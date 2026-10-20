import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from datetime import datetime
import os
import math
import random
import re
import json
import matplotlib.pyplot as plt
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---- project imports ----
from microbiome_model.models import build_model
from microbiome_model.data.dataset import Arguments, DataProcessor
from microbiome_model.losses.core import compute_count_loss, PairwiseLoss, distance_weighted_ce
from microbiome_model.eval.evaluate import predict
from microbiome_model.utils.misc import _mean_absolute_error, compute_diversity_indices, create_augmented_batch_DDD 
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn, DonorAwareSampler
#from microbiome_model.training.pre_training_masked import MaskedAbundancePretraining
from torch.optim.swa_utils import AveragedModel


# ============================================================
# ABLATION CONFIG 
# ============================================================
ABLATION = {
    "name": "test_new_t_pred",        # goes into output path + saved config
    "metadata_columns": ["env", "bi_month_name"],  # metadata columns to use "env", "bi_month_name"
    "use_count_loss": True,              # auxiliary abundance head
    "grl": False,                          # use gradient reversal layer for adversarial training
    "clr": False,
}

METADATA_COLUMNS = list(ABLATION["metadata_columns"])

def build_features(batch, meta_spec, device):
    """Return the collated categorical features in the configured column order."""
    if not meta_spec:
        return None
    expected_names = [name for name, _cardinality in meta_spec]
    actual_names = batch.get("meta_names", [])
    if actual_names != expected_names:
        raise ValueError(
            f"Metadata columns mismatch: expected {expected_names}, got {actual_names}"
        )
    features = batch.get("features")
    if features is None:
        raise ValueError(f"Batch is missing configured metadata columns: {expected_names}")
    return features.to(device=device, dtype=torch.long)


def float_mask(tensor , dtype=torch.float32):

    return (tensor != 0).to(dtype)

def compute_unifrac_loss(y_true, embeddings, pairwise_loss_fn):

    loss = pairwise_loss_fn(y_true, embeddings)  # [B, B]
    mask = float_mask(loss)                      # [B, B]
    num_samples = mask.sum()
    total_loss = loss.sum()
    return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0, device=loss.device)

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Calculate standard squared error
        sq_error = (pred - target) ** 2
        
        # Create weights based on target magnitude. 
        # Adding 1.0 ensures the zeros still get a baseline weight of 1.
        # If a target is 3.0 (300 in real scale), its error is multiplied by 4.
        weights = target + 1.0 
        
        # Apply weights and return the mean
        weighted_loss = sq_error * weights
        return weighted_loss.mean()

class WarmupCosineDecay:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def lr_lambda(self, step):
        """Defines learning rate schedule"""
        if step < self.warmup_steps:
            return self.base_lr
        else:
            # Cosine decay phase
            decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))  # Cosine decay

    def get_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)


def validate_with_donor_metrics(model, val_loader, criterion_age, device, data_processor):
    model.eval()
    val_loss_epoch = 0
    donor_errors = {} # Store absolute errors per donor
    
    sampleID_map_donorID = data_processor.sampleID_map_donorID

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4
            targets = batch['outdoor_add_0'].to(device)
            masks = batch['masks'].to(device)
            env = batch['env'].to(device, dtype=torch.float32)
            features = batch['season'].to(device)
            sample_ids = batch["SampleID"]

            outputs, _, _, _, _ = model(embeddings, abundances, masks, features, env=env)
            
            loss = criterion_age(outputs, targets)
            val_loss_epoch += loss.item()
            
            # Calculate absolute errors for this batch
            abs_errors = torch.abs(outputs - targets).cpu().numpy().flatten()
            
            # Map errors back to their specific DonorID
            for i, sample_id in enumerate(sample_ids):
                donor_id = sampleID_map_donorID[sample_id]
                if donor_id not in donor_errors:
                    donor_errors[donor_id] = []
                donor_errors[donor_id].append(abs_errors[i])

    # Calculate mean error per donor
    mean_donor_mae = {d: np.mean(errs) for d, errs in donor_errors.items()}
    
    # Sort by error to find the "problem" donors
    worst_donors = sorted(mean_donor_mae.items(), key=lambda x: x[1], reverse=True)[:5]
    
    avg_val_mae = np.mean(list(mean_donor_mae.values()))
    val_loss_epoch /= len(val_loader)

    print(f"\n--- Validation Report ---")
    print(f"Avg Val MAE: {avg_val_mae:.4f}")
    print(f"Top 5 Worst Donors: {worst_donors}")
    
    return avg_val_mae, val_loss_epoch, mean_donor_mae


def train_model_GRL(model, num_epochs, learning_rate, device, out_dir, data_processor, use_unifrac_loss=False):
    model = model.to(device)
    
    # Losses
    criterion_age = nn.MSELoss()
    criterion_subj = nn.CrossEntropyLoss() # For the adversary
    pairwise_loss_fn = PairwiseLoss() # Assuming this is defined in your utils
    count_loss_type = "mse"

    sampleID_map_donorID = data_processor.sampleID_map_donorID



    # Optimization
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Tracking
    train_losses, val_losses = [], []
    best_mae = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_warmup = 50
    patience = 30
    
    # Hyperparameters
    reg_count_loss = 0.1
    lambda_adv = 0.1

    # Initial Data Load
    train_dataset, val_dataset = data_processor.sample_data(epoch=0)

    unique_donors = set(sampleID_map_donorID.values())
    print("Unique donors in dataset: ", len(unique_donors))
    donor_to_idx = {donor: i for i, donor in enumerate(sorted(unique_donors))}
    print("Donor to Index Mapping: ", donor_to_idx)
    print(f"Adversarial Training Setup: Found {len(unique_donors)} unique donors.")

    # Assuming sparse_collate_fn is defined in your scope
    train_loader = DataLoader(train_dataset,  collate_fn=sparse_collate_fn, shuffle=True, batch_size = 128)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=sparse_collate_fn)

    train_donors = set([data_processor.sampleID_map_donorID[sid] for sid in train_dataset.sample_ids])
    val_donors = set([data_processor.sampleID_map_donorID[sid] for sid in val_dataset.sample_ids])

    print(f"Unique donors in training set: {len(train_donors)}")
    print(f"Unique donors in validation set: {len(val_donors)}")
    print(f"Intersection of donors between train and val: {len(train_donors.intersection(val_donors))} (should be 0 for proper generalization test)")

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    loss_record_val = []
    loss_record_train = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        
        # --- Alpha Scheduler (Gradient Reversal Strength) ---
        # Slowly ramp alpha from 0.0 to 1.0 over the first ~50 epochs
        p = min(1.0, epoch / 20.0)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        # Update the model's GRL alpha
        if hasattr(model, 'grl'):
            print(f"Epoch {epoch+1}: Setting GRL alpha to {alpha:.4f}")
            model.grl_layer.alpha = alpha

        # Resample training data if needed (for epoch-dependent sampling)
        train_dataset.sample_epoch_init(epoch)
        #train_loader = DataLoader(train_dataset,  collate_fn=sparse_collate_fn, batch_sampler=DonorAwareSampler(train_dataset.sample_ids, data_processor.sampleID_map_donorID))
        train_loader = DataLoader(train_dataset,  collate_fn=sparse_collate_fn, shuffle=True, batch_size = 64)

        predictions = []
        labels = []

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Alpha: {alpha:.2f}]')
        max_idx = 0
        for batch in loop:
            # Move data to device
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4
            
            targets = batch['outdoor_add_0'].to(device)
            masks = batch['masks'].to(device)
            env = batch['env'].to(device, dtype=torch.float32)
            features = batch['season'].to(device)        
            sample_ids = batch["SampleID"]
            donor_ids = [sampleID_map_donorID[sid] for sid in sample_ids]

            max_idx = max(max_idx, max([donor_to_idx[d] for d in donor_ids]))
            # Convert to tensor indices
            subj_targets = torch.tensor([donor_to_idx[d] for d in donor_ids], device=device)
            if torch.isnan(targets).any():
                continue

            optimizer.zero_grad()

            outputs, counts_pred, unifrac_emb, diversity_pred, subj_logits = model(
                embeddings, abundances, masks, features, env=env
            )

            # 1. Main Task Loss
            loss_age = criterion_age(outputs, targets)
            
            # 2. Adversarial Loss (Subject ID)
            loss_subj = criterion_subj(subj_logits, subj_targets)
            # 3. Auxiliary Losses
            loss_count = compute_count_loss(abundances, counts_pred, loss_type=count_loss_type).mean()
            
            adv_acc = (subj_logits.argmax(dim=1) == subj_targets).float().mean().item()
            adv_chance = 1.0 / len(unique_donors)
            
            #print(f"Batch {loop.n}/{len(train_loader)} - Age Loss: {loss_age.item():.4f}, Subj Loss: {loss_subj.item():.4f}, Count Loss: {loss_count.item():.4f}, Adv Acc: {adv_acc:.4f} (Chance: {adv_chance:.4f})")
            # Combine
            # The GRL handles the "min-max" logic automatically via backward()
            #print(f"Batch Losses - Age: {loss_age.item():.4f}, Subj: {loss_subj.item():.4f}, Count: {loss_count.item():.4f}")
            total_loss = loss_age  + (loss_subj * lambda_adv) +  (reg_count_loss * loss_count)
            #(lambda_adv * loss_subj)
            # (Optional: Add Unifrac or Augmented loss here as needed, simplified for clarity)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += total_loss.item()
            
            # Store for metrics
            predictions.append(outputs.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

        scheduler.step()

        # Calculate Train Metrics
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        train_mae = mean_absolute_error(labels, predictions)
        train_loss_epoch /= len(train_loader)
        loss_record_train.append(train_loss_epoch)
        # --- Validation Loop ---
        model.eval()
        val_loss_epoch = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                abundances = batch['abundances'].to(device) / 1e4
                targets = batch['outdoor_add_0'].to(device)
                masks = batch['masks'].to(device)
                env = batch['env'].to(device, dtype=torch.float32)
                features = batch['season'].to(device)

                outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks, features, env=env)
                
                loss = criterion_age(outputs, targets)
                val_loss_epoch += loss.item()
                
                val_preds.append(outputs.cpu().numpy())
                val_labels.append(targets.cpu().numpy())

        val_loss_epoch /= len(val_loader)
        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)
        val_mae = mean_absolute_error(val_labels, val_preds)
        loss_record_val.append(val_loss_epoch)

        print(f"Train Loss: {train_loss_epoch:.4f} | Train MAE: {train_mae:.4f}")
        print(f"Valid Loss: {val_loss_epoch:.4f} | Valid MAE: {val_mae:.4f}")

        # Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            best_val_loss = val_loss_epoch
            if epoch >= early_stop_warmup:
                patience_counter = 0
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
            print(f"--> Saved Best Model (MAE: {best_mae:.4f})")
        else:
            if epoch >= early_stop_warmup:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        with open(f'{out_dir}/training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}: Train Loss={train_loss_epoch:.4f}, Train MAE={train_mae:.4f}, Val Loss={val_loss_epoch:.4f}, Val MAE={val_mae:.4f}\n")   
        validate_with_donor_metrics(model, val_loader, criterion_age, device, data_processor)

    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae)}

def next_abundance_sse(pred, batch, masks, device, scale_abundance):
    """Squared error of next-timepoint abundance predictions.

    Only real (unpadded) ASVs of samples that have a next timepoint count.
    Returns (sum of squared errors, number of positions) so epoch means are exact.
    """
    target = batch['next_abundances'].to(device)
    target = target / 1e4 if scale_abundance else target          # match the input scale
    # print("target", target[0])
    # print("pred", pred[0])
    valid = masks.bool() & batch['has_next'].to(device)[:, None]  # (B, L)
    if pred is None or not valid.any():
        return torch.zeros((), device=device), 0
    sq = (pred.squeeze(-1) - target.squeeze(-1)) ** 2
    return sq[valid].sum(), int(valid.sum())


def train_model(model, train_config, device, out_dir, data_processor, use_unifrac_loss=False, scale_abundance = True, multitask=False, grl=False, meta_spec=None): #abundance_table , distances, train_data, val_data, use_unifrac_loss=False, embedding_path=None,kmer_seqs=None, random_vector=False, kmer_embedding=False):
    count_loss_type = "mse"
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss(beta=0.5)
    print("Training model...")
    print(model)
    train_losses = []
    train_count_losses = []
    train_regression_losses = []
    val_losses = []
    val_count_losses = []
    val_regression_losses = []
    train_next_losses = []
    val_next_losses = []
    val_next_baseline_losses = []
    train_maes = []
    val_maes = []
    lrs = []
    model = model.to(device)
    eta_min = 1e-6
    T_0 = 10
    T_mult = 2
    lr = train_config.get("learning_rate", 1e-4)
    batch_size = train_config.get("batch_size", 64)
    reg_count_loss = train_config.get("reg_count_loss", 1.0)
    epochs = train_config.get("num_epochs", 1000)
    patience = train_config.get("patience", 30)
    grl_loss_weight = train_config.get("grl_loss_weight", 0.1)
    use_count_loss = train_config.get("use_count_loss", True)
    next_loss_weight = train_config.get("next_loss_weight", 1.0)
    next_loss_weight = 10
    meta_spec = meta_spec or []
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    subsample_table = train_config.get("subsample", True)

    if hasattr(model, 'basemodel'):
        model.basemodel.requires_grad_(False)
        
    if hasattr(model, 'subject_head'):
        head_params = list(model.subject_head.parameters())
        head_ids = {id(p) for p in head_params}
        base_params = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = AdamW([
            {"params": base_params, "lr": lr},
            {"params": head_params, "lr": lr * 10, "weight_decay": 0.0},
        ], weight_decay=0.001)


    



    best_val_loss = float('inf')
    early_stop_warmup = 30
    patience_counter = 0
    best_mae = 1000
    step = 0
    smooth_window      = 1
    best_smoothed_mae  = float('inf')
    best_model_path    = f'{out_dir}/model.pt'
    

    unique_donors = set(data_processor.sampleID_map_donorID[sid] for sid in data_processor.train_data[0])


    train_dataset, val_dataset = data_processor.sample_data(epoch=0, subsample=subsample_table)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=False, collate_fn=sparse_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  pin_memory=False, collate_fn=sparse_collate_fn)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps
# )
#     swa_model = AveragedModel(model)
#     swa_start = int(0.6 * epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=0.00005
        )
    
    distances_target = None
    bins = False
    
    donor_to_idx = {donor: i for i, donor in enumerate(sorted(unique_donors))}



    
    print("SCALE ABUNDANCE: +++++", scale_abundance)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        per_batch_count_loss = 0
        per_batch_unifrac_loss = 0
        regression_loss = 0.0
        next_sse, next_n = 0.0, 0
        adv_loss_sum = 0.0
        adv_correct = 0
        adv_total = 0



        train_dataset.sample_epoch_init(epoch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, collate_fn=sparse_collate_fn)

        predictions = []
        labels = []
        if bins and epoch == 0:
            all_abundances = []
            for sample in train_loader:
                abs_vals = sample['abundances'] / 1e4 if scale_abundance else sample['abundances']
                all_abundances.extend(abs_vals[abs_vals > 0].numpy().tolist())
            all_abundances = np.array(all_abundances)

            log_abs = np.log(all_abundances + 1e-6)

            n_nonzero_bins = 6
            quantile_points = np.linspace(0, 1, n_nonzero_bins + 1)[1:-1]
            log_boundaries = np.quantile(log_abs, quantile_points)
            boundaries = np.exp(log_boundaries) - 1e-6 
            print("Computed bin boundaries (non-zero values):", boundaries)
            bin_assignments = np.digitize(all_abundances, boundaries)
            unique, counts = np.unique(bin_assignments, return_counts=True)
            print("Bin occupancy:", dict(zip(unique, counts)))
        
        if grl:
            
            warmup_epochs = 15  # off the LR restart at epoch 10 (CosineAnnealingWarmRestarts T_0=10, T_mult=2)

            ramp = 30
            alpha_max = 0.3
            alpha = 0.0 if epoch < warmup_epochs else alpha_max * min(1.0, (epoch - warmup_epochs) / ramp)

             #int(0.1 * epochs)

            # if epoch < warmup_epochs:
            #     alpha = 0.0
            # else:
            #     p = (epoch - warmup_epochs) / max(
            #         1, epochs - warmup_epochs - 1
            #     )
            #     alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            # if model.grl:
            #     model.grl_layer.alpha = alpha
    
            # p = min(1.0, epoch / 80.0)
            # alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            # Update the model's GRL alpha
            if hasattr(model, 'grl'):
                print(f"Epoch {epoch+1}: Setting GRL alpha to {alpha:.4f}")
                model.grl_layer.alpha = alpha

        print(model.grl_layer.alpha if grl else "GRL not used")
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4 if scale_abundance else batch['abundances'].to(device)
            targets = batch['outdoor_add_0'].to(device)
            masks = batch['masks'].to(device)
            sample_ids = batch["SampleID"]
            bins_abundance = batch['binned_abundances'].to(device) if 'binned_abundances' in batch else None
            
            if bins:
                abundances = bins_abundance

            bimonth_idx = batch['season'].to(device)

            features = build_features(batch, meta_spec, device)



            if torch.isnan(targets).any():
                print("NaN detected in targets!")
                continue

            optimizer.zero_grad()
            model_out = model(embeddings, abundances, masks, features)
            outputs, counts_pred, unifrac_embeddings, diversity_pred, subj_pred = model_out[:5]
            d_next = model_out[5] if len(model_out) > 5 else None
            next_count_pred = abundances + d_next

            loss = criterion(outputs, targets)
            regression_loss += loss.item()

            sse, n = next_abundance_sse(next_count_pred, batch, masks, device, scale_abundance)
            if n > 0:
                loss = loss + next_loss_weight * (sse / n)
                next_sse += sse.item()
                next_n += n


            if grl:
                subj_targets = torch.tensor([donor_to_idx[data_processor.sampleID_map_donorID[sid]] for sid in sample_ids], device=device)
                criterion_subj = nn.CrossEntropyLoss()
                loss_subj = criterion_subj(subj_pred, subj_targets)
                #print(f"Batch Losses - Age: {loss.item():.4f}, Subj: {loss_subj.item():.4f}")
                loss = loss + (grl_loss_weight * loss_subj) #earlier exps were with 0.1

                adv_correct += (subj_pred.argmax(1) == subj_targets).sum().item()
                adv_total += len(subj_targets)
                adv_loss_sum += loss_subj.item()
            print(counts_pred[0,:10,:])
            print(abundances[0,:10])



            if use_count_loss and counts_pred is not None:
                if bins:
                    count_loss = distance_weighted_ce(abundances, counts_pred, num_bins=50, mask=masks)
                else:
                    count_loss = compute_count_loss(abundances, counts_pred, masks=masks, loss_type=count_loss_type).mean()
            else:
                count_loss = torch.tensor(0.0, device=device)

            # Unifrac loss (only on clean pass)
            if use_unifrac_loss:
                distances_target = torch.from_numpy(distances.filter(sample_ids).data)
                unifrac_loss = compute_unifrac_loss(distances_target, unifrac_embeddings, pairwise_loss_fn)
                per_batch_unifrac_loss += unifrac_loss.item()
                loss = loss + (0.1 * unifrac_loss)

            # Combined loss
            #total_loss = loss + 0.1 * loss_aug + reg_count_loss * (count_loss + count_loss_aug)


            total_loss = loss + (reg_count_loss * count_loss)
            predictions.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())

            total_loss.backward()
            if grl:
                torch.nn.utils.clip_grad_norm_(base_params, 1.0)
                torch.nn.utils.clip_grad_norm_(head_params, 5.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


            step += 1
            train_loss += total_loss.item()
            per_batch_count_loss += count_loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
            

        scheduler.step()

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        train_mae = mean_absolute_error(labels, predictions)
        train_maes.append(train_mae)
        train_loss /= len(train_loader)
        per_batch_count_loss /= len(train_loader)
        per_batch_unifrac_loss /= len(train_loader)
        regression_loss /= len(train_loader)
        next_loss = next_sse / max(next_n, 1)


        predictions = []
        labels = []

        # Validation
        model.eval()
        val_loss = 0
        per_batch_count_loss_val = 0
        per_batch_unifrac_loss_val = 0
        all_ids = []
        print("..................Running validation.....................")
        print(val_dataset.biom_data.shape)
        regression_loss_val = 0.0
        next_sse_val, next_n_val, next_sse_base = 0.0, 0, 0.0
        for i in range(1):
            #val_dataset.sample_epoch_init(epoch=epoch+i)
            #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, collate_fn=sparse_collate_fn)


            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch['embeddings'].to(device)
                    abundances = batch['abundances'].to(device) / 1e4 if scale_abundance else batch['abundances'].to(device)
                    targets = batch['outdoor_add_0'].to(device)
                    masks = batch['masks'].to(device)
                    all_ids.extend(batch["SampleID"])
                    sample_ids = batch["SampleID"]
                    bins_abundance = batch['binned_abundances'].to(device) if 'binned_abundances' in batch else None
                    features = build_features(batch, meta_spec, device)

                    if bins:
                        abundances = bins_abundance
                        
                    model_out = model(embeddings, abundances, masks, features)
                    outputs, counts_pred, unifrac_embeddings, diversity_pred, _ = model_out[:5]
                    d_next = model_out[5] if len(model_out) > 5 else None
                    next_count_pred = abundances + d_next if d_next is not None else None

                    # Next-timepoint loss, and the "no change" baseline (predict next = current)
                    sse, n = next_abundance_sse(next_count_pred, batch, masks, device, scale_abundance)
                    if n > 0:
                        current = batch['abundances'].to(device)
                        current = current / 1e4 if scale_abundance else current
                        sse_base, _ = next_abundance_sse(current, batch, masks, device, scale_abundance)
                        next_sse_val += sse.item()
                        next_sse_base += sse_base.item()
                        next_n_val += n

                    if use_count_loss and counts_pred is not None:
                        if bins:
                            count_loss = distance_weighted_ce(abundances, counts_pred, num_bins=50, mask=masks)
                        else:
                            count_loss = compute_count_loss(abundances, counts_pred, masks=masks, loss_type=count_loss_type).mean()
                    else:
                        count_loss = torch.tensor(0.0, device=device)
                    loss = criterion(outputs, targets)
                    regression_loss_val += loss.item()

                    if use_unifrac_loss:
                        distances_target = torch.from_numpy(distances.filter(sample_ids).data).to(device)
                        unifrac_loss = compute_unifrac_loss(distances_target, unifrac_embeddings, pairwise_loss_fn)
                        per_batch_unifrac_loss_val += unifrac_loss.item()
                        loss = loss + (0.1 * unifrac_loss)
                    


                    total_loss = loss + (reg_count_loss * count_loss)
                    if n > 0:
                        total_loss = total_loss + next_loss_weight * (sse / n)

                    val_loss += total_loss.item()
                    per_batch_count_loss_val += count_loss.item()

                    predictions.append(outputs.cpu().detach().numpy())
                    labels.append(targets.cpu().detach().numpy())
        val_loss = val_loss / (len(val_loader) * 1)  # Average over all validation runs
        regression_loss_val /= (len(val_loader) * 1)
        print("All IDs in validation set: ", len(all_ids))
        #val_loss /= len(val_loader)
        per_batch_count_loss_val /= (len(val_loader) * 1)
        per_batch_unifrac_loss_val /= (len(val_loader) * 1)
        next_loss_val = next_sse_val / max(next_n_val, 1)
        next_loss_base = next_sse_base / max(next_n_val, 1)
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        val_mae = mean_absolute_error(labels, predictions)
        val_maes.append(val_mae)
        window = val_maes[-smooth_window:]
        smoothed_mae = float(np.mean(window))
        warm = len(val_maes) >= smooth_window   # wait until the window is full


        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Regression Loss: {regression_loss:.4f}, Count Loss: {per_batch_count_loss:.4f}, Unifrac Loss: {per_batch_unifrac_loss:.4f}, Train MAE: {train_mae:.4f}")
        if grl:
            print(f"adv acc {adv_correct/adv_total:.3f} (chance {1/len(donor_to_idx):.3f}), "
                f"adv CE {adv_loss_sum/len(train_loader):.3f} (log K = {np.log(len(donor_to_idx)):.3f})")
        print(f"Validation Loss: {val_loss:.4f}, Regression Loss: {regression_loss_val:.4f}, Count Loss: {per_batch_count_loss_val:.4f}, Unifrac Loss: {per_batch_unifrac_loss_val:.4f}, Validation MAE: {val_mae:.4f}")
        print(f"Next-abundance MSE: train {next_loss:.3e} | val {next_loss_val:.3e} "
              f"| val no-change baseline {next_loss_base:.3e} ({next_n_val} ASV positions)")
        print(f"Val MAE: {val_mae:.4f} | smoothed({len(window)}): {smoothed_mae:.4f} "
        f"| best smoothed: {best_smoothed_mae:.4f}")
        if hasattr(model, '_count_alpha'):
            print(f"Alpha: {model._count_alpha.item():.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_count_losses.append(per_batch_count_loss)
        train_regression_losses.append(regression_loss)
        val_count_losses.append(per_batch_count_loss_val)
        val_regression_losses.append(regression_loss_val)
        train_next_losses.append(next_loss)
        val_next_losses.append(next_loss_val)
        val_next_baseline_losses.append(next_loss_base)

        print("difference between smoothed mae and best smoothed mae: ", smoothed_mae - best_smoothed_mae)
        if warm and smoothed_mae < best_smoothed_mae - 0.003:

            best_smoothed_mae = smoothed_mae
            best_val_loss = val_loss
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
            best_model_path = f'{out_dir}/model.pt'
            print(f"--> Saved (smoothed {best_smoothed_mae:.4f}, raw {val_mae:.4f})")
            print("Running predictions on validation set")
            labels_p, predictions_p, per_sample_pred = predict(model, val_loader, device, scale_abundances=scale_abundance, val = True, multitask=multitask, meta_spec=meta_spec)
            err = _mean_absolute_error(predictions_p, labels_p, f"{out_dir}/best_model_valid_2.png")
            err = _mean_absolute_error(predictions_p * 100, labels_p * 100, f"{out_dir}/best_model_valid.png")
            print(f"MAE on validation set: {err}")
            with open(f"{out_dir}/val_predictions.json", "w") as fs:
                json.dump(per_sample_pred,  fs, indent=4)
        elif warm and epoch >= early_stop_warmup:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.plot(train_maes, label='Train MAE', marker='x')
    plt.plot(val_maes, label='Validation MAE', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/loss_curve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_count_losses, label='Train Count Loss', marker='o')
    plt.plot(val_count_losses, label='Validation Count Loss', marker='s')
    plt.plot(train_regression_losses, label='Train Regression Loss', marker='x')
    plt.plot(val_regression_losses, label='Validation Regression Loss', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/count_loss_curve.png')
    plt.close()

    

    pd.DataFrame({'lrs': lrs}).to_csv(f'{out_dir}/lrs.csv', index=False)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses,
                  'train_next_loss': train_next_losses, 'val_next_loss': val_next_losses,
                  'val_next_baseline_loss': val_next_baseline_losses}).to_csv(f'{out_dir}/losses.csv', index=False)

    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae), "best_model": best_model_path}




def cleanup_resources():
    """Clean up resources between training runs"""
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Close any remaining file handles
    try:
        import resource
        print(f"Open file handles: {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except:
        pass

def setup_seed(seed):
    """Set random seed for reproducibility"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass

def main():
    setup_seed(42) # Set a global seed for reproducibility
    
    mode = "train"
    # Setup
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    #train_loader, val_loader, test_loader = reload_data()
    
    # Initialize and train model
        ## all outdoors ###
    # output_dir = "finetune_dnabert_results/all_outdoors/"
    # biom_file = "../data/new_data/table_all/feature-table.biom"
    # metadata_file = "../data/new_data/metadata_all_outdoor.tsv"
    # embedding_path = "../data/embeddings/al_outdoors.h5"
    # heldout_samples =['D4', 'D17', 'D29', 'D12', 'D6',
    #    'D8', 'D10', 'D13', 'D15', 'D19', 'D21', 'D23', 'D25', 'D27',
    #    'D11']
    ### all outdoors ###
    ## sheds data ###
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"

    biom_file = os.path.join(data_dir, "process_data_all/exported-feature-table/feature-table.biom")
    biom_file_sheds = os.path.join(data_dir, "data/new_data/feature-table/feature-table.biom")
    new_biom_file = os.path.join(data_dir, "data/new_pmi_modeling_files/merged_latest_table.biom")

    sheds_file = os.path.join(data_dir, "data/new_data/metadata_sheds.tsv")
    metadata_file = os.path.join(data_dir, "data/new_data/merged_metadata_final.csv") # combined_metadata_merged.tsv
    new_metadata_file = os.path.join(data_dir, "data/new_pmi_modeling_files/train_metadata_latest.csv")
    
    embedding_path = os.path.join(data_dir, "data/embeddings/all_data.h5") #all_data_new.h5"
    heldout_samples = pd.read_csv(sheds_file, sep="\t")['DonorID'].unique().tolist()

    
    datasets = { "SHEDS" : [biom_file_sheds, metadata_file, embedding_path],
                "ALL_DATA" : [biom_file, metadata_file, embedding_path],
                "NEW_DATA" : [new_biom_file,new_metadata_file, embedding_path ]
    }


    key_data = "ALL_DATA"
    EXP_DATA = datasets[key_data] # Change to switch dataset
    biom_file, metadata_file, embedding_path = EXP_DATA
    
    
    experiment_name = f"{key_data.lower()}_{ABLATION['name']}"
    output_dir = os.path.join(data_dir, f"microbiome_model/results/final/{experiment_name}/") #clusteredbaseline2

    #computed = ["D4","D6", "D8", "D10", "D11", "D12", "D14", "D16", "D17","D18", "D19", "D21", "D7","D29", "D13", "D15", "D20", "D29", "D30"]
    #computed = ["D4","D6", "D8", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17","D18", "D19", "D20", "D21", "D27", "D29"]# For testing purposes, limit to a few donors
    computed = []
    heldout_samples = [s for s in heldout_samples if s not in computed]
    heldout_samples = ["D15", "D20", "D9", "D27"]# For testing purposes, limit to a few donors
    #heldout_samples = ["D5", "D7","D9"] # For testing purposes, limit to a few donors
    print(f"Heldout samples for this experiment: {heldout_samples}")
    
    #pd.read_csv(metadata_file)[""]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    seed_values = [42, 1337, 2048]
    grl = ABLATION["grl"]
    clr = False
    multitask = True
    model_config = {
            "model_name": "BasicRegressorFilm",
            "input_dim": 128,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "pe": False,
            "multitask": multitask,
            "data": "all_data",
            "grl": ABLATION["grl"],
            "clr": ABLATION["clr"],
            "metadata_columns": METADATA_COLUMNS,
            "metadata_categories": {},
            "metadata_cardinalities": None,
            "pretrained_weights": os.path.join(data_dir, "microbiome_model/results/MaskedPretrainSkin/pretrained_best.pt"),
            "num_donors": 74
        }
    train_config = {
                "batch_size": 64,
                "learning_rate": 5e-05,
                "reg_count_loss": 10.0,  # Weight for count reconstruction loss
                "early_stop_warmup": 15,
                "patience": 15,
                "num_epochs": 200,
                "use_unifrac_loss": False,
                "scale_abundance": True,
                "seeds": seed_values,
                "grl": grl, 
                "grl_loss_weight": 0.5,  # Weight for GRL loss
                "subsample": True,  
                "embedding_path": embedding_path,
                'embedding_type': "DNABERT",  # or "DNABERT" or "RANDOM"
            }
        
    embedding_path = None if train_config.get("embedding_type", "RANDOM") == "RANDOM" else embedding_path

    # Create a directory for each heldout sample
    for j in range(len(heldout_samples)):
        
        heldout = heldout_samples[j]
        print(f"Running for heldout sample: {heldout}")
        if heldout is None:
            out_dir = os.path.join(output_dir, "no_heldout")
        else:
            out_dir = os.path.join(output_dir, heldout)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            

        runs = 3

        #abundance_table, train_data , test_data, embedding_path, distances = load_data(args
        

        #for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        for i in range(runs):
            print(f"Run {i+1}/{runs} for heldout sample {heldout}")
            setup_seed(seed_values[i])  # Different seed for each run


            cleanup_resources()
            args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding=train_config["embedding_type"], #,
            heldout=heldout,
            sort_asvs = True,
            clr=clr
                )
            data_processor = DataProcessor(args)
            # data_processor.load_biom()
            # data_processor.load_sequences()
            # if data_processor.seq_to_idx is None:
            #     print("Sequence to index mapping not found. Ensure that sequences are loaded correctly.")
            # else:
            #     print(f"Loaded {len(data_processor.seq_to_idx)} unique sequences.")

            data_processor.load_data(
                multitask=multitask,
                column="bi_month_name",
                metadata_columns=METADATA_COLUMNS,
            )
            meta_spec = [
                (column, len(data_processor.metadata_categories[column]))
                for column in METADATA_COLUMNS
            ]
            model_config["metadata_categories"] = data_processor.metadata_categories
            model_config["metadata_cardinalities"] = [cardinality for _, cardinality in meta_spec] or None

            ### number of donors in training and validation sets
            train_donors = set([data_processor.sampleID_map_donorID[sid] for sid in data_processor.train_data[0]])
            val_donors = set([data_processor.sampleID_map_donorID[sid] for sid in data_processor.test_data[0]])
            print(f"Number of unique donors in training set: {len(train_donors)}")
            print(f"Number of unique donors in validation set: {len(val_donors)}")
            model_config["num_donors"] = len(train_donors)  # Pass this to the model config for GRL or donor-aware components
            print("Model Config:", model_config)
            model = build_model(model_config, str(device))

            result_dir = os.path.join(out_dir, f"run_{i+1}")
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            with open(os.path.join(result_dir, "model_config.json"), "w") as f:
                json.dump(model_config, f, indent=4)
            with open(os.path.join(result_dir, "training_config.json"), "w") as f:
                json.dump({**train_config, "seed": seed_values[i]}, f, indent=4)
            with open(os.path.join(result_dir, "ablation_config.json"), "w") as f:
                json.dump(ABLATION, f, indent=4)
            print("Ablation Config:", ABLATION)
            print(result_dir)

            res = train_model(
                    model=model,
                    train_config=train_config,
                    device=device,
                    out_dir=result_dir,
                    data_processor=data_processor,
                    use_unifrac_loss=False,
                    scale_abundance = train_config["scale_abundance"],
                    grl=grl,
                    meta_spec=meta_spec,
                )
            print(res)            
            res["seed"] = seed_values[i]
                
                # Save results
            with open(result_dir + "/res.json", "w") as json_file:
                    json.dump(res, json_file, indent=4)

        


if __name__ == "__main__":
    main()
    

