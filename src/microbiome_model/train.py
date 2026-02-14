import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from models import SampleLevelRegressor, BasicRegressorWithASVEncoder
from datetime import datetime
from other_models import random_forest
from sklearn.metrics import mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import load_data, sample_data, sample_test_data
import os 
import math 
import torch.optim as optim
from losses import compute_count_loss, PairwiseLoss,compute_nuc_loss
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import _mean_absolute_error, predict

def float_mask(tensor , dtype=torch.float32):

    return (tensor != 0).to(dtype)

def compute_unifrac_loss(y_true, embeddings, pairwise_loss_fn):

    loss = pairwise_loss_fn(y_true, embeddings)  # [B, B]
    mask = float_mask(loss)                      # [B, B]
    num_samples = mask.sum()
    total_loss = loss.sum()
    return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0, device=loss.device)


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



def train_model(model, num_epochs, learning_rate, device, out_dir, abundance_table , distances, train_data, val_data, embedding_path=None,kmer_seqs=None,one_hot_seqs = None,  random_vector=False, kmer_embedding=False):
    """Training function similar to before but adapted for sample-level predictions"""
    print("Using device:", device)
    model = model.to(device)
    criterion = nn.MSELoss()
    unifrac_loss_fn = PairwiseLoss()
    print("Training model...")
    print(model)
    train_losses = []
    val_losses = []
    warmup_steps = 100000 
    total_steps = 30000
    
    # steps_per_epoch = len(train_loader)  # Number of batches per epoch
    # total_steps = num_epochs * steps_per_epoch  # Total training steps

    # # Set warmup steps dynamically (e.g., first 10% of training)
    # warmup_steps = int(0.1 * total_steps)  

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)
    # Cosine Decay with Restarts Scheduler (Equivalent to CosineDecayRestarts in TF)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=warmup_steps,  # Number of steps before first restart
        T_mult=1)  # Multiplicative factor for decay period
    
    
    best_val_loss = float('inf')
    early_stop_warmup = 90
    patience = 30
    patience_counter = 0
    best_mae = 0
    step = 0
    val_maes = []
    train_maes = []
    lrs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        per_batch_count_loss = 0
        per_batch_unifrac_loss = 0
        per_batch_nuc_loss = 0
        train_loader, val_loader = sample_data(abundance_table , train_data, val_data,kmer_seqs=kmer_seqs, embedding_path=embedding_path, one_hot_seqs = one_hot_seqs, random_vector=random_vector, batch_size=4)
        predictions = []
        labels = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            embeddings = batch ['embeddings'].to(device, dtype=torch.int64)
            abundances = batch['abundances'].to(device)
            targets = batch['outdoor_add_0'].to(device)
            masks = batch['masks'].to(device)
            sample_ids = batch['SampleID']
            ditances_target =torch.from_numpy(distances.filter(sample_ids).data).to(device)
            #print(embeddings.shape, abundances.shape, targets.shape, masks.shape)
            if torch.isnan(targets).any():
                print("NaN detected in targets!")
            
            optimizer.zero_grad()
            outputs , counts_pred , unifrac_embeddings, nuc_pred = model(embeddings, abundances, masks)
            if torch.isnan(outputs).any():
                print("NaN detected in outputs!")
                
            loss = criterion(outputs, targets) 
            predictions.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
            
            count_loss = compute_count_loss(abundances, counts_pred).mean()  
            if unifrac_embeddings is not None and nuc_pred is None: 
                unifrac_loss = compute_unifrac_loss(ditances_target, unifrac_embeddings, unifrac_loss_fn)
                loss = loss + count_loss + (0.01 * unifrac_loss)
            elif nuc_pred is not None and unifrac_embeddings is not None:
                nuc_loss = compute_nuc_loss(embeddings, nuc_pred, mask=masks)
                unifrac_loss = compute_unifrac_loss(ditances_target, unifrac_embeddings, unifrac_loss_fn)

                loss = loss + (0.1 * nuc_loss) + count_loss + (0.01 * unifrac_loss)
            else:
                unifrac_loss = torch.tensor(0.0, device=device)
                loss = loss + count_loss
            loss.backward()
            optimizer.step()
            #scheduler.step()
            # Increase step counter and log learning rate periodically
            step += 1
            if step % 101 == 0:
                print(f"Step {step}: Learning Rate = {scheduler.get_last_lr()[0]:.6f}")
                print(f"Step {step}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")
       
            lrs.append(optimizer.param_groups[0]['lr'])

            train_loss += loss.item()
            per_batch_count_loss += count_loss.item()
            per_batch_unifrac_loss += unifrac_loss.item()
            per_batch_nuc_loss += nuc_loss.item()
            
        print("total steps ", step)
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        train_mae = mean_absolute_error(labels ,predictions)
        train_maes.append(train_mae)
        train_loss /= len(train_loader)
        per_batch_count_loss /= len(train_loader)
        per_batch_unifrac_loss /= len(train_loader)
        per_batch_nuc_loss /= len(train_loader)
        
        predictions = []
        labels = []

        # Validation
        model.eval()
        val_loss = 0
        per_batch_count_loss_val = 0
        per_batch_unifrac_loss_val = 0
        per_batch_nuc_loss_val = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device, dtype=torch.int64)
                abundances = batch['abundances'].to(device)
                targets = batch['outdoor_add_0'].to(device)
                masks = batch['masks'].to(device)
                sample_ids = batch['SampleID']
                ditances_target =torch.from_numpy(distances.filter(sample_ids).data).to(device)
                
                outputs,counts_pred, unifrac_embeddings, nuc_pred  = model(embeddings, abundances, masks)
                #print(outputs.shape, counts_pred.shape, unifrac_embeddings.shape, nuc_pred.shape)
                count_loss = compute_count_loss(abundances, counts_pred).mean()  
                loss = criterion(outputs, targets) 
                if unifrac_embeddings is not None and nuc_pred is None: 
                    unifrac_loss = compute_unifrac_loss(ditances_target, unifrac_embeddings, unifrac_loss_fn)
                    loss = loss + count_loss + (0.01 * unifrac_loss)
                    
                elif nuc_pred is not None and unifrac_embeddings is not None:
                    nuc_loss = compute_nuc_loss(embeddings, nuc_pred, mask=masks)
                    unifrac_loss = compute_unifrac_loss(ditances_target, unifrac_embeddings, unifrac_loss_fn)

                    loss = loss + (0.1 * nuc_loss) + count_loss + (0.01 * unifrac_loss)

                else:
                    unifrac_loss = torch.tensor(0.0, device=device)
                    loss = loss + count_loss
                
                val_loss += loss.item()
                per_batch_count_loss_val += count_loss.item()
                per_batch_unifrac_loss_val += unifrac_loss.item()
                per_batch_nuc_loss_val += nuc_loss.item()
                
                
                predictions.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        per_batch_count_loss_val /= len(val_loader)
        per_batch_unifrac_loss_val /= len(val_loader)
        per_batch_nuc_loss_val /= len(val_loader)
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        val_mae = mean_absolute_error(labels ,predictions)
        val_maes.append(val_mae)
        print(f"Epoch {epoch+1}/{num_epochs}", "Count Alpha: ", model._count_alpha.item())
        print(f"Train Loss: {train_loss:.4f}, Count Loss: {per_batch_count_loss:.4f} ,Unifrac Loss : {per_batch_unifrac_loss:.5f} ,Nuc Loss : {per_batch_nuc_loss:.5f},Train MAE: {train_mae:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Count Loss: {per_batch_count_loss_val:.4f} ,Unifrac Loss : {per_batch_unifrac_loss_val:.5f} ,Nuc Loss : {per_batch_nuc_loss_val:.5f},Validation MAE: {val_mae:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_mae = val_mae
            if epoch >= early_stop_warmup:
                patience_counter = 0
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
            best_model_path = f'{out_dir}/model.pt'
            #labels, predictions = predict(model,val_loader  , device)
        
            #_ = _mean_absolute_error(predictions*100, labels*100, f"{out_dir}/best_model_valid.png")

        else:
            if epoch >= early_stop_warmup:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
        # Plot training and validation loss
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
    plt.savefig(f'{out_dir}/loss_curve.png')  # Save to output directory
    plt.show()
    
    # Save lrs to CSV
    pd.DataFrame({'lrs': lrs}).to_csv(f'{out_dir}/lrs.csv', index=False)

    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'{out_dir}/losses.csv', index=False)


    return {"best_loss" : float(best_val_loss), "best_mae" : float(best_mae) , "best_model" : best_model_path}


def main():
    mode = "train"
    # Setup
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #train_loader, val_loader, test_loader = reload_data()
    
    # Initialize and train model
    
    from sklearn.model_selection import KFold

    heldout_samples = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
 'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']#
    heldout_samples2 = ["D27", "D15", "D12", "D24"] #worst samples  
    set1 = set(heldout_samples)
    set2 = set(heldout_samples2)
    heldout_samples = list(set1.difference(set2))
    print("Heldout samples: ", heldout_samples)
    output_dir = "deeper_models/exp_5/"
    random_vector = False
    kmer_embedding = False
    embedding_path = None
    kmer_seqs = None
    train_asv_encoder = True
    one_hot_seqs = None
    heldout_samples = ["D15"]#["D27", "D15", "D12", "D24"] #worst samples
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a directory for each heldout sample
    for j in range(len(heldout_samples)):
        
        heldout = heldout_samples[j]
        print(f"Running for heldout sample: {heldout}")
        out_dir = os.path.join(output_dir, heldout)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if kmer_embedding:
            abundance_table, train_data , test_data, kmer_seqs, distances = load_data(heldout = heldout, kmer_embedding=True)
        elif train_asv_encoder:
            abundance_table, train_data , test_data, one_hot_seqs, distances = load_data(heldout = heldout, train_encoder=True)
        else:
            abundance_table, train_data , test_data, embedding_path, distances = load_data(heldout = heldout, kmer_embedding=False)
    
        train_samples, train_target = train_data
        runs = 3

        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of test samples: {len(test_data[0])}")

        
        num_train_samples = list(range(len(train_samples)))

        train_ids, val_ids = train_test_split(num_train_samples, test_size=0.1 , random_state=42) #42

        best_mae = 1000
        #for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        for i in range(runs):
               
            # Split train and validation sets
            X_train, X_val = [train_samples[i] for i in train_ids], [train_samples[i] for i in val_ids]
            y_train, y_val = {i:train_target[i] for i in X_train}, {i:train_target[i] for i in X_val}

            model = SampleLevelRegressor(use_nt_encoder=train_asv_encoder, pe=True, kmer_embedding=kmer_embedding )
            #model = BasicRegressorWithASVEncoder(pe=True)
            result_dir = os.path.join(out_dir, f"run_{i+1}")
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            res = train_model(
                model=model,
                num_epochs=1000,
                learning_rate=0.0001,
                device=device,
                out_dir=result_dir,
                abundance_table = abundance_table,
                distances = distances,
                train_data = (X_train, y_train),
                val_data = (X_val , y_val),
                embedding_path = embedding_path, 
                kmer_seqs = kmer_seqs,
                one_hot_seqs= one_hot_seqs,
                random_vector=random_vector, kmer_embedding = kmer_embedding
            )
            print(res)            
            
            try:
                
                # Load the best model
                with open(result_dir + "/res.json", "w") as json_file:
                    json.dump(res, json_file, indent=4)
            except:
                print("Error saving results to JSON file.")
            



if __name__ == "__main__":
    main()
    
