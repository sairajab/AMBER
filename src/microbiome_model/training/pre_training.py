"""
Contrastive Pre-training for Microbiome ADD Prediction.

Usage:
    1. Standalone pre-training:
        model = BasicRegressor(128, 512, 4, 2, 0.3, pe=True, GRL=True)
        pretrained_model = pretrain_contrastive(model, data_processor, device, out_dir)
    
    2. Full pipeline (pretrain + finetune):
        model = BasicRegressor(128, 512, 4, 2, 0.3, pe=True, GRL=True)
        results = train_with_contrastive_pretraining(model, data_processor, device, out_dir)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from microbiome_model.models.orig import BasicRegressor, BasicRegressorwithUnifrac
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn


# ---------------------------------------------------------------------------
# Import your project modules here — adjust paths as needed
# ---------------------------------------------------------------------------
from microbiome_model.training.train_orig import train_model, setup_seed, cleanup_resources
from microbiome_model.data.dataset import DataProcessor, Arguments




# ===========================================================================
# Contrastive Loss
# ===========================================================================
class SoftADDContrastiveLoss(nn.Module):
    """
    Soft contrastive loss that pulls together samples with similar ADD
    across different donors, and pushes apart samples with different ADD.
    """

    def __init__(self, temperature=0.07, sigma=1.0, donor_penalty=0.5):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.donor_penalty = donor_penalty

    def forward(self, embeddings, add_values, donor_ids):
        """
        Args:
            embeddings: (B, D) projected embeddings
            add_values: (B,) ADD targets
            donor_ids:  (B,) integer donor IDs (as a tensor)
        """
        embeddings = F.normalize(embeddings, dim=1)
        batch_size = embeddings.shape[0]

        if batch_size < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Cosine similarity scaled by temperature
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # ADD distance → soft target weights (close ADD = high weight)
        add_diff = torch.abs(add_values.unsqueeze(0) - add_values.unsqueeze(1))
        target_weights = torch.exp(-add_diff ** 2 / (2 * self.sigma ** 2))

        # Penalize same-donor pairs to encourage donor-invariant representations
        same_donor = (donor_ids.unsqueeze(0) == donor_ids.unsqueeze(1)).float()
        target_weights = target_weights * (1.0 - self.donor_penalty * same_donor)

        # Zero out diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        target_weights = target_weights * mask.float()

        # Normalize to form target distribution per row
        row_sums = target_weights.sum(dim=1, keepdim=True) + 1e-8
        target_dist = target_weights / row_sums

        # Log softmax of similarities (masked diagonal)
        sim_masked = sim_matrix.masked_fill(~mask, -1e9)
        log_probs = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

        # Cross-entropy between target distribution and learned similarity
        loss = -(target_dist * log_probs * mask.float()).sum(dim=1).mean()
        return loss


# ===========================================================================
# Projection Head Wrapper
# ===========================================================================
class ContrastiveProjector(nn.Module):
    """
    Wraps your base model's encoder and adds a projection head
    for contrastive pre-training. The projection head is discarded
    after pre-training.
    """

    def __init__(self, base_model, encoder_dim=256, proj_dim=64):
        super().__init__()
        self.base_model = base_model
        self.encoder_dim = encoder_dim

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, proj_dim),
        )

    def forward(self, embeddings, abundances, masks, features, env=None):
        """
        Returns:
            encoder_output: (B, encoder_dim) — pooled representation
            projections:    (B, proj_dim)    — for contrastive loss
        """
        encoder_output = self.base_model.encode(
            embeddings, abundances, masks, features, env=env
        )
        projections = self.projector(encoder_output)
        return encoder_output, projections


# ===========================================================================
# Helper: Extract donor IDs from sample ID strings
# ===========================================================================
def extract_donor_ids(sample_ids, sampleID_map_donorID, donor_to_idx, device):
    """
    Convert a list of sample ID strings to a tensor of integer donor indices.

    Args:
        sample_ids:           list[str]  — e.g. ["SKF.SHED.D30.W1", ...]
        sampleID_map_donorID: dict       — maps sample ID → donor string
        donor_to_idx:         dict       — maps donor string → integer index
        device:               torch.device

    Returns:
        torch.LongTensor of shape (B,) with donor indices
    """
    donor_indices = []
    for sid in sample_ids:
        donor_str = sampleID_map_donorID[sid]
        donor_indices.append(donor_to_idx[donor_str])
    return torch.tensor(donor_indices, dtype=torch.long, device=device)


# ===========================================================================
# Contrastive Pre-training Loop
# ===========================================================================
def pretrain_contrastive(
    model,
    data_processor,
    device,
    out_dir,
    num_epochs=150,
    learning_rate=5e-4,
    temperature=0.07,
    sigma=50.0,
    batch_size=64,
    patience=25,
    warmup_epochs=10,
):
    """
    Contrastive pre-training loop.

    Args:
        model:          ContrastiveProjector wrapping your base model
        data_processor: your DataProcessor (must have .sampleID_map_donorID
                        and .sample_data(epoch))
        device:         torch device
        out_dir:        directory to save checkpoints
        num_epochs:     max pre-training epochs
        learning_rate:  peak learning rate
        temperature:    contrastive loss temperature (lower = sharper)
        sigma:          ADD distance scale for soft positives
        batch_size:     larger is better for contrastive (64–128)
        patience:       early stopping patience
        warmup_epochs:  linear warmup epochs

    Returns:
        model with best checkpoint loaded
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)

    # ---- Build donor index mapping ----
    sampleID_map_donorID = data_processor.sampleID_map_donorID
    unique_donors = sorted(set(sampleID_map_donorID.values()))
    donor_to_idx = {donor: i for i, donor in enumerate(unique_donors)}
    print(f"Contrastive pre-training: {len(unique_donors)} unique donors")

    # ---- Loss, optimizer, scheduler ----
    criterion = SoftADDContrastiveLoss(
        temperature=temperature, sigma=sigma, donor_penalty=0.5
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # ---- Data ----
    train_dataset, val_dataset = data_processor.sample_data(epoch=0)

    # NOTE: import sparse_collate_fn from your data module
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn
    )

    # ---- Tracking ----
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lrs = []

    print("=" * 60)
    print("CONTRASTIVE PRE-TRAINING")
    print(f"  Temperature : {temperature}")
    print(f"  Sigma       : {sigma}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {learning_rate}")
    print(f"  Warmup      : {warmup_epochs} epochs")
    print(f"  Patience    : {patience}")
    print("=" * 60)

    for epoch in range(num_epochs):
        # ---- Linear warmup ----
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Re-sample if your dataset supports it
        if hasattr(train_dataset, 'sample_epoch_init'):
            train_dataset.sample_epoch_init(epoch)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn
            )

        loop = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}/{num_epochs}')
        for batch in loop:
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4
            masks = batch['masks'].to(device)
            targets = batch['outdoor_add_0'].to(device)
            features = batch['season'].to(device)
            env = batch['env'].to(device, dtype=torch.float32)

            # Build donor ID tensor
            donor_ids = extract_donor_ids(
                batch['SampleID'], sampleID_map_donorID, donor_to_idx, device
            )

            if torch.isnan(targets).any():
                continue

            optimizer.zero_grad()

            # Forward: encoder + projection head
            encoder_output, projections = model(
                embeddings, abundances, masks, features, env=env
            )

            loss = criterion(projections, targets, donor_ids)

            if torch.isnan(loss) or loss.item() == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=loss.item())

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss_epoch = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['outdoor_add_0'].to(device)
                features = batch['season'].to(device)
                env = batch['env'].to(device, dtype=torch.float32)

                donor_ids = extract_donor_ids(
                    batch['SampleID'], sampleID_map_donorID, donor_to_idx, device
                )

                _, projections = model(
                    embeddings, abundances, masks, features, env=env
                )
                loss = criterion(projections, targets, donor_ids)

                if not torch.isnan(loss):
                    val_loss_epoch += loss.item()
                    val_batches += 1

        val_loss = val_loss_epoch / max(val_batches, 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        # ---- Checkpointing / early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'base_model_state_dict': model.base_model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'donor_to_idx': donor_to_idx,
            }, os.path.join(out_dir, 'contrastive_pretrained.pt'))
            print(f"  -> Saved best checkpoint (val_loss: {val_loss:.4f})")
        else:
            if epoch >= warmup_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # ---- Plot training curves ----
    _plot_pretrain_curves(train_losses, val_losses, lrs, out_dir)

    # ---- Save logs ----
    pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'lr': lrs,
    }).to_csv(os.path.join(out_dir, 'contrastive_pretrain_logs.csv'), index=False)

    # ---- Load best checkpoint ----
    ckpt_path = os.path.join(out_dir, 'contrastive_pretrained.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best contrastive checkpoint from epoch {checkpoint['epoch']+1}")
    print(f"Best val contrastive loss: {checkpoint['val_loss']:.4f}")

    return model


def _plot_pretrain_curves(train_losses, val_losses, lrs, out_dir):
    """Save loss and LR curves to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Contrastive Loss')
    axes[0].set_title('Contrastive Pre-training Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lrs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'contrastive_pretrain_curves.png'), dpi=150)
    plt.close(fig)


# ===========================================================================
# Freeze / Unfreeze helpers
# ===========================================================================
def freeze_encoder(model):
    """
    Freeze everything except the final regression / prediction heads.
    Adjust the name filters to match your BasicRegressor architecture.
    """
    head_keywords = ['regressor', 'regression', 'head', 'fc_out', 'output_layer',
                     'grl', 'adversary', 'subject_classifier']

    for name, param in model.named_parameters():
        is_head = any(kw in name.lower() for kw in head_keywords)
        param.requires_grad = is_head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen encoder: {trainable:,}/{total:,} parameters trainable "
          f"({100 * trainable / total:.1f}%)")


def unfreeze_all(model):
    """Unfreeze all parameters for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen: {trainable:,} parameters trainable")


# ===========================================================================
# Full Pipeline: Pretrain → Fine-tune
# ===========================================================================
def train_with_contrastive_pretraining(
    base_model,
    data_processor,
    device,
    out_dir,
    encoder_dim=256,
    proj_dim=64,
    # Pre-training params
    pretrain_epochs=150,
    pretrain_lr=5e-4,
    pretrain_batch_size=64,
    temperature=0.07,
    sigma=50.0,
    # Fine-tuning params
    finetune_epochs=1000,
    finetune_lr=1e-4,
):
    """
    Full pipeline:
      1. Contrastive pre-training (learn donor-invariant temporal representations)
      2a. Fine-tune regression head only (encoder frozen)
      2b. End-to-end fine-tuning with lower LR
    """
    os.makedirs(out_dir, exist_ok=True)

    # ==================================================
    # PHASE 1: Contrastive Pre-training
    # ==================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Contrastive Pre-training")
    print("=" * 60)

    contrastive_model = ContrastiveProjector(
        base_model, encoder_dim=encoder_dim, proj_dim=proj_dim
    )

    contrastive_model = pretrain_contrastive(
        model=contrastive_model,
        data_processor=data_processor,
        device=device,
        out_dir=os.path.join(out_dir, 'phase1_contrastive'),
        num_epochs=pretrain_epochs,
        learning_rate=pretrain_lr,
        temperature=temperature,
        sigma=sigma,
        batch_size=pretrain_batch_size,
    )

    # Extract the pre-trained base model (discard projection head)
    pretrained_base = contrastive_model.base_model

    # ==================================================
    # PHASE 2a: Train regression head only (encoder frozen)
    # ==================================================
    print("\n" + "=" * 60)
    print("PHASE 2a: Training regression head (encoder frozen)")
    print("=" * 60)

    freeze_encoder(pretrained_base)

    phase2a_dir = os.path.join(out_dir, 'phase2a_frozen')
    os.makedirs(phase2a_dir, exist_ok=True)

    # Use your existing GRL training function
    results_frozen = train_model(
        model=pretrained_base,
        num_epochs=min(100, finetune_epochs // 4),
        learning_rate=finetune_lr * 5,  # Higher LR for head-only training
        device=device,
        out_dir=phase2a_dir,
        data_processor=data_processor,
    )

    print(f"Phase 2a results: {results_frozen}")

    # ==================================================
    # PHASE 2b: End-to-end fine-tuning
    # ==================================================
    print("\n" + "=" * 60)
    print("PHASE 2b: End-to-end fine-tuning")
    print("=" * 60)

    unfreeze_all(pretrained_base)

    phase2b_dir = os.path.join(out_dir, 'phase2b_finetune')
    os.makedirs(phase2b_dir, exist_ok=True)

    results_full = train_model(
        model=pretrained_base,
        num_epochs=finetune_epochs,
        learning_rate=finetune_lr,
        device=device,
        out_dir=phase2b_dir,
        data_processor=data_processor,
    )

    print(f"Phase 2b results: {results_full}")
    return results_full


# ===========================================================================
# Integration into your existing main() loop
# ===========================================================================
def main_with_pretraining():
    """
    Drop-in replacement for your main() that adds contrastive pre-training.
    Mirrors the structure of your existing main().
    """
    setup_seed(42)

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Paths (same as your main) ----
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
    output_dir = os.path.join(data_dir, "microbiome_model/results/ContrastivePretrain/")
    biom_file = os.path.join(data_dir, "process_data_all/exported-feature-table/feature-table.biom")
    sheds_file = os.path.join(data_dir, "data/new_data/metadata_sheds.tsv")
    metadata_file = os.path.join(data_dir, "data/new_data/combined_metadata_merged.tsv")
    embedding_path = os.path.join(data_dir, "data/embeddings/all_data.h5")

    heldout_samples = pd.read_csv(sheds_file, sep="\t")['DonorID'].unique().tolist()
    computed = ["D28"]
    heldout_samples = [s for s in heldout_samples if s not in computed]

    os.makedirs(output_dir, exist_ok=True)
    seed_values = [42, 1337, 2048]
    runs = 2

    for j, heldout in enumerate(heldout_samples):
        print(f"\n{'='*60}")
        print(f"Heldout sample: {heldout} ({j+1}/{len(heldout_samples)})")
        print(f"{'='*60}")

        out_dir = os.path.join(output_dir, heldout)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(runs):
            print(f"\n--- Run {i+1}/{runs} for {heldout} ---")
            cleanup_resources()

            setup_seed(seed_values[i])

            args = Arguments(
                biom_file=biom_file,
                metadata_file=metadata_file,
                tree_path=None,
                embedding_file=embedding_path,
                embedding="DNABERT",
                heldout=heldout,
                sort_asvs=True,
            )
            data_processor = DataProcessor(args)
            data_processor.load_data(multitask=True, column="bi_month_name")

            # Build fresh model
            model = BasicRegressor(128, 512, 4, 2, 0.3, pe=True, GRL=True)

            result_dir = os.path.join(out_dir, f"run_{i+1}")
            os.makedirs(result_dir, exist_ok=True)

            # Save config
            model_config = {
                "input_dim": 128,
                "hidden_dim": 512,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "pe": True,
                "GRL": True,
                "pretrain": "contrastive",
            }
            with open(os.path.join(result_dir, f"model_config_{i+1}.json"), "w") as f:
                json.dump(model_config, f, indent=4)

            # ---- Run full pipeline ----
            res = train_with_contrastive_pretraining(
                base_model=model,
                data_processor=data_processor,
                device=device,
                out_dir=result_dir,
                encoder_dim=128,   # must match your model's encode() output dim
                proj_dim=64,
                pretrain_epochs=150,
                pretrain_lr=5e-4,
                pretrain_batch_size=64,
                temperature=0.5,
                sigma=1.0,
                finetune_epochs=1000,
                finetune_lr=1e-4,
            )

            res["seed"] = seed_values[i]
            with open(os.path.join(result_dir, "res.json"), "w") as f:
                json.dump(res, f, indent=4)

            print(f"Run {i+1} results: {res}")

            del model
            cleanup_resources()


if __name__ == "__main__":
    import logging
    main_with_pretraining()
