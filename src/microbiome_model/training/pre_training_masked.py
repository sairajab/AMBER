"""
Masked Abundance Pre-training + Task-Specific Fine-tuning for Microbiome Models.

Pipeline:
  Phase 1:  Masked abundance pre-training (task-agnostic, no labels)
  Phase 2a: Fine-tune regression/GRL heads only (encoder frozen)
  Phase 2b: End-to-end fine-tuning with GRL

Usage:
    # Option A: Run full pipeline from main()
    python pretrain_masked_abundance.py

    # Option B: Import and use programmatically
    from pretrain_masked_abundance import run_pretrain_then_finetune
    results = run_pretrain_then_finetune(model, data_processor, device, out_dir)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import biom

# ---------------------------------------------------------------------------
# Import your project modules here — adjust paths as needed
# ---------------------------------------------------------------------------
from microbiome_model.models.orig import BasicRegressor, BasicRegressorwithUnifrac
# from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn # Overridden below
from microbiome_model.data.optimized_embedding_loader import MemoryMappedEmbeddingLoader as EmbeddingLoader 
from microbiome_model.data.dataset import finetuned_dna_bert
from microbiome_model.data.dataset import DataProcessor, Arguments
from torch.utils.data import Dataset, Subset
mask_ratio = 0.15

# ===========================================================================
# Data Loader Logic
# ===========================================================================


class DataLoaderMirobiome(Dataset): 
    """
    Custom Dataset that can re-sample data each epoch if the underlying
    dataset supports it (e.g., by implementing sample_epoch_init()).
    """

    def __init__(self, biom_file, embedding_path, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
        self.biom_file = biom_file
        self.fulltable = biom.load_table(biom_file)
        self.embedding_path = embedding_path
        print("Original table shape: ", self.fulltable.shape)
        
        self.sequences = self.fulltable.ids(axis='observation')
        if os.path.exists(embedding_path):
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        else:
            self.create_embeddings(output_path=embedding_path)
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
            
        self.table = self.fulltable.subsample(5000, axis="observation")
        self.load_sequences()
        self.sample_ids = self.table.ids(axis='sample')
        self.sort_asvs = True
        self.num_bins = 100
        self.obs_ids = self.table.ids(axis='observation') 
        self.table_data = self._table_data(self.table)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # --- NEW: Generate 80-10-10 split indices ---
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self._create_splits()

    def _create_splits(self):
        """Calculates exact integer boundaries for an 80-10-10 split."""
        num_samples = self.table.shape[1]
        
        # Randomize indices if shuffle is True, otherwise keep sequential
        if self.shuffle:
            # Using a fixed generator seed here is highly recommended 
            # so your splits stay identical across script restarts.
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(num_samples, generator=generator).tolist()
        else:
            indices = list(range(num_samples))
            
        train_split = int(0.8 * num_samples)
        val_split = int(0.9 * num_samples) # 80% + 10%
        
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:val_split]
        self.test_indices = indices[val_split:]
        
        print(f"Dataset split: {len(self.train_indices)} Train | {len(self.val_indices)} Val | {len(self.test_indices)} Test")

    def sample_data(self, epoch=None):
        """
        Returns PyTorch Subsets for Train, Val, and Test splits.
        If you resample the table for this epoch, the Subsets will 
        automatically reflect the new observation data while keeping 
        the sample routing consistent.
        """
        if epoch is not None and hasattr(self, 'sample_epoch_init'):
             self.sample_epoch_init(epoch)
             
        train_subset = Subset(self, self.train_indices)
        val_subset = Subset(self, self.val_indices)
        test_subset = Subset(self, self.test_indices)
        
        return train_subset, val_subset, test_subset
    
    
    def create_embeddings(self, output_path="embeddings.h5"):
        seq_dict = {seq: seq for seq in self.sequences}
        print("Creating embeddings for ", len(seq_dict), " sequences")
        finetuned_dna_bert(output_path, seq_dict, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Embeddings created and saved to ", output_path)

    def _table_data(self, table):
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, (coo.row, coo.col))
        table_mask = data > 0
        data = data[table_mask]
        row = row[table_mask]
        col = col[table_mask]
        return data, row, col, shape

    def load_sequences(self):
        if self.table is None:
            raise ValueError("BIOM table not loaded yet. Call load_biom() first.")
        s_ids = self.table.ids(axis="observation")
        self.sequences = {s_id: s_id for s_id in s_ids}
        return self.sequences

    def sample_epoch_init(self, epoch):
        print(f"Epoch {epoch+1} initialization complete. Resampling table.")
        self.table = self.fulltable.subsample(5000, axis="observation", seed=epoch) 
        self.obs_ids = self.table.ids(axis='observation')
        self.sample_ids = self.table.ids()
        self.table_data = self._table_data(self.table)
    
    def __len__(self):
        return self.table.shape[1] 
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        s_mask = self.table_data[1] == idx
        abundances = self.table_data[0][s_mask]

        s_obs = self.table_data[2][s_mask]
        s_obs_ids = self.obs_ids[s_obs]
        
        abundances = abundances / (abundances.sum() + 1e-6) 
        abundances = abundances * 1e4  

        if len(abundances) > 0:
            abundances_t = torch.tensor(abundances, dtype=torch.float32)
            ranks = torch.argsort(torch.argsort(abundances_t)).float()  
            binned_abundances = (ranks / len(abundances_t) * self.num_bins).long() + 1
        else:
            binned_abundances = torch.empty(0, dtype=torch.long)
            abundances_t = torch.empty(0, dtype=torch.float32)

        if self.sort_asvs:
            sorted_order = torch.argsort(abundances_t, descending=True)
        else:
            sorted_order = torch.arange(len(abundances_t), dtype=torch.long)

        if hasattr(self.embedding_loader, 'get_embeddings_batch'):
            sample_embeddings_list = self.embedding_loader.get_embeddings_batch(s_obs_ids)
            sample_embeddings = torch.stack(sample_embeddings_list)
        else:
            sample_embeddings = torch.stack([
                self.embedding_loader.get_embedding(seq_id) for seq_id in s_obs_ids
            ])

        abundances_t = abundances_t[sorted_order].reshape(-1, 1)  
        sample_embeddings = sample_embeddings[sorted_order]  
        seqs_ids = [s_obs_ids[i] for i in sorted_order.numpy()]  
        binned_abundances = binned_abundances[sorted_order]  

        # Masking logic
        mask_indices = torch.rand(len(binned_abundances)) < mask_ratio
        binned_abundances_masked = binned_abundances.clone()
        binned_abundances_masked[mask_indices] = 0  # 0 is the masked token

        return {
            'SampleID': sample_id,
            'embeddings': sample_embeddings,
            'abundances': abundances_t,
            'binned_abundances': binned_abundances,
            'binned_abundances_masked': binned_abundances_masked,
            'seqs_ids': seqs_ids
        }

def collate_fn(batch):
    sample_ids = [item['SampleID'] for item in batch]
    seqs_ids = [item['seqs_ids'] for item in batch]

    embeddings = [item['embeddings'] for item in batch]
    abundances = [item['abundances'] for item in batch]
    binned_abundances = [item['binned_abundances'] for item in batch]
    masked_binned_abundances = [item['binned_abundances_masked'] for item in batch]

    max_len = max(e.shape[0] for e in embeddings)
    
    if len(embeddings[0].shape) == 2:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1])  
    else:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2])  
        
    padded_abundances = torch.zeros(len(batch), max_len, 1)  
    padded_seqs_ids = [None] * len(batch)  
    batch_masks = torch.zeros(len(batch), max_len)
    padded_binned_abundances = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_masked_binned_abundances = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb  
        padded_abundances[i, :length, :] = abun  
        padded_binned_abundances[i, :length] = binned_abundances[i]  
        padded_masked_binned_abundances[i, :length] = masked_binned_abundances[i]  
        batch_masks[i, :length] = 1

    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))
        
    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings, 
        'abundances': padded_abundances, 
        'masks': batch_masks,
        'seqs_ids': padded_seqs_ids,
        'binned_abundances': padded_binned_abundances, 
        'masked_binned_abundances': padded_masked_binned_abundances 
    }

# ===========================================================================
# Masked Abundance Pre-training Wrapper
# ===========================================================================

class MaskedAbundancePretraining(nn.Module):
    def __init__(self, base_model=None, input_dim=128, bins=100):
        super().__init__()
        self.base_model = base_model # Kept for weight extraction later

        # +1 for a dedicated mask/pad token index (0)
        self.bins_embeddings = nn.Embedding(bins + 1, input_dim) 

        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bins + 1), # Output shape matches number of bins (classes)
        )

    def forward(self, embeddings, masked_binned_abundances, masks):
        # 1. Get binned embeddings from the ALREADY MASKED input
        bin_embs = self.bins_embeddings(masked_binned_abundances) 

        # print("Embeddings Shape:", embeddings.shape)
        # print("Bin Embeddings Shape:", bin_embs.shape)
        # 2. Combine with base embeddings
        input_embeddings = embeddings + bin_embs 

        # 3. Transformer
        padding_mask = ~masks.bool() 
        out = self.transformer_encoder(input_embeddings, src_key_padding_mask=padding_mask) 
        
        # 4. Predict probabilities across all bins per token
        logits = self.decoder(out) # (B, L, bins+1)
        
        return out, logits


def compute_loss_and_metrics(logits, true_bins, masked_inputs, padding_masks):
    """
    Computes CrossEntropyLoss only on the tokens that were masked.
    logits: (B, L, bins+1)
    true_bins: (B, L)
    masked_inputs: (B, L) - contains 0 where masked
    padding_masks: (B, L) - 1 if valid ASV, 0 if padding
    """
    # Create a mask identifying which tokens were hidden AND are valid
    loss_mask = (masked_inputs == 0) & (padding_masks == 1)
    
    if not loss_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True), {'loss': 0.0, 'acc': 0.0, 'n_masked': 0}
        
    masked_logits = logits[loss_mask]     # (N, bins+1)
    masked_targets = true_bins[loss_mask] # (N)
    
    loss = F.cross_entropy(masked_logits, masked_targets)
    
    # Calculate accuracy
    preds = masked_logits.argmax(dim=-1)
    acc = (preds == masked_targets).float().mean().item()
    
    return loss, {'loss': loss.item(), 'acc': acc, 'n_masked': loss_mask.sum().item()}

# ===========================================================================
# Pretraining Loop
# ===========================================================================

def pretrain_masked_abundance(
    input_dim,
    data_processor, 
    device,
    out_dir,
    base_model=None, # Passed in to save weights
    num_epochs=200,
    learning_rate=5e-4,
    batch_size=32,
    patience=30,
    warmup_epochs=15,
    log_every=5,
):
    os.makedirs(out_dir, exist_ok=True)

    pretrain_model = MaskedAbundancePretraining(
        base_model=base_model,
        input_dim=input_dim,
        bins=100
    ).to(device)

    optimizer = AdamW(pretrain_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # ---- Data ----
    # Note: Adapting this slightly depending on what data_processor.sample_data returns.
    # Assuming it returns instances of the DataLoaderMirobiome defined above.
    train_dataset, val_dataset, test_dataset = data_processor.sample_data()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn, # Using local collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_fn,
    )

    # ---- Tracking ----
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [],
    }

    print("=" * 70)
    print("PHASE 1: MASKED ABUNDANCE PRE-TRAINING")
    print("=" * 70)

    for epoch in range(num_epochs):

        # ---- Warmup ----
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # ---- Train ----
        pretrain_model.train()
        epoch_metrics = {'loss': 0, 'acc': 0}
        n_batches = 0

        train_dataset.dataset.sample_epoch_init(epoch)
    
        loop = tqdm(train_loader, desc=f'Pretrain {epoch+1}/{num_epochs}', leave=False)
        for batch in loop:
            embeddings = batch['embeddings'].to(device)
            masks = batch['masks'].to(device)
            true_bins = batch['binned_abundances'].to(device)
            masked_bins = batch['masked_binned_abundances'].to(device)

            optimizer.zero_grad()
            logits = pretrain_model(embeddings, masked_bins, masks)
            
            loss, metrics = compute_loss_and_metrics(logits, true_bins, masked_bins, masks)

            if torch.isnan(loss) or metrics['n_masked'] == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_metrics['loss'] += metrics['loss']
            epoch_metrics['acc'] += metrics['acc']
            n_batches += 1
            loop.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['acc']:.3f}")

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_metrics['loss'] / max(n_batches, 1)
        train_acc = epoch_metrics['acc'] / max(n_batches, 1)

        # ---- Validate ----
        pretrain_model.eval()
        val_metrics = {'loss': 0, 'acc': 0}
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                masks = batch['masks'].to(device)
                true_bins = batch['binned_abundances'].to(device)
                masked_bins = batch['masked_binned_abundances'].to(device)

                logits = pretrain_model(embeddings, masked_bins, masks)
                loss, metrics = compute_loss_and_metrics(logits, true_bins, masked_bins, masks)
                
                if not torch.isnan(loss) and metrics['n_masked'] > 0:
                    val_metrics['loss'] += metrics['loss']
                    val_metrics['acc'] += metrics['acc']
                    val_batches += 1

        val_loss = val_metrics['loss'] / max(val_batches, 1)
        val_acc = val_metrics['acc'] / max(val_batches, 1)

        # ---- Record ----
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss:{train_loss:.4f} Acc:{train_acc:.3f} | "
                  f"Val Loss:{val_loss:.4f} Acc:{val_acc:.3f} | "
                  f"LR:{current_lr:.6f}")

        # ---- Checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_dict = {
                'full_model_state_dict': pretrain_model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            if base_model is not None:
                save_dict['base_model_state_dict'] = pretrain_model.base_model.state_dict()
                
            torch.save(save_dict, os.path.join(out_dir, 'pretrained_best.pt'))
            print(f"  -> Saved best (val_loss:{val_loss:.4f} Acc:{val_acc:.3f})")
        else:
            if epoch >= warmup_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        with open(os.path.join(out_dir, 'pretrain_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                    f"lr={current_lr:.8f}\n")

    # ---- Plots ----
    _plot_pretrain_history(history, out_dir)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'pretrain_history.csv'), index=False)

    # ---- Load best weights back into base_model ----
    ckpt = torch.load(os.path.join(out_dir, 'pretrained_best.pt'), map_location=device)
    if base_model is not None and 'base_model_state_dict' in ckpt:
        base_model.load_state_dict(ckpt['base_model_state_dict'])
        print(f"\nLoaded best pre-trained weights from epoch {ckpt['epoch']+1}")
        
    return base_model


def _plot_pretrain_history(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0].plot(history['val_loss'], label='Val', alpha=0.8)
    axes[0].set_title('Masked Bin CrossEntropy Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', alpha=0.8)
    axes[1].plot(history['val_acc'], label='Val', alpha=0.8)
    axes[1].set_title('Bin Prediction Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['lr'], alpha=0.8)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pretrain_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)




# ===========================================================================
# Full Pipeline
# ===========================================================================

def main():
    setup_seed(42)
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Paths ----
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
    output_dir = os.path.join(data_dir, "microbiome_model/results/MaskedPretrainSkin/")
    biom_file = os.path.join(data_dir, "AGP/skin_all.biom")
    embedding_path = os.path.join(data_dir, "data/embeddings/skin_data.h5")


    dataset = DataLoaderMirobiome(biom_file=biom_file, embedding_path=embedding_path)

    os.makedirs(output_dir, exist_ok=True)
    seed_values = [42, 1337, 2048]
    runs = 2
    pretrain_masked_abundance(
        input_dim=768, data_processor=dataset, device=device, out_dir=output_dir,)
    

if __name__ == "__main__":
    main()