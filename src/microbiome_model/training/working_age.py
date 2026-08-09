from microbiome_model.data.dataset import finetuned_dna_bert, dna_bert
from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.optimized_embedding_loader import MemoryMappedEmbeddingLoaderNew as EmbeddingLoader 
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import biom
import os 
import numpy as np
from microbiome_model.models.orig import BasicRegressor,BasicRegressorNew, BasicRegressorwithUnifrac
from microbiome_model.models.zoo import BasicRegressorGRL, ClusteredRegressor , MicrobiomeTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import torch.nn as nn
from microbiome_model.losses.core import compute_count_loss, PairwiseLoss
from sklearn.model_selection import GroupShuffleSplit

class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim+1, 1)

    def forward(self, embeddings, abundances, masks):
            # 1. Concatenate so the network SEES the abundance to make its decision
            # embeddings: [B, Seq_Len, Features], abundances: [B, Seq_Len, 1]
            concat_feats = torch.cat([embeddings, abundances], dim=-1)
            
            # 2. Calculate scores based on BOTH the sequence and its abundance
            raw_scores = self.attention(concat_feats)  # [B, Seq_Len, 1]
            
            # 3. Mask out padding
            raw_scores = raw_scores.masked_fill(masks.unsqueeze(-1) == 0, -1e9)
            
            # 4. Softmax (Guarantees the weights sum to exactly 1.0)
            attn_weights = torch.softmax(raw_scores, dim=1)  # [B, Seq_Len, 1]
            
            # 5. Pool using the learned weights ONLY (preserves full vector magnitude)
            pooled = torch.sum(attn_weights * embeddings, dim=1)  # [B, Features]
            
            return pooled
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, abundances, masks):

        scaled_embeddings = (embeddings * abundances) + embeddings   

        x = self.attention(scaled_embeddings)
        x = x.masked_fill(masks.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(x, dim=1)  # [B, Seq_Len, 1]
        pooled = torch.sum(attn_weights * embeddings, dim=1)            # [B, Features]
        return pooled
      
    
class AgeDatasetLoader:

    def __init__(self, biom_table_path, metadata_df, target_column, embedding_path=None, n_bins=50):
        self.table = biom.load_table(biom_table_path)
        self.metadata_df = metadata_df
        self.target_column = target_column
        self.sample_ids = self.table.ids(axis='sample')
        self.sample_ids = self.metadata_df.index.intersection(self.sample_ids).tolist()
        self.subsampled =self.table.subsample(5000, axis = "observation")
        print("Original table shape: ", self.table.shape)
        self.sequences = self.table.ids(axis='observation')
        self.n_bins = n_bins
        if os.path.exists(embedding_path):
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        else:
            self.create_embeddings(output_path=embedding_path)
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
    
    def create_embeddings(self, output_path="embeddings.h5"):
        #map seq to seq in dict
        seq_dict = {seq: seq for seq in self.sequences}
        print("Creating embeddings for ", len(seq_dict), " sequences")
        finetuned_dna_bert(output_path, seq_dict, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #dna_bert(output_path, seq_dict, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Embeddings created and saved to ", output_path)
    
    def create_splits(self, test_split=0.2, random_seed=42):
        import random
        random.seed(random_seed)
        shuffled = list(self.sample_ids)
        random.shuffle(shuffled)
        #self.sample_ids = shuffled
        # Split sample IDs into train and test
        train_ids = self.sample_ids[:int(len(self.sample_ids) * (1 - test_split))]
        test_ids = self.sample_ids[int(len(self.sample_ids) * (1 - test_split)):]
        val_ids = test_ids[:int(len(test_ids)*0.5)]
        test_ids = test_ids[int(len(test_ids)*0.5):]
        return train_ids, val_ids, test_ids


    
    def get_dataset(self, split="train", test_split=0.2, random_seed=42):
        import random
        random.seed(random_seed)
        shuffled = list(self.sample_ids)
        random.shuffle(shuffled)
        #self.sample_ids = shuffled
        # Split sample IDs into train and test
        # if split == "train":
        #     sample_ids = self.sample_ids[:int(len(self.sample_ids) * (1 - test_split))]
        # elif split == "val":
        #     sample_ids = self.sample_ids[int(len(self.sample_ids) * (1 - test_split) * 0.5):int(len(self.sample_ids) * (1 - test_split))]
        # else:
        #     sample_ids = self.sample_ids[int(len(self.sample_ids) * (1 - test_split)):]

        n = len(self.sample_ids)
        train_end = int(n * 0.8)
        val_end   = int(n * 0.9)

        if split == "train":
            sample_ids = shuffled[:train_end]
        elif split == "val":
            sample_ids = shuffled[train_end:val_end]
        else:
            sample_ids = shuffled[val_end:]
        
        #sample_ids = train_sample_ids + val_sample_ids + test_sample_ids

        missing = set(sample_ids) - set(self.table.ids(axis='sample'))
        if missing:
            raise ValueError(f"{len(missing)} sample_ids not in BIOM. Examples: {list(missing)[:10]}")


        #check overlap between train and test sample ids
        # print("Overlap between train and val: ", set(train_sample_ids).intersection(set(val_sample_ids)))
        # print("Overlap between val and test: ", set(val_sample_ids).intersection(set(test_sample_ids)))
        # train_table = self.table.filter(train_sample_ids, axis='sample', inplace=False)
        # val_table = self.table.filter(val_sample_ids, axis='sample', inplace=False)
        # test_table = self.table.filter(test_sample_ids, axis='sample', inplace=False)
        split_table = self.table.filter(sample_ids, axis='sample', inplace=False)
        #print("Subsampling..." , split,  split_table.shape)
        split_table = split_table.subsample(5000, axis = "observation")
        # print(split_table.shape)
        
        # Create dataset
        dataset = MicrobiomeSparseDataset(
            biom_table=split_table,
            sample_targets=self.metadata_df.loc[sample_ids][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            test_split=(split=="test"),
            seed=random_seed,
            n_bins=self.n_bins
        )

        # val_dataset = MicrobiomeSparseDataset(
        #     biom_table=val_table,
        #     sample_targets=self.metadata_df.loc[val_sample_ids][self.target_column].to_dict(),
        #     sort_asvs=True,
        #     embedding_loader=self.embedding_loader,
        #     kmer_seqs=None,
        #     one_hot_seqs=None,
        #     test_split=(split=="test"),
        #     seed=random_seed,
        #     n_bins=self.n_bins
        # )

        # test_dataset = MicrobiomeSparseDataset(
        #     biom_table=test_table,
        #     sample_targets=self.metadata_df.loc[test_sample_ids][self.target_column].to_dict(),
        #     sort_asvs=True,
        #     embedding_loader=self.embedding_loader,
        #     kmer_seqs=None,
        #     one_hot_seqs=None,
        #     test_split=(split=="test"),
        #     seed=random_seed,
        #     n_bins=self.n_bins
        # )

        # self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        # self.test_dataset = test_dataset

        # self.train_targets = self.metadata_df.loc[train_sample_ids][self.target_column].to_list()
        # self.val_targets = self.metadata_df.loc[val_sample_ids][self.target_column].to_list()
        # self.test_targets = self.metadata_df.loc[test_sample_ids][self.target_column].to_list()

        return dataset, split_table, self.metadata_df.loc[sample_ids][self.target_column].to_list()

    def get_dataset_by_json(self, json_path="/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/group_assignments_gut_data.json"):
        import json
        with open(json_path, "r") as f:
            group_assignments = json.load(f)
        
        train_indices = group_assignments["train_indices"]
        test_indices = group_assignments["test_indices"]

        self.json = json_path
        


        dataset_name = "skin"
        val_indices = []
        val_indices_file = f"/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/val_samples_{dataset_name}.txt"
        if not os.path.exists(val_indices_file):
            
            train_host_ids = self.metadata_df.loc[train_indices]['host_subject_id']

            ## group by host_subject_id and then split by host_subject_id to ensure no overlap between train and val
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            _ , val_idx = next(gss.split(train_indices, groups=train_host_ids))
            for i in val_idx:
                val_indices.append(train_indices[i])
            print(val_indices)
        else:

            with open(val_indices_file, "r") as f:
                for line in f.readlines():
                    val_indices.append(line.strip())

        train_indices = list(set(train_indices) - set(val_indices))
        train_table = self.table.filter(train_indices, axis='sample', inplace=False)
        test_table = self.table.filter(test_indices, axis='sample', inplace=False)
        val_table = self.table.filter(val_indices, axis='sample', inplace=False)

        print(len(val_indices), len(train_indices), len(test_indices))

        # split_table = split_table.subsample(5000, axis = "observation")
        # print(split_table.shape)
        
        print("Train table shape: ", train_table.shape)
        # Create dataset
        self.train_dataset = MicrobiomeSparseDataset(
            biom_table=train_table,
            sample_targets=self.metadata_df.loc[train_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=42,
            n_bins=self.n_bins,
            subsample= True
        )




        # print("Validation table shape: ", val_table.shape)
        self.val_dataset = MicrobiomeSparseDataset(
            biom_table=val_table,
            sample_targets=self.metadata_df.loc[val_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=42,
            n_bins=self.n_bins,
            subsample=True
        )
        print("Test table shape: ", test_table.shape)
        self.test_dataset = MicrobiomeSparseDataset(
            biom_table=test_table,
            sample_targets=self.metadata_df.loc[test_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=42,
            n_bins=self.n_bins,
            subsample=True
        )
        self.train_targets = self.metadata_df.loc[train_indices][self.target_column].to_list()
        self.val_targets = self.metadata_df.loc[val_indices][self.target_column].to_list()
        self.test_targets = self.metadata_df.loc[test_indices][self.target_column].to_list()

        # get train targets mean and std
        mean_value = np.mean(self.train_targets)
        std_value = np.std(self.train_targets)

        print("mean value", mean_value )
        print("std value", std_value)
    

    

    
def collate_fn(batch):
        
    sample_ids = [item['SampleID'] for item in batch]
    targets = torch.stack([item['outdoor_add_0']/10 for item in batch])
    seqs_ids = [item['seqs_ids'] for item in batch]

    # Extract embeddings and abundances
    embeddings = [item['embeddings'] for item in batch]
    abundances = [item['abundances'] for item in batch]
    bins_abundance = [item['binned_abundances'] for item in batch]
    print("EMbeddings shape:", embeddings[0].shape)
    # print("Abundances shape:", abundances[0].shape)
    # print(abundances[0].max(), abundances[0].min(), abundances[0].sum())

    # Find the max length of embeddings in the batch for padding
    max_len = max(e.shape[0] for e in embeddings)
    # Pad embeddings and abundances to ensure equal shape in batch
    if len(embeddings[0].shape) == 2:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1])  # (Batch, Max_Len, Emb_Dim)
    else:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2])  # (Batch, Max_Len, Emb_Dim)
    padded_abundances = torch.zeros(len(batch), max_len, 1)  # (Batch, Max_Len, 1)
    padded_seqs_ids = [None] * len(batch)  # Placeholder for seqs_ids
    batch_masks = torch.zeros(len(batch), max_len)
    padded_bins_abundance = torch.zeros((len(batch), max_len), dtype=torch.long)  # (Batch, Max_Len, Num_Bins)
    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb  # Copy actual data
        padded_abundances[i, :length, :] = abun  # Copy actual abundances
        padded_bins_abundance[i, :length] = bins_abundance[i]  # Copy actual bins abundances
        batch_masks[i, :length] = 1
    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))
    

    actual_lengths = batch_masks.sum(dim=1, keepdim=True)
    weighted_sum = (padded_embeddings * padded_abundances).sum(dim=1) / actual_lengths
    #print("Batch shape:", padded_abundances.shape)
    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings,  # (Batch, Max_Len, Emb_Dim)
        'abundances': padded_abundances,  # (Batch, Max_Len, 1)
        'masks': batch_masks,
        'label': targets,  # (Batch, 1),
        'seqs_ids': padded_seqs_ids,
        'weighted_sum': weighted_sum,  # (Batch, Emb_Dim),
        'bins_abundance': padded_bins_abundance  # (Batch, Max_Len, Num_Bins)
    }

class CLR(nn.Module):
    """Project embeddings to lower dim, then pool"""
    def __init__(self, emb_dim=768, proj_dim=64, hidden_dim=256, dropout=0.3):
        super().__init__()
        # Compress each ASV embedding first
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        
        # Attention pooling on compressed embeddings
        self.attn = nn.Linear(proj_dim + 1, 1)  # +1 for CLR abundance
        
        self.regressor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings, abundances, masks):
        # 1. Project embeddings down
        proj = self.proj(embeddings)  # [B, seq, 64]
        
        # 2. CLR transform abundances
        ab = abundances.squeeze(-1).clone()
        nonzero = (ab > 0) & (masks > 0)
        log_ab = torch.zeros_like(ab)
        log_ab[nonzero] = torch.log(ab[nonzero])
        count = nonzero.float().sum(1, keepdim=True).clamp(min=1)
        geo_mean = (log_ab * nonzero.float()).sum(1, keepdim=True) / count
        clr = ((log_ab - geo_mean) * nonzero.float()).unsqueeze(-1)  # [B, seq, 1]
        
        # 3. Attention using projected embeddings + CLR abundance
        attn_input = torch.cat([proj, clr], dim=-1)  # [B, seq, 65]
        scores = self.attn(attn_input).squeeze(-1)    # [B, seq]
        scores = scores.masked_fill(~nonzero, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        
        # 4. Pool projected embeddings
        pooled = (proj * weights).sum(dim=1)  # [B, 64]
        
        return self.regressor(pooled), None, None, None, None
       

def train_model(model, criterion, optimizer, loader, train_loader, val_loader, test_loader, device, num_epochs, out_dir, count_reg=1.0):

    best_val_loss = float('inf')
    best_mae = float('inf')
    best_model_path = None
    patience = 100
    counter = 0

    #count_reg = 0.0
    train_dataset = train_loader.dataset
    from transformers import get_linear_schedule_with_warmup
    
    total_steps = 200 * len(train_loader)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # warmup for first 10% of training
        num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        #train_dataset, train_split_table, train_targets = loader.get_dataset(split="train")
        train_dataset.sample_epoch_init(epoch)
        train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
        count_loss_value = 0.0
        count_loss_val = 0.0
        train_preds = []
        train_targets = []
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)  # Scale down to prevent instability
            abundances = batch['abundances'].to(device)/ 1e4
            masks = batch['masks'].to(device)
            targets = batch['label'].to(device)
            w_emb = batch['weighted_sum'].to(device)
            bins_abundance = batch['bins_abundance'].to(device)
            ##abundances = bins_abundance
            optimizer.zero_grad()


            outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
            if counts_pred is not None:
                count_loss = compute_count_loss(counts_pred, abundances).mean()
            else:
                count_loss = torch.tensor(0.0, device=device)

            loss = criterion(outputs, targets) + (count_loss * count_reg)
            count_loss_batch = count_loss.item() if count_loss is not None else 0.0

            loss.backward()
            #Check early vs late layers
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            train_targets.append(targets.cpu().detach())
            train_preds.append(outputs.cpu().detach())

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            total_loss += loss.item()
            count_loss_value += count_loss_batch

        avg_train_loss = total_loss / len(train_loader)
        count_loss_value = count_loss_value / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device) 
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)
                w_emb = batch['weighted_sum'].to(device)
                bins_abundance = batch['bins_abundance'].to(device)
                #abundances = bins_abundance


                outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
                #outputs, _, _, _, _ = model(w_emb)
                #outputs = model(abundances)['regression_output']
                if counts_pred is not None:
                    loss_count = compute_count_loss(counts_pred, abundances).mean()#  compute_count_loss(abundances, counts_pred, loss_type="mse").mean()
                else:
                    loss_count = torch.tensor(0.0, device=device)
                loss = criterion(outputs, targets)  + (loss_count * count_reg)
                val_loss += loss.item()
                count_loss_val += loss_count.item() if loss_count is not None else 0.0

                all_preds.append(outputs.cpu() * 10)  # Scale back predictions to original range
                all_targets.append(targets.cpu() * 10)

        avg_val_loss = val_loss / len(val_loader)
        count_loss_val = count_loss_val / len(val_loader)
        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()

        train_mae = torch.mean(torch.abs(torch.cat(train_preds).squeeze() - torch.cat(train_targets).squeeze())).item()


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - LR: {current_lr:.8f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train MAE: {train_mae:.4f} | Val MAE: {mae:.4f}")
        print(f"Count Loss - Train: {count_loss_value:.4f} | Val: {count_loss_val:.4f}")

        # Save best model
        if best_mae > mae:
            best_val_loss = avg_val_loss
            best_mae = mae
            best_model_path = os.path.join(out_dir, f"best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            counter = 0  # Reset counter when a better model is found   
            print(f"New best model found at epoch {epoch+1} with Val Loss: {best_val_loss:.4f} and Val MAE: {best_mae:.4f}. Saving model.")

            import matplotlib.pyplot as plt
            plt.scatter(all_targets, all_preds)
            plt.xlabel("True Age")
            plt.ylabel("Predicted Age")
            plt.title(f"Epoch {epoch+1} - True vs Predicted Age - Mae {mae:.4f}")
            plt.savefig(os.path.join(out_dir, f"true_vs_pred_epoch_val.png"))
            plt.close()
        else:
            counter += 1  # Increment counter if no improvement
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device) 
            abundances = batch['abundances'].to(device)/ 1e4
            masks = batch['masks'].to(device)
            targets = batch['label'].to(device)
            w_emb = batch['weighted_sum'].to(device)
            bins_abundance = batch['bins_abundance'].to(device)
            #abundances = bins_abundance

            outputs, _, _, _, _ = model(embeddings, abundances, masks)
            #outputs, _, _, _, _ = model(w_emb)
            all_preds.append(outputs.cpu() * 10)  # Scale back predictions to original range
            all_targets.append(targets.cpu() * 10)

        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        test_mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        print(f"Test MAE: {test_mae:.4f}")

    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae), "best_model": best_model_path}

    #

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MLPRegressor, self).__init__()
        
        self.attention_pooling = AttentionPooling(input_dim)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, embeddings, abundances, masks=None):
        x = self.attention_pooling(embeddings, abundances, masks)
        return self.regressor(x), None, None, None, None
    
def train_model_masked(model, optimizer, train_loader, val_loader, 
                       device, num_epochs, out_dir, 
                       recon_weight=0.1):
    
    regression_loss_fn = nn.MSELoss()
    reconstruction_loss_fn = nn.MSELoss(reduction='none')
    
    best_val_loss = float('inf')
    best_mae = float('inf')
    best_model_path = None
    patience = 15
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_reg_loss = 0
        total_recon_loss = 0
        
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device)
            masks = batch['masks'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            age_pred, reconstructed, mask_selected, original_emb, _ = model(
                embeddings, abundances, masks
            )
            
            # Regression loss
            reg_loss = regression_loss_fn(age_pred.squeeze(), targets)
            
            # Reconstruction loss (only at masked positions)
            if mask_selected.any():
                recon_loss_all = reconstruction_loss_fn(
                    reconstructed, original_emb
                )  # [B, seq, 768]
                # Average over embedding dim, then only take masked positions
                recon_loss_per_pos = recon_loss_all.mean(dim=-1)  # [B, seq]
                recon_loss = (recon_loss_per_pos * mask_selected.float()).sum() / mask_selected.float().sum().clamp(min=1)
            else:
                recon_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = reg_loss + recon_weight * recon_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_recon_loss += recon_loss.item()
        
        n = len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                abundances = batch['abundances'].to(device)
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)
                
                age_pred, _, _, _, _ = model(embeddings, abundances, masks)
                loss = regression_loss_fn(age_pred.squeeze(), targets)
                val_loss += loss.item()
                
                all_preds.append(age_pred.cpu())
                all_targets.append(targets.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train: {total_loss/n:.4f} (reg: {total_reg_loss/n:.4f}, "
              f"recon: {total_recon_loss/n:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {mae:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_mae = mae
            print(f"New best model found at epoch {epoch+1} with Val Loss: {best_val_loss:.4f} and Val MAE: {best_mae:.4f}. Saving model.")
            best_model_path = os.path.join(out_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae), 
            "best_model": best_model_path}

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, abundances, masks):

        scaled_embeddings = (embeddings * abundances) + embeddings   

        x = self.attention(scaled_embeddings)
        x = x.masked_fill(masks.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(x, dim=1)  # [B, Seq_Len, 1]
        pooled = torch.sum(attn_weights * embeddings, dim=1)            # [B, Features]
        return pooled
    
    
class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MLPRegressor, self).__init__()
        
        self.attention_pooling = AttentionPooling(input_dim)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, embeddings, abundances, masks=None):
        x = self.attention_pooling(embeddings, abundances, masks)
        return self.regressor(x), None, None, None, None
    
if __name__ == "__main__":

    datasets = {"gut_data" : "gut_4434" , "skin_data": "skin_1975", "oral_data": "oral_2550"}
    dataset_name = "skin_data"
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/Input/{}/" 
  
    output_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/my_outputs/"
    biom_table_path = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + ".biom")
    metadata_file = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + "_map.txt")
    embedding_path = os.path.join(output_dir, datasets[dataset_name] + "token_finetuned_embeddings.h5")

    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)
    print("Metadata shape: ", metadata_df.columns)
    target_column = "qiita_host_age"

    # loader = AgeDatasetLoader(biom_table_path, metadata_df, target_column, embedding_path)
    # train_dataset, train_split_table, train_targets = loader.get_dataset(split="train")
    # val_dataset, val_split_table, val_targets = loader.get_dataset(split="val")
    # test_dataset, test_split_table, test_targets = loader.get_dataset(split="test")
    # print("Train dataset size: ", len(train_dataset))
    # print("Test dataset size: ", len(test_dataset))

    #print("median target ", np.median(train_targets))
    multitask = False
    model_config = {
                "input_dim": 128,
                "hidden_dim": 1024,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "pe": False,

            }

    # train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)



#     model = MicrobiomeTransformer(
#     embed_dim=128,
#     num_layers=4,
#     num_heads=4,
#     dropout=0.3,
#     fusion_type="gating",
#     pooling="attention",
#     num_outputs=1,
#     task="regression",
#     num_datasets=2
# )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

    #optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    print("=========================================================")

    #model = BasicRegressorGRL(**model_config)
    #model = BottleneckRegressor(input_dim=model_config["input_dim"], hidden_dim=model_config["hidden_dim"], dropout=model_config["dropout"])
    #model = MLPRegressor(input_dim=768, hidden_dim=1024, dropout=model_config["dropout"])
    #model = CLR(emb_dim=768, proj_dim=64, hidden_dim=512, dropout=model_config["dropout"])

    #model = ClusteredRegressor()
    #for n_bins in [50]: #, 100, 150]:
    n_bins = 50
    for count_reg in [0.1, 1, 10, 100]: #, 0.5, 1.0]:    
        loader = AgeDatasetLoader(biom_table_path, metadata_df, target_column, embedding_path, n_bins=n_bins)

        # train_dataset, train_split_table, train_targets = loader.get_dataset(split="train")
        # val_dataset, val_split_table, val_targets = loader.get_dataset(split="val")
        # test_dataset, test_split_table, test_targets = loader.get_dataset(split="test")

        loader.get_dataset_by_json(json_path=f"/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/group_assignments_{dataset_name}.json")
        train_dataset = loader.train_dataset
        val_dataset = loader.val_dataset
        test_dataset = loader.test_dataset

        train_ids = set()
        for item in train_dataset:
            train_ids.add(item['SampleID'])
        val_ids = set()
        for item in val_dataset:
            val_ids.add(item['SampleID'])
        test_ids = set()
        for item in test_dataset:
            test_ids.add(item['SampleID'])
        print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}, Test samples: {len(test_ids)}")
        print(f"Train-Val overlap: {len(train_ids.intersection(val_ids))}, Train-Test overlap: {len(train_ids.intersection(test_ids))}, Val-Test overlap: {len(val_ids.intersection(test_ids))}")

        #val_dataset, val_split_table, val_targets = loader.get_dataset(split="val")
        #test_dataset, test_split_table, test_targets = loader.get_dataset(split="test")
        # print("Train dataset size: ", len(train_dataset))
        # print("Test dataset size: ", len(test_dataset))

        # print("median target ", np.median(train_targets))
        multitask = False
        model_config = {
                    "input_dim": 128,
                    "hidden_dim": 1024,
                    "num_heads": 4,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "pe": False,

                }
        

        train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)



        for seed in range(3):  # try a few seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            
            # model = BasicRegressorNew(
            #     input_dim=128, hidden_dim=1024, 
            #     num_heads=4, num_layers=2, 
            #     dropout=0.3, pe=False, GRL=False, bins=True
            # )
            model = BasicRegressor(
                input_dim=128, hidden_dim=256, 
                num_heads=4, num_layers=2, 
                dropout=0.05, pe=False, grl=False
            )  ###############works well with count_reg=0.1, 1, 10, 100, but not with 0.5

            model = MLPRegressor(input_dim=768, hidden_dim=1024, dropout=0.3)
            # Update bin embedding size
            #model.bin_embeddings = nn.Embedding(n_bins + 1, 128, padding_idx=0)
            #model.reset_parameters()
            model = model.to(device)
            
            optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
            criterion = torch.nn.MSELoss()
            
            out_dir = f"results/sweep_improved_split_finetuned/{dataset_name}_{count_reg}_2seed_{seed}"
            os.makedirs(out_dir, exist_ok=True)
            
            res = train_model(model, criterion, optimizer, loader, 
                            train_loader, val_loader, test_loader, 
                            device, num_epochs=1000, out_dir=out_dir, count_reg=count_reg)
            print(f"count_reg={count_reg}, seed={seed}: MAE={res['best_mae']:.4f}")
            with open(os.path.join(out_dir, "results.txt"), "w") as f:
                f.write(f"Best Val Loss: {res['best_loss']:.4f}\n")
                f.write(f"Best Val MAE: {res['best_mae']:.4f}\n")
                f.write(f"Best Model Path: {res['best_model']}\n")
                




