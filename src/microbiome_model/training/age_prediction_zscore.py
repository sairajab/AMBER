"""
Age prediction transformer with z-score normalized target.

Target normalization:
  - Fit mean/std on TRAIN targets only (never touch val/test stats).
  - collate_fn normalizes the target to z-space.
  - Model learns in z-space.
  - For MAE/plots, invert: y_hat_years = y_hat_z * std + mean.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import biom
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from microbiome_model.data.dataset import finetuned_dna_bert
from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.optimized_embedding_loader import (
    MemoryMappedEmbeddingLoader as EmbeddingLoader,
)
from microbiome_model.models.orig import BasicRegressor
from microbiome_model.losses.core import compute_count_loss


# ---------------------------------------------------------------------------
# Global target scaler (set once from TRAIN targets before training)
# ---------------------------------------------------------------------------
class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        arr = np.asarray(list(values), dtype=np.float64)
        self.mean = float(arr.mean())
        self.std = float(arr.std() + 1e-8)
        print(f"[TargetScaler] fit on {len(arr)} values: mean={self.mean:.3f}, std={self.std:.3f}")

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, y_z):
        return y_z * self.std + self.mean

    def is_fit(self):
        return self.mean is not None


TARGET_SCALER = TargetScaler()  # shared, set in main() from train targets


# ---------------------------------------------------------------------------
# Collate (normalizes target using TARGET_SCALER)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    sample_ids = [item["SampleID"] for item in batch]

    # raw ages in years -> z-space
    raw_targets = torch.stack([item["outdoor_add_0"] for item in batch]).float()  # shape (B, 1)
    if not TARGET_SCALER.is_fit():
        raise RuntimeError("TARGET_SCALER must be fit on train targets before training")
    targets_z = (raw_targets - TARGET_SCALER.mean) / TARGET_SCALER.std

    seqs_ids = [item["seqs_ids"] for item in batch]
    embeddings = [item["embeddings"] for item in batch]
    abundances = [item["abundances"] for item in batch]
    bins_abundance = [item["binned_abundances"] for item in batch]

    max_len = max(e.shape[0] for e in embeddings)
    if len(embeddings[0].shape) == 2:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1])
    else:
        padded_embeddings = torch.zeros(
            len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2]
        )
    padded_abundances = torch.zeros(len(batch), max_len, 1)
    padded_seqs_ids = [None] * len(batch)
    batch_masks = torch.zeros(len(batch), max_len)
    padded_bins = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb
        padded_abundances[i, :length, :] = abun
        padded_bins[i, :length] = bins_abundance[i]
        batch_masks[i, :length] = 1
    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))

    actual_lengths = batch_masks.sum(dim=1, keepdim=True)
    weighted_sum = (padded_embeddings * padded_abundances).sum(dim=1) / actual_lengths

    return {
        "SampleID": sample_ids,
        "embeddings": padded_embeddings,
        "abundances": padded_abundances,
        "masks": batch_masks,
        "label": targets_z,          # z-scored target
        "label_raw": raw_targets,    # keep original age (years) for eval
        "seqs_ids": padded_seqs_ids,
        "weighted_sum": weighted_sum,
        "bins_abundance": padded_bins,
    }


# ---------------------------------------------------------------------------
# Dataset loader (unchanged split logic; no target scaling happens here)
# ---------------------------------------------------------------------------
class AgeDatasetLoader:
    def __init__(self, biom_table_path, metadata_df, target_column, embedding_path=None, n_bins=50):
        self.table = biom.load_table(biom_table_path)
        self.metadata_df = metadata_df
        self.target_column = target_column
        self.n_bins = n_bins

        if os.path.exists(embedding_path):
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        else:
            seq_dict = {s: s for s in self.table.ids(axis="observation")}
            finetuned_dna_bert(embedding_path, seq_dict,
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)

    def get_dataset_by_json(self, split, json_path):
        with open(json_path, "r") as f:
            groups = json.load(f)
        train_ids_all = list(groups["train_indices"])
        test_ids = list(groups["test_indices"])
        cut = int(len(train_ids_all) * 0.9)
        val_ids = train_ids_all[cut:]
        train_ids = train_ids_all[:cut]

        if split == "train":
            sample_ids = train_ids
        elif split == "val":
            sample_ids = val_ids
        else:
            sample_ids = test_ids

        biom_ids = set(self.table.ids(axis="sample"))
        sample_ids = [s for s in sample_ids if s in biom_ids]
        print(f"  {split}: {len(sample_ids)} samples")

        split_table = self.table.filter(sample_ids, axis="sample", inplace=False)
        seed = 9999 if split == "val" else None
        targets = self.metadata_df.loc[sample_ids][self.target_column].to_dict()

        dataset = MicrobiomeSparseDataset(
            biom_table=split_table,
            sample_targets=targets,
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            test_split=(split == "test"),
            seed=seed,
            n_bins=self.n_bins,
            subsample=(split == "train"),
        )
        return dataset, split_table, self.metadata_df.loc[sample_ids][self.target_column].to_list()


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
def z_to_years(z_tensor):
    """Tensor in z-space -> years (numpy)."""
    return (z_tensor.detach().cpu().numpy() * TARGET_SCALER.std) + TARGET_SCALER.mean


def train_model(model, criterion, optimizer, loader, train_loader, val_loader, test_loader,
                device, num_epochs, out_dir, count_reg=1.0):
    best_val_loss = float("inf")
    best_mae = float("inf")
    best_model_path = None
    patience, counter = 10, 0

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    train_dataset, _, _ = loader.get_dataset_by_json(split="train", json_path=JSON_PATH)

    for epoch in range(num_epochs):
        model.train()
        train_dataset.sample_epoch_init(epoch, seed=True)
        train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)

        total_loss, count_loss_value = 0.0, 0.0
        for batch in train_loader:
            embeddings = batch["embeddings"].to(device)
            abundances = batch["abundances"].to(device) / 1e4
            masks = batch["masks"].to(device)
            targets_z = batch["label"].to(device)  # z-space

            optimizer.zero_grad()
            outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
            count_loss = compute_count_loss(abundances, counts_pred, masks).mean()
            loss = criterion(outputs, targets_z) + count_loss * count_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count_loss_value += count_loss.item()
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        count_loss_value /= len(train_loader)

        # Validation — compute MAE in YEARS
        model.eval()
        val_loss, count_loss_val = 0.0, 0.0
        preds_years_all, targets_years_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embeddings"].to(device)
                abundances = batch["abundances"].to(device) / 1e4
                masks = batch["masks"].to(device)
                targets_z = batch["label"].to(device)
                targets_raw = batch["label_raw"].to(device)

                outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
                loss_count = compute_count_loss(abundances, counts_pred, masks).mean()
                loss = criterion(outputs, targets_z) + loss_count * count_reg
                val_loss += loss.item()
                count_loss_val += loss_count.item()

                preds_years_all.append(z_to_years(outputs))
                targets_years_all.append(targets_raw.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        count_loss_val /= len(val_loader)
        preds_years = np.concatenate(preds_years_all).squeeze()
        targets_years = np.concatenate(targets_years_all).squeeze()
        mae = float(np.mean(np.abs(preds_years - targets_years)))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} "
              f"- Val Loss: {avg_val_loss:.4f} - Val MAE: {mae:.4f} yrs")
        print(f"Count Loss - Train: {count_loss_value:.7f} | Val: {count_loss_val:.7f}")

        if mae < best_mae:
            best_val_loss = avg_val_loss
            best_mae = mae
            best_model_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            counter = 0

            plt.scatter(targets_years, preds_years, alpha=0.5)
            plt.plot([targets_years.min(), targets_years.max()],
                     [targets_years.min(), targets_years.max()], "r--")
            plt.xlabel("True Age"); plt.ylabel("Predicted Age")
            plt.title(f"Epoch {epoch+1} - Val (MAE {mae:.2f})")
            plt.savefig(os.path.join(out_dir, "true_vs_pred_epoch_val.png"))
            plt.close()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae),
            "best_model": best_model_path}


def evaluate_model(model, test_loader, device, output_dir):
    model.eval()
    preds_years_all, targets_years_all, ids_all = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embeddings"].to(device)
            abundances = batch["abundances"].to(device) / 1e4
            masks = batch["masks"].to(device)
            targets_raw = batch["label_raw"].to(device)

            outputs, _, _, _, _ = model(embeddings, abundances, masks)
            preds_years_all.append(z_to_years(outputs))
            targets_years_all.append(targets_raw.detach().cpu().numpy())
            ids_all.extend(batch["SampleID"])

    preds = np.concatenate(preds_years_all).squeeze()
    targets = np.concatenate(targets_years_all).squeeze()
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.5)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", label="Perfect")
    ax.set_xlabel("True age"); ax.set_ylabel("Predicted age")
    ax.set_title(f"Test | MAE {mae:.3f} | R² {r2:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "true_vs_pred_final.png"))
    plt.close(fig)

    pd.DataFrame({"sample_id": ids_all, "y_true": targets, "y_pred": preds})\
      .to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    print(f"Test MAE: {mae:.4f} | R²: {r2:.4f}")
    return mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = {"gut_data": "gut_4434", "skin_data": "skin_1975", "oral_data": "oral_2550"}
    dataset_name = "gut_data"
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/Input/{}/"
    output_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/my_outputs/"

    best_params = {
        "gut_data":  {"count_reg": 0.1, "seed": 1},
        "skin_data": {"count_reg": 1.0, "seed": 0},
        "oral_data": {"count_reg": 0.1, "seed": 0},
    }

    biom_path = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + ".biom")
    meta_path = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + "_map.txt")
    embedding_path = os.path.join(output_dir, datasets[dataset_name] + "_embeddings.h5")
    JSON_PATH = f"/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/group_assignments_{dataset_name}.json"

    metadata_df = pd.read_csv(meta_path, sep="\t", index_col=0)
    target_column = "age"

    loader = AgeDatasetLoader(biom_path, metadata_df, target_column, embedding_path, n_bins=10)
    train_dataset, _, train_targets = loader.get_dataset_by_json(split="train", json_path=JSON_PATH)
    val_dataset,   _, val_targets   = loader.get_dataset_by_json(split="val",   json_path=JSON_PATH)
    test_dataset,  _, test_targets  = loader.get_dataset_by_json(split="test",  json_path=JSON_PATH)

    # Fit target scaler on TRAIN only
    TARGET_SCALER.fit(train_targets)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, collate_fn=collate_fn, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [best_params[dataset_name]["seed"], 42, 123]
    count_reg = best_params[dataset_name]["count_reg"]

    for seed in seeds:
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)

        model = BasicRegressor(
            input_dim=256, hidden_dim=768,
            num_heads=8, num_layers=4,
            dropout=0.2, pe=True, grl=False,
        )
        model.reset_parameters()
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()  # in z-space; loss magnitude is comparable across datasets

        out_dir = f"results/age_prediction/{dataset_name}/zscore/seed_{seed}"
        os.makedirs(out_dir, exist_ok=True)

        res = train_model(model, criterion, optimizer, loader,
                          train_loader, val_loader, test_loader,
                          device, num_epochs=300, out_dir=out_dir, count_reg=count_reg)

        print(f"seed={seed}: best val MAE={res['best_mae']:.4f} yrs")
        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(f"Best Val Loss (z-space): {res['best_loss']:.4f}\n")
            f.write(f"Best Val MAE (years):    {res['best_mae']:.4f}\n")
            f.write(f"Target mean/std:         {TARGET_SCALER.mean:.3f} / {TARGET_SCALER.std:.3f}\n")
            f.write(f"Best Model Path:         {res['best_model']}\n")

        model.load_state_dict(torch.load(res["best_model"]))
        print("Evaluating best model on test set...")
        evaluate_model(model, test_loader, device, out_dir)