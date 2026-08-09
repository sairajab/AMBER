"""
Optuna hyperparameter optimization for microbiome age prediction.

Searches over: learning rate, weight decay, hidden dim, num layers, num heads,
dropout, count_reg, loss function, batch size, and optimizer betas.

Usage:
    python optuna_train.py                          # Run all datasets
    python optuna_train.py --dataset gut_data        # Run one dataset
    python optuna_train.py --n_trials 200            # More trials
"""

from microbiome_model.data.dataset import finetuned_dna_bert
from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.optimized_embedding_loader import MemoryMappedEmbeddingLoader as EmbeddingLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import biom
import os
import json
import numpy as np
import random
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from microbiome_model.models.orig import BasicRegressor


# =============================================================================
# Data loading (unchanged from original)
# =============================================================================

class AgeDatasetLoader:

    def __init__(self, biom_table_path, metadata_df, target_column, embedding_path=None, n_bins=50):
        self.table = biom.load_table(biom_table_path)
        self.metadata_df = metadata_df
        self.target_column = target_column
        self.sample_ids = self.table.ids(axis='sample')
        self.sample_ids = self.metadata_df.index.intersection(self.sample_ids).tolist()
        self.subsampled = self.table.subsample(5000, axis="observation")
        print("Original table shape:", self.table.shape)
        self.sequences = self.table.ids(axis='observation')
        self.n_bins = n_bins
        if os.path.exists(embedding_path):
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        else:
            self.create_embeddings(output_path=embedding_path)
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        self.json = None

    def create_embeddings(self, output_path="embeddings.h5"):
        seq_dict = {seq: seq for seq in self.sequences}
        print("Creating embeddings for", len(seq_dict), "sequences")
        finetuned_dna_bert(output_path, seq_dict,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Embeddings created and saved to", output_path)

    def get_dataset_by_json(self, json_path):
        with open(json_path, "r") as f:
            group_assignments = json.load(f)
        

        self.json = json_path
        train_indices_full = group_assignments["train_indices"]
        test_indices = group_assignments["test_indices"]

        
        val_indices = []

        if "host_subject_id" in self.metadata_df.columns:
            print("Using host_subject_id for grouping")
            train_host_ids = self.metadata_df.loc[train_indices_full]['host_subject_id']

            ## group by host_subject_id and then split by host_subject_id to ensure no overlap between train and val
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            _ , val_idx = next(gss.split(train_indices_full, groups=train_host_ids))
            for i in val_idx:
                val_indices.append(train_indices_full[i])
        else:
            _, val_idx = train_test_split(train_indices_full, test_size=0.1, random_state=42)
            val_indices = val_idx


        train_indices = [idx for idx in train_indices_full if idx not in val_indices]

        train_table = self.table.filter(train_indices, axis='sample', inplace=False)
        val_table = self.table.filter(val_indices, axis='sample', inplace=False)
        test_table = self.table.filter(test_indices, axis='sample', inplace=False)

        print("Train table shape:", train_table.shape)
        self.train_dataset = MicrobiomeSparseDataset(
            biom_table=train_table,
            sample_targets=self.metadata_df.loc[train_indices][self.target_column].to_dict(),
            sort_asvs=True, embedding_loader=self.embedding_loader,
            kmer_seqs=None, one_hot_seqs=None, seed=0,
            n_bins=self.n_bins, subsample=True,
        )

        print("Val table shape:", val_table.shape)
        self.val_dataset = MicrobiomeSparseDataset(
            biom_table=val_table,
            sample_targets=self.metadata_df.loc[val_indices][self.target_column].to_dict(),
            sort_asvs=True, embedding_loader=self.embedding_loader,
            kmer_seqs=None, one_hot_seqs=None, seed=0,
            n_bins=self.n_bins, subsample=True,
        )


        print("Test table shape:", test_table.shape)
        self.test_dataset = MicrobiomeSparseDataset(
            biom_table=test_table,
            sample_targets=self.metadata_df.loc[test_indices][self.target_column].to_dict(),
            sort_asvs=True, embedding_loader=self.embedding_loader,
            kmer_seqs=None, one_hot_seqs=None, seed=None,
            n_bins=self.n_bins, subsample=True,
        )

        self.train_targets = self.metadata_df.loc[train_indices][self.target_column].to_list()
        self.test_targets = self.metadata_df.loc[test_indices][self.target_column].to_list()

    def get_test_dataset(self, seed=0):
        if not self.json:
            raise ValueError("JSON path not set. Run get_dataset_by_json first.")
        with open(self.json, "r") as f:
            group_assignments = json.load(f)
        test_indices = group_assignments["test_indices"]
        test_table = self.table.filter(test_indices, axis='sample', inplace=False)
        self.test_dataset = MicrobiomeSparseDataset(
            biom_table=test_table,
            sample_targets=self.metadata_df.loc[test_indices][self.target_column].to_dict(),
            sort_asvs=True, embedding_loader=self.embedding_loader,
            kmer_seqs=None, one_hot_seqs=None, seed=seed,
            n_bins=self.n_bins, subsample=True,
        )


# =============================================================================
# Collate (unchanged)
# =============================================================================

def collate_fn(batch):
    from microbiome_model.losses.core import compute_count_loss  # keep import local

    sample_ids = [item['SampleID'] for item in batch]
    targets = torch.stack([item['outdoor_add_0'] / 10 for item in batch])
    seqs_ids = [item['seqs_ids'] for item in batch]
    embeddings = [item['embeddings'] for item in batch]
    abundances = [item['abundances'] for item in batch]
    bins_abundance = [item['binned_abundances'] for item in batch]

    max_len = max(e.shape[0] for e in embeddings)

    if len(embeddings[0].shape) == 2:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1])
    else:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2])

    padded_abundances = torch.zeros(len(batch), max_len, 1)
    padded_seqs_ids = [None] * len(batch)
    batch_masks = torch.zeros(len(batch), max_len)
    padded_bins_abundance = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb
        padded_abundances[i, :length, :] = abun
        padded_bins_abundance[i, :length] = bins_abundance[i]
        batch_masks[i, :length] = 1

    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))

    actual_lengths = batch_masks.sum(dim=1, keepdim=True)
    weighted_sum = (padded_embeddings * padded_abundances).sum(dim=1) / actual_lengths

    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings,
        'abundances': padded_abundances,
        'masks': batch_masks,
        'label': targets,
        'seqs_ids': padded_seqs_ids,
        'weighted_sum': weighted_sum,
        'bins_abundance': padded_bins_abundance,
    }


# =============================================================================
# Training loop used inside each Optuna trial
# =============================================================================

def train_one_trial(model, criterion, optimizer, loader, device,
                    num_epochs, count_reg, patience, trial,
                    warmup_epochs=20, batch_size=64, val_split=0.1):
    """Train with early stopping & Optuna pruning. Returns best val MAE."""

    from microbiome_model.losses.core import compute_count_loss

    # --- schedulers ---
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # --- train / val split ---
    train_dataset = loader.train_dataset
    val_dataset = loader.val_dataset
    # torch.manual_seed(42)
    # val_size = int(len(train_dataset_full) * val_split)
    # train_size = len(train_dataset_full) - val_size
    # train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    best_mae = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        # ---------- train ----------
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4
            masks = batch['masks'].to(device)
            targets = batch['label'].to(device)

            optimizer.zero_grad()
            outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
            count_loss = compute_count_loss(abundances, counts_pred, masks).mean()
            loss = criterion(outputs, targets) + count_loss * count_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ---------- validate ----------
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)

                outputs, _, _, _, _ = model(embeddings, abundances, masks)
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item() * 10  # scale back

        # report to Optuna for pruning
        trial.report(mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if mae < best_mae:
            best_mae = mae
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_mae


# =============================================================================
# Optuna objective
# =============================================================================

def make_objective(loader, device, num_epochs, patience):
    """Returns a closure that Optuna calls for each trial."""

    from microbiome_model.losses.core import compute_count_loss  # noqa

    def objective(trial: optuna.Trial) -> float:
        # ----- hyper-parameters to search -----
        input_dim = trial.suggest_categorical("input_dim", [256, 512, 128])  # fixed for now
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        count_reg = trial.suggest_float("count_reg", 0.0, 1.0, step=0.05)
        pe =  False #trial.suggest_categorical("pe", [True, False])
        batch_size = 32#trial.suggest_categorical("batch_size", [32])
        loss_name = trial.suggest_categorical("loss_fn", ["mse", "smooth_l1", "huber"])
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 40, step=5)

        # ----- loss -----
        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "smooth_l1":
            beta = trial.suggest_float("smooth_l1_beta", 0.5, 5.0, step=0.5)
            criterion = nn.SmoothL1Loss(beta=beta)
        else:  # huber
            delta = trial.suggest_float("huber_delta", 0.5, 5.0, step=0.5)
            criterion = nn.HuberLoss(delta=delta)

        # ----- seed for reproducibility within trial -----
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ----- model -----
        model = BasicRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pe=pe,
            grl=False,
        )
        model.reset_parameters()
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_mae = train_one_trial(
            model, criterion, optimizer, loader, device,
            num_epochs=num_epochs, count_reg=count_reg,
            patience=patience, trial=trial,
            warmup_epochs=warmup_epochs, batch_size=batch_size,
        )

        return best_mae  # Optuna minimises this

    return objective


# =============================================================================
# Full retrain with best params, then evaluate on test
# =============================================================================

def retrain_best(params, loader, device, out_dir, num_epochs=300, patience=30):
    """Retrain 3 seeds with the best hyperparameters and evaluate."""

    from microbiome_model.losses.core import compute_count_loss

    seeds = [0, 42, 123]

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # loss
        loss_name = params["loss_fn"]
        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "smooth_l1":
            criterion = nn.SmoothL1Loss(beta=params.get("smooth_l1_beta", 1.0))
        else:
            criterion = nn.HuberLoss(delta=params.get("huber_delta", 1.0))

        model = BasicRegressor(
            input_dim=params.get("input_dim", 256),
            hidden_dim=params["hidden_dim"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            pe=params["pe"],
            grl=False,
        )
        model.reset_parameters()
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=params["lr"],
                                weight_decay=params["weight_decay"])

        count_reg = params["count_reg"]
        batch_size = params["batch_size"]
        warmup_epochs = params["warmup_epochs"]

        # schedulers
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        # data
        train_dataset_full = loader.train_dataset
        torch.manual_seed(42)
        val_size = int(len(train_dataset_full) * 0.1)
        train_size = len(train_dataset_full) - val_size
        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        test_loader = DataLoader(loader.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        best_mae = float('inf')
        counter = 0
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        best_model_path = os.path.join(seed_dir, "best_model.pt")

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                embeddings = batch['embeddings'].to(device)
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)
                optimizer.zero_grad()
                outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks)
                count_loss = compute_count_loss(abundances, counts_pred, masks).mean()
                loss = criterion(outputs, targets) + count_loss * count_reg
                loss.backward()
                optimizer.step()
            scheduler.step()

            # val
            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch['embeddings'].to(device)
                    abundances = batch['abundances'].to(device) / 1e4
                    masks = batch['masks'].to(device)
                    targets = batch['label'].to(device)
                    outputs, _, _, _, _ = model(embeddings, abundances, masks)
                    preds.append(outputs.cpu())
                    tgts.append(targets.cpu())
            preds = torch.cat(preds).squeeze()
            tgts = torch.cat(tgts).squeeze()
            mae = torch.mean(torch.abs(preds - tgts)).item() * 10

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), best_model_path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  [seed={seed}] Early stop at epoch {epoch+1}")
                    break

        # --- test evaluation (10 sub-samples) ---
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_maes = []
        for k in range(10):
            loader.get_test_dataset(seed=k)
            test_loader = DataLoader(loader.test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)
            preds, tgts = [], []
            with torch.no_grad():
                for batch in test_loader:
                    embeddings = batch['embeddings'].to(device)
                    abundances = batch['abundances'].to(device) / 1e4
                    masks = batch['masks'].to(device)
                    targets = batch['label'].to(device)
                    outputs, _, _, _, _ = model(embeddings, abundances, masks)
                    preds.append(outputs.cpu() * 10)
                    tgts.append(targets.cpu() * 10)
            preds = torch.cat(preds).squeeze()
            tgts = torch.cat(tgts).squeeze()
            test_maes.append(torch.mean(torch.abs(preds - tgts)).item())

            # plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(tgts.numpy(), preds.numpy(), alpha=0.4, s=10)
            ax.plot([tgts.min(), tgts.max()], [tgts.min(), tgts.max()], 'r--')
            r2 = r2_score(tgts.numpy(), preds.numpy())
            ax.set_title(f"seed={seed} eval={k}  MAE={test_maes[-1]:.2f}  R²={r2:.3f}")
            ax.set_xlabel("True Age"); ax.set_ylabel("Predicted Age")
            fig.savefig(os.path.join(seed_dir, f"test_eval_{k}.png"), dpi=100)
            plt.close(fig)

        mean_test = np.mean(test_maes)
        std_test = np.std(test_maes)
        print(f"  [seed={seed}] Val MAE={best_mae:.4f} | Test MAE={mean_test:.4f} ± {std_test:.4f}")

        with open(os.path.join(seed_dir, "results.txt"), "w") as f:
            f.write(f"Best Val MAE: {best_mae:.4f}\n")
            f.write(f"Test MAE: {mean_test:.4f} ± {std_test:.4f}\n")
            f.write(f"Params: {json.dumps(params, indent=2)}\n")

class TestModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings, abundances, masks):
        batch_size, max_len, emb_dim = embeddings.shape

        weighted_sum = (embeddings * abundances).sum(dim=1) / (masks.sum(dim=1, keepdim=True) + 1e-8)
        output = self.model(weighted_sum)
        return output, None, None, None, None
    

        
    

# =============================================================================
# Main
# =============================================================================

def run_for_dataset(dataset_name, n_trials=100, num_epochs_search=150,
                    num_epochs_retrain=300, patience_search=15, patience_retrain=30):
    """Run Optuna search + retrain for one dataset."""

    base_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project//"
    datasets_map = {"gut_data": "gut_4434", "skin_data": "skin_1975", "oral_data": "oral_2550"}
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/Input/{}/"
    embedding_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/my_outputs/"
    target_column = "qiita_host_age"
    if dataset_name == "gut_data":
        target_column = "age"
    biom_table_path = os.path.join(data_dir.format(dataset_name), datasets_map[dataset_name] + ".biom")
    metadata_file = os.path.join(data_dir.format(dataset_name), datasets_map[dataset_name] + "_map.txt")
    embedding_path = os.path.join(embedding_dir, datasets_map[dataset_name] + "_new_embeddings.h5")
    json_path = os.path.join(base_dir, f"TRPCA/2_trails_results/group_assignments_{dataset_name}.json")

    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}  ({datasets_map[dataset_name]})")
    print(f"{'='*70}")
    print(metadata_df[target_column].describe())

    loader = AgeDatasetLoader(biom_table_path, metadata_df, target_column, embedding_path)
    loader.get_dataset_by_json(json_path=json_path)
    print(f"Train size: {len(loader.train_dataset)}, Test size: {len(loader.test_dataset)}")

    output_folder = os.path.join(base_dir, "microbiome_model", "results", "optuna", dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # --- Optuna study ---
    study_name = f"age_pred_{dataset_name}"
    storage_path = os.path.join(output_folder, f"optuna_{dataset_name}.db")
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    objective = make_objective(loader, device, num_epochs=num_epochs_search, patience=patience_search)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # --- results ---
    best = study.best_trial
    print(f"\nBest trial #{best.number}: Val MAE = {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # save study summary
    results_dir = os.path.join(output_folder, f"optuna_results/{dataset_name}")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "best_params.json"), "w") as f:
        json.dump(best.params, f, indent=2)

    # save top-10 trials
    trials_df = study.trials_dataframe().sort_values("value")
    trials_df.head(20).to_csv(os.path.join(results_dir, "top20_trials.csv"), index=False)

    # importance plot
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        fig = plot_param_importances(study)
        fig.figure.savefig(os.path.join(results_dir, "param_importances.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"Skipping importance plot: {e}")

    # --- retrain with best params ---
    print(f"\nRetraining with best params across 3 seeds...")
    retrain_dir = os.path.join(results_dir, "retrain")
    retrain_best(best.params, loader, device, retrain_dir,
                 num_epochs=num_epochs_retrain, patience=patience_retrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="One of: gut_data, skin_data, oral_data. If omitted, runs all.")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials per dataset")
    parser.add_argument("--search_epochs", type=int, default=150,
                        help="Max epochs during search (shorter = faster search)")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="Max epochs for final retrain")
    args = parser.parse_args()

    all_datasets = ["skin_data"]
    targets = [args.dataset] if args.dataset else all_datasets

    for ds in targets:
        run_for_dataset(ds, n_trials=args.n_trials,
                        num_epochs_search=args.search_epochs,
                        num_epochs_retrain=args.retrain_epochs)