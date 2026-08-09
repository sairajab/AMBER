from microbiome_model.data.dataset import finetuned_dna_bert, dna_bert
from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.optimized_embedding_loader import MemoryMappedEmbeddingLoader as EmbeddingLoader 
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import biom
import os 
import numpy as np
from microbiome_model.models.orig import BasicRegressor, AMBERRegressor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
import torch.nn as nn
from microbiome_model.losses.core import compute_count_loss, PairwiseLoss, DensityWeightedMSELoss
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import random
import json
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import wandb


COUNTRY_ALIASES = {
    "usa": "united states of america",
    "united states": "united states of america",
}

def build_categorical_map(series, aliases=None):
    """Map a metadata column to integer category indices.

    Index 0 is reserved for missing/unknown values, matching
    BasicRegressor's `card + 1` embedding convention (pad/unknown at 0).

    Returns (sample_id -> index dict, category_name -> index dict).
    """
    aliases = aliases or {}

    def normalize(v):
        if pd.isna(v):
            return None
        v = str(v).strip().lower()
        if v in ("", "not applicable", "not provided", "unknown", "na", "n/a"):
            return None
        v = v.removeprefix("gaz:").strip()
        return aliases.get(v, v)

    normalized = series.map(normalize)
    categories = sorted(normalized.dropna().unique())
    cat_to_idx = {cat: i + 1 for i, cat in enumerate(categories)}
    sample_to_idx = {sid: cat_to_idx.get(val, 0) for sid, val in normalized.items()}
    return sample_to_idx, cat_to_idx


class AgeDatasetLoader:

    def __init__(self, biom_table_path, metadata_df, target_column, embedding_path=None, n_bins=50,
                 categorical_columns=None):
        """
        Args:
            categorical_columns (dict[str, str]): Optional mapping of
                metadata field name -> metadata_df column name, e.g.
                {"sex": "sex", "country": "country"}. Each gets its own
                embedding in BasicRegressor via `metadata_cardinalities`.
        """
        self.table = biom.load_table(biom_table_path)
        self.metadata_df = metadata_df
        self.target_column = target_column
        self.sample_ids = self.table.ids(axis='sample')
        self.sample_ids = self.metadata_df.index.intersection(self.sample_ids).tolist()
        n_samples = len(self.sample_ids)
        # min_prev = 0.05 * n_samples # for gut 

        # self.table = self.table.filter(
        #     lambda values, id_, md: (values > 0).sum() >= min_prev,
        #     axis='observation',
        #     inplace=False
        # )

        self.subsampled =self.table.subsample(5000, axis = "observation")
        print("Original table shape: ", self.table.shape)
        self.sequences = self.table.ids(axis='observation')
        self.n_bins = n_bins
        if os.path.exists(embedding_path):
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)
        else:
            self.create_embeddings(output_path=embedding_path)
            self.embedding_loader = EmbeddingLoader(embedding_path, preload_all=True)

        self.json = None

        # Build categorical metadata maps (e.g. gender/country) once, over
        # the full metadata_df, so every split's dataset can share them.
        self.categorical_columns = categorical_columns or {}
        self.categorical_metadata = {}
        self.metadata_cardinalities = []
        for name, column in self.categorical_columns.items():
            print(f"Building categorical map for metadata '{name}' (column '{column}')...")
            aliases = COUNTRY_ALIASES if name == "country" else None
            sample_to_idx, cat_to_idx = build_categorical_map(self.metadata_df[column], aliases=aliases)
            print(f"Metadata '{name}' (column '{column}'): {len(cat_to_idx)} categories -> {list(cat_to_idx)[:10]}"
                  + (" ..." if len(cat_to_idx) > 10 else ""))
            self.categorical_metadata[name] = sample_to_idx
            self.metadata_cardinalities.append(len(cat_to_idx))
    
    def create_embeddings(self, output_path="embeddings.h5"):
        #map seq to seq in dict
        seq_dict = {seq: seq for seq in self.sequences}
        print("Creating embeddings for ", len(seq_dict), " sequences")
        dna_bert(output_path, seq_dict, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Embeddings created and saved to ", output_path)
    
    def create_splits(self, test_split=0.2, random_seed=42):
        import random
        random.seed(random_seed)
        shuffled = list(self.sample_ids)
        random.shuffle(shuffled)
        self.sample_ids = shuffled
        # Split sample IDs into train and test
        train_ids = self.sample_ids[:int(len(self.sample_ids) * (1 - test_split))]
        test_ids = self.sample_ids[int(len(self.sample_ids) * (1 - test_split)):]
        val_ids = test_ids[:int(len(test_ids)*0.5)]
        test_ids = test_ids[int(len(test_ids)*0.5):]
        return train_ids, val_ids, test_ids

    def get_dataset_by_group(self, split="train", test_split=0.2, random_seed=42):
        # 1. Get unique subject IDs and shuffle them
        # Using 'host_subject_id' as identified in the skin_1975_map.txt
        unique_subjects = sorted(self.metadata_df['host_subject_id'].unique())
        random.seed(random_seed)
        random.shuffle(unique_subjects)
        
        num_subjects = len(unique_subjects)
        test_idx = int(num_subjects * (1 - test_split))
        val_idx = int(test_idx * 0.9) # Example: uses 10% of training subjects for validation

        # 2. Assign subjects to splits
        if split == "train":
            selected_subjects = unique_subjects[:val_idx]
        elif split == "val":
            selected_subjects = unique_subjects[val_idx:test_idx]
        else: # test
            selected_subjects = unique_subjects[test_idx:]

        # 3. Map subjects back to Sample IDs
        # This ensures ALL samples for the chosen subjects are included
        sample_ids = self.metadata_df[
            self.metadata_df['host_subject_id'].isin(selected_subjects)
        ].index.tolist()

        # Verify sample IDs exist in the BIOM table
        biom_sample_ids = set(self.table.ids(axis='sample'))
        sample_ids = [s for s in sample_ids if s in biom_sample_ids]
        
        if not sample_ids:
            raise ValueError(f"No samples found for the selected {split} subjects.")

        # 4. Filter BIOM table
        split_table = self.table.filter(sample_ids, axis='sample', inplace=False)
        print(f"Subsampling {split}... Subjects: {len(selected_subjects)}, Samples: {split_table.shape[1]}")
        
        if split != "train":
            print("Setting subsample to false for val/test to ensure consistency across epochs")
            subsample = False

        dataset = MicrobiomeSparseDataset(
            biom_table=split_table,
            sample_targets=self.metadata_df.loc[sample_ids][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            test_split=(split == "test"),
            seed=None,
            n_bins=self.n_bins,
            subsample=True,
            categorical_metadata=self.categorical_metadata
        )

        return dataset, split_table, self.metadata_df.loc[sample_ids][self.target_column].to_list()


    def get_dataset_by_json(self, json_path="/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/group_assignments_skin_data.json"):
        import json
        with open(json_path, "r") as f:
            group_assignments = json.load(f)
        
        train_indices = group_assignments["train_indices"]
        test_indices = group_assignments["test_indices"]

        self.json = json_path
        

        # split by host_subject_id

        #if "host_subject_id" not in self.metadata_df.columns:
            


        from sklearn.model_selection import GroupShuffleSplit




        dataset_name = "gut" if "gut" in json_path else "skin" if "skin" in json_path else "oral"
        val_indices = []
        #/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/val_samples_gut.txt
        val_indices_file = f"/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/val_samples_{dataset_name}.txt"
        print("Val indices file: ", val_indices_file, os.path.exists(val_indices_file))
        if not os.path.exists(val_indices_file):
            # create split 
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
            seed=0,
            n_bins=self.n_bins,
            subsample= True,
            categorical_metadata=self.categorical_metadata
        )




        # print("Validation table shape: ", val_table.shape)
        self.val_dataset = MicrobiomeSparseDataset(
            biom_table=val_table,
            sample_targets=self.metadata_df.loc[val_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=0,
            n_bins=self.n_bins,
            subsample=True,
            categorical_metadata=self.categorical_metadata
        )
        print("Test table shape: ", test_table.shape)
        self.test_dataset = MicrobiomeSparseDataset(
            biom_table=test_table,
            sample_targets=self.metadata_df.loc[test_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=None,
            n_bins=self.n_bins,
            subsample=True,
            categorical_metadata=self.categorical_metadata
        )
        self.train_targets = self.metadata_df.loc[train_indices][self.target_column].to_list()
        self.val_targets = self.metadata_df.loc[val_indices][self.target_column].to_list()
        self.test_targets = self.metadata_df.loc[test_indices][self.target_column].to_list()

        # get train targets mean and std
        mean_value = np.mean(self.train_targets)
        std_value = np.std(self.train_targets)

        print("mean value", mean_value )
        print("std value", std_value)
    
    def get_test_dataset(self, seed = 0):
        import json

        if not self.json:
                raise ValueError("JSON path not set. Please run get_dataset_by_json first to set the JSON path for splits.")
        
        with open(self.json, "r") as f:
            group_assignments = json.load(f)
        
        test_indices = group_assignments["test_indices"]

        test_table = self.table.filter(test_indices, axis='sample', inplace=False)

        test_dataset = MicrobiomeSparseDataset(
            biom_table=test_table,
            sample_targets=self.metadata_df.loc[test_indices][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=seed,
            n_bins=self.n_bins,
            subsample=True,
            categorical_metadata=self.categorical_metadata
        )

        self.test_dataset = test_dataset

    
def inspect_embeddings_by_age_group(loader, n_samples_per_group=3, top_n_asvs=10):
    """
    Inspect embeddings for top ASVs across age groups (20+, 60+, 80+).
    Uses the loader's table, metadata_df, and embedding_loader directly.
    """
    import numpy as np
    import pandas as pd

    age_groups = {
        "20+": (20, 40),
        "60+": (60, 80),
        "80+": (80, 120),
    }

    results = {}

    for group_name, (age_min, age_max) in age_groups.items():
        print(f"\n{'='*60}")
        print(f"Age Group: {group_name} ({age_min}-{age_max})")
        print(f"{'='*60}")

        # Filter metadata for this age group
        mask = (loader.metadata_df[loader.target_column] >= age_min) & \
               (loader.metadata_df[loader.target_column] < age_max)
        group_meta = loader.metadata_df[mask]

        # Only keep samples that exist in the BIOM table
        biom_ids = set(loader.table.ids(axis='sample'))
        valid_ids = [sid for sid in group_meta.index if sid in biom_ids]

        if len(valid_ids) == 0:
            print(f"  No valid samples found for group {group_name}")
            continue

        # Sample n_samples_per_group randomly
        chosen_ids = pd.Series(valid_ids).sample(
            n=min(n_samples_per_group, len(valid_ids)), random_state=42
        ).tolist()

        group_results = []

        for sid in chosen_ids:
            age = group_meta.loc[sid, loader.target_column]
            print(f"\n  Sample: {sid}  |  Age: {age:.1f}")

            # Filter table to this single sample and get abundance vector
            sample_table = loader.table.filter([sid], axis='sample', inplace=False)
            
            # Convert to dense: shape (n_obs,)
            dense = sample_table.matrix_data.toarray().squeeze()  # (n_asvs,)
            asv_ids = list(sample_table.ids(axis='observation'))

            # Get top N ASVs by abundance
            top_indices = np.argsort(dense)[::-1][:top_n_asvs]
            top_asvs = [(asv_ids[i], dense[i]) for i in top_indices if dense[i] > 0]

            print(f"  Top {top_n_asvs} ASVs:")
            asv_embedding_info = []

            for rank, (asv_id, abundance) in enumerate(top_asvs):
                # Try to load embedding for this ASV
                try:
                    emb = loader.embedding_loader.get_embedding(asv_id)
                    emb_shape = emb.shape
                    emb_f = emb.detach().cpu().float()
                    emb_shape = tuple(emb_f.shape)
                    emb_norm  = float(emb_f.norm())
                    emb_mean  = float(emb_f.mean())
                    emb_std   = float(emb_f.std())
                    has_emb = True
                except Exception as e:
                    print(f"    Warning: Could not load embedding for ASV {asv_id}: {e}")
                    emb_shape = None
                    emb_norm = emb_mean = emb_std = None
                    has_emb = False

                print(
                    f"    [{rank+1:2d}] ASV: {asv_id[:30]}..."
                    f"  Abundance: {abundance:8.1f}"
                    f"  Emb: {'OK' if has_emb else 'MISSING':7s}"
                    + (f"  shape={emb_shape}  norm={emb_norm:.3f}  mean={emb_mean:.4f}  std={emb_std:.4f}" if has_emb else "")
                )

                asv_embedding_info.append({
                    "asv_id": asv_id,
                    "abundance": abundance,
                    "has_embedding": has_emb,
                    "emb_norm": emb_norm,
                    "emb_mean": emb_mean,
                    "emb_std": emb_std,
                })

            group_results.append({
                "sample_id": sid,
                "age": age,
                "top_asvs": asv_embedding_info,
            })

        results[group_name] = group_results

    # Summary: check for missing embeddings
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for group_name, group_results in results.items():
        total_asvs = sum(len(s["top_asvs"]) for s in group_results)
        missing = sum(
            1 for s in group_results
            for a in s["top_asvs"] if not a["has_embedding"]
        )
        print(f"  {group_name}: {total_asvs} ASVs checked, {missing} missing embeddings")

    return results


# Targets are carried through the model as age/10 for numerical conditioning;
# every reported metric is converted back to years with this factor. Keep all
# MAE reporting in years so val and test numbers are directly comparable.
LABEL_TO_YEARS = 10.0


def collate_fn(batch):

    sample_ids = [item['SampleID'] for item in batch]
    targets = torch.stack([item['outdoor_add_0'] / LABEL_TO_YEARS for item in batch])
    seqs_ids = [item['seqs_ids'] for item in batch]

    # Extract embeddings and abundances
    embeddings = [item['embeddings'] for item in batch]
    abundances = [item['abundances'] for item in batch]
    bins_abundance = [item['binned_abundances'] for item in batch]

    # print("Abundances shape:", abundances[0].shape)
    # print(abundances[0].max(), abundances[0].min(), abundances[0].sum())

    # Find the max length of embeddings in the batch for padding
    max_len = max(e.shape[0] for e in embeddings)
    #max_len = 512
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

    # for i in range(len(targets)):

    #     if targets[i] > 8:
    #         print("\nWarning: Target value greater than 9 found in batch at index ", i)
    #         print("Target value greater than 9 found: ", targets[i].item())
    #         print("Abundances for this sample: ", abundances[i][:10])
    #         print("seq ids", seqs_ids[i][:5])

    # Generic categorical metadata (e.g. sex, country) -> [B, n_meta] LongTensor.
    meta_names = list(batch[0].get('categorical_metadata', {}).keys())
    if meta_names:
        features = torch.tensor(
            [[item['categorical_metadata'][name] for name in meta_names] for item in batch],
            dtype=torch.long
        )
    else:
        features = None

    actual_lengths = batch_masks.sum(dim=1, keepdim=True)
    weighted_sum = (padded_embeddings * padded_abundances).sum(dim=1) / actual_lengths
    #print("Batch shape:", padded_abundances.shape)
    #print("padded_abundances[0, :10, 0]:", padded_abundances[0, :10, 0], padded_abundances.shape)
    
    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings,  # (Batch, Max_Len, Emb_Dim)
        'abundances': padded_abundances,  # (Batch, Max_Len, 1)
        'masks': batch_masks,
        'label': targets,  # (Batch, 1),
        'seqs_ids': padded_seqs_ids,
        'weighted_sum': weighted_sum,  # (Batch, Emb_Dim),
        'bins_abundance': padded_bins_abundance,  # (Batch, Max_Len, Num_Bins)
        'features': features,  # (Batch, n_meta) or None
    }


class TestModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, embeddings, abundances, masks):
        batch_size, max_len, emb_dim = embeddings.shape

        #print("embeddings", embeddings[0, :5, :2])
        #print("abundances", abundances[0, :10, 0])
        #weighted_sum = (embeddings * abundances).sum(dim=1) / (masks.sum(dim=1, keepdim=True) + 1e-8)
        masked_embeddings = embeddings * masks.unsqueeze(-1)  # (Batch, Max_Len, Emb_Dim)
        embeddings_summed = masked_embeddings.sum(dim=1)  # (Batch, Emb_Dim)
        #print("Weighted sum shape: ", weighted_sum[0, :10])
        output = self.model(embeddings_summed)  # (Batch, Max_Len) -> (Batch, 1)
        return output, None, None, None, None
    
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


class StratifiedAgeBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that puts every age group into every batch.

    With a skewed cohort (oral: 84% of samples are 18-40) a random batch of 32
    frequently contains nobody over 50, so the sparse ages reach the gradient only
    intermittently. This allocates a fixed number of slots per age group in each
    batch, so every update sees the whole range.

    Slots per group are allocated as n_g ** beta, normalised to batch_size:
        beta = 1.0  -> natural proportions (equivalent to plain shuffling)
        beta = 0.5  -> sqrt damping: sparse groups upscaled, not equalised
        beta = 0.0  -> every group represented equally
    Each group is guaranteed at least `min_per_group` slots. Groups whose sample
    count is smaller than the draws they need are cycled (reshuffled) within the
    epoch; dense groups are drawn without replacement so they still cover.

    `labels` must be aligned with dataset.sample_ids as it currently stands --
    sample_epoch_init() re-subsamples the BIOM table and can change both the
    membership and the order of sample_ids, so rebuild this every epoch.
    """

    def __init__(self, labels, batch_size=32, bin_edges=(30, 40, 50, 60),
                 beta=0.5, min_per_group=1, seed=None):
        self.labels = np.asarray(labels, dtype=np.float64)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.n = len(self.labels)

        group_of = np.digitize(self.labels, list(bin_edges))
        self.groups = []
        for g in np.unique(group_of):
            idx = np.flatnonzero(group_of == g)
            if len(idx):
                self.groups.append(idx)

        counts = np.array([len(g) for g in self.groups], dtype=np.float64)
        raw = counts ** beta
        slots = raw / raw.sum() * batch_size

        alloc = np.maximum(np.floor(slots).astype(int), min_per_group)
        # Reconcile to exactly batch_size: give leftovers to the largest
        # fractional parts, take back from the largest groups if we overshot.
        frac = slots - np.floor(slots)
        while alloc.sum() < batch_size:
            alloc[np.argmax(frac - (alloc - np.floor(slots)))] += 1
        while alloc.sum() > batch_size:
            shrinkable = np.flatnonzero(alloc > min_per_group)
            alloc[shrinkable[np.argmax(counts[shrinkable])]] -= 1
        self.alloc = alloc
        self.num_batches = max(1, self.n // batch_size)

        self.bin_edges = bin_edges
        self.counts = counts.astype(int)

    def describe(self):
        edges = [-np.inf, *self.bin_edges, np.inf]
        lines = []
        for i, (c, a) in enumerate(zip(self.counts, self.alloc)):
            lo, hi = edges[i], edges[i + 1]
            name = f"<{hi:.0f}" if np.isinf(lo) else (f">={lo:.0f}" if np.isinf(hi) else f"{lo:.0f}-{hi - 1:.0f}")
            lines.append(f"   age {name:>7}: n={c:5d} ({100 * c / self.n:4.1f}% of data) "
                         f"-> {a:2d}/{self.batch_size} slots per batch ({100 * a / self.batch_size:4.1f}%)")
        return "\n".join(lines)

    def _pools(self):
        """One reshuffled draw queue per group, long enough for a full epoch."""
        pools = []
        for idx, need in zip(self.groups, self.alloc):
            total = need * self.num_batches
            chunks, drawn = [], 0
            while drawn < total:
                perm = self.rng.permutation(idx)
                chunks.append(perm)
                drawn += len(perm)
            pools.append(np.concatenate(chunks)[:total])
        return pools

    def __iter__(self):
        pools = self._pools()
        cursors = [0] * len(pools)
        for _ in range(self.num_batches):
            batch = []
            for gi, need in enumerate(self.alloc):
                batch.extend(pools[gi][cursors[gi]:cursors[gi] + need].tolist())
                cursors[gi] += need
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def _dataset_ages(dataset):
    """Ages aligned to dataset.sample_ids, i.e. to __getitem__'s indexing."""
    return np.array([float(dataset.sample_targets[sid]) for sid in dataset.sample_ids])


def train_model(model, criterion, optimizer, loader, device, num_epochs, out_dir, count_reg=1.0,
                stratify_beta=None, stratify_edges=(30, 40, 50, 60), batch_size=32):

    best_val_loss = float('inf')
    best_mae = float('inf')
    best_model_path = None
    patience = 15
    counter = 0
    early_stop_warmup_epochs = max(10, int(0.1 * num_epochs))
    val_mae_history = []
    train_mae_history = []
    train_loss_history = []
    val_loss_history = []

    train_dataset = loader.train_dataset
    val_dataset = loader.val_dataset
    test_dataset = loader.test_dataset

    def build_train_loader(epoch=0):
        """Rebuilt every epoch: sample_epoch_init() can change sample_ids, so the
        sampler's index->age mapping has to be recomputed alongside it."""
        if stratify_beta is None:
            return DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=collate_fn, shuffle=True)
        sampler = StratifiedAgeBatchSampler(
            _dataset_ages(train_dataset), batch_size=batch_size,
            bin_edges=stratify_edges, beta=stratify_beta, seed=1234 + epoch)
        if epoch == 0:
            print(f"Stratified age batches ON (beta={stratify_beta}, "
                  f"{len(sampler)} batches/epoch):\n{sampler.describe()}")
        return DataLoader(train_dataset, collate_fn=collate_fn, batch_sampler=sampler)

    train_loader = build_train_loader(epoch=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # results = inspect_embeddings_by_age_group(loader, n_samples_per_group=3, top_n_asvs=10)

    # print("Results of embedding inspection by age group:")
    # for group_name, group_results in results.items():
    #     print(f"\nAge Group: {group_name}")
    #     for sample_info in group_results:
    #         print(f"  Sample ID: {sample_info['sample_id']}, Age: {sample_info['age']:.1f}")
    #         for asv_info in sample_info['top_asvs']:
    #             print(
    #                 f"    ASV ID: {asv_info['asv_id'][:30]}..., "
    #                 f"Abundance: {asv_info['abundance']:.1f}, "
    #                 f"Has Embedding: {'Yes' if asv_info['has_embedding'] else 'No'}, "
    #                 f"Emb Norm: {asv_info['emb_norm']:.3f}, "
    #                 f"Emb Mean: {asv_info['emb_mean']:.4f}, "
    #                 f"Emb Std: {asv_info['emb_std']:.4f}"
    #             )





    
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=5e-5,                # Your peak learning rate
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.1,              # Warm up for the first 10% of training!
    #     div_factor=25.0,            # Start at max_lr / 25
    #     final_div_factor=1000.0     # End at a very tiny learning rate
    # )
    # warmup_epochs = 20
    # eta_min = 1e-6
    # T_0 = 10
    # T_mult = 2

    # warmup_scheduler = LinearLR(
    #     optimizer,
    #     start_factor=0.01,   # start at 1% of target LR
    #     end_factor=1.0,      # ramp to 100%
    #     total_iters=warmup_epochs
    # )

    # cosine_scheduler = CosineAnnealingWarmRestarts(
    #     optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    # )
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs]
    # )



    # train_loader_full = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    # val_split = 0.1
    # torch.manual_seed(42)
    # dataset = train_loader_full.dataset
    # val_size = int(len(dataset) * val_split)
    # train_size = len(dataset) - val_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])




    # save train and val indices for reproducibility
    # with open(os.path.join(out_dir, "train_indices.txt"), "w") as f:
    #     for idx in train_loader.dataset.indices:
    #           f.write(f"{idx}\n")
    # with open(os.path.join(out_dir, "val_indices.txt"), "w") as f:
    #     for idx in val_loader.dataset.indices:
    #          f.write(f"{idx}\n")
    # # save test indices as well 
    


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        #train_loader.dataset.dataset.sample_epoch_init(epoch)  # Reinitialize dataset for new epoch

        train_dataset.sample_epoch_init(epoch)
        train_loader = build_train_loader(epoch=epoch)
        count_loss_value = 0.0
        count_loss_val = 0.0
        train_targets= []
        train_predictions = []
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)  # Scale down to prevent instability
            abundances = batch['abundances'].to(device) / 1e4
            #abundances = torch.log1p(batch['abundances'].to(device))
            masks = batch['masks'].to(device)
            targets = batch['label'].to(device)
            w_emb = batch['weighted_sum'].to(device)
            bins_abundance = batch['bins_abundance'].to(device)
            features = batch['features'].to(device) if batch['features'] is not None else None
            #abundances = bins_abundance
            optimizer.zero_grad()

            #print("abundances", abundances[0, :10, 0])
            # print("masks", masks[0, :10])

            #print("embeddings", embeddings.shape)




            outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks, features)
            # print("outputs", outputs[:10])
            # print("targets", targets[:10])
            # print("counts pred", counts_pred[0, :10])
            # print("abundances", abundances[0, :10, 0])

            #print("abundances", abundances[0, :10, 0])

            #print("count pred", counts_pred[0, :10, 0])
            # counts_pred = torch.log1p(counts_pred)
            # print("count pred log1p", counts_pred[0, :10, 0])

            #abundances = torch.log1p(abundances)
            if counts_pred is not None:
                count_loss = compute_count_loss(abundances, counts_pred, masks).mean()
            
            else:
                count_loss = 0.0

            loss = criterion(outputs, targets) + (count_loss * count_reg)
            count_loss_value += count_loss.item() if count_loss > 0.0 else 0.0

            loss.backward()

            # if counter == 0: # Print just once per epoch
            #     grad_norm = model.proj_embeddings[0].weight.grad.norm().item()
            #     print(f"\nGradient Norm at Proj Layer: {grad_norm:.8f}")

            train_predictions.append(outputs.cpu() )  # Scale back predictions to original range
            train_targets.append(targets.cpu() )
            # print("Gradients for pool_proj layer:")
            # print(model.pool_proj.weight.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

        #scheduler.step() 
        #cosine_scheduler.step()


        avg_train_loss = total_loss / len(train_loader)
        count_loss_value = count_loss_value / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        all_sample_ids = []
        print("Running validation...")
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device) 
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)
                w_emb = batch['weighted_sum'].to(device)
                bins_abundance = batch['bins_abundance'].to(device)
                features = batch['features'].to(device) if batch['features'] is not None else None
                #abundances = bins_abundance

                #abundances = bins_abundance
                sample_ids = batch['SampleID']
                all_sample_ids.extend(sample_ids)



                outputs, counts_pred, _, _, _ = model(embeddings, abundances, masks, features)
                #outputs, _, _, _, _ = model(w_emb)
                #outputs = model(abundances)['regression_output']
                if counts_pred is not None:
                    loss_count = compute_count_loss(abundances, counts_pred, masks).mean()#  compute_count_loss(abundances, counts_pred, loss_type="mse").mean()
                else:
                    loss_count = 0.0
                loss = criterion(outputs, targets)  + (loss_count * count_reg)
                val_loss += loss.item()
                count_loss_val += loss_count.item() if loss_count > 0.0 else 0.0

                all_preds.append(outputs.cpu() )  # Scale back predictions to original range
                all_targets.append(targets.cpu() )

        print("Save sample ids for validation set...")
        with open(os.path.join(out_dir, "val_indices.txt"), "w") as f:
            for idx in all_sample_ids:
                f.write(f"{idx}\n")
        
        avg_val_loss = val_loss / len(val_loader)
        count_loss_val = count_loss_val / len(val_loader)
        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        # Reported in YEARS, matching the test-set metric below.
        mae = torch.mean(torch.abs(all_preds - all_targets)).item() * LABEL_TO_YEARS
        train_mae = torch.mean(torch.abs(torch.cat(train_predictions).squeeze() - torch.cat(train_targets).squeeze())).item() * LABEL_TO_YEARS
        x = torch.cat(train_predictions).squeeze()
        print("Train predictions ", x.min(), x.max(), x.mean(), x.std())

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - train MAE: {train_mae:.4f} yrs | Val Loss: {avg_val_loss:.4f} - Val MAE: {mae:.4f} yrs")
        print(f"Count Loss - Train: {count_loss_value:.7f} | Val: {count_loss_val:.7f}")
        train_mae_history.append(train_mae)
        val_loss_history.append(avg_val_loss)
        val_mae_history.append(mae)

        # Save best model
        if best_mae > mae:
            best_val_loss = avg_val_loss
            best_mae = mae
            best_model_path = os.path.join(out_dir, f"best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            
            counter = 0  # Reset counter when a better model is found
            import matplotlib.pyplot as plt
            y_true = all_targets * LABEL_TO_YEARS  # rescale age/10 -> real age in years
            y_pred = all_preds * LABEL_TO_YEARS
            r2_true = r2_score(y_true, y_pred)
            plt.scatter(y_true, y_pred)
            plt.xlabel("True Age")
            plt.ylabel("Predicted Age")
            plt.title(f"Epoch {epoch+1} - True vs Predicted Age" f"\nMAE: {mae:.2f} yrs, R²: {r2_true:.3f}")
            plt.savefig(os.path.join(out_dir, f"true_vs_pred_epoch_val.png"))
            plt.close()
        else:
            # Warm-up is counted in EPOCHS. `num_warmup_steps` is an optimizer-step
            # count (~0.1 * batches * epochs), so comparing against it here meant the
            # counter never incremented and early stopping never fired.
            if epoch >= early_stop_warmup_epochs:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_train_loss,
            "train/count_loss": count_loss_value,
            "train/train mae": train_mae,
            "val/loss": avg_val_loss,
            "val/mae": mae,
            "lr": optimizer.param_groups[0]['lr']
        })
    
    # NOTE: the in-loop test evaluation that used to live here scored the LAST-epoch
    # weights, not the best checkpoint, and printed it as "Test MAE" alongside the
    # authoritative number from evaluate_model(). Removed to keep one test metric.
    # plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.plot(train_mae_history, label='Train MAE')
    plt.plot(val_mae_history, label='Validation MAE')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()
    return {"best_loss": float(best_val_loss), "best_mae": float(best_mae), "best_model": best_model_path}

def plot_true_vs_pred(true, pred, epoch, out_dir):
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    
    # Create evaluation plot
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 8))    
    # Regression scatter plot
    ax1.scatter(true, pred, alpha=0.5)
    ax1.plot([true.min(), true.max()], 
             [true.min(), true.max()], 
             'r--', label='Perfect Prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Regression Results\nMAE: {mae:.3f}, R²: {r2:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if epoch is not None:
        plt.savefig(os.path.join(out_dir, f"true_vs_pred_epoch_{epoch+1}.png"))
    else:
        plt.savefig(os.path.join(out_dir, f"true_vs_pred_final.png"))
    plt.close()

def evaluate_model(model, loader, device, output_dir):
    model.eval()

    evals = 10
    test_all = []
    for k in range(evals):
        all_preds = []
        all_targets = []

        loader.get_test_dataset(seed=k)
        test_dataset = loader.test_dataset

        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch['embeddings'].to(device) 
                abundances = batch['abundances'].to(device) / 1e4
                masks = batch['masks'].to(device)
                targets = batch['label'].to(device)
                w_emb = batch['weighted_sum'].to(device)
                bins_abundance = batch['bins_abundance'].to(device)
                features = batch['features'].to(device) if batch['features'] is not None else None
                #abundances = bins_abundance

                outputs, _, _, _, _ = model(embeddings, abundances, masks, features)
                #outputs, _, _, _, _ = model(w_emb)
                all_preds.append(outputs.cpu() * LABEL_TO_YEARS)  # back to years
                all_targets.append(targets.cpu() * LABEL_TO_YEARS)


        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        test_mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        test_all.append(test_mae)
        plot_true_vs_pred(all_targets, all_preds, epoch=k, out_dir=output_dir)

    mae_final = float(np.mean(test_all))
    std_final = float(np.std(test_all))
    print(f"Final Test MAE: {mae_final:.4f} +/- {std_final:.4f} yrs (over {evals} subsample seeds)")
    # Previously returned `test_mae` (the last iteration only) instead of the mean.
    return mae_final, std_final

if __name__ == "__main__":

    datasets = {"gut_data" : "gut_4434" , "skin_data": "skin_1975", "oral_data": "oral_2550"}
    dataset_name = "gut_data"
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/Input/{}/" 
    best_params = {
        "gut_data": {"count_reg": 0.1, "seed": 1, "target_column": "age", },
        "skin_data": {"count_reg": 0.1, "seed": 0, "target_column": "qiita_host_age", },
        "oral_data": {"count_reg": 0.1, "seed": 0, "target_column": "qiita_host_age"}
    }
  
    output_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/my_outputs/"
    biom_table_path = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + ".biom")
    metadata_file = os.path.join(data_dir.format(dataset_name), datasets[dataset_name] + "_map.txt")

    embedding_path = os.path.join(output_dir, datasets[dataset_name] + "_embeddings.h5")

    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)



    print("Metadata shape: ", metadata_df.columns)
    target_column = best_params[dataset_name]["target_column"] 
    json_path_gut = f"/s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/2_trails_results/group_assignments_{dataset_name}.json"


    loader = AgeDatasetLoader(biom_table_path, metadata_df, target_column, embedding_path,)
                               #categorical_columns={"sex": "sex"}) #, "country": "country"})
    loader.get_dataset_by_json(json_path=json_path_gut) 

    #val_dataset, val_split_table, val_targets = loader.get_dataset_by_json(split="val", json_path=json_path_gut)
    #test_dataset, test_split_table, test_targets = loader.get_dataset_by_json(split="test", json_path=json_path_gut)
    print("Train dataset size: ", len(loader.train_dataset))
    print("Test dataset size: ", len(loader.test_dataset))

    print("median target ", np.median(loader.train_targets))
    multitask = False
    model_config = {
                "input_dim": 256,
                "hidden_dim": 1024,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "pe": False,

            }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("=========================================================")


      
    multitask = False
    model_config = {
                    "input_dim": 256,
                    "hidden_dim": 1024,
                    "num_heads": 4,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "pe": False,

                }

    seed = best_params[dataset_name]["seed"]
    count_reg = best_params[dataset_name]["count_reg"]

    # Stratified age batches: every batch carries all age groups, sparse groups
    # mildly upscaled. beta=0.5 (sqrt) upscales; None disables (plain shuffling).
    STRATIFY_BETA = 0.5
    STRATIFY_EDGES = (30, 40, 50, 60)

    wandb.init(
    project="microbiome-age-prediction",
    config=model_config # Save your hyperparameters too!
)
    
    res = {"best_loss": float('inf'), "best_mae": float('inf'), "best_model": None}
    seeds = [seed, 42, 123]  # Example: 3 different seeds for robustness
    for i in range(3):
        seed = seeds[i] #+ 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        print("Metadata cardinalities:", loader.metadata_cardinalities)
        model = AMBERRegressor(
                    input_dim=model_config["input_dim"], hidden_dim=model_config["hidden_dim"],
                    num_heads=model_config["num_heads"], num_layers=model_config["num_layers"],
                    dropout=model_config["dropout"],pe= model_config["pe"], grl=False,
                    metadata_cardinalities=loader.metadata_cardinalities
                )
        
        model.reset_parameters()
        #nn.init.constant_(model.output_head[-1].bias, 2.3)  #4.75
        # model = AttnRegressor(
        #             input_dim=model_config["input_dim"], hidden_dim=model_config["hidden_dim"], 
        #             num_heads=model_config["num_heads"], num_layers=model_config["num_layers"], 
        #             dropout=model_config["dropout"],pe= model_config["pe"], grl=False,
                    
        #         )
        # model = MLPRegressor(input_dim=model_config["input_dim"], hidden_dim=model_config["hidden_dim"], 
        #             dropout=model_config["dropout"])
                # Update bin embedding size
        #model.bin_embeddings = nn.Embedding(n_bins + 1, 256, padding_idx=0)
        #model.reset_parameters()
        model = model.to(device)
        wandb.watch(model, log="all", log_freq=10)

                
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        # Plain MSE regresses sparse ages (e.g. the older tail) toward the
        # dense mid-range mean; reweight by inverse label density instead.
        #criterion = DensityWeightedMSELoss(loader.train_targets, target_scale=0.1).to(device)
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.SmoothL1Loss(beta=0.3)  # Huber loss with beta=2.0, which is less sensitive to outliers than MSE
                
        out_dir = f"results/age_prediction/{dataset_name}/Final_final_huber_str_no_metadata/seed_{seed}"
        os.makedirs(out_dir, exist_ok=True)
                
        res = train_model(model, criterion, optimizer, loader,
                                device, num_epochs=300, out_dir=out_dir, count_reg=count_reg,
                                stratify_beta=STRATIFY_BETA, stratify_edges=STRATIFY_EDGES)
        print(f"count_reg={count_reg}, seed={seed}: val MAE={res['best_mae']:.4f} yrs")

        if res["best_model"] is None:
            best_model_path = os.path.join(out_dir, f"best_model.pt")
            print(f"Loading best model from {best_model_path} for evaluation...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print(f"Best model already saved at {res['best_model']}. Loading for evaluation...")
            model.load_state_dict(torch.load(res['best_model'], map_location=device))
            
        print("Evaluating best model on test set...")
        test_mae, test_std = evaluate_model(model, loader, device, out_dir)

        # Both metrics are in YEARS. Test MAE is the number comparable to the
        # published RF/SVM/TRPCA baselines; it used to be printed and discarded.
        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(f"Best Val Loss: {res['best_loss']:.4f}\n")
            f.write(f"Best Val MAE (years): {res['best_mae']:.4f}\n")
            f.write(f"Test MAE (years): {test_mae:.4f} +/- {test_std:.4f}\n")
            f.write(f"Best Model Path: {res['best_model']}\n")
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({
                "dataset": dataset_name,
                "seed": seed,
                "count_reg": count_reg,
                "stratify_beta": STRATIFY_BETA,
                "stratify_edges": list(STRATIFY_EDGES) if STRATIFY_BETA is not None else None,
                "units": "years",
                "best_val_loss": res["best_loss"],
                "best_val_mae": res["best_mae"],
                "test_mae": test_mae,
                "test_mae_std": test_std,
                "best_model": res["best_model"],
            }, f, indent=2)





