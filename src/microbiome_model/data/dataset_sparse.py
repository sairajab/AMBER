import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import re
np.set_printoptions(threshold=np.inf)
from microbiome_model.data.embedding_loader import *
from collections import defaultdict

# Timepoint suffix in sample IDs: "...T14", "...T14.1" (13810), "...day8", "...16day2" (14989/15700)
TIMEPOINT_RE = re.compile(r'(?:\.T|day)(\d+)(?:\.\d+)?$')


def parse_timepoint(sample_id):
    m = TIMEPOINT_RE.search(str(sample_id))
    return int(m.group(1)) if m else None


def build_series_keys(metadata_df, donor_col="host_subject_id", site_col="body_site"):
    """sample_id -> (donor, body_site); samples sharing a key form one time series."""
    return dict(zip(metadata_df["sample_name"].astype(str),
                    zip(metadata_df[donor_col], metadata_df[site_col])))


def build_sample_times(metadata_df, time_col="days_since_placement"):
    """sample_id -> elapsed days. Preferred over parse_timepoint: ID labels count
    visits (14989 'day2' can be 2 days after placement when a day was skipped)."""
    return dict(zip(metadata_df["sample_name"].astype(str), metadata_df[time_col].astype(float)))

class DonorAwareSampler(torch.utils.data.Sampler):
    def __init__(self, sample_ids, sampleID_map_donorID, batch_size=64, pairs_per_batch=8):
        self.batch_size = batch_size
        self.pairs_per_batch = pairs_per_batch
        self.n_samples = len(sample_ids)
        
        self.donor_to_indices = defaultdict(list)
        for idx, sid in enumerate(sample_ids):
            donor = sampleID_map_donorID.get(sid)
            if donor is not None:
                self.donor_to_indices[donor].append(idx)
        
        self.paired_donors = [d for d, idxs in self.donor_to_indices.items() if len(idxs) >= 2]
        print(f"DonorAwareSampler: {len(self.paired_donors)} donors with 2+ samples")
    
    def __iter__(self):
        all_indices = list(range(self.n_samples))
        random.shuffle(all_indices)
        used = set()
        batches = []
        
        # 1. Pre-calculate all available pairs for this epoch
        available_pairs = []
        for donor in self.paired_donors:
            idxs = self.donor_to_indices[donor].copy()
            random.shuffle(idxs)
            # Create as many non-overlapping pairs as possible for this donor
            for i in range(0, len(idxs) - 1, 2):
                available_pairs.append([idxs[i], idxs[i+1]])
                
        random.shuffle(available_pairs)
        idx_queue = iter(all_indices)
        
        # 2. Build batches
        while True:
            batch = []
            
            # Inject pairs from the pool
            pairs_added = 0
            while pairs_added < self.pairs_per_batch and available_pairs:
                pair = available_pairs.pop()
                # Ensure neither index was used (edge case if they were pulled from idx_queue earlier)
                if pair[0] not in used and pair[1] not in used:
                    batch.extend(pair)
                    used.update(pair)
                    pairs_added += 1
            
            # Fill remainder from the random queue
            while len(batch) < self.batch_size:
                try:
                    idx = next(idx_queue)
                    if idx not in used:
                        batch.append(idx)
                        used.add(idx)
                except StopIteration:
                    break
            
            if len(batch) < self.batch_size // 2:
                break
            
            random.shuffle(batch)
            batches.append(batch)
        
        random.shuffle(batches)
        
        # Yield as a BatchSampler (Lists of indices)
        for b in batches:
            yield b
    
    def __len__(self):
        return self.n_samples // self.batch_size
    
    
class MicrobiomeSparseDataset(Dataset):
    def __init__(self, biom_table, sample_targets, sort_asvs=True, embedding_loader=None, learnable_embeddings=False, seq_to_idx = None, kmer_seqs=None, one_hot_seqs=None, test_split = False,random_vec=False, seed = None, env = None, sample_metadata = None, n_bins = 50, subsample= True, clr = False, categorical_metadata = None, asv_stats = None, series_keys = None, sample_times = None, max_time_gap = None):
        """
        Dataset class for sample-level microbiome data using disk-based embeddings.

        Args:
            biom_data (pd.DataFrame): Abundance data
            embedding_path (str): Path to H5 file containing embeddings
            sample_targets (dict): Dictionary mapping sample IDs to target values
            categorical_metadata (dict[str, dict]): Optional mapping of
                metadata_name -> {sample_id: category_index}. Any number of
                fields can be supplied (e.g. {"sex": {...}, "country": {...}});
                index 0 is reserved for unknown/missing, matching
                BasicRegressor's metadata embedding convention. Samples not
                present in a mapping default to 0.
            series_keys (dict): Optional sample_id -> series key (e.g. (donor, body_site),
                see build_series_keys). When given, each item also returns the abundances
                of the next timepoint in the same series ('next_abundances').
            sample_times (dict): Optional sample_id -> time (see build_sample_times).
                If None, the time is parsed from the sample ID suffix (T3 / day3).
            max_time_gap (int): If set, only link to a next timepoint at most this far
                ahead (1 = strictly t+1). None = next available timepoint.
        """
        self.biom_table =  biom_table.copy()
        
        self.sample_targets = sample_targets
        self.env = env
        if subsample:
            if seed is not None:
                subsample_seed = seed + 42
                print("Subsampling dataaaaa", subsample_seed)
                self.biom_data = biom_table.subsample(5000, axis = "observation", seed = subsample_seed)
            else:
                self.biom_data = biom_table.subsample(5000 , axis = "observation")
        else:
            self.biom_data = biom_table.copy()
            print("Not subsampling data", biom_table.shape)
        self.sort_asvs = sort_asvs
        print("In data loader ", self.biom_data.shape)
        self.obs_ids = self.biom_data.ids(axis='observation')
        self.sample_ids = self.biom_data.ids()
        self.sample_targets =  {k: v for k, v in sample_targets.items() if k in self.sample_ids}
        self.table_data = self._table_data(self.biom_data)
        self.random_vec = random_vec
        self.kmer_seqs = kmer_seqs
        self.one_hot_seqs = one_hot_seqs
        # Open embedding file in read mode
        self.embedding_loader = embedding_loader
        self.learnable_embeddings = learnable_embeddings
        self.seq_to_idx = seq_to_idx
        self.sample_seasons = sample_metadata
        self.categorical_metadata = categorical_metadata or {}
        self.env_bool = None
        self.n_bins = n_bins
        self.clr = clr
        self.max_len = 512
        self.asv_stats = asv_stats
        self.series_keys = series_keys
        self.sample_times = sample_times
        self.max_time_gap = max_time_gap
        self.next_idx, self.next_dt = self._build_next_index()
        
        if isinstance(self.env, tuple):
            if len(self.env) != 2:
                raise ValueError("Expected tuple with exactly 2 elements (indoor_samples, outdoor_samples)")
            if self.env[0] is None or self.env[1] is None:
                print("No metadata in use")
            else:
                self.indoor_samples, self.outdoor_samples = self.env
                self.env_bool = {sample_id: 0 if sample_id in self.indoor_samples else 1 for sample_id in self.sample_ids}
        
    
    
    def sample_epoch_init(self, epoch, seed = True):
        
        if epoch > 0:
            print("Resubsampling data for epoch ", epoch)
            if seed:
                subsample_seed = epoch + 42
                print("Subsampling dataaaaa", subsample_seed)
                print(self.biom_table.shape)
                self.biom_data = self.biom_table.subsample(5000, axis = "observation", seed = subsample_seed)
                
            else:
                self.biom_data = self.biom_table.subsample(5000, axis = "observation")

        print("In epoch ", epoch, self.biom_data.shape)
        self.obs_ids = self.biom_data.ids(axis='observation')
        self.sample_ids = self.biom_data.ids()
        self.sample_targets =  {k: v for k, v in self.sample_targets.items() if k in self.sample_ids}
        self.table_data = self._table_data(self.biom_data)
        self.next_idx, self.next_dt = self._build_next_index()

    def _build_next_index(self):
        """Map each sample index to the index of the next timepoint in its series.

        Only samples present in this dataset are linked, so the next sample never
        crosses a train/test split. Replicates at the same timepoint all point to
        the first replicate of the next timepoint.
        """
        if self.series_keys is None:
            return {}, {}
        series = defaultdict(lambda: defaultdict(list))
        for i, sid in enumerate(self.sample_ids):
            key = self.series_keys.get(sid)
            t = self.sample_times.get(sid) if self.sample_times is not None else parse_timepoint(sid)
            if key is None or t is None:
                continue
            series[key][t].append(i)

        next_idx, next_dt = {}, {}
        for by_t in series.values():
            times = sorted(by_t)
            for t, t_next in zip(times, times[1:]):
                if self.max_time_gap is not None and t_next - t > self.max_time_gap:
                    continue
                for i in by_t[t]:
                    next_idx[i] = by_t[t_next][0]
                    next_dt[i] = t_next - t
        print(f"Next-timepoint links: {len(next_idx)}/{len(self.sample_ids)} samples")
        return next_idx, next_dt

    def _raw_sample(self, idx):
        """Raw counts and ASV ids for one sample index."""
        s_mask = self.table_data[1] == idx
        return self.table_data[0][s_mask], self.obs_ids[self.table_data[2][s_mask]]
        
    def _table_data(self, table):
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, (coo.row, coo.col))
        # only keep observations with count > 0
        table_mask = data > 0
        data = data[table_mask]
        row = row[table_mask]
        col = col[table_mask]

        
        return data, row, col, shape

    def __len__(self):
        return len(self.sample_ids)

    def get_targets(self):
        """
        Returns the target values for all samples in the dataset.
        
        Returns:
            dict: Dictionary mapping sample IDs to target values.
        """
        return self.sample_targets
    
    def clr_transform(self, abundances, pseudocount=1e-6):
        """Centered log-ratio transform — standard for compositional data"""
        abundances = abundances + pseudocount
        log_abundances = torch.log(abundances)
        geometric_mean = log_abundances.mean(dim=-1, keepdim=True)
        return log_abundances - geometric_mean
    
    def __getitem__(self, idx):
                
        sample_id = self.sample_ids[idx]

        # Get abundances for this sample
        s_mask = self.table_data[1] == idx
        #print(s_mask)
        abundances = self.table_data[0][s_mask]
        
        # coverages = []

        # for i in range(len(self.sample_ids)):
        #     s_mask = self.table_data[1] == i
        #             #print(s_mask)
        #     values = self.table_data[0][s_mask]
        #     #values = np.asarray(sample)
        #     top256 = np.sort(values)[::-1][:512]
        #     coverage = top256.sum() / (values.sum() + 1e-6)
        #     coverages.append(coverage)

        # print("Mean coverage:", np.mean(coverages))
        # print("Median coverage:", np.median(coverages))
        # print("Minimum coverage:", np.min(coverages))
        # print("Samples below 95%:", np.mean(np.array(coverages) < 0.95))
            
        s_obs = self.table_data[2][s_mask]
        s_obs_ids = self.obs_ids[s_obs]

        # # In your dataset __getitem__:
        # if len(abundances) > self.max_len:
        #     # Keep top-k by abundance
        #     top_k = np.argsort(abundances)[-self.max_len:]
        #     s_obs_ids = s_obs_ids[top_k]
        #     abundances = abundances[top_k]



            
        if self.clr:
            
            abundances = self.clr_transform(torch.tensor(abundances, dtype=torch.float32)).numpy()
        else:
            abundances = abundances / (abundances.sum() + 1e-6) #account for varying sequencing depth by converting to relative abundances

        if self.asv_stats is not None:
            means = np.array([self.asv_stats[asv_id][0] for asv_id in s_obs_ids])

            stds = np.array([self.asv_stats[asv_id][1] for asv_id in s_obs_ids])

            normalized_abundances = torch.tensor((abundances - means) / stds, dtype=torch.float32)
            # print("normalized abundances", normalized_abundances)
            # print("abundances", abundances)
        abundances = abundances * 1e4  # Scale abundances for better numerical stability

            
        if len(abundances) > 0:
            abundances = torch.tensor(abundances, dtype=torch.float32)
            ranks = torch.argsort(torch.argsort(abundances)).float()  # Get ranks of abundances
            
            binned_abundances = (ranks / len(abundances) * self.n_bins).long() + 1
        else:
            binned_abundances = torch.empty(0, dtype=torch.long)

        if self.clr and self.sort_asvs:
            print("Warning: sort asvs is {self.sort_asvs} but CLR transform is applied, setting it to False.")
            self.sort_asvs = False

        if self.sort_asvs:
            sorted_order = torch.argsort(abundances, descending=True)
            #sorted_order = np.argsort(-normalized_abundances) # sort by normalized abundances in descending order

        else:
            sorted_order = torch.arange(len(abundances), dtype=torch.long)

        #abundances = self.clr_transform(torch.tensor(abundances, dtype=torch.float32)).numpy()
        if self.kmer_seqs is not None:
            sample_embeddings = torch.stack([
                    self.kmer_seqs[seq_id] for seq_id in s_obs_ids
                ])
        else:
            if self.random_vec:
                    # Generate a random vector for the sequence
                    sample_embeddings = torch.stack([
                        self.embedding_loader._generate_deterministic_vector(seq_id) for seq_id in s_obs_ids
                    ])
            elif self.embedding_loader is None and self.one_hot_seqs is not None:
                    sample_embeddings = torch.stack([
                        self.one_hot_seqs[seq_id] for seq_id in s_obs_ids
                    ])
            elif self.embedding_loader is None and self.learnable_embeddings and self.seq_to_idx is not None:
                # Use learnable embeddings for sequences
                sample_embeddings = torch.tensor([
                                        self.seq_to_idx[seq_id] for seq_id in s_obs_ids
                                    ], dtype=torch.long)
            else:
                # Load the embeddings in batch for better I/O efficiency
                start = time.time()
                if hasattr(self.embedding_loader, 'get_embeddings_batch'):
                    sample_embeddings_list = self.embedding_loader.get_embeddings_batch(s_obs_ids)
                    sample_embeddings = torch.stack(sample_embeddings_list)
                else:
                    
                    sample_embeddings = torch.stack([
                        self.embedding_loader.get_embedding(seq_id) for seq_id in s_obs_ids
                    ])
                end = time.time()
                # Sort indices in descending order based on abundances


        if self.sort_asvs and len(abundances) > self.max_len:
            sorted_order = sorted_order[:self.max_len]
            
            
        # Apply sorting
        abundances = abundances[sorted_order.numpy()].reshape(-1, 1)  # NumPy array indexing
        
        sample_embeddings = sample_embeddings[sorted_order]  # PyTorch tensor indexing
        seqs_ids = [s_obs_ids[i] for i in sorted_order.numpy()]  # Get sorted sequence IDs
        binned_abundances = binned_abundances[sorted_order.numpy()]  # Sort binned abundances as well

        # Abundances at the next timepoint, aligned to this sample's (sorted) ASVs.
        # Relative abundance * 1e4 like 'abundances' (no CLR); ASVs absent at t+1 are 0.
        next_abundances = torch.zeros(len(seqs_ids), 1, dtype=torch.float32)
        n_idx = self.next_idx.get(idx)
        if n_idx is not None:
            n_counts, n_obs_ids = self._raw_sample(n_idx)
            n_rel = n_counts / (n_counts.sum() + 1e-6) * 1e4
            lookup = dict(zip(n_obs_ids, n_rel))
            next_abundances = torch.tensor(
                [lookup.get(s, 0.0) for s in seqs_ids], dtype=torch.float32
            ).reshape(-1, 1)

        return {
            'SampleID': sample_id,
            'embeddings': sample_embeddings,
            'abundances': abundances,
            'binned_abundances': binned_abundances,
            'outdoor_add_0': torch.FloatTensor([self.sample_targets[sample_id]]),
            'seqs_ids': seqs_ids,
            'env': self.env_bool[sample_id] if self.env_bool is not None else -1,
            'season': self.sample_seasons[sample_id] if self.sample_seasons is not None else -1,
            'next_abundances': next_abundances,
            'has_next': n_idx is not None,
            'next_dt': self.next_dt.get(idx, 0),
            'next_SampleID': self.sample_ids[n_idx] if n_idx is not None else None,
            'categorical_metadata': {
                name: mapping.get(sample_id, 0)
                for name, mapping in self.categorical_metadata.items()
            }
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length embeddings and batch data properly.
    
    Args:
        batch (list of dicts): List of samples from the dataset, each containing:
            - 'SampleID': Sample identifier
            - 'embeddings': Tensor of embeddings (variable length)
            - 'abundances': Tensor of abundances
            - 'binned_abundances': Tensor of binned abundances
            - 'outdoor_add_0': Tensor of target values
    
    Returns:
        dict: Batched data with padded embeddings.
    """
    sample_ids = [item['SampleID'] for item in batch]
    envs = [item['env'] for item in batch]
    targets = torch.stack([item['outdoor_add_0'] for item in batch])
    seqs_ids = [item['seqs_ids'] for item in batch]
    seasons = [item['season'] for item in batch]
    seasons = np.array(seasons)

    # Extract embeddings and abundances
    embeddings = [item['embeddings'] for item in batch]
    abundances = [item['abundances'] for item in batch]
    binned_abundances = [item['binned_abundances'] for item in batch]

    # Find the max length of embeddings in the batch for padding
    max_len = max(e.shape[0] for e in embeddings)
    # Pad embeddings and abundances to ensure equal shape in batch
    
    if len(embeddings[0].shape) == 2:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1])  # (Batch, Max_Len, Emb_Dim)
    elif len(embeddings[0].shape) == 1:
        padded_embeddings = torch.zeros(len(batch), max_len)  # (Batch, Max_Len)
    else:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2])  # (Batch, Max_Len, Emb_Dim)
    padded_abundances = torch.zeros(len(batch), max_len, 1)  # (Batch, Max_Len, 1)
    padded_seqs_ids = [None] * len(batch)  # Placeholder for seqs_ids
    batch_masks = torch.zeros(len(batch), max_len)
    padded_binned_abundances = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_next_abundances = torch.zeros(len(batch), max_len, 1)

    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        if padded_embeddings.ndim == 3:
            padded_embeddings[i, :length, :] = emb  # Copy actual data
        elif padded_embeddings.ndim == 2:
            padded_embeddings[i, :length] = emb  # Copy actual data
            
        padded_abundances[i, :length, :] = abun  # Copy actual abundances
        padded_binned_abundances[i, :length] = binned_abundances[i]  # Copy actual binned abundances
        if 'next_abundances' in batch[i]:
            padded_next_abundances[i, :length, :] = batch[i]['next_abundances']
        batch_masks[i, :length] = 1
    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))

    # Generic categorical metadata -> [B, n_meta] LongTensor, in whatever
    # order the dataset's categorical_metadata dict was built with.
    meta_names = list(batch[0].get('categorical_metadata', {}).keys())
    if meta_names:
        features = torch.tensor(
            [[item['categorical_metadata'][name] for name in meta_names] for item in batch],
            dtype=torch.long
        )
    else:
        features = None

    #targets = torch.log1p(targets)  # Log-transform targets for better numerical stability
    #print("Batch shape:", padded_abundances.shape)
    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings,  # (Batch, Max_Len, Emb_Dim)
        'abundances': padded_abundances,  # (Batch, Max_Len, 1)
        'masks': batch_masks,
        'outdoor_add_0': targets,  # (Batch, 1),
        'seqs_ids': padded_seqs_ids,
        'env': torch.tensor(envs),
        'season': torch.tensor(seasons, dtype=torch.float32),
        'binned_abundances': padded_binned_abundances,  # (Batch, Max_Len)
        'features': features,  # (Batch, n_meta) or None
        'meta_names': meta_names,
        'next_abundances': padded_next_abundances,  # (Batch, Max_Len, 1), aligned to 'abundances'
        'has_next': torch.tensor([item.get('has_next', False) for item in batch]),  # (Batch,)
        'next_dt': torch.tensor([item.get('next_dt', 0) for item in batch], dtype=torch.float32),
        'next_SampleID': [item.get('next_SampleID') for item in batch],
    }


