import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
np.set_printoptions(threshold=np.inf)
from microbiome_model.data.embedding_loader import *
from collections import defaultdict

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
    def __init__(self, biom_table, sample_targets, sort_asvs=True, embedding_loader=None, kmer_seqs=None, one_hot_seqs=None, test_split = False,random_vec=False, seed = None, env = None, sample_metadata = None, n_bins = 50, subsample= True, clr = False):
        """
        Dataset class for sample-level microbiome data using disk-based embeddings.
        
        Args:
            biom_data (pd.DataFrame): Abundance data
            embedding_path (str): Path to H5 file containing embeddings
            sample_targets (dict): Dictionary mapping sample IDs to target values
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
        self.sample_seasons = sample_metadata
        self.env_bool = None
        self.n_bins = n_bins
        self.clr = clr
        self.max_len = 1024
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

        return {
            'SampleID': sample_id,
            'embeddings': sample_embeddings,
            'abundances': abundances,
            'binned_abundances': binned_abundances,
            'outdoor_add_0': torch.FloatTensor([self.sample_targets[sample_id]]),
            'seqs_ids': seqs_ids,
            'env': self.env_bool[sample_id] if self.env_bool is not None else -1,
            'season': self.sample_seasons[sample_id] if self.sample_seasons is not None else -1
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
    else:
        padded_embeddings = torch.zeros(len(batch), max_len, embeddings[0].shape[1], embeddings[0].shape[2])  # (Batch, Max_Len, Emb_Dim)
    padded_abundances = torch.zeros(len(batch), max_len, 1)  # (Batch, Max_Len, 1)
    padded_seqs_ids = [None] * len(batch)  # Placeholder for seqs_ids
    batch_masks = torch.zeros(len(batch), max_len)
    padded_binned_abundances = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb  # Copy actual data
        padded_abundances[i, :length, :] = abun  # Copy actual abundances
        padded_binned_abundances[i, :length] = binned_abundances[i]  # Copy actual binned abundances
        batch_masks[i, :length] = 1
    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))
        
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
        'binned_abundances': padded_binned_abundances  # (Batch, Max_Len)
    }


