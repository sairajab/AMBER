import random
import torch
from torch.utils.data import Dataset, DataLoader
from embedding_loader import *
import numpy as np
import time
import os
np.set_printoptions(threshold=np.inf)

class MicrobiomeSparseDataset(Dataset):
    def __init__(self, biom_table, sample_targets, embedding_loader=None, kmer_seqs=None, one_hot_seqs=None, test_split = False,random_vec=False, seed = None, env = None, sample_metadata = None):
        """
        Dataset class for sample-level microbiome data using disk-based embeddings.
        
        Args:
            biom_data (pd.DataFrame): Abundance data
            embedding_path (str): Path to H5 file containing embeddings
            sample_targets (dict): Dictionary mapping sample IDs to target values
        """
        self.biom_table = biom_table
        self.sample_targets = sample_targets
        self.env = env
        if not test_split:
            if seed is not None:
                subsample_seed = seed + 42
                print("Subsampling dataaaaa", subsample_seed)
                self.biom_data = biom_table.subsample(5000, axis = "observation", seed = subsample_seed)
            else:
                self.biom_data = biom_table.subsample(5000,axis = "observation")
        else:
            print("Subsampling data", random_vec)
            subsample_seed2 = random.randint(0, 2**32 - 1)
            self.biom_data = biom_table.subsample(5000, axis = "observation", seed = subsample_seed2)
            #print("Not subsampling data", biom_table.shape)

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
        
        if self.env is not None:
            self.indoor_samples, self.outdoor_samples = self.env
            self.env_bool = {sample_id: 0 if sample_id in self.indoor_samples else 1 for sample_id in self.sample_ids}
        
    
    
    def sample_epoch_init(self, epoch, seed = False):
        
        if epoch > 0:
            print("Resubsampling data for epoch ", epoch)
            if seed:
                subsample_seed = epoch + 42
                print("Subsampling dataaaaa", subsample_seed)
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
    
    def __getitem__(self, idx):
                
        sample_id = self.sample_ids[idx]

        # Get abundances for this sample
        s_mask = self.table_data[1] == idx
        #print(s_mask)
        abundances = self.table_data[0][s_mask]
        
        
        s_obs = self.table_data[2][s_mask]
        s_obs_ids = self.obs_ids[s_obs]
        abundances = abundances / (abundances.sum() + 1e-6)

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
        sorted_order = np.argsort(abundances)[::-1].copy()  # Descending order
        sorted_order = torch.tensor(sorted_order, dtype=torch.long)  # Convert to PyTorch tensor

        # Apply sorting
        abundances = abundances[sorted_order.numpy()].reshape(-1, 1)  # NumPy array indexing
        sample_embeddings = sample_embeddings[sorted_order]  # PyTorch tensor indexing
        seqs_ids = [s_obs_ids[i] for i in sorted_order.numpy()]  # Get sorted sequence IDs
        abundances_tensor = torch.tensor(abundances, dtype=torch.float32)  # Ensuring float32 dtype

        return {
            'SampleID': sample_id,
            'embeddings': sample_embeddings,
            'abundances': abundances_tensor,
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
    
    for i, (emb, abun) in enumerate(zip(embeddings, abundances)):
        length = emb.shape[0]
        padded_embeddings[i, :length, :] = emb  # Copy actual data
        padded_abundances[i, :length, :] = abun  # Copy actual abundances
        batch_masks[i, :length] = 1
    for i, seqs in enumerate(seqs_ids):
        padded_seqs_ids[i] = seqs + [None] * (max_len - len(seqs))
        
    #print("Batch shape:", padded_abundances.shape)
    return {
        'SampleID': sample_ids,
        'embeddings': padded_embeddings,  # (Batch, Max_Len, Emb_Dim)
        'abundances': padded_abundances,  # (Batch, Max_Len, 1)
        'masks': batch_masks,
        'outdoor_add_0': targets,  # (Batch, 1),
        'seqs_ids': padded_seqs_ids,
        'env': torch.tensor(envs),
        'season': torch.tensor(seasons, dtype=torch.float32)
    }


