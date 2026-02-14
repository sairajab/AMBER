import torch
import h5py
import numpy as np
from collections import OrderedDict
import pickle
import os
from tqdm import tqdm
import mmap

class OptimizedEmbeddingLoader:
    """Optimized embedding loader with multiple strategies for large datasets"""
    
    def __init__(self, embedding_path, embedding_dim=128, strategy='adaptive', max_cache_size=10000):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.strategy = strategy
        self.max_cache_size = max_cache_size
        
        # Different caching strategies
        self.cache = OrderedDict()  # LRU cache
        self.file = None
        self.memory_mapped = None
        
        # Index for fast lookup
        self.sequence_index = {}
        self._build_index()
        
    def _build_index(self):
        """Build an index of sequence IDs for fast lookup"""
        with h5py.File(self.embedding_path, 'r') as f:
            emb_group = f['embeddings']
            for i, seq_id in enumerate(emb_group.keys()):
                self.sequence_index[seq_id] = i
        print(f"Built index for {len(self.sequence_index)} sequences")
    
    def preload_subset(self, sequence_ids, priority='frequency'):
        """Preload a subset of embeddings based on priority"""
        print(f"Preloading {len(sequence_ids)} high-priority embeddings...")
        
        if priority == 'frequency':
            # Load most frequently accessed sequences first
            sequence_ids = sorted(sequence_ids, key=lambda x: self._get_access_frequency(x), reverse=True)
        
        with h5py.File(self.embedding_path, 'r') as f:
            emb_group = f['embeddings']
            for seq_id in tqdm(sequence_ids[:self.max_cache_size], desc="Preloading"):
                if seq_id in emb_group:
                    self.cache[seq_id] = torch.from_numpy(emb_group[seq_id][()])
    
    def _get_access_frequency(self, seq_id):
        """Placeholder for access frequency tracking"""
        return hash(seq_id) % 1000  # Simple hash-based priority
    
    def get_embedding(self, sequence_id):
        """Get embedding with LRU caching"""
        # Check cache first
        if sequence_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(sequence_id)
            return self.cache[sequence_id]
        
        # Load from disk
        embedding = self._load_from_disk(sequence_id)
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[sequence_id] = embedding
        return embedding
    
    def _load_from_disk(self, sequence_id):
        """Load single embedding from disk"""
        if self.file is None:
            self.file = h5py.File(self.embedding_path, 'r')
        
        return torch.from_numpy(self.file['embeddings'][sequence_id][()])
    
    def get_embeddings_batch_optimized(self, sequence_ids):
        """Optimized batch loading with intelligent caching"""
        embeddings = []
        cache_hits = []
        cache_misses = []
        
        # Separate cached and uncached
        for seq_id in sequence_ids:
            if seq_id in self.cache:
                cache_hits.append(seq_id)
                embeddings.append(self.cache[seq_id])
                # Update LRU
                self.cache.move_to_end(seq_id)
            else:
                cache_misses.append(seq_id)
                embeddings.append(None)  # Placeholder
        
        # Batch load cache misses
        if cache_misses:
            if self.file is None:
                self.file = h5py.File(self.embedding_path, 'r')
            
            emb_group = self.file['embeddings']
            miss_embeddings = []
            
            for seq_id in cache_misses:
                emb = torch.from_numpy(emb_group[seq_id][()])
                miss_embeddings.append(emb)
                
                # Add to cache with eviction
                if len(self.cache) >= self.max_cache_size:
                    self.cache.popitem(last=False)
                self.cache[seq_id] = emb
            
            # Fill placeholders
            miss_idx = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embeddings[i] = miss_embeddings[miss_idx]
                    miss_idx += 1
        
        print(f"Cache hits: {len(cache_hits)}, Cache misses: {len(cache_misses)}")
        return embeddings

class MemoryMappedEmbeddingLoader:
    """Memory-mapped embedding loader for very large datasets"""
    
    def __init__(self, embedding_path, convert_to_mmap=False, preload_all=False):
        self.embedding_path = embedding_path
        self.mmap_path = embedding_path.replace('.h5', '.mmap')
        self.index_path = embedding_path.replace('.h5', '_index.pkl')
        
        if convert_to_mmap or not os.path.exists(self.mmap_path):
            self._convert_to_mmap()
        
        self._load_mmap()
    
    def _convert_to_mmap(self):
        """Convert H5 to memory-mapped format (one-time operation)"""
        print("Converting H5 to memory-mapped format...")
        
        embeddings_list = []
        sequence_index = {}
        
        with h5py.File(self.embedding_path, 'r') as f:
            emb_group = f['embeddings']
            embedding_dim = None
            
            for i, seq_id in enumerate(tqdm(emb_group.keys(), desc="Converting")):
                emb = emb_group[seq_id][()]
                if embedding_dim is None:
                    embedding_dim = emb.shape[-1]
                
                embeddings_list.append(emb.flatten())
                sequence_index[seq_id] = i
        
        # Stack all embeddings
        all_embeddings = np.vstack(embeddings_list).astype(np.float32)
        
        # Save as memory-mapped array
        mmap_array = np.memmap(self.mmap_path, dtype=np.float32, mode='w+', 
                              shape=all_embeddings.shape)
        mmap_array[:] = all_embeddings[:]
        del mmap_array  # Close file
        
        # Save index
        with open(self.index_path, 'wb') as f:
            pickle.dump({'index': sequence_index, 'embedding_dim': embedding_dim}, f)
        
        print(f"Saved {len(sequence_index)} embeddings to memory-mapped format")
    
    def _load_mmap(self):
        """Load memory-mapped array and index"""
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.sequence_index = data['index']
            self.embedding_dim = data['embedding_dim']
        
        # Open memory-mapped array
        num_sequences = len(self.sequence_index)
        self.embeddings = np.memmap(self.mmap_path, dtype=np.float32, mode='r',
                                   shape=(num_sequences, self.embedding_dim))
    
    def get_embedding(self, sequence_id):
        """Get embedding using memory mapping (very fast)"""
        idx = self.sequence_index[sequence_id]
        return torch.from_numpy(self.embeddings[idx].copy())
    
    def get_embeddings_batch(self, sequence_ids):
        """Batch get embeddings (very fast)"""
        indices = [self.sequence_index[seq_id] for seq_id in sequence_ids]
        batch_embeddings = self.embeddings[indices]
        return [torch.from_numpy(emb.copy()) for emb in batch_embeddings]

class StreamingEmbeddingLoader:
    """Streaming loader that processes embeddings on-demand with minimal memory"""
    
    def __init__(self, embedding_path, chunk_size=1000):
        self.embedding_path = embedding_path
        self.chunk_size = chunk_size
        self.current_chunk = {}
        self.current_chunk_ids = set()
        
    def _load_chunk(self, target_seq_ids):
        """Load a chunk of embeddings that covers the target sequences"""
        with h5py.File(self.embedding_path, 'r') as f:
            emb_group = f['embeddings']
            
            # Find sequences to load
            all_seq_ids = list(emb_group.keys())
            target_indices = [i for i, seq_id in enumerate(all_seq_ids) 
                            if seq_id in target_seq_ids]
            
            if not target_indices:
                return
            
            # Load chunk around target sequences
            start_idx = max(0, min(target_indices) - self.chunk_size // 2)
            end_idx = min(len(all_seq_ids), max(target_indices) + self.chunk_size // 2)
            
            chunk_seq_ids = all_seq_ids[start_idx:end_idx]
            
            # Load embeddings for chunk
            self.current_chunk = {}
            for seq_id in chunk_seq_ids:
                self.current_chunk[seq_id] = torch.from_numpy(emb_group[seq_id][()])
            
            self.current_chunk_ids = set(chunk_seq_ids)
    
    def get_embeddings_batch(self, sequence_ids):
        """Get batch of embeddings with streaming loading"""
        target_set = set(sequence_ids)
        
        # Check if we need to load new chunk
        if not target_set.issubset(self.current_chunk_ids):
            self._load_chunk(target_set)
        
        return [self.current_chunk[seq_id] for seq_id in sequence_ids 
                if seq_id in self.current_chunk]