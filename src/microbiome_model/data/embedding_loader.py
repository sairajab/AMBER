import torch
import os
from tqdm import tqdm
import h5py
import hashlib
import numpy as np
import pandas as pd

def compute_and_save_embeddings(sequences, tokenizer, model, output_path, batch_size=1, device='cuda', max_length=None):
    """
    Compute DNABert embeddings and save to H5 file.
    
    Args:
        sequences (dict): Dictionary of sequence_id to sequence
        tokenizer: DNABert2 tokenizer
        model: DNABert2 model
        output_path (str): Path to save H5 file
        batch_size (int): Batch size for computing embeddings
        device (str): Device to use for computation
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process sequences in batches to save memory
    sequence_items = list(sequences.items())
    
    with h5py.File(output_path, 'w') as f:
        # Create a group for embeddings
        emb_group = f.create_group('embeddings')
        #print("sequence items ", sequence_items)
        # Process in batches
        for i in tqdm(range(0, len(sequence_items), batch_size), desc="Computing embeddings"):
            batch_items = sequence_items[i:i + batch_size]
            batch_ids, batch_seqs = zip(*batch_items)
            print("BATCH ", batch_ids, batch_seqs)  # Debugging output
            # Compute embeddings for batch
            if max_length:
                inputs = tokenizer(list(batch_seqs), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
            else:
                inputs = tokenizer(list(batch_seqs), return_tensors="pt").to(device)
                
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                
            # for dna bert cls_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()  # shape: [hidden_dim]
            print("Outputs keys:", outputs[0].shape, outputs[1].shape)  # Debugging output
            #print(outputs.keys(), outputs.hidden_states)
            #print("CLS embedding shape:", cls_embedding.shape)
            hidden_states = outputs[1]#.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]
            #print("Hidden states shape:", hidden_states.shape)
            embeddings =  hidden_states.cpu().numpy()  #cls_embedding #outputs[1].cpu().numpy()
            #print("Embeddings shape:", embeddings.shape)
            # Save each sequence's embedding
            for j, seq_id in enumerate(batch_ids):
                emb_group.create_dataset(seq_id, data=embeddings[j])
        
        # Save metadata
        f.attrs['embedding_dim'] = embeddings.shape[-1]
        f.attrs['creation_date'] = str(pd.Timestamp.now())

class EmbeddingLoader:
    """Memory-efficient embedding loader using H5 files"""
    def __init__(self, embedding_path, embedding_dim=128, preload_all=False):
        self.embedding_path = embedding_path
        self.file = h5py.File(self.embedding_path, 'r')
        self.embedding_dim = embedding_dim
        self.cache = {}
        self.preload_all = preload_all
        
        if preload_all:
            self._preload_embeddings()
            self.file.close()
            self.file = None  # Close file handle if all preloaded
        
    def _preload_embeddings(self):
        """Preload all embeddings into memory for faster access"""
        print("Preloading all embeddings into memory...")
        emb_group = self.file['embeddings']
        for seq_id in tqdm(emb_group.keys(), desc="Loading embeddings"):
            self.cache[seq_id] = torch.from_numpy(emb_group[seq_id][()])
        print(f"Loaded {len(self.cache)} embeddings into memory")
    
    def preload_sequences(self, sequence_ids):
        """Preload specific sequences into memory"""
        print(f"Preloading {len(sequence_ids)} embeddings...")
        emb_group = self.file['embeddings']
        for seq_id in tqdm(sequence_ids, desc="Loading embeddings"):
            if seq_id in emb_group and seq_id not in self.cache:
                self.cache[seq_id] = torch.from_numpy(emb_group[seq_id][()])
        print(f"Cache now contains {len(self.cache)} embeddings")
        
    def __enter__(self):
        self.file = h5py.File(self.embedding_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            
    def _generate_deterministic_vector(self, sequence_id):
        """Generates a deterministic random vector from sequence_id"""
        # Create a consistent seed from the sequence_id
        
        seed = int(hashlib.sha256(sequence_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return torch.from_numpy(vec)

    def get_embedding(self, sequence_id):
        # Check cache first
        if sequence_id in self.cache:
            return self.cache[sequence_id]
        
        # If not in cache, load from disk
        embedding = torch.from_numpy(self.file['embeddings'][sequence_id][()])
        
        # Optionally add to cache (memory permitting)
        if len(self.cache) < 50000:  # Limit cache size
            self.cache[sequence_id] = embedding
            
        return embedding
    
    def get_embeddings_batch(self, sequence_ids):
        """Load multiple embeddings at once for better I/O efficiency"""
        embeddings = []
        uncached_ids = []
        
        # Collect cached embeddings and identify uncached ones
        for seq_id in sequence_ids:
            if seq_id in self.cache:
                embeddings.append(self.cache[seq_id])
            else:
                uncached_ids.append(seq_id)
                embeddings.append(None)  # Placeholder
        
        # Load uncached embeddings in batch
        if uncached_ids:
            emb_group = self.file['embeddings']
            uncached_embeddings = []
            for seq_id in uncached_ids:
                emb = torch.from_numpy(emb_group[seq_id][()])
                uncached_embeddings.append(emb)
                
                # Add to cache if space available
                if len(self.cache) < 50000:
                    self.cache[seq_id] = emb
            
            # Fill in the None placeholders
            uncached_idx = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embeddings[i] = uncached_embeddings[uncached_idx]
                    uncached_idx += 1
        
        return embeddings

        
def test_generate_deterministic_vector():
    
    loader = EmbeddingLoader("dummy_path.h5", embedding_dim=768)

    # Generate vector for same ID twice
    vec1 = loader._generate_deterministic_vector("ffffa3c5e73b91b8e18b4b59fafeb83e")
    vec2 = loader._generate_deterministic_vector("ffffa3c5e73b91b8e18b4b59fafeb83e")

    print("Shape:", vec1.shape)
    assert vec1.shape == (768,), "Output shape mismatch"

    print("Deterministic test (should be True):", torch.allclose(vec1, vec2))
    assert torch.allclose(vec1, vec2), "Vectors should match for same ID"

    # Check that different IDs give different vectors
    vec3 = loader._generate_deterministic_vector("fffa52555f0d542613a26955a558d76d")
    print("Different ID test (should be False):", torch.allclose(vec1, vec3))
    assert not torch.allclose(vec1, vec3), "Different IDs should not produce same vector"

    print("All deterministic vector tests passed ✅")
        
if __name__ == "__main__":
    
    test_generate_deterministic_vector()