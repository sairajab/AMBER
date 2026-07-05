import torch
import os
from tqdm import tqdm
import h5py
import hashlib
import numpy as np
import pandas as pd

# def compute_and_save_embeddings(sequences, tokenizer, model, output_path, batch_size=1, device='cuda', max_length=None):
#     """
#     Compute DNABert embeddings and save to H5 file.
    
#     Args:
#         sequences (dict): Dictionary of sequence_id to sequence
#         tokenizer: DNABert2 tokenizer
#         model: DNABert2 model
#         output_path (str): Path to save H5 file
#         batch_size (int): Batch size for computing embeddings
#         device (str): Device to use for computation
#     """
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     if isinstance(sequences, dict):
#         sequence_items = list(sequences.items())
#     else:
#         sequence_items = list(sequences)
    
#     with h5py.File(output_path, 'w') as f:
#         # Create a group for embeddings
#         emb_group = f.create_group('embeddings')
#         #print("sequence items ", sequence_items)
#         # Process in batches
#         for i in tqdm(range(0, len(sequence_items), batch_size), desc="Computing embeddings"):
#             batch_items = sequence_items[i:i + batch_size]
#             batch_ids, batch_seqs = zip(*batch_items)
#             print("BATCH ", batch_ids, batch_seqs)  # Debugging output
#             # Compute embeddings for batch
#             if max_length:
#                 inputs = tokenizer(list(batch_seqs), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
#             else:
#                 inputs = tokenizer(list(batch_seqs), return_tensors="pt").to(device)
                
#             with torch.no_grad():
#                 outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                
#             # for dna bert cls_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()  # shape: [hidden_dim]
#             print("Outputs keys:", outputs[0].shape, outputs[1].shape)  # Debugging output
#             #print(outputs.keys(), outputs.hidden_states)
#             #print("CLS embedding shape:", cls_embedding.shape)
#             hidden_states = outputs[1]#.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]
#             #print("Hidden states shape:", hidden_states.shape)
#             embeddings =  hidden_states.cpu().numpy()  #cls_embedding #outputs[1].cpu().numpy()
#             #print("Embeddings shape:", embeddings.shape)
#             # Save each sequence's embedding
#             for j, seq_id in enumerate(batch_ids):
#                 emb_group.create_dataset(seq_id, data=embeddings[j])
        
#         # Save metadata
#         f.attrs['embedding_dim'] = embeddings.shape[-1]
#         f.attrs['creation_date'] = str(pd.Timestamp.now())


def compute_and_save_embeddings_fast(
    sequences,
    tokenizer,
    model,
    output_path,
    batch_size=64,
    device='cuda',
    max_length=512,
    pooling='mean',   # 'mean' or 'cls'
    compression="lzf"  # None, "lzf", or "gzip"
):
    import os
    import h5py
    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    assert pooling in ('mean', 'cls')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize input
    sequence_items = list(sequences.items()) if isinstance(sequences, dict) else list(sequences)
    total = len(sequence_items)

    # (Optional but recommended) sort by length → reduces padding waste
    sequence_items.sort(key=lambda x: len(x[1]))

    model.eval()
    model.to(device)

    # ---- Probe embedding dimension with a small batch ----
    sample_seqs = [sequence_items[0][1]]
    inputs = tokenizer(
        sample_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        hidden = outputs[0]

    embedding_dim = hidden.shape[-1]

    # ---- Create HDF5 datasets (CONTIGUOUS, chunked) ----
    with h5py.File(output_path, 'w') as f:
        emb_ds = f.create_dataset(
            "embeddings",
            shape=(total, embedding_dim),
            dtype="float32",
            chunks=(batch_size, embedding_dim),
            compression=compression,
        )

        id_ds = f.create_dataset(
            "ids",
            shape=(total,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        # ---- Main loop ----
        write_idx = 0

        for i in tqdm(range(0, total, batch_size), desc="Computing embeddings"):
            batch_items = sequence_items[i:i + batch_size]
            batch_ids, batch_seqs = zip(*batch_items)

            inputs = tokenizer(
                list(batch_seqs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
                hidden_states = outputs[0]

            if pooling == 'mean':
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embeddings = summed / counts
            else:
                embeddings = hidden_states[:, 0, :]

            # Move once per batch (not per sequence)
            embeddings = embeddings.cpu().numpy().astype("float32")

            bsz = len(batch_ids)

            # ---- Fast contiguous write ----
            emb_ds[write_idx:write_idx + bsz] = embeddings
            id_ds[write_idx:write_idx + bsz] = batch_ids

            write_idx += bsz

        # ---- Metadata ----
        f.attrs['embedding_dim'] = embedding_dim
        f.attrs['total_sequences'] = total
        f.attrs['pooling_method'] = pooling
        f.attrs['max_length'] = max_length
        f.attrs['batch_size'] = batch_size
        f.attrs['creation_date'] = str(pd.Timestamp.now())

    print(f"Saved {total} embeddings to {output_path} (dim={embedding_dim}, pooling={pooling})")


def compute_and_save_embeddings(
    sequences,
    tokenizer,
    model,
    output_path,
    batch_size=128,
    device='cuda',
    max_length=512,
    pooling='cls'  # 'mean' or 'cls'
):
    """
    Compute DNABERT-2 embeddings and save to H5 file.

    Args:
        sequences (dict): Dictionary of sequence_id -> sequence string
        tokenizer:        DNABERT-2 tokenizer
        model:            DNABERT-2 model
        output_path (str): Path to save .h5 file
        batch_size (int): Number of sequences per batch
        device (str):     'cuda' or 'cpu'
        max_length (int): Max token length (hard BERT limit is 512)
        pooling (str):    'mean' (recommended) or 'cls'
    """
    assert pooling in ('mean', 'cls'), "pooling must be 'mean' or 'cls'"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize input
    sequence_items = list(sequences.items()) if isinstance(sequences, dict) else list(sequences)
    total = len(sequence_items)

    model.eval()
    model.to(device)

    with h5py.File(output_path, 'w') as f:
        emb_group = f.create_group('embeddings')
        embedding_dim = None

        for i in tqdm(range(0, total, batch_size), desc="Computing embeddings"):
            batch_items = sequence_items[i:i + batch_size]
            batch_ids, batch_seqs = zip(*batch_items)
            batch_seqs = list(batch_seqs)

            # Tokenize — pad to longest in batch, truncate at hard BERT limit
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,        # pad to longest in THIS batch only
                truncation=True,     # truncate sequences exceeding max_length
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            # outputs[0] is the last hidden state: [batch, seq_len, hidden_dim]
            hidden_states = outputs[0]

            if pooling == 'mean':
                # Masked mean pooling — exclude padding tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()  # [batch, seq_len, 1]
                sum_embeddings = (hidden_states * mask).sum(dim=1)     # [batch, hidden_dim]
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)             # avoid div by zero
                embeddings = (sum_embeddings / sum_mask).cpu().numpy() # [batch, hidden_dim]
            else:
                # CLS token — first token only
                embeddings = hidden_states[:, 0, :].cpu().numpy()      # [batch, hidden_dim]

            # Save each sequence embedding individually
            for j, seq_id in enumerate(batch_ids):
                emb_group.create_dataset(
                    str(seq_id),           # ensure string key
                    data=embeddings[j][None, :],    # shape: [hidden_dim]
                    compression="gzip",    # compress to save disk space
                    compression_opts=4,
                )

            if embedding_dim is None:
                embedding_dim = embeddings.shape[-1]

        # Save metadata
        f.attrs['embedding_dim'] = embedding_dim
        f.attrs['total_sequences'] = total
        f.attrs['pooling_method'] = pooling
        f.attrs['max_length'] = max_length
        f.attrs['batch_size'] = batch_size
        f.attrs['creation_date'] = str(pd.Timestamp.now())

    print(f"Saved {total} embeddings to {output_path}  (dim={embedding_dim}, pooling={pooling})")



def compute_and_save_token_embeddings(
    sequences,
    tokenizer,
    model,
    output_path,
    batch_size=16,
    device='cuda',
    max_length=512,
    dtype="float16"  # use float16 to save space
):
    """
    Compute DNABERT-2 per-token embeddings and save to HDF5.

    Each sequence is stored as:
        [seq_len, hidden_dim]  (no padding)

    Args:
        sequences: dict {seq_id: sequence} or list of tuples
        tokenizer: DNABERT-2 tokenizer
        model: DNABERT-2 model
        output_path: .h5 file
        batch_size: batch size
        device: 'cuda' or 'cpu'
        max_length: truncation limit (BERT max = 512)
        dtype: 'float16' or 'float32'
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sequence_items = (
        list(sequences.items())
        if isinstance(sequences, dict)
        else list(sequences)
    )

    model.eval()
    model.to(device)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    with h5py.File(output_path, 'w') as f:
        emb_group = f.create_group('embeddings')

        for i in tqdm(range(0, len(sequence_items), batch_size),
                      desc="Token embeddings"):

            batch_items = sequence_items[i:i + batch_size]
            batch_ids, batch_seqs = zip(*batch_items)

            inputs = tokenizer(
                list(batch_seqs),
                return_tensors="pt",
                padding=True,          # pad within batch
                truncation=True,
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # [batch, seq_len, hidden_dim]
            hidden = outputs[0]#.last_hidden_state

            attention_mask = inputs["attention_mask"]  # [batch, seq_len]

            for j, seq_id in enumerate(batch_ids):
                seq_len = int(attention_mask[j].sum().item())

                # remove padding
                token_emb = hidden[j, :seq_len, :].detach().cpu().numpy()

                if dtype == "float16":
                    token_emb = token_emb.astype(np.float16)

                emb_group.create_dataset(
                    str(seq_id),
                    data=token_emb,
                    compression="gzip",
                    compression_opts=4
                )

        # metadata
        f.attrs["embedding_type"] = "per_token"
        f.attrs["model"] = "DNABERT-2"
        f.attrs["max_length"] = max_length
        f.attrs["dtype"] = dtype
        f.attrs["creation_date"] = str(pd.Timestamp.now())

    print(f"Saved token embeddings to {output_path}")

class EmbeddingLoader:
    """Memory-efficient embedding loader using H5 files"""
    def __init__(self, embedding_path, embedding_dim=128, preload_all=False):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        if embedding_path and os.path.exists(embedding_path):
            self.file = h5py.File(self.embedding_path, 'r')
            
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