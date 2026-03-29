import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# def create_augmented_batch_old(embeddings, abundances, masks, augmentation_prob=0.3):
#     """
#     Data augmentation for better generalization.
    
#     Strategies:
#     1. Dropout random microbes
#     2. Add Gaussian noise to abundances
#     3. Permute microbe order (set-invariant)
#     """
#     batch_size, seq_len, _ = embeddings.shape
    
#     if torch.rand(1).item() < augmentation_prob:
#         # Randomly drop some microbes
#         dropout_mask = torch.bernoulli(torch.ones_like(masks) * 0.8)  # Keep 80%
#         masks = masks * dropout_mask
        
#         # Add small noise to abundances
#         noise = torch.randn_like(abundances) * 0.01
#         abundances = abundances + noise
#         abundances = abundances.clamp(min=0)
    
#     return embeddings, abundances, masks


def create_augmented_batch_DDD(embeddings, abundances, masks, augmentation_prob=0.3):
    batch_size, seq_len, _ = embeddings.shape
    
    # Clone everything upfront — never touch originals
    aug_embeddings = embeddings.clone()
    aug_abundances = abundances.clone()
    aug_masks = masks.clone()

    # 1. ASV dropout
    dropout_mask = torch.bernoulli(torch.ones_like(aug_masks) * 0.85)
    aug_masks = aug_masks * dropout_mask

    # 2. CLR-safe abundance noise
    noise = torch.randn_like(aug_abundances) * 0.05
    aug_abundances = aug_abundances + noise

    # 3. Rank perturbation — swap pairs
    if torch.rand(1).item() < 0.3:
        perm = torch.randperm(seq_len, device=embeddings.device)[:max(2, seq_len // 10)]
        for i in range(0, len(perm) - 1, 2):
            idx1, idx2 = perm[i], perm[i + 1]
            # Use temporary variables instead of in-place swap
            tmp_abund = aug_abundances[:, idx1].clone()
            aug_abundances[:, idx1] = aug_abundances[:, idx2]
            aug_abundances[:, idx2] = tmp_abund

            tmp_emb = aug_embeddings[:, idx1].clone()
            aug_embeddings[:, idx1] = aug_embeddings[:, idx2]
            aug_embeddings[:, idx2] = tmp_emb

            tmp_mask = aug_masks[:, idx1].clone()
            aug_masks[:, idx1] = aug_masks[:, idx2]
            aug_masks[:, idx2] = tmp_mask

    # 4. Mixup (abundance-only)
    if torch.rand(1).item() < 0.3 and batch_size > 1:
        lam = torch.distributions.Beta(0.4, 0.4).sample()
        mix_idx = torch.randperm(batch_size, device=embeddings.device)
        aug_abundances = lam * aug_abundances + (1 - lam) * aug_abundances[mix_idx]
        aug_masks = aug_masks * aug_masks[mix_idx]
        return aug_embeddings, aug_abundances, aug_masks, lam, mix_idx

    return aug_embeddings, aug_abundances, aug_masks, None, None

def compute_diversity_indices(abundances, masks):
    """
    Compute Shannon and Simpson diversity indices.
    
    Args:
        abundances: [B, L] relative abundances
        masks: [B, L] valid positions
    
    Returns:
        [B, 2] Shannon and Simpson indices
    """
    # Ensure abundances sum to 1 per sample
    masks = masks.unsqueeze(-1)  # [B, L, 1]
    masked_abundances = abundances * masks  # Apply mask to abundances
    abundance_sums = masked_abundances.sum(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = masked_abundances / abundance_sums
    
    # Shannon index: -sum(p_i * log(p_i))
    log_p = torch.log(normalized.clamp(min=1e-10))
    shannon = -(normalized * log_p * masks).sum(dim=1)
    
    # Simpson index: 1 - sum(p_i^2)
    simpson = 1 - (normalized ** 2 * masks).sum(dim=1)
    
    return torch.stack([shannon, simpson], dim=1)

def _mean_absolute_error(pred_val, true_val, fname, labels=None):
    pred_val = np.squeeze(pred_val)
    true_val = np.squeeze(true_val)
    mae = np.mean(np.abs(true_val - pred_val))

    min_x = np.min(true_val)
    max_x = np.max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 50)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 50)
    diag_yy = p(diag_xx)
    data = {"pred": pred_val, "true": true_val}
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred")
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae = "%.4g" % mae
    plot.set(xlabel="True")
    plot.set(ylabel="Predicted")
    plot.set(title=f"MAE: {mae}")
    plt.savefig(fname)
    plt.close()
    return mae

def float_mask(tensor: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """
    Creates a mask for nonzero elements of a tensor. 
    I.e., mask * tensor = tensor and (1 - mask) * tensor = 0.

    Args:
        tensor (torch.Tensor): A tensor of type float.
        dtype (torch.dtype): The data type for the mask (default: torch.float32).

    Returns:
        torch.Tensor: A mask tensor with the same shape as input.
    """
    mask = (tensor != 0).to(dtype)  # Create a boolean mask and convert to float
    return mask



def _relative_abundance(counts: torch.Tensor):
    counts = counts.to(dtype=torch.float32)  # Adjust dtype as needed
    count_sums = counts.sum(dim=1, keepdim=True)  # PyTorch equivalent of tf.reduce_sum
    rel_abundance = counts / count_sums
    return rel_abundance


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        return self.pe[:x.size(1)]

class PositionEmbedding(nn.Module):
    def __init__(self, max_position: int, embedding_dim: int):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position, embedding_dim)
        nn.init.zeros_(self.embedding.weight)  # Initialize with zeros, similar to TensorFlow's "zeros" initializer

    def forward(self, position_ids: torch.Tensor):
        return self.embedding(position_ids)



class CountEncoderNetwork(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, dropout_rate, intermediate_size, max_length):
        super().__init__()
        
        # Embedding layers
        self.positional_encoding = PositionEmbedding(max_length , embedding_dim)

        # Transformer encoder (count_encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            activation="relu",
        )
        self.count_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.count_out = nn.Linear(embedding_dim, 1)  # For regression tasks
        
        # Embedding dimension
        self.embedding_dim = embedding_dim

    def forward(self, sequence_embeddings, relative_abundances, attention_mask=None, count_mask=None):
        """
        tokens: [batch_size, seq_len]
        relative_abundances: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        count_mask: [batch_size, seq_len] (optional, for masked token prediction)
        """
        # Step 2: Add positional encoding and relative abundance bias
        count_embeddings = sequence_embeddings + self.positional_encoding(sequence_embeddings) * (1 + relative_abundances.unsqueeze(-1))

        # Step 3: Apply Transformer encoder
        # Transformer expects input shape: [seq_len, batch_size, embedding_dim]
        count_embeddings = count_embeddings.permute(1, 0, 2)  # Permute to [seq_len, batch_size, embedding_dim]
        
        # Attention mask: Transformer expects it as [batch_size, seq_len]
        encoded = self.count_encoder(count_embeddings, src_key_padding_mask=attention_mask)

        # Permute back: [batch_size, seq_len, embedding_dim]
        encoded = encoded.permute(1, 0, 2)

        # Step 4: Masked token prediction (optional)
        if count_mask is not None:
            # Flatten tensors for masked token prediction
            count_mask = count_mask.view(-1)
            count_pred = encoded.view(-1, self.embedding_dim)[count_mask.bool()]
            count_pred = self.count_out(count_pred) 
        else:
            count_pred = encoded.view(-1, self.embedding_dim)
            count_pred = self.count_out(count_pred)  

        return encoded, count_pred

def create_random_mask(shape, percent, dtype=torch.float32):
    # Generate random values uniformly between 0 and 1
    random_mask = torch.rand(shape, dtype=dtype)
    
    # Compare random values with the percentage threshold
    random_mask = (random_mask <= percent).to(dtype)  # Convert boolean to the specified dtype
    return random_mask


def mask_counts(counts, training=False):
    count_shape = counts.shape
    valid_mask = (counts > 0).to(torch.float32)
    random_mask = (
        create_random_mask(count_shape, percent=0.15, dtype=torch.float32)
        * valid_mask
    )

    if training:
        random_non_mask = (
            create_random_mask(count_shape, percent=0.2, dtype=torch.float32)
            * random_mask
        )
        random_change = create_random_mask(
            count_shape, percent=0.5, dtype=torch.float32
        )
        random_keep = random_non_mask * random_change
        random_change = (1 - random_keep) * valid_mask * random_non_mask
        masked_input = counts * (1 - random_mask)
        masked_input = (
            masked_input + counts * random_keep * random_mask * valid_mask
        )
        random_tokens = torch.rand(count_shape, dtype=torch.float32)
        masked_input = (
            masked_input + random_tokens * random_change * random_mask * valid_mask
        )
        counts = masked_input

    random_mask = random_mask > 0
    return counts, random_mask



# count_mask = float_mask(counts)
# rel_abundance =_relative_abundance(counts)
# training = True
# count_attention_mask = count_mask
# rel_abundance, count_mask = mask_counts(rel_abundance, training=training)
# count_encoder = CountEncoderNetwork(embedding_dim=128, num_layers=2, num_heads=4, dropout_rate=0.1, intermediate_size=512, max_length=1000)
# base_embeddings = "DNABERT embeddings"
# count_gated_embeddings, count_pred = count_encoder(
#             base_embeddings,
#             rel_abundance,
#             attention_mask=count_attention_mask,
#             count_mask=count_mask,
#             training=training,
#         )
