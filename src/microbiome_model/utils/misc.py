import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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



count_mask = float_mask(counts)
rel_abundance =_relative_abundance(counts)
training = True
count_attention_mask = count_mask
rel_abundance, count_mask = mask_counts(rel_abundance, training=training)
count_encoder = CountEncoderNetwork(embedding_dim=128, num_layers=2, num_heads=4, dropout_rate=0.1, intermediate_size=512, max_length=1000)
base_embeddings = "DNABERT embeddings"
count_gated_embeddings, count_pred = count_encoder(
            base_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
            count_mask=count_mask,
            training=training,
        )
