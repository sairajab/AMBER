import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class NucleotideAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, intermediate_ff=1024, activation='gelu'):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim)
        self.inter_ff = nn.Linear(embed_dim, intermediate_ff)
        self.outer_ff = nn.Linear(intermediate_ff, embed_dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(self, x):
        """
        x: [B, A, N, E]
        """
        B, A, N, E = x.shape
        x = x.view(B * A, N, E)  # Merge B and A for batch

        # Attention
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_output)

        # Feedforward
        ff_input = self.ff_norm(x)
        ff_output = self.outer_ff(self.activation(self.inter_ff(ff_input)))
        x = x + self.ff_dropout(ff_output)

        return x.view(B, A, N, E)  # Reshape back


class ASVEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers=4, dropout=0.2, max_seq_len=512):
        super(ASVEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_projection = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.max_seq_len, self.embed_dim))

        self.encoder_layer =  nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
        # Transformer encoder layer for sequence encoding
        self.sequence_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.num_layers
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        seq_len = x.size(2)
        token_ids = x  # keep for masking
        x = self.input_projection(x) + self.pos_embedding[:,:, :seq_len, :]
        B, A, L, E = x.shape
        
        key_padding_mask = (token_ids == 0)
        key_padding_mask = key_padding_mask.view(B * A, L) 


        x = x.view(B*A, L,E)
        x = self.sequence_encoder(x, src_key_padding_mask=key_padding_mask)
        # Reshape back to [B, A, L, E]
        x = x.view(B, A, L, E)
        # Mean pooling across sequence
        x = x[:, :, 0, :]

        return x

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: [B, ..., L]  (we assume last dim is position axis)
        seq_len = x.size(-2)  # assuming position is the second-last dim
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Broadcast position across batch
        pos = pos.expand(x.size(0), seq_len)  # [B, L]

        return self.pos_emb(pos)  # returns [B, L, D]


class NucleotideTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=32,       # Projection dimension
                 num_heads=4,
                 ff_dim=256,
                 num_layers=4,
                 max_seq_len=512,
                 dropout=0.1,
                 output_dim=1):       # For regression or binary classification
        super(NucleotideTransformer, self).__init__()

        # Project one-hot vectors to embedding dimension
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, max_seq_len+10, embed_dim))
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = 1e-6
        self.intermediate_activation = 'gelu'
        intermediate_ff = ff_dim
        print("Number of attention layers:", self.num_layers)
        # Create multiple attention layers
        self.asv_attention = nn.ModuleList([
                NucleotideAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    intermediate_ff=intermediate_ff,
                    activation=self.intermediate_activation
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            for _ in range(self.num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))  # shape: [1, 1, 1, C]
        self.output_normalization = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # x: [batch_size, seq_len, 4] (one-hot encoded input)
        seq_len = x.size(2)
        x = x + self.pos_embedding[:,:, :seq_len, :]
        # add cls token
        # Pad the third dimension (L) with (0 before, 1 after)
        B, A, L, C = x.shape
        # cls_token = self.cls_token.expand(B, A, 1, C)  # [B, A, 1, C]
        # x_padded = torch.cat([cls_token, x], dim=2)  # along sequence dim

        # x = x_padded
        for layer_idx in range(self.num_layers):
            x = self.asv_attention[layer_idx](
                x
            )

        return self.output_normalization(x)

class ASVEncoderWithTransformer(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=32, hidden_dim=256, num_heads=4, num_layers=4, dropout=0.1, max_seq_len=512):
        super(ASVEncoderWithTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.asv_attention = NucleotideTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.base_tokens = 5
        self.max_bp = max_seq_len
        self.nuc_positions = torch.arange(0, self.base_tokens * self.max_bp, self.base_tokens, dtype=torch.int32)

        #self.positional_embedding = nn.Parameter(torch.arange(1, 1, max_seq_len, embed_dim))
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim)
        self.asv_token = 1

    def forward(self, x):
        
        self.nuc_positions = self.nuc_positions.to(x.device)  # Ensure positions are on the same device
        seq = x
        seq = seq + self.nuc_positions.unsqueeze(0).unsqueeze(0)
        # add class token

        # Create a tensor of shape [B, A, 1] filled with cls_token
        cls_column = torch.full(
            (seq.size(0), seq.size(1), 1),  # [B, A, 1]
            fill_value=self.asv_token,
            dtype=seq.dtype,
            device=seq.device
        )
        seq = torch.cat([seq, cls_column], dim=2)  # Final shape: [4, 81, 251]

        output = self.embedding_layer(seq)
        return self.asv_attention(output)