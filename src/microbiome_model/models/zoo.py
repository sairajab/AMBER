
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from transformers_model import TransformerEncoder, ReZeroTransformerEncoder
from nt_model import NucleotideTransformer, ASVEncoder, ASVEncoderWithTransformer

bases = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in product(bases, repeat=3)]
kmer_to_index = {kmer: idx for idx, kmer in enumerate(kmers)}


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)  # Learnable embeddings

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns: Position embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_length, _ = x.shape  # Extract seq_len from input
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
        
        return self.position_embeddings(positions).expand(batch_size, -1, -1)  # Shape: (batch_size, seq_len, d_model)


class SampleLevelRegressor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, num_heads=4, num_layers=4, dropout=0.2, use_nt_encoder=False, pe=False, kmer_embedding=False):
        """
        Regressor that processes multiple sequences per sample.
        
        Args:
            input_dim (int): Dimension of input embeddings from DNABert2
            hidden_dim (int): Dimension of hidden layers
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dropout (float): Dropout rate
        """
        super().__init__()
        self.pe = pe
        self.use_nt_encoder = use_nt_encoder
        self.kmer_embedding = kmer_embedding
        self.input_dim = input_dim
        # Add input normalization
        #self.input_norm = nn.LayerNorm(input_dim)
        # Initialize a learned query vector
        if self.use_nt_encoder:
            self.nt_encoder = ASVEncoderWithTransformer(
                vocab_size=1260,  # One-hot A/C/G/T → 4 channels
                embed_dim=32,  # Output embedding dimension
                hidden_dim=hidden_dim,
                num_heads=1,
                num_layers=3,
                dropout=dropout,
                max_seq_len=250
            )
            self.asv_scale = nn.Linear(32, input_dim)

        self.query_vector = nn.Parameter(torch.randn(1, 1, input_dim) / np.sqrt(input_dim))
        
        self._count_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._sample_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._unifrac_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        #self.sequence_embedding = nn.Linear(in_features=768, out_features=input_dim)
        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim)
        if self.kmer_embedding:
            self.kmer_embedding = nn.Embedding(len(kmer_to_index)+2, input_dim, padding_idx=0)
        
        self.layer_norm = nn.LayerNorm(input_dim)
        # Sequence-level transformer
        self.count_encoder = ReZeroTransformerEncoder(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        num_heads = num_heads,
        num_layers = 2,
        dropout = dropout,
        activation = "gelu",
        batch_first = True
        )
        self.sample_attention = ReZeroTransformerEncoder(
        input_dim = input_dim,
        hidden_dim = 1024,
        num_heads =4,
        num_layers = 4,
        dropout = dropout,
        activation = "gelu",
        batch_first = True
        )

        self.unifrac_encoder = ReZeroTransformerEncoder(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        num_heads = 4,
        num_layers = 4,
        dropout = dropout,
        activation = "gelu",
        batch_first = True
        )      
        
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)
        
        # # Sample-level transformer to aggregate sequence information
        self.target_encoder = ReZeroTransformerEncoder(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        num_heads = num_heads,
        num_layers = 4,
        dropout = dropout,
        activation = "gelu",
        batch_first = True
        )
        
        # Final regression layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1) #, bias=False
        self.nuc_logits = nn.Linear(input_dim, 6, bias=False)  # Output logits for A/C/G/T/N
        self.asv_norm = nn.LayerNorm(input_dim)
        self.asv_pos = LearnablePositionalEmbedding(input_dim)
        #self.fc3 = nn.Linear(512, 1)
        
        self.unifrac_ff = nn.Linear(input_dim, input_dim, bias=False)


    
    def _split_asvs(self, embeddings):
        """
        Split the embeddings into ASV and nucleotide components.
        
        Args:
            embeddings (Tensor): Input embeddings of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tuple: ASV embeddings and nucleotide embeddings
        """
        nuc_embeddings = embeddings[:, :, :-1, :]
        nucleotides = self.nuc_logits(nuc_embeddings)
        nucleotides = F.softmax(nucleotides, dim=-1)  # Convert logits to probabilities

        asv_embeddings = embeddings[:, :, 0, :]
        asv_embeddings = self.asv_norm(asv_embeddings)

        asv_embeddings = asv_embeddings + self.asv_pos(asv_embeddings)

        return asv_embeddings, nucleotides
    
    def forward(self, input_embeddings, abundances, masks):
        """
        Forward pass with debugging information
        """

        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)  # shape: [B, 1, D]
        zero_token = torch.zeros(batch_size, 1, self.input_dim, device="cuda", dtype=torch.float32)

        if self.kmer_embedding:
            # Get the kmer embeddings
            embeddings = self.kmer_embedding(input_embeddings)
            mask = (input_embeddings != 0).unsqueeze(-1)  # mask out paddings (0 index)
            summed = (embeddings * mask).sum(dim=2)
            lengths = mask.sum(dim=2).clamp(min=2)
            asv_embeddings = summed / lengths  # [batch_size, embedding_dim]
        elif self.use_nt_encoder:
            # Use the nucleotide transformer encoder
            if input_embeddings.dtype != torch.int64:
                input_embeddings = input_embeddings.to(torch.int64)
            embeddings = self.nt_encoder(input_embeddings)   
            embeddings = self.asv_scale(embeddings)  # Project to input_dim
            asv_embeddings, nucleotides = self._split_asvs(embeddings)
            
        else:
            #asv_embeddings = self.sequence_embedding(input_embeddings)  # [B, L, D]
            asv_embeddings = input_embeddings
        
        padded_asv_embeddings = torch.cat([zero_token, asv_embeddings], dim=1)  # [B, L+1, D]
        asv_embeddings = torch.cat([query_token, asv_embeddings], dim=1)  # [B, 1 + L, D]

        abundances = abundances.transpose(1, 2)  # [B, C, L]
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)  # only pad left of L
        abundances = abundances.transpose(1, 2)  # [B, L+1, C]

        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        
        sample_embeddings = self.sample_attention(asv_embeddings,padding_mask=attention_mask)
        
        sample_embeddings = padded_asv_embeddings + sample_embeddings * self._sample_alpha # Residual connection   

        unifrac_gated_embeddings = self.unifrac_encoder(sample_embeddings, padding_mask=attention_mask)
        
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)
        
        unifrac_embeddings = sample_embeddings  + unifrac_gated_embeddings * self._unifrac_alpha # Residual connection

        if self.pe:
            weighted_embeddings = unifrac_embeddings + self.pos_embedding(unifrac_embeddings) * abundances
        else:
            weighted_embeddings = sample_embeddings + sample_embeddings * abundances
        
        # Process sequences with gradient checking
        count_embeddings = self.count_encoder(
            weighted_embeddings,
            padding_mask=attention_mask
        )  # 8 x no of ASvs x 786  
        count_pred = count_embeddings[:, 1:, :]
        count_pred = self.count_out(count_pred)

        count_alpha = F.softplus(self._count_alpha)

        sequence_encoded = unifrac_embeddings + count_embeddings * count_alpha

        target_encoded = self.target_encoder(
            sequence_encoded,
            padding_mask=attention_mask
        )
        summary_token = target_encoded[:, 0, :]  # shape: [B, D]

        # Final regression
        x = F.relu(self.fc1(summary_token))
        x = self.fc2(x)  # shape: [B, 1]
        
        return x, count_pred , unifrac_pred, nucleotides


class BasicRegressorWithASVEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.1, pe = False):
        """
        Regressor that processes multiple sequences per sample.
        
        Args:
            input_dim (int): Dimension of input embeddings from DNABert2
            hidden_dim (int): Dimension of hidden layers
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dropout (float): Dropout rate
        """
        super().__init__()
        self.pe = pe
        self.input_dim = input_dim
        # Add input normalization
        #self.input_norm = nn.LayerNorm(input_dim)
        # Initialize a learned query vector
        


        self.query_vector = nn.Parameter(torch.randn(1, 1, input_dim) / np.sqrt(input_dim))
        
        self._count_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        #self.sequence_embedding = nn.Linear(in_features=768, out_features=input_dim)
        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim ,max_len=768)
                # Sequence-level transformer
                
        #original model has one layer only, I am trying to add one more layer to see the improvement
        self.sequence_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True
        )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)

        self.asv_encoder = ASVEncoder(
            vocab_size=1026,  
            embed_dim=32,  # Output embedding dimension
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=512  # Adjust as needed
        )
        self.dense_layer = nn.Linear(32, input_dim)
        
        self.sample_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True
        )
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)


        
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)

        # Final regression layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1) #, bias=False
        #self.fc3 = nn.Linear(512, 1)
        

    
    
    def forward(self, input_embeddings, abundances, masks):
        """
        Forward pass with debugging information
        """

        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)  # shape: [B, 1, D]
        
        asv_embeddings = self.asv_encoder(input_embeddings)   
        asv_embeddings = self.dense_layer(asv_embeddings) 


        asv_embeddings = torch.cat([query_token, asv_embeddings], dim=1)  # [B, 1 + L, D]

        abundances = abundances.transpose(1, 2)  # [B, C, L]
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)  # only pad left of L
        abundances = abundances.transpose(1, 2)  # [B, L+1, C]

        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        
        
        if self.pe:
            weighted_embeddings = asv_embeddings + self.pos_embedding(asv_embeddings) * abundances
        else:
            weighted_embeddings = asv_embeddings + asv_embeddings * abundances
        
        # Process sequences with gradient checking
        
        count_embeddings = self.sequence_attention(
            weighted_embeddings,
            src_key_padding_mask=attention_mask
        )  # 8 x no of ASvs x 786  
        count_pred = count_embeddings[:, 1:, :]
        count_pred = self.count_out(count_pred)
        
        count_alpha = F.softplus(self._count_alpha) 
        sequence_encoded = asv_embeddings + count_embeddings * count_alpha

        target_encoded = self.sample_attention(
            sequence_encoded    ,
            src_key_padding_mask=attention_mask
        )
        summary_token = target_encoded[:, 0, :]  # shape: [B, D]

        # Final regression
        x = F.relu(self.fc1(summary_token))
        x = self.fc2(x)  # shape: [B, 1]
        
        return x, count_pred , None

        