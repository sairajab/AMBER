
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from transformers.activations import gelu_new as gelu
from torch.autograd import Function
from einops.layers.torch import Rearrange


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.alpha = 0.0

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)



#from transformers_model import TransformerEncoder, ReZeroTransformerEncoder

bases = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in product(bases, repeat=3)]
kmer_to_index = {kmer: idx for idx, kmer in enumerate(kmers)}

class RZTXEncoderLayer(nn.Module):
    r"""RZTXEncoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src            
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src
    
# class AttentionPooling(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(hidden_dim, 1)

#     def forward(self, x):  # x: (batch, num_asvs, seq_len, hidden_dim)
#         # Flatten batch and ASVs to apply attention per sequence
#         b, s, h = x.shape
#         x_reshaped = x.view(b, s, h)               # (b*n, seq_len, hidden_dim)
#         attn_scores = self.attn(x)            # (b*n, seq_len, 1)
#         attn_weights = F.softmax(attn_scores, dim=1)   # (b*n, seq_len, 1)
#         pooled = (x * attn_weights).sum(dim=1)  # (b*n, hidden_dim)

#         # Reshape back to (batch, num_asvs, hidden_dim)
#         return pooled.view(b,h)



class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, L, D)
        # mask: (B, L), False = valid, True = masked

        attn_scores = self.attn(x)  # (B, L, 1)

        if mask is not None:
            
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(-1), float('-inf')
            )
            
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)
        return pooled

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


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.1):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Sequential()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.norm(self.fc2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d -> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x, mask=None):
        """
        x:    [B, L, D]
        mask: [B, L] — 1 for valid positions, 0 for padding (optional)
        """
        attn_logits = torch.einsum('b n d, d e -> b n e', x, self.to_attn_logits)  # [B, L, D]

        # if mask is not None:
        #     # If a sample has no valid positions, leave at least one (the first) attended
        #     all_masked = (~mask.bool()).all(dim=1)  # [B]
        #     if all_masked.any():
        #         mask = mask.clone()
        #         mask[all_masked, 0] = 1
        #     mask_expanded = mask.unsqueeze(-1).bool()
        #     attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))

        if mask is not None:
            # Expand mask to [B, L, D] and set padded positions to -inf
            mask_expanded = mask.unsqueeze(-1).bool()  # [B, L, 1]
            attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))

        x = self.pool_fn(x)                     # [B, 1, L, D]
        logits = self.pool_fn(attn_logits)      # [B, 1, L, D]

        attn = logits.softmax(dim=-2)           # softmax over L; padded positions → 0
        return (x * attn).sum(dim=-2).squeeze(1)  # [B, D]


class AbundanceAdaptiveLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(1, dim)
        self.beta_proj = nn.Linear(1, dim)
        # Init so that initially this is identity-like
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, x, abundance):
        # x: [B, L, D], log_abundance: [B, L, 1]
        normed = self.norm(x)
        gamma = self.gamma_proj(abundance)
        beta = self.beta_proj(abundance)
        out = normed.clone()
        out = gamma * normed + beta
        return out

class BasicRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.2, pe = False, grl = False, clr = False, unique_donors_train = 106):
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

        self.scale_embeddings = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.LayerNorm(input_dim)
        )

        self.query_vector = nn.Parameter(torch.randn(1, 1, input_dim) / np.sqrt(input_dim))
        
        self._count_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))


        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim ,max_len=5000)
                # Sequence-level transformer
                        
        #original model has one layer only, I am trying to add one more layer to see the improvement
        self.sequence_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation= gelu,
                batch_first=True,  
              )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)
        
        self.season_feature = nn.Linear(6, input_dim//2)
        self.env_encoder = nn.Linear(1, input_dim//2)

        self.env_embed = nn.Embedding(num_embeddings=3, embedding_dim=input_dim) 
        self.bimonth_embed = nn.Embedding(num_embeddings=6, embedding_dim=input_dim)

        # self.metadata_encoder = nn.Sequential(
        #         nn.Linear(7, input_dim),  # 6 features + env
        #         nn.GELU(),
        #         nn.LayerNorm(input_dim),
        #         nn.Linear(input_dim, input_dim),
        #     )

        self.sample_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=gelu,
                batch_first=True,
                )
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)

        self.norm = nn.LayerNorm(input_dim)
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)


        self.grl = grl
        if self.grl:
            self.grl_layer = GradientReversalLayer()

            self.subject_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, unique_donors_train) # Classifies "Who is this?"
            )

        self.output_head = nn.Sequential(
            nn.Linear(input_dim , input_dim// 2),
            # nn.GELU(), # Smoother than ReLU
            # nn.Dropout(0.1),
            
            # nn.Linear(hidden_dim * 2, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Helps with stable convergence
            nn.GELU(),
            nn.Dropout(dropout),
            

            nn.Linear(input_dim//2, 1, bias=True) ) # Bias=True handles baseline shifts

        self.clr = clr

        if clr:
            self.abundance_proj = nn.Sequential(
                nn.Linear(1, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
            )
        # self.to_gamma = nn.Linear(input_dim, input_dim)
        # self.to_beta  = nn.Linear(input_dim, input_dim)
        # nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)   # start as identity
        # nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)
                
        #self.abundance_proj = nn.Linear(1, input_dim)

        #self.fc3 = nn.Linear(512, 1)
        #self.fusion_layer = nn.Linear(input_dim + 1, input_dim)
        #self.asv_dropout = nn.Dropout(0.1)
        #self.donor_embedding = nn.Embedding(100, 768)
        self.attn_pooling = AttentionPool(input_dim)
        self.pre_norm = AbundanceAdaptiveLayerNorm(input_dim)
        self.reset_parameters()
        



    def reset_parameters(self):
        """Initialize model parameters for better reproducibility and performance"""
        
        # For transformer-based models, use scaled initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use scaled normal initialization (similar to what BERT uses)
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0.0, std=0.02)

        # for name, param in self.named_parameters():
        #     if 'sequence_attention' in name or 'sample_attention' in name:
        #         if param.dim() > 1:  # Weights
        #             nn.init.normal_(param, mean=0.0, std=0.02)
        #         else:  # Biases or LayerNorm params
        #             nn.init.zeros_(param)
        #     elif 'pos_embedding' in name:
        #         nn.init.normal_(param, mean=0.0, std=0.02)
        #     elif 'bins_embedding' in name:
        #         nn.init.normal_(param, mean=0.0, std=0.02)
        #     else:
        #         init_weights(param)

    def forward(self, input_embeddings, abundances, masks, features=None, env=None, donor_ids=None):
        """
        Forward pass with debugging information
        """

        batch_size = input_embeddings.shape[0]
        asv_embeddings = self.scale_embeddings(input_embeddings)

        if self.clr:
            weighted_embeddings = asv_embeddings + abundances
        else:
            if self.pe:
                weighted_embeddings = asv_embeddings + self.pos_embedding(asv_embeddings)  * abundances
            else:

                weighted_embeddings =  (asv_embeddings * abundances) + asv_embeddings 

        if self.training: 

            abundances = abundances + torch.randn_like(abundances) * 0.01

            dropout_mask = torch.rand_like(masks.float()) < 0.1
            masks = masks.bool() & ~dropout_mask.bool()
        
        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype) # for 1 query token, 1 for metadata token
        masks = torch.cat([cls_mask, masks], dim=1)
        asv_attention_mask = ~masks.bool()  # Invert mask for transformer (True where we want to attend)
        weighted_embeddings = torch.cat([self.query_vector.expand(batch_size, -1, -1), weighted_embeddings], dim=1)  # Prepend CLS token to the sequence
        
        asv_refined = self.sequence_attention(weighted_embeddings , src_key_padding_mask=asv_attention_mask)
        # Per-ASV count prediction from refined representations
        count_pred = self.count_out(asv_refined[:, 1:, :])  # Skip CLS token for count prediction


        if env is not None and features is not None:
            # meta_features = torch.cat([features, env.unsqueeze(-1)], dim=-1)  # shape: [B, 7]
            # metadata_encoder = self.metadata_encoder(meta_features)  # shape: [B, D]
            #env_embeddings = self.env_encoder(env.unsqueeze(-1))  # shape: [B, D]
            #meta_embeddings = torch.cat([self.gating_feature(features), env_embeddings], dim=1).unsqueeze(1)  # shape: [B, 1, D]
            env = env + 1  # Shift to make 0-based for embedding
            env_embeddings = self.env_embed(env.long()).unsqueeze(1)  # shape: [B, D//2]
            bimonth_embeddings = self.bimonth_embed(features.long()).unsqueeze(1)  # shape: [B, 1, D//2]

            #meta_embeddings = metadata_encoder.unsqueeze(1)  # [B, 1, D]
            asv_refined = torch.cat([env_embeddings, bimonth_embeddings, asv_refined], dim=1)

        

        if self.clr:
            cls_token_abundance = 0
        else:
            cls_token_abundance = 1


        if self.clr:
            abundances = self.abundance_proj(abundances)  # Project abundances to match embedding dimension

            abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=cls_token_abundance)  # only pad left of L, one for metadata one for query token

        
        meta_mask = torch.ones(batch_size, 2, device=masks.device, dtype=masks.dtype) # for 1 query token, 1 for metadata token
        masks = torch.cat([meta_mask, masks], dim=1)
        full_attention_mask = ~masks.bool() 


        target_encoded = self.sample_attention(
            asv_refined,
            src_key_padding_mask=full_attention_mask
        )

        #summary_token = target_encoded[:, 2, :]  # shape: [B, D]
        
        summary_token =self.attn_pooling(target_encoded[:, :, :], masks)  # shape: [B, D]

        if self.grl:
            out = self.grl_layer(summary_token)
            subj_pred = self.subject_head(out)
    
        x = self.output_head(summary_token)  # shape: [B, 1]

        if self.grl:
                return x, count_pred , None, None, subj_pred
    
        return x, count_pred , None, None, None



class BasicRegressorMultiTask(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.2, pe = False):
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
        self.scale_embeddings = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.LayerNorm(input_dim)
        )

        self.query_vector = nn.Parameter(torch.randn(1, 1, input_dim) / np.sqrt(input_dim))
        
        self._count_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        #self.sequence_embedding = nn.Linear(in_features=768, out_features=input_dim)
        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim ,max_len=5000)
                # Sequence-level transformer
                
        #original model has one layer only, I am trying to add one more layer to see the improvement
        self.sequence_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation= gelu,
                batch_first=True,
                )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)

        
        # self.sequence_encoder = TransformerEncoder(input_dim = input_dim,
        #     hidden_dim = hidden_dim,
        #     num_heads = num_heads,
        #     num_layers = num_layers,
        #     dropout= 0.2,
        #     activation= "gelu",
        #     batch_first = True) 
        
        self.gating_feature = nn.Linear(6, 128)
        self.sample_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=gelu,
                batch_first=True,)
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)

        self.norm = nn.LayerNorm(input_dim)


        
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        # Final regression layers
        self.indoor_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.outdoor_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1, bias=False)
        #self.fc3 = nn.Linear(512, 1)
        #self.fusion_layer = nn.Linear(input_dim + 1, input_dim)
        #self.asv_dropout = nn.Dropout(0.1)
        #self.donor_embedding = nn.Embedding(100, 768)
        #self.attn_pooling = AttentionPooling(input_dim)
        self.reset_parameters()




    def reset_parameters(self):
        """Initialize model parameters for better reproducibility and performance"""
        
        # For transformer-based models, use scaled initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use scaled normal initialization (similar to what BERT uses)
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0.0, std=0.02)

        # Apply to custom layers only (don't reinitialize transformer layers)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights) 
        self.count_out.apply(init_weights)
        
        # Initialize positional embeddings if they exist
        if self.pe:
            nn.init.normal_(self.pos_embedding.position_embeddings.weight, mean=0.0, std=0.02)

        # Initialize learnable parameters consistently
        nn.init.normal_(self.query_vector, mean=0.0, std=0.02)
        nn.init.zeros_(self._count_alpha)

    def forward(self, input_embeddings, abundances, masks, features,   donor_ids=None):
        """
        Forward pass with debugging information
        """
        #input_embeddings = self.asv_dropout(input_embeddings)
        #input_embeddings = self.attn_pooling(input_embeddings)  # Apply attention pooling
        #print("Input Embeddings Shape:", input_embeddings.shape)
        input_embeddings = self.scale_embeddings(input_embeddings)
        feature_embeddings = nn.Sigmoid()(self.gating_feature(features))
        input_embeddings = input_embeddings * feature_embeddings.unsqueeze(1)
        
        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)  # shape: [B, 1, D]
        #donor_embeddings = self.donor_embedding(donor_ids).unsqueeze(1)

        asv_embeddings = torch.cat([query_token, input_embeddings], dim=1)  # [B, 1 + L, D]

        abundances = abundances.transpose(1, 2)  # [B, C, L]
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)  # only pad left of L
        abundances = abundances.transpose(1, 2)  # [B, L+1, C]
        
        if self.training: 
            abundances += torch.randn_like(abundances) * 0.01
            dropout_mask = torch.rand_like(masks.float()) < 0.1
            masks = masks.bool() & ~dropout_mask.bool()


        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        
        if self.pe:
            #print(self.pos_embedding(asv_embeddings).max(), self.pos_embedding(asv_embeddings).min())
            weighted_embeddings = asv_embeddings + self.pos_embedding(asv_embeddings) * abundances
            #weighted_embeddings = torch.cat([weighted_embeddings, donor_embeddings], dim=1)

            #fusion_input = torch.cat([self.pos_embedding(asv_embeddings), abundances], dim=-1)
            #weighted_embeddings = self.fusion_layer(fusion_input)
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
        summary_token = self.norm(summary_token)
        # Final regression
        # x = self.dropout(F.relu(self.fc1(summary_token)))
        # x = self.fc2(x)  # shape: [B, 1]
        
        x1 = self.indoor_head(summary_token)
        x2 = self.outdoor_head(summary_token)
        x = x1, x2
        
        return x, count_pred , None
    
class BasicModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.2, pe = False):
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
        self.scale_embeddings = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.LayerNorm(input_dim)
        )

        self.query_vector = nn.Parameter(torch.randn(1, 1, input_dim) / np.sqrt(input_dim))
        
        self._count_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        #self.sequence_embedding = nn.Linear(in_features=768, out_features=input_dim)
        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim ,max_len=5000)
                # Sequence-level transformer
                
        #original model has one layer only, I am trying to add one more layer to see the improvement
        self.sequence_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation= gelu,
                batch_first=True,
                )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)

        
        # self.sequence_encoder = TransformerEncoder(input_dim = input_dim,
        #     hidden_dim = hidden_dim,
        #     num_heads = num_heads,
        #     num_layers = num_layers,
        #     dropout= 0.2,
        #     activation= "gelu",
        #     batch_first = True) 
        
        self.sample_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=gelu,
                batch_first=True,)
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)

        self.norm = nn.LayerNorm(input_dim)
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1, bias=False)
        #self.fc3 = nn.Linear(512, 1)
        #self.fusion_layer = nn.Linear(input_dim + 1, input_dim)
        #self.asv_dropout = nn.Dropout(0.1)
        #self.donor_embedding = nn.Embedding(100, 768)
        #self.attn_pooling = AttentionPooling(input_dim)
        self.reset_parameters()




    def reset_parameters(self):
        """Initialize model parameters for better reproducibility and performance"""
        
        # For transformer-based models, use scaled initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use scaled normal initialization (similar to what BERT uses)
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0.0, std=0.02)

        # Apply to custom layers only (don't reinitialize transformer layers)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights) 
        self.count_out.apply(init_weights)
        
        # Initialize positional embeddings if they exist
        if self.pe:
            nn.init.normal_(self.pos_embedding.position_embeddings.weight, mean=0.0, std=0.02)

        # Initialize learnable parameters consistently
        nn.init.normal_(self.query_vector, mean=0.0, std=0.02)
        nn.init.zeros_(self._count_alpha)

    def forward(self, input_embeddings, abundances, masks, donor_ids=None):
        """
        Forward pass with debugging information
        """
        #input_embeddings = self.asv_dropout(input_embeddings)
        #input_embeddings = self.attn_pooling(input_embeddings)  # Apply attention pooling
        #print("Input Embeddings Shape:", input_embeddings.shape)
        input_embeddings = self.scale_embeddings(input_embeddings)
        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)  # shape: [B, 1, D]
        #donor_embeddings = self.donor_embedding(donor_ids).unsqueeze(1)

        asv_embeddings = torch.cat([query_token, input_embeddings], dim=1)  # [B, 1 + L, D]

        abundances = abundances.transpose(1, 2)  # [B, C, L]
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)  # only pad left of L
        abundances = abundances.transpose(1, 2)  # [B, L+1, C]
        
        if self.training: 
            abundances += torch.randn_like(abundances) * 0.01
            dropout_mask = torch.rand_like(masks.float()) < 0.1
            masks = masks.bool() & ~dropout_mask.bool()


        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        
        if self.pe:
            #print(self.pos_embedding(asv_embeddings).max(), self.pos_embedding(asv_embeddings).min())
            weighted_embeddings = asv_embeddings + self.pos_embedding(asv_embeddings) * abundances
            #weighted_embeddings = torch.cat([weighted_embeddings, donor_embeddings], dim=1)

            #fusion_input = torch.cat([self.pos_embedding(asv_embeddings), abundances], dim=-1)
            #weighted_embeddings = self.fusion_layer(fusion_input)
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
        summary_token = self.norm(summary_token)
        # Final regression
        x = self.dropout(F.relu(self.fc1(summary_token)))
        x = self.fc2(x)  # shape: [B, 1]
    
        
        return x, count_pred , None
    
    

class BasicRegressorwithUnifrac(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.2, pe = False):
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
        self._unifrac_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        #self.sequence_embedding = nn.Linear(in_features=768, out_features=input_dim)
        # Positional Embedding 
        if self.pe:
            self.pos_embedding = LearnablePositionalEmbedding(input_dim ,max_len=5000)
                # Sequence-level transformer
                
        #original model has one layer only, I am trying to add one more layer to see the improvement
        self.sequence_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
                )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)

        self.sample_encoder = nn.TransformerEncoderLayer(
        d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True)
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)


        self.unifrac_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        ),
        num_layers=num_layers
    )
                
        self.norm = nn.LayerNorm(input_dim)


        
        self.count_out = nn.Linear(in_features= input_dim , out_features=1, bias=False)
        self.unifrac_ff = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Final regression layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1, bias=False)
        self.reset_parameters()




    def reset_parameters(self):
        """Initialize model parameters for better reproducibility and performance"""
        
        # For transformer-based models, use scaled initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use scaled normal initialization (similar to what BERT uses)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)

        # Apply to custom layers only (don't reinitialize transformer layers)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights) 
        self.count_out.apply(init_weights)
        
        # Initialize positional embeddings if they exist
        if self.pe:
            nn.init.normal_(self.pos_embedding.position_embeddings.weight, mean=0.0, std=0.02)

        # Initialize learnable parameters consistently
        nn.init.normal_(self.query_vector, mean=0.0, std=0.02)
        nn.init.zeros_(self._count_alpha)

    def forward(self, input_embeddings, abundances, masks, donor_ids=None):
        """
        Forward pass with debugging information
        """
        #input_embeddings = self.asv_dropout(input_embeddings)
        #input_embeddings = self.attn_pooling(input_embeddings)  # Apply attention pooling
        #print("Input Embeddings Shape:", input_embeddings.shape)
        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)  # shape: [B, 1, D]
        #donor_embeddings = self.donor_embedding(donor_ids).unsqueeze(1)

        asv_embeddings = torch.cat([query_token, input_embeddings], dim=1)  # [B, 1 + L, D]

        abundances = abundances.transpose(1, 2)  # [B, C, L]
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)  # only pad left of L
        abundances = abundances.transpose(1, 2)  # [B, L+1, C]
        
        if self.training: 
            abundances += torch.randn_like(abundances) * 0.01
            dropout_mask = torch.rand_like(masks.float()) < 0.1
            masks = masks.bool() & ~dropout_mask.bool()


        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        

        unifrac_gated_embeddings = self.unifrac_encoder(asv_embeddings, src_key_padding_mask=attention_mask)
        asv_embeddings = asv_embeddings + unifrac_gated_embeddings * self._unifrac_alpha
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)
        
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
        summary_token = self.norm(summary_token)
        # Final regression
        x = self.dropout(F.relu(self.fc1(summary_token)))
        x = self.fc2(x)  # shape: [B, 1]
        
        return x, count_pred , unifrac_pred

    