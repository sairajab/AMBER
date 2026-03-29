
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from itertools import product
from microbiome_model.models.transformers import TransformerEncoder, ReZeroTransformerEncoder
from microbiome_model.models.nt_model import NucleotideTransformer, ASVEncoder, ASVEncoderWithTransformer

bases = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in product(bases, repeat=3)]
kmer_to_index = {kmer: idx for idx, kmer in enumerate(kmers)}


import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

# --- 1. Gradient Reversal Layer ---
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




class ASVClusterLayer(nn.Module):
    def __init__(
        self,
        emb_dim=768,
        num_clusters=16,      # hyperparameter K
        temperature=0.1,
        use_cosine=True
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.use_cosine = use_cosine

        # Learnable global cluster prototypes
        self.prototypes = nn.Parameter(
            torch.randn(num_clusters, emb_dim)
        )

    def forward(self, x, abundance=None):
        """
        x: (B, N_asv, D)
        abundance: (B, N_asv) or None
        """

        B, N, D = x.shape
        K = self.num_clusters

        if self.use_cosine:
            x_norm = F.normalize(x, dim=-1)
            prot_norm = F.normalize(self.prototypes, dim=-1)

            # similarity (B, N, K)
            sim = torch.matmul(x_norm, prot_norm.T)
            assignments = F.softmax(sim / self.temperature, dim=-1)

        else:
            # Euclidean
            x_exp = x.unsqueeze(2)                  # (B, N, 1, D)
            prot_exp = self.prototypes.unsqueeze(0).unsqueeze(0)  # (1,1,K,D)
            dist = torch.sum((x_exp - prot_exp) ** 2, dim=-1)
            assignments = F.softmax(-dist / self.temperature, dim=-1)

        # ---- Abundance weighting ----
        if abundance is not None:
            assignments = assignments * abundance   # weight by counts

        # Normalize across ASVs so clusters don't explode
        cluster_mass = assignments.sum(dim=1, keepdim=True) + 1e-8
        assignments_norm = assignments / cluster_mass

        # ---- Compute cluster representations ----
        # weighted sum of embeddings per cluster
        # (B, K, D)
        cluster_repr = torch.einsum("bnk,bnd->bkd", assignments_norm, x)

        return cluster_repr, assignments
    
class SoftADDContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma=50.0):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma  # Controls how quickly similarity falls off with ADD distance
    
    def forward(self, embeddings, add_values, donor_ids):
        embeddings = F.normalize(embeddings, dim=1)
        batch_size = embeddings.shape[0]
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Soft target: high weight for close ADD, low for far
        add_diff = torch.abs(add_values.unsqueeze(0) - add_values.unsqueeze(1))
        target_weights = torch.exp(-add_diff ** 2 / (2 * self.sigma ** 2))
        
        # Penalize same-donor similarity to encourage donor-invariance
        same_donor = (donor_ids.unsqueeze(0) == donor_ids.unsqueeze(1)).float()
        target_weights = target_weights * (1.0 - 0.5 * same_donor)
        
        # Zero out diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        target_weights = target_weights * mask.float()
        
        # Normalize weights to form a distribution per row
        target_dist = target_weights / (target_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Log softmax of similarities
        log_probs = sim_matrix - torch.logsumexp(
            sim_matrix.masked_fill(~mask, -1e9), dim=1, keepdim=True
        )
        
        # Cross-entropy between target distribution and similarity distribution
        loss = -(target_dist * log_probs * mask.float()).sum(dim=1).mean()
        
        return loss

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, proj_dim=64):
        super().__init__()
        # This becomes your encoder for downstream regression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Projection head: discarded after pre-training
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
    
    def forward(self, x):
        h = self.encoder(x)           # Keep this
        z = self.projector(h)          # Discard after pre-training
        return h, z


class CLR(nn.Module):
    """Project embeddings to lower dim, then pool"""
    def __init__(self, emb_dim=768, proj_dim=64, hidden_dim=256, dropout=0.3):
        super().__init__()
        # Compress each ASV embedding first
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        
        # Attention pooling on compressed embeddings
        self.attn = nn.Linear(proj_dim + 1 + 7, 1)  # +1 for CLR abundance +7 for metadata
        
        self.regressor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings, abundances, masks, features = None, env = None):
        # 1. Project embeddings down
        proj = self.proj(embeddings)  # [B, seq, 64]
        
        # 2. CLR transform abundances
        ab = abundances.squeeze(-1).clone()
        nonzero = (ab > 0) & (masks > 0)
        log_ab = torch.zeros_like(ab)
        log_ab[nonzero] = torch.log(ab[nonzero])
        count = nonzero.float().sum(1, keepdim=True).clamp(min=1)
        geo_mean = (log_ab * nonzero.float()).sum(1, keepdim=True) / count
        clr = ((log_ab - geo_mean) * nonzero.float()).unsqueeze(-1)  # [B, seq, 1]
        
        # 3. Attention using projected embeddings + CLR abundance
        if features is not None and env is not None:
            env_emb = env.unsqueeze(-1)
            meta_emb = torch.cat([features, env_emb], dim=-1)
            meta_emb = F.relu(meta_emb)
            meta_emb = meta_emb.unsqueeze(1).expand(-1, proj.shape[1], -1)
            attn_input = torch.cat([proj, clr, meta_emb], dim=-1)  # [B, seq, 65 + meta_dim]
        else:
            attn_input = torch.cat([proj, clr], dim=-1)  # [B, seq, 65]
        scores = self.attn(attn_input).squeeze(-1)    # [B, seq]
        scores = scores.masked_fill(~nonzero, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        
        # 4. Pool projected embeddings
        pooled = (proj * weights).sum(dim=1)  # [B, 64]
        
        return self.regressor(pooled), None, None, None, None
    
class NormalizedTransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.2):
        super(NormalizedTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # Swapped to GELU for smoother gradients on the sphere
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.alphaA = nn.Parameter(torch.tensor(1.0))
        self.alphaM = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, padding_mask=None):
        # Normalize input
        x = F.normalize(x, p=2, dim=-1)

        # Attention block (with masking!)
        hA, _ = self.attention(
            query=x, key=x, value=x, 
            key_padding_mask=padding_mask
        )
        hA = F.normalize(hA, p=2, dim=-1)
        x = F.normalize(x + self.alphaA * (hA - x), p=2, dim=-1)

        # MLP block
        hM = self.mlp(x)
        hM = F.normalize(hM, p=2, dim=-1)
        x = F.normalize(x + self.alphaM * (hM - x), p=2, dim=-1)

        return x


class ClusteredRegressor(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=512,
        num_clusters=32,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim

        # ---- 1. Projection ----
        self.scale_embeddings = nn.Sequential(
            nn.Linear(768, input_dim),
            nn.LayerNorm(input_dim)
        )

        # ---- 2. ASV Clustering ----
        self.cluster_layer = ASVClusterLayer(
            emb_dim=input_dim,
            num_clusters=num_clusters,
            temperature=0.05,
            use_cosine=True
        )

        # ---- 3. Metadata Encoders ----
        self.gating_feature = nn.Linear(6, input_dim // 2)
        self.env_encoder = nn.Linear(1, input_dim // 2)

        # ---- 4. Cluster Self-Attention ----
        self.cluster_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Learned CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # ---- 5. Regression Head ----
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_embeddings, abundances, masks=None,
                features=None, env=None):

        B = input_embeddings.size(0)

        # --------------------------------------------------
        # 1. Projection
        # --------------------------------------------------
        x = self.scale_embeddings(input_embeddings)  # (B, N, D)
        # Log-transform raw counts
        abundances = torch.log1p(abundances)

        # --------------------------------------------------
        # 2. Clustering
        # --------------------------------------------------
        cluster_repr, assignments = self.cluster_layer(x, abundances)

        # Normalize cluster representations (stability)
        cluster_repr = F.layer_norm(cluster_repr, (self.input_dim,))

        # --------------------------------------------------
        # 3. Metadata Gating (applied to clusters)
        # --------------------------------------------------
        if features is not None and env is not None:
            valid_meta = (features != -1).any() and (env != -1).any()

            if valid_meta:
                env_emb = self.env_encoder(env.unsqueeze(-1))
                feat_emb = self.gating_feature(features)

                meta_emb = torch.cat([feat_emb, env_emb], dim=1)
                gate = torch.sigmoid(meta_emb)  # (B, D)

                cluster_repr = cluster_repr * gate.unsqueeze(1)

        # --------------------------------------------------
        # 4. Cluster Self-Attention with Residual
        # --------------------------------------------------
        attn_out, _ = self.cluster_attention(
            cluster_repr,
            cluster_repr,
            cluster_repr
        )

        cluster_repr = cluster_repr + attn_out  # residual

        # --------------------------------------------------
        # 5. Learned CLS Pooling
        # --------------------------------------------------
        cls_token = self.cls_token.expand(B, -1, -1)
        attn_input = torch.cat([cls_token, cluster_repr], dim=1)

        pooled_out, _ = self.cluster_attention(
            attn_input,
            attn_input,
            attn_input
        )

        pooled = pooled_out[:, 0]  # CLS token

        # --------------------------------------------------
        # 6. Regression
        # --------------------------------------------------
        age_prediction = self.regression_head(pooled)

        return age_prediction, None, None, None
    
# --- 2. Main Architecture ---
class BasicRegressorGRL(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_heads=4, num_layers=2, dropout=0.2, pe=False, num_subjects=74):
        """
        Args:
            num_subjects (int): Number of unique donors/subjects for the adversary to classify.
        """
        super().__init__()
        self.pe = pe
        self.input_dim = input_dim

        # Input projection
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
            self.pos_embedding = LearnablePositionalEmbedding(input_dim, max_len=5000)
                
        # Sequence Encoder
        self.sequence_encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.sequence_attention = nn.TransformerEncoder(self.sequence_encoder, num_layers=num_layers)

        # Environmental Features
        self.gating_feature = nn.Linear(6, input_dim//2)
        self.env_encoder = nn.Linear(1, input_dim//2)

        # Sample Encoder
        self.sample_encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.sample_attention = nn.TransformerEncoder(self.sample_encoder, num_layers=num_layers)

        self.norm = nn.LayerNorm(input_dim)
        self.count_out = nn.Linear(in_features=input_dim, out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Main Task Head (Age Prediction)
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.grl = GradientReversalLayer()
        self.subject_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subjects) # Classifies "Who is this?"
        )
        self.abundance_proj = nn.Linear(1, input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
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

        self.output_head.apply(init_weights)
        self.count_out.apply(init_weights)
        # Initialize the new subject head
        self.subject_head.apply(init_weights)
        
        if self.pe:
            nn.init.normal_(self.pos_embedding.position_embeddings.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.query_vector, mean=0.0, std=0.02)
        nn.init.zeros_(self._count_alpha)

    def forward(self, input_embeddings, abundances, masks, features=None, env=None, donor_ids=None):
        # 1. Input Processing
        input_embeddings = self.scale_embeddings(input_embeddings)
        env_metadata = env is not None and (env != -1).any()
        feat_metadata = features is not None and (features != -1).any()
        
        if env_metadata and feat_metadata:
            print("Using both environmental and feature metadata for gating.")
            env_embeddings = self.env_encoder(env.unsqueeze(-1))
            meta_embeddings = torch.cat([self.gating_feature(features), env_embeddings], dim=1)
            feature_embeddings = torch.sigmoid(meta_embeddings)
            input_embeddings = input_embeddings * feature_embeddings.unsqueeze(1)
        
        batch_size = input_embeddings.shape[0]
        query_token = self.query_vector.expand(batch_size, -1, -1)

        asv_embeddings = torch.cat([query_token, input_embeddings], dim=1)

        # Abundance padding/transpose logic
        abundances = abundances.transpose(1, 2)
        abundances = F.pad(abundances, pad=(1, 0), mode='constant', value=1)
        abundances = abundances.transpose(1, 2)
        
        # Training noise injection
        if self.training: 
            abundances += torch.randn_like(abundances) * 0.01
            dropout_mask = torch.rand_like(masks.float()) < 0.1
            masks = masks.bool() & ~dropout_mask.bool()

        cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
        masks = torch.cat([cls_mask, masks], dim=1)
        attention_mask = ~masks.bool()  
        abundances = self.abundance_proj(abundances)  # Project abundances to match embedding dimension
        
        if self.pe:
            weighted_embeddings = asv_embeddings + (self.pos_embedding(asv_embeddings) * abundances)
        else:
            weighted_embeddings = asv_embeddings + (asv_embeddings * abundances)

        # 2. Sequence Attention
        count_embeddings = self.sequence_attention(
            weighted_embeddings,
            src_key_padding_mask=attention_mask
        )
        
        # Count prediction head
        count_pred = count_embeddings[:, 1:, :]
        count_pred = self.count_out(count_pred)
        
        # Combine for sample attention
        count_alpha = F.softplus(self._count_alpha) 
        sequence_encoded = asv_embeddings + count_embeddings * count_alpha

        # 3. Sample Attention
        target_encoded = self.sample_attention(
            sequence_encoded,
            src_key_padding_mask=attention_mask
        )
        
        # Extract Summary Token
        summary_token = target_encoded[:, 0, :] 
        summary_token = self.norm(summary_token)

        # 4. Final Heads
        # A. Age Prediction (Main Task)
        age_prediction = self.output_head(summary_token)
        
        # B. Subject Classification (Adversarial Task)
        # Pass through GRL first!
        #reversed_features = self.grl(summary_token)
        subject_logits = self.subject_head(summary_token)
        
        # Return 5 values to match the training loop:
        # age, counts, unifrac(None), diversity(None), subject_logits
        return age_prediction, count_pred, None, None, subject_logits


import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Abundance Gating Module
# -----------------------------
class AbundanceGating(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, embeddings, abundance):
        # abundance: (B, N)
        a = abundance.unsqueeze(-1)  # (B, N, 1)
        gate = self.mlp(a)           # (B, N, D)
        return embeddings * gate


# -----------------------------
# Attention Pooling
# -----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x, mask):
        # x: (B, N, D)
        scores = torch.matmul(x, self.query)  # (B, N)

        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)  # (B, N)

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


# -----------------------------
# Main Model
# -----------------------------
class MicrobiomeTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        fusion_type="gating",      # "gating" | None
        use_attention_bias=False,
        abundance_hidden=64,
        pooling="attention",       # "attention" | "mean"
        num_outputs=1,
        task="regression",         # "regression" | "classification"
        num_datasets=None
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.use_attention_bias = use_attention_bias
        self.pooling_type = pooling
        self.task = task

        if num_datasets is not None:
            self.dataset_embedding = nn.Embedding(num_datasets, embed_dim)
        else:
            self.dataset_embedding = None

        # Abundance gating
        if fusion_type == "gating":
            self.gating = AbundanceGating(embed_dim, abundance_hidden)
        else:
            self.gating = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pooling
        if pooling == "attention":
            self.pool = AttentionPooling(embed_dim)
        else:
            self.pool = None

        # Task head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_outputs)
        )

    # -------------------------
    # Optional abundance bias
    # -------------------------
    def compute_attention_bias(self, abundance):
        # log abundance prior
        bias = torch.log(abundance + 1e-6)
        return bias

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, embeddings, abundance, mask, dataset_id=None):
        """
        embeddings: (B, N, D)
        abundance:  (B, N)
        mask:       (B, N)  1=valid, 0=pad
        dataset_id: (B,) optional
        """

        x = embeddings

        # Add dataset embedding
        if self.dataset_embedding is not None and dataset_id is not None:
            d_emb = self.dataset_embedding(dataset_id)  # (B, D)
            x = x + d_emb.unsqueeze(1)

        # Abundance gating
        if self.gating is not None:
            x = self.gating(x, abundance)

        # Transformer requires key_padding_mask: True = ignore
        key_padding_mask = mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Pooling
        if self.pooling_type == "attention":
            pooled = self.pool(x, mask)
        else:
            masked_x = x * mask.unsqueeze(-1)
            pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

        out = self.head(pooled)

        if self.task == "regression":
            return out.squeeze(-1), None, None, None, None
        else:
            return out, None, None, None, None
        
class GeneralizedRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.3):
        """
        Improved Regressor with Abundance Embeddings and Set-Invariant Attention.
        """
        super().__init__()
        self.input_dim = input_dim

        # 1. DNA Embedding Adapter (Projects DNABert size to model size)
        self.dna_adapter = nn.Sequential(
            nn.Linear(768, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )

        # 2. Continuous Abundance Encoder
        # Projects scalar abundance (0-1) into a high-dim vector
        self.abundance_encoder = nn.Sequential(
            nn.Linear(1, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Dropout(dropout)
        )

        # 3. Metadata Encoder (Env + Features)
        # We fuse these into a single Context Token
        self.meta_encoder = nn.Sequential(
            nn.Linear(6 + 1, input_dim), # 6 features + 1 env
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        # Learnable Query Token (The "CLS" token that aggregates sample info)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        
        # 4. The Core Transformer (Single Deep Stack is usually better than splitting)
        # Norm_first=True is crucial for convergence in deep transformers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Heads
        self.norm = nn.LayerNorm(input_dim)
        
        # Main Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Reconstruction Head (Helps learn generalized embeddings)
        # Tries to predict the abundance back from the embedding
        self.abundance_decoder = nn.Linear(input_dim, 1)
        self.diversity_decoder = nn.Linear(input_dim, 2)  # For Shannon and Simpson indices

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_embeddings, abundances, masks, features, env=None, donor_ids=None):
        """
        Args:
            input_embeddings: [Batch, Seq_Len, 768] (DNABert vectors)
            abundances: [Batch, Seq_Len] (Normalized counts/CLR)
            masks: [Batch, Seq_Len] (1=Real data, 0=Padding)
            features: [Batch, 6] (Sample metadata)
            env: [Batch, 1] (Environmental feature)
        """
        batch_size, seq_len, _ = input_embeddings.shape

        # --- A. Prepare Semantic Embeddings ---
        # [B, L, D]
        dna_embeds = self.dna_adapter(input_embeddings)

        # --- B. Prepare Abundance Embeddings ---
        # Don't multiply! Add information.
        # [B, L, 1] -> [B, L, D]
        abund_embeds = self.abundance_encoder(abundances.unsqueeze(-1))  # Log-transform to handle skewness
        abund_embeds = abund_embeds.squeeze(2)  # [B, L, D]
        token_embeddings = dna_embeds + abund_embeds

        # --- C. Prepare Context Token (Metadata) ---
        if env is not None:
            # Combine env and features: [B, 7]
            meta_input = torch.cat([features, env.unsqueeze(-1)], dim=1)
            meta_token = self.meta_encoder(meta_input).unsqueeze(1)
        else:
            meta_token = torch.zeros(batch_size, 1, self.input_dim, device=dna_embeds.device)

        # --- D. Assemble Sequence ---
        # Structure: [CLS Token, Meta Token, Microbe_1, Microbe_2, ...]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Final Input: [B, 2 + L, D]
        x = torch.cat((cls_tokens, meta_token, token_embeddings), dim=1)

        # --- E. Attention Masking ---
        # We need to mask padding, but allow attention to CLS and Meta tokens
        # Create mask for CLS and Meta (always 1/True)
        special_token_mask = torch.ones((batch_size, 2), device=masks.device)
        full_mask = torch.cat((special_token_mask, masks), dim=1) 
        
        # Transformer expects src_key_padding_mask where True = IGNORE (inverted in newer PyTorch versions)
        # Check your PyTorch version. Standard approach: 
        # Float mask: 0.0 for include, -inf for ignore is safest for 'attn_mask'
        # Bool mask: True for ignore is standard for 'key_padding_mask'
        key_padding_mask = (full_mask == 0) # True where padding exists

        # --- F. Transformer Pass ---
        # Output: [B, 2 + L, D]
        x_out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x_out = self.norm(x_out)

        # --- G. Downstream Tasks ---
        
        # 1. Regression Task (Use the CLS token at index 0)
        cls_out = x_out[:, 0, :] # [B, D]
        regression_pred = self.regressor(cls_out)

        # 2. Auxiliary Task: Reconstruct Abundance (Use Microbe tokens start at index 2)
        # This forces the model to retain "How much" info in the generalized embedding
        microbe_out = x_out[:, 2:, :] # [B, L, D]
        abundance_pred = self.abundance_decoder(microbe_out)
        diversity_pred = self.diversity_decoder(cls_out)

        return regression_pred, abundance_pred, None, diversity_pred
    

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

        