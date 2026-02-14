import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True
    ):
        super().__init__()
        
        # Define one encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        # Stack the layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=None  # optional final norm
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
            padding_mask: Bool tensor of shape [B, L] (True for positions to mask)

        Returns:
            Output tensor of shape [B, L, D]
        """
        return self.encoder(x, src_key_padding_mask=padding_mask)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ReZeroBlock(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero scaling

    def forward(self, x, **kwargs):
        return x + self.alpha * self.layer(x, **kwargs)


class ReZeroTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        final_norm: bool = True
    ):
        super().__init__()
        
        # Define a single encoder layer template
        self.layers = nn.ModuleList([
            ReZeroBlock(nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first
            )) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(input_dim) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, L, D]
            padding_mask: Bool tensor of shape [B, L], where True = pad
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.norm(x)
