import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings"""

    def __init__(
            self,
            d_model: int,
            max_seq_length: int = 512,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    