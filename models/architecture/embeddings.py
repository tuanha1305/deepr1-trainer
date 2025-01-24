import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .. import LayerNorm
from ..components.hyperbolic import PoincareBallLayer
from ..components.embeddings import PositionalEmbedding


class TextEmbedding(nn.Module):
    """
    Text embedding module with positional encoding
    """

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            max_seq_length: int = 512,
            padding_idx: int = 0,
            dropout: float = 0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        self.positional = PositionalEmbedding(
            embedding_dim,
            max_seq_length,
            dropout
        )

        self.layer_norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens

        Args:
            x: Input token ids [batch_size, seq_len]

        Returns:
            Embedded tensor [batch_size, seq_len, embedding_dim]
        """
        x = self.token_embedding(x)
        x = self.positional(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class HyperbolicEmbedding(nn.Module):
    """
    Embedding module operating in hyperbolic space
    """

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            max_seq_length: int = 512,
            padding_idx: int = 0,
            dropout: float = 0.1,
            hyperbolic_c: float = 1.0
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        self.positional = PositionalEmbedding(
            embedding_dim,
            max_seq_length,
            dropout
        )

        self.hyperbolic = PoincareBallLayer(
            embedding_dim,
            embedding_dim,
            c=hyperbolic_c
        )

        self.layer_norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens in hyperbolic space
        """
        x = self.token_embedding(x)
        x = self.positional(x)
        x = self.hyperbolic(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
