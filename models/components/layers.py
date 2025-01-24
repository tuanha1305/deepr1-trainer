import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class FeedForward(nn.Module):
    """Feed-forward network with residual connection"""

    def __init__(
            self,
            d_model: int,
            d_ff: int = 2048,
            dropout: float = 0.1,
            activation: Callable = F.relu
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation"""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    """Layer normalization with optional bias"""

    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            bias: bool = True
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        normalized = (x - mean) / (std + self.eps)

        if self.beta is not None:
            return self.gamma * normalized + self.beta
        return self.gamma * normalized
