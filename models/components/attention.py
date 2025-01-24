import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            bias: bool = True
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply masks if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Attention weights and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        output = self.out_proj(attn_output)

        return output, attn_weights
    