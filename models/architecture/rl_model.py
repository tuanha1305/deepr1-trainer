import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from ..components.hyperbolic import PoincareBallLayer
from ..components.attention import MultiHeadAttention
from ..components.embeddings import PositionalEmbedding
from ..components.layers import FeedForward, LayerNorm


class SmallRLModel(nn.Module):
    """
    Main RL model with hyperbolic embeddings and attention
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int = 8,
            num_layers: int = 2,
            dropout: float = 0.1,
            max_seq_length: int = 512,
            use_hyperbolic: bool = True,
            hyperbolic_c: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_hyperbolic = use_hyperbolic

        # Embeddings
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional = PositionalEmbedding(hidden_dim, max_seq_length, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim,
                num_heads,
                dropout=dropout,
                use_hyperbolic=use_hyperbolic,
                hyperbolic_c=hyperbolic_c
            ) for _ in range(num_layers)
        ])

        # Output layers
        if use_hyperbolic:
            self.output_layer = PoincareBallLayer(hidden_dim, output_dim, c=hyperbolic_c)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.layer_norm = LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            output: Output tensor [batch_size, seq_len, output_dim]
            attentions: Dictionary of attention weights from each layer
        """
        # Input embedding
        x = self.embedding(x)
        x = self.positional(x)
        x = self.dropout(x)

        # Store attention weights
        attentions = {}

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, mask)
            attentions[f'layer_{i}'] = attn

        # Output projection
        x = self.output_layer(x)
        x = self.layer_norm(x)

        return x, attentions

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation"""
        x = self.embedding(x)
        x = self.positional(x)

        for layer in self.layers:
            x, _ = layer(x)

        return x

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation to output"""
        x = self.output_layer(h)
        x = self.layer_norm(x)
        return x


class TransformerLayer(nn.Module):
    """
    Single transformer layer with optional hyperbolic operations
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            use_hyperbolic: bool = True,
            hyperbolic_c: float = 1.0
    ):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(hidden_dim, dropout=dropout)

        # Layer normalization
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

        # Optional hyperbolic layer
        if use_hyperbolic:
            self.hyperbolic = PoincareBallLayer(hidden_dim, hidden_dim, c=hyperbolic_c)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer layer

        Returns:
            output: Transformed tensor
            attention_weights: Attention weights
        """
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        # Optional hyperbolic transformation
        if self.use_hyperbolic:
            x = self.hyperbolic(x)

        return x, attn_weights


class SmallerRLModel(nn.Module):
    """
    Lightweight version of the RL model for distillation
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 1,
            dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Simple feed-forward architecture
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.layers = nn.Sequential(*layers[:-2])  # Remove last ReLU and dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.layers(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate representation"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) // 2:  # Middle layer
                return x
        return x
