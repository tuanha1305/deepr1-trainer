import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from geoopt import ManifoldTensor, PoincareBall


class HyperbolicLayer(nn.Module):
    """Base layer for hyperbolic operations"""

    def __init__(self, manifold: PoincareBall):
        super().__init__()
        self.manifold = manifold

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform matrix-vector multiplication in hyperbolic space"""
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
        return self.manifold.expmap0(mx)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Addition in hyperbolic space"""
        return self.manifold.mobius_add(x, y)

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map"""
        return self.manifold.expmap(x, v)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map"""
        return self.manifold.logmap(x, y)

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance in hyperbolic space"""
        return self.manifold.dist(x, y)


class PoincareBallLayer(HyperbolicLayer):
    """Neural network layer operating in Poincare ball model"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            c: float = 1.0,
            dropout: float = 0.0,
            bias: bool = True
    ):
        manifold = PoincareBall(c=c)
        super().__init__(manifold)

        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout

        # Trainable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass in hyperbolic space"""
        # Apply weight matrix
        output = self.mobius_matvec(self.weight, input)

        # Apply bias in hyperbolic space if present
        if self.bias is not None:
            output = self.mobius_add(output, self.manifold.expmap0(self.bias))

        # Apply dropout if needed
        if self.training and self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=True)

        return output
