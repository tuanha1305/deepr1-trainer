from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # Model dimensions
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 2

    # Hyperbolic space parameters
    use_hyperbolic: bool = True
    hyperbolic_c: float = 1.0  # Curvature of hyperbolic space

    # Activation and dropout
    hidden_activation: str = "relu"
    dropout: float = 0.1

    # Model type selection
    model_type: str = "SmallRLModel"  # or "SmallerRLModel"

    # Advanced options
    use_layer_norm: bool = True
    use_residual: bool = True
    embedding_dim: Optional[int] = None

    def __post_init__(self):
        if self.embedding_dim is None:
            self.embedding_dim = self.hidden_dim