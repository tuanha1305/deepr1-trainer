from typing import Dict, Type

# Import model architectures
from .architecture.rl_model import SmallRLModel, SmallerRLModel
from .architecture.embeddings import TextEmbedding, HyperbolicEmbedding

# Import model components
from .components.hyperbolic import HyperbolicLayer, PoincareBallLayer
from .components.attention import MultiHeadAttention
from .components.embeddings import PositionalEmbedding
from .components.layers import FeedForward, LayerNorm

# Registry for available models
MODEL_REGISTRY: Dict[str, Type] = {
    # Main models
    'small_rl': SmallRLModel,
    'smaller_rl': SmallerRLModel,

    # Embeddings
    'text_embedding': TextEmbedding,
    'hyperbolic_embedding': HyperbolicEmbedding,
}

# Registry for model components
COMPONENT_REGISTRY: Dict[str, Type] = {
    # Layers
    'hyperbolic_layer': HyperbolicLayer,
    'poincare_ball_layer': PoincareBallLayer,
    'attention': MultiHeadAttention,
    'positional_embedding': PositionalEmbedding,
    'feed_forward': FeedForward,
    'layer_norm': LayerNorm,
}


def get_model(model_name: str, **kwargs):
    """
    Get model by name from registry

    Args:
        model_name: Name of the model to instantiate
        **kwargs: Arguments to pass to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def get_component(component_name: str, **kwargs):
    """
    Get component by name from registry

    Args:
        component_name: Name of the component to instantiate
        **kwargs: Arguments to pass to component constructor

    Returns:
        Instantiated component

    Raises:
        ValueError: If component_name is not in registry
    """
    if component_name not in COMPONENT_REGISTRY:
        raise ValueError(
            f"Component {component_name} not found. Available components: {list(COMPONENT_REGISTRY.keys())}"
        )
    return COMPONENT_REGISTRY[component_name](**kwargs)


def register_model(name: str, model_class: Type):
    """
    Register a new model class

    Args:
        name: Name for the model
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class


def register_component(name: str, component_class: Type):
    """
    Register a new component class

    Args:
        name: Name for the component
        component_class: Component class to register
    """
    COMPONENT_REGISTRY[name] = component_class


# Version info
__version__ = '1.0.0'

# Export all public interfaces
__all__ = [
    # Main functions
    'get_model',
    'get_component',
    'register_model',
    'register_component',

    # Models
    'SmallRLModel',
    'SmallerRLModel',
    'TextEmbedding',
    'HyperbolicEmbedding',

    # Components
    'HyperbolicLayer',
    'PoincareBallLayer',
    'MultiHeadAttention',
    'PositionalEmbedding',
    'FeedForward',
    'LayerNorm',

    # Registries
    'MODEL_REGISTRY',
    'COMPONENT_REGISTRY',
]