# data/processors/__init__.py
from typing import Dict, Type

from .advanced_text_processor import AdvancedTextProcessor
from .basic_text_processor import BasicTextProcessor
from .text_processor import (
    TextProcessor
)

from .base_processor import BaseProcessor
from .tokenizer_processor import TokenizerProcessor
from .augmentation_processor import DataAugmentationProcessor

# Registry for available processors
PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {
    # Basic processors
    "text": TextProcessor,
    "basic_text": BasicTextProcessor,
    "advanced_text": AdvancedTextProcessor,
    "mlm": MLMTextProcessor,

    # Special processors
    "tokenizer": TokenizerProcessor,
    "augmentation": DataAugmentationProcessor,
}


def get_processor(processor_name: str, **kwargs) -> BaseProcessor:
    """
    Get processor by name from registry

    Args:
        processor_name: Name of the processor to instantiate
        **kwargs: Arguments to pass to processor constructor

    Returns:
        Instantiated processor

    Raises:
        ValueError: If processor_name is not in registry
    """
    if processor_name not in PROCESSOR_REGISTRY:
        raise ValueError(
            f"Processor {processor_name} not found. Available processors: {list(PROCESSOR_REGISTRY.keys())}"
        )
    return PROCESSOR_REGISTRY[processor_name](**kwargs)


def register_processor(name: str, processor_class: Type[BaseProcessor]):
    """
    Register a new processor class

    Args:
        name: Name for the processor
        processor_class: Processor class to register
    """
    PROCESSOR_REGISTRY[name] = processor_class


__all__ = [
    # Main functions
    'get_processor',
    'register_processor',

    # Base classes
    'BaseProcessor',

    # Processor classes
    'TextProcessor',
    'BasicTextProcessor',
    'AdvancedTextProcessor',
    'MLMTextProcessor',
    'TokenizerProcessor',
    'DataAugmentationProcessor',

    # Registry
    'PROCESSOR_REGISTRY',
]