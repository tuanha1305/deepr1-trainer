from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os


@dataclass
class DataConfig:
    """Configuration for data processing and loading"""
    # Data paths
    train_data_path: str = "data/train"
    eval_data_path: str = "data/eval"
    test_data_path: str = "data/test"

    # Data processing
    max_seq_length: int = 512
    vocab_size: int = 50000
    pad_token_id: int = 0
    eos_token_id: int = 2

    # Tokenizer settings
    tokenizer_name: str = "gpt2"
    tokenizer_kwargs: Dict[str, Any] = None

    # Dataset settings
    dataset_name: str = "custom"  # or "huggingface"
    dataset_config: Optional[Dict[str, Any]] = None

    # Data loading
    num_workers: int = os.cpu_count()
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Data augmentation
    use_augmentation: bool = False
    augmentation_methods: List[str] = None

    def __post_init__(self):
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}
        if self.dataset_config is None:
            self.dataset_config = {}
        if self.augmentation_methods is None:
            self.augmentation_methods = []
