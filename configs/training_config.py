from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Basic training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    max_seq_length: int = 512

    # Optimizer settings
    optimizer: str = "RiemannianAdam"
    optimizer_params: Dict[str, Any] = None
    scheduler: Optional[str] = "cosine"
    warmup_steps: int = 1000

    # RL specific settings
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000

    # Reward weights
    accuracy_reward_weight: float = 1.0
    format_reward_weight: float = 0.5

    # Training devices
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()

    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Distillation parameters
    distillation_temperature: float = 2.0
    teacher_model_path: Optional[str] = None

    def __post_init__(self):
        if self.optimizer_params is None:
            self.optimizer_params = {
                "lr": self.learning_rate,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999)
            }