from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseReward(ABC):
    """Base class for all reward functions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.eps = 1e-8  # Small value for numerical stability

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """Calculate reward value"""
        pass

    def normalize_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Normalize reward to specified range"""
        return (max_val - min_val) * (reward - min_val) / (max_val - min_val + self.eps) + min_val

    def clip_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clip reward to specified range"""
        return max(min_val, min(max_val, reward))

    def decay_reward(self, reward: float, decay_factor: float = 0.99) -> float:
        """Apply decay to reward"""
        return reward * decay_factor
