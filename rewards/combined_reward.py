from typing import Optional, Dict, Any, List
import torch
from .base_reward import BaseReward
from .accuracy_reward import AccuracyReward
from .format_reward import FormatReward


class CombinedReward(BaseReward):
    """Combine multiple reward functions with weights"""

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            reward_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)

        # Initialize component rewards
        self.accuracy_reward = AccuracyReward(config)
        self.format_reward = FormatReward(config)

        # Set reward weights
        self.reward_weights = reward_weights or {
            'accuracy': 0.6,
            'format': 0.4
        }

    def __call__(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            response: str,
            **kwargs
    ) -> float:
        """
        Calculate combined reward

        Args:
            output: Model output embeddings
            target: Target embeddings
            response: Generated text response
        """
        # Calculate individual rewards
        accuracy_reward = self.accuracy_reward(output=output, target=target)
        format_reward = self.format_reward(response=response)

        # Combine rewards using weights
        combined_reward = (
                self.reward_weights['accuracy'] * accuracy_reward +
                self.reward_weights['format'] * format_reward
        )

        # Additional reward shaping
        if kwargs.get('use_reward_shaping', False):
            combined_reward = self._shape_reward(combined_reward, **kwargs)

        return self.clip_reward(combined_reward)

    def _shape_reward(
            self,
            reward: float,
            step: int = 0,
            episode: int = 0,
            **kwargs
    ) -> float:
        """Apply reward shaping"""
        # Decay reward over time
        decay_rate = 0.999
        reward *= (decay_rate ** episode)

        # Add exploration bonus
        if step < 1000:  # Early training steps
            exploration_bonus = 0.1 * (1000 - step) / 1000
            reward += exploration_bonus

        return reward
