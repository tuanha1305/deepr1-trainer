import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .base_reward import BaseReward


class AccuracyReward(BaseReward):
    """Reward based on output accuracy and similarity metrics"""

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            similarity_threshold: float = 0.8,
            use_cosine: bool = True
    ):
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.use_cosine = use_cosine

    def __call__(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            **kwargs
    ) -> float:
        """
        Calculate accuracy-based reward

        Args:
            output: Model output embeddings
            target: Target embeddings
        """
        with torch.no_grad():
            if self.use_cosine:
                # Cosine similarity reward
                similarity = F.cosine_similarity(output, target, dim=-1)
                reward = similarity.mean().item()

                # Bonus for high similarity
                if reward > self.similarity_threshold:
                    reward *= 1.2  # Bonus multiplier
            else:
                # L2 distance-based reward
                distance = torch.norm(output - target, dim=-1)
                reward = 1.0 / (1.0 + distance.mean().item())

            # Normalize and clip
            reward = self.normalize_reward(reward)
            reward = self.clip_reward(reward)

            return reward

    def compute_perplexity_reward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Additional reward based on perplexity"""
        with torch.no_grad():
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            perplexity = torch.exp(loss).item()
            reward = 1.0 / (1.0 + perplexity)
            return self.normalize_reward(reward)
