import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base_trainer import BaseTrainer
from ..rewards import CombinedReward


class RLTrainer(BaseTrainer):
    """Trainer for reinforcement learning"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize reward function
        self.reward_fn = CombinedReward(self.config.get('reward_config'))

        # RL specific parameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 1000)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step with RL"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch.get('target_ids')
        if target_ids is not None:
            target_ids = target_ids.to(self.device)

        # Get model outputs
        outputs, attentions = self.model(input_ids, attention_mask)

        # Compute rewards
        rewards = []
        for i in range(len(outputs)):
            reward = self.reward_fn(
                output=outputs[i],
                target=target_ids[i] if target_ids is not None else None,
                response=self.tokenizer.decode(outputs[i].argmax(-1))
            )
            rewards.append(reward)
        rewards = torch.tensor(rewards, device=self.device)

        # Compute policy loss
        policy_loss = -torch.mean(rewards * outputs.log_prob(target_ids))

        # Add value loss if using critic
        value_loss = 0
        if hasattr(self.model, 'value_head'):
            values = self.model.value_head(outputs.detach())
            value_loss = F.mse_loss(values, rewards)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay
        )

        metrics = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if value_loss else 0,
            'mean_reward': rewards.mean().item(),
            'epsilon': self.epsilon
        }

        return loss, metrics

    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step for RL"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch.get('target_ids')
        if target_ids is not None:
            target_ids = target_ids.to(self.device)

        outputs, _ = self.model(input_ids, attention_mask)

        # Compute validation rewards
        rewards = []
        for i in range(len(outputs)):
            reward = self.reward_fn(
                output=outputs[i],
                target=target_ids[i] if target_ids is not None else None,
                response=self.tokenizer.decode(outputs[i].argmax(-1))
            )
            rewards.append(reward)
        rewards = torch.tensor(rewards, device=self.device)

        return {
            'val_reward': rewards.mean().item()
        }

