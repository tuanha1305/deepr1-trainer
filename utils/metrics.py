from typing import Dict, List, Optional
import numpy as np
import torch
from collections import defaultdict


class MetricTracker:
    """Track and compute various training metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)

    def update(self, metric_dict: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metric_dict.items():
            self.metrics[key].append(value)

    def get_mean(self, metric_name: str) -> float:
        """Get mean value for a specific metric"""
        values = self.metrics.get(metric_name, [])
        return np.mean(values) if values else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in self.metrics.items()
        }


def calculate_metrics(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor
) -> Dict[str, float]:
    """Calculate training metrics"""
    with torch.no_grad():
        # Accuracy metrics
        accuracy = torch.mean((outputs.argmax(dim=-1) == targets).float()).item()

        # Loss metrics
        mse_loss = torch.nn.functional.mse_loss(outputs, targets).item()

        # Reward metrics
        mean_reward = torch.mean(rewards).item()
        std_reward = torch.std(rewards).item()

        metrics = {
            'accuracy': accuracy,
            'mse_loss': mse_loss,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }

        return metrics
