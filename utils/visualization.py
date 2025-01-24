import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import numpy as np


def plot_training_curves(
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
):
    """Plot training metrics over time"""
    plt.figure(figsize=(12, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)

    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_rewards(
        rewards: List[float],
        window_size: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
):
    """Plot reward distribution and moving average"""
    plt.figure(figsize=(12, 8))

    # Plot 1: Reward Distribution
    plt.subplot(2, 1, 1)
    sns.histplot(rewards, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Reward Value')
    plt.ylabel('Count')

    # Plot 2: Moving Average
    plt.subplot(2, 1, 2)
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
    plt.title('Reward Moving Average')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()