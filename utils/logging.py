import logging
import os
import sys
from typing import Optional, Dict, Any
import json
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Custom logger for training process"""

    def __init__(
            self,
            exp_name: str,
            log_dir: str = "logs",
            use_tensorboard: bool = True,
            use_wandb: bool = False
    ):
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup file logger
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, f"{exp_name}.log"))
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=exp_name, dir=self.log_dir)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all enabled platforms"""
        # Log to file
        self.logger.info(f"Step {step} - Metrics: {json.dumps(metrics, indent=2)}")

        # Log to TensorBoard
        if self.use_tensorboard:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(metric_name, metric_value, step)

        # Log to W&B
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_model_graph(self, model: torch.nn.Module, input_size: tuple):
        """Log model architecture"""
        if self.use_tensorboard:
            dummy_input = torch.zeros(input_size)
            self.writer.add_graph(model, dummy_input)

    def close(self):
        """Close all handlers"""
        if self.use_tensorboard:
            self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()


def setup_logging(exp_name: str) -> Logger:
    """Setup logging for the experiment"""
    return Logger(exp_name)