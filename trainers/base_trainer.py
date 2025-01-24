import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import os
import time
from tqdm.auto import tqdm

from ..utils.logging import Logger
from ..utils.metrics import MetricTracker
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class BaseTrainer:
    """Base class for all trainers"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}

        # Setup logging and metrics
        self.logger = Logger(self.config.get("exp_name", "training"))
        self.metric_tracker = MetricTracker()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

    def train(self, epochs: int):
        """Main training loop"""
        for epoch in range(epochs):
            self.current_epoch = epoch
            self._train_epoch()

            if self.val_loader is not None:
                metrics = self._validate()
                # Save checkpoint if improved
                if metrics['val_loss'] < self.best_metric:
                    self.best_metric = metrics['val_loss']
                    self._save_checkpoint('best.pt')

            # Save regular checkpoint
            self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # Forward pass and loss computation
            loss, metrics = self._train_step(batch)

            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            epoch_metrics.append(metrics)
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix(loss=loss.item())

            # Log metrics
            self.logger.log_metrics(metrics, self.global_step)

        # Compute epoch metrics
        epoch_metrics = self._aggregate_metrics(epoch_metrics)
        self.logger.log_metrics(
            {f"epoch_{k}": v for k, v in epoch_metrics.items()},
            self.current_epoch
        )

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step"""
        raise NotImplementedError

    def _validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                metrics = self._validation_step(batch)
                val_metrics.append(metrics)

        # Aggregate validation metrics
        val_metrics = self._aggregate_metrics(val_metrics)
        self.logger.log_metrics(
            {f"val_{k}": v for k, v in val_metrics.items()},
            self.global_step
        )
        return val_metrics

    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step"""
        raise NotImplementedError

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple steps"""
        aggregated = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = sum(values) / len(values)
        return aggregated

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(state, save_path)

    def _load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        path = os.path.join(self.config['checkpoint_dir'], filename)
        if not os.path.exists(path):
            self.logger.info(f"Checkpoint {path} does not exist.")
            return

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        if self.scheduler and state['scheduler_state']:
            self.scheduler.load_state_dict(state['scheduler_state'])

        self.current_epoch = state['epoch']
        self.global_step = state['global_step']
        self.best_metric = state['best_metric']
