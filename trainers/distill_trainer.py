import torch
import torch.nn.functional as F
from typing import Dict, Any

from .base_trainer import BaseTrainer


class DistillationTrainer(BaseTrainer):
    """Trainer for knowledge distillation"""

    def __init__(
            self,
            student_model: torch.nn.Module,
            teacher_model: torch.nn.Module,
            *args,
            temperature: float = 2.0,
            alpha: float = 0.5,
            **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()

        self.temperature = temperature
        self.alpha = alpha

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step with distillation"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch.get('target_ids')
        if target_ids is not None:
            target_ids = target_ids.to(self.device)

        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs, _ = self.teacher_model(input_ids, attention_mask)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=-1)

        # Get student predictions
        student_outputs, _ = self.model(input_ids, attention_mask)
        student_logits = student_outputs / self.temperature
        student_probs = F.softmax(student_logits, dim=-1)

        # Compute distillation loss
        distill_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Compute standard cross-entropy loss if targets available
        if target_ids is not None:
            ce_loss = F.cross_entropy(student_outputs, target_ids)
            loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
        else:
            loss = distill_loss

        metrics = {
            'loss': loss.item(),
            'distill_loss': distill_loss.item(),
            'ce_loss': ce_loss.item() if target_ids is not None else 0
        }

        return loss, metrics

    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step for distillation"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch.get('target_ids')
        if target_ids is not None:
            target_ids = target_ids.to(self.device)

        # Get predictions
        with torch.no_grad():
            teacher_outputs, _ = self.teacher_model(input_ids, attention_mask)
            student_outputs, _ = self.model(input_ids, attention_mask)

        # Compute validation metrics
        metrics = {
            'val_teacher_acc': (teacher_outputs.argmax(-1) == target_ids).float().mean().item(),
            'val_student_acc': (student_outputs.argmax(-1) == target_ids).float().mean().item()
        }

        return metrics