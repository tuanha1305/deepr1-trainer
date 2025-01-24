import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        save_dir: str,
        filename: Optional[str] = None
):
    """Save model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_path: str
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }