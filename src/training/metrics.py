"""Evaluation metrics for node classification."""
from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Classification accuracy over the nodes selected by ``mask``."""
    pred = logits[mask].argmax(dim=1)
    correct = (pred == y[mask]).sum().item()
    total = int(mask.sum())
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate(model: torch.nn.Module, data) -> Dict[str, float]:
    """Return train/val/test accuracy for a trained model in eval mode."""
    model.eval()
    logits = model(data.x, data.edge_index)
    return {
        "train_acc": accuracy(logits, data.y, data.train_mask),
        "val_acc": accuracy(logits, data.y, data.val_mask),
        "test_acc": accuracy(logits, data.y, data.test_mask),
    }
