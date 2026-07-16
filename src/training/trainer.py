"""Single-run training loop with validation-based model selection.

The training recipe is shared by every model so comparisons are fair: same
optimizer (Adam), same cross-entropy loss on the training nodes, and the reported
test accuracy is taken at the epoch of best *validation* accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import copy
import torch
import torch.nn.functional as F

from ..utils.config import Config
from ..utils.seed import resolve_device
from .metrics import accuracy


@dataclass
class TrainResult:
    best_val_acc: float
    test_acc: float                     # test acc at the best-val epoch
    best_epoch: int
    train_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    test_acc_curve: List[float] = field(default_factory=list)


def train(model: torch.nn.Module, data, cfg: Config, verbose: bool = False) -> TrainResult:
    """Train ``model`` on ``data`` and return curves + best-val test accuracy."""
    device = resolve_device(cfg.device)
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    result = TrainResult(best_val_acc=0.0, test_acc=0.0, best_epoch=-1)
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        # --- train step ---
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- eval step ---
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        result.train_loss.append(float(loss.item()))
        result.val_acc.append(val_acc)
        result.test_acc_curve.append(test_acc)

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.test_acc = test_acc
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} "
                  f"| val {val_acc:.4f} | test {test_acc:.4f}")

        if cfg.patience and epochs_no_improve >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {cfg.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best-val weights
    return result
