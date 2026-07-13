"""High-level runners: train a model across multiple seeds and aggregate.

Experiment scripts call ``run_seeds`` to get mean ± std test accuracy, which is
what the report tables use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..data import get_data
from ..models import build_model
from ..utils.config import Config
from ..utils.seed import set_seed
from .trainer import train, TrainResult


@dataclass
class SeedSummary:
    model: str
    test_acc_mean: float
    test_acc_std: float
    val_acc_mean: float
    test_accs: List[float]
    runs: List[TrainResult]

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "test_acc_mean": round(self.test_acc_mean, 4),
            "test_acc_std": round(self.test_acc_std, 4),
            "val_acc_mean": round(self.val_acc_mean, 4),
            "n_seeds": len(self.test_accs),
        }


def run_single(cfg: Config, model_name: str | None = None, seed: int | None = None,
               verbose: bool = False) -> TrainResult:
    """Load data, build the model, train once for a single seed."""
    model_name = model_name or cfg.model
    seed = cfg.seed if seed is None else seed
    set_seed(seed)

    data, in_dim, out_dim = get_data(
        name=cfg.dataset, root=cfg.data_root,
        normalize_features=cfg.normalize_features,
    )
    model = build_model(model_name, in_dim, out_dim, cfg)
    return train(model, data, cfg, verbose=verbose)


def run_seeds(cfg: Config, model_name: str | None = None,
              verbose: bool = False) -> SeedSummary:
    """Train a model over ``cfg.seeds`` seeds and summarise test accuracy."""
    model_name = model_name or cfg.model
    runs: List[TrainResult] = []
    for i in range(cfg.seeds):
        res = run_single(cfg, model_name=model_name, seed=cfg.seed + i, verbose=verbose)
        runs.append(res)
        if verbose:
            print(f"[{model_name}] seed {cfg.seed + i}: "
                  f"test={res.test_acc:.4f} (best val={res.best_val_acc:.4f})")

    test_accs = [r.test_acc for r in runs]
    val_accs = [r.best_val_acc for r in runs]
    return SeedSummary(
        model=model_name,
        test_acc_mean=float(np.mean(test_accs)),
        test_acc_std=float(np.std(test_accs)),
        val_acc_mean=float(np.mean(val_accs)),
        test_accs=test_accs,
        runs=runs,
    )
