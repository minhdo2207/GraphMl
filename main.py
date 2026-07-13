"""Train a single model on Cora.

Usage:
    python main.py --model gcn
    python main.py --model proposed --seeds 5 --drop-edge 0.2
"""
from __future__ import annotations

import argparse

from src.models import MODEL_NAMES
from src.training import run_seeds
from src.utils import load_config
from src.utils.plots import print_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GNN/MLP on Cora.")
    p.add_argument("--model", choices=MODEL_NAMES, default="gcn")
    p.add_argument("--config", default=None, help="optional YAML config path")
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None, dest="hidden_dim")
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--drop-edge", type=float, default=None, dest="drop_edge")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "verbose") and v is not None}
    cfg = load_config(args.config, **overrides)

    print(f"Training '{cfg.model}' on {cfg.dataset} over {cfg.seeds} seed(s)...")
    summary = run_seeds(cfg, model_name=cfg.model, verbose=args.verbose)
    print_table([summary.as_row()])


if __name__ == "__main__":
    main()
