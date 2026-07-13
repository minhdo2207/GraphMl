"""Experiment: MLP vs GCN baseline (Person 2).

Answers "does the graph help?" — compares a feature-only MLP against a 2-layer GCN.
Writes results/baselines.csv and figures/baselines_curves.png.

Usage:
    python experiments/run_baselines.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path)

from src.training import run_seeds
from src.utils import load_config
from src.utils.plots import plot_curves, print_table, save_results_table

MODELS = ["mlp", "gcn"]


def main() -> None:
    cfg = load_config()
    rows = []
    best_runs = {}

    for model in MODELS:
        print(f"\n=== {model.upper()} ===")
        summary = run_seeds(cfg, model_name=model, verbose=True)
        rows.append(summary.as_row())
        best_runs[model] = summary.runs[0]  # curves from the first seed

    print("\n=== Baseline results (MLP vs GCN) ===")
    print_table(rows)
    save_results_table(rows, "results/baselines.csv")
    plot_curves(best_runs, "figures/baselines_curves.png",
                title="MLP vs GCN on Cora")
    print("\nSaved: results/baselines.csv, figures/baselines_curves.png")


if __name__ == "__main__":
    main()
