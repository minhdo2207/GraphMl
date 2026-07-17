"""Experiment: architecture comparison — GCN vs GraphSAGE vs GAT.

Writes results/comparison.csv and figures/comparison_curves.png.

Usage:
    python experiments/run_comparison.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path)

from src.training import run_seeds
from src.utils import load_config
from src.utils.plots import plot_curves, print_table, save_results_table

MODELS = ["gcn", "graphsage", "gat"]


def main() -> None:
    cfg = load_config()
    rows = []
    best_runs = {}

    for model in MODELS:
        print(f"\n=== {model.upper()} ===")
        summary = run_seeds(cfg, model_name=model, verbose=True)
        rows.append(summary.as_row())
        best_runs[model] = summary.runs[0]

    print("\n=== Architecture comparison (GCN / GraphSAGE / GAT) ===")
    print_table(rows)
    save_results_table(rows, "results/comparison.csv")
    plot_curves(best_runs, "figures/comparison_curves.png",
                title="GCN vs GraphSAGE vs GAT on Cora")
    print("\nSaved: results/comparison.csv, figures/comparison_curves.png")


if __name__ == "__main__":
    main()
