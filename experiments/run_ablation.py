"""Experiment: proposed-model ablation.

Ablates the two contributions of the Residual Multi-Head Mixing GAT:
  * residual skip connection  (H0 @ W_skip) on / off
  * DropEdge                  on / off
and sweeps the number of attention heads.

Writes results/ablation.csv and figures/ablation_curves.png.

Usage:
    python experiments/run_ablation.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path)

from src.training import run_seeds
from src.utils import load_config
from src.utils.plots import plot_curves, print_table, save_results_table


def main() -> None:
    base = load_config()

    # Each variant overrides a field of the proposed model.
    variants = {
        "GAT (no residual, no dropedge)": base.update(model="proposed", residual=False, drop_edge=0.0),
        "+ residual skip":               base.update(model="proposed", residual=True,  drop_edge=0.0),
        "+ residual + dropedge (full)":  base.update(model="proposed", residual=True,  drop_edge=0.2),
        "full, heads=4":                 base.update(model="proposed", residual=True,  drop_edge=0.2, heads=4),
    }

    rows = []
    best_runs = {}
    for label, cfg in variants.items():
        print(f"\n=== {label} ===")
        summary = run_seeds(cfg, model_name="proposed", verbose=True)
        row = summary.as_row()
        row["variant"] = label
        row["residual"] = cfg.residual
        row["drop_edge"] = cfg.drop_edge
        row["heads"] = cfg.heads
        rows.append(row)
        best_runs[label] = summary.runs[0]

    print("\n=== Ablation: Residual Multi-Head Mixing GAT ===")
    print_table(rows)
    save_results_table(rows, "results/ablation.csv")
    plot_curves(best_runs, "figures/ablation_curves.png",
                title="Ablation of Residual Multi-Head Mixing GAT on Cora")
    print("\nSaved: results/ablation.csv, figures/ablation_curves.png")


if __name__ == "__main__":
    main()
