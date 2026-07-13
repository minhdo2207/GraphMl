"""Plotting and results-table helpers (Person 5).

Uses a non-interactive matplotlib backend so figures can be generated in Colab or
headless CI without a display.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(histories: Mapping[str, "object"], out_path: str,
                title: str = "Training curves") -> str:
    """Plot validation-accuracy and training-loss curves for several models.

    ``histories`` maps a model name to any object exposing ``val_acc`` and
    ``train_loss`` lists (e.g. a ``TrainResult``).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, hist in histories.items():
        ax_loss.plot(hist.train_loss, label=name)
        ax_acc.plot(hist.val_acc, label=name)

    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.legend()

    ax_acc.set_title("Validation accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_results_table(rows: Sequence[Mapping[str, object]], out_path: str) -> str:
    """Write a list of result dicts to a CSV file."""
    if not rows:
        raise ValueError("No rows to write.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames: List[str] = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def print_table(rows: Sequence[Mapping[str, object]]) -> None:
    """Print a simple aligned table to stdout."""
    if not rows:
        print("(no results)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
