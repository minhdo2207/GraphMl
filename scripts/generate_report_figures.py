"""Generate all publication-quality figures for the Graph ML report.

Produces 5 figures in figures/:
  1. model_comparison_curves.png  – train loss + val acc curves for all 5 models
  2. accuracy_bar_chart.png       – horizontal bar chart, test acc ± std
  3. ablation_bar_chart.png       – ablation study bar chart
  4. confusion_matrix.png         – proposed model predictions on Cora test set
  5. val_vs_test_scatter.png      – val vs test scatter for all 25 runs (5 models × 5 seeds)

Usage:
  source .venv/bin/activate && python scripts/generate_report_figures.py
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F

from src.training import run_seeds, run_single
from src.training.trainer import train
from src.utils.config import load_config
from src.data_loader import get_data
from src.models import build_model

# ── Global style ──────────────────────────────────────────────────────────────
DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 12
LEGEND_SIZE = 10

# Consistent color palette across all figures
COLORS = {
    "mlp": "#e74c3c",       # red
    "gcn": "#3498db",       # blue
    "graphsage": "#2ecc71", # green
    "gat": "#f39c12",       # orange
    "proposed": "#9b59b6",  # purple
}
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "proposed": "Proposed (ours)",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Fallback if style not available
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3, linestyle="--")


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [saved] {name} ({size_kb:.0f} KB)")


# ── Figure 1: Model comparison curves ────────────────────────────────────────

def fig1_comparison_curves(all_runs: dict[str, list]):
    """Plot train loss + val acc curves. Uses seed 0 from each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in ["mlp", "gcn", "graphsage", "gat", "proposed"]:
        result = all_runs[model_name][0]  # seed 0
        color = COLORS[model_name]
        label = MODEL_LABELS[model_name]
        epochs = range(1, len(result.train_loss) + 1)

        axes[0].plot(epochs, result.train_loss, label=label, color=color, linewidth=1.5)
        axes[1].plot(epochs, result.val_acc, label=label, color=color, linewidth=1.5)

    _style_ax(axes[0], "Training Loss", "Epoch", "Loss")
    axes[0].legend(fontsize=LEGEND_SIZE, loc="upper right")

    _style_ax(axes[1], "Validation Accuracy", "Epoch", "Accuracy")
    axes[1].legend(fontsize=LEGEND_SIZE, loc="lower right")
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Model Comparison: Training Dynamics", fontsize=TITLE_SIZE + 1, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "model_comparison_curves.png")


# ── Figure 2: Accuracy bar chart ─────────────────────────────────────────────

def fig2_accuracy_bar_chart():
    """Horizontal bar chart comparing test accuracy of all 5 models."""
    # Read from CSV files
    data = {}
    # Baselines
    with open("results/baselines.csv") as f:
        for row in csv.DictReader(f):
            data[row["model"]] = (float(row["test_acc_mean"]), float(row["test_acc_std"]))
    # Comparison (may overwrite gcn with same values)
    with open("results/comparison.csv") as f:
        for row in csv.DictReader(f):
            data[row["model"]] = (float(row["test_acc_mean"]), float(row["test_acc_std"]))

    # Sort by accuracy
    sorted_models = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
    names = [MODEL_LABELS.get(m, m) for m, _ in sorted_models]
    means = [v[0] for _, v in sorted_models]
    stds = [v[1] for _, v in sorted_models]
    model_keys = [m for m, _ in sorted_models]
    colors = [COLORS.get(m, "#333333") for m in model_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, height=0.6, color=colors,
                   edgecolor="white", linewidth=1.2, capsize=5,
                   error_kw={"linewidth": 1.2, "capthick": 1.2})

    # Highlight proposed model bar with a thicker border
    for i, mk in enumerate(model_keys):
        if mk == "proposed":
            bars[i].set_edgecolor("#2c3e50")
            bars[i].set_linewidth(2.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=LABEL_SIZE)
    ax.invert_yaxis()

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f"{m:.4f} ± {s:.4f}", va="center",
                fontsize=LEGEND_SIZE, fontweight="bold")

    _style_ax(ax, "Test Accuracy Comparison (Cora)", "Test Accuracy", "")
    ax.set_xlim(0.5, 0.9)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "accuracy_bar_chart.png")


# ── Figure 3: Ablation bar chart ─────────────────────────────────────────────

def fig3_ablation_bar_chart():
    """Bar chart for ablation study with 4 variants."""
    with open("results/ablation.csv") as f:
        rows = list(csv.DictReader(f))

    variant_labels = [
        "GAT baseline\n(no res, no dropedge)",
        "+ residual\nskip",
        "+ residual\n+ dropedge (full)",
        "full,\nheads=4",
    ]
    colors_abl = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12"]

    means = [float(r["test_acc_mean"]) for r in rows]
    stds = [float(r["test_acc_std"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(len(variant_labels))
    bars = ax.bar(x_pos, means, yerr=stds, width=0.6, color=colors_abl,
                  edgecolor="white", linewidth=1.2, capsize=6,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})

    # Highlight the "full" model (index 2)
    bars[2].set_edgecolor("#2c3e50")
    bars[2].set_linewidth(2.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_labels, fontsize=LEGEND_SIZE)

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.003, f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom",
                fontsize=LEGEND_SIZE - 1, fontweight="bold")

    _style_ax(ax, "Ablation Study: Test Accuracy", "", "Test Accuracy")
    ax.set_ylim(0.78, 0.86)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "ablation_bar_chart.png")


# ── Figure 4: Confusion matrix ───────────────────────────────────────────────

def fig4_confusion_matrix(cfg):
    """Confusion matrix for the proposed model on Cora test set."""
    data, in_dim, out_dim = get_data(name="Cora", root="data", normalize_features=True)
    model = build_model("proposed", in_dim, out_dim, cfg)
    train(model, data, cfg, verbose=False)  # train inline

    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()

    num_classes = out_dim
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    # Normalize per row for better visualization
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Class labels for Cora
    class_names = [f"Class {i}" for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, fontsize=LEGEND_SIZE, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=LEGEND_SIZE)
    ax.set_xlabel("Predicted", fontsize=LABEL_SIZE)
    ax.set_ylabel("True", fontsize=LABEL_SIZE)
    ax.set_title("Confusion Matrix — Proposed Model (Cora Test Set)",
                 fontsize=TITLE_SIZE, fontweight="bold")

    # Annotate cells with raw counts
    thresh = cm_norm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=LEGEND_SIZE, fontweight="bold", color=text_color)

    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


# ── Figure 5: Val vs Test scatter ────────────────────────────────────────────

def fig5_val_vs_test_scatter(all_runs: dict[str, list]):
    """Scatter plot of best val accuracy vs test accuracy for all 25 runs."""
    fig, ax = plt.subplots(figsize=(8, 7))

    all_val, all_test = [], []
    for model_name in ["mlp", "gcn", "graphsage", "gat", "proposed"]:
        runs = all_runs[model_name]
        val_accs = [r.best_val_acc for r in runs]
        test_accs = [r.test_acc for r in runs]
        all_val.extend(val_accs)
        all_test.extend(test_accs)

        ax.scatter(val_accs, test_accs, c=COLORS[model_name], label=MODEL_LABELS[model_name],
                   s=60, alpha=0.8, edgecolors="white", linewidth=0.8, zorder=3)

    # Diagonal line (val == test)
    lo = min(min(all_val), min(all_test)) - 0.02
    hi = max(max(all_val), max(all_test)) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1.2, label="val = test", zorder=2)

    _style_ax(ax, "Validation vs Test Accuracy (All Seeds)",
              "Best Validation Accuracy", "Test Accuracy")
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    _save(fig, "val_vs_test_scatter.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating publication-quality figures for report")
    print("=" * 60)

    cfg = load_config()

    # Train all 5 models × 5 seeds = 25 runs (reused across figures)
    print("\n[1/5] Running training for all 5 models × 5 seeds...")
    all_runs: dict[str, list] = {}
    for model_name in ["mlp", "gcn", "graphsage", "gat", "proposed"]:
        print(f"  Training {model_name} (5 seeds, {cfg.epochs} epochs)...")
        summary = run_seeds(cfg, model_name=model_name, verbose=False)
        all_runs[model_name] = summary.runs
        print(f"    test_acc = {summary.test_acc_mean:.4f} ± {summary.test_acc_std:.4f}")

    # Figure 1: Model comparison curves (uses seed 0 per model)
    print("\n[2/5] Generating model comparison curves...")
    fig1_comparison_curves(all_runs)

    # Figure 2: Accuracy bar chart (from CSV data)
    print("\n[3/5] Generating accuracy bar chart...")
    fig2_accuracy_bar_chart()

    # Figure 3: Ablation bar chart (from CSV data)
    print("\n[4/5] Generating ablation bar chart...")
    fig3_ablation_bar_chart()

    # Figure 4: Confusion matrix (train proposed model inline)
    print("\n[5/5] Generating confusion matrix...")
    fig4_confusion_matrix(cfg)

    # Figure 5: Val vs test scatter (uses all 25 runs)
    print("\n[6/6] Generating val vs test scatter...")
    fig5_val_vs_test_scatter(all_runs)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

    # List generated files with sizes
    print("\nGenerated figures:")
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith(".png"):
            fpath = os.path.join(OUT_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname:40s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
