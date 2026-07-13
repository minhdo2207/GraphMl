# Residual Multi-Head Mixing GAT for Semi-Supervised Node Classification on Cora

Course project — Hanoi University of Science and Technology.

We study **semi-supervised node classification** on the **Cora** citation network.
The project compares a feature-only **MLP** against graph neural networks (**GCN**,
**GraphSAGE**, **GAT**) and introduces a **Residual Multi-Head Mixing GAT** that combines
multi-head attention with an input-feature skip connection and DropEdge regularization.

## Proposed model

For input features `H0 = X`:

```
H1 = ELU( GATConv_multi(H0, E) )        # K heads, concat  -> local / one-hop
H2 = GATConv_single(H1, E)              # single head       -> two-hop
Z  = softmax( H2 + H0 @ W_skip )        # residual skip from raw features
```

DropEdge (random edge removal) is applied to `E` during training as structural
augmentation and regularization.

## Repository structure

```
GraphMl/
├── src/
│   ├── data/dataset.py        # Cora loader (Planetoid) + statistics
│   ├── models/                # mlp, gcn, graphsage, gat, proposed (+ registry)
│   ├── training/              # trainer, metrics, multi-seed runner
│   └── utils/                 # config, seeding, plotting
├── experiments/
│   ├── run_baselines.py       # [P2] MLP vs GCN
│   ├── run_comparison.py      # [P3] GCN vs GraphSAGE vs GAT
│   └── run_ablation.py        # [P4] proposed model ablation
├── scripts/dataset_stats.py   # [P2] dataset statistics
├── configs/default.yaml       # default hyperparameters
├── notebooks/                 # Colab / exploratory notebooks
├── results/                   # metrics CSVs (generated)
├── figures/                   # curves / plots (generated)
└── report/                    # LaTeX / Word report + slides
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> PyTorch Geometric wheels depend on your CUDA/CPU + torch version.
> If `pip install -r requirements.txt` fails on `torch-geometric` extras, follow
> https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Quick start

```bash
# Dataset statistics (Person 2)
python scripts/dataset_stats.py

# Train a single model
python main.py --model gcn
python main.py --model proposed

# Experiment suites
python experiments/run_baselines.py     # MLP vs GCN
python experiments/run_comparison.py    # GCN / GraphSAGE / GAT
python experiments/run_ablation.py      # proposed model ablation
```

Each experiment writes a CSV to `results/` and (optionally) curves to `figures/`.

## Team & task split

| # | Member (ID)                     | Responsibility                                   |
|---|---------------------------------|--------------------------------------------------|
| 1 | Do Tuan Minh (20261057M)        | Final integration, report/presentation, QA       |
| 2 | Tran Tien Dung (20252574M)      | Dataset + MLP/GCN baseline                        |
| 3 | Tran Manh Tien (20252762M)      | GraphSAGE + GAT baseline, architecture comparison |
| 4 | Hoang Huy Chien (20261069M)     | Proposed model + ablation                         |
| 5 | Nguyen Tien Duc (20252076M)     | Analysis, figures, slides, discussion             |

## Reproducibility

All experiments seed Python/NumPy/PyTorch (`src/utils/seed.py`) and report
**mean ± std** over multiple seeds. Model selection uses best **validation**
accuracy; the reported number is **test** accuracy at that epoch.
