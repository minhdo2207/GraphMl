# Residual Multi-Head Mixing GAT — Node Classification on Cora

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minhdo2207/GraphMl/blob/main/notebooks/demo.ipynb)

Semi-supervised **node classification** on the **Cora** citation network. Compares a
feature-only **MLP** against **GCN**, **GraphSAGE**, **GAT**, and a proposed
**Residual Multi-Head Mixing GAT** (multi-head attention + input-feature skip
connection + DropEdge).

---

## 🚀 Chạy nhanh

### Cách 1 — Colab (không cần cài gì)
Bấm badge **Open In Colab** ở trên → `Runtime > Run all`. Notebook tự clone repo, cài
PyTorch Geometric và chạy full pipeline (dataset stats → baselines → comparison →
ablation → vẽ curve).

### Cách 2 — Local
```bash
git clone https://github.com/minhdo2207/GraphMl.git
cd GraphMl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/dataset_stats.py        # thống kê dataset
python main.py --model proposed        # train 1 model bất kỳ
```
> PyG wheel phụ thuộc torch/CUDA của máy. Nếu `pip install` lỗi ở `torch-geometric`,
> theo hướng dẫn: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Chạy các experiment

```bash
python experiments/run_baselines.py    # MLP vs GCN
python experiments/run_comparison.py   # GCN vs GraphSAGE vs GAT
python experiments/run_ablation.py     # ablation của proposed model
```
Mỗi experiment tự ghi metrics vào `results/*.csv` và curve vào `figures/*.png`.

## `main.py` — các cờ hay dùng

```bash
python main.py --model gcn                       # gcn | mlp | graphsage | gat | proposed
python main.py --model proposed --drop-edge 0.3  # override DropEdge
python main.py --model gat --seeds 10            # nhiều seed hơn
python main.py --model proposed --device cpu     # ép chạy CPU
```
Tất cả hyperparameter mặc định nằm ở [configs/default.yaml](configs/default.yaml).

---

## Proposed model

Cho input features `H0 = X`:

```
H1 = ELU( GATConv_multi(H0, E) )        # K heads, concat  -> one-hop
H2 = GATConv_single(H1, E)              # single head       -> two-hop
Z  = H2 + H0 @ W_skip                   # residual skip từ raw features
```

DropEdge (xóa ngẫu nhiên một phần cạnh) áp dụng cho `E` lúc train như structural
augmentation + regularization. Cờ `residual` và `drop_edge` tách riêng để ablation
bật/tắt từng đóng góp độc lập.

---

## Cấu trúc repo

```
GraphMl/
├── main.py                        # train 1 model: python main.py --model <name>
├── configs/default.yaml           # toàn bộ hyperparameter
├── notebooks/demo.ipynb           # Colab chạy end-to-end (Run all)
├── src/
│   ├── data_loader/data_loader.py # Cora loader (Planetoid) + statistics
│   ├── models/                    # mlp, gcn, graphsage, gat, proposed (+ registry)
│   ├── training/                  # trainer, metrics, multi-seed runner
│   └── utils/                     # config, seeding, plotting
├── experiments/
│   ├── run_baselines.py           # MLP vs GCN
│   ├── run_comparison.py          # GCN vs GraphSAGE vs GAT
│   └── run_ablation.py            # proposed model ablation
├── scripts/dataset_stats.py       # dataset statistics
├── results/                       # metrics CSVs (generated)
├── figures/                       # curves / plots (generated)
└── report/                        # report (NeurIPS LaTeX) + slides
```

> Thêm model mới: tạo file trong `src/models/`, khai báo ở
> [src/models/registry.py](src/models/registry.py). Không phải sửa training/experiment script.

---

## Reproducibility

Mọi experiment seed Python/NumPy/PyTorch ([src/utils/seed.py](src/utils/seed.py)) và
report **mean ± std** trên 5 seeds. Model selection theo best **validation** accuracy;
số báo cáo là **test** accuracy tại epoch đó.
