# Residual Multi-Head Mixing GAT — Node Classification on Cora

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minhdo2207/GraphMl/blob/main/notebooks/demo.ipynb)

Course project — Hanoi University of Science and Technology.

We study **semi-supervised node classification** on the **Cora** citation network.
The project compares a feature-only **MLP** against graph neural networks (**GCN**,
**GraphSAGE**, **GAT**) and introduces a **Residual Multi-Head Mixing GAT** that combines
multi-head attention with an input-feature skip connection and DropEdge regularization.

> **New here? The fastest way to see everything run is the Colab badge above** →
> `Runtime > Run all`. No local setup needed.

---

## 🚀 Chạy nhanh (2 cách)

### Cách 1 — Colab (khuyến nghị, không cần cài gì)
Bấm badge **Open In Colab** ở trên → `Runtime > Run all`. Notebook sẽ tự clone repo,
cài PyTorch Geometric, chạy full pipeline (dataset stats → baselines → comparison → ablation → vẽ curve).

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

## 👥 Ai làm gì & chạy phần của mình thế nào

| # | Thành viên (MSSV)           | Phụ trách                         | Chạy phần của mình                    |
|---|-----------------------------|-----------------------------------|---------------------------------------|
| 1 | Đỗ Tuấn Minh (20261057M)    | Final integration, report, QA     | tổng hợp `results/*.csv`, `figures/*` |
| 2 | Trần Tiến Dũng (20252574M)  | Dataset + MLP/GCN baseline        | `python experiments/run_baselines.py` |
| 3 | Trần Mạnh Tiến (20252762M)  | GraphSAGE + GAT, so sánh kiến trúc| `python experiments/run_comparison.py`|
| 4 | Hoàng Huy Chiến (20261069M) | Proposed model + ablation         | `python experiments/run_ablation.py`  |
| 5 | Nguyễn Tiến Đức (20252076M) | Phân tích, figure, slide          | curves/CSV sinh ra ở `figures/` & `results/` |

Mỗi experiment tự ghi CSV vào `results/` và ảnh curve vào `figures/`.

**File cần đụng tới cho từng người:**
- **P2 (Dũng):** [src/data/dataset.py](src/data/dataset.py), [src/models/mlp.py](src/models/mlp.py), [src/models/gcn.py](src/models/gcn.py)
- **P3 (Tiến):** [src/models/graphsage.py](src/models/graphsage.py), [src/models/gat.py](src/models/gat.py)
- **P4 (Chiến):** [src/models/proposed.py](src/models/proposed.py) (cờ `residual`, `drop_edge`, `heads`)
- **P5 (Đức):** [src/utils/plots.py](src/utils/plots.py)
- **P1 (Minh):** [main.py](main.py), [configs/default.yaml](configs/default.yaml), [report/](report/)

> Thêm model mới: tạo file trong `src/models/`, khai báo ở [src/models/registry.py](src/models/registry.py). Không phải sửa training/experiment script.

---

## Proposed model

Cho input features `H0 = X`:

```
H1 = ELU( GATConv_multi(H0, E) )        # K heads, concat  -> local / one-hop
H2 = GATConv_single(H1, E)              # single head       -> two-hop
Z  = softmax( H2 + H0 @ W_skip )        # residual skip from raw features
```

DropEdge (xóa ngẫu nhiên một phần cạnh) được áp dụng cho `E` trong lúc train như
structural augmentation + regularization. Cờ `residual` và `drop_edge` tách riêng để
ablation bật/tắt từng đóng góp độc lập.

---

## Cấu trúc repo

```
GraphMl/
├── main.py                     # train 1 model: python main.py --model <name>
├── configs/default.yaml        # toàn bộ hyperparameter
├── notebooks/demo.ipynb        # Colab chạy end-to-end (Run all)
├── src/
│   ├── data/dataset.py         # [P2] Cora loader (Planetoid) + statistics
│   ├── models/                 # mlp, gcn, graphsage, gat, proposed (+ registry)
│   ├── training/               # trainer, metrics, multi-seed runner
│   └── utils/                  # config, seeding, plotting
├── experiments/
│   ├── run_baselines.py        # [P2] MLP vs GCN
│   ├── run_comparison.py       # [P3] GCN vs GraphSAGE vs GAT
│   └── run_ablation.py         # [P4] proposed model ablation
├── scripts/dataset_stats.py    # [P2] dataset statistics
├── results/                    # metrics CSVs (generated)
├── figures/                    # curves / plots (generated)
└── report/                     # LaTeX / Word report + slides
```

---

## `main.py` — các cờ hay dùng

```bash
python main.py --model gcn                       # gcn | mlp | graphsage | gat | proposed
python main.py --model proposed --drop-edge 0.3  # override DropEdge
python main.py --model gat --seeds 10            # nhiều seed hơn
python main.py --model proposed --device cpu     # ép chạy CPU
```
Tất cả hyperparameter mặc định nằm ở [configs/default.yaml](configs/default.yaml).

---

## Format bảng kết quả (đã thống nhất)

| Model | Test acc (mean ± std) | Val acc | #seeds |
|-------|-----------------------|---------|--------|

Báo cáo **test accuracy tại epoch có validation tốt nhất**, trung bình trên 5 seeds.

## Reproducibility

Mọi experiment seed Python/NumPy/PyTorch ([src/utils/seed.py](src/utils/seed.py)) và
report **mean ± std**. Model selection theo best **validation** accuracy; số báo cáo là
**test** accuracy tại epoch đó.
