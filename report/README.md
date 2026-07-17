# Report & Presentation

The submittable report is in [`latex/`](latex/) (NeurIPS 2025 style). See
[`latex/README.md`](latex/README.md) for how to build it.

## Structure
- `latex/` — LaTeX source + compiled `main.pdf` of the final report.
- Figures are generated into `../figures/` by the experiment scripts.
- Result tables are generated into `../results/` as CSV.

## Report sections
1. **Introduction** + **Problem formulation**
2. **Dataset** (Cora statistics)
3. **Background** (message-passing framework)
4. **Models** — MLP / GCN / GraphSAGE / GAT / proposed Residual Multi-Head Mixing GAT
5. **Experiments & Results**
   - MLP vs GCN (`results/baselines.csv`)
   - Architecture comparison (`results/comparison.csv`)
   - Ablation (`results/ablation.csv`)
6. **Discussion & Limitations**
7. **Conclusion**

## Experiment table format
| Model | Test acc (mean ± std) | Val acc | #seeds |
|-------|-----------------------|---------|--------|

Report **test** accuracy selected at best **validation** epoch, averaged over 5 seeds.
