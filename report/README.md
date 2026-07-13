# Report & Presentation

Owner: Person 1 (final integration) + Person 5 (analysis/figures/slides).

## Structure
- `report/` — LaTeX or Word source of the final report.
- Figures are generated into `../figures/` by the experiment scripts.
- Result tables are generated into `../results/` as CSV (paste into LaTeX/Word).

## Suggested report sections (maps to task split)
1. **Introduction** + **Problem formulation** — Person 1
2. **Dataset** (Cora statistics) — Person 2
3. **Methodology**
   - MLP / GCN — Person 2
   - GraphSAGE / GAT — Person 3
   - Proposed: Residual Multi-Head Mixing GAT — Person 4
4. **Experiments & Results**
   - MLP vs GCN (`results/baselines.csv`)
   - Architecture comparison (`results/comparison.csv`)
   - Ablation (`results/ablation.csv`)
5. **Discussion & Limitations** — Person 5
6. **Conclusion** — Person 1

## Experiment table format (agreed default)
| Model | Test acc (mean ± std) | Val acc | #seeds |
|-------|-----------------------|---------|--------|

Report **test** accuracy selected at best **validation** epoch, averaged over 5 seeds.
