# S1 — Dettling reproduction, full grid (LRZ cluster)

Figures 3 and 5 of Dettling, Drton & Kolar (2024), reproduced with the Direct Lyapunov Lasso.

| | |
|---|---|
| grid | p ∈ {10,15,20,25,30,40,50} × k ∈ {1,2,3,4} × 4 C-choices × 100 reps = **11,200 datasets** |
| config | lasso, solver `fista`, tol 1e-8, standardized Σ̂, 100 λ, λ_min = λ_max/1e4, seed 20260922 |
| where | LRZ Linux Cluster, `serial_std`, 64-task array on 7 cm4 nodes; Python 3.10.12, NumPy 2.2.6 |
| compute | 50.2 CPU-hours; finished 2026-09-29 09:03–09:24 UTC |

| file | contents |
|---|---|
| `m0_reps100.csv`, `.npz` | Figure 3: per-path metrics, and per-λ confusion counts |
| `s1_shards/` | Figure 5 raw shards (64). **Gitignored**: large, and reproducible from their `config_json` |
| `s1_summary.csv` | Figure 5 itself: mean ± SE per (p, C-choice), 400 datasets per point |
| `s1_per_dataset.csv`, `s1_curves.csv` | per-dataset metrics; mean ROC/PR curves |
| `comparison_vs_dettling.csv` | ours vs values extracted from the published figure, with z-scores |
| `figures/` | `figure3_reproduction`, `figure5_reproduction` (.png, .pdf) |

Checks: every (p, k, C) cell has exactly 100 replicates. Re-aggregating the shards with the
current code reproduces `s1_summary.csv` exactly. Figure 3's `max_acc`, `max_f1` and `auc` are
bit-identical to an independent laptop run (macOS, NumPy 2.0) across all 2,202 paths.

Interpretation: `simulations/S1_reproduction.md` §8.10.
