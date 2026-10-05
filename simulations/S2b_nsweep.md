# S2b: the n-sweep. Three losses × three penalties at $n = 10^3 \dots \infty$, $p = 10, 20$

*Extends S1b and S2 ([`S2_penalties_losses.md`](S2_penalties_losses.md)) from $n = 1000$ to larger
sample sizes. Run on the LRZ cluster on 2–3 October 2026, analysed on 4 October. Numbers:
`runs/nsweep_p10-20/` (per cell `s1_per_dataset.csv`, `s1_summary.csv`, `s1_curves.csv`; across cells
`nsweep_*.csv`). Figures: `runs/nsweep_p10-20/figures/`. How it was run:
[`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) §2.6.*

**This sweep is the baseline: Dettling's Figure 5 pipeline** (standardised data, $C = 2I$ in the
fit) **and paths that run from $\lambda_{\max}$ to the dense end.** The two changes that the
3 October notes found to matter, the rescaled $C$ and dense → sparse paths
([`../next_steps/031026/next_steps_031026.md`](../next_steps/031026/next_steps_031026.md)), are not
part of it.

---

## 0. Summary

1. **The sweep is complete and consistent.** 36 cells, 800 graphs each, 28 800 paths, 720 CPU-hours.
   The direct-loss cells reproduce the local runs (§2).
2. **MCP and SCAD lose to the lasso in every cell.** That is 24 combinations of loss, $p$ and $n$,
   on `max_f1`, `auc` and `aupr`, with paired $z$ between −12 and −56. MCP has the higher `max_f1`
   than the lasso on 3–14 % of the graphs, SCAD on 5–27 %.
3. **More data makes the gap larger, not smaller.** For example MCP − lasso in `max_f1` on the
   direct loss at $p = 10$: −0.096 at $n = 10^3$, −0.117 at $n = \infty$. The lasso gains 0.05–0.10
   from $n = 10^3$ to $\infty$; MCP gains 40–70 % of that and SCAD 45–80 %.
4. **The error is mostly orientation, and it does not go away.** With growing $n$ the lasso reverses
   fewer true edges and keeps both directions of more. MCP keeps reversing the same number: about 4
   per graph at $p = 10$ and 8–9 at $p = 20$, at every $n$ (§3.6).
5. **The loss matters less than the penalty.** With the lasso, the log-likelihood loss is the best
   of the three and the Frobenius loss the worst; the differences are 0.01–0.05 in `max_f1`. The
   penalty gaps are 0.03–0.16.
6. **$n = 10^5$ is the population limit** for all practical purposes; the curves are flat from
   there to $n = \infty$.
7. **The covariance losses are numerically fragile.** The same path computed on the laptop and on
   the cluster differs on most datasets, although the code and data are identical. The averages
   agree within about 0.01 (§2).

This confirms the pilot on all three losses and at every sample size: on Dettling's pipeline, with
the usual path, nonconvex penalties do not help. It is the reference against which the new arms
(rescaled $C$, dense → sparse) have to be measured.

![max F1 against n](../runs/nsweep_p10-20/figures/metric_vs_n_max_f1.png)

*Figure 1. Maximum $F_1$ along the path against the sample size, mean over 400 graphs per point
with ± 1 standard error. Figures 2 and 3 show the other two metrics.*

---

## 1. What was run

| | |
|---|---|
| graphs | Dettling's Figure 5 generator and seeds: $p \in \{10, 20\}$, $k = 1..4$, four $C$ choices, 25 reps: 800 graphs per cell, the same 800 in every cell |
| sample sizes | $n = 10^3, 10^4, 10^5, \infty$ ($\infty$: the population covariance) |
| input | standardised (correlation matrix), $C = 2I$ in the fit, as in S1 |
| losses | direct (Dettling), log-likelihood and Frobenius (Varando & Hansen) |
| penalties | lasso, MCP ($\gamma = 3$), SCAD ($\gamma = 3.7$), textbook convention |
| paths | 100 $\lambda$ from $\lambda_{\max}$ down to $10^{-4}\lambda_{\max}$, warm-started from the sparse end |
| metrics | `max_f1`, `auc`, `aupr`, `max_acc` as in S1; the best-F1 estimate of every path is stored |

All comparisons below are paired over the same graphs. $z$ is the mean paired difference divided
by its standard error.

## 2. Validation

**Completeness.** Every cell has all its shards and 800 datasets (400 per $p$), and the settings
stored in the shards match the cell (`nsweep_audit.csv`).

**Cells that repeat local runs** (`nsweep_validation.txt`), compared dataset by dataset:

| cluster cell | local run | datasets | differ | mean abs. difference in `max_f1` |
|---|---|---|---|---|
| direct, lasso, $n = 10^3$ | `runs/s1_dettling_reproduction` (the September cluster run) | 800 | **0** | 0 |
| direct, MCP, $n = 10^3$ | `runs/s1b_pilot_p10-20/MCP` (laptop) | 800 | 4 | $9 \cdot 10^{-6}$ |
| direct, SCAD, $n = 10^3$ | `runs/s1b_pilot_p10-20/SCAD` (laptop) | 800 | 6 | 0 (`auc` differs) |
| direct, lasso / MCP / SCAD, $n = \infty$, $p = 10$ | the local pilot of 2 October | 210 each | 1 / 6 / 5 | 0 / 0.002 / 0.001 |
| log-likelihood, lasso / MCP / SCAD, $n = 10^3$, $p = 10$ | `runs/s2_pilot_p10` (laptop) | 160 each | **63 / 101 / 74** | 0.014 / 0.053 / 0.031 |
| Frobenius, lasso / MCP / SCAD, $n = 10^3$, $p = 10$ | `runs/s2_pilot_p10` (laptop) | 160 each | **39 / 79 / 59** | 0.003 / 0.020 / 0.012 |

- **The direct loss reproduces.** The lasso is identical on all 800 datasets. MCP and SCAD differ on
  a handful, as expected for a nonconvex objective computed on two machines.
- **The covariance losses do not reproduce dataset by dataset.** The cause is the platform, not the
  code or the data:
  - The settings are the same, and $\lambda_{\max}$ agrees to $3 \cdot 10^{-15}$ on every dataset.
  - Rerunning one pilot shard on the laptop with today's code (20 log-likelihood paths each for
    MCP and the lasso) gives the pilot's numbers **exactly**, on all 40 paths and all 100
    $\lambda$'s. On the cluster, 7 of the 20 MCP paths and 16 of the 20 lasso paths are identical.
  - The two MCP paths separate early (median: the 12th $\lambda$ from $\lambda_{\max}$) and then
    follow different stationary points. Neither machine ends lower: the laptop has the lower objective on
    34 % of the later $\lambda$'s.
- **The averages agree.** On the 160 common datasets, laptop against cluster:

  | | `max_f1` | `auc` | `aupr` |
  |---|---|---|---|
  | log-likelihood, lasso | 0.593 / 0.595 | 0.708 / 0.708 | 0.510 / 0.512 |
  | log-likelihood, MCP | 0.481 / 0.493 | 0.594 / 0.603 | 0.429 / 0.447 |
  | log-likelihood, SCAD | 0.502 / 0.509 | 0.602 / 0.605 | 0.438 / 0.442 |
  | Frobenius, lasso | 0.565 / 0.568 | 0.692 / 0.692 | 0.472 / 0.473 |
  | Frobenius, MCP | 0.450 / 0.452 | 0.577 / 0.580 | 0.407 / 0.408 |
  | Frobenius, SCAD | 0.469 / 0.472 | 0.582 / 0.584 | 0.408 / 0.410 |

  The largest difference is 0.012 in `max_f1` and 0.018 in `aupr` (log-likelihood, MCP; $z = 1.5$
  and 2.3).

**Consequence.** For the covariance losses, a single dataset's number depends on the machine.
Means over hundreds of graphs are stable to about 0.01, which is small against the effects reported
below (0.03–0.21). It is a direct view of the many stationary points described in
`docs/LIKELIHOOD.md`, and it holds even for the lasso, because these losses are not convex in $M$.

## 3. Results

### 3.1 Means

`max_f1`, lasso / MCP / SCAD (400 graphs per number; `nsweep_means.csv`):

| $p$ | loss | $n = 10^3$ | $n = 10^4$ | $n = 10^5$ | $n = \infty$ |
|---|---|---|---|---|---|
| 10 | direct | 0.589 / 0.493 / 0.535 | 0.629 / 0.518 / 0.561 | 0.635 / 0.518 / 0.566 | 0.638 / 0.520 / 0.568 |
| 10 | log-likelihood | 0.601 / 0.493 / 0.516 | 0.634 / 0.513 / 0.538 | 0.641 / 0.518 / 0.543 | 0.651 / 0.512 / 0.544 |
| 10 | Frobenius | 0.573 / 0.455 / 0.478 | 0.607 / 0.469 / 0.498 | 0.621 / 0.472 / 0.500 | 0.628 / 0.478 / 0.501 |
| 20 | direct | 0.531 / 0.455 / 0.500 | 0.601 / 0.510 / 0.557 | 0.616 / 0.517 / 0.570 | 0.617 / 0.516 / 0.571 |
| 20 | log-likelihood | 0.569 / 0.461 / 0.487 | 0.631 / 0.498 / 0.530 | 0.643 / 0.506 / 0.539 | 0.644 / 0.504 / 0.540 |
| 20 | Frobenius | 0.486 / 0.383 / 0.411 | 0.554 / 0.415 / 0.450 | 0.574 / 0.424 / 0.460 | 0.581 / 0.424 / 0.462 |

The ordering lasso > SCAD > MCP holds in all 24 rows-by-columns.

![AUC against n](../runs/nsweep_p10-20/figures/metric_vs_n_auc.png)

*Figure 2. Area under the ROC curve against the sample size. The gap between the lasso and the
nonconvex penalties is wider than in `max_f1`, and MCP and SCAD are close to each other.*

![AUPR against n](../runs/nsweep_p10-20/figures/metric_vs_n_aupr.png)

*Figure 3. Area under the precision-recall curve against the sample size.*

### 3.2 Penalty against the lasso

![paired differences](../runs/nsweep_p10-20/figures/paired_vs_lasso_max_f1.png)

*Figure 4. Paired difference in `max_f1` (penalty − lasso, same loss, same $n$, same graphs) with
± 2 standard errors.*

![paired differences, AUC](../runs/nsweep_p10-20/figures/paired_vs_lasso_auc.png)

*Figure 5. The same paired difference for the area under the ROC curve.*

![paired differences, AUPR](../runs/nsweep_p10-20/figures/paired_vs_lasso_aupr.png)

*Figure 6. The same paired difference for the area under the precision-recall curve.*

`max_f1`, paired difference to the lasso, $z$ in brackets (`nsweep_paired_vs_lasso.csv`):

| $p$ | loss | penalty | $n = 10^3$ | $n = 10^4$ | $n = 10^5$ | $n = \infty$ |
|---|---|---|---|---|---|---|
| 10 | direct | MCP | −0.096 (−19) | −0.110 (−21) | −0.117 (−22) | −0.117 (−22) |
| 10 | direct | SCAD | −0.053 (−14) | −0.068 (−15) | −0.069 (−15) | −0.070 (−15) |
| 10 | log-likelihood | MCP | −0.108 (−20) | −0.121 (−21) | −0.124 (−22) | −0.139 (−23) |
| 10 | log-likelihood | SCAD | −0.086 (−19) | −0.096 (−19) | −0.098 (−20) | −0.107 (−19) |
| 10 | Frobenius | MCP | −0.118 (−25) | −0.137 (−26) | −0.149 (−28) | −0.150 (−27) |
| 10 | Frobenius | SCAD | −0.096 (−21) | −0.109 (−23) | −0.121 (−24) | −0.127 (−25) |
| 20 | direct | MCP | −0.076 (−20) | −0.091 (−21) | −0.099 (−21) | −0.102 (−22) |
| 20 | direct | SCAD | −0.031 (−12) | −0.044 (−13) | −0.046 (−12) | −0.047 (−12) |
| 20 | log-likelihood | MCP | −0.109 (−26) | −0.133 (−30) | −0.138 (−30) | −0.140 (−32) |
| 20 | log-likelihood | SCAD | −0.082 (−22) | −0.101 (−25) | −0.105 (−24) | −0.105 (−24) |
| 20 | Frobenius | MCP | −0.103 (−31) | −0.139 (−39) | −0.150 (−36) | −0.156 (−38) |
| 20 | Frobenius | SCAD | −0.075 (−24) | −0.104 (−31) | −0.113 (−30) | −0.119 (−31) |

The other two metrics, as ranges over the four sample sizes:

| $p$ | loss | MCP − lasso: `auc` | `aupr` | SCAD − lasso: `auc` | `aupr` |
|---|---|---|---|---|---|
| 10 | direct | −0.105 … −0.128 | −0.061 … −0.077 | −0.098 … −0.113 | −0.056 … −0.069 |
| 10 | log-likelihood | −0.110 … −0.132 | −0.074 … −0.097 | −0.098 … −0.118 | −0.067 … −0.080 |
| 10 | Frobenius | −0.120 … −0.151 | −0.070 … −0.094 | −0.111 … −0.137 | −0.065 … −0.084 |
| 20 | direct | −0.134 … −0.161 | −0.073 … −0.095 | −0.108 … −0.125 | −0.056 … −0.075 |
| 20 | log-likelihood | −0.145 … −0.180 | −0.108 … −0.141 | −0.130 … −0.158 | −0.091 … −0.116 |
| 20 | Frobenius | −0.165 … −0.211 | −0.089 … −0.132 | −0.146 … −0.191 | −0.076 … −0.114 |

- **Every difference is negative,** with $z$ between −12 and −56.
- **The deficit grows with $n$** and levels off at $n = 10^5$. It never shrinks.
- **It is larger on the covariance losses than on the direct loss,** and largest on the Frobenius
  loss. SCAD's advantage over MCP is 0.04–0.055 on the direct loss and 0.02–0.04 on the other
  two.
- **Per graph:** MCP has the higher `max_f1` than the lasso on 3–14 % of the graphs, depending on
  the cell; SCAD on 5–27 %. For `auc` it is 0–13 % and 0–15 %.

### 3.3 By $C$ choice

![by C choice](../runs/nsweep_p10-20/figures/by_c_choice_MCP.png)

*Figure 7. MCP − lasso in `max_f1`, separately for Dettling's four volatility settings
(100 graphs per point).*

![by C choice, SCAD](../runs/nsweep_p10-20/figures/by_c_choice_SCAD.png)

*Figure 8. SCAD − lasso in `max_f1`, by volatility setting.*

- **The result holds in each of the four settings,** in all 96 combinations of setting, loss, $p$
  and $n$: MCP − lasso between −0.058 and −0.186 ($z \le -8.5$), SCAD − lasso between −0.023 and
  −0.144 ($z \le -5.6$).
- **`C_Random_Full` is different in level.** There the fitted model is far from the truth (the true
  $C$ is not diagonal), and the lasso itself hardly improves with $n$: 0.513 → 0.518 at $p = 10$ and
  0.441 → 0.468 at $p = 20$ on the direct loss, against gains of 0.06–0.11 in the other three
  settings. The penalty gap is smaller there and flat in $n$.

### 3.4 The lasso under the three losses

![lasso by loss](../runs/nsweep_p10-20/figures/lasso_by_loss.png)

*Figure 9. The lasso on the three losses.*

Paired difference to the direct loss, same penalty and $n$ (`nsweep_paired_vs_direct.csv`), as
ranges over $n$:

| | $p = 10$: `max_f1` | `aupr` | $p = 20$: `max_f1` | `aupr` |
|---|---|---|---|---|
| lasso: log-likelihood − direct | +0.005 … +0.013 ($z$ 2–5) | +0.043 … +0.048 ($z$ 9–10) | +0.027 … +0.038 ($z$ 12–13) | +0.077 … +0.078 ($z$ 19–21) |
| lasso: Frobenius − direct | −0.009 … −0.022 | +0.002 … +0.006 | −0.037 … −0.047 ($z$ −16 … −19) | −0.010 … −0.014 |
| MCP: log-likelihood − direct | −0.008 … 0.000 | +0.026 … +0.034 | −0.012 … +0.006 | +0.030 … +0.042 |
| MCP: Frobenius − direct | −0.038 … −0.049 | −0.005 … −0.015 | −0.072 … −0.095 | −0.027 … −0.051 |
| SCAD: log-likelihood − direct | −0.020 … −0.023 | +0.032 … +0.036 | −0.013 … −0.031 | +0.035 … +0.043 |
| SCAD: Frobenius − direct | −0.058 … −0.066 | −0.005 … −0.012 | −0.089 … −0.110 | −0.030 … −0.052 |

- **With the lasso, the log-likelihood loss is the best of the three** at every $n$, most clearly in
  `aupr` (+0.04 at $p = 10$, +0.08 at $p = 20$) and more at $p = 20$ than at $p = 10$. This is
  Varando & Hansen's finding and S2 §3.3, now for all sample sizes.
- **The Frobenius loss is the worst,** clearly so at $p = 20$.
- **With MCP and SCAD the log-likelihood loss gives no gain in `max_f1`.** The penalty gap is
  therefore larger on the log-likelihood loss than on the direct loss.

### 3.5 The effect of the sample size

Gain in `max_f1` from $n = 10^3$ to $n = 10^4$ / $10^5$ / $\infty$ (`nsweep_paired_vs_n1000.csv`):

| $p$ | loss | lasso | MCP | SCAD |
|---|---|---|---|---|
| 10 | direct | +0.040 / +0.046 / +0.049 | +0.025 / +0.025 / +0.027 | +0.026 / +0.031 / +0.032 |
| 10 | log-likelihood | +0.033 / +0.040 / +0.050 | +0.020 / +0.024 / +0.019 | +0.022 / +0.028 / +0.029 |
| 10 | Frobenius | +0.033 / +0.048 / +0.055 | +0.014 / +0.017 / +0.023 | +0.020 / +0.023 / +0.024 |
| 20 | direct | +0.070 / +0.085 / +0.086 | +0.054 / +0.061 / +0.060 | +0.057 / +0.070 / +0.070 |
| 20 | log-likelihood | +0.062 / +0.074 / +0.075 | +0.037 / +0.045 / +0.043 | +0.043 / +0.052 / +0.052 |
| 20 | Frobenius | +0.068 / +0.088 / +0.095 | +0.032 / +0.041 / +0.041 | +0.039 / +0.049 / +0.050 |

- **Most of the gain comes between $n = 10^3$ and $10^4$.** From $10^5$ to $\infty$ the change is at
  most 0.01.
- **Even with the exact covariance the lasso stops at `max_f1` 0.58–0.65.** Its errors on this
  pipeline are built in, not noise.
- **The nonconvex paths gain less than the lasso** (MCP 40–70 % of the lasso's gain, SCAD 45–80 %),
  which is why the gap widens.

### 3.6 Skeleton and orientation at the best-F1 point

From the stored best-F1 estimates (`nsweep_orientation.csv`), with the categories of S3a: a true
single edge is found in the correct direction only, in both directions ("hedged"), or in the
reversed direction only; a true 2-cycle is found in both directions or in one.

![skeleton vs directed](../runs/nsweep_p10-20/figures/skeleton_vs_directed.png)

*Figure 10. Paired difference to the lasso at each method's best-F1 point: directed $F_1$ (filled)
and skeleton $F_1$ (hollow).*

![orientation counts](../runs/nsweep_p10-20/figures/orientation_counts_p20.png)

*Figure 11. True single edges found in the reversed direction only (top) and in both directions
(bottom), per graph, $p = 20$.*

![orientation counts, p = 10](../runs/nsweep_p10-20/figures/orientation_counts_p10.png)

*Figure 12. The same for $p = 10$.*

Per graph, lasso / MCP / SCAD, at $n = 10^3$ and $n = \infty$:

| $p$ | loss | reversed, $10^3$ | reversed, $\infty$ | hedged, $10^3$ | hedged, $\infty$ | 2-cycles in both directions, $10^3$ | $\infty$ |
|---|---|---|---|---|---|---|---|
| 10 | direct | 1.7 / 4.0 / 2.9 | 1.5 / 4.2 / 3.0 | 4.5 / 0.1 / 1.3 | 5.0 / 0.2 / 1.2 | 1.38 / 0.04 / 0.49 | 1.51 / 0.03 / 0.43 |
| 10 | log-likelihood | 1.8 / 4.4 / 3.9 | 1.6 / 4.4 / 3.8 | 3.4 / 0.1 / 0.2 | 4.1 / 0.1 / 0.2 | 1.17 / 0.01 / 0.05 | 1.38 / 0.03 / 0.05 |
| 10 | Frobenius | 2.1 / 4.5 / 4.1 | 1.7 / 4.5 / 4.2 | 3.6 / 0.1 / 0.1 | 4.5 / 0.1 / 0.1 | 1.31 / 0.01 / 0.05 | 1.45 / 0.01 / 0.03 |
| 20 | direct | 2.9 / 7.5 / 4.7 | 2.1 / 7.9 / 5.0 | 12.3 / 0.8 / 5.8 | 14.5 / 0.9 / 4.8 | 1.47 / 0.12 / 0.81 | 1.56 / 0.08 / 0.57 |
| 20 | log-likelihood | 3.6 / 7.7 / 7.1 | 2.4 / 8.5 / 7.4 | 8.0 / 0.1 / 0.3 | 11.4 / 0.2 / 0.6 | 1.07 / 0.01 / 0.06 | 1.44 / 0.02 / 0.07 |
| 20 | Frobenius | 4.7 / 8.3 / 7.5 | 2.8 / 9.3 / 7.9 | 8.0 / 0.0 / 0.3 | 12.4 / 0.1 / 0.3 | 1.03 / 0.01 / 0.02 | 1.46 / 0.01 / 0.03 |

Paired differences to the lasso at the best-F1 point (directed $F_1$ / skeleton $F_1$ / orientation
recall):

| $p$ | loss | penalty | $n = 10^3$ | $n = \infty$ |
|---|---|---|---|---|
| 10 | direct | MCP | −0.096 / −0.027 / −0.194 | −0.117 / −0.034 / −0.223 |
| 10 | direct | SCAD | −0.053 / −0.022 / −0.113 | −0.070 / −0.033 / −0.132 |
| 10 | log-likelihood | MCP | −0.108 / −0.021 / −0.201 | −0.139 / −0.042 / −0.224 |
| 10 | log-likelihood | SCAD | −0.086 / −0.018 / −0.160 | −0.107 / −0.035 / −0.177 |
| 10 | Frobenius | MCP | −0.118 / −0.044 / −0.185 | −0.150 / −0.061 / −0.235 |
| 10 | Frobenius | SCAD | −0.096 / −0.032 / −0.160 | −0.127 / −0.049 / −0.214 |
| 20 | direct | MCP | −0.076 / −0.026 / −0.192 | −0.102 / −0.055 / −0.211 |
| 20 | direct | SCAD | −0.031 / −0.019 / −0.080 | −0.047 / −0.039 / −0.109 |
| 20 | log-likelihood | MCP | −0.109 / −0.052 / −0.176 | −0.140 / −0.076 / −0.217 |
| 20 | log-likelihood | SCAD | −0.082 / −0.036 / −0.149 | −0.105 / −0.059 / −0.179 |
| 20 | Frobenius | MCP | −0.103 / −0.055 / −0.170 | −0.156 / −0.083 / −0.255 |
| 20 | Frobenius | SCAD | −0.075 / −0.040 / −0.136 | −0.119 / −0.062 / −0.210 |

*Orientation recall*: among the true single edges whose pair is found, the share whose true
direction is present.

- **With more data the lasso orients better; MCP does not.**
  - The lasso's reversed edges fall with $n$ (at $p = 20$ from 2.9–4.7 to 2.1–2.8 per graph), and
    its hedges rise (from 8–12 to 11–15).
  - MCP's reversed edges stay at about 4 per graph at $p = 10$ and rise from 7.5–8.3 to 7.9–9.3 at
    $p = 20$. It keeps both directions of almost no pair.
  - MCP recovers both directions of a true 2-cycle about 0.01–0.12 times per graph, against 1.0–1.6
    for the lasso.
- **The S3a mechanism holds on every loss and at every $n$.** The first direction to enter is kept;
  more data does not correct it.
- **On the covariance losses SCAD behaves like MCP.** On the direct loss SCAD still hedges (1.2–5.8
  pairs per graph); on the log-likelihood and Frobenius losses it does not (0.1–0.6). That is why
  SCAD's advantage over MCP is small there.
- **The skeleton gap is not negligible any more.** At the directed best-F1 point it is 0.02–0.03 on
  the direct loss at $n = 10^3$, as in S2 §3.4. It grows with $n$, with $p$ and on the covariance
  losses, up to 0.06–0.08. For MCP it is a quarter to a half of the directed gap; the rest is
  orientation. For SCAD on the direct loss at $p = 20$, where the directed gap is small and SCAD
  still hedges, the skeleton gap is most of it (0.019 of 0.031 at $n = 10^3$, 0.039 of 0.047 at
  $n = \infty$).
  The skeleton here is measured at the $\lambda$ that is best for the directed graph, so this is an
  upper bound on the skeleton gap along the path.

## 4. Convergence and cost

Per cell, means over the four sample sizes (`nsweep_audit.csv`):

| loss | penalty | CPU-h per cell | seconds per path, $p = 10$ / $20$ | slowest path | longest shard | $\lambda$'s at the 50 000-step cap | largest first-order violation | $\lambda$'s with violation $> 10^{-4}$ |
|---|---|---|---|---|---|---|---|---|
| direct | lasso | 1.4 | 4 / 9 | 2 min | 0.4 h | — | — | — |
| direct | MCP | 6.2 | 16 / 39 | 7 min | 1.6 h | — | — | — |
| direct | SCAD | 8.4 | 22 / 54 | 8 min | 2.3 h | — | — | — |
| log-likelihood | lasso | 24.1 | 47 / 169 | 34 min | 3.5 h | 0.12 % | $6.5 \cdot 10^{-6}$ | 0 |
| log-likelihood | MCP | 6.8 | 7 / 55 | 7 min | 1.1 h | 0.05 % | $1.1 \cdot 10^{-6}$ | 0 |
| log-likelihood | SCAD | 6.8 | 8 / 52 | 9 min | 1.2 h | 0.04 % | $1.5 \cdot 10^{-6}$ | 0 |
| Frobenius | lasso | 51.9 | 107 / 360 | 42 min | 4.9 h | 0.78 % | $1.1 \cdot 10^{-4}$ | 1 |
| Frobenius | MCP | 37.3 | 98 / 237 | 65 min | 4.9 h | 2.71 % | $1.8 \cdot 10^{-3}$ | 6 |
| Frobenius | SCAD | 37.1 | 95 / 239 | 60 min | 4.6 h | 2.33 % | $6.6 \cdot 10^{-4}$ | 3 |

- **Total: 720 CPU-hours,** of which the Frobenius loss takes 505, the log-likelihood loss 151 and
  the direct loss 64. The sample size does not change the cost.
- **Convergence.**
  - The direct-loss solvers stop on the coefficient change and report no violation.
  - On the log-likelihood loss every $\lambda$ has a first-order violation below $10^{-5}$.
  - On the Frobenius loss 0.8–2.7 % of the $\lambda$'s reach the step cap. At those $\lambda$'s the
    violation is still small (median $3$–$5 \cdot 10^{-7}$), and 10 $\lambda$'s out of 960 000 exceed
    $10^{-4}$.
- **For future runs:** the longest shard took 4.9 h, so the 24 h limit of `serial_std` is ample.
  The earlier estimates in `docs/REPRODUCTION.md` §2.6 were too high by a factor of two to three
  for the Frobenius loss and of two for the log-likelihood lasso; the table there now holds the
  measured values.

## 5. What this means

- **The pilot's conclusion stands on all three losses and at every sample size.** On Dettling's
  pipeline with the standard path, MCP and SCAD recover the directed graph worse than the lasso,
  and more data widens the gap. "The nonconvex penalties need more data" is ruled out.
- **The cause is the one identified in S2, S3a and the 3 October notes:** the direction of an edge
  is fixed when it first enters the path, on information that does not determine it, and the
  nonconvex penalties keep it. The lasso keeps both directions and lets the data sort them out as
  $n$ grows.
- **This is the baseline, not the last word.** The independent study and its replication
  (`next_steps/031026/`) find that MCP and SCAD beat the lasso when two things are changed: the
  volatility is rescaled to match the standardised data, and the path runs dense → sparse. Neither
  is in this sweep. The corresponding cluster arms are Step 3 of the 3 October plan.
- **The misspecified setting shows in the data:** in `C_Random_Full` the lasso does not improve
  with $n$ at all. Even the "well-specified" `C_ID` is misspecified on this pipeline, because of the
  standardisation.

## 6. Caveats

- **Oracle tuning.** `max_f1` and the best-F1 estimates use the true graph to choose $\lambda$;
  `auc` and `aupr` summarise the whole path. BIC-tuned comparisons are in S3b.
- **$p \le 20$ and 25 reps.** The effects are large against the standard errors (0.003–0.007), but
  larger $p$ was not run.
- **Machine dependence of the covariance losses** (§2): trust the means, not single datasets.
- **$\gamma$ fixed** at 3 and 3.7. The independent study's sweep finds that a larger $\gamma$ only
  moves the path towards the lasso.

## 7. Reproducing

```bash
# on LRZ: docs/REPRODUCTION.md §2.6 (cluster/submit_nsweep.sh), then per cell
python simulations/aggregate_s1.py --in-dir runs/nsweep_p10-20/<cell>/s1_shards
# locally, after copying runs/nsweep_p10-20 home
python simulations/diagnostics/nsweep.py          # validation, paired tables, orientation, audit
python simulations/diagnostics/plot_nsweep.py     # the figures
python -m pytest tests/test_nsweep_analysis.py    # the paired statistics
```
