# S1b and S2 — MCP / SCAD on the direct loss, and all three penalties on Varando's losses

*Pilot results, laptop scale. Companion to [`S1_reproduction.md`](S1_reproduction.md) (the lasso
baseline, Figure 5 at full scale) and to [`docs/NONCONVEX.md`](../docs/NONCONVEX.md) and
[`docs/LIKELIHOOD.md`](../docs/LIKELIHOOD.md) (the methods). Numbers in `runs/s1b_pilot_p10-20/`
and `runs/s2_pilot_p10/`.*

## 1. What is compared

Three penalties × three losses, every combination fitted along the same 100-point $\lambda$ path
on the same datasets, scored with the same four support-recovery metrics as Figure 5:

| | lasso | MCP ($\gamma=3$) | SCAD ($\gamma=3.7$) |
|---|---|---|---|
| **direct** $\tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2$ — Dettling | S1 (full grid) | S1b | S1b |
| **loglik** $\log\det\Sigma(M)+\operatorname{tr}(\Sigma(M)^{-1}\hat\Sigma)$ — Varando | S2 | S2 | S2 |
| **frobenius** $\tfrac12\|\Sigma(M)-\hat\Sigma\|_F^2$ — Varando | S2 | S2 | S2 |

The question of the thesis is the columns: does a nonconvex penalty recover the support better
than the lasso, and does the answer depend on the loss?

## 2. Setup

Everything is inherited from S1 (`S1_reproduction.md` §3–4): $N=1000$ standardised observations,
$C_{\text{est}}=2I$, Dettling's four $C$ choices, $k\in\{1,2,3,4\}$, the grid
$\lambda_{\max}\cdot10^{\text{linspace}(-4,0,100)}$ with $\lambda_{\max}$ computed per dataset and
per loss, metrics on the off-diagonal support with the dense anchor (§8.8). The datasets are
**identical across all nine cells**: the random stream is seeded by `(seed, p, k, C, rep)`, so
`rep < 25` of the S1 cluster run, the S1b pilot and the S2 pilot are the same draws. Differences
between cells are therefore paired, not just independent samples.

Per study:

| | $p$ | reps per $(p,k,C)$ | datasets per point | run |
|---|---|---|---|---|
| S1 lasso baseline | 10, 20 (of 10…50) | 100, restricted to the first 25 / 10 | 100 / 40 | `runs/s1_dettling_reproduction` |
| S1b MCP, SCAD | 10, 20 | 25 | 100 | `runs/s1b_pilot_p10-20/{MCP,SCAD}` |
| S2 loglik / frobenius × 3 | 10 | 10 | 40 | `runs/s2_pilot_p10/{loss}_{penalty}` |

Solver settings: direct loss — FISTA / monotone APG, textbook convention, `tol = 1e-8` on the
coefficient change (docs/NONCONVEX.md §4); covariance losses — accelerated proximal gradient,
continuation from $\lambda_{\max}$ downwards, `tol = 1e-8` on the gradient mapping (the drivers
pass `S1Config.tol` to every loss; tighter than the library default of `1e-6`), at most 50,000
proximal steps per $\lambda$ (docs/LIKELIHOOD.md §4.3 and §5, which explain why these losses need
the path-following rule spelled out).

## 3. Results

*(filled in from the pilot runs; see the figures under each run's `figures/`)*

### 3.1 S1b — MCP and SCAD on the direct loss, $p=10,20$

**Both nonconvex penalties recover the support worse than the lasso**, on every metric except
accuracy, in every $C$ setting, at both $p$. Paired differences (penalty − lasso) over the 100
datasets per $(p, C)$ point, range over the four $C$ choices; $z$ is the paired $z$-score; "better"
is the fraction of datasets on which the penalty beats the lasso
(`runs/s1b_pilot_p10-20/{MCP,SCAD}/paired_vs_lasso.csv`):

| | $p$ | `max_acc` | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|---|
| **MCP** $\gamma=3$ | 10 | −0.015 … −0.004 ($z$ −5 … −2) | **−0.12 … −0.07** ($z$ −12 … −9) | **−0.12 … −0.07** ($z$ −13 … −8) | −0.08 … −0.04 ($z$ −11 … −7) |
| | 20 | +0.001 … +0.002 | **−0.09 … −0.06** ($z$ −12 … −9) | **−0.15 … −0.11** ($z$ −18 … −17) | −0.08 … −0.05 ($z$ −15 … −12) |
| **SCAD** $\gamma=3.7$ | 10 | −0.008 … −0.003 | −0.06 … −0.05 ($z$ −8 … −6) | **−0.11 … −0.06** ($z$ −12 … −8) | −0.07 … −0.03 ($z$ −11 … −8) |
| | 20 | ±0.001 | −0.04 … −0.02 ($z$ −7 … −6) | **−0.12 … −0.09** ($z$ −17 … −15) | −0.07 … −0.04 ($z$ −18 … −14) |

MCP beats the lasso on `auc` on 0–19 % of the datasets at $p=10$ and on 0–2 % at $p=20$; SCAD
on 1–22 % and 1–2 %. SCAD is consistently the less harmful of the two (SCAD − MCP: `max_f1`
+0.03 … +0.06, `auc` 0 … +0.035, $z$ up to +9). `max_acc` is flat because at these edge
densities the all-zero graph already scores 0.8–0.9 and the maximum is reached at the sparse end
of every path. Figure: `runs/s1b_pilot_p10-20/figures/penalties_direct.{png,pdf}`.

**Where the paths differ.** The mean path at $p=20$, `C_ID` (`s1_curves.csv`), by $\lambda$ index
from the dense end (0) to $\lambda_{\max}$ (99):

| | nnz | tpr | fpr |
|---|---|---|---|
| lasso, index 0 / 70 / 80 | 190 / 114 / 63 | 0.79 / 0.75 / 0.64 | 0.46 / 0.24 / 0.10 |
| MCP, index 0 / 70 / 80 | 190 / 99 / 51 | 0.60 / 0.53 / 0.46 | 0.49 / 0.22 / 0.09 |
| SCAD, index 0 / 70 / 80 | 189 / 108 / 58 | 0.63 / 0.58 / 0.54 | 0.48 / 0.24 / 0.10 |

All three saturate at the same $p(p+1)/2 - p = 190$ nonzeros, the rank of the design
(S1_reproduction.md §2.3). The lasso's saturated support holds 79 % of the true edges, MCP's 60 %.
At equal false-positive rate along the path (index 70–80) the nonconvex penalties find 10–20
points less recall. The difference is not in how many entries are selected but in *which*.

**Reading.** The direct loss is a quadratic whose design $A(\hat\Sigma)$ has a null space of
dimension $p(p-1)/2$, so it is never strongly convex, and at the dense end its minimisers form an
affine set of that dimension. For the lasso the problem stays convex: the path is unique and its
dense end is the minimum-$\ell_1$ point of that affine set, which — empirically — is a good
support. For MCP/SCAD the penalty is concave on $[\gamma\lambda, \infty)$ (MCP: everywhere above
0 with curvature $-1/\gamma$), so the objective is nonconvex exactly in the directions where the
loss has little or no curvature. Loh & Wainwright's guarantee for local optima of nonconvex
$M$-estimators needs restricted strong convexity with constant $\alpha > \mu = 1/\gamma$; here
the curvature of the loss along a coordinate is $v_{ij} = 2(\|\hat\Sigma_{j,\cdot}\|^2 +
\hat\Sigma_{ij}^2) \approx 2$–$4$ for a correlation matrix, but along many sparse combinations
of coordinates it is far smaller, and with $\gamma = 3$ the condition fails there. The
continuation path then settles into local minima that keep whatever entered first: once an entry
exceeds $\gamma\lambda$ it is no longer shrunk, and a false positive that enters early is locked
in. SCAD, which still shrinks on $[\lambda, \gamma\lambda]$ with the weaker slope, locks in less,
which matches its intermediate position.

**It is not the optimiser.** `skglm`'s coordinate descent (an independent implementation of
textbook MCP) returns the same `max_f1` / `auc` as our monotone APG on five S1 datasets, with
objectives within $3\cdot10^{-4}$ along the whole path (one dataset: MCP 0.638 vs lasso 0.596;
the other four: 0.545 vs 0.696, 0.333 vs 0.560, 0.556 vs 0.558, 0.643 vs 0.812).

**The shrinkage is removed; the estimate is not better.** At each method's best-F1 point
(`diagnostics/estimation_error_best_f1.py`, same 800 datasets):

| $p$ | penalty | $\|\hat M-M^*\|_F/\|M^*\|_F$ | $\sum\lvert\hat M_{ij}\rvert/\sum\lvert M^*_{ij}\rvert$ on selected true edges | rel. error on selected true edges | true edges selected / total | false edges |
|---|---|---|---|---|---|---|
| 10 | lasso | 0.98 | **0.56** | 1.05 | 14.4 / 22.5 | 12.2 |
| 10 | MCP | 1.04 | **1.03** | 1.22 | 10.8 / 22.5 | 10.3 |
| 10 | SCAD | 0.82 | 0.60 | 0.88 | 11.7 / 22.5 | 9.6 |
| 20 | lasso | 0.72 | **0.31** | 0.76 | 29.1 / 47.4 | 34.5 |
| 20 | MCP | 0.79 | **0.56** | 0.76 | 21.0 / 47.4 | 24.2 |
| 20 | SCAD | 0.77 | 0.43 | 0.81 | 24.5 / 47.4 | 26.6 |

MCP does exactly what it is designed to do — the selected true edges are no longer shrunk
(ratio 1.03 at $p=10$ against the lasso's 0.56) — and yet their relative error is no smaller
(1.22 vs 1.05) and the overall error no smaller either. The bias it removes is replaced by
variance: on a design with a $p(p-1)/2$-dimensional null space, an entry that is free of
shrinkage absorbs whatever its collinear partners would have carried. SCAD, which shrinks almost
as much as the lasso at the best-F1 point, has the smallest overall error at $p=10$.

Two cheap follow-ups would pin this down and are not yet run: (i) a $\gamma$ sweep
($\gamma \in \{3, 10, 30, 100\}$; as $\gamma \to \infty$ both penalties become the lasso, so the
curves must meet it, and the question is how large $\gamma$ has to be), and (ii) the ncvreg
convention (`--convention ncvreg`, docs/NONCONVEX.md §2), which measures $\gamma$ relative to
each coordinate's curvature $v_{ij}$ and so keeps every one-dimensional sub-problem convex. Both
are ~10 min per setting at $p=10$. These are the natural next experiment for S1b.

### 3.2 S2 — the three penalties on the log-likelihood and Frobenius losses, $p=10$

### 3.3 Across losses

## 4. Cost and the cluster budget

**Direct loss (S1b).** Per 100-$\lambda$ path, textbook convention, `tol = 1e-8`, Apple M2, six
shards running at once (so the laptop numbers carry some contention):

| penalty | 800 datasets at $p=10,20$ | ratio to the lasso | full S1 grid (11,200 datasets) |
|---|---|---|---|
| lasso (cluster run, S1) | 0.84 CPU-h | 1 | 38.6 CPU-h |
| MCP | 4.9 CPU-h | ≈ 6× | ≈ 230 CPU-h |
| SCAD | 6.1 CPU-h | ≈ 7× | ≈ 280 CPU-h |

The ratios match the per-path benchmark in docs/NONCONVEX.md §7 (5× / 6.5× without contention).
On 64 cluster cores the full MCP grid is about 4 h wall-clock plus the tail, SCAD about 5 h;
`--time=06:00:00` in `cluster/s1_array.sbatch` is too tight for the $p=50$ shards of either —
raise it to 12 h or use 128 shards.

**Covariance losses (S2).** *(filled in from `runs/s2_pilot_p10/timing.txt` once the pilot has
finished.)*

## 5. Verification of the comparisons

Checks run on the finished runs (scripts inline in the session; the reusable ones are under
`simulations/diagnostics/`), all passed:

| check | result |
|---|---|
| **Same datasets.** $M^*$ stored in the shards is bit-identical for the same $(p,k,C,\text{rep})$ across the S1 cluster run, S1b and S2; $\lambda_{\max}$ is bit-identical across penalties within a loss (800/800 direct, 160/160 per covariance loss). | pass |
| **Same settings.** `config_json` of every run: seed 20260922, $N=1000$, 100 $\lambda$'s, ratio $10^{-4}$, correlation input, diagonal unpenalised, textbook convention, $\gamma=3$ / $3.7$, `tol = 1e-8`. | pass |
| **End to end.** Re-running three datasets (lasso, MCP, SCAD; $p=10$ and $20$) from the seed reproduces the stored confusion counts exactly — including the cluster's lasso counts on the laptop — and `aggregate_s1.metrics_from_counts` equals `metrics.evaluate_path` to $10^{-12}$ on the CSV rows. | pass |
| **Stationarity, direct loss.** On the refit paths the first-order violation (`penalties.stationarity`) is $\le 2\cdot10^{-7}$ at every $\lambda$, $\le 1.2\cdot10^{-3}$ relative to $\lambda$, for all three penalties. | pass |
| **Convergence, covariance losses.** Of 16,000 $\lambda$'s per cell, 11 (loglik lasso), 0 (MCP), 1 (SCAD) and 67 (Frobenius lasso) hit the 50,000-step cap, all with violation $\le 10^{-5}$; the median violation is $10^{-8}$. | pass |
| **Not the optimiser.** `skglm`'s coordinate descent gives the same MCP metrics as our solver (§3.1). | pass |
| **Metric construction vs. non-nested paths.** `auc_roc`/`aupr` integrate the operating points in path order (the R recipe). MCP paths are in fact *more* nested than lasso paths (0.6–1.0 vs 6–10 support reversals per path). Recomputing both areas from the points sorted by fpr / recall changes them by $\le 0.002$ (`auc`) and $\le 0.011$ (`aupr`), and the MCP − lasso gap by $\le 0.002$. | pass |
| **Pairing in `compare_runs.py`.** Keys $(p,k,C,\text{rep})$; $z = \bar d / \mathrm{se}(\bar d)$ with `ddof=1`; "better" is a strict inequality. | by construction |

## 6. Reproducing

```bash
# S1b pilot (direct loss): 2 penalties x 800 datasets, 3 shards each, ~1 h on 6 cores
for pen in MCP SCAD; do for i in 0 1 2; do
  python simulations/run_s1_shard.py --shard $i --n-shards 3 --p 10 20 --reps 25 \
      --penalty $pen --out-dir runs/s1b_pilot_p10-20/$pen/s1_shards &
done; done; wait
for pen in MCP SCAD; do python simulations/aggregate_s1.py --in-dir runs/s1b_pilot_p10-20/$pen/s1_shards; done

# S2 pilot (Varando's losses): runs/s2_pilot_p10/run_pilot.sh  (6 combos x 160 datasets, 8 shards)

# figures
python simulations/plot_penalties.py --reps 25 --p 10 20 \
    --run "lasso=runs/s1_dettling_reproduction" \
    --run "MCP=runs/s1b_pilot_p10-20/MCP" --run "SCAD=runs/s1b_pilot_p10-20/SCAD" \
    --title "Direct loss: lasso vs. MCP vs. SCAD" --out runs/s1b_pilot_p10-20/figures/penalties_direct.png
```

The full-scale versions are one `sbatch cluster/s1_array.sbatch <run> --loss L --penalty P` per
cell (docs/REPRODUCTION.md §2.4); §4 gives the budget per cell.
