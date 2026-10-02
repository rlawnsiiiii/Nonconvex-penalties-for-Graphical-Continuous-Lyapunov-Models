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

Figures under `runs/s1b_pilot_p10-20/figures/` and `runs/s2_pilot_p10/figures/`; every number below
is a paired comparison on identical datasets (`compare_runs.py`, CSVs next to each run).

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
dimension $p(p-1)/2$, so it is never strongly convex. The null space is $\{W\hat\Sigma^{-1}: W
\text{ skew-symmetric}\}$, and for a correlation matrix $\hat\Sigma \approx I$ it is close to the
skew-symmetric matrices themselves: adding $t$ to $M_{ij}$ and subtracting $t$ from $M_{ji}$
changes the fit by almost nothing. **The weakly identified quantity in this model is the
direction of an edge**, $i \to j$ versus $j \to i$, and the two penalties resolve it differently.

**It is not the optimiser.** `skglm`'s coordinate descent (an independent implementation of
textbook MCP) returns the same `max_f1` / `auc` as our monotone APG on five S1 datasets, with
objectives within $3\cdot10^{-4}$ along the whole path (one dataset: MCP 0.638 vs lasso 0.596;
the other four: 0.545 vs 0.696, 0.333 vs 0.560, 0.556 vs 0.558, 0.643 vs 0.812).

**Bias, not variance — seen with $M^*$ fixed.** The Figure 5 metrics cannot separate bias from
variance (one dataset per $M^*$). `diagnostics/bias_variance_fixed_mstar.py` fixes one $M^*$
($p=10$, $k=2$, 14 true edges, $|M^*_{ij}|$ from 0.11 to 1.77), draws 200 datasets, and fits each
down to $\lambda = 0.1\lambda_{\max}$, where $F_1$ peaks
(`runs/s1b_pilot_p10-20/bias_variance_fixed_mstar.txt`):

| | true edges selected | mean bias | mean variance | mean MSE | false entries selected | variance along the flattest direction of $\Gamma_{SS}$ | along the steepest |
|---|---|---|---|---|---|---|---|
| lasso | 76 % | +0.08 | 0.009 | 0.175 | 16.5 | 0.0006 | 0.0051 |
| MCP | 48 % | +0.23 | 0.009 | **0.383** | 18.0 | 0.0025 | 0.0033 |
| SCAD | 62 % | +0.12 | 0.046 | 0.302 | 16.6 | 0.0018 | 0.0141 |

MCP's error is twice the lasso's, and it is **bias**: its per-edge variance equals the lasso's.
The variance inflation one expects from a slope-less penalty is there, but confined to the two
flattest curvature directions of the loss on the true support (ratio MCP/lasso 4.1 and 3.0;
about 1 elsewhere) and small in absolute terms. Per edge:

| $\lvert M^*_{ij}\rvert$ | lasso: selected / bias | MCP: selected / bias | SCAD: selected / bias (variance) |
|---|---|---|---|
| 1.77, edge (3,0) | 100 % / +1.17 | **0 %** / +1.77 | 10 % / +1.64 (0.14) |
| 1.33 | 100 % / −0.66 | 98 % / −0.51 | 81 % / −0.67 (0.11) |
| 1.00, edge (7,1) | 100 % / +0.28 | **1 %** / +0.99 | 66 % / +0.35 (0.23) |
| 0.79, edge (2,6) | 98 % / +0.31 | **1 %** / +0.78 | 78 % / +0.24 (0.09) |
| 0.76 … 0.53 (4 edges) | 100 % | 95–99 % | 95–100 % |
| ≤ 0.38 (6 weak edges) | 10–99 % | 9–69 % | 8–96 % |

Three of the four strongest edges are selected by the lasso in every dataset and by MCP in none.
And the false entries MCP selects in $\ge 90$ % of the datasets are exactly their transposes —
(0,3), (1,7), (6,2) — whose design columns correlate 0.61, 0.74 and 0.81 with the dropped edge
and $\approx 0$ with anything else. MCP has not lost these edges; it has **reversed them**, in 199
of 200 datasets. SCAD reverses them in some datasets and not in others, which is where its large
per-edge variance comes from.

**The mechanism: a barrier against reversing an edge.** Along a path from $\lambda_{\max}$
downwards, $M_{ij}$ and $M_{ji}$ have nearly the same gradient (their design columns correlate
0.6–0.8; the curvature of the loss along the reversal direction $e_{ij}-e_{ji}$ has median 0.75
on true edges, against $\ge 2$ along a single coordinate), and which one enters first is decided
by the geometry of $\Sigma(M^*)$, not by noise — hence the reproducibility. What happens next
differs by penalty:

- *Lasso.* Along the reversal segment $(a-t,\,t)$ the $\ell_1$ norm is constant,
  $|a-t|+|t| = |a|$, so the split between the two directions is decided by the loss alone. When
  the loss is nearly indifferent the lasso keeps **both** directions, at reduced size: at the
  best-F1 point it holds both directions of 4–5 true pairs per dataset at $p=10$ and 12 at $p=20$
  (§3.4, "hedged"), and the true direction wins as $\lambda$ decreases — lasso paths show 6–10
  support reversals per 100 $\lambda$'s, which are these corrections. $F_1$ and AUC reward the
  hedge, since a true positive plus a false positive beats a false positive plus a false negative.
- *MCP.* Once the first-chosen direction has $|a| \ge \gamma\lambda$ its penalty is flat, so
  moving mass $t$ to the other direction costs $\lambda t$ on the entering entry and refunds
  nothing on the exiting one: every reversal faces a first-order barrier of $\lambda$, which the
  loss, with its small curvature along the swap, rarely overcomes. The orientation picked early is
  frozen (0.6–1.0 reversals per path), and the two representations are two local minima of
  almost the same objective. It is the **kink** of the penalty at zero combined with the flat
  part, not the concavity as such: only 13 % of true edges have reversal curvature below
  $1/\gamma$, so the problem is not that the objective is nonconvex along the swap.
- *SCAD* still refunds part of the slope on $[\lambda, \gamma\lambda]$, so its barrier is
  smaller — hence its intermediate position on every metric and its larger variance (it
  reverses in some datasets and not in others).

For Varando's losses the start is even less informative: the gradient at the diagonal fit is
$\hat\Sigma - I$, exactly symmetric, so both directions enter with identical gradient (§3.2).

**Better optimisation does not fix it.** Started from the least-squares fit on the *true* support
at the continuation path's best-F1 $\lambda$, MCP reaches $F_1 = 0.68$ instead of 0.54 (40
`C_ID` datasets, $p=10$) — the right basin exists and is much better — but that basin has the
lower MCP objective in only 42 % of the datasets. So the objective does not reliably prefer the
truth at $n = 1000$, and a solver that finds lower objectives finds worse supports, which is what
the Newton experiment of docs/LIKELIHOOD.md §5 showed from the other side.

**Estimation error across many $M^*$, at each method's best-F1 point**
(`diagnostics/estimation_error_best_f1.py`, the same 800 datasets as the metrics):

| $p$ | penalty | $\|\hat M-M^*\|_F/\|M^*\|_F$ | $\sum\lvert\hat M_{ij}\rvert/\sum\lvert M^*_{ij}\rvert$ on selected true edges | rel. error on selected true edges | true edges selected / total | false edges |
|---|---|---|---|---|---|---|
| 10 | lasso | 0.98 | **0.56** | 1.05 | 14.4 / 22.5 | 12.2 |
| 10 | MCP | 1.04 | **1.03** | 1.22 | 10.8 / 22.5 | 10.3 |
| 10 | SCAD | 0.82 | 0.60 | 0.88 | 11.7 / 22.5 | 9.6 |
| 20 | lasso | 0.72 | **0.31** | 0.76 | 29.1 / 47.4 | 34.5 |
| 20 | MCP | 0.79 | **0.56** | 0.76 | 21.0 / 47.4 | 24.2 |
| 20 | SCAD | 0.77 | 0.43 | 0.81 | 24.5 / 47.4 | 26.6 |

Decomposing the error on the selected true edges ($p=10$): rescaling the lasso's values by
$1/0.56$, i.e. removing its shrinkage entirely, lowers its error only from 1.05 to 0.69; the rest
is misallocation — the correlation between $\hat M_{ij}$ and $M^*_{ij}$ over those edges is 0.77,
6 % have the wrong sign. MCP has no bias to remove and a slightly larger misallocation (0.72,
correlation 0.76, 8.5 % wrong signs); the true edges the lasso misses are weak (mean
$|M^*_{ij}| = 0.51$), the ones MCP misses include strong ones (0.74).

**Re-read with the mechanism in mind:** MCP removes the shrinkage of the edges it keeps
(ratio 1.03 vs 0.56), but its selected set is the wrong one for the strong edges, so the relative
error on "selected true edges" is no better and the overall error is worse; the "misallocation"
in that table (correlation 0.76–0.77 between $\hat M_{ij}$ and $M^*_{ij}$, 6–8 % wrong signs) is
the same reversal seen across many $M^*$. SCAD, which shrinks almost as much as the lasso at its
best-F1 point, has the smallest overall error at $p=10$.

Two cheap follow-ups would pin this down and are not yet run: (i) a $\gamma$ sweep
($\gamma \in \{3, 10, 30, 100\}$; as $\gamma \to \infty$ both penalties become the lasso, so the
curves must meet it, and the question is how large $\gamma$ has to be), and (ii) the ncvreg
convention (`--convention ncvreg`, docs/NONCONVEX.md §2), which measures $\gamma$ relative to
each coordinate's curvature $v_{ij}$ and so keeps every one-dimensional sub-problem convex. Both
are ~10 min per setting at $p=10$. These are the natural next experiment for S1b.

### 3.2 S2 — the three penalties on the log-likelihood and Frobenius losses, $p=10$

**The same picture on both of Varando's losses.** Paired differences (penalty − lasso, same
loss) over the 40 datasets per $C$ choice, range over the four choices
(`runs/s2_pilot_p10/{loss}_{penalty}/paired_vs_{loss}_lasso.csv`):

| loss | penalty | `max_f1` | `auc` | `aupr` | penalty better on |
|---|---|---|---|---|---|
| log-likelihood | MCP | −0.14 … −0.08 ($z$ −7 … −5) | −0.16 … −0.06 ($z$ −9 … −4) | −0.11 … −0.04 ($z$ −7 … −3) | 3–23 % of datasets |
| log-likelihood | SCAD | −0.11 … −0.07 ($z$ −8 … −5) | −0.13 … −0.06 ($z$ −9 … −5) | −0.09 … −0.04 ($z$ −7 … −5) | 7–23 % |
| Frobenius | MCP | −0.13 … −0.09 ($z$ −8 … −7) | −0.14 … −0.06 ($z$ −9 … −4) | −0.08 … −0.04 ($z$ −7 … −4) | 5–23 % |
| Frobenius | SCAD | −0.11 … −0.08 ($z$ −7 … −6) | −0.14 … −0.07 ($z$ −9 … −5) | −0.08 … −0.05 ($z$ −9 … −6) | 5–23 % |

The sizes are those of the direct loss at $p=10$ (§3.1), with the lasso baselines themselves
close across losses (next section). Figures: `penalties_loglik.png`, `penalties_frobenius.png`.

The mechanism of §3.1 carries over unchanged, because it lives in the map $M \mapsto \Sigma(M)$,
not in the loss: for these losses the fibre $\{M : \Sigma(M) = \hat\Sigma\} = M_0 + W\hat\Sigma^{-1}$
is *exactly* flat (docs/LIKELIHOOD.md §3), so swapping $M_{ij}$ against $M_{ji}$ costs even less
than for the direct loss, and a concave penalty again freezes whichever direction the path meets
first. The covariance losses are themselves nonconvex, so here the lasso is not globally convex
either — which is why the lasso's advantage does not grow on these losses (it would if convexity
were the only thing at work): what the lasso keeps is the slope $\lambda$ along the flat
directions, which the concave penalties give up once an entry passes $\gamma\lambda$.

### 3.3 Across losses

With the penalty fixed, the three losses are close, and the log-likelihood is the best of them
for the lasso (paired, same datasets, 40 per point; `paired_vs_direct_*.csv`):

| penalty | log-likelihood − direct | Frobenius − direct |
|---|---|---|
| lasso | `max_f1` +0.01 … +0.02, `auc` +0.005 … +0.017 ($z$ up to +4), **`aupr` +0.03 … +0.06** ($z$ +2 … +4) | `max_f1` −0.03 … +0.02 (mixed), `auc` −0.013 … +0.008, `aupr` ±0.04 |
| MCP | all $\lvert\text{diff}\rvert \le 0.02$, $\lvert z\rvert \le 0.8$ | `max_f1` −0.07 … 0 ($z$ down to −3), `auc` −0.04 … +0.01 |
| SCAD | `max_f1` −0.05 … +0.01 ($z$ down to −3), `auc` ±0.02 | `max_f1` −0.09 … −0.02 ($z$ down to −5), `auc` −0.04 … 0 |

Two things to take from this. First, Varando & Hansen's finding reproduces: with the lasso, the
log-likelihood loss recovers the support slightly better than the direct lasso path (their
`mloglik-inf` vs `lasso`), most visibly in `aupr`. Second, the ordering of the *penalties* is the
same under every loss, and the gaps between penalties (0.06–0.16) are several times the gaps
between losses (≤ 0.06): the choice of penalty matters more than the choice of loss, and in the
direction opposite to the thesis hypothesis. Figures: `losses_lasso.png`, `losses_MCP.png`,
`losses_SCAD.png`.

All results in this section are at $p=10$ with 40 datasets per point; §4 explains why the larger
$p$ go to the cluster.

### 3.4 Skeleton vs. orientation: where the directed errors come from

`gclm.metrics.orientation_breakdown` splits the directed confusion counts exactly into pair-level
categories (tested): a true single-direction edge is **correct** (estimate has that direction
only), **reversed** (the other one only) or **hedged** (both); a true 2-cycle is recovered
**both** or **half**; the rest are skeleton errors (missed pairs, false pairs).
`diagnostics/orientation.py` reads the stored best-F1 estimates of every run.

**At the best-F1 point** (paired differences vs. the lasso on the same loss, pooled over the four
$C$ choices; `runs/*/orientation_best_f1*.txt`):

| loss | penalty | $p$ | directed $F_1$ | **skeleton $F_1$** | orientation accuracy | orientation recall |
|---|---|---|---|---|---|---|
| direct | MCP | 10 | −0.096 ($z$ −19) | −0.027 ($z$ −6) | −0.123 ($z$ −14) | −0.194 ($z$ −27) |
| direct | MCP | 20 | −0.076 ($z$ −20) | −0.026 ($z$ −9) | −0.129 ($z$ −19) | −0.192 ($z$ −33) |
| direct | SCAD | 10 | −0.053 ($z$ −14) | −0.022 ($z$ −5) | −0.073 ($z$ −8) | −0.113 ($z$ −16) |
| direct | SCAD | 20 | −0.031 ($z$ −12) | −0.019 ($z$ −8) | −0.061 ($z$ −10) | −0.080 ($z$ −18) |
| log-likelihood | MCP | 10 | −0.112 ($z$ −12) | −0.030 ($z$ −4) | −0.148 ($z$ −10) | −0.199 ($z$ −15) |
| log-likelihood | SCAD | 10 | −0.090 ($z$ −12) | −0.032 ($z$ −4) | −0.104 ($z$ −8) | −0.150 ($z$ −13) |
| Frobenius | MCP | 10 | −0.115 ($z$ −15) | −0.042 ($z$ −5) | −0.119 ($z$ −8) | −0.184 ($z$ −13) |
| Frobenius | SCAD | 10 | −0.096 ($z$ −14) | −0.027 ($z$ −4) | −0.108 ($z$ −8) | −0.165 ($z$ −12) |

*Orientation accuracy*: among detected single-direction true edges on which the estimate committed
to one direction, the fraction oriented correctly. *Orientation recall*: among detected
single-direction true edges, the fraction whose true direction is present (hedges count). The
skeleton loss is small (0.02–0.04); three quarters of the directed loss is orientation. The
categories behind it, `C_ID`, $p=10$, means per dataset (direct loss; 100 datasets):

| | correct | reversed | hedged | 2-cycle: both | 2-cycle: half | missed pairs | false pairs |
|---|---|---|---|---|---|---|---|
| lasso | 5.9 | 1.4 | **4.6** | **1.4** | 1.5 | 4.2 | 4.1 |
| MCP | 7.7 | **3.5** | 0.2 | 0.1 | 2.9 | 4.7 | 4.4 |
| SCAD | 7.3 | 2.7 | 1.3 | 0.6 | 2.3 | 4.9 | 3.9 |

MCP commits to one direction per pair: it hedges on 0.2 pairs where the lasso hedges on 4.6,
reverses 3.5 true edges where the lasso reverses 1.4, and recovers both directions of a true
2-cycle 0.05 times per dataset where the lasso does 1.4 times. The 2-cycles are a structural
handicap: Dettling's DGP draws each direction independently with probability $k/p$, so 10–40 % of
true edges at $p=10$ have their reverse present too, and a penalty that keeps one direction per
pair cannot recover them. The same signature appears on both of Varando's losses.

**Along the whole path** (refits, direct loss, $p=10$, 160 datasets, paired vs. the lasso;
`runs/s1b_pilot_p10-20/orientation_path_p10.csv`):

| penalty | directed `max_f1` / `auc` / `aupr` | **skeleton** `max_f1` / `auc` / `aupr` |
|---|---|---|
| MCP | −0.094 ($z$ −12) / −0.104 ($z$ −13) / −0.056 ($z$ −9) | −0.017 ($z$ −4) / −0.010 ($z$ −2) / −0.001 ($z$ −0.2) |
| SCAD | −0.054 ($z$ −9) / −0.102 ($z$ −13) / −0.056 ($z$ −12) | −0.008 ($z$ −2) / −0.011 ($z$ −3) / −0.010 ($z$ −3) |

On the undirected graph the three penalties are within 0.02 of each other, and on `C_ID` MCP's
skeleton AUC and AUPR are marginally above the lasso's (0.822 vs 0.817, 0.787 vs 0.774). The same
refit on the **log-likelihood loss** (`runs/s2_pilot_p10/orientation_path_loglik_p10.csv`):

| penalty | directed `max_f1` / `auc` / `aupr` | **skeleton** `max_f1` / `auc` / `aupr` |
|---|---|---|
| MCP | −0.112 ($z$ −12) / −0.114 ($z$ −14) / −0.081 ($z$ −10) | −0.035 ($z$ −6) / −0.026 ($z$ −5) / −0.022 ($z$ −4) |
| SCAD | −0.090 ($z$ −12) / −0.106 ($z$ −13) / −0.072 ($z$ −13) | −0.026 ($z$ −5) / −0.023 ($z$ −5) / −0.020 ($z$ −6) |

There the skeleton gap is real but small (0.02–0.035), three to four times smaller than the
directed one. **The nonconvex penalties find the skeleton about as well as the lasso — on a par
on the direct loss, slightly behind on the log-likelihood — and lose on orientation.**

### 3.5 Example 2 as a positive control

Dettling's Example 2 (`gclm.data.examples`), $n = \infty$, i.e. $\hat\Sigma = \Sigma(M^*)$:

- **Path $G_1$** (1→2→3→4→5): all three penalties recover it exactly (`max_f1` = `auc` = 1).
- **5-cycle $G_2$** (plus 5→1): all three give `max_f1` 0.800, `auc` 0.833 — identical — and the
  best-F1 support is $\{1\to2, 2\to3, 3\to4, 4\to5, \mathbf{1\to5}\}$: the skeleton is perfect,
  one edge is reversed. This is Dettling's irrepresentability failure, and it is an orientation
  failure.
- **Under MCP the truth is the better solution, and the path does not reach it.** $M^*$ is a
  stationary point of the MCP objective (violation $3\cdot10^{-15}$) with a lower objective than
  the continuation solution for $\lambda \le 0.1$ (0.0748 vs 0.0862 at $\lambda = 0.100$). At
  $\lambda = 0.03$, reversing the one wrong edge of the continuation solution and re-solving lands
  exactly on $M^*$ ($\|\hat M - M^*\|_\infty = 3\cdot10^{-11}$).

So in the population the nonconvex objective prefers the truth and continuation never finds it;
at $n = 1000$ on random drift matrices (the oracle-start probe in §3.1) the objective does not
even reliably prefer it. Both point the same way: what is missing is a move that *reverses* an
edge, not a penalty with less bias.

### 3.6 What this changes about the next steps

1. **Hold the large-$p$ runs of the current cells.** The pilot settles $\gamma = 3$ / $3.7$ at
   $p = 10, 20$; scaling to $p = 50$ (230–280 CPU-h per direct-loss cell) would measure the same
   thing more precisely.
2. **The $\gamma$ sweep is still informative**, because the barrier against a reversal is
   $\min(|a|/\gamma, \lambda)$ for MCP — larger $\gamma$ keeps more entries in the refund zone
   and should approach the lasso from below. The ncvreg convention is not: with $v_{ij} \ge 2$ it
   amounts to a textbook $\gamma / v_{ij} \le 1.5$, i.e. *more* concave.
3. **Target orientation.** Two cheap candidates: a search with add / delete / *reverse* moves
   started from the lasso path (S3 with a reversal move, which advances planned work), and
   MCP plus a small ridge term (Mnet; the ridge rewards splitting mass across the two correlated
   directions, so it can hedge like the lasso while debiasing large entries).
4. **Report skeleton and orientation metrics alongside the directed ones**, and a tuning-based
   comparison (BIC / eBIC) next to the path maxima, which are oracle tuning.
5. **Map where nonconvexity pays off**: sweep $n$ (10³ to ∞) with the Example 2 harness and
   random drift matrices with edge weights bounded away from zero, with and without 2-cycles.


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

**Covariance losses (S2).** Per 100-$\lambda$ path at $p=10$, `apg`, `tol = 1e-8`, eight shards
in parallel on the same 8-core laptop (so with contention), from the `seconds` stored in the
shards:

| loss | penalty | mean | median | max | 160 datasets | $\lambda$'s hitting the 50,000-step cap |
|---|---|---|---|---|---|---|
| log-likelihood | lasso | 33 s | 23 s | 552 s | 1.5 CPU-h | 11 / 16,000 |
| log-likelihood | MCP | 4.4 s | 2.7 s | 55 s | 0.2 CPU-h | 0 |
| log-likelihood | SCAD | 5.5 s | 3.8 s | 56 s | 0.25 CPU-h | 1 |
| Frobenius | lasso | 108 s | 58 s | 1,202 s | 4.8 CPU-h | 67 |
| Frobenius | MCP | 63 s | 11 s | 1,661 s | 2.8 CPU-h | 379 |
| Frobenius | SCAD | 74 s | 14 s | 1,816 s | 3.3 CPU-h | 393 |

For comparison the direct lasso costs 3.2 s per path at $p=10$ on the cluster. The lasso is the
expensive penalty on these losses (its slope keeps pulling inside the flat valley; MCP/SCAD stop
pulling past $\gamma\lambda$), the Frobenius loss costs 3× the log-likelihood, and the
distribution has a long tail: a few datasets take 20–30 min at $p=10$. Every $\lambda$ that hit
the step cap still had a first-order violation $\le 1.5\cdot10^{-5}$ (§5). The cost per
iteration is $O(p^3)$ and the iteration count grows with $p$ too, so $p=20$ is of the order of
10× and $p=50$ far beyond that; the full S1 grid on these losses is a cluster job with a budget
to be measured first (§6).

## 5. Verification of the comparisons

Checks run on the finished runs (scripts inline in the session; the reusable ones are under
`simulations/diagnostics/`), all passed:

| check | result |
|---|---|
| **Same datasets.** $M^*$ stored in the shards is bit-identical for the same $(p,k,C,\text{rep})$ across the S1 cluster run, S1b and S2; $\lambda_{\max}$ is bit-identical across penalties within a loss (800/800 direct, 160/160 per covariance loss). | pass |
| **Same settings.** `config_json` of every run: seed 20260922, $N=1000$, 100 $\lambda$'s, ratio $10^{-4}$, correlation input, diagonal unpenalised, textbook convention, $\gamma=3$ / $3.7$, `tol = 1e-8`. | pass |
| **End to end.** Re-running three datasets (lasso, MCP, SCAD; $p=10$ and $20$) from the seed reproduces the stored confusion counts exactly — including the cluster's lasso counts on the laptop — and `aggregate_s1.metrics_from_counts` equals `metrics.evaluate_path` to $10^{-12}$ on the CSV rows. | pass |
| **Stationarity, direct loss.** On the refit paths the first-order violation (`penalties.stationarity`) is $\le 2\cdot10^{-7}$ at every $\lambda$, $\le 1.2\cdot10^{-3}$ relative to $\lambda$, for all three penalties. | pass |
| **Convergence, covariance losses.** Of 16,000 $\lambda$'s per cell, 11 / 0 / 1 (log-likelihood lasso / MCP / SCAD) and 67 / 379 / 393 (Frobenius lasso / MCP / SCAD) hit the 50,000-step cap, all with first-order violation $\le 1.5\cdot10^{-5}$; the median violation is $10^{-8}$. | pass |
| **Not the optimiser.** `skglm`'s coordinate descent gives the same MCP metrics as our solver (§3.1). | pass |
| **Metric construction vs. non-nested paths.** `auc_roc`/`aupr` integrate the operating points in path order (the R recipe). MCP paths are in fact *more* nested than lasso paths (0.6–1.0 vs 6–10 support reversals per path). Recomputing both areas from the points sorted by fpr / recall changes them by $\le 0.002$ (`auc`) and $\le 0.011$ (`aupr`), and the MCP − lasso gap by $\le 0.002$. | pass |
| **Pairing in `compare_runs.py`.** Keys $(p,k,C,\text{rep})$; $z = \bar d / \mathrm{se}(\bar d)$ with `ddof=1`; "better" is a strict inequality. | by construction |
| **Orientation categories.** `orientation_breakdown` reproduces the directed and skeleton confusion counts exactly on random patterns (`test_orientation_breakdown_decomposes_the_directed_confusion`); Example 2 and the curvature figures were re-derived independently from a scratch re-implementation and match (`max_f1` 0.800 / `auc` 0.833 on the 5-cycle, 13 % of true edges below $1/\gamma$). | pass |

## 6. Reproducing

```bash
# S1b pilot (direct loss): 2 penalties x 800 datasets, 3 shards each, ~1.5 h on 6 cores
for pen in MCP SCAD; do for i in 0 1 2; do
  python simulations/run_s1_shard.py --shard $i --n-shards 3 --p 10 20 --reps 25 \
      --penalty $pen --out-dir runs/s1b_pilot_p10-20/$pen/s1_shards &
done; done; wait
for pen in MCP SCAD; do python simulations/aggregate_s1.py --in-dir runs/s1b_pilot_p10-20/$pen/s1_shards; done

# S2 pilot (Varando's losses, p = 10): runs/s2_pilot_p10/run_pilot.sh
#   6 cells x 160 datasets, 8 shards each, aggregated per cell; ~2.5 h on 8 cores

# paired comparisons and figures
python simulations/compare_runs.py --baseline runs/s1_dettling_reproduction \
    --run runs/s1b_pilot_p10-20/MCP --reps 25 --p 10 20
python simulations/plot_penalties.py --reps 25 --p 10 20 \
    --run "lasso=runs/s1_dettling_reproduction" \
    --run "MCP=runs/s1b_pilot_p10-20/MCP" --run "SCAD=runs/s1b_pilot_p10-20/SCAD" \
    --title "Direct loss: lasso vs. MCP vs. SCAD" --out runs/s1b_pilot_p10-20/figures/penalties_direct.png
python simulations/diagnostics/estimation_error_best_f1.py --reps 25 --p 10 20 \
    --run lasso=runs/s1_dettling_reproduction --run MCP=runs/s1b_pilot_p10-20/MCP --run SCAD=runs/s1b_pilot_p10-20/SCAD
```

**Larger $p$ on the cluster.** The S2 cells at $p \ge 20$ are not laptop work (§4). One array job per
cell, each in its own folder, restricted to the $p$ values wanted and to the replicates that pair
with the S1 run (docs/REPRODUCTION.md §2.4 for the environment):

```bash
mkdir -p logs
for loss in loglik frobenius; do
  for pen in lasso MCP SCAD; do
    sbatch cluster/s1_array.sbatch "runs/s2_${loss}_${pen}_p20-30" --loss "$loss" --penalty "$pen" --p 20 30 --reps 25
  done
done
```

Start with $p = 20, 30$ and read the per-dataset `seconds` from the shards before committing to
$p = 40, 50$: at $p=10$ a single lasso path on the log-likelihood already takes up to 9 min and on
the Frobenius loss up to 20 min (§4), the cost per iteration grows as $p^3$ and the iteration count
grows with $p$ as well. Set `--time` in `s1_array.sbatch` accordingly (the default 6 h was chosen
for the direct lasso) and use more than 64 shards so that no single shard carries several slow
datasets. The direct-loss MCP/SCAD grid (S1b at full scale) is the same call without `--loss`.

**γ sweep (not run).** `runs/s1b_gamma_sweep/run_gamma_sweep.sh` runs the follow-up of §3.1 —
$\gamma \in \{3, 10, 30, 100\}$ for MCP and SCAD plus the ncvreg convention — on `C_ID`, $p=10$,
100 datasets per setting, through `run_s1.py`; about 40 min on 6 laptop cores, and its per-dataset
CSVs drop straight into `compare_runs.py`.
