# Are the repository's default solvers correct, and as accurate as the packages? (12 October 2026)

*Question (Joon, 12 October): is the default optimisation solver implemented correctly, and is its
accuracy comparable to that of package solvers? Checked for the three solvers the campaign runs,
at the campaign's settings (tolerance $10^{-8}$, 100-λ grid down to $10^{-4}\lambda_{\max}$,
warm-started paths, standardised data from the campaign's generator, true $C = 2I$, $C$ in the
fit $2I$ or rescaled). Scripts and R helpers: `solver_accuracy/` next to this note; the commands
are in §7.*

| loss and penalty | solver (repository) | where |
|---|---|---|
| direct, lasso | FISTA with adaptive restart (Beck & Teboulle 2009; O'Donoghue & Candès 2015) | `src/gclm/solvers/proxgrad.py::solve_fista` |
| direct, MCP / SCAD | monotone accelerated proximal gradient (Li & Lin 2015, Algorithm 1) | `proxgrad.py::_solve_mapg` |
| log-likelihood (all penalties) | accelerated proximal gradient with Barzilai–Borwein steps and backtracking | `src/gclm/solvers/covariance.py::solve`, `method="apg"` |

## 0. Answer

- **The implementations are correct.** The three algorithms are implemented as published
  (§5). The one defect is the one already listed as A1 in the code review
  (`code_review_121026.md`): the log-likelihood solver's step test uses an absolute slack, so it
  sometimes stops early or runs to its iteration cap.
- **Direct lasso: more accurate than glmnet, ncvreg and skglm at their defaults, and as accurate
  as ncvreg and skglm run at $10^{-12}$, in 20 to 40 times less time.** Measured against the exact
  lasso solution (certified by its KKT conditions at every λ), the campaign's FISTA run is within
  $2.4\times10^{-7}$ of the optimal objective; its support differs from the exact one at 0 to 9 of
  100 λ (median 0 at $p = 10$, 2 at $p = 20$), one entry each, which leaves the oracle $F_1$
  unchanged and moves AUPR by at most 0.0006. The packages at their defaults miss the exact
  support at 35 to 87 of 100 λ. At tolerance $10^{-10}$ FISTA is exact at every λ, for 1.6 times
  the time.
- **Direct MCP and SCAD: as stationary as the packages at tight tolerances, but nonconvex paths
  are sensitive.** At $10^{-8}$ the monotone APG ends within $3\times10^{-4}$ (relative) of the
  objective of the same path run to $10^{-14}$, at the same stationary points up to 2 of 100 λ,
  in 13 of the 14 paths per penalty (7 data sets, both path orders). In one data set the dense → sparse path
  starts from a slightly inexact lasso solution and follows another branch at 43 to 46 of 100 λ,
  with path metrics unchanged to 0.001; at $10^{-10}$ that disappears. ncvreg and skglm, run
  tightly, are as stationary as ours but sit at different stationary points at up to 98 of 100 λ:
  coordinate descent and proximal gradient follow different branches. Their path metrics differ
  from ours by up to 0.05 per path (for MCP with ncvreg never upwards); no difference is large
  enough to reverse a finding.
- **Log-likelihood loss: more accurate than Varando's own package.** At $p = 10$ the solver
  reached a stationarity residual below $1.5\times10^{-7}$ at every λ, and polishing every
  solution with Newton's method changes the support at no more than one λ per path (AUPR moved by
  0.010 in one path of 16, by 0.003 in another, oracle $F_1$ never). At $p = 20$ with the rescaled
  $C$, where the stored campaign shows the most capped solves, 0 to 5 solves per path hit the cap
  and polishing changes the support at up to 4 λ, with oracle $F_1$ and AUPR unchanged.
  Varando's `gclm` package stalls: even with no stopping slack it ends with residuals up to 0.23
  and an objective above ours at every λ.

## 1. How accuracy was measured

- **Direct lasso (convex): against the exact solution.** A path run to $10^{-15}$ gives the active
  set and signs at each λ; the optimum then solves a linear system on that set, and the KKT
  conditions certify it as the global minimiser. All 1,200 λ (12 data sets) passed: signs
  consistent and KKT violation below $10^{-8}\max(1, \lambda)$ ($3.5\times10^{-14}$ in the pilot data
  set). Reported: the relative objective gap, the largest coefficient error, the number of
  λ whose sign pattern differs from the exact one, and the change in the path's oracle $F_1$ and
  AUPR against the truth.
- **MCP and SCAD (nonconvex): stationarity and agreement.** There is no global optimum to compare
  with, and two correct solvers can end at different stationary points. Reported: the first-order
  stationarity residual divided by λ; the objective and sign pattern relative to the same solver's
  path run to $10^{-14}$, or, for a package, relative to our path run to $10^{-14}$ in the
  package's convention (ncvreg measures MCP's concavity relative to each coordinate's curvature;
  `convention="ncvreg"` solves that problem).
- **Log-likelihood (nonconvex): Newton polishing.** Each returned solution is refined at the same λ
  by the repository's active-set Newton method to a residual of $10^{-10}$, which stays in the
  same basin. Differences between the two measure the solver's error.
- **Packages at their defaults and tight.** glmnet (`thresh` $10^{-7}$; and the repository's
  ladder from $10^{-14}$), ncvreg's `ncvfit` (`eps` $10^{-5}$, 1,000 iterations; and $10^{-12}$),
  skglm `AndersonCD` (tol $10^{-4}$; and $10^{-12}$), CVXPY with the interior-point solver
  Clarabel (defaults), Varando's `gclm` (`eps` 0.01, 100 iterations; and `eps` 0 or $10^{-10}$ with
  up to $10^6$ iterations). Packages fit the same paths with warm starts; small entries
  ($<10^{-12}$) are set to zero as in `lasso_path`.
- **Data.** $p = 10$: $n = 10^3, 10^4$, both $C$ in the fit, two replicates each (8 data sets);
  $p = 20$: one replicate of each (4 data sets); true $C = 2I$, $k = 2$, seeds as in the campaign.
  Timings are wall-clock on the laptop with other benchmarks running alongside; ratios are
  meaningful, absolute values only roughly.

## 2. Direct loss, lasso

| solver | time per path | worst relative objective gap | λ with a different sign pattern (of 100) | worst change in oracle $F_1$ | worst change in AUPR |
|---|---|---|---|---|---|
| *$p = 10$, 8 data sets* | | | | | |
| **FISTA, tol $10^{-8}$ (campaign)** | 1.2 s | $2.4\times10^{-7}$ | 0–7 (median 0) | 0 | 0.0006 |
| FISTA, tol $10^{-10}$ | 1.9 s | $2.0\times10^{-12}$ | 0 | 0 | 0 |
| glmnet, default | 0.7 s | $1.1\times10^{-2}$ | 71–84 | 0.012 | 0.039 |
| glmnet, repository ladder ($10^{-14}$ …) | 0.9 s | $7.9\times10^{-4}$ | 0–64 (median 39) | 0 | 0.051 |
| ncvreg, default | 0.2 s | $8.5\times10^{-4}$ | 41–63 | 0 | 0.024 |
| ncvreg, eps $10^{-12}$ | 22.6 s | $7.8\times10^{-7}$ | 0–4 | 0 | 0.0002 |
| skglm, default | 0.4 s | $5.5\times10^{-4}$ | 35–61 | 0 | 0.023 |
| skglm, tol $10^{-12}$ | 45.4 s | $1.4\times10^{-7}$ | 0 | 0 | 0 |
| CVXPY + Clarabel, default | 0.4 s | $6.6\times10^{-8}$ | 44–77 (no exact zeros) | 0.025 | 0.054 |
| *$p = 20$, 4 data sets* | | | | | |
| **FISTA, tol $10^{-8}$ (campaign)** | 3.5 s | $4.5\times10^{-8}$ | 1–9 (median 2) | 0 | < 0.0001 |
| FISTA, tol $10^{-10}$ | 5.9 s | $1.2\times10^{-11}$ | 0 | 0 | 0 |
| glmnet, default | 1.0 s | $1.1\times10^{-2}$ | 80–87 | 0.018 | 0.006 |
| glmnet, repository ladder | 5.1 s | $1.1\times10^{-3}$ | 35–66 | 0 | 0.002 |
| ncvreg, default | 1.7 s | $1.1\times10^{-3}$ | 55–66 | 0.009 | 0.003 |
| skglm, default | 2.4 s | $7.4\times10^{-4}$ | 42–57 | 0 | 0.004 |
| CVXPY + Clarabel, default | 5.0 s | $2.8\times10^{-8}$ | 83–91 (no exact zeros) | 0.014 | 0.039 |

ncvreg and skglm at $10^{-12}$ take minutes per path at $p = 20$ and were run at $p = 10$ only.

**Reading.** Against the coordinate-descent packages at their defaults, the campaign's FISTA run is
three to five orders of magnitude closer to the optimal objective and misses the support at about ten
times fewer λ; only the interior-point solver gets closer in the objective, and it returns no exact
zeros. FISTA's remaining error comes
from its stopping rule, a coefficient change below $10^{-8}$ per iteration: with a step of
$1/L$ and a nearly singular design (the design $A(\hat\Sigma)$ has rank $p(p+1)/2$ of $p^2$), a
small step does not yet mean a small distance to the optimum, so the support can still be one
entry off. glmnet cannot reach high accuracy on this design: at tight thresholds its coordinate
descent stops converging and the repository's ladder falls back to looser ones. The interior-point
solver is accurate in the objective but returns no exact zeros, so its supports depend on a
threshold and are not comparable.

## 3. Direct loss, MCP and SCAD

$p = 10$, 7 data sets (the eighth was not run). Residual: first-order stationarity residual
divided by λ, worst and median over the 100 λ of all paths. Reference: the same path run to
$10^{-14}$ — our own solver for the first rows; for a package, our solver in the package's
convention.

| penalty | solver | time per path | worst residual / λ | median | objective relative to the reference | λ at a different stationary point (of 100) | change in oracle $F_1$ | change in AUPR |
|---|---|---|---|---|---|---|---|---|
| MCP | **ours, sparse → dense, tol $10^{-8}$ (campaign)** | 4.1 s | $1.3\times10^{-3}$ | $1.1\times10^{-5}$ | $[0, +1.8\times10^{-7}]$ | 0 | 0 | 0 |
| | **ours, dense → sparse, tol $10^{-8}$ (campaign)** | 12.3 s | $1.2\times10^{-3}$ | $6.9\times10^{-6}$ | $[-5.0, +5.9]\times10^{-2}$ | 0–46 (one data set; else 0–1) | 0 | ≤ 0.0004 |
| | ours, both orders, tol $10^{-10}$ | 8.5 / 46.1 s | $1.9\times10^{-5}$ | $1.2\times10^{-7}$ | $< 2\times10^{-8}$ | 0 | 0 | 0 |
| | ncvreg, default | 0.2 s | 0.13 | $3.9\times10^{-3}$ | $[-0.14, +0.05]$ | 0–99 | −0.045 – 0 | −0.045 – 0 |
| | ncvreg, eps $10^{-12}$ | 0.3 s | $2.1\times10^{-8}$ | $6.2\times10^{-10}$ | $[-0.14, +0.05]$ | 0–98 | −0.045 – 0 | −0.044 – 0 |
| | skglm, default | 0.6 s | 0.67 | $6.3\times10^{-4}$ | $[-0.02, +0.03]$ | 1–71 | 0 | −0.050 – +0.019 |
| | skglm, tol $10^{-12}$ | 0.1 s | $6.7\times10^{-9}$ | $1.7\times10^{-10}$ | $[-0.01, +0.03]$ | 0–70 | 0 | −0.001 – 0 |
| SCAD | **ours, sparse → dense, tol $10^{-8}$ (campaign)** | 5.5 s | $1.6\times10^{-3}$ | $1.2\times10^{-5}$ | $[0, +4.5\times10^{-8}]$ | 0 | 0 | 0 |
| | **ours, dense → sparse, tol $10^{-8}$ (campaign)** | 14.9 s | $1.2\times10^{-3}$ | $8.2\times10^{-6}$ | $[-4.8, +6.3]\times10^{-2}$ | 0–43 (one data set; else 0–2) | 0 | ≤ 0.0006 |
| | ours, both orders, tol $10^{-10}$ | 42.5 / 54.7 s | $1.8\times10^{-5}$ | $1.1\times10^{-7}$ | $< 10^{-8}$ | 0 | 0 | 0 |
| | ncvreg, default | 0.2 s | 0.39 | $4.7\times10^{-3}$ | $[-0.04, +0.25]$ | 0–87 | 0 – +0.014 | −0.042 – +0.037 |
| | ncvreg, eps $10^{-12}$ | 0.3 s | $5.9\times10^{-8}$ | $6.5\times10^{-10}$ | $[-0.04, +0.25]$ | 0–87 | 0 – +0.014 | −0.017 – +0.037 |

skglm has no weighted SCAD, so it cannot leave the diagonal unpenalised and is not compared for SCAD.
The packages walk the path sparse → dense only, as in their own path functions.

**The one data set where our dense → sparse path depends on the tolerance** ($n = 10^4$, rescaled
$C$) is the one where the lasso at $10^{-8}$ missed the exact support at 7 λ (§2). The dense → sparse
path starts from the lasso solution at the smallest λ, computed at the same tolerance; from that
slightly inexact start it follows another branch at 46 of 100 λ, with objectives up to 6 % lower or
higher than on the branch of the exact start. Its oracle $F_1$ is unchanged and its AUPR moves by
0.0004 (MCP) and 0.0006 (SCAD). At $10^{-10}$ the path is identical to the $10^{-14}$ one.

**Reading.** Our solver is accurate at the campaign's tolerance: apart from the one dense → sparse
start above, tightening it to $10^{-14}$ moves the objective by at most $3\times10^{-4}$ (relative)
and changes the support at 0 to 2 of 100 λ, leaving the path metrics unchanged. The packages at their defaults are much less stationary
(residual up to 0.39λ for ncvreg, 0.67λ for skglm). At tight tolerances they are as stationary as
ours, but along most of the path they converge to *other* stationary points: the objective can be
lower or higher than ours by several percent. Which stationary point a nonconvex path follows is
part of the estimator, along with the penalty and the path order. For the findings this means: with
ncvreg's coordinate descent, MCP sparse → dense scores equal or lower AUPR than with our solver (by up
to 0.044), and SCAD lower or higher (−0.017 to +0.037); the lasso's lead over both (0.05 to 0.15 in
oracle $F_1$) is larger than these differences.

## 4. Log-likelihood loss

Our solver at the campaign's settings (tol $10^{-8}$, at most 50,000 iterations per λ), then every
solution polished by Newton's method at the same λ. "Residual" is the first-order stationarity
residual; "false converged" counts solves reported as converged with a residual above $10^{-7}$.

| data set | path | time | worst residual | λ with residual $> 10^{-6}$ | capped solves | false converged | λ whose support changes when polished | change in oracle $F_1$ / AUPR |
|---|---|---|---|---|---|---|---|---|
| $p = 10$, $n = 10^3$, $C = 2I$ | lasso | 19 s | $7.3\times10^{-8}$ | 0 | 0 | 0 | 1 | 0 / +0.010 |
| | MCP sparse → dense | 13 s | $5.6\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP dense → sparse, exact fit | 13 s | $6.3\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP dense → sparse, lasso | 29 s | $4.6\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| $p = 10$, $n = 10^3$, rescaled $C$ | lasso | 27 s | $1.3\times10^{-7}$ | 0 | 0 | 2 | 1 | 0 / 0 |
| | MCP sparse → dense | 3 s | $8.3\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP dense → sparse, exact fit | 12 s | $1.4\times10^{-7}$ | 0 | 0 | 2 | 0 | 0 / 0 |
| | MCP dense → sparse, lasso | 32 s | $4.7\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| $p = 10$, $n = 10^4$, $C = 2I$ | lasso | 38 s | $9.3\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP sparse → dense | 5 s | $4.7\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP dense → sparse, exact fit | 13 s | $8.4\times10^{-8}$ | 0 | 0 | 0 | 1 | 0 / +0.003 |
| | MCP dense → sparse, lasso | 47 s | $5.0\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| $p = 10$, $n = 10^4$, rescaled $C$ | lasso | 23 s | $1.2\times10^{-7}$ | 0 | 0 | 1 | 0 | 0 / 0 |
| | MCP sparse → dense | 2 s | $7.9\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| | MCP dense → sparse, exact fit | 16 s | $1.5\times10^{-7}$ | 0 | 0 | 2 | 0 | 0 / 0 |
| | MCP dense → sparse, lasso | 28 s | $5.3\times10^{-8}$ | 0 | 0 | 0 | 0 | 0 / 0 |
| $p = 20$, $n = 10^4$, rescaled $C$ | lasso | 248 s | $3.4\times10^{-6}$ | 1 | 1 | 1 | 0 | 0 / 0 |
| | MCP dense → sparse, exact fit | 294 s | $3.7\times10^{-7}$ | 0 | 3 | 1 | 0 | 0 / 0 |
| | MCP dense → sparse, lasso | 608 s | $3.1\times10^{-6}$ | 1 | 5 | 0 | 4 | 0 / 0 |
| $p = 20$, $n = 10^3$, rescaled $C$ | lasso | 406 s | $1.7\times10^{-6}$ | 1 | 5 | 0 | 2 | 0 / 0 |
| | MCP dense → sparse, exact fit | 184 s | $1.0\times10^{-7}$ | 0 | 0 | 1 | 0 | 0 / 0 |
| | MCP dense → sparse, lasso | 585 s | $6.8\times10^{-8}$ | 0 | 0 | 0 | 1 | 0 / 0 |

Two changes of AUPR at $p = 10$ come from a single λ each. There the Newton step moved the solution by
0.02 from a point whose residual was already $7\times10^{-8}$: the solution lies in a nearly flat
valley of the likelihood, so it is poorly determined by the data, not badly computed.

**Varando's `gclm` (R, v0.0.1)** on the same lasso paths, warm-started along the same grid, against
our polished solutions:

| data set | `gclm` setting | time | worst residual | median residual | λ with a different support (of 100) | change in oracle $F_1$ / AUPR |
|---|---|---|---|---|---|---|
| $p = 10$, $n = 10^3$, $C = 2I$ | default (eps 0.01, 100 iterations) | 0.3 s | 0.26 | $2.2\times10^{-2}$ | 99 | 0 / +0.018 |
| | eps $10^{-10}$, $10^6$ iterations | 0.6 s | 0.26 | $2.3\times10^{-3}$ | 99 | +0.107 / +0.086 |
| $p = 10$, $n = 10^3$, rescaled $C$ | default | 0.2 s | 0.29 | $4.9\times10^{-2}$ | 99 | −0.097 / −0.084 |
| | eps $10^{-10}$, $10^6$ iterations | 0.9 s | 0.29 | $5.0\times10^{-3}$ | 99 | 0 / −0.027 |
| $p = 10$, $n = 10^4$, $C = 2I$ | default | 0.2 s | 0.27 | $2.0\times10^{-2}$ | 99 | −0.047 / −0.024 |
| | eps $10^{-10}$, $10^6$ iterations | 0.6 s | 0.26 | $3.1\times10^{-3}$ | 99 | +0.072 / +0.036 |
| $p = 10$, $n = 10^4$, rescaled $C$ | default | 0.2 s | 0.28 | $4.4\times10^{-2}$ | 99 | −0.157 / −0.112 |
| | eps $10^{-10}$, $10^6$ iterations | 0.8 s | 0.28 | $5.0\times10^{-3}$ | 99 | −0.061 / −0.075 |

`gclm` stops when an iteration lowers the objective by less than `eps` or when its backtracking
finds no decrease. With `eps` 0 and up to 200,000 iterations (`probe_gclm.py`, 10 of the λ of the
first data set) it still stops after 60 iterations at the second λ with residual 0.23, 3 edges
where ours has 6, and an objective 0.09 higher; its objective is above ours at every λ, and its
residuals range from $1.5\times10^{-6}$ to 0.23 against ours of $10^{-10}$ to $10^{-7}$.

**How often the solver stops short, over the whole stored campaign** (100 λ per path, all
log-likelihood cells of waves 3 and 7; the shards store the residual and the iteration count of
every solve):

| $C$ in the fit | solves with residual $> 10^{-6}$ | with residual $> 10^{-4}$ | that hit the 50,000-iteration cap |
|---|---|---|---|
| $2I$ | 0 – 0.10 % per cell | 0 – 0.01 % | 0 – 0.8 % |
| rescaled, $p = 10$ | 0.3 – 1.7 % | 0.01 – 0.45 % | 0.4 – 2.2 % |
| rescaled, $p = 20$ | 0.9 – 4.9 % | 0.01 – 1.2 % | 1.2 – 6.7 % |

**Reading.** On the data sets checked the solver's imperfect stopping changes supports rarely and
path metrics not at all. It is more accurate than Varando's `gclm`, whose backtracking line search
gives up long before stationarity (§0). Bug A1 is still worth fixing: up to 7 % of the solves run
50,000 iterations for nothing, and a "converged" flag can be wrong.

## 5. Is the implementation correct?

- **FISTA** (`solve_fista`): gradient $2R\hat\Sigma$ (checked against finite differences and the
  design matrix in the tests), step $1/L$ with $L = (2\lambda_{\max}(\hat\Sigma))^2 \ge \lVert A\rVert_2^2$,
  weighted soft-thresholding with the diagonal unpenalised, momentum $(t_k-1)/t_{k+1}$, and the
  gradient-based restart of O'Donoghue & Candès, $(Y_k - M_{k+1})^\top(M_{k+1} - M_k) > 0$. All as
  published. Stopping rule: largest coefficient change below `tol`.
- **Monotone APG** (`_solve_mapg`): Li & Lin's Algorithm 1 line by line — extrapolation from the
  last accelerated candidate, two proximal steps of length $0.99/L < 1/L$, the lower objective
  kept, $t_0 = 0$, $t_1 = 1$. The MCP and SCAD proximal maps were checked against brute force on
  23,694 cases in the code review. Stopping rule as for FISTA.
- **Covariance-loss APG** (`covariance.solve`): accelerated proximal gradient with backtracking from
  a Barzilai–Borwein trial step, every iterate kept stable, the momentum step accepted only if the
  objective does not rise. Correct except for finding A1: the curvature and descent tests add an
  absolute slack of $10^{-13}\max(1, |F|)$, which accepts uncontrolled steps once the step's
  curvature term falls below it.
- The packages agree with ours where they should: ncvreg and skglm at $10^{-12}$ reach the exact
  lasso solution; on MCP and SCAD every package solution checked is a stationary point of the same
  objective.

## 6. Recommendations

1. **Direct loss: use `tol = 1e-10` for new runs** (S1Config.tol). It makes the lasso exact at every
   λ for 1.6 times the time, and every MCP and SCAD path, in both orders, then agrees with a
   $10^{-14}$ run (2 to 4 times the time for dense → sparse). No re-run of the campaign is needed:
   the measured effect of $10^{-8}$ on any path metric is below 0.001.
2. **Log-likelihood loss: fix A1** (stop on the KKT residual of the accepted iterate; a
   rounding-safe step test). The measured effect on supports is small, so re-running the
   log-likelihood cells is not required for the current findings.
3. **In the thesis, define the nonconvex estimators by penalty, path order and algorithm**, and
   report the comparison of §3: with coordinate descent the MCP and SCAD paths follow other
   stationary points, without changing the conclusions.
4. **Keep the packages as validation backends, not as the default.** At their defaults they are
   less accurate than our solver on this problem; tightened, they are 20 to 40 times slower and, for
   MCP and SCAD, define a different estimator.

## 7. Reproduce

```bash
cd next_steps/121026/solver_accuracy
python3 bench_lasso.py     # direct lasso, all solvers vs the exact solution (~1 h; ncvreg/skglm tight at p = 10 only)
python3 bench_mcp.py       # direct MCP / SCAD vs tight runs and the packages (p = 10 part: ~2 h)
python3 bench_loglik.py    # log-likelihood paths, Newton polishing, Varando's gclm (~1.5 h)
python3 probe_gclm.py      # gclm's stall at 10 lambdas, with and without its stopping slack
```

Needs R with glmnet, ncvreg and gclm, and Python with cvxpy (Clarabel) and skglm.
