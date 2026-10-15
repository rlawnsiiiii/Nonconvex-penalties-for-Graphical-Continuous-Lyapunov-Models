# Starting from the lasso: the rescaled $C$, dense → sparse paths, LLA, the adaptive lasso

How the estimators added for the campaign of October 2026 are defined, implemented and validated.
The reasons for them are in [`../simulations/VERDICTS.md`](../simulations/VERDICTS.md) (verdicts 3 to 7)
and in [`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md);
the theory behind "the standard path fixes a direction too early" is in
[`../next_steps/051026/orientation_lock_in.md`](../next_steps/051026/orientation_lock_in.md).

Code: `src/gclm/data/simulate.py` (`estimation_volatility`), `src/gclm/solvers/path.py`
(`lasso_path(direction="up")`, `lla_path`, `adaptive_lasso_path`), `src/gclm/objective/penalties.py`
(`lla_weights`). Tests: `tests/test_direction_cscale.py` (5), `tests/test_lla_adaptive.py` (16).

```bash
python simulations/run_s1.py --penalty MCP --direction up                    # MCP dense -> sparse
python simulations/run_s1.py --penalty SCAD --method lla --c-scale variance  # SCAD by LLA, rescaled C
python simulations/run_s1.py --method adaptive --c-scale variance            # adaptive lasso
```

All of this is for the **direct loss** with the default `fista` solver and an unpenalised diagonal.

---

## 1. The problem these estimators answer

The standard way to compute an MCP or SCAD path is continuation from the sparse end: start at
$\lambda_{\max}$ with the empty graph, lower $\lambda$, and start each problem from the previous
solution. In a GCLM this fixes the direction of an edge at the moment its pair first enters, when
the fit contains almost nothing else, and the nonconvex penalty then keeps that direction
(S3a; `orientation_lock_in.md`). The lasso is not affected in the same way: its problem is convex,
so it does not depend on where it was started, and when the direction is unclear it keeps both.

The three estimators below use the nonconvex penalty differently. They **start from a lasso
solution**, which still contains both directions of uncertain edges, and prune from there.

| estimator | start | then | convex steps | depends on the order of $\lambda$ |
|---|---|---|---|---|
| MCP / SCAD, sparse → dense (standard) | empty graph at $\lambda_{\max}$ | MCP / SCAD at each smaller $\lambda$, warm-started | no | yes |
| **MCP / SCAD, dense → sparse** | lasso solution at $\lambda_{\min}$ | MCP / SCAD at each larger $\lambda$, warm-started | no | yes |
| **MCP / SCAD by LLA** | lasso solution at the same $\lambda$ | two weighted lassos | yes | no |
| **adaptive lasso** | lasso solution at $\lambda_{\min}$ | one weighted lasso path | yes | no |

---

## 2. The rescaled $C$ (`--c-scale variance`)

**What it is.** The simulations standardise the data: every variable is divided by its standard
deviation $s_i$. That is a change of variables, and it changes the model:

$$M\Sigma+\Sigma M^\top+C=0
\quad\Longrightarrow\quad
\tilde M R+R\tilde M^\top+\tilde C=0,\qquad
\tilde M_{ij}=M_{ij}\,\frac{s_j}{s_i},\quad \tilde C=D^{-1}CD^{-1},\quad D=\mathrm{diag}(s).$$

$R$ is the correlation matrix. The support of $\tilde M$ is that of $M$. If the data were generated
with $C=2I$, then $\tilde C = 2\,\mathrm{diag}(1/s_i^2)$: the **rescaled $C$**.

**Code.** `draw_instance(..., return_scale=True)` returns $s$ next to the standardised covariance,
and

```python
estimation_volatility(scale, "identity")   # 2 I            -- Dettling's pipeline (default)
estimation_volatility(scale, "variance")   # 2 diag(1/s^2)  -- the rescaled C
```

(`src/gclm/data/simulate.py`) returns the matrix handed to the estimator. `S1Config.c_scale` and
`--c-scale` select it in both drivers. With unstandardised data $s=1$ and the two coincide.

**What it assumes.** That $C=2I$ on the scale on which the data were measured. That is exact for
Dettling's `C_ID` setting and only partly right for the other three (campaign note §2.2). It is a
device for the simulation study, not a recipe for real data.

**Test.** `test_standardized_population_model_is_misspecified_with_identity_and_exact_with_variance`:
at $n=\infty$ the true support fits the standardised covariance with zero loss under the rescaled
$C$ and with positive loss under $2I$, and $D^{-1}MD$ solves the rescaled equation.

---

## 3. MCP / SCAD dense → sparse (`--direction up`)

**Definition.** On the grid $\lambda_1<\dots<\lambda_{100}=\lambda_{\max}$:

1. $\hat M^{\text{lasso}}(\lambda_1)$: the lasso solution at the smallest $\lambda$. It is dense
   (close to the minimum-$\ell_1$ exact solution of the Lyapunov equation) and well defined,
   because the lasso is convex.
2. For $t=1,\dots,100$: solve the MCP / SCAD problem at $\lambda_t$ with the monotone accelerated
   proximal gradient of `docs/NONCONVEX.md` §4, started from the solution at $\lambda_{t-1}$
   (from the lasso start for $t=1$).
3. At $\lambda_{\max}$ the path is set to the diagonal fit, which is a stationary point there, so
   both orders end at the empty graph.

The objective is the same as for the standard path. Only the order differs, and with a nonconvex
penalty the order decides which stationary point is reached.

**Code.** `lasso_path(sigma, c, penalty="MCP", direction="up")` in `src/gclm/solvers/path.py`: the
block that begins with `if direction == "up" and penalty != "lasso":`. For the lasso `direction`
has no effect.

**Tests** (`tests/test_direction_cscale.py`): every point of the path is a stationary point of the
MCP objective; the last point is diagonal; the path differs from the standard one; the default is
unchanged; for the lasso both orders return the same path.

**Cost.** One lasso path plus an MCP / SCAD path that starts dense: 2 to 5 times the standard
MCP / SCAD path (measured, campaign note §3.4).

### 3.1 The exact fits behind the dense start (`src/gclm/objective/exact_fits.py`)

The lasso at $\lambda_1$ is an approximation of a well-defined object: the exact solutions of
$M\hat\Sigma+\hat\Sigma M^\top+C=0$ form the affine space $\{M_0+W\hat\Sigma^{-1}: W \text{ skew}\}$
with $M_0=-\tfrac12 C\hat\Sigma^{-1}$ (`dense_fit`), of dimension $p(p-1)/2$ (Dettling, Homs,
Améndola, Drton & Hansen 2023, Lemma 27), and as $\lambda\to 0$ the lasso tends to the member with
the smallest off-diagonal $\ell_1$ norm (the theory note of 8 October 2026, §2). The module added on
14 October 2026 for the dense-start experiments of the plan of 13 October (§1.1) gives members of
that space exactly, the diagonal free as everywhere else in the code:

| function | member | method |
|---|---|---|
| `min_l2_exact_fit(sigma_hat, c)` | smallest off-diagonal Frobenius norm | linear least squares in the skew coordinates $w$, $M(w)=M_0+\sum_k w_k B_k\hat\Sigma^{-1}$ with $B_k=E_{ij}-E_{ji}$ (`skew_basis`, `exact_fit_jacobian`) |
| `min_l1_exact_fit(sigma_hat, c)` | smallest off-diagonal $\ell_1$ norm ("basis pursuit", the limit of the lasso) | linear program in $(w,t)$, $\lvert M(w)_{ij}\rvert\le t_{ij}$, `scipy.optimize.linprog` with HiGHS at tolerance $10^{-9}$; 0.2 s at $p=20$, 1 s at $p=30$, 10–26 s at $p=50$ |
| `noisy_truth_start(m_true, tau, rng)` | $M^*+\tau\,N(0,1)$ off the diagonal | a ceiling for the dense starts, not an estimator; on standardised data pass `standardized_drift(M*, s)` |
| `exact_fit_residual(m, sigma_hat, c)` | $\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F$ | for tests and diagnostics |

**Uniqueness of the minimum-$\ell_1$ fit.** The theory's *BP condition* (8 October note, §4) asks
that the standardised truth $\tilde M$ be the *unique* minimiser. `min_l1_exact_fit(..., probes=k,
return_info=True)` decides this exactly: at the solution, split the off-diagonal entries into the
nonzero set $N$ (signs $s$) and the zero set $Z$; for a small step $d$ the norm changes by
$s^\top J_N d+\lVert J_Z d\rVert_1$ exactly, so the minimiser is unique iff this is positive for
every $d\ne 0$ (a sharp minimum, Burke & Ferris 1993), i.e. iff the polyhedral cone
$\{d: s^\top J_N d+\lVert J_Z d\rVert_1\le 0\}$ is $\{0\}$. Random linear objectives maximised
over that cone in the unit box are 0 iff it is, and positive with probability one otherwise
(`L1Fit.face_extent`). The constraints are homogeneous, so no tolerance on the minimal norm enters;
only the classification of entries as zero (`zero_tol`). On the campaign graphs the extent is either
exactly 0 or above 0.047. The earlier idea, re-solving with the norm held at its minimum up to a
slack, flagged generic sample instances as non-unique (the slack is amplified along weakly sharp
minima) and was dropped.

**Regression (`tests/test_exact_fits.py::test_bp_condition_count_on_the_campaign_graphs`).** On the
thesis DGP at $p=10$ ($k=1,\dots,4$, 25 graphs each, $C=2I$, the campaign's seed), population
covariance on the correlation scale with the rescaled $C$: $\tilde M$ is the unique minimum-$\ell_1$
fit in **11 of 100** graphs (7 at $k=1$, 4 at $k=2$, none at $k=3,4$), the 7 October count
reproduced exactly. It attains the minimal norm in 17; in the other 6 the optimal face is flat (the
direction of an isolated edge is free) and which vertex the LP returns depends on the solver: HiGHS
1.15 returns $\tilde M$ in 14, the 8 October note counted 15. About 5 s.

**Tests** (`tests/test_exact_fits.py`, 12): the basis and Jacobian against an explicit construction
and full column rank; all three fits exact to $10^{-8}\lVert C\rVert_F$ at $p=5,10,20$ and distinct;
each minimum-norm member smallest in its norm by a strict margin; the LP against an independent
formulation in the $p^2$ entries with the design matrix as equality constraints; the uniqueness
probe on the flat two-node case, a sharp two-node case, a generic sample and a population graph with
an isolated edge; the noisy truth and the residual; refused inputs; the regression above.

---

## 4. MCP / SCAD by local linear approximation (`--method lla`)

**Definition.** At every $\lambda$ separately, with $P_\lambda$ the MCP or SCAD penalty:

1. start from the lasso solution $M^{(0)}=\hat M^{\text{lasso}}(\lambda)$ at that same $\lambda$;
2. for $s=1,2$: compute weights from the current estimate and solve a weighted lasso,

$$w_{ij}=\frac{P'_\lambda(|M^{(s-1)}_{ij}|)}{\lambda}\in[0,1],\qquad
M^{(s)}=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}| .$$

The weight is 1 at a zero entry and 0 beyond $\gamma\lambda$, where the penalty is flat: large
entries are no longer shrunk, small ones are penalised like in the lasso. Each step replaces the
concave penalty by its tangent at the current estimate, which is a convex problem (Zou & Li 2008).

**Why two steps from the lasso.** Fan, Xue & Zou (2014): if the start is close enough to the truth
and the nonzero entries are large enough, the first step gives the oracle estimator and the second
step reproduces it; the lasso is a suitable start under a restricted-eigenvalue condition. The
irrepresentability condition, which the lasso needs for support recovery and which fails for most
of these graphs, is not required. A fixed point of the iteration is a stationary point of the
MCP / SCAD objective.

**Code.** `lla_path` in `src/gclm/solvers/path.py`; the weights are `lla_weights` in
`src/gclm/objective/penalties.py`; each weighted lasso is `solve_fista(..., weights=w)`, started
from the current estimate. Nothing is carried from one $\lambda$ to the next.

**A caveat at the dense end.** The design $A(\hat\Sigma)$ has rank $p(p+1)/2$. When more entries
than that are unpenalised (the diagonal plus every entry beyond $\gamma\lambda$), the weighted
lasso has flat directions and more than one minimiser; the estimate is then the one FISTA reaches
from the lasso solution. This concerns the smallest $\lambda$ only, where the fit is dense anyway.

**Tests** (`tests/test_lla_adaptive.py`): the weights equal $P'/\lambda$; one step satisfies the
optimality conditions of its weighted lasso at every $\lambda$ and agrees with coordinate descent
on the explicit design where the problem is well posed; iterated to a fixed point it is stationary
for the nonconvex objective; the estimate at one $\lambda$ does not depend on the rest of the grid;
it ends at the diagonal fit and tends to the lasso as $\gamma\to\infty$.

**Cost.** About three lasso paths.

---

## 5. Adaptive lasso (`--method adaptive`)

**Definition** (Zou 2006, with a lasso pilot):

1. pilot $M^0=\hat M^{\text{lasso}}(\lambda_{\min})$, the dense end of the lasso path;
2. weights $w_{ij}=1/|M^0_{ij}|$, divided by the smallest one, so that the largest pilot entry has
   weight 1; entries with $M^0_{ij}=0$ are excluded;
3. the weighted lasso path

$$\hat M(\lambda)=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}|$$

on its own grid: 100 log-spaced values up to $\lambda^{w}_{\max}=\max_{i\ne j}|\nabla f(M_D)_{ij}|/w_{ij}$,
the smallest $\lambda$ at which the solution is the diagonal fit $M_D$.

It follows the same idea as the dense → sparse path (start dense, remove the weak entries first),
with a convex second step. It is the control for the question whether the nonconvex penalty is
needed, or only the dense start.

**Code.** `adaptive_lasso_path` in `src/gclm/solvers/path.py`. It returns an `AdaptiveLassoPath`,
which also holds the pilot and the weights (`inf` marks an excluded entry).

**Tests** (`tests/test_lla_adaptive.py`): the weights; the optimality conditions of the weighted
problem at every $\lambda$ and agreement with coordinate descent; excluded entries stay zero and
every support is contained in the pilot's; with equal weights it is the lasso path.

**Cost.** About two lasso paths.

### 5.1 Any matrix as the start or the pilot (`--up-start`, `--pilot`; 14 October 2026)

The dense → sparse paths and the adaptive lasso share one ingredient, a dense matrix to start
from, and until 14 October 2026 it was always the lasso's dense end (or, on the covariance losses,
Varando & Hansen's exact fit). Experiment E1 of the plan of 13 October asks which property of that
start matters, so the start became an argument: `lasso_path(..., direction="up", start=M)`,
`covloss_path(..., direction="up", start=M)`, `adaptive_lasso_path(pilot=M)` and
`adaptive_covloss_path(pilot=M)` take any `(p, p)` matrix; without it they compute what they always
did (`start=` equal to the lasso's dense end reproduces the stored paths bit for bit,
`tests/test_starts.py`). The path records the start it used (`LassoPath.start`,
`AdaptiveLassoPath.pilot`). The runner's `--up-start` (paths) and `--pilot` (adaptive and thresholded
lasso) choose among

| name | the matrix | property kept (plan §2.0) |
|---|---|---|
| `lasso` | the lasso solution at the smallest $\lambda$ (the default for the direct loss) | exact fit up to $O(\lambda)$, $\ell_1$-smallest, oriented |
| `exact` | $-\tfrac12 C\hat\Sigma^{-1}$ (Varando & Hansen; the default for the covariance losses) | exact fit only |
| `l1` | the minimum-$\ell_1$ exact fit, §3.1 | exact fit, $\ell_1$-smallest, oriented (the limit the lasso approximates) |
| `l2` | the minimum-Frobenius exact fit, §3.1 | exact fit, dense, both directions of a pair graded |
| `truth-<τ>` | the standardised truth $\tilde M$ plus $\tau\,N(0,1)$ off the diagonal | oriented only (a ceiling, not an estimator) |

Every cell with a dense start stores the start's own diagnostics per data set (`start_*` in the
shard: norms, Lyapunov residual, support size at $10^{-8}$, confusion counts and the orientation
breakdown of that support, i.e. the "dense-start content" of the 7 October note), and `campaign.py`
turns them into the `start_f1`, `start_recall`, `start_correct`, `start_hedged`, `start_reversed`, …
columns. The fully dense starts (`l2`, `exact`, `truth-τ`) hit the step cap of the MCP solver at a
few $\lambda$ in the flat valley of the dense end (`converged` flags them; `LIKELIHOOD.md` §4.3);
the `l1` start reproduces the lasso-started path on the test instance, as the theory predicts.

### 5.2 The thresholded lasso (`--method threshold`, `--method threshold-norefit`)

Experiment E2 asks whether the adaptive lasso's gain is in its pilot or in its reweighting, so the
simplest pruning of the same pilot is an estimator of its own: rank the pilot's off-diagonal
entries by size, keep those with $|M^0_{ij}| \ge t$, refit the kept support by least squares on the
direct loss with the diagonal free (`DirectRefit`, the closed form behind the score), and let the
score choose $t$. `thresholded_lasso_path(sigma, c, pilot=None, n_thresholds=100, refit="direct")`
computes the pilot exactly as `adaptive_lasso_path` does (pass the same `pilot` to both to make the
pairing explicit), lays a grid of 100 thresholds — 0 at the dense end, 98 values log-spaced between
the smallest and the largest pilot entry above $10^{-8}$, $\infty$ at the sparse end — and returns a
`ThresholdedLassoPath`, a `LassoPath` whose `lambdas` are the thresholds in increasing order, so
`evaluate_path`, `bic_along_path`, the selection and the search treat it like any path. The
supports are nested. `refit="none"` (`--method threshold-norefit`) keeps the pilot's values on the kept
support: the pilot's ranking as a path, whose `aupr` measures how well the dense end *orders* the
edges; `refit="loglik"` is available for the likelihood refit. In the shard the threshold path has
`lambda_max` = the largest pilot entry, the loss as `objective` (no penalty), `kkt` NaN and no
solver steps. Tests: `tests/test_threshold.py`; cost: one lasso path plus at most 100 closed-form
refits.

---

## 6. On the cluster

`simulations/run_s1_shard.py` takes the same options (`--c-scale`, `--direction`, `--method`,
and since 14 October 2026 `--up-start`, `--pilot`, `--weights`, `--w-min`, `--graph`).
With `--select bic` or `--select search` it also records, for each graph, the support at every
$\lambda$, the graph that the score selects on the path and the graph the greedy search of
[`SEARCH.md`](SEARCH.md) reaches from it. `cluster/submit_campaign.sh` defines the cells of the
campaign; wave 9 holds the mechanism experiments of the plan of 13 October 2026 in five groups
(`--group starts | threshold | betamin | class | starts-loglik`). Details, the list of cells and the
commands: campaign note §4 and §5, and [`REPRODUCTION.md`](REPRODUCTION.md) §2.7.

---

## 7. Why the rescaled $C$ helps these estimators, and not the lasso

*Numbers: `next_steps/051026/files/why_rescaling_helps.py` (C_ID graphs, $p = 10$: 20 graphs,
$p = 20$: 8) and the campaign tables (`runs/campaign/campaign_paired.csv`, 800 graphs per cell).*

### 7.1 $C$ is the right-hand side, not a tuning constant

The direct loss is $\tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2$. The drift matrices that fit
exactly are the solutions of the linear system

$$M\hat\Sigma+\hat\Sigma M^\top=-C ,$$

an affine space of dimension $p(p-1)/2$ (the design $A(\hat\Sigma)$ has rank $p(p+1)/2$ and
maps onto the symmetric matrices, so the system is consistent for every $C$). $C$ decides which
$M$ fit the data at all. Change $C$ and the whole solution set moves.

With the rescaled $C$ the true (standardised) drift matrix $\tilde M$ is in the set: at $n=\infty$
its misfit is 0. With $C=2I$ it is not: the best fit on the true support misses by 0.8 % of the
diagonal fit's loss at $p=10$ and 1.8 % at $p=20$ (`misfit` in the script; the table in the
campaign note §2.2 uses another normalisation and reads 1.4 % at $p=10$).

### 7.2 A diagonal error in $C$ is a dense error in $M$

Let $C$ be wrong by a diagonal $\Delta$. Moving from the true solution set to the wrong one means
adding to $M$ a solution $X$ of $X\hat\Sigma+\hat\Sigma X^\top=-\Delta$. If $\hat\Sigma$ were
diagonal, $X$ could be diagonal, and the unpenalised diagonal of $M$ would absorb the error for
free. It is not. For two nodes with correlation $\rho$ and $\Delta=\mathrm{diag}(\delta,0)$, writing
$X=\begin{pmatrix}a&b\\c&d\end{pmatrix}$, the three equations are $a+b\rho=-\delta/2$,
$c\rho+d=0$ and $a\rho+b+c+d\rho=0$, which give

$$b+c=\frac{\rho\,\delta}{2(1-\rho^2)} .$$

An error $\delta$ in $C_{11}$ forces off-diagonal mass of size $\rho\delta/(2(1-\rho^2))$ between
the two nodes: a false edge, or a changed true one, proportional to the correlation. In general
the smallest correction is $X=-A(\hat\Sigma)^{+}\mathrm{vec}(\Delta)$, and it is dense.

How large is it here? On standardised C_ID data $\Delta=2I-2\,\mathrm{diag}(1/s_i^2)$ with
$s_i^2$ the raw stationary variances, which range from 0.3 to 2.1 (10th to 90th percentile at
$p=10$), so $|\Delta_{ii}|$ has median 1.5: the same order as $C$ itself. Most of it is absorbed by
the diagonal of $M$; what leaks into the off-diagonal entries is the shift of §7.1: median entry
0.06 ($p=10$) and 0.03 ($p=20$), largest entry 1.0 and 1.8, against true edge weights with median
0.7 and 10th percentile 0.16. Small on most entries, as large as the weakest true edges on some,
larger than a typical true edge on a few.

### 7.3 Where each estimator makes its decision

- **The lasso** takes its best graph in the middle of the path, where the fit is a compromise with
  sparsity anyway. Only the *order* in which entries enter matters there, and a shift of this size
  hardly changes the order. In the campaign the lasso path's `max_f1` is unchanged by rescaling
  at $p=10$ (0.589 against 0.584 at $n=10^3$) and slightly worse at $p=20$ (0.531 against 0.507).
  Dettling's robustness of the lasso to a misspecified $C$ is this.
- **The dense-start estimators** (§3–§5) decide at the dense end: they take the lasso solution at
  the smallest $\lambda$, a point of the solution set, and keep its large entries. That works when
  the dense end is "truth plus a null-space component", which pruning can strip. With $C=2I$ it is
  truth plus null-space component plus the shift of §7.2, and the pruning drops weak true edges and
  keeps shifted false ones. Measured on the dense end (the sizes $|M^0_{ij}|$, which are exactly
  the adaptive lasso's weights and the dense → sparse paths' start):

  | C_ID, $n=\infty$ | ranking of true edges above non-edges by $\lvert M^0_{ij}\rvert$ (AUC) | true edges whose reversed entry is the larger one |
  |---|---|---|
  | $p=10$, $C=2I$ | 0.71 | 22 % |
  | $p=10$, rescaled $C$ | 0.76 | 14 % |
  | $p=20$, $C=2I$ | 0.88 | 10 % |
  | $p=20$, rescaled $C$ | 0.93 | 1 % |

  At $n=10^4$ the same pattern with smaller differences (0.66 → 0.73 and 0.86 → 0.87 in AUC).

### 7.4 Two mechanisms that add, and a graded response

The dense start helps on its own, with $C=2I$, because it avoids the early orientation lock-in of
the standard path (`next_steps/051026/orientation_lock_in.md`); that has nothing to do with $C$.
The correct $C$ then makes the dense end informative. Campaign numbers, MCP dense → sparse against
the lasso with the same $C$, `max_f1`, $p=20$, $n=10^4$:

| true $C$ | gain with $C=2I$ | gain with the rescaled $C$ | extra from rescaling | misfit removed by rescaling |
|---|---|---|---|---|
| `C_ID` | +0.077 | +0.154 | +0.077 | 100 % |
| `C_Random_Min_Diag` | +0.081 | +0.141 | +0.060 | 95 % |
| `C_Random_Diag` | +0.079 | +0.122 | +0.043 | 40 % |
| `C_Random_Full` | +0.004 | +0.042 | +0.038 | 30 % |

The extra gain follows how much of the misfit the rescaling removes (misfit figures: campaign note
§2.2).

### 7.5 A third channel: the score

The score's loss term, and with it the search, uses the likelihood of $\Sigma(M,C)$. With the wrong $C$ the
best-scoring graph is a different graph (S3b §9; `next_steps/051026/next_steps_051026.md` §2).
This is why lasso + search gains from rescaling (0.614 → 0.650 at $p=20$, $n=10^4$) although the
lasso path does not.

**In one sentence:** with the wrong $C$ the data say that no sparse graph fits; the lasso, which
never insists on an exact fit, is barely affected, while the estimators that start from an exact
fit and prune are pruning the wrong object.

## References

- Fan, J., Xue, L. & Zou, H. (2014). Strong oracle optimality of folded concave penalized
  estimation. *Annals of Statistics* 42(3), 819–849. LLA from a lasso start.
- Zou, H. & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models.
  *Annals of Statistics* 36(4), 1509–1533. The local linear approximation.
- Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA* 101(476), 1418–1429.
- Varando, G. & Hansen, N. R. (2020). Graphical continuous Lyapunov models. *UAI*. Their paths run
  from a dense start towards sparser solutions.
- Dettling, P., Drton, M. & Kolar, M. (2024). On the Lasso for graphical continuous Lyapunov
  models. *CLeaR*. The direct loss and the simulation design.
- Dettling, P., Homs, R., Améndola, C., Drton, M. & Hansen, N. R. (2023). Identifiability in
  continuous Lyapunov models. *SIAM Journal on Matrix Analysis and Applications* 44(4). Lemma 27:
  the exact fits $(W-C/2)\hat\Sigma^{-1}$, $W$ skew (§3.1).
- Burke, J. V. & Ferris, M. C. (1993). Weak sharp minima in mathematical programming. *SIAM
  Journal on Control and Optimization* 31(5), 1340–1359. A polyhedral function has a unique minimiser
  iff the minimum is sharp (§3.1, the uniqueness probe).
