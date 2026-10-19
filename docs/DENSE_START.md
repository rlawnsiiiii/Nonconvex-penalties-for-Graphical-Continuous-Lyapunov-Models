# Starting from the lasso: the rescaled $C$, dense → sparse paths, LLA, the adaptive lasso

Definitions, code and validation of the estimators added for the campaign of October 2026, all for the direct loss with the default `fista` solver and an unpenalised diagonal.

| | |
|---|---|
| reasons | [`simulations/VERDICTS.md`](../simulations/VERDICTS.md) verdicts 3 to 7; [`cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md) (the campaign note) |
| lock-in theory | [`orientation_lock_in_051026.md`](../claude_docs/orientation_lock_in_051026.md) |
| solver | monotone accelerated proximal gradient, `docs/NONCONVEX.md` §4 |

```text
src/gclm/
├── data/simulate.py          draw_instance(..., return_scale=True) -> s      estimation_volatility(scale, "identity" | "variance")
│                             standardized_drift(M*, s)                        S1Config.c_scale (src/gclm/config.py)
├── objective/penalties.py    lla_weights(m, lam, penalty, gamma)
├── objective/exact_fits.py   dense_fit   min_l2_exact_fit   min_l1_exact_fit(..., probes=k, return_info=True) -> L1Fit.face_extent
│                             noisy_truth_start   exact_fit_residual   skew_basis   exact_fit_jacobian
└── solvers/path.py           lasso_path(sigma, c, penalty, direction="up", start=M)      -> LassoPath.start
                              covloss_path(..., direction="up", start=M)
                              lla_path                                                     each step: solve_fista(..., weights=w)
                              adaptive_lasso_path(pilot=M)                                 -> AdaptiveLassoPath.pilot, .weights
                              adaptive_covloss_path(pilot=M)
                              thresholded_lasso_path(sigma, c, pilot=None, n_thresholds=100, refit="direct") -> ThresholdedLassoPath
src/gclm/solvers/search.py    DirectRefit   (closed-form least-squares refit behind the score)
simulations/run_s1.py, run_s1_shard.py      --c-scale --direction --method --up-start --pilot --weights --w-min --graph --select
tests/   test_direction_cscale.py (5)   test_lla_adaptive.py (16)   test_exact_fits.py (12)   test_starts.py   test_threshold.py
```

```bash
python simulations/run_s1.py --penalty MCP --direction up                    # MCP dense -> sparse
python simulations/run_s1.py --penalty SCAD --method lla --c-scale variance  # SCAD by LLA, rescaled C
python simulations/run_s1.py --method adaptive --c-scale variance            # adaptive lasso
```

## 1. The problem these estimators answer

```text
standard MCP / SCAD path, continuation from the sparse end
  λ_max ──► λ_min :  empty graph ─► pair (i,j) first enters, fit almost empty ─► one direction fixed ─► nonconvex penalty keeps it
                     = orientation lock-in (S3a; orientation_lock_in_051026.md)
lasso
  convex ─► independent of the start; direction unclear ─► keeps both M_ij and M_ji (hedge)
dense start (§3, §4, §5)
  λ_min ──► λ_max :  lasso solution, both directions of uncertain pairs still present ─► MCP / SCAD or a weighted lasso prunes
```

| estimator | start | then | convex steps | depends on the order of $\lambda$ |
|---|---|---|---|---|
| MCP / SCAD, sparse → dense (standard) | empty graph at $\lambda_{\max}$ | MCP / SCAD at each smaller $\lambda$, warm-started | no | yes |
| **MCP / SCAD, dense → sparse** | lasso solution at $\lambda_{\min}$ | MCP / SCAD at each larger $\lambda$, warm-started | no | yes |
| **MCP / SCAD by LLA** | lasso solution at the same $\lambda$ | two weighted lassos | yes | no |
| **adaptive lasso** | lasso solution at $\lambda_{\min}$ | one weighted lasso path | yes | no |

## 2. The rescaled $C$ (`--c-scale variance`)

Standardising every variable by its standard deviation $s_i$ changes the model:

$$M\Sigma+\Sigma M^\top+C=0\;\Longrightarrow\;\tilde M R+R\tilde M^\top+\tilde C=0,\qquad \tilde M_{ij}=M_{ij}\,\frac{s_j}{s_i},\quad \tilde C=D^{-1}CD^{-1},\quad D=\mathrm{diag}(s),\quad R\ \text{the correlation matrix}.$$

$$\operatorname{supp}\tilde M=\operatorname{supp}M,\qquad C=2I\ \Longrightarrow\ \tilde C=2\,\mathrm{diag}(1/s_i^2)\quad(\text{the rescaled } C).$$

```python
estimation_volatility(scale, "identity")   # 2 I            -- Dettling's pipeline (default)
estimation_volatility(scale, "variance")   # 2 diag(1/s^2)  -- the rescaled C
```

| | |
|---|---|
| code | `draw_instance(..., return_scale=True)` returns $s$ next to the standardised covariance; `S1Config.c_scale` and `--c-scale` select the matrix in both drivers; unstandardised data: $s=1$, both coincide |
| assumes | $C=2I$ on the scale on which the data were measured: exact for Dettling's `C_ID`, only partly right for the other three (campaign note §2.2); a device for the simulation study, not a recipe for real data |
| test | `test_standardized_population_model_is_misspecified_with_identity_and_exact_with_variance`: at $n=\infty$ the true support fits the standardised covariance with zero loss under the rescaled $C$, positive loss under $2I$; $D^{-1}MD$ solves the rescaled equation |

## 3. MCP / SCAD dense → sparse (`--direction up`)

$$\lambda_1<\dots<\lambda_{100}=\lambda_{\max},\qquad M^{(0)}=\hat M^{\text{lasso}}(\lambda_1),\qquad M^{(t)}=\text{MCP / SCAD stationary point at }\lambda_t\text{ reached from }M^{(t-1)},\qquad M^{(100)}=M_D\ \text{(diagonal fit)}.$$

```text
t = 0     lasso at λ_1         dense (close to the minimum-ℓ1 exact solution of the Lyapunov equation), well defined: convex
t = 1..100  MCP / SCAD at λ_t  monotone accelerated proximal gradient (NONCONVEX.md §4), warm start from λ_{t-1}
t = 100   diagonal fit         a stationary point at λ_max, so both orders end at the empty graph
same objective as the standard path; only the order differs, and with a nonconvex penalty the order decides which stationary point is reached
```

| | |
|---|---|
| code | `lasso_path(sigma, c, penalty="MCP", direction="up")`, the block `if direction == "up" and penalty != "lasso":`; for the lasso `direction` has no effect |
| tests, `tests/test_direction_cscale.py` | every point of the path stationary for the MCP objective; last point diagonal; differs from the standard path; default unchanged; lasso: both orders return the same path |
| cost | one lasso path + an MCP / SCAD path that starts dense: 2 to 5 times the standard MCP / SCAD path (measured, campaign note §3.4) |

### 3.1 The exact fits behind the dense start (`src/gclm/objective/exact_fits.py`)

$$\{M:\ M\hat\Sigma+\hat\Sigma M^\top+C=0\}=\{M_0+W\hat\Sigma^{-1}:\ W\ \text{skew}\},\qquad M_0=-\tfrac12 C\hat\Sigma^{-1}\ (\texttt{dense\_fit}),\qquad \dim=p(p-1)/2,\qquad \hat M^{\text{lasso}}(\lambda)\xrightarrow[\lambda\to 0]{}\ \text{the member with the smallest off-diagonal }\ell_1\text{ norm}.$$

| | |
|---|---|
| sources | Dettling, Homs, Améndola, Drton & Hansen 2023, Lemma 27; the lasso limit: theory note of 8 October 2026, §2 |
| added | 14 October 2026, for the dense-start experiments of the plan of 13 October, §1.1; the diagonal free as everywhere in the code |

| function | member | method |
|---|---|---|
| `min_l2_exact_fit(sigma_hat, c)` | smallest off-diagonal Frobenius norm | linear least squares in the skew coordinates $w$, $M(w)=M_0+\sum_k w_k B_k\hat\Sigma^{-1}$ with $B_k=E_{ij}-E_{ji}$ (`skew_basis`, `exact_fit_jacobian`) |
| `min_l1_exact_fit(sigma_hat, c)` | smallest off-diagonal $\ell_1$ norm ("basis pursuit", the limit of the lasso) | linear program in $(w,t)$, $\lvert M(w)_{ij}\rvert\le t_{ij}$, `scipy.optimize.linprog` with HiGHS at tolerance $10^{-9}$; 0.2 s at $p=20$, 1 s at $p=30$, 10–26 s at $p=50$ |
| `noisy_truth_start(m_true, tau, rng)` | $M^*+\tau\,N(0,1)$ off the diagonal | a ceiling for the dense starts, not an estimator; on standardised data pass `standardized_drift(M*, s)` |
| `exact_fit_residual(m, sigma_hat, c)` | $\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F$ | for tests and diagnostics |

Uniqueness of the minimum-$\ell_1$ fit, the BP condition of the 8 October note §4: $\tilde M$ must be the unique minimiser.

$$N=\{(i,j):\ M(w^\star)_{ij}\ne 0\},\ \ s=\operatorname{sign}M(w^\star)_N,\ \ Z=\{(i,j):\ M(w^\star)_{ij}=0\}:\qquad \lVert M(w^\star+d)\rVert_1-\lVert M(w^\star)\rVert_1=s^\top J_N d+\lVert J_Z d\rVert_1\ \ \text{exactly, for small } d.$$

$$\text{unique}\iff s^\top J_N d+\lVert J_Z d\rVert_1>0\ \ \forall d\ne 0\iff\{d:\ s^\top J_N d+\lVert J_Z d\rVert_1\le 0\}=\{0\}\qquad(\text{sharp minimum, Burke \& Ferris 1993}).$$

| | |
|---|---|
| probe, `min_l1_exact_fit(..., probes=k, return_info=True)` | random linear objectives maximised over that polyhedral cone in the unit box: 0 iff the cone is $\{0\}$, positive with probability one otherwise (`L1Fit.face_extent`); the constraints are homogeneous, so no tolerance on the minimal norm enters, only the classification of entries as zero (`zero_tol`) |
| campaign graphs | the extent is either exactly 0 or above 0.047 |
| dropped | re-solving with the norm held at its minimum up to a slack: flagged generic sample instances as non-unique (the slack is amplified along weakly sharp minima) |

Regression, `tests/test_exact_fits.py::test_bp_condition_count_on_the_campaign_graphs`:

| | |
|---|---|
| setting | thesis DGP at $p=10$, $k=1,\dots,4$, 25 graphs each, $C=2I$, the campaign's seed; population covariance on the correlation scale with the rescaled $C$ |
| $\tilde M$ the unique minimum-$\ell_1$ fit | **11 of 100**: 7 at $k=1$, 4 at $k=2$, none at $k=3,4$; the 7 October count reproduced exactly |
| $\tilde M$ attains the minimal norm | 17; in the other 6 the optimal face is flat (the direction of an isolated edge is free) and the vertex the LP returns depends on the solver: HiGHS 1.15 returns $\tilde M$ in 14, the 8 October note counted 15 |
| run time | about 5 s |
| the other tests, 12 in all | basis and Jacobian against an explicit construction, full column rank; all three fits exact to $10^{-8}\lVert C\rVert_F$ at $p=5,10,20$ and distinct; each minimum-norm member smallest in its norm by a strict margin; the LP against an independent formulation in the $p^2$ entries with the design matrix as equality constraints; the uniqueness probe on the flat two-node case, a sharp two-node case, a generic sample and a population graph with an isolated edge; the noisy truth and the residual; refused inputs |

## 4. MCP / SCAD by local linear approximation (`--method lla`)

At every $\lambda$ separately, with $P_\lambda$ the MCP or SCAD penalty, $M^{(0)}=\hat M^{\text{lasso}}(\lambda)$ and two steps $s=1,2$:

$$w_{ij}=\frac{P'_\lambda(|M^{(s-1)}_{ij}|)}{\lambda}\in[0,1],\qquad M^{(s)}=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}| .$$

```text
w_ij = 1   at a zero entry                      penalised like in the lasso
w_ij = 0   beyond γλ, where the penalty is flat  large entries no longer shrunk
each step:  concave penalty replaced by its tangent at the current estimate -> a convex problem (Zou & Li 2008)
fixed point of the iteration -> stationary point of the MCP / SCAD objective
```

| | |
|---|---|
| why two steps from the lasso | Fan, Xue & Zou 2014: start close enough to the truth and nonzero entries large enough ⇒ the first step gives the oracle estimator and the second reproduces it; the lasso is a suitable start under a restricted-eigenvalue condition; the irrepresentability condition, which the lasso needs for support recovery and which fails for most of these graphs, is not required |
| code | `lla_path`; weights `lla_weights`; each weighted lasso `solve_fista(..., weights=w)` started from the current estimate; nothing carried from one $\lambda$ to the next |
| caveat at the dense end | the design $A(\hat\Sigma)$ has rank $p(p+1)/2$; more unpenalised entries than that (the diagonal plus every entry beyond $\gamma\lambda$) ⇒ flat directions and more than one minimiser, the estimate is the one FISTA reaches from the lasso solution; the smallest $\lambda$ only, where the fit is dense anyway |
| tests, `tests/test_lla_adaptive.py` | weights $=P'/\lambda$; one step satisfies the optimality conditions of its weighted lasso at every $\lambda$ and agrees with coordinate descent on the explicit design where the problem is well posed; iterated to a fixed point it is stationary for the nonconvex objective; the estimate at one $\lambda$ does not depend on the rest of the grid; ends at the diagonal fit; tends to the lasso as $\gamma\to\infty$ |
| cost | about three lasso paths |

## 5. Adaptive lasso (`--method adaptive`)

Zou 2006 with a lasso pilot:

$$M^0=\hat M^{\text{lasso}}(\lambda_{\min}),\qquad w_{ij}=\frac{1/|M^0_{ij}|}{\min_{k\ne l}1/|M^0_{kl}|}\ \ (\text{largest pilot entry: } w=1;\ \text{entries with } M^0_{ij}=0\text{ excluded}),$$

$$\hat M(\lambda)=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}|,\qquad \lambda\in\text{100 log-spaced values up to }\lambda^{w}_{\max}=\max_{i\ne j}\frac{|\nabla f(M_D)_{ij}|}{w_{ij}} .$$

| | |
|---|---|
| $\lambda^{w}_{\max}$ | the smallest $\lambda$ at which the solution is the diagonal fit $M_D$ |
| idea | start dense, remove the weak entries first, as the dense → sparse path, with a convex second step: the control for whether the nonconvex penalty is needed, or only the dense start |
| code | `adaptive_lasso_path` returns an `AdaptiveLassoPath`, which also holds the pilot and the weights (`inf` marks an excluded entry) |
| tests, `tests/test_lla_adaptive.py` | the weights; the optimality conditions of the weighted problem at every $\lambda$ and agreement with coordinate descent; excluded entries stay zero and every support is contained in the pilot's; with equal weights it is the lasso path |
| cost | about two lasso paths |

### 5.1 Any matrix as the start or the pilot (`--up-start`, `--pilot`; 14 October 2026)

E1 of the plan of 13 October asks which property of the dense start matters, so the start became an argument.

```text
lasso_path(..., direction="up", start=M)   covloss_path(..., direction="up", start=M)   -> LassoPath.start          runner: --up-start (paths)
adaptive_lasso_path(pilot=M)               adaptive_covloss_path(pilot=M)               -> AdaptiveLassoPath.pilot  runner: --pilot (adaptive, thresholded lasso)
any (p, p) matrix; omitted -> what they always computed; start = the lasso's dense end reproduces the stored paths bit for bit (tests/test_starts.py)
before 14 October 2026 the start was always the lasso's dense end (covariance losses: Varando & Hansen's exact fit)
```

| name | the matrix | property kept (plan §2.0) |
|---|---|---|
| `lasso` | the lasso solution at the smallest $\lambda$ (the default for the direct loss) | exact fit up to $O(\lambda)$, $\ell_1$-smallest, oriented |
| `exact` | $-\tfrac12 C\hat\Sigma^{-1}$ (Varando & Hansen; the default for the covariance losses) | exact fit only |
| `l1` | the minimum-$\ell_1$ exact fit, §3.1 | exact fit, $\ell_1$-smallest, oriented (the limit the lasso approximates) |
| `l2` | the minimum-Frobenius exact fit, §3.1 | exact fit, dense, both directions of a pair graded |
| `truth-<τ>` | the standardised truth $\tilde M$ plus $\tau\,N(0,1)$ off the diagonal | oriented only (a ceiling, not an estimator) |

| | |
|---|---|
| diagnostics of the start, per data set | `start_*` in the shard: norms, Lyapunov residual, support size at $10^{-8}$, confusion counts and the orientation breakdown of that support (the "dense-start content" of the 7 October note); `campaign.py` → `start_f1`, `start_recall`, `start_correct`, `start_hedged`, `start_reversed`, … |
| fully dense starts `l2`, `exact`, `truth-τ` | hit the step cap of the MCP solver at a few $\lambda$ in the flat valley of the dense end (`converged` flags them; `LIKELIHOOD.md` §4.3) |
| `l1` start | reproduces the lasso-started path on the test instance, as the theory predicts |

### 5.2 The thresholded lasso (`--method threshold`, `--method threshold-norefit`)

E2 asks whether the adaptive lasso's gain is in its pilot or in its reweighting.

$$\hat S(t)=\{(i,j):\ |M^0_{ij}|\ge t\},\qquad \hat M(t)=\text{least-squares refit of }\hat S(t)\text{ on the direct loss, diagonal free }(\texttt{DirectRefit}),\qquad t\ \text{chosen by the score}.$$

```text
the simplest pruning of the same pilot, as an estimator of its own:
thresholded_lasso_path(sigma, c, pilot=None, n_thresholds=100, refit="direct")
pilot    computed exactly as adaptive_lasso_path does; pass the same pilot to both to make the pairing explicit
grid     100 thresholds: 0 at the dense end, 98 log-spaced between the smallest and the largest pilot entry above 1e-8, ∞ at the sparse end; nested supports
returns  ThresholdedLassoPath, a LassoPath whose lambdas are the thresholds in increasing order -> evaluate_path, bic_along_path, selection, search unchanged
refit    "direct" (default)  |  "none" = --method threshold-norefit: pilot values kept, its aupr = how well the dense end orders the edges  |  "loglik": likelihood refit
shard    lambda_max = the largest pilot entry; objective = the loss (no penalty); kkt NaN; no solver steps
tests    tests/test_threshold.py                cost   one lasso path + at most 100 closed-form refits
```

## 6. On the cluster

| | |
|---|---|
| `simulations/run_s1_shard.py` | the same options: `--c-scale`, `--direction`, `--method`, and since 14 October 2026 `--up-start`, `--pilot`, `--weights`, `--w-min`, `--graph` |
| `--select bic`, `--select search` | records per graph the support at every $\lambda$, the graph the score selects on the path and the graph the greedy search of [`SEARCH.md`](SEARCH.md) reaches from it |
| `cluster/submit_campaign.sh` | the cells of the campaign; wave 9 = the mechanism experiments of the plan of 13 October 2026, five groups |
| cells and commands | campaign note §4 and §5; [`REPRODUCTION.md`](REPRODUCTION.md) §2.7 |

```text
wave 9:   --group starts | threshold | betamin | class | starts-loglik
```

## 7. Why the rescaled $C$ helps these estimators, and not the lasso

| numbers | |
|---|---|
| `next_steps/051026/files/why_rescaling_helps.py` | `C_ID` graphs, $p=10$: 20 graphs, $p=20$: 8 |
| `runs/campaign/campaign_paired.csv` | the campaign tables, 800 graphs per cell |

### 7.1 $C$ is the right-hand side, not a tuning constant

$$\text{exact fits of } \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2:\qquad M\hat\Sigma+\hat\Sigma M^\top=-C,\qquad \text{affine, } \dim=p(p-1)/2\qquad(\operatorname{rank}A(\hat\Sigma)=p(p+1)/2,\ \text{onto the symmetric matrices, consistent for every } C).$$

$C$ decides which $M$ fit the data at all; change $C$ and the whole solution set moves.

| misfit of the true standardised drift $\tilde M$, $n=\infty$ | rescaled $C$ | $C=2I$ |
|---|---|---|
| $p=10$ | 0, $\tilde M$ is in the solution set | 0.8 % of the diagonal fit's loss (the best fit on the true support) |
| $p=20$ | 0 | 1.8 % |
| normalisation | `misfit` in the script | the table in the campaign note §2.2 uses another one and reads 1.4 % at $p=10$ |

### 7.2 A diagonal error in $C$ is a dense error in $M$

$$C\ \text{wrong by a diagonal }\Delta:\quad M\mapsto M+X,\quad X\hat\Sigma+\hat\Sigma X^\top=-\Delta;\qquad \text{smallest correction } X=-A(\hat\Sigma)^{+}\mathrm{vec}(\Delta),\ \text{dense; diagonal only if }\hat\Sigma\text{ were diagonal}.$$

Two nodes with correlation $\rho$, $\Delta=\mathrm{diag}(\delta,0)$, $X=\begin{pmatrix}a&b\\c&d\end{pmatrix}$:

$$a+b\rho=-\delta/2,\qquad c\rho+d=0,\qquad a\rho+b+c+d\rho=0\quad\Longrightarrow\quad b+c=\frac{\rho\,\delta}{2(1-\rho^2)}\ :\ \text{off-diagonal mass proportional to the correlation, a false edge or a changed true one}.$$

| standardised `C_ID` data, $\Delta=2I-2\,\mathrm{diag}(1/s_i^2)$ | |
|---|---|
| raw stationary variances $s_i^2$, 10th to 90th percentile at $p=10$ | 0.3 to 2.1 |
| median $\lvert\Delta_{ii}\rvert$ | 1.5, the same order as $C$ itself; most of it absorbed by the diagonal of $M$ |
| off-diagonal shift of §7.1, median entry | 0.06 ($p=10$), 0.03 ($p=20$) |
| off-diagonal shift, largest entry | 1.0 ($p=10$), 1.8 ($p=20$) |
| true edge weights, median / 10th percentile | 0.7 / 0.16 |
| verdict | small on most entries, as large as the weakest true edges on some, larger than a typical true edge on a few |

### 7.3 Where each estimator makes its decision

```text
lasso                    decides in the MIDDLE of the path, where the fit is a compromise with sparsity anyway:
                         only the ORDER in which entries enter matters, and a shift of this size hardly changes it
                         (Dettling's robustness of the lasso to a misspecified C is this)
dense-start estimators   decide at the DENSE END (§3–§5): the lasso solution at the smallest λ, a point of the solution set; keep its large entries
  rescaled C             dense end = truth + null-space component                  -> pruning strips the null-space part
  C = 2I                 dense end = truth + null-space component + shift (§7.2)   -> pruning drops weak true edges, keeps shifted false ones
hedge and lock-in        lasso: direction unclear -> keeps both M_ij, M_ji (hedge); pruning resolves the hedge by |M⁰_ij| vs |M⁰_ji|
                         sparse → dense path: pair enters alone -> one direction fixed -> MCP / SCAD keeps it (lock-in, §1)
```

| lasso path `max_f1` under rescaling, $n=10^3$ | |
|---|---|
| $p=10$ | unchanged: 0.589 against 0.584 |
| $p=20$ | slightly worse: 0.531 against 0.507 |

| `C_ID`, $n=\infty$ | ranking of true edges above non-edges by $\lvert M^0_{ij}\rvert$ (AUC) | true edges whose reversed entry is the larger one |
|---|---|---|
| $p=10$, $C=2I$ | 0.71 | 22 % |
| $p=10$, rescaled $C$ | 0.76 | 14 % |
| $p=20$, $C=2I$ | 0.88 | 10 % |
| $p=20$, rescaled $C$ | 0.93 | 1 % |
| $n=10^4$ | same pattern, smaller differences: 0.66 → 0.73 and 0.86 → 0.87 | |
| measured on | the dense end: the sizes $\lvert M^0_{ij}\rvert$, exactly the adaptive lasso's weights and the dense → sparse paths' start | |

### 7.4 Two mechanisms that add, and a graded response

```text
dense start    avoids the early orientation lock-in of the standard path (orientation_lock_in_051026.md): helps on its own, with C = 2I; nothing to do with C
+ rescaled C   makes the dense end informative (§7.3)
extra gain from rescaling follows how much of the misfit the rescaling removes (misfit figures: campaign note §2.2)
```

| true $C$; MCP dense → sparse against the lasso with the same $C$, `max_f1`, $p=20$, $n=10^4$ | gain with $C=2I$ | gain with the rescaled $C$ | extra from rescaling | misfit removed by rescaling |
|---|---|---|---|---|
| `C_ID` | +0.077 | +0.154 | +0.077 | 100 % |
| `C_Random_Min_Diag` | +0.081 | +0.141 | +0.060 | 95 % |
| `C_Random_Diag` | +0.079 | +0.122 | +0.043 | 40 % |
| `C_Random_Full` | +0.004 | +0.042 | +0.038 | 30 % |

### 7.5 A third channel: the score

| | |
|---|---|
| the score's loss term, and with it the search | the likelihood of $\Sigma(M,C)$; with the wrong $C$ the best-scoring graph is a different graph (S3b §9; `next_steps/051026/next_steps_051026.md` §2) |
| lasso + search, $p=20$, $n=10^4$ | gains from rescaling, 0.614 → 0.650, although the lasso path does not |

In one sentence: with the wrong $C$ the data say that no sparse graph fits; the lasso, which never insists on an exact fit, is barely affected, while the estimators that start from an exact fit and prune are pruning the wrong object.

## References

| reference | used for |
|---|---|
| Fan, J., Xue, L. & Zou, H. (2014). Strong oracle optimality of folded concave penalized estimation. *Annals of Statistics* 42(3), 819–849. | LLA from a lasso start, §4 |
| Zou, H. & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics* 36(4), 1509–1533. | the local linear approximation, §4 |
| Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA* 101(476), 1418–1429. | §5 |
| Varando, G. & Hansen, N. R. (2020). Graphical continuous Lyapunov models. *UAI*. | their paths run from a dense start towards sparser solutions |
| Dettling, P., Drton, M. & Kolar, M. (2024). On the Lasso for graphical continuous Lyapunov models. *CLeaR*. | the direct loss and the simulation design |
| Dettling, P., Homs, R., Améndola, C., Drton, M. & Hansen, N. R. (2023). Identifiability in continuous Lyapunov models. *SIAM Journal on Matrix Analysis and Applications* 44(4). | Lemma 27: the exact fits $(W-C/2)\hat\Sigma^{-1}$, $W$ skew, §3.1 |
| Burke, J. V. & Ferris, M. C. (1993). Weak sharp minima in mathematical programming. *SIAM Journal on Control and Optimization* 31(5), 1340–1359. | a polyhedral function has a unique minimiser iff the minimum is sharp, §3.1 |
