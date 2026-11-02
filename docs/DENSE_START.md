# Dense-start estimators

| | |
|---|---|
| reasons | [`simulations/VERDICTS.md`](../simulations/VERDICTS.md) 3–7; [`cluster_campaign_051026.md`](../next_steps/Oct2026/051026/cluster_campaign_051026.md) (campaign note) |
| lock-in | [`orientation_lock_in_051026.md`](../claude_docs/orientation_lock_in_051026.md) |
| solver | [`NONCONVEX.md`](NONCONVEX.md) §4 (monotone accelerated proximal gradient) |

```text
scope: direct loss, default `fista` solver, unpenalised diagonal; the estimators added for the campaign of 2026-10

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

## 1. Problem

```text
standard MCP / SCAD path, continuation from the sparse end
  λ_max ──► λ_min :  empty graph ─► pair (i,j) first enters, fit almost empty ─► one direction fixed ─► nonconvex penalty keeps it
                     = orientation lock-in (S3a; orientation_lock_in_051026.md)
lasso
  convex ─► independent of the start; direction unclear ─► keeps both M_ij and M_ji (hedge)
dense start (§3, §4, §5)
  λ_min ──► λ_max :  lasso solution, both directions of uncertain pairs still present ─► MCP / SCAD or a weighted lasso prunes
```

| estimator | start | then | convex | $\lambda$-order |
|---|---|---|---|---|
| MCP / SCAD, sparse → dense (standard) | $\emptyset$ at $\lambda_{\max}$ | MCP / SCAD, warm, $\lambda\downarrow$ | no | yes |
| **MCP / SCAD, dense → sparse**, §3 | lasso at $\lambda_{\min}$ | MCP / SCAD, warm, $\lambda\uparrow$ | no | yes |
| **MCP / SCAD by LLA**, §4 | lasso at the same $\lambda$ | two weighted lassos | yes | no |
| **adaptive lasso**, §5 | lasso at $\lambda_{\min}$ | one weighted lasso path | yes | no |

## 2. Rescaled $C$ (`--c-scale variance`)

$$M\Sigma+\Sigma M^\top+C=0\ \Longrightarrow\ \tilde M R+R\tilde M^\top+\tilde C=0,\qquad \tilde M=D^{-1}MD,\ \ \tilde M_{ij}=M_{ij}\,\frac{s_j}{s_i},\qquad \tilde C=D^{-1}CD^{-1},\qquad D=\mathrm{diag}(s),\qquad R=D^{-1}\Sigma D^{-1}.$$

$$\operatorname{supp}\tilde M=\operatorname{supp}M,\qquad C=2I\ \Longrightarrow\ \tilde C=2\,\mathrm{diag}(1/s_i^2)\quad(\text{the rescaled }C).$$

```python
estimation_volatility(scale, "identity")   # 2 I            -- Dettling's pipeline (default)
estimation_volatility(scale, "variance")   # 2 diag(1/s^2)  -- the rescaled C
# draw_instance(..., return_scale=True) returns s next to the standardised covariance; S1Config.c_scale and --c-scale select
#   the matrix in both drivers; unstandardised data: s = 1, both coincide
# assumes C = 2I on the scale on which the data were measured: exact for Dettling's C_ID, only partly right for the other
#   three (campaign note §2.2); a device for the simulation study, not a recipe for real data
# test  test_standardized_population_model_is_misspecified_with_identity_and_exact_with_variance:
#   n = inf, the true support fits the standardised covariance with loss 0 under 2 diag(1/s^2), loss > 0 under 2I;
#   D^-1 M D solves the rescaled equation
```

## 3. Dense → sparse (`--direction up`)

$$\lambda_1<\dots<\lambda_{100}=\lambda_{\max},\qquad M^{(0)}=\hat M^{\mathrm{lasso}}(\lambda_1),\qquad M^{(t)}=\text{MCP / SCAD stationary point at }\lambda_t\text{ reached from }M^{(t-1)},\qquad M^{(100)}=M_D\ \ (\text{diagonal fit}).$$

```text
t = 0       lasso at λ_1        dense (close to the minimum-ℓ1 exact fit of the Lyapunov equation, §3.1); convex, well defined
t = 1..100  MCP / SCAD at λ_t   monotone accelerated proximal gradient (NONCONVEX.md §4), warm start from λ_{t-1}
t = 100     diagonal fit M_D    a stationary point at λ_max, so both orders end at the empty graph
same objective as the standard path; only the order differs; with a nonconvex penalty the order decides which stationary point is reached

code    lasso_path(sigma, c, penalty="MCP", direction="up"), the block `if direction == "up" and penalty != "lasso":`
        for the lasso `direction` has no effect
tests   tests/test_direction_cscale.py (5): every point stationary for the MCP objective; last point diagonal; differs from the
        standard path; default unchanged; lasso: both orders return the same path
cost    one lasso path + an MCP / SCAD path that starts dense = 2 to 5 x the standard MCP / SCAD path (measured, campaign note §3.4)
```

### 3.1 Exact fits (`src/gclm/objective/exact_fits.py`)

$$\{M:\ M\hat\Sigma+\hat\Sigma M^\top+C=0\}=\{M_0+W\hat\Sigma^{-1}:\ W^\top=-W\},\qquad M_0=-\tfrac12 C\hat\Sigma^{-1},\qquad \dim=p(p-1)/2,\qquad \hat M^{\mathrm{lasso}}(\lambda)\xrightarrow[\lambda\to 0]{}\ \arg\min\textstyle\sum_{i\ne j}\lvert M_{ij}\rvert\ \text{over the set}.$$

$$M(w)=M_0+\sum_k w_k B_k\hat\Sigma^{-1},\qquad B_k=E_{ij}-E_{ji},\qquad \ell_2:\ \min_w\sum_{i\ne j}M(w)_{ij}^2,\qquad \ell_1:\ \min_{w,t}\sum_{i\ne j}t_{ij}\ \ \text{s.t.}\ \lvert M(w)_{ij}\rvert\le t_{ij}.$$

| function | member | method |
|---|---|---|
| `dense_fit(sigma_hat, c)` | $M_0$ | $-\tfrac12 C\hat\Sigma^{-1}$ |
| `min_l2_exact_fit(sigma_hat, c)` | $\min\sum_{i\ne j}M_{ij}^2$ | least squares in $w$ (`skew_basis`, `exact_fit_jacobian`) |
| `min_l1_exact_fit(sigma_hat, c)` | $\min\sum_{i\ne j}\lvert M_{ij}\rvert$ (basis pursuit, the lasso limit) | LP in $(w,t)$; `scipy.optimize.linprog`, HiGHS, $10^{-9}$ |
| `noisy_truth_start(m_true, tau, rng)` | $M^*+\tau\,N(0,1)$ off-diagonal | ceiling, not an estimator; standardised: `standardized_drift(M*, s)` |
| `exact_fit_residual(m, sigma_hat, c)` | $\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F$ | tests, diagnostics |

| `min_l1_exact_fit`, $p$ | 20 | 30 | 50 |
|---|---|---|---|
| $t\ [\mathrm{s}]$ | 0.2 | 1 | 10–26 |

```text
sources   Dettling, Homs, Améndola, Drton & Hansen 2023, Lemma 27 (the solution set); the lasso limit: theory note 2026-10-08 §2
added     2026-10-14 for the dense-start experiments of the plan of 2026-10-13 §1.1; diagonal free, as everywhere in the code
```

$$N=\{(i,j):\ M(w^\star)_{ij}\ne 0\},\quad s=\operatorname{sign}M(w^\star)_N,\quad Z=\{(i,j):\ M(w^\star)_{ij}=0\}:\qquad \lVert M(w^\star+d)\rVert_1-\lVert M(w^\star)\rVert_1=s^\top J_N d+\lVert J_Z d\rVert_1\ \ \text{exactly, for small }d.$$

$$\tilde M\ \text{unique minimiser}\iff s^\top J_N d+\lVert J_Z d\rVert_1>0\ \ \forall d\ne 0\iff\{d:\ s^\top J_N d+\lVert J_Z d\rVert_1\le 0\}=\{0\}\qquad(\text{sharp minimum, Burke and Ferris 1993}).$$

```text
BP condition (note 2026-10-08 §4): M̃ must be the unique minimum-ℓ1 fit
probe     min_l1_exact_fit(..., probes=k, return_info=True) -> L1Fit.face_extent: random linear objectives maximised over the
          cone above, in the unit box; 0 iff the cone is {0}, > 0 with probability one otherwise. The constraints are homogeneous:
          no tolerance on the minimal norm enters, only the classification of entries as zero (zero_tol)
campaign  the extent is either exactly 0 or above 0.047
dropped   re-solving with the norm held at its minimum up to a slack: flagged generic sample instances as non-unique
          (the slack is amplified along weakly sharp minima)
```

| `test_bp_condition_count_on_the_campaign_graphs`, $k$ | 1 | 2 | 3 | 4 | $\sum$ |
|---|---|---|---|---|---|
| $\tilde M$ unique | 7 | 4 | 0 | 0 | 11 / 100 |
| $\lVert\tilde M\rVert_1=\min$ | | | | | 17 |
| HiGHS 1.15 $\to\tilde M$ | | | | | 14 |

```text
setting   thesis DGP, p = 10, k = 1..4, 25 graphs each, C = 2I, the campaign's seed; population covariance on the correlation
          scale with the rescaled C; about 5 s
11 unique the count of 2026-10-07 reproduced exactly
17 vs 14  in the other 6 the optimal face is flat (the direction of an isolated edge is free) and the vertex the LP returns
          depends on the solver: HiGHS 1.15 returns M̃ in 14, the note of 2026-10-08 counted 15
tests     tests/test_exact_fits.py, 12 in all: basis and Jacobian against an explicit construction, full column rank; all three
          fits exact to 1e-8 ‖C‖_F at p = 5, 10, 20 and distinct; each minimum-norm member smallest in its norm by a strict
          margin; the LP against an independent formulation in the p^2 entries with the design matrix as equality constraints;
          the uniqueness probe on the flat two-node case, a sharp two-node case, a generic sample and a population graph with
          an isolated edge; the noisy truth and the residual; refused inputs
```

## 4. LLA (`--method lla`)

$$M^{(0)}=\hat M^{\mathrm{lasso}}(\lambda),\qquad w_{ij}=\frac{P_\lambda'(\lvert M^{(s-1)}_{ij}\rvert)}{\lambda}\in[0,1],\qquad M^{(s)}=\arg\min_M\ \tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2+\lambda\sum_{i\ne j}w_{ij}\lvert M_{ij}\rvert,\qquad s=1,2,\ \ \text{each }\lambda\text{ separately}.$$

```text
w_ij = 1   at a zero entry                      penalised like in the lasso
w_ij = 0   beyond γλ, where the penalty is flat  large entries no longer shrunk
each step:  concave penalty replaced by its tangent at the current estimate -> a convex problem (Zou & Li 2008)
fixed point of the iteration -> stationary point of the MCP / SCAD objective

two steps  Fan, Xue & Zou 2014: start close enough to the truth and nonzero entries large enough => the first step gives the
           oracle estimator and the second reproduces it; the lasso is a suitable start under a restricted-eigenvalue condition;
           the irrepresentability condition (needed by the lasso for support recovery, fails for most of these graphs) is not required
caveat     rank A(Σ̂) = p(p+1)/2; more unpenalised entries than that (the diagonal plus every entry beyond γλ) => flat directions
           and more than one minimiser; the estimate is the one FISTA reaches from the lasso solution; the smallest λ only,
           where the fit is dense anyway
code       lla_path; weights lla_weights; each weighted lasso solve_fista(..., weights=w) started from the current estimate;
           nothing carried from one λ to the next
tests      tests/test_lla_adaptive.py: weights = P'/λ; one step satisfies the optimality conditions of its weighted lasso at every λ
           and agrees with coordinate descent on the explicit design where the problem is well posed; iterated to a fixed point it
           is stationary for the nonconvex objective; the estimate at one λ does not depend on the rest of the grid; ends at the
           diagonal fit; tends to the lasso as γ -> inf
cost       about three lasso paths
```

## 5. Adaptive lasso (`--method adaptive`)

$$M^0=\hat M^{\mathrm{lasso}}(\lambda_{\min}),\qquad w_{ij}=\frac{1/\lvert M^0_{ij}\rvert}{\min_{k\ne l}1/\lvert M^0_{kl}\rvert}\qquad(\text{largest pilot entry: }w=1;\ \ M^0_{ij}=0:\ w_{ij}=\infty,\ \text{excluded}),$$

$$\hat M(\lambda)=\arg\min_M\ \tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2+\lambda\sum_{i\ne j}w_{ij}\lvert M_{ij}\rvert,\qquad \lambda\in\{100\ \text{log-spaced}\}\ \text{up to}\ \lambda^{w}_{\max}=\max_{i\ne j}\frac{\lvert\nabla f(M_D)_{ij}\rvert}{w_{ij}}=\min\{\lambda:\ \hat M(\lambda)=M_D\}.$$

```text
Zou 2006 with a lasso pilot: start dense, remove the weak entries first, as the dense -> sparse path, with a convex second step
= the control for whether the nonconvex penalty is needed, or only the dense start
code    adaptive_lasso_path -> AdaptiveLassoPath, which also holds the pilot and the weights (inf marks an excluded entry)
tests   tests/test_lla_adaptive.py: the weights; the optimality conditions of the weighted problem at every λ and agreement with
        coordinate descent; excluded entries stay zero and every support is contained in the pilot's; equal weights -> the lasso path
cost    about two lasso paths
```

### 5.1 Start, pilot (`--up-start`, `--pilot`)

```text
2026-10-14; E1 of the plan of 2026-10-13: which property of the dense start matters -> the start became an argument
lasso_path(..., direction="up", start=M)   covloss_path(..., direction="up", start=M)   -> LassoPath.start          runner: --up-start (paths)
adaptive_lasso_path(pilot=M)               adaptive_covloss_path(pilot=M)               -> AdaptiveLassoPath.pilot  runner: --pilot (adaptive, thresholded lasso)
any (p, p) matrix; omitted -> what they always computed; start = the lasso's dense end reproduces the stored paths bit for bit (tests/test_starts.py)
before 2026-10-14 the start was always the lasso's dense end (covariance losses: Varando & Hansen's exact fit)
```

| `--up-start`, `--pilot` | matrix | kept (plan §2.0) |
|---|---|---|
| `lasso` | $\hat M^{\mathrm{lasso}}(\lambda_{\min})$; default, direct loss | exact fit $+O(\lambda)$, $\ell_1$-smallest, oriented |
| `exact` | $-\tfrac12 C\hat\Sigma^{-1}$ (Varando & Hansen); default, covariance losses | exact fit |
| `l1` | $\arg\min\ell_1$ exact fit, §3.1 | exact fit, $\ell_1$-smallest, oriented (lasso limit) |
| `l2` | $\arg\min\lVert\cdot\rVert_F$ exact fit, §3.1 | exact fit, dense, both directions graded |
| `truth-<τ>` | $\tilde M+\tau\,N(0,1)$ off-diagonal | oriented (ceiling, not an estimator) |

```text
diagnostics   start_* in the shard: norms, Lyapunov residual, support size at 1e-8, confusion counts and the orientation breakdown
              of that support (the "dense-start content" of the note of 2026-10-07); campaign.py -> start_f1, start_recall,
              start_correct, start_hedged, start_reversed, ...
l2, exact, truth-τ   fully dense: hit the step cap of the MCP solver at a few λ in the flat valley of the dense end
                     (`converged` flags them; LIKELIHOOD.md §4.3)
l1                   reproduces the lasso-started path on the test instance, as the theory predicts
```

### 5.2 Thresholded lasso (`--method threshold`, `--method threshold-norefit`)

$$\hat S(t)=\{(i,j):\ \lvert M^0_{ij}\rvert\ge t\},\qquad \hat M(t)=\arg\min_{\operatorname{supp}M\subseteq\hat S(t)}\ \tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2\ \ (\mathrm{DirectRefit},\ \text{diagonal free}),\qquad t\ \text{chosen by the score}.$$

```text
E2: is the adaptive lasso's gain in its pilot or in its reweighting? -> the simplest pruning of the same pilot, as an estimator of its own
thresholded_lasso_path(sigma, c, pilot=None, n_thresholds=100, refit="direct")
pilot    computed exactly as adaptive_lasso_path does; pass the same pilot to both to make the pairing explicit
grid     100 thresholds: 0 at the dense end, 98 log-spaced between the smallest and the largest pilot entry above 1e-8, inf at the sparse end; nested supports
returns  ThresholdedLassoPath, a LassoPath whose lambdas are the thresholds in increasing order -> evaluate_path, bic_along_path, selection, search unchanged
refit    "direct" (default)  |  "none" = --method threshold-norefit: pilot values kept, its aupr = how well the dense end orders the edges  |  "loglik": likelihood refit
shard    lambda_max = the largest pilot entry; objective = the loss (no penalty); kkt NaN; no solver steps
tests    tests/test_threshold.py                cost   one lasso path + at most 100 closed-form refits
```

## 6. Cluster

| | |
|---|---|
| `simulations/run_s1_shard.py` | `--c-scale --direction --method`; since 2026-10-14 `--up-start --pilot --weights --w-min --graph` |
| `--select bic`, `--select search` | per graph: the support at every $\lambda$, the graph the score selects on the path, the graph the greedy search of [`SEARCH.md`](SEARCH.md) reaches from it |
| `cluster/submit_campaign.sh` | the cells; wave 9 = mechanism experiments, plan 2026-10-13, five groups |
| cells, commands | campaign note §4, §5; [`REPRODUCTION.md`](REPRODUCTION.md) §2.7 |

```bash
# wave 9
cluster/submit_campaign.sh --wave 9 --group starts          # | threshold | betamin | class | starts-loglik
```

## 7. Why rescaling helps

```text
numbers   next_steps/051026/files/why_rescaling_helps.py   C_ID graphs, p = 10: 20 graphs, p = 20: 8
          runs/campaign/campaign_paired.csv                 the campaign tables, 800 graphs per cell
```

### 7.1 Solution set

$$\arg\min_M\ \tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2=\{M:\ M\hat\Sigma+\hat\Sigma M^\top=-C\},\quad\text{affine},\ \dim=p(p-1)/2;\qquad \operatorname{rank}A(\hat\Sigma)=p(p+1)/2\ \ (\text{onto the symmetric matrices, consistent for every }C).$$

```text
C is the right-hand side, not a tuning constant: it decides which M fit the data at all; change C and the whole solution set moves
misfit = loss of the best fit on the true support / loss of the diagonal fit (`misfit` in the script); the campaign note §2.2
         uses another normalisation and reads 1.4 % at p = 10
```

| $\mathrm{misfit}(\tilde M)$, $n=\infty$ | $\tilde C=2\,\mathrm{diag}(1/s_i^2)$ | $C=2I$ |
|---|---|---|
| $p=10$ | 0 ($\tilde M$ in the solution set) | 0.8 % |
| $p=20$ | 0 | 1.8 % |

### 7.2 Diagonal $\Delta$, dense $X$

$$C\ \text{wrong by a diagonal }\Delta:\qquad M\mapsto M+X,\qquad X\hat\Sigma+\hat\Sigma X^\top=-\Delta,\qquad X_{\min}=-A(\hat\Sigma)^{+}\mathrm{vec}(\Delta)\ \ \text{dense; diagonal only if }\hat\Sigma\text{ were diagonal}.$$

$$p=2,\ \hat\Sigma=\begin{pmatrix}1&\rho\\ \rho&1\end{pmatrix},\ \Delta=\mathrm{diag}(\delta,0),\ X=\begin{pmatrix}a&b\\ c&d\end{pmatrix}:\qquad a+b\rho=-\tfrac{\delta}{2},\quad c\rho+d=0,\quad a\rho+b+c+d\rho=0\ \Longrightarrow\ b+c=\frac{\rho\,\delta}{2(1-\rho^2)}.$$

```text
off-diagonal mass proportional to the correlation: a false edge, or a changed true one
standardised C_ID data, Δ = 2I - 2 diag(1/s_i^2): median |Δ_ii| = 1.5, the same order as C itself; most of it absorbed by the
diagonal of M; what leaks off the diagonal is the shift X of §7.1
verdict   small on most entries, as large as the weakest true edges on some, larger than a typical true edge on a few
```

| `C_ID`, $\Delta=2I-2\,\mathrm{diag}(1/s_i^2)$ | |
|---|---|
| $s_i^2$, $q_{0.1}$–$q_{0.9}$, $p=10$ | 0.3–2.1 |
| $\operatorname{median}\lvert\Delta_{ii}\rvert$, $p=10$ | 1.5 |
| $\operatorname{median}_{i\ne j}\lvert X_{ij}\rvert$ | 0.06 ($p=10$), 0.03 ($p=20$) |
| $\max_{i\ne j}\lvert X_{ij}\rvert$ | 1.0 ($p=10$), 1.8 ($p=20$) |
| $\lvert M^*_{ij}\rvert$, $(i,j)\in\operatorname{supp}M^*$: $\operatorname{median}$, $q_{0.1}$ | 0.7, 0.16 |

### 7.3 Mechanism

$$\tilde C:\ \ M^0=\tilde M+W\hat\Sigma^{-1}\ (W\ \text{skew});\qquad C=2I:\ \ M^0=\tilde M+W\hat\Sigma^{-1}+X\ (\S 7.2);\qquad \text{prune: keep }(i,j)\ \text{with }\lvert M^0_{ij}\rvert\ \text{large};\qquad \text{hedge }(i,j),(j,i)\ \text{resolved by }\lvert M^0_{ij}\rvert\ \text{vs}\ \lvert M^0_{ji}\rvert.$$

```text
lasso                    decides in the MIDDLE of the path, where the fit is a compromise with sparsity anyway:
                         only the ORDER in which entries enter matters, and a shift of this size hardly changes it
                         (Dettling's robustness of the lasso to a misspecified C is this)
dense-start estimators   decide at the DENSE END (§3–§5): the lasso solution at the smallest λ, a point of the solution set; keep its large entries
  rescaled C             dense end = truth + null-space component                  -> pruning strips the null-space part
  C = 2I                 dense end = truth + null-space component + shift (§7.2)   -> pruning drops weak true edges, keeps shifted false ones
hedge and lock-in        lasso: direction unclear -> keeps both M_ij, M_ji (hedge); pruning resolves the hedge by |M⁰_ij| vs |M⁰_ji|
                         sparse → dense path: pair enters alone -> one direction fixed -> MCP / SCAD keeps it (lock-in, §1)
measured on              the dense end M⁰: the sizes |M⁰_ij|, exactly the adaptive lasso's weights and the dense → sparse paths' start
```

| lasso `max_f1`, $n=10^3$ | $C=2I$ | $\tilde C$ |
|---|---|---|
| $p=10$ | 0.589 | 0.584 |
| $p=20$ | 0.531 | 0.507 |

| `C_ID` | $\mathrm{AUC}(\lvert M^0_{ij}\rvert)$ | $(i,j)\in\operatorname{supp}\tilde M:\ \lvert M^0_{ji}\rvert>\lvert M^0_{ij}\rvert$ |
|---|---|---|
| $p=10$, $n=\infty$, $C=2I$ | 0.71 | 22 % |
| $p=10$, $n=\infty$, $\tilde C$ | 0.76 | 14 % |
| $p=20$, $n=\infty$, $C=2I$ | 0.88 | 10 % |
| $p=20$, $n=\infty$, $\tilde C$ | 0.93 | 1 % |
| $p=10$, $n=10^4$, $C=2I\to\tilde C$ | 0.66 → 0.73 | |
| $p=20$, $n=10^4$, $C=2I\to\tilde C$ | 0.86 → 0.87 | |

### 7.4 Additivity

```text
dense start    avoids the early orientation lock-in of the standard path (orientation_lock_in_051026.md): helps on its own, with C = 2I; nothing to do with C
+ rescaled C   makes the dense end informative (§7.3)
the extra gain from rescaling follows how much of the misfit the rescaling removes (misfit figures: campaign note §2.2)
table: MCP dense -> sparse against the lasso with the same C, max_f1, p = 20, n = 1e4
```

| true $C$ | gain, $C=2I$ | gain, $\tilde C$ | extra | misfit removed |
|---|---|---|---|---|
| `C_ID` | +0.077 | +0.154 | +0.077 | 100 % |
| `C_Random_Min_Diag` | +0.081 | +0.141 | +0.060 | 95 % |
| `C_Random_Diag` | +0.079 | +0.122 | +0.043 | 40 % |
| `C_Random_Full` | +0.004 | +0.042 | +0.038 | 30 % |

### 7.5 Score

```text
the score's loss term, and with it the search: the likelihood of Σ(M, C); with the wrong C the best-scoring graph is a different
graph (S3b §9; next_steps/051026/next_steps_051026.md §2)
in one sentence: with the wrong C the data say that no sparse graph fits; the lasso, which never insists on an exact fit, is
barely affected, while the estimators that start from an exact fit and prune are pruning the wrong object
```

| lasso + search `max_f1`, $p=20$, $n=10^4$ | $C=2I$ | $\tilde C$ |
|---|---|---|
| | 0.614 | 0.650 |

## References

```text
Fan, J., Xue, L. & Zou, H. (2014). Strong oracle optimality of folded concave penalized estimation. Annals of Statistics 42(3), 819–849.      LLA from a lasso start, §4
Zou, H. & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. Annals of Statistics 36(4), 1509–1533.     the local linear approximation, §4
Zou, H. (2006). The adaptive lasso and its oracle properties. JASA 101(476), 1418–1429.                                                  §5
Varando, G. & Hansen, N. R. (2020). Graphical continuous Lyapunov models. UAI.                                                           paths from a dense start towards sparser solutions
Dettling, P., Drton, M. & Kolar, M. (2024). On the Lasso for graphical continuous Lyapunov models. CLeaR.                                 the direct loss and the simulation design
Dettling, P., Homs, R., Améndola, C., Drton, M. & Hansen, N. R. (2023). Identifiability in continuous Lyapunov models.
    SIAM Journal on Matrix Analysis and Applications 44(4).                                                                               Lemma 27: the exact fits (W - C/2) Σ̂^-1, W skew, §3.1
Burke, J. V. & Ferris, M. C. (1993). Weak sharp minima in mathematical programming. SIAM Journal on Control and Optimization 31(5),
    1340–1359.                                                                                                                            a polyhedral function has a unique minimiser iff the minimum is sharp, §3.1
```
