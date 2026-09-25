# S1 — Reproduction: Direct Lyapunov Lasso (quadratic loss + $\ell_1$)

**Status:** specification + implementation plan. Baseline for the thesis; the SCAD/MCP
variants (S1b) plug into the same harness by swapping only the proximal operator.

**Target:** reproduce Dettling, Drton & Kolar (2024), *On the Lasso for Graphical Continuous
Lyapunov Models*, PMLR v236 — Section 5 (Figure 5) and Section 1.3 (Figure 3).
Data-generating setup follows Varando & Hansen (2020), PMLR v124.

---

## 1. Model

Each observation is one cross-section of an independent $p$-dimensional Ornstein–Uhlenbeck
process in equilibrium,

$$dX(t) = M\big(X(t)-a\big)\,dt + D\,dW(t),$$

with **drift matrix** $M \in \mathbb{R}^{p\times p}$ *stable* (all eigenvalues have strictly
negative real part) and **volatility** $C = DD^\top \succ 0$. We take $a = 0$ throughout.
The equilibrium distribution is $\mathcal{N}(0,\Sigma)$ where $\Sigma$ is the unique solution of the
**continuous Lyapunov equation**

$$M\Sigma + \Sigma M^\top + C = 0. \tag{1.2}$$

The graph $G=(V,E)$ has $V=\{1,\dots,p\}$ and $i \to j \in E \iff M_{ji} \neq 0$.
Stability forces $M_{ii} < 0$, so all self-loops are always present and are not drawn.

**Scaling.** $(M,C) \mapsto (\gamma M, \gamma C)$ leaves $\Sigma$ unchanged, so $C$ is only
identifiable up to a positive scalar. This is why estimating with $C = 2I_p$ while generating
with $C=I_p$ is harmless for *support* recovery — it only rescales $\hat M$.

## 2. Estimator

**Direct Lyapunov Lasso** (Dettling eq. 1.4):

$$\hat M(\lambda) \;=\; \arg\min_{M \in \mathbb{R}^{p\times p}}\;
\tfrac12\big\|M\hat\Sigma + \hat\Sigma M^\top + C\big\|_F^2 \;+\; \lambda\|M\|_1,
\qquad
\hat\Sigma = \tfrac1n\sum_{i=1}^n X_iX_i^\top. \tag{1.3–1.4}$$

Note $\hat\Sigma$ uses $1/n$ (not $1/(n-1)$) and no centering, since $a=0$.

### 2.1 Vectorized form (Dettling §2) — the regression view

$$A(\Sigma) \;=\; (\Sigma \otimes I_p) + (I_p \otimes \Sigma)K_{(p,p)} \;\in\; \mathbb{R}^{p^2\times p^2}, \tag{2.2}$$

with $K_{(p,p)}$ the commutation matrix ($K\,\mathrm{vec}(A) = \mathrm{vec}(A^\top)$) and
$\mathrm{vec}$ column-stacking. Then (1.2) reads $A(\Sigma)\mathrm{vec}(M) + \mathrm{vec}(C) = 0$
and the objective becomes an **ordinary lasso regression**

$$\tfrac12\big\|\,y - X\beta\,\big\|_2^2 + \lambda\|\beta\|_1,
\qquad X = A(\hat\Sigma),\quad y = -\mathrm{vec}(C),\quad \beta = \mathrm{vec}(M).$$

Gram matrix and linear term (Dettling Lemma 1, eqs. 2.3–2.4):

$$\Gamma(\Sigma) = 2(\Sigma^2\otimes I_p) + (\Sigma\otimes\Sigma)K_{(p,p)} + K_{(p,p)}(\Sigma\otimes\Sigma),
\qquad g(\Sigma) = -A(\Sigma)\mathrm{vec}(C).$$

$A(\Sigma)$ deliberately keeps **two identical rows** for each off-diagonal entry of the
(symmetric) Lyapunov residual. This redundancy is what makes the regression loss equal
$\tfrac12\|\cdot\|_F^2$, which counts off-diagonal residuals twice. **Do not deduplicate rows.**

### 2.2 Matrix-free form — what we actually optimize

Forming $X$ costs $O(p^4)$ memory. With $R(M) := M\hat\Sigma + \hat\Sigma M^\top + C$,

$$f(M) = \tfrac12\|R(M)\|_F^2, \qquad \nabla f(M) = 2\,R(M)\,\hat\Sigma \quad [O(p^3)],$$

and $\nabla f$ is Lipschitz with $L = \|A(\hat\Sigma)\|_2^2 \le 4\,\lambda_{\max}(\hat\Sigma)^2$.
This is the form used for production runs (and the one that generalises to SCAD/MCP by
swapping the prox).

### 2.3 The design matrix is rank deficient — always

Because the Lyapunov residual is symmetric, only $p(p+1)/2$ of the $p^2$ rows of $A(\Sigma)$ are
independent.

> **Read the next equation as a statement about the null space of a matrix, not as a Lyapunov
> equation.** $V$ below is *not* a drift matrix and $C$ is *not* being set to zero. In the
> regression view of §2.1, $A(\Sigma)$ is the design matrix $X$ and $C$ is only the response
> $y = -\mathrm{vec}(C)$; the rank and null space of a design matrix are properties of $X$ alone.
> The zero on the right-hand side arises by **differencing**: if two candidate drift matrices
> $M_1, M_2$ leave the same residual, then subtracting
> $A\,\mathrm{vec}(M_1) + \mathrm{vec}(C) = r$ from
> $A\,\mathrm{vec}(M_2) + \mathrm{vec}(C) = r$ cancels $\mathrm{vec}(C)$ and leaves
> $A\,\mathrm{vec}(M_1 - M_2) = 0$. So $V = M_1 - M_2$ is a *direction* in parameter space along
> which the loss does not change — for **any** $C$.

Concretely, using $\Sigma^\top = \Sigma$ so that $\Sigma V^\top = (V\Sigma)^\top$,

$$A(\Sigma)\,\mathrm{vec}(V) = 0
\iff V\Sigma + \Sigma V^\top = 0
\iff V\Sigma + (V\Sigma)^\top = 0
\iff V\Sigma \text{ is skew-symmetric}.$$

Since $V \mapsto V\Sigma$ is a bijection for $\Sigma \succ 0$, the null space is
$\{\,W\Sigma^{-1} : W^\top = -W\,\}$ and has dimension exactly $p(p-1)/2$ — the dimension of the
skew-symmetric matrices — for **every** $\Sigma$, independently of $C$. Hence

$$\mathrm{rank}\,A(\Sigma) = \mathrm{rank}\,\Gamma(\Sigma) = \tfrac{p(p+1)}{2} < p^2 .$$

Verified numerically in `tests/test_lasso.py::test_design_matrix_is_rank_deficient`, and the
$C$-independence in `test_null_space_is_flat_for_every_C`.

Three consequences that matter downstream:

1. **The smooth part is never strongly convex.** $\Gamma(\Sigma)$ is singular regardless of $n$ or
   $p$. Global strong convexity is therefore unavailable *by construction*, not merely in the
   $p > n$ regime. This is exactly why the nonconvex extension (S1b) must argue via **restricted
   strong convexity** on a cone rather than plain convexity — it pins down the condition flagged
   in `plan.md` ("we would need RSC instead").
2. **The minimizer can be non-unique.** Uniqueness requires the active columns $A_{\cdot S}$ to be
   linearly independent; at the dense end of the path they are not. Any comparison of *coefficients*
   between implementations must be guarded by that rank check (`solution_is_unique` in the tests).
   Support-recovery metrics are unaffected in practice.
3. **Never solve via $\Gamma^{-1}$.** Use the KKT system restricted to the active set, which is
   well conditioned whenever $A_{\cdot S}$ has full column rank.

## 3. Data-generating process

### 3.1 Drift matrix (Dettling §5, following Varando & Hansen §4)

$$M_{ij} = \omega_{ij}\,\varepsilon_{ij}\ (i\neq j), \qquad
M_{ii} = -\sum_{j\neq i}|M_{ij}| - |\varepsilon_{ii}|,$$

with $\omega_{ij}\sim\mathrm{Bernoulli}(d)$ and $\varepsilon_{ij}\sim\mathcal{N}(0,1)$, all independent.
The diagonal rule makes $M$ strictly diagonally dominant with negative diagonal, so by
Gershgorin **every draw is stable** — no rejection sampling needed.

> ⚠️ **Discrepancy to record.** Varando's released code (`functions/util.R::rStableMetzler`)
> uses `abs(rfun(...))`, producing **non-negative** off-diagonals (a Metzler matrix), whereas
> both papers' *text* writes $M_{ij}=\omega_{ij}\varepsilon_{ij}$ with $\varepsilon_{ij}\sim N(0,1)$, i.e. **signed**.
> We follow the text (signed) as default and expose `metzler: bool` to switch.

### 3.2 Volatility matrix — Dettling's four choices

| # | Label | Definition |
|---|-------|------------|
| 1 | `C_ID` | $C = 2I_p$ |
| 2 | `C_RANDOM_DIAG` | diagonal, $C_{ii}\sim\mathrm{Unif}[0.5,4]$ |
| 3 | `C_RANDOM_MIN_DIAG` | diagonal, $C_{ii}\sim\mathrm{Unif}[2,4]$ |
| 4 | `C_RANDOM_FULL` | $\tilde\omega_{ij}\sim\mathrm{Bernoulli}(2/p)$, $\tilde\varepsilon_{ij}\sim N(0,1)$; off-diagonals $C_{ij}=\tilde\omega_{ij}\tilde\varepsilon_{ij}+\tilde\omega_{ji}\tilde\varepsilon_{ji}$; diagonal $C_{ii}=\sum_{j\neq i}\lvert C_{ij}\rvert+\lvert\tilde\varepsilon_{ii}\rvert+0.5$ |

Choice 4 is symmetric and diagonally dominant, hence positive definite. Choices 2–4 are
**misspecification** settings: estimation always uses $C = 2I_p$.

### 3.3 Sampling

$\Sigma = $ solution of (1.2) for the drawn $(M,C)$; then $X_1,\dots,X_N \stackrel{iid}{\sim} \mathcal{N}(0,\Sigma)$.

## 4. Configuration

```yaml
# S1 — Dettling Section 5 / Figure 5
p:            [10, 15, 20, 25, 30, 40, 50]   # p^2 > N for p in {40, 50}: high-dimensional
k:            [1, 2, 3, 4]                   # edge probability d = k / p
C_choice:     [C_ID, C_RANDOM_DIAG, C_RANDOM_MIN_DIAG, C_RANDOM_FULL]
n_rep:        100                            # (M, C) pairs per (p, k, C_choice)
N:            1000                           # observations per dataset
C_estimation: 2 * I_p                        # always, regardless of C_choice
n_lambda:     100
lambda_ratio: 1.0e-4                         # lambda_1 = lambda_max / 1e4
lambda_grid:  log-equidistant, lambda_max * 10 ** linspace(-4, 0, 100)
```

Grid definition (Dettling §5): $0 < \lambda_{\max}/10^4 = \lambda_1 < \dots < \lambda_{100} = \lambda_{\max}$,
with $\lambda_{\max}$ the **smallest** $\lambda$ for which $\hat M$ is diagonal.
Matches Varando's `lambdaseq <- 10^(seq(-4, 0, length = 100))`.

**Aggregation.** Each metric is averaged over the 4 sparsity levels $k$ and the 100 replicates
$\Rightarrow$ $4\times100 = 400$ drift matrices per $(p, \texttt{C\_choice})$ point in Figure 5.
This matches the Figure 5 caption ("over the 400 randomly generated drift matrices") and is a
useful arithmetic check that the loop structure is right. Error bars = standard error of that mean.

**Total cost.** $7\times4\times4\times100 = 11{,}200$ datasets $\times$ 100 $\lambda$ values
$= 1.12$M lasso fits. Embarrassingly parallel over `(p, k, C_choice, rep)`.

### 4.1 Open settings — flags, not assumptions

These are places where the paper text is silent or the two sources disagree. Each is a config
flag defaulting to the value we believe matches Dettling, with the alternative testable.

| Flag | Default | Why it is open |
|---|---|---|
| `penalize_diagonal` | `False` | Eq. (1.4) literally writes $\lambda\|M\|_1$ (all entries). But Dettling used `glmnet`, and Varando's `lassoB` passes `penalty.factor = 1 - diag(p)`, i.e. **diagonal unpenalized**. Unpenalized diagonal is also what makes "smallest $\lambda$ such that $\hat M$ is diagonal" a well-posed, closed-form quantity — if the diagonal were penalized, large $\lambda$ would drive $\hat M \to 0$, which is *also* diagonal. |
| `metrics_include_diagonal` | `False` | **RESOLVED — see §8.5.** Definition G.4 counts over all $\hat M_{ij}$ with no stated exclusion, but Varando's `evaluatePathB` scores `lower.tri \| upper.tri` only. Settled empirically in favour of `False`. |
| `standardize` | `True` | **RESOLVED — see §8.6.** Varando's `simulate.R` uses `cor(data)`; Dettling never mentions it. Settled empirically in favour of `True`: it is what brings all four metrics inside Figure 5's ranges across $10 \le p \le 50$. Standardization does **not** change the support: if $\Sigma = D R D$ then the correlation-scale drift is $D^{-1}MD$, same zero pattern. |
| `cov_denominator` | `n` | Dettling eq. (1.3) is explicit: $1/n$. |

> **Floor argument for `metrics_include_diagonal`.** If diagonal entries are scored, they are
> always true positives (they never get zeroed), so at the sparsest end of the path
> ($\hat M$ diagonal) we get $tp=p$, $fp=0$, $fn = \#\{\text{off-diag nonzeros}\}$. At $p=30$,
> $k=2.5$ on average, that is $F_1 = 2\cdot30/(2\cdot 30+75) \approx 0.44$ — suspiciously close to
> the lower end of the reported `max_f1` range (0.45–0.60) in Figure 5. So including the
> diagonal is *not* ruled out by the figure. **Resolved in §8.5: the answer is `False`.** The
> argument below was the reason to check, not the conclusion — kept for the record, superseded.

## 5. Metrics (Dettling Definitions G.4, G.5)

Per $\lambda$, comparing $\hat M$ to $M^*$ over the scored index set (see `metrics_include_diagonal`):

$$tp=\#\{\hat M_{ij}\neq0,\,M^*_{ij}\neq0\},\quad fp=\#\{\hat M_{ij}\neq0,\,M^*_{ij}=0\},\quad
tn, fn \text{ analogously}$$

$$tpr=\tfrac{tp}{tp+fn},\quad fpr=\tfrac{fp}{fp+tn},\quad acc=\tfrac{tp+tn}{tp+tn+fp+fn},\quad
F_1=\tfrac{2tp}{2tp+fp+fn},\quad pr=\tfrac{tp}{tp+fp}$$

Reported per dataset: `max_acc`, `max_f1` (maxima over the $\lambda$-path), `auc`, `aupr`.

**Curve construction** (following Dettling's "interpolation and extrapolation if necessary",
implemented as in Varando's `util.R`, which is the concrete recipe Dettling inherited):

- **ROC**: collect $(fpr,tpr)$ along the path, order by increasing $fpr$, prepend $(0,0)$ and
  append $(1,1)$, integrate by the trapezoid rule.
- **PR**: plot $pr$ vs. $tpr$ (recall), trapezoid rule, plus a closing segment from the
  highest-recall point to $(recall=0, pr=1)$ — this mirrors `AUCPR` in `util.R`.
- Degenerate cases: $tpr := 1$ if $tp+fn=0$; $fpr := 1$ if $fp+tn=0$; $pr := 1$ if $tp+fp=0$;
  $F_1 := 0$ if undefined. (Exactly the `NaN` handling in `evaluatePathB`.)

## 6. Milestones

**M0 — Example 2 / Figure 3 (cheap smoke test, do this first).**
$p=5$. $G_1$ = path $1\to2\to3\to4\to5$, $M^*_1$ with diagonal $(-2,-3,-4,-5,-6)$ and the four
sub-diagonal entries $=0.65$. $G_2$ = 5-cycle, adding $m_{15}$: (a) fixed $=0.65$, (b) 100 draws
$\sim\mathrm{Unif}[0.5,1]$. $C = 2I_p$. 100 datasets per sample size,
$n \in \{100, 200, 500, 1000, 5000, 10^4, \mathbf{10^5}, \infty\}$ ($n=\infty$ means feed the population
$\Sigma$). Report `max_acc`, `max_f1`, `auc`.
*Expected qualitative result:* the path is recovered nearly perfectly by $n=10^4$; the cycle is
**not** recovered even as $n\to\infty$ — this is the irrepresentability failure, and it is the single
sharpest check that the whole pipeline is correct.

> Note: Appendix A lists $n$ up to $10^5$, but the Figure 3 axis runs $100,\dots,10^4,\infty$.
> Minor inconsistency in the paper; we compute $10^5$ and simply have one extra point.

**M1 — Figure 5.** Full grid of §4. Run `C_ID` alone first ($p \le 20$) to fix the open flags in
§4.1 against the published curve, then extend.

**M2 — hand-off to S1b.** Freeze the harness; swap $\lambda\|M\|_1$ for MCP/SCAD by replacing the
proximal operator only. Everything in §3–§5 is reused unchanged so the comparison is apples-to-apples.

## 7. Implementation plan

### 7.1 Layout

```
src/gclm/
  lyap.py       vec/unvec, commutation_matrix, design_matrix A(Sigma), solve_lyapunov
  dgp.py        sample_drift, sample_volatility (4 choices), sample_data
  loss.py       frobenius_loss, frobenius_grad, lipschitz_bound
  lasso.py      4 solver backends + lambda_max + path + lambda conversions
  metrics.py    Definitions G.4/G.5, ROC/PR curves, auc/aupr
  examples.py   the fixed 5-node models of Example 2
  rbridge.py    subprocess/JSON bridge to R; R stays optional
  config.py     S1Config / M0Config dataclasses
simulations/
  S1_reproduction.md      (this file)
  run_m0.py               driver for Figure 3
  run_s1.py               driver for Figure 5; --solver, --penalty, --shard
  results/                committed run outputs
R/
  backend_glmnet.R        production backend + validation; port of Varando's lassoB()
  backend_ncvreg.R        production backend via ncvreg::ncvfit; MCP/SCAD for S1b
  reference_metrics.R     port of evaluatePathB()/AUROC()/AUCPR(); validation only
tests/                    pytest
```

### 7.2 Four solver backends

Selected with `lasso_path(..., solver=...)`, or `run_s1.py --solver`. All four minimize the *same*
objective and are cross-checked against each other in `test_all_four_backends_agree`.

| `solver` | implementation | accuracy vs. exact KKT | cost, 100-λ path | use it for |
|---|---|---|---|---|
| **`fista`** *(default)* | Python, matrix-free accelerated proximal gradient | 2e-13 at `tol=1e-14`; ~1e-4 at the production `tol=1e-8` | $p{=}10$ 0.9 s, $p{=}20$ 5 s | **everything at scale** — the only backend that reaches $p=50$ |
| **`ncvreg`** | R, `ncvreg::ncvfit` | **4e-12**, in 8–48 iterations | $p{=}10$ 10 s, $p{=}20$ 244 s | accuracy, and **MCP/SCAD in S1b** |
| **`glmnet`** | R, transcription of Varando's `lassoB()` | 1e-4 – 7e-2 (see below) | $p{=}10$ ~1 s | **fidelity to Dettling**, who used it |
| **`design`** | Python, coordinate descent on explicit $A(\hat\Sigma)$ | 4e-8 | slow (~58 s at $p{=}8$) | a transparent reference in tests |

#### What the papers specify

Dettling gives no bespoke algorithm for S1 — Appendix A says the Direct Lyapunov Lasso is fitted
with *"the R package `glmnet`, which runs a coordinate descent algorithm"*, and Varando's
`lassoB()` does the same. That is the `glmnet` backend, transcribed in `R/backend_glmnet.R`.

Varando's **Algorithm 1** *is* detailed proximal-gradient pseudocode, but for a different
objective: $L(\Sigma(B,C)) + \lambda\rho_1(B) + \kappa\|C-I_p\|_F^2$ with $L$ the negative Gaussian
log-likelihood or $\|\Sigma-\hat\Sigma\|_F^2$. It optimizes over $B$ *and* $C$, solves a Lyapunov
equation every iteration, and line-searches to keep $B$ stable — none of which applies here.
**Algorithm 1 is the reference for S2, not S1**, and should be transcribed there.

`solve_fista` is **not** taken from either paper. It is standard FISTA
(Beck & Teboulle 2009) with the gradient-based adaptive restart of
O'Donoghue & Candès (2015); `ncvreg` is Breheny & Huang (2011). See References.

#### Two defects in `glmnet` on this problem

Both were found by cross-checking against the other backends, and both are why `glmnet` is offered
for fidelity rather than as the default.

1. **It silently truncates the λ path.** On this rank-deficient design (§2.3) glmnet's coordinate
   descent stops converging near the dense end, and its response is to return *fewer* λ values, not
   an error. Measured at $p=8$ over 100 λ: `thresh=1e-10` fits all 100, `1e-12` only 48, `1e-14`
   only 42. `R/backend_glmnet.R` therefore walks a **threshold ladder**
   (`1e-14 → 1e-12 → 1e-10 → 1e-8 → 1e-7`), takes the tightest setting that fits the whole path, and
   returns `thresh_used`. Pinned by `test_glmnet_backend_returns_every_requested_lambda`.
2. **Its accuracy is path-dependent.** Where the ladder reaches `1e-14` (e.g. $p=5$, 40 λ) glmnet
   lands ~3.6e-4 from the exact KKT solution; where it must loosen to `1e-10`, up to **6.8e-2**, and
   supports then differ at 5 of 40 λ values. That is visible in the *reported metrics*: on a small
   $p=8$ run, `glmnet` gives `max_f1` 0.677 against 0.685 for `fista` and `ncvreg`, which agree
   exactly.

**Consequence for the thesis.** Support-recovery conclusions are robust — all four backends produce
the same qualitative picture. But S1b's claim is about *estimation error*, and there `glmnet` is not
accurate enough to measure what is being claimed. Use `fista` at a tight `tol`, or `ncvreg`.

#### MCP / SCAD

`plan.md` records the advisor's suggestion to use an off-the-shelf package for the nonconvex
penalties: *"you can use any package that does linear regression with such penalties, you have to
create a suitable response vector y and a design matrix X out of your input."* That is `ncvreg`,
and it is wired up now:

```bash
python simulations/run_s1.py --solver ncvreg --penalty MCP   --gamma 3
python simulations/run_s1.py --solver ncvreg --penalty SCAD  --gamma 3.7
```

`ncvfit` is the right entry point, not `ncvreg()`: the latter always standardizes the design and
always fits an intercept, neither of which this problem wants ($y = -\mathrm{vec}(C)$ has no
constant term). `ncvfit` does neither and accepts `penalty.factor`, so the unpenalized diagonal is
expressible. Smoke-tested in `test_ncvreg_backend_supports_nonconvex_penalties`; the statistical
study itself is S1b and is not yet run.

#### λ conversions

Each backend has its own parameterization; the conversions are in `lasso.py` and asserted by tests,
not trusted from documentation.

| backend | objective | conversion |
|---|---|---|
| `glmnet` | $\frac{1}{2n}\|y-X\beta\|^2 + \lambda_g\sum_j v_j\lvert\beta_j\rvert$, $n=p^2$, **and `penalty.factor` internally rescaled to sum to `nvars`** | $\lambda = \lambda_g\,p^3/(p-1)$ |
| `ncvfit` | $\frac{1}{2n}\|y-X\beta\|^2 + \lambda_n\sum_j \mathrm{pf}_j\,\mathrm{pen}(\beta_j)$, $n=p^2$, **no rescaling** | $\lambda = \lambda_n\,p^2$ |

The presence of the rescaling in `glmnet` and its absence in `ncvfit` were both established
empirically (`test_glmnet_lambda_scaling`, `test_fista_matches_ncvreg`).

**[`R/ENCODING.md`](../R/ENCODING.md)** walks the whole encoding — $X$, $y$, `penalty.factor`, the
index map from coefficient to $M_{ik}$, every argument, and both λ conversions — through a $p=3$
example small enough to check by hand. Its numbers are pinned by `tests/test_encoding.py`.

### 7.3 $\lambda_{\max}$

With the diagonal unpenalized, $\hat M(\lambda)$ is diagonal iff the KKT condition holds at the
diagonal-only least-squares solution $\hat M_{\mathrm{diag}}$, giving a **closed form**:

$$\lambda_{\max} = \max_{i \neq j} \big| [\,2\,R(\hat M_{\mathrm{diag}})\,\hat\Sigma\,]_{ij} \big|,
\qquad
\hat M_{\mathrm{diag}} = \arg\min_{M \text{ diagonal}} f(M),$$

where the inner problem is a $p$-variable least squares (residual entries are
$R_{ij} = (m_i+m_j)\hat\Sigma_{ij} + C_{ij}$, linear in $m$).
If `penalize_diagonal=True`, fall back to bisection on $\log\lambda$ for the smallest $\lambda$
with zero off-diagonal.

### 7.4 Validation against R

| Test | Against | Tolerance |
|---|---|---|
| `test_lyap_solution` | `scipy.linalg.solve_continuous_lyapunov` vs. direct residual $\|M\Sigma+\Sigma M^\top+C\|_F$ | $10^{-10}$ |
| `test_design_matrix_matches_r` | $A(\Sigma)$ vs. R's `Sigma %x% diag(p) + (diag(p) %x% Sigma) %*% TT` | exact (1e-12) |
| `test_gradient` | $\nabla f = 2R\Sigma$ vs. finite differences and vs. $X^\top(X\beta-y)$ | 1e-6 |
| `test_lasso_path_matches_glmnet` | objective + support vs. `glmnet` on a shared $(\hat\Sigma, C, \lambda)$ grid | obj 1e-8; coef 1e-3 (see §8.2) |
| `test_solvers_match_exact_restricted_solution` | both solvers vs. the closed-form KKT solution on the active set | 1e-9 |
| `test_design_and_fista_agree` | `solve_fista` vs. `solve_design` | 1e-6 |
| `test_fista_matches_ncvreg` | `solve_fista` vs. `ncvreg::ncvfit` | **1e-9** — the sharpest external check |
| `test_all_four_backends_agree` | all four backends: identical supports and metrics | metrics exact |
| `test_glmnet_backend_returns_every_requested_lambda` | glmnet never returns a truncated path | — |
| `test_ncvreg_backend_supports_nonconvex_penalties` | MCP/SCAD plumbing (S1b entry point) | smoke |
| `test_example2_*` | Dettling Figure 3 population-limit values | 0.02–0.03 abs |
| `test_glmnet_lambda_scaling` | our $\lambda_{\max}$ vs. `glmnet`'s first all-diagonal $\lambda$, through $p^3/(p-1)$ | 3% in $\log\lambda$ |
| `test_all_metrics_match_r` | `acc/f1/tpr/fpr/auc/aupr` vs. a port of `evaluatePathB`/`AUROC`/`AUCPR` | 1e-12, exact |
| `test_dgp_stability` | every sampled $M$ stable; every sampled $C \succ 0$ and symmetric | — |

R is invoked via `Rscript` from `pytest`; the R-dependent tests skip cleanly (`pytest.mark.skipif`)
when `Rscript` or `glmnet` is unavailable, so CI and the cluster do not need R.

**Seeding.** Python uses `numpy.random.default_rng(seed)`; R fixtures are generated *once* from
Python-written inputs (Python generates $\hat\Sigma$, $C$, $\lambda$; R only fits), so the two
languages never need matching RNG streams.

### 7.5 Compute

Per-$\lambda$ FISTA cost is $O(p^3)$ per iteration, but wall-clock is dominated by iteration
count rather than by $p$ (see §8.3): roughly 5 s per dataset at `tol=1e-8`, so the full M1 grid is
~55 core-hours. Parallelize over the `(p, k, C_choice, rep)` grid with one process per core;
each task writes one row of metrics. No inter-task communication, so this maps directly onto a
cluster array job once access is available. Until then M0 and the small-$p$ slice of M1 run on a
laptop.

---

## 8. Findings so far

### 8.1 M0 reproduces Figure 3 (validated)

`python simulations/run_m0.py --reps 100`. Five replicates per cell already match the published
figure closely:

| setting | n=100 | 200 | 500 | 1000 | 5000 | 10⁴ | 10⁵ | ∞ |
|---|---|---|---|---|---|---|---|---|
| path `max_acc` | 0.820 | 0.860 | 0.870 | 0.940 | 0.960 | **1.000** | 1.000 | 1.000 |
| path `max_f1` | 0.594 | 0.685 | 0.722 | 0.865 | 0.916 | **1.000** | 1.000 | 1.000 |
| path `auc` | 0.764 | 0.823 | 0.875 | 0.975 | 0.984 | **1.000** | 1.000 | 1.000 |
| cycle-fixed `max_acc` | 0.750 | 0.790 | 0.850 | 0.850 | 0.890 | 0.880 | 0.900 | **0.900** |
| cycle-fixed `max_f1` | 0.502 | 0.573 | 0.656 | 0.688 | 0.777 | 0.771 | 0.800 | **0.800** |
| cycle-fixed `auc` | 0.581 | 0.663 | 0.777 | 0.784 | 0.825 | 0.827 | 0.833 | **0.833** |

All three of Dettling's qualitative claims hold:

- the path beats the cycle at **every** sample size and **every** metric;
- the path reaches perfect recovery at $n = 10^4$;
- the cycle plateaus at $(0.90, 0.80, 0.833)$ and does **not** improve with more data — the
  irrepresentability failure, visible even at $n=\infty$;
- averaging over random completions $m_{15}\sim\mathrm{Unif}[0.5,1]$ gives essentially the same
  numbers as the fixed $m_{15}=0.65$, matching "averaging over various completions does not
  improve the metrics".

The $n=\infty$ values are exact and deterministic, so they are locked in as regression tests
(`test_example2_path_is_recovered_in_population_limit`, `..._cycle_...`).

### 8.2 `glmnet` is the less accurate solver at the dense end of the path

Comparing on a shared problem at $\lambda \approx 8\times10^{-4}$ (the sparse end agrees to
machine precision):

| solver | objective | max abs. deviation from the exact KKT solution |
|---|---|---|
| our FISTA | 0.003553737286006 | 2.0e-13 |
| our coordinate descent | 0.003553737286006 | 4.5e-08 |
| `glmnet` (`thresh = 1e-14`) | 0.003553737331701 | **3.6e-04** |

The "exact KKT solution" here is computed in closed form from the active set
($b_S = (A_S^\top A_S)^{-1}(A_S^\top y - \lambda w_S \mathrm{sign}(b_S))$), so it does not depend on
any iterative solver. `glmnet` stops on a relative objective change, and near the dense end the
objective is very flat (§2.3), so it terminates early.

**Why this matters for the thesis.** It is harmless for Dettling's published results, which are all
*support-recovery* metrics — the supports still agree. But the case for SCAD/MCP in `plan.md` is
partly about **estimation error / MSE**, and at $10^{-4}$ accuracy `glmnet` would contaminate exactly
that comparison. S1b must therefore not use `glmnet` as the $\ell_1$ baseline; use our solvers,
which are verified against the closed-form KKT solution.

### 8.3 Solver tolerance: 1e-8 is the right production default

FISTA stops on the max-norm coefficient change. One dataset per $p$, $k=2$, `C_ID`, $N=1000$,
full 100-point path:

| $p$ | tol | path time | `max_acc` | `max_f1` | `auc` | `aupr` | supports identical to 1e-10 |
|---|---|---|---|---|---|---|---|
| 10 | 1e-10 | 1.33 s | 0.8556 | 0.4800 | 0.6969 | 0.3741 | 1.00 |
| 10 | 1e-8 | 0.91 s | 0.8556 | 0.4800 | 0.6964 | 0.3630 | 0.99 |
| 10 | 1e-6 | 0.34 s | 0.8556 | 0.4800 | 0.6911 | 0.3744 | 0.71 |
| 20 | 1e-10 | 9.18 s | 0.8921 | 0.4375 | 0.8021 | 0.3089 | 1.00 |
| 20 | 1e-8 | 5.10 s | 0.8921 | 0.4375 | 0.8019 | 0.3089 | 0.94 |
| 20 | 1e-6 | 1.12 s | 0.8921 | 0.4375 | 0.8041 | 0.3132 | 0.66 |
| 30 | 1e-10 | 10.17 s | 0.9368 | 0.4444 | 0.7455 | 0.3529 | 1.00 |
| 30 | 1e-8 | 6.49 s | 0.9368 | 0.4444 | 0.7453 | 0.3529 | 0.95 |
| 30 | 1e-6 | 1.89 s | 0.9368 | 0.4444 | 0.7453 | 0.3530 | 0.67 |
| 40 | 1e-10 | 13.74 s | 0.9564 | 0.3681 | 0.8356 | 0.2881 | 1.00 |
| 40 | 1e-8 | 6.73 s | 0.9564 | 0.3681 | 0.8356 | 0.2881 | 0.91 |
| 40 | 1e-6 | 2.35 s | 0.9564 | 0.3681 | 0.8344 | 0.2881 | 0.62 |

`max_acc` and `max_f1` are completely insensitive — they depend only on the best point of the path.
`auc`/`aupr` use the whole curve and so are mildly sensitive, but still agree to ~0.001 at 1e-8.
**Default: `tol = 1e-8`** (`S1Config.tol`). The looser settings are not worth the risk for a
1–3 s saving, and any *estimation-error* study (S1b's MSE comparison) should tighten to 1e-10.

Note that the cost is dominated by iteration count, not by $p$: $p=40$ costs only ~30% more than
$p=20$, nowhere near the $O(p^3)$-per-iteration scaling would suggest. The single-dataset figures
above understate the full-grid average, which the 1,120-dataset run of §8.7 measures at ~17 s: the
full M1 grid (11,200 datasets) is ~55 core-hours, about 7 h on 8 cores, and trivial as a cluster
array job.

### 8.4 First M1 slice: ordering reproduces; levels initially did not (superseded by §8.6)

`python simulations/run_s1.py --p 10 15 20 --reps 10` — 480 datasets, 4.2 min on 8 cores,
averaged over $k=1,\dots,4$ as in Figure 5, default flags (off-diagonal metrics, no standardization):

| $p$ | C choice | `max_acc` | `max_f1` | `auc` | `aupr` | s/dataset |
|---|---|---|---|---|---|---|
| 10 | `C_ID` | 0.807 | 0.566 | 0.700 | 0.384 | 1.5 |
| 10 | `C_Random_Min_Diag` | 0.803 | 0.561 | 0.713 | 0.367 | 2.1 |
| 10 | `C_Random_Diag` | 0.790 | 0.518 | 0.669 | 0.331 | 1.8 |
| 10 | `C_Random_Full` | 0.775 | 0.486 | 0.629 | 0.277 | 1.9 |
| 15 | `C_ID` | 0.853 | 0.482 | 0.715 | 0.324 | 3.4 |
| 15 | `C_Random_Min_Diag` | 0.859 | 0.493 | 0.740 | 0.340 | 3.8 |
| 15 | `C_Random_Diag` | 0.849 | 0.448 | 0.687 | 0.297 | 4.0 |
| 15 | `C_Random_Full` | 0.839 | 0.403 | 0.655 | 0.217 | 4.7 |
| 20 | `C_ID` | 0.886 | 0.443 | 0.743 | 0.296 | 6.6 |
| 20 | `C_Random_Min_Diag` | 0.885 | 0.440 | 0.735 | 0.291 | 5.7 |
| 20 | `C_Random_Diag` | 0.885 | 0.425 | 0.709 | 0.288 | 5.6 |
| 20 | `C_Random_Full` | 0.879 | 0.354 | 0.682 | 0.193 | 8.6 |

**What matches Figure 5** — every qualitative claim in Dettling §5:

- `C_ID` and `C_Random_Min_Diag` are best and nearly indistinguishable ("few differences in all
  metrics among the choices between choice 1) and choice 3)");
- `C_Random_Diag` sits below them and its gap widens with $p$;
- `C_Random_Full` is worst on every metric at every $p$ — the expected penalty for generating with a
  non-diagonal $C$ and estimating with $C = 2I_p$;
- `max_acc` rises with $p$ while `max_f1` and `aupr` fall.

**What can and cannot be checked from the paper.** Figure 5 is a plot; the text gives us its axis
tick ranges but not the per-curve values. So the available checks are (i) the ordering of the four
$C$ choices, above, and (ii) whether our numbers fall inside the published axis ranges:

| metric | Figure 5 axis range | our range, $p \in \{10,15,20\}$ | inside? |
|---|---|---|---|
| `max_acc` | 0.80 – 0.95 | 0.775 – 0.886 | yes, except `C_Random_Full` at $p=10$ |
| `max_f1` | 0.45 – 0.60 | 0.354 – 0.566 | high end yes; $p=20$ dips below |
| `auc` | 0.70 – 0.85 | 0.629 – 0.743 | low end only |
| `aupr` | 0.30 – 0.45 | 0.193 – 0.384 | low end only |

Our values sit at or below the bottom of each range. **This was the `standardize` flag; see §8.6, which closes the gap.** Note the axes span all four $C$ choices and all
seven $p$, and the published ranges must also accommodate $p$ up to 50, so partial overlap at
$p \le 20$ is weak evidence either way — this comparison cannot be sharpened without the figure's
underlying data.

### 8.5 Resolving `metrics_include_diagonal`

Direct test, `C_ID`, averaged over $k = 1,\dots,4$ with 15 replicates each:

| $p$ | `include_diagonal` | `max_acc` | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|---|
| 10 | `False` | 0.802 | 0.535 | 0.688 | 0.363 |
| 10 | `True` | 0.822 | **0.702** | 0.793 | **0.673** |
| 20 | `False` | 0.886 | 0.445 | 0.725 | 0.300 |
| 20 | `True` | 0.892 | **0.635** | 0.816 | **0.633** |

Scoring the diagonal inflates everything, and the curve-based metrics most — as predicted, since the
diagonal is recovered at every $\lambda$ and so contributes true positives all along the path.

**Conclusion: keep `metrics_include_diagonal = False`.** With it on, `max_f1` and `aupr` land
*outside* the published axis ranges at both $p$ — 0.702 / 0.673 at $p=10$ and 0.635 / 0.633 at
$p=20$, against axes that top out at 0.60 and 0.45 — and Figure 5 must additionally fit $p$ up to 50
on those same axes. Off-diagonal scoring keeps both inside at both $p$.
This also agrees with Varando's `evaluatePathB`, which scores `lower.tri | upper.tri` only.

The §4.1 floor argument pointed the other way and is now retired: an $F_1$ floor near 0.44 is
consistent with the axis *minimum*, but the corresponding *maximum* would have to exceed the axis,
and it does not. Default confirmed; flag retained for the record.

**Still open:** the remaining gap in `auc`/`aupr` is not explained by the diagonal. Next test is
`standardize=True` (Varando's `cor(data)`), then a larger $p$ sweep — the published curves are
plotted against $p$ up to 50, and our $p \le 20$ slice may simply be sampling the low-$p$ end.

### 8.6 Resolved: Dettling standardizes. Figure 5 then reproduces across $10 \le p \le 50$

The §8.4 gap was the `standardize` flag. `C_ID`, all seven $p$, $k=1,\dots,4$, 5 replicates,
identical in every other respect — the only difference is $\hat\Sigma$ vs. the empirical
correlation matrix:

| $p$ | `max_acc` | `max_f1` | `auc` | `aupr` | | `max_acc` | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|---|---|---|---|---|
| | *covariance* | | | | | **correlation** | | | |
| 10 | 0.806 | 0.576 | 0.699 | 0.390 | | 0.812 | 0.619 | 0.716 | 0.386 |
| 15 | 0.849 | 0.495 | 0.718 | 0.328 | | 0.854 | 0.567 | 0.759 | 0.379 |
| 20 | 0.889 | 0.446 | 0.750 | 0.297 | | 0.894 | 0.574 | 0.811 | 0.382 |
| 25 | 0.905 | 0.409 | 0.747 | 0.265 | | 0.908 | 0.540 | 0.810 | 0.366 |
| 30 | 0.919 | 0.382 | 0.749 | 0.227 | | 0.926 | 0.545 | 0.814 | 0.381 |
| 40 | 0.938 | 0.355 | 0.761 | 0.227 | | 0.941 | 0.528 | 0.835 | 0.369 |
| 50 | 0.952 | 0.328 | 0.774 | 0.195 | | 0.955 | 0.512 | 0.851 | 0.370 |

Against Figure 5's axis ranges (`max_acc` 0.80–0.95, `max_f1` 0.45–0.60, `auc` 0.70–0.85,
`aupr` 0.30–0.45):

- **Standardized:** all four metrics lie inside their ranges at every $p$ from 10 to 50. `auc`
  runs 0.716 → 0.851, essentially spanning the published axis end to end; `max_f1` declines
  0.619 → 0.512 within the 0.45–0.60 band; `aupr` is flat at 0.37–0.39 inside 0.30–0.45;
  `max_acc` rises 0.812 → 0.955 along 0.80–0.95.
- **Unstandardized:** `max_f1` falls to 0.328 and `aupr` to 0.195 by $p=50$, far below the axis
  floors of 0.45 and 0.30, and `auc` never approaches the 0.85 top.

An axis has to contain its curves, so the unstandardized configuration is ruled out and the
standardized one is not. **`S1Config.standardize` now defaults to `True`**; `run_s1.py
--no-standardize` opts out.

This also makes sense of the setup: Dettling writes only that he uses "a similar setting as in
Varando and Hansen (2020)", and Varando's `simulate.R` passes `cor(exper$data)` to every method.
Standardizing leaves the true support untouched (§4.1), so nothing about the estimand changes —
only the conditioning of $\hat\Sigma$, and with it the whole regularization path.

Raw output: `results/s1_allp_CID_reps5_raw.csv`, `results/s1_allp_CID_reps5_standardized.csv`.

### 8.7 The $C$-choice ordering survives standardization

`run_s1.py --p 10 15 20 25 30 40 50 --reps 10`, all four $C$ choices, standardized —
1,120 datasets, 40.6 min on 8 cores. Averaged over $k=1,\dots,4$:

| metric | $p$ | `C_ID` | `C_Random_Min_Diag` | `C_Random_Diag` | `C_Random_Full` |
|---|---|---|---|---|---|
| `max_acc` | 10 | 0.819 | 0.817 | 0.801 | 0.774 |
| | 30 | 0.924 | 0.921 | 0.922 | 0.920 |
| | 50 | 0.954 | 0.953 | 0.951 | 0.950 |
| `max_f1` | 10 | 0.622 | 0.618 | 0.583 | 0.496 |
| | 30 | 0.537 | 0.531 | 0.529 | 0.418 |
| | 50 | 0.509 | 0.510 | 0.482 | 0.386 |
| `auc` | 10 | 0.725 | 0.739 | 0.704 | 0.625 |
| | 30 | 0.814 | 0.814 | 0.809 | 0.777 |
| | 50 | 0.846 | 0.838 | 0.834 | 0.805 |
| `aupr` | 10 | 0.407 | 0.404 | 0.370 | 0.273 |
| | 30 | 0.371 | 0.362 | 0.350 | 0.267 |
| | 50 | 0.359 | 0.352 | 0.321 | 0.252 |

Every structural feature of Figure 5 is present simultaneously: the ordering
`C_ID ≈ C_Random_Min_Diag > C_Random_Diag > C_Random_Full` holds at every $p$ on all four metrics;
`max_acc` and `auc` rise with $p$ while `max_f1` and `aupr` fall; and `C_Random_Full` is separated
from the other three by a clear margin, widening on `max_f1` and `aupr`.

**One blemish.** `C_Random_Full`'s `aupr` (0.252–0.273) sits just below Figure 5's 0.30 axis floor,
and its `max_f1` (0.386–0.496) crosses the 0.45 floor for $p \ge 20$. The other three curves stay
inside on every metric. So the worst-case curve is not fully accounted for — plausibly the residual
of some remaining convention difference in how $C$ is drawn for choice 4, or simply 10 replicates
rather than 100. Worth resolving before the figure is treated as fully reproduced.

Raw output: `results/s1_allC_allp_reps10_standardized.csv`.

---

## 9. Status and what is left

Run outputs backing §8 are committed under [`results/`](results/).

### 9.1 Done and verified

| | evidence |
|---|---|
| Model, DGP, Lyapunov solve, both solvers, metrics implemented | `src/gclm/`, 77 passing tests |
| Design matrix identical to Varando's R construction | `test_design_matrix_matches_r` |
| All six metrics identical to `evaluatePathB`/`AUROC`/`AUCPR` | `test_all_metrics_match_r`, 1e-12 |
| Solvers hit the exact KKT solution, not just a low objective | `test_solvers_match_exact_restricted_solution`, 1e-9 |
| Agreement with `glmnet` on objective and support along a path | `test_lasso_path_matches_glmnet` |
| `glmnet` λ-scaling $p^3/(p-1)$ recovered empirically | `test_glmnet_lambda_scaling` |
| **M0 = Figure 3 reproduced**, incl. the irrepresentability failure at $n=\infty$ | §8.1, `results/m0_reps5.csv` |
| $\Gamma(\Sigma)$ singular with null space $p(p-1)/2$ | §2.3, `test_design_matrix_is_rank_deficient` |
| Production tolerance chosen by measurement | §8.3, `results/tolerance_benchmark.txt` |
| `metrics_include_diagonal = False` settled | §8.5, `results/diagonal_flag.txt` |
| Figure 5's **ordering** of the four $C$ choices reproduced, standardized, $10 \le p \le 50$ | §8.7, `results/s1_allC_allp_reps10_standardized.csv` |
| `standardize = True` settled; **Figure 5's four metric ranges reproduced for $10 \le p \le 50$** | §8.6, `results/s1_allp_CID_reps5_*.csv` |

### 9.2 Open — in priority order

1. **Run M1 in full.** Everything needed is settled; what remains is compute. Current coverage is
   `C_ID` at 5 replicates across all seven $p$ (§8.6) plus all four $C$ choices at 10 replicates
   (§8.4, $p \le 20$). The paper uses 100 replicates and all four $C$ choices at all seven $p$:
   11,200 datasets, ~7 h on 8 cores — measured from the 1,120-dataset run in §8.7, which
   spans the same $p$ range and $C$ choices. Nothing blocks this on cluster access.
2. **`C_Random_Full` sits below two axis floors** (§8.7): `aupr` 0.25–0.27 against a floor of 0.30,
   and `max_f1` below 0.45 for $p \ge 20$. The other three $C$ curves are inside on every metric.
   Re-run choice 4 at 100 replicates first; if the gap persists, re-read §3.2 against the paper.
3. **`penalize_diagonal`** is still inferred from Varando's `penalty.factor = 1 - diag(p)`, not
   confirmed against Dettling. Cheap to test the same way as §8.5 and §8.6.
4. **`metzler`** (§3.1): the papers' text and Varando's code disagree on whether off-diagonal
   entries are signed. Defaulting to the text; the effect on the metrics is unmeasured.
5. **Figure 5 is matched at the level of axis ranges, not curve values.** Only tick ranges are
   recoverable from the paper text. §8.6 is strong evidence (the unstandardized run falls outside
   the axes, the standardized one inside), but an exact per-curve check needs the figure's data.
   Worth asking Dettling at the Oct 10–11 meeting.

### 9.3 Not started

- **M2 / S1b** — MCP and SCAD. The harness is ready: only the proximal operator changes
  (`gclm.lasso._soft_threshold`), and §2.3 already identifies RSC as the condition to argue under.
  Add `ncvreg` as the R-side reference (§7.2), mirroring `glmnet`'s role in S1. Use our solvers,
  not `ncvreg`/`glmnet`, to produce the numbers in any MSE comparison (§8.2).
- **S2** (likelihood loss) and **S3** (score-based search with BIC) — see `plan.md`. The CRAN
  package `gclm` is the reference for S2, not for S1.

---

## References

- Dettling, Drton, Kolar (2024). *On the Lasso for Graphical Continuous Lyapunov Models.* PMLR 236.
- Varando, Hansen (2020). *Graphical Continuous Lyapunov Models.* PMLR 124.

**Algorithms used by the solvers:**

- Beck, Teboulle (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
  Problems.* SIAM J. Imaging Sci. 2(1):183–202. — FISTA; `solve_fista`.
- O'Donoghue, Candès (2015). *Adaptive Restart for Accelerated Gradient Schemes.* Found. Comput.
  Math. 15:715–732. — the restart condition at `lasso.py`'s FISTA loop.
- Nesterov (1983). *A method of solving a convex programming problem with convergence rate
  $O(1/k^2)$.* Soviet Math. Dokl. 27:372–376. — the underlying acceleration.
- Parikh, Boyd (2014). *Proximal Algorithms.* Found. Trends Optim. 1(3):127–239. — also cited by
  Varando for Algorithm 1.
- Breheny, Huang (2011). *Coordinate descent algorithms for nonconvex penalized regression.* Ann.
  Appl. Stat. 5(1):232–253. — `ncvreg`.
- Friedman, Hastie, Tibshirani (2010). *Regularization Paths for Generalized Linear Models via
  Coordinate Descent.* J. Stat. Softw. 33(1). — `glmnet`; cited by Dettling in Appendix A.
- Loh, Wainwright (2015). *Regularized M-estimators with nonconvexity.* JMLR 16:559–616. — the
  RSC framework and composite gradient descent for S1b (already in `plan.md`).
- Code: [`gherardovarando/gclm_experiments`](https://github.com/gherardovarando/gclm_experiments)
  (`functions/util.R::lassoB`, `evaluatePathB`; `simulate.R`), R package
  [`gclm`](https://cran.r-project.org/package=gclm).

> Note: the CRAN package `gclm` fits the *likelihood* / *Frobenius-to-$\hat\Sigma$* losses
> (`gclm()` minimizes $L(\Sigma(B,C)) + \lambda\rho(B) + \lambda_C\|C-C_0\|_F^2$), **not** the
> Direct Lyapunov Lasso. It is the reference for **S2**, not for S1. For S1 the reference is
> `lassoB()` + `glmnet`.
