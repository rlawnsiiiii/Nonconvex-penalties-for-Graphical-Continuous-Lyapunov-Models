# MCP and SCAD for the Direct Lyapunov estimator

How the nonconvex penalties of study S1b are defined, implemented and validated. Code:
`src/gclm/objective/penalties.py` (the penalties), `src/gclm/solvers/proxgrad.py` (`_solve_mapg`),
`src/gclm/solvers/backends.py` (the package backends), `src/gclm/solvers/path.py` (the path),
tests in `tests/test_nonconvex.py` (83 tests).

```bash
python simulations/run_s1.py --penalty MCP                                  # fista, textbook
python simulations/run_s1.py --penalty SCAD --gamma 3.7
python simulations/run_s1.py --penalty MCP --solver skglm                   # package, textbook
python simulations/run_s1.py --penalty MCP --solver ncvreg --convention ncvreg
```

---

## 1. The estimator

Dettling's eq. (1.4) with the ℓ1 penalty replaced by a nonconvex one:

$$\hat M(\lambda)=\arg\min_{M}\ \tfrac12\big\|M\hat\Sigma+\hat\Sigma M^\top+C\big\|_F^2
\;+\;\sum_{i\neq j} P_{\lambda,\gamma}(M_{ij}) .$$

The diagonal stays unpenalized (weights $w_{ij}=0$), and the loss keeps Dettling's scale: factor
$\tfrac12$, **no $1/n$**. The penalties, with $x\ge0$ and extended symmetrically:

| | $P_{\lambda,\gamma}(x)$ | $P'(x)$ | default $\gamma$ |
|---|---|---|---|
| **lasso** | $\lambda x$ | $\lambda$ | — |
| **MCP** (Zhang 2010) | $\lambda x - \dfrac{x^2}{2\gamma}$ for $x\le\gamma\lambda$; $\ \dfrac{\gamma\lambda^2}{2}$ beyond | $(\lambda - x/\gamma)_+$ | 3 |
| **SCAD** (Fan & Li 2001) | $\lambda x$ for $x\le\lambda$; $\ \dfrac{2\gamma\lambda x - x^2-\lambda^2}{2(\gamma-1)}$ up to $\gamma\lambda$; $\ \dfrac{\lambda^2(\gamma+1)}{2}$ beyond | $\lambda$; $\ \dfrac{\gamma\lambda-x}{\gamma-1}$; $\ 0$ | 3.7 |

Both behave like the lasso near zero (slope $\lambda$ at $0^+$, so they still select), then taper
off and become **constant beyond $\gamma\lambda$**, so large coefficients are not shrunk at all.
That is the property S1b is about. $\gamma$ is restricted to $\gamma>1$ (MCP) and $\gamma>2$
(SCAD), the same domain ncvreg enforces.

The parameterization is the one ncvreg, skglm and pyproximal all use, verified on their
implementations (§6).

---

## 2. Two conventions — and why this matters for ncvreg

This section records a finding that decides which packages can be used for S1b.

### 2.1 What ncvreg actually minimizes

`ncvreg::ncvfit` does **not** minimize the textbook MCP/SCAD objective on a design whose columns
are not standardized. On a plain regression, independent of any Lyapunov code
(`test_what_ncvfit_actually_minimises`), its solution is a stationary point of

$$\frac{1}{2n}\|y-X\beta\|^2 \;+\; \sum_j \frac{P_{\lambda,\gamma}(v_j\beta_j)}{v_j},
\qquad v_j=\frac{x_j^\top x_j}{n},$$

to $10^{-16}$. It is **not** a stationary point of the textbook objective
$\frac1{2n}\|y-X\beta\|^2+\sum_j P_{\lambda,\gamma}(\beta_j)$ (violations 0.2–0.8) once
coefficients reach the penalty's curved region. For the lasso the two coincide, because
$\lambda|v\beta|/v=\lambda|\beta|$, which is why ncvreg's lasso agreed with every other backend to
$10^{-12}$. For MCP/SCAD they coincide only when every $v_j=1$.

This is not a bug. It is Breheny & Huang's standardized-covariate convention, in which γ measures
the penalty's concavity **relative to each coordinate's loss curvature**, so the same γ gives the
same convexity margin in every coordinate. But it is a different estimator. For MCP it amounts to
a per-coordinate $\gamma_j=\gamma/v_j$.

### 2.2 Why it is large here

In our problem the "columns" are the columns of $A(\hat\Sigma)$, one per entry of $M$, with
squared norms (closed form, `lyap.design_column_sq_norms`)

$$v_{ik}=\|A(\hat\Sigma)e_{ik}\|^2 = 2\big(\|\hat\Sigma_{k,\cdot}\|^2+\hat\Sigma_{ik}^2\big).$$

These vary with the entry, so the two conventions give genuinely different estimates. On a $p=6$
problem, ncvreg's MCP/SCAD solutions violate the textbook stationarity condition by 0.28–1.3, and
satisfy the ncvreg-convention condition to $10^{-11}$.

### 2.3 The two conventions in this code

`convention=` on `lasso_path` / `solve_fista` / `run_s1.py`:

| | penalty on entry $ij$ | solved by | use when |
|---|---|---|---|
| **`"textbook"`** *(default)* | $P_{\lambda,\gamma}(M_{ij})$ | `fista`, `skglm` | the estimator as written in Fan & Li, Zhang and Loh & Wainwright: a fixed concavity $\mu=1/\gamma$ (MCP) or $1/(\gamma-1)$ (SCAD), as the RSC theory is stated |
| **`"ncvreg"`** | $P_{\lambda,\gamma}(v_{ij}M_{ij})/v_{ij}$ | `fista`, `ncvreg` | you want ncvreg's scale-free γ, or ncvreg as the solver |

The backends **refuse** a convention they do not solve (`test_backends_refuse_a_convention_they_do_not_solve`):
asking `solver="ncvreg"` for the textbook convention raises instead of silently returning a
different estimator. `fista` does either, so every configuration is available at large $p$.

**The choice between them is a modeling decision for the thesis, not an implementation detail.**
The default is `"textbook"` because it is the estimator the nonconvex-penalty literature (and
`plan.md`'s references) analyses. `"ncvreg"` is there because the advisor's suggestion was to use
ncvreg, and it gives ncvreg a verified role rather than a silently different one.

### 2.4 Why the package calls use a scaled design

Both ncvfit and skglm's `Quadratic` datafit use $\frac{1}{2n}\|y-X\beta\|^2$ with
$n=\mathrm{nrow}(X)=p^2$. Passing $\sqrt n\,X$ and $\sqrt n\,y$ turns that into exactly
$\tfrac12\|y-X\beta\|^2$, so **λ and γ go through unchanged** for every penalty.

The alternative, an unscaled design with converted parameters, cannot work for SCAD under the
textbook convention. $n\cdot\mathrm{SCAD}_{\lambda',\gamma'}$ is not a SCAD penalty for any
$(\lambda,\gamma)$, because SCAD's kinks sit at $\lambda$ itself: the best candidate differs by up
to 0.48. MCP survives the rescaling, but ncvreg's `gamma=3` would become $\gamma=3/p^2$, outside the
valid range. (The ncvreg convention is itself invariant to column scaling, so for ncvreg the two
encodings give identical results. The scaled one is used everywhere for uniformity.)

---

## 3. Proximal operators

The default solver needs $\mathrm{prox}_{tP}(z)=\arg\min_x \tfrac12(x-z)^2+tP(x)$, elementwise.
Setting the derivative to zero on each piece gives, for $z>0$:

**MCP** (firm thresholding; needs $\gamma>t$)
$$x^\star=\begin{cases}0 & z\le t\lambda\\[2pt] \dfrac{z-t\lambda}{1-t/\gamma} & t\lambda<z\le\gamma\lambda\\[6pt] z & z>\gamma\lambda\end{cases}$$

**SCAD** (needs $\gamma>1+t$)
$$x^\star=\begin{cases}(z-t\lambda)_+ & z\le(1+t)\lambda\\[2pt] \dfrac{(\gamma-1)z-t\gamma\lambda}{\gamma-1-t} & (1+t)\lambda<z\le\gamma\lambda\\[6pt] z & z>\gamma\lambda\end{cases}$$

Under the stated conditions each scalar problem is strictly convex, so the closed form is its
**unique global minimizer**. `prox` raises otherwise. Both pieces are continuous at the knots,
and the last row, $x^\star=z$, is the "no shrinkage" property.

For the ncvreg convention, substituting $u=vx$ reduces $\min_x\tfrac12(x-z)^2+\tfrac tv P(vx)$ to
the textbook problem with step $tv$ at $vz$, so $x^\star=\mathrm{prox}_{tvP}(vz)/v$.

**Why not call a package prox here?** Measured on $p^2=2500$ entries:

| | one prox call |
|---|---|
| skglm `prox_1d` (scalar, looped from Python) | 0.62 ms |
| pyproximal `SCAD.prox` (vectorized) | 0.052 ms |
| our vectorized closed form | 0.030 ms |

A path at $p=50$ needs $10^5$–$10^6$ prox calls. skglm's scalar interface would cost minutes per
dataset, and pyproximal has no MCP and takes a single scalar threshold (no per-entry weights). So
the closed forms are ours, and **all three references check them**: brute-force minimization
(knows nothing of the formulas), skglm's `prox_1d` to $10^{-12}$, and pyproximal's SCAD to
$10^{-12}$, at several step sizes.

---

## 4. The algorithm: monotone accelerated proximal gradient

For the lasso, `solve_fista` is unchanged (FISTA with adaptive restart; the validated default,
bit-identical: `test_default_path_is_unchanged_for_the_lasso`).

For MCP/SCAD plain FISTA has **no convergence guarantee**: with a nonconvex penalty the momentum
can increase the objective indefinitely. `_solve_mapg` implements Li & Lin (2015), *Accelerated
Proximal Gradient Methods for Nonconvex Programming*, NeurIPS, **Algorithm 1**:

$$\begin{aligned}
y_k &= x_k + \tfrac{t_{k-1}}{t_k}(z_k-x_k) + \tfrac{t_{k-1}-1}{t_k}(x_k-x_{k-1})\\
z_{k+1} &= \mathrm{prox}_{aP}\big(y_k-a\nabla f(y_k)\big)\\
v_{k+1} &= \mathrm{prox}_{aP}\big(x_k-a\nabla f(x_k)\big)\\
t_{k+1} &= \big(\sqrt{4t_k^2+1}+1\big)/2\\
x_{k+1} &= z_{k+1}\ \text{if}\ F(z_{k+1})\le F(v_{k+1}),\ \text{else}\ v_{k+1}
\end{aligned}$$

Comparing against the plain step $v$ makes the objective **monotone**, and gives convergence to a
**critical point** for nonconvex penalties with the Kurdyka–Łojasiewicz property. MCP and SCAD are
semi-algebraic, so they qualify (Li & Lin, Theorem 1). The step is $a=0.99/L$ with
$L=4\lambda_{\max}(\hat\Sigma)^2$, since the theorem needs $a<1/L$. With the standardized
$\hat\Sigma$ used throughout, every $v_{ij}\ge2\hat\Sigma_{kk}^2=2$, so $L\ge\max_{ij}v_{ij}\ge2$ and
$a<\tfrac12$. The prox conditions ($\gamma>a$, resp. $\gamma>1+a$, with $av_{ij}\le0.99$ in place
of $a$ under the ncvreg convention) then hold for any valid γ. `prox` checks them regardless and
raises if they fail.

Each iteration takes two proximal-gradient steps and two objective evaluations. Residuals
$R=M\hat\Sigma+\hat\Sigma M^\top+C$ are cached so each candidate costs one residual, reused for both
its objective value and the next gradient. Stopping rule: $\|x_{k+1}-x_k\|_\infty<\texttt{tol}$.

### Path, λ_max and warm starts

All three penalties have $P'(0^+)=\lambda$, so the diagonal fit is stationary exactly when it is
for the lasso: **λ_max and the grid are unchanged**, as is the short-circuit that returns the
diagonal fit for $\lambda\ge\lambda_{\max}$ (`test_lambda_max_is_the_same_as_for_the_lasso`). The
path is warm-started from the sparse end, which matters more here than for the lasso. With a
nonconvex penalty the starting point decides which local minimum is found; ncvfit's help page
says the same.

---

## 5. What to expect from nonconvex penalties

### 5.1 They remove the lasso's shrinkage bias

> **What the simulations found (S1b, simulations/S2_penalties_losses.md §3.1–3.6):** they do
> remove it, and they find the *undirected skeleton* as well as the lasso — but they orient
> edges worse, and three quarters of their loss on the directed metrics is orientation. The
> design $A(\hat\Sigma)$ has a $p(p-1)/2$-dimensional null space (for a correlation matrix,
> nearly the skew-symmetric matrices), so $M_{ij}$ and $M_{ji}$ are almost interchangeable. The
> $\ell_1$ norm is constant along that swap, so the lasso lets the loss decide and keeps both
> directions when it is unsure; a penalty with a kink at zero and a flat part charges $\lambda$
> on the entering direction and refunds nothing on the exiting one, which freezes whichever
> direction entered first. Read that section before expecting the bias removal below to
> translate into better graphs.


On Dettling's Example 2 path graph with the population covariance, at every λ where both methods
recover the support exactly (`test_nonconvex_penalties_remove_the_lasso_bias`):

| λ | lasso: max \|M̂ − M*\| on the true edges | MCP | SCAD |
|---|---|---|---|
| 0.0365 | 0.448 | 1.2e-12 | 1.1e-12 |
| 0.0410 | 0.503 | 1.2e-12 | 1.2e-12 |
| 0.0461 | 0.566 | 1.3e-12 | 1.7e-11 |

The true edge weights are 0.65, so the lasso shrinks them almost to zero while MCP and SCAD return
them exactly. This is the effect S1b sets out to measure at scale.

### 5.2 Different solvers can reach different local minima

This is inherent to nonconvex penalties, and it shows up between backends:

| | vs | coefficients | same support | both stationary |
|---|---|---|---|---|
| SCAD, textbook (diag. penalized) | skglm | **1.7e-9** | 15/15 | yes |
| SCAD, ncvreg convention | ncvreg | 2.9e-3 | 14/15 | yes |
| MCP, textbook | skglm | 0.33 | 14/15 | yes |
| MCP, ncvreg convention | ncvreg | 0.11 | 9/15 | yes |

Where coefficients differ, both points are genuine critical points of the same objective. Started
at the package's solution, our solver moves by less than $10^{-11}$, so each solution is a fixed
point of the other's iteration. Neither is systematically lower in objective (ours lower at 2
λ, the package's at 5, equal at 23). The tests therefore require stationarity and the fixed-point
property, and coefficient agreement only where the solvers do share a path (SCAD vs skglm).

For the thesis, reported MCP/SCAD results are properties of **estimator + path strategy**. Loh &
Wainwright (2015) is the relevant theory: under restricted strong convexity every stationary point
in a neighbourhood of the truth is statistically close to it, which is why the choice among such
local minima need not matter asymptotically.

### 5.3 A SCAD-specific edge case

At the sparse edge of the exact-recovery window (λ ≈ 0.17·λ_max in Example 2), SCAD's warm-started
path can settle in a local minimum with one weak edge still inside its lasso-like region
$|x|\le\lambda$, and so stays biased there. MCP did not do this, since its slope starts decreasing
immediately at 0. It happened at one λ out of fifteen, and the test allows exactly that.

---

## 6. Validation (`tests/test_nonconvex.py`)

| group | against | result |
|---|---|---|
| penalty values | textbook formulas, skglm, pyproximal | exact ($10^{-14}$) |
| penalty shape | $P(0)=0$, slope λ at $0^+$, continuity at knots, flat beyond $\gamma\lambda$, $\gamma\to\infty$ gives the lasso | exact |
| derivatives | central finite differences, both conventions | $10^{-5}$ |
| proximal operators | **brute-force minimization**, skglm `prox_1d`, pyproximal SCAD | $2\cdot10^{-6}$ (grid), $10^{-12}$, $10^{-12}$ |
| scaled proximal operators | brute-force minimization | $2\cdot10^{-6}$ |
| what ncvfit minimizes | a generic regression, no Lyapunov code | ncvreg convention stationary to $10^{-10}$; textbook violated by $>10^{-2}$ |
| solver | stationarity (both conventions, two λ), monotone objective, $\gamma\to\infty\Rightarrow$ lasso solution | $10^{-8}$; monotone; $10^{-6}$ |
| backends | skglm (textbook MCP, SCAD), ncvreg (ncvreg MCP, SCAD) | stationarity + fixed point, $10^{-8}$ |
| guards | wrong convention, ℓ1-only backends, unknown convention, fractional weights | raise |
| motivation | Example 2: bias removal | $<10^{-6}$ vs lasso $>0.05$ |

---

## 7. Cost

100-λ path, production tolerance `tol=1e-8`, one core (Apple M2):

| $p$ | lasso | MCP | SCAD |
|---|---|---|---|
| 10 | 0.96 s | 4.5 s | 6.2 s |
| 20 | 3.1 s | 15.1 s | 21.3 s |

MCP costs about 5× the lasso and SCAD about 6.5×: the monotone scheme takes two
proximal-gradient steps per iteration, and nonconvex paths need more iterations. Scaling the full
Figure 5 grid (38.6 CPU-h for the lasso) gives **about 190 CPU-h for MCP and 250 for SCAD**. That
is roughly 3–4 hours on the 64-task array in `cluster/s1_array.sbatch` with `--penalty` added,
and not a laptop job.

If this becomes the bottleneck, Li & Lin's Algorithm 2 (non-monotone APG) skips the second step
when the extrapolated point already decreases the objective enough. It was not needed to establish
correctness, so it is not implemented.

---

## 8. Backend summary

| backend | lasso | MCP | SCAD | convention for MCP/SCAD |
|---|---|---|---|---|
| **`fista`** (default) | ✓ | ✓ | ✓ | textbook or ncvreg |
| **`ncvreg`** | ✓ | ✓ | ✓ | **ncvreg only** |
| **`skglm`** | ✓ | ✓ | only with the diagonal penalized (no weighted SCAD) | **textbook only** |
| `glmnet` | ✓ | — | — | — |
| `pyproximal` | ✓ | — | — | — |
| `design` | ✓ | — | — | — |

All of these minimise the **direct** loss. For Varando's losses on the implied covariance
(`--loss loglik|frobenius`) the same three penalties are solved by the active-set Newton method of
`src/gclm/solvers/covariance.py` under the textbook convention only — there is no design matrix, so the
ncvreg convention does not apply. See docs/LIKELIHOOD.md.

---

## References

- Fan, J. & Li, R. (2001). *Variable selection via nonconcave penalized likelihood and its oracle
  properties.* JASA 96(456), 1348–1360. — SCAD.
- Zhang, C.-H. (2010). *Nearly unbiased variable selection under minimax concave penalty.* Annals
  of Statistics 38(2), 894–942. — MCP.
- Breheny, P. & Huang, J. (2011). *Coordinate descent algorithms for nonconvex penalized
  regression, with applications to biological feature selection.* Annals of Applied Statistics
  5(1), 232–253. — ncvreg, and its standardized-covariate convention.
- Li, H. & Lin, Z. (2015). *Accelerated proximal gradient methods for nonconvex programming.*
  Advances in Neural Information Processing Systems 28. — Algorithm 1, used here.
- Loh, P.-L. & Wainwright, M. J. (2015). *Regularized M-estimators with nonconvexity: statistical
  and algorithmic theory for local optima.* JMLR 16, 559–616. — RSC and local optima.
- Loh, P.-L. & Wainwright, M. J. (2017). *Support recovery without incoherence: a case for
  nonconvex regularization.* Annals of Statistics 45(6), 2455–2482.
- Beck, A. & Teboulle, M. (2009). *A fast iterative shrinkage-thresholding algorithm for linear
  inverse problems.* SIAM J. Imaging Sciences 2(1), 183–202. — the lasso path (docs/FISTA.md).
- skglm, JMLR 26 (2025), paper 24-0008; pyproximal (PyLops project). — reference implementations.
