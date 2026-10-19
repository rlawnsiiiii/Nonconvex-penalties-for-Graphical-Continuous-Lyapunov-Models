# MCP and SCAD for the Direct Lyapunov estimator

How the nonconvex penalties of study S1b are defined, implemented and validated.

```
simulations/run_s1.py   --penalty {MCP,SCAD}   --solver {fista,skglm,ncvreg}   --convention {textbook,ncvreg}   --gamma
        |
        v
src/gclm/solvers/path.py            lasso_path(solver=, **solver_kwargs)          lambda_max, grid, warm starts   Section 4
        |
        |--> src/gclm/solvers/proxgrad.py    solve_fista(penalty=, gamma=, convention=) -> _solve_mapg              Section 4
        |           |--> src/gclm/objective/penalties.py   P, P', prox, both conventions                            Sections 1-3
        |           '--> src/gclm/lyapunov.py              design_column_sq_norms -> v_ik                           Section 2.2
        |
        |--> src/gclm/solvers/backends.py    skglm | ncvreg | glmnet | pyproximal | design,  scaled design sqrt(n) X  Sections 2.4, 8
        '--> src/gclm/solvers/covariance.py  --loss loglik|frobenius, active-set Newton, textbook only               Section 8

tests/test_nonconvex.py             83 tests                                                                       Section 6
```

```bash
python simulations/run_s1.py --penalty MCP                                  # fista, textbook
python simulations/run_s1.py --penalty SCAD --gamma 3.7
python simulations/run_s1.py --penalty MCP --solver skglm                   # package, textbook
python simulations/run_s1.py --penalty MCP --solver ncvreg --convention ncvreg
```

---

## 1. The estimator

Dettling's eq. 1.4 with the $\ell_1$ penalty replaced by a nonconvex one:

$$\hat M(\lambda)=\arg\min_{M}\ \tfrac12\big\|M\hat\Sigma+\hat\Sigma M^\top+C\big\|_F^2\;+\;\sum_{i\neq j} P_{\lambda,\gamma}(M_{ij}) .$$

| convention | rule |
|---|---|
| diagonal | unpenalized, weights $w_{ij}=0$ |
| loss scale | Dettling's: factor $\tfrac12$, no $1/n$ |
| $\gamma$ domain | $\gamma>1$ for MCP, $\gamma>2$ for SCAD, the domain ncvreg enforces |
| parameterization | the one ncvreg, skglm and pyproximal use, verified on their implementations in Section 6 |
| near zero | slope $\lambda$ at $0^+$, so MCP and SCAD still select |
| beyond $\gamma\lambda$ | $P$ constant, so large coefficients are not shrunk at all; the property S1b is about |

Penalties, for $x\ge0$ and extended symmetrically:

| | $P_{\lambda,\gamma}(x)$ | $P'(x)$ | default $\gamma$ |
|---|---|---|---|
| **lasso** | $\lambda x$ | $\lambda$ | — |
| **MCP**, Zhang 2010 | $\lambda x - \dfrac{x^2}{2\gamma}$ for $x\le\gamma\lambda$; $\ \dfrac{\gamma\lambda^2}{2}$ beyond | $(\lambda - x/\gamma)_+$ | 3 |
| **SCAD**, Fan & Li 2001 | $\lambda x$ for $x\le\lambda$; $\ \dfrac{2\gamma\lambda x - x^2-\lambda^2}{2(\gamma-1)}$ up to $\gamma\lambda$; $\ \dfrac{\lambda^2(\gamma+1)}{2}$ beyond | $\lambda$; $\ \dfrac{\gamma\lambda-x}{\gamma-1}$; $\ 0$ | 3.7 |

---

## 2. Two conventions, and why this matters for ncvreg

### 2.1 What ncvreg actually minimizes

ncvreg objective:

$$\frac{1}{2n}\|y-X\beta\|^2 \;+\; \sum_j \frac{P_{\lambda,\gamma}(v_j\beta_j)}{v_j},\qquad v_j=\frac{x_j^\top x_j}{n}$$

textbook objective:

$$\frac{1}{2n}\|y-X\beta\|^2 \;+\; \sum_j P_{\lambda,\gamma}(\beta_j)$$

| `ncvreg::ncvfit` solution, plain regression, unstandardized design, no Lyapunov code, `test_what_ncvfit_actually_minimises` | stationary point? |
|---|---|
| of the ncvreg objective | yes, to $10^{-16}$ |
| of the textbook objective | no, violations 0.2 to 0.8 once coefficients reach the penalty's curved region |
| lasso | both objectives coincide, since $\lambda\lvert v\beta\rvert/v=\lambda\lvert\beta\rvert$; ncvreg's lasso agreed with every other backend to $10^{-12}$ |
| MCP, SCAD | coincide only when every $v_j=1$ |
| meaning | not a bug: Breheny & Huang's standardized-covariate convention, in which $\gamma$ measures the penalty's concavity relative to each coordinate's loss curvature, so the same $\gamma$ gives the same convexity margin in every coordinate; for MCP a per-coordinate $\gamma_j=\gamma/v_j$; a different estimator |

### 2.2 Why it is large here

The columns are those of $A(\hat\Sigma)$, one per entry of $M$; squared norms in closed form:

$$v_{ik}=\|A(\hat\Sigma)e_{ik}\|^2 = 2\big(\|\hat\Sigma_{k,\cdot}\|^2+\hat\Sigma_{ik}^2\big)$$

| | |
|---|---|
| code | `design_column_sq_norms` in `src/gclm/lyapunov.py` |
| $v_{ik}$ | varies with the entry, so the two conventions give genuinely different estimates |
| $p=6$, ncvreg's MCP/SCAD solutions | violate the textbook stationarity condition by 0.28 to 1.3; satisfy the ncvreg-convention condition to $10^{-11}$ |

### 2.3 The two conventions in this code

`convention=` on `lasso_path`, `solve_fista` and `run_s1.py`:

| `convention=` | penalty on entry $ij$ | solved by | use when |
|---|---|---|---|
| `"textbook"`, the default | $P_{\lambda,\gamma}(M_{ij})$ | `fista`, `skglm` | the estimator as written in Fan & Li, Zhang and Loh & Wainwright: fixed concavity $\mu=1/\gamma$ for MCP, $1/(\gamma-1)$ for SCAD, as the RSC theory is stated |
| `"ncvreg"` | $P_{\lambda,\gamma}(v_{ij}M_{ij})/v_{ij}$ | `fista`, `ncvreg` | you want ncvreg's scale-free $\gamma$, or ncvreg as the solver |

| rule | |
|---|---|
| guard | the backends refuse a convention they do not solve, `test_backends_refuse_a_convention_they_do_not_solve`; `solver="ncvreg"` with the textbook convention raises instead of silently returning a different estimator |
| `fista` | solves either, so every configuration is available at large $p$ |
| decision | the choice is a modeling decision for the thesis, not an implementation detail |
| why `"textbook"` is the default | the estimator the nonconvex-penalty literature and `plan.md`'s references analyse |
| why `"ncvreg"` exists | the advisor's suggestion was ncvreg; this gives ncvreg a verified role rather than a silently different one |

### 2.4 Why the package calls use a scaled design

| | |
|---|---|
| package datafit, ncvfit and skglm's `Quadratic` | $\frac{1}{2n}\lVert y-X\beta\rVert^2$ with $n=\mathrm{nrow}(X)=p^2$ |
| passed instead | $\sqrt n\,X$ and $\sqrt n\,y$, which gives exactly $\tfrac12\lVert y-X\beta\rVert^2$ |
| effect | $\lambda$ and $\gamma$ go through unchanged for every penalty |
| alternative, unscaled design with converted parameters | cannot work for SCAD under the textbook convention: $n\cdot\mathrm{SCAD}_{\lambda',\gamma'}$ is not a SCAD penalty for any $(\lambda,\gamma)$, because SCAD's kinks sit at $\lambda$ itself; the best candidate differs by up to 0.48 |
| MCP under the alternative | survives the rescaling, but ncvreg's `gamma=3` would become $\gamma=3/p^2$, outside the valid range |
| ncvreg convention | invariant to column scaling, so for ncvreg both encodings give identical results; the scaled one is used everywhere for uniformity |

---

## 3. Proximal operators

Elementwise, derivative set to zero on each piece, $z>0$:

$$\mathrm{prox}_{tP}(z)=\arg\min_x \tfrac12(x-z)^2+tP(x)$$

MCP, firm thresholding, needs $\gamma>t$:

$$x^\star=\begin{cases}0 & z\le t\lambda\\[2pt] \dfrac{z-t\lambda}{1-t/\gamma} & t\lambda<z\le\gamma\lambda\\[6pt] z & z>\gamma\lambda\end{cases}$$

SCAD, needs $\gamma>1+t$:

$$x^\star=\begin{cases}(z-t\lambda)_+ & z\le(1+t)\lambda\\[2pt] \dfrac{(\gamma-1)z-t\gamma\lambda}{\gamma-1-t} & (1+t)\lambda<z\le\gamma\lambda\\[6pt] z & z>\gamma\lambda\end{cases}$$

ncvreg convention, substitute $u=vx$ to reduce to the textbook problem with step $tv$ at $vz$:

$$\min_x\tfrac12(x-z)^2+\tfrac tv P(vx)\quad\Rightarrow\quad x^\star=\mathrm{prox}_{tvP}(vz)/v$$

| property | |
|---|---|
| uniqueness | under the stated conditions each scalar problem is strictly convex, so the closed form is its unique global minimizer; `prox` raises otherwise |
| knots | both pieces continuous |
| last row, $x^\star=z$ | the no-shrinkage property |

| one prox call, $p^2=2500$ entries | time |
|---|---|
| skglm `prox_1d`, scalar, looped from Python | 0.62 ms |
| pyproximal `SCAD.prox`, vectorized | 0.052 ms |
| our vectorized closed form | 0.030 ms |

| why the closed forms are ours | |
|---|---|
| a path at $p=50$ | $10^5$ to $10^6$ prox calls; skglm's scalar interface would cost minutes per dataset |
| pyproximal | no MCP; a single scalar threshold, no per-entry weights |
| three references check them | brute-force minimization, which knows nothing of the formulas; skglm `prox_1d` to $10^{-12}$; pyproximal's SCAD to $10^{-12}$; at several step sizes |

---

## 4. The algorithm: monotone accelerated proximal gradient

| penalty | solver |
|---|---|
| lasso | `solve_fista` unchanged: FISTA with adaptive restart, the validated default, bit-identical, `test_default_path_is_unchanged_for_the_lasso` |
| MCP, SCAD | `_solve_mapg`: Li & Lin 2015, Algorithm 1, below; plain FISTA has no convergence guarantee here, since with a nonconvex penalty the momentum can increase the objective indefinitely |

$$y_k = x_k + \tfrac{t_{k-1}}{t_k}(z_k-x_k) + \tfrac{t_{k-1}-1}{t_k}(x_k-x_{k-1})$$

$$z_{k+1} = \mathrm{prox}_{aP}\big(y_k-a\nabla f(y_k)\big)$$

$$v_{k+1} = \mathrm{prox}_{aP}\big(x_k-a\nabla f(x_k)\big)$$

$$t_{k+1} = \big(\sqrt{4t_k^2+1}+1\big)/2$$

$$x_{k+1} = z_{k+1}\ \text{if}\ F(z_{k+1})\le F(v_{k+1}),\ \text{else}\ v_{k+1}$$

| | |
|---|---|
| monotone | comparing against the plain step $v$ makes the objective monotone |
| convergence | to a critical point for nonconvex penalties with the Kurdyka–Łojasiewicz property; MCP and SCAD are semi-algebraic, so they qualify, Li & Lin Theorem 1 |
| step | $a=0.99/L$ with $L=4\lambda_{\max}(\hat\Sigma)^2$, since the theorem needs $a<1/L$ |
| standardized $\hat\Sigma$ | every $v_{ij}\ge2\hat\Sigma_{kk}^2=2$, so $L\ge\max_{ij}v_{ij}\ge2$ and $a<\tfrac12$ |
| prox conditions | $\gamma>a$ for MCP, $\gamma>1+a$ for SCAD, with $av_{ij}\le0.99$ in place of $a$ under the ncvreg convention; they hold for any valid $\gamma$; `prox` checks them regardless and raises if they fail |
| per iteration | two proximal-gradient steps and two objective evaluations |
| cache | residuals $R=M\hat\Sigma+\hat\Sigma M^\top+C$; each candidate costs one residual, reused for its objective value and the next gradient |
| stopping rule | $\lVert x_{k+1}-x_k\rVert_\infty<\texttt{tol}$ |

### Path, λ_max and warm starts

| | |
|---|---|
| $P'(0^+)=\lambda$ for all three penalties | the diagonal fit is stationary exactly when it is for the lasso, so $\lambda_{\max}$ and the grid are unchanged, `test_lambda_max_is_the_same_as_for_the_lasso` |
| $\lambda\ge\lambda_{\max}$ | the short-circuit returning the diagonal fit is unchanged |
| warm starts | from the sparse end; with a nonconvex penalty the starting point decides which local minimum is found, so this matters more than for the lasso; ncvfit's help page says the same |

---

## 5. What to expect from nonconvex penalties

### 5.1 They remove the lasso's shrinkage bias

| S1b finding, `simulations/S2_penalties_losses.md` §3.1–3.6 | read it before expecting the bias removal below to translate into better graphs |
|---|---|
| shrinkage bias | removed |
| undirected skeleton | found as well as the lasso finds it |
| edge orientation | worse; three quarters of their loss on the directed metrics is orientation |
| cause | $A(\hat\Sigma)$ has a $p(p-1)/2$-dimensional null space, for a correlation matrix nearly the skew-symmetric matrices, so $M_{ij}$ and $M_{ji}$ are almost interchangeable |
| lasso | the $\ell_1$ norm is constant along that swap, so the loss decides and both directions are kept when it is unsure |
| MCP, SCAD | a kink at zero plus a flat part charges $\lambda$ on the entering direction and refunds nothing on the exiting one, which freezes whichever direction entered first |

Dettling's Example 2 path graph, population covariance, every $\lambda$ where both methods recover the support exactly, `test_nonconvex_penalties_remove_the_lasso_bias`; the true edge weights are 0.65, which the lasso shrinks almost to zero while MCP and SCAD return them exactly:

| λ | lasso: max \|M̂ − M*\| on the true edges | MCP | SCAD |
|---|---|---|---|
| 0.0365 | 0.448 | 1.2e-12 | 1.1e-12 |
| 0.0410 | 0.503 | 1.2e-12 | 1.2e-12 |
| 0.0461 | 0.566 | 1.3e-12 | 1.7e-11 |

### 5.2 Different solvers can reach different local minima

| | vs | coefficients | same support | both stationary |
|---|---|---|---|---|
| SCAD, textbook, diagonal penalized | skglm | **1.7e-9** | 15/15 | yes |
| SCAD, ncvreg convention | ncvreg | 2.9e-3 | 14/15 | yes |
| MCP, textbook | skglm | 0.33 | 14/15 | yes |
| MCP, ncvreg convention | ncvreg | 0.11 | 9/15 | yes |

| | |
|---|---|
| where coefficients differ | both points are genuine critical points of the same objective |
| fixed point | started at the package's solution, our solver moves by less than $10^{-11}$, so each solution is a fixed point of the other's iteration |
| objective | neither systematically lower: ours lower at 2 $\lambda$, the package's at 5, equal at 23 |
| the tests require | stationarity and the fixed-point property; coefficient agreement only where the solvers share a path, SCAD vs skglm |
| for the thesis | reported MCP/SCAD results are properties of estimator plus path strategy |
| theory | Loh & Wainwright 2015: under restricted strong convexity every stationary point in a neighbourhood of the truth is statistically close to it, so the choice among such local minima need not matter asymptotically |

### 5.3 A SCAD-specific edge case

| | |
|---|---|
| where | the sparse edge of the exact-recovery window, $\lambda\approx 0.17\,\lambda_{\max}$ in Example 2 |
| what | SCAD's warm-started path settles in a local minimum with one weak edge still inside its lasso-like region $\lvert x\rvert\le\lambda$, so it stays biased there |
| MCP | did not do this, since its slope starts decreasing immediately at 0 |
| frequency | one $\lambda$ out of fifteen; the test allows exactly that |

---

## 6. Validation: `tests/test_nonconvex.py`

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

100-λ path, production tolerance `tol=1e-8`, one core, Apple M2:

| $p$ | lasso | MCP | SCAD |
|---|---|---|---|
| 10 | 0.96 s | 4.5 s | 6.2 s |
| 20 | 3.1 s | 15.1 s | 21.3 s |

| | |
|---|---|
| ratio | MCP about 5× the lasso, SCAD about 6.5×: two proximal-gradient steps per iteration, and nonconvex paths need more iterations |
| full Figure 5 grid | 38.6 CPU-h for the lasso, so about 190 CPU-h for MCP and 250 for SCAD |
| cluster | roughly 3 to 4 hours on the 64-task array in `cluster/s1_array.sbatch` with `--penalty` added; not a laptop job |
| if cost becomes the bottleneck | Li & Lin's Algorithm 2, non-monotone APG, skips the second step when the extrapolated point already decreases the objective enough; not needed to establish correctness, so not implemented |

---

## 8. Backend summary

| backend | lasso | MCP | SCAD | convention for MCP/SCAD |
|---|---|---|---|---|
| **`fista`**, default | ✓ | ✓ | ✓ | textbook or ncvreg |
| **`ncvreg`** | ✓ | ✓ | ✓ | **ncvreg only** |
| **`skglm`** | ✓ | ✓ | only with the diagonal penalized, no weighted SCAD | **textbook only** |
| `glmnet` | ✓ | — | — | — |
| `pyproximal` | ✓ | — | — | — |
| `design` | ✓ | — | — | — |

| loss | solver | penalties | convention |
|---|---|---|---|
| direct | every backend above | lasso, MCP, SCAD | as above |
| implied covariance, `--loss loglik` or `--loss frobenius` | active-set Newton, `src/gclm/solvers/covariance.py`, docs/LIKELIHOOD.md | the same three | textbook only; there is no design matrix, so the ncvreg convention does not apply |

---

## References

| | reference | used for |
|---|---|---|
| Fan & Li 2001 | Variable selection via nonconcave penalized likelihood and its oracle properties. JASA 96(456), 1348–1360 | SCAD |
| Zhang 2010 | Nearly unbiased variable selection under minimax concave penalty. Annals of Statistics 38(2), 894–942 | MCP |
| Breheny & Huang 2011 | Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection. Annals of Applied Statistics 5(1), 232–253 | ncvreg, and its standardized-covariate convention |
| Li & Lin 2015 | Accelerated proximal gradient methods for nonconvex programming. Advances in Neural Information Processing Systems 28 | Algorithm 1, used here |
| Loh & Wainwright 2015 | Regularized M-estimators with nonconvexity: statistical and algorithmic theory for local optima. JMLR 16, 559–616 | RSC and local optima |
| Loh & Wainwright 2017 | Support recovery without incoherence: a case for nonconvex regularization. Annals of Statistics 45(6), 2455–2482 | |
| Beck & Teboulle 2009 | A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM J. Imaging Sciences 2(1), 183–202 | the lasso path, docs/FISTA.md |
| skglm; pyproximal | skglm, JMLR 26 (2025), paper 24-0008; pyproximal, PyLops project | reference implementations |
