# The FISTA backends: packaged and hand-written

Purpose: document the default solver `fista`, the packaged FISTA `pyproximal`, and the package survey behind the choice.

```
src/gclm/solvers/
├── proxgrad.py ─ solve_fista()          hand-written FISTA, the default       solver="fista"
└── backends.py ─ _pyproximal_path()     packaged FISTA, pyproximal + pylops   solver="pyproximal"
tests/
├── test_fista.py                        22 tests, Section 4: analytic solution, duality-gap
│                                        certificate, interior-point solver, every other backend
└── test_loss.py                         gradient checks, Section 1
```

---

## 1. The algorithm

Minimize $F(M)=f(M)+g(M)$, $f$ smooth and convex, $g$ convex and non-smooth:

$$f(M)=\tfrac12\big\|M\hat\Sigma+\hat\Sigma M^\top+C\big\|_F^2$$
$$g(M)=\lambda\sum_{ij}v_{ij}\lvert M_{ij}\rvert$$

FISTA iteration, Beck and Teboulle 2009, constant step $t=1/L$:

$$M_{k+1}=\mathrm{prox}_{t g}\!\big(Y_k-t\,\nabla f(Y_k)\big)$$
$$t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}$$
$$Y_{k+1}=M_{k+1}+\frac{t_k-1}{t_{k+1}}\big(M_{k+1}-M_k\big)$$
$$\big[\mathrm{prox}_{tg}(Z)\big]_{ij}=\mathrm{sign}(Z_{ij})\max\big(\lvert Z_{ij}\rvert-t\lambda v_{ij},\,0\big)$$
$$F(M_k)-F(M^\star)=O(1/k^2)\qquad\text{(ISTA: }O(1/k)\text{; Beck and Teboulle 2009, Theorem 4.4)}$$

Gradient, with $R(M)=M\hat\Sigma+\hat\Sigma M^\top+C$:

$$\nabla f(M)=2\,R(M)\,\hat\Sigma$$
$$f=\tfrac12\operatorname{tr}(R^2),\qquad df=\operatorname{tr}(R\,dR)=\operatorname{tr}\!\big(R(dM\,\hat\Sigma+\hat\Sigma\,dM^\top)\big)=2\langle R\hat\Sigma,\,dM\rangle$$

Step size $t=1/L$:

$$L=\big\|A(\hat\Sigma)\big\|_2^2\le\big(2\lambda_{\max}(\hat\Sigma)\big)^2$$
$$A=(\Sigma\otimes I)+(I\otimes\Sigma)K,\qquad\big\|\Sigma\otimes I\big\|_2=\big\|(I\otimes\Sigma)K\big\|_2=\lambda_{\max}(\Sigma),\qquad K\ \text{orthogonal}$$

Adaptive restart, gradient scheme of O'Donoghue and Candès 2015: reset $t_k\leftarrow1$ whenever

$$\big\langle Y_k-M_{k+1},\;M_{k+1}-M_k\big\rangle>0$$

Stopping rule:

$$\lVert M_{k+1}-M_k\rVert_\infty<\texttt{tol}$$

| part | fact |
|---|---|
| gradient cost | three $p\times p$ products, $O(p^3)$; the design matrix $A(\hat\Sigma)$ is never formed |
| gradient checks | `tests/test_loss.py`: finite differences and $X^\top(X\beta-y)$ |
| bound tightness | measured ratio 1.000, exactly tight, so nothing to gain from computing the true constant by power iteration |
| bound test | `test_lipschitz_bound_dominates_true_constant` asserts $L\ge\lambda_{\max}(A^\top A)$; otherwise FISTA would diverge |
| why restart | plain FISTA is non-monotone: the momentum overshoots and the objective oscillates, severe here because the Hessian is singular, §2.3 of `S1_reproduction.md` |
| restart meaning | the momentum points against the direction the proximal step just took |
| restart cost | no function evaluations and no strong-convexity parameter, which does not exist here |

---

## 2. References

| | |
|---|---|
| **Beck, A. & Teboulle, M. (2009).** *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sciences **2**(1), 183–202. [doi:10.1137/080716542](https://doi.org/10.1137/080716542) | FISTA itself: the update above, the $O(1/k^2)$ rate (Thm 4.4), and the constant-step variant we use (§4). |
| **O'Donoghue, B. & Candès, E. (2015).** *Adaptive Restart for Accelerated Gradient Schemes.* Foundations of Computational Mathematics **15**, 715–732. [doi:10.1007/s10208-013-9150-2](https://doi.org/10.1007/s10208-013-9150-2) | The restart condition. Their "gradient scheme", §3.2: the inner-product test, which is what the code implements. |
| **Nesterov, Y. (1983).** *A method of solving a convex programming problem with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady **27**(2), 372–376. | The original acceleration that FISTA extends to the composite setting. |
| **Parikh, N. & Boyd, S. (2014).** *Proximal Algorithms.* Foundations and Trends in Optimization **1**(3), 127–239. [doi:10.1561/2400000003](https://doi.org/10.1561/2400000003) | Reference for proximal operators generally, incl. the soft-threshold. **Cited by Varando & Hansen for their Algorithm 1**, so the same family the GCLM literature already uses. |
| **Beck, A. & Teboulle, M. (2010).** *Gradient-based algorithms with applications to signal recovery.* In *Convex Optimization in Signal Processing and Communications*, CUP, 42–88. | Backtracking line search; cited by Varando & Hansen for the step-size rule in Algorithm 1. We use the closed-form bound instead, since it is tight. |

Relation to the GCLM papers:

| paper | method | relation |
|---|---|---|
| Dettling, Appendix A | Direct Lyapunov Lasso fitted with *"the R package `glmnet`, which runs a coordinate descent algorithm"* | the `glmnet` backend |
| Varando, Algorithm 1 | a proximal-gradient method, citing Parikh & Boyd and Beck & Teboulle 2010, for $L(\Sigma(B,C))+\lambda\rho_1(B)+\kappa\lVert C-I_p\rVert_F^2$ over $B$ **and** $C$; a Lyapunov equation solved every iteration; a line search keeps $B$ stable | the reference for S2, not S1 |
| `solve_fista` | a standard method applied to our problem | a transcription of neither paper; neither prescribes FISTA for this estimator |

---

## 3. Package survey

Every candidate was installed and run on the real problem; the outcomes are measured, not assumed.

### Packaged FISTA: `pyproximal`

[`pyproximal`](https://pyproximal.readthedocs.io) with `pylops`, `solver="pyproximal"`, matrix-free through `pyproximal.L2(Op=...)` on any `pylops` `LinearOperator`:

$$\texttt{matvec}:\ v\mapsto\mathrm{vec}(V\hat\Sigma+\hat\Sigma V^\top)$$
$$\texttt{rmatvec}:\ u\mapsto\mathrm{vec}\big((U+U^\top)\hat\Sigma\big)$$

| item | fact |
|---|---|
| cost | both $O(p^3)$; no $p^2\times p^2$ matrix is ever formed |
| weights | `pyproximal.L1(sigma=ndarray)` takes per-element weights, so the unpenalized diagonal is expressible |
| operator check | `test_pyproximal_operator_is_the_design_matrix`: both directions, against the explicit $A(\hat\Sigma)$ |

Measured, single $\lambda=0.1\lambda_{\max}$, run to convergence rather than to a fixed iteration budget:

| | $p=10$ | $p=20$ |
|---|---|---|
| `fista` (hand-written), time to machine precision | **19.5 ms** | **38 ms** |
| `pyproximal`, time to machine precision | 1007 ms | 1561 ms |
| ratio | ~50× | ~41× |
| `pyproximal`, `F − F*` reached | ≈ −6.7e-16 | 1.4e-14 |

| | |
|---|---|
| verdict | both converge to the same optimum; the `tol` of `pyproximal` terminates early rather than burning the whole budget; the difference is wall-clock, not accuracy |
| correction | an earlier version of this file reported `pyproximal` reaching only 2.7e-05: an artifact of capping `niter` in the benchmark, not a limit of the package; the speed gap is real, the accuracy gap was not |

#### Exactly what differs

`pyproximal.optimization.primal.ProximalGradient` with `acceleration='fista'`:

```python
told = t; t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0; omega = (told - 1.0) / t
y = x + omega * (x - xold)
```

Character for character the same recursion as ours, with the same $\tau=1/L$ and the same prox; the only algorithmic difference is the restart branch.

| source of the runtime gap | kind | factor | how measured |
|---|---|---|---|
| **adaptive restart** | algorithmic | ~2–3× (more as $p$ grows) | toggling restart on/off in *identical* code: 63 vs 187 iterations to reach `F−F* < 1e-12` |
| per-iteration overhead | implementation | ~4× | 42 µs/iter vs 10 µs/iter on the same problem |

| overhead component | detail |
|---|---|
| stopping test | when `tol is not None`, `ProximalGradient` evaluates `proxf(x)` every iteration; with `L2(Op=...)` that is one extra operator application on top of the two the gradient already costs |
| dispatch | `pylops`/`scipy` `LinearOperator` dispatch and the `vec`/`unvec` reshapes on each application |
| restart | a published technique, O'Donoghue & Candès 2015, that `pyproximal` does not expose; its options are `None`, `'vandenberghe'`, `'fista'`; with restart and a cheaper stopping test the gap would largely close |

### The rest

| package | verdict |
|---|---|
| `sklearn.linear_model.Lasso` | **Cannot express the problem.** No per-feature penalty weights, so the unpenalized diagonal is impossible. The workaround (profile the diagonal out by projection) is correct but converges badly on the rank-deficient design. |
| `copt` | **Unusable here**: segfaults on import in this environment (`mutex lock failed`). Its `minimize_proximal_gradient(accelerated=True)` would otherwise have been a good matrix-free fit. |
| `proxmin` | Installs, but is a general multi-block PGM/APGM toolkit with no weighted-L1 prox out of the box; subsumed by `pyproximal` for our purposes. |
| `celer` | `skglm`'s predecessor; superseded by it. |
| **`skglm`** (JMLR 26, 2025) | **Works**: `WeightedL1`, `WeightedMCPenalty`, `Quadratic` datafit, no intercept, and MCP **without R**. Wired in as `solver="skglm"`. But it operates on the explicit $p^2\times p^2$ design, so it is $O(p^4)$ per sweep against our $O(p^3)$ per iteration. Measured below. `SCAD` has no weighted variant, so the unpenalized diagonal is not expressible there. |
| `ncvreg::ncvfit` (R) | **Works and is the most accurate**, see `R/ENCODING.md`. Also $O(p^4)$; 48× slower than `fista` at $p=20$. |
| `glmnet` (R) | Dettling's own choice. Least accurate, and silently truncates the λ path when it fails to converge. |
| `cvxpy` | General convex modelling; would need the explicit design and is far slower for a 100-point path. |

### Measured, 100-λ path, warm-started, `tol = 1e-12`

| $p$ | backend | time | max distance to exact KKT solution |
|---|---|---|---|
| 10 | `fista` (ours) | **1.78 s** | **8.9e-09** |
| 10 | `skglm` AndersonCD | 2.24 s | 8.9e-07 |
| 10 | `skglm` `FISTA`, no warm starts | 42 s | |
| 20 | `fista` (ours) | **6.84 s** | **1.2e-07** |
| 20 | `skglm` AndersonCD | 30.45 s | 3.1e-06 |

---

## 4. Evidence that the hand-written solver is correct

`tests/test_fista.py` attacks `solve_fista` from five independent directions; none relies on agreement with another first-order method.

### 4.1 An analytic solution, no optimizer involved

$\hat\Sigma=\mathrm{diag}(s)$ and $C=\mathrm{diag}(c)$ with diagonal $M$ give a diagonal residual with entries $2m_is_i+c_i$, so

$$M^\star=\mathrm{diag}\!\big(-c_i/(2s_i)\big)$$

| fact | value |
|---|---|
| loss at $M^\star$ | exactly zero at zero penalty cost: the global minimum, for every $\lambda$ |
| `solve_fista` error | **2.4e-14**, at $\lambda=10^{-3}$, $0.5$ and $10$ alike |

### 4.2 A duality-gap certificate, a bound, not a comparison

$$P(\beta)=\tfrac12\lVert y-X\beta\rVert^2+\lambda\sum_jw_j\lvert\beta_j\rvert$$
$$\theta=\text{the residual, rescaled so that }\lvert x_j^\top\theta\rvert\le\lambda w_j\ \text{for all }j\quad\Rightarrow\quad\theta\ \text{dual feasible}$$
$$D(\theta)=\tfrac12\lVert y\rVert^2-\tfrac12\lVert y-\theta\rVert^2\le\min_\beta P(\beta)$$

The gap $P-D$ bounds suboptimality from above with no reference solution at all.

| $\lambda$ | 0.05 | 0.3 | 2.0 |
|---|---|---|---|
| gap | **1.3e-13** | **6.7e-16** | **-7.4e-15** |
| control | a second test: the gap widens when `tol` is loosened, so the certificate tracks the solver | | |

### 4.3 An independent solver class

| solver | class | objective | agreement |
|---|---|---|---|
| `cvxpy` with **CLARABEL** | interior-point conic; no shared code path or failure mode with a first-order proximal method | 0.759801966624 | **2.9e-08** |
| `fista` | first-order proximal | 0.759801964632, the lower one | |

### 4.4 Invariances the true minimizer must satisfy

| property | why it must hold | measured |
|---|---|---|
| initialization independence | the problem is convex, so the optimal *value* is unique | spread **0.0** over 5 starts |
| permutation equivariance | relabelling variables may only relabel the answer | **4.4e-16** |
| scale equivariance | $(C,\lambda)\mapsto(aC,a\lambda)$ scales the minimizer by $a$ | 1e-9 |
| $\lambda=0\Rightarrow\nabla f=0$ | nothing left to balance the gradient | < 1e-6 |

### 4.5 The promised convergence rate

$$F(x_k)-F^\star=O(1/k^2)\ \text{(Beck and Teboulle)}\quad\Rightarrow\quad k^2\big(F(x_k)-F^\star\big)\ \text{stays bounded}$$

| $k$ | 10 | 20 | 50 | 200 |
|---|---|---|---|---|
| $k^2\big(F(x_k)-F^\star\big)$ | 4.31 | 0.56 | 1.2e-06 | |
| $F(x_k)-F^\star$ | | | | machine precision |

Decreasing, so the adaptive restart buys better than guaranteed convergence; broken momentum would show as growth.

### 4.6 Cross-checks against the other backends

| check | against | result |
|---|---|---|
| `test_gradient_matches_design_matrix` | $X^\top(X\beta-y)$ | exact |
| `test_gradient_matches_finite_differences` | central differences | 1e-5 |
| `test_lipschitz_bound_dominates_true_constant` | $\lambda_{\max}(A^\top A)$ | bound holds, within 4× |
| `test_solvers_match_exact_restricted_solution` | **closed-form KKT solution** | **1e-9** |
| `test_fista_matches_ncvreg` | `ncvreg::ncvfit` | 1e-9 |
| `test_all_backends_agree` | `design`, `glmnet`, `ncvreg`, `skglm` | identical supports and metrics |
| `test_example2_path_is_recovered_in_population_limit` | Dettling Figure 3, $n=\infty$ | exact |

| | |
|---|---|
| closed-form KKT check | the important one: given the active set and signs the optimum solves a linear system, so the target involves no optimizer at all; it also established that `glmnet`, not our code, is the inaccurate one at the dense end of the path |

### 4.7 Accuracy at the campaign's tolerance, against the packages, 12 October 2026

| quantity | value |
|---|---|
| tolerance in 4.1 to 4.6 | $10^{-12}$ or tighter |
| campaign tolerance | $10^{-8}$ |
| reference | 12 campaign data sets, the exact KKT-certified lasso solution at every λ |
| FISTA objective, relative | within $2.4\times10^{-7}$ |
| FISTA support | differs at 0 to 9 of 100 λ, one entry each |
| oracle $F_1$ | unchanged |
| AUPR | within 0.0006 |
| glmnet, ncvreg, skglm at their defaults | miss the support at 35 to 87 of 100 λ |
| FISTA at $10^{-10}$ | exact everywhere, for 1.6 times the time |
| also checked in the same study | MCP, SCAD, the log-likelihood solver |

[`next_steps/121026/solver_accuracy_121026.md`](../next_steps/121026/solver_accuracy_121026.md)

---

## 5. Why `fista` is the default

Default `fista` for speed on the full grid, not because it is a better or more accurate algorithm: it is the same algorithm as any FISTA and converges to the same point, Section 3.

| against | advantage | reason |
|---|---|---|
| `ncvreg`, `skglm`, `glmnet` | *structural* | they need the explicit $p^2\times p^2$ design, so $O(p^4)$ per sweep against the matrix-free $O(p^3)$ |
| `pyproximal`, also matrix-free here | *incidental* | ~4× lower per-iteration overhead and ~2–3× fewer iterations from adaptive restart; a package with restart would be competitive |

| $p$ | `fista` (hand-written) | `ncvreg` | `skglm` |
|---|---|---|---|
| 10 | 1.8 s | 10 s | 2.2 s |
| 20 | 6.8 s | 244 s | 30 s |

| fact | consequence |
|---|---|
| `ncvreg` already 48× slower at $p=20$ | the gap grows with $p$ |
| the Figure 5 grid runs to $p=50$ | no package backend reaches it in practical time, so the default has to be `fista` for the reproduction to be finishable at all |
| the burden of proof | falls on Section 4, whose analytic solution, duality-gap certificate and interior-point cross-check are independent of FISTA's own failure modes |
| `ncvreg` reaches 4e-12 at moderate $p$ | one flag away; the right choice when accuracy matters more than speed, in particular S1b's estimation-error comparison |

```bash
python simulations/run_s1.py                                   # fista (default)
python simulations/run_s1.py --p 10 15 --solver ncvreg         # accuracy-critical runs
python simulations/run_s1.py --penalty MCP                     # S1b (docs/NONCONVEX.md)
```
