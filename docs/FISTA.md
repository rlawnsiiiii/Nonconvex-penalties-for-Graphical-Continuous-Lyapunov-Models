# FISTA backends

```
src/gclm/solvers/
├── proxgrad.py ─ solve_fista()          hand-written FISTA + adaptive restart   solver="fista"       (default)
└── backends.py ─ _pyproximal_path()     packaged FISTA, pyproximal + pylops     solver="pyproximal"
tests/
├── test_fista.py                        22 tests, Section 4: analytic solution, duality-gap
│                                        certificate, interior-point solver, every other backend
└── test_loss.py                         gradient checks, Section 1
```

---

## 1. Algorithm

$$\min_M\ F(M)=f(M)+g(M)$$

$$f(M)=\tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2,\qquad g(M)=\lambda\sum_{ij}v_{ij}\lvert M_{ij}\rvert$$

Constant step $t=1/L$:

$$M_{k+1}=\mathrm{prox}_{tg}\big(Y_k-t\,\nabla f(Y_k)\big)$$

$$t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}$$

$$Y_{k+1}=M_{k+1}+\frac{t_k-1}{t_{k+1}}\big(M_{k+1}-M_k\big)$$

$$\big[\mathrm{prox}_{tg}(Z)\big]_{ij}=\mathrm{sign}(Z_{ij})\max\big(\lvert Z_{ij}\rvert-t\lambda v_{ij},\,0\big)$$

$$F(M_k)-F(M^\star)=O(1/k^2)\qquad\text{(ISTA: }O(1/k)\text{; Beck and Teboulle 2009, Theorem 4.4)}$$

Gradient:

$$R(M)=M\hat\Sigma+\hat\Sigma M^\top+C,\qquad\nabla f(M)=2\,R(M)\,\hat\Sigma$$

$$f=\tfrac12\operatorname{tr}(R^2),\qquad df=\operatorname{tr}(R\,dR)=\operatorname{tr}\big(R(dM\,\hat\Sigma+\hat\Sigma\,dM^\top)\big)=2\langle R\hat\Sigma,\,dM\rangle$$

Lipschitz bound:

$$L=\lVert A(\hat\Sigma)\rVert_2^2\le\big(2\lambda_{\max}(\hat\Sigma)\big)^2$$

$$A=(\Sigma\otimes I)+(I\otimes\Sigma)K,\qquad\lVert\Sigma\otimes I\rVert_2=\lVert(I\otimes\Sigma)K\rVert_2=\lambda_{\max}(\Sigma),\qquad K\ \text{orthogonal}$$

Restart: $t_k\leftarrow1$ if

$$\langle Y_k-M_{k+1},\;M_{k+1}-M_k\rangle>0$$

Stop:

$$\lVert M_{k+1}-M_k\rVert_\infty<\texttt{tol}$$

```
solve_fista(Sigma, C, lam, v, tol, max_iter):             # src/gclm/solvers/proxgrad.py
    L = (2 lambda_max(Sigma))^2;  t = 1/L                 # bound / true constant: measured ratio 1.000
    M = Y = M0;  tk = 1
    repeat max_iter times:
        R     = Y Sigma + Sigma Y^T + C                   # three p x p products, O(p^3); A(Sigma) never formed
        grad  = 2 R Sigma
        M_new = soft_threshold(Y - t grad, t lam v)       # v_ii = 0: diagonal unpenalized
        if <Y - M_new, M_new - M> > 0:                    # restart (O'Donoghue & Candes 2015, gradient scheme):
            tk_new = 1;  Y = M_new                        #   momentum points against the prox step; no function
        else:                                             #   evaluations, no strong-convexity parameter (none
            tk_new = (1 + sqrt(1 + 4 tk^2)) / 2           #   exists here); plain FISTA is non-monotone, severe
            Y = M_new + (tk - 1)/tk_new (M_new - M)       #   for a singular Hessian, S1_reproduction.md Section 2.3
        delta = max_ij |M_new - M|
        M, tk = M_new, tk_new
        if delta < tol: break
    return M
```

---

## 2. References

```bibtex
@article{BeckTeboulle2009,
  author  = {Beck, A. and Teboulle, M.},
  title   = {A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems},
  journal = {SIAM J. Imaging Sciences}, volume = {2}, number = {1}, pages = {183--202}, year = {2009},
  doi     = {10.1137/080716542}}
@article{ODonoghueCandes2015,
  author  = {O'Donoghue, B. and Cand{\`e}s, E.},
  title   = {Adaptive Restart for Accelerated Gradient Schemes},
  journal = {Foundations of Computational Mathematics}, volume = {15}, pages = {715--732}, year = {2015},
  doi     = {10.1007/s10208-013-9150-2}}
@article{Nesterov1983,
  author  = {Nesterov, Y.},
  title   = {A method of solving a convex programming problem with convergence rate $O(1/k^2)$},
  journal = {Soviet Mathematics Doklady}, volume = {27}, number = {2}, pages = {372--376}, year = {1983}}
@article{ParikhBoyd2014,
  author  = {Parikh, N. and Boyd, S.},
  title   = {Proximal Algorithms},
  journal = {Foundations and Trends in Optimization}, volume = {1}, number = {3}, pages = {127--239}, year = {2014},
  doi     = {10.1561/2400000003}}
@incollection{BeckTeboulle2010,
  author    = {Beck, A. and Teboulle, M.},
  title     = {Gradient-based algorithms with applications to signal recovery},
  booktitle = {Convex Optimization in Signal Processing and Communications}, publisher = {CUP}, pages = {42--88}, year = {2010}}
```

| | | Varando & Hansen, Algorithm 1 |
|---|---|---|
| `BeckTeboulle2009` | updates; rate, Theorem 4.4; constant step, §4 | |
| `ODonoghueCandes2015` | restart, §3.2 | |
| `Nesterov1983` | acceleration | |
| `ParikhBoyd2014` | prox, soft threshold | ✓ |
| `BeckTeboulle2010` | backtracking; closed-form bound instead | ✓ |

| | | |
|---|---|---|
| Dettling, Appendix A | `glmnet`, coordinate descent | `glmnet` |
| Varando, Algorithm 1 | proximal gradient over $(B,C)$, $L(\Sigma(B,C))+\lambda\rho_1(B)+\kappa\lVert C-I_p\rVert_F^2$; Lyapunov solve, line search per iteration | S2 |
| `solve_fista` | FISTA, Section 1 | neither paper |

---

## 3. Packages

### 3.1 `pyproximal`

`pyproximal.L2(Op=...)`, `pylops` `LinearOperator`:

$$\texttt{matvec}:\ v\mapsto\mathrm{vec}(V\hat\Sigma+\hat\Sigma V^\top),\qquad\texttt{rmatvec}:\ u\mapsto\mathrm{vec}\big((U+U^\top)\hat\Sigma\big),\qquad O(p^3)$$

| | |
|---|---|
| weights | `pyproximal.L1(sigma=ndarray)` |
| `test_pyproximal_operator_is_the_design_matrix` | both directions vs explicit $A(\hat\Sigma)$ |
| restart | none; `acceleration` ∈ {`None`, `'vandenberghe'`, `'fista'`} |

`ProximalGradient(acceleration='fista')`: same recursion, $\tau=1/L$, prox:

```python
told = t; t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0; omega = (told - 1.0) / t
y = x + omega * (x - xold)
```

| ms to machine precision (to convergence), $\lambda=0.1\lambda_{\max}$ | $p=10$ | $p=20$ |
|---|---|---|
| `fista` | **19.5** | **38** |
| `pyproximal` | 1007 | 1561 |
| ratio | ~50× | ~41× |
| `pyproximal`, $F-F^\star$ | $\approx-6.7\times10^{-16}$ | $1.4\times10^{-14}$ |

Same optimum, wall-clock gap only; earlier $2.7\times10^{-5}$: `niter` cap artifact.

| `pyproximal` → `fista` | factor | measurement |
|---|---|---|
| restart (algorithmic) | ~2–3×, grows with $p$ | 187 → 63 iterations to $F-F^\star<10^{-12}$, identical code |
| per-iteration overhead (implementation) | ~4× | 42 → 10 $\mu\mathrm{s}$; `proxf(x)` per iteration when `tol is not None` (+1 operator application), `pylops`/`scipy` `LinearOperator` dispatch, `vec`/`unvec` reshapes |

### 3.2 Others

| | weighted $\ell_1$ | matrix-free | |
|---|---|---|---|
| `skglm` | `WeightedL1`, `WeightedMCPenalty`, `Quadratic`, no intercept; `SCAD` unweighted | ✗, $O(p^4)$ | `solver="skglm"`, JMLR 26 (2025); MCP without R |
| `ncvreg::ncvfit` (R) | ✓ | ✗, $O(p^4)$ | most accurate (`R/ENCODING.md`); 48× slower at $p=20$ |
| `glmnet` (R) | ✓ | ✗, $O(p^4)$ | Dettling's; least accurate; silent $\lambda$-path truncation on non-convergence |
| `cvxpy` | ✓ | ✗ | 100-point path far slower; 4.3 |
| `sklearn.linear_model.Lasso` | ✗ | | projection workaround converges badly |
| `copt` | | ✓ `minimize_proximal_gradient(accelerated=True)` | `mutex lock failed` on import |
| `proxmin` | ✗ | | subsumed by `pyproximal` |
| `celer` | | | superseded by `skglm` |

### 3.3 Path, 100 $\lambda$, warm starts, `tol = 1e-12`

| $p$ | | $t\,[\mathrm{s}]$ | max distance, exact KKT |
|---|---|---|---|
| 10 | `fista` | **1.78** | $8.9\times10^{-9}$ |
| 10 | `skglm` `AndersonCD` | 2.24 | $8.9\times10^{-7}$ |
| 10 | `skglm` `FISTA`, no warm starts | 42 | |
| 20 | `fista` | **6.84** | $1.2\times10^{-7}$ |
| 20 | `skglm` `AndersonCD` | 30.45 | $3.1\times10^{-6}$ |

---

## 4. Evidence

`tests/test_fista.py`: five directions, none via another first-order method.

### 4.1 Analytic solution

$$\hat\Sigma=\mathrm{diag}(s),\quad C=\mathrm{diag}(c),\quad M=\mathrm{diag}(m)\ \Rightarrow\ R=\mathrm{diag}(2m_is_i+c_i)$$

$$M^\star=\mathrm{diag}\big(-c_i/(2s_i)\big),\qquad f(M^\star)=g(M^\star)=0\ \text{for every }\lambda$$

| $\lambda$ | $10^{-3}$ | $0.5$ | $10$ |
|---|---|---|---|
| `solve_fista` error | $2.4\times10^{-14}$ | $2.4\times10^{-14}$ | $2.4\times10^{-14}$ |

### 4.2 Duality gap

$$P(\beta)=\tfrac12\lVert y-X\beta\rVert^2+\lambda\sum_jw_j\lvert\beta_j\rvert$$

$$\theta=\text{residual, rescaled so that }\lvert x_j^\top\theta\rvert\le\lambda w_j\ \text{for all }j\quad\Rightarrow\quad\theta\ \text{dual feasible}$$

$$D(\theta)=\tfrac12\lVert y\rVert^2-\tfrac12\lVert y-\theta\rVert^2\le\min_\beta P(\beta)$$

$$0\le P(\hat\beta)-\min_\beta P\le P(\hat\beta)-D(\theta)$$

| $\lambda$ | 0.05 | 0.3 | 2.0 |
|---|---|---|---|
| $P-D$ | $1.3\times10^{-13}$ | $6.7\times10^{-16}$ | $-7.4\times10^{-15}$ |

Control: gap widens as `tol` loosens.

### 4.3 Interior point

| | class | $F$ | $\Delta F$ |
|---|---|---|---|
| `cvxpy` + `CLARABEL` | interior-point conic | 0.759801966624 | $2.9\times10^{-8}$ |
| `fista` | first-order proximal | 0.759801964632 (lower) | |

### 4.4 Invariances

| | measured |
|---|---|
| $F^\star$ over 5 starts | spread **0.0** |
| permutation equivariance | $4.4\times10^{-16}$ |
| scale equivariance, $(C,\lambda)\mapsto(aC,a\lambda)\Rightarrow M^\star\mapsto aM^\star$ | $10^{-9}$ |
| $\lambda=0\Rightarrow\nabla f=0$ | $<10^{-6}$ |

### 4.5 Rate

$$F(x_k)-F^\star=O(1/k^2)\ \Rightarrow\ k^2\big(F(x_k)-F^\star\big)\ \text{bounded}$$

| $k$ | 10 | 20 | 50 | 200 |
|---|---|---|---|---|
| $k^2\big(F(x_k)-F^\star\big)$ | 4.31 | 0.56 | $1.2\times10^{-6}$ | |
| $F(x_k)-F^\star$ | | | | machine precision |

Decreasing (growth: broken momentum).

### 4.6 Cross-checks

| | against | result |
|---|---|---|
| `test_gradient_matches_design_matrix` | $X^\top(X\beta-y)$ | exact |
| `test_gradient_matches_finite_differences` | central differences | $10^{-5}$ |
| `test_lipschitz_bound_dominates_true_constant` | $L\ge\lambda_{\max}(A^\top A)$ | holds, within 4× |
| `test_solvers_match_exact_restricted_solution` | closed-form KKT solution, no optimizer | $10^{-9}$; `glmnet` inaccurate, dense end |
| `test_fista_matches_ncvreg` | `ncvreg::ncvfit` | $10^{-9}$ |
| `test_all_backends_agree` | `design`, `glmnet`, `ncvreg`, `skglm` | identical supports and metrics |
| `test_example2_path_is_recovered_in_population_limit` | Dettling Figure 3, $n=\infty$ | exact |

### 4.7 Campaign tolerance, 12 October 2026

| | |
|---|---|
| tolerance, 4.1–4.6 | $\le10^{-12}$ |
| campaign | $10^{-8}$ |
| reference | 12 campaign data sets; KKT-certified lasso, every $\lambda$ |
| FISTA objective, relative | $\le2.4\times10^{-7}$ |
| FISTA support | differs, 0–9 of 100 $\lambda$, one entry each |
| oracle $F_1$ | unchanged |
| AUPR | within 0.0006 |
| `glmnet`, `ncvreg`, `skglm` defaults | support missed, 35–87 of 100 $\lambda$ |
| FISTA, $10^{-10}$ | exact everywhere, 1.6× time |
| also | MCP, SCAD, log-likelihood solver |

[`../next_steps/Oct2026/121026`](../next_steps/Oct2026/121026/solver_accuracy_121026.md)

---

## 5. Default

Speed only, same optimum (Section 3).

| path $[\mathrm{s}]$ | `fista` | `ncvreg` | `skglm` |
|---|---|---|---|
| $p=10$ | 1.8 | 10 | 2.2 |
| $p=20$ | 6.8 | 244 | 30 |

| | |
|---|---|
| `ncvreg`, `skglm`, `glmnet` | structural: explicit $p^2\times p^2$ design, $O(p^4)$ per sweep vs matrix-free $O(p^3)$; `ncvreg` 48× slower at $p=20$, gap grows with $p$ |
| `pyproximal` | incidental: ~4× overhead, ~2–3× iterations (restart); restart package would be competitive |
| Figure 5 grid, $p\le50$ | `fista` only in practical time |
| `ncvreg`, $4\times10^{-12}$ at moderate $p$ | `--solver ncvreg` for accuracy-critical runs, S1b estimation error |
| burden of proof | Section 4, independent of FISTA |

```bash
python simulations/run_s1.py                                   # fista (default)
python simulations/run_s1.py --p 10 15 --solver ncvreg         # accuracy-critical runs
python simulations/run_s1.py --penalty MCP                     # S1b (docs/NONCONVEX.md)
```
