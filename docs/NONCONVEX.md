# MCP and SCAD for the direct Lyapunov estimator

```
simulations/run_s1.py        --penalty {lasso,MCP,SCAD}   --solver {fista,ncvreg,skglm,glmnet,pyproximal,design}
                             --convention {textbook,ncvreg}   --gamma   --loss {direct,loglik,frobenius}
src/gclm/
  solvers/path.py            lasso_path(solver=, **solver_kwargs)   lambda_max, grid, warm starts, convention guard   Sections 2, 4
  solvers/proxgrad.py        solve_fista(penalty=, gamma=, convention=) -> _solve_mapg                                  Section 4
  objective/penalties.py     P, P', prox, penalty_scale; both conventions                                               Sections 1-3
  lyapunov.py                design_column_sq_norms -> v_ik                                                             Section 2.2
  solvers/backends.py        skglm | ncvreg | glmnet | pyproximal | design;  scaled design sqrt(n) X, sqrt(n) y         Sections 2.4, 8
  solvers/covariance.py      --loss loglik | frobenius;  active-set Newton;  textbook only                              Section 8
tests/test_nonconvex.py      83 tests                                                                                   Section 6
```

```bash
python simulations/run_s1.py --penalty MCP                                  # fista, textbook
python simulations/run_s1.py --penalty SCAD --gamma 3.7
python simulations/run_s1.py --penalty MCP --solver skglm                   # package, textbook
python simulations/run_s1.py --penalty MCP --solver ncvreg --convention ncvreg
```

---

## 1. Estimator

Dettling, eq. 1.4:

$$\hat M(\lambda)=\arg\min_{M}F(M),\qquad F(M)=\tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2+\sum_{i,j}w_{ij}P_{\lambda,\gamma}(M_{ij}),\qquad w_{ii}=0,\quad w_{ij}=1\ (i\ne j)$$

$$x\ge0,\qquad P(-x)=P(x)$$

$$P^{\mathrm{lasso}}_{\lambda}(x)=\lambda x,\qquad P'(x)=\lambda$$

$$P^{\mathrm{MCP}}_{\lambda,\gamma}(x)=\begin{cases}\lambda x-\dfrac{x^2}{2\gamma} & x\le\gamma\lambda\\ \dfrac{\gamma\lambda^2}{2} & x>\gamma\lambda\end{cases},\qquad P'(x)=(\lambda-x/\gamma)_+$$

$$P^{\mathrm{SCAD}}_{\lambda,\gamma}(x)=\begin{cases}\lambda x & x\le\lambda\\ \dfrac{2\gamma\lambda x-x^2-\lambda^2}{2(\gamma-1)} & \lambda<x\le\gamma\lambda\\ \dfrac{\lambda^2(\gamma+1)}{2} & x>\gamma\lambda\end{cases},\qquad P'(x)=\begin{cases}\lambda & x\le\lambda\\ \dfrac{\gamma\lambda-x}{\gamma-1} & \lambda<x\le\gamma\lambda\\ 0 & x>\gamma\lambda\end{cases}$$

| | `lasso` | `MCP`, Zhang 2010 | `SCAD`, Fan & Li 2001 |
|---|---|---|---|
| $\gamma$ default | — | 3 | 3.7 |
| $\gamma$ domain, `test_gamma_domain_matches_ncvreg` | — | $\gamma>1$ | $\gamma>2$ |
| $P'(0^+)$ | $\lambda$ | $\lambda$ | $\lambda$ |
| $P'(x)$, $x>\gamma\lambda$ | $\lambda$ | $0$ | $0$ |

---

## 2. Conventions

### 2.1 What `ncvfit` minimizes

$$F_{\mathrm{ncvreg}}(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert^2+\sum_j\frac{P_{\lambda,\gamma}(v_j\beta_j)}{v_j},\qquad v_j=\frac{x_j^\top x_j}{n}$$

$$F_{\mathrm{textbook}}(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert^2+\sum_j P_{\lambda,\gamma}(\beta_j)$$

$$\lambda\lvert v\beta\rvert/v=\lambda\lvert\beta\rvert\ \Rightarrow\ F_{\mathrm{ncvreg}}=F_{\mathrm{textbook}}\ \ \text{(lasso)};\qquad \text{MCP, SCAD:}\ \ F_{\mathrm{ncvreg}}=F_{\mathrm{textbook}}\iff v_j=1\ \ \forall j$$

$$\text{MCP:}\qquad P_{\lambda,\gamma}(v\beta)/v=P_{\lambda,\gamma/v}(\beta)$$

`test_conventions_coincide_for_the_lasso`, `test_conventions_coincide_when_all_curvatures_are_one`

| `ncvfit` solution | $F_{\mathrm{ncvreg}}$ | $F_{\mathrm{textbook}}$ |
|---|---|---|
| stationarity violation, plain regression, `MCP`, `SCAD`, `test_what_ncvfit_actually_minimises` | $10^{-16}$ | 0.2–0.8 |
| stationarity violation, Lyapunov design, $p=6$, `MCP`, `SCAD`, `test_ncvreg_is_not_stationary_for_the_textbook_objective` | $10^{-11}$ | 0.28–1.3 |
| `lasso` vs every other backend | $10^{-12}$ | $10^{-12}$ |

### 2.2 Lyapunov design

$$v_{ik}=\lVert A(\hat\Sigma)e_{ik}\rVert^2=2\big(\lVert\hat\Sigma_{k,\cdot}\rVert^2+\hat\Sigma_{ik}^2\big)$$

`design_column_sq_norms`, `src/gclm/lyapunov.py`, `test_column_sq_norms_closed_form`

### 2.3 `convention=`

`lasso_path`, `solve_fista`, `run_s1.py`:

| `convention=` | $P_{ij}(M_{ij})$ | `--solver` | concavity |
|---|---|---|---|
| `"textbook"` (default) | $P_{\lambda,\gamma}(M_{ij})$ | `fista`, `skglm` | $\mu_{\mathrm{MCP}}=1/\gamma$, $\mu_{\mathrm{SCAD}}=1/(\gamma-1)$ |
| `"ncvreg"` | $P_{\lambda,\gamma}(v_{ij}M_{ij})/v_{ij}$ | `fista`, `ncvreg` | $\gamma$ relative to $v_{ij}$ |
| mismatch | | `ValueError`, `test_backends_refuse_a_convention_they_do_not_solve` | |

### 2.4 Scaled design, `ncvfit` and `skglm` `Quadratic`

$$n=\mathrm{nrow}(X)=p^2,\qquad X\mapsto\sqrt n\,X,\ \ y\mapsto\sqrt n\,y:\qquad \frac{1}{2n}\lVert y-X\beta\rVert^2\ \mapsto\ \tfrac12\lVert y-X\beta\rVert^2,\qquad(\lambda,\gamma)\ \text{unchanged}$$

| unscaled design, converted $(\lambda',\gamma')$ | |
|---|---|
| `SCAD`, `"textbook"` | $n\,\mathrm{SCAD}_{\lambda',\gamma'}\ne\mathrm{SCAD}_{\lambda,\gamma}\ \ \forall(\lambda,\gamma)$; best candidate differs by up to 0.48 |
| `MCP` | `gamma=3` $\to\gamma=3/p^2$, outside $\gamma>1$ |
| `"ncvreg"` | scaled = unscaled |

---

## 3. Proximal operators

$$\mathrm{prox}_{tP}(z)=\arg\min_x\tfrac12(x-z)^2+tP(x),\qquad z>0$$

$$\gamma>t:\qquad \mathrm{prox}^{\mathrm{MCP}}_{tP}(z)=\begin{cases}0 & z\le t\lambda\\ \dfrac{z-t\lambda}{1-t/\gamma} & t\lambda<z\le\gamma\lambda\\ z & z>\gamma\lambda\end{cases}$$

$$\gamma>1+t:\qquad \mathrm{prox}^{\mathrm{SCAD}}_{tP}(z)=\begin{cases}(z-t\lambda)_+ & z\le(1+t)\lambda\\ \dfrac{(\gamma-1)z-t\gamma\lambda}{\gamma-1-t} & (1+t)\lambda<z\le\gamma\lambda\\ z & z>\gamma\lambda\end{cases}$$

$$\min_x\tfrac12(x-z)^2+\tfrac tv P(vx):\qquad u=vx,\qquad x^\star=\mathrm{prox}_{tvP}(vz)/v$$

| `prox`, `src/gclm/objective/penalties.py` | |
|---|---|
| $\gamma>t$ (`MCP`), $\gamma>1+t$ (`SCAD`) | strictly convex, unique minimizer; else `ValueError`, `test_prox_refuses_steps_where_it_is_not_unique` |
| knots | continuous |
| $z>\gamma\lambda\Rightarrow x^\star=z$ | `test_prox_leaves_large_entries_unshrunk` |
| $w_{ij}=0$ | `test_prox_leaves_unpenalised_entries_alone` |

| one `prox` call, $p^2=2500$ | ms |
|---|---|
| `skglm` `prox_1d`, scalar loop | 0.62 |
| `pyproximal` `SCAD.prox` | 0.052 |
| `gclm.objective.penalties.prox` | 0.030 |

$p=50$ path: $10^5$–$10^6$ `prox` calls.

---

## 4. Algorithm: MAPG, Li & Lin 2015

```
solve_fista(penalty="lasso")                  FISTA + adaptive restart, unchanged, bit-identical
                                              (test_default_path_is_unchanged_for_the_lasso)
solve_fista(penalty="MCP" | "SCAD")  ->  _solve_mapg      Li & Lin 2015, Algorithm 1
                                              (plain FISTA: no convergence guarantee for a nonconvex penalty)

_solve_mapg(Sigma, C, lam, weights, m_init, tol, max_iter, gamma, convention):
    L = 4 lambda_max(Sigma)^2                 # lipschitz_bound
    a = 0.99 / L                              # a < 1/L, Theorem 1
    R(M)    = M Sigma + Sigma M' + C          # residual, cached: one per candidate point
    f(M)    = 0.5 ||R(M)||_F^2 ;  grad f(M) = 2 R(M) Sigma
    F(M)    = f(M) + sum_ij w_ij P_ij(M_ij)
    x = m_init (warm start) or 0 ;  x_prev = z = x ;  t_prev, t = 0, 1
    repeat up to max_iter:
        y = x + (t_prev / t) (z - x) + ((t_prev - 1) / t) (x - x_prev)
        z = prox_aP(y - a grad f(y))          # extrapolated step
        v = prox_aP(x - a grad f(x))          # plain step
        t_prev, t = t, (sqrt(4 t^2 + 1) + 1) / 2
        x_prev = x
        x = z if F(z) <= F(v) else v          # monotone objective
        stop if max |x - x_prev| < tol
    # per iteration: two proximal-gradient steps, two objective values
    # converges to a critical point: MCP and SCAD are semi-algebraic, hence Kurdyka-Lojasiewicz (Li & Lin, Theorem 1)
    # not implemented: Li & Lin Algorithm 2, non-monotone APG, which skips the second step when the
    #                  extrapolated point already decreases the objective enough
```

$$y_k=x_k+\tfrac{t_{k-1}}{t_k}(z_k-x_k)+\tfrac{t_{k-1}-1}{t_k}(x_k-x_{k-1})$$

$$z_{k+1}=\mathrm{prox}_{aP}\big(y_k-a\nabla f(y_k)\big)$$

$$v_{k+1}=\mathrm{prox}_{aP}\big(x_k-a\nabla f(x_k)\big)$$

$$t_{k+1}=\big(\sqrt{4t_k^2+1}+1\big)/2$$

$$x_{k+1}=z_{k+1}\ \text{if}\ F(z_{k+1})\le F(v_{k+1}),\ \text{else}\ v_{k+1}$$

$$L=4\lambda_{\max}(\hat\Sigma)^2,\qquad a=0.99/L<1/L$$

$$\hat\Sigma_{kk}=1\ \Rightarrow\ v_{ij}\ge2\hat\Sigma_{kk}^2=2,\qquad L\ge\max_{ij}v_{ij}\ge2,\qquad a<\tfrac12$$

$$\gamma>a\ (\mathrm{MCP}),\qquad \gamma>1+a\ (\mathrm{SCAD}),\qquad \text{ncvreg: }a\,v_{ij}\le0.99\ \text{in place of }a$$

$$\lVert x_{k+1}-x_k\rVert_\infty<\mathtt{tol}$$

```
lasso_path(solver=, penalty=, gamma=, convention=)          # src/gclm/solvers/path.py
    P'(0+) = lam  for lasso, MCP, SCAD   =>  lambda_max, grid unchanged     (test_lambda_max_is_the_same_as_for_the_lasso)
    lam >= lambda_max                     =>  diagonal fit, short-circuit unchanged
    warm starts from the sparse end       # the start decides which local minimum is found; ncvfit's help page says the same
```

---

## 5. Behaviour

### 5.1 Bias

S1b, `simulations/S2_penalties_losses.md` §3.1–3.6:

| | `MCP`, `SCAD` vs `lasso` |
|---|---|
| shrinkage bias | removed |
| skeleton | equal |
| orientation | worse; 3/4 of the directed-metric loss |

$$\dim\ker A(\hat\Sigma)=\tfrac{p(p-1)}{2},\qquad \ker A(\hat\Sigma)\approx\{S:\ S^\top=-S\}\ \ (\hat\Sigma\ \text{a correlation matrix})$$

Example 2, path graph, population $\hat\Sigma$, true edges $0.65$, $\max\lvert\hat M_{ij}-M^\star_{ij}\rvert$, `test_nonconvex_penalties_remove_the_lasso_bias`:

| $\lambda$ | `lasso` | `MCP` | `SCAD` |
|---|---|---|---|
| 0.0365 | 0.448 | $1.2\cdot10^{-12}$ | $1.1\cdot10^{-12}$ |
| 0.0410 | 0.503 | $1.2\cdot10^{-12}$ | $1.2\cdot10^{-12}$ |
| 0.0461 | 0.566 | $1.3\cdot10^{-12}$ | $1.7\cdot10^{-11}$ |

### 5.2 Local minima

| `fista` vs | `--penalty`, `--convention` | $\max\lvert\Delta\hat M\rvert$ | support | both stationary |
|---|---|---|---|---|
| `skglm` | `SCAD`, `textbook`, $w_{ij}\equiv1$ | $1.7\cdot10^{-9}$ | 15/15 | ✓ |
| `ncvreg` | `SCAD`, `ncvreg` | $2.9\cdot10^{-3}$ | 14/15 | ✓ |
| `skglm` | `MCP`, `textbook` | 0.33 | 14/15 | ✓ |
| `ncvreg` | `MCP`, `ncvreg` | 0.11 | 9/15 | ✓ |

| | |
|---|---|
| `fista` restarted at the package solution | $<10^{-11}$ |
| lower objective, `fista` : package : equal | 2 : 5 : 23 |

### 5.3 SCAD edge case

| Example 2, `SCAD`, warm-started path | |
|---|---|
| $\lambda\approx0.17\,\lambda_{\max}$ | local minimum, one weak edge in $\lvert x\rvert\le\lambda$, biased |
| frequency | 1/15 $\lambda$; `MCP` 0/15 |

---

## 6. Validation

| `tests/test_nonconvex.py`, 83 tests | against | tolerance |
|---|---|---|
| `test_values_match_skglm`, `test_scad_value_matches_pyproximal` | formulas, `skglm`, `pyproximal` | $10^{-14}$ |
| `test_penalty_shape`, `test_large_gamma_recovers_the_lasso` | $P(0)=0$, $P'(0^+)=\lambda$, continuous at knots, $P'=0$ for $x>\gamma\lambda$, $\gamma\to\infty\Rightarrow$ `lasso` | exact |
| `test_derivative_matches_finite_differences` | central differences, both conventions | $10^{-5}$ |
| `test_prox_is_the_global_minimiser`; `test_prox_matches_skglm`; `test_scad_prox_matches_pyproximal` | brute force; `skglm` `prox_1d`; `pyproximal` | $2\cdot10^{-6}$ (grid); $10^{-12}$; $10^{-12}$ |
| `test_scaled_prox_is_the_global_minimiser` | brute force | $2\cdot10^{-6}$ |
| `test_what_ncvfit_actually_minimises` | plain regression | $F_{\mathrm{ncvreg}}$ $10^{-10}$; $F_{\mathrm{textbook}}$ $>10^{-2}$ |
| `test_solution_is_stationary`; `test_objective_is_monotone`; `test_huge_gamma_recovers_the_lasso_solution` | both conventions, two $\lambda$; monotone; $\gamma\to\infty$ | $10^{-8}$; monotone; $10^{-6}$ |
| `test_fista_and_skglm_agree_on_textbook_mcp`, `test_fista_matches_skglm_on_textbook_scad`, `test_fista_and_ncvreg_agree_under_the_ncvreg_convention` | stationarity + fixed point | $10^{-8}$ |
| `test_backends_refuse_a_convention_they_do_not_solve`, `test_l1_only_backends_refuse_nonconvex_penalties`, `test_unknown_convention_is_rejected`, `test_nonconvex_penalties_reject_fractional_weights` | | `ValueError` |
| `test_nonconvex_penalties_remove_the_lasso_bias` | Example 2 | $<10^{-6}$; `lasso` $>0.05$ |

---

## 7. Cost

$100$-$\lambda$ path, `tol=1e-8`, one core, Apple M2, seconds:

| $p$ | `lasso` | `MCP` | `SCAD` |
|---|---|---|---|
| 10 | 0.96 | 4.5 | 6.2 |
| 20 | 3.1 | 15.1 | 21.3 |
| ratio | 1 | $\approx5$ | $\approx6.5$ |
| Figure 5 grid, CPU-h | 38.6 | $\approx190$ | $\approx250$ |
| `cluster/s1_array.sbatch` `--penalty`, 64 tasks, hours | — | 3–4 | 3–4 |

---

## 8. Backends

| `--solver` | `lasso` | `MCP` | `SCAD` | `--convention` |
|---|---|---|---|---|
| `fista` (default) | ✓ | ✓ | ✓ | `textbook`, `ncvreg` |
| `ncvreg` | ✓ | ✓ | ✓ | `ncvreg` |
| `skglm` | ✓ | ✓ | ✓, $w_{ij}\equiv1$ | `textbook` |
| `glmnet` | ✓ | — | — | — |
| `pyproximal` | ✓ | — | — | — |
| `design` | ✓ | — | — | — |

| `--loss` | solver | `--convention` |
|---|---|---|
| `direct` | table above | `textbook`, `ncvreg` |
| `loglik`, `frobenius` | active-set Newton, `src/gclm/solvers/covariance.py`, `docs/LIKELIHOOD.md` | `textbook` (no design matrix) |

---

## References

| | | |
|---|---|---|
| Fan & Li 2001 | JASA 96(456), 1348–1360 | SCAD |
| Zhang 2010 | Annals of Statistics 38(2), 894–942 | MCP |
| Breheny & Huang 2011 | Annals of Applied Statistics 5(1), 232–253 | `ncvreg` |
| Li & Lin 2015 | NeurIPS 28 | `_solve_mapg`, Algorithm 1 |
| Loh & Wainwright 2015 | JMLR 16, 559–616 | RSC, local optima |
| Loh & Wainwright 2017 | Annals of Statistics 45(6), 2455–2482 | |
| Beck & Teboulle 2009 | SIAM J. Imaging Sciences 2(1), 183–202 | `docs/FISTA.md` |
| `skglm`; `pyproximal` | JMLR 26 (2025), 24-0008; PyLops | reference implementations |
