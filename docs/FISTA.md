# The FISTA backends: packaged and hand-written

> **`fista` (hand-written) is the default solver.** It is the only backend that reaches $p=50$ in
> practical time, and it is validated in `tests/test_fista.py` (22 tests) against an analytic
> solution, a duality-gap certificate, an interior-point solver, and every other backend.
> This note covers both FISTA backends — `pyproximal` (packaged) and `fista` — and the package
> survey behind the choice.

Implementation: `src/gclm/solvers/proxgrad.py::solve_fista` (hand-written) and
`src/gclm/solvers/backends.py::_pyproximal_path` (packaged).

---

## 1. The algorithm

We minimize $F(M)=f(M)+g(M)$ with

$$f(M)=\tfrac12\big\|M\hat\Sigma+\hat\Sigma M^\top+C\big\|_F^2
\quad\text{(smooth, convex)},\qquad
g(M)=\lambda\sum_{ij}v_{ij}\lvert M_{ij}\rvert\quad\text{(convex, non-smooth)} .$$

FISTA (Beck & Teboulle 2009) is the accelerated proximal-gradient method for exactly this
splitting. One iteration:

$$M_{k+1}=\mathrm{prox}_{t g}\!\big(Y_k-t\,\nabla f(Y_k)\big),\qquad
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},\qquad
Y_{k+1}=M_{k+1}+\frac{t_k-1}{t_{k+1}}\big(M_{k+1}-M_k\big)$$

with, for the weighted $\ell_1$ penalty, the elementwise soft-threshold

$$\big[\mathrm{prox}_{tg}(Z)\big]_{ij}=\mathrm{sign}(Z_{ij})\max\big(\lvert Z_{ij}\rvert-t\lambda v_{ij},\,0\big).$$

FISTA converges as $F(M_k)-F(M^\star)=O(1/k^2)$, against $O(1/k)$ for plain proximal gradient
(ISTA) — Beck & Teboulle 2009, Theorem 4.4.

### The two problem-specific parts

**Gradient.** With $R(M)=M\hat\Sigma+\hat\Sigma M^\top+C$,

$$\nabla f(M)=2\,R(M)\,\hat\Sigma .$$

*Derivation.* $f=\tfrac12\operatorname{tr}(R^2)$ since $R$ is symmetric, so
$df=\operatorname{tr}(R\,dR)=\operatorname{tr}\!\big(R(dM\,\hat\Sigma+\hat\Sigma\,dM^\top)\big)
=2\langle R\hat\Sigma,\,dM\rangle$.
Cost is three $p\times p$ products, i.e. $O(p^3)$ — the design matrix $A(\hat\Sigma)$ is never
formed. Checked against finite differences and against $X^\top(X\beta-y)$ in
`tests/test_loss.py`.

**Step size.** $t=1/L$ with

$$L=\big\|A(\hat\Sigma)\big\|_2^2\le\big(2\lambda_{\max}(\hat\Sigma)\big)^2,$$

since $A=(\Sigma\otimes I)+(I\otimes\Sigma)K$, both summands have spectral norm
$\lambda_{\max}(\Sigma)$, and $K$ is orthogonal. Measured to be **exactly tight** on our problems
(ratio 1.000), so there is nothing to gain from computing the true constant by power iteration.
`test_lipschitz_bound_dominates_true_constant` asserts $L\ge\lambda_{\max}(A^\top A)$ — if it did
not, FISTA would diverge.

### Adaptive restart

Plain FISTA is non-monotone: the momentum term can overshoot and the objective oscillates, which is
severe here because the Hessian is singular (§2.3 of `S1_reproduction.md`). We use the
**gradient scheme** of O'Donoghue & Candès (2015): reset $t_k\leftarrow1$ whenever

$$\big\langle Y_k-M_{k+1},\;M_{k+1}-M_k\big\rangle>0,$$

i.e. when the momentum points against the direction the proximal step just took. This needs no
function evaluations and no knowledge of the strong-convexity parameter (which does not exist here).

Stopping rule: $\lVert M_{k+1}-M_k\rVert_\infty<\texttt{tol}$.

---

## 2. References

| | |
|---|---|
| **Beck, A. & Teboulle, M. (2009).** *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sciences **2**(1), 183–202. [doi:10.1137/080716542](https://doi.org/10.1137/080716542) | FISTA itself: the update above, the $O(1/k^2)$ rate (Thm 4.4), and the constant-step variant we use (§4). |
| **O'Donoghue, B. & Candès, E. (2015).** *Adaptive Restart for Accelerated Gradient Schemes.* Foundations of Computational Mathematics **15**, 715–732. [doi:10.1007/s10208-013-9150-2](https://doi.org/10.1007/s10208-013-9150-2) | The restart condition. Their "gradient scheme", §3.2 — the inner-product test, which is what the code implements. |
| **Nesterov, Y. (1983).** *A method of solving a convex programming problem with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady **27**(2), 372–376. | The original acceleration that FISTA extends to the composite setting. |
| **Parikh, N. & Boyd, S. (2014).** *Proximal Algorithms.* Foundations and Trends in Optimization **1**(3), 127–239. [doi:10.1561/2400000003](https://doi.org/10.1561/2400000003) | Reference for proximal operators generally, incl. the soft-threshold. **Cited by Varando & Hansen for their Algorithm 1**, so the same family the GCLM literature already uses. |
| **Beck, A. & Teboulle, M. (2010).** *Gradient-based algorithms with applications to signal recovery.* In *Convex Optimization in Signal Processing and Communications*, CUP, 42–88. | Backtracking line search; cited by Varando & Hansen for the step-size rule in Algorithm 1. We use the closed-form bound instead, since it is tight. |

### Relation to the GCLM papers

Neither Dettling nor Varando prescribes FISTA for **this** estimator:

- **Dettling**, Appendix A, fits the Direct Lyapunov Lasso with *"the R package `glmnet`, which
  runs a coordinate descent algorithm"*. That is the `glmnet` backend.
- **Varando's Algorithm 1** *is* a proximal-gradient method — same family, citing Parikh & Boyd and
  Beck & Teboulle (2010) — but for a different objective:
  $L(\Sigma(B,C))+\lambda\rho_1(B)+\kappa\lVert C-I_p\rVert_F^2$, optimizing over $B$ **and** $C$,
  solving a Lyapunov equation every iteration and line-searching to keep $B$ stable. That is the
  reference for **S2**, not S1.

So `solve_fista` is a standard method applied to our problem, not a transcription of either paper.

---

## 3. Package survey

Every candidate was installed and run on the real problem; the outcomes are measured, not assumed.

### Packaged FISTA: `pyproximal`

There **is** an off-the-shelf FISTA that fits: [`pyproximal`](https://pyproximal.readthedocs.io)
(with `pylops`). It is available as `solver="pyproximal"`, and — importantly — it runs
**matrix-free**, because `pyproximal.L2(Op=...)` accepts any `pylops` `LinearOperator`. We supply

$$\texttt{matvec}:\ v\mapsto\mathrm{vec}(V\hat\Sigma+\hat\Sigma V^\top),\qquad
\texttt{rmatvec}:\ u\mapsto\mathrm{vec}\big((U+U^\top)\hat\Sigma\big)$$

both $O(p^3)$, so no $p^2\times p^2$ matrix is ever formed. `pyproximal.L1(sigma=ndarray)` takes
per-element weights, so the unpenalized diagonal is expressible. Correctness of the operator (both
directions, against the explicit $A(\hat\Sigma)$) is asserted in
`test_pyproximal_operator_is_the_design_matrix`.

**Measured**, single λ = $0.1\lambda_{\max}$, run to convergence (not to a fixed iteration budget):

| | $p=10$ | $p=20$ |
|---|---|---|
| `fista` (hand-written), time to machine precision | **19.5 ms** | **38 ms** |
| `pyproximal`, time to machine precision | 1007 ms | 1561 ms |
| ratio | ~50× | ~41× |

**Both converge to the same answer.** `pyproximal` reaches `F − F* ≈ −6.7e-16` at $p=10$ and
`1.4e-14` at $p=20$ — the same optimum, and its `tol` does terminate early rather than burning the
whole budget. The difference is **wall-clock, not accuracy**.

> **Correction.** An earlier version of this file reported `pyproximal` as reaching only 2.7e-05.
> That was an artifact of capping `niter` in the benchmark, not a limit of the package: given
> enough iterations it converges to machine precision like any correct FISTA. The speed gap is
> real; the accuracy gap was not.

#### Exactly what differs

Reading `pyproximal.optimization.primal.ProximalGradient`, its `acceleration='fista'` momentum is

```python
told = t; t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0; omega = (told - 1.0) / t
y = x + omega * (x - xold)
```

which is **character-for-character the same recursion as ours**, and we pass it the same
$\tau=1/L$ and the same prox. So the *only algorithmic* difference is the restart branch.

The runtime gap, however, has two sources, and restart is the smaller one:

| source | kind | factor | how measured |
|---|---|---|---|
| **adaptive restart** | algorithmic | ~2–3× (more as $p$ grows) | toggling restart on/off in *identical* code: 63 vs 187 iterations to reach `F−F* < 1e-12` |
| per-iteration overhead | implementation | ~4× | 42 µs/iter vs 10 µs/iter on the same problem |

The overhead is identifiable, not mysterious. When `tol is not None`, `ProximalGradient` evaluates
`proxf(x)` **every iteration** for its stopping test — with `L2(Op=...)` that is an extra operator
application on top of the two the gradient already costs. The rest is `pylops`/`scipy`
`LinearOperator` dispatch and the `vec`/`unvec` reshapes on each application.

Neither source is deep. Restart is a published technique (O'Donoghue & Candès 2015) that
`pyproximal` does not expose — its options are `None`, `'vandenberghe'`, `'fista'`. **If
`pyproximal` added restart and a cheaper stopping test, the gap would largely close.**

### The rest### The rest

| package | verdict |

| package | verdict |
|---|---|
| `sklearn.linear_model.Lasso` | **Cannot express the problem.** No per-feature penalty weights, so the unpenalized diagonal is impossible. The workaround (profile the diagonal out by projection) is correct but converges badly on the rank-deficient design. |
| `copt` | **Unusable here** — segfaults on import in this environment (`mutex lock failed`). Its `minimize_proximal_gradient(accelerated=True)` would otherwise have been a good matrix-free fit. |
| `proxmin` | Installs, but is a general multi-block PGM/APGM toolkit with no weighted-L1 prox out of the box; subsumed by `pyproximal` for our purposes. |
| `celer` | `skglm`'s predecessor; superseded by it. |
| **`skglm`** (JMLR 26, 2025) | **Works** — `WeightedL1`, `WeightedMCPenalty`, `Quadratic` datafit, no intercept, and MCP **without R**. Wired in as `solver="skglm"`. But it operates on the explicit $p^2\times p^2$ design, so it is $O(p^4)$ per sweep against our $O(p^3)$ per iteration. Measured below. `SCAD` has no weighted variant, so the unpenalized diagonal is not expressible there. |
| `ncvreg::ncvfit` (R) | **Works and is the most accurate** — see `R/ENCODING.md`. Also $O(p^4)$; 48× slower than `fista` at $p=20$. |
| `glmnet` (R) | Dettling's own choice. Least accurate, and silently truncates the λ path when it fails to converge. |
| `cvxpy` | General convex modelling; would need the explicit design and is far slower for a 100-point path. |

### Measured, 100-λ path, warm-started, `tol = 1e-12`

| $p$ | backend | time | max distance to exact KKT solution |
|---|---|---|---|
| 10 | `fista` (ours) | **1.78 s** | **8.9e-09** |
| 10 | `skglm` AndersonCD | 2.24 s | 8.9e-07 |
| 20 | `fista` (ours) | **6.84 s** | **1.2e-07** |
| 20 | `skglm` AndersonCD | 30.45 s | 3.1e-06 |

(skglm's own `FISTA` solver, without warm starts, took 42 s at $p=10$.)

## 4. Evidence that the hand-written solver is correct

Because `fista` is the default, it carries more of the burden of proof than a library would.
`tests/test_fista.py` attacks it from five independent directions; nothing below relies on
agreement with another *first-order* method.

### 4.1 An analytic solution — no optimizer involved

With diagonal $\hat\Sigma=\mathrm{diag}(s)$ and $C=\mathrm{diag}(c)$, a diagonal $M$ makes the
residual diagonal with entries $2m_is_i+c_i$, so

$$M^\star=\mathrm{diag}\!\big(-c_i/(2s_i)\big)$$

drives the loss to exactly zero at zero penalty cost — the global minimum, **for every $\lambda$**.
`solve_fista` returns it to **2.4e-14**, at $\lambda=10^{-3}$, $0.5$ and $10$ alike.

### 4.2 A duality-gap certificate — a bound, not a comparison

For $\min\tfrac12\lVert y-X\beta\rVert^2+\lambda\sum_jw_j\lvert\beta_j\rvert$, any residual
rescaled to satisfy $\lvert x_j^\top\theta\rvert\le\lambda w_j$ is dual feasible, and
$D(\theta)=\tfrac12\lVert y\rVert^2-\tfrac12\lVert y-\theta\rVert^2$ lower-bounds the optimum.
The gap $P-D$ therefore **bounds suboptimality from above with no reference solution at all**.
Measured: **1.3e-13, 6.7e-16, -7.4e-15** at $\lambda=0.05,\,0.3,\,2.0$. A second test confirms the
gap widens when `tol` is loosened, so the certificate is actually tracking the solver.

### 4.3 An independent solver class

`cvxpy` with **CLARABEL** is an interior-point conic solver — no shared code path or failure mode
with a first-order proximal method. Agreement to **2.9e-08**, and FISTA lands at the *lower*
objective (0.759801964632 vs 0.759801966624).

### 4.4 Invariances the true minimizer must satisfy

| property | why it must hold | measured |
|---|---|---|
| initialization independence | the problem is convex, so the optimal *value* is unique | spread **0.0** over 5 starts |
| permutation equivariance | relabelling variables may only relabel the answer | **4.4e-16** |
| scale equivariance | $(C,\lambda)\mapsto(aC,a\lambda)$ scales the minimizer by $a$ | 1e-9 |
| $\lambda=0\Rightarrow\nabla f=0$ | nothing left to balance the gradient | < 1e-6 |

### 4.5 The promised convergence rate

Beck & Teboulle guarantee $F(x_k)-F^\star=O(1/k^2)$, so $k^2(F(x_k)-F^\star)$ must stay bounded.
Measured it is *decreasing* — 4.31, 0.56, 1.2e-06 at $k=10,20,50$ — which is the adaptive restart
buying better-than-guaranteed convergence. Broken momentum would show as growth. Machine precision
is reached by $k=200$.

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

The closed-form KKT check is the important one among these: given the active set and signs, the
optimum solves a linear system, so the target involves no optimizer at all. That check is also what
established that `glmnet`, not our code, is the inaccurate one at the dense end of the path.

---

## 5. Why `fista` is the default

**Default: `fista`** — for **speed on the full grid**, not because it is a better or more accurate
algorithm. It is the same algorithm as any FISTA and converges to the same point (§3).

Two distinct reasons, which should not be conflated:

1. **Against `ncvreg` / `skglm` / `glmnet`** the advantage is *structural*: they need the explicit
   $p^2\times p^2$ design, so they are $O(p^4)$ per sweep against the matrix-free $O(p^3)$.
2. **Against `pyproximal`** — which also runs matrix-free here — the advantage is *incidental*:
   ~4× lower per-iteration overhead and ~2–3× fewer iterations from adaptive restart. A package
   with restart would be competitive.

The $O(p^4)$ figures:

| $p$ | `fista` (hand-written) | `ncvreg` | `skglm` |
|---|---|---|---|
| 10 | 1.8 s | 10 s | 2.2 s |
| 20 | 6.8 s | 244 s | 30 s |

At $p=20$ `ncvreg` is already 48× slower, and the gap grows with $p$. **The Figure 5 grid runs to
$p=50$, which no package backend reaches in practical time** — so the default has to be `fista` for
the reproduction to be finishable at all.

That places the burden of proof on §4, which is why the evidence there is deliberately not of the
"agrees with another first-order solver" kind: an analytic solution, a duality-gap certificate and
an interior-point cross-check are independent of FISTA's own failure modes.

The packages remain one flag away and are the right choice when accuracy matters more than speed —
in particular for **S1b's estimation-error comparison at moderate $p$**, where `ncvreg` reaches
4e-12:

```bash
python simulations/run_s1.py                                   # fista (default)
python simulations/run_s1.py --p 10 15 --solver ncvreg         # accuracy-critical runs
python simulations/run_s1.py --penalty MCP                     # S1b (docs/NONCONVEX.md)
```

---
