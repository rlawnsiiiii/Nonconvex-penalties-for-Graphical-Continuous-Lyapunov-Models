# Likelihood and Frobenius losses on the implied covariance

*Varando & Hansen (2020), "Graphical continuous Lyapunov models", Section 3 — implemented in
`src/gclm/objective/covariance.py` (losses) and `src/gclm/solvers/covariance.py` (solvers),
validated in `tests/test_covloss.py` (63 tests), reference backend
`R/backend_gclm.R` (the authors' CRAN package `gclm`).*

**In one paragraph.** Study S2 attaches the lasso, MCP and SCAD to the two losses Varando &
Hansen fit instead of Dettling's direct loss: the Gaussian negative log-likelihood and the squared
Frobenius distance, both evaluated at the covariance *implied* by $M$, $\Sigma(M)$, rather than at
the Lyapunov residual. Both are nonconvex in $M$, and both are nearly flat along $p(p-1)/2$
directions that leave $\Sigma(M)$ unchanged. Two consequences drove the implementation: (i) the
proximal-gradient method of the paper converges so slowly that the published results depend on
its iteration cap, and (ii) the problems have several stationary points per $\lambda$, so the
*estimator* is only defined together with its path-following algorithm. The estimator's solver is
an accelerated proximal-gradient method (Section 4.3), run as a continuation from $\lambda_{\max}$
with warm starts (Section 5). An active-set Newton method with the exact Hessian was built along
the way and is kept in the code **for the experiment of Section 5 only**: it converges faster,
but to the wrong stationary points.

---

## 1. The estimator

Varando & Hansen, eq. (7), with $C$ held fixed ($\kappa = \infty$ in their notation) and our $M$
for their $B$:

$$\hat M(\lambda)\in\arg\min_{M\ \text{stable}}\ L\big(\Sigma(M, C)\big)+\sum_{i\neq j}P_\lambda(M_{ij}),
\qquad M\Sigma+\Sigma M^\top+C=0,$$

| `loss` | $L(\Sigma)$ | $\nabla_\Sigma L$ |
|---|---|---|
| `"loglik"` | $\log\det\Sigma+\operatorname{tr}(\Sigma^{-1}\hat\Sigma)$ | $P-P\hat\Sigma P$, $P=\Sigma^{-1}$ |
| `"frobenius"` | $\tfrac12\|\Sigma-\hat\Sigma\|_F^2$ | $\Sigma-\hat\Sigma$ |

The names and scales are those of the R package `gclm` (`loss = "loglik"` / `"frobenius"`), so
$\lambda$ passes to it unchanged. The log-likelihood omits $n/2$ and constants. The paper writes
the Frobenius loss without the $\tfrac12$; the package has it, and so do we.

**Direct vs. implied.** Dettling's loss $\tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2$
(`src/gclm/objective/direct.py`, study S1) is also a Frobenius norm — of the Lyapunov *residual* at the sample
covariance. It is a convex quadratic in $M$. The two losses above are compositions with the
rational map $M\mapsto\Sigma(M, C)$ and are nonconvex; they also need $M$ stable, since
$\Sigma(M,C)$ is only defined (and positive definite) there. Varando calls Dettling's estimator the
"direct lasso path" (his Section 3.2); we use `loss="direct"` for it throughout the drivers.

**Penalty.** Off-diagonal entries only, as in both papers; $P_\lambda$ is the lasso, MCP or SCAD
of `src/gclm/objective/penalties.py` under the textbook convention. There is no design matrix here, so the
ncvreg convention of docs/NONCONVEX.md §2 has no meaning.

**The scale of $C$.** Since $\Sigma(cM, cC)=\Sigma(M, C)$, fitting with $C=2I$ is the same
problem as fitting with $C=I$ in the variable $M'=M/2$, with the penalty evaluated at $2M'$. For
the lasso that is a rescaling of $\lambda$ — $\hat M(\lambda;\,2I)=2\hat M(2\lambda;\,I)$, tested
— so the support path is the same and the choice is immaterial. For MCP and SCAD it is not:
$\gamma$ is measured on the scale of $M$, and doubling $C$ doubles the coefficients. We keep
Dettling's $C=2I$ for every loss, so that S1 and S2 compare penalties on identical data and an
identical model; Varando used $C=I$ on correlation matrices. (For the direct loss the same
scaling leaves even $\gamma$ untouched, because the quadratic loss scales with $c^2$.)

**Standardisation.** As in S1, the input is the empirical correlation matrix (Varando: "Data was
standardized"). With $\hat\Sigma_{ii}=1$ the diagonal fit of Section 3 is $M_D=-I$ for $C=2I$.

## 2. Gradient

Prop. 3.1 of the paper, derived by differentiating the Lyapunov equation. Writing
$G=\nabla_\Sigma L$ at $\Sigma=\Sigma(M,C)$ and $Z$ for the solution of the *adjoint* equation
$M^\top Z+ZM=G$,

$$\nabla_M\,L(\Sigma(M,C))=-2\,Z\,\Sigma .$$

Two notes on the paper's statement. It prints the product as $2\Sigma\,\Sigma(B^\top,\nabla L)$,
i.e. with the factors in the other order; the package's Fortran (`GRAD`: `DSYMM("R", ...)`,
`DD = D * S`) computes $2\,X\Sigma$ with $X=-Z$, which is what the finite-difference check
confirms (`test_m_gradient_matches_finite_differences`, relative error $2\cdot10^{-9}$). And the
package's gradient in $C$ carries an extra factor 2 relative to the derivative (`DC(J) =
2*TMP(J,J)`); it only affects step lengths in the $C$-update, which we do not use.

One real Schur factorisation $M=QTQ^\top$ per iterate serves three purposes — the stability check
(the diagonal of the standardised quasi-triangular $T$ holds the real parts of the eigenvalues),
$\Sigma(M,C)$, and the adjoint solve — each Lyapunov solve being one LAPACK `dtrsyl` call. That is
the same trick as in `gclm`'s `DGELYP`, and it is what makes an iteration $O(p^3)$
(`SchurLyapunov`).

## 3. The two ends of the path

- **Diagonal fit** (sparse end). For diagonal $M$ and $C$, $\Sigma_{ii}=-C_{ii}/(2M_{ii})$ and
  both losses separate over $i$, minimised at $\Sigma_{ii}=\hat\Sigma_{ii}$:
  $M_{D,ii}=-C_{ii}/(2\hat\Sigma_{ii})$. The diagonal part of the gradient vanishes there.
- **$\lambda_{\max}$** $=\max_{i\neq j}\lvert\nabla_M L(M_D)_{ij}\rvert$: the smallest $\lambda$
  at which $M_D$ is stationary. The paper gives no $\lambda_{\max}$ for these losses (it fixes
  $\lambda_{\max}=6$ on standardised data; only for the direct lasso does it use "the smallest
  $\lambda$ such that $B$ is diagonal"); this is that definition carried over. The derivation: the
  diagonal part of the gradient vanishes at $M_D$, and the subdifferential of $\lambda|\cdot|$ at
  zero is $[-\lambda,\lambda]$, so $M_D$ satisfies the first-order condition iff
  $|\nabla_M L(M_D)_{ij}|\le\lambda$ off the diagonal. The same for all three penalties. For
  correlation input it has a closed form: $\Sigma(M_D)=I$, both $\Sigma$-gradients reduce to
  $I-\hat\Sigma$, the adjoint solution with $M_D=-I$ is $Z=-(I-\hat\Sigma)/2$, and
  $\nabla_M=-2Z\Sigma=\hat\Sigma-I$, hence

  $$\lambda_{\max}=\max_{i\neq j}\lvert\hat\Sigma_{ij}\rvert\qquad(\hat\Sigma\text{ a correlation matrix, }C=2I),$$

  the largest absolute correlation — the same $\lambda_{\max}$ Varando uses for the graphical lasso
  (`test_lambda_max_is_the_largest_correlation`). We use Dettling's grid
  $\lambda_{\max}\cdot10^{\text{linspace}(-4,0,100)}$ below it. With a nonconvex objective this is
  the smallest $\lambda$ for which the *diagonal* solution is stationary; non-diagonal stationary
  points may exist above it (Section 5).
- **Dense fit** (dense end). $M_0=-\tfrac12 C\hat\Sigma^{-1}$ gives
  $M_0\hat\Sigma+\hat\Sigma M_0^\top=-C$ exactly, so $\Sigma(M_0)=\hat\Sigma$ and both losses are
  at their global minimum ($\log\det\hat\Sigma+p$, resp. $0$). This is Varando's starting point
  $B_0=-\tfrac12\hat R^{-1}$. It is not unique: $M_0+W\hat\Sigma^{-1}$ for any skew-symmetric $W$
  gives the same $\Sigma$ — the fibre of the map $M\mapsto\Sigma(M)$ has dimension $p(p-1)/2$,
  exactly as the null space of Dettling's design $A(\hat\Sigma)$ (S1_reproduction.md §2.3).

## 4. Why a first-order method is not enough, and the solver

### 4.1 The flat valley

Along the fibre directions $W\Sigma^{-1}$ the loss does not change at all at the unpenalised
optimum, and changes only slightly nearby. The penalty is the only thing that selects a point along
them, and for small $\lambda$ it is weak. A proximal-gradient method therefore crawls: the gradient
mapping is tiny, every step is accepted, and the iterate moves a long way at a slow rate.

Measured at $p=10$ (loglik, lasso, one dataset, path of 100 $\lambda$'s with warm starts,
accelerated proximal gradient with backtracking; `tol` is the gradient-mapping threshold):

| `tol` | path time | iterations | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|---|
| $10^{-3}$ | 1.0 s | 4.8k | 0.467 | 0.744 | 0.354 |
| $10^{-4}$ | 3.0 s | 13.6k | 0.500 | 0.727 | 0.354 |
| $10^{-5}$ | 8.1 s | 35k | 0.444 | 0.692 | 0.312 |
| $10^{-6}$ | 20.5 s | 92k | 0.400 | 0.699 | 0.314 |
| $10^{-7}$ | 38 s | 189k | 0.400 | 0.700 | 0.314 |

The support metrics keep changing until about $10^{-6}$, and under-converged solutions look
*better* (they stay closer to the sparse warm start). Varando & Hansen's Algorithm 1 stops after
$M=100$ iterations or a relative objective decrease below $10^{-4}$ — the first row of this table,
roughly. The Frobenius loss is worse: 55 s to $10^{-6}$ for the same dataset, and its `auc` moves
from 0.81 to 0.71 between the loose and the converged solution. At $p=50$ the cost per iteration
is ~100× higher. A converged first-order path for the full S1 grid is out of reach, and a loosely
converged one measures the optimiser as much as the estimator.

### 4.2 The Hessian

On an index set $A$ of entries of $M$ (the active set), with $J_a=\partial\Sigma/\partial M_a$
the solution of $MJ+JM^\top=-(E_a\Sigma+\Sigma E_a^\top)$ and $Z$ the adjoint solution of the
gradient,

$$H_{ab}=\nabla^2_\Sigma L\,[J_a,J_b]\;-\;2\,(ZJ_b)_a\;-\;2\,(ZJ_a)_b .$$

The first term is the Gauss–Newton (for the log-likelihood: Fisher-scoring) part — $\langle J_a,
J_b\rangle$ for the Frobenius loss, $-\operatorname{tr}(PJ_aPJ_b)+\operatorname{tr}(PJ_aPJ_bP\hat\Sigma)
+\operatorname{tr}(PJ_bPJ_aP\hat\Sigma)$ for the log-likelihood; the second is the curvature of
$M\mapsto\Sigma(M)$ itself. Everything is computed in the Schur basis, where the right-hand side
of each $J_a$-equation is the rank-2 matrix $uv^\top+vu^\top$ with $u=Q_{i,\cdot}$,
$v=(\Sigma Q)_{j,\cdot}$: $|A|$ `dtrsyl` calls, $O(|A|p^3)$, then matrix products,
$O(|A|^2p^2)$. Checked against finite differences of the gradient to $2\cdot10^{-9}$
(`test_hessian_matches_finite_differences`). The Hessian is indefinite away from a minimiser —
the losses are nonconvex — and singular along the fibre at the dense end.

### 4.3 The solvers (`solve(method=...)`)

All three are descent methods on $F=L+P$ — the objective never rises, every iterate is stable —
and all three use the same backtracking acceptance test as Algorithm 1: a trial step $t$ is halved
until $M^+$ is stable and $L(M^+)\le L(Y)+\langle\nabla L(Y),M^+-Y\rangle+\|M^+-Y\|^2/(2t)$. The
trial step comes from Barzilai–Borwein (SpaRSA, Wright, Nowak & Figueiredo 2009) rather than
being reset to $t=1$ every iteration, which is what makes the Frobenius loss, whose curvature is
far from 1, workable. For MCP/SCAD the step is capped below $\gamma$ (resp. $\gamma-1$), where
the proximal problem is strictly convex and the closed-form prox of docs/NONCONVEX.md §3 exact.

**`"apg"` — accelerated proximal gradient with adaptive restart (the default).** Nesterov
momentum on top of the proximal step; a momentum step is accepted only if it lowers $F$ below the
current iterate (Beck & Teboulle's MFISTA rule), otherwise a plain step is taken and the momentum
reset (O'Donoghue & Candès 2015). It stops when the gradient mapping $\max|M^+-Y|/t$ is below
`tol` $=10^{-6}$, the level at which the support metrics of Section 4.1 have stabilised. It
converges to the same stationary points as the plain method (Section 5) in a fraction of the
iterations.

**`"prox"` — the plain method**, Algorithm 1 with BB trial steps. Kept as the reference the
others are compared with; slow.

**`"newton"` — active-set Newton, experimental; not used by any simulation.** In the manner of
FPC_AS (Wen, Yin, Goldfarb & Zhang 2010):

```
repeat until stationarity(M) <= tol:
    [identify]  proximal-gradient steps until the support is unchanged for two steps
                and the gradient mapping is below `newton_after`
    [converge]  on A = {M_ij != 0} ∪ diagonal:
                g_A = grad_A L + P'(M_A),  H = Hessian_A + diag(P''(M_A))
                shift H by -lambda_min(H) + margin so it is positive definite
                d = -H^{-1} g_A;  backtrack alpha from 1 with the Armijo test on F;
                an entry whose sign would flip is set to zero instead (orthant projection)
                ... until g_A is below 0.1 tol, the support changes, or no step is accepted
```

It reaches `penalties.stationarity` $\le10^{-8}$ ($|\nabla L_{ij}|\le\lambda$ on the zeros,
$\nabla L_{ij}+P'(M_{ij})=0$ elsewhere) in a few hundred steps per path, 3–150× faster than the
first-order methods. It was written because the first-order methods crawl (Section 4.1), and it
is what revealed the problem of Section 5: it converges to different, worse-recovering stationary
points. It stays in the code so that Section 5 can be reproduced; the estimator never uses it,
and the thesis should describe the `"apg"` path as the method.

Also tried and rejected, at $p=10$: L-BFGS-B on the split $M=M^+-M^-$ (fast, but stalls with
stationarity $\sim10^{-2}$); OWL-QN (Andrew & Gao 2007; millions of function evaluations,
repeated line-search failures); a nonmonotone BB variant (no faster than the monotone one).

### 4.4 What the cost is spent on

$p=10$, one dataset, `apg`, by blocks of ten $\lambda$'s from the dense end (index 0,
$\lambda=10^{-4}\lambda_{\max}$) to the sparse end; `nnz` is the off-diagonal support size:

| $\lambda/\lambda_{\max}$ | nnz (loglik, lasso) | iterations | nnz (Frobenius, lasso) | iterations |
|---|---|---|---|---|
| $10^{-4}$ – $2\cdot10^{-4}$ | 46–47 | 10k | 44–46 | 43k |
| $6\cdot10^{-4}$ – $2\cdot10^{-3}$ | 42–45 | 12k | 34–39 | 42k |
| $4\cdot10^{-3}$ – $10^{-2}$ | 33–40 | 20k | 32–35 | 62k |
| $10^{-2}$ – $2\cdot10^{-2}$ | 23–31 | 13k | 25–31 | 40k |
| $3\cdot10^{-2}$ – $6\cdot10^{-2}$ | 13–23 | 7k | 15–23 | 10k |
| $0.4$ – $1$ | 0–5 | 0.3k | 0–12 | 0.8k |

The path saturates at $p(p+1)/2-p=45$ off-diagonal nonzeros, exactly as the direct lasso does
(S1_reproduction.md §2.3) and for the same reason — the fibre has dimension $p(p-1)/2$. The
expensive part is the middle of the path, where the support is still being decided inside the
flat valley; MCP spends 5–10× fewer iterations there than the lasso, because its saturating
penalty stops pulling on the large entries. Dettling's grid is kept unchanged for comparability
with S1.

## 5. Nonconvexity: which stationary point?

The losses have several stationary points per $\lambda$, and a descent method converges to the one
whose basin it starts in. Measured on $p=10$ datasets, all with the same warm-started path from
$\lambda_{\max}$ downwards:

1. **Independent implementations differ.** From the *same* start, our plain proximal method
   and `gclm`'s Algorithm 1 can converge to different stationary points, one with a lower
   objective than the other (loglik, $\lambda=0.1\lambda_{\max}$: objectives differ by
   $1.5\cdot10^{-4}$, $\|B-B'\|_\infty=1.5$).
2. **Newton finds different, lower, worse points.** The table below compares the Newton path,
   with its Newton phase allowed only once the proximal phase's gradient mapping is below
   `newton_after`, against the plain proximal path converged to $10^{-7}$ (the reference).
   `dF` is the objective difference Newton − reference over the 100 $\lambda$'s, `supp` the
   number of (λ, entry) pairs whose zero/nonzero status differs.

   | loss, penalty, dataset | `newton_after` | time | `max_f1` / `auc` / `aupr` | `dF` min … max | `supp` (λ's) |
   |---|---|---|---|---|---|
   | loglik, lasso, #0 | reference | 377 s | 0.444 / 0.699 / 0.309 | — | — |
   | | $\infty$ | 25 s | 0.444 / 0.701 / 0.276 | $-1.1\cdot10^{-2}$ … $0$ | 12 (9) |
   | | $10^{-4}$ | 34 s | 0.444 / 0.701 / 0.276 | $-1.1\cdot10^{-2}$ … $0$ | 12 (9) |
   | | $10^{-5}$ | 50 s | 0.444 / 0.700 / 0.276 | $-1.1\cdot10^{-2}$ … $0$ | 8 (8) |
   | loglik, MCP, #0 | reference | 392 s | 0.455 / 0.681 / 0.325 | — | — |
   | | $\infty$ | 0.6 s | 0.286 / 0.471 / 0.253 | $-5.0\cdot10^{-2}$ … $+3.8\cdot10^{-3}$ | 2087 (99) |
   | | $10^{-4}$ | 8.7 s | 0.286 / 0.467 / 0.253 | $-5.0\cdot10^{-2}$ … $+3.8\cdot10^{-3}$ | 2212 (99) |
   | | $10^{-5}$ | 55 s | 0.286 / 0.459 / 0.251 | $-5.0\cdot10^{-2}$ … $+3.8\cdot10^{-3}$ | 2226 (99) |
   | loglik, MCP, #1 | reference | 2.2 s | 0.489 / 0.618 / 0.443 | — | — |
   | | $\infty$ | 0.5 s | 0.421 / 0.524 / 0.304 | $-7.9\cdot10^{-2}$ … $+1.3\cdot10^{-2}$ | 3920 (99) |
   | | $10^{-5}$ | 1.8 s | 0.421 / 0.524 / 0.304 | $-7.9\cdot10^{-2}$ … $+1.3\cdot10^{-2}$ | 3920 (99) |

   For the lasso the Newton path is close to the reference: a handful of $\lambda$'s at the dense
   end, where the fibre directions lie inside the active set and Newton slides along them to a
   point with a smaller $\ell_1$ norm. For MCP it is a different path at every $\lambda$, however
   late Newton is allowed in: with a concave penalty the proximal phase settles into shallow
   stationary points, Newton (seeing their near-zero or negative curvature) leaves them for
   lower ones, and the lower ones recover the support worse — `auc` 0.47 instead of 0.68, below
   chance. A lower objective is not a better estimate: along the fibre every $M$ has the same
   loss, and the sparsest point on it need not be near the truth.
3. **The first-order methods agree with each other.** `apg` reproduces the plain method's
   metrics (MCP #0: 0.455 / 0.683 / 0.326; MCP #1: 0.489 / 0.618 / 0.443) with 5–50× fewer
   iterations; for the lasso on #0 it lands on 0.400 / 0.699 / 0.314, the same values as the
   independent accelerated implementation of the tolerance study, with the plain BB method at
   0.444 on `max_f1` — a residual solver dependence at a few $\lambda$'s that is small compared
   with the Newton effect but not zero.

So the estimator is only defined together with the path-following rule, and the rule used
throughout is: **continuation from $\lambda_{\max}$ downwards** (`direction="down"`) from the
diagonal fit, each $\lambda$ warm-started from the previous solution, each solved by the
accelerated proximal method to gradient-mapping tolerance $10^{-6}$ — the same algorithm for all
three penalties, so that the comparison between them is a comparison of estimators, not of
optimisers. Varando & Hansen walk the other way (`direction="up"`, from the dense fit) and report
it empirically better for their loosely converged algorithm; it is available but not used.

## 6. Validation against the `gclm` package

`R/backend_gclm.R` calls `gclm::gclm()` with `lambdac = -1` (its Fortran then never updates $C$),
every $\lambda$ from the same start, `eps = 0` (run until no decrease is representable). Tests:

| test | what it establishes |
|---|---|
| `test_gclm_objective_scale_matches` | `gclm`'s reported loss plus $\lambda\|B_{\text{off}}\|_1$ equals our objective at its $B$, to $10^{-9}$ — same losses, same scale |
| `test_gclm_converged_points_are_fixed_points_of_our_solver` | where `gclm` reaches a stationary point (by our KKT measure), our Newton solver started there does not move |
| `test_same_basin_agreement_with_gclm` | at $\lambda=0.02\lambda_{\max}$ from the diagonal fit, both reach the same point: objective to $10^{-6}$, $B$ to $10^{-2}$ (the valley is flat, so $B$ is less precise than $F$) |

What does **not** agree, and why: `gclm` stops whenever an iteration produces no decrease
(`(F+G+H-FNW-GNW-HNW) <= EPS` with `EPS = 0`), which happens as soon as a line search returns the
current point — at $\lambda=\tfrac12\lambda_{\max}$ it stops after 33 iterations with the objective
0.19 above the stationary point. That is a property of its stopping rule, not of the objective; the
tests therefore check fixed points and same-basin agreement rather than equality of outputs.

## 7. Cost

Per 100-$\lambda$ path, warm-started, `apg` at `tol = 1e-6`, Apple M2, one core, $p=10$:
loglik 10–50 s (lasso), 1–10 s (MCP/SCAD); Frobenius 2–3× that. Each iteration is $O(p^3)$ and
the iteration count grows with $p$ as well, so $p=20$ costs roughly 10× and $p=50$ far more —
the full S1 grid is not within reach for these losses. The measured table and the cluster budget
are in simulations/S2_penalties_losses.md §4.

## 8. Running it

```bash
python simulations/run_s1.py --loss loglik    --penalty lasso --p 10 20 --reps 10
python simulations/run_s1.py --loss frobenius --penalty MCP   --p 10 20 --reps 10
python simulations/run_s1_shard.py --loss loglik --penalty SCAD --shard 0 --n-shards 64 \
       --out-dir runs/s2_loglik_scad/s1_shards
sbatch cluster/s1_array.sbatch runs/s2_loglik_scad --loss loglik --penalty SCAD
```

`--solver` is ignored for these losses; `--direction up|down` selects the path order. The shard
output carries, per $\lambda$, the first-order violation (`kkt`), the number of proximal and
Newton steps, and the objective, so convergence can be audited after the fact.

```python
from gclm.objective.covariance import lambda_max
from gclm.solvers.covariance import solve
from gclm.solvers.path import covloss_path
path = covloss_path(sigma_hat, 2 * np.eye(p), "loglik", penalty="MCP")   # 100 lambdas
m_hat, info = solve(sigma_hat, 2 * np.eye(p), lam, "frobenius", return_info=True)
```

## References

- Varando, G. & Hansen, N. R. (2020). *Graphical continuous Lyapunov models.* UAI 2020, PMLR 124.
  — eq. (7), Prop. 3.1, Algorithm 1. R package `gclm` 0.0.1 (CRAN).
- Dettling, P., Drton, M. & Kolar, M. (2024). *On the Lasso for graphical continuous Lyapunov
  models.* CLeaR 2024, PMLR 236. — the direct loss and the simulation design.
- Wen, Z., Yin, W., Goldfarb, D. & Zhang, Y. (2010). *A fast algorithm for sparse reconstruction
  based on shrinkage, subspace optimization, and continuation.* SIAM J. Sci. Comput. 32(4),
  1832–1857. — FPC_AS: the identify/converge structure.
- Wright, S. J., Nowak, R. D. & Figueiredo, M. A. T. (2009). *Sparse reconstruction by separable
  approximation.* IEEE Trans. Signal Process. 57(7), 2479–2493. — SpaRSA: Barzilai–Borwein steps
  with a backtracking acceptance test for $f+\lambda\|\cdot\|_1$.
- Andrew, G. & Gao, J. (2007). *Scalable training of $L_1$-regularized log-linear models.* ICML.
  — the orthant projection used in the Newton line search.
- Bartels, R. H. & Stewart, G. W. (1972). *Solution of the matrix equation AX + XB = C.* Comm.
  ACM 15(9), 820–826. — the Schur-based Lyapunov solver (LAPACK `dtrsyl`).
