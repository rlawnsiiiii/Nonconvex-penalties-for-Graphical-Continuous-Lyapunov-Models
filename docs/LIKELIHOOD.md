# Likelihood and Frobenius losses

```
Study S2: lasso | MCP | SCAD under the two losses of Varando & Hansen (2020), Section 3,
evaluated at the implied covariance Sigma(M), not at the Lyapunov residual.

Varando & Hansen (2020), Section 3        R/backend_gclm.R -> gclm::gclm()   reference backend, CRAN package gclm 0.0.1
                                           ^ tests/test_covloss.py (63 tests)
src/gclm/objective/penalties.py    P_lambda: lasso | MCP | SCAD (textbook convention), stationarity
               |
src/gclm/objective/covariance.py   L(Sigma(M, C)) for loss = "loglik" | "frobenius", gradient in M, lambda_max
               |                    (src/gclm/objective/direct.py: loss = "direct", the Lyapunov residual, study S1)
src/gclm/solvers/covariance.py     solve(sigma_hat, C, lam, loss, ..., method = "apg" | "prox" | "newton")
               |
src/gclm/solvers/path.py           covloss_path(direction = "down" | "up", start = "exact" | "lasso"), adaptive_covloss_path
               |                    tests/test_covloss_dense_start.py
simulations/run_s1.py, simulations/run_s1_shard.py, cluster/s1_array.sbatch      (Section 8)
```

---

## 1. Estimator

`varando2020` eq. (7): $C$ fixed, $\kappa=\infty$, $B\to M$.

$$\hat M(\lambda)\in\arg\min_{M\ \text{stable}}\ L\big(\Sigma(M, C)\big)+\sum_{i\neq j}P_\lambda(M_{ij}),\qquad M\Sigma+\Sigma M^\top+C=0$$

| `loss` | $L(\Sigma)$ | $\nabla_\Sigma L$ | convex | $\min L$ | $\arg\min L$ |
|---|---|---|---|---|---|
| `"loglik"` | $\log\det\Sigma+\operatorname{tr}(\Sigma^{-1}\hat\Sigma)$ | $P-P\hat\Sigma P$, $P=\Sigma^{-1}$ | $\times$ | $\log\det\hat\Sigma+p$ | $M_0+W\hat\Sigma^{-1}$, $W^\top=-W$ |
| `"frobenius"` | $\tfrac12\lVert\Sigma-\hat\Sigma\rVert_F^2$ | $\Sigma-\hat\Sigma$ | $\times$ | $0$ | $M_0+W\hat\Sigma^{-1}$, $W^\top=-W$ |
| `"direct"` (S1, `src/gclm/objective/direct.py`) | $\tfrac12\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2$ | | $\checkmark$ | $0$ | $M_0+W\hat\Sigma^{-1}$, $W^\top=-W$ |

| | $C$ | $\hat\Sigma$ | $M_D$ | $P_\lambda$ |
|---|---|---|---|---|
| S1, S2 (`dettling2024`) | $2I$ | $\hat\Sigma_{ii}=1$ | $-I$ | $i\neq j$ only; textbook (`docs/NONCONVEX.md` §2), not `ncvreg` |
| `varando2020` | $I$ | $\hat\Sigma_{ii}=1$ | | $i\neq j$ only |

$$\Sigma(cM,\,cC)=\Sigma(M,\,C)\qquad\text{(fitting with } C=2I \text{ equals fitting with } C=I \text{ in } M'=M/2 \text{, penalty evaluated at } 2M')$$

$$\hat M(\lambda;\,2I)=2\,\hat M(2\lambda;\,I)\qquad\text{(lasso, tested)}$$

$$\texttt{MCP}, \texttt{SCAD}:\ \gamma \text{ on the scale of } M,\quad C\to2C\ \Rightarrow\ \hat M\to2\hat M;\qquad \texttt{direct}:\ L(cM,\,cC)=c^2\,L(M,\,C)$$

## 2. Gradient

$$G=\nabla_\Sigma L\ \text{ at }\ \Sigma=\Sigma(M,C)$$

$$M^\top Z+ZM=G\qquad\text{(adjoint equation)}$$

$$\nabla_M\,L(\Sigma(M,C))=-2\,Z\,\Sigma\qquad\text{(Prop. 3.1)}$$

| source | value |
|---|---|
| `varando2020`, Prop. 3.1 as printed | $2\Sigma\,\Sigma(B^\top,\nabla L)$ |
| `gclm` Fortran `GRAD`: `DSYMM("R", ...)`, `DD = D * S` | $2\,X\Sigma$, $X=-Z$ |
| `test_m_gradient_matches_finite_differences` | $2\cdot10^{-9}$ |
| `gclm` Fortran `DC(J) = 2*TMP(J,J)` | $2\,\nabla_C L$ (unused) |
| `SchurLyapunov` ($M=QTQ^\top$), `dtrsyl` (as `DGELYP`) | $\operatorname{Re}\,\mathrm{eig}(M)=\operatorname{diag}(T)$; $\Sigma(M,C)$, $Z$: one `dtrsyl` each; $O(p^3)$ per iteration |

## 3. Path ends

$$\text{diagonal } M, C:\quad \Sigma_{ii}=-C_{ii}/(2M_{ii}),\qquad M_{D,ii}=-C_{ii}/(2\hat\Sigma_{ii})\qquad\text{(diagonal fit, sparse end: } \Sigma_{ii}=\hat\Sigma_{ii}, \text{ diagonal gradient } 0)$$

$$\lambda_{\max}=\max_{i\neq j}\lvert\nabla_M L(M_D)_{ij}\rvert\qquad(\partial(\lambda\lvert\cdot\rvert)(0)=[-\lambda,\lambda]:\ M_D \text{ stationary iff } \lambda\ge\lambda_{\max};\ \text{same for lasso, MCP, SCAD; other stationary points possible above it, Section 5)}$$

$$\hat\Sigma_{ii}=1,\ C=2I:\quad M_D=-I,\quad \Sigma(M_D)=I,\quad G=I-\hat\Sigma,\quad Z=-(I-\hat\Sigma)/2,\quad \nabla_M L(M_D)=I-\hat\Sigma,\quad \lambda_{\max}=\max_{i\neq j}\lvert\hat\Sigma_{ij}\rvert\quad\text{(as the graphical lasso)}$$

$$\lambda_{\max}=6\ \text{ fixed in varando2020 (standardised data); its smallest } \lambda \text{ with } B \text{ diagonal only for the direct lasso}$$

$$\text{grid: }\ \lambda_{\max}\cdot10^{\mathrm{linspace}(-4,0,100)}\qquad\text{(dettling2024)}$$

$$M_0=-\tfrac12 C\hat\Sigma^{-1},\qquad M_0\hat\Sigma+\hat\Sigma M_0^\top=-C,\qquad \Sigma(M_0)=\hat\Sigma\qquad\text{(dense fit, dense end; } B_0=-\tfrac12\hat R^{-1} \text{ in varando2020)}$$

$$\Sigma(M_0+W\hat\Sigma^{-1})=\hat\Sigma\ \ \forall\,W^\top=-W,\qquad \dim(\text{fibre})=p(p-1)/2=\dim\ker A(\hat\Sigma)\qquad\text{(the design of dettling2024)}$$

$$\lambda\to0:\quad \hat M_{\mathrm{lasso}}(\lambda)\to\tilde M=\arg\min\Big\{\textstyle\sum_{i\neq j}\lvert M_{ij}\rvert:\ \Sigma(M)=\hat\Sigma\Big\},\qquad \lvert\{i\neq j:\tilde M_{ij}=0\}\rvert=p(p-1)/2,\qquad \tilde M=\lim_{\lambda\to0}\hat M^{\mathrm{direct}}_{\mathrm{lasso}}(\lambda)$$

Tests: `test_lambda_max_is_the_largest_correlation`, `tests/test_covloss_dense_start.py`; fibre: `simulations/S1_reproduction.md` §2.3.

### 3.1 Dense starts (wave 7)

$$w_{ij}=\frac{\max_{kl}\lvert\tilde M_{kl}\rvert}{\lvert\tilde M_{ij}\rvert}\ \ (\tilde M_{ij}\neq0),\qquad w_{ij}=10^{12}\ \ (\tilde M_{ij}=0;\ \infty \text{ would give NaN}),\qquad \lambda^w_{\max}=\max_{i\neq j}\frac{\lvert\nabla_M L(M_D)_{ij}\rvert}{w_{ij}}$$

| | `covloss_path(direction="up", start="exact")` | `covloss_path(direction="up", start="lasso")` | `adaptive_covloss_path` |
|---|---|---|---|
| runner | `--direction up` | `--direction up --up-start lasso` | `--loss loglik --method adaptive` |
| start | $M_0$ | $\tilde M$ | $M_D$ |
| order | $\lambda_{\min}\to\lambda_{\max}$ | $\lambda_{\min}\to\lambda_{\max}$ | $\lambda^w_{\max}\to\lambda^w_{\min}$ |
| `lasso` at $\lambda_{\min}$ | $\tilde M$ | $\tilde M$ | |
| `MCP` at $\lambda_{\min}$ | $\approx M_0$ | $\approx\tilde M$ | |
| `MCP` at $\lambda_{\max}$ | $\neq M_D$: $\lvert M_{ij}\rvert>\gamma\lambda$ survive | $\neq M_D$ | |
| `loss="direct"` analogue | `lasso_path(direction="up")`, `docs/DENSE_START.md`: $\to M_D$ | | `adaptive_lasso_path` |

## 4. Solver

### 4.1 Flat valley

| `tol` ($p=10$, `loglik`, `lasso`, 1 dataset, 100 $\lambda$, warm starts, `apg`) | $t\,[\mathrm{s}]$ | $k\,[10^3]$ | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|---|
| $10^{-3}$ | 1.0 | 4.8 | 0.467 | 0.744 | 0.354 |
| $10^{-4}$ | 3.0 | 13.6 | 0.500 | 0.727 | 0.354 |
| $10^{-5}$ | 8.1 | 35 | 0.444 | 0.692 | 0.312 |
| $10^{-6}$ | 20.5 | 92 | 0.400 | 0.699 | 0.314 |
| $10^{-7}$ | 38 | 189 | 0.400 | 0.700 | 0.314 |

$$\texttt{tol}:\ \max\lvert M^+-Y\rvert/t\le\texttt{tol}\qquad\text{(gradient mapping)}$$

$$\text{Algorithm 1 of varando2020 stops at } k=100 \text{ or } \lvert\Delta F/F\rvert<10^{-4}:\ \approx\ \text{row } 10^{-3}$$

$$\texttt{frobenius}, \text{ same dataset: } 55\ \mathrm{s}\ \text{ to } 10^{-6},\qquad \texttt{auc}\ 0.81\to0.71\ \text{ (loose to converged)}$$

$$p=50:\quad \text{cost per iteration } \approx100\times$$

### 4.2 Hessian

$$A=\{(i,j):M_{ij}\neq0\}\cup\{(i,i)\},\qquad J_a=\partial\Sigma/\partial M_a,\qquad MJ_a+J_aM^\top=-(E_a\Sigma+\Sigma E_a^\top)$$

$$H_{ab}=\underbrace{\nabla^2_\Sigma L\,[J_a,J_b]}_{\text{Gauss-Newton; Fisher scoring for loglik}}\;-\;\underbrace{2\,(ZJ_b)_a\;-\;2\,(ZJ_a)_b}_{\text{curvature of } M\mapsto\Sigma(M)}$$

$$\nabla^2_\Sigma L\,[J_a,J_b]=\langle J_a,J_b\rangle\qquad\text{(Frobenius)}$$

$$\nabla^2_\Sigma L\,[J_a,J_b]=-\operatorname{tr}(PJ_aPJ_b)+\operatorname{tr}(PJ_aPJ_bP\hat\Sigma)+\operatorname{tr}(PJ_bPJ_aP\hat\Sigma)\qquad\text{(log-likelihood)}$$

$$\text{Schur basis: rhs}=uv^\top+vu^\top,\ \ u=Q_{i,\cdot},\ v=(\Sigma Q)_{j,\cdot};\qquad \lvert A\rvert\ \texttt{dtrsyl}\text{ calls: } O(\lvert A\rvert p^3);\quad \text{products: } O(\lvert A\rvert^2p^2)$$

$$H\not\succeq0\ \text{ away from a minimiser};\qquad H \text{ singular along the fibre at } M_0$$

`test_hessian_matches_finite_differences`: $2\cdot10^{-9}$.

### 4.3 Solvers: `solve(method=...)`

$$F=L+P,\qquad t\leftarrow t/2\ \text{ until } M^+ \text{ stable and accepted}$$

$$L(M^+)\le L(Y)+\langle\nabla L(Y),\,M^+-Y\rangle+\lVert M^+-Y\rVert^2/(2t)\qquad\text{(acceptance test on loss values, Algorithm 1)}$$

$$\langle\nabla L(M^+)-\nabla L(Y),\,M^+-Y\rangle\le\lVert M^+-Y\rVert^2/t\qquad\text{(the same bound on gradients: secant condition, exact for quadratic } L)$$

$$\mathrm{pred}=\langle\nabla L(Y),M^+-Y\rangle+\lVert M^+-Y\rVert^2/(2t)\approx-t\lVert G\rVert^2/2\ \text{ near a stationary point},\qquad u=10^{-13}\max(1,\lvert L\rvert)\ \text{ (rounding level of } L)$$

```
solve(sigma_hat, C, lam, loss, method, tol):      # every method: descent on F = L + P, every iterate stable
    trial step t: Barzilai-Borwein (SpaRSA, wright2009), not a reset to t = 1 each iteration
                  (this makes the Frobenius loss, curvature far from 1, workable)
                  capped below gamma (MCP) / gamma - 1 (SCAD): proximal problem strictly convex,
                  closed-form prox of docs/NONCONVEX.md Section 3 exact
    halve t until M+ is stable and the step test (table below) holds
    prox   : Algorithm 1 of varando2020 with the BB trial step                 # the reference; slow
    apg    : prox + Nesterov momentum, adaptive restart: a momentum step is kept only if it
             lowers F below the current iterate (Beck & Teboulle, MFISTA rule), else a plain
             step and the momentum is reset (O'Donoghue & Candes 2015)          # the estimator's solver
    newton : active-set Newton in the manner of FPC_AS (wen2010)                # experimental, used by no simulation
        repeat until stationarity(M) <= tol:
            [identify]  proximal-gradient steps until the support is unchanged for two steps
                        and the gradient mapping is below `newton_after`
            [converge]  on A = {M_ij != 0} ∪ diagonal:
                        g_A = grad_A L + P'(M_A),  H = Hessian_A + diag(P''(M_A))
                        shift H by -lambda_min(H) + margin so it is positive definite
                        d = -H^{-1} g_A;  backtrack alpha from 1 with the Armijo test on F;
                        an entry whose sign would flip is set to zero instead (orthant projection, andrew2007)
                        ... until g_A is below 0.1 tol, the support changes, or no step is accepted
# tried at p = 10 and rejected:
#   L-BFGS-B on the split M = M+ - M-     fast, but stalls with stationarity ~ 1e-2
#   OWL-QN (andrew2007)                   millions of function evaluations, repeated line-search failures
#   a nonmonotone BB variant              no faster than the monotone one
```

| step test | before 2026-10-13 | since 2026-10-13 (code review A1) |
|---|---|---|
| accept | $L(M^+)-L(Y)\le\mathrm{pred}+u$ | $\lvert\mathrm{pred}\rvert\ge100u$: loss test; else gradient test |
| `kkt(m_init)` $\le$ `tol` | 1 step at $t=1$; `kkt` ratio $\le3000$ | $M=$ `m_init` |
| $\max_iC_{ii}/\min_iC_{ii}\ge100$ (rescaled $C$) | $255\text{ to }970\ /\ 40000$ solves at $k_{\max}=50000$ | |
| `converged` | $\max\lvert M^+-Y\rvert/t\le$ `tol` | `kkt` $\le$ `tol` |

| $p=10$, rescaled $C$, ratio 250, $n=10^4$, `tol` $=10^{-8}$; old → new | `lasso` | `MCP`, `direction="up"`, `start="exact"` |
|---|---|---|
| support changes, $\lambda$'s | 0 / 100 | 8 / 100, by 1–3 entries |
| oracle $F_1$ | | 0.493 → 0.507 |
| `converged=False` | 1 → 0 | 21 → 33 |
| `kkt` | $\le9\times10^{-6}$ → $\le10^{-8}$ | → $2\times10^{-8}$ to $7\times10^{-4}$ |
| steps | 939,000 → 604,000 | |

| `MCP`, $k=k_{\max}=50000$ | `kkt` | `newton` from there |
|---|---|---|
| dense end | $7\times10^{-4}$ ($k=500$) to $2\times10^{-8}$ ($k=50000$) | $\lVert\Delta M\rVert_\infty=2\times10^{-7}$ |
| $\lambda\approx10^{-3}\lambda_{\max}$ | $10^{-3}$ to $5\times10^{-5}$, oscillating (`prox` alike) | $\lVert\Delta M\rVert_\infty\in[0.8,1.1]$, $\lvert S\,\triangle\,S'\rvert\in[20,22]$, $\Delta F<0$ (Section 5) |
| `loglik` cells, rescaled $C$ | 0.6–2.4 % of solves; `converged` | |
| waves 3, 7, 8 (`loglik`, before 2026-10-13) | old test; $\lvert\Delta\rvert<\mathrm{s.e.}$ (`next_steps/121026/code_review_121026.md`, A1) | re-run open |

$$\texttt{penalties.stationarity}(M)\le\texttt{tol}:\qquad \lvert\nabla L_{ij}\rvert\le\lambda\ \text{ on the zeros},\qquad \nabla L_{ij}+P'(M_{ij})=0\ \text{ elsewhere}$$

| `method` | `tol` | status |
|---|---|---|
| `"apg"` (default) | $10^{-6}$; campaign $10^{-8}$ | estimator's solver |
| `"prox"` | $10^{-6}$ | reference |
| `"newton"` | $10^{-8}$ in $k\sim10^2$; $3$ to $150\times$ faster | experimental, unused |

### 4.4 Cost along the path

| $\lambda/\lambda_{\max}$ (blocks of 10; $p=10$, `apg`) | `nnz`, `loglik`, `lasso` | $k\,[10^3]$ | `nnz`, `frobenius`, `lasso` | $k\,[10^3]$ |
|---|---|---|---|---|
| $10^{-4}$ to $2\cdot10^{-4}$ | 46–47 | 10 | 44–46 | 43 |
| $6\cdot10^{-4}$ to $2\cdot10^{-3}$ | 42–45 | 12 | 34–39 | 42 |
| $4\cdot10^{-3}$ to $10^{-2}$ | 33–40 | 20 | 32–35 | 62 |
| $10^{-2}$ to $2\cdot10^{-2}$ | 23–31 | 13 | 25–31 | 40 |
| $3\cdot10^{-2}$ to $6\cdot10^{-2}$ | 13–23 | 7 | 15–23 | 10 |
| $0.4$ to $1$ | 0–5 | 0.3 | 0–12 | 0.8 |

$$\max\texttt{nnz}=p(p+1)/2-p=45\quad\text{(as the direct lasso: fibre of dimension } p(p-1)/2),\qquad k_{\mathrm{lasso}}/k_{\mathrm{MCP}}\in[5,10]\ \text{ mid-path}$$

## 5. Stationary points

$$\texttt{prox} \text{ vs gclm Algorithm 1, same start, loglik}, \lambda=0.1\lambda_{\max}:\qquad \lvert\Delta F\rvert=1.5\cdot10^{-4},\qquad \lVert B-B'\rVert_\infty=1.5$$

$$k_{\mathrm{apg}}\approx k_{\mathrm{prox}}/(5\text{ to }50)$$

| `max_f1` / `auc` / `aupr`, `loglik`, $p=10$ | `apg` | `prox` |
|---|---|---|
| `MCP`, #0 | 0.455 / 0.683 / 0.326 | 0.455 / 0.681 / 0.325 |
| `MCP`, #1 | 0.489 / 0.618 / 0.443 | 0.489 / 0.618 / 0.443 |
| `lasso`, #0 | 0.400 / 0.699 / 0.314 (Section 4.1, `tol` $\le10^{-6}$) | 0.444 / 0.699 / 0.309 |

$$\texttt{dF}=F_{\mathrm{newton}}-F_{\mathrm{prox}}\ \text{ over the } 100\ \lambda\text{'s},\qquad \texttt{supp}=\lvert\{(\lambda,ij):\ [\hat M_{ij}\neq0]\ \text{differs}\}\rvert\ \ (\text{number of } \lambda\text{'s in brackets})$$

| `loss`, `penalty`, # | `newton_after` | $t\,[\mathrm{s}]$ | `max_f1` / `auc` / `aupr` | `dF` $[\min,\max]$ | `supp` ($\lambda$'s) |
|---|---|---|---|---|---|
| `loglik`, `lasso`, #0 | `prox`, `tol` $=10^{-7}$ | 377 | 0.444 / 0.699 / 0.309 | | |
| | $\infty$ | 25 | 0.444 / 0.701 / 0.276 | $[-1.1\cdot10^{-2},\,0]$ | 12 (9) |
| | $10^{-4}$ | 34 | 0.444 / 0.701 / 0.276 | $[-1.1\cdot10^{-2},\,0]$ | 12 (9) |
| | $10^{-5}$ | 50 | 0.444 / 0.700 / 0.276 | $[-1.1\cdot10^{-2},\,0]$ | 8 (8) |
| `loglik`, `MCP`, #0 | `prox`, `tol` $=10^{-7}$ | 392 | 0.455 / 0.681 / 0.325 | | |
| | $\infty$ | 0.6 | 0.286 / 0.471 / 0.253 | $[-5.0\cdot10^{-2},\,+3.8\cdot10^{-3}]$ | 2087 (99) |
| | $10^{-4}$ | 8.7 | 0.286 / 0.467 / 0.253 | $[-5.0\cdot10^{-2},\,+3.8\cdot10^{-3}]$ | 2212 (99) |
| | $10^{-5}$ | 55 | 0.286 / 0.459 / 0.251 | $[-5.0\cdot10^{-2},\,+3.8\cdot10^{-3}]$ | 2226 (99) |
| `loglik`, `MCP`, #1 | `prox`, `tol` $=10^{-7}$ | 2.2 | 0.489 / 0.618 / 0.443 | | |
| | $\infty$ | 0.5 | 0.421 / 0.524 / 0.304 | $[-7.9\cdot10^{-2},\,+1.3\cdot10^{-2}]$ | 3920 (99) |
| | $10^{-5}$ | 1.8 | 0.421 / 0.524 / 0.304 | $[-7.9\cdot10^{-2},\,+1.3\cdot10^{-2}]$ | 3920 (99) |

```python
# the estimator = this path rule, the same algorithm for lasso, MCP and SCAD (a comparison of
# estimators, not of optimisers): lambda_max -> lambda_min from M_D, each lambda warm-started
# from the previous solution, apg to tol = 1e-6 with the stopping rule of Section 4.3
covloss_path(sigma_hat, C, loss, penalty=penalty, direction="down", method="apg", tol=1e-6)
# direction="up" (Varando & Hansen: from the dense fit, reported better for their loosely
# converged algorithm): available, not used
```

## 6. Validation: `gclm`

```r
# R/backend_gclm.R: every lambda from the same start B0
gclm(Sigma, B = B0, C = Cdiag, C0 = Cdiag, loss = loss, eps = eps, alpha = 0.5, maxIter = maxIter,
     lambda = lambdas[i], lambdac = -1, job = 0)
# lambdac = -1: its Fortran never updates C;  eps = 0: runs until no decrease is representable
```

| test | $\lambda/\lambda_{\max}$ | agreement |
|---|---|---|
| `test_gclm_objective_scale_matches` | | $L_{\texttt{gclm}}(B)+\lambda\lVert B_{\mathrm{off}}\rVert_1=F(B)$ to $10^{-9}$ |
| `test_gclm_converged_points_are_fixed_points_of_our_solver` | | `newton` from `gclm`'s stationary points (`kkt`): $\Delta M=0$ |
| `test_same_basin_agreement_with_gclm` | 0.02 | from $M_D$: $\lvert\Delta F\rvert\le10^{-6}$, $\lVert\Delta B\rVert\le10^{-2}$ |
| `gclm` at 33 iterations, `(F+G+H-FNW-GNW-HNW) <= EPS`, `EPS = 0` | 0.5 | $F-F^\star=0.19$ |

## 7. Cost

| $p=10$, 100 $\lambda$, warm starts, `apg`, `tol = 1e-6`, Apple M2, 1 core; $t\,[\mathrm{s}]$ | `lasso` | `MCP`, `SCAD` |
|---|---|---|
| `loglik` | 10–50 | 1–10 |
| `frobenius` | $\times\,2$ to $3$ | $\times\,2$ to $3$ |
| $p=20$ | $\approx\times10$ | $\approx\times10$ |
| $p=50$ | $\gg$ | $\gg$ |
| measured | `simulations/S2_penalties_losses.md` §4 | |

$$\text{per iteration } O(p^3),\qquad k \text{ grows with } p$$

## 8. Running

```bash
python simulations/run_s1.py --loss loglik    --penalty lasso --p 10 20 --reps 10
python simulations/run_s1.py --loss frobenius --penalty MCP   --p 10 20 --reps 10
python simulations/run_s1_shard.py --loss loglik --penalty SCAD --shard 0 --n-shards 64 \
       --out-dir runs/s2_loglik_scad/s1_shards
sbatch cluster/s1_array.sbatch runs/s2_loglik_scad --loss loglik --penalty SCAD
# --solver      refused for these losses (no effect)
# --direction   up | down, the path order
# shard output per lambda: kkt (first-order violation), proximal and Newton step counts,
#     converged (solver met its tolerance; stored since 2026-10-13), objective
#     -> convergence can be audited after the fact
```

```python
from gclm.objective.covariance import lambda_max
from gclm.solvers.covariance import solve
from gclm.solvers.path import covloss_path
path = covloss_path(sigma_hat, 2 * np.eye(p), "loglik", penalty="MCP")   # 100 lambdas
m_hat, info = solve(sigma_hat, 2 * np.eye(p), lam, "frobenius", return_info=True)
```

## References

```bibtex
@inproceedings{varando2020, author = {Varando, G. and Hansen, N. R.}, year = {2020},
  title = {Graphical continuous Lyapunov models}, booktitle = {UAI 2020}, series = {PMLR}, volume = {124},
  note = {eq. (7), Prop. 3.1, Algorithm 1; R package gclm 0.0.1 (CRAN)}}
@inproceedings{dettling2024, author = {Dettling, P. and Drton, M. and Kolar, M.}, year = {2024},
  title = {On the Lasso for graphical continuous Lyapunov models}, booktitle = {CLeaR 2024}, series = {PMLR}, volume = {236},
  note = {the direct loss and the simulation design}}
@article{wen2010, author = {Wen, Z. and Yin, W. and Goldfarb, D. and Zhang, Y.}, year = {2010},
  title = {A fast algorithm for sparse reconstruction based on shrinkage, subspace optimization, and continuation},
  journal = {SIAM J. Sci. Comput.}, volume = {32}, number = {4}, pages = {1832--1857},
  note = {FPC_AS: the identify/converge structure}}
@article{wright2009, author = {Wright, S. J. and Nowak, R. D. and Figueiredo, M. A. T.}, year = {2009},
  title = {Sparse reconstruction by separable approximation},
  journal = {IEEE Trans. Signal Process.}, volume = {57}, number = {7}, pages = {2479--2493},
  note = {SpaRSA: Barzilai-Borwein steps with a backtracking acceptance test for f + lambda ||.||_1}}
@inproceedings{andrew2007, author = {Andrew, G. and Gao, J.}, year = {2007},
  title = {Scalable training of $L_1$-regularized log-linear models}, booktitle = {ICML},
  note = {the orthant projection used in the Newton line search}}
@article{bartels1972, author = {Bartels, R. H. and Stewart, G. W.}, year = {1972},
  title = {Solution of the matrix equation AX + XB = C}, journal = {Comm. ACM}, volume = {15}, number = {9}, pages = {820--826},
  note = {the Schur-based Lyapunov solver (LAPACK dtrsyl)}}
```
