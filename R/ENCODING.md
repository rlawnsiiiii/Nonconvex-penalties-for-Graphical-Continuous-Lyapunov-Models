# How our problem is encoded into `glmnet` and `ncvreg`

A line-by-line account of how the Direct Lyapunov Lasso becomes a call to `glmnet()` or
`ncvreg::ncvfit()`, with a fully worked $p=3$ example whose numbers you can check by hand.

Implemented in [`backend_glmnet.R`](backend_glmnet.R) and [`backend_ncvreg.R`](backend_ncvreg.R);
selected from Python with `lasso_path(..., solver="glmnet" | "ncvreg")`.

---

## 1. The problem

$$\hat M(\lambda)\;=\;\arg\min_{M\in\mathbb{R}^{p\times p}}\;
\underbrace{\tfrac12\big\|M\hat\Sigma+\hat\Sigma M^\top+C\big\|_F^2}_{\text{smooth}}
\;+\;\lambda\!\!\sum_{i\neq j}\lvert M_{ij}\rvert$$

Note the two conventions this repo uses throughout, both matching Varando's `lassoB()`:

- the **diagonal of $M$ is not penalized** (it is never zero — $M$ must be stable);
- the loss carries a factor $\tfrac12$ and the Frobenius norm counts off-diagonal residuals twice.

## 2. Turning it into a linear regression

The Lyapunov equation is linear in $M$, so with $\mathrm{vec}$ stacking **columns**:

$$\mathrm{vec}\big(M\hat\Sigma+\hat\Sigma M^\top\big)=A(\hat\Sigma)\,\mathrm{vec}(M),
\qquad A(\Sigma)=(\Sigma\otimes I_p)+(I_p\otimes\Sigma)K_{(p,p)}$$

so the objective is an ordinary lasso in $\beta=\mathrm{vec}(M)$:

$$\tfrac12\lVert y-X\beta\rVert_2^2+\lambda\sum_j v_j\lvert\beta_j\rvert,
\qquad \boxed{X=A(\hat\Sigma)},\quad \boxed{y=-\mathrm{vec}(C)},\quad
\boxed{v=\mathrm{vec}(\mathbf{1}\mathbf{1}^\top-I_p)}$$

**`X` is $p^2\times p^2$, `y` has length $p^2$.** There are $p^2$ "observations" — one per entry of
the residual matrix — and $p^2$ "predictors" — one per entry of $M$.

### R code (identical in both backends)

```r
p  <- nrow(Sigma)
MM <- matrix(nrow = p, ncol = p, 1:(p^2))   # MM[i,k] = (k-1)*p + i, column-major
TT <- diag(p^2)[c(t(MM)), ]                 # commutation matrix K(p,p)
X  <- Sigma %x% diag(p) + ((diag(p) %x% Sigma) %*% TT)
y  <- -c(C)                                 # c() is column-major = vec()
pf <- c(1 - diag(p))                        # penalty.factor: 0 on M's diagonal
```

`c()` in R and `vec` in the paper are the same operation (stack columns), so `c(C)` and `c(1-diag(p))`
need no reordering. `TT` is built by row-reading a column-major fill, which is exactly $K_{(p,p)}$.

### Index map — which coefficient is which entry of `M`

Column $j$ of `X` (1-based) is the coefficient of $M_{ik}$ with

$$j=(k-1)p+i,\qquad\text{i.e.}\qquad i=((j-1)\bmod p)+1,\quad k=\big\lfloor (j-1)/p\big\rfloor+1 .$$

To recover the matrix: `M <- matrix(beta, p, p)` — R fills column-major, so this is exactly
$\mathrm{vec}^{-1}$. In Python, `unvec(beta, p)` = `beta.reshape((p, p), order="F")`.

---

## 3. Worked example, $p=3$

$$\hat\Sigma=\begin{pmatrix}2&0.5&0\\0.5&3&1\\0&1&4\end{pmatrix},\qquad C=2I_3$$

**Response and penalty factors** (`vec(C)` picks out the diagonal of $C$ at positions 1, 5, 9):

```
y  = -2  0  0  0 -2  0  0  0 -2
pf =  0  1  1  1  0  1  1  1  0        # zeros at j = 1, 5, 9  <->  M[1,1], M[2,2], M[3,3]
```

**Design matrix** `X = A(Sigma)`, $9\times 9$:

```
        M11  M21  M31  M12  M22  M32  M13  M23  M33
 R11 [  4.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0 ]
 R21 [  0.5  2.0  0.0  3.0  0.5  0.0  1.0  0.0  0.0 ]
 R31 [  0.0  0.0  2.0  1.0  0.0  0.5  4.0  0.0  0.0 ]
 R12 [  0.5  2.0  0.0  3.0  0.5  0.0  1.0  0.0  0.0 ]
 R22 [  0.0  1.0  0.0  0.0  6.0  0.0  0.0  2.0  0.0 ]
 R32 [  0.0  0.0  0.5  0.0  1.0  3.0  0.0  4.0  1.0 ]
 R13 [  0.0  0.0  2.0  1.0  0.0  0.5  4.0  0.0  0.0 ]
 R23 [  0.0  0.0  0.5  0.0  1.0  3.0  0.0  4.0  1.0 ]
 R33 [  0.0  0.0  0.0  0.0  0.0  2.0  0.0  0.0  8.0 ]
```

Three things to check by eye:

1. **Row `R11`.** The $(1,1)$ residual is $R_{11}=2(M\hat\Sigma)_{11}+C_{11}
   =2(M_{11}\Sigma_{11}+M_{12}\Sigma_{21}+M_{13}\Sigma_{31})+C_{11}$. With
   $\Sigma_{11}=2,\Sigma_{21}=0.5,\Sigma_{31}=0$ that is $4M_{11}+1\cdot M_{12}+0$ — the row reads
   `4.0` under `M11` and `1.0` under `M12`, everything else zero. ✓
2. **Row `R33`.** $2(M_{31}\Sigma_{13}+M_{32}\Sigma_{23}+M_{33}\Sigma_{33})
   =0+2M_{32}+8M_{33}$. ✓
3. **Duplicate rows.** `R21 == R12`, `R31 == R13`, `R32 == R23`. The residual is symmetric, so each
   off-diagonal equation appears twice. **This is intentional** — it is what makes
   $\lVert y-X\beta\rVert_2^2$ equal $\lVert M\hat\Sigma+\hat\Sigma M^\top+C\rVert_F^2$, which also
   counts off-diagonal entries twice. Do not deduplicate.

   It also means $X$ has rank $p(p+1)/2=6$, not 9 — see `S1_reproduction.md` §2.3.

### The two calls, at $\lambda_{\text{paper}}=0.7$

```r
## glmnet --------------------------------------------------------------------
lam_g <- lambda_paper * (p - 1) / p^3                    # = 0.0518519
fit_g <- glmnet(X, y,
                intercept      = FALSE,   # our model has no constant term
                standardize    = FALSE,   # columns of A(Sigma) are already meaningful
                lambda         = lam_g,
                thresh         = 1e-14,   # glmnet >= 5.0: control = list(thresh = 1e-14)
                penalty.factor = pf)
Mg <- matrix(as.numeric(fit_g$beta), p, p)

## ncvreg --------------------------------------------------------------------
lam_n <- lambda_paper / p^2                              # = 0.0777778
fit_n <- ncvfit(X, y,
                xtx            = apply(X, 2, crossprod) / nrow(X),
                penalty        = "lasso",                # or "MCP" / "SCAD"
                lambda         = lam_n,
                penalty.factor = pf,
                eps = 1e-12, max.iter = 1e5)
Mn <- matrix(fit_n$beta, p, p)
```

Both return, to every printed digit:

```
          [,1]      [,2]      [,3]
[1,] -0.514089  0.095056  0.000000
[2,]  0.000000 -0.365853  0.127540
[3,]  0.000000  0.000000 -0.246797

objective  glmnet: 0.20219587245   ncvfit: 0.20219587245
```

where the objective is evaluated *directly on the matrices*, independently of either package:

```r
obj <- function(M) 0.5*sum((M %*% Sigma + Sigma %*% t(M) + C)^2) +
                   lambda_paper*sum(abs(M[row(M) != col(M)]))
```

The diagonal is negative (as a stable drift matrix requires) and unpenalized; the off-diagonal is
sparse. The full script is reproduced in §6.

---

## 4. The λ conversions — where the constants come from

Neither package minimizes our objective as written. Both use a $1/(2n)$ factor with
$n=\texttt{nrow(X)}=p^2$, and they differ in how they treat `penalty.factor`.

### `glmnet`: $\lambda_{\text{paper}} = \lambda_g\cdot p^3/(p-1)$

glmnet minimizes

$$\frac{1}{2n}\lVert y-X\beta\rVert^2+\lambda_g\sum_j \tilde v_j\lvert\beta_j\rvert ,\qquad n=p^2 .$$

Two adjustments:

1. Multiply through by $n=p^2$ to match our $\tfrac12\lVert\cdot\rVert^2$: contributes a factor $p^2$.
2. **glmnet rescales `penalty.factor` so that it sums to `nvars`.** Our `pf` has $p^2-p$ ones and
   $p$ zeros, summing to $p^2-p$; rescaled to sum to $p^2$, each nonzero entry becomes
   $\tilde v_j=\dfrac{p^2}{p^2-p}=\dfrac{p}{p-1}$.

Together: $\lambda_{\text{paper}}=\lambda_g\cdot p^2\cdot\frac{p}{p-1}=\lambda_g\,p^3/(p-1)$.
At $p=3$: $0.7 = \lambda_g\cdot 27/2 \Rightarrow \lambda_g=0.0518519$. ✓

### `ncvfit`: $\lambda_{\text{paper}} = \lambda_n\cdot p^2$

`ncvfit` uses the same $1/(2n)$ factor but **does not rescale `penalty.factor`**, so only
adjustment 1 applies: $\lambda_{\text{paper}}=\lambda_n\,p^2$. At $p=3$:
$0.7=\lambda_n\cdot 9\Rightarrow\lambda_n=0.0777778$. ✓

### Confirmed against the packages' own documentation

Both rules were first established empirically (`test_glmnet_lambda_scaling`,
`test_fista_matches_ncvreg`) and then checked against the installed help pages
(`tools::Rd_db`), quoted verbatim below.

**`glmnet`**, `?glmnet`, argument `penalty.factor` — the rescaling is stated outright:

> "Separate penalty factors can be applied to each coefficient. This is a number that multiplies
> `lambda` to allow differential shrinkage. Can be 0 for some variables, which implies no shrinkage,
> and that variable is always included in the model. […]
> **Note: the penalty factors are internally rescaled to sum to nvars, and the lambda sequence will
> reflect this change.**"

**`ncvreg`**, `?ncvfit`, argument `penalty.factor`:

> "A multiplicative factor for the penalty applied to each coefficient. […] In particular,
> `penalty.factor` can be 0, in which case the coefficient is always in the model without
> shrinkage."

No rescaling is mentioned — and the string *"rescale"* does not occur anywhere on the `ncvreg` or
`ncvfit` help pages. Absence from the docs is not proof on its own, which is why the empirical test
is the primary evidence; the documentation corroborates it.

> The worked example is the eyeball version of the same check: the two λ values differ by exactly
> $p/(p-1)=1.5$, and both produce the same $\hat M$.

In Python these live in `gclm.lasso` as `to_glmnet_lambda`, `from_glmnet_lambda`,
`to_ncvreg_lambda`.

---

## 5. Argument-by-argument justification

| argument | value | why |
|---|---|---|
| `intercept` | `FALSE` | The Lyapunov equation has no constant term; $y=-\mathrm{vec}(C)$ is fully explained by $X\beta$. A free intercept would absorb signal — note $\bar y=-2/p\neq 0$. |
| `standardize` | `FALSE` | Columns of $A(\hat\Sigma)$ carry the problem's own scaling. Standardizing changes the penalty from $\lambda\lvert\beta_j\rvert$ to $\lambda\,\mathrm{sd}_j\lvert\beta_j\rvert$, i.e. a different estimator. |
| `penalty.factor` | `c(1 - diag(p))` | Leaves the diagonal of $M$ unpenalized, matching Varando's `lassoB()`. See `S1_reproduction.md` §4.1. |
| `lambda` | see §4 | Converted from our scale. |
| `thresh` / `eps` | `1e-14` / `1e-12` | Tight, because $X$ is rank-deficient and the objective is flat near the dense end of the path. |
| `dfmax`, `pmax` | $p^2+1$ | glmnet only: stops it capping the active set. |
| `xtx` | `apply(X, 2, crossprod)/nrow(X)` | `ncvfit` only: precomputed column norms it would otherwise expect. |

**Use `ncvfit()`, not `ncvreg()`.** This is the single most important detail on the ncvreg side,
and both help pages say so explicitly.

`?ncvreg`, argument `X`:

> "The design matrix, without an intercept. **`ncvreg` standardizes the data and includes an
> intercept by default.**"

Neither is what we want: $y=-\mathrm{vec}(C)$ has no constant term (and $\bar y=-2/p\neq0$, so a
free intercept absorbs signal), and the columns of $A(\hat\Sigma)$ carry the problem's own scaling.

`?ncvfit`, description:

> "This function is intended for users who know exactly what they're doing and want complete control
> over the fitting process: **no standardization is applied, no intercept is included, no path is
> fit.** […] You should provide initial values for the coefficients; **in nonconvex optimization,
> initial values are very important in determining which local solution an algorithm converges to.**"

The last sentence matters for S1b: MCP and SCAD are nonconvex, so the answer depends on the
starting point. `backend_ncvreg.R` walks the λ-grid from large to small and passes each solution as
`init` for the next — pathwise continuation, which is what `ncvreg()` does internally and what
`ncvfit()` leaves to the caller.

**glmnet ≥ 5.0** deprecates `thresh=` in favour of `control = list(thresh = ...)`;
`backend_glmnet.R` detects which the installed version wants.

### One glmnet trap

On this design glmnet's coordinate descent sometimes fails to converge near the dense end of the
path, and then **returns fewer λ values rather than raising an error** (measured at $p=8$ over 100 λ:
`thresh=1e-10` fits all 100, `1e-12` only 48, `1e-14` only 42). `backend_glmnet.R` therefore tries
`1e-14 → 1e-12 → 1e-10 → 1e-8 → 1e-7`, takes the tightest that returns the full path, reports
`thresh_used`, and errors if none does. If you call `glmnet()` yourself, **check
`length(fit$lambda)` against what you asked for.**

---

## 6. Reproducing the example

```r
suppressPackageStartupMessages({library(glmnet); library(ncvreg)})
p <- 3
Sigma <- matrix(c(2,0.5,0, 0.5,3,1, 0,1,4), 3, 3, byrow = TRUE)
C     <- 2*diag(3)
lambda_paper <- 0.7

MM <- matrix(nrow = p, ncol = p, 1:(p^2))
TT <- diag(p^2)[c(t(MM)), ]
X  <- Sigma %x% diag(p) + ((diag(p) %x% Sigma) %*% TT)
y  <- -c(C)
pf <- c(1 - diag(p))

fit_g <- glmnet(X, y, intercept = FALSE, standardize = FALSE,
                lambda = lambda_paper*(p-1)/p^3, thresh = 1e-14, penalty.factor = pf)
fit_n <- ncvfit(X, y, xtx = apply(X, 2, crossprod)/nrow(X), penalty = "lasso",
                lambda = lambda_paper/p^2, penalty.factor = pf, eps = 1e-12, max.iter = 1e5)

print(round(matrix(as.numeric(fit_g$beta), p, p), 6))
print(round(matrix(fit_n$beta, p, p), 6))

obj <- function(M) 0.5*sum((M %*% Sigma + Sigma %*% t(M) + C)^2) +
                   lambda_paper*sum(abs(M[row(M) != col(M)]))
cat(obj(matrix(as.numeric(fit_g$beta), p, p)),
    obj(matrix(fit_n$beta, p, p)), "\n")
```

The same numbers come out of the Python side with

```python
from gclm.lasso import lasso_path
lasso_path(sigma, c, lambdas=[0.7], solver="glmnet")   # or solver="ncvreg"
```

and `test_all_four_backends_agree` asserts that all four backends return identical supports and
identical metrics on a shared problem.

## 7. MCP and SCAD

Only the `penalty` argument changes; `X`, `y`, `pf` and the λ conversion are untouched:

```r
ncvfit(X, y, xtx = ..., penalty = "MCP",  gamma = 3,   lambda = lambda_paper/p^2, penalty.factor = pf)
ncvfit(X, y, xtx = ..., penalty = "SCAD", gamma = 3.7, lambda = lambda_paper/p^2, penalty.factor = pf)
```

From Python: `run_s1.py --solver ncvreg --penalty MCP --gamma 3`. `glmnet` cannot do this — the
backend raises rather than silently falling back to the lasso.
