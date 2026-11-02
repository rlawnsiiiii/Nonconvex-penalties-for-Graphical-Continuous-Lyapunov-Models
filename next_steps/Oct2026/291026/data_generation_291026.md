# How the data were generated (29 October 2026)

For [part1_291026.md](part1_291026.md). Code: `part1_population.py` (`GRAPHS`, `drift`, `model`) and `part1_finite_n.py`
(`fit_data`, `class_points`).

## 1. Drift matrix $M^\ast$

$$
M^\ast=-\mathrm{diag}(d_1,\dots,d_p)+\varepsilon B,\qquad B_{ht}=s_e\ \text{for each edge }e=t\to h,\qquad B_{ht}=0\ \text{otherwise}.
$$

- $M^\ast_{ij}\ne0$ for $i\ne j$ means the edge $j\to i$: row is child, column is parent, as in DDK.
- Every edge has the same weight $\varepsilon$, up to its sign $s_e=\pm1$. $s_e=-1$ only on $1\to2$ in the two variants
  with $M_{21}=-\varepsilon$; every other sign is $+1$.
- $d_i=1+\mathrm{depth}(i)$, where $\mathrm{depth}(i)$ counts the edges on the longest directed path into $i$. So $d_t\lt d_h$ on
  every edge $t\to h$: the tail $t$ (the parent) decays more slowly than the head $h$ (the child). The one exception is
  the close chain.
- $\varepsilon=0.02,0.04,\dots,6$, 300 values. Nothing in $M^\ast$ is random.
- The nodes are numbered in a topological order, so $M^\ast$ is lower triangular with eigenvalues $-d_i\lt0$: stable for
  every $\varepsilon$.

| graph | $d$ | edges |
|---|---|---|
| 3 parallel paths | $(1,2,2,2,3)$ | $1\to2,\ 1\to3,\ 1\to4,\ 2\to5,\ 3\to5,\ 4\to5$ |
| diamond | $(1,2,2,3)$ | $1\to2,\ 1\to3,\ 2\to4,\ 3\to4$ |
| chain of $p=3,4,5$ | $(1,2,\dots,p)$ | $1\to2\to\dots\to p$ |
| close chain | $(1,1.2,1.44)$ | $1\to2\to3$ |
| collider | $(1,1,2)$ | $1\to3,\ 2\to3$ |
| fork | $(1,2,2)$ | $1\to2,\ 1\to3$ |
| triangle | $(1,2,3)$ | $1\to2,\ 2\to3,\ 1\to3$ |

The two variants are the 3 parallel paths and the diamond with $M_{21}=-\varepsilon$. Pictures and matrices:
[graphs_291026.md](graphs_291026.md).

## 2. Volatility and covariance

The true volatility is $C=2I$ (DDK's $C_{\mathrm{ID}}$), and $\Sigma^\ast$ solves (`gclm.lyapunov.solve_lyapunov`)

$$
M^\ast\Sigma^\ast+\Sigma^\ast M^{\ast\top}+C=0 .
$$

## 3. Samples, finite $n$ only

$X_1,\dots,X_n$ iid $N(0,\Sigma^\ast)$ (`sample_data`) and $\hat\Sigma=X^\top X/n$ (`sample_covariance`: no centring, divisor
$n$), as in the campaign. $n\in\lbrace10^2,10^3,10^4,10^5\rbrace$, 50 replicates, seed `[20261029, graph, class, n, replicate]`.
At $n=\infty$, $\hat\Sigma=\Sigma^\ast$.

Per graph there are three values of $\varepsilon$, one per class, taken from the population run on the same scale:

- the middle of the A/B range;
- the middle of the R range;
- the F point nearest to $1.25$ times the end of the first range with $m(0^+)=M^\ast$.

A class the graph does not have is left out.

## 4. What the estimator is given

| scale | covariance | $C$ in the fit | truth |
|---|---|---|---|
| not standardised | $\hat\Sigma$ | $2I$ | $M^\ast$ |
| standardised, as in the campaign | $\hat R=\hat D^{-1}\hat\Sigma\hat D^{-1}$ | $2\hat D^{-2}$ | $D^{-1}M^\ast D$ |

Here $\hat D=\mathrm{diag}(\hat\Sigma_{ii}^{1/2})$ and $D=\mathrm{diag}(\Sigma^{\ast\,1/2}_{ii})$, with $\hat D=D$ at $n=\infty$. Standardising is the
change of variables $X\mapsto\hat D^{-1}X$:

$$
(\hat D^{-1}M\hat D)\,\hat R+\hat R\,(\hat D^{-1}M\hat D)^\top+\hat D^{-1}C\hat D^{-1}=\hat D^{-1}\big(M\hat\Sigma+\hat\Sigma M^\top+C\big)\hat D^{-1}.
$$

So for standardised data the model's volatility is $\hat D^{-1}C\hat D^{-1}=2\hat D^{-2}$, the rescaled $C$
(`estimation_volatility(sd, "variance")`). The drift becomes $\hat D^{-1}M\hat D$, which has the support and signs of $M$.

- Sign recovery is checked against $\mathrm{sign}(M^\ast)$.
- $P$, $\rho$ and $\rho_{\mathrm{weak}}$ use the population model on the same scale: $\Sigma^\ast$ unstandardised,
  $\mathrm{corr}(\Sigma^\ast)$ standardised.
- The pilot and both lasso paths are fitted to the covariance and $C$ in the table, with every entry of $M$ penalised.

## 5. Difference from the campaign's generator

The campaign draws random graphs: each ordered pair is an edge with probability $k/p$, with $N(0,1)$ weights and
$M_{ii}=-\sum_{j}\lvert M_{ij}\rvert-\lvert\varepsilon_{ii}\rvert$. Here the graph, the edge weight $\varepsilon$ and the rates $d$ are fixed by
hand, and only the samples are random.
