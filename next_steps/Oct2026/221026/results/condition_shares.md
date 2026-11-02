# Shares of graphs meeting the conditions (%)

## Graphs and data

- 400 drift matrices $M^\ast$: $p \in \lbrace 10, 20 \rbrace$ times generator `random` or `dag` times $k = 1, \dots, 4$ times 25 replicates, i.e. 100 per ($p$, generator); seed `[20260922, p, k, 0, rep]` (campaign seeds, true $C = 2I$). Each $M^\ast$ serves every $n$ and both $C$.
- `random` (Dettling et al. 2024, §5): $M_{ij} = \omega_{ij}\varepsilon_{ij}$, $\omega_{ij} \sim \mathrm{Bern}(k/p)$, $\varepsilon_{ij} \sim N(0,1)$, $M_{ii} = -\sum_{j \ne i}|M_{ij}| - |\varepsilon_{ii}|$.
- `dag` (`sample_drift(graph="dag")`): random order $\pi$; edge $j \to i$ allowed iff $\pi(j) \lt \pi(i)$, kept with probability $\min(2k/p, 1)$ (same expected edge count); same $\varepsilon$ and diagonal.
- $\Sigma^\ast$ solves $M^\ast\Sigma^\ast + \Sigma^\ast M^{\ast\top} + 2I = 0$; $X_1, \dots, X_n \sim N(0, \Sigma^\ast)$, $\hat\Sigma = X^\top X / n$ ($n = \infty$: $\hat\Sigma = \Sigma^\ast$); standardised, $\hat R = D^{-1}\hat\Sigma D^{-1}$, $D = \mathrm{diag}(\hat\Sigma_{ii}^{1/2})$.

## Pilot and weights

- $C$ in the fit: `Cresc` $= 2D^{-2}$ (exact for standardised data), `C2I` $= 2I$.
- $\lambda_{\max}$, from the data: the smallest $\lambda$ with a diagonal estimate, $\min\lbrace\lambda : \lvert[2R(D_\lambda)\hat R]_{ij}\rvert \le \lambda\ \forall i \ne j\rbrace$, $D_\lambda$ the diagonal-only solution, $R(M) = M\hat R + \hat R M^\top + C$ (bisection; `gclm.objective.direct.lambda_max`).
- Grid $\lambda_j = \lambda_{\max}\,10^{-4(1 - j/99)}$, $j = 0, \dots, 99$; so $\lambda_{\min} = 10^{-4}\lambda_{\max}$ is a fixed ratio, not estimated (`--ratio` changes it).
- Pilot: FISTA down the grid from $\lambda_{\max}$, warm-started, tolerance $10^{-8}$, at most 50 000 iterations per $\lambda$; $\hat M^0$ = the solution at $\lambda_{\min}$,

$$
\hat M^0 = \arg\min_M \tfrac12\lVert M\hat R + \hat R M^\top + C\rVert_F^2 + \lambda_{\min}\lVert M\rVert_1 \quad (\text{every entry penalised})
$$

- Weights $\hat w_{jk} = \max_{kl}|\hat M^0_{kl}| / |\hat M^0_{jk}|$, $\infty$ where $\hat M^0_{jk} = 0$ (`adaptive_lasso_path(..., penalize_diagonal=True).weights`).

## Columns

- $S = \mathrm{supp}(M^\ast)$ with the diagonal; $P = \Gamma^\ast_{S^cS}(\Gamma^\ast_{SS})^{-1}$ at $R^\ast = \mathrm{corr}(\Sigma^\ast)$; $S^c_+$: non-edges with finite weight.
- (W2): $\rho_w \lt 1$, $\rho_w = \max_{j \in S^c_+}\hat w_j^{-1}\sum_{k \in S}|P_{jk}|\hat w_k$.
- lasso: $\rho = \max_j\sum_{k \in S}|P_{jk}| \lt 1$ (DDK (4.1)); weak: $\max_j\lvert\sum_{k \in S}P_{jk}\,\mathrm{sign}(M^\ast_k)\rvert \lt 1$ (DDK (G.1)), $\mathrm{sign}(M^\ast_{ii}) = -1$; the same for both $C$.
- `keeps S`: $\hat M^0_{jk} \ne 0$ on all of $S$. Otherwise $\bar w_S = \infty$, Theorem W.1 does not apply, $\rho_w := \infty$, (W2) fails.
- (W2) given `keeps S` = `a/b`: $b$ graphs keep $S$, $a$ of them satisfy (W2); so (W2) $= a$ and `keeps S` $= b$ (of 100).

Per $k$ and the medians: `rho_w_summary.csv`; per graph: `rho_w_per_pilot.csv`.

## Shares (%)

| p | graphs | n | lasso `rho < 1` | lasso weak | (W2) `Cresc` | `keeps S` `Cresc` | (W2) given `keeps S` `Cresc` | (W2) `C2I` | `keeps S` `C2I` | (W2) given `keeps S` `C2I` |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | `random` | 10^3 | 0 | 0 | 0 | 6 | 0/6 | 0 | 7 | 0/7 |
| 10 | `random` | 10^4 | 0 | 0 | 4 | 11 | 4/11 | 1 | 7 | 1/7 |
| 10 | `random` | inf | 0 | 0 | 11 | 18 | 11/18 | 2 | 15 | 2/15 |
| 10 | `dag` | 10^3 | 0 | 0 | 1 | 5 | 1/5 | 0 | 4 | 0/4 |
| 10 | `dag` | 10^4 | 0 | 0 | 6 | 15 | 6/15 | 0 | 10 | 0/10 |
| 10 | `dag` | inf | 0 | 0 | 18 | 22 | 18/22 | 1 | 18 | 1/18 |
| 20 | `random` | 10^3 | 0 | 0 | 1 | 2 | 1/2 | 0 | 2 | 0/2 |
| 20 | `random` | 10^4 | 0 | 0 | 0 | 6 | 0/6 | 0 | 2 | 0/2 |
| 20 | `random` | inf | 0 | 0 | 19 | 24 | 19/24 | 0 | 12 | 0/12 |
| 20 | `dag` | 10^3 | 0 | 0 | 0 | 3 | 0/3 | 0 | 2 | 0/2 |
| 20 | `dag` | 10^4 | 0 | 0 | 1 | 6 | 1/6 | 0 | 4 | 0/4 |
| 20 | `dag` | inf | 0 | 0 | 23 | 28 | 23/28 | 0 | 12 | 0/12 |

## Per k (counts of the 25 graphs per cell)

`a/b`: a graphs satisfy (W2), b keep S. The lasso's `rho < 1` holds for no k.

### `Cresc`

| p | graphs | n | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|
| 10 | `random` | 10^3 | 0/6 | 0/0 | 0/0 | 0/0 |
| 10 | `random` | 10^4 | 3/8 | 1/2 | 0/1 | 0/0 |
| 10 | `random` | inf | 7/14 | 4/4 | 0/0 | 0/0 |
| 10 | `dag` | 10^3 | 1/5 | 0/0 | 0/0 | 0/0 |
| 10 | `dag` | 10^4 | 6/12 | 0/3 | 0/0 | 0/0 |
| 10 | `dag` | inf | 12/15 | 6/7 | 0/0 | 0/0 |
| 20 | `random` | 10^3 | 1/2 | 0/0 | 0/0 | 0/0 |
| 20 | `random` | 10^4 | 0/6 | 0/0 | 0/0 | 0/0 |
| 20 | `random` | inf | 8/13 | 7/7 | 4/4 | 0/0 |
| 20 | `dag` | 10^3 | 0/3 | 0/0 | 0/0 | 0/0 |
| 20 | `dag` | 10^4 | 1/5 | 0/1 | 0/0 | 0/0 |
| 20 | `dag` | inf | 8/13 | 11/11 | 4/4 | 0/0 |

### `C2I`

| p | graphs | n | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|
| 10 | `random` | 10^3 | 0/6 | 0/1 | 0/0 | 0/0 |
| 10 | `random` | 10^4 | 1/6 | 0/1 | 0/0 | 0/0 |
| 10 | `random` | inf | 2/12 | 0/3 | 0/0 | 0/0 |
| 10 | `dag` | 10^3 | 0/4 | 0/0 | 0/0 | 0/0 |
| 10 | `dag` | 10^4 | 0/9 | 0/1 | 0/0 | 0/0 |
| 10 | `dag` | inf | 0/13 | 1/5 | 0/0 | 0/0 |
| 20 | `random` | 10^3 | 0/2 | 0/0 | 0/0 | 0/0 |
| 20 | `random` | 10^4 | 0/2 | 0/0 | 0/0 | 0/0 |
| 20 | `random` | inf | 0/10 | 0/2 | 0/0 | 0/0 |
| 20 | `dag` | 10^3 | 0/2 | 0/0 | 0/0 | 0/0 |
| 20 | `dag` | 10^4 | 0/3 | 0/1 | 0/0 | 0/0 |
| 20 | `dag` | inf | 0/7 | 0/5 | 0/0 | 0/0 |
