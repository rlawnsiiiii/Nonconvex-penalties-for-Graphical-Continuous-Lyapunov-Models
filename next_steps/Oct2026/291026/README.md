# How to read the figures (29 October 2026)

Results: [part1_291026.md](part1_291026.md) ($n=\infty$) and [finite_n_291026.md](finite_n_291026.md). Graphs: [graphs_291026.md](graphs_291026.md).
Data: [data_generation_291026.md](data_generation_291026.md). Two scales, with figures in `figures/cov` and `figures/corr/`:

- `cov`: data not standardised, $C=2I$ in the fit (step 2's setup);
- `corr`: data standardised, rescaled $C$ in the fit, as in the campaign and in the (W2) study of 22 October.

## The four classes

Each pair (graph, $\varepsilon$) falls into one class on each scale, decided at $n=\infty$:

| class | condition | the lasso | (W2) of the adaptive lasso |
|---|---|---|---|
| A | $\rho\lt1$ | recovers | holds |
| B | $\rho\ge1$, $\rho_{\mathrm{weak}}\lt1$ | open | holds |
| R | $\rho_{\mathrm{weak}}>1$, $m(0^+)=M^\ast$ | fails | holds |
| F | $m(0^+)\ne M^\ast$ | fails | fails |

- A: DDK's sufficient condition; B: only the necessary one. R, the gain: no $\lambda$ gives the lasso the signs of $M^\ast$
  (DDK Prop. G.2), but the pilot is exact. F: $m(0^+)$ is zero on an entry of $S$, almost always a true edge
  ([m0_vs_mstar_291026.md](m0_vs_mstar_291026.md)).
- $m(0^+)$: the population lasso path's end as $\lambda\to0$, the minimum-$\ell_1$ exact fit. The pilot, the lasso at
  $\lambda_0=10^{-4}\lambda_{\max}$, sits next to it.

## `population_conditions.png`

Per graph and scale, four lanes: where $\rho\lt1$, $\rho_{\mathrm{weak}}\lt1$, $\rho_w\lt1$ and $m(0^+)=M^\ast$ hold along $\varepsilon$; an
empty lane holds nowhere.

## `population_constants.png`

One panel per graph, $\varepsilon$ against a log y-axis, the black line at $1$: a condition holds where its curve lies below.
Blue $\rho$ and orange $\rho_{\mathrm{weak}}$ are the lasso's constants, aqua $\rho_w$ the adaptive lasso's. Background: white A/B,
light grey R, darker grey F.

**Reading the aqua curve.** With $\hat w=\max\lvert\hat M^0\rvert/\lvert\hat M^0\rvert$ and $\hat M^0\approx M^\ast$ on $S$,

$$
\rho_w=\max_{j\ \text{kept non-edge}}\ \lvert\hat M^0_j\rvert\sum_{k\in S}\frac{\lvert P_{jk}\rvert}{\lvert\hat M^0_k\rvert}
\approx c\cdot\max_j\lvert\hat M^0_j\rvert ,
$$

the largest non-edge the pilot keeps. Each feature is that size changing (chain 1→2→3, not standardised):

| $\varepsilon$ | $t^\ast$ | $\rho_{\mathrm{weak}}$ | kept non-edges, in units of $\lambda_0$ | $\rho_w$ |
|---|---|---|---|---|
| $0.60$ | $0.41$ | $0.997$ | none | $0$ |
| $0.62$ | $0.41$ | $1.016$ | $M_{31}=0.01$ | $9\cdot10^{-7}$ |
| $1.50$ | $0.29$ | $1.34$ | $M_{12}=1.50$ | $1.5\cdot10^{-4}$ |
| $2.46$ | $-0.004$ | $2.34$ | $M_{12}=17605$, $M_{23}=33143$; $M_{32}=0$ | $\infty$ |

- Rise from $0$ (before it, the support is $S$: the bottom strip): $\rho_{\mathrm{weak}}$ crosses $1$ and a non-edge enters at
  size $0$. Step: the pilot switches which non-edge it keeps ($M_{31}$ to the six times larger $M_{12}$).
- Hump: $\rho_{\mathrm{weak}}$ comes back down to $1$ and the kept non-edges shrink to $0$ (fork, $\varepsilon=2.02$ to $2.10$).
- Spike: $t^\ast\approx0$ makes the objective flat along the competitor's direction and FISTA stops short of the lasso
  solution, with up to $56\lambda_0$ on a reverse against the exact $0.8$ to $2.3\lambda_0$; in the sliver $t^\ast\lt0$
  ($\varepsilon=1.74$ to $1.78$) it leaves $1$ to $8\lambda_0$ on the lost edge ($\rho_w\approx10^3$, not $\infty$). The exact
  $\rho_w$ is smooth, also at the standardised left end (`../301026/rho_w_steps_301026.md`).

Not standardised, $t^\ast\lt0.05$ only just before F ($9\,\%$ of R, $61\,\%$ standardised), so the curves stay
smooth.

**One caveat on the values.** In graphs with symmetric parallel paths the lasso at $\lambda_0$ has a flat optimum (the loss
does not see mass moved between the paths), so its solution is not unique. The pilot of the campaign's FISTA is the
symmetric one; an exact solve gives a vertex with the same objective to 12 digits but other non-edge entries, and
$\rho_w$ differs by up to a factor $2$. The curves show the implemented pilot.

## `population_rho_w_ratio.png`

$\rho_w$ for pilots at $\lambda_0=10^{-2},10^{-3},10^{-4}\lambda_{\max}$, darker for smaller $\lambda_0$: a decade apart. Not standardised,
only $10^{-2}$ reaches $1$ inside R (chain of 5); standardised, $10^{-2}$ and $10^{-3}$ do in several graphs.

## `finite_n.png`

Rows are graphs, columns one $\varepsilon$ per class (in the panel title); $n$ on the x-axis, the dot at $\infty$ the population.
The y-axis: the share of 50 replicates in which

- blue: some $\lambda$ on the adaptive lasso's path gives the signs of $M^\ast$ in every entry;
- orange: the same for the lasso;
- aqua: (W2) holds with the pilot fitted to the same data.

Where lines coincide, only the top one shows.
