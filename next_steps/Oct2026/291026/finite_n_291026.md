# Part 1 graphs at finite $n$ (29 October 2026)

Design in [data_generation_291026.md](data_generation_291026.md): per graph one $\varepsilon$ per class (A/B, R, F), $n=10^2,\dots,10^5$
and $\infty$, 50 replicates. The graph is fixed; each replicate draws new data, so $\hat\Sigma$ and the estimates differ
between replicates, and a share estimates the probability of recovery. At $n=\infty$ there is no sampling: one run. Per replicate: whether (W2) holds with the implemented pilot ($\lambda_0=10^{-4}\lambda_{\max}$), and
whether some $\lambda$ on the path of the adaptive lasso, and of the lasso, gives the signs of $M^\ast$ in every entry. How to
read the figure: [README](README.md). Code: `part1_finite_n.py`.

## Data not standardised, $C=2I$

![finite n, not standardised](figures/cov/finite_n.png)

Class R, % of replicates with the signs of $M^\ast$ on the path, adaptive lasso / lasso:

| graph | $\varepsilon$ | $n=10^2$ | $10^3$ | $10^4$ | $10^5$ | $\infty$ |
|---|---|---|---|---|---|---|
| 3 parallel paths | $0.9$ | $30/0$ | $92/4$ | $100/0$ | $100/4$ | $100/0$ |
| diamond | $1.14$ | $54/8$ | $100/12$ | $100/28$ | $100/10$ | $100/0$ |
| chain 1→2→3 | $1.52$ | $90/62$ | $100/46$ | $100/46$ | $100/40$ | $100/0$ |
| chain of 4 | $1.92$ | $68/6$ | $100/4$ | $100/2$ | $100/0$ | $100/0$ |
| chain of 5 | $2.28$ | $66/0$ | $100/0$ | $100/2$ | $100/0$ | $100/0$ |
| chain, close d | $0.74$ | $48/40$ | $84/54$ | $100/56$ | $100/50$ | $100/0$ |
| collider | $1.4$ | $68/44$ | $100/52$ | $100/44$ | $100/58$ | $100/0$ |
| 3 paths, $M_{21}=-\varepsilon$ | $1.38$ | $14/0$ | $98/0$ | $100/2$ | $100/0$ | $100/0$ |
| fork | $1.66$ | $46/40$ | $92/52$ | $100/48$ | $100/42$ | $100/0$ |
| diamond, $M_{21}=-\varepsilon$ | $1.66$ | $22/10$ | $94/20$ | $100/26$ | $100/28$ | $100/0$ |

- R: the adaptive lasso recovers the signs in every replicate from $n=10^4$ on, in all ten graphs with a class R. The
  lasso does in at most $62\,\%$, and never at $n=\infty$, as $\rho_{\mathrm{weak}}>1$ predicts. At finite $n$ the noise
  sometimes lets it through.
- (W2) never holds in more replicates than the adaptive lasso recovers, and from $n=10^3$ on it holds in at least $82\,\%$.
- A/B: both reach $100\,\%$ by $n=10^5$. The adaptive lasso is mostly ahead (chain of 5 at $n=10^4$: $86$ against $10$),
  and behind only in the fork, the close chain and the triangle at $n\le10^3$, by at most 16 points.
- F: both fail at every $n$.
- With the pilot at $\lambda_0=10^{-2}\lambda_{\max}$ the adaptive lasso's shares change by at most 4 points.

## Data standardised, rescaled $C$ (as in the campaign)

The class of each $\varepsilon$ is taken on this scale; several graphs have no class A/B or F here.

![finite n, standardised](figures/corr/finite_n.png)

Class R, adaptive lasso / lasso as above:

| graph | $\varepsilon$ | $n=10^2$ | $10^3$ | $10^4$ | $10^5$ | $\infty$ |
|---|---|---|---|---|---|---|
| 3 parallel paths | $3.24$ | $16/0$ | $84/0$ | $100/0$ | $100/0$ | $100/0$ |
| diamond | $3.32$ | $42/8$ | $94/0$ | $100/2$ | $100/6$ | $100/0$ |
| chain 1→2→3 | $3.9$ | $92/8$ | $100/8$ | $100/8$ | $100/8$ | $100/0$ |
| chain of 4 | $3$ | $34/0$ | $78/0$ | $90/0$ | $100/0$ | $100/0$ |
| chain of 5 | $3$ | $26/0$ | $86/0$ | $98/0$ | $100/0$ | $100/0$ |
| chain, close d | $3$ | $80/14$ | $100/14$ | $100/18$ | $100/18$ | $100/0$ |
| collider | $3$ | $90/56$ | $100/52$ | $100/44$ | $100/58$ | $100/0$ |
| 3 paths, $M_{21}=-\varepsilon$ | $0.56$ | $2/0$ | $20/2$ | $80/2$ | $100/8$ | $100/0$ |
| fork | $1.38$ | $78/40$ | $100/60$ | $100/48$ | $100/42$ | $100/0$ |
| diamond, $M_{21}=-\varepsilon$ | $0.38$ | $0/0$ | $8/12$ | $54/16$ | $98/18$ | $100/0$ |

- R: the adaptive lasso recovers the signs in at least $98\,\%$ of replicates at $n=10^5$ in all ten graphs, and in
  every replicate from $n=10^4$ on in six. It needs more data than without standardising, since the margins $t^\ast$ are
  small here. The lasso does in at most $60\,\%$.
- A/B exists only at small $\varepsilon$, in four graphs, and there the lasso is ahead in three (3 parallel paths at
  $n=10^4$: $88$ against $18$; the pilot keeps $S$ in $38\,\%$). In the triangle both are equal.
- F: both fail at $n=\infty$. At finite $n$ the noise lets some replicates through near the edge of F (chain 1→2→3 at
  $\varepsilon=1.78$, up to $52\,\%$).
- The pilot at $10^{-2}\lambda_{\max}$ is much worse for the 3 parallel paths and the diamond (3 parallel paths at
  $n=10^4$: $16$ against $100$).
- 28 of 3,819 replicates hit the iteration cap of a path.
