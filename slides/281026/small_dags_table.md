# Small DAGs at $n=\infty$: the table of 26 October, verified (2 November 2026)

Source: `../../next_steps/Oct2026/261026`, "Scratch results". Those numbers came from scratch scripts outside
the repository, so their tolerance was not recorded. Recomputed here with `../../next_steps/Oct2026/291026`
(`population`, `certificate`, `min_l1_fit`): every entry is reproduced.

Setting: $M^\ast=-\mathrm{diag}(d)+\varepsilon B$, true and fitted $C=2I$, data not standardised, every entry of $M$
penalised (diagonal included), $\hat\Sigma=\Sigma^\ast$.

## Table 1: the seven rows

| graph | $d$ | $\varepsilon$ | $\rho$ | $\rho_{\mathrm{weak}}$ | $t^\ast$ | $\lVert m(0^+)-m^\ast\rVert_\infty$ | $m(0^+)=m^\ast$ |
|---|---|---|---|---|---|---|---|
| chain $1\to2\to3$ | $(1,2,3)$ | $0.01$ | $0.671$ | $0.667$ | $0.335$ | $2\cdot10^{-18}$ | yes |
| chain | $(3,2,1)$ | $0.01$ | $2.007$ | $2.000$ | $-0.997$ | $1.0\cdot10^{-2}$ | no |
| chain | $(1,2,3)$ | $1$ | $1.446$ | $1.249$ | $0.373$ | $4\cdot10^{-16}$ | yes |
| chain | $(1,2,3)$ | $2$ | $3.066$ | $1.648$ | $0.159$ | $4\cdot10^{-16}$ | yes |
| collider $1\to3\leftarrow2$ | $(1,1,2)$ | $1.5$ | $2.833$ | $1.833$ | $0.167$ | $0$ | yes |
| diamond | $(1,2,2,3)$ | $0.5$ | $1.524$ | $1.474$ | $0.402$ | $1\cdot10^{-15}$ | yes |
| diamond, $M_{21}\lt0$ | $(1,2,2,3)$ | $0.5$ | $1.396$ | $0.678$ | $0.364$ | $1\cdot10^{-16}$ | yes |

The "no" row is the near-diagonal lemma (b) of `steps_1_2_261026.md`: $m(0^+)$ is $0$ on both edges and puts
$\varepsilon d_h/d_t$ on the reverses, $M_{12}=0.0067$, $M_{23}=0.005$.

## Table 2: largest $\varepsilon$ up to which each condition holds

Grid $\varepsilon=0.02,0.04,\dots,6$; the condition fails at every larger grid point.

| graph | $d$ | $\rho\lt1$ | $\rho_{\mathrm{weak}}\lt1$ | $t^\ast>0$ | $m(0^+)=m^\ast$ |
|---|---|---|---|---|---|
| chain $1\to2\to3$ | $(1,2,3)$ | $0.58$ | $0.60$ | $2.44$ | $2.44$ |
| chain | $(3,2,1)$ | never | never | never | never |
| collider $1\to3\leftarrow2$ | $(1,1,2)$ | $0.52$ | $0.96$ | $1.82$ | $1.82$ |
| diamond | $(1,2,2,3)$ | $0.26$ | $0.26$ | $2.02$ | $2.02$ |
| diamond, $M_{21}\lt0$ | $(1,2,2,3)$ | $0.28$ | $1.50$ | $1.82$ | $1.82$ |

## Resolution

- $m(0^+)=\arg\min\lVert m\rVert_1$ subject to $A^\ast m=-\mathrm{vec}(C)$: a linear programme, `scipy.optimize.linprog`
  with HiGHS.
- $m(0^+)=m^\ast$ means $\lVert m(0^+)-m^\ast\rVert_\infty\le10^{-8}\max\lvert m^\ast\rvert$. The verdict does not depend
  on this threshold: where it holds the error is at most $3\cdot10^{-15}$ (machine precision), where it fails at
  least $10^{-2}$ (the whole edge mass has moved; at least $1.8$ in class F of Table 2).
- $t^\ast$ is the LP of the certificate (`certificate`), counted positive above $10^{-9}$. $t^\ast>0$ and $m(0^+)=m^\ast$
  agree at every one of the $5\times300$ grid points.
- Diagonal penalised or not: the same $t^\ast$ and the same verdict in every row of Table 1.
- $\rho$, $\rho_{\mathrm{weak}}$ at $\Sigma^\ast$ itself; the 26 October values are these rounded to two decimals.
