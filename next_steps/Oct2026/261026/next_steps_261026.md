# Next steps: when does $\mathcal E_w$ hold? (26 October 2026)

Theorem W.2 (`../211026/adaptive_lasso_theorem_211026.tex`) holds with probability
$1-c_2e^{-\tau_1p}-P(\mathcal E_w^c)$, and $P(\mathcal E_w^c)$ is unknown. By Corollary W.3 a bound on the pilot suffices:

$$
P\big(\lVert\hat M^0-M^\ast\rVert_\infty\le\delta\big)\ge1-c\,e^{-\tau p},\qquad \delta\lt\beta_{\min}/(1+\rho).
$$

**Target theorem** (to be proved): this holds for a pilot at $\lambda_0$ near the noise level whenever $t^\ast>0$,
without any irrepresentability condition. Start after confirmation.

## Notation

$A^\ast=A(\Sigma^\ast)$, $S$ includes the diagonal (sign $-1$), $m(\lambda)$ is the population lasso, $m(0^+)$ its
dense end, the minimum-$\ell_1$ exact fit (Remark W.6):

$$
P=\Gamma^\ast_{S^cS}(\Gamma^\ast_{SS})^{-1},\qquad \rho_{\mathrm{weak}}=\lVert P\,\mathrm{sign}(m^\ast_S)\rVert_\infty,\qquad
$$

$$
\rho\lt1\ \Longrightarrow\ \rho_{\mathrm{weak}}\lt1\ \Longrightarrow\ t^\ast>0\ \Longrightarrow\ m(0^+)=m^\ast\qquad(\Gamma^\ast_{SS}\succ0).
$$

**What $t^\ast$ means.** Every exact fit is $m^\ast+v$ with $A^\ast v=0$. Moving along $v$ saves $\ell_1$ norm on the true
support and costs $\lVert v_{S^c}\rVert_1$ on the non-edges; $1-t^\ast$ is the best exchange rate:

$$
1-t^\ast=\max_{A^\ast v=0,\ v_{S^c}\ne0}\ \frac{-\mathrm{sign}(m^\ast_S)^\top v_S}{\lVert v_{S^c}\rVert_1},
\qquad \lVert m\rVert_1\ge\lVert m^\ast\rVert_1+t^\ast\lVert m_{S^c}\rVert_1\ \ \text{for every exact fit}\ m .
$$

So $t^\ast$ is the price per unit of non-edge mass. If $t^\ast>0$ no trade pays and the minimum-$\ell_1$ fit is $m^\ast$; if
$t^\ast\lt0$ one does, and the pilot moves mass onto non-edges. The certificate $h$ is the LP dual of this maximum
(scratch check: both equal to four decimals). The lasso at $\lambda>0$ has no choice of trade: shrinking $m^\ast_S$ by
$\lambda(\Gamma^\ast_{SS})^{-1}\mathrm{sign}(m^\ast_S)$ shows up on the non-edges as $\lambda P\,\mathrm{sign}(m^\ast_S)$, the least-norm
certificate. Hence $t^\ast\ge1-\rho_{\mathrm{weak}}$, and the gap is what the adaptive lasso can gain. Chain $d=(1,2,3)$,
$\varepsilon=1$: $\rho_{\mathrm{weak}}=1.25$ (lasso fails), $t^\ast=0.37$ (pilot exact).

## Small DAGs

$$
M=-\mathrm{diag}(d_1,\dots,d_p)+\varepsilon B,\qquad B_{ht}=1\ \text{for each edge}\ t\to h,\qquad C=2I.
$$

- $d_i>0$: decay rate of node $i$, the diagonal of DDK Theorem 3; $\Sigma=\mathrm{diag}(1/d_i)$ at $\varepsilon=0$.
- $\varepsilon$: common edge weight (DDK's $e$, p. 9); Theorem 3 is $\varepsilon\to0$.
- Right ordering: $d_t\lt d_h$ on every edge $t\to h$.

Scratch results ($n=\infty$):

| graph | $d$ | $\varepsilon$ | $\rho$ | $\rho_{\mathrm{weak}}$ | $m(0^+)=m^\ast$ |
|---|---|---|---|---|---|
| chain $1\to2\to3$ | $(1,2,3)$ | $0.01$ | $0.67$ | $0.67$ | yes |
| chain | $(3,2,1)$ | $0.01$ | $2.01$ | $2.00$ | no |
| chain | $(1,2,3)$ | $1$ | $1.45$ | $1.25$ | yes |
| chain | $(1,2,3)$ | $2$ | $3.07$ | $1.65$ | yes |
| collider $1\to3\leftarrow2$ | $(1,1,2)$ | $1.5$ | $2.83$ | $1.83$ | yes |
| diamond | $(1,2,2,3)$ | $0.5$ | $1.52$ | $1.47$ | yes |
| diamond, $M_{21}\lt0$ | $(1,2,2,3)$ | $0.5$ | $1.40$ | $0.68$ | yes |

The gain: right ordering with strong edges, where the lasso fails ($\rho_{\mathrm{weak}}>1$, DDK Prop. G.2) but $t^\ast>0$.

## Steps

1. **Near-diagonal lemma:** for small $\varepsilon$, $m(0^+)=m^\ast$ iff right ordering, so no gain there.
2. **Population map:** where $\rho_{\mathrm{weak}}>1$ and $t^\ast>0$ on small DAGs.
3. **Population path:** $\lVert m(\lambda)-m^\ast\rVert_\infty\le L\lambda$ when $t^\ast>0$.
4. **Finite $n$:** the pilot bound above, then the target theorem.
5. **Monte Carlo:** $P(\mathcal E_w)$ against $n$.

Details: [steps 1–2](steps_1_2_261026.md), [step 2 in full](step2_261026.md), [steps 3–5](steps_3_5_261026.md); literature: [reading](reading_261026.md).
Order: 1 and 2 first (laptop), reading for 3–4 alongside, 5 with 4. First task: the `diag_penalized` flag in
`bp_certificate` with its test, then `population_map.py` for the 3-chain.
