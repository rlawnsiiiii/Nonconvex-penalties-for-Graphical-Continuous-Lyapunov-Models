# Orientation lock-in on the sparse → dense path

*Statement, proof, numerical evidence and references for the claim in the memo
`next_steps/independent_study_021026.md` (§3–§4):*

> At the sparse end of the path, the two orientations of an edge fit almost equally well, so the
> sparse → dense MCP/SCAD path keeps whichever orientation entered first.

Everything below was derived and checked numerically in the study's sandbox (5 October 2026). Parts
marked **exact** hold for every sample covariance; parts marked **leading order** are expansions in
the edge weights; parts marked **numerical** are illustrations on the thesis DGP. §6 says what would
be needed to turn this into a lemma for the thesis.

---

## 0. Short answer

The claim splits into two halves, and both can be proved.

1. **The loss is nearly flat between the two orientations.** For the direct loss
   $L(M) = \tfrac12\lVert M\hat\Sigma + \hat\Sigma M^\top + C\rVert_F^2$, restricted to the pair
   $(M_{ij}, M_{ji})$, the Hessian is *exactly* a rank-one matrix plus a diagonal correction whose
   entries are sums of squared covariances of nodes $i$ and $j$ with the other nodes (Lemma 2).
   The rank-one part says the loss sees only the combination $\hat\Sigma_{jj}M_{ij} + \hat\Sigma_{ii}M_{ji}$;
   the correction is what distinguishes the orientations, and in the population it is second order in
   the edge weights (Lemma 3). In the two-node case the orientation is not identifiable at all.
2. **The path cannot cross from one orientation to the other.** For MCP and SCAD the penalised objective,
   restricted to the line along which one orientation is traded for the other at constant loss, is a
   *concave* function with a hump between the two orientations (Proposition 5). The orientation that
   entered first is therefore a strict local minimum at every later $\lambda$, and a descent method
   warm-started there (the repo's monotone APG, or coordinate descent) stays. Which orientation enters
   first is decided by the gradient at the diagonal fit, whose ratio between the two orientations is
   $\hat\Sigma_{jj} : \hat\Sigma_{ii}$ up to second-order terms (Proposition 4): the variances, not the
   true direction. For the lasso the same restriction is *linear*, there is no hump, and the convex
   problem does not depend on the warm start, which is why the lasso does not have this failure mode
   (it hedges instead).

Reference situation: I am not aware of a paper that states this for GCLMs. The ingredients are published
(the exact null space of the Lyapunov loss in the identifiability paper of Dettling, Homs, Améndola, Drton
& Hansen; the path dependence of folded-concave penalties in Breheny & Huang 2011, Mazumder–Friedman–Hastie
2011, Wang–Kim–Li 2013, Fan–Xue–Zou 2014, Loh & Wainwright 2015; Varando & Hansen's remark that the
increasing-$\lambda$ sequence works better for their likelihood path). The assembly, the rank-one Hessian
and the weak-coupling expansion are from the study (§7).

---

## 1. Setting and notation

- $p$ nodes, drift matrix $M \in \mathbb R^{p\times p}$ with $M_{ij} \neq 0$ for $i \ne j$ meaning the edge
  $j \to i$ (the repo's convention). Diagonal $d_i = M_{ii} < 0$. $M$ Hurwitz.
- Volatility $C \succ 0$, diagonal; `C_ID` is $C = 2I$. Stationary covariance $\Sigma = \Sigma(M, C)$ solves
  $M\Sigma + \Sigma M^\top + C = 0$; explicitly $\Sigma = \int_0^\infty e^{tM} C e^{tM^\top}\,dt$.
- Sample covariance $\hat\Sigma$ (a correlation matrix after standardizing). $C_{\mathrm{est}}$ is the
  volatility matrix handed to the estimator; on the correlation scale the correct one is
  $2\,\mathrm{diag}(1/\hat s_{ii})$ (`variance`), the pilot uses $2I$ (`identity`).
- Direct loss $L(M) = \tfrac12\lVert R(M)\rVert_F^2$ with residual $R(M) = M\hat\Sigma + \hat\Sigma M^\top + C_{\mathrm{est}}$,
  which is linear in $M$. The map $\mathcal A(M) = M\hat\Sigma + \hat\Sigma M^\top$ is linear from
  $\mathbb R^{p\times p}$ ($p^2$ dimensions) onto the symmetric matrices ($p(p+1)/2$ dimensions).
- Penalised objective $F_\lambda(M) = L(M) + \sum_{i\ne j} P_\lambda(\lvert M_{ij}\rvert)$, diagonal unpenalised.
  - lasso: $P_\lambda(t) = \lambda t$.
  - MCP ($\gamma = 3$ in the pilot): $P_\lambda(t) = \lambda t - t^2/(2\gamma)$ for $t \le \gamma\lambda$,
    $= \gamma\lambda^2/2$ for $t \ge \gamma\lambda$; derivative $P'_\lambda(t) = (\lambda - t/\gamma)_+$.
  - SCAD ($\gamma = 3.7$): $P'_\lambda(t) = \lambda$ for $t \le \lambda$,
    $= (\gamma\lambda - t)_+/(\gamma - 1)$ for $t > \lambda$.
  - Both MCP and SCAD are concave on $[0,\infty)$, have slope $\lambda$ at $0^+$ and slope $0$ beyond $\gamma\lambda$.
- $\lambda$ grid: 100 points $\lambda_{\max}\cdot 10^{\,\mathrm{linspace}(-4, 0)}$.
  **Sparse → dense path** (`mcp_mapg`, the pilot): solve at $\lambda_{\max}$ (diagonal-only fit), then at each next
  smaller $\lambda$ warm-start from the previous solution. **Dense → sparse path** (`mcp_up`): start from the
  lasso solution at $\lambda_{\min}$ and warm-start upwards.
- For a fixed pair $i \ne j$ write $u = M_{ij}$ (edge $j \to i$), $v = M_{ji}$ (edge $i \to j$), and
  $\rho = \hat\Sigma_{jj}/\hat\Sigma_{ii}$ ($= 1$ on the correlation scale).

---

## 2. Part I — the two orientations are nearly interchangeable for the loss

### Lemma 1 (exact flat directions of the direct loss)

For every skew-symmetric $W$ and every $M$: $L(M + W\hat\Sigma^{-1}) = L(M)$.
The null space of $\mathcal A$ is exactly $\{W\hat\Sigma^{-1} : W^\top = -W\}$, of dimension $p(p-1)/2$.

*Proof.* $\mathcal A(W\hat\Sigma^{-1}) = W\hat\Sigma^{-1}\hat\Sigma + \hat\Sigma(W\hat\Sigma^{-1})^\top = W + \hat\Sigma\hat\Sigma^{-1}W^\top = W + W^\top = 0$.
The map $W \mapsto W\hat\Sigma^{-1}$ is injective, so this gives a $p(p-1)/2$-dimensional subspace of the null
space; since $\mathcal A$ maps onto the symmetric matrices (the Lyapunov equation is solvable for every
symmetric right-hand side when $\hat\Sigma \succ 0$), the null space has dimension exactly
$p^2 - p(p+1)/2 = p(p-1)/2$. $\square$

*Remarks.* (i) This is the non-identifiability of $M$ from $\Sigma$ without sparsity that the
identifiability paper of Dettling et al. starts from: every $\Sigma$ is compatible with a
$\binom p2$-dimensional affine space of drift matrices, and sparsity is what selects one.
(ii) For $W = E_{ij} - E_{ji}$ the null direction $N = W\hat\Sigma^{-1}$ has row $i$ equal to
$(\hat\Sigma^{-1})_{j\cdot}$ and row $j$ equal to $-(\hat\Sigma^{-1})_{i\cdot}$. Moving along it by
$t = -M_{ij}/(\hat\Sigma^{-1})_{jj}$ deletes the edge $j \to i$, creates $M_{ji} + M_{ij}(\hat\Sigma^{-1})_{ii}/(\hat\Sigma^{-1})_{jj}$
in the other direction, and leaves the loss *unchanged*; the price is a leakage into the other entries of
rows $i$ and $j$ proportional to the partial correlations of $i$ and $j$ with the remaining nodes. When
those are zero, reversing an edge (with the right weight) is exactly free.

### Lemma 2 (exact gradient and Hessian in the pair)

With all other entries of $M$ held fixed,

$$
\nabla^2_{(u,v)} L \;=\; 2\begin{pmatrix}\hat\Sigma_{jj}^2 & \hat\Sigma_{ii}\hat\Sigma_{jj}\\ \hat\Sigma_{ii}\hat\Sigma_{jj} & \hat\Sigma_{ii}^2\end{pmatrix}
\;+\; 2\begin{pmatrix} a_j & 0\\ 0 & a_i\end{pmatrix},
\qquad a_j = \sum_{k\neq j}\hat\Sigma_{jk}^2 + \hat\Sigma_{ij}^2,\quad a_i = \sum_{k \neq i}\hat\Sigma_{ik}^2 + \hat\Sigma_{ij}^2,
$$

$$
\nabla_{(u,v)} L \;=\; 2\,R_{ij}\,\big(\hat\Sigma_{jj},\ \hat\Sigma_{ii}\big) \;+\; 2\Big(\sum_{b\neq j} R_{ib}\hat\Sigma_{bj},\ \sum_{b \neq i} R_{jb}\hat\Sigma_{bi}\Big),
$$

where $R = R(M)$ is the (symmetric) residual at the current $M$.

*Proof.* $\partial R/\partial u = E_{ij}\hat\Sigma + \hat\Sigma E_{ji} =: A^{(ij)}$, the matrix whose row $i$
is $\hat\Sigma_{j\cdot}$, whose column $i$ is $\hat\Sigma_{\cdot j}$, and whose $(i,i)$ entry is $2\hat\Sigma_{ij}$.
Then $\partial L/\partial u = \langle R, A^{(ij)}\rangle = \sum_b R_{ib}\hat\Sigma_{jb} + \sum_a R_{ai}\hat\Sigma_{aj} = 2(R\hat\Sigma)_{ij}$,
and splitting off the $b = j$ term gives the stated form. For the Hessian,
$H_{uu} = \lVert A^{(ij)}\rVert_F^2 = \lVert E_{ij}\hat\Sigma\rVert^2 + \lVert\hat\Sigma E_{ji}\rVert^2 + 2\,\mathrm{tr}(\hat\Sigma E_{ji}\hat\Sigma E_{ji})
= 2(\hat\Sigma^2)_{jj} + 2\hat\Sigma_{ij}^2 = 2\hat\Sigma_{jj}^2 + 2a_j$,
and $H_{uv} = \langle A^{(ij)}, A^{(ji)}\rangle$ has four terms, of which
$\mathrm{tr}(\hat\Sigma E_{ji}E_{ji}\hat\Sigma) = 0$, $\mathrm{tr}(\hat\Sigma E_{ji}\hat\Sigma E_{ij}) = \hat\Sigma_{ii}\hat\Sigma_{jj}$,
$\mathrm{tr}(E_{ij}\hat\Sigma E_{ji}\hat\Sigma) = \hat\Sigma_{ii}\hat\Sigma_{jj}$, $\mathrm{tr}(E_{ij}\hat\Sigma^2 E_{ij}) = 0$,
so $H_{uv} = 2\hat\Sigma_{ii}\hat\Sigma_{jj}$. $\square$ (Checked against finite differences on a random
$6\times 6$ SPD matrix: agreement to $10^{-6}$.)

**Corollary 2.1 (correlation scale).** With $\hat\Sigma_{ii} = \hat\Sigma_{jj} = 1$,
$\nabla^2_{(u,v)}L = 2\binom{1\ 1}{1\ 1} + 2\,\mathrm{diag}(a_j, a_i)$. The curvature in the *presence*
direction $(1,1)/\sqrt2$ is $4 + a_i + a_j$; in the *reversal* direction $(1,-1)/\sqrt2$ it is $a_i + a_j$,
a sum of squared correlations of $i$ and $j$ with the other nodes. The ratio
$(a_i + a_j)/(4 + a_i + a_j)$ is how much weaker the loss's grip on the direction is than on the presence.

**Remarks.** (i) The rank-one term says that, apart from the correction, $L$ depends on $(u,v)$ only through
$\hat\Sigma_{jj}u + \hat\Sigma_{ii}v$. The **reversal line** $\{\hat\Sigma_{jj}u + \hat\Sigma_{ii}v = \text{const}\}$
is where $j \to i$ is traded for $i \to j$ at the exchange rate $\rho = \hat\Sigma_{jj}/\hat\Sigma_{ii}$.
(ii) The Hessian is constant (the loss is quadratic), so $a_i, a_j$ are the same at the sparse and at the
dense end of the path. What differs between the ends is the residual $R$, i.e. the gradient.
(iii) Letting the diagonal re-optimise (profiling $M_{ii}, M_{jj}$) subtracts a Schur-complement term of the
same order as $a_i + a_j$ and only lowers the curvature along the reversal line further.

### Lemma 3 (weak-coupling expansion; leading order)

Write $M = D + B$ with $D = \mathrm{diag}(d_1,\dots,d_p)$ and $B$ off-diagonal, $C = 2I$, and expand
$\Sigma(D + B) = \Sigma_0 + \Sigma_1 + \Sigma_2 + O(\lVert B\rVert^3)$. Then
$\Sigma_0 = \mathrm{diag}(\sigma_i)$ with $\sigma_i = -1/d_i$, $(\Sigma_1)_{ii} = 0$, and for $i \ne j$

$$
(\Sigma_1)_{ij} \;=\; \frac{d_i B_{ij} + d_j B_{ji}}{d_i d_j\,(d_i + d_j)} ,
\qquad
(\Sigma_2)_{ab} \;=\; -\,\frac{(B\Sigma_1)_{ab} + (\Sigma_1 B^\top)_{ab}}{d_a + d_b} .
$$

*Proof.* Insert the expansion into $(D+B)\Sigma + \Sigma(D+B)^\top + 2I = 0$ and collect orders:
$D\Sigma_0 + \Sigma_0 D + 2I = 0$ gives $\Sigma_0$; $D\Sigma_1 + \Sigma_1 D + B\Sigma_0 + \Sigma_0 B^\top = 0$
gives, entrywise, $(d_i + d_j)(\Sigma_1)_{ij} = -(B_{ij}\sigma_j + \sigma_i B_{ji})$, which is the stated
formula after substituting $\sigma = -1/d$; the diagonal of $B\Sigma_0 + \Sigma_0 B^\top$ vanishes because
$B_{ii} = 0$. The second-order equation is $D\Sigma_2 + \Sigma_2 D + B\Sigma_1 + \Sigma_1 B^\top = 0$. $\square$

**Corollary 3.1 (first-order non-identifiability of direction).** $\Sigma_1$ depends on $B$ only through the
symmetric matrix $d_iB_{ij} + d_jB_{ji}$. Hence the model with the edge $j \to i$ of weight $w$ and the
model with the reversed edge $i \to j$ of weight

$$ w' = w\,\frac{d_i}{d_j} = w\,\frac{\sigma_j}{\sigma_i} \qquad (i = \text{child of the true edge},\ j = \text{parent}) $$

have the same covariance up to $O(\lVert B\rVert^2)$. Presence of an edge is a first-order effect on $\Sigma$;
its direction is a second-order one.

**Corollary 3.2 (the two-node case is exactly unidentifiable).** For $p = 2$ a single-edge GCLM has three
parameters $(d_1, d_2, w)$ and $\Sigma$ has three entries. Solving the reversed model's equations shows that
for every covariance generated by $1 \to 2$ there is a stable $(d_1', d_2', w')$ with $2 \to 1$ reproducing it
exactly, with $w'/w \to d_2/d_1$ as $w \to 0$. (Numerically: $d = (-2,-3)$, $w = 0.02$ gives $w' = 0.0300$;
$w = 0.4$ gives $w' = 0.591$ against the first-order value $0.600$.) So the direction of an isolated edge is
identified only through third nodes.

**Corollary 3.3 (where direction lives: the three-node motifs).** For nodes $a, k, b$ with edges only between
$a$–$k$ and $k$–$b$ (weights $w_2$ on the $a$–$k$ edge, $w_1$ on the $k$–$b$ edge), Lemma 3 gives for the
covariance between the two outer nodes

| motif | $(\Sigma_2)_{ab}$ |
|---|---|
| chain $b \to k \to a$ | $-\dfrac{w_1 w_2}{d_b\,(d_k + d_b)\,(d_a + d_b)}$ |
| fork $a \leftarrow k \to b$ | $-\dfrac{w_1 w_2}{d_k\,(d_a + d_b)}\Big(\dfrac{1}{d_k + d_b} + \dfrac{1}{d_a + d_k}\Big)$ |
| collider $a \to k \leftarrow b$ | $0$ (exactly, at every order: row $a$ and row $b$ of $M$ are diagonal) |

(All three checked numerically to 10 digits.) These are the GCLM analogues of the DAG facts "parents of a
collider are marginally independent" and "chain and fork are Markov equivalent but differ in the
coefficients". They are the second-order information an estimator must use to orient an edge, and they
involve the *other* edges at $i$ and $j$: orienting an edge needs its neighbours to be in the fit.

**Corollary 3.4 (standardized scale, matching $C$).** Standardizing maps $M_{ij} \mapsto \tilde M_{ij} = M_{ij}D_j/D_i$
with $D_i = \Sigma_{ii}^{1/2}$. Under Corollary 3.1, $\tilde w = w\,(\sigma_j/\sigma_i)^{1/2}$ and
$\tilde w' = w'(\sigma_i/\sigma_j)^{1/2} = w\,(\sigma_j/\sigma_i)^{1/2} = \tilde w$. On the correlation
scale with the matching $C$ the two first-order-equivalent orientations have *equal* weights: the exchange rate
$\rho$ of Lemma 2 is 1, and any $\ell_1$-type tie-breaker between them disappears. On the raw scale the
lasso prefers the orientation with the smaller weight, i.e. the one whose parent has the larger stationary
variance (the var-sortability of §2 of the memo; in the thesis DGP this is the true orientation for about
65 % of the edges).

**Numerical illustration (three-node chain).** $1 \to 2 \to 3$, $d = (-2,-3,-4)$, both weights $0.5$,
$C = 2I$, population covariance. Best fit with $1 \to 2$ reversed (keeping $2 \to 3$): loss $1.5\cdot10^{-4}$.
Best fit with $1 \to 2$ deleted: loss $3.0\cdot10^{-2}$. The direction signal is 200 times weaker than
the presence signal at this weight.

---

## 3. Part II — the path keeps the orientation that entered first

### Proposition 4 (which orientation enters first)

On the sparse → dense path, let $M^{(0)}$ be the diagonal-only fit (the solution at $\lambda_{\max}$) and
$R^{(0)}$ its residual. A coordinate enters when $\lvert\partial L/\partial M_{ij}(M^{(0)})\rvert$ exceeds
$\lambda$, for the lasso and for MCP/SCAD alike (all three penalties have slope $\lambda$ at zero). By Lemma 2,

$$
\frac{\partial L}{\partial u} = 2R^{(0)}_{ij}\hat\Sigma_{jj} + \varepsilon_{ij},\qquad
\frac{\partial L}{\partial v} = 2R^{(0)}_{ij}\hat\Sigma_{ii} + \varepsilon_{ji},
$$

with $\varepsilon_{ij} = 2R^{(0)}_{ii}\hat\Sigma_{ij} + 2\sum_{b \ne i,j}(d_i + d_b)\hat\Sigma_{ib}\hat\Sigma_{bj}$
(using $R^{(0)}_{ib} = (d_i + d_b)\hat\Sigma_{ib}$ for $b \ne i$). In the population with weak edges,
$R^{(0)}_{ij} = O(w)$, the two-step term is $O(w^2)$ (products of two correlations along a path
$i - b - j$: this is exactly where the motif information of Corollary 3.3 sits), and
$R^{(0)}_{ii}\hat\Sigma_{ij} = O(w^3)$. Hence

$$
\frac{\lvert\partial L/\partial u\rvert}{\lvert\partial L/\partial v\rvert} = \frac{\hat\Sigma_{jj}}{\hat\Sigma_{ii}}\,\big(1 + O(w)\big).
$$

**Reading.** To leading order the orientation that enters first is the one whose *parent* has the larger
variance, whatever the true direction. On the raw scale this is the variance ordering (right 65 % of the
time in the thesis DGP). On the correlation scale the leading terms tie exactly and the entry is decided by
the $O(w)$ relative correction, i.e. by second-order signal plus sampling noise of order $n^{-1/2}$ in the
correlations.

### Proposition 5 (the barrier)

Let $(u^*, 0)$ be the solution just after $u$ has entered (so $v = 0$), and parametrise the reversal line
through it by

$$ u(t) = (1-t)\,u^*, \qquad v(t) = t\,\rho\,u^*, \qquad t \in [0,1], $$

so that $\hat\Sigma_{jj}u(t) + \hat\Sigma_{ii}v(t) = \hat\Sigma_{jj}u^*$ is constant and the endpoints are the two
orientations with first-order-equivalent weights. Write the penalised objective along the line as
$F(t) = \ell(t) + P_\lambda(\lvert u(t)\rvert) + P_\lambda(\lvert v(t)\rvert)$ with $\ell(t) = L(u(t), v(t))$.

**(a) Flat case.** If $a_i = a_j = 0$ (Lemma 2), $\ell$ is constant on the line and
$F(t) - \ell = P_\lambda((1-t)\lvert u^*\rvert) + P_\lambda(t\rho\lvert u^*\rvert)$ is **concave** in $t$ for MCP and
SCAD (a sum of concave functions of affine arguments). A concave function on $[0,1]$ attains its minimum at an
endpoint and lies above its chord in between. Its right derivative at $t = 0$ is

$$ F'(0^+) = \lvert u^*\rvert\,\big(\rho\,P'_\lambda(0^+) - P'_\lambda(\lvert u^*\rvert)\big) = \lvert u^*\rvert\,\big(\rho\lambda - P'_\lambda(\lvert u^*\rvert)\big), $$

which is $> 0$ for MCP whenever $\lvert u^*\rvert > 0$ and $\rho \ge 1$ (since $P'_\lambda(\lvert u\rvert) < \lambda$ for
$\lvert u\rvert > 0$), and equals $\rho\lambda\lvert u^*\rvert > 0$ once $\lvert u^*\rvert \ge \gamma\lambda$. So
$(u^*, 0)$ is a strict local minimum along the line, and symmetrically so is $(0, \rho u^*)$; between them is a
hump. If moreover $\lvert u^*\rvert \ge \gamma\lambda(1 + 1/\rho)$, then at $t_1 = \gamma\lambda/(\rho\lvert u^*\rvert)$
both $\lvert v(t_1)\rvert = \gamma\lambda$ and $\lvert u(t_1)\rvert \ge \gamma\lambda$ are in the saturated regime, so
$F(t_1) - F(0) = \gamma\lambda^2/2$: the hump is at least one full saturated penalty.

**(b) General case.** With $a_i + a_j > 0$ the loss adds to $F$ the convex term
$\ell(t) = \ell(0) + \ell'(0)\,t + \tfrac12\,\kappa\,t^2$ with $\kappa = 2(a_j + \rho^2 a_i)\,u^{*2}$ (from Lemma 2).
This tilts the hump toward the orientation with the lower loss, but it removes the barrier only when the
loss difference between the endpoints, $\lvert\ell(1) - \ell(0)\rvert$, exceeds the height of the hump. By Lemma 3
the loss difference between the two orientations is second order in the edge weights, while the hump is of
order $\lambda\lvert u^*\rvert$ (first order in the entered coefficient, and $\gamma\lambda^2/2$ in the saturated regime).
At the sparse end of the path the loss difference is the smaller of the two.

**(c) Persistence along the path.** The argument applies at every $\lambda$ separately. So at each later
(smaller) $\lambda$ the configuration with $v = 0$ remains a strict local minimum along the reversal line,
with a barrier toward $u = 0$. The repo's monotone APG and the coordinate-descent solver are descent methods:
warm-started inside this basin they converge to a stationary point in it and cannot cross a hump. The
orientation that entered first is therefore frozen for the rest of the path. $\square$

**(d) Lasso.** For $P_\lambda(t) = \lambda t$ the flat-case restriction is $F(t) - \ell = \lambda\lvert u^*\rvert\,(1 - t + \rho t)$,
*linear* in $t$: no hump. The minimum is at the endpoint with the smaller weighted $\ell_1$ norm (the orientation
whose parent has the larger variance), and on the correlation scale ($\rho = 1$) the whole segment is optimal,
which is the lasso's hedging between the two directions. Since the lasso problem is convex, its solution does not
depend on the warm start at all, so it cannot have the lock-in.

**(e) Why larger $\gamma$ helps only a little.** As $\gamma \to \infty$ the MCP penalty tends to the lasso, the
concavity and the hump shrink, and lock-in weakens; but the penalty also loses its unbiasing effect. E5 in the
memo shows exactly this: the deficit of the sparse → dense path shrinks with $\gamma$ but no $\gamma$ reaches the
lasso at $n = 10^3$.

### What the dense → sparse path does differently

It starts at the lasso solution at $\lambda_{\min}$, which is close to the least-squares fit: the residual
$R$ is nearly zero, so the first-order gradient of Proposition 4 plays no role. Both $u$ and $v$ are present
with all other edges in the fit. As $\lambda$ grows, the concave penalty forces a decision between the two
endpoints of the reversal line, and the only thing that tilts the hump is the loss difference, i.e. the
second-order information of Corollary 3.3 evaluated with the neighbours already fitted. The barrier of
Proposition 5 still exists at every $\lambda$; the dense → sparse path merely makes the choice at the point
where the loss, not the variances, decides it. The same holds for LLA from the lasso and for the adaptive
lasso with dense weights, which is why those estimators behave like `mcp_up` in the bake-off.

### Numerical illustration (thesis DGP, $p = 10$, $k = 2$, first drift matrix, population, `variance` scale)

For each single-direction true edge $j \to i$, two MCP ($\gamma = 3$) solves at the same $\lambda$ from two starts:
A = least squares on the true support, B = least squares on the support with that edge reversed. Both converge
to stationary points (stationarity $3\cdot10^{-10}$). "Hump" is the maximum of the objective on the straight segment
between the two solutions minus the larger endpoint value.

| edge | $\lambda$ | solution A $(M_{ij}, M_{ji})$ | solution B $(M_{ij}, M_{ji})$ | objective A | objective B | hump |
|---|---|---|---|---|---|---|
| $2 \leftarrow 3$ | 0.093 | $(-0.967,\ 0)$ | $(0,\ -0.758)$ | 0.23316 | 0.26438 | $7.4\cdot10^{-3}$ |
| $2 \leftarrow 3$ | 0.377 | $(-0.911,\ 0)$ | $(0,\ -0.702)$ | 2.51958 | 2.61988 | $2.1\cdot10^{-2}$ |
| $1 \leftarrow 8$ | 0.093 | $(0.717,\ 0)$ | $(0,\ 0.911)$ | 0.23316 | 0.24493 | $1.3\cdot10^{-2}$ |
| $2 \leftarrow 8$ | 0.093 | $(0.189,\ 0)$ | $(0,\ 0.160)$ | 0.23316 | 0.23790 | $4.4\cdot10^{-4}$ |
| $3 \leftarrow 1$ | 0.093 | $(0.934,\ 0)$ | $(0.917,\ 0)$ | 0.23316 | 0.25049 | $4.8\cdot10^{-3}$ |

Reading: at the same $\lambda$ the two orientations are two distinct stationary points of the same objective, the
wrong one has the higher objective (by 2–13 %), and there is a hump on the segment between them, of the order of
$\gamma\lambda^2/2 = 0.013$ at $\lambda = 0.093$ as Proposition 5(a) predicts for saturated coefficients. For
$3 \leftarrow 1$ the reversed start returned to the true orientation but to a different, worse stationary point
(the leakage of Lemma 1, remark (ii), had moved other entries). A hump on the straight segment shows that the
two points are separated along that path; it does not by itself exclude a lower saddle elsewhere, which is why
the proof goes through the concavity on the reversal line rather than through this table.

---

## 4. Empirical evidence from the study

`max_f1` of the same MCP objective ($\gamma = 3$) computed in the two directions, with the lasso for reference
(paired $z$ against the lasso in brackets; `variance` scale, $p = 10$, $k = 1,\dots,4$).

| DGP | path | $n = 10^3$ | $10^4$ | $10^5$ | $\infty$ |
|---|---|---|---|---|---|
| thesis (100 matrices) | lasso | 0.621 | 0.667 | 0.684 | 0.688 |
| | MCP sparse → dense | 0.553 (−6.6) | 0.595 (−6.6) | 0.603 (−7.4) | 0.612 (−6.9) |
| | MCP dense → sparse | 0.646 (+2.1) | 0.734 (+6.3) | 0.773 (+8.3) | 0.778 (+7.1) |
| strong signals, no 2-cycles (52) | lasso | 0.622 | 0.665 | — | 0.686 |
| | MCP sparse → dense | 0.581 (−2.8) | 0.588 (−4.8) | — | 0.593 (−5.7) |
| | MCP dense → sparse | 0.662 (+2.8) | 0.781 (+6.5) | — | 0.820 (+8.8) |

Further checks in the memo: (i) the two solvers (coordinate descent, monotone APG) give the same numbers in
each direction (within 0.003 per dataset on average for dense → sparse), so the effect is the path, not the
solver; (ii) at the same $\lambda$ the dense → sparse solution has the *lower* penalised objective at 64–67 % of
the 60 largest $\lambda$'s (strong-signal DGP) and the higher one at 20–26 %, i.e. the sparse → dense path is
usually at a worse local minimum of the same objective; (iii) MCP started at the truth reaches `max_f1` 0.999
on the `variance` scale at $n = \infty$ (E1), so the objective's minimum near the truth is fine and the path is
what misses it; (iv) the deficit of the sparse → dense path does not shrink with $n$, as a path effect should
not; (v) the $\gamma$ sweep (E5) behaves as Proposition 5(e) predicts.

---

## 5. Consequences for the thesis

- "MCP/SCAD" without a path direction is not a well-defined estimator for this problem; the thesis should name
  the direction, and the natural definition of the nonconvex estimator is dense → sparse (or LLA from the
  lasso), as Varando & Hansen already do for their likelihood path.
- The sparse → dense result of the pilot is a statement about the continuation algorithm, not about the
  penalty: the same objective, started elsewhere, beats the lasso.
- On the correlation scale with the matching $C$, the entry-order tie of Proposition 4 means the sparse → dense
  path orients edges by second-order signal plus noise and then locks them. This is why its orientation
  accuracy is at chance level for weakly identified edges (memo §3) and why abstaining (memo §12) helps the
  committing estimators.
- The loss's exact null space (Lemma 1) is why the curvature condition of Loh & Wainwright ($1/\gamma$ below the
  restricted-strong-convexity constant) cannot hold here in the reversal directions: their guarantee that all
  stationary points are good is unavailable, and multiple stationary points (the two orientations) are the norm.

## 6. What is proved and what remains

Proved as stated: Lemma 1, Lemma 2 and Corollaries 2.1 (exact); Lemma 3 and Corollaries 3.1, 3.3 (leading
order in the weights, with the error terms as stated); Corollary 3.2 (exact for $p = 2$); Proposition 5(a), (d)
(exact in the flat case). Leading-order or conditional: Proposition 4 (the $O(w)$ relative correction),
Proposition 5(b)–(c) (lock-in holds when the loss difference between the orientations is below the hump; the
comparison of scales is leading order, not a bound). To make a thesis lemma out of it one would (i) state
Proposition 5 with an explicit inequality: lock-in at $\lambda$ whenever $\lvert\ell(1) - \ell(0)\rvert < h(\lambda, u^*, \rho, \gamma)$
for the hump height $h$ computed from the concave function in (a); (ii) bound $\lvert\ell(1) - \ell(0)\rvert$ using
Lemma 2 (it is at most $\tfrac12\kappa + \lvert\ell'(0)\rvert$ with $\kappa = 2(a_j + \rho^2 a_i)u^{*2}$) and, in the
population, Lemma 3; (iii) add the sampling error of the correlations for finite $n$; (iv) handle several
edges entering between two grid points (the argument above is for one pair at a time with the rest fixed);
(v) say precisely which descent methods are covered (monotone APG and cyclic coordinate descent both decrease
the objective monotonically, which is all that (c) uses).

## 7. References

| reference | used for | status |
|---|---|---|
| Dettling, Homs, Améndola, Drton, Hansen, *Identifiability in continuous Lyapunov models*, [arXiv:2209.03835](https://arxiv.org/abs/2209.03835) (SIAM J. Matrix Anal. Appl. 2023) | the exact null space of the Lyapunov map; non-identifiability of $M$ without sparsity | abstract and existence verified; the exact statement of the null space should be cited from the paper |
| *Identifiability and Estimation in Continuous Lyapunov Models*, [arXiv:2603.17142](https://arxiv.org/abs/2603.17142) (2026) | possibly related | exists; not read |
| Breheny & Huang 2011, *Coordinate descent algorithms for nonconvex penalized regression*, Ann. Appl. Stat., [arXiv:1104.2748](https://arxiv.org/abs/1104.2748) | local convexity diagnostics along the $\lambda$ path; CD lands in different local minima | existence verified |
| Mazumder, Friedman, Hastie 2011, *SparseNet*, JASA | warm starts from the lasso toward MCP to avoid bad local minima | from memory |
| Wang, Kim, Li 2013, *Calibrating nonconvex penalized regression in ultra-high dimension*, Ann. Statist. | $\lambda$-continuation from $\lambda_{\max}$ can miss the oracle solution | from memory |
| Fan, Xue, Zou 2014, *Strong oracle optimality of folded concave penalized estimation*, Ann. Statist. | LLA from a good initial estimator reaches the oracle in one or two steps | from memory |
| Loh & Wainwright 2015, *Regularized M-estimators with nonconvexity*, JMLR | all stationary points are good only under restricted strong convexity with constant $> 1/\gamma$ | from memory |
| Varando & Hansen 2020, *Graphical continuous Lyapunov models*, UAI | the increasing-$\lambda$ (dense → sparse) sequence works better for their likelihood path | in the project files |
| Dettling, master's thesis, Theorem 3 | variance ordering and the lasso's orientation on the raw scale | see the note below |

The weak-coupling expansion (Lemma 3), the rank-one Hessian (Lemma 2) and the barrier argument
(Proposition 5) are from this study; I have not seen them in the literature, but I did not search exhaustively.

## 8. Note on the memo

Memo §8 quotes Dettling's Theorem 3 as "$d_i < d_j$ along every edge $i \to j$". Corollary 3.4 says the lasso on
the raw scale prefers the true orientation when the *parent* has the larger stationary variance, i.e.
$\lvert d_{\text{parent}}\rvert < \lvert d_{\text{child}}\rvert$, which is what Example 2's diagonal $(-2,\dots,-6)$ satisfies along
the path $1 \to 2 \to 3 \to 4 \to 5$. The memo's inequality should be read in absolute values; check the exact
form and the edge convention in the thesis before citing it.

## Appendix: the numerical checks (sandbox, `simulations/independent_study`)

- Lemma 2: finite-difference Hessian of `0.5*||M S + S M' + C||_F^2` in $(M_{ij}, M_{ji})$ on a random $6\times6$
  SPD $S$ against the formula: $(5.732464, 3.747887, 2.963732)$ vs $(5.732464, 3.747886, 2.963733)$.
- Lemma 1: loss change along $W S^{-1}$ for a random skew $W$: $1.4\cdot10^{-14}$.
- Corollary 3.2: `scipy.optimize.least_squares` on the reversed two-node model, residual $2\cdot10^{-9}$.
- Corollary 3.3: $(\Sigma_2)_{ab}$ from `solve_continuous_lyapunov` with weights $10^{-3}$ against the formulas,
  agreement to 10 digits; collider gives exactly 0.
- §3 table: `gx.draw_dettling(10, 2, 0, 0, inf)`, `gx.formulation(., "corrC")`, `fl.cd_solve(S, C, lam, "mcp", 3.0, m0=.)`,
  `fl.objective`, `fl.stationarity`.
