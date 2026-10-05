# Starting from the lasso: the rescaled $C$, dense → sparse paths, LLA, the adaptive lasso

How the estimators added for the campaign of October 2026 are defined, implemented and validated.
The reasons for them are in [`../simulations/VERDICTS.md`](../simulations/VERDICTS.md) (verdicts 3 to 7)
and in [`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md);
the theory behind "the standard path fixes a direction too early" is in
[`../next_steps/051026/orientation_lock_in.md`](../next_steps/051026/orientation_lock_in.md).

Code: `src/gclm/data/simulate.py` (`estimation_volatility`), `src/gclm/solvers/path.py`
(`lasso_path(direction="up")`, `lla_path`, `adaptive_lasso_path`), `src/gclm/objective/penalties.py`
(`lla_weights`). Tests: `tests/test_direction_cscale.py` (5), `tests/test_lla_adaptive.py` (16).

```bash
python simulations/run_s1.py --penalty MCP --direction up                    # MCP dense -> sparse
python simulations/run_s1.py --penalty SCAD --method lla --c-scale variance  # SCAD by LLA, rescaled C
python simulations/run_s1.py --method adaptive --c-scale variance            # adaptive lasso
```

All of this is for the **direct loss** with the default `fista` solver and an unpenalised diagonal.

---

## 1. The problem these estimators answer

The standard way to compute an MCP or SCAD path is continuation from the sparse end: start at
$\lambda_{\max}$ with the empty graph, lower $\lambda$, and start each problem from the previous
solution. In a GCLM this fixes the direction of an edge at the moment its pair first enters, when
the fit contains almost nothing else, and the nonconvex penalty then keeps that direction
(S3a; `orientation_lock_in.md`). The lasso is not affected in the same way: its problem is convex,
so it does not depend on where it was started, and when the direction is unclear it keeps both.

The three estimators below use the nonconvex penalty differently. They **start from a lasso
solution**, which still contains both directions of uncertain edges, and prune from there.

| estimator | start | then | convex steps | depends on the order of $\lambda$ |
|---|---|---|---|---|
| MCP / SCAD, sparse → dense (standard) | empty graph at $\lambda_{\max}$ | MCP / SCAD at each smaller $\lambda$, warm-started | no | yes |
| **MCP / SCAD, dense → sparse** | lasso solution at $\lambda_{\min}$ | MCP / SCAD at each larger $\lambda$, warm-started | no | yes |
| **MCP / SCAD by LLA** | lasso solution at the same $\lambda$ | two weighted lassos | yes | no |
| **adaptive lasso** | lasso solution at $\lambda_{\min}$ | one weighted lasso path | yes | no |

---

## 2. The rescaled $C$ (`--c-scale variance`)

**What it is.** The simulations standardise the data: every variable is divided by its standard
deviation $s_i$. That is a change of variables, and it changes the model:

$$M\Sigma+\Sigma M^\top+C=0
\quad\Longrightarrow\quad
\tilde M R+R\tilde M^\top+\tilde C=0,\qquad
\tilde M_{ij}=M_{ij}\,\frac{s_j}{s_i},\quad \tilde C=D^{-1}CD^{-1},\quad D=\mathrm{diag}(s).$$

$R$ is the correlation matrix. The support of $\tilde M$ is that of $M$. If the data were generated
with $C=2I$, then $\tilde C = 2\,\mathrm{diag}(1/s_i^2)$: the **rescaled $C$**.

**Code.** `draw_instance(..., return_scale=True)` returns $s$ next to the standardised covariance,
and

```python
estimation_volatility(scale, "identity")   # 2 I            -- Dettling's pipeline (default)
estimation_volatility(scale, "variance")   # 2 diag(1/s^2)  -- the rescaled C
```

(`src/gclm/data/simulate.py`) returns the matrix handed to the estimator. `S1Config.c_scale` and
`--c-scale` select it in both drivers. With unstandardised data $s=1$ and the two coincide.

**What it assumes.** That $C=2I$ on the scale on which the data were measured. That is exact for
Dettling's `C_ID` setting and only partly right for the other three (campaign note §2.2). It is a
device for the simulation study, not a recipe for real data.

**Test.** `test_standardized_population_model_is_misspecified_with_identity_and_exact_with_variance`:
at $n=\infty$ the true support fits the standardised covariance with zero loss under the rescaled
$C$ and with positive loss under $2I$, and $D^{-1}MD$ solves the rescaled equation.

---

## 3. MCP / SCAD dense → sparse (`--direction up`)

**Definition.** On the grid $\lambda_1<\dots<\lambda_{100}=\lambda_{\max}$:

1. $\hat M^{\text{lasso}}(\lambda_1)$: the lasso solution at the smallest $\lambda$. It is dense
   (close to the minimum-$\ell_1$ exact solution of the Lyapunov equation) and well defined,
   because the lasso is convex.
2. For $t=1,\dots,100$: solve the MCP / SCAD problem at $\lambda_t$ with the monotone accelerated
   proximal gradient of `docs/NONCONVEX.md` §4, started from the solution at $\lambda_{t-1}$
   (from the lasso start for $t=1$).
3. At $\lambda_{\max}$ the path is set to the diagonal fit, which is a stationary point there, so
   both orders end at the empty graph.

The objective is the same as for the standard path. Only the order differs, and with a nonconvex
penalty the order decides which stationary point is reached.

**Code.** `lasso_path(sigma, c, penalty="MCP", direction="up")` in `src/gclm/solvers/path.py`: the
block that begins with `if direction == "up" and penalty != "lasso":`. For the lasso `direction`
has no effect.

**Tests** (`tests/test_direction_cscale.py`): every point of the path is a stationary point of the
MCP objective; the last point is diagonal; the path differs from the standard one; the default is
unchanged; for the lasso both orders return the same path.

**Cost.** One lasso path plus an MCP / SCAD path that starts dense: 2 to 5 times the standard
MCP / SCAD path (measured, campaign note §3.4).

---

## 4. MCP / SCAD by local linear approximation (`--method lla`)

**Definition.** At every $\lambda$ separately, with $P_\lambda$ the MCP or SCAD penalty:

1. start from the lasso solution $M^{(0)}=\hat M^{\text{lasso}}(\lambda)$ at that same $\lambda$;
2. for $s=1,2$: compute weights from the current estimate and solve a weighted lasso,

$$w_{ij}=\frac{P'_\lambda(|M^{(s-1)}_{ij}|)}{\lambda}\in[0,1],\qquad
M^{(s)}=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}| .$$

The weight is 1 at a zero entry and 0 beyond $\gamma\lambda$, where the penalty is flat: large
entries are no longer shrunk, small ones are penalised like in the lasso. Each step replaces the
concave penalty by its tangent at the current estimate, which is a convex problem (Zou & Li 2008).

**Why two steps from the lasso.** Fan, Xue & Zou (2014): if the start is close enough to the truth
and the nonzero entries are large enough, the first step gives the oracle estimator and the second
step reproduces it; the lasso is a suitable start under a restricted-eigenvalue condition. The
irrepresentability condition, which the lasso needs for support recovery and which fails for most
of these graphs, is not required. A fixed point of the iteration is a stationary point of the
MCP / SCAD objective.

**Code.** `lla_path` in `src/gclm/solvers/path.py`; the weights are `lla_weights` in
`src/gclm/objective/penalties.py`; each weighted lasso is `solve_fista(..., weights=w)`, started
from the current estimate. Nothing is carried from one $\lambda$ to the next.

**A caveat at the dense end.** The design $A(\hat\Sigma)$ has rank $p(p+1)/2$. When more entries
than that are unpenalised (the diagonal plus every entry beyond $\gamma\lambda$), the weighted
lasso has flat directions and more than one minimiser; the estimate is then the one FISTA reaches
from the lasso solution. This concerns the smallest $\lambda$ only, where the fit is dense anyway.

**Tests** (`tests/test_lla_adaptive.py`): the weights equal $P'/\lambda$; one step satisfies the
optimality conditions of its weighted lasso at every $\lambda$ and agrees with coordinate descent
on the explicit design where the problem is well posed; iterated to a fixed point it is stationary
for the nonconvex objective; the estimate at one $\lambda$ does not depend on the rest of the grid;
it ends at the diagonal fit and tends to the lasso as $\gamma\to\infty$.

**Cost.** About three lasso paths.

---

## 5. Adaptive lasso (`--method adaptive`)

**Definition** (Zou 2006, with a lasso pilot):

1. pilot $M^0=\hat M^{\text{lasso}}(\lambda_{\min})$, the dense end of the lasso path;
2. weights $w_{ij}=1/|M^0_{ij}|$, divided by the smallest one, so that the largest pilot entry has
   weight 1; entries with $M^0_{ij}=0$ are excluded;
3. the weighted lasso path

$$\hat M(\lambda)=\arg\min_M\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{i\ne j}w_{ij}|M_{ij}|$$

on its own grid: 100 log-spaced values up to $\lambda^{w}_{\max}=\max_{i\ne j}|\nabla f(M_D)_{ij}|/w_{ij}$,
the smallest $\lambda$ at which the solution is the diagonal fit $M_D$.

It follows the same idea as the dense → sparse path (start dense, remove the weak entries first),
with a convex second step. It is the control for the question whether the nonconvex penalty is
needed, or only the dense start.

**Code.** `adaptive_lasso_path` in `src/gclm/solvers/path.py`. It returns an `AdaptiveLassoPath`,
which also holds the pilot and the weights (`inf` marks an excluded entry).

**Tests** (`tests/test_lla_adaptive.py`): the weights; the optimality conditions of the weighted
problem at every $\lambda$ and agreement with coordinate descent; excluded entries stay zero and
every support is contained in the pilot's; with equal weights it is the lasso path.

**Cost.** About two lasso paths.

---

## 6. On the cluster

`simulations/run_s1_shard.py` takes the same options (`--c-scale`, `--direction`, `--method`).
With `--select bic` or `--select search` it also records, for each graph, the support at every
$\lambda$, the graph that BIC selects on the path and the graph the BIC search of
[`SEARCH.md`](SEARCH.md) reaches from it. `cluster/submit_campaign.sh` defines the cells of the
campaign. Details, the list of cells and the commands: campaign note §4 and §5, and
[`REPRODUCTION.md`](REPRODUCTION.md) §2.7.

---

## References

- Fan, J., Xue, L. & Zou, H. (2014). Strong oracle optimality of folded concave penalized
  estimation. *Annals of Statistics* 42(3), 819–849. LLA from a lasso start.
- Zou, H. & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models.
  *Annals of Statistics* 36(4), 1509–1533. The local linear approximation.
- Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA* 101(476), 1418–1429.
- Varando, G. & Hansen, N. R. (2020). Graphical continuous Lyapunov models. *UAI*. Their paths run
  from a dense start towards sparser solutions.
- Dettling, P., Drton, M. & Kolar, M. (2024). On the Lasso for graphical continuous Lyapunov
  models. *CLeaR*. The direct loss and the simulation design.
