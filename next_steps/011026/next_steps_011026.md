# Next steps — 1 October 2026

*Where the simulations stand, what the literature says about the pattern they found, which
remedies exist, and in which order to try them. Evidence: [`../../simulations/S2_penalties_losses.md`](../../simulations/S2_penalties_losses.md)
(§3.1–3.6); methods: [`../../docs/NONCONVEX.md`](../../docs/NONCONVEX.md), [`../../docs/LIKELIHOOD.md`](../../docs/LIKELIHOOD.md).*

## 1. Where we are

- **Result.** On Dettling's direct loss and on Varando's log-likelihood and Frobenius losses, MCP
  ($\gamma=3$) and SCAD ($\gamma=3.7$) recover the directed support *worse* than the lasso
  (`max_f1` −0.05 … −0.14, paired $z$ −4 … −18 on identical datasets, $p=10,20$). On the
  **undirected skeleton** they are close to the lasso (skeleton AUC / AUPR within 0.01–0.02 on the
  direct loss, 0.02–0.035 on the log-likelihood).
  Three quarters of the directed loss is **orientation**: the nonconvex penalties pick the wrong
  direction of an edge, and they cannot recover 2-cycles (one direction per pair).
- **Mechanism** (verified, including against an independent re-implementation). The design
  $A(\hat\Sigma)$ has a $p(p-1)/2$-dimensional null space, nearly the skew-symmetric matrices for
  correlation input, so $M_{ij}$ and $M_{ji}$ are almost interchangeable (loss curvature along the
  swap: median 0.75, vs. $\ge 2$ along a coordinate). The $\ell_1$ norm is constant along the swap,
  so the lasso lets the loss decide and **hedges** (keeps both directions, 4–5 pairs per dataset)
  until the data settle it. A penalty with a kink at zero and a flat part charges $\lambda$ on the
  entering direction and refunds nothing on the exiting one: a **barrier of
  $\min(|a|/\gamma, \lambda)$ against every reversal**, so whichever direction entered the path
  first is frozen. It is the kink plus the flat part, not the concavity (only 13 % of true edges
  have swap curvature below $1/\gamma$); it is bias, not variance (per-edge variance equal to the
  lasso's); it is not the optimiser (`skglm` agrees; the oracle basin is better but has the lower
  objective in only 42 % of datasets).
- **Clean case.** Example 2's 5-cycle: all three penalties return the correct skeleton with the
  edge 5→1 reversed; under MCP the truth is a stationary point with the lower objective, reachable
  by one reversal move. In the population the objective prefers the truth and continuation never
  finds it; at $n=1000$ it does not even reliably prefer it.

## 2. Is this known?

The ingredients are; the combination in Lyapunov models is not something I have seen written
down, and the orientation framing is new as far as I know.

**The ambiguity: edge direction is the weakly identified part of a Lyapunov model.**
The equation sees $M$ through $M\Sigma + \Sigma M^\top$; at $\Sigma = I$ only the symmetric part
$M + M^\top$ enters, so the skew part (which way an edge points) is identified only through the
off-diagonal correlations.
- Varando & Hansen (2020): the fibre $\{M + W\Sigma^{-1}\}$, $W$ skew.
- Dettling, Homs, Améndola, Drton & Hansen (2023), *Identifiability in continuous Lyapunov
  models*: when a sparse $M$ is identifiable from $\Sigma$ given the graph; 2-cycles are the
  problematic structure. **Verify the exact statement before citing** — my recollection is that
  simple graphs (no 2-cycles) are the generically identifiable class.
- Dettling, Drton & Kolar (2024), Example 2: the lasso fails irrepresentability on the 5-cycle.
  Our result sharpens "fails" to "reverses one edge".

**Nonconvex penalties under correlated designs: path dependence and lock-in** — documented, in
regression language:
- Breheny & Huang (2011): the *local convexity diagnostic* (`ncvreg`'s `convex.min`): the MCP /
  SCAD path is trustworthy only while $\lambda_{\min}(X_S^\top X_S) > 1/\gamma$ on the active set;
  beyond that the solution is a path-dependent local minimum. Our $\Gamma_{SS}$ eigenvalues reach
  0.33 ≈ $1/\gamma$, so their diagnostic flags exactly these paths.
- Mazumder, Friedman & Hastie (2011), *SparseNet*: under correlation the solution depends on the
  initialisation; they warm-start each $\lambda$ from the **lasso solution and decrease $\gamma$
  gradually**. The closest existing answer to "the first-entered direction is frozen".
- Zou & Li (2008), LLA; Fan, Xue & Zou (2014): the MCP / SCAD solution with the oracle property is
  the one reached from a good initialiser (usually the lasso), under conditions on that
  initialiser. Wang, Kim & Li (2013): the same point with a calibrated path.
- Theory: Zhang (2010) needs sparse-eigenvalue conditions; Loh & Wainwright (2015, 2017) need
  restricted strong convexity with $\alpha > 1/\gamma$, and their bounds scale with
  $1/(\alpha-\mu)$, which is loose when the swap curvature is 0.3–0.8.

**Why the lasso keeps both directions.** The behaviour the elastic net was built to strengthen:
Zou & Hastie (2005), the grouping effect (a ridge term forces near-identical columns to share
mass). Tibshirani (2013) on what the lasso does with near-collinear columns.

**Framing.** Fan & Li (2001) list unbiasedness, sparsity and *continuity* as the desirable
properties of a penalty. In a model where edge direction is nearly non-identifiable, continuity
is the property that decides support recovery, and it is the one folded-concave penalties give up.

## 3. Candidate remedies

| remedy | reference | what it does for the orientation barrier | cost |
|---|---|---|---|
| **Mnet / Snet**: MCP or SCAD + ridge | Huang, Breheny, Lee, Ma & Zhang (2016); `ncvreg(alpha < 1)` | the ridge rewards splitting mass across $M_{ij}$ / $M_{ji}$ — restores the hedge — while the concave part still debiases large entries | ~20 lines: add $(\alpha/2)\|M_{\text{off}}\|_F^2$ to the direct loss (gradient $+\alpha M_{\text{off}}$, Lipschitz $+\alpha$); or `--solver ncvreg` with `alpha` |
| **$\gamma$-continuation from the lasso** | Mazumder, Friedman & Hastie (2011) | the loss resolves the split while the penalty is still convex; concavity is added after the orientation is decided by the data | ~30 lines of warm starts over $\gamma$ (e.g. $\infty \to 100 \to 30 \to 10 \to 3$) per $\lambda$ |
| **LLA / one-step from the lasso** | Zou & Li (2008); Fan, Xue & Zou (2014) | one weighted lasso with weights $P'(\lvert\hat\beta_{\text{lasso}}\rvert)/\lambda$; keeps the hedge where the lasso hedged, fails where the lasso itself reversed (the 5-cycle) | trivial: `penalty_weights` already takes arbitrary weights for the lasso |
| **Score-based search with a *reverse* move** | hill-climbing with add / delete / reverse: Heckerman, Geiger & Chickering (1995); GES: Chickering (2002); cyclic: Améndola, Dettling, Drton, Onori & Wu (2020) — in `literature/`, **check which moves it uses** | attacks orientation directly; Example 2 shows one reversal move reaches the truth | this is S3 from `../../plan.md` with one extra move |
| **Skeleton first, orientation second** | pair-level group penalty: Yuan & Lin (2006) | penalise $\{M_{ij}, M_{ji}\}$ as a group (every penalty is good at the skeleton), then orient each pair by comparing the two fits; 2-cycles stay possible | a new estimator; moderate |
| **$\gamma$ sweep** of plain MCP / SCAD | — | the barrier is $\min(\lvert a\rvert/\gamma, \lambda)$: larger $\gamma$ keeps more entries in the refund zone and should approach the lasso from below; shows how much of the gap is $\gamma$ | script ready: `../../runs/s1b_gamma_sweep/run_gamma_sweep.sh`, ~40 min |
| **Convexity diagnostics** | Breheny & Huang (2011) | not a fix: report where on the path the solution is locally convex and compare only there | free |

Not worth running: the ncvreg convention. With $v_{ij} \ge 2$ it amounts to a textbook
$\gamma/v_{ij} \le 1.5$ for MCP, i.e. *more* concave.

## 4. Plan, in order

1. **Hold the large-$p$ cluster runs of the current cells.** The pilot settles $\gamma = 3 / 3.7$
   at $p = 10, 20$; $p = 50$ (230–280 CPU-h per direct-loss cell) would measure the same thing
   more precisely. Cluster time is better spent on steps 3–4.
2. **Measure orientation everywhere, by default.** `orientation_breakdown`,
   `skeleton_confusion` and `evaluate_skeleton_path` exist (`gclm.metrics`,
   `../../simulations/diagnostics/orientation.py`); add the skeleton metrics and the category counts to
   `run_s1_shard.py` / `aggregate_s1.py` so every future run reports them, and add a tuning-based
   comparison (BIC / eBIC at one $\lambda$) next to the path maxima, which are oracle tuning.
3. **Try the two cheap remedies at $p=10$ on `C_ID`** (~10 min each with the orientation
   diagnostic): $\gamma$-continuation from the lasso, and Mnet. Then the $\gamma$ sweep. Success
   criterion: skeleton metrics unchanged, orientation recall back to the lasso's (≈ 0.9), directed
   `max_f1` at or above the lasso.
4. **Target orientation directly: S3 with a reverse move.** Greedy search over add / delete /
   reverse with a BIC-type score, started from the lasso path (or from the skeleton of step 3).
   This advances the planned score-based study and is where the Example 2 positive control
   belongs.
5. **Map where nonconvexity pays off.** With the Example 2 harness: sweep $n$ ($10^3$ to
   $\infty$); random drift matrices with edge weights bounded away from zero; with and without
   2-cycles (the DGP draws each direction independently with probability $k/p$, so 10–40 % of
   true edges at $p=10$ have their reverse present — a structural handicap for any
   one-direction-per-pair penalty).
6. **Then** send the winning configuration to the cluster for the full $p$ range.

## 5. Questions for the supervisor

- Is "orientation is the hard part of GCLM structure learning, and the lasso's hedging is what
  makes it work" an acceptable reframing of the thesis question? The positive side would be
  the remedies of §3, with S3-with-reversal as the main contribution.
- Should 2-cycles stay in the data-generating process, given the identifiability results of
  Dettling et al. (2023)? If they are not identifiable, every method should be scored on simple
  graphs as well, or the metric should treat a 2-cycle as one undirected edge.
- Is a tuning-based comparison (BIC at one $\lambda$) required, or are Dettling's path-maximum
  metrics acceptable for the thesis?

## References

- Améndola, C., Dettling, P., Drton, M., Onori, F. & Wu, J. (2020). *Structure learning for
  cyclic linear causal models.* UAI 2020, PMLR 124.
- Breheny, P. & Huang, J. (2011). *Coordinate descent algorithms for nonconvex penalized
  regression, with applications to biological feature selection.* Ann. Appl. Stat. 5(1), 232–253.
- Chickering, D. M. (2002). *Optimal structure identification with greedy search.* JMLR 3,
  507–554.
- Dettling, P., Drton, M. & Kolar, M. (2024). *On the Lasso for graphical continuous Lyapunov
  models.* CLeaR 2024, PMLR 236.
- Dettling, P., Homs, R., Améndola, C., Drton, M. & Hansen, N. R. (2023). *Identifiability in
  continuous Lyapunov models.* SIAM J. Matrix Anal. Appl. 44(4).
- Fan, J. & Li, R. (2001). *Variable selection via nonconcave penalized likelihood and its
  oracle properties.* JASA 96(456), 1348–1360.
- Fan, J., Xue, L. & Zou, H. (2014). *Strong oracle optimality of folded concave penalized
  estimation.* Ann. Statist. 42(3), 819–849.
- Heckerman, D., Geiger, D. & Chickering, D. M. (1995). *Learning Bayesian networks: the
  combination of knowledge and statistical data.* Machine Learning 20, 197–243.
- Huang, J., Breheny, P., Lee, S., Ma, S. & Zhang, C.-H. (2016). *The Mnet method for variable
  selection.* Statistica Sinica 26, 903–923.
- Loh, P.-L. & Wainwright, M. J. (2015). *Regularized M-estimators with nonconvexity: statistical
  and algorithmic theory for local optima.* JMLR 16, 559–616.
- Loh, P.-L. & Wainwright, M. J. (2017). *Support recovery without incoherence: a case for
  nonconvex regularization.* Ann. Statist. 45(6), 2455–2482.
- Mazumder, R., Friedman, J. H. & Hastie, T. (2011). *SparseNet: coordinate descent with
  nonconvex penalties.* JASA 106(495), 1125–1138.
- Tibshirani, R. J. (2013). *The lasso problem and uniqueness.* Electron. J. Statist. 7,
  1456–1490.
- Varando, G. & Hansen, N. R. (2020). *Graphical continuous Lyapunov models.* UAI 2020, PMLR 124.
- Wang, L., Kim, Y. & Li, R. (2013). *Calibrating nonconvex penalized regression in ultra-high
  dimension.* Ann. Statist. 41(5), 2505–2536.
- Yuan, M. & Lin, Y. (2006). *Model selection and estimation in regression with grouped
  variables.* JRSS-B 68(1), 49–67.
- Zhang, C.-H. (2010). *Nearly unbiased variable selection under minimax concave penalty.* Ann.
  Statist. 38(2), 894–942.
- Zou, H. & Hastie, T. (2005). *Regularization and variable selection via the elastic net.*
  JRSS-B 67(2), 301–320.
- Zou, H. & Li, R. (2008). *One-step sparse estimates in nonconcave penalized likelihood models.*
  Ann. Statist. 36(4), 1509–1533.
