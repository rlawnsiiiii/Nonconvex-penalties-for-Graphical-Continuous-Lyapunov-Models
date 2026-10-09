# Nonconvex penalties for GCLMs — where the pilot leaves us, and what to do next

*Working notes, 2026-10-01, written after reviewing the S1b/S2 pilot. Suggested home in the
repo: `simulations/next_steps_2026-10-01.md` (companion to `S2_penalties_losses.md`).*

**Provenance tags used throughout**

- **[repo]**: numbers already committed in the repo (`S2_penalties_losses.md`,
  `S1_reproduction.md`, `docs/LIKELIHOOD.md`, `docs/NONCONVEX.md`).
- **[scratch]**: numbers from a quick, independent mini re-implementation (`scratch_probes.zip`,
  see Appendix B). It reproduces the repo exactly where checked:
  - Example 2 at $n=\infty$: lasso path 1.000/1.000, cycle 0.800/0.833.
  - The committed lasso `max_f1` for $p=10$, $k=1$, `C_ID`, reps 0–5.
  - The dataset $p=10$, $k=2$, `C_ID`, rep 0, with lasso 0.596 and MCP 0.638 (the numbers quoted in §3.1).

  Even so, regenerate every [scratch] number with the repo before it goes into the thesis.

---

## 0. Summary

- **The S1b result is real.** On the direct loss, textbook MCP ($\gamma=3$) and SCAD ($\gamma=3.7$)
  do worse than the lasso on `max_f1`, `auc` and `aupr`. This holds in every $C$ setting at
  $p=10,20$, with $z$ down to −18 [repo].
  - The checks behind it: paired design on identical datasets, the optimiser cross-checked with
    skglm, stationarity verified.
  - An independent re-implementation gives the same direction and size [scratch].
- **The loss is in orientation, not in the skeleton.** At each method's best-F1 point, MCP finds
  the undirected skeleton about as well as the lasso (skeleton F1 0.760 vs 0.785). But it almost
  never keeps both directions of a pair, so it loses two things [scratch]:
  - single edges that it orients the wrong way;
  - the second edge of every 2-cycle.
- **Mechanism.**
  - For each pair $(M_{ij},M_{ji})$ the data determine the *sum* well and the *difference*
    (orientation) poorly. The two design columns have median correlation 0.77 [scratch].
  - The lasso penalty is flat along the difference. The lasso therefore reports the data's split
    and keeps both directions when unsure.
  - MCP and SCAD are concave, so they amplify the split into an all-or-nothing choice. When the
    data are ambiguous, that choice is close to a coin flip.
  - Debiasing then acts on the poorly determined direction, trading bias for variance. This is
    what the repo's estimation-error table shows: shrinkage removed, error not lower.
- **Example 2 (Dettling's 5-cycle) is the cleanest case** [scratch].
  - The lasso's irrepresentability failure is a single reversed edge.
  - MCP/SCAD continuation inherits the same error.
  - Yet the truth is a lower-objective local minimum of the MCP problem, and one "reverse this
    edge" move reaches it.
- **Consequence for the thesis.** "Nonconvex beats lasso" does not hold out of the box for GCLMs.
  The reason is specific and explainable: orientation is weakly identified, and
  $\operatorname{rank}A(\hat\Sigma)=p(p+1)/2$. A positive story has to come from one of three places:
  - regimes where orientation is decisive (large $n$, strong edges, no 2-cycles);
  - estimators that handle orientation explicitly (reversal moves, i.e. S3; Mnet);
  - data-driven tuning.

---

## 1. What we know

### 1.1 S1b — direct loss [repo]

Source: `S2_penalties_losses.md` §3.1. Each cell is the paired difference (penalty − lasso), with
the range taken over the four $C$ choices and 100 datasets per point.

| | $p$ | `max_f1` | `auc` | `aupr` |
|---|---|---|---|---|
| MCP $\gamma=3$ | 10 | −0.12 … −0.07 | −0.12 … −0.07 | −0.08 … −0.04 |
| | 20 | −0.09 … −0.06 | −0.15 … −0.11 | −0.08 … −0.05 |
| SCAD $\gamma=3.7$ | 10 | −0.06 … −0.05 | −0.11 … −0.06 | −0.07 … −0.03 |
| | 20 | −0.04 … −0.02 | −0.12 … −0.09 | −0.07 … −0.04 |

- **Paths** ($p=20$, `C_ID`).
  - At the same λ index MCP is sparser: 99 vs 114 nonzeros at index 70, and 51 vs 63 at index 80.
  - Yet at matched FPR it has 10–20 points less recall.
  - All three penalties saturate at $p(p-1)/2=190$ nonzeros.
  - TPR at saturation: lasso 0.79, MCP 0.60, SCAD 0.63.
- **Nestedness.** MCP paths are more nested: 0.6–1.0 support reversals per path, against 6–10 for
  the lasso.
- **Estimation error at the best-F1 point.** MCP removes the shrinkage: on selected true edges,
  $|\hat M|/|M^*|$ is 1.03 vs 0.56 at $p=10$. But the relative error is not smaller (1.22 vs 1.05).

### 1.2 S2 — Varando's losses

- **Not written up yet.** §3.2, §3.3 and the S2 part of §4 are still empty. The pilot itself ran:
  the §5 convergence audit covers 16,000 λ's per cell, and the few that hit the step cap all have
  violation ≤ 1e−5 [repo].
- **Evidence so far, from LIKELIHOOD.md §4–5** [repo].
  - Under-converged solutions look *better*, because they stay closer to the sparse start.
  - Newton reaches lower-objective MCP points with much worse support (`auc` 0.47 vs 0.68 on one
    dataset). A lower objective is not a better estimate.
- **The start of the path carries no orientation information.** With correlation input and
  $C=2I$, the gradient at the diagonal fit is exactly $\hat\Sigma-I$. That matrix is symmetric, so
  both orientations of every pair enter with identical gradient.

### 1.3 Decomposing the S1b deficit [scratch]

Setup: $p=10$, `C_ID`, $k=1..4$ × reps 0–9, i.e. 40 datasets (the same draws as the repo).

**Paired differences.**

- MCP − lasso: `max_f1` −0.080 (se 0.017, $z$ −4.8); `auc` −0.085 ($z$ −6.6).
- SCAD − lasso: `max_f1` −0.044 ($z$ −4.0); `auc` −0.101 ($z$ −6.9).

**Breakdown at each method's best-F1 point** (means per dataset). On average a dataset has 15.0
single true edges and 3.55 two-cycles (7.1 edges), so about a third of all true edges sit in
2-cycles.

| | lasso | MCP | SCAD |
|---|---|---|---|
| `max_f1` | 0.622 | 0.542 | 0.578 |
| TP / FP / FN | 14.3 / 9.9 / 7.8 | 11.1 / 7.7 / 11.0 | 12.0 / 7.9 / 10.1 |
| skeleton F1 (unordered pairs) | 0.785 | 0.760 | 0.775 |
| single edges whose pair is found | 11.5 | 11.3 | 11.4 |
| … correct direction only | 5.6 (49 %) | 7.8 (69 %) | 7.4 (65 %) |
| … both directions (hedge) | 4.5 (39 %) | 0.3 (2 %) | 1.2 (11 %) |
| … reversed only | 1.5 (13 %) | 3.3 (29 %) | 2.8 (24 %) |
| 2-cycles (of 3.55): both edges found | 1.35 | 0.05 | 0.45 |
| 2-cycles: one edge found | 1.50 | 2.92 | 2.48 |
| false pairs (skeleton false positives) | 3.6 | 4.2 | 3.7 |
| pairs with both directions at the dense end | 9.2 | 0.4 | 0.2 |

How to read the table:

- **Same skeleton; the difference is inside the pair.** The lasso keeps both directions on 39 % of
  the found single edges. MCP does so on 2 %.
- **The lasso's hedges turn into near coin flips under MCP.** In aggregate, those 39 % become about
  +20 points "correct" and +16 points "reversed". This is an aggregate reading; the pair-matched
  cross-tab in Step 1 will confirm or refute it.
- **Where MCP's true positives go.** Its deficit of 3.2 true positives splits into about 2.0 from
  single edges and about 1.2 from 2-cycles. It almost never recovers both directions of a 2-cycle
  (0.05 of 3.55).
- **A metric correction.** My earlier chat number, "orientation accuracy 0.86 vs 0.68", counted
  hedges as correct. Use the strict split above instead.

**Other checks** [scratch]:

- **Curvature along the reversal direction** $V_{ij}=e_ie_j^\top-e_je_i^\top$.
  - Median ≈ 0.84 on true edges, against coordinate curvature $v_{ij}\ge 2$.
  - Only about 13 % of true edges have it below $1/\gamma=1/3$.
  - That fraction does not predict the per-dataset MCP deficit (correlation +0.23).
- **Design columns of $M_{ij}$ and $M_{ji}$.** Median correlation 0.77; per-dataset medians range
  from 0.36 to 0.94.
- **Oracle basin.** At MCP's best-F1 λ, MCP started from the least-squares fit on the true support
  reaches F1 0.674, against 0.542 for the continuation solution. But that basin has the *lower*
  MCP objective in only 45 % of datasets. So a better optimiser of the same objective would not
  systematically help, which is consistent with LIKELIHOOD.md §5.
- **Greedy reversal search** on the MCP objective at the best-F1 λ ($k=1,2$; 20 datasets). Flips
  were accepted on 60 % of datasets, and the F1 change averaged −0.004 (better on 20 %, worse on
  35 %).

### 1.4 Example 2 (5-cycle, $n=\infty$) [scratch]

**Headline numbers.** Lasso, MCP and SCAD all give `max_f1` 0.800 and `auc` 0.833. These equal the
M0 lasso numbers [repo].

**What goes wrong for the lasso**:

- The failure is **one reversed edge**. The lasso selects `M[4,0]` (1→5) instead of `M[0,4]` (5→1);
  in the repo's convention, edge $i\to j$ is `M[j, i]`. Skeleton F1 = 1.
- At the λ's inspected along the path, `M[0,4]` is exactly 0, while `M[4,0]` grows from
  λ/λ_max ≈ 0.19 down to the dense end.

**What MCP's objective prefers**:

- The true support is the **unique** support with ≤ 5 off-diagonal edges that fits Σ exactly
  (brute force over all supports of size 3–5).
- $M^*$ is a fixed point of the MCP solver at λ ∈ {0.02, 0.05, 0.1, 0.2}.
- MCP objective at $M^*$ vs at the continuation solution:

  | λ | at $M^*$ | at continuation solution |
  |---|---|---|
  | 0.02 | 0.00300 | 0.00326 |
  | 0.05 | 0.01875 | 0.01901 |
  | 0.1 | 0.0750 | 0.0863 |
  | 0.2 | 0.300 | 0.167 |

  So for λ ≤ 0.1 the truth is the lower of the two; at λ = 0.2 it is not.

**What fixes it and what doesn't**:

- One reversal move at λ = 0.05 (move `M[4,0]` to `M[0,4]`, then re-solve) lowers the objective to
  0.01875 and gives exactly the true support.
- One-step LLA (reweighted ℓ1) from the lasso does **not** fix it. The lasso gives the true
  direction zero weight, so reweighting cannot revive it.

**Why two orientations are both stable here.** The reversal curvature on the five cycle pairs is
0.002–0.12 (raw covariance scale), all below $1/\gamma$. So every pair has two stable orientations
(§2.5).

---

## 2. Why — material for the thesis text

### 2.1 $A(\hat\Sigma)$ has rank $p(p+1)/2$

$A(\Sigma)\operatorname{vec}(M)=\operatorname{vec}(M\Sigma+\Sigma M^\top)$, with $\Sigma\succ0$.

1. $M\Sigma+\Sigma M^\top$ is symmetric, so rank ≤ $p(p+1)/2$. This is the count in Dettling et
   al. (2024), App. B, Remark B.2: the Lyapunov equation has $(p+1)p/2$ equations for $p^2$
   unknowns, so $M$ needs extra structure such as sparsity.
2. Every symmetric $S$ is attained: $M=\tfrac12S\Sigma^{-1}$ gives $M\Sigma+\Sigma M^\top=S$. So the
   rank is exactly $p(p+1)/2$.
3. Null space: $M\Sigma+(M\Sigma)^\top=0$ iff $M\Sigma$ is skew-symmetric iff $M=W\Sigma^{-1}$ with
   $W^\top=-W$. Its dimension is $p(p-1)/2$.

This holds for every $\Sigma\succ0$, every $n$ and every $C$.

- In the repo it is derived in `S1_reproduction.md` §2.3 and tested by
  `test_design_matrix_is_rank_deficient` and `test_null_space_is_flat_for_every_C`.
- Varando's losses share the same fibre at the dense end (LIKELIHOOD.md §3).
- **Consequence.** $c_*=\lambda_{\min}(A^\top A)=0$. The MCP/SCAD objective is convex only if
  $\gamma>1/c_*$ (MCP) or $\gamma>1+1/c_*$ (SCAD) (Zhang 2010; Breheny & Huang 2011, §2.3).
  So it is **never globally convex** here, for any $\gamma$ or $n$; only local convexity on an
  active set is possible.

### 2.2 Orientation is the weakly identified direction

- **At a diagonal Σ, the two columns are parallel.** For a diagonal $\Sigma^0$, the columns of
  $A(\Sigma^0)$ indexed $(k,l)$ and $(l,k)$ are nonzero only in rows $(k,l)$ and $(l,k)$, so they
  are parallel. Hence $\Gamma^0_{SS}$ is singular whenever the graph contains a 2-cycle. Source:
  Dettling et al. (2024), App. G.1, proof of Theorem 3, Case I. Orientation information exists only
  through the off-diagonal correlations.
- **Curvature along the reversal direction, for general $\hat\Sigma$** (unit-norm direction;
  derived and checked numerically [scratch]). For a correlation matrix,
  $$\kappa_{ij}=\tfrac12\big\|[V_{ij},\hat\Sigma]\big\|_F^2=\sum_{k\neq i,j}(\rho_{ik}^2+\rho_{jk}^2)+4\rho_{ij}^2 .$$
  Compare the coordinate curvature $v_{ij}=2(\|\hat\Sigma_{j\cdot}\|^2+\hat\Sigma_{ij}^2)\ge 2$
  (NONCONVEX.md §2.2). The two columns are therefore strongly correlated (median 0.77 [scratch]).
- **Orientation is already the crux for the lasso.** Dettling's Theorem 3: near a diagonal
  $M^0=\operatorname{diag}(-d_1,\dots,-d_p)$, irrepresentability holds uniformly iff $d_i<d_j$ for
  every edge $i\to j$. In particular the graph must be a DAG.

### 2.3 The lasso leaves the difference alone; MCP amplifies it (two-predictor toy)

**Setup.** Two standardized predictors with correlation ρ, truth $(b,0)$. Write
$s=\beta_1+\beta_2$ and $d=\beta_1-\beta_2$. Then
$$\tfrac12(\beta-\hat\beta)^\top G(\beta-\hat\beta)=\tfrac{1+\rho}{4}(s-\hat s)^2+\tfrac{1-\rho}{4}(d-\hat d)^2,
\qquad \hat d\sim N(b,\tau^2),\quad \tau^2=\frac{2\sigma^2}{n(1-\rho)} .$$

**What each penalty does with the difference** (both coefficients active, same sign):

- **Lasso.** $\beta=\hat\beta-\frac{\lambda}{1+\rho}(1,1)$, so $d=\hat d$: the difference is left
  untouched. Both predictors are kept iff $|\hat d|<s$.
- **MCP**, with both coefficients in its curved region. From $(G-I/\gamma)\beta=G\hat\beta-\lambda(1,1)$,
  $$d=\hat d\cdot\frac{1-\rho}{1-\rho-1/\gamma}.$$
  For $\gamma<1/(1-\rho)$ there is no stable interior solution, and the estimate goes to the corner
  on $\hat d$'s side. With $\gamma=3$ that happens once $\rho>2/3$.
- **Mnet** (adds a ridge term $\tfrac{\lambda_2}{2}\|\beta\|^2$). The factor becomes
  $\frac{1-\rho}{1-\rho+\lambda_2-1/\gamma}$.

**Probability that the true predictor is kept.** This ignores shrinkage of the sum ($s\approx b$);
write $x=b/\tau$ for how decisive the comparison is.

| $x$ | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| lasso, $\Phi(2x)$ | 0.69 | 0.84 | 0.98 | 1.00 |
| MCP, winner-takes-all, $\Phi(x)$ | 0.60 | 0.69 | 0.84 | 0.98 |

**Who wins depends on how decisive the data are.**

- The lasso pays a price: in the ambiguous zone it keeps the impostor as well, which is a false
  positive.
- MCP wins when the comparison is decisive (large $x$).
- The lasso's hedge wins when it is ambiguous. At the operating points here, one extra true positive
  is worth about two avoided false positives in F1 (TP ≈ 14, FP ≈ 10, 22 true edges: +0.029 vs
  +0.013).

**GCLM analogue.** Let κ be the unit curvature along the reversal direction and $1/\gamma$ MCP's
concavity.

- While both entries are in the curved region, MCP amplifies the split by $\kappa/(\kappa-1/\gamma)$,
  about 1.7 for κ = 0.84 and γ = 3.
- Mnet changes this to $\kappa/(\kappa+\lambda_2-1/\gamma)$.

### 2.4 Same entry threshold; sparsity differences come from correlation

- **Same entry threshold.** $P'(0^+)=\lambda$ for all three penalties, so λ_max is the same. With an
  orthonormal design, soft (lasso), firm (MCP) and SCAD thresholding zero out exactly the same
  coefficients, those with $|z|\le\lambda$ (Breheny & Huang 2011, §2.1–2.2).
- **With correlated predictors MCP is usually sparser.** Two effects:
  - debiasing: less leftover signal for correlated predictors to chase;
  - concentration: it keeps one member of a near-interchangeable group.
- **Sparser is not the problem.** At matched FPR, MCP has less recall. As §3.1 puts it, the
  difference is not how many entries are selected but which.

### 2.5 When can a pair get stuck in the wrong orientation?

Take the reversal segment $(a-t,\,t)$, $t\in[0,a]$; the loss has curvature $2\kappa$ in $t$. Both
corners are local minima iff
$$\kappa\,a \;<\; P'(0^+)-P'(a),$$
where the right-hand side is the slope drop, i.e. the concavity integrated over $[0,a]$.

- **MCP.** The slope drop is $\min(a/\gamma,\lambda)$, so this requires $\kappa<1/\gamma$ (and
  $\kappa a<\lambda$).
- **Lasso.** The slope drop is 0, so it never happens.
- **Random DGP.** Only about 13 % of true edges satisfy $\kappa<1/\gamma$ [scratch]. For the rest,
  MCP's optimum along the pair is unique but pulled to a corner, i.e. it does not hedge. Genuine
  multiple local minima in the random design must therefore come from multi-entry directions: the
  exact fibre $W\hat\Sigma^{-1}$, where the curvature is 0.
- **Example 2.** All five cycle pairs are bistable.

### 2.6 Corrections to the current text

- **`S2_penalties_losses.md` §3.1, "Reading".** The RSC paragraph argues that the loss curvature
  along sparse combinations falls below $1/\gamma$. For single reversal directions this is false
  for about 87 % of true edges [scratch]. Suggested replacement:
  1. global convexity is impossible ($c_*=0$);
  2. the orientation direction is weakly curved (κ ≈ 0.84 vs $v\ge 2$), and MCP's concavity
     amplifies the noisy split;
  3. quantify local convexity with the Breheny–Huang diagnostic (Step 2b) rather than the RSC
     argument.
- **"It's the kink, not the concavity"** (my chat claim) was the wrong framing. The corner
  preference *is* concavity: the slope drop over $[0,a]$.
- **The ncvreg convention** (`docs/NONCONVEX.md` §2; follow-ups in S2 §3.1). It is the adaptive
  rescaling of Breheny & Huang (2011, §3.2), $\gamma^*=\gamma/v_j$.
  - With $v_{ij}\ge 2$ it is *more* concave than textbook $\gamma$, not a convexity fix.
  - The 1-D subproblems are already convex under the textbook convention:
    $v_{ij}-1/\gamma\ge 2-1/3$.

---

## 3. Next steps, in priority order

Every step below is judged by one question: does it fix orientation without hurting the skeleton?

### Step 1 — Measure orientation (no reruns needed)

**Why.** This turns the negative result into an explained one, and it gives every later
experiment a yardstick.

**What.**

1. Write `simulations/diagnostics/orientation.py`. For each dataset, take $M^*$ and $\hat M$ at the
   best-F1 index; both are stored in the shards as sparse triples, `m_true_{i,j,v}` and
   `m_best_f1_{i,j,v}`. Compute the breakdown of Appendix A.3:
   - skeleton precision, recall and F1;
   - single edges: correct-only / hedged / reversed-only / missed;
   - 2-cycles: both / one / none;
   - false pairs.
2. Run it on these runs, at $p=10,20$ and all four $C$ choices:
   - lasso: the cluster run, restricted to reps < 25;
   - MCP and SCAD: the S1b pilot.
3. Cross-tabulate per pair: lasso outcome × MCP outcome on the same dataset. This tests the "coin
   flip on the lasso's hedges" reading. The best-F1 λ differs per method, so repeat the comparison
   at matched sparsity as well (the λ index whose number of nonzeros is closest).
4. Record per-dataset reversal curvature κ and the column correlation for the true pairs
   (Appendix A.3).

**Expected.** The same picture as §1.3. At $p=20$ the 2-cycles matter less: they make up a
fraction $k/p$ of the true edges, i.e. 5–20 % instead of 10–40 %. So single-edge orientation should
dominate there.

**Deliverable.** A table like §1.3 in `S2_penalties_losses.md` §3.1, plus the rewritten "Reading"
from §2.6.

**Effort.** 2–3 h. The lasso shards from the cluster run are gitignored, so fetch them from LRZ if
they are not local.

### Step 2 — γ sweep on the laptop; hold the cluster runs

**Why.**

- The pilot already settles γ = 3 / 3.7. The full grid would cost about 230 CPU-h (MCP) and 280
  CPU-h (SCAD) for the direct loss alone [repo §4], without changing the conclusion.
- The open question is whether *any* γ beats the lasso. A larger γ shrinks both the amplification
  $\kappa/(\kappa-1/\gamma)$ and the pull towards the corners.

**What.**

- Grid: MCP γ ∈ {6, 10, 30, 100} and SCAD γ ∈ {6, 10, 30, 100}, next to the existing 3 / 3.7.
- Setting: $p=10,20$, 25 reps, paired with the lasso baseline.
- Commands are in Appendix A.4.
- Evaluate with `compare_runs.py` and with the Step 1 diagnostic. The key curve is hedge share
  vs γ.

**Expected.** My guess is a monotone approach to the lasso from below. If some γ beats the lasso
anywhere, that is the thesis estimator to scale up on the cluster.

**About the ncvreg convention.** Deprioritise it. It equals textbook $\gamma/v_{ij}\le 1.5$, which
is more concave, not less. Run it once for completeness, since the advisor suggested ncvreg, but
expect it to be worse.

**Effort.** About 10 min per setting at $p=10$ [repo estimate], and 3–4× that at $p=20$. Run it in
the background.

### Step 2b — Local convexity diagnostic (Breheny & Huang 2011, §4.2)

**Why.** It gives a citable, quantitative replacement for the RSC paragraph: where along the path
is the MCP/SCAD objective locally convex?

**What.**

1. Along each path compute $c_*(\lambda)=\lambda_{\min}(A_U^\top A_U)$. Here $U$ is the active set,
   plus the diagonal, plus the entries that enter at the next λ.
2. MCP is locally convex iff $\gamma\,c_*(\lambda)>1$; SCAD iff $(\gamma-1)\,c_*(\lambda)>1$.
3. Let $\lambda^*$ be the largest λ at which the condition fails. Code is in Appendix A.5.
4. Plot a few paths with the nonconvex region shaded, like their Fig. 2. Also plot the
   distribution of $\lambda^*/\lambda_{\text{best-F1}}$.

**Expected.** For most datasets, convexity is lost before the best-F1 λ; for γ ≥ 30, much later.

**Effort.** 1–2 h at $p=10$. The explicit design is $p^2\times p^2$, which is fine up to $p=20$.

### Step 3 — Example 2 as a positive control, then orientation-aware estimators

**Why.** Example 2 is the one place where nonconvexity is known to hold the right answer:
- the truth is a lower-objective local minimum;
- the truth is ℓ0-identifiable;
- the lasso fails by exactly one reversal.

That makes it the right showcase, and the test bed for fixes.

**What.**

1. **Reproduce §1.4 with the repo code** (Appendix A.1–A.2). Then add MCP/SCAD to the M0
   sample-size sweep ($n=10^2,\dots,10^5,\infty$); add a `--penalty` option to `run_m0.py` if it
   does not have one yet.
2. **Reversal-aware search** (Appendix A.6). Start from the lasso or MCP solution at each λ, try
   add / delete / reverse moves, and accept a move when the criterion improves.
   - The criterion matters. The MCP objective itself fixed Example 2 but not the random design at
     $n=1000$ [scratch].
   - So use the score with the BIC or the eBIC penalty on a refit, or a held-out loss. This is the
     bridge to S3.
3. **Mnet**: MCP plus a ridge term on the off-diagonal entries (Huang et al. 2016).
   - Implementation: the gradient gains $+\lambda_2 M_{\text{off}}$, the Lipschitz constant
     $+\lambda_2$, and the prox is unchanged (Appendix A.7).
   - Grid: $\lambda_2\gamma\in\{0, 0.25, 0.5, 0.75\}$. At $\lambda_2=1/\gamma$ the penalty is exactly
     the lasso on $[-\gamma\lambda,\gamma\lambda]$, so there is no debiasing left.
   - Measure: hedge share, orientation errors, F1/AUC, estimation error.
4. **LLA / adaptive lasso from the lasso** (low priority). It helps estimation error, but it cannot
   fix an orientation that the lasso gives zero weight, as in Example 2 [scratch].

**Expected.**

- Mnet with a moderate λ₂: hedge share up, reversed-only down, F1 between MCP and the lasso, and
  possibly better estimation error than the lasso.
- Reversal search on the score (BIC penalty): fixes Example 2 at large $n$; the random design is
  still to be determined.

**Effort.** Example 2 reproduction: 1 h. Mnet: half a day. Reversal search: 1 day (overlaps with
S3).

### Step 4 — Map where nonconvexity pays off

**Why.** Meeting 1 asked for which drift matrices RSC holds and irrepresentability fails. The
pilot reframes the question: nonconvexity should pay off when orientation is decisive. A map of
that regime would be a positive thesis result.

**What.** Vary three knobs, each with the lasso/MCP/SCAD triple, starting with $p=10$ and `C_ID`:

1. **Sample size** $n\in\{10^3,10^4,10^5,\infty\}$. `draw_instance` already handles $n=\infty$
   (population covariance), and `run_s1.py` has `--n-obs`. In the toy, orientation information
   grows like $\sqrt n$, since $x=b/\tau$.
2. **Edge strength (beta-min).** Draw magnitudes bounded away from 0, e.g. $|M_{ij}|\in[0.5,1]$
   with random sign, instead of $N(0,1)$.
3. **2-cycles on/off.** Drop one direction of every 2-cycle.

Implement knobs 2 and 3 as `sample_drift` flags whose defaults reproduce the current DGP
bit-for-bit. Appendix A.8 does this with deterministic transforms of the existing draws, which keeps
the random stream, and with it the pairing.

**Expected.** MCP's deficit should shrink with $n$, with edge strength, and with 2-cycles switched
off. If the theory story holds, it flips sign somewhere. Population-level failures of the
Example 2 type remain.

**Effort.** Half a day of code, plus background runs.

### Step 5 — Tuning-based comparison and better metrics

**Why.**

- `max_f1` is tuned by an oracle, and `auc`/`aupr` summarise the whole path. Much of the practical
  case for MCP/SCAD is about the model chosen at a data-driven λ.
- `max_acc` says little here, because the empty graph already scores 0.8–0.9.

**What.**

- **Choose λ by the score, with the BIC or the eBIC penalty.** The direct loss has no likelihood,
  so score the Gaussian log-likelihood of the implied covariance $\Sigma(\hat M)$ (Varando's loss;
  it needs $\hat M$ stable). Prefer an unpenalised refit on the support:
  $$\mathrm{score}=L+\mathrm{pen}_{\mathrm{BIC}},\quad L=n\big[\log\det\Sigma(\hat M)+\operatorname{tr}(\Sigma(\hat M)^{-1}\hat\Sigma)\big],\quad \mathrm{pen}_{\mathrm{BIC}}=\log(n)\,(p+|\hat S|).$$
  The eBIC penalty adds $2\gamma_e\log\binom{p(p-1)}{|\hat S|}$ with $\gamma_e=0.5$ (Chen & Chen
  2008). At the selected λ, report F1, precision, recall, skeleton and orientation metrics.
- **Caveats.**
  - Breheny & Huang (2011, §4.3) report that the score with an AIC or BIC penalty sometimes selects
    local minima in the nonconvex region.
  - With 2-cycles, the model dimension can be smaller than $p+|S|$ (identifiability; see Dettling
    et al. 2023). The cyclic-SEM paper in the project folder computes the model dimension as the
    maximal rank of a Jacobian, and it assumes *simple* graphs (no 2-cycles) to get the expected
    dimension. This matters for S3 as well.
- **Metrics.** Add partial AUC (FPR ≤ 0.1), TPR at FPR = 0.05, and the skeleton and orientation
  metrics. Drop `max_acc` from the headline tables.

**Effort.** Half a day.

### Step 6 — Fill S2 §3.2, §3.3 and §4

**Why.** The pilot has run; only the write-up is missing. The symmetric start
($\nabla=\hat\Sigma-I$) predicts the same orientation signature as on the direct loss.

**What.**

- §3.2: paired comparisons of MCP and SCAD against each loss's *own* lasso.
- §3.3: each loss's lasso against the direct lasso.
- Run the Step 1 diagnostic on the S2 shards.
- §4: the timing table from `runs/s2_pilot_p10/timing.txt`.
- Commands are in Appendix A.9.

With 40 datasets per $(p,C)$, expect paired standard errors of about 0.01–0.02 [scratch].

**Effort.** 1–2 h.

---

## 4. Plan for tomorrow

1. **(1 h)** Reproduce Example 2 with the repo code (A.1, A.2). If it matches §1.4, the
   positive-control story is solid.
2. **(2–3 h)** Step 1: run `diagnostics/orientation.py` on the S1b and lasso shards. Update §3.1
   with the table and the new "Reading" (§2.6).
3. **(15 min, then in the background)** Launch the γ sweep at $p=10$ (A.4).
4. **(1–2 h)** Step 6: write S2 §3.2/§3.3 using `compare_runs.py`.
5. **If time is left:** the Step 2b local convexity plot for 3–4 datasets.

## 5. Questions for the advisor / the Oct 10–11 meeting

- Is it known that Example 2's irrepresentability failure is a single reversed edge? Does Dettling
  read it the same way?
- Which convention should define the thesis estimator, textbook or ncvreg? Note that the ncvreg
  convention is Breheny & Huang's adaptive rescaling, which is more concave here.
- Is it acceptable to frame the S1b result around orientation and 2-cycles as a main thesis
  finding, with Example 2 and the regime map as the positive part?
- For S3: should the score be the likelihood of $\Sigma(M)$ with the BIC or the eBIC penalty? How
  should the dimension be counted for graphs with 2-cycles? Should the neighbourhood include
  reversal moves?
- Already open in `S1_reproduction.md` §9.2: the data behind Figure 5, and the `C_Random_Full` gap.

---

## 6. References

- Dettling, P., Drton, M., Kolar, M. (2024). On the Lasso for Graphical Continuous Lyapunov
  Models. *CLeaR 2024*, PMLR 236. Relevant parts: App. B, Remark B.2 (equation count);
  Theorem 3 and App. G.1, Case I (parallel columns at diagonal Σ; 2-cycles).
- Dettling, P., Homs, R., Améndola, C., Drton, M., Hansen, N. R. (2023). Identifiability in
  continuous Lyapunov models. *SIAM J. Matrix Anal. Appl.* 44(4):1799–1821.
- Varando, G., Hansen, N. R. (2020). Graphical continuous Lyapunov models. *UAI 2020*, PMLR 124.
- Améndola, C., Dettling, P., Drton, M., Onori, F., Wu, J. Structure learning for cyclic linear
  causal models. *UAI 2020* (PDF in the project folder).
- Fan, J., Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle
  properties. *JASA* 96:1348–1360.
- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty.
  *Ann. Statist.* 38(2):894–942.
- Breheny, P., Huang, J. (2011). Coordinate descent algorithms for nonconvex penalized regression,
  with applications to biological feature selection. *Ann. Appl. Stat.* 5(1):232–253. Relevant
  parts:
  - §2.3: convexity iff $\gamma>1/c_*$ (MCP) or $\gamma>1+1/c_*$ (SCAD);
  - §3.2: adaptive rescaling, i.e. the "ncvreg convention";
  - §4: convexity, stability, the local-convexity diagnostic, and the choice of γ and λ.
- Huang, J., Breheny, P., Lee, S., Ma, S., Zhang, C.-H. (2016). The Mnet method for variable
  selection. *Statistica Sinica* 26(3):903–923.
- Lee, S., Breheny, P. Strong rules for nonconvex penalties and their implications for efficient
  algorithms in high-dimensional regression. arXiv:1403.2963. Its §3.1 covers ℓ2-stabilisation
  (Mnet) and the local convexity of Mnet.
- Mazumder, R., Friedman, J., Hastie, T. (2011). SparseNet: coordinate descent with nonconvex
  penalties. *JASA* 106(495):1125–1138.
- Zou, H., Hastie, T. (2005). Regularization and variable selection via the elastic net.
  *JRSS-B* 67(2):301–320. Relevant for the grouping effect.
- Loh, P.-L., Wainwright, M. J. (2015). Regularized M-estimators with nonconvexity: statistical and
  algorithmic theory for local optima. *JMLR* 16:559–616.
- Loh, P.-L., Wainwright, M. J. (2017). Support recovery without incoherence: a case for nonconvex
  regularization. *Ann. Statist.* 45(6):2455–2482.
- Zou, H., Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models.
  *Ann. Statist.* 36(4):1509–1533. Relevant for LLA.
- Fan, J., Xue, L., Zou, H. (2014). Strong oracle optimality of folded concave penalized estimation.
  *Ann. Statist.* 42(3):819–849.
- Chen, J., Chen, Z. (2008). Extended Bayesian information criteria for model selection with large
  model spaces. *Biometrika* 95(3):759–771.

---

## Appendix A — code sketches

The function signatures below are taken from the repo as I read them (`run_s1.py`,
`run_s1_shard.py`, `solvers/proxgrad.py`). Adjust them if they differ.

### A.1 Example 2 with the repo code

```python
import numpy as np
from gclm.data.examples import example2_cycle
from gclm.lyapunov import solve_lyapunov
from gclm.solvers.path import lasso_path
from gclm.metrics import evaluate_path

m_true = example2_cycle()                 # edge i -> j is M[j, i]; so 5 -> 1 is M[0, 4]
c = 2 * np.eye(5)
sigma = solve_lyapunov(m_true, c)         # population covariance (n = inf), raw scale as in M0

for pen in ["lasso", "MCP", "SCAD"]:
    path = lasso_path(sigma, c, penalty=pen, tol=1e-10)
    ev = evaluate_path(path.estimates, m_true, include_diagonal=False)
    print(pen, round(ev["max_f1"], 3), round(ev["auc"], 3))   # [scratch]: 0.8 / 0.833 for all three
    for m in path.estimates[::11]:                              # increasing-lambda order
        print(f"   M[0,4] (5->1, true) = {m[0, 4]: .4f}   M[4,0] (1->5) = {m[4, 0]: .4f}")
```

### A.2 $M^*$ as an MCP local minimum, and one reversal move

```python
from gclm.objective.direct import lambda_max, objective
from gclm.objective.penalties import penalty_weights
from gclm.solvers.path import lambda_grid
from gclm.solvers.proxgrad import solve_fista

lam, gamma = 0.05, 3.0
w = penalty_weights(5, penalize_diagonal=False)
lams = np.sort(np.append(lambda_grid(lambda_max(sigma, c, penalize_diagonal=False)), lam))
path = lasso_path(sigma, c, lambdas=lams, penalty="MCP", gamma=gamma, tol=1e-12)
m_cont = path.estimates[int(np.argmin(np.abs(lams - lam)))]        # continuation solution at lam

F = lambda m: objective(m, sigma, c, lam, w, "MCP", gamma, "textbook")
m_fix = solve_fista(sigma, c, lam, weights=w, m_init=m_true, penalty="MCP", gamma=gamma, tol=1e-12)
print(np.abs(m_fix - m_true).max())         # [scratch] ~1e-15: M* is a fixed point
print(F(m_cont), F(m_true))                 # [scratch] 0.01901 vs 0.01875

trial = m_cont.copy(); trial[0, 4], trial[4, 0] = trial[4, 0], 0.0     # reverse 1->5 into 5->1
m_rev = solve_fista(sigma, c, lam, weights=w, m_init=trial, penalty="MCP", gamma=gamma, tol=1e-12)
off = ~np.eye(5, dtype=bool)
print(F(m_rev), np.argwhere((m_rev != 0) & off))   # [scratch] 0.01875 and exactly the true support
```

### A.3 Orientation breakdown (pure numpy)

```python
import numpy as np

def dense(i, j, v, p):
    """Rebuild a dense matrix from the shard's sparse triples (load them as aggregate_s1.py does)."""
    m = np.zeros((p, p)); m[np.asarray(i), np.asarray(j)] = v
    return m

def orientation_breakdown(m_true, m_hat, zero_tol=0.0):
    p = m_true.shape[0]
    off = ~np.eye(p, dtype=bool)
    up = np.triu(np.ones((p, p), dtype=bool), 1)           # each unordered pair once
    T = (m_true != 0) & off
    S = (np.abs(m_hat) > zero_tol) & off
    single = T & ~T.T                                       # true edge whose reverse is absent
    twocyc = up & T & T.T                                   # 2-cycles, counted once per pair
    hit = S | S.T                                           # pair selected in either direction
    true_pair, sel_pair = up & (T | T.T), up & hit
    sk_tp = int((true_pair & sel_pair).sum())
    sk_fp = int((~true_pair & sel_pair).sum())
    sk_fn = int((true_pair & ~sel_pair).sum())
    return {
        "skeleton_f1": 2 * sk_tp / max(2 * sk_tp + sk_fp + sk_fn, 1),
        "single_correct_only": int((single & S & ~S.T).sum()),
        "single_hedged":       int((single & S & S.T).sum()),
        "single_reversed_only": int((single & ~S & S.T).sum()),
        "single_missed":       int((single & ~hit).sum()),
        "twocyc_both":         int((twocyc & S & S.T).sum()),
        "twocyc_one":          int((twocyc & (S ^ S.T)).sum()),
        "twocyc_none":         int((twocyc & ~hit).sum()),
        "false_pairs":         sk_fp,
    }

def reversal_curvature(sigma_hat, i, j):
    """Curvature of the direct loss along the unit reversal direction (V_ij / sqrt 2)."""
    p = sigma_hat.shape[0]
    v = np.zeros((p, p)); v[i, j], v[j, i] = 1.0, -1.0
    x = v @ sigma_hat - sigma_hat @ v
    return 0.5 * float(np.sum(x * x))

def column_correlation(sigma_hat, i, j):
    """Correlation between the design columns of M_ij and M_ji."""
    p = sigma_hat.shape[0]
    def col(a, b):
        e = np.zeros((p, p)); e[a, b] = 1.0
        return (e @ sigma_hat + sigma_hat @ e.T).ravel()
    u, w = col(i, j), col(j, i)
    return float(u @ w / np.sqrt((u @ u) * (w @ w)))
```

### A.4 γ sweep

```bash
for pen in MCP SCAD; do
  for g in 6 10 30 100; do
    run=runs/s1b_gamma/${pen}_g${g}
    for i in 0 1 2; do
      python simulations/run_s1_shard.py --shard $i --n-shards 3 --p 10 20 --reps 25 \
          --penalty $pen --gamma $g --out-dir $run/s1_shards &
    done
    wait
    python simulations/aggregate_s1.py --in-dir $run/s1_shards
    python simulations/compare_runs.py --baseline runs/s1_dettling_reproduction --run $run \
        --reps 25 --p 10 20 --csv $run/paired_vs_lasso.csv
  done
done
# ncvreg convention, once:  ... --penalty MCP --convention ncvreg --out-dir runs/s1b_gamma/MCP_ncvreg/s1_shards
```

### A.5 Local convexity diagnostic (Breheny & Huang §4.2)

```python
import numpy as np
from gclm.lyapunov import design_matrix, vec

def column_of(i, j, p):
    e = np.zeros((p, p)); e[i, j] = 1.0
    return int(np.argmax(vec(e)))             # robust to the vec ordering

def c_star(sigma_hat, U):
    """Smallest eigenvalue of A_U' A_U; U is a boolean p x p mask (include the diagonal)."""
    p = sigma_hat.shape[0]
    A = design_matrix(sigma_hat)              # loss = 0.5 ||A vec(M) + vec(C)||^2, Hessian A'A
    AU = A[:, [column_of(i, j, p) for i, j in zip(*np.nonzero(U))]]
    return float(np.linalg.eigvalsh(AU.T @ AU)[0])

# Along a path (estimates in increasing-lambda order), for k = 1..K-1:
#   U_k = (est[k] != 0) | (est[k-1] != 0) | np.eye(p, dtype=bool)   # active now or at the next smaller lambda
#   MCP locally convex at lambda_k  iff  gamma * c_star(sigma_hat, U_k) > 1
#   SCAD locally convex at lambda_k iff  (gamma - 1) * c_star(sigma_hat, U_k) > 1
```

### A.6 Reversal-aware greedy search (sketch)

```python
def local_search(m0, score, refit, max_moves=50):
    """Greedy search over supports. score(m): lower is better (e.g. the score of Sigma(m) with the
    eBIC penalty on a refit, a held-out loss, or the penalised objective).
    refit(support_mask, m_init): re-optimise on the proposed support (e.g. unpenalised least
    squares on the support columns of A, diagonal included)."""
    cur, best = m0.copy(), score(m0)
    p = m0.shape[0]
    for _ in range(max_moves):
        sel = [(i, j) for i, j in zip(*np.nonzero(cur)) if i != j]
        props = []
        for i, j in sel:                                   # reverse i,j -> j,i
            s = cur != 0; s[i, j] = False; s[j, i] = True
            t = cur.copy(); t[j, i] += t[i, j]; t[i, j] = 0.0
            props.append((s, t))
        for i, j in sel:                                   # delete
            s = cur != 0; s[i, j] = False
            t = cur.copy(); t[i, j] = 0.0
            props.append((s, t))
        # optional: add the zero entry with the largest |gradient|
        cand = [refit(s, t) for s, t in props]
        sc = [score(m) for m in cand]
        k = int(np.argmin(sc))
        if sc[k] >= best - 1e-10:
            break
        cur, best = cand[k], sc[k]
    return cur
```

### A.7 Mnet (MCP + ridge on the off-diagonal)

```python
# objective: 0.5*||M S + S M' + C||_F^2 + sum_{i != j} [ MCP_{lam,gamma}(M_ij) + 0.5*lam2*M_ij^2 ]
# Put the ridge into the smooth part, e.g. as an extra `ridge` argument of _solve_mapg:
#   grad  = direct_grad(m, sigma, c) + lam2 * m * offdiag_mask
#   L     = lipschitz_bound(sigma) + lam2
#   prox  = unchanged MCP prox (diagonal unpenalised)
# Grid: lam2 * gamma in {0, 0.25, 0.5, 0.75}. At lam2 = 1/gamma the penalty equals the lasso on
# [-gamma*lam, gamma*lam] (MCP is 1/gamma-weakly convex), so there is no debiasing left.
# ncvreg's parametrisation instead uses lam1 = alpha*lam, lam2 = (1-alpha)*lam, which vanishes at the dense end.
```

### A.8 DGP flags that keep the random stream (and the pairing)

```python
import numpy as np
from scipy.stats import norm

def sample_drift(p, d, rng, metzler=False, two_cycles=True, magnitude=None):
    eps = rng.normal(size=(p, p))
    omega = rng.binomial(1, d, size=(p, p))
    if not two_cycles:                       # keep the direction with the larger |eps|: no extra draws
        both = (omega == 1) & (omega.T == 1)
        omega = np.where(both & (np.abs(eps) < np.abs(eps.T)), 0, omega)
    w = np.abs(eps) if metzler else eps
    if magnitude is not None:                # |w| in [a, b], monotone in |eps|: no extra draws
        a, b = magnitude
        u = 2 * norm.cdf(np.abs(eps)) - 1    # Uniform(0, 1) when eps ~ N(0, 1)
        w = np.sign(w) * (a + (b - a) * u)
    off = omega * w
    np.fill_diagonal(off, 0.0)
    m = off.copy()
    np.fill_diagonal(m, -np.abs(off).sum(axis=1) - np.abs(np.diag(eps)))
    return m
# With the defaults (two_cycles=True, magnitude=None) this is the current function, draw for draw.
```

### A.9 S2 write-up

```bash
# §3.2: penalties against each loss's own lasso (adjust folder names to runs/s2_pilot_p10/{loss}_{penalty})
for loss in loglik frobenius; do
  for pen in MCP SCAD; do
    python simulations/compare_runs.py --baseline runs/s2_pilot_p10/${loss}_lasso \
        --run runs/s2_pilot_p10/${loss}_${pen} --reps 10 --p 10 \
        --csv runs/s2_pilot_p10/${loss}_${pen}/paired_vs_lasso.csv
  done
done
# §3.3: each loss's lasso against the direct lasso on the same 160 datasets
for loss in loglik frobenius; do
  python simulations/compare_runs.py --baseline runs/s1_dettling_reproduction \
      --run runs/s2_pilot_p10/${loss}_lasso --reps 10 --p 10 \
      --csv runs/s2_pilot_p10/${loss}_lasso/paired_vs_direct_lasso.csv
done
# figure
python simulations/plot_penalties.py --p 10 --reps 10 \
    --run "lasso=runs/s2_pilot_p10/loglik_lasso" --run "MCP=runs/s2_pilot_p10/loglik_MCP" \
    --run "SCAD=runs/s2_pilot_p10/loglik_SCAD" --title "loglik: lasso vs MCP vs SCAD" \
    --out runs/s2_pilot_p10/figures/penalties_loglik.png
```

---

## Appendix B — scratch probes (`scratch_probes.zip`)

This is an independent mini re-implementation, for mechanism checks only: numpy + scipy, single
core. The README inside lists every script and how to run it.

| file | what it does |
|---|---|
| `gclm_mini.py` | DGP with the repo's seeding, direct loss, FISTA (lasso), monotone APG (MCP/SCAD), continuation paths, metrics, reversal curvature |
| `test_basic.py`, `test_cycle.py`, `test_cycle2.py` | the Example 2 results of §1.4 |
| `batch_p10.py` + `summarize.py` | the 40-dataset results of §1.3: paired differences, curvature, oracle basin, reversal search |
| `pairs.py` + `summarize_pairs.py` | the single-edge vs 2-cycle table of §1.3 |
| `*.json` | the raw outputs behind the [scratch] numbers |
