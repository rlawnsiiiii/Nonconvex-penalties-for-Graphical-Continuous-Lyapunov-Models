# 5 October 2026: the question about $C$, and the cluster campaign

*Working notes. They answer the three questions of 5 October (has MCP / SCAD been run with the
rescaled $C$? does rescaling $C$ make sense? what does Varando & Hansen's paper do?) and turn
directions A, D, E and F of [`next_steps_051026.md`](next_steps_051026.md) into one cluster
campaign that includes the side experiments: adaptive lasso, search, log-likelihood loss,
larger $p$. The verdicts so far are in [`../../simulations/VERDICTS.md`](../../simulations/VERDICTS.md).*

**Status (7 October): waves 1 – 3 submitted on 5 October, wave 1 complete, wave 2 complete but
for one resubmitted shard, wave 3 half complete, wave 4 started. The results are in
[`../../simulations/S4_campaign.md`](../../simulations/S4_campaign.md); §4 says where everything
is implemented, §5 gives the commands for the cluster.** §1 – §3 are the plan as proposed in the afternoon; where the implementation
differs, the text says so. The new numbers in §2 and the cost estimates in §3 come from small
laptop checks; their scripts and outputs are in [`files/`](files/) and listed in §8.

---

## 0. Summary

1. **Has MCP / SCAD been run with the rescaled $C$?** Yes, but only in small studies (24 to 100
   graphs) and only for `C_ID`. Nothing on the cluster, nothing for the other three settings of the
   true $C$, nothing for the log-likelihood loss (§2.1).
2. **Does rescaling $C$ make sense?** Yes, as the correct specification of the simulation.
   Standardising changes $C$: data generated with $C = 2I$ have $C = 2\,\mathrm{diag}(1/s_i^2)$
   afterwards ($s_i$: the standard deviations). It is the same model as fitting the raw covariance
   with $C = 2I$. Three caveats: it is exact only for `C_ID`; it is not a recipe for real data;
   with a known $C$ the variances carry information about direction. A check shows that the last
   point does not explain the gain (§2.2).
3. **What do the papers do?** Neither rescales. Dettling fixes $C = 2I$ everywhere. Varando &
   Hansen estimate a diagonal $C$. With $C$ estimated no rescaling is needed; with $C$ fixed, which
   is what the repository implements, the likelihood loss has the same problem as the direct
   loss (§2.3).
4. **Does their estimated $C$ solve the problem?** No. In their package the estimated diagonal
   either stays at $I$ or shrinks towards zero. It does not follow the true diagonal (§2.4).
5. **Their paths run dense → sparse,** and their package beats our log-likelihood baseline on
   the same graphs (+0.02 to +0.04 in `max_f1`). With our solver the order alone gives almost
   nothing for the lasso, so most of that difference comes with their solver, which stops its
   fits much earlier than ours (§2.4, §2.5).
6. **The campaign** (§3). Its core is the 2 × 2 "choice of $C$ × order of the path" for MCP and
   SCAD against the lasso, in all four settings of the true $C$, on the 800 graphs of the baseline.
   Controls and extensions: LLA, adaptive lasso, BIC and search, log-likelihood loss, larger $p$.
   Four waves, 1,150 to 1,550 CPU-h in total (the baseline was 720). Wave 1 answers the main
   question: 600 CPU-h, about 6 hours of computing on LRZ.
7. **The code is written** (§4 says where each piece is), a first look on the laptop at the other
   three settings of the true $C$ is in §3.7, and **the commands for the cluster are in §5**.

---

## 1. The target, in one paragraph

The thesis asks whether MCP and SCAD recover the graph of a GCLM better than the lasso.

- **Used the standard way they do not.** This is settled: all 24 cells of the cluster baseline
  (S2b).
- **The small studies say they do when two things are changed together:** the path is run from
  the dense end, and $C$ is specified correctly. So far this rests on 24 to 100 graphs with one of
  Dettling's four choices of the true $C$ (`C_ID`).

The campaign tests that claim at the scale of the baseline. Its core is a 2 × 2 design for MCP and
SCAD,

| | sparse → dense (the standard way) | dense → sparse |
|---|---|---|
| **$C = 2I$** (Dettling's pipeline) | cluster baseline: loses everywhere | small studies: ties the lasso at $n = 10^3$, small gain at larger $n$ |
| **rescaled $C$** | small studies: MCP loses, SCAD loses a little or ties | small studies: wins |

against the lasso, on the same 800 graphs per cell, in all four of Dettling's settings for the
true $C$. Everything else in the campaign is a control or an extension of this table:

- **MCP / SCAD computed by a few weighted lassos started from the lasso (LLA):** the version of
  "start from the lasso" that has theory behind it. Does it do as well?
- **adaptive lasso:** is it the nonconvex penalty that helps, or only the dense start?
- **BIC and search:** does the gain survive when $\lambda$ is chosen from the data, and does a
  search add to it?
- **log-likelihood loss:** does the same table hold for the other loss?
- **larger $p$:** does it hold up to $p = 50$, the range of Dettling's Figure 5?

---

## 2. The question about $C$

### 2.1 What has already been run with a correctly specified $C$

MCP and SCAD *have* been run with the rescaled $C$, but only in small studies and only for `C_ID`.
S3b used the raw scale instead, which for `C_ID` is the same model. The last column is `max_f1`
minus the lasso's, both fitted with the same $C$, in the order of the sample sizes.

| study | solvers | graphs | true $C$ | $n$ | result |
|---|---|---|---|---|---|
| independent study §5 | its own | 100, $p = 10$ | `C_ID` | $10^3, 10^4, 10^5, \infty$ | standard MCP −0.07 / −0.07 / −0.08 / −0.08; standard SCAD −0.02 / −0.03 / −0.03 / −0.03; MCP dense → sparse +0.03 / +0.07 / +0.09 / +0.09; adaptive lasso +0.03 / +0.06 / +0.08 / +0.08 |
| independent study §11 | its own | 24, $p = 20$ | `C_ID` | $10^3, 10^4, \infty$ | standard MCP −0.07 / −0.05 / −0.10; standard SCAD 0.00 / +0.03 / 0.00; MCP dense → sparse +0.07 / +0.16 / +0.18; adaptive lasso +0.10 / +0.16 / +0.17 |
| my replication (031026 §4) | repository | 40, $p = 10$ | `C_ID` | $10^3, 10^4, \infty$ | standard MCP −0.05 / −0.05 / −0.06; MCP dense → sparse +0.03 / +0.07 / +0.10; SCAD dense → sparse +0.02 / +0.08 / +0.10 |
| S3b, raw scale¹ | repository | 40 + 120, $p = 10$ | all four | $10^3, 10^4, \infty$ | `C_ID`: standard MCP −0.02 / −0.02 / −0.03, standard SCAD −0.02 / −0.01 / 0.00; random $C$: standard MCP −0.02, standard SCAD −0.01 to 0.00 |

¹ Unstandardised data with $C = 2I$. For `C_ID` this is the same model as the rescaled $C$, with
a different weighting of loss and penalty (§2.2). The lasso itself is weaker on the raw scale.

**Not run yet:**

- the rescaled $C$ in the three settings where the true $C$ is not $2I$;
- the standard SCAD path with the rescaled $C$ and the repository's solvers;
- more than 24 graphs at $p = 20$;
- the log-likelihood and Frobenius losses with the rescaled $C$;
- anything with the rescaled $C$ on the cluster. The baseline's 36 cells all used $C = 2I$.

**What the small runs say, and what they leave open:**

- The rescaled $C$ alone does not rescue the standard MCP path (−0.05 to −0.10).
- For the standard SCAD path it closes most of the gap, and at $p = 20$ SCAD ties the lasso. With
  24 graphs that is not settled; with 800 it will be.
- Dense → sparse with the rescaled $C$ wins at every $n$, more at $p = 20$ than at $p = 10$.

### 2.2 Does rescaling $C$ make sense?

**What standardising does to the model.** Let the data follow $M\Sigma + \Sigma M^\top + C = 0$.
Divide every variable by its standard deviation $s_i$. The standardised data follow the same kind of
model, with

$$\tilde M_{ji} = M_{ji}\,\frac{s_i}{s_j}, \qquad \tilde C = \mathrm{diag}(1/s)\;C\;\mathrm{diag}(1/s).$$

The zeros of $\tilde M$ are the zeros of $M$, so the graph is the same. But $C$ changes. If the
data were generated with $C = 2I$, the standardised data have $\tilde C = 2\,\mathrm{diag}(1/s_i^2)$.
That is the "rescaled $C$".

**There are four ways to fit, and they assume different things:**

| what is fitted | what it assumes about the noise | true for `C_ID` data | depends on the units of the data |
|---|---|---|---|
| raw covariance, $C = 2I$ | every variable receives noise of the same size, in the units of the data | yes | yes |
| correlation matrix, $C = 2\,\mathrm{diag}(1/s_i^2)$ (**rescaled $C$**) | the same assumption, written in standardised units | yes | yes |
| correlation matrix, $C = 2I$ (**Dettling's pipeline**; Varando & Hansen with $C$ fixed) | every variable receives noise in proportion to its own variance | no | no |
| correlation matrix, diagonal $C$ **estimated** (Varando & Hansen's general estimator) | the noise of different variables is independent; its sizes are unknown | yes, and for the two random diagonal settings | no |

**Check: how well does the true graph fit?** Least-squares loss of the best fit on the *true*
support at $n = \infty$, relative to the diagonal fit; 0 means the true graph fits exactly
(`files/fit_check_free_diagonal.py`, 40 graphs per setting, $p = 10$):

| true $C$ | $C = 2I$ | rescaled $C$ | diagonal $C$ left free |
|---|---|---|---|
| `C_ID` | 0.0137 | **0.0000** | **0.0000** |
| `C_Random_Diag` | 0.0152 | 0.0095 | **0.0000** |
| `C_Random_Min_Diag` | 0.0180 | 0.0009 | **0.0000** |
| `C_Random_Full` | 0.2740 | 0.1964 | 0.0676² |

² Not attainable: in some graphs the best "diagonal" has negative entries.
(The middle column was 0.0077 and 0.1939 in the 3 October note for the two random settings; that
table scaled the loss differently. The conclusions are the same.)

**My answer: yes, with three caveats.** (Why the correct $C$ matters for the dense-start
estimators in particular, and not for the lasso, is worked out with numbers in
[`docs/DENSE_START.md`](../../docs/DENSE_START.md) §7, added on 7 October.)

- **It is the correct specification of Dettling's own benchmark.** In that simulation the data
  are generated with $C = 2I$, and the paper calls this the setting in which the fitted $C$ is
  right: "Choice 1) for $C$ is used when applying Direct Lyapunov Lasso for model selection. Thus,
  it is natural to expect the best results for this choice" (§5 of the paper). The simulations
  standardise the data (S1 §8.6: stated in Dettling's dissertation, and needed to reproduce
  Figure 5). After standardising, the sentence is true only with the rescaled $C$.
- **It uses no extra information.** It is the model "$C = 2I$ in the units of the data", which is
  the assumption of the paper's theory ("it mirrors the equal variance assumption for structural
  equation models", Remark B.1), plus the standard deviations that the analyst computes anyway. It is
  algebraically the same as fitting the raw covariance with $C = 2I$; only the weighting of the
  loss and of the penalty differs.
- **Why not simply fit the raw covariance?** Because the lasso is much worse there: `max_f1` 0.576
  against 0.619 at $p = 10$ and 0.328 against 0.512 at $p = 50$ (S1 §8.6). Without standardising,
  loss and penalty are badly balanced across variables. Standardising is right; the question is
  only which $C$ goes with it.
- **Without a correct $C$ the question of the thesis cannot be tested.** What theory promises for
  nonconvex penalties (less bias, recovery without the irrepresentability condition) is promised
  for a correctly specified model. With $C = 2I$ on standardised data the true graph does not fit
  the covariance even at $n = \infty$, so nothing guarantees that any estimator converges to it.
  In the baseline the gap between MCP and the lasso even *grows* with $n$ (S2b).

**The caveats:**

1. **It is exact only for `C_ID`.** In the two random diagonal settings it removes part of the
   misfit, in `C_Random_Full` little (table above). Whether the gains survive there is the first
   question of the campaign.
2. **It is not a recipe for real data.** Real data have no scale on which $C = 2I$ is known. There
   the rescaled $C$ means "trust the units of the measurements", which is arbitrary. Dettling's
   convention at least does not depend on the units. So the rescaled $C$ belongs to the simulation
   study. For real data the honest options are a sensitivity analysis or an estimated $C$.
3. **With a known $C$ the variances carry information about direction.** In this DGP 67 % of the
   true edges point from the node with the larger variance to the one with the smaller. An
   estimator with the rescaled $C$ can use that; one with $C = 2I$ cannot. Using it is legitimate
   under the model, as in equal-variance SEMs. The worry is that the gain is nothing but this.
   **It is not** (check below).

**Check for caveat 3: a rule that uses only the variances.** Take the pairs the lasso finds and
orient each from the larger to the smaller variance. `max_f1`, 40 `C_ID` graphs, $p = 10$, rescaled
$C$ (`files/variance_ordering_baseline.py`):

| | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|
| lasso | 0.637 | 0.673 | 0.696 |
| MCP dense → sparse | 0.662 | 0.747 | 0.798 |
| the lasso's pairs, oriented from the larger to the smaller variance | 0.570 | 0.596 | 0.611 |
| the lasso's pairs, oriented by a coin | 0.463 | 0.488 | 0.499 |
| the lasso's pairs, oriented by the truth (one direction per pair) | 0.739 | 0.783 | 0.803 |

The variance rule loses to the lasso by 0.07 to 0.09 ($z \approx -5$). Being right for 67 % of the
edges is below the 70 % that committing to one direction needs (verdict 5). So the +0.03 to +0.10
of MCP dense → sparse comes from somewhere else.

### 2.3 What the two papers do

**Neither paper rescales $C$.**

**Dettling, Drton & Kolar (2024)** fit with $C = 2I_p$ throughout: in all four simulation settings
(§5), and on standardised real data, including inside the likelihood of their extended BIC
(§6, eq. 6.1). A misspecified $C$ is treated as something the lasso is robust to.

**Varando & Hansen (2020)** go another way: they estimate $C$.

- **The estimator** (their eq. 7) minimises
  $L(\Sigma(B, C)) + \lambda\rho_1(B) + \kappa\lVert C - I_p\rVert_F^2$ over stable $B$ and
  **diagonal** $C$. "The penalization term … is necessary, since the pair $(B, C)$ can only be
  identified up to a multiplicative constant. Letting $\kappa = +\infty$, we obtain as a special
  case an estimator of $B$ with $C = I_p$ fixed."
- **Their simulation:** true $C$ diagonal with $C_{ii} \sim \mathrm{Uniform}([0,1])$. "Data was
  standardized, which means that all methods used the empirical correlation matrix." The methods
  `mloglik-inf`, `frob-inf` and `lasso` fix $C = I_p$; `mloglik-0.01` estimates $C$ with
  $\kappa = 0.01$.
- **Their result:** the two likelihood versions "were highly similar with the exception of the
  precision-recall curve where `mloglik-0.01` obtained consistently higher results".

**So, is rescaling "not needed" for the likelihood loss?**

- **With $C$ fixed it is needed exactly as for the direct loss.** This is the version the
  repository implements (`docs/LIKELIHOOD.md` §1) and the one all cluster runs used. Their fixed-$C$
  methods are misspecified in their own simulation, as the lasso is in Dettling's.
- **With $C$ estimated it is not needed.** If $(M, C)$ with a diagonal $C$ fits $\Sigma$, then
  $(\tilde M, \tilde C)$ fits the correlation matrix and $\tilde C$ is again diagonal. A free
  diagonal absorbs the rescaling, and also the random diagonals of two of Dettling's settings
  (last column of the table in §2.2).

**A second point in their paper matters for us.** Their paths run dense → sparse: the algorithm
"starts from a dense estimate and moves along the regularization parameters in increasing order
toward sparser and sparser solutions … we empirically observed that better results were obtained
using an increasing sequence" (§3.1). All our log-likelihood runs went sparse → dense. So what the
baseline calls the log-likelihood lasso is not quite their estimator. The option exists in the
repository (`covloss_path(direction="up")`) and had not been used; §2.5 has a first check.

### 2.4 A check with their package: does estimating $C$ do the job?

`files/estimate_c_check.py` runs their package (`gclm`, log-likelihood loss, lasso) on 100 of our
graphs at $p = 10$: 40 for `C_ID`, 20 for each of the other three settings, standardised as
always. Four treatments of $C$: fixed at $I$ (their `mloglik-inf`, and our pipeline up to a factor
2); fixed at the rescaled value; estimated with $\kappa = 0.01$ (their `mloglik-0.01`); estimated
with $\kappa = 1$. All paths run dense → sparse as in their paper; the first one also
sparse → dense. The first row is the cluster baseline on the same graphs.

In brackets: $z$ of the paired difference to the third row.

**`C_ID`, 40 graphs**

| | `max_f1`, $n = 10^3$ | `max_f1`, $10^4$ | `aupr`, $10^3$ | `aupr`, $10^4$ |
|---|---|---|---|---|
| cluster baseline: our solver, $C = 2I$, sparse → dense | 0.633 (−2.1) | 0.661 (−3.7) | 0.544 (−3.7) | 0.573 (−5.4) |
| their package, $C = I$, sparse → dense | 0.592 (−4.9) | 0.630 (−5.0) | 0.520 (−6.0) | 0.560 (−4.2) |
| their package, $C = I$ (`mloglik-inf`) | 0.654 | 0.688 | 0.603 | 0.638 |
| their package, rescaled $C$ | 0.666 (+1.8) | 0.715 (+3.9) | 0.637 (+2.6) | 0.694 (+3.6) |
| their package, $C$ estimated, $\kappa = 0.01$ (`mloglik-0.01`) | 0.642 (−0.9) | 0.692 (+0.4) | 0.632 (+1.9) | 0.694 (+3.9) |
| their package, $C$ estimated, $\kappa = 1$ | 0.658 (+0.4) | 0.682 (−0.6) | 0.616 (+1.1) | 0.638 (+0.0) |

**random diagonal $C$ (`C_Random_Diag`, `C_Random_Min_Diag`), 40 graphs**

| | `max_f1`, $n = 10^3$ | `max_f1`, $10^4$ | `aupr`, $10^3$ | `aupr`, $10^4$ |
|---|---|---|---|---|
| cluster baseline: our solver, $C = 2I$, sparse → dense | 0.610 (−2.1) | 0.663 (−3.7) | 0.522 (−2.8) | 0.575 (−5.4) |
| their package, $C = I$, sparse → dense | 0.550 (−7.1) | 0.583 (−6.4) | 0.478 (−5.5) | 0.512 (−6.2) |
| their package, $C = I$ (`mloglik-inf`) | 0.647 | 0.697 | 0.585 | 0.663 |
| their package, rescaled $C$ | 0.635 (−1.0) | 0.691 (−0.6) | 0.601 (+0.9) | 0.682 (+1.6) |
| their package, $C$ estimated, $\kappa = 0.01$ (`mloglik-0.01`) | 0.625 (−1.8) | 0.679 (−1.5) | 0.603 (+0.9) | 0.659 (−0.3) |
| their package, $C$ estimated, $\kappa = 1$ | 0.630 (−1.3) | 0.666 (−3.2) | 0.585 (−0.0) | 0.614 (−3.2) |

**What it shows:**

1. **Their estimated $C$ is not an estimate of the true $C$.** With $\kappa = 0.01$ the diagonal
   only shrinks: at the best $\lambda$ the smallest entry has a median of 0.03 (the start is 1),
   in 40 % of the graphs at least one entry is below 0.01, and no entry ever exceeds 1.09. With
   $\kappa = 1$ it hardly moves (typically every entry stays within 5 % of 1). In neither case
   does the estimate follow the true diagonal: the correlation of the logarithms is 0.07 to 0.20,
   against 1.00 (`C_ID`) and 0.84 (random diagonal) for the rescaled $C$.
   *Why (my reading, not from the paper):* with a free diagonal the likelihood cannot tell
   different $C$ apart, because for every diagonal $C$ the dense matrix $-\tfrac12 C\hat\Sigma^{-1}$
   fits the covariance exactly. Only the penalty on $M$ distinguishes them, and a smaller $C$
   allows a smaller $M$. So the estimate drifts towards zero until the ridge stops it.
2. **On `C_ID` it still behaves as their paper reports:** `max_f1` as with $C$ fixed, `aupr`
   higher by 0.03 to 0.06, which is as much as with the rescaled $C$. So that gain does not come
   from learning $C$. In the random diagonal settings it gains nothing (`max_f1` −0.02).
3. **The rescaled $C$ helps the log-likelihood lasso on `C_ID`:** +0.01 to +0.03 in `max_f1`
   and +0.03 to +0.06 in `aupr`. With a random true diagonal nothing moves beyond noise. For
   `C_Random_Full` (20 graphs) all treatments are within 0.04 of each other.
4. **With their package the order of the path matters a lot** (0.06 to 0.11 in `max_f1`). In
   their order it beats our baseline on the same graphs: +0.02 to +0.04 in `max_f1` and +0.06 to
   +0.09 in `aupr`. Their solver stops much earlier than ours, so this may be the order, the
   early stopping, or both. §2.5 separates the two: the order explains little of it.

**Limits of this check.** $p = 10$ only. Their stopping rule is weak in the flat directions of
this loss, so single paths are not converged; I capped the iterations (10,000 per $\lambda$,
50 values of $\lambda$) to keep the check under an hour. At $n = \infty$ their solver stops almost
at once, because the start already fits exactly; the results are well below ours (`max_f1` 0.61
against 0.70 for `C_ID`), so I do not report that column.

### 2.5 Consequences for the campaign

- **Both choices of $C$ stay in every wave.** $C = 2I$ is Dettling's pipeline and the link to
  Figure 5. The rescaled $C$ is the correctly specified benchmark for `C_ID`. Showing both shows
  how much of each result depends on $C$.
- **In the thesis the rescaled $C$ is an assumption, stated as one:** "$C$ is known on the scale
  on which the data were measured". That is Dettling's own assumption, carried through the
  standardisation. It is not offered as a method for real data.
- **Estimating $C$ is a research question, not a quick fix.** Their ridge either leaves $C$ at $I$
  or lets it shrink towards zero (§2.4). A version that could work would fix the size of $C$ by a
  constraint, for example on its trace, instead of a ridge. It also needs an identifiability
  argument, because a free diagonal uses up $p - 1$ of the $p(p+1)/2$ equations. That is a possible
  later chapter, not part of this campaign.
- **The log-likelihood loss is run in both path orders** (wave 3). For the lasso the order turns
  out to matter little with our solver (`files/loglik_order_check.py`, the 40 `C_ID` graphs,
  $C = 2I$):

  | our solver, log-likelihood lasso | `max_f1`, $n = 10^3$ | `max_f1`, $10^4$ | `aupr`, $10^3$ | `aupr`, $10^4$ |
  |---|---|---|---|---|
  | sparse → dense | 0.630 | 0.671 | 0.537 | 0.574 |
  | dense → sparse | 0.641 (+1.2) | 0.663 (−0.7) | 0.569 (+2.1) | 0.591 (+0.9) |
  | their package, dense → sparse (§2.4) | 0.654 | 0.688 | 0.603 | 0.638 |

  (In brackets: $z$ of the paired difference to the first row.) So the order explains little of
  the advantage of their package. In the same order it is still ahead of our solver by 0.01 to
  0.03 in `max_f1` and 0.03 to 0.05 in `aupr` ($z$ 1.7 to 5.2). The two solve the same problem;
  they differ in how far each fit is converged (theirs stops early, ours runs to a tight
  tolerance) and in the grid of $\lambda$. My guess, not tested: a path that is stopped early
  stays closer to the dense start, and that helps. It would fit the earlier finding that a lower
  objective is not a better graph (`docs/LIKELIHOOD.md` §5). It matters whenever our numbers are
  compared with those of their paper.
- **Two questions for the 10 – 11 October meeting.** Is $C = 2I$ on standardised data intended
  in Dettling's paper, given that `C_ID` is described as the setting in which the fitted $C$ is
  right? And how far were the fits in Varando & Hansen's simulations converged?

---

## 3. The campaign

### 3.1 Common to all waves

- **Graphs and data.** Figure 5's generator with the seeds of the baseline. Per cell: $p = 10$ and
  $20$, 4 densities ($k = 1..4$), 4 choices of the true $C$, 25 replicates, so 800 graphs. Every
  cell sees the same graphs, so every comparison is paired, also with the baseline.
- **Standardised data, as in every run so far.** Each estimator is run twice: with $C = 2I$ and
  with the rescaled $C$.
- **Settings as before:** MCP with $\gamma = 3$, SCAD with $\gamma = 3.7$, 100 values of $\lambda$
  from $\lambda_{\max}$ to $\lambda_{\max}/10^4$.
- **The Frobenius loss is left out.** It is the worst loss for every penalty and took 70 % of the
  baseline's compute (S2b §4).
- **The cluster writes numbers only.** Figures and tables are made on the laptop.

### 3.2 The estimators

| estimator | penalty | how the path is computed | convex |
|---|---|---|---|
| lasso | $\ell_1$ | the order does not matter | yes |
| MCP, SCAD, **sparse → dense** | MCP, SCAD | from $\lambda_{\max}$ downwards, starting from the empty graph. The standard way, used in every thesis run so far. | no |
| MCP, SCAD, **dense → sparse** | MCP, SCAD | from the smallest $\lambda$ upwards, starting from the lasso solution at the smallest $\lambda$ | no |
| MCP, SCAD, **by LLA from the lasso** | MCP, SCAD | at every $\lambda$: start from the lasso solution at that $\lambda$, then two weighted lassos (below) | each step is |
| **adaptive lasso** | $\ell_1$ with weights $1/\lvert M^0_{ij}\rvert$, where $M^0$ is the lasso solution at the smallest $\lambda$ | the order does not matter | yes |

Eight estimators, each with two choices of $C$: 16 for the direct loss.

**What LLA is.** The local linear approximation solves an MCP or SCAD problem by a sequence of
weighted lassos. Each entry gets the penalty weight that the penalty's slope has at the entry's
current size: large entries are not penalised any more, small ones keep the full weight. Started
from the lasso solution, two such steps reach the oracle estimator with high probability under
conditions that do not include irrepresentability (Fan, Xue & Zou 2014; the one-step idea is Zou &
Li 2008). This is the textbook reason to expect a gain from MCP / SCAD where the lasso's condition
fails, and it is a statement about a lasso start, not about the standard path. Every step is
convex, so the result does not depend on the machine. In the independent study it does as well as
MCP dense → sparse (`max_f1` 0.643 / 0.723 / 0.764 / 0.777 with the rescaled $C$, $p = 10$).

**Why the adaptive lasso is in.** It also starts from the dense lasso solution and then prunes,
but its second step is convex. If it matches MCP dense → sparse, the gain comes from starting
dense, not from the nonconvex penalty. In the independent study it does match (§2.1), with a better
`aupr`. The thesis needs to know this either way.

### 3.3 What is recorded for every estimator and graph

1. **Along the path** (as in Figure 5, $\lambda$ chosen knowing the truth): `max_f1`, `auc`, `aupr`,
   `max_acc`.
2. **The graph that BIC selects on the path:** directed $F_1$, precision, recall, and the split
   of the true edges into correct / reversed / both directions kept / missed.
3. **That graph after the BIC search** with add / delete / reverse moves (`gclm.solvers.search`,
   S3b).
4. **The support at every $\lambda$,** so that other selection rules can be tried later without
   rerunning anything.

Item 1 answers "is the right graph on the path?", items 2 and 3 "does one get it without knowing
the truth?". The three are defined side by side, with the refit behind the BIC, in
[`docs/SEARCH.md`](../../docs/SEARCH.md) §2a.

### 3.4 The waves

| wave | question | cells | CPU-h (estimate) | code needed |
|---|---|---|---|---|
| **1** | the 2 × 2, LLA, the adaptive lasso, BIC and search; direct loss, $p = 10, 20$, $n = 10^3, 10^4, \infty$ | 16 × 3 = 48 (128 tasks per $n$) | 600 | the patch; LLA; adaptive lasso; selection inside the runner; submit script |
| **2** | search without any penalty (Améndola et al. 2020), and the search started from the true graph as a ceiling; $n = 10^3, 10^4, \infty$ | 3 × 3 = 9 (20 tasks per $n$) | 170 | a sharded version of the S3b driver |
| **3** | the log-likelihood loss at $p = 10$: the 2 × 2 for MCP, with the lasso in both orders; $n = 10^3, 10^4, \infty$ | 8 × 3 = 24 (56 tasks per $n$) | 180; $p = 20$ at one $n$: + 340 | none beyond wave 1 |
| **4** | larger $p$ (15, 25, 30, 40, 50) at $n = 10^3$, the setting of Figure 5 | chosen after wave 1 | 200 – 600 | none |
| **5** | three checks of the selection step: (a) at $p = 10$ the BIC with the maximised likelihood instead of the least-squares refit, with the search, for lasso / MCP dense → sparse / adaptive lasso, both $C$; (b) 100 randomly drawn starting graphs for the pure search, sparse and uniform (and 30 at $p = 20$ on 5 replicates); (c) the extended BIC term ($\gamma = 0.5, 1$) inside the selection and the search, next to the plain BIC on the same path, $p = 10, 20$, and in the pure search with $\gamma = 1$ | 6 + 2 (+ 1) + 6 + 3 per $n$ | (a) 15 – 30 per cell; (b) 15 per cell, the $p = 20$ cell 50 – 100; (c) about 190 per $n$, the pure search 170 for three $n$ | `--refit loglik`, `--starts`, `--ebic-gamma`, the result of every start; wave 5 in the submit script; `restarts.py` |
| later | a diagonal $C$ that is estimated; a DGP in which direction is identifiable | | | a new solver feature |

For comparison: the baseline was 36 cells and 720 CPU-h. With the 96 cores LRZ gives one user,
wave 1 is about 6 hours of computing and all four waves 12 to 16 (16 to 20 with $p = 20$ in
wave 3), plus waiting in the queue.

**Wave 1** *(the answer to the main question)*

- **Cells.** 8 estimators × 2 choices of $C$ × 3 sample sizes. Three of the sixteen exist in the
  baseline ($C = 2I$: lasso, standard MCP, standard SCAD). They are rerun, for two reasons: the
  baseline did not store items 2 – 4 of §3.3, and their path metrics must reproduce the October
  numbers, which tests the changed code on the cluster (cost: 16 CPU-h per $n$).
- **Sample sizes.** $10^3$, $10^4$, $\infty$. In the baseline $10^5$ and $\infty$ were nearly the
  same; $10^5$ can be added later for 200 CPU-h.
- **Cost.** Measured per cell on the cluster: lasso 1.4, standard MCP 6.5, standard SCAD 8.7
  CPU-h. From a timing pilot on the laptop (`files/time_direct_up.py`, four graphs): a
  dense → sparse path costs 2 to 5 standard paths. Estimated: LLA about three lasso paths, the
  adaptive lasso about two. BIC and search: about 3.5 per cell (15 – 40 s per graph at $p = 20$,
  measured in S3b). Together about 200 CPU-h per sample size, a quarter of it the search.
- **Submission.** 4 tasks per cell for the cheap paths (lasso, LLA, adaptive lasso), 8 for the
  standard MCP / SCAD paths and 16 for the dense → sparse paths, which cost most: 128 tasks per
  sample size, each of 1 to 3 hours. One sample size per round keeps each round under LRZ's
  limit of about 200 queued tasks.

**Wave 2** *(the fourth method of S3b, at scale)*

- **Search without a penalty:** the greedy BIC search from the empty graph and from 10 random
  graphs, best result kept (Améndola, Dettling, Drton, Onori & Wu 2020, §5).
- **Search started from the true graph:** not an estimator, but the ceiling for any search with
  this score. It separates "the score prefers another graph" from "the search gets stuck"
  (`next_steps_051026.md` §2).
- **Scope.** $p = 10$ with both choices of $C$; $p = 20$ with the rescaled $C$ only, because it is
  expensive there (200 – 560 s per graph, measured) and S3b already showed that with $C = 2I$ the
  search does not help. Three cells per sample size: `search_p10_C2I`, `search_p10_Cresc`,
  `search_p20_Cresc`.

**Wave 3** *(the other loss)*

- **Cells.** lasso and MCP × both path orders × both choices of $C$. Here the lasso has two
  orders as well, because this loss is not convex in $M$. The existing cells ($C = 2I$,
  sparse → dense) are rerun for items 2 and 4 of §3.3; the search is left out.
- **Left out for this loss:** LLA and the adaptive lasso (each of their steps is a full
  log-likelihood path, the expensive kind), and at first SCAD (decision 3 in §7).
- **Dense → sparse** starts from the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, as in Varando &
  Hansen, and is already implemented.
- **Cost, and why $p = 10$ first.** This is the expensive loss: a lasso cell cost 26 CPU-h in the
  baseline, nine tenths of it at $p = 20$. Timing pilot on the laptop (`files/time_loglik_up.py`,
  three graphs at $p = 10$, two at $p = 20$): the lasso costs about the same in both orders; at
  $p = 20$ an MCP dense → sparse path took 53 to 338 s where a lasso path took 120 to 180 s; and
  with the rescaled $C$ everything was slower on the one $p = 20$ graph timed (lasso 700 s against
  170 s, standard MCP 274 s against 16 s). From these few timings: about 60 CPU-h per sample size
  at $p = 10$, about 340 at $p = 20$.
- **So:** $p = 10$ at all three sample sizes first (180 CPU-h). $p = 20$ only if $p = 10$ shows
  a pattern worth it, and then at $n = 10^4$ first (340 CPU-h; the timing is uncertain).
- **Caution.** These fits depend on the machine for single graphs (S2b §2). Only means over the
  graphs of a cell count.
- **BIC** uses the same least-squares refit as for the direct loss, so that the selection rule is
  identical for all estimators and only the paths differ.

**Wave 4** *(the thesis figure; decided on 7 October from the finite-$n$ results of wave 1)*

- **Estimators:** lasso, MCP and SCAD on the standard path (the reference, which Figure 5 never
  had beyond the lasso), MCP and SCAD dense → sparse, the adaptive lasso; each with $C = 2I$ and
  the rescaled $C$. LLA is left out: it gains about half of dense → sparse and adds nothing to
  the story. 12 cells per $p$, $p = 15, 25, 30, 40, 50$, $n = 1000$, 25 replicates (400 graphs
  per cell); with wave 1's $p = 10, 20$ this gives Figure 5's whole axis.
- **BIC but no search:** one search at $p \ge 30$ would take many minutes per graph. `ebic1_f1`
  comes for free from the stored scores.
- **Cost.** One $p = 50$ graph ($k = 1$; denser ones cost 2 to 3 times more), BIC selection
  included: lasso 31 s, adaptive lasso 32 s, standard MCP 220 s, MCP dense → sparse 328 s. The
  cells at $p = 40, 50$ have twice the shards and 24 h; a task then takes half an hour to about
  three hours. About 700 CPU-h for the five sizes, $p = 50$ alone half of it; the population
  version of a subset (`--n inf --only Cresc`, $p = 30, 50$) would add about 250.
- **Open there:** at $p = 40, 50$ with $n = 1000$ there are more parameters than observations,
  and the dense start rests on a noisy fit. Whether dense → sparse still helps is not known.
  *Answered on 8 October (S4 §6a): it helps most there; the gap to the lasso grows with $p$ to
  +0.04 / +0.07 for MCP dense → sparse and +0.08 / +0.13 for the adaptive lasso at $p = 50$.*
- `cluster/submit_campaign.sh --wave 4 --p <p>`; cells are named `..._p<p>_n1000`.

**Wave 5** *(two checks of the selection step; decided on 7 October, implemented, not yet submitted)*

- **(a) The BIC proper.** The campaign's BIC scores a support by the Gaussian likelihood at the
  least-squares refit on the direct loss, not at the maximum-likelihood refit (`docs/SEARCH.md`
  §2a); Améndola et al. and Dettling use the maximised likelihood. `--refit loglik` switches the
  refit (`CovRefit`, iterative, through `Scorer(..., "loglik")`) for the BIC along the path and for
  the search alike. Cells `direct_<lasso|MCP-up|adaptive>-ml_<C2I|Cresc>`, $p = 10$, 800 graphs,
  `--select search`. On the laptop: 1 to 4 s per graph for the selection, 13 to 100 s for the
  search with every add move scored (`--add-screen` would cut that, but is not used, so that only
  the score differs from wave 1's search). On those three graphs (`files/time_refit_loglik.txt`)
  the two refits selected the same $\lambda$ twice and neighbouring ones once, where the likelihood
  refit was better by 0.03 to 0.04 in $F_1$. About 15 to 30 CPU-h per cell; 6 cells per $n$.
- **(b) The starting graphs of the pure search.** Ten had not saturated (S4 §4a). `--restarts 100`
  with `--starts sparse` (as in wave 2) and `--starts uniform` (a fair coin per entry: Nowzohour et
  al.'s uniform draw, which for our graph class needs no MCMC). The graph and the BIC of every
  start are stored (`m_pure_starts_support`, `pure_scores`, `pure_start_moves`), so one run gives
  the best of the first $r$ starts for every $r \le 100$ (`simulations/diagnostics/restarts.py`,
  table and `campaign_restarts.csv`). Cells `search100s_p10_Cresc`, `search100u_p10_Cresc`: 38 to
  46 s per graph on the laptop, about 15 CPU-h per cell. `search30s_p20_Cresc` (30 starts, 5
  replicates = 160 graphs, 32 shards, 24 h) is optional, 50 to 100 CPU-h. On the one graph timed
  per $p$ (`files/time_starts.py`) the uniform starts ended worse than the sparse ones: at $p = 20$
  the best of 5 uniform starts had 114 false positives against 3, and cost twice as much; uniform
  starting graphs are what the reference does, not necessarily what works in this class.
- **(c) The extended term, inside the search.** `docs/SEARCH.md` §2 writes the BIC with an
  optional extended term. It was used in S3b (Chen & Chen's form, $\gamma_e = 1$: the same graphs
  within 0.01) and in the campaign only offline, on the supports of a path (Dettling's
  $4\gamma|E|\log p$, columns `ebic05_*`, `ebic1_*`), never inside the selection-plus-search or
  the pure search. Since the plain BIC over-selects and the search prunes, the term may change
  where the search ends. `run_s1_shard.py --ebic-gamma 0.5 1` selects and searches once more per
  $\gamma$ on the same path (Dettling's form; fields `ebic05_*`, `ebic1_*` next to the plain-BIC
  ones, so with / without is paired graph by graph) and `run_search_shard.py --ebic-gamma 1`
  scores both of its searches with it. Cells `direct_<lasso|MCP-up|adaptive>-ebic_<C>` at
  $p = 10, 20$ (the paths are computed again, which is most of the cost: about 130 CPU-h per $n$,
  MCP dense → sparse 100 of them; the two extra searches about 60) and `searche1_p10_C2I`,
  `searche1_p10_Cresc`, `searche1_p20_Cresc` (as wave 2: about 170 CPU-h for three $n$, the
  $p = 20$ cell most of it). Cheapest informative subset: the lasso and adaptive-lasso cells at
  $n = 10^4$ (about 40 CPU-h) and `searche1_p10_Cresc` (a few CPU-h per $n$).
- **Found while timing.** With the rescaled $C$ the least-squares refit of the empty graph is
  unstable, so its BIC is $+\infty$ and the search cannot leave it (a dead start), for 6 to 7 % of
  the wave 2 graphs at $p = 10$ and 24 to 25 % at $p = 20$; the refit of the true support is
  unstable for 2 to 5 % (mostly `C_Random_Full`). The best of the 11 starts was dead once in 2 375
  graphs. The likelihood refit cannot be unstable. Numbers in `docs/SEARCH.md` §2a and S4 §4a.
- **Suggested order:** (a) at $n = 10^4$ and (b) at $p = 10$ for all three $n$ first (about 200
  CPU-h); the rest of (a), and the $p = 20$ cell, if those show something. Block 11 of
  `cluster_commands_051026.md`.

**Later, not in this campaign**

- **A diagonal $C$ that is estimated** (§2.4, §2.5).
- **A second data-generating process** in which direction is identifiable (edge weights bounded
  away from zero, no 2-cycles; direction B of `next_steps_051026.md`). It costs wave 1 once more.
- **The variance-ordering baseline** of caveat 3 (§2.2) for all cells needs no cluster: it is
  computed on the laptop from the stored supports.
- **An idea, not worked out:** a DGP whose variables all have variance 1. Then $C = 2I$ is right on
  the standardised scale, the two choices of $C$ coincide, and the variances carry no information
  about direction. My first construction (rescaling the diagonal of $M$ until the variances are 1)
  makes about half of the drift matrices unstable, so this needs more thought.

### 3.5 What I expect, written down before the run

1. **`C_ID`, rescaled $C$, dense → sparse:** MCP and SCAD beat the lasso in `max_f1` by about
   +0.03 at $n = 10^3$ and +0.07 to +0.10 at $n \ge 10^4$ for $p = 10$, and by more for $p = 20$.
2. **`C_Random_Min_Diag`** behaves like `C_ID`. **`C_Random_Diag`** shows smaller gains.
   **`C_Random_Full`** shows none.
3. **With $C = 2I$,** dense → sparse ties the lasso at $n = 10^3$ and gains at most +0.05 at
   large $n$.
4. **The standard paths lose with either $C$:** MCP clearly, SCAD slightly.
5. **LLA and the adaptive lasso** match MCP dense → sparse in `max_f1`; the adaptive lasso is
   better in `aupr`. With $C = 2I$, LLA gains a little at $n \ge 10^4$ (+0.03 to +0.04 in the
   independent study).
6. **BIC-selected graphs:** the same order, with smaller differences at $n = 10^3$. The search
   adds +0.01 to +0.03 with the rescaled $C$ and nothing with $C = 2I$.
7. **Log-likelihood:** for the lasso the order makes little difference (§2.5). For MCP it is
   open: in the timing pilot (four graphs, so only a hint) dense → sparse helped with the
   rescaled $C$ and hurt with $C = 2I$.

### 3.6 How the results will be read

- **"MCP / SCAD dense → sparse beat the lasso when $C$ is specified correctly"** stands if the
  paired difference to the lasso with the same $C$ is positive with $z > 3$ for `C_ID` at
  $n \ge 10^4$, for both $p$.
- **It extends to a mildly wrong $C$** if the same holds for `C_Random_Min_Diag`, and to a wrong
  diagonal in general if it holds for `C_Random_Diag`.
- **"The nonconvex penalty is needed"** only if MCP / SCAD (dense → sparse or LLA) also beat the
  adaptive lasso. Otherwise the message is "start from the lasso, then prune", and MCP is one way
  of doing it. I expect this second outcome.
- **If nothing survives outside `C_ID`,** $C$ is the bottleneck, and estimating $C$ moves from
  "later" to "next".

### 3.7 A first look on the laptop (5 October, evening)

While the code was being written, the laptop ran the replication of 3 October on the three
settings where the true $C$ is not $2I$: 20 graphs per setting, $p = 10$, the repository's solvers
(`files/replicate_other_c.csv`, `files/summarize_other_c.py`). It is the question of wave 1 in
small. $n = 10^3$ and $10^4$ finished; $n = \infty$ did not for the denser graphs and is left to
the cluster.

`max_f1`: lasso → MCP dense → sparse, both fitted with the same $C$. In brackets: $z$ of the
paired difference. The `C_ID` rows are the 40 graphs of 3 October.

| true $C$ | fitted with | $n = 10^3$ | $10^4$ |
|---|---|---|---|
| `C_ID` (40) | $C = 2I$ | 0.622 → 0.625 (+0.2) | 0.664 → 0.675 (+0.6) |
| `C_ID` (40) | rescaled $C$ | 0.637 → 0.662 (+1.2) | 0.673 → 0.747 (+4.2) |
| `C_Random_Min_Diag` (20) | $C = 2I$ | 0.602 → 0.637 (+1.4) | 0.678 → 0.712 (+1.6) |
| `C_Random_Min_Diag` (20) | rescaled $C$ | 0.598 → 0.637 (+2.0) | 0.672 → 0.744 (+3.1) |
| `C_Random_Diag` (20) | $C = 2I$ | 0.589 → 0.594 (+0.1) | 0.631 → 0.676 (+1.8) |
| `C_Random_Diag` (20) | rescaled $C$ | 0.559 → 0.621 (+2.9) | 0.608 → 0.701 (+2.8) |
| `C_Random_Full` (20) | $C = 2I$ | 0.500 → 0.455 (−4.0) | 0.505 → 0.465 (−3.2) |
| `C_Random_Full` (20) | rescaled $C$ | 0.525 → 0.500 (−1.6) | 0.536 → 0.518 (−1.0) |

**Reading, with 20 graphs per setting in mind:**

- **The gain is there when the true $C$ is diagonal,** also when it is not $2I$: +0.04 to +0.09
  with the rescaled $C$ in both random diagonal settings, as large as for `C_ID`. Expectation 2
  of §3.5 was too cautious about `C_Random_Diag`.
- **It is not there when the true $C$ is not diagonal** (`C_Random_Full`): MCP dense → sparse is
  0.02 to 0.05 below the lasso.
- **SCAD dense → sparse** gives the same picture (rescaled $C$: +0.04 / +0.09 for
  `C_Random_Min_Diag`, +0.05 / +0.08 for `C_Random_Diag`, −0.00 / −0.01 for `C_Random_Full`).
- **The standard MCP path loses in every setting** with either $C$ (−0.05 to −0.15).
- **After BIC and the search the picture is less clear** at these sample sizes: +0.02 to +0.06 at
  $n = 10^3$ and −0.01 to +0.03 at $10^4$ for the diagonal settings, −0.01 to −0.02 for
  `C_Random_Full`. Most of these are within noise for 20 graphs.

So the first look agrees with the plan: the claim seems to extend to a wrong but diagonal $C$,
and to stop at a non-diagonal one. Wave 1 has 200 graphs per setting and $p$ instead of 20.

---

## 4. Where the code is

*Written on 5 October after the go. Everything below has tests, and the defaults of the library
are unchanged. Line numbers are those of that day.*

### 4.1 The map

| what | where | function or place | switched on by |
|---|---|---|---|
| **the rescaled $C$** | `src/gclm/data/simulate.py:129` | `estimation_volatility`; the scales $s_i$ come from `draw_instance(..., return_scale=True)` (line 90) | `--c-scale variance` |
| lasso; MCP / SCAD on the **standard path** (sparse → dense) | `src/gclm/solvers/path.py:110` | `lasso_path`; the loop from line 254 | default |
| MCP / SCAD **dense → sparse** | `src/gclm/solvers/path.py:226` | `lasso_path`, the block `if direction == "up" and penalty != "lasso":` | `--direction up` |
| MCP / SCAD **by LLA** | `src/gclm/solvers/path.py:267` | `lla_path`; its weights: `lla_weights`, `src/gclm/objective/penalties.py:148` | `--method lla` |
| **adaptive lasso** | `src/gclm/solvers/path.py:348` | `adaptive_lasso_path` | `--method adaptive` |
| the solver under all of them | `src/gclm/solvers/proxgrad.py:22` | `solve_fista`: FISTA for the lasso with any weights; for MCP / SCAD it hands over to `_solve_mapg` (line 79) | |
| log-likelihood paths in both orders | `src/gclm/solvers/path.py:426` | `covloss_path(direction=...)`; the dense start is `dense_fit`, `src/gclm/objective/covariance.py:130` | `--loss loglik --direction up` |
| **BIC along a path** | `src/gclm/solvers/search.py:282` | `bic_along_path`; refit `DirectRefit` (line 50), score `bic` (127), cache `Scorer` (167) | `--select bic` |
| **BIC search** (add / delete / reverse) | `src/gclm/solvers/search.py:216` | `greedy_search`; the moves: `neighbours` (187) | `--select search` |
| what one cell computes and stores | `simulations/run_s1_shard.py:175` | `run_one` (the path) and `select_graph` (line 134: BIC and search) | |
| **search without a penalty**, and from the truth | `simulations/run_search_shard.py:71` | `run_one`; it uses `multistart_search` and `random_support` (`search.py:270`, `257`) | wave 2 |
| the cells of each wave | `cluster/submit_campaign.sh:63` | `cells()`: one line per cell | `--wave` |
| one cell as a SLURM job array | `cluster/campaign_array.sbatch` | | |
| a rehearsal without a cluster | `cluster/local/sbatch` | a stand-in for `sbatch` | |
| tables from the shards | `simulations/diagnostics/campaign.py` | `load` (138), `means_table` (178), `paired_table` (194), `check_baseline` (251) | |
| the settings | `src/gclm/config.py` | `S1Config.direction` (71), `.c_scale` (77), `.method` (83) | |

The definitions with formulas, and what each test checks, are in
[`docs/DENSE_START.md`](../../docs/DENSE_START.md) (the new estimators) and
[`docs/SEARCH.md`](../../docs/SEARCH.md) (BIC and search). The cluster procedure is also in
[`docs/REPRODUCTION.md`](../../docs/REPRODUCTION.md) §2.7.

**Added for wave 5 (7 October).** `simulations/run_s1_shard.py --refit {direct,loglik}`,
`--add-screen N` and `--ebic-gamma G...` (`select_graph`; the likelihood refit is
`gclm.solvers.search.CovRefit` behind `Scorer(..., "loglik")`; the extended term is
`gclm.solvers.search.bic(..., ebic_form)` with `EBIC_FORM = "dettling"`, fields `ebic<G>_*`);
`simulations/run_search_shard.py --starts {sparse,uniform}`, `--ebic-gamma G` and the per-start
fields `m_pure_starts_support`, `pure_start_moves`; `simulations/diagnostics/restarts.py`
(best of the first $r$ starts); `cluster/submit_campaign.sh --wave 5`; the analysis script knows the
`-ml` estimators and the `search100s` / `search100u` / `search30s` cells. Tests:
`tests/test_campaign_runner.py` (the likelihood refit), `tests/test_search_shard.py` (uniform starts,
per-start results), `tests/test_campaign_submit.py` (wave 5), `tests/test_restarts_analysis.py`.

### 4.2 The main pieces, with the code (abridged)

**The rescaled $C$** (`src/gclm/data/simulate.py:129`). `scale` holds the standard deviations the
data were divided by:

```python
def estimation_volatility(scale, c_scale="identity"):
    if c_scale == "identity":
        return 2.0 * np.eye(scale.shape[0])      # C = 2I: Dettling's pipeline
    return np.diag(2.0 / scale ** 2)             # C = 2 diag(1/s_i^2): the rescaled C
```

**MCP / SCAD dense → sparse** (`src/gclm/solvers/path.py:226`, inside `lasso_path`). `fit` is
`solve_fista` with `penalty="MCP"` or `"SCAD"`; `warm` carries each solution into the next problem:

```python
if direction == "up" and penalty != "lasso":
    # 1. the dense start: the LASSO solution at the smallest lambda
    dense = lasso_path(sigma, c, lambdas=lambdas, solver="fista", zero_tol=0.0, ...)
    # 2. walk the grid from the smallest lambda to the largest
    order = np.argsort(lambdas)
    warm = dense.estimates[order[0]]
    for i in order:
        lam = lambdas[i]
        if lam >= lam_max:
            warm = diagonal_fit(sigma, c)        # 3. the path ends at the empty graph
        else:
            warm = fit(sigma, c, lam, weights=weights, m_init=warm, **solver_kwargs)
        up[i] = warm.copy()
```

The standard path is the loop right below it (line 254): the same call to `fit`, but over
`lambdas[::-1]`, from $\lambda_{\max}$ downwards, starting from nothing. That is the whole
difference between the two estimators.

**MCP / SCAD by LLA** (`src/gclm/solvers/path.py:267`, `lla_path`). For every $\lambda$, starting
from the lasso solution `m` at that $\lambda$:

```python
for _ in range(steps):                                   # steps = 2
    w = off * lla_weights(m, lam, penalty, gamma)        # P'(|m|) / lam: 1 at zero, 0 beyond gamma*lam
    m_new = solve_fista(sigma, c, lam, weights=w, m_init=m, tol=tol)   # a weighted LASSO: convex
    fixed = np.array_equal(m_new != 0, m != 0) and np.max(np.abs(m_new - m)) < 10 * tol
    m = m_new
    if fixed:
        break
```

**Adaptive lasso** (`src/gclm/solvers/path.py:348`, `adaptive_lasso_path`). `pilot` is the lasso
solution at the smallest $\lambda$:

```python
size = np.abs(pilot)
kept = off & (size > 0)
weights = np.where(off, np.inf, 0.0)             # inf: excluded; 0: the diagonal, unpenalised
weights[kept] = size[kept] ** (-power)           # power = 1: weight 1 / |pilot entry|
weights[kept] /= weights[kept].min()             # the largest pilot entry gets weight 1
...
warm = solve_fista(sigma, c, lam, weights=weights, m_init=warm, tol=tol)   # along its own grid
```

**BIC and search in a cell** (`simulations/run_s1_shard.py:134`, `select_graph`). `supports` are
the supports of the path, `c_est` the same $C$ the path was fitted with:

```python
scorer = Scorer(sigma_hat, c_est, n_obs, "direct")       # least-squares refit + Gaussian BIC, cached
ib, scores = bic_along_path(sigma_hat, c_est, n_obs, supports, scorer=scorer)   # ib: the BIC-selected lambda
...
res = greedy_search(sigma_hat, c_est, n_obs, supports[ib], scorer=scorer,       # start: the BIC-selected graph
                    max_steps=p * (p - 1))                                       # a guard; it stops by itself
```

**Which function a cell calls** (`simulations/run_s1_shard.py:175`, `run_one`):

| cell name contains | runner arguments | library call |
|---|---|---|
| `lasso` | `--penalty lasso` | `lasso_path(...)` |
| `MCP`, `SCAD` | `--penalty MCP` | `lasso_path(..., penalty="MCP")` |
| `MCP-up`, `SCAD-up` | `--penalty MCP --direction up` | `lasso_path(..., penalty="MCP", direction="up")` |
| `MCP-lla`, `SCAD-lla` | `--penalty MCP --method lla` | `lla_path(..., penalty="MCP")` |
| `adaptive` | `--method adaptive` | `adaptive_lasso_path(...)` |
| `loglik_...` | `--loss loglik [--direction up]` | `covloss_path(..., "loglik", direction=...)` |
| `..._C2I`, `..._Cresc` | `--c-scale identity`, `variance` | `estimation_volatility(scale, ...)` |

`bash cluster/submit_campaign.sh --wave 1 --list` prints the full table of cells with their
arguments.

### 4.3 What a shard file contains

One `.npz` per array task, one entry per graph.

| fields | content |
|---|---|
| `p`, `k`, `c_choice`, `rep` | the graph; `c_choice` indexes `c_choice_names` |
| `lambdas`, `lambda_max` | the grid (for the adaptive lasso: its own) |
| `conf_offdiag` | tp, fp, tn, fn at each of the 100 $\lambda$: everything behind `max_f1`, `auc`, `aupr` |
| `m_true_i/j/v` | the true drift matrix, sparse |
| `supports_packed`, `scale` | the support at every $\lambda$ (packed bits) and the standard deviations $s_i$ |
| `bic_index`, `bic_scores`, `bic_conf`, `bic_orient` | the BIC-selected $\lambda$, the BIC of all 100 supports, the counts of the selected graph and its orientation breakdown |
| `m_search_support`, `m_search_i/j/v`, `search_conf`, `search_orient`, `search_score`, `search_moves` | the graph after the BIC search: support, unpenalised refit, counts, BIC, number of add / delete / reverse moves |
| `config_json`, `provenance_json` | every setting of the run; host, versions, time |

The orientation breakdown has the order `ORIENT` of `run_s1_shard.py`: correct, reversed, hedged
(both directions kept for a single true edge), both, half (for a true 2-cycle), missed single,
missed double, false-positive single, false-positive double. Wave 2 files have the same kind of
fields under `pure_*` and `truth_*`.

```python
import sys
import numpy as np
sys.path.insert(0, "simulations")
from run_s1_shard import ORIENT, unpack_supports

d = np.load("runs/campaign/direct_MCP-up_Cresc_n1000/shards/shard_0000_of_0016.npz", allow_pickle=True)
i = 0                                                    # the first graph of this shard
p, n_lambda = int(d["p"][i]), len(d["lambdas"][i])
supports = unpack_supports(d["supports_packed"][i], n_lambda, p)   # (100, p, p) booleans
bic_graph = supports[d["bic_index"][i]]                  # [j, i] true: the edge i -> j is selected
searched = unpack_supports(d["m_search_support"][i], 1, p)[0]
print(dict(zip(ORIENT, d["search_orient"][i])))
```

### 4.4 Tests

| file | tests | what they establish |
|---|---|---|
| `tests/test_direction_cscale.py` | 5 | the rescaled $C$ makes the true graph fit exactly at $n = \infty$ and $2I$ does not; the dense → sparse path is stationary at every $\lambda$ and differs from the standard one; defaults unchanged |
| `tests/test_lla_adaptive.py` | 16 | the LLA weights; every weighted lasso satisfies its optimality conditions and agrees with a second solver; a fixed point of LLA is stationary for MCP / SCAD; LLA at one $\lambda$ does not depend on the grid; adaptive weights, excluded entries, equality with the lasso for equal weights |
| `tests/test_campaign_runner.py` | 12 | the runner end to end: default output unchanged; supports, BIC and search fields consistent with each other and with the truth; unimplemented combinations refused |
| `tests/test_search_shard.py` | 3 | the wave 2 runner: stored scores are the BIC of the stored graphs; random starts depend on the graph only; at $n = \infty$ with the right $C$ the search stays at the truth |
| `tests/test_campaign_submit.py` | 7 | the submit script against a stub `sbatch`: the cells of each wave, every cell accepted by its runner, one submission per cell, `--status`, `--fill`, a refused cell |
| `tests/test_campaign_analysis.py` | 4 | the tables: metrics from stored counts equal the library's; loader and paired differences on a tiny campaign |

```bash
uv run --with pytest python -m pytest -q -m "not r"        # laptop: the whole suite without R
```

### 4.5 What was checked before submitting

1. **The defaults are unchanged.** Without the new options the runner writes exactly the fields
   it wrote before (`test_default_output_is_what_it_was_before_the_campaign_options`). On 54
   graphs taken from the cluster shards of the n-sweep (`direct_lasso_n1000`, `direct_MCP_n1000`,
   `direct_SCAD_ninf`; $p = 10$ and $20$) the changed code reproduces the stored counts at all 100
   $\lambda$: lasso 18 of 18, MCP 18 of 18, SCAD 16 of 18. On the SCAD graphs it is identical to
   the laptop run of 2 October made with the old code (10 of 10), so the two differences are the
   known dependence of nonconvex paths on the machine (S2b §2), not a change in the code.
2. **A rehearsal of the whole chain.** All cells of waves 1 and 3 and the $p = 10$ cells of wave 2
   (26 cells) ran on the laptop the way they will run on LRZ: `submit_campaign.sh` → a stand-in
   for `sbatch` (`cluster/local/sbatch`) → the runners → `--status` → `campaign.py`. Reduced to
   $p = 10$ and one replicate: 16 graphs per cell, $n = 1000$. Every cell completed and the tables
   came out. Minutes per cell on the laptop:

   | 16 graphs, $p = 10$ | $C = 2I$ | rescaled $C$ |
   |---|---|---|
   | direct: lasso / adaptive lasso | 0.6 / 0.7 | 0.6 / 0.7 |
   | direct: MCP / SCAD, standard path | 2.3 / 3.3 | 2.0 / 2.9 |
   | direct: MCP / SCAD, dense → sparse | 5.8 / 6.7 | 4.9 / 6.1 |
   | direct: MCP / SCAD by LLA | 1.4 / 1.3 | 1.2 / 1.2 |
   | search without a penalty, and from the truth | 1.8 | 1.9 |
   | log-likelihood: lasso, sparse → dense / dense → sparse | 8.2 / 9.1 | 22.7 / 23.8 |
   | log-likelihood: MCP, sparse → dense / dense → sparse | 1.3 / 4.3 | 14.8 / 16.9 |

   The direct-loss cells include BIC and the search. The log-likelihood loss is three times
   slower with the rescaled $C$ for the lasso and more than ten times for the standard MCP path;
   the shard counts of wave 3 allow for it.
3. **Four heavy cells at $p = 20$,** two graphs each, through the runners: SCAD dense → sparse
   with BIC and search took 87 and 334 s per graph, MCP by LLA 15 and 304 s, the adaptive lasso 13
   and 106 s, the search without a penalty 304 s. The slow graph is the same one each time: its
   lasso path is slow, its BIC-selected graph is dense, and the search then needs about a hundred
   deletions (90 to 120 s). The shard counts are sized for this: about 1 to 3 hours per task, with
   a limit of 12 hours.
4. **The tests:** 342 pass on the laptop (320 without R, 22 with R); 24 are skipped because
   optional packages are not installed.

---

## 5. What you run

*The same commands as a stand-alone sheet, block by block with what to expect and a submission
log to fill in: [`cluster_commands_051026.md`](cluster_commands_051026.md).*

Copy from top to bottom. What each block does, and what to expect, is below the sheet.

```bash
# ---- 1. laptop: commit and push the code (the cluster gets it with git pull)
git add .gitignore README.md ARCHITECTURE.md docs src cluster tests \
        simulations/run_s1.py simulations/run_s1_shard.py simulations/run_search_shard.py \
        simulations/diagnostics
git status --short | grep -v '^??'          # what is staged; no shards, no runs/
git commit -m "campaign: rescaled C, dense-to-sparse, LLA, adaptive lasso, BIC selection and search on the cluster"
git push

# ---- 2. LRZ: every login
ssh -Y ge47xod3@cool.hpc.lrz.de
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate

# ---- 3. LRZ: update and check (about 3 minutes)
git pull
python -m pytest -q tests/test_campaign_runner.py tests/test_campaign_submit.py \
    tests/test_search_shard.py tests/test_lla_adaptive.py tests/test_direction_cscale.py
bash cluster/submit_campaign.sh --wave 1 --list
bash cluster/submit_campaign.sh --wave 1 --dry-run | tail -2

# ---- 4. LRZ: a canary of 4 small tasks, to see that the new job script starts
bash cluster/submit_campaign.sh --wave 2 --n 1000 --only p10
squeue -M serial -u $USER
tail -n 4 logs/camp_*.out                   # after 2 to 3 minutes: see below
cat logs/camp_*.err

# ---- 5. LRZ: the first round (128 + 56 tasks)
bash cluster/submit_campaign.sh --wave 1 --n 1000
bash cluster/submit_campaign.sh --wave 2

# ---- 6. LRZ: the next rounds, each once the queue has room
squeue -M serial -u $USER -h -r | wc -l     # tasks queued or running right now
bash cluster/submit_campaign.sh --wave 1 --n 1e4       # 128 tasks: needs the count below about 70
bash cluster/submit_campaign.sh --wave 1 --n inf       # 128 tasks: again below about 70
bash cluster/submit_campaign.sh --wave 3               # 168 tasks: below about 30

# ---- 7. LRZ: watch
bash cluster/submit_campaign.sh --wave 1 --status
bash cluster/submit_campaign.sh --wave 2 --status
bash cluster/submit_campaign.sh --wave 3 --status
sacct -M serial -X -u $USER -S now-2days --format=JobName%10,JobID%18,State,Elapsed | grep -v COMPLETED
less logs/camp_<jobid>_<task>.err

# ---- 8. LRZ: repair, only after every job of that wave has ended
bash cluster/submit_campaign.sh --wave 1 --fill
bash cluster/submit_campaign.sh --wave 1 --fill --time 24:00:00       # after a TIMEOUT

# ---- 9. laptop, from the repository root: bring it home and look
scp -r ge47xod3@cool.hpc.lrz.de:~/repo/runs/campaign runs/
python simulations/diagnostics/campaign.py --check-baseline
```

**What each block does, and what to expect**

1. **Commit and push.** You commit; I have not. The `git add` line stages 30 files: the library,
   the runners, the cluster scripts, the tests and the documentation.
   - It leaves out `runs/` and `next_steps/` (this note included), which you may want to commit
     separately.
   - `src/gclm/solvers/search.py` and its tests have been untracked since S3b. The cluster needs
     them, and `git add src tests` picks them up.
   - What you staged earlier (the independent study's README, figures and tables) goes into the
     same commit unless you unstage it first.
2. **Every login.** `sbatch` needs nothing activated. `python` needs the module and the venv.
3. **Update and check.**
   - If `git pull` is refused because of local edits on LRZ: `git stash && git pull && git stash pop`.
   - The tests should end with `43 passed`. They run the runners on tiny problems and the submit
     script against a stand-in for `sbatch`; nothing is submitted.
   - `--list` prints the 16 cells of wave 1. The dry run should end with
     `would submit 384 tasks in 48 cells`.
4. **The canary.** `cluster/campaign_array.sbatch` is new, so four small tasks go first: the search
   without a penalty at $p = 10$ (two cells, two tasks each, 20 to 40 minutes per task).
   - After two or three minutes each `.out` file should show a line `python=...`, a line
     `host=... runner=simulations/run_search_shard.py ...`, a line `shard 0/2: 200 datasets [...]`,
     and soon after a first progress line `10/200 ...`.
   - The `.err` files should be empty. If a task has failed, send me its `.err`.
   - These four tasks are part of wave 2; they are not thrown away.
5. **The first round.** Wave 1 at $n = 1000$ (16 cells, 128 tasks) and the rest of wave 2 (7 cells,
   56 tasks). With the canary that is 188 tasks, just under LRZ's limit of about 200. If the
   second command is refused part of the way, run it again once some tasks have finished.
6. **The next rounds.** LRZ runs 96 of your tasks at a time and accepts about 200 in the queue.
   - Each command needs room for all its tasks; the comment gives the queue count below which
     it fits.
   - If `sbatch` refuses a cell (`AssocMaxSubmitJobLimit`), the script stops and says how many
     tasks went in. Nothing is recorded for the refused cell. Run the same command again later; it
     skips what is already submitted.
   - Wave 3 can also go in pieces: `--wave 3 --n 1000`, then `--n 1e4`, then `--n inf` (56 tasks
     each).
7. **Watch.** `--status` prints one line per cell: `complete`, `submitted (k/N shards written)` or
   `not started`. `sacct ... | grep -v COMPLETED` lists jobs that failed or ran out of time.
8. **Repair.** `--fill` resubmits exactly the missing shards. Use it only when `squeue` shows no
   job of that wave any more; otherwise shards that are still running are submitted twice.
9. **Bring it home.** About 350 MB with all three waves. `campaign.py` writes the tables into
   `runs/campaign/` and, with `--check-baseline`, compares the three cells that repeat the n-sweep
   with `runs/nsweep_p10-20` graph by graph. The lasso must agree exactly; MCP and SCAD may differ
   on a handful of graphs (S2b §2).

**Feeding the queue automatically.** From 7 October `cluster/feed_queue.sh` runs a plan of submit
commands (`cluster/plan_071026.txt`: the rest of wave 4, the suggested subset of wave 5, the
repairs) as the queue has room, so nobody has to watch `squeue`; block 12 of
`cluster_commands_051026.md`.

**How long.** Per task 1 to 3 hours for waves 1 and 2, up to a few hours for wave 3.

| round | tasks | CPU-h (estimate) | at 96 cores |
|---|---|---|---|
| wave 1, one sample size | 128 | 200 | 2 to 4 hours |
| wave 2, all three sample sizes | 60 | 170 | 2 to 4 hours |
| wave 3, all three sample sizes | 168 | 180 – 300 | 2 to 4 hours |
| wave 4, all five $p$ | 784 | about 700 | 8 hours, over a day with queueing |
| wave 5a, one sample size | 48 | 90 – 180 | 1 to 3 hours |
| wave 5b at $p = 10$, three sample sizes | 48 | about 45 | about an hour |
| wave 5b at $p = 20$ (optional) | 32 | 50 – 100 | 2 to 3 hours |
| wave 5c, paths with three scores, one sample size | 64 | about 190 | 2 to 4 hours |
| wave 5c, pure search with $\gamma = 1$, three sample sizes | 60 | about 170 | 2 to 4 hours |

So the first results (wave 1 at $n = 1000$) are back a few hours after block 5, and everything
within about a day if the rounds follow each other.

---

## 6. Order of work until the 10 – 11 October meeting

| when | what |
|---|---|
| done on 5 October | the code for waves 1 – 3 with tests; the rehearsal of every cell on the laptop (§4.5); the first look at the other settings of the true $C$ (§3.7) |
| now | you: commit, push, and blocks 2 – 5 of §5 |
| while wave 1 runs | I: the figures for the campaign and the skeleton of the write-up `simulations/S4_campaign.md` |
| wave 1 back | the analysis of the 2 × 2; choose the estimators for wave 4 (larger $p$) |
| last | wave 4; the two-page summary for the meeting |

## 7. Decided on 5 October

You confirmed the plan as written, so I took the option I had recommended at each open point. Say
so if you want one of them changed; each is a one-line change in `cluster/submit_campaign.sh`.

1. **Waves 1 – 3 go ahead;** wave 4 (larger $p$) after wave 1.
2. **The patch of the independent study is in `src/`,** without its `.gitignore` line, which
   pointed at a folder that does not exist here. Defaults are unchanged (§4.5).
3. **Wave 3 has the lasso and MCP at $p = 10$.** SCAD and $p = 20$ can be added once it shows a
   pattern worth completing.
4. **Three sample sizes** ($10^3$, $10^4$, $\infty$) in every wave. `--n 1e5` adds the fourth.
5. **The estimated $C$** stays for after the meeting, unless wave 1 shows that $C$ is the
   bottleneck.
6. **Wave 5 (7 October):** implemented and tested, in the submit script as `--wave 5`; whether and
   which subset to submit is your call (§3.4 suggests (a) at $n = 10^4$, (b) at $p = 10$ and the
   cheap cells of (c) first).

## 8. Scripts of this note

All in [`files/`](files/); run them from the repository root. None of them changes `src/`.

| script | what it does | output | time |
|---|---|---|---|
| `fit_check_free_diagonal.py` | fit of the true support under three assumptions about $C$ (§2.2) | printed | seconds |
| `variance_ordering_baseline.py` | the rule that orients the lasso's pairs by variance (§2.2) | `variance_ordering_baseline.csv` | 3 min |
| `estimate_c_check.py`, `estimate_c_check.R` | Varando & Hansen's package with four treatments of $C$ (§2.4); needs R with `gclm`; `--markdown` prints the tables of §2.4 | `estimate_c_check.csv` | 30 min |
| `loglik_order_check.py` | the repository's log-likelihood lasso in both path orders (§2.5) | `loglik_order_check.csv` | 25 min |
| `time_direct_up.py`, `time_loglik_up.py` | timing pilots for waves 1 and 3 (§3.4) | `time_direct_up.txt`, `time_loglik_up.txt` | 15 min, 1 h |
| `time_refit_loglik.py`, `time_starts.py` | timing pilots for wave 5 (§3.4): the BIC selection and the search with the least-squares against the maximised-likelihood refit on three $p = 10$ graphs; the pure search from 100 sparse and 100 uniform starting graphs at $p = 10$ and 5 of each at $p = 20$ (7 October) | `time_refit_loglik.txt`, `time_starts.txt` | 3 min, 8 min |
| `why_rescaling_helps.py` | why the rescaled $C$ helps the dense-start estimators: misfit of the truth, the dense shift of the solution set, ranking quality of the dense end, under both $C$ (7 October) | printed | 5 min |
| `summarize_other_c.py` | the first look at the other settings of the true $C$ (§3.7); the run itself is `../031026/files/replicate_dense_to_sparse.py --c C_Random_Diag C_Random_Min_Diag C_Random_Full --reps 5` | `replicate_other_c.csv` | 80 min, stopped before $n = \infty$ finished |
