# 5 October 2026: the question about $C$, and the cluster campaign

*Working notes. They answer the three questions of 5 October (has MCP / SCAD been run with the
rescaled $C$? does rescaling $C$ make sense? what does Varando & Hansen's paper do?) and turn
directions A, D, E and F of [`next_steps_051026.md`](next_steps_051026.md) into one cluster
campaign that includes the side experiments: adaptive lasso, search, log-likelihood loss,
larger $p$. The verdicts so far are in [`../../simulations/VERDICTS.md`](../../simulations/VERDICTS.md).*

**Status: a proposal. No library code is changed and nothing is submitted until Joon confirms.**
The new numbers in §2 and the cost estimates in §3 come from small laptop checks made to answer
the questions, about an hour of wall-clock time in all. Their scripts and outputs are in
[`files/`](files/) and listed in §8.

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
7. **Before anything is submitted** I need about half a day for the code of wave 1 (§4), and
   **your go** (§7).

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

**My answer: yes, with three caveats.**

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
the truth?".

### 3.4 The waves

| wave | question | cells | CPU-h (estimate) | code needed |
|---|---|---|---|---|
| **1** | the 2 × 2, LLA, the adaptive lasso, BIC and search; direct loss, $p = 10, 20$, $n = 10^3, 10^4, \infty$ | 16 × 3 = 48 | 600 | the patch; LLA; adaptive lasso; selection inside the runner; submit script |
| **2** | search without any penalty (Améndola et al. 2020), and the search started from the true graph as a ceiling; $n = 10^3, 10^4, \infty$ | 6 | 170 | a sharded version of the S3b driver |
| **3** | the log-likelihood loss at $p = 10$: the 2 × 2 for MCP, with the lasso in both orders; $n = 10^3, 10^4, \infty$ | 8 × 3 = 24 | 180; $p = 20$ at one $n$: + 340 | none beyond wave 1 |
| **4** | larger $p$ (15, 25, 30, 40, 50) at $n = 10^3$, the setting of Figure 5 | chosen after wave 1 | 200 – 600 | none |
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
- **Submission.** 4 tasks per cell: 64 tasks per sample size, 192 for all three. That is at
  LRZ's limit of about 200 queued tasks, so two rounds (two sample sizes, then the third).

**Wave 2** *(the fourth method of S3b, at scale)*

- **Search without a penalty:** the greedy BIC search from the empty graph and from 10 random
  graphs, best result kept (Améndola, Dettling, Drton, Onori & Wu 2020, §5).
- **Search started from the true graph:** not an estimator, but the ceiling for any search with
  this score. It separates "the score prefers another graph" from "the search gets stuck"
  (`next_steps_051026.md` §2).
- **Scope.** $p = 10$ with both choices of $C$; $p = 20$ with the rescaled $C$ only, because it is
  expensive there (200 – 560 s per graph, measured) and S3b already showed that with $C = 2I$ the
  search does not help.

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

**Wave 4** *(the thesis figure)*

- **Why after wave 1.** With all 16 estimators the five larger values of $p$ cost about 1,400
  CPU-h for one sample size; with a reduced list (lasso, standard MCP, and the two or three best
  of wave 1) 200 to 600. Wave 1 shows which estimators are worth it.
- **BIC but no search** at $p \ge 30$: one search would take many minutes per graph.
- **Open there:** at $p = 40, 50$ with $n = 1000$ there are more parameters than observations,
  and the dense start rests on a noisy fit. Whether dense → sparse still helps is not known.

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

---

## 4. Code to write before anything is submitted

All with tests, defaults unchanged, nothing committed by me.

1. **The patch of the independent study** into `src/` and the two runners: `--c-scale variance`
   (the rescaled $C$) and `--direction up` for MCP / SCAD on the direct loss. It applies cleanly
   (`git apply --check`) and brings its own tests.
2. **LLA and the adaptive lasso** as path functions next to `lasso_path`. Tests: both equal the
   lasso when all weights are 1; every solution satisfies the optimality conditions of its
   weighted problem; an LLA path that has converged is a stationary point of the MCP / SCAD
   objective; the adaptive lasso never selects an entry that its first step excluded.
3. **Selection inside the shard runner** (items 2 – 4 of §3.3), switched on by an option, so that
   the old output is reproduced bit by bit without it. Tests on a small grid.
4. **The submit script** for the campaign: one command per wave, with `--dry-run`, `--status` and
   `--fill` as in `submit_nsweep.sh`. Tested against the stub `sbatch`.
5. **The wave 2 runner:** the `random` command of `search_study.py`, cut into shards.
6. **The analysis** (`nsweep.py` extended by the two new factors, figures) and the write-up
   `simulations/S4_campaign.md`. This can be written while the cluster runs.
7. **The command sheet** in `docs/REPRODUCTION.md`, as for the n-sweep.

Items 1 – 4 and 7 are needed for wave 1: about half a day. Item 5 for wave 2: about two hours.

## 5. What you will do on the cluster

The exact commands come with the code, as on 2 October. In outline:

```bash
# laptop: review, commit and push the new code.  The cluster also needs files that are
# untracked today (git status): src/gclm/solvers/search.py, tests/test_search.py,
# simulations/diagnostics/search_study.py
# LRZ:
ssh ge47xod3@cool.hpc.lrz.de
cd ~/repo && git pull
bash cluster/submit_campaign.sh --wave 1 --dry-run        # prints what would be submitted
bash cluster/submit_campaign.sh --wave 1 --n 1000 1e4     # 128 tasks
bash cluster/submit_campaign.sh --wave 1 --n inf          # once the first round is running
bash cluster/submit_campaign.sh --status
```

Waves 2 and 3 do not depend on the results of wave 1 and can be submitted as soon as the queue
has room. Wave 4 waits for the analysis of wave 1.

## 6. Order of work until the 10 – 11 October meeting

| when | what |
|---|---|
| after your go | code for wave 1 (§4, items 1 – 4, 7); a local run of one small cell end to end. Meanwhile the laptop runs the 3 October replication on the three other settings of the true $C$ (20 graphs each, about an hour), as a first look before the cluster results |
| then | you submit wave 1; I write the wave 2 runner and the analysis |
| wave 1 back | analysis and figures for the 2 × 2; decide the estimators for wave 4 |
| in parallel | waves 2 and 3 on the cluster |
| last | wave 4; the two-page summary for the meeting |

## 7. What I need from you

1. **Go ahead with waves 1 – 3 as described?** Wave 4 after wave 1. If time gets short before
   the meeting, wave 4 matters more for the thesis than wave 3.
2. **May the patch go into `src/`?** Defaults stay as they are; the three baseline cells of wave 1
   check that nothing changed.
3. **Wave 3 without SCAD and at $p = 10$ first?** The log-likelihood loss is the expensive one.
   Wave 1 covers both penalties on the direct loss, and in every result so far SCAD sits between
   the lasso and MCP. SCAD and $p = 20$ can be added once wave 3 shows a pattern worth completing.
4. **Three sample sizes** ($10^3$, $10^4$, $\infty$) in all waves? In the baseline $10^5$ and
   $\infty$ were nearly the same.
5. **The estimated $C$:** after the meeting, unless wave 1 shows that $C$ is the bottleneck?

## 8. Scripts of this note

All in [`files/`](files/); run them from the repository root. None of them changes `src/`.

| script | what it does | output | time |
|---|---|---|---|
| `fit_check_free_diagonal.py` | fit of the true support under three assumptions about $C$ (§2.2) | printed | seconds |
| `variance_ordering_baseline.py` | the rule that orients the lasso's pairs by variance (§2.2) | `variance_ordering_baseline.csv` | 3 min |
| `estimate_c_check.py`, `estimate_c_check.R` | Varando & Hansen's package with four treatments of $C$ (§2.4); needs R with `gclm`; `--markdown` prints the tables of §2.4 | `estimate_c_check.csv` | 30 min |
| `loglik_order_check.py` | the repository's log-likelihood lasso in both path orders (§2.5) | `loglik_order_check.csv` | 25 min |
| `time_direct_up.py`, `time_loglik_up.py` | timing pilots for waves 1 and 3 (§3.4) | `time_direct_up.txt`, `time_loglik_up.txt` | 15 min, 1 h |
