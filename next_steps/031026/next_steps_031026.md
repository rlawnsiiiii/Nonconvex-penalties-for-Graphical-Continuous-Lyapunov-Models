# 3 October 2026: the overnight search study and the independent study, combined

*Working notes. Three sources:*

- *my overnight run of the reversal-search study ([`simulations/S3b_reversal_search.md`](../../simulations/S3b_reversal_search.md), `runs/s3b_search/`);*
- *the independent study ([`../021026/independent_study/independent_study_021026.md`](../021026/independent_study/independent_study_021026.md)), computed in a cloud sandbox with its own C solvers;*
- *a replication of that study's two central claims with the repository's own solvers ([`files/replicate_dense_to_sparse.py`](files/replicate_dense_to_sparse.py)).*

*Numbers marked [IS] are taken from the independent study and not recomputed here. Everything else
was computed with the repository code.*

**Status: the next steps of §8 are a proposal. Nothing in it is started until Joon confirms.**

*A one-page list of the verdicts so far is in [`../../simulations/VERDICTS.md`](../../simulations/VERDICTS.md).*

---

## 0. Summary

**Verdict: the independent study is promising, and it changes the plan.**

1. **The two studies agree wherever they overlap.** On the same 40 graphs, the two independent
   implementations of the BIC search give the same directed $F_1$ to three decimals in 14 of 18
   shared numbers and within 0.01 in the others (§2).
2. **It adds two levers that I did not test, and both hold up in my replication** (§4):
   - **The right $C$ after standardising.** Standardised data follow the model with
     $C = 2\,\mathrm{diag}(1/s_{ii}^2)$, not $C = 2I$. With that $C$ the true graph fits exactly for
     `C_ID`, and the misfit also shrinks in the three random-$C$ settings (§3.1).
   - **The direction of the path.** MCP and SCAD run *dense → sparse* (from the lasso solution at
     the smallest $\lambda$ up to $\lambda_{\max}$) beat the lasso. Run *sparse → dense*, as in
     every run of the thesis so far, they lose at every $n$.
   - **Both are needed together.** On the 40 `C_ID` graphs, with the repository's solvers and the
     rescaled $C$, MCP dense → sparse reaches `max_f1` 0.662 / 0.747 / 0.798 at
     $n = 10^3 / 10^4 / \infty$ against 0.637 / 0.673 / 0.696 for the lasso. With $C = 2I$ the same
     path only ties the lasso up to $n = 10^4$.
3. **One principle explains both studies: decide the direction of an edge late** (§5). Every
   estimator that commits to a direction when the edge first enters loses. Every estimator that
   starts with both directions present and prunes afterwards wins, given a correctly specified $C$.
4. **For the thesis question** this is a positive answer with conditions: nonconvex penalties help
   as a pruning step on a dense $\ell_1$ solution, on a correctly specified scale, mostly for sparse
   graphs and $n \ge 10^4$. They do not help as a replacement for the lasso on the usual path.
5. **Caveats** (§6): almost everything in the independent study is `C_ID` at $p = 10$; the gains at
   $n = 1000$ are small; the headline numbers use oracle tuning.
6. **Proposed next steps** (§8), in order:
   1. put the two options into the repository;
   2. test them on all four $C$ settings on the laptop;
   3. add them to the cluster sweep;
   4. rerun the search study on the corrected scale;
   5. add a second data-generating process in which direction is identifiable.

---

## 1. The two studies side by side

| | my overnight run (S3b) | independent study |
|---|---|---|
| code | repository solvers; new `gclm.solvers.search` | own C solvers, validated against the repository on 20 datasets |
| question | does a greedy BIC search with add / delete / reverse moves improve on the path estimators, and from which start? | why is the pilot negative, and which estimators do beat the lasso? |
| graphs | Figure 5 generator, all four $C$ settings, $p = 10$ (160 per $n$) and $p = 20$; Example 2 | Figure 5 generator, **`C_ID` only**, $p = 10$ (52–100) and $p = 20$ (24); Example 2; a strong-signal DGP without 2-cycles |
| scales | Dettling's pipeline (standardised, $C = 2I$) and the raw covariance | Dettling's pipeline ("identity") and standardised with $C = 2\,\mathrm{diag}(1/s_{ii}^2)$ ("variance") |
| estimators | lasso / MCP / SCAD paths (sparse → dense); BIC search from each; pure search with random restarts | 13 estimators, among them MCP dense → sparse, LLA, adaptive lasso, backward elimination with exchange moves; a trial of the S3b search |
| tuning | BIC-selected $\lambda$ (and the oracle best-F1 $\lambda$ as reference) | oracle `max_f1` along the path, and BIC-selected graphs |

The raw scale and the "variance" scale describe the same model: "variance" is the raw-scale problem
written in the standardised variable. They differ in the lasso penalty, which is unweighted on the
raw scale and weighted by $s_j/s_i$ on the "variance" scale.

## 2. Where the two studies agree

**The same cells, computed twice.** Directed $F_1$ on the 40 `C_ID` graphs ($p = 10$, $k = 1..4$,
reps 0–9): the path's support at its BIC $\lambda$ → after the BIC search.

| | mine | independent study [IS] |
|---|---|---|
| Dettling's pipeline, lasso start, $n = 10^3$ | 0.549 → 0.548 | 0.549 → 0.548 |
| … $n = 10^4$ | 0.624 → 0.632 | 0.624 → 0.632 |
| … $n = \infty$ | 0.601 → 0.610 | 0.597 → 0.614 |
| Dettling's pipeline, MCP (sparse → dense) start, $n = 10^3$ | 0.488 → 0.517 | 0.488 → 0.517 |
| … $n = 10^4$ | 0.477 → 0.499 | 0.477 → 0.490 |
| … $n = \infty$ | 0.454 → 0.468 | 0.448 → 0.468 |
| Dettling's pipeline, SCAD start, $n = 10^3$ / $10^4$ | 0.503 → 0.506 / 0.518 → 0.524 | 0.503 → 0.506 / 0.518 → 0.524 |
| raw scale, lasso start, $n = 10^4$ | 0.566 → 0.715 | 0.566 → 0.715 |

Two implementations, written independently and in different languages, give the same numbers. The
small differences at $n = \infty$ are within what tie-breaking between equal-score moves explains.

**The same conclusions:**

- **On Dettling's pipeline the model is misspecified**, and a likelihood-based score suffers from it.
  I saw it as "BIC prefers wrong graphs"; the independent study measures it directly: a
  likelihood-ratio test rejects the *true* graph in 34 % of datasets at $n = 10^3$ and 75 % at
  $10^4$ [IS].
- **The pilot's MCP and SCAD paths are the worst starting points for a search.** The search does
  not undo their reversed edges.
- **The search is a finishing step.** Its gain depends on the start: nothing on a good start, a lot
  on a dense one.
- **Greedy from the empty graph is poor.** It commits to the first direction it sees.
- **On Example 2 the score, not the search, is the limit at $n \le 10^4$.** The exhaustive BIC
  optimum is not the true graph in any of 5 datasets there [IS]; my exhaustive check found the same
  on two datasets. At $n = 10^5$ and $\infty$ reversal moves recover the 5-cycle.
- **The lasso's habit of keeping both directions is the right response to ambiguity** under
  directed $F_1$ (break-even accuracy about 2/3 to 0.7).

## 3. What the independent study adds

### 3.1 After standardising, $C = 2I$ is the wrong model

With $D = \mathrm{diag}(s)$ the standard deviations, standardising maps
$M\Sigma + \Sigma M^\top + C = 0$ to $\tilde M R + R\tilde M^\top + D^{-1}CD^{-1} = 0$ with
$\tilde M = D^{-1}MD$. The support is unchanged, but the volatility becomes $D^{-1}CD^{-1}$. For
`C_ID` that is $2\,\mathrm{diag}(1/s_{ii}^2)$, not $2I$.

My check, in all four of Dettling's settings. It is the least-squares loss of the best fit on the
*true* support at $n = \infty$, relative to the loss of the diagonal fit; 0 means the truth fits
exactly (40 graphs per setting, $p = 10$):

| true $C$ | standardised, $C = 2I$ (every run so far) | standardised, $C = 2\,\mathrm{diag}(1/s_{ii}^2)$ | raw, $C = 2I$ |
|---|---|---|---|
| `C_ID` | 0.0137 | **0.0000** | 0.0000 |
| `C_Random_Diag` | 0.0152 | 0.0077 | 0.0096 |
| `C_Random_Min_Diag` | 0.0180 | **0.0009** | 0.0013 |
| `C_Random_Full` | 0.2740 | 0.1939 | 0.2465 |

- **The rescaled $C$ is better in every setting,** and exact for `C_ID`. (The independent study
  reports 1.5 % for `C_ID` under $C = 2I$; I get 1.4 %.)
- **`C_Random_Full` stays badly misspecified** under any diagonal $C$. Expect likelihood-based
  methods to struggle there whatever is done.
- **For the lasso this hardly matters** (`max_f1` 0.620 vs 0.621 at $n = 10^3$ [IS]), which is why
  the Figure 5 reproduction was unaffected. It matters for everything that commits to a sparse fit.
- **Why not simply the raw scale?** On the raw scale the lasso orients edges by the variance
  ordering ("the high-variance node is the parent"), which this DGP satisfies for 65 % of edges
  [IS]. That is information a benchmark should not hand out for free. Standardising removes it;
  the rescaled $C$ keeps the model right.

### 3.2 The direction of the path

The same penalty, $\gamma$ and objective give different stationary points depending on where the
path starts [IS, `C_ID`, $p = 10$, 52 graphs, rescaled $C$, `max_f1` with paired $z$ against the
lasso]:

| | $n = 10^3$ | $10^4$ | $10^5$ | $\infty$ |
|---|---|---|---|---|
| lasso | 0.633 | 0.669 | 0.680 | 0.685 |
| MCP, sparse → dense (the pilot) | 0.570 (−4.0) | 0.606 (−4.1) | 0.603 (−5.1) | 0.619 (−4.0) |
| MCP, dense → sparse | 0.648 (+0.9) | 0.742 (+4.9) | 0.773 (+6.7) | 0.777 (+6.1) |
| SCAD, dense → sparse | 0.647 (+1.0) | 0.746 (+5.7) | 0.773 (+6.7) | 0.778 (+6.1) |

![independent study: max F1 against n](../021026/independent_study/independent_study/figures/f1_vs_n.png)

*The independent study's own figure [IS]: 100 `C_ID` graphs, $p = 10$. Left: Dettling's pipeline.
Right: the rescaled $C$.*

- **The mechanism** [IS §3]: to first order in the coupling strength, the covariance sees only that
  a pair is connected, not in which direction. Direction is a second-order effect that shows in the
  covariance of *non-adjacent* nodes. A path that starts sparse must choose a direction when only
  first-order information is in the fit. A path that starts dense begins with both directions
  present and only has to drop one.
- **This matches S3a:** the first direction to enter is the true one only 51–61 % of the time, for
  every penalty.
- **$\gamma$ is a weak lever** [IS §6]: increasing it only moves the pilot's path towards the lasso.

### 3.3 How much information the data hold

For an oracle that knows the rest of the graph and uses the likelihood [IS §3]:

| thesis DGP ($N(0,1)$ weights) | $n = 10^3$ | $10^4$ | $10^5$ |
|---|---|---|---|
| true edges with a clear signal for *presence* | 41 % | 70 % | 87 % |
| … for *direction* | 27 % | 59 % | 81 % |
| oracle orientation accuracy | 0.77 | 0.88 | 0.95 |

![information budget](../021026/independent_study/independent_study/figures/information_budget.png)

*[IS] Share of true edges with a clear signal (expected likelihood-ratio statistic above 10).*

At $n = 1000$ the thesis DGP is a low-information problem. This is consistent with my runs: even
the search started from the true graph reaches only $F_1$ 0.73 / 0.87 / 0.98 at
$n = 10^3 / 10^4 / \infty$ (`C_ID`, raw scale).

### 3.4 Smaller results

- **Example 2 depends on the scale** [IS §8]. On the raw scale the minimum-$\ell_1$ exact fit is the
  wrong DAG (as I found). On the standardised scale with the rescaled $C$ it *is* the 5-cycle, but
  the lasso then loses the path graph it recovers on the raw scale. The example is var-sortable.
- **Reporting both directions when unsure** adds 0.02–0.03 $F_1$ for estimators that commit
  [IS §12].
- **GMC** (a nonconvex penalty with a convex objective) is only marginally better than the lasso
  [IS §10]. Forward stepwise selection and GLS-type weighting did not help.
- **At $p = 20$** (24 graphs) the BIC-selected graphs of dense → sparse MCP have a structural
  Hamming distance of 23.5 against 54.3 for the lasso at $n = 10^4$ [IS §11].

## 4. Replication with the repository's solvers

`files/replicate_dense_to_sparse.py`: the 40 `C_ID` graphs ($p = 10$, $k = 1..4$, reps 0–9, the
repository's seeds) at three sample sizes, fitted with `lasso_path` and `solve_fista`. Nothing in
`src/` was changed; the dense → sparse path is implemented in the script exactly as in the patch.
Output: `files/replicate_dense_to_sparse.csv` (`--summarize` prints the tables). Paired $z$ against the lasso on the same
scale in brackets.

![replication](files/replication_dense_to_sparse.png)

*Replication with the repository's solvers (40 `C_ID` graphs, $p = 10$; means ± 1 standard error).
Hollow orange: the path of every thesis run so far. Filled: dense → sparse.
`files/plot_replication.py` draws it.*

**Oracle path maximum (`max_f1`):**

| | $n = 10^3$ | $n = 10^4$ | $n = \infty$ |
|---|---|---|---|
| **rescaled $C$** | | | |
| lasso | 0.637 | 0.673 | 0.696 |
| MCP, sparse → dense (pilot) | 0.587 (−2.7) | 0.626 (−2.6) | 0.634 (−3.3) |
| MCP, dense → sparse | 0.662 (+1.2) | **0.747 (+4.2)** | **0.798 (+5.6)** |
| SCAD, dense → sparse | 0.656 (+1.1) | **0.755 (+4.9)** | **0.794 (+5.3)** |
| **$C = 2I$ (Dettling's pipeline)** | | | |
| lasso | 0.622 | 0.664 | 0.679 |
| MCP, sparse → dense (pilot) | 0.542 (−4.8) | 0.556 (−5.4) | 0.577 (−5.6) |
| MCP, dense → sparse | 0.625 (+0.2) | 0.675 (+0.6) | 0.725 (+3.5) |
| SCAD, dense → sparse | 0.626 (+0.3) | 0.692 (+2.8) | 0.723 (+3.8) |

**Data-driven: directed $F_1$ at the BIC-selected $\lambda$, and after the BIC search from there:**

| | $n = 10^3$ | $n = 10^4$ | $n = \infty$ |
|---|---|---|---|
| **rescaled $C$** | | | |
| lasso: BIC $\lambda$ → + search | 0.567 → 0.569 | 0.625 → 0.670 | 0.644 → 0.724 |
| MCP sparse → dense: BIC $\lambda$ → + search | 0.514 → 0.510 | 0.566 → 0.577 | 0.534 → 0.588 |
| MCP dense → sparse: BIC $\lambda$ → + search | 0.571 → 0.574 | **0.691 → 0.702** | **0.739 → 0.760** |
| SCAD dense → sparse: BIC $\lambda$ | 0.571 | 0.695 | 0.737 |
| **$C = 2I$** | | | |
| lasso: BIC $\lambda$ → + search | 0.549 → 0.548 | 0.624 → 0.632 | 0.601 → 0.610 |
| MCP dense → sparse: BIC $\lambda$ → + search | 0.555 → 0.552 | 0.621 → 0.627 | 0.619 → 0.615 |

**Reading:**

- **The independent study's numbers reproduce.** Its values on 52 graphs: lasso 0.633 / 0.669 /
  0.685, MCP dense → sparse 0.648 / 0.742 / 0.777 (`max_f1`); BIC-selected 0.568 / 0.688 / 0.731,
  after the search 0.575 / 0.701 / 0.755. Mine on 40 graphs are within 0.02 of all of them.
- **Both levers are needed.** With $C = 2I$ the dense → sparse path only ties the lasso up to
  $n = 10^4$, and its BIC-selected graphs gain nothing. With the rescaled $C$ it gains +0.07 at
  $n = 10^4$ and +0.10 at $n = \infty$, in the oracle and in the BIC-selected numbers alike.
- **The pilot's direction loses on both scales.** The rescaled $C$ halves its deficit but does not
  remove it.
- **At $n = 1000$ the gain is small:** +0.025 `max_f1` ($z = 1.2$ on 40 graphs), nothing in the
  BIC-selected $F_1$. The independent study finds $z \approx 2$ with 52–100 graphs.
- **Dense → sparse MCP is the best start for the search,** and needs it least: 0.691 → 0.702 at
  $n = 10^4$. The lasso start gains more from the search (0.625 → 0.670) but ends lower.
- **Reversed edges tell the same story.** At the BIC-selected $\lambda$ ($n = 10^4$, rescaled $C$)
  the lasso has 1.8 reversed edges per graph, the pilot's MCP path 3.9 and MCP dense → sparse 2.2.

## 5. How the pieces fit: decide the direction late

| estimator | when it settles the direction of an edge | result against the lasso |
|---|---|---|
| lasso | never: keeps both directions when unsure | baseline |
| MCP / SCAD, sparse → dense (every thesis run so far) | when the edge enters, on first-order information | loses at every $n$, on every loss |
| forward stepwise; greedy search from the empty graph | when the edge enters | loses, or needs many random restarts |
| MCP / SCAD dense → sparse; LLA or adaptive lasso from a dense $\ell_1$ fit; backward elimination | when pruning an exact fit that holds both directions | wins on a correctly specified scale, more with larger $n$ |
| BIC search started from the lasso | after the path, by the likelihood | wins on a correctly specified scale |
| BIC search started from the pilot's MCP path | too late: the reversals are frozen | no gain |

Two conditions sit on top of this:

- **The score must be right.** A likelihood-based decision (BIC, or pruning towards an exact fit)
  needs the correct $C$. On Dettling's pipeline it is not.
- **The information must be there.** At $n = 1000$, with $N(0,1)$ weights, it mostly is not. The
  gains concentrate on sparse graphs ($k = 1, 2$) and $n \ge 10^4$.

The routes end close to each other. Directed $F_1$ of BIC-selected graphs on the 40 `C_ID` graphs:

| | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|
| plain lasso at its BIC $\lambda$, rescaled $C$ [IS] | 0.567 | 0.625 | 0.636 |
| lasso + BIC search, raw scale (mine) | 0.573 | 0.715 | 0.721 |
| pure greedy search, 10 restarts, raw scale (mine) | 0.565 | 0.666 | 0.747 |
| lasso + BIC search, rescaled $C$ [IS] | 0.569 | 0.670 | 0.726 |
| MCP dense → sparse (+ search), rescaled $C$ [IS] | 0.568 (0.575) | 0.688 (0.701) | 0.731 (0.755) |
| backward elimination with exchange moves, rescaled $C$ [IS] | 0.549 | 0.713 | 0.751 |
| *search started from the true graph (oracle), raw scale (mine)* | *0.732* | *0.865* | *0.980* |

- **The good methods are within about 0.03 of each other**: about 0.57 at $n = 10^3$, 0.67–0.72 at
  $10^4$, 0.72–0.75 at $\infty$.
- **The gap to the oracle start is large** (0.15–0.25) and is an optimisation gap: at $n = \infty$
  the truth is almost a fixed point of the search (0.98), but no data-driven start reaches it.
- **That gap sits in the dense graphs.** By density at $n = 10^4$, lasso + search reaches 0.76 and
  0.80 for $k = 1, 2$, and 0.67 and 0.64 for $k = 3, 4$.

## 6. Is it promising? Strengths and caveats

**Strengths:**

- **A mechanism, not only numbers.** First-order / second-order identifiability explains S1b, S2,
  S3a, the γ sweep, the forward methods and the search results with one argument.
- **Independent confirmation.** Its S3b trial matches my implementation; its misspecification claim
  matches my check; its path-direction claim replicates with the repository's solvers (§4).
- **A positive result for the thesis topic.** Nonconvex penalties beat the lasso, with a clear
  recipe and a clear boundary of validity.
- **The patch is small.** Two options, defaults unchanged; it applies cleanly to the current
  repository.

**Caveats:**

- **`C_ID` only.** The three misspecified settings were not run with the new estimators. §3.1 says
  the rescaled $C$ still helps the fit there, but `C_Random_Full` remains far off. This is the
  first thing to test.
- **Small gains at $n = 1000$,** the sample size of Figure 5: +0.02 to +0.03 `max_f1`, $z \approx 2$.
  The large gains need $n \ge 10^4$.
- **Sparse graphs only.** For $k = 3, 4$ the lasso is as good or better.
- **Oracle tuning in the headline tables.** With BIC the gains are smaller but have the same sign;
  structural Hamming distance improves more clearly than $F_1$.
- **$p = 20$ rests on 24 graphs,** and the covariance losses were not run with the new options.
- **Its literature list is partly unverified,** as it says itself.

## 7. Corrections to what I said earlier

- **"As an initialiser, nonconvexity adds nothing over the lasso."** True only for the pilot's
  sparse → dense path. A dense → sparse MCP path is a better start than the lasso [IS §9].
- **"Nonconvexity helps as the score of a search, not as a penalty on a path."** Too narrow. It
  also helps as a penalty when the path runs dense → sparse. The common factor is deciding late.
- **"The raw scale is correctly specified."** Only for `C_ID`. And the cleaner fix is the rescaled
  $C$ on standardised data, which does not leak the variance ordering.
- **"$\ell_1$ aims at the wrong graph in Example 2."** On the raw scale. On the standardised scale
  with the rescaled $C$ the minimum-$\ell_1$ fit is the true 5-cycle [IS §8].
- **$p = 20$ on Dettling's pipeline at $n = 10^4$.** The +0.08 for lasso + search came from sparse
  graphs only ($k = 1, 2$ and part of $k = 3$); the run was stopped before the dense ones.
  Corrected in S3b §9.4.

## 8. Proposed next steps

**Awaiting confirmation. Nothing below is started.**

**Step 1. Put the two options into the repository** (about 1 h).
- Apply `library_patch.diff`: `--c-scale variance` and `--direction up` for MCP/SCAD on the direct
  loss. The patch applies cleanly to the current HEAD.
- Run its tests and the full suite; confirm that the defaults reproduce the stored S1 / S1b
  estimates exactly.
- Document both in `docs/NONCONVEX.md` and `docs/REPRODUCTION.md`.

**Step 2. Does it hold beyond `C_ID`?** (laptop, a few hours in the background).
- $p = 10$, all four $C$ settings, $n = 10^3, 10^4, \infty$, 10 reps (480 datasets).
- Lasso, MCP and SCAD, each sparse → dense and dense → sparse, with $C = 2I$ and the rescaled $C$.
- Report `max_f1`, BIC-selected $F_1$, structural Hamming distance, skeleton and orientation.
- **This decides whether the result is a thesis result or a `C_ID` result.**

**Step 3. Add the new arms to the cluster sweep** (roughly 150 CPU-h, 80 tasks).
- Direct loss, $p = 10, 20$, 25 reps, $n = 10^3 \dots \infty$, rescaled $C$: lasso, and MCP / SCAD
  in both directions.
- The cells already submitted ($C = 2I$, sparse → dense) stay: they are the baseline and document
  the negative result on all three losses.
- Needs a small change to `cluster/submit_nsweep.sh` (pass the two options, name the folders).
- The covariance losses with the rescaled $C$ and Varando's dense start come afterwards; that
  combination is untested.

**Step 4. Rerun the search study on the corrected scale** (laptop for $p = 10$, cluster for $p = 20$).
- Starts: lasso, MCP dense → sparse, SCAD dense → sparse, each at its BIC $\lambda$; the pure search
  as reference.
- Add what both studies found missing: restarts or exchange moves, the objective-based search as an
  arm, and the "report both directions when unsure" layer.
- All four $C$ settings; the log-likelihood loss at $p = 10$.

**Step 5. A second DGP in which direction is identifiable** (half a day).
- Edge weights bounded away from zero, no 2-cycles; flags in `sample_drift` that leave the default
  DGP bit-identical.
- Separates "the estimator is weak" from "the information is not there". The independent study
  finds larger gains there [IS §7].

**Step 6. Reporting.**
- Structural Hamming distance and BIC-selected graphs next to the oracle path maximum in
  `aggregate_s1.py` / `compare_runs.py`.
- Drop from the plan: the γ sweep, Mnet, GMC, forward stepwise, LLA from the lasso as a separate
  item (it is one of the dense → sparse family).

**Step 7. Write-up and the 10–11 October meeting.**
- Update the "Reading" of `S2_penalties_losses.md` with the misspecification and path-direction
  results.
- Questions for the meeting:
  - Was $C$ rescaled in Dettling's standardised experiments?
  - Does Example 2 rely on the variance ordering (his Theorem 3)?
  - Is "nonconvex penalties as a late, pruning step" an acceptable framing of the thesis result?

**Decisions needed from Joon:**

1. May the patch go into `src/` (Step 1)?
2. Should the rescaled $C$ become the default for new experiments, with $C = 2I$ kept as the
   reproduction of Dettling?
3. Cluster: submit the new arms as soon as the queue has room, or after the current sweep has
   finished?
4. How far to go today: Steps 1–2 only, or also 4?

## 9. Open points between the two studies

- **BIC search or objective search?** On random graphs the independent study finds the search on
  the MCP objective ahead of the BIC search for MCP starts (0.547 / 0.623 / 0.650 against 0.510 /
  0.577 / 0.597) [IS §9]. On Example 2 I found the opposite. Step 4 keeps both.
- **Which scale for the lasso that starts the search?** The search landscape is nearly the same on
  the raw and the rescaled scale, since the likelihood is invariant. Yet the raw-scale lasso start
  ends at 0.715 and the rescaled one at 0.670 at $n = 10^4$, although the raw lasso itself is the
  worse estimator (0.566 against 0.625). Possibly the variance ordering again.
- **Random restarts.** The pure search with 10 random restarts is the best data-driven method at
  $n = \infty$ on `C_ID` (0.747). The independent study tried only the empty start.

## 10. Status of runs and files

- **Stopped:** the $p = 20$ search run on Dettling's pipeline (259 of 480 graphs; rows kept).
- **Finished:** the rest of Example 2 ($n = 10^3$ and $\infty$) and the raw-scale $p = 20$ run at
  $n = \infty$. Both are written up in S3b §9.1 and §9.4.
- **Cluster n-sweep (the baseline cells): complete and analysed on 4 October,**
  `simulations/S2b_nsweep.md`. MCP and SCAD lose to the lasso in all 24 cells, and the gap grows
  with $n$, as §0 predicts for Dettling's pipeline with sparse → dense paths. Measured cost: 720
  CPU-h in total; a direct-loss cell costs 1.4 / 6 / 8 CPU-h (lasso / MCP / SCAD), so the new arms
  of Step 3 come to roughly 130 CPU-h.
- **Files of this note:** `files/replicate_dense_to_sparse.py` and its CSVs.
- **Not committed:** everything of today.
