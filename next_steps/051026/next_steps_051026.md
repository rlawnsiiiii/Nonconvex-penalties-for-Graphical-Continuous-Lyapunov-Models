# 5 October 2026: where to go next

*Working notes. They update §8 of [`../031026/next_steps_031026.md`](../031026/next_steps_031026.md)
with what has been learnt since: the cluster baseline
([`../../simulations/S2b_nsweep.md`](../../simulations/S2b_nsweep.md)) and one new diagnostic (§2).
The verdicts so far are in [`../../simulations/VERDICTS.md`](../../simulations/VERDICTS.md).*

**Status: a proposal. Nothing below is started until Joon confirms.**

**Update, later on 5 October.** Directions A, D and F and the search part of E are now planned as
one cluster campaign in [`cluster_campaign_051026.md`](cluster_campaign_051026.md). That note also
answers the question whether rescaling $C$ makes sense and what Varando & Hansen do instead (they
estimate $C$), and it supersedes §5 below. This note remains the overview of the directions.

---

## 0. Summary

1. **The core plan of 3 October stands:** put the two options (rescaled $C$, dense → sparse paths)
   into the repository, test them beyond `C_ID`, and scale them up on the cluster.
2. **One simplification:** the test "does it hold for the other three $C$ settings?" does not need a
   separate laptop run. The cluster cells already contain all four settings with 25 reps, so the
   first cluster wave of the new arms *is* that test, at $p = 10$ and $20$, in a few hours (A).
3. **One new finding changes what is worth doing after that** (§2). What limits the methods depends
   on the sample size:
   - at $n = 1000$ the **score** is the limit: the best-scoring graph is not the true one, so
     searching harder makes things worse;
   - at large $n$ in dense graphs the **search** is the limit: the true graph scores best but is
     not reached.
4. **Directions, in order** (§3):
   - A. the new arms on the cluster, all four $C$ settings;
   - B. a data-generating process in which direction is identifiable, plus a variance-ordering
     baseline;
   - C. a small theory part;
   - D. the log-likelihood loss with the new options;
   - E. a finishing step chosen by regime;
   - F. larger $p$ for the thesis figure;
   - G. estimating $C$, only if A shows that $C$ is the bottleneck;
   - H. one real-data example.
5. **Drop** (§4): the Frobenius loss in new runs, more search effort at $n = 1000$, the γ sweep,
   Mnet, GMC, forward selection.

---

## 1. What changed since 3 October

- **The cluster baseline is in** (S2b). On Dettling's pipeline with the usual path, MCP and SCAD
  lose to the lasso in all 24 cells of loss × $p$ × $n$, and the gap grows with $n$. This is the
  reference for every new arm.
- **Measured cost.** A direct-loss cell (800 graphs, $p \le 20$) costs 1.4 / 6 / 8 CPU-h for lasso /
  MCP / SCAD. The Frobenius loss took 505 of the 720 CPU-h and is the worst loss on every metric.
- **Covariance-loss fits depend on the machine** for single datasets (S2b §2). This is an argument
  for the direct loss and for convex steps wherever they do the same job.
- **Figures.** The write-ups now embed their figures: S3b (9), S2b (12), the independent study
  (its own 3, whose links were broken, plus 4 from the repository's runs), the 3 October note (3).

## 2. What limits the methods now

`files/score_vs_optimisation.py`, from the raw-scale S3b runs at $p = 10$.

**The comparison.** For each graph, take the score of the graph reached by the search *started
from the truth*, and the best score reached by any data-driven search.

- If the truth-started search scores better, the data-driven searches are stuck, and a better
  optimiser would help.
- If a data-driven search scores better, the score itself prefers another graph, and a better
  optimiser would not help.

**Result** (number of graphs: truth-start better / equal / data-driven better; directed $F_1$ of
the two graphs in brackets):

| true $C$ | density | $n = 10^3$ | $n = 10^4$ | $n = \infty$ |
|---|---|---|---|---|
| `C_ID` (model correct) | $k = 1, 2$ | 2 / 7 / **11** (0.76, 0.66) | 5 / 11 / 4 (0.89, 0.82) | 1 / **19** / 0 (0.98, 0.92) |
| `C_ID` (model correct) | $k = 3, 4$ | 0 / 1 / **19** (0.71, 0.52) | 8 / 2 / 10 (0.84, 0.66) | **16** / 3 / 1 (0.98, 0.68) |
| random $C$ (model wrong) | $k = 1, 2$ | 7 / 17 / **36** (0.70, 0.56) | 15 / 17 / 28 (0.77, 0.63) | 10 / 28 / 22 (0.70, 0.61) |
| random $C$ (model wrong) | $k = 3, 4$ | 2 / 3 / **55** (0.69, 0.49) | 10 / 0 / **50** (0.80, 0.51) | 17 / 4 / **39** (0.86, 0.52) |

20 graphs per cell for `C_ID`, 60 for random $C$.

**Reading:**

- **$n = 1000$: the score is the limit, even with the correct model.** In 30 of 40 `C_ID` graphs a
  data-driven search finds a graph with a *better* score than the one next to the truth, and that
  graph is much worse ($F_1$ 0.52–0.66 against 0.71–0.76). The data cannot tell many graphs apart,
  and the score picks a sparser or re-oriented one. A more thorough search would find more of these.
- **$n = \infty$, sparse graphs: solved.** The data-driven searches reach the same score as the
  truth-started one in 19 of 20 graphs. The remaining $F_1$ difference (0.92 against 0.98) comes
  from graphs with the same score, i.e. graphs the data cannot distinguish at all.
- **$n = \infty$, dense graphs: the search is the limit.** The truth-started search scores better in
  16 of 20 graphs, by a wide margin, and reaches $F_1$ 0.98 against 0.68. The data-driven searches
  stop at graphs with too many edges: 7–10 false pairs and 4 reversed edges per graph, and 4 of
  the 6 true 2-cycles found in one direction only. A likely reason: these graphs fit the covariance
  exactly, removing any single edge destroys the exact fit, and so no single move improves the
  score.
- **$n = 10^4$: in between.** Sparse graphs are mostly solved; dense ones are split.
- **With the wrong $C$ the score prefers another graph at every $n$,** including $n = \infty$: in
  most dense graphs (39 to 55 of 60) and in a third to a half of the sparse ones. That is the S3b
  result on Dettling's pipeline seen once more, and the reason $C$ comes first below.

## 3. Directions

### A. The new arms on the cluster, all four $C$ settings *(first; decides the scope of the claim)*

- **Why.** Every positive result so far is for `C_ID`. §2 and S3b §9.3 show that a wrong $C$ is
  exactly what breaks likelihood-based decisions. The fit check of 3 October predicts a graded
  outcome: the rescaled $C$ removes almost all misfit for `C_Random_Min_Diag`, half of it for
  `C_Random_Diag`, and little for `C_Random_Full`.
- **Experiment.** After the patch: direct loss, $p = 10, 20$, $n = 10^3 \dots \infty$, the same 800
  graphs per cell as the baseline, rescaled $C$, with these arms:
  - the lasso;
  - MCP and SCAD in both path directions;
  - the adaptive lasso weighted by the minimum-$\ell_1$ exact fit. It belongs to the same
    dense → sparse family, but it is convex. In the independent study it matches MCP dense → sparse
    and has the best `aupr` by a wide margin [IS §5]. Being convex, it is free of the machine
    dependence of S2b §2, and it has known theory.
- **Expected.** The gain holds for `C_ID` and `C_Random_Min_Diag`, shrinks for `C_Random_Diag` and
  vanishes for `C_Random_Full`. At $n = 1000$ it is small everywhere.
- **Also settles** whether the gain at $n = 1000$ is real: 800 graphs instead of 40.
- **Cost.** About 35 CPU-h per sample size, 140 in total; a few hours of wall-clock with 96 cores.
  Uses `--shards 4`, as measured.

### B. A setting in which direction is identifiable, and a variance-ordering baseline *(cheap)*

- **Why.** With $N(0,1)$ edge weights and 2-cycles, an oracle orients an edge correctly with
  probability 0.77 at $n = 1000$ [IS §3], and §2 shows the score cannot pick the truth there. A
  second data-generating process (edge weights bounded away from zero, no 2-cycles) separates
  "the estimator is weak" from "the information is not there". The independent study finds larger
  gains and 40–52 % exact recovery there [IS §7].
- **Baseline.** On the raw scale, 65 % of the true edges point from the higher-variance node to the
  lower one [IS §2]. A trivial rule ("orient every found pair from high to low variance") shows how
  much of any raw-scale result is that artefact. It justifies standardising, and it is a fair
  criticism to raise at the meeting.
- **Cost.** Half a day: two flags in `sample_drift` that keep the default bit-identical, plus the
  runs of A on the new graphs.

### C. A small theory part *(no compute; gives the thesis a spine)*

- **The standardisation lemma:** standardising maps $C$ to $D^{-1}CD^{-1}$. Trivial to prove, and it
  explains verdict 6.
- **Presence is first order, direction second order** [IS §3]: to first order the covariance
  depends on a pair only through $d_iB_{ij} + d_jB_{ji}$. Consequences to state and prove:
  - on the correlation scale both directions of a pair enter the lasso path at the same $\lambda$;
  - which of them a sparse → dense MCP path keeps is decided by second-order terms and noise. This
    is the 51–61 % of S3a.
- **A worked three-node example** (chain, fork, collider): which orientations the covariance
  distinguishes, and what the two path directions return.

### D. The log-likelihood loss with the new options

- **Why.** With the lasso the log-likelihood loss is the best of the three (`aupr` +0.04 at
  $p = 10$, +0.08 at $p = 20$; S2b §3.4). Varando & Hansen already report that paths started from
  the dense fit work better. The combination "log-likelihood, rescaled $C$, dense start, MCP" has
  never been run and could be the best estimator overall.
- **Caution.** These fits are machine-dependent for single datasets, so only means over many graphs
  count. The dense-start option of `covloss_path` is untested.
- **Cost.** 7–24 CPU-h per cell on the cluster; a day of code and tests first.

### E. The finishing step, chosen by regime

- **Small $n$: do not search harder; aggregate.** At $n = 1000$ the single best-scoring graph is
  unreliable (§2). Instead of one graph, report how often each edge appears among the good graphs
  (restarts, bootstrap samples, or all graphs within a few score units of the best), and threshold
  that. This is the lasso's hedge done on purpose. The "report both directions when unsure" layer
  of the independent study (+0.02 to +0.03, [IS §12]) is the simplest version. Exploratory: there
  is no evidence yet that it beats the lasso.
- **Large $n$, dense graphs: better moves.** §2 shows a large optimisation gap there. Candidates:
  - exchange moves (the best method of the independent study at large $n$ uses them);
  - moves on a whole pair (try $i \to j$, $j \to i$ and both);
  - raising the weight of the fit gradually instead of starting at its final value.
- **Cost.** A day each. Both reuse `gclm.solvers.search`.

### F. Larger $p$ for the thesis figure

- **Why.** Dettling's Figure 5 runs to $p = 50$, and the thesis will want the same axis. At
  $p = 40, 50$ with $n = 1000$ there are more parameters than observations, and the dense start
  rests on a noisy $\hat\Sigma^{-1}$. Whether dense → sparse still helps there is open.
- **Cost.** About 350 CPU-h per sample size for the full $p$ range with 25 reps. After A.

### G. Only if A shows that $C$ is the bottleneck: treat $C$ as unknown

- **Sensitivity first:** fit with $C = 2I$ and with the rescaled $C$, and report the edges on which
  the two agree.
- **Estimating the diagonal of $C$** is a research question of its own. Freeing it removes $p$ of
  the $p(p+1)/2$ equations, and identifiability is unclear. Not before the meeting.

### H. One real-data example, near the end

- Dettling's own application (the Sachs protein data, §6 of the paper): the lasso selected by the
  score with the eBIC penalty, then the rescaled $C$, then MCP dense → sparse, compared with the
  consensus network. Small ($p = 11$) and directly comparable to the paper.

## 4. What I would drop

- **The Frobenius loss in new runs.** It is the worst loss for every penalty, costs 70 % of the
  compute and has the most convergence problems (S2b §4).
- **More search effort at $n = 1000$** (more restarts, larger neighbourhoods). §2 says it finds
  better-scoring but worse graphs.
- **Starting anything from the standard MCP / SCAD paths.** They are the worst starts in both
  studies.
- **The γ sweep, Mnet, GMC, forward selection.** Tried or superseded (VERDICTS, "Tried, with little
  or no gain").

## 5. Order, and what I need from Joon

**Before the 10–11 October meeting:**

1. The patch into the repository, with tests (about 1 h).
2. A on the cluster (submit, a few hours, then the analysis with `nsweep.py`).
3. B: the second data-generating process and the variance-ordering baseline.
4. C: the lemma and the first-order statement written down; the worked example if time allows.
5. A two-page summary for the meeting from `VERDICTS.md` and the figures.

**After the meeting:** D, E, F, then H; G only if needed.

**Decisions:**

1. May the patch go into `src/`? Defaults stay unchanged.
2. Should the rescaled $C$ become the default for new experiments, with $C = 2I$ kept as the
   reproduction of Dettling?
3. Is it fine to go straight to the cluster for A (my recommendation) instead of a laptop run first?
4. Drop the Frobenius loss from all new arms?
5. Should the adaptive lasso be an arm of A? It is convex, so strictly it is not a "nonconvex
   penalty", but it is the one-step version of one.
