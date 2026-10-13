# Code review of the repository, 10–12 October 2026

*Six reviews, each of one part of the repository, run against the code at commit f226361 and
against the stored campaign shards in `runs/campaign/`: the core numerics (`lyapunov`,
`objective/`, `data/`), the solvers and paths (`solvers/proxgrad.py`, `covariance.py`,
`path.py`), the score and the greedy search (`solvers/search.py`, `metrics.py`), the cluster
runners (`simulations/run_s1_shard.py`, `run_search_shard.py`, `rescore_shard.py`), the cluster
scripts (`cluster/submit_campaign.sh`, `feed_queue.sh`, the plans) and the analysis
(`simulations/diagnostics/campaign.py`, `restarts.py`, `plot_campaign.py`). Every finding below
was reproduced with a script; the scripts lived in the session's scratch folder and are not part
of the repository. Severity: **BUG** wrong output; **RISK** wrong output in a reachable case, or
silent where it should fail; **NIT** wrong wording, dead or misleading code; **GAP** important
behaviour without a test.*

## 0. Bottom line

- **No finding changes a conclusion of the campaign.** Every effect on stored numbers that could
  be measured is at most about a third of a standard error of a cell mean, or leaves the
  selected graphs unchanged.
- **Three analysis bugs would have corrupted the figures of 12 October**; they were fixed before
  the figures were drawn (§1).
- **Two numerical issues touch every log-likelihood result** and are open: the covariance-loss
  solver can stop without having converged (A1), and the likelihood refit behind the score has
  no maximiser on some graphs with 2-cycles (A2). Both were measured to change no support and no
  search decision in the cases checked, but both are silent. A1 is the one to fix before the
  log-likelihood numbers go into the thesis; fixing it means re-running the log-likelihood cells.
- **The cluster scripts have several ways to loop or double-submit** (C1, C2). None has
  happened in a way that changed results.

## 1. Fixed (12 October)

| finding | where | what was wrong | fix |
|---|---|---|---|
| the 100-start likelihood-refit searches compared with the wrong lasso | `campaign.py:302` | `search-pure-100sml` and `search-truth-100sml` do not end in `-ml`, so they were paired with the least-squares lasso, against the docstring | the reference is `lasso-ml` for every name ending in `-ml` or `sml` |
| restart curves of different cells merged into one | `plot_campaign.py`, `fig_restarts` | cells were keyed by (draw, p, C, n), so the 100- and 300-start cells became one zig-zag series, and a 32-graph cell could replace a 400-graph one | one curve per cell; reference lines computed on the cell's own graphs |
| the eBIC rule blanked the eBIC-scored searches | `plot_campaign.py`, `apply_rule("ebic1")` | rows renamed from `-e1` had their real `search_f1` overwritten by an empty `ebic1_search_f1`; `figures_ebic1/search_ceilings` was empty | renamed rows are not overwritten |
| `KeyError: 'MCP-up-lasso'` | `plot_campaign.py`, `STYLE` | wave 7's estimator had no style | style added |
| sign of the gradient at the diagonal fit | `docs/LIKELIHOOD.md` | the text said $\nabla_M = \hat\Sigma - I$; the code (correctly) gives $I - \hat\Sigma$. $\lambda_{\max}$ uses the absolute value and was never affected | text corrected |
| `scancel` without `-M serial` | `next_steps/091026/next_steps_091026.md:286` | on LRZ the command would act on the wrong cluster | `-M serial` added; no other instruction in the repository lacks it |
| one wave 5a cell at 350 of 400 graphs | `runs/campaign/direct_lasso-ml_Cresc_n1e4` | shard 2 was missing | arrived with the rsync of 12 October; the tables now hold 400 graphs. **Still to do:** the "350 graphs" remarks in `S4_campaign.md` (lines 10, 97, 105, 427, 439) and the numbers of S4 §5 that used them |

The full suite passes: 426 tests on 12 October (`pytest`, 8.5 min; the README count is
updated). Tests added during the review: `test_covloss.py` (gradient and Hessian against
finite differences, $\lambda_{\max}$ in closed form and as the exact threshold, with a full and
with the rescaled $C$), `test_nonconvex.py` (the MCP/SCAD prox near and at its step cap),
`test_lyap.py` (the Schur solver against scipy; the stability flag near the boundary),
`test_rescore_shard.py` (a source that cannot be rebuilt is refused in both modes),
`test_search.py` (the direct-loss score on three examples; searches restricted to simple
graphs), `test_restarts_analysis.py` (one curve set per true $C$).

## 2. Open findings, most severe first

### A. Numerics behind stored results

**A1. BUG — the covariance-loss solver accepts steps on an absolute slack, which gives false
"converged" results and stalls.** `src/gclm/solvers/covariance.py:89-90, 116-117, 230, 261-263`.
The step test allows a slack of $10^{-13}\max(1, |F|)$. Once the curvature term of a step is
below that, every step length passes, so the step is no longer controlled. Two effects:
(a) when the warm start is already stationary, which is common on a dense → sparse path because
MCP has zero slope beyond $\gamma\lambda$, one step at $t = 1$ is accepted, reported as converged,
and ends at a point up to 3,000 times less stationary than the start; (b) in stiff directions the
accepted steps are too long and the solve runs to the iteration cap. Measured on stored data
sets: 255–970 capped solves per log-likelihood cell with the rescaled $C$ (of 40,000), KKT up to
0.17. They concentrate on data sets whose $C$ has a ratio of largest to smallest diagonal entry
of 100 or more: all 16 such data sets for MCP dense → sparse and 13 of 16 for the lasso, against
51 and 16 of the 261 with a ratio below 10. A copy of the
loop with a gradient-based step test converges in 200–900 iterations where the current one needs
50,000. **Effect on results:** the one-step solves move $M$ by about $10^{-8}$ and change no
support; a full path with the corrected test differs at 2 of 100 $\lambda$ by one entry, with the
same oracle $F_1$. Capped solves account for 17–40 % of all solver iterations in these cells,
which is cluster time. **Fix:** return
at once if the start is stationary; judge convergence on the KKT residual of the accepted iterate;
replace the absolute slack by a rounding-safe test (gradient-based when the decrease is at
rounding level). **Tests:** from a stationary start on a badly scaled instance the solver returns
KKT ≤ tol; `converged` implies stationarity ≤ 10·tol. Because paths are warm-started, the
log-likelihood cells of waves 3 and 7 would need re-running after the fix; that decision is
best taken with the supervisor's feedback.

**A2. RISK — the likelihood refit behind the score is not checked for convergence, and on some
graphs with 2-cycles it has no maximiser.** `src/gclm/solvers/search.py:152-166`
(`CovRefit`, tolerance $10^{-8}$, 3,000 iterations). On a support with a 2-cycle the
unpenalised likelihood can approach its infimum only as the two entries of the 2-cycle grow
without bound (seen: $|M_{ij}|$ from 460 to 2,000 over 60,000 iterations, the score still
falling). The score of such a graph is then the value at the cap, 5–19 units above the infimum
at $n = 10^3$, about one to three edges' worth of BIC penalty: a bias against 2-cycles in every
search with the likelihood refit (waves 5a, 8 (b), 8 (c)). **Effect on results, measured:** on
the stored wave 8 cells, 0 of 16 selected and searched supports were capped, and in one full
search step 7 of 106 neighbours were capped, all adds that create a 2-cycle; tightening them
left the chosen move unchanged. Late steps with small improvements were not tested. **Fix:**
let `CovRefit` report `converged`, continue or flag unconverged scores, and record the number of
capped refits per search in the shards. **Related (A2b):** `Scorer.fit` recomputes the stored
matrix of the end graph from a different warm start than the one that was scored
(`search.py:295-300`), so with the likelihood refit `m_search_v` need not be the matrix behind
`search_score` (seen: max $|\Delta M| = 0.74$, $\Delta$score 0.04). No analysis reads these
values today.

**A3. RISK — a path on which every score is $+\infty$ is stored as a real selection.**
`run_s1_shard.py:182-183` with `search.py:417`; `run_search_shard.py:134-137, 153-155`. When
every least-squares refit on the path is unstable, `argmin` returns index 0, the densest graph.
25 of 99,200 stored rows (15 cells, all with the rescaled $C$, mostly the adaptive lasso at
$p \ge 15$); one searched graph still at $+\infty$; one pure search with all 11 starts dead; six
searches from the truth that end at $+\infty$ with $F_1 = 1$ and serve as the ceiling. Largest
effect on a cell mean: 0.0027 in `direct_adaptive_Cresc_p50_n1000` (standard error 0.0077).
**Fix:** store a flag (`bic_dead`, `pure_dead`, `truth_dead`) and let `campaign.py` drop or
count those rows.

**A4. RISK — the direct-loss solvers return silently at the iteration cap.**
`proxgrad.py:73-77, 147-149`. No iteration count or convergence flag leaves the solver; the
shards store NaN and −1. No cap was hit in the checks (KKT ≤ $8\times10^{-7}$), but the largest
counts reach 74–85 % of the cap at $p = 50$. **Fix:** return and store `iterations` and
`converged`.

**A5. RISK — with the likelihood refit and start blocks, the same end graph can get different
scores in different blocks.** Each block has its own score cache. On 36 end graphs the scores of
two caches agree to $10^{-15}$ in the median, but 4 differ by more than $10^{-9}$, up to
$7.8\times10^{-3}$ (once from the iteration cap, once from two stationary points of the
likelihood on one support). The best of $r$ starts across blocks and `restarts.py`'s "reached"
share can then be decided by optimisation noise. Concerns wave 8 (c) only; fixing A2 removes
most of it.

### B. Analysis

**B1. RISK — a searched graph compared with a selected one under the name `search_f1_diff`.**
`campaign.py:310`. When the reference lasso has no search (wave 3 ran `--select bic`), the
fallback pairs the estimator's searched graph with the lasso's selected graph. This hits the
wave 7 cells at $p = 10$ (`loglik_MCP-up-lasso`, `loglik_adaptive`): a "difference" of 0.21
there is mostly the search. No document quotes these numbers. **Fix:** leave the column empty or
name it differently. The fallback for `ebic1_search_f1` looks for a column that does not exist
and always gives NaN.

**B2. RISK — no completeness or duplicate checks when loading shards.** `campaign.load` and
`restarts.load_cell` read `shard_*.npz` without checking that all shards of a cell are there
(B2a; a cell at 350 of 400 graphs went unnoticed next to complete ones), that no graph appears
twice (B2b; a stale shard from the old 32-task layout next to the 128-task one would add
duplicate rows to the means but not to the paired table) or that a file is complete (B2c; a
truncated copy aborts the whole analysis with an unnamed `BadZipFile`). **Fix:** compare the
shards found with `n_shards`, refuse duplicate graph ids, name the file on read errors.

**B3. NIT — always-empty and misleading columns.** `ebic1_skeleton_f1` is listed in `REPORT`
but never computed, so it is NaN everywhere (and `apply_rule("ebic1")` copies the NaN over
`bic_skeleton_f1`). The rescored cells keep the source's `seconds`, so `cpu_hours` of every
`-ml` cell is the cost of the source path, not of the likelihood refit (up to 157 times too
small). The overlay dictionary is keyed by source cell only, so two rescorings of one cell
overwrite each other. The eBIC-scored searches (`search-pure-e1`) are compared with the lasso's
BIC search.

**B4. NIT — the documented AUC/AUPR convention differs from the code.** `S1_reproduction.md` §5
says the ROC points are sorted by false-positive rate and the PR curve closes from the
highest-recall point; the code (which matches Varando's R package and its test) does not sort
and closes from the lowest-recall point. On real paths the two conventions differ by about
0.005 in AUPR. **Fix:** correct the document.

### C. Cluster scripts and runners

**C1. RISK — six cell names belong to two waves, and `--fill` never gives up.**
`direct_{lasso,MCP-up,adaptive}-ml_{C2I,Cresc}` are defined in wave 5 (`run_s1_shard`, 8 shards)
and in wave 8 (`rescore_shard`, 4 or 16 shards). `--wave 8 --fill` on a wave 5 folder submits
the wrong runner with the wrong shard count; the task exits with an error, and because the
feeder counts only refused submissions, it resubmits the failing task forever and never reaches
the lines after it. The same loop follows from any shard that fails deterministically, including
one that always exceeds its time limit. Latent now, since the missing shard arrived. **Fix:**
record runner, arguments and shard count in each cell folder and let `--fill` use them, or rename
the wave 8 cells; give a fill line up when the number of missing shards stops falling.

**C2. RISK — feeder safeguards.** The check for running jobs of a wave fails open on an
`squeue` error (`grep -c … || true`) and then fills while the wave is still running; two feeders
started together submit every cell twice (no lock file; the log of 8 October shows two feeders
interleaved); a plan line that matches no cell is logged as "nothing left to submit, done".

**C3. GAP — no provenance.** Shards and the submission log record no commit hash; a queued
task runs whatever is checked out when it starts. **Fix:** write `git rev-parse HEAD` into
`config_json` and the submission log.

**C4. RISK — a task killed while saving leaves a truncated shard that `--fill` counts as
written.** `np.savez_compressed` writes straight to the final name. **Fix:** write to a
temporary name and `os.replace`.

**C5. NITs in the runners.** More shards than tasks writes a shard without `p`, which the
analysis cannot read; `--restarts 0 --start-blocks 2` crashes; `--reps 0` and `--tol 0`
silently fall back to the defaults; `--ebic-gamma 1.5` and `15` give the same tag; settings that
are ignored (`--direction up` with `--method adaptive`, `--convention ncvreg` with the covariance
losses, `--refit` with `--select none`) are still recorded in `config_json`; the covariance-loss
`converged` flag is not stored (135 of 400 rows of `loglik_MCP-up_Cresc_n1e4` hit the 50,000
cap); `run_m0.py` uses one random stream, so a data set depends on `--reps`; the rescore check
against the source uses a tolerance that fails on 5 of 800 laptop recomputations of cluster
shards (never on the cluster). Stale docstrings in `run_search_shard.py:33-34` (unpacking with
start blocks) and `run_s1_shard.py:25-28` ("least-squares refit always").

### D. Core numerics and documentation (no effect on stored results)

- `SchurLyapunov.solve` symmetrises its output, which is wrong for a non-symmetric right-hand
  side that the docstring allows; no caller passes one. `solve_lyapunov` does not check
  stability.
- $p = 1$: the direct $\lambda_{\max}$ raises, the covariance one returns 0 (a grid of zeros);
  `lambda_grid(n_lambda=1)` contradicts its docstring; non-finite $M$ raises instead of giving
  $+\infty$; the direct gradient assumes a symmetric $C$ without checking.
- `sample_drift` with $k > p$ fails with numpy's message; `draw_instance` with `n` NaN, $-\infty$
  or `None` silently returns the population covariance (the command lines reject these).
- `bic_direct` counts off-diagonal residuals twice (score unused in the campaign).
- `docs/SEARCH.md:54` gives the refit tolerance as $10^{-9}$ (the code uses $10^{-8}$, 3,000
  iterations); `docs/SEARCH.md:211` says the campaign does not use `--add-screen`, but wave 8 (c)
  uses `--add-screen 20`.
- `path.py:257` walks `lambdas[::-1]` and so assumes an increasing grid; every other path sorts.
  No caller passes another order.
- `greedy_search` does not flag a search that stops at `max_steps`; `Scorer.score` does not mask
  the diagonal (callers do).

## 3. Checked and found correct

Gradients and Hessians of both losses against finite differences (also near instability and
with a full or rescaled $C$); $\lambda_{\max}$ as the exact threshold for lasso, MCP and SCAD on
both losses; the diagonal and the exact fits; the MCP/SCAD prox against brute force on 23,694
cases; the Schur stability flag on 3,000 near-boundary matrices; FISTA and MAPG against
coordinate descent; the starts and directions of every path type; the adaptive-lasso weights and
their $\lambda_{\max}$; the direct-loss refit and the search invariants (strict decrease, stored
score equals a fresh score, no improving neighbour at the end); the cache keys; the data
generator against Dettling's recipe (7,600 drifts, all stable); the rescaled $C$ exact for
`C_ID` at $n = \infty$; task slicing for 1 to 259 shards, start blocks included; every stored
selection index equal to the arg-min of its stored scores (99,200 rows); the rebuild of every
rescored data set from its source shard; pairing by (p, k, true $C$, replicate) valid in every
stored cell; offline and online eBIC selections identical (9,600 graphs); the restart curves
reproduced by brute force; the paired differences, standard errors and $z$ against a direct
computation; cluster scores bit-identical across nodes.

## 4. Recommended order

1. **A1 with its tests**, then decide with the supervisor whether to re-run the log-likelihood
   cells of waves 3 and 7 (about 700 CPU-h per sample size for wave 7). Until then, the
   log-likelihood results are stated as they are, with A1 noted.
2. **A2 and A3:** convergence and dead-selection flags in the shards, so the next runs report
   them; cheap, no re-runs needed.
3. **C1, C2, C4** before the next large submission: the feeder is the part most likely to waste
   cluster time.
4. **B1–B3** in the analysis, and the "350 graphs" remarks in `S4_campaign.md` (§1).
5. **B4, C5, D:** documentation and small fixes, whenever convenient.
