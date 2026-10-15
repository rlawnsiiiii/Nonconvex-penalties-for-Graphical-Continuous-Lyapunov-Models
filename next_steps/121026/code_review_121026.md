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
- **Two numerical issues touch every log-likelihood result**: the covariance-loss solver could stop
  without having converged (A1), and the likelihood refit behind the score has no maximiser on some
  graphs with 2-cycles (A2). Both were measured to change no support and no search decision in the
  cases checked, but both were silent. Both are fixed as of 13 October (§1b): the solver's step test
  and stopping rule, and convergence flags everywhere. What A1's fix does not remove is the
  first-order method's crawl along flat valleys of the nonconvex MCP objective, which keeps a few
  per cent of the MCP dense → sparse solves at the iteration cap on badly scaled data sets; those
  are now flagged. Whether to re-run the log-likelihood cells with the corrected solver is the one
  decision left open.
- **The cluster scripts had several ways to loop or double-submit** (C1, C2); none had happened in
  a way that changed results, and all are closed (§1b).
- **Status on 13 October:** every finding of §2 is fixed, tested (464 tests pass) and documented,
  except the re-run decision and the flat-valley caps just named; §1b lists the fixes, §4 what
  remains.

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

## 1b. Fixed (13 October)

Every finding of §2 was fixed on 13 October 2026, with a test for each change; the full suite has
464 tests (426 on 12 October). The log-likelihood cells of waves 3, 7 and 8 stored before that
date were computed with the old solver (A1) and are stated as stored (`S4_campaign.md` §6 note).

| finding | fix | where | test |
|---|---|---|---|
| A1 | the backtracking test uses the loss values while they resolve the comparison and the secant condition on the gradients at rounding level; convergence is judged on the first-order residual of the accepted iterate; a stationary start is returned unchanged | `solvers/covariance.py` (`_Problem._bound_holds`, `solve`); `docs/LIKELIHOOD.md` §4.3 | `test_covloss.py`: stationary start, badly scaled $C$ converges, converged ⇒ residual ≤ tol |
| A1, measured | worst-scaled stored data set ($C$ ratio 250, $p = 10$, rescaled $C$, $n = 10^4$): lasso path identical at all 100 $\lambda$, no cap, residual at tolerance, a third fewer steps (604k vs 939k); MCP dense → sparse from the exact fit: 8 of 100 supports differ by 1 to 3 entries, oracle $F_1$ 0.493 → 0.507, 33 of 100 solves still at the cap (21 before). Of those, the dense end is slow convergence (residual $2\times10^{-8}$ at 50,000 steps; a Newton polish moves $2\times10^{-7}$); at $\lambda \approx 10^{-3}\lambda_{\max}$ the iterate crawls along a flat valley (residual $10^{-3}$ to $5\times10^{-5}$, the plain method no better) and a Newton step from there reaches another stationary point 0.8 to 1.1 away with 20 to 22 support changes. The flat valley is the problem's (`docs/LIKELIHOOD.md` §5), not the test's; such solves are flagged by `converged` | | |
| A2 | `CovRefit` and `DirectRefit` count their refits and those stopped at the cap (`unconverged`, `last_converged`); `Scorer.unconverged`; `SearchResult.unconverged` / `capped`; stored per graph as `bic_unconverged`, `search_unconverged`, `search_capped`, `pure_*`, `truth_*`. Scores unchanged | `solvers/search.py`, the three runners; `docs/SEARCH.md` §3 | `test_search.py::test_refits_report_whether_they_converged`, `test_campaign_runner.py`, `test_search_shard.py` |
| A2b | the search keeps the matrix of its best move when that support was scored in the step; a recomputed matrix (cache hit) is rescored, a lower score replaces the cached one, a higher one is counted (`mismatches`, stored as `search_mismatches`) | `solvers/search.py` (`Scorer.score_new`, `_consistent`) | `test_search.py::test_search_result_flags_and_the_matrix_behind_the_score` |
| A3 | `bic_dead` / `search_dead` / `pure_dead` / `truth_dead` stored; `campaign.py` derives them for every shard (old ones from the stored scores), counts them per cell (`campaign_means.csv`) and blanks them with `--drop-dead`. Found: 22 dead paths (15 cells, rescaled $C$, mostly the adaptive lasso at $p \ge 15$), 1 searched graph, 1 pure search, 6 searches from the truth | runners, `campaign.py` | `test_campaign_analysis.py::test_dead_selections_*` |
| A4 | `solve_fista` / `_solve_mapg` return `ProxInfo(iterations, converged)` on request; every path reports `iterations` and `converged` per $\lambda$; the shards store them for the direct loss too, with the first-order residual `kkt` | `solvers/proxgrad.py`, `solvers/path.py`, `run_s1_shard.py` | `test_fista.py`, `test_lasso.py::test_path_diagnostics_and_grid_order`, `test_campaign_runner.py` |
| A5 | follows from A2 (counted, documented); no code beyond A2 | `docs/SEARCH.md` §3 | |
| B1 | no fallback: a column the reference lacks gives NaN | `campaign.py::paired_table` | `test_campaign_analysis.py::test_means_and_paired_differences` |
| B2 | `campaign.load_shards`: unreadable file named, duplicate graphs refused, incomplete cells reported (stderr; `shards` / `n_shards` columns; "incomplete:" lines); `restarts.py` uses it | `campaign.py`, `restarts.py` | `test_campaign_analysis.py::test_loader_checks_the_shards` |
| B3 | `ebic<g>_skeleton_f1` computed (every selected graph gets `graph_metrics`); rescored cells' `seconds` is their own cost (`rescore_shard.py` keeps the source's as `source_seconds`; old refit shards read `select_seconds`); one overlay per rescoring folder; `-e1` searches compared with the lasso's `ebic1_search_*` | `campaign.py`, `rescore_shard.py` | `test_campaign_analysis.py`, `test_rescore_shard.py` |
| B4 | the ROC and PR text now describes the code (no sorting; closing segment from the lowest-recall point) | `S1_reproduction.md` §5 | existing R-port test |
| C1 | each cell folder records `submission` (wave, runner, arguments, time); `--fill` resubmits from it and reports a cell recorded by another wave; a legacy folder is filled only when its shard count matches; the feeder gives a fill line up when its jobs ended without reducing the missing shards | `cluster/submit_campaign.sh`, `cluster/feed_queue.sh` | `test_campaign_submit.py::test_fill_uses_the_cells_record_*`, `test_feed_queue.py` |
| C2 | lock directory with the feeder's pid (stale locks taken over); `squeue` failure makes a fill wait; a plan line whose dry run fails (e.g. `--only` matching nothing, now exit 2) is skipped with the message | `cluster/feed_queue.sh`, `cluster/submit_campaign.sh` | `test_feed_queue.py::test_one_feeder_at_a_time_*` |
| C3 | `provenance_json` carries the git commit and a dirty flag (the task's code, not the submission's); `logs/submissions.log` records every `sbatch` call with the commit | runners (`git_revision`), `submit_campaign.sh` | `test_campaign_runner.py` |
| C4 | shards written to a temporary name and renamed into place (`save_npz`) | runners | `*.tmp` absent after every runner test |
| C5 | empty shards keep the id columns; `--restarts 0 --start-blocks 2`, `--reps 0`, `--tol 0`, `--ebic-gamma` outside (0, 10) or with two decimals, duplicate tags, `--direction` with `lla`/`adaptive`, `--convention` or `--solver` with a covariance loss, `--refit` without `--select`, `--add-screen` without the likelihood search are refused; `run_m0.py` seeds per (setting, n, replicate); the rebuild tolerance of `rescore_shard.py` is $10^{-6}$ relative; docstrings corrected | runners | `test_campaign_runner.py::test_settings_without_effect_*`, `test_search_shard.py` |
| D | `SchurLyapunov` symmetrises only a symmetric right-hand side; `solve_lyapunov(check=True)` refuses unstable or non-finite $M$; both `lambda_max` refuse $p = 1$; `lambda_grid(n_lambda=1)` gives $\lambda_{\max}$; a non-finite $M$ has loss $+\infty$; the direct-loss solvers check that $C$ is symmetric; `bic_direct` sums the $N$ distinct residuals once; `lasso_path` walks any grid order; `greedy_search` flags `capped`; `Scorer` masks the diagonal; `sample_drift` refuses $k > p$, `draw_instance` refuses `nan` / `-inf`; `docs/SEARCH.md` tolerances and `--add-screen` text corrected | `lyapunov.py`, `objective/*.py`, `solvers/*.py`, `data/simulate.py` | `test_lyap.py`, `test_covloss.py::test_degenerate_inputs`, `test_fista.py::test_c_must_be_symmetric`, `test_search.py`, `test_dgp.py` |
| §1 "still to do" | the "350 graphs" remarks removed from `S4_campaign.md`, the wave 5a table restated at 400 graphs (lasso, rescaled $C$: 0.566 → 0.568 selected, 0.597 → 0.600 after the search), the figures redrawn from the complete cells | `S4_campaign.md` §0, §5, §6a | |

## 2. The findings in detail (as of 12 October; fixes in §1b)

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

## 4. What remains (13 October)

1. **Whether to re-run the log-likelihood cells of waves 3, 7 and 8 with the corrected solver**
   (about 700 CPU-h per sample size for wave 7): the measured changes (§1b, A1) are well inside the
   cells' standard errors, so the current findings stand; the decision is the supervisor's.
2. **The flat-valley caps of the MCP dense → sparse path** on badly scaled data sets (§1b, A1):
   a property of the first-order method on this nonconvex objective, not a defect of the test. Now
   flagged per $\lambda$ (`converged`). A second-order polish would reach other stationary points
   and change the estimator; left as it is.
3. **`S1Config.tol`:** the solver study of 12 October recommends $10^{-10}$ for new direct-loss
   runs (`solver_accuracy_121026.md` §6); not changed here, since the campaign's cells are stored
   at $10^{-8}$ and pair with each other.
4. **Pulling these changes on LRZ:** safe for the scripts at any time (the running feeder picks up
   the new `submit_campaign.sh`; legacy folders are filled when their shard counts match), but the
   solver change should not reach the cluster while wave 8 at $n = 10^3$ is running, or those
   cells would mix two solver versions. Pull after it ends.
