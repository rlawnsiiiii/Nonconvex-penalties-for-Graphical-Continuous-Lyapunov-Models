# Plan: four mechanism experiments (13 October 2026)

*Working plan for Claude Code. Repository: `Nonconvex-penalties-for-Graphical-Continuous-Lyapunov-Models`,
branch `main` at `d24a32b` or later. Suggested location for this file: `next_steps/131026/plan_mechanism_131026.md`.*

**Where things run.** Every experiment that could end up in the thesis, even as a small part, runs on
LRZ as **wave 9** of the campaign (`cluster/submit_campaign.sh`, fed by `cluster/feed_queue.sh` from a
new `cluster/plan_131026.txt`). The laptop is for three things only: (i) code and tests, (ii) a rehearsal of
one shard per new cell type with `--reps 1` through `cluster/local/sbatch`, to check that the cells run and
to time them, and (iii) the analysis scripts, which by convention run locally on the shards brought home
(`simulations/diagnostics/*.py`, `plot_*.py`). Nothing with more than a handful of datasets runs on the
laptop.

**Why these four.** The campaign (waves 1–8, `simulations/VERDICTS.md`) established that MCP/SCAD on
the standard sparse → dense path lose to the lasso, and that pruning a dense start (MCP dense → sparse,
LLA, adaptive lasso) wins, with the convex adaptive lasso as good as or better than MCP. The open
question is *why*. The four experiments below each isolate one part of the mechanism:

| # | question | what changes | computation |
|---|---|---|---|
| E1 | Which property of the dense start matters? | the start (min-ℓ1, min-ℓ2, exact fit, truth + noise) | cluster, wave 9 |
| E2 | Does plain thresholding of the dense end match the adaptive lasso? | the pruning step | cluster, wave 9 |
| E3 | Does the gain grow with the minimum edge weight (beta-min)? | the weight distribution of the DGP | analysis of stored shards first (laptop), then cluster cells |
| E4 | Is the gain concentrated in identifiable graph classes? | the graph class of the DGP (DAG / no 2-cycles / random) | analysis of stored shards first (laptop), then cluster cells |

Everything goes through the existing shard runner (`simulations/run_s1_shard.py`), so that
`simulations/diagnostics/campaign.py` aggregates the new cells like any other and every comparison is
paired with wave 1 through `(p, k, c_choice, rep)`.

---

## 0. Before writing code

Read, in this order: `ARCHITECTURE.md`, `docs/DENSE_START.md`, `docs/REPRODUCTION.md`,
`simulations/S4_campaign.md` §0, `simulations/run_s1_shard.py` (module docstring and `main`),
`src/gclm/solvers/path.py` (`lasso_path`, `lla_path`, `adaptive_lasso_path`), `src/gclm/data/simulate.py`,
`simulations/diagnostics/campaign.py` (`CELL` regex, `load_estimator_cell`), `tests/test_lla_adaptive.py`,
`tests/test_dgp.py`, `cluster/submit_campaign.sh` (`cells()`, `--only`, `--status`, `--dry-run`, `--fill`),
`cluster/feed_queue.sh`, `cluster/plan_101026.txt`, `cluster/local/sbatch`,
`next_steps/091026/next_steps_091026.md` §2 (the cluster commands) and
`next_steps/051026/cluster_commands_051026.md` (block 9: `rsync`).

House rules (from the existing code and memos):

- **Defaults never change.** Every new behaviour is behind a new option whose default reproduces the
  current output bit for bit. The existing seeds must produce the existing graphs: `tests/test_dgp.py`
  and `tests/test_campaign_analysis.py` must pass unchanged.
- **Numbers only on runners**; figures only in `simulations/diagnostics/plot_*.py`, local.
- Terms: a graph is selected by its **score = loss + penalty** (BIC penalty or eBIC penalty); "sparse →
  dense" and "dense → sparse", never "standard path" in new labels (see `docs/SEARCH.md` §2).
- Every new function gets a test in `tests/`; run the full suite (`pytest -q`, ~400 tests) before reporting.
- A cell is what wave 1 calls a cell: `--p 10 20 --reps 25` with all four true-C settings and k = 1…4, i.e.
  400 datasets per p, 800 per cell; shards keep one task under about 6 h (a quarter of the 24 h limit),
  with LRZ at about twice the laptop time per dataset.
- One dated memo at the end: `next_steps/131026/next_steps_131026.md` with what was built, how to run
  it, the cluster commands, and the first numbers. Do not edit `simulations/VERDICTS.md`; leave verdict
  wording to the author.

---

## 1. Library additions (shared by E1–E4)

### 1.1 Exact fits of the Lyapunov equation — new module `src/gclm/objective/exact_fits.py`

The exact solutions of `M Σ̂ + Σ̂ Mᵀ + C = 0` form the affine space `{ M0 + W Σ̂⁻¹ : W skew }` with
`M0 = -½ C Σ̂⁻¹` (`gclm.objective.covariance.dense_fit`), dimension `p(p-1)/2`. Implement three
members of that space, all with the **diagonal free** (only off-diagonal entries enter the norm, matching the
unpenalised diagonal everywhere else in the code):

```python
def skew_basis(p) -> np.ndarray            # (p(p-1)/2, p, p) basis E_ij - E_ji, i < j
def min_l2_exact_fit(sigma_hat, c) -> np.ndarray
    # argmin ||M_off||_F over exact solutions: linear least squares in the skew coordinates w,
    # M(w) = M0 + sum_k w_k B_k Σ̂⁻¹, minimise the sum of squares of the off-diagonal entries of M(w).
    # p(p-1)/2 unknowns, p(p-1) equations; numpy.linalg.lstsq.  p ≤ 50 is trivial.
def min_l1_exact_fit(sigma_hat, c, *, solver="highs") -> np.ndarray
    # argmin ||M_off||_1 over exact solutions: LP in (w, t) with |M(w)_ij| ≤ t_ij, scipy.optimize.linprog.
    # This is the "BP" solution of claude_docs/.. main_claim_and_theory_071026.md §2.4 / §4.
def noisy_truth_start(m_true, tau, rng) -> np.ndarray
    # M* + tau * N(0,1) on every off-diagonal entry (dense); diagonal unchanged.  A ceiling, not an estimator.
def exact_fit_residual(m, sigma_hat, c) -> float   # ||M Σ̂ + Σ̂ Mᵀ + C||_F, for tests and diagnostics
```

`scipy` is already a dependency (`pyproject.toml`); the LP needs nothing else. Make sure the cluster
environment (`cluster/setup_env.sh`) has a scipy with HiGHS (≥ 1.6); if not, add it there.

Tests (`tests/test_exact_fits.py`):
- all three fits satisfy the Lyapunov equation to `1e-8` (relative to `||C||_F`), `p ∈ {5, 10, 20}`;
- `min_l2` has the smallest off-diagonal Frobenius norm among `{min_l2, min_l1, dense_fit}` and `min_l1`
  the smallest off-diagonal ℓ1 norm, each by a strict margin on random instances;
- regression against the 7 October note: on the thesis DGP at p = 10, correlation scale, rescaled C,
  `n = inf`, the repository seeds, `M*` is the unique min-ℓ1 solution in **11 of 100** graphs (the note's
  "BP condition" count; uniqueness probed with a few random linear objectives as the note did). This is the
  one laptop-scale computation allowed in this step (about 2 minutes). If the seeds cannot be matched
  exactly, assert a count in `[8, 14]` and say so in the memo.

### 1.2 Any dense start for the direct-loss dense → sparse path — `src/gclm/solvers/path.py`

`lasso_path(direction="up", penalty="MCP"/"SCAD")` computes its own start (the lasso at the smallest λ).
Add a keyword `start: np.ndarray | None = None`: when given, use it instead of that lasso solution as the
warm start at the smallest λ. Everything else unchanged; the runner records the start in the config.

For the covariance losses `covloss_path(..., direction="up", start=...)` already takes `"exact" | "lasso"`
(`UP_STARTS`); extend it to accept an array as well, for the optional loglik arms of §2.6.

Tests: `start=` equal to the internally computed lasso start reproduces the current path exactly; a different
start changes only the warm starts, not the grid; a `start` with the wrong shape raises.

### 1.3 Thresholded lasso — `src/gclm/solvers/path.py`

```python
@dataclass
class ThresholdedLassoPath(LassoPath):
    thresholds: np.ndarray      # the "lambdas" of this path are thresholds on |pilot_ij|
    pilot: np.ndarray

def thresholded_lasso_path(sigma, c, pilot=None, n_thresholds=100, refit="direct", loss="direct",
                           tol=..., max_iter=...) -> ThresholdedLassoPath
```

- Pilot: the lasso solution at the smallest λ of its own grid, exactly as `adaptive_lasso_path` does
  (`pilot=None` → compute it; pass the same pilot to both for a paired comparison).
- Threshold grid: `n_thresholds` values, log-spaced between the smallest and the largest nonzero `|pilot_ij|`
  (off-diagonal), plus `0` at the dense end and `+inf` at the sparse end, so that the path runs from the pilot's
  full support to the empty graph. Store the grid in `thresholds` and also in `lambdas` (increasing order), so
  that `evaluate_path`, `bic_along_path` and the shard runner treat it like any path.
- Estimate at threshold `t`: support `|pilot_ij| ≥ t` (off-diagonal), **refit** on that support with
  `gclm.solvers.search.make_refit(sigma, c, loss)` (least squares on the direct loss; the likelihood refit
  for `loss="loglik"`), diagonal always free. Keep a `refit="none"` option that returns the thresholded
  pilot itself (for AUPR this is the ranking by `|pilot|`).
- Diagonal unpenalised, textbook convention, nothing else.

Tests (`tests/test_threshold.py`): threshold 0 reproduces the pilot's support; threshold `inf` gives the
diagonal fit; supports are nested and monotone along the grid; the refit on the true support equals
`DirectRefit.fit(S*)`; the path object works with `evaluate_path` and `bic_along_path`.

### 1.4 DGP options — `src/gclm/data/simulate.py` and `src/gclm/config.py`

`sample_drift(p, d, rng, metzler=False, *, weights="normal", w_min=0.0, graph="random")`:

- `weights="normal"` (default): `N(0,1)` as now. `weights="uniform"`: `|w| ~ U[w_min, 1]` with an independent
  random sign. The diagonal rule (`-Σ|M_ij| - |eps_ii|`) stays, so stability is unchanged.
- `graph="random"` (default): `Bernoulli(d)` on every ordered pair, as now. `graph="no2c"`: draw as now, then for
  every 2-cycle drop one direction at random (the 7 October note's `unif_no2c` variant). `graph="dag"`: draw a
  random permutation π and keep only edges `j → i` with `π(j) < π(i)`, each with probability `d'` chosen so that
  the expected number of edges equals the random class (`d' = 2d` capped at 1; document the choice). Every DAG is
  also "no2c".
- **Random-number stream.** The default call must consume the generator exactly as today. Implement the new
  options so that `weights="normal", graph="random"` takes the current code path verbatim; the new branches may
  draw extra numbers after it.

`draw_instance(..., weights=, w_min=, graph=)` passes these through. `S1Config` gets `weights`, `w_min`, `graph`
with the current defaults; they are written into the shard's stored config.

Tests (`tests/test_dgp.py`): defaults reproduce the stored reference drift matrices; `uniform` respects
`w_min ≤ |w| ≤ 1`; `no2c` has no pair with both directions; `dag` is acyclic (Kahn's algorithm, implemented in
§1.5); all stable; expected edge count of `dag` within 10 % of `random` over 200 draws.

### 1.5 Graph-class diagnostics — `src/gclm/graphs.py` (new, no internal dependencies)

```python
def is_dag(support: np.ndarray) -> bool                 # Kahn's algorithm on the off-diagonal support
def two_cycles(support) -> int                          # number of pairs with both directions
def frac_edges_in_two_cycles(support) -> float
def graph_class(support) -> str                         # "dag" | "simple" (cyclic, no 2-cycles) | "two_cycles"
def identified_on_support(m_true, c_true) -> float      # min eigenvalue of Γ_{S̄S̄} = (Aᵀ A) restricted to the
                                                        # true support plus the diagonal, A = design_matrix(Σ*),
                                                        # Σ* = solve_lyapunov(M*, C*).  > 0 means identified.
```

Tests: hand-built 3- and 4-node examples for each class; `identified_on_support` is 0 for a graph with an
isolated 2-cycle and > 0 for a chain.

---

## 2. E1 — Which dense start?

### 2.1 Arms

All on the **direct loss**, textbook convention, diagonal unpenalised, p ∈ {10, 20} (one cell, as wave 1),
n ∈ {10³, 10⁴, ∞}, both C (`--c-scale identity` and `variance`), wave 1's datasets (same seeds, so everything
is paired with the stored lasso, MCP-up and adaptive cells).

| start | how | code |
|---|---|---|
| `lasso` | lasso at the smallest λ of the grid (`ratio · λ_max`) — the current start | existing (wave 1) |
| `l1` | the exact min-ℓ1 fit | `min_l1_exact_fit` |
| `l2` | the exact min-ℓ2 fit | `min_l2_exact_fit` |
| `exact` | Varando & Hansen's `-½ C Σ̂⁻¹` | `dense_fit` |
| `truth-0.05`, `truth-0.2` | `M* + τ N(0,1)` off the diagonal (ceiling) | `noisy_truth_start` |

Prune each start two ways: **MCP dense → sparse** (`lasso_path(direction="up", penalty="MCP", start=...)`) and
the **adaptive lasso** (`adaptive_lasso_path(pilot=start)`). The `lasso`-start arms are wave 1's `MCP-up` and
`adaptive` cells; the five new starts × two pruners give 10 new arms per (C, n).

### 2.2 Runner

`simulations/run_s1_shard.py`: let `--up-start` apply to the direct loss too, with choices
`lasso | exact | l1 | l2 | truth-<tau>`; for `--method adaptive` add `--pilot` with the same choices (default
`lasso`, the current behaviour). The start is computed once per dataset and handed to the path function.
Record the start's own diagnostics per dataset in the `.npz` (`start_*` keys): its off-diagonal ℓ1 and ℓ2
norms, its Lyapunov residual, its number of nonzero off-diagonal entries, and the **dense-start content**
of the 7 October note: among true edges (outside 2-cycles), the count with the true direction present
alone / both directions present / only the reverse present / neither. (Use `|M_ij| > 1e-8` as "present" for
the dense fits, which have no exact zeros.) Also store the start's confusion counts, so that "F1 of the
start" is available.

### 2.3 Cells (wave 9, group `starts`)

```
direct_MCP-up-<start>_<C>_n<n>        e.g. direct_MCP-up-l2_Cresc_n1e4
direct_adaptive-<start>_<C>_n<n>      e.g. direct_adaptive-truth0.2_C2I_n1000
```

with `<start> ∈ {l1, l2, exact, truth0.05, truth0.2}`, C ∈ {C2I, Cresc}, n ∈ {1000, 1e4, inf}: 60 cells of 800
datasets, `--p 10 20 --reps 25 --select search`. Extend the `CELL` regex in `campaign.py` so that the estimator
token may contain a dot. Shard counts: start from wave 1's (4 for adaptive, 16 for MCP-up, `12:00:00`) and
correct them with the rehearsal timing (§6, step 5): LRZ seconds per dataset ≈ 2 × laptop, task ≤ 6 h.

### 2.4 Cost

Wave 1 ran 16 cells of this shape. E1 is 60 cells, with the LP and the min-ℓ2 fit adding well under a second
per dataset. Expect roughly four times wave 1's CPU-hours; put the exact figure in the memo from the
rehearsal before submitting. If it has to be cut, drop `n = inf` and `truth-0.05` first (that leaves 32 cells).

### 2.5 Analysis — `simulations/diagnostics/starts.py` and `plot_starts.py` (local, on the shards brought home)

From `campaign_per_dataset.csv` (extend `campaign.py` to read the `start_*` keys into columns):

1. Table per (p, C, n): `max_f1`, `aupr`, `bic_f1`, `search_f1` for the 12 arms (10 new + the two wave 1
   arms), paired differences to the lasso and to the `lasso`-start arm, with standard errors over graphs.
2. Table "what the start contains": the dense-start content per start (true direction alone / both / only
   reverse / neither), its F1, its ℓ1 and ℓ2 norms.
3. Figure `starts_by_pruner.png`: `max_f1` of the pruned estimator against the start's own orientation
   recall (share of true edges whose true direction is present in the start), one point per (start, pruner),
   panels (p, C). This is the test of "the start matters because it already orients most edges": the points
   should line up.
4. Figure `starts_bars.png`: paired gain over the lasso per start, MCP and adaptive side by side, over n.

Readings to decide between (write them in the memo as hypotheses, let the numbers pick):
(a) `l1` ≈ `lasso` ≫ `l2` ≈ `exact`: the ℓ1 geometry of the start matters (it hedges and orients); (b)
`l2` ≈ `l1`: any exact fit does, the message is "start dense"; (c) `truth-τ` far above everything: the
pruners are fine, the starts are the bottleneck.

### 2.6 Optional (a second plan line, submitted only if E1 on the direct loss is clear)

The same 10 arms on the log-likelihood loss at p = 10, n ∈ {10³, 10⁴} (`covloss_path(start=array)`,
`adaptive_covloss_path(pilot=array)`), cells `loglik_MCP-up-<start>_<C>_n<n>` and
`loglik_adaptive-<start>_<C>_n<n>`, shards as wave 7's p = 10 cells. Group `starts-loglik` in wave 9.

---

## 3. E2 — Thresholding the dense end

### 3.1 Design

Wave 1's datasets. One new estimator arm, `threshold`, on the direct loss: the thresholded lasso of §1.3 with
the **same pilot** as the adaptive lasso, so the comparison is paired at the pilot level. Two variants:
`threshold` (refit on the support, the estimator) and `threshold-norefit` (the pilot cut at `t`, no refit; for
AUPR only). Compare with wave 1's `adaptive` and `MCP-up` cells, p ∈ {10, 20}, n ∈ {10³, 10⁴, ∞}, both C.

Metrics as always: `max_f1` over the threshold grid (the oracle threshold, the analogue of the oracle λ),
`aupr` (the pilot's ranking), `bic_f1` with the score selecting the threshold (`bic_along_path` over the
supports of the thresholded path), and `search_f1` from the selected support. The thresholded path goes
through `--select search` exactly as any other path.

### 3.2 Cells (wave 9, group `threshold`)

`--method threshold` and `--method threshold-norefit` in `run_s1_shard.py` (`METHODS`); cells
`direct_threshold_<C>_n<n>` and `direct_threshold-norefit_<C>_n<n>`, `--p 10 20 --reps 25 --select search`:
12 cells. Cost per dataset: one lasso path plus ≤ 100 closed-form refits, cheaper than the adaptive lasso;
4 shards, `12:00:00`, as wave 1's adaptive cells.

### 3.3 Analysis (local)

In `starts.py`: a table `threshold | adaptive | MCP-up | lasso` per (p, C, n) with paired differences and `z`;
figure `threshold_vs_adaptive.png`: paired `max_f1` difference adaptive − threshold per graph, histogram per
(p, C, n), plus the same for `bic_f1`. Reading: a difference within ±0.01 means pruning is the whole story;
a clear positive difference means the reweighting matters, and the memo should then say *where* (edges in
2-cycles? weak edges? use the E3/E4 columns).

Also report, per graph, the share of pairs in which the pilot keeps both directions and which direction the
thresholded and the adaptive estimator keep: the thresholded one keeps the larger `|pilot_ij|`, the adaptive
one may not. That difference is the candidate mechanism if the adaptive lasso wins.

---

## 4. E3 — Beta-min curve

### 4.1 Analysis first, on the stored campaign (laptop, no runs)

Every campaign shard stores `M*` (`m_true_i/j/v`) and the selected / searched graphs. Add to
`campaign.py` an edge-level output `campaign_edges.csv`: one row per (cell, graph, true edge) with `|w|`,
whether the edge is in a 2-cycle, and whether it is recovered (a) in the best-F1 graph, (b) in the selected
graph, (c) after the search; plus the false positives per graph binned by `|M̂_ij|`. Then
`simulations/diagnostics/betamin.py` produces recall curves against `|w|` (bins `[0, .1), [.1, .25), [.25, .5),
[.5, 1), [1, 2), ≥ 2`) per estimator, for the wave 1 cells at p = 10 and 20, both C, n ∈ {10³, 10⁴, ∞}.
Figure `recall_by_weight.png`. Reading: the dense-start estimators should separate from the lasso on the
*strong* edges (direction) and tie on the weak ones (nobody finds them).

### 4.2 Cells (wave 9, group `betamin`): the weight distribution as a knob

DGP `weights="uniform"` with `w_min ∈ {0, 0.1, 0.25, 0.5}`, `graph="random"`, `--p 10 20 --reps 25 --select
search`, n ∈ {10³, 10⁴}, both C, four estimators: `lasso`, `MCP` (sparse → dense), `MCP-up`, `adaptive`.
Cell names

```
direct_<estimator>_<C>_w<wmin>_n<n>      wmin written without the dot: w0, w01, w025, w05
```

(`CELL` regex: an optional `_w[0-9]+` group before `_n`; carry it as a column `w_min`). 4 × 4 × 2 × 2 = 64 cells
of 800 datasets; shards as wave 1 per estimator (4 / 8 / 16 / 4). Runner flags `--weights uniform --w-min 0.25`
(`S1Config.weights / w_min`). This is about four times wave 1; if it has to be cut, run p = 10 only (`--p 10`,
cells named with `_p10` as wave 4 does) or drop `w_min = 0.1`.

Note that `w_min = 0` with the uniform law is not the thesis DGP (`N(0,1)` has a different shape and heavier
tails); include one `weights="normal"` reference line from wave 1 in every figure.

### 4.3 Analysis (local)

`betamin.py`: `max_f1`, `bic_f1`, `search_f1` against `w_min`, per estimator, panels (p, C, n); paired gains
over the lasso against `w_min` with standard errors; figure `gain_by_wmin.png`. Prediction to check: the
dense-start gain increases with `w_min`; the sparse → dense deficit does not shrink (its problem is
direction, not signal).

---

## 5. E4 — Graph class

### 5.1 Analysis first, on the stored campaign (laptop, no runs)

Add the §1.5 diagnostics to `campaign_per_dataset.csv` as columns `graph_class`, `n_two_cycles`,
`frac_edges_2c`, `identified_min_eig` (regenerate `Σ*` from the stored `M*` and `C*`; if `C*` is not in the
shard, regenerate the instance from its stored `seed` with `draw_instance`, which is deterministic, and assert
that `M*` matches). Then `simulations/diagnostics/graph_class.py`: means and paired gains per
(graph_class, k) for the wave 1 cells. **Compare within k only**: at p = 10 the class is almost determined
by k (k = 1 ⇒ mostly no 2-cycles, k = 4 ⇒ almost all with 2-cycles), so a pooled comparison would just be
the density effect of verdict 12. Figure `gain_by_class.png`: paired gain over the lasso per class, one
panel per k, both C, n = 10³ and 10⁴. Report the number of graphs per (class, k) in the figure; some cells will
be nearly empty, say so rather than plotting a mean of three graphs.

### 5.2 Cells (wave 9, group `class`): the graph class as a knob

DGP `graph ∈ {no2c, dag}` with the thesis weights, `--p 10 20 --reps 25 --select search`, n ∈ {10³, 10⁴},
both C, the same four estimators as E3. Cell names `direct_<estimator>_<C>_<graph>_n<n>` (`random` is wave 1;
`CELL` regex: an optional `_(no2c|dag)` group before `_n`, column `graph`). 2 × 4 × 2 × 2 = 32 cells of 800
datasets; shards as wave 1. Runner flag `--graph`.

### 5.3 Analysis (local)

`graph_class.py`: the same tables and `gain_by_class.png` with the three generated classes side by side,
within k. Add the share of graphs with `identified_min_eig > 0` per class (should be ≈ 1 for `dag` and
`no2c`). Reading to check: the dense-start gain is largest for `dag` / `no2c` and smallest with 2-cycles, and
the sparse → dense deficit is present in every class (the lock-in does not need 2-cycles).

Also record, per graph, the number of true edges inside 2-cycles that each estimator recovers *as a 2-cycle*
(both directions); `orientation_breakdown` already has the hedged / reversed counts, add "both directions of
a true 2-cycle recovered".

---

## 6. Order of work

1. §1.5 `graphs.py` + tests; §5.1 columns in `campaign.py` + `graph_class.py` on the stored campaign →
   first numbers in the memo. (Laptop, analysis only.)
2. §4.1 `campaign_edges.csv` + `betamin.py` on the stored campaign. (Laptop, analysis only.)
3. §1.1 `exact_fits.py` + tests, including the 11-of-100 regression (the one laptop-scale computation).
4. §1.2, §1.3, §1.4 library changes + tests; runner flags; wave 9 in `cluster/submit_campaign.sh` with the
   four groups `starts`, `threshold`, `betamin`, `class` (and `starts-loglik`) selectable with `--only`, as
   wave 5 does; `CELL` regex and the new columns in `campaign.py`; tests for the submit script
   (`tests/test_campaign_submit.py`) and the analysis (`tests/test_campaign_analysis.py`).
5. **Rehearsal on the laptop**, one shard per new cell type, `--reps 1`, through `cluster/local/sbatch`
   (`bash cluster/submit_campaign.sh --wave 9 --dry-run` first). Check that every cell runs end to end,
   that `campaign.py` reads its shards, and record seconds per dataset per arm. Set the shard counts and
   time limits of wave 9 from that (task ≤ 6 h at 2 × laptop speed). Nothing larger runs on the laptop.
6. Write `cluster/plan_131026.txt` (§8), commit, push. Then the cluster steps of §8.
7. While wave 9 runs: finish `starts.py`, `betamin.py`, `graph_class.py` and the plots against the rehearsal
   shards, so that the analysis is ready when the results come home.
8. When the cells are complete: `rsync` (block 9), `python simulations/diagnostics/campaign.py`, the three
   analysis scripts, the figures, the memo `next_steps/131026/next_steps_131026.md`.

Report after steps 1–2, before step 3: the stored-data answers to E3/E4 may already settle part of the
question and change which cells are worth submitting.

## 7. What not to do

- Do not change the seeds, the λ grid (`n_lambda=100`, `ratio=1e-4`), the standardisation, or `C` handling.
- Do not re-run existing cells; everything new pairs with wave 1 through `(p, k, c_choice, rep)`.
- Do not run anything beyond `--reps 1` rehearsals and the 11-of-100 regression on the laptop.
- Do not interrupt wave 8 on LRZ (`cluster/plan_101026.txt`); the new plan file carries its remaining lines.
- Do not write figures from the shard runners, and do not add plotting dependencies to `src/gclm`.
- Do not edit `VERDICTS.md`, `S4_campaign.md` or the slides; put results in the dated memo.

## 8. Cluster recipe

`cluster/plan_131026.txt`, one feeder for everything; it carries the remaining lines of `plan_101026.txt`
first (lines whose cells are submitted or complete are skipped), then wave 9 in the order of the questions.
The analysis-only steps (E3 §4.1, E4 §5.1) decide whether the `betamin` and `class` lines stay in:

```
# remaining lines of plan_101026.txt (wave 8 and its fills), verbatim, first
...
# wave 9 (plan_mechanism_131026.md): E2, then E1, then E3 and E4
--wave 9 --only threshold
--wave 9 --only starts --n 1e4
--wave 9 --only starts --n 1000
--wave 9 --only starts --n inf
--wave 9 --only betamin --n 1e4
--wave 9 --only betamin --n 1000
--wave 9 --only class --n 1e4
--wave 9 --only class --n 1000
--wave 9 --fill --time 24:00:00
```

On LRZ, after `git pull` (the running wave 8 tasks keep their code; the queued ones accept the same arguments):

```bash
cd ~/repo && git pull
bash cluster/submit_campaign.sh --wave 9 --dry-run | tail -3           # cells and tasks, nothing submitted
pgrep -fl feed_queue.sh; ssh cm4login1 pgrep -fl feed_queue.sh; ssh cm4login2 pgrep -fl feed_queue.sh
ssh cm4login1 pkill -f feed_queue.sh; ssh cm4login2 pkill -f feed_queue.sh   # only the one that runs
nohup bash cluster/feed_queue.sh cluster/plan_131026.txt > logs/feed.out 2>&1 &
sleep 5 && tail -3 logs/feed_queue.log
```

Progress: `bash cluster/submit_campaign.sh --wave 9 --status | grep -v complete` (empty = done),
`tail -5 logs/feed_queue.log`, `squeue -M serial -u $USER -h -r | wc -l`. When a group is complete, bring it
home with the `rsync` lines of block 9 and run `campaign.py`; the groups are independent, so E2 can be
analysed while E1 still runs.

Put the measured per-dataset times, the final shard counts and the CPU-hour estimate per group in the memo
next to these commands, as `next_steps_091026.md` §2–§4 did for waves 7 and 8.
