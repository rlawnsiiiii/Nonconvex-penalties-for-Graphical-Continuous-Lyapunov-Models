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

Each experiment's section opens with its motivation (§2.0, §3.0, §4.0, §5.0; added 13 October): where
the question comes from in the campaign and the theory note, what each arm isolates, and what each
possible outcome would mean for the thesis.

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
    # This is the "BP" solution of claude_docs/dense_start_theory_081026.md §2 / §4 (the count in §4).
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

### 2.0 Motivation

**Where the question comes from.** The campaign's positive result is a statement about a *start*, not
about a penalty: MCP and SCAD run dense → sparse from the lasso's dense end beat the lasso by up to +0.15 in
oracle $F_1$ (S4 §0, item 2), the adaptive lasso built on the same dense end does as well or better (item 3),
and on the log-likelihood loss the same estimators win only once their start is the lasso's dense end
(wave 7) and not when it is Varando & Hansen's exact fit $-\tfrac12 C\hat\Sigma^{-1}$ (wave 3; S4 §6,
"the start matters, not the loss"). So the estimator that works is "a dense start, then pruning", and the
one dense start that has been tried in both losses is the lasso at $\lambda_{\min} = 10^{-4}\lambda_{\max}$.
The theory note (`claude_docs/dense_start_theory_081026.md`, §3–§4) says what that start is: up to
$O(\lambda_{\min})$ the minimum-$\ell_1$ exact fit of the Lyapunov equation, which (Lemma 1) has at most
$p(p-1)/2$ nonzero off-diagonal entries with independent design columns, and which under the BP condition
(the true $M^*$ is the unique minimum-$\ell_1$ exact fit; 11 to 15 of 100 graphs at $p = 10$) contains every
true edge and gives the adaptive lasso the oracle property (Theorem 5). Three properties of this start are
therefore candidates for *why* it works, and they are confounded in the campaign because the lasso's dense
end has all three at once:

1. it is an **exact fit** of the sample Lyapunov equation, i.e. it sits on the fibre
   $\{M : M\hat\Sigma + \hat\Sigma M^\top + C = 0\}$ on which the direct loss is zero;
2. it is the **$\ell_1$-smallest** point of that fibre: a vertex, sparse (saturated at $p(p-1)/2$ off-diagonal
   entries in two thirds of the $p = 10$ graphs), keeping *both* directions of an uncertain pair ("hedging",
   S3a) rather than committing to one, as the sparse → dense path does at its first entry (a coin flip; the
   lock-in of `next_steps/051026/orientation_lock_in.md`);
3. it is **oriented**: on the population scale the dense end contains every true edge whenever the BP
   condition holds, and in the campaign's graphs it contains the true direction of most true edges even when
   that condition fails (the "dense-start content" counts of the 7 October note).

E1 separates these by changing the start and keeping everything else fixed: the data sets, the two pruners
(MCP dense → sparse and the adaptive lasso), the grid, the selection by the score and the search.

**Why these starts.**

- `lasso` (wave 1's `MCP-up` and `adaptive` cells; nothing new runs): the reference, with all three
  properties. It is also an *approximate* minimum-$\ell_1$ fit (FISTA at $\lambda_{\min}$, tolerance
  $10^{-8}$), so the difference to `l1` measures how much the approximation costs.
- `l1`, the exact minimum-$\ell_1$ fit from the LP: properties 1–3 exactly, with a vertex solution whose
  columns are independent (Lemma 1), which removes the question whether FISTA returns the sparse minimiser
  (Lemma 3's caveat). If `l1` ≈ `lasso` the dense end is what the theory says it is; if `l1` is clearly
  better, the thesis should use the LP as the pilot (it costs well under a second per data set).
- `l2`, the minimum-Frobenius exact fit: property 1 without property 2. It is on the same fibre, as close to
  the origin as an exact fit can be, fully dense (no exact zeros: the $\ell_2$ ball has no corners) and
  symmetric in its treatment of the two directions of a pair, so it hedges in a different way — both
  directions are present with graded sizes. It asks whether the *geometry* ($\ell_1$ vs $\ell_2$) of the
  start matters once the start is exact.
- `exact`, Varando & Hansen's $-\tfrac12 C\hat\Sigma^{-1}$: property 1 only, and a particular point of the
  fibre with no minimum-norm property at all (the one that solves $M\hat\Sigma = -\tfrac12 C$ entrywise).
  Wave 3 showed it fails as a start on the log-likelihood loss at finite $n$; E1 runs it on the direct loss
  with the two pruners that work from the lasso's dense end. If it fails here too, the failure is the
  start's, not the loss's; if it works on the direct loss, the loss-specific part of the wave 3 result needs
  its own explanation.
- `truth-0.05` and `truth-0.2`, the true $M^*$ plus Gaussian noise on every off-diagonal entry: property 3
  taken to its limit and the other two dropped (a dense matrix that is *not* an exact fit of the sample
  equation). This is a ceiling, not an estimator: it says how good the two pruners are when the start is
  right up to noise, and therefore whether the gap between the data-driven starts and the truth (the search
  from the truth ends 0.10 to 0.24 above every data-driven method, S4 §0 item 6) is in the start or in the
  pruner. Two noise levels because the pruners may tolerate small perturbations of a correctly oriented start
  (every true edge still the larger of its pair at $\tau = 0.05$, when the smallest true weights are around
  0.1) and not large ones ($\tau = 0.2$ swaps the order of a sizeable share of pairs).

**Why both pruners.** MCP dense → sparse and the adaptive lasso prune in different ways — a warm-started
nonconvex continuation against a convex reweighted problem — and in the campaign the adaptive lasso is
ahead in `aupr` and in the selected graph while MCP is slightly ahead in oracle $F_1$ at $p = 10$. If the
ranking of the starts is the same under both pruners, the result is about the start; if a start helps one
pruner and not the other, it is about the interaction, which the thesis should say.

**Readings** (§2.5 lists them as hypotheses): (a) `l1` ≈ `lasso` ≫ `l2` ≈ `exact` means the $\ell_1$
geometry of the start is the mechanism — the hedge and the vertex structure, as in the theory note; (b)
`l2` ≈ `l1` ≈ `lasso` ≫ `exact` or `l2` ≈ `l1` ≈ `exact` means any exact fit (or any minimum-norm exact fit)
does, and the message is simply "start on the fibre and prune"; (c) `truth-τ` far above every data start
means the pruners are not the bottleneck and the remaining gap to the search from the truth is in the start's
orientation, which points to a start that uses the score or the search rather than a norm. The start's own
diagnostics stored per data set (§2.2: its orientation content, $F_1$, norms, residual) make (c) checkable
graph by graph: `starts_by_pruner.png` plots the pruned estimator's $F_1$ against the start's orientation
recall, and under hypothesis 3 above the points of all starts should fall on one line.

**What E1 cannot settle.** It uses the campaign's data-generating process and $p \le 20$; whether the ranking
of the starts changes with the graph class or the weights is E3's and E4's question, and the log-likelihood
loss is only an optional second line (§2.6).

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

### 3.0 Motivation

**Where the question comes from.** The adaptive lasso is the best estimator of the campaign: convex,
deterministic, cheapest, and as good as or better than MCP dense → sparse on every metric (S4 §0 item 3;
§8). It has two ingredients: the dense end of the lasso path as the pilot, and a *reweighted* $\ell_1$
problem with weights $1/\lvert\hat M^{\mathrm{pilot}}_{ij}\rvert$ whose own path is then followed from
sparse to dense. The thesis needs to know which ingredient carries the gain, because the two lead to
different statements. If it is the pilot alone, then the simplest estimator that starts from the pilot — rank
its entries by size, keep the $k$ largest, refit by least squares, choose $k$ by the score — must do as
well, and "start dense, then prune" is the whole finding, with the reweighting an implementation detail. If
the reweighting matters, then *how* the pruning weighs the evidence matters, and the thesis should say where
it does.

**Why thresholding is the right comparison.** Thresholding the pilot is pruning in its purest form: the only
information used is the ranking of the pilot's entries by absolute size. The adaptive lasso uses the same
ranking as weights but lets the loss re-decide: an entry with a small pilot value can survive if the data
insist, and a pair whose two directions are both in the pilot is resolved by the loss *and* the weights
together rather than by size alone. For an orthogonal design the two select the same entries (the adaptive
lasso is then a thresholding rule itself, with shrunken values); in the general case the literature compares
them directly (van de Geer, Bühlmann & Zhou 2011, *The adaptive
and the thresholded Lasso for potentially misspecified models*; Zhou 2010 on the thresholded lasso — check
the statements) and finds similar selection guarantees under similar conditions, with thresholding the
simpler of the two. Here the design has a $p(p-1)/2$-dimensional null space and the pilot sits on its
boundary (Lemma 1 of the theory note), so the comparison is not covered by that literature and has to be
run.

**Why the same pilot in both arms.** The comparison is paired at the pilot level: the thresholded lasso and
the adaptive lasso of the same graph start from the identical matrix, so every difference between them is
the pruning step. (`adaptive_lasso_path` computes the pilot internally; `thresholded_lasso_path` takes it as
an argument and recomputes the same one when it is not given — §1.3.)

**Why two variants.** `threshold` refits the kept support by least squares on the direct loss, which is what
the score and the search need and what makes it an estimator of $M$. `threshold-norefit` keeps the pilot's
own values on the kept support: it is the pilot's ranking as a path, and its `aupr` is the area under the
precision–recall curve of ranking the entries of the dense end by size — the cleanest measure of how well the
dense end *orders* the edges, which the adaptive lasso's `aupr` (+0.07 to +0.14 over the lasso) is then
compared with. The two variants share their supports at every threshold, so their `max_f1` and `bic_f1`
coincide and only `aupr` differs; the second variant is for that one number.

**Readings.** A paired difference adaptive − threshold within ±0.01 in `max_f1` and `bic_f1` at every (p, C,
n) means the pruning is the whole story and the thesis can present the thresholded lasso as the minimal
version of the method, with the adaptive lasso as its smooth relative. A clear positive difference means the
reweighting matters; §3.3 then locates it with the edge-level columns of E3 and the class columns of E4:
is the gain on the pairs where the pilot keeps both directions (the thresholded estimator keeps the larger
pilot entry, the adaptive one may keep the other), on the weak edges (which thresholding removes first and
the reweighted loss can keep), or on graphs with 2-cycles (where the pilot's size ranking is not
informative about direction at all)? A negative difference — thresholding better — would mean the reweighted
loss re-admits false edges, which the false-positive bins of `campaign_false_positives.csv` would show.

**What E2 cannot settle.** It compares two pruners of one pilot; which pilot is E1's question. And at
$n = \infty$ with the rescaled $C$ both are expected to be near their ceiling, so the informative cells are
$n = 10^3$ and $10^4$.

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

### 4.0 Motivation

**Where the question comes from.** Every selection guarantee for a lasso-type estimator has a minimum-signal
("beta-min") condition: the nonzero coefficients must exceed a multiple of the noise level for the estimator
to find them. In the theory note the condition appears concretely: the population adaptive-lasso path misses
the true support in 1 of 15 BP graphs at $p = 10$ and 3 of 7 at $p = 20$, and every miss has a true entry of
size 0.003 to 0.008 that the smallest $\lambda$ of the grid already removes (`dense_start_theory_081026.md`
§4). The campaign's weights are $N(0, 1)$ draws (Dettling's generator), so about 8 % of the true edges have
$\lvert w\rvert < 0.1$ and 20 % have $\lvert w\rvert < 0.25$: a fixed share of edges that no method at these
$n$ can be expected to find, and that caps the $F_1$ of every estimator alike. The question for the mechanism
is whether the *gain* of the dense-start estimators over the lasso lives on the strong edges — where the
problem is to orient an edge whose presence is clear — or on the weak ones, where the problem is to detect
it at all.

**Why this separates the two mechanisms.** The campaign's explanation of the gain is about direction: the
lasso hedges, keeping both directions of a pair, and the pruning resolves the hedge in favour of the right
one (S3a; `docs/DENSE_START.md` §7.3); the sparse → dense path's deficit is also about direction: it commits
to one direction at the first entry and MCP freezes it (lock-in). Both mechanisms act on edges strong enough
to be detected; neither can create a weak edge. So the prediction is sharp: recall curves against
$\lvert w\rvert$ should show the dense-start estimators separated from the lasso on the strong edges (where
the lasso's hedging costs it a false positive per pair, hence precision, and the sparse → dense path's lock-in
costs it the true direction, hence recall) and tied with it on the weak edges, where nobody finds the edge.
If instead the gain sits on the weak edges, the mechanism is detection, not direction — a different story,
closer to the usual "nonconvex penalties have less bias" argument, which the campaign's other findings
(the adaptive lasso matching MCP) speak against.

**Why the analysis comes first.** Every stored shard has $M^*$ and the three graphs (best-$F_1$, selected,
searched), so the recall curve by weight bin can be read off the campaign without a single new run
(`campaign.py --edges`, §4.1, done 13 October). That costs nothing and may already answer the question;
the new cells are then a confirmation with the weight distribution as an *exogenous* knob rather than a
split of one distribution into bins, which conditions on the realised weights.

**Why the knob is `weights="uniform"` with `w_min`.** Replacing the $N(0, 1)$ law by $\lvert w\rvert \sim
U[w_{\min}, 1]$ with a random sign fixes a lower bound on every true weight and changes nothing else in the
generator: the same supports (the Bernoulli draws are unchanged), the same diagonal rule, the same
stability. Four levels — $w_{\min} = 0$ (a uniform law without a gap, the reference for the knob itself),
0.1, 0.25 and 0.5 — give a curve rather than a before/after. The $w_{\min} = 0$ arm is needed because the
uniform law differs from the thesis DGP in shape (no heavy tails, no mass near zero beyond the uniform
density), so the comparison with wave 1 has to go through it, and the figures carry one wave 1 line as the
reference. The independent study of 2 October ran one such cell (`|w| ~ U[0.5, 1]`, $n = 10^4$) and found
much larger gains than on the thesis weights, which is the observation this experiment turns into a curve.

**Why these estimators.** `lasso` (reference), `MCP` sparse → dense (the lock-in: its deficit should *not*
shrink with $w_{\min}$, because the first entry's direction is a coin flip whatever the weight), `MCP-up`
and `adaptive` (the two pruners: their gain should grow with $w_{\min}$, because a strong true edge whose
reverse the lasso hedges is exactly the case the pruning resolves).

**Readings.** Gains growing with $w_{\min}$ for the dense-start estimators and a flat sparse → dense deficit
confirm the direction mechanism. A gain that does not grow with $w_{\min}$ while the recall of the strong
edges is already near one would mean the remaining errors are false positives, and the figures should then
be read together with `campaign_false_positives.csv`. If the sparse → dense deficit shrinks with $w_{\min}$,
the lock-in is partly a signal-strength effect, and the thesis should soften that claim.

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

### 5.0 Motivation

**Where the question comes from.** A graphical continuous Lyapunov model is identified from its covariance
only up to what the Lyapunov equation can tell apart, and the equation is blind in one well-defined place:
an isolated pair with both directions, $i \leftrightarrow j$, has two off-diagonal unknowns for the one
off-diagonal equation of the pair, so no estimator can recover both directions from $\Sigma$. The campaign's
generator draws every ordered pair independently with probability $d = k/p$, so a 2-cycle appears on a pair
with probability $d^2$, and at $k = 4$, $p = 10$ almost every graph has one. S3a showed that the lasso
keeps both directions of such pairs (and hedges on single edges) while MCP keeps one; the orientation
breakdown stored with every selected graph (`hedged`, `reversed`, `both`, `half`) counts exactly these cases.
`gclm.graphs.identified_on_support` turns the identifiability question into a number per graph: the smallest
eigenvalue of the design's Gram matrix restricted to the true support, positive iff $M^*$ is the only drift
matrix with that support that fits the population covariance.

**Why the gain may be concentrated in a class.** The direction mechanism of E3 and of the campaign's
explanation can only work where direction *is* identified. In a DAG every edge's direction is identified on
the true support; in a cyclic graph without 2-cycles it is as well (the restricted design has full column
rank); with a 2-cycle the two entries of the pair are confounded, so the dense end contains both, no pruning
can tell them apart, and the pruned estimator's choice between them is arbitrary — a case in which the lasso's
hedge (keep both) is *right* and the pruning is wrong. So the prediction is that the dense-start gain is
largest on DAGs and on cyclic graphs without 2-cycles and smallest on graphs with 2-cycles, and that the
sparse → dense deficit, which comes from the lock-in at the first entry rather than from identifiability, is
present in every class. If the gain turned out to be *largest* with 2-cycles, the explanation would have to
change: the pruning would then be helping by removing one direction of a 2-cycle at random, gaining
precision at the cost of a true edge, and the skeleton metrics would show it.

**Why within $k$.** At $p = 10$ the class is almost a function of the density: $k = 1$ graphs rarely have a
2-cycle (45 pairs, each with probability $d^2 = 0.01$: 0.45 expected), $k = 4$ graphs almost always do
(45 pairs at $d^2 = 0.16$: about 7). The campaign
already knows that the gain depends on density (verdict 12 of `VERDICTS.md`), so a comparison of classes
pooled over $k$ would mostly repeat that. Comparing within $k$ holds the density fixed and lets the class
vary through the randomness of the draw; the price is that some (class, $k$) cells are nearly empty at
$p = 10$, which is why the figure prints the number of graphs per cell and why the generated classes of
§5.2 are needed for a clean answer.

**Why the analysis comes first, and why generated classes after.** As for E3, the stored shards already
hold $M^*$ and the three graphs per data set, so the class columns and the within-$k$ gains cost nothing
(`campaign.py`, `graph_class.py`, done 13 October). But in the stored campaign the class is realised, not
assigned: a $k = 2$ graph with a 2-cycle differs from one without in more than the 2-cycle (it has, on
average, one more edge among fewer pairs). The generated classes `no2c` (drop one direction of every
2-cycle at random) and `dag` (orient every edge along a random permutation, at a density that keeps the
expected number of edges) make the class exogenous and leave everything else of the generator alone: same
weights, same diagonal rule, same $C$. `dag` is also the class in which the direction of *every* edge is
identified and the one most of the graphical-model literature studies, so the thesis can state the gain
there directly; `no2c` sits between `dag` and the thesis generator and isolates the 2-cycle question from the
question of long cycles. The share of graphs with `identified_min_eig > 0` per class is the check that the
classes do what they are meant to (it should be about 1 for `dag` and `no2c`).

**Why the same four estimators as E3.** The lasso as the reference; the sparse → dense MCP, whose deficit
is predicted to be class-independent; the two pruners, whose gain is predicted to be class-dependent. If
the four behave as predicted, E3 and E4 together give the thesis a mechanism with two testable parts:
the dense start wins by orienting strong, identifiable edges, and the sparse → dense path loses by committing
to a direction before the data can tell.

**Readings.** Gains ordered `dag` ≥ `no2c` > `random` (with 2-cycles) for the pruners, a deficit of about
the same size in every class for `MCP` sparse → dense, and the orientation breakdown of the 2-cycle graphs
showing the pruners at `half` where the lasso is at `both`, confirm the picture. An equal gain in every
class would mean identifiability is not the limiting factor at these sizes — the gain would then come from
the hedge on *single* edges alone, which is consistent with the theory note's finding that the dense end
contains the true direction of most true edges even where the BP condition fails. The within-$k$ requirement
applies to the generated classes too, since `dag` and `no2c` change the edge count per pair as well.

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
