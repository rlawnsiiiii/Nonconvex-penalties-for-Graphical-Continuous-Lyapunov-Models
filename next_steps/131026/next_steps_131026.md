# Mechanism experiments: what was built, and the first numbers (13–14 October 2026)

*Plan: [`plan_mechanism_131026.md`](plan_mechanism_131026.md). This memo follows its order of work
(§6 there) and grows with it. Status: steps 1 and 2 (the analyses on the stored campaign) are done
and reported below; steps 3 to 8 are pending.*

## 1. Built on 13 October (steps 1 and 2)

| what | where | test |
|---|---|---|
| graph-class diagnostics: `is_dag`, `two_cycles`, `frac_edges_in_two_cycles`, `graph_class`, `identified_on_support` (smallest eigenvalue of the design's Gram matrix on the true support plus the diagonal, from `Σ* = Σ(M*, C*)`; positive iff `M*` is the only drift matrix with that support fitting the population covariance), `design_columns` (the columns of `A(Σ)` without forming it) | `src/gclm/graphs.py` | `tests/test_graphs.py` |
| the class of every graph in the per-dataset table: columns `graph_class`, `n_two_cycles`, `frac_edges_2c`, `identified_min_eig`, computed once per graph from the stored `M*` and the regenerated `C*` (the regenerated `M*` is checked against the stored one) | `simulations/diagnostics/campaign.py` (`graph_diagnostics`) | `tests/test_campaign_analysis.py::test_graph_class_columns_and_edge_tables` |
| the edge tables: `campaign.py --edges` writes `campaign_edges.csv` (one row per true edge of the 48 wave 1 cells at p = 10, 20: 1,341,744 edges, 178 MB) and `campaign_false_positives.csv` (per graph and estimate, the false positives of the best-F1 estimate and of the searched refit by `|M̂_ij|` bin; 76,800 rows) | `campaign.py` (`edge_rows`, `edge_tables`) | same test |
| E4 tables: paired gains over the lasso per (p, C, n, k, graph class) and estimator, with the number of graphs and the share identified on the true support → `campaign_graph_class.csv`; figure `gain_by_class_p<p>.png` (and `_bic_f1`) | `simulations/diagnostics/graph_class.py`, `plot_graph_class.py` | same test |
| E3 tables: recall of the true edges by weight bin per (p, C, n, estimator), all edges and split by 2-cycle membership, in the best-F1 graph, the selected graph and after the search → `campaign_betamin.csv`; figures `recall_by_weight.png`, `recall_by_weight_search.png` | `simulations/diagnostics/betamin.py`, `plot_betamin.py` | same test |

Runtime on the laptop: `campaign.py --edges` 17 minutes (the class columns add about a minute);
the two analyses a few seconds. The figures follow the campaign's style (`plot_campaign.py`).

### 1.1 Step 3, built on 14 October: the exact fits (plan §1.1)

| what | where | test |
|---|---|---|
| the affine space of exact fits in skew coordinates: `skew_basis`, `skew_from_coordinates`, `exact_fit_jacobian`, `offdiag_indices`; the members `min_l2_exact_fit` (least squares), `min_l1_exact_fit` (LP, HiGHS at 1e-9; with `probes > 0` an exact uniqueness test, `L1Fit.face_extent`), `noisy_truth_start`; `exact_fit_residual` | `src/gclm/objective/exact_fits.py` | `tests/test_exact_fits.py` (12 tests, 4 s) |
| `standardized_drift(m, scale)` = D⁻¹ M D, the truth on the correlation scale (the value the dense-start experiments compare with) | `src/gclm/data/simulate.py` | `tests/test_dgp.py::test_standardized_drift_solves_the_equation_of_the_standardized_data` |

Documented in `docs/DENSE_START.md` §3.1. The seeds match the campaign exactly (`S1Config.seed`,
`default_rng([seed, p, k, c_index, rep])`), so the regression is exact, not a range:

- **BP condition, 11 of 100** at p = 10 (7 at k = 1, 4 at k = 2, none at k = 3, 4): the 7 October count
  reproduced. The standardised truth attains the minimal ℓ1 norm in 17 graphs; in the other 6 the optimal
  face is flat (the direction of an isolated edge is free) and the LP returns one vertex at random: HiGHS
  1.15 returns the truth in 14, the 8 October note counted 15. The separations are clean: the face extent
  is exactly 0 or above 0.047, the ℓ1 gap of the unique-but-wrong fits is above 0.033, the deviation of
  the returned fit from the truth is below 1e-6 or above 0.03.
- The uniqueness test had to change from the plan's wording. Re-solving with the norm held at its
  minimum up to a slack flagged generic sample instances as non-unique (extent 2.5e-4 at p = 8, n = 1000):
  the slack is amplified along weakly sharp minima. The implemented test is exact: the directional
  derivative at the solution is a polyhedral function of the step, and random linear objectives over the
  cone where it is nonpositive are 0 iff the minimiser is unique (`docs/DENSE_START.md` §3.1).
- LP cost on the laptop: 0.2 s at p = 20, 1 s at p = 30, 10 s (`highs-ipm`) to 26 s (simplex) at p = 50.
  Nothing to change in `cluster/setup_env.sh`: `scipy >= 1.10` has HiGHS.
- The plan's pointer `claude_docs/.. main_claim_and_theory_071026.md §2.4 / §4` does not exist; the BP
  solution and the count live in `claude_docs/dense_start_theory_081026.md` §2 and §4 (fixed in the plan).

### 1.2 Step 4, built on 14 October: the library options, the runner, wave 9 (plan §1.2–1.4, §2–§5)

| what | where | test |
|---|---|---|
| `lasso_path(start=)` and `covloss_path(start=<matrix>)`: any matrix as the dense start of a dense → sparse path; the start used is returned (`LassoPath.start`); `start=` equal to the lasso's dense end reproduces the stored paths bit for bit | `src/gclm/solvers/path.py` | `tests/test_starts.py` (4) |
| `thresholded_lasso_path` / `ThresholdedLassoPath`: the lasso's dense end cut by size (100 thresholds: 0, 98 log-spaced, ∞), refitted by least squares (`refit="direct"`), by likelihood (`"loglik"`) or not at all (`"none"`); the same pilot as the adaptive lasso; entries above `PRESENT = 1e-8` count | `path.py`, `gclm.objective.exact_fits.PRESENT` | `tests/test_threshold.py` (5) |
| DGP knobs `weights="uniform"`, `w_min`, `graph="no2c"\|"dag"` in `sample_drift` / `draw_instance`; the default call consumes the generator exactly as before | `src/gclm/data/simulate.py` | `tests/test_dgp.py` (+3) |
| `S1Config`: `up_start` (None = the loss's default; `start_name` resolves it), `pilot`, `weights`, `w_min`, `graph` | `src/gclm/config.py` | runner tests |
| runner: `--up-start lasso\|exact\|l1\|l2\|truth-<τ>` on both losses, `--pilot` (same choices) for `adaptive` / `threshold` / `threshold-norefit`, `--method threshold[-norefit]`, `--weights`, `--w-min`, `--graph`; `start_*` diagnostics per data set (norms, residual, `start_nnz`, `start_conf`, `start_orient` = the dense-start content); refused combinations; config records the resolved start | `simulations/run_s1_shard.py` (+ `rescore_shard.py` rebuilds knob cells from their config) | `tests/test_campaign_runner.py` (+13) |
| analysis: `CELL` accepts a dot in the estimator, `_w<wmin>` and `_no2c\|_dag`; columns `weights`, `w_min`, `graph`, `dgp`; `means_table` / `paired_table` group by `dgp` (a knob cell pairs with the lasso drawn with the same knob); `start_*` columns via `graph_metrics`; `graph_diagnostics` regenerates knob graphs from their config | `simulations/diagnostics/campaign.py` | `tests/test_wave9_analysis.py` (3; `test_campaign_analysis.py` unchanged and passing) |
| wave 9 in five groups, `--group`; a knob cell or a loglik start asked for at n = ∞ is reported and skipped; shard counts provisional (wave 1's per estimator) | `cluster/submit_campaign.sh` | `tests/test_campaign_submit.py` (+1, wave 9 in the runner-acceptance test) |

Documented in `docs/DENSE_START.md` §5.1–5.2, `docs/REPRODUCTION.md` (wave table, commands) and
`simulations/S1_reproduction.md` §3.1 (the knobs). Deviations from the plan's wording, all small:

- `thresholded_lasso_path` has no `loss` argument (the plan's `loss="direct"`): the pilot is always the
  direct-loss lasso's dense end, as the adaptive lasso's; `refit` picks the refit. E2 is direct-loss only.
- `S1Config.up_start` defaults to `None` rather than `"exact"`, so that the direct loss's default
  (`lasso`) and the covariance losses' (`exact`) are both the old behaviour; the shard records the resolved
  value (`up_start` in `config_json`), new shards of old-style direct cells therefore say `lasso` where
  older ones said `exact` without meaning it.
- The pilot's and the start's "present" rule is one constant, `PRESENT = 1e-8`: the l1 fit's structural
  zeros come back from the LP as dust around 1e-10 and must not count as edges of the pilot.
- The l1 start reproduces the lasso-started MCP path on the test instance (p = 8, n = 1000): the theory's
  prediction, and a first hint for E1's reading (a).

Step 7, also built on 14 October (the analyses, ready before the cells come home; tested on the rehearsal
shards and on a tiny campaign, `tests/test_wave9_analysis.py`):

| what | where | output |
|---|---|---|
| E1: per (p, C, n, pruner, start) the mean and the paired gains over the lasso and over the pruner's lasso-start arm; what each start contains (F1 of its support, the shares of true single edges with the true direction alone / both / only the reverse / neither, orientation recall, norms, residual); E2: the five estimators against the thresholded lasso with paired differences | `simulations/diagnostics/starts.py` | `campaign_starts.csv`, `campaign_start_content.csv`, `campaign_threshold.csv` |
| figures: `starts_bars[_<metric>].png`, `starts_by_pruner[_n<n>].png` (pruned F1 against the start's orientation recall), `threshold_vs_adaptive[_<metric>].png` (histograms of the paired difference per graph) | `plot_starts.py` | `figures/` |
| E4: the generated classes (`no2c`, `dag`) as their own generator in the within-k table, each against the lasso of the same generator; `table_by_generator` pools the classes; figure `gain_by_generator_p<p>[_bic_f1].png` next to `gain_by_class` | `graph_class.py`, `plot_graph_class.py` | `campaign_graph_class.csv` (column `generator`), `campaign_generator.csv` |
| E3: `max_f1`, `aupr`, `bic_f1`, `search_f1` against `w_min` per estimator with the paired gain over the lasso drawn with the same weights, wave 1 as the `normal` reference; figure `gain_by_wmin[_<metric>].png` | `betamin.py` (`wmin_table`), `plot_betamin.py` | `campaign_wmin.csv` |

### 1.3 Step 5, the rehearsal (14 October): every cell type runs end to end; what wave 9 costs

One cell per type through `cluster/local/sbatch` (`--shards 4`, `LOCAL_EXTRA="--p 10 --reps 1"`, shard 0 = the
four graphs k = 1..4 at C_ID, rep 0, p = 10, n = 1000), 11 cells, 4 minutes wall time with three processes,
no errors; `campaign.py` reads the shards (incomplete cells, as expected) and every analysis and figure
script of §1.2 runs on them. Laptop seconds for the four graphs, against wave 1's LRZ seconds on the same
four graphs (LRZ ≈ 2 × laptop, so the ratio in the last column is 2 × laptop / LRZ):

| rehearsed cell | laptop s (4 graphs) | wave 1 reference (LRZ s, same graphs) | cost ratio |
|---|---|---|---|
| `direct_MCP-up-l2_Cresc` | 84 | MCP-up Cresc 172 | 1.0 |
| `direct_MCP-up-truth0.2_C2I` | 95 | MCP-up C2I 100 | 1.9 |
| `direct_adaptive-l1_Cresc` | 2.4 | adaptive Cresc 19.5 | 0.25 (no lasso path needed) |
| `direct_threshold_Cresc`, `-norefit` | 7.2, 7.3 | adaptive Cresc 19.5 | 0.75 |
| `direct_lasso_Cresc_w025` | 5.1 | lasso Cresc 16.2 | 0.6 |
| `direct_MCP-up_Cresc_w025` | 32 | MCP-up Cresc 172 | 0.4 |
| `direct_MCP_Cresc_dag` | 90 (49 at k = 4) | MCP Cresc 50.6 | 3.5 |
| `direct_adaptive_Cresc_no2c` | 12 | adaptive Cresc 19.5 | 1.2 |
| `loglik_MCP-up-l1_C2I` | 28 | — | about 3 CPU-h per cell (p = 10) |
| `loglik_adaptive-l2_Cresc` | 159 (100 at k = 4) | — | about 9 CPU-h per cell |

The exact-fit and truth starts are fully dense (90 entries present) and the MCP solver crawls through the
flat valley of the dense end at a few λ (the `converged` flags say so); the l1 start has 45 entries and the
adaptive lasso from it is four times cheaper than from the lasso's dense end. On the four graphs the l1 start
contains the true direction of 37 % of the true single edges alone, both directions of 40 %, only the reverse
of 16 % and neither of 8 % (`campaign_start_content.csv`).

**Cost.** Wave 1's measured CPU-hours per cell (`campaign_means.csv`, LRZ, 800 graphs) are, at n = 1000,
about ten times those at n = 10⁴ and n = ∞ (MCP-up: 198 and 167 CPU-h at n = 1000 against 15 to 20): the
direct-loss paths converge slowly on the noisier sample covariance. Scaling them with the ratios above
(unmeasured arms at 1.0) gives, per group:

| group | n = 10⁴ | n = ∞ | n = 1000 |
|---|---|---|---|
| threshold (12 cells) | 9 | 13 | 52 |
| starts (60 cells) | 260 | 270 | 2,600 |
| class (32 cells) | 230 | — | 1,900 |
| betamin (64 cells) | 180 | — | 1,400 |
| starts-loglik (optional, 40 cells) | ~150 | — | ~150 |

about 1,000 CPU-h for everything at n = 10⁴ and n = ∞ and 5,900 CPU-h for the n = 1000 groups, 6,900 in
all — the whole of `plan_101026.txt` again. The plan's §2.4 expected "four times wave 1" for E1, which holds
(wave 1 ≈ 960 CPU-h; E1 ≈ 3,100); E3 and E4 add as much again. Shards: wave 1's counts (the script's
defaults) keep every task under 6 h at n = 10⁴ and n = ∞; at n = 1000 they would not (MCP-up from the
truth: 24 h per task at 16 shards; MCP sparse → dense on DAGs: 33 h at 8), so `cluster/plan_131026.txt`
submits the n = 1000 groups with `--shards 32 --time 24:00:00` (betamin: 16), which keeps the longest task
at about 12 h. The plan file orders the cheap groups first (E2, then E1, E4, E3 at n = 10⁴ and n = ∞), and
the n = 1000 groups after a `--fill`, with the recommendation of §4 (class as planned, betamin the group to
cut first; three weight levels are `--only _w0$ _w025 _w05`, the `$` anchoring the pattern at the end of
the name, added to `submit_campaign.sh` for this). Whether to run the n = 1000 groups at all is a decision
for when the n = 10⁴ results are in. The estimates rest on four graphs at p = 10; the first complete cells
at n = 10⁴ will say how good they are.

## 2. E4 on the stored campaign: the gain by graph class, within k

Paired gain in oracle F1 over the lasso with the same C, by the class of the true graph, within k
(wave 1, 400 graphs per (p, C, n); `campaign_graph_class.csv`; standard errors 0.01 to 0.03 for the
groups with 37 graphs or more). "cyclic" = a directed cycle but no 2-cycle; "2-cycles" = at least
one pair with both directions.

| p, C, n = 10⁴ | k | class (graphs) | MCP sparse → dense | MCP dense → sparse | adaptive lasso |
|---|---|---|---|---|---|
| 10, C = 2I | 1 | DAG (52) / cyclic (11) / 2-cycles (37) | −0.11 / −0.12 / −0.13 | +0.06 / +0.07 / +0.02 | +0.07 / +0.06 / +0.03 |
| 10, C = 2I | 2 | cyclic (11) / 2-cycles (87) | −0.11 / −0.13 | +0.10 / +0.02 | +0.10 / +0.03 |
| 10, rescaled C | 1 | DAG / cyclic / 2-cycles | −0.13 / −0.03 / −0.12 | +0.12 / +0.09 / +0.09 | +0.13 / +0.09 / +0.08 |
| 10, rescaled C | 2 | cyclic / 2-cycles | −0.08 / −0.08 | +0.19 / +0.06 | +0.15 / +0.06 |
| 20, C = 2I | 1 | DAG (47) / cyclic (15) / 2-cycles (38) | −0.13 / −0.11 / −0.14 | +0.12 / +0.12 / +0.05 | +0.13 / +0.12 / +0.07 |
| 20, C = 2I | 2 | cyclic (12) / 2-cycles (85) | −0.08 / −0.07 | +0.09 / +0.09 | +0.07 / +0.10 |
| 20, rescaled C | 1 | DAG / cyclic / 2-cycles | −0.07 / −0.07 / −0.10 | +0.20 / +0.17 / +0.11 | +0.22 / +0.17 / +0.13 |
| 20, rescaled C | 2 | cyclic / 2-cycles | −0.04 / −0.04 | +0.17 / +0.15 | +0.18 / +0.15 |

At k = 3 and 4 nearly every graph has 2-cycles (96 to 100 of 100), so there is no within-k
comparison; the gain there falls with k and turns negative at p = 10, k = 4 (−0.03 / −0.04 with
either C), as verdict 12 says.

1. **At k = 1 the dense-start gain is class-dependent, in the predicted direction.** DAGs and cyclic
   graphs without 2-cycles gain the same; graphs with a 2-cycle gain half as much or less: at p = 20,
   n = 10⁴, +0.12 / +0.12 / +0.05 (MCP dense → sparse, C = 2I) and +0.20 / +0.17 / +0.11 (rescaled
   C); at p = 10, +0.06 / +0.07 / +0.02 and +0.12 / +0.09 / +0.09. At n = 10³ the same ordering
   (p = 20, rescaled C: +0.15 / +0.12 / +0.09 for MCP, +0.20 / +0.17 / +0.12 for the adaptive lasso).
2. **At k = 2 the picture splits by p.** At p = 10 the gap persists (cyclic +0.10 / +0.19 against
   2-cycles +0.02 / +0.06); at p = 20 it is gone (+0.09 / +0.17 against +0.09 / +0.15). One 2-cycle
   among 38 edges at p = 20 costs the pruners little; among 19 edges at p = 10 it costs a lot. The
   effect is about the share of pairs that are 2-cycles, not their presence.
3. **The sparse → dense deficit is class-independent within k**, as predicted: p = 10, C = 2I,
   n = 10⁴, k = 1: −0.11 / −0.12 / −0.13; p = 20: −0.13 / −0.11 / −0.14 (rescaled C: −0.07 / −0.07 /
   −0.10). The lock-in at the first entry does not need 2-cycles.
4. **Identification on the true support is almost never lost:** 95 to 100 % of the graphs in every
   class have `identified_min_eig > 0` (the isolated pair of the plan's motivation is rare; a 2-cycle
   inside a larger graph is identified through its other edges). So the class effect is not a pure
   identifiability effect; it is about where the lasso's hedges sit and what the pruning does with
   them. The within-class orientation counts (`hedged`, `both`, `half`) are the next thing to read.
5. **The within-k groups are small** at k = 1 (11 to 15 cyclic graphs per cell; 2 to 3 DAGs at
   k = 2) and the class is realised, not assigned: a k = 2 graph with a 2-cycle has, on average, more
   edges among fewer pairs. The generated classes of the plan (§5.2, `no2c` and `dag`) are needed
   for a clean statement; the stored data say the effect is there and worth the cells.

## 3. E3 on the stored campaign: recall by weight, and where the gain comes from

Recall of the true edges in the best-F1 graph of the path (the oracle λ), by |w| (the generator's
N(0, 1) weight), p = 20, n = 10⁴, all 400 graphs (`campaign_betamin.csv`):

| |w| (edges) | lasso | MCP sparse → dense | MCP dense → sparse | adaptive lasso |
|---|---|---|---|---|
| C = 2I: 0–0.1 (1483) | 0.19 | 0.14 | 0.16 | 0.13 |
| 0.1–0.25 (2339) | 0.39 | 0.27 | 0.33 | 0.29 |
| 0.25–0.5 (3512) | 0.61 | 0.45 | 0.56 | 0.54 |
| 0.5–1 (5576) | 0.81 | 0.59 | 0.72 | 0.75 |
| 1–2 (5125) | 0.90 | 0.63 | 0.83 | 0.86 |
| ≥ 2 (905) | 0.96 | 0.61 | 0.91 | 0.94 |
| rescaled C: 0.5–1 / 1–2 / ≥ 2 | 0.81 / 0.90 / 0.95 | 0.61 / 0.66 / 0.67 | 0.76 / 0.85 / 0.92 | 0.75 / 0.87 / 0.95 |

False positives per graph in the same best-F1 graphs (47.4 true edges per graph), and how many of
them are tiny (`|M̂_ij| < 0.1`):

| p = 20, n = 10⁴ | lasso | MCP sparse → dense | MCP dense → sparse | adaptive lasso |
|---|---|---|---|---|
| C = 2I: false positives (of which tiny) | 32.1 (18.4) | 22.2 (8.6) | 17.8 (8.1) | 16.8 (4.6) |
| rescaled C: false positives (of which tiny) | 34.0 (17.0) | 20.0 (6.4) | 14.1 (5.8) | 11.8 (2.6) |

1. **The lasso has the highest recall in every weight bin**, at every p, C and n. The dense-start
   estimators recover *fewer* true edges than the lasso, by 0.02 to 0.07; the gap narrows with |w|
   and closes at |w| ≥ 2 with the rescaled C (0.945 against 0.947). Their F1 gain is therefore a
   gain in **precision**: at p = 20, n = 10⁴ the lasso's best graph carries 32 to 34 false positives
   against 12 to 18 for the pruned estimators, and the difference is almost entirely in tiny
   entries (17 to 18 of the lasso's false positives are below 0.1 in size, against 3 to 8). Those
   tiny entries are the hedges: the reverse direction of an edge the lasso also has right. The
   pruning removes them, and in doing so also loses a few true edges, mostly weak ones.
2. **The sparse → dense MCP's recall is flat beyond |w| = 0.5** — 0.59 / 0.63 / 0.61 for the three
   strongest bins at C = 2I, 0.61 / 0.66 / 0.67 with the rescaled C, and the same at n = ∞ (0.63 /
   0.67 / 0.67) — while every other estimator's recall rises to 0.9 and beyond. A strong true edge is
   missed as often as a medium one, and more data do not help: that is the lock-in of the first entry's
   direction, seen directly. (Its false positives are fewer than the lasso's but larger: 1.4 to 2.0
   per graph above 1 in size, the reversed edges.)
3. **After the search the lasso-based estimators coincide** (recall at |w| ≥ 2, C = 2I, n = 10⁴:
   0.93 / 0.91 / 0.94; false positives 31 / 31 / 31): the search equalises, as S4 §5 found; the
   sparse → dense MCP stays behind (0.66, with 43 false positives).
4. **Reading against the plan's prediction (§4.0).** The prediction "the dense-start estimators
   separate from the lasso on the strong edges and tie on the weak ones" was stated in recall and is
   wrong as stated: in recall the lasso is ahead everywhere. The mechanism it meant is confirmed in
   the precision column instead — the gain is the removal of hedges, which sit next to strong, correctly
   recovered edges — and the lock-in prediction for the sparse → dense path holds exactly. The
   beta-min knob (§4.2) therefore tests a sharper statement: with a larger minimum weight the hedge
   is easier to resolve (the true direction's entry is larger relative to its reverse), so the
   pruners' precision gain should grow and their recall loss shrink, while the sparse → dense recall
   stays flat.

## 4. What this means for the wave 9 cells

- **E4 `class` cells (32):** worth running as planned. The stored data show the class effect at k = 1
  on small, realised groups and a p-dependent picture at k = 2; the generated `no2c` and `dag` classes
  make it exogenous and give every k the three classes.
- **E3 `betamin` cells (64):** the qualitative answer is already in the stored data (the gain is
  precision on the hedges next to strong edges; the sparse → dense deficit is signal-independent).
  The knob is still informative for the thesis (a curve of the gain against the minimum weight), but
  it is the group to cut first if wave 9 is too large: p = 10 only, or `w_min ∈ {0, 0.25, 0.5}`.
- **E1 and E2:** unaffected by these findings; E2 should also report precision and recall per weight
  bin, since that is where the two pruners could differ.
- One more laptop-only analysis before submitting: the within-class orientation counts (`hedged`,
  `both`, `half`, `reversed`) of the selected graphs per estimator, to see whether the 2-cycle
  graphs lose their gain on the 2-cycles themselves (the pruners keeping `half` where the lasso keeps
  `both`) or elsewhere.

## 5. Next (steps 3 to 8 of the plan)

Steps 3 to 7 are done (§1.1–1.3; step 7's analyses are written and tested, they wait for data).
Step 6: `cluster/plan_131026.txt` is written; submitting is Joon's call: `git pull` on LRZ (the solver
change of 13 October is in it: do not pull while wave 8's n = 1000 tasks run, or accept that the
remaining tasks of a cell run on the new solver), stop the old feeder, start the new one on the new
plan file (plan §8). Step 8 when the cells come home: `rsync`, `campaign.py`, `starts.py`, `graph_class.py`,
`betamin.py`, the three plot scripts, and the results here.
