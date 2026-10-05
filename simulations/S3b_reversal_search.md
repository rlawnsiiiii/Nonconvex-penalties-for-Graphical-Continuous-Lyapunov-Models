# S3b: a BIC search with reversal moves, started from the lasso, MCP and SCAD paths, on all three losses

*Second part of S3 (`plan.md`: score-based search with a BIC-type penalty). Written 2 October 2026.
S3a (`S3a_bidirectional_edges.md`) showed that the direction entering the path first is close to a
coin flip for every penalty. The lasso later corrects it by keeping both directions; MCP freezes it.
Example 2 (`../next_steps/021026/next_steps_021026.md` §3) showed that the MCP objective prefers the
truth at large $n$, and that a reversal search on that objective reaches it while the continuation
path does not.*

**Status (3 October 2026): implemented and validated; phases 1–2 complete for the direct loss (the $p = 20$ run on Dettling's pipeline was stopped early). Phase 3, the covariance losses on random graphs, was not run. The follow-up is planned in `next_steps/031026/next_steps_031026.md`.**

## Summary (morning of 3 October)

![Overview](../runs/s3b_search/figures/overview_f1_vs_plain_lasso.png)

*Figure 1. Directed $F_1$ minus plain lasso at its BIC-selected $\lambda$ (paired means; direct loss,
BIC) for every setting run; the number of graphs is in brackets. Grey: Dettling's oracle-tuned
lasso. Right of the line is better than plain lasso. The last row was stopped early and covers
sparse graphs only (§9.4).*

1. **The four methods** (lasso / MCP / SCAD + greedy BIC search, and the pure greedy search of
   Améndola et al. 2020) are implemented, tested and validated. The exhaustive check on Example 2
   shows that greedy reaches the global BIC optimum from every penalty start (§9.0).
2. **On Dettling's Figure 5 pipeline** (standardised input, $C = 2I$ assumed), no search improves on
   the plain lasso at its BIC $\lambda$, and all stay below Dettling's oracle-tuned lasso (§9.2).
   - The reason is the score, not the search: on that pipeline the likelihood is misspecified, and
     BIC prefers wrong graphs in about 70 % of cases.
3. **On the raw scale** the search clearly helps (§9.3). Most where the model is exactly right
   (`C_ID`, true $C = 2I$): +0.14 to +0.27 over plain lasso, well above Dettling's oracle lasso.
   Less, but still positive, with random $C$, where $C = 2I$ is also misspecified.
   - **Lasso + search** gains +0.07 / +0.10 / +0.06 $F_1$ at $n = 10^3$ / $10^4$ / $\infty$ over plain
     lasso at its BIC $\lambda$, and reaches or beats Dettling's *oracle-tuned* lasso with a
     data-driven $\lambda$.
   - The **pure search** matches it at large $n$ and is weaker at $n = 10^3$.
   - **MCP and SCAD starts** are as good as the lasso start at $n = 10^3$, and worse at larger $n$,
     where their frozen reversals survive the search.
4. **On Example 2 at $n = 10^5$** every penalty-started search recovers the 5-cycle exactly (9 of 10
   datasets, on all three losses), which the lasso never does. At $n = 10^4$, BIC prefers a wrong
   orientation and the search makes $F_1$ worse. The pure search is the weakest method: random
   restarts get stuck in the reversed cycle (§9.1).
5. **Nonconvexity:**
   - **As an initialiser** (MCP/SCAD paths), it brings nothing over the lasso and is sometimes
     worse.
   - **As a score**, the edge-counting BIC is what fixes orientation; the MCP-objective ablation
     helps less.

   This supports the 2 October framing: nonconvexity helps as the score of a discrete search, not
   as a penalty on a continuation path.

6. **At $p = 20$ (§9.4)** lasso + search matches Dettling's oracle-tuned lasso on his pipeline for
   sparse graphs ($k = 1, 2$: +0.07 to +0.09 over BIC-tuned lasso) and does nothing for dense ones
   at $n = 10^3$. On the raw scale ($n = 10^4$, all $k$) every search gains +0.16 to +0.21.
7. **Combined with the independent study** (`next_steps/031026/next_steps_031026.md`): the
   standardised pipeline needs the rescaled $C = 2\,\mathrm{diag}(1/s_{ii}^2)$, and MCP/SCAD should
   be run dense → sparse. Both change the conclusions about the penalties.

All runs of this study have ended. The $p = 20$ run on Dettling's pipeline was stopped early
(§9.4); everything else is complete.

---

## 1. Questions

1. Does a score-based search with add / delete / reverse moves fix the orientation errors of the
   penalised path estimators, on each loss: direct, log-likelihood, Frobenius?
2. Once there is a search, does the penalty matter? That is, does starting from MCP or SCAD beat
   starting from the lasso, or from the empty graph? This separates the contribution of
   nonconvexity from that of the search.
3. Does the score matter? An edge-counting BIC on a refit, against the penalised MCP/SCAD objective
   at a fixed $\lambda$ (the score of the Example 2 test).
4. How do the answers change with $n$ (1000, $10^4$, $\infty$), with irrepresentability failing
   (Example 2), and with 2-cycles?
5. With a data-driven choice of model (BIC) instead of the oracle best-F1 point, how do all methods
   compare?

## 2. The estimator

**State.** A support $S$ of off-diagonal entries; the diagonal is always free (it is not penalised
in S1/S2 either).

**Refit.** An unpenalised fit of $M$ on $S \cup \mathrm{diag}$ under the arm's loss:

| loss | refit on a fixed support |
|---|---|
| direct | least squares on the columns of $A(\hat\Sigma)$ belonging to $S \cup \mathrm{diag}$: closed form, about 1 ms at $p = 20$ |
| log-likelihood | maximum likelihood on the support; L-BFGS or projected gradient through `objective.covariance.loss_value` / `loss_grad`, warm-started from the current estimate |
| Frobenius | the same with the Frobenius loss |

**Score.** One score for every arm, so the arms are comparable: the BIC of the Gaussian model with
covariance $\Sigma(\hat M_S)$,

$$\mathrm{BIC}(S) = n\,\big[\log\det\Sigma(\hat M_S) + \operatorname{tr}(\Sigma(\hat M_S)^{-1}\hat\Sigma)\big] + \log(n)\,(p + |S|),$$

with $C = 2I$ as in the fits.
- **Unstable refits:** a refit that is not stable (an eigenvalue with non-negative real part) has no
  $\Sigma(M)$ and scores $+\infty$.
- **The direct arm:** the refit is not the MLE, so its likelihood is slightly conservative. Using
  the likelihood as the score keeps it on the same footing as the other arms.
- **At $n = \infty$:** BIC needs a finite $n$. With $\hat\Sigma = \Sigma$ exact, a nominal
  $n = 10^6$ makes it prefer exact fits, then fewer edges, i.e. the $\ell_0$ target of the 1 October
  linear program. Sensitivity checked with $10^5$ and $10^7$.

**Moves**, greedy best improvement (evaluate all candidates, take the best, repeat; stop when
nothing lowers the score; at most 200 steps):

| move | candidates | why |
|---|---|---|
| delete $(i,j)$ | every entry of $S$ | removes false edges, and commits a hedge to one direction |
| reverse $(i,j) \to (j,i)$ | every entry of $S$ whose reverse is not in $S$ | undoes a frozen wrong direction (S3a §4.4) |
| add $(i,j)$ | entries not in $S$ | adds missed edges; adding the reverse of a selected entry creates a 2-cycle, which a one-direction-per-pair start cannot represent otherwise |

Cost control for the covariance losses, where a refit is iterative:
- **The direct arm** evaluates every candidate.
- **The other two arms** evaluate every delete and reverse move, but only the $K = 2p$ add moves
  with the largest gradient of the loss at the current refit.

Phase 0 (§7) measures whether this screen loses anything, by running the direct arm with and without
it.

**The four methods compared** (added 3 October at Joon's suggestion):

| method | start of the search | what it tests |
|---|---|---|
| **pure greedy BIC search**, as in Améndola, Dettling, Drton, Onori & Wu (2020), §5 | random starting graphs plus the empty graph; the best-scoring result is reported | no penalty at all: the paper's method adapted to GCLMs |
| **lasso + search** | the lasso path's support at its BIC-selected $\lambda$ | the convex penalty as an initialiser |
| **MCP + search** | the MCP path's support at its BIC-selected $\lambda$ | does nonconvexity help as an initialiser? |
| **SCAD + search** | the SCAD path's support at its BIC-selected $\lambda$ | the same |

References next to the four, not competitors:
- each path at its BIC-selected $\lambda$ *without* search;
- each path at its best-F1 $\lambda$ (oracle), with and without search;
- the search started from the true graph (oracle, as in the paper's Table 1).

**Adapting the paper's search to GCLMs:**
- **Same:** the neighbourhood (add, remove, reverse one edge), best-improvement hill climbing, random
  restarts with the best score kept, the standard BIC penalty $\tfrac12(p+k)\log n$ (in the
  $-2\ell$ scale used here: $(p+k)\log n$).
- **Graphs:**
  - The paper works with *simple mixed* graphs: at most one edge per pair, bidirected edges for
    latent confounding, a linear SEM.
  - A GCLM is a *directed* graph that may contain 2-cycles, with $C$ known.
  - So 2-cycles are allowed by default, and the paper's restriction to simple graphs is a switch
    for a sensitivity run.
- **Increased penalty:** the paper's $\log(p^{2k}3^k)$ counts simple mixed graphs with $k$ edges.
  The analogue for directed graphs is $\log\binom{p(p-1)}{k}$, i.e. eBIC (Chen & Chen 2008) with
  $\gamma_e = 1$ in the $-2\ell$ scale. Reported next to the standard BIC.
- **Random starts:** the paper draws 300 graphs uniformly. That is affordable here at $p = 5$ but
  not at $p = 20$.
  - Restart counts: 50 (Example 2), 10 ($p = 10$), 5 ($p = 20$).
  - Starting density: each restart draws its edge probability uniformly from $[0, 0.3]$, then a
    random graph with that density.
  - Every method's compute (score evaluations, time) is reported, so the comparison can also be
    read at equal cost.
- **Maximum likelihood:** the paper uses block coordinate descent (its reference [6]). Here: the
  refits of §2.

The BIC-selected $\lambda$ comes from refitting the support at each of the 100 $\lambda$'s of the
path and scoring it with the BIC above.

**Ablation: score = the penalised objective.** For MCP and SCAD on the direct loss only: the same
moves, each re-solved with the penalised solver at the start's $\lambda$, accepted if the MCP/SCAD
objective decreases. This is the Example 2 search (`next_steps/021026/files/m0_objective.py`), now
from the data-driven start. For the lasso this is pointless: its objective is convex, so no move
can lower it.

## 3. Design

| | Example 2 | random graphs (Figure 5 generator) |
|---|---|---|
| graphs | path $G_1$; 5-cycle $G_2$ with $m_{15} = 0.65$ and with $m_{15}$ random | $p = 10, 20$; $k = 1..4$; four $C$ choices; reps 0–9, the 640 graphs of S3a |
| random restarts (pure search) | 50 | 10 ($p = 10$), 5 ($p = 20$) |
| $n$ | $10^3, 10^4, 10^5, \infty$ | $10^3, 10^4, \infty$ |
| scale | raw ($C = 2I$ exactly right) | standardised (the Figure 5 pipeline); raw as the correctly specified control at $10^4$ and $\infty$ |
| losses | direct, log-likelihood, Frobenius | direct; then log-likelihood and Frobenius (§7) |
| methods | the four of §2 plus the references | the same |
| reps | 100 per $n$ (as M0) | 10 per $(k, C)$, i.e. 160 graphs per $(p, n)$ |

The datasets are the existing ones (same seeds), so everything is paired with S1, S1b, S2, S3a and
the cluster n-sweep. The direct-loss paths of S3a are stored (`runs/s3a_bidirectional/raw/`), so
for those cells the starting supports are already available.

## 4. Comparisons and metrics

**Metrics at the selected model:**
- directed $F_1$, precision, recall;
- skeleton $F_1$;
- the orientation breakdown of S3a (correct / hedged / reversed / missed; 2-cycles both / half /
  none);
- exact recovery rate and model size;
- the moves accepted, by type;
- score evaluations and run time.

**Contrasts,** all paired over the same graphs (differences with standard errors):

| | contrast | answers |
|---|---|---|
| C1 | start + search vs. the same start without search | question 1 |
| C2 | the four methods: pure greedy BIC search vs. lasso / MCP / SCAD + search, also at equal compute | question 2 |
| C3 | BIC search vs. penalised-objective search (MCP, SCAD; direct loss) | question 3 |
| C4 | direct vs. log-likelihood vs. Frobenius, after the search | the loss |
| C5 | everything against $n$, and Example 2 against random graphs | question 4 |
| C6 | BIC-selected path estimates and searches vs. the oracle `max_f1` of S1/S2 | question 5 |

## 5. Expectations

- **H1.** The search improves directed $F_1$ over the path estimates at their BIC-selected
  $\lambda$, mostly by turning reversed edges and hedges into correct ones.
- **H2a.** The penalty-started searches beat the pure search at equal compute, because the paths
  start them near the right skeleton. With many restarts the pure search should catch up at small $p$
  (Example 2), less so at $p = 20$.
- **H2.** Starting from the lasso is at least as good as starting from MCP: the true direction is
  already inside the lasso's hedges (S3a). If H2 holds, the gain comes from the search and the
  edge-counting score, not from a nonconvex penalty. If MCP starts win, nonconvexity is useful as an
  initialiser.
- **H3.** The BIC score is better than the MCP objective at $n = 1000$, where the objective prefers
  the truth's basin only about half the time (1 October, Example 2). The two converge at large $n$.
- **H4.** On Example 2 every start reaches the 5-cycle at large $n$, from about $10^5$ as in the
  MCP-objective test.
- **H5.** 2-cycles: the add move recovers some of them, but at $n = 1000$ BIC's $\log n$ penalty may
  prefer single edges.
- **H6.** The loss matters less than the start and the search. S2 found the three losses close,
  with the log-likelihood slightly ahead.

## 6. Validation and tests (`tests/test_search.py`)

- **The refits are right.** At $n = \infty$, the refit on the true support returns $M^*$ for every
  loss. The direct-loss refit equals `numpy.linalg.lstsq` on the explicit design restricted to the
  support.
- **The search never increases its score.** The moves are generated correctly: a reverse turns
  $(i,j)$ into $(j,i)$, add-reverse creates a 2-cycle, nothing is duplicated, the diagonal is never
  touched.
- **BIC bookkeeping:** the parameter count is $p + |S|$; the $n = \infty$ convention; unstable
  refits score $+\infty$.
- **Exhaustive check on Example 2** ($p = 5$, 20 off-diagonal entries). Enumerate every support of
  up to 7 edges (about 140 000 direct-loss refits, minutes), find the global BIC optimum, and check
  that the greedy search reaches it from each start. No package implements this search for Lyapunov
  models, so the exhaustive optimum is the reference.
- **The known case:** from the lasso's best-F1 support of Example 2 at $n = \infty$, which has 1→5
  reversed, one reversal move must reach the true support.

## 7. Implementation and phases

**Files:**

| file | role |
|---|---|
| `src/gclm/solvers/search.py` | support refits per loss, the BIC score, the greedy add / delete / reverse search |
| `tests/test_search.py` | §6 |
| `docs/SEARCH.md` | the method: score, moves, refits, $n = \infty$, screening, complexity |
| `simulations/diagnostics/search_study.py` | `run` / `summarize`, numbers only, like `bidirectional.py`; reuses the S3a paths where they exist |
| `simulations/diagnostics/plot_search.py` | figures from the CSVs |
| `runs/s3b_search/` | outputs |

**Phases,** each ending with a look at the numbers before the next:

| phase | what | cost (estimate) |
|---|---|---|
| 0 | implement, test, exhaustive Example 2 check, screening check | about 1 day of work |
| 1 | Example 2, all three losses, all starts, $n = 10^3 \dots \infty$, plus the ablation | about 1 h on the laptop |
| 2 | random graphs, direct loss, $p = 10, 20$, $n = 10^3, 10^4, \infty$, all starts, plus the ablation | a few hours on the laptop (S3a supports reused) |
| 3 | random graphs, log-likelihood and Frobenius losses: $p = 10$ on the laptop, $p = 20$ on the cluster (as one more option of `run_s1_shard.py`, after the n-sweep) | the iterative refits make this the expensive phase; measured in phase 0 |
| 4 (optional) | where the search pays off: edge strength bounded away from 0, graphs without 2-cycles, the raw scale | half a day of code, background runs |

## 8. Open choices to settle before phase 2

- **BIC or eBIC.** At $p = 20$ the model space is large, and eBIC (Chen & Chen 2008, $\gamma_e = 0.5$)
  is the usual guard against over-selection. Plan: report both, choose the main one after phase 1.
- **Refit criterion for the direct arm.** Least squares on the direct loss (cheap, planned) or the
  likelihood refit for every arm (cleaner, expensive). Phase 0 compares them on Example 2.
- **The parameter count with 2-cycles.** $p + |S|$ assumes the model is identifiable. With 2-cycles
  the true dimension can be smaller (Dettling et al. 2023; the cyclic-SEM paper computes it as a
  Jacobian rank). Plan: use $p + |S|$, and check the Jacobian rank on a sample of the selected
  models.
- **Greedy vs. restarts.** Pure greedy can stop in a local optimum. A few random restarts per graph
  are cheap to add if the exhaustive check in phase 0 shows that greedy misses the optimum.

---

## 9. Results

### 9.0 Implementation and validation (phase 0)

**Code:**
- `src/gclm/solvers/search.py`: refits, BIC, neighbourhood, greedy and multistart search, BIC
  along a path.
- `docs/SEARCH.md`: the method.
- `tests/test_search.py`: 12 tests, all passing.
- `simulations/diagnostics/search_study.py`: `validate`, `example2`, `random`, `summarize`.
- `simulations/diagnostics/plot_search.py`.
- Outputs in `runs/s3b_search/`. Each result row stores the selected graph's edge list, so any
  estimate can be drawn.

**Two deviations from the plan, both measured:**
- **Covariance-loss refits inside the search** use tolerance $10^{-8}$ and at most 3 000 APG
  iterations, instead of $10^{-9}$ and 50 000.
  - On Example 2 ($n = 10^4$, three random starts) the final scores and graphs were identical to 4
    decimals, and the search was 5–30× faster. With the strict settings, one log-likelihood search
    took 487 s, because some refits on flat supports crawled to the iteration cap.
  - The unit tests keep the strict settings.
- **The run sizes are smaller than planned,** to fit the night:
  - Example 2: 10 datasets per $n$.
  - Restarts of the pure search: 20 on the direct loss and 10 on the covariance losses (Example 2);
    10 at $p = 10$ and 5 at $p = 20$.
  - At $p = 20$: BIC only, without the oracle-start searches.

**Exhaustive check** (`runs/s3b_search/validate_direct/validate.json`). Example 2's 5-cycle, direct
loss: every support with up to 7 edges, i.e. 137 980 refits per dataset.

| dataset | global BIC optimum | greedy from the lasso / MCP / SCAD start | from the empty graph | from 10 random graphs |
|---|---|---|---|---|
| $n = \infty$ | **the truth** | reaches it | does not | 4 of 10 |
| $n = 10^4$, rep 0 | a 5-edge graph with the true skeleton and 3 edges flipped (BIC 8 below the truth's) | reaches it | reaches it | 7 of 10 |
| $n = 10^4$, rep 1 | the same kind (BIC 1.5 below the truth's) | reaches it | reaches it | 7 of 10 |

- **The greedy search optimises well** from the penalty starts.
- **In the population the BIC optimum is the truth.** Example 2 is identified at $n = \infty$.
- **At $n = 10^4$ BIC prefers a wrong orientation of the right skeleton,** with the log-likelihood
  refit too: the same graph and score as with the direct refit. The orientation of the 5-cycle is
  barely determined by the likelihood at that $n$. The difference is 1.5–8 BIC units, and the
  reversed cycle is just as close.
- **From the empty graph, at $n = \infty$,** the greedy search builds the whole cycle in the wrong
  direction on the direct and Frobenius losses, and then stops. The log-likelihood search gets it
  right. This is the "first direction is a coin flip" problem of S3a, now inside the greedy search,
  and the reason the paper uses many random restarts.

### 9.1 Example 2 (phase 1)

10 datasets per $n$ (the M0 data), all three losses, BIC (eBIC gives the same graphs here)
(`runs/s3b_search/example2_rows.csv`, `example2_summary.csv`). Directed $F_1$ / share recovered
exactly. At $n = \infty$ the two fixed graphs are one dataset each.

| graph, $n$ | plain lasso (BIC $\lambda$) | plain MCP (BIC $\lambda$) | **lasso / MCP / SCAD + search** | **pure search** | objective search (MCP / SCAD) | search from the truth |
|---|---|---|---|---|---|---|
| path (irrepresentable), $10^3$ | 0.56–0.61 / 0 | 0.57–0.64 / 0 | 0.38–0.42 / 0.0 | 0.33–0.40 / 0.0 | 0.25 / 0.25 | 0.38–0.43 / 0.0 |
| path (irrepresentable), $10^4$ | 0.98 / 0.8 | 1.00 / 1.0 | 0.68 / 0.1 | 0.45 / 0.0 | 0.72 / 0.73 | 0.68 / 0.1 |
| path, $10^5$ | 1.00 / 1.0 | 1.00 / 1.0 | 0.97 / 0.9 | 0.70 / 0.6 | 0.92 / 0.97 | 0.97 / 0.9 |
| path, $\infty$ | 1.00 / 1.0 | 1.00 / 1.0 | 1.00 / 1.0 | 1.00 / 1.0 | 1.00 / 1.00 | 1.00 / 1.0 |
| 5-cycle, $m_{15} = 0.65$, $10^3$ | 0.53–0.54 / 0 | 0.50–0.54 / 0 | 0.39–0.42 / 0.0 | 0.32–0.36 / 0.0 | 0.32 / 0.30 | 0.39–0.42 / 0.0 |
| 5-cycle, $m_{15} = 0.65$, $10^4$ | 0.76–0.79 / 0 | 0.73–0.80 / 0 | 0.68 / 0.0 | 0.56 / 0.0 | 0.68 / 0.71 | 0.68 / 0.0 |
| 5-cycle, $10^5$ | 0.74–0.78 / 0 | 0.76–0.79 / 0 | **0.98 / 0.9** | 0.76 / 0.6 | 0.87 / 0.87 | 0.98 / 0.9 |
| 5-cycle, $\infty$ | 0.62–0.67 / 0 | 0.67–0.73 / 0 | **1.00 / 1.0** | 1.00 / 1.0 | 1.00 / 1.00 | 1.00 / 1.0 |
| 5-cycle, random $m_{15}$, $10^3$ | 0.52–0.61 / 0 | 0.55–0.64 / 0 | 0.43–0.49 / 0.0 | 0.33–0.36 / 0.0 | 0.32 / 0.46 | 0.43–0.49 / 0.0 |
| 5-cycle, random $m_{15}$, $10^4$ | 0.71–0.80 / 0 | 0.74–0.80 / 0 | 0.58 / 0.1 | 0.54 / 0.1 | 0.69 / 0.62 | 0.62 / 0.1 |
| 5-cycle, random $m_{15}$, $10^5$ | 0.76–0.77 / 0 | 0.77 / 0 | **0.89–0.91 / 0.7** | 0.72 / 0.5 | 0.80 / 0.84 | 0.94 / 0.8 |
| 5-cycle, random $m_{15}$, $\infty$ (10 datasets) | 0.62–0.67 / 0 | 0.67–0.71 / 0 | **1.00 / 1.0** | 1.00 / 1.0 | 1.00 / 1.00 | 1.00 / 1.0 |

Ranges are over the three losses for the plain paths. The searches give the same graphs on every
loss: they share the BIC score and only refit differently.

![Example 2: exact recovery](../runs/s3b_search/figures/example2_exact_direct_bic.png)

*Figure 2. Example 2, share of datasets recovered exactly, direct loss. Series with identical
values are drawn side by side: the three penalty-started searches coincide. The plain paths
(hollow markers) never recover the 5-cycle; the searches do from $n = 10^5$ on.*

![Example 2: directed F1](../runs/s3b_search/figures/example2_f1_direct_bic.png)

*Figure 3. Example 2, mean directed $F_1$, direct loss. At $n = 10^3$ and $10^4$ every search is
below the plain paths. The same figures for the other two losses are
`example2_{exact,f1}_{loglik,frobenius}_bic.png`.*

![Example 2 at n = 1e5](../runs/s3b_search/figures/example2_graphs_cycle_fixed_0_100000p0_direct_bic.png)

*Figure 4. One dataset of the 5-cycle at $n = 10^5$. The plain lasso has 1→5 instead of 5→1 plus
one false edge. The three penalty-started searches return exactly the truth. The pure search ends
on the reversed cycle.*

![Example 2 at n = 1e4](../runs/s3b_search/figures/example2_graphs_cycle_fixed_0_10000p0_direct_bic.png)

*Figure 5. The same graph at $n = 10^4$. Every search ends on the right skeleton with three edges
reversed: that graph has the lower BIC at this sample size (§9.0).*

**Reading:**
- **At $n = 10^5$ the search does what the lasso cannot.** The 5-cycle, Dettling's
  irrepresentability counterexample, is recovered exactly in 9 of 10 datasets from every penalty
  start, on every loss. The plain paths never recover it (Figures 2 and 4).
- **At $n = \infty$ every search recovers every graph exactly,** including the pure search and the
  objective search. With an exact covariance the BIC optimum is the truth and all starts reach it.
- **At $n = 10^3$ and $10^4$ the search makes things worse, even on the path graph,** where the lasso works.
  BIC prefers a re-orientation of the right skeleton (§9.0), and even the search started from the
  truth walks there. With five nodes, $n = 10^4$ is not enough for the likelihood to settle
  orientation.
- **The pure search is the weakest method throughout.** At $n = 10^5$ its best restart is often
  the *reversed* cycle ($F_1$ 0.20 in Figure 4). Twenty random restarts get locked into the wrong
  orientation basin: the S3a coin flip, now inside the search. The penalty paths provide starts in
  the right basin.
- **The penalty that starts the search does not matter on Example 2:** lasso, MCP and SCAD all
  converge to the same graphs.
- **The objective ablation** (reversal moves on the MCP/SCAD objective, §2) helps at $n = 10^5$
  (0.80–0.87), but less than the BIC search. The edge-counting BIC score matters, not only the
  reversal moves.

### 9.2 Random graphs, direct loss, $p = 10$ (phase 2a: Dettling's standardised pipeline)

160 graphs per $n$, all paired (`runs/s3b_search/random_direct_p10_rows.csv`, `…_summary.csv`).
Directed $F_1$, and its paired difference to plain lasso at its BIC-selected $\lambda$:

| method | $n = 10^3$ | $n = 10^4$ | $n = \infty$ |
|---|---|---|---|
| plain lasso, BIC $\lambda$ (no search) | 0.504 | 0.569 | 0.539 |
| plain lasso, oracle best-F1 $\lambda$ (Dettling's `max_f1`) | 0.580 (+0.08) | 0.622 (+0.05) | 0.630 (+0.09) |
| plain MCP / SCAD, BIC $\lambda$ | 0.419 / 0.436 | 0.439 / 0.455 | 0.399 / 0.415 |
| **lasso + search** | 0.489 (−0.02) | 0.571 (±0.00) | 0.547 (+0.01) |
| **MCP + search** | 0.438 (−0.07) | 0.456 (−0.11) | 0.412 (−0.13) |
| **SCAD + search** | 0.443 (−0.06) | 0.472 (−0.10) | 0.428 (−0.11) |
| **pure greedy search** | 0.478 (−0.03) | 0.507 (−0.06) | 0.499 (−0.04) |
| search from the true graph (oracle) | 0.679 (+0.18) | 0.768 (+0.20) | 0.771 (+0.23) |

eBIC changes these by at most 0.01. With $z$ up to ±25 on 160 paired graphs, all differences beyond
0.02 are clear.

![Random graphs, Dettling's pipeline](../runs/s3b_search/figures/random_direct_p10_f1_direct_bic.png)

*Figure 6. Random graphs at $p = 10$ on Dettling's pipeline, mean directed $F_1$. Hollow markers:
the paths at their BIC $\lambda$. Filled: after the search. No search rises above the plain lasso
(hollow blue); the oracle-tuned lasso (grey) and the search from the truth (black) are references.*

**Reading:**
- **On Dettling's pipeline the search does not improve on the plain lasso.** None of the four
  methods beats plain lasso at its BIC $\lambda$ by more than 0.01, and all stay 0.05–0.09 below
  Dettling's oracle-tuned lasso.
- **The MCP and SCAD starts stay worse after the search.** The search does not undo their reversed
  edges: 3.8–6.3 per graph, against 3.0 for lasso + search.
- **The pure search sits between them.**
- **The problem is the score, not the optimiser.** In 67–75 % of the graphs, a data-driven search
  ends with a *lower* BIC than the search started from the truth, while having much lower $F_1$.
  The truth-started search reaches the lower score in only 12–30 %. The best optimiser by score is
  the pure search, which finds the lowest BIC in 50–80 % of the graphs, yet has lower $F_1$ than
  lasso + search.
- **On this pipeline, BIC prefers wrong graphs.** The suspected reason is that the pipeline
  standardises $\hat\Sigma$ but assumes $C = 2I$. On the correlation scale the true volatility is
  $2D^{-1}$, so even at $n = \infty$ the true graph does not fit exactly, and the likelihood is free
  to prefer other graphs.

  **Check:** the same comparison on the raw scale, where $C = 2I$ is correct (§9.3).

### 9.3 Raw scale, $p = 10$ (the correctly specified control)

The same 160 graphs per $n$, but $\hat\Sigma$ is the raw covariance, so $C = 2I$ is the true
volatility and the Gaussian likelihood in the BIC is correctly specified
(`runs/s3b_search/random_direct_p10_raw_rows.csv`). Directed $F_1$, and its paired difference to
plain lasso at its BIC-selected $\lambda$:

| method | $n = 10^3$ | $n = 10^4$ | $n = \infty$ |
|---|---|---|---|
| plain lasso, BIC $\lambda$ (no search) | 0.476 | 0.514 | 0.518 |
| plain lasso, oracle best-F1 $\lambda$ (Dettling's `max_f1`) | 0.533 (+0.06) | 0.564 (+0.05) | 0.590 (+0.07) |
| plain MCP / SCAD, BIC $\lambda$ | 0.468 / 0.477 | 0.488 / 0.508 | 0.468 / 0.490 |
| **lasso + search** | **0.549 (+0.07, $z$ 7.6)** | **0.611 (+0.10, $z$ 9.4)** | **0.579 (+0.06, $z$ 6.6)** |
| **MCP + search** | 0.543 (+0.07) | 0.577 (+0.06) | 0.510 (−0.01) |
| **SCAD + search** | 0.547 (+0.07) | 0.586 (+0.07) | 0.533 (+0.02) |
| **pure greedy search** | 0.514 (+0.04) | 0.581 (+0.07) | 0.584 (+0.07) |
| search from the true graph (oracle) | 0.705 (+0.23) | 0.803 (+0.29) | 0.829 (+0.31) |

![Random graphs, raw scale](../runs/s3b_search/figures/random_direct_p10_raw_f1_direct_bic.png)

*Figure 7. The same graphs on the raw scale. Now every search (filled) lies above its starting
path (hollow), and lasso + search reaches the oracle-tuned lasso (grey).*

**Correction (3 October, morning).** "Raw scale" is correctly specified only for `C_ID`, where the
true $C$ is $2I$. In the three random-$C$ settings the fit's $C = 2I$ is wrong on the raw scale as
well, only less so than after standardising. Split by $C$ (lasso + search minus plain lasso at its
BIC $\lambda$; Dettling's oracle lasso in brackets):

| | $p = 10$, $n = 10^3$ | $p = 10$, $n = 10^4$ | $p = 10$, $n = \infty$ | $p = 20$, $n = 10^4$ |
|---|---|---|---|---|
| raw, `C_ID`: $C = 2I$ is the truth | +0.08 (+0.07) | **+0.15** (+0.04) | **+0.14** (+0.06); pure search +0.17 | **+0.27** (+0.03) |
| raw, random $C$ | +0.07 (+0.05) | +0.08 (+0.05) | +0.04 (+0.08) | +0.17 (+0.04) |
| Dettling's standardised pipeline (every $C$) | −0.00 to −0.02 (+0.07) | ±0.00 (+0.04 to +0.06) | +0.01 (+0.08 to +0.10) | +0.06 to +0.09 (+0.05 to +0.08) |

**The better the assumed $C$ matches the truth, the more the BIC search gains.** With the model
exactly right, it beats Dettling's oracle-tuned lasso by 0.1–0.25.

**Reading:**
- **On the correctly specified scale the search clearly helps.**
  - Lasso + search improves the data-driven lasso by 0.06–0.10 and **reaches Dettling's oracle-tuned
    lasso** (+0.016 at $n = 10^3$, +0.046 at $10^4$, −0.010 at $\infty$), with $\lambda$ chosen
    from the data.
  - The pure search does as well at large $n$ (+0.07), but worse at $n = 10^3$ (+0.04), where its
    random restarts miss.
  - At $n = 10^3$ the three penalty starts are equivalent (+0.07 each).
  - So the null result of §9.2 comes from the misspecified standardised pipeline, not from the
    search.
- **The penalty that starts the search matters.**
  - The lasso start is the best of the three, consistent with S3a: its hedges contain the true
    direction.
  - MCP and SCAD starts gain from the search at $n = 10^4$ (+0.06, +0.07), but they stay below the
    lasso start, and at $n = \infty$ the search barely helps them.
  - In the paired orientation counts, MCP + search keeps 4.7 reversed edges per graph at
    $n = \infty$, against 3.3 for lasso + search. The search does not undo all of MCP's frozen
    reversals.
- **Much is still left on the table.** The search started from the truth reaches $F_1$ 0.80–0.83.
  - At $n = \infty$ the truth-started search ends with a lower BIC than every data-driven search in
    45 of 104 graphs (§9.0 counting). In those, greedy from the data-driven starts stops in a worse
    local optimum, and better optimisation (more restarts, tabu, larger moves) would pay.
  - In the others the BIC optimum itself is not the truth, typically because BIC drops weak edges:
    the edge weights are $N(0,1)$, and many are small.
- **Visual check (Figure 8).** On one graph ($k = 2$, `C_ID`, rep 0, $n = 10^4$; 20 true edges),
  the plain lasso keeps 10 false edges and 8 wrong directions ($F_1$ 0.54). Every search prunes
  this to 2–3 false edges and 3–6 wrong directions ($F_1$ 0.56–0.67).

![Example graph, raw scale](../runs/s3b_search/figures/random_direct_p10_raw_graphs_2_C_ID_0_10000_bic.png)

*Figure 8. One random graph on the raw scale ($p = 10$, $k = 2$, `C_ID`, rep 0, $n = 10^4$): the
truth, the plain lasso at its BIC $\lambda$, and the four methods. The same graph on Dettling's
pipeline is `random_direct_p10_graphs_2_C_ID_0_10000_bic.png`.*

### 9.4 Random graphs, direct loss, $p = 20$ (phase 2b)

BIC only, no oracle-start searches, 5 random restarts for the pure search
(`random_direct_p20_rows.csv`: Dettling's pipeline; `random_direct_p20_raw_n10000_rows.csv` and
`random_direct_p20_raw_ninf_rows.csv`: raw scale, 3 reps, i.e. 48 graphs per $n$). Directed $F_1$, and its paired difference to plain lasso at its BIC
$\lambda$:

| method | Dettling's pipeline, $n = 10^3$ (160) | Dettling's pipeline, $n = 10^4$ (first 64 graphs, sparse only; see the correction) | raw scale, $n = 10^4$ (48) |
|---|---|---|---|
| plain lasso, BIC $\lambda$ | 0.483 | 0.570 | 0.441 |
| plain lasso, oracle best-F1 $\lambda$ (Dettling) | 0.538 (+0.06) | 0.635 (+0.06) | 0.479 (+0.04) |
| **lasso + search** | 0.524 (+0.04) | **0.653 (+0.08)** | **0.636 (+0.20, $z$ 11)** |
| **MCP + search** | 0.433 (−0.05) | 0.466 (−0.10) | 0.603 (+0.16) |
| **SCAD + search** | 0.471 (−0.01) | 0.512 (−0.06) | 0.620 (+0.18) |
| **pure greedy search** | 0.474 (−0.01) | 0.566 (±0.00) | **0.646 (+0.21)** |
| search from the truth (oracle) | 0.656 (+0.17) | 0.731 (+0.16) | 0.770 (+0.33) |

**Correction (3 October).** The run on Dettling's pipeline was stopped after 259 of 480 graphs. It
is complete at $n = 10^3$, but at $n = 10^4$ it covers only the sparse graphs: $k = 1, 2$ (80
graphs) and 19 graphs of $k = 3$. The "$n = 10^4$" column above is therefore a sparse-graph result
and overstates the average gain. Split by density (lasso + search minus plain lasso at its BIC
$\lambda$; Dettling's oracle lasso in brackets):

| Dettling's pipeline, $p = 20$ | $k = 1, 2$ | $k = 3, 4$ |
|---|---|---|
| $n = 10^3$ (80 + 80 graphs) | +0.085, $z$ 7.2 (+0.072) | −0.004 (+0.038) |
| $n = 10^4$ (80 + 19 graphs) | +0.074, $z$ 8.3 (+0.076) | +0.062 on 19 graphs of $k = 3$ only (+0.021) |

So on Dettling's pipeline the lasso-started search pays on sparse graphs, where it matches the
oracle-tuned lasso, and does nothing on dense ones at $n = 10^3$. $n = \infty$ was not run.

![p = 20, raw scale](../runs/s3b_search/figures/random_direct_p20_raw_f1_direct_bic.png)

*Figure 9. $p = 20$ on the raw scale (48 graphs per $n$). The lasso start and the pure search keep
their gain at $n = \infty$; the MCP and SCAD starts lose most of theirs. There is no such figure for
Dettling's pipeline at $p = 20$, because only $n = 10^3$ was run in full.*

**Reading:**
- **At $p = 20$ the search helps even on Dettling's misspecified pipeline,** for the lasso start
  only: +0.04 at $n = 10^3$, and +0.08 at $10^4$, which beats the oracle-tuned lasso. The MCP and
  SCAD starts lose.
  - The lasso's paths at $p = 20$ hold many more hedges (10–12 per graph). The search turns them
    into single correct directions: lasso + search keeps 1.8–3.7 hedges. MCP and SCAD reach the
    search with their reversals already frozen: 5–8 reversed edges after the search, against 2–6
    for lasso + search.
- **On the raw scale the gains are large:** +0.16 to +0.21 for all four methods, all well above the
  oracle-tuned lasso. The pure search is best here (with 5 restarts), lasso + search a close second.
- **Raw scale at $n = \infty$** (48 graphs, `random_direct_p20_raw_ninf_rows.csv`): plain lasso at
  its BIC $\lambda$ 0.433 and at the oracle $\lambda$ 0.524; lasso + search 0.590 (+0.16, $z$ 6.5);
  pure search 0.630 (+0.20); MCP + search 0.444 (+0.01) and SCAD + search 0.486 (+0.05); search from
  the truth 0.734. The lasso start and the pure search keep their gains. The MCP and SCAD starts
  lose theirs, as at $p = 10$.
