# S3b: a BIC search with reversal moves, started from the lasso, MCP and SCAD paths, on all three losses

*Second part of S3 (`plan.md`: score-based search with a BIC-type penalty). Written 2 October 2026.
S3a (`S3a_bidirectional_edges.md`) showed that the direction entering the path first is close to a
coin flip for every penalty. The lasso later corrects it by keeping both directions; MCP freezes it.
Example 2 (`../next_steps/021026/next_steps_021026.md` §3) showed that the MCP objective prefers the
truth at large $n$, and that a reversal search on that objective reaches it while the continuation
path does not.*

**Status: plan, nothing implemented yet.**

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

**Starting points.** Each run starts from the support of one estimate:

| start | what it tests |
|---|---|
| lasso path at its BIC-selected $\lambda$ | the realistic, data-driven version |
| MCP path at its BIC-selected $\lambda$ | does nonconvexity help as an initialiser? |
| SCAD path at its BIC-selected $\lambda$ | the same |
| each path at its best-F1 $\lambda$ | oracle start; links to S1–S3a |
| the empty graph | the pure search, no penalty at all |

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
| $n$ | $10^3, 10^4, 10^5, \infty$ | $10^3, 10^4, \infty$ |
| scale | raw ($C = 2I$ exactly right) | standardised (the Figure 5 pipeline); raw as the correctly specified control at $10^4$ and $\infty$ |
| losses | direct, log-likelihood, Frobenius | direct; then log-likelihood and Frobenius (§7) |
| starts | all five of §2 | all five |
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
| C2 | after the search: lasso vs. MCP vs. SCAD vs. empty start | question 2 |
| C3 | BIC search vs. penalised-objective search (MCP, SCAD; direct loss) | question 3 |
| C4 | direct vs. log-likelihood vs. Frobenius, after the search | the loss |
| C5 | everything against $n$, and Example 2 against random graphs | question 4 |
| C6 | BIC-selected path estimates and searches vs. the oracle `max_f1` of S1/S2 | question 5 |

## 5. Expectations

- **H1.** The search improves directed $F_1$ over the path estimates at their BIC-selected
  $\lambda$, mostly by turning reversed edges and hedges into correct ones.
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

*(to be filled in phase by phase)*
