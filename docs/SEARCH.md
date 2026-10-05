# Greedy BIC search over GCLM supports (`gclm.solvers.search`)

The score-based search of study S3b (`simulations/S3b_reversal_search.md`). It follows the greedy
search of Améndola, Dettling, Drton, Onori & Wu (2020), *Structure learning for cyclic linear causal
models*, §5, adapted to graphical continuous Lyapunov models. Tests: `tests/test_search.py`.

## 1. The search space

A state is a **support** $S$: the set of off-diagonal entries of $M$ allowed to be nonzero. The
diagonal is always free. The edge $i \to j$ is the entry $M_{ji}$.
- **2-cycles are allowed by default.** A GCLM can have them.
- **The paper's restriction to simple graphs** (at most one edge per pair) is the
  `allow_two_cycles=False` switch.
- **Unlike the paper there are no bidirected edges.** The volatility $C = 2I$ is known.

## 2. Scoring a support

1. **Refit** $M$ without penalty on $S \cup \mathrm{diag}$:

   | loss | refit | class |
   |---|---|---|
   | direct, $\tfrac12\lVert A(\hat\Sigma)\operatorname{vec}(M) + \operatorname{vec}(C)\rVert^2$ | least squares on the columns of $A$ in $S \cup \mathrm{diag}$, via the normal equations on the precomputed Gram matrix ($\texttt{lstsq}$ as fallback) | `DirectRefit` |
   | log-likelihood or Frobenius, as in docs/LIKELIHOOD.md | the package's APG solver (`solvers.covariance.solve`) with weight 1 and $\lambda = 10^8$ outside the support (those entries stay exactly zero) and weight 0 on it; warm-started from the current estimate when that is stable, otherwise from the diagonal fit; tolerance $10^{-9}$ | `CovRefit` |

2. **Score** the implied covariance $\Sigma(\hat M_S)$, the stationary covariance from
   $M\Sigma + \Sigma M^\top + C = 0$:

   $$\mathrm{BIC}(S) = n\big[\log\det\Sigma(\hat M_S) + \operatorname{tr}(\Sigma(\hat M_S)^{-1}\hat\Sigma)\big] + \log(n)\,(p + |S|) \;\big[+\;2\gamma_e \log\textstyle\binom{p(p-1)}{|S|}\big].$$

   This is $-2n$ times the paper's score (16) up to a constant: the standard BIC penalty
   $\tfrac12(p+k)\log n$, with model dimension $p + k$.
   - **The bracketed term** is the extended BIC (Chen & Chen 2008). With $\gamma_e = 1$ it is the
     analogue of the paper's increased penalty (17), whose $\log(p^{2k}3^k)$ counts simple mixed
     graphs. Here the count is of directed graphs with $k$ edges among $p(p-1)$ possible ones.
   - **Stability is part of the model.** A refit with an eigenvalue whose real part is $\ge 0$ has
     no stationary covariance, so it scores $+\infty$. The direct-loss refit can produce such an
     $M$; the covariance-loss solver accepts only stable iterates.
   - **$n = \infty$** ($\hat\Sigma = \Sigma$ exactly) uses a nominal $n = 10^6$ in the weight. The
     score then prefers exact fits first and fewer edges second: the $\ell_0$ target.

## 3. The moves and the search

**Neighbourhood** of $S$, as in the paper:

| move | candidates |
|---|---|
| delete | every entry of $S$ |
| reverse | every $(i,j) \in S$ whose reverse $(j,i)$ is not in $S$; it is replaced by $(j,i)$ |
| add | every off-diagonal entry not in $S$. Adding the reverse of a selected entry creates a 2-cycle |

**Best improvement** (`greedy_search`): score every neighbour, move to the best one if it lowers
the score, and repeat. Stop when no neighbour improves, or after 200 steps.
- Scored supports are cached, so no support is refitted twice.
- **For the covariance losses**, only the $2p$ add moves with the largest $\lvert\nabla L\rvert$ at
  the current refit are scored (`add_screen`), because those refits are iterative. Deletes and
  reverses are always all scored.

**Starts:**

| function | start |
|---|---|
| `greedy_search` | one given support, e.g. a penalised path's support at its BIC-selected $\lambda$ (`bic_along_path`) |
| `multistart_search` | several supports with one shared cache; the best-scoring result is returned. The pure search uses random graphs (`random_support`: edge probability drawn uniformly from $[0, 0.3]$) plus the empty graph |

## 4. Cost

At $p = 20$ a step scores about 400 candidates. On the M2 a direct-loss search takes about 4 s from
the empty graph and about 20 s from a random graph. The covariance-loss refits are about 10–100×
more expensive per candidate. From a dense start it takes much longer: a BIC-selected graph at
$p = 20$ can be more than a hundred deletions away from where the search ends, which is one to two
minutes.

## 4a. On the cluster (campaign of October 2026)

- **After a path:** `simulations/run_s1_shard.py --select bic` scores every support of the path
  (`bic_along_path`) and stores the BIC-selected graph; `--select search` then runs
  `greedy_search` from it. Function `select_graph`. The refit is the direct-loss one for every
  loss, so that the selection rule is the same for all estimators.
- **Without a path:** `simulations/run_search_shard.py` runs `multistart_search` from the empty
  graph and 10 random graphs, and `greedy_search` from the true graph as a ceiling.
- **The volatility matrix** in the score is the one the estimator was fitted with: $2I$, or the
  rescaled $C$ of `docs/DENSE_START.md` §2 (`--c-scale variance`).
- **The step limit** `max_steps` is raised from the default 200 to $p(p-1)$ in both runners. Every
  accepted move lowers the score, so the search ends by itself; the limit is only a guard.

## 5. Validation

- **Unit tests** (`tests/test_search.py`): the refits recover $M^*$ on the true support; the
  direct refit satisfies the normal equations; the BIC bookkeeping, eBIC and unstable cases; the
  neighbourhood moves; the score never increases along a search; multistart keeps the best; the
  Example 2 reversal.
- **Exhaustive check** (`simulations/diagnostics/search_study.py validate`): on Example 2 every
  support with up to 7 edges (direct loss) or up to 5 (log-likelihood) is scored. The global
  optimum is compared with the truth and with the result of the greedy search from every start.
  Results: S3b §9.

## 6. References

- Améndola, C., Dettling, P., Drton, M., Onori, F. & Wu, J. (2020). Structure learning for cyclic
  linear causal models. *UAI 2020*. The greedy search of §5 and the scores (16)–(17).
- Chen, J. & Chen, Z. (2008). Extended Bayesian information criteria for model selection with large
  model spaces. *Biometrika* 95(3), 759–771.
- Chickering, D. M. (2002). Optimal structure identification with greedy search. *JMLR* 3, 507–554.
