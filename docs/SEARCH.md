# Greedy search over GCLM supports (`gclm.solvers.search`)

Score-based greedy search over the supports of $M$: the greedy search of Améndola, Dettling, Drton, Onori & Wu 2020, §5, adapted to graphical continuous Lyapunov models; used in study S3b and in the campaign of October 2026.

| | |
|---|---|
| study | S3b, `simulations/S3b_reversal_search.md` |
| tests | `tests/test_search.py` |
| reference | Améndola, Dettling, Drton, Onori & Wu (2020), *Structure learning for cyclic linear causal models*, §5; §6 below |

## 1. The search space

| term | definition |
|---|---|
| state | a support $S$: the set of off-diagonal entries of $M$ allowed to be nonzero; the diagonal is always free |
| edge $i \to j$ | the entry $M_{ji}$; the edge is in the graph iff $M_{ji} \ne 0$ |
| 2-cycles | allowed by default: a GCLM can have them, and Dettling, Drton & Kolar's model and simulation do, their Section 5, reproduced by `gclm.data.simulate` |
| `allow_two_cycles=False` | the paper's restriction to simple graphs, at most one edge per pair |
| bidirected edges | none, unlike the paper: the volatility $C = 2I$ is known |

$$\text{every off-diagonal entry } M_{ij} = \omega_{ij}\varepsilon_{ij} \text{ independently},\qquad \omega_{ij} \sim \mathrm{Bernoulli}(d),\qquad d = k/p$$
$$P(\text{a pair carries a 2-cycle}) = d^2,\qquad \text{fraction of the true edges in a 2-cycle} = d$$

| $p$ | fraction of the true edges in a 2-cycle |
|---|---|
| 10 | 10 to 40 % |
| 20 | 5 to 20 % |

## 2. Scoring a support

Supports are compared by their score, a loss plus a penalty; the lowest score wins. BIC and eBIC name only the penalty.

$$\mathrm{score}(S) = L(S) + \mathrm{pen}(S)$$
$$L(S) = n\big[\log\det\Sigma_S + \operatorname{tr}(\Sigma_S^{-1}\hat\Sigma)\big],\qquad \Sigma_S = \Sigma(\hat M_S):\quad \hat M_S\Sigma_S + \Sigma_S\hat M_S^\top + C = 0$$

| term | meaning |
|---|---|
| $L(S)$ | $n$ times the log-likelihood loss of `docs/LIKELIHOOD.md` at the refit $\hat M_S$, i.e. $-2$ times the Gaussian log-likelihood up to a constant, Dettling, Drton & Kolar eq. (6.1) |
| $\hat M_S$ | the refit of $M$ without penalty on $S \cup \mathrm{diag}$, step 1 below |
| eBIC term | the part by which an eBIC penalty exceeds the BIC penalty; Dettling's "EBIC$_\gamma$", the whole sum of his eq. (6.2), is here the score with the eBIC penalty |

| penalty | $\mathrm{pen}(S)$ | where |
|---|---|---|
| BIC | $(p + \lvert S\rvert)\log n$ | the default, $\gamma = 0$ |
| eBIC, Dettling's form | $(p + \lvert S\rvert)\log n + 4\gamma\lvert S\rvert\log p$, his eq. (6.2) | the campaign from wave 5c on, `--ebic-gamma` |
| eBIC, Chen & Chen's form | $(p + \lvert S\rvert)\log n + 2\gamma\log\binom{p(p-1)}{\lvert S\rvert}$ | S3b, $\gamma = 1$ |

Relation to the paper's scores 16 and 17, with $k = \lvert S\rvert$ edges:

$$-2n \cdot \tfrac1n\big[\ell - \tfrac12(p+k)\log n\big] = -2\ell + (p+k)\log n = \mathrm{score}_{\mathrm{BIC}}(S) + np\log 2\pi$$
$$\log\binom{N}{k} = k\log(N/k) + k - O(\log k),\quad N = p(p-1) \approx p^2,\qquad 2k\log p = k\log p^2,\qquad \text{Dettling's form larger by about } k(\log k - 1)$$

| | |
|---|---|
| the $-2\ell$ scale | the paper maximises $\tfrac1n[\ell - \tfrac12(p+k)\log n]$, penalty $\tfrac12(p+k)\log n$ with model dimension $p + k$; times $-2n$ it is the score with the BIC penalty plus the constant $np\log 2\pi$. The ordering of graphs, the greedy moves and the local optima are the same; Schwarz's BIC penalty, Chen & Chen's term and Dettling's eq. (6.2) are written on the $-2\ell$ scale, so they are added literally |
| Chen & Chen's term, $\gamma = 1$ | the analogue of the paper's increased penalty (17), whose $\log(p^{2k}3^k)$ counts simple mixed graphs; here the count is of directed graphs with $k$ edges among $p(p-1)$ possible ones |
| the two eBIC forms | close for sparse graphs, by the expansion above |

Step 1, refit $M$ without penalty on $S \cup \mathrm{diag}$:

| loss | refit | class |
|---|---|---|
| direct, $\tfrac12\lVert A(\hat\Sigma)\operatorname{vec}(M) + \operatorname{vec}(C)\rVert^2$ | least squares on the columns of $A$ in $S \cup \mathrm{diag}$, via the normal equations on the precomputed Gram matrix, `lstsq` as fallback | `DirectRefit` |
| log-likelihood or Frobenius, as in `docs/LIKELIHOOD.md` | the package's APG solver, `solvers.covariance.solve`, with weight 1 and $\lambda = 10^8$ outside the support, so those entries stay exactly zero, and weight 0 on it; warm-started from the current estimate when that is stable, otherwise from the diagonal fit; inside the selection and the search tolerance $10^{-8}$ with at most 3,000 steps, `SEARCH_TOL`, `SEARCH_MAX_ITER`, whose scores and selected graphs equal those at $10^{-9}$ / 50,000, S3b §9.0; the class default is $10^{-9}$ / 50,000. A refit that stops at the cap is counted, `CovRefit.unconverged`, code review A2, §3 | `CovRefit` |

Step 2, score the implied covariance, the stationary covariance from $M\Sigma + \Sigma M^\top + C = 0$:

$$\mathrm{score}(S) = n\big[\log\det\Sigma(\hat M_S) + \operatorname{tr}(\Sigma(\hat M_S)^{-1}\hat\Sigma)\big] + \mathrm{pen}(S)$$

| case | rule |
|---|---|
| unstable refit, an eigenvalue of $\hat M_S$ with real part $\ge 0$ | stability is part of the model: no stationary covariance, score $+\infty$. The direct-loss refit can produce such an $M$; the covariance-loss solver accepts only stable iterates |
| $n = \infty$, $\hat\Sigma = \Sigma$ exactly | nominal $n = 10^6$ in the loss term and in the penalty; the score then prefers exact fits first and fewer edges second, the $\ell_0$ target |
| singular Gram matrix $A_S^\top A_S$ | happens when a nonzero $M$ supported in $S$ solves $M\hat\Sigma + \hat\Sigma M^\top = 0$. At $n = \infty$ for real: two nodes $i, j$ isolated in the true graph have $\hat\Sigma_{ij} = 0$ exactly, the columns of $A$ for $M_{ij}$ and $M_{ji}$ are identical, so any support holding both directions of such a pair, as the random starts of the pure search do, is rank deficient. Fallbacks: least squares on the columns, the minimum-norm solution; then SciPy's QR-based driver, because LAPACK's SVD-based one failed to converge on one such matrix on the cluster, campaign, 5 October; if nothing works the support scores $+\infty$ and the search skips it. At finite $n$ the columns are only nearly dependent |

**Names in the code.** They are older than these terms and are kept, because the stored results and the command-line flags use them.

| name | meaning |
|---|---|
| `bic()` | the score with the loss term $L$ |
| `bic_direct()` | the direct-loss score below |
| `bic_along_path()` | the scores of the supports of a path |
| `bic(..., ebic_form="binom")` | Chen & Chen's eBIC term; S3b, $\gamma = 1$ |
| `bic(..., ebic_form="dettling")` | Dettling's eBIC term, eq. (6.2), $4\gamma\lvert S\rvert\log p$; the form the campaign uses |
| `--select bic` | selects by the score |
| `--ebic-gamma` | the $\gamma$ of the eBIC penalty, inside the selection and the search from wave 5c on; waves 1, 2, 5a and 5b run with $\gamma = 0$, the BIC penalty |
| `bic_*` | fields of the graph selected with the BIC penalty |
| `ebic05_*`, `ebic1_*` | fields of the graph selected with the eBIC penalty, $\gamma = 0.5$, 1, computed offline on the supports of a path, `campaign.py` |
| `ebic1_search_*` | selected and searched with the eBIC penalty inside the run, wave 5c on |

Three scores, by how $M_S$ is fitted and by what it is scored:

| score | fit of $M_S$ on $S \cup \mathrm{diag}$ | loss term | code | where |
|---|---|---|---|---|
| likelihood, least-squares fit | least squares on the direct loss, `DirectRefit`; closed form, microseconds | $L$: the Gaussian likelihood at the least-squares fit, not the maximised likelihood | `bic()`, `--refit direct` | the campaign's default; in between the two scores named in the meeting notes, `plan.md`, Simulations: fitted by the quadratic loss, scored by the likelihood loss term |
| likelihood, likelihood fit | maximised Gaussian likelihood, `CovRefit`; iterative, 0.05 to 1 s | $L$ | `bic()`, `--refit loglik` | "likelihood + BIC-type penalty" of `plan.md`, as in Améndola et al. and Dettling et al.; S3b for the covariance losses; waves 5a and 8 |
| direct loss | least squares on the direct loss, `DirectRefit` | $N\log(\mathrm{RSS}_S/N)$ | `bic_direct()`, `Scorer(..., score="direct")`, 9 October | "quadratic loss + BIC-type penalty" of `plan.md`; not used in the campaign |

The direct-loss score:

$$\mathrm{score}_{\mathrm{dir}}(S) = N\log(\mathrm{RSS}_S/N) + (p + \lvert S\rvert)\log n$$
$$N = \tfrac{p(p+1)}{2},\qquad \mathrm{RSS}_S = \sum_{i\le j} R_{ij}^2,\qquad R = M_S\hat\Sigma + \hat\Sigma M_S^\top + C,\qquad \mathrm{RSS}_S \text{ floored at } 10^{-14}\lVert C\rVert_F^2,\ \texttt{RSS\_FLOOR}$$

| property | |
|---|---|
| meaning | the score of a Gaussian regression with unknown error variance on the $N$ distinct equations of the Lyapunov system, with the BIC penalty; $R$ is symmetric and its upper triangle holds the $N$ equations once each |
| not a likelihood of the data | the $N$ residuals are neither independent nor of equal variance, and their number does not grow with $n$ |
| pull toward dense graphs | a support with $p + \lvert S\rvert \ge N$ free entries generically fits the $N$ equations exactly, RSS zero up to rounding, floored at $10^{-14}\lVert C\rVert_F^2$, `RSS_FLOOR`, so its loss term lies far below that of every graph that does not fit exactly, at every finite $n$; at $n = \infty$ the true support already fits exactly and the penalty decides among exact fits |
| until 13 October 2026 | the full Frobenius norm $\lVert R\rVert_F^2$ was used, which counts every off-diagonal equation twice, code review D; the check table of §2a was computed with that version |

## 2a. From a path to one graph: oracle $\lambda$, score, search

A regularisation path gives 100 graphs per dataset, one per $\lambda$; the campaign records all three ways to one graph for every estimator, `run_s1_shard.py --select search`, campaign note §3.3.

| | how the graph is chosen | uses the truth | column |
|---|---|---|---|
| oracle $\lambda$ | every graph on the path is compared with the true graph; the best $F_1$ is reported | yes | `max_f1`, Dettling's Figure 5; likewise `max_acc`, and `auc` / `aupr`, which summarise the whole path |
| selected by the score | every support on the path is refitted as in §2, least squares on the direct loss, and scored, with the BIC penalty unless stated; the lowest score wins, `bic_along_path` | no | `bic_f1` |
| after the search | the selected graph is the start of the greedy search of §3 | no | `search_f1` |
| reading | the oracle value says whether a good graph is on the path; the other two say whether one gets it without the truth | | |
| they can disagree | at $p = 10$, $n = 10^3$ the dense → sparse MCP path gains +0.016 in `max_f1` over the lasso and nothing in `bic_f1` | | |

Why refit, and why by least squares:

| | |
|---|---|
| why refit | the estimate at a given $\lambda$ is shrunk by the path's penalty, so its likelihood says little about the support; the score evaluates each support at the best unpenalised fit on it, $M$ restricted to the support plus the diagonal |
| why least squares | the direct loss $\tfrac12\lVert M\hat\Sigma + \hat\Sigma M^\top + C\rVert_F^2$ has a residual linear in $M$, so the fit is an ordinary least-squares problem, `DirectRefit`; the refit is then plugged into the loss term $L$ of §2, the Gaussian log-likelihood |
| consequences | the same selection rule applies to every estimator and every loss, so only the paths differ; the refit minimises the direct loss rather than maximising the likelihood, so $L$ is the likelihood at the least-squares fit, not the maximised likelihood; at $n = \infty$ with the correct $C$ the two coincide on the true support, both fit exactly; at finite $n$ they differ slightly |
| the likelihood version | `CovRefit`, `--refit loglik`; used in S3b for the covariance losses and in wave 5a, cells `direct_<estimator>-ml_<C>`; about 0.05 to 0.1 s per support at $p = 10$ against microseconds for least squares |
| where a loss enters | a score-based search has no $\lambda$ and no penalty on $M$, but needs a fitted model for every graph: the score is the loss term, $-2$ times the log-likelihood of the best model with that graph, plus the penalty, and that best model is an estimate of $M$ on $S$. Every score is two steps, the fit and the loss term, and a loss enters both; the search compares graphs only by this number |

| check of 10 October | 32 graphs: $p = 10$, $C = 2I$, $k = 1, \dots, 4$, the four true $C$, two replicates; the lasso path of the direct loss; every score with the BIC penalty; the true graphs have 23.7 edges on average; `next_steps/091026/score_check.py` and `.txt` |
|---|---|

| $n$ | loss term of the score | from the truth: $F_1$ (edges) | selected on the path | after the search | s per graph |
|---|---|---|---|---|---|
| $10^3$ | likelihood, least-squares fit (the campaign's) | 0.684 (17.8) | 0.515 (23.3) | 0.482 (17.0) | 0.8 |
| $10^3$ | likelihood, likelihood fit (Améndola, Dettling) | 0.703 (17.4) | 0.510 (22.8) | 0.478 (16.5) | 565 |
| $10^3$ | direct loss, $N\log(\mathrm{RSS}_S/N)$ | 0.679 (35.0) | 0.426 (45.0) | 0.426 (45.0) | 0.5 |
| $10^4$ | likelihood, least-squares fit | 0.788 (23.7) | 0.565 (29.5) | 0.558 (23.4) | 0.6 |
| $10^4$ | likelihood, likelihood fit | 0.801 (23.5) | 0.566 (29.3) | 0.560 (23.2) | 623 |
| $10^4$ | direct loss | 0.747 (31.0) | 0.440 (45.0) | 0.440 (45.0) | 0.4 |

| finding | |
|---|---|
| the direct-loss score is degenerate | on the path it selects the densest graph, the lasso's dense end with $p(p-1)/2 = 45$ edges, in every one of the 64 graphs, and the search stays there; from the true graph it adds 7 to 11 edges; not used in the campaign |
| the likelihood fit and the least-squares fit select the same graphs | the selected and the searched graphs differ by at most 0.005 in $F_1$; from the truth the likelihood fit is 0.01 to 0.02 higher; it costs 700 to 1,000 times as much; wave 5a's result again, on other data |

When the least-squares refit is unstable:

| | |
|---|---|
| mechanism | the least-squares fit of a sparse support is not entry by entry: the off-diagonal residuals $(M_{ii} + M_{jj})\hat\Sigma_{ij}$ couple the diagonal entries, and with the rescaled $C$, whose diagonal varies by an order of magnitude, the compromise can put a diagonal entry above zero. Then $M$ is unstable, the score is $+\infty$, and a search started there cannot leave unless a single move makes the refit stable |
| wave 2, the empty graph dead | 6 to 7 % of the graphs at $p = 10$, 24 to 25 % at $p = 20$; rescaled $C$, never with $C = 2I$; most under `C_Random_Min_Diag` and `C_Random_Full` |
| wave 2, the true support dead | 2 to 5 %, almost all `C_Random_Full` |
| the best of the 11 starts dead | one graph of 2 375 |
| the likelihood refit | maximises over stable $M$ only, the covariance solver accepts no unstable iterate, so it has no dead starts. On the first such $p = 20$ graph the least-squares refit of the empty graph has an eigenvalue at $+0.11$, the likelihood refit has its largest real part at $-0.36$ and a finite score, and a search with the likelihood refit from the empty graph makes 43 moves; `next_steps/051026/files/time_starts.py` for the dataset |

## 3. The moves and the search

Neighbourhood of $S$, as in the paper:

| move | candidates |
|---|---|
| delete | every entry of $S$ |
| reverse | every $(i,j) \in S$ whose reverse $(j,i)$ is not in $S$; it is replaced by $(j,i)$ |
| add | every off-diagonal entry not in $S$; adding the reverse of a selected entry creates a 2-cycle |

Best improvement, `greedy_search`:

```
greedy_search(S_0):
    S <- S_0
    repeat up to max_steps times:                  # 200 by default; the runners allow p(p-1); SearchResult.capped says the cap stopped it
        score every neighbour of S                 # scored supports are cached: no support is refitted twice
        S* <- argmin over neighbours(S) of score
        if score(S*) < score(S) - 1e-9 |score(S)|:  S <- S*
        else:                                       stop     # no neighbour improves
    return S
```

| option or finding | |
|---|---|
| `add_screen`, with an iterative refit | restricts the add moves scored per step to the ones with the largest $\lvert\nabla L\rvert$ at the current refit; deletes and reverses are always all scored. `--add-screen` is off in waves 5a and 8 (b), so those searches differ from wave 1 in the score only, not in the neighbourhood; scoring every add move with the likelihood refit costs 15 to 100 s per graph at $p = 10$. Wave 8 (c), the 100-start search with the likelihood refit, uses `--add-screen 20`: 100 starts of one graph with every add move scored would take longer than a cluster task may run |
| iterative refits that do not converge, code review A2, 13 October 2026 | the covariance refits are capped at 3,000 steps. On a support with a 2-cycle the likelihood need not have a maximiser: the two entries of the cycle can grow without bound while the loss keeps falling, seen $\lvert M_{ij}\rvert$ from 460 to 2,000 over 60,000 steps with the score still falling, so the refit stops at the cap and the score is the value there, 5 to 19 units above the infimum at $n = 10^3$, one to three edges' worth of BIC penalty: a bias against such graphs in every search with the likelihood refit, waves 5a, 8 (b), 8 (c). Measured on the stored wave 8 cells, 0 of 16 selected and searched supports were capped, and in one full search step 7 of 106 neighbours were, all adds that create a 2-cycle; tightening them left the chosen move unchanged. The scores themselves are left as they are |
| convergence flags | `Scorer.unconverged` counts the capped refits; every `SearchResult` carries `unconverged`, the number during its search, and `capped`, whether its step cap stopped it; the cluster runners store them per graph, `bic_unconverged`, `search_unconverged`, `search_capped`, and `pure_*`, `truth_*` |
| the matrix behind the end graph, code review A2b | the scorer caches scores, not matrices; `greedy_search` keeps the matrix of its best move when that support was scored for the first time in the step, so the stored refit is exactly the one that was scored. A support scored in an earlier step, a cache hit, typically "reverse" after "add", has its matrix recomputed from the current warm start, which for an iterative refit can be another stationary point: if it scores lower, that better value replaces the cached score, still a strict descent; if higher, the cached score stands and the case is counted, `mismatches`, stored as `search_mismatches`. With the least-squares refit the recomputed matrix is identical |
| code review A5 | the same recomputation is why two start blocks of wave 8 (c), each with its own cache, can score one end graph differently: 4 of 36 end graphs by more than $10^{-9}$, up to $7.8\times10^{-3}$ |

Starts:

| function | start |
|---|---|
| `greedy_search` | one given support, e.g. a penalised path's support at the $\lambda$ selected by the score, `bic_along_path` |
| `multistart_search` | several supports with one shared cache; the best-scoring result is returned. The pure search uses randomly drawn graphs plus the empty graph |
| `random_support`, "sparse", `--starts sparse`, the default | one edge probability $d \sim U[0, 0.3]$ for the graph, then every off-diagonal entry independently with probability $d$; 2-cycles allowed, no self-loops |
| "uniform", `--starts uniform` | every off-diagonal entry with probability $\tfrac12$, i.e. uniform over all directed graphs on $p$ nodes |
| seed | `(RESTART_SEED, p, k, C, rep)` per data set in `run_search_shard.py`, so the starts of a data set are the same at every $n$, for both $C$ and whatever the shard layout |

## 3a. Side by side: the search of Améndola et al. (2020) and the search here

Both are greedy hill-climbing over graphs with a score of likelihood loss term plus penalty and single-edge moves; they differ in the graph class, the score, the number and distribution of the random starting graphs, and in how the likelihood is maximised.

### The paper's method

```
Améndola, Dettling, Drton, Onori & Wu (2020), §5–6

Input:  sample covariance S (n observations, p variables), penalty (BIC or increased)
Graph class:  simple mixed graphs G = (V, D, B): directed edges D and bidirected edges B,
              at most ONE edge of any type between a pair of nodes, cycles allowed

score(G):
    Σ̂_G   <- argmax over Σ in the model of G of the Gaussian log-likelihood
              ℓ(Σ; S) = -(n/2) [ log det(2πΣ) + tr(Σ⁻¹ S) ]
              (block coordinate descent of Drton, Fox & Wang 2019)
    k      <- |D| + |B|
    return (1/n) [ ℓ(Σ̂_G; S) - penalty(p, k, n) ]
    with  penalty = ½ (p + k) log n                         (BIC penalty)
       or penalty = ½ (p + k) log n + log(p^{2k} 3^k)      (increased penalty, eq. 17)

neighbours(G):  every simple mixed graph obtained from G by
    adding one edge (directed or bidirected, where no edge exists yet),
    removing one edge,
    reversing one directed edge

greedy(G_0):
    G <- G_0
    repeat up to 10^4 times:
        G* <- argmax over neighbours(G) of score
        if score(G*) > score(G):  G <- G*
        else:                     stop
    return G

main:
    for r = 1 .. 300:   G_r <- greedy(random graph)       # 300 randomly drawn starting graphs per data set
    (and, as a reference, greedy(true graph))
    return the G_r with the highest score

Simulations: p = 5 and 6, 100 graphs, n = 10², 10³, 10⁴.
How the 300 starting graphs are drawn is not stated; the MCMC of their §6.1 draws the TRUE graphs
uniformly over simple mixed graphs, "in analogy to [22]" = Nowzohour, Maathuis, Evans & Bühlmann (2017).
```

### The method here

```
gclm.solvers.search

Input:  sample covariance Σ̂ (or the population Σ, "n = ∞"), the volatility matrix C the path used
Graph class:  directed graphs on p nodes: any set S of off-diagonal entries of M,
              2-cycles allowed, no bidirected edges (C is known)

score(S):
    M̂_S   <- argmin over M with zeros outside S ∪ diag of ½ ‖MΣ̂ + Σ̂Mᵀ + C‖²_F
              (least squares on the direct loss: normal equations, closed form)
    Σ(M̂_S) <- stationary covariance of M̂_S under C   (solve M Σ + Σ Mᵀ + C = 0)
    if M̂_S is not stable, or no finite refit exists:  return +∞
    return L(S) + pen(S)
    with  L(S)   = n [ log det Σ(M̂_S) + tr(Σ(M̂_S)⁻¹ Σ̂) ]      (loss term: -2 ℓ up to a constant)
          pen(S) = log(n) (p + |S|)                           (BIC penalty)
              or   log(n) (p + |S|) + 4 γ |S| log p           (eBIC penalty, Dettling's form;
                                                               from wave 5c on, --ebic-gamma)
    (n = ∞ uses a nominal n = 10^6)

neighbours(S):  every support obtained from S by
    adding one off-diagonal entry (the reverse of a present entry makes a 2-cycle),
    deleting one entry,
    reversing one entry (i, j) whose reverse (j, i) is not in S

greedy(S_0):
    S <- S_0
    repeat up to p(p-1) times:                       # a guard; never reached
        S* <- argmin over neighbours(S) of score     # scores cached; refits warm-started
        if score(S*) < score(S) - 1e-9 |score(S)|:  S <- S*
        else:                                        stop
    return S

main, three uses:
    wave 1 ("after the search"):      greedy(support of an estimator's path selected by the score)
    wave 2 ("search-pure"):           for r = 1 .. R: S_r <- greedy(random graph);    R = 10 (wave 2), 100 (wave 5b)
                                      S_0 <- greedy(empty graph); return the S_r with the lowest score
    wave 2 ("search-truth"):          greedy(true support)
    random graph, "sparse":   edge probability d ~ U[0, 0.3], then each off-diagonal entry with probability d
    random graph, "uniform":  each off-diagonal entry with probability 1/2  (wave 5b; uniform over the class)
```

### The starting graphs

Nowzohour et al. 2017 adapted to this class:

| | Nowzohour, Maathuis, Evans & Bühlmann (2017) | Améndola et al. (2020) | here, `--starts uniform` | here, `--starts sparse`, the default |
|---|---|---|---|---|
| class | bow-free acyclic path diagrams; the constraints, acyclicity and no bow, are global | simple mixed graphs | all directed graphs on $p$ nodes, $2^{p(p-1)}$ of them, no global constraint | the same |
| draw | uniform over the class by an MCMC sampler, their Algorithm 1; uniform sampling of a constrained class is not trivial | not stated for the starting graphs; the true graphs of their simulation by the same sampler | the uniform distribution factorises: every off-diagonal entry in $S$ independently with probability $\tfrac12$, exactly uniform, no MCMC and no burn-in; expected density $\tfrac12$, 45 edges at $p = 10$ and 190 at $p = 20$, against true graphs with density below 0.4 and 0.2; the only published recipe among the references | one edge probability $d \sim U[0, 0.3]$ per starting graph, then every off-diagonal entry independently with probability $d$: a mixture of Erdős–Rényi digraphs over densities 0 to 0.3, chosen so that the starts are about as dense as the true graphs and the searches short; this repository's own recipe from S3b and wave 2, the bound 0.3 and the mixture not taken from a paper |
| $R$ | 100 | 300 | 100, wave 5b | 10 plus the empty graph, wave 2; 100, wave 5b |
| remark | | | had the class been the paper's, at most one edge per pair, uniform would draw each of the $\binom p2$ pairs from $\{\text{none}, i \to j, j \to i\}$ with probability $\tfrac13$, density $\tfrac23$ of the pairs; with 2-cycles allowed the fair coin per entry is the right analogue | |
| timed $p = 20$ graph, `next_steps/051026/files/time_starts.txt` | | | the best of five uniform starts ended at a dense local optimum with 114 false positives | the best of five sparse starts at 3 |
| seeding | | | generator seeded with `(RESTART_SEED, p, k, C, rep)`: the same starting graphs at every $n$, under both $C$ and whatever the shard layout; the empty graph appended last | the same |
| wave 5b | | | both recipes with $R = 100$ on the same data sets, `search100s_p10_Cresc`, `search100u_p10_Cresc`; `simulations/diagnostics/restarts.py` reads off the best of the first $r$ starts of either for every $r \le 100$ | the same |

### The differences

| | Améndola et al. 2020 | here |
|---|---|---|
| graph class | simple mixed graphs: directed and bidirected edges, at most one edge per pair | directed graphs: 2-cycles allowed, no bidirected edges ($C$ known) |
| model | linear SEM with error covariance $\Omega$ (bidirected edges) | GCLM, $M\Sigma + \Sigma M^\top + C = 0$ |
| maximised fit in the score | Gaussian likelihood, by block coordinate descent | least squares on the direct loss (closed form), plugged into the Gaussian likelihood; the likelihood refit (`CovRefit`) is the `--refit loglik` of wave 5a |
| dimension in the penalty | $p + k$, $k$ = number of edges (justified by their expected-dimension theorem for simple mixed graphs) | $p + \lvert S\rvert$; exact when the support is identifiable, over-counts for both directions between two isolated nodes |
| penalties | BIC; "increased" $+\log(p^{2k}3^k)$ | BIC; eBIC in Chen & Chen's form $+2\gamma \log\binom{p(p-1)}{\lvert S\rvert}$ (S3b, $\gamma = 1$) and in Dettling's form $+4\gamma\lvert E\rvert\log p$: offline on the path (S4 §5) and inside the selection and the search (wave 5c, `--ebic-gamma`) |
| moves | add / remove / reverse one edge | add / delete / reverse one entry: the same |
| acceptance | strictly higher score | strictly lower score (sign convention only) |
| iteration cap | $10^4$ | $p(p-1)$, never reached |
| randomly drawn starting graphs | 300 per data set (reference [22]: 100, uniform over the class via MCMC) | **10** plus the empty graph in wave 2; **100** in wave 5b |
| distribution of the starting graphs | not stated; uniform via MCMC in [22] | wave 2: edge probability $U[0, 0.3]$ per graph, i.e. sparser than uniform; this repository's own recipe from S3b, chosen so that the starts are about as dense as the true graphs and the searches short, not taken from a paper; wave 5b: that, and the uniform draw (a fair coin per entry, density 0.5; for this class no MCMC is needed), which is the only published recipe among the references |
| problem size | $p = 5, 6$; 100 graphs; $n \le 10^4$ | $p = 10, 20$; 800 graphs per cell; $n \le \infty$ |
| other starts | the true graph | the true graph; and the graph selected by the score on every estimator's path |
| bottom line | | the moves and the hill-climbing are the paper's; the score is the GCLM analogue of theirs with a cheaper refit; the starting graphs are where the two part: 10 sparse ones here against 300, or 100 uniform ones, and 10 had not saturated (S4 §4a), so the `search-pure` numbers of S4 are a lower bound on the paper's method in this setting; wave 5b runs 100 of each kind |

## 4. Cost

| setting | cost |
|---|---|
| $p = 20$, one step | about 400 candidates scored |
| M2, direct-loss search from the empty graph | about 4 s |
| M2, direct-loss search from a random graph | about 20 s |
| covariance-loss refits | about 10 to 100 times more expensive per candidate |
| dense start: a graph selected by the score at $p = 20$ | can be more than a hundred deletions away from where the search ends; one to two minutes |

## 4a. On the cluster (campaign of October 2026)

```
# after a path: simulations/run_s1_shard.py, function select_graph
run_s1_shard.py --select bic           # bic_along_path scores every support of the path; stores the selected graph
run_s1_shard.py --select search        # then greedy_search from the selected graph
run_s1_shard.py --refit direct         # the default: least squares on the direct loss, for every loss
run_s1_shard.py --refit loglik         # the maximised likelihood; wave 5a at p = 10
run_s1_shard.py --ebic-gamma 0.5 1     # wave 5c on: selects and searches once more per gamma on the same path;
                                       # fields ebic05_*, ebic1_* next to the BIC ones, the two penalties paired graph by graph
run_s1_shard.py --c-scale variance     # the rescaled C of docs/DENSE_START.md §2 in the score; otherwise 2I

# without a path: simulations/run_search_shard.py
run_search_shard.py --restarts 10      # wave 2: multistart_search from the empty graph and 10 random graphs,
                                       # greedy_search from the true graph as a ceiling; end graph and score of every start stored
run_search_shard.py --restarts 100     # wave 5b
run_search_shard.py --starts sparse    # the default; or --starts uniform
run_search_shard.py --start-blocks B   # B contiguous blocks of starts per graph, one (graph, block) pair per task,
                                       # for searches too long for one task: wave 8 (c), 100 starts with the likelihood refit
run_search_shard.py --ebic-gamma 1     # scores both of its searches with the eBIC penalty, Dettling's form

# wave 6: rescoring a finished cell
simulations/rescore_shard.py           # re-selects and re-searches the stored supports of a cell's paths with the eBIC
                                       # penalty after rebuilding each data set from its seed, checked against the stored
                                       # BIC scores; no path recomputed; cells rescore1_<source cell>, overlaid by campaign.py
plot_campaign.py --rule ebic1          # every figure with the eBIC penalty in the score; figures_ebic1/
simulations/diagnostics/restarts.py    # best of the first r starts for every r; with campaign.py puts the start blocks of a graph back together
```

| rule | |
|---|---|
| the refit behind the score | least squares on the direct loss for every loss, `--refit direct`, so that the selection rule is the same for all estimators; or the maximised likelihood, `--refit loglik` |
| the eBIC penalty | off, the BIC penalty is used, except from wave 5c on |
| the volatility matrix in the score | the one the estimator was fitted with: $2I$, or the rescaled $C$ of `docs/DENSE_START.md` §2 |
| the step limit | `max_steps` raised from the default 200 to $p(p-1)$ in both runners; every accepted move lowers the score, so the search ends by itself, and the limit is only a guard |

## 5. Validation

| check | content |
|---|---|
| unit tests, `tests/test_search.py` | the refits recover $M^*$ on the true support; the direct refit satisfies the normal equations; the score's penalties, BIC and both eBIC forms, and unstable cases; the direct-loss score; the neighbourhood moves; the score never increases along a search; multistart keeps the best; the Example 2 reversal |
| exhaustive check, `simulations/diagnostics/search_study.py validate` | on Example 2 every support with up to 7 edges, direct loss, or up to 5, log-likelihood, is scored; the global optimum is compared with the truth and with the result of the greedy search from every start; results in S3b §9 |

## 6. References

| | |
|---|---|
| Améndola, C., Dettling, P., Drton, M., Onori, F. & Wu, J. (2020) | Structure learning for cyclic linear causal models. *UAI 2020*. The greedy search of §5 and the scores (16), (17) |
| Chen, J. & Chen, Z. (2008) | Extended Bayesian information criteria for model selection with large model spaces. *Biometrika* 95(3), 759–771 |
| Chickering, D. M. (2002) | Optimal structure identification with greedy search. *JMLR* 3, 507–554 |
