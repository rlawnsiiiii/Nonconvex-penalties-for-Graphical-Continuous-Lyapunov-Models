# Greedy search (`gclm.solvers.search`)

| | |
|---|---|
| reference | Améndola, Dettling, Drton, Onori & Wu (2020) §5; §6 |
| study | `simulations/S3b_reversal_search.md`; `simulations/S4_campaign.md` |
| tests | `tests/test_search.py` |

## 1. Search space

$$S \subseteq \{(i,j) : i \ne j\},\qquad M_{ij} = 0\ \ \text{for } (i,j) \notin S \cup \mathrm{diag},\qquad i \to j \iff M_{ji} \ne 0$$

$$M_{ij} = \omega_{ij}\varepsilon_{ij}\ \text{independently},\qquad \omega_{ij} \sim \mathrm{Bernoulli}(d),\qquad d = k/p\qquad (\texttt{gclm.data.simulate};\ \text{Dettling, Drton, Kolar, Sect. 5})$$

$$P(\text{2-cycle on a pair}) = d^2,\qquad \text{share of the true edges in a 2-cycle} = d$$

| $p$ | $d$ |
|---|---|
| 10 | 10–40 % |
| 20 | 5–20 % |

| 2-cycles | allowed; `allow_two_cycles=False`: the paper's simple graphs |
|---|---|
| bidirected edges | none; $C = 2I$ known |

## 2. Scoring a support

$$\mathrm{score}(S) = L(S) + \mathrm{pen}(S)\quad\text{(lowest wins)},\qquad L(S) = n\big[\log\det\Sigma_S + \operatorname{tr}(\Sigma_S^{-1}\hat\Sigma)\big] = -2\ell(\hat M_S) + \mathrm{const}$$

$$\hat M_S\Sigma_S + \Sigma_S\hat M_S^\top + C = 0,\qquad \hat M_S = \text{the unpenalised refit on } S \cup \mathrm{diag}\qquad (L/n:\ \texttt{docs/LIKELIHOOD.md};\ \text{Dettling, Drton, Kolar eq. 6.1})$$

$$\mathrm{pen}_{\mathrm{BIC}}(S) = (p + \lvert S\rvert)\log n\qquad(\gamma = 0,\ \text{the default})$$

$$\mathrm{pen}_{\mathrm{eBIC}}(S) = (p + \lvert S\rvert)\log n + 4\gamma\lvert S\rvert\log p\ \ \text{(Dettling eq. 6.2; wave 5c on)}\qquad\text{or}\qquad (p + \lvert S\rvert)\log n + 2\gamma\log\binom{p(p-1)}{\lvert S\rvert}\ \ \text{(Chen and Chen; S3b, } \gamma = 1\text{)}$$

$$\text{eBIC term} = \mathrm{pen}_{\mathrm{eBIC}} - \mathrm{pen}_{\mathrm{BIC}},\qquad \text{Dettling's } \mathrm{EBIC}_\gamma = L + \mathrm{pen}_{\mathrm{eBIC}}$$

$$\text{paper's (16): } -2n\cdot\tfrac1n\big[\ell - \tfrac12(p+k)\log n\big] = -2\ell + (p+k)\log n = \mathrm{score}_{\mathrm{BIC}}(S) + np\log 2\pi,\qquad k = \lvert S\rvert$$

$$\log\binom{N}{k} = k\log\tfrac{N}{k} + k - O(\log k),\quad N = p(p-1) \approx p^2,\quad 2k\log p = k\log p^2\quad\Rightarrow\quad\text{Dettling's form larger by about } k(\log k - 1)$$

$$\text{paper's (17): } \log(p^{2k}3^k) \text{ counts simple mixed graphs; here } \binom{p(p-1)}{k} \text{ counts directed graphs with } k \text{ edges}$$

```
refit(S):                                 # step 1: M without penalty on S ∪ diag
  direct loss ½‖A(Σ̂) vec M + vec C‖²  ->  DirectRefit: least squares on the columns of A in S ∪ diag; normal
                                           equations on the precomputed Gram matrix, lstsq as fallback
  log-likelihood / Frobenius loss     ->  CovRefit: the APG solver solvers.covariance.solve; weight 0 on S ∪ diag,
    (docs/LIKELIHOOD.md)                   weight 1 and λ = 1e8 outside (those entries stay exactly zero); warm start
                                           = the current estimate if stable, else the diagonal fit; tol 1e-8, at most
                                           3,000 steps (SEARCH_TOL, SEARCH_MAX_ITER; same scores and graphs as
                                           1e-9 / 50,000, S3b §9.0; class default 1e-9 / 50,000); a refit stopped at
                                           the cap is counted, CovRefit.unconverged (code review A2, §3)

score(S):                                 # step 2
  M̂_S <- refit(S)
  if max Re λ(M̂_S) ≥ 0 or no finite refit:  return +∞   # stability is part of the model: no stationary covariance.
                                                          # DirectRefit can produce this; CovRefit accepts only stable iterates
  Σ_S <- solve M̂_S Σ + Σ M̂_Sᵀ + C = 0
  return n [log det Σ_S + tr(Σ_S⁻¹ Σ̂)] + pen(S)
  # n = ∞ (Σ̂ = Σ exactly): nominal n = 1e6 (N_INF) in loss and penalty -> exact fits first, fewer edges second: the ℓ0 target
  # singular Gram A_Sᵀ A_S: a nonzero M on S solves M Σ̂ + Σ̂ Mᵀ = 0. At n = ∞ for real: nodes i, j isolated in the
  #   truth have Σ̂_ij = 0 exactly, the columns for M_ij and M_ji coincide, so a support with both directions (as the
  #   random starts of the pure search) is rank deficient. Fallbacks: lstsq (minimum norm); SciPy's QR driver (LAPACK's
  #   SVD driver failed to converge on one such matrix, cluster, 5 Oct); else +∞ and the search skips the support.
  #   Finite n: the columns are only nearly dependent
```

Three scores:

| $\hat M_S$ | loss term | code | use |
|---|---|---|---|
| `DirectRefit`, $\mu$s | $L$ at the least-squares fit | `bic()`, `--refit direct` | campaign default |
| `CovRefit`, 0.05–1 s; $p = 10$: 0.05–0.1 s | $L$ maximised | `bic()`, `--refit loglik` | `simulations/S3b_reversal_search.md` (covariance losses); waves 5a, 8; cells `direct_<estimator>-ml_<C>` |
| `DirectRefit` | $N\log(\mathrm{RSS}_S/N)$ | `bic_direct()`, `Scorer(..., score="direct")` | none |

$$\mathrm{score}_{\mathrm{dir}}(S) = N\log(\mathrm{RSS}_S/N) + (p + \lvert S\rvert)\log n,\qquad N = \tfrac{p(p+1)}{2},\qquad \mathrm{RSS}_S = \max\Big(\sum_{i \le j} R_{ij}^2,\ 10^{-14}\lVert C\rVert_F^2\Big),\qquad R = M_S\hat\Sigma + \hat\Sigma M_S^\top + C$$

$$p + \lvert S\rvert \ge N\ \Rightarrow\ \mathrm{RSS}_S = \text{the floor, generically, at every finite } n;\qquad n = \infty:\ \text{the true support fits exactly, the penalty decides}$$

### Names in the code

```
# older than the terms above; kept, because the stored results and the flags use them
src/gclm/solvers/search.py
  bic(...)                      the score with the loss term L. plan.md: "likelihood + BIC-type penalty" with CovRefit
                                (Améndola et al., Dettling et al.); with DirectRefit (the campaign) it sits between
                                plan.md's two scores: fitted by the quadratic loss, scored by the likelihood loss term
    ebic_form="binom"           Chen & Chen's eBIC term; S3b, γ = 1
    ebic_form="dettling"        Dettling's eBIC term, eq. (6.2), 4 γ |S| log p; the campaign's form
  bic_direct(...)               the direct-loss score (9 Oct 2026); plan.md: "quadratic loss + BIC-type penalty".
                                A Gaussian regression with unknown error variance on the N distinct equations of the
                                Lyapunov system (R is symmetric; its upper triangle holds each once). Not a likelihood
                                of the data: the N residuals are dependent, of unequal variance, and N does not grow
                                with n. Until 13 Oct 2026: ‖R‖²_F, every off-diagonal equation twice (code review D);
                                the check table of §2a was computed with that version
  bic_along_path(...)           the scores of the supports of a path
  RSS_FLOOR = 1e-14             the floor on RSS_S, times ‖C‖²_F
simulations/run_s1_shard.py
  --select bic                  select by the score
  --ebic-gamma G                the γ of the eBIC penalty, inside the selection and the search (wave 5c on; also
                                run_search_shard.py); waves 1, 2, 5a, 5b: γ = 0, the BIC penalty
stored fields, campaign.py
  bic_*                         the graph selected with the BIC penalty
  ebic05_*, ebic1_*             selected with the eBIC penalty, γ = 0.5, 1, computed offline on the supports of a path
  ebic1_search_*                selected and searched with the eBIC penalty inside the run, wave 5c on
```

## 2a. From a path to one graph

| 100 supports per path (`run_s1_shard.py --select search`) | truth | field |
|---|---|---|
| $\max_\lambda F_1$ | yes | `max_f1` (Dettling, Fig. 5); `max_acc`; `auc`, `aupr` |
| $\arg\min_\lambda \mathrm{score}$, `bic_along_path` | no | `bic_f1` |
| `greedy_search` from that support, §3 | no | `search_f1` |
| $p = 10$, $n = 10^3$: MCP dense $\to$ sparse minus lasso | | `max_f1` $+0.016$; `bic_f1` $0$ |

$$\hat M_S = \arg\min\big\{\tfrac12\lVert M\hat\Sigma + \hat\Sigma M^\top + C\rVert_F^2 : M_{ij} = 0\ \forall (i,j) \notin S \cup \mathrm{diag}\big\}\quad\text{(the residual is linear in } M\text{: ordinary least squares, } \texttt{DirectRefit})$$

$$L(S) = -2\ell(\hat M_S) + \mathrm{const} \ \ge\ \min_{M \text{ on } S \cup \mathrm{diag}}\big[-2\ell(M)\big] + \mathrm{const}\ \ (\texttt{CovRefit});\qquad \text{equal at } n = \infty \text{ with the correct } C \text{ on the true support}$$

`next_steps/091026/score_check.py`, `.txt`: 32 graphs, $p = 10$, $C = 2I$, $k = 1, \dots, 4$, 4 true $C$ $\times$ 2 replicates, 23.7 true edges on average; lasso path, direct loss; $\mathrm{pen}_{\mathrm{BIC}}$:

| $n$ | refit, loss term | truth: $F_1$ (edges) | path | search | s/graph |
|---|---|---|---|---|---|
| $10^3$ | `DirectRefit`, $L$ | 0.684 (17.8) | 0.515 (23.3) | 0.482 (17.0) | 0.8 |
| $10^3$ | `CovRefit`, $L$ | 0.703 (17.4) | 0.510 (22.8) | 0.478 (16.5) | 565 |
| $10^3$ | `DirectRefit`, $N\log(\mathrm{RSS}_S/N)$ | 0.679 (35.0) | 0.426 (45.0) | 0.426 (45.0) | 0.5 |
| $10^4$ | `DirectRefit`, $L$ | 0.788 (23.7) | 0.565 (29.5) | 0.558 (23.4) | 0.6 |
| $10^4$ | `CovRefit`, $L$ | 0.801 (23.5) | 0.566 (29.3) | 0.560 (23.2) | 623 |
| $10^4$ | `DirectRefit`, $N\log(\mathrm{RSS}_S/N)$ | 0.747 (31.0) | 0.440 (45.0) | 0.440 (45.0) | 0.4 |

| | `bic_direct()` | `CovRefit` vs `DirectRefit` |
|---|---|---|
| path, search | $p(p-1)/2 = 45$ edges, 64 of 64 | $\Delta F_1 \le 0.005$ |
| from the truth | $+7$ to $+11$ edges | $\Delta F_1 = +0.01$ to $+0.02$ |
| cost | | $\times 700$ to $1{,}000$ (wave 5a again) |

$$\text{the term } (M_{ii} + M_{jj})\hat\Sigma_{ij} \text{ of } R_{ij} \text{ couples the diagonal};\quad \text{rescaled } C\ (\mathrm{diag}\,C \text{ varying by } 10\times) \Rightarrow \hat M_{ii} > 0 \text{ possible} \Rightarrow \mathrm{score} = +\infty\text{: a dead start}$$

| wave 2 dead starts: `DirectRefit`, rescaled $C$ only (never $C = 2I$) | |
|---|---|
| empty graph, $p = 10$ / $20$ | 6–7 % / 24–25 % (`C_Random_Min_Diag`, `C_Random_Full`) |
| true support | 2–5 % (`C_Random_Full`) |
| best of 11 starts | 1 of 2 375 |
| first such $p = 20$ graph, empty start (`next_steps/051026/files/time_starts.py`): $\max\operatorname{Re}\lambda$ | `DirectRefit` $+0.11$; `CovRefit` $-0.36$, finite, 43 moves |

## 3. Moves and search

$$\mathcal N(S) = \underbrace{\{S \setminus \{e\} : e \in S\}}_{\text{delete}}\ \cup\ \underbrace{\{S \cup \{e\} : e \notin S\}}_{\text{add; } e = (j,i),\ (i,j) \in S \text{ makes a 2-cycle}}\ \cup\ \underbrace{\{S \setminus \{(i,j)\} \cup \{(j,i)\} : (i,j) \in S,\ (j,i) \notin S\}}_{\text{reverse}}$$

$$S \leftarrow S^\star = \arg\min_{S' \in \mathcal N(S)} \mathrm{score}(S')\quad\text{iff}\quad \mathrm{score}(S^\star) < \mathrm{score}(S) - 10^{-9}\,\lvert \mathrm{score}(S)\rvert$$

```
greedy_search(S_0, max_steps=200):          # the runners pass p(p-1); SearchResult.capped: the cap stopped it
  S <- S_0
  repeat up to max_steps times:
    score every neighbour of S              # Scorer caches scores, not matrices: no support is refitted twice.
                                            # add_screen (iterative refits): score only the adds with the largest |∇L|
                                            # at the current refit; deletes and reverses always all scored. --add-screen
                                            # off in waves 5a, 8 (b): neighbourhood as wave 1, score differs; every add
                                            # with CovRefit costs 15 to 100 s per graph at p = 10. --add-screen 20 in
                                            # wave 8 (c), 100 starts with CovRefit: unscreened, longer than a task may run
    S* <- argmin over neighbours(S) of score
    if score(S*) < score(S) - 1e-9 |score(S)|:  S <- S*
    else:                                       stop        # no neighbour improves
  return S, M(S)
  # M(S): the matrix kept from the step that first scored S (code review A2b). A cache hit (typically reverse after
  #   add) is recomputed from the current warm start; CovRefit can reach another stationary point: lower -> replaces
  #   the cached score (still a strict descent); higher -> the cached score stands, counted in SearchResult.mismatches,
  #   stored as search_mismatches. DirectRefit: identical. A5: two start blocks of wave 8 (c), each with its own cache,
  #   scored 4 of 36 end graphs differently, by more than 1e-9, up to 7.8e-3
  # CovRefit at the 3,000-step cap (code review A2, 13 Oct 2026): on a support with a 2-cycle the likelihood need not
  #   have a maximiser; both entries grow while the loss falls (|M_ij| 460 -> 2,000 over 60,000 steps, score still
  #   falling). Score at the cap: 5 to 19 above the infimum at n = 1e3, one to three edges of BIC penalty -> a bias
  #   against such graphs in waves 5a, 8 (b), 8 (c). Stored wave 8 cells: 0 of 16 selected and searched supports
  #   capped; one full step: 7 of 106 neighbours, all adds that create a 2-cycle; tightening left the move unchanged.
  #   Scores left as they are
  # flags: Scorer.unconverged (capped refits); SearchResult.unconverged, .capped; per graph bic_unconverged,
  #   search_unconverged, search_capped, pure_*, truth_*

multistart_search(starts):                  # one shared cache; returns the best-scoring end graph
random_support(p, rng, max_density=0.3):    # "sparse", --starts sparse (default): d ~ U[0, 0.3] per graph, then every
                                            #   off-diagonal entry independently with probability d; 2-cycles allowed, no self-loops
                                            # "uniform", --starts uniform: probability 1/2 per entry, uniform over directed graphs
RESTART_SEED, run_search_shard.py:          # rng seeded with (RESTART_SEED, p, k, C, rep) per data set: the same starts
                                            #   at every n, under both C, whatever the shard layout; the empty graph appended last
```

## 3a. Side by side

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
    with  penalty = ½ (p + k) log n                         (BIC penalty; dimension p + k by their
                                                             expected-dimension theorem for simple mixed graphs)
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

```
gclm.solvers.search

Input:  sample covariance Σ̂ (or the population Σ, "n = ∞"), the volatility matrix C the path used
Graph class:  directed graphs on p nodes: any set S of off-diagonal entries of M,
              2-cycles allowed, no bidirected edges (C is known);
              allow_two_cycles=False: at most one edge per pair, the paper's simple graphs

score(S):
    M̂_S   <- argmin over M with zeros outside S ∪ diag of ½ ‖MΣ̂ + Σ̂Mᵀ + C‖²_F
              (least squares on the direct loss: normal equations, closed form;
               the likelihood refit CovRefit is --refit loglik, wave 5a)
    Σ(M̂_S) <- stationary covariance of M̂_S under C   (solve M Σ + Σ Mᵀ + C = 0)
    if M̂_S is not stable, or no finite refit exists:  return +∞
    return L(S) + pen(S)
    with  L(S)   = n [ log det Σ(M̂_S) + tr(Σ(M̂_S)⁻¹ Σ̂) ]      (loss term: -2 ℓ up to a constant)
          pen(S) = log(n) (p + |S|)                           (BIC penalty; dimension p + |S|: exact when the
                                                               support is identifiable, over-counts both
                                                               directions between two isolated nodes)
              or   log(n) (p + |S|) + 4 γ |S| log p           (eBIC penalty, Dettling's form; from wave 5c on,
                                                               --ebic-gamma; offline on the path: S4 §5)
              or   log(n) (p + |S|) + 2 γ log C(p(p-1), |S|)  (Chen & Chen's form; S3b, γ = 1)
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
                              (a mixture of Erdős–Rényi digraphs over densities 0 to 0.3: starts about as dense as
                               the true graphs, short searches; this repository's recipe from S3b and wave 2, the
                               bound 0.3 and the mixture not taken from a paper)
    random graph, "uniform":  each off-diagonal entry with probability 1/2  (wave 5b; uniform over the class;
                              the only published recipe among the references, [22])

Simulations: p = 10 and 20, 800 graphs per cell, n = 10³, 10⁴, ∞.
Bottom line: moves and hill-climbing are the paper's; the score is the GCLM analogue with a cheaper refit;
the starting graphs differ: 10 sparse ones against 300, or 100 uniform ones. R = 10 had not saturated
(S4 §4a), so the search-pure numbers of S4 bound the paper's method from below; wave 5b runs 100 of each kind.
```

### The starting graphs

| | Nowzohour et al. (2017) | paper (2020) | `--starts uniform` | `--starts sparse` (default) |
|---|---|---|---|---|
| class | bow-free acyclic path diagrams | simple mixed graphs | directed graphs, $2^{p(p-1)}$ | directed graphs, $2^{p(p-1)}$ |
| $P(e \in S)$ | uniform over the class by MCMC (Algorithm 1) | not stated | $\tfrac12$ independently: exactly uniform, no MCMC | $d \sim U[0, 0.3]$ per graph, then $d$ |
| mean edges, $p = 10$; $20$ | | | 45; 190 (truth: density $< 0.4$; $< 0.2$) | density $\le 0.3$ |
| $R$ | 100 | 300 | 100 (wave 5b) | 10 + $\varnothing$ (wave 2); 100 (wave 5b) |
| false positives, best of 5, `next_steps/051026/files/time_starts.txt` | | | 114 | 3 |
| wave 5b cell, $R = 100$ | | | `search100u_p10_Cresc` | `search100s_p10_Cresc` |

$$\text{the paper's class, uniform: each of the } \binom p2 \text{ pairs} \in \{\varnothing,\ i \to j,\ j \to i\} \text{ with probability } \tfrac13,\ \text{density } \tfrac23;\qquad \text{with 2-cycles: a fair coin per entry}$$

### The differences

| | paper (2020) | here |
|---|---|---|
| graphs | simple mixed: directed + bidirected, $\le 1$ edge per pair | directed, 2-cycles, no bidirected ($C$ known) |
| model | linear SEM, error covariance $\Omega$ | GCLM, $M\Sigma + \Sigma M^\top + C = 0$ |
| fit | $\max \ell$, block coordinate descent | least squares into $L$ |
| penalty | $\tfrac12(p+k)\log n$; $+\log(p^{2k}3^k)$ | $(p+\lvert S\rvert)\log n$; $+4\gamma\lvert S\rvert\log p$; $+2\gamma\log\binom{p(p-1)}{\lvert S\rvert}$ |
| acceptance, cap | strictly higher; $10^4$ | strictly lower; $p(p-1)$ |
| random starts | 300 ([22]: 100, uniform by MCMC) | 10 + $\varnothing$ (wave 2); 100 (wave 5b) |
| $p$; graphs; $n$ | 5, 6; 100; $\le 10^4$ | 10, 20; 800 per cell; $\le \infty$ |
| other starts | truth | truth; the score-selected support of every path |

## 4. Cost

| $p = 20$ | `DirectRefit` | `CovRefit` |
|---|---|---|
| candidates per step | $\approx 400$ | per candidate $\times 10$ to $100$ |
| M2, from $\varnothing$ | $\approx 4$ s | |
| M2, from a random graph | $\approx 20$ s | |
| from the score-selected graph | $> 100$ deletions; 1–2 min | |

## 4a. Cluster

```bash
# after a path: simulations/run_s1_shard.py, function select_graph
run_s1_shard.py --select bic           # bic_along_path scores every support of the path; stores the selected graph
run_s1_shard.py --select search        # then greedy_search from the selected graph
run_s1_shard.py --refit direct         # the default: least squares on the direct loss, for every loss, so that the
                                       # selection rule is the same for all estimators
run_s1_shard.py --refit loglik         # the maximised likelihood; wave 5a at p = 10
run_s1_shard.py --ebic-gamma 0.5 1     # wave 5c on: selects and searches once more per gamma on the same path;
                                       # fields ebic05_*, ebic1_* next to the BIC ones, the two penalties paired graph by
                                       # graph. Otherwise the BIC penalty
run_s1_shard.py --c-scale variance     # the rescaled C of docs/DENSE_START.md §2 in the score; otherwise 2I:
                                       # the score uses the volatility matrix the estimator was fitted with

# without a path: simulations/run_search_shard.py
run_search_shard.py --restarts 10      # wave 2: multistart_search from the empty graph and 10 random graphs,
                                       # greedy_search from the true graph as a ceiling; end graph and score of every start stored
run_search_shard.py --restarts 100     # wave 5b
run_search_shard.py --starts sparse    # the default; or --starts uniform
run_search_shard.py --start-blocks B   # B contiguous blocks of starts per graph, one (graph, block) pair per task,
                                       # for searches too long for one task: wave 8 (c), 100 starts with the likelihood refit
run_search_shard.py --ebic-gamma 1     # scores both of its searches with the eBIC penalty, Dettling's form
# both runners: max_steps raised from the default 200 to p(p-1); every accepted move lowers the score, so the search
# ends by itself and the limit is only a guard

# wave 6: rescoring a finished cell
simulations/rescore_shard.py           # re-selects and re-searches the stored supports of a cell's paths with the eBIC
                                       # penalty after rebuilding each data set from its seed, checked against the stored
                                       # BIC scores; no path recomputed; cells rescore1_<source cell>, overlaid by campaign.py
plot_campaign.py --rule ebic1          # every figure with the eBIC penalty in the score; figures_ebic1/
simulations/diagnostics/restarts.py    # best of the first r starts for every r <= 100; with campaign.py puts the start
                                       # blocks of a graph back together
```

## 5. Validation

```
tests/test_search.py                              the refits recover M* on the true support; DirectRefit satisfies the
                                                  normal equations; the penalties (BIC, both eBIC forms) and the unstable
                                                  cases; the direct-loss score; the neighbourhood moves; the score never
                                                  increases along a search; multistart keeps the best; the Example 2 reversal
simulations/diagnostics/search_study.py validate  Example 2: every support with up to 7 edges (direct loss) or up to 5
                                                  (log-likelihood) scored; the global optimum against the truth and against
                                                  the greedy search from every start; results in S3b §9
```

## 6. References

```
Amendola, Dettling, Drton, Onori, Wu (2020)  Structure learning for cyclic linear causal models. UAI 2020; §5, scores (16), (17)
Chen, Chen (2008)                            Extended Bayesian information criteria for model selection with large model spaces.
                                             Biometrika 95(3), 759-771
Chickering (2002)                            Optimal structure identification with greedy search. JMLR 3, 507-554
```
