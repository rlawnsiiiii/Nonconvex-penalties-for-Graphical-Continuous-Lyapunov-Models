# Greedy search over GCLM supports (`gclm.solvers.search`)

The score-based search of study S3b (`simulations/S3b_reversal_search.md`). It follows the greedy
search of Améndola, Dettling, Drton, Onori & Wu (2020), *Structure learning for cyclic linear causal
models*, §5, adapted to graphical continuous Lyapunov models. Tests: `tests/test_search.py`.

## 1. The search space

A state is a **support** $S$: the set of off-diagonal entries of $M$ allowed to be nonzero. The
diagonal is always free. The edge $i \to j$ is the entry $M_{ji}$.
- **2-cycles are allowed by default.** A GCLM can have them, and Dettling, Drton & Kolar's model
  and simulation do: the edge $i \to j$ is in the graph iff $M_{ji} \ne 0$, every off-diagonal
  entry is drawn independently as $\omega_{ij}\varepsilon_{ij}$ with $\omega_{ij} \sim
  \mathrm{Bernoulli}(d)$, $d = k/p$ (their Section 5, reproduced by `gclm.data.simulate`), so a
  pair carries a 2-cycle with probability $d^2$ and a fraction $d$ of the true edges sits in one:
  10 to 40 % at $p = 10$, 5 to 20 % at $p = 20$.
- **The paper's restriction to simple graphs** (at most one edge per pair) is the
  `allow_two_cycles=False` switch.
- **Unlike the paper there are no bidirected edges.** The volatility $C = 2I$ is known.

## 2. Scoring a support

**Terms.** Supports are compared by their **score**, the sum of a **loss** and a **penalty**:

$$\mathrm{score}(S) = L(S) + \mathrm{pen}(S), \qquad
L(S) = n\big[\log\det\Sigma_S + \operatorname{tr}(\Sigma_S^{-1}\hat\Sigma)\big],$$

with $\Sigma_S = \Sigma(\hat M_S)$ the stationary covariance of the refit $\hat M_S$ of step 1 below.
The loss term $L$ is $n$ times the log-likelihood loss of `docs/LIKELIHOOD.md` at $\hat M_S$, i.e.
$-2$ times the Gaussian log-likelihood up to a constant (Dettling, Drton & Kolar, eq. 6.1). **BIC
and eBIC name only the penalty**; the lowest score wins.

| penalty | $\mathrm{pen}(S)$ | where |
|---|---|---|
| BIC | $(p + \lvert S\rvert)\log n$ | the default ($\gamma = 0$) |
| eBIC, Dettling's form | $(p + \lvert S\rvert)\log n + 4\gamma\lvert S\rvert\log p$ (his eq. 6.2) | the campaign from wave 5c on (`--ebic-gamma`) |
| eBIC, Chen & Chen's form | $(p + \lvert S\rvert)\log n + 2\gamma\log\binom{p(p-1)}{\lvert S\rvert}$ | S3b, $\gamma = 1$ |

The part by which an eBIC penalty exceeds the BIC penalty is the **eBIC term**. Dettling calls the
whole sum of his eq. (6.2) "EBIC$_\gamma$"; here that sum is the score with the eBIC penalty.

**Names in the code.** They are older than these terms and are kept, because the stored results
and the command-line flags use them: `bic()` computes the score with the loss term $L$,
`bic_direct()` the direct-loss score of §2a, `bic_along_path()` the scores of the supports of a
path; `--select bic` selects by the score; `--ebic-gamma` sets the $\gamma$ of the eBIC penalty;
the fields `bic_*` describe the graph selected with the BIC penalty, `ebic05_*` and `ebic1_*` the
one selected with the eBIC penalty ($\gamma = 0.5$, 1).

1. **Refit** $M$ without penalty on $S \cup \mathrm{diag}$:

   | loss | refit | class |
   |---|---|---|
   | direct, $\tfrac12\lVert A(\hat\Sigma)\operatorname{vec}(M) + \operatorname{vec}(C)\rVert^2$ | least squares on the columns of $A$ in $S \cup \mathrm{diag}$, via the normal equations on the precomputed Gram matrix ($\texttt{lstsq}$ as fallback) | `DirectRefit` |
   | log-likelihood or Frobenius, as in docs/LIKELIHOOD.md | the package's APG solver (`solvers.covariance.solve`) with weight 1 and $\lambda = 10^8$ outside the support (those entries stay exactly zero) and weight 0 on it; warm-started from the current estimate when that is stable, otherwise from the diagonal fit; tolerance $10^{-9}$ | `CovRefit` |

2. **Score** the implied covariance $\Sigma(\hat M_S)$, the stationary covariance from
   $M\Sigma + \Sigma M^\top + C = 0$:

   $$\mathrm{score}(S) = n\big[\log\det\Sigma(\hat M_S) + \operatorname{tr}(\Sigma(\hat M_S)^{-1}\hat\Sigma)\big] + \mathrm{pen}(S).$$

   With the BIC penalty this is $-2n$ times the paper's score (16) up to a constant: their penalty
   is $\tfrac12(p+k)\log n$, with model dimension $p + k$.
   - **The eBIC term** in Chen & Chen's (2008) form, $2\gamma\log\binom{p(p-1)}{|S|}$, is with
     $\gamma = 1$ the analogue of the paper's increased penalty (17), whose $\log(p^{2k}3^k)$ counts
     simple mixed graphs. Here the count is of directed graphs with $k$ edges among $p(p-1)$
     possible ones. `bic(..., ebic_form)` has both forms of the term: `"binom"`, Chen & Chen's, used
     in S3b with $\gamma = 1$; and `"dettling"`, his eq. (6.2), $4\gamma|S|\log p$, the form the
     campaign uses: offline on the supports of a path (`campaign.py`, $\gamma = 0.5, 1$, columns
     `ebic05_*`, `ebic1_*`) and, from wave 5c on, inside the selection and the search
     (`--ebic-gamma`, columns `ebic1_search_*`). Waves 1, 2, 5a and 5b run with $\gamma = 0$, the
     BIC penalty. For sparse
     graphs the two forms are close: $\log\binom{N}{k} = k\log(N/k) + k - O(\log k)$ with
     $N = p(p-1) \approx p^2$, against $2k\log p = k\log p^2$, so Dettling's is the larger by about
     $k(\log k - 1)$.
   - **Why the $-2\ell$ scale.** The paper maximises $\tfrac1n[\ell - \tfrac12(p+k)\log n]$; times
     $-2n$ that is $-2\ell + (p+k)\log n$, our score with the BIC penalty plus the constant
     $np\log 2\pi$. The ordering
     of graphs, the greedy moves and the local optima are the same; the $-2\ell$ scale is the one
     Schwarz's BIC penalty, Chen & Chen's term and Dettling's eq. (6.2) are written in, so these
     penalties are added literally.
   - **Stability is part of the model.** A refit with an eigenvalue whose real part is $\ge 0$ has
     no stationary covariance, so it scores $+\infty$. The direct-loss refit can produce such an
     $M$; the covariance-loss solver accepts only stable iterates.
   - **$n = \infty$** ($\hat\Sigma = \Sigma$ exactly) uses a nominal $n = 10^6$ in the loss term
     and in the penalty. The score then prefers exact fits first and fewer edges second: the
     $\ell_0$ target.
   - **A singular refit.** The Gram matrix $A_S^\top A_S$ is singular when a nonzero $M$ supported
     in $S$ solves $M\hat\Sigma + \hat\Sigma M^\top = 0$. At $n = \infty$ this happens for real: two
     nodes $i, j$ that are isolated in the true graph have $\hat\Sigma_{ij} = 0$ exactly, and then
     the columns of $A$ for $M_{ij}$ and $M_{ji}$ are identical, so any support holding both
     directions of such a pair (the random starts of the pure search do) is rank deficient. The
     refit then falls back to least squares on the columns (minimum-norm solution), with SciPy's
     QR-based driver as a second fallback, because LAPACK's SVD-based one failed to converge on
     one such matrix on the cluster (campaign, 5 October); if nothing works the support scores
     $+\infty$ and the search skips it. At finite $n$ the columns are only nearly dependent.

## 2a. From a path to one graph: oracle $\lambda$, score, search

A regularisation path gives 100 graphs per dataset, one per $\lambda$. Three ways to turn it into
one graph are used in the thesis, and the campaign of October 2026 records all three for every
estimator (`run_s1_shard.py --select search`; campaign note §3.3):

| | how the graph is chosen | uses the truth | column |
|---|---|---|---|
| **oracle $\lambda$** | every graph on the path is compared with the true graph; the best $F_1$ is reported | yes | `max_f1` (Dettling's Figure 5; likewise `max_acc`, and `auc` / `aupr`, which summarise the whole path) |
| **selected by the score** | every support on the path is refitted as in §2 (least squares on the direct loss) and scored, with the BIC penalty unless stated; the lowest score wins (`bic_along_path`) | no | `bic_f1` |
| **after the search** | the selected graph is the start of the greedy search of §3 | no | `search_f1` |

The oracle value says whether a good graph is *on* the path; the other two say whether one *gets*
it without the truth. They can disagree: at $p = 10$, $n = 10^3$ the dense → sparse MCP path gains
+0.016 in `max_f1` over the lasso and nothing in `bic_f1`.

**Why refit, and why by least squares.** The estimate at a given $\lambda$ is shrunk by the
path's penalty, so its likelihood says little about the *support*. The score therefore evaluates
each support at the best unpenalised fit on it: $M$ restricted to the support plus the diagonal,
minimising the direct loss $\tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2$, whose residual is
linear in $M$, so an ordinary least-squares problem (`DirectRefit`). That refit is then plugged
into the loss term $L$ of §2, the Gaussian log-likelihood. Two consequences: the same selection
rule applies to every estimator and every loss, so only the paths differ; and the refit minimises
the direct loss rather than maximising the likelihood, so $L$ is the likelihood *at the
least-squares fit*, not the maximised likelihood. At $n = \infty$ with the correct $C$ the two coincide on the true support (both fit
exactly); at finite $n$ they differ slightly. The `CovRefit` of §2 is the likelihood version; it was
used in S3b for the covariance losses and is the runner's option `--refit loglik` (wave 5a of the
campaign, cells `direct_<estimator>-ml_<C>`), at about 0.05 to 0.1 s per support at $p = 10$
against microseconds for least squares.

**Where a loss enters a search over graphs.** A score-based search has no $\lambda$ and no penalty
on $M$, but it still needs a fitted model for every graph it looks at: the score of a graph $S$ is
the loss term, $-2$ times the log-likelihood of *the best model with that graph*, plus the penalty,
and "the best model with that graph" is an estimate of $M$ on the support $S$. So every score is
two steps, and a loss enters both: the fit, and the loss term.

1. **fit** $M_S$ on the support $S \cup \mathrm{diag}$, by one of
   - least squares on the direct loss (`DirectRefit`; closed form, microseconds; the default), or
   - maximising the Gaussian likelihood (`CovRefit`; iterative, 0.05 to 1 s; `--refit loglik`);
2. **score** the fitted model: $\mathrm{score}(S) = L(S) + \mathrm{pen}(S)$ with
   $L(S) = n[\log\det\Sigma_S + \mathrm{tr}(\Sigma_S^{-1}\hat\Sigma)]$, $\Sigma_S$ the covariance
   of $M_S$, and the BIC penalty $(p+|S|)\log n$ or the eBIC penalty; the search compares graphs
   only by this number.

The meeting notes (`plan.md`, Simulations) name two scores, "quadratic loss + BIC-type penalty" and
"likelihood + BIC-type penalty". The second is the likelihood loss term with the likelihood fit
(`--refit loglik`, waves 5a and 8), as in Améndola et al. and Dettling et al. The campaign's
default is in between: fitted by the quadratic loss, scored by the likelihood loss term.

**The direct-loss score.** The first of the two, with the quadratic loss itself as the loss term,
is `Scorer(..., score="direct")` (`bic_direct`, 9 October; tests in `tests/test_search.py`):

$$\mathrm{score}_{\mathrm{dir}}(S) = N\log(\mathrm{RSS}_S/N) + (p+|S|)\log n, \qquad
N = \tfrac{p(p+1)}{2},\quad \mathrm{RSS}_S = \|M_S\hat\Sigma + \hat\Sigma M_S^\top + C\|_F^2,$$

the score of a Gaussian regression with unknown error variance on the $N$ distinct equations of
the Lyapunov system, with the BIC penalty. It is not a likelihood of the data: the $N$ residuals
are neither independent nor of equal variance, and their number does not grow with $n$. A support
with $p + |S| \ge N$ free entries generically fits the $N$ equations exactly: its RSS is zero up to
rounding (floored at $10^{-14}\|C\|_F^2$, `RSS_FLOOR`), so its loss term lies far below that of
every graph that does not fit exactly, and the score pulls toward dense graphs at every finite $n$.
At $n = \infty$ the true support already fits exactly and the penalty decides among exact fits.

**Checked on 32 graphs** (10 October; $p = 10$, $C = 2I$, $k = 1, \dots, 4$, the four true $C$, two
replicates; the lasso path of the direct loss; every score with the BIC penalty;
`next_steps/091026/score_check.py` and `.txt`):

| $n$ | loss term of the score | from the truth: $F_1$ (edges) | selected on the path | after the search | s per graph |
|---|---|---|---|---|---|
| $10^3$ | likelihood, least-squares fit (the campaign's) | 0.684 (17.8) | 0.515 (23.3) | 0.482 (17.0) | 0.8 |
| $10^3$ | likelihood, likelihood fit (Améndola, Dettling) | 0.703 (17.4) | 0.510 (22.8) | 0.478 (16.5) | 565 |
| $10^3$ | direct loss, $N\log(\mathrm{RSS}_S/N)$ | 0.679 (35.0) | 0.426 (45.0) | 0.426 (45.0) | 0.5 |
| $10^4$ | likelihood, least-squares fit | 0.788 (23.7) | 0.565 (29.5) | 0.558 (23.4) | 0.6 |
| $10^4$ | likelihood, likelihood fit | 0.801 (23.5) | 0.566 (29.3) | 0.560 (23.2) | 623 |
| $10^4$ | direct loss | 0.747 (31.0) | 0.440 (45.0) | 0.440 (45.0) | 0.4 |

The true graphs have 23.7 edges on average.
- **The direct-loss score is degenerate.** On the path it selects the densest graph, the lasso's
  dense end with $p(p-1)/2 = 45$ edges, in every one of the 64 graphs, and the search stays there.
  From the true graph it adds 7 to 11 edges. It is not used in the campaign.
- **The likelihood fit and the least-squares fit select the same graphs.** The selected and the
  searched graphs differ by at most 0.005 in $F_1$; from the truth the likelihood fit is 0.01 to
  0.02 higher. It costs 700 to 1,000 times as much. This is wave 5a's result again, on other data.

**When the least-squares refit is unstable.** The least-squares fit of a sparse support is not
entry by entry: the off-diagonal residuals $(M_{ii} + M_{jj})\hat\Sigma_{ij}$ couple the diagonal
entries, and with the rescaled $C$, whose diagonal varies by an order of magnitude, the compromise
can put a diagonal entry above zero. Then $M$ is unstable, the score is $+\infty$, and a search started
there cannot leave unless a single move makes the refit stable. In wave 2 this happens to the
**empty graph** in 6 to 7 % of the graphs at $p = 10$ and 24 to 25 % at $p = 20$ (rescaled $C$;
never with $C = 2I$; most under `C_Random_Min_Diag` and `C_Random_Full`), and to the **true
support** in 2 to 5 %, almost all `C_Random_Full`. The best of the 11 starts was dead for one graph
of 2 375. The likelihood refit maximises over stable $M$ only (the covariance solver accepts no
unstable iterate), so it has no dead starts: on the first such $p = 20$ graph the least-squares
refit of the empty graph has an eigenvalue at $+0.11$, the likelihood refit has its largest real
part at $-0.36$ and a finite score, and a search with the likelihood refit from the empty graph makes 43
moves (`next_steps/051026/files/time_starts.py` for the dataset).

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
- **With an iterative refit** (the covariance losses; the likelihood refit of wave 5a),
  `add_screen` can restrict the add moves scored per step to the ones with the largest
  $\lvert\nabla L\rvert$ at the current refit; deletes and reverses are always all scored. The
  campaign does not use it (`--add-screen` is off; scoring every add move with the likelihood refit
  costs 15 to 100 s per graph at $p = 10$), so the two searches of wave 1 and wave 5a differ in the
  score only, not in the neighbourhood.

**Starts:**

| function | start |
|---|---|
| `greedy_search` | one given support, e.g. a penalised path's support at the $\lambda$ selected by the score (`bic_along_path`) |
| `multistart_search` | several supports with one shared cache; the best-scoring result is returned. The pure search uses randomly drawn graphs plus the empty graph. **How a start is drawn** (`random_support`, "sparse"): one edge probability $d \sim U[0, 0.3]$ for the graph, then every off-diagonal entry independently with probability $d$ (2-cycles allowed, no self-loops); "uniform" (`--starts uniform`): every off-diagonal entry with probability $\tfrac12$, i.e. uniform over all directed graphs on $p$ nodes. The generator is seeded per data set with `(RESTART_SEED, p, k, C, rep)` in `run_search_shard.py`, so the starts of a data set are the same at every $n$, for both $C$ and whatever the shard layout |

## 3a. Side by side: the search of Améndola et al. (2020) and the search here

Both are the same algorithm in outline, greedy hill-climbing over graphs with a score = likelihood
loss term + penalty and single-edge moves. The differences are in the graph class, the score, the number and the
distribution of the randomly drawn starting graphs, and in how the likelihood is maximised.

### The paper's method (Améndola, Dettling, Drton, Onori & Wu 2020, §5–6)

```
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
```

How the 300 random starting graphs are drawn is not stated; the MCMC of their §6.1 draws the
*true* graphs of the simulation uniformly over simple mixed graphs, "in analogy to [22]".
Their reference [22], Nowzohour, Maathuis, Evans & Bühlmann (2017), draws the *starting graphs*
uniformly over the graph class by an MCMC algorithm (their Algorithm 1; uniform sampling of
constrained graph classes is not trivial) and uses R = 100 of them. Their simulations have
p = 5 and 6, 100 graphs, n = 10², 10³, 10⁴.

### The method here (`gclm.solvers.search`)

```
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

### The starting graphs: Nowzohour et al. (2017) adapted to this class

Nowzohour, Maathuis, Evans & Bühlmann (2017) start their greedy search from $R = 100$ graphs
drawn *uniformly* over their model class, bow-free acyclic path diagrams. A uniform draw over a
constrained class is not trivial, because the constraints (acyclicity, no bow) are global, so they
use an MCMC sampler (their Algorithm 1) whose stationary distribution is uniform over the class.
Améndola et al. (2020) use the same sampler to draw the *true* graphs of their simulation and do
not say how their 300 starting graphs are drawn.

**The adaptation.** The class searched here is all directed graphs on $p$ nodes: any set $S$ of
off-diagonal entries of $M$, with 2-cycles allowed, no self-loops (the diagonal is always free)
and no bidirected edges ($C$ is known). There are $2^{p(p-1)}$ such graphs and no global
constraint, so the uniform distribution factorises: every off-diagonal entry is in $S$
independently with probability $\tfrac12$. That is `--starts uniform` in `run_search_shard.py`,
and it is exactly uniform over the class, with no MCMC and no burn-in to argue about. Its expected
density is $\tfrac12$, i.e. 45 edges at $p = 10$ and 190 at $p = 20$, against true graphs with
density below 0.4 and 0.2. Two remarks:

- Had the class been Améndola's "at most one edge per pair" (no 2-cycles), uniform would mean
  drawing each of the $\binom p2$ pairs from $\{\text{none}, i \to j, j \to i\}$ with probability
  $\tfrac13$ each, i.e. density $\tfrac23$ of the pairs. The search here allows 2-cycles, so the
  fair coin per entry is the right analogue.
- Uniform over the class is what the reference does, not necessarily what works: on the one
  $p = 20$ graph timed (`next_steps/051026/files/time_starts.txt`) the best of five uniform starts
  ended at a dense local optimum with 114 false positives, the best of five sparse starts at 3.

**The sparse recipe** (`random_support`, `--starts sparse`, the default and the choice of S3b
and wave 2) is this repository's own: one edge probability $d \sim U[0, 0.3]$ per starting graph,
then every off-diagonal entry independently with probability $d$. It is a mixture of Erdős–Rényi
digraphs over densities from 0 to 0.3, chosen so that the starts are about as dense as the true
graphs and the searches short; the bound 0.3 and the mixture are not taken from a paper.

**Seeding.** Both recipes draw from a generator seeded with `(RESTART_SEED, p, k, C, rep)`, so a
data set has the same starting graphs at every $n$, under both $C$ and whatever the shard layout;
the empty graph is appended last. **Wave 5b** runs both recipes with $R = 100$ on the same data
sets (`search100s_p10_Cresc`, `search100u_p10_Cresc`), and `simulations/diagnostics/restarts.py`
reads off the best of the first $r$ starts of either for every $r \le 100$.

### The differences

| | Améndola et al. 2020 | here |
|---|---|---|
| graph class | simple mixed graphs: directed and bidirected edges, at most one edge per pair | directed graphs: 2-cycles allowed, no bidirected edges ($C$ known) |
| model | linear SEM with error covariance $\Omega$ (bidirected edges) | GCLM, $M\Sigma + \Sigma M^\top + C = 0$ |
| maximised fit in the score | Gaussian likelihood, by block coordinate descent | least squares on the direct loss (closed form), plugged into the Gaussian likelihood; the likelihood refit (`CovRefit`) is the `--refit loglik` of wave 5a |
| dimension in the penalty | $p + k$, $k$ = number of edges (justified by their expected-dimension theorem for simple mixed graphs) | $p + |S|$; exact when the support is identifiable, over-counts for both directions between two isolated nodes |
| penalties | BIC; "increased" $+\log(p^{2k}3^k)$ | BIC; eBIC in Chen & Chen's form $+2\gamma \log\binom{p(p-1)}{|S|}$ (S3b, $\gamma = 1$) and in Dettling's form $+4\gamma|E|\log p$: offline on the path (S4 §5) and inside the selection and the search (wave 5c, `--ebic-gamma`) |
| moves | add / remove / reverse one edge | add / delete / reverse one entry: the same |
| acceptance | strictly higher score | strictly lower score (sign convention only) |
| iteration cap | $10^4$ | $p(p-1)$, never reached |
| randomly drawn starting graphs | 300 per data set (reference [22]: 100, uniform over the class via MCMC) | **10** plus the empty graph in wave 2; **100** in wave 5b |
| distribution of the starting graphs | not stated; uniform via MCMC in [22] | wave 2: edge probability $U[0, 0.3]$ per graph, i.e. sparser than uniform; this repository's own recipe from S3b, chosen so that the starts are about as dense as the true graphs and the searches short, not taken from a paper; wave 5b: that, and the uniform draw (a fair coin per entry, density 0.5; for this class no MCMC is needed), which is the only published recipe among the references |
| problem size | $p = 5, 6$; 100 graphs; $n \le 10^4$ | $p = 10, 20$; 800 graphs per cell; $n \le \infty$ |
| other starts | the true graph | the true graph; and the graph selected by the score on every estimator's path |

The moves and the hill-climbing are the paper's; the score is the GCLM analogue of theirs with a
cheaper refit; the starting graphs are the point where the two part: 10 sparse ones here against
300 (or 100 uniform ones), and 10 had not saturated (S4 §4a). The `search-pure` numbers of S4 are
therefore a lower bound on the paper's method in this setting; wave 5b runs 100 of each kind.

## 4. Cost

At $p = 20$ a step scores about 400 candidates. On the M2 a direct-loss search takes about 4 s from
the empty graph and about 20 s from a random graph. The covariance-loss refits are about 10–100×
more expensive per candidate. From a dense start it takes much longer: a graph selected by the score at
$p = 20$ can be more than a hundred deletions away from where the search ends, which is one to two
minutes.

## 4a. On the cluster (campaign of October 2026)

- **After a path:** `simulations/run_s1_shard.py --select bic` scores every support of the path
  (`bic_along_path`) and stores the graph selected by the score; `--select search` then runs
  `greedy_search` from it. Function `select_graph`. The refit is the direct-loss one for every
  loss, so that the selection rule is the same for all estimators.
- **Without a path:** `simulations/run_search_shard.py` runs `multistart_search` from the empty
  graph and `--restarts` randomly drawn graphs (10 in wave 2, 100 in wave 5b; `--starts sparse` or
  `uniform`), and `greedy_search` from the true graph as a ceiling. The graph and the score every
  start ends at are stored, and `simulations/diagnostics/restarts.py` reads off the best of the
  first $r$ starts for every $r$. With `--start-blocks B` the starts of every graph are split into
  $B$ contiguous blocks, one (graph, block) pair per task, for searches too long for one task
  (wave 8 (c): 100 starts with the likelihood refit); `restarts.py` and `campaign.py` put the
  blocks of a graph back together.
- **The refit behind the score** is least squares on the direct loss for every loss (`--refit
  direct`, the default), or the maximised likelihood (`--refit loglik`, wave 5a at $p = 10$).
- **The eBIC penalty** is off (the BIC penalty is used) except from wave 5c on:
  `run_s1_shard.py --ebic-gamma 0.5 1` selects and searches once more per $\gamma$ on the same path
  (fields `ebic05_*`, `ebic1_*` next to the ones with the BIC penalty, so the two penalties are
  paired graph by graph) and `run_search_shard.py --ebic-gamma 1` scores both of its searches with
  it; Dettling's form.
- **Rescoring a finished cell** (wave 6, `simulations/rescore_shard.py`): the stored supports of a
  cell's paths are re-selected and re-searched with the eBIC penalty, after rebuilding each data
  set from its seed (checked against the stored scores with the BIC penalty), so no path is
  recomputed. Cells `rescore1_<source cell>`; `campaign.py` overlays them on the source cell's
  rows, and `plot_campaign.py --rule ebic1` draws every figure with the eBIC penalty in the score
  (`figures_ebic1/`).
- **The volatility matrix** in the score is the one the estimator was fitted with: $2I$, or the
  rescaled $C$ of `docs/DENSE_START.md` §2 (`--c-scale variance`).
- **The step limit** `max_steps` is raised from the default 200 to $p(p-1)$ in both runners. Every
  accepted move lowers the score, so the search ends by itself; the limit is only a guard.

## 5. Validation

- **Unit tests** (`tests/test_search.py`): the refits recover $M^*$ on the true support; the
  direct refit satisfies the normal equations; the score's penalties (BIC, both eBIC forms) and
  unstable cases; the direct-loss score; the
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
