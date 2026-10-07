the# S4 — The campaign: every estimator with $C = 2I$ and with the rescaled $C$, at scale

*Results of the cluster campaign planned in
[`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md)
(the plan, the reasons, the code map). Status on 7 October 2026: wave 1 complete, wave 2 complete
but for one shard, wave 3 (log-likelihood loss) half complete, wave 4 (larger $p$) started; wave 5
(the BIC with the likelihood refit and the extended BIC inside the search, §5; 100 starting graphs
for the pure search, §4a) implemented on 7 October and not yet submitted. The sections on waves 3 and 4 are preliminary and marked as such. Numbers:
`runs/campaign/campaign_{means,paired,per_dataset}.csv` from `simulations/diagnostics/campaign.py`;
figures: `simulations/diagnostics/plot_campaign.py` → `runs/campaign/figures/`.*

---

## 0. Summary

The question of the thesis is whether MCP and SCAD recover the graph of a GCLM better than the
lasso. On 800 graphs per cell (Figure 5's generator, $p = 10, 20$, $n = 10^3, 10^4, \infty$, all four
settings of the true $C$), the answer is:

1. **Used the standard way, no.** MCP loses to the lasso by 0.04 to 0.12 in `max_f1` in every cell,
   SCAD by 0.03 to 0.07 with $C = 2I$; the gap grows with $n$. (As in S2 and S2b.)
2. **Started from the dense lasso solution and pruned, yes.** MCP and SCAD run dense → sparse beat
   the lasso in every cell except one tie, by up to +0.146 at $p = 20$, $n = \infty$ with the rescaled
   $C$ and +0.086 with $C = 2I$. The gain grows with $p$ and with $n$.
3. **The gain does not need the nonconvex penalty.** The adaptive lasso, which starts from the same
   dense solution and prunes with a convex step, matches MCP dense → sparse in `max_f1` (differences
   of 0.00 to 0.045, in its favour) and is better on everything else: `aupr` by +0.08 to +0.14, the
   BIC-selected graph by +0.02 to +0.08. The message is "start from the lasso, then prune"; MCP and
   SCAD are one way to do it, the adaptive lasso the best of the ways tried.
4. **The gain holds for every diagonal true $C$,** known or not: `C_ID`, `C_Random_Min_Diag` and
   `C_Random_Diag` show gains of the same size. For the non-diagonal `C_Random_Full` there is none
   at $p = 10$ and a small one at $p = 20$.
5. **The correctly specified $C$ roughly doubles the gain** (verdict 6 of `VERDICTS.md`, mechanism
   in `docs/DENSE_START.md` §7), and it is the only setting in which the BIC-selected graph of the
   dense-start estimators clearly beats the lasso's at $n = 10^3$.
6. **Without an oracle $\lambda$:** the BIC-selected graphs keep the ranking, with smaller gains at
   $n = 10^3$. The BIC search then equalises almost everything: it lifts the lasso by +0.09 to +0.14
   at $p = 20$ and the dense-start estimators by less, so all lasso-based starts end within 0.02 of
   each other, the adaptive lasso on top. The greedy search from random starts is 0.02 to 0.07
   worse; the greedy search from the truth is 0.10 to 0.24 better, so most of the remaining gap is in the
   search, not in the paths.
7. **Preliminary, wave 3:** on the log-likelihood loss MCP dense → sparse does *not* beat the lasso
   at finite $n$ (−0.05 to −0.07 with $C = 2I$, 0.00 and −0.04 with the rescaled $C$), only at
   $n = \infty$ with the rescaled $C$ (+0.02). Its dense start is the exact fit $-\tfrac12 C\hat\Sigma^{-1}$,
   not the lasso's dense end; see §6 for why that may be the difference.

Against the expectations written down before the run (campaign note §3.5): five of seven held,
one was too cautious (`C_Random_Diag` gains as much as `C_ID`) and one was wrong in size (the
search helps the lasso far more than expected). Details in §7.

---

## 1. What was run

| | |
|---|---|
| graphs | Figure 5's generator with the seeds of the n-sweep: $p \in \{10, 20\}$, $k = 1..4$, four true $C$, 25 replicates; 800 graphs per cell, the same in every cell |
| data | standardised; $n = 10^3, 10^4, \infty$ (population covariance) |
| $C$ in the fit | `C2I`: $C = 2I$ (Dettling's pipeline); `Cresc`: $C = 2\,\mathrm{diag}(1/s_i^2)$, the rescaled $C$ (`docs/DENSE_START.md` §2) |
| estimators, wave 1 (direct loss) | lasso; MCP and SCAD on the standard path (sparse → dense); MCP and SCAD dense → sparse (`--direction up`); MCP and SCAD by LLA (`--method lla`); adaptive lasso (`--method adaptive`). Definitions: `docs/DENSE_START.md` §3–§5 |
| per graph and estimator | the path (100 $\lambda$); its `max_f1`, `auc`, `aupr`; the BIC-selected graph; that graph after the greedy BIC search; Dettling's eBIC selection, computed afterwards from the stored scores (`docs/SEARCH.md` §2a) |
| wave 2 | the greedy search from random starts (empty graph + 10 random graphs; the method of Améndola et al. 2020, §4a) and the same search started from the true graph, $p = 10$ with both $C$, $p = 20$ with the rescaled $C$ |
| wave 3 | log-likelihood loss, $p = 10$: lasso and MCP, both path orders, both $C$ |
| wave 4 | $p = 15 \dots 50$ at $n = 1000$: running |
| cost | 915 CPU-h so far (wave 1: 600) |

**Validity.** The three cells that repeat the n-sweep (direct lasso / MCP / SCAD, $C = 2I$)
reproduce `runs/nsweep_p10-20` on all 800 graphs at every $n$, zero differences
(`campaign_baseline_check.txt`). One task failed and was rerun (a singular refit at $n = \infty$,
`docs/SEARCH.md` §2); `search_p20_Cresc_ninf` has 375 of 400 graphs until it returns.

---

## 2. The 2 × 2: choice of $C$ × order of the path

![](../runs/campaign/figures/twobytwo_max_f1.png)

*`max_f1` (the best graph on the path) against $n$; rows $p = 10, 20$, columns $C = 2I$ and the
rescaled $C$. Error bars: standard errors over 400 graphs per point. Hollow: the standard path.*

Paired difference to the lasso fitted with the same $C$ (`max_f1`, 400 graphs per cell, $z$ in
brackets):

| | $p = 10$, $n = 10^3$ | $10^4$ | $\infty$ | $p = 20$, $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|---|---|---|
| MCP standard, $C = 2I$ | −0.096 (−19) | −0.110 (−21) | −0.117 (−22) | −0.076 (−20) | −0.091 (−21) | −0.102 (−22) |
| MCP standard, rescaled $C$ | −0.074 (−16) | −0.092 (−17) | −0.094 (−17) | −0.043 (−12) | −0.052 (−11) | −0.072 (−14) |
| SCAD standard, rescaled $C$ | −0.034 (−8) | −0.038 (−8) | −0.042 (−8) | +0.002 (+1) | +0.017 (+5) | +0.009 (+2) |
| MCP dense → sparse, $C = 2I$ | −0.009 (−2) | +0.012 (+2) | +0.026 (+6) | +0.018 (+4) | +0.060 (+14) | +0.086 (+22) |
| MCP dense → sparse, rescaled $C$ | +0.016 (+3) | +0.045 (+8) | +0.058 (+11) | +0.056 (+12) | +0.115 (+23) | +0.146 (+30) |
| SCAD dense → sparse, rescaled $C$ | +0.021 (+5) | +0.048 (+10) | +0.057 (+12) | +0.038 (+12) | +0.092 (+20) | +0.129 (+27) |
| MCP by LLA, rescaled $C$ | +0.011 (+3) | +0.033 (+9) | +0.048 (+11) | +0.023 (+10) | +0.080 (+23) | +0.112 (+26) |
| adaptive lasso, rescaled $C$ | +0.020 (+4) | +0.039 (+7) | +0.047 (+9) | +0.080 (+15) | +0.125 (+26) | +0.149 (+32) |

- **The standard path loses everywhere,** by more as $n$ grows: the failure is not a small-sample
  effect (S2b). The rescaled $C$ narrows the gap but does not close it; for SCAD at $p = 20$ it
  turns into a tie.
- **The dense start wins,** and the two factors add. With $C = 2I$ the dense → sparse paths tie the
  lasso at $p = 10$, $n = 10^3$ and gain up to +0.086 at $p = 20$, $n = \infty$; the rescaled $C$ adds
  about as much again (+0.146). Both gains grow with $p$ and with $n$.
- **By density** (MCP dense → sparse, rescaled $C$, diagonal true $C$): at $p = 10$, $n = 10^3$ the
  gain is +0.11 for $k = 1$, +0.05 for $k = 2$, and −0.02 / −0.03 for $k = 3, 4$; at $p = 20$,
  $n = 10^4$ it is +0.18 / +0.18 / +0.13 / +0.06. The dense start helps most where the graph is
  sparse relative to $p$, and only there at the smallest sample size.

## 3. Does it hold beyond `C_ID`?

![](../runs/campaign/figures/by_true_c_max_f1.png)

*Paired difference to the lasso (same $C$) in `max_f1`, rescaled $C$, one column per setting of the
true $C$. 100 graphs per point.*

| MCP dense → sparse − lasso, rescaled $C$ | $p = 10$: $10^3$ / $10^4$ / $\infty$ | $p = 20$: $10^3$ / $10^4$ / $\infty$ |
|---|---|---|
| `C_ID` (true $C = 2I$; the rescaled $C$ is exact) | +0.026 / +0.068 / +0.095 | +0.081 / +0.154 / +0.210 |
| `C_Random_Min_Diag` (diagonal, $U[2, 4]$) | +0.027 / +0.071 / +0.083 | +0.062 / +0.141 / +0.185 |
| `C_Random_Diag` (diagonal, $U[0.5, 4]$) | +0.025 / +0.051 / +0.056 | +0.063 / +0.122 / +0.155 |
| `C_Random_Full` (non-diagonal) | −0.013 / −0.010 / −0.003 | +0.016 / +0.042 / +0.036 |

The claim extends to any diagonal true $C$: the rescaled $C$ is only partly right in the two random
settings (it removes 95 % and 40 % of the misfit, campaign note §2.2), yet the gains are of the same
size as for `C_ID`. With a non-diagonal true $C$ there is nothing to gain at $p = 10$ and a little at
$p = 20$. With $C = 2I$ the same picture holds at a lower level (`campaign_paired.csv`).

## 4. Is it the nonconvex penalty?

Three estimators start from the dense lasso solution. Two prune with MCP / SCAD (dense → sparse
and LLA), one with a weighted $\ell_1$ penalty (the adaptive lasso). Adaptive lasso minus MCP
dense → sparse, rescaled $C$:

| | $p = 10$: $10^3$ / $10^4$ / $\infty$ | $p = 20$: $10^3$ / $10^4$ / $\infty$ |
|---|---|---|
| `max_f1` | +0.022 / +0.045 / 0.000 | +0.024 / +0.011 / +0.002 |
| `aupr` | +0.085 / +0.116 / +0.078 | +0.128 / +0.143 / +0.144 |
| BIC-selected $F_1$ | +0.052 / +0.075 / +0.021 | +0.029 / +0.017 / +0.040 |
| after the search | +0.016 / +0.023 / +0.013 | +0.004 / +0.008 / +0.012 |

All but the $n = \infty$ `max_f1` entries have $z > 3$. The convex estimator is at least as good on
the oracle metric and better on every data-driven one. LLA, the version of MCP with theory behind
it (Fan, Xue & Zou 2014), gains about three quarters of dense → sparse at $p = 20$ (−0.03 in
`max_f1`) and the same at $p = 10$.

So the nonconvex penalty is not what helps. What helps is the start: a fit that has seen all the
edges, from which the weak ones are removed, instead of a fit that adds edges one by one and commits
to a direction the moment a pair enters (`next_steps/051026/orientation_lock_in.md`).

![](../runs/campaign/figures/orientation_p20.png)

*What the BIC-selected graphs consist of, $p = 20$, $n = 10^4$, rescaled $C$, diagonal true $C$ (300
graphs). The lasso keeps both directions of 14 true pairs and 29 false edges; the standard MCP path
commits, and is wrong 9 times; the dense-start estimators commit and are right (orientation
accuracy 0.89 against the lasso's 0.87) with a third to a half of the false edges.*

## 4a. The greedy search, in detail

Three of the rows above and below use the same search, the greedy hill-climbing of Améndola,
Dettling, Drton, Onori & Wu (2020, *Structure learning for cyclic linear causal models*, §5)
adapted to GCLMs. It is `gclm.solvers.search` (`docs/SEARCH.md`); its three uses in the campaign
differ only in the starting graph:

| name in this write-up | start | where |
|---|---|---|
| "after the BIC search" (`search_f1` of an estimator) | the estimator's BIC-selected graph | wave 1, every cell |
| "greedy search from random starts" (`search-pure`) | the empty graph and 10 random graphs; best final score kept | wave 2 |
| "greedy search from the truth" (`search-truth`) | the true graph | wave 2, as a ceiling |

**State.** A directed graph, stored as its support $S$: the set of off-diagonal entries of $M$
allowed to be nonzero. The diagonal is always free. 2-cycles are allowed (a GCLM can have them;
the paper restricts to simple graphs).

**Score.** Refit $M$ without penalty on $S \cup \mathrm{diag}$ by least squares on the direct loss
(the normal equations on the columns of the design matrix that belong to $S$; `DirectRefit`), under
the same $C$ the paths used, then

$$\mathrm{BIC}(S)=n\big[\log\det\Sigma(\hat M_S)+\operatorname{tr}\big(\Sigma(\hat M_S)^{-1}\hat\Sigma\big)\big]+\log(n)\,(p+|S|),$$

with $\Sigma(\hat M_S)$ the stationary covariance of the refit (the solution of
$M\Sigma+\Sigma M^\top+C=0$). This is $-2$ times the Gaussian log-likelihood plus the BIC penalty
with model dimension $p+|S|$, the number of free entries of $M$; the paper's score (16) up to the
factor $-2n$. A refit that is not stable, or whose refit cannot be computed, scores $+\infty$. At
$n=\infty$ a nominal $n=10^6$ is used, so the score prefers exact fits first and fewer edges second.

**Neighbourhood** of $S$, as in the paper: every single-edge change,

- *delete* one entry of $S$;
- *reverse* one entry $(i,j)\in S$ whose reverse is not in $S$, i.e. replace it by $(j,i)$;
- *add* one off-diagonal entry not in $S$ (adding the reverse of a selected entry makes a 2-cycle).

At $p=20$ that is about 400 neighbours per step.

**Iteration** (best improvement): score every neighbour; if the best of them has a strictly lower
score than $S$, move there and repeat; otherwise stop. Scores are cached, so no support is refitted
twice within one run, and the refit of a neighbour is warm-started from the current one. There is
no restart and no randomness inside a run; the only randomness is in the random starting graphs,
which are drawn per dataset from a fixed seed (edge probability itself drawn from $U[0, 0.3]$, then
every entry independently). A cap of $p(p-1)$ steps exists as a guard and was never reached.

In pseudo-code:

```
greedy_search(S):
    best_score <- BIC(S)
    repeat
        candidates <- {delete(e) : e in S} ∪ {reverse(e) : e in S, reverse(e) not in S}
                      ∪ {add(e) : e off-diagonal, e not in S}
        S', score' <- the candidate with the lowest BIC (cached refits)
        if score' < best_score:  S <- S'; best_score <- score'
        else:                    return S
multistart(S_1 .. S_11):  return the greedy_search(S_k) with the lowest final BIC
```

**Differences from the paper.** The paper's method and this one are written out side by side as
pseudo-code, with a table of every difference, in [`../docs/SEARCH.md`](../docs/SEARCH.md) §3a.
In short: the moves and the hill-climbing are the paper's; the graph class differs (directed
graphs with 2-cycles and known $C$ here, simple mixed graphs with bidirected edges there); the
score is the GCLM analogue of theirs with a least-squares refit instead of a likelihood
maximisation and the dimension $p+|S|$; and the randomly drawn starting graphs differ most, see next.

**The randomly drawn starting graphs are not the paper's, and there are far fewer of them.**
Améndola et al. start the search from **300** randomly drawn graphs per data set (maximum $10^4$
iterations each) on graphs with $p = 5, 6$; their reference for the procedure, Nowzohour, Maathuis,
Evans & Bühlmann (2017), uses 100 starting graphs drawn *uniformly* over the graph class by an MCMC algorithm (uniform sampling of
bow-free acyclic path diagrams is not trivial; the MCMC in Améndola et al. §6.1 serves to draw
the *true* graphs of their simulation the same way). Here:

- the number of randomly drawn starting graphs is **10** plus the empty graph, a choice made in
  S3b for cost (one start at $p = 20$ takes about a minute; 300 would be five hours per graph), not
  taken from the papers;
- the starting graphs are **not uniform** over the class: for our class (directed graphs, 2-cycles
  allowed) a uniform draw is a fair coin per entry and needs no MCMC, but it gives density 0.5,
  i.e. 190 edges at $p = 20$; the starts here have an edge probability drawn from $U[0, 0.3]$,
  sparser and closer to the truth's density ($\le 0.4$ at $p = 10$, $\le 0.2$ at $p = 20$). One
  probability $d$ per starting graph, then every off-diagonal entry with probability $d$; seeded
  per data set, so the same 10 graphs are used at every $n$. This recipe is the repository's own;
  the uniform draw of Nowzohour et al., adapted to this class (a fair coin per entry, exact without
  MCMC because the class has no global constraint), is the second recipe of wave 5b
  (`docs/SEARCH.md` §3a, "The starting graphs").

Both choices favour the search in cost, and the first works against it in quality: **ten starting
graphs do not saturate.** The 11 starts of a graph end at 10 distinct local optima on average, and
the best score of the 11 is reached within the first 5 starts for only 41 to 56 % of the graphs,
within the first 10 for 82 to 93 % (`search_restarts` in `campaign_search_stats.txt`). More
starting graphs would keep improving the score, so the `search-pure` rows of this write-up are a
lower bound on what the method with 300 would reach, not its value. Whether 100 or 300 close the
gap of 0.02 to 0.07 to the lasso-based starts is open. **Wave 5b** answers it at $p = 10$
(`cluster/submit_campaign.sh --wave 5`, cells `search100s_p10_Cresc` and `search100u_p10_Cresc`):
100 starting graphs per data set, drawn sparse as in wave 2 and uniformly as in Nowzohour et al.
(for our graph class a fair coin per entry, no MCMC needed), about 15 CPU-h per cell and sample
size; the graph every start ends at is stored, so `simulations/diagnostics/restarts.py` gives the
$F_1$ of the best of the first $r$ starts for every $r \le 100$ from one run. A smaller version at
$p = 20$ (30 starts on 5 replicates, `search30s_p20_Cresc`) costs 50 to 100 CPU-h.

**A start can be dead.** With the rescaled $C$ the least-squares refit of a very sparse support is
not always stable (`docs/SEARCH.md` §2a): the empty graph then has BIC $= +\infty$ and the search
cannot leave it. In wave 2 this is so for 6 to 7 % of the graphs at $p = 10$ and 24 to 25 % at
$p = 20$ (most under `C_Random_Min_Diag` and `C_Random_Full`; never with $C = 2I$), and for the true
support of 2 to 5 % of the graphs, almost all `C_Random_Full`. The best of the 11 starts was dead
for one graph of 2 375, so the `search-pure` numbers stand; but "the empty graph is the best start"
in §4 is understated for the rescaled $C$, and a search from the truth that starts at $+\infty$
makes its first move without a warm start. The likelihood refit of wave 5a has no dead starts.

**Validation** (`tests/test_search.py`, S3b §9.0): the refits recover the true $M$ on the true
support; on Example 2 every support with up to 7 edges was scored exhaustively and the greedy
search compared with the global optimum from every start.

## 5. Without the oracle: BIC, eBIC and the search

![](../runs/campaign/figures/selection_p20.png)

*$p = 20$, rescaled $C$: the same four estimators under the three ways to get one graph from a path.*

- **BIC-selected graphs keep the ranking** (figure `by_true_c_bic_f1`): MCP dense → sparse minus the
  lasso +0.05 / +0.11 / +0.09 at $p = 20$, ≈ 0 / +0.03 / +0.03 at $p = 10$; the adaptive lasso
  +0.08 / +0.13 / +0.13 and +0.02 / +0.04 / +0.04. With $C = 2I$ the gains are a third of that.
  Plain BIC over-selects on every path at these sizes (lasso at $p = 20$: 95, 102, 124 edges for 47
  true ones; MCP dense → sparse 62, 68, 100; the $n = \infty$ scores use a nominal $n = 10^6$).
- **Dettling's extended BIC** ($\gamma = 1$, his eq. 6.2) changes the selected $F_1$ by at most
  ±0.03 and the ranking not at all; its sparser choice still leaves the lasso with 80 to 93 edges at
  $p = 20$. The path, not the rule, is what matters (`docs/SEARCH.md` §2a).
- **The search equalises.** It lifts the lasso's BIC graph by +0.09 / +0.14 / +0.12 at $p = 20$,
  MCP dense → sparse by +0.06 / +0.04 / +0.03, the adaptive lasso by +0.03 / +0.03 / 0.00; afterwards
  all lasso-based starts lie within 0.02 of each other, the adaptive lasso on top (+0.01 over
  lasso + search, $z$ 3 to 5). At $p = 10$ the search adds little to anyone.

![](../runs/campaign/figures/search_ceilings.png)

![](../runs/campaign/figures/by_true_c_search_f1.png)

*Above: the graph the search ends at, from five starts, rescaled $C$. Below: the same as paired
differences to lasso + search, per setting of the true $C$.*

- **The greedy search from randomly drawn starting graphs** (empty graph and 10 random graphs, §4a) ends 0.02 to 0.07 below any
  lasso-based start: the paths are a better start than random graphs, as in S3b. With the caveat
  of §4a: 10 starting graphs are far fewer than the 300 of Améndola et al., and 10 have not
  saturated, so this is a lower bound on the method (wave 5b runs 100). It is also by far
  the most expensive method: a median of 23,000 (p = 10) to 490,000 (p = 20, $n = \infty$) supports
  scored per graph, 9 s to 20 minutes, against 2 to 60 moves and under a minute for a search from a
  path's BIC-selected graph. The empty graph is the best of its 11 starts in only 7 to 18 % of the
  graphs; none of the starts ends exactly at the truth at $n = 10^3$, and at $n = \infty$ about 0.2
  of the 11 do.
- **The greedy search started from the truth** ends 0.10 to 0.24 above the best data-driven method. With
  `C_ID` at $n = \infty$ it stays at the truth or next to it ($F_1$ 0.98 to 0.99; 55 % of the graphs
  unchanged, the rest lose an edge of negligible weight); in the misspecified settings it moves away
  (0.81 to 0.87 for the random diagonals, 0.43 to 0.66 for `C_Random_Full`): there the score prefers
  another graph, and no search can fix that. At $p = 20$, $n = \infty$ the data-driven searches end
  lower than at $n = 10^4$ (0.62 against 0.65): the dense graphs where the search gets stuck
  (`next_steps/051026/next_steps_051026.md` §2).

**Planned check (wave 5a).** The BIC above scores a support by the likelihood *at the
least-squares refit*, not the maximised likelihood (`docs/SEARCH.md` §2a); Améndola et al. and
Dettling use the maximised one. The cells `direct_lasso-ml`, `direct_MCP-up-ml` and
`direct_adaptive-ml` (both $C$, $p = 10$, `--refit loglik`) repeat the selection and the search with
the maximised likelihood, so that the ranking of the BIC-selected and searched graphs can be
confirmed under the BIC proper. On three $p = 10$ graphs timed on the laptop
(`../next_steps/051026/files/time_refit_loglik.txt`) the two refits selected the same $\lambda$ on
two and neighbouring ones on the third; there the likelihood refit's selected graph ($F_1$ 0.76
against 0.73) and searched graph (0.80 against 0.76) were the better ones, on the other two the
searches ended at the same graphs. The cost is 15 to 100 s per graph against under a second.

**Planned check (wave 5c).** The search above, and the pure search of §4a, run with the plain
BIC; the extended term was used offline on the path only (the `ebic1_f1` column), never inside
the search. Since the plain BIC over-selects (the lasso's BIC-selected graph has 95 to 124 edges
for 47 true ones at $p = 20$) and the search then prunes, the term may change where the search
ends. The cells `direct_<lasso|MCP-up|adaptive>-ebic_<C>` ($p = 10, 20$, `--ebic-gamma 0.5 1`)
select and search once more per $\gamma$ on the same path, next to the plain BIC, and
`searche1_p10_C2I`, `searche1_p10_Cresc`, `searche1_p20_Cresc` repeat wave 2 with $\gamma = 1$;
the form is Dettling's $4\gamma|E|\log p$ throughout (`docs/SEARCH.md` §2).

## 6. Preliminary: the log-likelihood loss (wave 3, $p = 10$)

![](../runs/campaign/figures/loglik_p10.png)

*The rescaled-$C$ cells are at 200 to 300 of 400 graphs; the figure will be redrawn when they are
complete.*

| MCP dense → sparse − lasso, log-likelihood loss, `max_f1` | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|
| $C = 2I$ (400 graphs) | −0.052 (−8) | −0.068 (−9) | −0.018 (−3) |
| rescaled $C$ (300 / 250 / 200 graphs) | 0.000 (0) | −0.036 (−4) | +0.024 (+3) |

On this loss the dense → sparse MCP path does not beat the lasso at finite $n$. Two things are
different from the direct loss, and the second is probably the reason:

- the lasso's own order matters little here (dense → sparse +0.00 to +0.02 over sparse → dense),
  so the lasso is not the problem;
- **the dense start is a different one.** For the covariance losses `covloss_path(direction="up")`
  starts from the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, as Varando & Hansen do: a fully dense matrix
  with no preference for sparsity. The direct-loss dense → sparse path starts from the *lasso's*
  dense end, which is close to the minimum-$\ell_1$ exact fit and already favours sparse solutions.
  The natural next cell is the log-likelihood MCP path started from the log-likelihood lasso's dense
  end; it needs one option in `covloss_path` and about 60 CPU-h.

The standard MCP path loses here as on the direct loss (−0.11 to −0.14). All log-likelihood fits
are machine-dependent for single graphs (S2b §2); only the means are meaningful.

## 7. Expectations against outcome

Written down before the run (campaign note §3.5):

| expectation | outcome |
|---|---|
| 1. `C_ID`, rescaled $C$, dense → sparse: +0.03 at $n = 10^3$, +0.07 to +0.10 at $n \ge 10^4$ for $p = 10$, more for $p = 20$ | held: +0.03 / +0.07 / +0.10 at $p = 10$, +0.08 / +0.15 / +0.21 at $p = 20$ |
| 2. `C_Random_Min_Diag` like `C_ID`; `C_Random_Diag` smaller; `C_Random_Full` none | `C_Random_Min_Diag` yes; `C_Random_Diag` **as large as `C_ID`**; `C_Random_Full` none at $p = 10$, small at $p = 20$ |
| 3. with $C = 2I$: a tie at $n = 10^3$, at most +0.05 at large $n$ | tie at $p = 10$; at $p = 20$ up to **+0.09** |
| 4. standard paths lose with either $C$, MCP clearly, SCAD slightly | held; SCAD ties with the rescaled $C$ at $p = 20$ |
| 5. LLA and the adaptive lasso match dense → sparse in `max_f1`; the adaptive lasso better in `aupr` | adaptive lasso: held, and better on every data-driven metric too; LLA: held at $p = 10$, **−0.03 at $p = 20$** |
| 6. BIC keeps the order with smaller gains at $n = 10^3$; the search adds +0.01 to +0.03 with the rescaled $C$, nothing with $C = 2I$ | first half held; the search adds **+0.09 to +0.14 to the lasso** at $p = 20$ and less to the others, with either $C$ (more with the rescaled one) |
| 7. log-likelihood: open | preliminary: dense → sparse does not beat the lasso at finite $n$ on this loss (§6) |

The reading rules of §3.6 of the note: the claim "MCP / SCAD dense → sparse beat the lasso when $C$
is specified correctly" stands ($z$ 6 to 30 for `C_ID` at $n \ge 10^4$, both $p$); it extends to a
mildly wrong and to a wrong diagonal $C$; and "the nonconvex penalty is needed" fails, because the
adaptive lasso does at least as well. $C$ is the bottleneck only when it is not diagonal.

## 8. What this means for the thesis

- The result on nonconvex penalties is two-sided and should be stated that way: used as published
  they are worse than the lasso, for a reason that can be shown (the early commitment to a
  direction); used from a dense start they are better, for a reason that can be shown too (the
  lasso hedges, the pruning resolves the hedge). The second half is not specific to MCP / SCAD.
- The cleanest positive statement is about the adaptive lasso: convex, deterministic, cheapest,
  best on every metric here, with Zou's theory behind it.
- The rescaled $C$ belongs in the simulation as the correctly specified benchmark, with the
  $C = 2I$ rows as the comparison with Dettling's pipeline as published; both are reported.
- Open: wave 3 complete and the log-likelihood path from the lasso's dense end; wave 4 for the
  figure over $p$; the theory of the lock-in and of why the dense start resolves it.

## 9. Files

| | |
|---|---|
| `runs/campaign/<cell>/shards/*.npz` | the raw results (gitignored; `campaign note §4.3` lists the fields) |
| `runs/campaign/campaign_per_dataset.csv` | one row per graph and estimator: path metrics, BIC / eBIC / search graphs with their orientation breakdown |
| `runs/campaign/campaign_means.csv`, `campaign_paired.csv` | the tables of this write-up |
| `runs/campaign/campaign_baseline_check.txt` | the reproduction of the n-sweep cells |
| `runs/campaign/figures/` | the figures above and `twobytwo_{bic_f1,search_f1,aupr}`, `by_true_c_bic_f1` |
| `simulations/diagnostics/campaign.py`, `plot_campaign.py` | tables and figures |
| `simulations/diagnostics/restarts.py` → `runs/campaign/campaign_restarts.csv` | the best of the first $r$ randomly drawn starting graphs of the pure search, for every $r$ (wave 2: scores only; wave 5b: also $F_1$) |
