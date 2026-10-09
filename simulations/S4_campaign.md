# S4 — The campaign: every estimator with $C = 2I$ and with the rescaled $C$, at scale

*Results of the cluster campaign planned in
[`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md)
(the plan, the reasons, the code map). Status on 8 October 2026: waves 1 to 4 complete (one
$p = 50$ cell, SCAD dense → sparse with the rescaled $C$, at 350 of 400 graphs, its last shards
still running); wave 5 (the score with the likelihood refit and the eBIC penalty inside the search,
§5; 100 starting graphs for the pure search, §4a) complete on 9 October but for one shard of
one cell. Numbers:
`runs/campaign/campaign_{means,paired,per_dataset}.csv` from `simulations/diagnostics/campaign.py`;
figures: `simulations/diagnostics/plot_campaign.py` → `runs/campaign/figures/`. Terms: a graph is
selected by its score, the sum of a loss and a penalty (the BIC penalty or the eBIC penalty);
definitions in [`../docs/SEARCH.md`](../docs/SEARCH.md) §2.*

---

## 0. Summary

The question of the thesis is whether MCP and SCAD recover the graph of a GCLM better than the
lasso. On 800 graphs per cell (Figure 5's generator, $p = 10, 20$, $n = 10^3, 10^4, \infty$, all four
settings of the true $C$), the answer is:

1. **Used the standard way, no.** MCP loses to the lasso by 0.04 to 0.12 in `max_f1` in every cell,
   SCAD by 0.03 to 0.07 with $C = 2I$; the gap grows with $n$. (As in S2 and S2b.)
2. **Started from the dense lasso solution and pruned, yes.** MCP and SCAD run dense → sparse beat
   the lasso in every cell except one tie, by up to +0.146 at $p = 20$, $n = \infty$ with the rescaled
   $C$ and +0.086 with $C = 2I$. The gain grows with $p$ and with $n$: at $n = 10^3$ it rises from
   about zero at $p = 10$ to +0.044 ($C = 2I$) and +0.065 (rescaled $C$) for MCP at $p = 50$ (§6a).
3. **The gain does not need the nonconvex penalty.** The adaptive lasso, which starts from the same
   dense solution and prunes with a convex step, matches MCP dense → sparse in `max_f1` at $p = 10$
   and 20 (paired differences −0.010 to +0.024; MCP dense → sparse slightly ahead at $p = 10$,
   $n \ge 10^4$, the adaptive lasso ahead at $p = 20$) and is better in `aupr` (+0.07 to +0.14) and
   in the graph selected by the score (+0.01 to +0.04). Over $p$ at $n = 10^3$ its `max_f1` hardly
   falls (0.58 at $p = 10$, 0.55 to 0.57 at $p = 50$) while the lasso's falls from 0.59 to 0.43 to
   0.47, so the gap reaches +0.08 ($C = 2I$) and +0.13 (rescaled $C$) at $p = 50$. The message is
   "start from the lasso, then prune"; MCP and SCAD are one way to do it, the adaptive lasso the best
   of the ways tried.
4. **The gain holds for every diagonal true $C$,** known or not: `C_ID`, `C_Random_Min_Diag` and
   `C_Random_Diag` show gains of the same size. For the non-diagonal `C_Random_Full` there is none
   at $p = 10$ and a small one at $p = 20$.
5. **The correctly specified $C$ roughly doubles the gain** (verdict 6 of `VERDICTS.md`, mechanism
   in `docs/DENSE_START.md` §7), and it is the only setting in which the selected graph of the
   dense-start estimators clearly beats the lasso's at $n = 10^3$.
6. **Without an oracle $\lambda$:** the graphs the score selects keep the ranking, with smaller gains at
   $n = 10^3$. The greedy search then equalises almost everything: it lifts the lasso by +0.09 to +0.14
   at $p = 20$ and the dense-start estimators by less, so all lasso-based starts end within 0.02 of
   each other, the adaptive lasso on top. The greedy search from random starts is 0.02 to 0.07
   worse; the greedy search from the truth is 0.10 to 0.24 better, so most of the remaining gap is in the
   search, not in the paths.
7. **On the log-likelihood loss (wave 3, complete)** MCP dense → sparse does *not* beat the lasso
   at finite $n$ (−0.05 to −0.07 with $C = 2I$, −0.00 and −0.03 with the rescaled $C$), only at
   $n = \infty$ with the rescaled $C$ (+0.025); its `aupr` is better with the rescaled $C$ (+0.06 to
   +0.14). Its dense start is the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, not the lasso's dense end;
   §6 for why that is probably the difference. The log-likelihood lasso matches the direct-loss
   lasso; the best estimators on either loss are the direct-loss dense-start ones.
8. **At large $p$ the BIC penalty is much too weak and the score selects far too many edges,** and
   more so with the rescaled $C$: at $p = 50$, $n = 10^3$ the lasso's selected graph has 313
   ($C = 2I$) and 610 (rescaled $C$) edges for 123 true ones, the adaptive lasso's 161 and 301. The
   $F_1$ of the selected graph therefore falls with $p$ for every estimator, fastest with the
   rescaled $C$, and the ranking (adaptive lasso, then MCP / SCAD dense → sparse, then the lasso,
   then the standard paths) stays. Dettling's eBIC penalty with $\gamma = 1$ halves the excess and is
   the better penalty from $p \approx 30$ on at $n = 10^3$ and from $p = 20$ on at $n \ge 10^4$; where
   the BIC penalty is not too weak (small $p$, $C = 2I$, $n = 10^3$) the eBIC term costs up to 0.04
   (§5, figure `bic_vs_ebic`).
9. **The pure search needs many sparse starting graphs, and the right kind** (wave 5b, $p = 10$,
   rescaled $C$). With 100 sparse starts instead of 10 it gains +0.05 at $n \ge 10^4$ and ends within
   0.02 of the lasso + search and 0.01 to 0.03 below the adaptive lasso + search, at 10 to 30 times
   their cost and still not saturated; at $n = 10^3$ it gains +0.01 and stays 0.03 to 0.04 behind.
   Uniform starts, the recipe of Nowzohour et al., are far worse than sparse ones (−0.06 to −0.15
   with 100 of each): the start distribution matters more than the number (§4a).
10. **The selection rule is confirmed, with one amendment** (waves 5a, 5c, $n = 10^4$). The score with
   the maximised likelihood gives the same graphs as the least-squares refit (within 0.01, at 30 to
   180 times the cost). Dettling's eBIC term *inside* the selection and the search adds +0.02 to
   every estimator at $p = 20$ and nothing at $p = 10$, with the ranking unchanged; as the BIC penalty
   is too weak at large $p$ (point 8), the eBIC penalty is the one to use from $p = 20$ on (§5).

**The two headline figures.** Over $p$ at Figure 5's sample size, and over $n$ at $p = 10, 20$:

![](../runs/campaign/figures/by_p.png)

![](../runs/campaign/figures/twobytwo_max_f1.png)

**Figure index** (all in `../runs/campaign/figures/`, drawn on 8 October from the complete data by
`simulations/diagnostics/plot_campaign.py`; the numbers are in the CSVs next to them):

| figure | what | section | data behind it |
|---|---|---|---|
| `by_p`, `gain_by_p` | the six estimators over $p = 10 \dots 50$ at $n = 10^3$, both $C$; and as paired gains over the lasso | §6a | final (one $p = 50$ cell at 350 of 400 graphs) |
| `twobytwo_max_f1`, `_aupr`, `_bic_f1`, `_search_f1` | the 2 × 2 over $n$ at $p = 10, 20$ | §2, §5 | final |
| `by_true_c_max_f1`, `_bic_f1`, `_search_f1` | the gains per setting of the true $C$ | §3 | final |
| `selection_p20`, `orientation_p20` | oracle $\lambda$ against the selected graph (BIC penalty) against the search; what the selected graphs consist of | §4, §5 | final |
| `search_ceilings` | the search from the estimators' graphs, from random starts and from the truth | §5 | final (wave 2 complete) |
| `loglik_p10` | the log-likelihood loss | §6 | final (wave 3 complete) |
| `restarts` | the pure search with 100 starting graphs, sparse and uniform, over $n$ | §4a | final (wave 5b complete) |
| `selection_checks` | the score with the likelihood refit; the eBIC term inside the search | §5 | final (one cell at 350 of 400 graphs) |
| `bic_vs_ebic` | the BIC penalty against the eBIC penalty: $F_1$ difference and edges selected, over $p$ and $n$ | §5 | final for the selection; the search part fills with wave 6 |
| `../runs/campaign/figures_ebic1/*` | every rule-dependent figure with the eBIC penalty in the score | §5 | selection only until wave 6 is in |

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
| per graph and estimator | the path (100 $\lambda$); its `max_f1`, `auc`, `aupr`; the graph selected by the score with the BIC penalty; that graph after the greedy search; the graph selected with Dettling's eBIC penalty, computed afterwards from the stored scores (`docs/SEARCH.md` §2a) |
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
| $F_1$ of the selected graph | +0.052 / +0.075 / +0.021 | +0.029 / +0.017 / +0.040 |
| after the search | +0.016 / +0.023 / +0.013 | +0.004 / +0.008 / +0.012 |

All but the $n = \infty$ `max_f1` entries have $z > 3$. The convex estimator is at least as good on
the oracle metric and better on every data-driven one. LLA, the version of MCP with theory behind
it (Fan, Xue & Zou 2014), gains about three quarters of dense → sparse at $p = 20$ (−0.03 in
`max_f1`) and the same at $p = 10$.

So the nonconvex penalty is not what helps. What helps is the start: a fit that has seen all the
edges, from which the weak ones are removed, instead of a fit that adds edges one by one and commits
to a direction the moment a pair enters (`next_steps/051026/orientation_lock_in.md`).

![](../runs/campaign/figures/orientation_p20.png)

*What the selected graphs (BIC penalty) consist of, $p = 20$, $n = 10^4$, rescaled $C$, diagonal
true $C$ (300 graphs). The lasso keeps both directions of 14 true pairs and 29 false edges; the
standard MCP path commits, and is wrong 9 times; the dense-start estimators commit and are right
(orientation accuracy 0.89 against the lasso's 0.87) with a third to a half of the false edges.*

## 4a. The greedy search, in detail

Three of the rows above and below use the same search, the greedy hill-climbing of Améndola,
Dettling, Drton, Onori & Wu (2020, *Structure learning for cyclic linear causal models*, §5)
adapted to GCLMs. It is `gclm.solvers.search` (`docs/SEARCH.md`); its three uses in the campaign
differ only in the starting graph:

| name in this write-up | start | where |
|---|---|---|
| "after the search" (`search_f1` of an estimator) | the estimator's selected graph | wave 1, every cell |
| "greedy search from random starts" (`search-pure`) | the empty graph and 10 random graphs; best final score kept | wave 2 |
| "greedy search from the truth" (`search-truth`) | the true graph | wave 2, as a ceiling |

**State.** A directed graph, stored as its support $S$: the set of off-diagonal entries of $M$
allowed to be nonzero. The diagonal is always free. 2-cycles are allowed (a GCLM can have them;
the paper restricts to simple graphs).

**Score.** Refit $M$ without penalty on $S \cup \mathrm{diag}$ by least squares on the direct loss
(the normal equations on the columns of the design matrix that belong to $S$; `DirectRefit`), under
the same $C$ the paths used, then

$$\mathrm{score}(S)=L(S)+\mathrm{pen}(S),\qquad L(S)=n\big[\log\det\Sigma(\hat M_S)+\operatorname{tr}\big(\Sigma(\hat M_S)^{-1}\hat\Sigma\big)\big],\qquad \mathrm{pen}_{\mathrm{BIC}}(S)=\log(n)\,(p+|S|),$$

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
    best_score <- score(S)
    repeat
        candidates <- {delete(e) : e in S} ∪ {reverse(e) : e in S, reverse(e) not in S}
                      ∪ {add(e) : e off-diagonal, e not in S}
        S', score' <- the candidate with the lowest score (cached refits)
        if score' < best_score:  S <- S'; best_score <- score'
        else:                    return S
multistart(S_1 .. S_11):  return the greedy_search(S_k) with the lowest final score
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

![](../runs/campaign/figures/restarts.png)

**Result (9 October; the six cells, 400 graphs each, 10 to 14 CPU-h per cell).** The $F_1$ of
the best of the first $r$ starting graphs plus the empty graph, and where the lasso-based searches
and the truth end on the same graphs:

| $p = 10$, rescaled $C$ | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|
| sparse starts, $r = 10$ (= wave 2) | 0.475 | 0.535 | 0.537 |
| sparse starts, $r = 30$ | 0.479 | 0.567 | 0.567 |
| sparse starts, $r = 100$ | 0.486 | **0.581** | **0.589** |
| uniform starts, $r = 100$, without the empty graph | 0.413 | 0.434 | 0.429 |
| uniform starts, $r = 100$, with the empty graph | 0.441 | 0.488 | 0.484 |
| lasso + search | 0.511 | 0.597 | 0.592 |
| adaptive lasso + search | 0.527 | 0.608 | 0.602 |
| search started from the truth | 0.695 | 0.800 | 0.829 |

- **At $n \ge 10^4$ the number of starting graphs was the limitation.** With 100 sparse starts the
  pure search gains +0.046 and +0.052 over its 10-start version and ends within 0.016 and 0.003 of
  the lasso + search, 0.03 and 0.01 below the adaptive lasso + search. Its score has still not
  saturated at $r = 100$ (the best of 100 is reached within the first 50 starts for 61 to 64 % of
  the graphs) and the $F_1$ curve still rises (+0.008 and +0.014 from $r = 50$ to 100), so with the
  300 of Améndola et al. it would probably match the lasso-based starts at these sample sizes, at
  10 to 30 times their cost (14 CPU-h per cell against 0.4 to 4).
- **At $n = 10^3$ it was not:** +0.011 from 10 to 100 starts, +0.005 from 50 to 100, and the gap of
  0.025 to 0.04 to the lasso-based starts stays. There the score has more local optima than
  starts can fix, and the path's graph is the better start.
- **The uniform draw is the wrong one for this class.** The best of 100 uniform starts, density
  one half, ends 0.06 to 0.15 below the best of 10 sparse ones; adding the empty graph lifts it
  only because the empty start then wins. What Nowzohour et al. do for their class does not
  transfer: a start with 45 edges for 22 true ones descends into a dense local optimum. The
  recipe of the starts matters more than their number.
- The "search-pure" rows of this write-up therefore stay the 10-start version (the lower bound);
  the 100-start version is reported here and in verdict 24.

**A start can be dead.** With the rescaled $C$ the least-squares refit of a very sparse support is
not always stable (`docs/SEARCH.md` §2a): the empty graph then has score $= +\infty$ and the search
cannot leave it. In wave 2 this is so for 6 to 7 % of the graphs at $p = 10$ and 24 to 25 % at
$p = 20$ (most under `C_Random_Min_Diag` and `C_Random_Full`; never with $C = 2I$), and for the true
support of 2 to 5 % of the graphs, almost all `C_Random_Full`. The best of the 11 starts was dead
for one graph of 2 375, so the `search-pure` numbers stand; but "the empty graph is the best start"
in §4 is understated for the rescaled $C$, and a search from the truth that starts at $+\infty$
makes its first move without a warm start. The likelihood refit of wave 5a has no dead starts.

**Validation** (`tests/test_search.py`, S3b §9.0): the refits recover the true $M$ on the true
support; on Example 2 every support with up to 7 edges was scored exhaustively and the greedy
search compared with the global optimum from every start.

## 5. Without the oracle: the score, the eBIC penalty and the search

![](../runs/campaign/figures/selection_p20.png)

*$p = 20$, rescaled $C$: the same four estimators under the three ways to get one graph from a path.*

- **The graphs the score selects keep the ranking** (figure `by_true_c_bic_f1`): MCP dense → sparse
  minus the lasso +0.05 / +0.11 / +0.09 at $p = 20$, ≈ 0 / +0.03 / +0.03 at $p = 10$; the adaptive
  lasso +0.08 / +0.13 / +0.13 and +0.02 / +0.04 / +0.04. With $C = 2I$ the gains are a third of
  that. The BIC penalty is too weak on every path at these sizes: the score selects too many edges
  (lasso at $p = 20$: 95, 102, 124 edges for 47 true ones; MCP dense → sparse 62, 68, 100; the
  $n = \infty$ scores use a nominal $n = 10^6$).
- **Dettling's eBIC penalty** ($\gamma = 1$, his eq. 6.2) changes the selected $F_1$ by at most
  ±0.03 and the ranking not at all; its sparser choice still leaves the lasso with 80 to 93 edges at
  $p = 20$. The path, not the penalty, is what matters (`docs/SEARCH.md` §2a).
- **The search equalises.** It lifts the lasso's selected graph by +0.09 / +0.14 / +0.12 at $p = 20$,
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
  path's selected graph. The empty graph is the best of its 11 starts in only 7 to 18 % of the
  graphs; none of the starts ends exactly at the truth at $n = 10^3$, and at $n = \infty$ about 0.2
  of the 11 do.
- **The greedy search started from the truth** ends 0.10 to 0.24 above the best data-driven method. With
  `C_ID` at $n = \infty$ it stays at the truth or next to it ($F_1$ 0.98 to 0.99; 55 % of the graphs
  unchanged, the rest lose an edge of negligible weight); in the misspecified settings it moves away
  (0.81 to 0.87 for the random diagonals, 0.43 to 0.66 for `C_Random_Full`): there the score prefers
  another graph, and no search can fix that. At $p = 20$, $n = \infty$ the data-driven searches end
  lower than at $n = 10^4$ (0.62 against 0.65): the dense graphs where the search gets stuck
  (`next_steps/051026/next_steps_051026.md` §2).

![](../runs/campaign/figures/selection_checks.png)

*Left: the selected and the searched graph under the campaign's least-squares refit and under
the likelihood refit (wave 5a). Right: the searched graph with the BIC penalty and with Dettling's
eBIC penalty inside the selection and the search (wave 5c). $n = 10^4$.*

**The score with the likelihood refit changes nothing (wave 5a).** The score above uses the likelihood
*at the least-squares refit*, not the maximised likelihood (`docs/SEARCH.md` §2a); Améndola et al.
and Dettling use the maximised one. The cells `direct_<lasso|MCP-up|adaptive>-ml_<C>` ($p = 10$,
$n = 10^4$, `--refit loglik`; 400 graphs, the lasso cell with the rescaled $C$ at 350) repeat the
selection and the search with the maximised likelihood on the same paths:

| $p = 10$, $n = 10^4$, $F_1$ | selected, LS → ML | after the search, LS → ML |
|---|---|---|
| lasso, $C = 2I$ / rescaled $C$ | 0.570 → 0.571 / 0.566 → 0.572 | 0.570 → 0.570 / 0.597 → 0.602 |
| MCP dense → sparse | 0.564 → 0.563 / 0.591 → 0.594 | 0.566 → 0.565 / 0.596 → 0.605 |
| adaptive lasso | 0.575 → 0.575 / 0.606 → 0.608 | 0.567 → 0.570 / 0.608 → 0.612 |

Every difference is within 0.01 and within one standard error; the ranking is unchanged; the
likelihood refit costs 30 to 180 times more (12 to 67 CPU-h per cell against 0.4 to 5). The
least-squares refit is a valid stand-in for the likelihood refit at this size.

**The eBIC term inside the search helps at $p = 20$, not at $p = 10$ (wave 5c).** The cells
`direct_<lasso|MCP-up|adaptive>-ebic_<C>` ($p = 10, 20$, $n = 10^4$, `--ebic-gamma 0.5 1`) select and
search once more per $\gamma$ on the same path, next to the BIC penalty, with Dettling's eBIC term
$4\gamma|E|\log p$:

| $n = 10^4$, $F_1$ after the search: BIC penalty → eBIC penalty, $\gamma = 1$ (paired $z$) | $p = 10$ | $p = 20$ |
|---|---|---|
| lasso, $C = 2I$ / rescaled $C$ | 0.570 → 0.570 (0) / 0.597 → 0.598 (0) | 0.614 → 0.632 (6) / 0.650 → 0.666 (6) |
| MCP dense → sparse | 0.566 → 0.562 (−1) / 0.596 → 0.602 (2) | 0.602 → 0.624 (9) / 0.658 → 0.677 (8) |
| adaptive lasso | 0.567 → 0.571 (1) / 0.608 → 0.605 (−1) | 0.613 → 0.634 (9) / 0.666 → 0.682 (7) |

At $p = 20$ the term inside the search adds +0.016 to +0.022 to every estimator (edges after the
search 60 to 63 → 51 to 52 for 47 true ones), $\gamma = 0.5$ three quarters of that, the selection
alone +0.01 to +0.02; the ranking stays (adaptive lasso, then MCP dense → sparse, then the lasso,
within 0.016). At $p = 10$ it changes nothing (±0.006, $|z| \le 1.7$). In the pure search
(`searche1_p10_*`) it is a wash: −0.016 and −0.012 at $n = 10^3$, +0.013 and +0.009 at $10^4$, 0.00 at
$\infty$ ($C = 2I$ / rescaled $C$), and it pulls the truth-started search away from the truth at
$n = 10^3$ (0.676 → 0.637, 0.695 → 0.655): at small $n$ the term penalises true edges with small
weights.

![](../runs/campaign/figures/bic_vs_ebic.png)

*The BIC penalty against Dettling's eBIC penalty in the score that picks one graph from a path,
for the lasso, MCP dense → sparse and the adaptive lasso. Top: paired difference in $F_1$ of the
selected graph, with the eBIC penalty minus with the BIC penalty, $\gamma = 1$ (solid) and
$\gamma = 0.5$ (dotted, hollow squares); the x-marked dotted lines are the difference after the
search with $\gamma = 1$ inside it, where that was run (wave 5c; wave 6 fills the rest). Bottom:
edges selected over true edges, log scale; dashed = BIC penalty. Columns: over $p$ at $n = 10^3$
with $C = 2I$ and with the rescaled $C$; over $n$ at $p = 20$ with the rescaled $C$.*

**When the eBIC term helps, and when it hurts.** It is the right correction where the BIC penalty
is too weak and the score selects too many edges, and the wrong one where it is not. Bottom row:
with the BIC penalty the score picks 1.2 to 3.3 times the true number of edges for the lasso and
0.8 to 1.6 times for the dense-start estimators with $C = 2I$, and up to 5 times with the rescaled
$C$; $\gamma = 1$ halves the excess, and for the adaptive lasso with $C = 2I$ it goes below the
truth (0.8). Top row, accordingly: with $C = 2I$ at $n = 10^3$ the eBIC penalty *loses* for every
estimator up to $p = 25$ to $30$ (the lasso by 0.03 to 0.04, the dense-start ones by 0.01 to 0.02)
and wins from $p = 40$ on; with the rescaled $C$ it wins from $p = 30$ on (up to +0.05 for the
adaptive lasso at $p = 50$) and loses for the lasso below; over $n$ at $p = 20$ it wins at
$n \ge 10^4$ (+0.01 to +0.02) and loses for the lasso at $n = 10^3$. $\gamma = 0.5$ sits between
the two penalties throughout. The ranking of the estimators is the same with either penalty
everywhere. So: the eBIC penalty from $p \approx 30$ on at $n = 10^3$, and from $p = 20$ on at
$n \ge 10^4$; below that the BIC penalty, or $\gamma = 0.5$ as the compromise. Wave 6
(`cluster/plan_091026.txt`) rescores every wave 1 cell with the eBIC term inside the search, after
which every figure of this write-up exists with that penalty in `../runs/campaign/figures_ebic1/`
(`plot_campaign.py --rule ebic1`; until then those figures show the selection only).

## 6. The log-likelihood loss (wave 3, $p = 10$, complete)

![](../runs/campaign/figures/loglik_p10.png)

*The log-likelihood loss, $p = 10$, 400 graphs per cell: lasso and MCP in both path orders, both
$C$, $F_1$ at the oracle $\lambda$.*

| paired difference to the log-likelihood lasso, same $C$ ($z$) | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|
| MCP dense → sparse, `max_f1`, $C = 2I$ | −0.052 (−8) | −0.068 (−9) | −0.018 (−3) |
| MCP dense → sparse, `max_f1`, rescaled $C$ | −0.004 (−1) | −0.028 (−4) | **+0.025 (+4)** |
| MCP dense → sparse, `aupr`, $C = 2I$ | −0.054 (−8) | −0.062 (−7) | +0.007 (+1) |
| MCP dense → sparse, `aupr`, rescaled $C$ | **+0.059 (+9)** | **+0.062 (+8)** | **+0.135 (+18)** |
| MCP dense → sparse, `bic_f1`, $C = 2I$ / rescaled $C$ | −0.100 / −0.035 | −0.106 / −0.062 | −0.012 / +0.009 |
| lasso dense → sparse, `max_f1`, $C = 2I$ / rescaled $C$ | +0.003 / +0.019 | +0.004 / +0.008 | +0.002 / +0.008 |
| lasso dense → sparse, `aupr`, $C = 2I$ / rescaled $C$ | +0.018 / +0.065 | +0.022 / +0.069 | +0.021 / +0.074 |
| MCP standard path, `max_f1`, $C = 2I$ / rescaled $C$ | −0.108 / −0.121 | −0.121 / −0.147 | −0.139 / −0.161 |

On this loss the dense → sparse MCP path does not beat the lasso in `max_f1` at finite $n$; only at
$n = \infty$ with the rescaled $C$ does it (+0.025), and in `aupr` it is better with the rescaled $C$
at every $n$ (its path ranks the entries better, but its best single graph is not better). The
standard MCP path loses here as on the direct loss, by 0.11 to 0.16. Two things differ from the
direct loss, and the second is probably the reason for the missing gain:

- the lasso's own order matters little here (dense → sparse +0.00 to +0.02 in `max_f1`, +0.02 to
  +0.07 in `aupr` over sparse → dense), so the lasso is not the problem;
- **the dense start is a different one.** For the covariance losses `covloss_path(direction="up")`
  starts from the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, as Varando & Hansen do: a fully dense matrix
  with no preference for sparsity. The direct-loss dense → sparse path starts from the *lasso's*
  dense end, which is close to the minimum-$\ell_1$ exact fit and already favours sparse solutions.
  The natural next cell is the log-likelihood MCP path started from the log-likelihood lasso's dense
  end; it needs one option in `covloss_path` and about 60 CPU-h.

**Across the two losses.** The log-likelihood lasso and the direct-loss lasso are within 0.01 of
each other in `max_f1` at every $n$ and $C$ (0.60 / 0.63 / 0.65 against 0.59 / 0.63 / 0.64 with
$C = 2I$). The best estimators overall are the direct-loss dense-start ones: at $n = 10^4$ with the
rescaled $C$ the direct-loss MCP dense → sparse reaches 0.673 and the adaptive lasso 0.666, against
0.638 for the best log-likelihood estimator (the lasso dense → sparse) and 0.602 for the
log-likelihood MCP dense → sparse. The thesis can therefore stay with the direct loss. All
log-likelihood fits are machine-dependent for single graphs (S2b §2); only the means are meaningful.

## 6a. Over $p$: the thesis figure (wave 4 with wave 1, $n = 10^3$)

![](../runs/campaign/figures/by_p.png)

*Figure 5's setting, $n = 10^3$, 800 graphs per cell at $p = 10, 20$ and 400 at $p = 15, 25, 30, 40,
50$ (the SCAD dense → sparse cell with the rescaled $C$ at $p = 50$: 350). Rows: $F_1$ at the oracle
$\lambda$, area under the precision–recall curve, $F_1$ of the selected graph (BIC penalty).*

![](../runs/campaign/figures/gain_by_p.png)

*The same as paired differences to the lasso with the same $C$; error bars are one standard error
of the paired difference.*

| `max_f1` − lasso, same $C$ | $p = 10$ | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| MCP dense → sparse, $C = 2I$ | −0.009 | +0.002 | +0.018 | +0.022 | +0.028 | +0.043 | +0.044 |
| MCP dense → sparse, rescaled $C$ | +0.016 | +0.040 | +0.056 | +0.060 | +0.064 | +0.067 | +0.065 |
| SCAD dense → sparse, $C = 2I$ / rescaled $C$ | −0.005 / +0.021 | +0.003 / +0.030 | +0.011 / +0.038 | +0.012 / +0.035 | +0.010 / +0.033 | +0.019 / +0.032 | +0.017 / +0.025 |
| adaptive lasso, $C = 2I$ | −0.007 | +0.017 | +0.027 | +0.045 | +0.053 | +0.070 | +0.078 |
| adaptive lasso, rescaled $C$ | +0.020 | +0.055 | +0.080 | +0.094 | +0.102 | +0.124 | +0.133 |
| MCP standard, $C = 2I$ / rescaled $C$ | −0.096 / −0.074 | −0.083 / −0.061 | −0.076 / −0.043 | −0.058 / −0.027 | −0.052 / −0.014 | −0.038 / −0.004 | −0.028 / +0.006 |
| SCAD standard, $C = 2I$ / rescaled $C$ | −0.053 / −0.034 | −0.036 / −0.009 | −0.031 / +0.002 | −0.024 / +0.007 | −0.019 / +0.009 | −0.013 / +0.012 | −0.008 / +0.011 |

Every entry beyond ±0.01 has $|z| \ge 3$; the adaptive lasso's and MCP dense → sparse's gains have
$z$ from 4 at $p = 20$ to 18 to 34 at $p = 50$.

- **The gap opens with $p$ because the lasso deteriorates and the dense-start estimators hardly
  do.** The lasso's `max_f1` falls from 0.59 at $p = 10$ to 0.47 ($C = 2I$) and 0.43 (rescaled $C$)
  at $p = 50$; the adaptive lasso's from 0.58 / 0.60 to 0.55 / 0.57; MCP dense → sparse's from 0.58 /
  0.60 to 0.52 / 0.50. The `aupr` says the same more strongly: the adaptive lasso's is +0.14 and
  +0.22 above the lasso's at $p = 50$.
- **The standard paths "catch up" only because the lasso comes down to them.** Their `max_f1`
  is nearly constant in $p$ (MCP 0.49 → 0.44), so the deficit shrinks from −0.10 to −0.03 with
  $C = 2I$ and turns into a tie with the rescaled $C$; they never beat the dense-start paths.
- **The rescaled $C$ helps the dense-start estimators and hurts the lasso at large $p$.** With it
  the adaptive lasso is better than with $C = 2I$ at every $p$ (0.566 against 0.549 at $p = 50$), the
  lasso worse (0.433 against 0.471). This is the mechanism of `docs/DENSE_START.md` §7: the
  correctly specified $C$ improves the ranking at the dense end, which only the dense-start
  estimators use.
- **$p > n$ is no obstacle.** At $p = 40$ and $50$ the drift matrix has 1 600 and 2 500 free entries
  against $n = 10^3$ observations, the dense start is a least-squares fit on a rank-deficient design,
  and the gains are the largest of the sweep.
- **At large $p$ the BIC penalty is much too weak and the score selects far too many edges, worst
  with the rescaled $C$.** Mean edges of the selected graph against the true number:

| $n = 10^3$, edges | $p = 10$ (23 true) | 20 (47) | 30 (73) | 40 (98) | 50 (123) |
|---|---|---|---|---|---|
| lasso, BIC penalty, $C = 2I$ / rescaled $C$ | 24 / 27 | 71 / 95 | 129 / 205 | 216 / 384 | 313 / 610 |
| adaptive lasso, BIC penalty | 18 / 19 | 47 / 57 | 81 / 111 | 120 / 188 | 161 / 301 |
| MCP dense → sparse, BIC penalty | 18 / 19 | 50 / 62 | 88 / 128 | 134 / 227 | 183 / 365 |
| lasso, eBIC penalty, $\gamma = 1$ | 20 / 23 | 53 / 80 | 92 / 155 | 129 / 282 | 164 / 440 |
| adaptive lasso, eBIC penalty, $\gamma = 1$ | 14 / 16 | 33 / 42 | 51 / 79 | 70 / 135 | 89 / 203 |

  The $F_1$ of the selected graph therefore falls with $p$ for every estimator (lasso 0.51 → 0.40 with
  $C = 2I$, 0.51 → 0.28 with the rescaled $C$), and the dense-start estimators, which select fewer
  edges to begin with, keep their lead: +0.06 (MCP dense → sparse) and +0.10 to +0.12 (adaptive
  lasso) at $p = 50$. Dettling's eBIC penalty with $\gamma = 1$ halves the excess edges and gives a
  higher $F_1$ than the BIC penalty for every estimator from $p = 25$ on (lasso 0.405 against 0.397 at
  $p = 50$ with $C = 2I$, 0.311 against 0.279 with the rescaled $C$; adaptive lasso 0.506 against
  0.492 and 0.444 against 0.395), without changing the ranking. Why the rescaled $C$ makes the
  score with the BIC penalty select more edges is not worked out; the eBIC term inside the search
  (wave 5c) is the pragmatic answer.

## 7. Expectations against outcome

Written down before the run (campaign note §3.5):

| expectation | outcome |
|---|---|
| 1. `C_ID`, rescaled $C$, dense → sparse: +0.03 at $n = 10^3$, +0.07 to +0.10 at $n \ge 10^4$ for $p = 10$, more for $p = 20$ | held: +0.03 / +0.07 / +0.10 at $p = 10$, +0.08 / +0.15 / +0.21 at $p = 20$ |
| 2. `C_Random_Min_Diag` like `C_ID`; `C_Random_Diag` smaller; `C_Random_Full` none | `C_Random_Min_Diag` yes; `C_Random_Diag` **as large as `C_ID`**; `C_Random_Full` none at $p = 10$, small at $p = 20$ |
| 3. with $C = 2I$: a tie at $n = 10^3$, at most +0.05 at large $n$ | tie at $p = 10$; at $p = 20$ up to **+0.09** |
| 4. standard paths lose with either $C$, MCP clearly, SCAD slightly | held; SCAD ties with the rescaled $C$ at $p = 20$ |
| 5. LLA and the adaptive lasso match dense → sparse in `max_f1`; the adaptive lasso better in `aupr` | adaptive lasso: held, and better on every data-driven metric too; LLA: held at $p = 10$, **−0.03 at $p = 20$** |
| 6. selection by the score keeps the order with smaller gains at $n = 10^3$; the search adds +0.01 to +0.03 with the rescaled $C$, nothing with $C = 2I$ | first half held; the search adds **+0.09 to +0.14 to the lasso** at $p = 20$ and less to the others, with either $C$ (more with the rescaled one) |
| 7. log-likelihood: open | dense → sparse MCP does not beat the lasso in `max_f1` at finite $n$ on this loss, does in `aupr` with the rescaled $C$ and in `max_f1` at $n = \infty$ (§6) |
| 8. larger $p$ (campaign note §3.4, wave 4): "whether dense → sparse still helps at $p = 40, 50$ with $n = 1000$ is not known" | it helps most there: the gap to the lasso grows monotonically with $p$, to +0.04 / +0.07 (MCP dense → sparse) and +0.08 / +0.13 (adaptive lasso) at $p = 50$ (§6a); at large $p$ the BIC penalty is much too weak and the score selects far too many edges |

The reading rules of §3.6 of the note: the claim "MCP / SCAD dense → sparse beat the lasso when $C$
is specified correctly" stands ($z$ 6 to 30 for `C_ID` at $n \ge 10^4$, both $p$); it extends to a
mildly wrong and to a wrong diagonal $C$; and "the nonconvex penalty is needed" fails, because the
adaptive lasso does as well in `max_f1` (±0.01 at $p = 10$, ahead from $p = 20$ on) and better in
`aupr` and the selected graph. $C$ is the bottleneck only when it is not diagonal.

## 8. What this means for the thesis

- The result on nonconvex penalties is two-sided and should be stated that way: used as published
  they are worse than the lasso, for a reason that can be shown (the early commitment to a
  direction); used from a dense start they are better, for a reason that can be shown too (the
  lasso hedges, the pruning resolves the hedge). The second half is not specific to MCP / SCAD.
- The cleanest positive statement is about the adaptive lasso: convex, deterministic, cheapest,
  best or tied on every metric here (MCP dense → sparse is ahead by at most 0.01 in oracle $F_1$
  at $p = 10$, $n \ge 10^4$), with Zou's theory behind it.
- The rescaled $C$ belongs in the simulation as the correctly specified benchmark, with the
  $C = 2I$ rows as the comparison with Dettling's pipeline as published; both are reported.
- Open: wave 3 complete and the log-likelihood path from the lasso's dense end; wave 4 for the
  figure over $p$; the theory of the lock-in and of why the dense start resolves it.

## 9. Files

| | |
|---|---|
| `runs/campaign/<cell>/shards/*.npz` | the raw results (gitignored; `campaign note §4.3` lists the fields) |
| `runs/campaign/campaign_per_dataset.csv` | one row per graph and estimator: path metrics, the graphs selected with the BIC penalty and with the eBIC penalty and the searched graphs, with their orientation breakdown |
| `runs/campaign/campaign_means.csv`, `campaign_paired.csv` | the tables of this write-up |
| `runs/campaign/campaign_baseline_check.txt` | the reproduction of the n-sweep cells |
| `runs/campaign/figures/` | the figures above and `twobytwo_{bic_f1,search_f1,aupr}`, `by_true_c_bic_f1`; `by_p`, `gain_by_p` (§6a), `restarts` (§4a), `selection_checks`, `bic_vs_ebic` (§5) |
| `runs/campaign/figures_ebic1/` | the rule-dependent figures with the eBIC penalty in the score (`plot_campaign.py --rule ebic1`) |
| `runs/campaign/campaign_restarts.csv`, `.txt` | the best of the first $r$ starting graphs, every cell of waves 2 and 5b (`simulations/diagnostics/restarts.py`) |
| `simulations/diagnostics/campaign.py`, `plot_campaign.py` | tables and figures |
| `simulations/diagnostics/restarts.py` → `runs/campaign/campaign_restarts.csv` | the best of the first $r$ randomly drawn starting graphs of the pure search, for every $r$ (wave 2: scores only; wave 5b: also $F_1$) |
