# Verdicts so far

*As of 9 October 2026. The bottom line, the verdicts as a table, then the evidence for them figure by figure; the evidence is in the documents named in each row. "IS" is the
independent study (`../next_steps/021026/independent_study/independent_study_021026.md`), "031026"
the combined note (`../next_steps/031026/next_steps_031026.md`), "campaign note"
`../next_steps/051026/cluster_campaign_051026.md`, "S4" the campaign's results (`S4_campaign.md`).*

## Bottom line

Used the standard way, nonconvex penalties do not beat the lasso for GCLMs, on any loss and at any
sample size. They fail on the direction of edges, because the standard path fixes a direction too
early. Started from the dense lasso solution and used to prune it, they beat the lasso, by more as
$p$ and $n$ grow (at $n = 10^3$ from a tie at $p = 10$ to +0.04 / +0.07 at $p = 50$), for every
diagonal true $C$ and under Dettling's own pipeline as well as with the correctly specified $C$
(S4, 800 graphs per cell at $p = 10, 20$, 400 at $p = 15 \dots 50$). But the convex adaptive lasso,
which prunes the same start, does at least as well on every metric and keeps its $F_1$ nearly
constant in $p$ where the lasso's falls (+0.08 / +0.13 over the lasso at $p = 50$): the gain comes
from the start, not from the nonconvex penalty.

![](../runs/campaign/figures/by_p.png)

*The thesis figure: $F_1$ at the oracle $\lambda$, area under the precision–recall curve and $F_1$ of
the BIC-selected graph over $p$ at $n = 10^3$, Figure 5's setting, both choices of $C$ (S4 §6a,
8 October; 800 graphs per cell at $p = 10, 20$, 400 at the other sizes).*

![](../runs/campaign/figures/gain_by_p.png)

*The same as paired differences to the lasso with the same $C$. The 2 × 2 over $n$ at $p = 10, 20$
and the other figures are indexed in S4 §0.*

## The verdicts

*How to read this page: each row is one claim with its numbers; the evidence column names the
figure that shows it (all in `../runs/campaign/figures/`, listed with a one-line reading under
"The figures" below) and the document section with the table, the $z$-values and the reasoning.
S4 is the long version of rows 7 to 26; the earlier studies (S2, S3a, S3b, IS) carry the rest.*

| # | verdict | evidence | status |
|---|---|---|---|
| 1 | **Standard MCP / SCAD lose to the lasso.** Dettling's pipeline (standardised data, $C = 2I$), path from $\lambda_{\max}$ to the dense end: worse than the lasso on `max_f1`, `auc`, `aupr`, for the direct, log-likelihood and Frobenius losses, $p = 10, 20$, all four $C$ settings. | `twobytwo_max_f1` (hollow markers below the lasso); S2 §3.1–3.3; S2b §3 | settled |
| 2 | **More data does not help them.** The gap grows from $n = 10^3$ to $\infty$ (MCP − lasso in `max_f1`: −0.08 … −0.12 at $10^3$, −0.10 … −0.16 at $\infty$). | `twobytwo_max_f1` (the gap over n); S2b §3.2, §3.5 | settled |
| 3 | **The failure is edge direction.** MCP and SCAD find nearly the same pairs as the lasso (skeleton $F_1$ 0.02–0.08 lower) but keep one direction per pair; MCP reverses 2–4 times as many edges. They almost never recover both directions of a 2-cycle. | `orientation_p20`; S2 §3.4; S3a; S2b §3.6 | settled |
| 4 | **The direction is fixed too early.** The first direction to enter the path is the true one only 51–61 % of the time, for every penalty. The lasso later corrects it by keeping both directions; MCP keeps the first one. | S3a §4.4 | settled |
| 5 | **Keeping both directions is the right response under $F_1$.** Committing to one direction pays only if it is right more than about 70 % of the time; at $n = 1000$ the estimators manage 54–63 % on the pairs in question. | 021026 §1; IS §3 | settled |
| 6 | **Dettling's pipeline fits a misspecified model, even for `C_ID`.** Standardising changes the volatility to $C = 2\,\mathrm{diag}(1/s_i^2)$ ($s_i$: the standard deviations); the pipeline keeps $C = 2I$. The lasso is robust to it; BIC and other likelihood-based decisions are not. | IS §2; 031026 §3.1 | settled |
| 7 | **MCP / SCAD run dense → sparse beat the lasso.** Paired `max_f1` gain over the lasso with the same $C$, rescaled $C$: +0.016 / +0.045 / +0.058 at $p = 10$ and +0.056 / +0.115 / +0.146 at $p = 20$ ($n = 10^3 / 10^4 / \infty$); with $C = 2I$: −0.009 / +0.012 / +0.026 and +0.018 / +0.060 / +0.086. The gain is of the same size for all three diagonal settings of the true $C$, absent for the non-diagonal one at $p = 10$, small at $p = 20$. | `twobytwo_max_f1`, `by_true_c_max_f1`; S4 §2–3 (800 graphs per cell); IS §4–5; 031026 §4 | settled for the direct loss |
| 8 | **A BIC search with add / delete / reverse moves is a useful finishing step, on a correctly specified model only.** On Dettling's pipeline it does not improve on the lasso. | S3b §9.2–9.4; IS §9 | settled for `C_ID`; weaker with random $C$ |
| 9 | **The search equalises the lasso-based starts.** It lifts the lasso's BIC-selected graph by +0.09 to +0.14 at $p = 20$ and the dense-start estimators by less; afterwards all lasso-based starts lie within 0.02 of each other, the adaptive lasso on top by +0.01. Starts from the standard MCP / SCAD paths stay far behind (their reversed edges survive). The greedy search from random starts (Améndola et al. 2020) ends 0.02 to 0.07 lower, but with 10 restarts instead of the paper's 300, and the restarts had not saturated: a lower bound on that method. | `selection_p20`, `search_ceilings`; S4 §4a, §5; S3b §9.3 | revises the earlier "the start matters more than the search"; the random-start search is undersampled |
| 10 | **Example 2 (5-cycle, where the lasso provably fails):** the search recovers the graph exactly from $n = 10^5$ on, on all three losses. At $n \le 10^4$ BIC prefers a re-oriented graph and the search hurts. | S3b §9.1; IS §8 | settled |
| 11 | **The ceiling is information and optimisation.** An oracle that knows the rest of the graph orients an edge correctly with probability 0.77 at $n = 10^3$. The best data-driven methods reach $F_1$ 0.57 / 0.70 / 0.76; a search started from the truth reaches 0.73 / 0.87 / 0.98. | IS §3; 031026 §5 | settled for `C_ID` |
| 12 | **The gains sit in sparse graphs, larger $n$ and larger $p$.** At $p = 10$, $n = 10^3$ the dense start wins for $k = 1, 2$ and loses for $k = 3, 4$; at $p = 20$, $n = 10^4$ it wins at every density (+0.18 … +0.06). | `by_true_c_max_f1`, `by_p`; S4 §2; IS §5 | settled for the direct loss |
| 13 | **With the lasso, the log-likelihood loss is the best of the three and the Frobenius loss the worst.** The loss matters less than the penalty. | S2 §3.3; S2b §3.4 | settled |
| 14 | **Covariance-loss fits depend on the machine** for single datasets (many stationary points); averages agree within 0.01. | S2b §2 | settled |
| 15 | **What limits the search depends on $n$.** At $n = 10^3$ the best-scoring graph is not the true one even with the correct $C$ (30 of 40 graphs), so searching harder does not help. At $n = \infty$ in dense graphs the true graph scores best but the search does not reach it (16 of 20). | 051026 §2 | `C_ID`, $p = 10$, 40 graphs |
| 16 | **The gain with the rescaled $C$ is not "the high-variance node is the parent".** 67 % of the true edges point from the larger to the smaller variance, but a rule that orients the lasso's pairs by variance alone loses to the lasso: `max_f1` 0.570 / 0.596 / 0.611 against 0.637 / 0.673 / 0.696 (MCP dense → sparse: 0.662 / 0.747 / 0.798). | campaign note §2.2 | `C_ID`, $p = 10$, 40 graphs |
| 17 | **Estimating $C$ the way Varando & Hansen do does not recover $C$.** In their package the diagonal either stays at $I$ ($\kappa = 1$) or shrinks towards zero ($\kappa = 0.01$); its correlation with the true diagonal is 0.07 to 0.20. Path metrics change little. | campaign note §2.4 | $p = 10$, 100 graphs, their solver |
| 18 | **For the log-likelihood lasso the order of the path matters little with our solver** (`max_f1` ±0.01, `aupr` +0.02 to +0.03). Varando & Hansen's package, whose fits stop much earlier, is still ahead of our solver in the same order by 0.01 to 0.03 in `max_f1` and 0.03 to 0.05 in `aupr`. | campaign note §2.4, §2.5 | `C_ID`, $p = 10$, 40 graphs; why is open |
| 19 | **The nonconvex penalty is not what helps.** The adaptive lasso (weights from the dense end of the lasso path, convex) matches MCP dense → sparse in `max_f1` and beats it in `aupr` (+0.08 … +0.14), in the BIC-selected graph (+0.02 … +0.08) and after the search (+0.01). LLA gains three quarters of dense → sparse at $p = 20$. | `twobytwo_aupr`, `by_p`; S4 §4 | settled for the direct loss |
| 20 | **Dettling's extended BIC does not change the ranking; it is the better rule where the plain BIC over-selects and the worse one where it does not.** The ranking of the estimators is the same under either rule everywhere. The plain BIC selects 1.2 to 3.3 times the true edges for the lasso at $n = 10^3$ with $C = 2I$, up to 5 times with the rescaled $C$, and the extended term ($\gamma = 1$) halves the excess; it gains up to +0.05 in $F_1$ at $p = 50$ and +0.01 to +0.02 at $p = 20$, $n \ge 10^4$, and loses up to 0.04 (lasso) and 0.02 (dense-start estimators) with $C = 2I$ at $n = 10^3$, $p \le 30$, where the plain BIC is about right. $\gamma = 0.5$ sits between. The path decides the ranking, the rule the level. | `bic_vs_ebic`, `by_p` (bottom row), `selection_checks` (right); S4 §5, §6a; `docs/SEARCH.md` §2a | revised 9 Oct |
| 21 | **With the model exact, the score is right; with a wrong $C$ it is not.** The search started from the truth stays at it for `C_ID` at $n = \infty$ ($F_1$ 0.98 – 0.99) and leaves it in every misspecified setting (0.43 – 0.87). The best data-driven searches end 0.10 – 0.24 below the truth-started one. | `search_ceilings`, `by_true_c_search_f1`; S4 §5; 051026 §2 | settled |
| 22 | **On the log-likelihood loss the dense → sparse MCP path does not beat the lasso in `max_f1` at finite $n$** (−0.05 … −0.07 with $C = 2I$; −0.00 and −0.03 with the rescaled $C$; +0.025 at $n = \infty$), though its `aupr` is better with the rescaled $C$ (+0.06 … +0.14). Its dense start is the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, not the lasso's dense end, which is the likely reason. The log-likelihood lasso matches the direct-loss lasso, and the best estimators on either loss are the direct-loss dense-start ones (0.67 against 0.64 at $p = 10$, $n = 10^4$, rescaled $C$). | `loglik_p10`; S4 §6 | settled for the exact-fit start; the lasso-start variant not run |
| 23 | **The gain grows with $p$ because the lasso deteriorates and the dense-start estimators hardly do.** At $n = 10^3$ the lasso's `max_f1` falls from 0.59 at $p = 10$ to 0.47 / 0.43 at $p = 50$ ($C = 2I$ / rescaled $C$), the adaptive lasso's to 0.55 / 0.57, MCP dense → sparse's to 0.52 / 0.50; the paired gains reach +0.044 / +0.065 (MCP) and +0.078 / +0.133 (adaptive), $z$ 18 … 34. The standard paths only close their deficit because the lasso comes down to them. $p > n$ ($p = 40, 50$) is no obstacle: the gains are largest there. The rescaled $C$ helps the dense-start estimators and hurts the lasso at large $p$. | `by_p`, `gain_by_p`; S4 §6a | settled |
| 24 | **The pure search needs many sparse starting graphs, and the right kind.** At $p = 10$ with the rescaled $C$, 100 sparse starts instead of 10 lift it by +0.05 at $n \ge 10^4$, to within 0.02 of the lasso + search and 0.01 to 0.03 of the adaptive lasso + search, at 10 to 30 times the cost and still rising at $r = 100$; at $n = 10^3$ by +0.01 only, 0.03 to 0.04 behind. 100 uniform starts (Nowzohour et al.'s recipe) end 0.06 to 0.15 below 10 sparse ones: in this class a start of density one half descends into a dense local optimum. | `restarts`; S4 §4a | settled at $p = 10$; $p = 20$ not run |
| 25 | **The least-squares refit is a valid stand-in for the BIC proper.** With the maximised likelihood behind the BIC, the selected and the searched graphs of the lasso, MCP dense → sparse and the adaptive lasso move by at most 0.01 ($p = 10$, $n = 10^4$, both $C$), within one standard error, at 30 to 180 times the cost. | `selection_checks` (left); S4 §5 | settled |
| 26 | **The extended term inside the search helps at $p = 20$.** Dettling's $4\gamma|E|\log p$ with $\gamma = 1$ in the selection and the search adds +0.016 to +0.022 in $F_1$ to every estimator at $p = 20$, $n = 10^4$ ($z$ 6 to 9; 60 → 51 edges for 47 true), nothing at $p = 10$, and leaves the ranking unchanged; in the pure search it is a wash, and at $n = 10^3$ it pulls the truth-started search away from the truth. With verdict 20, the rule to use from $p = 20$ on. | `selection_checks` (right); S4 §5 | settled at $n = 10^4$ |

## The figures

One line on how to read each (`../runs/campaign/figures/`; drawn by
`simulations/diagnostics/plot_campaign.py` from the CSVs next to them; details in S4):

| figure | what it shows | how to read it |
|---|---|---|
| `by_p` | the six estimators over $p = 10 \dots 50$ at $n = 10^3$, both $C$; rows: oracle $F_1$, precision–recall area, BIC-selected $F_1$ | hollow markers are the standard paths, filled the dense-start ones; the lasso (blue) falls with $p$, the adaptive lasso (green) does not |
| `gain_by_p` | `by_p` as paired differences to the lasso, with one standard error | above zero = better than the lasso; the gap opens with $p$ |
| `twobytwo_max_f1` (`_aupr`, `_bic_f1`, `_search_f1`) | the 2 × 2 at $p = 10, 20$ over $n = 10^3, 10^4, \infty$, $C = 2I$ and the rescaled $C$ | same marker code; the gap of the hollow markers grows with $n$, the filled ones sit above the lasso |
| `by_true_c_max_f1` (`_bic_f1`, `_search_f1`) | the gain over the lasso per setting of the true $C$, rescaled $C$ | the gain is the same for every diagonal true $C$ and absent for the non-diagonal one |
| `selection_p20` | $p = 20$, rescaled $C$: oracle $\lambda$, BIC-selected, after the search | the BIC keeps the ranking; the search lifts the lasso most |
| `orientation_p20` | what the BIC-selected graphs consist of: correct, hedged, reversed, false edges | the standard path's loss is reversed edges; the lasso hedges |
| `search_ceilings` | the search from three estimators' graphs (coloured), from 10 random starts (grey) and from the truth (dark) | the random-start search is below, the truth-started one far above everything |
| `restarts` | the pure search with $r = 1 \dots 100$ starting graphs; rows: uniform / sparse draw; columns: $n$ | dashed: best of the first $r$ random starts; solid: plus the empty graph; dotted: wave 2's 10 starts + empty; blue / green: lasso / adaptive lasso + search. Sparse starts reach the blue line at $n \ge 10^4$, uniform ones never |
| `selection_checks` | left: least-squares (grey) against likelihood refit (brown) behind the BIC; right: the term $4\gamma\lvert E\rvert\log p$ in the search, $\gamma = 0$ (grey), 0.5, 1 (purple) | left: the markers coincide; right: purple above grey at $p = 20$ only |
| `loglik_p10` | the log-likelihood loss at $p = 10$ | the dense → sparse MCP (filled orange) is not above the lasso at finite $n$ |

## The evidence, theme by theme

Each theme embeds the figure that carries its verdicts and explains what the figure shows, how
to read it, and which numbers in it support which verdict. The figures are drawn from the
campaign's CSVs (`../runs/campaign/`); the marker code is the same throughout: **blue** the
lasso, **orange** MCP, **aqua** SCAD, **magenta** MCP by LLA, **green** the adaptive lasso;
**hollow** markers with dashed lines are the standard sparse → dense paths, **filled** markers with
solid lines the paths that start from the dense end of the lasso path. Error bars are one
standard error over the graphs of a cell. "Rescaled $C$" means $C = 2\,\mathrm{diag}(1/s_i^2)$ after
standardising, the volatility matrix that is correct for Dettling's generator; "$C = 2I$" is his
pipeline, which ignores the standardisation.

### A. The standard paths lose, and they lose on direction (verdicts 1 – 5)

![](../runs/campaign/figures/twobytwo_max_f1.png)

*Directed $F_1$ at the oracle $\lambda$, the best graph on each path, over $n = 10^3, 10^4, \infty$;
rows $p = 10, 20$; columns $C = 2I$ and the rescaled $C$; 800 graphs per cell.*

**What it shows.** The hollow orange and aqua markers, MCP and SCAD on the standard path, lie
below the blue lasso in every panel and at every $n$. The gap is 0.04 to 0.12 for MCP and 0.03 to
0.07 for SCAD, and it does not close as $n$ grows: at $n = \infty$, where the data are the exact
population covariance, it is the widest. That is verdicts 1 and 2: more data does not help the
standard paths, so the failure is not variance.

![](../runs/campaign/figures/orientation_p20.png)

*The BIC-selected graphs at $p = 20$, $n = 10^4$, rescaled $C$, decomposed over the true
single-direction edges: found with the correct direction, hedged (both directions kept), reversed,
missed; plus false edges.*

**What it shows.** The standard paths find about as many true pairs as the lasso but put more of
them in the wrong direction, while the lasso keeps both directions of a pair far more often.
That is verdict 3: the failure is edge direction, not the skeleton. Verdicts 4 and 5 are the
mechanism, established in S3a and S3b on single paths: for a pair $i, j$ the first of $M_{ij}$,
$M_{ji}$ to enter the path is close to a coin flip, the lasso later adds the other direction when
the data ask for it and corrects, whereas a nonconvex penalty stops shrinking an entry once it is
large, so the first choice is frozen and the reverse entry never gets in. Under directed $F_1$
keeping both directions costs one false positive and gains one true positive, so hedging is the
right response, and the lasso hedges.

### B. Started from the dense end of the lasso path and pruned, they win (verdicts 6, 7, 12, 16, 17, 19)

**What `twobytwo_max_f1` shows for this.** The filled orange and aqua markers, MCP and SCAD run
dense → sparse, sit above the lasso in every cell but one tie: with the rescaled $C$ by +0.016 /
+0.045 / +0.058 at $p = 10$ and +0.056 / +0.115 / +0.146 at $p = 20$ ($n = 10^3 / 10^4 / \infty$), with
$C = 2I$ by −0.009 / +0.012 / +0.026 and +0.018 / +0.060 / +0.086. That is verdict 7. The gains are
larger at larger $n$ and larger $p$, and (S4 §2) they sit in the sparser graphs, verdict 12.

![](../runs/campaign/figures/by_true_c_max_f1.png)

*The same gains as paired differences to the lasso, split by the true $C$ of the data: $C = 2I$
exactly; a random diagonal in $[2, 4]$; a random diagonal in $[0.5, 4]$; a non-diagonal $C$. Rescaled
$C$ in the fit throughout.*

**What it shows.** The gain is the same for every diagonal true $C$, including the ones the fit
gets wrong, and absent for the non-diagonal one. So the dense start does not need the true $C$;
it needs the *structure* of a diagonal $C$ on the standardised scale (verdicts 6 and 16). Verdict 6
also says why $C = 2I$ is a misspecified model even for `C_ID` data: standardising the data
rescales $C$ to $2\,\mathrm{diag}(1/s_i^2)$, and `docs/DENSE_START.md` §7 shows that a diagonal error in
$C$ becomes a *dense* error in $M$, which hurts a method that starts from the dense solution and
not one that starts from zero. Verdict 17 (campaign note §2.4) adds that estimating $C$ the way
Varando & Hansen do does not recover this gain.

![](../runs/campaign/figures/twobytwo_aupr.png)

*Area under the precision–recall curve of the whole path, the same layout as above.*

**What it shows.** The green adaptive lasso, which starts from the same dense solution and prunes
it with a convex weighted-$\ell_1$ step, is the top line in every panel: +0.08 to +0.14 above the
lasso at $p = 10, 20$ and also above MCP dense → sparse. In `twobytwo_max_f1` it matches MCP dense →
sparse within 0.00 to 0.045 in its favour, and in `twobytwo_bic_f1` it leads by +0.02 to +0.08.
That is verdict 19: the nonconvex penalty is not what helps; the dense start is.

### C. The gain grows with $p$ (verdict 23)

The two headline figures at the top of this page, `by_p` and `gain_by_p`, carry this. In `by_p`
the lasso's oracle $F_1$ falls from 0.59 at $p = 10$ to 0.47 ($C = 2I$) and 0.43 (rescaled $C$) at
$p = 50$, the adaptive lasso's stays at 0.55 to 0.57, MCP dense → sparse's ends at 0.52 and 0.50. In
`gain_by_p` the differences to the lasso rise monotonically to +0.078 / +0.133 (adaptive lasso)
and +0.044 / +0.065 (MCP dense → sparse) with $z$ 18 to 34, while the hollow standard paths rise
towards zero only because the lasso comes down to them; they never cross the filled ones. At
$p = 40, 50$ the drift matrix has more free entries than the $n = 10^3$ observations and the gains
are the largest of the sweep. The bottom row, the BIC-selected graph, falls for everyone with $p$
and faster with the rescaled $C$, which is the over-selection of verdict 20 (theme D).

### D. One graph without the truth: BIC, the extended term, the search, and the ceiling (verdicts 8, 9, 11, 15, 20, 21, 25, 26)

![](../runs/campaign/figures/selection_p20.png)

*$p = 20$, rescaled $C$, four estimators under the three ways to get one graph from a path: the
oracle $\lambda$ (needs the truth), the BIC-selected graph, and that graph after the greedy BIC
search.*

**What it shows.** The BIC-selected graphs keep the ranking of the oracle ones with smaller gains
at $n = 10^3$ (MCP dense → sparse +0.05 / +0.11 / +0.09 over the lasso, the adaptive lasso +0.08 /
+0.13 / +0.13). The search, which adds, deletes and reverses single edges while the BIC improves,
then lifts the lasso by +0.09 / +0.14 / +0.12 and the dense-start estimators by less, so that
afterwards every lasso-based start ends within 0.02 of the others, the adaptive lasso on top.
That is verdict 9, and verdict 8's "the search is a useful final step".

![](../runs/campaign/figures/search_ceilings.png)

*The graph the search ends at, from five starts: three estimators' BIC-selected graphs
(coloured), 10 randomly drawn sparse graphs plus the empty graph (grey; the method of Améndola
et al. 2020), and the true graph (dark); rescaled $C$.*

**What it shows.** The grey line, the search without any path, ends 0.02 to 0.07 below every
lasso-based start; verdict 24 (theme E) shows what more starting graphs do about that. The dark
line, the search started from the truth, is a ceiling no method reaches, 0.10 to 0.24 above the
best data-driven one: with `C_ID` at $n = \infty$ it stays at the truth ($F_1$ 0.98 to 0.99), so the
score is right there and the remaining gap is the search getting stuck (verdict 11: information
and optimisation); in the misspecified settings it walks away from the truth (0.81 to 0.87 for the
random diagonals, 0.43 to 0.66 for the non-diagonal $C$), so there the score itself prefers a
wrong graph and no search can fix it (verdict 21). Verdict 15 is the $n$-dependence of this
(campaign note §2): at $n = 10^3$ the best data-driven start is already near what the score can
tell apart; at $n = \infty$ the search is what limits.

**The plain BIC over-selects at large $p$ (verdict 20).** In `by_p`, bottom row, the BIC-selected
$F_1$ falls with $p$ for every estimator, and the table in S4 §6a gives the cause: at $p = 50$ the
lasso's BIC graph has 313 edges with $C = 2I$ and 610 with the rescaled $C$ for 123 true ones, the
adaptive lasso's 161 and 301. Dettling's extended BIC adds $4\gamma\lvert E\rvert\log p$ to the score,
an extra charge per edge that grows with $p$.

![](../runs/campaign/figures/bic_vs_ebic.png)

*The two rules compared, for the lasso, MCP dense → sparse and the adaptive lasso. Top: $F_1$ of
the graph the extended BIC selects minus the one the plain BIC selects, $\gamma = 1$ solid and
$\gamma = 0.5$ dotted; the x-marked dotted lines are the same difference after the search with the
term inside it, where run. Bottom: selected edges over true edges on a log scale, dashed = plain
BIC, solid = $\gamma = 1$; the grey line is the truth. Columns: over $p$ at $n = 10^3$ with $C = 2I$,
the same with the rescaled $C$, over $n$ at $p = 20$ with the rescaled $C$.*

**What it shows.** The bottom row is the diagnosis: the plain BIC selects 1.2 to 3.3 times the true
number of edges for the lasso and up to 5 times with the rescaled $C$, and $\gamma = 1$ halves the
excess, for the adaptive lasso with $C = 2I$ to below the truth. The top row is the consequence:
the extended BIC wins exactly where the plain one over-selects a lot, from $p \approx 30$ on at
$n = 10^3$ (up to +0.05 with the rescaled $C$ at $p = 50$) and from $p = 20$ on at $n \ge 10^4$, and
it *loses* where the plain BIC is already about right, with $C = 2I$ at $n = 10^3$ and $p \le 30$ by
up to 0.04 for the lasso and 0.02 for the dense-start estimators. $\gamma = 0.5$ is the compromise
between the two everywhere. The ranking of the estimators never changes under either rule. That
is the refined verdict 20: the path decides the ranking, the rule the level, and the right rule
depends on whether the plain BIC over-selects.

![](../runs/campaign/figures/selection_checks.png)

*Two checks of the selection step at $n = 10^4$. Left: the BIC-selected (hollow) and searched
(filled) graphs with the campaign's least-squares refit (grey) and with the maximised likelihood
behind the BIC (brown), $p = 10$. Right: the searched graph with the plain BIC (grey, $\gamma = 0$)
and with the extended term inside the selection and the search, $\gamma = 0.5$ and $1$ (purple),
$p = 10$ and $20$.*

**What it shows.** Left: our BIC evaluates the Gaussian likelihood at a cheap least-squares refit
of each support rather than at the maximised likelihood that Améndola et al. and Dettling use. The
brown markers sit on the grey ones for every estimator, selected and searched, within 0.01 and
one standard error, at 30 to 180 times the cost. That is verdict 25: the cheap refit is a valid
stand-in for the BIC proper. Right: at $p = 10$, the three left groups, the purple markers sit on
the grey; at $p = 20$, the three right groups, both purple markers sit about 0.02 above the grey for
every estimator ($z$ 6 to 9), and the graph after the search has 51 edges instead of 60 for 47 true
ones. That is verdict 26: the extended term *inside* the search helps from $p = 20$ on and does not
change the ranking. In the pure search it is a wash, and at $n = 10^3$ it pulls the truth-started
search away from the truth, because at small $n$ it penalises true edges with small weights (S4 §5).

### E. The pure search and its starting graphs (verdict 24)

![](../runs/campaign/figures/restarts.png)

*The greedy BIC search without a path, $p = 10$, rescaled $C$, 400 graphs per panel. Columns:
$n = 10^3, 10^4, \infty$. Rows: the starting graphs drawn uniformly over all directed graphs
(every entry with probability $\tfrac12$, the recipe of Nowzohour et al. 2017 adapted to this
class) or sparsely (edge probability $d \sim U[0, 0.3]$ per graph, the repository's recipe). The
$x$-axis is $r$, the number of randomly drawn starting graphs used; the $y$-axis the $F_1$ of the
graph that the best-scoring of those starts ends at.*

**How to read the lines.** The **dashed grey** curve is the best of the first $r$ random starts
alone. The **solid dark** curve is the same with the empty graph added as one more start, which is
how the pure search was always run. The **dotted horizontal line** is the pure search as run in
wave 2 and shown as the grey line of `search_ceilings`: 10 sparse starts plus the empty graph; by
construction the solid sparse curve meets it at $r = 10$. The **blue** and **green** lines are the
lasso and the adaptive lasso with the BIC search on the same graphs, the data-driven methods the
pure search competes with.

**What it shows.** Bottom row, $n = 10^4$ and $\infty$: the solid curve keeps climbing past $r = 10$
and reaches 0.581 and 0.589 at $r = 100$, within 0.016 and 0.003 of the lasso + search and 0.03 and
0.01 below the adaptive lasso + search, and it has not flattened; the best score of 100 starts is
reached within the first 50 for only 61 to 64 % of the graphs. So at these sample sizes the
10-start pure search of wave 2 was limited by its starts, and with the 300 of Améndola et al. it
would probably match the lasso-based methods, at 10 to 30 times their cost (14 CPU-h per cell
against 0.4 to 4). Bottom left, $n = 10^3$: the curve flattens at $r \approx 20$ at 0.48, below
both lines (0.511, 0.527); more starts do not help there. Top row: 100 uniform starts end 0.06 to
0.15 below 10 sparse ones (dashed curve), and the solid curve is high only because the empty graph
wins; a start with 45 edges for 22 true ones descends into a dense local optimum. That is verdict
24: the pure search needs many sparse starting graphs, and the recipe of the starts matters more
than their number. The "search-pure" rows elsewhere on this page remain the 10-start version, a
lower bound.

### F. The other losses (verdicts 13, 14, 18, 22)

![](../runs/campaign/figures/loglik_p10.png)

*The log-likelihood loss, $p = 10$, 400 graphs per cell: the lasso and MCP in both path orders,
both $C$; $F_1$ at the oracle $\lambda$.*

**What it shows.** On this loss the dense → sparse MCP path (filled orange) is not above the lasso
at finite $n$ (−0.05 to −0.07 with $C = 2I$, −0.00 to −0.03 with the rescaled $C$) and only at
$n = \infty$ with the rescaled $C$ (+0.025); its precision–recall area is better with the rescaled $C$
at every $n$, so its path ranks the entries well but its best single graph is not better. The
likely reason (S4 §6): this path starts from the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, as in Varando
& Hansen, not from the lasso's dense end, which already favours sparse solutions; the variant with
the lasso start is not run (open item 1). The lasso in both orders is within 0.02 of itself
(verdict 18), and the log-likelihood lasso is within 0.01 of the direct-loss lasso (verdict 13
qualified: it was the best of the three losses for the lasso in S2, but the direct-loss dense-start
estimators beat everything on either loss, 0.67 against 0.64 at $n = 10^4$ with the rescaled $C$).
Verdict 14, that covariance-loss fits are machine-dependent for single graphs, is why only means
are reported for this loss.

## The methods side by side

The campaign's numbers (S4; 800 graphs per cell, all four settings of the true $C$, 8 October).
Directed $F_1$ at $p = 20$ with the rescaled $C$, for the three ways to get one graph: the oracle
$\lambda$ (`max_f1`), the BIC-selected graph (`bic_f1`) and that graph after the BIC search
(`search_f1`); $n = 10^3$ / $10^4$ / $\infty$. The same table for $p = 10$ and for $C = 2I$ is in
`runs/campaign/campaign_means.csv`; the figures over $p$ are above and over $n$ in S4 §2.

| method, $p = 20$, rescaled $C$ | oracle $\lambda$ | BIC-selected | after the BIC search |
|---|---|---|---|
| lasso | 0.507 / 0.587 / 0.618 | 0.431 / 0.508 / 0.507 | 0.525 / 0.650 / 0.623 |
| MCP, standard path | 0.464 / 0.534 / 0.546 | 0.379 / 0.428 / 0.356 | 0.464 / 0.510 / 0.398 |
| SCAD, standard path | 0.509 / 0.604 / 0.628 | 0.410 / 0.491 / 0.432 | 0.504 / 0.586 / 0.486 |
| **MCP dense → sparse** | 0.563 / **0.701** / **0.765** | 0.483 / 0.618 / 0.595 | 0.539 / 0.658 / 0.623 |
| SCAD dense → sparse | 0.546 / 0.679 / 0.747 | 0.470 / 0.600 / 0.585 | 0.543 / 0.662 / 0.626 |
| MCP by LLA | 0.531 / 0.666 / 0.731 | 0.446 / 0.570 / 0.584 | 0.545 / 0.655 / 0.629 |
| **adaptive lasso** | **0.587** / **0.712** / **0.767** | **0.512** / **0.635** / **0.635** | 0.543 / **0.666** / **0.635** |
| pure greedy search, 10 randomly drawn starting graphs + empty | – | – | 0.475 / 0.579 / 0.554 |
| *search started from the true graph (ceiling)* | – | – | *0.680 / 0.772 / 0.728* |

With $C = 2I$ (Dettling's pipeline) the same ranking holds with smaller gaps: oracle $\lambda$ at
$n = 10^4$: lasso 0.601, MCP dense → sparse 0.661, adaptive lasso 0.671; after the search 0.614,
0.602, 0.613. At $p = 10$ the gaps are a third of these. Over $p$ at $n = 10^3$ they grow to
+0.065 (MCP dense → sparse) and +0.133 (adaptive lasso) at $p = 50$ (verdict 23).

## Tried, with little or no gain

- **A larger $\gamma$ on the standard path:** it only moves the path towards the lasso. A small
  gain appears only at $n = \infty$ with the rescaled $C$ and $\gamma \ge 30$ (IS §6).
- **GMC,** a nonconvex penalty with a convex objective: +0.01 to +0.04 over the lasso (IS §10).
- **Forward stepwise selection and greedy search from the empty graph,** at $p = 10$: they commit
  early, like the standard MCP path (IS §5; S3b §9.0). Forward stepwise does better at $p = 20$
  with the rescaled $C$ (IS §11).
- **The pure search from randomly drawn graphs** on Example 2 and at $n = 10^3$: the starts get stuck in
  reversed orientations (S3b §9.1, §9.3).

## What the two papers do about $C$

*Longer, with the argument for and against rescaling:
[`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md) §2.*

**Neither paper rescales $C$.** Dettling, Drton & Kolar (2024) fit with $C = 2I_p$ throughout.

- **Simulations (their §5):** four choices of the true $C$; "we apply the Direct Lyapunov Lasso with
  $C = 2I_p$" in all of them. Misspecified $C$ is treated as something the lasso is robust to.
- **Data application (their §6):** every column is standardised, the lasso is run with $C = 2I_p$,
  and the graph is chosen by an extended BIC whose likelihood uses $\Sigma(M, 2I_p)$.
- **Rescaling $C$ after standardising is not mentioned.** For the lasso path it makes almost no
  difference (`max_f1` 0.620 against 0.621), so their Figure 5 looks the same either way. For their
  BIC step it should matter (verdict 6). A question for the 10–11 October meeting.
- **A caveat for real data:** there the true $C$ is unknown on any scale. "$C = 2I$ on the raw
  scale" and "$C = 2I$ on the standardised scale" are two different assumptions, and neither is the
  correct one a priori. In simulations whose data are generated with $C = 2I$, the rescaled $C$ is
  the correct one.

Varando & Hansen (2020) estimate $C$ instead.

- **Estimator (their eq. 7):** a diagonal $C$ is estimated together with the drift matrix, with a
  ridge penalty $\kappa\lVert C - I\rVert_F^2$ that fixes its scale. $\kappa = \infty$ fixes
  $C = I$; that is the version the repository implements and every run so far used.
- **Their simulation:** true $C$ diagonal and random, data standardised, so their fixed-$C$ methods
  are misspecified as Dettling's are. The version that estimates $C$ ($\kappa = 0.01$) is "highly
  similar", with a better precision-recall curve.
- **Why an estimated diagonal needs no rescaling:** standardising turns a diagonal $C$ into another
  diagonal $C$, so the model stays inside the family.
- **Their paths run dense → sparse,** and they report that order as better. All log-likelihood
  runs of the thesis so far ran sparse → dense.

## Open

*The campaign is planned in [`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md)
and reported in [`S4_campaign.md`](S4_campaign.md).*

1. The log-likelihood MCP path started from the log-likelihood lasso's dense end (S4 §6; one
   option in `covloss_path`, about 60 CPU-h): does the gain of the direct loss appear on this
   loss too once the start is the same?
2. Why the plain BIC selects more edges with the rescaled $C$ than with $C = 2I$ at large $p$
   (S4 §6a). The extended term in the search removes part of the over-selection at $p = 20$
   (verdict 26); whether it does at $p = 40, 50$ was not run.
3. A data-generating process in which direction is identifiable (stronger edges, no 2-cycles): are
   the gains larger there, as IS §7 finds?
4. Theory: why the standard path locks in a direction, and why the dense start resolves the hedge
   (`../next_steps/051026/orientation_lock_in.md`, `docs/DENSE_START.md` §7).
5. The gain at $n = 1000$ is real at $p = 20$ (+0.056, $z = 12$) and small at $p = 10$ (+0.016,
   $z = 3$); for the BIC-selected graph at $p = 10$ it is zero.
6. Settled (verdict 25). Wave 5a: the BIC with the maximised likelihood instead of the
   least-squares refit (`--refit loglik`) for the lasso, MCP dense → sparse and the adaptive lasso
   at $p = 10$. Does the ranking of verdicts 7 and 9 hold under the BIC proper?
7. Settled at $p = 10$ (verdict 24). Wave 5b: 100 randomly drawn starting graphs, sparse and uniform,
   for the pure search. Is its gap of 0.02 to 0.07 to the lasso-based starts a matter of too few
   starts? Found on the way: with the rescaled $C$ the empty start is dead (BIC $= +\infty$) in
   6 to 25 % of the graphs (S4 §4a).
8. Settled at $n = 10^4$ (verdict 26); wave 6 (`cluster/plan_091026.txt`, not yet submitted)
   rescores every wave 1 cell with the term inside the search, so that every figure exists under
   that rule (`runs/campaign/figures_ebic1/`), and runs 300 starting graphs for the pure search.
   Wave 5c: the extended BIC term ($4\gamma|E|\log p$, $\gamma = 0.5,
   1$) inside the selection and the search, next to the plain BIC on the same path, and in the
   pure search. So far it was used offline on the path only (verdict 20: $\pm 0.03$ at $\gamma = 1$).
