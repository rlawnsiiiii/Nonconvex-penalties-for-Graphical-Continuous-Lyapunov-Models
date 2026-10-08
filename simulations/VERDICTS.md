# Verdicts so far

*As of 8 October 2026. One page; the evidence is in the documents named in each row. "IS" is the
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

| # | verdict | evidence | status |
|---|---|---|---|
| 1 | **Standard MCP / SCAD lose to the lasso.** Dettling's pipeline (standardised data, $C = 2I$), path from $\lambda_{\max}$ to the dense end: worse than the lasso on `max_f1`, `auc`, `aupr`, for the direct, log-likelihood and Frobenius losses, $p = 10, 20$, all four $C$ settings. | S2 §3.1–3.3; S2b §3 | settled |
| 2 | **More data does not help them.** The gap grows from $n = 10^3$ to $\infty$ (MCP − lasso in `max_f1`: −0.08 … −0.12 at $10^3$, −0.10 … −0.16 at $\infty$). | S2b §3.2, §3.5 | settled |
| 3 | **The failure is edge direction.** MCP and SCAD find nearly the same pairs as the lasso (skeleton $F_1$ 0.02–0.08 lower) but keep one direction per pair; MCP reverses 2–4 times as many edges. They almost never recover both directions of a 2-cycle. | S2 §3.4; S3a; S2b §3.6 | settled |
| 4 | **The direction is fixed too early.** The first direction to enter the path is the true one only 51–61 % of the time, for every penalty. The lasso later corrects it by keeping both directions; MCP keeps the first one. | S3a §4.4 | settled |
| 5 | **Keeping both directions is the right response under $F_1$.** Committing to one direction pays only if it is right more than about 70 % of the time; at $n = 1000$ the estimators manage 54–63 % on the pairs in question. | 021026 §1; IS §3 | settled |
| 6 | **Dettling's pipeline fits a misspecified model, even for `C_ID`.** Standardising changes the volatility to $C = 2\,\mathrm{diag}(1/s_i^2)$ ($s_i$: the standard deviations); the pipeline keeps $C = 2I$. The lasso is robust to it; BIC and other likelihood-based decisions are not. | IS §2; 031026 §3.1 | settled |
| 7 | **MCP / SCAD run dense → sparse beat the lasso.** Paired `max_f1` gain over the lasso with the same $C$, rescaled $C$: +0.016 / +0.045 / +0.058 at $p = 10$ and +0.056 / +0.115 / +0.146 at $p = 20$ ($n = 10^3 / 10^4 / \infty$); with $C = 2I$: −0.009 / +0.012 / +0.026 and +0.018 / +0.060 / +0.086. The gain is of the same size for all three diagonal settings of the true $C$, absent for the non-diagonal one at $p = 10$, small at $p = 20$. | S4 §2–3 (800 graphs per cell); IS §4–5; 031026 §4 | settled for the direct loss |
| 8 | **A BIC search with add / delete / reverse moves is a useful finishing step, on a correctly specified model only.** On Dettling's pipeline it does not improve on the lasso. | S3b §9.2–9.4; IS §9 | settled for `C_ID`; weaker with random $C$ |
| 9 | **The search equalises the lasso-based starts.** It lifts the lasso's BIC-selected graph by +0.09 to +0.14 at $p = 20$ and the dense-start estimators by less; afterwards all lasso-based starts lie within 0.02 of each other, the adaptive lasso on top by +0.01. Starts from the standard MCP / SCAD paths stay far behind (their reversed edges survive). The greedy search from random starts (Améndola et al. 2020) ends 0.02 to 0.07 lower, but with 10 restarts instead of the paper's 300, and the restarts had not saturated: a lower bound on that method. | S4 §4a, §5; S3b §9.3 | revises the earlier "the start matters more than the search"; the random-start search is undersampled |
| 10 | **Example 2 (5-cycle, where the lasso provably fails):** the search recovers the graph exactly from $n = 10^5$ on, on all three losses. At $n \le 10^4$ BIC prefers a re-oriented graph and the search hurts. | S3b §9.1; IS §8 | settled |
| 11 | **The ceiling is information and optimisation.** An oracle that knows the rest of the graph orients an edge correctly with probability 0.77 at $n = 10^3$. The best data-driven methods reach $F_1$ 0.57 / 0.70 / 0.76; a search started from the truth reaches 0.73 / 0.87 / 0.98. | IS §3; 031026 §5 | settled for `C_ID` |
| 12 | **The gains sit in sparse graphs, larger $n$ and larger $p$.** At $p = 10$, $n = 10^3$ the dense start wins for $k = 1, 2$ and loses for $k = 3, 4$; at $p = 20$, $n = 10^4$ it wins at every density (+0.18 … +0.06). | S4 §2; IS §5 | settled for the direct loss |
| 13 | **With the lasso, the log-likelihood loss is the best of the three and the Frobenius loss the worst.** The loss matters less than the penalty. | S2 §3.3; S2b §3.4 | settled |
| 14 | **Covariance-loss fits depend on the machine** for single datasets (many stationary points); averages agree within 0.01. | S2b §2 | settled |
| 15 | **What limits the search depends on $n$.** At $n = 10^3$ the best-scoring graph is not the true one even with the correct $C$ (30 of 40 graphs), so searching harder does not help. At $n = \infty$ in dense graphs the true graph scores best but the search does not reach it (16 of 20). | 051026 §2 | `C_ID`, $p = 10$, 40 graphs |
| 16 | **The gain with the rescaled $C$ is not "the high-variance node is the parent".** 67 % of the true edges point from the larger to the smaller variance, but a rule that orients the lasso's pairs by variance alone loses to the lasso: `max_f1` 0.570 / 0.596 / 0.611 against 0.637 / 0.673 / 0.696 (MCP dense → sparse: 0.662 / 0.747 / 0.798). | campaign note §2.2 | `C_ID`, $p = 10$, 40 graphs |
| 17 | **Estimating $C$ the way Varando & Hansen do does not recover $C$.** In their package the diagonal either stays at $I$ ($\kappa = 1$) or shrinks towards zero ($\kappa = 0.01$); its correlation with the true diagonal is 0.07 to 0.20. Path metrics change little. | campaign note §2.4 | $p = 10$, 100 graphs, their solver |
| 18 | **For the log-likelihood lasso the order of the path matters little with our solver** (`max_f1` ±0.01, `aupr` +0.02 to +0.03). Varando & Hansen's package, whose fits stop much earlier, is still ahead of our solver in the same order by 0.01 to 0.03 in `max_f1` and 0.03 to 0.05 in `aupr`. | campaign note §2.4, §2.5 | `C_ID`, $p = 10$, 40 graphs; why is open |
| 19 | **The nonconvex penalty is not what helps.** The adaptive lasso (weights from the dense end of the lasso path, convex) matches MCP dense → sparse in `max_f1` and beats it in `aupr` (+0.08 … +0.14), in the BIC-selected graph (+0.02 … +0.08) and after the search (+0.01). LLA gains three quarters of dense → sparse at $p = 20$. | S4 §4 | settled for the direct loss |
| 20 | **Dettling's extended BIC does not change the ranking, and at large $p$ it is the better rule.** With $\gamma = 1$ the selected $F_1$ moves by at most ±0.03 at $p = 10, 20$ and the ranking of the estimators not at all; the lasso still selects 80 to 93 edges for 47 true ones at $p = 20$. From $p = 25$ on the plain BIC over-selects badly (the lasso 313 / 610 edges for 123 true at $p = 50$ with $C = 2I$ / the rescaled $C$) and the extended BIC halves the excess and gives a higher $F_1$ for every estimator (adaptive lasso 0.444 against 0.395 with the rescaled $C$ at $p = 50$). The path decides the ranking, the rule the level. | S4 §5, §6a; `docs/SEARCH.md` §2a | revised 8 Oct |
| 21 | **With the model exact, the score is right; with a wrong $C$ it is not.** The search started from the truth stays at it for `C_ID` at $n = \infty$ ($F_1$ 0.98 – 0.99) and leaves it in every misspecified setting (0.43 – 0.87). The best data-driven searches end 0.10 – 0.24 below the truth-started one. | S4 §5; 051026 §2 | settled |
| 22 | **On the log-likelihood loss the dense → sparse MCP path does not beat the lasso in `max_f1` at finite $n$** (−0.05 … −0.07 with $C = 2I$; −0.00 and −0.03 with the rescaled $C$; +0.025 at $n = \infty$), though its `aupr` is better with the rescaled $C$ (+0.06 … +0.14). Its dense start is the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, not the lasso's dense end, which is the likely reason. The log-likelihood lasso matches the direct-loss lasso, and the best estimators on either loss are the direct-loss dense-start ones (0.67 against 0.64 at $p = 10$, $n = 10^4$, rescaled $C$). | S4 §6 | settled for the exact-fit start; the lasso-start variant not run |
| 23 | **The gain grows with $p$ because the lasso deteriorates and the dense-start estimators hardly do.** At $n = 10^3$ the lasso's `max_f1` falls from 0.59 at $p = 10$ to 0.47 / 0.43 at $p = 50$ ($C = 2I$ / rescaled $C$), the adaptive lasso's to 0.55 / 0.57, MCP dense → sparse's to 0.52 / 0.50; the paired gains reach +0.044 / +0.065 (MCP) and +0.078 / +0.133 (adaptive), $z$ 18 … 34. The standard paths only close their deficit because the lasso comes down to them. $p > n$ ($p = 40, 50$) is no obstacle: the gains are largest there. The rescaled $C$ helps the dense-start estimators and hurts the lasso at large $p$. | S4 §6a | settled |
| 24 | **Preliminary: ten times the starting graphs do not rescue the pure search.** At $p = 10$, rescaled $C$, $n = 10^3$, 100 sparse starts raise its $F_1$ from 0.475 to 0.486; the score has not saturated but the $F_1$ has (+0.005 from 50 to 100 starts); the lasso + search is at 0.511, the adaptive lasso + search at 0.527. | S4 §4a | one of six cells; uniform starts and $n \ge 10^4$ running |

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
   (S4 §6a), and whether the search with the extended term (wave 5c) removes the over-selection.
3. A data-generating process in which direction is identifiable (stronger edges, no 2-cycles): are
   the gains larger there, as IS §7 finds?
4. Theory: why the standard path locks in a direction, and why the dense start resolves the hedge
   (`../next_steps/051026/orientation_lock_in.md`, `docs/DENSE_START.md` §7).
5. The gain at $n = 1000$ is real at $p = 20$ (+0.056, $z = 12$) and small at $p = 10$ (+0.016,
   $z = 3$); for the BIC-selected graph at $p = 10$ it is zero.
6. Wave 5a (running since 8 October): the BIC with the maximised likelihood instead of the
   least-squares refit (`--refit loglik`) for the lasso, MCP dense → sparse and the adaptive lasso
   at $p = 10$. Does the ranking of verdicts 7 and 9 hold under the BIC proper?
7. Wave 5b (running since 8 October; first cell in, verdict 24): 100 randomly drawn starting graphs, sparse and uniform,
   for the pure search. Is its gap of 0.02 to 0.07 to the lasso-based starts a matter of too few
   starts? Found on the way: with the rescaled $C$ the empty start is dead (BIC $= +\infty$) in
   6 to 25 % of the graphs (S4 §4a).
8. Wave 5c (running since 8 October): the extended BIC term ($4\gamma|E|\log p$, $\gamma = 0.5,
   1$) inside the selection and the search, next to the plain BIC on the same path, and in the
   pure search. So far it was used offline on the path only (verdict 20: $\pm 0.03$ at $\gamma = 1$).
