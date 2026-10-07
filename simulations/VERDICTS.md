# Verdicts so far

*As of 7 October 2026. One page; the evidence is in the documents named in each row. "IS" is the
independent study (`../next_steps/021026/independent_study/independent_study_021026.md`), "031026"
the combined note (`../next_steps/031026/next_steps_031026.md`), "campaign note"
`../next_steps/051026/cluster_campaign_051026.md`, "S4" the campaign's results (`S4_campaign.md`).*

## Bottom line

Used the standard way, nonconvex penalties do not beat the lasso for GCLMs, on any loss and at any
sample size. They fail on the direction of edges, because the standard path fixes a direction too
early. Started from the dense lasso solution and used to prune it, they beat the lasso, by more as
$p$ and $n$ grow, for every diagonal true $C$ and under Dettling's own pipeline as well as with the
correctly specified $C$ (S4, 800 graphs per cell). But the convex adaptive lasso, which prunes the
same start, does at least as well on every metric: the gain comes from the start, not from the
nonconvex penalty.

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
| 20 | **Dettling's extended BIC does not change the picture.** With $\gamma = 1$ the selected $F_1$ moves by at most ±0.03 and the ranking of the estimators not at all; the lasso still selects 80 to 93 edges for 47 true ones at $p = 20$. The path, not the selection rule, decides. | S4 §5; `docs/SEARCH.md` §2a | settled |
| 21 | **With the model exact, the score is right; with a wrong $C$ it is not.** The search started from the truth stays at it for `C_ID` at $n = \infty$ ($F_1$ 0.98 – 0.99) and leaves it in every misspecified setting (0.43 – 0.87). The best data-driven searches end 0.10 – 0.24 below the truth-started one. | S4 §5; 051026 §2 | settled |
| 22 | **Preliminary: on the log-likelihood loss the dense → sparse MCP path does not beat the lasso at finite $n$** (−0.05 … −0.07 with $C = 2I$; 0.00 and −0.04 with the rescaled $C$; +0.02 at $n = \infty$). Its dense start is the exact fit $-\tfrac12 C\hat\Sigma^{-1}$, not the lasso's dense end, which is the likely reason. | S4 §6 | wave 3 half complete; the lasso-start variant not run |

## The methods side by side

Directed $F_1$ of the graph each method selects from the data (BIC), `C_ID`, $p = 10$, 40 graphs
(3 October; the campaign's version with 800 graphs per cell and all four settings is S4 §5):

| method | model | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|---|
| lasso (Dettling) | standardised, $C = 2I$ | 0.549 | 0.624 | 0.601 |
| lasso | standardised, rescaled $C$ | 0.567 | 0.625 | 0.644 |
| MCP, standard path | rescaled $C$ | 0.514 | 0.566 | 0.534 |
| lasso + BIC search | rescaled $C$ | 0.569 | 0.670 | 0.724 |
| **MCP dense → sparse** | rescaled $C$ | 0.571 | **0.691** | **0.739** |
| **MCP dense → sparse + BIC search** | rescaled $C$ | 0.574 | **0.702** | **0.760** |
| lasso + BIC search | raw scale | 0.573 | 0.715 | 0.721 |
| pure greedy search, 10 randomly drawn starting graphs | raw scale | 0.565 | 0.666 | 0.747 |
| *search started from the true graph (oracle)* | raw scale | *0.732* | *0.865* | *0.980* |

For `C_ID` the raw scale and the rescaled $C$ are the same, correctly specified model. They differ
only in how the loss and the penalty weight the entries of $M$.

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

1. Wave 3 complete, and the log-likelihood MCP path started from the log-likelihood lasso's dense
   end (S4 §6).
2. Wave 4: $p = 15 \dots 50$ at $n = 1000$ for the thesis figure (running).
3. A data-generating process in which direction is identifiable (stronger edges, no 2-cycles): are
   the gains larger there, as IS §7 finds?
4. Theory: why the standard path locks in a direction, and why the dense start resolves the hedge
   (`../next_steps/051026/orientation_lock_in.md`, `docs/DENSE_START.md` §7).
5. The gain at $n = 1000$ is real at $p = 20$ (+0.056, $z = 12$) and small at $p = 10$ (+0.016,
   $z = 3$); for the BIC-selected graph at $p = 10$ it is zero.
6. Wave 5a (implemented, not submitted): the BIC with the maximised likelihood instead of the
   least-squares refit (`--refit loglik`) for the lasso, MCP dense → sparse and the adaptive lasso
   at $p = 10$. Does the ranking of verdicts 7 and 9 hold under the BIC proper?
7. Wave 5b (implemented, not submitted): 100 randomly drawn starting graphs, sparse and uniform,
   for the pure search. Is its gap of 0.02 to 0.07 to the lasso-based starts a matter of too few
   starts? Found on the way: with the rescaled $C$ the empty start is dead (BIC $= +\infty$) in
   6 to 25 % of the graphs (S4 §4a).
8. Wave 5c (implemented, not submitted): the extended BIC term ($4\gamma|E|\log p$, $\gamma = 0.5,
   1$) inside the selection and the search, next to the plain BIC on the same path, and in the
   pure search. So far it was used offline on the path only (verdict 20: $\pm 0.03$ at $\gamma = 1$).
