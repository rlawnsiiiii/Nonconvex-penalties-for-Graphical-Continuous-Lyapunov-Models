# Verdicts so far

*As of 5 October 2026. One page; the evidence is in the documents named in each row. "IS" is the
independent study (`../next_steps/021026/independent_study/independent_study_021026.md`), "031026"
the combined note (`../next_steps/031026/next_steps_031026.md`), "campaign note"
`../next_steps/051026/cluster_campaign_051026.md`.*

## Bottom line

Used the standard way, nonconvex penalties do not beat the lasso for GCLMs, on any loss and at any
sample size. They fail on the direction of edges, because the standard path fixes a direction too
early. Used to prune a dense lasso solution, on a correctly specified model, they do beat the
lasso. That positive result is so far shown for one of Dettling's four settings (`C_ID`).

## The verdicts

| # | verdict | evidence | status |
|---|---|---|---|
| 1 | **Standard MCP / SCAD lose to the lasso.** Dettling's pipeline (standardised data, $C = 2I$), path from $\lambda_{\max}$ to the dense end: worse than the lasso on `max_f1`, `auc`, `aupr`, for the direct, log-likelihood and Frobenius losses, $p = 10, 20$, all four $C$ settings. | S2 §3.1–3.3; S2b §3 | settled |
| 2 | **More data does not help them.** The gap grows from $n = 10^3$ to $\infty$ (MCP − lasso in `max_f1`: −0.08 … −0.12 at $10^3$, −0.10 … −0.16 at $\infty$). | S2b §3.2, §3.5 | settled |
| 3 | **The failure is edge direction.** MCP and SCAD find nearly the same pairs as the lasso (skeleton $F_1$ 0.02–0.08 lower) but keep one direction per pair; MCP reverses 2–4 times as many edges. They almost never recover both directions of a 2-cycle. | S2 §3.4; S3a; S2b §3.6 | settled |
| 4 | **The direction is fixed too early.** The first direction to enter the path is the true one only 51–61 % of the time, for every penalty. The lasso later corrects it by keeping both directions; MCP keeps the first one. | S3a §4.4 | settled |
| 5 | **Keeping both directions is the right response under $F_1$.** Committing to one direction pays only if it is right more than about 70 % of the time; at $n = 1000$ the estimators manage 54–63 % on the pairs in question. | 021026 §1; IS §3 | settled |
| 6 | **Dettling's pipeline fits a misspecified model, even for `C_ID`.** Standardising changes the volatility to $C = 2\,\mathrm{diag}(1/s_i^2)$ ($s_i$: the standard deviations); the pipeline keeps $C = 2I$. The lasso is robust to it; BIC and other likelihood-based decisions are not. | IS §2; 031026 §3.1 | settled |
| 7 | **MCP / SCAD run dense → sparse beat the lasso,** with the rescaled $C$. `max_f1`, lasso vs. MCP dense → sparse: 0.637 / 0.662 ($n = 10^3$), 0.673 / 0.747 ($10^4$), 0.696 / 0.798 ($\infty$). With $C = 2I$ they only tie the lasso up to $n = 10^4$. | IS §4–5; 031026 §4 (replicated with the repository's solvers) | holds for `C_ID`, $p = 10$; **open for the other three $C$ settings** |
| 8 | **A BIC search with add / delete / reverse moves is a useful finishing step, on a correctly specified model only.** On Dettling's pipeline it does not improve on the lasso. | S3b §9.2–9.4; IS §9 | settled for `C_ID`; weaker with random $C$ |
| 9 | **The graph the search starts from matters more than the search.** Best start: MCP dense → sparse. Then the lasso. Worst: the standard MCP / SCAD paths, whose reversed edges survive the search. | 031026 §4; S3b §9.3; IS §9 | settled for `C_ID` |
| 10 | **Example 2 (5-cycle, where the lasso provably fails):** the search recovers the graph exactly from $n = 10^5$ on, on all three losses. At $n \le 10^4$ BIC prefers a re-oriented graph and the search hurts. | S3b §9.1; IS §8 | settled |
| 11 | **The ceiling is information and optimisation.** An oracle that knows the rest of the graph orients an edge correctly with probability 0.77 at $n = 10^3$. The best data-driven methods reach $F_1$ 0.57 / 0.70 / 0.76; a search started from the truth reaches 0.73 / 0.87 / 0.98. | IS §3; 031026 §5 | settled for `C_ID` |
| 12 | **The gains sit in sparse graphs and larger $n$.** For $k = 3, 4$ and at $n = 10^3$ the lasso is as good or within noise. | IS §5; 031026 §4 | settled for `C_ID` |
| 13 | **With the lasso, the log-likelihood loss is the best of the three and the Frobenius loss the worst.** The loss matters less than the penalty. | S2 §3.3; S2b §3.4 | settled |
| 14 | **Covariance-loss fits depend on the machine** for single datasets (many stationary points); averages agree within 0.01. | S2b §2 | settled |
| 15 | **What limits the search depends on $n$.** At $n = 10^3$ the best-scoring graph is not the true one even with the correct $C$ (30 of 40 graphs), so searching harder does not help. At $n = \infty$ in dense graphs the true graph scores best but the search does not reach it (16 of 20). | 051026 §2 | `C_ID`, $p = 10$, 40 graphs |
| 16 | **The gain with the rescaled $C$ is not "the high-variance node is the parent".** 67 % of the true edges point from the larger to the smaller variance, but a rule that orients the lasso's pairs by variance alone loses to the lasso: `max_f1` 0.570 / 0.596 / 0.611 against 0.637 / 0.673 / 0.696 (MCP dense → sparse: 0.662 / 0.747 / 0.798). | campaign note §2.2 | `C_ID`, $p = 10$, 40 graphs |
| 17 | **Estimating $C$ the way Varando & Hansen do does not recover $C$.** In their package the diagonal either stays at $I$ ($\kappa = 1$) or shrinks towards zero ($\kappa = 0.01$); its correlation with the true diagonal is 0.07 to 0.20. Path metrics change little. | campaign note §2.4 | $p = 10$, 100 graphs, their solver |
| 18 | **For the log-likelihood lasso the order of the path matters little with our solver** (`max_f1` ±0.01, `aupr` +0.02 to +0.03). Varando & Hansen's package, whose fits stop much earlier, is still ahead of our solver in the same order by 0.01 to 0.03 in `max_f1` and 0.03 to 0.05 in `aupr`. | campaign note §2.4, §2.5 | `C_ID`, $p = 10$, 40 graphs; why is open |

## The methods side by side

Directed $F_1$ of the graph each method selects from the data (BIC), `C_ID`, $p = 10$, 40 graphs:

| method | model | $n = 10^3$ | $10^4$ | $\infty$ |
|---|---|---|---|---|
| lasso (Dettling) | standardised, $C = 2I$ | 0.549 | 0.624 | 0.601 |
| lasso | standardised, rescaled $C$ | 0.567 | 0.625 | 0.644 |
| MCP, standard path | rescaled $C$ | 0.514 | 0.566 | 0.534 |
| lasso + BIC search | rescaled $C$ | 0.569 | 0.670 | 0.724 |
| **MCP dense → sparse** | rescaled $C$ | 0.571 | **0.691** | **0.739** |
| **MCP dense → sparse + BIC search** | rescaled $C$ | 0.574 | **0.702** | **0.760** |
| lasso + BIC search | raw scale | 0.573 | 0.715 | 0.721 |
| pure greedy search, 10 random restarts | raw scale | 0.565 | 0.666 | 0.747 |
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
- **The pure search with random restarts** on Example 2 and at $n = 10^3$: restarts get stuck in
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

*The proposed order of work is in [`../next_steps/051026/next_steps_051026.md`](../next_steps/051026/next_steps_051026.md);
the cluster campaign that covers items 1 – 3 is planned in
[`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md).*

1. Do verdicts 7–9 hold in the three settings where the true $C$ is not $2I$? (031026, Step 2)
2. $p = 20$ and larger $n$ on the cluster with the rescaled $C$ and both path directions. (Step 3)
3. The log-likelihood and Frobenius losses with the rescaled $C$ and a dense start. Untested.
4. A data-generating process in which direction is identifiable (stronger edges, no 2-cycles): are
   the gains larger there, as IS §7 finds? (Step 5)
5. Is the gain at $n = 1000$ real? It is +0.025 in `max_f1` with $z \approx 1$–2 so far.
