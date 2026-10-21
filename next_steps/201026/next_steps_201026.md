# Next steps: the adaptive lasso as the first theory target (20 October 2026)

*Decision taken on 20 October: the theory part starts with the adaptive lasso, not with MCP/SCAD. Reasons:
the problem is convex, so the theorem is about the estimator that is actually computed (global minimiser);
it is the arm that won the campaign (`VERDICTS.md`, dense-start estimators); the deterministic theorem is
already written (`claude/theory_conditions_and_proofs_201026.md`, §3) and everything that is still open sits in
one object, the pilot. MCP/SCAD come afterwards as "LLA = adaptive lasso with saturated weights".*

Status words as in the other memos: **proved / provable / cite / numerical / open.**

---

## 1. What is settled (do not redo)

| statement | status | where |
|---|---|---|
| Weighted-lasso PDW for GCLMs: with $\Gamma^*_{SS}$ invertible, DDK's noise bounds, $\rho_{\rm ad}\le1-\alpha$, (B1) $\lambda\min_{S^c}\hat w_j\,\alpha>(1+\hat\rho)\varepsilon$, (B2) $\beta_{\min}>2c_{\Gamma^*}(\varepsilon+\lambda\max_S\hat w_k)$ ⟹ unique solution, $\hat S=S$, correct signs | proved (line by line against DDK Thm D.1) | 20 Oct note §3.3 |
| Necessity: fixed limiting weights with $\rho_{\rm ad}>1$ ⟹ $P(\hat S=S)\to0$ | proved (DDK Prop. G.2 verbatim) | 20 Oct note §3.3 |
| Pilot accuracy ⟹ weighted condition: $\|\text{pilot}-m^*\|_\infty\le\delta<\beta_{\min}/(1+\hat\rho)$ ⟹ $\rho_{\rm ad}\le\hat\rho\,\delta/(\beta_{\min}-\delta)<1$ | proved (ZvdGB (8.5) in our notation) | 20 Oct note §3.4 |
| Population limit of the lasso dense end = minimum-$\ell_1$ exact fit (BP solution); BP solution is either $M^*$ ($\rho_{\rm ad}=0$) or misses a true edge ($\rho_{\rm ad}=\infty$) | numerical (2 160 instances) + the LP `min_l1_exact_fit` | 15 Oct memo §3, Table 2 |
| On the correlation scale $\rho\ge1$ for every graph; the violating non-edges are the reversed true edges | proved near the diagonal, numerical away from it | 15 Oct memo §2.3 |
| Exact characterisation of the BP condition by a dual certificate (LP) | cite (Fuchs 2004, Tropp 2006) + `bp_certificate` | §4 below |

Definitions used below (diagonal unpenalised, $\hat w_j = 1/|\hat m^{\rm pilot}_j|$, $\hat w_j=\infty$ if the pilot entry is 0):
$$\rho_{\rm ad}(\text{pilot}) = \max_{j\in S^c}\ \frac{1}{\hat w_j}\Big|\sum_{k\in S_{\rm off}}\big[\Gamma_{S^cS}\Gamma_{SS}^{-1}\big]_{jk}\,\hat w_k\,\operatorname{sign}(m^*_k)\Big| ,
\qquad\text{near the diagonal (corr. scale)}:\ \ \rho_{\rm ad}\approx\max_{l\to k\in E}\frac{|\hat m^{\rm pilot}_{(l,k)}|}{|\hat m^{\rm pilot}_{(k,l)}|}.$$
Two failure modes of a pilot, and only two: it **misses** a true edge (then $\hat w=\infty$ there and the adaptive
lasso can never recover it), or it puts **more weight on the reversed than on the true direction** of some edge.

---

## 2. Reading (in this order, with what to extract)

1. **Zhou, van de Geer & Bühlmann (2009)**, *Adaptive Lasso for high dimensional regression and Gaussian graphical modeling*, arXiv:0903.2515. Read §2 (Thm 2.1: any pilot with sup-norm accuracy $\tilde\delta_S,\tilde\delta_{S^c}$), §8 (Def. 8.1 "$(\vec w,S)$-incoherence", eq. (8.5), Lemma 8.2) and §12 (the KKT proof = Wainwright's Lemmas 2–3). Extract: the exact form of (8.4a) and how (2.12)–(2.14) are derived from it — our (B1)–(B2) should match after the substitution $X^\top X/n\to\hat\Gamma$, $X^\top\varepsilon/n\to w$.
2. **Wainwright (2009)**, *Sharp thresholds …*, IEEE Trans. IT 55(5): §II–III, Lemma 1 (uniqueness from strict dual feasibility), the PDW construction. This is the machinery DDK use; read it once properly.
3. **Huang, Ma & Zhang (2008)**, *Adaptive Lasso for sparse high-dimensional regression models*, Statist. Sinica 18(4):1603–1618. Extract their "adaptive irrepresentable" condition and the notion of a *zero-consistent* pilot; note that their partial-orthogonality route (marginal regression as pilot) does not transfer — our design columns are all correlated by construction. Check the exact form of their condition against ours (the 20 Oct note cites it from the abstract only).
4. **Zou (2006)**, JASA 101(476), Thm 2: fixed $p$, $\sqrt n$-consistent pilot, $\lambda_n/\sqrt n\to0$, $\lambda_nn^{(\gamma-1)/2}\to\infty$; proof by epi-convergence. Only $C_{SS}$ enters the limit, so the fixed-$p$ GCLM version needs $\Gamma^*_{SS}\succ0$ only — a one-paragraph remark for the thesis.
5. **Fan, Xue & Zou (2014)**, Ann. Statist. 42(3), Thm 1–2 and §5.1: LLA weights make the weighted condition void ($\hat w_S=0$); this is how MCP/SCAD re-enter later.
6. **Bühlmann & van de Geer (2011)**, Ch. 7 (§7.5 irrepresentable condition, §7.8–7.9 adaptive and thresholded lasso) for a textbook exposition.
7. For §4: **Fuchs (2004)**, IEEE Trans. IT 50(6):1341–1344; **Tropp (2006)**, IEEE Trans. IT 52(3):1030–1051 (§3, exact recovery coefficient); **Tibshirani (2013)**, EJS 7:1456–1490 (lasso uniqueness and the $\lambda\to0^+$ limit); Gribonval & Nielsen (2003), IEEE Trans. IT 49(12) and Cohen, Dahmen & DeVore (2009), JAMS 22(1) for the null-space property.

---

## 3. Empirical programme: which pilots satisfy the condition, and how often

All functions are in `gclmth.py` (numpy/scipy only; `python gclmth.py` runs the self-test).

**3.1 Per-dataset diagnostics for the campaign (laptop, analysis only; then cluster columns).** For every
dataset of wave 1 and every E1 start (`lasso` dense end, `l1`, `l2`, `exact`, `truth-τ`), from the stored $M^*$
and the stored start, add to `campaign_per_dataset.csv`:

| column | meaning | function |
|---|---|---|
| `pilot_missed` | number of true edges with pilot entry 0 (or $<10^{-8}$) | `adaptive_conditions(S, C, M_true, pilot)["n_missed"]` |
| `rho_ad` | the weighted constant (∞ if `pilot_missed` > 0); regenerate $\Sigma^*$ from $M^*$, $C^*$ for the population version, or use $\hat\Sigma$ for the sample version | `adaptive_conditions(...)["rho_ad"]` |
| `pilot_orient_frac` | share of true edges with $|\text{pilot}_{\rm true}|>|\text{pilot}_{\rm rev}|$ | 5 lines |
| `pilot_worst_ratio` | $\max_{\rm edges}|\text{pilot}_{\rm rev}|/|\text{pilot}_{\rm true}|$ (the near-diagonal form of $\rho_{\rm ad}$) | 5 lines |

Then split every paired gain (adaptive − lasso, MCP dense→sparse − lasso) by `pilot_missed = 0 ∧ rho_ad < 1`
versus the rest. Prediction from the pilot runs (15 Oct memo, Fig. 7): exact recovery happens if and only if
`rho_ad < 1`; the F1 gain grows with `pilot_orient_frac`. This is E1 of `plan_mechanism_131026.md` with the
mechanism made explicit, and it decides whether the "start matters because it orients" reading is right.

**3.2 Population: how often does the BP condition hold, by DGP.** `bp_certificate(S, C, M)` (§4) per graph;
report the fraction with $t^*>0$ (unique), $t^*=0$ (tie, isolated edges), $t^*<0$, split by $k$, by graph class,
and by "contains a 2-cycle". Already computed (50 graphs per $k$, correlation scale, thesis DGP $p=10$):

| $k$ | identified ($\Gamma_{SS}\succ0$) | weak IC ($\rho_{\rm weak}<1$) | BP unique ($t^*>0$) | BP minimiser ($t^*\ge0$) | median 2-cycles |
|---|---|---|---|---|---|
| 1 | 98 % | 2 % | 48 % | 72 % | 0 |
| 2 | 100 % | 0 % | 14 % | 14 % | 2 |
| 3 | 100 % | 0 % | 0 % | 0 % | 4 |
| 4 | 96 % | 0 % | 0 % | 0 % | 7 |

Among graphs without 2-cycles: 45 %; with 2-cycles: 8 %. Raw scale ($C=2I$): 18 / 4 / 0 / 0 %. (The 7 October
note found 11 of 100 with random-objective probing; the certificate is exact and also separates ties.)
To do: the same table for the DGP variants of the plan (`weights="uniform"`, `w_min`; `graph="no2c"`, `"dag"`),
$p=10$ and $20$, and for the designed classes — this gives the thesis a table "for which DGPs is the dense-start
estimator provably consistent".

**3.3 Finite samples: the sample dense end vs the population BP solution.** For $n\in\{10^3,10^4,10^5\}$:
fraction of datasets with `pilot_missed = 0`, with `rho_ad < 1`, and the exact-recovery rate of the adaptive
lasso, against the population $t^*$ of the graph. Hypothesis to test: for graphs with $t^*>0$ the three
quantities converge to 1 at a rate governed by $t^*$ (the certificate margin), for $t^*\le0$ they do not. The
runner `finite_sample_adaptive.py` already does this on 14 designed instances; the campaign version is the
columns of 3.1 at the three $n$.

**3.4 A ceiling.** `truth-τ` starts (E1) are the $\delta$-consistent pilots of Cor. B.1: with $\tau\le0.2\,\beta_{\min}$
the adaptive lasso was exact from $n=250$–$500$ in every identifiable instance. Report where the lasso dense end
sits between "no pilot" and this ceiling.

---

## 4. The minimum-$\ell_1$ exact fit: literature and the exact check

The population dense end solves $\min\|M_{\rm off}\|_1$ s.t. $M\Sigma^*+\Sigma^*M^\top+C=0$: **basis pursuit**
(Chen, Donoho & Saunders 1998) with an underdetermined linear system ($p(p+1)/2$ independent equations,
$p^2$ unknowns, kernel $\{W\Sigma^{*-1}\}$) and free (unpenalised) diagonal variables. The "BP condition" of the
7 October note ($M^*$ is the unique minimiser) has an exact characterisation:

> **Dual certificate** (Fuchs 2004, Thm 4; Tropp 2006, §3). $M^*$ is the unique minimiser iff $A_{\bar S}$ has
> full column rank ($\Gamma_{\bar S\bar S}\succ0$) and there is $h\in\mathbb R^{p^2}$ with
> $A_S^\top h=\operatorname{sign}(m^*_S)$ on the off-diagonal support, $A_D^\top h=0$ on the diagonal columns, and
> $\|A_{S^c}^\top h\|_\infty<1$.

`bp_certificate` solves $\max t$ subject to these constraints with $\|A_{S^c}^\top h\|_\infty\le1-t$:
$t^*>0$ unique minimiser, $t^*=0$ minimiser but tied, $t^*<0$ not a minimiser; it agreed with the direct LP
solution in all test instances. Three facts to use in the thesis:

1. The least-norm certificate $h_0=A_{\bar S}(A_{\bar S}^\top A_{\bar S})^{-1}\operatorname{sign}(\cdot)$ gives
   $\|A_{S^c}^\top h_0\|_\infty=\rho_{\rm weak}$ (DDK's weak irrepresentability constant). Hence
   $$\text{weak IC (DDK G.1)}\ \Longrightarrow\ \text{BP condition}\ \Longrightarrow\ \rho_{\rm ad}(\text{dense-end pilot})=0\ \Longrightarrow\ \text{adaptive lasso sign-consistent},$$
   each arrow strict: weak IC holds in 0–2 % of thesis graphs, the BP condition in 15 % (45 % without 2-cycles).
   This is the chain that explains "the lasso fails everywhere but the dense-start estimators recover some graphs".
2. The uniform-over-signs version of the certificate is Tropp's exact recovery coefficient = DDK's condition (3.1);
   the all-sign-patterns version is the null-space property $\theta(S)>1$ of the 7 October note. Both are far too
   strong here (they fail in every graph); the sign-specific certificate is the right object.
3. Ties ($t^*=0$) are the isolated-edge graphs (both orientations have the same $\ell_1$ norm on the correlation
   scale because every exact solution has the same trace); the thesis should state the BP condition "for graphs
   without isolated edges, or up to their orientation".

**A possible small theorem.** Expand the certificate near the diagonal as DDK did for $\rho$: at $M^0$ the design is
explicit ($A(\Sigma^0)$ entries $1/d_l$, $2/d_l$), the equality constraints decouple into pair blocks, and the
question "does a certificate with margin exist" becomes a condition on the diagonal ratios $d_l/d_k$ and the signs,
edge by edge, with second-order corrections from the three-node motifs (lock-in note, Cor. 3.3). If it works out
it gives "the BP condition holds near the diagonal for graph class X iff …", the analogue of DDK Thm 3 for the
dense-start estimators. Open.

---

## 5. Theorem to write up (after §2 and §3.1)

**Theorem (adaptive lasso for GCLMs).** Hypotheses of DDK Thm D.1 without the irrepresentability condition,
plus $\rho_{\rm ad}\le1-\alpha$, (B1), (B2). Conclusion: unique solution, $\hat S=S$, signs of $M^*$,
$\|\hat M-M^*\|_\infty\le2c_{\Gamma^*}(\varepsilon+\lambda\max_S\hat w_k)$. Probabilistic version: DDK Lemma E.6 + Thm E.4
unchanged. **Corollary 1**: any pilot with $\|\hat M^{\rm pilot}-M^*\|_\infty<\beta_{\min}/(1+\rho)$. **Corollary 2**:
the lasso-dense-end pilot, in the population, iff the BP certificate exists; in samples, iff the dense end misses no
edge and orients every edge. **Proposition (necessity)**: $\rho_{\rm ad}>1$ ⟹ inconsistent. Then the table of §3.2
("for which DGPs") and the campaign split of §3.1 are the empirical half.

What *not* to attempt now: a restricted-eigenvalue route (ZvdGB Thm 4.3, van de Geer–Bühlmann–Zhou 2011) — RE
fails for the Lyapunov design (7 Oct note §4); an "all stationary points" statement — irrelevant for a convex
problem; rates in $(n,p)$ better than DDK's — a concentration problem, separate.

---

## 6. Order of work

1. Read ZvdGB §2, §8, §12 and Wainwright §II–III (one day); check (B1)–(B2) against (2.12)–(2.14).
2. Add the four columns of §3.1 to `campaign.py` for the stored wave-1 shards (laptop); produce the split tables.
3. Run `bp_certificate` over the DGP variants (§3.2; minutes on the laptop at $p=10$, cluster for $p=20$–50 with
   many graphs); produce the "for which DGPs" table.
4. Decide from 2–3 whether the thesis's main theorem is stated for "any orientation-consistent pilot" (Cor. 1,
   clean, always true) with the lasso dense end as the worked example (Cor. 2), or the other way round.
5. Write §5; then return to MCP/SCAD via FXZ Thm 1 (one paragraph: same theorem, saturated weights).
6. Optional: the near-diagonal certificate (end of §4).

## 7. Files

`gclmth.py` (library: `adaptive_conditions`, `min_l1_exact_fit`, `bp_certificate`, `adaptive_lasso_path`, `conditions`, `oracle_window`),
`sweep_adaptive.py`, `finite_sample_adaptive.py`, `bp_certificate_thesis.csv` (the table of §3.2), and the two
theory notes `claude/theory_nonconvex_adaptive_151026.md`, `claude/theory_conditions_and_proofs_201026.md`.
