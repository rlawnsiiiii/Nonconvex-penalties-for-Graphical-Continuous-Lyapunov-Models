# Meeting with the supervisor, 15 October 2026 — preparation

*What to discuss (the direction of the thesis, with wave 9 as the candidate experiments), what to do
before the meeting, the commands for the cluster, and the short glossary. Sources: `simulations/S4_campaign.md`
§0, `simulations/VERDICTS.md`, `next_steps/131026/plan_mechanism_131026.md` (the plan, with the motivation
of each experiment in §2.0–§5.0), `next_steps/131026/next_steps_131026.md` (what is built, the first numbers,
the costs), `claude_docs/dense_start_theory_081026.md` (the theory). The last thing he has seen is the
briefing of 9 October (`slides/091026/briefing.pdf`: model and estimators, V1–V6, next steps).*

---

## 1. The main item: the direction of the thesis

### 1.1 What the evidence says now (one paragraph to open with)

The thesis asked whether MCP and SCAD recover the graph of a GCLM better than the lasso. Used the
standard way (the path from the empty graph), no: they lose on every loss and at every sample size,
and they lose on the *direction* of edges, because the sparse → dense path commits to a direction
before the data can tell (the lock-in). Started from the dense end of the lasso path and used to
prune it, yes: by up to +0.15 in oracle $F_1$, more as $p$ and $n$ grow, for every diagonal true $C$,
under Dettling's pipeline and under the correctly specified $C$. But the convex adaptive lasso, which
prunes the same start, does as well or better, so the gain comes from the start, not from the nonconvex
penalty. Without an oracle $\lambda$ the score keeps the ranking and the greedy search then brings
every lasso-based estimator within 0.02 of each other, with the search from the truth 0.10 to 0.24
above all of them (S4 §0, items 1–3, 6; VERDICTS "Bottom line"). On the log-likelihood loss the same
holds once the start is the lasso's dense end (item 7): the start matters, not the loss.

Since the 9 October briefing: waves 5–8 (likelihood refit ≈ least-squares refit; the eBIC penalty
helps from $p \approx 20$–30; 100 to 300 sparse starts of the pure search only reach the lasso + search
at 10 to 100 times the cost), the code review (every finding fixed, 518 tests), the solver check
(solvers correct), and the mechanism plan of 13 October with its first answers from the stored data
(§2.3 below) and its cluster cells ready (wave 9).

### 1.2 Three ways to frame the thesis (bring them, let him pick or combine)

| framing | main claim | what it still needs | risk |
|---|---|---|---|
| **A. "Start dense, then prune"** | the dense end of the lasso path removes the null space of the Lyapunov design; pruning it (adaptive lasso, MCP / SCAD dense → sparse) recovers the graph better than the lasso; MCP / SCAD are one way to prune, the adaptive lasso the best tried | the mechanism (wave 9, §2), the theory chapter (§1.3), the selection story (score, eBIC, search) as a second part | the broadest; the mechanism experiments must come home in time |
| **B. The original question, answered** | nonconvex penalties do not beat the lasso the standard way; from the dense end they do, and so does the convex adaptive lasso | nothing on the cluster; writing and the theory | the thesis reads as "the answer is no, and the interesting part is something else" |
| **C. Methods comparison for GCLM support recovery** | lasso, MCP / SCAD (both orders), adaptive lasso, greedy search, two losses, two choices of $C$, selection by the score: what works at which $p$, $n$ | already run (waves 1–8); figures exist | breadth without a mechanism; the search part is cost-heavy and less clean |

My recommendation to put forward: **A**, with B's negative result as its first chapter and C's
selection material as a shorter later chapter. Wave 9 is then the mechanism chapter's evidence, and the
theory note's results are the theory chapter.

### 1.3 How much theory (the 8 October note, `claude_docs/dense_start_theory_081026.md`)

The note's central idea: the direct loss is a regression whose design has a null space of dimension
$p(p-1)/2$ (every $W\hat\Sigma^{-1}$ with $W$ skew); the dense end of the lasso path is, up to
$O(\lambda_{\min})$, the minimum-$\ell_1$ exact fit of the Lyapunov equation, and its support indexes
linearly independent columns (Lemma 1), so the adaptive lasso from it solves an ordinary full-rank
weighted lasso. From this:

- **Theorem 5** (fixed $p$, $n \to \infty$): under the **BP condition** (the true $M^*$ is the unique
  minimum-$\ell_1$ exact fit of the population equation; glossary §6) the adaptive lasso from the dense
  end has Zou's oracle property; MCP / SCAD by two LLA steps too (Corollary 6).
- **Proposition 7**: strict irrepresentability ⇒ BP condition ⇒ local identifiability; on the thesis
  graphs at $p = 10$: 0, 11 (15 with a non-unique minimiser) and 98 of 100. So the lasso's own guarantee
  never applies, the dense-start guarantee applies to about one graph in nine, and a good local minimum
  exists almost always.
- **Theorem 8**: without the BP condition the path still converges, to a computable population target
  whose quality is the dense end's recall and orientation (both improve with $p$: the "gain grows with
  $p$").
- **Proposition 9**: why the dense end orients edges (reversing an edge inside the exact fits leaks into
  the other entries of the two rows and costs $\ell_1$).

Status words in the note: proved / sketch / numerical / cite. Question for him: which of these belong
in the thesis, at what level of rigour, and whether Theorem 5 with the BP condition (a guarantee that
holds in 11 % of the thesis's own graphs) is worth a theorem or a remark.

The lock-in itself (the negative result) has its own note, `../../claude_docs/orientation_lock_in_051026.md`:
Lemma 1, the exactly flat directions of the direct loss (the null space of the design, $W\hat\Sigma^{-1}$
with $W$ skew); Lemma 2 and Corollary 2.1, exact: restricted to one pair the loss sees only
$\hat\Sigma_{jj}M_{ij} + \hat\Sigma_{ii}M_{ji}$ at first order ($M_{ij} + M_{ji}$ on the correlation
scale), the two orientations fit almost equally well; Proposition 4, leading order: which orientation
enters the sparse → dense path first is decided by the variances, not the true direction; Proposition 5,
the barrier: along the line that trades one orientation for the other the lasso penalty is constant (the
lasso is indifferent, it hedges) while the sum of two concave penalties is concave with a hump, so the
first orientation is a strict local minimum at every later $\lambda$ and a warm-started descent stays. §6
lists what a thesis lemma would still need. The empirical side: `simulations/S3a_bidirectional_edges.md`
§4.4 (which direction enters first is a coin flip: 51–54 % true, 39–46 % reversed, for every penalty) and
`simulations/VERDICTS.md` §A.

### 1.4 Questions to put to him

1. Is framing A the thesis? Which of the results in §1.1 is the headline, and which are supporting?
2. Theory: §1.3. Is a theorem under the BP condition enough, or does he want a statement without it
   (Theorem 8's target)?
3. Wave 9 (§2): which of E1–E4 does he consider decisive for the mechanism chapter? Is the budget
   acceptable (about 1,000 CPU-h for the cheap part, 5,900 more at $n = 10^3$)?
4. Three open cost decisions: $p = 30$ on the log-likelihood loss stays dropped; whether the
   log-likelihood cells are re-run with the corrected solver (code review A1; the direct-loss cells are
   unaffected); the $n = 10^3$ groups of wave 9.
5. Timeline: when writing starts, and the outline (chapters of framing A).

---

## 2. Wave 9: the candidate experiments, and what you must be able to say about each

The campaign's explanation of the gain is a *mechanism*: the lasso hedges (keeps both directions of an
uncertain pair), the pruning resolves the hedge in favour of the true direction, and the sparse → dense
path cannot because it commits early. The numbers behind `orientation_p20.png` (p = 20, n = 10⁴, rescaled
C, selected graphs, means over 800 graphs): the lasso and the sparse → dense MCP find the same number of
true single pairs (34.5 and 34.8 of about 40); the lasso hedges 14.4 of them and reverses 3.5, MCP hedges
0.5 and reverses 10.4, so the lasso has 31.0 true directions present against 24.4 (recall 0.78 against
0.59). The figure's "both directions kept" bar counts both entries of a hedged pair, which is why the
lasso's bar is the longest. MCP dense → sparse: 29.1 true directions, 5.2 reversed, 2.1 hedged; adaptive
lasso: 29.4, 4.2, 5.2, with the fewest false edges (20.6 against the lasso's 33.4). Each experiment of wave 9 varies one thing the explanation
depends on and holds everything else fixed; every cell uses wave 1's graphs (same seeds), so every
comparison is paired graph by graph. Read §2.0, §3.0, §4.0, §5.0 of the plan for the full motivations;
the table below is what you should be able to say without notes. **Todo (§3, item 3): work through
these until you can answer the four questions of §3.3 for each experiment.**

### 2.1 E1 — which dense start? (group `starts`, 60 cells)

- **Question.** The only dense start tried so far is the lasso's dense end. It has three properties at
  once: it is an exact fit of the sample Lyapunov equation; it is the $\ell_1$-smallest point of the
  fibre of exact fits (a vertex, sparse, hedging); it is oriented (contains the true direction of most
  true edges). Which of the three carries the gain?
- **Varied.** The start of MCP dense → sparse (`--up-start`) and the pilot of the adaptive lasso
  (`--pilot`): `exact` (Varando & Hansen's $-\tfrac12 C\hat\Sigma^{-1}$: an exact fit only), `l1` (the
  minimum-$\ell_1$ exact fit from an LP: exactly what the lasso approximates), `l2` (the
  minimum-Frobenius exact fit: exact and dense, no vertex), `truth-0.05` and `truth-0.2` (the standardised
  truth plus Gaussian noise: oriented only, not an exact fit; a ceiling, not an estimator).
- **Fixed.** The data sets, the two pruners, the grid, the selection by the score, the search.
- **Readings.** (a) `l1` ≈ `lasso` ≫ `l2` ≈ `exact`: the $\ell_1$ geometry (hedge and vertex) is the
  mechanism. (b) `l2` ≈ `l1`: any exact fit does, "start on the fibre and prune". (c) `truth-τ` far above
  every data start: the pruners are not the bottleneck, the start's orientation is.
- **Cannot settle.** Dependence on the graph class or the weights (E3, E4); the log-likelihood loss is
  an optional second line (`starts-loglik`).
- **Already known.** On a test instance the `l1` start reproduces the lasso-started MCP path; the
  rehearsal's four graphs: the `l1` start contains the true direction alone for 37 % of the true single
  edges, both directions for 40 %, only the reverse for 16 %, neither for 8 %.

### 2.2 E2 — thresholding the dense end (group `threshold`, 12 cells)

- **Question.** The adaptive lasso has two ingredients, the pilot and the reweighted $\ell_1$ problem
  with weights $1/|\hat M^0_{ij}|$. Is the gain in the pilot alone? Then the simplest pruning of the same
  pilot, keep the $k$ largest entries and refit, must do as well.
- **Varied.** The pruning step only: the thresholded lasso (rank the pilot's entries by size, keep those
  above $t$, refit by least squares, let the score choose $t$) against the adaptive lasso, with the
  *same* pilot (paired at the pilot). `threshold-norefit` keeps the pilot's values and is for AUPR only:
  how well the dense end *orders* the edges.
- **Readings.** adaptive − threshold within ±0.01 in oracle $F_1$ and selected-graph $F_1$: the pruning
  is the whole story, the thesis can present the thresholded lasso as the minimal method. A clear positive
  difference: the reweighting matters, and E3's and E4's columns locate where (pairs where the pilot
  keeps both directions? weak edges? 2-cycles?).
- **Cannot settle.** Which pilot (E1). At $n = \infty$ with the rescaled $C$ both are near their ceiling.

### 2.3 E3 — beta-min: does the gain live on the strong or the weak edges? (group `betamin`, 64 cells)

- **Question.** Every selection guarantee has a minimum-signal condition. The campaign's weights are
  $N(0, 1)$: about 8 % of true edges have $|w| < 0.1$ and 20 % have $|w| < 0.25$, edges no method finds at
  these $n$. Is the dense-start gain on the strong edges (a *direction* mechanism: orient an edge whose
  presence is clear) or on the weak ones (a *detection* mechanism, the usual "less bias" story)?
- **Already answered from the stored data (memo §3).** The lasso has the highest recall in every weight
  bin; the pruners' gain is a *precision* gain: at $p = 20$, $n = 10^4$ the lasso's best graph carries
  32 to 34 false positives per graph against 12 to 18, almost all tiny hedged reverse directions. The
  sparse → dense MCP's recall is flat beyond $|w| = 0.5$ (0.59 / 0.63 / 0.61 in the three strongest bins,
  the same at $n = \infty$) while every other estimator passes 0.9: the lock-in seen directly. So the
  plan's prediction, phrased in recall, was wrong as stated; the mechanism shows in precision.
- **Varied in the cells.** The weight law as an exogenous knob: $|w| \sim U[w_{\min}, 1]$ with a random
  sign, $w_{\min} \in \{0, 0.1, 0.25, 0.5\}$; same supports, same diagonal rule. Four estimators: lasso,
  MCP (sparse → dense), MCP dense → sparse, adaptive lasso. Prediction now: the pruners' gain (precision)
  survives at every $w_{\min}$, the sparse → dense deficit does not shrink with $w_{\min}$.
- **Cost and status.** The most expensive group relative to what it adds; the group to cut first (three
  levels: `--only _w0$ _w025 _w05`).

### 2.4 E4 — the graph class: where direction is identified (group `class`, 32 cells)

- **Question.** The Lyapunov equation cannot tell the two directions of an isolated 2-cycle apart (two
  unknowns, one equation); in a DAG every direction is identified on the true support, in a cyclic
  graph without 2-cycles as well. The direction mechanism can only work where direction is identified,
  so the gain should be largest on DAGs and cyclic graphs without 2-cycles and smallest with 2-cycles,
  while the sparse → dense deficit (lock-in) should be present in every class.
- **Already answered from the stored data (memo §2).** Within $k$ (the class is almost a function of the
  density, so pooled comparisons would repeat the density effect): at $k = 1$, $p = 20$, $n = 10^4$,
  $C = 2I$ the gain of MCP dense → sparse is +0.12 on DAGs, +0.12 on cyclic graphs without 2-cycles,
  +0.05 with 2-cycles (rescaled $C$: +0.20 / +0.17 / +0.11); at $k = 2$ the gap survives at $p = 10$ and
  vanishes at $p = 20$; the sparse → dense deficit is −0.11 to −0.14 in every class; identification on the
  true support holds in 95 to 100 % of the graphs in every class. Groups are small (11 to 15 cyclic
  graphs per cell at $k = 1$) and the class is *realised*, not assigned.
- **Varied in the cells.** The class as an exogenous knob: `no2c` (one direction of every 2-cycle
  dropped at random) and `dag` (edges oriented along a random permutation at the same expected number
  of edges); same weights, same $C$. The same four estimators; the comparison is again within $k$.
- **Status.** Run as planned (memo §4).

### 2.5 Cost and the proposal (memo §1.3, `cluster/plan_131026.txt`)

| group | $n = 10^4$ | $n = \infty$ | $n = 10^3$ |
|---|---|---|---|
| threshold | 9 | 13 | 52 |
| starts | 260 | 270 | 2,600 |
| class | 230 | — | 1,900 |
| betamin | 180 | — | 1,400 |

CPU-hours, estimated from the laptop rehearsal of 14 October scaled with wave 1's measured costs; wave 1's
$n = 10^3$ cells cost about ten times its $n = 10^4$ cells (slow convergence on the noisier sample
covariance). The plan file runs the cheap part first (about 1,000 CPU-h, one to two days of wall time
with the queue's 200-task cap), then the $n = 10^3$ groups (5,900 CPU-h, about a week) after a fill, with
larger shard counts and 24 h limits. Proposal to him: start the cheap part now; decide the $n = 10^3$
groups from its results; betamin is the group to cut first; `starts-loglik` only if E1 is clear.

---

### 2.6 Dettling's whole-path comparison at C = 2I (15 October; no selection rule)

Dettling's Figure 5 summarises each estimator's whole path per graph by four numbers (the largest accuracy
and the largest F₁ over the path, the areas under the ROC and the precision–recall curves) and never
selects a graph. Under his pipeline (C = 2I in the estimation, n = 10³, the four true C pooled, 400 graphs
per p) the paired gains over the lasso on the same graphs are (± standard error; numbers in
`runs/campaign/campaign_figure5.csv` from `simulations/diagnostics/figure5_table.py`, figures
`figure5_estimators_C2I.png` and `figure5_gain_C2I.png` in `runs/campaign/figures/figure5/`):

Largest F₁ over the path (the lasso: 0.589 at p = 10, 0.471 at p = 50)

| p | MCP dense → sparse | SCAD dense → sparse | adaptive lasso |
|---|---|---|---|
| 10 | −0.009 ± 0.005 | −0.005 ± 0.004 | −0.007 ± 0.005 |
| 15 | +0.002 ± 0.004 | +0.003 ± 0.003 | +0.017 ± 0.005 |
| 20 | +0.018 ± 0.004 | +0.011 ± 0.003 | +0.027 ± 0.005 |
| 25 | +0.022 ± 0.004 | +0.012 ± 0.002 | +0.045 ± 0.004 |
| 30 | +0.028 ± 0.004 | +0.010 ± 0.002 | +0.053 ± 0.004 |
| 40 | +0.043 ± 0.003 | +0.019 ± 0.002 | +0.070 ± 0.004 |
| 50 | +0.044 ± 0.002 | +0.017 ± 0.002 | +0.078 ± 0.003 |

Area under the precision–recall curve (the lasso: 0.475 at p = 10, 0.332 at p = 50)

| p | MCP dense → sparse | SCAD dense → sparse | adaptive lasso |
|---|---|---|---|
| 10 | +0.007 ± 0.004 | −0.013 ± 0.003 | +0.074 ± 0.007 |
| 15 | +0.004 ± 0.003 | −0.018 ± 0.002 | +0.091 ± 0.007 |
| 20 | +0.014 ± 0.003 | −0.015 ± 0.002 | +0.111 ± 0.006 |
| 25 | +0.008 ± 0.003 | −0.017 ± 0.002 | +0.121 ± 0.006 |
| 30 | +0.010 ± 0.003 | −0.016 ± 0.002 | +0.124 ± 0.005 |
| 40 | +0.017 ± 0.002 | −0.012 ± 0.001 | +0.141 ± 0.005 |
| 50 | +0.015 ± 0.002 | −0.011 ± 0.001 | +0.141 ± 0.004 |

Area under the ROC curve (the lasso: 0.704 at p = 10, 0.831 at p = 50)

| p | MCP dense → sparse | SCAD dense → sparse | adaptive lasso |
|---|---|---|---|
| 10 | −0.016 ± 0.002 | −0.021 ± 0.002 | −0.001 ± 0.002 |
| 15 | −0.023 ± 0.002 | −0.027 ± 0.001 | −0.006 ± 0.002 |
| 20 | −0.028 ± 0.001 | −0.030 ± 0.001 | −0.012 ± 0.002 |
| 25 | −0.031 ± 0.001 | −0.032 ± 0.001 | −0.014 ± 0.001 |
| 30 | −0.032 ± 0.001 | −0.031 ± 0.001 | −0.015 ± 0.001 |
| 40 | −0.035 ± 0.001 | −0.033 ± 0.001 | −0.016 ± 0.001 |
| 50 | −0.038 ± 0.001 | −0.034 ± 0.001 | −0.020 ± 0.001 |

Largest accuracy over the path (the lasso: 0.804 at p = 10, 0.952 at p = 50)

| p | MCP dense → sparse | SCAD dense → sparse | adaptive lasso |
|---|---|---|---|
| 10 | +0.010 ± 0.002 | +0.006 ± 0.001 | +0.009 ± 0.002 |
| 15 | +0.013 ± 0.001 | +0.006 ± 0.001 | +0.017 ± 0.001 |
| 20 | +0.011 ± 0.001 | +0.003 ± 0.000 | +0.016 ± 0.001 |
| 25 | +0.009 ± 0.000 | +0.002 ± 0.000 | +0.015 ± 0.001 |
| 30 | +0.008 ± 0.000 | +0.001 ± 0.000 | +0.013 ± 0.000 |
| 40 | +0.006 ± 0.000 | +0.001 ± 0.000 | +0.011 ± 0.000 |
| 50 | +0.004 ± 0.000 | +0.000 ± 0.000 | +0.009 ± 0.000 |

How to read it.

- The adaptive lasso beats the lasso on three of his four summaries from p = 15 on: F₁ by +0.02 (p = 20)
  to +0.08 (p = 50), the precision–recall area by +0.07 to +0.14 (the largest effect in the tables), the
  accuracy by +0.01 to +0.02. At p = 10 it ties the lasso in F₁.
- MCP dense → sparse shows the same pattern with half the F₁ effect (+0.02 to +0.04 from p = 20 on) and a
  small precision–recall gain (+0.01); SCAD dense → sparse gains +0.01 to +0.02 in F₁ and sits 0.01 to 0.02
  below the lasso in the precision–recall area.
- The ROC area is the one summary the lasso wins against all three, by 0.01 to 0.04. It weighs the ordering
  of all entries including the dense part of the path, where the lasso's hedging keeps both directions of
  the true pairs present; the precision–recall area weighs the sparse part, which the dense-start paths trace
  better. That the dense-start paths order the late entries worse is a reading of the figure, not something
  tested. At n = 10⁴ (p = 10, 20) the ROC deficit vanishes for MCP dense → sparse (−0.001, −0.005) and turns
  into a gain for the adaptive lasso (+0.012, +0.009), while the other gains grow (adaptive lasso at p = 20:
  F₁ +0.070, precision–recall area +0.187).
- MCP and SCAD sparse → dense lose on F₁, ROC area and precision–recall area everywhere (F₁ −0.03 to −0.10,
  ROC −0.10 to −0.13, precision–recall −0.04 to −0.07); in accuracy they tie the lasso (within 0.002) because
  the accuracy is dominated by the true absences.
- Per true C (the four columns of `figure5_gain_C2I.png`): the three diagonal settings show the pooled
  pattern; the non-diagonal true C (the misspecified one) is the hard case, with F₁ below the lasso at
  p ≤ 15 for all three estimators and the adaptive lasso's precision–recall gain halved (+0.03 to +0.11); the
  MCP dense → sparse precision–recall gain is absent there until p = 40.
- The caveats of §5 apply: 100 graphs per (p, true C) against Dettling's 400, and the lasso's reproduction
  is below his curves for the non-diagonal true C.

---

## 3. Before the meeting

1. **Commit** the 14 October work (10 min): `git status` lists it; `cluster/plan_wave7.txt` is staged as
   added but deleted in the tree (`git add -A cluster/` or `git rm --cached cluster/plan_wave7.txt` first).
2. **Cluster state and results home** (20 min): §4, then `campaign.py` and the figures, so the numbers
   you quote are today's. If wave 8's $n = 10^3$ tasks are still running, do **not** `git pull` on LRZ
   (the solver change of 13 October is in the tree).
3. **Understand the motivation and setting of each wave 9 experiment** (60 min, the item you asked for).
   Reading order: the plan's "Why these four" table (§0) → §2.0, §3.0, §4.0, §5.0 (the motivations, about
   two pages each) → memo §2–§4 (what the stored data already say and what it means for the cells) →
   theory note §0 and §4 (the BP condition and Theorem 5). Then, for each of E1–E4, answer the four
   questions of §3.3 in your own words, on paper, without looking.
4. **Decide what to show** (20 min): talk from this document and the memo with four figures open
   (§5), or ask for a 3–5 slide addendum to the briefing (I can draft it in the house style).
5. **Prepare the questions of §1.4** with your own preference stated first for each.

### 3.3 The four questions you should be able to answer for each experiment

- What is varied, and what is held fixed (so that the comparison is paired)?
- What does the thesis's explanation predict, and what result would contradict it?
- What has the stored campaign already said about it (E3, E4), and why are the cells still worth running
  (the knob is exogenous; the realised classes are small and confounded with density)?
- What can the experiment *not* settle, and which other experiment covers that?

---

## 4. Commands: the cluster state, and the results home

On LRZ (`ssh ge47xod3@cool.hpc.lrz.de`, repository in `~/repo`; every SLURM command needs `-M serial`):

```bash
cd ~/repo
squeue -M serial -u $USER -h -r | wc -l                       # queued + running tasks (cap 200)
squeue -M serial -u $USER -o "%.10i %.12j %.9T %.11M %.5C" | sort -k3 | head -40
pgrep -fl feed_queue.sh; tail -5 logs/feed_queue.log           # is the feeder alive, what did it do last
bash cluster/submit_campaign.sh --wave 7 --status | grep -v complete   # empty = wave 7 done
bash cluster/submit_campaign.sh --wave 8 --status | grep -v complete   # the likelihood refit and the 100-start search
sacct -M serial -u $USER -S $(date -d '-2 days' +%F) -X --format=JobID,JobName%14,State,Elapsed,Timelimit | grep -v COMPLETED | tail -20
git log --oneline -1; git status --short | head                # which commit the cluster runs
```

On the laptop (incremental; run as often as you like):

```bash
cd ~/Desktop/MastersThesis/repo
rsync -av ge47xod3@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/          # the shards of every cell
mkdir -p logs/lrz && rsync -av ge47xod3@cool.hpc.lrz.de:repo/logs/feed_queue.log logs/lrz/
python3 simulations/diagnostics/campaign.py            # tables: campaign_{per_dataset,means,means_by_true_c,paired}.csv
python3 simulations/diagnostics/plot_campaign.py       # the figures, redrawn from the complete cells
bash cluster/submit_campaign.sh --wave 8 --status | grep -v complete   # the same picture from the local copy
```

`campaign.py` prints which cells are incomplete and how many dead selections there are; the figures use
complete cells only. `--edges` (the per-edge table, 17 minutes) is not needed today.

---

## 5. Figures to have open

- `runs/campaign/figures/by_p.png` and `gain_by_p.png`: the thesis figure (oracle $F_1$, AUPR, selected
  graph over $p$ at $n = 10^3$) and the paired gains.
- `runs/campaign/figures/gain_by_class_p20.png`: E4 from the stored data, within $k$.
- `runs/campaign/figures/recall_by_weight.png`: E3 from the stored data (the lasso's recall is the
  highest everywhere; the flat sparse → dense curve).
- `runs/campaign/figures/selection_checks.png` or the `bic_vs_ebic` figure if the selection story comes up.
- `runs/campaign/figures/figure5/figure5_grid_C2I.png` and `figure5_grid_Cresc.png` (15 October): Dettling's
  Figure 5 layout for every estimator, his four whole-path summaries (largest accuracy and F₁ over the path,
  areas under the ROC and precision–recall curves) over p at n = 10³, one column per true C; and
  `figure5_lasso_C2I.png`, the lasso under his pipeline with his own digitised curves overlaid. These use no
  selection rule at all, so they answer "how does each estimator's whole path compare with the lasso's" in
  his own terms; the selected-graph figures are the data-driven complement.
- `runs/campaign/figures/figure5/figure5_estimators_C2I.png` and `figure5_gain_C2I.png` (15 October, the
  figures for §2.6): his 2 × 2 layout at C = 2I with one curve per estimator, the four true C pooled
  (400 graphs per point), and the paired gain of every estimator over the lasso with ± 2 standard errors,
  pooled and per true C. The numbers are in `runs/campaign/campaign_figure5.csv`
  (`python3 simulations/diagnostics/figure5_table.py` prints them).

---

## 6. Glossary (the words that will come up)

- **Dense end.** The lasso solution at the smallest $\lambda$ of the grid ($10^{-4}\lambda_{\max}$):
  dense, both directions of an uncertain pair present, an approximation of the minimum-$\ell_1$ exact
  fit of the Lyapunov equation.
- **Exact fits / the fibre.** All $M$ with $M\hat\Sigma + \hat\Sigma M^\top + C = 0$: an affine space
  $\{M_0 + W\hat\Sigma^{-1} : W \text{ skew}\}$ of dimension $p(p-1)/2$, $M_0 = -\tfrac12 C\hat\Sigma^{-1}$.
  The direct loss is zero on it; the null space of the design is what makes the problem hard.
- **BP condition, BP count.** "BP" is basis pursuit, the minimum-$\ell_1$ solution of a linear system.
  The BP condition of the theory note says: the true drift matrix (on the correlation scale, with the
  rescaled $C$) is the *unique* minimum-$\ell_1$ exact fit of the *population* Lyapunov equation. Under
  it, Theorem 5 gives the adaptive lasso from the dense end the oracle property as $n \to \infty$. The
  BP count is how often the condition holds in the thesis's own generator: **11 of 100 graphs at
  $p = 10$** (7 at $k = 1$, 4 at $k = 2$, none at $k = 3, 4$), reproduced exactly on 14 October with the new
  exact-fit module; the truth attains the minimal norm in 17 graphs, but in 6 of them the minimiser is
  not unique (an isolated edge can be reversed at the same $\ell_1$ cost), so the LP returns the truth by
  chance (14 of 100 with the current solver, 15 in the 8 October note). Meaning: the guarantee applies
  to about one graph in nine and never at $k \ge 3$; the finite-$n$ gains come mostly from graphs where
  it does not apply, which is why the mechanism experiments exist. For comparison, the lasso's own
  condition (strict irrepresentability) holds in 0 of 100 graphs.
- **Hedge.** The lasso keeping both directions of a single true edge: one true positive, one false
  positive, and the pruning's opportunity.
- **Lock-in.** The sparse → dense path enters a pair with one direction at the first $\lambda$ (a coin
  flip for MCP / SCAD) and never corrects it; the lasso corrects, MCP freezes.
- **Rescaled $C$.** $2\,\mathrm{diag}(1/s_i^2)$ on standardised data: the model "$C = 2I$ on the
  measurement scale" written on the correlation scale; exact for `C_ID`, a device for simulations.
- **Score.** A loss plus a penalty (BIC or eBIC); the graph selected by the score, and the greedy
  search on the score with add / delete / reverse moves.
- **Oracle $F_1$ (`max_f1`).** The best $F_1$ along the path, the quality of the path itself; AUPR the
  area under its precision–recall curve; `bic_f1`, `search_f1` the data-driven graphs.
- **Within $k$.** Comparing classes at fixed density, because at $p = 10$ the class is almost a function
  of $k$ ($k = 1$: 0.45 expected 2-cycles; $k = 4$: about 7).

---

## 7. After the meeting

If wave 9 is approved: `git pull` on LRZ once wave 8's $n = 10^3$ tasks are done (or accept the solver
change for their remaining shards), stop the old feeder, start `cluster/feed_queue.sh
cluster/plan_131026.txt` (plan §8 has the exact lines). The cheap groups come home in one to two days;
E2 can be analysed while E1 runs (`starts.py`, `plot_starts.py`), then `graph_class.py` and `betamin.py`
with their figures, and the results go into the memo. Writing can start on the chapters that do not
depend on wave 9: the model, the negative result, the selection story, the theory.
