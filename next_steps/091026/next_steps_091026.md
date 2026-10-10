# 9 October 2026: what is done, and what runs next on the cluster

*Working notes. The results so far are in [`../../simulations/VERDICTS.md`](../../simulations/VERDICTS.md)
and [`../../simulations/S4_campaign.md`](../../simulations/S4_campaign.md); the campaign plan is
[`../051026/cluster_campaign_051026.md`](../051026/cluster_campaign_051026.md) (§3.4 lists every
wave), the commands are in [`../051026/cluster_commands_051026.md`](../051026/cluster_commands_051026.md).*

**Terms used from today on.** A graph is selected by its **score = loss + penalty**:
$\mathrm{score}(S) = L(S) + \mathrm{pen}(S)$ with the loss term
$L(S) = n[\log\det\Sigma_S + \mathrm{tr}(\Sigma_S^{-1}\hat\Sigma)]$ at the refit $M_S$, and the
**BIC penalty** $(p + |S|)\log n$ or the **eBIC penalty** $(p + |S|)\log n + 4\gamma|S|\log p$
(Dettling et al., eq. 6.1–6.2). BIC and eBIC name only the penalty
([`../../docs/SEARCH.md`](../../docs/SEARCH.md) §2).

---

## 0. Summary

1. **Waves 1 to 5 are complete and written up**; wave 6 (the eBIC penalty everywhere, 300 starting
   graphs) was started with the feeder on 9 October and still has to come home.
2. **Waves 7 and 8 are implemented and tested** (10 October, after your feedback): (a) the
   log-likelihood loss over $p = 10$ and $20$ with the dense-start estimators; (b) the likelihood
   refit, as Améndola et al. and Dettling et al. fit, behind the score and the search on the stored
   paths; (c) the search from 100 random graphs with the likelihood refit. Every part runs at
   $n = 10^3$ and $10^4$. About 7,100 CPU-h in all by estimate, more than half of it part (c);
   $p = 30$ is left out for cost for now (§2 to §4).
3. **The whole repository now uses the new terms** (score = loss + penalty); code names, flags and
   stored fields are unchanged, so nothing on the cluster breaks.
4. **What you do now** (§2): commit and push; check that wave 6 is done and bring it home; start
   waves 7 and 8 with `cluster/plan_101026.txt`, which also carries wave 6's remaining lines.

---

## 1. Done since the morning of 9 October

**Results and documents**

- Waves 3, 4 and 5 analysed. `simulations/VERDICTS.md` (verdicts 19 to 26, every claim with its
  figure) and `simulations/S4_campaign.md` (§4a the starting graphs, §5 the selection step and the
  BIC against the eBIC penalty, §6 the log-likelihood loss, §6a the trend over $p$) are up to date.
- Figures: `runs/campaign/figures/` (BIC penalty) and `runs/campaign/figures_ebic1/` (eBIC
  penalty), with the new labels; "standard path" is now "sparse → dense" in every legend.

**Slides for the professor**

- `slides/091026/briefing.pdf` (11 pages: model and estimators, data and selection, verdicts,
  four evidence slides, next steps, references, one backup slide on the log-likelihood loss).
- Slide 2 shows the score as loss + penalty with the two penalties; slide 7 separates the two
  levels of the greedy search: the score of one graph (fit $M$ on it, then the score), and the
  search from a graph (score every neighbour, move to the best).
- `slides/SLIDE_STYLE.md` collects your rules, `slides/check_slides.py` checks the mechanical ones;
  the deck passes and every page was looked at.

**Code (all with tests; the full suite, 395 tests, passes)**

| what | where | used by |
|---|---|---|
| MCP dense → sparse for the covariance losses started at the lasso solution instead of the exact fit | `covloss_path(..., start="lasso")`, `--up-start lasso` | wave 7 (a) |
| adaptive lasso for the covariance losses | `adaptive_covloss_path` | wave 7 (a) |
| a finished cell rescored with the likelihood refit, as a cell of its own | `simulations/rescore_shard.py --refit loglik` | wave 8 (b) |
| the search from random graphs with the likelihood refit | `run_search_shard.py --refit loglik --add-screen 20` | wave 8 (c) |
| the direct-loss score $N\log(\mathrm{RSS}_S/N) + (p + \lvert S\rvert)\log n$ | `Scorer(..., score="direct")`, `bic_direct` | not on the cluster, see below |
| the cells of waves 7 and 8; the queue feeder's plan | `cluster/submit_campaign.sh --wave 7`, `--wave 8`, `cluster/plan_101026.txt` | §2 |

**The direct-loss score is not worth cluster time.** It is the score of a Gaussian regression on
the $N = p(p+1)/2$ Lyapunov equations, which are not independent and whose number does not grow
with $n$. A graph with enough edges fits all $N$ equations exactly, so the score pulls toward dense
graphs. The check on 32 graphs at $p = 10$ (`score_check.txt`, table in `docs/SEARCH.md` §2a)
confirms it: on the lasso path the score selects the densest graph, 45 edges for about 24 true
ones, in all 64 graphs, with $F_1$ 0.43 / 0.44 at $n = 10^3$ / $10^4$ against 0.52 / 0.57 for the
campaign's score. The same check repeats wave 5a's result: the likelihood fit and the
least-squares fit select graphs within 0.005 in $F_1$, at 700 to 1,000 times the cost.

**Terms.** `docs/SEARCH.md` §2 defines the score, the loss term and the two penalties and lists
the code names that keep the old words (`bic()`, `--select bic`, `bic_f1`, `ebic1_*`). Every
document, note, slide, figure label, docstring and help text was rewritten to the new terms;
the 1–3 October notes too. A memory note keeps the rule for future sessions.

---

## 2. What to do now

**Laptop: commit and push.** Everything above is uncommitted. New files: `cluster/plan_101026.txt`,
`next_steps/091026/` (this note, the two timing pilots `time_wave7*.py` with their `.txt`, `score_check.py`),
`tests/test_covloss_dense_start.py`, `slides/` (deck, figures, `make_figures.py`, `SLIDE_STYLE.md`,
`check_slides.py`). `claude_docs/dense_start_theory_081026.md` and `slides/091026/briefing.pdf`
are staged but have changed since: add them again.

**LRZ: wave 6 first.** It was started on 9 October (submission log of the command sheet).

```bash
cd ~/repo && git pull
bash cluster/submit_campaign.sh --wave 6 --status | grep -v complete    # empty = done
tail -5 logs/feed_queue.log                                             # "plan complete"?
```

When it is done, on the laptop: block 9 of the command sheet (the two `rsync` lines), then
`python simulations/diagnostics/campaign.py` and tell me; I redraw `figures_ebic1/`, the
`bic_vs_ebic` figure, the 300-start curves, and update VERDICTS and S4.

**LRZ: waves 7 and 8** (block 14 of the command sheet). The plan also carries the lines of
`plan_091026.txt`, so one feeder runs everything. Stop the wave 6 feeder first if it still runs;
its jobs keep running, and its cells are skipped as already submitted.

```bash
pgrep -fl feed_queue.sh; ssh cm4login1 pgrep -fl feed_queue.sh; ssh cm4login2 pgrep -fl feed_queue.sh
ssh cm4login1 pkill -f feed_queue.sh; ssh cm4login2 pkill -f feed_queue.sh                          # only if one runs
bash cluster/submit_campaign.sh --wave 7 --n 1e4 --dry-run | tail -1   # would submit 244 tasks in 14 cells
nohup bash cluster/feed_queue.sh cluster/plan_101026.txt > logs/feed.out 2>&1 &
sleep 5 && tail -3 logs/feed_queue.log
```

The plan runs (a) at $n = 10^4$, then (c) at both sample sizes, then (a) at $n = 1000$. It then
waits until no wave 7 job is left, because (b) rescores wave 7's paths, and runs (b) at both sample
sizes.

### The three parts, and why each is needed

They replace the (a) to (e) of 9 October:
- **Old (a) and (b) were one experiment at two values of $p$.** Old (a) ran $p = 20$ with all five
  estimators and the selection only; old (b) ran $p = 10$ with the two new estimators only,
  because wave 3 already has the other three there, and with the search, which is affordable at
  $p = 10$. Both are now part (a), over $p = 10$ and $20$.
- **Old (c) and (d) both scored stored paths again with the likelihood refit**, for the
  log-likelihood paths of wave 3 and the direct-loss paths of wave 1. Both are now part (b).
- **Old (e) is part (c)**, now with 100 random starting graphs instead of 10.

**(a) The log-likelihood loss over $p$** (wave 7). Five estimators: the lasso, MCP sparse → dense,
MCP dense → sparse from the exact fit, MCP dense → sparse from the lasso solution, and the adaptive
lasso. $p = 10$ and $20$, both $C$, $n = 10^3$ and $10^4$; $p = 30$ is left out for now (§3). The
graph is selected by the score and then searched, with the least-squares refit as in waves 1 to 4.
At $p = 10$ only the two new estimators run; the other three come from wave 3, which has the
selected graph but no search.
- *Why.* Verdicts V1 to V4 hold for the direct loss. On the log-likelihood loss, wave 3 found at
  $p = 10$ that MCP dense → sparse from the exact fit does not beat the lasso at finite $n$. Part
  (a) asks whether the two dense starts that work for the direct loss, from the lasso solution and
  the adaptive lasso, carry over to the likelihood loss, and whether their gain grows with $p$ as
  it does for the direct loss. Without it, the main result is about one loss only.

**(b) The likelihood refit after a path** (wave 8). The stored paths are scored again, with the
model on every graph fitted by maximum likelihood instead of least squares. At $p = 10$ the
selection and the search: the six log-likelihood estimators of waves 3 and 7, and the lasso, MCP
dense → sparse and adaptive lasso of the direct loss from wave 1. At $p = 20$ the selection
only, on wave 7's paths.
- *Why.* Améndola et al. and Dettling et al. fit every candidate graph by maximum likelihood. The
  campaign fits by least squares, which is 30 to 180 times cheaper. Wave 5a showed a difference of
  at most 0.01 for direct-loss paths at $p = 10$, $n = 10^4$. Part (b) checks it at $n = 10^3$, on
  the log-likelihood paths, and at $p = 20$, so that the results can be stated for the
  papers' procedure. The search stays at $p = 10$: with the likelihood refit it takes about 90
  minutes per graph at $p = 20$ on the laptop, against about 2 minutes at $p = 10$.

**(c) The search from 100 random graphs with the likelihood refit** (wave 8): Améndola et al.'s
procedure as published. $p = 10$, both $C$, $n = 10^3$ and $10^4$; 100 sparse random starting
graphs and the empty graph, plus the search from the true graph as a ceiling. Two replicates, i.e.
32 graphs per cell. Only the 20 add moves with the largest gradient are scored per step. The 100
starts of a graph are split into 4 tasks of 25 (`--start-blocks 4`, added on the night of
10 October), because one graph takes about 17 laptop hours and would exceed the 24 h task limit;
the analysis puts the four blocks back together (§4).
- *Why.* Wave 5b ran this search with the least-squares fit. With 100 starts it nearly reached the
  lasso followed by the search at $n \ge 10^4$, and it was still improving. Part (c) runs it with
  the likelihood fit, to see whether that conclusion holds for the method as published. Only
  $p = 10$ and 32 graphs per cell, because 100 starts with the likelihood refit cost about 17
  laptop hours per graph at $p = 10$, and far more at $p = 20$.

### Cost, estimated

| part | $p$ | cells per $n$ | tasks per $n$ | CPU-h per $n$ |
|---|---|---|---|---|
| (a) | 10 | 4 | 48 | about 45 |
| (a) | 20 | 10 | 196 | about 650 |
| (b) | 10 | 18, of which 12 at $n = 10^4$ | 152, of which 104 at $n = 10^4$ | about 560, 370 at $n = 10^4$ |
| (b) | 20 | 10 | 196 | about 210 |
| (c) | 10 | 2 | 256 | about 2,150 |

About 7,100 CPU-h for both sample sizes: three days or more of the 96 tasks that LRZ runs at
once. Part (c) is more than half of it; the timing behind that is in §4.

**Where the numbers come from.** Laptop seconds per data set, times 2 for LRZ, times 400 data sets
per cell (32 graphs in (c)). Shards keep the estimated time per task under about 6 hours, a factor
4 under the 24 h limit.

| laptop seconds per data set, path only, $p = 20$ | $C = 2I$ | rescaled $C$ |
|---|---|---|
| lasso | 90 | 219 |
| MCP sparse → dense | 73 | 132 |
| MCP dense → sparse, exact fit | 117 | 290 |
| MCP dense → sparse, from the lasso | 180 | 599 |
| adaptive lasso | 101 | 248 |

- **Measured.** The $p = 20$ paths and the likelihood refit along a path at $p = 20$, 80 to 110 s
  per data set (`time_wave7.txt`, 9 October). The search with the likelihood refit, 133 s per graph
  at $p = 10$ and about 90 minutes at $p = 20$ (`time_wave7_p30.txt`, 10 October, stopped early).
- **Estimated, not measured.** The least-squares selection and search at $p = 20$, about 1.5
  minutes per data set. One random start of the likelihood search at $p = 10$ was measured on
  the night of 10 October: 10.1 minutes on average (§4).

**Afterwards on the laptop:** the `rsync` lines of block 9, `python simulations/diagnostics/campaign.py`,
and tell me. I then add the figures for the new cells (none is written yet), update S4 §6, the
verdicts, and the slides (slide 1 still says the log-likelihood loss runs only at $p = 10$).
The first lines of the plan give a complete picture at $n = 10^4$ on their own, so the feeder can
be stopped there if the queue is slow.

---

## 3. Decisions for you

1. **$p = 30$ is left out for now.** Adding it later costs about 3,400 CPU-h per sample size by
   estimate (the log-likelihood lasso path at $p = 30$ took 3.8 times its $p = 20$ time), or about
   1,350 with 10 replicates instead of 25. The cells only need $p = 30$ added to the loops of waves
   7 and 8 in `cluster/submit_campaign.sh`; the shard counts are in a comment there.
2. **Part (c) costs about 2,150 CPU-h per sample size**, more than half of the plan. Its
   least-squares partner with 100 starts, wave 5b, exists for the rescaled $C$ only: restricted to
   the rescaled $C$, part (c) costs half and keeps the paired comparison. One replicate instead of
   two halves it again. Tell me and I change the cells.
3. **Part (c) scores only the 20 add moves with the largest gradient per step**, for cost. Its
   comparison with wave 5b, which scores all of them, therefore changes two things at once.
4. **The search with the likelihood refit stays at $p = 10$.** At $p = 20$, part (b)
   gives the selection with the likelihood refit, and part (a) the search with the least-squares
   refit.
5. **Code names.** `bic()`, `--select bic`, `bic_f1` and the like keep the old word, because the
   stored shards and the queued jobs use them. Renaming them means a translation layer for the
   stored results; I would leave them.

---

## 4. The night of 10 October: checks, and one change to part (c)

**Part (c) now runs the starts of a graph in 4 tasks, and it costs more than estimated.** On the
laptop (partly while the rehearsal below ran) one start of the search with the likelihood refit
takes 1 to 29 minutes at $p = 10$, 10.1 minutes on average over 12 starts on four graphs
(`next_steps/091026/time_wave8c.txt`). So the 100 starts of one graph take about 17 laptop hours
on average, about 34 on LRZ: with one graph per task most tasks would exceed the 24 h limit, and a
task that times out loses all of its 100 starts. `run_search_shard.py --start-blocks 4` therefore
splits the starts of every graph into 4 tasks of 25; `restarts.py` and `campaign.py` put the
blocks back together. A test checks that the split run gives exactly the unsplit result. The
cells `search100sml_p10_<C>` now have 128 tasks each, 25 starts per task. Part (c) is now the
largest part of the plan, about 2,150 CPU-h per sample size (§3, item 2).

**Checked, nothing to do:**
- **A rehearsal of waves 7 and 8 on the laptop**, one or two data sets per cell with the cells'
  own arguments: the log-likelihood cells of wave 7 at $p = 10$ and $20$ with the search; the
  rescoring of wave 8 with the likelihood refit, selection and search at $p = 10$ and selection at
  $p = 20$; the search from random graphs with the likelihood refit; then `campaign.py` and
  `restarts.py` on the result. Every step ran without error, and the tables and the restart curves
  read every new cell. One log-likelihood lasso path at $p = 20$, $n = 1000$ took 25 minutes,
  against 1.5 for another data set; the n-sweep ran the same paths on LRZ in about 3 hours per
  task of 100 graphs, so wave 7's shard counts are safe.
- The direct-loss score: the 32-graph check is in `docs/SEARCH.md` §2a. The score always selects
  the densest graph on the path; the likelihood fit and the least-squares fit select the same
  graphs.
- **Tests:** the full suite passes, 395 tests, among them three new ones for the start blocks
  (the split run equals the unsplit one; the curves and the table rows are the same; blocks must be
  whole).

**In the morning.**

1. Laptop: commit and push the night's changes. Yesterday's commit has everything the cluster
   needs, but the slide sources and the pilot scripts were left out (probably a path in the long
   `git add` line that did not match, which makes git add nothing from that line); they are
   named one by one here.

   ```bash
   cd ~/Desktop/MastersThesis/repo
   git add -u
   git add slides/SLIDE_STYLE.md slides/check_slides.py slides/091026/briefing.tex slides/091026/make_figures.py slides/091026/figures
   git add next_steps/091026/score_check.py next_steps/091026/score_check.txt next_steps/091026/time_wave7.py next_steps/091026/time_wave7.txt
   git add next_steps/091026/time_wave7_p30.py next_steps/091026/time_wave7_p30.txt next_steps/091026/time_wave8c.py next_steps/091026/time_wave8c.txt
   git status --short | grep -v "^??"        # what goes into the commit
   git commit -m "Start blocks for the 100-start likelihood search (wave 8 c); slides; pilots and checks"
   git push
   ```

2. LRZ: did the feeder submit part (c) overnight, in the old layout?

   ```bash
   cd ~/repo && ls -d runs/campaign/search100sml_p10_* 2>/dev/null
   ```

   - **Nothing listed:** `git pull`. The feeder submits part (c) in the new layout when it gets there.
   - **Listed:** those tasks hold all 100 starts of a graph, and most of them would run past
     24 h. Cancel them, remove the cells and pull. The feeder submits them again, in the new
     layout, with its wave 8 lines after wave 7 has ended; to start them sooner, run the two
     `--only search100sml` lines of the plan by hand once the queue has room.

     ```bash
     for j in 8si4 8sr4 8si3 8sr3; do scancel -u $USER --name=$j; done   # part (c) at n = 1e4 and 1000
     rm -rf runs/campaign/search100sml_p10_*
     git pull
     ```

   Pulling while waves 7 and 8 run is safe: the running tasks keep their code, and the queued
   ones accept the same arguments. If you want part (c) smaller (§3, item 2), tell me before it is
   submitted again.
