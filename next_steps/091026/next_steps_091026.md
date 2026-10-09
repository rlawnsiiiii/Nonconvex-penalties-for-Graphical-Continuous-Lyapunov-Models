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
   $n = 10^3$ and $10^4$. About 4,000 CPU-h in all by estimate; $p = 30$ is left out for cost for
   now (§2, §3).
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

**Code (all with tests; the full suite, 392 tests, passes)**

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
graphs: on the first test graph, the search from the true graph added 24 to 27 false edges at
finite $n$. A check on 32 graphs runs on the laptop (`score_check.py`, output `score_check.txt`);
I read it next time and add the numbers to `docs/SEARCH.md` §2a.

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
32 graphs per cell, one graph per task. Only the 20 add moves with the largest gradient are scored
per step.
- *Why.* Wave 5b ran this search with the least-squares fit. With 100 starts it nearly reached the
  lasso followed by the search at $n \ge 10^4$, and it was still improving. Part (c) runs it with
  the likelihood fit, to see whether that conclusion holds for the method as published. Only
  $p = 10$ and 32 graphs per cell, because 100 starts with the likelihood refit cost about 5 laptop
  hours per graph at $p = 10$, and far more at $p = 20$.

### Cost, estimated

| part | $p$ | cells per $n$ | tasks per $n$ | CPU-h per $n$ |
|---|---|---|---|---|
| (a) | 10 | 4 | 48 | about 45 |
| (a) | 20 | 10 | 196 | about 650 |
| (b) | 10 | 18, of which 12 at $n = 10^4$ | 152, of which 104 at $n = 10^4$ | about 560, 370 at $n = 10^4$ |
| (b) | 20 | 10 | 196 | about 210 |
| (c) | 10 | 2 | 64 | about 640 |

About 4,000 CPU-h for both sample sizes: about two days of the 96 tasks that LRZ runs at once,
more with queueing.

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
  minutes per data set; one random start of the likelihood search at $p = 10$, about 3 minutes,
  from the pilot of 9 October.

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
2. **Part (c) at 2 replicates**, 32 graphs per cell. Enough to see whether the likelihood fit
   changes the search from random graphs; every further replicate adds about 640 CPU-h.
3. **Part (c) scores only the 20 add moves with the largest gradient per step**, for cost. Its
   comparison with wave 5b, which scores all of them, therefore changes two things at once.
4. **The search with the likelihood refit stays at $p = 10$.** At $p = 20$, part (b)
   gives the selection with the likelihood refit, and part (a) the search with the least-squares
   refit.
5. **Code names.** `bic()`, `--select bic`, `bic_f1` and the like keep the old word, because the
   stored shards and the queued jobs use them. Renaming them means a translation layer for the
   stored results; I would leave them.
