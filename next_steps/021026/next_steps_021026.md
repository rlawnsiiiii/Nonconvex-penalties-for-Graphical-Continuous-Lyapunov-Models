# 2 October 2026: orientation, Example 2 at large $n$, and the cluster n-sweep

*Working notes. Numbers marked [scratch] come from the scripts in [`files/`](files/) (run them from
the repository root; they import `gclm` from `src/` and read the committed runs). Regenerate them
with the repository drivers before they go into the thesis. The notes of the day before are in
[`../011026/`](../011026/).*

---

## 0. Summary

- **Why the lasso wins even though MCP orients edges about as well as a coin flip.** A coin flip
  only ties another coin flip, and the lasso does not flip: when the direction is unclear it keeps
  both directions (it *hedges*). Measured against a hedge, a right call saves one false positive and
  a wrong call loses one true positive. Committing therefore pays only if it is right more than
  $1 - F_1/2 \approx 71\,\%$ ($p=10$) / $73\,\%$ ($p=20$) of the time. MCP is right on 54 % / 59 %
  of the lasso's hedges. Settling the lasso's own hedges by a fair coin and cutting its 2-cycles to
  one direction reproduces nearly all of MCP's deficit (§1).
- **Nonconvexity helps as an objective, not as a continuation path (Example 2).** At $n=\infty$
  the smallest-$\ell_1$ exact fit is a 10-edge DAG with the cycle-closing edge reversed ($\ell_1$
  2.88 against 3.25 for the true 5-cycle). The MCP and SCAD paths converge to the lasso's wrong
  graph at every $n$. The MCP objective with a greedy reversal search recovers the truth exactly in
  90 % of datasets at $n=10^5$ and in all at $n=\infty$ (§2, §3).
- **Random graphs at $n=\infty$ (Figure 5 DGP, $p=10$, direct loss).** The lasso moves only from
  `max_f1` 0.621 to 0.681 and recovers no graph exactly. MCP's deficit *grows* (`C_ID`: −0.093 at
  $n=1000$, −0.115 at $n=\infty$). Getting stuck on a direction is not a small-sample effect (§4).
- **Cluster n-sweep.** Figure 5's DGP at $p=10,20$ for $n=10^3,\dots,\infty$, three losses × three
  penalties. Code, tests and a submission script were added to the repository; submission started
  today with a canary (§5).
- **Next.** Move the reversal search into the repository, then compare with $\lambda$ chosen by
  BIC: lasso + reversal search vs. MCP + reversal search vs. the lasso alone (§6).

---

## 1. Why the lasso wins when MCP's orientation is a coin flip [scratch: `coinflip.py`, `hedge_split.py`]

**The asymmetry.** For a true edge $i\to j$ on which the lasso keeps both directions:

| | TP | FP | FN | compared with the hedge |
|---|---|---|---|---|
| lasso keeps both | 1 | 1 | 0 | — |
| MCP picks right (probability $q$) | 1 | 0 | 0 | one FP fewer |
| MCP picks wrong (probability $1-q$) | 0 | 1 | 1 | one TP fewer |

- **The break-even.** $F_1 = 2\,TP/(|S^*| + |\hat S|)$. Both outcomes shrink $|\hat S|$ by one;
  only the wrong one also costs a TP. So settling one hedge changes the expected $F_1$ from
  $2TP/D$ to $2(TP - (1-q))/(D-1)$. This is exact, since $D-1$ is the same in both outcomes, and it
  is an improvement iff $q > 1 - F_1/2$.
- **Equivalently:** an entry is worth keeping when its chance of being a true edge exceeds $F_1/2$.
  This is the general $F_1$-thresholding result (Lipton, Elkan & Narayanaswamy 2014; check before
  citing).
- **AUC weighs it more heavily.** A missed true edge costs $1/|P|$ of TPR, a false edge only
  $1/|N|$ of FPR. $|N|/|P| \approx 3$ at $p=10$ and $\approx 7$ at $p=20$. That fits MCP's AUC gap
  growing from $p=10$ to $20$ while its $F_1$ gap shrinks (S2 §3.1).
- **2-cycles involve no coin at all.** Both directions are true, so committing to one costs a TP
  and saves nothing.

**On your shards.** Best-F1 estimates, pair-matched on identical datasets, 400 per $p$: the lasso
from `runs/s1_dettling_reproduction`, MCP and SCAD from `runs/s1b_pilot_p10-20`.

| per dataset | $p=10$ | $p=20$ |
|---|---|---|
| true single edges the lasso hedges | 4.50 | 12.27 |
| … MCP picks the true direction, when it commits | 54.1 % | 58.6 % |
| … SCAD, the same | 59.1 % | 66.8 % |
| … the lasso's larger entry is the true direction | 53.8 % | 63.3 % |
| break-even $1 - F_1/2$ (mean over datasets) | 70.6 % | 73.4 % |
| true 2-cycles the lasso finds in both directions → MCP keeps one direction | 1.30 of 1.37 | 1.33 of 1.47 |

Counterfactuals on the lasso's own best-F1 estimate:

| $F_1$ | $p=10$ | $p=20$ |
|---|---|---|
| lasso as is | 0.589 | 0.531 |
| every hedge settled by a fair coin | 0.544 (−0.045) | 0.471 (−0.060) |
| … and every found 2-cycle cut to one direction | 0.506 (−0.082) | 0.452 (−0.079) |
| every hedge settled by its larger entry (2-cycles spared)¹ | 0.557 (−0.032) | 0.512 (−0.019) |
| every hedge settled correctly (oracle) | 0.654 (+0.065) | 0.602 (+0.071) |
| *for comparison: MCP − lasso at the best-F1 point (S2 §3.4)* | *−0.096* | *−0.076* |

¹ Also drops the smaller entry of hedged false pairs, which can only help.

**Reading.** At $n=1000$ the data favour the true direction only slightly within a hedge, so any
commit rule loses to keeping both, MCP's included. The upside of a good orientation step is real:
+0.065 / +0.071 if every hedge were settled correctly.

**Corrections to the 1 October notes** (`../011026/files/next_steps_2026-10-01.md`):

- "Close to a coin flip" holds across *edges*, not across *datasets*. With $M^*$ fixed, MCP makes
  the same reversal in 199 of 200 datasets (S2 §3.1).
- "Debiasing … trading bias for variance" is superseded by S2 §3.1: MCP's extra error is bias, and
  its per-edge variance equals the lasso's.
- The family "keep the hedge, remove the shrinkage afterwards" does not help support recovery (§2).

---

## 2. Orientation-aware estimators: intuition and references

**Where the direction information lives.** $M\Sigma + \Sigma M^\top = -C$; the edge $i\to j$ is
$M_{ji}$.

| | $i\to j$ ($M_{ji}$) | $j\to i$ ($M_{ij}$) |
|---|---|---|
| the pair's own $(i,j)$ equation | coefficient $\Sigma_{ii}$ (= 1) | coefficient $\Sigma_{jj}$ (= 1) |
| equations with a third node $k$ | enters $(j,k)$, coefficient $\Sigma_{ik}$ | enters $(i,k)$, coefficient $\Sigma_{jk}$ |
| diagonal equations (where $C$ enters) | $j$'s, coefficient $2\Sigma_{ij}$ | $i$'s, coefficient $2\Sigma_{ij}$ |

The two directions are interchangeable in the pair's own equation, which is why the pair's sum is
well identified. They differ only in their side effects, and the reversal curvature
$\kappa_{ij} = \sum_k(\rho_{ik}^2 + \rho_{jk}^2) + 4\rho_{ij}^2$ measures exactly those. Direction
is a property of how the edge fits into the rest of the graph, and the evidence is spread thinly
over many equations.

**Four families of estimators.**

| family | idea | references | verdict |
|---|---|---|---|
| keep the hedge, remove the shrinkage afterwards | LLA, $\gamma$-continuation, Mnet | Zou & Li 2008; Fan, Xue & Zou 2014; Mazumder, Friedman & Hastie 2011; Huang et al. 2016 | no new direction information. The concavity settles hedges by magnitude, which costs $F_1$ (§1), so expect it between MCP and the lasso on support metrics. Its gain is estimation error, and a refit on the lasso's support gives that too. **Drop for support recovery.** |
| skeleton first, direction second | penalise each pair $\{M_{ij}, M_{ji}\}$ as a group, then orient each pair by refitting $i\to j$ / $j\to i$ / both | Yuan & Lin 2006; Huang, Breheny & Ma 2012; Breheny & Huang 2015; Friedman, Hastie & Tibshirani 2010; PC (Spirtes et al. 2000; Kalisch & Bühlmann 2007); MMHC (Tsamardinos et al. 2006) | works with a lasso first stage |
| move directly between orientations | add / delete / reverse search; pair-block coordinate descent | Heckerman, Geiger & Chickering 1995; Chickering 2002; Nandy, Hauser & Maathuis 2018; Améndola et al. 2020; Fu & Zhou 2013 and Aragam & Zhou 2015 (pair-block solver: check) | needs a score that prefers the truth |
| commit only when it pays | commit iff the evidence clears the §1 threshold; otherwise keep both or report the pair as undirected | Andersson, Madigan & Perlman 1997; Lipton et al. 2014 | bootstrap or stability selection cannot set the threshold here: the reversal is bias, so it looks stable |

**Families 2–4 do not need a nonconvex penalty in their first stage.** The lasso finds the skeleton
as well as MCP and keeps the hedges a second stage can test.

**Where nonconvexity still has a role** [scratch: `bp_example2.py`]. In Example 2 at $n=\infty$ the
exact solutions are $M^* + W\Sigma^{-1}$, $W$ skew-symmetric. A linear program finds the one with
the smallest off-diagonal $\ell_1$ norm (the lasso's limit as $\lambda\to0$):

| | edges | $\ell_1$ |
|---|---|---|
| truth, the 5-cycle | 1→2, 2→3, 3→4, 4→5, **5→1** | 3.25 |
| smallest-$\ell_1$ exact fit | all 10 edges of the DAG 1<2<3<4<5, including **1→5** | 2.88 |

- **$\ell_1$ aims at the wrong graph** in the population. An edge count ($\ell_0$; MCP behaves like
  one for large entries) prefers the truth, which is the only exact fit with at most 5 edges (1
  October [scratch]).
- **This is compressed sensing.** Finding the sparsest solution of an underdetermined linear system
  with $\ell_1$ needs a null-space property. Here the null space consists of the orientation swaps,
  so $\ell_1$'s population failures are orientation failures.
- **Framing:** nonconvexity helps as the *score of a discrete search*, not as a penalty on a
  continuation path. The spectrum:
  - $\ell_1$: easy to compute, hedges well, but can target the wrong graph;
  - continuous MCP/SCAD on a path: the right target, but unreachable by coordinate-wise moves;
  - $\ell_0$/BIC with reversal moves: the right target, and reachable.

---

## 3. Example 2 at large $n$ with all three penalties [scratch: `m0_penalties.py`, `m0_objective.py`]

**Setup.** The datasets of `simulations/run_m0.py`, 100 per $n$, with its random stream replayed;
the lasso numbers equal `runs/s1_dettling_reproduction/m0_reps100.csv` exactly. MCP $\gamma=3$ and
SCAD $\gamma=3.7$ use the continuation path, as in S1b.

**Irrepresentability, computed on the population covariance** ($\Gamma = A^\top A$, diagonal
unpenalised):
- path graph: 0.823 (holds);
- 5-cycle: 3.00 at $m_{15} = 0.65$, and 2.99–3.04 over $m_{15}\in[0.5,1]$ (fails).

The reversal curvatures of the five cycle pairs are 0.043, 0.018, 0.0055, 0.0023 and 0.119, all
below $1/\gamma = 0.33$.

**The continuation path** (`max_f1` lasso / MCP / SCAD):

| $n$ | path graph (irrepresentable, 0.82) | 5-cycle, $m_{15}=0.65$ (3.0) |
|---|---|---|
| 100 | 0.611 / 0.596 / 0.603 | 0.547 / 0.549 / 0.550 |
| 1000 | 0.813 / 0.815 / 0.820 | 0.687 / 0.691 / 0.696 |
| 5000 | 0.966 / 0.961 / 0.964 | 0.772 / 0.775 / 0.779 |
| $10^4$ | 0.980 / **0.992** / **0.992** | 0.780 / 0.794 / 0.792 |
| $10^5$ | 1.000 / 1.000 / 1.000 | 0.800 / 0.800 / 0.800 |
| $\infty$ | 1.000 / 1.000 / 1.000 | 0.800 / 0.800 / 0.800 |

- **5-cycle.** No penalty recovers the truth at any $n$. 5→1 never appears at the best-F1 point,
  and from $n = 10^5$ every estimate has 1→5 instead. The random-$m_{15}$ setting is the same.
- **Path graph.** This is the only clear win for the path: exact recovery at $n=10^4$ is 93 % for
  MCP and SCAD against 83 % for the lasso. That is where the lasso already works.

**The MCP objective vs. the path** (5-cycle, 20 datasets per $n$, every 4th $\lambda$):

| $n$ | truth's basin has the lower objective at some $\lambda$ | reversal search (no oracle): exact recovery / `max_f1` | path: exact recovery / `max_f1` |
|---|---|---|---|
| $10^3$ | 45 % | 0 % / 0.659 | 0 % / 0.661 |
| $10^4$ | 100 % | 15 % / 0.779 | 0 % / 0.785 |
| $10^5$ | 100 % | **90 % / 0.980** | 0 % / 0.800 |
| $\infty$ (1 dataset) | 100 % | **100 % / 1.000** | 0 % / 0.800 |

Among the $\lambda$'s at which the truth's basin has exactly the true support, it has the lower
objective at 50 % ($10^3$), 65 % ($10^4$), 82 % ($10^5$) and 96 % ($\infty$) of them.

**How the search works.** It runs at a fixed $\lambda$ and starts from the path's solution there.
For each selected off-diagonal entry it moves the entry's mass to the reverse direction, re-solves
MCP from that start (monotone APG), and keeps the move that lowers

$$F(M) = \tfrac12\|M\hat\Sigma + \hat\Sigma M^\top + C\|_F^2 + \textstyle\sum_{i\neq j}\mathrm{MCP}_{\lambda,\gamma}(M_{ij})$$

the most. It repeats this for at most six rounds.

- **Score:** the MCP objective at that $\lambda$, not BIC.
- **$\lambda$:** the best of the inspected ones, i.e. oracle tuning, like `max_f1`.
- **Moves:** reversals only.
- **"Truth's basin"** is MCP re-solved from $M^*$. It uses the truth, so it shows what the objective
  prefers; it is not an estimator.

**Reading.**
- **Path estimators never beat the lasso where irrepresentability fails.** The MCP objective
  prefers the truth more and more as $n$ grows, and a reversal search reaches it from about
  $n = 10^5$.
- **The theory does not cover this.** Loh & Wainwright (2017) need the curvature to exceed
  $1/\gamma$; here it is below, so the objective has several local minima.

Open questions:
1. Does the gain come from MCP, or from the reversal move plus an edge-counting score? Lasso + BIC
   reversal search is untested.
2. What happens when $\lambda$ is chosen from the data?
3. Does any of this carry over to random graphs?

---

## 4. Random graphs (Figure 5 DGP) at $n=\infty$ [scratch: `ninf_standardize.py`, `s1_nsweep_pilot.py`]

**The lasso, $p=10$, `C_ID`**, on the same 100 graphs as S1 (reps < 25):

| | `max_f1` | `auc` | skeleton $F_1$ | reversed | hedged | exact recovery |
|---|---|---|---|---|---|---|
| $n=1000$ (committed run) | 0.621 | 0.731 | | | | |
| $n=\infty$, standardised (Figure 5 pipeline) | 0.681 | 0.788 | 0.849 | 1.28 | 5.10 | 0 of 100 |
| $n=\infty$, raw scale ($C = 2I$ exactly right) | 0.632 | 0.756 | 0.787 | 2.10 | 1.90 | 0 of 100 |

**All three penalties at $n=\infty$.** This is the local pilot, stopped once the cluster sweep was
ready; it is complete for `C_ID` and `C_Random_Diag`. The pilot's lasso refit at $n=1000$ (80
datasets) equals the committed S1 run exactly.

| | `max_f1` lasso / MCP / SCAD, $n=1000$ | $n=\infty$ | MCP − lasso, $1000 \to \infty$ |
|---|---|---|---|
| `C_ID` | 0.621 / 0.527 / 0.567 | 0.681 / 0.566 / 0.607 | −0.093 → −0.115 ($z$ −10) |
| `C_Random_Diag` | 0.605 / 0.505 / 0.554 | 0.668 / 0.536 / 0.586 | −0.100 → −0.132 ($z$ −13) |

At $n=\infty$, per dataset at the best-F1 point:

| | reversed | hedged | 2-cycles in both directions | skeleton $F_1$ |
|---|---|---|---|---|
| lasso | 1.28 | 5.06 | 1.46 | 0.843 |
| MCP | 3.96 | 0.19 | 0.04 | 0.802 |
| SCAD | 2.87 | 1.30 | 0.44 | 0.808 |

These are 210 datasets: `C_ID`, `C_Random_Diag` and 10 of `C_Random_Min_Diag`. No penalty
recovers a single graph exactly.

**Reading.**
- **The lasso's errors are built in, not noise.** $n=1000$ is within 0.06 of its limit.
- **MCP's path deficit grows with $n$.** Getting stuck on a direction is bias, as in S2 §3.1.
- **Standardising still helps.** It misspecifies $C$ on the correlation scale (the truth there is
  $2D^{-1}$), yet it still beats the correctly specified raw scale for the lasso, as in S1 §8.6.

---

## 5. Cluster n-sweep (set up and submitted 2 October)

**Design.** Figure 5's random graphs at larger $n$, with every loss and penalty:

| | Figure 5 (the existing run) | n-sweep |
|---|---|---|
| random graphs | edge probability $k/p$, $k=1..4$, four $C$ choices, standardised input, $C=2I$ in the fit | identical (same seeds, hence the same graphs) |
| $p$ | 10, 15, 20, 25, 30, 40, 50 | 10, 20 |
| reps | 100 | 25 |
| $n$ | 1000 | $10^3, 10^4, 10^5, \infty$ |
| losses | direct | direct, log-likelihood, Frobenius |
| penalties | lasso | lasso, MCP ($\gamma=3$), SCAD ($\gamma=3.7$) |
| estimator | continuation path, 100 $\lambda$ | the same path; **no reversal search** |

$M^*$ and $C$ are drawn before the data, so all cells see the same 800 graphs and every comparison
is paired, across penalties, losses and $n$.

**Code added to the repository today:**

- `src/gclm/config.py`: `parse_n_obs` (accepts `1000`, `1e5`, `inf`).
- `simulations/run_s1_shard.py`, `simulations/run_s1.py`: a `--n-obs` option. The shard runner
  also checks that the shard index is in range.
- `cluster/s1_array.sbatch`: the shard count comes from `<run>/n_shards` (default 64), and BLAS is
  pinned to one thread.
- `cluster/submit_nsweep.sh`: one array per (loss, penalty, $n$), with `--dry-run`, `--status`,
  `--fill` (resubmit missing shards), `--time` and subset filters.
- `tests/test_shard_runner.py` (16 tests): $n$ parsing; $M^*$ and $C$ identical across $n$; the
  runner end to end at $n=\infty$; the submit script's bookkeeping against a stub `sbatch`.
  - The full suite without R passes (272).
  - On a clean checkout of HEAD plus these files, the tests pass and both covariance losses run at
    $n=\infty$ (APG only; the Newton solver is experimental and unused).
- `docs/REPRODUCTION.md` §2.6: the commands and the cost table.

**Cost.**
- **The September lasso run on LRZ** (64 shards, full grid): all shards started within a minute;
  the slowest took 0.99 h; 50.2 CPU-h in total.
- **Estimates for one cell of the sweep** (800 datasets):
  - direct loss: about 1.4 / 6 / 7 CPU-h (lasso / MCP / SCAD);
  - log-likelihood: about 45 / 6 / 7;
  - Frobenius: about 150 / 90 / 110, most likely on the high side.
- **One $p=20$ dataset on the M2** took 4.5 CPU-min for the log-likelihood lasso and 9 CPU-min for
  the Frobenius lasso, 5–8× their $p=10$ means. No $\lambda$ hit the solver's step limit.

**What the sweep can show:**
- whether the path estimators' deficit persists across losses, at $p=20$, and as $n\to\infty$.

**What it cannot show:**
- anything about the reversal search, which is not in the repository yet.

### Running it on LRZ

**Status (evening of 2 October).**

| | |
|---|---|
| complete | `direct_{lasso,MCP,SCAD}_n1000` (the canary) |
| running or queued | `loglik_*_n1000` (jobs 5640958, 5640990, 5640991); `direct_*_n1e4` (5641059, 5641060, 5641061); `loglik_{lasso,MCP}_n1e4` (5641062, 5641072) |
| not submitted yet | every `frobenius_*` cell; `loglik_SCAD_n1e4`; `direct_*` and `loglik_*` at $n = 10^5$ and $\infty$ |
| LRZ limits | **96 cores at once per user**: further tasks wait as `PD (QOSMaxCpuPerUserLimit)` and start by themselves. **About 200 tasks queued or running per user**: beyond that `sbatch` refuses with `AssocMaxSubmitJobLimit`. |
| to do | the command sheet below, from block 1 on |

#### Command sheet

Copy from top to bottom. Each block is explained below the sheet.

```bash
# ---- 0. every login: shell, repository, Python (the venv is needed for python, not for sbatch)
ssh -Y <your-lrz-id>@cool.hpc.lrz.de
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate

# ---- 1. pull the submit-script fixes (--shards; Frobenius 24 h by default); safe while jobs run
git pull
bash cluster/submit_nsweep.sh --status

# ---- 2. check the canary against the September lasso run (on LRZ it lives in results/)
ls results/s1 | head -3
python simulations/aggregate_s1.py --in-dir results/s1 --out-dir runs/s1_dettling_reproduction
python simulations/aggregate_s1.py --in-dir runs/nsweep_p10-20/direct_lasso_n1000/s1_shards
python simulations/compare_runs.py --baseline runs/s1_dettling_reproduction \
    --run runs/nsweep_p10-20/direct_lasso_n1000 --reps 25 --p 10 20

# ---- 3. the remaining cells, with fewer and longer tasks, in two rounds
squeue -M serial -u $USER -h -r | wc -l              # tasks queued or running right now
# round A (128 tasks), once that count is below about 70
bash cluster/submit_nsweep.sh --n 1000 --loss frobenius --shards 16
bash cluster/submit_nsweep.sh --n 1e4 1e5 inf --loss loglik --shards 8
bash cluster/submit_nsweep.sh --n 1e4 1e5 inf --loss direct --shards 4
# round B (144 tasks), once the count is below about 55
bash cluster/submit_nsweep.sh --n 1e4 1e5 inf --loss frobenius --shards 16

# ---- 4. watch
squeue -M serial -u $USER
bash cluster/submit_nsweep.sh --status
sacct -M serial -X -u $USER -S now-2days --format=JobName%10,JobID%18,State,Elapsed | grep -v COMPLETED
less logs/s1_<jobid>_<task>.err

# ---- 5. repair, only after every job of that cell has ended
bash cluster/submit_nsweep.sh --fill
bash cluster/submit_nsweep.sh --fill --time 24:00:00
scancel -M serial <jobid>

# ---- 6. once --status reports every cell complete
for d in runs/nsweep_p10-20/*/; do python simulations/aggregate_s1.py --in-dir "$d/s1_shards"; done

# ---- 7. on the laptop, from the repository root
scp -r <your-lrz-id>@cool.hpc.lrz.de:~/repo/runs/nsweep_p10-20 runs/
for pen in MCP SCAD; do
  python simulations/compare_runs.py --baseline runs/s1b_pilot_p10-20/$pen \
      --run runs/nsweep_p10-20/direct_${pen}_n1000 --reps 25 --p 10 20
done
for loss in loglik frobenius; do for pen in lasso MCP SCAD; do
  python simulations/compare_runs.py --baseline runs/s2_pilot_p10/${loss}_${pen} \
      --run runs/nsweep_p10-20/${loss}_${pen}_n1000 --reps 10 --p 10
done; done
```

#### What each block does, and what to expect

0. **Every login.** `sbatch` needs nothing activated: each job sets up its own environment. The
   `python` commands (blocks 2 and 6) need the venv.
1. **Pull the fixes.** First commit and push them on the laptop:
   ```bash
   git add cluster/submit_nsweep.sh tests/test_shard_runner.py docs/REPRODUCTION.md
   git commit -m "submit_nsweep.sh: --shards override; Frobenius limit 24 h"
   git push
   ```
   - Pulling is safe while jobs run. It changes only the submit script, its test and the docs,
     not `s1_array.sbatch` or `run_s1_shard.py`, which the jobs use.
   - After the pull, Frobenius cells get 24 h by default and `--shards` is available.
   - `--status` prints one line per cell: `complete`, `submitted (k/N shards written)` or
     `not started`.
2. **Canary check.** The September run predates the repository restructuring, so on LRZ it is in
   `results/`, with its shards in `results/s1/`.
   - Re-aggregating it into `runs/s1_dettling_reproduction` puts the current aggregator on both
     sides of the comparison and leaves `results/` untouched.
   - The aggregator should print `datasets : 11200 (expected 11200)` for the September run and
     `datasets : 800 (expected 800)` for the canary.
   - `compare_runs.py` prints a mean difference per $(p, C)$ and metric: all should be `+0.000`.
     If not, stop and send the output.
3. **The remaining cells.** At the default shard counts (16 / 32 / 64) they are almost 900 tasks,
   against a queue of about 200. `--shards` makes fewer, longer tasks for the cells not yet
   submitted; cells already submitted keep their counts.

   | loss | `--shards` | datasets per task | expected time per task |
   |---|---|---|---|
   | direct | 4 | 200 | about 2 h (limit 6 h) |
   | log-likelihood | 8 | 100 | about 5–6 h (limit 24 h) |
   | Frobenius | 16 | 50 | about 6 h (limit 24 h) |

   - **Round A** is Frobenius at $n = 1000$, the log-likelihood cells left at $10^4$, $10^5$ and
     $\infty$, and the direct cells at $10^5$ and $\infty$: 128 tasks.
   - **Round B** is Frobenius at the three larger $n$: 144 tasks.
   - Submit each round once the queue has room. If `sbatch` still refuses, nothing is recorded for
     that cell; run the same command again later and it continues where it stopped.
   - With 96 cores at a time, everything takes about 12–15 hours of continuous running, so expect
     about a day.
4. **Watch.**
   - `squeue`: `PD` waiting, `R` running.
   - `--status`: shards written per cell.
   - `sacct … | grep -v COMPLETED`: jobs that failed or ran out of time.
   - The `.err` log of a task shows its Python traceback.
5. **Repair.**
   - `--fill` resubmits exactly the shards that are missing. Use it only once `squeue` shows no job
     of that cell; otherwise shards that are still running get submitted twice.
   - After a TIMEOUT, `--time 24:00:00` helps only the direct-loss cells, which ran with 6 h. A
     covariance-loss shard at the 24 h cap needs more shards instead; ask before deleting
     anything.
6. **Aggregate.** This writes `s1_per_dataset.csv`, `s1_summary.csv` and `s1_curves.csv` into each
   cell's folder; each should report `datasets : 800 (expected 800)`.
7. **Bring it home and check.**
   - About 90 MB including the shards.
   - The loops compare the cells that repeat local runs: MCP and SCAD at $n = 1000$ against the S1b
     pilot, and the covariance losses at $p = 10$, reps < 10, against the S2 pilot.
   - All differences should be 0; nonconvex fits may differ in the last digits across machines.
     Then the analysis (§6).

**Rough timeline.** About 1,100–1,400 CPU-h are left in total, and LRZ runs 96 of your tasks at a
time, so allow about a day with the two submission rounds.

#### Done so far (for the record)

```bash
# step 0, laptop: commit and push the new code
git add src/gclm/config.py simulations/run_s1_shard.py simulations/run_s1.py \
        cluster/s1_array.sbatch cluster/submit_nsweep.sh tests/test_shard_runner.py docs/REPRODUCTION.md
git commit -m "n-sweep: --n-obs (incl. inf), per-run shard count, cluster/submit_nsweep.sh"
git push
# step 1, LRZ: update (if refused because of local edits: git stash && git pull && git stash pop)
git pull
# step 2, LRZ: quick checks -- all passed; the dry run printed the expected sbatch lines
python -m pytest tests/test_shard_runner.py -q
python simulations/run_s1_shard.py --shard 0 --n-shards 800 --p 10 20 --reps 25 \
    --n-obs inf --loss loglik --penalty MCP --out-dir /tmp/$USER-smoke
bash cluster/submit_nsweep.sh --dry-run | head -3
# step 3, LRZ: the canary -- complete
bash cluster/submit_nsweep.sh --n 1000 --loss direct
# step 4, LRZ: n = 1000 -- log-likelihood queued, Frobenius refused at 48 h (block 1 above redoes it)
bash cluster/submit_nsweep.sh --n 1000
```

The canary ran at $n = 1000$, which only reproduces existing results. The $n = \infty$ variant
discussed the same day (new results, checked against `files/s1_nsweep_p10.csv`) was not used.

**Submission log (2 October).**

- Canary: `direct_{lasso,MCP,SCAD}_n1000`, all three cells complete.
- `--n 1000`: `loglik_{lasso,MCP,SCAD}_n1000` submitted (jobs 5640958, 5640990, 5640991, 24 h
  limit). `frobenius_lasso_n1000` was refused: "Requested time limit is invalid". `serial_std` does
  not allow 48 h. Nothing was recorded; the Frobenius cells go in with `--time 24:00:00` (block 1).
- The script's default for Frobenius is now 24 h (not yet committed).
- `--n 1e4 1e5 inf --loss direct loglik`: `direct_*_n1e4` (5641059, 5641060, 5641061) and
  `loglik_{lasso,MCP}_n1e4` (5641062, 5641072) were submitted. `loglik_SCAD_n1e4` was refused with
  `AssocMaxSubmitJobLimit` and nothing was recorded. Running tasks: exactly 96; the rest wait with
  `QOSMaxCpuPerUserLimit`. The Frobenius cells at $n = 1000$ are not in the queue.
- Response: a `--shards` option (committed as "fix slurm batch"); the remaining cells go in, with fewer shards,
  in two rounds (block 3).
- Round B, `--n 1e4 1e5 inf --loss frobenius --shards 16`: 7 of 9 cells submitted (jobs
  5641425–5641431, i.e. all three penalties at $10^4$ and $10^5$ plus `frobenius_lasso_ninf`).
  `frobenius_MCP_ninf` was refused (`AssocMaxSubmitJobLimit`) and `frobenius_SCAD_ninf` was not
  attempted. To do: rerun the same command once
  `squeue -M serial -u $USER -h -r | wc -l` is below about 150; it submits only those two cells.

---

## 6. Next steps

1. **Canary check, then the rest of the sweep** (§5, steps 3–7).
2. **When the sweep is back:**
   - aggregate it;
   - paired comparisons per (loss, $n$);
   - orientation breakdown and skeleton metrics;
   - write it up as a new section of `simulations/S2_penalties_losses.md`.
3. **Reversal search into the repository** (planned in `simulations/S3b_reversal_search.md`):
   - the solver, with tests (Example 2 at $n=\infty$ is turned into $M^*$; the objective never
     increases) and documentation;
   - then the fair comparison, with $\lambda$ chosen by BIC: lasso + BIC reversal search, MCP + the
     same search, and the lasso alone;
   - run it first on the Example 2 $n$-sweep, then on random graphs at $n=\infty$, on both the raw
     and the standardised scale;
   - on the cluster, as one more option of the shard runner.
4. **Small fixes to the docs:**
   - S2 §4 gives the lasso's full-grid cost as 38.6 CPU-h, which is the M2 figure; the cluster run
     used 50.2;
   - REPRODUCTION §5 says cluster core-hours are "roughly comparable" to the M2; they are about
     1.3×.
5. **For the 10–11 October meeting:**
   - Framing: orientation is the hard part of GCLM structure learning. Nonconvexity helps as the
     score of a discrete search (Example 2), not as a penalty on a continuation path.
   - Is BIC tuning required?
   - Should 2-cycles stay in the DGP?

---

## Files ([`files/`](files/))

| file | what it does | § |
|---|---|---|
| `coinflip.py` | lasso × MCP/SCAD outcome per true pair; fair-coin counterfactual; break-even | 1 |
| `hedge_split.py` | inside the lasso's hedges: is the larger entry the true direction; prune counterfactual | 1 |
| `bp_example2.py` | smallest-$\ell_1$ exact fit of Example 2 (linear program) | 2 |
| `m0_penalties.py` → `m0_penalties_reps100.csv` | `run_m0.py`'s datasets with the three penalties; population irrepresentability | 3 |
| `m0_objective.py` → `m0_objective.csv`, `m0_objective_summary.py` | truth's basin vs. the path; greedy reversal search on the MCP objective | 3 |
| `ninf_standardize.py` | lasso at $n=\infty$, standardised vs. raw | 4 |
| `s1_nsweep_pilot.py` → `s1_nsweep_p10.csv` (partial), `s1_nsweep_summary.py` | the stopped $p=10$ pilot, and its summary with the $n=1000$ validation | 4, 5 |

---

## References added today

Check every one of these before citing. The references of the 1 October notes are not repeated.

- Andersson, S. A., Madigan, D. & Perlman, M. D. (1997). A characterization of Markov equivalence
  classes for acyclic digraphs. *Ann. Statist.* 25(2), 505–541.
- Aragam, B. & Zhou, Q. (2015). Concave penalized estimation of sparse Gaussian Bayesian networks.
  *JMLR* 16, 2273–2328.
- Breheny, P. & Huang, J. (2015). Group descent algorithms for nonconvex penalized linear and
  logistic regression models with grouped predictors. *Stat. Comput.* 25(2), 173–187.
- Candès, E. J., Wakin, M. B. & Boyd, S. P. (2008). Enhancing sparsity by reweighted $\ell_1$
  minimization. *J. Fourier Anal. Appl.* 14, 877–905.
- Chartrand, R. (2007). Exact reconstruction of sparse signals via nonconvex minimization. *IEEE
  Signal Process. Lett.* 14(10), 707–710.
- Foucart, S. & Lai, M.-J. (2009). Sparsest solutions of underdetermined linear systems via
  $\ell_q$-minimization for $0<q\le1$. *Appl. Comput. Harmon. Anal.* 26(3), 395–407.
- Foucart, S. & Rauhut, H. (2013). *A Mathematical Introduction to Compressive Sensing.*
  Birkhäuser (ch. 4, the null-space property).
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). Applications of the lasso and grouped lasso to
  the estimation of sparse graphical models. Technical report, Stanford University.
- Fu, F. & Zhou, Q. (2013). Learning sparse causal Gaussian networks with experimental
  intervention: regularization and coordinate descent. *JASA* 108, 288–300.
- Huang, J., Breheny, P. & Ma, S. (2012). A selective review of group selection in
  high-dimensional models. *Statist. Sci.* 27(4), 481–499.
- Kalisch, M. & Bühlmann, P. (2007). Estimating high-dimensional directed acyclic graphs with the
  PC-algorithm. *JMLR* 8, 613–636.
- Lipton, Z. C., Elkan, C. & Narayanaswamy, B. (2014). Optimal thresholding of classifiers to
  maximize F1 measure. *ECML PKDD 2014* (arXiv:1402.1892).
- Loh, P.-L. & Wainwright, M. J. (2017). Support recovery without incoherence: a case for
  nonconvex regularization. *Ann. Statist.* 45(6), 2455–2482.
- Meinshausen, N. & Bühlmann, P. (2010). Stability selection. *JRSS-B* 72(4), 417–473.
- Nandy, P., Hauser, A. & Maathuis, M. H. (2018). High-dimensional consistency in score-based and
  hybrid structure learning. *Ann. Statist.* 46(6A), 3151–3183.
- Peters, J. & Bühlmann, P. (2014). Identifiability of Gaussian structural equation models with
  equal error variances. *Biometrika* 101(1), 219–228.
- Reisach, A. G., Seiler, C. & Weichwald, S. (2021). Beware of the simulated DAG! Causal discovery
  benchmarks may be easy to game. *NeurIPS 2021*.
- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.).
  MIT Press.
- Tsamardinos, I., Brown, L. E. & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian
  network structure learning algorithm. *Machine Learning* 65(1), 31–78.
- van de Geer, S. & Bühlmann, P. (2013). $\ell_0$-penalized maximum likelihood for sparse directed
  acyclic graphs. *Ann. Statist.* 41(2), 536–567.
