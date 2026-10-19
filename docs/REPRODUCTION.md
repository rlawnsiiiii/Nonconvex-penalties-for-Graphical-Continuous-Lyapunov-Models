# Reproducing Dettling Figures 3 and 5

Commands in order, with what each produces. The cluster writes numbers only, the plots are made
locally, and everything a run produces lives in one folder, `runs/<run name>/`.

| stage | where | script | output |
|---|---|---|---|
| Figure 3 (Example 2) | cluster, one serial job | `simulations/run_m0.py` | `<run>/m0_reps100.csv` + `.npz` |
| Figure 5 (Section 5) | cluster, 64-task array | `simulations/run_s1_shard.py` | `<run>/s1_shards/shard_*.npz` |
| aggregation | cluster login node or local | `simulations/aggregate_s1.py` | `<run>/s1_*.csv` (3 tidy CSVs) |
| plots | local only | `simulations/plot_figures.py` | `<run>/figures/*.png`, `*.pdf` |

```
runs/s1_dettling_reproduction/
  m0_reps100.csv, m0_reps100.npz       Figure 3 data
  s1_shards/shard_*_of_0064.npz        Figure 5 raw data (gitignored: large, reproducible)
  s1_summary.csv, s1_per_dataset.csv, s1_curves.csv    Figure 5 aggregated
  figures/                              the plots, made locally
```

Studies and their notes:

| study | section | scripts | results | note |
|---|---|---|---|---|
| S1, the lasso on Dettling's grid, Figures 3 and 5 | §2.4, §3, §4 | `run_m0.py`, `run_s1_shard.py`, `aggregate_s1.py`, `plot_figures.py` | `runs/s1_dettling_reproduction/` | [`S1_reproduction.md`](../simulations/S1_reproduction.md) |
| S1b, MCP and SCAD textbook paths on the direct loss | §2.4 | `s1_array.sbatch <folder> --penalty`, `aggregate_s1.py` | `runs/s1b_mcp_textbook/`, `runs/s1b_scad_textbook/` | [`S2_penalties_losses.md`](../simulations/S2_penalties_losses.md) |
| S2, three penalties on Varando's losses | §2.4 | `s1_array.sbatch <folder> --loss --penalty`, `aggregate_s1.py`, `plot_penalties.py` | `runs/s2_<loss>_<penalty>/` | [`S2_penalties_losses.md`](../simulations/S2_penalties_losses.md), §4 for the costs; [`LIKELIHOOD.md`](LIKELIHOOD.md) for the losses |
| S2b, the n-sweep | §2.6 | `cluster/submit_nsweep.sh`, `aggregate_s1.py`, `compare_runs.py`, `diagnostics/orientation.py`, `diagnostics/nsweep.py`, `diagnostics/plot_nsweep.py` | `runs/nsweep_p10-20/<loss>_<penalty>_n<n>/` | [`S2b_nsweep.md`](../simulations/S2b_nsweep.md) |
| S4, the campaign, waves 1 to 9 | §2.7 | `cluster/submit_campaign.sh`, `cluster/feed_queue.sh`, `cluster/campaign_array.sbatch` with `run_s1_shard.py`, `run_search_shard.py` or `rescore_shard.py`; `diagnostics/campaign.py`, `plot_campaign.py`, `restarts.py`, `plot_figure5.py`, `figure5_table.py` | `runs/campaign/<cell>_n<n>/`; tables and `figures/` in `runs/campaign/` | [`S4_campaign.md`](../simulations/S4_campaign.md); plan, reasons and code map [`cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md); estimators [`DENSE_START.md`](DENSE_START.md), [`SEARCH.md`](SEARCH.md); plan files `cluster/plan_071026.txt`, `cluster/plan_101026.txt` |
| wave 9, the mechanism experiments E1 to E4 | §2.7 | `submit_campaign.sh --wave 9 --group`, `diagnostics/graph_class.py`, `betamin.py`, `twocycles.py` and their plot scripts | `runs/campaign/`; illustrations in `docs/figures/` | [`plan_mechanism_131026.md`](../next_steps/131026/plan_mechanism_131026.md); [`GRAPH_CLASSES_AND_BETAMIN.md`](GRAPH_CLASSES_AND_BETAMIN.md); `DENSE_START.md` §5.1–5.2 |

---

## 1. What the cluster writes

Shards store the raw per-$\lambda$ confusion counts, not just the four reported metrics, so every
metric of Definitions G.4/G.5 can be recomputed later. Per shard, `shard_NNNN_of_0064.npz`, compressed:

| array | shape | what |
|---|---|---|
| `p`, `k`, `c_choice`, `rep` | (n,) | which cell of the grid |
| `lambdas` | (n, 100) | the λ grid actually used |
| `lambda_max` | (n,) | its right endpoint |
| `conf_offdiag` | (n, 100, 4) | tp, fp, tn, fn, off-diagonal scoring |
| `conf_incdiag` | (n, 100, 4) | the same, including the diagonal: the `include_diagonal` variant, stored in parallel |
| `nnz` | (n, 100) | selected off-diagonal entries per λ |
| `objective` | (n, 100) | penalised objective value per λ |
| `n_true_edges` | (n,) | true off-diagonal support size |
| `m_true_{i,j,v}` | ragged | $M^\*$ as a sparse triple, exact edge weights |
| `m_best_f1_{i,j,v}` | ragged | $\hat M$ at the best-$F_1$ λ, sparse |
| `m_best_acc_{i,j,v}` | ragged | $\hat M$ at the best-accuracy λ, sparse |
| `best_f1_index`, `best_acc_index` | (n,) | which λ those were |
| `seconds` | (n,) | per-dataset runtime |
| `config_json` | scalar | the complete `S1Config`, so the run is self-describing |
| `provenance_json` | scalar | host, Python/NumPy versions, platform, wall time, UTC finish |
| not stored | | $\hat\Sigma$ and the 100 full $\hat M$ matrices per dataset, ~22 GB at $p = 50$; exactly reproducible from `config_json` plus `(p, k, c_choice, rep)`, because the per-task RNG is seeded as `default_rng([seed, p, k, c_index, rep])`, independent of worker count and scheduling order (§6) |
| total output | | roughly 100 MB for the whole grid |

---

## 2. Running it on the LRZ Linux Cluster

| where | what |
|---|---|
| every job | the LRZ serial-job template: `--clusters=serial`, `--partition=serial_std`, `--export=NONE` with `--get-user-env`, the mandatory `module load slurm_setup`; job names under 10 characters as LRZ asks |
| your login shell: setup, smoke test, aggregation | you run `module load python` and `source ~/venvs/gclm/bin/activate` yourself, in every new session (§2.2) |
| a batch job, `sbatch` | the `.sbatch` script does both itself (§2.4) |

### 2.1 One-time setup

```bash
ssh -Y $USER@cool.hpc.lrz.de          # 2FA; see the LRZ access documentation
git clone <this repo> ~/repo
cd ~/repo
bash cluster/setup_env.sh             # builds the venv at ~/venvs/gclm; runs the five lines below in one go
```

```bash
# what setup_env.sh runs; you can also type it by hand
module load python                    # first, always: a venv has no Python of its own, it points back at the interpreter it was
                                      # created with, and that interpreter is on the path only while its module is loaded;
                                      # load the SAME module every time you use the venv, interactively or in a job
python3 -m venv ~/venvs/gclm          # the batch scripts use exactly this path; if you move it, change VENV= in
                                      # cluster/setup_env.sh, cluster/s1_array.sbatch and cluster/m0.sbatch
source ~/venvs/gclm/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.24" "scipy>=1.10"   # all that is needed: the default solver (fista) is pure Python,
                                                    # so no R, glmnet, ncvreg or skglm is required on the cluster
```

```bash
module avail python                   # if `module load python` fails: pick one, e.g. `module load python/3.x`, and use that
                                      # name in cluster/setup_env.sh AND both .sbatch files; pinning an explicit version in
                                      # all three scripts avoids being surprised by a changed default
rm -rf ~/venvs/gclm && bash cluster/setup_env.sh    # recreate the venv when the cluster's default Python changes or you switch versions
```

### 2.2 Every new login session

```bash
ssh -Y $USER@cool.hpc.lrz.de          # a fresh session has neither the module nor the venv
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate
which python                          # should print ~/venvs/gclm/bin/python
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
deactivate                            # leaves the venv again
```

### 2.3 Smoke test before submitting

```bash
# about a minute on the login node, environment of §2.2 active; the code path of a real array task, on one dataset
python simulations/run_s1_shard.py --shard 0 --n-shards 400 --p 10 --reps 1 --out-dir /tmp/gclm_smoke   # ends with "done: 1 datasets ..."
python simulations/aggregate_s1.py --in-dir /tmp/gclm_smoke --out-dir /tmp/gclm_smoke                    # warns that datasets are missing: expected,
                                                                                                         # only one dataset of the grid was run
python -m pip install pytest          # optional: the test suite; tests that need R, skglm, pyproximal or cvxpy skip themselves
python -m pytest -m "not r" -q
```

### 2.4 Submit

Nothing needs to be activated before `sbatch`.

```bash
# --export=NONE: a job starts with a clean environment and inherits nothing from your shell, no modules, no venv;
# after module load slurm_setup each .sbatch script runs `module load python; source ~/venvs/gclm/bin/activate`
# itself and prints the interpreter it ended up with into its log, so a wrong environment shows up there
mkdir -p logs
sbatch cluster/m0.sbatch              # Figure 3: one serial job, ~5 min   -> runs/s1_dettling_reproduction/m0_reps100.{csv,npz}
sbatch cluster/s1_array.sbatch        # Figure 5: array 0-63               -> runs/s1_dettling_reproduction/s1_shards/
```

```bash
# any other run: its own folder as the first argument, or it overwrites the old shards (same file names);
# further arguments go straight to run_s1_shard.py (arguments, not environment variables: with --export=NONE
# a job inherits no environment but does receive its arguments)
sbatch cluster/s1_array.sbatch runs/s1b_mcp_textbook --penalty MCP      # S1b
sbatch cluster/s1_array.sbatch runs/s1b_scad_textbook --penalty SCAD
for loss in loglik frobenius; do      # S2: the same grid on Varando's losses (docs/LIKELIHOOD.md), one submission per (loss, penalty)
  for pen in lasso MCP SCAD; do       # same output layout, so aggregate_s1.py and plot_penalties.py apply unchanged
    sbatch cluster/s1_array.sbatch "runs/s2_${loss}_${pen}" --loss "$loss" --penalty "$pen"
  done                                # heavier than the direct loss: check simulations/S2_penalties_losses.md §4 against the array's --time first
done
```

```bash
squeue -M serial -u $USER             # progress
sacct  -M serial -j <jobid> --format=JobID,State,Elapsed,MaxRSS
ls runs/s1_dettling_reproduction/s1_shards/*.npz | wc -l      # expect 64 when complete
head -3 logs/s1_<jobid>_0.out         # host, shard, and the python actually used
```

```bash
# verify array support before relying on it: the LRZ pages consulted document serial jobs and --clusters/--partition
# but not --array limits; if arrays are restricted on your segment, submit the shards as independent jobs
# (the runner takes --shard i --n-shards N and needs nothing else); each wrapped job sets up its environment itself
for i in $(seq 0 63); do
  sbatch --clusters=serial --partition=serial_std --time=06:00:00 \
         --ntasks=1 --mem=2G -J gclm-s$i --export=NONE --get-user-env \
         --wrap "module load slurm_setup; module load python; \
                 source \$HOME/venvs/gclm/bin/activate; \
                 python simulations/run_s1_shard.py --shard $i --n-shards 64 \
                        --out-dir runs/s1_dettling_reproduction/s1_shards"
done
```

### 2.5 Resource choices, and why

| setting | value | why |
|---|---|---|
| `--cpus-per-task` | 1 | each array task is single-core; parallelism comes from the array |
| `--mem` | 2G | one worker peaks near 70 MB even at $p = 50$; the solver is matrix-free, so nothing scales with $p^2$; 2G is generous |
| `--time` | 06:00:00 | the slowest shard should finish well inside 2 h; the margin covers the long tail (a few $p = 50$ datasets take ~200 s) and a busy node. `serial_std` caps the time: on 2 October 2026 it refused 48 h and accepted 24 h; `sinfo -M serial -p serial_std -o "%P %l"` shows the current cap |
| shards | 64 | datasets are assigned stride-wise (`tasks[i::64]`), so every shard gets a mix of $p$ values and they finish at roughly the same time; no straggler shard of pure $p = 50$ |

### 2.6 The n-sweep: $p = 10, 20$ at growing $n$, every loss and penalty

Thirty-six cells, one array each; `cluster/submit_nsweep.sh` does the bookkeeping.

| item | detail |
|---|---|
| grid | Figure 5's data-generating process at $p \in \{10, 20\}$: 4 $k$ × 4 $C$ × 25 reps = 800 datasets per cell |
| cells | loss (direct, log-likelihood, Frobenius) × penalty (lasso, MCP, SCAD) × $n$ ($10^3, 10^4, 10^5, \infty$) = 36, in `runs/nsweep_p10-20/<loss>_<penalty>_n<n>/`; `--n-obs inf` feeds the population covariance |
| paired | $M^*$ and $C$ are drawn before the data, so all 36 cells see the same 800 drift matrices and every comparison, across penalties, losses or $n$, is paired (`test_drift_and_volatility_do_not_depend_on_n`) |
| bookkeeping | `submit_nsweep.sh` writes each cell's shard count into its folder (`n_shards`, which `s1_array.sbatch` reads), never submits a cell twice, and resubmits only the missing shards when asked |
| not rerun | the direct loss at $n = 1000$: those datasets are $p = 10, 20$, reps < 25 of the full lasso run (`runs/s1_dettling_reproduction`) and of the S1b pilot (MCP, SCAD), so its three cells stay "not started" |
| checks | the canary cells, aggregated, against the local pilot `next_steps/021026/files/s1_nsweep_p10.csv` at $p = 10$, `C_ID`, `C_Random_Diag`: they must agree exactly; the covariance-loss cells at $n = 1000$, $p = 10$, reps < 10 repeat the S2 pilot (`runs/s2_pilot_p10`) and should agree with it |
| LRZ limits, measured 2 October 2026 | at most 96 of a user's tasks run at once, further ones wait as `PD (QOSMaxCpuPerUserLimit)` and start by themselves; about 200 tasks may be queued or running, beyond that `sbatch` refuses with `AssocMaxSubmitJobLimit`; a refused cell records nothing, so rerunning the same command continues where it stopped |
| analysis across cells | `simulations/diagnostics/nsweep.py` and `plot_nsweep.py`; results in [`S2b_nsweep.md`](../simulations/S2b_nsweep.md); all 36 cells with their shards are about 120 MB |

```bash
cd ~/repo && git pull
bash cluster/submit_nsweep.sh --dry-run            # the 36 sbatch commands; submits nothing
bash cluster/submit_nsweep.sh --status             # complete / k of N shards written / not started
# the order used on 2 October 2026
bash cluster/submit_nsweep.sh --n inf --loss direct   # 1. canary: the direct loss at n = inf, 48 tasks, under an hour; a cheap first run of the new
                                                      #    code path (--n-obs, n_shards, the array override) that produces new results; when --status
                                                      #    shows its three cells complete and logs/*.err are empty, aggregate and compare with the pilot
bash cluster/submit_nsweep.sh --n 1e4 1e5 inf                    # 2. everything else in one go, 1,248 tasks; skips the canary cells
bash cluster/submit_nsweep.sh --n 1000 --loss loglik frobenius
# at the default shard counts the sweep is far more tasks than the queue takes: give the cells not yet submitted fewer, longer
# tasks with --shards, for example 4 (direct), 8 (log-likelihood) and 16 (Frobenius), and submit them in rounds as the queue
# empties; if sbatch refuses with a per-user limit, submit fewer cells at a time
sacct -M serial -X -u $USER -S now-2days --format=JobName%10,JobID%18,State,Elapsed   # how long each shard took; if a loss needs more time or
                                                                                      # more shards, edit shards_for / time_for at the top of the
                                                                                      # script before submitting more of it (a cell already
                                                                                      # submitted keeps the shard count it recorded)
bash cluster/submit_nsweep.sh --fill               # once a cell's jobs have ended, --status shows shards never written (TIMEOUT, node failure:
                                                   # check sacct) and --fill resubmits exactly those; after a TIMEOUT add --time, up to the partition's cap
```

| loss | shards (default) | `--time` | CPU-h per cell: lasso / MCP / SCAD (measured) | longest shard |
|---|---|---|---|---|
| direct | 16 | 6 h | 1.4 / 6.2 / 8.4 | 2.3 h (4 shards) |
| log-likelihood | 32 | 24 h | 24 / 6.8 / 6.8 | 3.5 h (8 shards) |
| Frobenius | 64 | 24 h | 52 / 37 / 37 | 4.9 h (16 shards) |
| all, measured on the run of 2–3 October 2026 ([`S2b_nsweep.md`](../simulations/S2b_nsweep.md) §4) | | | 180 CPU-h per sample size, 720 for all four, 70 % of it on the Frobenius loss; the sample size does not change the cost | with `--shards 4 / 8 / 16` (direct / log-likelihood / Frobenius) no task ran longer than 5 h, so those counts fit the 24 h limit with a wide margin and keep a whole sweep under the per-user job limit; wall-clock depends on how many tasks LRZ runs at once (96 on that run) |

```bash
for d in runs/nsweep_p10-20/*/; do python simulations/aggregate_s1.py --in-dir "$d/s1_shards"; done   # aggregate every cell (§3), on the cluster
scp -r $USER@cool.hpc.lrz.de:~/repo/runs/nsweep_p10-20 runs/                                          # then locally (§4)
# each penalty against the lasso on the same loss and n
for d in runs/nsweep_p10-20/*_{MCP,SCAD}_n*; do
  pen=$(basename "$d" | cut -d_ -f2)
  python simulations/compare_runs.py --baseline "${d/_${pen}_/_lasso_}" --run "$d" \
      --reps 25 --p 10 20 --csv "$d/paired_vs_lasso.csv"
done
# skeleton vs. orientation at the best-F1 point (reads the shards)
python simulations/diagnostics/orientation.py --shards \
    lasso=runs/nsweep_p10-20/direct_lasso_ninf MCP=runs/nsweep_p10-20/direct_MCP_ninf \
    SCAD=runs/nsweep_p10-20/direct_SCAD_ninf --reps 25 --p 10 20
```

### 2.7 The campaign: every estimator with $C = 2I$ and with the rescaled $C$

Same graphs as the n-sweep, same seeds, so everything is paired with `runs/nsweep_p10-20` too.

| wave | cells per $n$ | tasks per $n$ | what |
|---|---|---|---|
| 1 | 16 | 128 | direct loss, $p = 10, 20$: lasso, MCP / SCAD (standard, dense → sparse, LLA), adaptive lasso; each with $C = 2I$ (`C2I`) and the rescaled $C$ (`Cresc`); path, graph selected by the score, graph after the greedy search on the score |
| 2 | 3 | 20 | search from random graphs and search started from the truth: $p = 10$ (`C2I`, `Cresc`), $p = 20$ (`Cresc`) |
| 3 | 8 | 56 | log-likelihood loss, $p = 10$: lasso and MCP in both path orders, `C2I` and `Cresc`; path and the graph selected by the score |
| 4 | 12 per $p$ | 112 – 224 per $p$ | larger $p$ at $n = 1000$ ($p = 15, 25, 30, 40, 50$): lasso, MCP, SCAD, MCP / SCAD dense → sparse, adaptive lasso, `C2I` and `Cresc`; path and the graph selected by the score, no search. Selected with `--p`, not `--n`; cells `..._p<p>_n1000` |
| 5 | 6 + 3 + 6 + 3 | 48 + 48 (+ 32) + 64 + 20 | three checks of the selection step: (a) at $p = 10$ the score with the maximised likelihood (`--refit loglik`) for the lasso, MCP dense → sparse and the adaptive lasso, `C2I` and `Cresc`, with the search (cells `direct_<estimator>-ml_<C>`); (b) the pure search from 100 randomly drawn starting graphs, sparse and uniform (`search100s_p10_Cresc`, `search100u_p10_Cresc`), and 30 at $p = 20$ on 5 replicates (`search30s_p20_Cresc`); `simulations/diagnostics/restarts.py` reads off the best of the first $r$ starts; (c) the eBIC term inside the selection and the search (`--ebic-gamma 0.5 1`, cells `direct_<estimator>-ebic_<C>`, $p = 10, 20$) and in the pure search (`searche1_p<p>_<C>`) |
| 6 | 16 | 128 | every wave 1 cell rescored with the eBIC penalty inside the selection and the search, from its stored supports (`simulations/rescore_shard.py`; no path recomputed); cells `rescore1_<source cell>`, overlaid on the source rows by `campaign.py` |
| 7 | 14 | 244 | the log-likelihood loss over $p = 10, 20$ ($p = 30$ left out for cost), both $C$: lasso, MCP sparse → dense, MCP dense → sparse from the exact fit and from the lasso solution (`--up-start lasso`), adaptive lasso (`--method adaptive` on `--loss loglik`); selection and search with the least-squares refit; at $p = 10$ only the two estimators wave 3 lacks; cells `loglik_<estimator>_<C>[_p20]` |
| 8 | 30 | 604 (556 at $n = 10^4$, where the six direct-loss cells are wave 5a's) | the likelihood refit: the stored paths of waves 1, 3 and 7 scored again with the model on every graph fitted by maximum likelihood (`rescore_shard.py --refit loglik`), selection and search at $p = 10$, the selection only at $p = 20$, cells `<loss>_<estimator>-ml_<C>[_p20]`; the search from 100 random graphs, the empty graph and the truth with the likelihood refit, $p = 10$, 2 replicates, the starts of a graph in 4 tasks (`--start-blocks 4`; `search100sml_p10_<C>`). Plan file `cluster/plan_101026.txt` (waves 7 and 8 at $n = 10^4$ and 1000; wave 8 after wave 7) |
| 9 | 20 + 4 + 24 + 6 + 30 + 20 | 200 + 16 + 192 + 48 + 240 + 240 | the mechanism experiments of `next_steps/131026/plan_mechanism_131026.md`, direct loss, wave 1's graphs, selection and search, in six groups selected with `--group`: `starts` (E1: MCP dense → sparse and the adaptive lasso started from `exact` / `l1` / `l2` / `truth-0.05` / `truth-0.2` instead of the lasso's dense end, `--up-start` / `--pilot`; cells `direct_MCP-up-<start>_<C>`, `direct_adaptive-<start>_<C>`), `threshold` (E2: `--method threshold[-norefit]`; `direct_threshold[-norefit]_<C>`), `betamin` (E3: `--weights uniform --w-min` 0 / 0.1 / 0.25 / 0.5 for lasso, MCP-up, adaptive, the true C = 2I graphs, every reduction; `direct_<estimator>_<C>_w<wmin>`, $n = 10^3, 10^4, \infty$), `class` (E4: `--graph dag`; `direct_<estimator>_<C>_dag`, $n = 10^3, 10^4, \infty$), `twocycle` (E5: `--graph pairs --two-cycle-share` 0 / 0.25 / 0.5 / 0.75 / 1; `direct_<estimator>_<C>_s<share>`, $n = 10^3, 10^4, \infty$), `starts-loglik` (E1 on the log-likelihood loss at $p = 10$, optional). `DENSE_START.md` §5.1–5.2 |

| item | detail |
|---|---|
| notes | plan, reasons and code map: [`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md); the estimators: [`DENSE_START.md`](DENSE_START.md) and [`SEARCH.md`](SEARCH.md); results: [`S4_campaign.md`](../simulations/S4_campaign.md) |
| `cluster/submit_campaign.sh` | bookkeeping exactly as `submit_nsweep.sh`: one submission per cell, `--status`, `--fill`; a cell is a folder `runs/campaign/<cell>_n<n>/` with its shards in `shards/` |
| `cluster/campaign_array.sbatch` | the job script; its first argument is the runner: `simulations/run_s1_shard.py`, `simulations/run_search_shard.py` or `simulations/rescore_shard.py` |
| `submission`, in the cell folder since 13 October 2026 | what the cell was submitted with: wave, runner, arguments, time limit; `--fill` resubmits from that record and reports a cell recorded by another wave instead of filling it with the wrong runner (six cell names belong to both wave 5 and wave 8); a folder from before the record is filled only when its shard count matches the wave's definition |
| `logs/submissions.log` | every `sbatch` call with the time and the git commit; every shard records the commit the task ran from (`provenance_json`) |
| temporary shard names | shards are written to a temporary name and renamed into place, so a task killed while saving leaves nothing `--fill` would count as written |
| the queue | LRZ accepts about 200 queued or running tasks per user and runs 96 at a time; when `sbatch` refuses a cell, the script records nothing for it and stops, and the same command, run again later, continues there; `squeue -M serial -u $USER -h -r \| wc -l` shows the room |
| `cluster/feed_queue.sh PLAN` | runs the submit commands of a plan file (one argument list per line, e.g. `cluster/plan_071026.txt`) in order, each as soon as the queue has room for all of its tasks (a line larger than the cap goes in parts, cell by cell, whenever its largest cell fits); holds a `--fill` line until no job of its wave is queued; logs to `logs/feed_queue.log`; start it with `nohup ... &` on the login node |
| one feeder at a time | a lock directory, `logs/feed_queue.lock`, holds its pid, and a second feeder exits with a message; while `squeue` fails a `--fill` line waits rather than proceeds; a `--fill` line whose jobs have ended without reducing the number of missing shards is given up on with a warning (its tasks fail every time: check `logs/camp_*.err` and `sacct -M serial`); a plan line whose dry run fails, for instance `--only` matching no cell, is skipped with the dry run's message |

```bash
cd ~/repo && git pull
bash cluster/submit_campaign.sh --wave 1 --list          # the cells: name, job tag, shards, time, arguments
bash cluster/submit_campaign.sh --wave 1 --dry-run       # the sbatch commands; submits nothing
bash cluster/submit_campaign.sh --wave 1 --n 1000        # one sample size: 128 tasks
bash cluster/submit_campaign.sh --wave 1 --status        # complete / k of N shards written / not started
bash cluster/submit_campaign.sh --wave 1 --fill          # resubmit missing shards, once the jobs have ended
bash cluster/submit_campaign.sh --wave 1 --only MCP-up --n inf    # a subset of cells
bash cluster/submit_campaign.sh --wave 4 --p 15 25                 # wave 4: by p, n = 1000
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only -ml        # wave 5a at one n; --only search100 for 5b
bash cluster/submit_campaign.sh --wave 9 --group class --n 1000 1e4 # one group of wave 9 (a knob cell at n = inf is skipped)
```

```bash
# back on the laptop
rsync -av $USER@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/     # incremental; rerun as cells finish
python simulations/diagnostics/campaign.py --check-baseline   # -> runs/campaign/campaign_per_dataset.csv, campaign_means.csv, campaign_paired.csv;
                                                              #    compares the three cells that repeat the n-sweep (direct loss, C2I, standard paths)
                                                              #    with runs/nsweep_p10-20 graph by graph
python simulations/diagnostics/plot_campaign.py               # the figures of simulations/S4_campaign.md -> runs/campaign/figures/ (needs matplotlib)
python simulations/diagnostics/restarts.py                    # the search cells (waves 2 and 5b): the best of the first r randomly drawn starting
                                                              #    graphs, for every r -> campaign_restarts.csv
python3 simulations/diagnostics/plot_campaign.py --rule ebic1 # the rule-dependent figures with Dettling's eBIC penalty (gamma = 1) in the selection
                                                              #    rule and, where wave 5c or 6 ran it, inside the search -> runs/campaign/figures_ebic1/
python3 simulations/diagnostics/plot_figure5.py               # Dettling's Figure 5 layout for every estimator: his four whole-path summaries (the largest
                                                              #    accuracy and F1 over the path, the areas under the ROC and precision-recall curves) over p
                                                              #    at n = 1e3, one curve per true C; the lasso under his pipeline with his digitised values
                                                              #    overlaid; and a grid of all estimators per true C -> runs/campaign/figures/figure5/
                                                              #    (max_acc is in the means and paired tables since 15 October 2026)
python3 simulations/diagnostics/figure5_table.py              # the same four summaries per estimator with the paired gain over the lasso on the same graphs,
                                                              #    per true C and pooled over the four settings -> campaign_figure5.csv; from that table
                                                              #    plot_figure5.py also draws, per C, his 2 x 2 layout with one curve per estimator
                                                              #    (figure5_estimators_<C>.png, the four true C pooled) and the gains over the lasso with
                                                              #    +- 2 standard errors (figure5_gain_<C>.png)
python3 simulations/diagnostics/figure5_table.py --c Cresc --n 1e4 --true-c C_Random_Full   # prints the table for one C and n
```

Graph classes, beta-min and 2-cycles, wave 9:

| script | what | writes |
|---|---|---|
| `python simulations/diagnostics/graph_class.py` | the graph-class analysis: the stored campaign and the wave 9 class cells | `campaign_graph_class.csv`, `campaign_generator.csv`, `campaign_two_cycles.csv` |
| `python simulations/diagnostics/betamin.py` | the beta-min knob: the wave 9 `_w<wmin>` cells only | `campaign_wmin.csv`, `campaign_wmin_orientation.csv`, `campaign_edge_outcomes.csv`, `campaign_wmin_bins.csv` |
| `python simulations/diagnostics/twocycles.py` | the share of 2-cycles: the `_s<share>` cells | `campaign_share*.csv` |
| `python3 simulations/diagnostics/plot_graph_class.py`, `python3 simulations/diagnostics/plot_betamin.py`, `python3 simulations/diagnostics/plot_twocycles.py` | draw the three analyses above | |
| `python3 simulations/diagnostics/plot_generator_knobs.py` | the illustrations of [`GRAPH_CLASSES_AND_BETAMIN.md`](GRAPH_CLASSES_AND_BETAMIN.md), the note that describes the two knobs from the generator to the figures | `docs/figures/` |

```bash
# a rehearsal without a cluster: cluster/local/sbatch stands in for sbatch and turns every array task into a small shell script
export LOCAL_TASKS=/tmp/tasks.txt CAMPAIGN_ROOT=runs/local/campaign_rehearsal
export LOCAL_EXTRA="--p 10 --reps 1"                      # 16 graphs per cell instead of 800
PATH="$PWD/cluster/local:$PATH" bash cluster/submit_campaign.sh --wave 1 --n 1000 --shards 1
xargs -P 6 -n 1 sh < "$LOCAL_TASKS"                       # run them, six at a time
bash cluster/submit_campaign.sh --wave 1 --n 1000 --status
python simulations/diagnostics/campaign.py --root "$CAMPAIGN_ROOT"
```

---

## 3. Aggregating

```bash
# cheap enough for a login node, with the environment of §2.2
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate
python simulations/aggregate_s1.py --in-dir runs/s1_dettling_reproduction/s1_shards   # the CSVs next to the shard folder, in the run directory;
                                                                                      # warns loudly if the dataset count falls short of the grid,
                                                                                      # which is how a silently failed array task gets caught
```

A complete run reports:

```
shards      : 64
datasets    : 11200  (expected 11200)
compute     : <sum of per-dataset runtimes> CPU-hours
```

| file | contents |
|---|---|
| `s1_per_dataset.csv` | one row per dataset: all four metrics under both scoring conventions, `lambda_max`, `n_true_edges`, runtime |
| `s1_summary.csv` | Figure 5 itself: mean ± standard error per $(p, C)$, averaged over $k$ and reps |
| `s1_curves.csv` | mean tpr/fpr/precision/nnz per $(p, C, \lambda)$, enough to redraw ROC and PR curves |
| compute | §5 estimates about 39 CPU-hours for the full grid; the cluster's figure depends on its nodes |
| metrics | recomputed from the stored counts by `metrics_from_counts`, a second implementation of Definitions G.4/G.5; `test_aggregator_metrics_match_evaluate_path` asserts it agrees with `gclm.metrics.evaluate_path` to 1e-12, so the published numbers do not depend on which code path produced them |

---

## 4. Plotting locally

```bash
# on your own machine, from the repository root, in whatever Python environment you develop in (no module system here);
# plotting is the only step that needs matplotlib, which is why the cluster venv does not have it
mkdir -p runs
scp -r $USER@cool.hpc.lrz.de:~/repo/runs/s1_dettling_reproduction runs/   # the shards are not needed for plotting, only to re-aggregate locally
python -m pip install matplotlib      # once
python simulations/plot_figures.py --results runs/s1_dettling_reproduction   # -> runs/s1_dettling_reproduction/figures/figure3_reproduction.{png,pdf}
                                                                             #    and figure5_reproduction.{png,pdf}, next to the numbers they are made from
```

| design choice | detail |
|---|---|
| four categorical hues | blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, violet `#4a3aa7`, chosen by running a colour validator rather than by eye |
| the rejected set | blue/orange/aqua/yellow: orange and yellow sit at ΔE 13.7 in normal vision, below the 15 threshold at which full-colour readers can still separate them |
| the chosen set passes | the lightness band, the chroma floor, colour-vision-deficiency separation (worst pair ΔE 9.2, deutan) and the normal-vision floor (worst pair ΔE 16.3) |
| marker shapes | aqua falls below 3:1 contrast against the chart surface, so every series also carries a distinct marker shape and identity never rests on colour alone; every plotted number ships as CSV beside the figure |
| error bars | the standard error of the mean over the 400 drift matrices behind each point (4 sparsity levels × 100 replicates), matching the Figure 5 caption |

---

## 5. Cost

| $p$ | mean s/dataset | CPU-h for the full grid at this $p$ |
|---|---|---|
| 10 | 1.3 | 0.6 |
| 15 | 3.8 | 1.7 |
| 20 | 6.3 | 2.8 |
| 25 | 6.2 | 2.8 |
| 30 | 11.1 | 4.9 |
| 40 | 22.5 | 10.0 |
| 50 | 35.6 | 15.8 |
| **total** | | **38.6** |
| measured on | an Apple M2 (8 cores), per 100-λ path at the production tolerance; cluster core-hours are roughly comparable to these figures, modulo the node's clock | |
| the full grid | 7 × 4 × 4 × 100 = 11,200 datasets | about 40 minutes on 64 cluster cores plus the long tail; about 5 hours on the 8-core laptop, which is why it goes to the cluster |

---

## 6. Reproducing a single dataset locally

```python
# every dataset is addressable, so any outlier in the aggregate can be pulled apart on a laptop;
# this regenerates bit-for-bit what the cluster saw
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance

cfg = S1Config()
p, k, c, rep = 50, 3, CChoice.RANDOM_FULL, 17
rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c), rep])
m_true, c_true, sigma_true, sigma_hat = draw_instance(
    p, k, cfg.n_obs, c, rng, metzler=cfg.metzler, standardize=cfg.standardize)
```
