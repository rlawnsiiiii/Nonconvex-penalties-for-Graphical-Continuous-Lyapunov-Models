# Reproducing Dettling Figures 3 and 5

End-to-end recipe: what runs where, what comes back, and how the plots are made.

**Division of labour.** The cluster produces **numbers only** — no plots, no figures, no
matplotlib. It writes everything needed to reconstruct any downstream quantity, so a re-run is
never required just because a metric or a plot style changed. Plotting happens locally.

| stage | where | script | output |
|---|---|---|---|
| Figure 3 (Example 2) | cluster, one serial job | `simulations/run_m0.py` | `<run>/m0_reps100.csv` + `.npz` |
| Figure 5 (Section 5) | cluster, 64-task array | `simulations/run_s1_shard.py` | `<run>/s1_shards/shard_*.npz` |
| aggregation | cluster login node **or** local | `simulations/aggregate_s1.py` | `<run>/s1_*.csv` (3 tidy CSVs) |
| plots | **local only** | `simulations/plot_figures.py` | `<run>/figures/*.png`, `*.pdf` |

Everything a run produces lives in one folder, **`runs/<run name>/`**. The Dettling reproduction
is `runs/s1_dettling_reproduction/`:

```
runs/s1_dettling_reproduction/
  m0_reps100.csv, m0_reps100.npz       Figure 3 data
  s1_shards/shard_*_of_0064.npz        Figure 5 raw data (gitignored: large, reproducible)
  s1_summary.csv, s1_per_dataset.csv, s1_curves.csv    Figure 5 aggregated
  figures/                              the plots, made locally
```

---

## 1. What the cluster writes

The key design decision: shards store the **raw per-λ confusion counts**, not just the four
reported metrics. From `tp/fp/tn/fn` at each of the 100 λ values, every metric in Definitions
G.4/G.5 can be recomputed — including ones not yet defined, and including the
`include_diagonal` variant, which is stored in parallel. Nothing about the metric definitions is
baked in at compute time.

Per shard (`shard_NNNN_of_0064.npz`, compressed):

| array | shape | what |
|---|---|---|
| `p`, `k`, `c_choice`, `rep` | (n,) | which cell of the grid |
| `lambdas` | (n, 100) | the λ grid actually used |
| `lambda_max` | (n,) | its right endpoint |
| `conf_offdiag` | (n, 100, 4) | tp, fp, tn, fn — **off-diagonal scoring** |
| `conf_incdiag` | (n, 100, 4) | the same, **including the diagonal** |
| `nnz` | (n, 100) | selected off-diagonal entries per λ |
| `objective` | (n, 100) | penalised objective value per λ |
| `n_true_edges` | (n,) | true off-diagonal support size |
| `m_true_{i,j,v}` | ragged | $M^\*$ as a sparse triple — exact edge weights |
| `m_best_f1_{i,j,v}` | ragged | $\hat M$ at the best-$F_1$ λ, sparse |
| `m_best_acc_{i,j,v}` | ragged | $\hat M$ at the best-accuracy λ, sparse |
| `best_f1_index`, `best_acc_index` | (n,) | which λ those were |
| `seconds` | (n,) | per-dataset runtime |
| `config_json` | scalar | the complete `S1Config` — the run is self-describing |
| `provenance_json` | scalar | host, Python/NumPy versions, platform, wall time, UTC finish |

Not stored: $\hat\Sigma$ and the 100 full $\hat M$ matrices per dataset — at $p=50$ that would be
~22 GB. They are exactly reproducible from `config_json` plus `(p, k, c_choice, rep)`, because the
per-task RNG is seeded as `default_rng([seed, p, k, c_index, rep])` — independent of worker count
and scheduling order.

Total output: roughly **100 MB** for the whole grid.

---

## 2. Running it on the LRZ Linux Cluster

Scripts follow the LRZ serial-job template: `--clusters=serial`, `--partition=serial_std`,
`--export=NONE` with `--get-user-env`, and the mandatory `module load slurm_setup`.
Job names are kept under 10 characters as LRZ asks.

Two environments are involved, and they are set up differently:

| where | how Python is provided |
|---|---|
| **your login shell** (setup, smoke test, aggregation) | you run `module load python` and `source ~/venvs/gclm/bin/activate` yourself, **in every new session** |
| **a batch job** (`sbatch`) | the `.sbatch` script does both itself — see §2.4 |

### 2.1 One-time setup

```bash
ssh -Y $USER@cool.hpc.lrz.de          # 2FA; see the LRZ access documentation
git clone <this repo> ~/repo
cd ~/repo
```

Then build the virtual environment. The script does it in one go:

```bash
bash cluster/setup_env.sh
```

It runs the following, which you can also type by hand:

```bash
module load python                    # the Python the venv is built on
python3 -m venv ~/venvs/gclm
source ~/venvs/gclm/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.24" "scipy>=1.10"
```

Only NumPy and SciPy are needed: the default solver (`fista`) is pure Python, so **no R, glmnet,
ncvreg or skglm is required on the cluster**.

**Why `module load python` comes first.** A venv does not contain its own copy of Python; it
points back at the interpreter it was created with. That interpreter, and the libraries it needs,
are only on the path while its module is loaded. So:

- load the **same** Python module every time you use the venv, interactively or in a job;
- if `module load python` fails, run `module avail python`, pick one (e.g. `module load python/3.x`),
  and use that name in `cluster/setup_env.sh` **and** both `.sbatch` files;
- if the cluster's default Python changes, or you switch versions, recreate the venv:
  `rm -rf ~/venvs/gclm && bash cluster/setup_env.sh`. Pinning an explicit version in all three
  scripts avoids being surprised by a changed default.

The venv lives at `~/venvs/gclm`. The batch scripts use exactly that path; if you move it, change
`VENV=` in `cluster/setup_env.sh`, `cluster/s1_array.sbatch` and `cluster/m0.sbatch`.

### 2.2 Every new login session

A fresh SSH session has neither the module nor the venv. Before running anything by hand:

```bash
ssh -Y $USER@cool.hpc.lrz.de
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate
```

Check that the venv is the one in use:

```bash
which python                          # should print ~/venvs/gclm/bin/python
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

`deactivate` leaves the venv again.

### 2.3 Smoke test before submitting

About a minute on the login node, with the environment of §2.2 active. It exercises the same code
path as the real array task, on a single dataset:

```bash
python simulations/run_s1_shard.py --shard 0 --n-shards 400 --p 10 --reps 1 --out-dir /tmp/gclm_smoke
python simulations/aggregate_s1.py --in-dir /tmp/gclm_smoke --out-dir /tmp/gclm_smoke
```

The first command should end with `done: 1 datasets ...`. The aggregator will warn that datasets
are missing; that is expected here, since only one dataset of the grid was run.

Optionally, the test suite. The tests that need R, skglm, pyproximal or cvxpy skip themselves, so
NumPy, SciPy and pytest are enough:

```bash
python -m pip install pytest
python -m pytest -m "not r" -q
```

### 2.4 Submit

Nothing needs to be activated before `sbatch`. Because of `--export=NONE`, a job starts with a
clean environment and inherits **nothing** from your shell: no loaded modules, no active venv.
Each `.sbatch` script therefore sets up its own environment. After `module load slurm_setup` it runs

```bash
module load python
source ~/venvs/gclm/bin/activate
```

and prints the interpreter it ended up with into its log, so a wrong environment shows up there.
From `~/repo`:

```bash
mkdir -p logs
sbatch cluster/m0.sbatch              # Figure 3   — one serial job, ~5 min
sbatch cluster/s1_array.sbatch        # Figure 5   — array 0-63

Both write to `runs/s1_dettling_reproduction/` by default. For any other run, give it its own
folder as the first argument; further arguments to `s1_array.sbatch` go straight to
`run_s1_shard.py`. Without a separate folder a new run would overwrite the shards of the old one,
since the shard file names are the same:

```bash
sbatch cluster/s1_array.sbatch runs/s1b_mcp_textbook --penalty MCP
sbatch cluster/s1_array.sbatch runs/s1b_scad_textbook --penalty SCAD
```

(Arguments rather than environment variables: with `--export=NONE` a job inherits no
environment, but it does receive its arguments.)

Study S2 — the same grid on Varando's losses (docs/LIKELIHOOD.md) — is six more submissions, one
per (loss, penalty), each in its own folder:

```bash
for loss in loglik frobenius; do
  for pen in lasso MCP SCAD; do
    sbatch cluster/s1_array.sbatch "runs/s2_${loss}_${pen}" --loss "$loss" --penalty "$pen"
  done
done
```

The output layout is identical, so `aggregate_s1.py` and `plot_penalties.py` apply unchanged.
Costs per loss and penalty are in simulations/S2_penalties_losses.md §4; the S2 fits are heavier
than the direct loss, so check that table against the array's `--time` before submitting.
```

Check progress:

```bash
squeue -M serial -u $USER
sacct  -M serial -j <jobid> --format=JobID,State,Elapsed,MaxRSS
ls runs/s1_dettling_reproduction/s1_shards/*.npz | wc -l      # expect 64 when complete
head -3 logs/s1_<jobid>_0.out         # host, shard, and the python actually used
```

> **Verify array support before relying on it.** The LRZ pages consulted document serial jobs and
> `--clusters/--partition` but do not discuss `--array` limits. If arrays are restricted on your
> segment, submit the shards as independent jobs instead — the runner takes `--shard i --n-shards N`
> and needs nothing else. Note that each wrapped job has to set up the environment itself, for the
> same reason as above:
> ```bash
> for i in $(seq 0 63); do
>   sbatch --clusters=serial --partition=serial_std --time=06:00:00 \
>          --ntasks=1 --mem=2G -J gclm-s$i --export=NONE --get-user-env \
>          --wrap "module load slurm_setup; module load python; \
>                  source \$HOME/venvs/gclm/bin/activate; \
>                  python simulations/run_s1_shard.py --shard $i --n-shards 64 \
>                         --out-dir runs/s1_dettling_reproduction/s1_shards"
> done
> ```

### 2.5 Resource choices, and why

- **`--cpus-per-task=1`** — each array task is single-core; parallelism comes from the array.
- **`--mem=2G`** — one worker peaks near 70 MB even at $p=50$; the solver is matrix-free, so
  nothing scales with $p^2$. 2G is generous.
- **`--time=06:00:00`** — the slowest shard should finish well inside 2 h. The margin covers the
  long tail (a few $p=50$ datasets take ~200 s) and a busy node. `serial_std` caps the time: on
  2 October 2026 it refused 48 h and accepted 24 h; `sinfo -M serial -p serial_std -o "%P %l"`
  shows the current cap.
- **64 shards** — datasets are assigned stride-wise (`tasks[i::64]`), so every shard gets a mix of
  $p$ values and they finish at roughly the same time. No straggler shard of pure $p=50$.

### 2.6 The n-sweep: $p = 10, 20$ at growing $n$, every loss and penalty

Figure 5's data-generating process at $p \in \{10, 20\}$ (4 $k$ × 4 $C$ × 25 reps = 800 datasets)
for every loss (direct, log-likelihood, Frobenius), penalty (lasso, MCP, SCAD) and sample size
($n = 10^3, 10^4, 10^5, \infty$): 36 cells, one array each, in
`runs/nsweep_p10-20/<loss>_<penalty>_n<n>/`. `--n-obs inf` feeds the population covariance.
$M^*$ and $C$ are drawn before the data, so all 36 cells see the same 800 drift matrices and every
comparison, across penalties, losses or $n$, is paired (`test_drift_and_volatility_do_not_depend_on_n`).

`cluster/submit_nsweep.sh` does the bookkeeping: it writes each cell's shard count into its folder
(`n_shards`, which `s1_array.sbatch` reads), never submits a cell twice, and resubmits only the
missing shards when asked:

```bash
cd ~/repo && git pull
bash cluster/submit_nsweep.sh --dry-run            # the 36 sbatch commands; submits nothing
bash cluster/submit_nsweep.sh --status             # complete / k of N shards written / not started
```

The direct loss at $n = 1000$ needs no rerun: those datasets are $p = 10, 20$, reps < 25 of the
full lasso run (`runs/s1_dettling_reproduction`) and of the S1b pilot (MCP, SCAD), so its three
cells stay "not started". The order used on 2 October:

1. **Canary: the direct loss at $n = \infty$** (48 tasks, under an hour), a cheap first run of the
   new code path (`--n-obs`, `n_shards`, the array override) that produces new results:
   ```bash
   bash cluster/submit_nsweep.sh --n inf --loss direct
   ```
   When `--status` shows its three cells complete and `logs/*.err` are empty, aggregate them and
   compare $p = 10$, `C_ID` and `C_Random_Diag` with the local pilot of that day
   (`next_steps/021026/files/s1_nsweep_p10.csv`); they must agree exactly.
2. **Everything else in one go** (1,248 tasks):
   ```bash
   bash cluster/submit_nsweep.sh --n 1e4 1e5 inf                    # skips the canary cells
   bash cluster/submit_nsweep.sh --n 1000 --loss loglik frobenius
   ```
   The covariance-loss cells at $n = 1000$, $p = 10$, reps < 10 repeat the S2 pilot
   (`runs/s2_pilot_p10`) and should agree with it.

The shard counts and limits below are estimates; `sacct -M serial -X -u $USER -S now-2days
--format=JobName%10,JobID%18,State,Elapsed` shows how long each shard took. If a loss needs more
time or more shards, edit `shards_for` / `time_for` at the top of the script before submitting more
of it (a cell already submitted keeps the shard count it recorded).

**LRZ limits (measured 2 October 2026).** At most 96 of a user's tasks run at once; further ones
wait as `PD (QOSMaxCpuPerUserLimit)` and start by themselves. About 200 tasks may be queued or
running; beyond that `sbatch` refuses with `AssocMaxSubmitJobLimit`. At the default shard counts
the sweep is far more tasks than that, so give the cells not yet submitted fewer, longer tasks
with `--shards`, for example 4 (direct), 8 (log-likelihood) and 16 (Frobenius), and submit them in rounds as the queue
empties.

If `sbatch` refuses with a per-user limit, submit fewer cells at a
time; a refused cell records nothing, so rerunning the same command continues where it stopped.
Once a cell's jobs have ended, `--status` shows shards that were never written (TIMEOUT, node
failure: check `sacct`), and `--fill` resubmits exactly those:

```bash
bash cluster/submit_nsweep.sh --fill               # after a TIMEOUT add --time, up to the partition's cap
```

| loss | shards (default) | `--time` | CPU-h per cell: lasso / MCP / SCAD (measured) | longest shard |
|---|---|---|---|---|
| direct | 16 | 6 h | 1.4 / 6.2 / 8.4 | 2.3 h (4 shards) |
| log-likelihood | 32 | 24 h | 24 / 6.8 / 6.8 | 3.5 h (8 shards) |
| Frobenius | 64 | 24 h | 52 / 37 / 37 | 4.9 h (16 shards) |

Measured on the run of 2–3 October 2026 (simulations/S2b_nsweep.md §4): 180 CPU-h per sample size
and 720 for all four, 70 % of it on the Frobenius loss. The sample size does not change the cost.
With `--shards 4 / 8 / 16` (direct / log-likelihood / Frobenius) no task ran longer than 5 h, so
those counts fit the 24 h limit with a wide margin and keep a whole sweep under the per-user job
limit. Wall-clock depends on how many tasks LRZ runs at once (96 on that run).

**Aggregate and compare** (§3, §4; all 36 cells with their shards are about 120 MB). The analysis
across cells is `simulations/diagnostics/nsweep.py` and `plot_nsweep.py`; results in
simulations/S2b_nsweep.md.

```bash
for d in runs/nsweep_p10-20/*/; do python simulations/aggregate_s1.py --in-dir "$d/s1_shards"; done
scp -r $USER@cool.hpc.lrz.de:~/repo/runs/nsweep_p10-20 runs/
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

The plan, the reasons and the code map are in
[`../next_steps/051026/cluster_campaign_051026.md`](../next_steps/051026/cluster_campaign_051026.md);
the estimators are defined in [`DENSE_START.md`](DENSE_START.md) and [`SEARCH.md`](SEARCH.md).
Same graphs as the n-sweep (same seeds), so everything is paired with `runs/nsweep_p10-20` too.

| wave | cells per $n$ | tasks per $n$ | what |
|---|---|---|---|
| 1 | 16 | 128 | direct loss, $p = 10, 20$: lasso, MCP / SCAD (standard, dense → sparse, LLA), adaptive lasso; each with $C = 2I$ (`C2I`) and the rescaled $C$ (`Cresc`); path, graph selected by the score, graph after the greedy search on the score |
| 2 | 3 | 20 | search from random graphs and search started from the truth: $p = 10$ (`C2I`, `Cresc`), $p = 20$ (`Cresc`) |
| 3 | 8 | 56 | log-likelihood loss, $p = 10$: lasso and MCP in both path orders, `C2I` and `Cresc`; path and the graph selected by the score |
| 4 | 12 per $p$ | 112 – 224 per $p$ | larger $p$ at $n = 1000$ ($p = 15, 25, 30, 40, 50$): lasso, MCP, SCAD, MCP / SCAD dense → sparse, adaptive lasso, `C2I` and `Cresc`; path and the graph selected by the score, no search. Selected with `--p`, not `--n`; cells `..._p<p>_n1000` |
| 7 | 14 | 244 | the log-likelihood loss over $p = 10, 20$ ($p = 30$ left out for cost), both $C$: lasso, MCP sparse → dense, MCP dense → sparse from the exact fit and from the lasso solution (`--up-start lasso`), adaptive lasso (`--method adaptive` on `--loss loglik`); selection and search with the least-squares refit; at $p = 10$ only the two estimators wave 3 lacks; cells `loglik_<estimator>_<C>[_p20]` |
| 8 | 30 | 604 (556 at $n = 10^4$, where the six direct-loss cells are wave 5a's) | the likelihood refit: the stored paths of waves 1, 3 and 7 scored again with the model on every graph fitted by maximum likelihood (`rescore_shard.py --refit loglik`), selection and search at $p = 10$, the selection only at $p = 20$, cells `<loss>_<estimator>-ml_<C>[_p20]`; the search from 100 random graphs, the empty graph and the truth with the likelihood refit, $p = 10$, 2 replicates, the starts of a graph in 4 tasks (`--start-blocks 4`; `search100sml_p10_<C>`). Plan file `cluster/plan_101026.txt` (waves 7 and 8 at $n = 10^4$ and 1000; wave 8 after wave 7) |
| 6 | 16 | 128 | every wave 1 cell rescored with the eBIC penalty inside the selection and the search, from its stored supports (`simulations/rescore_shard.py`; no path recomputed); cells `rescore1_<source cell>`, overlaid on the source rows by `campaign.py` |
| 5 | 6 + 3 + 6 + 3 | 48 + 48 (+ 32) + 64 + 20 | three checks of the selection step: (a) at $p = 10$ the score with the maximised likelihood (`--refit loglik`) for the lasso, MCP dense → sparse and the adaptive lasso, `C2I` and `Cresc`, with the search (cells `direct_<estimator>-ml_<C>`); (b) the pure search from 100 randomly drawn starting graphs, sparse and uniform (`search100s_p10_Cresc`, `search100u_p10_Cresc`), and 30 at $p = 20$ on 5 replicates (`search30s_p20_Cresc`); `simulations/diagnostics/restarts.py` reads off the best of the first $r$ starts; (c) the eBIC term inside the selection and the search (`--ebic-gamma 0.5 1`, cells `direct_<estimator>-ebic_<C>`, $p = 10, 20$) and in the pure search (`searche1_p<p>_<C>`) |

`cluster/submit_campaign.sh` does the bookkeeping exactly as `submit_nsweep.sh` does (one
submission per cell, `--status`, `--fill`); a cell is a folder
`runs/campaign/<cell>_n<n>/` with its shards in `shards/`. `cluster/campaign_array.sbatch` is the
job script; its first argument is the runner (`simulations/run_s1_shard.py` or
`simulations/run_search_shard.py`).

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
```

LRZ accepts about 200 queued or running tasks per user and runs 96 at a time. When `sbatch`
refuses a cell, the script records nothing for it and stops; the same command, run again later,
continues there. Check the room with `squeue -M serial -u $USER -h -r | wc -l`, or let
`cluster/feed_queue.sh PLAN` do it: it runs the submit commands of a plan file (one argument list
per line, e.g. `cluster/plan_071026.txt`) in order, each as soon as the queue has room for all of
its tasks (a line larger than the cap goes in parts, cell by cell, whenever its largest cell
fits), holds a `--fill` line until no job of its wave is queued, and logs to
`logs/feed_queue.log`; start it with `nohup ... &` on the login node.

Back on the laptop:

```bash
rsync -av $USER@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/     # incremental; rerun as cells finish
python simulations/diagnostics/campaign.py --check-baseline
```

writes `campaign_per_dataset.csv`, `campaign_means.csv` and `campaign_paired.csv` into
`runs/campaign/`, and compares the three cells that repeat the n-sweep (direct loss, `C2I`,
standard paths) with `runs/nsweep_p10-20` graph by graph. `python simulations/diagnostics/plot_campaign.py`
(needs matplotlib) draws the figures of `simulations/S4_campaign.md` into `runs/campaign/figures/`.
`python simulations/diagnostics/restarts.py` reads the search cells (waves 2 and 5b) and tabulates
the best of the first $r$ randomly drawn starting graphs for every $r$ (`campaign_restarts.csv`).
`python3 simulations/diagnostics/plot_campaign.py --rule ebic1` redraws the rule-dependent figures
with Dettling's eBIC penalty ($\gamma = 1$) in the selection rule and, where wave 5c or 6 ran it,
inside the search, into `runs/campaign/figures_ebic1/`.

**A rehearsal without a cluster.** `cluster/local/sbatch` is a stand-in for `sbatch` that turns
every array task into a small shell script instead of submitting it:

```bash
export LOCAL_TASKS=/tmp/tasks.txt CAMPAIGN_ROOT=runs/local/campaign_rehearsal
export LOCAL_EXTRA="--p 10 --reps 1"                      # 16 graphs per cell instead of 800
PATH="$PWD/cluster/local:$PATH" bash cluster/submit_campaign.sh --wave 1 --n 1000 --shards 1
xargs -P 6 -n 1 sh < "$LOCAL_TASKS"                       # run them, six at a time
bash cluster/submit_campaign.sh --wave 1 --n 1000 --status
python simulations/diagnostics/campaign.py --root "$CAMPAIGN_ROOT"
```

---

## 3. Aggregating

Cheap enough for a login node. With the environment of §2.2:

```bash
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate
python simulations/aggregate_s1.py --in-dir runs/s1_dettling_reproduction/s1_shards
```

The CSVs are written next to the shard folder, in the run directory. It warns loudly if the
dataset count falls short of the grid, which is how a silently failed array
task gets caught. A complete run reports:

```
shards      : 64
datasets    : 11200  (expected 11200)
compute     : <sum of per-dataset runtimes> CPU-hours
```

(§5 estimates about 39 CPU-hours for the full grid; the cluster's figure depends on its nodes.)

Produces:

| file | contents |
|---|---|
| `s1_per_dataset.csv` | one row per dataset: all four metrics under **both** scoring conventions, `lambda_max`, `n_true_edges`, runtime |
| `s1_summary.csv` | **Figure 5 itself** — mean ± standard error per $(p, C)$, averaged over $k$ and reps |
| `s1_curves.csv` | mean tpr/fpr/precision/nnz per $(p, C, \lambda)$ — enough to redraw ROC and PR curves |

The metrics are recomputed here from the stored counts by `metrics_from_counts`, which is a second
implementation of Definitions G.4/G.5. `test_aggregator_metrics_match_evaluate_path` asserts it
agrees with `gclm.metrics.evaluate_path` to 1e-12, so the published numbers do not depend on which
code path produced them.

---

## 4. Plotting — locally

On your own machine, from the repository root. There is no module system here — use whatever
Python environment you develop in. Plotting is the only step that needs matplotlib, which is why the
cluster venv does not have it:

```bash
mkdir -p runs
scp -r $USER@cool.hpc.lrz.de:~/repo/runs/s1_dettling_reproduction runs/
python -m pip install matplotlib      # once
python simulations/plot_figures.py --results runs/s1_dettling_reproduction
```

Writes `runs/s1_dettling_reproduction/figures/figure3_reproduction.{png,pdf}` and
`.../figure5_reproduction.{png,pdf}`, next to the numbers they are made from. The shards are not
needed for plotting; copying them is only necessary to re-aggregate locally.

**Design choices.** Four categorical hues — blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`,
violet `#4a3aa7` — chosen by running a colour validator rather than by eye. The obvious
blue/orange/aqua/**yellow** set fails: orange and yellow sit at ΔE 13.7 in normal vision, below the
15 threshold at which full-colour readers can still separate them. The chosen set passes the
lightness band, chroma floor, colour-vision-deficiency separation (worst pair ΔE 9.2, deutan) and
the normal-vision floor (worst pair ΔE 16.3).

Aqua falls below 3:1 contrast against the chart surface, which obliges relief: every series also
carries a **distinct marker shape**, so identity never rests on colour alone, and every plotted
number ships as CSV beside the figure.

Error bars are the standard error of the mean over the 400 drift matrices behind each point
(4 sparsity levels × 100 replicates), matching the Figure 5 caption.

---

## 5. Cost

Measured on an Apple M2 (8 cores), per 100-λ path at the production tolerance:

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

The full grid is 7 × 4 × 4 × 100 = **11,200 datasets**. On 64 cluster cores that is about 40
minutes plus the long tail; the same job on the 8-core laptop is about 5 hours, which is why it
goes to the cluster.

Cluster core-hours are roughly comparable to the laptop figures above, modulo the node's clock.

---

## 6. Reproducing a single dataset locally

Every dataset is addressable, so any outlier in the aggregate can be pulled apart on a laptop:

```python
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance

cfg = S1Config()
p, k, c, rep = 50, 3, CChoice.RANDOM_FULL, 17
rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c), rep])
m_true, c_true, sigma_true, sigma_hat = draw_instance(
    p, k, cfg.n_obs, c, rng, metzler=cfg.metzler, standardize=cfg.standardize)
```

This regenerates bit-for-bit what the cluster saw.
