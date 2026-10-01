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
  long tail (a few $p=50$ datasets take ~200 s) and a busy node. `serial_std` allows up to 48 h.
- **64 shards** — datasets are assigned stride-wise (`tasks[i::64]`), so every shard gets a mix of
  $p$ values and they finish at roughly the same time. No straggler shard of pure $p=50$.

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
