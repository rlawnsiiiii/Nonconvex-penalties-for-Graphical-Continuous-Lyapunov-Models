# Cluster commands for the campaign (5 October 2026)

*The commands only. The plan, the reasons and the code map are in
[`cluster_campaign_051026.md`](cluster_campaign_051026.md) (§3 the waves, §4 where the code is).
Work through the blocks from top to bottom; what to expect is written under each block.*

**What gets submitted**

| wave | what | cells per $n$ | tasks per $n$ |
|---|---|---|---|
| 1 | direct loss, $p = 10, 20$: lasso; MCP / SCAD (standard, dense → sparse, LLA); adaptive lasso; each with $C = 2I$ and the rescaled $C$; path, BIC-selected graph, graph after the BIC search | 16 | 128 |
| 2 | search without a penalty, and search started from the true graph | 3 | 20 |
| 3 | log-likelihood loss, $p = 10$: lasso and MCP, both path orders, both $C$ | 8 | 56 |

Sample sizes: $n = 1000$, $10^4$, $\infty$. LRZ runs 96 of your tasks at a time and accepts about
200 queued or running.

---

## 1. Laptop: commit and push

```bash
cd ~/Desktop/MastersThesis/repo
git add .gitignore README.md ARCHITECTURE.md docs src cluster tests \
        simulations/run_s1.py simulations/run_s1_shard.py simulations/run_search_shard.py \
        simulations/diagnostics
git status --short | grep -v '^??'
git commit -m "campaign: rescaled C, dense-to-sparse, LLA, adaptive lasso, BIC selection and search on the cluster"
git push
```

- The `git add` stages 30 files: library, runners, cluster scripts, tests, documentation.
- It leaves out `runs/` and `next_steps/`. Commit those separately if you want them.
- What you staged earlier (the independent study's README, figures, tables) goes into the same
  commit unless you unstage it first (`git restore --staged <path>`).

## 2. LRZ: every login

```bash
ssh -Y ge47xod3@cool.hpc.lrz.de
cd ~/repo
module load python
source ~/venvs/gclm/bin/activate
```

- `sbatch` needs nothing activated. `python` needs the module and the venv.

## 3. LRZ: update and check (about 3 minutes)

```bash
git pull
python -m pytest -q tests/test_campaign_runner.py tests/test_campaign_submit.py \
    tests/test_search_shard.py tests/test_lla_adaptive.py tests/test_direction_cscale.py
bash cluster/submit_campaign.sh --wave 1 --list
bash cluster/submit_campaign.sh --wave 1 --dry-run | tail -2
```

- If `git pull` is refused because of local edits: `git stash && git pull && git stash pop`.
- The tests should end with `43 passed`. Nothing is submitted by them.
- `--list` prints the 16 cells of wave 1 with their arguments.
- The dry run should end with `would submit 384 tasks in 48 cells`.
- **If anything here differs, stop and send me the output.**

## 4. LRZ: a canary of 4 small tasks

The job script `cluster/campaign_array.sbatch` is new, so four small tasks go first.

```bash
bash cluster/submit_campaign.sh --wave 2 --n 1000 --only p10
squeue -M serial -u $USER
```

After two or three minutes:

```bash
tail -n 4 logs/camp_*.out
cat logs/camp_*.err
```

- Each `.out` file should show:
  - a line `python=...`;
  - a line `host=... runner=simulations/run_search_shard.py ...`;
  - a line `shard 0/2: 200 datasets [...]` (or `shard 1/2`);
  - soon after, a first progress line `10/200 ...`.
- The `.err` files should be empty. If a task failed, send me its `.err`.
- These four tasks belong to wave 2. They are not thrown away. Each takes 20 to 40 minutes.

## 5. LRZ: the first round (128 + 56 tasks)

Once the canary shows progress lines:

```bash
bash cluster/submit_campaign.sh --wave 1 --n 1000
bash cluster/submit_campaign.sh --wave 2
```

- With the canary that is 188 tasks, just under the limit.
- If `sbatch` refuses a cell (`AssocMaxSubmitJobLimit`), the script stops, says how many tasks
  went in, and records nothing for the refused cell. Run the same command again later; it skips
  what is already submitted.

## 6. LRZ: the next rounds, each once the queue has room

```bash
squeue -M serial -u $USER -h -r | wc -l                # tasks queued or running right now
```

| command | tasks | run it when the count is below about |
|---|---|---|
| `bash cluster/submit_campaign.sh --wave 1 --n 1e4` | 128 | 70 |
| `bash cluster/submit_campaign.sh --wave 1 --n inf` | 128 | 70 |
| `bash cluster/submit_campaign.sh --wave 3` | 168 | 30 |

- Wave 3 can also go in three pieces of 56 tasks: `--wave 3 --n 1000`, then `--n 1e4`, then
  `--n inf`.

## 7. LRZ: watch

```bash
squeue -M serial -u $USER
bash cluster/submit_campaign.sh --wave 1 --status
bash cluster/submit_campaign.sh --wave 2 --status
bash cluster/submit_campaign.sh --wave 3 --status
sacct -M serial -X -u $USER -S now-2days --format=JobName%10,JobID%18,State,Elapsed | grep -v COMPLETED
less logs/camp_<jobid>_<task>.err
```

- `squeue`: `PD` waiting, `R` running. `PD (QOSMaxCpuPerUserLimit)` is normal: 96 run at a time.
- `--status` prints one line per cell: `complete`, `submitted (k/N shards written)` or
  `not started`.
- The `sacct` line lists jobs that failed or ran out of time. Nothing listed is good.
- Job names: wave, estimator, $C$, sample size. For example `1Mur3` is wave 1, MCP dense → sparse
  (`u`), rescaled $C$ (`r`), $n = 10^3$ (`3`; `4` is $10^4$, `i` is $\infty$). `--list` shows the
  tag of every cell.

## 8. LRZ: repair, only after every job of that wave has ended

```bash
bash cluster/submit_campaign.sh --wave 1 --fill
bash cluster/submit_campaign.sh --wave 1 --fill --time 24:00:00       # after a TIMEOUT
scancel -M serial <jobid>                                              # to cancel a job
```

- `--fill` resubmits exactly the missing shards.
- Use it only when `squeue` shows no job of that wave any more. Otherwise shards that are still
  running are submitted twice.
- The same works for `--wave 2` and `--wave 3`.

## 9. Laptop: bring the results home

Once `--status` reports every cell of a wave `complete` (or earlier, for a first look at what is
finished):

```bash
cd ~/Desktop/MastersThesis/repo
scp -r ge47xod3@cool.hpc.lrz.de:~/repo/runs/campaign runs/
python simulations/diagnostics/campaign.py --check-baseline
```

- About 350 MB with all three waves.
- `campaign.py` prints the table of means and writes `campaign_per_dataset.csv`,
  `campaign_means.csv` and `campaign_paired.csv` into `runs/campaign/`.
- `--check-baseline` compares the three cells that repeat the n-sweep (direct loss, $C = 2I$,
  standard paths) with `runs/nsweep_p10-20`, graph by graph. The lasso must agree exactly; MCP and
  SCAD may differ on a handful of graphs.
- Then tell me, and I do the analysis.

---

## How long

| round | tasks | CPU-h (estimate) | at 96 cores |
|---|---|---|---|
| wave 1, one sample size | 128 | 200 | 2 to 4 hours |
| wave 2, all three sample sizes | 60 | 170 | 2 to 4 hours |
| wave 3, all three sample sizes | 168 | 180 – 300 | 2 to 4 hours |

- Each task takes about 1 to 3 hours (limit 12 h; wave 3: 24 h).
- The first results (wave 1 at $n = 1000$) are back a few hours after block 5.
- Everything is back within about a day if the rounds follow each other.

## Submission log

*Fill in as you go: date, command, what `sbatch` answered.*

| when | command | result |
|---|---|---|
| 051026| bash cluster/submit_campaign.sh --wave 1 --n 1000, bash cluster/submit_campaign.sh --wave 2| |
| 061026| bash cluster/submit_campaign.sh --wave 1 --n 1e4| |
| 071026| bash cluster/submit_campaign.sh --wave 1 --n inf| |
