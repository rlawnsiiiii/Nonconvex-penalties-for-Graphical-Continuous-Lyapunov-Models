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
| 4 | larger $p$ for the thesis figure: $p = 15, 25, 30, 40, 50$ at $n = 1000$; lasso, MCP, SCAD (standard), MCP / SCAD dense → sparse, adaptive lasso; both $C$; BIC but no search | 12 per $p$ | 112 ($p \le 30$), 224 ($p = 40, 50$) |

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
rsync -av ge47xod3@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/
python simulations/diagnostics/campaign.py --check-baseline
```

- `rsync` copies only what is new, so run it as often as you like: once wave 1 is complete,
  again when waves 2 – 4 come in. Cells still in progress come along with the shards they have
  so far and show fewer graphs in the tables; the next `rsync` completes them.
- About 350 MB for waves 1 – 3, roughly as much again for wave 4.
- `campaign.py` prints the table of means and writes `campaign_per_dataset.csv`,
  `campaign_means.csv` and `campaign_paired.csv` into `runs/campaign/`.
- `--check-baseline` compares the three cells that repeat the n-sweep (direct loss, $C = 2I$,
  standard paths) with `runs/nsweep_p10-20`, graph by graph. The lasso must agree exactly; MCP and
  SCAD may differ on a handful of graphs.
- Then tell me, and I do the analysis.

## 10. LRZ: wave 4 (larger $p$), one $p$ at a time

Wave 4 varies $p$ instead of $n$ ($n = 1000$ throughout), so it takes `--p` and ignores `--n`.
Cells are named `direct_<estimator>_<C>_p<p>_n1000`.

```bash
bash cluster/submit_campaign.sh --wave 4 --list              # the 60 cells
squeue -M serial -u $USER -h -r | wc -l                      # room: 200 minus this
bash cluster/submit_campaign.sh --wave 4 --p 15              # 112 tasks, short
bash cluster/submit_campaign.sh --wave 4 --p 25              # 112 tasks
bash cluster/submit_campaign.sh --wave 4 --p 30              # 112 tasks, 24 h limit
bash cluster/submit_campaign.sh --wave 4 --p 40              # 224 tasks: needs an almost empty queue
bash cluster/submit_campaign.sh --wave 4 --p 50              # 224 tasks
bash cluster/submit_campaign.sh --wave 4 --status
bash cluster/submit_campaign.sh --wave 4 --fill --time 24:00:00   # after the jobs of a p have ended
```

- Submit the next $p$ whenever the queue count allows it (each command needs room for all its
  tasks; if `sbatch` refuses part of the way, rerun the same command later).
- $p = 40$ and $50$ are the expensive ones: one $p = 50$ graph takes 30 s (lasso, adaptive
  lasso) to 5 to 15 minutes (MCP / SCAD, standard or dense → sparse). Their cells have twice
  the shards and 24 h; a task then runs half an hour to about three hours. About 700 CPU-h for
  all five $p$: expect a day.
- On the laptop afterwards: the same `rsync` and `campaign.py` as in block 9; the tables then
  show every $p$ from 10 to 50.
- Optional, if cluster time is left: the population version for a subset,
  `bash cluster/submit_campaign.sh --wave 4 --p 30 50 --n inf --only Cresc` (12 cells, about
  150 CPU-h; cells `..._p30_ninf`).

## 11. LRZ: wave 5 (two checks of the selection step), when the queue has room

Wave 5a repeats the selection and the search of three estimators with the BIC proper (the maximised
likelihood behind the score, `--refit loglik`); wave 5b runs the pure search from 100 randomly drawn
starting graphs instead of 10 (sparse ones as in wave 2, and uniform ones); wave 5c runs the
selection and the search with the extended BIC term ($\gamma = 0.5, 1$) next to the plain BIC on the
same path, and the pure search with $\gamma = 1$. Cells `direct_<estimator>-ml_<C>_n<n>`,
`search100s_p10_Cresc_n<n>`, `search100u_p10_Cresc_n<n>`, the optional `search30s_p20_Cresc_n<n>`,
`direct_<estimator>-ebic_<C>_n<n>` and `searche1_p<p>_<C>_n<n>`. Nothing of this is submitted yet;
the suggested first subset is the second, third, fourth and fifth command.

```bash
bash cluster/submit_campaign.sh --wave 5 --list                        # the 18 cells per n
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only -ml            # 5a at one n: 48 tasks
bash cluster/submit_campaign.sh --wave 5 --only search100              # 5b, p = 10, three n: 48 tasks
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only lasso-ebic adaptive-ebic   # 5c, the cheap paths: 32 tasks
bash cluster/submit_campaign.sh --wave 5 --only searche1_p10           # 5c pure search, p = 10, three n: 12 tasks
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only MCP-up-ebic    # 5c, the expensive paths: 32 tasks
bash cluster/submit_campaign.sh --wave 5 --n 1000 inf --only -ml       # the rest of 5a, if wanted: 96 tasks
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only search30s      # optional, p = 20 restarts: 32 tasks, 24 h
bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only searche1_p20   # optional, p = 20 pure search: 16 tasks
bash cluster/submit_campaign.sh --wave 5 --status
bash cluster/submit_campaign.sh --wave 5 --fill --time 24:00:00        # after the jobs of the wave have ended
```

- A task of 5a takes 1 to 3 hours (100 graphs at 15 to 100 s each on the laptop); of 5b at $p = 10$
  about an hour; of 5c as the matching wave 1 cell plus two more searches per graph, 2 to 4 hours.
- On the laptop afterwards: the `rsync` of block 9, then `campaign.py` as before (the `-ml`
  estimators appear as rows of their own next to the least-squares ones) and
  `python simulations/diagnostics/restarts.py`, which prints the $F_1$ and the share of graphs at
  the best score for the best of the first $r$ starting graphs, $r = 1 \dots 100$, and writes
  `runs/campaign/campaign_restarts.csv`.

---

## How long

| round | tasks | CPU-h (estimate) | at 96 cores |
|---|---|---|---|
| wave 1, one sample size | 128 | 200 | 2 to 4 hours |
| wave 2, all three sample sizes | 60 | 170 | 2 to 4 hours |
| wave 3, all three sample sizes | 168 | 180 – 300 | 2 to 4 hours |
| wave 4, all five $p$ | 784 | about 700 | 8 hours, over a day with queueing |
| wave 5a, one sample size | 48 | 90 – 180 | 1 to 3 hours |
| wave 5b at $p = 10$, three sample sizes | 48 | about 45 | about an hour |
| wave 5b at $p = 20$ (optional) | 32 | 50 – 100 | 2 to 3 hours |
| wave 5c, paths with three scores, one sample size | 64 | about 190 | 2 to 4 hours |
| wave 5c, pure search with $\gamma = 1$, three sample sizes | 60 | about 170 | 2 to 4 hours |

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
| 071026| bash cluster/submit_campaign.sh --wave 3| |
| 071026| bash cluster/submit_campaign.sh --wave 2 --fill --time 24:00:00| |
| 071026| bash cluster/submit_campaign.sh --wave 4 --p 15 | |
| 071026| bash cluster/submit_campaign.sh --wave 4 --p 25 | |
| 071026| bash cluster/submit_campaign.sh --wave 4 --p 30 | |
