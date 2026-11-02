# Generator knobs: E4 class, E3 beta-min, E5 2-cycles

Three generator knobs; estimators `lasso`, `MCP-up` (dense → sparse), `adaptive`. Status 19 Oct 2026: built,
tested, rehearsed; nothing submitted.

## 1. Files

```
src/gclm/data/simulate.py        sample_drift(p, d, rng, weights=, w_min=, graph=, two_cycle_share=)   the knobs
src/gclm/config.py               S1Config: weights, w_min, graph, two_cycle_share (defaults = thesis generator)
src/gclm/graphs.py               is_dag, two_cycles, graph_class, identified_on_support
simulations/run_s1_shard.py      --weights --w-min --graph --two-cycle-share --c-choices --select --ebic-gamma
         │   runs/campaign/<cell>/shards/*.npz   (M*, s, λ grid, every support, the seven graphs' counts)
         ▼
simulations/diagnostics/campaign.py        cell name → knob columns; class columns per graph; campaign_per_dataset.csv
   ├── graph_class.py  → plot_graph_class.py        E4   campaign_graph_class.csv, campaign_two_cycles.csv
   ├── betamin.py      → plot_betamin.py            E3   campaign_wmin*.csv, campaign_edge_outcomes.csv   (Knob machinery)
   └── twocycles.py    → plot_twocycles.py          E5   campaign_share*.csv
cluster/submit_campaign.sh  --wave 9 --group class | betamin | twocycle        cluster/plan_131026.txt
simulations/diagnostics/plot_generator_knobs.py → docs/figures/                 this note's illustrations
tests/  test_dgp.py  test_graphs.py  test_campaign_runner.py  test_campaign_submit.py  test_wave9_analysis.py  test_generator_knobs.py
```

## 2. Data-generating process

Per graph, seed `[20260922, p, k, C-index, rep]`, $d = k/p$:

$$M_{ij} = \omega_{ij} w_{ij}\ (i \ne j), \qquad M_{ii} = -\sum_{j \ne i}|M_{ij}| - |\varepsilon_{ii}| \quad\Rightarrow\quad \text{Gershgorin: } \operatorname{Re}\lambda(M) \le -|\varepsilon_{ii}| < 0,\ M \text{ stable}$$

$$C \in \lbrace 2I,\ \mathrm{diag}\ U[0.5,4],\ \mathrm{diag}\ U[2,4],\ \text{sparse symmetric, diagonally dominant}\rbrace \qquad (\text{E3, E5: } 2I \text{ only})$$

$$M\Sigma + \Sigma M^\top + C = 0 \ \Rightarrow\ \Sigma = \int_0^\infty e^{tM} C e^{tM^\top} dt \quad (\text{Bartels-Stewart, SciPy}); \qquad X_1, \dots, X_n \sim N(0, \Sigma),\ \hat\Sigma = X^\top X / n;\ n = \infty: \hat\Sigma = \Sigma$$

$$s_i = \hat\Sigma_{ii}^{1/2},\quad \hat R = D^{-1}\hat\Sigma D^{-1},\quad \tilde M_{ij} = M_{ij}\  s_j / s_i,\quad \tilde C = 2I\ (\text{C2I})\ \text{or}\ 2\ \mathrm{diag}(1/s_i^2)\ (\text{Cresc})$$

$$\lambda_j = \lambda_{\max}\ 10^{-4(1 - j/99)},\ j = 0..99 \ (\text{log-spaced: a constant factor } 10^{4/99});\qquad \lambda_{\max} = \max_{i \ne j}\bigl|[2(M_{\mathrm{diag}}\hat R + \hat R M_{\mathrm{diag}}^\top + \tilde C)\hat R]_{ij}\bigr|$$

No centring (mean zero); divisor $n$: Dettling eq. (1.3).

## 3. Knobs and expected edges

| knob | construction | $\mathbb{E}\ \lvert E \rvert$ |
|---|---|---|
| `random` | $\omega_{ij} \overset{\mathrm{iid}}{\sim} \mathrm{Bern}(d)$ | $p(p-1)d$; $P(\text{2-cycle}) = d^2$ |
| `dag` (E4) | $\pi$ random permutation; $\omega_{ij} \sim \mathrm{Bern}(\min(2d,1))\cdot\mathbf{1}[\pi(j) < \pi(i)]$ | $\binom{p}{2}\cdot 2d = p(p-1)d$ for $d \le 1/2$ |
| `pairs` (E5) | $P(\lbrace i,j\rbrace \text{ connected}) = q = \min(2d/(1+s), 1)$; $P(\text{2-cycle} \mid \text{connected}) = s$; else fair-coin direction | $\binom{p}{2}\ q\ (1+s) = p(p-1)d$; $s = d/(2-d)$ gives `random` in law |
| `uniform` (E3) | $\lvert w_{ij}\rvert \sim U[w_{\min}, 1]$, sign fair coin; supports unchanged | unchanged |

```python
    if graph == "dag":
        position = rng.permutation(p)                               # position[node] = pi(node); pi cannot increase
        allowed = position[None, :] < position[:, None]             # around a cycle, so the graph is acyclic
        omega = rng.binomial(1, min(2.0 * d, 1.0), size=(p, p)) * allowed   # half the ordered pairs are open: 2d
    elif graph == "pairs":
        q = min(2.0 * d / (1.0 + two_cycle_share), 1.0)             # a 2-cycle has two edges: q (1 + s) = 2d
        upper = np.triu(np.ones((p, p), dtype=bool), 1)
        connected = (rng.random(size=(p, p)) < q) & upper
        two = connected & (rng.random(size=(p, p)) < two_cycle_share)
        single_up = (connected & ~two) & (rng.random(size=(p, p)) < 0.5)
        single_down = (connected & ~two) & ~single_up
        omega = np.zeros((p, p), dtype=int)
        omega[two | two.T] = 1
        omega[single_up] = 1                                        # entry (i, j), i < j: the edge j -> i
        omega[single_down.T] = 1
    if weights == "uniform":                                        # after the two default draws: seeds untouched
        magnitude = rng.uniform(w_min, 1.0, size=(p, p))
        sign = np.where(rng.random(size=(p, p)) < 0.5, -1.0, 1.0)
        values = magnitude if metzler else sign * magnitude
```

![](figures/generator_classes.png)

![](figures/dag_construction.png)

![](figures/weight_laws.png)

## 4. Seven graphs per data set

$$\text{oracle: } \arg\max_\lambda F_1(S_\lambda, S^\ast)\qquad \text{selection: } \arg\min_\lambda\ \hat L(S_\lambda) + \log n\ (p + |S_\lambda|)\ [+\ 4\gamma|S_\lambda|\log p,\ \gamma \in \lbrace 0.5, 1\rbrace]\qquad \text{search: greedy single-edge moves from each selection}$$

$\hat L$ refitted loss; $n := 10^6$ at $n = \infty$; only the oracle sees $S^\ast$.

## 5. Cluster runs (wave 9)

All groups: $p \in \lbrace 10, 20\rbrace$ per cell, $n \in \lbrace 10^3, 10^4, \infty\rbrace$, `C2I` and `Cresc`, the seven graphs;
`--shards` 4 / 16 / 4 (`lasso` / `MCP-up` / `adaptive`), 16 and $24\ \mathrm{h}$ at $n = 10^3$.

| | `class` (E4) | `betamin` (E3) | `twocycle` (E5) |
|---|---|---|---|
| cells | `direct_<est>_<C>_dag_n<n>`: $3 \cdot 2 \cdot 1 \cdot 3 = 18$ | `direct_<est>_<C>_w<wmin>_n<n>`: $3 \cdot 2 \cdot 4 \cdot 3 = 72$ | `direct_<est>_<C>_s<share>_n<n>`: $3 \cdot 2 \cdot 5 \cdot 3 = 90$ |
| levels | `dag` | $w_{\min} \in \lbrace 0, 0.1, 0.25, 0.5\rbrace$; reference $N(0,1)$ | $s \in \lbrace 0, 0.25, 0.5, 0.75, 1\rbrace$; reference `random` |
| true C | all four | `C_ID` | `C_ID` |
| graphs per cell | $4k \cdot 4C \cdot 25 \cdot 2p = 800$ | $4k \cdot 25 \cdot 2p = 200$ | $200$ |
| runner flags | `--graph dag` | `--c-choices C_ID --weights uniform --w-min W` | `--c-choices C_ID --graph pairs --two-cycle-share S` |
| CPU-h | $\approx 105$ per $n$, $115$ at $\infty$ | $\approx 110$, $120$ at $\infty$ | $\approx 130$, $150$ at $\infty$ |
| analysis | `graph_class.py`, `plot_graph_class.py` | `betamin.py --c-choices C_ID`, `plot_betamin.py` | `twocycles.py --c-choices C_ID`, `plot_twocycles.py` |
| rehearsed | 14, 17 Oct: $p = 10$, 1 replicate | 19 Oct: $p = 10$, $2I$, 4 replicates | 19 Oct: $p = 10$, $2I$, 4 replicates |

### Commands, in order

```bash
# 1. Laptop: stage what the run and its analysis need, test, commit, push
cd ~/Desktop/MastersThesis/repo
git add src/gclm/objective/exact_fits.py src/gclm/solvers/path.py src/gclm/data/simulate.py src/gclm/config.py \
        simulations/run_s1_shard.py cluster/submit_campaign.sh cluster/plan_131026.txt \
        simulations/diagnostics/campaign.py simulations/diagnostics/betamin.py simulations/diagnostics/plot_betamin.py \
        simulations/diagnostics/twocycles.py simulations/diagnostics/plot_twocycles.py \
        simulations/diagnostics/graph_class.py simulations/diagnostics/plot_graph_class.py \
        simulations/diagnostics/starts.py simulations/diagnostics/plot_starts.py simulations/diagnostics/plot_generator_knobs.py \
        simulations/diagnostics/figure5_table.py simulations/diagnostics/plot_figure5.py \
        tests/test_dgp.py tests/test_campaign_runner.py tests/test_campaign_submit.py tests/test_campaign_analysis.py \
        tests/test_wave9_analysis.py tests/test_generator_knobs.py tests/test_figure5_table.py tests/test_plot_figure5.py \
        docs/GRAPH_CLASSES_AND_BETAMIN.md docs/figures docs/REPRODUCTION.md docs/DENSE_START.md docs/FISTA.md \
        docs/LIKELIHOOD.md docs/NONCONVEX.md docs/SEARCH.md README.md next_steps/131026/next_steps_131026.md
python -m pytest -q                      # 533 tests, about 15 min
git commit -m "E3 and E5 on the cluster: every reduction, true C = 2I, n = 1e3, 1e4, inf; the 2-cycle share; generator and cluster notes"
git push

# 2. LRZ: pull, check that the committed tree imports, dry-run
ssh ge47xod3@cool.hpc.lrz.de
cd ~/repo && git pull
module load python 2>/dev/null || module load anaconda3; source ~/venvs/gclm/bin/activate
PYTHONPATH=src python -c "import gclm.solvers.path, gclm.objective.exact_fits; print('imports ok')"   # the venv holds numpy + scipy only; the runners add src/ themselves
bash cluster/submit_campaign.sh --wave 9 --group betamin --n 1e4 --dry-run    # "would submit 192 tasks in 24 cells"
bash cluster/submit_campaign.sh --wave 9 --group twocycle --n 1e4 --dry-run   # "would submit 240 tasks in 30 cells"
bash cluster/submit_campaign.sh --wave 9 --group class --n 1e4 --dry-run      # "would submit 48 tasks in 6 cells"

# 3. LRZ: submit, one line at a time when the queue has room
bash cluster/submit_campaign.sh --wave 9 --group betamin --n 1e4
bash cluster/submit_campaign.sh --wave 9 --group betamin --n inf
bash cluster/submit_campaign.sh --wave 9 --group twocycle --n 1e4
bash cluster/submit_campaign.sh --wave 9 --group twocycle --n inf
bash cluster/submit_campaign.sh --wave 9 --group class --n 1e4
bash cluster/submit_campaign.sh --wave 9 --group class --n inf
bash cluster/submit_campaign.sh --wave 9 --group betamin --n 1000 --shards 16 --time 24:00:00
bash cluster/submit_campaign.sh --wave 9 --group twocycle --n 1000 --shards 16 --time 24:00:00
bash cluster/submit_campaign.sh --wave 9 --group class --n 1000 --shards 16 --time 24:00:00
#    or unattended, in the plan file's order:  nohup bash cluster/feed_queue.sh cluster/plan_131026.txt > logs/feed.out 2>&1 &

# 4. LRZ: look in on it
squeue -M serial -u $USER -h -r | wc -l
bash cluster/submit_campaign.sh --wave 9 --group betamin --n 1e4 --status    # likewise the other groups and n
sacct -M serial -X -u $USER -S now-2days --format=JobName%10,JobID%18,State,Elapsed | grep -v COMPLETED
grep -il "error\|traceback" logs/camp_*.err                                   # must print nothing

# 5. Laptop: bring the cells home and analyse (rerun as cells arrive)
rsync -av ge47xod3@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/
python simulations/diagnostics/campaign.py                                    # prints any incomplete cell
python simulations/diagnostics/graph_class.py && python3 simulations/diagnostics/plot_graph_class.py
python simulations/diagnostics/betamin.py --c-choices C_ID && python3 simulations/diagnostics/plot_betamin.py
python simulations/diagnostics/twocycles.py --c-choices C_ID && python3 simulations/diagnostics/plot_twocycles.py
```

## 6. Analysis and figures

```
campaign_per_dataset.csv ─┬─► graph_class.py:  gain within k and class, 2-cycle recovery      → gain_by_class_p<p>, two_cycle_recovery_p<p>
                          ├─► betamin.py:      Knob = level column + row/cell → level + sort/label/select
                          │      knob_table            levels ± se, paired gain over the lasso at the same level, per rule   → gain_by_wmin[_<rule>_f1], levels_by_wmin_n<n>
                          │      orientation_by_knob   per graph: single edges right/hedged/reversed/missed, 2-cycles both/half/neither, fp, identified → orientation_by_wmin_n<n>_<rule>, pair_outcomes_by_wmin_...
                          │      edge_outcomes         per true edge from the shards: |w|, |w| s_j/s_i, fate under every rule        → campaign_edge_outcomes.csv
                          │      binned_orientation    fates by |w| bin within a level (bins 0, .1, .25, .5, .75, 1, 2, ∞)           → orientation_by_weight_..., orientation_bin_across_laws_...
                          └─► twocycles.py:    the same Knob with the share as the level                                                  → gain_by_share, pair_outcomes_by_share_...
```

Single-edge fates ($j \to i$ true); 2-cycle edges `present` or `missed`; shares average over graphs, never levels:

| fate | $\hat S_{ij}$ | $\hat S_{ji}$ |
|---|---|---|
| `correct` | 1 | 0 |
| `hedged` | 1 | 1 |
| `reversed` | 0 | 1 |
| `missed` | 0 | 0 |

![](figures/gain_by_wmin_rehearsal.png)

![](figures/orientation_by_wmin_n1e4_bic_rehearsal.png)

![](figures/gain_by_share_rehearsal.png)

![](figures/pair_outcomes_by_share_p10_Cresc_n1e4_bic_rehearsal.png)

Previews: $p = 10$, `Cresc`, $n = 10^4$, 16 graphs; gain oracle / BIC:

| E3 | $N(0,1)$ | $U[0,1]$ | $U[0.1,1]$ | $U[0.25,1]$ | $U[0.5,1]$ |
|---|---|---|---|---|---|
| `lasso` oracle $F_1$ | 0.64 | 0.66 | 0.66 | 0.67 | 0.66 |
| `MCP-up` gain | +0.11 / +0.10 | +0.09 / +0.05 | +0.06 / +0.05 | +0.10 / +0.11 | +0.13 / +0.13 |
| `adaptive` gain | +0.07 / +0.07 | +0.09 / +0.08 | +0.09 / +0.08 | +0.10 / +0.12 | +0.12 / +0.12 |
| `lasso` single edges `hedged` / `missed` | 0.33 / 0.15 | 0.35 / 0.12 | 0.35 / 0.12 | 0.39 / 0.05 | 0.43 / 0.02 |

| E5 | `random` | $s = 0$ | $0.25$ | $0.5$ | $0.75$ | $1$ |
|---|---|---|---|---|---|---|
| `lasso` oracle $F_1$ | 0.64 | 0.61 | 0.67 | 0.71 | 0.74 | 0.75 |
| `MCP-up` gain | +0.11 / +0.10 | +0.11 / +0.07 | +0.08 / +0.07 | +0.04 / +0.04 | −0.04 / −0.08 | −0.09 / −0.12 |
| `adaptive` gain | +0.07 / +0.07 | +0.08 / +0.07 | +0.08 / +0.09 | +0.06 / +0.10 | 0.00 / 0.00 | −0.09 / −0.09 |
| true 2-cycles kept whole, `lasso` / `MCP-up` / `adaptive` | 0.37 / 0.08 / 0.16 | – | 0.46 / 0.20 / 0.25 | 0.50 / 0.21 / 0.37 | 0.40 / 0.11 / 0.21 | 0.42 / 0.11 / 0.17 |
| identified graphs | 1.00 | 1.00 | 0.94 | 0.81 | 0.75 | 0.62 |

## 7. Expected outcomes

The gain lives on detected true single edges. A miss costs everyone one false negative; on a detection the
lasso keeps both directions a third of the time, the pruners one. True 2-cycles reverse the roles.

| knob | pruners' gain | after search | refuted if |
|---|---|---|---|
| E3, $w_{\min} \uparrow$ | grows (rescued edges: lasso hedges, pruners correct) | $\approx 0$ at $n \ge 10^4$ | lasso hedge share falls |
| E4, `dag` | largest, every $k$ (every hedge is false) | $\approx 0$ | no gain over `random` within $k$ |
| E5, $s \uparrow$ | falls linearly; zero at $s \approx 0.6$ / $0.75$ (preview) | $\approx 0$; isolated 2-cycles at $s = 1$ | pruners not losing at $s = 1$ |

MCP sparse → dense not run: lock-in settled in wave 1, previews flat. At $n = \infty$: E3 flat at its largest
gain; E5's identified share and oracle levels fall with $s$.

## References

Dettling, Drton, Kolar (2024) §5; Varando & Hansen (2020) §4; Kalisch & Bühlmann (2007) §4.1; pcalg
`randomDAG`; NOTEARS `simulate_dag`; plan and memo in `../next_steps/Oct2026/131026`.
