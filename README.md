# Master's thesis — nonconvex penalties for Graphical Continuous Lyapunov Models

Research plan and meeting notes: [`plan.md`](plan.md).
Code map and diagrams: [`ARCHITECTURE.md`](ARCHITECTURE.md).
How the problem is encoded into glmnet / ncvreg: [`R/ENCODING.md`](R/ENCODING.md).
The default solver and why it is hand-written: [`docs/FISTA.md`](docs/FISTA.md).
MCP and SCAD — definitions, the two conventions, validation: [`docs/NONCONVEX.md`](docs/NONCONVEX.md).
Varando's likelihood and Frobenius losses, and their Newton solver: [`docs/LIKELIHOOD.md`](docs/LIKELIHOOD.md).
Running the full reproduction on a cluster: [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Layout

```
plan.md                      topic, references, meeting notes
next_steps/                  dated next-steps memos (next_steps_011026.md: orientation finding, literature, remedies, plan)
ARCHITECTURE.md              repository structure, with diagrams
simulations/
  S1_reproduction.md         spec + implementation plan for study S1
  run_m0.py                  Dettling Example 2 / Figure 3   (minutes)
  run_s1.py                  Dettling Section 5 / Figure 5   (local, multiprocessing)
  run_s1_shard.py            the same grid, one shard -> .npz  (cluster)
  aggregate_s1.py            shards -> tidy CSVs
  plot_figures.py            CSVs -> figures            (LOCAL ONLY)
  plot_penalties.py          overlay runs: penalties / losses vs. p (LOCAL ONLY)
  S2_penalties_losses.md     S1b + S2: setup and pilot results
  results/                   committed run outputs
src/gclm/                    the library: lyapunov.py, data/, objective/, solvers/, metrics.py (ARCHITECTURE.md §1)
cluster/                     LRZ SLURM batch scripts + env setup
runs/                        one folder per simulation run (raw shards gitignored)
  s1_dettling_reproduction/  the full cluster run: data, figures, comparison with the paper
figures/                     generated plots (local)
R/                           glmnet + ncvreg solver backends (see R/ENCODING.md); gclm reference
tests/                       pytest; the `r` marker needs Rscript + glmnet/ncvreg/gclm
```

## Getting started

```bash
pip install -e ".[dev]"
pytest                       # 377 tests; R-backed ones skip if Rscript is absent
pytest -m "not r"            # skip the R validation explicitly

python simulations/run_m0.py --reps 100          # reproduces Figure 3
python simulations/run_s1.py --p 10 --reps 10    # small slice of Figure 5
```

### Solver backends

All four minimize the same objective and are cross-checked against each other
(`test_all_four_backends_agree`). See S1_reproduction.md §7.2.

```bash
python simulations/run_s1.py                      # default: fista (reaches p = 50)
python simulations/run_s1.py --solver ncvreg      # most accurate (4e-12 vs exact KKT)
python simulations/run_s1.py --solver skglm       # pure Python, MCP, no R
python simulations/run_s1.py --solver glmnet      # Dettling's own choice (Appendix A)
python simulations/run_s1.py --solver pyproximal  # packaged FISTA, matrix-free
python simulations/run_s1.py --solver design      # transparent Python reference

# MCP / SCAD (study S1b) -- see docs/NONCONVEX.md, in particular Section 2
python simulations/run_s1.py --penalty MCP                         # fista, textbook MCP
python simulations/run_s1.py --penalty SCAD --gamma 3.7
python simulations/run_s1.py --penalty MCP --solver skglm          # package, textbook
python simulations/run_s1.py --penalty MCP --solver ncvreg --convention ncvreg

# Varando's losses on the implied covariance (study S2) -- see docs/LIKELIHOOD.md
python simulations/run_s1.py --loss loglik    --penalty lasso   # own Newton solver; --solver ignored
python simulations/run_s1.py --loss frobenius --penalty SCAD
```

The R backends and validation tests need:

```r
install.packages(c("glmnet", "ncvreg", "gclm", "jsonlite"))
```

Optional Python backends and test oracles: `pip install skglm pyproximal pylops cvxpy`.

R is optional: nothing imports the bridge unless an R backend is selected, and every
R-backed test skips cleanly without `Rscript`.

## Studies

| id | loss | penalty | status |
|---|---|---|---|
| **S1** | direct (quadratic) Lyapunov loss | $\ell_1$ | Figure 5 reproduced at full scale; two $C$ settings differ from the paper ([`simulations/S1_reproduction.md`](simulations/S1_reproduction.md) §8.10) |
| S1b | direct | MCP / SCAD | pilot at $p=10,20$: both find the **skeleton as well as the lasso but orient edges worse** (directed $z$ −6…−18); mechanism in [`simulations/S2_penalties_losses.md`](simulations/S2_penalties_losses.md) §3.1 |
| S2 | Gaussian log-likelihood, Frobenius on $\Sigma(M)$ | $\ell_1$ vs. MCP / SCAD | pilot at $p=10$: same ordering on both losses; lasso + log-likelihood slightly beats the direct lasso (§3.2–3.3); larger $p$ on the cluster (§6) |
| S3 | quadratic or likelihood | BIC-type, score-based search | planned |
