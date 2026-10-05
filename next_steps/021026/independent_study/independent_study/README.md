# Independent study (2 October 2026)

Code, raw results and figures behind
[`../independent_study_021026.md`](../independent_study_021026.md).
Self-contained: it imports the thesis library (`src/gclm`) for the data-generating process, the
metrics and the Gaussian likelihood, and brings its own fast solvers.

```
lyapcd.c            C solvers for the penalised direct loss: coordinate descent (lasso / MCP / SCAD,
                    per-entry weights), CD with edge-reversal moves, and line-by-line ports of the
                    repo's FISTA and monotone APG.  Compiled on first import (needs gcc).
fastlyap.py         ctypes wrapper: cd_path, mapg_path (= the repo's MCP/SCAD estimator),
                    mapg_up_path / cd_swap_path, objective, stationarity, lambda grid
gx.py               instances (the repo's RNG stream), the three input formulations, the explicit
                    design, exact (weighted) lasso paths by LARS, metrics, likelihood, EBIC
methods.py          the estimators: LLA, adaptive lasso, thresholding, dense-to-sparse continuation,
                    forward / backward selection with exchange moves, basis pursuit as an LP
e0_misspec.py       goodness of fit of the TRUE graph under the two formulations
e1_formulations.py  formulation x penalty x n, and MCP started at the truth
e2_identifiability.py   information per true edge (presence / direction) in the population
e3_bakeoff.py       all estimators, path metrics and BIC-selected graphs   (also used for p = 20)
e5_gamma.py         gamma sweep for the repo's MCP / SCAD path
e6_abstain.py       orient-or-abstain on BIC-selected graphs
e7_direction.py     path direction x solver x penalty
e8_gmc.py / gmc.py / lyapgmc.c   GMC penalty (Selesnick 2017), exact-KKT path solver and trial
e9_bp_population.py minimum-l1 exact solution (basis pursuit) at the population level
e10_example2.py     Dettling's Example 2, every estimator, three formulations
s3b.py, e11_s3b.py  the BIC search of simulations/S3b_reversal_search.md (direct-loss arm) and its trial
s3b_checks.py, s3b_refit_check.py   phase-0 checks of that search on Example 2
analyze.py          every table of the memo, from results/           ->  results/tables.md
make_figures.py     the three figures                                 ->  figures/
check_vs_repo_driver.py   this study's numbers against simulations/run_s1.py on the same datasets
results/            one JSON line per (dataset, n, formulation, method); gzip (analyze.py reads .jsonl or .jsonl.gz)
```

## Running it

```bash
cd simulations/independent_study
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1      # the scripts use 2 worker processes
python e0_misspec.py 10 25                           #  2 min
python e2_identifiability.py 10 25 dettling          #  3 min   (also: unif, unif_no2c)
python e1_formulations.py 10 13                      # 30 min   (52 drift matrices)
python e3_bakeoff.py 10 25 dettling 1000 1e4 1e5 inf # 45 min   (100 drift matrices)
python e7_direction.py 10 13 dettling 1000 1e4 1e5 inf
python e5_gamma.py 10 13
python e6_abstain.py 10 25 dettling 1000 1e4
python e3_bakeoff.py 10 13 unif_no2c 1000 1e4 inf
E3_METHODS="mcp_mapg,scad_mapg,adalasso_bp,thr_bp,mcp_up,bwd,fwd" E3_ORACLE=0 E3_TAG=reduced E3_NTOP=3 \
    python e3_bakeoff.py 20 6 dettling 1000 1e4 inf
python e8_gmc.py 10 13 1000 1e4 inf
python e9_bp_population.py corrC; python e9_bp_population.py raw2I
python e10_example2.py 100
python e11_s3b.py 10 10 1000 1e4 inf
python e6_abstain.py 10 13 unif_no2c 1000 1e4
python e7_direction.py 10 13 unif_no2c 1000 1e4 inf
python analyze.py e1 e2 e3 e7 e5 e6 e3u e4 e6u e7u e10 e11 memo_e8 > results/tables.md
python make_figures.py
```

Timings are for two cores of a small cloud machine. Needs `numpy`, `scipy`, `scikit-learn`
(LARS), `joblib`, `pandas`, `matplotlib`, and `gcc`. No R.

Set the BLAS thread count to 1: with the default, three or four Python processes on two cores
ran 10-20 times slower.

## Conventions

Everything follows the repo: `M[i, k] != 0` is the edge `k -> i`; estimates along a path are
ordered by increasing lambda (dense to sparse); the diagonal is unpenalised and unscored; the
objective is `0.5 * ||M S + S M' + C||_F^2 + sum P(M_ij)` with the textbook MCP / SCAD.

`gx.formulation(s_hat, kind)` returns the `(S, C)` handed to the estimator:

| kind | S | C | |
|---|---|---|---|
| `corr2I` | correlation matrix | `2 I` | the pilot's setting; `--c-scale identity` in the patched drivers |
| `corrC` | correlation matrix | `2 diag(1 / s_ii)` | well specified; `--c-scale variance` |
| `raw2I` | covariance matrix | `2 I` | well specified, badly scaled |
