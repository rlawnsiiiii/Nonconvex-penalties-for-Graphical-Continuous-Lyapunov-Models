# Master's thesis — nonconvex penalties for Graphical Continuous Lyapunov Models

Research plan and meeting notes: [`plan.md`](plan.md).
Code map and diagrams: [`ARCHITECTURE.md`](ARCHITECTURE.md).
How the problem is encoded into glmnet / ncvreg: [`R/ENCODING.md`](R/ENCODING.md).

## Layout

```
plan.md                      topic, references, meeting notes
ARCHITECTURE.md              repository structure, with diagrams
simulations/
  S1_reproduction.md         spec + implementation plan for study S1
  run_m0.py                  Dettling Example 2 / Figure 3   (minutes)
  run_s1.py                  Dettling Section 5 / Figure 5   (~7 h on 8 cores)
  results/                   committed run outputs
src/gclm/                    the library (see S1_reproduction.md Section 7.1)
R/                           glmnet + ncvreg solver backends (see R/ENCODING.md)
tests/                       pytest; the `r` marker needs Rscript + glmnet/ncvreg
```

## Getting started

```bash
pip install -e ".[dev]"
pytest                       # 85 tests; R-backed ones skip if Rscript is absent
pytest -m "not r"            # skip the R validation explicitly

python simulations/run_m0.py --reps 100          # reproduces Figure 3
python simulations/run_s1.py --p 10 --reps 10    # small slice of Figure 5
```

### Solver backends

All four minimize the same objective and are cross-checked against each other
(`test_all_four_backends_agree`). See S1_reproduction.md §7.2.

```bash
python simulations/run_s1.py --solver fista    # default; the only one that reaches p = 50
python simulations/run_s1.py --solver glmnet   # Dettling's own choice (Appendix A)
python simulations/run_s1.py --solver ncvreg   # most accurate (4e-12 vs exact KKT)
python simulations/run_s1.py --solver design   # transparent Python reference

# MCP / SCAD -- the S1b entry point, via ncvreg::ncvfit
python simulations/run_s1.py --solver ncvreg --penalty MCP  --gamma 3
python simulations/run_s1.py --solver ncvreg --penalty SCAD --gamma 3.7
```

The R backends and validation tests need:

```r
install.packages(c("glmnet", "ncvreg", "jsonlite"))
```

R is optional: nothing imports the bridge unless an R backend is selected, and every
R-backed test skips cleanly without `Rscript`.

## Studies

| id | loss | penalty | status |
|---|---|---|---|
| **S1** | quadratic (Frobenius) Lyapunov loss | $\ell_1$ | reproduction in progress |
| S1b | quadratic | MCP / SCAD | planned — swaps the prox only |
| S2 | Gaussian log-likelihood | $\ell_1$ vs. nonconvex | planned |
| S3 | quadratic or likelihood | BIC-type, score-based search | planned |
