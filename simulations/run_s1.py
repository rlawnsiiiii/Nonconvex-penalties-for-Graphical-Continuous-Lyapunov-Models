#!/usr/bin/env python3
"""S1 (M1) -- Dettling et al. (2024) Section 5 / Figure 5.

Writes one CSV row per (p, k, C_choice, rep) with the four reported metrics.
Tasks are independent, so this maps directly onto a cluster array job: use
--shard i/N to run a slice.

    python simulations/run_s1.py --p 10 15 --reps 20 --out results/s1.csv
    python simulations/run_s1.py --shard 3/64 --out results/s1_shard03.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gclm.config import S1Config
from gclm.dgp import CChoice, draw_instance
from gclm.lasso import lasso_path
from gclm.metrics import evaluate_path

FIELDS = ["p", "k", "c_choice", "rep", "max_acc", "max_f1", "auc", "aupr", "seconds"]

SOLVERS = ("fista", "ncvreg", "skglm", "glmnet", "pyproximal", "design")


def run_one(task):
    p, k, c_choice, rep, cfg = task
    # independent, reproducible stream per task -- order and parallelism agnostic
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c_choice), rep])
    t0 = time.perf_counter()
    m_true, _, _, sigma_hat = draw_instance(
        p, k, cfg.n_obs, c_choice, rng,
        metzler=cfg.metzler, standardize=cfg.standardize,
    )
    c_est = 2.0 * np.eye(p)                       # always C = 2 I for estimation
    path = lasso_path(
        sigma_hat, c_est,
        n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio,
        penalize_diagonal=cfg.penalize_diagonal, solver=cfg.solver, tol=cfg.tol,
        penalty=cfg.penalty, gamma=cfg.gamma,
    )
    ev = evaluate_path(path.estimates, m_true,
                       include_diagonal=cfg.metrics_include_diagonal)
    return {
        "p": p, "k": k, "c_choice": c_choice.value, "rep": rep,
        "max_acc": ev["max_acc"], "max_f1": ev["max_f1"],
        "auc": ev["auc"], "aupr": ev["aupr"],
        "seconds": round(time.perf_counter() - t0, 3),
    }


def main() -> None:
    base = S1Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=list(base.p_values))
    ap.add_argument("--k", type=int, nargs="+", default=list(base.k_values))
    ap.add_argument("--c", nargs="+", default=[c.value for c in base.c_choices])
    ap.add_argument("--reps", type=int, default=base.n_rep)
    ap.add_argument("--n-obs", type=int, default=base.n_obs)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard", default=None, help="i/N -- run shard i of N")
    ap.add_argument("--penalize-diagonal", action="store_true")
    ap.add_argument("--metrics-include-diagonal", action="store_true")
    ap.add_argument("--no-standardize", dest="standardize",
                    action="store_false", default=True,
                    help="use the covariance matrix instead of the correlation matrix")
    ap.add_argument("--tol", type=float, default=base.tol)
    ap.add_argument("--solver", default=base.solver,
                    choices=["fista", "ncvreg", "skglm", "glmnet", "pyproximal", "design"],
                    help="fista: default; the only one that reaches p=50. "
                         "ncvreg: most accurate, MCP and SCAD. "
                         "skglm: pure-Python MCP, no R. "
                         "glmnet: Dettling's own choice. "
                         "pyproximal: packaged FISTA.")
    ap.add_argument("--penalty", default=base.penalty,
                    choices=["lasso", "MCP", "SCAD"],
                    help="MCP needs --solver ncvreg or skglm; SCAD needs ncvreg")
    ap.add_argument("--gamma", type=float, default=base.gamma,
                    help="MCP/SCAD concavity parameter (ncvreg default: 3 / 3.7)")
    ap.add_argument("--out", type=Path, default=Path("results/s1.csv"))
    args = ap.parse_args()

    cfg = S1Config(
        p_values=tuple(args.p), k_values=tuple(args.k),
        c_choices=tuple(CChoice(c) for c in args.c),
        n_rep=args.reps, n_obs=args.n_obs,
        penalize_diagonal=args.penalize_diagonal,
        metrics_include_diagonal=args.metrics_include_diagonal,
        standardize=args.standardize, tol=args.tol,
        solver=args.solver, penalty=args.penalty, gamma=args.gamma,
    )

    tasks = [(p, k, c, r, cfg)
             for p in cfg.p_values for k in cfg.k_values
             for c in cfg.c_choices for r in range(cfg.n_rep)]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        tasks = tasks[i::n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks on {args.workers} workers "
          f"[solver={cfg.solver}, penalty={cfg.penalty}] -> {args.out}", flush=True)

    t0 = time.time()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
                writer.writerow(row)
                if i % 50 == 0 or i == len(tasks):
                    fh.flush()
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(tasks)}  {rate:.1f}/s  "
                          f"eta {(len(tasks) - i) / rate / 60:.1f}min", flush=True)

    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
