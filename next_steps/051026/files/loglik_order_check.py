#!/usr/bin/env python3
"""Does the order of the path matter for the log-likelihood LASSO with the repository's solver?

Varando & Hansen run their paths dense -> sparse and report that order as better.  With their
package (estimate_c_check.py) the order matters a lot, but their solver stops early, so that
could be an effect of the stopping rule.  This runs the repository's solver, which converges
every fit to a tight tolerance, in both orders on the same graphs: C = 2I, standardised data,
settings of run_s1_shard.py.

    python next_steps/051026/files/loglik_order_check.py --workers 5
    python next_steps/051026/files/loglik_order_check.py --summarize
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import evaluate_path  # noqa: E402
from gclm.objective import covariance  # noqa: E402
from gclm.solvers.path import covloss_path, lambda_grid  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)


def run(task):
    p, k, c_name, rep, n, direction = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
    m_true, _, _, r = draw_instance(p, k, n, CChoice(c_name), rng, metzler=cfg.metzler,
                                    standardize=True)
    c = 2.0 * np.eye(p)
    lams = lambda_grid(covariance.lambda_max(r, c, "loglik"), n_lambda=cfg.n_lambda,
                       ratio=cfg.lambda_ratio)
    t0 = time.perf_counter()
    path = covloss_path(r, c, "loglik", lambdas=lams, penalty="lasso", direction=direction,
                        tol=cfg.tol)
    ev = evaluate_path(path.estimates, m_true)
    return {"p": p, "k": k, "c_choice": c_name, "rep": rep, "n": n, "direction": direction,
            "max_f1": ev["max_f1"], "auc": ev["auc"], "aupr": ev["aupr"],
            "not_converged": int((~path.converged).sum()),
            "seconds": time.perf_counter() - t0}


def summarize(path: Path):
    rows = list(csv.DictReader(path.open()))
    by = defaultdict(dict)
    for r in rows:
        by[(r["k"], r["c_choice"], r["rep"], r["n"])][r["direction"]] = r
    for n in sorted({k[3] for k in by}, key=float):
        keys = [k for k in by if k[3] == n and len(by[k]) == 2]
        line = f"n = {n} [{len(keys)} graphs]"
        for col in ("max_f1", "aupr", "auc"):
            d = np.array([float(by[k]["down"][col]) for k in keys])
            u = np.array([float(by[k]["up"][col]) for k in keys])
            diff = u - d
            se = diff.std(ddof=1) / math.sqrt(len(diff))
            line += (f"  {col}: sparse->dense {d.mean():.3f}, dense->sparse {u.mean():.3f} "
                     f"(z {diff.mean() / se if se > 0 else 0:+.1f})")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000])
    ap.add_argument("--c", nargs="+", default=["C_ID"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=HERE / "loglik_order_check.csv")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if not args.summarize:
        tasks = [(args.p, k, c, rep, n, d) for n in args.n for k in (1, 2, 3, 4) for c in args.c
                 for rep in range(args.reps) for d in ("down", "up")]
        with ProcessPoolExecutor(args.workers) as ex, args.out.open("w", newline="") as fh:
            writer = None
            for fut in as_completed([ex.submit(run, task) for task in tasks]):
                row = fut.result()
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
    summarize(args.out)


if __name__ == "__main__":
    main()
