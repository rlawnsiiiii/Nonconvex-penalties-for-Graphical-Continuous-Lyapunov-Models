#!/usr/bin/env python3
"""Timing pilot for the plan of 5 October: what does a log-likelihood path cost when it is run
dense -> sparse (direction="up", Varando & Hansen's order) instead of sparse -> dense, and with
the rescaled C?  A handful of the repository's datasets, same settings as run_s1_shard.py.
Prints one line per (dataset, C, penalty, direction); the numbers go into next_steps_051026.md.

    python next_steps/051026/files/time_loglik_up.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import evaluate_path  # noqa: E402
from gclm.objective import covariance  # noqa: E402
from gclm.solvers.path import covloss_path, lambda_grid  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)
DATASETS = ((10, 2, "C_ID", 0), (10, 4, "C_Random_Diag", 0), (10, 1, "C_ID", 1),
            (20, 2, "C_ID", 0), (20, 4, "C_Random_Diag", 0))


def main():
    cfg = S1Config()
    n = 1000
    for p, k, c_name, rep in DATASETS:
        rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
        m_true, _, _, s_raw = draw_instance(p, k, n, CChoice(c_name), rng, metzler=cfg.metzler,
                                            standardize=False)
        s = np.sqrt(np.diag(s_raw))
        r = s_raw / np.outer(s, s)
        for c_scale in ("identity", "variance"):
            c = 2.0 * np.eye(p) if c_scale == "identity" else np.diag(2.0 / s ** 2)
            lams = lambda_grid(covariance.lambda_max(r, c, "loglik"), n_lambda=cfg.n_lambda,
                               ratio=cfg.lambda_ratio)
            for pen in ("lasso", "MCP"):
                for direction in ("down", "up"):
                    t0 = time.perf_counter()
                    path = covloss_path(r, c, "loglik", lambdas=lams, penalty=pen,
                                        direction=direction, tol=cfg.tol)
                    dt = time.perf_counter() - t0
                    ev = evaluate_path(path.estimates, m_true)
                    print(f"p={p} k={k} {c_name} rep={rep} C={c_scale:<8} {pen:<5} {direction:<4} "
                          f"{dt:8.1f}s  max_f1 {ev['max_f1']:.3f} aupr {ev['aupr']:.3f}  "
                          f"iters {int(path.iterations.sum())}  "
                          f"not converged {int((~path.converged).sum())}", flush=True)


if __name__ == "__main__":
    main()
