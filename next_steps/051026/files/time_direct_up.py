#!/usr/bin/env python3
"""Timing pilot for wave 1 of the campaign (direct loss): what do the new paths cost relative to
the standard MCP / SCAD path?  Dense -> sparse is implemented as in
next_steps/031026/files/replicate_dense_to_sparse.py (identical to library_patch.diff).

    python next_steps/051026/files/time_direct_up.py
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
sys.path.insert(0, str(ROOT / "next_steps" / "031026" / "files"))

from gclm.config import S1Config  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import evaluate_path  # noqa: E402
from gclm.solvers.path import lasso_path  # noqa: E402
from replicate_dense_to_sparse import path_up  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)
DATASETS = ((10, 2, "C_ID", 0), (20, 2, "C_ID", 0), (20, 4, "C_Random_Diag", 0), (20, 1, "C_ID", 1))


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
            t0 = time.perf_counter()
            lasso = lasso_path(r, c, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, tol=cfg.tol)
            line = f"p={p} k={k} {c_name} rep={rep} C={c_scale:<8} lasso {time.perf_counter() - t0:6.1f}s"
            f1 = f" | max_f1 lasso {evaluate_path(lasso.estimates, m_true)['max_f1']:.3f}"
            for pen in ("MCP", "SCAD"):
                t0 = time.perf_counter()
                down = lasso_path(r, c, lambdas=lasso.lambdas, penalty=pen, tol=cfg.tol).estimates
                t1 = time.perf_counter()
                up = path_up(r, c, lasso.lambdas, lasso.estimates[0], pen, cfg.tol)
                t2 = time.perf_counter()
                line += f"  {pen} standard {t1 - t0:6.1f}s, dense->sparse {t2 - t1:6.1f}s"
                f1 += (f", {pen} {evaluate_path(down, m_true)['max_f1']:.3f} -> "
                       f"{evaluate_path(up, m_true)['max_f1']:.3f}")
            print(line + f1, flush=True)


if __name__ == "__main__":
    main()
