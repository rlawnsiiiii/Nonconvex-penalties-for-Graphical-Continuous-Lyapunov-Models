#!/usr/bin/env python3
"""Replication, with the repository's own solvers, of the two central claims of the
independent study (next_steps/021026/independent_study):

  1. on standardised data the matching volatility is C = 2 diag(1 / s_ii^2) ("variance"),
     not 2 I ("identity");
  2. MCP run dense -> sparse (started from the lasso solution at the smallest lambda and
     walked up to lambda_max) beats the lasso, while the pilot's sparse -> dense path loses.

and the combination with the BIC search of S3b (gclm.solvers.search): which start is best?

Same datasets as S1 / S3a / S3b (repo seeds), p = 10 by default.  Nothing in src/ is changed:
"up" is implemented here with solve_fista, exactly as in the study's library_patch.diff.

    python next_steps/031026/files/replicate_dense_to_sparse.py --workers 4
    python next_steps/031026/files/replicate_dense_to_sparse.py --summarize
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import confusion, evaluate_path, orientation_breakdown  # noqa: E402
from gclm.objective.direct import diagonal_fit, lambda_max  # noqa: E402
from gclm.objective.penalties import penalty_weights  # noqa: E402
from gclm.solvers.path import lasso_path  # noqa: E402
from gclm.solvers.proxgrad import solve_fista  # noqa: E402
from gclm.solvers.search import Scorer, bic_along_path, greedy_search  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)
GAMMA = {"MCP": 3.0, "SCAD": 3.7}


def path_up(sigma, c, lambdas, dense, pen, tol):
    """MCP / SCAD from the lasso solution at the smallest lambda up to lambda_max."""
    p = sigma.shape[0]
    w = penalty_weights(p, penalize_diagonal=False)
    lmax = lambda_max(sigma, c, penalize_diagonal=False)
    warm, out = dense, []
    for lam in lambdas:                                   # increasing
        if lam >= lmax:
            warm = diagonal_fit(sigma, c)
        else:
            warm = solve_fista(sigma, c, float(lam), weights=w, m_init=warm, penalty=pen,
                               gamma=GAMMA[pen], tol=tol)
        m = warm.copy()
        m[np.abs(m) < 1e-12] = 0.0
        out.append(m)
    return out


def run(task):
    p, k, c_name, rep, n, c_scales, search = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
    m_true, _, _, s_raw = draw_instance(p, k, n, CChoice(c_name), rng, metzler=cfg.metzler,
                                        standardize=False)
    s = np.sqrt(np.diag(s_raw))
    r = s_raw / np.outer(s, s)                            # what standardize=True returns
    off = ~np.eye(p, dtype=bool)
    rows = []
    for c_scale in c_scales:
        c = 2.0 * np.eye(p) if c_scale == "identity" else np.diag(2.0 / s ** 2)
        t0 = time.perf_counter()
        lasso = lasso_path(r, c, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, tol=cfg.tol)
        paths = {"lasso": lasso.estimates,
                 "MCP_down": lasso_path(r, c, lambdas=lasso.lambdas, penalty="MCP", tol=cfg.tol).estimates,
                 "MCP_up": path_up(r, c, lasso.lambdas, lasso.estimates[0], "MCP", cfg.tol),
                 "SCAD_up": path_up(r, c, lasso.lambdas, lasso.estimates[0], "SCAD", cfg.tol)}
        t_paths = time.perf_counter() - t0
        sc = Scorer(r, c, n, "direct")
        for name, est in paths.items():
            ev = evaluate_path(est, m_true)
            sups = [(m != 0) & off for m in est]
            ib, _ = bic_along_path(r, c, n, sups, scorer=sc)
            b = orientation_breakdown(sups[ib].astype(float), m_true)
            row = {"p": p, "k": k, "c_choice": c_name, "rep": rep, "n": n, "c_scale": c_scale,
                   "method": name, "max_f1": ev["max_f1"], "auc": ev["auc"], "aupr": ev["aupr"],
                   "bic_f1": confusion(sups[ib].astype(float), m_true).f1,
                   "bic_edges": int(sups[ib].sum()), "bic_reversed": b["reversed"],
                   "bic_hedged": b["hedged"], "bic_skeleton_f1": b["skeleton_f1"],
                   "search_f1": "", "search_reversed": "", "search_skeleton_f1": "",
                   "seconds_paths": t_paths}
            if search and name in ("lasso", "MCP_down", "MCP_up"):
                res = greedy_search(r, c, n, sups[ib], scorer=sc)
                bs = orientation_breakdown(res.support.astype(float), m_true)
                row.update(search_f1=confusion(res.support.astype(float), m_true).f1,
                           search_reversed=bs["reversed"], search_skeleton_f1=bs["skeleton_f1"])
            rows.append(row)
    return rows


def summarize(path: Path):
    rows = list(csv.DictReader(path.open()))
    by = defaultdict(dict)
    for r in rows:
        by[(r["p"], r["k"], r["c_choice"], r["rep"], r["n"], r["c_scale"])][r["method"]] = r
    cells = sorted({(k[0], k[4], k[5]) for k in by}, key=lambda t: (int(t[0]), float(t[1]), t[2]))
    for p, n, cs in cells:
        keys = [k for k in by if (k[0], k[4], k[5]) == (p, n, cs)]
        print(f"\np = {p}, n = {n}, C scale = {cs}  [{len(keys)} graphs]")
        for col in ("max_f1", "bic_f1", "search_f1"):
            line = f"  {col:<10}"
            for m in ("lasso", "MCP_down", "MCP_up", "SCAD_up"):
                v = [float(by[k][m][col]) for k in keys if m in by[k] and by[k][m][col] != ""]
                if not v:
                    continue
                d = np.array([float(by[k][m][col]) - float(by[k]["lasso"][col]) for k in keys
                              if m in by[k] and by[k][m][col] != "" and by[k]["lasso"][col] != ""])
                se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
                z = f" (z {d.mean() / se:+.1f})" if m != "lasso" and se > 0 else ""
                line += f"  {m} {np.mean(v):.3f}{z}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000, math.inf])
    ap.add_argument("--c", nargs="+", default=["C_ID"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--c-scales", nargs="+", default=["identity", "variance"])
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "replicate_dense_to_sparse.csv")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.out)
    tasks = [(args.p, k, c, rep, n, tuple(args.c_scales), not args.no_search)
             for n in args.n for k in (1, 2, 3, 4) for c in args.c for rep in range(args.reps)]
    t0, writer = time.time(), None
    with ProcessPoolExecutor(args.workers) as ex, args.out.open("w", newline="") as fh:
        for i, rows in enumerate(ex.map(run, tasks, chunksize=1), 1):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            if i % 10 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} datasets, {(time.time() - t0) / 60:.1f} min", flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
