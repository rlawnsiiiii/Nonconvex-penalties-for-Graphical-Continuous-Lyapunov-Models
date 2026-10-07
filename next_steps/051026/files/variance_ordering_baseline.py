#!/usr/bin/env python3
"""How much of the gain with the rescaled C is "the high-variance node is the parent"?

With C = 2I known on the measurement scale, the variances carry information about direction: in
Dettling's DGP most true edges point from the node with the larger variance to the one with the
smaller.  A trivial rule that uses only this: take the pairs the lasso finds (its skeleton, at
every lambda) and orient each pair from the larger raw variance to the smaller.  If that rule did
as well as MCP dense -> sparse, the gain would be an artefact of the DGP.

Same 40 C_ID graphs (p = 10) as next_steps/031026/files/replicate_dense_to_sparse.py, whose
csv supplies the lasso and MCP numbers with the rescaled C for comparison.  Also reported:
the same skeleton oriented by the truth (the ceiling for any orientation rule that keeps one
direction per pair) and by a coin.

    python next_steps/051026/files/variance_ordering_baseline.py --workers 3
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import evaluate_path  # noqa: E402
from gclm.solvers.path import lasso_path  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)
REPLICATION = ROOT / "next_steps" / "031026" / "files" / "replicate_dense_to_sparse.csv"


def orient(skeleton, parent_first):
    """One direction per pair of the skeleton.  Entry [j, i] is the edge i -> j;
    ``parent_first[i, j]`` says that i is taken to be the parent in the pair {i, j}."""
    out = np.zeros(skeleton.shape)
    i, j = np.nonzero(np.triu(skeleton, 1))
    for a, b in zip(i, j):
        if parent_first[a, b]:
            out[b, a] = 1.0
        else:
            out[a, b] = 1.0
    return out


def run(task):
    p, k, c_name, rep, n = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
    m_true, _, _, s_raw = draw_instance(p, k, n, CChoice(c_name), rng, metzler=cfg.metzler,
                                        standardize=False)
    s = np.sqrt(np.diag(s_raw))
    r = s_raw / np.outer(s, s)
    c = np.diag(2.0 / s ** 2)
    path = lasso_path(r, c, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, tol=cfg.tol)
    off = ~np.eye(p, dtype=bool)
    truth = (m_true != 0) & off
    by_variance = s[:, None] > s[None, :]                 # [i, j]: i has the larger variance
    # truth as the orientation rule: i is the parent if i -> j is a true edge; pairs without a
    # true edge get the variance rule (they are false positives either way)
    by_truth = np.where(truth.T | truth, truth.T, by_variance)
    coin = np.random.default_rng([cfg.seed, p, k, rep, 7]).random((p, p)) < 0.5
    paths = {"lasso": path.estimates, "variance_rule": [], "truth_rule": [], "coin": []}
    for m in path.estimates:
        sk = (m != 0) & off
        sk = sk | sk.T
        paths["variance_rule"].append(orient(sk, by_variance))
        paths["truth_rule"].append(orient(sk, by_truth))
        paths["coin"].append(orient(sk, coin))
    # share of the true single edges that point from the larger to the smaller variance
    single = truth & ~truth.T
    jj, ii = np.nonzero(single)                           # edge ii -> jj
    share = float(np.mean(s[ii] > s[jj])) if len(ii) else float("nan")
    rows = []
    for name, est in paths.items():
        ev = evaluate_path(est, m_true)
        rows.append({"p": p, "k": k, "c_choice": c_name, "rep": rep, "n": n, "method": name,
                     "max_f1": ev["max_f1"], "aupr": ev["aupr"], "high_to_low_share": share})
    return rows


def summarize(path: Path):
    rows = list(csv.DictReader(path.open()))
    val = defaultdict(dict)
    for r in rows:
        val[(r["k"], r["rep"], r["n"])][r["method"]] = float(r["max_f1"])
    for r in csv.DictReader(REPLICATION.open()):
        if r["c_scale"] == "variance" and r["method"] in ("MCP_up", "MCP_down"):
            key = (r["k"], r["rep"], r["n"])
            if key in val:
                val[key][r["method"]] = float(r["max_f1"])
    share = np.nanmean([float(r["high_to_low_share"]) for r in rows if r["method"] == "lasso"])
    print(f"true single edges pointing from the larger to the smaller variance: {share:.2f}")
    ns = sorted({k[2] for k in val}, key=float)
    methods = ("lasso", "MCP_down", "MCP_up", "variance_rule", "coin", "truth_rule")
    print(f"{'max_f1, rescaled C':<22}" + "".join(f"n = {n:<16}" for n in ns))
    for m in methods:
        line = f"{m:<22}"
        for n in ns:
            keys = [k for k in val if k[2] == n and m in val[k]]
            x = np.array([val[k][m] for k in keys])
            d = x - np.array([val[k]["lasso"] for k in keys])
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
            z = f"({d.mean() / se:+.1f})" if m != "lasso" and se > 0 else ""
            line += f"{x.mean():.3f} {z:<7} [{len(x)}]   "
        print(line)
    for dens, ks in (("k = 1, 2", ("1", "2")), ("k = 3, 4", ("3", "4"))):
        print(dens)
        for m in ("lasso", "MCP_up", "variance_rule", "truth_rule"):
            line = f"  {m:<20}"
            for n in ns:
                x = [val[k][m] for k in val if k[2] == n and k[0] in ks and m in val[k]]
                line += f"{np.mean(x):.3f}                  "
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000, math.inf])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "variance_ordering_baseline.csv")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if not args.summarize:
        tasks = [(args.p, k, "C_ID", rep, n) for n in args.n for k in (1, 2, 3, 4)
                 for rep in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as ex, args.out.open("w", newline="") as fh:
            writer = None
            for rows in ex.map(run, tasks, chunksize=2):
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
    summarize(args.out)


if __name__ == "__main__":
    main()
