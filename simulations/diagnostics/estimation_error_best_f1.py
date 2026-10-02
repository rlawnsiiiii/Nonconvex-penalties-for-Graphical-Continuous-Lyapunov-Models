#!/usr/bin/env python3
"""Estimation error and shrinkage at each method's best-F1 lambda, paired over
identical datasets.

The support metrics of Figure 5 say nothing about the *values* of the selected
entries, and the advertised advantage of MCP/SCAD is precisely there: less
shrinkage of the large coefficients.  Every shard stores M_hat at the path's
best-F1 point as a sparse triple, so this can be read off without refitting.

    python simulations/diagnostics/estimation_error_best_f1.py \\
        --run lasso=runs/s1_dettling_reproduction \\
        --run MCP=runs/s1b_pilot_p10-20/MCP --run SCAD=runs/s1b_pilot_p10-20/SCAD \\
        --reps 25 --p 10 20

Per (p, method), averaged over the paired datasets:
  rel_err_all        ||M_hat - M*||_F / ||M*||_F
  rel_err_offdiag    the same on the off-diagonal part
  rel_err_sel_true   the same restricted to true edges that were selected
  shrink_sel_true    sum |M_hat_ij| / sum |M*_ij| over selected true edges
                     (< 1: shrunk towards zero; ~1: unbiased in size)
  n_sel_true/n_true  how many true edges the best-F1 support contains
  n_false            false edges in the best-F1 support
  diag_rel_err       relative error of the (unpenalised) diagonal
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run: Path, reps: int | None, ps):
    out = {}
    for f in sorted(glob.glob(str(run / "s1_shards" / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["c_choice_names"]]
        for i in range(len(d["p"])):
            p, rep = int(d["p"][i]), int(d["rep"][i])
            if (reps is not None and rep >= reps) or (ps is not None and p not in ps):
                continue
            key = (p, int(d["k"][i]), names[int(d["c_choice"][i])], rep)
            mt = np.zeros((p, p))
            mt[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
            mh = np.zeros((p, p))
            mh[d["m_best_f1_i"][i], d["m_best_f1_j"][i]] = d["m_best_f1_v"][i]
            out[key] = (mt, mh)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=run directory")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--p", type=int, nargs="+", default=None)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, _, path = spec.rpartition("=")
        runs[label] = load(Path(path), args.reps, args.p)
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(keys)} paired datasets; quantities at each method's own best-F1 lambda\n")

    stats = defaultdict(lambda: defaultdict(list))
    for key in keys:
        p = key[0]
        off = ~np.eye(p, dtype=bool)
        for name, run in runs.items():
            mt, mh = run[key]
            true = (mt != 0) & off
            sel = (mh != 0) & true
            st = stats[(p, name)]
            st["rel_err_all"].append(np.linalg.norm(mh - mt) / np.linalg.norm(mt))
            st["rel_err_offdiag"].append(np.linalg.norm((mh - mt)[off]) / np.linalg.norm(mt[off]))
            st["rel_err_sel_true"].append(
                np.linalg.norm((mh - mt)[sel]) / np.linalg.norm(mt[sel]) if sel.any() else np.nan)
            st["shrink_sel_true"].append(
                np.abs(mh[sel]).sum() / np.abs(mt[sel]).sum() if sel.any() else np.nan)
            st["n_sel_true"].append(sel.sum())
            st["n_true"].append(true.sum())
            st["n_false"].append(((mh != 0) & ~true & off).sum())
            st["diag_rel_err"].append(np.linalg.norm(np.diag(mh - mt)) / np.linalg.norm(np.diag(mt)))

    cols = ["rel_err_all", "rel_err_offdiag", "rel_err_sel_true", "shrink_sel_true", "n_sel_true", "n_true", "n_false", "diag_rel_err"]
    print(f"{'p':>3} {'method':<8}" + "".join(f"{c:>18}" for c in cols))
    for (p, name), st in sorted(stats.items(), key=lambda kv: (kv[0][0], list(runs).index(kv[0][1]))):
        print(f"{p:>3} {name:<8}" + "".join(f"{np.nanmean(st[c]):>18.3f}" for c in cols))


if __name__ == "__main__":
    main()
