#!/usr/bin/env python3
"""Skeleton vs. orientation: does a penalty lose on *which pairs* are connected
or on *which way* the edge points?

Two modes.

``--shards``  reads the stored best-F1 estimates of finished runs (no refits)
and reports, per (p, C choice, run), the mean of each orientation category of
``gclm.metrics.orientation_breakdown`` plus skeleton F1 and orientation
accuracy, with paired differences against the first run given:

    python simulations/diagnostics/orientation.py --shards \\
        lasso=runs/s1_dettling_reproduction MCP=runs/s1b_pilot_p10-20/MCP SCAD=runs/s1b_pilot_p10-20/SCAD \\
        --reps 25 --p 10 20

``--refit``   regenerates the datasets from the seed and refits the paths, so
that the skeleton metrics can be computed along the whole path (max skeleton
F1, skeleton AUC / AUPR -- the Figure 5 metrics for the undirected graph), not
only at the directed best-F1 point:

    python simulations/diagnostics/orientation.py --refit --loss direct \\
        --penalties lasso MCP SCAD --p 10 --reps 10 --workers 6 \\
        --out runs/s1b_pilot_p10-20/orientation_path_p10.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gclm.config import S1Config  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import (  # noqa: E402
    ORIENTATION_KEYS,
    confusion,
    evaluate_path,
    evaluate_skeleton_path,
    orientation_breakdown,
)
from gclm.solvers.path import fit_path  # noqa: E402

CATS = list(ORIENTATION_KEYS) + ["n_true_single", "n_true_double"]
SUMMARY = ["directed_f1", "skeleton_f1", "orientation_accuracy", "orientation_recall"]


# --------------------------------------------------------------------------- #
# shards
# --------------------------------------------------------------------------- #


def load_best_f1(run: Path, reps, ps):
    out = {}
    for f in sorted(glob.glob(str(run / "s1_shards" / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["c_choice_names"]]
        for i in range(len(d["p"])):
            p, rep = int(d["p"][i]), int(d["rep"][i])
            if (reps is not None and rep >= reps) or (ps is not None and p not in ps):
                continue
            mt = np.zeros((p, p))
            mt[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
            mh = np.zeros((p, p))
            mh[d["m_best_f1_i"][i], d["m_best_f1_j"][i]] = d["m_best_f1_v"][i]
            out[(p, int(d["k"][i]), names[int(d["c_choice"][i])], rep)] = (mt, mh)
    return out


def rows_from_shards(label, data):
    rows = []
    for key, (mt, mh) in data.items():
        b = orientation_breakdown(mh, mt)
        b["directed_f1"] = confusion(mh, mt).f1
        rows.append({"run": label, "p": key[0], "k": key[1], "c_choice": key[2], "rep": key[3], **b})
    return rows


# --------------------------------------------------------------------------- #
# refits
# --------------------------------------------------------------------------- #


def refit_one(args):
    p, k, c, rep, loss, penalties, cfg = args
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(CChoice(c)), rep])
    m_true, _, _, sigma_hat = draw_instance(p, k, cfg.n_obs, CChoice(c), rng,
                                            metzler=cfg.metzler, standardize=cfg.standardize)
    c_est = 2.0 * np.eye(p)
    rows = []
    for pen in penalties:
        kw = dict(n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, penalty=pen, tol=cfg.tol)
        if loss == "direct":
            kw.update(solver=cfg.solver, convention=cfg.convention)
        path = fit_path(sigma_hat, c_est, loss=loss, **kw)
        ev, sk = evaluate_path(path.estimates, m_true), evaluate_skeleton_path(path.estimates, m_true)
        i_dir, i_sk = int(np.nanargmax(ev["f1"])), int(np.nanargmax(sk["f1"]))
        b_dir = orientation_breakdown(path.estimates[i_dir], m_true)
        b_sk = orientation_breakdown(path.estimates[i_sk], m_true)
        rows.append({
            "run": pen, "loss": loss, "p": p, "k": k, "c_choice": c, "rep": rep,
            "max_f1": ev["max_f1"], "auc": ev["auc"], "aupr": ev["aupr"],
            "skel_max_f1": sk["max_f1"], "skel_auc": sk["auc"], "skel_aupr": sk["aupr"],
            "directed_f1": confusion(path.estimates[i_dir], m_true).f1,
            **{f"{k_}": b_dir[k_] for k_ in CATS + ["skeleton_f1", "orientation_accuracy", "orientation_recall"]},
            **{f"atskel_{k_}": b_sk[k_] for k_ in ("correct", "reversed", "hedged", "skeleton_f1", "orientation_accuracy")},
        })
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def report(rows, runs, extra_cols=()):
    by = defaultdict(list)
    for r in rows:
        by[(r["run"], r["p"], r["c_choice"])].append(r)
    cols = list(extra_cols) + SUMMARY + ["correct", "reversed", "hedged", "both", "half",
                                          "missed_single", "missed_double", "fp_single", "fp_double"]
    print(f"{'p':>3} {'C choice':<18} {'run':<6}" + "".join(f"{c:>{max(8, len(c) + 1)}}" for c in cols))
    for (p, c) in sorted({(r["p"], r["c_choice"]) for r in rows}):
        for run in runs:
            g = by.get((run, p, c), [])
            if not g:
                continue
            vals = [np.nanmean([float(r[col]) for r in g]) for col in cols]
            print(f"{p:>3} {c:<18} {run:<6}" + "".join(f"{v:>{max(8, len(col) + 1)}.3f}" for v, col in zip(vals, cols)))
    # paired differences against the first run
    base = runs[0]
    keyed = {(r["run"], r["p"], r["k"], r["c_choice"], r["rep"]): r for r in rows}
    print(f"\npaired differences vs {base} (mean ± se, z), pooled over C choices:")
    for run in runs[1:]:
        for p in sorted({r["p"] for r in rows}):
            diffs = defaultdict(list)
            for (rn, pp, k, c, rep), r in keyed.items():
                if rn != run or pp != p:
                    continue
                b = keyed.get((base, pp, k, c, rep))
                if b is None:
                    continue
                for col in list(extra_cols) + SUMMARY:
                    x, y = float(r[col]), float(b[col])
                    if np.isfinite(x) and np.isfinite(y):
                        diffs[col].append(x - y)
            parts = []
            for col in list(extra_cols) + SUMMARY:
                d = np.array(diffs[col])
                if len(d) > 1:
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    parts.append(f"{col} {d.mean():+.3f}±{se:.3f} (z {d.mean() / se if se > 0 else np.nan:+.1f})")
            print(f"  {run:<6} p={p}: " + "; ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", help="label=run directory, first one is the baseline")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--loss", default="direct", choices=["direct", "loglik", "frobenius"])
    ap.add_argument("--penalties", nargs="+", default=["lasso", "MCP", "SCAD"])
    ap.add_argument("--c", nargs="+", default=[c.value for c in CChoice])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--p", type=int, nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.shards:
        rows, runs = [], []
        for spec in args.shards:
            label, _, path = spec.rpartition("=")
            runs.append(label)
            rows += rows_from_shards(label, load_best_f1(Path(path), args.reps, args.p))
        print(f"best-F1 estimates from the shards; {len(rows)} rows\n")
        report(rows, runs)
    if args.refit:
        cfg = S1Config()
        ps = args.p or [10]
        reps = args.reps or 10
        tasks = [(p, k, c, r, args.loss, args.penalties, cfg)
                 for p in ps for k in args.k for c in args.c for r in range(reps)]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for rs in ex.map(refit_one, tasks) for r in rs]
        print(f"\nrefit: loss={args.loss}, {len(tasks)} datasets, penalties {args.penalties}; "
              f"metrics along the whole path (directed and skeleton) and the breakdown at the directed best-F1 point\n")
        report(rows, args.penalties, extra_cols=("max_f1", "auc", "aupr", "skel_max_f1", "skel_auc", "skel_aupr"))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
