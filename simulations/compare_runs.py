#!/usr/bin/env python3
"""Paired comparison of two runs on identical datasets.

Every run in this repository draws dataset ``(p, k, C, rep)`` from the same seeded
stream, so two runs that differ only in penalty or loss are *paired*: the right
summary is the per-dataset difference of each metric, averaged, with its own
standard error -- far tighter than comparing two independent means.

    python simulations/compare_runs.py \\
        --baseline runs/s1_dettling_reproduction --run runs/s1b_pilot_p10-20/MCP \\
        --reps 25 --p 10 20

prints, per (p, C choice) and metric, mean(run - baseline) +- se and the paired
z-score, plus the fraction of datasets on which the run is better.  ``--csv``
also writes the table.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("max_acc", "max_f1", "auc", "aupr")


def load(run: Path, reps: int | None, ps):
    rows = {}
    for r in csv.DictReader((run / "s1_per_dataset.csv").open()):
        key = (int(r["p"]), int(r["k"]), r["c_choice"], int(r["rep"]))
        if (reps is None or key[3] < reps) and (ps is None or key[0] in ps):
            rows[key] = {m: float(r[m]) for m in METRICS}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--p", type=int, nargs="+", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    base = load(args.baseline, args.reps, args.p)
    run = load(args.run, args.reps, args.p)
    common = sorted(set(base) & set(run))
    if not common:
        raise SystemExit("no datasets in common")
    missing = len(run) - len(common)
    print(f"{len(common)} paired datasets" + (f" ({missing} in --run without a baseline)" if missing else ""))

    groups = defaultdict(list)
    for key in common:
        groups[(key[0], key[2])].append(key)

    out = []
    print(f"\n{'p':>3} {'C choice':<18} {'metric':<8} {'baseline':>9} {'run':>8} {'diff':>8} {'se':>7} {'z':>6} {'run better':>11}")
    for (p, c), keys in sorted(groups.items()):
        for m in METRICS:
            b = np.array([base[k][m] for k in keys])
            r = np.array([run[k][m] for k in keys])
            d = r - b
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            z = d.mean() / se if se > 0 else np.nan
            better = float(np.mean(d > 0))
            print(f"{p:>3} {c:<18} {m:<8} {b.mean():>9.3f} {r.mean():>8.3f} {d.mean():>+8.3f} {se:>7.3f} {z:>+6.1f} {better:>11.2f}")
            out.append({"p": p, "c_choice": c, "metric": m, "n": len(d), "baseline": b.mean(),
                        "run": r.mean(), "diff": d.mean(), "se": se, "z": z, "frac_run_better": better})
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
