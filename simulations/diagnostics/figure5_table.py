#!/usr/bin/env python3
"""Dettling's whole-path comparison between the estimators at one C (15 October 2026).  Numbers only.

Dettling, Drton & Kolar (2024) never reduce a path to one graph: their Figure 5 reports four summaries of the
whole path of 100 estimates -- the largest accuracy and the largest F1 over the path (the best graph the path
contains) and the areas under the ROC and the precision-recall curves it traces.  The campaign stores the
four numbers per graph and estimator (``campaign_per_dataset.csv`` of ``campaign.py``).  This script averages
them per (C, n, true C, p, estimator), pooled over the four settings of the true C as well (``true_c = "all"``,
400 graphs per p), and pairs every estimator with the lasso under the same C on the same graphs: the mean
gain, its standard error and z.  It writes ``<root>/campaign_figure5.csv`` (long format, one row per metric)
and prints, for one C, n and true-C setting, the lasso's mean and the gains of the other estimators over p.
The figures are drawn by ``plot_figure5.py``.

    python simulations/diagnostics/figure5_table.py                      # prints C = 2I, n = 1000, pooled
    python simulations/diagnostics/figure5_table.py --c Cresc --n 1e4 --true-c C_Random_Full
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import ID, mean_se, paired  # noqa: E402

ESTS = ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")
#: Dettling's four summaries of the whole path
METRICS = ("max_acc", "max_f1", "auc", "aupr")
PS = (10, 15, 20, 25, 30, 40, 50)
#: the pooled setting first, then Dettling's four settings of the true C
SETTINGS = ("all", "C_ID", "C_Random_Diag", "C_Random_Min_Diag", "C_Random_Full")
NS = ("1000", "1e4", "inf")
COLUMNS = ("c", "n", "true_c", "p", "estimator", "graphs", "metric", "mean", "se", "gain", "gain_se", "z")
TRUE_C = ID.index("c_choice")


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def table(rows, c="C2I", n="1000", ests=ESTS, metrics=METRICS, ps=PS, settings=SETTINGS) -> list[dict]:
    """One row per (true-C setting, p, estimator, metric) of the direct-loss cells of the thesis generator at
    C ``c`` and sample size ``n``: the mean over the graphs with its standard error, and the paired gain over
    the lasso on the same graphs with its standard error and z (``paired`` of ``campaign.py``; the lasso's
    own row has gain 0, an estimator without a lasso partner an empty gain)."""
    by = defaultdict(dict)                                    # (p, estimator) -> graph id -> row
    for r in rows:
        if (r["loss"] == "direct" and r["c"] == c and r["n"] == n and not r.get("dgp") and r["estimator"] in ests
                and int(r["p"]) in ps):
            by[(int(r["p"]), r["estimator"])][tuple(r[k] for k in ID)] = r
    out = []
    for setting in settings:
        for p in ps:
            ref = by.get((p, "lasso"), {})
            for est in ests:
                cell = by.get((p, est))
                if not cell:
                    continue
                ids = [g for g in cell if setting in ("all", g[TRUE_C])]
                shared = [g for g in ids if g in ref]
                if not ids:
                    continue
                for metric in metrics:
                    mean, se, k = mean_se([f(cell[g].get(metric)) for g in ids])
                    stat = paired([f(cell[g].get(metric)) for g in shared], [f(ref[g].get(metric)) for g in shared])
                    out.append({"c": c, "n": n, "true_c": setting, "p": p, "estimator": est, "graphs": k,
                                "metric": metric, "mean": mean, "se": se, "gain": stat["diff"],
                                "gain_se": stat["se"], "z": stat["z"]})
    return out


def print_table(out: list[dict], metric: str, true_c: str = "all", ests=ESTS) -> None:
    """The lasso's mean and the paired gain of every other estimator over it (± its standard error), over p."""
    cells = {(r["p"], r["estimator"]): r for r in out if r["metric"] == metric and r["true_c"] == true_c}
    if not cells:
        return
    ps = sorted({p for p, _ in cells})
    others = [e for e in ests if e != "lasso" and any((p, e) in cells for p in ps)]
    first = next(iter(cells.values()))
    print(f"\n{metric}: {first['c']}, n = {first['n']}, true C = {true_c} -- the lasso's mean and the paired gain of "
          f"each estimator over it (± standard error)")
    print(f"  {'p':<4}{'graphs':>7}{'lasso':>8}" + "".join(f"{e:>20}" for e in others))
    for p in ps:
        lasso = cells.get((p, "lasso"))
        graphs = (lasso or next(r for (q, _), r in cells.items() if q == p))["graphs"]
        vals = "".join(f"{cells[(p, e)]['gain']:+13.3f} ±{cells[(p, e)]['gain_se']:.3f}"
                       if (p, e) in cells and math.isfinite(cells[(p, e)]["gain"]) else f"{'':>20}" for e in others)
        print(f"  {p:<4}{graphs:>7}{(lasso['mean'] if lasso else math.nan):>8.3f}{vals}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--c", default="C2I", choices=("C2I", "Cresc"), help="the C printed (the table holds both)")
    ap.add_argument("--n", default="1000", choices=NS, help="the n printed (the table holds all three)")
    ap.add_argument("--true-c", default="all", choices=SETTINGS, help="the true-C setting printed")
    args = ap.parse_args()
    rows = list(csv.DictReader((args.root / "campaign_per_dataset.csv").open()))
    out = [r for c in ("C2I", "Cresc") for n in NS for r in table(rows, c, n)]
    with (args.root / "campaign_figure5.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    shown = [r for r in out if r["c"] == args.c and r["n"] == args.n]
    for metric in METRICS:
        print_table(shown, metric, args.true_c)
    print(f"\nwrote {args.root / 'campaign_figure5.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
