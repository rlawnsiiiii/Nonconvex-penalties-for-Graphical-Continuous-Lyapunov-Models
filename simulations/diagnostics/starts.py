#!/usr/bin/env python3
"""E1 and E2 of the plan of 13 October 2026 (next_steps/131026/plan_mechanism_131026.md §2.5, §3.3):
which dense start, and pruning by thresholding against the adaptive lasso's reweighting, from
``campaign_per_dataset.csv`` (``campaign.py``; the thesis generator's cells only, ``dgp == ""``).

E1.  The arms ``direct_MCP-up-<start>_<C>_n<n>`` and ``direct_adaptive-<start>_<C>_n<n>`` of wave 9
next to wave 1's ``MCP-up`` and ``adaptive`` (the ``lasso`` start) and the lasso itself.  Per
(p, C, n, pruner, start) and metric: the mean, the paired gain over the lasso and the paired gain
over the same pruner started from the lasso's dense end -> ``campaign_starts.csv``.  What the start
contains (the ``start_*`` columns): the F1 of the start's support at 1e-8, the shares of the true
single edges whose true direction is present alone / together with its reverse / only as the
reverse / not at all (the dense-start content of the 7 October note), its orientation recall, the
norms, the Lyapunov residual -> ``campaign_start_content.csv``.

E2.  ``threshold`` and ``threshold-norefit`` against ``adaptive``, ``MCP-up`` and ``lasso`` per
(p, C, n) and metric, with the paired difference to the thresholded lasso (adaptive - threshold is
the number the plan reads) -> ``campaign_threshold.csv``.

Numbers only; the figures are drawn by ``plot_starts.py``.

    python simulations/diagnostics/starts.py                  # runs/campaign
    python simulations/diagnostics/starts.py --root some/dir
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import paired  # noqa: E402

STARTS = ("lasso", "exact", "l1", "l2", "truth0.05", "truth0.2")
PRUNERS = ("MCP-up", "adaptive")
METRICS = ("max_f1", "aupr", "bic_f1", "search_f1")
E2 = ("threshold", "threshold-norefit", "adaptive", "MCP-up", "lasso")
#: the start's own diagnostics (means over graphs) and the orientation content of its support
CONTENT = ("start_f1", "start_precision", "start_recall", "start_l1", "start_l2", "start_nnz", "start_residual")
SHARES = (("true_alone", "start_correct"), ("both", "start_hedged"), ("reverse_alone", "start_reversed"),
          ("neither", "start_missed_single"))
STARTS_COLUMNS = ("p", "c", "n", "pruner", "start", "estimator", "n_graphs", "metric", "mean",
                  "gain_lasso", "gain_lasso_se", "gain_lasso_z", "pairs_lasso",
                  "gain_lasso_start", "gain_lasso_start_se", "gain_lasso_start_z", "pairs_lasso_start")
CONTENT_COLUMNS = ("p", "c", "n", "pruner", "start", "estimator", "n_graphs", *CONTENT,
                   *(s for s, _ in SHARES), "orientation_recall")
THRESHOLD_COLUMNS = ("p", "c", "n", "estimator", "n_graphs", "metric", "mean",
                     "diff_vs_threshold", "diff_se", "diff_z", "pairs")


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def arm(estimator: str):
    """``(pruner, start)`` of an E1 arm: ``MCP-up`` / ``adaptive`` are the lasso start,
    ``MCP-up-l1`` is ``("MCP-up", "l1")``; None for anything else."""
    for pruner in PRUNERS:
        if estimator == pruner:
            return pruner, "lasso"
        if estimator.startswith(pruner + "-") and estimator[len(pruner) + 1:] in STARTS:
            return pruner, estimator[len(pruner) + 1:]
    return None


def gid(r: dict) -> tuple:
    return (r["p"], r["k"], r["c_choice"], r["rep"])


def load_rows(path: Path) -> list[dict]:
    """The direct-loss rows of the thesis generator (the knob cells are E3's and E4's)."""
    return [r for r in csv.DictReader(path.open()) if r["loss"] == "direct" and not r.get("dgp")]


def by_cell(rows: list[dict]) -> dict:
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["p"], r["c"], r["n"], r["estimator"])][gid(r)] = r
    return cells


def _mean(values) -> float:
    v = np.asarray([x for x in values if not math.isnan(x)], float)
    return float(v.mean()) if v.size else math.nan


def _gain(cell: dict, ref: dict, metric: str) -> dict:
    ids = [g for g in cell if g in ref]
    return paired([f(cell[g][metric]) for g in ids], [f(ref[g][metric]) for g in ids])


def _sort_key(key):
    p, c, n, est = key[:4]
    return (int(p), c, math.inf if n == "inf" else float(n), est)


def starts_table(rows: list[dict]) -> list[dict]:
    """E1: per (p, C, n, pruner, start) and metric, the mean and the paired gains over the lasso and
    over the pruner's lasso-start arm (zero for that arm itself)."""
    cells = by_cell(rows)
    out = []
    for key, cell in sorted(cells.items(), key=lambda kv: _sort_key(kv[0])):
        a = arm(key[3])
        if a is None:
            continue
        p, c, n, est = key
        pruner, start = a
        lasso, base = cells.get((p, c, n, "lasso"), {}), cells.get((p, c, n, pruner), {})
        for metric in METRICS:
            row = {"p": p, "c": c, "n": n, "pruner": pruner, "start": start, "estimator": est,
                   "n_graphs": len(cell), "metric": metric, "mean": _mean(f(r[metric]) for r in cell.values())}
            for name, ref in (("lasso", lasso), ("lasso_start", base)):
                st = _gain(cell, ref, metric)
                row.update({f"gain_{name}": st["diff"], f"gain_{name}_se": st["se"], f"gain_{name}_z": st["z"],
                            f"pairs_{name}": st["pairs"]})
            out.append(row)
    return out


def content_table(rows: list[dict]) -> list[dict]:
    """What each start contains, per (p, C, n, pruner, start): means of the start columns and the
    orientation content of its support among the true single edges."""
    cells = by_cell(rows)
    out = []
    for key, cell in sorted(cells.items(), key=lambda kv: _sort_key(kv[0])):
        a = arm(key[3])
        if a is None or not any("start_f1" in r for r in cell.values()):
            continue
        p, c, n, est = key
        rs = [r for r in cell.values() if r.get("start_f1") not in ("", None)]
        row = {"p": p, "c": c, "n": n, "pruner": a[0], "start": a[1], "estimator": est, "n_graphs": len(rs)}
        for col in CONTENT:
            row[col] = _mean(f(r[col]) for r in rs)
        single = [sum(f(r[src]) for _, src in SHARES) for r in rs]
        for name, src in SHARES:
            row[name] = _mean(f(r[src]) / s for r, s in zip(rs, single) if s > 0)
        row["orientation_recall"] = _mean((f(r["start_correct"]) + f(r["start_hedged"])) / s
                                          for r, s in zip(rs, single) if s > 0)
        out.append(row)
    return out


def threshold_table(rows: list[dict]) -> list[dict]:
    """E2: the five estimators per (p, C, n) and metric, each with its paired difference to the
    thresholded lasso (``threshold``) on the same graphs."""
    cells = by_cell(rows)
    keys = sorted({k[:3] for k in cells if k[3] in E2}, key=lambda t: _sort_key((*t, "")))
    out = []
    for p, c, n in keys:
        thr = cells.get((p, c, n, "threshold"), {})
        for est in E2:
            cell = cells.get((p, c, n, est))
            if not cell:
                continue
            for metric in METRICS:
                st = _gain(cell, thr, metric) if est != "threshold" else {"diff": 0.0, "se": 0.0, "z": math.nan,
                                                                          "pairs": len(cell)}
                out.append({"p": p, "c": c, "n": n, "estimator": est, "n_graphs": len(cell), "metric": metric,
                            "mean": _mean(f(r[metric]) for r in cell.values()), "diff_vs_threshold": st["diff"],
                            "diff_se": st["se"], "diff_z": st["z"], "pairs": st["pairs"]})
    return out


def print_starts(out: list[dict], metric: str = "max_f1") -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out}, key=lambda t: _sort_key((*t, "")))
    for p, c, n in keys:
        rows = {(r["start"], r["pruner"]): r for r in out if (r["p"], r["c"], r["n"], r["metric"]) == (p, c, n, metric)}
        if not rows:
            continue
        print(f"\np = {p}, {c}, n = {n}: {metric}, mean and paired gain over the lasso (z), by start and pruner")
        print(f"  {'start':<11}" + "".join(f"{pr:>30}" for pr in PRUNERS))
        for start in STARTS:
            cells = [rows.get((start, pr)) for pr in PRUNERS]
            if all(r is None for r in cells):
                continue
            print(f"  {start:<11}" + "".join(
                f"{r['mean']:>10.3f} {r['gain_lasso']:+8.3f} (z {r['gain_lasso_z']:5.1f})" if r and not math.isnan(r["gain_lasso"])
                else f"{r['mean']:>10.3f}{'':>20}" if r else f"{'':>30}" for r in cells))


def print_content(out: list[dict]) -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out}, key=lambda t: _sort_key((*t, "")))
    for p, c, n in keys:
        rows = [r for r in out if (r["p"], r["c"], r["n"]) == (p, c, n)]
        print(f"\np = {p}, {c}, n = {n}: what the start contains (true single edges: true direction alone / both "
              f"directions / only the reverse / neither), its F1, nonzeros and residual")
        print(f"  {'pruner':<9}{'start':<11}{'alone':>7}{'both':>7}{'rev.':>7}{'none':>7}{'F1':>7}{'nnz':>7}{'resid':>9}")
        for r in rows:
            print(f"  {r['pruner']:<9}{r['start']:<11}{r['true_alone']:>7.2f}{r['both']:>7.2f}{r['reverse_alone']:>7.2f}"
                  f"{r['neither']:>7.2f}{r['start_f1']:>7.3f}{r['start_nnz']:>7.1f}{r['start_residual']:>9.2e}")


def print_threshold(out: list[dict], metric: str = "max_f1") -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out}, key=lambda t: _sort_key((*t, "")))
    for p, c, n in keys:
        rows = {r["estimator"]: r for r in out if (r["p"], r["c"], r["n"], r["metric"]) == (p, c, n, metric)}
        if "threshold" not in rows:
            continue
        print(f"\np = {p}, {c}, n = {n}: {metric}, mean and paired difference to the thresholded lasso (z)")
        for est in E2:
            r = rows.get(est)
            if r is None:
                continue
            d = f"{r['diff_vs_threshold']:+8.3f} (z {r['diff_z']:5.1f}, {r['pairs']} pairs)" if not math.isnan(r["diff_z"]) else ""
            print(f"  {est:<20}{r['mean']:>8.3f}  {d}")


def write(path: Path, rows: list[dict], columns) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    args = ap.parse_args()
    rows = load_rows(args.root / "campaign_per_dataset.csv")
    if not rows:
        raise SystemExit(f"no direct-loss rows under {args.root}")
    starts, content, threshold = starts_table(rows), content_table(rows), threshold_table(rows)
    write(args.root / "campaign_starts.csv", starts, STARTS_COLUMNS)
    write(args.root / "campaign_start_content.csv", content, CONTENT_COLUMNS)
    write(args.root / "campaign_threshold.csv", threshold, THRESHOLD_COLUMNS)
    for metric in ("max_f1", "bic_f1", "search_f1"):
        print_starts(starts, metric)
    print_content(content)
    for metric in ("max_f1", "aupr", "bic_f1", "search_f1"):
        print_threshold(threshold, metric)
    print(f"\nwrote campaign_starts.csv ({len(starts)} rows), campaign_start_content.csv ({len(content)}), "
          f"campaign_threshold.csv ({len(threshold)}) under {args.root}")


if __name__ == "__main__":
    main()
