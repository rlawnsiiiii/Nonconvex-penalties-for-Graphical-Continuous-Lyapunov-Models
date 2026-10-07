#!/usr/bin/env python3
"""The n-sweep (runs/nsweep_p10-20, simulations/S2b_nsweep.md): validation against
the local runs, paired comparisons, orientation breakdown, convergence audit, cost.

Reads the 36 cells ``<loss>_<penalty>_n<n>/`` written by the cluster (their
``s1_per_dataset.csv`` and shards) and writes, next to them:

  nsweep_means.csv            mean metric per (loss, penalty, n, p)
  nsweep_paired_vs_lasso.csv  penalty - lasso on the same loss and n, per p, and per (p, C choice)
  nsweep_paired_vs_direct.csv loss - direct loss for the same penalty and n, per p
  nsweep_paired_vs_n1000.csv  n - (n = 1000) for the same loss and penalty, per p
  nsweep_orientation.csv      best-F1 point: directed / skeleton F1, orientation, pair categories
  nsweep_audit.csv            per cell: CPU-h, lambdas at the step cap, first-order violation
  nsweep_validation.txt       the cells that repeat local runs, compared dataset by dataset

    python simulations/diagnostics/nsweep.py            # everything
    python simulations/diagnostics/plot_nsweep.py       # the figures

All comparisons are paired: every cell holds the same 800 drift matrices (the
data-generating seed depends on (p, k, C choice, rep) only).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gclm.metrics import confusion, orientation_breakdown  # noqa: E402
from orientation import load_best_f1  # noqa: E402

LOSSES = ("direct", "loglik", "frobenius")
PENALTIES = ("lasso", "MCP", "SCAD")
NS = ("1000", "1e4", "1e5", "inf")
METRICS = ("max_f1", "auc", "aupr", "max_acc")
ORIENT = ("directed_f1", "skeleton_f1", "orientation_accuracy", "orientation_recall",
          "correct", "reversed", "hedged", "both", "half", "missed_single", "missed_double",
          "fp_single", "fp_double")
DEFAULT = ROOT / "runs" / "nsweep_p10-20"


def paired(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean, standard error and z of the paired difference ``x - y``, and the
    share of pairs with ``x > y``.  NaNs (in either) are dropped pairwise."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    n = len(d)
    if n == 0:
        return {"pairs": 0, "diff": math.nan, "se": math.nan, "z": math.nan, "better": math.nan}
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
    z = d.mean() / se if n > 1 and se > 0 else (0.0 if n > 1 and d.mean() == 0 else math.nan)
    return {"pairs": n, "diff": float(d.mean()), "se": float(se), "z": float(z),
            "better": float(np.mean(d > 0))}


def load_cells(root: Path) -> dict:
    """``cells[(loss, pen, n)][(p, k, c, rep)] = {metric: value}``."""
    cells = {}
    for loss in LOSSES:
        for pen in PENALTIES:
            for n in NS:
                f = root / f"{loss}_{pen}_n{n}" / "s1_per_dataset.csv"
                if not f.exists():
                    continue
                rows = {}
                for r in csv.DictReader(f.open()):
                    rows[(int(r["p"]), int(r["k"]), r["c_choice"], int(r["rep"]))] = {
                        m: float(r[m]) for m in METRICS + ("seconds",)}
                cells[(loss, pen, n)] = rows
    return cells


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def paired_rows(a: dict, b: dict, base: dict, groups=("p",)) -> list[dict]:
    """Paired differences ``a - b`` per metric, grouped by p (and C choice)."""
    out = []
    keys = sorted(set(a) & set(b))
    def gkey(k):
        return tuple({"p": k[0], "c_choice": k[2]}[g] for g in groups)
    for g in sorted({gkey(k) for k in keys}):
        ks = [k for k in keys if gkey(k) == g]
        for m in METRICS:
            st = paired([a[k][m] for k in ks], [b[k][m] for k in ks])
            out.append({**base, **dict(zip(groups, g)), "metric": m,
                        "mean_a": float(np.mean([a[k][m] for k in ks])),
                        "mean_b": float(np.mean([b[k][m] for k in ks])), **st})
    return out


def validation(cells: dict, root: Path) -> list[str]:
    """Cells that repeat local runs, dataset by dataset."""
    lines = []

    def compare(label, cell, ref_rows, metrics=("max_f1", "auc", "aupr")):
        keys = sorted(set(cell) & set(ref_rows))
        if not keys:
            lines.append(f"{label}: no common datasets")
            return
        d = np.array([[abs(cell[k][m] - ref_rows[k][m]) for m in metrics] for k in keys])
        differ = int(np.sum(d.max(axis=1) > 1e-9))
        lines.append(f"{label}: {len(keys)} datasets, {differ} differ, max |diff| "
                     + ", ".join(f"{m} {d[:, i].max():.1e}" for i, m in enumerate(metrics))
                     + f"; mean |diff| max_f1 {d[:, 0].mean():.1e}")

    def per_dataset(path, reps, ps):
        rows = {}
        if not path.exists():
            return rows
        for r in csv.DictReader(path.open()):
            key = (int(r["p"]), int(r["k"]), r["c_choice"], int(r["rep"]))
            if key[3] < reps and key[0] in ps:
                rows[key] = {m: float(r[m]) for m in ("max_f1", "auc", "aupr")}
        return rows

    runs = ROOT / "runs"
    for pen, ref in (("lasso", runs / "s1_dettling_reproduction"), ("MCP", runs / "s1b_pilot_p10-20" / "MCP"),
                     ("SCAD", runs / "s1b_pilot_p10-20" / "SCAD")):
        if ("direct", pen, "1000") in cells:
            compare(f"direct {pen}, n = 1000 vs {ref.relative_to(ROOT)}", cells[("direct", pen, "1000")],
                    per_dataset(ref / "s1_per_dataset.csv", 25, (10, 20)))
    for loss in ("loglik", "frobenius"):
        for pen in PENALTIES:
            ref = runs / "s2_pilot_p10" / f"{loss}_{pen}"
            if (loss, pen, "1000") in cells:
                compare(f"{loss} {pen}, n = 1000 vs {ref.relative_to(ROOT)}", cells[(loss, pen, "1000")],
                        per_dataset(ref / "s1_per_dataset.csv", 10, (10,)))
    pilot = ROOT / "next_steps" / "021026" / "files" / "s1_nsweep_p10.csv"
    if pilot.exists():
        ref = defaultdict(dict)
        for r in csv.DictReader(pilot.open()):
            if r["n"] in ("inf", "Infinity"):
                ref[r["pen"]][(10, int(r["k"]), r["c"], int(r["rep"]))] = {
                    m: float(r[m]) for m in ("max_f1", "auc", "aupr")}
        for pen in PENALTIES:
            if ("direct", pen, "inf") in cells and ref[pen]:
                compare(f"direct {pen}, n = inf, p = 10 vs the local pilot of 2 October",
                        cells[("direct", pen, "inf")], ref[pen])
    return lines


def orientation_rows(root: Path) -> list[dict]:
    out = []
    per_cell = {}
    for loss in LOSSES:
        for pen in PENALTIES:
            for n in NS:
                run = root / f"{loss}_{pen}_n{n}"
                if not (run / "s1_shards").exists():
                    continue
                data = load_best_f1(run, None, None)
                rows = {}
                for key, (mt, mh) in data.items():
                    b = orientation_breakdown(mh, mt)
                    b["directed_f1"] = confusion(mh, mt).f1
                    rows[key] = b
                per_cell[(loss, pen, n)] = rows
    for (loss, pen, n), rows in per_cell.items():
        base = per_cell.get((loss, "lasso", n), {})
        for p in sorted({k[0] for k in rows}):
            ks = [k for k in rows if k[0] == p]
            row = {"loss": loss, "penalty": pen, "n": n, "p": p, "datasets": len(ks)}
            for col in ORIENT:
                row[col] = float(np.nanmean([rows[k][col] for k in ks]))
            for col in ("directed_f1", "skeleton_f1", "orientation_accuracy", "orientation_recall"):
                st = paired([rows[k][col] for k in ks], [base[k][col] for k in ks]) if base else {}
                row[f"{col}_minus_lasso"] = st.get("diff", math.nan)
                row[f"{col}_z"] = st.get("z", math.nan)
            out.append(row)
    return out


def audit_rows(root: Path) -> list[dict]:
    out = []
    for loss in LOSSES:
        for pen in PENALTIES:
            for n in NS:
                run = root / f"{loss}_{pen}_n{n}"
                files = sorted(glob.glob(str(run / "s1_shards" / "*.npz")))
                if not files:
                    continue
                secs, by_p, n_lam, capped, kkts, kkt_capped, walls, hosts = 0.0, defaultdict(list), 0, 0, [], [], [], set()
                for f in files:
                    z = np.load(f, allow_pickle=True)
                    for p, s in zip(z["p"], z["seconds"]):
                        by_p[int(p)].append(float(s))
                    secs += float(np.sum(z["seconds"]))
                    pv = json.loads(str(z["provenance_json"]))
                    walls.append(float(pv["wall_seconds"]))
                    hosts.add(pv["host"])
                    n_lam += int(np.asarray(z["lambdas"]).size)
                    if "iterations" in z.files and loss != "direct":
                        it, kk = np.asarray(z["iterations"]), np.asarray(z["kkt"])
                        cap = it >= 50_000
                        capped += int(cap.sum())
                        kkts.append(kk[np.isfinite(kk)].ravel())
                        kkt_capped.append(kk[cap & np.isfinite(kk)].ravel())
                kk = np.concatenate(kkts) if kkts else np.array([])
                kc = np.concatenate(kkt_capped) if kkt_capped else np.array([])
                out.append({
                    "loss": loss, "penalty": pen, "n": n, "shards": len(files), "cpu_hours": secs / 3600,
                    "sec_per_path_p10": float(np.mean(by_p[10])), "sec_per_path_p20": float(np.mean(by_p[20])),
                    "max_sec_per_path": float(max(max(v) for v in by_p.values())),
                    "longest_shard_hours": max(walls) / 3600, "hosts": len(hosts),
                    "lambdas": n_lam, "lambdas_at_step_cap": capped,
                    "share_at_step_cap": capped / n_lam if n_lam else math.nan,
                    "kkt_max": float(kk.max()) if kk.size else math.nan,
                    "kkt_q999": float(np.quantile(kk, 0.999)) if kk.size else math.nan,
                    "kkt_capped_median": float(np.median(kc)) if kc.size else math.nan,
                    "lambdas_kkt_above_1e-4": int(np.sum(kk > 1e-4)) if kk.size else 0,
                    "lambdas_kkt_above_1e-5": int(np.sum(kk > 1e-5)) if kk.size else 0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT)
    ap.add_argument("--skip-shards", action="store_true", help="skip the orientation and audit tables")
    args = ap.parse_args()
    root = args.root
    cells = load_cells(root)
    print(f"{len(cells)} cells, {sum(len(v) for v in cells.values())} datasets in total")

    lines = validation(cells, root)
    (root / "nsweep_validation.txt").write_text("\n".join(lines) + "\n")
    print("\nvalidation:")
    for line in lines:
        print("  " + line)

    means = []
    for (loss, pen, n), rows in cells.items():
        for p in sorted({k[0] for k in rows}):
            ks = [k for k in rows if k[0] == p]
            r = {"loss": loss, "penalty": pen, "n": n, "p": p, "datasets": len(ks)}
            for m in METRICS:
                v = np.array([rows[k][m] for k in ks])
                r[m] = float(v.mean())
                r[f"{m}_se"] = float(v.std(ddof=1) / math.sqrt(len(v)))
            means.append(r)
    write(root / "nsweep_means.csv", means)

    vs_lasso, vs_direct, vs_n = [], [], []
    for (loss, pen, n), rows in cells.items():
        if pen != "lasso" and (loss, "lasso", n) in cells:
            base = {"loss": loss, "penalty": pen, "n": n}
            vs_lasso += [{**r, "c_choice": "all"} for r in paired_rows(rows, cells[(loss, "lasso", n)], base)]
            vs_lasso += paired_rows(rows, cells[(loss, "lasso", n)], base, groups=("p", "c_choice"))
        if loss != "direct" and ("direct", pen, n) in cells:
            vs_direct += paired_rows(rows, cells[("direct", pen, n)], {"loss": loss, "penalty": pen, "n": n})
        if n != "1000" and (loss, pen, "1000") in cells:
            vs_n += paired_rows(rows, cells[(loss, pen, "1000")], {"loss": loss, "penalty": pen, "n": n})
    order = ["loss", "penalty", "n", "p", "c_choice", "metric", "mean_a", "mean_b", "pairs", "diff", "se", "z", "better"]
    vs_lasso = [{c: r[c] for c in order} for r in vs_lasso]
    write(root / "nsweep_paired_vs_lasso.csv", vs_lasso)
    write(root / "nsweep_paired_vs_direct.csv", vs_direct)
    write(root / "nsweep_paired_vs_n1000.csv", vs_n)

    if not args.skip_shards:
        write(root / "nsweep_orientation.csv", orientation_rows(root))
        write(root / "nsweep_audit.csv", audit_rows(root))


if __name__ == "__main__":
    main()
