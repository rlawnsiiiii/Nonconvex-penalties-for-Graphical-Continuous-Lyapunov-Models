#!/usr/bin/env python3
"""Merge the Figure 5 shards into tidy tables.  Numbers only, no plots.

Reads every ``shard_*.npz`` written by ``run_s1_shard.py`` and produces:

  s1_per_dataset.csv   one row per dataset: the four reported metrics, plus
                       lambda_max, n_true_edges, runtime.  Metrics are computed
                       here from the stored confusion counts, both off-diagonal
                       and including the diagonal.
  s1_summary.csv       Figure 5 itself: mean and standard error of each metric
                       per (p, C_choice), averaged over k and the replicates.
  s1_curves.csv        mean tpr/fpr/precision/nnz per (p, C_choice, lambda
                       index) -- enough to redraw ROC/PR curves.

Run on the cluster login node or locally; it is cheap either way.

    python simulations/aggregate_s1.py --in-dir runs/s1_dettling_reproduction/s1_shards

writes the CSVs next to the shard folder (``--out-dir`` defaults to its parent).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gclm.metrics import auc_roc, aupr

METRICS = ("max_acc", "max_f1", "auc", "aupr")


def metrics_from_counts(counts: np.ndarray, anchor_dense: bool = True) -> dict[str, float]:
    """Definitions G.4/G.5 from a (n_lambda, 4) array of tp, fp, tn, fn.

    Same conventions as ``gclm.metrics``: tpr := 1 when there are no true
    edges, fpr := 1 when there are no true non-edges, precision := 1 when
    nothing is selected, f1 := 0 when undefined.

    ``anchor_dense`` prepends the fully dense point (tpr = fpr = 1) -- see
    ``gclm.metrics.evaluate_path`` for why the saturating path makes this
    necessary.  ``max_acc`` and ``max_f1`` are computed *before* the anchor, so
    the anchor can only affect the curve areas.
    """
    tp, fp, tn, fn = counts.T.astype(float)
    tpr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), 1.0)
    total = tp + tn + fp + fn
    acc = np.where(total > 0, (tp + tn) / np.maximum(total, 1), np.nan)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)

    # curve inputs, optionally extended by the dense anchor; the per-lambda
    # arrays returned below stay at grid length (the anchor is not a fit), and
    # max_acc / max_f1 are maxima over the grid only
    ctpr, cfpr, cprec = tpr, fpr, prec
    if anchor_dense and len(counts):
        total = tp[0] + fn[0] + fp[0] + tn[0]
        base = 1.0 if total == 0 else (tp[0] + fn[0]) / total
        ctpr = np.concatenate(([1.0], tpr))
        cfpr = np.concatenate(([1.0], fpr))
        cprec = np.concatenate(([base], prec))

    return {
        "max_acc": float(np.nanmax(acc)), "max_f1": float(np.max(f1)),
        "auc": auc_roc(cfpr, ctpr), "aupr": aupr(ctpr, cprec),
        "_tpr": tpr, "_fpr": fpr, "_prec": prec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: the parent of --in-dir, i.e. the run directory")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = args.in_dir.parent
    shards = sorted(args.in_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shard_*.npz under {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, curves = [], defaultdict(list)
    names, config = None, None
    n_expected = None

    for f in shards:
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["c_choice_names"]]
        config = json.loads(str(d["config_json"]))
        if n_expected is None:
            n_expected = (len(config["p_values"]) * len(config["k_values"])
                          * len(config["c_choices"]) * config["n_rep"])
        for i in range(len(d["p"])):
            m_off = metrics_from_counts(d["conf_offdiag"][i])
            m_inc = metrics_from_counts(d["conf_incdiag"][i])
            key = (int(d["p"][i]), names[int(d["c_choice"][i])])
            rows.append({
                "p": int(d["p"][i]), "k": int(d["k"][i]),
                "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i]),
                **{k: m_off[k] for k in METRICS},
                **{f"{k}_incdiag": m_inc[k] for k in METRICS},
                "lambda_max": float(d["lambda_max"][i]),
                "n_true_edges": int(d["n_true_edges"][i]),
                "seconds": float(d["seconds"][i]),
            })
            curves[key].append((m_off["_tpr"], m_off["_fpr"], m_off["_prec"],
                                d["nnz"][i]))

    per = args.out_dir / "s1_per_dataset.csv"
    with per.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Figure 5: average over k and reps -> mean +- standard error per (p, C)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["p"], r["c_choice"])].append(r)
    summ = args.out_dir / "s1_summary.csv"
    with summ.open("w", newline="") as fh:
        cols = (["p", "c_choice", "n_datasets"]
                + [f"{m}{s}" for m in METRICS for s in ("", "_se")]
                + [f"{m}_incdiag" for m in METRICS])
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for (p, c), g in sorted(agg.items()):
            row = {"p": p, "c_choice": c, "n_datasets": len(g)}
            for m in METRICS:
                v = np.array([x[m] for x in g], float)
                row[m] = float(v.mean())
                row[f"{m}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                row[f"{m}_incdiag"] = float(np.mean([x[f"{m}_incdiag"] for x in g]))
            w.writerow(row)

    crv = args.out_dir / "s1_curves.csv"
    with crv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p", "c_choice", "lambda_index", "tpr", "fpr", "precision", "nnz"])
        for (p, c), vals in sorted(curves.items()):
            tpr = np.mean([v[0] for v in vals], axis=0)
            fpr = np.mean([v[1] for v in vals], axis=0)
            prec = np.mean([v[2] for v in vals], axis=0)
            nnz = np.mean([v[3] for v in vals], axis=0)
            for j in range(len(tpr)):
                w.writerow([p, c, j, tpr[j], fpr[j], prec[j], nnz[j]])

    total = sum(r["seconds"] for r in rows)
    print(f"shards      : {len(shards)}")
    print(f"datasets    : {len(rows)}" +
          (f"  (expected {n_expected})" if n_expected else ""))
    if n_expected and len(rows) != n_expected:
        print(f"  !! WARNING: {n_expected - len(rows)} datasets missing -- "
              f"check for failed array tasks")
    print(f"compute     : {total/3600:.1f} CPU-hours")
    print(f"wrote       : {per}\n              {summ}\n              {crv}")


if __name__ == "__main__":
    main()
