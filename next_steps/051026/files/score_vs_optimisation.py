#!/usr/bin/env python3
"""Is the gap between the data-driven searches and the search started from the true graph
an optimisation gap or a score gap?  (next_steps/051026/next_steps_051026.md, section 2)

For every graph of the raw-scale S3b runs (runs/s3b_search/random_direct_p10_raw*_rows.csv)
compare the BIC of the graph reached from the TRUE start with the best BIC reached by any
data-driven search (lasso / MCP / SCAD starts at the BIC and at the oracle lambda, pure search):

  truth-start lower   -> the data-driven searches are stuck: a better optimiser would help
  data-driven lower   -> the score itself prefers another graph: a better optimiser would not

Also prints the error decomposition of the main methods.  Writes score_vs_optimisation.csv.

    python next_steps/051026/files/score_vs_optimisation.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "runs" / "s3b_search"
DATA_DRIVEN = ("search_lasso", "search_MCP", "search_SCAD", "search_pure",
               "search_lasso_oracle", "search_MCP_oracle", "search_SCAD_oracle")
RANDOM_C = ("C_Random_Diag", "C_Random_Min_Diag", "C_Random_Full")


def main() -> None:
    rows = []
    for f in ("random_direct_p10_raw_n1000", "random_direct_p10_raw"):
        rows += [r for r in csv.DictReader((RUNS / f"{f}_rows.csv").open()) if r["score"] == "bic"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["c_choice"], r["k"], r["rep"], r["n"])][r["method"]] = r
    out = []
    for cs, cname in ((("C_ID",), "C_ID"), (RANDOM_C, "random C")):
        for n in ("1000", "10000", "inf"):
            for ks, kname in ((("1", "2"), "k = 1, 2"), (("3", "4"), "k = 3, 4")):
                keys = [k for k in by if k[0] in cs and k[3] == n and k[1] in ks and "search_truth" in by[k]]
                lower = same = higher = 0
                f1_t, f1_d, gap = [], [], []
                for k in keys:
                    t = float(by[k]["search_truth"]["bic_value"])
                    cand = {m: float(by[k][m]["bic_value"]) for m in DATA_DRIVEN if m in by[k]}
                    best = min(cand, key=cand.get)
                    tol = 1e-6 * max(1.0, abs(t))
                    lower += t < cand[best] - tol
                    same += abs(t - cand[best]) <= tol
                    higher += t > cand[best] + tol
                    f1_t.append(float(by[k]["search_truth"]["f1"]))
                    f1_d.append(float(by[k][best]["f1"]))
                    gap.append(cand[best] - t)
                out.append({"true_C": cname, "n": n, "density": kname, "graphs": len(keys),
                            "truth_start_lower_bic": lower, "equal_bic": same, "data_driven_lower_bic": higher,
                            "f1_truth_start": float(np.mean(f1_t)), "f1_best_bic_data_driven": float(np.mean(f1_d)),
                            "median_bic_gap": float(np.median(gap))})
    with (HERE / "score_vs_optimisation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{'true C':<9} {'n':>6} {'density':<9} graphs | truth-start lower / equal / data-driven lower BIC | F1 truth-start, best-BIC data-driven")
    for r in out:
        print(f"{r['true_C']:<9} {r['n']:>6} {r['density']:<9} {r['graphs']:>5}  | {r['truth_start_lower_bic']:>3} / {r['equal_bic']:>3} / {r['data_driven_lower_bic']:>3}"
              f"   | {r['f1_truth_start']:.3f}, {r['f1_best_bic_data_driven']:.3f}")

    print("\nerror decomposition, C_ID, raw scale (means per graph)")
    cols = ("f1", "skeleton_f1", "correct", "reversed", "hedged", "missed_single", "both", "half", "edges")
    for n in ("1000", "10000", "inf"):
        for ks, kname in ((("1", "2"), "k = 1, 2"), (("3", "4"), "k = 3, 4")):
            keys = [k for k in by if k[0] == "C_ID" and k[3] == n and k[1] in ks]
            for m in ("path_lasso_bic", "search_lasso", "search_pure", "search_truth"):
                v = {c: np.mean([float(by[k][m][c]) for k in keys]) for c in cols}
                fp = np.mean([float(by[k][m]["fp_single"]) + float(by[k][m]["fp_double"]) for k in keys])
                print(f"  n={n:>5} {kname} {m:<15} F1 {v['f1']:.3f} skeleton {v['skeleton_f1']:.3f} | correct {v['correct']:.1f} "
                      f"reversed {v['reversed']:.1f} hedged {v['hedged']:.1f} missed {v['missed_single']:.1f} | "
                      f"2-cycles both {v['both']:.2f} half {v['half']:.2f} | false pairs {fp:.1f} | edges {v['edges']:.1f}")


if __name__ == "__main__":
    main()
