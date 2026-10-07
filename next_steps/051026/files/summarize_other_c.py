#!/usr/bin/env python3
"""First look, on the laptop, at the question wave 1 of the campaign answers at scale: do the
gains of MCP / SCAD dense -> sparse survive when the true C is not 2I?

Reads the output of

    python next_steps/031026/files/replicate_dense_to_sparse.py \
        --c C_Random_Diag C_Random_Min_Diag C_Random_Full --reps 5 \
        --out next_steps/051026/files/replicate_other_c.csv

(20 graphs per setting, p = 10, repository solvers) and, for comparison, the C_ID run of
3 October (40 graphs).  The run of 5 October was stopped after 80 minutes: n = 1000 and
10000 are complete, n = inf has the sparsest graphs only and is not reported.  Prints, per true-C setting and per C used in the fit, the mean of each
metric and the z of the paired difference to the lasso fitted with the same C.

    python next_steps/051026/files/summarize_other_c.py               # all methods, plain text
    python next_steps/051026/files/summarize_other_c.py --markdown    # the tables of the note
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = (HERE.parents[1] / "031026" / "files" / "replicate_dense_to_sparse.csv",
         HERE / "replicate_other_c.csv")
METHODS = ("lasso", "MCP_down", "MCP_up", "SCAD_up")
NAMES = {"lasso": "lasso", "MCP_down": "MCP standard", "MCP_up": "MCP dense->sparse",
         "SCAD_up": "SCAD dense->sparse"}
C_FIT = {"identity": "C = 2I", "variance": "rescaled C"}


def main():
    by = defaultdict(dict)        # (c_choice, n, c_scale, k, rep) -> method -> row
    for f in FILES:
        if not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            by[(r["c_choice"], r["n"], r["c_scale"], r["k"], r["rep"])][r["method"]] = r
    settings = ("C_ID", "C_Random_Min_Diag", "C_Random_Diag", "C_Random_Full")
    ns = sorted({k[1] for k in by}, key=float)
    for col, title in (("max_f1", "max_f1 (best of the path)"), ("bic_f1", "F1 of the BIC-selected graph"),
                       ("search_f1", "F1 after the BIC search")):
        print(f"\n=== {title}; in brackets z of the paired difference to the lasso with the same C ===")
        for c_choice in settings:
            for c_scale in ("identity", "variance"):
                graphs = len({k[3:] for k in by if k[0] == c_choice and k[2] == c_scale and k[1] == ns[0]})
                if not graphs:
                    continue
                print(f"{c_choice}, fitted with {C_FIT[c_scale]}  [{graphs} graphs]")
                for m in METHODS:
                    line = f"   {NAMES[m]:<20}"
                    for n in ns:
                        keys = [k for k in by if k[:3] == (c_choice, n, c_scale)
                                and m in by[k] and by[k][m][col] != "" and by[k]["lasso"][col] != ""]
                        if not keys:
                            line += f"{'':>20}"
                            continue
                        x = np.array([float(by[k][m][col]) for k in keys])
                        d = x - np.array([float(by[k]["lasso"][col]) for k in keys])
                        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
                        z = f"({d.mean() / se:+.1f})" if m != "lasso" and se > 0 else ""
                        line += f"   n={n:<6} {x.mean():.3f} {z:<7}"
                    print(line)


def markdown():
    """The two tables of cluster_campaign_051026.md, Section 3.7:
    ``python summarize_other_c.py --markdown``."""
    by = defaultdict(dict)
    for f in FILES:
        if f.exists():
            for r in csv.DictReader(f.open()):
                by[(r["c_choice"], r["n"], r["c_scale"], r["k"], r["rep"])][r["method"]] = r
    ns = sorted({k[1] for k in by}, key=float)
    lab = {"1000": "$n = 10^3$", "10000": "$10^4$", "inf": "$\\infty$"}
    for col, method, title in (
            ("max_f1", "MCP_up", "`max_f1`: lasso → MCP dense → sparse"),
            ("max_f1", "SCAD_up", "`max_f1`: lasso → SCAD dense → sparse"),
            ("max_f1", "MCP_down", "`max_f1`: lasso → MCP on the standard path"),
            ("search_f1", "MCP_up", "$F_1$ of the BIC-selected graph after the BIC search: from the lasso → from MCP dense → sparse")):
        print(f"\n{title}\n")
        print("| true $C$ | fitted with | " + " | ".join(lab[n] for n in ns) + " |")
        print("|---|---|" + "---|" * len(ns))
        for c_choice in ("C_ID", "C_Random_Min_Diag", "C_Random_Diag", "C_Random_Full"):
            for c_scale in ("identity", "variance"):
                cells, graphs = [], 0
                for n in ns:
                    keys = [k for k in by if k[:3] == (c_choice, n, c_scale) and method in by[k]
                            and by[k][method][col] != "" and by[k]["lasso"][col] != ""]
                    graphs = max(graphs, len(keys))
                    # a sample size is reported only if all its graphs finished: the run
                    # writes sparse graphs first, so a partial cell would be biased
                    if not keys or len(keys) < graphs:
                        cells.append("not finished")
                        continue
                    x = np.array([float(by[k][method][col]) for k in keys])
                    y = np.array([float(by[k]["lasso"][col]) for k in keys])
                    d = x - y
                    se = d.std(ddof=1) / math.sqrt(len(d))
                    z = f"{d.mean() / se:+.1f}".replace("-", "−") if se > 0 else "0"
                    cells.append(f"{y.mean():.3f} → {x.mean():.3f} ({z})")
                print(f"| `{c_choice}` ({graphs}) | {'$C = 2I$' if c_scale == 'identity' else 'rescaled $C$'} | "
                      + " | ".join(cells) + " |")


if __name__ == "__main__":
    import sys

    markdown() if "--markdown" in sys.argv else main()
