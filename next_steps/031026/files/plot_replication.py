#!/usr/bin/env python3
"""Figure for the replication of the independent study's central claims
(replicate_dense_to_sparse.csv -> replication_dense_to_sparse.{png,pdf}).

Rows: the oracle path maximum (max_f1) and the directed F1 of the graph selected by the score (BIC penalty).
Columns: the two volatility matrices fitted to the standardised data.
Encoding as elsewhere in the repository (lasso blue circle, MCP orange square, SCAD aqua
triangle); hollow + thin = the path used in every thesis run so far (sparse -> dense),
filled + thick = dense -> sparse.

    python next_steps/031026/files/plot_replication.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "simulations"))
from plot_figures import INK, INK_2, style_axes  # noqa: E402

SURFACE = "#fcfcfb"
SERIES = [("lasso", "lasso", "#2a78d6", "o", True),
          ("MCP_down", "MCP, sparse → dense (every run so far)", "#eb6834", "s", False),
          ("MCP_up", "MCP, dense → sparse", "#eb6834", "s", True),
          ("SCAD_up", "SCAD, dense → sparse", "#1baf7a", "^", True)]
NS = ("1000", "10000", "inf")
N_LABEL = {"1000": "10³", "10000": "10⁴", "inf": "∞"}
SCALES = (("variance", "rescaled C = 2·diag(1/s²)"), ("identity", "C = 2I (Dettling's pipeline)"))
METRICS = (("max_f1", "maximum F1 along the path\n(oracle λ)"), ("bic_f1", "directed F1 of the\nselected graph (BIC penalty)"))


def main() -> None:
    rows = list(csv.DictReader((HERE / "replicate_dense_to_sparse.csv").open()))
    val = defaultdict(list)
    for r in rows:
        for m, _ in METRICS:
            val[(r["c_scale"], r["n"], r["method"], m)].append(float(r[m]))
    fig, axes = plt.subplots(len(METRICS), len(SCALES), figsize=(8.6, 6.4), sharey=True, squeeze=False)
    x = np.arange(len(NS))
    for a, (metric, ylab) in enumerate(METRICS):
        for b, (scale, name) in enumerate(SCALES):
            ax = axes[a][b]
            style_axes(ax)
            for k, (method, label, col, mk, filled) in enumerate(SERIES):
                v = [np.array(val[(scale, n, method, metric)]) for n in NS]
                y = [u.mean() for u in v]
                se = [u.std(ddof=1) / math.sqrt(len(u)) for u in v]
                xs = x + (k - 1.5) * 0.03
                ax.errorbar(xs, y, yerr=se, color=col, lw=0, elinewidth=1.0, capsize=2.5, zorder=2)
                ax.plot(xs, y, color=col, lw=2.0 if filled else 1.0, zorder=3)
                ax.plot(xs, y, ls="none", marker=mk, ms=7, color=col if filled else SURFACE,
                        mec=SURFACE if filled else col, mew=1.3, zorder=4, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([N_LABEL[n] for n in NS])
            ax.set_ylim(0.42, 0.84)
            if a == 0:
                ax.set_title(name, loc="left", fontsize=10, color=INK)
            else:
                ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel(ylab, fontsize=9, color=INK_2)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"replication_dense_to_sparse.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", HERE / "replication_dense_to_sparse.png")


if __name__ == "__main__":
    main()
