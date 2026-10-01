#!/usr/bin/env python3
"""Compare penalties and losses: the four Figure 5 metrics vs. p, one line per
run.  **Run locally, not on the cluster.**

Every run is a directory holding an ``s1_summary.csv`` from ``aggregate_s1.py``
(the same layout as the Figure 5 reproduction), so any set of runs that share
the grid can be overlaid -- lasso vs. MCP vs. SCAD on one loss, or one penalty
across the three losses.

    python simulations/plot_penalties.py \\
        --run "lasso=runs/s1_dettling_reproduction" \\
        --run "MCP=runs/s1b_pilot_p10-20/MCP" --run "SCAD=runs/s1b_pilot_p10-20/SCAD" \\
        --title "Direct loss, p = 10, 20" --out runs/s1b_pilot_p10-20/figures/penalties.png

One panel per (metric, C choice); ``--c`` restricts the C choices.  ``--p``
restricts the p values (e.g. to the ones every run covers).  The baseline run
may cover more replicates than the others; ``--reps`` keeps only ``rep < reps``
from each run's ``s1_per_dataset.csv`` so that all runs are averaged over the
same datasets, and the summary is recomputed from that subset.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_figures import C_STYLE, INK, INK_2, METRICS, style_axes

# validated categorical palette, one slot per run (see plot_figures.py)
RUN_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#4a3aa7", "D"),
             ("#52514e", "v"), ("#0b0b0b", "x")]


def load_summary(run: Path, reps: int | None, metrics):
    """{(p, c_choice): {metric: (mean, se)}} from a run directory."""
    per = run / "s1_per_dataset.csv"
    if reps is None or not per.exists():
        rows = list(csv.DictReader((run / "s1_summary.csv").open()))
        return {(int(r["p"]), r["c_choice"]):
                {m: (float(r[m]), float(r[f"{m}_se"])) for m, _ in metrics} for r in rows}
    groups = defaultdict(list)
    for r in csv.DictReader(per.open()):
        if int(r["rep"]) < reps:
            groups[(int(r["p"]), r["c_choice"])].append(r)
    out = {}
    for key, g in groups.items():
        out[key] = {}
        for m, _ in metrics:
            v = np.array([float(r[m]) for r in g])
            out[key][m] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=directory; repeat for each run to overlay")
    ap.add_argument("--c", nargs="+", default=list(C_STYLE),
                    help="C choices to show (default: all four)")
    ap.add_argument("--p", type=int, nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="use only rep < reps from every run (same datasets everywhere)")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, _, path = spec.rpartition("=")      # the label may itself contain "="
        runs.append((label, load_summary(Path(path), args.reps, METRICS)))

    c_choices = [c for c in C_STYLE if c in args.c]
    fig, axes = plt.subplots(len(METRICS), len(c_choices),
                             figsize=(3.4 * len(c_choices) + 1, 2.6 * len(METRICS) + 1),
                             facecolor="white", squeeze=False)
    for i, (key, mtitle) in enumerate(METRICS):
        for j, c_name in enumerate(c_choices):
            ax = axes[i, j]
            style_axes(ax)
            for (label, summ), (colour, marker) in zip(runs, RUN_STYLE):
                ps = sorted(p for (p, c) in summ if c == c_name and (args.p is None or p in args.p))
                if not ps:
                    continue
                mu = np.array([summ[(p, c_name)][key][0] for p in ps])
                se = np.array([summ[(p, c_name)][key][1] for p in ps])
                ax.errorbar(ps, mu, yerr=se, color=colour, marker=marker, markersize=4,
                            linewidth=1.8, capsize=2.5, capthick=1, elinewidth=1,
                            label=label, zorder=3)
                ax.set_xticks(ps)
            if i == 0:
                ax.set_title(C_STYLE[c_name][2], color=INK, fontsize=10, loc="left", pad=6)
            if j == 0:
                ax.set_ylabel(mtitle, color=INK_2, fontsize=10)
            if i == len(METRICS) - 1:
                ax.set_xlabel("$p$", color=INK_2, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(runs), 6), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=12, x=0.04, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    if args.out.suffix == ".png":
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
