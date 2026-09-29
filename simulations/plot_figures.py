#!/usr/bin/env python3
"""Reproduce Dettling Figures 3 and 5 as plots.  **Run locally, not on the cluster.**

The cluster writes numbers only; this script turns them into figures.  It reads
nothing but the aggregated CSV/NPZ files, so it can be re-run and restyled
without touching the simulation.

    python simulations/plot_figures.py --results results --out figures

Palette: categorical slots blue / orange / aqua / violet, validated for
colourblind separation and normal-vision separation (see docs/REPRODUCTION.md).
Each series also carries a distinct marker, so identity never rests on colour
alone, and every plotted number is available as CSV alongside.
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

# validated categorical palette (light surface #fcfcfb)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

C_STYLE = {                      # Dettling's four volatility choices
    "C_ID": (BLUE, "o", "C = 2I"),
    "C_Random_Min_Diag": (ORANGE, "s", "C random diag, U[2,4]"),
    "C_Random_Diag": (AQUA, "^", "C random diag, U[0.5,4]"),
    "C_Random_Full": (VIOLET, "D", "C random, non-diagonal"),
}
M0_STYLE = {
    "path": (BLUE, "o", "path $G_1$"),
    "cycle_fixed": (ORANGE, "s", "5-cycle $G_2$, $m_{15}=0.65$"),
    "cycle_random": (AQUA, "^", "5-cycle $G_2$, $m_{15}$ random"),
}
METRICS = [("max_acc", "maximum accuracy"), ("max_f1", "maximum $F_1$"),
           ("auc", "area under ROC"), ("aupr", "area under PR")]


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def plot_figure5(results: Path, out: Path):
    rows = list(csv.DictReader((results / "s1_summary.csv").open()))
    if not rows:
        raise SystemExit("s1_summary.csv is empty")
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["c_choice"]].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
    for ax, (key, title) in zip(axes.ravel(), METRICS):
        style_axes(ax)
        for c_name in C_STYLE:
            g = sorted(by_c.get(c_name, []), key=lambda r: int(r["p"]))
            if not g:
                continue
            colour, marker, label = C_STYLE[c_name]
            ps = [int(r["p"]) for r in g]
            mu = np.array([float(r[key]) for r in g])
            se = np.array([float(r[f"{key}_se"]) for r in g])
            ax.errorbar(ps, mu, yerr=se, color=colour, marker=marker,
                        markersize=5, linewidth=2, capsize=3, capthick=1,
                        elinewidth=1, label=label, zorder=3)
        ax.set_xlabel("$p$", color=INK_2, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
        # tick only the p values actually present, so a partial run does not
        # render as a mostly-empty axis stretched to p = 50
        ax.set_xticks(sorted({int(r["p"]) for r in rows}))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    fig.suptitle("Figure 5 — Direct Lyapunov Lasso, support recovery vs. $p$",
                 color=INK, fontsize=13, x=0.055, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure5_reproduction.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure5_reproduction.png / .pdf   ({len(rows)} (p, C) points)")


def plot_figure3(results: Path, out: Path):
    path_csv = results / "m0_reps100.csv"
    if not path_csv.exists():
        cands = sorted(results.glob("m0*.csv"))
        if not cands:
            print("  (no m0*.csv found -- skipping Figure 3)")
            return
        path_csv = cands[-1]
    rows = list(csv.DictReader(path_csv.open()))

    agg = defaultdict(list)
    for r in rows:
        agg[(r["setting"], float(r["n"]))].append(r)

    keys = ("max_acc", "max_f1", "auc")
    titles = ("maximum accuracy", "maximum $F_1$", "area under ROC")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="white")
    sizes = sorted({float(r["n"]) for r in rows})
    xs = np.arange(len(sizes))
    def tick(n):
        # plain integers up to 5000, powers of ten beyond -- as in Dettling Fig. 3
        if np.isinf(n):
            return "$\\infty$"
        return f"$10^{{{int(np.log10(n))}}}$" if n >= 1e4 else f"{int(n)}"

    labels = [tick(s) for s in sizes]

    for ax, key, title in zip(axes, keys, titles):
        style_axes(ax)
        for setting in M0_STYLE:
            colour, marker, label = M0_STYLE[setting]
            mu, se = [], []
            for s in sizes:
                g = agg.get((setting, s), [])
                v = np.array([float(r[key]) for r in g], float)
                mu.append(v.mean() if len(v) else np.nan)
                se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            ax.errorbar(xs, mu, yerr=se, color=colour, marker=marker,
                        markersize=5, linewidth=2, capsize=3, capthick=1,
                        elinewidth=1, label=label, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("sample size $n$", color=INK_2, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)

    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), labelcolor=INK_2)
    fig.suptitle("Figure 3 — path vs. 5-cycle: the irrepresentability failure",
                 color=INK, fontsize=13, x=0.045, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure3_reproduction.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure3_reproduction.png / .pdf   ({len(rows)} rows from {path_csv.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"reading {args.results}, writing {args.out}")
    plot_figure3(args.results, args.out)
    if (args.results / "s1_summary.csv").exists():
        plot_figure5(args.results, args.out)
    else:
        print("  (no s1_summary.csv -- run aggregate_s1.py first)")


if __name__ == "__main__":
    main()
