#!/usr/bin/env python3
"""Figures for S3a (simulations/S3a_bidirectional_edges.md).  **Run locally.**

Reads only the CSVs that ``bidirectional.py summarize`` writes, so every plotted
number is also in a table next to the figure:

    python simulations/diagnostics/plot_bidirectional.py [--in runs/s3a_bidirectional]

writes ``<in>/figures/``:
  bidirectional_path.{png,pdf}  pairs with both directions selected vs. selected
                                pairs along the path, per (p, n), the true graph
                                as a reference point          (path_means.csv)
  best_f1_outcomes.{png,pdf}    what happens to true single edges and true
                                2-cycles at the best-F1 point (best_f1_outcomes.csv)
  example_graph.{png,pdf}       one p = 10 graph: the truth and the three best-F1
                                estimates as networks           (example.csv)

Style follows plot_figures.py / plot_penalties.py: lasso blue o, MCP orange s,
SCAD aqua ^ (validated for colour-vision deficiency; aqua is below 3:1 contrast,
so every series also has its own marker and the numbers are in the CSVs).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulations"))
from plot_figures import GRID, INK, INK_2, style_axes  # noqa: E402

PEN_STYLE = {"lasso": ("#2a78d6", "o"), "MCP": ("#eb6834", "s"), "SCAD": ("#1baf7a", "^")}
SURFACE = "#fcfcfb"
WRONG = "#d03b3b"        # status "critical": an edge pointing the wrong way
FALSE = "#a19f99"        # a selected edge in a pair without any true edge
MISSED = "#c9c7c1"       # a true edge the estimate does not select


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def cells_of(rows):
    ps = sorted({int(r["p"]) for r in rows})
    ns = sorted({r["n"] for r in rows}, key=float)
    return ps, ns


# --------------------------------------------------------------------------- #
# 1. along the path
# --------------------------------------------------------------------------- #


def fig_path(rows, summary, out: Path) -> None:
    ps, ns = cells_of(rows)
    best = {(int(r["p"]), r["n"], r["penalty"]): (float(r["selected_pairs_at_best"]), float(r["bidir_at_best"]))
            for r in summary}
    fig, axes = plt.subplots(len(ps), len(ns), figsize=(4.6 * len(ns), 3.7 * len(ps)), squeeze=False)
    for a, p in enumerate(ps):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            sub = [r for r in rows if int(r["p"]) == p and r["n"] == n]
            if not sub:
                ax.set_visible(False)
                continue
            style_axes(ax)
            for pen, (col, mk) in PEN_STYLE.items():
                rr = sorted((r for r in sub if r["penalty"] == pen), key=lambda r: -float(r["lambda_rel"]))
                if not rr:
                    continue
                x = np.array([float(r["selected_pairs"]) for r in rr])
                y = np.array([float(r["bidirectional"]) for r in rr])
                ax.plot(x, y, color=col, lw=1.6, solid_capstyle="round", zorder=3)
                ax.plot(x[::11], y[::11], ls="none", marker=mk, ms=6.5, color=col, mec=SURFACE,
                        mew=1.2, zorder=4, label=pen)
                if (p, n, pen) in best:
                    ax.plot(*best[(p, n, pen)], ls="none", marker=mk, ms=11, color=col, mec=INK,
                            mew=1.3, zorder=6)
                if pen == "lasso":
                    ax.annotate("lasso", (x[-1], y[-1]), xytext=(-10, 8), textcoords="offset points",
                                ha="right", fontsize=8.5, color=INK_2)
            tp, tc = float(sub[0]["true_pairs"]), float(sub[0]["true_2cycles"])
            ax.plot([tp], [tc], ls="none", marker="*", ms=13, color=INK, mec=SURFACE, mew=1.2,
                    zorder=5, label="true graph")
            ax.annotate("true graph", (tp, tc), xytext=(9, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_2, va="center")
            ax.set_xlim(0, p * (p - 1) / 2 * 1.03)
            ax.set_ylim(bottom=0)
            ax.set_title(f"p = {p}, n = {int(float(n)):,}", loc="left", fontsize=10, color=INK)
            if a == len(ps) - 1:
                ax.set_xlabel("selected pairs (either direction), mean per graph", fontsize=9, color=INK_2)
            if b == 0:
                ax.set_ylabel("pairs with both directions selected", fontsize=9, color=INK_2)
    handles, labels = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", ms=11, color="white", mec=INK, mew=1.3))
    labels.append("best-F1 point (mean)")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "bidirectional_path")


# --------------------------------------------------------------------------- #
# 2. at the best-F1 point
# --------------------------------------------------------------------------- #

ROWS = [("single edge", "correct", "i→j: correct direction only"),
        ("single edge", "hedged", "i→j: both directions"),
        ("single edge", "reversed", "i→j: reversed direction only"),
        ("single edge", "missed", "i→j: missed"),
        None,
        ("2-cycle", "both", "i⇄j: both directions"),
        ("2-cycle", "half", "i⇄j: one direction"),
        ("2-cycle", "missed", "i⇄j: missed")]


def fig_best(rows, out: Path) -> None:
    ps, ns = cells_of(rows)
    fig, axes = plt.subplots(len(ps), len(ns), figsize=(4.6 * len(ns), 3.5 * len(ps)),
                             squeeze=False, sharey=True)
    dodge = {"lasso": -0.2, "MCP": 0.0, "SCAD": 0.2}
    for a, p in enumerate(ps):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            sub = {(r["penalty"], r["group"], r["outcome"]): float(r["share"])
                   for r in rows if int(r["p"]) == p and r["n"] == n}
            if not sub:
                ax.set_visible(False)
                continue
            style_axes(ax)
            ax.grid(False, axis="y")
            for y, row in enumerate(ROWS):
                if row is None:
                    ax.axhline(y, color=GRID, lw=0.8, zorder=0)
                    continue
                group, outcome, _ = row
                vals = [sub[(pen, group, outcome)] for pen in PEN_STYLE if (pen, group, outcome) in sub]
                ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.6, solid_capstyle="round", zorder=1)
                for pen, (col, mk) in PEN_STYLE.items():
                    if (pen, group, outcome) in sub:
                        ax.plot([sub[(pen, group, outcome)]], [y + dodge[pen]], ls="none", marker=mk,
                                ms=7, color=col, mec=SURFACE, mew=1.2, zorder=3,
                                label=pen if y == 0 else None)
            ax.set_yticks([y for y, r in enumerate(ROWS) if r is not None])
            ax.set_yticklabels([r[2] for r in ROWS if r is not None], fontsize=8.5)
            ax.set_ylim(len(ROWS) - 0.5, -0.6)
            ax.set_xlim(0, 1)
            ax.set_title(f"p = {p}, n = {int(float(n)):,}", loc="left", fontsize=10, color=INK)
            if a == len(ps) - 1:
                ax.set_xlabel("share of the true pairs of that kind", fontsize=9, color=INK_2)
    handles, labels = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "best_f1_outcomes")


# --------------------------------------------------------------------------- #
# 3. one example graph
# --------------------------------------------------------------------------- #


def directed_edges(codes, iu, ju) -> set[tuple[int, int]]:
    """Directed edges a -> b from pair codes (edge a -> b is M[b, a]): code 1
    (M[i, j]) is j -> i, code 2 (M[j, i]) is i -> j, code 3 both."""
    edges = set()
    for c, i, j in zip(codes, iu, ju):
        if c in (1, 3):
            edges.add((j, i))
        if c in (2, 3):
            edges.add((i, j))
    return edges


def draw_graph(ax, pos, edges, colors, styles, title, subtitle) -> None:
    r = 0.11
    for (a, b) in sorted(edges):
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=0.2",
                                     arrowstyle="-|>", mutation_scale=11, lw=1.4,
                                     color=colors[(a, b)], linestyle=styles[(a, b)],
                                     shrinkA=13, shrinkB=13, zorder=2))
    for v, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=1.0, zorder=3))
        ax.text(x, y, str(v + 1), ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.75, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.text(-1.25, -1.3, subtitle, fontsize=8, color=INK_2, va="top")


def fig_example(rows, out: Path) -> None:
    if not rows:
        return
    p = int(rows[0]["p"])
    iu = [int(r["i"]) for r in rows]
    ju = [int(r["j"]) for r in rows]
    ang = np.pi / 2 - 2 * np.pi * np.arange(p) / p
    pos = list(zip(np.cos(ang), np.sin(ang)))
    truth = directed_edges([int(r["truth"]) for r in rows], iu, ju)
    n_cyc = sum(int(r["truth"]) == 3 for r in rows)
    pens = [pen for pen in PEN_STYLE if pen in rows[0]]
    fig, axes = plt.subplots(1, 1 + len(pens), figsize=(3.6 * (1 + len(pens)), 4.6))
    draw_graph(axes[0], pos, truth, {e: INK for e in truth}, {e: "-" for e in truth},
               "true graph", f"{len(truth)} edges, {n_cyc} 2-cycles")
    for ax, pen in zip(axes[1:], pens):
        est = directed_edges([int(r[pen]) for r in rows], iu, ju)
        colors, styles = {}, {}
        for e in est:
            colors[e] = INK if e in truth else (WRONG if e[::-1] in truth else FALSE)
            styles[e] = "-"
        for e in truth - est:
            colors[e], styles[e] = MISSED, (0, (1.5, 2))
        both = sum(int(r[pen]) == 3 for r in rows)
        n_ok = sum(e in truth for e in est)
        n_wrong = sum(e not in truth and e[::-1] in truth for e in est)
        n_false = len(est) - n_ok - n_wrong
        draw_graph(ax, pos, est | (truth - est), colors, styles, f"{pen}, best-F1 estimate",
                   f"correct {n_ok} · wrong direction {n_wrong} · false {n_false}\n"
                   f"pairs with both directions: {both}")
    legend = [Line2D([], [], color=INK, lw=1.4, label="true edge, selected"),
              Line2D([], [], color=WRONG, lw=1.4, label="selected, but the true edge points the other way"),
              Line2D([], [], color=FALSE, lw=1.4, label="selected, no true edge in this pair"),
              Line2D([], [], color=MISSED, lw=1.4, ls=(0, (1.5, 2)), label="true edge, not selected")]
    r0 = rows[0]
    fig.suptitle(f"example graph (p = {p}, k = {r0['k']}, {r0['c_choice']}, rep {r0['rep']}, "
                 f"n = {int(float(r0['n'])):,}), chosen to make the effects visible",
                 x=0.01, y=0.995, ha="left", fontsize=10, color=INK)
    fig.legend(handles=legend, loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.005, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out, "example_graph")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "runs" / "s3a_bidirectional")
    args = ap.parse_args()
    out = args.inp / "figures"
    fig_path(read(args.inp / "path_means.csv"), read(args.inp / "summary.csv"), out)
    fig_best(read(args.inp / "best_f1_outcomes.csv"), out)
    if (args.inp / "example.csv").exists():
        fig_example(read(args.inp / "example.csv"), out)


if __name__ == "__main__":
    main()
