#!/usr/bin/env python3
"""Figure for E4 of the plan of 13 October 2026: the paired gain over the lasso by the class of
the true graph, within k (``campaign_graph_class.csv`` from ``graph_class.py``).  **Run locally.**

One figure per p (``gain_by_class_p<p>.png``): rows (C, n) for n = 10^3 and 10^4, columns k;
x = graph class, one series per estimator with standard errors; the number of graphs per
(class, k) is printed under the class.  Classes with fewer than ``MIN_GRAPHS`` graphs are shown
hollow and without error bars, so that a mean of three graphs is not read as a result.  With the
wave 9 class cells (``campaign_generator.csv``, plan §5.3) also ``gain_by_generator_p<p>.png``: x = the
generator (the thesis generator / dag), classes pooled, within k.
With ``campaign_two_cycles.csv`` also ``two_cycle_recovery_p<p>.png``: how each estimator keeps the
true 2-cycles and the true single edges of the selected graph (n = 10^4, rescaled C), means per graph.

    python simulations/diagnostics/plot_graph_class.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from gclm.graphs import CLASSES  # noqa: E402
from plot_bidirectional import save  # noqa: E402
from plot_campaign import C_NAME, GREY, INK, INK_2, N_LABEL, STYLE, SURFACE, f, legend, read, style_axes  # noqa: E402

CLASS_LABEL = {"dag": "DAG", "simple": "cyclic,\nno 2-cycle", "two_cycles": "with\n2-cycles"}
MIN_GRAPHS = 20


GENERATOR_LABEL = {"random": "thesis\ngenerator", "dag": "DAG"}
#: the parts of a true 2-cycle and of a true single edge in ``two_cycle_recovery_p<p>.png``
PAIR_PARTS = (("both", "#2a78d6", "both directions kept"), ("half", "#8fbbea", "one direction kept"), ("neither", GREY, "neither"))
SINGLE_PARTS = (("correct", "#2a78d6", "right direction only"), ("hedged", "#8fbbea", "both directions kept (a hedge)"),
                ("reversed", "#eb6834", "reverse only"), ("missed", GREY, "missed"))
RULE_TITLE = {"bic": "selected graph (BIC penalty)", "search": "graph after the search (BIC penalty)"}


def fig_gain_by_class(table, out, p="10", ests=("MCP", "MCP-up", "SCAD-up", "adaptive"), metric="max_f1",
                      ns=("1000", "1e4"), ks=("1", "2", "3", "4")):
    rows = [r for r in table if r["p"] == p and r["metric"] == metric and r["n"] in ns
            and r.get("generator", "random") == "random"]
    if not rows:
        return
    panels = [(c, n) for c in ("C2I", "Cresc") for n in ns]
    fig, axes = plt.subplots(len(panels), len(ks), figsize=(2.6 * len(ks) + 1.2, 2.5 * len(panels)),
                             sharey=True, squeeze=False)
    x = np.arange(len(CLASSES))
    for a, (c, n) in enumerate(panels):
        for b, k in enumerate(ks):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0, color=GREY, lw=0.8, zorder=1)
            counts = {}
            for e_idx, est in enumerate(ests):
                col, mk, filled, _ = STYLE[est]
                for ci, cls in enumerate(CLASSES):
                    r = next((r for r in rows if (r["c"], r["n"], r["k"], r["graph_class"], r["estimator"]) == (c, n, k, cls, est)), None)
                    if r is None or math.isnan(f(r["gain"])):
                        continue
                    counts[cls] = int(r["n_graphs"])
                    xx = x[ci] + (e_idx - (len(ests) - 1) / 2) * 0.16
                    enough = int(r["n_graphs"]) >= MIN_GRAPHS
                    if enough and not math.isnan(f(r["gain_se"])):
                        ax.errorbar(xx, f(r["gain"]), yerr=f(r["gain_se"]), color=col, lw=0, elinewidth=1.0, capsize=2.5, zorder=2)
                    ax.plot(xx, f(r["gain"]), ls="none", marker=mk, ms=6.5, color=col if (filled and enough) else SURFACE,
                            mec=SURFACE if (filled and enough) else col, mew=1.3, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{CLASS_LABEL[cls]}\n({counts.get(cls, 0)})" for cls in CLASSES], fontsize=7.5)
            ax.set_title(f"k = {k}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}", loc="left", fontsize=8.5, color=INK)
        axes[a][0].set_ylabel("gain over the lasso\n(F₁ at the oracle λ)" if metric == "max_f1" else f"gain over the lasso ({metric})",
                              fontsize=8.5, color=INK_2)
    fig.suptitle(f"p = {p}: paired gain over the lasso by the class of the true graph, within k "
                 f"(hollow: fewer than {MIN_GRAPHS} graphs)", fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    legend(fig, ests, y=-0.005, ncol=4)
    save(fig, out, f"gain_by_class_p{p}" + ("" if metric == "max_f1" else f"_{metric}"))


def fig_gain_by_generator(gens, out, p="10", ests=("MCP", "MCP-up", "adaptive"), metric="max_f1",
                          ns=("1000", "1e4"), ks=("1", "2", "3", "4")):
    """The paired gain over the lasso drawn with the same generator, classes pooled, within k."""
    from graph_class import GENERATORS
    rows = [r for r in gens if r["p"] == p and r["metric"] == metric and r["n"] in ns]
    if not rows:
        return
    panels = [(c, n) for c in ("C2I", "Cresc") for n in ns if any(r["c"] == c and r["n"] == n for r in rows)]
    fig, axes = plt.subplots(len(panels), len(ks), figsize=(2.6 * len(ks) + 1.2, 2.5 * len(panels)),
                             sharey=True, squeeze=False)
    x = np.arange(len(GENERATORS))
    for a, (c, n) in enumerate(panels):
        for b, k in enumerate(ks):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0, color=GREY, lw=0.8, zorder=1)
            counts = {}
            for e_idx, est in enumerate(ests):
                col, mk, filled, _ = STYLE[est]
                for gi, gen in enumerate(GENERATORS):
                    r = next((r for r in rows if (r["c"], r["n"], r["k"], r["generator"], r["estimator"]) == (c, n, k, gen, est)), None)
                    if r is None or math.isnan(f(r["gain"])):
                        continue
                    counts[gen] = int(r["n_graphs"])
                    xx = x[gi] + (e_idx - (len(ests) - 1) / 2) * 0.16
                    enough = int(r["n_graphs"]) >= MIN_GRAPHS
                    if enough and not math.isnan(f(r["gain_se"])):
                        ax.errorbar(xx, f(r["gain"]), yerr=f(r["gain_se"]), color=col, lw=0, elinewidth=1.0, capsize=2.5, zorder=2)
                    ax.plot(xx, f(r["gain"]), ls="none", marker=mk, ms=6.5, color=col if (filled and enough) else SURFACE,
                            mec=SURFACE if (filled and enough) else col, mew=1.3, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{GENERATOR_LABEL[g]}\n({counts.get(g, 0)})" for g in GENERATORS], fontsize=7.5)
            ax.set_title(f"k = {k}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}", loc="left", fontsize=8.5, color=INK)
        axes[a][0].set_ylabel("gain over the lasso\n(F₁ at the oracle λ)" if metric == "max_f1" else f"gain over the lasso ({metric})",
                              fontsize=8.5, color=INK_2)
    fig.suptitle(f"p = {p}: paired gain over the lasso of the same generator, within k "
                 f"(hollow: fewer than {MIN_GRAPHS} graphs)", fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    legend(fig, ests, y=-0.005, ncol=3)
    save(fig, out, f"gain_by_generator_p{p}" + ("" if metric == "max_f1" else f"_{metric}"))


def fig_two_cycles(table, out, p="20", n="1e4", c="Cresc", rule="bic", generator="random",
                   ests=("lasso", "MCP", "MCP-up", "SCAD-up", "adaptive")):
    """How each estimator keeps the true 2-cycles (left) and the true single edges (right) of the
    graphs of one cell: means per graph from ``campaign_two_cycles.csv`` (``graph_class.py``).  A
    2-cycle is kept with both directions, one or neither; a single edge with its direction alone,
    with both directions (the lasso's hedge), with the reverse alone, or not at all."""
    rows = {r["estimator"]: r for r in table
            if (r["p"], r["n"], r["c"], r["rule"], r.get("generator") or "random") == (p, n, c, rule, generator)}
    ests = [e for e in ests if e in rows]
    if not ests:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.5 * len(ests) + 2.8), gridspec_kw={"width_ratios": [1, 1.7]})
    y = np.arange(len(ests))[::-1]
    for ax, parts, total, title, unit in ((axes[0], PAIR_PARTS, "two_cycles", "true 2-cycles", "pairs per graph"),
                                          (axes[1], SINGLE_PARTS, "single_edges", "true single edges", "edges per graph")):
        style_axes(ax)
        ax.grid(False, axis="y")
        left = np.zeros(len(ests))
        for part, colour, label in parts:
            vals = np.array([f(rows[e][part]) for e in ests])
            ax.barh(y, vals, left=left, color=colour, edgecolor=SURFACE, linewidth=1.2, height=0.62, label=label, zorder=3)
            left += vals
        truth = f(rows[ests[0]][total])
        ax.axvline(truth, color=INK_2, lw=0.9, ls=":", zorder=4)
        ax.text(truth, len(ests) - 0.4, f"{title} per graph: {truth:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([STYLE[e][3] for e in ests] if ax is axes[0] else [""] * len(ests), fontsize=9)
        ax.set_xlabel(unit, fontsize=9, color=INK_2)
        ax.set_xlim(0, truth * 1.15)
        ax.set_ylim(-0.6, len(ests) + 0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.5)
    n_graphs = int(f(rows[ests[0]]["n_graphs"]))
    fig.suptitle(f"{RULE_TITLE[rule]}, p = {p}, n = {N_LABEL[n]}, {C_NAME[c].split(' (')[0]}: how the true pairs are kept, "
                 f"means per graph over {n_graphs} graphs", fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    suffix = "" if (n, c, rule, generator) == ("1e4", "Cresc", "bic", "random") else f"_n{n}_{c}_{rule}_{generator}"
    save(fig, out, f"two_cycle_recovery_p{p}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    args = ap.parse_args()
    table = read(args.root / "campaign_graph_class.csv")
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    for p in sorted({r["p"] for r in table}, key=int):
        for metric in ("max_f1", "bic_f1"):
            fig_gain_by_class(table, out, p=p, metric=metric)
    gens_path = args.root / "campaign_generator.csv"
    if gens_path.exists():
        gens = read(gens_path)
        for p in sorted({r["p"] for r in gens}, key=int):
            for metric in ("max_f1", "bic_f1"):
                fig_gain_by_generator(gens, out, p=p, metric=metric)
    pairs_path = args.root / "campaign_two_cycles.csv"
    if pairs_path.exists():
        pairs = read(pairs_path)
        for p in sorted({r["p"] for r in pairs}, key=int):
            ns = {r["n"] for r in pairs if r["p"] == p}
            fig_two_cycles(pairs, out, p=p, n="1e4" if "1e4" in ns else sorted(ns, key=float)[0])


if __name__ == "__main__":
    main()
