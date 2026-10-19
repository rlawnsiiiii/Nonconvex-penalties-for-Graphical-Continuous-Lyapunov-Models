#!/usr/bin/env python3
"""Figures for the generator knobs: E3, the beta-min knob (``campaign_wmin*.csv`` from ``betamin.py``),
and, through ``plot_twocycles.py``, E5, the share of 2-cycles (``campaign_share*.csv``).  **Run locally.**

``gain_by_<tag>[_<rule>_f1].png``: the paired gain over the lasso drawn at the same level of the knob,
against the level (the reference, wave 1's thesis generator, at the left); rows (p, C), columns n.
``levels_by_<tag>_n<n>[_<rule>].png``: the F1, precision and recall of each estimator against the
level -- do all estimators rise, and is the pruners' edge precision or recall?
``orientation_by_<tag>_n<n>_<rule>.png``: what the graph of the rule does with the true single edges,
per estimator and level: right direction only / both directions (a hedge) / reverse only / missed,
stacked, with the false positives per graph written above each bar.
``pair_outcomes_by_<tag>_n<n>_<rule>.png``: the same for the true 2-cycles (both directions kept / one /
neither, with the identified share above) over the true single edges.
``orientation_by_weight_...`` and ``orientation_bin_across_laws_...`` (E3 only): the fates within each
level by |w| bin, and one bin across the levels (``campaign_wmin_bins.csv``).

    python3 simulations/diagnostics/plot_betamin.py                       # runs/campaign -> runs/campaign/figures
    python3 simulations/diagnostics/plot_betamin.py --root <rehearsal root> --out docs/figures \\
        --suffix _rehearsal --note "laptop rehearsal: p = 10, one replicate"
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from betamin import BIN_LABELS, KNOB_ESTIMATORS, KNOB_RULES, OUTCOMES, WMIN, Knob  # noqa: E402
from plot_bidirectional import save  # noqa: E402
from plot_campaign import C_NAME, GREY, INK, INK_2, N_LABEL, STYLE, SURFACE, f, legend, read, style_axes  # noqa: E402
from plot_graph_class import PAIR_PARTS, SINGLE_PARTS  # noqa: E402

RULE_SHORT = {"bic": "selected graph (BIC)", "search": "search from the BIC selection",
              "ebic05": "selected graph (eBIC, γ = 0.5)", "ebic05_search": "search from the eBIC(0.5) selection",
              "ebic1": "selected graph (eBIC, γ = 1)", "ebic1_search": "search from the eBIC(1) selection"}
METRIC_LABEL = {"max_f1": "F₁ at the oracle λ", "aupr": "AUPR",
                **{f"{r}_{m}": f"{lab} of the {RULE_SHORT[r]}" for r in KNOB_RULES
                   for m, lab in (("f1", "F₁"), ("precision", "precision"), ("recall", "recall"))}}
#: how each graph of a data set is picked, for the footnotes of the figures
GRAPH_OF = {"max_f1": "the path's support with the highest F₁ against the truth (oracle λ, not available in practice)",
            "bic": "the support that minimises loss + BIC penalty over the path (no truth used)",
            "search": "the greedy search started from the BIC-selected support",
            "ebic05": "the support that minimises loss + eBIC penalty (γ = 0.5) over the path",
            "ebic05_search": "the greedy search started from the eBIC(0.5)-selected support",
            "ebic1": "the support that minimises loss + eBIC penalty (γ = 1) over the path",
            "ebic1_search": "the greedy search started from the eBIC(1)-selected support"}
FOOTER = 1.0          # inches reserved under the panels for the legend and the footnote


def levels(rows: list[dict], knob: Knob) -> list[str]:
    return sorted({r[knob.name] for r in rows}, key=knob.key)


def is_reference(level: str, knob: Knob) -> bool:
    return knob.key(level) < 0


def panel_title(p, c, n=None) -> str:
    return f"p = {p}, {C_NAME[c].split(' (')[0]}" + (f", n = {N_LABEL[n]}" if n else "")


def suptitle(fig, text: str, note: str) -> None:
    fig.suptitle(text + (f"\n[{note}]" if note else ""), fontsize=9.5, color=INK, x=0.02, ha="left")


def footnote(fig, text: str, y: float = 0.01) -> None:
    """One line under the figure saying how to read it: which graph of the path, what is averaged."""
    fig.text(0.02, y, text, fontsize=7.2, color=INK_2, ha="left", va="bottom", wrap=True)


def footer_layout(fig, top: float = 0.96) -> tuple[tuple, float]:
    """The tight-layout rect that leaves ``FOOTER`` inches free at the bottom, and the legend's y."""
    h = fig.get_size_inches()[1]
    return (0, (FOOTER + 0.1) / h, 1, top), (FOOTER - 0.45) / h


def rule_of(metric: str) -> str:
    """The rule a metric or rule name belongs to: ``ebic1_search_f1`` -> ``ebic1_search``."""
    if metric in ("max_f1", "best", "aupr"):
        return "max_f1"
    for r in sorted(KNOB_RULES, key=len, reverse=True):
        if metric == r or metric.startswith(r + "_"):
            return r
    return "bic"


def graph_sentence(rule: str) -> str:
    return f"Graph per data set: {GRAPH_OF[rule_of(rule)]}."


def plot_series(ax, ws, rows, ests, value, error, knob: Knob, width: float = 0.12) -> None:
    """One series per estimator over the levels ``ws``: points with error bars, the knob's levels joined
    by a line and the reference level left on its own."""
    for k, est in enumerate(ests):
        col, mk, filled, _ = STYLE[est]
        pts = [(i, r) for i, w in enumerate(ws) for r in rows if (r[knob.name], r["estimator"]) == (w, est)
               and not math.isnan(f(r[value]))]
        if not pts:
            continue
        xs = np.array([i for i, _ in pts]) + (k - (len(ests) - 1) / 2) * width
        ys = np.array([f(r[value]) for _, r in pts])
        es = np.array([f(r[error]) for _, r in pts])
        ax.errorbar(xs, ys, yerr=np.where(np.isfinite(es), es, 0), color=col, lw=0, elinewidth=1.0, capsize=2.5, zorder=2)
        own = [j for j, (i, _) in enumerate(pts) if not is_reference(ws[i], knob)]
        if len(own) > 1:
            ax.plot(xs[own], ys[own], color=col, lw=1.4 if filled else 1.0, ls="-" if filled else "--", zorder=3)
        ax.plot(xs, ys, ls="none", marker=mk, ms=6.5, color=col if filled else SURFACE, mec=SURFACE if filled else col,
                mew=1.3, zorder=4)


def fig_gain_by_wmin(table, out, metric="max_f1", ests=("MCP", "MCP-up", "adaptive"), ps=None,
                     ns=("1000", "1e4", "inf"), suffix="", note="", knob: Knob = WMIN) -> None:
    """The paired gain over the lasso at the same level against the level; the reference (the thesis
    generator) as the leftmost point.  Rows (p, C), columns n."""
    rows = [r for r in table if r["metric"] == metric]
    ps = sorted({r["p"] for r in rows}, key=int) if ps is None else ps
    panels = [(p, c) for p in ps for c in ("C2I", "Cresc") if any(r["p"] == p and r["c"] == c for r in rows)]
    ns = [n for n in ns if any(r["n"] == n for r in rows)]
    if not panels or not ns:
        return
    ws = levels(rows, knob)
    fig, axes = plt.subplots(len(panels), len(ns), figsize=(max(3.4 * len(ns) + 0.8, 7.0), 2.6 * len(panels) + FOOTER),
                             sharey=True, squeeze=False)
    x = np.arange(len(ws))
    for a, (p, c) in enumerate(panels):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0, color=GREY, lw=0.8, zorder=1)
            if any(is_reference(w, knob) for w in ws):
                ax.axvline(0.5, color=GREY, lw=0.6, ls=":", zorder=1)
            plot_series(ax, ws, [r for r in rows if (r["p"], r["c"], r["n"]) == (p, c, n)], ests, "gain", "gain_se", knob)
            ax.set_xticks(x)
            ax.set_xticklabels([knob.label(w) for w in ws], fontsize=7.5)
            ax.set_title(panel_title(p, c, n), loc="left", fontsize=8.5, color=INK)
        axes[a][0].set_ylabel(f"gain over the lasso\n({METRIC_LABEL.get(metric, metric)})", fontsize=8.5, color=INK_2)
    suptitle(fig, f"{knob.title}, direct loss -- paired gain over the lasso drawn at the same level", note)
    rect, y_leg = footer_layout(fig)
    fig.tight_layout(rect=rect)
    legend(fig, ests, y=y_leg, ncol=3)
    footnote(fig, graph_sentence(metric) + "  Gain = mean over the cell's graphs of F₁(estimator) − F₁(lasso) on the "
                  "same graph (same seed, same generator); bars = one standard error of that mean.")
    save(fig, out, f"gain_by_{knob.tag}" + ("" if metric == "max_f1" else f"_{metric}") + suffix)


def fig_levels_by_wmin(table, out, n="1e4", metrics=("bic_f1", "bic_precision", "bic_recall"), ests=KNOB_ESTIMATORS,
                       suffix="", note="", knob: Knob = WMIN) -> None:
    """The level of each estimator (not the gain) against the knob at one n: rows the metrics,
    columns (p, C); error bars are the standard error of the mean over the graphs."""
    rows = [r for r in table if r["n"] == n and r["metric"] in metrics and r["estimator"] in ests]
    panels = sorted({(r["p"], r["c"]) for r in rows}, key=lambda t: (int(t[0]), t[1]))
    if not panels:
        return
    ws = levels(rows, knob)
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(max(3.4 * len(panels) + 0.6, 7.0), 2.5 * len(metrics) + FOOTER),
                             sharey="row", squeeze=False)
    x = np.arange(len(ws))
    for a, metric in enumerate(metrics):
        for b, (p, c) in enumerate(panels):
            ax = axes[a][b]
            style_axes(ax)
            if any(is_reference(w, knob) for w in ws):
                ax.axvline(0.5, color=GREY, lw=0.6, ls=":", zorder=1)
            plot_series(ax, ws, [r for r in rows if (r["p"], r["c"], r["metric"]) == (p, c, metric)], ests, "mean", "mean_se",
                        knob, width=0.1)
            ax.set_xticks(x)
            ax.set_xticklabels([knob.label(w) for w in ws], fontsize=7.5)
            ax.set_ylim(0, 1.02)
            if a == 0:
                ax.set_title(panel_title(p, c), loc="left", fontsize=8.5, color=INK)
        axes[a][0].set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=8.5, color=INK_2)
    suptitle(fig, f"{knob.title}, direct loss -- the level of each estimator, n = {N_LABEL[n]}", note)
    rect, y_leg = footer_layout(fig)
    fig.tight_layout(rect=rect)
    legend(fig, ests, y=y_leg, ncol=len(ests))
    footnote(fig, graph_sentence(metrics[0]) + "  Each point: the mean over the cell's graphs; bars = one standard error.")
    rule = rule_of(metrics[0])
    save(fig, out, f"levels_by_{knob.tag}_n{n}" + ("" if rule == "bic" else f"_{rule}") + suffix)


def stacked(ax, x, cells: dict, keys, parts, count_key="n_edges", y_text=1.03, fmt="{:.0f}", prefix="") -> None:
    """Stacked shares of ``parts`` at the positions ``x`` for ``keys`` (looked up in ``cells``), a number
    above each bar."""
    bottom = np.zeros(len(keys))
    for part, colour, _ in parts:
        h = np.array([f(cells[k][part]) if k in cells and not math.isnan(f(cells[k][part])) else 0.0 for k in keys])
        ax.bar(x, h, bottom=bottom, width=0.66, color=colour, edgecolor=SURFACE, lw=0.6, zorder=2)
        bottom += h
    for i, k in enumerate(keys):
        if k in cells and not math.isnan(f(cells[k][count_key])):
            ax.text(i, y_text, prefix + fmt.format(f(cells[k][count_key])), ha="center", va="bottom", fontsize=6.5, color=INK_2)


def fig_orientation_by_wmin(orient, out, n="1e4", rule="bic", ests=KNOB_ESTIMATORS, suffix="", note="",
                            knob: Knob = WMIN) -> None:
    """What the rule's graph does with the true single edges, per estimator (columns) and level (one
    stacked bar each): right direction only / both directions / reverse only / missed, as the mean share
    per graph; the false positives per graph above each bar.  Rows (p, C)."""
    rows = [r for r in orient if r["n"] == n and r["rule"] == rule and r["estimator"] in ests]
    panels = sorted({(r["p"], r["c"]) for r in rows}, key=lambda t: (int(t[0]), t[1]))
    if not panels:
        return
    ws = levels(rows, knob)
    fig, axes = plt.subplots(len(panels), len(ests), figsize=(2.7 * len(ests) + 0.6, 2.9 * len(panels)), sharey=True, squeeze=False)
    x = np.arange(len(ws))
    for a, (p, c) in enumerate(panels):
        for b, est in enumerate(ests):
            ax = axes[a][b]
            style_axes(ax)
            cells = {r[knob.name]: r for r in rows if (r["p"], r["c"], r["estimator"]) == (p, c, est)}
            stacked(ax, x, cells, ws, SINGLE_PARTS, count_key="fp", fmt="{:.1f}", prefix="fp ")
            ax.set_xticks(x)
            ax.set_xticklabels([knob.label(w) for w in ws], fontsize=7, rotation=35, ha="right")
            ax.set_ylim(0, 1.16)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_title(f"{STYLE[est][3]}\n{panel_title(p, c)}", loc="left", fontsize=8, color=INK)
        axes[a][0].set_ylabel("share of the true single edges", fontsize=8.5, color=INK_2)
    graph = "the " + RULE_SHORT.get(rule, rule)
    suptitle(fig, f"{knob.title}, direct loss -- the true single edges in {graph}, n = {N_LABEL[n]}  (fp: false positives per graph)", note)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.legend(handles=[Patch(facecolor=colour, label=label) for _, colour, label in SINGLE_PARTS],
               loc="lower center", ncol=len(SINGLE_PARTS), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.05))
    footnote(fig, graph_sentence(rule) + "  A bar splits the TRUE single edges (true edges outside 2-cycles) of a graph by their fate "
                  "and always sums to one; it is the mean over the cell's graphs of those shares.  The estimated graph's size is not "
                  "shown: fp = its false positives per graph (hedges + reversals + edges between unconnected nodes).")
    save(fig, out, f"orientation_by_{knob.tag}_n{n}_{rule}{suffix}")


def fig_pair_outcomes(orient, out, n="1e4", rule="bic", ests=KNOB_ESTIMATORS, suffix="", note="", knob: Knob = WMIN) -> None:
    """The true 2-cycles (top: both directions kept / one / neither, the identified share above) and
    the true single edges (bottom: the four fates, the false positives above) in the rule's graph, per
    estimator (columns) and level; one figure per (p, C)."""
    rows = [r for r in orient if r["n"] == n and r["rule"] == rule and r["estimator"] in ests]
    panels = sorted({(r["p"], r["c"]) for r in rows}, key=lambda t: (int(t[0]), t[1]))
    if not rows:
        return
    for p, c in panels:
        sub = [r for r in rows if (r["p"], r["c"]) == (p, c)]
        ws = levels(sub, knob)
        ests_here = [e for e in ests if any(r["estimator"] == e for r in sub)]
        fig, axes = plt.subplots(2, len(ests_here), figsize=(2.7 * len(ests_here) + 0.6, 6.2), sharey=True, squeeze=False)
        x = np.arange(len(ws))
        for b, est in enumerate(ests_here):
            cells = {r[knob.name]: r for r in sub if r["estimator"] == est}
            ax = axes[0][b]
            style_axes(ax)
            stacked(ax, x, cells, ws, PAIR_PARTS, count_key="identified", fmt="{:.2f}", prefix="id ")
            ax.set_title(f"{STYLE[est][3]}", loc="left", fontsize=8.5, color=INK)
            ax = axes[1][b]
            style_axes(ax)
            stacked(ax, x, cells, ws, SINGLE_PARTS, count_key="fp", fmt="{:.1f}", prefix="fp ")
            for ax in (axes[0][b], axes[1][b]):
                ax.set_xticks(x)
                ax.set_xticklabels([knob.label(w) for w in ws], fontsize=7, rotation=35, ha="right")
                ax.set_ylim(0, 1.18)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axes[0][0].set_ylabel("share of the true 2-cycles", fontsize=8.5, color=INK_2)
        axes[1][0].set_ylabel("share of the true single edges", fontsize=8.5, color=INK_2)
        graph = "the " + RULE_SHORT.get(rule, rule)
        suptitle(fig, f"{knob.title}, direct loss -- the true 2-cycles (top) and the true single edges (bottom) in {graph}; "
                      f"{panel_title(p, c, n)}", note)
        fig.tight_layout(rect=(0, 0.14, 1, 0.94))
        fig.legend(handles=[Patch(facecolor=colour, label=label) for _, colour, label in PAIR_PARTS]
                   + [Patch(facecolor=colour, label=label) for _, colour, label in SINGLE_PARTS],
                   loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.045))
        footnote(fig, graph_sentence(rule) + "  Top: mean over the graphs with a 2-cycle of the share of their true 2-cycles kept with "
                      "both directions / one / neither; id = the share of graphs whose truth is identified on its support.  Bottom: the "
                      "true single edges as in the orientation figure; fp = false positives per graph.")
        save(fig, out, f"pair_outcomes_by_{knob.tag}_p{p}_{c}_n{n}_{rule}{suffix}")


def fig_orientation_by_weight(bins, out, n="1e4", rule="bic", var="abs_w", ests=KNOB_ESTIMATORS, suffix="", note="",
                              knob: Knob = WMIN) -> None:
    """Within each level (columns), the outcome of the true single edges in the rule's graph by |w| bin
    (x), one row per estimator; the edge count above each bar.  Bins a level does not populate are
    left out, so the uniform laws stop at 1 and U[0.5, 1] has two bins."""
    rows = [r for r in bins if r["n"] == n and r["rule"] == rule and r["var"] == var and r["estimator"] in ests]
    panels = sorted({(r["p"], r["c"]) for r in rows}, key=lambda t: (int(t[0]), t[1]))
    if not rows:
        return
    for p, c in panels:
        sub = [r for r in rows if (r["p"], r["c"]) == (p, c)]
        ws = levels(sub, knob)
        ests_here = [e for e in ests if any(r["estimator"] == e for r in sub)]
        fig, axes = plt.subplots(len(ests_here), len(ws), figsize=(2.3 * len(ws) + 0.8, 2.3 * len(ests_here) + 0.6),
                                 sharey=True, squeeze=False)
        for a, est in enumerate(ests_here):
            for b, w in enumerate(ws):
                ax = axes[a][b]
                style_axes(ax)
                cells = {r["weight_bin"]: r for r in sub if (r["estimator"], r[knob.name]) == (est, w)}
                keys = [wb for wb in BIN_LABELS if wb in cells and int(float(cells[wb]["n_edges"])) > 0]
                stacked(ax, np.arange(len(keys)), cells, keys, SINGLE_PARTS)
                ax.set_xticks(np.arange(len(keys)))
                ax.set_xticklabels(keys, fontsize=6.5, rotation=35, ha="right")
                ax.set_xlim(-0.6, max(len(keys), 1) - 0.4)
                ax.set_ylim(0, 1.16)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if a == 0:
                    ax.set_title(knob.label(w), loc="left", fontsize=8.5, color=INK)
            axes[a][0].set_ylabel(f"{STYLE[est][3]}\nshare of the true single edges", fontsize=7.5, color=INK_2)
        graph = "the " + RULE_SHORT.get(rule, rule)
        what = "|w| on the generator's scale" if var == "abs_w" else "standardised |w| = |M_ij| s_j / s_i"
        suptitle(fig, f"{knob.title}, direct loss -- the true single edges in {graph} by weight within each level -- {what}; "
                      f"p = {p}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}  (numbers: edges in the bin)", note)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.legend(handles=[Patch(facecolor=colour, label=label) for _, colour, label in SINGLE_PARTS],
                   loc="lower center", ncol=len(SINGLE_PARTS), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.025))
        footnote(fig, graph_sentence(rule) + "  Shares of the true single edges of the bin, pooled over all graphs of the cell "
                      "(half-open bins on |w|); the number above a bar is the edge count behind it.")
        tag = "" if var == "abs_w" else "_std"
        save(fig, out, f"orientation_by_weight_p{p}_{c}_n{n}_{rule}{tag}{suffix}")


def fig_bin_across_laws(bins, out, n="1e4", rule="bic", var="abs_w", ests=KNOB_ESTIMATORS,
                        shown=("0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"), suffix="", note="", knob: Knob = WMIN) -> None:
    """One |w| bin (columns) across the levels (x): the same weight inside a graph without weak edges
    against one with them.  Rows the estimators; the edge count above each bar."""
    rows = [r for r in bins if r["n"] == n and r["rule"] == rule and r["var"] == var and r["estimator"] in ests
            and r["weight_bin"] in shown]
    panels = sorted({(r["p"], r["c"]) for r in rows}, key=lambda t: (int(t[0]), t[1]))
    if not rows:
        return
    for p, c in panels:
        sub = [r for r in rows if (r["p"], r["c"]) == (p, c)]
        ws = levels(sub, knob)
        cols = [wb for wb in shown if any(r["weight_bin"] == wb for r in sub)]
        ests_here = [e for e in ests if any(r["estimator"] == e for r in sub)]
        fig, axes = plt.subplots(len(ests_here), len(cols), figsize=(2.4 * len(cols) + 0.8, 2.3 * len(ests_here) + 0.6),
                                 sharey=True, squeeze=False)
        for a, est in enumerate(ests_here):
            for b, wb in enumerate(cols):
                ax = axes[a][b]
                style_axes(ax)
                cells = {r[knob.name]: r for r in sub if (r["estimator"], r["weight_bin"]) == (est, wb)}
                stacked(ax, np.arange(len(ws)), cells, ws, SINGLE_PARTS)
                ax.set_xticks(np.arange(len(ws)))
                ax.set_xticklabels([knob.label(w) for w in ws], fontsize=6.5, rotation=35, ha="right")
                ax.set_ylim(0, 1.16)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if a == 0:
                    ax.set_title(f"|w| in {wb}", loc="left", fontsize=8.5, color=INK)
            axes[a][0].set_ylabel(f"{STYLE[est][3]}\nshare of the true single edges", fontsize=7.5, color=INK_2)
        graph = "the " + RULE_SHORT.get(rule, rule)
        what = "|w| on the generator's scale" if var == "abs_w" else "standardised |w|"
        suptitle(fig, f"{knob.title}, direct loss -- the same weight bin across the levels -- {what}, the true single edges in {graph}; "
                      f"p = {p}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}  (numbers: edges in the bin)", note)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.legend(handles=[Patch(facecolor=colour, label=label) for _, colour, label in SINGLE_PARTS],
                   loc="lower center", ncol=len(SINGLE_PARTS), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.025))
        footnote(fig, graph_sentence(rule) + "  Shares of the true single edges of the bin, pooled over all graphs of the cell; "
                      "a level without edges in the bin leaves its slot empty.")
        tag = "" if var == "abs_w" else "_std"
        save(fig, out, f"orientation_bin_across_laws_p{p}_{c}_n{n}_{rule}{tag}{suffix}")


def pick_n(rows: list[dict]) -> str | None:
    ns = {r["n"] for r in rows}
    if not ns:
        return None
    return "1e4" if "1e4" in ns else min(ns, key=lambda n: math.inf if n == "inf" else float(n))


def draw(root: Path, out: Path, n: str | None, suffix: str, note: str, knob: Knob = WMIN,
         gain_ests=("MCP", "MCP-up", "adaptive"), ests=KNOB_ESTIMATORS, bins: bool = True) -> None:
    """Every figure of one knob from the CSVs under ``root`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    knob_csv = root / f"campaign_{knob.tag}.csv"
    if not knob_csv.exists():
        print(f"{knob_csv} is missing (run the numbers script first)")
        return
    table = read(knob_csv)
    present = lambda metric: any(r["metric"] == metric and not math.isnan(f(r["gain"])) for r in table)  # noqa: E731
    for metric in ["max_f1"] + [f"{r}_f1" for r in KNOB_RULES]:
        if present(metric):
            fig_gain_by_wmin(table, out, metric, ests=gain_ests, suffix=suffix, note=note, knob=knob)
    n = n or pick_n(table)
    if n:
        for rule in KNOB_RULES:
            if present(f"{rule}_f1"):
                fig_levels_by_wmin(table, out, n, (f"{rule}_f1", f"{rule}_precision", f"{rule}_recall"), ests=ests,
                                   suffix=suffix, note=note, knob=knob)
    orient_path = root / f"campaign_{knob.tag}_orientation.csv"
    if orient_path.exists() and n:
        orient = read(orient_path)
        for rule in KNOB_RULES:
            fig_orientation_by_wmin(orient, out, n, rule, ests=ests, suffix=suffix, note=note, knob=knob)
            fig_pair_outcomes(orient, out, n, rule, ests=ests, suffix=suffix, note=note, knob=knob)
    bins_path = root / f"campaign_{knob.tag}_bins.csv"
    if bins and bins_path.exists() and n:
        table_bins = read(bins_path)
        for rule in KNOB_RULES:
            for var in (("abs_w", "abs_w_std") if rule == "bic" else ("abs_w",)):
                fig_orientation_by_weight(table_bins, out, n, rule, var, ests=ests, suffix=suffix, note=note, knob=knob)
                fig_bin_across_laws(table_bins, out, n, rule, var, ests=ests, suffix=suffix, note=note, knob=knob)


def parse(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--out", type=Path, default=None, help="figure directory (default <root>/figures)")
    ap.add_argument("--n", default=None, help="the n of the level and orientation figures (default 1e4 if present, else the smallest)")
    ap.add_argument("--suffix", default="", help="appended to every figure name")
    ap.add_argument("--note", default="", help="appended to every title, e.g. 'laptop rehearsal: p = 10, one replicate'")
    args = ap.parse_args()
    args.out = args.out or args.root / "figures"
    return args


def main() -> None:
    args = parse("E3 figures: the beta-min knob")
    draw(args.root, args.out, args.n, args.suffix, args.note, WMIN)


if __name__ == "__main__":
    main()
