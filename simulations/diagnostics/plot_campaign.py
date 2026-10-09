#!/usr/bin/env python3
"""Figures for the campaign of October 2026 (simulations/S4_campaign.md).  **Run locally.**

Reads the tables that simulations/diagnostics/campaign.py writes into runs/campaign/ and
draws, into runs/campaign/figures/:

  twobytwo_<metric>      the 2 x 2 of the campaign: rows p = 10, 20; columns C = 2I and the
                         rescaled C; x = n; one series per estimator (direct loss)
  by_true_c_<metric>     paired difference to the lasso with the same C, per setting of the
                         true C (columns), rescaled C, rows p = 10, 20
  selection_p20          p = 20, rescaled C: the same estimators under the three ways to get one
                         graph (oracle lambda, BIC-selected, after the BIC search)
  orientation_p20        what the BIC-selected graphs consist of (correct / hedged / reversed /
                         false edges), p = 20, n = 1e4, rescaled C
  search_ceilings        the BIC search from three estimators' graphs, from random starts
                         (Amendola et al. 2020) and from the truth, rescaled C
  loglik_p10             wave 3: the log-likelihood loss, lasso and MCP in both path orders
  by_p                   wave 4 with wave 1: the six estimators of the thesis figure over
                         p = 10 ... 50 at n = 1000, both C; rows max_f1, aupr, bic_f1
  gain_by_p              the same as paired differences to the lasso with the same C
  restarts               wave 5b: F1 of the best of the first r randomly drawn starting graphs
                         of the pure search, r = 1 ... 100, sparse and uniform starts, against
                         the 10 of wave 2 and the lasso-based search (campaign_restarts.csv)
  selection_checks       waves 5a and 5c: the BIC with the likelihood refit, and the extended
                         term inside the search, next to the campaign's rule
  bic_vs_ebic            the plain BIC against Dettling's extended BIC (gamma = 0.5, 1) as the
                         rule that picks one graph: F1 difference and edges selected relative to
                         the truth, over p and over n; the search with the term inside where run
  With --rule ebic1 the rule-dependent figures are redrawn with the extended BIC (gamma = 1) as
  the selection rule and, where waves 5c / 6 ran it, inside the search, into figures_ebic1/.

Series carry a fixed hue per estimator (lasso blue, MCP orange, SCAD aqua, LLA magenta,
adaptive lasso green) and a distinct marker; the standard paths are hollow, the dense-start
paths filled.  Palette validated for colour-vision deficiency with the dataviz checks (the
magenta-aqua pair is at the floor and is separated by the markers).  Every number plotted is in
the CSVs next to the figures.

    python simulations/diagnostics/plot_campaign.py
    python simulations/diagnostics/plot_campaign.py --root runs/campaign
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bidirectional import save  # noqa: E402
from plot_figures import INK, INK_2, style_axes  # noqa: E402

SURFACE = "#fcfcfb"
GREY = "#8f8d88"
#: estimator -> (hue, marker, filled, label)
STYLE = {
    "lasso": ("#2a78d6", "o", True, "lasso"),
    "MCP": ("#eb6834", "s", False, "MCP, standard path"),
    "SCAD": ("#1baf7a", "^", False, "SCAD, standard path"),
    "MCP-up": ("#eb6834", "s", True, "MCP, dense → sparse"),
    "SCAD-up": ("#1baf7a", "^", True, "SCAD, dense → sparse"),
    "MCP-lla": ("#e87ba4", "D", True, "MCP by LLA"),
    "adaptive": ("#008300", "v", True, "adaptive lasso"),
    "lasso-up": ("#2a78d6", "o", False, "lasso, dense → sparse"),
    "search-pure": (GREY, "x", True, "greedy search from random starts"),
    "search-truth": (INK_2, "+", True, "greedy search from the truth"),
}
MAIN = ("lasso", "MCP", "MCP-up", "SCAD-up", "MCP-lla", "adaptive")
NS = ("1000", "1e4", "inf")
N_LABEL = {"1000": "10³", "1e4": "10⁴", "inf": "∞"}
C_NAME = {"C2I": "C = 2I (Dettling's pipeline)", "Cresc": "rescaled C"}
TRUE_C = (("C_ID", "true C = 2I"), ("C_Random_Min_Diag", "diagonal, U[2, 4]"),
          ("C_Random_Diag", "diagonal, U[0.5, 4]"), ("C_Random_Full", "non-diagonal"))
METRIC_NAME = {"max_f1": "F₁ at the oracle λ (best on the path)",
               "bic_f1": "F₁ of the BIC-selected graph",
               "search_f1": "F₁ after the BIC search",
               "aupr": "area under the precision–recall curve"}
#: short panel titles of the two data-driven rules; replaced under --rule ebic1
RULE_SHORT = {"bic_f1": "BIC-selected", "search_f1": "after the BIC search"}
X = np.arange(len(NS))


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def series(ax, xs, y, est, err=None, lw=1.8):
    col, mk, filled, label = STYLE[est]
    y = np.asarray(y, float)
    if err is not None:
        ax.errorbar(xs, y, yerr=err, color=col, lw=0, elinewidth=1.0, capsize=2.5, zorder=2)
    ax.plot(xs, y, color=col, lw=lw if filled else 1.1, ls="-" if filled else "--", zorder=3)
    ax.plot(xs, y, ls="none", marker=mk, ms=7, color=col if filled else SURFACE,
            mec=SURFACE if filled else col, mew=1.3, zorder=4, label=label)


def legend(fig, ests, y=1.0, ncol=3):
    handles = [Line2D([], [], color=STYLE[e][0], marker=STYLE[e][1], ls="-" if STYLE[e][2] else "--",
                      lw=1.6 if STYLE[e][2] else 1.1, ms=7,
                      markerfacecolor=STYLE[e][0] if STYLE[e][2] else SURFACE,
                      markeredgecolor=SURFACE if STYLE[e][2] else STYLE[e][0], markeredgewidth=1.3,
                      label=STYLE[e][3]) for e in ests]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=9)


def dodge(k, n_series):
    return (k - (n_series - 1) / 2) * 0.05


# --------------------------------------------------------------------------- #


def fig_twobytwo(means, metric, out, ests=MAIN):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), sharey=True, squeeze=False)
    for a, p in enumerate(("10", "20")):
        for b, c in enumerate(("C2I", "Cresc")):
            ax = axes[a][b]
            style_axes(ax)
            for k, est in enumerate(ests):
                rr = {r["n"]: r for r in means if r["loss"] == "direct" and r["p"] == p
                      and r["c"] == c and r["estimator"] == est}
                if len(rr) < len(NS):
                    continue
                series(ax, X + dodge(k, len(ests)), [f(rr[n][metric]) for n in NS], est,
                       err=[f(rr[n][metric + "_se"]) for n in NS])
            ax.set_xticks(X)
            ax.set_xticklabels([N_LABEL[n] for n in NS])
            ax.set_title(f"p = {p}, {C_NAME[c]}", loc="left", fontsize=9.5, color=INK)
            if a == 1:
                ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel(METRIC_NAME[metric], fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests)
    save(fig, out, f"twobytwo_{metric}")


def fig_by_true_c(paired, metric, out, c="Cresc", ests=("MCP", "MCP-up", "SCAD-up", "MCP-lla", "adaptive")):
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), sharey=True, squeeze=False)
    for a, p in enumerate(("10", "20")):
        for b, (tc, tc_name) in enumerate(TRUE_C):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0, color=INK_2, lw=0.9, zorder=1)
            for k, est in enumerate(ests):
                rr = {r["n"]: r for r in paired if r["loss"] == "direct" and r["p"] == p and r["c"] == c
                      and r["estimator"] == est and r["reference"] == "same_c" and r["true_c"] == tc}
                if len(rr) < len(NS):
                    continue
                series(ax, X + dodge(k, len(ests)), [f(rr[n][metric + "_diff"]) for n in NS], est)
            ax.set_xticks(X)
            ax.set_xticklabels([N_LABEL[n] for n in NS])
            ax.set_title(f"p = {p}, {tc_name}", loc="left", fontsize=9.5, color=INK)
            if a == 1:
                ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel(f"{METRIC_NAME[metric]}:\ndifference to the lasso, same C",
                              fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=5)
    save(fig, out, f"by_true_c_{metric}")


def fig_selection(means, out, p="20", c="Cresc", ests=("lasso", "MCP", "MCP-up", "adaptive")):
    rules = (("max_f1", "oracle λ"), ("bic_f1", RULE_SHORT["bic_f1"]), ("search_f1", RULE_SHORT["search_f1"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=True, squeeze=False)
    for b, (metric, name) in enumerate(rules):
        ax = axes[0][b]
        style_axes(ax)
        for k, est in enumerate(ests):
            rr = {r["n"]: r for r in means if r["loss"] == "direct" and r["p"] == p and r["c"] == c
                  and r["estimator"] == est}
            if len(rr) < len(NS):
                continue
            series(ax, X + dodge(k, len(ests)), [f(rr[n][metric]) for n in NS], est,
                   err=[f(rr[n][metric + "_se"]) for n in NS])
        ax.set_xticks(X)
        ax.set_xticklabels([N_LABEL[n] for n in NS])
        ax.set_title(f"{name}", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
    axes[0][0].set_ylabel(f"directed F₁, p = {p}, {C_NAME[c]}", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=4)
    save(fig, out, f"selection_p{p}")


def fig_orientation(rows, out, p="20", n="1e4", c="Cresc",
                    ests=("lasso", "MCP", "MCP-up", "MCP-lla", "adaptive")):
    """Composition of the BIC-selected graph, mean counts per graph (diagonal true C only)."""
    parts = (("correct", "#2a78d6", "true edge, right direction"),
             ("hedged", "#8fb8ea", "true edge, both directions kept"),
             ("reversed", "#eb6834", "true edge, reversed"),
             ("fp", GREY, "false edge"))
    sel = [r for r in rows if r["loss"] == "direct" and r["p"] == p and r["n"] == n and r["c"] == c
           and r["c_choice"] != "C_Random_Full"]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    style_axes(ax)
    ax.grid(False, axis="y")
    ys = np.arange(len(ests))[::-1]
    truth = np.mean([f(r["n_true_edges"]) for r in sel if r["estimator"] == "lasso"])
    for yi, est in zip(ys, ests):
        g = [r for r in sel if r["estimator"] == est]
        if not g:
            continue
        vals = {"correct": np.mean([f(r["bic_correct"]) + 2 * f(r["bic_both"]) + f(r["bic_half"]) for r in g]),
                "hedged": np.mean([2 * f(r["bic_hedged"]) for r in g]),
                "reversed": np.mean([f(r["bic_reversed"]) for r in g]),
                "fp": np.mean([f(r["bic_fp_single"]) + 2 * f(r["bic_fp_double"]) for r in g])}
        left = 0.0
        for key, col, _ in parts:
            ax.barh(yi, vals[key], left=left, color=col, height=0.62, edgecolor=SURFACE, linewidth=2, zorder=3)
            left += vals[key]
        ax.text(left + 1.0, yi, f"{left:.0f} edges", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(truth, color=INK_2, lw=0.9, ls=":", zorder=4)
    ax.text(truth, len(ests) - 0.35, f"true edges: {truth:.0f}", fontsize=8.5, color=INK_2, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([STYLE[e][3] for e in ests], fontsize=9)
    ax.set_xlabel(f"entries of the BIC-selected graph, mean per graph (p = {p}, n = {N_LABEL[n]}, "
                  f"{C_NAME[c]}, diagonal true C)", fontsize=9, color=INK_2)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    fig.legend(handles=[Patch(facecolor=col, label=name) for _, col, name in parts], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out, f"orientation_p{p}")


def fig_search(means, out, c="Cresc", ests=("lasso", "MCP-up", "adaptive", "search-pure", "search-truth")):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, squeeze=False)
    for b, p in enumerate(("10", "20")):
        ax = axes[0][b]
        style_axes(ax)
        for k, est in enumerate(ests):
            rr = {r["n"]: r for r in means if r["loss"] == "direct" and r["p"] == p and r["c"] == c
                  and r["estimator"] == est}
            if len(rr) < len(NS):
                continue
            series(ax, X + dodge(k, len(ests)), [f(rr[n]["search_f1"]) for n in NS], est,
                   err=[f(rr[n]["search_f1_se"]) for n in NS])
        ax.set_xticks(X)
        ax.set_xticklabels([N_LABEL[n] for n in NS])
        ax.set_title(f"p = {p}, {C_NAME[c]}", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
    axes[0][0].set_ylabel("directed F₁ of the graph the search ends at", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=3)
    save(fig, out, "search_ceilings")


def fig_loglik(means, out, ests=("lasso", "lasso-up", "MCP", "MCP-up")):
    sel = [r for r in means if r["loss"] == "loglik"]
    if not sel:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, squeeze=False)
    for b, c in enumerate(("C2I", "Cresc")):
        ax = axes[0][b]
        style_axes(ax)
        for k, est in enumerate(ests):
            rr = {r["n"]: r for r in sel if r["p"] == "10" and r["c"] == c and r["estimator"] == est}
            if len(rr) < len(NS):
                continue
            series(ax, X + dodge(k, len(ests)), [f(rr[n]["max_f1"]) for n in NS], est,
                   err=[f(rr[n]["max_f1_se"]) for n in NS])
        ax.set_xticks(X)
        ax.set_xticklabels([N_LABEL[n] for n in NS])
        ax.set_title(f"log-likelihood loss, p = 10, {C_NAME[c]}", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
    axes[0][0].set_ylabel(METRIC_NAME["max_f1"], fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=4)
    save(fig, out, "loglik_p10")


PS = ("10", "15", "20", "25", "30", "40", "50")
P_EST = ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")


def fig_by_p(means, out, ests=P_EST, metrics=("max_f1", "aupr", "bic_f1"), n="1000"):
    """Wave 4 together with wave 1's p = 10, 20: the estimators over p at one n."""
    sel = [r for r in means if r["loss"] == "direct" and r["n"] == n and r["p"] in PS]
    ps = sorted({int(r["p"]) for r in sel})
    if len(ps) < 3:
        return
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.6, 2.9 * len(metrics)), sharex=True,
                             sharey="row", squeeze=False)
    for a, metric in enumerate(metrics):
        for b, c in enumerate(("C2I", "Cresc")):
            ax = axes[a][b]
            style_axes(ax)
            for est in ests:
                rr = {int(r["p"]): r for r in sel if r["c"] == c and r["estimator"] == est}
                xs = [p for p in ps if p in rr]
                if len(xs) < 3:
                    continue
                series(ax, xs, [f(rr[p][metric]) for p in xs], est,
                       err=[f(rr[p][f"{metric}_se"]) for p in xs])
            ax.set_xticks(ps)
            if a == 0:
                ax.set_title(f"{C_NAME[c]}, n = {N_LABEL[n]}", loc="left", fontsize=9.5, color=INK)
            if a == len(metrics) - 1:
                ax.set_xlabel("number of nodes p", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel(METRIC_NAME[metric], fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=3)
    save(fig, out, "by_p")


def fig_gain_by_p(paired, out, ests=("MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive"),
                  metrics=("max_f1", "bic_f1"), n="1000"):
    """Paired differences to the lasso with the same C over p (all settings of the true C)."""
    sel = [r for r in paired if r["loss"] == "direct" and r["n"] == n and r["p"] in PS
           and r["reference"] == "same_c" and r["true_c"] == "all"]
    ps = sorted({int(r["p"]) for r in sel})
    if len(ps) < 3:
        return
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.6, 2.9 * len(metrics)), sharex=True,
                             sharey="row", squeeze=False)
    for a, metric in enumerate(metrics):
        for b, c in enumerate(("C2I", "Cresc")):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0.0, color=GREY, lw=0.8, zorder=1)
            for est in ests:
                rr = {int(r["p"]): r for r in sel if r["c"] == c and r["estimator"] == est}
                xs = [p for p in ps if p in rr]
                if len(xs) < 3:
                    continue
                d = [f(rr[p][f"{metric}_diff"]) for p in xs]
                z = [f(rr[p][f"{metric}_z"]) for p in xs]
                se = [abs(di / zi) if zi else math.nan for di, zi in zip(d, z)]
                series(ax, xs, d, est, err=se)
            ax.set_xticks(ps)
            if a == 0:
                ax.set_title(f"{C_NAME[c]}, n = {N_LABEL[n]}", loc="left", fontsize=9.5, color=INK)
            if a == len(metrics) - 1:
                ax.set_xlabel("number of nodes p", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel(f"{METRIC_NAME[metric]}\nminus the lasso's", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, ests, ncol=3)
    save(fig, out, "gain_by_p")


def fig_restarts(root: Path, means, out):
    """Wave 5b: the best of the first r randomly drawn starting graphs, from campaign_restarts.csv
    (cells with per-start supports only, i.e. wave 5), against the 10 starts of wave 2 and
    the lasso-based search on the same graphs.  Rows: how the starts are drawn; columns: n."""
    path = root / "campaign_restarts.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r["f1_mean"] not in ("", "nan")]
    cells = sorted({(r["starts"], r["p"], r["c"], r["n"]) for r in rows})
    if not cells:
        return
    kinds = sorted({k for k, _, _, _ in cells}, reverse=True)          # sparse before uniform
    ns = [n for n in NS if any(c[3] == n for c in cells)]
    fig, axes = plt.subplots(len(kinds), len(ns), figsize=(3.6 * len(ns) + 0.6, 3.2 * len(kinds)),
                             sharey=True, squeeze=False)
    for a, kind in enumerate(kinds):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            style_axes(ax)
            sel = [c for c in cells if c[0] == kind and c[3] == n]
            if not sel:
                ax.set_visible(False)
                continue
            _, p, c, _ = sel[0]
            rr = [r for r in rows if (r["starts"], r["p"], r["c"], r["n"]) == sel[0]]
            for with_empty, col, ls, label in ((0, GREY, "--", "best of the first r random starts"),
                                               (1, INK_2, "-", "the same plus the empty graph")):
                pts = sorted((int(r["r"]), f(r["f1_mean"]), f(r["f1_se"])) for r in rr
                             if r["with_empty"] == str(with_empty))
                ax.errorbar([x for x, _, _ in pts], [y for _, y, _ in pts], yerr=[e for _, _, e in pts],
                            color=col, ls=ls, lw=1.6, marker="o", ms=3.5, capsize=2, label=label, zorder=3)
            ref = {r["estimator"]: f(r["search_f1"]) for r in means
                   if r["p"] == p and r["c"] == c and r["n"] == n and r["loss"] == "direct"}
            for est, label in (("search-pure", "wave 2: 10 sparse starts + empty"),
                               ("lasso", "lasso + BIC search"), ("adaptive", "adaptive lasso + BIC search")):
                if est in ref and not math.isnan(ref[est]):
                    ax.axhline(ref[est], color=STYLE[est][0], lw=1.0,
                               ls=":" if est == "search-pure" else "-", zorder=2)
                    if b == 0:
                        ax.annotate(label, (1.05, ref[est]), fontsize=7, color=STYLE[est][0], va="bottom")
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 10, 30, 100])
            ax.set_xticklabels(["1", "3", "10", "30", "100"])
            ax.set_title(f"{kind} starts, p = {p}, {C_NAME[c]}, n = {N_LABEL.get(n, n)}",
                         loc="left", fontsize=8.5, color=INK)
            if a == len(kinds) - 1:
                ax.set_xlabel("r = randomly drawn starting graphs", fontsize=9, color=INK_2)
        axes[a][0].set_ylabel("F₁ of the graph the best start ends at", fontsize=9, color=INK_2)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, out, "restarts")


def fig_selection_checks(means, rows, out, n="1e4"):
    """Waves 5a and 5c: the selection step under the BIC with the likelihood refit (p = 10) and
    with the extended term inside the search (p = 10, 20), next to the campaign's rule."""
    ests = ("lasso", "MCP-up", "adaptive")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey="row", squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.6]})
    short = {"lasso": "lasso", "MCP-up": "MCP d→s", "adaptive": "adaptive"}
    for a, c in enumerate(("C2I", "Cresc")):
        # ---- left: least-squares refit against likelihood refit, p = 10
        ax = axes[a][0]
        style_axes(ax)
        xs = np.arange(len(ests))
        for j, (suffix, col, mk, label) in enumerate((("", INK_2, "o", "least-squares refit (campaign)"),
                                                     ("-ml", "#b5651d", "D", "likelihood refit (wave 5a)"))):
            for metric, dx, mfc in (("bic_f1", -0.18, "none"), ("search_f1", 0.18, None)):
                ys, es = [], []
                for e in ests:
                    r = [r for r in means if r["loss"] == "direct" and r["p"] == "10" and r["c"] == c
                         and r["n"] == n and r["estimator"] == e + suffix]
                    ys.append(f(r[0][metric]) if r else math.nan)
                    es.append(f(r[0][f"{metric}_se"]) if r else math.nan)
                ax.errorbar(xs + dx + (0.06 if j else -0.06), ys, yerr=es, ls="none", marker=mk, ms=6,
                            color=col, mfc=mfc if mfc else col, mew=1.3, capsize=2,
                            label=f"{label}, {'BIC-selected' if metric == 'bic_f1' else 'after the search'}")
        ax.set_xticks(xs)
        ax.set_xticklabels([STYLE[e][3] for e in ests], fontsize=8)
        ax.set_title(f"refit behind the BIC: p = 10, {C_NAME[c]}", loc="left", fontsize=9, color=INK)
        ax.set_ylabel("F₁ (hollow: selected; filled: after the search)", fontsize=8.5, color=INK_2)
        # ---- right: plain BIC against the extended term in the search, p = 10 and 20
        ax = axes[a][1]
        style_axes(ax)
        labels, k = [], 0
        for p in ("10", "20"):
            for e in ests:
                sel = [r for r in rows if r["estimator"] == e + "-ebic" and r["p"] == p and r["c"] == c
                       and r["n"] == n]
                if not sel:
                    continue
                for col_name, dx, col, mk in (("search_f1", -0.22, INK_2, "o"), ("ebic05_search_f1", 0.0, "#7c6ab8", "s"),
                                              ("ebic1_search_f1", 0.22, "#5a3fa0", "D")):
                    v = np.array([f(r[col_name]) for r in sel])
                    v = v[~np.isnan(v)]
                    ax.errorbar([k + dx], [v.mean()], yerr=[v.std(ddof=1) / math.sqrt(len(v))], ls="none",
                                marker=mk, ms=6, color=col, capsize=2)
                labels.append(f"{short[e]}\np = {p}")
                k += 1
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(f"the term 4γ|E| log p in the search: {C_NAME[c]}", loc="left", fontsize=9, color=INK)
    handles = [Line2D([], [], ls="none", marker="o", color=INK_2, mfc="none", label="least-squares refit, BIC-selected"),
               Line2D([], [], ls="none", marker="o", color=INK_2, label="least-squares refit, after the search"),
               Line2D([], [], ls="none", marker="D", color="#b5651d", mfc="none", label="likelihood refit, BIC-selected"),
               Line2D([], [], ls="none", marker="D", color="#b5651d", label="likelihood refit, after the search"),
               Line2D([], [], ls="none", marker="o", color=INK_2, label="after the search, plain BIC (γ = 0)"),
               Line2D([], [], ls="none", marker="s", color="#7c6ab8", label="after the search, γ = 0.5"),
               Line2D([], [], ls="none", marker="D", color="#5a3fa0", label="after the search, γ = 1")]
    fig.tight_layout()
    fig.suptitle(f"n = {N_LABEL[n]}", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.5)
    save(fig, out, "selection_checks")


def apply_rule(records: list[dict], rule: str) -> list[dict]:
    """Under ``rule="ebic1"`` the selected graph is the one Dettling's extended BIC picks and
    the searched graph the one the search with the term inside ends at: every ``ebic1_*``
    column is copied over its ``bic_*`` / ``search_*`` counterpart (cells without the rescored
    search get an empty ``search_*``, so that they are left out rather than mislabelled), and
    the pure-search estimators are replaced by their ``-e1`` variants."""
    if rule == "bic":
        return records
    out = []
    for r in records:
        r = dict(r)
        if r.get("estimator") in ("search-pure", "search-truth"):
            continue
        if r.get("estimator", "").endswith("-e1"):
            r["estimator"] = r["estimator"][:-3]
        for key in list(r):
            if key.startswith("ebic1_search_"):
                r[key.replace("ebic1_search_", "search_", 1)] = r[key]
            elif key.startswith("ebic1_"):
                r[key.replace("ebic1_", "bic_", 1)] = r[key]
        for key in list(r):
            if key.startswith("search_") and "ebic1_" + key not in r:
                r[key] = ""                                   # no rescored search for this cell
        out.append(r)
    return out


def fig_bic_vs_ebic(rows, out, ests=("lasso", "MCP-up", "adaptive")):
    """The plain BIC against Dettling's extended BIC (gamma = 0.5, 1) as the rule that picks one
    graph from a path: paired difference in F1 and the number of edges selected relative to the
    truth, over p at n = 1000 (left two columns) and over n at p = 20 (right column); both C.
    Where the extended term was also used inside the search (waves 5c and 6), the searched
    graphs are compared the same way (dotted)."""
    sel = [r for r in rows if r["loss"] == "direct" and r["estimator"] in ests]
    if not sel:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), squeeze=False)
    panels = (("C2I", "p", "1000"), ("Cresc", "p", "1000"), ("Cresc", "n", "20"))
    for b, (c, axis, fixed) in enumerate(panels):
        rr = [r for r in sel if r["c"] == c and (r["n"] == fixed if axis == "p" else r["p"] == fixed)]
        xs = sorted({int(r["p"]) for r in rr}) if axis == "p" else [n for n in NS if any(r["n"] == n for r in rr)]
        xpos = xs if axis == "p" else list(range(len(xs)))
        for a, kind in enumerate(("diff", "edges")):
            ax = axes[a][b]
            style_axes(ax)
            if kind == "diff":
                ax.axhline(0.0, color=GREY, lw=0.8, zorder=1)
            for est in ests:
                col = STYLE[est][0]
                for gamma, ls, mk in (("ebic05", ":", "s"), ("ebic1", "-", STYLE[est][1])):
                    ys, es = [], []
                    for x in xs:
                        g = [r for r in rr if r["estimator"] == est and (int(r["p"]) == x if axis == "p" else r["n"] == x)]
                        if kind == "diff":
                            d = np.array([f(r[f"{gamma}_f1"]) - f(r["bic_f1"]) for r in g])
                        else:
                            d = np.array([f(r[f"{gamma}_edges"]) / max(1.0, f(r["n_true_edges"])) for r in g])
                        d = d[~np.isnan(d)]
                        ys.append(d.mean() if len(d) else math.nan)
                        es.append(d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan)
                    if gamma == "ebic1":
                        series(ax, xpos, ys, est, err=es)
                    else:
                        ax.plot(xpos, ys, color=col, ls=ls, lw=1.2, marker=mk, ms=4, mfc=SURFACE, zorder=3)
                if kind == "edges":                         # the plain BIC's edge ratio, dashed, same hue
                    ys = []
                    for x in xs:
                        g = [r for r in rr if r["estimator"] == est and (int(r["p"]) == x if axis == "p" else r["n"] == x)]
                        d = np.array([f(r["bic_edges"]) / max(1.0, f(r["n_true_edges"])) for r in g])
                        ys.append(d[~np.isnan(d)].mean() if len(d) else math.nan)
                    ax.plot(xpos, ys, color=col, ls="--", lw=1.4, zorder=2)
                if kind == "diff":                          # the search, where the term was inside it
                    ys = []
                    for x in xs:
                        g = [r for r in rr if r["estimator"] == est and (int(r["p"]) == x if axis == "p" else r["n"] == x)]
                        d = np.array([f(r.get("ebic1_search_f1", "")) - f(r.get("search_f1", "")) for r in g])
                        d = d[~np.isnan(d)]
                        ys.append(d.mean() if len(d) else math.nan)
                    if not all(math.isnan(y) for y in ys):
                        ax.plot(xpos, ys, color=col, ls=(0, (1, 1)), lw=1.8, marker="x", ms=5, zorder=4)
            if kind == "edges":
                ax.axhline(1.0, color=GREY, lw=0.8, zorder=1)
                ax.set_yscale("log")
            if axis == "p":
                ax.set_xticks(xs)
                if a == 1:
                    ax.set_xlabel("number of nodes p", fontsize=9, color=INK_2)
            else:
                ax.set_xticks(xpos)
                ax.set_xticklabels([N_LABEL[n] for n in xs])
                if a == 1:
                    ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
            if a == 0:
                ax.set_title(f"{C_NAME[c]}, " + (f"n = {N_LABEL[fixed]}" if axis == "p" else f"p = {fixed}"),
                             loc="left", fontsize=9.5, color=INK)
    axes[0][0].set_ylabel("F₁ with the extended BIC minus with the plain BIC", fontsize=8.5, color=INK_2)
    axes[1][0].set_ylabel("selected edges / true edges (log scale)", fontsize=8.5, color=INK_2)
    handles = [Line2D([], [], color=STYLE[e][0], marker=STYLE[e][1], lw=1.8, ms=6, label=STYLE[e][3]) for e in ests]
    handles += [Line2D([], [], color=INK_2, lw=1.8, label="γ = 1, selected graph (top: F₁ difference; bottom: edge ratio)"),
                Line2D([], [], color=INK_2, ls=":", lw=1.2, marker="s", ms=4, mfc=SURFACE, label="γ = 0.5, selected graph"),
                Line2D([], [], color=INK_2, ls="--", lw=1.4, label="plain BIC, selected graph (bottom)"),
                Line2D([], [], color=INK_2, ls=(0, (1, 1)), lw=1.8, marker="x", ms=5, label="γ = 1 inside the search, F₁ difference after the search (top; waves 5c, 6)")]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5)
    save(fig, out, "bic_vs_ebic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--rule", default="bic", choices=("bic", "ebic1"),
                    help="the selection rule behind the 'selected' and 'after the search' series: "
                         "the plain BIC (default; figures/) or Dettling's extended BIC with gamma = 1 "
                         "inside the selection and the search (figures_ebic1/; wave 5c and 6 cells)")
    args = ap.parse_args()
    means = apply_rule(read(args.root / "campaign_means.csv"), args.rule)
    paired = apply_rule(read(args.root / "campaign_paired.csv"), args.rule)
    rows = apply_rule(read(args.root / "campaign_per_dataset.csv"), args.rule)
    out = args.root / ("figures" if args.rule == "bic" else "figures_ebic1")
    if args.rule == "ebic1":
        METRIC_NAME["bic_f1"] = "F₁ of the eBIC-selected graph (γ = 1)"
        METRIC_NAME["search_f1"] = "F₁ after the search with the eBIC (γ = 1)"
        RULE_SHORT["bic_f1"], RULE_SHORT["search_f1"] = "eBIC-selected (γ = 1)", "after the eBIC search (γ = 1)"
    for metric in ("max_f1", "bic_f1", "search_f1", "aupr"):
        fig_twobytwo(means, metric, out)
    for metric in ("max_f1", "bic_f1"):
        fig_by_true_c(paired, metric, out)
    fig_by_true_c(paired, "search_f1", out,
                  ests=("MCP", "MCP-up", "adaptive", "search-pure", "search-truth"))
    fig_selection(means, out)
    fig_orientation(rows, out)
    fig_search(means, out)
    fig_loglik(means, out)
    fig_by_p(means, out)
    fig_gain_by_p(paired, out)
    if args.rule == "bic":                 # the checks of the rule are rule-independent
        fig_restarts(args.root, means, out)
        fig_selection_checks(means, rows, out)
        fig_bic_vs_ebic(rows, out)


if __name__ == "__main__":
    main()
