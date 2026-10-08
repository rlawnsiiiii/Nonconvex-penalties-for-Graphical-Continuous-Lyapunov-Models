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
                         of the pure search, r = 1 ... 100, against the 10 of wave 2 and the
                         lasso-based search (campaign_restarts.csv)

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
    rules = (("max_f1", "oracle λ"), ("bic_f1", "BIC-selected"), ("search_f1", "after the BIC search"))
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
    the lasso-based search on the same graphs."""
    path = root / "campaign_restarts.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r["f1_mean"] not in ("", "nan")]
    cells = sorted({r["cell"] for r in rows})
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells), figsize=(4.3 * len(cells), 3.6), sharey=True, squeeze=False)
    for b, cell in enumerate(cells):
        ax = axes[0][b]
        style_axes(ax)
        rr = [r for r in rows if r["cell"] == cell]
        p, c, n, starts = rr[0]["p"], rr[0]["c"], rr[0]["n"], rr[0]["starts"]
        for with_empty, col, ls, label in ((0, GREY, "--", "best of the first r random starts"),
                                           (1, INK_2, "-", "the same plus the empty graph")):
            pts = sorted((int(r["r"]), f(r["f1_mean"]), f(r["f1_se"])) for r in rr if r["with_empty"] == str(with_empty))
            ax.errorbar([x for x, _, _ in pts], [y for _, y, _ in pts], yerr=[e for _, _, e in pts],
                        color=col, ls=ls, lw=1.6, marker="o", ms=4, capsize=2, label=label, zorder=3)
        ref = {r["estimator"]: f(r["search_f1"]) for r in means
               if r["p"] == p and r["c"] == c and r["n"] == n and r["loss"] == "direct"}
        for est, label in (("search-pure", "wave 2: 10 random starts + empty"),
                           ("lasso", "lasso + BIC search"), ("adaptive", "adaptive lasso + BIC search")):
            if est in ref and not math.isnan(ref[est]):
                ax.axhline(ref[est], color=STYLE[est][0], lw=1.0, ls=":" if est == "search-pure" else "-",
                           zorder=2)
                ax.annotate(label, (1.05, ref[est]), fontsize=7.5, color=STYLE[est][0], va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 30, 50, 100])
        ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "30", "50", "100"])
        ax.set_title(f"p = {p}, {C_NAME[c]}, n = {N_LABEL.get(n, n)}, {starts} starts", loc="left",
                     fontsize=9.5, color=INK)
        ax.set_xlabel("r = number of randomly drawn starting graphs", fontsize=9, color=INK_2)
    axes[0][0].set_ylabel("F₁ of the graph the best start ends at", fontsize=9, color=INK_2)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out, "restarts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    args = ap.parse_args()
    means = read(args.root / "campaign_means.csv")
    paired = read(args.root / "campaign_paired.csv")
    rows = read(args.root / "campaign_per_dataset.csv")
    out = args.root / "figures"
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
    fig_restarts(args.root, means, out)


if __name__ == "__main__":
    main()
