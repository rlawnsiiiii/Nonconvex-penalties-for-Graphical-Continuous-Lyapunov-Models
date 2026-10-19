#!/usr/bin/env python3
"""Figures for E1 and E2 of the plan of 13 October 2026 (``starts.py`` tables and
``campaign_per_dataset.csv``).  **Run locally.**

``starts_bars[_<metric>].png``   paired gain over the lasso per start, MCP dense -> sparse and the adaptive
                                 lasso side by side; rows (p, C), columns n.
``starts_by_pruner[_n<n>].png``  the pruned estimator's F1 against the start's orientation recall (the share of
                                 true single edges whose true direction the start contains), one point per
                                 (start, pruner), panels (p, C): the test of "the start matters because it
                                 already orients most edges".
``threshold_vs_adaptive[_<metric>].png``  the paired difference adaptive - threshold per graph, histograms
                                 per (p, C, n).

    python simulations/diagnostics/plot_starts.py
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
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bidirectional import save  # noqa: E402
from plot_campaign import C_NAME, GREY, INK, INK_2, METRIC_NAME, N_LABEL, STYLE, SURFACE, f, read, style_axes  # noqa: E402
from starts import PRUNERS, STARTS, gid  # noqa: E402

START_LABEL = {"lasso": "lasso\n(dense end)", "exact": "exact fit\n−½CΣ̂⁻¹", "l1": "min-ℓ₁\nexact fit",
               "l2": "min-ℓ₂\nexact fit", "truth0.05": "truth\nτ = 0.05", "truth0.2": "truth\nτ = 0.2"}
THRESHOLD = "#8e5bd1"


def _pruner_style(pruner):
    col, mk, _, label = STYLE[pruner]
    return col, mk, label


def _panel_rows(table, metric, p, c, n):
    return {(r["start"], r["pruner"]): r for r in table
            if (r["p"], r["c"], r["n"], r["metric"]) == (p, c, n, metric)}


def _ps(rows, ps):
    return sorted({r["p"] for r in rows}, key=int) if ps is None else ps


def fig_starts_bars(table, out, metric="max_f1", ps=None, ns=("1000", "1e4", "inf")):
    ps = _ps(table, ps)
    panels = [(p, c) for p in ps for c in ("C2I", "Cresc") if any(r["p"] == p and r["c"] == c for r in table)]
    ns = [n for n in ns if any(r["n"] == n for r in table)]
    if not panels or not ns:
        return
    fig, axes = plt.subplots(len(panels), len(ns), figsize=(3.9 * len(ns) + 0.8, 2.6 * len(panels)), sharey=True,
                             squeeze=False)
    x = np.arange(len(STARTS))
    for a, (p, c) in enumerate(panels):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0, color=GREY, lw=0.8, zorder=1)
            rows = _panel_rows(table, metric, p, c, n)
            for k, pruner in enumerate(PRUNERS):
                col, mk, _ = _pruner_style(pruner)
                for i, start in enumerate(STARTS):
                    r = rows.get((start, pruner))
                    if r is None or math.isnan(f(r["gain_lasso"])):
                        continue
                    xx = x[i] + (k - 0.5) * 0.22
                    if not math.isnan(f(r["gain_lasso_se"])):
                        ax.errorbar(xx, f(r["gain_lasso"]), yerr=f(r["gain_lasso_se"]), color=col, lw=0, elinewidth=1.0,
                                    capsize=2.5, zorder=2)
                    ax.plot(xx, f(r["gain_lasso"]), ls="none", marker=mk, ms=7, color=col, mec=SURFACE, mew=1.3, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([START_LABEL[s] for s in STARTS], fontsize=7)
            ax.set_title(f"p = {p}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}", loc="left", fontsize=8.5, color=INK)
        axes[a][0].set_ylabel(f"gain over the lasso\n({METRIC_NAME.get(metric, metric)})", fontsize=8, color=INK_2)
    handles = [Line2D([], [], color=_pruner_style(pr)[0], marker=_pruner_style(pr)[1], ls="none", ms=7,
                      markeredgecolor=SURFACE, markeredgewidth=1.3, label=_pruner_style(pr)[2]) for pr in PRUNERS]
    fig.suptitle(f"E1: the dense start, pruned two ways -- paired gain over the lasso ({metric})", fontsize=9.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False, fontsize=9)
    save(fig, out, "starts_bars" + ("" if metric == "max_f1" else f"_{metric}"))


def fig_starts_by_pruner(table, content, out, n="1e4", metric="max_f1", ps=None):
    ps = _ps(table, ps)
    panels = [(p, c) for p in ps for c in ("C2I", "Cresc") if any(r["p"] == p and r["c"] == c and r["n"] == n for r in table)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 0.6, 3.3), sharey=True, squeeze=False)
    for a, (p, c) in enumerate(panels):
        ax = axes[0][a]
        style_axes(ax)
        rows = _panel_rows(table, metric, p, c, n)
        rec = {(r["start"], r["pruner"]): f(r["orientation_recall"]) for r in content
               if (r["p"], r["c"], r["n"]) == (p, c, n)}
        lasso = next((f(r["mean"]) - f(r["gain_lasso"]) for r in rows.values() if not math.isnan(f(r["gain_lasso"]))), math.nan)
        if not math.isnan(lasso):
            ax.axhline(lasso, color=STYLE["lasso"][0], lw=1.0, ls="--", zorder=1)
            ax.annotate("lasso", (0.01, lasso), xycoords=("axes fraction", "data"), fontsize=7, color=STYLE["lasso"][0],
                        va="bottom")
        for pruner in PRUNERS:
            col, mk, _ = _pruner_style(pruner)
            for start in STARTS:
                r = rows.get((start, pruner))
                xr = rec.get((start, pruner), math.nan)
                if r is None or math.isnan(xr):
                    continue
                ax.plot(xr, f(r["mean"]), ls="none", marker=mk, ms=7.5, color=col, mec=SURFACE, mew=1.3, zorder=4)
                ax.annotate(start, (xr, f(r["mean"])), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=col)
        ax.set_xlabel("orientation recall of the start", fontsize=8, color=INK_2)
        ax.set_title(f"p = {p}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}", loc="left", fontsize=8.5, color=INK)
    axes[0][0].set_ylabel(METRIC_NAME.get(metric, metric), fontsize=8, color=INK_2)
    handles = [Line2D([], [], color=_pruner_style(pr)[0], marker=_pruner_style(pr)[1], ls="none", ms=7,
                      markeredgecolor=SURFACE, markeredgewidth=1.3, label=_pruner_style(pr)[2]) for pr in PRUNERS]
    fig.suptitle("E1: does the start matter because it already orients the edges?  one point per (start, pruner)",
                 fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False, fontsize=9)
    save(fig, out, "starts_by_pruner" + ("" if n == "1e4" else f"_n{n}"))


def fig_threshold_vs_adaptive(rows, out, metric="max_f1", ps=None, ns=("1000", "1e4", "inf")):
    """Histograms of the paired difference adaptive - threshold per graph."""
    ps = _ps(rows, ps)
    cells = {}
    for r in rows:
        if r["loss"] == "direct" and not r.get("dgp") and r["estimator"] in ("adaptive", "threshold"):
            cells.setdefault((r["p"], r["c"], r["n"], r["estimator"]), {})[gid(r)] = f(r[metric])
    panels = [(p, c) for p in ps for c in ("C2I", "Cresc") if any(k[:2] == (p, c) and k[3] == "threshold" for k in cells)]
    ns = [n for n in ns if any(k[2] == n and k[3] == "threshold" for k in cells)]
    if not panels or not ns:
        return
    fig, axes = plt.subplots(len(panels), len(ns), figsize=(3.0 * len(ns) + 0.6, 2.4 * len(panels)), squeeze=False)
    for a, (p, c) in enumerate(panels):
        for b, n in enumerate(ns):
            ax = axes[a][b]
            style_axes(ax)
            ada, thr = cells.get((p, c, n, "adaptive"), {}), cells.get((p, c, n, "threshold"), {})
            d = np.array([ada[g] - thr[g] for g in thr if g in ada])
            d = d[np.isfinite(d)]
            if d.size:
                ax.hist(d, bins=np.linspace(-0.3, 0.3, 31), color=THRESHOLD, alpha=0.85, zorder=2)
                ax.axvline(d.mean(), color=INK, lw=1.0, zorder=3)
                ax.text(0.98, 0.95, f"mean {d.mean():+.3f}\n{int((d > 0).sum())} > 0, {int((d < 0).sum())} < 0 of {d.size}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
            ax.axvline(0, color=GREY, lw=0.8, zorder=1)
            ax.set_title(f"p = {p}, {C_NAME[c].split(' (')[0]}, n = {N_LABEL[n]}", loc="left", fontsize=8.5, color=INK)
            if a == len(panels) - 1:
                ax.set_xlabel(f"adaptive − threshold ({metric}) per graph", fontsize=8, color=INK_2)
    fig.suptitle("E2: the adaptive lasso against the thresholded lasso on the same pilot", fontsize=9.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "threshold_vs_adaptive" + ("" if metric == "max_f1" else f"_{metric}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    args = ap.parse_args()
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    table = read(args.root / "campaign_starts.csv")
    content = read(args.root / "campaign_start_content.csv")
    for metric in ("max_f1", "bic_f1", "search_f1"):
        fig_starts_bars(table, out, metric)
    for n in sorted({r["n"] for r in table}):
        fig_starts_by_pruner(table, content, out, n=n)
    rows = read(args.root / "campaign_per_dataset.csv")
    for metric in ("max_f1", "bic_f1"):
        fig_threshold_vs_adaptive(rows, out, metric)


if __name__ == "__main__":
    main()
