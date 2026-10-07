#!/usr/bin/env python3
"""Figures for the n-sweep (simulations/S2b_nsweep.md).  **Run locally.**

Reads only the CSVs written by ``nsweep.py`` and writes ``<root>/figures/``:

  metric_vs_n_<metric>.{png,pdf}    mean of the metric against n; lasso / MCP / SCAD;
                                    rows p = 10, 20, columns the three losses      (nsweep_means.csv)
  paired_vs_lasso_<metric>.{png,pdf} paired difference (penalty - lasso) with +-2 se (nsweep_paired_vs_lasso.csv)
  by_c_choice_<penalty>.{png,pdf}   the same difference in max_f1, per C choice     (nsweep_paired_vs_lasso.csv)
  skeleton_vs_directed.{png,pdf}    best-F1 point: paired difference to the lasso in
                                    directed F1 and in skeleton F1                  (nsweep_orientation.csv)
  orientation_counts_p<p>.{png,pdf} reversed and hedged true edges per graph         (nsweep_orientation.csv)
  lasso_by_loss.{png,pdf}           the lasso under the three losses                 (nsweep_means.csv)

    python simulations/diagnostics/plot_nsweep.py [--root runs/nsweep_p10-20]

Encoding as in the other figures of the repository: lasso blue circle, MCP orange square,
SCAD aqua triangle; every series has its own marker, and every number is in the CSVs.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bidirectional import save  # noqa: E402
from plot_figures import C_STYLE, INK, INK_2, style_axes  # noqa: E402

SURFACE = "#fcfcfb"
PEN = {"lasso": ("#2a78d6", "o"), "MCP": ("#eb6834", "s"), "SCAD": ("#1baf7a", "^")}
LOSS_STYLE = {"direct": ("#2a78d6", "o"), "loglik": ("#eb6834", "s"), "frobenius": ("#1baf7a", "^")}
LOSSES = ("direct", "loglik", "frobenius")
LOSS_NAME = {"direct": "direct loss", "loglik": "log-likelihood loss", "frobenius": "Frobenius loss"}
NS = ("1000", "1e4", "1e5", "inf")
N_LABEL = {"1000": "10³", "1e4": "10⁴", "1e5": "10⁵", "inf": "∞"}
METRIC_NAME = {"max_f1": "maximum F1 along the path", "auc": "area under the ROC curve",
               "aupr": "area under the PR curve"}
X = np.arange(len(NS))


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def grid(ps, cols=LOSSES, width=3.7, height=3.0):
    fig, axes = plt.subplots(len(ps), len(cols), figsize=(width * len(cols), height * len(ps)),
                             squeeze=False, sharey="row")
    for a, p in enumerate(ps):
        for b, col in enumerate(cols):
            ax = axes[a][b]
            style_axes(ax)
            ax.set_xticks(X)
            ax.set_xticklabels([N_LABEL[n] for n in NS])
            ax.set_title(f"{LOSS_NAME.get(col, col)}, p = {p}", loc="left", fontsize=9.5, color=INK)
            if a == len(ps) - 1:
                ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
    return fig, axes


def series(ax, y, col, mk, label, err=None, filled=True, lw=1.8):
    if err is not None:
        ax.errorbar(X, y, yerr=err, color=col, lw=0, elinewidth=1.1, capsize=2.5, zorder=2)
    ax.plot(X, y, color=col, lw=lw, zorder=3)
    ax.plot(X, y, ls="none", marker=mk, ms=7, color=col if filled else SURFACE,
            mec=SURFACE if filled else col, mew=1.3, zorder=4, label=label)


def legend(fig, axes, ncol, y=1.0):
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=9)


def fig_metric(means, metric, out):
    ps = sorted({int(r["p"]) for r in means})
    fig, axes = grid(ps)
    for a, p in enumerate(ps):
        for b, loss in enumerate(LOSSES):
            for pen, (col, mk) in PEN.items():
                rr = {r["n"]: r for r in means if int(r["p"]) == p and r["loss"] == loss and r["penalty"] == pen}
                series(axes[a][b], [float(rr[n][metric]) for n in NS], col, mk, pen,
                       err=[float(rr[n][f"{metric}_se"]) for n in NS])
        axes[a][0].set_ylabel(METRIC_NAME[metric], fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, axes, 3)
    save(fig, out, f"metric_vs_n_{metric}")


def fig_paired(paired, metric, out):
    rows = [r for r in paired if r["c_choice"] == "all" and r["metric"] == metric]
    ps = sorted({int(r["p"]) for r in rows})
    fig, axes = grid(ps)
    for a, p in enumerate(ps):
        for b, loss in enumerate(LOSSES):
            ax = axes[a][b]
            ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
            for pen in ("MCP", "SCAD"):
                rr = {r["n"]: r for r in rows if int(r["p"]) == p and r["loss"] == loss and r["penalty"] == pen}
                series(ax, [float(rr[n]["diff"]) for n in NS], *PEN[pen], f"{pen} − lasso",
                       err=[2 * float(rr[n]["se"]) for n in NS])
        axes[a][0].set_ylabel(f"paired difference in\n{METRIC_NAME[metric]}", fontsize=9, color=INK_2)
    for ax in axes.flat:
        ax.set_ylim(top=0.02)
    fig.tight_layout()
    legend(fig, axes, 2)
    save(fig, out, f"paired_vs_lasso_{metric}")


def fig_by_c(paired, pen, out):
    rows = [r for r in paired if r["c_choice"] != "all" and r["metric"] == "max_f1" and r["penalty"] == pen]
    ps = sorted({int(r["p"]) for r in rows})
    fig, axes = grid(ps)
    for a, p in enumerate(ps):
        for b, loss in enumerate(LOSSES):
            ax = axes[a][b]
            ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
            for c, (col, mk, name) in C_STYLE.items():
                rr = {r["n"]: r for r in rows if int(r["p"]) == p and r["loss"] == loss and r["c_choice"] == c}
                series(ax, [float(rr[n]["diff"]) for n in NS], col, mk, name, lw=1.5)
        axes[a][0].set_ylabel(f"{pen} − lasso,\nmaximum F1 (paired)", fontsize=9, color=INK_2)
    for ax in axes.flat:
        ax.set_ylim(top=0.02)
    fig.tight_layout()
    legend(fig, axes, 4)
    save(fig, out, f"by_c_choice_{pen}")


def fig_skeleton(orient, out):
    ps = sorted({int(r["p"]) for r in orient})
    fig, axes = grid(ps)
    for a, p in enumerate(ps):
        for b, loss in enumerate(LOSSES):
            ax = axes[a][b]
            ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
            for pen in ("MCP", "SCAD"):
                rr = {r["n"]: r for r in orient if int(r["p"]) == p and r["loss"] == loss and r["penalty"] == pen}
                col, mk = PEN[pen]
                series(ax, [float(rr[n]["directed_f1_minus_lasso"]) for n in NS], col, mk, f"{pen}: directed F1")
                series(ax, [float(rr[n]["skeleton_f1_minus_lasso"]) for n in NS], col, mk,
                       f"{pen}: skeleton F1", filled=False, lw=1.0)
        axes[a][0].set_ylabel("paired difference to the lasso\nat the best-F1 point", fontsize=9, color=INK_2)
    for ax in axes.flat:
        ax.set_ylim(top=0.02)
    fig.tight_layout()
    legend(fig, axes, 4)
    save(fig, out, "skeleton_vs_directed")


def fig_counts(orient, out):
    kinds = [("reversed", "true single edges found\nin the reversed direction only"),
             ("hedged", "true single edges found\nin both directions")]
    for p in sorted({int(r["p"]) for r in orient}):
        fig, axes = plt.subplots(len(kinds), len(LOSSES), figsize=(3.7 * len(LOSSES), 5.6), squeeze=False)
        for a, (kind, ylab) in enumerate(kinds):
            top = 0.0
            for b, loss in enumerate(LOSSES):
                ax = axes[a][b]
                style_axes(ax)
                ax.set_xticks(X)
                ax.set_xticklabels([N_LABEL[n] for n in NS])
                if a == 0:
                    ax.set_title(f"{LOSS_NAME[loss]}, p = {p}", loc="left", fontsize=9.5, color=INK)
                else:
                    ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
                for pen, (col, mk) in PEN.items():
                    rr = {r["n"]: r for r in orient if int(r["p"]) == p and r["loss"] == loss and r["penalty"] == pen}
                    y = [float(rr[n][kind]) for n in NS]
                    top = max(top, max(y))
                    series(ax, y, col, mk, pen)
                if b:
                    ax.tick_params(labelleft=False)
            for ax in axes[a]:
                ax.set_ylim(0, top * 1.1)
            axes[a][0].set_ylabel(f"{ylab}\n(mean per graph)", fontsize=9, color=INK_2)
        fig.tight_layout()
        legend(fig, axes, 3)
        save(fig, out, f"orientation_counts_p{p}")


def fig_lasso_by_loss(means, out):
    ps = sorted({int(r["p"]) for r in means})
    metrics = ("max_f1", "auc", "aupr")
    fig, axes = plt.subplots(len(ps), len(metrics), figsize=(3.7 * len(metrics), 3.0 * len(ps)), squeeze=False)
    for a, p in enumerate(ps):
        for b, m in enumerate(metrics):
            ax = axes[a][b]
            style_axes(ax)
            ax.set_xticks(X)
            ax.set_xticklabels([N_LABEL[n] for n in NS])
            ax.set_title(f"lasso, {METRIC_NAME[m]}, p = {p}", loc="left", fontsize=9, color=INK)
            for loss, (col, mk) in LOSS_STYLE.items():
                rr = {r["n"]: r for r in means if int(r["p"]) == p and r["loss"] == loss and r["penalty"] == "lasso"}
                series(ax, [float(rr[n][m]) for n in NS], col, mk, LOSS_NAME[loss],
                       err=[float(rr[n][f"{m}_se"]) for n in NS])
            if a == len(ps) - 1:
                ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend(fig, axes, 3)
    save(fig, out, "lasso_by_loss")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "nsweep_p10-20")
    args = ap.parse_args()
    out = args.root / "figures"
    means = read(args.root / "nsweep_means.csv")
    paired = read(args.root / "nsweep_paired_vs_lasso.csv")
    orient = read(args.root / "nsweep_orientation.csv")
    for metric in ("max_f1", "auc", "aupr"):
        fig_metric(means, metric, out)
        fig_paired(paired, metric, out)
    for pen in ("MCP", "SCAD"):
        fig_by_c(paired, pen, out)
    fig_skeleton(orient, out)
    fig_counts(orient, out)
    fig_lasso_by_loss(means, out)


if __name__ == "__main__":
    main()
