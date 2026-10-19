#!/usr/bin/env python3
"""Dettling's Figure 5 for every estimator of the campaign (15 October 2026).  **Run locally.**

Dettling, Drton & Kolar (2024) never reduce a path to one graph in their simulation: Figure 5 reports four
summaries of the whole path of 100 estimates against the truth -- the largest accuracy and the largest F1
over the path (oracle numbers, the best graph the path contains), and the areas under the ROC and the
precision-recall curves traced by the path -- over p at n = 1000, one curve per setting of the true C.
The campaign stores exactly these four numbers per graph and estimator (``campaign.py``,
``campaign_means_by_true_c.csv``: means and standard errors over the 100 graphs per (p, true C); Dettling
averages 400), so his figure can be drawn for every estimator, under his pipeline (C = 2I, ``C2I``) and
under the rescaled C (``Cresc``), with p = 10, 20 from wave 1 and p = 15, 25, 30, 40, 50 from wave 4.

``figure5/figure5_<estimator>_<C>.png``   his 2 x 2 layout for one estimator: the four metrics over p, one
                                          curve per true C.  For the lasso under his pipeline his own
                                          digitised values (``comparison_vs_dettling.csv`` of the S1
                                          reproduction) are overlaid as dashed lines.
``figure5/figure5_grid_<C>.png``          the comparison: rows = the four metrics, columns = the four
                                          settings of the true C, one curve per estimator.
``figure5/figure5_estimators_<C>.png``    his 2 x 2 layout for one C with one curve per estimator, the four
                                          settings of the true C pooled (400 graphs per point).
``figure5/figure5_gain_<C>.png``          the paired gain of every estimator over the lasso on the same
                                          graphs (bars: ± 2 standard errors): rows = the four metrics,
                                          columns = pooled and the four settings of the true C.

The last two read ``campaign_figure5.csv`` of ``figure5_table.py`` (run it first); the first two the means
table of ``campaign.py``.

    python simulations/diagnostics/figure5_table.py
    python simulations/diagnostics/plot_figure5.py            # runs/campaign -> runs/campaign/figures/figure5/
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
from figure5_table import SETTINGS  # noqa: E402
from plot_bidirectional import save  # noqa: E402
from plot_campaign import C_NAME, GREY, INK, INK_2, STYLE, SURFACE, f, legend, read, series, style_axes  # noqa: E402
from plot_figures import C_STYLE, METRICS  # noqa: E402

ESTS = ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")
PS = (10, 15, 20, 25, 30, 40, 50)
REFERENCE = ROOT / "runs" / "s1_dettling_reproduction" / "comparison_vs_dettling.csv"
SETTING_LABEL = {"all": "the four true C pooled", **{k: v[2] for k, v in C_STYLE.items()}}


def load(root: Path, n: str = "1000") -> list[dict]:
    """The direct-loss rows of the thesis generator at one n, per true C."""
    return [r for r in read(root / "campaign_means_by_true_c.csv")
            if r["loss"] == "direct" and r["n"] == n and not r.get("dgp") and r["estimator"] in ESTS
            and int(r["p"]) in PS]


def load_reference(path: Path) -> dict:
    """Dettling's digitised Figure 5 values: (metric, true C, p) -> (value, standard error)."""
    if not path.exists():
        return {}
    return {(r["metric"], r["c_choice"], int(r["p"])): (float(r["dettling"]), float(r["dettling_se"]))
            for r in csv.DictReader(path.open())}


def load_table(root: Path, c: str | None = None, n: str = "1000") -> list[dict]:
    """The long table of ``figure5_table.py`` (``campaign_figure5.csv``) at one n, optionally one C; empty
    when the table has not been written."""
    path = root / "campaign_figure5.csv"
    if not path.exists():
        return []
    return [r for r in read(path) if r["n"] == n and (c is None or r["c"] == c) and r["estimator"] in ESTS]


def _points(rows, est, c, true_c, metric):
    rr = sorted((r for r in rows if (r["estimator"], r["c"], r["true_c"]) == (est, c, true_c)), key=lambda r: int(r["p"]))
    xs = [int(r["p"]) for r in rr]
    return xs, [f(r[metric]) for r in rr], [f(r[f"{metric}_se"]) for r in rr]


def _curve(tab, est, c, true_c, metric, value="mean", err="se"):
    rr = sorted((r for r in tab if (r["estimator"], r["c"], r["true_c"], r["metric"]) == (est, c, true_c, metric)),
                key=lambda r: int(r["p"]))
    return [int(r["p"]) for r in rr], [f(r[value]) for r in rr], [f(r[err]) for r in rr]


def fig_estimator(rows, est, c, out, reference=None):
    """Dettling's 2 x 2 layout for one estimator and one C: the four metrics over p, one curve per true C."""
    sel = [r for r in rows if r["estimator"] == est and r["c"] == c]
    if not sel:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), facecolor="white")
    ps_all = sorted({int(r["p"]) for r in sel})
    overlaid = False
    for ax, (metric, title) in zip(axes.ravel(), METRICS):
        style_axes(ax)
        for true_c, (colour, marker, label) in C_STYLE.items():
            xs, mu, se = _points(sel, est, c, true_c, metric)
            if not xs:
                continue
            ax.errorbar(xs, mu, yerr=se, color=colour, marker=marker, markersize=5, linewidth=1.8, capsize=3,
                        capthick=1, elinewidth=1, label=label, zorder=3)
            if reference:
                ref = [(p, *reference[(metric, true_c, p)]) for p in xs if (metric, true_c, p) in reference]
                if ref:
                    ax.plot([p for p, _, _ in ref], [v for _, v, _ in ref], color=colour, lw=1.0, ls="--", alpha=0.7,
                            zorder=2)
                    overlaid = True
        ax.set_xticks(ps_all)
        ax.set_xlabel("number of nodes p", color=INK_2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if overlaid:
        handles.append(Line2D([], [], color=GREY, lw=1.0, ls="--"))
        labels.append("Dettling et al. (2024), Figure 5")
    fig.legend(handles, labels, loc="lower center", ncol=5 if overlaid else 4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    fig.suptitle(f"Figure 5 layout: {STYLE[est][3]}, {C_NAME[c]}, n = 10³ (100 graphs per point)",
                 color=INK, fontsize=11.5, x=0.04, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    save(fig, out, f"figure5_{est}_{c}")


def fig_grid(rows, c, out, ests=ESTS):
    """Rows = the four metrics, columns = the four settings of the true C, one curve per estimator."""
    sel = [r for r in rows if r["c"] == c]
    if not sel:
        return
    settings = list(C_STYLE)
    fig, axes = plt.subplots(len(METRICS), len(settings), figsize=(3.3 * len(settings) + 0.5, 2.7 * len(METRICS)),
                             sharex=True, sharey="row", squeeze=False)
    ps_all = sorted({int(r["p"]) for r in sel})
    for a, (metric, title) in enumerate(METRICS):
        for b, true_c in enumerate(settings):
            ax = axes[a][b]
            style_axes(ax)
            for est in ests:
                xs, mu, se = _points(sel, est, c, true_c, metric)
                if len(xs) >= 2:
                    series(ax, xs, mu, est, err=se, lw=1.5)
            ax.set_xticks(ps_all)
            if a == 0:
                ax.set_title(C_STYLE[true_c][2], color=INK, fontsize=9.5, loc="left")
            if a == len(METRICS) - 1:
                ax.set_xlabel("number of nodes p", color=INK_2, fontsize=9)
        axes[a][0].set_ylabel(title, color=INK_2, fontsize=9)
    fig.suptitle(f"Dettling's four path summaries for every estimator, {C_NAME[c]}, n = 10³ "
                 f"(100 graphs per point)", color=INK, fontsize=11, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    legend(fig, ests, y=-0.005, ncol=6)
    save(fig, out, f"figure5_grid_{c}")


def fig_estimators(tab, c, out, ests=ESTS, true_c="all"):
    """Dettling's 2 x 2 layout for one C with one curve per estimator: the four metrics over p, the four
    settings of the true C pooled (400 graphs per point) unless ``true_c`` names one of them."""
    sel = [r for r in tab if r["c"] == c and r["true_c"] == true_c]
    if not sel:
        return
    shown = [e for e in ests if any(r["estimator"] == e for r in sel)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), facecolor="white")
    ps_all = sorted({int(r["p"]) for r in sel})
    for ax, (metric, title) in zip(axes.ravel(), METRICS):
        style_axes(ax)
        for est in shown:
            xs, mu, se = _curve(sel, est, c, true_c, metric)
            if len(xs) >= 2:
                series(ax, xs, mu, est, err=se, lw=1.6)
        ax.set_xticks(ps_all)
        ax.set_xlabel("number of nodes p", color=INK_2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=6)
    graphs = max(int(float(r["graphs"])) for r in sel)
    fig.suptitle(f"Dettling's four path summaries per estimator, {C_NAME[c]}, n = 10³, {SETTING_LABEL[true_c]} "
                 f"({graphs} graphs per point; bars: standard errors)", color=INK, fontsize=11, x=0.04, ha="left",
                 y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    legend(fig, shown, y=-0.005, ncol=6)
    save(fig, out, f"figure5_estimators_{c}" + ("" if true_c == "all" else f"_{true_c}"))


def fig_gain(tab, c, out, ests=ESTS):
    """Rows = the four metrics, columns = the pooled setting and the four settings of the true C; one curve
    per estimator: its paired gain over the lasso on the same graphs, bars = ± 2 standard errors."""
    sel = [r for r in tab if r["c"] == c]
    if not sel:
        return
    settings = [s for s in SETTINGS if any(r["true_c"] == s for r in sel)]
    shown = [e for e in ests if e != "lasso" and any(r["estimator"] == e for r in sel)]
    fig, axes = plt.subplots(len(METRICS), len(settings), figsize=(3.1 * len(settings) + 0.5, 2.6 * len(METRICS)),
                             sharex=True, sharey="row", squeeze=False)
    ps_all = sorted({int(r["p"]) for r in sel})
    for a, (metric, title) in enumerate(METRICS):
        for b, true_c in enumerate(settings):
            ax = axes[a][b]
            style_axes(ax)
            ax.axhline(0.0, color=GREY, lw=1.0, zorder=1)
            for est in shown:
                xs, g, se = _curve(sel, est, c, true_c, metric, "gain", "gain_se")
                if len(xs) >= 2:
                    series(ax, xs, g, est, err=2 * np.asarray(se, float), lw=1.5)
            ax.set_xticks(ps_all)
            if a == 0:
                ax.set_title(SETTING_LABEL[true_c], color=INK, fontsize=9.5, loc="left")
            if a == len(METRICS) - 1:
                ax.set_xlabel("number of nodes p", color=INK_2, fontsize=9)
        axes[a][0].set_ylabel(f"gain in {title}", color=INK_2, fontsize=9)
    fig.suptitle(f"Paired gain over the lasso on the same graphs, {C_NAME[c]}, n = 10³ (bars: ± 2 standard errors)",
                 color=INK, fontsize=11, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    legend(fig, shown, y=-0.005, ncol=5)
    save(fig, out, f"figure5_gain_{c}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--reference", type=Path, default=REFERENCE,
                    help="comparison_vs_dettling.csv of the S1 reproduction (his digitised values; overlaid on "
                         "the lasso under his pipeline); 'none' to skip")
    args = ap.parse_args()
    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no direct-loss rows at n = 1000 in {args.root / 'campaign_means_by_true_c.csv'}")
    reference = {} if str(args.reference) == "none" else load_reference(args.reference)
    out = args.root / "figures" / "figure5"
    for c in ("C2I", "Cresc"):
        for est in ESTS:
            fig_estimator(rows, est, c, out, reference if (est == "lasso" and c == "C2I") else None)
        fig_grid(rows, c, out)
    tab = load_table(args.root)
    if not tab:
        print(f"{args.root / 'campaign_figure5.csv'} is missing (run simulations/diagnostics/figure5_table.py for the "
              "per-estimator and gain figures)")
    for c in ("C2I", "Cresc"):
        fig_estimators(tab, c, out)
        fig_gain(tab, c, out)


if __name__ == "__main__":
    main()
