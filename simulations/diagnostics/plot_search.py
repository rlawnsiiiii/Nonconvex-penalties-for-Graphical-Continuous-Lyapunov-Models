#!/usr/bin/env python3
"""Figures for S3b (simulations/S3b_reversal_search.md).  **Run locally.**

Reads the row files of search_study.py and draws, for each of them:

  <label>_f1_<loss>_<score>.{png,pdf}      mean directed F1 against n, per method
  <label>_exact_<loss>_<score>.{png,pdf}   share of datasets recovered exactly
  <label>_graphs_<...>.{png,pdf}           one dataset drawn as networks: the truth,
                                            plain lasso (no search), and the four methods

    python simulations/diagnostics/plot_search.py [--in runs/s3b_search]

Methods and their encoding (hue = where the method comes from; filled = with search):
  plain lasso / MCP / SCAD at the BIC-selected lambda     hollow marker, thin line
  lasso / MCP / SCAD + search                             filled marker, thick line
  pure greedy search (random restarts)                    violet diamond, thick line
  lasso at the oracle best-F1 lambda (Dettling's max_f1)  grey x, thin line
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bidirectional import FALSE, MISSED, WRONG, draw_graph, save  # noqa: E402
from plot_figures import INK, INK_2, style_axes  # noqa: E402

SURFACE = "#fcfcfb"
PEN = {"lasso": ("#2a78d6", "o"), "MCP": ("#eb6834", "s"), "SCAD": ("#1baf7a", "^")}
PURE = ("#4a3aa7", "D")
ORACLE = ("#8a8984", "x")
SERIES = ([(f"path_{p}_bic", f"{p}, no search (BIC λ)", PEN[p][0], PEN[p][1], False) for p in PEN]
          + [(f"search_{p}", f"{p} + search", PEN[p][0], PEN[p][1], True) for p in PEN]
          + [("search_pure", "pure greedy search", PURE[0], PURE[1], True),
             ("path_lasso_oracle", "lasso, oracle best-F1 λ (Dettling's max_f1)", ORACLE[0], ORACLE[1], False),
             ("search_truth", "search started from the true graph (oracle)", INK, "*", True)])


def nkey(n: str) -> float:
    return math.inf if n in ("inf", "Infinity") else float(n)


def nlabel(n: str) -> str:
    x = nkey(n)
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "∞" if math.isinf(x) else ("10" + str(int(round(math.log10(x)))).translate(sup)) if x >= 1000 else str(int(x))


def group_of(r) -> str:
    return r["setting"] if r.get("setting") else f"p = {r['p']}"


GROUP_NAME = {"path": "Example 2: path", "cycle_fixed": "Example 2: 5-cycle, m₁₅ = 0.65",
              "cycle_random": "Example 2: 5-cycle, random m₁₅"}
LOSS_NAME = {"direct": "direct loss", "loglik": "log-likelihood loss", "frobenius": "Frobenius loss"}


def family_of(label: str) -> str:
    """Row files of one setting that were run per sample size belong together:
    ``random_direct_p10_raw_n1000`` -> ``random_direct_p10_raw``."""
    return re.sub(r"_n(1000|10000|inf)$", "", label)


def scale_of(family: str) -> str:
    if family == "example2":
        return "raw scale"
    return "raw scale" if family.endswith("_raw") else "Dettling's pipeline"


def complete_cells(rows) -> set[tuple[str, str]]:
    """(group, n) cells that were run in full: a finite-n cell with fewer than 90 % of
    the datasets of the group's largest cell was stopped early and is left out of the
    by-n figures (n = inf legitimately has one dataset for the fixed Example 2 graphs)."""
    count = defaultdict(int)
    for r in rows:
        if r["method"] == "path_lasso_bic":
            count[(group_of(r), r["n"])] += 1
    top = defaultdict(int)
    for (g, n), c in count.items():
        top[g] = max(top[g], c)
    return {(g, n) for (g, n), c in count.items() if math.isinf(nkey(n)) or c >= 0.9 * top[g]}


def fig_by_n(rows, label, loss, score, metric, ylabel, out):
    rows = [r for r in rows if r["loss"] == loss and r["score"] == score]
    if not rows:
        return
    keep = complete_cells(rows)
    rows = [r for r in rows if (group_of(r), r["n"]) in keep]
    groups = sorted({group_of(r) for r in rows})
    ns = sorted({r["n"] for r in rows}, key=nkey)
    if len(ns) < 2:
        return                                  # nothing to draw against n
    agg = defaultdict(list)
    for r in rows:
        agg[(group_of(r), r["n"], r["method"])].append(float(r[metric]))
    fig, axes = plt.subplots(1, len(groups), figsize=(max(4.4 * len(groups), 7.6), 3.6), squeeze=False,
                             sharey=True)
    for ax, g in zip(axes[0], groups):
        style_axes(ax)
        x = np.arange(len(ns))
        for k, (method, name, col, mk, filled) in enumerate(SERIES):
            y = [np.mean(agg[(g, n, method)]) if agg.get((g, n, method)) else np.nan for n in ns]
            if np.all(np.isnan(y)):
                continue
            # small horizontal dodge, so that series with identical values stay visible
            xs = x + (k - (len(SERIES) - 1) / 2) * 0.028 * (len(ns) - 1) / 3
            reference = method in ("search_truth", "path_lasso_oracle")
            ax.plot(xs, y, color=col, lw=1.1 if reference or not filled else 2.0,
                    zorder=3 if filled else 2, alpha=1.0 if filled else 0.9)
            ax.plot(xs, y, ls="none", marker=mk, ms=11 if mk == "*" else 7 if filled else 6.5,
                    color=col if filled else SURFACE, mec=col if not filled else SURFACE,
                    mew=1.4, zorder=4, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([nlabel(n) for n in ns])
        ax.set_xlabel("sample size n", fontsize=9, color=INK_2)
        ax.set_title(f"{GROUP_NAME.get(g, g)} · {scale_of(family_of(label))}\n"
                     f"{LOSS_NAME.get(loss, loss)}, {score.upper()}", loc="left", fontsize=9.5, color=INK)
        ax.set_ylim(0, 1.03)
    axes[0][0].set_ylabel(ylabel, fontsize=9, color=INK_2)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, out, f"{label}_{metric}_{loss}_{score}")


def edges_of(s: str) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in e.split(">")) for e in s.split(";") if e} if s else set()


def fig_graphs(rows, label, select: dict, out, title):
    """Truth + plain lasso + the four methods for one dataset (``select``: column -> value)."""
    sub = [r for r in rows if all(r.get(k) == v for k, v in select.items())]
    if not sub:
        return
    by = {r["method"]: r for r in sub}
    order = [("path_lasso_bic", "plain lasso (BIC λ)"), ("search_lasso", "lasso + search"),
             ("search_MCP", "MCP + search"), ("search_SCAD", "SCAD + search"),
             ("search_pure", "pure greedy search")]
    order = [(m, t) for m, t in order if m in by]
    truth = edges_of(sub[0]["true_edge_list"])
    p = int(sub[0].get("p") or 5)
    ang = np.pi / 2 - 2 * np.pi * np.arange(p) / p
    pos = list(zip(np.cos(ang), np.sin(ang)))
    fig, axes = plt.subplots(1, 1 + len(order), figsize=(3.3 * (1 + len(order)), 4.4))
    n_cyc = sum((b, a) in truth for a, b in truth) // 2
    draw_graph(axes[0], pos, truth, {e: INK for e in truth}, {e: "-" for e in truth},
               "true graph", f"{len(truth)} edges, {n_cyc} 2-cycles")
    for ax, (method, name) in zip(axes[1:], order):
        est = edges_of(by[method]["edge_list"])
        colors, styles = {}, {}
        for e in est:
            colors[e] = INK if e in truth else (WRONG if e[::-1] in truth else FALSE)
            styles[e] = "-"
        for e in truth - est:
            colors[e], styles[e] = MISSED, (0, (1.5, 2))
        n_ok = sum(e in truth for e in est)
        n_wrong = sum(e not in truth and e[::-1] in truth for e in est)
        draw_graph(ax, pos, est | (truth - est), colors, styles, name,
                   f"correct {n_ok} · wrong dir. {n_wrong} · false {len(est) - n_ok - n_wrong}\n"
                   f"F1 {float(by[method]['f1']):.2f}")
    legend = [Line2D([], [], color=INK, lw=1.4, label="true edge, selected"),
              Line2D([], [], color=WRONG, lw=1.4, label="selected, true edge points the other way"),
              Line2D([], [], color=FALSE, lw=1.4, label="selected, no true edge in this pair"),
              Line2D([], [], color=MISSED, lw=1.4, ls=(0, (1.5, 2)), label="true edge, not selected")]
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=10, color=INK)
    fig.legend(handles=legend, loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.005, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out, f"{label}_graphs_" + "_".join(f"{v}" for v in select.values()).replace(".", "p"))


OVERVIEW = [("path_lasso_oracle", "lasso, oracle λ (Dettling)", ORACLE[0], ORACLE[1]),
            ("search_lasso", "lasso + search", PEN["lasso"][0], PEN["lasso"][1]),
            ("search_MCP", "MCP + search", PEN["MCP"][0], PEN["MCP"][1]),
            ("search_SCAD", "SCAD + search", PEN["SCAD"][0], PEN["SCAD"][1]),
            ("search_pure", "pure greedy search", PURE[0], PURE[1])]


def fig_overview(inp: Path, out: Path):
    """Paired difference in directed F1 to plain lasso at its BIC lambda, per setting
    (rows) and method (dots), from every *_summary.csv (direct loss, BIC)."""
    rows = []
    for f in sorted(inp.glob("*_summary.csv")):
        label = f.stem.replace("_summary", "").replace("random_direct_", "")
        for r in csv.DictReader(f.open()):
            if r["loss"] != "direct" or r["score"] != "bic":
                continue
            if label == "example2" and r["group"] == "path":
                continue
            scale = "raw" if "raw" in label else ("Example 2" if label == "example2" else "Dettling")
            name = (GROUP_NAME.get(r["group"], r["group"]) if label == "example2"
                    else f"{r['group']}, {scale}") + f", n = {nlabel(r['n'])}"
            rows.append((name, r["method"], float(r["f1_minus_path_lasso_bic"]), int(r["datasets"]),
                         (family_of(f.stem.replace("_summary", "")), r["group"])))
    names = list(dict.fromkeys(n for n, *_ in rows))
    if not names:
        return
    top = defaultdict(int)                       # largest cell per (setting, group)
    for _, _, _, c, fam in rows:
        top[fam] = max(top[fam], c)

    def count_label(name):
        c, fam = next((c, fam) for nn, _, _, c, fam in rows if nn == name)
        early = "∞" not in name and c < 0.9 * top[fam]
        return f"{name}  [{c}{', stopped early' if early else ''}]"
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(names) + 1.4))
    style_axes(ax)
    ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
    dodge = np.linspace(-0.24, 0.24, len(OVERVIEW))
    for k, (method, lab, col, mk) in enumerate(OVERVIEW):
        pts = [(names.index(n), d) for n, m, d, _, _ in rows if m == method]
        if pts:
            ax.plot([d for _, d in pts], [y + dodge[k] for y, _ in pts], ls="none", marker=mk,
                    ms=7, color=col, mec=SURFACE if mk != "x" else col, mew=1.2, label=lab, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([count_label(n) for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("directed F1 minus plain lasso at its BIC λ (paired mean; [graphs])", fontsize=9, color=INK_2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "overview_f1_vs_plain_lasso")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "runs" / "s3b_search")
    ap.add_argument("--graphs", nargs="*", default=[],
                    help="extra datasets to draw, as label:col=val,col=val (e.g. random_direct:p=10,k=2,...)")
    args = ap.parse_args()
    out = args.inp / "figures"
    families = defaultdict(list)
    for f in sorted(args.inp.glob("*_rows.csv")):
        families[family_of(f.stem.replace("_rows", ""))] += list(csv.DictReader(f.open()))
    for label, rows in families.items():
        for loss in sorted({r["loss"] for r in rows}):
            for score in sorted({r["score"] for r in rows}):
                fig_by_n(rows, label, loss, score, "f1", "directed F1 (mean)", out)
                fig_by_n(rows, label, loss, score, "exact", "share recovered exactly", out)
    fig_overview(args.inp, out)
    for spec in args.graphs:
        label, _, sel = spec.partition(":")
        select = dict(kv.split("=") for kv in sel.split(","))
        rows = list(csv.DictReader((args.inp / f"{label}_rows.csv").open()))
        what = ", ".join(f"n = {nlabel(v)}" if k == "n" else GROUP_NAME.get(v, v) if k == "setting"
                         else f"{k} = {v}" if k in ("k", "rep") else v
                         for k, v in select.items() if k not in ("score", "loss"))
        title = (f"{what} · {scale_of(family_of(label))} · "
                 f"{LOSS_NAME.get(select.get('loss', 'direct'))}, {select.get('score', 'bic').upper()}")
        fig_graphs(rows, label, select, out, title)


if __name__ == "__main__":
    main()
