#!/usr/bin/env python3
"""E3 of the plan of 13 October 2026 (next_steps/131026/plan_mechanism_131026.md §4.2–4.3), the
beta-min knob: the wave 9 cells ``direct_<estimator>_<C>_w<wmin>_n<n>`` draw the true weights as
``|w| ~ U[w_min, 1]`` with a random sign, ``w_min`` in {0, 0.1, 0.25, 0.5}, on the same supports as
wave 1 (``N(0, 1)`` weights).  From ``campaign_per_dataset.csv`` (``campaign.py``), per
(p, C, n, w_min, estimator):

* ``wmin_table`` -> ``campaign_wmin.csv``: the mean (with its standard error) of ``max_f1``, ``aupr``
  and of the F1, precision and recall of every reduction of the path to one graph (``KNOB_RULES``:
  the BIC and the eBIC(0.5, 1) selections and the search from each), and the paired gain over the
  lasso drawn with the same weights (the same graphs); wave 1's rows appear as ``w_min = "normal"``.
* ``orientation_by_wmin`` -> ``campaign_wmin_orientation.csv``: what the graph of each rule does with
  the true single edges -- right direction only (``correct``), both directions (``hedged``), the
  reverse only (``reversed``), missed -- and with the true 2-cycles (``both`` / ``half`` / ``neither``),
  as mean shares per graph, with the false positives per graph and the share of graphs whose truth is
  identified on its support.  This is the mechanism the knob is meant to move.
* ``edge_outcome_tables`` -> ``campaign_edge_outcomes.csv``: one row per true edge of every graph of
  those cells, from the shards: its weight on the generator's scale, its standardised weight
  ``|M_ij| s_j / s_i`` (the ``scale`` the shard stores), whether it lies in a 2-cycle, and its outcome
  in the graph of every rule.
* ``binned_orientation`` -> ``campaign_wmin_bins.csv``: the outcome shares of the true single edges
  per (p, C, n, w_min, estimator, rule) and |w| bin, pooled over the edges of the bin (``WEIGHT_BINS``,
  fixed on the generator's scale), once for the raw and once for the standardised weight.  Within a
  law only -- laws are never pooled.

The machinery is written for any generator knob (:class:`Knob`): ``twocycles.py`` reuses it for E5,
the share of 2-cycles.  Numbers only; ``plot_betamin.py`` draws them.

    python simulations/diagnostics/campaign.py        # first: campaign_per_dataset.csv
    python simulations/diagnostics/betamin.py         # runs/campaign
    python simulations/diagnostics/betamin.py --root <some campaign root> --c-choices C_ID
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import CELL, cell_meta, ebic_selection, load_shards, true_matrix, unpack_supports  # noqa: E402

#: the four estimators of E3 (plan §4.2): the reference, the lock-in, the two pruners
KNOB_ESTIMATORS = ("lasso", "MCP", "MCP-up", "adaptive")
#: the reductions of a path to one graph, besides the oracle (max_f1): the selection with the BIC
#: penalty and with the eBIC penalty (gamma = 0.5, 1; decoded from the stored path for every cell),
#: and the greedy search from each (stored only by cells run with --select search [--ebic-gamma ...])
KNOB_RULES = ("bic", "search", "ebic05", "ebic05_search", "ebic1", "ebic1_search")
RULE_LABEL = {"bic": "selected, BIC penalty", "search": "searched from the BIC selection",
              "ebic05": "selected, eBIC penalty (γ = 0.5)", "ebic05_search": "searched from the eBIC(0.5) selection",
              "ebic1": "selected, eBIC penalty (γ = 1)", "ebic1_search": "searched from the eBIC(1) selection"}
KNOB_METRICS = ("max_f1", "aupr") + tuple(f"{r}_{m}" for r in KNOB_RULES for m in ("f1", "precision", "recall"))
#: the orientation of the true single edges in the graph of a rule: (name, per-dataset column suffix)
ORIENT_RULES = KNOB_RULES
ORIENT_PARTS = (("correct", "correct"), ("hedged", "hedged"), ("reversed", "reversed"), ("missed", "missed_single"))
#: the fate of a true 2-cycle: both directions kept, one, neither
PAIR_PARTS = (("both", "both"), ("half", "half"), ("neither", "missed_double"))
#: the within-law binning of the true edges by |w| (note §5.2): the bin edges are fixed on the
#: generator's scale, so the same bin is the same weight in every law; the uniform laws stop at 1
WEIGHT_BINS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, math.inf)
BIN_LABELS = ("0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-2", "2+")
OUTCOMES = ("correct", "hedged", "reversed", "missed")
BIN_VARS = ("abs_w", "abs_w_std")


class Knob(NamedTuple):
    """A generator knob as the tables see it: the column that names its level, how a per-data-set
    row and a cell's metadata map to a level, how levels sort and are labelled, and which rows and
    cells belong to the experiment (the reference level included)."""

    name: str                         # the column name of the level ("w_min", "share")
    tag: str                          # the token in file names ("wmin", "share")
    title: str                        # for figure titles
    of: Callable[[dict], str]         # per-data-set row -> level
    of_meta: Callable[[dict], str]    # cell metadata (campaign.cell_meta) -> level
    key: Callable[[str], float]       # sort key of a level
    label: Callable[[str], str]       # axis label of a level
    select: Callable[[dict], bool]    # which per-data-set rows belong (loss, graph class, weights)


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def law(r: dict) -> str:
    """The weight law of a per-dataset row: ``w_min`` as a string for the uniform cells, ``"normal"``
    for the thesis generator."""
    return str(r["w_min"]) if r.get("weights") == "uniform" else "normal"


def law_of_meta(meta: dict) -> str:
    return str(meta["w_min"]) if meta.get("weights") == "uniform" else "normal"


def law_key(w: str) -> float:
    return -1.0 if w == "normal" else float(w)


def law_label(w: str) -> str:
    return "N(0,1)" if w == "normal" else f"U[{float(w):g}, 1]"


def is_random_class(r: dict) -> bool:
    """A direct-loss row of the random graph class (the thesis generator or the weight knob)."""
    return r.get("loss", "direct") == "direct" and r.get("graph", "random") == "random"


WMIN = Knob(name="w_min", tag="wmin", title="E3: the weight distribution as a knob", of=law, of_meta=law_of_meta,
            key=law_key, label=law_label, select=is_random_class)


def knob_columns(knob: Knob) -> tuple:
    return ("p", "c", "n", knob.name, "estimator", "n_graphs", "metric", "mean", "mean_se", "gain", "gain_se", "z")


def orient_columns(knob: Knob) -> tuple:
    return ("p", "c", "n", knob.name, "estimator", "rule", "n_graphs", "single_edges", "correct", "hedged", "reversed",
            "missed", "fp", "edges", "n_graphs_2c", "two_cycles", "both", "half", "neither", "identified")


def edge_columns(knob: Knob) -> tuple:
    return ("cell", "loss", "estimator", "c", "n", "p", "k", "c_choice", "rep", knob.name, "i", "j", "w", "abs_w",
            "abs_w_std", "two_cycle") + KNOB_RULES


def bin_columns(knob: Knob) -> tuple:
    return ("p", "c", "n", knob.name, "estimator", "rule", "var", "weight_bin", "n_edges") + OUTCOMES


KNOB_COLUMNS, ORIENT_COLUMNS, EDGE_COLUMNS, BIN_COLUMNS = (knob_columns(WMIN), orient_columns(WMIN), edge_columns(WMIN),
                                                           bin_columns(WMIN))


def group_by_knob(rows: list[dict], estimators=KNOB_ESTIMATORS, knob: Knob = WMIN) -> dict:
    """``{(p, C, n, level, estimator): {graph: row}}`` over the rows the knob selects, the graph keyed
    by (p, k, c_choice, rep) so that cells pair graph by graph."""
    by = defaultdict(dict)
    for r in rows:
        if r["estimator"] not in estimators or not knob.select(r):
            continue
        by[(r["p"], r["c"], r["n"], knob.of(r), r["estimator"])][(r["p"], r["k"], r["c_choice"], r["rep"])] = r
    return dict(sorted(by.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2], knob.key(kv[0][3]),
                                                   estimators.index(kv[0][4]))))


def group_by_law(rows: list[dict], estimators=KNOB_ESTIMATORS) -> dict:
    return group_by_knob(rows, estimators, WMIN)


def knob_table(rows: list[dict], estimators=KNOB_ESTIMATORS, metrics=KNOB_METRICS, knob: Knob = WMIN) -> list[dict]:
    """Means and paired gains over the lasso at the same level per (p, C, n, level, estimator)."""
    by = group_by_knob(rows, estimators, knob)
    out = []
    for (p, c, n, w, est), cell in by.items():
        ref = by.get((p, c, n, w, "lasso"), {})
        for metric in metrics:
            x = np.array([f(r.get(metric)) for r in cell.values()])
            x = x[np.isfinite(x)]
            d = np.array([f(cell[g].get(metric)) - f(ref[g].get(metric)) for g in cell if g in ref])
            d = d[np.isfinite(d)]
            row = {"p": p, "c": c, "n": n, knob.name: w, "estimator": est, "n_graphs": len(cell), "metric": metric,
                   "mean": float(x.mean()) if len(x) else math.nan,
                   "mean_se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan,
                   "gain": float(d.mean()) if len(d) else math.nan,
                   "gain_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan}
            row["z"] = row["gain"] / row["gain_se"] if row["gain_se"] and row["gain_se"] > 0 else math.nan
            out.append(row)
    return out


def wmin_table(rows: list[dict], estimators=KNOB_ESTIMATORS, metrics=KNOB_METRICS) -> list[dict]:
    """E3 §4.3: means and paired gains over the lasso with the same weights per (p, C, n, w_min) and
    estimator, from the per-dataset rows (``weights``, ``w_min`` columns of ``campaign.py``); the thesis
    generator's rows (``N(0, 1)``) appear with ``w_min = "normal"``."""
    return knob_table(rows, estimators, metrics, WMIN)


def print_knob(out: list[dict], metric="max_f1", estimators=("MCP", "MCP-up", "adaptive"), knob: Knob = WMIN) -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out}, key=lambda t: (int(t[0]), t[1], t[2]))
    for p, c, n in keys:
        cells = {(r[knob.name], r["estimator"]): r for r in out if (r["p"], r["c"], r["n"], r["metric"]) == (p, c, n, metric)}
        ws = sorted({w for w, _ in cells}, key=knob.key)
        print(f"\np = {p}, {c}, n = {n}: {metric} of the lasso and the paired gain over it, by {knob.name}")
        print(f"  {knob.name:<8}{'graphs':>7}{'lasso':>8}" + "".join(f"{e:>22}" for e in estimators))
        for w in ws:
            lasso = cells.get((w, "lasso"))
            vals = "".join(f"{cells[(w, e)]['gain']:+13.3f} ±{cells[(w, e)]['gain_se']:.3f}"
                           if (w, e) in cells and not math.isnan(cells[(w, e)]["gain"]) else f"{'':>22}" for e in estimators)
            print(f"  {w:<8}{(lasso or next(iter(cells.values())))['n_graphs']:>7}"
                  f"{lasso['mean'] if lasso else math.nan:>8.3f}{vals}")


def print_wmin(out: list[dict], metric="max_f1", estimators=("MCP", "MCP-up", "adaptive")) -> None:
    print_knob(out, metric, estimators, WMIN)


def orientation_by_knob(rows: list[dict], estimators=KNOB_ESTIMATORS, rules=ORIENT_RULES, knob: Knob = WMIN) -> list[dict]:
    """Per (p, C, n, level, estimator) and rule: the mean over graphs of the shares of the true single
    edges (true edges outside 2-cycles) that the rule's graph keeps in the right direction only, in both
    directions (a hedge), in the reverse direction only, or not at all; the mean shares of the true
    2-cycles kept with both directions, one, or neither (over the graphs that have a 2-cycle); the
    false positives per graph (``hedged + reversed + fp_single + 2 fp_double``, the stored orientation
    counts of ``campaign.py``), the graph's number of edges, and the share of graphs whose truth is
    identified on its support.  A cell without the rule's graph (no search) contributes nothing."""
    out = []
    for (p, c, n, w, est), cell in group_by_knob(rows, estimators, knob).items():
        for rule in rules:
            acc = defaultdict(list)
            for r in cell.values():
                counts = {name: f(r.get(f"{rule}_{col}")) for name, col in ORIENT_PARTS + PAIR_PARTS}
                if any(math.isnan(v) for v in counts.values()):
                    continue
                two = f(r.get("n_two_cycles"))
                single = f(r.get("n_true_edges")) - 2 * two
                if single > 0:
                    for name, _ in ORIENT_PARTS:
                        acc[name].append(counts[name] / single)
                    acc["single_edges"].append(single)
                if two > 0:
                    for name, _ in PAIR_PARTS:
                        acc[name].append(counts[name] / two)
                acc["two_cycles"].append(two)
                acc["fp"].append(counts["hedged"] + counts["reversed"] + f(r.get(f"{rule}_fp_single"))
                                 + 2 * f(r.get(f"{rule}_fp_double")))
                acc["edges"].append(f(r.get(f"{rule}_edges")))
                acc["identified"].append(float(f(r.get("identified_min_eig")) > 1e-10))
            if not acc["fp"]:
                continue
            row = {"p": p, "c": c, "n": n, knob.name: w, "estimator": est, "rule": rule,
                   "n_graphs": len(acc["single_edges"]), "n_graphs_2c": len(acc["both"])}
            for key in ("single_edges", "correct", "hedged", "reversed", "missed", "fp", "edges", "two_cycles", "both",
                        "half", "neither", "identified"):
                row[key] = float(np.mean(acc[key])) if acc[key] else math.nan
            out.append(row)
    return out


def orientation_by_wmin(rows: list[dict], estimators=KNOB_ESTIMATORS, rules=ORIENT_RULES) -> list[dict]:
    return orientation_by_knob(rows, estimators, rules, WMIN)


def print_orientation(out: list[dict], rule="bic", estimators=KNOB_ESTIMATORS, knob: Knob = WMIN) -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out if r["rule"] == rule}, key=lambda t: (int(t[0]), t[1], t[2]))
    for p, c, n in keys:
        cells = {(r[knob.name], r["estimator"]): r for r in out if (r["p"], r["c"], r["n"], r["rule"]) == (p, c, n, rule)}
        ws = sorted({w for w, _ in cells}, key=knob.key)
        print(f"\np = {p}, {c}, n = {n}: the true single edges in the {rule} graph -- "
              f"right only / hedged / reversed / missed (shares), false positives per graph")
        print(f"  {knob.name:<8}" + "".join(f"{e:>38}" for e in estimators))
        for w in ws:
            cols = []
            for e in estimators:
                r = cells.get((w, e))
                cols.append(f"{r['correct']:.2f} / {r['hedged']:.2f} / {r['reversed']:.2f} / {r['missed']:.2f}  fp {r['fp']:5.1f}"
                            if r else "")
            print(f"  {w:<8}" + "".join(f"{s:>38}" for s in cols))


def print_pairs(out: list[dict], rule="bic", estimators=KNOB_ESTIMATORS, knob: Knob = WMIN) -> None:
    keys = sorted({(r["p"], r["c"], r["n"]) for r in out if r["rule"] == rule}, key=lambda t: (int(t[0]), t[1], t[2]))
    for p, c, n in keys:
        cells = {(r[knob.name], r["estimator"]): r for r in out if (r["p"], r["c"], r["n"], r["rule"]) == (p, c, n, rule)}
        ws = sorted({w for w, _ in cells}, key=knob.key)
        print(f"\np = {p}, {c}, n = {n}: the true 2-cycles in the {rule} graph -- both / half / neither (shares), "
              f"2-cycles per graph, identified share")
        print(f"  {knob.name:<8}" + "".join(f"{e:>40}" for e in estimators))
        for w in ws:
            cols = []
            for e in estimators:
                r = cells.get((w, e))
                cols.append(f"{r['both']:.2f} / {r['half']:.2f} / {r['neither']:.2f}  2c {r['two_cycles']:4.1f}  id {r['identified']:.2f}"
                            if r and not math.isnan(r["both"]) else "")
            print(f"  {w:<8}" + "".join(f"{s:>40}" for s in cols))


def outcome(kept: bool, reverse_kept: bool, two_cycle: bool) -> str:
    """The fate of one true edge j -> i in an estimated graph.  A single edge (its reverse is not a
    true edge): ``correct`` = kept and the reverse not, ``hedged`` = both directions kept, ``reversed``
    = only the reverse kept, ``missed`` = neither.  An edge of a true 2-cycle (the reverse is true too,
    so "reversed" has no meaning): ``present`` or ``missed``."""
    if two_cycle:
        return "present" if kept else "missed"
    if kept:
        return "hedged" if reverse_kept else "correct"
    return "reversed" if reverse_kept else "missed"


def weight_bin(abs_w: float, bins=WEIGHT_BINS, labels=BIN_LABELS) -> str:
    """The label of the half-open bin ``[bins[k], bins[k+1])`` that holds ``abs_w``."""
    return labels[int(np.searchsorted(bins, abs_w, side="right")) - 1]


def rule_supports(d, i: int) -> dict:
    """The graph of every rule for data set ``i`` of shard ``d``, as boolean supports: the BIC
    selection at ``bic_index``, the eBIC selections decoded from the stored path exactly as
    ``campaign.ebic_selection`` does, and the searched graphs the runner stored (``m_search_support``,
    ``m_ebic<gamma>_search_support``); a rule the cell did not run is absent."""
    if "bic_index" not in d.files:
        return {}
    p, n_lambda = int(d["p"][i]), len(d["lambdas"][i])
    path = unpack_supports(d["supports_packed"][i], n_lambda, p)
    out = {"bic": path[int(d["bic_index"][i])]}
    for gamma, tag in ((0.5, "ebic05"), (1.0, "ebic1")):
        out[tag] = path[int(ebic_selection(d, i, gamma)[f"{tag}_index"])]
    for tag, field in (("search", "m_search_support"), ("ebic05_search", "m_ebic05_search_support"),
                       ("ebic1_search", "m_ebic1_search_support")):
        if field in d.files:
            out[tag] = unpack_supports(d[field][i], 1, p)[0]
    return out


def edge_outcomes(folder: Path, meta: dict, knob: Knob = WMIN) -> list[dict]:
    """One row per true edge of every graph of a cell, from its shards (``m_true``, the graph of every
    rule (:func:`rule_supports`), the standardisation ``scale``): the weight on the generator's scale,
    the standardised weight ``|M_ij| s_j / s_i`` the estimators see, the 2-cycle flag and the outcome
    in the graph of every rule of ``KNOB_RULES``, blank when the cell has no such graph."""
    rows = []
    files, _ = load_shards(folder)
    base_meta = {"cell": folder.name, "loss": meta["loss"], "estimator": meta["estimator"], "c": meta["c"], "n": meta["n"],
                 knob.name: knob.of_meta(meta)}
    for _, d in files:
        names = [str(x) for x in d["c_choice_names"]]
        for i in range(len(d["p"])):
            p = int(d["p"][i])
            m_true = true_matrix(d, i)
            truth = m_true != 0
            np.fill_diagonal(truth, False)
            scale = np.asarray(d["scale"][i], dtype=float).reshape(-1) if "scale" in d.files else np.ones(p)
            graphs = rule_supports(d, i)
            base = {**base_meta, "p": p, "k": int(d["k"][i]), "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i])}
            for a, b in zip(*np.nonzero(truth)):                    # the true edge b -> a
                w, tc = float(m_true[a, b]), bool(truth[b, a])
                rows.append({**base, "i": int(a), "j": int(b), "w": w, "abs_w": abs(w),
                             "abs_w_std": abs(w) * float(scale[b] / scale[a]), "two_cycle": int(tc),
                             **{rule: outcome(bool(g[a, b]), bool(g[b, a]), tc) if (g := graphs.get(rule)) is not None else ""
                                for rule in KNOB_RULES}})
    return rows


def edge_outcome_tables(root: Path, estimators=KNOB_ESTIMATORS, knob: Knob = WMIN) -> list[dict]:
    """The edge rows of every direct-loss cell of ``estimators`` that the knob selects under ``root``
    -- the reference cells (the thesis generator) and the knob's cells."""
    rows = []
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        m = CELL.match(folder.name)
        if not m or not (folder / "shards").is_dir():
            continue
        meta = cell_meta(m)
        if meta["estimator"] not in estimators or not knob.select(meta):
            continue
        rows += edge_outcomes(folder, meta, knob)
    return rows


def binned_orientation(edges: list[dict], var: str = "abs_w", rules=ORIENT_RULES, estimators=KNOB_ESTIMATORS,
                       knob: Knob = WMIN) -> list[dict]:
    """The outcome shares of the true single edges per (p, C, n, level, estimator, rule) and bin of
    ``var`` (``abs_w`` or ``abs_w_std``), pooled over the edges of the bin -- a bin holds too few edges
    per graph for per-graph shares.  Edges of 2-cycles and edges without the rule's graph are left out."""
    groups = defaultdict(list)
    for e in edges:
        if int(e["two_cycle"]) or e["estimator"] not in estimators:
            continue
        for rule in rules:
            if e[rule] in OUTCOMES:
                groups[(e["p"], e["c"], e["n"], str(e[knob.name]), e["estimator"], rule, weight_bin(float(e[var])))].append(e[rule])
    out = []
    for key, fates in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2], knob.key(kv[0][3]),
                                                             estimators.index(kv[0][4]), kv[0][5], BIN_LABELS.index(kv[0][6]))):
        p, c, n, w, est, rule, wb = key
        out.append({"p": p, "c": c, "n": n, knob.name: w, "estimator": est, "rule": rule, "var": var, "weight_bin": wb,
                    "n_edges": len(fates), **{o: fates.count(o) / len(fates) for o in OUTCOMES}})
    return out


def print_bins(out: list[dict], rule="bic", var="abs_w", estimators=KNOB_ESTIMATORS, knob: Knob = WMIN) -> None:
    keys = sorted({(r["p"], r["c"], r["n"], r[knob.name]) for r in out if r["rule"] == rule and r["var"] == var},
                  key=lambda t: (int(t[0]), t[1], t[2], knob.key(t[3])))
    for p, c, n, w in keys:
        cells = {(r["weight_bin"], r["estimator"]): r for r in out
                 if (r["p"], r["c"], r["n"], r[knob.name], r["rule"], r["var"]) == (p, c, n, w, rule, var)}
        print(f"\np = {p}, {c}, n = {n}, {knob.name} = {w}: the true single edges in the {rule} graph by "
              f"{'|w|' if var == 'abs_w' else 'standardised |w|'} -- right only / hedged / reversed / missed (edges)")
        print(f"  {'bin':<10}" + "".join(f"{e:>36}" for e in estimators))
        for wb in BIN_LABELS:
            if not any((wb, e) in cells for e in estimators):
                continue
            cols = [f"{r['correct']:.2f} / {r['hedged']:.2f} / {r['reversed']:.2f} / {r['missed']:.2f} ({r['n_edges']})"
                    if (r := cells.get((wb, e))) else "" for e in estimators]
            print(f"  {wb:<10}" + "".join(f"{x:>36}" for x in cols))


def write(path: Path, rows: list[dict], columns) -> None:
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns)
        wr.writeheader()
        wr.writerows(rows)


def run_knob(root: Path, knob: Knob, estimators=KNOB_ESTIMATORS, c_choices=None, bins: bool = True,
             edges_name: str | None = None, reference: str = "normal") -> bool:
    """The whole pipeline for one knob: the tables of this module written next to the campaign's
    CSVs as ``campaign_<tag>*.csv`` and printed.  Returns False when no cell of the knob exists."""
    per_dataset = root / "campaign_per_dataset.csv"
    if not per_dataset.exists():
        raise SystemExit(f"{per_dataset} is missing (run simulations/diagnostics/campaign.py first)")
    rows = list(csv.DictReader(per_dataset.open()))
    if c_choices:
        rows = [r for r in rows if r["c_choice"] in set(c_choices)]
    table = knob_table(rows, estimators, knob=knob)
    if not any(r[knob.name] != reference for r in table):
        print(f"no cell of the knob {knob.name} under {root}")
        return False
    write(root / f"campaign_{knob.tag}.csv", table, knob_columns(knob))
    for metric in ("max_f1", "bic_f1", "search_f1"):
        print_knob(table, metric, [e for e in estimators if e != "lasso"], knob)
    orient = orientation_by_knob(rows, estimators, knob=knob)
    write(root / f"campaign_{knob.tag}_orientation.csv", orient, orient_columns(knob))
    print_orientation(orient, "bic", estimators, knob)
    print_pairs(orient, "bic", estimators, knob)
    edges = edge_outcome_tables(root, estimators, knob)
    if c_choices:
        edges = [e for e in edges if e["c_choice"] in set(c_choices)]
    edges_name = edges_name or f"campaign_{knob.tag}_edges.csv"
    write(root / edges_name, edges, edge_columns(knob))
    written = [f"campaign_{knob.tag}.csv ({len(table)} rows)", f"campaign_{knob.tag}_orientation.csv ({len(orient)} rows)",
               f"{edges_name} ({len(edges)} true edges)"]
    if bins:
        table_bins = [r for var in BIN_VARS for r in binned_orientation(edges, var, estimators=estimators, knob=knob)]
        write(root / f"campaign_{knob.tag}_bins.csv", table_bins, bin_columns(knob))
        print_bins(table_bins, "bic", "abs_w", estimators, knob)
        written.append(f"campaign_{knob.tag}_bins.csv ({len(table_bins)} rows)")
    print(f"\nwrote under {root}: " + ", ".join(written))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--c-choices", nargs="+", default=None, metavar="C",
                    help="keep the graphs of these true C only (e.g. C_ID when the knob cells were run with "
                         "--c-choices C_ID, so that the wave 1 reference is the same graphs); default all")
    args = ap.parse_args()
    run_knob(args.root, WMIN, KNOB_ESTIMATORS, args.c_choices, bins=True, edges_name="campaign_edge_outcomes.csv")


if __name__ == "__main__":
    main()
