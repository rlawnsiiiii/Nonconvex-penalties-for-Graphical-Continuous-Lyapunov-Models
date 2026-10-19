#!/usr/bin/env python3
"""E4 of the plan of 13 October 2026 (next_steps/131026/plan_mechanism_131026.md §5.1, §5.3): the
gain of the dense-start estimators over the lasso by the class of the true graph, **within k**.

Reads ``campaign_per_dataset.csv`` (which carries ``graph_class``, ``n_two_cycles``,
``frac_edges_2c`` and ``identified_min_eig`` per graph since 13 October; ``campaign.py``) and, for
the wave 1 cells ``direct_<estimator>_<C>_n<n>`` at the given p, groups the graphs by
(p, C, n, k, graph_class).  At p = 10 the class is almost determined by k (k = 1: mostly no
2-cycles; k = 4: almost every graph has one), so a comparison pooled over k would only show the
density effect of verdict 12 of VERDICTS.md; every number here is within one k.

Writes ``<root>/campaign_graph_class.csv`` (long format: p, c, n, generator, k, graph_class, n_graphs,
identified_share, estimator, metric, mean, gain, gain_se, z) and prints the paired gains in
max_f1, bic_f1 and search_f1 over the lasso with the same C.  Since 14 October the wave 9 cells with
a generated class (plan §5.2: ``generator`` = ``dag``, column ``graph`` of the per-dataset
table; ``random`` is the thesis generator) are in the same table, each compared with the lasso drawn
with the same generator, and ``table_by_generator`` pools the classes per generator (within k) for
``gain_by_generator.png``.  ``two_cycle_table`` (17 October) reads the stored orientation counts of
the selected and the searched graph and writes ``campaign_two_cycles.csv``: how each estimator keeps
the true 2-cycles (both directions / one / neither) and the true single edges (right direction alone /
both directions, the hedge / reversed / missed), means per graph -- the counts behind the reading of
the classes in ``docs/GRAPH_CLASSES_AND_BETAMIN.md``.  Numbers only; the figures are drawn by
``plot_graph_class.py``.

    python simulations/diagnostics/graph_class.py                 # runs/campaign, p = 10 and 20
    python simulations/diagnostics/graph_class.py --p 5 --root some/dir
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gclm.graphs import CLASSES  # noqa: E402

CELL = re.compile(r"^direct_(?P<estimator>[A-Za-z-]+)_(?P<c>C2I|Cresc)(?:_(?P<graph>dag))?_n(?P<n>1000|1e4|inf)$")
ESTIMATORS = ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "MCP-lla", "SCAD-lla", "adaptive")
GENERATORS = ("random", "dag")
METRICS = ("max_f1", "aupr", "bic_f1", "search_f1")
COLUMNS = ("p", "c", "n", "generator", "k", "graph_class", "n_graphs", "identified_share", "estimator", "metric",
           "mean", "gain", "gain_se", "z")
#: the orientation counts stored per selected (``bic_*``) and searched (``search_*``) graph
#: (gclm.metrics.orientation_breakdown), per unordered pair of nodes
ORIENT = ("correct", "reversed", "hedged", "both", "half", "missed_single", "missed_double", "fp_single", "fp_double")
TWO_CYCLE_COLUMNS = ("p", "c", "n", "generator", "estimator", "rule", "n_graphs", "true_edges", "two_cycles", "both", "half",
                     "neither", "single_edges", "correct", "hedged", "reversed", "missed")


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path, ps) -> list[dict]:
    """The wave 1 rows and the wave 9 class cells at the sizes ``ps``, with the class columns; each row
    gets ``generator`` (``random`` for the thesis generator, else ``dag``)."""
    rows = []
    for r in csv.DictReader(path.open()):
        m = CELL.match(r["cell"])
        if not m or m["estimator"] not in ESTIMATORS or int(r["p"]) not in ps or r["loss"] != "direct":
            continue
        if not r.get("graph_class"):
            raise SystemExit(f"{path} has no graph_class column: rerun simulations/diagnostics/campaign.py")
        r["generator"] = m["graph"] or "random"
        rows.append(r)
    return rows


def table(rows: list[dict], estimators=ESTIMATORS, metrics=METRICS, pooled: bool = False) -> list[dict]:
    """Means and paired gains over the lasso (same p, C, n, generator; the same graphs) per
    (p, C, n, generator, k, graph_class) and estimator.  ``pooled`` adds rows with ``graph_class = "all"``
    (the classes of a generator pooled, within k), which :func:`table_by_generator` uses."""
    by = defaultdict(dict)                       # (p, c, n, generator, estimator) -> graph id -> row
    for r in rows:
        gen = r.get("generator") or "random"
        by[(r["p"], r["c"], r["n"], gen, r["estimator"])][(r["p"], r["k"], r["c_choice"], r["rep"])] = r
    out = []
    for (p, c, n, gen, est), cell in sorted(by.items(), key=lambda kv: (kv[0][:3], GENERATORS.index(kv[0][3]), kv[0][4])):
        if est not in estimators:
            continue
        ref = by.get((p, c, n, gen, "lasso"), {})       # the lasso drawn with the same generator
        groups = defaultdict(list)
        for g, r in cell.items():
            groups[(r["k"], r["graph_class"])].append(g)
            if pooled:
                groups[(r["k"], "all")].append(g)
        for (k, cls), ids in sorted(groups.items()):
            ident = np.mean([f(cell[g]["identified_min_eig"]) > 1e-10 for g in ids])
            for metric in metrics:
                x = np.array([f(cell[g].get(metric)) for g in ids])        # a cell without the search: NaN
                paired = [g for g in ids if g in ref]
                d = np.array([f(cell[g].get(metric)) - f(ref[g].get(metric)) for g in paired])
                d = d[np.isfinite(d)]
                row = {"p": p, "c": c, "n": n, "generator": gen, "k": k, "graph_class": cls, "n_graphs": len(ids),
                       "identified_share": float(ident), "estimator": est, "metric": metric,
                       "mean": float(np.nanmean(x)) if np.isfinite(x).any() else math.nan,
                       "gain": float(d.mean()) if len(d) else math.nan,
                       "gain_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan}
                row["z"] = row["gain"] / row["gain_se"] if row["gain_se"] and row["gain_se"] > 0 else math.nan
                out.append(row)
    return out


def table_by_generator(rows: list[dict], estimators=ESTIMATORS, metrics=METRICS) -> list[dict]:
    """E4 §5.3: the gain per (p, C, n, generator, k) with the classes pooled -- the generated classes
    ``dag`` next to the thesis generator, within k."""
    return [r for r in table(rows, estimators, metrics, pooled=True) if r["graph_class"] == "all"]


def two_cycle_table(rows: list[dict], estimators=ESTIMATORS, rules=("bic", "search")) -> list[dict]:
    """How each estimator treats the true 2-cycles and the true single edges of a graph, per
    (p, C, n, generator, estimator) and rule (``bic``: the graph selected by the score; ``search``:
    the graph after the search), as means per graph of the stored orientation counts.  A true
    2-cycle is kept with ``both`` directions, with one (``half``) or with ``neither``; a true single
    edge is ``correct`` (its direction alone), ``hedged`` (both directions kept), ``reversed`` (the
    reverse alone) or ``missed``.  Per graph, both + half + neither = the number of true 2-cycles and
    correct + hedged + reversed + missed = the number of true single edges.  A cell without the
    rule's columns (no selection stored) is skipped for that rule."""
    by = defaultdict(list)
    for r in rows:
        if r["estimator"] in estimators:
            by[(r["p"], r["c"], r["n"], r.get("generator") or "random", r["estimator"])].append(r)
    out = []
    for (p, c, n, gen, est), cell in sorted(by.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2],
                                                                        GENERATORS.index(kv[0][3]), estimators.index(kv[0][4]))):
        for rule in rules:
            vals = defaultdict(list)
            for r in cell:
                counts = {k: f(r.get(f"{rule}_{k}")) for k in ORIENT}
                if any(math.isnan(v) for v in counts.values()):
                    continue
                vals["true_edges"].append(f(r["n_true_edges"]))
                vals["two_cycles"].append(f(r["n_two_cycles"]))
                for col, key in (("both", "both"), ("half", "half"), ("neither", "missed_double"), ("correct", "correct"),
                                 ("hedged", "hedged"), ("reversed", "reversed"), ("missed", "missed_single")):
                    vals[col].append(counts[key])
            if not vals["both"]:
                continue
            row = {"p": p, "c": c, "n": n, "generator": gen, "estimator": est, "rule": rule, "n_graphs": len(vals["both"])}
            row.update({k: float(np.mean(v)) for k, v in vals.items()})
            row["single_edges"] = row["true_edges"] - 2 * row["two_cycles"]
            out.append(row)
    return out


def print_two_cycles(out: list[dict], rule="bic", estimators=("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")) -> None:
    keys = sorted({(r["p"], r["c"], r["n"], r["generator"]) for r in out if r["rule"] == rule},
                  key=lambda t: (int(t[0]), t[1], t[2], GENERATORS.index(t[3])))
    for p, c, n, gen in keys:
        cells = {r["estimator"]: r for r in out if (r["p"], r["c"], r["n"], r["generator"], r["rule"]) == (p, c, n, gen, rule)}
        first = next(iter(cells.values()))
        print(f"\np = {p}, {c}, n = {n}, generator {gen}, {rule} graph: how the true pairs are kept, means per graph "
              f"({first['n_graphs']} graphs; {first['two_cycles']:.2f} true 2-cycles, {first['single_edges']:.2f} true single edges)")
        print(f"  {'estimator':<10}{'both':>7}{'half':>7}{'neither':>8}   |{'correct':>9}{'hedged':>8}{'reversed':>9}{'missed':>8}")
        for est in estimators:
            r = cells.get(est)
            if r is None:
                continue
            print(f"  {est:<10}{r['both']:>7.2f}{r['half']:>7.2f}{r['neither']:>8.2f}   |{r['correct']:>9.2f}{r['hedged']:>8.2f}"
                  f"{r['reversed']:>9.2f}{r['missed']:>8.2f}")


def print_table(out: list[dict], estimators=("MCP", "MCP-up", "adaptive"), metric="max_f1") -> None:
    keys = sorted({(r["p"], r["c"], r["n"], r.get("generator", "random")) for r in out},
                  key=lambda t: (int(t[0]), t[1], t[2], GENERATORS.index(t[3])))
    for p, c, n, gen in keys:
        print(f"\np = {p}, {c}, n = {n}, generator {gen}: paired gain in {metric} over the lasso, within k; "
              f"graphs per (k, class), share identified on the true support")
        print(f"  {'k':>2} {'class':<11}{'graphs':>7}{'ident.':>7}" + "".join(f"{e:>22}" for e in estimators))
        cells = {(r["k"], r["graph_class"], r["estimator"]): r for r in out
                 if (r["p"], r["c"], r["n"], r.get("generator", "random"), r["metric"]) == (p, c, n, gen, metric)}
        for k in sorted({r["k"] for r in out if (r["p"], r["c"], r["n"], r.get("generator", "random")) == (p, c, n, gen)}, key=int):
            for cls in CLASSES:
                row0 = next((cells[(k, cls, e)] for e in estimators if (k, cls, e) in cells), None)
                if row0 is None:
                    continue
                vals = "".join(f"{cells[(k, cls, e)]['gain']:+13.3f} ±{cells[(k, cls, e)]['gain_se']:.3f}"
                               if (k, cls, e) in cells and not math.isnan(cells[(k, cls, e)]["gain"]) else f"{'':>22}"
                               for e in estimators)
                print(f"  {k:>2} {cls:<11}{row0['n_graphs']:>7}{row0['identified_share']:>7.2f}{vals}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--p", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()
    rows = load_rows(args.root / "campaign_per_dataset.csv", set(args.p))
    if not rows:
        raise SystemExit(f"no wave 1 rows at p = {args.p} under {args.root}")
    out = table(rows)
    with (args.root / "campaign_graph_class.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    for metric in ("max_f1", "bic_f1", "search_f1"):
        print_table(out, metric=metric)
    classes = defaultdict(int)
    for r in rows:
        if r["estimator"] == "lasso":
            classes[(r["p"], r.get("generator", "random"), r["k"], r["graph_class"])] += 1
    print("\ngraphs per (p, generator, k, class), over the C and n of the lasso cells:")
    for (p, gen, k, cls), count in sorted(classes.items(), key=lambda kv: (int(kv[0][0]), GENERATORS.index(kv[0][1]),
                                                                            int(kv[0][2]), kv[0][3])):
        print(f"  p = {p}, {gen:<7} k = {k}, {cls:<11} {count}")
    print(f"\nwrote {args.root / 'campaign_graph_class.csv'} ({len(out)} rows)")
    pairs = two_cycle_table(rows)
    if pairs:
        with (args.root / "campaign_two_cycles.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TWO_CYCLE_COLUMNS)
            w.writeheader()
            w.writerows(pairs)
        print_two_cycles(pairs)
        print(f"wrote {args.root / 'campaign_two_cycles.csv'} ({len(pairs)} rows: the true 2-cycles and single edges per estimator)")
    gens = table_by_generator(rows)
    if any(r["generator"] != "random" for r in gens):
        with (args.root / "campaign_generator.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(gens)
        print(f"wrote {args.root / 'campaign_generator.csv'} ({len(gens)} rows: the classes pooled per generator)")


if __name__ == "__main__":
    main()
