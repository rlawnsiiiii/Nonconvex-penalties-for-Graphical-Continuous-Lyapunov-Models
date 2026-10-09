#!/usr/bin/env python3
"""Tables for the campaign of October 2026 (next_steps/051026/cluster_campaign_051026.md).
Numbers only; it reads the shards that ``cluster/submit_campaign.sh`` produces,

    <root>/<loss>_<estimator>_<C>_n<n>/shards/shard_*.npz      run_s1_shard.py  (waves 1, 3)
    <root>/search<v>_p<p>_<C>_n<n>/shards/shard_*.npz          run_search_shard.py  (wave 2; wave 5:
                                                               v = 100s, 100u, 30s, e1)

and writes into ``<root>``:

  campaign_per_dataset.csv   one row per (cell, graph).  For an estimator cell: the
                             path metrics of Figure 5 (max_f1, auc, aupr, max_acc; lambda
                             chosen knowing the truth), the BIC-selected graph (bic_*),
                             the graph Dettling's extended BIC with gamma = 0.5 / 1 picks
                             (ebic05_* / ebic1_*, computed here from the stored scores)
                             and the BIC-selected graph after the BIC search (search_*).  For a
                             wave 2 cell the two searches are two rows, estimators
                             "search-pure" and "search-truth", in the search_* columns.
  campaign_means.csv         means and standard errors per (loss, estimator, C, n, p)
  campaign_paired.csv        paired differences to the lasso with the same C, loss, n
                             and p, and to Dettling's lasso (C = 2I), per true-C setting
  campaign_baseline_check.txt  (--check-baseline) the three wave 1 cells that repeat the
                             n-sweep baseline, compared with runs/nsweep_p10-20 graph by graph

    python simulations/diagnostics/campaign.py                       # runs/campaign
    python simulations/diagnostics/campaign.py --root runs/local/campaign_rehearsal
    python simulations/diagnostics/campaign.py --check-baseline

Everything is paired: a graph is identified by (p, k, c_choice, rep) and is the same
graph in every cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))

from aggregate_s1 import metrics_from_counts  # noqa: E402
from gclm.metrics import confusion  # noqa: E402
from run_s1_shard import ORIENT, unpack_supports  # noqa: E402

DEFAULT = ROOT / "runs" / "campaign"
BASELINE = ROOT / "runs" / "nsweep_p10-20"
#: <loss>_<estimator>_<C>_n<n>, or with _p<p> before _n<n> for the wave 4 cells (one p each)
CELL = re.compile(r"^(?P<loss>direct|loglik|frobenius)_(?P<estimator>[A-Za-z-]+)_(?P<c>C2I|Cresc)(?:_p(?P<p_label>[0-9]+))?_n(?P<n>[0-9e]+|inf)$")
#: wave 2 cells (search_p10_Cresc) and the restart cells of wave 5 (search100s_p10_Cresc:
#: 100 sparse starts; search100u_...: 100 uniform starts; searche1_...: the extended BIC with
#: gamma = 1 in the score); the variant becomes part of the estimator label ("search-pure-100s")
#: wave 6: a source cell rescored with the extended term inside the selection and the search
#: (simulations/rescore_shard.py); its shards overlay the source cell's rows, matched by file name
RESCORE_CELL = re.compile(r"^rescore(?P<gamma>[0-9]+)_(?P<source>.+)$")
SEARCH_CELL = re.compile(r"^search(?P<variant>[0-9a-z]+)?_p(?P<p>[0-9]+)_(?P<c>C2I|Cresc)_n(?P<n>[0-9e]+|inf)$")
PATH_METRICS = ("max_f1", "aupr", "auc", "max_acc")
#: Dettling's extended BIC (his eq. 6.2): BIC + 4 gamma |E| log p, for these gammas.  Computed
#: offline from the stored BIC of every support of the path; column prefix "ebic<gamma>"
EBIC_GAMMAS = (0.5, 1.0)
#: the columns reported in the tables: path metrics, then the data-driven graphs
REPORT = ("max_f1", "aupr", "auc", "bic_f1", "ebic05_f1", "ebic1_f1", "search_f1", "ebic1_search_f1",
          "bic_skeleton_f1", "ebic1_skeleton_f1", "search_skeleton_f1", "ebic1_search_skeleton_f1")
ID = ("p", "k", "c_choice", "rep")
#: the order of the estimators in the tables: reference first, then the standard
#: paths, then the three that start from the lasso, then the same with the
#: likelihood refit behind the BIC (wave 5a, "-ml"), then the searches without a path
ORDER = ("lasso", "lasso-up", "MCP", "SCAD", "MCP-up", "SCAD-up", "MCP-lla", "SCAD-lla",
         "adaptive", "lasso-ml", "MCP-up-ml", "adaptive-ml",
         "search-pure", "search-pure-100s", "search-pure-100u", "search-pure-30s", "search-pure-e1",
         "search-truth", "search-truth-e1")


def graph_metrics(conf: np.ndarray, orient: np.ndarray, prefix: str) -> dict:
    """Directed F1, precision, recall, number of edges, skeleton F1 and the
    orientation counts of one selected graph, from the stored counts.

    ``conf`` is (tp, fp, tn, fn) over the off-diagonal entries; ``orient`` is the
    breakdown of gclm.metrics.orientation_breakdown in the order ``ORIENT``.  A pair
    counts for the skeleton if either direction is present, so the skeleton's true
    positives are the detected true pairs whatever their orientation."""
    tp, fp, _, fn = (int(x) for x in conf)
    o = dict(zip(ORIENT, (int(x) for x in orient)))
    sk_tp = o["correct"] + o["reversed"] + o["hedged"] + o["both"] + o["half"]
    sk_fp = o["fp_single"] + o["fp_double"]
    sk_fn = o["missed_single"] + o["missed_double"]
    committed = o["correct"] + o["reversed"]
    return {
        f"{prefix}_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        f"{prefix}_precision": tp / (tp + fp) if tp + fp else 1.0,
        f"{prefix}_recall": tp / (tp + fn) if tp + fn else 1.0,
        f"{prefix}_edges": tp + fp,
        f"{prefix}_skeleton_f1": 2 * sk_tp / (2 * sk_tp + sk_fp + sk_fn) if 2 * sk_tp + sk_fp + sk_fn else 0.0,
        # of the true single-direction edges on which the graph commits to one direction
        f"{prefix}_orientation_accuracy": o["correct"] / committed if committed else math.nan,
        **{f"{prefix}_{key}": o[key] for key in ORIENT},
    }


def ebic_selection(d, i: int, gamma: float) -> dict:
    """The graph Dettling's extended BIC picks on the path of dataset ``i`` of shard ``d``:
    his eq. (6.2), ``(|E| + p) log n + 4 gamma |E| log p + L``, where the stored ``bic_scores``
    already hold ``L + (|E| + p) log n``.  Decoded from the stored supports; no refit."""
    p, n_lambda = int(d["p"][i]), len(d["lambdas"][i])
    scores = d["bic_scores"][i] + 4.0 * gamma * d["nnz"][i] * math.log(p)
    idx = int(np.argmin(scores))
    support = unpack_supports(d["supports_packed"][i], n_lambda, p)[idx]
    m_true = np.zeros((p, p))
    m_true[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
    cf = confusion(support.astype(float), m_true)
    tag = f"ebic{gamma:g}".replace(".", "")          # ebic05, ebic1
    return {f"{tag}_f1": cf.f1, f"{tag}_precision": cf.precision, f"{tag}_recall": cf.tpr,
            f"{tag}_edges": int(support.sum()), f"{tag}_index": idx}


def load_estimator_cell(folder: Path, meta: dict, rescored: dict | None = None) -> list[dict]:
    """One row per graph of a ``run_s1_shard.py`` cell.  ``rescored`` maps a shard file name
    to a rescoring of it (wave 6); its ``ebic<gamma>_search_*`` fields are overlaid."""
    rows = []
    for f in sorted((folder / "shards").glob("shard_*.npz")):
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["c_choice_names"]]
        has_bic, has_search = "bic_index" in d.files, "search_conf" in d.files
        extra = None
        if rescored and f.name in rescored:
            extra = np.load(rescored[f.name], allow_pickle=True)
            for key in ID:
                if not np.array_equal(extra[key], d[key]):
                    raise ValueError(f"{rescored[f.name]} does not line up with {f}")
        for i in range(len(d["p"])):
            m = metrics_from_counts(d["conf_offdiag"][i])
            row = {**meta, "p": int(d["p"][i]), "k": int(d["k"][i]),
                   "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i]),
                   "n_true_edges": int(d["n_true_edges"][i]),
                   **{key: m[key] for key in PATH_METRICS}, "seconds": float(d["seconds"][i])}
            if has_bic:
                row.update(graph_metrics(d["bic_conf"][i], d["bic_orient"][i], "bic"))
                row["bic_index"] = int(d["bic_index"][i])
                for gamma in EBIC_GAMMAS:
                    row.update(ebic_selection(d, i, gamma))
                    tag = f"ebic{gamma:g}".replace(".", "")
                    src = d if f"{tag}_search_conf" in d.files else \
                        extra if extra is not None and f"{tag}_search_conf" in extra.files else None
                    if src is not None:                      # wave 5c or a wave 6 rescoring
                        row.update(graph_metrics(src[f"{tag}_search_conf"][i],
                                                 src[f"{tag}_search_orient"][i], f"{tag}_search"))
                        row[f"{tag}_search_moves"] = int(src[f"{tag}_search_moves"][i].sum())
            if has_search:
                row.update(graph_metrics(d["search_conf"][i], d["search_orient"][i], "search"))
                row["search_moves"] = int(d["search_moves"][i].sum())
                row["search_seconds"] = float(d["search_seconds"][i])
            rows.append(row)
    return rows


def load_search_cell(folder: Path, meta: dict, variant: str = "") -> list[dict]:
    """Two rows per graph of a ``run_search_shard.py`` cell: the search without a
    penalty ("search-pure", or "search-pure-<variant>" for the restart cells of
    wave 5) and the search started from the truth ("search-truth"), both in the
    ``search_*`` columns so that they line up with the estimators."""
    rows = []
    for f in sorted((folder / "shards").glob("shard_*.npz")):
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["c_choice_names"]]
        for i in range(len(d["p"])):
            base = {**meta, "p": int(d["p"][i]), "k": int(d["k"][i]),
                    "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i]),
                    "n_true_edges": int(d["n_true_edges"][i])}
            for name in ("pure", "truth"):
                if f"{name}_conf" not in d.files:
                    continue
                label = f"search-{name}" + (f"-{variant}" if variant else "")
                row = {**base, "estimator": label,
                       **graph_metrics(d[f"{name}_conf"][i], d[f"{name}_orient"][i], "search"),
                       "seconds": float(d[f"{name}_seconds"][i])}
                if name == "truth":
                    row["search_moves"] = int(d["truth_moves"][i].sum())
                rows.append(row)
    return rows


def load(root: Path) -> list[dict]:
    """All rows of all cells under ``root``; cells without shards are skipped."""
    rows = []
    rescored: dict[str, dict[str, Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        r = RESCORE_CELL.match(folder.name)
        if r and (folder / "shards").is_dir():
            for f in (folder / "shards").glob("shard_*.npz"):
                rescored.setdefault(r["source"], {})[f.name] = f
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if RESCORE_CELL.match(folder.name):
            continue
        m, s = CELL.match(folder.name), SEARCH_CELL.match(folder.name)
        if m:
            meta = {k: v for k, v in m.groupdict().items() if k != "p_label"}
            rows += load_estimator_cell(folder, {"cell": folder.name, **meta}, rescored.get(folder.name))
        elif s:
            rows += load_search_cell(folder, {"cell": folder.name, "loss": "direct",
                                              "c": s["c"], "n": s["n"]}, s["variant"] or "")
    return rows


def n_key(n: str) -> float:
    return math.inf if n == "inf" else float(n)


def estimator_key(name: str) -> tuple[int, str]:
    return (ORDER.index(name) if name in ORDER else len(ORDER), name)


def mean_se(values) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return math.nan, math.nan, 0
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


def paired(x, y) -> dict:
    """Mean, standard error and z of the paired difference ``x - y`` (NaNs dropped pairwise)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if len(d) < 2:
        return {"pairs": len(d), "diff": float(d.mean()) if len(d) else math.nan, "se": math.nan, "z": math.nan}
    se = d.std(ddof=1) / math.sqrt(len(d))
    return {"pairs": len(d), "diff": float(d.mean()), "se": float(se),
            "z": float(d.mean() / se) if se > 0 else (0.0 if d.mean() == 0 else math.nan)}


def means_table(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["loss"], r["estimator"], r["c"], r["n"], r["p"])].append(r)
    out = []
    for (loss, est, c, n, p), g in sorted(groups.items(), key=lambda kv: (
            kv[0][0], n_key(kv[0][3]), kv[0][4], kv[0][2], estimator_key(kv[0][1]))):
        row = {"loss": loss, "estimator": est, "c": c, "n": n, "p": p, "graphs": len(g)}
        for col in REPORT:
            mean, se, _ = mean_se([r.get(col) for r in g])
            row[col], row[f"{col}_se"] = mean, se
        row["cpu_hours"] = sum(r["seconds"] for r in g) / 3600
        out.append(row)
    return out


def paired_table(rows: list[dict]) -> list[dict]:
    """Every estimator against two references on the same graphs: the lasso with
    the same loss, C, n and p ("same_c"), and Dettling's lasso, C = 2I ("dettling").
    Per true-C setting and pooled ("all")."""
    by = defaultdict(dict)                       # (loss, c, n, p, estimator) -> graph id -> row
    for r in rows:
        by[(r["loss"], r["c"], r["n"], r["p"], r["estimator"])][tuple(r[k] for k in ID)] = r
    out = []
    for (loss, c, n, p, est), cell in sorted(by.items(), key=lambda kv: (
            kv[0][0], n_key(kv[0][2]), kv[0][3], kv[0][1], estimator_key(kv[0][4]))):
        for ref_name, ref_key in (("same_c", (loss, c, n, p, "lasso")),
                                  ("dettling", (loss, "C2I", n, p, "lasso"))):
            ref = by.get(ref_key)
            if ref is None or ref is cell:
                continue
            for setting in ("all", *sorted({g[2] for g in cell})):
                ids = [g for g in cell if g in ref and setting in ("all", g[2])]
                if not ids:
                    continue
                row = {"loss": loss, "estimator": est, "c": c, "n": n, "p": p,
                       "reference": ref_name, "true_c": setting}
                for col in REPORT:
                    # an estimator's searched graph is compared with the reference's
                    # BIC-selected graph when the reference has no search of its own
                    ref_col = col if any(col in ref[g] for g in ids) else col.replace("search_", "bic_")
                    x = [cell[g].get(col, math.nan) for g in ids]
                    y = [ref[g].get(ref_col, math.nan) for g in ids]
                    stat = paired(x, y)
                    row.update({f"{col}_diff": stat["diff"], f"{col}_z": stat["z"]})
                    row["pairs"] = max(row.get("pairs", 0), stat["pairs"])
                out.append(row)
    return out


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_means(table: list[dict]) -> None:
    last = None
    for r in table:
        head = (r["loss"], r["n"], r["p"])
        if head != last:
            print(f"\n{r['loss']} loss, n = {r['n']}, p = {r['p']}")
            print(f"  {'estimator':<14}{'C':<7}{'graphs':>7}" + "".join(f"{c:>20}" for c in REPORT[:6]))
            last = head
        cells = "".join(f"{r[c]:>12.3f} ±{r[c + '_se']:.3f}" if not math.isnan(r[c]) else f"{'':>20}"
                        for c in REPORT[:6])
        print(f"  {r['estimator']:<14}{r['c']:<7}{r['graphs']:>7}{cells}")


def check_baseline(rows: list[dict], out: Path) -> list[str]:
    """Wave 1 repeats three cells of the n-sweep (direct loss, C = 2I, standard
    paths) with the changed code.  Their path metrics must agree with the October
    run graph by graph: exactly for the lasso, and up to the handful of graphs on
    which the nonconvex paths depend on the machine (S2b Section 2)."""
    lines = []
    for pen in ("lasso", "MCP", "SCAD"):
        for n in sorted({r["n"] for r in rows}, key=n_key):
            new = {tuple(r[k] for k in ID): r for r in rows
                   if (r["loss"], r["estimator"], r["c"], r["n"]) == ("direct", pen, "C2I", n)}
            f = BASELINE / f"direct_{pen}_n{n}" / "s1_per_dataset.csv"
            if not new or not f.exists():
                continue
            old = {(int(r["p"]), int(r["k"]), r["c_choice"], int(r["rep"])): r
                   for r in csv.DictReader(f.open())}
            ids = [g for g in new if g in old]
            diff = {m: np.array([abs(new[g][m] - float(old[g][m])) for g in ids]) for m in PATH_METRICS}
            differ = int(np.sum(np.max([diff[m] for m in PATH_METRICS], axis=0) > 1e-12))
            lines.append(f"direct {pen}, C = 2I, n = {n}: {len(ids)} graphs in both runs, {differ} differ; "
                         + ", ".join(f"max |diff| {m} {diff[m].max():.1e}" for m in PATH_METRICS[:3]))
    out.write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT)
    ap.add_argument("--check-baseline", action="store_true",
                    help="compare the repeated baseline cells with runs/nsweep_p10-20")
    args = ap.parse_args()
    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no campaign shards under {args.root}")
    write(args.root / "campaign_per_dataset.csv", rows)
    table = means_table(rows)
    write(args.root / "campaign_means.csv", table)
    write(args.root / "campaign_paired.csv", paired_table(rows))
    print_means(table)
    cells = sorted({r["cell"] for r in rows})
    print(f"\n{len(rows)} rows from {len(cells)} cells; "
          f"{sum(r['seconds'] for r in rows) / 3600:.1f} CPU-hours")
    print(f"wrote {args.root / 'campaign_per_dataset.csv'}, campaign_means.csv, campaign_paired.csv")
    if args.check_baseline:
        for line in check_baseline(rows, args.root / "campaign_baseline_check.txt"):
            print(line)


if __name__ == "__main__":
    main()
