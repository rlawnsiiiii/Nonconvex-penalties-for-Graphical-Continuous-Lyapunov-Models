#!/usr/bin/env python3
"""Tables for the campaign of October 2026 (next_steps/051026/cluster_campaign_051026.md).
Numbers only; it reads the shards that ``cluster/submit_campaign.sh`` produces,

    <root>/<loss>_<estimator>_<C>_n<n>/shards/shard_*.npz      run_s1_shard.py  (waves 1, 3)
    <root>/search<v>_p<p>_<C>_n<n>/shards/shard_*.npz          run_search_shard.py  (wave 2; wave 5:
                                                               v = 100s, 100u, 30s, e1)

and writes into ``<root>``:

  campaign_per_dataset.csv   one row per (cell, graph).  For an estimator cell: the
                             path metrics of Figure 5 (max_f1, auc, aupr, max_acc; lambda
                             chosen knowing the truth), the graph selected by the score
                             with the BIC penalty (bic_*), the graph selected with
                             Dettling's eBIC penalty, gamma = 0.5 / 1 (ebic05_* / ebic1_*,
                             computed here from the stored scores) and the selected graph
                             after the greedy search (search_*; BIC penalty).  For a
                             wave 2 cell the two searches are two rows, estimators
                             "search-pure" and "search-truth", in the search_* columns.
  campaign_means.csv         means and standard errors per (loss, estimator, C, n, p), with
                             the cell's shard count (shards of n_shards), its CPU-hours and
                             the number of dead selections (bic_dead, search_dead: a graph
                             on which every score of the path, or the searched graph, is
                             +inf -- no stable refit; the stored index is then not a
                             selection, code review A3)
  campaign_paired.csv        paired differences to the lasso with the same C, loss, n
                             and p, and to Dettling's lasso (C = 2I), per true-C setting
  campaign_baseline_check.txt  (--check-baseline) the three wave 1 cells that repeat the
                             n-sweep baseline, compared with runs/nsweep_p10-20 graph by graph

Every row of the per-dataset table also carries the class of the true graph (plan E4, §5.1):
graph_class (dag / simple / two_cycles), n_two_cycles, frac_edges_2c and identified_min_eig, the
smallest eigenvalue of the design's Gram matrix on the true support (> 0: M* is identified on its
support), computed once per graph from the stored M* and the regenerated C* (gclm.graphs).

    python simulations/diagnostics/campaign.py                       # runs/campaign
    python simulations/diagnostics/campaign.py --root runs/local/campaign_rehearsal
    python simulations/diagnostics/campaign.py --check-baseline
    python simulations/diagnostics/campaign.py --drop-dead          # dead selections as NaN

Everything is paired: a graph is identified by (p, k, c_choice, rep) and is the same
graph in every cell.  Loading checks every cell (code review B2, 13 October 2026): a file
that cannot be read is named, a graph that appears in two shards of a cell is refused, and
a cell with fewer shards than its ``n_shards`` says is reported on stderr and in the tables.
Dead selections are kept in the means by default (as the densest graph, which is what the
stored index points at; the largest effect on a cell mean is 0.003) and counted;
``--drop-dead`` blanks their graph metrics instead.  CPU-hours of a cell rescored with the
likelihood refit (wave 8) are the rescoring's own (code review B3).
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
from gclm.data.simulate import CChoice, sample_drift, sample_volatility  # noqa: E402
from gclm.graphs import frac_edges_in_two_cycles, graph_class, identified_on_support, two_cycles  # noqa: E402
from run_s1_shard import ORIENT, _counts, unpack_supports  # noqa: E402

DEFAULT = ROOT / "runs" / "campaign"
BASELINE = ROOT / "runs" / "nsweep_p10-20"
#: <loss>_<estimator>_<C>_n<n>, or with _p<p> before _n<n> for the wave 4 cells (one p each); the wave 9
#: cells (plan of 13 October 2026) may carry a dot in the estimator (MCP-up-truth0.05), a weight knob
#: _w<wmin> (w0, w01, w025, w05), _s<share> (E5) and a graph class _dag before _n<n>
CELL = re.compile(r"^(?P<loss>direct|loglik|frobenius)_(?P<estimator>[A-Za-z][A-Za-z0-9.-]*)_(?P<c>C2I|Cresc)"
                  r"(?:_p(?P<p_label>[0-9]+))?(?:_w(?P<w_min>[0-9]+))?(?:_s(?P<share>[0-9]+))?(?:_(?P<graph>dag))?"
                  r"_n(?P<n>[0-9e]+|inf)$")
#: the generator columns of a cell drawn with the thesis generator (every cell before wave 9)
DGP_DEFAULT = {"weights": "normal", "w_min": "", "graph": "random", "two_cycle_share": "", "dgp": ""}
#: wave 2 cells (search_p10_Cresc) and the restart cells of wave 5 (search100s_p10_Cresc:
#: 100 sparse starts; search100u_...: 100 uniform starts; searche1_...: the eBIC penalty with
#: gamma = 1 in the score); the variant becomes part of the estimator label ("search-pure-100s")
#: wave 6: a source cell rescored with the eBIC term inside the selection and the search
#: (simulations/rescore_shard.py); its shards overlay the source cell's rows, matched by file name
RESCORE_CELL = re.compile(r"^rescore(?P<gamma>[0-9]+)_(?P<source>.+)$")
SEARCH_CELL = re.compile(r"^search(?P<variant>[0-9a-z]+)?_p(?P<p>[0-9]+)_(?P<c>C2I|Cresc)_n(?P<n>[0-9e]+|inf)$")
PATH_METRICS = ("max_f1", "aupr", "auc", "max_acc")
#: Dettling's eBIC penalty (his eq. 6.2): pen_eBIC = pen_BIC + 4 gamma |E| log p, for these
#: gammas.  Computed offline from the stored score (BIC penalty) of every support of the path;
#: column prefix "ebic<gamma>"
EBIC_GAMMAS = (0.5, 1.0)
#: the columns reported in the tables: path metrics, then the data-driven graphs; max_acc last (Dettling's
#: fourth path summary, added 15 October 2026 for the Figure 5 layouts of plot_figure5.py)
REPORT = ("max_f1", "aupr", "auc", "bic_f1", "ebic05_f1", "ebic1_f1", "search_f1", "ebic1_search_f1",
          "bic_skeleton_f1", "ebic1_skeleton_f1", "search_skeleton_f1", "ebic1_search_skeleton_f1", "max_acc")
ID = ("p", "k", "c_choice", "rep")
#: the per-graph metrics of a selected graph (graph_metrics), blanked by --drop-dead
GRAPH_METRICS = ("f1", "precision", "recall", "edges", "skeleton_f1", "orientation_accuracy") + ORIENT

_GRAPH_CACHE: dict[tuple, dict] = {}


def cell_meta(m: re.Match) -> dict:
    """The columns a cell's name carries: ``loss``, ``estimator``, ``c``, ``n``, and the generator's
    knobs of the wave 9 cells: ``weights`` / ``w_min`` from ``_w<wmin>`` (the digits after the leading
    0 are decimals: w025 = 0.25), ``graph`` from ``_dag``, and ``graph = "pairs"`` with
    ``two_cycle_share`` from ``_s<share>`` (s05 = 0.5; E5, 19 October 2026).  ``dgp`` is the one token
    that names the generator ("" for the thesis generator, else "w025", "dag", "s05"): the
    tables group by it, so that a knob cell is compared with the lasso drawn with the same knob."""
    g = m.groupdict()
    w, sh = g.get("w_min"), g.get("share")
    graph = g.get("graph") or "random"
    meta = {"loss": g["loss"], "estimator": g["estimator"], "c": g["c"], "n": g["n"], **DGP_DEFAULT}
    decimal = lambda digits: float(f"{digits[0]}.{digits[1:]}") if len(digits) > 1 else float(digits)  # noqa: E731
    if w:
        meta.update(weights="uniform", w_min=decimal(w), dgp=f"w{w}")
    if sh:
        meta.update(graph="pairs", two_cycle_share=decimal(sh), dgp=f"s{sh}")
    if graph != "random":
        meta.update(graph=graph, dgp=graph)
    return meta


def true_matrix(d, i: int) -> np.ndarray:
    p = int(d["p"][i])
    m = np.zeros((p, p))
    m[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
    return m


def graph_diagnostics(d, i: int, cfg: dict) -> dict:
    """The class of the true graph of data set ``i`` (plan of 13 October 2026, E4 §5.1):
    ``graph_class``, ``n_two_cycles`` and ``frac_edges_2c`` from the stored ``M*``, and
    ``identified_min_eig`` (:func:`gclm.graphs.identified_on_support`), which needs ``C*``: the
    instance is regenerated from its seed -- the two draws of ``draw_instance`` before the data --
    and the regenerated ``M*`` is checked against the stored one.  Cached per graph, since the same
    graph appears in every cell."""
    p, k, ci, rep = (int(d[key][i]) for key in ID)
    knobs = {"weights": cfg.get("weights", "normal"), "w_min": cfg.get("w_min", 0.0),
             "graph": cfg.get("graph", "random"),
             "two_cycle_share": cfg.get("two_cycle_share", 0.0)}      # absent before wave 9: the thesis generator
    key = (int(cfg["seed"]), p, k, ci, rep, bool(cfg["metzler"]), *knobs.values())
    if key not in _GRAPH_CACHE:
        m_true = true_matrix(d, i)
        rng = np.random.default_rng([cfg["seed"], p, k, ci, rep])
        m_again = sample_drift(p, k / p, rng, metzler=cfg["metzler"], **knobs)
        c_true = sample_volatility(p, list(CChoice)[ci], rng)
        if not np.allclose(m_again, m_true, rtol=0.0, atol=1e-12):
            raise ValueError(f"the stored M* of graph (p={p}, k={k}, c_choice={ci}, rep={rep}) is not the one "
                             f"its seed draws")
        support = m_true != 0
        _GRAPH_CACHE[key] = {"graph_class": graph_class(support), "n_two_cycles": two_cycles(support),
                             "frac_edges_2c": frac_edges_in_two_cycles(support),
                             "identified_min_eig": identified_on_support(m_true, c_true)}
    return _GRAPH_CACHE[key]
#: the order of the estimators in the tables: reference first, then the standard
#: paths, then the three that start from the lasso, then the same with the
#: likelihood refit behind the score (wave 5a, "-ml"), then the searches without a path
ORDER = ("lasso", "lasso-up", "MCP", "SCAD", "MCP-up", "MCP-up-lasso", "SCAD-up", "MCP-lla", "SCAD-lla",
         "adaptive",
         "MCP-up-exact", "MCP-up-l1", "MCP-up-l2", "MCP-up-truth0.05", "MCP-up-truth0.2",       # wave 9: E1
         "adaptive-exact", "adaptive-l1", "adaptive-l2", "adaptive-truth0.05", "adaptive-truth0.2",
         "threshold", "threshold-norefit",                                                      # wave 9: E2
         "lasso-pd", "MCP-pd", "SCAD-pd", "MCP-up-pd", "SCAD-up-pd", "adaptive-pd",             # wave 10
         "lasso-ml", "lasso-up-ml", "MCP-ml", "MCP-up-ml", "MCP-up-lasso-ml", "adaptive-ml",
         "search-pure", "search-pure-100s", "search-pure-100u", "search-pure-30s", "search-pure-300s",
         "search-pure-e1", "search-pure-ml", "search-pure-100sml", "search-truth", "search-truth-e1",
         "search-truth-ml", "search-truth-100sml")


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


def dead(scores) -> int:
    """1 if every score is +inf: no support with a stable refit, so argmin's 0 is not a selection."""
    return int(not np.isfinite(np.asarray(scores, float)).any())


def ebic_selection(d, i: int, gamma: float) -> dict:
    """The graph selected with Dettling's eBIC penalty on the path of dataset ``i`` of shard ``d``:
    the score of his eq. (6.2), ``(|E| + p) log n + 4 gamma |E| log p + L``, where the stored
    ``bic_scores`` already hold ``L + (|E| + p) log n`` (the score with the BIC penalty).
    Decoded from the stored supports; no refit.  The same metrics as for the other selected
    graphs (:func:`graph_metrics`, so ``ebic<gamma>_skeleton_f1`` is computed too; it used to be
    listed in ``REPORT`` but never filled, code review B3), the index and the dead flag."""
    p, n_lambda = int(d["p"][i]), len(d["lambdas"][i])
    scores = d["bic_scores"][i] + 4.0 * gamma * d["nnz"][i] * math.log(p)
    idx = int(np.argmin(scores))
    support = unpack_supports(d["supports_packed"][i], n_lambda, p)[idx]
    m_true = np.zeros((p, p))
    m_true[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
    conf, orient = _counts(support, m_true)
    tag = f"ebic{gamma:g}".replace(".", "")          # ebic05, ebic1
    return {**graph_metrics(conf, orient, tag), f"{tag}_index": idx, f"{tag}_dead": dead(scores)}


def load_shards(folder: Path) -> tuple[list[tuple[Path, object]], int | None]:
    """The shard files of a cell, with the checks of code review B2 (13 October 2026): a file
    that cannot be read is named (instead of an anonymous ``BadZipFile`` ending the whole
    analysis); a graph that appears in two files of the cell -- a stale shard of another layout
    next to the current one -- is refused; and when the submit script's ``n_shards`` is present
    the number of files is compared with it: an incomplete cell is reported on stderr and in the
    tables (columns ``shards``, ``n_shards``), not refused, because the campaign is analysed
    while it runs.  Returns the ``(file, npz)`` pairs and the expected count (``None`` without
    ``n_shards``).  A cell run with start blocks holds one row per graph and block, so the block
    is part of the identity."""
    files = sorted((folder / "shards").glob("shard_*.npz"))
    expected = None
    if (folder / "n_shards").exists():
        expected = int((folder / "n_shards").read_text().strip())
        if len(files) != expected:
            print(f"warning: {folder.name}: {len(files)} of {expected} shards present", file=sys.stderr)
    out, seen = [], {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            ids = [tuple(int(x) for x in g) for g in zip(*(d[k] for k in ID))]
            blocks = [int(b) for b in d["start_block"]] if "start_block" in d.files else [0] * len(ids)
        except Exception as err:            # noqa: BLE001 -- zipfile.BadZipFile, OSError, KeyError, ...
            raise RuntimeError(f"cannot read {f}: {type(err).__name__}: {err}") from err
        for g, b in zip(ids, blocks):
            if (g, b) in seen:
                raise ValueError(f"{folder.name}: graph {g} appears in both {seen[(g, b)]} and {f.name}")
            seen[(g, b)] = f.name
        out.append((f, d))
    return out, expected


def load_estimator_cell(folder: Path, meta: dict, rescored: list[dict] | None = None) -> list[dict]:
    """One row per graph of a ``run_s1_shard.py`` cell.  ``rescored`` holds, per rescoring of
    this cell (wave 6; one dictionary per rescoring folder, so two rescorings with different
    gammas do not overwrite each other -- code review B3), the map from a shard file name to the
    rescored file; their ``ebic<gamma>_search_*`` fields are overlaid."""
    rows = []
    files, expected = load_shards(folder)
    meta = {**meta, "shards": len(files), "n_shards": expected if expected is not None else len(files)}
    for f, d in files:
        names = [str(x) for x in d["c_choice_names"]]
        cfg = json.loads(str(d["config_json"]))
        has_bic, has_search = "bic_index" in d.files, "search_conf" in d.files
        extras = []
        for overlay in rescored or ():
            if f.name in overlay:
                e = np.load(overlay[f.name], allow_pickle=True)
                for key in ID:
                    if not np.array_equal(e[key], d[key]):
                        raise ValueError(f"{overlay[f.name]} does not line up with {f}")
                extras.append(e)
        # a cell rescored with the likelihood refit (wave 8) and written before 13 October 2026
        # carries the source path's time as `seconds`; its own cost is the selection and the
        # search (code review B3).  Newer refit shards store their own cost as `seconds`
        own_cost = "source_json" in d.files and "source_seconds" not in d.files
        for i in range(len(d["p"])):
            m = metrics_from_counts(d["conf_offdiag"][i])
            secs = float(d["select_seconds"][i]) if own_cost else float(d["seconds"][i])
            row = {**meta, "p": int(d["p"][i]), "k": int(d["k"][i]),
                   "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i]),
                   "n_true_edges": int(d["n_true_edges"][i]),
                   **{key: m[key] for key in PATH_METRICS}, "seconds": secs}
            row.update(graph_diagnostics(d, i, cfg))
            if has_bic:
                row.update(graph_metrics(d["bic_conf"][i], d["bic_orient"][i], "bic"))
                row["bic_index"] = int(d["bic_index"][i])
                row["bic_dead"] = dead(d["bic_scores"][i])
                for gamma in EBIC_GAMMAS:
                    row.update(ebic_selection(d, i, gamma))
                    tag = f"ebic{gamma:g}".replace(".", "")
                    src = d if f"{tag}_search_conf" in d.files else \
                        next((e for e in extras if f"{tag}_search_conf" in e.files), None)
                    if src is not None:                      # wave 5c or a wave 6 rescoring
                        row.update(graph_metrics(src[f"{tag}_search_conf"][i],
                                                 src[f"{tag}_search_orient"][i], f"{tag}_search"))
                        row[f"{tag}_search_moves"] = int(src[f"{tag}_search_moves"][i].sum())
                        row[f"{tag}_search_dead"] = dead([src[f"{tag}_search_score"][i]])
            if has_search:
                row.update(graph_metrics(d["search_conf"][i], d["search_orient"][i], "search"))
                row["search_moves"] = int(d["search_moves"][i].sum())
                row["search_seconds"] = float(d["search_seconds"][i])
                row["search_dead"] = dead([d["search_score"][i]])
            if "start_conf" in d.files:                 # a dense start or pilot (wave 9, plan §2.2)
                row.update(graph_metrics(d["start_conf"][i], d["start_orient"][i], "start"))
                for key in ("start_l1", "start_l2", "start_residual"):
                    row[key] = float(d[key][i])
                row["start_nnz"] = int(d["start_nnz"][i])
            rows.append(row)
    return rows


def load_search_cell(folder: Path, meta: dict, variant: str = "") -> list[dict]:
    """Two rows per graph of a ``run_search_shard.py`` cell: the search without a
    penalty ("search-pure", or "search-pure-<variant>" for the restart cells of
    wave 5) and the search started from the truth ("search-truth"), both in the
    ``search_*`` columns so that they line up with the estimators.  ``search_dead``
    flags a search that ended with score +inf (code review A3): for the search from
    the truth that is a truth without a stable least-squares refit, whose F1 of 1 is
    then no ceiling."""
    rows = []
    blocks: dict[tuple, dict] = {}      # start blocks (run_search_shard.py --start-blocks): per graph
    files, expected = load_shards(folder)
    meta = {**meta, "shards": len(files), "n_shards": expected if expected is not None else len(files)}
    for f, d in files:
        names = [str(x) for x in d["c_choice_names"]]
        cfg = json.loads(str(d["config_json"]))
        for i in range(len(d["p"])):
            base = {**meta, "p": int(d["p"][i]), "k": int(d["k"][i]),
                    "c_choice": names[int(d["c_choice"][i])], "rep": int(d["rep"][i]),
                    "n_true_edges": int(d["n_true_edges"][i]), **graph_diagnostics(d, i, cfg)}
            if "start_block" in d.files:
                key = (base["p"], base["k"], base["c_choice"], base["rep"])
                entry = blocks.setdefault(key, {"base": base, "n": int(d["start_blocks"][i]), "pure": {}})
                entry["pure"][int(d["start_block"][i])] = (
                    float(d["pure_score"][i]), d["pure_conf"][i], d["pure_orient"][i],
                    float(d["pure_seconds"][i]))
                if "truth_conf" in d.files:
                    entry["truth"] = (d["truth_conf"][i], d["truth_orient"][i],
                                      float(d["truth_seconds"][i]), int(d["truth_moves"][i].sum()),
                                      float(d["truth_score"][i]))
                continue
            for name in ("pure", "truth"):
                if f"{name}_conf" not in d.files:
                    continue
                label = f"search-{name}" + (f"-{variant}" if variant else "")
                row = {**base, "estimator": label,
                       **graph_metrics(d[f"{name}_conf"][i], d[f"{name}_orient"][i], "search"),
                       "seconds": float(d[f"{name}_seconds"][i]),
                       "search_dead": dead([d[f"{name}_score"][i]])}
                if name == "truth":
                    row["search_moves"] = int(d["truth_moves"][i].sum())
                rows.append(row)
    suffix = f"-{variant}" if variant else ""
    for key in sorted(blocks):          # the blocks of a graph back together; incomplete graphs are left out
        entry = blocks[key]
        if len(entry["pure"]) == entry["n"]:
            score, conf, orient, _ = min(entry["pure"].values(), key=lambda x: x[0])
            rows.append({**entry["base"], "estimator": f"search-pure{suffix}",
                         **graph_metrics(conf, orient, "search"),
                         "seconds": sum(x[3] for x in entry["pure"].values()), "search_dead": dead([score])})
        if "truth" in entry:
            conf, orient, seconds, moves, score = entry["truth"]
            rows.append({**entry["base"], "estimator": f"search-truth{suffix}",
                         **graph_metrics(conf, orient, "search"), "seconds": seconds,
                         "search_moves": moves, "search_dead": dead([score])})
    return rows


def load(root: Path) -> list[dict]:
    """All rows of all cells under ``root``; cells without shards are skipped."""
    rows = []
    rescored: dict[str, list[dict[str, Path]]] = {}      # source cell -> one file map per rescoring
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        r = RESCORE_CELL.match(folder.name)
        if r and (folder / "shards").is_dir():
            rescored.setdefault(r["source"], []).append(
                {f.name: f for f in (folder / "shards").glob("shard_*.npz")})
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if RESCORE_CELL.match(folder.name):
            continue
        m, s = CELL.match(folder.name), SEARCH_CELL.match(folder.name)
        if m:
            rows += load_estimator_cell(folder, {"cell": folder.name, **cell_meta(m)}, rescored.get(folder.name))
        elif s:
            rows += load_search_cell(folder, {"cell": folder.name, "loss": "direct",
                                              "c": s["c"], "n": s["n"], **DGP_DEFAULT}, s["variant"] or "")
    return rows


def blank_dead(rows: list[dict]) -> int:
    """``--drop-dead``: the graph metrics of a dead selection (``<prefix>_dead`` = 1) become NaN,
    so that they drop out of the means and the paired differences.  Returns how many flags were
    set.  The default keeps them: the stored index then points at the densest graph of the
    path, and the largest effect on a cell mean measured on the campaign of October 2026 is
    0.003 (25 of 99,200 rows; code review A3)."""
    count = 0
    for r in rows:
        for key in [k for k in r if k.endswith("_dead")]:
            if r[key]:
                count += 1
                prefix = key[:-len("_dead")]
                for suffix in GRAPH_METRICS:
                    if f"{prefix}_{suffix}" in r:
                        r[f"{prefix}_{suffix}"] = math.nan
    return count


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


def means_table(rows: list[dict], by_true_c: bool = False) -> list[dict]:
    """Means with standard errors per (loss, estimator, C, n, p, generator) over all graphs of a cell
    (the four settings of the true C pooled), or, with ``by_true_c``, per setting of the true
    C as well (column ``true_c``: 100 graphs each at p = 10, 20).  ``dgp`` names the generator
    ("" for the thesis generator; the wave 9 knobs otherwise), so that the knob cells form their
    own groups."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["loss"], r["estimator"], r["c"], r["n"], r["p"], r.get("dgp", "")) + \
            ((r["c_choice"],) if by_true_c else ())
        groups[key].append(r)
    out = []
    for key, g in sorted(groups.items(), key=lambda kv: (
            kv[0][0], n_key(kv[0][3]), kv[0][4], kv[0][2], kv[0][5], estimator_key(kv[0][1]), kv[0][6:])):
        loss, est, c, n, p, dgp = key[:6]
        row = {"loss": loss, "estimator": est, "c": c, "n": n, "p": p, "dgp": dgp, "graphs": len(g),
               "shards": g[0].get("shards"), "n_shards": g[0].get("n_shards")}
        if by_true_c:
            row["true_c"] = key[6]
        for col in REPORT:
            mean, se, _ = mean_se([r.get(col) for r in g])
            row[col], row[f"{col}_se"] = mean, se
        row["cpu_hours"] = sum(r["seconds"] for r in g) / 3600
        row["bic_dead"] = sum(int(r.get("bic_dead", 0)) for r in g)       # dead selections (code review A3)
        row["search_dead"] = sum(int(r.get("search_dead", 0)) for r in g)
        out.append(row)
    return out


def paired_table(rows: list[dict]) -> list[dict]:
    """Every estimator against two references on the same graphs: the lasso with
    the same loss, C, n and p ("same_c"), and Dettling's lasso, C = 2I ("dettling").
    Per true-C setting and pooled ("all").  An estimator whose score uses the likelihood
    refit ("-ml", waves 5a and 8) is compared with the lasso under the same refit
    ("lasso-ml"), so that the selected and searched graphs are judged by the same rule; a
    search scored with the eBIC penalty ("-e1", wave 5c) is compared with the lasso's search
    under that penalty (its ``ebic1_search_*`` columns, from wave 5c or the wave 6 rescoring).
    A column the reference lacks gives NaN: until 13 October 2026 an estimator's searched graph
    was then compared with the reference's selected graph under the name ``search_*_diff``
    (code review B1; it hit the wave 7 cells at p = 10 against wave 3's lasso).
    An estimator with the diagonal penalised ("-pd", wave 10) is compared with the lasso with the
    diagonal penalised ("lasso-pd"), and also with itself with the diagonal unpenalised
    ("unpenalised": e.g. adaptive-pd against adaptive on the same graphs)."""
    by = defaultdict(dict)                       # (loss, c, n, p, estimator, dgp) -> graph id -> row
    for r in rows:
        by[(r["loss"], r["c"], r["n"], r["p"], r["estimator"], r.get("dgp", ""))][tuple(r[k] for k in ID)] = r
    out = []
    for (loss, c, n, p, est, dgp), cell in sorted(by.items(), key=lambda kv: (
            kv[0][0], n_key(kv[0][2]), kv[0][3], kv[0][1], kv[0][5], estimator_key(kv[0][4]))):
        ref_est = ("lasso-ml" if est.endswith(("-ml", "sml"))        # the "-ml" cells and the 100-start
                   else "lasso-pd" if est.endswith("-pd") else "lasso")   # likelihood-refit searches; wave 10
        # the reference is drawn with the same generator (the knob cells of wave 9 pair with their own lasso)
        refs = [("same_c", (loss, c, n, p, ref_est, dgp)), ("dettling", (loss, "C2I", n, p, ref_est, dgp))]
        if est.endswith("-pd"):
            refs.append(("unpenalised", (loss, c, n, p, est[:-len("-pd")], dgp)))
        for ref_name, ref_key in refs:
            ref = by.get(ref_key)
            if ref is None or ref is cell:
                continue
            for setting in ("all", *sorted({g[2] for g in cell})):
                ids = [g for g in cell if g in ref and setting in ("all", g[2])]
                if not ids:
                    continue
                row = {"loss": loss, "estimator": est, "c": c, "n": n, "p": p, "dgp": dgp,
                       "reference": ref_name, "true_c": setting}
                for col in REPORT:
                    ref_col = f"ebic1_{col}" if est.endswith("-e1") and col.startswith("search_") else col
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
        head = (r["loss"], r["n"], r["p"], r.get("dgp", ""))
        if head != last:
            dgp = f", generator {r['dgp']}" if r.get("dgp") else ""
            print(f"\n{r['loss']} loss, n = {r['n']}, p = {r['p']}{dgp}")
            print(f"  {'estimator':<20}{'C':<7}{'graphs':>7}" + "".join(f"{c:>20}" for c in REPORT[:6]))
            last = head
        cells = "".join(f"{r[c]:>12.3f} ±{r[c + '_se']:.3f}" if not math.isnan(r[c]) else f"{'':>20}"
                        for c in REPORT[:6])
        print(f"  {r['estimator']:<20}{r['c']:<7}{r['graphs']:>7}{cells}")


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
    ap.add_argument("--drop-dead", action="store_true",
                    help="blank the graph metrics of dead selections (every score +inf) instead of "
                         "keeping them as the densest graph; they are counted either way")
    args = ap.parse_args()
    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no campaign shards under {args.root}")
    if args.drop_dead:
        print(f"{blank_dead(rows)} dead selections blanked (--drop-dead)")
    write(args.root / "campaign_per_dataset.csv", rows)
    table = means_table(rows)
    write(args.root / "campaign_means.csv", table)
    write(args.root / "campaign_means_by_true_c.csv", means_table(rows, by_true_c=True))
    write(args.root / "campaign_paired.csv", paired_table(rows))
    print_means(table)
    cells = sorted({r["cell"] for r in rows})
    print(f"\n{len(rows)} rows from {len(cells)} cells; "
          f"{sum(r['seconds'] for r in rows) / 3600:.1f} CPU-hours")
    incomplete = sorted({(r["cell"], r["shards"], r["n_shards"]) for r in rows if r["shards"] != r["n_shards"]})
    for cell, found, expected in incomplete:
        print(f"incomplete: {cell} has {found} of {expected} shards")
    n_dead = sum(int(r.get("bic_dead", 0)) + int(r.get("search_dead", 0)) for r in rows)
    print(f"{n_dead} dead selections (score +inf on every support, or a searched graph at +inf); "
          f"kept as stored{'' if not args.drop_dead else ' -- blanked (--drop-dead)'}")
    print(f"wrote {args.root / 'campaign_per_dataset.csv'}, campaign_means.csv, campaign_means_by_true_c.csv, "
          f"campaign_paired.csv")
    if args.check_baseline:
        for line in check_baseline(rows, args.root / "campaign_baseline_check.txt"):
            print(line)


if __name__ == "__main__":
    main()
