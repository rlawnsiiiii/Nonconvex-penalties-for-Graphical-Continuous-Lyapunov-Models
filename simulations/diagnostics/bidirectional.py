#!/usr/bin/env python3
"""S3a -- what each penalty does with the two directions of a pair.

``run`` refits the Figure 5 datasets (generator and seeds of run_s1_shard.py:
standardised input, C = 2I in the fit) at the requested p and n with every
penalty, and stores, for every lambda of the path, the selection pattern of
every unordered pair next to the true pattern: one compressed .npz per
(p, n, penalty) under <out>/raw/ (numbers only, not tracked by git).

``summarize`` reads them back, checks the n = 1000 refits against the stored
S1 / S1b best-F1 estimates, prints the tables of
simulations/S3a_bidirectional_edges.md and writes the CSVs the figures are
drawn from (plot_bidirectional.py).

    python simulations/diagnostics/bidirectional.py run --p 10 20 --n 1000 10000 --reps 10 --workers 6
    python simulations/diagnostics/bidirectional.py summarize

Pair codes (gclm.metrics._pair_patterns, pairs i < j in np.triu_indices order):
0 neither entry, 1 M[i, j] only, 2 M[j, i] only, 3 both.  The edge i -> j is M[j, i].
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import _pair_patterns, evaluate_path  # noqa: E402
from gclm.solvers.path import fit_path  # noqa: E402

PENALTIES = ("lasso", "MCP", "SCAD")
C_NAMES = tuple(c.value for c in CChoice)
FIRST = ("true", "reversed", "both", "never")
SINGLE = ("correct", "hedged", "reversed", "missed")
DEFAULT_OUT = ROOT / "runs" / "s3a_bidirectional"
# the stored best-F1 estimates the n = 1000 refits must reproduce
STORED = {"lasso": ROOT / "runs/s1_dettling_reproduction",
          "MCP": ROOT / "runs/s1b_pilot_p10-20/MCP",
          "SCAD": ROOT / "runs/s1b_pilot_p10-20/SCAD"}


# --------------------------------------------------------------------------- #
# pair-level bookkeeping (tested in tests/test_bidirectional.py)
# --------------------------------------------------------------------------- #


def outcome_counts(patterns: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Counts per estimate (last axis = pairs) of the categories of
    ``gclm.metrics.orientation_breakdown``, plus the selected and the
    bidirectional pairs.  ``patterns`` has shape (..., n_pairs), ``truth``
    (n_pairs,)."""
    single = (truth == 1) | (truth == 2)
    double, none = truth == 3, truth == 0
    one = (patterns == 1) | (patterns == 2)
    both = patterns == 3
    return {
        "correct": np.sum(single & (patterns == truth), axis=-1),
        "hedged": np.sum(single & both, axis=-1),
        "reversed": np.sum(single & one & (patterns != truth), axis=-1),
        "missed_single": np.sum(single & (patterns == 0), axis=-1),
        "both": np.sum(double & both, axis=-1),
        "half": np.sum(double & one, axis=-1),
        "missed_double": np.sum(double & (patterns == 0), axis=-1),
        "fp_single": np.sum(none & one, axis=-1),
        "fp_double": np.sum(none & both, axis=-1),
        "selected_pairs": np.sum(patterns != 0, axis=-1),
        "bidirectional": np.sum(both, axis=-1),
    }


def first_entry(patterns: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """For every pair, what appears first going from lambda_max towards the
    dense end.  ``patterns`` is (n_lambda, n_pairs) in the path's own order
    (increasing lambda, so the dense end first).  Codes: 0 the true direction
    alone, 1 the other direction alone, 2 both at the same lambda, 3 never.
    Meaningful for true single-direction pairs (truth 1 or 2)."""
    sparse_first = patterns[::-1]
    selected = sparse_first != 0
    ever = selected.any(axis=0)
    first = sparse_first[np.argmax(selected, axis=0), np.arange(patterns.shape[1])]
    out = np.full(patterns.shape[1], 3, dtype=np.int8)
    out[ever & (first == truth)] = 0
    out[ever & (first == 3)] = 2
    out[ever & (first != truth) & (first != 3)] = 1
    return out


def single_outcome(pattern: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Outcome of each true single-direction pair under one estimate:
    0 correct, 1 hedged, 2 reversed, 3 missed (index into ``SINGLE``)."""
    out = np.full(pattern.shape, 3, dtype=np.int8)      # missed
    one = (pattern == 1) | (pattern == 2)
    out[one & (pattern == truth)] = 0                   # correct
    out[one & (pattern != truth)] = 2                   # reversed
    out[pattern == 3] = 1                               # hedged
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def fit_task(task):
    p, n, k, c, rep, penalties = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c), rep])
    m_true, _, _, s = draw_instance(p, k, n, CChoice(c), rng,
                                    metzler=cfg.metzler, standardize=cfg.standardize)
    fits = {}
    for pen in penalties:
        path = fit_path(s, 2.0 * np.eye(p), loss="direct", n_lambda=cfg.n_lambda,
                        ratio=cfg.lambda_ratio, penalty=pen, tol=cfg.tol,
                        solver=cfg.solver, convention=cfg.convention)
        pats = np.stack([_pair_patterns(m) for m in path.estimates]).astype(np.int8)
        f1 = np.asarray(evaluate_path(path.estimates, m_true)["f1"], dtype=np.float32)
        fits[pen] = (pats, f1, path.lambdas / path.lambdas.max())
    return (k, C_NAMES.index(c), rep), _pair_patterns(m_true).astype(np.int8), fits


def n_label(n) -> str:
    return "inf" if np.isinf(n) else str(int(n))


def cmd_run(args) -> None:
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cfg = S1Config()
    for p in args.p:
        for n in args.n:
            tasks = [(p, n, k, c, r, tuple(args.penalties))
                     for k in cfg.k_values for c in C_NAMES for r in range(args.reps)]
            t0 = time.time()
            with ProcessPoolExecutor(args.workers) as ex:
                res = list(ex.map(fit_task, tasks, chunksize=1))
            meta = np.array([r[0] for r in res], dtype=np.int16)
            truth = np.stack([r[1] for r in res])
            for pen in args.penalties:
                lam_rel = res[0][2][pen][2]
                assert all(np.allclose(r[2][pen][2], lam_rel) for r in res)
                np.savez_compressed(
                    raw / f"p{p}_n{n_label(n)}_{pen}.npz",
                    k=meta[:, 0], c_choice=meta[:, 1], rep=meta[:, 2], truth=truth,
                    patterns=np.stack([r[2][pen][0] for r in res]),
                    f1=np.stack([r[2][pen][1] for r in res]), lam_rel=lam_rel,
                    config_json=json.dumps({
                        "p": p, "n_obs": n, "penalty": pen, "reps": args.reps,
                        "k_values": list(cfg.k_values), "c_choices": list(C_NAMES),
                        "n_lambda": cfg.n_lambda, "lambda_ratio": cfg.lambda_ratio,
                        "tol": cfg.tol, "standardize": cfg.standardize, "seed": cfg.seed,
                        "solver": cfg.solver, "convention": cfg.convention,
                    }))
            print(f"p={p} n={n_label(n)}: {len(tasks)} datasets x {len(args.penalties)} penalties "
                  f"in {(time.time() - t0) / 60:.1f} min -> {raw}", flush=True)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def load(raw: Path):
    cells = {}
    for f in sorted(raw.glob("p*_n*_*.npz")):
        d = np.load(f, allow_pickle=False)
        cfg = json.loads(str(d["config_json"]))
        cells[(cfg["p"], n_label(cfg["n_obs"]), cfg["penalty"])] = {
            k: d[k] for k in ("k", "c_choice", "rep", "truth", "patterns", "f1", "lam_rel")}
    return cells


def best_index(f1: np.ndarray) -> np.ndarray:
    """The best-F1 lambda of every path, first maximum as in run_s1_shard.py."""
    return np.argmax(np.nan_to_num(f1, nan=-1.0), axis=1)


def validate(cells) -> list[str]:
    """Best-F1 pair patterns of the n = 1000 refits vs. the stored S1 / S1b runs."""
    from orientation import load_best_f1
    lines = []
    for pen, run in STORED.items():
        keys = [key for key in cells if key[1] == "1000" and key[2] == pen]
        if not keys or not (run / "s1_shards").exists():
            continue
        ps = sorted({key[0] for key in keys})
        reps = int(max(cells[keys[0]]["rep"])) + 1
        stored = load_best_f1(run, reps, ps)
        mismatched = total = 0
        for key in keys:
            c = cells[key]
            best = best_index(c["f1"])
            for i in range(len(best)):
                skey = (key[0], int(c["k"][i]), C_NAMES[int(c["c_choice"][i])], int(c["rep"][i]))
                if skey not in stored:
                    continue
                total += 1
                mt, mh = stored[skey]
                same_truth = np.array_equal(_pair_patterns(mt), c["truth"][i])
                same_est = np.array_equal(_pair_patterns(mh), c["patterns"][i, best[i]])
                mismatched += not (same_truth and same_est)
        lines.append(f"{pen}: {total} datasets at n = 1000 compared with {run.relative_to(ROOT)}, "
                     f"{mismatched} differ")
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_summarize(args) -> None:
    cells = load(args.out / "raw")
    if not cells:
        sys.exit(f"no raw files under {args.out / 'raw'}")
    pens = [p for p in PENALTIES if any(key[2] == p for key in cells)]
    grid = sorted({(key[0], key[1]) for key in cells}, key=lambda t: (t[0], float(t[1])))

    print("validation:")
    for line in validate(cells):
        print("  " + line)

    # truth check (the graphs are the same for every n and penalty)
    truth_rows = []
    for p in sorted({key[0] for key in cells}):
        c = cells[next(key for key in cells if key[0] == p)]
        for k in sorted(set(c["k"].tolist())):
            t = c["truth"][c["k"] == k]
            edges = np.sum((t == 1) | (t == 2), axis=1) + 2 * np.sum(t == 3, axis=1)
            cyc = np.sum(t == 3, axis=1)
            truth_rows.append({"p": p, "k": k, "graphs": len(t), "true_edges": edges.mean(),
                               "two_cycles": cyc.mean(),
                               "share_of_edges_in_2cycles": (2 * cyc).sum() / edges.sum()})

    path_rows, best_rows, first_rows, cell_rows = [], [], [], []
    for (p, n) in grid:
        for pen in pens:
            c = cells[(p, n, pen)]
            pats, truth, f1 = c["patterns"], c["truth"], c["f1"]
            n_data, n_lam, _ = pats.shape
            best = best_index(f1)
            counts = [outcome_counts(pats[i], truth[i]) for i in range(n_data)]   # each (n_lam,)
            path_mean = {key: np.mean([cnt[key] for cnt in counts], axis=0) for key in counts[0]}
            true_pairs = np.mean([np.sum(t != 0) for t in truth])
            true_cyc = np.mean([np.sum(t == 3) for t in truth])
            for li in range(n_lam):
                path_rows.append({
                    "p": p, "n": n, "penalty": pen, "lambda_index": li,
                    "lambda_rel": float(c["lam_rel"][li]),
                    **{key: float(path_mean[key][li]) for key in (
                        "selected_pairs", "bidirectional", "both", "hedged", "fp_double",
                        "correct", "reversed")},
                    "true_pairs": true_pairs, "true_2cycles": true_cyc})
            at_best = {key: np.array([cnt[key][best[i]] for i, cnt in enumerate(counts)])
                       for key in counts[0]}
            n_single = sum(np.sum((t == 1) | (t == 2)) for t in truth)
            n_double = sum(np.sum(t == 3) for t in truth)
            for group, keys, denom in (("single edge", ("correct", "hedged", "reversed", "missed_single"), n_single),
                                       ("2-cycle", ("both", "half", "missed_double"), n_double)):
                for key in keys:
                    best_rows.append({"p": p, "n": n, "penalty": pen, "group": group,
                                      "outcome": key.replace("_single", "").replace("_double", ""),
                                      "count_per_dataset": at_best[key].mean(),
                                      "share": at_best[key].sum() / denom})
            # which direction enters first, and what is left of it at the best-F1 point
            trans = np.zeros((len(FIRST), len(SINGLE)), dtype=int)
            for i in range(n_data):
                single = (truth[i] == 1) | (truth[i] == 2)
                fe = first_entry(pats[i], truth[i])[single]
                ob = single_outcome(pats[i, best[i]], truth[i])[single]
                np.add.at(trans, (fe, ob), 1)
            for a, fname in enumerate(FIRST):
                for b, oname in enumerate(SINGLE):
                    first_rows.append({"p": p, "n": n, "penalty": pen, "first": fname,
                                       "at_best_f1": oname, "pairs": int(trans[a, b])})
            dense = {key: float(path_mean[key][0]) for key in ("selected_pairs", "bidirectional")}
            cell_rows.append({
                "p": p, "n": n, "penalty": pen, "datasets": n_data,
                "max_f1": float(np.mean(np.nanmax(f1, axis=1))),
                "selected_pairs_at_best": at_best["selected_pairs"].mean(),
                "bidir_at_best": at_best["bidirectional"].mean(),
                "datasets_with_bidir_at_best": float(np.mean(at_best["bidirectional"] > 0)),
                "bidir_max_along_path": float(path_mean["bidirectional"].max()),
                "datasets_with_bidir_anywhere": float(np.mean(
                    [cnt["bidirectional"].max() > 0 for cnt in counts])),
                "two_cycles_both_at_best": at_best["both"].sum() / n_double,
                "two_cycles_both_anywhere": sum(
                    np.sum(np.any(pats[i] == 3, axis=0) & (truth[i] == 3)) for i in range(n_data)) / n_double,
                "selected_pairs_dense_end": dense["selected_pairs"],
                "bidir_dense_end": dense["bidirectional"],
                "first_true": trans[0].sum() / trans.sum(),
                "first_reversed": trans[1].sum() / trans.sum(),
                "first_both": trans[2].sum() / trans.sum(),
                "first_reversed_still_reversed_at_best": trans[1, 2] / max(trans[1].sum(), 1),
                "first_reversed_corrected_at_best": (trans[1, 0] + trans[1, 1]) / max(trans[1].sum(), 1),
            })

    out = args.out
    write_csv(out / "truth_check.csv", truth_rows)
    write_csv(out / "path_means.csv", path_rows)
    write_csv(out / "best_f1_outcomes.csv", best_rows)
    write_csv(out / "first_entry.csv", first_rows)
    write_csv(out / "summary.csv", cell_rows)
    write_example(cells, out / "example.csv")

    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    for name, rows in (("truth", truth_rows), ("summary", cell_rows)):
        print(f"\n{name}:")
        print("  " + " | ".join(rows[0]))
        for r in rows:
            print("  " + " | ".join(fmt(v) for v in r.values()))
    print(f"\nwrote {', '.join(f.name for f in sorted(out.glob('*.csv')))} -> {out}")


def write_example(cells, path: Path) -> None:
    """One p = 10, n = 1000 graph for the network figure.  Sparse enough to read
    (k = 2, C_ID preferred) and with at least two true 2-cycles; among those, the
    graph on which the lasso's bidirectional pairs plus MCP's reversed and
    half-found pairs are most numerous at the best-F1 point.  An illustration
    chosen for visibility, not a typical case."""
    keys = {pen: (10, "1000", pen) for pen in PENALTIES}
    if not all(k in cells for k in keys.values()):
        return
    lasso, mcp = cells[keys["lasso"]], cells[keys["MCP"]]
    bl, bm = best_index(lasso["f1"]), best_index(mcp["f1"])
    score = []
    for i in range(len(bl)):
        t = lasso["truth"][i]
        cl = outcome_counts(lasso["patterns"][i, bl[i]], t)
        cm = outcome_counts(mcp["patterns"][i, bm[i]], t)
        k2, cid = int(lasso["k"][i]) == 2, int(lasso["c_choice"][i]) == 0
        ok = np.sum(t == 3) >= 2
        score.append((ok and k2 and cid, ok and k2, ok,
                      cl["bidirectional"] + cm["reversed"] + cm["half"], -i))
    i = max(range(len(score)), key=lambda j: score[j])
    iu, ju = np.triu_indices(10, k=1)
    rows = []
    for a, (pi, pj) in enumerate(zip(iu, ju)):
        row = {"p": 10, "n": 1000, "k": int(lasso["k"][i]), "c_choice": C_NAMES[int(lasso["c_choice"][i])],
               "rep": int(lasso["rep"][i]), "i": int(pi), "j": int(pj),
               "truth": int(lasso["truth"][i][a])}
        for pen in PENALTIES:
            c = cells[keys[pen]]
            row[pen] = int(c["patterns"][i, best_index(c["f1"])[i], a])
        rows.append(row)
    write_csv(path, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--p", type=int, nargs="+", default=[10, 20])
    r.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000])
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--penalties", nargs="+", default=list(PENALTIES), choices=PENALTIES)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    main()
