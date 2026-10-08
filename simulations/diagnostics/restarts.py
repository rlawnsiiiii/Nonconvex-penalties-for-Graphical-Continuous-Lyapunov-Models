#!/usr/bin/env python3
"""The restart study (wave 5b of the campaign, next_steps/051026/cluster_campaign_051026.md
Section 3.4): how many randomly drawn starting graphs does the pure greedy BIC search need?

Reads the cells of ``run_search_shard.py`` (``<root>/search*_p<p>_<C>_n<n>/shards``; the wave 2
cells with 10 starts and the wave 5 cells with 100) and, for every r on a grid up to the number
of random starts R of the cell, takes the best-scoring result of the FIRST r random starts --
without and with the empty graph added, the latter being the estimator exactly as wave 2 ran it
when r = 10.  Per cell and r it reports

  score_reached   share of graphs whose best BIC over all R + 1 starts is already reached
                  within the first r random starts (the saturation check of S4 Section 4a)
  f1_mean, f1_se  mean directed F1 of the graph that best-of-r ends at (needs the per-start
                  supports, stored from wave 5 on; wave 2 cells give the score column only)
  empty_dead      share of graphs whose empty start has BIC = +inf (least-squares refit of the
                  diagonal support unstable; it happens with the rescaled C only)

Writes ``<root>/campaign_restarts.csv`` (long format) and prints one table per cell.  Numbers
only; the figure is drawn by plot_campaign.py.

    python simulations/diagnostics/restarts.py                 # runs/campaign
    python simulations/diagnostics/restarts.py --root some/dir --grid 1 3 10
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))

from gclm.metrics import confusion  # noqa: E402
from run_s1_shard import unpack_supports  # noqa: E402

CELL = re.compile(r"^search(?P<variant>[0-9]+[su])?_p(?P<p>[0-9]+)_(?P<c>C2I|Cresc)_n(?P<n>[0-9e]+|inf)$")
GRID = (1, 2, 3, 5, 10, 20, 30, 50, 100)
COLUMNS = ("cell", "p", "c", "n", "starts", "R", "r", "with_empty", "n_graphs", "f1_mean", "f1_se",
           "score_reached", "empty_dead")


def f1_of(support: np.ndarray, m_true: np.ndarray) -> float:
    cf = confusion(support.astype(float), m_true)
    return 2 * cf.tp / max(1, 2 * cf.tp + cf.fp + cf.fn)


def load_cell(folder: Path) -> list[dict]:
    """One dict per graph: the final score of every start (random ones first, the empty
    graph last), the final supports if stored, and the true matrix."""
    graphs, starts, restarts = [], None, None
    for f in sorted((folder / "shards").glob("shard_*.npz")):
        d = np.load(f, allow_pickle=True)
        import json
        cfg = json.loads(str(d["config_json"]))
        starts, restarts = cfg.get("starts", "sparse"), int(cfg["restarts"])
        for i in range(len(d["p"])):
            p = int(d["p"][i])
            m_true = np.zeros((p, p))
            m_true[d["m_true_i"][i], d["m_true_j"][i]] = d["m_true_v"][i]
            g = {"p": p, "scores": np.asarray(d["pure_scores"][i], float), "m_true": m_true}
            if "m_pure_starts_support" in d.files:
                g["ends"] = unpack_supports(d["m_pure_starts_support"][i], restarts + 1, p)
            graphs.append(g)
    return graphs, starts, restarts


def best_of(scores: np.ndarray, r: int, with_empty: bool) -> int:
    """Index of the best-scoring start among the first r random starts (and the empty
    graph, which is the last entry, if asked)."""
    idx = list(range(r)) + ([len(scores) - 1] if with_empty else [])
    return idx[int(np.argmin(scores[idx]))]


def curves(graphs: list[dict], restarts: int, grid=GRID) -> list[dict]:
    rows = []
    rs = [r for r in grid if r <= restarts] + ([restarts] if restarts not in grid else [])
    dead = float(np.mean([np.isinf(g["scores"][-1]) for g in graphs]))
    for r in sorted(set(rs)):
        for with_empty in (False, True):
            reached, f1s = [], []
            for g in graphs:
                b = best_of(g["scores"], r, with_empty)
                reached.append(g["scores"][b] <= g["scores"].min() + 1e-9 * max(1.0, abs(g["scores"].min())))
                if "ends" in g:
                    f1s.append(f1_of(g["ends"][b], g["m_true"]))
            f1 = np.asarray(f1s, float)
            rows.append({"r": r, "with_empty": int(with_empty), "n_graphs": len(graphs),
                         "f1_mean": float(f1.mean()) if len(f1) else float("nan"),
                         "f1_se": float(f1.std(ddof=1) / np.sqrt(len(f1))) if len(f1) > 1 else float("nan"),
                         "score_reached": float(np.mean(reached)), "empty_dead": dead})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    args = ap.parse_args()
    out = []
    for folder in sorted(p for p in args.root.iterdir() if p.is_dir()):
        m = CELL.match(folder.name)
        if not m or not (folder / "shards").is_dir():
            continue
        graphs, starts, restarts = load_cell(folder)
        if not graphs:
            continue
        rows = curves(graphs, restarts, args.grid)
        meta = {"cell": folder.name, "p": int(m["p"]), "c": m["c"], "n": m["n"], "starts": starts,
                "R": restarts}
        out += [{**meta, **row} for row in rows]
        print(f"\n{folder.name}: {len(graphs)} graphs, R = {restarts} {starts} starts, "
              f"empty start dead in {rows[0]['empty_dead']:.2f}")
        print(f"  {'r':>4s} {'reached':>8s} {'F1 best-of-r':>13s} {'F1 +empty':>10s}")
        for r in sorted({row["r"] for row in rows}):
            a = next(x for x in rows if x["r"] == r and not x["with_empty"])
            b = next(x for x in rows if x["r"] == r and x["with_empty"])
            print(f"  {r:4d} {a['score_reached']:8.2f} {a['f1_mean']:13.3f} {b['f1_mean']:10.3f}")
    if out:
        path = args.root / "campaign_restarts.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
