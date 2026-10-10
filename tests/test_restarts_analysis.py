"""simulations/diagnostics/restarts.py: best-of-the-first-r starting graphs, read off the
per-start results of run_search_shard.py (wave 5b)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))
sys.path.insert(0, str(ROOT / "simulations"))

import restarts  # noqa: E402


def test_curves_match_the_stored_best_and_are_monotone(tmp_path):
    cell = tmp_path / "search3u_p5_C2I_n1e4"
    cmd = [sys.executable, str(ROOT / "simulations" / "run_search_shard.py"), "--shard", "0",
           "--n-shards", "4", "--p", "5", "--reps", "1", "--restarts", "3", "--starts", "uniform",
           "--methods", "pure", "--n-obs", "10000", "--out-dir", str(cell / "shards")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "restarts.py"),
                          "--root", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    rows = list(csv.DictReader(open(tmp_path / "campaign_restarts.csv")))
    assert {r["cell"] for r in rows} == {"search3u_p5_C2I_n1e4"}
    assert sorted({int(r["r"]) for r in rows}) == [1, 2, 3] and all(r["starts"] == "uniform" for r in rows)
    # with all R random starts and the empty graph, best-of-r is the stored pure result
    (f,) = (cell / "shards").glob("shard_*.npz")
    d = np.load(f, allow_pickle=True)
    f1 = [2 * c[0] / max(1, 2 * c[0] + c[1] + c[3]) for c in d["pure_conf"]]
    full = next(r for r in rows if int(r["r"]) == 3 and r["with_empty"] == "1")
    assert np.isclose(float(full["f1_mean"]), np.mean(f1)) and float(full["score_reached"]) == 1.0
    assert int(full["n_graphs"]) == len(d["p"])
    # the share that reaches the best score cannot fall as r grows, nor when the empty graph is added
    for with_empty in ("0", "1"):
        reached = [float(r["score_reached"]) for r in sorted(
            (r for r in rows if r["with_empty"] == with_empty), key=lambda r: int(r["r"]))]
        assert reached == sorted(reached)
    for r in (1, 2, 3):
        a = next(x for x in rows if int(x["r"]) == r and x["with_empty"] == "0")
        b = next(x for x in rows if int(x["r"]) == r and x["with_empty"] == "1")
        assert float(a["score_reached"]) <= float(b["score_reached"])


def test_start_blocks_give_the_same_curves_and_table_rows(tmp_path):
    """A cell run with --start-blocks 2 (two tasks per graph) is put back together by
    restarts.py and campaign.py: the same curves and the same rows as the unsplit cell."""
    import campaign
    for name, shards, extra in (("search4s_p5_C2I_n1e4", 1, []),
                                ("search4sb_p5_C2I_n1e4", 2, ["--start-blocks", "2"])):
        for s in range(shards):
            cmd = [sys.executable, str(ROOT / "simulations" / "run_search_shard.py"), "--shard", str(s),
                   "--n-shards", str(shards), "--p", "5", "--reps", "1", "--restarts", "4",
                   "--n-obs", "10000", "--out-dir", str(tmp_path / name / "shards"), *extra]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            assert res.returncode == 0, res.stderr
    whole, _, r_whole = restarts.load_cell(tmp_path / "search4s_p5_C2I_n1e4")
    split, _, r_split = restarts.load_cell(tmp_path / "search4sb_p5_C2I_n1e4")
    assert r_whole == r_split == 4 and len(whole) == len(split) == 16
    for a, b in zip(whole, split):
        assert np.array_equal(a["scores"], b["scores"]) and np.array_equal(a["ends"], b["ends"])
    assert restarts.curves(whole, 4) == restarts.curves(split, 4)
    meta = {"loss": "direct", "c": "C2I", "n": "1e4"}
    key = lambda r: (r["estimator"], r["p"], r["k"], r["c_choice"], r["rep"])
    rows_whole = sorted(campaign.load_search_cell(tmp_path / "search4s_p5_C2I_n1e4", meta), key=key)
    rows_split = sorted(campaign.load_search_cell(tmp_path / "search4sb_p5_C2I_n1e4", meta), key=key)
    assert len(rows_whole) == len(rows_split) == 32                  # pure and truth per graph
    for a, b in zip(rows_whole, rows_split):
        assert key(a) == key(b)
        assert {k: v for k, v in a.items() if k.startswith("search_") and k != "search_moves"} == \
               {k: v for k, v in b.items() if k.startswith("search_") and k != "search_moves"}
