"""simulations/rescore_shard.py (wave 6): the extended BIC inside the selection and the search,
computed from a cell's stored supports without recomputing the paths.  Checked against the
runner's own --ebic-gamma output on the same data sets, and overlaid by campaign.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import campaign  # noqa: E402

COMMON = ["--shard", "0", "--n-shards", "8", "--p", "5", "--reps", "1", "--n-obs", "10000",
          "--c-scale", "variance", "--select", "search"]


def run(cmd, cwd=ROOT):
    res = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, cwd=cwd)
    assert res.returncode == 0, res.stderr
    return res


def test_rescoring_reproduces_the_runner_and_is_overlaid(tmp_path):
    root = tmp_path
    src = root / "direct_lasso_Cresc_n1e4"
    run([str(ROOT / "simulations" / "run_s1_shard.py"), *COMMON, "--out-dir", str(src / "shards")])
    (src / "n_shards").write_text("8\n")                       # as the submit script writes it
    # the same cell run with the term inside, for comparison
    direct = root / "ref"
    run([str(ROOT / "simulations" / "run_s1_shard.py"), *COMMON, "--ebic-gamma", "1",
         "--out-dir", str(direct)])
    # the rescoring: one task per source shard, folder layout of the submit script
    out = root / "rescore1_direct_lasso_Cresc_n1e4" / "shards"
    res = run([str(ROOT / "simulations" / "rescore_shard.py"), "--shard", "0", "--n-shards", "8",
               "--n-obs", "1e4", "--source-cell", "direct_lasso_Cresc", "--ebic-gamma", "1",
               "--select", "search", "--out-dir", str(out)])
    assert "rescoring" in res.stdout
    (f,) = out.glob("shard_*.npz")
    assert f.name == "shard_0000_of_0008.npz"
    r = np.load(f, allow_pickle=True)
    (g,) = direct.glob("shard_*.npz")
    d = np.load(g, allow_pickle=True)
    assert bool(r["rebuilt_exact"].all())
    for key in ("p", "k", "c_choice", "rep", "ebic1_bic_index", "ebic1_search_conf", "ebic1_search_orient",
                "ebic1_search_moves"):
        assert np.array_equal(r[key], d[key]), key
    for i in range(len(r["p"])):
        assert np.allclose(r["ebic1_bic_scores"][i], d["ebic1_bic_scores"][i])
        assert np.isclose(r["ebic1_search_score"][i], d["ebic1_search_score"][i])
        assert np.array_equal(r["m_ebic1_search_support"][i], d["m_ebic1_search_support"][i])
    cfg = json.loads(str(r["config_json"]))
    assert (cfg["source_cell"], cfg["ebic_gammas"], cfg["select"]) == ("direct_lasso_Cresc", [1.0], "search")
    # campaign.py overlays the rescored search on the source cell's rows
    rows = campaign.load(root)
    rows = [x for x in rows if x["cell"] == "direct_lasso_Cresc_n1e4"]
    assert len(rows) == len(r["p"])
    for x in rows:
        assert {"ebic1_f1", "ebic1_search_f1", "ebic1_search_moves", "search_f1"} <= set(x)
    # a wrong shard count or a missing source is refused
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "rescore_shard.py"), "--shard", "0",
                          "--n-shards", "4", "--n-obs", "1e4", "--source-cell", "direct_lasso_Cresc",
                          "--out-dir", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 2 and "8 shards" in res.stderr
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "rescore_shard.py"), "--shard", "0",
                          "--n-shards", "8", "--n-obs", "inf", "--source-cell", "direct_lasso_Cresc",
                          "--out-dir", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 2 and "no such source cell" in res.stderr
