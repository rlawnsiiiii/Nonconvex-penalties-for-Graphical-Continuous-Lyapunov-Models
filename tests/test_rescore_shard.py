"""simulations/rescore_shard.py (wave 6): the eBIC penalty inside the selection and the search,
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


def test_refit_mode_writes_a_complete_cell_with_the_likelihood_bic(tmp_path):
    """Wave 7: --refit loglik redoes the selection and the search with the maximised likelihood
    on the stored paths and writes a complete cell: the path fields are the source's, the
    selection fields are new, and --p keeps only the data sets of the given sizes."""
    root = tmp_path
    src = root / "direct_lasso_Cresc_n1e4"
    run([str(ROOT / "simulations" / "run_s1_shard.py"), "--shard", "0", "--n-shards", "4", "--p", "5", "6",
         "--reps", "1", "--n-obs", "10000", "--c-scale", "variance", "--select", "search",
         "--out-dir", str(src / "shards")])
    (src / "n_shards").write_text("4\n")
    out = root / "direct_lasso-ml_Cresc_n1e4" / "shards"
    run([str(ROOT / "simulations" / "rescore_shard.py"), "--shard", "0", "--n-shards", "4", "--n-obs", "1e4",
         "--source-cell", "direct_lasso_Cresc", "--refit", "loglik", "--select", "search", "--p", "5",
         "--out-dir", str(out)])
    (f,) = out.glob("shard_*.npz")
    r = np.load(f, allow_pickle=True)
    (g,) = (src / "shards").glob("shard_*.npz")
    s = np.load(g, allow_pickle=True)
    keep = [i for i in range(len(s["p"])) if int(s["p"][i]) == 5]
    assert list(r["p"]) == [5] * len(keep) and len(keep) > 0
    for key in ("k", "c_choice", "rep", "lambdas", "conf_offdiag", "nnz"):
        assert np.array_equal(r[key], s[key][keep]), key
    cfg = json.loads(str(r["config_json"]))
    assert (cfg["refit"], cfg["select"], cfg["p_values"]) == ("loglik", "search", [5])
    from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
    from gclm.solvers.search import Scorer
    from run_s1_shard import unpack_supports
    for j, i in enumerate(keep):
        rng = np.random.default_rng([cfg["seed"], 5, int(s["k"][i]), int(s["c_choice"][i]), int(s["rep"][i])])
        _, _, _, sigma_hat, scale = draw_instance(5, int(s["k"][i]), 10_000, list(CChoice)[int(s["c_choice"][i])],
                                                  rng, standardize=True, return_scale=True)
        scorer = Scorer(sigma_hat, estimation_volatility(scale, "variance"), 10_000, "loglik")
        supports = unpack_supports(r["supports_packed"][j], len(r["lambdas"][j]), 5)
        ib = int(r["bic_index"][j])
        assert np.isclose(scorer.score(supports[ib]), r["bic_scores"][j][ib], rtol=1e-8)
        assert r["search_score"][j] <= r["bic_scores"][j][ib] + 1e-9
    rows = [x for x in campaign.load(root) if x["cell"] == "direct_lasso-ml_Cresc_n1e4"]
    assert len(rows) == len(keep) and all(x["estimator"] == "lasso-ml" for x in rows)
