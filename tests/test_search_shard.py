"""The wave 2 runner of the campaign (simulations/run_search_shard.py): the greedy
search on the score without a penalised path -- from the empty graph and random graphs --
and the same search started from the true graph.  Run end to end on p = 5; the
search itself is tested in tests/test_search.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.metrics import confusion
from gclm.solvers.search import Scorer

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulations" / "run_search_shard.py"
sys.path.insert(0, str(ROOT / "simulations"))

from run_s1_shard import unpack_supports  # noqa: E402

P = 5


def run(tmp_path, *extra, name="out"):
    out = tmp_path / name
    cmd = [sys.executable, str(RUNNER), "--shard", "0", "--n-shards", "8", "--p", str(P),
           "--reps", "1", "--restarts", "3", "--out-dir", str(out), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    (f,) = out.glob("shard_*.npz")
    return np.load(f, allow_pickle=True)


def dataset(shard, i, n_obs, c_scale):
    """Rebuild dataset ``i`` of the shard from its seed, as the runner does."""
    cfg = S1Config()
    p, k, c, rep = (int(shard[key][i]) for key in ("p", "k", "c_choice", "rep"))
    rng = np.random.default_rng([cfg.seed, p, k, c, rep])
    m_true, _, _, sigma_hat, scale = draw_instance(p, k, n_obs, list(CChoice)[c], rng,
                                                   standardize=True, return_scale=True)
    return m_true, sigma_hat, estimation_volatility(scale, c_scale)


def test_both_searches_are_recorded_consistently(tmp_path):
    shard = run(tmp_path, "--n-obs", "10000", "--c-scale", "variance")
    config = json.loads(str(shard["config_json"]))
    assert (config["c_scale"], config["restarts"], config["methods"]) == ("variance", 3, ["pure", "truth"])
    assert len(shard["p"]) == 2
    for i in range(2):
        m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "variance")
        truth = (m_true != 0) & ~np.eye(P, dtype=bool)
        scorer = Scorer(sigma_hat, c_est, 10_000, "direct")
        for name in ("pure", "truth"):
            support = unpack_supports(shard[f"m_{name}_support"][i], 1, P)[0]
            cf = confusion(support.astype(float), m_true)
            assert tuple(shard[f"{name}_conf"][i]) == (cf.tp, cf.fp, cf.tn, cf.fn)
            # the stored score is that of the stored support under the same C
            assert np.isclose(scorer(support)[0], shard[f"{name}_score"][i], rtol=1e-10)
        # pure search: the best of the empty start and the random starts
        scores = shard["pure_scores"][i]
        assert scores.shape == (4,) and np.isclose(shard["pure_score"][i], scores.min())
        assert 0 <= shard["pure_exact_starts"][i] <= 4
        # search from the truth: a descent, and the start score is the truth's score
        assert np.isclose(scorer(truth)[0], shard["truth_start_score"][i], rtol=1e-10)
        assert shard["truth_score"][i] <= shard["truth_start_score"][i] + 1e-9
        stayed = shard["truth_moves"][i].sum() == 0
        assert stayed == np.array_equal(unpack_supports(shard["m_truth_support"][i], 1, P)[0], truth)


def test_random_starts_depend_on_the_dataset_only(tmp_path):
    """Same dataset, different shard layout and sample size bookkeeping: the pure
    search is reproducible because its random starts are seeded per dataset."""
    a = run(tmp_path, "--n-obs", "inf", name="a")
    b = run(tmp_path, "--n-obs", "inf", "--methods", "pure", name="b")
    assert np.array_equal(a["pure_scores"], b["pure_scores"])
    assert "truth_score" in a.files and "truth_score" not in b.files


def test_every_start_is_recorded_and_uniform_starts_are_an_option(tmp_path):
    """Wave 5b: the graph and the score every start ended at are stored (random starts
    first, the empty graph last), so that the best of the first r starts can be read
    off for any r; ``--starts uniform`` draws the starts with a fair coin per entry
    instead of sparsely.  The empty start does not depend on that choice."""
    sparse = run(tmp_path, "--n-obs", "10000", "--methods", "pure", name="s")
    uniform = run(tmp_path, "--n-obs", "10000", "--methods", "pure", "--starts", "uniform", name="u")
    assert json.loads(str(sparse["config_json"]))["starts"] == "sparse"
    assert json.loads(str(uniform["config_json"]))["starts"] == "uniform"
    for shard in (sparse, uniform):
        for i in range(2):
            m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "identity")
            scorer = Scorer(sigma_hat, c_est, 10_000, "direct")
            ends = unpack_supports(shard["m_pure_starts_support"][i], 4, P)
            scores = shard["pure_scores"][i]
            assert ends.shape == (4, P, P) and shard["pure_start_moves"][i].shape == (4,)
            for s, score in zip(ends, scores):             # every stored score is that of its end
                assert np.isclose(scorer(s)[0], score, rtol=1e-10)
            best = unpack_supports(shard["m_pure_support"][i], 1, P)[0]
            assert np.array_equal(ends[int(np.argmin(scores))], best)
            # the search from the empty graph (last) can only add at first, so it made moves
            assert shard["pure_start_moves"][i][-1] > 0 or not ends[-1].any()
    assert np.allclose(sparse["pure_scores"][:, -1], uniform["pure_scores"][:, -1])


def test_the_extended_bic_scores_both_searches(tmp_path):
    """Wave 5c: ``--ebic-gamma 1`` adds Dettling's eBIC term to the score of the pure
    search and of the search from the truth."""
    shard = run(tmp_path, "--n-obs", "10000", "--c-scale", "variance", "--ebic-gamma", "1")
    config = json.loads(str(shard["config_json"]))
    assert (config["ebic_gamma"], config["ebic_form"]) == (1.0, "dettling")
    for i in range(2):
        m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "variance")
        truth = (m_true != 0) & ~np.eye(P, dtype=bool)
        scorer = Scorer(sigma_hat, c_est, 10_000, "direct", 1.0, "dettling")
        for name in ("pure", "truth"):
            support = unpack_supports(shard[f"m_{name}_support"][i], 1, P)[0]
            assert np.isclose(scorer(support)[0], shard[f"{name}_score"][i], rtol=1e-10)
        assert np.isclose(scorer(truth)[0], shard["truth_start_score"][i], rtol=1e-10)


def test_population_covariance_with_the_right_c_keeps_the_truth(tmp_path):
    """n = inf, data generated with C = 2I (the first dataset of the shard is C_ID),
    scored with the rescaled C: the model is exact, so the truth cannot be improved
    on by a single move and the search started there stays."""
    shard = run(tmp_path, "--n-obs", "inf", "--c-scale", "variance", "--methods", "truth")
    assert str(shard["c_choice_names"][int(shard["c_choice"][0])]) == "C_ID"
    assert shard["truth_moves"][0].sum() == 0
    assert tuple(shard["truth_conf"][0][[1, 3]]) == (0, 0)             # no false positives or negatives


def test_the_likelihood_refit_scores_both_searches(tmp_path):
    """Wave 7: --refit loglik fits each graph by maximising the Gaussian likelihood (the
    likelihood refit); the stored scores are those of that scorer."""
    shard = run(tmp_path, "--n-obs", "10000", "--refit", "loglik", "--add-screen", "8")
    cfg = json.loads(str(shard["config_json"]))
    assert (cfg["refit"], cfg["add_screen"]) == ("loglik", 8)
    for i in range(2):
        m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "identity")
        scorer = Scorer(sigma_hat, c_est, 10_000, "loglik")
        for name in ("pure", "truth"):
            support = unpack_supports(shard[f"m_{name}_support"][i], 1, P)[0]
            assert np.isclose(scorer(support)[0], shard[f"{name}_score"][i], rtol=1e-8)
