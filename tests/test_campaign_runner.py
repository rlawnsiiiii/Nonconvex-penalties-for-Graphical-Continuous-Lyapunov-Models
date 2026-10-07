"""The shard runner's options for the campaign of October 2026
(next_steps/051026/cluster_campaign_051026.md): the rescaled C, the new ways to
compute a path (``--direction up``, ``--method lla``, ``--method adaptive``) and
the selection outputs (``--select bic`` / ``search``).

The runner is exercised end to end, as the cluster calls it, on p = 5.  What is
checked is the bookkeeping: that the default output is unchanged, that the new
fields are consistent with each other and with the stored truth, and that
combinations that are not implemented are refused before anything runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.metrics import confusion, orientation_breakdown
from gclm.solvers.search import Scorer

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulations" / "run_s1_shard.py"
sys.path.insert(0, str(ROOT / "simulations"))

from run_s1_shard import ORIENT, unpack_supports  # noqa: E402

#: the fields of a shard before the campaign options existed
LEGACY = {"p", "k", "c_choice", "rep", "lambda_max", "lambdas", "conf_offdiag", "conf_incdiag",
          "nnz", "objective", "kkt", "iterations", "newton_steps", "n_true_edges",
          "m_true_i", "m_true_j", "m_true_v", "best_f1_index", "best_acc_index",
          "m_best_f1_i", "m_best_f1_j", "m_best_f1_v", "m_best_acc_i", "m_best_acc_j",
          "m_best_acc_v", "seconds", "config_json", "c_choice_names", "provenance_json"}
BIC = {"supports_packed", "scale", "bic_index", "bic_scores", "bic_evaluations", "bic_conf",
       "bic_orient", "select_seconds"}
SEARCH = {"m_search_support", "m_search_i", "m_search_j", "m_search_v", "search_conf",
          "search_orient", "search_score", "search_evaluations", "search_moves", "search_seconds"}
P = 5


def run(tmp_path, *extra, n_shards=16, name="out"):
    """One shard at p = 5 with one replicate: 16 datasets in all, so ``n_shards=16``
    gives a single dataset (k = 1, C_ID) and ``n_shards=8`` two (C_ID, C_Random_Min_Diag)."""
    out = tmp_path / name
    cmd = [sys.executable, str(RUNNER), "--shard", "0", "--n-shards", str(n_shards),
           "--p", str(P), "--reps", "1", "--out-dir", str(out), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        return res, None
    (f,) = out.glob("shard_*.npz")
    return res, np.load(f, allow_pickle=True)


def true_matrix(shard, i):
    m = np.zeros((P, P))
    m[shard["m_true_i"][i], shard["m_true_j"][i]] = shard["m_true_v"][i]
    return m


def test_default_output_is_what_it_was_before_the_campaign_options(tmp_path):
    res, shard = run(tmp_path)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY
    config = json.loads(str(shard["config_json"]))
    assert (config["method"], config["c_scale"], config["direction"], config["select"]) == \
        ("path", "identity", "down", "none")


def test_select_bic_adds_the_supports_and_a_consistent_bic_selection(tmp_path):
    res, shard = run(tmp_path, "--select", "bic", "--c-scale", "variance", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC
    assert json.loads(str(shard["config_json"]))["c_scale"] == "variance"
    for i in range(len(shard["p"])):
        n_lambda = len(shard["lambdas"][i])
        supports = unpack_supports(shard["supports_packed"][i], n_lambda, P)
        assert supports.shape == (n_lambda, P, P)
        assert not supports[:, np.arange(P), np.arange(P)].any()          # off-diagonal only
        assert np.array_equal(supports.sum(axis=(1, 2)), shard["nnz"][i])
        truth = true_matrix(shard, i)
        for lam_index in (0, n_lambda // 2, n_lambda - 1):               # same supports as the counts
            cf = confusion(supports[lam_index].astype(float), truth)
            assert tuple(shard["conf_offdiag"][i][lam_index]) == (cf.tp, cf.fp, cf.tn, cf.fn)
        ib, scores = int(shard["bic_index"][i]), shard["bic_scores"][i]
        assert scores.shape == (n_lambda,) and scores[ib] == np.min(scores) and np.isfinite(scores[ib])
        assert np.array_equal(shard["bic_conf"][i], shard["conf_offdiag"][i][ib])
        ob = orientation_breakdown(supports[ib].astype(float), truth)
        assert np.array_equal(shard["bic_orient"][i], [ob[key] for key in ORIENT])
        scale = shard["scale"][i]
        assert scale.shape == (P,) and np.all(scale > 0)
        assert 0 < shard["bic_evaluations"][i] <= n_lambda               # equal supports are scored once


def test_select_search_improves_the_bic_and_stores_the_searched_graph(tmp_path):
    res, shard = run(tmp_path, "--penalty", "MCP", "--direction", "up", "--c-scale", "variance",
                     "--select", "search", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | SEARCH
    config = json.loads(str(shard["config_json"]))
    assert (config["penalty"], config["direction"], config["select"]) == ("MCP", "up", "search")
    for i in range(len(shard["p"])):
        start_score = shard["bic_scores"][i][int(shard["bic_index"][i])]
        assert shard["search_score"][i] <= start_score + 1e-9             # a descent on the BIC
        support = unpack_supports(shard["m_search_support"][i], 1, P)[0]
        truth = true_matrix(shard, i)
        cf = confusion(support.astype(float), truth)
        assert tuple(shard["search_conf"][i]) == (cf.tp, cf.fp, cf.tn, cf.fn)
        ob = orientation_breakdown(support.astype(float), truth)
        assert np.array_equal(shard["search_orient"][i], [ob[key] for key in ORIENT])
        # the stored refit lives on the searched support plus the diagonal
        refit = np.zeros((P, P))
        refit[shard["m_search_i"][i], shard["m_search_j"][i]] = shard["m_search_v"][i]
        assert not np.any((refit != 0) & ~support & ~np.eye(P, dtype=bool))
        moved = int(shard["search_moves"][i].sum())
        assert (moved == 0) == np.array_equal(
            support, unpack_supports(shard["supports_packed"][i], len(shard["lambdas"][i]), P)[
                int(shard["bic_index"][i])])
        assert shard["search_evaluations"][i] >= shard["bic_evaluations"][i]


@pytest.mark.parametrize("extra, method, penalty", [
    (("--method", "lla", "--penalty", "MCP"), "lla", "MCP"),
    (("--method", "lla", "--penalty", "SCAD", "--c-scale", "variance"), "lla", "SCAD"),
    (("--method", "adaptive"), "adaptive", "lasso"),
])
def test_lla_and_adaptive_lasso_run_through_the_runner(tmp_path, extra, method, penalty):
    res, shard = run(tmp_path, *extra, "--select", "bic")
    assert res.returncode == 0, res.stderr
    config = json.loads(str(shard["config_json"]))
    assert (config["method"], config["penalty"]) == (method, penalty)
    assert shard["nnz"][0][-1] == 0 and shard["nnz"][0][0] > 0            # empty at lambda_max, dense end not
    assert np.all(np.isfinite(shard["objective"][0]))
    assert np.all(np.diff(shard["lambdas"][0]) > 0)


def test_adaptive_lasso_has_its_own_lambda_grid(tmp_path):
    _, lasso = run(tmp_path, name="lasso")
    _, adaptive = run(tmp_path, "--method", "adaptive", name="adaptive")
    assert not np.isclose(lasso["lambda_max"][0], adaptive["lambda_max"][0])
    assert np.isclose(adaptive["lambdas"][0][-1], adaptive["lambda_max"][0])
    assert np.isclose(adaptive["lambdas"][0][0], 1e-4 * adaptive["lambda_max"][0])


@pytest.mark.parametrize("extra, message", [
    (("--method", "lla"), "needs --penalty MCP or SCAD"),
    (("--method", "adaptive", "--penalty", "MCP"), "adaptive lasso"),
    (("--method", "lla", "--penalty", "MCP", "--loss", "loglik"), "direct loss"),
    (("--method", "lla", "--penalty", "MCP", "--convention", "ncvreg"), "textbook"),
])
def test_combinations_that_are_not_implemented_are_refused(tmp_path, extra, message):
    res, shard = run(tmp_path, *extra)
    assert res.returncode != 0 and shard is None
    assert message in res.stderr
    assert not list((tmp_path / "out").glob("*.npz"))


def dataset(shard, i, n_obs, c_scale):
    """Rebuild dataset ``i`` of the shard from its seed, as the runner does."""
    cfg = S1Config()
    p, k, c, rep = (int(shard[key][i]) for key in ("p", "k", "c_choice", "rep"))
    rng = np.random.default_rng([cfg.seed, p, k, c, rep])
    m_true, _, _, sigma_hat, scale = draw_instance(p, k, n_obs, list(CChoice)[c], rng,
                                                   standardize=True, return_scale=True)
    return m_true, sigma_hat, estimation_volatility(scale, c_scale)


def test_the_likelihood_refit_behind_the_bic_is_an_option(tmp_path):
    """Wave 5a: ``--refit loglik`` scores every support by the maximised Gaussian
    likelihood instead of the least-squares refit (the BIC proper).  Same fields;
    the stored scores are those of the likelihood scorer, the search descends on
    them, and the likelihood cannot be worse than least squares on a support."""
    res, shard = run(tmp_path, "--method", "adaptive", "--c-scale", "variance", "--n-obs", "10000",
                     "--select", "search", "--refit", "loglik", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | SEARCH
    config = json.loads(str(shard["config_json"]))
    assert (config["refit"], config["add_screen"], config["select"]) == ("loglik", None, "search")
    for i in range(len(shard["p"])):
        m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "variance")
        ib = int(shard["bic_index"][i])
        supports = unpack_supports(shard["supports_packed"][i], len(shard["lambdas"][i]), P)
        ml, ls = Scorer(sigma_hat, c_est, 10_000, "loglik"), Scorer(sigma_hat, c_est, 10_000, "direct")
        for l in (0, ib, len(supports) - 1):
            assert np.isclose(ml.score(supports[l]), shard["bic_scores"][i][l], rtol=1e-8)
            assert ml.score(supports[l]) <= ls.score(supports[l]) + 1e-6 * abs(ls.score(supports[l]))
        assert shard["search_score"][i] <= shard["bic_scores"][i][ib] + 1e-9
        searched = unpack_supports(shard["m_search_support"][i], 1, P)[0]
        assert np.isclose(ml.score(searched), shard["search_score"][i], rtol=1e-8)
    # the default is the least-squares refit, and the screening is recorded
    res, shard = run(tmp_path, "--method", "adaptive", "--select", "bic", n_shards=16, name="ls")
    assert json.loads(str(shard["config_json"]))["refit"] == "direct"
    res, shard = run(tmp_path, "--select", "search", "--refit", "loglik", "--add-screen", "3",
                     n_shards=16, name="screened")
    assert res.returncode == 0, res.stderr
    assert json.loads(str(shard["config_json"]))["add_screen"] == 3


def test_the_extended_bic_can_score_the_selection_and_the_search(tmp_path):
    """Wave 5c: ``--ebic-gamma 0.5 1`` selects and searches once more per gamma with
    Dettling's extended BIC (4 gamma |E| log p) on the same path.  The plain-BIC
    fields are unchanged; the extra ones carry the prefix ebic<gamma>_, their
    selection is the offline rule of campaign.py, and their search descends on the
    extended score."""
    res, shard = run(tmp_path, "--c-scale", "variance", "--n-obs", "10000", "--select", "search",
                     "--ebic-gamma", "0.5", "1", n_shards=8)
    assert res.returncode == 0, res.stderr
    tags = ("ebic05", "ebic1")
    extra = {f"{t}_{k}" for t in tags for k in BIC - {"supports_packed", "scale"} | SEARCH
             if not k.startswith("m_")} | {f"m_{t}_search_{k}" for t in tags for k in ("support", "i", "j", "v")}
    assert set(shard.files) == LEGACY | BIC | SEARCH | extra
    config = json.loads(str(shard["config_json"]))
    assert (config["ebic_gammas"], config["ebic_form"]) == ([0.5, 1.0], "dettling")
    _, plain = run(tmp_path, "--c-scale", "variance", "--n-obs", "10000", "--select", "search",
                   n_shards=8, name="plain")
    assert np.array_equal(plain["bic_index"], shard["bic_index"])
    assert np.array_equal(plain["search_conf"], shard["search_conf"])
    for i in range(len(shard["p"])):
        m_true, sigma_hat, c_est = dataset(shard, i, 10_000, "variance")
        supports = unpack_supports(shard["supports_packed"][i], len(shard["lambdas"][i]), P)
        edges = supports.sum(axis=(1, 2))
        for gamma, tag in zip((0.5, 1.0), tags):
            scores = shard["bic_scores"][i] + 4 * gamma * edges * np.log(P)       # the offline rule
            assert np.allclose(shard[f"{tag}_bic_scores"][i], scores)
            ib = int(shard[f"{tag}_bic_index"][i])
            assert np.isclose(scores[ib], scores.min())
            sc = Scorer(sigma_hat, c_est, 10_000, "direct", gamma, "dettling")
            searched = unpack_supports(shard[f"m_{tag}_search_support"][i], 1, P)[0]
            assert np.isclose(sc.score(searched), shard[f"{tag}_search_score"][i], rtol=1e-10)
            assert shard[f"{tag}_search_score"][i] <= scores[ib] + 1e-9
    res, _ = run(tmp_path, "--select", "bic", "--ebic-gamma", "0", n_shards=16, name="zero")
    assert res.returncode == 2
    res, _ = run(tmp_path, "--ebic-gamma", "1", n_shards=16, name="noselect")
    assert res.returncode == 2


def test_selection_is_recorded_for_the_log_likelihood_loss_too(tmp_path):
    """Wave 3: the covariance-loss paths get the same BIC selection (least-squares
    refit of each support), in both path orders and with the rescaled C."""
    res, shard = run(tmp_path, "--loss", "loglik", "--penalty", "MCP", "--direction", "up",
                     "--c-scale", "variance", "--select", "bic")
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC
    config = json.loads(str(shard["config_json"]))
    assert (config["loss"], config["direction"], config["c_scale"]) == ("loglik", "up", "variance")
    assert np.isfinite(shard["bic_scores"][0][int(shard["bic_index"][0])])
