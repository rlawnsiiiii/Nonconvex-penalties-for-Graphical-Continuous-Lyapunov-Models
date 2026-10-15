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

#: the fields of a shard before the campaign options existed, plus ``converged`` (the solver's
#: flag per lambda, stored for every loss since 13 October 2026, code review A4)
LEGACY = {"p", "k", "c_choice", "rep", "lambda_max", "lambdas", "conf_offdiag", "conf_incdiag",
          "nnz", "objective", "kkt", "iterations", "newton_steps", "converged", "n_true_edges",
          "m_true_i", "m_true_j", "m_true_v", "best_f1_index", "best_acc_index",
          "m_best_f1_i", "m_best_f1_j", "m_best_f1_v", "m_best_acc_i", "m_best_acc_j",
          "m_best_acc_v", "seconds", "config_json", "c_choice_names", "provenance_json"}
BIC = {"supports_packed", "scale", "bic_index", "bic_scores", "bic_evaluations", "bic_conf",
       "bic_orient", "select_seconds", "bic_dead", "bic_unconverged"}
SEARCH = {"m_search_support", "m_search_i", "m_search_j", "m_search_v", "search_conf",
          "search_orient", "search_score", "search_evaluations", "search_moves", "search_seconds",
          "search_dead", "search_unconverged", "search_capped", "search_mismatches"}
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
    assert set(shard.files) == LEGACY | BIC | SEARCH | START     # a dense start: its diagnostics too
    config = json.loads(str(shard["config_json"]))
    assert (config["penalty"], config["direction"], config["select"]) == ("MCP", "up", "search")
    for i in range(len(shard["p"])):
        start_score = shard["bic_scores"][i][int(shard["bic_index"][i])]
        assert shard["search_score"][i] <= start_score + 1e-9             # a descent on the score
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
    likelihood (the likelihood refit) instead of the least-squares refit.  Same fields;
    the stored scores are those of the likelihood scorer, the search descends on
    them, and the likelihood cannot be worse than least squares on a support."""
    res, shard = run(tmp_path, "--method", "adaptive", "--c-scale", "variance", "--n-obs", "10000",
                     "--select", "search", "--refit", "loglik", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | SEARCH | START
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
    Dettling's eBIC penalty (eBIC term 4 gamma |E| log p) on the same path.  The fields
    with the BIC penalty are unchanged; the extra ones carry the prefix ebic<gamma>_, their
    selection is the offline rule of campaign.py, and their search descends on the
    score with the eBIC penalty."""
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
    """Wave 3: the covariance-loss paths get the same selection by the score
    (least-squares refit of each support), in both path orders and with the rescaled C."""
    res, shard = run(tmp_path, "--loss", "loglik", "--penalty", "MCP", "--direction", "up",
                     "--c-scale", "variance", "--select", "bic")
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | START
    config = json.loads(str(shard["config_json"]))
    assert (config["loss"], config["direction"], config["c_scale"]) == ("loglik", "up", "variance")
    assert np.isfinite(shard["bic_scores"][0][int(shard["bic_index"][0])])


def test_log_likelihood_adaptive_lasso_and_lasso_start_run_through_the_runner(tmp_path):
    """Wave 7: the adaptive lasso and the dense -> sparse MCP path started from the lasso solution,
    both on the log-likelihood loss, with the selection by the score and the search under the
    likelihood refit."""
    res, ada = run(tmp_path, "--loss", "loglik", "--method", "adaptive", "--select", "search",
                   "--refit", "loglik", "--n-obs", "10000", name="ada")
    assert res.returncode == 0, res.stderr
    cfg = json.loads(str(ada["config_json"]))
    assert (cfg["loss"], cfg["method"], cfg["refit"], cfg["up_start"]) == ("loglik", "adaptive", "loglik", "exact")
    assert set(ada.files) == LEGACY | BIC | SEARCH | START
    assert ada["search_score"][0] <= ada["bic_scores"][0][int(ada["bic_index"][0])] + 1e-9
    res, up = run(tmp_path, "--loss", "loglik", "--penalty", "MCP", "--direction", "up", "--up-start", "lasso",
                  "--select", "bic", "--refit", "loglik", "--n-obs", "10000", name="up")
    assert res.returncode == 0, res.stderr
    assert json.loads(str(up["config_json"]))["up_start"] == "lasso"
    # the start needs a dense -> sparse path: refused without --direction up, and for the lasso (no order)
    for extra in (("--loss", "loglik", "--up-start", "lasso", "--penalty", "MCP"),
                  ("--up-start", "l1", "--penalty", "MCP"),
                  ("--up-start", "l1", "--direction", "up")):
        res, _ = run(tmp_path, *extra, name="bad")
        assert res.returncode != 0 and "up_start" in res.stderr


def test_the_direct_loss_stores_solver_diagnostics_and_provenance(tmp_path):
    """Code review A4 / C3 / C4 (13 October 2026): FISTA's step counts and convergence flag per
    lambda and the first-order residual are stored for the direct loss too, the provenance
    names the git commit, no selection is dead or unconverged here, and no temporary file is
    left behind by the atomic write."""
    res, shard = run(tmp_path, "--select", "bic")
    assert res.returncode == 0, res.stderr
    assert shard["converged"].all() and shard["iterations"].min() >= 0
    assert shard["iterations"][0][-1] == 0                    # lambda_max: the closed form, no steps
    assert shard["iterations"][0][0] > 0
    assert np.all(np.isfinite(shard["kkt"])) and shard["kkt"].max() < 1e-4
    assert np.all(shard["newton_steps"] == -1)
    assert not shard["bic_dead"].any() and np.all(shard["bic_unconverged"] == 0)
    prov = json.loads(str(shard["provenance_json"]))
    assert {"commit", "dirty", "host", "numpy"} <= set(prov)
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_more_shards_than_tasks_gives_a_readable_empty_shard(tmp_path):
    """Code review C5: a shard with no data set still has the identifier columns, so the
    analysis reads it instead of failing on a missing ``p``."""
    out = tmp_path / "cell" / "shards"
    cmd = [sys.executable, str(RUNNER), "--shard", "63", "--n-shards", "64", "--p", str(P), "--reps", "1",
           "--out-dir", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    (f,) = out.glob("shard_*.npz")
    d = np.load(f, allow_pickle=True)
    assert len(d["p"]) == 0 and len(d["rep"]) == 0 and "config_json" in d.files


@pytest.mark.parametrize("extra, message", [
    (("--reps", "0"), "--reps"),
    (("--tol", "0"), "--tol"),
    (("--method", "adaptive", "--direction", "up"), "--direction"),
    (("--loss", "loglik", "--convention", "ncvreg"), "--convention"),
    (("--loss", "loglik", "--solver", "skglm"), "--solver"),
    (("--refit", "loglik"), "--refit"),
    (("--select", "bic", "--refit", "loglik", "--add-screen", "3"), "--add-screen"),
    (("--select", "search", "--add-screen", "3"), "--add-screen"),
    (("--select", "bic", "--ebic-gamma", "15"), "one decimal"),
    (("--select", "bic", "--ebic-gamma", "1.5", "1.5"), "distinct"),
])
def test_settings_without_effect_or_out_of_range_are_refused(tmp_path, extra, message):
    """Code review C5: a setting that would be recorded in config_json without having any effect,
    or a value that silently fell back to the default, is refused before anything runs."""
    res, shard = run(tmp_path, *extra)
    assert res.returncode != 0 and shard is None
    assert message in res.stderr, res.stderr


#: the per-data-set diagnostics of a dense start (plan of 13 October 2026, §2.2)
START = {"start_l1", "start_l2", "start_residual", "start_nnz", "start_conf", "start_orient"}


def dataset_std(shard, i, n_obs, c_scale):
    """The standardised truth, Sigma_hat and the estimation C of data set ``i``."""
    from gclm.data.simulate import standardized_drift
    cfg = S1Config()
    p, k, c, rep = (int(shard[key][i]) for key in ("p", "k", "c_choice", "rep"))
    rng = np.random.default_rng([cfg.seed, p, k, c, rep])
    m_true, _, _, sigma_hat, scale = draw_instance(p, k, n_obs, list(CChoice)[c], rng,
                                                   standardize=True, return_scale=True)
    return standardized_drift(m_true, scale), sigma_hat, estimation_volatility(scale, c_scale)


@pytest.mark.parametrize("start", ["lasso", "exact", "l1", "l2", "truth-0.1"])
def test_any_dense_start_runs_the_mcp_path_and_records_itself(tmp_path, start):
    """E1: ``--up-start`` on the direct loss.  The shard stores the start's diagnostics; the explicit
    ``lasso`` is the old default and stores the same path as a run without the option."""
    res, shard = run(tmp_path, "--penalty", "MCP", "--direction", "up", "--up-start", start,
                     "--select", "bic", "--c-scale", "variance", "--n-obs", "inf", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | START
    cfg = json.loads(str(shard["config_json"]))
    assert cfg["up_start"] == start and cfg["pilot"] == "lasso"
    assert (cfg["weights"], cfg["w_min"], cfg["graph"]) == ("normal", 0.0, "random")
    for i in range(len(shard["p"])):
        m_tilde, sigma_hat, c_est = dataset_std(shard, i, np.inf, "variance")
        conf, orient = shard["start_conf"][i], shard["start_orient"][i]
        assert conf.sum() == P * (P - 1) and conf.shape == (4,) and orient.shape == (len(ORIENT),)
        assert shard["start_nnz"][i] == conf[0] + conf[1]                    # tp + fp = present entries
        assert shard["start_l1"][i] >= shard["start_l2"][i] > 0
        if start in ("exact", "l1", "l2"):
            assert shard["start_residual"][i] < 1e-8                           # exact fits
        if start == "truth-0.1":
            assert shard["start_residual"][i] > 1e-3 and shard["start_nnz"][i] == P * (P - 1)
        if start == "l1":                                                      # the LP's value is the l1 norm
            from gclm.objective.exact_fits import min_l1_exact_fit
            _, info = min_l1_exact_fit(sigma_hat, c_est, return_info=True)
            assert shard["start_l1"][i] == pytest.approx(info.value, rel=1e-9)
    if start == "lasso":
        res, plain = run(tmp_path, "--penalty", "MCP", "--direction", "up", "--select", "bic",
                         "--c-scale", "variance", "--n-obs", "inf", n_shards=8, name="plain")
        assert res.returncode == 0, res.stderr
        for key in ("conf_offdiag", "bic_index", "nnz"):
            assert np.array_equal(plain[key], shard[key]), key
        assert np.array_equal(plain["start_l1"], shard["start_l1"])


def test_truth_start_at_tau_zero_is_the_standardised_truth(tmp_path):
    res, shard = run(tmp_path, "--penalty", "MCP", "--direction", "up", "--up-start", "truth-0",
                     "--c-scale", "variance", "--n-obs", "inf", n_shards=8)
    assert res.returncode == 0, res.stderr
    for i in range(len(shard["p"])):
        m_tilde, _, _ = dataset_std(shard, i, np.inf, "variance")
        conf = shard["start_conf"][i]
        assert conf[1] == 0 and conf[3] == 0 and conf[0] == int(shard["n_true_edges"][i])   # no fp, no fn
        assert shard["start_residual"][i] < 1e-8                                            # exact at n = inf
        assert shard["start_l1"][i] == pytest.approx(np.abs(m_tilde[~np.eye(P, dtype=bool)]).sum())


@pytest.mark.parametrize("method, pilot", [("adaptive", "l2"), ("threshold", "lasso"), ("threshold", "l1"),
                                           ("threshold-norefit", "lasso")])
def test_pilot_methods_run_and_record_the_pilot(tmp_path, method, pilot):
    """E2: the thresholded lasso as a path through the runner, and ``--pilot`` for the adaptive and
    thresholded lassos; the pilot's diagnostics are stored like a start's."""
    res, shard = run(tmp_path, "--method", method, "--pilot", pilot, "--select", "search",
                     "--c-scale", "variance", "--n-obs", "10000", n_shards=8)
    assert res.returncode == 0, res.stderr
    assert set(shard.files) == LEGACY | BIC | SEARCH | START
    cfg = json.loads(str(shard["config_json"]))
    assert cfg["method"] == method and cfg["pilot"] == pilot and cfg["up_start"] == "lasso"
    for i in range(len(shard["p"])):
        lams, nnz = shard["lambdas"][i], shard["nnz"][i]
        assert len(lams) == 100 and np.all(np.diff(lams) >= 0)
        if method.startswith("threshold"):
            assert lams[0] == 0.0 and lams[-1] == np.inf and shard["lambda_max"][i] == lams[-2]
            assert np.all(np.diff(nnz) <= 0) and nnz[-1] == 0 and nnz[0] == shard["start_nnz"][i]
            assert np.isnan(shard["kkt"][i]).all() and np.all(shard["iterations"][i] == 0)
            assert np.all(shard["converged"][i])
            sup = unpack_supports(shard["supports_packed"][i], 100, P)
            assert all(np.all(b <= a) for a, b in zip(sup, sup[1:]))        # nested supports
        assert shard["search_score"][i] <= shard["bic_scores"][i][int(shard["bic_index"][i])] + 1e-9
    if pilot == "lasso" and method == "threshold":
        # the same pilot as the adaptive lasso: identical start diagnostics
        res, ada = run(tmp_path, "--method", "adaptive", "--select", "bic", "--c-scale", "variance",
                       "--n-obs", "10000", n_shards=8, name="ada")
        assert res.returncode == 0, res.stderr
        for key in ("start_l1", "start_nnz", "start_conf"):
            assert np.array_equal(ada[key], shard[key]), key


def test_generator_knobs_run_and_are_recorded(tmp_path):
    """E3 / E4: ``--weights uniform --w-min`` and ``--graph`` change the truth and are recorded; the
    default run is bit-identical to one without the options."""
    from gclm.graphs import is_dag
    res, uni = run(tmp_path, "--weights", "uniform", "--w-min", "0.25", "--n-obs", "1000", n_shards=4, name="uni")
    assert res.returncode == 0, res.stderr
    cfg = json.loads(str(uni["config_json"]))
    assert (cfg["weights"], cfg["w_min"], cfg["graph"]) == ("uniform", 0.25, "random")
    for i in range(len(uni["p"])):
        w = np.abs(uni["m_true_v"][i][uni["m_true_i"][i] != uni["m_true_j"][i]])
        assert w.size == 0 or (w.min() >= 0.25 and w.max() <= 1.0)
    res, dag = run(tmp_path, "--graph", "dag", "--n-obs", "1000", n_shards=4, name="dag")
    assert res.returncode == 0, res.stderr
    assert json.loads(str(dag["config_json"]))["graph"] == "dag"
    assert all(is_dag(true_matrix(dag, i) != 0) for i in range(len(dag["p"])))
    res, plain = run(tmp_path, "--n-obs", "1000", n_shards=4, name="plain")
    res2, explicit = run(tmp_path, "--weights", "normal", "--graph", "random", "--n-obs", "1000", n_shards=4,
                         name="explicit")
    assert res.returncode == 0 and res2.returncode == 0
    for key in ("m_true_v", "conf_offdiag"):
        assert all(np.array_equal(a, b) for a, b in zip(plain[key], explicit[key])), key
    assert not np.array_equal(plain["n_true_edges"], dag["n_true_edges"]) or \
        not all(np.array_equal(a, b) for a, b in zip(plain["m_true_v"], dag["m_true_v"]))


@pytest.mark.parametrize("extra, message", [
    (("--pilot", "l1"), "--pilot"),
    (("--method", "lla", "--penalty", "MCP", "--pilot", "l1"), "--pilot"),
    (("--w-min", "0.2"), "--w-min"),
    (("--method", "threshold", "--penalty", "MCP"), "thresholded lasso"),
    (("--method", "threshold", "--loss", "loglik"), "direct loss"),
    (("--up-start", "truth-x", "--penalty", "MCP", "--direction", "up"), "truth-<tau>"),
    (("--up-start", "truth--1", "--penalty", "MCP", "--direction", "up"), "truth-<tau>"),
    (("--pilot", "ridge", "--method", "adaptive"), "truth-<tau>"),
])
def test_start_and_generator_settings_without_effect_are_refused(tmp_path, extra, message):
    res, shard = run(tmp_path, *extra)
    assert res.returncode != 0 and shard is None
    assert message in res.stderr, res.stderr
