"""The thresholded lasso (plan of 13 October 2026, §1.3): the dense end of the lasso path cut by
size and refitted, as a path object that the metrics, the score and the shard runner read like
any other."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.data.simulate import CChoice, draw_instance, estimation_volatility, standardized_drift
from gclm.metrics import evaluate_path
from gclm.objective.direct import diagonal_fit
from gclm.objective.exact_fits import PRESENT, min_l1_exact_fit
from gclm.solvers.path import THRESHOLD_REFITS, ThresholdedLassoPath, adaptive_lasso_path, thresholded_lasso_path
from gclm.solvers.search import DirectRefit, bic_along_path


@pytest.fixture(scope="module")
def problem():
    m, _, _, r, s = draw_instance(7, 2, 2000, CChoice.ID, np.random.default_rng(4), standardize=True,
                                  return_scale=True)
    return standardized_drift(m, s), r, estimation_volatility(s, "variance")


def supports(path):
    p = path.estimates[0].shape[0]
    return [(m != 0) & ~np.eye(p, dtype=bool) for m in path.estimates]


def test_grid_supports_and_refits(problem):
    m_tilde, sigma, c = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    path = thresholded_lasso_path(sigma, c, n_thresholds=30)
    assert isinstance(path, ThresholdedLassoPath) and path.refit == "direct"
    assert np.array_equal(path.lambdas, path.thresholds) and len(path.thresholds) == 30
    assert path.thresholds[0] == 0.0 and path.thresholds[-1] == np.inf
    assert np.all(np.diff(path.thresholds) >= 0)
    size = np.abs(path.pilot[off])
    assert path.thresholds[1] == pytest.approx(size[size > PRESENT].min()) and path.thresholds[-2] == pytest.approx(size.max())
    sups = supports(path)
    assert np.array_equal(sups[0], (np.abs(path.pilot) > PRESENT) & off)   # threshold 0: the pilot's support
    assert np.array_equal(sups[0], (path.pilot != 0) & off)                 # (the lasso's has exact zeros)
    assert not sups[-1].any() and np.allclose(path.estimates[-1], diagonal_fit(sigma, c))
    assert all(np.all(b <= a) for a, b in zip(sups, sups[1:]))        # nested, monotone
    assert len({s.tobytes() for s in sups}) > 5                       # and the grid resolves the ranking
    assert np.array_equal(path.start, path.pilot) and np.all(path.converged) and np.all(path.iterations == 0)
    # every estimate is the least-squares refit of its support, diagonal free
    refit = DirectRefit(sigma, c)
    for s, m in zip(sups, path.estimates):
        np.testing.assert_allclose(m, refit.fit(s), atol=1e-10)
    # the pilot is the adaptive lasso's, so the two estimators are paired at the pilot
    assert np.array_equal(path.pilot, adaptive_lasso_path(sigma, c).pilot)


def test_refit_on_the_true_support_and_the_norefit_variant(problem):
    m_tilde, sigma, c = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    truth = (m_tilde != 0) & off
    path = thresholded_lasso_path(sigma, c, pilot=m_tilde, n_thresholds=10)
    np.testing.assert_allclose(path.estimates[0], DirectRefit(sigma, c).fit(truth), atol=1e-12)
    raw = thresholded_lasso_path(sigma, c, pilot=m_tilde, n_thresholds=10, refit="none")
    assert raw.refit == "none"
    assert all(np.array_equal(a, b) for a, b in zip(supports(path), supports(raw)))   # same supports ...
    for t, m in zip(raw.thresholds, raw.estimates):                   # ... with the pilot's own values
        kept = truth & (np.abs(m_tilde) >= t)
        assert np.array_equal(m[off], np.where(kept, m_tilde, 0.0)[off]) and np.array_equal(np.diag(m), np.diag(m_tilde))
    assert not np.array_equal(path.estimates[0], raw.estimates[0])
    assert THRESHOLD_REFITS == ("direct", "loglik", "none")
    ml = thresholded_lasso_path(sigma, c, pilot=m_tilde, n_thresholds=4, refit="loglik")
    assert ml.refit == "loglik" and np.all(ml.converged) and not np.allclose(ml.estimates[0], path.estimates[0])
    assert all(np.array_equal(a, b) for a, b in zip(supports(ml), supports(thresholded_lasso_path(
        sigma, c, pilot=m_tilde, n_thresholds=4))))


def test_the_path_object_works_with_the_metrics_and_the_score(problem):
    m_tilde, sigma, c = problem
    path = thresholded_lasso_path(sigma, c, n_thresholds=25)
    ev = evaluate_path(path.estimates, m_tilde)
    assert 0.0 <= ev["max_f1"] <= 1.0 and 0.0 <= ev["aupr"] <= 1.0
    idx, scores = bic_along_path(sigma, c, 2000, supports(path))
    assert len(scores) == 25 and 0 <= idx < 25 and np.isfinite(scores).all()
    # the ranking by |pilot| is the same with and without the refit: the same max_f1
    raw = thresholded_lasso_path(sigma, c, pilot=path.pilot, n_thresholds=25, refit="none")
    assert evaluate_path(raw.estimates, m_tilde)["max_f1"] == pytest.approx(ev["max_f1"])


def test_an_exact_fit_as_the_pilot_counts_entries_above_present(problem):
    """The minimum-l1 exact fit has structural zeros that the LP returns as dust (~1e-10); they are
    not part of the pilot's support, exactly as the shard runner's start diagnostics count it."""
    _, sigma, c = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    pilot = min_l1_exact_fit(sigma, c)
    dust = (np.abs(pilot) > 0) & (np.abs(pilot) <= PRESENT) & off
    assert dust.any() and (np.abs(pilot[off]) > PRESENT).sum() <= p * (p - 1) // 2   # a vertex: at most p(p-1)/2
    path = thresholded_lasso_path(sigma, c, pilot=pilot, n_thresholds=12)
    assert np.array_equal(supports(path)[0], (np.abs(pilot) > PRESENT) & off)
    strict = thresholded_lasso_path(sigma, c, pilot=pilot, n_thresholds=12, present=0.0)   # the dust counts
    assert supports(strict)[0].sum() > p * (p - 1) // 2 and strict.thresholds[1] < 1e-8


def test_degenerate_pilot_and_bad_inputs(problem):
    _, sigma, c = problem
    p = sigma.shape[0]
    empty = thresholded_lasso_path(sigma, c, pilot=np.diag(np.diag(sigma)) * -1.0, n_thresholds=5)
    assert all(not s.any() for s in supports(empty)) and np.isinf(empty.thresholds[1:]).all()
    with pytest.raises(ValueError):
        thresholded_lasso_path(sigma, c, refit="ridge")
    with pytest.raises(ValueError):
        thresholded_lasso_path(sigma, c, n_thresholds=1)
    with pytest.raises(ValueError):
        thresholded_lasso_path(sigma, c, pilot=np.zeros((p - 1, p - 1)))
