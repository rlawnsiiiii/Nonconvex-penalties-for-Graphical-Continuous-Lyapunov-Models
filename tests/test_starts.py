"""Any matrix as the dense start of a dense -> sparse path (plan of 13 October 2026, §1.2):
``lasso_path(start=)`` on the direct loss and ``covloss_path(start=)`` on the covariance losses,
``pilot=`` for the adaptive lassos, and the exact fits of gclm.objective.exact_fits as starts."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.data.simulate import CChoice, draw_instance, estimation_volatility, standardized_drift
from gclm.objective.covariance import dense_fit
from gclm.objective.direct import direct_grad, lambda_max
from gclm.objective.exact_fits import min_l1_exact_fit, min_l2_exact_fit, noisy_truth_start
from gclm.objective.penalties import penalty_weights, stationarity
from gclm.solvers.path import adaptive_covloss_path, adaptive_lasso_path, covloss_path, lambda_grid, lasso_path


@pytest.fixture(scope="module")
def problem():
    m, _, _, r, s = draw_instance(8, 2, 1000, CChoice.ID, np.random.default_rng(3), standardize=True,
                                  return_scale=True)
    c = estimation_volatility(s, "variance")
    return standardized_drift(m, s), r, c


def test_start_equal_to_the_lasso_dense_end_reproduces_the_current_path(problem):
    _, sigma, c = problem
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=20, ratio=1e-3)
    up = lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", tol=1e-8)
    dense = lasso_path(sigma, c, lambdas=lams, zero_tol=0.0, tol=1e-8).estimates[0]   # what the path computes itself
    assert np.array_equal(up.start, dense)
    same = lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", tol=1e-8, start=dense)
    assert all(np.array_equal(a, b) for a, b in zip(up.estimates, same.estimates))
    assert np.array_equal(same.lambdas, lams) and np.array_equal(same.start, dense)
    assert np.array_equal(same.iterations, up.iterations) and np.array_equal(same.converged, up.converged)
    # a sparse -> dense path has no start
    assert lasso_path(sigma, c, lambdas=lams, penalty="MCP", tol=1e-8).start is None


def test_another_start_changes_the_path_not_the_grid(problem):
    m_tilde, sigma, c = problem
    p = sigma.shape[0]
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=20, ratio=1e-3)
    w = penalty_weights(p)
    ref = lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", tol=1e-10)
    starts = {"l2": min_l2_exact_fit(sigma, c), "l1": min_l1_exact_fit(sigma, c), "exact": dense_fit(sigma, c),
              "truth": noisy_truth_start(m_tilde, 0.05, np.random.default_rng(0))}
    for name, start in starts.items():
        path = lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", tol=1e-10, start=start)
        assert np.array_equal(path.lambdas, lams) and np.array_equal(path.start, start), name
        assert path.start is not start                                # a copy, the caller's array untouched
        # every converged point is stationary; the fully dense starts (l2, exact, truth) hit the step cap
        # at a few lambdas in the flat valley of the dense end (docs/LIKELIHOOD.md §4.3), flagged as such
        for lam, m, ok in zip(lams[:-1], path.estimates[:-1], path.converged[:-1]):
            viol = stationarity(m, direct_grad(m, sigma, c), lam, w, "MCP")
            assert viol < (1e-6 if ok else 1e-4), (name, lam, viol, ok)
        assert (~path.converged).sum() <= 3, name
        assert np.all(path.estimates[-1][~np.eye(p, dtype=bool)] == 0)   # diagonal fit at lambda_max
        if name != "l1":        # the l1 fit is what the lasso's dense end approximates: the two paths may coincide
            assert any(not np.array_equal(a != 0, b != 0) for a, b in zip(ref.estimates, path.estimates)), name
    assert np.abs(starts["l1"] - ref.start).max() < 0.05 * np.abs(ref.start).max()   # and here they nearly do


def test_start_is_validated(problem):
    _, sigma, c = problem
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=5, ratio=1e-2)
    good = dense_fit(sigma, c)
    with pytest.raises(ValueError):
        lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", start=good[:4, :4])
    with pytest.raises(ValueError):
        lasso_path(sigma, c, lambdas=lams, penalty="MCP", direction="up", start=np.full_like(good, np.nan))
    with pytest.raises(ValueError):                                    # sparse -> dense: no start to give
        lasso_path(sigma, c, lambdas=lams, penalty="MCP", start=good)
    with pytest.raises(ValueError):                                    # the lasso has no order
        lasso_path(sigma, c, lambdas=lams, direction="up", start=good)
    with pytest.raises(ValueError):
        covloss_path(sigma, c, "loglik", n_lambda=4, ratio=1e-2, penalty="MCP", direction="up", start=good[:4, :4])
    with pytest.raises(ValueError):
        covloss_path(sigma, c, "loglik", n_lambda=4, ratio=1e-2, penalty="MCP", start=good)
    with pytest.raises(ValueError):
        covloss_path(sigma, c, "loglik", n_lambda=4, ratio=1e-2, penalty="MCP", direction="up", start="truth")


def test_covariance_loss_takes_a_matrix_start(problem):
    m_tilde, sigma, c = problem
    kw = dict(n_lambda=8, ratio=1e-2, penalty="MCP", direction="up")
    exact = covloss_path(sigma, c, "loglik", **kw)                     # start="exact"
    assert np.allclose(exact.start, dense_fit(sigma, c))
    same = covloss_path(sigma, c, "loglik", start=dense_fit(sigma, c), **kw)
    assert all(np.array_equal(a, b) for a, b in zip(exact.estimates, same.estimates))
    other = covloss_path(sigma, c, "loglik", start=min_l1_exact_fit(sigma, c), **kw)
    assert np.array_equal(other.lambdas, exact.lambdas)
    assert any(not np.array_equal(a != 0, b != 0) for a, b in zip(exact.estimates, other.estimates))
    assert covloss_path(sigma, c, "loglik", n_lambda=4, ratio=1e-2, penalty="MCP").start is None
    # the adaptive lassos take the same matrices as pilots
    pilot = min_l2_exact_fit(sigma, c)
    ada = adaptive_lasso_path(sigma, c, n_lambda=10, pilot=pilot)
    assert np.array_equal(ada.pilot, pilot) and np.all(ada.weights[~np.eye(8, dtype=bool)] < np.inf)
    ada_cov = adaptive_covloss_path(sigma, c, "loglik", n_lambda=6, ratio=1e-2, pilot=pilot)
    assert np.array_equal(ada_cov.pilot, pilot)
    noisy = noisy_truth_start(m_tilde, 0.0, np.random.default_rng(0))
    assert np.array_equal(adaptive_lasso_path(sigma, c, n_lambda=10, pilot=noisy).pilot, m_tilde)
