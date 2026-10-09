"""The covariance-loss estimators added for wave 7 of the campaign (October 2026):
dense -> sparse paths started from the lasso solution of the same loss
(covloss_path(direction="up", start="lasso")) and the adaptive lasso on a covariance loss
(adaptive_covloss_path).  Checked against the lasso path they are built from and against
the first-order conditions."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.data.simulate import CChoice, draw_instance
from gclm.objective import covariance as cov
from gclm.objective.penalties import stationarity
from gclm.solvers.path import EXCLUDED_WEIGHT, adaptive_covloss_path, covloss_path

P = 6


@pytest.fixture(scope="module")
def instance():
    rng = np.random.default_rng([20261009, P])
    m_true, c_true, _, sigma_hat = draw_instance(P, 2, 2000, CChoice.ID, rng, standardize=True)
    return sigma_hat, 2.0 * np.eye(P)


@pytest.fixture(scope="module")
def lasso(instance):
    sigma_hat, c = instance
    return covloss_path(sigma_hat, c, "loglik", n_lambda=30, penalty="lasso", direction="down", tol=1e-9)


def test_lasso_start_is_the_dense_end_of_the_lasso_path(instance, lasso):
    sigma_hat, c = instance
    off = ~np.eye(P, dtype=bool)
    up = covloss_path(sigma_hat, c, "loglik", lambdas=lasso.lambdas, penalty="lasso", direction="up",
                      start="lasso", tol=1e-9)
    i_min = int(np.argmin(lasso.lambdas))
    # started at the lasso's own solution at the smallest lambda, the lasso stays there
    assert np.allclose(up.estimates[i_min], lasso.estimates[i_min], atol=1e-7)
    # that solution is the exact fit with the smallest l1 norm: half the entries are zero
    # (the exact fits form an affine set of dimension p (p - 1) / 2)
    assert int(np.sum(lasso.estimates[i_min][off] != 0)) == P * (P - 1) // 2
    assert cov.loss_value(lasso.estimates[i_min], sigma_hat, c, "loglik") == pytest.approx(
        cov.loss_value(cov.dense_fit(sigma_hat, c), sigma_hat, c, "loglik"), abs=1e-4)
    # MCP is nearly flat at lambda_min, so there the start decides: from the exact fit
    # -C Sigma^-1 / 2 the path stays dense, from the lasso solution it stays sparse
    mcp = {s: covloss_path(sigma_hat, c, "loglik", lambdas=lasso.lambdas, penalty="MCP",
                           direction="up", start=s, tol=1e-9).estimates[i_min] for s in ("exact", "lasso")}
    assert int(np.sum(mcp["lasso"][off] != 0)) < int(np.sum(mcp["exact"][off] != 0))


def test_mcp_from_the_lasso_start_is_stationary(instance, lasso):
    sigma_hat, c = instance
    up = covloss_path(sigma_hat, c, "loglik", lambdas=lasso.lambdas, penalty="MCP", direction="up",
                      start="lasso", tol=1e-9)
    assert np.max(up.kkt) < 1e-6                      # first-order conditions at every lambda
    # MCP leaves entries beyond gamma * lambda unshrunk, so even at lambda_max the path keeps its
    # largest entries: unlike the direct loss's dense -> sparse path it is not reset to the
    # diagonal fit there, exactly as the exact-fit start (covloss_path's Varando & Hansen order)
    i_max = int(np.argmax(lasso.lambdas))
    off = ~np.eye(P, dtype=bool)
    kept = np.abs(up.estimates[i_max][off])
    assert np.all((kept == 0) | (kept > 3.0 * lasso.lambdas.max() * 0.5))


def test_adaptive_lasso_with_equal_weights_is_the_lasso(instance, lasso):
    """A pilot without zeros and power = 0 give every entry weight 1: the adaptive path is
    then the lasso path."""
    sigma_hat, c = instance
    ada = adaptive_covloss_path(sigma_hat, c, "loglik", lambdas=lasso.lambdas, power=0.0,
                                pilot=cov.dense_fit(sigma_hat, c), tol=1e-9)
    assert np.all(ada.weights[~np.eye(P, dtype=bool)] == 1.0)
    for a, b in zip(ada.estimates, lasso.estimates):
        assert np.allclose(a, b, atol=1e-5)              # the same estimator, up to solver tolerance


def test_adaptive_lasso_weights_grid_and_conditions(instance, lasso):
    sigma_hat, c = instance
    pilot = lasso.estimates[int(np.argmin(lasso.lambdas))].copy()
    pilot[0, 1] = 0.0                                   # one excluded entry
    ada = adaptive_covloss_path(sigma_hat, c, "loglik", n_lambda=30, pilot=pilot, tol=1e-9)
    off = ~np.eye(P, dtype=bool)
    kept = off & (pilot != 0)
    assert ada.weights[0, 1] == EXCLUDED_WEIGHT and np.all(np.diag(ada.weights) == 0)
    assert np.isclose(ada.weights[kept].min(), 1.0)
    expected = np.abs(pilot[kept]).max() / np.abs(pilot[kept])
    assert np.allclose(ada.weights[kept], expected)
    # its own grid: the largest lambda is where the diagonal fit becomes stationary
    g = np.abs(cov.loss_grad(cov.diagonal_fit(sigma_hat, c), sigma_hat, c, "loglik"))
    assert np.isclose(ada.lambdas.max(), np.max(g[kept] / ada.weights[kept]))
    assert not np.any(ada.estimates[int(np.argmax(ada.lambdas))][off])
    # the excluded entry stays zero along the whole path, every solve is stationary
    for lam, m in zip(ada.lambdas, ada.estimates):
        assert m[0, 1] == 0.0
        grad = cov.loss_grad(m, sigma_hat, c, "loglik")
        assert stationarity(m, grad, lam, ada.weights, "lasso", None) < 1e-6
