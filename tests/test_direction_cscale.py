"""Path direction for MCP/SCAD on the direct loss, and the volatility matrix on standardized
data (``c_scale``).  Plain asserts, no fixtures: runs under pytest and as a script."""

import numpy as np

from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.lyapunov import design_matrix, lyapunov_residual, solve_lyapunov, vec
from gclm.objective.direct import direct_grad, lambda_max
from gclm.objective.penalties import penalty_weights, stationarity
from gclm.solvers.path import lambda_grid, lasso_path


def _instance(p=6, k=2, rep=0, n=1000, standardize=True):
    rng = np.random.default_rng([20260922, p, k, 0, rep])
    return draw_instance(p, k, n, CChoice.ID, rng, standardize=standardize, return_scale=True)


def _restricted_loss(sigma, c, support):
    a = design_matrix(sigma)
    idx = np.flatnonzero(vec(support))
    b, *_ = np.linalg.lstsq(a[:, idx], -vec(c), rcond=None)
    r = a[:, idx] @ b + vec(c)
    return 0.5 * float(r @ r)


def test_return_scale_does_not_change_the_draw():
    rng1 = np.random.default_rng([1, 2, 3])
    rng2 = np.random.default_rng([1, 2, 3])
    a = draw_instance(6, 2, 500, CChoice.ID, rng1, standardize=True)
    b = draw_instance(6, 2, 500, CChoice.ID, rng2, standardize=True, return_scale=True)
    assert len(a) == 4 and len(b) == 5
    for x, y in zip(a, b[:4]):
        assert np.array_equal(x, y)
    assert np.all(b[4] > 0)


def test_identity_scale_is_two_identity():
    assert np.array_equal(estimation_volatility(np.array([0.5, 2.0, 3.0]), "identity"), 2.0 * np.eye(3))
    assert np.array_equal(estimation_volatility(np.ones(4), "variance"), 2.0 * np.eye(4))


def test_standardized_population_model_is_misspecified_with_identity_and_exact_with_variance():
    """At n = inf the true support fits the standardized covariance exactly iff C is rescaled."""
    m_true, c_true, sigma_true, sigma_hat, scale = _instance(n=np.inf)
    support = m_true != 0
    exact = _restricted_loss(sigma_hat, estimation_volatility(scale, "variance"), support)
    wrong = _restricted_loss(sigma_hat, estimation_volatility(scale, "identity"), support)
    assert exact < 1e-20
    assert wrong > 1e-6
    # and the rescaled drift D^-1 M D solves the rescaled equation
    m_std = m_true * scale[None, :] / scale[:, None]
    res = lyapunov_residual(m_std, sigma_hat, estimation_volatility(scale, "variance"))
    assert np.max(np.abs(res)) < 1e-10


def test_up_path_is_stationary_and_differs_from_down():
    m_true, _, _, sigma_hat, scale = _instance()
    p = sigma_hat.shape[0]
    c = estimation_volatility(scale, "variance")
    lams = lambda_grid(lambda_max(sigma_hat, c), n_lambda=15, ratio=1e-2)
    w = penalty_weights(p)
    down = lasso_path(sigma_hat, c, lambdas=lams, penalty="MCP", tol=1e-10)
    up = lasso_path(sigma_hat, c, lambdas=lams, penalty="MCP", tol=1e-10, direction="up")
    for lam, m in zip(lams[:-1], up.estimates[:-1]):
        assert stationarity(m, direct_grad(m, sigma_hat, c), lam, w, "MCP") < 1e-6
    off = ~np.eye(p, dtype=bool)
    assert np.all(up.estimates[-1][off] == 0)                 # diagonal fit at lambda_max
    assert any(not np.array_equal(a != 0, b != 0) for a, b in zip(down.estimates, up.estimates))


def test_default_direction_is_unchanged():
    _, _, _, sigma_hat, _ = _instance()
    c = 2.0 * np.eye(sigma_hat.shape[0])
    lams = lambda_grid(lambda_max(sigma_hat, c), n_lambda=12, ratio=1e-2)
    a = lasso_path(sigma_hat, c, lambdas=lams, penalty="MCP")
    b = lasso_path(sigma_hat, c, lambdas=lams, penalty="MCP", direction="down")
    for x, y in zip(a.estimates, b.estimates):
        assert np.array_equal(x, y)
    # for the lasso the order is irrelevant: "up" is accepted and returns the same path
    la = lasso_path(sigma_hat, c, lambdas=lams)
    lb = lasso_path(sigma_hat, c, lambdas=lams, direction="up")
    for x, y in zip(la.estimates, lb.estimates):
        assert np.array_equal(x, y)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
