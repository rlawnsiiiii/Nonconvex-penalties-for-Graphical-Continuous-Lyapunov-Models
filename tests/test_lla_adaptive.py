"""The two estimators that start from the lasso and then prune with convex steps
(next_steps/051026/cluster_campaign_051026.md):

- :func:`gclm.solvers.path.lla_path` -- MCP / SCAD by local linear approximation
  from the lasso solution at each lambda (Zou & Li 2008; Fan, Xue & Zou 2014);
- :func:`gclm.solvers.path.adaptive_lasso_path` -- adaptive lasso (Zou 2006) with
  weights from the dense end of the lasso path.

Both reduce to weighted lassos, which are convex, so each claim below is checked
against an optimality condition or against a solver that shares no code with the
one under test (coordinate descent on the explicit design).
"""

from __future__ import annotations

import numpy as np
import pytest

from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.lyapunov import design_matrix, vec
from gclm.objective.direct import diagonal_fit, direct_grad, lambda_max
from gclm.objective.penalties import DEFAULT_GAMMA, lla_weights, penalty_weights, stationarity
from gclm.solvers.coordinate import solve_design
from gclm.solvers.path import adaptive_lasso_path, lambda_grid, lasso_path, lla_path
from gclm.solvers.proxgrad import solve_fista

NONCONVEX = ("MCP", "SCAD")
TIGHT = dict(tol=1e-12, max_iter=200_000)


@pytest.fixture(scope="module")
def problem():
    """p = 6, standardised data with the rescaled C, a short grid."""
    rng = np.random.default_rng([20260922, 6, 2, 0, 1])
    m_true, _, _, sigma, scale = draw_instance(6, 2, 2000, CChoice.ID, rng, standardize=True,
                                               return_scale=True)
    c = estimation_volatility(scale, "variance")
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-3)
    return m_true, sigma, c, lams


# --------------------------------------------------------------------------- #
# the LLA weights
# --------------------------------------------------------------------------- #


def test_lla_weights_are_the_penalty_slopes_over_lambda():
    lam, x = 0.5, np.array([0.0, 0.1, 0.5, 1.0, 1.5, 1.85, 5.0])
    g = DEFAULT_GAMMA["MCP"]
    assert np.allclose(lla_weights(x, lam, "MCP"), np.maximum(1.0 - x / (g * lam), 0.0))
    g = DEFAULT_GAMMA["SCAD"]
    expected = np.where(x <= lam, 1.0, np.maximum(g * lam - x, 0.0) / ((g - 1.0) * lam))
    assert np.allclose(lla_weights(x, lam, "SCAD"), expected)
    for pen in NONCONVEX:
        w = lla_weights(-x, lam, pen)                       # symmetric in the sign
        assert np.array_equal(w, lla_weights(x, lam, pen))
        assert w[0] == 1.0 and w[-1] == 0.0 and np.all(np.diff(w) <= 0)


def test_lla_weights_reject_the_lasso_and_nonpositive_lambda():
    with pytest.raises(ValueError):
        lla_weights(np.ones(3), 0.5, "lasso")
    with pytest.raises(ValueError):
        lla_weights(np.ones(3), 0.0, "MCP")


# --------------------------------------------------------------------------- #
# lla_path
# --------------------------------------------------------------------------- #


def _well_posed(sigma, m, w, max_cond=1e4) -> bool:
    """Can a second solver be expected to reproduce the weighted-lasso solution ``m``?

    The design ``A(Sigma)`` has rank p(p+1)/2 only.  Once the active set (the
    nonzero entries plus every unpenalised one) reaches that size the fit
    interpolates: the minimiser may not be unique, and coordinate descent needs
    orders of magnitude more sweeps than a test should take (measured here: its
    optimality violation is still 1e-6 after 20,000 sweeps, FISTA's is 1e-12).
    Below that size, with a well-conditioned active design, the two solvers agree
    to 1e-11, and to 1e-6 one entry short of saturation."""
    p = sigma.shape[0]
    active = np.flatnonzero(vec((m != 0) | (w == 0)))
    if len(active) >= p * (p + 1) // 2:
        return False
    return np.linalg.cond(design_matrix(sigma)[:, active]) < max_cond


@pytest.mark.parametrize("penalty", NONCONVEX)
def test_one_lla_step_is_the_weighted_lasso_built_from_the_lasso(problem, penalty):
    """steps = 1: at every lambda the estimate satisfies the optimality conditions
    of the lasso whose weights are the penalty's slopes at the lasso solution.
    For a convex problem those conditions are necessary and sufficient, so this is
    the certificate.

    Where the problem is also well posed (:func:`_well_posed`) the estimate is
    compared with an independent solver, coordinate descent on the explicit design.
    At the dense end of the path, where most entries are beyond gamma * lambda and
    therefore unpenalised, only the certificate applies."""
    _, sigma, c, lams = problem
    off = penalty_weights(sigma.shape[0])
    lasso = lasso_path(sigma, c, lambdas=lams, zero_tol=0.0, **TIGHT)
    one = lla_path(sigma, c, lambdas=lams, penalty=penalty, steps=1, zero_tol=0.0, **TIGHT)
    compared = 0
    for lam, m0, m1 in zip(lams[:-1], lasso.estimates[:-1], one.estimates[:-1]):
        w = off * lla_weights(m0, lam, penalty)
        assert stationarity(m1, direct_grad(m1, sigma, c), lam, w, "lasso") < 1e-7
        if _well_posed(sigma, m1, w):
            compared += 1
            ref = solve_design(sigma, c, lam, weights=w, tol=1e-13, max_iter=20_000)
            assert np.max(np.abs(m1 - ref)) < 1e-5
    assert compared >= 4


@pytest.mark.parametrize("penalty", NONCONVEX)
def test_a_fixed_point_of_lla_is_stationary_for_the_nonconvex_objective(problem, penalty):
    """Iterated until nothing changes, LLA ends at a stationary point of the MCP /
    SCAD objective -- the sense in which it "computes MCP"."""
    _, sigma, c, lams = problem
    off = penalty_weights(sigma.shape[0])
    path = lla_path(sigma, c, lambdas=lams, penalty=penalty, steps=200, zero_tol=0.0, **TIGHT)
    fixed = 0
    for lam, m in zip(lams[:-1], path.estimates[:-1]):
        again = solve_fista(sigma, c, lam, weights=off * lla_weights(m, lam, penalty), m_init=m, **TIGHT)
        if np.max(np.abs(again - m)) < 1e-8:
            fixed += 1
            assert stationarity(m, direct_grad(m, sigma, c), lam, off, penalty) < 1e-6
    assert fixed >= len(lams) - 2


@pytest.mark.parametrize("penalty", NONCONVEX)
def test_lla_estimate_at_one_lambda_does_not_depend_on_the_rest_of_the_grid(problem, penalty):
    """No warm starts between lambdas: the estimate at a lambda computed alone
    equals the one computed as part of the path.  (The warm-started MCP / SCAD
    paths do not have this property; that is the point of the estimator.)"""
    _, sigma, c, lams = problem
    full = lla_path(sigma, c, lambdas=lams, penalty=penalty, **TIGHT)
    for i in (2, 6, 9):
        alone = lla_path(sigma, c, lambdas=lams[i:i + 1], penalty=penalty, **TIGHT)
        assert np.max(np.abs(alone.estimates[0] - full.estimates[i])) < 1e-6


def test_lla_ends_at_the_diagonal_fit_and_tends_to_the_lasso_for_large_gamma(problem):
    _, sigma, c, lams = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    lasso = lasso_path(sigma, c, lambdas=lams, **TIGHT)
    for penalty in NONCONVEX:
        path = lla_path(sigma, c, lambdas=lams, penalty=penalty, **TIGHT)
        assert np.all(path.estimates[-1][off] == 0)
        assert np.allclose(path.estimates[-1], diagonal_fit(sigma, c), atol=1e-10)
        flat = lla_path(sigma, c, lambdas=lams, penalty=penalty, gamma=1e9, **TIGHT)
        for a, b in zip(flat.estimates, lasso.estimates):   # gamma -> inf: every weight -> 1
            assert np.max(np.abs(a - b)) < 1e-6
        # with the default gamma it is a different estimator
        assert any(np.max(np.abs(a - b)) > 1e-3 for a, b in zip(path.estimates, lasso.estimates))


def test_lla_accepts_a_precomputed_lasso_path_and_validates_its_arguments(problem):
    _, sigma, c, lams = problem
    lasso = lasso_path(sigma, c, lambdas=lams, zero_tol=0.0, **TIGHT)
    a = lla_path(sigma, c, lambdas=lams, penalty="MCP", **TIGHT)
    b = lla_path(sigma, c, penalty="MCP", lasso=lasso, **TIGHT)
    assert np.array_equal(a.lambdas, b.lambdas)
    for x, y in zip(a.estimates, b.estimates):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError):
        lla_path(sigma, c, lambdas=lams, penalty="lasso")
    with pytest.raises(ValueError):
        lla_path(sigma, c, lambdas=lams, penalty="MCP", steps=0)


# --------------------------------------------------------------------------- #
# adaptive_lasso_path
# --------------------------------------------------------------------------- #


def test_adaptive_weights_come_from_the_dense_end_of_the_lasso_path(problem):
    _, sigma, c, _ = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    path = adaptive_lasso_path(sigma, c, n_lambda=12, ratio=1e-3, **TIGHT)
    pilot = lasso_path(sigma, c, n_lambda=12, ratio=1e-3, **TIGHT).estimates[0]
    assert np.array_equal(path.pilot, pilot)
    kept = off & (pilot != 0)
    assert kept.any() and (off & ~kept).any()               # the test covers both kinds of entries
    assert np.all(np.diag(path.weights) == 0)               # diagonal unpenalised
    assert np.all(np.isinf(path.weights[off & ~kept]))      # excluded
    expected = 1.0 / np.abs(pilot[kept])
    assert np.allclose(path.weights[kept], expected / expected.min())
    assert np.isclose(path.weights[kept].min(), 1.0)        # the largest pilot entry


def test_adaptive_lasso_solves_its_weighted_problem(problem):
    """Optimality conditions of the weighted lasso at every lambda, agreement with
    an independent solver, and the excluded entries stay exactly zero."""
    _, sigma, c, _ = problem
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    path = adaptive_lasso_path(sigma, c, n_lambda=12, ratio=1e-3, zero_tol=0.0, **TIGHT)
    finite = np.where(np.isfinite(path.weights), path.weights, 0.0)
    excluded = off & np.isinf(path.weights)
    compared = 0
    for lam, m in zip(path.lambdas[:-1], path.estimates[:-1]):
        assert np.all(m[excluded] == 0)
        grad = direct_grad(m, sigma, c)
        on = ~excluded                                      # the entries the problem is about
        assert stationarity(m[on], grad[on], lam, finite[on], "lasso") < 1e-7
        if _well_posed(sigma, m, path.weights):
            compared += 1
            ref = solve_design(sigma, c, lam, weights=path.weights, tol=1e-13, max_iter=20_000)
            assert np.max(np.abs(m - ref)) < 1e-5
    assert compared >= 4
    assert np.all(path.estimates[-1][off] == 0)             # diagonal at its own lambda_max
    below = path.estimates[-2]                               # ... and not just below it
    assert np.any(below[off] != 0)


def test_adaptive_lasso_only_selects_what_the_pilot_selected(problem):
    _, sigma, c, _ = problem
    off = ~np.eye(sigma.shape[0], dtype=bool)
    path = adaptive_lasso_path(sigma, c, n_lambda=12, ratio=1e-3, **TIGHT)
    pilot_support = (path.pilot != 0) & off
    for m in path.estimates:
        assert not np.any((m != 0) & off & ~pilot_support)
    sizes = [int(np.sum((m != 0) & off)) for m in path.estimates]
    assert sizes[0] > 0 and sizes[-1] == 0


def test_adaptive_lasso_with_unit_weights_is_the_lasso(problem):
    """A pilot whose off-diagonal entries all have the same size gives every entry
    weight 1: same lambda_max, same grid, same path as the lasso."""
    _, sigma, c, _ = problem
    p = sigma.shape[0]
    pilot = np.ones((p, p)) - 3.0 * np.eye(p)
    ada = adaptive_lasso_path(sigma, c, n_lambda=12, ratio=1e-3, pilot=pilot, **TIGHT)
    lasso = lasso_path(sigma, c, n_lambda=12, ratio=1e-3, **TIGHT)
    assert np.allclose(ada.lambdas, lasso.lambdas, rtol=1e-12)
    for a, b in zip(ada.estimates, lasso.estimates):
        assert np.max(np.abs(a - b)) < 1e-7


def test_adaptive_lasso_with_an_empty_pilot_returns_the_diagonal_fit(problem):
    _, sigma, c, _ = problem
    p = sigma.shape[0]
    path = adaptive_lasso_path(sigma, c, n_lambda=5, pilot=-np.eye(p))
    assert len(path.estimates) == 5
    for m in path.estimates:
        assert np.allclose(m, diagonal_fit(sigma, c))


# --------------------------------------------------------------------------- #
# what they are for
# --------------------------------------------------------------------------- #


def test_both_estimators_keep_large_entries_less_shrunk_than_the_lasso(problem):
    """The motivation in one check: at a moderate lambda the entries that all three
    estimators select are, on average, larger in absolute value under LLA and the
    adaptive lasso than under the lasso (less shrinkage of what is kept)."""
    _, sigma, c, lams = problem
    off = ~np.eye(sigma.shape[0], dtype=bool)
    i = 6
    lasso = lasso_path(sigma, c, lambdas=lams, **TIGHT).estimates[i]
    lla = lla_path(sigma, c, lambdas=lams, penalty="MCP", **TIGHT).estimates[i]
    common = off & (lasso != 0) & (lla != 0)
    assert common.sum() >= 2
    assert np.mean(np.abs(lla[common])) > np.mean(np.abs(lasso[common]))
