"""Correctness of the hand-written FISTA solver (`gclm.lasso.solve_fista`).

`solve_fista` is the default backend, so it carries more of the burden of proof
than a library would.  These tests attack it from several independent
directions, in rough order of strength:

1. an **analytic** solution known in closed form (no optimizer involved);
2. a **duality-gap certificate** -- a rigorous bound on suboptimality;
3. an **independent solver class**: `cvxpy` / CLARABEL is interior-point, not
   first-order, so it shares no code path or failure mode with FISTA;
4. **mathematical invariances** the true minimizer must satisfy (initialization
   independence, permutation equivariance, scale equivariance);
5. the **convergence rate** FISTA is supposed to achieve.

Agreement with `ncvreg`, `glmnet`, `skglm`, `pyproximal` and the closed-form KKT
solution lives in `test_lasso.py`; the published Figure 3 values in
`test_lasso.py::test_example2_*`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gclm.dgp import CChoice, draw_instance
from gclm.lasso import (
    lambda_grid,
    lambda_max,
    penalty_weights,
    solve_fista,
)
from gclm.loss import objective
from gclm.lyap import design_matrix, vec

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CVXPY_AVAILABLE = False

requires_cvxpy = pytest.mark.skipif(
    not CVXPY_AVAILABLE, reason="cvxpy not installed (pip install cvxpy)"
)

TIGHT = dict(tol=1e-14, max_iter=500_000)


def _problem(seed=3, p=5, n=300):
    rng = np.random.default_rng(seed)
    _, _, _, sigma = draw_instance(p, 2, n, CChoice.ID, rng)
    return sigma, 2 * np.eye(p), penalty_weights(p)


# --------------------------------------------------------------------------- #
# 1. Analytic solution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lam", [1e-3, 0.5, 10.0])
@pytest.mark.parametrize("p", [4, 8])
def test_matches_analytic_solution_for_diagonal_inputs(p, lam):
    """With diagonal `Sigma` and `C` the minimizer is known in closed form.

    A diagonal `M` makes `R = M Sigma + Sigma M' + C` diagonal with entries
    `2 m_i s_i + c_i`, so `m_i = -c_i / (2 s_i)` drives the loss to exactly zero
    while costing no penalty (the diagonal is unpenalized).  Objective zero is
    the global minimum, so this holds for *every* lambda -- no optimizer needed
    to know the answer.
    """
    rng = np.random.default_rng(p)
    s = rng.uniform(1.0, 4.0, p)
    c = rng.uniform(1.0, 3.0, p)
    sigma, cmat = np.diag(s), np.diag(c)
    analytic = np.diag(-c / (2 * s))

    m = solve_fista(sigma, cmat, lam, weights=penalty_weights(p), **TIGHT)
    assert np.allclose(m, analytic, atol=1e-10)
    assert objective(m, sigma, cmat, lam, penalty_weights(p)) < 1e-20


# --------------------------------------------------------------------------- #
# 2. Duality gap -- a certificate, not a comparison
# --------------------------------------------------------------------------- #


def _duality_gap(m, sigma, c, lam, w):
    """Primal minus dual value at a feasible dual point.

    For `min 0.5||y - Xb||^2 + lam * sum_j w_j |b_j|`, any residual `r = y - Xb`
    rescaled to satisfy `|x_j' theta| <= lam w_j` is dual feasible, and
    `D(theta) = 0.5||y||^2 - 0.5||y - theta||^2` lower-bounds the optimum.  The
    gap therefore bounds suboptimality from above with no reference solution.
    """
    x, y = design_matrix(sigma), -vec(c)
    b = vec(m)
    r = y - x @ b
    wv = vec(w)
    pen = wv > 0
    scale = max(1.0, float(np.max(np.abs(x.T @ r)[pen] / (lam * wv[pen]))))
    theta = r / scale
    primal = 0.5 * r @ r + lam * float(np.sum(wv * np.abs(b)))
    dual = 0.5 * y @ y - 0.5 * float(np.sum((y - theta) ** 2))
    return primal - dual


@pytest.mark.parametrize("lam", [0.05, 0.3, 2.0])
def test_duality_gap_certifies_optimality(lam):
    """The gap bounds suboptimality without needing a reference solution."""
    sigma, c, w = _problem()
    m = solve_fista(sigma, c, lam, weights=w, **TIGHT)
    assert abs(_duality_gap(m, sigma, c, lam, w)) < 1e-10


def test_duality_gap_shrinks_with_tolerance():
    """A looser tolerance must give a demonstrably worse certificate."""
    sigma, c, w = _problem()
    gaps = [
        abs(_duality_gap(solve_fista(sigma, c, 0.3, weights=w, tol=t,
                                     max_iter=500_000), sigma, c, 0.3, w))
        for t in (1e-4, 1e-8, 1e-14)
    ]
    assert gaps[0] > gaps[-1]
    assert gaps[-1] < 1e-10


# --------------------------------------------------------------------------- #
# 3. Independent solver class
# --------------------------------------------------------------------------- #


@requires_cvxpy
@pytest.mark.parametrize("lam", [0.05, 0.3, 2.0])
def test_matches_cvxpy_interior_point(lam):
    """Against an interior-point conic solver -- no shared code or failure mode.

    CLARABEL solves the problem from the conic side; FISTA from the first-order
    side.  Agreement across that divide is strong evidence neither is wrong.
    """
    sigma, c, w = _problem()
    p = sigma.shape[0]

    m_var = cp.Variable((p, p))
    resid = m_var @ sigma + sigma @ m_var.T + c
    prob = cp.Problem(cp.Minimize(
        0.5 * cp.sum_squares(resid) + lam * cp.sum(cp.multiply(w, cp.abs(m_var)))
    ))
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal"

    ours = solve_fista(sigma, c, lam, weights=w, **TIGHT)
    assert np.allclose(ours, m_var.value, atol=1e-6)
    # and FISTA is never at a worse objective than the interior-point solution
    assert (objective(ours, sigma, c, lam, w)
            <= objective(m_var.value, sigma, c, lam, w) + 1e-9)


# --------------------------------------------------------------------------- #
# 4. Invariances the true minimizer must satisfy
# --------------------------------------------------------------------------- #


def test_optimal_value_is_independent_of_initialization():
    """The problem is convex, so every starting point must reach the same value.

    (The *argmin* need not be unique -- the design is rank deficient -- but the
    optimal value is.)
    """
    sigma, c, w = _problem()
    p = sigma.shape[0]
    rng = np.random.default_rng(0)
    starts = [None, np.zeros((p, p)), -np.eye(p),
              rng.normal(size=(p, p)), 10 * rng.normal(size=(p, p))]
    values = [objective(solve_fista(sigma, c, 0.3, weights=w, m_init=s, **TIGHT),
                        sigma, c, 0.3, w) for s in starts]
    assert max(values) - min(values) < 1e-12


def test_is_permutation_equivariant():
    """Relabelling the variables must relabel the solution, nothing more."""
    sigma, c, w = _problem()
    p = sigma.shape[0]
    perm = np.eye(p)[np.random.default_rng(1).permutation(p)]

    base = solve_fista(sigma, c, 0.3, weights=w, **TIGHT)
    permuted = solve_fista(perm @ sigma @ perm.T, perm @ c @ perm.T, 0.3,
                           weights=w, **TIGHT)
    assert np.allclose(perm @ base @ perm.T, permuted, atol=1e-10)


def test_is_scale_equivariant_in_c():
    """Scaling `C` and `lambda` together scales the solution by the same factor.

    `f(aM; Sigma, aC) = a^2 f(M; Sigma, C)` and the penalty is 1-homogeneous, so
    the minimizer for `(aC, a*lam)` is `a` times the minimizer for `(C, lam)`.
    """
    sigma, c, w = _problem()
    a = 3.0
    base = solve_fista(sigma, c, 0.3, weights=w, **TIGHT)
    scaled = solve_fista(sigma, a * c, a * 0.3, weights=w, **TIGHT)
    assert np.allclose(a * base, scaled, atol=1e-9)


def test_zero_lambda_solves_the_unpenalized_problem():
    """At lambda = 0 the gradient must vanish (no penalty to balance it)."""
    from gclm.loss import frobenius_grad

    sigma, c, w = _problem()
    m = solve_fista(sigma, c, 0.0, weights=w, **TIGHT)
    assert np.max(np.abs(frobenius_grad(m, sigma, c))) < 1e-6


# --------------------------------------------------------------------------- #
# 5. The convergence rate FISTA promises
# --------------------------------------------------------------------------- #


def test_achieves_at_least_the_accelerated_rate():
    """`k^2 (F(x_k) - F*)` must stay bounded -- the FISTA guarantee.

    Beck & Teboulle (2009), Thm 4.4.  With the adaptive restart of
    O'Donoghue & Candes (2015) convergence is in practice much faster than
    `O(1/k^2)`, so the bounded sequence should be *decreasing*; a broken
    momentum term would show up as growth.
    """
    sigma, c, w = _problem()
    lam = 0.3
    star = objective(solve_fista(sigma, c, lam, weights=w, tol=1e-15,
                                 max_iter=1_000_000), sigma, c, lam, w)

    scaled = []
    for k in (10, 20, 50):
        m = solve_fista(sigma, c, lam, weights=w, tol=0.0, max_iter=k)
        gap = objective(m, sigma, c, lam, w) - star
        assert gap >= -1e-12, k                       # never below the optimum
        scaled.append(k * k * gap)
    assert scaled[0] > scaled[1] > scaled[2]          # faster than O(1/k^2)

    # and it reaches machine precision quickly
    m = solve_fista(sigma, c, lam, weights=w, tol=0.0, max_iter=200)
    assert objective(m, sigma, c, lam, w) - star < 1e-14


def test_objective_never_increases_along_the_path():
    """Warm-started path fits must each be optimal for their own lambda.

    A stale warm start would show up as an objective above the cold-started one.
    """
    sigma, c, w = _problem()
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=10, ratio=1e-2)
    warm = None
    for lam in lams[::-1]:
        warm = solve_fista(sigma, c, lam, weights=w, m_init=warm, **TIGHT)
        cold = solve_fista(sigma, c, lam, weights=w, **TIGHT)
        assert (objective(warm, sigma, c, lam, w)
                <= objective(cold, sigma, c, lam, w) + 1e-10), lam


@pytest.mark.parametrize("p", [3, 6, 10])
def test_converges_from_any_problem_size(p):
    """Smoke test across sizes: KKT satisfied and diagonal negative."""
    from gclm.loss import frobenius_grad

    rng = np.random.default_rng(p)
    _, _, _, sigma = draw_instance(p, 2, 500, CChoice.ID, rng)
    c, w = 2 * np.eye(p), penalty_weights(p)
    lam = 0.5 * lambda_max(sigma, c)
    m = solve_fista(sigma, c, lam, weights=w, **TIGHT)

    g = frobenius_grad(m, sigma, c)
    thr = lam * w
    active = m != 0
    viol = np.zeros_like(m)
    viol[active] = np.abs(g + thr * np.sign(m))[active]
    viol[~active] = np.maximum(np.abs(g) - thr, 0.0)[~active]
    assert np.max(viol) < 1e-6
    assert np.all(np.diag(m) < 0)
