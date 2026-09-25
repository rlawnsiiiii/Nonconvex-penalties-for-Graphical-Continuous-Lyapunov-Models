"""Solvers: internal consistency, KKT optimality, and agreement with glmnet."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.dgp import CChoice, draw_instance
from gclm.examples import example2_cycle, example2_path
from gclm.lasso import (
    to_glmnet_lambda,
    diagonal_fit,
    lambda_grid,
    lambda_max,
    lasso_path,
    penalty_weights,
    solve_design,
    solve_fista,
)
from gclm.loss import frobenius_grad, frobenius_loss, objective
from gclm.lyap import (
    design_matrix,
    gram_matrix,
    lyapunov_residual,
    solve_lyapunov,
    unvec,
    vec,
)
from gclm.metrics import evaluate_path

from conftest import requires_ncvreg, requires_r, requires_skglm, run_r

# glmnet's objective is (1/(2 n_obs))||y - X b||^2 + lam_g * sum(v_j |b_j|) with
# n_obs = p^2, and it rescales `penalty.factor` to sum to nvars = p^2.  With
# `1 - diag(p)` that turns each off-diagonal weight into p^2/(p^2-p) = p/(p-1).
# Hence lam_paper = lam_g * p^2 * p/(p-1) = lam_g * p^3/(p-1).
# This is asserted empirically in test_glmnet_lambda_scaling below.
def glmnet_to_paper(lam_g: float, p: int) -> float:
    return lam_g * p ** 3 / (p - 1)


def paper_to_glmnet(lam: float, p: int) -> float:
    return lam * (p - 1) / p ** 3


def exact_restricted_solution(m, sigma, c, lam, weights):
    """The closed-form optimum given the active set and signs of ``m``.

    On the active set ``S`` the KKT conditions are linear:
    ``b_S = (A_S' A_S)^-1 (A_S' y - lam * w_S * sign(b_S))``.  This is an
    algorithm-independent ground truth -- no iterative solver involved -- and is
    valid whenever ``A_S`` has full column rank and the recovered signs are right.
    """
    p = sigma.shape[0]
    active = vec(m) != 0
    a_s = design_matrix(sigma)[:, active]
    rhs = a_s.T @ (-vec(c)) - lam * vec(weights)[active] * np.sign(vec(m)[active])
    b = np.linalg.solve(a_s.T @ a_s, rhs)
    full = np.zeros(p * p)
    full[active] = b
    return unvec(full, p)


def solution_is_unique(m, sigma):
    """The lasso minimizer is unique iff the active columns of the design matrix
    are linearly independent.  Because ``A(Sigma)`` is rank deficient by
    ``p(p-1)/2``, this genuinely fails at the dense end of the path."""
    active = vec(m) != 0
    if not active.any():
        return True
    sub = design_matrix(sigma)[:, active]
    return np.linalg.matrix_rank(sub) == sub.shape[1]


def kkt_violation(m, sigma, c, lam, weights):
    """max |subgradient| violation; 0 at an exact optimum."""
    g = frobenius_grad(m, sigma, c)
    thr = lam * weights
    active = m != 0
    viol = np.zeros_like(m)
    viol[active] = np.abs(g + thr * np.sign(m))[active]
    viol[~active] = np.maximum(np.abs(g) - thr, 0.0)[~active]
    return float(np.max(viol))


def test_penalty_weights_leave_diagonal_free():
    w = penalty_weights(4)
    assert np.all(np.diag(w) == 0) and np.all(w[~np.eye(4, dtype=bool)] == 1)
    assert np.all(penalty_weights(4, penalize_diagonal=True) == 1)


@pytest.mark.parametrize("lam", [0.05, 0.5, 5.0])
def test_fista_satisfies_kkt(instance, lam):
    m_true, sigma, c = instance
    w = penalty_weights(sigma.shape[0])
    m = solve_fista(sigma, c, lam, weights=w, tol=1e-13, max_iter=200_000)
    assert kkt_violation(m, sigma, c, lam, w) < 1e-6


@pytest.mark.parametrize("lam", [0.05, 0.5, 5.0])
def test_design_and_fista_agree(instance, lam):
    """Two independent algorithms, same optimum."""
    m_true, sigma, c = instance
    w = penalty_weights(sigma.shape[0])
    a = solve_design(sigma, c, lam, weights=w, tol=1e-13)
    b = solve_fista(sigma, c, lam, weights=w, tol=1e-13, max_iter=200_000)
    assert np.allclose(a, b, atol=1e-6)
    assert np.isclose(objective(a, sigma, c, lam, w),
                      objective(b, sigma, c, lam, w), rtol=1e-9, atol=1e-10)


def test_large_lambda_gives_diagonal_fit(instance):
    """At lambda_max the estimate is exactly the diagonal least-squares fit."""
    _, sigma, c = instance
    lam = lambda_max(sigma, c)
    m = solve_fista(sigma, c, lam * 1.001, tol=1e-13, max_iter=200_000)
    off = ~np.eye(sigma.shape[0], dtype=bool)
    assert np.all(m[off] == 0.0)
    assert np.allclose(np.diag(m), np.diag(diagonal_fit(sigma, c)), atol=1e-6)


def test_lambda_max_is_sharp(instance):
    """Just below lambda_max at least one off-diagonal entry must switch on."""
    _, sigma, c = instance
    lam = lambda_max(sigma, c)
    off = ~np.eye(sigma.shape[0], dtype=bool)
    below = solve_fista(sigma, c, lam * 0.99, tol=1e-13, max_iter=200_000)
    assert np.any(below[off] != 0.0)


def test_diagonal_fit_is_least_squares_optimum(instance):
    """The diagonal fit must zero the gradient in the diagonal directions."""
    _, sigma, c = instance
    m = diagonal_fit(sigma, c)
    g = 2 * lyapunov_residual(m, sigma, c) @ sigma
    assert np.allclose(np.diag(g), 0.0, atol=1e-8)


def test_lambda_max_bisection_matches_closed_form(instance):
    """The penalized-diagonal fallback reproduces the closed form when the
    diagonal happens to stay unpenalized in the search."""
    _, sigma, c = instance
    closed = lambda_max(sigma, c, penalize_diagonal=False)
    assert closed > 0
    grid = lambda_grid(closed, n_lambda=100, ratio=1e-4)
    assert grid.size == 100
    assert np.isclose(grid[-1], closed) and np.isclose(grid[0], closed * 1e-4)
    assert np.all(np.diff(grid) > 0)                       # increasing, as in the paper
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0])                  # log-equidistant


def test_path_is_monotone_in_sparsity(instance):
    """Sparsity should grow with lambda (not guaranteed in theory, but a strong
    smoke test that warm starts are not leaving stale coefficients behind)."""
    m_true, sigma, c = instance
    path = lasso_path(sigma, c, n_lambda=30, solver="fista", tol=1e-12)
    off = ~np.eye(sigma.shape[0], dtype=bool)
    nnz = np.array([np.sum(m[off] != 0) for m in path.estimates])
    assert nnz[0] >= nnz[-1]
    assert nnz[-1] == 0                                    # diagonal at lambda_max
    # allow tiny non-monotonicity from thresholding, but not a trend reversal
    assert np.sum(np.diff(nnz) > 0) <= 2


def test_example2_path_is_recovered_in_population_limit():
    """Dettling Figure 3, n = inf, graph G1 (the path).

    The path's drift matrix satisfies the irrepresentability condition, so the
    published result is perfect recovery.  This is the sharpest end-to-end check
    we have: DGP, Lyapunov solve, path, and metrics must all be right for it.
    """
    m_star = example2_path()
    c = 2 * np.eye(5)
    sigma = solve_lyapunov(m_star, c)
    path = lasso_path(sigma, c, n_lambda=100, tol=1e-13)
    ev = evaluate_path(path.estimates, m_star)
    assert ev["max_acc"] == 1.0
    assert ev["max_f1"] == 1.0
    assert ev["auc"] == 1.0
    off = ~np.eye(5, dtype=bool)
    exact = [m for m in path.estimates if np.array_equal(m[off] != 0, m_star[off] != 0)]
    assert exact, "some lambda on the grid must give the exact support"


def test_example2_cycle_is_not_recovered_in_population_limit():
    """Dettling Figure 3, n = inf, graph G2 (the 5-cycle).

    Irrepresentability fails here, and no amount of data fixes it: even with the
    population covariance no lambda recovers the support.  The reference values
    are read off Figure 3 at the `Inf` tick.
    """
    m_star = example2_cycle()
    c = 2 * np.eye(5)
    sigma = solve_lyapunov(m_star, c)
    path = lasso_path(sigma, c, n_lambda=100, tol=1e-13)
    ev = evaluate_path(path.estimates, m_star)
    off = ~np.eye(5, dtype=bool)
    assert not any(np.array_equal(m[off] != 0, m_star[off] != 0) for m in path.estimates)
    assert ev["max_acc"] < 1.0 and ev["max_f1"] < 1.0 and ev["auc"] < 1.0
    assert np.isclose(ev["max_acc"], 0.90, atol=0.02)
    assert np.isclose(ev["max_f1"], 0.80, atol=0.03)
    assert np.isclose(ev["auc"], 0.833, atol=0.03)


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_solvers_match_exact_restricted_solution(seed):
    """Both solvers must hit the exact KKT solution, not merely a low objective.

    This is the primary correctness test: the target is computed in closed form
    from the active set, so it does not depend on any iterative algorithm.
    """
    rng = np.random.default_rng(seed)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    w = penalty_weights(p)

    checked = 0
    for lam in lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-3):
        m = solve_fista(sigma, c, lam, weights=w, tol=1e-15, max_iter=1_000_000)
        if not solution_is_unique(m, sigma):
            continue
        exact = exact_restricted_solution(m, sigma, c, lam, w)
        assert np.array_equal(exact != 0, m != 0), lam
        assert np.allclose(m, exact, atol=1e-9), lam
        cd = solve_design(sigma, c, lam, weights=w, tol=1e-14, max_iter=100_000)
        assert np.allclose(cd, exact, atol=1e-6), lam
        checked += 1
    assert checked >= 6


# --------------------------------------------------------------------------- #
# Validation against R / glmnet
# --------------------------------------------------------------------------- #


@requires_r
@pytest.mark.r
def test_glmnet_lambda_scaling(instance):
    """Recover glmnet's lambda scaling empirically instead of trusting the docs.

    Finds the smallest glmnet lambda whose solution is diagonal, and checks it
    maps onto our closed-form lambda_max through p^3/(p-1).
    """
    _, sigma, c = instance
    p = sigma.shape[0]
    ours = lambda_max(sigma, c)
    grid_g = paper_to_glmnet(ours, p) * np.logspace(-0.3, 0.3, 121)
    out = run_r("backend_glmnet.R",
                {"Sigma": sigma.tolist(), "C": c.tolist(),
                 "lambda_glmnet": sorted(grid_g, reverse=True)})
    lam_g = np.asarray(out["lambda_glmnet"])
    beta = np.asarray(out["beta"])                     # row i <-> lam_g[i]
    off = ~np.eye(p, dtype=bool)
    diagonal = np.array([np.all(b.reshape(p, p, order="F")[off] == 0) for b in beta])
    assert diagonal.any() and not diagonal.all(), "grid must bracket lambda_max"
    theirs = glmnet_to_paper(lam_g[diagonal].min(), p)
    # grid spacing is ~1.15% per step, so agreement to a couple of steps is exact
    assert abs(np.log(theirs / ours)) < 0.03


@pytest.mark.parametrize("p", [3, 5, 8])
def test_design_matrix_is_rank_deficient(p):
    """``A(Sigma)`` always has a null space of dimension ``p(p-1)/2``.

    The Lyapunov residual is symmetric, so only ``p(p+1)/2`` of the ``p^2``
    equations are independent: ``A vec(V) = 0`` iff ``V Sigma`` is skew-symmetric.
    Hence the Gram matrix ``Gamma = A'A`` is singular for *every* ``Sigma`` and
    the smooth part of the objective is never strongly convex.  This is why the
    minimizer can be non-unique at small lambda, and why the nonconvex
    extension (S1b) needs restricted strong convexity rather than plain
    convexity.  Recorded here so the fact cannot silently regress.
    """
    rng = np.random.default_rng(p)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    sv = np.linalg.svd(design_matrix(sigma), compute_uv=False)
    assert int(np.sum(sv > 1e-9 * sv[0])) == p * (p + 1) // 2
    assert np.linalg.matrix_rank(gram_matrix(sigma)) == p * (p + 1) // 2


@requires_r
@pytest.mark.r
@pytest.mark.parametrize("seed", [0, 3])
def test_lasso_path_matches_glmnet(seed):
    """Agreement with glmnet over a full path.

    Objectives must agree to near machine precision everywhere.  Coefficients are
    additionally compared, but only where the minimizer is unique: because
    ``A(Sigma)`` is rank deficient (see test_design_matrix_is_rank_deficient),
    at small lambda several matrices attain the same optimal value, and the two
    coordinate systems legitimately land on different ones.
    """
    rng = np.random.default_rng(seed)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)

    lams = lambda_grid(lambda_max(sigma, c), n_lambda=40, ratio=1e-3)
    out = run_r("backend_glmnet.R",
                {"Sigma": sigma.tolist(), "C": c.tolist(),
                 "lambda_glmnet": [paper_to_glmnet(l, p) for l in lams[::-1]]})
    lam_g = np.asarray(out["lambda_glmnet"])
    beta = np.asarray(out["beta"])
    w = penalty_weights(p)

    n_compared = 0
    for lam_g_i, b in zip(lam_g, beta):
        lam = glmnet_to_paper(lam_g_i, p)
        r_est = b.reshape(p, p, order="F")
        ours = solve_fista(sigma, c, lam, weights=w, tol=1e-14, max_iter=500_000)

        obj_ours = objective(ours, sigma, c, lam, w)
        obj_r = objective(r_est, sigma, c, lam, w)
        # our solution is never worse, and the two agree to solver precision
        assert obj_ours <= obj_r + 1e-9, lam
        assert abs(obj_ours - obj_r) <= 1e-8 * max(abs(obj_r), 1.0), lam
        # both must satisfy the KKT conditions of the same problem
        assert kkt_violation(ours, sigma, c, lam, w) < 1e-6, lam

        # Coefficients agree only to glmnet's own accuracy.  glmnet stops on a
        # relative objective change, and at the dense end of the path the
        # objective is very flat (the active Gram matrix is ill conditioned), so
        # glmnet lands ~1e-4 away from the exact KKT solution that our solvers
        # reach -- see test_solvers_match_exact_restricted_solution.  Support
        # recovery is unaffected, but this matters for any estimation-error
        # comparison, so it is asserted rather than papered over.
        assert np.array_equal(ours != 0, r_est != 0), lam
        assert np.allclose(ours, r_est, atol=1e-3), lam
        n_compared += 1
    assert n_compared == len(lam_g)


@requires_r
@pytest.mark.r
def test_design_solver_matches_glmnet(instance):
    """The coordinate-descent reference solver against glmnet's own CD."""
    _, sigma, c = instance
    p = sigma.shape[0]
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=10, ratio=1e-2)
    out = run_r("backend_glmnet.R",
                {"Sigma": sigma.tolist(), "C": c.tolist(),
                 "lambda_glmnet": [paper_to_glmnet(l, p) for l in lams[::-1]]})
    beta = np.asarray(out["beta"])
    cache: dict = {}
    for lam_g_i, b in zip(np.asarray(out["lambda_glmnet"]), beta):
        ours = solve_design(sigma, c, glmnet_to_paper(lam_g_i, p),
                            tol=1e-13, _cache=cache)
        assert np.allclose(ours, b.reshape(p, p, order="F"), atol=1e-5)


@pytest.mark.parametrize("p", [4, 6])
def test_null_space_is_flat_for_every_c(p):
    """Directions in the null space of ``A(Sigma)`` leave the loss unchanged --
    whatever ``C`` is.

    Guards the reading of Section 2.3: ``A(Sigma) vec(V) = 0`` is a statement
    about the null space of the design matrix, not a Lyapunov equation with
    ``C`` set to zero.  ``C`` is the response, so it cannot affect the null
    space; the flat directions are ``V = W Sigma^-1`` with ``W`` skew-symmetric.
    """
    rng = np.random.default_rng(p)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    w = rng.normal(size=(p, p))
    w = w - w.T                                   # skew-symmetric
    v = w @ np.linalg.inv(sigma)

    assert np.allclose(design_matrix(sigma) @ vec(v), 0.0, atol=1e-10)

    m = rng.normal(size=(p, p))
    spd = rng.normal(size=(p, p))
    for c in (2 * np.eye(p), spd @ spd.T + p * np.eye(p), np.zeros((p, p))):
        assert np.isclose(frobenius_loss(m, sigma, c),
                          frobenius_loss(m + v, sigma, c), rtol=1e-12)


@requires_ncvreg
@pytest.mark.r
@pytest.mark.parametrize("seed", [3, 11])
def test_fista_matches_ncvreg(seed):
    """Validate ``solve_fista`` against ``ncvreg::ncvfit`` (Breheny & Huang 2011).

    ``ncvfit`` is the low-level entry point: unlike ``ncvreg``, it neither
    standardizes the design nor fits an intercept, so it solves exactly our
    problem.  It is a coordinate-descent implementation independent of both our
    FISTA and of ``glmnet``, and it converges to the exact KKT solution -- so
    this is a far sharper external check than ``test_lasso_path_matches_glmnet``,
    where ``glmnet``'s own tolerance is the limiting factor (see §8.2).

    ncvfit minimizes ``(1/(2n))||y - Xb||^2 + lambda * pf_j * pen(b_j)`` with
    ``n = nrow(X) = p^2``, so ``lambda_paper = lambda_ncv * p^2``.
    """
    rng = np.random.default_rng(seed)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    w = penalty_weights(p)
    n = p * p

    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-2)
    out = run_r("backend_ncvreg.R",
                {"Sigma": sigma.tolist(), "C": c.tolist(),
                 "lambda_ncv": [lam / n for lam in lams], "penalty": "lasso"}, ("glmnet", "jsonlite"))
    beta = np.asarray(out["beta"])

    for lam, b in zip(lams, beta):
        ncv = b.reshape(p, p, order="F")
        ours = solve_fista(sigma, c, lam, weights=w, tol=1e-14, max_iter=500_000)
        assert np.allclose(ours, ncv, atol=1e-9), lam
        assert np.array_equal(ours != 0, ncv != 0), lam
        # neither implementation is systematically better than the other here
        assert abs(objective(ours, sigma, c, lam, w)
                   - objective(ncv, sigma, c, lam, w)) < 1e-12, lam


ALL_SOLVERS = ("fista", "design", "glmnet", "ncvreg", "skglm")


def _fit_with(solver, sigma, c, lams):
    """Fit the path with `solver`, at each backend's tightest usable setting."""
    kw = {
        "fista": dict(tol=1e-14, max_iter=500_000),
        "design": dict(tol=1e-13),
        "skglm": dict(tol=1e-12),
    }.get(solver, {})
    return lasso_path(sigma, c, lambdas=lams, solver=solver, **kw)


@requires_r
@requires_ncvreg
@requires_skglm
@pytest.mark.r
def test_all_backends_agree():
    """Every backend must solve the same problem.

    The claim the simulations rest on is that they select the same supports and
    therefore report the same metrics.  Coefficients are compared at each
    backend's own accuracy: `ncvreg` is the reference because it reaches the
    exact KKT solution (see test_fista_matches_ncvreg).
    """
    from gclm.metrics import evaluate_path

    rng = np.random.default_rng(3)
    p = 5
    m_true, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    w = penalty_weights(p)
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-2)

    paths = {s: _fit_with(s, sigma, c, lams) for s in ALL_SOLVERS}
    base = paths["ncvreg"].estimates
    ref = evaluate_path(base, m_true)
    coef_tol = {"fista": 1e-9, "design": 1e-6, "glmnet": 1e-5, "skglm": 1e-5}

    for name, path in paths.items():
        assert len(path.estimates) == len(lams), name
        assert np.allclose(path.lambdas, lams), name

        # (1) identical support selection
        for lam, a, b in zip(lams, path.estimates, base):
            assert np.array_equal(a != 0, b != 0), (name, lam)

        # (2) identical reported metrics
        got = evaluate_path(path.estimates, m_true)
        for key in ("max_acc", "max_f1", "auc", "aupr"):
            assert np.isclose(got[key], ref[key]), (name, key)

        if name == "ncvreg":
            continue

        # (3) coefficients agree to that backend's accuracy
        md = max(np.max(np.abs(a - b)) for a, b in zip(path.estimates, base))
        assert md < coef_tol[name], (name, md)

        # (4) and none of them is at a worse objective value
        for lam, a, b in zip(lams, path.estimates, base):
            assert (objective(a, sigma, c, lam, w)
                    <= objective(b, sigma, c, lam, w) + 1e-9), (name, lam)


@pytest.mark.parametrize("solver", ["fista", "design", "skglm"])
def test_python_backends_agree_without_r(solver):
    """The R-free subset must agree too, so CI and the cluster need no R."""
    if solver == "skglm":
        pytest.importorskip("skglm")
    rng = np.random.default_rng(11)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=10, ratio=1e-2)

    ref = _fit_with("fista", sigma, c, lams).estimates
    got = _fit_with(solver, sigma, c, lams).estimates
    for lam, a, b in zip(lams, got, ref):
        assert np.array_equal(a != 0, b != 0), lam
        assert np.allclose(a, b, atol=1e-5), lam


@requires_skglm
def test_skglm_backend_supports_weighted_mcp():
    """skglm reaches MCP with the diagonal unpenalized, via WeightedMCPenalty."""
    rng = np.random.default_rng(5)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=8, ratio=1e-2)
    path = lasso_path(sigma, c, lambdas=lams, solver="skglm", penalty="MCP", gamma=3.0)
    assert len(path.estimates) == len(lams)
    # the weights took effect: the diagonal is never driven to zero
    assert all(np.all(np.diag(m) != 0) for m in path.estimates)


@requires_skglm
def test_skglm_scad_rejects_unpenalized_diagonal():
    """skglm has no weighted SCAD; the backend says so instead of silently
    penalizing the diagonal."""
    rng = np.random.default_rng(5)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    with pytest.raises(ValueError, match="no weighted SCAD"):
        lasso_path(sigma, 2 * np.eye(p), n_lambda=4, solver="skglm", penalty="SCAD")


def test_unknown_solver_is_rejected(instance):
    _, sigma, c = instance
    with pytest.raises(ValueError, match="unknown solver"):
        lasso_path(sigma, c, n_lambda=3, solver="nope")


@requires_ncvreg
@pytest.mark.r
@pytest.mark.parametrize("penalty", ["MCP", "SCAD"])
def test_ncvreg_backend_supports_nonconvex_penalties(penalty):
    """The ncvreg backend already reaches MCP/SCAD -- the S1b entry point.

    Only a smoke test: it checks the plumbing (penalty passed through, path
    returned, sparser than lasso at the same lambda because nonconvex penalties
    do not shrink large coefficients).
    """
    rng = np.random.default_rng(5)
    p = 5
    _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
    c = 2 * np.eye(p)
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=8, ratio=1e-2)

    lasso = lasso_path(sigma, c, lambdas=lams, solver="ncvreg", penalty="lasso")
    ncvx = lasso_path(sigma, c, lambdas=lams, solver="ncvreg", penalty=penalty)
    assert len(ncvx.estimates) == len(lams)
    off = ~np.eye(p, dtype=bool)
    # nonconvex penalties leave large entries unshrunk: bigger max |off-diagonal|
    assert max(np.max(np.abs(m[off])) for m in ncvx.estimates) >= \
           max(np.max(np.abs(m[off])) for m in lasso.estimates) - 1e-9


def test_glmnet_backend_rejects_nonconvex_penalty(instance):
    _, sigma, c = instance
    with pytest.raises(ValueError, match="penalty='lasso' only"):
        lasso_path(sigma, c, n_lambda=3, solver="glmnet", penalty="MCP")


@requires_r
@pytest.mark.r
def test_glmnet_backend_returns_every_requested_lambda():
    """glmnet must never hand back a truncated path.

    Its coordinate descent does not converge at tight ``thresh`` on this
    rank-deficient design, and glmnet's response is to return *fewer* lambdas
    rather than to error.  The backend walks a threshold ladder until the whole
    path fits; this pins that behaviour, including the loosening on the harder
    100-point path.
    """
    from gclm.rbridge import run_r as _run_r

    for p, n_lambda, ratio in ((5, 40, 1e-3), (8, 100, 1e-4)):
        rng = np.random.default_rng(3)
        _, _, _, sigma = draw_instance(p, 2, 300, CChoice.ID, rng)
        c = 2 * np.eye(p)
        lams = lambda_grid(lambda_max(sigma, c), n_lambda=n_lambda, ratio=ratio)
        out = _run_r("backend_glmnet.R",
                     {"Sigma": sigma.tolist(), "C": c.tolist(),
                      "lambda_glmnet": sorted(to_glmnet_lambda(lams, p).tolist(),
                                              reverse=True)},
                     ("glmnet", "jsonlite"))
        assert len(out["lambda_glmnet"]) == n_lambda
        assert np.asarray(out["beta"]).shape == (n_lambda, p * p)
        assert 1e-14 <= out["thresh_used"] <= 1e-7

        # and the high-level path API returns exactly the lambdas it was given
        path = lasso_path(sigma, c, lambdas=lams, solver="glmnet")
        assert len(path.estimates) == n_lambda
        assert np.allclose(path.lambdas, lams)
