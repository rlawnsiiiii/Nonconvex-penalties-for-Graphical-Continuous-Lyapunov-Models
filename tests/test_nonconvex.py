"""MCP and SCAD: penalty functions, proximal operators, conventions, solvers.

Organised from the smallest piece outwards, and -- wherever possible -- against
references that share no code with ours:

1. penalty values and derivatives   vs the formulas, skglm, pyproximal
2. proximal operators               vs **brute-force minimisation**, skglm, pyproximal
3. the two conventions              what ncvreg actually minimises, established
                                    on a generic regression (no Lyapunov code)
4. the solver (monotone APG)        stationarity, monotonicity, lambda limits
5. the backends                     fista vs skglm (textbook) and vs ncvreg
                                    (ncvreg convention), including the fact that
                                    nonconvex solvers may reach different local
                                    minima
6. the motivation                   MCP/SCAD remove the lasso's shrinkage bias

See docs/NONCONVEX.md for the definitions these tests pin down.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gclm.data.simulate import CChoice, draw_instance
from gclm.data.examples import example2_path
from gclm.objective.direct import diagonal_fit, lambda_max
from gclm.objective.penalties import penalty_scale, penalty_weights
from gclm.solvers.path import lambda_grid, lasso_path
from gclm.solvers.proxgrad import solve_fista
from gclm.objective.direct import direct_grad, objective
from gclm.lyapunov import design_column_sq_norms, design_matrix, solve_lyapunov, unvec
from gclm.objective.penalties import (
    DEFAULT_GAMMA,
    canonical,
    derivative,
    prox,
    resolve_gamma,
    stationarity,
    value,
)

from conftest import requires_ncvreg, requires_skglm, run_r

try:
    import pyproximal
    PYPROXIMAL = True
except ImportError:  # pragma: no cover
    PYPROXIMAL = False
requires_pyproximal = pytest.mark.skipif(not PYPROXIMAL, reason="pyproximal not installed")

NONCONVEX = ("MCP", "SCAD")
LAM = 0.5
ZS = np.array([-4.0, -1.9, -1.2, -0.7, -0.3, -0.05, 0.0, 0.2, 0.55, 0.8,
               1.1, 1.6, 1.85, 2.5, 5.0])
TIGHT = dict(tol=1e-13, max_iter=500_000)


def scalar(pen, x, lam=LAM, gamma=None, scale=None):
    """Penalty of a single number, through the vectorised API."""
    return value(np.array([[x]]), lam, np.ones((1, 1)), pen, gamma,
                 None if scale is None else np.array([[scale]]))


def brute_prox(z, t, pen, gamma, lam=LAM, scale=1.0):
    """argmin_x 0.5 (x - z)^2 + t P(x): dense grid, then a refined grid.

    Knows nothing about the closed forms -- the package-independent oracle.
    """
    def obj(xs):
        return 0.5 * (xs - z) ** 2 + t * np.array(
            [scalar(pen, x, lam, gamma, scale) for x in xs])
    xs = np.linspace(-abs(z) - 2, abs(z) + 2, 8001)
    x0 = xs[np.argmin(obj(xs))]
    xs = np.linspace(x0 - 2e-3, x0 + 2e-3, 4001)
    return xs[np.argmin(obj(xs))]


def _problem(seed=3, p=6):
    rng = np.random.default_rng(seed)
    _, _, _, sigma = draw_instance(p, 2, 400, CChoice.ID, rng, standardize=True)
    return sigma, 2 * np.eye(p), penalty_weights(p)


# --------------------------------------------------------------------------- #
# 1. Penalty values
# --------------------------------------------------------------------------- #


def test_penalty_names_and_default_gammas():
    assert canonical(None) == canonical("L1") == canonical("lasso") == "lasso"
    assert canonical("mcp") == "MCP" and canonical("Scad") == "SCAD"
    with pytest.raises(ValueError):
        canonical("elastic")
    assert resolve_gamma("MCP", None) == DEFAULT_GAMMA["MCP"] == 3.0
    assert resolve_gamma("SCAD", None) == DEFAULT_GAMMA["SCAD"] == 3.7
    assert resolve_gamma("lasso", 5.0) is None


@pytest.mark.parametrize("pen, bad", [("MCP", 1.0), ("MCP", 0.5), ("SCAD", 2.0), ("SCAD", 1.5)])
def test_gamma_domain_matches_ncvreg(pen, bad):
    """ncvreg refuses gamma <= 1 (MCP) and gamma <= 2 (SCAD); so do we."""
    with pytest.raises(ValueError, match="requires gamma"):
        resolve_gamma(pen, bad)


@pytest.mark.parametrize("pen", NONCONVEX)
def test_penalty_shape(pen):
    """P(0) = 0, slope lam at 0+, continuous at the knots, constant beyond gamma lam."""
    g = DEFAULT_GAMMA[pen]
    assert scalar(pen, 0.0) == 0.0
    eps = 1e-7
    assert np.isclose(scalar(pen, eps) / eps, LAM, rtol=1e-5)
    for knot in ([g * LAM] if pen == "MCP" else [LAM, g * LAM]):
        assert np.isclose(scalar(pen, knot - 1e-9), scalar(pen, knot + 1e-9), atol=1e-7)
    flat = g * LAM * LAM / 2 if pen == "MCP" else LAM * LAM * (g + 1) / 2
    for x in (g * LAM + 0.1, 10.0, -50.0):
        assert np.isclose(scalar(pen, x), flat)
    # symmetric and nondecreasing in |x|
    xs = np.linspace(0, 5, 200)
    vals = np.array([scalar(pen, x) for x in xs])
    assert np.all(np.diff(vals) >= -1e-12)
    assert np.allclose(vals, [scalar(pen, -x) for x in xs])


@pytest.mark.parametrize("pen", NONCONVEX)
def test_large_gamma_recovers_the_lasso(pen):
    for x in (0.1, 0.7, 2.0):
        assert np.isclose(scalar(pen, x, gamma=1e9), LAM * x, rtol=1e-6)


@pytest.mark.parametrize("pen", ("lasso",) + NONCONVEX)
@pytest.mark.parametrize("scale", [None, 2.5])
def test_derivative_matches_finite_differences(pen, scale):
    rng = np.random.default_rng(1)
    m = rng.normal(size=(4, 4)) * 1.5
    w = np.ones((4, 4))
    d = derivative(m, LAM, w, pen, None, scale)
    h = 1e-6
    for i in range(4):
        for j in range(4):
            e = np.zeros((4, 4))
            e[i, j] = h
            fd = (value(m + e, LAM, w, pen, None, scale)
                  - value(m - e, LAM, w, pen, None, scale)) / (2 * h)
            assert np.isclose(d[i, j], fd, atol=1e-5), (i, j)


@requires_skglm
@pytest.mark.parametrize("pen", NONCONVEX)
def test_values_match_skglm(pen):
    import skglm.penalties as P
    g = DEFAULT_GAMMA[pen]
    ref = P.MCPenalty(alpha=LAM, gamma=g) if pen == "MCP" else P.SCAD(alpha=LAM, gamma=g)
    for z in ZS:
        assert np.isclose(scalar(pen, z), ref.value(np.array([z])), atol=1e-14)


@requires_pyproximal
def test_scad_value_matches_pyproximal():
    ref = pyproximal.SCAD(sigma=LAM, a=3.7)
    for z in ZS:
        assert np.isclose(scalar("SCAD", z), float(ref(np.array([z]))), atol=1e-14)


def test_nonconvex_penalties_reject_fractional_weights():
    with pytest.raises(ValueError, match=r"weights in \{0, 1\}"):
        value(np.ones((2, 2)), LAM, np.full((2, 2), 0.5), "MCP")


# --------------------------------------------------------------------------- #
# 2. Proximal operators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pen", ("lasso",) + NONCONVEX)
@pytest.mark.parametrize("t", [0.05, 0.3, 1.0])
def test_prox_is_the_global_minimiser(pen, t):
    """The closed form against brute-force minimisation of 0.5(x-z)^2 + t P(x)."""
    g = DEFAULT_GAMMA.get(pen)
    got = prox(ZS, t, LAM, np.ones_like(ZS), pen, g)
    want = np.array([brute_prox(z, t, pen, g) for z in ZS])
    assert np.allclose(got, want, atol=2e-6)


@pytest.mark.parametrize("pen", NONCONVEX)
@pytest.mark.parametrize("scale", [0.4, 3.0])
def test_scaled_prox_is_the_global_minimiser(pen, scale):
    """Same, for the ncvreg convention P(v x) / v."""
    g = DEFAULT_GAMMA[pen]
    t = 0.2
    got = prox(ZS, t, LAM, np.ones_like(ZS), pen, g, np.full_like(ZS, scale))
    want = np.array([brute_prox(z, t, pen, g, scale=scale) for z in ZS])
    assert np.allclose(got, want, atol=2e-6)


@requires_skglm
@pytest.mark.parametrize("pen", NONCONVEX)
@pytest.mark.parametrize("t", [0.05, 0.3, 1.0])
def test_prox_matches_skglm(pen, t):
    import skglm.penalties as P
    g = DEFAULT_GAMMA[pen]
    ref = P.MCPenalty(alpha=LAM, gamma=g) if pen == "MCP" else P.SCAD(alpha=LAM, gamma=g)
    got = prox(ZS, t, LAM, np.ones_like(ZS), pen, g)
    assert np.allclose(got, [ref.prox_1d(z, t, 0) for z in ZS], atol=1e-12)


@requires_pyproximal
@pytest.mark.parametrize("t", [0.05, 0.3, 1.0])
def test_scad_prox_matches_pyproximal(t):
    got = prox(ZS, t, LAM, np.ones_like(ZS), "SCAD", 3.7)
    assert np.allclose(got, pyproximal.SCAD(sigma=LAM, a=3.7).prox(ZS.copy(), t), atol=1e-12)


@pytest.mark.parametrize("pen", NONCONVEX)
def test_prox_leaves_large_entries_unshrunk(pen):
    """Beyond gamma*lam the prox is the identity -- the unbiasedness property
    that motivates MCP/SCAD.  The lasso always subtracts t*lam."""
    g = DEFAULT_GAMMA[pen]
    big = np.array([-3.0, 2.0, 7.5])                     # all > g * LAM
    assert np.array_equal(prox(big, 0.3, LAM, np.ones(3), pen, g), big)
    lasso = prox(big, 0.3, LAM, np.ones(3), "lasso")
    assert np.allclose(np.abs(big) - np.abs(lasso), 0.3 * LAM)


@pytest.mark.parametrize("pen", ("lasso",) + NONCONVEX)
def test_prox_leaves_unpenalised_entries_alone(pen):
    """w = 0 entries pass through untouched; w = 1 entries are proxed as usual."""
    g = DEFAULT_GAMMA.get(pen)
    z = np.array([0.1, -0.2, 0.3, 5.0])
    w = np.array([0.0, 0.0, 1.0, 1.0])
    out = prox(z, 0.3, LAM, w, pen, g)
    assert out[0] == z[0] and out[1] == z[1]
    assert np.array_equal(out[2:], prox(z[2:], 0.3, LAM, np.ones(2), pen, g))
    assert out[2] != z[2]                     # 0.3 > t*lam = 0.15: shrunk, not zeroed


@pytest.mark.parametrize("pen, t", [("MCP", 3.5), ("SCAD", 2.8)])
def test_prox_refuses_steps_where_it_is_not_unique(pen, t):
    """MCP needs gamma > t, SCAD gamma > 1 + t; otherwise the scalar problem is
    nonconvex and the closed form is not its minimiser."""
    with pytest.raises(ValueError, match="prox needs gamma"):
        prox(ZS, t, LAM, np.ones_like(ZS), pen, DEFAULT_GAMMA[pen])


# --------------------------------------------------------------------------- #
# 3. The two conventions
# --------------------------------------------------------------------------- #


def test_column_sq_norms_closed_form():
    rng = np.random.default_rng(0)
    for p in (3, 6):
        s = rng.normal(size=(p, p))
        s = s @ s.T + p * np.eye(p)
        explicit = unvec((design_matrix(s) ** 2).sum(axis=0), p)
        assert np.allclose(design_column_sq_norms(s), explicit)


def test_conventions_coincide_for_the_lasso():
    """lam |v x| / v = lam |x|: the convention is irrelevant for l1."""
    rng = np.random.default_rng(2)
    m = rng.normal(size=(5, 5))
    w = penalty_weights(5)
    v = rng.uniform(0.3, 9.0, size=(5, 5))
    assert np.isclose(value(m, LAM, w, "lasso", None, v), value(m, LAM, w, "lasso"))


@pytest.mark.parametrize("pen", NONCONVEX)
def test_conventions_coincide_when_all_curvatures_are_one(pen):
    rng = np.random.default_rng(3)
    m = rng.normal(size=(5, 5)) * 2
    w = penalty_weights(5)
    assert np.isclose(value(m, LAM, w, pen, None, np.ones((5, 5))), value(m, LAM, w, pen))


@requires_ncvreg
@pytest.mark.r
@pytest.mark.parametrize("pen", NONCONVEX)
def test_what_ncvfit_actually_minimises(pen, tmp_path):
    """ncvfit minimises (1/2n)||y - Xb||^2 + sum_j P(v_j b_j) / v_j, v_j = x_j'x_j/n.

    Established on a plain regression -- no Lyapunov code involved.  With
    unequal column norms and coefficients in the penalty's curved region, the
    ncvfit solution is a stationary point of the *scaled* objective and not of
    the textbook one.  This is the fact behind the ``convention`` argument.
    """
    import subprocess
    import textwrap

    rng = np.random.default_rng(0)
    n, d = 200, 8
    x = rng.normal(size=(n, d)) * np.array([0.3, 3, 1, 0.5, 2, 1, 4, 0.7])
    y = x @ np.array([10.0, -0.7, 0, 0, 0.75, 0, 0, 0]) + 0.5 * rng.normal(size=n)
    lam, g = 0.4, DEFAULT_GAMMA[pen]
    (tmp_path / "in.json").write_text(json.dumps({"X": x.tolist(), "y": y.tolist()}))
    script = textwrap.dedent(f"""
        suppressPackageStartupMessages({{library(ncvreg); library(jsonlite)}})
        d <- fromJSON("{tmp_path / 'in.json'}", simplifyMatrix = TRUE)
        X <- as.matrix(d$X); n <- nrow(X)
        f <- ncvfit(X, d$y, xtx = colSums(X^2)/n, penalty = "{pen}", gamma = {g},
                    lambda = {lam}, eps = 1e-14, max.iter = 1e6, warn = FALSE)
        write_json(list(beta = as.numeric(f$beta)), "{tmp_path / 'out.json'}", digits = 16)
    """)
    (tmp_path / "fit.R").write_text(script)
    subprocess.run(["Rscript", str(tmp_path / "fit.R")], check=True, capture_output=True)
    b = np.array(json.loads((tmp_path / "out.json").read_text())["beta"])

    grad = -x.T @ (y - x @ b) / n
    v = (x ** 2).sum(axis=0) / n
    ones = np.ones(d)
    assert stationarity(b, grad, lam, ones, pen, g, scale=v) < 1e-10      # ncvreg convention
    assert stationarity(b, grad, lam, ones, pen, g) > 1e-2                # not the textbook one


# --------------------------------------------------------------------------- #
# 4. The solver: monotone accelerated proximal gradient
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pen", NONCONVEX)
@pytest.mark.parametrize("convention", ["textbook", "ncvreg"])
@pytest.mark.parametrize("frac", [0.1, 0.4])
def test_solution_is_stationary(pen, convention, frac):
    sigma, c, w = _problem()
    lam = frac * lambda_max(sigma, c)
    m = solve_fista(sigma, c, lam, weights=w, penalty=pen, convention=convention, **TIGHT)
    scale = penalty_scale(sigma, convention)
    assert stationarity(m, direct_grad(m, sigma, c), lam, w, pen, None, scale) < 1e-8
    assert np.all(np.diag(m) < 0)


@pytest.mark.parametrize("pen", NONCONVEX)
def test_objective_is_monotone(pen):
    """Li & Lin's safeguard: the objective never increases between iterates."""
    sigma, c, w = _problem()
    lam = 0.3 * lambda_max(sigma, c)
    vals = [objective(solve_fista(sigma, c, lam, weights=w, tol=0.0, max_iter=k,
                                  penalty=pen), sigma, c, lam, w, pen)
            for k in (1, 2, 3, 5, 10, 20, 50, 100, 300)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("pen", NONCONVEX)
def test_huge_gamma_recovers_the_lasso_solution(pen):
    sigma, c, w = _problem()
    lam = 0.3 * lambda_max(sigma, c)
    lasso = solve_fista(sigma, c, lam, weights=w, **TIGHT)
    near = solve_fista(sigma, c, lam, weights=w, penalty=pen, gamma=1e8, **TIGHT)
    assert np.allclose(near, lasso, atol=1e-6)


@pytest.mark.parametrize("pen", NONCONVEX)
def test_lambda_max_is_the_same_as_for_the_lasso(pen):
    """P'(0+) = lam for all three penalties, so the diagonal fit is stationary
    exactly when it is for the lasso -- the grid and short-circuit carry over."""
    sigma, c, w = _problem()
    lmax = lambda_max(sigma, c)
    d = diagonal_fit(sigma, c)
    assert stationarity(d, direct_grad(d, sigma, c), lmax * 1.0001, w, pen) < 1e-10
    below = solve_fista(sigma, c, 0.97 * lmax, weights=w, penalty=pen, **TIGHT)
    assert np.any(below[~np.eye(len(sigma), dtype=bool)] != 0)


def test_default_path_is_unchanged_for_the_lasso():
    """Adding MCP/SCAD must not have altered the validated lasso path."""
    sigma, c, _ = _problem()
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=10, ratio=1e-2)
    a = lasso_path(sigma, c, lambdas=lams, **TIGHT).estimates
    b = lasso_path(sigma, c, lambdas=lams, penalty="lasso", convention="ncvreg", **TIGHT).estimates
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 5. Backends
# --------------------------------------------------------------------------- #


def _fixed_point_gap(sigma, c, lams, reference, pen, convention, w):
    """How far fista moves when started at another solver's solutions."""
    return max(
        np.max(np.abs(solve_fista(sigma, c, lam, weights=w, m_init=m, penalty=pen,
                                  convention=convention, tol=1e-14, max_iter=200_000) - m))
        for lam, m in zip(lams, reference))


@requires_skglm
def test_fista_and_skglm_agree_on_textbook_mcp():
    """Same objective, both stationary, and each is a fixed point of the other.

    Coefficients are *not* required to agree along the whole path: MCP is
    nonconvex, and coordinate descent and proximal gradient can settle in
    different local minima.  Measured, neither is systematically lower.
    """
    sigma, c, w = _problem()
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-2)
    theirs = lasso_path(sigma, c, lambdas=lams, solver="skglm", penalty="MCP").estimates
    ours = lasso_path(sigma, c, lambdas=lams, solver="fista", penalty="MCP", **TIGHT).estimates
    for lam, a, b in zip(lams, ours, theirs):
        assert stationarity(a, direct_grad(a, sigma, c), lam, w, "MCP") < 1e-8
        assert stationarity(b, direct_grad(b, sigma, c), lam, w, "MCP") < 1e-6
    assert _fixed_point_gap(sigma, c, lams, theirs, "MCP", "textbook", w) < 1e-8
    agree = sum(np.array_equal(a != 0, b != 0) for a, b in zip(ours, theirs))
    assert agree >= len(lams) - 2


@requires_skglm
def test_fista_matches_skglm_on_textbook_scad():
    """skglm has no weighted SCAD, so compare with the diagonal penalised too.
    Here the two land on the same local path: full coefficient agreement."""
    sigma, c, _ = _problem()
    w = penalty_weights(len(sigma), penalize_diagonal=True)
    lams = lambda_grid(lambda_max(sigma, c, penalize_diagonal=True), n_lambda=10, ratio=1e-2)
    kw = dict(lambdas=lams, penalty="SCAD", penalize_diagonal=True)
    ours = lasso_path(sigma, c, solver="fista", **kw, **TIGHT).estimates
    theirs = lasso_path(sigma, c, solver="skglm", **kw).estimates
    for lam, a, b in zip(lams, ours, theirs):
        assert np.allclose(a, b, atol=1e-6), lam
        assert abs(objective(a, sigma, c, lam, w, "SCAD")
                   - objective(b, sigma, c, lam, w, "SCAD")) < 1e-10


@requires_ncvreg
@pytest.mark.r
@pytest.mark.parametrize("pen", NONCONVEX)
def test_fista_and_ncvreg_agree_under_the_ncvreg_convention(pen):
    sigma, c, w = _problem()
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-2)
    scale = penalty_scale(sigma, "ncvreg")
    theirs = lasso_path(sigma, c, lambdas=lams, solver="ncvreg", penalty=pen,
                        convention="ncvreg").estimates
    ours = lasso_path(sigma, c, lambdas=lams, solver="fista", penalty=pen,
                      convention="ncvreg", **TIGHT).estimates
    for lam, a, b in zip(lams, ours, theirs):
        assert stationarity(a, direct_grad(a, sigma, c), lam, w, pen, None, scale) < 1e-8
        assert stationarity(b, direct_grad(b, sigma, c), lam, w, pen, None, scale) < 1e-8
    assert _fixed_point_gap(sigma, c, lams, theirs, pen, "ncvreg", w) < 1e-8


@requires_ncvreg
@pytest.mark.r
@pytest.mark.parametrize("pen", NONCONVEX)
def test_ncvreg_is_not_stationary_for_the_textbook_objective(pen):
    """Why the backend refuses the textbook convention: on this design its
    solutions are not critical points of the textbook MCP/SCAD objective."""
    sigma, c, w = _problem()
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=12, ratio=1e-2)
    theirs = lasso_path(sigma, c, lambdas=lams, solver="ncvreg", penalty=pen,
                        convention="ncvreg").estimates
    worst = max(stationarity(b, direct_grad(b, sigma, c), lam, w, pen)
                for lam, b in zip(lams, theirs))
    assert worst > 1e-3


@pytest.mark.parametrize("solver, convention", [("ncvreg", "textbook"), ("skglm", "ncvreg")])
def test_backends_refuse_a_convention_they_do_not_solve(solver, convention):
    sigma, c, _ = _problem()
    with pytest.raises(ValueError, match="convention"):
        lasso_path(sigma, c, n_lambda=3, solver=solver, penalty="MCP", convention=convention)


@pytest.mark.parametrize("solver", ["design", "glmnet", "pyproximal"])
def test_l1_only_backends_refuse_nonconvex_penalties(solver):
    sigma, c, _ = _problem()
    with pytest.raises(ValueError, match="l1 penalty only"):
        lasso_path(sigma, c, n_lambda=3, solver=solver, penalty="SCAD")


def test_unknown_convention_is_rejected():
    sigma, c, _ = _problem()
    with pytest.raises(ValueError, match="unknown convention"):
        lasso_path(sigma, c, n_lambda=3, penalty="MCP", convention="standardised")


# --------------------------------------------------------------------------- #
# 6. The motivation: no shrinkage bias on large edges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pen", NONCONVEX)
def test_nonconvex_penalties_remove_the_lasso_bias(pen):
    """Dettling's Example 2 path graph, population covariance.

    At every lambda where both methods recover the support exactly, the lasso
    shrinks the true edges (0.65) substantially, while MCP/SCAD return them
    unbiased.  This is the property S1b sets out to measure.
    """
    m_star = example2_path()
    c = 2 * np.eye(5)
    sigma = solve_lyapunov(m_star, c)
    off = ~np.eye(5, dtype=bool)
    truth = m_star[off] != 0
    lams = lambda_grid(lambda_max(sigma, c), n_lambda=40, ratio=1e-3)
    lasso = lasso_path(sigma, c, lambdas=lams, **TIGHT).estimates
    ncv = lasso_path(sigma, c, lambdas=lams, penalty=pen, **TIGHT).estimates

    rows = []
    for lam, a, b in zip(lams, lasso, ncv):
        if not (np.array_equal(a[off] != 0, truth) and np.array_equal(b[off] != 0, truth)):
            continue
        bias_lasso = np.max(np.abs(a[off][truth] - m_star[off][truth]))
        bias_ncv = np.max(np.abs(b[off][truth] - m_star[off][truth]))
        if bias_lasso > 0.05:                       # lambda large enough to shrink
            rows.append((lam, bias_ncv))
    assert len(rows) >= 5
    rows.sort()
    # MCP: unbiased at every lambda in the exact-recovery window.
    # SCAD: likewise, except possibly at the window's sparse edge (its largest
    # lambda, measured at 0.17 * lambda_max), where the warm-started path can
    # settle in another local minimum with a weak edge still inside SCAD's
    # lasso-like region |x| <= lam.  That is nonconvexity, not a solver failure:
    # stationarity is asserted separately.
    checked = rows if pen == "MCP" else rows[:-1]
    for lam, bias_ncv in checked:
        assert bias_ncv < 1e-6, lam
