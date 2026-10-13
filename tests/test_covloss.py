"""Losses on the implied covariance (Varando & Hansen 2020) and their solvers.

Organised from the smallest piece outwards, each against something that shares
no code with the piece under test:

1. the Lyapunov solves on one Schur factorisation   vs scipy, vs the equations
2. the losses and their Sigma-gradients              vs slogdet / finite differences
3. the M-gradient (adjoint equation)                  vs finite differences, vs Prop. 3.1
4. the exact Hessian on an index set                  vs finite differences
5. the two ends of the path                           closed forms, lambda_max
6. the solvers                                        stationarity, descent, limits,
                                                      scaling identity, prox == newton
7. the gclm R package                                 objective scale, fixed points,
                                                      same-basin agreement
8. the path                                           shapes, order, diagnostics

See docs/LIKELIHOOD.md for the definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from gclm.lyapunov import SchurLyapunov
from gclm.objective.covariance import (
    LOSSES,
    dense_fit,
    diagonal_fit,
    hessian,
    lambda_max,
    loss_grad,
    loss_value,
    objective,
    sigma_loss,
    sigma_loss_grad,
)
from gclm.solvers.covariance import solve
from gclm.solvers.path import covloss_path
from gclm.data.simulate import CChoice, draw_instance
from gclm.objective.penalties import penalty_weights
from gclm.lyapunov import is_stable, solve_lyapunov
from gclm.objective.penalties import stationarity, value

from conftest import r_available, run_r

GCLM_AVAILABLE = r_available("gclm", "jsonlite")
requires_gclm = pytest.mark.skipif(not GCLM_AVAILABLE, reason="R package gclm not installed")

PENALTIES = ("lasso", "MCP", "SCAD")


def _instance(p=6, k=2, n=500, seed=7):
    rng = np.random.default_rng(seed)
    m_true, _, _, s = draw_instance(p, k, n, CChoice.ID, rng, standardize=True)
    return m_true, s, 2.0 * np.eye(p)


def _random_stable(p, rng):
    m = rng.normal(size=(p, p))
    np.fill_diagonal(m, -np.abs(m).sum(axis=1) - 0.5)
    return m


def _perturbed(m, rng, scale=0.1):
    """``m`` plus noise, with the noise halved until the result is stable."""
    e = rng.normal(size=m.shape)
    while not is_stable(m + scale * e):
        scale /= 2
    return m + scale * e


# --------------------------------------------------------------------------- #
# 1. Lyapunov solves on the Schur factorisation
# --------------------------------------------------------------------------- #


def test_schur_sigma_matches_scipy():
    rng = np.random.default_rng(0)
    for p in (3, 7, 12):
        m, c = _random_stable(p, rng), np.diag(rng.uniform(0.5, 2, p))
        fac = SchurLyapunov(m)
        assert fac.stable
        np.testing.assert_allclose(fac.sigma(c), solve_lyapunov(m, c), atol=1e-12)


def test_schur_adjoint_solves_its_equation():
    rng = np.random.default_rng(1)
    m = _random_stable(8, rng)
    g = rng.normal(size=(8, 8))
    g = g + g.T
    z = SchurLyapunov(m).solve_adjoint(g)
    np.testing.assert_allclose(m.T @ z + z @ m, g, atol=1e-12)


def test_schur_stability_flag():
    rng = np.random.default_rng(2)
    m = _random_stable(6, rng)
    assert SchurLyapunov(m).stable == is_stable(m)
    m[0, 0] = 10.0
    assert SchurLyapunov(m).stable == is_stable(m) is False


def test_varando_adjoint_identity():
    """Prop. 3.1's key step: tr(Sigma(B, C) D) = tr(C Sigma(B', D))."""
    rng = np.random.default_rng(3)
    b = _random_stable(5, rng)
    c, d = (lambda x: x @ x.T)(rng.normal(size=(5, 5))), (lambda x: x @ x.T)(rng.normal(size=(5, 5)))
    lhs = np.trace(solve_lyapunov(b, c) @ d)
    rhs = np.trace(c @ solve_lyapunov(b.T, d))
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)


# --------------------------------------------------------------------------- #
# 2. losses as functions of Sigma
# --------------------------------------------------------------------------- #


def test_loglik_value_is_logdet_plus_trace():
    _, s_hat, _ = _instance()
    rng = np.random.default_rng(4)
    a = rng.normal(size=(6, 6))
    sig = a @ a.T + np.eye(6)
    expect = np.linalg.slogdet(sig)[1] + np.trace(np.linalg.solve(sig, s_hat))
    assert abs(sigma_loss(sig, s_hat, "loglik") - expect) < 1e-12
    assert sigma_loss(sig, s_hat, "frobenius") == pytest.approx(0.5 * np.sum((sig - s_hat) ** 2))


def test_loglik_not_pd_is_inf():
    _, s_hat, _ = _instance()
    assert sigma_loss(-np.eye(6), s_hat, "loglik") == np.inf


@pytest.mark.parametrize("loss", LOSSES)
def test_sigma_gradient_matches_finite_differences(loss):
    _, s_hat, _ = _instance()
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 6))
    sig = a @ a.T + np.eye(6)
    g = sigma_loss_grad(sig, s_hat, loss)
    for _ in range(3):
        e = rng.normal(size=(6, 6))
        e = e + e.T
        h = 1e-6
        fd = (sigma_loss(sig + h * e, s_hat, loss) - sigma_loss(sig - h * e, s_hat, loss)) / (2 * h)
        assert abs(fd - np.sum(g * e)) < 1e-6 * max(1.0, abs(fd))


# --------------------------------------------------------------------------- #
# 3. the M-gradient
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", LOSSES)
def test_m_gradient_matches_finite_differences(loss):
    m_true, s_hat, c = _instance()
    rng = np.random.default_rng(6)
    m = _perturbed(m_true, rng)
    g = loss_grad(m, s_hat, c, loss)
    for _ in range(4):
        e = rng.normal(size=m.shape)
        h = 1e-6
        fd = (loss_value(m + h * e, s_hat, c, loss) - loss_value(m - h * e, s_hat, c, loss)) / (2 * h)
        assert abs(fd - np.sum(g * e)) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("loss", LOSSES)
def test_m_gradient_is_minus_two_z_sigma(loss):
    """The closed form of Prop. 3.1 (with gclm's operand order, see docs)."""
    m_true, s_hat, c = _instance()
    sig = solve_lyapunov(m_true, c)
    g = sigma_loss_grad(sig, s_hat, loss)
    z = solve_lyapunov(m_true.T, -g)            # M' Z + Z M = G
    np.testing.assert_allclose(loss_grad(m_true, s_hat, c, loss), -2.0 * z @ sig, atol=1e-10)


def test_gradient_raises_for_unstable_m():
    _, s_hat, c = _instance()
    with pytest.raises(ValueError):
        loss_grad(np.eye(6), s_hat, c, "loglik")


# --------------------------------------------------------------------------- #
# 4. the Hessian
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", LOSSES)
def test_hessian_matches_finite_differences(loss):
    m_true, s_hat, c = _instance()
    p = m_true.shape[0]
    rng = np.random.default_rng(8)
    m = _perturbed(m_true, rng)
    rows, cols = np.nonzero(np.ones((p, p)))
    h = hessian(m, s_hat, c, loss, rows, cols)
    assert np.allclose(h, h.T)
    step = 1e-6
    fd = np.zeros((p * p, p * p))
    for b in range(p * p):
        e = np.zeros((p, p))
        e[rows[b], cols[b]] = step
        fd[:, b] = ((loss_grad(m + e, s_hat, c, loss) - loss_grad(m - e, s_hat, c, loss))
                    / (2 * step))[rows, cols]
    assert np.abs(h - fd).max() < 1e-6 * np.abs(fd).max()


def test_hessian_on_subset_is_a_submatrix():
    m_true, s_hat, c = _instance()
    p = m_true.shape[0]
    rows, cols = np.nonzero(np.ones((p, p)))
    full = hessian(m_true, s_hat, c, "loglik", rows, cols)
    idx = np.array([0, 5, 7, 20, 33])
    sub = hessian(m_true, s_hat, c, "loglik", rows[idx], cols[idx])
    np.testing.assert_allclose(sub, full[np.ix_(idx, idx)], atol=1e-12)


# --------------------------------------------------------------------------- #
# 5. the two ends of the path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", LOSSES)
def test_diagonal_fit_is_the_diagonal_minimiser(loss):
    _, s_hat, c = _instance()
    md = diagonal_fit(s_hat, c)
    g = loss_grad(md, s_hat, c, loss)
    assert np.abs(np.diag(g)).max() < 1e-10            # stationary over diagonal M
    f0 = loss_value(md, s_hat, c, loss)
    rng = np.random.default_rng(9)
    for _ in range(5):                                 # and a minimum, not a saddle
        d = np.diag(rng.normal(size=6) * 0.3)
        assert loss_value(md + d, s_hat, c, loss) > f0


def test_diagonal_fit_needs_diagonal_c():
    _, s_hat, c = _instance()
    c[0, 1] = c[1, 0] = 0.1
    with pytest.raises(ValueError):
        diagonal_fit(s_hat, c)


@pytest.mark.parametrize("loss", LOSSES)
def test_dense_fit_is_a_global_minimiser(loss):
    _, s_hat, c = _instance()
    m0 = dense_fit(s_hat, c)
    assert is_stable(m0)
    np.testing.assert_allclose(solve_lyapunov(m0, c), s_hat, atol=1e-12)
    assert np.abs(loss_grad(m0, s_hat, c, loss)).max() < 1e-10
    expect = np.linalg.slogdet(s_hat)[1] + 6 if loss == "loglik" else 0.0
    assert abs(loss_value(m0, s_hat, c, loss) - expect) < 1e-10


@pytest.mark.parametrize("loss", LOSSES)
def test_population_truth_is_stationary(loss):
    """With Sigma_hat = Sigma(M*, C) exactly, M* has zero gradient and the
    global minimum -- as any point on the fibre {Sigma(M) = Sigma_hat} does."""
    m_true, _, c = _instance()
    s_pop = solve_lyapunov(m_true, c)
    assert np.abs(loss_grad(m_true, s_pop, c, loss)).max() < 1e-10
    assert loss_value(m_true, s_pop, c, loss) <= loss_value(dense_fit(s_pop, c), s_pop, c, loss) + 1e-10


@pytest.mark.parametrize("loss", LOSSES)
def test_lambda_max_separates_diagonal_from_nondiagonal(loss):
    _, s_hat, c = _instance()
    lm = lambda_max(s_hat, c, loss)
    md = diagonal_fit(s_hat, c)
    above = solve(s_hat, c, 1.01 * lm, loss, m_init=md)
    np.testing.assert_allclose(above, md, atol=1e-12)   # KKT holds: nothing moves
    below = solve(s_hat, c, 0.9 * lm, loss, m_init=md)
    assert np.count_nonzero(below - np.diag(np.diag(below))) > 0


@pytest.mark.parametrize("loss", LOSSES)
def test_lambda_max_is_the_largest_correlation(loss):
    """Closed form for correlation input and C = 2I: at M_D = -I both
    Sigma-gradients are I - Sigma_hat and grad_M = I - Sigma_hat, so
    lambda_max = max_{i != j} |Sigma_hat_ij| -- the graphical lasso's."""
    _, s_hat, c = _instance()
    assert np.allclose(np.diag(s_hat), 1.0)
    np.testing.assert_allclose(diagonal_fit(s_hat, c), -np.eye(6), atol=1e-14)
    off = ~np.eye(6, dtype=bool)
    assert lambda_max(s_hat, c, loss) == pytest.approx(np.abs(s_hat[off]).max(), rel=1e-12)


def test_lambda_max_equal_across_losses_for_correlation_input():
    """At the diagonal fit Sigma = I for standardised data, where both
    Sigma-gradients are I - Sigma_hat; hence the same lambda_max."""
    _, s_hat, c = _instance()
    assert lambda_max(s_hat, c, "loglik") == pytest.approx(lambda_max(s_hat, c, "frobenius"), rel=1e-12)


# --------------------------------------------------------------------------- #
# 6. the solvers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("penalty", PENALTIES)
@pytest.mark.parametrize("method", ("newton", "prox"))
def test_solver_reaches_a_stationary_point(loss, penalty, method):
    _, s_hat, c = _instance()
    w = penalty_weights(6)
    lam = 0.1 * lambda_max(s_hat, c, loss)
    tol = 1e-8 if method == "newton" else 1e-6
    m, info = solve(s_hat, c, lam, loss, penalty=penalty, method=method, tol=tol, return_info=True)
    assert info.converged
    assert is_stable(m)
    assert stationarity(m, loss_grad(m, s_hat, c, loss), lam, w, penalty) <= 10 * tol
    assert info.objective == pytest.approx(objective(m, s_hat, c, lam, w, loss, penalty), abs=1e-12)
    # a descent method: below the start, which is the diagonal fit
    assert info.objective < objective(diagonal_fit(s_hat, c), s_hat, c, lam, w, loss, penalty)
    assert np.all(np.diag(m) < 0)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("penalty", PENALTIES)
def test_newton_polish_leaves_a_converged_prox_solution_alone(loss, penalty):
    """Both methods minimise the same function: the proximal solution is a
    fixed point of the Newton solver."""
    _, s_hat, c = _instance()
    w = penalty_weights(6)
    lam = 0.2 * lambda_max(s_hat, c, loss)
    m_prox = solve(s_hat, c, lam, loss, penalty=penalty, method="prox", tol=1e-9)
    m_newt = solve(s_hat, c, lam, loss, penalty=penalty, method="newton", m_init=m_prox, tol=1e-10)
    f_prox = objective(m_prox, s_hat, c, lam, w, loss, penalty)
    f_newt = objective(m_newt, s_hat, c, lam, w, loss, penalty)
    assert f_prox - 1e-9 <= f_newt <= f_prox + 1e-12
    assert np.abs(m_newt - m_prox).max() < 1e-5


@pytest.mark.parametrize("loss", LOSSES)
def test_mcp_with_huge_gamma_is_the_lasso(loss):
    _, s_hat, c = _instance()
    lam = 0.2 * lambda_max(s_hat, c, loss)
    m_l1 = solve(s_hat, c, lam, loss, penalty="lasso", tol=1e-10)
    m_mcp = solve(s_hat, c, lam, loss, penalty="MCP", gamma=1e8, tol=1e-10)
    np.testing.assert_allclose(m_mcp, m_l1, atol=1e-6)


@pytest.mark.parametrize("loss", LOSSES)
def test_scaling_c_rescales_lambda_for_the_lasso(loss):
    """Sigma(cM, cC) = Sigma(M, C), so M_hat(lam; 2I) = 2 M_hat(2 lam; I) for the
    lasso.  (Not for MCP/SCAD: their gamma is measured on the scale of M.)"""
    _, s_hat, _ = _instance()
    lam = 0.1 * lambda_max(s_hat, 2 * np.eye(6), loss)
    # Newton: its iterates are equivariant under the rescaling, so both runs
    # converge to the same stationary point; a first-order method's step-size
    # sequence is not, and the problem has more than one stationary point.
    m2 = solve(s_hat, 2 * np.eye(6), lam, loss, method="newton", tol=1e-10)
    m1 = solve(s_hat, np.eye(6), 2 * lam, loss, method="newton", tol=1e-10)
    np.testing.assert_allclose(m2, 2 * m1, atol=1e-7)


def test_objective_is_loss_plus_penalty():
    m_true, s_hat, c = _instance()
    w = penalty_weights(6)
    for pen in PENALTIES:
        expect = loss_value(m_true, s_hat, c, "loglik") + value(m_true, 0.3, w, pen)
        assert objective(m_true, s_hat, c, 0.3, w, "loglik", pen) == pytest.approx(expect)


def test_solver_rejects_unstable_start():
    _, s_hat, c = _instance()
    with pytest.raises(ValueError):
        solve(s_hat, c, 0.1, m_init=np.eye(6))


# --------------------------------------------------------------------------- #
# 7. the gclm R package
# --------------------------------------------------------------------------- #


def _gclm(s_hat, c, b0, lams, loss, eps=0.0, max_iter=200_000):
    return run_r("backend_gclm.R", {
        "Sigma": s_hat.tolist(), "C": np.diag(c).tolist(), "B0": b0.tolist(),
        "lambda": list(map(float, lams)), "loss": loss, "eps": eps, "maxIter": max_iter,
    }, packages=("gclm", "jsonlite"))


@requires_gclm
@pytest.mark.parametrize("loss", LOSSES)
def test_gclm_objective_scale_matches(loss):
    """gclm's reported loss plus lam * ||B_off||_1 is our objective at its B."""
    _, s_hat, c = _instance()
    w = penalty_weights(6)
    lams = [0.3, 0.05]
    out = _gclm(s_hat, c, diagonal_fit(s_hat, c), lams, loss, eps=1e-6, max_iter=50)
    for i, lam in enumerate(lams):
        b = np.array(out["B"][i])
        ours = objective(b, s_hat, c, lam, w, loss)
        theirs = out["loss"][i] + value(b, lam, w, "lasso")
        assert abs(ours - theirs) < 1e-9 * max(1.0, abs(ours))


@requires_gclm
@pytest.mark.parametrize("loss", LOSSES)
def test_gclm_converged_points_are_fixed_points_of_our_solver(loss):
    """Where gclm reaches a stationary point (by our KKT measure), the Newton
    solver started there does not move -- the two codes agree on the problem."""
    _, s_hat, c = _instance()
    w = penalty_weights(6)
    lm = lambda_max(s_hat, c, loss)
    lams = [0.1 * lm, 0.02 * lm]
    out = _gclm(s_hat, c, diagonal_fit(s_hat, c), lams, loss)
    checked = 0
    for i, lam in enumerate(lams):
        b = np.array(out["B"][i])
        if stationarity(b, loss_grad(b, s_hat, c, loss), lam, w) > 1e-6:
            continue                                   # gclm stalled here (its stopping rule)
        m = solve(s_hat, c, lam, loss, m_init=b, tol=1e-10)
        assert objective(b, s_hat, c, lam, w, loss) - objective(m, s_hat, c, lam, w, loss) < 1e-8
        assert np.abs(m - b).max() < 1e-3                # flat valley: B less precise than F
        checked += 1
    assert checked >= 1


@requires_gclm
@pytest.mark.parametrize("loss", LOSSES)
def test_same_basin_agreement_with_gclm(loss):
    """From the diagonal fit at a small lambda both codes converge to the same
    point (same basin): objective to 1e-6, B to 1e-2 (the valley is flat, so
    gclm's looser stopping leaves B less precise than F), supports differing
    in at most one near-threshold entry."""
    _, s_hat, c = _instance()
    w = penalty_weights(6)
    lam = 0.02 * lambda_max(s_hat, c, loss)
    b = np.array(_gclm(s_hat, c, diagonal_fit(s_hat, c), [lam], loss)["B"][0])
    m = solve(s_hat, c, lam, loss, tol=1e-10)
    assert abs(objective(m, s_hat, c, lam, w, loss) - objective(b, s_hat, c, lam, w, loss)) < 1e-6
    assert np.abs(m - b).max() < 1e-2
    assert np.sum((m != 0) != (b != 0)) <= 1


# --------------------------------------------------------------------------- #
# 8. the path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", LOSSES)
def test_path_shapes_order_and_diagnostics(loss):
    _, s_hat, c = _instance()
    path = covloss_path(s_hat, c, loss, n_lambda=12, tol=1e-8)
    assert len(path.estimates) == 12 and len(path.lambdas) == 12
    assert np.all(np.diff(path.lambdas) > 0)
    assert path.lambdas[-1] == pytest.approx(lambda_max(s_hat, c, loss))
    assert path.converged.all() and path.kkt.max() <= 10 * 1e-8   # KKT ~ gradient mapping
    assert np.all(np.diff(path.objective) > 0)        # F*(lam) increases with lam
    np.testing.assert_allclose(path.estimates[-1], diagonal_fit(s_hat, c), atol=1e-12)
    for m in path.estimates:
        assert is_stable(m)
    sup = path.supports()
    assert not sup[-1].any() and sup[0].sum() >= sup[-1].sum()


@pytest.mark.parametrize("loss", LOSSES)
def test_path_up_starts_dense(loss):
    _, s_hat, c = _instance()
    path = covloss_path(s_hat, c, loss, n_lambda=8, direction="up", tol=1e-8)
    assert path.converged.all()
    assert path.supports()[0].sum() > path.supports()[-1].sum()


@pytest.mark.parametrize("penalty", ("MCP", "SCAD"))
def test_path_nonconvex_penalties(penalty):
    _, s_hat, c = _instance()
    path = covloss_path(s_hat, c, "loglik", n_lambda=10, penalty=penalty, tol=1e-8)
    assert path.converged.all()
    w = penalty_weights(6)
    for lam, m in zip(path.lambdas, path.estimates):
        assert stationarity(m, loss_grad(m, s_hat, c, "loglik"), lam, w, penalty) <= 1e-7


def test_path_rejects_bad_arguments():
    _, s_hat, c = _instance()
    with pytest.raises(ValueError):
        covloss_path(s_hat, c, "direct")
    with pytest.raises(ValueError):
        covloss_path(s_hat, c, "loglik", direction="sideways")
    with pytest.raises(ValueError):
        solve(s_hat, c, 0.1, method="cg")


# --------------------------------------------------------------------------- #
# Other C than 2I (code review, 12 October 2026): the campaign fits with the rescaled
# C = 2 diag(1/s^2), and the tests above use C = 2I only.
# --------------------------------------------------------------------------- #


def _instance_c(kind, p=6, seed=7):
    """Standardised data drawn with C_Random_Diag, and the C of the fit: the rescaled C
    (correct after standardising) or a random non-diagonal positive definite C."""
    from gclm.data.simulate import estimation_volatility
    rng = np.random.default_rng(seed)
    _, _, _, s, scale = draw_instance(p, 2, 500, CChoice.RANDOM_DIAG, rng, standardize=True,
                                      return_scale=True)
    if kind == "rescaled":
        return s, estimation_volatility(scale, "variance")
    a = rng.normal(size=(p, p))
    return s, a @ a.T / p + np.eye(p)


def _fd_grad(m, s, c, loss, h=1e-6):
    g = np.zeros_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            e = np.zeros_like(m)
            e[i, j] = h
            g[i, j] = (loss_value(m + e, s, c, loss) - loss_value(m - e, s, c, loss)) / (2 * h)
    return g


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("kind", ["rescaled", "full"])
def test_gradient_matches_finite_differences_for_other_c(loss, kind):
    s, c = _instance_c(kind)
    m = _perturbed(dense_fit(s, c), np.random.default_rng(1), 0.2)   # stable, non-symmetric, off the optimum
    np.testing.assert_allclose(loss_grad(m, s, c, loss), _fd_grad(m, s, c, loss), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("loss", LOSSES)
def test_lambda_max_closed_form_for_a_diagonal_c(loss):
    """Correlation input and a diagonal C: the diagonal fit has Sigma = I and m_i = -C_ii / 2,
    and off the diagonal the gradient there is 2 Sigma_hat_ij / (m_i + m_j) for both losses
    (the adjoint equation reads (m_i + m_j) Z_ij = G_ij with G_ij = -Sigma_hat_ij); with C = 2I
    this is the largest correlation."""
    s, c = _instance_c("rescaled")
    assert np.allclose(np.diag(s), 1.0)
    m = np.diag(diagonal_fit(s, c))
    np.testing.assert_allclose(m, -np.diag(c) / 2, rtol=1e-12)
    off = ~np.eye(len(m), dtype=bool)
    want = (2 * np.abs(s) / np.abs(m[:, None] + m[None, :]))[off].max()
    assert lambda_max(s, c, loss) == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("pen", PENALTIES)
def test_lambda_max_is_the_threshold_with_the_rescaled_c(loss, pen):
    """Just above lambda_max the diagonal fit does not move; below it the solution leaves the
    diagonal -- for every penalty, with the C of the campaign."""
    s, c = _instance_c("rescaled")
    lm = lambda_max(s, c, loss)
    md = diagonal_fit(s, c)
    above = solve(s, c, 1.001 * lm, loss, m_init=md, penalty=pen)
    np.testing.assert_allclose(above, md, atol=1e-10)
    below = solve(s, c, 0.95 * lm, loss, m_init=md, penalty=pen)
    assert np.count_nonzero(below - np.diag(np.diag(below))) > 0


# --------------------------------------------------------------------------- #
# The step test and the stopping rule (code review A1, 13 October 2026), and the
# degenerate inputs of code review D
# --------------------------------------------------------------------------- #


def _stiff_instance(ratio=100.0, p=6, seed=7):
    """Standardised data with a diagonal C whose entries span ``ratio``: the badly scaled
    problems the rescaled C produces, on which the old step test ran to the iteration cap."""
    _, s_hat, _ = _instance(p=p, seed=seed)
    return s_hat, np.diag(np.logspace(0.0, np.log10(ratio), p))


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("penalty", PENALTIES)
def test_a_stationary_start_is_returned_unchanged(loss, penalty):
    """A warm start that already solves the problem (the flat end of a dense -> sparse MCP path,
    for instance) is returned as it is: no step, zero iterations.  The old solver took one
    uncontrolled step at t = 1 from it and reported convergence."""
    _, s_hat, c = _instance()
    lam = 0.2 * lambda_max(s_hat, c, loss)
    m = solve(s_hat, c, lam, loss, penalty=penalty, tol=1e-10)
    again, info = solve(s_hat, c, lam, loss, penalty=penalty, m_init=m, tol=1e-8, return_info=True)
    assert info.converged and info.iterations == 0
    assert np.array_equal(again, m)
    assert info.stationarity <= 1e-8


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("penalty", ("lasso", "MCP"))
def test_badly_scaled_c_converges_well_below_the_iteration_cap(loss, penalty):
    """A C with a diagonal ratio of 100: every solve of a path, sparse -> dense and dense ->
    sparse from the exact fit, meets tol = 1e-8 on the first-order residual before the cap of
    50,000 steps the old test ran into (255 to 970 capped solves per campaign cell), and most
    solves need a few thousand steps at most (the dense end, where the loss is flat along the
    fibre of exact fits, legitimately needs more)."""
    s_hat, c = _stiff_instance()
    w = penalty_weights(6)
    for direction in ("down", "up"):
        path = covloss_path(s_hat, c, loss, n_lambda=12, penalty=penalty, direction=direction, tol=1e-8)
        assert path.converged.all(), direction
        assert path.iterations.max() < 50_000 and np.median(path.iterations) < 5_000, (direction, path.iterations)
        for lam, m, kkt in zip(path.lambdas, path.estimates, path.kkt):
            assert kkt <= 1e-8
            assert stationarity(m, loss_grad(m, s_hat, c, loss), lam, w, penalty) <= 1.01e-8


@pytest.mark.parametrize("loss", LOSSES)
def test_converged_means_the_first_order_residual_is_below_tol(loss):
    """The flag is judged on the accepted iterate itself, not on the size of the last step."""
    _, s_hat, c = _instance()
    for penalty in PENALTIES:
        path = covloss_path(s_hat, c, loss, n_lambda=10, penalty=penalty, tol=1e-8)
        assert path.converged.all() and np.all(path.kkt <= 1e-8)


def test_degenerate_inputs():
    """p = 1 has no off-diagonal entry (lambda_max undefined); a non-finite M is treated as
    unstable (loss +inf) instead of raising inside scipy; a grid of one value is lambda_max."""
    from gclm.solvers.path import lambda_grid
    _, s_hat, c = _instance()
    for loss in LOSSES:
        with pytest.raises(ValueError):
            lambda_max(np.eye(1), np.eye(1), loss)
        assert loss_value(np.full((6, 6), np.nan), s_hat, c, loss) == np.inf
    assert lambda_grid(2.0, n_lambda=1).tolist() == [2.0]
    with pytest.raises(ValueError):
        lambda_grid(2.0, n_lambda=0)
    assert np.all(np.diff(lambda_grid(2.0, n_lambda=5)) > 0) and lambda_grid(2.0, 5)[-1] == 2.0
