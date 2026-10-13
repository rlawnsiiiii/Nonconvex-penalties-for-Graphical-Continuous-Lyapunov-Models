"""Vectorization identities and the Lyapunov solve."""

from __future__ import annotations

import numpy as np
import pytest

from scipy.linalg import solve_continuous_lyapunov

from gclm.lyapunov import (
    SchurLyapunov,
    commutation_matrix,
    design_matrix,
    gram_matrix,
    is_stable,
    lyapunov_residual,
    solve_lyapunov,
    unvec,
    vec,
)

from conftest import requires_r, run_r


@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_vec_is_column_stacking(p):
    rng = np.random.default_rng(p)
    a = rng.normal(size=(p, p))
    assert np.array_equal(vec(a), a.T.ravel())          # columns stacked
    assert np.array_equal(unvec(vec(a), p), a)


@pytest.mark.parametrize("p", [2, 3, 5])
def test_commutation_matrix(p):
    rng = np.random.default_rng(p)
    a = rng.normal(size=(p, p))
    k = commutation_matrix(p)
    assert np.allclose(k @ vec(a), vec(a.T))
    assert np.allclose(k @ k, np.eye(p * p))            # K is an involution


@pytest.mark.parametrize("p", [2, 3, 6])
def test_design_matrix_vectorizes_lyapunov(p):
    """A(Sigma) vec(M) == vec(M Sigma + Sigma M')."""
    rng = np.random.default_rng(p)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    m = rng.normal(size=(p, p))
    assert np.allclose(design_matrix(sigma) @ vec(m), vec(m @ sigma + sigma @ m.T))


@pytest.mark.parametrize("p", [2, 4])
def test_gram_matrix_closed_form(p):
    """Dettling Lemma 1 against the explicit product A'A."""
    rng = np.random.default_rng(p)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    a = design_matrix(sigma)
    assert np.allclose(gram_matrix(sigma), a.T @ a)


@pytest.mark.parametrize("p", [3, 8])
def test_solve_lyapunov_residual(p):
    rng = np.random.default_rng(p)
    m = -np.eye(p) * (p + 1) + rng.normal(size=(p, p))
    c = rng.normal(size=(p, p))
    c = c @ c.T + p * np.eye(p)
    sigma = solve_lyapunov(m, c)
    assert is_stable(m)
    assert np.allclose(lyapunov_residual(m, sigma, c), 0.0, atol=1e-10)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)


@requires_r
@pytest.mark.r
@pytest.mark.parametrize("p", [3, 5])
def test_design_matrix_matches_r(p):
    """Our A(Sigma) is bit-for-bit R's `Sigma %x% I + (I %x% Sigma) %*% K`."""
    rng = np.random.default_rng(p)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    out = run_r(
        "backend_glmnet.R",
        {"Sigma": sigma.tolist(), "C": (2 * np.eye(p)).tolist(),
         "lambda_glmnet": [1.0], "return_design": True},
        ("glmnet", "jsonlite"),
    )
    assert np.allclose(design_matrix(sigma), np.asarray(out["A"]), atol=1e-12)


def test_schur_solves_match_scipy_for_symmetric_right_hand_sides():
    """SchurLyapunov.solve and solve_adjoint against scipy, for the symmetric right-hand sides
    the package uses (-C and dL/dSigma); the output is then symmetrised."""
    rng = np.random.default_rng(3)
    for p in (1, 2, 5, 9):
        m = rng.normal(size=(p, p))
        np.fill_diagonal(m, -np.abs(m).sum(axis=1) - 0.3)
        y = rng.normal(size=(p, p))
        y = y + y.T
        fac = SchurLyapunov(m)
        np.testing.assert_allclose(fac.solve(y), solve_continuous_lyapunov(m, y), atol=1e-10)
        np.testing.assert_allclose(fac.solve_adjoint(y), solve_continuous_lyapunov(m.T, y), atol=1e-10)


def test_schur_stability_flag_near_the_boundary():
    """The flag (largest diagonal entry of the real Schur form below zero) agrees with the
    eigenvalues for matrices whose spectral abscissa lies within 1e-6 to 1e-2 of zero, on
    either side, with real eigenvalues and complex pairs."""
    rng = np.random.default_rng(4)
    for _ in range(300):
        p = int(rng.integers(2, 8))
        a = rng.normal(size=(p, p))
        shift = np.linalg.eigvals(a).real.max() + rng.choice([-1, 1]) * 10 ** rng.uniform(-6, -2)
        m = a - shift * np.eye(p)
        assert SchurLyapunov(m).stable == (np.linalg.eigvals(m).real.max() < 0)



def test_schur_solves_non_symmetric_right_hand_sides_too():
    """A non-symmetric y is solved as it is (code review D, 13 October 2026: the output used to be
    symmetrised always, which is wrong then); a symmetric y still gives a symmetric X."""
    rng = np.random.default_rng(5)
    for p in (2, 4, 7):
        m = rng.normal(size=(p, p))
        np.fill_diagonal(m, -np.abs(m).sum(axis=1) - 0.3)
        y = rng.normal(size=(p, p))
        fac = SchurLyapunov(m)
        x = fac.solve(y)
        np.testing.assert_allclose(m @ x + x @ m.T, y, atol=1e-10)
        xa = fac.solve_adjoint(y)
        np.testing.assert_allclose(m.T @ xa + xa @ m, y, atol=1e-10)
        xs = fac.solve(y + y.T)
        assert np.array_equal(xs, xs.T)


def test_solve_lyapunov_checks_stability():
    """The equation is solvable for many unstable M too, but only a stable M has a stationary
    covariance: the default refuses the rest; check=False is for callers that just checked."""
    rng = np.random.default_rng(6)
    m = rng.normal(size=(4, 4))
    np.fill_diagonal(m, -np.abs(m).sum(axis=1) - 0.5)
    c = np.eye(4)
    np.testing.assert_allclose(solve_lyapunov(m, c), solve_lyapunov(m, c, check=False))
    with pytest.raises(ValueError):
        solve_lyapunov(-m, c)
    with pytest.raises(ValueError):
        solve_lyapunov(np.full((4, 4), np.nan), c)
