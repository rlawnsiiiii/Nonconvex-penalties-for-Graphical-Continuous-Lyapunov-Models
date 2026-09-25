"""Vectorization identities and the Lyapunov solve."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.lyap import (
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
