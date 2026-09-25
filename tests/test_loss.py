"""The matrix-free gradient must agree with the design-matrix one."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.loss import frobenius_grad, frobenius_loss, lipschitz_bound
from gclm.lyap import design_matrix, unvec, vec


def _problem(p, seed=0):
    rng = np.random.default_rng(seed)
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    c = 2 * np.eye(p)
    m = rng.normal(size=(p, p))
    return m, sigma, c


@pytest.mark.parametrize("p", [2, 4, 7])
def test_loss_matches_regression_form(p):
    """0.5||R||_F^2 == 0.5||X beta - y||^2 with X = A(Sigma), y = -vec(C)."""
    m, sigma, c = _problem(p)
    x, y = design_matrix(sigma), -vec(c)
    assert np.isclose(frobenius_loss(m, sigma, c), 0.5 * np.sum((x @ vec(m) - y) ** 2))


@pytest.mark.parametrize("p", [2, 4, 7])
def test_gradient_matches_design_matrix(p):
    """grad = 2 R Sigma  ==  X'(X beta - y)."""
    m, sigma, c = _problem(p)
    x, y = design_matrix(sigma), -vec(c)
    assert np.allclose(vec(frobenius_grad(m, sigma, c)), x.T @ (x @ vec(m) - y))


@pytest.mark.parametrize("p", [3, 5])
def test_gradient_matches_finite_differences(p):
    m, sigma, c = _problem(p, seed=p)
    g = frobenius_grad(m, sigma, c)
    h = 1e-6
    for i in range(p):
        for j in range(p):
            e = np.zeros((p, p)); e[i, j] = h
            fd = (frobenius_loss(m + e, sigma, c) - frobenius_loss(m - e, sigma, c)) / (2 * h)
            assert np.isclose(g[i, j], fd, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [3, 6])
def test_lipschitz_bound_dominates_true_constant(p):
    """L must be >= the top eigenvalue of the Gram matrix, or FISTA diverges."""
    _, sigma, _ = _problem(p, seed=p + 1)
    a = design_matrix(sigma)
    true_l = np.linalg.eigvalsh(a.T @ a).max()
    bound = lipschitz_bound(sigma)
    assert bound >= true_l - 1e-8
    assert bound <= 4 * true_l          # and is not absurdly loose
