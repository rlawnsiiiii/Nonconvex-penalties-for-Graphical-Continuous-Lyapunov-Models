"""Quadratic (Frobenius) Lyapunov loss and its gradient.

``f(M) = 0.5 * ||M Sigma + Sigma M' + C||_F^2``  -- the smooth part of eq. (1.4).

The gradient is computed matrix-free in ``O(p^3)``:

    grad f(M) = 2 * R(M) @ Sigma,      R(M) = M Sigma + Sigma M' + C

which avoids ever forming the ``p^2 x p^2`` design matrix.  This is the form that
carries over unchanged to the SCAD/MCP variants in S1b.
"""

from __future__ import annotations

import numpy as np

from gclm.lyap import lyapunov_residual


def frobenius_loss(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> float:
    r = lyapunov_residual(m, sigma, c)
    return 0.5 * float(np.sum(r * r))


def frobenius_grad(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 2.0 * lyapunov_residual(m, sigma, c) @ sigma


def lipschitz_bound(sigma: np.ndarray) -> float:
    """Upper bound on the Lipschitz constant of ``frobenius_grad``.

    ``L = ||A(Sigma)||_2^2 <= (2 * lambda_max(Sigma))^2``, since both summands of
    ``A`` have spectral norm ``lambda_max(Sigma)`` (the commutation matrix is
    orthogonal).
    """
    lam = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))))
    return 4.0 * lam * lam


def objective(
    m: np.ndarray,
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray,
) -> float:
    """Full penalized objective ``f(M) + lam * sum(weights * |M|)``."""
    return frobenius_loss(m, sigma, c) + lam * float(np.sum(weights * np.abs(m)))
