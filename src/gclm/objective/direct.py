"""Dettling's direct loss: the quadratic loss on the Lyapunov residual.

    f(M) = 0.5 * ||M Sigma + Sigma M' + C||_F^2            -- eq. (1.4) of
    Dettling, Drton & Kolar (2024), the smooth part of the Direct Lyapunov Lasso.

The gradient is matrix-free in ``O(p^3)``:

    grad f(M) = 2 * R(M) @ Sigma,      R(M) = M Sigma + Sigma M' + C

so the ``p^2 x p^2`` design matrix is never formed.  Also here: the Lipschitz
constant of the gradient, the diagonal fit and ``lambda_max`` (the sparse end of
the path, closed forms specific to this loss), and the penalised objective.

Not to be confused with Varando & Hansen's Frobenius loss on the *implied*
covariance, ``0.5 * ||Sigma(M) - Sigma_hat||_F^2`` (:mod:`gclm.objective.covariance`).
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import lyapunov_residual, vec
from gclm.objective.penalties import penalty_scale, penalty_weights, value


def direct_loss(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> float:
    r = lyapunov_residual(m, sigma, c)
    return 0.5 * float(np.sum(r * r))

def direct_grad(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 2.0 * lyapunov_residual(m, sigma, c) @ sigma

def lipschitz_bound(sigma: np.ndarray) -> float:
    """Upper bound on the Lipschitz constant of ``direct_grad``.

    ``L = ||A(Sigma)||_2^2 <= (2 * lambda_max(Sigma))^2``, since both summands of
    ``A`` have spectral norm ``lambda_max(Sigma)`` (the commutation matrix is
    orthogonal).
    """
    lam = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))))
    return 4.0 * lam * lam

def diagonal_fit(sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unpenalized least-squares fit restricted to diagonal ``M``.

    The residual entries are ``R_ij = (m_i + m_j) Sigma_ij + C_ij``, linear in the
    diagonal vector ``m``, so this is a ``p``-variable least-squares problem.
    """
    p = sigma.shape[0]
    rows = np.zeros((p * p, p))
    idx = np.arange(p * p)
    i = idx % p
    j = idx // p
    s = sigma[i, j]
    np.add.at(rows, (idx, i), s)
    np.add.at(rows, (idx, j), s)
    m_diag, *_ = np.linalg.lstsq(rows, -vec(c), rcond=None)
    return np.diag(m_diag)

def lambda_max(
    sigma: np.ndarray,
    c: np.ndarray,
    penalize_diagonal: bool = False,
    **solver_kwargs,
) -> float:
    """Smallest ``lambda`` for which the estimate is diagonal.

    With the diagonal unpenalized this has a closed form from the KKT conditions
    at the diagonal-only least-squares fit:

        lambda_max = max_{i != j} |[2 R(M_diag) Sigma]_ij|

    If the diagonal *is* penalized, "diagonal" is no longer a single well-defined
    stopping point (large lambda drives M to 0, which is also diagonal), so fall
    back to bisection on log-lambda for the smallest lambda with zero off-diagonal.
    """
    p = sigma.shape[0]
    if not penalize_diagonal:
        m_diag = diagonal_fit(sigma, c)
        grad = 2.0 * lyapunov_residual(m_diag, sigma, c) @ sigma
        off = ~np.eye(p, dtype=bool)
        return float(np.max(np.abs(grad[off])))

    off = ~np.eye(p, dtype=bool)
    weights = penalty_weights(p, penalize_diagonal=True)

    def is_diagonal(lam: float) -> bool:
        from gclm.solvers.proxgrad import solve_fista   # local: solvers import this module

        m = solve_fista(sigma, c, lam, weights=weights, **solver_kwargs)
        return bool(np.all(m[off] == 0.0))

    hi = float(np.max(np.abs(2.0 * lyapunov_residual(np.zeros((p, p)), sigma, c) @ sigma)))
    hi = max(hi, 1e-8)
    while not is_diagonal(hi):
        hi *= 2.0
    lo = hi
    while is_diagonal(lo):
        lo /= 2.0
        if lo < 1e-14:
            return lo
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if is_diagonal(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)

def objective(
    m: np.ndarray,
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray,
    penalty: str = "lasso",
    gamma: float | None = None,
    convention: str = "textbook",
) -> float:
    """Full penalised objective ``f(M) + sum_ij w_ij P_ij(M_ij)``.

    ``penalty`` is ``"lasso"`` (``P = lam |x|``, the default), ``"MCP"`` or
    ``"SCAD"``.  ``convention`` is ``"textbook"`` or ``"ncvreg"``; the two differ
    only for MCP/SCAD.  See :mod:`gclm.objective.penalties`.
    """
    scale = penalty_scale(sigma, convention)
    return direct_loss(m, sigma, c) + value(m, lam, weights, penalty, gamma, scale)
