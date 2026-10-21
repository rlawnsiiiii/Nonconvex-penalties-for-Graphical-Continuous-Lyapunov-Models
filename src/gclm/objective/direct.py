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
from gclm.objective.penalties import penalty_scale, penalty_weights, prox, value


def direct_loss(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> float:
    r = lyapunov_residual(m, sigma, c)
    return 0.5 * float(np.sum(r * r))

def direct_grad(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``2 R(M) Sigma`` -- the gradient of :func:`direct_loss` for a **symmetric** ``C``.

    In general the gradient is ``(R + R') Sigma``; it equals ``2 R Sigma`` when the
    residual is symmetric, i.e. when ``C`` is.  Every ``C`` in this repository is,
    and the solvers check it once at entry (``gclm.solvers.proxgrad``), not here in
    the inner loop.
    """
    return 2.0 * lyapunov_residual(m, sigma, c) @ sigma

def lipschitz_bound(sigma: np.ndarray) -> float:
    """Upper bound on the Lipschitz constant of ``direct_grad``.

    ``L = ||A(Sigma)||_2^2 <= (2 * lambda_max(Sigma))^2``, since both summands of
    ``A`` have spectral norm ``lambda_max(Sigma)`` (the commutation matrix is
    orthogonal).
    """
    lam = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))))
    return 4.0 * lam * lam

def _diagonal_design(sigma: np.ndarray) -> np.ndarray:
    """The ``p^2 x p`` design of the residual of a diagonal ``M``: ``vec(R) = B m + vec(C)``,
    ``R_ij = (m_i + m_j) Sigma_ij + C_ij``."""
    p = sigma.shape[0]
    rows = np.zeros((p * p, p))
    idx = np.arange(p * p)
    i = idx % p
    j = idx // p
    s = sigma[i, j]
    np.add.at(rows, (idx, i), s)
    np.add.at(rows, (idx, j), s)
    return rows

def diagonal_fit(sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unpenalized least-squares fit restricted to diagonal ``M``.

    The residual entries are ``R_ij = (m_i + m_j) Sigma_ij + C_ij``, linear in the
    diagonal vector ``m``, so this is a ``p``-variable least-squares problem.
    """
    m_diag, *_ = np.linalg.lstsq(_diagonal_design(sigma), -vec(c), rcond=None)
    return np.diag(m_diag)

def diagonal_solution(
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    penalty: str = "lasso",
    gamma: float | None = None,
    weights: np.ndarray | None = None,
    start: np.ndarray | None = None,
    tol: float = 1e-13,
    max_sweeps: int = 10_000,
) -> np.ndarray:
    """The estimate when it is diagonal:

        argmin_m  0.5 ||B m + vec(C)||^2 + sum_i w_i P_lam(|m_i|),     M = diag(m),

    with ``B`` the design of :func:`diagonal_fit`.  ``weights``: the diagonal's penalty weights,
    length p (default 1; 0 = unpenalised; ``inf`` = held at zero, lasso only).  With every
    weight 0 this is :func:`diagonal_fit`, returned as such: the diagonal unpenalised -- every
    run before wave 10 -- is untouched.

    Coordinate descent with exact coordinate minimisers: the proximal map of the penalty
    (:func:`gclm.objective.penalties.prox`, textbook convention) with step ``1 / H_ii``,
    ``H = B'B``, started from the diagonal fit (or ``start``).  The problem is strictly convex
    when ``H`` exceeds the penalty's concavity, ``1 / gamma`` for MCP and ``1 / (gamma - 1)`` for
    SCAD: on the correlation scale ``H = 2 (Sigma o Sigma) + 2 I >= 2 I``, so always.  Then the
    solution is unique and is the limit.
    """
    p = sigma.shape[0]
    w = np.ones(p) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (p,):
        raise ValueError(f"weights must have length p = {p}")
    if not lam > 0 or np.all(w == 0):
        return diagonal_fit(sigma, c)
    b = _diagonal_design(sigma)
    h = b.T @ b
    m = np.diag(diagonal_fit(sigma, c)).copy() if start is None else np.array(start, dtype=float)
    grad = h @ m + b.T @ vec(c)                          # the gradient in m: B'(B m + vec C)
    hii = np.diag(h)
    for _ in range(max_sweeps):
        change = 0.0
        for i in range(p):
            z = m[i] - grad[i] / hii[i]
            new = float(prox(np.array([z]), 1.0 / hii[i], lam, w[i:i + 1], penalty, gamma)[0])
            if new != m[i]:
                grad += h[:, i] * (new - m[i])
                change = max(change, abs(new - m[i]))
                m[i] = new
        if change <= tol * max(1.0, float(np.max(np.abs(m)))):
            break
    return np.diag(m)

def lambda_max(
    sigma: np.ndarray,
    c: np.ndarray,
    penalize_diagonal: bool = False,
    weights: np.ndarray | None = None,
    penalty: str = "lasso",
    gamma: float | None = None,
    tol: float = 1e-13,
) -> float:
    """Smallest ``lambda`` for which the estimate is diagonal.

    The estimate is diagonal at ``lambda`` exactly when the diagonal-only solution ``D(lambda)``
    (:func:`diagonal_solution`) meets the first-order condition of every off-diagonal entry at zero,

        |[2 R(D) Sigma]_ij| <= lambda w_ij        for all i != j,

    the same condition for the lasso, MCP and SCAD, whose slope at 0+ is ``lambda``.

    Diagonal unpenalised (every run before wave 10): ``D`` is the diagonal least-squares fit
    whatever ``lambda`` and the penalty, so ``lambda_max = max_{i != j} |[2 R(M_diag) Sigma]_ij| / w_ij``
    in closed form, the same for every penalty.

    Diagonal penalised (Dettling's eq. (1.4); wave 10): ``D(lambda)`` is shrunk, by an amount that
    depends on the penalty, so each penalty has its own ``lambda_max``.  It is found by bisection on
    the condition -- one diagonal-only solve (p unknowns) per step, no solve of the whole problem --
    to relative precision ``tol``; the upper end of the final bracket is returned, where the
    condition holds.

    ``weights``: per-entry penalty weights, ``inf`` = excluded (the adaptive lasso's); default 1 off
    the diagonal, and 1 or 0 on it by ``penalize_diagonal``.  Needs ``p >= 2``: a ``1 x 1`` drift
    matrix has no off-diagonal entry, so the quantity is undefined (``ValueError``).
    """
    p = sigma.shape[0]
    if p < 2:
        raise ValueError("lambda_max needs p >= 2: a 1 x 1 drift matrix has no off-diagonal entry")
    off = ~np.eye(p, dtype=bool)
    if not penalize_diagonal:
        if weights is not None and np.any(np.diag(weights) != 0):
            raise ValueError("weights penalise the diagonal: pass penalize_diagonal=True")
        m_diag = diagonal_fit(sigma, c)
        grad = 2.0 * lyapunov_residual(m_diag, sigma, c) @ sigma
        if weights is None:
            return float(np.max(np.abs(grad[off])))
        w_off = np.asarray(weights, dtype=float)[off]
        return float(np.max(np.where(np.isinf(w_off), 0.0, np.abs(grad[off]) / w_off)))

    w = penalty_weights(p, penalize_diagonal=True) if weights is None else np.asarray(weights, dtype=float)
    w_off, w_diag = w[off], np.diag(w).copy()
    last = {"m": None}

    def diagonal(lam: float) -> bool:                    # the condition at D(lam)
        d = diagonal_solution(sigma, c, lam, penalty, gamma, weights=w_diag, start=last["m"])
        last["m"] = np.diag(d)
        g = np.abs(2.0 * lyapunov_residual(d, sigma, c) @ sigma)[off]
        return bool(np.all((g <= lam * w_off) | np.isinf(w_off)))

    # at and above the largest |grad_ij(0)| / w_ij, zero solves the diagonal-only problem (strictly
    # convex) and meets the condition; the doubling only guards the strict-convexity assumption
    g0 = np.abs(2.0 * lyapunov_residual(np.zeros((p, p)), sigma, c) @ sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = max(float(np.max(np.where(w > 0, g0 / w, 0.0))), 1e-12)
    while not diagonal(hi):
        hi *= 2.0
    lo = hi / 2.0
    while diagonal(lo):
        hi, lo = lo, lo / 2.0
        if lo < 1e-14:
            return hi
    while hi / lo - 1.0 > tol:
        mid = float(np.sqrt(lo * hi))
        if diagonal(mid):
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
