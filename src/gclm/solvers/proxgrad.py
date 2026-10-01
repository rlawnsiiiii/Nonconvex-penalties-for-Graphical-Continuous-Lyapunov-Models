"""Proximal-gradient solvers for the direct loss, matrix-free.

:func:`solve_fista` -- FISTA with the adaptive restart of O'Donoghue & Candes
(2015) for the lasso (docs/FISTA.md); for MCP/SCAD it hands over to
:func:`_solve_mapg`, the monotone accelerated proximal gradient of Li & Lin
(2015) (docs/NONCONVEX.md Section 4).  ``O(p^3)`` per iteration, never forms the
design matrix; the only backend that reaches ``p = 50``.  The default solver.
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import lyapunov_residual
from gclm.objective.direct import direct_grad, lipschitz_bound
from gclm.objective.penalties import canonical, penalty_scale, penalty_weights, prox, resolve_gamma, value


def _soft_threshold(z: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def solve_fista(
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
    m_init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    step: float | None = None,
    penalty: str = "lasso",
    gamma: float | None = None,
    convention: str = "textbook",
) -> np.ndarray:
    """Accelerated proximal gradient.

    ``penalty="lasso"``: FISTA with the adaptive restart of O'Donoghue &
    Candes (2015) -- convex, converges to the global minimiser.

    ``penalty="MCP"`` / ``"SCAD"``: the monotone accelerated proximal gradient
    of Li & Lin (2015), which converges to a stationary point.  See
    :func:`_solve_mapg` and docs/NONCONVEX.md.

    Stops when the max-norm coefficient change falls below ``tol``.
    """
    if canonical(penalty) != "lasso":
        return _solve_mapg(sigma, c, lam, weights=weights, m_init=m_init, tol=tol,
                           max_iter=max_iter, step=step, penalty=penalty, gamma=gamma,
                           convention=convention)
    p = sigma.shape[0]
    weights = penalty_weights(p) if weights is None else weights
    lstep = lipschitz_bound(sigma) if step is None else 1.0 / step
    if lstep <= 0:
        lstep = 1.0
    t_step = 1.0 / lstep
    thresh = lam * weights * t_step

    m = np.zeros((p, p)) if m_init is None else np.array(m_init, dtype=float)
    y = m.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = direct_grad(y, sigma, c)
        m_new = _soft_threshold(y - t_step * grad, thresh)
        # adaptive restart: momentum is dropped when it points uphill
        if np.sum((y - m_new) * (m_new - m)) > 0:
            t = 1.0
            y = m_new.copy()
            t_next = 1.0
        else:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = m_new + ((t - 1.0) / t_next) * (m_new - m)
        delta = np.max(np.abs(m_new - m))
        m, t = m_new, t_next
        if delta < tol:
            break
    return m

def _solve_mapg(
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
    m_init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    step: float | None = None,
    penalty: str = "MCP",
    gamma: float | None = None,
    convention: str = "textbook",
) -> np.ndarray:
    """Monotone accelerated proximal gradient for MCP / SCAD.

    Li & Lin (2015), "Accelerated Proximal Gradient Methods for Nonconvex
    Programming", NeurIPS, Algorithm 1.  Each iteration takes two proximal
    gradient steps -- one from the extrapolated point ``y`` and one from the
    current iterate ``x`` -- and keeps whichever has the lower objective:

        y_k     = x_k + (t_{k-1}/t_k)(z_k - x_k) + ((t_{k-1}-1)/t_k)(x_k - x_{k-1})
        z_{k+1} = prox(y_k - a grad f(y_k))
        v_{k+1} = prox(x_k - a grad f(x_k))
        t_{k+1} = (sqrt(4 t_k^2 + 1) + 1) / 2
        x_{k+1} = z_{k+1} if F(z_{k+1}) <= F(v_{k+1}) else v_{k+1}

    The comparison with the plain step ``v`` makes the objective monotone and
    gives convergence to a critical point for nonconvex, KL-type penalties such
    as MCP and SCAD (their Theorem 1), with step ``a < 1/L``.  Plain FISTA has
    no such guarantee once the penalty is nonconvex.

    Residuals are cached so each candidate point costs one residual, used for
    both its objective value and (if it is kept) the next gradient.
    """
    penalty = canonical(penalty)
    gamma = resolve_gamma(penalty, gamma)
    p = sigma.shape[0]
    weights = penalty_weights(p) if weights is None else weights
    scale = penalty_scale(sigma, convention)
    lip = lipschitz_bound(sigma) if step is None else 1.0 / step
    a = 0.99 / lip if step is None else step          # a < 1/L, as the theorem needs

    def resid(m):
        return lyapunov_residual(m, sigma, c)

    def fval(m, r):
        return 0.5 * float(np.sum(r * r)) + value(m, lam, weights, penalty, gamma, scale)

    def pstep(m, r):
        return prox(m - a * (2.0 * r @ sigma), a, lam, weights, penalty, gamma, scale)

    x = np.zeros((p, p)) if m_init is None else np.array(m_init, dtype=float)
    r_x = resid(x)
    x_prev, z = x.copy(), x.copy()
    t_prev, t = 0.0, 1.0

    for _ in range(max_iter):
        y = x + (t_prev / t) * (z - x) + ((t_prev - 1.0) / t) * (x - x_prev)
        z = pstep(y, resid(y))
        v = pstep(x, r_x)
        r_z, r_v = resid(z), resid(v)
        f_z, f_v = fval(z, r_z), fval(v, r_v)
        t_prev, t = t, 0.5 * (np.sqrt(4.0 * t * t + 1.0) + 1.0)
        x_prev = x
        if f_z <= f_v:
            x, r_x = z, r_z
        else:
            x, r_x = v, r_v
        if np.max(np.abs(x - x_prev)) < tol:
            break
    return x
