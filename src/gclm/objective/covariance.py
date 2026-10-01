"""Losses on the implied covariance: Varando & Hansen (2020), eq. (7).

The objective half; the solvers are in :mod:`gclm.solvers.covariance`.

    minimise   L(Sigma(M, C)) + sum_ij w_ij P(M_ij)      subject to M stable

with ``C`` fixed and ``Sigma(M, C)`` the solution of the Lyapunov equation
``M Sigma + Sigma M' + C = 0``.  Two losses, named as in the R package
``gclm`` (``gclm(..., loss = )``):

``"loglik"``     ``log det Sigma + tr(Sigma^{-1} Sigma_hat)`` -- the negative
                 Gaussian log-likelihood, without the factor n/2 and constants.
``"frobenius"``  ``0.5 * ||Sigma - Sigma_hat||_F^2``.

Not to be confused with the *direct* loss of Dettling et al. (2024),
``0.5 * ||M Sigma_hat + Sigma_hat M' + C||_F^2`` (:mod:`gclm.objective.direct`), which is also a
Frobenius norm but of the Lyapunov *residual* at the sample covariance.  The
direct loss is a convex quadratic in ``M``; both losses here are nonconvex in
``M`` even before any penalty, and need ``M`` stable.  See docs/LIKELIHOOD.md.

``P`` is the lasso, MCP or SCAD (:mod:`gclm.objective.penalties`, textbook convention --
there is no design matrix, so the ncvreg convention has no meaning here).

Gradient (Varando & Hansen, Prop. 3.1).  With ``G = dL/dSigma`` at
``Sigma = Sigma(M, C)`` and ``Z`` the solution of the *adjoint* Lyapunov equation
``M' Z + Z M = G``,

    grad_M L(Sigma(M, C)) = -2 Z Sigma.

One real Schur factorisation ``M = Q T Q'`` per iterate serves three purposes:
the stability check (the diagonal of ``T`` holds the real parts of the
eigenvalues), ``Sigma(M, C)``, and the adjoint solve -- the same trick ``gclm``'s
Fortran uses, so one iteration costs ``O(p^3)`` (:class:`gclm.lyapunov.SchurLyapunov`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.lapack import dtrsyl

from gclm.lyapunov import SchurLyapunov
from gclm.objective.penalties import value


LOSSES = ("loglik", "frobenius")

def check_loss(loss: str) -> str:
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    return loss

def sigma_loss(sigma: np.ndarray, sigma_hat: np.ndarray, loss: str) -> float:
    """``L(Sigma)``; ``inf`` for ``"loglik"`` if ``Sigma`` is not positive definite."""
    check_loss(loss)
    if loss == "frobenius":
        d = sigma - sigma_hat
        return 0.5 * float(np.sum(d * d))
    try:
        cf = cho_factor(sigma, lower=True, check_finite=False)
    except LinAlgError:
        return np.inf
    logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    return logdet + float(np.trace(cho_solve(cf, sigma_hat, check_finite=False)))

def sigma_loss_grad(sigma: np.ndarray, sigma_hat: np.ndarray, loss: str) -> np.ndarray:
    """``dL/dSigma``: ``P - P Sigma_hat P`` (``P = Sigma^{-1}``) or ``Sigma - Sigma_hat``."""
    check_loss(loss)
    if loss == "frobenius":
        return sigma - sigma_hat
    cf = cho_factor(sigma, lower=True, check_finite=False)
    prec = cho_solve(cf, np.eye(sigma.shape[0]), check_finite=False)
    g = prec - prec @ sigma_hat @ prec
    return 0.5 * (g + g.T)

@dataclass
class _Point:
    """One iterate with everything computed at it."""

    m: np.ndarray
    fac: SchurLyapunov
    sigma: np.ndarray | None
    f: float                          # smooth loss; inf if M is not stable

def _evaluate(m, c, sigma_hat, loss) -> _Point:
    fac = SchurLyapunov(m)
    if not fac.stable:
        return _Point(m, fac, None, np.inf)
    sig = fac.sigma(c)
    return _Point(m, fac, sig, sigma_loss(sig, sigma_hat, loss))

def _grad_at(pt: _Point, sigma_hat, loss) -> np.ndarray:
    z = pt.fac.solve_adjoint(sigma_loss_grad(pt.sigma, sigma_hat, loss))
    return -2.0 * z @ pt.sigma

def loss_value(m, sigma_hat, c, loss="loglik") -> float:
    """``L(Sigma(M, C))``; ``inf`` if ``M`` is not stable."""
    return _evaluate(np.asarray(m, float), c, sigma_hat, check_loss(loss)).f

def loss_grad(m, sigma_hat, c, loss="loglik") -> np.ndarray:
    """``grad_M L(Sigma(M, C)) = -2 Z Sigma`` with ``M' Z + Z M = dL/dSigma``."""
    pt = _evaluate(np.asarray(m, float), c, sigma_hat, check_loss(loss))
    if not np.isfinite(pt.f):
        raise ValueError("M is not stable (or Sigma(M, C) is not positive definite)")
    return _grad_at(pt, sigma_hat, loss)

def objective(m, sigma_hat, c, lam, weights, loss="loglik", penalty="lasso",
              gamma=None) -> float:
    """``L(Sigma(M, C)) + sum_ij w_ij P(M_ij)``."""
    return loss_value(m, sigma_hat, c, loss) + value(m, lam, weights, penalty, gamma)

def diagonal_fit(sigma_hat: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimiser over diagonal ``M``, for either loss: ``M_ii = -C_ii / (2 Sigma_hat_ii)``.

    With ``M = diag(m)`` and ``C`` diagonal the Lyapunov equation reads
    ``(m_i + m_j) Sigma_ij + C_ij = 0``: off the diagonal ``C_ij = 0`` forces
    ``Sigma_ij = 0``, on it ``Sigma_ii = -C_ii / (2 m_i)``.  Both losses then
    separate over ``i`` -- ``log s + Sigma_hat_ii / s`` and ``(s - Sigma_hat_ii)^2 / 2``
    -- and are minimised at ``Sigma_ii = Sigma_hat_ii``, i.e. ``m_i = -C_ii /
    (2 Sigma_hat_ii)``; the off-diagonal part of the Frobenius loss does not
    depend on ``M``.  For a correlation matrix and ``C = 2 I`` this is ``-I``.
    """
    c = np.asarray(c, dtype=float)
    if np.any(c != np.diag(np.diag(c))):
        raise ValueError("diagonal_fit needs a diagonal C")
    return np.diag(-np.diag(c) / (2.0 * np.diag(sigma_hat)))

def dense_fit(sigma_hat: np.ndarray, c: np.ndarray) -> np.ndarray:
    """An exact unpenalised minimiser: ``M0 = -C Sigma_hat^{-1} / 2``.

    ``M0 Sigma_hat + Sigma_hat M0' = -C/2 - C/2 = -C``, so ``Sigma(M0, C) =
    Sigma_hat`` and both losses reach their global minimum.  ``M0`` is stable
    (a product of two positive definite matrices has positive eigenvalues).
    It is Varando's ``B0 = -R^{-1}/2`` for ``C = I`` -- the dense end of the path.
    The minimiser is not unique: ``M0 + W Sigma_hat^{-1}`` for any skew ``W``
    gives the same ``Sigma``.
    """
    return -0.5 * np.asarray(c, float) @ np.linalg.inv(sigma_hat)

def lambda_max(sigma_hat: np.ndarray, c: np.ndarray, loss: str = "loglik") -> float:
    """Smallest ``lam`` at which :func:`diagonal_fit` is a stationary point.

    Dettling's definition ("the smallest lambda for which the estimate is
    diagonal") carried over to these losses, which the paper gives no
    ``lambda_max`` for.  At the diagonal fit the diagonal part of the gradient
    is zero, so the first-order condition reduces to ``|grad_ij| <= lam`` off
    the diagonal.  The same value for lasso, MCP and SCAD, whose subdifferential
    at zero is ``[-lam, lam]``.  For a correlation matrix and ``C = 2 I`` it is
    ``max_{i != j} |Sigma_hat_ij|``, the largest absolute correlation -- the
    graphical lasso's ``lambda_max`` (docs/LIKELIHOOD.md Section 3).  For these
    nonconvex problems it is the smallest ``lam`` for which the *diagonal*
    solution is stationary; other, non-diagonal stationary points may exist
    above it.
    """
    g = loss_grad(diagonal_fit(sigma_hat, c), sigma_hat, c, loss)
    np.fill_diagonal(g, 0.0)
    return float(np.max(np.abs(g)))

def _trsyl(t: np.ndarray, y: np.ndarray, trana: str, tranb: str) -> np.ndarray:
    x, scale, info = dtrsyl(t, t, y, trana=trana, tranb=tranb)
    return x / scale

def hessian(m, sigma_hat, c, loss, rows, cols) -> np.ndarray:
    """Exact Hessian of ``L(Sigma(M, C))`` restricted to the entries ``(rows, cols)``.

    Used only by the experimental Newton solver (``solve(method="newton")``,
    docs/LIKELIHOOD.md Section 5) and its tests; the estimator's solver is
    first-order and never needs it.

    Everything is done in the Schur basis of ``M = Q T Q'``, where every
    Lyapunov solve is one ``dtrsyl`` call.  With ``J_a = dSigma/dM_a`` for
    ``a = (i, j)`` -- the solution of ``M J + J M' = -(E_a Sigma + Sigma E_a')``,
    whose right-hand side is the rank-2 matrix ``u v' + v u'`` with
    ``u = Q[i, :]``, ``v = (Sigma Q)[j, :]`` in that basis -- and ``Z`` the adjoint
    solution of the gradient (``M' Z + Z M = dL/dSigma``),

        H_ab = d2L/dSigma2 [J_a, J_b]  -  2 (Z J_b)_a  -  2 (Z J_a)_b .

    The first term is the Gauss-Newton / Fisher part: ``<J_a, J_b>`` for the
    Frobenius loss and ``-tr(P J_a P J_b) + tr(P J_a P J_b P Sigma_hat)
    + tr(P J_b P J_a P Sigma_hat)`` for the log-likelihood (``P = Sigma^{-1}``).
    The second comes from the curvature of ``M -> Sigma(M)`` itself.  Both are
    assembled as matrix products: ``O(|A| p^3)`` for the ``|A|`` solves and
    ``O(|A|^2 p^2)`` for the products.  Checked against finite differences in
    tests/test_covloss.py.
    """
    loss = check_loss(loss)
    pt = _evaluate(np.asarray(m, float), c, sigma_hat, loss)
    if not np.isfinite(pt.f):
        raise ValueError("M is not stable")
    return _hessian_at(pt, sigma_hat, loss, np.asarray(rows), np.asarray(cols))

def _hessian_at(pt: _Point, sigma_hat, loss, rows, cols) -> np.ndarray:
    q, t = pt.fac.q, pt.fac.t
    sig = pt.sigma
    p = sig.shape[0]
    n_a = len(rows)
    u = q[rows, :]
    v = (sig @ q)[cols, :]
    x = np.empty((n_a, p, p))
    for a in range(n_a):
        r = np.outer(u[a], v[a])
        x[a] = _trsyl(t, -(r + r.T), "N", "T")          # J_a = Q x_a Q'
    xf = x.reshape(n_a, -1)

    if loss == "frobenius":
        h = xf @ xf.T
    else:
        prec = q.T @ np.linalg.inv(sig) @ q                # P in the Schur basis
        w = prec @ (q.T @ sigma_hat @ q)                   # P Sigma_hat
        k = prec[None] @ x                                 # K_a = P J_a
        kf = k.reshape(n_a, -1)
        kt = np.transpose(k, (0, 2, 1)).reshape(n_a, -1)
        kw = np.transpose(k @ w[None], (0, 2, 1)).reshape(n_a, -1)
        kk = kf @ kt.T                                     # tr(K_a K_b)
        kkw = kf @ kw.T                                    # tr(K_a K_b P Sigma_hat)
        h = -kk + kkw + kkw.T

    z = _trsyl(t, q.T @ sigma_loss_grad(sig, sigma_hat, loss) @ q, "T", "N")
    y = (z[None] @ x).reshape(n_a, -1)                     # Z J_b, Schur basis
    uw = (u[:, :, None] * q[cols, :][:, None, :]).reshape(n_a, -1)
    mz = uw @ y.T                                          # (Z J_b)_a
    h = h - 2.0 * (mz + mz.T)
    return 0.5 * (h + h.T)
