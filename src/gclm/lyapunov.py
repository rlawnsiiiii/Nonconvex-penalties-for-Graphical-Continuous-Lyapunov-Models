"""Vectorization helpers and the continuous Lyapunov equation.

Dettling et al. (2024), Section 2.  The Lyapunov equation

    M @ Sigma + Sigma @ M.T + C = 0                                      (1.2)

is linear in ``M`` and vectorizes to ``A(Sigma) @ vec(M) + vec(C) = 0`` with

    A(Sigma) = kron(Sigma, I_p) + kron(I_p, Sigma) @ K                   (2.2)

where ``K`` is the commutation matrix.  ``vec`` stacks columns (Fortran order),
matching both the paper and R's ``c()``.

:class:`SchurLyapunov` solves the equation and its adjoint on one real Schur
factorisation of ``M`` -- the form the covariance-loss solvers need.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur, solve_continuous_lyapunov
from scipy.linalg.lapack import dtrsyl


def vec(a: np.ndarray) -> np.ndarray:
    """Stack the columns of ``a`` (the paper's and R's convention)."""
    return np.asarray(a).flatten(order="F")


def unvec(v: np.ndarray, p: int | None = None) -> np.ndarray:
    """Inverse of :func:`vec`."""
    v = np.asarray(v)
    if p is None:
        p = int(round(np.sqrt(v.size)))
    return v.reshape((p, p), order="F")


def commutation_matrix(p: int) -> np.ndarray:
    """``K`` with ``K @ vec(A) == vec(A.T)`` for every ``p x p`` matrix ``A``."""
    idx = np.arange(p * p).reshape((p, p), order="F").ravel(order="C")
    return np.eye(p * p)[idx, :]


def design_matrix(sigma: np.ndarray) -> np.ndarray:
    """``A(Sigma)`` from eq. (2.2).

    Rows are indexed by the entries of the (symmetric) Lyapunov residual, so each
    off-diagonal equation appears twice.  That redundancy is intentional: it is
    exactly what makes ``0.5 * ||A vec(M) + vec(C)||^2`` equal the squared
    Frobenius loss of eq. (1.4).  Do not deduplicate.
    """
    sigma = np.asarray(sigma, dtype=float)
    p = sigma.shape[0]
    eye = np.eye(p)
    return np.kron(sigma, eye) + np.kron(eye, sigma) @ commutation_matrix(p)


def gram_matrix(sigma: np.ndarray) -> np.ndarray:
    """``Gamma(Sigma) = A.T @ A`` via Dettling's Lemma 1 (closed form)."""
    sigma = np.asarray(sigma, dtype=float)
    p = sigma.shape[0]
    eye = np.eye(p)
    k = commutation_matrix(p)
    ss = np.kron(sigma, sigma)
    return 2.0 * np.kron(sigma @ sigma, eye) + ss @ k + k @ ss


def design_column_sq_norms(sigma: np.ndarray) -> np.ndarray:
    """Squared column norms of ``A(Sigma)``, arranged as a ``p x p`` matrix.

    Entry ``(i, k)`` is ``||A(Sigma) e_ik||^2``, the loss curvature along the
    coordinate ``M_ik``.  The column is ``vec(E_ik Sigma + Sigma E_ki)``: row ``i``
    holds ``Sigma[k, :]``, column ``i`` holds ``Sigma[:, k]``, and they overlap at
    ``(i, i)`` with ``2 Sigma_ik``.  Hence, in ``O(p^2)`` without forming ``A``,

        ||A e_ik||^2 = 2 (||Sigma[k, :]||^2 + Sigma_ik^2).
    """
    sigma = np.asarray(sigma, dtype=float)
    return 2.0 * ((sigma ** 2).sum(axis=1)[None, :] + sigma ** 2)


def solve_lyapunov(m: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Equilibrium covariance: the unique ``Sigma`` with ``M Sigma + Sigma M' + C = 0``.

    Requires ``M`` stable.  The result is symmetrized to kill roundoff asymmetry.
    """
    m = np.asarray(m, dtype=float)
    c = np.asarray(c, dtype=float)
    sigma = solve_continuous_lyapunov(m, -c)
    return 0.5 * (sigma + sigma.T)


def is_stable(m: np.ndarray) -> bool:
    """True iff every eigenvalue of ``m`` has strictly negative real part."""
    return bool(np.all(np.linalg.eigvals(np.asarray(m)).real < 0))


def lyapunov_residual(m: np.ndarray, sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``R(M) = M Sigma + Sigma M' + C``."""
    return m @ sigma + sigma @ m.T + c


class SchurLyapunov:
    """``M = Q T Q'`` (real Schur form) and the two Lyapunov solves it enables.

    LAPACK returns ``T`` in standardised form: every 2x2 block has equal
    diagonal entries, which are then the real part of its eigenvalue pair.  So
    ``M`` is stable iff ``max(diag(T)) < 0`` -- no separate eigenvalue call.
    """

    def __init__(self, m: np.ndarray):
        self.t, self.q = schur(np.asarray(m, dtype=float), output="real")

    @property
    def stable(self) -> bool:
        return bool(np.max(np.diag(self.t)) < 0.0)

    def _trsyl(self, y: np.ndarray, trana: str, tranb: str) -> np.ndarray:
        x, scale, info = dtrsyl(self.t, self.t, self.q.T @ y @ self.q,
                                trana=trana, tranb=tranb)
        if info < 0:                                       # pragma: no cover
            raise RuntimeError(f"dtrsyl: illegal argument {-info}")
        x = self.q @ (x / scale) @ self.q.T
        return 0.5 * (x + x.T)                             # symmetric input -> symmetric output

    def solve(self, y: np.ndarray) -> np.ndarray:
        """``X`` with ``M X + X M' = y``."""
        return self._trsyl(y, "N", "T")

    def solve_adjoint(self, y: np.ndarray) -> np.ndarray:
        """``X`` with ``M' X + X M = y``."""
        return self._trsyl(y, "T", "N")

    def sigma(self, c: np.ndarray) -> np.ndarray:
        """``Sigma(M, C)``: the solution of ``M Sigma + Sigma M' + C = 0``."""
        return self.solve(-np.asarray(c, dtype=float))
