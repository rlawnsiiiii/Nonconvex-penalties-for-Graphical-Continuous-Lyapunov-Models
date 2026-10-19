"""Exact fits of the Lyapunov equation (plan of 13 October 2026, Section 1.1).

The exact solutions of ``M Sigma_hat + Sigma_hat M' + C = 0`` form the affine space

    { M0 + W Sigma_hat^{-1} : W skew-symmetric },      M0 = -C Sigma_hat^{-1} / 2,

of dimension ``p (p - 1) / 2`` (Dettling, Homs, Amendola, Drton & Hansen 2023, Lemma 27; ``M0`` is
:func:`gclm.objective.covariance.dense_fit`).  Every direct-loss dense -> sparse path starts in or
near this space: the lasso at the smallest lambda of the grid is, up to ``O(lambda)``, its
minimum-l1 member (claude_docs/dense_start_theory_081026.md, Section 2).  This module gives three
members explicitly, the diagonal free -- only the off-diagonal entries enter the norms, matching the
unpenalised diagonal everywhere else in the code:

* :func:`min_l2_exact_fit`   the smallest off-diagonal Frobenius norm: linear least squares in the
  skew coordinates;
* :func:`min_l1_exact_fit`   the smallest off-diagonal l1 norm ("basis pursuit", the BP solution of
  the theory note): a linear program solved by HiGHS, with an optional probe of the minimiser's
  uniqueness (the "BP condition" of the note is: the standardized truth is the unique minimiser);
* :func:`noisy_truth_start`  ``M* + tau N(0, 1)`` off the diagonal: a ceiling for the dense starts,
  not an estimator (it uses the truth).

Coordinates.  ``skew_basis(p)[k] = E_ij - E_ji`` for the ``k``-th pair ``i < j`` in the order of
``numpy.triu_indices(p, 1)``, and ``M(w) = M0 + sum_k w_k B_k Sigma_hat^{-1}``.  The off-diagonal
entries of ``M(w)`` are ``m0_off + J w`` with ``J = exact_fit_jacobian(sigma_hat)``, rows in the
row-major order of :func:`offdiag_indices`.  ``J`` has full column rank (``W Sigma_hat^{-1}``
diagonal forces ``W = 0`` for an invertible ``Sigma_hat``), so a fit that is unique in ``M`` is
unique in ``w`` and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, identity, vstack

from gclm.lyapunov import lyapunov_residual

LP_METHODS = ("highs", "highs-ds", "highs-ipm")
#: an entry of a dense fit counts as present above this size: the exact fits have no exact zeros
#: (the l2 fit and M0 are fully dense, the LP's structural zeros come out as solver dust ~1e-10)
PRESENT = 1e-8
LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}


def offdiag_indices(p: int) -> tuple[np.ndarray, np.ndarray]:
    """The off-diagonal positions ``(i, j)``, ``i != j``, in row-major order."""
    return np.nonzero(~np.eye(p, dtype=bool))


def skew_basis(p: int) -> np.ndarray:
    """The ``p (p - 1) / 2`` matrices ``E_ij - E_ji``, ``i < j``, in the order of ``numpy.triu_indices(p, 1)``."""
    if p < 2:
        raise ValueError(f"p must be at least 2, got {p}")
    iu, ju = np.triu_indices(p, 1)
    basis = np.zeros((len(iu), p, p))
    k = np.arange(len(iu))
    basis[k, iu, ju] = 1.0
    basis[k, ju, iu] = -1.0
    return basis


def skew_from_coordinates(w: np.ndarray, p: int) -> np.ndarray:
    """``sum_k w_k B_k``: the skew matrix with upper triangle ``w`` in the order of ``numpy.triu_indices(p, 1)``."""
    iu, ju = np.triu_indices(p, 1)
    w = np.asarray(w, dtype=float)
    if w.shape != (len(iu),):
        raise ValueError(f"expected {len(iu)} skew coordinates for p = {p}, got shape {w.shape}")
    skew = np.zeros((p, p))
    skew[iu, ju] = w
    skew[ju, iu] = -w
    return skew


def _jacobian(k_mat: np.ndarray) -> np.ndarray:
    """Column ``k``: the off-diagonal entries of ``B_k K`` (row ``i`` is ``K[j, :]``, row ``j`` is ``-K[i, :]``)."""
    p = k_mat.shape[0]
    iu, ju = np.triu_indices(p, 1)
    k = np.arange(len(iu))
    prod = np.zeros((len(iu), p, p))
    prod[k, iu, :] = k_mat[ju, :]
    prod[k, ju, :] = -k_mat[iu, :]
    rows, cols = offdiag_indices(p)
    return prod[:, rows, cols].T


def exact_fit_jacobian(sigma_hat: np.ndarray) -> np.ndarray:
    """``J`` of shape ``(p (p - 1), p (p - 1) / 2)``: the off-diagonal entries of ``M(w)`` are ``m0_off + J w``."""
    return _jacobian(np.linalg.inv(_check_square(sigma_hat, "sigma_hat")))


def _check_square(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix with p >= 2, got shape {a.shape}")
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return a


@dataclass(frozen=True)
class _Space:
    """The affine space of exact fits in skew coordinates, computed once per ``(Sigma_hat, C)``."""

    p: int
    k_mat: np.ndarray            # Sigma_hat^{-1}
    m0: np.ndarray               # the dense fit -C Sigma_hat^{-1} / 2
    jac: np.ndarray              # (p(p-1), p(p-1)/2)
    rows: np.ndarray
    cols: np.ndarray

    @property
    def m0_off(self) -> np.ndarray:
        return self.m0[self.rows, self.cols]

    def offdiag(self, w: np.ndarray) -> np.ndarray:
        return self.m0_off + self.jac @ w

    def matrix(self, w: np.ndarray) -> np.ndarray:
        return self.m0 + skew_from_coordinates(w, self.p) @ self.k_mat


def _space(sigma_hat, c) -> _Space:
    sigma_hat = _check_square(sigma_hat, "sigma_hat")
    c = _check_square(c, "c")
    if c.shape != sigma_hat.shape:
        raise ValueError(f"sigma_hat and c differ in shape: {sigma_hat.shape} vs {c.shape}")
    k_mat = np.linalg.inv(sigma_hat)
    rows, cols = offdiag_indices(sigma_hat.shape[0])
    return _Space(sigma_hat.shape[0], k_mat, -0.5 * c @ k_mat, _jacobian(k_mat), rows, cols)


def min_l2_exact_fit(sigma_hat: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The exact fit with the smallest off-diagonal Frobenius norm.

    ``argmin_w || m0_off + J w ||_2``: ``p (p - 1) / 2`` unknowns, ``p (p - 1)`` equations, full column
    rank, so the minimiser is unique.  ``numpy.linalg.lstsq``; trivial for ``p <= 50``.
    """
    sp = _space(sigma_hat, c)
    w, *_ = np.linalg.lstsq(sp.jac, -sp.m0_off, rcond=None)
    return sp.matrix(w)


@dataclass(frozen=True)
class L1Fit:
    """What :func:`min_l1_exact_fit` found besides the matrix."""

    w: np.ndarray                # the skew coordinates of the returned fit
    value: float                 # its off-diagonal l1 norm, the minimum
    status: int                  # ``linprog`` status (0: optimal)
    unique: bool | None          # None when the minimiser was not probed (``probes=0``)
    face_extent: float           # largest probe maximum over the optimal face (0 for a unique minimiser)


def _lp_constraints(sp: _Space):
    """``A_ub (w, t) <= b_ub`` encoding ``|m0_off + J w| <= t`` in the variables ``(w, t)``, and the bounds."""
    nt, nw = sp.jac.shape
    jac = coo_matrix(sp.jac)
    eye = identity(nt, format="coo")
    a_ub = vstack([hstack([jac, -eye]), hstack([-jac, -eye])]).tocsc()
    b_ub = np.concatenate([-sp.m0_off, sp.m0_off])
    bounds = [(None, None)] * nw + [(0.0, None)] * nt
    return a_ub, b_ub, bounds


def _solve_lp(cost, a_ub, b_ub, bounds, method: str):
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=method, options=LP_OPTIONS)
    if res.status != 0:
        raise RuntimeError(f"the exact-fit linear program did not solve (status {res.status}): {res.message}")
    return res


def _face_extent(sp: _Space, w_star, probes, rng, zero_tol, method):
    """How far the minimal l1 norm can be left unchanged: the optimal face of the LP, probed with
    random linear objectives (the 7 October 2026 check, made exact).

    At the solution ``w*`` split the off-diagonal entries into the nonzero set ``N`` (signs ``s``) and the
    zero set ``Z`` (``|m| <= zero_tol``).  For a step ``d`` small enough to keep the signs on ``N``,
    ``f(w* + d) - f(w*) = s' J_N d + || J_Z d ||_1`` exactly.  The minimiser is unique iff this directional
    derivative is positive for every ``d != 0`` (a sharp minimum, Burke & Ferris 1993), i.e. iff the
    convex cone ``K = { d : s' J_N d + || J_Z d ||_1 <= 0 }`` is ``{0}``.  ``K`` is polyhedral, so for a
    random direction ``g`` the maxima of ``g' d`` and ``-g' d`` over ``K`` intersected with the unit box
    are both zero iff ``K = {0}``, and one of them is positive with probability one otherwise.  Returns
    the largest such maximum over ``probes`` directions: 0 (to solver accuracy) for a unique minimiser,
    of order one for an optimal face of dimension >= 1.  Homogeneous constraints, so no tolerance on
    the minimal norm enters -- only the classification of the entries of ``M(w*)`` as zero or not."""
    rng = np.random.default_rng(0) if rng is None else rng
    nt, nw = sp.jac.shape
    m_off = sp.offdiag(w_star)
    zero = np.abs(m_off) <= zero_tol
    jac_z, jac_n = sp.jac[zero], sp.jac[~zero]
    nz = int(zero.sum())
    grad_n = np.sign(m_off[~zero]) @ jac_n                              # s' J_N
    rows = [hstack([coo_matrix(grad_n[None, :]), coo_matrix(np.ones((1, nz)))])]
    if nz:
        eye = identity(nz, format="coo")
        rows += [hstack([coo_matrix(jac_z), -eye]), hstack([-coo_matrix(jac_z), -eye])]
    a_ub = vstack(rows).tocsc()
    b_ub = np.zeros(a_ub.shape[0])
    bounds = [(-1.0, 1.0)] * nw + [(0.0, None)] * nz
    extent = 0.0
    for _ in range(probes):
        g = rng.normal(size=nw)
        g /= np.linalg.norm(g)
        for sign in (1.0, -1.0):
            res = _solve_lp(np.concatenate([sign * g, np.zeros(nz)]), a_ub, b_ub, bounds, method)
            extent = max(extent, float(-res.fun))
    return extent


def min_l1_exact_fit(sigma_hat: np.ndarray, c: np.ndarray, *, solver: str = "highs", probes: int = 0,
                     rng: np.random.Generator | None = None, zero_tol: float = 1e-7, atol: float = 1e-7,
                     return_info: bool = False):
    """The exact fit with the smallest off-diagonal l1 norm ("basis pursuit").

    The linear program ``min sum(t)`` subject to ``|m0_off + J w| <= t`` in ``(w, t)``, solved with
    ``scipy.optimize.linprog(method=solver)`` (one of the HiGHS methods, tolerances 1e-9).  It is the
    ``lambda -> 0`` limit of the direct-loss lasso and the target of the dense end of the dense -> sparse
    paths (theory note of 8 October 2026, Section 2).  Measured on a laptop (sample correlation matrices):
    0.2 s at ``p = 20``, about 1 s at ``p = 30``, 10 s (``highs-ipm``) to 26 s (simplex) at ``p = 50``.

    ``probes > 0`` additionally probes whether the minimiser is unique: ``probes`` random linear
    objectives are maximised over the directions along which the l1 norm does not increase (its
    optimal face; entries of the fit within ``zero_tol`` count as zero), and the fit counts as unique
    when every maximum is at most ``atol`` (``L1Fit.face_extent`` holds the largest; see
    ``_face_extent``).  The minimiser is not unique, for example, for a graph with an isolated edge
    on the correlation scale: both directions of the edge fit exactly with the same l1 norm, so the
    LP's answer is one of the two at random.  Any minimiser serves as a start; uniqueness matters for
    the theory's "BP condition".

    Returns the matrix, or ``(matrix, L1Fit)`` with ``return_info=True``.  Raises ``RuntimeError`` when
    the solver does not report an optimal solution.
    """
    if solver not in LP_METHODS:
        raise ValueError(f"solver must be one of {LP_METHODS}, got {solver!r}")
    if probes < 0:
        raise ValueError(f"probes must be >= 0, got {probes}")
    sp = _space(sigma_hat, c)
    nt, nw = sp.jac.shape
    a_ub, b_ub, bounds = _lp_constraints(sp)
    res = _solve_lp(np.concatenate([np.zeros(nw), np.ones(nt)]), a_ub, b_ub, bounds, solver)
    w = res.x[:nw]
    value = float(np.abs(sp.offdiag(w)).sum())
    unique, extent = None, 0.0
    if probes:
        extent = _face_extent(sp, w, probes, rng, zero_tol * max(1.0, float(np.abs(sp.offdiag(w)).max())), solver)
        unique = extent <= atol
    m = sp.matrix(w)
    if return_info:
        return m, L1Fit(w=w, value=value, status=int(res.status), unique=unique, face_extent=extent)
    return m


def noisy_truth_start(m_true: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    """``M* + tau N(0, 1)`` on every off-diagonal entry (dense), the diagonal unchanged.

    A ceiling for the dense starts, not an estimator: it uses the truth.  ``tau = 0`` returns a copy
    of ``M*``.  On standardized data pass the standardized truth
    (:func:`gclm.data.simulate.standardized_drift`).  The result need not be stable, which the direct
    loss does not need.
    """
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    m = np.array(m_true, dtype=float, copy=True)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"m_true must be square, got shape {m.shape}")
    noise = rng.normal(size=m.shape)
    np.fill_diagonal(noise, 0.0)
    return m + tau * noise


def exact_fit_residual(m: np.ndarray, sigma_hat: np.ndarray, c: np.ndarray) -> float:
    """``|| M Sigma_hat + Sigma_hat M' + C ||_F``, zero for every exact fit; for tests and diagnostics."""
    return float(np.linalg.norm(lyapunov_residual(np.asarray(m, dtype=float), sigma_hat, c)))
