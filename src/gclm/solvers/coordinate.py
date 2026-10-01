"""Reference solver for the direct loss: cyclic coordinate descent on the
explicit design ``A(Sigma)``, pure Python.  Transparent, ``O(p^4)`` per sweep;
used in tests and at small ``p`` (``solver="design"``).
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import design_matrix, unvec, vec
from gclm.objective.penalties import penalty_weights
from gclm.solvers.proxgrad import _soft_threshold


def solve_design(
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
    m_init: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 10_000,
    _cache: dict | None = None,
) -> np.ndarray:
    """Cyclic coordinate descent on ``X = A(Sigma)``, ``y = -vec(C)``.

    ``_cache`` optionally carries the precomputed Gram matrix across a path so it
    is built once; pass the same dict on every call for a fixed ``sigma``.
    """
    p = sigma.shape[0]
    weights = penalty_weights(p) if weights is None else weights

    if _cache is not None and "gram" in _cache:
        gram, xty = _cache["gram"], _cache["xty"]
    else:
        x = design_matrix(sigma)
        gram = x.T @ x
        xty = x.T @ (-vec(c))
        if _cache is not None:
            _cache["gram"], _cache["xty"] = gram, xty

    diag = np.diag(gram).copy()
    w = vec(weights)
    b = np.zeros(p * p) if m_init is None else vec(m_init).copy()
    gb = gram @ b

    for _ in range(max_iter):
        delta_max = 0.0
        for j in range(p * p):
            if diag[j] <= 0.0:
                continue
            rho = xty[j] - gb[j] + diag[j] * b[j]
            new = _soft_threshold(rho, lam * w[j]) / diag[j]
            delta = new - b[j]
            if delta != 0.0:
                gb += gram[:, j] * delta
                b[j] = new
                delta_max = max(delta_max, abs(delta))
        if delta_max < tol:
            break
    return unvec(b, p)
