"""Direct Lyapunov Lasso solvers.

Four interchangeable backends for

    argmin_M  0.5 * ||M Sigma + Sigma M' + C||_F^2 + lam * sum(W * |M|)      (1.4)

selected with ``lasso_path(..., solver=...)``:

===========  ===================================================================
``fista``    matrix-free accelerated proximal gradient (default).  ``O(p^3)`` per
             iteration, never forms the design matrix.  The only backend that
             scales to the full ``p = 50`` grid.
``design``   cyclic coordinate descent on the explicit ``A(Sigma)``, pure Python.
             Transparent; used in tests and at small ``p``.
``glmnet``   R, via ``R/backend_glmnet.R`` -- **Dettling's own choice**
             (Appendix A) and Varando's ``lassoB()``.  Coordinate descent.
             Its ``thresh`` cannot go below ~1e-10 here: tighter and it stops
             converging near the dense end of the path.
``ncvreg``   R, via ``R/backend_ncvreg.R`` using ``ncvreg::ncvfit`` -- the
             low-level fitter, which neither standardizes nor adds an intercept.
             Reaches the exact KKT solution in tens of iterations, and is the
             route to MCP/SCAD in S1b (``penalty=``).
``skglm``    Python, ``skglm`` (Bertrand et al., JMLR 2025).  Coordinate descent
             with Anderson acceleration on the explicit design.  Offers MCP
             **without R**.
``pyproxim`` Python, ``pyproximal`` + ``pylops``.  A packaged FISTA, driven
             matrix-free through a custom ``LinearOperator``.  Its FISTA has no
             adaptive restart, which costs it badly on this singular-Hessian
             problem: ~38x slower than ``fista`` for far lower accuracy.
``fista``    Python, hand-written accelerated proximal gradient with the adaptive
             restart of O'Donoghue & Candes (2015).  **The default.**  The only
             backend that scales to ``p = 50``; validated in
             ``tests/test_fista.py`` against an analytic solution, a duality-gap
             certificate, ``cvxpy``/CLARABEL and every other backend.
===========  ===================================================================

Accuracy and cost differ; see ``simulations/S1_reproduction.md`` Section 7.2.
Briefly: ``ncvreg`` is the most accurate, ``glmnet`` the least, ``fista`` the
only one viable for the full grid.

``W`` is the per-entry penalty weight matrix.  ``penalty_weights`` builds the
default used throughout S1: 1 off the diagonal, 0 on it (matching Varando's
``penalty.factor = 1 - diag(p)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gclm.loss import frobenius_grad, lipschitz_bound, objective
from gclm.lyap import design_matrix, lyapunov_residual, unvec, vec


def penalty_weights(p: int, penalize_diagonal: bool = False) -> np.ndarray:
    """Penalty weight matrix; diagonal unpenalized by default."""
    w = np.ones((p, p))
    if not penalize_diagonal:
        np.fill_diagonal(w, 0.0)
    return w


def _soft_threshold(z: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


# --------------------------------------------------------------------------- #
# Reference solver: coordinate descent on the explicit design matrix
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Workhorse: matrix-free FISTA
# --------------------------------------------------------------------------- #


def solve_fista(
    sigma: np.ndarray,
    c: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
    m_init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    step: float | None = None,
) -> np.ndarray:
    """Accelerated proximal gradient with adaptive restart.

    Stops when the max-norm coefficient change falls below ``tol``.
    """
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
        grad = frobenius_grad(y, sigma, c)
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


# --------------------------------------------------------------------------- #
# lambda_max and the regularization path
# --------------------------------------------------------------------------- #


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


def lambda_grid(lam_max: float, n_lambda: int = 100, ratio: float = 1e-4) -> np.ndarray:
    """``lambda_1 = lam_max * ratio < ... < lambda_n = lam_max``, log-equidistant.

    Dettling Section 5 / Varando ``10^(seq(-4, 0, length = 100))``.
    Returned in *increasing* order, as written in the papers.
    """
    return lam_max * np.logspace(np.log10(ratio), 0.0, n_lambda)


# --------------------------------------------------------------------------- #
# R backends
# --------------------------------------------------------------------------- #

#: script + required R packages for each R-backed solver
_R_BACKENDS = {
    "glmnet": ("backend_glmnet.R", ("glmnet", "jsonlite")),
    "ncvreg": ("backend_ncvreg.R", ("ncvreg", "jsonlite")),
}


def to_glmnet_lambda(lam, p: int):
    """Our lambda -> glmnet's.

    glmnet minimizes ``(1/(2n))||y-Xb||^2 + lam_g * v_j |b_j|`` with
    ``n = nrow(X) = p^2``, and internally rescales ``penalty.factor`` to sum to
    ``nvars``.  With ``1 - diag(p)`` that turns each off-diagonal weight into
    ``p^2/(p^2-p) = p/(p-1)``, so ``lam = lam_g * p^3/(p-1)``.
    """
    return np.asarray(lam) * (p - 1) / p ** 3


def from_glmnet_lambda(lam_g, p: int):
    """Inverse of :func:`to_glmnet_lambda`."""
    return np.asarray(lam_g) * p ** 3 / (p - 1)


def to_ncvreg_lambda(lam, p: int):
    """Our lambda -> ncvfit's.

    ``ncvfit`` minimizes ``(1/(2n))||y-Xb||^2 + lam_n * pf_j * pen(b_j)`` with
    ``n = p^2``.  Unlike glmnet it does **not** rescale ``penalty.factor``
    (established empirically in ``test_fista_matches_ncvreg``), so
    ``lam = lam_n * p^2``.
    """
    return np.asarray(lam) / p ** 2


def _pyproximal_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray,
    weights: np.ndarray,
    penalty: str = "lasso",
    niter: int = 20_000,
    tol: float = 1e-14,
    **_ignored,
) -> list[np.ndarray]:
    """Fit the path with ``pyproximal``'s FISTA, warm-started.

    The design is applied matrix-free through a ``pylops`` ``LinearOperator``
    that evaluates ``v -> vec(V Sigma + Sigma V')`` in ``O(p^3)``, so no
    ``p^2 x p^2`` matrix is ever formed -- the same trick the hand-written
    backend uses.  Only the iteration itself comes from the package.
    """
    try:
        import pyproximal
        from pylops import LinearOperator
        from pyproximal.optimization.primal import ProximalGradient
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "solver='pyproximal' needs: pip install pyproximal pylops"
        ) from exc

    if penalty != "lasso":
        raise ValueError("the pyproximal backend supports penalty='lasso' only")

    p = sigma.shape[0]
    s_mat = np.asarray(sigma, dtype=float)

    class _LyapunovOperator(LinearOperator):
        """``vec(M) -> vec(M Sigma + Sigma M')`` and its adjoint, both O(p^3)."""

        def __init__(self):
            super().__init__(dtype=np.dtype(float), shape=(p * p, p * p))

        def _matvec(self, v):
            m = unvec(v, p)
            return vec(m @ s_mat + s_mat @ m.T)

        def _rmatvec(self, u):
            m = unvec(u, p)
            return vec((m + m.T) @ s_mat)

    op = _LyapunovOperator()
    y = -vec(c)
    wv = vec(weights)
    tau = 1.0 / lipschitz_bound(s_mat)
    smooth = pyproximal.L2(Op=op, b=y)

    x = np.zeros(p * p)
    estimates: list[np.ndarray] = []
    for lam in np.asarray(lambdas)[::-1]:
        x = ProximalGradient(
            smooth, pyproximal.L1(sigma=float(lam) * wv), x0=x, tau=tau,
            acceleration="fista", niter=niter, tol=tol, show=False,
        )
        estimates.append(unvec(x, p).copy())
    estimates.reverse()
    return estimates


def _skglm_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray,
    weights: np.ndarray,
    penalty: str = "lasso",
    gamma: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
    **_ignored,
) -> list[np.ndarray]:
    """Fit the path with ``skglm``, warm-started from large to small lambda.

    ``skglm``'s ``Quadratic`` datafit is ``1/(2 n_samples) ||y - Xw||^2`` with
    ``n_samples = p^2``, and its penalties apply ``alpha * weights`` with no
    rescaling, so ``alpha = lam / p^2`` -- the same convention as ``ncvfit``.
    """
    try:
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import Quadratic
        from skglm.penalties import SCAD, WeightedL1, WeightedMCPenalty
        from skglm.solvers import AndersonCD
    except ImportError as exc:  # pragma: no cover - exercised only without skglm
        raise ImportError(
            "solver='skglm' needs the skglm package: pip install skglm"
        ) from exc

    p = sigma.shape[0]
    x = design_matrix(sigma)
    y = -vec(c)
    n = x.shape[0]
    wv = vec(weights)

    if penalty == "lasso":
        pen = WeightedL1(alpha=1.0, weights=wv)
    elif penalty == "MCP":
        pen = WeightedMCPenalty(alpha=1.0, gamma=gamma if gamma is not None else 3.0,
                                weights=wv)
    elif penalty == "SCAD":
        # skglm has no weighted SCAD, so the diagonal cannot be left unpenalized
        if np.any(wv == 0):
            raise ValueError(
                "skglm has no weighted SCAD, so the unpenalized diagonal cannot "
                "be expressed; use solver='ncvreg' for SCAD"
            )
        pen = SCAD(alpha=1.0, gamma=gamma if gamma is not None else 3.7)
    else:
        raise ValueError(f"unknown penalty {penalty!r}")

    est = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=pen,
        solver=AndersonCD(max_iter=max_iter, max_epochs=100_000, tol=tol,
                          fit_intercept=False, warm_start=True),
    )
    estimates: list[np.ndarray] = []
    for lam in np.asarray(lambdas)[::-1]:
        est.penalty.alpha = float(lam) / n
        est.fit(x, y)
        estimates.append(unvec(est.coef_, p).copy())
    estimates.reverse()
    return estimates


def _r_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray,
    solver: str,
    penalty: str = "lasso",
    gamma: float | None = None,
    thresh: float | None = None,
    **_ignored,
) -> list[np.ndarray]:
    """Fit the whole path in one R call and return estimates in lambda order."""
    from gclm.rbridge import run_r  # local import: R stays optional

    script, packages = _R_BACKENDS[solver]
    p = sigma.shape[0]
    payload = {"Sigma": np.asarray(sigma).tolist(), "C": np.asarray(c).tolist()}

    if solver == "glmnet":
        if penalty != "lasso":
            raise ValueError("the glmnet backend supports penalty='lasso' only; "
                             "use solver='ncvreg' for MCP/SCAD")
        # glmnet wants lambdas decreasing and returns them in that order
        payload["lambda_glmnet"] = sorted(to_glmnet_lambda(lambdas, p).tolist(),
                                          reverse=True)
        if thresh is not None:
            payload["thresh"] = float(thresh)
        out = run_r(script, payload, packages)
        lam_back = from_glmnet_lambda(np.asarray(out["lambda_glmnet"]), p)
        beta = np.asarray(out["beta"])
        order = np.argsort(lam_back)            # back to increasing lambda
        beta = beta[order]
    else:
        payload["lambda_ncv"] = to_ncvreg_lambda(lambdas, p).tolist()
        payload["penalty"] = penalty
        if gamma is not None:
            payload["gamma"] = float(gamma)
        out = run_r(script, payload, packages)
        beta = np.asarray(out["beta"])          # already in input order

    if beta.shape[0] != len(lambdas):
        raise RuntimeError(
            f"{solver} returned {beta.shape[0]} fits for {len(lambdas)} lambdas"
        )
    return [unvec(b, p) for b in beta]


def _snap(estimates: list[np.ndarray], zero_tol: float) -> list[np.ndarray]:
    """Set entries below ``zero_tol`` in absolute value to exactly zero."""
    if zero_tol <= 0:
        return estimates
    out = []
    for m in estimates:
        m = np.array(m, dtype=float, copy=True)
        m[np.abs(m) < zero_tol] = 0.0
        out.append(m)
    return out


@dataclass
class LassoPath:
    """Solutions along the regularization path, ordered by increasing lambda."""

    lambdas: np.ndarray
    estimates: list[np.ndarray] = field(default_factory=list)

    def supports(self, include_diagonal: bool = False) -> list[np.ndarray]:
        out = []
        for m in self.estimates:
            s = m != 0.0
            if not include_diagonal:
                s = s.copy()
                np.fill_diagonal(s, False)
            out.append(s)
        return out


def lasso_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray | None = None,
    n_lambda: int = 100,
    ratio: float = 1e-4,
    penalize_diagonal: bool = False,
    solver: str = "fista",
    zero_tol: float = 1e-12,
    **solver_kwargs,
) -> LassoPath:
    """Fit the whole path, warm-starting from the sparse end downwards.

    ``zero_tol``: entries with ``|M_ij| < zero_tol`` are snapped to exactly zero.

    Support recovery is decided by ``M_hat != 0`` (Dettling Definition G.4), so
    it matters that "zero" means the same thing in every backend.  FISTA's
    proximal step produces exact zeros; the coordinate-descent backends can leave
    dust of order 1e-13, most visibly at ``lambda_max``, where one off-diagonal
    coefficient sits exactly on the threshold.  Without this snap the same
    solution can be scored with a different number of edges depending on the
    solver.  Set ``zero_tol=0`` to disable.
    """
    p = sigma.shape[0]
    weights = penalty_weights(p, penalize_diagonal=penalize_diagonal)
    if lambdas is None:
        lambdas = lambda_grid(
            lambda_max(sigma, c, penalize_diagonal=penalize_diagonal),
            n_lambda=n_lambda,
            ratio=ratio,
        )
    lambdas = np.asarray(lambdas, dtype=float)
    solver_kwargs = dict(solver_kwargs)

    if solver == "skglm":
        estimates = _skglm_path(sigma, c, lambdas, weights, **solver_kwargs)
        return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))

    if solver == "pyproximal":
        estimates = _pyproximal_path(sigma, c, lambdas, weights, **solver_kwargs)
        return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))

    if solver in _R_BACKENDS:
        if penalize_diagonal:
            raise ValueError(
                f"solver {solver!r} always leaves the diagonal unpenalized; "
                "use solver='fista' or 'design' for penalize_diagonal=True"
            )
        estimates = _r_path(sigma, c, lambdas, solver, **solver_kwargs)
        return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))

    if solver not in ("fista", "design"):
        raise ValueError(
            f"unknown solver {solver!r}; expected one of "
            f"{('fista', 'design', 'skglm', 'pyproximal') + tuple(_R_BACKENDS)}"
        )
    # the Python solvers implement the l1 prox only; drop the penalty knobs that
    # are meaningful to the ncvreg backend alone
    penalty = solver_kwargs.pop("penalty", "lasso") or "lasso"
    solver_kwargs.pop("gamma", None)
    if penalty != "lasso":
        raise ValueError(
            f"solver {solver!r} implements the l1 penalty only; "
            f"use solver='ncvreg' for {penalty}"
        )

    fit = solve_fista if solver == "fista" else solve_design
    cache: dict = {}
    if solver == "design":
        solver_kwargs = {**solver_kwargs, "_cache": cache}

    estimates: list[np.ndarray] = []
    warm = None
    for lam in lambdas[::-1]:  # decreasing lambda, warm start
        warm = fit(sigma, c, lam, weights=weights, m_init=warm, **solver_kwargs)
        estimates.append(warm.copy())
    estimates.reverse()
    return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))
