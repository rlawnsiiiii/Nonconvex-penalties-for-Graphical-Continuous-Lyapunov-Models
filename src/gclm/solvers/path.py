"""Regularisation paths: the lambda grid, warm-started continuation, and the
entry points :func:`lasso_path` (direct loss), :func:`covloss_path` (Varando's
losses) and :func:`fit_path` (either, by name).

For the direct loss, interchangeable backends minimise

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
             MCP/SCAD under the *ncvreg* convention only (``convention=``).
``skglm``    Python, ``skglm`` (JMLR 26, 2025).  Coordinate descent with
             Anderson acceleration on the explicit design.  Textbook MCP
             without R; SCAD only with the diagonal penalised.
``pyproxim`` Python, ``pyproximal`` + ``pylops``, driven matrix-free through a
             custom ``LinearOperator``.  Defaults to ``AndersonProximalGradient``
             (``method="anderson"``); ``method="fista"`` selects the plain FISTA,
             which is ~10x slower because ``pyproximal`` offers no adaptive
             restart.
``fista``    Python, hand-written accelerated proximal gradient with the adaptive
             restart of O'Donoghue & Candes (2015).  **The default.**  MCP/SCAD
             via the monotone APG of Li & Lin (2015), under either convention
             (docs/NONCONVEX.md).  The only backend that scales to ``p = 50``;
             validated in
             ``tests/test_fista.py`` against an analytic solution, a duality-gap
             certificate, ``cvxpy``/CLARABEL and every other backend.
===========  ===================================================================

Accuracy and cost differ; see ``simulations/S1_reproduction.md`` Section 7.2.
Briefly: ``ncvreg`` is the most accurate, ``glmnet`` the least, ``fista`` the
only one viable for the full grid.

``W`` is the per-entry penalty weight matrix from
:func:`gclm.objective.penalties.penalty_weights`: 1 off the diagonal, 0 on it.

The covariance losses have one solver family of their own
(:mod:`gclm.solvers.covariance`); ``covloss_path`` drives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gclm.objective import covariance as cov
from gclm.objective.direct import diagonal_fit, lambda_max
from gclm.objective.penalties import CONVENTIONS, canonical, penalty_scale, penalty_weights, resolve_gamma
from gclm.solvers.backends import _R_BACKENDS, _pyproximal_path, _r_path, _skglm_path
from gclm.solvers.coordinate import solve_design
from gclm.solvers.covariance import solve as solve_covariance
from gclm.solvers.proxgrad import solve_fista


def lambda_grid(lam_max: float, n_lambda: int = 100, ratio: float = 1e-4) -> np.ndarray:
    """``lambda_1 = lam_max * ratio < ... < lambda_n = lam_max``, log-equidistant.

    Dettling Section 5 / Varando ``10^(seq(-4, 0, length = 100))``.
    Returned in *increasing* order, as written in the papers.
    """
    return lam_max * np.logspace(np.log10(ratio), 0.0, n_lambda)

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

    # lambda_max is needed for the grid, and again below to short-circuit the
    # sparse end of the path: for lam >= lambda_max the minimiser is exactly the
    # diagonal least-squares fit, by the definition of lambda_max.  With the
    # diagonal unpenalised it is a closed form (O(p^3)), so computing it even
    # when the caller supplied its own grid is cheap.
    lam_max: float | None = None
    if lambdas is None or not penalize_diagonal:
        lam_max = lambda_max(sigma, c, penalize_diagonal=penalize_diagonal)
    if lambdas is None:
        lambdas = lambda_grid(lam_max, n_lambda=n_lambda, ratio=ratio)

    lambdas = np.asarray(lambdas, dtype=float)
    solver_kwargs = dict(solver_kwargs)

    # Penalty and convention are resolved once, here, so that every backend
    # receives the same canonical (penalty, gamma) -- and so that no backend is
    # ever asked for a convention it does not actually solve.
    penalty = canonical(solver_kwargs.pop("penalty", "lasso"))
    gamma = resolve_gamma(penalty, solver_kwargs.pop("gamma", None))
    convention = solver_kwargs.pop("convention", "textbook")
    penalty_scale(sigma, convention)                       # validates the name
    if penalty != "lasso":
        solves = {"fista": CONVENTIONS, "ncvreg": ("ncvreg",), "skglm": ("textbook",)}
        if solver not in solves:
            raise ValueError(
                f"solver {solver!r} implements the l1 penalty only; use 'fista', "
                f"'ncvreg' or 'skglm' for {penalty}"
            )
        if convention not in solves[solver]:
            raise ValueError(
                f"solver {solver!r} solves {penalty} under the "
                f"{' / '.join(solves[solver])} convention only, not {convention!r}. "
                "ncvreg measures gamma relative to each coordinate's loss curvature, "
                "skglm applies the textbook penalty; 'fista' does either. "
                "See docs/NONCONVEX.md Section 2."
            )
    solver_kwargs.update(penalty=penalty, gamma=gamma)

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
    fit = solve_fista if solver == "fista" else solve_design
    if solver == "fista":
        solver_kwargs["convention"] = convention
    else:                                   # design: l1 only, validated above
        solver_kwargs.pop("penalty")
        solver_kwargs.pop("gamma")
        solver_kwargs["_cache"] = {}

    estimates: list[np.ndarray] = []
    warm = None
    for lam in lambdas[::-1]:
        if lam_max is not None and not penalize_diagonal and lam >= lam_max:
            warm = diagonal_fit(sigma, c)  # exact: KKT holds by definition of lam_max
        else:
            warm = fit(sigma, c, lam, weights=weights, m_init=warm, **solver_kwargs)
        estimates.append(warm.copy())
    estimates.reverse()
    return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))


DIRECTIONS = ("up", "down")

@dataclass
class CovlossPath(LassoPath):
    """:class:`LassoPath` plus per-lambda solver diagnostics."""

    iterations: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    newton_steps: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    converged: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    kkt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    objective: np.ndarray = field(default_factory=lambda: np.zeros(0))

def covloss_path(
    sigma_hat: np.ndarray,
    c: np.ndarray,
    loss: str = "loglik",
    lambdas: np.ndarray | None = None,
    n_lambda: int = 100,
    ratio: float = 1e-4,
    penalty: str = "lasso",
    gamma: float | None = None,
    direction: str = "down",
    method: str = "apg",
    tol: float | None = None,
    newton_after: float = 1e-4,
    max_iter: int = 50_000,
) -> CovlossPath:
    """Solutions along a lambda grid, with warm starts, off-diagonal penalised.

    ``direction="down"`` (default) walks from ``lambda_max`` to the dense end,
    starting from :func:`gclm.objective.covariance.diagonal_fit` -- the order used for every other fit in
    this repository and by glmnet/ncvreg.  ``direction="up"`` is Varando &
    Hansen's order (Section 3.1): start from the exact unpenalised minimiser
    :func:`gclm.objective.covariance.dense_fit` and increase ``lam``.  The problems are nonconvex, so the
    two orders can reach different stationary points; docs/LIKELIHOOD.md
    Section 5 compares them.

    The grid defaults to Dettling's: 100 log-spaced values from
    ``lambda_max / 1e4`` to ``lambda_max`` (Varando fixed ``lambda_max = 6``).
    Estimates are returned in increasing-lambda order, like ``lasso_path``.
    """
    loss = cov.check_loss(loss)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    penalty = canonical(penalty)
    gamma = resolve_gamma(penalty, gamma)
    p = sigma_hat.shape[0]
    weights = penalty_weights(p)
    lam_max = cov.lambda_max(sigma_hat, c, loss)
    if lambdas is None:
        lambdas = lambda_grid(lam_max, n_lambda=n_lambda, ratio=ratio)
    lambdas = np.asarray(lambdas, dtype=float)
    n = len(lambdas)

    est = [None] * n
    iters = np.zeros(n, dtype=np.int32)
    newt = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    kkt = np.zeros(n)
    obj = np.zeros(n)

    if direction == "down":
        order, warm = np.argsort(-lambdas, kind="stable"), cov.diagonal_fit(sigma_hat, c)
    else:
        order, warm = np.argsort(lambdas, kind="stable"), cov.dense_fit(sigma_hat, c)
    for i in order:
        warm, info = solve_covariance(sigma_hat, c, lambdas[i], loss=loss, weights=weights,
                           m_init=warm, penalty=penalty, gamma=gamma, method=method,
                           tol=tol, newton_after=newton_after, max_iter=max_iter,
                           return_info=True)
        est[i] = warm.copy()
        iters[i], newt[i], conv[i] = info.iterations, info.newton_steps, info.converged
        kkt[i], obj[i] = info.stationarity, info.objective

    return CovlossPath(lambdas=lambdas, estimates=est, iterations=iters,
                       newton_steps=newt, converged=conv, kkt=kkt, objective=obj)


def fit_path(sigma_hat, c, loss="direct", **kwargs):
    """One entry point for every loss: ``loss="direct"`` -> :func:`lasso_path`,
    ``"loglik"`` / ``"frobenius"`` -> :func:`covloss_path`.  Keyword arguments
    are passed through; the drivers in ``simulations/`` call this."""
    if loss == "direct":
        return lasso_path(sigma_hat, c, **kwargs)
    return covloss_path(sigma_hat, c, loss, **kwargs)
