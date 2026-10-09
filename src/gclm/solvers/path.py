"""Regularisation paths: the lambda grid, warm-started continuation, and the
entry points :func:`lasso_path` (direct loss), :func:`covloss_path` (Varando's
losses) and :func:`fit_path` (either, by name).

Three more ways to compute a path for the direct loss, all of which start from
the lasso instead of the empty graph (next_steps/051026/cluster_campaign_051026.md):
``lasso_path(..., direction="up")`` (MCP / SCAD dense -> sparse), :func:`lla_path`
(MCP / SCAD by local linear approximation) and :func:`adaptive_lasso_path`.

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
from gclm.objective.direct import diagonal_fit, direct_grad, lambda_max
from gclm.objective.penalties import (CONVENTIONS, canonical, lla_weights, penalty_scale,
                                      penalty_weights, resolve_gamma)
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
    direction: str = "down",
    **solver_kwargs,
) -> LassoPath:
    """Fit the whole path by continuation: each lambda is solved starting from
    the solution at the neighbouring lambda (a "warm start").

    ``direction="down"`` (default): sparse -> dense.  Start at ``lambda_max``,
    where the solution is the diagonal fit (the empty graph), and decrease
    lambda.  This is the standard order (glmnet, ncvreg) and the one used in
    every thesis run before October 2026.

    ``direction="up"`` (MCP / SCAD, ``solver="fista"`` only): dense -> sparse.
    Start from the *lasso* solution at the smallest lambda -- dense, with both
    directions of an uncertain edge still present -- and increase lambda, each
    MCP / SCAD problem warm-started from the previous one.  Edges are then
    removed one by one from a fit that has seen all of them, instead of being
    added one by one to a fit that has seen none.

    For the lasso the two orders give the same path (the problem is convex), and
    ``"up"`` simply returns the ``"down"`` path.  For MCP / SCAD they reach
    different stationary points of the same objective: the estimator is the
    objective *plus* the order.  See next_steps/051026/cluster_campaign_051026.md
    and next_steps/051026/orientation_lock_in.md for why the order matters here.

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

    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'")
    if direction == "up" and penalty != "lasso":
        # ---- MCP / SCAD, dense -> sparse -------------------------------------
        if solver != "fista":
            raise ValueError("direction='up' is implemented for solver='fista' only")
        # 1. The dense start: the LASSO solution at the smallest lambda.  The lasso
        #    is convex, so this start is well defined; it is computed by the usual
        #    sparse -> dense lasso path on the same grid (not snapped: zero_tol=0).
        lasso_kwargs = {k: v for k, v in solver_kwargs.items()
                        if k not in ("penalty", "gamma", "convention")}
        dense = lasso_path(sigma, c, lambdas=lambdas, penalize_diagonal=penalize_diagonal,
                           solver="fista", zero_tol=0.0, **lasso_kwargs)
        # 2. Walk the grid from the smallest lambda to the largest.  `fit` is
        #    solve_fista with penalty=MCP/SCAD (the monotone APG of proxgrad.py);
        #    `warm` carries the previous solution into the next problem.
        order = np.argsort(lambdas, kind="stable")
        warm = dense.estimates[order[0]]
        up: list[np.ndarray | None] = [None] * len(lambdas)
        for i in order:
            lam = lambdas[i]
            if lam_max is not None and not penalize_diagonal and lam >= lam_max:
                # 3. At lambda_max the path ends at the diagonal fit, as the
                #    standard path does (it is a stationary point there).
                warm = diagonal_fit(sigma, c)
            else:
                warm = fit(sigma, c, lam, weights=weights, m_init=warm, **solver_kwargs)
            up[i] = warm.copy()
        return LassoPath(lambdas=lambdas, estimates=_snap(up, zero_tol))

    # ---- sparse -> dense (every penalty; and the lasso in either "direction") ----
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


def lla_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray | None = None,
    n_lambda: int = 100,
    ratio: float = 1e-4,
    penalty: str = "MCP",
    gamma: float | None = None,
    steps: int = 2,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    zero_tol: float = 1e-12,
    lasso: LassoPath | None = None,
) -> LassoPath:
    """MCP / SCAD by local linear approximation (LLA), started from the lasso.

    At every lambda the start is the *lasso* solution at that same lambda.  Each
    step then solves a weighted lasso whose weights are the slopes of the penalty
    at the current estimate (:func:`gclm.objective.penalties.lla_weights`):

        w_ij = P'_lam(|M_ij|) / lam           1 at zero, 0 beyond gamma * lam
        M   <- argmin 0.5 ||M Sigma + Sigma M' + C||_F^2 + lam * sum_ij w_ij |M_ij|

    (Zou & Li 2008).  ``steps=2`` is the two-step estimator of Fan, Xue & Zou
    (2014): from a lasso start the first step reaches the oracle estimator with
    high probability and the second confirms it, under conditions that do not
    include irrepresentability.  The loop stops early at a fixed point, and a
    fixed point is a stationary point of the MCP / SCAD objective.

    Unlike the warm-started paths of :func:`lasso_path`, nothing is carried from
    one lambda to the next: every step is a convex problem started from the lasso
    solution at that lambda -- there is no "which direction entered first".
    Textbook convention, diagonal unpenalised.

    One caveat at the dense end.  The design has rank ``p (p + 1) / 2``, so when
    more entries than that have weight 0 (the diagonal plus every entry beyond
    ``gamma * lam``) the weighted lasso has flat directions and its minimiser is not
    unique.  The estimate is then the minimiser that FISTA reaches from the lasso
    solution.  This concerns small lambdas only (tests/test_lla_adaptive.py).

    ``lasso``: a precomputed lasso path on the same grid (optional; saves time).
    The grid defaults to the lasso's.  Estimates in increasing-lambda order.
    """
    penalty = canonical(penalty)
    gamma = resolve_gamma(penalty, gamma)
    if penalty == "lasso":
        raise ValueError("lla_path is for MCP / SCAD; for the lasso use lasso_path")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    p = sigma.shape[0]
    off = penalty_weights(p)
    if lasso is None:
        lasso = lasso_path(sigma, c, lambdas=lambdas, n_lambda=n_lambda, ratio=ratio,
                           tol=tol, max_iter=max_iter, zero_tol=0.0)
    lambdas = np.asarray(lasso.lambdas, dtype=float)
    lam_max = lambda_max(sigma, c)

    estimates: list[np.ndarray] = []
    for lam, m in zip(lambdas, lasso.estimates):
        m = np.array(m, dtype=float)
        if lam < lam_max:          # at lam_max the start is diagonal and every weight is 1
            for _ in range(steps):
                w = off * lla_weights(m, lam, penalty, gamma)
                m_new = solve_fista(sigma, c, lam, weights=w, m_init=m, tol=tol, max_iter=max_iter)
                fixed = np.array_equal(m_new != 0, m != 0) and np.max(np.abs(m_new - m)) < 10 * tol
                m = m_new
                if fixed:
                    break
        estimates.append(m)
    return LassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol))


@dataclass
class AdaptiveLassoPath(LassoPath):
    """:class:`LassoPath` plus what defines the adaptive lasso: the pilot estimate
    and the weights built from it (``inf`` = entry excluded)."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    pilot: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def adaptive_lasso_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray | None = None,
    n_lambda: int = 100,
    ratio: float = 1e-4,
    power: float = 1.0,
    pilot: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    zero_tol: float = 1e-12,
) -> AdaptiveLassoPath:
    """Adaptive lasso (Zou 2006) with weights from the dense end of the lasso path.

    Pilot ``M0``: the lasso solution at the smallest lambda of the lasso's own
    grid (``ratio * lambda_max``), which is close to the minimum-l1 exact solution
    of the Lyapunov equation and keeps both directions of an uncertain edge.

        w_ij = 1 / |M0_ij|^power    scaled so that the smallest weight is 1,
                                    inf where M0_ij = 0 (that entry is excluded)
        M(lam) = argmin 0.5 ||M Sigma + Sigma M' + C||_F^2 + lam * sum_ij w_ij |M_ij|

    Large pilot entries are penalised little, small ones heavily: the same
    "start dense, then prune" idea as MCP / SCAD run dense -> sparse, but with a
    convex second step, so the path does not depend on warm starts.

    The grid is the adaptive problem's own: 100 log-spaced values up to
    ``lam_max = max_ij |grad_ij(M_diag)| / w_ij``, the smallest lambda at which the
    solution is diagonal.  Diagonal unpenalised.  Increasing-lambda order.
    """
    p = sigma.shape[0]
    off = ~np.eye(p, dtype=bool)
    if pilot is None:
        pilot = lasso_path(sigma, c, n_lambda=n_lambda, ratio=ratio, tol=tol, max_iter=max_iter,
                           zero_tol=zero_tol).estimates[0]
    pilot = np.array(pilot, dtype=float)
    size = np.abs(pilot)
    kept = off & (size > 0)
    weights = np.where(off, np.inf, 0.0)
    m_diag = diagonal_fit(sigma, c)
    if not kept.any():             # nothing to select from: the path is the diagonal fit
        lambdas = lambda_grid(1.0, n_lambda, ratio) if lambdas is None else np.asarray(lambdas, float)
        return AdaptiveLassoPath(lambdas=lambdas, estimates=[m_diag.copy() for _ in lambdas],
                                 weights=weights, pilot=pilot)
    weights[kept] = size[kept] ** (-power)
    weights[kept] /= weights[kept].min()

    grad = np.abs(direct_grad(m_diag, sigma, c))
    lam_max = float(np.max(grad[kept] / weights[kept]))
    if lambdas is None:
        lambdas = lambda_grid(lam_max, n_lambda=n_lambda, ratio=ratio)
    lambdas = np.asarray(lambdas, dtype=float)

    estimates: list[np.ndarray | None] = [None] * len(lambdas)
    warm = m_diag
    for i in np.argsort(-lambdas, kind="stable"):
        lam = lambdas[i]
        if lam >= lam_max:
            warm = m_diag
        else:
            warm = solve_fista(sigma, c, lam, weights=weights, m_init=warm, tol=tol, max_iter=max_iter)
        estimates[i] = warm.copy()
    return AdaptiveLassoPath(lambdas=lambdas, estimates=_snap(estimates, zero_tol),
                             weights=weights, pilot=pilot)


DIRECTIONS = ("up", "down")
#: where a covariance-loss dense -> sparse path starts (covloss_path, direction="up")
UP_STARTS = ("exact", "lasso")
#: weight of an entry that the adaptive lasso's pilot sets to zero, on the covariance losses:
#: large and finite, so that the proximal step keeps the entry at zero while its penalty value
#: 0 * weight stays 0 (an infinite weight would make the objective NaN there)
EXCLUDED_WEIGHT = 1e12

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
    start: str = "exact",
) -> CovlossPath:
    """Solutions along a lambda grid, with warm starts, off-diagonal penalised.

    ``direction="down"`` (default) walks from ``lambda_max`` to the dense end,
    starting from :func:`gclm.objective.covariance.diagonal_fit` -- the order used for every other fit in
    this repository and by glmnet/ncvreg.  ``direction="up"`` walks from the dense
    end to ``lambda_max``, starting from

      ``start="exact"`` (default): the exact unpenalised minimiser
          :func:`gclm.objective.covariance.dense_fit`, ``-C Sigma_hat^{-1} / 2`` --
          Varando & Hansen's order (Section 3.1);
      ``start="lasso"``: the lasso solution of the same loss at the smallest lambda
          of the grid, the dense end of its own sparse -> dense lasso path -- the start
          the direct loss's dense -> sparse MCP / SCAD paths use (``lasso_path``).

    Both starts reach the same minimum of the loss (every exact solution of the
    Lyapunov equation does), but the lasso's is the exact fit with the smallest
    l1 norm, which keeps both directions of an uncertain edge and favours sparse
    graphs; ``dense_fit`` has no such preference (campaign of October 2026, wave 7).
    The problems are nonconvex, so the orders and starts can reach different
    stationary points; docs/LIKELIHOOD.md Section 5 compares the orders.

    The grid defaults to Dettling's: 100 log-spaced values from
    ``lambda_max / 1e4`` to ``lambda_max`` (Varando fixed ``lambda_max = 6``).
    Estimates are returned in increasing-lambda order, like ``lasso_path``.
    """
    loss = cov.check_loss(loss)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if start not in UP_STARTS:
        raise ValueError(f"start must be one of {UP_STARTS}")
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
    elif start == "exact":
        order, warm = np.argsort(lambdas, kind="stable"), cov.dense_fit(sigma_hat, c)
    else:
        # the dense end of the lasso path of the same loss, on the same grid
        lasso = covloss_path(sigma_hat, c, loss, lambdas=lambdas, penalty="lasso", direction="down",
                             method=method, tol=tol, newton_after=newton_after, max_iter=max_iter)
        order = np.argsort(lambdas, kind="stable")
        warm = lasso.estimates[int(order[0])]
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


@dataclass
class AdaptiveCovlossPath(CovlossPath):
    """:class:`CovlossPath` plus what defines the adaptive lasso: the pilot and its weights
    (``EXCLUDED_WEIGHT`` = entry excluded, 0 on the diagonal)."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    pilot: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def adaptive_covloss_path(
    sigma_hat: np.ndarray,
    c: np.ndarray,
    loss: str = "loglik",
    lambdas: np.ndarray | None = None,
    n_lambda: int = 100,
    ratio: float = 1e-4,
    power: float = 1.0,
    pilot: np.ndarray | None = None,
    method: str = "apg",
    tol: float | None = None,
    newton_after: float = 1e-4,
    max_iter: int = 50_000,
) -> AdaptiveCovlossPath:
    """Adaptive lasso (Zou 2006) on a covariance loss, built exactly as on the direct loss
    (:func:`adaptive_lasso_path`):

        pilot  M0 = the lasso solution of the same loss at the smallest lambda of its grid,
               the dense end of its sparse -> dense lasso path
        w_ij   = 1 / |M0_ij|^power, scaled so that the smallest weight is 1;
               EXCLUDED_WEIGHT where M0_ij = 0 (the entry stays zero); 0 on the diagonal
        M(lam) = argmin  L(Sigma(M)) + lam * sum_ij w_ij |M_ij|,   L = loss

    on its own grid of 100 log-spaced values up to ``lam_max = max_ij |grad_ij(M_diag)| /
    w_ij``, the smallest lambda at which the diagonal fit is stationary; sparse -> dense from
    :func:`gclm.objective.covariance.diagonal_fit` with warm starts.  Increasing-lambda order,
    like every path here.
    """
    loss = cov.check_loss(loss)
    p = sigma_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    solver_kw = dict(method=method, tol=tol, newton_after=newton_after, max_iter=max_iter)
    if pilot is None:
        lasso = covloss_path(sigma_hat, c, loss, n_lambda=n_lambda, ratio=ratio, penalty="lasso",
                             direction="down", **solver_kw)
        pilot = lasso.estimates[int(np.argmin(lasso.lambdas))]
    pilot = np.array(pilot, dtype=float)
    size = np.abs(pilot)
    kept = off & (size > 0)
    weights = np.where(off, EXCLUDED_WEIGHT, 0.0)
    m_diag = cov.diagonal_fit(sigma_hat, c)
    if kept.any():
        weights[kept] = size[kept] ** (-power)
        weights[kept] /= weights[kept].min()
        grad = np.abs(cov.loss_grad(m_diag, sigma_hat, c, loss))
        lam_max = float(np.max(grad[kept] / weights[kept]))
    else:                                           # nothing to select from
        lam_max = 1.0
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
    warm = m_diag
    for i in np.argsort(-lambdas, kind="stable"):
        if lambdas[i] >= lam_max or not kept.any():
            warm = m_diag                           # stationary there by the definition of lam_max
            conv[i], obj[i] = True, cov.loss_value(m_diag, sigma_hat, c, loss)
        else:
            warm, info = solve_covariance(sigma_hat, c, lambdas[i], loss=loss, weights=weights,
                                          m_init=warm, penalty="lasso", return_info=True, **solver_kw)
            iters[i], newt[i], conv[i] = info.iterations, info.newton_steps, info.converged
            kkt[i], obj[i] = info.stationarity, info.objective
        est[i] = warm.copy()
    return AdaptiveCovlossPath(lambdas=lambdas, estimates=est, iterations=iters, newton_steps=newt,
                               converged=conv, kkt=kkt, objective=obj, weights=weights, pilot=pilot)


def fit_path(sigma_hat, c, loss="direct", **kwargs):
    """One entry point for every loss: ``loss="direct"`` -> :func:`lasso_path`,
    ``"loglik"`` / ``"frobenius"`` -> :func:`covloss_path`.  Keyword arguments
    are passed through; the drivers in ``simulations/`` call this."""
    if loss == "direct":
        return lasso_path(sigma_hat, c, **kwargs)
    return covloss_path(sigma_hat, c, loss, **kwargs)
