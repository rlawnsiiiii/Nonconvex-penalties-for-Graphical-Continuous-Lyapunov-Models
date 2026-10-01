"""Package backends for the direct loss: glmnet and ncvreg (R, through
:mod:`gclm.rbridge`), skglm and pyproximal (Python).  Each fits a whole lambda
path on the regression form ``y = -vec(C)``, ``X = A(Sigma)`` -- see R/ENCODING.md
for the encoding and the lambda conversions.  Selected through
``lasso_path(..., solver=...)``; none is imported unless chosen.
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import design_matrix, unvec, vec
from gclm.objective.direct import lipschitz_bound


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
    """Our lambda -> the lambda handed to ``ncvfit``: the same number.

    ``ncvfit`` minimises ``(1/(2n))||y - Xb||^2 + sum_j P_{lam pf_j, gamma}(b_j)``
    with ``n = nrow(X) = p^2``.  ``R/backend_ncvreg.R`` passes ``sqrt(n) X`` and
    ``sqrt(n) y``, which turns that loss into exactly ``0.5 ||y - Xb||^2`` -- the
    paper scale -- so lambda and gamma need no conversion for any penalty.

    (With an unscaled design the lasso would need ``lam / p^2`` and MCP a
    rescaled gamma, and SCAD would have no equivalent at all: see
    docs/NONCONVEX.md Section 2.)  ``ncvfit`` does not rescale
    ``penalty.factor``, unlike glmnet (``test_fista_matches_ncvreg``).
    ``p`` is kept in the signature for symmetry with :func:`to_glmnet_lambda`.
    """
    return np.asarray(lam, dtype=float)

def _pyproximal_path(
    sigma: np.ndarray,
    c: np.ndarray,
    lambdas: np.ndarray,
    weights: np.ndarray,
    penalty: str = "lasso",
    gamma: float | None = None,
    method: str = "anderson",
    niter: int = 200_000,
    nhistory: int = 5,
    tol: float = 1e-14,
    **_ignored,
) -> list[np.ndarray]:
    """Fit the path with ``pyproximal``, warm-started.

    ``method``: ``"anderson"`` (default) uses ``AndersonProximalGradient``;
    ``"fista"`` uses ``ProximalGradient(acceleration="fista")``.

    Anderson is the better choice here by a wide margin (~10x).  ``pyproximal``
    offers no adaptive restart, so its plain FISTA suffers the momentum
    overshoot that restart exists to prevent; Anderson acceleration cures the
    same pathology by a different mechanism.

    The design is applied matrix-free through a ``pylops`` ``LinearOperator``
    that evaluates ``v -> vec(V Sigma + Sigma V')`` in ``O(p^3)``, so no
    ``p^2 x p^2`` matrix is ever formed -- the same trick the hand-written
    backend uses.  Only the iteration itself comes from the package.
    """
    try:
        import pyproximal
        from pylops import LinearOperator
        from pyproximal.optimization.primal import (
            AndersonProximalGradient,
            ProximalGradient,
        )
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

    if method not in ("anderson", "fista"):
        raise ValueError(f"unknown pyproximal method {method!r}")

    x = np.zeros(p * p)
    estimates: list[np.ndarray] = []
    for lam in np.asarray(lambdas)[::-1]:
        g = pyproximal.L1(sigma=float(lam) * wv)
        if method == "anderson":
            x = AndersonProximalGradient(smooth, g, x0=x, tau=tau,
                                         nhistory=nhistory, niter=niter,
                                         tol=tol, show=False)
        else:
            x = ProximalGradient(smooth, g, x0=x, tau=tau,
                                 acceleration="fista", niter=niter,
                                 tol=tol, show=False)
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
    ``n_samples = p^2``.  As for ncvfit, the design and response are passed
    scaled by ``sqrt(n_samples)``, which makes that exactly ``0.5 ||y - Xw||^2``
    (the paper scale); ``alpha`` and ``gamma`` then equal our lambda and gamma.

    ``WeightedMCPenalty`` multiplies the whole penalty by the weight while
    ncvreg multiplies lambda by it -- identical for the 0/1 weights used here,
    which :mod:`gclm.objective.penalties` enforces.  skglm has no weighted SCAD.
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

    from gclm.objective.penalties import canonical, resolve_gamma

    penalty = canonical(penalty)
    gamma = resolve_gamma(penalty, gamma)
    p = sigma.shape[0]
    n = p * p
    x = np.sqrt(n) * design_matrix(sigma)          # paper scale, see docstring
    y = -np.sqrt(n) * vec(c)
    wv = vec(weights)

    if penalty == "lasso":
        pen = WeightedL1(alpha=1.0, weights=wv)
    elif penalty == "MCP":
        if not np.all((wv == 0) | (wv == 1)):
            raise ValueError("MCP supports penalty weights in {0, 1} only")
        pen = WeightedMCPenalty(alpha=1.0, gamma=gamma, weights=wv)
    else:
        # skglm has no weighted SCAD, so the diagonal cannot be left unpenalized
        if np.any(wv == 0):
            raise ValueError(
                "skglm has no weighted SCAD, so the unpenalized diagonal cannot "
                "be expressed; use solver='fista' or 'ncvreg' for SCAD"
            )
        pen = SCAD(alpha=1.0, gamma=gamma)

    est = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=pen,
        solver=AndersonCD(max_iter=max_iter, max_epochs=100_000, tol=tol,
                          fit_intercept=False, warm_start=True),
    )
    estimates: list[np.ndarray] = []
    for lam in np.asarray(lambdas)[::-1]:
        est.penalty.alpha = float(lam)
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
        from gclm.objective.penalties import canonical

        if canonical(penalty) != "lasso":
            raise ValueError("the glmnet backend supports penalty='lasso' only; "
                             "use solver='fista', 'ncvreg' or 'skglm' for MCP/SCAD")
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
        from gclm.objective.penalties import canonical, resolve_gamma

        penalty = canonical(penalty)
        payload["lambda"] = to_ncvreg_lambda(lambdas, p).tolist()
        payload["penalty"] = penalty
        g = resolve_gamma(penalty, gamma)
        if g is not None:
            payload["gamma"] = g
        out = run_r(script, payload, packages)
        beta = np.asarray(out["beta"])          # already in input order

    if beta.shape[0] != len(lambdas):
        raise RuntimeError(
            f"{solver} returned {beta.shape[0]} fits for {len(lambdas)} lambdas"
        )
    return [unvec(b, p) for b in beta]
