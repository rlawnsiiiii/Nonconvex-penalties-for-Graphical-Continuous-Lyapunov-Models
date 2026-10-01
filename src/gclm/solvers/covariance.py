"""Solvers for the covariance losses (:mod:`gclm.objective.covariance`).

Three descent methods for  min_M  L(Sigma(M, C)) + sum_ij w_ij P(M_ij)  over stable M.

``"apg"``     accelerated proximal gradient with restart -- **the estimator's
              solver**, used by every simulation run.
``"prox"``    the plain proximal-gradient method, Varando & Hansen's Algorithm 1
              with Barzilai-Borwein steps -- the reference ``"apg"`` is checked
              against; slow.
``"newton"``  active-set Newton with the exact Hessian -- **experimental only**.
              Built to converge faster through the flat valley of these losses;
              it does, but to different stationary points than the first-order
              path, with worse support recovery.  Kept so that the comparison
              in docs/LIKELIHOOD.md Section 5 can be re-run; not used by any
              simulation and not part of the estimator.

Which method is used is part of the definition of the estimator, because the
problems are nonconvex (docs/LIKELIHOOD.md Sections 4-5).  The path is driven
from :func:`gclm.solvers.path.covloss_path`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gclm.objective.covariance import _Point, _evaluate, _grad_at, _hessian_at, check_loss, diagonal_fit
from gclm.objective.penalties import (
    canonical,
    derivative,
    penalty_weights,
    prox,
    resolve_gamma,
    stationarity,
    value,
)


METHODS = ("apg", "prox", "newton")

@dataclass
class FitInfo:
    iterations: int                   # proximal-gradient steps
    newton_steps: int
    converged: bool
    stationarity: float               # max first-order violation (penalties.stationarity)
    objective: float

def _step_cap(penalty: str, gamma: float | None) -> float:
    """Largest step for which the MCP/SCAD proximal problem stays convex."""
    if penalty == "MCP":
        return 0.99 * gamma
    if penalty == "SCAD":
        return 0.99 * (gamma - 1.0)
    return np.inf

def _penalty_curvature(m, lam, penalty, gamma) -> np.ndarray:
    """``P''(|M_ij|)`` where it exists: 0 (lasso), ``-1/gamma`` (MCP, inside
    ``gamma lam``), ``-1/(gamma-1)`` (SCAD, between ``lam`` and ``gamma lam``)."""
    a = np.abs(m)
    if penalty == "MCP":
        return np.where(a < gamma * lam, -1.0 / gamma, 0.0)
    if penalty == "SCAD":
        return np.where((a > lam) & (a < gamma * lam), -1.0 / (gamma - 1.0), 0.0)
    return np.zeros_like(a)

class _Problem:
    """One ``(sigma_hat, c, lam, loss, penalty)`` instance and its iterate."""

    def __init__(self, sigma_hat, c, lam, loss, weights, penalty, gamma, m_init):
        self.sigma_hat, self.c, self.lam, self.loss = sigma_hat, c, lam, loss
        self.weights, self.penalty, self.gamma = weights, penalty, gamma
        self.off = weights == 1
        self.pt = _evaluate(np.array(m_init, float), c, sigma_hat, loss)
        if not np.isfinite(self.pt.f):
            raise ValueError("m_init must be stable")
        self.grad = _grad_at(self.pt, sigma_hat, loss)
        self.big_f = self.pt.f + self.pen(self.pt.m)
        self.n_prox = self.n_newton = 0

    def pen(self, m):
        return value(m, self.lam, self.weights, self.penalty, self.gamma)

    def kkt(self) -> float:
        return stationarity(self.pt.m, self.grad, self.lam, self.weights,
                            self.penalty, self.gamma)

    def _slack(self) -> float:
        return 1e-13 * max(1.0, abs(self.big_f))

    def accept(self, new: _Point):
        self.pt = new
        self.grad = _grad_at(new, self.sigma_hat, self.loss)
        self.big_f = new.f + self.pen(new.m)

    def prox_step(self, t: float, base: _Point | None = None,
                  base_grad: np.ndarray | None = None) -> tuple[bool, float, np.ndarray | None]:
        """One proximal-gradient step from ``base`` (default: the iterate), with
        backtracking on ``t``.

        Accepts ``M+ = prox_{tP}(Y - t grad L(Y))`` iff ``M+`` is stable, the
        quadratic upper bound ``L(M+) <= L(Y) + <grad, M+ - Y> + ||M+ - Y||^2 / (2t)``
        holds, and the objective does not rise above the current iterate's --
        Varando & Hansen's test, made monotone in the iterate.  Returns
        ``(accepted, t, s)`` with ``s = M+ - Y``.
        """
        pt = self.pt if base is None else base
        grad = self.grad if base_grad is None else base_grad
        while t >= 1e-20:
            m_new = prox(pt.m - t * grad, t, self.lam, self.weights, self.penalty, self.gamma)
            new = _evaluate(m_new, self.c, self.sigma_hat, self.loss)
            if np.isfinite(new.f):
                d = m_new - pt.m
                quad = pt.f + float(np.sum(grad * d)) + float(np.sum(d * d)) / (2.0 * t)
                if new.f <= quad + self._slack():          # curvature condition met at this t
                    if new.f + self.pen(m_new) <= self.big_f + self._slack():
                        self.accept(new)
                        self.n_prox += 1
                        return True, t, d
                    return False, t, None                   # uphill from the iterate: caller restarts
            t *= 0.5
        return False, t, None

    def newton_step(self) -> tuple[bool, bool]:
        """One modified-Newton step on the active set, with a sign-preserving
        line search.  Returns ``(accepted, support_changed)``.

        The active set is every nonzero entry plus the (unpenalised) diagonal.
        On it the objective is smooth: ``L`` plus the penalty with the signs
        frozen.  The Hessian is shifted to be positive definite (its smallest
        eigenvalue plus a margin), which turns negative-curvature directions into
        descent directions.  A penalised entry whose sign would flip is set to
        zero instead (the orthant projection of Andrew & Gao 2007), and the
        caller returns to proximal-gradient steps if that happens.
        """
        pt, lam, pen, gam, w = self.pt, self.lam, self.penalty, self.gamma, self.weights
        rows, cols = np.nonzero((pt.m != 0) | ~self.off)
        g_a = (self.grad + derivative(pt.m, lam, w, pen, gam))[rows, cols]
        h = _hessian_at(pt, self.sigma_hat, self.loss, rows, cols)
        h[np.diag_indices_from(h)] += (_penalty_curvature(pt.m, lam, pen, gam) * w)[rows, cols]
        evals, evecs = np.linalg.eigh(h)
        shift = max(0.0, -evals[0]) + 1e-10 * max(1.0, abs(evals[-1]))
        d_a = -(evecs @ ((evecs.T @ g_a) / (evals + shift)))
        slope = float(g_a @ d_a)
        if slope >= 0:                                      # not a descent direction
            return False, False
        alpha = 1.0
        while alpha >= 1e-12:
            m_new = pt.m.copy()
            m_new[rows, cols] += alpha * d_a
            cross = self.off & (pt.m != 0) & (np.sign(m_new) != np.sign(pt.m))
            m_new[cross] = 0.0
            new = _evaluate(m_new, self.c, self.sigma_hat, self.loss)
            if np.isfinite(new.f):
                f_new = new.f + self.pen(m_new)
                if f_new <= self.big_f + 1e-4 * alpha * slope + self._slack():
                    self.accept(new)
                    self.n_newton += 1
                    return True, bool(cross.any())
            alpha *= 0.5
        return False, False

def solve(
    sigma_hat: np.ndarray,
    c: np.ndarray,
    lam: float,
    loss: str = "loglik",
    weights: np.ndarray | None = None,
    m_init: np.ndarray | None = None,
    penalty: str = "lasso",
    gamma: float | None = None,
    method: str = "apg",
    tol: float | None = None,
    newton_after: float = 1e-4,
    max_iter: int = 50_000,
    return_info: bool = False,
):
    """Minimise ``L(Sigma(M, C)) + sum w_ij P(M_ij)`` from ``m_init`` (default:
    the diagonal fit) to a stationary point.  All methods are descent methods:
    the objective never increases, and every iterate is stable.  The problems
    are nonconvex with several stationary points per ``lam``; which one is
    reached depends on the method -- docs/LIKELIHOOD.md Section 5 -- so the
    choice of method is part of the definition of the estimator.

    ``method="apg"`` (default) -- accelerated proximal gradient with adaptive
    restart: Varando & Hansen's Algorithm 1 with Nesterov momentum, the momentum
    step accepted only if it lowers the objective (Beck & Teboulle 2009b,
    MFISTA) and otherwise replaced by a plain step with the momentum reset
    (O'Donoghue & Candes 2015).  Step sizes by backtracking from a
    Barzilai-Borwein trial.  Follows the same stationary points as ``"prox"``
    at a fraction of the iterations.  Stops when the gradient mapping
    ``max|M+ - Y| / t`` falls below ``tol`` (default ``1e-6``).

    ``method="prox"`` -- the plain proximal-gradient method: Algorithm 1 with
    Barzilai-Borwein trial steps (monotone SpaRSA, Wright, Nowak & Figueiredo
    2009) instead of a line search restarted at ``t = 1``.  The reference the
    others are compared with; slow.

    ``method="newton"`` -- **experimental, not used by the simulations.**
    Active-set Newton in the manner of FPC_AS (Wen, Yin, Goldfarb & Zhang
    2010): proximal-gradient steps identify the support, then exact Newton steps
    on that support (:func:`gclm.objective.covariance.hessian`) converge to it;
    on a support change, back to proximal steps.  Stops when
    :func:`gclm.objective.penalties.stationarity` is below ``tol`` (default
    ``1e-8``).  Fast and precise, but it converges to *different* stationary
    points than the first-order methods -- for MCP/SCAD at nearly every
    ``lam``, with lower objectives and worse support recovery.  Kept only so
    that the experiment in docs/LIKELIHOOD.md Section 5 can be reproduced.

    ``newton_after`` (Newton only) -- Newton steps are taken once the gradient
    mapping of the proximal phase is below this value; ``inf`` at once, ``0``
    never.

    For MCP/SCAD the proximal step is capped below ``gamma`` (``gamma - 1``) so
    that the proximal problem is strictly convex and its closed form exact.
    """
    loss = check_loss(loss)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if tol is None:
        tol = 1e-8 if method == "newton" else 1e-6
    penalty = canonical(penalty)
    gamma = resolve_gamma(penalty, gamma)
    p = sigma_hat.shape[0]
    weights = penalty_weights(p) if weights is None else weights
    m0 = diagonal_fit(sigma_hat, c) if m_init is None else m_init
    prob = _Problem(sigma_hat, c, lam, loss, weights, penalty, gamma, m0)
    t_cap = _step_cap(penalty, gamma)
    t = min(1.0, t_cap)
    converged = False

    def bb_step(t, s, grad_old=None):
        """Barzilai-Borwein step from the last accepted move ``s``."""
        y = prob.grad - grad_old
        sy = float(np.sum(s * y))
        t = float(np.sum(s * s)) / sy if sy > 0 else 2.0 * t
        return min(max(t, 1e-12), t_cap)

    if method == "apg":
        x_prev = prob.pt.m
        tk = 1.0
        for _ in range(max_iter):
            tk1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y = _evaluate(prob.pt.m + ((tk - 1.0) / tk1) * (prob.pt.m - x_prev),
                          c, sigma_hat, loss)
            if np.isfinite(y.f):
                grad_y = _grad_at(y, sigma_hat, loss)
            else:                                       # extrapolated point unstable
                y, grad_y = prob.pt, prob.grad
            x_prev = prob.pt.m
            ok, t, s = prob.prox_step(t, y, grad_y)
            if ok:
                tk = tk1
            else:                                       # momentum pointed uphill: restart
                y, grad_y = prob.pt, prob.grad
                ok, t, s = prob.prox_step(t)
                tk = 1.0
                if not ok:
                    break
            if float(np.max(np.abs(s))) / t <= tol:
                converged = True
                break
            t = min(bb_step(t, s, grad_y), 2.0 * t)
    elif method == "prox":
        for _ in range(max_iter):
            grad_old = prob.grad
            ok, t, s = prob.prox_step(t)
            if not ok:
                break
            if float(np.max(np.abs(s))) / t <= tol:
                converged = True
                break
            t = bb_step(t, s, grad_old)
    else:
        # Outer loop: identify (proximal gradient) -> converge (Newton) -> re-check.
        for _ in range(max_iter):
            if prob.kkt() <= tol:
                converged = True
                break
            # identification: proximal steps until the support has settled and
            # the gradient mapping is below newton_after
            same, ok = 0, True
            for _ in range(max_iter):
                support = prob.pt.m != 0
                grad_old = prob.grad
                ok, t, s = prob.prox_step(t)
                if not ok:
                    break
                gmap = float(np.max(np.abs(s))) / t
                t = bb_step(t, s, grad_old)
                same = same + 1 if np.array_equal(prob.pt.m != 0, support) else 0
                if gmap <= tol or (same >= 2 and gmap <= newton_after):
                    break
            if not ok and prob.n_newton + prob.n_prox > 0 and prob.kkt() > tol:
                break                                      # no descent step representable
            # Newton on the current support until it is stationary there
            for _ in range(30):
                accepted, changed = prob.newton_step()
                if not accepted or changed:
                    break
                rows, cols = np.nonzero((prob.pt.m != 0) | ~prob.off)
                g_a = (prob.grad + derivative(prob.pt.m, lam, weights, penalty, gamma))[rows, cols]
                if np.max(np.abs(g_a)) <= 0.1 * tol:
                    break
        else:
            converged = prob.kkt() <= tol
        if not converged:
            converged = prob.kkt() <= tol

    if not return_info:
        return prob.pt.m
    info = FitInfo(iterations=prob.n_prox, newton_steps=prob.n_newton,
                   converged=converged, stationarity=prob.kkt(), objective=prob.big_f)
    return prob.pt.m, info
