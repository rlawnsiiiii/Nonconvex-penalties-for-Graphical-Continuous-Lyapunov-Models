"""Penalties for the Direct Lyapunov estimator: l1 (lasso), MCP and SCAD.

Every backend minimises the paper-scale objective (docs/NONCONVEX.md)

    F(M) = 0.5 * ||M Sigma + Sigma M' + C||_F^2  +  sum_ij  w_ij * P_ij(M_ij)

with ``w_ij`` the penalty weights (1 off the diagonal, 0 on it).  The loss
carries the factor 0.5 and no 1/n: that is Dettling's eq. (1.4).

The scalar penalties
--------------------
    lasso   P(x) = lam |x|

    MCP     P(x) = lam |x| - x^2 / (2 gamma)                  |x| <= gamma lam
                 = gamma lam^2 / 2                             otherwise
            (Zhang 2010)

    SCAD    P(x) = lam |x|                                     |x| <= lam
                 = (2 gamma lam |x| - x^2 - lam^2) / (2 (gamma - 1))
                                                               lam < |x| <= gamma lam
                 = lam^2 (gamma + 1) / 2                        otherwise
            (Fan & Li 2001, their ``a`` written as gamma)

Two conventions for how they are applied (``scale``)
---------------------------------------------------
``scale=None`` -- the **textbook** convention, ``P_ij(x) = P(x)``.  The same
    penalty on every entry, as in Fan & Li, Zhang and Loh & Wainwright.  Solved
    by the ``fista`` and ``skglm`` backends.

``scale=v`` -- the **ncvreg** convention, ``P_ij(x) = P(v_ij x) / v_ij`` with
    ``v_ij`` the loss curvature along coordinate ``ij`` (the squared column norm
    of the design).  This is what ``ncvreg::ncvfit`` minimises on a design whose
    columns are not standardised -- established to 1e-16 in
    tests/test_nonconvex.py -- and it makes gamma scale-free: every coordinate
    gets the same convexity margin.  It coincides with the textbook convention
    when all ``v_ij = 1``, and for the lasso it coincides always, because
    ``lam |v x| / v = lam |x|``.

All functions are vectorised over the entries of ``M``; ``scale`` broadcasts.

Also here: :func:`penalty_weights` (which entries are penalised -- the diagonal
is not, matching Varando's ``penalty.factor = 1 - diag(p)``) and
:func:`penalty_scale`, the per-entry scale of the ncvreg convention.
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import design_column_sq_norms

#: package defaults (ncvreg, skglm, pyproximal); Fan & Li recommend 3.7 for SCAD
DEFAULT_GAMMA = {"MCP": 3.0, "SCAD": 3.7}

_ALIASES = {"lasso": "lasso", "l1": "lasso", "mcp": "MCP", "scad": "SCAD"}


def canonical(penalty: str | None) -> str:
    """Normalise a penalty name to ``"lasso"``, ``"MCP"`` or ``"SCAD"``."""
    key = "lasso" if penalty is None else str(penalty).lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown penalty {penalty!r}; expected lasso, MCP or SCAD")
    return _ALIASES[key]


def resolve_gamma(penalty: str, gamma: float | None) -> float | None:
    """The concavity parameter, with package defaults filled in.

    Enforces the same domain as ncvreg: MCP needs gamma > 1, SCAD gamma > 2.
    """
    penalty = canonical(penalty)
    if penalty == "lasso":
        return None
    g = DEFAULT_GAMMA[penalty] if gamma is None else float(gamma)
    lower = 1.0 if penalty == "MCP" else 2.0
    if not g > lower:
        raise ValueError(f"{penalty} requires gamma > {lower:g}, got {g}")
    return g


def _check_binary_weights(weights: np.ndarray, penalty: str) -> None:
    """MCP/SCAD are supported with 0/1 weights only.

    For binary weights every convention in use coincides -- ncvreg scales lam
    by the weight, skglm's WeightedMCPenalty scales the whole penalty -- so the
    backends provably solve the same problem.  For other weights they differ.
    """
    if penalty != "lasso" and not np.all((weights == 0) | (weights == 1)):
        raise ValueError(f"{penalty} supports penalty weights in {{0, 1}} only")


def _scalar_value(a, lam, penalty, g):
    """P(a) for a >= 0."""
    if penalty == "lasso":
        return lam * a
    if penalty == "MCP":
        return np.where(a <= g * lam, lam * a - a * a / (2.0 * g), g * lam * lam / 2.0)
    return np.where(
        a <= lam, lam * a,
        np.where(a <= g * lam,
                 (2.0 * g * lam * a - a * a - lam * lam) / (2.0 * (g - 1.0)),
                 lam * lam * (g + 1.0) / 2.0))


def _scalar_slope(a, lam, penalty, g):
    """P'(a) for a > 0."""
    if penalty == "lasso":
        return np.full_like(a, lam, dtype=float)
    if penalty == "MCP":
        return np.maximum(lam - a / g, 0.0)
    return np.where(a <= lam, lam,
                    np.where(a <= g * lam, (g * lam - a) / (g - 1.0), 0.0))


def _prep(m, weights, scale, penalty):
    m = np.asarray(m, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), m.shape)
    v = np.ones_like(m) if scale is None else np.broadcast_to(
        np.asarray(scale, dtype=float), m.shape)
    if np.any(v <= 0):
        raise ValueError("scale must be positive")
    _check_binary_weights(w, penalty)
    return m, w, v


def value(m, lam, weights, penalty="lasso", gamma=None, scale=None) -> float:
    """``sum_ij w_ij P_ij(M_ij)`` -- see the module docstring for ``scale``."""
    penalty = canonical(penalty)
    g = resolve_gamma(penalty, gamma)
    m, w, v = _prep(m, weights, scale, penalty)
    return float(np.sum(w * _scalar_value(v * np.abs(m), lam, penalty, g) / v))


def derivative(m, lam, weights, penalty="lasso", gamma=None, scale=None) -> np.ndarray:
    """``w_ij * d/dx P_ij(x)`` at the nonzero entries, 0 where ``M_ij == 0``.

    ``d/dx [P(v x)/v] = P'(v x)``.  At zero every penalty here has the
    subdifferential ``[-lam w, lam w]`` for any scale; :func:`stationarity`
    handles that case.
    """
    penalty = canonical(penalty)
    g = resolve_gamma(penalty, gamma)
    m, w, v = _prep(m, weights, scale, penalty)
    slope = _scalar_slope(v * np.abs(m), lam, penalty, g)
    return np.where(m != 0, w * np.sign(m) * slope, 0.0)


def prox(z, t, lam, weights, penalty="lasso", gamma=None, scale=None) -> np.ndarray:
    """Elementwise ``argmin_x 0.5 (x - z)^2 + t * w P_ij(x)``.

    Textbook scale (docs/NONCONVEX.md Section 3):

    lasso  soft thresholding at ``t lam``.
    MCP    firm thresholding: 0 for |z| <= t lam; (|z| - t lam) / (1 - t/gamma)
           up to gamma lam; z unchanged beyond.  Needs gamma > t.
    SCAD   soft thresholding up to (1 + t) lam; ((gamma - 1) z - sign(z) t gamma lam)
           / (gamma - 1 - t) up to gamma lam; z unchanged beyond.
           Needs gamma > 1 + t.

    With a scale ``v`` the substitution ``u = v x`` reduces the problem to the
    textbook one with step ``t v`` at ``v z``:  ``x = prox_{t v}(v z) / v``.
    Under the stated conditions (with ``t v`` in place of ``t``) the scalar
    problem is strictly convex, so the closed form is its unique global
    minimiser.  Entries with ``w = 0`` are returned unchanged.
    """
    penalty = canonical(penalty)
    g = resolve_gamma(penalty, gamma)
    z, w, v = _prep(z, weights, scale, penalty)

    if penalty == "lasso":
        return np.sign(z) * np.maximum(np.abs(z) - t * lam * w, 0.0)

    ts = t * v                                  # effective step per entry
    pen = w == 1
    if np.any(pen):
        tmax = float(np.max(ts[pen]))
        if penalty == "MCP" and not g > tmax:
            raise ValueError(f"MCP prox needs gamma > step ({g} <= {tmax})")
        if penalty == "SCAD" and not g > 1.0 + tmax:
            raise ValueError(f"SCAD prox needs gamma > 1 + step ({g} <= {1.0 + tmax})")

    u = v * z                                   # work in u = v x
    a = np.abs(u)
    s = np.sign(u)
    out = u.copy()
    if penalty == "MCP":
        zero = pen & (a <= ts * lam)
        mid = pen & (a > ts * lam) & (a <= g * lam)
        out[zero] = 0.0
        out[mid] = s[mid] * (a[mid] - ts[mid] * lam) / (1.0 - ts[mid] / g)
    else:
        soft = pen & (a <= (1.0 + ts) * lam)
        mid = pen & (a > (1.0 + ts) * lam) & (a <= g * lam)
        out[soft] = s[soft] * np.maximum(a[soft] - ts[soft] * lam, 0.0)
        out[mid] = ((g - 1.0) * u[mid] - s[mid] * ts[mid] * g * lam) / (g - 1.0 - ts[mid])
    out = np.where(pen, out / v, z)             # back to x; unpenalised entries untouched
    return out


def stationarity(m, grad, lam, weights, penalty="lasso", gamma=None, scale=None) -> float:
    """Largest violation of the first-order condition ``0 in grad + dP(M)``.

    Nonzero entry: ``grad + w P_ij'(M) = 0``.  Zero entry: ``|grad| <= lam w``
    (the Clarke subdifferential of all three penalties at 0, for any scale).
    Zero at any local minimiser -- and, for the lasso, only at the global one.
    """
    m = np.asarray(m, dtype=float)
    grad = np.asarray(grad, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), m.shape)
    viol = np.where(m != 0,
                    np.abs(grad + derivative(m, lam, w, penalty, gamma, scale)),
                    np.maximum(np.abs(grad) - lam * w, 0.0))
    return float(np.max(viol)) if viol.size else 0.0


def penalty_weights(p: int, penalize_diagonal: bool = False) -> np.ndarray:
    """Penalty weight matrix; diagonal unpenalized by default."""
    w = np.ones((p, p))
    if not penalize_diagonal:
        np.fill_diagonal(w, 0.0)
    return w

CONVENTIONS = ("textbook", "ncvreg")

def penalty_scale(sigma: np.ndarray, convention: str = "textbook") -> np.ndarray | None:
    """Per-entry scale for the MCP/SCAD penalty under a given convention.

    ``"textbook"``: ``None`` -- ``P(M_ij)`` on every entry, as in Fan & Li,
    Zhang and Loh & Wainwright.

    ``"ncvreg"``: the loss curvature ``v_ij = ||A(Sigma) e_ij||^2`` along each
    coordinate, giving ``P(v_ij M_ij) / v_ij``.  That is exactly the objective
    ``ncvreg::ncvfit`` minimises on this (unstandardised) design.  The lasso is
    the same under both conventions.  See docs/NONCONVEX.md Section 2.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; expected one of {CONVENTIONS}")
    return None if convention == "textbook" else design_column_sq_norms(sigma)
