"""Data-generating process for S1.

Dettling et al. (2024), Section 5, following Varando & Hansen (2020), Section 4.
See ``simulations/S1_reproduction.md`` Section 3.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from gclm.lyapunov import solve_lyapunov


class CChoice(str, Enum):
    """Dettling's four volatility settings (labels as in his Figure 5)."""

    ID = "C_ID"
    RANDOM_DIAG = "C_Random_Diag"
    RANDOM_MIN_DIAG = "C_Random_Min_Diag"
    RANDOM_FULL = "C_Random_Full"


#: the law of the nonzero weights (plan of 13 October 2026, §1.4): Dettling's N(0, 1), or
#: |w| ~ U[w_min, 1] with a random sign (the beta-min knob of experiment E3)
WEIGHTS = ("normal", "uniform")
#: the graph class (plan §1.4): every ordered pair independently (Dettling), the same without
#: 2-cycles (one direction of each dropped at random), or a random DAG at the same expected
#: number of edges (experiment E4)
GRAPHS = ("random", "no2c", "dag")


def sample_drift(
    p: int,
    d: float,
    rng: np.random.Generator,
    metzler: bool = False,
    *,
    weights: str = "normal",
    w_min: float = 0.0,
    graph: str = "random",
) -> np.ndarray:
    """Draw a stable drift matrix.

    ``M_ij = omega_ij * eps_ij`` for ``i != j`` with ``omega_ij ~ Bernoulli(d)`` and
    ``eps_ij ~ N(0, 1)``; ``M_ii = -sum_{j != i} |M_ij| - |eps_ii|``.

    The diagonal rule makes ``M`` strictly diagonally dominant with a negative
    diagonal, so by Gershgorin every draw is stable -- no rejection sampling.

    Parameters
    ----------
    metzler:
        If True, use ``|eps_ij|`` off the diagonal, reproducing Varando's released
        code (``rStableMetzler``) rather than the papers' text.  Default False
        follows the text.  See S1_reproduction.md Section 3.1.
    weights, w_min, graph:
        The knobs of the mechanism experiments of 13 October 2026 (plan §1.4), off by
        default.  **The default call consumes the generator exactly as before**; the new
        branches draw their numbers after the two default draws, so every existing seed
        still produces its graph.

        ``weights="uniform"``: ``|M_ij| ~ U[w_min, 1]`` with an independent random sign
        (no sign under ``metzler``), a lower bound on every true weight (E3).  The supports
        (the Bernoulli draws) and the diagonal rule are unchanged, with the ``|eps_ii|`` of the
        normal draw.  ``graph="no2c"``: the Bernoulli(d) draw, then one direction of every
        2-cycle is dropped at random.  ``graph="dag"``: a random permutation ``pi`` orders the
        nodes and the edge ``j -> i`` (entry ``(i, j)``) is kept with probability
        ``d' = min(2 d, 1)`` if ``pi(j) < pi(i)`` and never otherwise, so the expected number of
        edges is the random class's ``p (p - 1) d`` for ``d <= 1/2`` (``k <= p / 2``) and
        ``p (p - 1) / 2`` above.  Every DAG is also free of 2-cycles.
    """
    if not 0.0 <= d <= 1.0:
        raise ValueError(f"the edge probability d = k / p must lie in [0, 1], got {d} (k > p?)")
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
    if graph not in GRAPHS:
        raise ValueError(f"graph must be one of {GRAPHS}, got {graph!r}")
    if not 0.0 <= w_min < 1.0:
        raise ValueError(f"w_min must lie in [0, 1), got {w_min}")
    eps = rng.normal(size=(p, p))
    omega = rng.binomial(1, d, size=(p, p))
    if graph == "no2c":
        i, j = np.nonzero(np.triu((omega == 1) & (omega.T == 1), 1))
        drop_upper = rng.random(size=len(i)) < 0.5
        omega[i[drop_upper], j[drop_upper]] = 0
        omega[j[~drop_upper], i[~drop_upper]] = 0
    elif graph == "dag":
        position = rng.permutation(p)                               # position[node] = pi(node)
        allowed = position[None, :] < position[:, None]             # entry (i, j): pi(j) < pi(i)
        omega = rng.binomial(1, min(2.0 * d, 1.0), size=(p, p)) * allowed
    if weights == "uniform":
        magnitude = rng.uniform(w_min, 1.0, size=(p, p))
        sign = np.where(rng.random(size=(p, p)) < 0.5, -1.0, 1.0)
        values = magnitude if metzler else sign * magnitude
    else:
        values = np.abs(eps) if metzler else eps
    off = omega * values
    np.fill_diagonal(off, 0.0)
    m = off.copy()
    np.fill_diagonal(m, -np.abs(off).sum(axis=1) - np.abs(np.diag(eps)))
    return m


def sample_volatility(p: int, choice: CChoice, rng: np.random.Generator) -> np.ndarray:
    """Draw ``C`` under one of Dettling's four choices (S1_reproduction.md Sec. 3.2)."""
    choice = CChoice(choice)
    if choice is CChoice.ID:
        return 2.0 * np.eye(p)
    if choice is CChoice.RANDOM_DIAG:
        return np.diag(rng.uniform(0.5, 4.0, size=p))
    if choice is CChoice.RANDOM_MIN_DIAG:
        return np.diag(rng.uniform(2.0, 4.0, size=p))

    # RANDOM_FULL: symmetric, diagonally dominant (hence positive definite).
    eps = rng.normal(size=(p, p))
    omega = rng.binomial(1, min(2.0 / p, 1.0), size=(p, p))
    w = omega * eps
    np.fill_diagonal(w, 0.0)
    c = w + w.T
    np.fill_diagonal(c, np.abs(c).sum(axis=1) + np.abs(np.diag(eps)) + 0.5)
    return c


def sample_data(
    n: int,
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` iid draws from ``N(0, sigma)``, shape ``(n, p)``."""
    p = sigma.shape[0]
    return rng.multivariate_normal(np.zeros(p), sigma, size=n, method="cholesky")


def sample_covariance(x: np.ndarray) -> np.ndarray:
    """``Sigma_hat = X' X / n`` -- eq. (1.3): divisor ``n``, no centering."""
    return x.T @ x / x.shape[0]


def draw_instance(
    p: int,
    k: int,
    n: int,
    c_choice: CChoice,
    rng: np.random.Generator,
    metzler: bool = False,
    standardize: bool = False,
    return_scale: bool = False,
    weights: str = "normal",
    w_min: float = 0.0,
    graph: str = "random",
):
    """One replicate: ``(M_true, C_true, Sigma_true, Sigma_hat)``.

    ``n`` is the sample size; ``inf`` (or ``None``) feeds the population covariance
    instead of a sample.  Anything else that is not a positive integer is refused
    (``nan`` and ``-inf`` used to give the population covariance silently; code
    review D, 13 October 2026).

    Edge probability is ``d = k / p``.  ``standardize=True`` reproduces Varando's
    ``simulate.R`` (which feeds the empirical *correlation* matrix); this leaves the
    support of the drift matrix unchanged.

    ``return_scale=True`` appends the standard deviations ``s`` the data were divided
    by (ones if ``standardize=False``).  They are what :func:`estimation_volatility`
    needs: standardizing the data changes the volatility matrix of the model, see there.
    ``weights``, ``w_min`` and ``graph`` are the generator's knobs of 13 October 2026
    (:func:`sample_drift`); the defaults are the thesis generator.
    """
    m_true = sample_drift(p, k / p, rng, metzler=metzler, weights=weights, w_min=w_min, graph=graph)
    c_true = sample_volatility(p, c_choice, rng)
    sigma_true = solve_lyapunov(m_true, c_true)
    if n is None or n == np.inf:
        sigma_hat = sigma_true                          # the population covariance
    elif not (np.isfinite(n) and n >= 1 and n == int(n)):
        raise ValueError(f"n must be a positive integer or inf (the population covariance), got {n!r}")
    else:
        sigma_hat = sample_covariance(sample_data(int(n), sigma_true, rng))
    s = np.ones(p)
    if standardize:
        s = np.sqrt(np.diag(sigma_hat))
        sigma_hat = sigma_hat / np.outer(s, s)
    if return_scale:
        return m_true, c_true, sigma_true, sigma_hat, s
    return m_true, c_true, sigma_true, sigma_hat


C_SCALES = ("identity", "variance")


def estimation_volatility(scale: np.ndarray, c_scale: str = "identity") -> np.ndarray:
    """The volatility matrix ``C`` handed to the estimator.

    Standardizing the data is a change of variables ``X -> D^{-1} X`` with
    ``D = diag(s)``.  It maps the Lyapunov equation ``M Sigma + Sigma M' + C = 0`` to

        (D^{-1} M D) R + R (D^{-1} M D)' + D^{-1} C D^{-1} = 0,     R = D^{-1} Sigma D^{-1},

    so the drift matrix keeps its support, but the volatility matrix of the standardized
    data is ``D^{-1} C D^{-1}``, not ``C``.

    ``"identity"``  ``2 I`` whatever the scale of the input -- the setting of every run so
                    far, and the one that reproduces Dettling's Figure 5.  On standardized
                    data it fits a model the data do not follow, even when the data were
                    generated with ``C = 2 I``.
    ``"variance"``  ``2 diag(1 / s_i^2)``: the model "C = 2 I on the measurement scale",
                    written on the correlation scale.  Identical to ``"identity"`` when the
                    data are not standardized (``s = 1``).  This is the "rescaled C" of
                    next_steps/051026/cluster_campaign_051026.md (Section 2.2): exact when
                    the data were generated with ``C = 2 I`` (Dettling's ``C_ID``), only
                    partly right for his three random choices of ``C``, and a device for
                    simulations, not a recipe for real data.
    """
    if c_scale not in C_SCALES:
        raise ValueError(f"unknown c_scale {c_scale!r}; expected one of {C_SCALES}")
    scale = np.asarray(scale, dtype=float)
    if c_scale == "identity":
        return 2.0 * np.eye(scale.shape[0])
    return np.diag(2.0 / scale ** 2)


def standardized_drift(m: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The drift matrix of the standardized data, ``D^{-1} M D`` with ``D = diag(scale)``.

    The counterpart of :func:`estimation_volatility`: with ``R = D^{-1} Sigma D^{-1}`` and
    ``D^{-1} C D^{-1}`` it solves the Lyapunov equation of the standardized data exactly, so
    it is the truth an estimator working on the correlation scale is compared with *by value*
    (the support is the same, ``M_ij s_j / s_i`` is nonzero iff ``M_ij`` is).  Used by the
    dense-start experiments of 13 October 2026 (the noisy-truth start, the BP condition).
    """
    scale = np.asarray(scale, dtype=float)
    if scale.ndim != 1 or scale.shape[0] != m.shape[0] or np.any(scale <= 0):
        raise ValueError("scale must be the vector of positive standard deviations the data were divided by")
    return np.asarray(m, dtype=float) * np.outer(1.0 / scale, scale)
