"""Data-generating process for S1.

Dettling et al. (2024), Section 5, following Varando & Hansen (2020), Section 4.
See ``simulations/S1_reproduction.md`` Section 3.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from gclm.lyap import solve_lyapunov


class CChoice(str, Enum):
    """Dettling's four volatility settings (labels as in his Figure 5)."""

    ID = "C_ID"
    RANDOM_DIAG = "C_Random_Diag"
    RANDOM_MIN_DIAG = "C_Random_Min_Diag"
    RANDOM_FULL = "C_Random_Full"


def sample_drift(
    p: int,
    d: float,
    rng: np.random.Generator,
    metzler: bool = False,
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
    """
    eps = rng.normal(size=(p, p))
    omega = rng.binomial(1, d, size=(p, p))
    off = omega * (np.abs(eps) if metzler else eps)
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
):
    """One replicate: ``(M_true, C_true, Sigma_true, Sigma_hat)``.

    Edge probability is ``d = k / p``.  ``standardize=True`` reproduces Varando's
    ``simulate.R`` (which feeds the empirical *correlation* matrix); this leaves the
    support of the drift matrix unchanged.
    """
    m_true = sample_drift(p, k / p, rng, metzler=metzler)
    c_true = sample_volatility(p, c_choice, rng)
    sigma_true = solve_lyapunov(m_true, c_true)
    if n is None or not np.isfinite(n):
        sigma_hat = sigma_true
    else:
        sigma_hat = sample_covariance(sample_data(int(n), sigma_true, rng))
    if standardize:
        s = np.sqrt(np.diag(sigma_hat))
        sigma_hat = sigma_hat / np.outer(s, s)
    return m_true, c_true, sigma_true, sigma_hat
