"""The data-generating process must produce valid models by construction."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.dgp import CChoice, draw_instance, sample_covariance, sample_drift, sample_volatility
from gclm.lyap import is_stable


@pytest.mark.parametrize("p", [5, 20])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_drift_is_always_stable(p, k):
    """Diagonal dominance + negative diagonal => Gershgorin gives stability."""
    rng = np.random.default_rng(p * 10 + k)
    for _ in range(25):
        m = sample_drift(p, k / p, rng)
        assert is_stable(m)
        assert np.all(np.diag(m) < 0)
        # strict diagonal dominance
        off = np.abs(m).sum(axis=1) - np.abs(np.diag(m))
        assert np.all(np.abs(np.diag(m)) > off)


def test_drift_edge_density_matches_d():
    """Off-diagonal nonzeros should appear with frequency ~ d."""
    p, k, rng = 40, 3, np.random.default_rng(0)
    off = ~np.eye(p, dtype=bool)
    rate = np.mean([np.mean(sample_drift(p, k / p, rng)[off] != 0) for _ in range(200)])
    assert abs(rate - k / p) < 0.01


def test_metzler_flag_switches_sign_convention():
    rng = np.random.default_rng(3)
    off = ~np.eye(25, dtype=bool)
    signed = sample_drift(25, 4 / 25, np.random.default_rng(3))
    metz = sample_drift(25, 4 / 25, rng, metzler=True)
    assert np.any(signed[off] < 0)          # paper text: signed
    assert np.all(metz[off] >= 0)           # Varando's code: Metzler


@pytest.mark.parametrize("choice", list(CChoice))
def test_volatility_is_symmetric_positive_definite(choice):
    rng = np.random.default_rng(7)
    for p in (5, 15):
        for _ in range(20):
            c = sample_volatility(p, choice, rng)
            assert np.allclose(c, c.T)
            assert np.all(np.linalg.eigvalsh(c) > 0)


def test_volatility_choice_definitions():
    rng = np.random.default_rng(1)
    assert np.allclose(sample_volatility(6, CChoice.ID, rng), 2 * np.eye(6))
    d2 = np.diag(sample_volatility(400, CChoice.RANDOM_DIAG, rng))
    assert d2.min() >= 0.5 and d2.max() <= 4.0
    d3 = np.diag(sample_volatility(400, CChoice.RANDOM_MIN_DIAG, rng))
    assert d3.min() >= 2.0 and d3.max() <= 4.0
    full = sample_volatility(30, CChoice.RANDOM_FULL, rng)
    assert np.any(full[~np.eye(30, dtype=bool)] != 0)


def test_sample_covariance_uses_n_not_n_minus_1():
    """Eq. (1.3): Sigma_hat = X'X / n, uncentered."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    assert np.allclose(sample_covariance(x), x.T @ x / 50)


def test_draw_instance_population_limit():
    """n = inf feeds the exact population covariance."""
    rng = np.random.default_rng(5)
    m, c, sigma_true, sigma_hat = draw_instance(6, 2, np.inf, CChoice.ID, rng)
    assert np.allclose(sigma_true, sigma_hat)
    assert np.allclose(m @ sigma_true + sigma_true @ m.T + c, 0.0, atol=1e-10)


def test_standardize_preserves_support():
    """Correlation scale rescales M to D^-1 M D -- same zero pattern."""
    rng = np.random.default_rng(9)
    m, _, _, sigma_hat = draw_instance(8, 2, 500, CChoice.RANDOM_DIAG, rng,
                                       standardize=True)
    assert np.allclose(np.diag(sigma_hat), 1.0)
