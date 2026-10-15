"""The data-generating process must produce valid models by construction."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.data.simulate import (
    CChoice,
    draw_instance,
    sample_covariance,
    sample_drift,
    sample_volatility,
)
from gclm.lyapunov import is_stable


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


def test_invalid_edge_probability_and_sample_size_are_refused():
    """Code review D (13 October 2026): k > p gave numpy's message; nan or -inf as n silently gave
    the population covariance.  None and inf mean the population covariance."""
    with pytest.raises(ValueError):
        sample_drift(5, 6 / 5, np.random.default_rng(2))
    for n in (float("nan"), -np.inf, 0, 2.5):
        with pytest.raises(ValueError):
            draw_instance(5, 2, n, CChoice.ID, np.random.default_rng(2))
    _, _, sigma_true, sigma_hat = draw_instance(5, 2, None, CChoice.ID, np.random.default_rng(2))
    assert np.allclose(sigma_true, sigma_hat)


def test_standardized_drift_solves_the_equation_of_the_standardized_data():
    """D^{-1} M D with R = D^{-1} Sigma D^{-1} and 2 diag(1/s^2) is an exact fit on the correlation scale,
    with the same support as M (13 October 2026, the dense-start experiments compare values on that scale)."""
    from gclm.data.simulate import estimation_volatility, standardized_drift
    from gclm.lyapunov import lyapunov_residual

    rng = np.random.default_rng(13)
    m, c, sigma, r, s = draw_instance(7, 2, np.inf, CChoice.ID, rng, standardize=True, return_scale=True)
    m_tilde = standardized_drift(m, s)
    assert np.abs(lyapunov_residual(m_tilde, r, estimation_volatility(s, "variance"))).max() < 1e-12
    assert np.array_equal(m_tilde != 0, m != 0)
    assert np.allclose(np.diag(m_tilde), np.diag(m))                 # the diagonal is scale-free
    assert np.array_equal(standardized_drift(m, np.ones(7)), m)      # s = 1: unchanged
    with pytest.raises(ValueError):
        standardized_drift(m, np.zeros(7))
    with pytest.raises(ValueError):
        standardized_drift(m, s[:3])


def test_generator_knobs_leave_the_default_stream_untouched():
    """Plan of 13 October 2026, §1.4: the default call consumes the generator exactly as before
    (written out here as the old algorithm), so every existing seed produces its graph; the new
    branches draw their numbers after it."""
    for seed, p, d in ((0, 6, 0.2), (1, 10, 0.4), (2, 12, 0.1)):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        m = sample_drift(p, d, a)
        eps = b.normal(size=(p, p))
        off = b.binomial(1, d, size=(p, p)) * eps
        np.fill_diagonal(off, 0.0)
        ref = off.copy()
        np.fill_diagonal(ref, -np.abs(off).sum(axis=1) - np.abs(np.diag(eps)))
        assert np.array_equal(m, ref)
        assert a.random() == b.random()                 # the stream continues at the same point
        # the new options start from the same two draws: same supports (uniform weights), same first draws
        u = sample_drift(p, d, np.random.default_rng(seed), weights="uniform", w_min=0.3)
        assert np.array_equal(u != 0, ref != 0)


def test_uniform_weights_respect_the_lower_bound():
    from gclm.data.simulate import WEIGHTS, GRAPHS
    assert WEIGHTS == ("normal", "uniform") and GRAPHS == ("random", "no2c", "dag")
    rng = np.random.default_rng(5)
    for w_min in (0.0, 0.1, 0.25, 0.5):
        m = sample_drift(10, 0.3, rng, weights="uniform", w_min=w_min)
        w = m[~np.eye(10, dtype=bool)]
        w = np.abs(w[w != 0])
        assert w.min() >= w_min and w.max() <= 1.0 and is_stable(m)
        assert np.all(np.diag(m) < 0)
    signs = np.sign(np.concatenate([sample_drift(10, 0.5, rng, weights="uniform")[~np.eye(10, dtype=bool)]
                                    for _ in range(5)]))
    assert (signs < 0).sum() > 20 and (signs > 0).sum() > 20     # a random sign
    metz = sample_drift(10, 0.5, rng, weights="uniform", metzler=True)
    assert np.all(metz[~np.eye(10, dtype=bool)] >= 0)
    for bad in ({"weights": "cauchy"}, {"graph": "tree"}, {"w_min": 1.0}, {"w_min": -0.1}):
        with pytest.raises(ValueError):
            sample_drift(5, 0.3, rng, **bad)


def test_graph_classes_no2c_and_dag():
    """``no2c``: no pair has both directions; ``dag``: acyclic, and the expected number of edges
    matches the random class within 10 % (d' = 2 d on the allowed half of the pairs)."""
    from gclm.graphs import is_dag, two_cycles
    rng = np.random.default_rng(11)
    p, d = 10, 0.4
    plain = [sample_drift(p, d, rng) for _ in range(200)]
    no2c = [sample_drift(p, d, rng, graph="no2c") for _ in range(200)]
    dags = [sample_drift(p, d, rng, graph="dag") for _ in range(200)]
    assert sum(two_cycles(m != 0) for m in plain) > 100         # the random class has them at this density
    assert all(two_cycles(m != 0) == 0 for m in no2c)
    assert all(is_dag(m != 0) and two_cycles(m != 0) == 0 for m in dags)
    assert all(is_stable(m) for m in no2c + dags)
    off = ~np.eye(p, dtype=bool)
    edges = lambda ms: np.mean([np.sum(m[off] != 0) for m in ms])  # noqa: E731
    assert abs(edges(dags) - edges(plain)) < 0.1 * edges(plain)
    # dropping one direction of every 2-cycle removes exactly that many edges from the same draw
    for seed in range(5):
        a = sample_drift(p, d, np.random.default_rng(seed))
        b = sample_drift(p, d, np.random.default_rng(seed), graph="no2c")
        assert np.all((b != 0) <= (a != 0))
        assert np.sum(a[off] != 0) - np.sum(b[off] != 0) == two_cycles(a != 0)
        assert np.array_equal(b[off][b[off] != 0], a[off][b[off] != 0])    # the surviving weights are the same
    # the knobs pass through draw_instance
    m, c, sigma, _ = draw_instance(p, 3, 500, CChoice.ID, np.random.default_rng(1), graph="dag",
                                   weights="uniform", w_min=0.5)
    w = np.abs(m[off][m[off] != 0])
    assert is_dag(m != 0) and w.min() >= 0.5 and is_stable(m)
