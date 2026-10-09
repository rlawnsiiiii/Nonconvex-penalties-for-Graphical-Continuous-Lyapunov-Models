"""The greedy search of gclm.solvers.search (S3b, docs/SEARCH.md): refits on a
fixed support, the score, the neighbourhood, and the search itself on Example 2,
where the answer is known."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gclm.data.examples import example2_cycle
from gclm.lyapunov import design_matrix, solve_lyapunov
from gclm.solvers.search import (N_INF, CovRefit, DirectRefit, Scorer, bic, bic_along_path, bic_direct,
                                 greedy_search, log_binom, multistart_search, neighbours, random_support)

C5 = 2.0 * np.eye(5)
OFF5 = ~np.eye(5, dtype=bool)


@pytest.fixture(scope="module")
def cycle():
    m = example2_cycle()
    return m, solve_lyapunov(m, C5), (m != 0) & OFF5


@pytest.mark.parametrize("loss", ["direct", "loglik", "frobenius"])
def test_refit_on_the_true_support_recovers_m_star(cycle, loss):
    m, sigma, truth = cycle
    refit = DirectRefit(sigma, C5) if loss == "direct" else CovRefit(sigma, C5, loss)
    assert np.allclose(refit.fit(truth), m, atol=1e-5)


def test_direct_refit_is_least_squares_on_the_support_columns():
    rng = np.random.default_rng(0)
    m = example2_cycle()
    sigma = solve_lyapunov(m, C5) + 0.05 * np.diag(rng.random(5))
    support = OFF5 & (rng.random((5, 5)) < 0.4)
    fitted = DirectRefit(sigma, C5).fit(support)
    assert np.all(fitted[OFF5 & ~support] == 0)
    # the gradient of the direct loss vanishes on the free entries (normal equations)
    a = design_matrix(sigma)
    grad = a.T @ (a @ fitted.flatten(order="F") + C5.flatten(order="F"))
    free = (support | np.eye(5, dtype=bool)).flatten(order="F")
    assert np.max(np.abs(grad[free])) < 1e-10


def test_bic_counts_parameters_and_rejects_unstable(cycle):
    m, sigma, truth = cycle
    k = int(truth.sum())
    base = bic(m, sigma, C5, 1000, k)
    assert bic(m, sigma, C5, 1000, k + 1) - base == pytest.approx(math.log(1000))
    assert bic(m, sigma, C5, 1000, k, ebic_gamma=0.5) - base == pytest.approx(log_binom(20, k))
    # Dettling's form of the eBIC term: 4 gamma k log p
    assert bic(m, sigma, C5, 1000, k, ebic_gamma=0.5, ebic_form="dettling") - base == \
        pytest.approx(2.0 * k * math.log(5))
    with pytest.raises(ValueError):
        bic(m, sigma, C5, 1000, k, ebic_gamma=0.5, ebic_form="chen")
    assert bic(m, sigma, C5, math.inf, k) == pytest.approx(bic(m, sigma, C5, N_INF, k))
    assert bic(np.eye(5), sigma, C5, 1000, 0) == math.inf          # unstable


def test_neighbourhood_moves():
    s = np.zeros((4, 4), bool)
    s[1, 0] = True                     # one entry
    moves = list(neighbours(s))
    kinds = [mv[0] for mv in moves]
    assert kinds.count("delete") == 1 and kinds.count("reverse") == 1
    assert kinds.count("add") == 4 * 3 - 1                        # every other off-diagonal entry
    rev = next(mv[3] for mv in moves if mv[0] == "reverse")
    assert rev[0, 1] and not rev[1, 0] and rev.sum() == 1
    adds = [mv[3] for mv in moves if mv[0] == "add"]
    assert any(a[0, 1] and a[1, 0] for a in adds)                  # add-reverse makes a 2-cycle
    assert not any(a[0, 1] for a in (mv[3] for mv in neighbours(s, allow_two_cycles=False)
                                     if mv[0] == "add"))
    for mv in moves:
        assert not np.any(np.diag(mv[3]))


@pytest.mark.parametrize("loss", ["direct", "loglik", "frobenius"])
def test_one_reversal_fixes_the_lassos_example2_graph(cycle, loss):
    """At n = inf the lasso's best-F1 support is the 5-cycle with 5 -> 1 reversed;
    one reverse move reaches the truth."""
    m, sigma, truth = cycle
    start = truth.copy()
    start[0, 4], start[4, 0] = False, True
    res = greedy_search(sigma, C5, math.inf, start, loss=loss)
    assert np.array_equal(res.support, truth)
    assert res.moves == [("reverse", 4, 0)]


def test_search_scores_never_increase(cycle):
    m, sigma, truth = cycle
    rng = np.random.default_rng(3)
    for _ in range(5):
        res = greedy_search(sigma, C5, 1e4, random_support(5, rng, 0.6))
        assert all(b < a for a, b in zip(res.scores, res.scores[1:]))


def test_multistart_keeps_the_best_and_shares_the_cache(cycle):
    m, sigma, truth = cycle
    rng = np.random.default_rng(1)
    starts = [random_support(5, rng) for _ in range(6)] + [np.zeros((5, 5), bool)]
    best, results = multistart_search(sigma, C5, math.inf, starts)
    assert best.score == min(r.score for r in results)
    assert best.evaluations <= sum(r.evaluations for r in results)


def test_bic_along_path_picks_the_true_support(cycle):
    m, sigma, truth = cycle
    supports = [np.zeros((5, 5), bool), truth, OFF5]
    idx, scores = bic_along_path(sigma, C5, 1e4, supports)
    assert idx == 1 and scores[1] < scores[0]


def test_a_support_whose_refit_cannot_be_computed_scores_inf_and_the_search_goes_on(monkeypatch):
    """At p = 20, n = inf one dense random start made LAPACK's SVD fail inside the
    least-squares fallback (campaign, 5 October).  Such a support must score inf
    rather than end the task."""
    import gclm.solvers.search as S
    m = example2_cycle()
    sigma = solve_lyapunov(m, C5)
    truth = (m != 0) & OFF5
    refit = DirectRefit(sigma, C5)

    def failing_solve(*a, **k):
        raise np.linalg.LinAlgError("Singular matrix")

    def failing_lstsq(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")

    monkeypatch.setattr(S.np.linalg, "solve", failing_solve)
    monkeypatch.setattr(S.np.linalg, "lstsq", failing_lstsq)
    fitted = refit.fit(truth)                               # SciPy's QR driver takes over
    assert np.all(np.isfinite(fitted)) and np.allclose(fitted, m, atol=1e-5)
    import scipy.linalg
    monkeypatch.setattr(scipy.linalg, "lstsq", failing_lstsq)
    fitted = refit.fit(truth)                               # nothing works: unfittable
    assert np.all(np.isnan(fitted[truth | np.eye(5, dtype=bool)]))
    assert bic(fitted, sigma, C5, 1000, int(truth.sum())) == math.inf
    res = greedy_search(sigma, C5, 1000, truth, scorer=S.Scorer(sigma, C5, 1000))
    assert np.isinf(res.score) and res.moves == []        # no finite neighbour: it stops


# --------------------------------------------------------------------------- #
# the direct-loss score (docs/SEARCH.md Section 2)
# --------------------------------------------------------------------------- #


def _noisy(sigma, seed=1):
    e = 0.02 * np.random.default_rng(seed).standard_normal(sigma.shape)
    return sigma + e @ e.T + 0.01 * np.eye(sigma.shape[0])


def test_direct_loss_score_is_n_log_rss_plus_the_bic_penalty(cycle):
    m, sigma, truth = cycle
    sigma_hat, k, n = _noisy(sigma), int(truth.sum()), 1000
    fitted = DirectRefit(sigma_hat, C5).fit(truth)
    r = fitted @ sigma_hat + sigma_hat @ fitted.T + C5
    big_n = 5 * 6 / 2                                   # distinct equations of the Lyapunov system
    expected = big_n * math.log(float(np.sum(r * r)) / big_n) + math.log(n) * (5 + k)
    base = bic_direct(fitted, sigma_hat, C5, n, k)
    assert base == pytest.approx(expected)
    assert bic_direct(fitted, sigma_hat, C5, n, k + 1) - base == pytest.approx(math.log(n))
    assert bic_direct(fitted, sigma_hat, C5, n, k, ebic_gamma=0.5) - base == pytest.approx(log_binom(20, k))
    assert bic_direct(fitted, sigma_hat, C5, n, k, ebic_gamma=0.5, ebic_form="dettling") - base == \
        pytest.approx(2.0 * k * math.log(5))
    assert bic_direct(fitted, sigma_hat, C5, math.inf, k) == pytest.approx(bic_direct(fitted, sigma_hat, C5, N_INF, k))
    assert bic_direct(np.eye(5), sigma_hat, C5, n, 0) == math.inf          # unstable: not a GCLM
    # the scorer refits by least squares and applies the same formula
    assert Scorer(sigma_hat, C5, n, "direct", score="direct").score(truth) == pytest.approx(expected)


def test_direct_loss_score_ties_exact_fits_so_the_penalty_decides(cycle):
    """At the population covariance the true graph and its supergraphs fit exactly: their RSS is
    rounding noise, floored at RSS_FLOOR ||C||^2, so they differ by the penalty alone and the
    search started at the truth stays there."""
    m, sigma, truth = cycle
    sc = Scorer(sigma, C5, math.inf, "direct", score="direct")
    extra = truth.copy()
    i, j = (int(a[0]) for a in np.nonzero(OFF5 & ~truth))
    extra[i, j] = True
    assert sc.score(extra) - sc.score(truth) == pytest.approx(math.log(N_INF))
    res = greedy_search(sigma, C5, math.inf, truth, scorer=sc)
    assert np.array_equal(res.support, truth) and res.moves == []


def test_direct_loss_score_needs_the_least_squares_refit(cycle):
    m, sigma, truth = cycle
    with pytest.raises(ValueError):
        Scorer(sigma, C5, 1000, "loglik", score="direct")
    with pytest.raises(ValueError):
        Scorer(sigma, C5, 1000, "direct", score="rss")
