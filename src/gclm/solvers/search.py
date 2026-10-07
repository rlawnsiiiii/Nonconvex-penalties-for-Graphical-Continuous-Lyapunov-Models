"""Score-based greedy search over GCLM supports (S3b, docs/SEARCH.md).

The state is a support ``S``: the off-diagonal entries of ``M`` allowed to be
nonzero.  The diagonal is always free.  A support is scored by refitting ``M``
without penalty on ``S`` plus the diagonal, under one of the three losses,
and evaluating the Gaussian BIC of the implied covariance ``Sigma(M)``:

    BIC(S) = n [log det Sigma(M_S) + tr(Sigma(M_S)^{-1} Sigma_hat)] + log(n) (p + |S|)
             (+ 2 gamma_e log binom(p (p - 1), |S|)  for the extended BIC)

i.e. minus twice the maximised log-likelihood (up to a constant) plus the
dimension penalty, the scale of Amendola, Dettling, Drton, Onori & Wu (2020),
eq. (16)-(17), multiplied by -2n.  A refit that is not stable (an eigenvalue of
``M`` with real part >= 0) has no stationary covariance and scores ``inf``.

The search is best-improvement hill climbing over the neighbourhood "add one
entry / delete one entry / reverse one entry" (as in that paper, Section 5),
started from a given support and stopped when no neighbour lowers the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gclm.lyapunov import design_matrix, solve_lyapunov
from gclm.objective.covariance import check_loss, diagonal_fit, loss_grad, loss_value
from gclm.solvers.covariance import solve as cov_solve

N_INF = 1e6        # nominal sample size standing in for n = inf in the BIC weight
_BIG_LAMBDA = 1e8  # keeps every entry outside the support at zero in the covariance refits


def vec_index(p: int) -> np.ndarray:
    """``idx[i, j]``: position of ``M[i, j]`` in ``vec(M)`` (column-major)."""
    return np.arange(p * p).reshape((p, p), order="F")


def is_stable(m: np.ndarray) -> bool:
    """Every eigenvalue of ``m`` has a negative real part.  A matrix that is not
    finite, or whose eigenvalues LAPACK cannot compute, counts as not stable."""
    if not np.all(np.isfinite(m)):
        return False
    try:
        return bool(np.all(np.linalg.eigvals(m).real < 0.0))
    except np.linalg.LinAlgError:
        return False


# --------------------------------------------------------------------------- #
# refits on a fixed support
# --------------------------------------------------------------------------- #


class DirectRefit:
    """Least squares on the direct loss ``0.5 ||A vec(M) + vec(C)||^2`` restricted
    to the support plus the diagonal, in closed form: the normal equations on the
    precomputed Gram matrix ``A'A`` (Cholesky), with ``numpy.linalg.lstsq`` on the
    columns as the fallback when the restricted Gram matrix is singular."""

    def __init__(self, sigma_hat: np.ndarray, c: np.ndarray):
        self.p = sigma_hat.shape[0]
        self.a = design_matrix(sigma_hat)
        self.rhs = -np.asarray(c, float).flatten(order="F")
        self.gram = self.a.T @ self.a
        self.atb = self.a.T @ self.rhs
        self.idx = vec_index(self.p)

    def fit(self, support: np.ndarray, m_init: np.ndarray | None = None) -> np.ndarray:
        free = support | np.eye(self.p, dtype=bool)
        cols = self.idx[free]
        g = self.gram[np.ix_(cols, cols)]
        try:
            x = np.linalg.solve(g, self.atb[cols])
            if not np.all(np.isfinite(x)) or np.linalg.cond(g) > 1e12:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            x = self._lstsq(cols)
        m = np.zeros(self.p * self.p)
        m[cols] = x
        return m.reshape((self.p, self.p), order="F")

    def _lstsq(self, cols: np.ndarray) -> np.ndarray:
        """Least squares on the design columns themselves, for a singular or badly
        conditioned Gram matrix.  NumPy's driver is SVD-based and can fail to converge
        on badly scaled columns (it did once, at p = 20, n = inf, on a dense random
        start of the pure search: ``LinAlgError: SVD did not converge``).  SciPy's
        QR-based driver then takes over; if that fails too, the support is marked
        unfittable (NaN), which :func:`bic` scores as ``inf``, so the search skips it."""
        try:
            return np.linalg.lstsq(self.a[:, cols], self.rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
        try:
            from scipy.linalg import lstsq
            return lstsq(self.a[:, cols], self.rhs, lapack_driver="gelsy")[0]
        except Exception:                      # noqa: BLE001  -- any LAPACK failure
            return np.full(len(cols), np.nan)


class CovRefit:
    """Unpenalised fit of a covariance loss (``"loglik"`` / ``"frobenius"``) on the
    support plus the diagonal, with the package's APG solver: entries outside the
    support carry weight 1 and a huge lambda (so they stay exactly zero), entries
    on it weight 0 (unpenalised).  Warm-started from ``m_init`` when that is
    stable, otherwise from the diagonal fit."""

    def __init__(self, sigma_hat: np.ndarray, c: np.ndarray, loss: str, tol: float = 1e-9,
                 max_iter: int = 50_000):
        self.sigma_hat, self.c, self.loss, self.tol = sigma_hat, np.asarray(c, float), check_loss(loss), tol
        self.max_iter = max_iter
        self.p = sigma_hat.shape[0]
        self.m_diag = diagonal_fit(sigma_hat, self.c)

    def fit(self, support: np.ndarray, m_init: np.ndarray | None = None) -> np.ndarray:
        eye = np.eye(self.p, dtype=bool)
        weights = np.where(support | eye, 0.0, 1.0)
        start = self.m_diag
        if m_init is not None:
            cand = np.where(support | eye, m_init, 0.0)
            if is_stable(cand) and np.isfinite(loss_value(cand, self.sigma_hat, self.c, self.loss)):
                start = cand
        return cov_solve(self.sigma_hat, self.c, _BIG_LAMBDA, loss=self.loss, weights=weights,
                         m_init=start, penalty="lasso", method="apg", tol=self.tol,
                         max_iter=self.max_iter)


# inside the search the covariance refits only need an accurate *score*: at tol 1e-8 with
# 3000 iterations the scores and selected graphs equal those at 1e-9 / 50 000 (S3b §9.0)
# while being 5-30x faster
SEARCH_TOL, SEARCH_MAX_ITER = 1e-8, 3000


def make_refit(sigma_hat, c, loss: str):
    if loss == "direct":
        return DirectRefit(sigma_hat, c)
    return CovRefit(sigma_hat, c, loss, tol=SEARCH_TOL, max_iter=SEARCH_MAX_ITER)


# --------------------------------------------------------------------------- #
# the score
# --------------------------------------------------------------------------- #


def log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def bic(m: np.ndarray, sigma_hat: np.ndarray, c: np.ndarray, n: float, n_edges: int,
        ebic_gamma: float = 0.0) -> float:
    """``n [log det Sigma(M) + tr(Sigma(M)^{-1} Sigma_hat)] + log(n) (p + n_edges)``,
    plus ``2 gamma_e log binom(p (p - 1), n_edges)``; ``inf`` if ``M`` is not stable.
    ``n = inf`` uses the nominal ``N_INF``."""
    if not is_stable(m):
        return math.inf
    try:
        sigma = solve_lyapunov(m, c)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0 or not np.isfinite(logdet):
            return math.inf
        nll = logdet + float(np.trace(np.linalg.solve(sigma, sigma_hat)))
    except np.linalg.LinAlgError:      # a degenerate refit has no usable likelihood
        return math.inf
    if not np.isfinite(nll):
        return math.inf
    nn = N_INF if math.isinf(n) else float(n)
    p = m.shape[0]
    score = nn * nll + math.log(nn) * (p + n_edges)
    if ebic_gamma:
        score += 2.0 * ebic_gamma * log_binom(p * (p - 1), n_edges)
    return score


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    support: np.ndarray
    m: np.ndarray
    score: float
    moves: list[tuple[str, int, int]] = field(default_factory=list)
    evaluations: int = 0
    scores: list[float] = field(default_factory=list)    # score after each accepted move


class Scorer:
    """Refit + BIC with a cache keyed by the support, so no support is scored twice.

    Only the score is cached.  A search at p = 20 evaluates up to a few hundred
    thousand supports (S3b: 380,000 for one graph at n = inf), and caching the
    refitted matrix with each of them took more than a cluster task's 2 GB.  The
    matrix is recomputed on demand (:meth:`fit`): once per accepted move in the
    search, against hundreds of scores per move.  For the direct loss the refit is
    a closed form, so the recomputed matrix is identical; for the covariance losses
    it is the same iterative solve from the same warm start.
    """

    def __init__(self, sigma_hat, c, n, loss="direct", ebic_gamma=0.0):
        self.sigma_hat, self.c, self.n, self.ebic_gamma = sigma_hat, np.asarray(c, float), n, ebic_gamma
        self.loss = loss
        self.refit = make_refit(sigma_hat, self.c, loss)
        self.cache: dict[bytes, float] = {}
        self.evaluations = 0
        self._last: tuple[bytes, np.ndarray] | None = None     # the refit of the last new support

    def score(self, support: np.ndarray, m_init: np.ndarray | None = None) -> float:
        key = np.packbits(support).tobytes()
        if key not in self.cache:
            self.evaluations += 1
            m = self.refit.fit(support, m_init)
            self.cache[key] = bic(m, self.sigma_hat, self.c, self.n, int(support.sum()),
                                  self.ebic_gamma)
            self._last = (key, m)
        return self.cache[key]

    def fit(self, support: np.ndarray, m_init: np.ndarray | None = None) -> np.ndarray:
        """The unpenalised refit on ``support`` (not cached, see above)."""
        key = np.packbits(support).tobytes()
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        return self.refit.fit(support, m_init)

    def __call__(self, support: np.ndarray, m_init: np.ndarray | None = None):
        """``(score, refit)`` of a support."""
        score = self.score(support, m_init)
        return score, self.fit(support, m_init)


def neighbours(support: np.ndarray, allow_two_cycles: bool = True):
    """Yield ``(move, i, j, new_support)`` for every add / delete / reverse."""
    p = support.shape[0]
    for i, j in zip(*np.nonzero(support)):
        s = support.copy()
        s[i, j] = False
        yield "delete", int(i), int(j), s
        if not support[j, i]:
            s = s.copy()
            s[j, i] = True
            yield "reverse", int(i), int(j), s
    off = ~np.eye(p, dtype=bool)
    for i, j in zip(*np.nonzero(off & ~support)):
        if not allow_two_cycles and support[j, i]:
            continue
        s = support.copy()
        s[i, j] = True
        yield "add", int(i), int(j), s


def _warm_start(m: np.ndarray, move: str, i: int, j: int) -> np.ndarray:
    w = m.copy()
    if move == "delete":
        w[i, j] = 0.0
    elif move == "reverse":
        w[j, i], w[i, j] = w[i, j], 0.0
    return w


def greedy_search(sigma_hat, c, n, start: np.ndarray, loss: str = "direct", ebic_gamma: float = 0.0,
                  max_steps: int = 200, allow_two_cycles: bool = True, add_screen: int | None = None,
                  scorer: Scorer | None = None) -> SearchResult:
    """Best-improvement hill climbing from the support ``start``.

    ``add_screen``: if given, only that many add moves are scored per step -- the
    entries with the largest |gradient| of the loss at the current refit (used for
    the covariance losses, whose refits are iterative).  Deletes and reverses are
    always all scored.  Pass a shared ``scorer`` to reuse its cache across starts.
    """
    p = sigma_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    sc = scorer or Scorer(sigma_hat, c, n, loss, ebic_gamma)
    cur = np.asarray(start, bool) & off
    if not allow_two_cycles:
        cur = cur & ~(cur & cur.T & np.triu(np.ones((p, p), bool), 1))
    cur_score, cur_m = sc(cur)
    res = SearchResult(cur.copy(), cur_m, cur_score)
    for _ in range(max_steps):
        allowed_adds = None
        if add_screen is not None and loss != "direct" and np.isfinite(cur_score):
            g = np.abs(loss_grad(cur_m, sigma_hat, np.asarray(c, float), loss))
            g[~off | cur] = -1.0
            top = np.argsort(g, axis=None)[::-1][:add_screen]
            allowed_adds = {(int(a), int(b)) for a, b in zip(*np.unravel_index(top, g.shape))}
        best = None
        for move, i, j, s in neighbours(cur, allow_two_cycles):
            if move == "add" and allowed_adds is not None and (i, j) not in allowed_adds:
                continue
            warm = _warm_start(cur_m, move, i, j) if np.isfinite(cur_score) else None
            score = sc.score(s, warm)
            if best is None or score < best[0]:
                best = (score, s, (move, i, j), warm)
        if best is None or not best[0] < cur_score - 1e-9 * max(1.0, abs(cur_score) if np.isfinite(cur_score) else 1.0):
            break
        cur_score, cur = best[0], best[1]
        cur_m = sc.fit(cur, best[3])                 # one refit per accepted move
        res.moves.append(best[2])
        res.scores.append(cur_score)
    res.support, res.m, res.score, res.evaluations = cur, cur_m, cur_score, sc.evaluations
    return res


def random_support(p: int, rng: np.random.Generator, max_density: float = 0.3,
                   allow_two_cycles: bool = True) -> np.ndarray:
    """A random starting graph: edge probability drawn uniformly from
    ``[0, max_density]``, then every off-diagonal entry independently."""
    d = rng.uniform(0.0, max_density)
    s = rng.random((p, p)) < d
    np.fill_diagonal(s, False)
    if not allow_two_cycles:
        both = s & s.T & np.triu(np.ones((p, p), bool), 1)
        s &= ~both
    return s


def multistart_search(sigma_hat, c, n, starts: list[np.ndarray], **kw) -> tuple[SearchResult, list[SearchResult]]:
    """Run :func:`greedy_search` from every start with one shared cache; return
    the best-scoring result and all of them."""
    loss, ebic_gamma = kw.pop("loss", "direct"), kw.pop("ebic_gamma", 0.0)
    sc = Scorer(sigma_hat, c, n, loss, ebic_gamma)
    results = [greedy_search(sigma_hat, c, n, s, loss=loss, ebic_gamma=ebic_gamma, scorer=sc, **kw)
               for s in starts]
    best = min(results, key=lambda r: r.score)
    best.evaluations = sc.evaluations
    return best, results


def bic_along_path(sigma_hat, c, n, supports: list[np.ndarray], loss: str = "direct",
                   ebic_gamma: float = 0.0, scorer: Scorer | None = None) -> tuple[int, list[float]]:
    """Refit and score the support of every estimate of a path; return the index
    of the lowest score (the BIC-selected lambda) and all scores."""
    sc = scorer or Scorer(sigma_hat, c, n, loss, ebic_gamma)
    p = sigma_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    scores = [sc.score(np.asarray(s, bool) & off) for s in supports]
    return int(np.argmin(scores)), scores
