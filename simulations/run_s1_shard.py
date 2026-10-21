#!/usr/bin/env python3
"""S1 / Figure 5 -- cluster shard runner.

Writes **numbers only**, no plots.  One compressed ``.npz`` per shard containing
everything needed to recompute any downstream quantity without re-running:

  per dataset      p, k, c_choice, rep, seed, lambda_max, timing, n_true_edges
  per (dataset, l) lambda, tp/fp/tn/fn off-diagonal AND including the diagonal,
                   nnz, penalised objective, the first-order violation (kkt), the
                   solver's step count (iterations; newton_steps for the covariance
                   losses) and whether the solver met its tolerance (converged) --
                   for the direct loss as well since 13 October 2026 (code review A4;
                   older shards hold NaN / -1 there and no ``converged``)
  per dataset      M* as a sparse triple, and M_hat at the best-F1 and best-acc
                   lambdas (also sparse)
  once             the full S1Config, so the run is self-describing, and the
                   provenance (host, versions, git commit, timing)

Storing the raw confusion counts (rather than only max_acc/max_f1/auc/aupr)
means every metric -- including ones not yet defined, and the
``include_diagonal`` variant -- can be recomputed locally from the output.

With ``--select bic`` or ``--select search`` (the campaign of October 2026,
next_steps/051026/cluster_campaign_051026.md) each dataset additionally gets

  supports_packed  the off-diagonal support at every lambda (np.packbits; decode
                   with :func:`unpack_supports`), and ``scale``, the standard
                   deviations the data were divided by
  bic_*            the likelihood score (BIC penalty) of every support of the
                   path after an unpenalised refit -- least squares on the direct
                   loss by default, the maximised likelihood with ``--refit loglik``
                   (gclm.solvers.search) -- the index of the best one, its
                   confusion counts and its orientation breakdown (order: ORIENT);
                   ``bic_dead`` flags a path on which every score is +inf (no stable
                   refit: the index is then argmin's 0 and not a selection, code
                   review A3), ``bic_unconverged`` counts refits that stopped at the
                   solver's iteration cap (likelihood refit; code review A2)
  search_*         (``search`` only) the graph reached from the selected support
                   by the greedy search on the score with add / delete / reverse
                   moves: its support (``m_search_support``, packed like one
                   lambda of ``supports_packed``), its unpenalised refit
                   ``m_search_i/j/v`` (sparse), counts, score, moves; and
                   ``search_dead``, ``search_unconverged``, ``search_capped`` (stopped
                   by the step cap), ``search_mismatches`` (gclm.solvers.search)

Without ``--select`` the output is what it was before these options existed, plus
``converged``.  Shards are written to a temporary name and renamed into place, so a
task killed while saving leaves no truncated file (code review C4).

Dense starts (plan of 13 October 2026, E1 and E2; 14 October).  ``--up-start`` chooses where a
dense -> sparse path starts, on both losses: ``lasso`` (the lasso solution at the smallest
lambda; the direct loss's default), ``exact`` (``-C Sigma_hat^-1 / 2``, Varando & Hansen; the
covariance losses' default), ``l1`` / ``l2`` (the minimum-l1 / minimum-Frobenius exact fits,
gclm.objective.exact_fits) or ``truth-<tau>`` (the standardised truth plus ``tau N(0, 1)`` off the
diagonal, drawn from the data set's generator after the data; a ceiling, not an estimator).
``--pilot`` takes the same choices for the adaptive lasso and for the thresholded lasso
(``--method threshold`` refits every kept support by least squares, ``threshold-norefit`` keeps the
pilot's values; gclm.solvers.path.thresholded_lasso_path).  Every cell with a dense start stores
the start's own diagnostics per data set:

  start_l1, start_l2   its off-diagonal l1 and Frobenius norms
  start_residual       ``||M Sigma_hat + Sigma_hat M' + C||_F`` (0 for an exact fit)
  start_nnz            its off-diagonal entries with ``|M_ij| > 1e-8`` ("present")
  start_conf           tp, fp, tn, fn of those entries against the truth (the start's own F1)
  start_orient         their orientation breakdown (order: ORIENT): among the true edges outside
                       2-cycles, ``correct`` = the true direction present alone, ``hedged`` = both
                       directions present, ``reversed`` = only the reverse, ``missed_single`` =
                       neither -- the "dense-start content" of the theory note of 7 October 2026

The generator's knobs ``--weights uniform --w-min W`` (E3), ``--graph dag`` (E4) and ``--graph pairs --two-cycle-share S`` (E5) are
recorded in the config; the defaults are the thesis generator bit for bit.

    python simulations/run_s1_shard.py --shard 0 --n-shards 64 \
        --out-dir runs/s1_dettling_reproduction/s1_shards
    python simulations/run_s1_shard.py --shard 0 --n-shards 64 --loss loglik \
        --penalty MCP --out-dir runs/s2_loglik_mcp/s1_shards
    python simulations/run_s1_shard.py --shard 0 --n-shards 64 --n-obs inf \
        --p 10 20 --reps 25 --out-dir runs/nsweep_p10-20/direct_lasso_ninf/s1_shards
    python simulations/run_s1_shard.py --shard 0 --n-shards 16 --p 10 20 --reps 25 \
        --penalty MCP --direction up --c-scale variance --select search \
        --out-dir runs/campaign/direct_MCP-up_Cresc_n1000/s1_shards
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gclm import __name__ as _pkg  # noqa: F401  (ensures src/ is importable)
from gclm.objective import covariance
from gclm.config import S1Config, parse_n_obs
from gclm.data.simulate import (C_SCALES, GRAPHS, WEIGHTS, CChoice, draw_instance, estimation_volatility,
                                standardized_drift)
from gclm.objective.direct import direct_grad, direct_loss, lambda_max
from gclm.objective.exact_fits import (PRESENT, exact_fit_residual, min_l1_exact_fit, min_l2_exact_fit,
                                       noisy_truth_start)
from gclm.objective.penalties import penalty_scale, penalty_weights, stationarity
from gclm.solvers.path import (adaptive_covloss_path, adaptive_lasso_path, covloss_path, lambda_grid, lasso_path,
                               lla_path, thresholded_lasso_path)
from gclm.objective.direct import objective
from gclm.metrics import confusion, orientation_breakdown
from gclm.objective.penalties import resolve_gamma
from gclm.solvers.search import Scorer, bic_along_path, greedy_search

METHODS = ("path", "lla", "adaptive", "threshold", "threshold-norefit")
#: the methods that start from a dense pilot (--pilot) rather than from the empty graph
PILOT_METHODS = ("adaptive", "threshold", "threshold-norefit")
SELECT = ("none", "bic", "search")
#: where a dense -> sparse path, or the pilot of the adaptive / thresholded lasso, comes from (plan
#: of 13 October 2026, E1): the lasso's dense end, Varando & Hansen's exact fit, the minimum-l1 /
#: minimum-l2 exact fits, or ``truth-<tau>``: the standardised truth plus tau N(0, 1) off the diagonal
STARTS = ("lasso", "exact", "l1", "l2")
TRUTH_START = "truth-"
#: an entry of a dense start counts as present above this size (the exact fits have no exact zeros);
#: the thresholded lasso's rule for its pilot, so that start_nnz is its first support
START_PRESENT = PRESENT
#: how a support is refitted for the score: least squares on the direct loss (closed form, the
#: campaign's default) or the maximised Gaussian likelihood (iterative; 200 to 1000 times slower)
REFITS = ("direct", "loglik")
#: the form of the eBIC term used by --ebic-gamma: Dettling's 4 gamma |E| log p (his
#: eq. 6.2), the same rule campaign.py applies offline to the path (columns ebic05_*, ebic1_*)
EBIC_FORM = "dettling"


def ebic_tag(gamma: float) -> str:
    """Field prefix of one gamma of the eBIC penalty: 0.5 -> ``ebic05``, 1 -> ``ebic1``,
    1.5 -> ``ebic15``.  The prefix drops the decimal point, so gamma is restricted to
    ``0 < gamma < 10`` with at most one decimal, which keeps it unambiguous (code review C5,
    13 October 2026: 1.5 and 15 used to give the same prefix)."""
    if not 0 < gamma < 10 or round(gamma, 1) != gamma:
        raise ValueError(f"--ebic-gamma takes values in (0, 10) with at most one decimal, got {gamma}")
    return f"ebic{gamma:g}".replace(".", "")


def start_name(text: str) -> str:
    """A start's name, validated (the ``type`` of --up-start and --pilot): one of ``STARTS`` or
    ``truth-<tau>`` with ``tau >= 0``."""
    if text in STARTS:
        return text
    if text.startswith(TRUTH_START):
        try:
            if float(text[len(TRUTH_START):]) >= 0:
                return text
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"expected one of {STARTS} or truth-<tau> with tau >= 0, got {text!r}")


def make_start(name: str, sigma_hat, c_est, m_true, scale, rng) -> np.ndarray | None:
    """The dense start ``name`` for one data set, or None for ``lasso`` (the path functions compute
    the lasso's dense end themselves, exactly as before this option existed).  The noisy truth is
    drawn from the data set's generator after the data, so it is reproducible per data set."""
    if name == "lasso":
        return None
    if name == "exact":
        return covariance.dense_fit(sigma_hat, c_est)
    if name == "l1":
        return min_l1_exact_fit(sigma_hat, c_est)
    if name == "l2":
        return min_l2_exact_fit(sigma_hat, c_est)
    return noisy_truth_start(standardized_drift(m_true, scale), float(name[len(TRUTH_START):]), rng)


def start_diagnostics(start: np.ndarray, sigma_hat, c_est, m_true) -> dict:
    """The ``start_*`` fields: norms, Lyapunov residual, size of the support at ``START_PRESENT``, and
    that support's confusion counts and orientation breakdown against the truth."""
    off = ~np.eye(start.shape[0], dtype=bool)
    present = (np.abs(start) > START_PRESENT) & off
    conf, orient = _counts(present, m_true)
    return {"start_l1": float(np.abs(start[off]).sum()), "start_l2": float(np.linalg.norm(start[off])),
            "start_residual": exact_fit_residual(start, sigma_hat, c_est), "start_nnz": int(present.sum()),
            "start_conf": conf, "start_orient": orient}


def git_revision() -> dict:
    """The commit the code was run from (code review C3): its short hash and whether the
    working tree had uncommitted changes; ``"unknown"`` when git or the repository is not
    available.  A queued cluster task runs whatever is checked out when it starts, so the
    shard records it, not the submission."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=20, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=20, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": head, "dirty": dirty}


def provenance(shard: int, n_shards: int, t0: float) -> str:
    """The provenance record of a shard as JSON: host, versions, the git commit, timing."""
    return json.dumps({
        "host": socket.gethostname(), "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), **git_revision(),
        "shard": shard, "n_shards": n_shards, "wall_seconds": time.time() - t0,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def save_npz(out: Path, payload: dict) -> None:
    """Write a shard atomically: to a temporary name first, then renamed into place, so that a
    task killed while saving leaves no truncated shard that ``--fill`` would count as written
    (code review C4, 13 October 2026)."""
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out)
#: order of the entries of ``bic_orient`` / ``search_orient`` (gclm.metrics.orientation_breakdown)
ORIENT = ("correct", "reversed", "hedged", "both", "half", "missed_single", "missed_double",
          "fp_single", "fp_double")
#: per-dataset fields whose length depends on p or on the graph: stored as object arrays
RAGGED = ("m_", "supports_packed", "scale")


def pack(store: dict[str, list], ids=("p", "k", "c_choice", "rep")) -> dict:
    """The arrays of a shard from per-data-set lists.  Ragged fields (sparse triples, packed
    supports, the scale: different lengths per data set) become object arrays, built
    explicitly because ``np.array(list_of_arrays, dtype=object)`` silently collapses to 2-D
    when the lengths happen to coincide, which would make the layout depend on the data.  A
    shard with no data set at all (more shards than tasks) still gets the identifier columns,
    empty, so that the analysis can read it (code review C5, 13 October 2026)."""
    payload = {}
    for key, vals in store.items():
        if key.startswith(RAGGED):
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
        else:
            arr = np.array(vals)
        payload[key] = arr
    for key in ids:
        payload.setdefault(key, np.zeros(0, dtype=int))
    return payload


def task_list(cfg: S1Config):
    """The full (p, k, C_choice, rep) grid, in a fixed, reproducible order."""
    return [
        (p, k, c, r)
        for p in cfg.p_values
        for k in cfg.k_values
        for c in cfg.c_choices
        for r in range(cfg.n_rep)
    ]


def sparse_triple(m: np.ndarray):
    """(rows, cols, values) of the nonzero entries -- compact and lossless."""
    i, j = np.nonzero(m)
    return i.astype(np.int16), j.astype(np.int16), m[i, j].astype(np.float64)


def unpack_supports(packed: np.ndarray, n_lambda: int, p: int) -> np.ndarray:
    """Inverse of the ``supports_packed`` field: a ``(n_lambda, p, p)`` boolean array,
    ``[l, i, j]`` true iff ``M_hat[i, j] != 0`` at the l-th lambda (off-diagonal)."""
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: n_lambda * p * p]
    return bits.reshape(n_lambda, p, p).astype(bool)


def check_method(cfg: S1Config) -> None:
    """The combinations of ``method`` with the other settings that are implemented."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}; expected one of {METHODS}")
    for field, value in (("up_start", cfg.up_start), ("pilot", cfg.pilot)):
        if value is not None:
            try:
                start_name(value)
            except argparse.ArgumentTypeError as err:
                raise ValueError(f"{field}: {err}") from None
    if cfg.up_start is not None and (cfg.direction != "up" or cfg.method != "path"
                                     or (cfg.loss == "direct" and cfg.penalty == "lasso")):
        raise ValueError("up_start chooses where a dense -> sparse path starts: it needs --direction up "
                         "with --method path (and MCP or SCAD on the direct loss, whose lasso path has "
                         "no order)")
    if cfg.pilot != "lasso" and cfg.method not in PILOT_METHODS:
        raise ValueError(f"pilot applies to --method {', '.join(PILOT_METHODS)}")
    if cfg.weights not in WEIGHTS or cfg.graph not in GRAPHS:
        raise ValueError(f"weights must be one of {WEIGHTS} and graph one of {GRAPHS}")
    if cfg.weights == "normal" and cfg.w_min != 0.0:
        raise ValueError("w_min applies to --weights uniform (the normal weights have no lower bound)")
    if cfg.graph != "pairs" and cfg.two_cycle_share != 0.0:
        raise ValueError("two_cycle_share applies to --graph pairs")
    if cfg.penalize_diagonal:                   # wave 10 (21 October 2026): Dettling's eq. (1.4)
        if cfg.loss != "direct":
            raise ValueError("--penalize-diagonal applies to the direct loss only (the covariance losses leave "
                             "the diagonal unpenalised)")
        if cfg.method not in ("path", "adaptive"):
            raise ValueError("--penalize-diagonal is implemented for --method path and adaptive only")
        if cfg.start_name != "lasso" or cfg.pilot != "lasso":
            raise ValueError("--penalize-diagonal starts from the lasso's dense end only")
        if cfg.penalty != "lasso" and cfg.convention != "textbook":
            raise ValueError("--penalize-diagonal with MCP / SCAD uses the textbook convention")
    if cfg.method == "path":
        return
    if cfg.method == "adaptive" and cfg.loss != "direct":
        if cfg.penalty != "lasso":
            raise ValueError("method 'adaptive' is the adaptive lasso: leave --penalty at lasso")
        return
    if cfg.loss != "direct" or cfg.solver != "fista":
        raise ValueError(f"method {cfg.method!r} is implemented for the direct loss with the fista solver")
    if cfg.method == "lla" and cfg.penalty == "lasso":
        raise ValueError("method 'lla' needs --penalty MCP or SCAD")
    if cfg.method == "lla" and cfg.convention != "textbook":
        raise ValueError("method 'lla' uses the textbook convention")
    if cfg.method == "adaptive" and cfg.penalty != "lasso":
        raise ValueError("method 'adaptive' is the adaptive lasso: leave --penalty at lasso")
    if cfg.method.startswith("threshold") and cfg.penalty != "lasso":
        raise ValueError(f"method {cfg.method!r} is the thresholded lasso: leave --penalty at lasso")


def _counts(support: np.ndarray, m_true: np.ndarray):
    """Confusion counts (tp, fp, tn, fn) and the orientation breakdown of a support."""
    est = support.astype(float)
    cf = confusion(est, m_true, include_diagonal=False)
    ob = orientation_breakdown(est, m_true)
    return (np.array([cf.tp, cf.fp, cf.tn, cf.fn], dtype=np.int32),
            np.array([ob[key] for key in ORIENT], dtype=np.int32))


def select_graph(supports, sigma_hat, c_est, n_obs, m_true, search: bool,
                 refit: str = "direct", add_screen: int | None = None,
                 ebic_gamma: float = 0.0) -> dict:
    """What one gets from the path without knowing the truth.

    Every support of the path is refitted without penalty and given the likelihood
    score (BIC penalty) of the implied covariance, under the same ``C`` the path was
    fitted with (:class:`gclm.solvers.search.Scorer`; the same rule for every loss
    and penalty, so that only the paths differ).  ``refit="direct"``: least squares
    on the direct loss (closed form; the campaign's default).  ``refit="loglik"``:
    the maximised Gaussian likelihood on the support, i.e. the likelihood refit, as in
    Amendola et al. 2020 and in Dettling's eq. 6.1 (wave 5; an iterative refit,
    about 0.05 s per support at p = 10 against microseconds for least squares).
    The best-scoring support is the selected graph.  With ``search`` it is then
    the start of :func:`gclm.solvers.search.greedy_search`; ``add_screen`` limits
    the add moves scored per step to the ones with the largest gradient (only
    meaningful with the likelihood refit; ``None`` scores them all, which at p = 10
    costs 15 to 60 s per graph and keeps the search identical in its moves to the
    least-squares one).  ``ebic_gamma > 0`` adds Dettling's eBIC term
    ``4 gamma |S| log p`` to every score, i.e. scores with the eBIC penalty
    (:data:`EBIC_FORM`; wave 5c), for the selection and the search alike.
    """
    t0 = time.perf_counter()
    if refit not in REFITS:
        raise ValueError(f"unknown refit {refit!r}; expected one of {REFITS}")
    scorer = Scorer(sigma_hat, c_est, n_obs, refit, ebic_gamma, EBIC_FORM)
    ib, scores = bic_along_path(sigma_hat, c_est, n_obs, supports, scorer=scorer)
    conf, orient = _counts(supports[ib], m_true)
    out = {"bic_index": ib, "bic_scores": np.array(scores, dtype=float),
           "bic_evaluations": scorer.evaluations, "bic_conf": conf, "bic_orient": orient,
           # every support of the path scored +inf (no stable refit): argmin then gives 0, the
           # densest graph, which is not a selection; the analysis counts these (code review A3)
           "bic_dead": not np.isfinite(scores).any(),
           # refits that stopped at the solver's iteration cap (likelihood refit; code review A2)
           "bic_unconverged": scorer.unconverged}
    if search:
        t1 = time.perf_counter()
        # Every accepted move lowers the score, so the search ends by itself.  The cap
        # only guards against a runaway; it is set well above the library's default
        # of 200, because a dense selected graph at p = 20 can need more than a
        # hundred deletions (the number of moves made is stored in search_moves).
        p = len(m_true)
        res = greedy_search(sigma_hat, c_est, n_obs, supports[ib], scorer=scorer,
                            max_steps=p * (p - 1), loss=refit, add_screen=add_screen)
        conf, orient = _counts(res.support, m_true)
        kinds = [move[0] for move in res.moves]
        si, sj, sv = sparse_triple(np.where(res.support | np.eye(len(m_true), dtype=bool), res.m, 0.0))
        out.update({
            "m_search_support": np.packbits(res.support),
            "m_search_i": si, "m_search_j": sj, "m_search_v": sv,
            "search_conf": conf, "search_orient": orient,
            "search_score": res.score, "search_evaluations": scorer.evaluations,
            "search_moves": np.array([kinds.count("add"), kinds.count("delete"),
                                      kinds.count("reverse")], dtype=np.int32),
            "search_seconds": time.perf_counter() - t1,
            "search_dead": not np.isfinite(res.score),           # ended at a graph without a stable refit
            "search_unconverged": res.unconverged,               # refits at the cap during the search
            "search_capped": res.capped,                         # stopped by max_steps, not by the score
            "search_mismatches": res.mismatches,                 # gclm.solvers.search._consistent
        })
    out["select_seconds"] = time.perf_counter() - t0
    return out


def _direct_kkt(path, sigma_hat, c_est, lams, weights, penalty, gamma, scale) -> np.ndarray:
    """The first-order violation of every estimate of a direct-loss path
    (:func:`gclm.objective.penalties.stationarity`), the same diagnostic the covariance-loss
    paths report.  An entry the adaptive lasso excludes (weight ``inf``) satisfies its condition
    whatever the gradient, which a large finite weight expresses."""
    w = np.where(np.isinf(weights), 1e300, weights)
    return np.array([stationarity(m, direct_grad(m, sigma_hat, c_est), lam, w, penalty, gamma, scale)
                     for lam, m in zip(lams, path.estimates)])


def run_one(p, k, c_choice, rep, cfg, select: str = "none", refit: str = "direct",
            add_screen: int | None = None, ebic_gammas=()):
    """One dataset -> a dict of arrays.  No metric is reduced away here.

    ``ebic_gammas``: besides the score with the BIC penalty (always recorded), select
    and search once more per gamma with the eBIC penalty on the same path; those fields
    carry the prefix ``ebic<gamma>_`` (``m_ebic<gamma>_...`` for the stored matrices), so
    that with and without the eBIC term are paired graph by graph.
    """
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c_choice), rep])
    t0 = time.perf_counter()

    m_true, c_true, _, sigma_hat, scale = draw_instance(
        p, k, cfg.n_obs, c_choice, rng,
        metzler=cfg.metzler, standardize=cfg.standardize, return_scale=True,
        weights=cfg.weights, w_min=cfg.w_min, graph=cfg.graph, two_cycle_share=cfg.two_cycle_share,
    )
    c_est = estimation_volatility(scale, cfg.c_scale)
    weights = penalty_weights(p, penalize_diagonal=cfg.penalize_diagonal)
    n_l = cfg.n_lambda
    # the dense start of a dense -> sparse path, or the pilot of the adaptive / thresholded lasso
    # (plan of 13 October 2026, E1 / E2): None = the path function computes the lasso's dense end
    # itself, as it always did; otherwise the matrix of make_start.  Its diagnostics go into the row.
    dense_start = cfg.method in PILOT_METHODS or (
        cfg.method == "path" and cfg.direction == "up" and not (cfg.loss == "direct" and cfg.penalty == "lasso"))
    start_matrix = None
    if dense_start:
        start_matrix = make_start(cfg.pilot if cfg.method in PILOT_METHODS else cfg.start_name,
                                  sigma_hat, c_est, m_true, scale, rng)
    # solver diagnostics per lambda: the first-order violation, the solver's steps and whether
    # it met its tolerance.  The covariance-loss paths report them; for the direct loss the
    # steps and the flag come from FISTA (code review A4) and the violation is computed here;
    # newton_steps exist for the covariance losses only (-1 otherwise)
    kkt = np.full(n_l, np.nan)
    iters = np.full(n_l, -1, dtype=np.int32)
    newton = np.full(n_l, -1, dtype=np.int32)
    converged = np.ones(n_l, dtype=bool)
    if cfg.loss == "direct" and cfg.method == "adaptive":
        # the adaptive lasso has its own weights and therefore its own grid
        path = adaptive_lasso_path(sigma_hat, c_est, n_lambda=n_l, ratio=cfg.lambda_ratio,
                                   tol=cfg.tol, pilot=start_matrix, penalize_diagonal=cfg.penalize_diagonal)
        lams, lmax = path.lambdas, float(path.lambdas[-1])
        obj = np.array([direct_loss(m_hat, sigma_hat, c_est)
                        + lam * float(np.sum(path.weights[m_hat != 0] * np.abs(m_hat[m_hat != 0])))
                        for lam, m_hat in zip(lams, path.estimates)])
        kkt = _direct_kkt(path, sigma_hat, c_est, lams, path.weights, "lasso", None, None)
        iters, converged = path.iterations.astype(np.int32), path.converged
    elif cfg.loss == "direct" and cfg.method.startswith("threshold"):
        # the thresholded lasso: its "lambdas" are thresholds on |pilot_ij|, increasing, from 0 (the
        # pilot's support) to inf (the empty graph); no penalty, so the objective is the loss itself
        # and the first-order violation is not defined (NaN); the refits are closed forms
        path = thresholded_lasso_path(sigma_hat, c_est, pilot=start_matrix, n_thresholds=n_l,
                                      refit="none" if cfg.method == "threshold-norefit" else "direct",
                                      n_lambda=n_l, ratio=cfg.lambda_ratio, tol=cfg.tol)
        lams, lmax = path.lambdas, float(path.thresholds[-2])     # the largest |pilot_ij|: above it, empty
        obj = np.array([direct_loss(m_hat, sigma_hat, c_est) for m_hat in path.estimates])
        iters, converged = path.iterations.astype(np.int32), path.converged
    elif cfg.loss == "direct":
        # the smallest lambda with a diagonal estimate under the cell's penalty: with the diagonal
        # unpenalised the same closed form for the lasso, MCP and SCAD (waves 1 - 9); with it penalised
        # each penalty's own (wave 10)
        lmax = lambda_max(sigma_hat, c_est, penalize_diagonal=cfg.penalize_diagonal, penalty=cfg.penalty,
                          gamma=cfg.gamma)
        lams = lambda_grid(lmax, n_lambda=n_l, ratio=cfg.lambda_ratio)
        if cfg.method == "lla":
            path = lla_path(sigma_hat, c_est, lambdas=lams, penalty=cfg.penalty,
                            gamma=cfg.gamma, tol=cfg.tol)
        else:
            path = lasso_path(
                sigma_hat, c_est, lambdas=lams,
                penalize_diagonal=cfg.penalize_diagonal, solver=cfg.solver,
                penalty=cfg.penalty, gamma=cfg.gamma, convention=cfg.convention, tol=cfg.tol,
                direction=cfg.direction, start=start_matrix,
            )
        obj = np.array([objective(m_hat, sigma_hat, c_est, lam, weights,
                                  cfg.penalty, cfg.gamma, cfg.convention)
                        for lam, m_hat in zip(lams, path.estimates)])
        kkt = _direct_kkt(path, sigma_hat, c_est, lams, weights, cfg.penalty, cfg.gamma,
                          penalty_scale(sigma_hat, cfg.convention))
        iters, converged = path.iterations.astype(np.int32), path.converged
    else:
        if cfg.penalize_diagonal:
            raise ValueError("the covariance losses leave the diagonal unpenalised")
        if cfg.method == "adaptive":
            # its own weights and therefore its own grid, as on the direct loss
            path = adaptive_covloss_path(sigma_hat, c_est, cfg.loss, n_lambda=n_l,
                                         ratio=cfg.lambda_ratio, tol=cfg.tol, pilot=start_matrix)
            lams, lmax = path.lambdas, float(path.lambdas[-1])
        else:
            lmax = covariance.lambda_max(sigma_hat, c_est, cfg.loss)
            lams = lambda_grid(lmax, n_lambda=n_l, ratio=cfg.lambda_ratio)
            path = covloss_path(
                sigma_hat, c_est, cfg.loss, lambdas=lams, penalty=cfg.penalty,
                gamma=cfg.gamma, direction=cfg.direction, tol=cfg.tol,
                start=cfg.start_name if start_matrix is None else start_matrix,
            )
        obj, kkt = path.objective, path.kkt
        iters, newton, converged = path.iterations, path.newton_steps, path.converged
    start_row = {}
    if dense_start:
        used = path.pilot if cfg.method in PILOT_METHODS else path.start
        start_row = start_diagnostics(used, sigma_hat, c_est, m_true)

    off = np.empty((n_l, 4), dtype=np.int32)      # tp, fp, tn, fn  (off-diagonal)
    inc = np.empty((n_l, 4), dtype=np.int32)      # ... including the diagonal
    nnz = np.empty(n_l, dtype=np.int32)
    eye = np.eye(p, dtype=bool)
    for i, m_hat in enumerate(path.estimates):
        a = confusion(m_hat, m_true, include_diagonal=False)
        b = confusion(m_hat, m_true, include_diagonal=True)
        off[i] = (a.tp, a.fp, a.tn, a.fn)
        inc[i] = (b.tp, b.fp, b.tn, b.fn)
        nnz[i] = int(np.sum(m_hat[~eye] != 0))

    # F1 / accuracy from the off-diagonal counts, to pick landmark lambdas
    tp, fp, tn, fn = off.T
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        acc = (tp + tn) / np.maximum(tp + tn + fp + fn, 1)
    i_f1, i_acc = int(np.argmax(f1)), int(np.argmax(acc))

    ti, tj, tv = sparse_triple(m_true)
    f1i, f1j, f1v = sparse_triple(path.estimates[i_f1])
    aci, acj, acv = sparse_triple(path.estimates[i_acc])

    row = {
        "p": p, "k": k, "c_choice": list(CChoice).index(c_choice), "rep": rep,
        "lambda_max": lmax, "lambdas": lams,
        "conf_offdiag": off, "conf_incdiag": inc, "nnz": nnz, "objective": obj,
        "kkt": kkt, "iterations": iters, "newton_steps": newton, "converged": converged,
        "n_true_edges": int(np.sum(m_true[~eye] != 0)),
        "m_true_i": ti, "m_true_j": tj, "m_true_v": tv,
        "best_f1_index": i_f1, "best_acc_index": i_acc,
        "m_best_f1_i": f1i, "m_best_f1_j": f1j, "m_best_f1_v": f1v,
        "m_best_acc_i": aci, "m_best_acc_j": acj, "m_best_acc_v": acv,
        **start_row,
    }
    if select != "none":
        supports = [(m_hat != 0) & ~eye for m_hat in path.estimates]
        row["supports_packed"] = np.packbits(np.array(supports))
        row["scale"] = np.asarray(scale, dtype=float)
        row.update(select_graph(supports, sigma_hat, c_est, cfg.n_obs, m_true,
                                search=select == "search", refit=refit, add_screen=add_screen))
        for gamma in ebic_gammas:
            tag = ebic_tag(gamma)
            extra = select_graph(supports, sigma_hat, c_est, cfg.n_obs, m_true,
                                 search=select == "search", refit=refit, add_screen=add_screen,
                                 ebic_gamma=gamma)
            for key, val in extra.items():
                row[f"m_{tag}_{key[2:]}" if key.startswith("m_") else f"{tag}_{key}"] = val
    row["seconds"] = time.perf_counter() - t0
    return row


def build_parser() -> argparse.ArgumentParser:
    """The command line.  Separate from :func:`main` so that the submit scripts'
    cell definitions can be checked against it (tests/test_campaign_submit.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=None, help="override n_rep (testing)")
    ap.add_argument("--p", type=int, nargs="+", default=None)
    ap.add_argument("--n-obs", type=parse_n_obs, default=None,
                    help="sample size, e.g. 1000, 1e5 or inf (the population "
                         "covariance); default 1000.  M* and C do not depend on it, "
                         "so runs at different n are paired dataset by dataset")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--penalize-diagonal", action="store_true",
                    help="penalise the diagonal of M as well (Dettling's eq. (1.4)); direct loss, --method path "
                         "or adaptive (wave 10)")
    ap.add_argument("--penalty", default=None, choices=["lasso", "MCP", "SCAD"])
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--convention", default=None, choices=["textbook", "ncvreg"])
    ap.add_argument("--loss", default=None, choices=["direct", "loglik", "frobenius"],
                    help="direct: Dettling's loss (default); loglik / frobenius: "
                         "Varando's losses on Sigma(M) (docs/LIKELIHOOD.md)")
    ap.add_argument("--up-start", default=None, type=start_name, metavar="START",
                    help="with --direction up (MCP/SCAD): where the dense -> sparse path starts: lasso "
                         "(the lasso solution at the smallest lambda; the direct loss's default), exact "
                         "(-C Sigma^-1/2, Varando & Hansen; the covariance losses' default), l1 or l2 "
                         "(the minimum-l1 / minimum-Frobenius exact fits), truth-<tau> (the standardised "
                         "truth plus tau N(0,1) off the diagonal; a ceiling)")
    ap.add_argument("--pilot", default=None, type=start_name, metavar="START",
                    help="with --method adaptive, threshold or threshold-norefit: the pilot, same "
                         "choices as --up-start (default lasso: the dense end of the lasso path)")
    ap.add_argument("--weights", default=None, choices=list(WEIGHTS),
                    help="law of the true weights: normal (N(0,1), default) or uniform "
                         "(|w| ~ U[w_min, 1] with a random sign)")
    ap.add_argument("--w-min", type=float, default=None, metavar="W",
                    help="with --weights uniform: the lower bound of |w| (default 0)")
    ap.add_argument("--c-choices", default=None, nargs="+", choices=[c.value for c in CChoice], metavar="C",
                    help="restrict the true C to one or more of Dettling's four choices (default all four); "
                         "a graph keeps its seed, so a C_ID graph is the same graph as in a full run")
    ap.add_argument("--two-cycle-share", type=float, default=None, metavar="S",
                    help="with --graph pairs: the share of connected pairs that are 2-cycles, in [0, 1]; the "
                         "expected number of edges stays p (p - 1) d (E5, 19 October 2026)")
    ap.add_argument("--graph", default=None, choices=list(GRAPHS),
                    help="graph class of the truth: random (default), dag (a random DAG at the same expected number of edges) or "
                         "pairs (--two-cycle-share: the share of connected pairs that are 2-cycles)")
    ap.add_argument("--direction", default=None, choices=["down", "up"],
                    help="path order (default: down): for the covariance losses, and for "
                         "MCP/SCAD on the direct loss (up = dense to sparse)")
    ap.add_argument("--c-scale", default=None, choices=list(C_SCALES),
                    help="volatility used for estimation: identity (2 I, default) or "
                         "variance (2 diag(1/s_i^2), the rescaled C)")
    ap.add_argument("--method", default=None, choices=list(METHODS),
                    help="direct loss: path (warm-started continuation, default), lla "
                         "(MCP/SCAD by local linear approximation from the lasso), "
                         "adaptive (adaptive lasso), threshold (the lasso's dense end thresholded "
                         "by size and refitted by least squares) or threshold-norefit (the same "
                         "supports with the pilot's values)")
    ap.add_argument("--select", default="none", choices=list(SELECT),
                    help="also record the graph of the path selected by the score (bic) and "
                         "the graph the greedy search reaches from it (search), plus the "
                         "support at every lambda; default none = the original output")
    ap.add_argument("--refit", default=None, choices=list(REFITS),
                    help="refit behind the score: least squares on the direct loss (default) "
                         "or the maximised Gaussian likelihood (the likelihood refit; slow); "
                         "needs --select")
    ap.add_argument("--add-screen", type=int, default=None, metavar="N",
                    help="with --refit loglik and --select search: score only the N add "
                         "moves with the largest gradient per step (default: all of them)")
    ap.add_argument("--ebic-gamma", type=float, nargs="+", default=[], metavar="G",
                    help="with --select: besides the BIC penalty, also select (and search) with "
                         "Dettling's eBIC penalty (eBIC term 4 G |E| log p) for each G > 0; "
                         "fields ebic<G>_* (e.g. ebic05_, ebic1_)")
    return ap


def config_from_args(args: argparse.Namespace) -> S1Config:
    """The run's configuration: the defaults of :class:`S1Config`, overridden by
    whatever was given on the command line.  Raises ``ValueError`` for a
    combination that is not implemented (:func:`check_method`)."""
    base = S1Config()
    if args.reps is not None and args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.tol is not None and not args.tol > 0:
        raise ValueError("--tol must be positive")
    if args.p is not None and (len(args.p) == 0 or any(p < 2 for p in args.p)):
        raise ValueError("--p takes sizes of at least 2")
    cfg = S1Config(
        p_values=tuple(args.p) if args.p else base.p_values,
        n_rep=args.reps if args.reps is not None else base.n_rep,
        n_obs=args.n_obs if args.n_obs is not None else base.n_obs,
        solver=args.solver or base.solver,
        tol=args.tol if args.tol is not None else base.tol,
        penalty=args.penalty or base.penalty,
        gamma=args.gamma if args.gamma is not None else base.gamma,
        convention=args.convention or base.convention,
        loss=args.loss or base.loss,
        direction=args.direction or base.direction,
        c_scale=args.c_scale or base.c_scale,
        method=args.method or base.method,
        up_start=args.up_start if args.up_start is not None else base.up_start,
        pilot=args.pilot or base.pilot,
        weights=args.weights or base.weights,
        w_min=args.w_min if args.w_min is not None else base.w_min,
        graph=args.graph or base.graph,
        two_cycle_share=args.two_cycle_share if args.two_cycle_share is not None else base.two_cycle_share,
        c_choices=tuple(CChoice(c) for c in args.c_choices) if args.c_choices else base.c_choices,
        penalize_diagonal=args.penalize_diagonal or base.penalize_diagonal,
    )
    # settings that would be recorded in config_json without having any effect (code review C5)
    if args.direction and cfg.method != "path":
        raise ValueError(f"--direction applies to --method path only; --method {cfg.method} ignores it")
    if args.pilot and cfg.method not in PILOT_METHODS:
        raise ValueError(f"--pilot applies to --method {', '.join(PILOT_METHODS)}; --method {cfg.method} ignores it")
    if args.w_min is not None and cfg.weights != "uniform":
        raise ValueError("--w-min applies to --weights uniform only")
    if args.convention and cfg.loss != "direct":
        raise ValueError("--convention applies to the direct loss only (the covariance losses have no design)")
    if args.solver and cfg.loss != "direct":
        raise ValueError("--solver applies to the direct loss only (the covariance losses have their own solver)")
    check_method(cfg)
    return cfg


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}, got {args.shard}")
    try:
        tags = [ebic_tag(g) for g in args.ebic_gamma]
    except ValueError as err:
        ap.error(str(err))
    if len(set(tags)) < len(tags):
        ap.error("--ebic-gamma takes distinct values (0 is the BIC penalty, always recorded)")
    if args.ebic_gamma and args.select == "none":
        ap.error("--ebic-gamma needs --select bic or search")
    if args.refit is not None and args.select == "none":
        ap.error("--refit has no effect without --select bic or search")
    refit = args.refit or "direct"
    if args.add_screen is not None and (refit != "loglik" or args.select != "search"):
        ap.error("--add-screen applies to the search with the likelihood refit (--select search --refit loglik)")
    try:
        cfg = config_from_args(args)
    except ValueError as err:
        ap.error(str(err))

    tasks = task_list(cfg)[args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:04d}_of_{args.n_shards:04d}.npz"

    print(f"shard {args.shard}/{args.n_shards}: {len(tasks)} datasets "
          f"[n={cfg.n_obs}, loss={cfg.loss}, penalty={cfg.penalty}, method={cfg.method}, "
          f"direction={cfg.direction}, start={cfg.start_name}, pilot={cfg.pilot}, c_scale={cfg.c_scale}, "
          f"weights={cfg.weights}, w_min={cfg.w_min}, graph={cfg.graph}, select={args.select}, "
          f"refit={refit}, ebic_gamma={args.ebic_gamma}] -> {out}",
          flush=True)

    store: dict[str, list] = {}
    t0 = time.time()
    for n, (p, k, c, r) in enumerate(tasks, 1):
        row = run_one(p, k, c, r, cfg, select=args.select, refit=refit,
                      add_screen=args.add_screen, ebic_gammas=tuple(args.ebic_gamma))
        for key, val in row.items():
            store.setdefault(key, []).append(val)
        if n % 25 == 0 or n == len(tasks):
            el = time.time() - t0
            print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                  f"eta {(len(tasks)-n)*el/n/60:.1f} min", flush=True)

    payload = pack(store)
    payload["config_json"] = json.dumps({
        **{f: getattr(cfg, f) for f in (
            "n_rep", "n_obs", "n_lambda", "lambda_ratio", "penalize_diagonal",
            "metrics_include_diagonal", "standardize", "metzler", "seed",
            "solver", "loss", "direction", "penalty", "convention", "tol", "c_scale",
            "method", "pilot", "weights", "w_min", "graph", "two_cycle_share")},
        "up_start": cfg.start_name,                  # the loss's default filled in
        "select": args.select, "refit": refit, "add_screen": args.add_screen,
        "ebic_gammas": list(args.ebic_gamma), "ebic_form": EBIC_FORM,
        "p_values": list(cfg.p_values),
        "k_values": list(cfg.k_values),
        "c_choices": [c.value for c in cfg.c_choices],
        # the gamma actually used (package default filled in), not just the request
        "gamma": resolve_gamma(cfg.penalty, cfg.gamma),
    })
    payload["c_choice_names"] = np.array([c.value for c in CChoice])
    payload["provenance_json"] = provenance(args.shard, args.n_shards, t0)
    save_npz(out, payload)
    print(f"done: {len(tasks)} datasets in {(time.time()-t0)/60:.1f} min -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
