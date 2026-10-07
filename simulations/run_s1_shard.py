#!/usr/bin/env python3
"""S1 / Figure 5 -- cluster shard runner.

Writes **numbers only**, no plots.  One compressed ``.npz`` per shard containing
everything needed to recompute any downstream quantity without re-running:

  per dataset      p, k, c_choice, rep, seed, lambda_max, timing, n_true_edges
  per (dataset, l) lambda, tp/fp/tn/fn off-diagonal AND including the diagonal,
                   nnz, penalised objective; for the covariance losses also the
                   first-order violation (kkt) and the solver's step counts
  per dataset      M* as a sparse triple, and M_hat at the best-F1 and best-acc
                   lambdas (also sparse)
  once             the full S1Config, so the run is self-describing

Storing the raw confusion counts (rather than only max_acc/max_f1/auc/aupr)
means every metric -- including ones not yet defined, and the
``include_diagonal`` variant -- can be recomputed locally from the output.

With ``--select bic`` or ``--select search`` (the campaign of October 2026,
next_steps/051026/cluster_campaign_051026.md) each dataset additionally gets

  supports_packed  the off-diagonal support at every lambda (np.packbits; decode
                   with :func:`unpack_supports`), and ``scale``, the standard
                   deviations the data were divided by
  bic_*            the Gaussian BIC of every support of the path after an
                   unpenalised least-squares refit (gclm.solvers.search), the
                   index of the best one, its confusion counts and its
                   orientation breakdown (order: ORIENT)
  search_*         (``search`` only) the graph reached from the BIC-selected
                   support by the greedy BIC search with add / delete / reverse
                   moves: its support (``m_search_support``, packed like one
                   lambda of ``supports_packed``), its unpenalised refit
                   ``m_search_i/j/v`` (sparse), counts, score, moves

Without ``--select`` the output is exactly what it was before these options existed.

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
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gclm import __name__ as _pkg  # noqa: F401  (ensures src/ is importable)
from gclm.objective import covariance
from gclm.config import S1Config, parse_n_obs
from gclm.data.simulate import C_SCALES, CChoice, draw_instance, estimation_volatility
from gclm.objective.direct import direct_loss, lambda_max
from gclm.objective.penalties import penalty_weights
from gclm.solvers.path import (adaptive_lasso_path, covloss_path, lambda_grid, lasso_path,
                               lla_path)
from gclm.objective.direct import objective
from gclm.metrics import confusion, orientation_breakdown
from gclm.objective.penalties import resolve_gamma
from gclm.solvers.search import Scorer, bic_along_path, greedy_search

METHODS = ("path", "lla", "adaptive")
SELECT = ("none", "bic", "search")
#: how a support is refitted for the BIC: least squares on the direct loss (closed form, the
#: campaign's default) or the maximised Gaussian likelihood (iterative; 200 to 1000 times slower)
REFITS = ("direct", "loglik")
#: the form of the extended-BIC term used by --ebic-gamma: Dettling's 4 gamma |E| log p (his
#: eq. 6.2), the same rule campaign.py applies offline to the path (columns ebic05_*, ebic1_*)
EBIC_FORM = "dettling"


def ebic_tag(gamma: float) -> str:
    """Field prefix of one extended-BIC gamma: 0.5 -> ``ebic05``, 1 -> ``ebic1``."""
    return f"ebic{gamma:g}".replace(".", "")
#: order of the entries of ``bic_orient`` / ``search_orient`` (gclm.metrics.orientation_breakdown)
ORIENT = ("correct", "reversed", "hedged", "both", "half", "missed_single", "missed_double",
          "fp_single", "fp_double")
#: per-dataset fields whose length depends on p or on the graph: stored as object arrays
RAGGED = ("m_", "supports_packed", "scale")


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
    if cfg.method == "path":
        return
    if cfg.loss != "direct" or cfg.solver != "fista" or cfg.penalize_diagonal:
        raise ValueError(f"method {cfg.method!r} is implemented for the direct loss with the "
                         "fista solver and an unpenalised diagonal")
    if cfg.method == "lla" and cfg.penalty == "lasso":
        raise ValueError("method 'lla' needs --penalty MCP or SCAD")
    if cfg.method == "lla" and cfg.convention != "textbook":
        raise ValueError("method 'lla' uses the textbook convention")
    if cfg.method == "adaptive" and cfg.penalty != "lasso":
        raise ValueError("method 'adaptive' is the adaptive lasso: leave --penalty at lasso")


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

    Every support of the path is refitted without penalty and scored by the
    Gaussian BIC of the implied covariance, under the same ``C`` the path was
    fitted with (:class:`gclm.solvers.search.Scorer`; the same rule for every loss
    and penalty, so that only the paths differ).  ``refit="direct"``: least squares
    on the direct loss (closed form; the campaign's default).  ``refit="loglik"``:
    the maximised Gaussian likelihood on the support, i.e. the BIC proper, as in
    Amendola et al. 2020 and in Dettling's eq. 6.1 (wave 5; an iterative refit,
    about 0.05 s per support at p = 10 against microseconds for least squares).
    The best-scoring support is the BIC-selected graph.  With ``search`` it is then
    the start of :func:`gclm.solvers.search.greedy_search`; ``add_screen`` limits
    the add moves scored per step to the ones with the largest gradient (only
    meaningful with the likelihood refit; ``None`` scores them all, which at p = 10
    costs 15 to 60 s per graph and keeps the search identical in its moves to the
    least-squares one).  ``ebic_gamma > 0`` adds Dettling's extended term
    ``4 gamma |S| log p`` to every score (:data:`EBIC_FORM`; wave 5c), for the
    selection and the search alike.
    """
    t0 = time.perf_counter()
    if refit not in REFITS:
        raise ValueError(f"unknown refit {refit!r}; expected one of {REFITS}")
    scorer = Scorer(sigma_hat, c_est, n_obs, refit, ebic_gamma, EBIC_FORM)
    ib, scores = bic_along_path(sigma_hat, c_est, n_obs, supports, scorer=scorer)
    conf, orient = _counts(supports[ib], m_true)
    out = {"bic_index": ib, "bic_scores": np.array(scores, dtype=float),
           "bic_evaluations": scorer.evaluations, "bic_conf": conf, "bic_orient": orient}
    if search:
        t1 = time.perf_counter()
        # Every accepted move lowers the BIC, so the search ends by itself.  The cap
        # only guards against a runaway; it is set well above the library's default
        # of 200, because a dense BIC-selected graph at p = 20 can need more than a
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
        })
    out["select_seconds"] = time.perf_counter() - t0
    return out


def run_one(p, k, c_choice, rep, cfg, select: str = "none", refit: str = "direct",
            add_screen: int | None = None, ebic_gammas=()):
    """One dataset -> a dict of arrays.  No metric is reduced away here.

    ``ebic_gammas``: besides the plain BIC (always recorded), select and search once
    more per gamma with the extended BIC on the same path; those fields carry the
    prefix ``ebic<gamma>_`` (``m_ebic<gamma>_...`` for the stored matrices), so that
    with and without the term are paired graph by graph.
    """
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c_choice), rep])
    t0 = time.perf_counter()

    m_true, c_true, _, sigma_hat, scale = draw_instance(
        p, k, cfg.n_obs, c_choice, rng,
        metzler=cfg.metzler, standardize=cfg.standardize, return_scale=True,
    )
    c_est = estimation_volatility(scale, cfg.c_scale)
    weights = penalty_weights(p, penalize_diagonal=cfg.penalize_diagonal)
    n_l = cfg.n_lambda
    # solver diagnostics; the direct-loss solvers report none (FISTA stops on
    # the coefficient change), so those columns are NaN / -1 for loss="direct"
    kkt = np.full(n_l, np.nan)
    iters = np.full(n_l, -1, dtype=np.int32)
    newton = np.full(n_l, -1, dtype=np.int32)
    if cfg.loss == "direct" and cfg.method == "adaptive":
        # the adaptive lasso has its own weights and therefore its own grid
        path = adaptive_lasso_path(sigma_hat, c_est, n_lambda=n_l, ratio=cfg.lambda_ratio,
                                   tol=cfg.tol)
        lams, lmax = path.lambdas, float(path.lambdas[-1])
        obj = np.array([direct_loss(m_hat, sigma_hat, c_est)
                        + lam * float(np.sum(path.weights[m_hat != 0] * np.abs(m_hat[m_hat != 0])))
                        for lam, m_hat in zip(lams, path.estimates)])
    elif cfg.loss == "direct":
        lmax = lambda_max(sigma_hat, c_est, penalize_diagonal=cfg.penalize_diagonal)
        lams = lambda_grid(lmax, n_lambda=n_l, ratio=cfg.lambda_ratio)
        if cfg.method == "lla":
            path = lla_path(sigma_hat, c_est, lambdas=lams, penalty=cfg.penalty,
                            gamma=cfg.gamma, tol=cfg.tol)
        else:
            path = lasso_path(
                sigma_hat, c_est, lambdas=lams,
                penalize_diagonal=cfg.penalize_diagonal, solver=cfg.solver,
                penalty=cfg.penalty, gamma=cfg.gamma, convention=cfg.convention, tol=cfg.tol,
                direction=cfg.direction,
            )
        obj = np.array([objective(m_hat, sigma_hat, c_est, lam, weights,
                                  cfg.penalty, cfg.gamma, cfg.convention)
                        for lam, m_hat in zip(lams, path.estimates)])
    else:
        if cfg.penalize_diagonal:
            raise ValueError("the covariance losses leave the diagonal unpenalised")
        lmax = covariance.lambda_max(sigma_hat, c_est, cfg.loss)
        lams = lambda_grid(lmax, n_lambda=n_l, ratio=cfg.lambda_ratio)
        path = covloss_path(
            sigma_hat, c_est, cfg.loss, lambdas=lams, penalty=cfg.penalty,
            gamma=cfg.gamma, direction=cfg.direction, tol=cfg.tol,
        )
        obj, kkt = path.objective, path.kkt
        iters, newton = path.iterations, path.newton_steps

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
        "kkt": kkt, "iterations": iters, "newton_steps": newton,
        "n_true_edges": int(np.sum(m_true[~eye] != 0)),
        "m_true_i": ti, "m_true_j": tj, "m_true_v": tv,
        "best_f1_index": i_f1, "best_acc_index": i_acc,
        "m_best_f1_i": f1i, "m_best_f1_j": f1j, "m_best_f1_v": f1v,
        "m_best_acc_i": aci, "m_best_acc_j": acj, "m_best_acc_v": acv,
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
    ap.add_argument("--penalty", default=None, choices=["lasso", "MCP", "SCAD"])
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--convention", default=None, choices=["textbook", "ncvreg"])
    ap.add_argument("--loss", default=None, choices=["direct", "loglik", "frobenius"],
                    help="direct: Dettling's loss (default); loglik / frobenius: "
                         "Varando's losses on Sigma(M) (docs/LIKELIHOOD.md)")
    ap.add_argument("--direction", default=None, choices=["down", "up"],
                    help="path order (default: down): for the covariance losses, and for "
                         "MCP/SCAD on the direct loss (up = dense to sparse)")
    ap.add_argument("--c-scale", default=None, choices=list(C_SCALES),
                    help="volatility used for estimation: identity (2 I, default) or "
                         "variance (2 diag(1/s_i^2), the rescaled C)")
    ap.add_argument("--method", default=None, choices=list(METHODS),
                    help="direct loss: path (warm-started continuation, default), lla "
                         "(MCP/SCAD by local linear approximation from the lasso) or "
                         "adaptive (adaptive lasso)")
    ap.add_argument("--select", default="none", choices=list(SELECT),
                    help="also record the BIC-selected graph of the path (bic) and the "
                         "graph the greedy BIC search reaches from it (search), plus the "
                         "support at every lambda; default none = the original output")
    ap.add_argument("--refit", default="direct", choices=list(REFITS),
                    help="refit behind the BIC: least squares on the direct loss (default) "
                         "or the maximised Gaussian likelihood (the BIC proper; slow)")
    ap.add_argument("--add-screen", type=int, default=None, metavar="N",
                    help="with --refit loglik and --select search: score only the N add "
                         "moves with the largest gradient per step (default: all of them)")
    ap.add_argument("--ebic-gamma", type=float, nargs="+", default=[], metavar="G",
                    help="with --select: besides the plain BIC, also select (and search) with "
                         "Dettling's extended BIC, 4 G |E| log p, for each G > 0; fields "
                         "ebic<G>_* (e.g. ebic05_, ebic1_)")
    return ap


def config_from_args(args: argparse.Namespace) -> S1Config:
    """The run's configuration: the defaults of :class:`S1Config`, overridden by
    whatever was given on the command line.  Raises ``ValueError`` for a
    combination that is not implemented (:func:`check_method`)."""
    base = S1Config()
    cfg = S1Config(
        p_values=tuple(args.p) if args.p else base.p_values,
        n_rep=args.reps if args.reps else base.n_rep,
        n_obs=args.n_obs if args.n_obs is not None else base.n_obs,
        solver=args.solver or base.solver,
        tol=args.tol if args.tol else base.tol,
        penalty=args.penalty or base.penalty,
        gamma=args.gamma if args.gamma is not None else base.gamma,
        convention=args.convention or base.convention,
        loss=args.loss or base.loss,
        direction=args.direction or base.direction,
        c_scale=args.c_scale or base.c_scale,
        method=args.method or base.method,
    )
    check_method(cfg)
    return cfg


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}, got {args.shard}")
    if any(g <= 0 for g in args.ebic_gamma) or len(set(args.ebic_gamma)) < len(args.ebic_gamma):
        ap.error("--ebic-gamma takes distinct positive values (0 is the plain BIC, always recorded)")
    if args.ebic_gamma and args.select == "none":
        ap.error("--ebic-gamma needs --select bic or search")
    try:
        cfg = config_from_args(args)
    except ValueError as err:
        ap.error(str(err))

    tasks = task_list(cfg)[args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:04d}_of_{args.n_shards:04d}.npz"

    print(f"shard {args.shard}/{args.n_shards}: {len(tasks)} datasets "
          f"[n={cfg.n_obs}, loss={cfg.loss}, penalty={cfg.penalty}, method={cfg.method}, "
          f"direction={cfg.direction}, c_scale={cfg.c_scale}, select={args.select}, "
          f"refit={args.refit}, ebic_gamma={args.ebic_gamma}] -> {out}",
          flush=True)

    store: dict[str, list] = {}
    t0 = time.time()
    for n, (p, k, c, r) in enumerate(tasks, 1):
        row = run_one(p, k, c, r, cfg, select=args.select, refit=args.refit,
                      add_screen=args.add_screen, ebic_gammas=tuple(args.ebic_gamma))
        for key, val in row.items():
            store.setdefault(key, []).append(val)
        if n % 25 == 0 or n == len(tasks):
            el = time.time() - t0
            print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                  f"eta {(len(tasks)-n)*el/n/60:.1f} min", flush=True)

    # Sparse-triple fields are ragged (different p and sparsity per dataset).
    # Build the object array explicitly: np.array(list_of_arrays, dtype=object)
    # silently collapses to 2-D when the lengths happen to coincide, which would
    # make the layout depend on the data.
    payload = {}
    for key, vals in store.items():
        if key.startswith(RAGGED):
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
        else:
            arr = np.array(vals)
        payload[key] = arr

    payload["config_json"] = json.dumps({
        **{f: getattr(cfg, f) for f in (
            "n_rep", "n_obs", "n_lambda", "lambda_ratio", "penalize_diagonal",
            "metrics_include_diagonal", "standardize", "metzler", "seed",
            "solver", "loss", "direction", "penalty", "convention", "tol", "c_scale",
            "method")},
        "select": args.select, "refit": args.refit, "add_screen": args.add_screen,
        "ebic_gammas": list(args.ebic_gamma), "ebic_form": EBIC_FORM,
        "p_values": list(cfg.p_values),
        "k_values": list(cfg.k_values),
        "c_choices": [c.value for c in cfg.c_choices],
        # the gamma actually used (package default filled in), not just the request
        "gamma": resolve_gamma(cfg.penalty, cfg.gamma),
    })
    payload["c_choice_names"] = np.array([c.value for c in CChoice])
    payload["provenance_json"] = json.dumps({
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "shard": args.shard, "n_shards": args.n_shards,
        "wall_seconds": time.time() - t0,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    np.savez_compressed(out, **payload)
    print(f"done: {len(tasks)} datasets in {(time.time()-t0)/60:.1f} min -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
