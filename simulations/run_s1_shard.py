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

    python simulations/run_s1_shard.py --shard 0 --n-shards 64 \
        --out-dir runs/s1_dettling_reproduction/s1_shards
    python simulations/run_s1_shard.py --shard 0 --n-shards 64 --loss loglik \
        --penalty MCP --out-dir runs/s2_loglik_mcp/s1_shards
    python simulations/run_s1_shard.py --shard 0 --n-shards 64 --n-obs inf \
        --p 10 20 --reps 25 --out-dir runs/nsweep_p10-20/direct_lasso_ninf/s1_shards
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
from gclm.data.simulate import CChoice, draw_instance
from gclm.objective.direct import lambda_max
from gclm.objective.penalties import penalty_weights
from gclm.solvers.path import covloss_path, lambda_grid, lasso_path
from gclm.objective.direct import objective
from gclm.metrics import confusion
from gclm.objective.penalties import resolve_gamma


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


def run_one(p, k, c_choice, rep, cfg):
    """One dataset -> a dict of arrays.  No metric is reduced away here."""
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(c_choice), rep])
    t0 = time.perf_counter()

    m_true, c_true, _, sigma_hat = draw_instance(
        p, k, cfg.n_obs, c_choice, rng,
        metzler=cfg.metzler, standardize=cfg.standardize,
    )
    c_est = 2.0 * np.eye(p)
    weights = penalty_weights(p, penalize_diagonal=cfg.penalize_diagonal)
    n_l = cfg.n_lambda
    # solver diagnostics; the direct-loss solvers report none (FISTA stops on
    # the coefficient change), so those columns are NaN / -1 for loss="direct"
    kkt = np.full(n_l, np.nan)
    iters = np.full(n_l, -1, dtype=np.int32)
    newton = np.full(n_l, -1, dtype=np.int32)
    if cfg.loss == "direct":
        lmax = lambda_max(sigma_hat, c_est, penalize_diagonal=cfg.penalize_diagonal)
        lams = lambda_grid(lmax, n_lambda=n_l, ratio=cfg.lambda_ratio)
        path = lasso_path(
            sigma_hat, c_est, lambdas=lams,
            penalize_diagonal=cfg.penalize_diagonal, solver=cfg.solver,
            penalty=cfg.penalty, gamma=cfg.gamma, convention=cfg.convention, tol=cfg.tol,
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

    return {
        "p": p, "k": k, "c_choice": list(CChoice).index(c_choice), "rep": rep,
        "lambda_max": lmax, "lambdas": lams,
        "conf_offdiag": off, "conf_incdiag": inc, "nnz": nnz, "objective": obj,
        "kkt": kkt, "iterations": iters, "newton_steps": newton,
        "n_true_edges": int(np.sum(m_true[~eye] != 0)),
        "m_true_i": ti, "m_true_j": tj, "m_true_v": tv,
        "best_f1_index": i_f1, "best_acc_index": i_acc,
        "m_best_f1_i": f1i, "m_best_f1_j": f1j, "m_best_f1_v": f1v,
        "m_best_acc_i": aci, "m_best_acc_j": acj, "m_best_acc_v": acv,
        "seconds": time.perf_counter() - t0,
    }


def main() -> None:
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
                    help="path order for the covariance losses (default: down)")
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}, got {args.shard}")

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
    )

    tasks = task_list(cfg)[args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:04d}_of_{args.n_shards:04d}.npz"

    print(f"shard {args.shard}/{args.n_shards}: {len(tasks)} datasets "
          f"[n={cfg.n_obs}, loss={cfg.loss}, penalty={cfg.penalty}] -> {out}", flush=True)

    store: dict[str, list] = {}
    t0 = time.time()
    for n, (p, k, c, r) in enumerate(tasks, 1):
        row = run_one(p, k, c, r, cfg)
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
        if key.startswith("m_"):
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
            "solver", "loss", "direction", "penalty", "convention", "tol")},
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
