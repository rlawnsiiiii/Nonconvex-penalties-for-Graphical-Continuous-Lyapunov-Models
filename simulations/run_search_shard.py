#!/usr/bin/env python3
"""Search without a penalty -- cluster shard runner (wave 2 of the campaign of
October 2026, next_steps/051026/cluster_campaign_051026.md).

The estimators of ``run_s1_shard.py`` all start from a regularisation path.  This
runner uses no path at all.  On the datasets of Figure 5 (same seeds, so every
result is paired with the path-based ones) it records two things:

  pure    the greedy BIC search with add / delete / reverse moves, started from
          the empty graph and from ``--restarts`` random graphs; the best-scoring
          result is kept (Amendola, Dettling, Drton, Onori & Wu 2020, Section 5;
          :func:`gclm.solvers.search.multistart_search`)
  truth   the same search started from the TRUE graph.  Not an estimator: it is
          the ceiling for any search with this score.  If it leaves the truth, the
          score prefers another graph; if the other searches end somewhere worse
          than it does, they got stuck.

The score is the Gaussian BIC of an unpenalised least-squares refit on the direct
loss, under the volatility matrix chosen with ``--c-scale`` -- exactly the score
that ``run_s1_shard.py --select`` uses.  Numbers only, one ``.npz`` per shard:

  per dataset   p, k, c_choice, rep, M* (sparse), scale
  pure_*        support (packed bits, ``m_pure_support``), refit (``m_pure_i/j/v``),
                confusion counts, orientation breakdown (order: ORIENT), BIC, the
                final BIC of every start, how many starts ended at the truth
  truth_*       the same for the search started from the truth, plus the BIC of
                the truth itself and the moves that left it (add, delete, reverse)

    python simulations/run_search_shard.py --shard 0 --n-shards 16 --p 20 --reps 25 \
        --n-obs 1e4 --c-scale variance \
        --out-dir runs/campaign/search_p20_Cresc_n1e4/search_shards
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import C_SCALES, CChoice, draw_instance, estimation_volatility  # noqa: E402
from gclm.solvers.search import Scorer, greedy_search, multistart_search, random_support  # noqa: E402
from run_s1_shard import ORIENT, RAGGED, _counts, sparse_triple, task_list  # noqa: E402,F401

METHODS = ("pure", "truth")
#: second seed word of the random starts; with (p, k, C, rep) it fixes them per dataset,
#: whatever the shard layout
RESTART_SEED = 20261003
#: largest edge probability of a random start (the value used in S3b)
MAX_DENSITY = 0.3


def _graph(prefix: str, support: np.ndarray, m: np.ndarray, m_true: np.ndarray) -> dict:
    """The fields that describe one selected graph: support, refit, counts."""
    conf, orient = _counts(support, m_true)
    i, j, v = sparse_triple(np.where(support | np.eye(len(m_true), dtype=bool), m, 0.0))
    return {f"m_{prefix}_support": np.packbits(support),
            f"m_{prefix}_i": i, f"m_{prefix}_j": j, f"m_{prefix}_v": v,
            f"{prefix}_conf": conf, f"{prefix}_orient": orient}


def run_one(p, k, c_choice, rep, cfg, restarts: int, methods=METHODS) -> dict:
    """Both searches on one dataset."""
    c_index = list(CChoice).index(c_choice)
    rng = np.random.default_rng([cfg.seed, p, k, c_index, rep])
    t0 = time.perf_counter()
    m_true, _, _, sigma_hat, scale = draw_instance(
        p, k, cfg.n_obs, c_choice, rng,
        metzler=cfg.metzler, standardize=cfg.standardize, return_scale=True,
    )
    c_est = estimation_volatility(scale, cfg.c_scale)
    off = ~np.eye(p, dtype=bool)
    truth = (m_true != 0) & off
    ti, tj, tv = sparse_triple(m_true)
    row = {"p": p, "k": k, "c_choice": c_index, "rep": rep,
           "m_true_i": ti, "m_true_j": tj, "m_true_v": tv,
           "n_true_edges": int(truth.sum()), "scale": np.asarray(scale, dtype=float)}

    if "pure" in methods:
        t1 = time.perf_counter()
        start_rng = np.random.default_rng([RESTART_SEED, p, k, c_index, rep])
        starts = [random_support(p, start_rng, MAX_DENSITY) for _ in range(restarts)]
        starts.append(np.zeros((p, p), dtype=bool))               # the empty graph, last
        # max_steps: a guard only, as in run_s1_shard.select_graph (the search stops by
        # itself when no move lowers the BIC)
        best, results = multistart_search(sigma_hat, c_est, cfg.n_obs, starts, loss="direct",
                                          max_steps=p * (p - 1))
        row.update(_graph("pure", best.support, best.m, m_true))
        row.update({
            "pure_score": best.score,
            "pure_scores": np.array([r.score for r in results], dtype=float),
            "pure_exact_starts": sum(np.array_equal(r.support, truth) for r in results),
            "pure_empty_start_exact": int(np.array_equal(results[-1].support, truth)),
            "pure_evaluations": best.evaluations,
            "pure_seconds": time.perf_counter() - t1,
        })

    if "truth" in methods:
        t1 = time.perf_counter()
        scorer = Scorer(sigma_hat, c_est, cfg.n_obs, "direct")
        truth_score, _ = scorer(truth)
        res = greedy_search(sigma_hat, c_est, cfg.n_obs, truth, scorer=scorer,
                            max_steps=p * (p - 1))
        kinds = [move[0] for move in res.moves]
        row.update(_graph("truth", res.support, res.m, m_true))
        row.update({
            "truth_score": res.score, "truth_start_score": truth_score,
            "truth_moves": np.array([kinds.count("add"), kinds.count("delete"),
                                     kinds.count("reverse")], dtype=np.int32),
            "truth_evaluations": res.evaluations,
            "truth_seconds": time.perf_counter() - t1,
        })
    row["seconds"] = time.perf_counter() - t0
    return row


def build_parser() -> argparse.ArgumentParser:
    """The command line (separate from :func:`main` so that tests can check the
    submit script's cell definitions against it)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=None, help="override n_rep")
    ap.add_argument("--p", type=int, nargs="+", default=None)
    ap.add_argument("--n-obs", type=parse_n_obs, default=None,
                    help="sample size, e.g. 1000, 1e4 or inf (the population covariance)")
    ap.add_argument("--c-scale", default=None, choices=list(C_SCALES),
                    help="volatility used in the score: identity (2 I, default) or "
                         "variance (2 diag(1/s_i^2), the rescaled C)")
    ap.add_argument("--restarts", type=int, default=10,
                    help="random starting graphs of the pure search, besides the empty graph")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}, got {args.shard}")
    if args.restarts < 0:
        ap.error("--restarts must not be negative")

    base = S1Config()
    cfg = S1Config(
        p_values=tuple(args.p) if args.p else base.p_values,
        n_rep=args.reps if args.reps else base.n_rep,
        n_obs=args.n_obs if args.n_obs is not None else base.n_obs,
        c_scale=args.c_scale or base.c_scale,
    )
    tasks = task_list(cfg)[args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:04d}_of_{args.n_shards:04d}.npz"
    print(f"shard {args.shard}/{args.n_shards}: {len(tasks)} datasets "
          f"[n={cfg.n_obs}, c_scale={cfg.c_scale}, methods={args.methods}, "
          f"restarts={args.restarts}] -> {out}", flush=True)

    store: dict[str, list] = {}
    t0 = time.time()
    for n, (p, k, c, r) in enumerate(tasks, 1):
        row = run_one(p, k, c, r, cfg, args.restarts, tuple(args.methods))
        for key, val in row.items():
            store.setdefault(key, []).append(val)
        if n % 10 == 0 or n == len(tasks):
            el = time.time() - t0
            print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                  f"eta {(len(tasks)-n)*el/n/60:.1f} min", flush=True)

    payload = {}
    for key, vals in store.items():
        if key.startswith(RAGGED):                       # length depends on p / the graph
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
        else:
            arr = np.array(vals)
        payload[key] = arr
    payload["config_json"] = json.dumps({
        **{f: getattr(cfg, f) for f in ("n_rep", "n_obs", "standardize", "metzler", "seed",
                                        "c_scale")},
        "p_values": list(cfg.p_values), "k_values": list(cfg.k_values),
        "c_choices": [c.value for c in cfg.c_choices],
        "methods": list(args.methods), "restarts": args.restarts,
        "restart_seed": RESTART_SEED, "max_density": MAX_DENSITY, "score": "bic",
        "refit": "direct",
    })
    payload["c_choice_names"] = np.array([c.value for c in CChoice])
    payload["provenance_json"] = json.dumps({
        "host": socket.gethostname(), "python": sys.version.split()[0],
        "numpy": np.__version__, "platform": platform.platform(),
        "shard": args.shard, "n_shards": args.n_shards,
        "wall_seconds": time.time() - t0,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    np.savez_compressed(out, **payload)
    print(f"done: {len(tasks)} datasets in {(time.time()-t0)/60:.1f} min -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
