#!/usr/bin/env python3
"""Search from random graphs -- cluster shard runner (wave 2 of the campaign of
October 2026, next_steps/051026/cluster_campaign_051026.md).

The estimators of ``run_s1_shard.py`` all start from a regularisation path.  This
runner uses no path at all.  On the datasets of Figure 5 (same seeds, so every
result is paired with the path-based ones) it records two things:

  pure    the greedy search with add / delete / reverse moves, started from
          the empty graph and from ``--restarts`` randomly drawn graphs (``--starts``:
          sparse ones, or uniform over all directed graphs); the best-scoring
          result is kept (Amendola, Dettling, Drton, Onori & Wu 2020, Section 5;
          :func:`gclm.solvers.search.multistart_search`).  The graph and the score
          every start ended at are stored too, so that "the best of the first r
          starts" can be read off afterwards for any r (the restart study, wave 5)
  truth   the same search started from the TRUE graph.  Not an estimator: it is
          the ceiling for any search with this score.  If it leaves the truth, the
          score prefers another graph; if the other searches end somewhere worse
          than it does, they got stuck.

The score of a graph is its likelihood loss term plus the BIC penalty, under the volatility
matrix chosen with ``--c-scale``, the score that ``run_s1_shard.py --select`` uses.  The
model on the graph is fitted either by least squares on the direct loss (``--refit direct``,
the default: closed form) or by maximising the Gaussian likelihood (``--refit loglik``: the
likelihood refit, as in Amendola et al. 2020; wave 7); ``--ebic-gamma G`` adds Dettling's
eBIC term ``4 G |E| log p``, i.e. scores with the eBIC penalty (wave 5c), for both searches.
Numbers only, one ``.npz`` per shard:

  per dataset   p, k, c_choice, rep, M* (sparse), scale
  pure_*        support (packed bits, ``m_pure_support``), refit (``m_pure_i/j/v``),
                confusion counts, orientation breakdown (order: ORIENT), score; per
                start (random ones first, the empty graph last): the final score
                (``pure_scores``), the final support (``m_pure_starts_support``,
                packed, unpack with ``unpack_supports(.., restarts + 1, p)``) and
                the number of moves made; how many starts ended at the truth
  truth_*       the same for the search started from the truth, plus the score of
                the truth itself and the moves that left it (add, delete, reverse)

Start blocks (``--start-blocks B``, wave 8 (c)).  With the likelihood refit, 100 random starts of
one graph take longer than a cluster task may run.  Then the starts of every graph are split into
B contiguous blocks, one (graph, block) pair per task: block b searches starts b R/B ... (b+1) R/B - 1
(the same graphs, drawn in the same order as without blocks), the last block also the empty graph,
and block 0 also the search from the truth.  ``--n-shards`` must be a multiple of B, so that every
shard holds one block only; every row records ``start_block`` and ``start_blocks``.  The analysis
(``simulations/diagnostics/restarts.py`` and ``campaign.py``) puts the blocks of a graph back
together: the per-start results in block order, and the best-scoring end graph over all blocks.

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
from run_s1_shard import EBIC_FORM, ORIENT, RAGGED, _counts, sparse_triple, task_list  # noqa: E402,F401

METHODS = ("pure", "truth")
#: second seed word of the random starts; with (p, k, C, rep) it fixes them per dataset,
#: whatever the shard layout
RESTART_SEED = 20261003
#: how the random starting graphs are drawn.  "sparse": edge probability drawn from
#: U[0, MAX_DENSITY] per graph, then each entry independently (the choice of S3b and of the
#: campaign's wave 2).  "uniform": each entry with probability 1/2, i.e. uniformly over all
#: directed graphs on p nodes (2-cycles allowed) -- the analogue of the uniformly drawn
#: restarts of Nowzohour et al. (2017); needs no MCMC for this graph class.
STARTS = ("sparse", "uniform")
#: largest edge probability of a "sparse" random start (the value used in S3b)
MAX_DENSITY = 0.3


def _graph(prefix: str, support: np.ndarray, m: np.ndarray, m_true: np.ndarray) -> dict:
    """The fields that describe one selected graph: support, refit, counts."""
    conf, orient = _counts(support, m_true)
    i, j, v = sparse_triple(np.where(support | np.eye(len(m_true), dtype=bool), m, 0.0))
    return {f"m_{prefix}_support": np.packbits(support),
            f"m_{prefix}_i": i, f"m_{prefix}_j": j, f"m_{prefix}_v": v,
            f"{prefix}_conf": conf, f"{prefix}_orient": orient}


def run_one(p, k, c_choice, rep, cfg, restarts: int, methods=METHODS, starts: str = "sparse",
            ebic_gamma: float = 0.0, refit: str = "direct", add_screen: int | None = None,
            block: int = 0, blocks: int = 1) -> dict:
    """Both searches on one dataset; with ``blocks > 1`` only start block ``block`` of the pure
    search (see the module docstring), and the search from the truth only in block 0."""
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
        if starts == "sparse":
            start_graphs = [random_support(p, start_rng, MAX_DENSITY) for _ in range(restarts)]
        else:
            start_graphs = []
            for _ in range(restarts):
                g = start_rng.random((p, p)) < 0.5
                np.fill_diagonal(g, False)
                start_graphs.append(g)
        if blocks > 1:                                              # this block's random starts
            size = restarts // blocks
            start_graphs = start_graphs[block * size:(block + 1) * size]
        if block == blocks - 1:
            start_graphs.append(np.zeros((p, p), dtype=bool))     # the empty graph, last
        # max_steps: a guard only, as in run_s1_shard.select_graph (the search stops by
        # itself when no move lowers the score)
        best, results = multistart_search(sigma_hat, c_est, cfg.n_obs, start_graphs, loss=refit,
                                          max_steps=p * (p - 1), ebic_gamma=ebic_gamma,
                                          ebic_form=EBIC_FORM, add_screen=add_screen)
        row.update(_graph("pure", best.support, best.m, m_true))
        row.update({
            "pure_score": best.score,
            "pure_scores": np.array([r.score for r in results], dtype=float),
            "m_pure_starts_support": np.packbits(np.array([r.support for r in results])),
            "pure_start_moves": np.array([len(r.moves) for r in results], dtype=np.int32),
            "pure_exact_starts": sum(np.array_equal(r.support, truth) for r in results),
            "pure_empty_start_exact": (int(np.array_equal(results[-1].support, truth))
                                       if block == blocks - 1 else -1),
            "pure_evaluations": best.evaluations,
            "pure_seconds": time.perf_counter() - t1,
        })

    if "truth" in methods and block == 0:
        t1 = time.perf_counter()
        scorer = Scorer(sigma_hat, c_est, cfg.n_obs, refit, ebic_gamma, EBIC_FORM)
        truth_score, _ = scorer(truth)
        res = greedy_search(sigma_hat, c_est, cfg.n_obs, truth, scorer=scorer, loss=refit,
                            max_steps=p * (p - 1), add_screen=add_screen)
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
    if blocks > 1:
        row.update({"start_block": block, "start_blocks": blocks})
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
    ap.add_argument("--starts", default="sparse", choices=list(STARTS),
                    help="random starting graphs: sparse (edge probability ~ U[0, 0.3], "
                         "default) or uniform (every entry with probability 1/2)")
    ap.add_argument("--ebic-gamma", type=float, default=0.0, metavar="G",
                    help="add Dettling's eBIC term 4 G |E| log p to the score, i.e. use the eBIC "
                         "penalty (default 0)")
    ap.add_argument("--refit", default="direct", choices=["direct", "loglik"],
                    help="how the model on a graph is fitted for its score: least squares on the "
                         "direct loss (default) or the maximised Gaussian likelihood (slow)")
    ap.add_argument("--add-screen", type=int, default=None, metavar="N",
                    help="with --refit loglik: score only the N add moves with the largest "
                         "gradient per step (default: all of them)")
    ap.add_argument("--start-blocks", type=int, default=1, metavar="B",
                    help="split the random starts of every graph into B blocks, one (graph, block) "
                         "pair per task (default 1: all starts of a graph in one task); --restarts "
                         "and --n-shards must be multiples of B")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}, got {args.shard}")
    if args.restarts < 0:
        ap.error("--restarts must not be negative")
    if args.ebic_gamma < 0:
        ap.error("--ebic-gamma must not be negative")
    blocks = args.start_blocks
    if blocks < 1:
        ap.error("--start-blocks must be at least 1")
    if blocks > 1 and ("pure" not in args.methods or args.restarts % blocks
                       or args.n_shards % blocks):
        ap.error("--start-blocks B needs the pure search, and --restarts and --n-shards "
                 "multiples of B (every shard then holds one block)")

    base = S1Config()
    cfg = S1Config(
        p_values=tuple(args.p) if args.p else base.p_values,
        n_rep=args.reps if args.reps else base.n_rep,
        n_obs=args.n_obs if args.n_obs is not None else base.n_obs,
        c_scale=args.c_scale or base.c_scale,
    )
    tasks = [(*t, b) for t in task_list(cfg) for b in range(blocks)][args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard:04d}_of_{args.n_shards:04d}.npz"
    print(f"shard {args.shard}/{args.n_shards}: {len(tasks)} datasets "
          f"[n={cfg.n_obs}, c_scale={cfg.c_scale}, methods={args.methods}, "
          f"restarts={args.restarts}, starts={args.starts}, ebic_gamma={args.ebic_gamma}, "
          f"refit={args.refit}, add_screen={args.add_screen}, start_blocks={blocks}] -> {out}",
          flush=True)

    store: dict[str, list] = {}
    t0 = time.time()
    for n, (p, k, c, r, b) in enumerate(tasks, 1):
        row = run_one(p, k, c, r, cfg, args.restarts, tuple(args.methods), args.starts,
                      args.ebic_gamma, args.refit, args.add_screen, block=b, blocks=blocks)
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
        "methods": list(args.methods), "restarts": args.restarts, "starts": args.starts,
        "restart_seed": RESTART_SEED, "max_density": MAX_DENSITY, "score": "bic",
        "refit": args.refit, "add_screen": args.add_screen, "start_blocks": blocks,
        "ebic_gamma": args.ebic_gamma, "ebic_form": EBIC_FORM,
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
