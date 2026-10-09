#!/usr/bin/env python3
"""Rescore an existing cell: the selection and the greedy search redone on the stored paths,
with the eBIC penalty (wave 6) or with the maximised likelihood behind the score (wave 8) --
cluster shard runner for the campaign of October 2026
(next_steps/051026/cluster_campaign_051026.md Section 3.4).

Two modes:

  --ebic-gamma G...   (wave 6) the eBIC term 4 G |E| log p inside the selection and the
                      search, written as overlay fields ebic<G>_* next to the source cell's own
                      (see below);
  --refit loglik      (wave 8) the model on each graph fitted by maximising the Gaussian
                      likelihood instead of least squares on the direct loss: the likelihood refit.
                      The output is a complete cell of its own (``<loss>_<estimator>-ml_...``):
                      every field of the source shard, with the selection (bic_*) and the search
                      (search_*, m_search_*) replaced.  ``--p`` keeps only the data sets of the
                      given sizes (the likelihood search is affordable at p = 10 only).

Every cell of ``run_s1_shard.py --select`` stores the support of every estimate of its path.
This runner reads those supports, rebuilds the data set from its seed (same generator, same
sample size, same C as the source cell: identical to the bit), and runs the selection and the
greedy search once more with Dettling's eBIC term ``4 gamma |E| log p`` inside the score
(``run_s1_shard.select_graph(..., ebic_gamma=gamma)``), for each ``--ebic-gamma``.  The paths are
not recomputed, which is most of a cell's cost; the output has the fields that
``run_s1_shard.py --ebic-gamma`` would have added (``ebic<gamma>_bic_*``, ``ebic<gamma>_search_*``,
``m_ebic<gamma>_search_*``), one file per source shard with the same name, so that
``simulations/diagnostics/campaign.py`` can overlay them on the source cell's rows.

As a check that the data set was rebuilt exactly, the scores (BIC penalty) are recomputed and
compared with the stored ones (``rebuilt_exact`` per data set; the runner stops on a mismatch).

    python simulations/rescore_shard.py --shard 0 --n-shards 4 --n-obs 1e4 \
        --source-cell direct_lasso_Cresc --ebic-gamma 1 --select search \
        --out-dir runs/campaign/rescore1_direct_lasso_Cresc_n1e4/shards

The source cell is ``<root>/<source-cell>_n<n-obs>``, with ``<root>`` the parent of the output
cell (the layout of ``cluster/submit_campaign.sh``).  ``--n-shards`` must equal the source cell's
shard count: array task i rescored source shard i.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gclm.data.simulate import CChoice, draw_instance, estimation_volatility  # noqa: E402
from run_s1_shard import RAGGED, ebic_tag, select_graph, unpack_supports  # noqa: E402

SELECT = ("bic", "search")
#: fields of a shard that are not one entry per data set
NOT_PER_DATASET = ("config_json", "c_choice_names", "provenance_json")


def n_label(n_obs: str) -> str:
    """The sample-size suffix of a cell folder, as the submit script writes it."""
    return f"n{n_obs}"


def rescore_file(src: Path, gammas, select: str) -> dict:
    """All data sets of one source shard, rescored; returns the arrays to store."""
    d = np.load(src, allow_pickle=True)
    cfg = json.loads(str(d["config_json"]))
    names = [str(x) for x in d["c_choice_names"]]
    n_obs = cfg["n_obs"]
    n_obs = math.inf if n_obs in ("inf", "Infinity") or (isinstance(n_obs, float) and math.isinf(n_obs)) else float(n_obs)
    store: dict[str, list] = {}
    for i in range(len(d["p"])):
        p, k, ci, rep = (int(d[key][i]) for key in ("p", "k", "c_choice", "rep"))
        rng = np.random.default_rng([cfg["seed"], p, k, ci, rep])
        m_true, _, _, sigma_hat, scale = draw_instance(
            p, k, n_obs, list(CChoice)[ci], rng, metzler=cfg["metzler"],
            standardize=cfg["standardize"], return_scale=True)
        c_est = estimation_volatility(scale, cfg["c_scale"])
        n_l = len(d["lambdas"][i])
        supports = list(unpack_supports(d["supports_packed"][i], n_l, p))
        row = {"p": p, "k": k, "c_choice": ci, "rep": rep}
        # recomputed with the BIC penalty: must reproduce the stored scores (same data, same refit)
        plain = select_graph(supports, sigma_hat, c_est, n_obs, m_true, search=False)
        stored = np.asarray(d["bic_scores"][i], float)
        same = np.allclose(plain["bic_scores"], stored, rtol=1e-8, atol=1e-6, equal_nan=True)
        row["rebuilt_exact"] = bool(same)
        if not same:
            raise RuntimeError(f"{src.name} data set {i} (p={p} k={k} C={names[ci]} rep={rep}): the "
                               f"recomputed scores differ from the stored ones; the data set was "
                               f"not rebuilt as the source cell saw it")
        for gamma in gammas:
            tag = ebic_tag(gamma)
            out = select_graph(supports, sigma_hat, c_est, n_obs, m_true, search=select == "search",
                               ebic_gamma=gamma)
            for key, val in out.items():
                row[f"m_{tag}_{key[2:]}" if key.startswith("m_") else f"{tag}_{key}"] = val
        for key, val in row.items():
            store.setdefault(key, []).append(val)
    payload = {}
    for key, vals in store.items():
        if key.startswith(RAGGED):
            arr = np.empty(len(vals), dtype=object)
            for j, v in enumerate(vals):
                arr[j] = v
        else:
            arr = np.array(vals)
        payload[key] = arr
    payload["c_choice_names"] = d["c_choice_names"]
    payload["source_config_json"] = d["config_json"]
    return payload


def _rebuild(d, i: int, cfg: dict, n_obs):
    """Data set ``i`` of a source shard, rebuilt from its seed: (p, m_true, sigma_hat, c_est,
    supports of the path)."""
    p, k, ci, rep = (int(d[key][i]) for key in ("p", "k", "c_choice", "rep"))
    rng = np.random.default_rng([cfg["seed"], p, k, ci, rep])
    m_true, _, _, sigma_hat, scale = draw_instance(
        p, k, n_obs, list(CChoice)[ci], rng, metzler=cfg["metzler"],
        standardize=cfg["standardize"], return_scale=True)
    c_est = estimation_volatility(scale, cfg["c_scale"])
    supports = list(unpack_supports(d["supports_packed"][i], len(d["lambdas"][i]), p))
    return p, m_true, sigma_hat, c_est, supports


def refit_file(src: Path, refit: str, select: str, p_keep=()) -> dict:
    """Refit mode: a complete shard with the selection and the search redone with ``refit``."""
    d = np.load(src, allow_pickle=True)
    cfg = json.loads(str(d["config_json"]))
    n_obs = cfg["n_obs"]
    n_obs = math.inf if n_obs in ("inf", "Infinity") or (isinstance(n_obs, float) and math.isinf(n_obs)) else float(n_obs)
    keep = [i for i in range(len(d["p"])) if not p_keep or int(d["p"][i]) in p_keep]
    stale = ("bic_", "search_", "m_search_", "select_seconds", "rebuilt_exact", "ebic", "m_ebic")
    payload = {key: d[key][keep] for key in d.files
               if key not in NOT_PER_DATASET and not key.startswith(stale)}
    new: dict[str, list] = {}
    for i in keep:
        p, m_true, sigma_hat, c_est, supports = _rebuild(d, i, cfg, n_obs)
        # check that the data set was rebuilt as the source cell saw it: its own scores
        own = select_graph(supports, sigma_hat, c_est, n_obs, m_true, search=False,
                           refit=cfg.get("refit", "direct"))
        if not np.allclose(own["bic_scores"], np.asarray(d["bic_scores"][i], float),
                           rtol=1e-8, atol=1e-6, equal_nan=True):
            raise RuntimeError(f"{src.name} data set {i}: the recomputed scores differ from the "
                               f"stored ones; the data set was not rebuilt as the source cell saw it")
        out = select_graph(supports, sigma_hat, c_est, n_obs, m_true, search=select == "search",
                           refit=refit)
        out["rebuilt_exact"] = True
        for key, val in out.items():
            new.setdefault(key, []).append(val)
    for key, vals in new.items():
        if key.startswith(RAGGED):
            arr = np.empty(len(vals), dtype=object)
            for j, v in enumerate(vals):
                arr[j] = v
        else:
            arr = np.array(vals)
        payload[key] = arr
    payload["c_choice_names"] = d["c_choice_names"]
    cfg.update({"select": select, "refit": refit, "add_screen": None, "ebic_gammas": [],
                "p_values": sorted({int(d["p"][i]) for i in keep})})
    payload["config_json"] = json.dumps(cfg)
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-obs", required=True, help="the source cell's sample size label: 1000, 1e4 or inf")
    ap.add_argument("--source-cell", required=True, help="e.g. direct_lasso_Cresc (without _n<n>)")
    ap.add_argument("--refit", default="direct", choices=["direct", "loglik"],
                    help="loglik: redo the selection and the search with the maximised likelihood "
                         "behind the score; writes a complete cell (refit mode, wave 8)")
    ap.add_argument("--p", type=int, nargs="+", default=[],
                    help="refit mode: keep only the data sets with these p")
    ap.add_argument("--ebic-gamma", type=float, nargs="+", default=[1.0], metavar="G",
                    help="gammas of the eBIC term 4 G |E| log p (default 1)")
    ap.add_argument("--select", default="search", choices=list(SELECT),
                    help="selection only, or selection and the greedy search (default)")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if any(g <= 0 for g in args.ebic_gamma):
        ap.error("--ebic-gamma takes positive values")
    root = args.out_dir.resolve().parent.parent
    source = root / f"{args.source_cell}_{n_label(args.n_obs)}"
    if not (source / "shards").is_dir():
        ap.error(f"no such source cell: {source}")
    if (source / "n_shards").exists():                          # the submit script's bookkeeping
        n_src = int((source / "n_shards").read_text())
    else:
        n_src = len(list((source / "shards").glob("shard_*.npz")))
    if args.n_shards != n_src:
        ap.error(f"--n-shards {args.n_shards} but the source cell has {n_src} shards; use that number")
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in 0..{args.n_shards - 1}")
    src = source / "shards" / f"shard_{args.shard:04d}_of_{n_src:04d}.npz"
    if not src.exists():
        ap.error(f"the source shard {src} has not been written")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / src.name
    t0 = time.time()
    if args.refit != "direct":
        print(f"refitting {src} with refit={args.refit}, select={args.select}, p={args.p or 'all'} -> {out}",
              flush=True)
        payload = refit_file(src, args.refit, args.select, tuple(args.p))
        payload["source_json"] = json.dumps({"source_cell": args.source_cell, "source_shard": src.name})
    else:
        if args.p:
            ap.error("--p only in refit mode (--refit loglik): an overlay must line up with its source")
        print(f"rescoring {src} with gamma {args.ebic_gamma}, select={args.select} -> {out}", flush=True)
        payload = rescore_file(src, args.ebic_gamma, args.select)
        payload["config_json"] = json.dumps({"source_cell": args.source_cell, "n_obs": args.n_obs,
                                             "source_shard": src.name, "ebic_gammas": list(args.ebic_gamma),
                                             "ebic_form": "dettling", "select": args.select, "refit": "direct"})
    payload["provenance_json"] = json.dumps({
        "host": socket.gethostname(), "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "shard": args.shard, "n_shards": args.n_shards,
        "wall_seconds": time.time() - t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    np.savez_compressed(out, **payload)
    print(f"done: {len(payload['p'])} data sets in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
