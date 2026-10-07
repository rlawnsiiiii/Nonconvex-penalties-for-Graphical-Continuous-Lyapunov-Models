#!/usr/bin/env python3
"""S3b -- greedy BIC search with add / delete / reverse moves, started from the
lasso / MCP / SCAD paths or (pure search, Amendola et al. 2020) from random graphs.

Commands (numbers only; plots: plot_search.py; write-up: simulations/S3b_reversal_search.md):

  validate   exhaustive check on Example 2: every support with up to K edges is
             refitted and scored; the global optimum is compared with the truth and
             with what the greedy search reaches from each start
  example2   phase 1: Example 2 (path, 5-cycle fixed / random m15), the datasets of
             run_m0.py, all three losses, all methods
  random     phase 2/3: the Figure 5 random graphs of S3a (same seeds), one loss
  summarize  tables and CSVs from the row files

    python simulations/diagnostics/search_study.py validate
    python simulations/diagnostics/search_study.py example2 --reps 20 --workers 6
    python simulations/diagnostics/search_study.py random --loss direct --p 10 20 --n 1000 10000 inf --workers 6
    python simulations/diagnostics/search_study.py summarize

Methods per dataset, loss and score (BIC, or eBIC with gamma_e = 1 = the paper's
"increased penalty" adapted to directed graphs):
  path_<pen>_bic          the path's support at its BIC-selected lambda, no search
  path_<pen>_oracle       the path's support at its best-F1 lambda (oracle), no search
  search_<pen>            greedy search started from path_<pen>_bic          <- the three penalised methods
  search_<pen>_oracle     greedy search started from path_<pen>_oracle       (if --oracle-starts)
  search_pure             best of greedy searches from R random graphs + the empty graph  <- the pure method
  search_truth            greedy search started from the true support (oracle reference)
  objsearch_<pen>         (Example 2, direct loss, MCP/SCAD) reversal-only search on the
                          penalised objective at the BIC-selected lambda -- the ablation
"""

from __future__ import annotations

import os

# one BLAS thread per worker: with Anaconda's multithreaded BLAS, a pool of workers
# otherwise oversubscribes the machine many times over (load ~70 on 8 cores observed)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gclm.config import M0Config, S1Config, parse_n_obs  # noqa: E402
from gclm.data.examples import example2_cycle, example2_path  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance, sample_covariance, sample_data  # noqa: E402
from gclm.lyapunov import solve_lyapunov  # noqa: E402
from gclm.metrics import confusion, orientation_breakdown  # noqa: E402
from gclm.solvers.path import fit_path  # noqa: E402
from gclm.solvers.search import (Scorer, bic_along_path, greedy_search,  # noqa: E402
                                 multistart_search, random_support)

OUT = ROOT / "runs" / "s3b_search"
PENALTIES = ("lasso", "MCP", "SCAD")
C_NAMES = tuple(c.value for c in CChoice)
SCORES = {"bic": 0.0, "ebic": 1.0}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def metrics(support: np.ndarray, m_true: np.ndarray) -> dict:
    est = support.astype(float)
    cf = confusion(est, m_true)
    ob = orientation_breakdown(est, m_true)
    truth = (m_true != 0) & ~np.eye(m_true.shape[0], dtype=bool)
    return {"edge_list": edge_list(support), "true_edge_list": edge_list(truth),
            "f1": cf.f1, "precision": cf.precision, "recall": cf.tpr,
            "skeleton_f1": ob["skeleton_f1"], "exact": int(np.array_equal(support, truth)),
            "edges": int(support.sum()), "true_edges": int(truth.sum()),
            **{k: ob[k] for k in ("correct", "hedged", "reversed", "missed_single",
                                  "both", "half", "missed_double", "fp_single", "fp_double")}}


def edge_list(support: np.ndarray) -> str:
    """``"a>b;..."``: the selected edges a -> b (0-based), i.e. the entries M[b, a]."""
    return ";".join(f"{a}>{b}" for b, a in zip(*np.nonzero(support)))


def fit_supports(sigma_hat, loss, pen):
    """Supports (off-diagonal) of the path, increasing lambda, plus the estimates."""
    cfg = S1Config()
    p = sigma_hat.shape[0]
    kw = dict(n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, penalty=pen)
    if loss == "direct":
        kw.update(tol=cfg.tol, solver=cfg.solver, convention=cfg.convention)
    path = fit_path(sigma_hat, 2.0 * np.eye(p), loss=loss, **kw)
    off = ~np.eye(p, dtype=bool)
    return [(m != 0) & off for m in path.estimates], path


def best_f1_index(supports, m_true):
    f1 = [confusion(s.astype(float), m_true).f1 for s in supports]
    return int(np.nanargmax(np.nan_to_num(f1, nan=-1.0)))


def objective_reversal_search(sigma_hat, lam, m_start, pen, max_rounds=6):
    """The Example 2 ablation: reversal moves only, each re-solved with the
    penalised (MCP / SCAD) solver at ``lam``, accepted if the penalised objective
    decreases (next_steps/021026/files/m0_objective.py)."""
    from gclm.objective.direct import objective
    from gclm.objective.penalties import penalty_weights
    from gclm.solvers.proxgrad import solve_fista
    p = sigma_hat.shape[0]
    c = 2.0 * np.eye(p)
    w = penalty_weights(p, penalize_diagonal=False)
    off = ~np.eye(p, dtype=bool)
    gamma = 3.0 if pen == "MCP" else 3.7
    obj = lambda m: objective(m, sigma_hat, c, lam, w, pen, gamma, "textbook")
    snap = lambda m: np.where(np.abs(m) < 1e-12, 0.0, m)
    cur, fc = m_start, obj(m_start)
    for _ in range(max_rounds):
        best = None
        for i, j in zip(*np.nonzero((cur != 0) & off)):
            t = cur.copy()
            t[j, i] += t[i, j]
            t[i, j] = 0.0
            m = snap(solve_fista(sigma_hat, c, lam, weights=w, m_init=t, penalty=pen, gamma=gamma, tol=1e-10))
            f = obj(m)
            if f < fc - 1e-12 and (best is None or f < best[1]):
                best = (m, f)
        if best is None:
            break
        cur, fc = best
    return (cur != 0) & off


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #


def run_dataset(task):
    """All methods on one dataset for the requested losses and scores."""
    meta, m_true, sigma_hat, n, opts = task
    p = sigma_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    truth = (m_true != 0) & off
    rows = []
    rng = np.random.default_rng(opts["restart_seed"])
    random_starts = [random_support(p, rng, 0.3, opts["two_cycles"]) for _ in range(opts["restarts"])]
    for loss in opts["losses"]:
        screen = None if loss == "direct" else 2 * p
        paths = {}
        for pen in PENALTIES:
            pre = opts.get("pre_supports", {}).get((loss, pen))
            if pre is not None:
                paths[pen] = (pre, None)
            else:
                paths[pen] = fit_supports(sigma_hat, loss, pen)
        for score_name in opts["scores"]:
            g = SCORES[score_name]

            def emit(method, support, t0, evaluations=None, score=None, extra=None):
                rows.append({**meta, "n": n, "loss": loss, "score": score_name, "method": method,
                             "seconds": time.perf_counter() - t0,
                             "evaluations": evaluations, "bic_value": score,
                             **metrics(support, m_true), **(extra or {})})

            for pen in PENALTIES:
                supports, path = paths[pen]
                t0 = time.perf_counter()
                sc = Scorer(sigma_hat, 2.0 * np.eye(p), n, loss, g)
                ib, path_scores = bic_along_path(sigma_hat, 2.0 * np.eye(p), n, supports, loss, g, scorer=sc)
                emit(f"path_{pen}_bic", supports[ib], t0, sc.evaluations, path_scores[ib],
                     {"lambda_index": ib})
                io = best_f1_index(supports, m_true)
                emit(f"path_{pen}_oracle", supports[io], t0, None, None, {"lambda_index": io})
                t0 = time.perf_counter()
                r = greedy_search(sigma_hat, 2.0 * np.eye(p), n, supports[ib], loss=loss, ebic_gamma=g,
                                  add_screen=screen, allow_two_cycles=opts["two_cycles"])
                emit(f"search_{pen}", r.support, t0, r.evaluations, r.score, _move_counts(r))
                if opts["oracle_starts"]:
                    t0 = time.perf_counter()
                    r = greedy_search(sigma_hat, 2.0 * np.eye(p), n, supports[io], loss=loss, ebic_gamma=g,
                                      add_screen=screen, allow_two_cycles=opts["two_cycles"])
                    emit(f"search_{pen}_oracle", r.support, t0, r.evaluations, r.score, _move_counts(r))
                if opts["ablation"] and loss == "direct" and pen != "lasso" and path is not None:
                    t0 = time.perf_counter()
                    s_obj = objective_reversal_search(sigma_hat, float(path.lambdas[ib]),
                                                      path.estimates[ib], pen)
                    emit(f"objsearch_{pen}", s_obj, t0)
            t0 = time.perf_counter()
            starts = random_starts if loss == "direct" else random_starts[: opts.get("restarts_cov", len(random_starts))]
            best, results = multistart_search(sigma_hat, 2.0 * np.eye(p), n,
                                              starts + [np.zeros((p, p), bool)], loss=loss,
                                              ebic_gamma=g, add_screen=screen,
                                              allow_two_cycles=opts["two_cycles"])
            emit("search_pure", best.support, t0, best.evaluations, best.score,
                 {"restarts": len(results),
                  "restarts_exact": sum(np.array_equal(r.support, truth) for r in results),
                  "empty_start_exact": int(np.array_equal(results[-1].support, truth))})
            t0 = time.perf_counter()
            r = greedy_search(sigma_hat, 2.0 * np.eye(p), n, truth, loss=loss, ebic_gamma=g,
                              add_screen=screen, allow_two_cycles=opts["two_cycles"])
            emit("search_truth", r.support, t0, r.evaluations, r.score, _move_counts(r))
    return rows


def _move_counts(r):
    kinds = [m[0] for m in r.moves]
    return {"moves_add": kinds.count("add"), "moves_delete": kinds.count("delete"),
            "moves_reverse": kinds.count("reverse")}


def run_tasks(tasks, workers, out: Path, label: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}_rows.csv"
    t0 = time.time()
    writer, fh, done = None, path.open("w", newline=""), 0
    with ProcessPoolExecutor(workers) as ex:
        for rows in ex.map(run_dataset, tasks, chunksize=1):
            if writer is None:
                fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                                if k in rows[0] else 999)
                fields = [f for f in fields if f not in ("edge_list", "true_edge_list")] + ["edge_list", "true_edge_list"]
                writer = csv.DictWriter(fh, fieldnames=fields + ["restarts", "restarts_exact",
                                        "empty_start_exact", "moves_add", "moves_delete",
                                        "moves_reverse", "lambda_index"], extrasaction="ignore")
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {label}: {done}/{len(tasks)} datasets, {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def m0_datasets(reps, ns):
    """The datasets of run_m0.py (its random stream replayed), filtered to ``ns``."""
    cfg = M0Config()
    rng = np.random.default_rng(cfg.seed)
    c = 2.0 * np.eye(5)
    out = []
    for setting in ("path", "cycle_fixed", "cycle_random"):
        for n in cfg.sample_sizes:
            for rep in range(cfg.n_rep):
                m = (example2_path() if setting == "path" else example2_cycle() if setting == "cycle_fixed"
                     else example2_cycle(rng.uniform(*cfg.m15_range)))
                sig = solve_lyapunov(m, c)
                s = sig if np.isinf(n) else sample_covariance(sample_data(int(n), sig, rng))
                if any(n == x for x in ns) and rep < reps:
                    out.append(({"setting": setting, "p": 5, "rep": rep}, m, s, n))
                if np.isinf(n) and setting in ("path", "cycle_fixed"):
                    break
    return out


def pattern_supports(codes_per_lambda, p):
    """S3a pair codes (n_lambda x n_pairs) -> list of off-diagonal supports."""
    iu, ju = np.triu_indices(p, 1)
    out = []
    for codes in codes_per_lambda:
        s = np.zeros((p, p), bool)
        s[iu[(codes == 1) | (codes == 3)], ju[(codes == 1) | (codes == 3)]] = True
        s[ju[(codes == 2) | (codes == 3)], iu[(codes == 2) | (codes == 3)]] = True
        out.append(s)
    return out


def random_datasets(ps, ns, reps, standardize=True):
    cfg = S1Config()
    out = []
    for p in ps:
        for n in ns:
            for k in cfg.k_values:
                for c in C_NAMES:
                    for rep in range(reps):
                        rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c), rep])
                        m, _, _, s = draw_instance(p, k, n, CChoice(c), rng, metzler=cfg.metzler,
                                                   standardize=standardize)
                        out.append(({"p": p, "k": k, "c_choice": c, "rep": rep}, m, s, n))
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_example2(args):
    data = m0_datasets(args.reps, args.n)
    opts = dict(losses=args.losses, scores=args.scores, restarts=args.restarts, two_cycles=True,
                oracle_starts=True, ablation=True, restarts_cov=args.restarts_cov)
    tasks = []
    for i, (meta, m, s, n) in enumerate(data):
        tasks.append((meta, m, s, n, {**opts, "restart_seed": [20261003, i]}))
    tasks.sort(key=lambda t: 0 if t[3] == math.inf else -t[3])     # large n first: those are fast
    print(f"example2: {len(tasks)} datasets, losses {args.losses}, scores {args.scores}")
    run_tasks(tasks, args.workers, args.out, "example2")


def cmd_random(args):
    raw = ROOT / "runs" / "s3a_bidirectional" / "raw"
    data = random_datasets(args.p, args.n, args.reps, standardize=not args.raw)
    opts = dict(losses=[args.loss], scores=args.scores, two_cycles=True,
                oracle_starts=args.oracle_starts, ablation=False)
    cache = {}
    tasks = []
    for i, (meta, m, s, n) in enumerate(data):
        p = meta["p"]
        pre = {}
        if args.loss == "direct" and not np.isinf(n) and not args.raw:     # S3a paths are standardised
            for pen in PENALTIES:
                f = raw / f"p{p}_n{int(n)}_{pen}.npz"
                if f.exists():
                    if f not in cache:
                        cache[f] = np.load(f)
                    d = cache[f]
                    hit = np.nonzero((d["k"] == meta["k"]) & (d["c_choice"] == C_NAMES.index(meta["c_choice"]))
                                     & (d["rep"] == meta["rep"]))[0]
                    if len(hit):
                        pre[("direct", pen)] = pattern_supports(d["patterns"][hit[0]], p)
        restarts = args.restarts if args.restarts is not None else (10 if p <= 10 else 5)
        tasks.append((meta, m, s, n, {**opts, "restarts": restarts, "pre_supports": pre,
                                      "restart_seed": [20261003, p, meta["k"],
                                                       C_NAMES.index(meta["c_choice"]), meta["rep"],
                                                       0 if np.isinf(n) else int(n)]}))
    # cheapest first, so partial results arrive early
    tasks.sort(key=lambda t: (t[2].shape[0], 0 if t[4]["pre_supports"] else 1))
    label = f"random_{args.loss}" + (f"_{args.label}" if args.label else "")
    print(f"random: {len(tasks)} datasets, loss {args.loss}, scores {args.scores}")
    run_tasks(tasks, args.workers, args.out, label)


def cmd_validate(args):
    """Exhaustive check on Example 2 (direct loss; up to --max-edges edges)."""
    from gclm.solvers.search import bic as bic_score, make_refit
    data = [d for d in m0_datasets(args.reps, [math.inf, 1e4]) if d[0]["setting"] == "cycle_fixed"]
    out = []
    for meta, m, s, n in data[: args.datasets]:
        p = 5
        off_idx = [(i, j) for i in range(p) for j in range(p) if i != j]
        truth = (m != 0) & ~np.eye(p, dtype=bool)
        for loss in args.losses:
            refit = make_refit(s, 2.0 * np.eye(p), loss)
            t0 = time.time()
            best = (math.inf, None)
            count = 0
            for k in range(args.max_edges + 1):
                for comb in itertools.combinations(range(len(off_idx)), k):
                    sup = np.zeros((p, p), bool)
                    for q in comb:
                        sup[off_idx[q]] = True
                    sc = bic_score(refit.fit(sup), s, 2.0 * np.eye(p), n, k)
                    count += 1
                    if sc < best[0]:
                        best = (sc, sup)
            truth_sc = bic_score(refit.fit(truth), s, 2.0 * np.eye(p), n, int(truth.sum()))
            starts = {"empty": np.zeros((p, p), bool)}
            for pen in PENALTIES:
                sups, _ = fit_supports(s, loss, pen)
                ib, _ = bic_along_path(s, 2.0 * np.eye(p), n, sups, loss)
                starts[f"{pen}_bic"] = sups[ib]
            rng = np.random.default_rng(7)
            for r in range(args.random_starts):
                starts[f"random{r}"] = random_support(p, rng, 0.3)
            reached = {}
            for name, st in starts.items():
                res = greedy_search(s, 2.0 * np.eye(p), n, st, loss=loss)
                reached[name] = bool(np.array_equal(res.support, best[1]))
            row = {"n": n, "rep": meta["rep"], "loss": loss, "supports_scored": count,
                   "seconds": round(time.time() - t0, 1),
                   "global_optimum_is_truth": bool(np.array_equal(best[1], truth)),
                   "global_optimum_edges": int(best[1].sum()), "global_bic": best[0], "truth_bic": truth_sc,
                   "reached": reached}
            print(json.dumps(row, default=float))
            out.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validate.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out / 'validate.json'}")


MAIN_METHODS = ("path_lasso_oracle", "path_lasso_bic", "path_MCP_bic", "path_SCAD_bic",
                "search_lasso", "search_MCP", "search_SCAD", "search_pure", "search_truth")


def cmd_summarize(args):
    """Means per (group, n, loss, score, method) and paired differences against
    plain lasso at its BIC lambda and at its oracle best-F1 lambda (Dettling's max_f1)."""
    for f in sorted(args.out.glob("*_rows.csv")):
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        key = lambda r: tuple(r.get(k, "") for k in ("setting", "p", "k", "c_choice", "rep", "n", "loss", "score"))
        group = lambda r: r["setting"] if r.get("setting") else f"p={r['p']}"
        nkey = lambda n: math.inf if n in ("inf", "Infinity") else float(n)
        by = {}
        for r in rows:
            by[(key(r), r["method"])] = r
        cells = sorted({(group(r), r["n"], r["loss"], r["score"]) for r in rows},
                       key=lambda c: (c[0], nkey(c[1]), c[2], c[3]))
        methods = [m for m in MAIN_METHODS if any(r["method"] == m for r in rows)]
        methods += sorted({r["method"] for r in rows} - set(methods))
        out_rows = []
        for g, n, loss, score in cells:
            sub_rows = [r for r in rows if group(r) == g and r["n"] == n and r["loss"] == loss and r["score"] == score]
            keys = sorted({key(r) for r in sub_rows})
            for m in methods:
                rr = [by[(k, m)] for k in keys if (k, m) in by]
                if not rr:
                    continue
                vals = lambda col: np.array([float(r[col]) for r in rr if r.get(col) not in ("", None)])
                row = {"group": g, "n": n, "loss": loss, "score": score, "method": m, "datasets": len(rr),
                       "f1": vals("f1").mean(), "exact": vals("exact").mean(),
                       "skeleton_f1": vals("skeleton_f1").mean(), "edges": vals("edges").mean(),
                       "reversed": vals("reversed").mean(), "hedged": vals("hedged").mean(),
                       "correct": vals("correct").mean(), "both_2cycles": vals("both").mean(),
                       "seconds": vals("seconds").mean(),
                       "evaluations": vals("evaluations").mean() if len(vals("evaluations")) else float("nan")}
                for base in ("path_lasso_bic", "path_lasso_oracle"):
                    d = np.array([float(by[(k, m)]["f1"]) - float(by[(k, base)]["f1"]) for k in keys
                                  if (k, m) in by and (k, base) in by])
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                    row[f"f1_minus_{base}"] = d.mean() if len(d) else float("nan")
                    row[f"z_vs_{base}"] = d.mean() / se if se and se > 0 else float("nan")
                out_rows.append(row)
        target = args.out / f.name.replace("_rows.csv", "_summary.csv")
        with target.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n=== {f.name} ===")
        for g, n, loss, score in cells:
            print(f"-- {g}, n={n}, {loss}, {score}")
            for r in out_rows:
                if (r["group"], r["n"], r["loss"], r["score"]) == (g, n, loss, score) and r["method"] in MAIN_METHODS:
                    print(f"   {r['method']:<20} F1 {r['f1']:.3f}  exact {r['exact']:.2f}  skel {r['skeleton_f1']:.3f}"
                          f"  rev {r['reversed']:.2f}  hedged {r['hedged']:.2f}  edges {r['edges']:.1f}"
                          f"  vs plain lasso(BIC) {r['f1_minus_path_lasso_bic']:+.3f} (z {r['z_vs_path_lasso_bic']:+.1f})"
                          f"  vs Dettling oracle {r['f1_minus_path_lasso_oracle']:+.3f}  [{r['datasets']}]")
        print(f"wrote {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--max-edges", type=int, default=7)
    v.add_argument("--losses", nargs="+", default=["direct"])
    v.add_argument("--reps", type=int, default=2)
    v.add_argument("--datasets", type=int, default=3)
    v.add_argument("--random-starts", type=int, default=10)
    e = sub.add_parser("example2")
    e.add_argument("--reps", type=int, default=20)
    e.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000, 100_000, math.inf])
    e.add_argument("--losses", nargs="+", default=["direct", "loglik", "frobenius"])
    e.add_argument("--scores", nargs="+", default=["bic", "ebic"])
    e.add_argument("--restarts", type=int, default=20)
    e.add_argument("--restarts-cov", type=int, default=10)
    e.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("random")
    r.add_argument("--loss", default="direct", choices=["direct", "loglik", "frobenius"])
    r.add_argument("--p", type=int, nargs="+", default=[10, 20])
    r.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000, math.inf])
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--scores", nargs="+", default=["bic"])
    r.add_argument("--restarts", type=int, default=None)
    r.add_argument("--oracle-starts", action="store_true")
    r.add_argument("--label", default="")
    r.add_argument("--raw", action="store_true",
                   help="raw covariance (C = 2I correctly specified) instead of the standardised Figure 5 input")
    r.add_argument("--workers", type=int, default=6)
    sm = sub.add_parser("summarize")
    for sp in (v, e, r, sm):
        sp.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    {"validate": cmd_validate, "example2": cmd_example2, "random": cmd_random,
     "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
