"""Check of three ways to score a graph in the greedy search (9 October 2026), before any cluster
time.  Every score is a loss term plus the BIC penalty (p + |S|) log n (docs/SEARCH.md §2):

    default             M_S by least squares on the direct loss; loss term n[log det Σ_S + tr(Σ_S⁻¹Σ̂)]
    Améndola/Dettling   M_S by maximising the likelihood; the same loss term (their eq. 16 / eq. 6.1)
    direct-loss score   M_S by least squares; loss term N log(RSS_S / N), N = p(p + 1)/2

p = 10, k = 1..4, the four true C, two replicates (32 graphs) at n = 10³ and 10⁴, C = 2I in the fit.
Per graph and score: the greedy search from the true graph, the graph the score selects on the lasso
path of the direct loss, and the greedy search from that graph.

Resumable and parallel: one JSON line per (n, graph) in score_check_rows.jsonl, written as soon as
it is done; a rerun skips the finished ones.  The summary goes to score_check.txt.
Run from the repository root:
    OMP_NUM_THREADS=1 nohup python3 next_steps/091026/score_check.py > next_steps/091026/score_check.log 2>&1 &
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "simulations")
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.solvers.path import lasso_path
from gclm.solvers.search import Scorer, bic_along_path, greedy_search

HERE = Path("next_steps/091026")
ROWS, TXT = HERE / "score_check_rows.jsonl", HERE / "score_check.txt"
P, NS, KS, REPS, WORKERS = 10, (1000, 10_000), (1, 2, 3, 4), (0, 1), 4
VARIANTS = (("default", "least-squares fit, likelihood loss term", "direct", "likelihood"),
            ("amendola", "likelihood fit, likelihood loss term (Améndola, Dettling)", "loglik", "likelihood"),
            ("direct", "least-squares fit, direct-loss term N log(RSS/N)", "direct", "direct"))


def f1(s, truth):
    tp = int((s & truth).sum()); fp = int((s & ~truth).sum()); fn = int((truth & ~s).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn), int(s.sum())


def one(task):
    n, k, ci, rep = task
    cfg = S1Config(n_obs=n)
    rng = np.random.default_rng([cfg.seed, P, k, ci, rep])
    m_true, _, _, sh, scale = draw_instance(P, k, n, list(CChoice)[ci], rng, metzler=cfg.metzler,
                                            standardize=cfg.standardize, return_scale=True)
    c = estimation_volatility(scale, "identity")
    eye = np.eye(P, dtype=bool)
    truth = (m_true != 0) & ~eye
    sup = [(m != 0) & ~eye for m in lasso_path(sh, c, n_lambda=100, tol=cfg.tol).estimates]
    out = {"n": n, "k": k, "c": ci, "rep": rep, "true_edges": int(truth.sum())}
    for tag, _, refit, score in VARIANTS:
        t0 = time.perf_counter()
        sc = Scorer(sh, c, n, refit, score=score)
        a = greedy_search(sh, c, n, truth, scorer=sc, loss=refit, max_steps=P * (P - 1))
        ib, _ = bic_along_path(sh, c, n, sup, scorer=sc)
        b = greedy_search(sh, c, n, sup[ib], scorer=sc, loss=refit, max_steps=P * (P - 1))
        for key, s in (("truth", a.support), ("selected", sup[ib]), ("path", b.support)):
            out[f"{tag}_{key}_f1"], out[f"{tag}_{key}_edges"] = f1(s, truth)
        out[f"{tag}_seconds"] = round(time.perf_counter() - t0, 2)
    return out


def summary(rows):
    lines = [f"Scores of a graph in the greedy search, p = {P}, C = 2I, lasso path of the direct loss "
             f"(next_steps/091026/score_check.py)", ""]
    for n in NS:
        r = [x for x in rows if x["n"] == n]
        if not r:
            continue
        lines.append(f"n = {n}: {len(r)} graphs, mean true edges {np.mean([x['true_edges'] for x in r]):.1f}")
        lines.append(f"  {'score (all with the BIC penalty)':58s} {'from the truth':>17s} {'selected on path':>17s}"
                     f" {'search from it':>17s} {'s/graph':>8s}")
        for tag, label, _, _ in VARIANTS:
            cols = []
            for key in ("truth", "selected", "path"):
                cols.append(f"F1 {np.mean([x[f'{tag}_{key}_f1'] for x in r]):.3f} "
                            f"{np.mean([x[f'{tag}_{key}_edges'] for x in r]):5.1f}e")
            lines.append(f"  {label:58s} {cols[0]:>17s} {cols[1]:>17s} {cols[2]:>17s} "
                         f"{np.mean([x[f'{tag}_seconds'] for x in r]):8.1f}")
        lines.append("")
    lines.append("e = mean number of edges of the graph")
    TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    done = [json.loads(s) for s in ROWS.read_text().splitlines()] if ROWS.exists() else []
    have = {(x["n"], x["k"], x["c"], x["rep"]) for x in done}
    tasks = [(n, k, ci, rep) for n in NS for k in KS for ci in range(len(CChoice)) for rep in REPS
             if (n, k, ci, rep) not in have]
    print(f"{len(have)} done, {len(tasks)} to go, {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(WORKERS) as pool, ROWS.open("a") as fh:
        futures = {pool.submit(one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            done.append(row)
            print(f"{i}/{len(tasks)} {futures[fut]} default {row['default_seconds']:.0f} s, "
                  f"amendola {row['amendola_seconds']:.0f} s", flush=True)
    summary(done)
