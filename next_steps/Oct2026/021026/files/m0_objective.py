"""Example 2 cycle (IC ~ 3): does the MCP *objective* prefer the truth at large n, and can a
non-oracle reversal search reach it?  Same datasets as m0_penalties.py (cycle_fixed, reps < 20).
At every 4th lambda of the MCP continuation path:
  cont   : the continuation solution (what the S1b pipeline reports)
  oracle : MCP re-solved from M* (oracle start; the truth's basin)
  search : greedy reversal search from cont on the MCP objective (no oracle): try reversing every
           selected entry, re-solve, accept the best improvement, repeat."""
import sys, csv, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
SCR = str(__import__('pathlib').Path(__file__).resolve().parent)   # this folder
sys.path.insert(0, SCR); sys.path.insert(0, "/Users/joonkim/Desktop/MastersThesis/repo/src")
from m0_penalties import SETTINGS, cfg, C                        # same constants as the sweep
from gclm.data.simulate import sample_covariance, sample_data
from gclm.data.examples import example2_cycle, example2_path
from gclm.lyapunov import solve_lyapunov
from gclm.solvers.path import lasso_path
from gclm.solvers.proxgrad import solve_fista
from gclm.objective.direct import objective
from gclm.objective.penalties import penalty_weights
from gclm.metrics import confusion

G, TOL = 3.0, 1e-11
W = penalty_weights(5, penalize_diagonal=False)
OFF = ~np.eye(5, dtype=bool)

def snap(m):
    m = m.copy(); m[np.abs(m) < 1e-12] = 0.0; return m

def search(m0, s, lam, F):
    cur, fc = m0, F(m0)
    for _ in range(6):
        best = None
        for i, j in zip(*np.nonzero((cur != 0) & OFF)):
            t = cur.copy(); t[j, i] += t[i, j]; t[i, j] = 0.0
            m = snap(solve_fista(s, C, lam, weights=W, m_init=t, penalty="MCP", gamma=G, tol=TOL))
            f = F(m)
            if f < fc - 1e-12 and (best is None or f < best[1]):
                best = (m, f)
        if best is None:
            break
        cur, fc = best
    return cur

def run(task):
    n, rep, m_star, s = task
    path = lasso_path(s, C, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, tol=TOL, penalty="MCP", gamma=G)
    out = []
    for idx in range(0, len(path.lambdas), 4):
        lam = float(path.lambdas[idx])
        F = lambda m: objective(m, s, C, lam, W, "MCP", G, "textbook")
        cont = path.estimates[idx]
        orac = snap(solve_fista(s, C, lam, weights=W, m_init=m_star, penalty="MCP", gamma=G, tol=TOL))
        srch = search(cont, s, lam, F)
        row = {"n": n, "rep": rep, "idx": idx, "lam": lam}
        for name, m in (("cont", cont), ("oracle", orac), ("search", srch)):
            row[f"F_{name}"] = F(m); row[f"f1_{name}"] = confusion(m, m_star).f1
        out.append(row)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(cfg.seed); tasks = []
    for setting in SETTINGS:                                       # replay the sweep's stream
        for n in cfg.sample_sizes:
            for rep in range(100):
                m_star = (example2_path() if setting == "path" else example2_cycle() if setting == "cycle_fixed"
                          else example2_cycle(rng.uniform(*cfg.m15_range)))
                sigma_true = solve_lyapunov(m_star, C)
                s = sigma_true if np.isinf(n) else sample_covariance(sample_data(int(n), sigma_true, rng))
                if setting == "cycle_fixed" and n in (1e3, 1e4, 1e5, float("inf")) and rep < 20:
                    tasks.append((n, rep, m_star, s))
                if np.isinf(n) and setting in ("path", "cycle_fixed"):
                    break
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(run, tasks) for r in rs]
    with open(f"{SCR}/m0_objective.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(tasks)} datasets, {time.time() - t0:.0f}s")
