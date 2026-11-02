"""M0 (Dettling Example 2) with lasso / MCP / SCAD on identical datasets, replaying run_m0.py's
random stream, plus the population irrepresentability constant of each target."""
import sys, time, csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
REPO = "/Users/joonkim/Desktop/MastersThesis/repo"
sys.path.insert(0, REPO + "/src")
from gclm.config import M0Config
from gclm.data.simulate import sample_covariance, sample_data
from gclm.data.examples import example2_cycle, example2_path
from gclm.solvers.path import lasso_path
from gclm.lyapunov import solve_lyapunov, design_matrix, vec
from gclm.metrics import evaluate_path

SETTINGS = ("path", "cycle_fixed", "cycle_random")
PENS = (("lasso", None), ("MCP", 3.0), ("SCAD", 3.7))
cfg = M0Config(); C = 2.0 * np.eye(5)

def col_index(i, j, p=5):
    e = np.zeros((p, p)); e[i, j] = 1.0
    return int(np.argmax(vec(e)))

def irrep(m_star):
    """max_{off-diag j not in S} |Gamma_{jT} Gamma_TT^-1 z_T|, T = true edges + diagonal, z = sign on
    edges, 0 on the (unpenalised) diagonal. < 1: lasso sign-consistent; > 1: fails."""
    p = m_star.shape[0]; sigma = solve_lyapunov(m_star, C)
    A = design_matrix(sigma); G = A.T @ A
    T = [col_index(i, j) for i in range(p) for j in range(p) if i == j or m_star[i, j] != 0]
    z = np.array([0.0 if i == j else np.sign(m_star[i, j]) for i in range(p) for j in range(p)
                  if i == j or m_star[i, j] != 0])
    Tc = [col_index(i, j) for i in range(p) for j in range(p) if i != j and m_star[i, j] == 0]
    v = G[np.ix_(Tc, T)] @ np.linalg.solve(G[np.ix_(T, T)], z)
    return float(np.max(np.abs(v)))

def fit(task):
    setting, n, rep, m_star, sigma_hat = task
    out = []
    for pen, g in PENS:
        kw = {} if pen == "lasso" else {"penalty": pen, "gamma": g}
        path = lasso_path(sigma_hat, C, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio, tol=1e-11, **kw)
        ev = evaluate_path(path.estimates, m_star)
        best = path.estimates[int(np.nanargmax(ev["f1"]))]
        out.append({"setting": setting, "n": n, "rep": rep, "pen": pen, "m15": m_star[0, 4],
                    "max_f1": ev["max_f1"], "auc": ev["auc"], "aupr": ev["aupr"],
                    "has_5to1": int(best[0, 4] != 0), "has_1to5": int(best[4, 0] != 0)})
    return out

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.n_rep
    rng = np.random.default_rng(cfg.seed); tasks = []
    for setting in SETTINGS:                         # identical draw order to run_m0.py
        for n in cfg.sample_sizes:
            for rep in range(reps):
                m_star = (example2_path() if setting == "path" else example2_cycle() if setting == "cycle_fixed"
                          else example2_cycle(rng.uniform(*M0Config.m15_range)))
                sigma_true = solve_lyapunov(m_star, C)
                sigma_hat = sigma_true if np.isinf(n) else sample_covariance(sample_data(int(n), sigma_true, rng))
                tasks.append((setting, n, rep, m_star, sigma_hat))
                if np.isinf(n) and setting in ("path", "cycle_fixed"):
                    break
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(fit, tasks, chunksize=4) for r in rs]
    print(f"{len(tasks)} datasets, {time.time() - t0:.0f}s")
    out = str(__import__('pathlib').Path(__file__).resolve().parent) + f"/m0_penalties_reps{reps}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
