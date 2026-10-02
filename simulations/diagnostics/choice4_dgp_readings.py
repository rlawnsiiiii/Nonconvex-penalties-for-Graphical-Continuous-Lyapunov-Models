"""Is the C_Random_Full gap Monte Carlo noise, or the DGP?  p=10, 400 datasets per config."""
import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/Users/joonkim/Desktop/MastersThesis/repo/src")
from gclm.config import S1Config
from gclm.data.simulate import (
    sample_drift,
    sample_volatility,
    sample_data,
    sample_covariance,
    CChoice,
)
from gclm.lyapunov import solve_lyapunov
from gclm.solvers.path import lasso_path
from gclm.metrics import evaluate_path

def c_full(p, rng, variant):
    eps = rng.normal(size=(p, p))
    if variant == "literal":          # ours: omega full matrix, C_ij = w_ij e_ij + w_ji e_ji
        om = rng.binomial(1, min(2/p, 1), size=(p, p)); w = om*eps
        np.fill_diagonal(w, 0); c = w + w.T
    elif variant == "upper":          # omega and eps only for i<j: C_ij = C_ji = w_ij e_ij
        om = rng.binomial(1, min(2/p, 1), size=(p, p)); w = np.triu(om*eps, 1); c = w + w.T
    elif variant == "sym_omega":      # one omega per pair: C_ij = w_ij (e_ij + e_ji)
        om = np.triu(rng.binomial(1, min(2/p, 1), size=(p, p)), 1); om = om + om.T
        c = om*(eps + eps.T); np.fill_diagonal(c, 0)
    np.fill_diagonal(c, np.abs(c).sum(axis=1) + np.abs(np.diag(eps)) + 0.5)
    return c

def one(args):
    p, k, rep, variant, seed, cchoice = args
    cfg = S1Config()
    rng = np.random.default_rng([seed, p, k, rep])
    m = sample_drift(p, k/p, rng)
    c = c_full(p, rng, variant) if cchoice == "full" else sample_volatility(p, CChoice.ID, rng)
    S = sample_covariance(sample_data(cfg.n_obs, solve_lyapunov(m, c), rng))
    s = np.sqrt(np.diag(S)); S = S/np.outer(s, s)
    ev = evaluate_path(lasso_path(S, 2*np.eye(p), n_lambda=100, tol=cfg.tol).estimates, m)
    nnz_c = int(np.sum(c[~np.eye(p, dtype=bool)] != 0)) / p
    return [ev["max_acc"], ev["max_f1"], ev["auc"], ev["aupr"], nnz_c]

if __name__ == "__main__":
    p, reps = 10, 100
    configs = [("C_ID  (control)", "literal", 111, "id"), ("C_ID  (control)", "literal", 222, "id"),
               ("full literal", "literal", 111, "full"), ("full literal", "literal", 222, "full"),
               ("full literal", "literal", 333, "full"),
               ("full upper (i<j)", "upper", 111, "full"), ("full sym_omega", "sym_omega", 111, "full")]
    out = {}
    with ProcessPoolExecutor(8) as ex:
        for label, var, seed, cc in configs:
            tasks = [(p, k, r, var, seed, cc) for k in (1,2,3,4) for r in range(reps)]
            res = np.array(list(ex.map(one, tasks, chunksize=8)))
            mu, se = res.mean(0), res.std(0, ddof=1)/np.sqrt(len(res))
            out[f"{label} seed={seed}"] = (mu.tolist(), se.tolist())
            print(f"{label:<18} seed={seed}: " + "  ".join(
                f"{n}={mu[i]:.3f}±{se[i]:.3f}" for i, n in enumerate(("max_acc","max_f1","auc","aupr"))) +
                f"   C offdiag nnz/row={mu[4]:.2f}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
