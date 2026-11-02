"""Lasso at n = inf on the S1 DGP (p = 10, C_ID, same M* as S1 reps < 25): standardised (Figure 5
pipeline, C = 2I assumed on the correlation scale -> misspecified) vs raw (C = 2I exactly right)."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "/Users/joonkim/Desktop/MastersThesis/repo/src")
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance
from gclm.solvers.path import lasso_path
from gclm.metrics import evaluate_path, orientation_breakdown

def run(task):
    k, rep, std = task
    cfg = S1Config(); p = 10
    rng = np.random.default_rng([cfg.seed, p, k, list(CChoice).index(CChoice.ID), rep])
    m_true, _, _, s = draw_instance(p, k, float("inf"), CChoice.ID, rng, standardize=std)
    path = lasso_path(s, 2.0 * np.eye(p), n_lambda=100, ratio=1e-4, tol=1e-8)
    ev = evaluate_path(path.estimates, m_true)
    b = orientation_breakdown(path.estimates[int(np.nanargmax(ev["f1"]))], m_true)
    return std, ev["max_f1"], ev["auc"], b["skeleton_f1"], b["reversed"], b["hedged"]

if __name__ == "__main__":
    tasks = [(k, r, std) for std in (True, False) for k in (1, 2, 3, 4) for r in range(25)]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, tasks))
    for std in (True, False):
        a = np.array([r[1:] for r in res if r[0] == std])
        print(f"{'standardised (Figure 5)' if std else 'raw scale (C correct)':<24} n=inf: max_f1 {a[:,0].mean():.3f}, "
              f"auc {a[:,1].mean():.3f}, skeleton F1 {a[:,2].mean():.3f}, reversed {a[:,3].mean():.2f}, hedged {a[:,4].mean():.2f}, "
              f"exact recovery {np.mean(a[:,0] > 1 - 1e-12):.2f}")
