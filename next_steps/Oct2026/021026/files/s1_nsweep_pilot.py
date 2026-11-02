"""Figure 5 pipeline (S1 DGP, standardised, C = 2I assumed) at larger n, p = 10, all four C choices,
k = 1..4, reps < 25, lasso / MCP / SCAD.  Same seeds as run_s1_shard.py, so M* and C are the S1 ones
at every n (they are drawn before the data); n = 1000 for these datasets already exists in the repo.
A validation block refits the lasso at n = 1000, reps < 5, to compare with the committed run."""
import sys, csv, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "/Users/joonkim/Desktop/MastersThesis/repo/src")
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance
from gclm.solvers.path import fit_path
from gclm.metrics import evaluate_path, orientation_breakdown

SCR = str(__import__('pathlib').Path(__file__).resolve().parent)   # this folder
P = 10

def run(task):
    c, k, rep, n, pens = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, P, k, list(CChoice).index(CChoice(c)), rep])
    m_true, _, _, s = draw_instance(P, k, n, CChoice(c), rng, metzler=cfg.metzler, standardize=cfg.standardize)
    out = []
    for pen in pens:
        path = fit_path(s, 2.0 * np.eye(P), loss="direct", n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio,
                        penalty=pen, tol=cfg.tol, solver=cfg.solver, convention=cfg.convention)
        ev = evaluate_path(path.estimates, m_true)
        b = orientation_breakdown(path.estimates[int(np.nanargmax(ev["f1"]))], m_true)
        out.append({"c": c, "k": k, "rep": rep, "n": n, "pen": pen, "max_f1": ev["max_f1"], "auc": ev["auc"],
                    "aupr": ev["aupr"], "exact": int(ev["max_f1"] > 1 - 1e-12),
                    **{key: b[key] for key in ("skeleton_f1", "correct", "reversed", "hedged", "both", "half")}})
    return out

if __name__ == "__main__":
    cs = [ch.value for ch in CChoice]
    tasks = [(c, k, r, 1000.0, ("lasso",)) for c in cs for k in (1, 2, 3, 4) for r in range(5)]
    tasks += [(c, k, r, n, ("lasso", "MCP", "SCAD")) for n in (float("inf"), 1e5, 1e4)
              for c in cs for k in (1, 2, 3, 4) for r in range(25)]
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex, open(f"{SCR}/s1_nsweep_p10.csv", "w", newline="") as fh:
        w = None
        for i, rows in enumerate(ex.map(run, tasks, chunksize=2)):
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
            w.writerows(rows); fh.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} datasets, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(tasks)} datasets, {time.time() - t0:.0f}s")
