"""Second timing pilot for wave 7 (10 October 2026), after Joon asked for p up to 30, the sample
sizes n = 1000 and 1e4 in every part, and 100 random starting graphs for the search with the
likelihood refit.  One data set per measurement (k = 2, true C = 2I, n = 1e4, replicate 0, the
campaign's generator).  Output: time_wave7_p30.txt, one line per measurement, appended as it runs.

  A <c_scale>  p = 30: the log-likelihood lasso path; the score with the likelihood refit along it
               (time, supports); the selection and the search with the least-squares refit on it.
               The other estimators are scaled from the lasso by their p = 20 ratios (time_wave7.txt).
  B <p>        rescaled C: the greedy search with the likelihood refit (20 add moves screened per
               step) from the graph selected on the direct-loss lasso path and from random sparse
               starts.  p = 10: full searches.  p = 20, 30: two steps, extrapolated with the number
               of moves the least-squares search makes from the same start.

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 next_steps/091026/time_wave7_p30.py A identity
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "simulations")
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.objective import covariance as cov
from gclm.solvers.path import covloss_path, lambda_grid, lasso_path
from gclm.solvers.search import Scorer, bic_along_path, greedy_search, random_support

OUT = Path("next_steps/091026/time_wave7_p30.txt")


def log(msg):
    with OUT.open("a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def data(p, c_scale, n=10_000):
    cfg = S1Config(n_obs=n, c_scale=c_scale)
    rng = np.random.default_rng([cfg.seed, p, 2, 0, 0])
    m_true, _, _, sh, scale = draw_instance(p, 2, n, list(CChoice)[0], rng, metzler=cfg.metzler,
                                            standardize=cfg.standardize, return_scale=True)
    return sh, estimation_volatility(scale, c_scale), cfg


def supports(path, p):
    eye = np.eye(p, dtype=bool)
    return [(m != 0) & ~eye for m in path.estimates]


def part_a(c_scale, p=30):
    sh, c, cfg = data(p, c_scale)
    lams = lambda_grid(cov.lambda_max(sh, c, "loglik"), 100, 1e-4)
    t0 = time.perf_counter()
    sup = supports(covloss_path(sh, c, "loglik", lambdas=lams, tol=cfg.tol, penalty="lasso"), p)
    log(f"A p={p} {c_scale:8s} log-likelihood lasso path: {time.perf_counter() - t0:8.1f} s")
    t0 = time.perf_counter()
    sc = Scorer(sh, c, cfg.n_obs, "loglik")
    ib, _ = bic_along_path(sh, c, cfg.n_obs, sup, scorer=sc)
    log(f"A p={p} {c_scale:8s} score along that path, likelihood refit: {time.perf_counter() - t0:8.1f} s "
        f"({sc.evaluations} supports, {int(sup[ib].sum())} edges selected)")
    t0 = time.perf_counter()
    sl = Scorer(sh, c, cfg.n_obs, "direct")
    jb, _ = bic_along_path(sh, c, cfg.n_obs, sup, scorer=sl)
    res = greedy_search(sh, c, cfg.n_obs, sup[jb], scorer=sl, loss="direct", max_steps=p * (p - 1))
    log(f"A p={p} {c_scale:8s} selection and search on that path, least-squares refit: "
        f"{time.perf_counter() - t0:8.1f} s ({len(res.moves)} moves)")


def part_b(p, c_scale="variance"):
    sh, c, cfg = data(p, c_scale)
    sup = supports(lasso_path(sh, c, n_lambda=100, tol=cfg.tol), p)
    jb, _ = bic_along_path(sh, c, cfg.n_obs, sup, scorer=Scorer(sh, c, cfg.n_obs, "direct"))
    rng = np.random.default_rng([20261010, p])
    starts = [("selected graph", sup[jb])] + [(f"random start {i + 1}", random_support(p, rng, 0.3))
                                              for i in range(3 if p == 10 else 1)]
    for name, s0 in starts:
        ls = greedy_search(sh, c, cfg.n_obs, s0, loss="direct", max_steps=p * (p - 1))
        steps = p * (p - 1) if p == 10 else 2
        t0 = time.perf_counter()
        ml = greedy_search(sh, c, cfg.n_obs, s0, loss="loglik", max_steps=steps, add_screen=20)
        dt = time.perf_counter() - t0
        if p == 10:
            log(f"B p={p} {c_scale} from the {name} ({int(s0.sum())} edges): likelihood search {dt:7.1f} s, "
                f"{len(ml.moves)} moves, {ml.evaluations} supports (least-squares search: {len(ls.moves)} moves)")
        else:
            per_step = dt / max(1, len(ml.moves))
            log(f"B p={p} {c_scale} from the {name} ({int(s0.sum())} edges): likelihood search {dt:7.1f} s for "
                f"{len(ml.moves)} steps ({ml.evaluations} supports); least-squares search {len(ls.moves)} moves; "
                f"full likelihood search about {per_step * max(1, len(ls.moves)) / 60:.0f} min")


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "A":
        part_a(sys.argv[2])
    else:
        part_b(int(sys.argv[2]))
