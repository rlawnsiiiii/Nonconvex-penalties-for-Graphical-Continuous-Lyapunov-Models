"""Timing of wave 8 (c) (10 October 2026, night): the greedy search with the likelihood refit
(20 add moves screened per step) from random sparse starting graphs and from the truth, p = 10,
on an idle laptop.  The pilot of 9 October measured 5549 s for 11 starts of one graph on a busy
machine; a task of (c) holds one graph with 100 starts + the empty graph + the truth, under a 24 h
limit.  One line per start, appended to time_wave8c.txt.

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 next_steps/091026/time_wave8c.py identity 2
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "simulations")
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.solvers.search import Scorer, greedy_search, random_support

OUT = Path("next_steps/091026/time_wave8c.txt")
c_scale, k = sys.argv[1], int(sys.argv[2])
p, n, starts = 10, 10_000, 3
cfg = S1Config(n_obs=n, c_scale=c_scale)
rng = np.random.default_rng([cfg.seed, p, k, 0, 0])
m_true, _, _, sh, scale = draw_instance(p, k, n, list(CChoice)[0], rng, metzler=cfg.metzler,
                                        standardize=cfg.standardize, return_scale=True)
c = estimation_volatility(scale, c_scale)
truth = (m_true != 0) & ~np.eye(p, dtype=bool)
sc = Scorer(sh, c, n, "loglik")                       # shared cache across starts, as in multistart_search
srng = np.random.default_rng([20261010, k, 0 if c_scale == "identity" else 1])
jobs = [(f"random start {i + 1}", random_support(p, srng, 0.3)) for i in range(starts)] + [("truth", truth)]
for name, s0 in jobs:
    t0 = time.perf_counter()
    res = greedy_search(sh, c, n, s0, loss="loglik", max_steps=p * (p - 1), add_screen=20, scorer=sc)
    dt = time.perf_counter() - t0
    with OUT.open("a") as fh:
        fh.write(f"p=10 k={k} {c_scale:8s} {name:15s} ({int(s0.sum()):2d} edges): {dt:7.1f} s, "
                 f"{len(res.moves):3d} moves, score {res.score:.1f}\n")
