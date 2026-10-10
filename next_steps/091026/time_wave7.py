"""Timing pilot for wave 7 (9 October 2026): the log-likelihood estimators at p = 20 (both C) and
the likelihood-refit searches at p = 10.  One data set per line (k = 2, C_ID, n = 1e4, replicate 0,
the generator of the campaign).  Output: time_wave7.txt.  Run from the repository root."""
import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "simulations")
import numpy as np
from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility
from gclm.objective import covariance as cov
from gclm.solvers.path import adaptive_covloss_path, covloss_path, lambda_grid
from gclm.solvers.search import Scorer, bic_along_path, greedy_search, multistart_search, random_support


def data(p, c_scale, n=10_000):
    cfg = S1Config(n_obs=n, c_scale=c_scale)
    rng = np.random.default_rng([cfg.seed, p, 2, 0, 0])
    m_true, _, _, sh, scale = draw_instance(p, 2, n, list(CChoice)[0], rng, metzler=cfg.metzler,
                                            standardize=cfg.standardize, return_scale=True)
    return sh, estimation_volatility(scale, c_scale), cfg


def supports(path, p):
    eye = np.eye(p, dtype=bool)
    return [(m != 0) & ~eye for m in path.estimates]


for p, c_scale in ((20, "identity"), (20, "variance")):
    sh, c, cfg = data(p, c_scale)
    lams = lambda_grid(cov.lambda_max(sh, c, "loglik"), 100, 1e-4)
    for name, kw in (("lasso", dict(penalty="lasso")), ("MCP", dict(penalty="MCP")),
                     ("MCP-up exact", dict(penalty="MCP", direction="up", start="exact")),
                     ("MCP-up lasso", dict(penalty="MCP", direction="up", start="lasso"))):
        t0 = time.perf_counter()
        path = covloss_path(sh, c, "loglik", lambdas=lams, tol=cfg.tol, **kw)
        print(f"p={p} {c_scale:8s} {name:13s} path {time.perf_counter() - t0:7.1f} s", flush=True)
        if name == "lasso":
            t0 = time.perf_counter()
            sc = Scorer(sh, c, cfg.n_obs, "loglik")
            ib, _ = bic_along_path(sh, c, cfg.n_obs, supports(path, p), scorer=sc)
            print(f"p={p} {c_scale:8s} lasso, score with the likelihood refit {time.perf_counter() - t0:7.1f} s "
                  f"({sc.evaluations} supports)", flush=True)
    t0 = time.perf_counter()
    ada = adaptive_covloss_path(sh, c, "loglik", tol=cfg.tol)
    print(f"p={p} {c_scale:8s} adaptive      path {time.perf_counter() - t0:7.1f} s", flush=True)

p = 10
sh, c, cfg = data(p, "variance")
lasso = covloss_path(sh, c, "loglik", lambdas=lambda_grid(cov.lambda_max(sh, c, "loglik"), 100, 1e-4), tol=cfg.tol)
sc = Scorer(sh, c, cfg.n_obs, "loglik")
ib, _ = bic_along_path(sh, c, cfg.n_obs, supports(lasso, p), scorer=sc)
t0 = time.perf_counter()
res = greedy_search(sh, c, cfg.n_obs, supports(lasso, p)[ib], scorer=sc, loss="loglik", max_steps=p * (p - 1))
print(f"p=10 variance search from the selected graph, likelihood refit, all adds: {time.perf_counter() - t0:7.1f} s, "
      f"{len(res.moves)} moves", flush=True)
rng = np.random.default_rng([20261003, 10])
starts = [random_support(p, rng, 0.3) for _ in range(10)] + [np.zeros((p, p), bool)]
for screen in (20, None):
    t0 = time.perf_counter()
    best, results = multistart_search(sh, c, cfg.n_obs, starts, loss="loglik", max_steps=p * (p - 1), add_screen=screen)
    print(f"p=10 variance pure search, 10 random starts + empty, likelihood refit, add_screen={screen}: "
          f"{time.perf_counter() - t0:7.1f} s, {best.evaluations} supports", flush=True)
