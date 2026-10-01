#!/usr/bin/env python3
"""M0 -- Dettling et al. (2024) Example 2 / Figure 3.

Path (G1) vs. 5-cycle (G2) across sample sizes.  Cheap, and the sharpest check
that the whole pipeline is right: G1 should approach perfect recovery, G2 should
not, even at n = infinity.

    python simulations/run_m0.py --reps 100 --out runs/local/m0_reps100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gclm.config import M0Config
from gclm.data.simulate import sample_covariance, sample_data
from gclm.data.examples import example2_cycle, example2_path
from gclm.solvers.path import lasso_path
from gclm.lyapunov import solve_lyapunov
from gclm.metrics import confusion, evaluate_path

SETTINGS = ("path", "cycle_fixed", "cycle_random")


def draw_target(setting: str, rng: np.random.Generator) -> np.ndarray:
    if setting == "path":
        return example2_path()
    if setting == "cycle_fixed":
        return example2_cycle()
    return example2_cycle(rng.uniform(*M0Config.m15_range))


def main() -> None:
    cfg = M0Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=cfg.n_rep)
    ap.add_argument("--out", type=Path, default=Path("runs/local/m0.csv"))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--solver", default="fista",
                    choices=["fista", "ncvreg", "skglm", "glmnet", "pyproximal", "design"])
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    c = 2.0 * np.eye(5)
    rng = np.random.default_rng(args.seed)
    rows = []
    detail = []
    t0 = time.time()

    for setting in SETTINGS:
        for n in cfg.sample_sizes:
            for rep in range(args.reps):
                m_star = draw_target(setting, rng)
                sigma_true = solve_lyapunov(m_star, c)
                if np.isinf(n):
                    sigma_hat = sigma_true
                else:
                    sigma_hat = sample_covariance(sample_data(int(n), sigma_true, rng))
                path = lasso_path(sigma_hat, c, n_lambda=cfg.n_lambda,
                                  ratio=cfg.lambda_ratio, solver=args.solver,
                                  tol=1e-11)
                ev = evaluate_path(path.estimates, m_star)
                rows.append({
                    "setting": setting, "n": n, "rep": rep,
                    "max_acc": ev["max_acc"], "max_f1": ev["max_f1"],
                    "auc": ev["auc"], "aupr": ev["aupr"],
                })
                # full per-lambda detail, so any metric can be recomputed later
                # without re-running (see run_s1_shard.py for the same idea)
                off = np.array([[(cf := confusion(m, m_star, False)).tp, cf.fp,
                                 cf.tn, cf.fn] for m in path.estimates], np.int32)
                inc = np.array([[(cf := confusion(m, m_star, True)).tp, cf.fp,
                                 cf.tn, cf.fn] for m in path.estimates], np.int32)
                detail.append({
                    "setting": SETTINGS.index(setting), "n": float(n), "rep": rep,
                    "lambdas": path.lambdas, "conf_offdiag": off,
                    "conf_incdiag": inc, "m15": m_star[0, 4],
                })
                # the fixed settings are deterministic at n = inf
                if np.isinf(n) and setting in ("path", "cycle_fixed"):
                    break
        print(f"  {setting} done  ({time.time() - t0:.1f}s)", flush=True)

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    npz = args.out.with_suffix(".npz")
    np.savez_compressed(
        npz,
        setting=np.array([d["setting"] for d in detail]),
        setting_names=np.array(SETTINGS),
        n=np.array([d["n"] for d in detail]),
        rep=np.array([d["rep"] for d in detail]),
        m15=np.array([d["m15"] for d in detail]),
        lambdas=np.array([d["lambdas"] for d in detail]),
        conf_offdiag=np.array([d["conf_offdiag"] for d in detail]),
        conf_incdiag=np.array([d["conf_incdiag"] for d in detail]),
        config_json=json.dumps({
            "n_rep": args.reps, "n_lambda": cfg.n_lambda,
            "lambda_ratio": cfg.lambda_ratio, "seed": args.seed,
            "solver": args.solver, "sample_sizes": [float(x) for x in cfg.sample_sizes],
            "diagonal": list(cfg.diagonal), "subdiagonal": cfg.subdiagonal,
            "m15_fixed": cfg.m15_fixed, "m15_range": list(cfg.m15_range),
        }),
    )

    print(f"\nwrote {len(rows)} rows to {args.out}")
    print(f"wrote per-lambda detail for {len(detail)} paths to {npz}\n")
    print(f"{'setting':<14}{'n':>8}  {'max_acc':>8}{'max_f1':>9}{'auc':>8}")
    for setting in SETTINGS:
        for n in cfg.sample_sizes:
            sel = [r for r in rows if r["setting"] == setting and r["n"] == n]
            if not sel:
                continue
            label = "Inf" if np.isinf(n) else f"{int(n)}"
            print(f"{setting:<14}{label:>8}  "
                  f"{np.mean([r['max_acc'] for r in sel]):8.3f}"
                  f"{np.mean([r['max_f1'] for r in sel]):9.3f}"
                  f"{np.mean([r['auc'] for r in sel]):8.3f}")


if __name__ == "__main__":
    main()
