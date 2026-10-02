#!/usr/bin/env python3
"""Bias and variance of the direct-loss estimators over replicates of the DATA,
with the drift matrix M* held fixed -- the decomposition the Figure 5 metrics
cannot give, because there every dataset has its own M*.

One M* (p = 10, k = 2, C = 2I), R datasets of N = 1000 observations, each fitted
with lasso / MCP / SCAD along the path from lambda_max down to 0.1 lambda_max
(the region where F1 peaks); the estimate is the path's last point.  Reports

  * per true edge: selection frequency, bias, variance over the R datasets;
  * the variance of the estimate projected on the eigenvectors of Gamma_SS, the
    loss curvature on the true support, from the flattest direction up -- the
    place where a penalty without slope leaves the data on their own;
  * which FALSE entries each method selects systematically (>= 90 % of the
    datasets), and their design-column correlation with the true edges the
    method drops -- the "explained away by a collinear partner" mechanism.

    python simulations/diagnostics/bias_variance_fixed_mstar.py --reps 200 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gclm.data.simulate import sample_covariance, sample_data, sample_drift  # noqa: E402
from gclm.lyapunov import design_matrix, gram_matrix, solve_lyapunov, unvec, vec  # noqa: E402
from gclm.objective.direct import lambda_max  # noqa: E402
from gclm.solvers.path import lasso_path  # noqa: E402

P, K, N = 10, 2, 1000
PENALTIES = ("lasso", "MCP", "SCAD")
SEED = 2026


def setup():
    rng = np.random.default_rng(SEED)
    m_true = sample_drift(P, K / P, rng)
    c = 2.0 * np.eye(P)
    sig = solve_lyapunov(m_true, c)
    d = np.sqrt(np.diag(sig))
    # the data are standardised, so the estimand is the drift on the correlation scale
    return m_true, c, sig, sig / np.outer(d, d), np.diag(1 / d) @ m_true @ np.diag(d)


def fit(r):
    m_true, c, sig, _, _ = setup()
    rng = np.random.default_rng([SEED, r])
    s = sample_covariance(sample_data(N, sig, rng))
    sd = np.sqrt(np.diag(s))
    s = s / np.outer(sd, sd)
    lams = lambda_max(s, c) * np.logspace(-1, 0, 50)
    return {pen: vec(lasso_path(s, c, lambdas=lams, penalty=pen, tol=1e-8).estimates[0])
            for pen in PENALTIES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    _, _, _, sig_corr, m_corr = setup()
    off = ~np.eye(P, dtype=bool)
    true = (m_corr != 0) & off
    idx_true = np.flatnonzero(vec(true))
    idx_false = np.flatnonzero(vec(off & ~true))
    truth = vec(m_corr)

    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(fit, range(args.reps)))
    est = {pen: np.array([r[pen] for r in res]) for pen in PENALTIES}      # R x p^2

    g = gram_matrix(sig_corr)[np.ix_(idx_true, idx_true)]
    ev, v = np.linalg.eigh(g)
    print(f"M*: {len(idx_true)} true edges, |M*| in [{np.abs(truth[idx_true]).min():.2f}, "
          f"{np.abs(truth[idx_true]).max():.2f}]; curvature on the true support: eigenvalues of "
          f"Gamma_SS {ev[0]:.3f} .. {ev[-1]:.2f}, condition number {ev[-1] / ev[0]:.0f}; {args.reps} datasets\n")

    print(f"{'':6s} {'sel.freq(true)':>14} {'mean bias':>10} {'mean |bias|':>11} {'mean var':>9} {'mean MSE':>9} "
          f"{'#false sel.':>11} {'var v_min':>10} {'var v_max':>10}")
    for pen in PENALTIES:
        a = est[pen][:, idx_true]
        bias = a.mean(0) - truth[idx_true]
        var = a.var(0, ddof=1)
        nf = (est[pen][:, idx_false] != 0).sum(1).mean()
        print(f"{pen:6s} {(a != 0).mean():>14.2f} {bias.mean():>+10.3f} {np.abs(bias).mean():>11.3f} "
              f"{var.mean():>9.3f} {(bias ** 2 + var).mean():>9.3f} {nf:>11.1f} "
              f"{(a @ v[:, 0]).var(ddof=1):>10.4f} {(a @ v[:, -1]).var(ddof=1):>10.4f}")

    al, am = est["lasso"][:, idx_true], est["MCP"][:, idx_true]
    print("\nvariance ratio MCP / lasso along the curvature directions of Gamma_SS, flattest first:")
    print("  eigenvalue:", np.round(ev, 2).tolist())
    print("  ratio     :", np.round([(am @ v[:, j]).var() / max((al @ v[:, j]).var(), 1e-12)
                                     for j in range(len(idx_true))], 1).tolist())

    print(f"\nper true edge (sorted by |M*|):  {'|M*|':>5}  " + "   ".join(f"{pen}: sel  bias   var" for pen in PENALTIES))
    for j in np.argsort(-np.abs(truth[idx_true])):
        cols = "   ".join(f"{(est[pen][:, idx_true[j]] != 0).mean():>9.2f} {est[pen][:, idx_true[j]].mean() - truth[idx_true[j]]:>+5.2f} {est[pen][:, idx_true[j]].var():>6.3f}"
                         for pen in PENALTIES)
        print(f"{'':32s}{abs(truth[idx_true[j]]):>5.2f}  {cols}")

    # where does the signal of a dropped true edge go?  false entries selected systematically
    a_design = design_matrix(sig_corr)
    corr = np.corrcoef(a_design.T)
    for pen in ("MCP", "SCAD"):
        dropped = [j for j in idx_true if (est[pen][:, j] != 0).mean() < 0.1 and (est["lasso"][:, j] != 0).mean() > 0.9]
        sub = [j for j in idx_false if (est[pen][:, j] != 0).mean() >= 0.9]
        print(f"\n{pen}: true edges selected < 10 % by {pen} but > 90 % by the lasso: "
              f"{[tuple(int(t) for t in np.unravel_index(j, (P, P), order='F')) for j in dropped]}")
        print(f"{pen}: false entries selected >= 90 % of the time: "
              f"{[(tuple(int(t) for t in np.unravel_index(j, (P, P), order='F')), round(float((est['lasso'][:, j] != 0).mean()), 2)) for j in sub]}  (value: lasso selection frequency)")
        for j in dropped:
            best = sorted(((abs(corr[j, s]), s) for s in sub), reverse=True)[:3]
            print(f"    dropped {tuple(int(t) for t in np.unravel_index(j, (P, P), order='F'))}: most correlated design columns among {pen}'s systematic false picks: "
                  + ", ".join(f"{tuple(int(t) for t in np.unravel_index(s, (P, P), order='F'))} |corr|={c:.2f}" for c, s in best))


if __name__ == "__main__":
    main()
