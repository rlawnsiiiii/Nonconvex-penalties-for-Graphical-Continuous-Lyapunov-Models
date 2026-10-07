#!/usr/bin/env python3
"""Why does the rescaled C help the estimators that start from the dense end of the
lasso path, while the lasso path itself hardly cares?

C is the right-hand side of the Lyapunov equation, M Sigma + Sigma M' = -C.  It decides
which drift matrices fit the data at all.  With the wrong C the true M does not fit, and
the whole solution set {M : M Sigma + Sigma M' = -C_wrong} is shifted away from it by a
DENSE matrix (a diagonal error in C needs a dense correction in M, because Sigma is dense).
Estimators that decide from the dense end -- MCP/SCAD dense -> sparse, the adaptive lasso --
start on that shifted set, so the entries they prune are "true weights + dense shift".
The lasso's best point is in the middle of the path, where the fit is a compromise anyway.

For C_ID graphs (data generated with C = 2I, then standardised) this prints, under C = 2I
and under the rescaled C:
  misfit     relative loss of the best fit on the TRUE support (0 = the truth fits)
  shift      the dense shift at n = inf: median |entry| of -A^+ vec(C_used - C_true), over
             the off-diagonal entries, against the true edge weights (median, 10th percentile)
  rank AUC   how well the size of the entries of the dense-end lasso solution ranks the true
             edges above the non-edges (the adaptive lasso's weights are exactly these sizes)
  reversed   share of true single edges whose REVERSED entry is larger than the true one at
             the dense end (the orientation the pruning would keep)

    python next_steps/051026/files/why_rescaling_helps.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility  # noqa: E402
from gclm.lyapunov import design_matrix, vec  # noqa: E402
from gclm.metrics import auc_roc  # noqa: E402
from gclm.objective.direct import diagonal_fit  # noqa: E402
from gclm.solvers.path import lasso_path  # noqa: E402


def misfit(sigma, c, support):
    a = design_matrix(sigma)
    idx = np.flatnonzero(vec(support | np.eye(len(support), dtype=bool)))
    b, *_ = np.linalg.lstsq(a[:, idx], -vec(c), rcond=None)
    r = a[:, idx] @ b + vec(c)
    m_d = diagonal_fit(sigma, c)
    ref = m_d @ sigma + sigma @ m_d.T + c
    return float(r @ r) / float(np.sum(ref ** 2))


def rank_auc(score, truth):
    """AUC of ranking the off-diagonal entries by `score` against the true support."""
    s, t = score.ravel(), truth.ravel()
    order = np.argsort(-s)
    tpr = np.cumsum(t[order]) / t.sum()
    fpr = np.cumsum(~t[order]) / (~t).sum()
    return auc_roc(fpr[::-1], tpr[::-1])          # auc_roc wants increasing-lambda order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5, help="graphs per density k")
    args = ap.parse_args()
    cfg, p = S1Config(), args.p
    rows = {("identity", n): [] for n in ("inf", "1e4")} | {("variance", n): [] for n in ("inf", "1e4")}
    for k in (1, 2, 3, 4):
        for rep in range(args.reps):
            for n_label, n in (("inf", np.inf), ("1e4", 10_000)):
                rng = np.random.default_rng([cfg.seed, p, k, 0, rep])
                m_true, c_true, _, r, s = draw_instance(p, k, n, CChoice.ID, rng, standardize=True,
                                                        return_scale=True)
                off = ~np.eye(p, dtype=bool)
                truth = (m_true != 0) & off
                m_std = m_true * s[None, :] / s[:, None]          # the true drift on the standardised scale
                c_std = np.diag(np.diag(c_true) / s ** 2)         # the true C on the standardised scale
                a_pinv = np.linalg.pinv(design_matrix(r)) if n_label == "inf" else None
                for c_scale in ("identity", "variance"):
                    c = estimation_volatility(s, c_scale)
                    dense = lasso_path(r, c, n_lambda=cfg.n_lambda, ratio=cfg.lambda_ratio,
                                       tol=cfg.tol).estimates[0]
                    size = np.abs(dense)
                    single = truth & ~truth.T
                    jj, ii = np.nonzero(single)                    # true edge ii -> jj is entry [jj, ii]
                    rev = float(np.mean(size[ii, jj] > size[jj, ii])) if len(ii) else np.nan
                    row = {"misfit": misfit(r, c, truth), "auc": rank_auc(size, truth), "reversed": rev,
                           "true_med": float(np.median(np.abs(m_std[truth]))),
                           "true_p10": float(np.quantile(np.abs(m_std[truth]), 0.1))}
                    if a_pinv is not None:
                        shift = -(a_pinv @ vec(c - c_std)).reshape((p, p), order="F")
                        row["shift_med"] = float(np.median(np.abs(shift[off])))
                        row["shift_max"] = float(np.max(np.abs(shift[off])))
                    rows[(c_scale, n_label)].append(row)
    print(f"C_ID, p = {p}, {4 * args.reps} graphs; medians over graphs\n")
    print(f"{'n':<6}{'C used':<12}{'misfit':>8}{'rank AUC':>10}{'reversed':>10}{'shift med':>11}{'shift max':>11}{'true med':>10}{'true p10':>10}")
    for n_label in ("inf", "1e4"):
        for c_scale in ("identity", "variance"):
            g = rows[(c_scale, n_label)]
            med = lambda key: np.nanmedian([x[key] for x in g]) if key in g[0] else np.nan
            print(f"{n_label:<6}{('C = 2I' if c_scale == 'identity' else 'rescaled'):<12}"
                  f"{med('misfit'):>8.4f}{med('auc'):>10.3f}{med('reversed'):>10.2f}"
                  f"{med('shift_med'):>11.3f}{med('shift_max'):>11.3f}{med('true_med'):>10.3f}{med('true_p10'):>10.3f}")


if __name__ == "__main__":
    main()
