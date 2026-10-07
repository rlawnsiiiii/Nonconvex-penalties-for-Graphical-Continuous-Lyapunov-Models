#!/usr/bin/env python3
"""How well does the TRUE support fit the standardised population covariance, depending on what
is assumed about C?  Extends the fit check of next_steps/031026 (Section 3.1) by a third option:
a diagonal C that is left free (only its trace is fixed, because (M, C) and (aM, aC) give the
same covariance).

For each graph: the least-squares loss  0.5 * || M R + R M' + C ||_F^2  minimised over all M with
the true support, divided by the loss of the diagonal fit under C = 2I.  0 = the truth fits.

    python next_steps/051026/files/fit_check_free_diagonal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.lyapunov import design_matrix, vec  # noqa: E402
from gclm.objective.direct import diagonal_fit  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)


def loss_fixed(a, idx, c):
    b, *_ = np.linalg.lstsq(a[:, idx], -vec(c), rcond=None)
    r = a[:, idx] @ b + vec(c)
    return 0.5 * float(r @ r)


def loss_free_diagonal(a, idx, p, trace):
    """min over M on the support and diagonal c with sum(c) = trace."""
    e = np.zeros((p * p, p))
    for i in range(p):
        d = np.zeros((p, p))
        d[i, i] = 1.0
        e[:, i] = vec(d)
    # c = trace / p + N z, with the columns of N spanning {z : sum(z) = 0}
    n_mat = np.linalg.svd(np.ones((1, p)))[2][1:].T
    x = np.hstack([a[:, idx], e @ n_mat])
    y = -e @ np.full(p, trace / p)
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    r = x @ b - y
    c = trace / p + n_mat @ b[len(idx):]
    return 0.5 * float(r @ r), c


def main():
    cfg, p, reps = S1Config(), 10, 10
    print(f"p = {p}, n = inf, {4 * reps} graphs per setting; mean relative loss of the best fit on "
          f"the true support")
    print(f"{'true C':<20}{'C = 2I':>10}{'rescaled C':>12}{'free diagonal':>15}{'min c_i':>10}")
    for c_name in C_NAMES:
        out = []
        for k in (1, 2, 3, 4):
            for rep in range(reps):
                rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
                m_true, _, _, s_raw = draw_instance(p, k, np.inf, CChoice(c_name), rng,
                                                    metzler=cfg.metzler, standardize=False)
                s = np.sqrt(np.diag(s_raw))
                r = s_raw / np.outer(s, s)
                a = design_matrix(r)
                idx = np.flatnonzero(vec(m_true != 0))
                c_id, c_resc = 2.0 * np.eye(p), np.diag(2.0 / s ** 2)
                m_d = diagonal_fit(r, c_id)
                res = m_d @ r + r @ m_d.T + c_id
                ref = 0.5 * float(np.sum(res ** 2))
                # the loss scales with the square of the overall size of C: bring the rescaled
                # and the free C to the trace of 2I before comparing
                scale = (2.0 * p / np.trace(c_resc)) ** 2
                free, c_hat = loss_free_diagonal(a, idx, p, 2.0 * p)
                out.append((loss_fixed(a, idx, c_id) / ref, scale * loss_fixed(a, idx, c_resc) / ref,
                            free / ref, c_hat.min()))
        o = np.array(out)
        print(f"{c_name:<20}{o[:, 0].mean():>10.4f}{o[:, 1].mean():>12.4f}{o[:, 2].mean():>15.4f}"
              f"{o[:, 3].min():>10.2f}")


if __name__ == "__main__":
    main()
