"""Support-recovery metrics -- Dettling et al. (2024), Definitions G.4 and G.5.

Curve construction follows Varando's ``functions/util.R`` (``AUROC``, ``AUCPR``,
``evaluatePathB``), which is the concrete recipe behind Dettling's
"interpolation and extrapolation if necessary".

Degenerate cases match that code exactly: tpr := 1 when there are no true edges,
fpr := 1 when there are no true non-edges, precision := 1 when nothing is
selected, f1 := 0 when undefined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def tpr(self) -> float:
        return 1.0 if self.tp + self.fn == 0 else self.tp / (self.tp + self.fn)

    @property
    def fpr(self) -> float:
        return 1.0 if self.fp + self.tn == 0 else self.fp / (self.fp + self.tn)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return float("nan") if total == 0 else (self.tp + self.tn) / total

    @property
    def precision(self) -> float:
        return 1.0 if self.tp + self.fp == 0 else self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tpr

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 0.0 if denom == 0 else 2 * self.tp / denom


def confusion(
    m_hat: np.ndarray,
    m_star: np.ndarray,
    include_diagonal: bool = False,
) -> Confusion:
    """Entrywise confusion counts (Definition G.4).

    ``include_diagonal=False`` scores only off-diagonal entries, matching
    Varando's ``evaluatePathB``.  See S1_reproduction.md Section 4.1 for why this
    is a configuration switch and not a settled fact.
    """
    mask = np.ones(np.shape(m_hat), dtype=bool)
    if not include_diagonal:
        np.fill_diagonal(mask, False)
    est = (np.asarray(m_hat) != 0.0)[mask]
    truth = (np.asarray(m_star) != 0.0)[mask]
    return Confusion(
        tp=int(np.sum(est & truth)),
        fp=int(np.sum(est & ~truth)),
        tn=int(np.sum(~est & ~truth)),
        fn=int(np.sum(~est & truth)),
    )


def auc_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve, Varando's ``AUROC``.

    ``fpr``/``tpr`` are given in order of *increasing* lambda (sparsest last).
    The sequence is reversed, anchored with (0, 0) and (1, 1), then integrated by
    the trapezoid rule -- the literal R recipe, including the fact that no sorting
    by fpr is performed.
    """
    x = np.concatenate(([0.0], np.asarray(fpr, float)[::-1], [1.0]))
    y = np.concatenate(([0.0], np.asarray(tpr, float)[::-1], [1.0]))
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))


def aupr(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, Varando's ``AUCPR``.

    Inputs are in order of increasing lambda (recall decreasing).  Trapezoid over
    consecutive points plus a closing segment from the lowest recall to
    ``recall = 0`` at ``precision = 1``.
    """
    r = np.asarray(recall, float)
    pr = np.asarray(precision, float)
    area = float(np.sum((pr[1:] + pr[:-1]) * (r[:-1] - r[1:]) / 2.0))
    return area + (1.0 + pr[-1]) * r[-1] / 2.0


def evaluate_path(
    estimates: list[np.ndarray],
    m_star: np.ndarray,
    include_diagonal: bool = False,
) -> dict[str, float | np.ndarray]:
    """All Definition G.5 quantities for one regularization path.

    ``estimates`` must be ordered by *increasing* lambda.
    """
    confs = [confusion(m, m_star, include_diagonal) for m in estimates]
    tpr = np.array([c.tpr for c in confs])
    fpr = np.array([c.fpr for c in confs])
    acc = np.array([c.accuracy for c in confs])
    f1 = np.array([c.f1 for c in confs])
    prec = np.array([c.precision for c in confs])
    return {
        "tpr": tpr,
        "fpr": fpr,
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "max_acc": float(np.nanmax(acc)),
        "max_f1": float(np.nanmax(f1)),
        "auc": auc_roc(fpr, tpr),
        "aupr": aupr(tpr, prec),
    }
