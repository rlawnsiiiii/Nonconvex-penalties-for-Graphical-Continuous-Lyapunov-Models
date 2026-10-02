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
    anchor_dense: bool = True,
) -> dict[str, float | np.ndarray]:
    """All Definition G.5 quantities for one regularization path.

    ``estimates`` must be ordered by *increasing* lambda.

    ``anchor_dense`` prepends a fully dense pseudo-estimate (every entry
    selected: tpr = fpr = 1, precision = the base rate).  This matters because
    the design matrix is rank deficient -- ``rank A(Sigma) = p(p+1)/2`` out of
    ``p^2`` -- so the lasso can never select more than ``p(p+1)/2`` entries and
    the path *saturates* well short of a dense model.  Measured at p = 20 the
    densest fit reaches only ``tpr = 0.79``, ``fpr = 0.46``.  Without the anchor
    the PR curve simply stops there and the area from that recall up to 1 is
    lost, which biases ``aupr`` down by 0.05-0.14.

    This is what Varando's ``lassoB()`` does by prepending
    ``list(B = Sigma, lambda = 0)`` to the path before scoring, and what
    Dettling's "interpolation and extrapolation if necessary" (Definition G.5)
    implies.  The ROC is unaffected: it already anchors at (1, 1).
    """
    confs = [confusion(m, m_star, include_diagonal) for m in estimates]
    tpr = np.array([c.tpr for c in confs])
    fpr = np.array([c.fpr for c in confs])
    acc = np.array([c.accuracy for c in confs])
    f1 = np.array([c.f1 for c in confs])
    prec = np.array([c.precision for c in confs])

    # curve inputs: optionally extended by the dense anchor.  The per-lambda
    # arrays returned below stay at grid length -- the anchor is not a fit.
    ctpr, cfpr, cprec = tpr, fpr, prec
    if anchor_dense and confs:
        c0 = confs[0]
        base = 1.0 if c0.tp + c0.fn + c0.fp + c0.tn == 0 else \
            (c0.tp + c0.fn) / (c0.tp + c0.fn + c0.fp + c0.tn)
        ctpr = np.concatenate(([1.0], tpr))
        cfpr = np.concatenate(([1.0], fpr))
        cprec = np.concatenate(([base], prec))

    return {
        "tpr": tpr,
        "fpr": fpr,
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "max_acc": float(np.nanmax(acc)),
        "max_f1": float(np.nanmax(f1)),
        "auc": auc_roc(cfpr, ctpr),
        "aupr": aupr(ctpr, cprec),
    }


# --------------------------------------------------------------------------- #
# skeleton and orientation: where does a directed error come from?
# --------------------------------------------------------------------------- #

ORIENTATION_KEYS = (
    "correct", "reversed", "hedged",            # truth has one direction, estimate detects the pair
    "both", "half",                             # truth has both directions (a 2-cycle)
    "missed_single", "missed_double",           # pair not detected at all
    "fp_single", "fp_double",                   # pair detected, truth has neither direction
)


def _pair_patterns(m: np.ndarray) -> np.ndarray:
    """For every unordered pair i < j: 0 none, 1 entry (i,j) only, 2 entry (j,i) only, 3 both."""
    nz = np.asarray(m) != 0.0
    iu, ju = np.triu_indices(nz.shape[0], k=1)
    return nz[iu, ju].astype(int) + 2 * nz[ju, iu].astype(int)


def skeleton_confusion(m_hat: np.ndarray, m_star: np.ndarray) -> Confusion:
    """Confusion counts of the *undirected* skeleton: a pair {i, j} is an edge
    if either direction is nonzero.  Same conventions as :func:`confusion`."""
    est, truth = _pair_patterns(m_hat) > 0, _pair_patterns(m_star) > 0
    return Confusion(
        tp=int(np.sum(est & truth)), fp=int(np.sum(est & ~truth)),
        tn=int(np.sum(~est & ~truth)), fn=int(np.sum(~est & truth)),
    )


def orientation_breakdown(m_hat: np.ndarray, m_star: np.ndarray) -> dict[str, float]:
    """Classify every unordered pair by what the truth has and what the estimate
    has, so that the off-diagonal directed confusion of :func:`confusion` splits
    exactly into orientation errors, 2-cycle errors and skeleton errors:

        tp = correct + hedged + 2 both + half
        fp = reversed + hedged + fp_single + 2 fp_double
        fn = reversed + half + missed_single + 2 missed_double

    ``correct``: one true direction, estimate has exactly that one.
    ``reversed``: one true direction, estimate has exactly the other one.
    ``hedged``: one true direction, estimate keeps both (true direction present,
    one false positive).  ``both`` / ``half``: a true 2-cycle recovered fully /
    as one direction.  Also returned: ``skeleton_f1`` (from
    :func:`skeleton_confusion`), ``orientation_accuracy`` = correct / (correct +
    reversed) over the detected single-direction true edges on which the
    estimate committed to one direction (nan if none), and
    ``orientation_recall`` = (correct + hedged) / (correct + reversed + hedged),
    the fraction of detected single-direction true edges whose true direction is
    present.
    """
    e, t = _pair_patterns(m_hat), _pair_patterns(m_star)
    single_t = (t == 1) | (t == 2)
    out = {
        "correct": int(np.sum(single_t & (e == t))),
        "reversed": int(np.sum(single_t & (e > 0) & (e < 3) & (e != t))),
        "hedged": int(np.sum(single_t & (e == 3))),
        "both": int(np.sum((t == 3) & (e == 3))),
        "half": int(np.sum((t == 3) & (e > 0) & (e < 3))),
        "missed_single": int(np.sum(single_t & (e == 0))),
        "missed_double": int(np.sum((t == 3) & (e == 0))),
        "fp_single": int(np.sum((t == 0) & (e > 0) & (e < 3))),
        "fp_double": int(np.sum((t == 0) & (e == 3))),
        "n_true_single": int(np.sum(single_t)),
        "n_true_double": int(np.sum(t == 3)),
    }
    committed = out["correct"] + out["reversed"]
    detected = committed + out["hedged"]
    out["orientation_accuracy"] = out["correct"] / committed if committed else float("nan")
    out["orientation_recall"] = (out["correct"] + out["hedged"]) / detected if detected else float("nan")
    out["skeleton_f1"] = skeleton_confusion(m_hat, m_star).f1
    return out


def evaluate_skeleton_path(
    estimates: list[np.ndarray],
    m_star: np.ndarray,
    anchor_dense: bool = True,
) -> dict[str, float | np.ndarray]:
    """The Definition G.5 quantities for the *skeleton* along a path -- the same
    curve construction as :func:`evaluate_path`, on :func:`skeleton_confusion`.
    """
    confs = [skeleton_confusion(m, m_star) for m in estimates]
    tpr = np.array([c.tpr for c in confs])
    fpr = np.array([c.fpr for c in confs])
    f1 = np.array([c.f1 for c in confs])
    prec = np.array([c.precision for c in confs])
    ctpr, cfpr, cprec = tpr, fpr, prec
    if anchor_dense and confs:
        c0 = confs[0]
        total = c0.tp + c0.fn + c0.fp + c0.tn
        base = 1.0 if total == 0 else (c0.tp + c0.fn) / total
        ctpr = np.concatenate(([1.0], tpr))
        cfpr = np.concatenate(([1.0], fpr))
        cprec = np.concatenate(([base], prec))
    return {"tpr": tpr, "fpr": fpr, "f1": f1, "precision": prec,
            "max_f1": float(np.nanmax(f1)), "auc": auc_roc(cfpr, ctpr), "aupr": aupr(ctpr, cprec)}
