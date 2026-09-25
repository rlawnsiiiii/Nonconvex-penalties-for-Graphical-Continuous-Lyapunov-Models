"""Definitions G.4/G.5, validated against a transcription of Varando's util.R."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.metrics import aupr, auc_roc, confusion, evaluate_path

from conftest import requires_r, run_r


def test_confusion_counts_off_diagonal_by_default():
    m_star = np.array([[-1.0, 0.5, 0.0], [0.0, -2.0, 0.0], [0.3, 0.0, -1.0]])
    m_hat = np.array([[-1.0, 0.5, 0.7], [0.0, -2.0, 0.0], [0.0, 0.0, -1.0]])
    c = confusion(m_hat, m_star)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 3)      # 6 off-diagonal entries
    with_diag = confusion(m_hat, m_star, include_diagonal=True)
    assert with_diag.tp == 4 and with_diag.tn == 3       # 3 diagonal true positives


def test_degenerate_conventions_match_r():
    """Empty denominators follow evaluatePathB's NaN handling."""
    zero = np.zeros((3, 3))
    c = confusion(zero, zero)
    assert c.tpr == 1.0 and c.precision == 1.0 and c.f1 == 0.0
    dense = np.ones((3, 3))
    assert confusion(dense, np.ones((3, 3))).fpr == 1.0   # no true negatives


def test_perfect_and_useless_classifiers():
    """A perfect path has auc 1; the anchors alone give the diagonal 0.5."""
    assert np.isclose(auc_roc(np.array([0.0]), np.array([1.0])), 1.0)
    assert np.isclose(auc_roc(np.array([0.5]), np.array([0.5])), 0.5)


def test_auc_roc_anchors_and_orientation():
    """Path is given in increasing lambda; the curve is reversed and anchored."""
    fpr = np.array([0.8, 0.4, 0.1])     # sparser as lambda grows
    tpr = np.array([0.9, 0.7, 0.3])
    x = np.array([0.0, 0.1, 0.4, 0.8, 1.0])
    y = np.array([0.0, 0.3, 0.7, 0.9, 1.0])
    assert np.isclose(auc_roc(fpr, tpr), np.trapezoid(y, x))


@requires_r
@pytest.mark.r
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_metrics_match_r(seed):
    """Full path metrics vs. R transcription of util.R (exact agreement)."""
    rng = np.random.default_rng(seed)
    p, n_lambda = 6, 12
    m_star = rng.normal(size=(p, p)) * (rng.random((p, p)) < 0.3)
    np.fill_diagonal(m_star, -3.0)
    # a nested sequence of supports, mimicking a real lasso path
    order = rng.permutation(p * p)
    estimates = []
    for i in range(n_lambda):
        keep = order[: p * p - i * 3]
        m = np.zeros(p * p)
        m[keep] = rng.normal(size=keep.size)
        estimates.append(m.reshape(p, p))

    got = evaluate_path(estimates, m_star)
    want = run_r("reference_metrics.R",
                 {"estimates": [e.tolist() for e in estimates],
                  "M_star": m_star.tolist()},
                 ("jsonlite",))

    for key in ("tpr", "fpr", "acc", "f1", "precision"):
        assert np.allclose(got[key], np.asarray(want[key]), atol=1e-12), key
    for key in ("max_acc", "max_f1", "auc", "aupr"):
        assert np.isclose(got[key], want[key], atol=1e-12), key
