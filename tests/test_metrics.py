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

    # the R port scores exactly the estimates it is given, so compare the
    # un-anchored metrics; the anchor has its own test below
    got = evaluate_path(estimates, m_star, anchor_dense=False)
    want = run_r("reference_metrics.R",
                 {"estimates": [e.tolist() for e in estimates],
                  "M_star": m_star.tolist()},
                 ("jsonlite",))

    for key in ("tpr", "fpr", "acc", "f1", "precision"):
        assert np.allclose(got[key], np.asarray(want[key]), atol=1e-12), key
    for key in ("max_acc", "max_f1", "auc", "aupr"):
        assert np.isclose(got[key], want[key], atol=1e-12), key


def test_aggregator_metrics_match_evaluate_path():
    """`aggregate_s1.metrics_from_counts` must reproduce `evaluate_path`.

    The cluster stores raw confusion counts and the aggregator recomputes the
    metrics from them.  That is a second implementation of Definitions G.4/G.5,
    so it has to agree with the one the simulations use -- otherwise the
    published numbers would depend on which code path produced them.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulations"))
    from aggregate_s1 import metrics_from_counts

    rng = np.random.default_rng(0)
    p, n_lambda = 7, 20
    m_star = rng.normal(size=(p, p)) * (rng.random((p, p)) < 0.3)
    np.fill_diagonal(m_star, -3.0)

    order = rng.permutation(p * p)
    estimates = []
    for i in range(n_lambda):
        keep = order[: max(p * p - i * 4, 0)]
        v = np.zeros(p * p)
        v[keep] = rng.normal(size=keep.size)
        estimates.append(v.reshape(p, p))

    for include_diagonal in (False, True):
        want = evaluate_path(estimates, m_star, include_diagonal=include_diagonal,
                             anchor_dense=False)
        counts = np.array([
            [(c := confusion(m, m_star, include_diagonal)).tp, c.fp, c.tn, c.fn]
            for m in estimates
        ], dtype=np.int32)
        got = metrics_from_counts(counts, anchor_dense=False)
        for key in ("max_acc", "max_f1", "auc", "aupr"):
            assert np.isclose(got[key], want[key], atol=1e-12), (include_diagonal, key)
        assert np.allclose(got["_tpr"], want["tpr"])
        assert np.allclose(got["_fpr"], want["fpr"])
        assert np.allclose(got["_prec"], want["precision"])


def test_dense_anchor_fixes_the_truncated_pr_curve():
    """The lasso path saturates, so `aupr` needs the dense anchor.

    `A(Sigma)` has rank `p(p+1)/2`, so the lasso can select at most that many
    entries and the path never reaches a dense model -- at p = 20 the densest
    fit on the real grid reaches only tpr 0.79.  Without an anchor at recall 1
    the PR curve stops there and the remaining area is silently dropped.
    """
    p = 10
    m_star = np.zeros((p, p))
    np.fill_diagonal(m_star, -3.0)
    true_edges = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]   # 5 true edges
    for i, j in true_edges:
        m_star[i, j] = 0.6

    # A saturating path: even the densest fit recovers only 3 of the 5 true
    # edges, so tpr never reaches 1 -- exactly what the rank cap produces.
    off = [(i, j) for i in range(p) for j in range(p) if i != j]
    recovered = true_edges[:3]
    estimates = []
    for n_sel in (20, 12, 6, 3, 1, 0):
        m = np.zeros((p, p))
        np.fill_diagonal(m, -3.0)
        chosen = recovered[:min(n_sel, 3)]
        extra = [e for e in off if e not in true_edges][: max(n_sel - 3, 0)]
        for i, j in chosen + extra:
            m[i, j] = 0.5
        estimates.append(m)

    plain = evaluate_path(estimates, m_star, anchor_dense=False)
    anchored = evaluate_path(estimates, m_star, anchor_dense=True)

    assert plain["tpr"].max() < 1.0, "the fixture must saturate for this to test anything"
    # the anchor only adds area; it never removes any
    assert anchored["aupr"] > plain["aupr"]
    # the per-lambda arrays stay at grid length -- the anchor is not a fit
    for key in ("tpr", "fpr", "acc", "f1", "precision"):
        assert len(anchored[key]) == len(estimates), key
    # max_acc / max_f1 are maxima over the grid and must not see the anchor
    assert anchored["max_acc"] == plain["max_acc"]
    assert anchored["max_f1"] == plain["max_f1"]


def test_dense_anchor_is_a_noop_when_the_path_already_reaches_recall_one():
    """If the densest fit already selects everything, the anchor changes nothing."""
    p = 5
    m_star = np.zeros((p, p))
    np.fill_diagonal(m_star, -2.0)
    m_star[1, 0] = m_star[2, 1] = 0.5
    dense = np.full((p, p), 0.3)
    np.fill_diagonal(dense, -2.0)
    estimates = [dense, m_star.copy()]

    plain = evaluate_path(estimates, m_star, anchor_dense=False)
    anchored = evaluate_path(estimates, m_star, anchor_dense=True)
    assert np.isclose(plain["aupr"], anchored["aupr"])
    assert np.isclose(plain["auc"], anchored["auc"])
