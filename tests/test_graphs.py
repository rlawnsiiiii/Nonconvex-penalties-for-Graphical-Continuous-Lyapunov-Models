"""gclm.graphs: the graph classes of a support and identification on the true support (plan of
13 October 2026, §1.5), on hand-built 3- and 4-node examples and against the design matrix."""

from __future__ import annotations

import numpy as np
import pytest

from gclm.graphs import (CLASSES, design_columns, frac_edges_in_two_cycles, graph_class, identified_on_support,
                         is_dag, two_cycles)
from gclm.lyapunov import design_matrix, solve_lyapunov


def support(p, edges):
    """``edges`` as (from, to) pairs: the entry (to, from) of the support."""
    s = np.zeros((p, p), dtype=bool)
    for j, i in edges:
        s[i, j] = True
    return s


def test_classes_on_small_examples():
    chain = support(3, [(0, 1), (1, 2)])                   # 1 -> 2 -> 3
    cycle3 = support(3, [(0, 1), (1, 2), (2, 0)])          # 1 -> 2 -> 3 -> 1
    pair = support(3, [(0, 1), (1, 0)])                    # 1 <-> 2, node 3 isolated
    mixed = support(4, [(0, 1), (1, 2), (2, 1), (2, 3)])   # a chain with one 2-cycle inside
    empty = np.zeros((4, 4), dtype=bool)
    assert is_dag(chain) and is_dag(empty) and not is_dag(cycle3) and not is_dag(pair) and not is_dag(mixed)
    assert two_cycles(chain) == 0 and two_cycles(cycle3) == 0 and two_cycles(pair) == 1 and two_cycles(mixed) == 1
    assert frac_edges_in_two_cycles(pair) == 1.0 and frac_edges_in_two_cycles(mixed) == pytest.approx(0.5)
    assert frac_edges_in_two_cycles(chain) == 0.0 and frac_edges_in_two_cycles(empty) == 0.0
    assert graph_class(chain) == "dag" and graph_class(empty) == "dag"
    assert graph_class(cycle3) == "simple" and graph_class(pair) == "two_cycles" and graph_class(mixed) == "two_cycles"
    assert {graph_class(s) for s in (chain, cycle3, pair)} == set(CLASSES)
    # the diagonal is ignored, and a float matrix is read by its nonzero pattern
    m = np.diag([-1.0, -2.0, -3.0])
    m[1, 0] = 0.5
    assert graph_class(m) == "dag" and two_cycles(m + m.T) == 1


def test_design_columns_match_the_design_matrix():
    rng = np.random.default_rng(0)
    p = 5
    sigma = rng.normal(size=(p, p))
    sigma = sigma @ sigma.T + p * np.eye(p)
    a = design_matrix(sigma)
    rows, cols = np.nonzero(rng.random((p, p)) < 0.5)
    idx = np.arange(p * p).reshape((p, p), order="F")[rows, cols]     # vec is column-major
    np.testing.assert_allclose(design_columns(sigma, rows, cols), a[:, idx], atol=1e-12)


def test_identification_on_the_true_support():
    """A chain is identified on its support (every entry of M* can be read off Sigma*); an isolated
    2-cycle is not: its two entries plus the diagonal are four unknowns for three equations."""
    chain = np.array([[-2.0, 0.0, 0.0], [0.7, -3.0, 0.0], [0.0, -0.6, -2.5]])
    pair = np.array([[-2.0, 0.6], [0.5, -3.0]])
    c = 2.0 * np.eye(3)
    assert identified_on_support(chain, c) > 1e-6
    assert identified_on_support(pair, 2.0 * np.eye(2)) == 0.0
    # the diagonal matrix is identified too, and the fully dense one never (p^2 unknowns, p(p+1)/2 equations)
    assert identified_on_support(np.diag([-1.0, -2.0, -3.0]), c) > 1e-6
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(4, 4))
    np.fill_diagonal(dense, -np.abs(dense).sum(axis=1) - 0.5)
    assert identified_on_support(dense, 2.0 * np.eye(4)) == 0.0
    # the value is the smallest eigenvalue of the restricted Gram matrix of the design at Sigma*
    sigma = solve_lyapunov(chain, c)
    a = design_matrix(sigma)
    free = (chain != 0) | np.eye(3, dtype=bool)
    idx = np.arange(9).reshape((3, 3), order="F")[free]
    gram = a[:, idx].T @ a[:, idx]
    assert identified_on_support(chain, c) == pytest.approx(np.linalg.eigvalsh(gram)[0])
