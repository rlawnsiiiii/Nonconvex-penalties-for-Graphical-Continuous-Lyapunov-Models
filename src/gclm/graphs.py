"""Graph-class diagnostics of a drift matrix's support (plan of 13 October 2026, §1.5).

A support is the off-diagonal pattern of ``M``: ``support[i, j]`` true means ``M_ij != 0``, the
edge ``j -> i`` of Dettling et al. (2024).  For the classes below the direction convention does
not matter.  Three classes, by the structure that decides whether the direction of an edge can
be read off the covariance:

``"dag"``         no directed cycle at all;
``"simple"``      a directed cycle of length three or more, but no pair with both directions;
``"two_cycles"``  at least one pair ``{i, j}`` with both ``i -> j`` and ``j -> i``.

:func:`identified_on_support` asks the question of the campaign's orientation findings directly:
is ``M*`` the only drift matrix with its support that solves the Lyapunov equation at the
population covariance?  It is iff the columns of the design ``A(Sigma*)`` that belong to the
support plus the diagonal are linearly independent, i.e. iff the smallest eigenvalue of the
restricted Gram matrix ``Gamma_SS`` is positive.  Only :mod:`gclm.lyapunov` is used.
"""

from __future__ import annotations

import numpy as np

from gclm.lyapunov import solve_lyapunov

CLASSES = ("dag", "simple", "two_cycles")


def _off(support: np.ndarray) -> np.ndarray:
    s = np.asarray(support).astype(bool).copy()
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("a support is a square boolean matrix")
    np.fill_diagonal(s, False)
    return s


def is_dag(support: np.ndarray) -> bool:
    """Is the directed graph of the off-diagonal support acyclic?  Kahn's algorithm: repeatedly
    remove a node without incoming edges; the graph is a DAG iff every node gets removed."""
    s = _off(support)
    p = s.shape[0]
    indeg = s.sum(axis=1)                       # row i: edges j -> i, i.e. the in-degree of i
    alive = np.ones(p, dtype=bool)
    removed = 0
    while True:
        free = np.flatnonzero(alive & (indeg == 0))
        if free.size == 0:
            break
        for v in free:
            alive[v] = False
            removed += 1
            indeg -= s[:, v]                    # v's outgoing edges (column v) no longer count
    return removed == p


def two_cycles(support: np.ndarray) -> int:
    """The number of unordered pairs with both directions present."""
    s = _off(support)
    return int(np.sum(np.triu(s & s.T, 1)))


def frac_edges_in_two_cycles(support: np.ndarray) -> float:
    """The share of directed edges that lie in a 2-cycle (0 for an empty graph)."""
    s = _off(support)
    n_edges = int(s.sum())
    return float(np.sum(s & s.T)) / n_edges if n_edges else 0.0


def graph_class(support: np.ndarray) -> str:
    """``"two_cycles"`` if a pair has both directions, else ``"dag"`` if acyclic, else ``"simple"``
    (cyclic without 2-cycles; the empty graph and every forest are DAGs)."""
    if two_cycles(support) > 0:
        return "two_cycles"
    return "dag" if is_dag(support) else "simple"


def design_columns(sigma: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The columns of ``A(Sigma)`` (:func:`gclm.lyapunov.design_matrix`) for the entries
    ``(rows[a], cols[a])`` of ``M``, as a ``(p*p, len(rows))`` array, without forming the whole
    ``p^2 x p^2`` design: the column of the entry ``(i, k)`` is ``vec(E_ik Sigma + Sigma E_ki)``,
    whose row ``i`` holds ``Sigma[k, :]`` and whose column ``i`` holds ``Sigma[:, k]``."""
    sigma = np.asarray(sigma, dtype=float)
    p = sigma.shape[0]
    out = np.zeros((p * p, len(rows)))
    for a, (i, k) in enumerate(zip(rows, cols)):
        x = np.zeros((p, p))
        x[i, :] += sigma[k, :]
        x[:, i] += sigma[:, k]
        out[:, a] = x.flatten(order="F")
    return out


def identified_on_support(m_true: np.ndarray, c_true: np.ndarray) -> float:
    """The smallest eigenvalue of ``Gamma_SS = (A' A)_SS`` with ``A = A(Sigma*)``,
    ``Sigma* = Sigma(M*, C*)`` and ``S`` the support of ``M*`` plus the diagonal.  Positive iff
    ``M*`` is the unique drift matrix with that support solving the Lyapunov equation at the
    population covariance (identified on its support); 0 otherwise -- for instance for an
    isolated 2-cycle, whose two entries cannot both be told from the covariance."""
    m = np.asarray(m_true, dtype=float)
    sigma = solve_lyapunov(m, np.asarray(c_true, dtype=float))
    free = (m != 0) | np.eye(m.shape[0], dtype=bool)
    rows, cols = np.nonzero(free)
    a = design_columns(sigma, rows, cols)
    gram = a.T @ a
    return float(max(np.linalg.eigvalsh(gram)[0], 0.0)) if gram.size else 0.0
