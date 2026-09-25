"""Fixed models from the papers, used as milestone M0 and as regression tests.

Dettling et al. (2024), Example 2 / Appendix A / Figure 3:
``G1`` is the path 1->2->3->4->5 and ``G2`` the 5-cycle obtained by adding 5->1.
The published result is that the Direct Lyapunov Lasso recovers ``G1`` (whose
drift matrix satisfies the irrepresentability condition) but *not* ``G2``, even
in the population limit.
"""

from __future__ import annotations

import numpy as np

DIAGONAL = (-2.0, -3.0, -4.0, -5.0, -6.0)
SUBDIAGONAL = 0.65
M15_FIXED = 0.65


def example2_path() -> np.ndarray:
    """``M*_1``: diagonal (-2,...,-6), four sub-diagonal entries 0.65.

    Edge ``i -> j`` corresponds to ``M[j, i]``, so the sub-diagonal entries
    ``M[i+1, i]`` are the path 1->2->3->4->5.
    """
    m = np.diag(np.array(DIAGONAL))
    for i in range(4):
        m[i + 1, i] = SUBDIAGONAL
    return m


def example2_cycle(m15: float = M15_FIXED) -> np.ndarray:
    """``M*_2``: the path plus the edge 5 -> 1, i.e. entry ``M[0, 4]``."""
    m = example2_path()
    m[0, 4] = m15
    return m
