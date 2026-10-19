#!/usr/bin/env python3
"""Figures for E5 (19 October 2026), the share of 2-cycles as a knob, from ``campaign_share*.csv``
(``twocycles.py``), with the figure functions of ``plot_betamin.py``: ``gain_by_share[_<rule>_f1].png``,
``levels_by_share_n<n>[_<rule>].png``, ``orientation_by_share_n<n>_<rule>.png`` and
``pair_outcomes_by_share_p<p>_<C>_n<n>_<rule>.png`` -- the last one is the point of the experiment: how
each estimator treats the true 2-cycles (both / one / neither direction kept) and the true single edges
as the share of 2-cycles grows.  **Run locally.**

    python3 simulations/diagnostics/plot_twocycles.py
    python3 simulations/diagnostics/plot_twocycles.py --root <rehearsal root> --out docs/figures --suffix _rehearsal --note "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_betamin import draw, parse  # noqa: E402
from twocycles import E5_ESTIMATORS, SHARE  # noqa: E402


def main() -> None:
    args = parse("E5 figures: the share of 2-cycles")
    draw(args.root, args.out, args.n, args.suffix, args.note, SHARE, gain_ests=("MCP-up", "adaptive"), ests=E5_ESTIMATORS,
         bins=False)


if __name__ == "__main__":
    main()
