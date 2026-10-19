#!/usr/bin/env python3
"""E5 (19 October 2026): the share of 2-cycles as a knob.  The wave 9 cells
``direct_<estimator>_<C>_s<share>_n<n>`` draw the graph pair by pair (``gclm.data.simulate.sample_drift``
with ``graph="pairs"``): a connected pair is a 2-cycle with probability ``s`` and the expected number of
directed edges stays the random class's ``p (p - 1) d``, so ``s`` moves only how the true edges are
arranged in pairs.  Wave 1's cells (Dettling's generator, whose pair law is the point ``s = d / (2 - d)``
of the family) are the reference level ``random``.

The tables are those of ``betamin.py`` with the share as the knob (:class:`betamin.Knob`), for the
lasso, MCP dense -> sparse and the adaptive lasso: ``campaign_share.csv`` (levels and paired gains over
the lasso at the same share, for the oracle and every reduction), ``campaign_share_orientation.csv``
(the fates of the true 2-cycles -- both / half / neither -- and of the true single edges, the false
positives per graph and the identified share) and ``campaign_share_edges.csv`` (one row per true edge).
Numbers only; ``plot_twocycles.py`` draws them.

    python simulations/diagnostics/campaign.py                 # first
    python simulations/diagnostics/twocycles.py --c-choices C_ID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from betamin import Knob, run_knob  # noqa: E402

#: the three estimators of E5
E5_ESTIMATORS = ("lasso", "MCP-up", "adaptive")


def share_of(r: dict) -> str:
    """The level of a per-data-set row: the 2-cycle share of a ``pairs`` cell, ``random`` for the
    thesis generator."""
    return str(r["two_cycle_share"]) if r.get("graph") == "pairs" else "random"


def share_key(x: str) -> float:
    return -1.0 if x == "random" else float(x)


def share_label(x: str) -> str:
    return "random\n(Dettling)" if x == "random" else f"s = {float(x):g}"


def is_e5_row(r: dict) -> bool:
    """A direct-loss row of the thesis generator (random class, normal weights) or of a ``pairs`` cell."""
    return (r.get("loss", "direct") == "direct" and r.get("graph", "random") in ("random", "pairs")
            and r.get("weights", "normal") != "uniform")


SHARE = Knob(name="share", tag="share", title="E5: the share of 2-cycles as a knob", of=share_of, of_meta=share_of,
             key=share_key, label=share_label, select=is_e5_row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "campaign")
    ap.add_argument("--c-choices", nargs="+", default=None, metavar="C",
                    help="keep the graphs of these true C only (C_ID when the cells were run with --c-choices C_ID)")
    args = ap.parse_args()
    run_knob(args.root, SHARE, E5_ESTIMATORS, args.c_choices, bins=False, reference="random")


if __name__ == "__main__":
    main()
