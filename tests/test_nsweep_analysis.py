"""The paired statistics behind simulations/S2b_nsweep.md (simulations/diagnostics/nsweep.py)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nsweep", ROOT / "simulations" / "diagnostics" / "nsweep.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)


def test_paired_matches_the_textbook_formulas():
    x = np.array([0.5, 0.7, 0.6, 0.9])
    y = np.array([0.4, 0.8, 0.3, 0.6])
    st = ns.paired(x, y)
    d = x - y
    assert st["pairs"] == 4
    assert math.isclose(st["diff"], d.mean())
    assert math.isclose(st["se"], d.std(ddof=1) / 2.0)
    assert math.isclose(st["z"], d.mean() / (d.std(ddof=1) / 2.0))
    assert st["better"] == 0.75


def test_paired_drops_nan_pairs_and_handles_no_variation():
    st = ns.paired([0.5, np.nan, 0.7], [0.4, 0.2, np.nan])
    assert st["pairs"] == 1 and math.isclose(st["diff"], 0.1) and math.isnan(st["z"])
    same = ns.paired([0.3, 0.3, 0.3], [0.3, 0.3, 0.3])
    assert same["diff"] == 0.0 and same["z"] == 0.0 and same["better"] == 0.0
    assert ns.paired([], [])["pairs"] == 0


def test_paired_rows_groups_by_p_and_c_choice_on_common_datasets():
    def cell(shift):
        return {(p, 1, c, rep): {m: 0.5 + shift + 0.01 * rep for m in ns.METRICS}
                for p in (10, 20) for c in ("C_ID", "C_Random_Full") for rep in range(3)}
    a, b = cell(0.1), cell(0.0)
    b.pop((20, 1, "C_ID", 2))                                   # one dataset missing in b
    rows = ns.paired_rows(a, b, {"loss": "direct"}, groups=("p", "c_choice"))
    assert len(rows) == 4 * len(ns.METRICS)
    by = {(r["p"], r["c_choice"], r["metric"]): r for r in rows}
    assert by[(10, "C_ID", "max_f1")]["pairs"] == 3 and by[(20, "C_ID", "max_f1")]["pairs"] == 2
    assert all(math.isclose(r["diff"], 0.1) for r in rows)
    pooled = ns.paired_rows(a, b, {"loss": "direct"})
    assert {r["p"] for r in pooled} == {10, 20} and pooled[0]["pairs"] == 6
