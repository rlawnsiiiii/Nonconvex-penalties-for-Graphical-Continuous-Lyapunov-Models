"""S3a bookkeeping (simulations/diagnostics/bidirectional.py): the per-lambda pair
categories must be those of orientation_breakdown, the first-entry codes must
follow the path from lambda_max downwards, and run + summarize must work end
to end."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gclm.metrics import _pair_patterns, orientation_breakdown

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "simulations" / "diagnostics" / "bidirectional.py"
spec = importlib.util.spec_from_file_location("bidirectional", SCRIPT)
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)


def _random_sparse(rng, p, density):
    m = rng.normal(size=(p, p)) * (rng.random((p, p)) < density)
    np.fill_diagonal(m, -1.0)
    return m


@pytest.mark.parametrize("seed", range(5))
def test_outcome_counts_match_orientation_breakdown(seed):
    rng = np.random.default_rng(seed)
    m_true = _random_sparse(rng, 8, 0.35)
    estimates = [_random_sparse(rng, 8, d) for d in (0.0, 0.2, 0.5, 0.9)]
    pats = np.stack([_pair_patterns(m) for m in estimates])
    counts = bd.outcome_counts(pats, _pair_patterns(m_true))
    for i, m_hat in enumerate(estimates):
        ref = orientation_breakdown(m_hat, m_true)
        for key in ("correct", "hedged", "reversed", "missed_single", "both", "half",
                    "missed_double", "fp_single", "fp_double"):
            assert counts[key][i] == ref[key], key
        assert counts["bidirectional"][i] == ref["hedged"] + ref["both"] + ref["fp_double"]


def test_first_entry_follows_the_path_from_lambda_max():
    """Five pairs, four lambdas; each column is written from lambda_max (left)
    to the dense end (right) and reversed into the path's own order."""
    truth = np.array([1, 1, 2, 2, 2], dtype=np.int8)
    sparse_to_dense = np.array([
        [0, 1, 1, 3],      # true direction first                 -> 0
        [2, 2, 3, 3],      # the other direction first            -> 1
        [0, 0, 3, 3],      # both at the same lambda              -> 2
        [0, 0, 0, 0],      # never selected                       -> 3
        [0, 2, 2, 2],      # truth 2, selected as 2 first         -> 0
    ], dtype=np.int8).T
    patterns = sparse_to_dense[::-1]                   # increasing lambda: dense end first
    assert bd.first_entry(patterns, truth).tolist() == [0, 1, 2, 3, 0]


def test_single_outcome_codes():
    truth = np.array([1, 1, 1, 1, 2, 2], dtype=np.int8)
    est = np.array([1, 3, 2, 0, 1, 2], dtype=np.int8)
    # correct, hedged, reversed, missed, reversed, correct
    assert bd.single_outcome(est, truth).tolist() == [0, 1, 2, 3, 2, 0]


def test_run_and_summarize_end_to_end(tmp_path):
    """p = 5 at n = inf, one rep: 16 datasets per penalty through both commands."""
    run = subprocess.run([sys.executable, str(SCRIPT), "run", "--p", "5", "--n", "inf",
                          "--reps", "1", "--workers", "2", "--out", str(tmp_path)],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    raw = sorted(f.name for f in (tmp_path / "raw").glob("*.npz"))
    assert raw == ["p5_ninf_MCP.npz", "p5_ninf_SCAD.npz", "p5_ninf_lasso.npz"]
    d = np.load(tmp_path / "raw" / "p5_ninf_MCP.npz")
    assert d["patterns"].shape == (16, 100, 10) and d["truth"].shape == (16, 10)

    summ = subprocess.run([sys.executable, str(SCRIPT), "summarize", "--out", str(tmp_path)],
                          capture_output=True, text=True, cwd=ROOT)
    assert summ.returncode == 0, summ.stderr
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert {r["penalty"] for r in rows} == {"lasso", "MCP", "SCAD"}
    shares = [float(r["share"]) for r in csv.DictReader((tmp_path / "best_f1_outcomes.csv").open())
              if r["penalty"] == "lasso" and r["group"] == "single edge"]
    assert abs(sum(shares) - 1.0) < 1e-9          # the four outcomes partition the single edges
