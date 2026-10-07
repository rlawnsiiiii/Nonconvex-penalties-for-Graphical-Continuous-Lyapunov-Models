"""simulations/diagnostics/campaign.py: the tables of the campaign are computed
from stored counts, so the formulas are checked against the library's metrics, and
the loader and the paired comparisons against a tiny campaign produced by the two
cluster runners (p = 5)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gclm.metrics import confusion, orientation_breakdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import campaign  # noqa: E402
from run_s1_shard import _counts  # noqa: E402


def test_graph_metrics_reproduce_the_library_metrics_from_the_stored_counts():
    rng = np.random.default_rng(3)
    for _ in range(30):
        p = int(rng.integers(4, 9))
        truth = (rng.random((p, p)) < 0.3) * rng.normal(size=(p, p))
        np.fill_diagonal(truth, -1.0)
        support = (rng.random((p, p)) < 0.35) & ~np.eye(p, dtype=bool)
        conf, orient = _counts(support, truth)
        got = campaign.graph_metrics(conf, orient, "x")
        cf = confusion(support.astype(float), truth)
        ob = orientation_breakdown(support.astype(float), truth)
        assert math.isclose(got["x_f1"], cf.f1) and math.isclose(got["x_precision"], cf.precision)
        assert math.isclose(got["x_recall"], cf.tpr) and got["x_edges"] == int(support.sum())
        assert math.isclose(got["x_skeleton_f1"], ob["skeleton_f1"])
        a, b = got["x_orientation_accuracy"], ob["orientation_accuracy"]
        assert (math.isnan(a) and math.isnan(b)) or math.isclose(a, b)
        assert got["x_reversed"] == ob["reversed"] and got["x_hedged"] == ob["hedged"]


@pytest.fixture(scope="module")
def tiny_campaign(tmp_path_factory):
    """Four cells at p = 5, one replicate (16 graphs each), named as on the cluster."""
    root = tmp_path_factory.mktemp("campaign")
    common = ["--shard", "0", "--n-shards", "1", "--p", "5", "--reps", "1", "--n-obs", "1000"]
    cells = {
        "direct_lasso_C2I_n1000": ("run_s1_shard.py", ["--select", "search"]),
        "direct_lasso_Cresc_n1000": ("run_s1_shard.py", ["--select", "search", "--c-scale", "variance"]),
        "direct_MCP-up_Cresc_n1000": ("run_s1_shard.py", ["--select", "search", "--c-scale", "variance",
                                                          "--penalty", "MCP", "--direction", "up"]),
        "search_p5_Cresc_n1000": ("run_search_shard.py", ["--c-scale", "variance", "--restarts", "2"]),
        # a wave 4 style name: the p in the folder name is a label, the rows carry the p
        "direct_adaptive_Cresc_p5_n1000": ("run_s1_shard.py", ["--select", "bic", "--c-scale", "variance",
                                                               "--method", "adaptive"]),
    }
    for name, (runner, extra) in cells.items():
        res = subprocess.run([sys.executable, str(ROOT / "simulations" / runner), *common, *extra,
                              "--out-dir", str(root / name / "shards")],
                             capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, res.stderr
    (root / "not_a_cell").mkdir()                       # ignored by the loader
    return root


def test_loader_reads_estimator_and_search_cells(tiny_campaign):
    rows = campaign.load(tiny_campaign)
    by = {}
    for r in rows:
        by.setdefault((r["loss"], r["estimator"], r["c"], r["n"]), []).append(r)
    assert {k: len(v) for k, v in by.items()} == {
        ("direct", "lasso", "C2I", "1000"): 16, ("direct", "lasso", "Cresc", "1000"): 16,
        ("direct", "MCP-up", "Cresc", "1000"): 16, ("direct", "adaptive", "Cresc", "1000"): 16,
        ("direct", "search-pure", "Cresc", "1000"): 16, ("direct", "search-truth", "Cresc", "1000"): 16}
    assert all(r["p"] == 5 and "p_label" not in r for r in by[("direct", "adaptive", "Cresc", "1000")])
    for r in by[("direct", "MCP-up", "Cresc", "1000")]:
        assert 0.0 <= r["bic_f1"] <= r["max_f1"] <= 1.0          # the best of the path bounds its BIC choice
        assert 0.0 <= r["search_f1"] <= 1.0 and r["p"] == 5
    assert all("max_f1" not in r for r in by[("direct", "search-pure", "Cresc", "1000")])


def test_means_and_paired_differences(tiny_campaign):
    rows = campaign.load(tiny_campaign)
    means = {(m["estimator"], m["c"]): m for m in campaign.means_table(rows)}
    mcp = [r for r in rows if r["estimator"] == "MCP-up"]
    assert means[("MCP-up", "Cresc")]["graphs"] == 16
    assert math.isclose(means[("MCP-up", "Cresc")]["max_f1"], np.mean([r["max_f1"] for r in mcp]))
    assert math.isnan(means[("search-pure", "Cresc")]["max_f1"])          # no path, no path metric
    assert not math.isnan(means[("search-pure", "Cresc")]["search_f1"])

    paired = {(r["estimator"], r["c"], r["reference"], r["true_c"]): r for r in campaign.paired_table(rows)}
    key = lambda r: (r["p"], r["k"], r["c_choice"], r["rep"])
    lasso = {c: {key(r): r for r in rows if r["estimator"] == "lasso" and r["c"] == c} for c in ("C2I", "Cresc")}
    for ref, c in (("same_c", "Cresc"), ("dettling", "C2I")):
        d = [r["max_f1"] - lasso[c][key(r)]["max_f1"] for r in mcp]
        row = paired[("MCP-up", "Cresc", ref, "all")]
        assert row["pairs"] == 16 and math.isclose(row["max_f1_diff"], np.mean(d), abs_tol=1e-12)
    # per true-C setting: 4 graphs each; the lasso is not compared with itself
    assert paired[("MCP-up", "Cresc", "same_c", "C_ID")]["pairs"] == 4
    assert ("lasso", "Cresc", "same_c", "all") not in paired
    assert ("lasso", "Cresc", "dettling", "all") in paired
    # a wave 2 row is compared with the lasso's BIC-selected graph
    d = [r["search_f1"] - lasso["Cresc"][key(r)]["search_f1"] for r in rows if r["estimator"] == "search-pure"]
    assert math.isclose(paired[("search-pure", "Cresc", "same_c", "all")]["search_f1_diff"], np.mean(d), abs_tol=1e-12)


def test_command_line_writes_the_tables(tiny_campaign):
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "campaign.py"),
                          "--root", str(tiny_campaign)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    for name in ("campaign_per_dataset.csv", "campaign_means.csv", "campaign_paired.csv"):
        assert (tiny_campaign / name).stat().st_size > 0
    assert "MCP-up" in res.stdout and "96 rows from 5 cells" in res.stdout
