"""simulations/diagnostics/campaign.py: the tables of the campaign are computed
from stored counts, so the formulas are checked against the library's metrics, and
the loader and the paired comparisons against a tiny campaign produced by the two
cluster runners (p = 5).  Also the checks the loader makes on the shards and the
flags for dead selections (code review, 13 October 2026)."""

from __future__ import annotations

import math
import shutil
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


#: the cells of the tiny campaign: name -> (runner, extra arguments).  The C = 2I lasso has no
#: search, so that Dettling's reference lacks the searched graph (code review B1); the rescaled-C
#: lasso also scores with the eBIC penalty (ebic1_search_*), the reference of the "-e1" search
CELLS = {
    "direct_lasso_C2I_n1000": ("run_s1_shard.py", ["--select", "bic"]),
    "direct_lasso_Cresc_n1000": ("run_s1_shard.py", ["--select", "search", "--c-scale", "variance",
                                                    "--ebic-gamma", "1"]),
    "direct_MCP-up_Cresc_n1000": ("run_s1_shard.py", ["--select", "search", "--c-scale", "variance",
                                                     "--penalty", "MCP", "--direction", "up"]),
    "search_p5_Cresc_n1000": ("run_search_shard.py", ["--c-scale", "variance", "--restarts", "2"]),
    "searche1_p5_Cresc_n1000": ("run_search_shard.py", ["--c-scale", "variance", "--restarts", "2",
                                                        "--ebic-gamma", "1"]),
    # a wave 4 style name: the p in the folder name is a label, the rows carry the p
    "direct_adaptive_Cresc_p5_n1000": ("run_s1_shard.py", ["--select", "bic", "--c-scale", "variance",
                                                           "--method", "adaptive"]),
}


@pytest.fixture(scope="module")
def tiny_campaign(tmp_path_factory):
    """Six cells at p = 5, one replicate (16 graphs each), named as on the cluster, each with
    the submit script's ``n_shards``."""
    root = tmp_path_factory.mktemp("campaign")
    common = ["--shard", "0", "--n-shards", "1", "--p", "5", "--reps", "1", "--n-obs", "1000"]
    for name, (runner, extra) in CELLS.items():
        res = subprocess.run([sys.executable, str(ROOT / "simulations" / runner), *common, *extra,
                              "--out-dir", str(root / name / "shards")],
                             capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, res.stderr
        (root / name / "n_shards").write_text("1\n")
    (root / "not_a_cell").mkdir()                       # ignored by the loader
    return root


def key(r):
    return (r["p"], r["k"], r["c_choice"], r["rep"])


def test_loader_reads_estimator_and_search_cells(tiny_campaign):
    rows = campaign.load(tiny_campaign)
    by = {}
    for r in rows:
        by.setdefault((r["loss"], r["estimator"], r["c"], r["n"]), []).append(r)
    assert {k: len(v) for k, v in by.items()} == {
        ("direct", "lasso", "C2I", "1000"): 16, ("direct", "lasso", "Cresc", "1000"): 16,
        ("direct", "MCP-up", "Cresc", "1000"): 16, ("direct", "adaptive", "Cresc", "1000"): 16,
        ("direct", "search-pure", "Cresc", "1000"): 16, ("direct", "search-truth", "Cresc", "1000"): 16,
        ("direct", "search-pure-e1", "Cresc", "1000"): 16, ("direct", "search-truth-e1", "Cresc", "1000"): 16}
    assert all(r["p"] == 5 and "p_label" not in r for r in by[("direct", "adaptive", "Cresc", "1000")])
    for r in by[("direct", "MCP-up", "Cresc", "1000")]:
        assert 0.0 <= r["bic_f1"] <= r["max_f1"] <= 1.0          # the best of the path bounds its selected graph
        assert 0.0 <= r["search_f1"] <= 1.0 and r["p"] == 5
        assert r["bic_dead"] == 0 and r["search_dead"] == 0 and r["ebic1_dead"] == 0
        assert r["shards"] == 1 and r["n_shards"] == 1
    assert all("max_f1" not in r and "search_dead" in r for r in by[("direct", "search-pure", "Cresc", "1000")])
    assert all("search_f1" not in r for r in by[("direct", "lasso", "C2I", "1000")])      # --select bic: no search
    # the eBIC selection carries the same metrics as every other selected graph, the skeleton too
    for r in by[("direct", "lasso", "Cresc", "1000")]:
        assert not math.isnan(r["ebic1_skeleton_f1"]) and not math.isnan(r["ebic05_skeleton_f1"])
        assert {"ebic1_search_f1", "ebic1_search_skeleton_f1", "ebic1_search_dead"} <= set(r)


def test_means_and_paired_differences(tiny_campaign):
    rows = campaign.load(tiny_campaign)
    means = {(m["estimator"], m["c"]): m for m in campaign.means_table(rows)}
    mcp = [r for r in rows if r["estimator"] == "MCP-up"]
    assert means[("MCP-up", "Cresc")]["graphs"] == 16
    assert math.isclose(means[("MCP-up", "Cresc")]["max_f1"], np.mean([r["max_f1"] for r in mcp]))
    assert math.isnan(means[("search-pure", "Cresc")]["max_f1"])          # no path, no path metric
    assert not math.isnan(means[("search-pure", "Cresc")]["search_f1"])
    assert not math.isnan(means[("lasso", "Cresc")]["ebic1_skeleton_f1"])  # B3: computed, not NaN
    assert means[("lasso", "Cresc")]["bic_dead"] == 0 and means[("lasso", "Cresc")]["search_dead"] == 0
    assert means[("lasso", "Cresc")]["shards"] == 1 and means[("lasso", "Cresc")]["n_shards"] == 1

    paired = {(r["estimator"], r["c"], r["reference"], r["true_c"]): r for r in campaign.paired_table(rows)}
    lasso = {c: {key(r): r for r in rows if r["estimator"] == "lasso" and r["c"] == c} for c in ("C2I", "Cresc")}
    for ref, c in (("same_c", "Cresc"), ("dettling", "C2I")):
        d = [r["max_f1"] - lasso[c][key(r)]["max_f1"] for r in mcp]
        row = paired[("MCP-up", "Cresc", ref, "all")]
        assert row["pairs"] == 16 and math.isclose(row["max_f1_diff"], np.mean(d), abs_tol=1e-12)
    # per true-C setting: 4 graphs each; the lasso is not compared with itself
    assert paired[("MCP-up", "Cresc", "same_c", "C_ID")]["pairs"] == 4
    assert ("lasso", "Cresc", "same_c", "all") not in paired
    assert ("lasso", "Cresc", "dettling", "all") in paired
    # a wave 2 row is compared with the lasso's searched graph
    d = [r["search_f1"] - lasso["Cresc"][key(r)]["search_f1"] for r in rows if r["estimator"] == "search-pure"]
    assert math.isclose(paired[("search-pure", "Cresc", "same_c", "all")]["search_f1_diff"], np.mean(d), abs_tol=1e-12)
    # B1: Dettling's lasso has no search, so the searched graph has no difference to it (it used
    # to be compared with the selected graph under this name)
    assert math.isnan(paired[("MCP-up", "Cresc", "dettling", "all")]["search_f1_diff"])
    assert not math.isnan(paired[("MCP-up", "Cresc", "same_c", "all")]["search_f1_diff"])
    # B3: a search scored with the eBIC penalty is compared with the lasso's search under it
    d = [r["search_f1"] - lasso["Cresc"][key(r)]["ebic1_search_f1"] for r in rows if r["estimator"] == "search-pure-e1"]
    assert math.isclose(paired[("search-pure-e1", "Cresc", "same_c", "all")]["search_f1_diff"], np.mean(d), abs_tol=1e-12)


def test_dead_selections_are_flagged_counted_and_blanked_on_request(tiny_campaign):
    """Code review A3: a selection whose every score is +inf is flagged; --drop-dead blanks its
    graph metrics and nothing else."""
    rows = campaign.load(tiny_campaign)
    assert campaign.dead([math.inf, math.inf]) == 1 and campaign.dead([math.inf, 3.0]) == 0
    r = next(x for x in rows if x["estimator"] == "lasso" and x["c"] == "Cresc").copy()   # has ebic1_search_*
    r["bic_dead"], r["ebic1_search_dead"] = 1, 1
    assert campaign.blank_dead([r]) == 2
    assert math.isnan(r["bic_f1"]) and math.isnan(r["bic_skeleton_f1"]) and math.isnan(r["ebic1_search_f1"])
    assert not math.isnan(r["ebic1_f1"]) and not math.isnan(r["search_f1"])      # other selections untouched
    assert isinstance(r["bic_index"], int)                                        # only the metrics are blanked
    assert campaign.blank_dead(campaign.load(tiny_campaign)) == 0                 # nothing dead here


def test_command_line_writes_the_tables(tiny_campaign):
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "campaign.py"),
                          "--root", str(tiny_campaign), "--drop-dead"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    for name in ("campaign_per_dataset.csv", "campaign_means.csv", "campaign_paired.csv"):
        assert (tiny_campaign / name).stat().st_size > 0
    assert "MCP-up" in res.stdout and "128 rows from 6 cells" in res.stdout
    assert "0 dead selections blanked" in res.stdout and "incomplete:" not in res.stdout


def test_loader_checks_the_shards(tiny_campaign, tmp_path, capsys):
    """Code review B2: a duplicate graph is refused, an unreadable file is named, an incomplete
    cell is reported (stderr and the shards / n_shards columns) but still loaded."""
    name = "direct_MCP-up_Cresc_n1000"
    root = tmp_path / "root"
    shutil.copytree(tiny_campaign / name, root / name)
    cell = root / name
    (shard,) = (cell / "shards").glob("shard_*.npz")
    shutil.copy(shard, cell / "shards" / "shard_0000_of_0002.npz")          # a stale shard of another layout
    with pytest.raises(ValueError, match="appears in both"):
        campaign.load(root)
    (cell / "shards" / "shard_0000_of_0002.npz").unlink()
    (cell / "n_shards").write_text("2\n")                                    # one of two shards present
    rows = campaign.load(root)
    assert len(rows) == 16 and all(r["shards"] == 1 and r["n_shards"] == 2 for r in rows)
    assert "1 of 2 shards present" in capsys.readouterr().err
    means = campaign.means_table(rows)
    assert means[0]["shards"] == 1 and means[0]["n_shards"] == 2
    truncated = cell / "shards" / "shard_0001_of_0002.npz"
    truncated.write_bytes(shard.read_bytes()[:1000])
    with pytest.raises(RuntimeError, match="shard_0001_of_0002.npz"):
        campaign.load(root)


def test_searched_extended_bic_graphs_are_loaded_next_to_the_offline_selection(tmp_path):
    """Wave 5c: a cell run with --ebic-gamma 1 gets ebic1_search_* columns from the stored
    counts, next to the offline ebic1_* selection every cell has."""
    cell = tmp_path / "direct_lasso_Cresc_n1000" / "shards"
    cmd = [sys.executable, str(ROOT / "simulations" / "run_s1_shard.py"), "--shard", "0",
           "--n-shards", "4", "--p", "5", "--reps", "1", "--n-obs", "1000", "--select", "search",
           "--c-scale", "variance", "--ebic-gamma", "1", "--out-dir", str(cell)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    rows = campaign.load(tmp_path)
    assert len(rows) == 4
    for row in rows:
        assert {"ebic1_f1", "ebic1_edges", "ebic1_search_f1", "ebic1_search_edges",
                "ebic1_search_moves", "search_f1"} <= set(row)
        assert 0 <= row["ebic1_search_f1"] <= 1 and "ebic05_search_f1" not in row


def test_graph_class_columns(tiny_campaign, tmp_path):
    """Plan of 13 October 2026, E3 / E4: every row carries the class of its true graph, computed
    from the stored M* and the regenerated C*; the edge table has one row per true edge with the
    recovery flags of the three graphs, and the false-positive table one row per graph and estimate."""
    import graph_class
    from gclm.graphs import CLASSES
    rows = campaign.load(tiny_campaign)
    for r in rows:
        assert r["graph_class"] in CLASSES and r["n_two_cycles"] >= 0 and 0.0 <= r["frac_edges_2c"] <= 1.0
        assert r["identified_min_eig"] >= 0.0
        assert (r["n_two_cycles"] > 0) == (r["graph_class"] == "two_cycles")
    # the same graph has the same class in every cell
    by_graph = {}
    for r in rows:
        by_graph.setdefault(key(r), set()).add((r["graph_class"], r["n_two_cycles"], r["identified_min_eig"]))
    assert all(len(v) == 1 for v in by_graph.values()) and len(by_graph) == 16
    # graph_class.py: within-k tables with paired gains over the lasso, on these p = 5 cells
    campaign.write(tmp_path / "campaign_per_dataset.csv", rows)
    gc_rows = graph_class.load_rows(tmp_path / "campaign_per_dataset.csv", {5})
    out = graph_class.table(gc_rows)
    assert out and all(r["graph_class"] in CLASSES and r["n_graphs"] > 0 for r in out)
    lasso = [r for r in out if r["estimator"] == "lasso" and r["metric"] == "max_f1"]
    assert lasso and all(abs(r["gain"]) < 1e-12 for r in lasso)                # the lasso against itself
    mcp_rows = [r for r in out if r["estimator"] == "MCP-up" and r["metric"] == "max_f1"]
    assert sum(r["n_graphs"] for r in mcp_rows) == 16                          # every graph in exactly one (k, class)
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "graph_class.py"),
                          "--root", str(tiny_campaign), "--p", "5"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "paired gain in max_f1" in res.stdout and (tiny_campaign / "campaign_graph_class.csv").exists()
