"""simulations/diagnostics/campaign.py on the cells of wave 9 (plan of 13 October 2026): estimator
names with a start (a dot allowed), the generator knobs in the cell name, the start diagnostics as
columns, and the grouping of the tables by generator, on a tiny campaign at p = 5."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import campaign  # noqa: E402

CELLS = {
    "direct_lasso_Cresc_n1000": ["--select", "bic"],
    "direct_MCP-up-l1_Cresc_n1000": ["--select", "bic", "--penalty", "MCP", "--direction", "up", "--up-start", "l1"],
    "direct_adaptive-truth0.05_Cresc_n1000": ["--select", "bic", "--method", "adaptive", "--pilot", "truth-0.05"],
    "direct_threshold_Cresc_n1000": ["--select", "search", "--method", "threshold"],
    "direct_lasso_Cresc_w025_n1000": ["--select", "bic", "--weights", "uniform", "--w-min", "0.25"],
    "direct_MCP-up_Cresc_w025_n1000": ["--select", "bic", "--weights", "uniform", "--w-min", "0.25",
                                       "--penalty", "MCP", "--direction", "up"],
    "direct_lasso_Cresc_dag_n1000": ["--select", "bic", "--graph", "dag"],
    # 19 October: the cluster's beta-min cells run every reduction -- the search from the BIC and from the
    # eBIC(gamma) selection (--ebic-gamma); here gamma = 1 only, to keep the fixture small
    "direct_lasso_Cresc_w05_n1000": ["--select", "search", "--ebic-gamma", "1", "--weights", "uniform", "--w-min", "0.5"],
    "direct_MCP-up_Cresc_w05_n1000": ["--select", "search", "--ebic-gamma", "1", "--weights", "uniform", "--w-min", "0.5",
                                      "--penalty", "MCP", "--direction", "up"],
    # E5 (19 October): the share of 2-cycles among the connected pairs
    "direct_lasso_Cresc_s05_n1000": ["--select", "bic", "--graph", "pairs", "--two-cycle-share", "0.5"],
    "direct_MCP-up_Cresc_s05_n1000": ["--select", "bic", "--graph", "pairs", "--two-cycle-share", "0.5",
                                      "--penalty", "MCP", "--direction", "up"],
}


@pytest.fixture(scope="module")
def wave9(tmp_path_factory):
    root = tmp_path_factory.mktemp("wave9")
    common = ["--shard", "0", "--n-shards", "1", "--p", "5", "--reps", "1", "--n-obs", "1000", "--c-scale", "variance"]
    for name, extra in CELLS.items():
        res = subprocess.run([sys.executable, str(ROOT / "simulations" / "run_s1_shard.py"), *common, *extra,
                              "--out-dir", str(root / name / "shards")], capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, res.stderr
        (root / name / "n_shards").write_text("1\n")
    return root


def test_cell_names_carry_the_start_and_the_generator():
    m = campaign.CELL.match("direct_MCP-up-truth0.05_Cresc_n1e4")
    assert m and campaign.cell_meta(m) == {"loss": "direct", "estimator": "MCP-up-truth0.05", "c": "Cresc", "n": "1e4",
                                           **campaign.DGP_DEFAULT}
    for name, w_min, dgp in (("direct_MCP_C2I_w0_n1000", 0.0, "w0"), ("direct_MCP_C2I_w01_n1000", 0.1, "w01"),
                             ("direct_adaptive_Cresc_w025_n1e4", 0.25, "w025"), ("direct_lasso_Cresc_w05_n1000", 0.5, "w05")):
        meta = campaign.cell_meta(campaign.CELL.match(name))
        assert (meta["weights"], meta["w_min"], meta["graph"], meta["dgp"]) == ("uniform", w_min, "random", dgp)
    meta = campaign.cell_meta(campaign.CELL.match("direct_SCAD-up_C2I_dag_n1e4"))
    assert (meta["weights"], meta["w_min"], meta["graph"], meta["dgp"], meta["estimator"]) == ("normal", "", "dag", "dag", "SCAD-up")
    assert campaign.CELL.match("direct_lasso_Cresc_tree_n1000") is None                 # an unknown class token does not parse
    for name, share in (("direct_lasso_Cresc_s05_n1e4", 0.5), ("direct_MCP-up_C2I_s1_ninf", 1.0), ("direct_adaptive_Cresc_s025_n1000", 0.25),
                        ("direct_lasso_C2I_s0_n1000", 0.0)):
        meta = campaign.cell_meta(campaign.CELL.match(name))                              # E5 (19 Oct)
        assert (meta["graph"], meta["two_cycle_share"], meta["weights"], meta["dgp"]) == ("pairs", share, "normal", "s" + name.rsplit("_s", 1)[1].split("_")[0])
    # the old names parse as before, and the p label stays out of the columns
    old = campaign.cell_meta(campaign.CELL.match("loglik_MCP-up-lasso-ml_C2I_p20_n1e4"))
    assert old == {"loss": "loglik", "estimator": "MCP-up-lasso-ml", "c": "C2I", "n": "1e4", **campaign.DGP_DEFAULT}
    assert campaign.CELL.match("direct_lasso_Cresc_w025") is None and campaign.CELL.match("direct__Cresc_n1000") is None


def test_loader_reads_starts_and_generators(wave9):
    rows = campaign.load(wave9)
    by = {}
    for r in rows:
        by.setdefault((r["estimator"], r["dgp"]), []).append(r)
    assert {k: len(v) for k, v in by.items()} == {("lasso", ""): 16, ("MCP-up-l1", ""): 16, ("adaptive-truth0.05", ""): 16,
                                                  ("threshold", ""): 16, ("lasso", "w025"): 16, ("MCP-up", "w025"): 16,
                                                  ("lasso", "dag"): 16, ("lasso", "w05"): 16, ("MCP-up", "w05"): 16,
                                                  ("lasso", "s05"): 16, ("MCP-up", "s05"): 16}
    for r in by[("lasso", "w025")]:
        assert (r["weights"], r["w_min"], r["graph"]) == ("uniform", 0.25, "random") and "start_f1" not in r
    for r in by[("lasso", "dag")]:
        assert (r["weights"], r["w_min"], r["graph"]) == ("normal", "", "dag")
        assert r["graph_class"] == "dag" and r["n_two_cycles"] == 0            # regenerated with the knob
    for r in by[("lasso", "")]:
        assert (r["weights"], r["w_min"], r["graph"], r["dgp"]) == ("normal", "", "random", "")
    # the start columns: the l1 fit is exact and sparse, the noisy truth dense and not exact
    for r in by[("MCP-up-l1", "")]:
        assert r["start_residual"] < 1e-8 and r["start_nnz"] <= 10 and r["start_edges"] == r["start_nnz"]
        assert 0.0 <= r["start_f1"] <= 1.0 and r["start_l1"] >= r["start_l2"] > 0
        assert {"start_precision", "start_recall", "start_skeleton_f1", "start_correct", "start_hedged",
                "start_reversed", "start_missed_single"} <= set(r)
    for r in by[("adaptive-truth0.05", "")]:
        assert r["start_nnz"] == 20 and r["start_residual"] > 1e-4 and r["start_recall"] == 1.0
    for r in by[("threshold", "")]:
        assert r["start_nnz"] >= 1 and 0.0 <= r["bic_f1"] <= r["max_f1"] <= 1.0 and "search_f1" in r
    # the graphs of a knob cell differ from the thesis generator's, those of the start cells do not
    key = lambda r: (r["p"], r["k"], r["c_choice"], r["rep"])  # noqa: E731
    plain = {key(r): r for r in by[("lasso", "")]}
    assert all(plain[key(r)]["n_true_edges"] == r["n_true_edges"] for r in by[("MCP-up-l1", "")])
    assert any(plain[key(r)]["n_true_edges"] != r["n_true_edges"] for r in by[("lasso", "dag")])


def test_tables_group_by_generator(wave9):
    rows = campaign.load(wave9)
    means = {(m["estimator"], m["dgp"]): m for m in campaign.means_table(rows)}
    assert set(means) == {("lasso", ""), ("MCP-up-l1", ""), ("adaptive-truth0.05", ""), ("threshold", ""),
                          ("lasso", "w025"), ("MCP-up", "w025"), ("lasso", "dag"), ("lasso", "w05"), ("MCP-up", "w05"),
                          ("lasso", "s05"), ("MCP-up", "s05")}
    assert all(m["graphs"] == 16 and m["dgp"] in ("", "w025", "w05", "dag", "s05") for m in means.values())
    assert means[("lasso", "w025")]["max_f1"] != means[("lasso", "")]["max_f1"]
    paired = {(r["estimator"], r["dgp"], r["reference"], r["true_c"]): r for r in campaign.paired_table(rows)}
    # the knob cell pairs with the lasso drawn with the same knob, not with the thesis lasso
    key = lambda r: (r["p"], r["k"], r["c_choice"], r["rep"])  # noqa: E731
    lasso_w = {key(r): r for r in rows if r["estimator"] == "lasso" and r["dgp"] == "w025"}
    d = [r["max_f1"] - lasso_w[key(r)]["max_f1"] for r in rows if r["estimator"] == "MCP-up" and r["dgp"] == "w025"]
    row = paired[("MCP-up", "w025", "same_c", "all")]
    assert row["pairs"] == 16 and math.isclose(row["max_f1_diff"], float(np.mean(d)), abs_tol=1e-12)
    assert ("lasso", "w025", "same_c", "all") not in paired and ("lasso", "dag", "same_c", "all") not in paired
    # the start cells pair with the thesis lasso
    lasso = {key(r): r for r in rows if r["estimator"] == "lasso" and r["dgp"] == ""}
    d = [r["max_f1"] - lasso[key(r)]["max_f1"] for r in rows if r["estimator"] == "threshold"]
    assert math.isclose(paired[("threshold", "", "same_c", "all")]["max_f1_diff"], float(np.mean(d)), abs_tol=1e-12)
    assert paired[("MCP-up-l1", "", "same_c", "all")]["pairs"] == 16
    # by_true_c keeps the generator column
    assert all("true_c" in m and "dgp" in m for m in campaign.means_table(rows, by_true_c=True))


def test_step7_analyses_and_figures_on_the_wave9_cells(wave9, tmp_path):
    """starts.py (E1/E2), graph_class.py with a generated class (E4) and betamin.py's knob table (E3),
    on the tiny campaign; then every script and figure script end to end."""
    import csv

    import betamin
    import graph_class
    import starts
    rows = campaign.load(wave9)
    per_dataset = tmp_path / "campaign_per_dataset.csv"
    campaign.write(per_dataset, rows)
    key = lambda r: (r["p"], r["k"], r["c_choice"], r["rep"])  # noqa: E731
    # E1 / E2: the thesis generator's cells only
    srows = starts.load_rows(per_dataset)
    assert {r["estimator"] for r in srows} == {"lasso", "MCP-up-l1", "adaptive-truth0.05", "threshold"}
    st = starts.starts_table(srows)
    assert {(r["pruner"], r["start"]) for r in st} == {("MCP-up", "l1"), ("adaptive", "truth0.05")}
    lasso = {key(r): r for r in srows if r["estimator"] == "lasso"}
    d = [float(r["max_f1"]) - float(lasso[key(r)]["max_f1"]) for r in srows if r["estimator"] == "MCP-up-l1"]
    row = next(r for r in st if r["estimator"] == "MCP-up-l1" and r["metric"] == "max_f1")
    assert row["pairs_lasso"] == 16 and math.isclose(row["gain_lasso"], float(np.mean(d)), abs_tol=1e-12)
    assert row["pairs_lasso_start"] == 0 and math.isnan(row["gain_lasso_start"])    # no plain MCP-up cell here
    content = starts.content_table(srows)
    c1 = next(r for r in content if r["estimator"] == "MCP-up-l1")
    assert c1["n_graphs"] == 16 and abs(c1["true_alone"] + c1["both"] + c1["reverse_alone"] + c1["neither"] - 1) < 1e-9
    assert c1["start_residual"] < 1e-8 and c1["start_nnz"] <= 10 and 0.0 <= c1["orientation_recall"] <= 1.0
    ct = next(r for r in content if r["estimator"] == "adaptive-truth0.05")
    assert ct["both"] == 1.0 and ct["orientation_recall"] == 1.0 and ct["start_nnz"] == 20
    th = starts.threshold_table(srows)
    assert {r["estimator"] for r in th} == {"threshold", "lasso"}
    thr = {key(r): r for r in srows if r["estimator"] == "threshold"}
    d = [float(lasso[g]["max_f1"]) - float(thr[g]["max_f1"]) for g in thr]
    t_lasso = next(r for r in th if r["estimator"] == "lasso" and r["metric"] == "max_f1")
    assert t_lasso["pairs"] == 16 and math.isclose(t_lasso["diff_vs_threshold"], float(np.mean(d)), abs_tol=1e-12)
    assert all(r["diff_vs_threshold"] == 0.0 for r in th if r["estimator"] == "threshold")
    # E4: the generated class is its own generator, compared with the lasso of the same generator
    gc_rows = graph_class.load_rows(per_dataset, {5})
    assert {r["generator"] for r in gc_rows} == {"random", "dag"}
    assert not any(r["cell"].endswith("_w025_n1000") for r in gc_rows)         # the knob cells stay out
    gt = graph_class.table(gc_rows)
    dag = [r for r in gt if r["generator"] == "dag"]
    assert dag and all(r["graph_class"] == "dag" and r["identified_share"] == 1.0 and r["estimator"] == "lasso" for r in dag)
    assert all(abs(r["gain"]) < 1e-12 for r in dag if r["metric"] != "search_f1")   # the lasso against itself ...
    assert all(math.isnan(r["gain"]) for r in dag if r["metric"] == "search_f1")     # ... and no search in that cell
    gens = graph_class.table_by_generator(gc_rows)
    assert all(r["graph_class"] == "all" for r in gens)
    assert sum(r["n_graphs"] for r in gens if r["generator"] == "dag" and r["metric"] == "max_f1") == 16
    # E3: the knob table pairs the w025 cells with each other and keeps N(0, 1) as "normal"
    knob = betamin.wmin_table(list(csv.DictReader(per_dataset.open())))
    assert {r["w_min"] for r in knob} == {"normal", "0.25", "0.5"}
    w_lasso = {key(r): r for r in rows if r["estimator"] == "lasso" and r["dgp"] == "w025"}
    d = [r["max_f1"] - w_lasso[key(r)]["max_f1"] for r in rows if r["estimator"] == "MCP-up" and r["dgp"] == "w025"]
    row = next(r for r in knob if (r["w_min"], r["estimator"], r["metric"]) == ("0.25", "MCP-up", "max_f1"))
    assert row["n_graphs"] == 16 and math.isclose(row["gain"], float(np.mean(d)), abs_tol=1e-12)
    assert {r["estimator"] for r in knob if r["w_min"] == "normal"} == {"lasso"}
    assert {r["metric"] for r in knob} == set(betamin.KNOB_METRICS)
    # every reduction: the eBIC selections are decoded for every cell, the searches only where they ran
    g = lambda w, est, metric: next(r for r in knob if (r["w_min"], r["estimator"], r["metric"]) == (w, est, metric))  # noqa: E731
    assert not math.isnan(g("0.25", "MCP-up", "ebic1_f1")["gain"]) and math.isnan(g("0.25", "MCP-up", "search_f1")["gain"])
    for metric in ("bic_f1", "search_f1", "ebic05_f1", "ebic1_f1", "ebic1_search_f1"):
        assert not math.isnan(g("0.5", "MCP-up", metric)["gain"]), metric
    assert math.isnan(g("0.5", "MCP-up", "ebic05_search_f1")["gain"])                # gamma = 1 only in the fixture
    se = next(r for r in knob if (r["w_min"], r["estimator"], r["metric"]) == ("0.25", "lasso", "bic_f1"))
    vals = [r["bic_f1"] for r in rows if r["estimator"] == "lasso" and r["dgp"] == "w025"]
    assert math.isclose(se["mean"], float(np.mean(vals))) and math.isclose(se["mean_se"], float(np.std(vals, ddof=1) / 4))
    # E3: the orientation of the true single edges by weight law, from the stored orientation counts
    orient = betamin.orientation_by_wmin(list(csv.DictReader(per_dataset.open())))
    assert {r["rule"] for r in orient if r["w_min"] == "0.25"} == {"bic", "ebic05", "ebic1"}          # --select bic: no search
    assert {r["rule"] for r in orient if r["w_min"] == "0.5"} == {"bic", "search", "ebic05", "ebic1", "ebic1_search"}
    assert all(abs(r["correct"] + r["hedged"] + r["reversed"] + r["missed"] - 1) < 1e-9 for r in orient)
    mcp = [r for r in rows if r["estimator"] == "MCP-up" and r["dgp"] == "w025" and r["n_true_edges"] - 2 * r["n_two_cycles"] > 0]
    fp = [r["bic_hedged"] + r["bic_reversed"] + r["bic_fp_single"] + 2 * r["bic_fp_double"] for r in mcp]
    assert all(r["bic_edges"] - round(r["bic_recall"] * r["n_true_edges"]) == x for r, x in zip(mcp, fp))   # fp = edges - TP
    row = next(r for r in orient if (r["w_min"], r["estimator"], r["rule"]) == ("0.25", "MCP-up", "bic"))
    assert row["n_graphs"] == len(mcp) and math.isclose(row["fp"], float(np.mean(fp))) and \
        math.isclose(row["correct"], float(np.mean([r["bic_correct"] / (r["n_true_edges"] - 2 * r["n_two_cycles"]) for r in mcp])))
    # E3: one row per true edge from the shards; the outcomes reproduce the stored orientation counts graph by graph
    edges = betamin.edge_outcome_tables(wave9)
    assert {e["estimator"] for e in edges} == {"lasso", "MCP-up"} and {e["w_min"] for e in edges} == {"normal", "0.25", "0.5"}
    per_graph = {}
    for e in edges:
        per_graph.setdefault((e["cell"], e["p"], e["k"], e["c_choice"], e["rep"]), []).append(e)
    checked = 0
    for r in rows:
        if r["estimator"] not in ("lasso", "MCP-up") or r["dgp"] not in ("", "w025", "w05"):
            continue
        mine = per_graph[(r["cell"], r["p"], r["k"], r["c_choice"], r["rep"])]
        single = [e for e in mine if e["two_cycle"] == 0]
        assert len(mine) == r["n_true_edges"] and len(single) == r["n_true_edges"] - 2 * r["n_two_cycles"]
        ran = ("bic", "ebic05", "ebic1") + (("search", "ebic1_search") if r["dgp"] == "w05" else ())
        for rule in betamin.KNOB_RULES:
            if rule not in ran:
                assert all(e[rule] == "" for e in mine), rule                         # the cell did not run it
                continue
            for name, col in betamin.ORIENT_PARTS:                                    # the per-edge fates sum to the stored counts
                assert sum(e[rule] == name for e in single) == r[f"{rule}_{col}"], (rule, name)
            assert sum(e[rule] == "present" for e in mine if e["two_cycle"] == 1) == 2 * r[f"{rule}_both"] + r[f"{rule}_half"]
        checked += 1
    assert checked == 80
    # the standardised weight is |M_ij| s_j / s_i with the scale the shard stores
    d = np.load(next((wave9 / "direct_lasso_Cresc_w025_n1000" / "shards").glob("*.npz")), allow_pickle=True)
    scale = np.asarray(d["scale"][0], dtype=float).reshape(-1)
    e0 = next(e for e in edges if e["cell"] == "direct_lasso_Cresc_w025_n1000" and
              (e["k"], e["c_choice"], e["rep"]) == (int(d["k"][0]), str(d["c_choice_names"][int(d["c_choice"][0])]), int(d["rep"][0])))
    assert math.isclose(e0["abs_w_std"], e0["abs_w"] * scale[e0["j"]] / scale[e0["i"]]) and 0.25 <= e0["abs_w"] <= 1.0
    assert betamin.outcome(True, False, False) == "correct" and betamin.outcome(True, True, False) == "hedged"
    assert betamin.outcome(False, True, False) == "reversed" and betamin.outcome(False, False, False) == "missed"
    assert betamin.outcome(True, True, True) == "present" and betamin.outcome(False, True, True) == "missed"
    assert betamin.weight_bin(0.0) == "0-0.1" and betamin.weight_bin(0.1) == "0.1-0.25" and betamin.weight_bin(0.75) == "0.75-1"
    assert betamin.weight_bin(1.0) == "1-2" and betamin.weight_bin(7.0) == "2+"
    # the within-law bins: shares of the single edges with an outcome, pooled over the edges of the bin
    bins = betamin.binned_orientation(edges)
    assert {b["var"] for b in bins} == {"abs_w"} and all(b["weight_bin"] in betamin.BIN_LABELS for b in bins)
    assert {b["rule"] for b in bins if b["w_min"] == "0.5"} == {"bic", "search", "ebic05", "ebic1", "ebic1_search"}
    assert all(abs(b["correct"] + b["hedged"] + b["reversed"] + b["missed"] - 1) < 1e-9 for b in bins)
    lasso25 = [e for e in edges if (e["estimator"], e["w_min"], e["two_cycle"]) == ("lasso", "0.25", 0)]
    assert sum(b["n_edges"] for b in bins if (b["w_min"], b["estimator"], b["rule"]) == ("0.25", "lasso", "bic")) == len(lasso25)
    assert not any(b["weight_bin"] in ("0-0.1", "0.1-0.25", "1-2", "2+") for b in bins if b["w_min"] == "0.25")   # U[0.25, 1]
    b1 = next(b for b in bins if (b["w_min"], b["estimator"], b["rule"], b["weight_bin"]) == ("0.25", "lasso", "bic", "0.5-0.75"))
    mine = [e for e in lasso25 if 0.5 <= e["abs_w"] < 0.75]
    assert b1["n_edges"] == len(mine) and math.isclose(b1["hedged"], sum(e["bic"] == "hedged" for e in mine) / len(mine))
    std = betamin.binned_orientation(edges, "abs_w_std")
    assert std and {b["var"] for b in std} == {"abs_w_std"}
    # the scripts and the figure scripts run end to end on the campaign root
    diag = ROOT / "simulations" / "diagnostics"
    for script, extra in (("campaign.py", []), ("starts.py", []), ("graph_class.py", ["--p", "5"]), ("betamin.py", []),
                          ("plot_starts.py", []), ("plot_graph_class.py", []), ("plot_betamin.py", [])):
        res = subprocess.run([sys.executable, str(diag / script), "--root", str(wave9), *extra],
                             capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, (script, res.stderr)
    for name in ("campaign_starts.csv", "campaign_start_content.csv", "campaign_threshold.csv", "campaign_graph_class.csv",
                 "campaign_generator.csv", "campaign_wmin.csv", "campaign_wmin_orientation.csv", "campaign_edge_outcomes.csv",
                 "campaign_wmin_bins.csv"):
        assert (wave9 / name).exists(), name
    for fig in ("starts_bars.png", "starts_by_pruner_n1000.png", "threshold_vs_adaptive.png", "gain_by_generator_p5.png",
                "gain_by_wmin.png", "gain_by_wmin_ebic1_search_f1.png", "levels_by_wmin_n1000.png",
                "levels_by_wmin_n1000_ebic1_search.png", "orientation_by_wmin_n1000_bic.png",
                "orientation_by_wmin_n1000_ebic1_search.png", "orientation_by_weight_p5_Cresc_n1000_bic.png",
                "orientation_by_weight_p5_Cresc_n1000_bic_std.png", "orientation_bin_across_laws_p5_Cresc_n1000_bic.png",
                "orientation_by_weight_p5_Cresc_n1000_ebic1.png"):
        assert (wave9 / "figures" / fig).exists(), fig
    assert not (wave9 / "figures" / "gain_by_wmin_ebic05_search_f1.png").exists()       # gamma = 0.5 search not run here


def test_two_cycle_share_tables_and_figures(wave9, tmp_path):
    """E5 (19 October 2026): the share knob's tables pair the s05 cells with their own lasso, keep the
    thesis generator as the ``random`` reference and leave the E3 and E4 cells out; the fates of the true
    2-cycles come from the stored counts; the per-edge table reproduces them; the scripts run end to end."""
    import csv

    import betamin
    import twocycles
    rows = campaign.load(wave9)
    per_dataset = tmp_path / "campaign_per_dataset.csv"
    campaign.write(per_dataset, rows)
    key = lambda r: (r["p"], r["k"], r["c_choice"], r["rep"])  # noqa: E731
    table = betamin.knob_table(list(csv.DictReader(per_dataset.open())), twocycles.E5_ESTIMATORS, knob=twocycles.SHARE)
    assert {r["share"] for r in table} == {"random", "0.5"}
    assert {r["estimator"] for r in table if r["share"] == "random"} == {"lasso"}       # the start cells are not E5's
    lasso_s = {key(r): r for r in rows if r["estimator"] == "lasso" and r["dgp"] == "s05"}
    d = [r["max_f1"] - lasso_s[key(r)]["max_f1"] for r in rows if r["estimator"] == "MCP-up" and r["dgp"] == "s05"]
    row = next(r for r in table if (r["share"], r["estimator"], r["metric"]) == ("0.5", "MCP-up", "max_f1"))
    assert row["n_graphs"] == 16 and math.isclose(row["gain"], float(np.mean(d)), abs_tol=1e-12)
    # E3's tables do not see the pairs cells, E5's do not see the uniform-weight cells
    assert "0.5" in {r["w_min"] for r in betamin.wmin_table(list(csv.DictReader(per_dataset.open())))}
    assert not any(r["dgp"] == "s05" for r in rows if betamin.WMIN.select(r) and r["estimator"] == "lasso" and r["dgp"] == "s05")
    assert all(r["dgp"] in ("", "s05") for r in rows if twocycles.SHARE.select(r))
    # the fates of the true 2-cycles, from the stored counts of the selected graph
    orient = betamin.orientation_by_knob(list(csv.DictReader(per_dataset.open())), twocycles.E5_ESTIMATORS, knob=twocycles.SHARE)
    o = next(r for r in orient if (r["share"], r["estimator"], r["rule"]) == ("0.5", "MCP-up", "bic"))
    mcp = [r for r in rows if r["estimator"] == "MCP-up" and r["dgp"] == "s05"]
    with_2c = [r for r in mcp if r["n_two_cycles"] > 0]
    assert o["n_graphs_2c"] == len(with_2c) > 0 and abs(o["both"] + o["half"] + o["neither"] - 1) < 1e-9
    assert math.isclose(o["both"], float(np.mean([r["bic_both"] / r["n_two_cycles"] for r in with_2c])))
    assert math.isclose(o["two_cycles"], float(np.mean([r["n_two_cycles"] for r in mcp])))
    assert 0.0 <= o["identified"] <= 1.0 and math.isclose(o["identified"], float(np.mean([r["identified_min_eig"] > 1e-10 for r in mcp])))
    # one row per true edge; the 2-cycle edges' fates reproduce both / half per graph
    edges = betamin.edge_outcome_tables(wave9, twocycles.E5_ESTIMATORS, knob=twocycles.SHARE)
    assert {e["share"] for e in edges} == {"random", "0.5"} and {e["estimator"] for e in edges} == {"lasso", "MCP-up"}
    for r in mcp:
        mine = [e for e in edges if e["cell"] == r["cell"] and key(e) == key(r)]
        assert len(mine) == r["n_true_edges"]
        assert sum(e["bic"] == "present" for e in mine if e["two_cycle"] == 1) == 2 * r["bic_both"] + r["bic_half"]
        assert sum(e["bic"] == "hedged" for e in mine if e["two_cycle"] == 0) == r["bic_hedged"]
    # the scripts and the figure script run end to end
    diag = ROOT / "simulations" / "diagnostics"
    for script in ("campaign.py", "twocycles.py", "plot_twocycles.py"):
        res = subprocess.run([sys.executable, str(diag / script), "--root", str(wave9)], capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, (script, res.stderr)
    for name in ("campaign_share.csv", "campaign_share_orientation.csv", "campaign_share_edges.csv"):
        assert (wave9 / name).exists(), name
    for fig in ("gain_by_share.png", "levels_by_share_n1000.png", "orientation_by_share_n1000_bic.png",
                "pair_outcomes_by_share_p5_Cresc_n1000_bic.png"):
        assert (wave9 / "figures" / fig).exists(), fig
    assert not (wave9 / "figures" / "gain_by_share_search_f1.png").exists()                 # --select bic: no search
