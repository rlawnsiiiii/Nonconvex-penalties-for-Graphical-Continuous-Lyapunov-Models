"""The illustrations of docs/GRAPH_CLASSES_AND_BETAMIN.md (simulations/diagnostics/plot_generator_knobs.py)
and the 2-cycle recovery table of graph_class.py with its figure (17 October 2026)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import graph_class  # noqa: E402
import plot_generator_knobs as knobs  # noqa: E402
from gclm.graphs import graph_class as class_of, is_dag, two_cycles  # noqa: E402

ORIENT_ZERO = {f"bic_{k}": "0" for k in graph_class.ORIENT}


def test_example_supports_show_the_three_classes():
    mats = knobs.example_supports(seed=3, p=8, k=2)
    assert list(mats) == ["random", "dag"]
    assert two_cycles(mats["random"] != 0) >= 1 and class_of(mats["random"] != 0) == "two_cycles"
    assert is_dag(mats["dag"] != 0)
    # the shares of weak edges the script prints: half-normal against the uniform laws
    assert knobs.share_below("normal", 0.1) == pytest.approx(0.0797, abs=1e-3)
    assert knobs.share_below("normal", 2.0) == pytest.approx(0.9545, abs=1e-3)
    assert knobs.share_below(0.25, 0.25) == 0.0 and knobs.share_below(0.25, 1.0) == 1.0
    assert knobs.share_below(0.0, 0.5) == pytest.approx(0.5)
    x = np.linspace(0, 3, 3001)
    assert np.trapezoid(knobs.density("normal", x), x) == pytest.approx(1.0, abs=1e-2)
    assert np.trapezoid(knobs.density(0.5, x), x) == pytest.approx(1.0, abs=1e-2)


def test_figure_script_end_to_end(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "plot_generator_knobs.py"),
                          "--out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    for name in ("generator_classes.png", "generator_classes.pdf", "weight_laws.png", "weight_laws.pdf",
                 "dag_construction.png", "dag_construction.pdf"):
        assert (tmp_path / name).exists(), name
    assert "share of true edges with |w| below x" in res.stdout and "U[0.5, 1]" in res.stdout
    assert "the dag branch, seed 7, p = 6, k = 1" in res.stdout
    assert "nodes in the order drawn (position 1 .. p): 3  2  4  6  5  1" in res.stdout
    assert "edge 3 -> 2   (position 1 -> 2)" in res.stdout and "edges kept: 5" in res.stdout


def test_dag_construction_steps_reproduce_the_generator():
    """The replayed steps of the ``dag`` branch give ``sample_drift``'s support exactly; half of the
    ordered pairs are open slots; sorted by the order the support is strictly lower triangular, and
    the walk along any edge climbs the positions -- the picture of "no directed cycle"."""
    for seed, p, k in ((7, 6, 1), (0, 10, 2), (5, 12, 6), (1, 8, 7)):
        st = knobs.dag_construction_steps(seed, p, k)
        d = k / p
        assert st["d"] == d and st["d_prime"] == min(2 * d, 1.0)
        omega, allowed, position, order = st["omega"], st["allowed"], st["position"], st["order"]
        support = st["m"] != 0
        np.fill_diagonal(support, False)
        assert np.array_equal(support, omega.astype(bool))
        assert allowed.sum() == p * (p - 1) // 2 and not np.any(np.diag(allowed))
        assert np.all(allowed | allowed.T | np.eye(p, dtype=bool)) and not np.any(allowed & allowed.T)
        assert np.all(omega <= allowed)
        sorted_omega = omega[np.ix_(order, order)]
        assert np.array_equal(sorted_omega, np.tril(sorted_omega, -1))
        for i, j in zip(*np.nonzero(omega)):                       # the edge j -> i
            assert position[j] < position[i]
        assert is_dag(omega.astype(bool)) and two_cycles(omega.astype(bool)) == 0
        assert np.linalg.matrix_power(omega, p).sum() == 0          # nilpotent: no walk of length p
        if 2 * d >= 1:                                              # the cap: every open slot is an edge
            assert np.array_equal(omega, allowed.astype(int))
    st = knobs.dag_construction_steps(7, 6, 1)                     # the example of the note
    assert [int(v) + 1 for v in st["order"]] == [3, 2, 4, 6, 5, 1]
    assert sorted((int(j) + 1, int(i) + 1) for i, j in zip(*np.nonzero(st["omega"]))) == \
        [(2, 1), (2, 4), (2, 6), (3, 2), (6, 1)]


def test_dag_construction_steps_refuse_a_replay_that_does_not_match(monkeypatch):
    """The replay is checked against the generator itself: if the generator returned something else
    (here: an empty graph), the helper raises instead of illustrating the wrong thing."""
    monkeypatch.setattr(knobs, "sample_drift", lambda p, d, rng, **kw: -np.eye(p))
    with pytest.raises(AssertionError, match="do not reproduce"):
        knobs.dag_construction_steps(7, 6, 1)


def test_print_dag_steps_text(capsys):
    st = knobs.dag_construction_steps(7, 6, 1)
    knobs.print_dag_steps(st)
    out = capsys.readouterr().out
    assert "open slots: 15 of 30 ordered pairs;  edges kept: 5" in out
    assert "node 3 (pi = 1)   -   -   -   -   -   -" in out
    assert "node 1 (pi = 6)   0   1   0   1   0   -" in out


def per_dataset_rows() -> list[dict]:
    """Two estimators on the same two graphs with the orientation counts of their selected graphs;
    the searched graph's counts only for the lasso.  Graph 1: 10 true edges, 2 of them 2-cycles (so
    6 single edges); graph 2: 7 true edges, one 2-cycle (5 single edges)."""
    base = {"loss": "direct", "c": "Cresc", "n": "1000", "p": "5", "c_choice": "C_ID", "generator": "random"}
    graphs = ({"k": "2", "rep": "0", "n_true_edges": "10", "n_two_cycles": "2"},
              {"k": "1", "rep": "1", "n_true_edges": "7", "n_two_cycles": "1"})
    counts = {                    # (both, half, missed_double, correct, hedged, reversed, missed_single)
        ("lasso", 0): (1, 1, 0, 3, 2, 1, 0), ("lasso", 1): (0, 1, 0, 2, 1, 1, 1),
        ("MCP-up", 0): (0, 2, 0, 5, 0, 1, 0), ("MCP-up", 1): (0, 0, 1, 3, 0, 1, 1),
    }
    rows = []
    for (est, g), (both, half, md, corr, hedged, rev, ms) in counts.items():
        row = {**base, **graphs[g], "estimator": est, **ORIENT_ZERO,
               "bic_both": str(both), "bic_half": str(half), "bic_missed_double": str(md), "bic_correct": str(corr),
               "bic_hedged": str(hedged), "bic_reversed": str(rev), "bic_missed_single": str(ms)}
        if est == "lasso":
            row.update({f"search_{k}": row[f"bic_{k}"] for k in graph_class.ORIENT})
        else:
            row.update({f"search_{k}": "" for k in graph_class.ORIENT})
        rows.append(row)
    return rows


def test_two_cycle_table_means_per_graph():
    out = graph_class.two_cycle_table(per_dataset_rows(), estimators=("lasso", "MCP-up"))
    assert [(r["estimator"], r["rule"]) for r in out] == [("lasso", "bic"), ("lasso", "search"), ("MCP-up", "bic")]
    lasso = out[0]
    assert lasso["n_graphs"] == 2 and lasso["two_cycles"] == 1.5 and lasso["true_edges"] == 8.5 and lasso["single_edges"] == 5.5
    assert (lasso["both"], lasso["half"], lasso["neither"]) == (0.5, 1.0, 0.0)
    assert (lasso["correct"], lasso["hedged"], lasso["reversed"], lasso["missed"]) == (2.5, 1.5, 1.0, 0.5)
    mcp = out[2]
    assert (mcp["both"], mcp["half"], mcp["neither"]) == (0.0, 1.0, 0.5)
    assert (mcp["correct"], mcp["hedged"], mcp["reversed"], mcp["missed"]) == (4.0, 0.0, 1.0, 0.5)
    for r in out:                                   # the parts add up to the true pairs of each kind
        assert math.isclose(r["both"] + r["half"] + r["neither"], r["two_cycles"])
        assert math.isclose(r["correct"] + r["hedged"] + r["reversed"] + r["missed"], r["single_edges"])
    assert out[1]["both"] == lasso["both"]          # the lasso's searched graph carries the same counts here
    assert graph_class.two_cycle_table(per_dataset_rows(), estimators=("SCAD",)) == []


def test_fig_two_cycles_draws(tmp_path):
    import plot_graph_class

    table = graph_class.two_cycle_table(per_dataset_rows(), estimators=("lasso", "MCP-up"))
    plot_graph_class.fig_two_cycles(table, tmp_path, p="5", n="1000", c="Cresc", rule="bic", ests=("lasso", "MCP-up"))
    assert (tmp_path / "two_cycle_recovery_p5_n1000_Cresc_bic_random.png").exists()
    plot_graph_class.fig_two_cycles(table, tmp_path, p="5", n="1e4")           # no such cell: nothing drawn
    assert not (tmp_path / "two_cycle_recovery_p5.png").exists()
