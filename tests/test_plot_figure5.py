"""simulations/diagnostics/plot_figure5.py: Dettling's Figure 5 layout for every estimator, drawn from a
small synthetic means table (the real one is runs/campaign/campaign_means_by_true_c.csv)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import plot_figure5  # noqa: E402

TRUE_C = ("C_ID", "C_Random_Diag", "C_Random_Min_Diag", "C_Random_Full")


def write_means(path: Path, ests=("lasso", "MCP", "adaptive"), ps=(10, 15, 20)):
    rng = np.random.default_rng(0)
    rows = []
    for est in ests:
        for c in ("C2I", "Cresc"):
            for true_c in TRUE_C:
                for p in ps:
                    row = {"loss": "direct", "estimator": est, "c": c, "n": "1000", "p": str(p), "dgp": "",
                           "graphs": "100", "shards": "4", "n_shards": "4", "true_c": true_c}
                    for m in ("max_f1", "aupr", "auc", "max_acc", "bic_f1"):
                        row[m], row[f"{m}_se"] = f"{rng.uniform(0.3, 0.9):.4f}", f"{rng.uniform(0.005, 0.02):.4f}"
                    rows.append(row)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_figures_are_drawn_for_every_estimator_and_the_grid(tmp_path):
    rows = write_means(tmp_path / "campaign_means_by_true_c.csv")
    ref = tmp_path / "reference.csv"
    with ref.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "c_choice", "p", "dettling", "dettling_se"])
        w.writeheader()
        for m in ("max_f1", "aupr"):
            for true_c in TRUE_C:
                for p in (10, 15, 20):
                    w.writerow({"metric": m, "c_choice": true_c, "p": p, "dettling": 0.5, "dettling_se": 0.01})
    loaded = plot_figure5.load(tmp_path)
    assert len(loaded) == len(rows) and {r["estimator"] for r in loaded} == {"lasso", "MCP", "adaptive"}
    reference = plot_figure5.load_reference(ref)
    assert reference[("max_f1", "C_ID", 10)] == (0.5, 0.01) and plot_figure5.load_reference(tmp_path / "missing.csv") == {}
    out = tmp_path / "figures" / "figure5"
    plot_figure5.fig_estimator(loaded, "lasso", "C2I", out, reference)
    plot_figure5.fig_estimator(loaded, "MCP", "Cresc", out)
    plot_figure5.fig_estimator(loaded, "SCAD", "C2I", out)                 # absent estimator: nothing drawn
    plot_figure5.fig_grid(loaded, "Cresc", out, ests=("lasso", "MCP", "adaptive"))
    assert (out / "figure5_lasso_C2I.png").exists() and (out / "figure5_MCP_Cresc.png").exists()
    assert not (out / "figure5_SCAD_C2I.png").exists() and (out / "figure5_grid_Cresc.png").exists()
    # the script end to end, with and without the reference
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "plot_figure5.py"), "--root",
                          str(tmp_path), "--reference", str(ref)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert (out / "figure5_grid_C2I.png").exists() and (out / "figure5_adaptive_Cresc.pdf").exists()
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "plot_figure5.py"), "--root",
                          str(tmp_path), "--reference", "none"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    # without the table of figure5_table.py the per-estimator and gain figures are skipped, with a hint
    assert "campaign_figure5.csv is missing" in res.stdout and not (out / "figure5_gain_C2I.png").exists()


def write_table(path: Path, ests=("lasso", "MCP", "adaptive"), ps=(10, 15, 20)):
    """A synthetic campaign_figure5.csv (figure5_table.py's long table): one row per C, true-C setting
    (pooled + four), p, estimator and metric."""
    rng = np.random.default_rng(1)
    rows = []
    for c in ("C2I", "Cresc"):
        for true_c in ("all", *TRUE_C):
            for p in ps:
                for est in ests:
                    for m in ("max_acc", "max_f1", "auc", "aupr"):
                        gain = 0.0 if est == "lasso" else rng.normal(0, 0.05)
                        rows.append({"c": c, "n": "1000", "true_c": true_c, "p": str(p), "estimator": est,
                                     "graphs": "400" if true_c == "all" else "100", "metric": m,
                                     "mean": f"{rng.uniform(0.3, 0.9):.4f}", "se": f"{rng.uniform(0.005, 0.02):.4f}",
                                     "gain": f"{gain:.4f}", "gain_se": "0.0" if est == "lasso" else "0.01",
                                     "z": "0.0" if est == "lasso" else f"{gain / 0.01:.2f}"})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_estimator_comparison_and_gain_figures_from_the_table(tmp_path):
    out = tmp_path / "figures" / "figure5"
    assert plot_figure5.load_table(tmp_path) == []                       # no table yet
    write_table(tmp_path / "campaign_figure5.csv")
    tab = plot_figure5.load_table(tmp_path, "C2I")
    assert {r["c"] for r in tab} == {"C2I"} and {r["true_c"] for r in tab} == {"all", *TRUE_C}
    assert len(plot_figure5.load_table(tmp_path)) == 2 * len(tab) and plot_figure5.load_table(tmp_path, n="1e4") == []
    ests = ("lasso", "MCP", "adaptive")
    plot_figure5.fig_estimators(tab, "C2I", out, ests=ests)
    plot_figure5.fig_estimators(tab, "C2I", out, ests=ests, true_c="C_Random_Full")
    plot_figure5.fig_gain(tab, "C2I", out, ests=ests)
    plot_figure5.fig_gain(tab, "Cresc", out)                              # no Cresc rows loaded: nothing drawn
    assert (out / "figure5_estimators_C2I.png").exists() and (out / "figure5_estimators_C2I_C_Random_Full.png").exists()
    assert (out / "figure5_gain_C2I.png").exists() and not (out / "figure5_gain_Cresc.png").exists()
    # the script end to end draws both C from the table
    write_means(tmp_path / "campaign_means_by_true_c.csv")
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "plot_figure5.py"), "--root",
                          str(tmp_path), "--reference", "none"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "is missing" not in res.stdout
    assert (out / "figure5_gain_Cresc.png").exists() and (out / "figure5_estimators_Cresc.pdf").exists()
