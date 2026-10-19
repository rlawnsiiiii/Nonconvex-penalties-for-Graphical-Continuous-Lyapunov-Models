"""simulations/diagnostics/figure5_table.py: Dettling's whole-path summaries per estimator with the paired gain
over the lasso, from a small synthetic per-dataset table (the real one is runs/campaign/campaign_per_dataset.csv)."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))

import figure5_table  # noqa: E402

TRUE_C = ("C_ID", "C_Random_Diag", "C_Random_Min_Diag", "C_Random_Full")
SHIFT = {"lasso": 0.0, "adaptive": 0.1, "MCP": -0.05}


def per_dataset_rows(ps=(10, 20), reps=3, ests=tuple(SHIFT), c="C2I", n="1000", seed=0):
    """Synthetic per-graph rows: every estimator's metrics are the lasso's shifted by a known amount plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in ps:
        for k in (1, 2):
            for true_c in TRUE_C:
                for rep in range(reps):
                    base = {m: rng.uniform(0.4, 0.6) for m in figure5_table.METRICS}
                    for est in ests:
                        row = {"cell": f"direct_{est}_{c}_n{n}", "loss": "direct", "estimator": est, "c": c, "n": n,
                               "dgp": "", "p": str(p), "k": str(k), "c_choice": true_c, "rep": str(rep)}
                        for m in figure5_table.METRICS:
                            row[m] = f"{base[m] + SHIFT[est] + rng.normal(0, 0.01):.6f}"
                        rows.append(row)
    return rows


def test_table_pairs_every_estimator_with_the_lasso_on_the_same_graphs():
    rows = per_dataset_rows()
    out = figure5_table.table(rows, "C2I", "1000")
    assert len(out) == 5 * 2 * 3 * 4                      # settings x p x estimators x metrics
    pooled = {(r["estimator"], r["metric"]): r for r in out if (r["true_c"], r["p"]) == ("all", 10)}
    assert pooled[("lasso", "max_f1")]["graphs"] == 2 * 4 * 3 and pooled[("lasso", "max_f1")]["gain"] == 0.0
    for est, shift in SHIFT.items():
        if est == "lasso":
            continue
        for m in figure5_table.METRICS:
            r = pooled[(est, m)]
            assert abs(r["gain"] - shift) < 0.01 and 0 < r["gain_se"] < 0.01 and abs(r["z"]) > 5
    one = [r for r in out if (r["true_c"], r["p"], r["estimator"], r["metric"]) == ("C_ID", 20, "adaptive", "aupr")]
    assert len(one) == 1 and one[0]["graphs"] == 2 * 3 and abs(one[0]["gain"] - 0.1) < 0.02
    # the other C or n, and a generator knob, are left out
    assert figure5_table.table(rows, "Cresc", "1000") == [] and figure5_table.table(rows, "C2I", "1e4") == []
    assert figure5_table.table([{**r, "dgp": "w025"} for r in rows]) == []
    # an estimator without a lasso partner keeps its means, with an empty gain
    alone = figure5_table.table([r for r in rows if r["estimator"] != "lasso"], "C2I", "1000")
    r = next(r for r in alone if (r["estimator"], r["true_c"], r["metric"]) == ("adaptive", "all", "max_acc"))
    assert r["graphs"] == 24 and math.isnan(r["gain"])


def test_script_writes_the_table_and_prints(tmp_path, capsys):
    rows = per_dataset_rows()
    with (tmp_path / "campaign_per_dataset.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure5_table.print_table(figure5_table.table(rows), "max_f1")
    text = capsys.readouterr().out
    assert "max_f1: C2I, n = 1000, true C = all" in text and "adaptive" in text and "+0.1" in text and "-0.05" in text
    res = subprocess.run([sys.executable, str(ROOT / "simulations" / "diagnostics" / "figure5_table.py"), "--root",
                          str(tmp_path), "--true-c", "C_Random_Full"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    written = list(csv.DictReader((tmp_path / "campaign_figure5.csv").open()))
    assert len(written) == 5 * 2 * 3 * 4 and set(written[0]) == set(figure5_table.COLUMNS)
    assert "true C = C_Random_Full" in res.stdout
