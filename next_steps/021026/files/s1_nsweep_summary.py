"""Join the p = 10 n-sweep pilot with the committed n = 1000 runs and print paired comparisons."""
import csv
from collections import defaultdict
import numpy as np
SCR = str(__import__('pathlib').Path(__file__).resolve().parent)   # this folder
REPO = "/Users/joonkim/Desktop/MastersThesis/repo"
CS = ["C_ID", "C_Random_Diag", "C_Random_Min_Diag", "C_Random_Full"]
PENS = ["lasso", "MCP", "SCAD"]

pilot = list(csv.DictReader(open(f"{SCR}/s1_nsweep_p10.csv")))
val = [r for r in pilot if float(r["n"]) == 1000.0]
sweep = [r for r in pilot if float(r["n"]) != 1000.0]

# n = 1000 from the committed runs (p = 10, reps < 25)
ref = {}
for pen, path in [("lasso", "runs/s1_dettling_reproduction"), ("MCP", "runs/s1b_pilot_p10-20/MCP"),
                  ("SCAD", "runs/s1b_pilot_p10-20/SCAD")]:
    for r in csv.DictReader(open(f"{REPO}/{path}/s1_per_dataset.csv")):
        if int(r["p"]) == 10 and int(r["rep"]) < 25:
            ref[(r["c_choice"], int(r["k"]), int(r["rep"]), 1000.0, pen)] = r
if val:
    d = [abs(float(r["max_f1"]) - float(ref[(r["c"], int(r["k"]), int(r["rep"]), 1000.0, "lasso")]["max_f1"])) for r in val]
    print(f"validation: lasso refit at n=1000 vs committed run, {len(d)} datasets, max |diff max_f1| = {max(d):.2e}\n")

data = dict(ref)
for r in sweep:
    data[(r["c"], int(r["k"]), int(r["rep"]), float(r["n"]), r["pen"])] = r
ns = sorted({k[3] for k in data})

def table(cs, title):
    print(title)
    print(f"{'n':>7} | {'max_f1 lasso / MCP / SCAD':>26} | {'auc lasso / MCP / SCAD':>24} | {'MCP-lasso f1 (z)':>17} | {'SCAD-lasso f1 (z)':>17} | {'exact L/M/S':>14}")
    for n in ns:
        keys = sorted({(c, k, rep) for (c, k, rep, nn, pen) in data if nn == n and c in cs
                       and all((c, k, rep, n, p) in data for p in PENS)})
        if not keys:
            continue
        g = {p: {m: np.array([float(data[(*key, n, p)][m]) for key in keys]) for m in ("max_f1", "auc")} for p in PENS}
        ex = {p: (np.mean([int(data[(*key, n, p)].get("exact", -1)) for key in keys]) if "exact" in data[(*keys[0], n, p)] else np.nan) for p in PENS}
        def dz(p):
            d = g[p]["max_f1"] - g["lasso"]["max_f1"]; se = d.std(ddof=1) / np.sqrt(len(d))
            return f"{d.mean():+.3f} ({d.mean() / se if se > 0 else 0:+.1f})"
        lab = "inf" if np.isinf(n) else f"1e{int(np.log10(n))}"
        exs = "  -" if np.isnan(ex["lasso"]) else f"{ex['lasso']:.2f}/{ex['MCP']:.2f}/{ex['SCAD']:.2f}"
        print(f"{lab:>7} | {g['lasso']['max_f1'].mean():.3f} / {g['MCP']['max_f1'].mean():.3f} / {g['SCAD']['max_f1'].mean():.3f}   | "
              f"{g['lasso']['auc'].mean():.3f} / {g['MCP']['auc'].mean():.3f} / {g['SCAD']['auc'].mean():.3f} | "
              f"{dz('MCP'):>17} | {dz('SCAD'):>17} | {exs:>14}   ({len(keys)} datasets)")
    print()

table(CS, "=== all four C choices pooled ===")
for c in CS:
    table([c], f"=== {c} ===")

print("orientation at the best-F1 point, means per dataset, all C pooled:")
for n in [n for n in ns if n != 1000.0]:
    for p in PENS:
        rs = [r for r in sweep if float(r["n"]) == n and r["pen"] == p]
        if rs:
            m = {k: np.mean([float(r[k]) for r in rs]) for k in ("skeleton_f1", "correct", "reversed", "hedged", "both", "half")}
            lab = "inf" if np.isinf(n) else f"1e{int(np.log10(n))}"
            print(f"  n={lab:>4} {p:<5}: skeleton F1 {m['skeleton_f1']:.3f}  correct {m['correct']:.2f}  reversed {m['reversed']:.2f}  "
                  f"hedged {m['hedged']:.2f}  2-cycle both {m['both']:.2f}  half {m['half']:.2f}")
