"""Is MCP a coin flip on the lasso's hedges, and would a coin flip break even?
Pair-matched on identical datasets, best-F1 estimates from the stored shards."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/joonkim/Desktop/MastersThesis/repo")
sys.path.insert(0, str(REPO / "simulations" / "diagnostics"))
sys.path.insert(0, str(REPO / "src"))
from orientation import load_best_f1          # noqa: E402
from gclm.metrics import _pair_patterns, confusion  # noqa: E402

lasso = load_best_f1(REPO / "runs/s1_dettling_reproduction", 25, [10, 20])
runs = {pen: load_best_f1(REPO / f"runs/s1b_pilot_p10-20/{pen}", 25, [10, 20]) for pen in ("MCP", "SCAD")}

def label(e, t):
    # t in {1,2}: single true direction; e: 0 none, 1/2 one entry, 3 both
    if e == 0: return "missed"
    if e == 3: return "hedged"
    return "correct" if e == t else "reversed"

for p in (10, 20):
    keys = [k for k in lasso if k[0] == p and k in runs["MCP"]]
    print(f"\n===== p = {p}: {len(keys)} paired datasets =====")
    # ---- (1) what each penalty does on the pairs the lasso hedged
    for pen in ("MCP", "SCAD"):
        xt = Counter(); xt2 = Counter()
        for key in keys:
            mt, ml = lasso[key]; mp = runs[pen][key][1]
            t, el, ep = _pair_patterns(mt), _pair_patterns(ml), _pair_patterns(mp)
            for a, b, c in zip(t, el, ep):
                if a in (1, 2):
                    xt[(label(b, a), label(c, a))] += 1
                elif a == 3:
                    f = lambda e: {0: "none", 3: "both"}.get(e, "half")
                    xt2[(f(b), f(c))] += 1
        n = len(keys)
        print(f"\n{pen}: lasso outcome (rows) x {pen} outcome (cols), true single edges, per dataset")
        cols = ["correct", "reversed", "hedged", "missed"]
        print(f"{'':>10}" + "".join(f"{c:>10}" for c in cols) + f"{'total':>10}")
        for r in cols:
            row = [xt[(r, c)] / n for c in cols]
            print(f"{r:>10}" + "".join(f"{v:>10.2f}" for v in row) + f"{sum(row):>10.2f}")
        h_c, h_r = xt[("hedged", "correct")], xt[("hedged", "reversed")]
        print(f"  on lasso-hedged pairs where {pen} commits: correct {h_c / (h_c + h_r):.3f} "
              f"(n = {h_c + h_r} pairs)")
        print(f"  true 2-cycles, lasso (rows) x {pen} (cols): " + ", ".join(
            f"{a}->{b}: {v / n:.2f}" for (a, b), v in sorted(xt2.items())))

    # ---- (2) counterfactual on the lasso's own best-F1 support: resolve every hedge by a coin
    f1_l, f1_coin, f1_mcpq, f1_oracle, f1_cyc, qstar = [], [], [], [], [], []
    for key in keys:
        mt, ml = lasso[key]
        c = confusion(ml, mt); tp, fp, fn = c.tp, c.fp, c.fn
        nstar, nsel = tp + fn, tp + fp
        t, el = _pair_patterns(mt), _pair_patterns(ml)
        h = int(np.sum(((t == 1) | (t == 2)) & (el == 3)))   # hedged single edges
        b2 = int(np.sum((t == 3) & (el == 3)))                # 2-cycles with both found
        f1 = 2 * tp / (nstar + nsel)
        f1_l.append(f1)
        qstar.append(1 - f1 / 2)
        # each resolved hedge: |S_hat| - 1 for sure, TP - 1 with prob (1 - q)  -> exact expectation
        ef1 = lambda q: 2 * (tp - h * (1 - q)) / (nstar + nsel - h)
        f1_coin.append(ef1(0.5)); f1_oracle.append(ef1(1.0))
        # and keep only one direction of every fully found 2-cycle (no coin there: always -1 TP)
        f1_cyc.append(2 * (tp - h * 0.5 - b2) / (nstar + nsel - h - b2))
    m = lambda x: float(np.mean(x))
    print(f"\ncounterfactual on the LASSO's best-F1 support, p = {p}:")
    print(f"  lasso as is                                  F1 {m(f1_l):.3f}")
    print(f"  every hedge -> one direction, fair coin       F1 {m(f1_coin):.3f}  ({m(f1_coin) - m(f1_l):+.3f})")
    print(f"  ... and every found 2-cycle -> one direction  F1 {m(f1_cyc):.3f}  ({m(f1_cyc) - m(f1_l):+.3f})")
    print(f"  every hedge -> the correct direction (oracle)  F1 {m(f1_oracle):.3f}  ({m(f1_oracle) - m(f1_l):+.3f})")
    print(f"  break-even accuracy q* = 1 - F1/2: mean {m(qstar):.3f}, range {min(qstar):.2f}-{max(qstar):.2f}")
