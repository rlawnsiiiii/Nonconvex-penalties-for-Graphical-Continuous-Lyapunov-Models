"""Inside the lasso's hedges: is the larger |M_hat| the true direction? And F1 if the lasso
pruned the smaller entry of every hedged pair (commit by magnitude)."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/joonkim/Desktop/MastersThesis/repo")
sys.path.insert(0, str(REPO / "simulations" / "diagnostics")); sys.path.insert(0, str(REPO / "src"))
from orientation import load_best_f1          # noqa: E402
from gclm.metrics import confusion            # noqa: E402

lasso = load_best_f1(REPO / "runs/s1_dettling_reproduction", 25, [10, 20])
for p in (10, 20):
    right = tot = 0; f1_l, f1_prune = [], []
    ratios = []
    for key, (mt, ml) in lasso.items():
        if key[0] != p: continue
        mp = ml.copy()
        for i in range(p):
            for j in range(i + 1, p):
                ti, tj = mt[i, j] != 0, mt[j, i] != 0
                if ml[i, j] != 0 and ml[j, i] != 0:            # hedged pair
                    big = (i, j) if abs(ml[i, j]) >= abs(ml[j, i]) else (j, i)
                    small = (j, i) if big == (i, j) else (i, j)
                    if ti != tj:                                # single true direction
                        tot += 1; right += mt[big] != 0
                        ratios.append(abs(ml[small]) / abs(ml[big]))
                    if not (ti and tj):                         # prune only non-2-cycles' smaller entry
                        mp[small] = 0.0
        f1_l.append(confusion(ml, mt).f1); f1_prune.append(confusion(mp, mt).f1)
    print(f"p={p}: lasso hedges on single true edges: larger entry is the true direction in "
          f"{right / tot:.3f} of {tot} pairs; median |small|/|big| = {np.median(ratios):.2f}")
    print(f"       F1 lasso {np.mean(f1_l):.3f} -> prune smaller entry of each hedge (2-cycles kept) "
          f"{np.mean(f1_prune):.3f} ({np.mean(f1_prune) - np.mean(f1_l):+.3f})")
