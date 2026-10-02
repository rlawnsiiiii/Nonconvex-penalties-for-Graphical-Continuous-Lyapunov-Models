import csv
from collections import defaultdict
import numpy as np
rows = list(csv.DictReader(open(str(__import__('pathlib').Path(__file__).resolve().parent) + "/m0_objective.csv")))
for r in rows:
    for k in r:
        r[k] = float(r[k])
by = defaultdict(list)
for r in rows:
    by[(r["n"], r["rep"])].append(r)
print(f"{'n':>7} {'#data':>6} | {'max_f1 cont / search / best-of-2 / oracle':>42} | {'exact rec. cont/search/best-of-2':>33} | {'truth basin lower (where it is exact)':>38}")
for n in sorted({k[0] for k in by}):
    ds = [v for k, v in by.items() if k[0] == n]
    mf = {e: [] for e in ("cont", "search", "best", "oracle")}
    ex = {e: [] for e in ("cont", "search", "best")}
    lower, cnt = 0, 0
    for path in ds:
        f1 = {e: [] for e in mf}
        for r in path:
            f1["cont"].append(r["f1_cont"]); f1["search"].append(r["f1_search"]); f1["oracle"].append(r["f1_oracle"])
            f1["best"].append(r["f1_oracle"] if r["F_oracle"] < r["F_cont"] - 1e-12 else r["f1_cont"])
            if r["f1_oracle"] > 1 - 1e-12:
                cnt += 1; lower += r["F_oracle"] < r["F_cont"] - 1e-12
        for e in mf:
            mf[e].append(max(f1[e]))
        for e in ex:
            ex[e].append(max(f1[e]) > 1 - 1e-12)
    lab = "inf" if np.isinf(n) else f"{int(n)}"
    print(f"{lab:>7} {len(ds):>6} | {np.mean(mf['cont']):.3f} / {np.mean(mf['search']):.3f} / {np.mean(mf['best']):.3f} / {np.mean(mf['oracle']):.3f}"
          f"{'':>8} | {np.mean(ex['cont']):.2f} / {np.mean(ex['search']):.2f} / {np.mean(ex['best']):.2f}{'':>14} | "
          f"{lower}/{cnt} lambdas ({lower / max(cnt, 1):.2f})")
# lambda range where the truth basin wins, per n (relative to lambda_max of each path)
print("\nwhere along the path the truth basin is exact AND lower (fraction of the 25 inspected lambdas, index 0 = densest):")
for n in sorted({k[0] for k in by}):
    ds = [v for k, v in by.items() if k[0] == n]
    hits = np.zeros(25); tot = 0
    for path in ds:
        tot += 1
        for t, r in enumerate(sorted(path, key=lambda r: r["idx"])):
            hits[t] += (r["f1_oracle"] > 1 - 1e-12) and (r["F_oracle"] < r["F_cont"] - 1e-12)
    lab = "inf" if np.isinf(n) else f"{int(n)}"
    print(f"{lab:>7}: " + " ".join(f"{h / tot:.1f}" for h in hits))
