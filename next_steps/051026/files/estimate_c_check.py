#!/usr/bin/env python3
"""Does estimating a diagonal C (Varando & Hansen, eq. (7), kappa < infinity) do the job of
rescaling C by hand?  A small check with their own package (gclm::gclm, log-likelihood loss,
lasso) on the repository's Figure-5 datasets, standardised as in every thesis run.

Treatments of C, all on the correlation matrix (Varando's scale: C = I rather than 2I, which
only rescales M and lambda):

  fixed_I      C = I held fixed            (their mloglik-inf; Dettling's convention)
  fixed_resc   C = diag(1/s^2) held fixed  (the "rescaled C" of next_steps/031026; mean 1)
  est_<kappa>  C diagonal, estimated with the ridge kappa * ||C - I||^2 (their mloglik-0.01)

Path directions: "down" = sparse -> dense from the diagonal fit (every thesis run so far),
"up" = dense -> sparse from B0 = -C R^{-1} / 2 (the order used in their simulations).  All
treatments are run "up"; fixed_I is also run "down" as the link to the thesis runs.

gclm stops on the decrease of the objective, which is a weak rule in the flat directions of this
loss: single paths still change between eps = 1e-8 and 1e-12, and some lambdas need more than
1e5 iterations.  The defaults (eps = 1e-9, at most 1e4 iterations, 50 lambdas) keep the check at
about an hour; the column max_iter_hit counts the lambdas that stopped at the cap.  Read the
means over graphs, not single graphs.

Besides the usual path metrics the script records how well the estimated diagonal follows the
true one, S^{-1} C* S^{-1} (correlation of the logarithms, scale-free).

    python next_steps/051026/files/estimate_c_check.py --workers 6
    python next_steps/051026/files/estimate_c_check.py --summarize
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gclm.config import S1Config, parse_n_obs  # noqa: E402
from gclm.data.simulate import CChoice, draw_instance  # noqa: E402
from gclm.metrics import evaluate_path  # noqa: E402

C_NAMES = tuple(c.value for c in CChoice)
VARIANTS = (("fixed_I", "down"), ("fixed_I", "up"), ("fixed_resc", "up"), ("est_0.01", "up"),
            ("est_1", "up"))


def log_corr(a, b):
    a, b = np.log(np.asarray(a, float)), np.log(np.asarray(b, float))
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(task):
    p, k, c_name, rep, n, eps, max_iter, n_lambda = task
    cfg = S1Config()
    rng = np.random.default_rng([cfg.seed, p, k, C_NAMES.index(c_name), rep])
    m_true, c_true, _, s_raw = draw_instance(p, k, n, CChoice(c_name), rng, metzler=cfg.metzler,
                                             standardize=False)
    s = np.sqrt(np.diag(s_raw))
    r = s_raw / np.outer(s, s)                            # what standardize=True returns
    off = ~np.eye(p, dtype=bool)
    c_std = np.diag(c_true) / s ** 2                      # diagonal of S^{-1} C* S^{-1}
    resc = 1.0 / s ** 2
    resc = resc / resc.mean()
    one = np.ones(p)
    lam_max = 2.0 * np.abs(r[off]).max()                  # lambda_max of the C = I problem
    lambdas = np.geomspace(cfg.lambda_ratio * lam_max, lam_max, n_lambda)
    variants = []
    for name, direction in VARIANTS:
        if name == "fixed_I":
            v = dict(C=one, C0=one, lambdac=-1.0)
        elif name == "fixed_resc":
            v = dict(C=resc, C0=resc, lambdac=-1.0)
        else:
            v = dict(C=one, C0=one, lambdac=float(name.split("_")[1]))
        variants.append(dict(name=f"{name}_{direction}", direction=direction, lambdac=v["lambdac"],
                             C=v["C"].tolist(), C0=v["C0"].tolist()))
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        inp.write_text(json.dumps(dict(R=r.tolist(), lambdas=lambdas.tolist(), variants=variants,
                                       eps=eps, maxIter=max_iter)))
        proc = subprocess.run(["Rscript", str(HERE / "estimate_c_check.R"), str(inp), str(out)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        res = json.loads(out.read_text())
    seconds = time.perf_counter() - t0
    rows = []
    for v in variants:
        o = res[v["name"]]
        est = [np.array(b, dtype=float) for b in o["B"]]
        ev = evaluate_path(est, m_true)
        best = int(np.nanargmax(ev["f1"]))
        cs = [np.array(c, dtype=float) for c in o["C"]]
        name, direction = v["name"].rsplit("_", 1)
        rows.append({"p": p, "k": k, "c_choice": c_name, "rep": rep, "n": n, "c_treatment": name,
                     "direction": direction, "max_f1": ev["max_f1"], "auc": ev["auc"],
                     "aupr": ev["aupr"], "max_acc": ev["max_acc"],
                     "edges_at_lambda_max": int((est[-1] != 0)[off].sum()),
                     "c_logcorr_best": log_corr(cs[best], c_std),
                     "c_logcorr_mid": log_corr(cs[len(cs) // 2], c_std),
                     "c_min_best": float(cs[best].min()), "c_max_best": float(cs[best].max()),
                     "resc_logcorr": log_corr(resc, c_std),
                     "max_iter_hit": int(sum(i >= max_iter for i in o["iter"])),
                     "seconds": seconds})
    return rows


def summarize(path: Path):
    rows = list(csv.DictReader(path.open()))
    by = defaultdict(dict)
    for r in rows:
        by[(r["k"], r["c_choice"], r["rep"], r["n"])][(r["c_treatment"], r["direction"])] = r
    ref = ("fixed_I", "down")
    groups = [("C_ID", ("C_ID",)), ("C_Random_Diag", ("C_Random_Diag",)),
              ("C_Random_Min_Diag", ("C_Random_Min_Diag",)), ("C_Random_Full", ("C_Random_Full",))]
    ns = sorted({k[3] for k in by}, key=float)
    for label, cs in groups:
        for n in ns:
            keys = [k for k in by if k[1] in cs and k[3] == n]
            if not keys:
                continue
            print(f"\n{label}, n = {n}  [{len(keys)} graphs]   mean (z of the paired difference to "
                  f"fixed_I / down)")
            for name, direction in VARIANTS:
                v = (name, direction)
                line = f"  {name + ' ' + direction:<16}"
                for col in ("max_f1", "aupr", "auc"):
                    x = np.array([float(by[k][v][col]) for k in keys])
                    d = x - np.array([float(by[k][ref][col]) for k in keys])
                    se = d.std(ddof=1) / math.sqrt(len(d))
                    z = f"({d.mean() / se:+.1f})" if v != ref and se > 0 else "      "
                    line += f"  {col} {x.mean():.3f} {z:<7}"
                if name.startswith("est"):
                    lc = np.array([float(by[k][v]["c_logcorr_best"]) for k in keys])
                    line += f"  corr(log C_hat, log C_true) {np.nanmean(lc):+.2f}"
                if name == "fixed_resc":
                    lc = np.array([float(by[k][v]["resc_logcorr"]) for k in keys])
                    line += f"  corr(log 1/s^2, log C_true) {np.nanmean(lc):+.2f}"
                print(line)


def markdown(path: Path):
    """The tables and numbers of next_steps/051026/cluster_campaign_051026.md, Section 2.4.
    The first row is the cluster baseline (repository's solver, C = 2I, sparse -> dense) on the
    same graphs, read from runs/nsweep_p10-20.  n = inf is left out: gclm's stopping rule ends
    almost at once there (see the write-up)."""
    sys.path.insert(0, str(ROOT / "simulations" / "diagnostics"))
    sys.path.insert(0, str(ROOT / "simulations"))
    from nsweep import DEFAULT, load_cells

    cells = load_cells(DEFAULT)
    rows = list(csv.DictReader(path.open()))
    by = defaultdict(dict)
    for r in rows:
        by[(int(r["k"]), r["c_choice"], int(r["rep"]), r["n"])][(r["c_treatment"], r["direction"])] = r
    cell_n = {"1000": "1000", "10000": "1e4"}
    names = [(("baseline",), "cluster baseline: our solver, $C = 2I$, sparse → dense"),
             (("fixed_I", "down"), "their package, $C = I$, sparse → dense"),
             (("fixed_I", "up"), "their package, $C = I$ (`mloglik-inf`)"),
             (("fixed_resc", "up"), "their package, rescaled $C$"),
             (("est_0.01", "up"), "their package, $C$ estimated, $\\kappa = 0.01$ (`mloglik-0.01`)"),
             (("est_1", "up"), "their package, $C$ estimated, $\\kappa = 1$")]
    ref = ("fixed_I", "up")
    groups = [("`C_ID`", ("C_ID",)),
              ("random diagonal $C$ (`C_Random_Diag`, `C_Random_Min_Diag`)",
               ("C_Random_Diag", "C_Random_Min_Diag")),
              ("`C_Random_Full`", ("C_Random_Full",))]

    def value(v, k, col):
        if v == ("baseline",):
            return cells[("loglik", "lasso", cell_n[k[3]])][(10, k[0], k[1], k[2])][col]
        return float(by[k][v][col])

    for label, cs in groups:
        n_graphs = len({k[:3] for k in by if k[1] in cs})
        print(f"\n**{label}, {n_graphs} graphs**\n")
        print("| | `max_f1`, $n = 10^3$ | `max_f1`, $10^4$ | `aupr`, $10^3$ | `aupr`, $10^4$ |")
        print("|---|---|---|---|---|")
        for v, name in names:
            out = []
            for col in ("max_f1", "aupr"):
                for n in ("1000", "10000"):
                    keys = [k for k in by if k[1] in cs and k[3] == n]
                    x = np.array([value(v, k, col) for k in keys])
                    d = x - np.array([value(ref, k, col) for k in keys])
                    se = d.std(ddof=1) / math.sqrt(len(d))
                    z = f" ({d.mean() / se:+.1f})".replace("-", "−") if v != ref and se > 0 else ""
                    out.append(f"{x.mean():.3f}{z}")
            print(f"| {name} | " + " | ".join(out) + " |")
    finite = [r for r in rows if r["n"] != "inf"]
    print("\nthe estimated diagonal at the lambda of the best F1 (n = 1e3 and 1e4):")
    for tr in ("est_0.01", "est_1"):
        r = [x for x in finite if x["c_treatment"] == tr]
        lo = np.array([float(x["c_min_best"]) for x in r])
        hi = np.array([float(x["c_max_best"]) for x in r])
        print(f"  {tr}: smallest entry median {np.median(lo):.3f}, below 0.01 in "
              f"{np.mean(lo < 0.01):.0%} of the graphs; largest entry at most {hi.max():.2f}")
    print("correlation of log C with log C_true:")
    for label, cs in groups:
        line = f"  {label}:"
        for tr, col in (("fixed_resc", "resc_logcorr"), ("est_0.01", "c_logcorr_best"),
                        ("est_1", "c_logcorr_best")):
            v = [float(x[col]) for x in finite if x["c_treatment"] == tr and x["direction"] == "up"
                 and x["c_choice"] in cs]
            line += f"  {tr} {np.nanmean(v):+.2f}"
        print(line)
    hit = defaultdict(list)
    for r in finite:
        hit[f"{r['c_treatment']} {r['direction']}"].append(int(r["max_iter_hit"]))
    print("lambdas stopped at the iteration cap, mean number per path:",
          {k: round(float(np.mean(v)), 1) for k, v in hit.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n", type=parse_n_obs, nargs="+", default=[1000, 10_000, math.inf])
    ap.add_argument("--reps-id", type=int, default=10, help="replicates per k for C_ID")
    ap.add_argument("--reps-other", type=int, default=5, help="replicates per k, other C choices")
    ap.add_argument("--eps", type=float, default=1e-9, help="gclm's stopping threshold")
    ap.add_argument("--max-iter", type=int, default=10_000, help="gclm's iteration cap per lambda")
    ap.add_argument("--n-lambda", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=HERE / "estimate_c_check.csv")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="print the tables of the write-up")
    args = ap.parse_args()
    if args.markdown:
        return markdown(args.out)
    if args.summarize:
        return summarize(args.out)
    tasks = [(args.p, k, c, rep, n, args.eps, args.max_iter, args.n_lambda)
             for n in args.n for k in (1, 2, 3, 4) for c in C_NAMES
             for rep in range(args.reps_id if c == "C_ID" else args.reps_other)]
    t0, writer = time.time(), None
    with ProcessPoolExecutor(args.workers) as ex, args.out.open("w", newline="") as fh:
        for i, fut in enumerate(as_completed([ex.submit(run, task) for task in tasks]), 1):
            rows = fut.result()
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} datasets, {(time.time() - t0) / 60:.1f} min", flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
