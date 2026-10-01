# Recorded run outputs

Raw output of the runs summarised in `../S1_reproduction.md` §8. Committed so the numbers in
the write-up can be traced back to a file rather than re-derived.

| file | produced by | notes |
|---|---|---|
| `m0_reps5.csv` | `run_m0.py --reps 5` | Figure 3 reproduction; §8.1 |
| `s1_p10-20_reps10.csv` | `run_s1.py --p 10 15 20 --reps 10` | 480 datasets, 4.2 min on 8 cores; §8.4 |
| `s1_allp_CID_reps5_raw.csv` / `..._standardized.csv` | `run_s1.py --c C_ID --reps 5`, all p | resolves `standardize`; §8.6 |
| `s1_allC_allp_reps10_standardized.csv` | `run_s1.py --reps 10`, all p, all C | 1120 datasets, 40.6 min; §8.7 |
| `diagonal_flag.txt` | ad-hoc script, §8.5 | resolves `metrics_include_diagonal` |
| `tolerance_benchmark.txt` | ad-hoc script, §8.3 | fixes `S1Config.tol = 1e-8` |

The two ad-hoc scripts were one-off probes and are not in the repo; both are reproducible from
the parameters stated in the corresponding section. The full M1 grid has **not** been run yet.

Full production runs live in `runs/<run name>/` at the repository root (e.g.
`runs/s1_dettling_reproduction/`). Their raw shards are gitignored; summaries and figures are not.
