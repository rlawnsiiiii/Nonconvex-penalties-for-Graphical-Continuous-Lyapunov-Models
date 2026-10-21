# The penalised diagonal (wave 10)

Dettling's eq. (1.4) penalises every entry, $\lambda\lVert M\rVert_1$. His simulations (glmnet through Varando's `lassoB`) and waves 1–9 leave the diagonal unpenalised. `--penalize-diagonal` switches to (1.4) for the direct loss.

The estimate is diagonal at $\lambda$ exactly when the diagonal-only solution $D_\lambda$ meets every off-diagonal first-order condition (the same for lasso, MCP and SCAD: slope $\lambda$ at $0^+$):

$$\lambda_{\max}=\min\lbrace\lambda:\ \lvert[2R(D_\lambda)\Sigma]_{ij}\rvert\le\lambda w_{ij}\ \ \forall i\ne j\rbrace,\qquad R(M)=M\Sigma+\Sigma M^\top+C$$

| | diagonal unpenalised | diagonal penalised |
|---|---|---|
| $D_\lambda$ | diagonal fit | `diagonal_solution`, shrunk by the penalty |
| $\lambda_{\max}$ | closed form, every penalty | bisection on the condition, per penalty |
| top of grid | diagonal fit | $D_{\lambda_{\max}}$ |
| adaptive weights | off-diagonal | diagonal too |

Code: `lambda_max`, `diagonal_solution` (`objective/direct.py`); `lasso_path`, `adaptive_lasso_path` (`solvers/path.py`); `--penalize-diagonal` (`run_s1_shard.py`).

Checks: unpenalised paths bit for bit as at HEAD (30 paths); tests in `test_lasso.py`, `test_nonconvex.py`, `test_lla_adaptive.py`, `test_campaign_*.py`; rehearsal on the laptop.

## Wave 10

The briefing's direct-loss cells: lasso, MCP, SCAD, both orders, adaptive lasso; both $C$; true $C=2I$; BIC and eBIC. Cells `direct_<estimator>-pd_<C>[_p<p>]`: $p=10,20$ at every $n$ with the search, $p=15$–$50$ at $n=10^3$. 276 tasks, about 370 CPU-h.

```bash
# laptop: commit, push.  LRZ:
cd ~/repo && git pull
bash cluster/submit_campaign.sh --wave 10 --dry-run            # "would submit 276 tasks in 96 cells"
nohup bash cluster/feed_queue.sh cluster/plan_211026.txt > logs/feed.out 2>&1 &
# laptop, when done:
rsync -av ge47xod3@cool.hpc.lrz.de:repo/runs/campaign/ runs/campaign/
python simulations/diagnostics/campaign.py   # campaign_paired.csv: reference "unpenalised" = X-pd against X
```
