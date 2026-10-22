# E4 figures: the graph class

Copied from `runs/campaign/figures/` on 21 Oct 2026. Experiment E4 of `../../plan_mechanism_131026.md`; cells of wave 9.

## The two generators

**Thesis generator** (Dettling's §5; waves 1–8): every ordered pair is an edge, independently.

$$P(M_{ij}\neq 0)=d=\tfrac{k}{p}\ \ (i\neq j),\qquad M_{ij}\sim N(0,1)\ \text{on the support},\qquad M_{ii}=-\textstyle\sum_{j\neq i}\lvert M_{ij}\rvert-\lvert\varepsilon_{ii}\rvert$$

**DAG generator** (E4): a random node order $\pi$; the edge $j\to i$ is possible only if $\pi(j)<\pi(i)$, with probability $\min(2d,1)$. Same expected number of edges, $p(p-1)d$; never a cycle.

## Class of a drawn graph

`gclm.graphs.graph_class`, from the true support: **with 2-cycles** if a pair has both directions; else **DAG** if acyclic; else **cyclic, no 2-cycle**. Nobody chooses the class. Under the thesis generator

$$P(\text{at least one 2-cycle})\approx 1-e^{-\binom{p}{2}d^2}:\qquad 0.38,\ 0.85,\ 0.99,\ 1.00\quad (p=20,\ k=1,\dots,4)$$

## Figures

| file | x-axis | panels |
|---|---|---|
| `gain_by_generator_p20.png`, `gain_by_generator_p10.png` | generator | rows $(C,n)$, columns $k$ |
| `gain_by_generator_p20_bic_f1.png` | generator | rows $(C,n)$, columns $k$ |
| `gain_by_class_p20.png` | class, thesis generator | rows $(C,n)$, columns $k$ |
| `two_cycle_recovery_p20.png` | pairs, edges per graph | estimators |

$$\text{gain}=\text{mean over the same graphs of }F_1(\text{estimator})-F_1(\text{lasso of the same generator})$$

- $F_1$ at the oracle $\lambda$ (best on the path); `_bic_f1`: the BIC-selected graph.
- In parentheses: number of graphs, 100 per $(p,k)$ (25 replicates, 4 true $C$). Hollow: fewer than 20.
- $C$ in the fit: $2I$ or $2\,\mathrm{diag}(1/s_i^2)$; $n=10^3,10^4$.
- `two_cycle_recovery_p20.png`: BIC graph, $n=10^4$, rescaled $C$, thesis generator; true 2-cycles by directions kept, true single edges by fate.

```bash
python simulations/diagnostics/graph_class.py && python3 simulations/diagnostics/plot_graph_class.py
```
