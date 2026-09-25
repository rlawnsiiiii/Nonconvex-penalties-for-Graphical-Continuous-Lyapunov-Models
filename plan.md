Thesis - Joon Kim
=====
## References:
### GCLM
- [On the Lasso for Graphical Continuous Lyapunov Models - Dettling](https://proceedings.mlr.press/v236/dettling24a/dettling24a.pdf)
- [Graphical Continuous Lyapunov Models - Varando](https://proceedings.mlr.press/v124/varando20a/varando20a.pdf)
- [Diss-Detttling](https://mediatum.ub.tum.de/doc/1745288/ayjlug6xhjco12g8qdle9flpo.DoktorarbeitPhilippDettlingLyapunovModels_adaptiertes_Titelblatt.pdf)
### Non-convex Penalties
- [Non-convex Penalties - Du](https://arxiv.org/abs/2502.07655)
- [Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties](https://www.tandfonline.com/doi/abs/10.1198/016214501753382273)
- [Regularized M-estimators with nonconvexity](https://arxiv.org/pdf/1305.2436)
- [Support recovery without incoherence: A case for nonconvex regularization](https://arxiv.org/abs/1412.5632)
- [Asymptotics for estimating a diverging number of parameters -- with and without sparsity](https://arxiv.org/abs/2411.17395)


## Meeting 1 (2026 Sept 22)
### Questions
- Why do we want to apply non-convex penalties like SCAD and MCP (instead of L1-Penalty) to the problem in Dettling-paper? Non-convex penalties should help reduce bias (smaller MSE) - especially for strong edges, while remaining smiliar to Lasso for weaker edges, i.e. similarly shrinking them towards zero. I.e. for high $\lambda$ - penalty coefficients, moderate/strong edges would still be recovered. But can we expect big improvements regarding support recovery (and not MSE) in general? Is the motivation for nonconvex penalties the failing irrepresentability condition?
    - failing ir. cond -> correlation among edges (and non-edges) -> shrinkage bias affecting other edges -> estimation error (error in recovered support)
    - for nonconvex penalties: no such shrinkage for strong edges, gradients for zero coefficients low enough. But we would need RSC(restricted strong convexity) instead.
    - Maybe check for which drift matrices RSC holds and ir.cond. fails?
    - -> Motivation correct, first focus on simulations.
- Which experiments should we do? Same ones as Section 5 and 6 of Dettling-paper for better comparisons?
    - -> see below
- In case the optimization takes long (for non-convex penalties), do we have some cluster-compute?
    - --> look for student clusters, else contact Stephan Haug <haug@tum.de>
- Registration
    - -> contact Prof. MD around Oct 10-11

- Score-based search with e.g. BIC
    - initialize random graphs, define neighborhood of graphs (define perturbations - how many edge changes e.g.?), compute BIC for newer graphs, greedily pick the best graph according to BIC, paper- not many local optima, thomas nagler - asymptotics for diverging number of parameters sparsity


### Simulations

- for compute :  student clusters?  maybe at LRZ?  If needed contact our group member Stephan Haug <haug@tum.de>
- same setup as Varando/Dettling
- quadratic loss + ell1 versus + nonconvex MCP/SCAD type penalty
    - (here you can use any package that does linear regression with such penalties, you have to create a suitable response vector y and a design matrix X out of your input)
- likelihood + ell1 versus + nonconvex (likelihood was considered in Varando Hansen)
    -  here you would need to code up something, if you do a gradient based method, then Varando Hansen explain how to compute the gradient of the Gaussian log-likelihood a bit more efficient than "naive"
- when this is done, you can consider a score-based search method as an alternative (with quadratic loss + BIC-type penalty or likelihood + BIC-type penalty)
    - [again you would need to code up something]
