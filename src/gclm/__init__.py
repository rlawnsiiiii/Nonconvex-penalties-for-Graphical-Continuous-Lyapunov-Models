"""Graphical Continuous Lyapunov Models -- simulation toolkit for the thesis.

Layout (see ARCHITECTURE.md):

    lyapunov      the model: vec, A(Sigma), Lyapunov solves
    data          simulated data and the Example 2 models
    objective     direct loss (Dettling), covariance losses (Varando), penalties
    solvers       proximal gradient, coordinate descent, package backends,
                  the covariance-loss solvers, and the paths
    metrics       Definitions G.4/G.5, ROC/PR curves
    config        S1Config / M0Config
"""

from gclm import config, data, lyapunov, metrics, objective, solvers
from gclm.lyapunov import design_matrix, solve_lyapunov, unvec, vec

__all__ = ["config", "data", "lyapunov", "metrics", "objective", "solvers",
           "design_matrix", "solve_lyapunov", "vec", "unvec"]
