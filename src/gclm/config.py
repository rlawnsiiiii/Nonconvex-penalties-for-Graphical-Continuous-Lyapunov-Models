"""Configuration for the S1 reproduction (see simulations/S1_reproduction.md Sec. 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gclm.data.simulate import CChoice


def parse_n_obs(text: str) -> int | float:
    """Sample size from the command line: ``"1000"`` or ``"1e5"`` -> an int,
    ``"inf"`` -> ``float("inf")``, which feeds the population covariance (no
    sampling; see :func:`gclm.data.simulate.draw_instance`)."""
    try:
        x = float(text)
    except ValueError:
        raise ValueError(f"sample size must be an integer or 'inf', got {text!r}") from None
    if math.isinf(x) and x > 0:
        return math.inf
    if not (math.isfinite(x) and x >= 2 and x == int(x)):
        raise ValueError(f"sample size must be an integer >= 2 or 'inf', got {text!r}")
    return int(x)


@dataclass(frozen=True)
class S1Config:
    """Dettling et al. (2024), Section 5 / Figure 5."""

    p_values: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50)
    k_values: tuple[int, ...] = (1, 2, 3, 4)
    c_choices: tuple[CChoice, ...] = tuple(CChoice)
    n_rep: int = 100
    n_obs: int | float = 1000       # math.inf -> the population covariance

    n_lambda: int = 100
    lambda_ratio: float = 1e-4

    # -- open settings, see S1_reproduction.md Section 4.1 --
    penalize_diagonal: bool = False
    metrics_include_diagonal: bool = False
    standardize: bool = True
    metzler: bool = False

    # Solver tolerance.  Benchmarked in S1_reproduction.md Section 8.3: 1e-8 is
    # 1.7x faster than 1e-10 with max_acc/max_f1 identical and auc/aupr within
    # 0.001.  Tighten for any estimation-error (as opposed to support) study.
    tol: float = 1e-8

    seed: int = 20260922

    # Solver backend.  Default "fista": the hand-written accelerated proximal
    # gradient.  It is the only backend that reaches p = 50 in practical time
    # (the others work on the explicit p^2 x p^2 design, O(p^4) per sweep), and
    # it is validated against an analytic solution, a duality-gap certificate,
    # cvxpy/CLARABEL, and every other backend -- see tests/test_fista.py and
    # docs/FISTA.md.  Alternatives: "ncvreg" (MCP/SCAD, most accurate),
    # "skglm" (pure-Python MCP), "glmnet" (Dettling's choice), "pyproximal",
    # "design".
    solver: str = "fista"
    # Which loss the penalty is attached to.  "direct": Dettling's quadratic
    # loss on the Lyapunov residual (the S1 estimator; src/gclm/objective/direct.py, every
    # `solver` above).  "loglik" / "frobenius": Varando & Hansen's losses on the
    # implied covariance Sigma(M) (src/gclm/objective/covariance.py, its own solvers;
    # `solver` is then ignored).  docs/LIKELIHOOD.md.
    loss: str = "direct"            # "direct" | "loglik" | "frobenius"
    # Path order.  "down" walks from lambda_max to the dense end.  "up" walks from
    # the dense end to lambda_max: for the covariance losses from the dense fit
    # (Varando's order), for the direct loss with MCP/SCAD from the lasso solution
    # at the smallest lambda.  Irrelevant for the direct lasso (convex).
    direction: str = "down"
    # Where a dense -> sparse path starts (direction "up").  None = the loss's default: the
    # lasso solution at the smallest lambda for the direct loss (every run before 14 October
    # 2026), the exact fit -C Sigma_hat^-1 / 2 (Varando & Hansen) for the covariance losses.
    # Explicit choices, both losses (plan of 13 October 2026, E1): "lasso", "exact", "l1" (the
    # minimum-l1 exact fit, gclm.objective.exact_fits), "l2" (the minimum-Frobenius exact fit),
    # "truth-<tau>" (the standardised truth plus tau N(0, 1) off the diagonal; a ceiling).
    # `start_name` is the resolved value, what the shard records.
    up_start: str | None = None
    # The pilot of the adaptive lasso and of the thresholded lasso (methods "adaptive",
    # "threshold", "threshold-norefit"): the same choices as up_start; "lasso" = the dense end of
    # the lasso path of the same loss (every run before 14 October 2026).
    pilot: str = "lasso"
    # The generator's knobs of 13 October 2026 (gclm.data.simulate.sample_drift): the law of the
    # weights ("normal" = N(0, 1), "uniform" = |w| ~ U[w_min, 1] with a random sign), the lower
    # bound of the uniform law, and the graph class ("random", "dag", "pairs").  Defaults = the
    # thesis generator, bit for bit.
    weights: str = "normal"
    w_min: float = 0.0
    graph: str = "random"
    # E5 (19 October 2026): with graph = "pairs", the share of connected pairs that are 2-cycles
    two_cycle_share: float = 0.0
    # The volatility matrix used for estimation: "identity" (2 I, Dettling's
    # pipeline, every run before the campaign of October 2026) or "variance"
    # (2 diag(1 / s_i^2) with s_i the standard deviations the data were divided by:
    # the model "C = 2 I on the measurement scale", the "rescaled C" --
    # gclm.data.simulate.estimation_volatility).
    c_scale: str = "identity"
    # How the path is computed.  "path": warm-started continuation in the order given
    # by `direction` (gclm.solvers.path.lasso_path / covloss_path).  "lla" (direct loss
    # only): MCP / SCAD by two local-linear-approximation steps from the lasso solution at
    # each lambda (lla_path).  "adaptive": adaptive lasso with weights from the dense end
    # of the lasso path of the same loss (adaptive_lasso_path / adaptive_covloss_path;
    # `penalty` must be "lasso").
    method: str = "path"
    penalty: str = "lasso"          # "lasso" | "MCP" | "SCAD"  (docs/NONCONVEX.md)
    gamma: float | None = None      # concavity; None -> 3 (MCP) / 3.7 (SCAD)
    # How MCP/SCAD are applied -- irrelevant for the lasso.  "textbook": P(M_ij)
    # on every entry (fista, skglm).  "ncvreg": P(v_ij M_ij)/v_ij with v_ij the
    # loss curvature, which is what ncvreg minimises (fista, ncvreg).
    convention: str = "textbook"

    @property
    def n_datasets(self) -> int:
        return len(self.p_values) * len(self.k_values) * len(self.c_choices) * self.n_rep

    @property
    def start_name(self) -> str:
        """``up_start`` with the loss's default filled in (the direct loss starts from the lasso,
        the covariance losses from the exact fit)."""
        if self.up_start is not None:
            return self.up_start
        return "lasso" if self.loss == "direct" else "exact"


@dataclass(frozen=True)
class M0Config:
    """Example 2 / Figure 3: path vs. 5-cycle (S1_reproduction.md Section 6)."""

    diagonal: tuple[float, ...] = (-2.0, -3.0, -4.0, -5.0, -6.0)
    subdiagonal: float = 0.65
    m15_fixed: float = 0.65
    m15_range: tuple[float, float] = (0.5, 1.0)
    sample_sizes: tuple[float, ...] = (100, 200, 500, 1000, 5000, 1e4, 1e5, float("inf"))
    n_rep: int = 100
    n_lambda: int = 100
    lambda_ratio: float = 1e-4
    seed: int = 20260922
