"""Configuration for the S1 reproduction (see simulations/S1_reproduction.md Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from gclm.dgp import CChoice


@dataclass(frozen=True)
class S1Config:
    """Dettling et al. (2024), Section 5 / Figure 5."""

    p_values: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50)
    k_values: tuple[int, ...] = (1, 2, 3, 4)
    c_choices: tuple[CChoice, ...] = tuple(CChoice)
    n_rep: int = 100
    n_obs: int = 1000
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

    # Solver backend: "fista" (default; the only one that scales to the full
    # p = 50 grid), "design", "glmnet" (Dettling's own choice), or "ncvreg"
    # (most accurate, and the only backend offering MCP/SCAD).  See
    # simulations/S1_reproduction.md Section 7.2.
    solver: str = "fista"
    penalty: str = "lasso"          # "MCP"/"SCAD" require solver="ncvreg"
    gamma: float | None = None      # concavity parameter for MCP/SCAD

    @property
    def n_datasets(self) -> int:
        return len(self.p_values) * len(self.k_values) * len(self.c_choices) * self.n_rep


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
