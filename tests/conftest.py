"""Shared fixtures and the R bridge used by the validation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gclm.dgp import CChoice, draw_instance  # noqa: E402


from gclm.rbridge import r_available, run_r  # noqa: E402,F401

R_AVAILABLE = r_available("glmnet", "jsonlite")
NCVREG_AVAILABLE = r_available("ncvreg", "jsonlite")

requires_r = pytest.mark.skipif(
    not R_AVAILABLE, reason="Rscript with glmnet + jsonlite not available"
)
requires_ncvreg = pytest.mark.skipif(
    not NCVREG_AVAILABLE, reason="Rscript with ncvreg + jsonlite not available"
)

try:
    import skglm  # noqa: F401
    SKGLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKGLM_AVAILABLE = False

requires_skglm = pytest.mark.skipif(
    not SKGLM_AVAILABLE, reason="skglm not installed (pip install skglm)"
)


@pytest.fixture
def instance():
    """A small, well-conditioned problem: p=5, k=2, n=200, C = 2 I."""
    rng = np.random.default_rng(11)
    m_true, _, _, sigma_hat = draw_instance(
        p=5, k=2, n=200, c_choice=CChoice.ID, rng=rng
    )
    return m_true, sigma_hat, 2.0 * np.eye(5)
