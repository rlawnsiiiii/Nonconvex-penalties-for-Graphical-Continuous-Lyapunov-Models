"""Pins the worked example in R/ENCODING.md.

That document is meant to be checkable by eye, which is only useful if its
numbers stay true.  These tests assert the exact design matrix, response,
penalty factors, lambda conversions and fitted estimate that it prints.
"""

from __future__ import annotations

import numpy as np
import pytest

from gclm.lasso import (
    lasso_path,
    penalty_weights,
    to_glmnet_lambda,
    to_ncvreg_lambda,
)
from gclm.lyap import design_matrix, unvec, vec

from conftest import requires_ncvreg, requires_r

SIGMA = np.array([[2.0, 0.5, 0.0],
                  [0.5, 3.0, 1.0],
                  [0.0, 1.0, 4.0]])
C = 2.0 * np.eye(3)
P = 3
LAMBDA = 0.7

#: A(Sigma) exactly as tabulated in ENCODING.md Section 3
EXPECTED_X = np.array([
    [4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 2.0, 0.0, 3.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 1.0, 0.0, 0.5, 4.0, 0.0, 0.0],
    [0.5, 2.0, 0.0, 3.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0],
    [0.0, 0.0, 2.0, 1.0, 0.0, 0.5, 4.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 8.0],
])

#: the estimate both packages return at lambda = 0.7
EXPECTED_M = np.array([[-0.514089, 0.095056, 0.000000],
                       [0.000000, -0.365853, 0.127540],
                       [0.000000, 0.000000, -0.246797]])


def test_design_matrix_matches_document():
    assert np.allclose(design_matrix(SIGMA), EXPECTED_X)


def test_response_and_penalty_factors_match_document():
    assert np.array_equal(-vec(C), np.array([-2, 0, 0, 0, -2, 0, 0, 0, -2], float))
    assert np.array_equal(vec(penalty_weights(P)),
                          np.array([0, 1, 1, 1, 0, 1, 1, 1, 0], float))


def test_row_arithmetic_stated_in_the_document():
    """The two rows the document invites you to verify by hand."""
    x = design_matrix(SIGMA)
    # R11 = 2 * sum_k M_1k Sigma_k1 + C_11 = 4*M11 + 1*M12
    assert np.array_equal(x[0], np.array([4, 0, 0, 1, 0, 0, 0, 0, 0], float))
    # R33 = 2 * sum_k M_3k Sigma_k3 = 2*M32 + 8*M33   (row index 8, column-major)
    assert np.array_equal(x[8], np.array([0, 0, 0, 0, 0, 2, 0, 0, 8], float))


def test_duplicate_rows_and_rank():
    """Off-diagonal equations appear twice; rank is p(p+1)/2, not p^2."""
    x = design_matrix(SIGMA)
    assert np.array_equal(x[1], x[3])          # R21 == R12
    assert np.array_equal(x[2], x[6])          # R31 == R13
    assert np.array_equal(x[5], x[7])          # R32 == R23
    assert np.linalg.matrix_rank(x) == P * (P + 1) // 2 == 6


def test_column_index_map():
    """Column j (0-based) is the coefficient of M[j % p, j // p]."""
    for j in range(P * P):
        e = np.zeros(P * P)
        e[j] = 1.0
        m = unvec(e, P)
        assert m[j % P, j // P] == 1.0
        assert m.sum() == 1.0


def test_lambda_conversions_match_document():
    assert np.isclose(to_glmnet_lambda(LAMBDA, P), 0.0518519, atol=1e-7)
    assert np.isclose(to_ncvreg_lambda(LAMBDA, P), 0.0777778, atol=1e-7)
    # the two differ by exactly p/(p-1) -- glmnet's penalty.factor rescaling
    assert np.isclose(to_ncvreg_lambda(LAMBDA, P) / to_glmnet_lambda(LAMBDA, P),
                      P / (P - 1))


@requires_r
@pytest.mark.r
def test_glmnet_reproduces_the_documented_estimate():
    path = lasso_path(SIGMA, C, lambdas=np.array([LAMBDA]), solver="glmnet")
    assert np.allclose(path.estimates[0], EXPECTED_M, atol=1e-6)


@requires_ncvreg
@pytest.mark.r
def test_ncvreg_reproduces_the_documented_estimate():
    path = lasso_path(SIGMA, C, lambdas=np.array([LAMBDA]), solver="ncvreg")
    assert np.allclose(path.estimates[0], EXPECTED_M, atol=1e-6)


def test_fista_reproduces_the_documented_estimate():
    """No R needed: the Python default must agree with the documented numbers."""
    path = lasso_path(SIGMA, C, lambdas=np.array([LAMBDA]), solver="fista",
                      tol=1e-14, max_iter=500_000)
    assert np.allclose(path.estimates[0], EXPECTED_M, atol=1e-6)
    m = path.estimates[0]
    assert np.all(np.diag(m) < 0)                       # stable-looking diagonal
    assert m[1, 0] == 0.0 and m[2, 1] == 0.0            # sparse off-diagonal
