"""gclm.objective.exact_fits (plan of 13 October 2026, Section 1.1): the exact fits of the Lyapunov
equation in skew coordinates, the minimum-l2 and minimum-l1 members, the uniqueness probe, and the
regression against the 7 October 2026 count of graphs satisfying the BP condition."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from gclm.config import S1Config
from gclm.data.simulate import CChoice, draw_instance, estimation_volatility, standardized_drift
from gclm.lyapunov import design_matrix, lyapunov_residual
from gclm.objective.covariance import dense_fit
from gclm.objective.exact_fits import (L1Fit, exact_fit_jacobian, exact_fit_residual, min_l1_exact_fit,
                                       min_l2_exact_fit, noisy_truth_start, offdiag_indices, skew_basis,
                                       skew_from_coordinates)


def instance(p, k=2, n=1000, seed=0):
    """A thesis data set on the correlation scale with the rescaled C."""
    rng = np.random.default_rng([seed, p, k])
    m, _, _, r, s = draw_instance(p, k, n, CChoice.ID, rng, standardize=True, return_scale=True)
    return standardized_drift(m, s), r, estimation_volatility(s, "variance")


def l1(m):
    return float(np.abs(m[~np.eye(m.shape[0], dtype=bool)]).sum())


def l2(m):
    return float(np.linalg.norm(m[~np.eye(m.shape[0], dtype=bool)]))


def test_skew_basis_coordinates_and_jacobian():
    p = 4
    basis = skew_basis(p)
    assert basis.shape == (6, p, p)
    assert all(np.array_equal(b, -b.T) for b in basis)
    iu, ju = np.triu_indices(p, 1)
    for k, (i, j) in enumerate(zip(iu, ju)):
        assert basis[k, i, j] == 1.0 and basis[k, j, i] == -1.0 and np.abs(basis[k]).sum() == 2.0
    w = np.arange(1.0, 7.0)
    skew = skew_from_coordinates(w, p)
    assert np.array_equal(skew, np.tensordot(w, basis, axes=1)) and np.array_equal(skew[iu, ju], w)
    # the jacobian: column k holds the off-diagonal entries of B_k Sigma^{-1}, rows in row-major order
    _, sigma, _ = instance(p)
    rows, cols = offdiag_indices(p)
    assert list(zip(rows, cols)) == [(i, j) for i in range(p) for j in range(p) if i != j]
    ref = np.einsum("kab,bc->kac", basis, np.linalg.inv(sigma))[:, rows, cols].T
    jac = exact_fit_jacobian(sigma)
    assert jac.shape == (p * (p - 1), p * (p - 1) // 2)
    np.testing.assert_allclose(jac, ref, atol=1e-13)
    assert np.linalg.matrix_rank(jac) == p * (p - 1) // 2         # full column rank: w is unique given M
    with pytest.raises(ValueError):
        skew_basis(1)
    with pytest.raises(ValueError):
        skew_from_coordinates(w[:5], p)


@pytest.mark.parametrize("p", [5, 10, 20])
def test_the_three_fits_are_exact_and_distinct(p):
    _, sigma, c = instance(p)
    fits = {"dense": dense_fit(sigma, c), "l2": min_l2_exact_fit(sigma, c), "l1": min_l1_exact_fit(sigma, c)}
    tol = 1e-8 * np.linalg.norm(c)
    for name, m in fits.items():
        assert exact_fit_residual(m, sigma, c) <= tol, name
        assert m.shape == (p, p) and np.isfinite(m).all()
    assert np.abs(fits["l2"] - fits["l1"]).max() > 1e-3 and np.abs(fits["l2"] - fits["dense"]).max() > 1e-3
    # the skew coordinates recover every member of the space: fit - M0 = W Sigma^{-1} with W skew
    for name in ("l2", "l1"):
        w_mat = (fits[name] - fits["dense"]) @ sigma
        np.testing.assert_allclose(w_mat, -w_mat.T, atol=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_minimum_norm_members_by_a_strict_margin(seed):
    _, sigma, c = instance(6, seed=seed)
    dense, m2, m1 = dense_fit(sigma, c), min_l2_exact_fit(sigma, c), min_l1_exact_fit(sigma, c)
    assert l2(m2) < min(l2(m1), l2(dense)) - 1e-6
    assert l1(m1) < min(l1(m2), l1(dense)) - 1e-6
    # the least-squares optimality condition: the residual m2_off is orthogonal to the columns of J
    rows, cols = offdiag_indices(6)
    assert np.abs(exact_fit_jacobian(sigma).T @ m2[rows, cols]).max() < 1e-9


def test_min_l1_agrees_with_an_explicit_design_lp():
    """The same program in the p^2 entries of M with the Lyapunov equation as equality constraints
    (A vec(M) = -vec(C), A the design matrix, one row per entry of the symmetric residual)."""
    p = 5
    _, sigma, c = instance(p, seed=4)
    a = design_matrix(sigma)
    keep = [i + j * p for j in range(p) for i in range(j + 1)]      # upper triangle of vec, column-major
    rows, cols = offdiag_indices(p)
    off = rows + cols * p                                             # vec indices of the off-diagonal entries
    nt = len(off)
    sel = np.zeros((nt, p * p))
    sel[np.arange(nt), off] = 1.0
    a_ub = np.block([[sel, -np.eye(nt)], [-sel, -np.eye(nt)]])
    a_eq = np.hstack([a[keep], np.zeros((len(keep), nt))])
    res = linprog(np.concatenate([np.zeros(p * p), np.ones(nt)]), A_ub=a_ub, b_ub=np.zeros(2 * nt), A_eq=a_eq,
                  b_eq=-c.flatten(order="F")[keep], bounds=[(None, None)] * (p * p) + [(0, None)] * nt,
                  method="highs", options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    assert res.status == 0
    m_ref = res.x[: p * p].reshape((p, p), order="F")
    m1, info = min_l1_exact_fit(sigma, c, probes=2, return_info=True)
    assert isinstance(info, L1Fit) and info.status == 0 and info.unique is True
    assert info.value == pytest.approx(l1(m_ref), abs=1e-7) and info.value == pytest.approx(l1(m1), abs=1e-12)
    np.testing.assert_allclose(m1, m_ref, atol=1e-6)
    assert np.allclose(skew_from_coordinates(info.w, p) @ np.linalg.inv(sigma) + dense_fit(sigma, c), m1)


def test_uniqueness_probe():
    """Two nodes on the correlation scale: the l1 norm is flat between the two single-edge fits (equal
    slopes 1/(1-r^2) on both entries), so the minimiser is not unique; off the correlation scale it is.
    An isolated edge inside a larger population graph has the same ambiguity of direction."""
    _, flat = min_l1_exact_fit(np.array([[1.0, 0.5], [0.5, 1.0]]), np.diag([2.0, 0.5]), probes=2, return_info=True)
    assert flat.unique is False and flat.face_extent > 0.5
    _, sharp = min_l1_exact_fit(np.array([[1.0, 0.3], [0.3, 2.0]]), 2.0 * np.eye(2), probes=2, return_info=True)
    assert sharp.unique is True and sharp.face_extent <= 1e-9
    # a generic sample instance has a unique minimiser; probes=0 leaves the question open
    _, sigma, c = instance(8, seed=5)
    _, info = min_l1_exact_fit(sigma, c, probes=3, return_info=True)
    assert info.unique is True and info.face_extent == 0.0
    _, info0 = min_l1_exact_fit(sigma, c, return_info=True)
    assert info0.unique is None and info0.face_extent == 0.0
    # population graph with one isolated edge (0 -> 1) and a chain 2 -> 3 -> 4: the edge's direction is free
    m = np.diag([-2.0, -2.5, -3.0, -2.0, -2.2])
    m[1, 0], m[3, 2], m[4, 3] = 0.8, -0.6, 0.7
    sigma_true = -np.linalg.solve(np.kron(np.eye(5), m) + np.kron(m, np.eye(5)), (2.0 * np.eye(5)).flatten(order="F")).reshape((5, 5), order="F")
    s = np.sqrt(np.diag(sigma_true))
    r, c_r = sigma_true / np.outer(s, s), np.diag(2.0 / s ** 2)
    m_tilde = standardized_drift(m, s)
    bp, info = min_l1_exact_fit(r, c_r, probes=3, return_info=True)
    assert info.unique is False and info.face_extent > 0.1
    assert info.value == pytest.approx(l1(m_tilde), abs=1e-7)       # the truth attains the minimum ...
    assert (abs(bp[1, 0]) < 1e-7) != (abs(bp[0, 1]) < 1e-7)           # ... and the LP picked one direction


def test_bad_inputs_are_refused():
    _, sigma, c = instance(4)
    with pytest.raises(ValueError):
        min_l1_exact_fit(sigma + np.triu(np.ones((4, 4)), 1) * 1e-3, c)
    with pytest.raises(ValueError):
        min_l2_exact_fit(sigma, c[:3, :3])
    with pytest.raises(ValueError):
        min_l1_exact_fit(sigma, c, solver="simplex")
    with pytest.raises(ValueError):
        min_l1_exact_fit(sigma, c, probes=-1)
    with pytest.raises(ValueError):
        exact_fit_jacobian(np.ones((1, 1)))


def test_noisy_truth_start_and_residual():
    m_true, sigma, c = instance(6)
    rng = np.random.default_rng(7)
    same = noisy_truth_start(m_true, 0.0, rng)
    assert np.array_equal(same, m_true) and same is not m_true
    noisy = noisy_truth_start(m_true, 0.1, np.random.default_rng(7))
    off = ~np.eye(6, dtype=bool)
    assert np.all(noisy[off] != m_true[off]) and np.array_equal(np.diag(noisy), np.diag(m_true))
    assert np.array_equal(noisy, noisy_truth_start(m_true, 0.1, np.random.default_rng(7)))
    assert np.std(noisy[off] - m_true[off]) == pytest.approx(0.1, rel=0.4)
    with pytest.raises(ValueError):
        noisy_truth_start(m_true, -0.1, rng)
    with pytest.raises(ValueError):
        noisy_truth_start(m_true[:2], 0.1, rng)
    # the residual: zero for an exact fit, the Frobenius norm of R(M) otherwise
    assert exact_fit_residual(dense_fit(sigma, c), sigma, c) < 1e-12
    assert exact_fit_residual(noisy, sigma, c) == pytest.approx(np.linalg.norm(lyapunov_residual(noisy, sigma, c)))
    assert exact_fit_residual(noisy, sigma, c) > 1e-3


def test_bp_condition_count_on_the_campaign_graphs():
    """Regression against the theory notes: on the thesis DGP at p = 10 (k = 1..4, 25 graphs each,
    C = 2I, the campaign's seed), population covariance on the correlation scale with the rescaled C,
    the standardized truth is the *unique* minimum-l1 exact fit in 11 of 100 graphs (7 October 2026),
    7 at k = 1 and 4 at k = 2.  It attains the minimal norm in 17; in the other 6 the face is flat
    (the direction of an isolated edge) and which vertex the LP returns depends on the solver -- the
    8 October note counted 15 returned truths, HiGHS 1.15 returns 14.  About 5 s."""
    seed = S1Config().seed
    off = ~np.eye(10, dtype=bool)
    unique_truth, attains, returned = {1: 0, 2: 0, 3: 0, 4: 0}, 0, 0
    for k in (1, 2, 3, 4):
        for rep in range(25):
            rng = np.random.default_rng([seed, 10, k, list(CChoice).index(CChoice.ID), rep])
            m, _, _, r, s = draw_instance(10, k, np.inf, CChoice.ID, rng, standardize=True, return_scale=True)
            c = estimation_volatility(s, "variance")
            m_tilde = standardized_drift(m, s)
            assert exact_fit_residual(m_tilde, r, c) <= 1e-8 * np.linalg.norm(c)
            bp, info = min_l1_exact_fit(r, c, probes=3, return_info=True)
            gap = np.abs(m_tilde[off]).sum() - info.value
            assert gap >= -1e-7                                       # the truth is feasible
            dev = np.abs(bp - m_tilde).max()
            assert dev <= 1e-6 or dev > 1e-2                          # the LP returns the truth or something else
            assert info.face_extent == 0.0 or info.face_extent > 1e-2  # unique or a face of real extent
            attains += gap <= 1e-7
            returned += dev <= 1e-6
            unique_truth[k] += info.unique and dev <= 1e-6
    assert unique_truth == {1: 7, 2: 4, 3: 0, 4: 0}
    assert attains == 17 and 11 <= returned <= 17
