"""Example 2 at n = inf: the exact solutions of M S + S M' = -C form M* + {W S^-1 : W skew}.
Which one has the smallest off-diagonal l1 norm (= the lasso's limit as lambda -> 0)?"""
import sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "/Users/joonkim/Desktop/MastersThesis/repo/src")
from gclm.data.examples import example2_cycle
from gclm.lyapunov import solve_lyapunov

m_true = example2_cycle(); p = 5; c = 2 * np.eye(p)
sigma = solve_lyapunov(m_true, c)
idx = [(a, b) for a in range(p) for b in range(p)]
rows = [(i, j) for i in range(p) for j in range(i, p)]
A = np.zeros((len(rows), p * p))
for col, (a, b) in enumerate(idx):
    e = np.zeros((p, p)); e[a, b] = 1.0
    x = e @ sigma + sigma @ e.T
    A[:, col] = [x[i, j] for i, j in rows]
rhs = np.array([-c[i, j] for i, j in rows])
off = np.array([a != b for a, b in idx])
# variables: M = Mp - Mn, both >= 0 off-diagonal; diagonal free (unpenalised)
n = p * p
cost = np.concatenate([off.astype(float), off.astype(float)])
bounds = [(0, None) if o else (None, None) for o in off] + [(0, None) if o else (0, 0) for o in off]
res = linprog(cost, A_eq=np.hstack([A, -A]), b_eq=rhs, bounds=bounds, method="highs")
m_bp = (res.x[:n] - res.x[n:]).reshape(p, p)
fmt = lambda m: sorted(f"{b + 1}->{a + 1}" for a in range(p) for b in range(p) if a != b and abs(m[a, b]) > 1e-9)
print("truth support      :", fmt(m_true), " l1 =", round(np.abs(m_true[~np.eye(p, dtype=bool)]).sum(), 4))
print("min-l1 exact fit   :", fmt(m_bp), " l1 =", round(res.fun, 4))
print("residual of min-l1 :", np.abs(m_bp @ sigma + sigma @ m_bp.T + c).max())
