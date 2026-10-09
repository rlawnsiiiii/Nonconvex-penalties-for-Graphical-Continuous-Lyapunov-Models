# Why the dense start works: theory for the adaptive lasso and MCP from the dense end (8 October 2026)

*Answers: "at the end we want theory why starting from dense graphs, e.g. the adaptive lasso, performs
well; at least some guarantees." Builds on `main_claim_and_theory_071026.md` (§2.4, §4) and on the
campaign results of 8 October (`simulations/S4_campaign.md`, `simulations/VERDICTS.md` at `b65ca5a`).
The raw campaign CSVs are not in the repository, so the campaign numbers below are quoted from S4; the
new numbers are population computations on the campaign's own graphs (Figure 5 generator, `C_ID`, the
repository's seeds, scripts in §9).*

*Revised 8 October (evening): new §2a explains the design $A(\Sigma)$ with Dettling's Appendix C, and
Lemma 1 now has an elementary proof with a two-node example.*

*Status words. **Proved**: full argument below (short, standard tools). **Sketch**: the argument is
standard but some details are not written out. **Numerical**: population computation only.
**Cite**: published result; check theorem numbers against the PDF before quoting.*

---

## 0. Short answer

**Yes, there is a theory, and it has one central idea: the dense start removes the null space.**

The direct loss is a regression whose design $A(\Sigma)$ has a null space of dimension $p(p-1)/2$
(every $W\Sigma^{-1}$ with $W$ skew). That null space is the source of every difficulty of the problem:
it is why the lasso needs an irrepresentability condition that fails in every graph, why no restricted
eigenvalue condition holds, and why reversing an edge is almost free for the loss (the lock-in). The
dense end of the lasso path is, up to $O(\lambda_{\min})$, the minimum-$\ell_1$ exact fit $M^{B}$ ("BP").
Generically its support plus the diagonal indexes **linearly independent** columns of the design
(Lemma 1; §2a explains the design and what independent columns mean). The adaptive lasso only ever selects inside that support, so **it solves an ordinary,
full-rank weighted lasso** (Lemma 3). Everything that is hard about the original problem is gone; what
is left is whether the dense end got the support right.

From this:

1. **Guarantee (Theorem 5, fixed $p$, $n \to \infty$).** If $M^*$ is the unique minimum-$\ell_1$ exact
   solution of the population Lyapunov equation (the *BP condition*), the adaptive lasso with the dense
   lasso pilot is selection-consistent and asymptotically normal on the true support (Zou's oracle
   property). No irrepresentability, no restricted eigenvalue condition. The same holds for MCP / SCAD by
   two LLA steps from the dense end (Corollary 6).
2. **It is strictly better than what the lasso can do (Proposition 7).** Strict irrepresentability ⇒
   BP condition ⇒ local identifiability. On the thesis graphs at $p = 10$: 0, 15 and 98 of 100 graphs.
3. **Without the BP condition it still converges, to a computable target (Theorem 8).** The sample
   path converges to a deterministic population path that selects only inside $\operatorname{supp}
   M^B$. Its quality is set by two properties of the dense end: how many true edges it contains, and
   whether it gives the true direction the larger weight. Both improve with $p$ (0.80 → 0.92 → 0.96 → 0.98
   recall, 0.76 → 0.92 → 0.99 → 0.99 direction at $p = 10, 20, 30, 50$), which is the campaign's "the gain grows
   with $p$". The population gain over the lasso correlates 0.79 ($p = 10$) and 0.75 ($p = 20$) with
   the recall of the dense end.
4. **Why the dense end gets directions right (Proposition 9, exact).** Inside the set of exact fits,
   reversing a true edge forces "leakage" into the other entries of the two rows, and that leakage costs
   $\ell_1$. The cost is a closed-form expression in $\Sigma^{-1}$; it is positive for 96 % (p = 10) and
   99 % (p = 20) of the true edges, while its variance part alone is a coin flip (44 %). So the dense end
   orients edges with exactly the second-order, neighbour-dependent information that the standard path
   cannot use (it orients before the neighbours are in the fit).
5. **Why the adaptive lasso is at least as good as MCP from the same start, and why the Varando–Hansen
   start does not help** (§6) follow from the same picture.

What is **not** available: a guarantee for the average $F_1$ gain where the BP condition fails, rates in
$p$ (in particular $p > n$), and a proof that the dense end orients most edges correctly at large $p$
(only the local version, Proposition 9).

---

## 1. What the theory has to explain (campaign, S4)

| | fact | S4 |
|---|---|---|
| F1 | MCP / SCAD dense → sparse and the adaptive lasso beat the lasso in every cell but one tie; up to +0.15 in `max_f1` | §2 |
| F2 | the adaptive lasso is at least as good as MCP dense → sparse in `max_f1` and better in `aupr` (+0.08 … +0.14), in $F_1$ of the selected graph and after the search; LLA gets three quarters of the MCP gain | §4 |
| F3 | the gain grows with $n$ and with $p$; at $n = 10^3$ the lasso's $F_1$ falls from 0.59 to 0.43–0.47 between $p = 10$ and 50, the adaptive lasso's only to 0.55–0.57 | §2, §6a |
| F4 | at $p = 10$, $k = 3, 4$ the dense start loses slightly | §2 |
| F5 | the correctly specified $C$ roughly doubles the gain; a non-diagonal true $C$ removes it at $p = 10$ | §2, §3 |
| F6 | on the log-likelihood loss, MCP dense → sparse from the Varando–Hansen start $-\tfrac12 C\hat\Sigma^{-1}$ does not beat the lasso | §6 |
| F7 | the dense-start estimators' graphs commit to one direction and are right (orientation accuracy 0.89) | §4 figure |

## 2. Setting

$m = \operatorname{vec}(M)$, $A = A(\Sigma)$ the design of $M \mapsto M\Sigma + \Sigma M^\top$, $c =
\operatorname{vec}(C)$, direct loss $L(m) = \tfrac12\lVert A m + c\rVert^2$, Gram $\Gamma = A^\top A$.
Hats for the sample versions. $f(M) = \sum_{i \ne j}\lvert M_{ij}\rvert$, the off-diagonal $\ell_1$ norm.

**Sets of positions.** A *position* is a pair $(i,j)$, i.e. one entry of $M$ and one column of $A$. All
sets of positions used below contain the whole diagonal, because the diagonal is never penalised and is
nonzero in every fit ($M_{ii} < 0$).

| symbol | meaning |
|---|---|
| $S$ | the true edges: positions $(i,j)$, $i \ne j$, with $M^*_{ij} \ne 0$ |
| $\bar S$ | $S$ plus the diagonal |
| $T$ | a generic set of positions, diagonal included: "only these entries of $M$ may be nonzero" |
| $T(M)$ | the positions a given matrix $M$ uses: its nonzero off-diagonal entries plus the diagonal |
| $T_B = T(M^B)$ | the positions the dense end uses; $T_B$ is to $M^B$ what $\bar S$ is to $M^*$ |

$A_T$ is the matrix of the columns of $A$ in $T$, and $\Gamma_{TT} = A_T^\top A_T$. "The columns of $T$ are
independent" means: if only the entries in $T$ may be nonzero, there is at most one exact fit (§2a item 5).

- **Exact fits.** $\mathcal E(\Sigma) = \{M : M\Sigma + \Sigma M^\top + C = 0\} = (W - C/2)\Sigma^{-1}$,
  $W$ skew: an affine space of dimension $p(p-1)/2$, the null space $N = \{W\Sigma^{-1}\}$.
- **The dense end.** $M^B \in \arg\min\{f(M) : M \in \mathcal E(\Sigma)\}$, a linear programme. The
  lasso $\hat M(\lambda)$ tends to it as $\lambda \to 0$ (Lemma 4b).
- **Adaptive lasso** (`--method adaptive`): pilot $M^0$ = lasso at $\lambda_{\min}$, weights $w_{ij} =
  1/\lvert M^0_{ij}\rvert$, $w_{ij} = \infty$ where $M^0_{ij} = 0$, then
  $\hat M^{\mathrm{ada}}(\lambda) = \arg\min L(M) + \lambda\sum w_{ij}\lvert M_{ij}\rvert$.
- **One family.** A weighted lasso with weights $P'(\lvert M^0_{ij}\rvert)/\lambda$ is one LLA step
  (Zou & Li 2008) of the concave penalty $P$ from the pilot $M^0$. The adaptive lasso is the case
  $P(t) = \log t$ ($P' = 1/t$), MCP-LLA the case $P' = (\lambda - t/\gamma)_+$. So all three campaign
  winners are "concave penalty, one or a few local steps, from the dense end". They differ in the weight
  curve: $1/t$ is unbounded at 0 (small pilot entries are penalised very hard, zeros are excluded), MCP's
  is capped at 1 (a small spurious entry is penalised like in the lasso, nothing is excluded). §6 uses
  this.

## 2a. The design $A(\Sigma)$, concretely

*Everything in §3 is a statement about the columns of $A(\Sigma)$, so this section spells out what they
are, using Dettling, Drton & Kolar (2024), Appendix C (Example C.1, the display of $A(\Sigma)$ for
$p = 3$).*

**1. Unknowns, equations, rows, columns.** The Lyapunov equation $M\Sigma + \Sigma M^\top = -C$ is a
system of linear equations:

- the **unknowns** are the $p^2$ entries of $M$, stacked column by column into $m = \operatorname{vec}(M)$:
  $(1,1), (2,1), (3,1), (1,2), \dots$ These are Dettling's **column labels**;
- the **equations** are the entries $(k,l)$ of $M\Sigma + \Sigma M^\top = -C$, one per entry. These are his
  **row labels**;
- $A(\Sigma)\,m = \operatorname{vec}(M\Sigma + \Sigma M^\top)$, so **row $(k,l)$ of $A$ is equation $(k,l)$**
  and **column $(i,j)$ of $A$ holds the coefficients of the unknown $M_{ij}$ in all equations.**

Since $(M\Sigma + \Sigma M^\top)_{kl} = \sum_m M_{km}\Sigma_{ml} + \sum_m \Sigma_{km}M_{lm}$, the entry of $A$ in
row $(k,l)$ and column $(i,j)$ is

$$A_{(k,l),(i,j)} = [k = i]\,\Sigma_{jl} + [l = i]\,\Sigma_{kj} .$$

($[\cdot]$ is 1 if the condition holds, 0 otherwise.) His display lists the rows row by row, (1,1), (1,2),
(1,3), (2,1), …, while the columns follow the column-stacking order; because $M\Sigma + \Sigma M^\top$ is
symmetric, the two orders describe the same set of equations.

**2. Reading a row.** Row (1,2) of his display says

$$\Sigma_{21}M_{11} + \Sigma_{11}M_{21} + \Sigma_{22}M_{12} + \Sigma_{12}M_{22} + \Sigma_{23}M_{13} + \Sigma_{13}M_{23} = -C_{12},$$

which is the (1,2) entry of $M\Sigma + \Sigma M^\top = -C$ written out.

**3. Reading a column.** Column (1,2) belongs to the unknown $M_{12}$ (the edge $2 \to 1$). Top to bottom:

| row | (1,1) | (1,2) | (1,3) | (2,1) | (2,2) | (2,3) | (3,1) | (3,2) | (3,3) |
|---|---|---|---|---|---|---|---|---|---|
| column (1,2) | $2\Sigma_{12}$ | $\Sigma_{22}$ | $\Sigma_{23}$ | $\Sigma_{22}$ | 0 | 0 | $\Sigma_{23}$ | 0 | 0 |

Folded back into a $3 \times 3$ grid (rows of the grid = first index of the row label):

$$G_{12} = \begin{pmatrix} 2\Sigma_{12} & \Sigma_{22} & \Sigma_{23}\\ \Sigma_{22} & 0 & 0\\ \Sigma_{23} & 0 & 0\end{pmatrix}
= E_{12}\Sigma + \Sigma E_{12}^\top .$$

In general **$G_{ij} = E_{ij}\Sigma + \Sigma E_{ij}^\top$ is what $M\Sigma + \Sigma M^\top$ becomes when $M = E_{ij}$**,
the matrix with a single 1 at position $(i,j)$; the column of $A$ for $M_{ij}$ is $G_{ij}$ listed as a
vector. $E_{ij}\Sigma$ puts row $j$ of $\Sigma$ into row $i$, $\Sigma E_{ij}^\top$ puts column $j$ of $\Sigma$
into column $i$, and the two overlap at $(i,i)$ with $2\Sigma_{ij}$. So the column for $M_{ij}$ is nonzero
only in row $i$ and column $i$ of the grid, and its values come from column $j$ of $\Sigma$. The diagonal
column (1,1), for example, is $G_{11}$ with first row $(2\Sigma_{11}, \Sigma_{12}, \Sigma_{13})$, first column
the same, zeros elsewhere: exactly column (1,1) of the display.

**4. Duplicate rows, rank, null space.** $M\Sigma + \Sigma M^\top$ is symmetric, so equation $(2,1)$ is
equation $(1,2)$ again. The italic rows of the display, (2,1), (3,1), (3,2), are copies. So $A$ has $p^2$
columns but only $p(p+1)/2$ different equations (6 for $p = 3$), and:

- **$\operatorname{rank} A = p(p+1)/2$.** It cannot be more (there are only that many different rows), and
  it is exactly that because every symmetric $Y$ is reached: $M = \tfrac12 Y\Sigma^{-1}$ gives $M\Sigma +
  \Sigma M^\top = \tfrac12 Y + \tfrac12\Sigma\Sigma^{-1}Y = Y$.
- **Null space of dimension $p^2 - p(p+1)/2 = p(p-1)/2$** (3 for $p = 3$): directions $h$ with $Ah = 0$,
  i.e. changes of $M$ that do not change the fit. They are $h = W\Sigma^{-1}$ with $W$ skew-symmetric: $W\Sigma^{-1}\Sigma + \Sigma\Sigma^{-1}W^\top = W + W^\top = 0$.
- **All exact fits:** one particular solution, $-\tfrac12 C\Sigma^{-1}$ (put $Y = -C$ above), plus the null
  space: $\mathcal E(\Sigma) = \{(W - \tfrac12 C)\Sigma^{-1} : W^\top = -W\}$. The choice $W = 0$ is the
  Varando–Hansen dense start of F6.

**5. "The columns of a set $T$ are linearly independent", in words.** ($T$ is a set of positions, diagonal
included; see the table in §2.) Allow only the unknowns in $T$ to be
nonzero. The columns of $T$ are independent if and only if no nonzero $h$ supported on $T$ has $Ah = 0$, if
and only if **there is at most one exact fit supported on $T$**, if and only if $\Gamma_{TT} = A_T^\top A_T$ is
positive definite (the loss restricted to $T$ is strictly convex). At most $\operatorname{rank}A =
p(p+1)/2$ columns can be independent.

The diagonal columns are always independent: a diagonal $D$ with $D\Sigma + \Sigma D = 0$ has $(i,i)$ entry
$2D_{ii}\Sigma_{ii} = 0$, and $\Sigma_{ii} > 0$, so $D = 0$.

**6. The two-node case.** $p = 2$, correlation scale: $\Sigma_{11} = \Sigma_{22} = 1$, $\Sigma_{12} = r$,
$\lvert r\rvert < 1$. In Dettling's format (columns in his order):

| row | $M_{11}$ | $M_{21}$ (edge $1 \to 2$) | $M_{12}$ (edge $2 \to 1$) | $M_{22}$ |
|---|---|---|---|---|
| (1,1) | 2 | 0 | $2r$ | 0 |
| (1,2) | $r$ | 1 | 1 | $r$ |
| *(2,1)* | $r$ | 1 | 1 | $r$ |
| (2,2) | 0 | $2r$ | 0 | 2 |

Four unknowns, three different equations, so a one-dimensional null space. It is $h = W\Sigma^{-1}$ with
$W = \bigl(\begin{smallmatrix}0 & 1\\ -1 & 0\end{smallmatrix}\bigr)$, proportional to

$$(h_{11}, h_{21}, h_{12}, h_{22}) = (-r,\ -1,\ 1,\ r):$$

$-r$ times the $M_{11}$ column, $-1$ times the $M_{21}$ column, $+1$ times the $M_{12}$ column and $+r$ times
the $M_{22}$ column add up to zero in every row. Adding $t\cdot h$ to any exact fit raises $M_{12}$, lowers
$M_{21}$ by the same amount and adjusts the diagonal, without changing the fit: **an edge reversal at no cost
to the loss.** This is the null space in its smallest form, and the reason two nodes alone cannot be
oriented (lock-in note, Corollary 3.2).

Delete the column of one direction, say $M_{21}$. The remaining three columns, restricted to the three
different rows, form the matrix with columns $(2, r, 0)$, $(0, r, 2)$, $(2r, 1, 0)$, whose determinant is
$-4(1 - r^2) \ne 0$: they are independent, and "diagonal plus one direction" pins down a unique exact fit.

(The spreadsheet `lemma1_columns_of_A.xlsx` contains this example with $r$ as an input: the columns, the
null combination, the determinant and the sliding step of Lemma 1's proof.)

## 3. Structure of the dense end

**Lemma 1 (the dense end has independent columns; proved).** Among the minimisers of $f$ over
$\mathcal E(\Sigma)$ there is one, $M^B$, for which the columns of $A$ indexed by
$T_B = T(M^B)$ (its nonzero off-diagonal entries plus the diagonal) are linearly independent. Consequently
$\lvert\operatorname{supp}_{\text{off}} M^B\rvert \le p(p+1)/2 - p = p(p-1)/2$. The LP solver returns such a
minimiser.

*Proof.* A minimiser exists: the diagonal of an exact fit is determined by its off-diagonal part (the
diagonal columns are independent, §2a item 5), so the exact fits with $f \le f(M_0)$, for any exact fit
$M_0$, form a closed and bounded set, on which the continuous $f$ attains its minimum. Take any minimiser $M$ and let $T = T(M)$ be the
positions it uses: its nonzero off-diagonal entries plus the diagonal.

0. If the columns of $T$ are independent, stop.
1. Otherwise there is $h \ne 0$ supported on $T$ with $Ah = 0$ (§2a item 5), so $M + t\cdot h$ is an exact
   fit for every real $t$.
2. $h$ has a nonzero off-diagonal entry, because the diagonal columns are independent (§2a item 5).
3. **$f$ is flat along $h$.** For small $\lvert t\rvert$ no nonzero entry of $M$ changes sign, so
   $f(M + t\cdot h) = f(M) + t\sum_{i \ne j}\operatorname{sign}(M_{ij})\,h_{ij}$ is linear in $t$ (entries outside
   $T$ stay 0, the diagonal does not count). A nonzero slope would let one of the two directions $t > 0$,
   $t < 0$ lower $f$, contradicting minimality. So the slope is 0.
4. Increase $\lvert t\rvert$ (in either direction) until the first off-diagonal entry of $M + t\cdot h$ reaches
   0. Up to that point no sign changes, so $f$ stays equal to $f(M)$: the new matrix is still a minimiser,
   with at least one fewer nonzero off-diagonal entry.
5. Repeat from 0. Each round removes an entry, so after finitely many rounds the process stops, at a
   minimiser whose columns are independent.

**Count.** Independent columns cannot be more than $\operatorname{rank}A = p(p+1)/2$ (§2a item 4). The
$p$ diagonal columns are among them, so at most $p(p-1)/2$ off-diagonal entries remain. $\square$

*Two-node illustration.* Take an exact fit with $M_{12} = 0.3$, $M_{21} = 0.2$, so $f = 0.5$. Along
$h = (-r, -1, 1, r)$ the slope of $f$ is $\operatorname{sign}(M_{12})\cdot 1 + \operatorname{sign}(M_{21})\cdot(-1) = 0$.
At $t = 0.2$ the entry $M_{21}$ reaches 0, $M_{12}$ becomes 0.5 and $f$ is still 0.5: a minimiser with one
direction only, whose columns are independent (§2a item 6). The bound $p(p-1)/2 = 1$ says the same: the
minimum-$\ell_1$ fit keeps one direction of the pair.

*In linear-programming language* (how the statement is usually quoted): writing $M_{ij} = x^+_{ij} -
x^-_{ij}$ with $x^\pm \ge 0$ turns the problem into a linear programme; a linear programme with an optimum
attains it at a vertex of the feasible polyhedron, and at a vertex the columns of the nonzero variables are
independent. The argument above is that fact proved by hand. HiGHS returns a vertex.

*Caveat.* The lemma says that *some* minimiser has independent columns. If the minimiser is not unique,
others can have more nonzeros: in the two-node example on the correlation scale every point of the segment
between "only $M_{12}$" and "only $M_{21}$" is a minimiser (the tie of an isolated edge), and its interior
points have both directions nonzero. The LP solver returns an end point.

At $p = 10$ the bound is reached ("saturated") in 66 % of the thesis graphs, at $p = 20$ in 57 %, at
$p = 30$ in 45 %, at $p = 50$ in 17 %.

**Why independence matters.** It means $M^B$ is pinned down by its own support: keeping all other entries
at zero, no change of its entries still fits exactly. In regression terms, there is no collinearity among
the selected variables; the typical collinearity here is an edge with both directions allowed, as in the
two-node example. Two places use it. The adaptive lasso only uses positions in $T_B$ (infinite weights
elsewhere), so independence makes it an ordinary regression with identified coefficients, without the free
reversal moves that defeat the lasso (Lemma 3, Theorem 5). And if the dense end contains every true edge, the
truth also lives on $T_B$, where only one exact fit exists (Lemma 2). We need the property only for the
dense end we actually use; Lemma 1 guarantees that one exists.

**Lemma 2 (screening ⟺ recovery; proved).** Let $M^B$ be the minimiser of Lemma 1, so that the columns of
$A$ for the positions in $T_B$ are linearly independent. (Each position $(i,j)$ has its own column of $A$,
the coefficients of the unknown $M_{ij}$; §2a items 1 and 3. Independence of these columns means that at
most one exact fit uses only positions in $T_B$; §2a item 5.) If every true edge is nonzero in $M^B$, i.e.
$S \subseteq \operatorname{supp}_{\text{off}} M^B$, then $M^B = M^*$.

*Proof.* $M^*$ uses only positions in $\bar S \subseteq T_B$, and $M^B$ uses only positions in $T_B$. Both
are exact fits, and at most one exact fit uses only positions in $T_B$. So $M^* = M^B$. $\square$

*Two-node example.* If $M^B$ keeps only the edge $2 \to 1$, then $T_B = \{(1,1), (2,2), (1,2)\}$, whose
columns $M_{11}, M_{22}, M_{12}$ in the table of §2a item 6 are independent (determinant $-4(1 - r^2)$). A
true graph consisting of the edge $2 \to 1$ uses only these positions, so it equals $M^B$.

*So the dense end either is the truth, or misses at least one true edge.* This is the 278-of-278
observation of 7 October, now explained. Numerically again today: `bp_true` = `S ⊆ supp BP` in all
380 graph-scale combinations.

**Lemma 3 (the null space is gone; proved).** The adaptive lasso is supported in $T_B$ and, restricted to
$T_B$, is a weighted lasso with positive definite Gram $\Gamma_{T_BT_B}$: strictly convex, unique
solution at every $\lambda$, path continuous in $\lambda$ and in $\hat\Sigma$.

*Proof.* Infinite weights fix the entries outside $T_B$ at 0; Lemma 1. $\square$

For the repository's pilot (the lasso at $\lambda_{\min}$, computed by FISTA, not the LP) the same holds
when its active columns are independent. A lasso problem always has a solution with independent active
columns, and under general position it is the only one (Tibshirani 2013, *The lasso problem and
uniqueness*); whether FISTA returns it is not guaranteed. Using the LP as the pilot removes the question.

This is the structural reason the dense start helps: the lasso, MCP on the standard path and any
sparse → dense method work on a design with a $p(p-1)/2$-dimensional null space; the adaptive lasso works
on a full-rank design selected by an $\ell_1$-optimal exact fit.

**Proposition 4 (the dense end is stable; proved, standard tools).**

(a) If the population LP has a unique solution $M^B$, then every minimiser $\hat M^B$ of the sample LP
satisfies $\lVert\hat M^B - M^B\rVert = O_P(\lVert\hat\Sigma - \Sigma\rVert) = O_P(n^{-1/2})$.

(b) The sample lasso at $\lambda_0$ satisfies $\lVert\hat M(\lambda_0) - M^B\rVert = O_P(n^{-1/2} +
\lambda_0)$.

*Proof.* (a) A polyhedral function with a unique minimiser on a polyhedron has a *sharp* minimum (Burke &
Ferris 1993): $f(M) - f(M^B) \ge \kappa\lVert M - M^B\rVert$ for $M \in \mathcal E(\Sigma)$. $A(\Sigma)$ is
onto the symmetric matrices with a right inverse that is bounded near $\Sigma$, so any $M$ can be moved
onto $\mathcal E(\Sigma)$ or $\mathcal E(\hat\Sigma)$ at cost $O(\varepsilon_n\lVert M\rVert)$,
$\varepsilon_n = \lVert\hat\Sigma - \Sigma\rVert$. Move $M^B$ into $\mathcal E(\hat\Sigma)$: $f(\hat M^B)
\le f(M^B) + O(\varepsilon_n)$. Move $\hat M^B$ into $\mathcal E(\Sigma)$ (it is bounded: its off-diagonal
by the previous inequality, its diagonal by the independent diagonal columns) to $M'$: $\kappa\lVert M' -
M^B\rVert \le f(M') - f(M^B) = O(\varepsilon_n)$. (b) Lasso optimality against $\hat M^B$ gives a residual
$\lVert r\rVert \le 2C\lambda_0$; projecting $\hat M(\lambda_0)$ onto $\mathcal E(\hat\Sigma)$ gives a
$O(\lambda_0)$-optimal point of the sample LP, and (a)'s argument converts near-optimality into distance.
$\square$

The repository's pilot uses $\lambda_{\min} = 10^{-4}\lambda_{\max}$, a fixed ratio; for the theorems it
should be read as $\lambda_0 \lesssim n^{-1/2}$, or the LP itself (`methods.bp_lp` of the independent
study) used as the pilot.

## 4. Guarantees under the BP condition

**BP condition.** $M^*$ is the unique minimiser of $f$ over $\mathcal E(\Sigma^*)$. Equivalently
(Lemma 2): the population dense end contains every true edge.

**Theorem 5 (oracle property of the adaptive lasso; proved along Zou 2006).** Fixed $p$, correctly
specified $C$, $\hat\Sigma$ from $n$ i.i.d. Gaussian observations, BP condition. Pilot $\hat M^B$ or
$\hat M(\lambda_0)$ with $\lambda_0 = O(n^{-1/2})$. Tuning $\lambda_n$ (on the scale of the repository's
objective $L + \lambda\sum w\lvert M\rvert$) with $\sqrt n\lambda_n \to 0$ and $n\lambda_n \to \infty$. Then

1. $P(\operatorname{supp}\hat M^{\mathrm{ada}}(\lambda_n) = S) \to 1$;
2. $\sqrt n(\hat M^{\mathrm{ada}}_{\bar S} - M^*_{\bar S}) \to N(0, \Gamma_{\bar S\bar S}^{-1}\,\Omega\,
   \Gamma_{\bar S\bar S}^{-1})$, $\Omega$ the covariance of $W_{\bar S}$ below (a sandwich, because the
   direct loss is not a likelihood).

*Proof.* Three ingredients.

- *$\Gamma_{\bar S\bar S} \succ 0$ follows from the BP condition.* If $h \ne 0$ were a null vector
  supported in $\bar S$, then $f(M^* + th)$ is linear in small $t$ (the signs on $S$ do not change), so one
  direction does not increase $f$, contradicting uniqueness; $h$ cannot be purely diagonal (§2a item
  5).
- *Pilot.* Proposition 4: $\sqrt n(\hat M^0 - M^*) = O_P(1)$. So the weights on $S$ converge to
  $1/\lvert M^*_{ij}\rvert$, and on $S^c$ they are at least of order $\sqrt n$ (or infinite).
- *Zou's argument with a singular Gram.* With $u = \sqrt n(m - m^*)$,
  $n[\hat L(m^* + u/\sqrt n) - \hat L(m^*)] = \tfrac12 u^\top\hat\Gamma u + u^\top\sqrt n\,\hat A^\top(\hat A
  m^* + c)$. Since $\hat A m^* + c = (\hat A - A)m^* = \operatorname{vec}(M^*\Delta + \Delta M^{*\top})$,
  $\Delta = \hat\Sigma - \Sigma^*$, the linear term converges to a Gaussian $W$. The penalty, multiplied by
  $n$, contributes $n\lambda_n\hat w_{ij}(\lvert M^*_{ij} + u_{ij}/\sqrt n\rvert - \lvert M^*_{ij}\rvert)
  \approx \sqrt n\lambda_n\hat w_{ij}\,\mathrm{sgn}(M^*_{ij})u_{ij} \to 0$ on $S$, and $n\lambda_n\lvert
  u_{ij}\rvert/\lvert\sqrt n\hat M^0_{ij}\rvert \to \infty$ on $S^c$ (because $n\lambda_n \to \infty$ and
  $\sqrt n\hat M^0_{ij} = O_P(1)$ there). The limit
  $V(u) = \tfrac12 u^\top\Gamma u + W^\top u$ if $u_{S^c} = 0$, $+\infty$ otherwise, has the unique minimiser
  $u_{\bar S} = -\Gamma_{\bar S\bar S}^{-1}W_{\bar S}$; Zou only uses positive definiteness of the Gram to
  get this uniqueness, and here it comes from $\Gamma_{\bar S\bar S} \succ 0$ plus the infinite penalty off
  $S$. The convexity lemma (Geyer 1994; Knight & Fu 2000) gives (2), and the KKT argument of Zou's
  Theorem 2 gives (1): an entry of $S^c$ can be nonzero only if the gradient of $n\hat L$ there, which is
  $O_P(\sqrt n)$ by (2), equals $n\lambda_n\hat w_{ij} = \sqrt n\cdot n\lambda_n/\lvert\sqrt n\hat
  M^0_{ij}\rvert \gg \sqrt n$. $\square$

**Corollary 6 (MCP / SCAD from the dense end; sketch).** Same assumptions, $\lambda_n \to 0$ with
$\sqrt n\lambda_n \to \infty$. (a) Two LLA steps for MCP or SCAD started from $\hat M^B$ give the oracle
estimator (least squares on $\bar S$) with probability → 1. This is Fan, Xue & Zou (2014), Theorems 1–2,
whose three failure probabilities all vanish: the pilot is within $a_0\lambda_n$ of $M^*$ (Proposition 4,
not a restricted eigenvalue condition), the oracle's gradient off $S$ is $O_P(n^{-1/2}) \ll \lambda_n$ (the
true model fits exactly), and the oracle's smallest entry exceeds $a\lambda_n$. Uniqueness of the weighted
lasso solution follows from strict dual feasibility and $\Gamma_{\bar S\bar S} \succ 0$. (b) In the
population, the dense → sparse MCP continuation starts at $M^B = M^*$, which is a stationary point and
strict local minimiser for $\lambda < \min_S\lvert M^*_{ij}\rvert/\gamma$
(`main_claim_and_theory_071026.md`, T8); the monotone APG started there does not move. So the population
dense → sparse path contains $S$.

*Note:* the repository's `--method lla` starts from the lasso at the same $\lambda$, not from the dense
end; it is not covered. The campaign's LLA gets three quarters of the dense → sparse gain (F2), which fits.

**Proposition 7 (a strict hierarchy; proved).** Population, correctly specified $C$.

$$\text{strict irrepresentability} \;\Longrightarrow\; \text{BP condition} \;\Longrightarrow\;
\Gamma_{\bar S\bar S} \succ 0 .$$

*Proof of the first arrow* (Fuchs 2004, with an unpenalised diagonal). Let $z$ be the sign of $M^*$ on
$S$ and 0 on the diagonal, $q = A_{\bar S}\Gamma_{\bar S\bar S}^{-1}z$. Then $A_{\bar S}^\top q = z$ and
strict irrepresentability says $\lvert A_j^\top q\rvert < 1$ off $S$. For an exact fit $M^* + h$
($Ah = 0$): $f(M^* + h) - f(M^*) \ge z^\top h_{\bar S} + \lVert h_{S^c}\rVert_1 = -q^\top A_{S^c}h_{S^c} +
\lVert h_{S^c}\rVert_1 \ge \sum_{S^c}(1 - \lvert A_j^\top q\rvert)\lvert h_j\rvert$, positive unless
$h_{S^c} = 0$, and then $h = 0$ because $\Gamma_{\bar S\bar S} \succ 0$. Second arrow: Theorem 5's proof.
$\square$

What each condition buys: the lasso path contains $S$ only under (weak) irrepresentability (Dettling,
Drton & Kolar 2024, Prop. G.2); the adaptive lasso, LLA from the dense end and the population dense →
sparse path under the BP condition; MCP has a good local minimum under local identifiability, but nothing
says an algorithm finds it.

| how often (population, rescaled $C$) | $p = 10$ (100 graphs) | $p = 20$ (40) | $p = 30$ (20) | $p = 50$ (12) |
|---|---|---|---|---|
| weak irrepresentability (lasso can recover) | 0 | 0 | — | — |
| BP condition (adaptive lasso can recover) | 15 | 7 | 5 | 1 |
| local identifiability (a good local minimum exists) | 98 | 40 | — | — |

(15 instead of 11 at $p = 10$: the 7 October count required a unique minimiser; here the LP solver
returned $M^*$, which happens in 4 more graphs whose only ambiguity is the direction of an isolated edge.
The population adaptive-lasso path on the 100-point grid recovers $S$ exactly in 14 of the 15 at
$p = 10$ and 4 of the 7 at $p = 20$; every miss has a true entry of size 0.003 to 0.008, which the
smallest $\lambda$ of the grid already removes: the minimum-signal condition of Theorem 5, not a failure
of the theory.)

## 5. Without the BP condition: what the adaptive lasso converges to

**Theorem 8 (convergence to the population adaptive path; sketch).** Fixed $p$, correctly specified $C$,
the population LP has a unique solution $M^B$ (not necessarily $M^*$), nondegenerate. For each fixed
$\lambda > 0$, $\hat M^{\mathrm{ada}}(\lambda) \to M^{\mathrm{ada}}(\lambda)$ in probability, the weighted
lasso of the *population* loss with weights $1/\lvert M^B\rvert$ on $T_B$ and $\infty$ elsewhere; at every
$\lambda$ where the population solution has strict dual feasibility, the supports converge too. Hence the
oracle-$\lambda$ $F_1$ on a fixed grid converges to a deterministic number, and

$$F_1 \;\le\; \frac{2\,\lvert S \cap \operatorname{supp}M^B\rvert}{\lvert S\cap\operatorname{supp}M^B\rvert + \lvert S\rvert}
\qquad\text{(the dense end's recall ceiling).}$$

*Sketch.* Proposition 4(a) with $M^B$ in place of $M^*$ gives the pilot; weights outside $T_B$ diverge;
on $T_B$ the limit objective is strictly convex (Lemma 3), so the argmin converges (epi-convergence of
convex functions). Support convergence: nonzeros of the limit stay nonzero, zeros with strict dual
feasibility stay zero. $\square$

So, at large $n$, the adaptive lasso is exactly as good as the population object "weighted lasso on the
dense end's support", which can be computed for every graph. The comparison with the lasso then becomes a
deterministic comparison of two population paths:

| population, rescaled $C$, `C_ID` | $p = 10$ (100) | $p = 20$ (40) | $p = 30$ (20) | $p = 50$ (12) |
|---|---|---|---|---|
| lasso path, best $F_1$ | 0.692 | 0.700 | — | — |
| adaptive lasso path, best $F_1$ | 0.774 | 0.884 | — | — |
| gain; graphs with gain > 0 / < 0 | +0.081 (se 0.012); 74 / 23 | +0.183 (se 0.012); 39 / 1 | — | — |
| recall ceiling of the dense end | 0.882 | 0.958 | — | — |
| dense end: recall of $S$ | 0.80 | 0.92 | 0.96 | 0.98 |
| dense end: true direction has the larger weight (single-direction edges) | 0.76 | 0.92 | 0.99 | 0.99 |
| correlation over graphs: gain vs recall of the dense end | 0.79 | 0.75 | | |

At $p = 30, 50$ only the dense end was computed (the population paths are too slow for this sandbox).
There the BP condition almost never holds (5 of 20, 1 of 12), so Theorem 5 rarely applies literally,
but the dense end contains 96–98 % of the true edges and gives the true direction the larger weight for
99 % of them; its many extra entries (about 1 160 of the 1 225 possible at $p = 50$, $k \ge 2$) are tiny
and carry the largest weights. Theorem 8's ceiling is then close to 1, which is what F3 needs: the
adaptive lasso's $F_1$ should hardly fall with $p$, and in the campaign it does not.

Checks against the campaign / independent study at $n = \infty$: lasso 0.692 here against 0.688 (IS,
same 100 graphs), adaptive lasso 0.774 against 0.763 (IS, pilot = lasso at $\lambda_{\min}$); at $p = 20$
the campaign's `C_ID` gain of MCP dense → sparse is +0.210 (S4 §3), the population adaptive gain here
+0.183. On the raw scale ($C = 2I$ on the covariance) the population gain at $p = 10$ is +0.102.

By density at $p = 10$ the population gain is +0.13 ($k = 1$), +0.16 ($k = 2$), +0.05 ($k = 3$) and
−0.02 ($k = 4$): at $k = 4$ the dense end is saturated (45 of 45 possible entries, Lemma 1) while 37 true
edges, 42 % of them in 2-cycles, compete for them; its recall is 0.67 and the true direction gets the
larger weight for only 58 % of the edges. That is F4.

**Why the gain grows with $p$ (F3).** The dense end can hold at most $p(p-1)/2$ entries (Lemma 1), the
graph has about $k(p-1)$, so the load is $2k/p$: 0.50 at $p = 10$, 0.25 at $p = 20$, 0.16 at $p = 30$,
0.10 at $p = 50$ (averaged over $k$). As in compressed sensing, $\ell_1$ recovery improves as the sparsity
falls relative to the number of equations; the measured recall and direction accuracy of the dense end
follow the load (table). This is a heuristic link (the Lyapunov design is not a random design, so the
Donoho–Tanner phase transition does not apply literally); the numbers are the evidence.

## 6. Why the dense end orients correctly, and the remaining facts

**Proposition 9 (the cost of reversing an edge inside the exact fits; exact).** Let $j \to i$ be a true
edge without its reverse, $K = \Sigma^{-1}$, $s = \operatorname{sign}M^*_{ij}$. The direction $v =
-s(E_{ij} - E_{ji})K$ stays in $\mathcal E(\Sigma)$ and trades $M_{ij}$ for $M_{ji}$. The directional
derivative of $f$ at $M^*$ along $v$ is

$$D_{ij} = (K_{ii} - K_{jj}) \;+\; \sum_{b \ne i,j}\Big(\lvert K_{jb}\rvert\,[(i,b)\notin S] +
\lvert K_{ib}\rvert\,[(j,b)\notin S]\Big) \;-\; \sum_{b \ne i,j}\Big(s\,\mathrm{sgn}(M^*_{ib})K_{jb}\,[(i,b)\in S] -
s\,\mathrm{sgn}(M^*_{jb})K_{ib}\,[(j,b)\in S]\Big).$$

*Proof.* Row $i$ of $(E_{ij} - E_{ji})K$ is $K_{j\cdot}$, row $j$ is $-K_{i\cdot}$; add up the changes of
$\lvert M_{ab}\rvert$ to first order. $\square$ (Checked against finite differences, error $< 3\cdot
10^{-7}$ on 4 700 edges.)

Reading:

- The first term is the only one a single pair "sees": on the correlation scale $K_{ii} = 1/(1 -
  R_i^2)$, how well node $i$ is explained by the others. It favours the truth for only 44 % of the edges at
  $p = 10$ and 45 % at $p = 20$: a coin flip, like the first-order tie of the lock-in note.
- The second term is the **leakage**: reversing the edge forces nonzeros into the other entries of rows
  $i$ and $j$, in proportion to the partial correlations of $i$ and $j$ with their neighbours. Where those
  entries are non-edges, each one costs $\ell_1$ (median total 1.4 at $p = 10$, 2.1 at $p = 20$). The third
  term is the rare case where the leakage lands on a true edge and can cancel (median 0.04 and 0.06).
- So $D_{ij} > 0$ — every exact fit near $M^*$ that starts reversing the edge has a larger $\ell_1$ norm —
  for 96 % of the edges at $p = 10$ and 99 % at $p = 20$. Where $D_{ij} \le 0$ the dense end gets the
  direction right only 34 % of the time; where $D_{ij} > 0$, 74 % ($p = 10$) and 91 % ($p = 20$). The
  remaining errors at $p = 10$ are non-local (several pairs move together) and concentrate at $k = 3, 4$,
  where the dense end is saturated.
- **This is the mechanism.** The leakage term is second-order information carried by the neighbours of
  the pair: the same information the lock-in note shows a sparse → dense path cannot use, because it
  orients a pair before the neighbours are in the fit. The dense end is an exact fit, so all neighbours are
  present and every edge is oriented using them, all at once. The standard path decides early with the
  first term only; the dense end decides with all three.

**F2: adaptive lasso ≥ MCP from the same start.** Theorem 5 and Corollary 6 give both the same
guarantee; the difference is in the weights (§2). The adaptive lasso removes entries in the order of their
size at the dense end (weight $1/\lvert M^B_{ij}\rvert$) and never re-admits an entry the dense end set to
zero; MCP's capped weights penalise a small spurious entry no more than the lasso does, and the
warm-started continuation can re-admit entries. Since the dense end ranks true above false entries well
(AUC 0.76 at $p = 10$, 0.93 at $p = 20$, `docs/DENSE_START.md` §7.3), ordering the whole path by that
ranking is what a good `aupr` needs. This is an explanation, not a proof; nothing above says one must win.

**F5: the rescaled $C$.** All of the above needs $M^* \in \mathcal E(\Sigma)$. With the wrong $C$ the
exact fits are shifted by a dense matrix (`docs/DENSE_START.md` §7.2), the dense end is the
$\ell_1$-optimal point of the wrong set, and Theorem 5 / 8 describe convergence to the wrong target. A
diagonal error is partly absorbed (rescaling removes 95 % / 40 % of the misfit for the random diagonal
settings), a non-diagonal one is not (`C_Random_Full`), which matches the campaign.

**F6: the Varando–Hansen start carries no direction information.** The log-likelihood path in the
campaign starts from $-\tfrac12 C\hat\Sigma^{-1}$, the exact fit with $W = 0$. Its entries are $-\tfrac12
C_{ii}K_{ij}$: **symmetric** in $(i,j)$ up to $C_{ii}/C_{jj}$. On the raw scale with $C = 2I$ it is exactly
symmetric (log-ratio 0 for every edge, population check), on the correlation scale with the rescaled $C$
the ratio is $s_j^2/s_i^2$, the variance ordering: the true direction is the larger entry for 65 % of the
edges, the same 65 % as "the parent has the larger variance" (IS §2). It is the precision matrix, an
undirected object, so the MCP path has to orient every edge itself, and it does so with the lock-in of the
standard path. The minimum-$\ell_1$ dense end puts the larger weight on the true direction for 76 / 92 / 99
% of the edges ($p = 10 / 20 / 30$). Prediction: the log-likelihood MCP path started from the
log-likelihood lasso's dense end (VERDICTS, open item 1) should recover the direct-loss gain.

**F7** is Proposition 9 plus Theorem 8: the dense end commits to one direction per pair and is right most
of the time; pruning keeps that direction.

## 7. What reweighting can and cannot add

Weighted $\ell_1$ with *good* weights recovers far more graphs than the BP condition allows: with oracle
weights $1/(\lvert M^*\rvert + \epsilon)$ the weighted minimum-$\ell_1$ exact fit is $M^*$ in 75 of 100
graphs at $p = 10$ and in 40 of 40 at $p = 20$ (weighted null space property: Mansour & Saab 2017;
Friedlander, Mansour, Saab & Yilmaz 2012). Iterating the reweighting from the dense end (Candès, Wakin &
Boyd 2008, five rounds, $\epsilon$ = a tenth of the median entry) recovers no more than the dense end
itself (14 against 15 of 100; 7 of 40 at $p = 20$), and adding $\epsilon$ to the adaptive lasso's weights
changes nothing (0.773 against 0.774). The reweighting is anchored by the dense end's support: once a true
edge is missing there, a small $\epsilon$ does not bring it back. **The pilot is the lever.** A better
dense start — different weights in the LP, an exact fit chosen by the likelihood, or a search that may add
edges (which is what the greedy search does, and why it lifts the lasso more than the adaptive lasso) — is
where further gains would come from; Theorem 5 / 8 would carry over to any pilot that converges.

## 8. What is still missing

- **Rates in $p$, and $p > n$.** Everything above is fixed $p$. The campaign's largest gains are at $p =
  40, 50$ with $n = 10^3$. A non-asymptotic version needs the sharpness constant $\kappa$ of the LP and the
  smallest eigenvalue of $\Gamma_{T_BT_B}$ as functions of $p$; both are computable per graph but have no
  bounds yet. The restricted-eigenvalue route of the high-dimensional adaptive-lasso literature (Huang, Ma
  & Zhang 2008; van de Geer, Bühlmann & Zhou 2011) is closed (7 October note, §4).
- **A global statement about the dense end's directions.** Proposition 9 is local (first order along one
  reversal at a time). That the dense end orients most edges correctly when the load $2k/p$ is small is
  numerical.
- **Finite $n$.** The population paths explain $n = \infty$; at $n = 10^3$ the campaign's gains are
  smaller at $p = 10$ and still large at $p \ge 20$. The sampling error of the dense end enters through
  Proposition 4 only asymptotically.
- **The comparison adaptive lasso vs MCP** (F2) is explained, not proved.

## 9. Scripts and references

Scripts (session sandbox; numpy, scipy, scikit-learn; they reuse `bp_condition_check.py` of 7 October):

- `dense_start_theory_check.py P REPS KIND [k-list] [out.jsonl]`: dense end, recall, direction weights,
  population lasso and adaptive-lasso paths ($p \le 20$), reweighting, oracle weights, Varando–Hansen
  start. `ds_summ.py` prints the tables.
- `reversal_derivative.py P REPS KIND`: Proposition 9 with the finite-difference check.
- `lemma1_columns_of_A.xlsx`: the two-node example of §2a and the proof of Lemma 1, with $r$ as an input.

| reference | used for | status |
|---|---|---|
| Zou (2006), *The adaptive lasso and its oracle properties*, JASA 101 | Theorem 5: the argument (epi-convergence via Geyer, KKT for selection); assumes a positive definite limit Gram, which we replace by $\Gamma_{\bar S\bar S} \succ 0$ + infinite penalty | assumptions confirmed from lecture slides; check the paper |
| Knight & Fu (2000), Ann. Statist.; Geyer (1994), Ann. Statist. | convexity lemma for argmin convergence | from memory |
| Fan, Xue & Zou (2014), Ann. Statist. 42 | Corollary 6 (LLA, Theorems 1–2) | literature note 7 Oct (T) |
| Zou & Li (2008), Ann. Statist. 36 | LLA; adaptive lasso as one LLA step | from memory |
| Fuchs (2004), IEEE Trans. Inf. Theory 50, *On sparse representations in arbitrary redundant bases* | Proposition 7, first arrow (dual certificate) | standard; statement from memory |
| Burke & Ferris (1993), *Weak sharp minima in mathematical programming*, SIAM J. Control Optim. 31 | Proposition 4 (sharp minimum of an LP with a unique solution) | existence confirmed |
| Mansour & Saab (2017), *Recovery analysis for weighted ℓ1-minimization using the null space property*, ACHA (arXiv:1412.1565); Friedlander, Mansour, Saab & Yilmaz (2012), IEEE TIT | §7, weighted null space property | abstract confirmed; journal from memory |
| Candès, Wakin & Boyd (2008), *Enhancing sparsity by reweighted ℓ1 minimization*, J. Fourier Anal. Appl. | §7 | from memory |
| Dettling, Drton & Kolar (2024) | Prop. G.2, weak irrepresentability necessary for the lasso; Appendix C (display of $A(\Sigma)$, used in §2a) | literature note (T); Appendix C from the paper |
| Dettling, Homs, Améndola, Drton & Hansen (2023) | exact fits $(W - C/2)\Sigma^{-1}$, Lemma 27 | literature note (T) |
| Huang, Ma & Zhang (2008); van de Geer, Bühlmann & Zhou (2011) | high-dimensional adaptive lasso; need RE-type conditions, fail here | literature note |
| Tibshirani (2013), *The lasso problem and uniqueness*, EJS 7 | existence of a lasso solution with independent active columns (Lemma 3 remark) | from memory |
