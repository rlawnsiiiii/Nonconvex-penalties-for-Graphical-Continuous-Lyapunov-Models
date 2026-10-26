# Support recovery for the adaptive (weighted) lasso in GCLMs — Theorem 2 / Theorem D.1 of DDK with weights, proved by the primal–dual witness (21 October 2026, revised)

*Revision of the same day, after the reading session. Changes against the first version:*

1. *The diagonal is now penalised, as in DDK's estimator (1.4).*
2. *The weights are the implemented ones (`gclm.solvers.path.adaptive_lasso_path`, `penalize_diagonal=True`):*
   $\hat w_{jk}=\max_{(k',l')}|\hat M^0_{k'l'}|\,/\,|\hat M^0_{jk}|$, *with the maximum over all entries, so every weight is $\ge1$.*
3. *The constants are rearranged so that the theorem reduces exactly to DDK's Theorem D.1 when all weights are $1$; that reduction is checked in §2.*
4. *The error bound is written in DDK's form.*

*Notation, constants and the order of steps follow Dettling, Drton & Kolar (2024) = DDK: Theorem 2 (p. 6), Theorem D.1 (App. D, pp. 530–533), Lemmas E.1–E.6 and App. F (pp. 533–537). Each line of DDK's proof that changes is marked ▶. The conclusion is support recovery: $\hat S\subseteq S$, and $\hat S=S$ under beta-min. Signs come out of the same argument.*

---

## 0. Setting (DDK §2–3)

**Data and model.** $X_1,\dots,X_n\in\mathbb R^p$ are i.i.d. $N(0,\Sigma^*)$ with $M^*\Sigma^*+\Sigma^*M^{*\top}+C=0$, where $M^*\in\mathrm{Stab}_p$ and $C\in PD_p$ is known. The sample covariance is $\hat\Sigma=\frac1n\sum_iX_iX_i^\top$.

**Design and Gram matrix.** For symmetric $\Sigma$:
$$A(\Sigma)=(\Sigma\otimes I_p)+(I_p\otimes\Sigma)K^{(p,p)},\qquad \Gamma(\Sigma)=A(\Sigma)^\top A(\Sigma),\qquad g(\Sigma)=-A(\Sigma)^\top\mathrm{vec}(C),$$
so that
$$\tfrac12\|M\Sigma+\Sigma M^\top+C\|_F^2=\tfrac12\mathrm{vec}(M)^\top\Gamma(\Sigma)\mathrm{vec}(M)-g(\Sigma)^\top\mathrm{vec}(M)+\tfrac12\|C\|_F^2 .$$
DDK write $g=-A(\Sigma)\mathrm{vec}(C)$. The two definitions agree when $C\propto I_p$; with the transpose the identity holds for every symmetric $C$, which the rescaled $C$ of the thesis needs.

**Shorthand.** $\hat\Gamma=\Gamma(\hat\Sigma)$, $\Gamma^*=\Gamma(\Sigma^*)$, $\hat g$, $g^*$, $\Delta_\Gamma=\hat\Gamma-\Gamma^*$, $\Delta_g=\hat g-g^*$. Indices are pairs $(j,k)$; $\mathrm{vec}$ stacks columns; $m=\mathrm{vec}(M)$, $m^*=\mathrm{vec}(M^*)$.

**Support.** $S=\{(j,k):M^*_{jk}\ne0\}$, which contains the diagonal; $d=|S|$; $S^c$ is its complement. Since $M^*$ solves the Lyapunov equation, $\Gamma^*m^*=g^*$ (DDK p. 531).

**Constants (DDK).**
$$c_{\Gamma^*}=|||(\Gamma^*_{SS})^{-1}|||_\infty,\qquad c_{M^*}=\|m^*\|_\infty,\qquad P:=\Gamma^*_{S^cS}(\Gamma^*_{SS})^{-1},\qquad \rho:=|||P|||_\infty\ \ (\text{DDK (4.1); no assumption on its size}).$$
Here $|||\cdot|||_\infty$ is the maximum absolute row sum and $\|\cdot\|_\infty$ the entrywise maximum. Write $\beta_{\min}:=\min_{(j,k)\in S}|M^*_{jk}|$.

## 1. Estimator, weights, and the weighted irrepresentability constant

**Pilot and weights (as implemented).**
- The pilot $\hat M^0$ is the lasso (1.4) with the diagonal penalised, at the smallest $\lambda$ of its grid: the dense end of the path.
- With $\hat\mu:=\max_{(k,l)}|\hat M^0_{kl}|$, the maximum over all entries including the diagonal,
$$\hat w_{jk}=\frac{\hat\mu}{|\hat M^0_{jk}|}\quad\text{if }\hat M^0_{jk}\ne0,\qquad \hat w_{jk}=\infty\ \ (\text{entry excluded, stays }0)\quad\text{if }\hat M^0_{jk}=0 .$$
- All finite weights are $\ge1$, with equality at the largest pilot entry.

The theorem below holds for **arbitrary** weights in $(0,\infty]$, possibly depending on the same data. It is deterministic, so no sample splitting is needed.

**Estimator** (weighted direct Lyapunov lasso, all entries penalised):
$$\hat M\in\arg\min_{M\in\mathbb R^{p\times p}}\ \tfrac12\|M\hat\Sigma+\hat\Sigma M^\top+C\|_F^2+\lambda\sum_{j,k}\hat w_{jk}|M_{jk}|
=\arg\min_m\ \tfrac12m^\top\hat\Gamma m-\hat g^\top m+\lambda\|\hat Wm\|_1,\qquad \hat W=\mathrm{diag}(\hat w). \tag{1.1}$$
An infinite weight means the constraint $m_j=0$.

**Scale invariance.** Replacing $(\lambda,\hat w)$ by $(\lambda/c,\,c\hat w)$ leaves (1.1) unchanged. Every hypothesis and conclusion below depends on the weights only through ratios of weights or through the products $\lambda\hat w_j$, so the normalisation by $\hat\mu$ only fixes the scale of $\lambda$ (as the docstring of `adaptive_lasso_path` notes).

**Quantities.** Let $E:=\{j:\hat w_j=\infty\}$ be the excluded entries and $S^c_+:=S^c\setminus E$. Define
$$\underline w:=\min_{j\in S^c_+}\hat w_j,\qquad \bar w_S:=\max_{k\in S}\hat w_k\lt \infty,\qquad
r:=\frac{\bar w_S}{\underline w}=\frac{\max_{j\in S^c_+}|\hat M^0_j|}{\min_{k\in S}|\hat M^0_k|}.$$
- $\bar w_S\lt \infty$ requires the pilot to have no zero on $S$, a necessary condition for recovery.
- $r$, the **separation ratio of the pilot**, is the largest pilot entry off the support divided by the smallest pilot entry on the support. It does not depend on the normalisation.
- DDK's lasso is $\hat w\equiv1$, which gives $\underline w=\bar w_S=r=1$.
- If $S^c_+=\emptyset$ (the pilot's support is contained in $S$), set $\underline w=\infty$ and $r=0$.

**Weighted irrepresentability constant** (Zhou, van de Geer & Bühlmann 2009, Def. 8.1, written as a constant in DDK's style):
$$\rho_w:=\max_{j\in S^c_+}\ \frac{1}{\hat w_j}\sum_{k\in S}|P_{jk}|\,\hat w_k\ \ \Big(=|||\hat W_{S^c_+}^{-1}P_{S^c_+\cdot}\hat W_{SS}|||_\infty\Big),\qquad \rho_w\le\rho\,r . \tag{1.2}$$
The bound $\rho_w\le\rho r$ follows by bounding every $\hat w_k$ by $\bar w_S$ and every $\hat w_j$ by $\underline w$. For $\hat w\equiv1$, $\rho_w=\rho$.

**A free constant.** Fix any $\kappa\ge1+\rho$. The natural choice is $\kappa=1+\rho$. In DDK's setting, $\rho\le1-\alpha$ and $\kappa=2-\alpha$ is admissible, and this is the choice that reproduces D.1.

## 2. Theorem W.1 (deterministic; the analogue of DDK Theorem D.1)

**Theorem W.1.** Let $M^*\in\mathrm{Stab}_p$ have support $S$, let $\hat w$ be weights with $\bar w_S\lt \infty$, and let $\kappa\ge1+\rho$. Assume:

* (W1) $\Gamma^*_{SS}$ is invertible;
* (W2) (weighted irrepresentability) $\rho_w\le1-\alpha$ for some $\alpha\in(0,1]$;
* (W3) $|||(\Delta_\Gamma)_{\cdot S}|||_\infty\le\varepsilon_1$ and $\|\Delta_g\|_\infty\le\varepsilon_2$, with
$$\varepsilon_1\ \le\ \frac{\alpha}{3c_{\Gamma^*}\,(\kappa r+\alpha)} ;$$
* (W4) the effective penalty off the support satisfies
$$\lambda\,\underline w\ >\ \frac{3\kappa}{\alpha}\,\max\{c_{M^*}\varepsilon_1,\ \varepsilon_2\}.$$

Then:

**(a)** The minimiser $\hat M$ of (1.1) is unique, its support is contained in the true support ($\hat S\subseteq S$), and
$$\|\hat M-M^*\|_\infty\ \le\ b_w:=\frac{c_{\Gamma^*}}{1-c_{\Gamma^*}\varepsilon_1}\big(c_{M^*}\varepsilon_1+\varepsilon_2+\lambda\bar w_S\big)\ \lt \ c_{\Gamma^*}\,\lambda\Big(\bar w_S+\frac{\alpha}{\kappa}\,\underline w\Big). \tag{2.1}$$
The last bound is informative only when $\underline w\lt \infty$.

**(b)** If moreover $\beta_{\min}=\min_{(j,k)\in S}|M^*_{jk}|>b_w$, then $\hat S=S$ and $\operatorname{sign}(\hat M_{jk})=\operatorname{sign}(M^*_{jk})$ for all $(j,k)\in S$.

**Reduction to DDK Theorem D.1.** Take $\hat w\equiv1$ and $\kappa=2-\alpha$; this needs $\rho\le1-\alpha$, which is D.1's assumption. Then:
- (W2) is DDK's (D.1), $\rho\le1-\alpha$;
- (W3) is DDK's $\varepsilon_1\le\alpha/(6c_{\Gamma^*})$, since $3c_{\Gamma^*}(\kappa\cdot1+\alpha)=6c_{\Gamma^*}$;
- (W4) is DDK's $\lambda>\frac{3(2-\alpha)}{\alpha}\max\{c_{M^*}\varepsilon_1,\varepsilon_2\}$;
- the right-hand bound in (2.1) is $c_{\Gamma^*}\lambda\big(1+\frac{\alpha}{2-\alpha}\big)=\frac{2c_{\Gamma^*}\lambda}{2-\alpha}$, which is DDK's bound;
- DDK's beta-min $\beta_{\min}>2c_{\Gamma^*}\lambda/(2-\alpha)$ implies the one in (b).

So D.1 is the special case of W.1 with unit weights.

**What the weights change.** Three things:
- the irrepresentability condition is on $\rho_w$ instead of $\rho$; $\rho$ survives only inside $\kappa$, as the factor in front of the noise;
- the noise is compared with the smallest effective penalty off the support, $\lambda\underline w$;
- the bias on the support is $\lambda\bar w_S$. Off-support weights strengthen the penalty where it is needed, support weights weaken it where it hurts, and $r$ measures how well the pilot separates the two.

**Implemented weights.** Since $\hat w\ge1$, we have $\lambda\underline w\ge\lambda$, so (W4) already holds when $\lambda$ satisfies DDK's own $\lambda$-condition with $2-\alpha$ replaced by $\kappa$.

## 3. Proof of Theorem W.1 (PDW, following DDK pp. 531–533)

**KKT conditions (replaces (D.2)).** Problem (1.1) is convex, since $\hat\Gamma\succeq0$, the weighted $\ell_1$ norm is convex, and the feasible set $\{m_E=0\}$ is a subspace. So $\hat m$ is a minimiser if and only if $\hat m_E=0$ and there is $\hat z$ with
$$\big(\hat\Gamma\hat m-\hat g\big)_j+\lambda\,\hat w_j\hat z_j=0\ \ (j\notin E),\qquad
\hat z_j=\operatorname{sign}(\hat m_j)\ \text{if }\hat m_j\ne0,\quad \hat z_j\in[-1,1]\ \text{if }\hat m_j=0 . \tag{3.1}$$
Coordinates in $E$ are fixed at $0$ and carry no condition.

▶ The only change from (D.2) is the factor $\hat w_j$ in front of $\hat z_j$. In words, at a zero coordinate the slope of the loss must not exceed the kink $\lambda\hat w_j$ of that coordinate's penalty. The PDW constructs a pair $(\tilde m,\tilde z)$ satisfying (3.1) with $\tilde m_{S^c}=0$.

**Step 1 (restricted problem; replaces (D.3)–(D.4)).** Let
$$\tilde m=\arg\min_{m:\ m_{S^c}=0}\ \tfrac12m^\top\hat\Gamma m-\hat g^\top m+\lambda\|\hat Wm\|_1 . \tag{3.2}$$

*Invertibility (as on p. 532).* By (W3), $c_{\Gamma^*}\varepsilon_1\le\alpha/(3(\kappa r+\alpha))\le\frac13$. Write $\hat\Gamma_{SS}=\Gamma^*_{SS}(I+X)$ with $X=(\Gamma^*_{SS})^{-1}(\Delta_\Gamma)_{SS}$ and $|||X|||_\infty\le c_{\Gamma^*}\varepsilon_1\lt 1$. The Neumann series converges, so $\hat\Gamma_{SS}$ is invertible and
$$|||(\hat\Gamma_{SS})^{-1}|||_\infty\le\frac{c_{\Gamma^*}}{1-c_{\Gamma^*}\varepsilon_1}. \tag{3.3}$$
The restricted objective is therefore strictly convex and $\tilde m$ is unique. Its optimality conditions (the $S$-rows of (3.1)) give $\tilde z_S$, with $\tilde z_k=\operatorname{sign}(\tilde m_k)$ if $\tilde m_k\ne0$ and $\tilde z_k\in[-1,1]$ otherwise, such that
$$\hat\Gamma_{SS}\tilde m_S-\hat g_S+\lambda(\hat w\circ\tilde z)_S=0. \tag{3.4}$$
▶ In (D.3) the last term is $\lambda\operatorname{sign}(\cdot)$; here every component of $(\hat w\circ\tilde z)_S$ is bounded by $\bar w_S$.

*Error bound.* Since $m^*_{S^c}=0$ and $\Gamma^*_{SS}m^*_S=g^*_S$, we have $\hat\Gamma_{SS}m^*_S-\hat g_S=(\Delta_\Gamma)_{SS}m^*_S-(\Delta_g)_S$. Subtracting this from (3.4) gives
$$\tilde\Delta_S:=\tilde m_S-m^*_S=-(\hat\Gamma_{SS})^{-1}\Big((\Delta_\Gamma)_{SS}m^*_S-(\Delta_g)_S+\lambda(\hat w\circ\tilde z)_S\Big).$$
Use $\|(\Delta_\Gamma)_{SS}m^*_S\|_\infty\le|||(\Delta_\Gamma)_{\cdot S}|||_\infty\,c_{M^*}\le c_{M^*}\varepsilon_1$ and (3.3):
$$\|\tilde\Delta_S\|_\infty\ \le\ \frac{c_{\Gamma^*}}{1-c_{\Gamma^*}\varepsilon_1}\big(c_{M^*}\varepsilon_1+\varepsilon_2+\lambda\bar w_S\big)=b_w. \tag{3.5}$$

*Converting the noise into $\lambda$ (as DDK do).* By (W4), $c_{M^*}\varepsilon_1+\varepsilon_2\le2\max\{c_{M^*}\varepsilon_1,\varepsilon_2\}\lt \frac{2\alpha}{3\kappa}\lambda\underline w$. By (W3), $\frac{1}{1-c_{\Gamma^*}\varepsilon_1}\le\frac{3(\kappa r+\alpha)}{3\kappa r+2\alpha}$. Hence
$$b_w\lt c_{\Gamma^*}\cdot\frac{3(\kappa r+\alpha)}{3\kappa r+2\alpha}\cdot\lambda\underline w\Big(r+\frac{2\alpha}{3\kappa}\Big)
=c_{\Gamma^*}\lambda\underline w\,\frac{\kappa r+\alpha}{\kappa}=c_{\Gamma^*}\lambda\Big(\bar w_S+\frac{\alpha}{\kappa}\underline w\Big). \tag{3.6}$$
▶ This is (D.4). With $\hat w\equiv1$ and $\kappa=2-\alpha$ the right-hand side is DDK's $\frac{6c_{\Gamma^*}}{6-\alpha}\cdot\frac{6-\alpha}{3(2-\alpha)}\lambda=\frac{2c_{\Gamma^*}\lambda}{2-\alpha}$.

**Step 2 (subgradient on $S$).** Take $\tilde z_S$ from (3.4). By construction it is a subgradient of $\|\cdot\|_1$ at $\tilde m_S$.

**Step 3 (dual vector on $S^c_+$; replaces (D.5)).** For $j\in S^c_+$ define $\tilde z_j$ by the $j$-th row of (3.1) with $\tilde m_{S^c}=0$:
$$\lambda\hat w_j\tilde z_j:=\hat g_j-\big[\hat\Gamma_{S^cS}\tilde m_S\big]_j .$$
Insert $\hat g_{S^c}=\Gamma^*_{S^cS}m^*_S+(\Delta_g)_{S^c}$ and $\hat\Gamma_{S^cS}=\Gamma^*_{S^cS}+(\Delta_\Gamma)_{S^cS}$. Then use the $S$-rows (3.4) in their population form,
$$m^*_S-\tilde m_S=(\Gamma^*_{SS})^{-1}\big((\Delta_\Gamma)_{SS}\tilde m_S-(\Delta_g)_S+\lambda(\hat w\circ\tilde z)_S\big).$$
This gives, with $P=\Gamma^*_{S^cS}(\Gamma^*_{SS})^{-1}$,
$$\tilde z_j=\underbrace{\frac{1}{\lambda\hat w_j}\Big(\big[P\big((\Delta_\Gamma)_{SS}\tilde m_S-(\Delta_g)_S\big)\big]_j-\big[(\Delta_\Gamma)_{S^cS}\tilde m_S\big]_j+(\Delta_g)_j\Big)}_{\text{noise}}
\;+\;\underbrace{\frac{1}{\hat w_j}\big[P(\hat w\circ\tilde z)_S\big]_j}_{\text{deterministic}} . \tag{3.7}$$
▶ This is (D.5) with $\operatorname{sign}(\mathrm{vec}(\tilde M)_S)$ replaced by $(\hat w\circ\tilde z)_S$ and row $j$ divided by $\hat w_j$. The signs of the noise terms differ from DDK's display; only absolute values enter below.

*Deterministic term.* Since $|\tilde z_k|\le1$,
$$\Big|\frac{1}{\hat w_j}\big[P(\hat w\circ\tilde z)_S\big]_j\Big|\ \le\ \frac{1}{\hat w_j}\sum_{k\in S}|P_{jk}|\hat w_k\ \le\ \rho_w\ \le\ 1-\alpha\qquad\text{by (W2)}. \tag{3.8}$$
▶ DDK bound this term by $\rho\le1-\alpha$. This is the only place where the weighted condition enters.

*Noise term.*
- $|[Pv]_j|\le\rho\|v\|_\infty$.
- The rows of $(\Delta_\Gamma)_{SS}$ and of $(\Delta_\Gamma)_{S^cS}$ are rows of $(\Delta_\Gamma)_{\cdot S}$, so their row sums are at most $\varepsilon_1$.
- $\|\Delta_g\|_\infty\le\varepsilon_2$, $\|\tilde m_S\|_\infty\le c_{M^*}+\|\tilde\Delta_S\|_\infty$, $1+\rho\le\kappa$, and $\hat w_j\ge\underline w$.

Together,
$$|\text{noise}_j|\ \le\ \frac{\kappa}{\lambda\underline w}\big(\varepsilon_1\|\tilde m_S\|_\infty+\varepsilon_2\big)\ \le\
\underbrace{\frac{\kappa\,c_{M^*}\varepsilon_1}{\lambda\underline w}}_{G_1}+\underbrace{\frac{\kappa\,\varepsilon_1\|\tilde\Delta_S\|_\infty}{\lambda\underline w}}_{G_2}+\underbrace{\frac{\kappa\,\varepsilon_2}{\lambda\underline w}}_{G_3}. \tag{3.9}$$
▶ These are DDK's $G_1,G_2,G_3$ (pp. 532–533), with $(2-\alpha)/\lambda$ replaced by $\kappa/(\lambda\underline w)$. Each is below $\alpha/3$:
- $G_1,G_3\lt \alpha/3$ by (W4);
- by (3.5)–(3.6), $G_2\lt \frac{\kappa\varepsilon_1}{\lambda\underline w}\,c_{\Gamma^*}\lambda\underline w\,\frac{\kappa r+\alpha}{\kappa}=c_{\Gamma^*}\varepsilon_1(\kappa r+\alpha)\le\alpha/3$ by (W3).

With (3.8),
$$\|\tilde z_{S^c_+}\|_\infty\lt (1-\alpha)+\tfrac\alpha3+\tfrac\alpha3+\tfrac\alpha3=1 . \tag{3.10}$$
If $S^c_+=\emptyset$ there is nothing to check.

**Step 4 (the witness is the unique minimiser; $\hat S\subseteq S$).** The pair $(\tilde m,\tilde z)$ satisfies (3.1):
- the $S$-rows by (3.4) and the $S^c_+$-rows by construction;
- $\tilde m_E=0$;
- $\tilde z$ is a valid subgradient, since $|\tilde z_j|\lt 1$ at the zero coordinates in $S^c_+$ by (3.10), and $\tilde z_S$ comes from Step 2.

So $\tilde m$ is a minimiser.

*Every minimiser has support in $S$* (Wainwright 2009, Lemma 1, cited by DDK on p. 532). Let $\hat m$ be any minimiser and $L(m)=\tfrac12m^\top\hat\Gamma m-\hat g^\top m$. On the feasible set, $\nabla_jL(\tilde m)=-\lambda\hat w_j\tilde z_j$ for $j\notin E$, and $\hat m_E=\tilde m_E=0$. Convexity of $L$ and $\langle\hat W\tilde z,\tilde m\rangle=\|\hat W\tilde m\|_1$ give
$$L(\hat m)\ge L(\tilde m)+\lambda\|\hat W\tilde m\|_1-\lambda\langle\hat W\tilde z,\hat m\rangle .$$
Both points minimise $L+\lambda\|\hat W\cdot\|_1$, so subtracting yields $\|\hat W\hat m\|_1\le\langle\hat W\tilde z,\hat m\rangle$. The right-hand side is at most $\sum_{k\in S}\hat w_k|\hat m_k|+\|\tilde z_{S^c_+}\|_\infty\sum_{j\in S^c_+}\hat w_j|\hat m_j|$, hence
$$\big(1-\|\tilde z_{S^c_+}\|_\infty\big)\sum_{j\in S^c_+}\hat w_j|\hat m_j|\le0 .$$
By (3.10) and $\hat w_j>0$, $\hat m_{S^c}=0$.

*Uniqueness.* Every minimiser lives on $\{m_{S^c}=0\}$, where the objective is strictly convex by (3.3). So the minimiser is unique and equals $\tilde m$. Part (a) follows from (3.5)–(3.6). $\square_{(a)}$

**Part (b).** For $(j,k)\in S$, (a) gives $|\hat M_{jk}-M^*_{jk}|\le b_w\lt \beta_{\min}\le|M^*_{jk}|$. Hence $\hat M_{jk}\ne0$ with the sign of $M^*_{jk}$, so $S\subseteq\hat S$, and with (a), $\hat S=S$. This is the last display of DDK p. 533, which also covers the diagonal entries, now that they are penalised. $\square$

## 4. Theorem W.2 (probabilistic; the analogue of DDK Theorem 2)

**DDK's tools, used unchanged.**
- Lemma E.6: $|||\Delta_\Sigma|||_2\lt \min\{\varepsilon_1/(\sqrt d(4+8c_{\Sigma^*})),\ \varepsilon_2/(2c_C)\}$ implies $|||(\Delta_\Gamma)_{\cdot S}|||_\infty\lt \varepsilon_1$ and $\|\Delta_g\|_\infty\lt \varepsilon_2$, with $c_{\Sigma^*}=|||\Sigma^*|||_2$ and $c_C=\|\mathrm{vec}(C)\|_2$.
- Corollary E.5 (from Wainwright 2019, Thm 6.5) bounds $|||\Delta_\Sigma|||_2$.
- Neither involves the penalty. As in DDK,
$$\tilde c=\max\Big\{\frac{4\max\{1,c_{\Sigma^*}^2\}(4+8c_{\Sigma^*})^2}{c_3},\ 16c_1^2c_{\Sigma^*}^2(4+8c_{\Sigma^*})^2,\ \frac{16\max\{1,c_{\Sigma^*}^2\}c_C^2}{c_3},\ 64c_1^2c_{\Sigma^*}^2c_C^2\Big\},$$
with the universal constants $c_1,c_2,c_3$ of Theorem E.4.

**Theorem W.2.** In the setting of DDK Theorem 2, let $\kappa\ge1+\rho$, $\alpha\in(0,1]$ and $\bar r\ge0$. Let $\mathcal E_w$ be an event on which the weights satisfy $\bar w_S\lt \infty$, $\rho_w\le1-\alpha$ and $r\le\bar r$. Put
$$\varepsilon:=\sqrt{\tau_1\tilde c\,d\,p/n},\qquad c_w:=\frac{3c_{\Gamma^*}(\kappa\bar r+\alpha)}{\alpha}\qquad\Big(\text{DDK: }c_*=\frac{6c_{\Gamma^*}}{\alpha}\Big).$$
Suppose $n>\tau_1\tilde c\,d\,p\max\{c_w^2,1/4\}$ for some $\tau_1>1$, and that on $\mathcal E_w$
$$\lambda\,\underline w\ >\ \frac{3\kappa}{\alpha}\max\{c_{M^*},1\}\,\varepsilon .$$
For the implemented weights ($\hat w\ge1$) it suffices that $\lambda>\frac{3\kappa}{\alpha}\max\{c_{M^*},1\}\varepsilon$; this is DDK's condition with $2-\alpha$ replaced by $\kappa$. Then, with probability at least $1-c_2\exp(-\tau_1p)-P(\mathcal E_w^c)$:

(a) the minimiser of (1.1) is unique, $\hat S\subseteq S$, and $\|\hat M-M^*\|_\infty\le\frac{c_{\Gamma^*}}{1-c_{\Gamma^*}\varepsilon}\big((c_{M^*}+1)\varepsilon+\lambda\bar w_S\big)\lt c_{\Gamma^*}\lambda(\bar w_S+\frac\alpha\kappa\underline w)$;

(b) if $\beta_{\min}$ exceeds the first bound in (a), then $\hat S=S$ with the signs of $M^*$.

*Proof.* DDK Appendix F, steps 1–3, with $\varepsilon_1=\varepsilon_2=\varepsilon$.
- **Steps 1–2 are unchanged.** $\varepsilon/(\sqrt d(4+8c_{\Sigma^*}))$ and $\varepsilon/(2c_C)$ lie in the admissible range of Corollary E.5. The upper end uses only $\max\{c_w^2,1/4\}\ge1/4$, the lower end only $\tilde c$. So with probability at least $1-c_2\exp(-\tau_1p)$, $|||(\Delta_\Gamma)_{\cdot S}|||_\infty\lt \varepsilon$ and $\|\Delta_g\|_\infty\lt \varepsilon$. Both are bounds on the same random variable $|||\Delta_\Sigma|||_2$, so no union bound is needed.
- **Step 3 replaces $c_*$ by $c_w$.** $\varepsilon\lt 1/\max\{c_w,1/2\}\le1/c_w=\alpha/(3c_{\Gamma^*}(\kappa\bar r+\alpha))\le\alpha/(3c_{\Gamma^*}(\kappa r+\alpha))$ on $\mathcal E_w$, which is (W3).
- **Conclude.** The $\lambda$-condition is (W4), since $\max\{c_{M^*}\varepsilon,\varepsilon\}=\max\{c_{M^*},1\}\varepsilon$. Apply Theorem W.1 on $\mathcal E_w\cap\{\text{concentration}\}$. $\square$

*Reading.* With $\hat w\equiv1$, $\kappa=2-\alpha$ and $\bar r=1$, Theorem W.2 is DDK's Theorem 2. With a good pilot, $\bar r\to0$ and one may take $\alpha\to1$, so $c_w\to3c_{\Gamma^*}$. The sample size then no longer depends on an irrepresentability gap: it becomes $n\gtrsim\tilde c\,dp\cdot9c_{\Gamma^*}^2$, against DDK's $\tilde c\,dp\cdot36c_{\Gamma^*}^2/\alpha^2$ with $\alpha$ the (often non-existent) lasso gap. The order $\Omega(dp)$ in $(d,p)$ is unchanged; it comes from the spectral-norm route of Appendix E, not from the penalty.

## 5. Corollaries, necessity, and the dense-end pilot

**Corollary W.3 (a pilot close to the truth).** Let $\|\hat M^0-M^*\|_\infty\le\delta\lt \beta_{\min}$. Here $\beta_{\min}$ includes the diagonal entries, which are typically the largest, so in practice it is the smallest true edge.
- On $S^c_+$: $|\hat M^0_j|\le\delta$. On $S$: $|\hat M^0_k|\ge\beta_{\min}-\delta>0$, so $\bar w_S\lt \infty$.
- Hence $r\le\delta/(\beta_{\min}-\delta)$ and
$$\rho_w\le\rho\,r\le\rho\,\frac{\delta}{\beta_{\min}-\delta}\le1-\alpha\qquad\text{as soon as}\qquad \delta\le\frac{(1-\alpha)\,\beta_{\min}}{\rho+1-\alpha}.$$
- With the implemented normalisation, $c_{M^*}-\delta\le\hat\mu\le c_{M^*}+\delta$, $\underline w\ge\hat\mu/\delta$ and $\bar w_S\le\hat\mu/(\beta_{\min}-\delta)$.

So Theorems W.1–W.2 hold for every pilot within $\beta_{\min}/(1+\rho)$ of the truth in sup norm, **whatever the size of DDK's constant $\rho$**. In W.2 take $\mathcal E_w=\{\|\hat M^0-M^*\|_\infty\le\delta\}$ and $\bar r=\delta/(\beta_{\min}-\delta)$.

This is ZvdGB's (8.5) / Theorem 2.1 in deterministic form. The LLA weights of Fan, Xue & Zou (2014), $\hat w_j=P'_\lambda(|\hat M^0_j|)/\lambda$, have $\bar w_S=0$ once the pilot is within $a_0\lambda$; then $\rho_w=0$ and only (W4) remains, which is their Theorem 1.

**Corollary W.4 (sign-weighted version).** Assume (W1), (W3), (W4) and $\beta_{\min}>b_w$. Replace (W2) by the weaker condition
$$\rho_w^{\rm sgn}:=\max_{j\in S^c_+}\frac{1}{\hat w_j}\Big|\sum_{k\in S}P_{jk}\,\hat w_k\operatorname{sign}(M^*_k)\Big|\le1-\alpha ,$$
where the sum includes the diagonal, with $\operatorname{sign}(M^*_{ii})=-1$ in the thesis DGP. Then the conclusions of W.1 hold.

*Proof.* (3.5) does not use (W2). With $\beta_{\min}>b_w$ it gives $\operatorname{sign}(\tilde m_S)=\operatorname{sign}(m^*_S)$, hence $\tilde z_S=\operatorname{sign}(m^*_S)$. The deterministic term of (3.7) then equals the expression inside $\rho_w^{\rm sgn}$; the rest of the proof is unchanged.

This is the weighted form of DDK's weak condition (G.1). It is the quantity to compute per dataset; `gclmth.adaptive_conditions` returns it for an unpenalised diagonal and needs the diagonal signs added for this version.

**Proposition W.5 (necessity; the analogue of DDK Prop. G.2).** Let $\lambda>0$ be fixed and let the weights converge in probability, $\hat w\to w$, with $0\lt w_j\lt \infty$ on $S^c$ and $w\lt \infty$ on $S$. Assume $\beta_{\min}>c_{\Gamma^*}\lambda\max_{k\in S}w_k$ and
$$\max_{j\in S^c}\frac{1}{w_j}\Big|\sum_{k\in S}P_{jk}\,w_k\operatorname{sign}(M^*_k)\Big|>1 .$$
Then $P(\hat S=S)\to0$.

*Proof.* As in G.2. On $\{\hat S=S\}$, $\hat M$ is the restricted solution (3.2), and by (3.5) and $\Delta_\Gamma,\Delta_g\to0$ in probability (Lemma E.6, Corollary E.5) it has the signs of $M^*$ with probability tending to one. Its dual vector is (3.7). The noise part tends to $0$, and the deterministic part tends to the displayed quantity, which exceeds $1$ in some row. That violates (3.1), so $P(\hat S=S)\to0$. $\square$

**Remark W.6 (the dense-end pilot when the diagonal is penalised).**
- As $\lambda\to0$, and in the population, the pilot tends to the minimum-$\|M\|_1$ exact solution (all entries) of $M\Sigma^*+\Sigma^*M^\top+C=0$ (Tibshirani 2013). Penalising the diagonal changes this basis-pursuit problem, but only in one direction.
- *Trace identity.* Every exact solution has $\mathrm{tr}(M)=-\tfrac12\mathrm{tr}(C\Sigma^{*-1})$. To see this, multiply the equation by $\Sigma^{*-1}$ from the right and take traces, using $\mathrm{tr}(\Sigma^*M^\top\Sigma^{*-1})=\mathrm{tr}(M)$. Hence $\sum_i|M_{ii}|\ge\tfrac12\mathrm{tr}(C\Sigma^{*-1})$, with equality if and only if all $M_{ii}\le0$.
- *Consequence.* Suppose $M^*$ has a non-positive diagonal (every DGP of the thesis does) and is the unique minimiser of $\|M_{\rm off}\|_1$ over exact solutions; this is the BP condition checked so far. Then for every exact $M\ne M^*$,
$$\|M\|_1\ge\|M_{\rm off}\|_1+\tfrac12\mathrm{tr}(C\Sigma^{*-1})>\|M^*_{\rm off}\|_1+\tfrac12\mathrm{tr}(C\Sigma^{*-1})=\|M^*\|_1 .$$
  So the off-diagonal BP condition implies the penalised-diagonal one. In the population, penalising the diagonal can only help the dense-end pilot, and the BP counts computed so far are lower bounds.
- For an exact check of the penalised version, the dual certificate requires $A_S^\top h=\operatorname{sign}(m^*_S)$ on all of $S$, diagonal included, instead of $A_D^\top h=0$ on the diagonal. This is a one-line change in `bp_certificate`.
- When the BP condition holds, $\hat M^0\to M^*$, so $E\supseteq S^c$ in the limit, $r\to0$, $\rho_w\to0$, and Corollary W.3 applies.

## 6. What changed relative to DDK, line by line

| DDK (lasso, $\hat w\equiv1$) | here (weights $\hat w_j=\hat\mu/|\hat M^0_j|\ge1$) | why |
|---|---|---|
| (D.1) $\rho\lt 1-\alpha$ | (W2) $\rho_w\le1-\alpha$, with $\rho_w\le\rho r$ | the deterministic part of the dual vector (3.8) is $\hat w_j^{-1}[P(\hat w\circ\tilde z)_S]_j$ |
| $\varepsilon_1\le\alpha/(6c_{\Gamma^*})$ | (W3) $\varepsilon_1\le\alpha/(3c_{\Gamma^*}(\kappa r+\alpha))$ | $G_2$ carries $\kappa r$: bias $\lambda\bar w_S$ on $S$ against kink $\lambda\underline w$ on $S^c$ |
| $\lambda>\frac{3(2-\alpha)}{\alpha}\max\{c_{M^*}\varepsilon_1,\varepsilon_2\}$ | (W4) $\lambda\underline w>\frac{3\kappa}{\alpha}\max\{c_{M^*}\varepsilon_1,\varepsilon_2\}$; implied by $\lambda>\frac{3\kappa}{\alpha}\max\{\cdot\}$ since $\hat w\ge1$ | noise is compared with the smallest effective kink; $2-\alpha\to\kappa\ge1+\rho$ because $\rho$ may exceed $1$ |
| (D.4) $\|\tilde\Delta_M\|_\infty\lt \frac{2c_{\Gamma^*}\lambda}{2-\alpha}$ | (3.5)–(3.6) $\le b_w\lt c_{\Gamma^*}\lambda(\bar w_S+\frac\alpha\kappa\underline w)$ | bias $\lambda\to\lambda\bar w_S$ |
| (D.5) | (3.7) | $\operatorname{sign}(\cdot)\to\hat w\circ\tilde z$, row $j$ divided by $\hat w_j$ |
| beta-min $>\frac{2c_{\Gamma^*}\lambda}{2-\alpha}$ | $\beta_{\min}>b_w$ (all of $S$, diagonal included) | same role |
| Thm 2: $c_*=6c_{\Gamma^*}/\alpha$ | $c_w=3c_{\Gamma^*}(\kappa\bar r+\alpha)/\alpha$ | from (W3) |
| Prop. G.2 | Prop. W.5 | dual vector with limiting weights |

Everything specific to GCLMs enters exactly where it enters in DDK: through the invertibility of $\Gamma^*_{SS}$, the constants $c_{\Gamma^*},\rho,c_{M^*},c_{\Sigma^*},c_C$, and Appendix E. Invertibility of $\Gamma^*_{SS}$ holds for every simple graph (Dettling et al. 2023).

## 7. Remarks

**Numerical check** (`check_W1.py`, `gclmth`; weighted lasso with the diagonal penalised and the implemented weights).
- *In-star, $p=6$, $|w|=0.7$, correlation scale.* DDK's constant is $\rho=1.78$, so the lasso's condition fails. The pilot is truth $+\,U(-0.05,0.05)$, giving $\rho_w=0.066$ and $r=0.052$. At $n=10^6$ and $10^7$, (W3) holds. With $\lambda$ at (W4) plus 1 %, the weighted lasso returned exactly $S$, with $\|\hat M-M^*\|_\infty=0.13$ and $0.057$, below $b_w=0.79$ and $0.31$, which are below the right-hand side of (2.1).
- *Unweighted 3-path with ordered diagonal, raw scale ($\rho=0.81$, DDK's case), $\kappa=2-\alpha$.* The right-hand side of (2.1) equals DDK's $2c_{\Gamma^*}\lambda/(2-\alpha)$ to machine precision. $\hat S\subseteq S$ held, and part (b) was not reached because $\lambda$ is large relative to $\beta_{\min}=0.2$.

The required $n$ is large because DDK's $\varepsilon_1$ (row sums of $\Delta_\Gamma$ over the support columns) is a conservative quantity. The theorems are sufficient conditions, as in DDK.

**Unpenalised diagonal** (the default runs, waves 1–9). Set $\hat w=0$ on the diagonal $D$, take the maximum defining $\hat\mu$ over off-diagonal entries, replace $S$ by $S\setminus D$ in $\bar w_S$, (1.2), Corollary W.4 and the beta-min, and read the diagonal rows of (3.1) as $(\hat\Gamma\hat m-\hat g)_D=0$. Nothing else changes; the proof is the one above.

**Points in DDK to be aware of when citing.**
1. $g$ should carry the transpose (§0).
2. D.1's $\lambda$-condition is printed as $\max\{c_{M^*},\varepsilon_1,\varepsilon_2\}$; the proof uses $\max\{c_{M^*}\varepsilon_1,\varepsilon_2\}$ ($G_1$, $G_3$ on pp. 532–533), and Theorem 2 uses $c_{M^*}\varepsilon$ (implicitly $c_{M^*}\ge1$).
3. The beta-min displays index $(j,k)\in S$ with $1\le j\lt k$, inherited from the symmetric setting of Lin et al.; for a drift matrix it must run over all of $S$.
4. The block form on p. 531 writes $+\Delta_g$; with $\Delta_g=\hat g-g^*$ the sign is $-\Delta_g$. Only norms enter, so nothing depends on it.
