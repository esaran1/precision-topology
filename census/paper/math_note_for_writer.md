# Math note v2: the conditional threshold in the scaling limit

Supersedes `scaling_proposition.md` for the conditional-threshold claim. The earlier note's steps 1–5
(series, remainder bounds, Ĝ = Kε^{3/2}(1 + O(ε))) are unchanged and are not repeated here. Checks:
`src/limit_bnb.py`, `src/math_note_v2_checks.py`, `src/profiled_bnb.py`, `src/conditional_certified.py`.

## Three objects

The earlier statement blurred these. They are distinct and are treated separately.

1. **The gap-maximising placement** — argmax of G over (w₁, b₁). It defines Ĝ, K and κ. The earlier
   note's H1 localises this object only.
2. **A local branch of the conditional loss** — a strict local minimiser of the profiled loss at fixed
   output scale, followed continuously in the scale. H2′ (transversality) concerns this object.
3. **The globally preferred conditional-loss solution** — the global minimiser of the profiled loss at
   fixed output scale. The paper's threshold R_glob is where its gap changes sign.

Objects 2 and 3 coincide near the switch exactly where a certificate says so (§4). Object 1 enters
only through the unit R = |w₂|Ĝ/2.

## 1. Setting: the profiled problem in rescaled coordinates

- f_a(t) = t + a sin t, a = 1 + ε. Windows I = [−0.8, 0.8] (class 0), O = ±[1.2, 2.0] (class 1).
- The objective is a **fixed finite point set**: the 800 quadrature points of the population objective
  (`blockB_landscape.population_data`), or a run's own 400 training points where stated.
  "Certified" below means exact for that objective, with floating-point rounding bounded (§6).
- Rescaled coordinates: w₁ = √ε p, b₁ = π + √ε q, σ = px + q, A = sε^{3/2} with s = |w₂|.
- With φ_ε(σ) = [f_a(π + √ε σ) − π]/ε^{3/2}, the logit is z = Aφ_ε(σ) + b, and
  L_ε(p, q, b; A) = mean ℓ(z, y), with ℓ(z, y) = log(1 + eᶻ) − yz.
- **Profiled loss**: L*_ε(p, q; A) = min_b L_ε. This is strictly convex in b whenever both classes
  are present, so b*(p, q; A) is unique and C^∞.
- **Limit problem**: φ_0 = h, h(σ) = −σ + σ³/6, giving L*_0(p, q; A). It does not depend on a.
- **Limit gap**: G_0(p, q) = min_O h(σ) − max_I h(σ) in the A > 0 orientation. The finite gap is
  ε^{3/2}(G_0 + O(ε)).
- **Units**: R = sĜ(a)/2 = KA(1 + δ_ε)/2 with |δ_ε| = O(ε). K ∈ [0.5794558, 0.5794951] (exact-extrema
  branch and bound, `limit_K_base.csv`; T58's 0.579454926 is a refinement estimate just below the
  supremum). Certified R intervals use K_lo·A_lo/2 and K_hi·A_hi/2.

## 2. Statement

**Theorem (limit problem, certified).** For the base window, on the quadrature objective:

- (a) **Localisation**, proved given computed bounds: for every A ∈ [0.66, 0.72], every global
  minimiser of L*_0(·, ·; A) lies in K(24) = {|p| ≤ 24, |q| ≤ 2√2 + 48}.
- (b) **Switch**: the global minimiser's gap changes sign at a unique A* ∈ (0.68125, 0.6875]. Hence
  **R_glob^∞ = KA*/2 ∈ [0.19738, 0.19920]** (`limit_K_base.csv`). The solve switch is certified in the
  same way at A_solve ∈ (1.05875, 1.06], giving **R_solve^∞ ∈ [0.30675, 0.30712]** (transversal, nondegenerate;
  `mn2_solve_limit.csv`).
  - **Global over the whole bracket** (§3(c), same four-link chain as (c)): for every A ∈ [1.05875, 1.06]
    the global minimiser of L*_0(·; A) is unique (up to p → −p) and is the branch minimiser
    (`certv2_solve_summary.csv`).
- (c) **Single branch, global** (certified by a four-link chain, §3(c)):
  - for every A ∈ [0.66, 0.71], L*_0(·; A) has a unique global minimiser (up to the reflection p → −p). It lies
    in the box U = [p₀ ± 0.05] × [q₀ ± 0.05], with (p₀, q₀) = (1.66858, 1.36972);
  - the (p, q, b) Hessian is positive definite on U × [0.66, 0.71], with λ_min ≥ 0.0473;
  - so the global minimiser is the unique critical point in U, and it is C^∞ in A.
- (d) **Transversality (H2′)**: dG_0/dA along the branch lies in **[1.666, 1.705]** at A = 0.68125, 0.684375
  and 0.6875 (certified intervals, `mn2_h2prime.csv`). It is positive, so the crossing is transversal. The
  earlier float 1.304 was a finite difference of coarse branch-and-bound argmins; it is superseded.

**Corollary (finite a, local).** For all sufficiently small ε, L*_ε has a unique critical point in U for
each A ∈ [0.66, 0.71]. It is a strict local minimum, and its gap changes sign transversally at
A_ε = A* + O(ε). In R units: **R_ε = KA*/2 + O(ε)**.

**Global preference at finite a is not claimed from the limit.**
- The isotonic localisation argument uses h's monotonicity for |σ| ≥ 2√2. f_a has periodic folds, so
  it does not transfer.
- Global preference at finite a is established only where Block 1c's certificate covers it: a = 1.30,
  1.35, 1.40, 1.45, 1.50 and 1.60 (§5).

## 3. Proof

**(a) Localisation.**
- h is increasing on |σ| ≥ √2, with h(2√2) = h(−√2) = 2√2/3 and h(−2√2) = h(√2) = −2√2/3. So h
  restricted to {|σ| ≥ 2√2} is increasing.
- For |p| ≥ P, the points with |σ(x)| < 2√2 lie in an x-window of length 4√2/|p| ≤ 4√2/P.
- On the remaining points the logit is monotone in x. So L*_0 ≥ (1/n) × (the best monotone logistic fit
  to those points).
- That best fit is the isotonic regression, for any Bregman loss, including log loss.
- Minimising over window positions gives B(P) ≤ L*_0 for all |p| ≥ P. If |q| > 2√2 + 2|p|, no point
  is in the window, and L*_0 ≥ B_full.
- **Exact values** (`mn2_bounds.csv`: integer PAVA block sums, logs at 50 digits):
  - B(24) = 0.38797358;
  - B_full = ln3/4 + ln(3/2)/2 = 0.47738563 (closed form, agreeing with PAVA to 7e−52).
- **Branch loss**: the certified upper bound on [0.66, 0.72] is 0.3768942 (`mn2_uniformity_summary.csv`),
  below B(24) by 0.0111 and below B_full by 0.1005. So every global minimiser lies in K(24).
- On the wider range [0.60, 0.76], the 0.02-grid bridge overshoots B(24) by 0.0032. It is not
  certified there at that resolution, and it is not needed.
- **Uniformity in A**: L*_0(θ; A) is convex in A (ℓ is convex and z is jointly linear in (A, b)).
  |∂_A L*_0| ≤ mean|h(σ)|.

**(b) Switch.**
- Profiled branch and bound on K(24) (`limit_bnb.certify`), for the regions G_0 ≤ 0 and G_0 > 0.
- **Cell lower bound**: L*(c) − |∂_p|h_p − |∂_q|h_q − ½A·mean(|σ|max·(|x|h_p + h_q)²).
  - The gradient term is exact (envelope theorem).
  - The profiled Hessian's Schur-complement correction is bounded by Cauchy–Schwarz, so the
    full-variable curvature term bounds the profiled curvature from below.
- **Gap step**: max|h′| over the cell × (2.8h_p + 2h_q).
- At A = 0.68125, min_{G≤0} < min_{G>0} with disjoint certified intervals. At A = 0.6875 the order
  is reversed. (`limit_switch.csv`, `limit_switch_evaluations.csv`.)
- **Uniqueness of the sign change** on [0.66, 0.71] follows from (c)–(d).

**(c) Single branch.**
- `mn2_neighbourhood.csv`: U × [0.66, 0.71] is split into 500 sub-boxes. On each:
  - b is enclosed by interval Newton (validated on all boxes);
  - the interval (p, q, b) Hessian has eigenvalues ≥ λ_mid − ‖radius‖_F > 0;
  - the minimum lower bound over all boxes is 0.0473.
- A C² function with a PD Hessian on a convex box has at most one critical point there.
- **Global uniqueness: the four-link chain** (`certificates_v2.py` → `certv2_annulus_parts.csv`,
  `certv2_annulus_summary.csv`; solve bracket → `certv2_solve_parts.csv`, `certv2_solve_pd.csv`,
  `certv2_solve_summary.csv`).
  - A is carried as an interval, never sampled. The chain holds on each A-sub-interval: 40 sub-intervals of
    width 0.00125 covering [0.66, 0.71], and the solve bracket [1.05875, 1.06] as one interval.
  - θ_c is U's centre (p₀, q₀). For the solve bracket it is the centre of the certified argmin enclosure at
    A = 1.06, with its own 0.05 box.
  - **Link 1, localisation**: every global minimiser lies in K(24) (by (a)).
    - For the solve bracket, the branch upper bound max(L*_0(θ_c; 1.05875), L*_0(θ_c; 1.06)) = 0.28117 is
      below B(24) = 0.38797.
  - **Link 2, outer exclusion**: for every θ ∈ K(24) outside the 0.15 box around θ_c, and every A in the
    sub-interval, L*_0(θ; A) > L*_0(θ_c; A). The margin is certified at ≥ 1.00e−4 on every annulus
    sub-interval and ≥ 1.48e−4 on the solve bracket.
    - **Branch side**: L*_0(θ_c; ·) is convex, so over the sub-interval it is at most the larger of its two
      endpoint values.
    - **Competitor side**: the §3(b) cell bound at the midpoint A_c, with curvature taken at A_hi, minus
      h_A·|∂_A L*_0| (convexity in A). By the envelope theorem, ∂_A L*_0 = mean((σ(z_i) − y_i)h(σ_i)).
    - The profiled bias gives mean(σ(z_i) − y_i) = 0. So for any constant c,
      |∂_A L*_0| ≤ mean|h(σ_i) − c| ≤ mean(|h(σ_i(θ_cell)) − c| + (1 + σmax_i²/2)(|x_i|h_p + h_q)),
      taking c as the median over points at the cell centre.
    - The cruder bound mean|h(σ)| reaches about 10⁴ near p = 0, |q| ≈ 36. There h(σ) is nearly constant
      across points and b absorbs it, and with the crude bound the search does not close.
    - A 1e−9 allowance covers float rounding in the profiled evaluations.
    - The search stops at 7 refinement rounds with ≤ 39,072 live cells, against a cap of 1.5M.
  - **Link 3, ring**: ∇_{(p,q)} L*_0(·; A) ≠ 0 on the ring 0.05 ≤ |θ − θ_c|_∞ ≤ 0.15, for every A in the
    sub-interval.
    - **Method**: vectorised interval arithmetic with outward rounding after every operation. exp and pow
      are widened by ≥ 4 ulps, and the mean carries a summation error bound. The profiled b is enclosed by
      certified sign changes of the monotone bounds mean σ(z_lo + b) − ȳ ≤ F(b) ≤ mean σ(z_hi + b) − ȳ.
    - The gradient ∂_p L*_0 = mean((σ(z) − y)Ah′(σ)x), ∂_q L*_0 = mean((σ(z) − y)Ah′(σ)) is enclosed on
      540 boxes of side 0.0125 covering the ring (padded by 1e−12). On every box, 0 is excluded from ∂_p
      or from ∂_q, with no subdivision needed.
    - The smallest certified |gradient component| is 3.28e−3 (annulus) and 8.47e−3 (solve bracket).
  - **Link 4, PD**: the (p, q, b) Hessian is PD on the 0.05 box × the A range. For the annulus this is
    `mn2_neighbourhood.csv` (λ_min ≥ 0.0473, 500 sub-boxes). For the solve bracket it is `certv2_solve_pd.csv`:
    100 sub-boxes over the whole bracket, b validated by interval Newton, λ_min ≥ 0.0209.
  - **Hence**:
    - A global minimiser exists in K(24) (Link 1, continuity).
    - It lies in the 0.15 box (Link 2).
    - It is an unconstrained minimiser, so it is a critical point, and therefore it is not in the ring
      (Link 3).
    - So it lies in the 0.05 box, where the critical point is unique (Link 4).
    - The global minimiser is therefore unique up to the reflection p → −p, and it is the branch minimiser.
      The data and windows are x-symmetric, and the search covers p ≥ 0, as in `limit_bnb.certify`.
  - **Tests**: `tests/test_certificates_v2.py`, 12 tests.
    - The interval primitives are checked against 50-digit values, and the b and gradient enclosures
      against float profiles at random interior points.
    - **Constructed fail cases**, each correctly not certified:
      - a centre shifted by 0.1, so the critical point is in the ring;
      - the inner box removed;
      - a centre shifted by 0.3, so the minimiser is outside the box;
      - a box too small for the A-interval;
      - the cell cap reached.
    - The halving-tree coverage bookkeeping and the three-link conjunction are also tested.
  - **Superseded first design** (`math_note_v2_checks.annulus`, `solve_competitor`; never produced an
    artifact). It compared the competitor bound directly against the rise at the edge of the 0.05 box,
    with a branch upper bound carrying h_A·mean|h|.
    - Its slack (7.2e−3 annulus, 1.2e−3 solve) exceeded that rise (3.3e−4 and 6.8e−4), so it could not
      close at any resolution.
    - The redesign widens the exclusion to the 0.15 box and closes the gap between 0.05 and 0.15 with the
      ring certificate instead.

**(d) Transversality.**
- Along the branch, G_0 is differentiable wherever its active set is unique: one extremal point of I
  and one of O.
- dG_0/dA = ∇G_0 · dθ/dA, with dθ/dA = −H⁻¹∂_A∇L_0 (implicit-function theorem on the full
  (p, q, b) gradient system).
- Evaluated in interval arithmetic over the certified argmin enclosure (`ift_limit`, `h2`, tolerance 1e−11).
  At all six (A, region) evaluations:
  - b is enclosed;
  - the Hessian is PD (λ_min ≥ 0.1470);
  - the active pair (outer −1.2, inner 0.8) is unique;
  - dG_0/dA ∈ [1.666, 1.705].
- The global minimiser's certified gap is [−0.00715, −0.00703] at A = 0.68125 and [0.00339, 0.00350] at
  0.6875. That agrees with the Krawczyk switch A* = 0.685445 (§8): the slope ≈ 1.68 predicts −0.00707 and
  +0.00346.

**Corollary: C² convergence.** On U, |σ| ≤ S_U = 2(|p₀| + ρ) + |q₀| + ρ = 4.857 (since |x| ≤ 2). For
ε ≤ 0.05 (`taylor_constants`, remainders derived in the docstring):

  |φ_ε^{(k)} − h^{(k)}| ≤ εM_k on |σ| ≤ S_U, with M₀ = 5.217, M₁ = 13.507, M₂ = 16.375, M₃ = 13.384.

- The remainder keeps the (1 + ε) factor, per the writer's correction.
- ∂^k_{(p,q)}z carries factors x^k ≤ 2^k.
- The logistic derivatives are bounded: |ℓ′| ≤ 1, ℓ″ ≤ ¼, |ℓ‴| ≤ 1/(6√3).
- So ‖L_ε − L_0‖_{C²(U×[−B,B]×[0.66,0.71])} ≤ C·ε, with C explicit in A ≤ 0.71, S_U and M₀–M₂.
- For ε with Cε below λ_min = 0.0473, the Hessian stays PD on U. The critical point persists, and it
  moves by O(ε)/λ_min, staying inside U for small ε.
- **The gap**: G is 2-Lipschitz in the sup norm of φ (writer's correction: each of its two terms
  is a max or min, and there are two of them). Its derivative along the branch converges at O(ε)
  under the unique active set.
- So the sign change persists at A_ε = A* + O(ε), and it stays transversal.
- **Units**: R = KA(1 + δ_ε)/2 with |δ_ε| = O(ε) (earlier note, step 4). ∎

## 4. Checklist: proved, certified or assumed

| Question | Status | Evidence | What would upgrade it |
|---|---|---|---|
| Loss-minimising branch (object 3) inside the rescaled region, limit problem | **Proved** given computed bounds, uniformly for A ∈ [0.66, 0.72] | isotonic bound; `mn2_bounds.csv`, `mn2_uniformity_summary.csv` | — |
| Same, finite a | **Certified at the reported a only** (Block 1c, on the certified domain \|w₁\| ≤ W(s, a)) | `cond_certified_brackets.csv` | a monotone-fold argument for f_a; not available (periodic folds) |
| Branch interior to where the expansion holds | **Certified**: U with S_U = 4.857, ε ≤ 0.05 | `mn2_neighbourhood.csv` | — |
| Uniform C² derivative control | **Proved**, with explicit constants M₀–M₃ | `taylor_constants` | — |
| Hessian PD on the neighbourhood (IFT) | **Certified**: λ_min ≥ 0.0473 on 500 sub-boxes | `mn2_neighbourhood.csv` | — |
| No competitor outside U, uniform in A | **Certified** for every A ∈ [0.66, 0.71] (40 A-sub-intervals, A as an interval): outer exclusion outside the 0.15 box (margin ≥ 1.00e−4), no critical point in the ring 0.05–0.15 (\|gradient component\| ≥ 3.28e−3), PD on U | `certv2_annulus_parts.csv`, `certv2_annulus_summary.csv` | — |
| Transversal crossing (H2′) at A* | **Certified**: dG₀/dA ∈ [1.666, 1.705] over the bracket | `mn2_h2prime.csv` | — |
| Local branch or global minimiser? | **Global and unique** (up to p → −p) in the limit for A ∈ [0.66, 0.71] and over the solve bracket (localisation, four-link chain §3(c)); **global at finite a** only at a ∈ {1.30, …, 1.60} (Block 1c); **local** otherwise | as above | — |
| Solve threshold: nondegeneracy and transversality | **Limit: certified** — A_solve ∈ (1.05875, 1.06]; the global minimiser's solve margin is certified negative then positive; Hessian PD (λ_min ≥ 0.1369); d(margin)/dA ∈ [0.532, 0.536]; localised (branch loss 0.281 < B(24) = 0.388). **Global over the whole bracket**: the four-link chain certifies a unique global minimiser for every A ∈ [1.05875, 1.06] (outer margin ≥ 1.48e−4; ring \|gradient component\| ≥ 8.47e−3; PD λ_min ≥ 0.0209; branch loss ≤ 0.28117 < B(24)) (`certv2_solve_summary.csv`). **Finite a: certified.** At a = 1.30–1.60, the global minimiser's solve margin over its certified enclosure (width ≤ 9e−6), with b validated, is certified negative at every lower bracket end and positive at every upper end. The smallest margin is [4.1e−6, 3.1e−4] at a = 1.45. | `mn2_solve_limit.csv`; `mn2_solve_finite.csv` | — |
| Other window geometries (**for the rebuttal**; computed after the submission set was scored) | **Switch bracketed on K(24) for all 9 windows** (width 0.003125; H10 0.00625, stopped at a tied midpoint with both ends certified). **Localised (global) for 7 of 9**. For G1 and G4 the branch loss over the bracket exceeds B(24) (0.408 > 0.353; 0.426 > 0.388), so A* there is certified only for the minimiser over K(24). Base window: A* ∈ [0.68375, 0.686875], consistent with the sharp A* = 0.685445. R∞ intervals: base [0.1981, 0.1990]; G1 [0.1964, 0.1982]; G2 [0.1230, 0.1232]; G3 [0.1931, 0.1939]; G4 [0.1344, 0.1361]; H10 [0.1864, 0.1875]; H35 [0.1978, 0.1987]; H65 [0.1989, 0.2002]; H90 [0.1984, 0.1999] | `limit_windows.csv`, `limit_windows_evaluations.csv` | G1, G4: localisation with a larger P |

## 5. Finite-a certificates (Block 1c)

### 5.0 Localisation of the finite-a search: the bound W(s, a)

The finite-a certificates (Block 1c: `conditional_certified.evaluate` → `profiled_bnb.certify`) search the
compact domain |w₁| ≤ W(s, a), b₁ ∈ [0, 2π). The lemma below is what makes that domain sufficient.

**Lemma (localisation of the finite-a conditional search).**

- **Setting**:
  - z(x) = s·f_a(w₁x + b₁) + b₂, with s = |w₂| > 0 in the orientation w₂ > 0, and f_a(t) = t + a sin t, so
    t − a ≤ f_a(t) ≤ t + a.
  - Class 1 lies on O = [−2, −1.2] ∪ [1.2, 2] and class 0 on I = [−0.8, 0.8], with n points in total and balanced
    classes (ȳ = ½; true of the 800-point population and of every 400-point training set).
  - L* is the profiled loss, minimised over b₂.
- **Data**: the population objective has n = 800 points:
  - 400 inner points x = linspace(−0.8, 0.8, 400), class 0;
  - 200 outer points x = linspace(1.2, 2.0, 200), class 1, and their 200 negatives, class 1.
  (`blockB_landscape.population_data`; a training set is `fold1d.make_data(200, seed)`.)
- **Notation**: for a cut c ∈ [0, 0.8), let n_O⁻ = #{class 1: x ≤ −1.2} and n_I^{≥c} = #{class 0: x ≥ c}. Set
  π(c) = min(n_O⁻, n_I^{≥c})/n and δ(π) = 2 log(2^{1/π} − 1).
- **The cut grid**: C = {0.799·k/79 : k = 0, 1, …, 79}, the 80 cuts the certified code evaluates
  (`np.linspace(0, 0.799, 80)`).
- **Claim**: if w₁ > W₊(s, a) = min over c ∈ C of (2a + δ(π(c))/s)/(1.2 + c), then L*(w₁, b₁; s) > log 2 for every b₁.
  (The argument holds for every single c; minimising over the grid C reproduces the code exactly.)
  The case w₁ < 0 is the mirror image: use the right-outer class-1 points (x ≥ 1.2) against the class-0 points
  with x ≤ −c, which gives W₋.
- **Conclusion**: with W(s, a) = max(W₊, W₋), every global conditional minimiser has |w₁| ≤ W. The constant
  predictor attains exactly log 2, so the global minimum is at most log 2.
- **b₁**: it ranges over [0, 2π). A shift of b₁ by 2π adds 2πs to every logit, which the profiled b₂ absorbs.

**Proof** (w₁ > 0).
1. For x ≤ −1.2, w₁x + b₁ ≤ −1.2w₁ + b₁, so z ≤ A := s(−1.2w₁ + b₁ + a) + b₂.
2. For x ≥ c, z ≥ s(cw₁ + b₁ − a) + b₂ = A + Δ, with Δ = s((1.2 + c)w₁ − 2a).
3. Take Δ > 0 and let M = A + Δ/2.
   - If M ≥ 0, every class-0 point with x ≥ c has z ≥ Δ/2, so its loss softplus(z) is at least softplus(Δ/2).
   - If M < 0, every class-1 point with x ≤ −1.2 has z < −Δ/2, so its loss softplus(−z) exceeds softplus(Δ/2).
4. All other losses are non-negative. So for every b₂, L ≥ π(c)·softplus(Δ/2), and hence
   L* ≥ π(c)·softplus(Δ/2).
5. π·softplus(Δ/2) > log 2 ⟺ Δ > δ(π) ⟺ w₁ > (2a + δ(π)/s)/(1.2 + c).
6. Every c gives a valid bound, so the minimum over the grid C does too.
   (For 1/π ≥ 1000 the code replaces δ by the larger 2 log 2/π, which is conservative; this never occurs here.) ∎

**Illustration** (a = 1.30, s = 4.95, the lower end of the certified R_glob bracket; population data):
- **With the cut c = 0.4**: n_O⁻ = 200, n_I^{≥0.4} = 100, π = 1/8, δ = 2 log 255 = 11.08. This gives
  W₊ = (2.6 + 11.08/4.95)/1.6 = 3.024.
- **With the grid-optimal cut c = 0.799·22/79 = 0.22251**: n_I^{≥c} = 145, π = 145/800 = 0.18125,
  δ = 2 log(2^{1/0.18125} − 1) = 7.6045, W₊ = (2.6 + 7.6045/4.95)/(1.2 + 0.22251) = 2.9077.
  W₋ is the same, because the data are x-symmetric, so **W = 2.908**: the value the certified search used.

**Checked numerically at every certified a** (`writer_patch.w_bound_table` → `writer_patch_w_bound.csv`: both
ends of all 12 certified R_glob and R_solve brackets, population objective):
- the formula reproduces the W used by the search to 1e−12;
- the profiled loss at |w₁| = W, 1.5W and 3W, over 720 values of b₁ and both signs of w₁, is at least 4.34 > log 2.
- W ranges from 2.56 (a = 1.30, R_solve bracket) to 4.86 (a = 1.60, R_glob bracket).


- Certified brackets for s = |w₂|, width 0.0125, at a = 1.30 … 1.60 (`cond_certified_brackets.csv`).
  For example, R_glob(1.30) ∈ (0.21310, 0.21364] and R_solve(1.30) ∈ (0.30566, 0.30620].
- The frozen Block B grid value is the first grid point above the bracket in every case.
- **Convergence, refitted on certified intervals** (`rglob_refit.py` → `rglob_convergence_refit.csv`,
  `rglob_convergence_points.csv`):
  - finite-a values exceed R_glob^∞ by +7.6 to +13.5% (midpoints), positive at the worst corner, monotone in a;
  - log-log slope 0.829 at midpoints, exact range [0.714, 0.955] over the certified box (the frozen-grid
    fit gave 0.947);
  - **no one-term law R_glob^∞(1 + c₁ε) passes through all certified intervals**; a two-term law does,
    with c₁ ∈ [0.243, 0.321] and a strictly negative ε² coefficient;
  - a straight line through the six finite a alone extrapolates to [0.2002, 0.2035], outside the
    certified limit.
- So the O(ε) rate is the corollary's, not the fit's. At ε = 0.30–0.60 the ε² term is not negligible,
  and the measured points do not isolate the leading coefficient better than [0.243, 0.321].

## 6. Chain table

| Statement | Hypothesis discharged | Script → artifact | Rounding margin |
|---|---|---|---|
| (a) localisation | H1′ (limit) | `math_note_v2_checks bounds`, `uniform` → `mn2_bounds.csv`, `mn2_uniformity*.csv` | B(24): exact rational/50-digit. Branch bound: margin 0.0111 ≫ float error ≤ 1.1e−16 (`mn2_rounding.csv`) |
| (b) switch at A* | global order at A = 0.68125, 0.6875 | `limit_bnb switch` → `limit_switch.csv` | certified interval gaps vs interval re-evaluation: \|float − interval\| ≤ 1.1e−16 (`mn2_rounding.csv`) |
| (c) uniqueness in U | Hessian PD | `math_note_v2_checks neighbourhood` → `mn2_neighbourhood.csv` | interval arithmetic throughout (mpmath.iv, 30 digits) |
| (c) competitors outside U | outer exclusion (0.15 box) + ring (no critical point, 0.05–0.15), uniform over 40 A-sub-intervals | `certificates_v2 annulus` → `certv2_annulus_parts.csv`, `certv2_annulus_summary.csv` | outer: float B&B with a 1e−9 rounding allowance against a certified margin ≥ 1.00e−4; ring: outward-rounded interval arithmetic |
| (d) transversality | H2′ | `math_note_v2_checks h2` → `mn2_h2prime.csv` | interval arithmetic (mpmath.iv), argmin enclosure at tolerance 1e−11 |
| C² convergence | derivative control | `taylor_constants` → `mn2_neighbourhood.csv` (M₀–M₃) | closed-form bounds, float max over 200,001 points of polynomials |
| finite-a global preference | Block 1c | `conditional_certified brackets` → `cond_certified_brackets.csv` | `mn2_rounding.csv`, a = 1.30 rows: ≤ 1.1e−16 against certified gaps ≥ 1.8e−8 |
| solve threshold (limit), branch | nondegeneracy, transversality | `math_note_v2_checks solve_limit` → `mn2_solve_limit.csv` | interval arithmetic over tolerance-1e−11 enclosures; b validated |
| solve threshold (limit), competitors | the same chain over [1.05875, 1.06]; Hessian PD on the 0.05 box × bracket | `certificates_v2 solve` → `certv2_solve_parts.csv`, `certv2_solve_pd.csv`, `certv2_solve_summary.csv` | as (c); PD in mpmath.iv (30 digits) |
| solve threshold (finite a) | sign of the solve margin at both bracket ends | `math_note_v2_checks solve_finite` → `mn2_solve_finite.csv` | interval arithmetic over the argmin enclosure (tolerance 1e−11); b validated by interval Newton |

## 7. Limits

- The corollary is asymptotic: "for all sufficiently small ε". No explicit ε₀ is claimed. The measured
  finite-a thresholds (ε = 0.30–0.60) sit 7–14% above the limit.
- Global preference at finite a rests on Block 1c at the six reported a, not on the limit.
- Everything is exact for the stated finite objective. For another sample (for example a run's own 400
  training points) the threshold differs; see `own_threshold_prediction.md`.

## 8. The first-order coefficient c₁ (`src/first_order.py` → `first_order_c1.csv`)

**Exact bookkeeping.** With A = sε^{3/2} and K(ε) := Ĝ(a)/ε^{3/2}, R = sĜ(a)/2 = A·K(ε)/2 exactly. So if the
switch sits at A_ε = A* + A′(0)ε + O(ε²) and K(ε) = K(1 + k₁ε + O(ε²)), then

  R_glob(ε) = R_glob^∞ (1 + c₁ε + O(ε²)),   **c₁ = A′(0)/A* + k₁**.

**Expansion.** φ_ε(σ) = [f_a(π + √εσ) − π]/ε^{3/2} = h(σ) + ε r(σ) + O(ε²), with r(σ) = σ³/6 − σ⁵/120.
Both terms come from the sine series, including the (1 + ε) factor.

**Switch shift, A′(0).**
- At the switch the gap's active pair is inner x = 0.8 and outer x = −1.2. On the box below, every other
  candidate is strictly dominated (margins 0.625 inner and 0.108 outer), so near the switch G₀ = h(q − 1.2p) − h(q + 0.8p)
  is smooth.
- The switch is a zero of Φ(p, q, b, A; ε) = (∇_{p,q,b} L_ε, G_ε).
- **At ε = 0**: a Krawczyk test on a box of radius 1e−9 certifies a unique zero, at
  (p, q, b, A*) = (1.6680839, 1.3692319, −0.6141196, **0.68544523757565**). This is 800-point quadrature
  objective, interval arithmetic at 30 digits. The Jacobian J = ∂_{(p,q,b,A)}Φ is invertible over the box.
- **Implicit-function theorem**: d(p, q, b, A)/dε = −J⁻¹∂_εΦ, where
  - ∂_ε∇_θL = mean[σ(z)(1 − σ(z))·A r(σ)·∂_θz + (σ(z) − y)·∂_θ(A r(σ))];
  - ∂_εG = r(σ_O) − r(σ_I).
  - The linear system is enclosed rigorously.
- Result: **A′(0)/A* ∈ [0.6621547, 0.6621550]**.

**Domain lemma for K = sup G₀ (added 2026-09-24).** The branch and bound for K searches (u, v) ∈ [0, 8] × [−12, 12];
this lemma shows nothing outside that box can exceed K.
- **Identity.** For h(σ) = −σ + σ³/6 and d ≥ 0: h(a) − h(a + d) = d·[1 − ((a + d/2)² + d²/12)/2].
- **Bounds.** For u ≥ 0, σ = ux + v increases with x. Each orientation of G₀ is at most its value at one outer point
  against one inner point (a minimum is at most any member of its set, a maximum at least any member):
  - O over I: ≤ h(σ(−2)) − h(σ(−0.8)) (d = 1.2u), and ≤ h(σ(−2)) − h(σ(0.8)) (d = 2.8u, a + d/2 = v − 0.6u);
  - I over O: ≤ h(σ(0.8)) − h(σ(2)) (d = 1.2u), and ≤ h(σ(−0.8)) − h(σ(2)) (d = 2.8u, a + d/2 = v + 0.6u).
- **Lemma.** G₀(u, v) ≤ 0 if u ≥ √(50/3) ≈ 4.0825 (then d²/12 ≥ 2 for d = 1.2u), or if |v| ≥ √2 + 0.6u (then
  (a + d/2)² ≥ 2 in both orientations). G₀(−u, v) = G₀(u, v) by x ↦ −x, since I and O are symmetric.
- **Consequence.** K > 0, so sup G₀ over the whole plane equals sup over {|u| < 4.0825, |v| < 3.8637}. With u ≥ 0 by
  the symmetry, that region lies inside the certified box. K's argmax (1.6056, 1.2042) lies inside the region.
- **Check** (`src/k_domain.py` → `k_domain_check.csv`, numerical, not a certificate):
  - the identity holds to 1.4e−12 relative at 100,000 random (a, d);
  - G₀ ≤ 0 at 4.0 million points of the excluded region (a dense grid on u ∈ [0, 60], v ∈ [−80, 80] and random
    points to 10⁴), with maximum exactly 0 at u = 0 and −1.19e−3 for u > 0;
  - the symmetry holds exactly on 10,000 random points.

**Gap-maximiser correction, k₁.**
- K = sup G₀ is attained at a vertex of the max–min: I(−0.8) = I(0.8) and O(−2.0) = O(−1.2). The mirror
  vertex is v ↦ −v.
- Krawczyk in 2D gives (u, v) = (1.6055757, 1.2041818) and K = 0.57945588342 (±2e−11). This lies inside the
  branch-and-bound enclosure [0.5794558833, 0.5794559217].
- It is a strict local maximum: 0 is interior to the convex hull of the four piece-gradient differences
  (largest angular gap 3.080 < π). The other pieces are dominated.
- The vertex equations persist under ε. Differentiating them gives K′(0) = ∂_ε(O − I) + ∇(O − I)·θ′, with
  J_Eθ′ = −∂_εE.
- Result: **k₁ ∈ [−0.37692472, −0.37692472]**.
- **Check against the exact φ_ε**: (K(ε)/K − 1)/ε = −0.3746, −0.3767, −0.3769 at ε = 10⁻², 10⁻³, 10⁻⁴.
  These were computed **with the active set fixed** (Newton on the two tie equations). They check the
  ε-dependence of that corner, not that the corner remains the maximiser. The free check below supplies
  that.

**Justification of the expansion at the tied corner** (`first_order corner` → `first_order_corner.csv`,
`first_order_corner_free.csv`).
- **Active edges** (the orientation min_O φ − max_I φ, v > 0):
  - the inner maximum is a tie between the two inner edges, x = −0.8 and x = +0.8;
  - the outer minimum is a tie between both edges of the negative outer window, x = −2.0 and x = −1.2.
  - The mirror maximiser (v < 0) uses the positive outer window, by x ↦ −x.
- **Uniform in ε**: φ_ε(σ) = −σ + (1 + ε)Σ_{k≥1}(−1)^{k+1}ε^{k−1}σ^{2k+1}/(2k+1)! is entire in ε. It is
  evaluated in interval arithmetic with a rigorous tail bound, for either sign of ε. On ε ∈ [−0.05, 0.05],
  split into 400 pieces, the following are certified on every piece:
  - **(i) Unique corner**: a parametric Krawczyk test gives a unique corner θ(ε) in a box of width ≤ 3e−4
    that serves every ε in the piece.
  - **(ii) Active set unchanged**:
    - φ_ε″ = (1 + ε)sin(√εσ)/√ε (the sinh form for ε < 0), and √|ε|·max|σ| < π. So φ_ε is concave for
      σ < 0 and convex for σ > 0.
    - φ_ε′ < 0 is certified on the inner window's negative σ-part. So the inner maximum sits at an edge.
    - The negative outer window lies at σ < 0, so its minimum sits at an edge.
    - φ_ε′ > 0 is certified on the positive outer window. Its minimum, the left edge, strictly exceeds
      the active outer value.
    - The other orientation is negative.
  - **(iii) Strict (sharp) local maximum**: 0 is strictly inside the convex hull of the four piece-gradient
    differences. The certified barycentric weights are ≥ 0.00434 on every piece (0.0047 at ε = 0).
- **Why the expansion is two-sided.**
  - At a non-smooth maximum of a min–max, the value function is in general only directionally
    differentiable (Danskin).
  - Its left and right derivatives differ when the active set changes at ε = 0.
  - Here the active set is the same on both sides, and the corner stays a sharp maximum. The tie
    equations E(θ, ε) = 0 are analytic, with J_E invertible.
  - So θ(ε) and the local maximum value K_loc(ε) = G_ε(θ(ε)) are analytic on the neighbourhood.
  - Hence k₁ = K_loc′(0)/K is an ordinary two-sided derivative.
- **Thin margin**: the smallest weight belongs to the O(x = −2.0) piece. The corner is sharp but close to
  degenerate in that direction. At some larger ε that tie could release; this is not tested here.
- **Global maximiser (object 1)**: K(ε) = Ĝ(a)/ε^{3/2} is the global supremum, so K = K_loc needs the global
  maximiser to be this corner.
  - At ε = 0 this is certified by branch and bound.
  - At ε = 0.005, 10⁻³ and 10⁻⁴, a **free** branch and bound (exact φ_ε extrema, no active set assumed,
    (u, v) ∈ [0, 8] × [−12, 12]) finds the maximiser at the corner or its mirror, with every surviving
    cell within 3e−6.
  - Its value encloses the fixed-active value at each of those ε. The free slopes (K/K₀ − 1)/ε are
    [−0.37576, −0.37574], [−0.37669, −0.37662] and [−0.37690, −0.37624].
  - ε = 10⁻² is omitted, because K(0.01) belongs to the registered test. That run records the global
    Ĝ(a) argmax at each test a.

**Result: c₁ ∈ [0.2852300, 0.2852303].**

**Chronology.**
1. **Before v4** (`scaling_limit_results.md`, T58): c₁ ≈ 0.49 was predicted from an incomplete argument.
   It used K(ε) with only the σ³/6 part of r and omitted the switch shift. It was set against the
   frozen-grid fit c₁ = 0.231 ("a factor 2.4 too large"), and that note already called the argument
   incomplete.
2. **Certified refit** (`a023380`, `6dc47f2`, §5): on certified intervals no one-term law fits. Two-term
   laws give c₁ ∈ [0.243, 0.321], so 0.49 is inconsistent with the data.
3. **Full first-order calculation** (`f92b1b5`, this section): c₁ ∈ [0.2852300, 0.2852303]. It was done
   after the refit had been seen. It has no fitted input, but it was not blind to the large-ε range.
4. **Independent check**: the registered small-ε test at a = 1.01–1.04 (`first_order_prediction.md`,
   registered in `f92b1b5` before any finite-a computation there).
   - **Scored: INCONCLUSIVE.** The feasible set [0.2252, 0.3302] is 0.105 wide, against a registered limit
     of 0.1. The prediction lies inside it, and all four competing values are excluded.
   - The component k₁ test PASSED.
   - A supplementary branch-root analysis, with no verdict attached, narrows the feasible set to
     [0.28460, 0.28590], which contains the prediction (`first_order_prediction.md`, Result).

**Why the earlier prediction (≈ 0.49, `scaling_limit_results.md`) was wrong.**
- It used K(ε) alone, with only the σ³/6 part of r.
- It omitted the switch shift, which is the larger term and has the opposite sign: the conditional
  minimiser trades gap against loss, as that note itself suspected.
- The complete first-order value, 0.285, lies inside the range [0.243, 0.321] allowed by the certified
  large-ε values (§5). So the "factor 2.4" discrepancy came from an incomplete calculation.
- Whether 0.285 is the actual first-order slope is tested prospectively at a = 1.01–1.04
  (`first_order_prediction.md`).

**Sharp limit threshold.**
- The Krawczyk A* combined with the K enclosure gives **R_glob^∞ ∈ [0.1985926, 0.1985927]**.
- This identifies the global switch given (c)'s competitor exclusion, which is now certified (§3(c)):
  - for every A ∈ [0.66, 0.71] the global minimiser is the unique critical point in U;
  - the Krawczyk switch point (p*, q*) = (1.66808, 1.36923) at A* = 0.685445 lies in U (5e−4 from its
    centre).
  - It lies inside the branch-and-bound bracket [0.19738, 0.19920].

## 9. Where the corner structure breaks (EXPLORATORY; `corner_tracking.py`)

- **Method**: the global certified Ĝ(a) branch and bound (full placement domain, relative target 1e−8), with
  no active set assumed. The maximiser's active edges and interior critical points are classified at every
  ε from 0.05 to 2.0 in steps of 0.025 (79 values), and each change is bisected.
- **Result: the active set never changes.** At every one of the 79 ε, the maximiser is the same four-piece
  corner:
  - an inner tie at x = ±0.8;
  - an outer tie at x = −2.0 and −1.2.
  - Every inactive candidate is separated by a relative margin ≥ 0.97 (`corner_tracking_changes.csv` is
    empty).
- **Confirmed directly**: the corner solved from its two tie equations with the exact f_a equals the
  branch-and-bound supremum, within 1e−9, at ε = 0.05, 0.3, 0.6, 1.0, 1.5 and 2.0.
- **Resolution caveat**: a change that reverses within 0.025 in ε would be missed.
- **Conditional minimiser**: its active pair (inner x = +0.8, outer x = −1.2) is also unchanged at all six
  certified switches, a = 1.30–1.60 (`corner_tracking_switch_active.csv`). The small multiplier on the
  O(−2.0) piece (0.004 at ε = 0) does not release anywhere up to ε = 2.0.
- **So the concave approach seen in the certified refit (§5) is not an active-set change.** It occurs where
  the corner persists, and it reflects genuine higher-order terms.
- **Decomposition** (`corner_tracking_decomposition.csv`): with α = A_ε/A* − 1 and κ = K(ε)/K − 1,
  R/R∞ − 1 = α + κ + ακ. At a = 1.60 the excess over the first-order line c₁ε is −0.038. It is the sum of
  three terms:

  | term | size at a = 1.60 |
  |---|---:|
  | product term ακ (its leading part a₁k₁ε² has certified coefficient 0.662 × (−0.377) = −0.250) | −0.060 |
  | the switch position's own curvature (α − a₁ε) | −0.037 |
  | the gap maximiser's own curvature (κ − k₁ε, positive) | +0.059 |

  The same signs hold at all six a. These use the sharp A* (certified once (c)'s chain closed) and are
  exploratory.

## 10. Small- and large-scale limits of the conditional minimiser (lemma; any width)

**Setting.**
- The hidden output is φ_θ(x): at width 1, f_a(w₁x + b₁); at width 2, Σᵢ ṽᵢu(αᵢx + βᵢ) with ‖ṽ‖₁ = 1.
- The logits are z = s·φ_θ(x) + b, with s = ‖w₂‖₁ > 0.
- The data have balanced classes: n_O = n_I = n/2 and ȳ = ½. This holds for the population and for every training
  set.
- L*(θ; s) = min_b mean ℓ(z, y), where ℓ is the logistic loss.
- μ_O(θ) and μ_I(θ) are the means of φ_θ over the class-1 (outer) and class-0 (inner) points. Δμ(θ) = μ_O − μ_I.
- Var(φ_θ) is the population variance of φ_θ over all n points.

**Lemma 1 (small s).** L*(θ; s) = log 2 − (s/4)·Δμ(θ) + (s²/8)·Var(φ_θ) + R(θ; s). Here |R| ≤ K·s⁴·m₄(θ), with
m₄ = mean |φ_θ − mean φ_θ|⁴ and K an absolute constant. The s³ term vanishes.
- *Proof.* g(s) = L*(θ; s) is smooth, because b*(s) is the unique root of mean σ(z) = ȳ, and b* is smooth by the
  implicit-function theorem.
  1. At s = 0 the logits are constant, so b*(0) = logit(ȳ) = 0 and g(0) = log 2.
  2. By the envelope theorem, g′(s) = mean((σ(z*) − y)·φ). At s = 0, σ = ½, so
     g′(0) = mean((½ − y)φ) = ½·½(μ_O + μ_I) − ½μ_O = −Δμ/4.
  3. Next, g″(s) = mean(σ′(z*)(φ + b*′)φ). Differentiating mean σ(z*) = ȳ gives b*′ = −mean(σ′φ)/mean(σ′).
     At s = 0, σ′ ≡ ¼, so b*′(0) = −mean φ and g″(0) = ¼·mean((φ − mean φ)φ) = ¼·Var(φ).
  4. For the third derivative, write z* = s·(φ − mean φ) + o(s) at s → 0. The integrand involves σ″(z*), which
     is odd at z = 0, so g‴(0) = 0.
  5. The fourth derivative is bounded by K·m₄, using |σ⁽ᵏ⁾| ≤ 1 and the same implicit differentiation. Taylor's
     theorem with remainder then gives the result. ∎
- *Numerical check* (random width-2 f_a configurations): the remainder is 1e−9 at s = 1e−2 and 1e−13 at s = 1e−3,
  scaling as s⁴.

**Corollary 1 (s → 0).** Let θ*(s) be conditional minimisers staying in a compact set as s → 0. Then:
- (i) every limit point maximises Δμ.
- (ii) If Δμ has quadratic growth away from its maximiser set M, the limit points minimise Var(φ_θ) over M.
- *Proof.* L* = log 2 − (s/4)[Δμ − (s/2)Var(φ) + O(s³)]. For (i), a point with Δμ below its supremum by δ loses to
  a maximiser once s·Var < δ. For (ii), inside a neighbourhood of M, moving a distance d away from M costs about
  c·d² in Δμ and gains at most C·s·d in Var, so the optimum stays within O(s) of M. On M itself, the next-order
  term −(s²/8)·(−Var) selects the minimum Var. ∎

**Corollary 2 (a placement threshold needs an unplaced small-scale limit).** Suppose every Var-minimising
Δμ-maximiser has G > 0. Then for all sufficiently small s the conditional minimiser is placed, and there is no
threshold below which it is unplaced. A threshold, in the sense of unplaced below and placed above, therefore
requires the selected Δμ-maximiser to have G ≤ 0.

**Lemma 2 (s → ∞).** Here G_n(θ) is the gap over the **data points**: the minimum of φ over the class-1 points minus
the maximum over the class-0 points. Let Γ_n = sup_θ G_n(θ) > 0, and suppose it is attained at some θ_Γ.
- (a) L*(θ_Γ; s) ≤ log(1 + e^{−sΓ_n/2}).
- (b) For every θ, L*(θ; s) ≥ (1/n)·log(1 + e^{−sG_n(θ)/2}).
- *Proof.* (a) Place b at the midpoint of the gap: every point then has margin at least sΓ_n/2. (b) For any b, the
  extreme class-1 and class-0 points cannot both have margin above sG_n(θ)/2. The one that doesn't contributes at
  least (1/n)·softplus(−sG_n/2). ∎
- **Consequence**: a θ with G_n(θ) ≤ Γ_n − δ cannot be a conditional minimiser once (1/n)·e^{−s(Γ_n−δ)/2}/2 >
  e^{−sΓ_n/2}, i.e. once s > (2/δ)·log(2n). So limit points of the minimisers, if they stay in a compact set,
  maximise the **data** gap G_n.
- G_n ≥ G, where G is the gap over the continuous windows. They differ by at most the Lipschitz constant of φ times
  the data's gap from the window edges; for the 800-point population that spacing is 0.004.
- **The condition matters.** If Γ_n is not attained, the minimisers need not converge. That is tanh at width 2:
  α diverges and the conditional infimum is not attained.

**Structure of the Δμ-maximisers (f_a).**
- The data are symmetric about 0 in both classes, so the linear part of f_a contributes nothing to Δμ. For one unit,
  Δμ = a·sin β·D(α), with D(α) = mean_O cos(αx) − mean_I cos(αx) over the data points.
- **Width 1**:
  - M = {w₁ = ±α*, sin b₁ = sign D(α*)}, with α* = argmax_{α>0} |D(α)|. α* does not depend on a.
  - Var is the same on all of M; the members are mirror images.
- **Width 2 (‖ṽ‖₁ = 1)**:
  - Δμ ≤ a·maxᵢ|D(αᵢ)| ≤ a·|D(α*)|, so M = {every unit with nonzero weight has αᵢ = ±α* and ṽᵢ sin βᵢ D(αᵢ) =
    |ṽᵢ||D(α*)|}.
  - On M, φ = c·x + const + a·sign(D(α*))·cos(α*x), with c = Σṽᵢαᵢ. Cov(x, cos α*x) = 0 by symmetry, so
    Var(φ) = c²Var(x) + a²Var(cos α*x).
  - The Var-minimising member has **c = 0**, which needs two units with ṽ₁α₁ = −ṽ₂α₂: the cosine pair. It is
    unavailable at width 1.

### 10.1 The second-order selection, in full (added at the author's request, 2026-09-24)

**The expansion, with every constant.**
- Let g(s) = L*(θ; s). The data are balanced (ȳ = ½), and b*(s) is the unique root of m(s, b) = mean σ(sφ + b) − ½.
- **g(0) = log 2.** At s = 0, b* = 0.
- **g′(0) = −Δμ/4.** By the envelope theorem, g′(s) = mean((σ(z*) − y)·φ), where ∂L/∂b = 0 at b*. At s = 0 this is
  mean((½ − y)φ) = ¼(μ_O + μ_I) − ½μ_O = −Δμ/4.
- **g″(0) = ¼·Var(φ).** g″(s) = mean(σ′(z*)·(φ + b*′)·φ).
  - Differentiating m(s, b*(s)) = 0 gives mean(σ′(z*)(φ + b*′)) = 0, so b*′ = −mean(σ′φ)/mean(σ′).
  - At s = 0, σ′ ≡ ¼, so b*′(0) = −φ̄ and g″(0) = ¼·mean((φ − φ̄)φ) = ¼·Var(φ).
- **g‴(0) = 0.** g‴(s) = mean(σ″(z*)(φ + b*′)²φ) + mean(σ′(z*)·b*″·φ).
  - Differentiating the identity mean(σ′(z*)(φ + b*′)) = 0 once more gives
    mean(σ″(z*)(φ + b*′)²) + b*″·mean(σ′(z*)) = 0.
  - At s = 0, z* ≡ 0 and σ″(0) = 0. So b*″(0) = 0, and both terms of g‴(0) vanish.
- **Hence L*(θ; s) = log 2 − (s/4)·Δμ(θ) + (s²/8)·Var(φ_θ) + O(s⁴).**
  - The remainder is bounded by K·s⁴·mean|φ − φ̄|⁴, with |σ⁽ᵏ⁾| ≤ 1.
  - Checked numerically: 1e−9 at s = 1e−2 and 1e−13 at s = 1e−3.

**Why the variance term favours the cancelling pair (f_a, width 2).**
- On the first-order maximiser set M, every unit with nonzero weight has α = ±α* and sin β = ±1, aligned so that
  the oscillatory part of φ is a·sign(D*)·cos(α*x) in every case. Δμ is the same on all of M.
- The members differ only in the **linear part** c·x, with c = Σṽᵢαᵢ.
  - The linear part contributes nothing to Δμ, because both classes are symmetric about 0.
  - It adds c²·Var(x) to Var(φ), because Cov(x, cos α*x) = 0 by the same symmetry.
- So on M, Var(φ) = c²·Var(x) + a²·Var(cos α*x). The s² term penalises the linear part and nothing else.
- The minimum is at c = 0. That needs two units with ṽ₁α₁ = −ṽ₂α₂: the cancelling pair φ = const + a·sign(D*)·cos(α*x).
  - A single unit has c = ±α* ≠ 0 and so has strictly larger Var.
  - At width 1 no cancellation is possible.
- **The second-order gap between the single unit and the pair is (s²/8)·α*²·Var(x).** It is small, which is why a
  finite restart budget at very small s can retain a single unit; the validated search's independent check exists
  for that reason.

**The criterion.** Suppose Γ_n > 0 is attained (Lemma 2), so the minimisers are placed at large s. Then **output
scale gates placement, meaning the conditional minimiser is unplaced below some scale and placed above it, if and
only if the small-scale conditional minimiser is unplaced.** When the first-order maximisers tie, "the small-scale
conditional minimiser" means the one the second-order term selects.
- *If*: unplaced at small s and placed at large s implies a sign change of G along the minimisers.
- *Only if*: Corollary 2.
- The criterion concerns the conditional minimiser, the object of the paper's threshold. Whether training reaches
  it is a separate question, tested by W4 at width 1 and by the fixed-scale design below at width 2.

**The three cases, and how each was established.**

| case | small-scale conditional minimiser | gating | evidence |
|---|---|---|---|
| width 1, f_a (a = 1.30–1.60) | unplaced (G = −3.66 … −3.35; `scale_limits_width1.csv`) | **yes**: certified thresholds | registered consistency check, passed |
| width 2, f_a (a = 1.30, 1.50) | placed: the cancelling pair (G = +0.890, +1.026) | **none** for the conditional minimiser; training at fixed small scale is nevertheless gated (Track 7, registered: WP-20) | registered prediction (`scale_limits_prediction.md`); decided by the second-order selection; direct check at finite small scale below |
| width 2, tanh | supremum Δμ = 1 **not attained** (proved: Δμ < 1 at every finite parameter); the Var-selected near-maximisers are the symmetric step pair, placed, with G₊ = 0.762, 0.964, 0.99933 and 0.9999998 at box sizes A = 5, 10, 20 and 40; single steps tie at first order and are unplaced | **none** in the limit sense; outside the criterion's attainment hypothesis | registered (`scale_limits_tanh_prediction.md`); **registered outcome "neither"**: every T1 number held, but the boundary condition failed at A = 40, where 9 of 1,107 near-maximisers inside the 1e−9 tie sit at 0.991–0.999·A (a criterion flaw, disclosed); recorded as "neither" by the author's decision, with no re-registration; non-attainment is proved analytically; earlier exploratory pilot scan (`width2_pilot_scan.csv`) |

## 11. The mechanism, stated analytically (harsh review A1–A2; for the submission, 2026-09-25)

Producer: `src/harsh_review_a.py` → `wp12_identity.csv`, `wp12_mechanism.csv`, `wp12_scaling.csv`,
`wp12_onset_bound.csv`, `wp12_adam_bound.csv`. The setting is §10's: width 1, f_a(t) = t + a sin t, φ_θ = sign(w₂)·f_a(w₁x + b₁),
s = |w₂|, balanced classes, and the 800-point population unless stated otherwise.

**Which gap each step uses.**
- G is the gap over the continuous windows (exact extrema). G_n is the gap over the data points.
- The four points used below, x = −2.0, −0.8, 0.8 and 2.0, are both window endpoints and data points of the
  population (`linspace` includes its ends). So Step 2 holds for G and for G_n alike.
- Step 3 (large s) is a statement about G_n. It transfers to G with the explicit correction η, stated there.

**Step 1: the class-mean gap (identity).** On windows or data symmetric about 0,
  Δμ(θ) = sign(w₂)·a·sin b₁·D(w₁),   D(α) = E_O cos αx − E_I cos αx.
- *Proof.* The linear part of f_a contributes w₁(E_O x − E_I x) + b₁ − b₁ = 0. For the sine part, sin(w₁x + b₁) =
  sin(w₁x)cos b₁ + cos(w₁x)sin b₁, and E sin(w₁x) = 0 in each class by symmetry. ∎
- On the continuous windows, D_c(α) = [sin 2α − sin 1.2α]/(0.8α) − sin(0.8α)/(0.8α).
- *Check.* Over 2,000 random (w₁, b₁) with |w₁| ≤ 10, at a = 1.30 and 1.50, the largest error was 3.3e−15 on the data
  points and 2.3e−14 on the continuous windows (Gauss–Legendre, 200 nodes per window, against the closed form).
- The maximiser of |D| is α* = 1.7913244 on the data points (the ledger value) and 1.7922917 on the continuous
  windows. |D*| = 1.571067 on the data points and 1.573242 on the continuous windows.
- α* does not depend on a, and the maximising bias is sin b₁ = ±1.

**Step 2: the domain lemma (placement needs a small first-layer weight).** For every θ,
  G(θ), G_n(θ) ≤ 2a − 2.8|w₁|, and more sharply ≤ 2a|sin(1.4w₁)| − 2.8|w₁|.
- *Proof.* Take w₁ > 0; w₁ < 0 is the mirror image.
  - G₊ ≤ φ(−2.0) − φ(0.8) = −2.8w₁ + a[sin(−2w₁ + b₁) − sin(0.8w₁ + b₁)].
  - G₋ ≤ φ(−0.8) − φ(2.0) = −2.8w₁ + a[sin(−0.8w₁ + b₁) − sin(2w₁ + b₁)].
  - Each bracket is at most 2|sin(1.4w₁)| ≤ 2 in absolute value, because |sin u − sin v| = 2|cos((u + v)/2)||sin((u − v)/2)|. ∎
- **Hence G > 0 or G_n > 0 requires |w₁| < a/1.4.** This is the same bound as the Ĝ branch-and-bound domain
  (`ghat_bnb.py`).
- Because α* is independent of a, the class-mean maximiser (|w₁| = α*) is unplaced for every a < 1.4·α* = 2.50785.
- At the maximiser itself, G = G_n = −3.66 at a = 1.30 and −3.45 at a = 1.50. Both lie below the sharp pair bound
  (−3.48 and −3.24).

**Proposition (width 1: an explicit analytic bracket for the placement switch).** Let 1 < a < 1.4α*. Define:
- D_P(a) = max over |α| ≤ a/1.4 of |D(α)|;
- θ* = (w₁, b₁) = (α*, sign(D*)·π/2) with w₂ > 0;
- s₀(a) = 2(Δμ(θ*) − a·D_P)/Var(φ_θ*);
- s₁(a) = 2·log(n/log 2)/(Ĝ(a) − η), with η = 0 for data placement and η = (a/1.4)(1 + a)(h_I + h_O) for window
  placement, where h_I and h_O are the grid half-spacings.

Then:
- (i) for every s < s₀, every placed θ (G > 0 or G_n > 0) has L*(θ; s) > L*(θ*; s), and θ* is unplaced;
- (ii) for every s > s₁, every θ with G_n(θ) ≤ η has a loss above that of Ĝ's witness, which is placed.

So no placed parameter comes within the (i) margin of the conditional infimum below s₀, and no unplaced one comes
within the (ii) margin above s₁. The placement switch lies in [s₀, s₁]. This needs no compactness and no attainment.

*Proof.*
- (i) The logistic loss is convex in z, so ℓ(z, y) ≥ log 2 + (½ − y)z. Averaging over balanced data, for every b,
  mean ℓ(sφ + b) ≥ log 2 − (s/4)Δμ(θ). Hence L*(θ; s) ≥ log 2 − (s/4)Δμ(θ) for every θ, with no remainder.
  - A placed θ has |w₁| < a/1.4 (Step 2), so Δμ(θ) ≤ a·D_P (Step 1).
  - For θ*, ℓ″ ≤ ¼ gives ℓ(z, y) ≤ log 2 + (½ − y)z + z²/8. Take b = −s·mean φ*. Then
    L*(θ*; s) ≤ log 2 − (s/4)Δμ(θ*) + (s²/8)Var(φ*).
  - Comparing the two bounds gives (i) whenever s < s₀. θ* is unplaced by Step 2.
- (ii) Lemma 2(b) of §10 with G_n(θ) ≤ η, and log(1 + e^{−u}) ≥ log 2·e^{−u} for u ≥ 0, give
  L*(θ; s) ≥ (log 2/n)·e^{−sη/2}.
  - For the witness, G_n ≥ G ≥ Ĝ_cert, so L* ≤ log(1 + e^{−sĜ/2}) < e^{−sĜ/2}.
  - Comparing the two gives (ii).
  - For window placement: if G(θ) ≤ 0 and |w₁| < a/1.4, then G_n(θ) ≤ G(θ) + Lip(φ)(h_I + h_O) ≤ η, since
    Lip(φ) ≤ |w₁|(1 + a). If |w₁| ≥ a/1.4, then G_n ≤ 0 ≤ η by Step 2. ∎

**The constants are four finite evaluations:** D*, D_P (a grid of step 1e−5 plus the Lipschitz step
(E_O|x| + E_I|x|)·h/2), Var(φ*), and Ĝ's rigorous witness value (`ghat_rigorous.csv`).

| a | s₀ | s₁ (data) | certified |w₂|_glob | R₀ = s₀Ĝ/2 | R₁ = log(n/log 2) |
|---|---|---|---|---|---|
| 1.30 | 0.342 | 163.8 | [4.950, 4.9625] | 0.0147 | 7.05 |
| 1.50 | 0.274 | 80.5 | [2.525, 2.5375] | 0.0240 | 7.05 |

- Every certified bracket (a = 1.30–1.60) lies inside [s₀, s₁] (`wp12_mechanism.csv`, `bracket_contains_cert`).
- The window-placement s₁ is defined wherever Ĝ > η, which holds at every tabulated a except 1.02.
- **The bracket is loose, by roughly 15× below and 30× above.** Its point is that the switch's *existence* is
  analytic, not that its location is.
- *Sanity check at s₀* (a = 1.30): L*(θ*; s₀) = 0.59659 lies below the placed lower bound, 0.60119.

**Step 3: what the loss rewards, and why the threshold is a change of objective.**
- The two bounds in the proof are the two regimes.
- For small s, L* = log 2 − (s/4)Δμ + O(s²), so the loss rewards the **class-mean gap**. Its maximiser uses the
  ramp-free cosine with |w₁| = α*, which Step 2 says is unplaced.
- For large s, the loss is controlled by e^{−sG_n/2}, so it rewards the **worst-case gap**, whose maximiser is placed.
- R_glob is where the global conditional minimiser switches from the first kind of solution to the second. The
  certified thresholds locate this switch; the proposition proves it exists.

**Step 4: width 2 in the same terms (§10.1).**
- On symmetric windows the ramp contributes nothing to Δμ at either width.
- With two units, the second-order term selects the pair with ṽ₁α₁ = −ṽ₂α₂. It cancels the ramp and leaves
  φ = const + a·sign(D*)·cos(α*x).
- That pair is placed: G = +0.890 at a = 1.30 and +1.026 at a = 1.50. So the class-mean maximiser is already
  placed, and no switch is needed (the registered width-2 verdict).
- The criterion predicts the opposite when the windows are asymmetric: there the ramp enters Δμ through
  m = E_O x − E_I x ≠ 0.
  - Registered and tested (Track 2, `asym_registration.md`): at a = 1.30, Δ = 0.4, a validated width-2 switch exists at
    s ∈ [0.4371, 0.4532] (T2-1 PASS).
  - Width-2 training crosses above it, but at a median 3.29× the threshold scale, so the registered training
    prediction fails (T2-3 FAIL; WP-15).

### 11.1 The scaling law and Adam's per-step bound (A2)

**Required scale.**
- From A = sε^{3/2} (§1) and the switch A_ε = A*(1 + (A′(0)/A*)ε + O(ε²)) (§8):
  |w₂|_glob(a) = A*·ε^{−3/2}·(1 + 0.66215ε + O(ε²)), with ε = a − 1.
- A* ∈ [0.68125, 0.6875] is certified and independently checked; the Krawczyk value is 0.6854452.
  A′(0)/A* ∈ [0.6621547, 0.6621550].
- (In R units the correction is c₁ = 0.28523. The difference is k₁ = −0.37692, from K(ε).)

| a | limit A*ε^{−3/2} | with first-order term | certified |w₂|_glob | first-order − certified (rel.) |
|---|---|---|---|---|
| 1.02 | 242.3 | 245.6 | — | — |
| 1.05 | 61.3 | 63.3 | — | — |
| 1.10 | 21.7 | 23.1 | — | — |
| 1.30 | 4.171 | 5.000 | [4.9500, 4.9625] | +0.9% |
| 1.40 | 2.709 | 3.427 | [3.3750, 3.3875] | +1.4% |
| 1.50 | 1.939 | 2.581 | [2.5250, 2.5375] | +1.9% |
| 1.60 | 1.475 | 2.061 | [2.0000, 2.0125] | +2.7% |

- The first-order law is within 3% of every certified finite-a threshold at ε ≤ 0.6.
- The error grows with ε, as an O(ε²) remainder should.
- Without the first-order term the limit value is 16–27% low at these ε.

**Adam's per-step bound (rigorous, for every gradient sequence).**
- The paper's training uses torch Adam with lr = 0.01, (β₁, β₂) = (0.9, 0.999), ε_Adam = 1e−8, full batch, no weight
  decay, and θ(0) ~ U(−1, 1)⁴.
- The update is Δθᵢ = −lr·m̂_t/(√v̂_t + ε_Adam), with m̂_t = m_t/(1 − β₁ᵗ) and v̂_t = v_t/(1 − β₂ᵗ).
- By Cauchy–Schwarz, |Σ_k β₁^{t−k}g_k| ≤ (Σ_k γ^{t−k})^{1/2}(Σ_k β₂^{t−k}g_k²)^{1/2}, with γ = β₁²/β₂. So
  |Δθᵢ| ≤ lr·B_t,   B_t = (1 − β₁)/(1 − β₁ᵗ)·√((1 − β₂ᵗ)/(1 − β₂))·√((1 − γᵗ)/(1 − γ)).
- Also B_t ≤ B_∞ = (1 − β₁)/√((1 − β₂)(1 − γ)) = 7.2703.
- The bound is **attained**: g_k ∝ (β₁/β₂)^{t−k} gives equality (checked at t = 400: 4.17529 both ways). ε_Adam only
  lowers the step.
- Summing, |w₂(T)| ≤ 1 + lr·Σ_{t≤T} B_t. That is **106.93 after 2,000 steps**, because the bias correction keeps B_t
  small early on (max over t ≤ 2,000 is 6.76).

**Consequence at a = 1.02.**
- The required |w₂| is 245.6: 242.3 in the limit, and at least 240.9 even with A* at its certified lower end.
  Any gradient sequence needs **at least 3,966 steps** to reach it.
- So within the paper's 2,000 steps, **no Adam run can reach the conditional threshold scale at a = 1.02**. This is
  a hard constraint, not a fitted one. It would still hold if the true threshold were 56% below the prediction.
- At the typical Adam rate (|m̂/√v̂| ≈ 1 for a consistent gradient sign) the requirement is about 24,000 steps.
- **Observed** (`fold1d_sweep.csv`): 0/200 solves at a = 1.02 within 2,000 steps. The terminal |w₂| has median 1.85
  and maximum 3.92, far below both the worst-case reachable 106.9 and the required 245.6.

**What this law explains and what it does not.**
- *Explains:*
  - why the required scale diverges as ε^{−3/2}, with its constant A* and first-order correction certified;
  - that with a bounded per-step move, the steps needed must diverge at least as fast, N_min ≳ (A*/(lr·B_∞))·ε^{−3/2}.
    This is why training near a = 1 is delayed.
  - A solve before the threshold scale is reached would need the trajectory to be placed below the conditional
    threshold, which the paper's crossing data do not show at this budget.
  - The 0/200 at a = 1.02 is therefore **consistent with, and forced by,** the threshold-crossing picture. It is
    not independent evidence for that picture.
- *Does not explain:*
  - **the actual growth rate.** The worst-case bound is loose by one to two orders of magnitude. At B = 2,000 it
    allows every a ≥ 1.035 (the lr-per-step rate allows a ≥ 1.107), but the observed onset is a = 1.60. At B = 128,000
    it allows a ≥ 1.0018; observed 1.03 (`wp12_onset_bound.csv`).
  - The observed onset follows the measured growth |w₂| ∝ B^α with α = 1.1172. α is a measured quantity: its SGD
    derivation failed (`alpha_derivation_results.md`). The onset exponent −2α/3 = −0.745 (registered) against the
    measured −0.734 therefore tests the *combination* of the derived ε^{−3/2} and the measured α, not a derivation
    of α.
  - The bound also says α ≤ 1 asymptotically. Over the budgets tested the measured |w₂| stays far below the bound's
    line, so this is no tension.
  - It also does not explain the residual, meaning why crossings sit above the conditional threshold.


## 12. H1 proved for small ε (Track 8; 2026-09-25)

**Status: PROVED for 0 < ε ≤ 0.029.** Every step below was checked by hand on 2026-09-25:
- Lemma 1, re-derived from the sum-to-product formula;
- the pairs, orders and d, m values in Lemma 2(a)–(e);
- the Taylor bounds on cos and sin/x, valid for all real arguments;
- the product of two positive inequalities;
- the Cauchy–Schwarz constant 3.10204;
- the two-pass bound x_a² ≤ 6ε, valid up to ε = 0.6883;
- β² ≤ ε(2 + ε), via arctan z ≤ z;
- the remainder of Lemma 4, re-derived from f_a(π + s) = π + s − a sin s;
- the four rational values of Lemma 5;
- K_low(0.029) = 0.4062;
- the convergence constant, sup|r| on |σ| ≤ 5.295 being 9.946.

The numerical sanity checks (`src/h1_checks.py`, output `h1_checks.log`) are not part of the proof.

**Consequences for this note.**
- The corollary of §2 is asymptotic ("for all sufficiently small ε"). H1 (localisation of the gap maximiser) is no
  longer a hypothesis for it: it holds for ε ≤ 0.029 with explicit constants.
- **What remains certified, not proved:** that the maximiser is the tied corner. That needs uniqueness, up to
  symmetry, of G₀'s maximiser, which K's branch and bound certifies. It is independently checked in Arb (Track 5,
  `certificate_checks/K_base*.json`).
- H1 at a ≥ 1.30 is not covered by this argument (§12.8).

### 12.0 The statement being proved

`scaling_proposition.md`, Hypotheses:

> **H1 (localisation).** For ε small, Ĝ(a) is attained at a placement whose limit coordinates lie in C, up to the
> symmetries t ↦ t + 2π (f_a(t + 2π) = f_a(t) + 2π) and σ ↦ −σ, x ↦ −x.
> C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}. Limit coordinates: w₁x + b₁ = π + √ε σ(x), σ(x) = ux + v.

Here (u, v) = (p, q) in the notation of `math_note_v2.md` §1: w₁ = √ε p, b₁ = π + √ε q. H1 is used in step 4 of
that note ("Taking sups over C (H1 for the left side)"). In `math_note_v2.md` it enters only through
K(ε) := Ĝ(a)/ε^{3/2} → K (§1 "Units", §8 bookkeeping).

**Setting.**
- f_a(t) = t + a sin t, a = 1 + ε with ε > 0. The windows are I = [−0.8, 0.8] and O = [−2, −1.2] ∪ [1.2, 2].
- φ(x) = f_a(w₁x + b₁).
- G(w₁, b₁) = max(G₁, G₂), with G₁ = min_O φ − max_I φ and G₂ = min_I φ − max_O φ.
- Ĝ(a) = sup G.
- Rescaled gap: G̃_ε(p, q) = G(√ε p, π + √ε q)/ε^{3/2}. This is the gap of φ_ε∘σ, with
  φ_ε(σ) = [f_a(π + √εσ) − π]/ε^{3/2}.
- G₀ is the same functional with h(σ) = −σ + σ³/6 in place of φ_ε, and K = sup G₀.

### 12.1. Theorem

**Theorem.** Let 0 < ε ≤ 0.029. Then:
1. Ĝ(a) > 0, and Ĝ(a) is attained.
2. Every maximiser (w₁, b₁) of G has w₁ ≠ 0. Using (w₁, b₁) ↦ (−w₁, −b₁) and b₁ ↦ b₁ + 2πk, it has a representative
   whose rescaled coordinates satisfy

   **1.0155 ≤ K(ε)/0.4 ≤ p < √6/1.4 = 1.74964,   |q| < 1.7956.**

In particular (p, q) ∈ C, so H1 holds as stated in `scaling_proposition.md`, with the explicit ε₀ = 0.029.

**Corollary (H1 in the "fixed compact" form, larger ε).**
- For every ε ∈ (0, 0.6878] with Ĝ(a) > 0, every maximiser has a representative with 0 < p < 1.74964 and
  |q| < √(2 + ε) + 0.6p < 2.690.
- Ĝ(a) > 0 is proved here for ε < 0.0859 (Lemma 5).

**Corollary (convergence, replacing H1 in step 4).** For 0 < ε ≤ 0.029,
  |K(ε) − K| ≤ 2ε[sup_{|σ|≤S}|r| + ε(S⁵/120 + (1+ε)S⁷/5040)] ≤ 23.3 ε,   with S = 5.295, r(σ) = σ³/6 − σ⁵/120.

**Limit version (ε = 0; sharpens the domain lemma of `math_note_v2.md` §8).**
- G₀(p, q) ≤ 0.4|p|.
- If G₀(p, q) > 0 (with p > 0), then (q − 0.6p)² + (2.8p)²/12 < 2 or (q + 0.6p)² + (2.8p)²/12 < 2.
- Hence G₀ > 0 only for |p| < √24/2.8 = 1.74964 and |q| < 1.76126. The note's lemma gives 4.0825 and 3.8637.
- The argmax of G₀ has p ≥ K/0.4 ≥ 1.4447.

### 12.2. Lemma 0: existence and symmetries

- **Continuity.** φ is jointly continuous in (x, w₁, b₁), and the windows are compact. So min_O φ, max_I φ, etc. are
  continuous, and G is continuous on ℝ².
- **Symmetries.** f_a(t + 2π) = f_a(t) + 2π, so b₁ ↦ b₁ + 2π shifts φ by a constant and leaves G unchanged.
  f_a is odd, so (w₁, b₁) ↦ (−w₁, −b₁) maps φ to −φ. This swaps G₁ and G₂ and leaves G unchanged.
- **w₁ = 0.** φ is constant and G = 0.
- **Attainment.** By Lemma 2(b), G ≤ 0 whenever 1.4|w₁| ≥ x_a. So when Ĝ(a) > 0 the supremum is a supremum over the
  compact set |w₁| ≤ x_a/1.4, b₁ ∈ [0, 2π], and it is attained. Every maximiser has G = Ĝ > 0, so w₁ ≠ 0. Take
  w₁ > 0 from now on.

### 12.3. Lemma 1: the pair identity (exact)

For t_o < t_i, with d = t_i − t_o and m = (t_o + t_i)/2 − π:

  **f_a(t_o) − f_a(t_i) = 2a cos(m) sin(d/2) − d.**

*Proof.*
- f_a(t_o) − f_a(t_i) = −d + a(sin t_o − sin t_i) = −d + 2a cos((t_o + t_i)/2)·sin(−d/2).
- cos((t_o + t_i)/2) = cos(π + m) = −cos m.
- So the difference is −d + 2a cos(m) sin(d/2). ∎

In rescaled units (t = π + √εσ, d = √εδ, m = √εμ) this is the exact analogue of the cubic identity
h(α) − h(α + δ) = δ[1 − ((α + δ/2)² + δ²/12)/2] of `math_note_v2.md` §8. Here α + δ/2 = μ, and the cubic identity
is its ε → 0 limit.

### 12.4. Lemma 2: the finite-ε domain lemma (every ε with 1 < a < π, w₁ > 0)

Let x_a be the unique root in (0, π) of sin x = x/a, and let β = arccos(1/a) ∈ (0, π/2). Write y = 1.4w₁.

**(a) G ≤ 0.4 ε w₁.** Equivalently, G̃_ε(p, q) ≤ 0.4p for every ε.
- G₁ ≤ φ(−1.2) − φ(−0.8), because a minimum is at most any member and a maximum at least any member.
- By Lemma 1 with d = 0.4w₁, this is 2a cos(m) sin(0.2w₁) − 0.4w₁ ≤ 2a|sin(0.2w₁)| − 0.4w₁ ≤ 0.4(a − 1)w₁.
- G₂ ≤ φ(0.8) − φ(1.2) gives the same bound.
- If G ≤ 0 the claim is trivial.

**(b) G > 0 ⇒ 0 < y < x_a (< π).**
- G₁ ≤ φ(−2) − φ(0.8). With d = 2.8w₁ = 2y, this is 2a cos(m₁) sin(y) − 2y ≤ 2a|sin y| − 2y, where
  m₁ = b₁ − 0.6w₁ − π.
- G₂ ≤ φ(−0.8) − φ(2) = 2a cos(m₂) sin(y) − 2y, where m₂ = b₁ + 0.6w₁ − π.
- So G > 0 ⇒ |sin y|/y > 1/a.
- For y ≥ π, |sin y|/y ≤ 1/π < 1/a. On (0, π), sin y/y is strictly decreasing. Hence y < x_a.

**(c) If G_j > 0, then cos m_j > 1/a.** So after b₁ ↦ b₁ − 2πk we may take |m_j| < β.
- By (b), 0 < sin y < y.
- G_j > 0 gives 2a cos(m_j) sin y > 2y > 0. So cos m_j > y/(a sin y) > 1/a.
- cos m > 1/a exactly when m ∈ (2πk − β, 2πk + β) for some k.

**(d) The ellipse.** Assume in addition ε ≤ 0.6878 (so that x_a² ≤ 6ε, Lemma 3). If G_j > 0 and |m_j| < β, then in
rescaled units μ_j = m_j/√ε:

  μ_j²/2 + c′p² < Λ(ε),   with c′ = 1.96/6,  Λ(ε) = 1/[(1 + ε)·min(κ₁, κ₃)],
  κ₁ = 1 − ε(2 + ε)/12,   κ₃ = (1 − 0.3ε)(1 − ε(2 + ε)/2).

*Proof.*
1. Use cos m ≤ 1 − (m²/2)(1 − m²/12) and sin y/y ≤ 1 − (y²/6)(1 − y²/20). Both are Taylor bounds with alternating
   remainder and hold for all real m and all y > 0.
2. By (b), (c) and Lemma 3, m² < β² ≤ ε(2 + ε) and y² < x_a² ≤ 6ε. So 1 − m²/12 ≥ κ₁ and 1 − y²/20 ≥ 1 − 0.3ε.
3. Each right side is at least the positive quantity it bounds. So
   1/a < cos m · sin y/y ≤ (1 − A)(1 − B), with A = κ₁m²/2 and B = (1 − 0.3ε)y²/6.
4. (1 − A)(1 − B) = 1 − A − B + AB, and AB ≤ B·m²/2 ≤ B·ε(2 + ε)/2.
5. So A + B(1 − ε(2 + ε)/2) < 1 − 1/a = ε/(1 + ε).
6. Divide by ε and use m² = εμ² and y² = 1.96εp². This gives
   κ₁μ²/2 + κ₃c′p² < 1/(1 + ε), which implies the stated inequality. ∎

**(e) q from μ.**
- q = μ₁ + 0.6p (from m₁) or q = μ₂ − 0.6p (from m₂).
- By Cauchy–Schwarz, |μ| + 0.6p ≤ √(2 + 0.36/c′)·√(μ²/2 + c′p²).
- Hence |q| < √(3.10204·Λ(ε)). Also, directly from (c) and Lemma 3, |q| < √(2 + ε) + 0.6p.

**Scope.** Only the points x = −2, −1.2, −0.8, 0.8, 1.2, 2 enter (a)–(e). These are window endpoints and also points
of the 800-point population data (`linspace` includes its ends). So Lemma 2 holds verbatim for the data gap G_n.
It does not hold for a random 400-point training set, whose points need not include the window endpoints.

### 12.5. Lemma 3: explicit bounds on x_a and β

**(i) x_a² ≤ 6ε for 0 < ε ≤ 0.6878.**
1. At x = x_a, 1/a = sin x/x ≤ 1 − (x²/6)(1 − x²/20).
2. So (x_a²/6)(1 − x_a²/20) ≤ ε/(1 + ε).
3. First pass: x_a < π gives 1 − x_a²/20 > 1 − π²/20 = 0.50652. So x_a² < 11.846ε, and hence
   1 − x_a²/20 > 1 − 0.5923ε.
4. Second pass: x_a² ≤ 6ε/[(1 + ε)(1 − 0.5923ε)].
5. The denominator is ≥ 1 if and only if ε(0.4077 − 0.5923ε) ≥ 0, that is, ε ≤ 0.6883. ∎

Hence G > 0 ⇒ p = y/(1.4√ε) < √6/1.4 = √24/2.8 = 1.74964. This is the ε-uniform bound, and it equals the limit
threshold exactly.

**(ii) β² ≤ ε(2 + ε) for all ε > 0.** Since tan β = √(a² − 1), β = arctan √(a² − 1) ≤ √(a² − 1) = √(ε(2 + ε)).

**Numerical values for ε ≤ 0.029.**
- κ₁ ≥ 0.99510 and κ₃ ≥ (1 − 0.0087)(1 − 0.029·2.029/2) = 0.96214.
- Since (1 + ε) ≥ 1, Λ ≤ 1/0.96214 = 1.03935.
- So |q| < √(3.10204 × 1.03935) = 1.7956. This bound is deliberately crude and monotone-safe.
- The value of Λ at ε = 0.029 itself gives 1.7701.

### 12.6. Lemmas 4–5: sup-norm remainder, and a lower bound on K(ε)

**Lemma 4 (remainder).**
- With s = √εσ and the Lagrange form sin s = s − s³/6 + s⁵/120 + R₇, |R₇| ≤ |s|⁷/5040 (all real s):

  φ_ε(σ) − h(σ) = ε[σ³/6 − (1 + ε)σ⁵/120] − aR₇/ε^{3/2},   |φ_ε − h| ≤ ε|σ³/6 − (1+ε)σ⁵/120| + (1+ε)ε²|σ|⁷/5040.

  This is the same remainder as step 2 of `scaling_proposition.md`.
- G is **2-Lipschitz** in the sup-norm of the profile on the windows. Each of G₁ and G₂ is a min minus a max, so each
  moves by at most 2δ, and so does their maximum.
- So |G̃_ε(p, q) − G₀(p, q)| ≤ 2·sup{|φ_ε − h|(σ) : σ ∈ σ(I ∪ O)}.

**Lemma 5 (exact value at (1.6, 1.2), and K(ε) ≥ K_low(ε)).**
1. At (p, q) = (8/5, 6/5), in exact rational arithmetic:
   - **I**: σ ∈ [−0.08, 2.48]. The only critical point of h inside is +√2, a local minimum, so the maximum is at an
     endpoint: max(3746/46875, 2914/46875) = 3746/46875.
   - **O₋**: σ ∈ [−2, −0.72]. The only critical point inside is −√2, a local maximum, so the minimum is at an
     endpoint: min(2/3, 10278/15625) = 10278/15625.
   - **O₊**: σ ∈ [3.12, 4.4], where h is increasing; the minimum is 30342/15625.
   - Membership of ±√2 is decided exactly, by comparing squares of rationals with 2.
   - Hence G₀(1.6, 1.2) ≥ G₁ = 10278/15625 − 3746/46875 = **27088/46875 = 0.5778773…**
2. Over all windows, σ ∈ [−2, 4.4]. The function g(σ) = σ³/6 − cσ⁵/120 (c = 1 + ε) is odd.
   - On [0, √(12/c)], 0 ≤ g ≤ (12/c)^{3/2}/15 ≤ 12^{3/2}/15 = 2.7713.
   - Beyond √(12/c), g decreases. For c ≤ 1.2, g(4.4) = 14.197 − 13.743c ≥ −2.296.
   - So sup_{|σ|≤4.4}|g| ≤ 2.7713 for ε ≤ 0.2.
3. By Lemma 4,
   K(ε) ≥ G̃_ε(1.6, 1.2) ≥ K_low(ε) := 27088/46875 − 2ε[2.7713 + (1 + ε)ε·4.4⁷/5040],  with 4.4⁷/5040 = 6.3349.
4. K_low > 0 for ε < 0.0859, and K_low > 0.4 for ε < 0.02998. K_low(0.029) = 0.4062.

### 12.7. Proof of the Theorem and corollaries

**Theorem.** Let 0 < ε ≤ 0.029.
1. By Lemma 5, Ĝ(a) ≥ ε^{3/2}K_low(ε) > 0. By Lemma 0 the supremum is attained, and every maximiser has w₁ ≠ 0.
   Reduce to w₁ > 0.
2. Let j be an orientation with G_j = Ĝ > 0. By Lemma 2(c), reduce b₁ so that |m_j| < β.
3. **Upper bound on p.** Lemma 2(b) and Lemma 3(i) give p < √6/1.4 = 1.74964 ≤ 2.2.
4. **Bound on q.** Lemma 2(d), (e) and the numerical values in Lemma 3 give |q| < 1.7956 ≤ 2.
5. **Lower bound on p.** Lemma 2(a) at the maximiser gives 0.4p ≥ G̃_ε = K(ε) ≥ K_low(ε) ≥ K_low(0.029) = 0.4062.
   K_low is decreasing in ε, so p ≥ 1.0155 ≥ 1. ∎

**Fixed-compact corollary.** For ε ≤ 0.6878 with Ĝ(a) > 0, steps 1–3 and 4 (with the direct bound of Lemma 2(e)) go
through unchanged, and give 0 < p < 1.74964 and |q| < √(2 + ε) + 0.6·1.74964. ∎

**Convergence corollary.**
- Let C′ = [1, 1.7497] × [−1.7956, 1.7956] ⊂ C. On C′, |σ| ≤ 2(1.7497) + 1.7956 = 5.295.
- For ε ≤ 0.029 every maximiser of G̃_ε lies in C′ (the Theorem), so sup_{C′} G̃_ε = K(ε).
- In the limit, Lemma 2(a) and the limit ellipse (§12.1, proved below) put every maximiser of G₀ in C′, since its p is at
  least 0.57788/0.4 = 1.4447. So sup_{C′} G₀ = K.
- Hence |K(ε) − K| ≤ sup_{C′}|G̃_ε − G₀| ≤ 2ε M(S, ε), by Lemma 4.
- With S = 5.295: sup_{|σ|≤S}|r| = 9.943, and the bound is ≤ 23.3ε for ε ≤ 0.029. The measured deviation is
  about 5ε (`scaling_proposition.md` step 4).
- This is exactly step 4 of `scaling_proposition.md` with C′ in place of C. Its own constant (S = 6.4, ≈ 94ε) is also
  valid, since C′ ⊂ C.

**Limit version.**
- Apply the cubic identity to the pairs (−1.2, −0.8) and (0.8, 1.2) (δ = 0.4p): G₀ ≤ δ = 0.4p.
- Apply it to the pairs (−2, 0.8) and (−0.8, 2) (δ = 2.8p, α + δ/2 = q ∓ 0.6p). A positive G_j needs
  (q ∓ 0.6p)² + (2.8p)²/12 < 2.
- That inequality is μ²/2 + c′p² < 1 with μ = q ∓ 0.6p. Cauchy–Schwarz as in Lemma 2(e) gives |q| < √3.10204 = 1.76126. ∎

(The §8 lemma used δ = 1.2p for the p-threshold. The pair (−2, 0.8) with δ = 2.8p is sharper and gives
1.74964 instead of 4.0825. The §8 lemma remains correct.)

### 12.8. What is NOT proved here (honest limits)

1. **The range of ε is small, and it is an artefact of the proof.**
   - ε₀ = 0.029 comes only from the crude lower bound K_low(ε), which uses one test point and a sup-norm remainder.
   - The localisation lemmas themselves hold up to ε ≈ 0.69 (the fixed compact form) and up to ε ≈ 0.29 (the |q| < 2
     bound, where √(3.102Λ(ε)) = 2).
   - For the C-form at a *given* larger ε, the only missing input is K(ε) ≥ 0.4 (for p ≥ 1).
   - At a = 1.02, 1.05, 1.10, 1.15 and 1.25, the rigorous Ĝ lower ends in `ghat_rigorous.csv` give K(ε) = 0.575,
     0.569, 0.559, 0.549 and 0.532. With Lemma 2 this gives the C-form at those a. That step is **certified, not
     proved**: it rests on the Ĝ branch-and-bound values.
   - At a = 1.30 (ε = 0.3) the q-bound of Lemma 2(d) gives 2.0008 > 2. It just fails, so the C-form at a ≥ 1.30 is not
     covered by this argument. The grid check in §12.9 still places the argmax well inside C.
   - Ĝ(a) > 0 for every a > 1 is proved only for ε < 0.0859. Beyond that it rests on the certified values.
2. **H1 localises; it does not identify the maximiser.**
   - The stronger claim in `math_note_v2.md` §8 is that K(ε) = K_loc(ε), that is, the global maximiser is the tied
     corner continued from ε = 0. §8 uses it to compute k₁ as a two-sided derivative of the global K(ε).
   - That claim needs, in addition, uniqueness (up to symmetry) of G₀'s global maximiser. That uniqueness is
     **certified** by the K branch and bound; it is not proved here.
   - Given it, the Theorem plus the convergence corollary put the finite-ε maximiser within o(1) of the corner. §8's
     parametric Krawczyk certificate (unique sharp corner for |ε| ≤ 0.05) then finishes the job.
   - So that chain is "proved + certified", not "proved".
3. **Only the continuous-window gap and the population data gap are covered.**
   - For a random training set, the endpoint pair bounds of Lemma 2 need the extremal data points in place of the
     endpoints. The analogous argument works with slightly different constants, but it is not written out here.
4. **H1 concerns object 1 only** (the gap maximiser). It says nothing about object 3 (the global conditional-loss
   minimiser at finite a). The note correctly leaves that to Block 1c.

### 12.9. Numerical sanity checks (not part of the proof)

`nice -n 15 python -m src.h1_checks`: one process, about 1 minute. The venv has no scipy, so root-finding is by
bisection.

- **(A) The exact value** 27088/46875 = 0.5778773 at (1.6, 1.2). The code asserts the critical-point membership
  pattern (I: +√2 only; O₋: −√2 only; O₊: none). The other orientation is ≤ −9.72 < 0.
- **(B) Lemma 3 bounds.** max x_a²/(6ε) = 1.000000 (→ 1 as ε → 0, never above) and max β²/(ε(2+ε)) = 1.000000, over
  800 values of ε in [1e−8, 0.6878].
- **(C) Constants.**
  - sup_{[0,4.4]}|g| = 2.7713 at ε = 0 and 2.2943 at ε = 0.2, against the bound 2.7713.
  - K_low > 0 for ε < 0.08593, and K_low > 0.4 for ε < 0.02998.
  - Λ(0.029) = 1.01006.
- **(D) 800,000 placements at each ε ∈ {0.5, 0.2, 0.05, 0.029, 0.01, 0.001}.**
  - Half are uniform over w₁ ∈ (0, 3], b₁ ∈ [0, 2π). Half are in the rescaled window p ∈ [0, 3], |q| ≤ 4.
  - Gap by exact extrema. The vectorised gap agrees with `src.exact_extrema.exact_gap` to 0.0 on 200 points.
  - 18,000–22,000 placements per ε have G > 0. All satisfy 1.4w₁ < x_a, |m_j| < β for the positive orientation, and
    the ellipse inequality of Lemma 2(d).
  - max G/(0.4εw₁) = 0.990–0.991, so Lemma 2(a) is nearly tight.
  - max G/(maximal drop 2(√(a²−1) − β)) ≈ 0.31.
- **(E) Grid argmax** of G̃_ε on p ∈ (0, 1.8] × q ∈ [−2.6, 2.6] (901 × 1301):

  | ε | K_grid | argmax (p, \|q\|) | max over p < 1 | max over \|q\| ≥ 2 | proved K_low |
  |---|---|---|---|---|---|
  | 0.2 | 0.54039 | (1.496, 1.132) | 0.392 | −0.0035 | (not valid) |
  | 0.05 | 0.56867 | (1.576, 1.184) | 0.387 | −0.0030 | 0.2675 |
  | 0.029 | 0.57302 | (1.588, 1.192) | 0.387 | −0.0029 | 0.4062 |
  | 0.01 | 0.57710 | (1.598, 1.200) | 0.386 | −0.0028 | 0.5212 |
  | 0.001 | 0.57922 | (1.604, 1.204) | 0.385 | −0.0028 | 0.5723 |
  | 0.0001 | 0.57927 | (1.604, 1.204) | 0.386 | −0.0028 | 0.5773 |

  The argmax is inside C at every ε checked, and it approaches K's argmax (1.6056, 1.2042). The maximum over p < 1
  stays below 0.4, consistent with Lemma 2(a).
  K_low is conservative by 0.17 at ε = 0.029. That slack is the only thing that limits ε₀.

### 12.10. Remark on the suggested route

The suggested route bounds G by the drop over the decreasing region plus the rise elsewhere, with a separate
many-period argument for w₁ = O(1). It is unnecessary.
- Lemma 1 turns every outer–inner pair into the closed form 2a cos(m) sin(d/2) − d. Choosing the pairs
  (−1.2, −0.8) [or (0.8, 1.2)] and (−2, 0.8) [or (−0.8, 2)] gives all three bounds: on G, on p and on q.
- The many-period regime is the single inequality |sin y|/y ≤ 1/π < 1/a for y = 1.4w₁ ≥ π.
- As a by-product, the finite-a branch-and-bound domain w₁ ∈ (0, a/1.4] (`ghat_bnb.py`) could be shrunk to
  w₁ < x_a/1.4. At a = 1.30, for example, that is 0.873 against 0.929.

## 13. The lag law: a no-fit constant κ(a) (Track 1A; derived after the fitted relationship was known; 2026-09-25)

**Status.** A first-order (linear, slaved) derivation, checked step by step. It is not a theorem about training: it
assumes that training tracks one conditional stationary branch and that the linearisation holds (§13.3(iv)). κ was computed
and every prediction committed (b6ae433) before any comparison. The comparison (899850d) used no refit. **Label
throughout: derived after the fitted relationship residual = α + β·ratio was known.** Code: `src/lag_law.py`; outputs:
`results/lag_law/`.

### 13.1 Setting and derivation

- Hidden coordinates θ = (w₁, b₁, b₂) at output scale s = |w₂|. The output bias b₂ is a trained coordinate in every
  protocol (its own gradient step each iteration; §13.3(ii)).
- Update: θ ← θ − ηP∇_θL(θ; s). P = I for SGD. For Adam, P = diag(1/(√v̂ + ε)), treated as fixed over the relaxation
  time.
- θ\*(s) is the tracked conditional stationary point (∇_θL(θ\*(s); s) = 0) with Hessian H (positive definite at every
  switch computed). Its tangent is θ\*′ = −H⁻¹∂ₛ∇_θL.
- Write θ = θ\*(s) + δ. To first order, with s advancing by ṡ per step:
  δ_{t+1} − δ_t = −ηPHδ_t − θ\*′ṡ.
- The slaved (steady) solution, valid when ṡ changes slowly on the relaxation time 1/(ηλ_min), is
  δ = −(ηPH)⁻¹θ\*′ṡ.
- The placement gap G depends on (w₁, b₁) only (∂G/∂b₂ = 0). Along the branch, g(s) = G(θ\*(s)) with g′(s\*) = ∇G·θ\*′,
  where s\* is the switch, g(s\*) = 0.
- The crossing G(θ\*(s_c) + δ) = 0 gives, to first order, g′(s\*)(s_c − s\*) + ∇G·δ = 0, i.e.
  g′(s\*)(s_c − s\*) = ∇G·(ηPH)⁻¹θ\*′ṡ.
- Dividing by s\*:

  **r = (s_c − s\*)/s\* = κ·χ,  χ = (ṡ/s\*)/(ηλ_min),  κ = λ_min·[∇G·(PH)⁻¹θ\*′]/[∇G·θ\*′],**

  with λ_min = λ_min(P^{1/2}HP^{1/2}), the slowest relaxation rate of the preconditioned dynamics.
- κ is dimensionless and invariant to rescaling P → cP (both λ_min and (PH)⁻¹ scale). So only P's relative shape
  enters κ, while P's absolute size enters χ.

### 13.2 Computation (`lag_law.run_kappa`, `kappa.csv`)

- Landscape: width 1, the 800-point population objective, at a = 1.30, 1.45, 1.50 and 1.60.
- Switch: s\* is the root of G(θ\*(s)) on the branch that is the certified global conditional minimiser. The search
  starts from the retained conditional-audit minimiser nearest the certified bracket, with Newton continuation and
  bisection in s. Every s\* lies inside its certified glob bracket.
- ∇G uses central differences (step 1e-6) of the exact-extrema gap. The one-sided differences agree to ≤ 5e-6
  relative, so the active extremal pair is unique.
- P: SGD uses I. Adam uses the per-coordinate median of the normalised 1/(√v̂ + ε) at the crossing over existing runs
  (48–80 runs per a). The interquartile range of κ over those runs' own P lies within 0.9% of the median.
- **The winding k.** f_a(t + 2π) = f_a(t) + 2π, so (w₁, b₁ + 2πk, b₂ − 2πk·s) has identical logits, loss, H, s\* and ∇G.
  The b₂ drift, however, changes: θ\*′ → θ\*′ − 2πk·e_{b₂}, and κ depends on k through (PH)⁻¹θ\*′ (`kappa_k`).
  - Each run's k is measured at its crossing as round((b₁ − b₁\*)/2π), with b₁ in w₂'s orientation.
  - All 1,750 predicted runs sit on k = −1 at a = 1.30 and on k = 0 at the other a. The largest |b₁ offset| from the
    copy is 0.039.
- Mirror branch: x → −x maps (w₁, b₁) → (−w₁, b₁). On the symmetric population it leaves G, s\*, H's spectrum and κ
  unchanged. 875 of the 1,750 runs sit on the mirror branch.

| a | s\* | k used | κ_Adam | κ_SGD |
|---|---|---|---|---|
| 1.30 | 4.9580 | −1 | 7.61 | 7.62 |
| 1.45 | 2.8962 | 0 | 4.59 | 4.61 |
| 1.50 | 2.5270 | 0 | 4.05 | 4.07 |
| 1.60 | 2.0022 | 0 | 3.29 | 3.32 |

(`kappa_by_winding.csv` gives κ for k = −1, 0, +1 at every a. At a = 1.30, κ = −7.53 for k = 0 and −22.7 for k = +1:
the sign itself depends on the winding.)

### 13.3 The four stated points

- **(i) χ against the paper's existing timescale ratio.**
  - The existing ratio (`residual_timescale.py`) is (d log s/dt at the crossing)/(lr·λ_min(D^{−1/2}HD^{−1/2})), with
    D = diag(√v̂ + ε), i.e. D⁻¹ = P, and each run's own D.
  - It equals χ except that it divides ṡ by s_c instead of s\*: ratio = χ·s\*/s_c = χ/(1 + r). The per-arm median r
    is at most 0.081, so the difference is second order in r.
  - The committed predictions use each run's measured ratio as χ.
- **(ii) The output bias is a trained coordinate, not profiled.**
  - If b₂ were profiled (always at its conditional optimum), θ = (w₁, b₁) with the Schur-complement Hessian, and b₂'s
    drift would not enter. κ would then be independent of the winding k.
  - The dynamics trains b₂ like any other coordinate. The controlled ramps below confirm the winding dependence, which
    exists only if b₂ is trained. So the trained-coordinate form is the one the dynamics implies.
- **(iii) Adam's momentum (β₁ = 0.9).**
  - In steady tracking δ is constant, so the gradient Hδ is constant and the momentum average equals it (m = Hδ).
    The steady lag is therefore unchanged by momentum.
  - Momentum only shapes the approach to the steady state. Its time constant is 1/(1 − β₁) = 10 steps.
  - With Adam's median absolute crossing preconditioner, the relaxation time 1/(ηλ_min(P^{1/2}HP^{1/2})) is 7.8, 12.1,
    14.7 and 16.2 steps at a = 1.30, 1.45, 1.50 and 1.60 (`lag_law.relaxation_rates`, `relaxation.csv`), so the two
    time scales are comparable. Momentum therefore affects
    transients but not the steady lag, and is not included.
  - A drift that changes on the momentum time scale adds a correction of relative order (β₁/(1 − β₁))·d log ṡ/dt.
    It is not computed here.
- **(iv) Where the linearisation should break down.**
  - (a) χ is not small, so δ is not small against the scale on which G is linear in θ. This is κχ ≳ 0.1 in practice.
  - (b) Near a fold of the branch, λ_min → 0 and the slaved solution does not exist.
  - (c) When P is not stationary on the relaxation time. For Adam this includes v̂ collapsing when gradients vanish, and
    Adam's per-step speed limit of about η per coordinate, which the ramp pilot hit (§13.6).
  - (d) Multiple branches, when a run is not tracking the branch whose switch is used. At width 2 this is the case in
    T2-3 (WP-15).

### 13.4 Winding verification by controlled ramps (before any comparison; `lag_law.winding_check`, `winding_check.csv`)

Population GD ramps (η = 0.3, P = I) on winding copy k, from 0.9·s\*, with s = s₀e^{γt} and γ = χηλ_min(H):

| a | k | χ | b₁ at crossing | κ_k (SGD) | predicted r | measured r |
|---|---|---|---|---|---|---|
| 1.30 | +0 | 0.003 | +3.832 | -7.44 | -0.0223 | -0.0215 |
| 1.30 | -1 | 0.003 | -2.452 | +7.62 | +0.0229 | +0.0236 |
| 1.30 | +1 | 0.003 | +10.115 | -22.50 | -0.0675 | -0.0607 |
| 1.50 | +0 | 0.002 | -2.291 | +4.07 | +0.0081 | +0.0082 |
| 1.50 | +0 | 0.005 | -2.291 | +4.07 | +0.0204 | +0.0207 |

The sign and size follow κ_k on every copy. The largest miss is k = +1 at a = 1.30 (−0.0607 measured, −0.0675 predicted), where |κχ| = 0.067 is closest to the nonlinear regime. This check was run before any comparison. It was quoted in the b6ae433 commit message and its producer was committed later (lag_law.winding_check, f08dc0d); the numbers reproduce exactly.

### 13.5 Comparison with existing runs (no refit; `compare_arms.csv`, `compare_per_a.csv`)

- **Arms:** 36/36 within the registered tolerance max(0.01, 0.25|pred|). The arms are the 32 intervention arms, SGD at
  two a and Task B at two a.
  - The four φ = 0.25 arms of the first lag test and the t\* rule, where w₂ is slowed from early on, are at
    0.70–0.76. They are within tolerance only through the 0.01 floor.
  - The other 32 arms are at 0.91–1.13. (The 899850d commit message said 0.99–1.13 and "the four φ = 0.25 arms"; the
    deconfounded primary rule's φ = 0.25 arms are at 1.03–1.07.)
- **Per a (post hoc, through the origin):** observed slope / committed κ = 0.89 (1.30), 0.83 (1.45), 0.91 (1.50),
  0.82 (1.60). This is within 30% at every a, with the law over-predicting by 9–18%.

### 13.6 The registered ramp test (Track 1B)

Registration: `results/ramp_registration.md` (c4b4c6d). Own thresholds were frozen before any crossing was read (83f66f5, 15b4744). Code: `src/ramp.py`.

- **Design.** s = s₀e^{γt} from 0.5·s\* after a 4,000-step warm-up. The hidden coordinates start on the population branch on winding k. 6 γ cells × 40 seeds per setting.
  - Settings: (1.30, −1) and (1.50, 0) are the natural windings; (1.30, 0) is the sign test, with κ < 0. Each is run with Adam and with SGD.
  - SGD's χ is a priori. Adam's χ uses the run's measured v̂ at crossing.
- **Registered verdicts.** All 1,440 runs crossed; none placed during warm-up; no run changed winding.

| a | k | opt | R1 slope | R2 signed Spearman | R3 slowest median |
|---|---|---|---|---|---|
| 1.30 | -1 | adam | -7.12 FAIL | 1.00 PASS | -0.0000 PASS |
| 1.30 | -1 | sgd | 0.78 PASS | 1.00 PASS | +0.0009 PASS |
| 1.30 | +0 | adam | 9.10 FAIL | 1.00 PASS | -0.0012 PASS |
| 1.30 | +0 | sgd | 1.30 FAIL | 1.00 PASS | -0.0014 PASS |
| 1.50 | +0 | adam | -9.99 FAIL | 0.71 FAIL | -0.0250 FAIL |
| 1.50 | +0 | sgd | -0.10 FAIL | 1.00 PASS | -0.0233 FAIL |

- **Why the registered R1 and R3 fail.** The observed residual was measured against each seed's own-sample *global* threshold, but the ramp forces one branch. For 35% of seeds at a = 1.30 and 68% at a = 1.50, that branch's own-sample switch lies more than 1% from the global threshold. Those seeds carry static offsets (5% quantile ≈ −0.25, 95% ≈ +0.09) in every γ cell.
  - This is a limit of the registered reference, not of the tracking law.
  - It is also a caution for §13.3(iv)(d): a residual is a lag only relative to the switch of the branch actually tracked.
- **POST HOC, against each seed's tracked-branch own-sample switch** (`ramp.posthoc_branch`):

| a | k | opt | R1 slope | R2 | R3 slowest median |
|---|---|---|---|---|---|
| 1.30 | -1 | adam | 0.52 FAIL | 1.00 PASS | +0.0002 PASS |
| 1.30 | -1 | sgd | 1.24 PASS | 1.00 PASS | +0.0010 PASS |
| 1.30 | +0 | adam | 1.49 FAIL | 1.00 PASS | -0.0009 PASS |
| 1.30 | +0 | sgd | 0.90 PASS | 1.00 PASS | -0.0010 PASS |
| 1.50 | +0 | adam | 0.90 PASS | 0.94 PASS | -0.0002 PASS |
| 1.50 | +0 | sgd | 1.14 PASS | 1.00 PASS | +0.0010 PASS |

  - SGD's observed/predicted cell medians are 0.99–1.06 in the four slowest cells, 0.97–1.11 in the fifth (κχ = 0.04) and 0.87–1.26 in the fastest (κχ = 0.1, the edge of the linear regime). The negative-κ copy crosses early, as predicted.
  - Adam's cell medians do not follow κχ (observed/predicted 0.35–0.94, 1.40–1.85 and −0.39 to 0.67 across the three settings), even post hoc. Its v̂ adapts during the ramp (§13.3(iv)(c)), and the post hoc R1 pass at a = 1.50 comes from the pooled slope only.
- **R4 (free Adam at η = 0.01, 0.005, 0.0025): PASS at both a.**
  - a = 1.30: median residuals 0.0296, 0.0304, 0.0308 (tolerance ±0.0100; 39 of 40 crossed at each η).
  - a = 1.50: median residuals 0.0619, 0.0625, 0.0647 (tolerance ±0.0155; 39 of 40 crossed at each η).
  - As the law predicts, χ is η-invariant in free training, so the lag does not vanish in the gradient-flow limit.
