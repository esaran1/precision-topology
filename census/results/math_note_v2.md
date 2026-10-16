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

- The corollary is asymptotic: "for all sufficiently small ε". No explicit ε₀ is claimed. (Forward reference, 2026-09-28: §16 gives an explicit ε₀ = 1.4·10⁻¹² for the switch in s-units and
  proves that the branch is the global minimiser there.) The measured
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
- Result: **c_s := A′(0)/A* ∈ [0.6621547, 0.6621550]**. This is the first-order coefficient of the switch in s-units. The symbol c_s keeps it distinct from c₁ = 0.28523, the coefficient in R-units.

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

**Follow-up (final round, 2026-09-25; registered e233ef5; WP-30).** Five added certified brackets (a = 1.08–1.12) and the
four original ones, with the registered estimator unchanged, give the feasible set C = [0.26531, 0.30687], of width
0.042. It contains the derived c₁ ∈ [0.2852300, 0.2852303]: PASS. The original test stays INCONCLUSIVE as registered. The coefficient is thus confirmed with certified thresholds at ε = 0.08 to 0.12. This is not a small-ε confirmation: at these ε the registered ±ε³ allowance is a leading part of the tolerance. All ten added bracket-end certificates (a = 1.08–1.12, both ends) were independently checked in Arb and pass. The rescaled Ĝ at these a has no independent checker.

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
  | product term ακ (its leading part c_s·k₁·ε² has certified coefficient 0.662 × (−0.377) = −0.250) | −0.060 |
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
- **Correction (final round, 2026-09-26; `src/theorem1_checks.py`, WP-34).**
  - On the 800-point grid population every point is a multiple of q = 0.4/79,401, so D(α) is periodic.
  - α* = 1.7913244 is the maximiser of |D| on the data points only for α ∈ (0, ≈6.2·10⁵). An aliased weight α ≈ π/q
    gives |D| = 2.
  - On the continuous windows, α* = 1.7922917 is the unique global maximiser (checked by computer).
  - Width 1 is unaffected: the alias beats α* only for s ≲ 2·10⁻¹², the localisation lemma excludes it for
    s > 7.2·10⁻⁶, and it is unplaced.
  - At width 2, state the s → 0 selection on the continuous windows, or for s above about 10⁻¹¹.

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
- From A = sε^{3/2} (§1) and the switch A_ε = A*(1 + c_s ε + O(ε²)) (§8), with c_s := A′(0)/A*:
  |w₂|_glob(a) = A*·ε^{−3/2}·(1 + c_s ε + O(ε²)), c_s = 0.66215, with ε = a − 1.
- A* ∈ [0.68125, 0.6875] is certified and independently checked; the Krawczyk value is 0.6854452.
  c_s = A′(0)/A* ∈ [0.6621547, 0.6621550].
- (In R units the correction is c₁ = 0.28523; c₁ = c_s + k₁ + O(ε) bookkeeping as in §8. The difference is k₁ = −0.37692, from K(ε).)

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
- **Why Adam differs in the ramp (POST HOC; `ramp.adam_contrast`).**
  - At crossing, √v̂ is 8–88× smaller than in free training, because the warm-up holds the run at a stationary point and v̂ decays.
  - The frozen-P relaxation time is then 0.17–0.19 steps, against 7.8–15.0 steps in free training. So ηλ_min > 1, and the linearisation of §13.1 (small ηλ, P fixed over the relaxation) does not apply. This is case (iv)(c).
  - The growth rate is only 2.3–2.7× the free rate, and the branch and winding are the same, so neither explains the difference.
  - The law's Adam form therefore requires a stationary preconditioner with ηλ_min ≪ 1. Free training meets this; the ramp does not.
- **The Adam ramp from initialisation (follow-up; not registered, infeasible by its design rules; `src/ramp2.py`).**
  - A γ was admissible only if ≥ 6 of 8 pilot runs cross and the median ηλ_min at crossing is ≤ 0.5. No γ qualified.
  - Slow ramps let v̂ collapse (ηλ_min 4.4–11.5). Where ηλ_min ≤ 0.5, at most 3 of 8 runs cross.
  - The law's Adam form therefore has support only in free training, where growth is not imposed.
- **R4 (free Adam at η = 0.01, 0.005, 0.0025): PASS at both a.**
  - a = 1.30: median residuals 0.0296, 0.0304, 0.0308 (tolerance ±0.0100; 39 of 40 crossed at each η).
  - a = 1.50: median residuals 0.0619, 0.0625, 0.0647 (tolerance ±0.0155; 39 of 40 crossed at each η).
  - As the law predicts, χ is η-invariant in free training, so the lag does not vanish in the gradient-flow limit.

### 13.7 The validity condition across activations (POST HOC; `src/act_posthoc2.py`)

- The linear lag law was verified at width 1 on runs with χ ≤ 0.064; the largest arm median χ at which it held is 0.0249.
- The Track 3A runs, scored against each run's tracked-branch switch on its own sample:
  - **GELU** (median χ 0.0137, inside the range): 96% of runs with a branch switch cross within 1% of it. The median
    residual is +0.0017, against κχ = +0.0021.
  - **SiLU and Mish** (median χ 0.22 and 0.26, about 10× beyond the range): the residual has the wrong sign for κχ.
- This is consistent with (iv)(a): the law needs χ small. It is not a test of the law outside that regime.
- GELU's 15 early crossings (s ≤ 0.45) lie on the tail-placed one-sided-ramp branch, where the continuation finds no
  switch.

### 13.8 Exact linear response along the trajectory (final round; POST HOC; `src/linear_response.py`, WP-31)

Setup:
- The three simplifications of §13.1 are removed: coefficients frozen at the switch, the slaved displacement and no
  momentum.
- The recursion δ_{t+1} = (I − ηP_tH_t)δ_t − Δθ\*_t, with Adam's first moment as a state, is iterated along each run's
  actual s_t and v̂_t.
- The start is at 0.7·s\* (and 0.5·s\*), with s\* the switch of the branch the run tracks on its own sample.
- All 1,750 runs were predicted, and the predictions were hashed (e86ca102…) before any observed crossing was read.

Results:
- **Full model:** observed/predicted lag is 1.02, 1.03, 1.04 and 1.05 at a = 1.30, 1.45, 1.50 and 1.60, and 1.00–1.05
  per arm.
- The two starting points give identical crossings. No run is on a different branch at either start.
- **What the 0.82–0.91 of §13.5 comes from:**
  - Mainly the reference. §13.5 measured the lag from each sample's global threshold. In 22.8% of runs that threshold
    differs by more than 1% from the switch of the tracked branch. Against the tracked switch, κχ gives 1.02 (a = 1.30)
    and 1.01 (a = 1.45).
  - Secondly the slaved approximation at the largest κχ: 0.96 and 0.90 at a = 1.50 and 1.60 with each run's own
    coefficients.
  - Time-varying coefficients move the slope by at most 0.01. Momentum does not change the lag.
- The remaining +2–5% is the linearisation of the gradient in δ. With the exact gradient, the observed crossing step
  is reproduced in every run checked (the first 12 seeds of each arm).

### 13.9 A registered test at an unseen activation value (final round; Track A, WP-35)

Setup:
- At a = 1.65, which had no prior crossing data, everything was frozen and hashed before any run (fcd2e46): the
  validated switch (s\*_pop = 1.81002), κ, the winding rule (canonical b₁ in (−π, π], defined from crossings at other
  a), the branch and preconditioner rules (taken at the last upward passage of 0.5·s\*), and the own-sample thresholds.
- 80 fresh seeds per optimiser.

Results:
- **SGD:** L1 (trajectory-integrated median ratio 1.056), L2 (closed form 1.024) and L3 (per-run Spearman 0.995) all
  PASS.
- **Adam:** L1 (1.065) and L2 (1.045) PASS. L3 (Spearman 0.25 < 0.5) FAILS: with P frozen at the rule point, the law
  predicts Adam's typical lag but not which runs lag more.
- 5 of 6 registered criteria pass. That P drifts after the rule point is a post hoc hypothesis and is untested here.

## 14. Fold tracking: crossing at the occupied branch's fold, and the (rate)^{2/3} delay (Track 3 (4); 2026-09-28)

**Status, stated per part.**
- **§14.2, Theorem F (the normal form): PROVED, every step checked.**
- **§14.3 (reduction of preconditioned gradient descent to the normal form; the constants m′, c′): derived, with the
  reduction step resting on cited results (centre manifold / fold-point analysis, §14.5). It is NOT proved here for
  the n-dimensional discrete map, and it is not presented as a theorem.**
- **§14.4 (the condition for crossing at the fold): a statement of hypotheses and their consequences, conditional on §14.3.**
- **Numerical checks, not part of any proof:** `src/theory_checks.py fold` → `theory_checks_fold.json`.
- Numbers for the paper's v3 fold come from Track 1A (`results/sb_fold/`). None are used here.

### 14.1 Setting

- Trained coordinates θ ∈ ℝⁿ at output scale s. Preconditioned gradient descent θ ← θ − ηP∇_θL(θ; s), with P
  symmetric positive definite and held fixed (no momentum).
- The scale moves slowly: s_t = s_F + ṡ·(t − t_F) near the point of interest, with ṡ > 0 per step.
- **Occupied branch U**: a curve θ_U(s) of strict local minima (∇_θL = 0, H = ∇²_θL ≻ 0) that training tracks.
- **Fold of U at s_F**: at (θ_F, s_F) the Hessian H_F has a simple zero eigenvalue, H_F r = 0, and U does not
  continue as a local minimum past s_F.
- **Normalisation**: r is normalised by rᵀP⁻¹r = 1.
- **Fold constants**:
  - m′ = rᵀ∂_s∇_θL(θ_F, s_F), the transversality;
  - c′ = ½∇³_θL(θ_F, s_F)[r, r, r], the quadratic term.
  - The fold is nondegenerate iff m′c′ ≠ 0.
  - Orient r so that the branch lies at y > 0 in the coordinate θ = θ_F + y·r + (hyperbolic part). Then c′ > 0, and
    the branch exists for s < s_F iff m′ > 0.

### 14.2 Theorem F (the Riccati normal form)

**Statement.** Let X solve X′ = T + X² on an interval (−∞, T_b), with X(T) + √(−T) → 0 as T → −∞, i.e. the solution
attached to the stable branch X = −√(−T). Then:
- (i) X is unique and X(T) = Ai′(−T)/Ai(−T);
- (ii) X is finite for T < Ω₀, and X(T) = 1/(Ω₀ − T) + O(Ω₀ − T) as T ↑ Ω₀;
- (iii) here **Ω₀ = −a₁ = 2.338107410459767…**, with a₁ the largest (least negative) zero of the Airy function Ai.

**Proof.**
1. **Linearisation.** On any interval where u ≠ 0, the substitution X = −u′/u gives
   X′ = −u″/u + (u′/u)² = −u″/u + X². So X′ = T + X² iff u″ + T·u = 0.
2. **General solution.** Ai″(z) = z·Ai(z), so d²/dT² Ai(−T) = Ai″(−T) = −T·Ai(−T). Likewise for Bi. So
   u = A·Ai(−T) + B·Bi(−T), and every solution X is −u′/u for some (A : B) ≠ (0 : 0).
3. **Branch condition, as z = −T → +∞.** Ai(z) ~ e^{−ζ}/(2√π z^{1/4}) and Bi(z) ~ e^{ζ}/(√π z^{1/4}), with
   ζ = (2/3)z^{3/2}. Also Ai′(z)/Ai(z) = −√z + O(z⁻¹) and Bi′(z)/Bi(z) = +√z + O(z⁻¹).
   - Since u′(T) = −A·Ai′(−T) − B·Bi′(−T), we have X = [A·Ai′(z) + B·Bi′(z)]/[A·Ai(z) + B·Bi(z)].
   - If B ≠ 0, the Bi terms dominate and X → +√z = +√(−T), the unstable branch.
   - So the stable branch forces B = 0. Then X = Ai′(z)/Ai(z) = −√(−T) + O(1/T), and uniqueness follows.
4. **Blow-up.** All zeros of Ai are real and negative. So Ai(−T) > 0 for −T > a₁, i.e. for T < Ω₀, and X is finite.
   - The zero a₁ is simple: Ai(a₁) = Ai′(a₁) = 0 together with Ai″ = z·Ai would force Ai ≡ 0.
   - Near T = Ω₀: Ai(−T) = −Ai′(a₁)(T − Ω₀) + O((T − Ω₀)²) and Ai′(−T) = Ai′(a₁) + O((T − Ω₀)²), since Ai″(a₁) = 0.
   - Hence X = 1/(Ω₀ − T) + O(Ω₀ − T). ∎

**Scaled form.** Consider the scalar flow dy/dt = η(m(s_F − s(t)) − c·y²), with m, c > 0 and s(t) = s_F + ṡ(t − t_F).
- Set β = (η²mcṡ)^{−1/3}, k = 1/(βηc), y = −kX and t − t_F = βT.
- Substituting gives exactly X′ = (η²mcṡβ³)·T + (ηckβ)·X² = T + X².
- The stable branch y = +√(m(s_F − s)/c) maps to X = −√(−T), since k²/(ṡβ) = m/c.
- Hence the solution that tracks the branch from the far past escapes (y → −∞) at t = t_F + Ω₀β, that is at
  **s_esc = s_F + Ω₀·ṡ^{2/3}·(η²mc)^{−1/3}**. This is exact for the flow.

### 14.3 Reduction of preconditioned gradient descent (derived; reduction step cited)

- **Continuous-time limit.** Take θ̇ = −ηP∇_θL(θ; s(t)), and write θ = θ_F + y·r + w with w in the complementary
  PH-invariant subspace.
- **Projection onto the centre direction.** The left null vector of PH is l = P⁻¹r: lᵀPH = rᵀH = 0. With lᵀr = 1:
  ẏ = lᵀθ̇ = −η·rᵀ∇_θL.
- **Expansion at the fold.** rᵀH_F·w = 0 because H_F is symmetric, and rᵀH_F·r = 0. So
  ẏ = −η[m′(s − s_F) + c′y² + O(|y|³ + |y||w| + |w|² + |y||s − s_F| + |s − s_F|²)].
- **The hyperbolic part.** w relaxes at the nonzero rates of PH, which are positive on the minimum branch. It is
  slaved to a centre manifold w = O(y² + |s − s_F|). That keeps every remainder term above of higher order.
- **Consequence.** With μ = m′(s_F − s) this is the scalar flow of §14.2 with m = m′, c = c′, plus higher-order terms.
  The leading escape is:

  **s_esc − s_F = Ω₀·ṡ^{2/3}·(η²|m′c′|)^{−1/3} + O(ṡ·log(1/ṡ)),**

  or, relative and dimensionless,

  **r_F := (s_esc − s_F)/s_F = Ω₀·ε_F^{2/3},  ε_F = (ṡ/s_F)/(η·Λ_F),  Λ_F = (|m′c′|·s_F)^{1/2}.**

- **The error term is not proved here** (§14.5). For the planar fold flow, [KS01] (Remark 2.11, citing [MR80]) gives
  the leading term −Ω₀ε^{2/3} and a next term O(ε ln ε). The extension to n fast variables with one centre direction
  (a centre-manifold reduction) is only a remark in [KS01] and is not proved here.
- **Measurable form of the constant.**
  - Along U, y_U(s) = (m′(s_F − s)/c′)^{1/2}. First-order perturbation of the simple zero eigenvalue of PH, with
    left vector l and lᵀr = 1, gives λ_min(P^{1/2}HP^{1/2})(s) = 2c′·y_U(s) + O(s_F − s).
  - Hence **λ_min(s)² = 4|m′c′|·(s_F − s) + O((s_F − s)^{3/2})**.
  - So |m′c′| is ¼ of the slope of λ_min² against s at the fold. It is read off the same preconditioned Hessian the
    lag law already uses (§13.1). Λ_F² is s_F times that quarter-slope.
- **Free training (corollary, conditional on the above).** When s is itself trained, ṡ = η·v with v = −p_s∂L/∂s at
  the fold. Then s_esc − s_F = Ω₀·v^{2/3}·|m′c′|^{−1/3}, which is independent of η at leading order. This is the
  analogue of χ's learning-rate invariance (§13.6, R4).
- **Discrete map.** One step of the map is a forward-Euler step of the flow. In the escape region its relative error
  per step is O(ηc′|y|). The cited result is for flows, and the discrete statement is checked only numerically (F2–F4).

### 14.4 When a discontinuous global switch leads training to cross at the fold (conditional on §14.3)

**Hypotheses:**
- **(H-F1) Discontinuous global switch.** At s_glob, two strict local-minimum branches U (unplaced, G < 0) and V
  (placed) have equal loss. The global minimiser jumps from U to V.
- **(H-F2) Occupation.** Training tracks U at s_glob, with the linear lag of §13 valid (κχ small against U's
  relaxation).
- **(H-F3) U continues unplaced to a nondegenerate fold.** For s ∈ [s_glob, s_F), U is a strict local minimum with
  G(θ_U(s)) < 0. At s_F it folds, with m′c′ ≠ 0.
- **(H-F4) Deterministic dynamics.** The dynamics are full-batch, with fixed P, and no noise large enough to cross
  the barrier between U and V before the fold. The barrier shrinks like (s_F − s)^{3/2} (the normal form's
  (4/3)·m′^{3/2}(s_F − s)^{3/2}/c′^{1/2}).
- **(H-F5) The escape leads to placement.** The unstable direction at the fold leads into V's basin, and G changes
  sign during that fast jump.
- **(H-F6) Slow passage.** ε_F ≪ 1, and ηλ_max(PH) < 2 (a stable map).

**Consequence.**
- Training does not cross at the global switch. Under deterministic tracking a strict local minimum is never left
  while it exists.
- It crosses at **s_c = s_F + s_F·Ω₀·ε_F^{2/3} + O(s_F·ε_F·log(1/ε_F))**.
  - The offset s_F − s_glob is static: it does not vanish as the rate goes to 0 (hysteresis).
  - The rate-dependent delay scales as (rate)^{2/3} with the universal constant Ω₀ = 2.33811. This contrasts with
    the linear lag κχ at a continuous switch.
  - The O(ε_F) term depends on where G = 0 lies along the jump (checked: F2 escape at y = −1).

**The general rule, which contains §13 as the continuous case.** Training crosses at the first placement event of
the branch it occupies: that branch's own continuous switch (lag κχ, §13) or its fold (delay Ω₀ε_F^{2/3}), whichever
comes first. The global threshold enters only through which branch is occupied.

**Where it fails:**
- **(a) Noise:** barrier crossing (Kramers) before s_F makes the crossing earlier.
- **(b) Adam's moving preconditioner:** v̂ changes on the escape time scale, the same failure mode as §13.3(iv)(c).
- **(c) Momentum:** β₁ turns the flow second order. The escape time β scales as ṡ^{−1/3} and eventually exceeds
  momentum's 1/(1 − β₁) = 10 steps, and the resulting correction is then O(1) steps, i.e. O(ṡ) in s. This is not
  proved.
- **(d) Degenerate fold** (m′c′ = 0, e.g. a cusp): a different exponent.
- **(e) Fast passage** (ε_F not small): no asymptotic statement.

### 14.5 What is proved, and what is cited

- **Proved here:** Theorem F; the exact scaling of the scalar flow; the formulas for m′, c′ and λ_min², as
  first-order perturbation identities.
- **Cited.** Each reference was checked against its publisher or DOI record (Crossref) on 2026-09-28. The Krupa–Szmolyan
  statements below were read in the paper itself.
  - **[KS01]** M. Krupa and P. Szmolyan, "Extending geometric singular perturbation theory to nonhyperbolic
    points—fold and canard points in two dimensions", *SIAM J. Math. Anal.* 33(2):286–314, 2001,
    doi:10.1137/S0036141099360919. The statements used:
    - **Theorem 2.1.** For the planar fold x′ = −y + x² + h, y′ = ε·g (g = −1 + O(x, y, ε); nondegeneracy (2.2)), with
      reduced flow toward the fold (jump point):
      - the attracting slow manifold S_{a,ε} reaches Δ_out = {x = ρ} at y = h(ε) = O(ε^{2/3});
      - the transition map is a contraction with rate O(e^{−c/ε}).
    - **Proposition 2.3 and Remark 2.4.**
      - The Riccati equation x′ = x² − y, y′ = −1 has a unique orbit asymptotic to the attracting branch.
      - That orbit has the horizontal asymptote y = −Ω₀.
      - Ω₀ is the smallest positive zero of J_{−1/3}(2z^{3/2}/3) + J_{1/3}(2z^{3/2}/3).
    - **Remark 2.11.** h(ε) = −Ω₀ε^{2/3} + o(ε^{2/3}). The remark adds that "the next term in the expansion is
      O(ε ln ε)", referring to [MR80].
    - **§1.** The analysis "carries over to higher-dimensional problems with one-dimensional critical manifolds
      containing fold points. By means of a center-manifold reduction, all normally hyperbolic directions can be
      eliminated". This is a remark in [KS01], not a theorem proved there.
  - **[MR80]** E. F. Mishchenko and N. Kh. Rozov, *Differential Equations with Small Parameters and Relaxation
    Oscillations*, Plenum Press / Springer US, New York, 1980, doi:10.1007/978-1-4615-9047-7.
    - [KS01] cites pp. 68–72 for the Riccati facts and for the O(ε ln ε) term.
    - I have not read [MR80] itself. The O(ε ln ε) error term is used here only as [KS01] Remark 2.11 reports it.
  - **Consistency with Theorem F.** J_{−1/3}(ζ) + J_{1/3}(ζ) = 3Ai(−z)/√z with ζ = (2/3)z^{3/2} (standard Airy–Bessel
    relation). So Ω₀ of [KS01] is −a₁, the Ω₀ of Theorem F. This is checked numerically (F0b).
    - Theorem F's X′ = T + X² is [KS01]'s (2.16) with x = X and y = −T. Their asymptote y = −Ω₀ is our blow-up
      time T = Ω₀.
  - **How §14.3 uses them.**
    - The leading delay Ω₀ε^{2/3} and the O(ε ln ε) remainder hold for **planar flows** ([KS01] Theorem 2.1 with
      Remark 2.11).
    - The reduction from n dimensions is [KS01]'s §1 remark, not a proved theorem there. It is **not proved here**.
    - A discrete-time theory near non-hyperbolic points exists (S. Jelbart and C. Kuehn, "Extending discrete
      geometric singular perturbation theory to non-hyperbolic points", *Nonlinearity* 37(10):105006, 2024,
      doi:10.1088/1361-6544/ad72c5; record checked). I have not read it, and nothing here relies on it.
- **Not proved:**
  - the statement for the discrete map (checked numerically: F2–F4);
  - the momentum and Adam cases.

- **Forward reference (2026-10-09).** The next-order terms (ε ln ε and ε, with their coefficients) are derived in §18.

### 14.6 Numerical checks (`theory_checks_fold.json`; not part of any proof)

- **F0.** Ω₀ = 2.33810741045976703848919725245 (mpmath: −a₁).
- **F0b.** The smallest positive zero of [KS01]'s J_{−1/3}(2z^{3/2}/3) + J_{1/3}(2z^{3/2}/3) equals −a₁ (difference < 1e−12).
- **F1.** RK4 on X′ = T + X² from the branch at T₀ = −4, −6, −8 blows up at 2.3381073, 2.33810741045 and
  2.33810741046. The exponentially small dependence on T₀ is as in step 3.
- **F2.** Scalar map, η = 0.3, m = c = p = 1. Observed/predicted delay is 0.979, 0.995, 1.0004 and 1.0024 at
  ṡ = 1e−3 … 1e−6.
  - A second case (η = 0.1, m = 2, c = 0.5, p = 0.4) gives 0.893, 0.949 and 0.977 at ṡ = 1e−4 … 1e−6.
  - The deficit is −ṡ/(ηpc·|y_esc|), from escape measured at y = −1: an O(ṡ) term, shrinking as ṡ^{1/3} relative.
- **F3.** Two-dimensional system with a coupled hyperbolic direction and non-diagonal P, with m′ and c′ computed from
  their definitions at the fold: 0.944, 0.971 and 0.986 at ṡ = 1e−4, 1e−5 and 1e−6.
- **F4.** Free training, ṡ = η·v at η = 0.2, 0.1 and 0.05: the delay in s is 0.00501, 0.00497 and 0.00495 against the
  η-free prediction 0.00504.
- **F5.** λ_min²/(s_F − s) is 2.206, 2.239, 2.262 and 2.279 at s_F − s = 1e−2 … 1.25e−3. It converges, as √(s_F − s),
  to 4|m′c′| = 2.318.

## 15. The lag law as a theorem (Track 3 (2); 2026-09-28)

**Status, stated per part.**
- **Theorems L1 and L2 and Corollary L3: PROVED, every step checked.**
  - They cover preconditioned gradient descent with a **fixed** preconditioner and no momentum, on a branch that is
    locally strongly convex with a Lipschitz Hessian, driven by a scale sequence s_t that increases slowly.
  - The rate may vary slowly: an imposed ramp, or s itself trained.
- **Not covered:** Adam's moving v̂ and momentum (§15.5). The theorems give κχ + O(χ²) with explicit constants plus a
  discreteness term of at most one step.
- **Numerical checks (not part of the proof):** `src/theory_checks.py lag` → `theory_checks_lag.json`.
- **§13 is unchanged.** §15 turns its first-order derivation (§13.1) into a bounded statement.

### 15.1 Setting and hypotheses

- **Coordinates.** Work in z = P^{−1/2}θ and set L̃(z, s) = L(P^{1/2}z, s).
  - Then θ ← θ − ηP∇_θL is exactly z ← z − η∇_zL̃, because ∇_zL̃ = P^{1/2}∇_θL.
  - The Hessian in z is P^{1/2}∇²_θL·P^{1/2}.
  - All norms below are Euclidean in z.
- **The tube.** J is a closed s-interval. The tube is T_ρ = {(z, s): s ∈ J, |z − z\*(s)| ≤ ρ}.

**Hypotheses:**
- **(L0) Branch.** z\* ∈ C²(J), with ∇_zL̃(z\*(s), s) = 0, |z\*′| ≤ D₁ and |z\*″| ≤ D₂.
- **(L1) Strong convexity on the tube.** λI ⪯ ∇²_zL̃ ⪯ ΛI on T_ρ, with λ > 0 and ηΛ ≤ 1.
- **(L2) Lipschitz Hessian.** ‖∇²_zL̃(z, s) − ∇²_zL̃(z′, s)‖ ≤ M|z − z′| on T_ρ.
- **(L3) Hessian along the branch.** H(s) := ∇²_zL̃(z\*(s), s) is C¹, with ‖H′‖ ≤ H₁.
- **(L4) Scale sequence.** s_{t+1} = s_t + ṡ_t, with 0 < ṡ_t ≤ ṡ and |ṡ_{t+1} − ṡ_t| ≤ K_s·ṡ_t².
  - K_s = 0 for a linear ramp.
  - K_s = 1/min s for s_{t+1} = (1 + γ)s_t.
  - For free training, see Corollary L3.

**Definitions:**
- **Rescaled rate:** q := ṡ/(ηλ), which has the units of s. The dimensionless χ of §13 is χ = ṡ/(s\*ηλ_min(H(s\*))).
- D̄₁ := D₁ + D₂ṡ/2 and S := q·D̄₁.
- C_e := 2MD̄₁²/λ + H₁D̄₁/λ + D₂ + D₁K_s, and E := C_e·q².
- **Displacements:** δ_t := z_t − z\*(s_t), Δ_t := z\*(s_{t+1}) − z\*(s_t), and the **slaved displacement**
  δ^sl_t := −(ηH(s_t))⁻¹Δ_t.

### 15.2 Theorem L1 (tracking)

**Statement.** Assume (L0)–(L4), E ≤ S and 2S ≤ ρ, and let |δ_{t₀} − δ^sl_{t₀}| ≤ S. Then for every t ≥ t₀ with
s_t, s_{t+1}, s_{t+2} ∈ J (step 5 evaluates z\* at s_{t+1} + ṡ_{t+1} = s_{t+2}),

  **|δ_t − δ^sl_t| ≤ E + (1 − ηλ)^{t−t₀}·|δ_{t₀} − δ^sl_{t₀}|,  with |δ^sl_t| ≤ S.**

So after the transient, the state follows the slaved displacement to O(q²), while the displacement itself is O(q).

**Proof.**
1. **Recursion.** δ_{t+1} = z_t − η∇L̃(z_t, s_t) − z\*(s_{t+1}) = δ_t − η∇L̃(z\*(s_t) + δ_t, s_t) − Δ_t.
   - While |δ_t| ≤ ρ, Taylor's theorem with (L2) gives ∇L̃(z\*(s_t) + δ_t, s_t) = H_tδ_t + R_t, with
     |R_t| ≤ (M/2)|δ_t|², because the segment lies in T_ρ.
   - Hence δ_{t+1} = (I − ηH_t)δ_t − Δ_t − ηR_t.
2. **The slaved point is a fixed point of the linear part.** (I − ηH_t)δ^sl_t − Δ_t = δ^sl_t + Δ_t − Δ_t = δ^sl_t. So
   e_t := δ_t − δ^sl_t satisfies e_{t+1} = (I − ηH_t)e_t − ηR_t − (δ^sl_{t+1} − δ^sl_t).
3. **Contraction.** spec(ηH_t) ⊂ [ηλ, ηΛ] ⊂ (0, 1] by (L1), so ‖I − ηH_t‖ ≤ 1 − ηλ.
4. **Size of the slaved point.** |Δ_t| ≤ D₁ṡ_t + D₂ṡ_t²/2 ≤ D̄₁ṡ, so |δ^sl_t| ≤ D̄₁ṡ/(ηλ) = S.
5. **Drift of the slaved point.**
   - δ^sl_{t+1} − δ^sl_t = −η⁻¹[H_{t+1}⁻¹(Δ_{t+1} − Δ_t) + (H_{t+1}⁻¹ − H_t⁻¹)Δ_t].
   - First, Δ_{t+1} − Δ_t = ∫_{s_t}^{s_t+ṡ_t}[z\*′(u + ṡ_t) − z\*′(u)]du + ∫_{s_{t+1}+ṡ_t}^{s_{t+1}+ṡ_{t+1}} z\*′.
     This uses s_{t+1} = s_t + ṡ_t. So |Δ_{t+1} − Δ_t| ≤ D₂ṡ_t² + D₁K_sṡ_t².
   - Second, ‖H_{t+1}⁻¹ − H_t⁻¹‖ = ‖H_{t+1}⁻¹(H_t − H_{t+1})H_t⁻¹‖ ≤ H₁ṡ_t/λ².
   - Hence V := |δ^sl_{t+1} − δ^sl_t| ≤ (ṡ²/(ηλ))·[D₂ + D₁K_s + H₁D̄₁/λ].
6. **Induction.** Suppose |e_t| ≤ S. Then |δ_t| ≤ 2S ≤ ρ, so step 1 applies, and
   - |e_{t+1}| ≤ (1 − ηλ)|e_t| + ηM(2S)²/2 + V = (1 − ηλ)|e_t| + ηλ·[2MS²/λ + V/(ηλ)] = (1 − ηλ)|e_t| + ηλE.
   - The last equality uses 2MS²/λ + V/(ηλ) = q²[2MD̄₁²/λ + D₂ + D₁K_s + H₁D̄₁/λ] = E.
   - Since E ≤ S, the right-hand side is ≤ S, so the hypothesis propagates.
   - Unrolling gives |e_t| ≤ E + (1 − ηλ)^{t−t₀}(|e_{t₀}| − E)₊. ∎

### 15.3 Theorem L2 (the lag)

**Further hypotheses.**
- **(L5) The gap is C² on the tube.** G ∈ C²(T_ρ) in z, with |∇G| ≤ g₁ and ‖∇²G‖ ≤ g₂.
  - For the placement gap this holds where the active extremal pair is unique and nondegenerate (§13.2 checks this
    at every switch used).
  - Let g(s) := G(z\*(s)), with g(s\*) = 0. On the window J′ = [s\* − w, s\* + w] ⊂ J, assume g′ ≥ γ > 0 and
    |g″| ≤ g₂″. One may take g₂″ = g₂D₁² + g₁D₂.
- **Reference rate.** v̂ is the rate ṡ_t̂ at the first step with s_t̂ ≥ s\*. Over the window,
  |ṡ_t − v̂| ≤ Δv := K_sṡ(2w + ṡ). (The increments sum to at most 2w + ṡ, and each rate change is at most K_sṡ·ṡ_t.)

**Definitions.**
- h(s) := −∇G(z\*(s))·(ηH(s))⁻¹[z\*(s + v̂) − z\*(s)] and F(s) := g(s) + h(s).
- **The lag prediction:** ℓ := −h(s\*)/g′(s\*) = ∇G·H(s\*)⁻¹[z\*(s\* + v̂) − z\*(s\*)]/(η·g′(s\*)).
- S₁ := q(D₂ + H₁D̄₁/λ), h′_max := g₂D₁S + g₁S₁, and E_w := sup of |e_t| over the steps in the window (Theorem L1).
- B := g₁E_w + g₂(S + E_w)²/2 + g₁D₁Δv/(ηλ).
- C_ℓ := [2g₂″g₁²S²/γ² + 2h′_max·g₁S/γ]/γ.

**Statement.** Assume Theorem L1's hypotheses, (L5), h′_max ≤ γ/2, and w > 2g₁S/γ + 2B/γ + ṡ. Let s_c be s_t at the
first step with s_t ∈ J′ and G(z_t) ≥ 0. Then

  **|s_c − s\* − ℓ| ≤ C_ℓ + 2B/γ + ṡ.**

**The same statement in §13's variables.**
- ℓ = s\*·κχ + ζ, with κ = λ_min·[∇G·H⁻¹z\*′]/[∇G·z\*′] and χ = v̂/(s\*ηλ_min), both at s\*, and |ζ| ≤ g₁D₂v̂²/(2ηλγ).
  In θ-coordinates this is §13's κ, since ∇_zG·H_z⁻¹z\*′_z = ∇_θG·(PH_θ)⁻¹θ\*′.
- C_ℓ, B and ζ are all O(q²) = O(χ²), with the constants above. So

  **r = (s_c − s\*)/s\* = κχ + O(χ²) + τ,  0 ≤ τ ≤ ṡ/s\* = ηλ_min·χ·(ṡ/v̂).**

- The last term τ is the discreteness of "the first step with G ≥ 0". It vanishes in the gradient-flow limit. A
  crossing located by interpolating G between steps removes it; that version is checked numerically only (L2 below).

**Proof.**
1. **G along the trajectory.** Write G(z_t) = g(s_t) + ∇G(z\*(s_t))·δ_t + ρ_t, with |ρ_t| ≤ g₂|δ_t|²/2 (Taylor, (L5)).
   Split δ_t = δ^sl_t + e_t.
2. **Replace the actual rate by v̂.** ∇G(z\*(s_t))·δ^sl_t differs from h(s_t) only through Δ_t versus
   z\*(s_t + v̂) − z\*(s_t). The difference is at most g₁D₁|ṡ_t − v̂|/(ηλ) ≤ g₁D₁Δv/(ηλ).
   So |G(z_t) − F(s_t)| ≤ g₁|e_t| + g₂(S + |e_t|)²/2 + g₁D₁Δv/(ηλ) ≤ B.
3. **F is increasing.**
   - |h| ≤ g₁S.
   - |h′| ≤ h′_max: d/ds ∇G(z\*(s)) contributes g₂D₁·S, and d/ds of (ηH)⁻¹[z\*(s + v̂) − z\*(s)] is at most
     (1/η)[H₁D̄₁v̂/λ² + D₂v̂/λ] ≤ S₁. This is step 5 of L1 with ṡ_t replaced by the fixed v̂.
   - Hence F′ ≥ g′ − h′_max ≥ γ/2 on J′.
4. **The zero of F.**
   - F(s\*) = h(s\*) with |h(s\*)| ≤ g₁S, and F(s\* ± w) has the sign of ±1 since w > 2g₁S/γ. So F has a unique zero
     s₀ ∈ J′, with |s₀ − s\*| ≤ 2g₁S/γ.
   - By Taylor, 0 = F(s₀) = g′(s\*)(s₀ − s\*) + ½g″(ξ)(s₀ − s\*)² + h(s\*) + h′(ξ′)(s₀ − s\*).
   - So g′(s\*)(s₀ − s\* − ℓ) = −½g″(s₀ − s\*)² − h′(s₀ − s\*), which gives |s₀ − s\* − ℓ| ≤ C_ℓ.
5. **Where the crossing can happen.**
   - For s_t ∈ J′ with s_t < s₀ − 2B/γ: F(s_t) < −B, so G(z_t) < 0.
   - For s_t ≥ s₀ + 2B/γ: F(s_t) ≥ B, so G(z_t) ≥ 0.
   - The window condition guarantees a step in [s₀ + 2B/γ, s₀ + 2B/γ + ṡ] ⊂ J′.
   - Hence s_c ∈ [s₀ − 2B/γ, s₀ + 2B/γ + ṡ].
6. **Combine** with step 4. ∎

### 15.4 Corollary L3 (free training: χ is learning-rate invariant, and so is the lag to leading order)

**Setting.** The scale is itself trained with the same fixed-P gradient step, s_{t+1} = s_t + ηp_s·σ_t. Here p_s > 0
is the preconditioner on s, and σ_t := −∂_sL(z_t, s_t) is the driving force.

**Assumptions** on the tube: σ_min ≤ σ ≤ σ_max, with σ_min > 0, and σ is L_σ-Lipschitz in (z, s).

**Statement.**
- **(i)** χ_t := ṡ_t/(s_tηλ) = p_sσ_t/(s_tλ) holds identically, so η cancels. After the transient,
  |χ_t − χ̄(s_t)| ≤ p_sL_σ·2S/(s_tλ), with χ̄(s) := p_sσ(z\*(s), s)/(sλ). Here χ̄ is a function of s alone, with no η in
  it.
- **(ii)** (L4) holds with K_s = L_σ(2ΛD̄₁/λ + 1)·σ_max/σ_min², which is η-free.
- **(iii)** q = ṡ/(ηλ) ≤ p_sσ_max/λ is η-free. So every constant of L1–L2 (E, C_ℓ and B) stays bounded as η → 0.
  Their only ṡ-dependence, through D̄₁ and Δv/(ηλ) = K_s·q·(2w + ṡ), is bounded.
- **(iv)** Hence r = κχ + O(χ²) + τ, with the O(χ²) constant η-free and 0 ≤ τ ≤ ṡ/s\* = O(η). As η → 0 the lag tends to
  κχ̄ + O(χ̄²), not to 0. This is the invariance observed in §13.6 (R4).

**Proof.**
- **(i)** χ_t = ηp_sσ_t/(s_tηλ). Then |σ_t − σ(z\*(s_t), s_t)| ≤ L_σ|δ_t| ≤ L_σ·2S (L1).
- **(ii)** |ṡ_{t+1} − ṡ_t| = ηp_s|σ_{t+1} − σ_t| ≤ ηp_sL_σ(|z_{t+1} − z_t| + ṡ_t).
  - |z_{t+1} − z_t| = η|∇L̃(z_t, s_t)| ≤ ηΛ|δ_t| ≤ 2ηΛS = 2(Λ/λ)D̄₁ṡ.
  - With ṡ ≤ ηp_sσ_max and ṡ_t ≥ ηp_sσ_min:
    |ṡ_{t+1} − ṡ_t| ≤ ηp_sL_σ(2ΛD̄₁/λ + 1)ṡ ≤ L_σ(2ΛD̄₁/λ + 1)(σ_max/σ_min²)·ṡ_t².
- **(iii)** Immediate.
- **(iv)** Theorem L2. ∎

### 15.5 Scope, and where it fails

**What is covered and what is not:**
- **Adam.** The theorems need a fixed P, so Adam is covered only with v̂ frozen. §13.6 shows that v̂ moving on the
  relaxation time breaks the law in forced ramps.
- **Momentum.** Not covered. §13.3(iii) argues that the steady lag is unchanged, but that is not proved here.
- **G must be C².** The placement gap is a min/max of extrema, so L2 applies only while the active pair is unique and
  the extrema are nondegenerate.
- **Near a fold.** λ → 0, so q = ṡ/(ηλ) blows up and the hypotheses E ≤ S and h′_max ≤ γ/2 fail. §14 takes over there.

**How tight the bounds are.**
- The constants are explicit but conservative. In the toy check, the lag error is 0.7–13% of the bound.
- For the sine family the constants would have to be enclosed on a tube around each branch. That has not been done,
  so the theorem is **not** yet a certified statement about the a = 1.30–1.60 runs. It is the exact form of the law
  those runs were compared against.

### 15.6 Numerical checks (`theory_checks_lag.json`; not part of the proof)

**Toy branch:** z\*(s) = (0.5 sin s, 0.3s), A(s) = diag(1 + 0.2s, 3), a cubic term (0.3/3)(z₁ − z₁\*)³, a nonlinear G,
and s\* = 2. The constants were computed on the tube ρ = 0.4, s ∈ [1, 3]: λ = 0.96, Λ = 3, M = 0.6, D₁ = 0.583,
D₂ = 0.5, H₁ = 0.2, g₁ = 1.61, g₂ = 0.2, γ = 0.292.

- **L1.** 10 linear ramps (η = 0.25 and 0.1, ṡ = 1e−3 … 6.25e−5) and 2 exponential ramps. All hypotheses held.
  - The deviation from the slaved displacement was at most 0.29 of E.
  - The lag error |s_c − s\* − ℓ| was 0.7–13% of the bound.
  - ℓ/s\* and κχ agree to 1e−6.
- **L2.** With the crossing interpolated between steps, (r − ℓ/s\*)/χ² = −0.933, −0.931, −0.930, −0.929 and −0.929
  at η = 0.1 as χ halves. The residual is O(χ²), as stated.
  - With first-step detection the residual is dominated by τ, which is less than one step.
- **L3.** Free training, ṡ = ηv(s): r_obs = 4.88e−4, 4.90e−4 and 4.91e−4 at η = 0.1, 0.05 and 0.025. They agree to
  0.6%, and the differences are the O(η) discreteness τ.

## 16. Local to global near the limit: the continued branch is the global minimiser for ε ≤ ε₀ = 1.4·10⁻¹² (Track 3 (1); 2026-09-28)

**Status.**
- **Theorem G: PROVED, every step checked.** It proves the **s-form** of the scaling law (5) as a statement about the
  **global** conditional minimiser, with an explicit ε₀.
- **The R-form (c₁) is not covered.** It also needs the unit Ĝ(a) to be attained at the tied corner for every
  ε ≤ ε₀. That is certified only at ε = 0, 10⁻⁴, 10⁻³ and 0.005 (§8), not for all small ε. This is stated in §16.5.
- **Producer:** `src/local_global.py` → `results/local_global.json`. One process, 7 minutes, mpmath interval
  arithmetic.
- **Inputs:** the committed limit certificates (`certv2_annulus_summary.csv`, `certv2_annulus_parts.csv`,
  `mn2_bounds.csv`, `first_order_c1.csv`).

### 16.0 Statement

**Theorem G.** Let 0 < ε ≤ ε₀ = 1.4·10⁻¹², a = 1 + ε, and the 800-point quadrature objective. For every A ∈ [0.66, 0.71]
(i.e. s = Aε^{−3/2}):
- **(i)** The conditional loss L\*_ε(·; A) has a unique global minimiser, up to the reflection p ↦ −p and the windings
  b₁ ↦ b₁ + 2πk. In rescaled coordinates it lies in the 0.05 box U around θ_c = (1.66858, 1.36972). It is the
  continued branch.
- **(ii)** The branch's gap changes sign exactly once in A ∈ [0.66, 0.71], at A_ε ∈ [0.68495, 0.68595]. It is
  negative before A_ε and positive after.
- **(iii)** Hence the global threshold in s-units, restricted to this A-range, is s_glob(ε) = A_ε·ε^{−3/2}, with
  **A_ε = A\*(1 + c_s·ε + O(ε²))**, A\* = 0.685445237…, and c_s ∈ [0.6621547, 0.6621550].
  - A_ε is the analytic continuation of the switch point of §8, by the implicit-function theorem at the
    Krawczyk-certified point with an invertible Jacobian. φ_ε is entire in ε.
  - The zero in (ii) is that point, by uniqueness in the switch box Z of §16.3.

**What ε₀ means.** ε₀ is set by the smallest certified limit margin (the outer exclusion, 1.0·10⁻⁴) against the
crudest perturbation bound (on K(24), where |σ| reaches 98.8). It is far below every tested a. At a = 1.01–1.12 the
global statements rest on the per-a certificates (§5, the c₁ tests), not on this theorem. The theorem closes (5)'s
s-form as an asymptotic statement about the global threshold.

### 16.1 Two lemmas

**Lemma G1 (pointwise remainder, all σ, all ε > 0).** With u = √ε·σ and φ_ε(σ) = (u − (1 + ε)sin u)/ε^{3/2}:
- |φ_ε − h| ≤ ε·m₀(|σ|), with m₀(S) = S⁵/120 + S³/6;
- |φ_ε′ − h′| ≤ ε·m₁, with m₁ = S⁴/24 + S²/2;
- |φ_ε″ − h″| ≤ ε·m₂, with m₂ = S³/6 + S.

*Proof.*
- φ_ε − h = [u − sin u − u³/6]/ε^{3/2} + [u − sin u]/√ε, and the two Taylor remainders of sin are bounded by |u|⁵/120
  and |u|³/6 for all real u.
- φ_ε′ − h′ = [1 − cos u − u²/2]/ε + (1 − cos u), with remainders u⁴/24 and u²/2.
- φ_ε″ − h″ = [sin u − u]/√ε + ε·sin u/√ε, with remainders |u|³/6 and |u|.
- Substituting u = √ε·σ gives the stated forms. ∎

**Lemma G2 (localisation at finite ε).** Let u₀ be the positive zero of g(u) = u − a·sin u, and β = arccos(1/a). Assume:
- (a) 4W(s, a) + u₀ < 2π − β, with W from the §5.0 lemma;
- (b) the branch-loss bound U_ε := max_A L\*_0(θ_c; A) + 0.71·ε·m₀(S_c) is below B(24), B_full and log 2.

Then every global minimiser of the finite-a conditional loss has |w₁| ≤ W. In rescaled coordinates relative to the
unique fold centre π + 2πk met by its t-range, it has (p, q) ∈ K(24).

*Proof.*
1. **|w₁| ≤ W.** For |w₁| > W, L\* > log 2 > U_ε (§5.0).
2. **At most one fold window.** For |w₁| ≤ W the t-range T = {w₁x + b₁ : x ∈ [−2, 2]} has length ≤ 4W. By (a) it meets
   at most one fold window [c − u₀, c + u₀], c = π + 2πk. If it meets none, f_a is increasing on T. Then the logit is
   monotone in x at every point, so L\* ≥ B_full.
3. **Ordering outside the window.** g is odd, decreasing on [0, β], increasing on [β, 2π − β], and g(u₀) = 0 with
   u₀ > β. So on the points with |u| ≥ u₀, where u = t − c:
   - u₁ < u₂ ≤ −u₀ and u₀ ≤ u₁ < u₂ are ordered, because g is increasing on [−(2π − β), −β] and [β, 2π − β], and
     T ⊂ [c − 2π + β, c + 2π − β] by (a);
   - u₁ ≤ −u₀ < u₀ ≤ u₂ gives g(u₁) ≤ g(−u₀) = 0 = g(u₀) ≤ g(u₂).
   So the logit is monotone in x on the retained points. Dropping the other points (nonnegative losses) gives
   L\* ≥ (1/n) × the best monotone logistic fit to the retained points.
4. **The window is short.** u₀ < 2√2·√ε. Indeed g(u) ≥ −εu + (1 + ε)u³/6 − (1 + ε)u⁵/120, and at u = 2√2·√ε this is
   ε^{3/2}·[−2√2 + (1 + ε)(2√2)³/6 − …] > 0, while g < 0 on (0, u₀).
   So the dropped points lie in σ ∈ (−2√2, 2√2), an x-window of length ≤ 4√2/|p|.
   - For |p| ≥ 24 the bound is B(24), the same exact PAVA value as in §3(a): dropping a sub-window only removes
     fewer points.
5. **What is left.** If |p| < 24 and some point has |σ| < 2√2, then |q| < 2√2 + 48. That is K(24).
6. By (b), every region other than K(24) has loss above the branch. ∎

### 16.2 Part (i): the four-link chain at finite ε

**Chain.** The limit chain of §3(c) holds on each of 40 A-sub-intervals of [0.66, 0.71] with θ_c fixed. Each link
transfers if its certified limit margin exceeds its perturbation. All bounds below are increasing in ε, so they hold
on (0, ε₀].

**Perturbation bounds**, from Lemma G1, |ℓ′| ≤ 1, σ′ ≤ ¼, |σ″| ≤ 1/(6√3), and |b\*_ε − b\*_0| ≤ max_i A|Δφ_i|:
- **Loss.** |L\*_ε(θ; A) − L\*_0(θ; A)| ≤ A·ε·m₀(S(θ)), where S(θ) = max_i |σ_i(θ)|. The profiled minimum over b moves
  at most by the sup of the change.
- **Gradient.** By the envelope theorem, ∂_pL\* = mean[(σ(z) − y)Aφ′(σ)x]. So
  |Δ∂_pL\*| ≤ 2Aε[m₁(S) + (S²/2 + 1)(A/2)m₀(S)], and the same without the factor 2 for ∂_q.
- **Hessian** of the (p, q, b) function at (p, q, b\*_ε):
  - (direct) each entry changes by at most
    (6√3)⁻¹Aεm₀d² + 2dAεm₁ + A²εm₀(S + εm₂) + 4Aεm₂, with d = max(1, 2A(S²/2 + 1 + εm₁));
  - (b-shift) plus a Lipschitz term in b, ((6√3)⁻¹d² + AS)·Aεm₀;
  - the Frobenius norm is at most 3 × the entry bound.
  - The Schur complement of a PD matrix is PD, so the profiled loss is strictly convex on U. That is what "at most
    one critical point in U" uses.

**Radii.** From θ_c = (1.668579, 1.369717):
- S_c = 4.707 (θ_c itself);
- S_U = 4.857 (0.05 box);
- S_R = 5.157 (0.15 box);
- S_K = 98.83 (K(24)).

**At ε₀ = 1.4·10⁻¹²** (`local_global.json`, `links`):

| link | certified limit margin | perturbation at ε₀ | slack |
|---|---|---|---|
| localisation (Lemma G2(b)) | B(24) − max branch loss = 0.38797 − 0.35804 | 0.71·ε₀·m₀(4.707) = 3.6·10⁻¹¹ | 0.0299 |
| 2, outer exclusion on K(24) ∖ 0.15 box | 1.0024·10⁻⁴ | 0.71·ε₀·(m₀(98.83) + m₀(4.707)) = 7.83·10⁻⁵ | 2.2·10⁻⁵ |
| 3, ring (gradient component) | 3.2786·10⁻³ | 6.2·10⁻¹⁰ | 3.28·10⁻³ |
| 4, PD on U | 0.047346 | 1.3·10⁻⁸ | 0.04735 |

- **Lemma G2(a):** W ≤ 1.0005 (s ≥ 0.66ε₀^{−3/2}, a ≤ 1 + ε₀; W is monotone in both), so 4W + u₀ = 4.0020 < 2π − β. The
  scaled value g(2√2√ε)/ε^{3/2} ≥ 0.943 > 0.
- **Why ε₀ is this value.** It is 0.8 × (the value at which the outer slack vanishes, 1.79·10⁻¹²), rounded down.
- **Hence (i):** the global minimiser exists (Lemma G2 bounds the domain), lies in K(24) (G2), lies in the 0.15 box
  (link 2), is not in the ring (link 3; it is a critical point), and is the unique critical point in the 0.05 box
  (link 4). ∎

### 16.3 Part (ii): one sign change, by parametric Krawczyk

**The ε-inflated enclosure.** Every quantity is evaluated in interval arithmetic with φ, φ′ and φ″ replaced by
h + [−1, 1]ε₀m₀(|σ|), h′ + [−1, 1]ε₀m₁ and h″ + [−1, 1]ε₀m₂. By Lemma G1 these enclose φ_ε for every ε ∈ [0, ε₀]. So a
Krawczyk test that passes holds for every such ε at once.

**The switch box.**
- The box is Z = (p, q, b, A) ∈ switch point ± (6, 6, 2, 5)·10⁻⁴, around the §8 point.
- The Krawczyk test for Φ = (∇_{p,q,b}L, φ(q − 1.2p) − φ(q + 0.8p)) passes on Z. So there is exactly one zero in Z for
  each ε ≤ ε₀.
- **The active pair is valid on Z (inflated).** The pair is inner x = 0.8, outer x = −1.2:
  - the other window endpoints are strictly dominated;
  - the inner σ-range stays above −√2 + 0.01;
  - the negative outer σ-range stays below √2 − 0.01;
  - the positive outer σ-range stays above √2 + 0.01.
  - φ_ε's critical points lie at ±β/√ε = ±√2(1 + O(ε)).
  - So φ_ε's window extrema are at the edges, and on Z the true gap equals that expression.
- Z's A-range is [0.684945, 0.685945].

**Branch boxes.**
- [0.66, 0.71] is covered by 110 closed A-sub-intervals: widths ≤ 2·10⁻³ away from Z, 2·10⁻⁵ inside Z, with bisection
  wherever the sign is undecided.
- On each, a parametric Krawczyk test for ∇_{p,q,b}L_ε(·; A) = 0, with A the sub-interval as an interval parameter
  and ε-inflation, passes on a box X_k (grown by ε-inflation). Each X_k lies in U.
- So X_k contains the unique critical point in U for every A in the sub-interval, i.e. θ_ε(A) by (i).
- **The gap over X_k** is enclosed over the continuous windows:
  - the lower bound is min over each window's σ-hull of the inflated φ (extrema at the ends or at ±√2), minus the
    inflated max over I;
  - the upper bound is φ at single window points.
- **Result** (`gap` in the JSON): all 110 boxes pass and lie in U.
  - Every sub-interval left of Z has gap < 0 (the largest upper bound is −2.5·10⁻⁴).
  - Every sub-interval right of Z has gap > 0 (the smallest lower bound is +7.8·10⁻⁴).
  - Every box for A inside Z lies inside Z's θ-range.

**Conclusion.**
- A ↦ G(θ_ε(A)) is continuous: θ_ε is the unique nondegenerate minimiser, hence C^∞ by the implicit-function
  theorem.
- It is negative on [0.66, 0.684945] and positive on [0.685945, 0.71].
- On Z's A-range its zeros are zeros of Φ in Z, because θ_ε(A) lies in Z's θ-range there. There is exactly one.
- So the sign changes exactly once. ∎

### 16.4 Part (iii)

- Φ(·; ε) is analytic in ε (φ_ε is entire in ε), and ∂Φ/∂(p, q, b, A) is invertible at the §8 point (Krawczyk).
- So the zero continues analytically: A_ε = A\* + A′(0)ε + O(ε²), with A′(0)/A\* = c_s ∈ [0.6621547, 0.6621550] (§8).
- By uniqueness in Z it is the zero of (ii). Finally s = A/ε^{3/2}. ∎

### 16.5 What is not proved, and what would enlarge ε₀

**The R-form, R_glob = R_glob^∞(1 + c₁ε + O(ε²)).**
- R = A·K(ε)/2 with K(ε) = Ĝ(a)/ε^{3/2}, the global supremum of the gap.
- **What is still needed.** For every ε ≤ ε₀, the maximiser of G̃_ε must be the tied corner. Two ingredients would
  give this:
  - a certified margin sup G₀ ≤ K − Δ outside a neighbourhood of ±corner;
  - the sharp-maximum argument on that neighbourhood, with interval gradients.
  Neither is done here.
- **What exists.** §12 proves that the maximiser lies in a compact set for ε ≤ 0.029. §8 certifies the corner at
  ε = 0 (branch and bound) and at 10⁻⁴, 10⁻³ and 0.005 (free branch and bound). So the R-form holds for the global
  switch in s-units, with a K(ε) whose global-ness is certified only at those ε.
- **The per-a certificates** cover the tested a directly: every registered c₁ bracket is a global certificate.

**Why ε₀ is tiny, and what would enlarge it.**
- The binding link is the outer exclusion. Its certified limit margin is 1.0·10⁻⁴, while |σ| reaches 98.8 on K(24),
  where m₀ ≈ 7.9·10⁷.
- A per-cell version would help: re-running the §3(c) outer exclusion with the perturbation 0.71·ε·m₀(cell σ-range)
  subtracted cell by cell. This is not done. It would lift ε₀ to about the near-branch limit,
  10⁻⁴/(0.71·2·m₀(5)) ≈ 10⁻⁶.
- Values of ε₀ comparable to the tested a need a finite-ε certificate with ε as an interval. That is a new search, not
  a perturbation of the limit certificate.

## 17. Small-scale compactness for the sine family, width 1 (Track 3 (3); 2026-09-28)

**Status, stated per part.**
- **PROVED, every step checked:**
  - **Lemma C1**, an exact decomposition of the conditional loss;
  - **Theorem C**: if Δμ attains its supremum, the conditional minimisers stay in an explicit compact set for all
    small s;
  - **Proposition C′**: if it does not, they leave every compact set.
- **Applied to the paper's three objects:**
  - **800-point population: compactness PROVED.** Δμ's supremum 2a is attained, exactly, at the aliased weight
    |w₁| = π/q.
  - **Continuous windows: compactness CHECKED.** It holds given that α\*_c is the global maximiser, which is
    computer-checked in floats, not proved.
  - **Training samples: compactness FAILS, proved.** For every sample with m = E_O x − E_I x ≠ 0 (all 50 seeds
    checked), the supremum is +∞.
- **Not covered:** width 2.
- **Producer:** `src/small_scale.py` → `results/small_scale_compactness.json` (the `ledger` block lists every number
  below). **Tests:** `tests/test_small_scale.py`.
- **What this means for Theorem 1.**
  - Its compactness hypothesis holds on the population and fails on samples.
  - Theorem 1's width-1 conclusion ("unplaced at small scale") does **not** need compactness. See §11's Proposition (i)
    for the population and Corollary C3 below for samples.

**Setting.** Width 1, φ_θ(x) = sign(w₂)·f_a(w₁x + b₁), s = |w₂|, and balanced classes (n_O = n_I), as in §10.
Δμ(θ) = mean_O φ − mean_I φ, and S := sup_θ Δμ(θ) ∈ (0, ∞].

### 17.1 Three lemmas

**Lemma C1 (exact decomposition).** For every θ and s > 0,

  **L\*(θ; s) = log 2 − (s/4)·Δμ(θ) + V_s(θ),  V_s(θ) := min_c mean_i ψ(s(φ_i − c)),  ψ(z) = log cosh(z/2) ≥ 0.**

*Proof.*
- log(1 + e^z) = z/2 + log(2cosh(z/2)), so ℓ(z, y) = log 2 + (½ − y)z + ψ(z).
- Put z_i = sφ_i + b. Then mean(½ − y_i) = 0 by balance, so the b-term drops.
- s·mean((½ − y)φ) = s(½·mean φ − ½·μ_O) = s(¼(μ_O + μ_I) − ½μ_O) = −(s/4)Δμ.
- Minimising over b is minimising over c = −b/s. ψ is convex and coercive, so the minimum is attained. ∎
- (Numerically: the largest error over 200 random (θ, s) is 1.8·10⁻¹⁵.)

**Lemma C2 (ψ).**
- ψ is even and increasing in |z|.
- ψ(z) ≤ z²/8, because (log cosh)″ = sech² ≤ 1.
- ψ(z) ≥ (5/48)z² for |z| ≤ 2. This uses log cosh u ≥ u²/2 − u⁴/12: the difference f satisfies f(0) = f′(0) = 0 and
  f″ = u² − tanh²u ≥ 0. Then z²/8 − z⁴/192 ≥ (5/48)z² for |z| ≤ 2.
- ψ(z) ≥ ψ(2) = log cosh 1 = 0.433781 for |z| ≥ 2.

**Lemma C3 (spreading).** Let N_τ be the largest number of data points in a closed x-interval of length 2τ, and
F := 1 − N_τ/n. Then for all c and all θ with τ|w₁| > a, at least F·n points satisfy |φ_i − c| ≥ τ|w₁| − a.
- *Proof:* |φ_i − c| ≥ |w₁x_i + b₁ − c| − a, and |w₁x_i + b₁ − c| < τ|w₁| only on an open x-interval of length 2τ. ∎
- **For the population with τ = 0.4:** N = 200 (exact sliding count; minimum spacing 0.00401), so F = 0.75. The
  continuous windows have uniform density 0.3125, so F = 0.75 there too.

### 17.2 Theorem C (attainment implies compactness, explicitly)

**Statement.** Suppose S is attained at θ₁, and let V₁ = Var(φ_{θ₁}). Put

  **s₁ := √(8Fψ(2)/V₁),  W_c := (a + √(1.2·V₁/F))/τ.**

Then for every 0 < s < s₁, every conditional minimiser θ_s satisfies |w₁| ≤ W_c. With b₁ mod 2π this is a compact set
that does not depend on s. So Corollary 1 of §10 applies: limit points maximise Δμ, and §10's (ii) applies when its
growth hypothesis holds.

**Proof.**
1. **Competitor bound.** By C1 and ψ ≤ z²/8 (with c = mean φ₁), L\*(θ₁; s) ≤ log 2 − (s/4)S + (s²/8)V₁.
2. **Minimiser bound.** A minimiser satisfies log 2 − (s/4)Δμ(θ_s) + V_s(θ_s) ≤ L\*(θ₁; s), and Δμ(θ_s) ≤ S. So
   V_s(θ_s) ≤ s²V₁/8.
3. **Spreading.** If τ|w₁| > a, let ζ = s(τ|w₁| − a). By C3 and the monotonicity of ψ, V_s ≥ F·ψ(ζ).
4. **Case ζ ≥ 2.** Then Fψ(2) ≤ s²V₁/8, which contradicts s < s₁.
5. **Case ζ < 2.** Then F(5/48)ζ² ≤ s²V₁/8, i.e. (τ|w₁| − a)² ≤ 48V₁/(40F) = 1.2V₁/F. ∎

### 17.3 Proposition C′ (non-attainment implies escape) and Corollary C3

**Proposition C′.** If S is not attained (including S = +∞), then for every W there is s_W > 0 such that every
conditional minimiser at s < s_W has |w₁| > W.

*Proof.*
1. On {|w₁| ≤ W} (b₁ mod 2π, both signs of w₂), Δμ is continuous on a compact set, so its maximum M_W is attained.
   Hence M_W < S.
2. Pick θ′ with η := Δμ(θ′) − M_W > 0.
3. For |w₁| ≤ W: L\* ≥ log 2 − (s/4)M_W, since V_s ≥ 0.
4. L\*(θ′; s) ≤ log 2 − (s/4)Δμ(θ′) + (s²/8)Var(φ_{θ′}), which is smaller than that once s < 2η/Var(φ_{θ′}). ∎

**Corollary C3 (unplaced at small scale, without compactness).**
- If S is not attained, every conditional minimiser at small s has |w₁| > a/1.4. So it is unplaced for the
  window gap G, by §11 Step 2, whose four points are window endpoints and do not depend on the sample.
- If S is attained, Theorem C applies instead.

### 17.4 The three objects

**800-point population: S = 2a is attained; compactness PROVED.**
- Every x is n·q with q = 0.4/79401: inner n even, outer n odd. This is the exact lattice of the linspace
  definition; the float values agree with n·q to 4.4·10⁻¹⁶.
- So at α = π/q = 623,613.996: D(α) = −2 exactly, and with sin b₁ = −1 (w₂ > 0), Δμ = 2a = S. Here |D| ≤ 2 always,
  and Δμ = a·sin b₁·D(w₁) on symmetric data (§11 Step 1).
- V₁ = α²·Var(x) + a²: cos αx_i = ±1 by class, and Cov(x, cos αx) = 0 by symmetry. Var(x) = 1.414136.

| a | V₁ | W_c | s₁ |
|---|---|---|---|
| 1.30 | 5.4995·10¹¹ | 2.34510·10⁶ | 2.175·10⁻⁶ |
| 1.60 | 5.4995·10¹¹ | 2.34510·10⁶ | 2.175·10⁻⁶ |

- (All nine a in the JSON, 1.02–1.60, give W_c = 3.76α.)
- **The limit object is the alias, not α\*.** By Corollary 1 the s → 0 limit points maximise Δμ, so they sit on the
  aliases. Among the aliases, Var is smallest at |w₁| = π/q (compare §10.1(ii)). The alias beats θ\* only below
  s ≈ 2·10⁻¹² (`theorem1_checks.json`).
- The alias is unplaced, as θ\* is, so the width-1 criterion's verdict is unchanged. This refines
  `theorem1_hypotheses.md`: the compactness hypothesis is now **proved** for the population. In D.1 it was "not proved".
- **Float caveat.** The float-rounded data are dyadic rationals, so their D is periodic too and S is attained. But
  the attaining point, and so W_c for the float objective, has not been computed. The theorem above is for the exact
  linspace population.

**Continuous windows (the idealised population measure): compactness CHECKED.**
- S = a·|D_c(α\*_c)| is attained at α\*_c = 1.7922917, given the computer-checked global-maximiser scan
  (`theorem1_checks.json`, float Lipschitz bounds). F = 0.75.

| a | V₁ = Var(φ\*_c) | W_c | s₁ |
|---|---|---|---|
| 1.30 | 5.6586 | 10.77 | 0.678 |
| 1.60 | 6.2344 | 11.90 | 0.646 |

- So for s < 0.64 every conditional minimiser has |w₁| ≤ 12. It is proved conditional on the maximiser scan.

**Training samples: compactness FAILS, proved whenever m ≠ 0.**
- A sample is not symmetric, so the linear part contributes w₁·m to Δμ, with m = E_O x − E_I x. Hence S = +∞, which
  is not attained.
- All 50 seeds (0–49, `make_data(200, seed)`) have m ≠ 0: min |m| = 0.0032, median 0.075.
- By C′ the minimisers leave every compact set as s → 0.
- By C3 they are unplaced there, so Theorem 1's conclusion holds on samples, but not through Corollary 1.
- **Illustration (seed 0; not part of the proof).** The pure ramp w₁ = m/(s·Var x) has a lower conditional loss than
  θ\* at s = 10⁻³ and 10⁻⁴ (w₁ = 58 and 585), though not at s = 10⁻².

### 17.5 Scope

- **Width 2 is not covered.** Directions with ṽᵢ → 0 and αᵢ → ∞ keep φ bounded, so Lemma C3's spreading argument does
  not bound the parameters. §10.1's width-2 selection keeps its compactness assumption.
- **Non-balanced data** would change C1 (the b-term no longer drops). Every object used in the paper is balanced.

## 18. The next-order term of the fold-passage delay, from the normal form (Track C; 2026-10-09)

**Status, stated per part.**
- **§18.2–18.4 (the expansion through O(ε)): derived by formal matched asymptotics, every algebraic step checked; it is
  NOT a theorem** (no remainder estimate is proved). It is checked against direct numerical integration of constructed
  normal forms, flow and map, where the error divided by ε goes to 0 and every dropped term leaves an O(1) error/ε
  (`tests/test_fold_next_order.py`).
- **§18.5–18.6 (Phase 2A's landscape): a PREDICTION computed from frozen landscape quantities only** (`src/fold_next_order.py
  predict` → `results/fold_next_order.json`). No run, observation or score file is read. It is committed before any
  comparison. The 2A data had been seen before this derivation began (the post hoc fit r = A ε^{2/3} + C ε ln(1/ε),
  `results/phase2a_posthoc.md`), so the comparison that follows in a later commit is labelled "derived after the data
  were seen".
- The reduction of the n-dimensional map to the planar system (§14.3) is used as there: derived, not proved.

### 18.1 Setting

- After the centre-manifold reduction of §14.3, take coordinates x along the fold's null vector e (|e| = 1 in the metric
  P = I of the fast coordinates) and σ = s − s_F. Orient e so that escape is toward x → +∞.
- One training step is the map
  - x_{n+1} = x_n + F(x_n, σ_n), F = pσ + qx² + f₃x³ + f₁₁xσ + (higher order),
  - σ_{n+1} = σ_n + ρ·G(x_n, σ_n), G = g₀ + g₁x + (higher order),
  - with p, q, g₀ > 0. The branch x = −(−pσ/q)^{1/2} attracts for σ < 0, and ρ is the slow rate.
- **The coefficients, in landscape terms** (η the fast step, H the fast Hessian at the fold, T = ∇³L, Q = ∇⁴L,
  H⁺ the pseudo-inverse on e⊥, b = ∂_s∇L):
  - q = −η·½T[e,e,e], p = −η·eᵀb (these are §14.1's c′, m′ with the escape orientation; pq = η²|m′c′|);
  - f₃ = −η(Q[e,e,e,e]/6 − ½T_eeᵀH⁺T_ee), f₁₁ = −η(∂_sH[e,e] − T_eeᵀH⁺b), with T_ee = T[e,e,·]. These are the cubic
    coefficients of the centre-manifold reduced gradient: the centre manifold is w = −H⁺(½T_ee x² + bσ) + …, and
    eᵀH w = 0 at the fold, so w enters F first through T[e,e,w]·x.
  - g₀ = ṡ/ρ at the fold point; g₁ = eᵀ∇G.
  - For Phase 2A's scale-only rule, G = −3η∂L/∂s (s = Σ|v| = √3·â·v; â·v moves with step ηρ). Then
    g₁ = −3η eᵀ∂_s∇L = **3p**: the coupling of x into ṡ is the same mixed derivative as the coupling of s into ẋ.
- **The crossing event** is a finite threshold, not the blow-up: the first step at which an observable (2A: ρ₂ ≥ q)
  is reached. Along the escape this happens at some x_q = O(1).

### 18.2 The inner problem at first order

- **Scaling.** Use σ as the independent variable: dx/dσ = F/(ρG). Set x = λX, σ = μT with
  μ = (g₀ρ)^{2/3}(pq)^{−1/3} and λ = (g₀ρp)^{1/3}q^{−2/3}. Then μλq = ρg₀ and pμ = qλ², and
  - dX/dT = T + X² + λ·(A X³ + B X T) + O(λ²),
  - **A = f₃/q − g₁/g₀, B = f₁₁/p − g₁/g₀** (flow). The −g₁/g₀ comes from 1/G = (1 − (g₁/g₀)x + …)/g₀ multiplying T + X².
  - Every other term is O(λ²) relative: x²σ, σ², g₂σ, x⁴.
- **Zeroth order** is Theorem F: X₀ = Ai′(−T)/Ai(−T) = −u′/u, u = Ai(−T).
- **First order.** X = X₀ + λX₁ with X₁′ = 2X₀X₁ + AX₀³ + BTX₀. The integrating factor is u⁻², so
  - X₁(T) = u(T)⁻² ∫_{−∞}^{T} u²(AX₀³ + BtX₀) dt.
  - This is the unique solution without the homogeneous part u⁻², which grows like exp((4/3)|T|^{3/2}) as T → −∞.
    So it is the one attached to the attracting slow manifold.
- **Near the blow-up** T = Ω₀ − τ, τ ↓ 0. Here u = kτ(1 + O(τ²)) with k = Ai′(a₁) = 0.70121…, and X₀ = 1/τ + O(τ).
  - So u²X₀³ = k²/τ + O(τ): the A-integral diverges logarithmically. The B-integral converges.
  - Hence ∫ = −Ak² ln τ + k²c₁ + O(τ²), with **c₁ = A·c_A + B·c_B**:
    - **c_A** = lim_{τ→0}[∫_{a₁+τ}^{∞} Ai′(z)³/Ai(z) dz + k² ln τ]/k² = **−1.0087907567050538** (mpmath, 30 digits;
      the integrand minus k²/(z − a₁) is integrated on [a₁, a₁ + 1]).
    - **c_B** = −∫_{a₁}^{∞} z·Ai(z)Ai′(z) dz/k² = **1/2 exactly**. The proof: ∫ z Ai Ai′ dz = ½zAi² − ½∫Ai² dz, and
      ∫Ai² dz = zAi² − Ai′², so ∫ z Ai Ai′ dz = ½Ai′². Evaluated from a₁ to ∞ this is −½k².
  - Inverting X = 1/τ + λ(−A ln τ + c₁)/τ² + … gives the inner time at which X is reached:
    - **T(X) = Ω₀ − 1/X − λ(A ln X + c₁) + O(λ², X⁻³)**, for 1 ≪ X ≪ 1/λ.
  - **The log is local.** Near the blow-up, dX/dT ≈ X²(1 + λAX), whose time to reach X is −1/X − λA ln X + const.
    The cubic part of the escape field sets it, not the far past.

### 18.3 The escape region and the matching

- For x = O(1), σ stays within O(μ) of 0 during the escape. To O(ρ),
  σ_q = σ_m + ρ·Σ_{steps from x_m to the threshold} G, evaluated with σ frozen at 0.
  - The σ-dependence of F there is relatively O(μ/x²). It matches the inner region's O(X⁻³) terms and contributes at
    O(ρμ) only.
- For a flow, Σ G becomes ∫_{x_m}^{x_q} G/F(x, 0) dx. Its small-x_m behaviour is g₀/(q x_m) + (g₀A/q) ln x_m + J + o(1).
  The 1/x coefficient of G/F is (g₁ − g₀f₃/q)/q = −g₀A/q, the same A as the inner one.
- This defines the **regularised escape sum**
  **J = lim_{x_m→0}[Σ_{x_m → threshold} G − g₀/(q x_m) − (g₀A/q) ln x_m]**.
  It is the only global quantity: it depends on the whole escape path to the threshold and on the observable.
- **Matching** at an intermediate x_m = λX_m, with the inner σ_m = μT(X_m):
  - the 1/x_m terms cancel, since μλ/x_m = ρg₀/(q x_m);
  - the ln x_m terms cancel, since μλ = ρg₀/q.
  - What remains:

  **σ_q = Ω₀μ + (ρg₀/q)(A ln λ − c₁) + ρJ + o(ρ).**

- Writing ln λ = ⅓ ln ρ + ln λ̂ with λ̂ = (g₀p)^{1/3}q^{−2/3}:

  **σ_q = Ω₀(g₀ρ)^{2/3}(pq)^{−1/3} + K·ρ ln(1/ρ) + D·ρ + O(ρ^{4/3} ln²ρ),**
  **K = −A g₀/(3q),  D = (g₀/q)(A ln λ̂ − A c_A − B/2) + J (+ g₀/2 for a map, §18.4).**

- **Agreement with [KS01].** Remark 2.11 states that the next term is O(ε ln ε), referring to [MR80]. Here that term is
  K·ρ ln(1/ρ), with **K set by the local cubic data alone**. The O(ε) term is not local: J carries the escape map to
  the threshold.

### 18.4 The discrete step (gradient descent)

- The map is the time-1 flow of the modified field F̃ = F − ½(F_xF + F_σ·ρG) + O(F²). Two terms matter at this order.
  - −½F_xF = −q²x³ − pq·xσ + …, so **f₃ → f₃ − q² and f₁₁ → f₁₁ − pq** in A and B.
  - −½F_σρG = −½pρg₀ is a constant forcing, equivalent to σ → σ − ½ρg₀. It adds **+½g₀ to D**.
  - The G-equation's correction −½ρG_xF is O(λ²) relative and is dropped.
- In the escape region, J is evaluated with the map itself (the frozen-σ step count). Its regularisation uses the map's
  A, and the Euler–Maclaurin boundary terms are part of J.
- **The threshold is detected at an integer step.** The first-crossing rule adds a jitter of at most one step's
  ρG(x_q) to σ_q. That is ≤ 0.02 % of 2A's Dρ, and it is reported, not modelled.
- **Hyperbolic modes enter at no lower order.** At the fold eᵀHw = 0. The O(ρ) lag of the fast modes behind the centre
  manifold therefore enters F only multiplied by x or σ, which is O(λ²) relative. Its size is ≈ (inner rate qλ)/λ₂
  times the first-order terms, which is large when λ₂ is small (§18.6).

### 18.5 What sets each coefficient

| order in σ_q | coefficient | set by |
|---|---|---|
| ρ^{2/3} | Ω₀(g₀)^{2/3}(pq)^{−1/3} | the fold constant pq = η²\|m′c′\| and the drift g₀ (§14) |
| ρ ln(1/ρ) | K = −A g₀/(3q) | local: the reduced cubic f₃ (Q₄ and T_eeᵀH⁺T_ee), the drift slope g₁ (= 3p for the scale-only rule), q; discrete −q² |
| ρ | D = (g₀/q)(A ln λ̂ − A c_A − B/2) + J + g₀/2 | local: A, B (f₁₁ via ∂_sH[e,e] and T_eeᵀH⁺b), λ̂; universal c_A, c_B; **global: J (escape to the threshold)**; discrete ½g₀ |

- Nothing at these orders is left undetermined. Every coefficient is a derivative of the loss at the fold, a universal
  Airy constant, or J. J is a deterministic, regularised escape computation on the frozen landscape, not a fitted
  constant.
- In r and ε units (r = σ_q/s_F, ε = (ρg₀/s_F)/(ηΛ_F), Λ_F = (|m′c′|s_F)^{1/2}, ρ = κε with κ = s_FηΛ_F/g₀):
  r = Ω₀ε^{2/3} + K_r ε ln(1/ε) + D_r ε, with K_r = Kκ/s_F and D_r = (D − K ln κ)κ/s_F.
- **The local exponent** is d ln r/d ln ε = d ln σ_q/d ln ρ = [⅔Ω₀μ + Kρ(ln(1/ρ) − 1) + Dρ]/σ_q, since ε ∝ ρ exactly.
- **Checks** (`tests/test_fold_next_order.py`): two polynomial flows and a map with η = 0.25, integrated to 1e-13. For
  the flow case (p, q, f₃, f₁₁, g₀, g₁) = (1, 1, 0.3, −0.4, 1, 0.5), x_q = 1:
  - error/ε at ε = 1e-3, 1e-4, 1e-5, 1e-6 is 0.0177, 0.0070, 0.0023, 0.0002 (o(1), falling like ε^{1/3});
  - the leading term alone leaves error/ε −0.324, −0.181, −0.033, +0.119. This drifts by K ln 10 = 0.1535 per decade,
    the log coefficient measured.
  - **FAIL cases:** dropping c₁, flipping K, using flow A, B on a map, or dropping the half step each leave an O(1)
    error/ε. The half step leaves exactly ½g₀.

### 18.6 Phase 2A's landscape: coefficients and the PREDICTION (before comparison)

**Inputs.** `results/phase2a/frozen.json` gives s_F, q and the ladder. The sb_fold continuation gives the M branch at
s_F − 0.002.

**Fold.** A Moore–Spence Newton solve in the 12 fast coordinates (W, c, v⊥â on the 3 active units, and b) gives
s_F = 4.7676894421067875. That is the frozen s_F to 4e−15. ρ₂ at the fold is 0.2801. The fast Hessian's eigenvalues
are 0, **λ₂ = 1.806e−4**, 3.96e−4, 4.75e−4, 5.77e−4, 3.7e−3, ….

| quantity | value |
|---|---|
| p = η\|m′\|, q = η\|c′\| | 1.06523e−3, 4.55732e−4 (\|m′c′\| = 4.8546e−7; frozen fit 4.7821e−7, −1.5 %) |
| Q₄/6, ½T_eeᵀH⁺T_ee → f₃ | 1.04793e−4, 1.35947e−4 → f₃ = 3.1154e−5 |
| ∂_sH[e,e], T_eeᵀH⁺b → f₁₁ | 6.3159e−4, 6.0927e−4 → f₁₁ = −2.2318e−5 |
| g₀ = −3η∂_sL, g₁ | 0.108120, 3.19569e−3 (= 3p to 1e−14) |
| A = f₃/q − q − g₁/g₀ | 0.068360 − 0.000456 − 0.029557 = **0.038348** |
| B = f₁₁/p − q − g₁/g₀ | −0.020952 − 0.000456 − 0.029557 = **−0.050964** |
| λ̂, c₁ = Ac_A + B/2 | 8.2157, −0.064167 |
| J (escape sum to ρ₂ ≥ q) | **494.06** (numerical ±0.03) |
| **K** | **−3.0326** |
| **D** = inner + J + ½g₀ | 34.38 + 494.06 + 0.054 = **528.50** |
| in ε units: K_r, D_r | **−0.0427, 7.32** |

- **J.** Frozen-s gradient descent (the scale-only rule at ρ = 0) starts at the centre-manifold point
  x = 0.005 and runs 442,408 steps until ρ₂ ≥ q, which is reached at x = 5.27.
  - The regularised tail sums at x_m = 0.4, 0.2, 0.1, 0.05, 0.025, 0.018, 0.0125, 0.009 converge linearly in x_m.
  - Two-point Richardson on (0.018, 0.009) gives 494.059, and (0.025, 0.0125) gives 494.030.
  - The run itself checks f₃ and g₁: (Δx − qx²)/x³ → f₃ to 2 %, and (G − g₀)/x = g₁ to 1e−4.
- **Consequences, stated before comparison.**
  - **The ε ln(1/ε) coefficient is small and NEGATIVE** (K_r = −0.043). At this order the derivation does NOT support
    reading the post hoc C = +0.80 as the ε ln ε term.
  - **The correction is carried by the O(ε) escape term.** About 93 % of D is J: the threshold ρ₂ = q lies far along the
    escape (x ≈ 5.3, ~4,570 steps of s-drift at g₀).
- **A priori limits** (computed from the landscape, per rate below):
  - δ₁ = λ|A| measures the first-order inner terms.
  - δ₂ = |f₄|λ²/q measures the second-order quartic term. f₄ ≈ −4.1e−4 is read off the escape run's increments; it is
    a diagnostic, not an input.
  - qλ/λ₂ measures the slow hyperbolic mode.
  - **δ₂ > δ₁ at every 2A rate, and qλ/λ₂ runs from 0.32 to 1.03.** So the formally neglected O(ρ^{4/3}) terms are not
    small at these rates. The expansion is expected to hold best at the slow end and not at all near 2⁻¹³ (qλ/λ₂ ≈ 1).

**Prediction table** (‡ = 2A's pre-observed rates). The ε column uses the exact Λ_F. The log term is Kρ ln(1/ρ)/s_F and
the linear term is Dρ/s_F.

| ρ | ε_F (exact Λ_F) | r: leading Ω₀ε^{2/3} | log term | linear term | r_pred | local slope d ln r/d ln ε | δ₁ = λ\|A\| | δ₂ = \|f₄\|λ²/q | qλ/λ₂ |
|---|---|---|---|---|---|---|---|---|---|
| 2^-13‡ | 1.8196e-03 | 0.034848 | -0.000700 | 0.013531 | 0.047680 | 0.7580 | 0.0156 | 0.1494 | 1.028 |
| 2^-14‡ | 9.0981e-04 | 0.021953 | -0.000377 | 0.006766 | 0.028342 | 0.7432 | 0.0124 | 0.0941 | 0.816 |
| 2^-15‡ | 4.5490e-04 | 0.013830 | -0.000202 | 0.003383 | 0.017011 | 0.7301 | 0.0098 | 0.0593 | 0.648 |
| 2^-15.125 | 4.1715e-04 | 0.013053 | -0.000187 | 0.003102 | 0.015969 | 0.7286 | 0.0096 | 0.0560 | 0.629 |
| 2^-15.25 | 3.8253e-04 | 0.012321 | -0.000173 | 0.002845 | 0.014993 | 0.7272 | 0.0093 | 0.0528 | 0.611 |
| 2^-15.375 | 3.5078e-04 | 0.011629 | -0.000160 | 0.002609 | 0.014078 | 0.7257 | 0.0090 | 0.0499 | 0.594 |
| 2^-15.5 | 3.2166e-04 | 0.010977 | -0.000147 | 0.002392 | 0.013221 | 0.7243 | 0.0088 | 0.0471 | 0.577 |
| 2^-15.625 | 2.9497e-04 | 0.010361 | -0.000136 | 0.002194 | 0.012418 | 0.7229 | 0.0085 | 0.0444 | 0.561 |
| 2^-15.75 | 2.7049e-04 | 0.009779 | -0.000126 | 0.002011 | 0.011664 | 0.7215 | 0.0083 | 0.0419 | 0.545 |
| 2^-15.875 | 2.4804e-04 | 0.009230 | -0.000116 | 0.001845 | 0.010958 | 0.7202 | 0.0080 | 0.0396 | 0.529 |
| 2^-16‡ | 2.2745e-04 | 0.008712 | -0.000108 | 0.001691 | 0.010296 | 0.7189 | 0.0078 | 0.0373 | 0.514 |
| 2^-16.125 | 2.0857e-04 | 0.008223 | -0.000099 | 0.001551 | 0.009675 | 0.7176 | 0.0076 | 0.0353 | 0.499 |
| 2^-16.25 | 1.9126e-04 | 0.007762 | -0.000092 | 0.001422 | 0.009092 | 0.7163 | 0.0074 | 0.0333 | 0.485 |
| 2^-16.375 | 1.7539e-04 | 0.007326 | -0.000085 | 0.001304 | 0.008545 | 0.7151 | 0.0072 | 0.0314 | 0.471 |
| 2^-16.5 | 1.6083e-04 | 0.006915 | -0.000078 | 0.001196 | 0.008032 | 0.7139 | 0.0070 | 0.0296 | 0.458 |
| 2^-16.625 | 1.4748e-04 | 0.006527 | -0.000073 | 0.001097 | 0.007551 | 0.7127 | 0.0068 | 0.0280 | 0.445 |
| 2^-16.75 | 1.3524e-04 | 0.006160 | -0.000067 | 0.001006 | 0.007099 | 0.7116 | 0.0066 | 0.0264 | 0.432 |
| 2^-16.875 | 1.2402e-04 | 0.005815 | -0.000062 | 0.000922 | 0.006675 | 0.7104 | 0.0064 | 0.0249 | 0.420 |
| 2^-17‡ | 1.1373e-04 | 0.005488 | -0.000057 | 0.000846 | 0.006277 | 0.7093 | 0.0062 | 0.0235 | 0.408 |
| 2^-17.125 | 1.0429e-04 | 0.005180 | -0.000053 | 0.000776 | 0.005903 | 0.7082 | 0.0060 | 0.0222 | 0.396 |
| 2^-17.25 | 9.5632e-05 | 0.004890 | -0.000049 | 0.000711 | 0.005552 | 0.7072 | 0.0059 | 0.0210 | 0.385 |
| 2^-17.375 | 8.7695e-05 | 0.004615 | -0.000045 | 0.000652 | 0.005222 | 0.7061 | 0.0057 | 0.0198 | 0.374 |
| 2^-17.5 | 8.0416e-05 | 0.004356 | -0.000042 | 0.000598 | 0.004912 | 0.7051 | 0.0055 | 0.0187 | 0.364 |
| 2^-17.625 | 7.3742e-05 | 0.004112 | -0.000038 | 0.000548 | 0.004622 | 0.7041 | 0.0054 | 0.0176 | 0.353 |
| 2^-17.75 | 6.7622e-05 | 0.003881 | -0.000036 | 0.000503 | 0.004348 | 0.7032 | 0.0052 | 0.0166 | 0.343 |
| 2^-17.875 | 6.2009e-05 | 0.003663 | -0.000033 | 0.000461 | 0.004091 | 0.7022 | 0.0051 | 0.0157 | 0.333 |
| 2^-18 | 5.6863e-05 | 0.003457 | -0.000030 | 0.000423 | 0.003850 | 0.7013 | 0.0049 | 0.0148 | 0.324 |

Two-point slopes between adjacent rates are in the JSON (`two_point`). Compute: one process, nice 15, one thread; peak
RSS 0.28 GB; 77 s. The memory gate was logged before the job.

### 18.7 Citations (records checked on Crossref, 2026-10-09)

- **[KS01]** Krupa & Szmolyan, *SIAM J. Math. Anal.* 33(2):286–314, 2001, doi:10.1137/S0036141099360919 (record
  matches). Used only as in §14.5:
  - Theorem 2.1 (the planar fold, leading O(ε^{2/3}));
  - Remark 2.11, "the next term in the expansion is O(ε ln ε)", which refers to [MR80].
  §18 derives that term's coefficient formally. It does not use a statement of [KS01] beyond the existence of the order.
- **[MR80]** Mishchenko & Rozov, *Differential Equations with Small Parameters and Relaxation Oscillations*, Springer US,
  1980, doi:10.1007/978-1-4615-9047-7 (record matches). **Not read; nothing here relies on it.**
- **R. Haberman**, "Slowly varying jump and transition phenomena associated with algebraic bifurcation problems",
  *SIAM J. Appl. Math.* 37(1):69–106, 1979, doi:10.1137/0137006 (record matches; abstract: interior nonlinear transition
  layers at the critical point, solutions that either grow algebraically or explode, "jump phenomena"). This is the
  classical source for this matched-asymptotics setting. **Not read beyond the abstract; nothing here relies on it.**
- Diminnie & Haberman (a saddle-centre passage) is not used.

### 18.8 Comparison with Phase 2A (DERIVED AFTER THE DATA WERE SEEN; prediction committed first in d31cedc)

`python -m src.fold_next_order compare` → `results/fold_next_order_compare.json`. It reads the committed prediction and
2A's committed s_obs per rate (`results/phase2a/observed.jsonl`). Nothing is fitted. Because ε ∝ ρ exactly, observed
slopes are Δ ln r_obs/Δ ln ρ.

**Size of the delay.**
- The leading term alone under-predicts: r_obs/r_lead runs from 1.099 to 1.170 over the 22 scored rates (1.225 at 2⁻¹³).
- With the next-order terms, r_obs/r_pred runs from 0.956 to 0.987 over the scored rates (0.895 at 2⁻¹³).
- The error shrinks monotonically toward the slow end: −1.3 % at 2⁻¹⁸.
- Measured in steps of drift at g₀, the observed excess over the leading term is 2,831 (2⁻¹³) rising to 3,958 (2⁻¹⁸).
  The predicted excess is 4,635 falling to 4,538.

**Local slope.** The prediction overshoots the observed two-point slope at every pair. The difference falls from 0.055
(2⁻¹³|2⁻¹⁴) to **0.0084 at 2⁻¹⁷·⁸⁷⁵|2⁻¹⁸**, where the prediction is 0.7017 and the observation 0.6934. Both lie above 2/3.

**The residual has the size and scaling of the neglected order (post hoc observation).** The residual
σ_obs − σ_pred, divided by ρ^{4/3}, is −3,926 to −4,013 across all 27 rates, almost constant over 5 octaves of ρ. This
fits the O(ρ^{4/3} ln²ρ) remainder of §18.3 with a large negative coefficient. It also fits the a priori diagnostics of
§18.6: δ₂ > δ₁, and qλ/λ₂ is up to 1.

**Reading.**
- The derivation closes through O(ε), and its prediction is about 4× closer to the data than the leading term at the
  slow end (−1.3 % vs +9.9 %).
- It does NOT reproduce the observed local exponents to better than 0.008–0.055 at these rates, because the next
  neglected order is numerically large here.
- The post hoc fit's C = +0.80 (`results/phase2a_posthoc.md`) is not the ε ln(1/ε) coefficient, which the normal form
  sets at −0.043. The data's curvature in ln r against ln ε comes from the O(ε) escape term (D_r = 7.32) together with a
  negative O(ε^{4/3}) remainder.
- No registered verdict is affected.

## 19. Track G, step G1: does the account apply to lazy-to-rich grokking? (derivation only, no compute; 2026-10-12)

**Status labels.** *Proved* means proved here, in full, from the stated model; the identities in (19.1) were also
checked symbolically (sympy; a seconds-long check, no training). *Argued* means a supporting argument that is not a
proof. *Literature* means a claim about published work, from the sources named, not re-proved here.

**Setting (G0).** Lyu, Jin, Li, Du, Lee and Hu, ICLR 2024, quoted from arXiv 2311.18817, whose numbering may differ from
the proceedings: the diagonal linear network f(θ, x) = ⟨u⊙u − v⊙v, x⟩, θ = (u, v) ∈ ℝ^d × ℝ^d, initialised at
θ(0) = α(1, 1); gradient flow on L_λ(θ) = L(θ) + (λ/2)‖θ‖² with the exponential loss L = (1/n)Σ_i exp(−y_i⟨w, x_i⟩),
w := u⊙u − v⊙v; separable data (sparse linear classification, n < d). Theorem 3.4 / Corollary 3.6: up to
(1−c)λ⁻¹ ln α the predictor is the max-L2-margin classifier; Theorem 3.5 / Corollary 3.7: by (1+c)λ⁻¹ ln α it is a
KKT point of min ‖θ‖² s.t. y_i f_i ≥ 1 (max-L1-margin). Both are limits α → ∞ with λ = Θ(α^{−p}).

### 19.1 Exact reduction (proved)

Write r := −∇_w L = (1/n)Σ_i y_i x_i e^{−y_i⟨w,x_i⟩}. Then u̇_k = (2r_k − λ)u_k and v̇_k = (−2r_k − λ)v_k, so u_k, v_k
stay positive and

1. **The product is an exact clock:** d(u_k v_k)/dt = −2λ u_k v_k, so u_k v_k = β(t) := α² e^{−2λt} for every k and
   every t, independent of the data and of the trajectory.
2. Since (u_k² + v_k²)² − (u_k² − v_k²)² = 4u_k²v_k², u_k² + v_k² = √(w_k² + 4β²), so ‖θ‖² = Σ_k √(w_k² + 4β²).
3. **The dynamics is an exact preconditioned gradient flow of a fixed-β landscape:**
   ẇ_k = 4r_k(u_k² + v_k²) − 2λw_k = −4 D_k ∂_k J_β(w), with D_k = √(w_k² + 4β²) and
   J_β(w) := L(w) + (λ/2) Σ_k √(w_k² + 4β²), evaluated at β = β(t).
4. J_β is L_λ restricted to the level set {u_k v_k = β ∀k}, written in w. It is strictly convex in w (L is convex,
   √(w² + c) is strictly convex for c > 0) and coercive for λ > 0, so it has a **unique** minimiser w*(β).
5. Limits (with Rosset, Zhu and Hastie 2004 for the regularisation paths; *literature*): for β ≫ |w|,
   J_β ≈ L + λβd + (λ/8β)‖w‖² (ridge, coefficient → 0 as β → ∞), whose minimiser's direction tends to the
   max-L2-margin classifier; as β → 0, J_β → L + (λ/2)‖w‖₁ (lasso), whose direction tends to the max-L1-margin
   classifier as λ → 0. These match Corollaries 3.6 and 3.7.

So the account's structure holds **exactly**, with one change: the slow "scale" is the conserved product β = u⊙v, not the
norm s = ‖θ‖. Its drift β̇/β = −2λ is imposed and exactly known, with no feedback from w.

### 19.2 The account's switch, as originally stated (scale s = ‖θ‖): exists, but is not grokking (proved / argued)

At fixed s, ‖w‖₁ ≤ Σ(u_k² + v_k²) = s², and every w with ‖w‖₁ ≤ s² is attained at norm exactly s (add equal mass to u_k
and v_k). So the fixed-s conditional minimiser is the L1-constrained problem min{L(w) : ‖w‖₁ ≤ s²} (*proved*). Along
s it runs from the single coordinate most correlated with the labels (small s) to the max-L1-margin direction
(large s), with breakpoints where coordinates enter (*argued*, by the KKT conditions of the constrained problem). The
fixed-s minimiser is never the kernel-phase *parameter* (*proved*: for separable data ∇L ≠ 0, so the minimum is on
‖w‖₁ = s², which forces u_k v_k = 0 for every k, while the kernel phase has u_k v_k = β ≫ 0); as a *predictor* it is
not the max-L2-margin classifier in this data (*argued*: it is L1-constrained and sparse at small s). In the setting's data (x uniform on {±1}^d, labels from a 3-sparse rule) the small-s minimiser
already uses a true coordinate. **The fixed-s switch exists but is a single-coordinate → sparse transition, at small
scale. It is not the kernel → rich transition, and training under weight decay does not move through it.** The kernel
phase is far from every fixed-s minimiser.

### 19.3 The fixed-β switch, and why the account cannot predict the grokking time from it

**Prediction it would give.** If training tracked w*(β(t)) quasi-statically, grokking would occur when β(t) passes β*,
the β at which w*(β)'s test error falls below the stated level: t_grok = (1/2λ) ln(α²/β*) + lag, i.e.
λ t_grok = ln α − ½ ln β* + λ·lag. β* comes from convex solves only (no restart ladder needed: the minimiser is unique).
This would add a computable O(1/λ) term to Lyu et al.'s leading (1/λ) ln α.

**Validity fails where it matters.**
- *Proved:* for the linearised flow δẇ = −4 D H_J δw (H_J = ∇²J_β at w*), the relaxation rates are the eigenvalues of
  the symmetric matrix A = 4 D^{1/2} H_J D^{1/2}. On the subspace S = {y : X D^{1/2} y = 0} (X the n × d data matrix;
  dim S ≥ d − n), the loss term vanishes and yᵀAy = Σ_k y_k² · 8λβ²/(w_k² + 4β²) ≤ 2λ‖y‖². By Courant–Fischer, at least
  d − n relaxation rates are ≤ 2λ at every β. The drift rate of the slow variable is exactly |β̇/β| = 2λ. So the
  adiabatic ratio (drift rate / relaxation rate) is **≥ 1 on a subspace of dimension ≥ d − n, for every λ**. Making λ
  smaller does not help: both rates scale with λ.
- *Argued:* that subspace carries the transition. The kernel predictor lies in the row space of X; the sparse predictor
  does not, and the change in test predictions is a change in the components that the training data do not constrain.
  In the row-space directions relaxation is fast (rate ≈ 8β × the loss curvature in the kernel phase), which is why the
  training fit tracks; it is the generalising change that does not.
- Consequence: the lag in §19.3's formula is O(1/λ), the same order as the −½ ln β* term it would correct. The
  quasi-static prediction is therefore **not sharp**. An exact finite-α time does follow from the exact solution
  w_k(t) = 2β(t) sinh(4ρ_k(t)), ρ_k = ∫₀ᵗ r_k (from 19.1), but that is the mirror-flow analysis already used in this
  literature (hypentropy; Woodworth et al. 2020; Lyu et al.'s proofs), not a prediction of the account.

**Literature (relevant prior work, from the sources named in the G1 report):** Boursier, Pesme and Dragomir (NeurIPS 2025,
arXiv 2505.20172) prove, for general losses as λ → 0, a fast phase to the critical manifold followed at time ≈ 1/λ by a
Riemannian gradient flow minimising ‖θ‖₂, and attribute grokking to it; their validation is on regression, and their
assumption (convergence of the unregularised flow) does not hold for the exponential loss on separable data. Kumar,
Bordelon, Gershman and Pehlevan (ICLR 2024) attribute grokking to the lazy → rich transition. The slow phase is a drift,
not quasi-static tracking of a fixed-scale minimiser, consistent with 19.3.

### 19.4 Alternative 1: grokking as the onset of tracking (no)

- With the validity measure of 19.3, there is no onset: on S the adiabatic ratio is ≥ 1 at every t (*proved*), and
  in the row space it is small throughout (*argued*). Nothing switches on at the grokking time.
- With a norm-based measure (the slack ‖θ‖² − ‖w‖₁ = 2Σ_k min(u_k², v_k²) small against ‖w‖₁, i.e. θ near the fixed-s
  minimiser manifold), the slack of an inactive coordinate is ≈ 2β(t), so onset needs 2dβ(t) ≪ ‖w‖₁, giving
  t ≈ (1/2λ) ln(2dα²/‖w‖₁) = (1/λ) ln α + (1/2λ) ln(2d/‖w‖₁) (*argued*). This carries a + (1/2λ) ln d term, so it falls
  after the leading-order grokking time by about (1/2λ) ln(2d/‖w‖₁) (illustrative arithmetic only: ≈ 3.5/λ at d = 10⁴ if ‖w‖₁ ≈ 20, a value not computed here). It is not the grokking time; it is when the unused
  coordinates have decayed.
- **No sharp prediction beyond (1/λ) ln α.**

### 19.5 Alternative 2: small initialisation, no weight decay, staged coordinate learning (no new prediction)

- With λ = 0, β = α² is constant (19.1), and ẇ = −4D∇L is the hypentropy mirror flow. For α → 0, coordinates
  activate in stages.
- *Literature:* Pesme and Flammarion (NeurIPS 2023, arXiv 2304.00488) characterise, for regression, the visited saddles
  and the jump times through a LARS-like recursion; the times are of order ln(1/α). I have not verified a published
  classification (exponential-loss) analogue; for separable data the flow converges in direction to max-L1-margin
  (Woodworth et al. 2020; Moroshko et al. 2020).
- The account's premise fails here. There is no imposed slow parameter: the time-scale separation comes from the
  exponential escape from near-zero coordinates, not from slow drift through a landscape, and the fixed-s path's
  breakpoints (19.2) are crossed by the loss-driven growth of s itself. Lag corrections would be O(1) against the
  O(ln 1/α) leading term, which the saddle-to-saddle analysis already gives.
- **No sharp prediction not already implied by existing results** (for regression); for classification, at most a
  re-derivation of the leading order, unverified as new.

### 19.6 Conclusion of G1

1. The account's switch with s = ‖θ‖ exists but is not the grokking transition (19.2).
2. Lyu et al.'s setting maps exactly onto the account's form with the conserved product β = u⊙v as the slow variable
   (19.1), an exact and possibly useful reduction; the fixed-β minimiser does switch from kernel-like to sparse.
3. But the generalising change happens in directions whose relaxation rate is never faster than the drift
   (19.3, proved bound), so the account's quasi-static prediction of the grokking time is not sharp. Its correction to
   (1/λ) ln α is the same order as the lag it ignores.
4. Neither alternative gives a sharp, testable prediction beyond existing results (19.4, 19.5).

Per the program's rule, Track G stops here: no G2 exploration, no design page.

## 20. Track U, step U1: the rescaled loss to second order, the margin near the switch, and an explicit ε₀ (2026-10-15)

**Status (checkpoint 1; no training, no seeds, paper untouched).**
- **Lemma U1** (φ_ε to second order, with derivatives): **PROVED**. Spot-checked in interval arithmetic at 80 digits.
- **Lemma U2** (the profiled loss to second order, |R₂| ≤ ε²C): **PROVED**.
- **Margin near the switch** (task 2): read off the committed limit certificates. It is **certified** by the producer
  (float branch and bound with a 10⁻⁹ allowance) and checked independently in Arb.
- **ε₀ for the task as posed** (one margin against a uniform bound on the perturbation of the difference): **PROVED** given
  that margin. It gives **ε₀ = 1.5·10⁻¹²**, the same order as §16's 1.4·10⁻¹². §20.4 explains why.
- **Theorem U1** (cell by cell): **ε₀ = 4.1·10⁻⁶**, about 2.9·10⁶ times §16's value. It has the same conclusions as
  Theorem G, for every A ∈ [0.66, 0.71]. Its status is mixed:
  - **certified, float with allowance**: the ε-uniform outer exclusion, a new branch and bound held to the producer
    standard of `certificates_v2.outer_exclusion`;
  - **proved**: the ring and PD links, from §16's first-order bounds;
  - **certified (interval arithmetic)**: the switch Krawczyk, with the second-order inflation of Lemma U1′.
  - **Not yet independently checked in Arb.** That is U3. §20.8 lists what it needs.
- **The binding link is now PD**, not the outer exclusion: §16's first-order Hessian bound limits it to ε < 5.17·10⁻⁶. The
  ring limit is 7.38·10⁻⁶. The outer link by itself certifies up to at least 6·10⁻³ (exploratory scan, §20.6).
- **A sharper localisation SET (finite ε, uniform in ε) of isotonic type is NOT available** (§20.2): B(P) falls below
  the branch loss at P ≈ 18, and the set that survives still has |σ| up to 75. What works instead is ε-uniform exclusion
  **cell by cell**, with exact bounds that do not use the expansion where |σ| is large (§20.5).
- **Producer:** `src/uniform_eps.py` → `results/uniform_eps/`:
  - `constants.json` (Lemmas U1–U2 constants, the margin, the uniform ε₀, the B(P) table, Lemma G2's ε-range);
  - `outer_eps_4.1e-6_k00.json` … `_k39.json`, with leaf files `…_k19_leaves.npz` and `…_k20_leaves.npz`;
  - `links_eps_4.1e-6.json`, `summary.json`, `outer_scan_19.json` (exploratory).
- **Runtime**, one process, nice 15, one thread, ≤ 0.5 GB:
  - constants 15 s;
  - outer exclusion 73 s for all 40 A-sub-intervals (1.5–2 s each);
  - links 452 s (the 110 Krawczyk boxes);
  - exploratory scan 13 s.
- **Tests:** `tests/test_uniform_eps.py` (11).

### 20.1 The rescaled conditional loss, exactly, and its expansion

**The exact loss.** f_a(π + u) − π = u + a·sin(π + u) = u − a·sin u. So with a = 1 + ε, u = √ε·σ and σ = px + q,
the rescaled feature and the loss are, exactly,

  φ_ε(σ) = [f_{1+ε}(π + √ε σ) − π]/ε^{3/2} = (u − (1 + ε) sin u)/ε^{3/2},
  L_ε(p, q, b; A) = (1/n) Σᵢ ℓ(A φ_ε(σᵢ) + b, yᵢ),   L\*_ε(p, q; A) = min_b L_ε.

φ_ε is entire in ε (§8). Its expansion is in integer powers of ε, so the remainder after the first-order term is
O(ε²), not O(ε^{3/2}).

**Lemma U1 (pointwise, all real σ, all ε > 0).** With h = −σ + σ³/6 and r = σ³/6 − σ⁵/120:
- φ_ε = h + εr + ε²ρ₀ with |ρ₀(σ)| ≤ m₂(|σ|) := |σ|⁷/5040 + |σ|⁵/120;
- (U1′) φ_ε′ = h′ + εr′ + ε²ρ₁ with |ρ₁| ≤ m₂₁ := σ⁶/720 + σ⁴/24;
- (U1′) φ_ε″ = h″ + εr″ + ε²ρ₂ with |ρ₂| ≤ m₂₂ := |σ|⁵/120 + |σ|³/6.

*Proof.* Use the Lagrange remainders of sin and cos, valid for all real u since |sin|, |cos| ≤ 1:
- |u − sin u − u³/6 + u⁵/120| ≤ |u|⁷/5040;
- |sin u − u + u³/6| ≤ |u|⁵/120;
- |1 − cos u − u²/2 + u⁴/24| ≤ u⁶/720;
- |cos u − 1 + u²/2| ≤ u⁴/24;
- |sin u − u + u³/6 − u⁵/120| ≤ |u|⁷/5040, and |sin u − u| ≤ |u|³/6.

Then:
- φ_ε = [u − sin u]/ε^{3/2} − ε sin u/ε^{3/2}. Expanding both terms gives h + εr, plus remainders bounded by ε²|σ|⁷/5040
  and ε·|u|⁵/(120ε^{3/2}) = ε²|σ|⁵/120.
- φ_ε′ = (1 − cos u)/ε − cos u = (σ²/2 − εσ⁴/24 + O₁) + (−1 + εσ²/2 + O₂), with |O₁| ≤ ε²σ⁶/720 and |O₂| ≤ ε²σ⁴/24.
- φ_ε″ = (1 + ε) sin u/√ε = (σ − εσ³/6 + O₃) + ε(σ + O₄), with |O₃| ≤ ε²|σ|⁵/120 and |O₄| ≤ ε|σ|³/6. ∎

*Checks.*
- 4,000 random (σ, ε) with σ ∈ [−100, 100] and ε ∈ [10⁻⁸, 0.6], in mpmath interval arithmetic at 80 digits: the largest
  ratio |remainder|/(ε²m₂) is 0.9999991.
- At 30 digits the remainder is lost to cancellation near ε = 10⁻⁸, so the check runs at 80.
- Tests: 1,500 + 1,000 random points at 60 digits for U1 and U1′.
- **Constructed fail:** m₂/2 fails at σ = 1, because the leading remainder there is ε²(σ⁷/5040 − σ⁵/120) = −0.00813ε².
  Dropping the εr′ term also violates the U1′ bound.

**Lemma U2 (the profiled loss to second order).** Fix θ = (p, q) and A. Let b₀ = b\*_0(θ; A) and zᵢ = A h(σᵢ) + b₀. Let
eᵢ = s(zᵢ) − yᵢ, with s the logistic function, so that mean eᵢ = 0. Define

  **L₁(θ; A) := A·mean_i eᵢ·r(σᵢ)**   (= ∂_ε L\*_ε at ε = 0, by the envelope theorem).

Then for every constant c, with mᵢ = m₂(|σᵢ|) and tᵢ = |r(σᵢ) − c| + ε mᵢ,

  −ε²[A·mean mᵢ + ¼A²·maxᵢ tᵢ·mean tᵢ]  ≤  L\*_ε − L\*_0 − εL₁  ≤  ε²[A·mean mᵢ + ⅛A²·mean tᵢ²].

So |R₂| ≤ ε²C(θ; A, ε) with **C = A·mean mᵢ + ¼A²·maxᵢ tᵢ·mean tᵢ**, which is nondecreasing in ε. Also
|L₁| ≤ A·mean|r(σᵢ) − c| for every c.

*Proof.*
1. **Constants are absorbed.** Let Δᵢ = A(φ_ε(σᵢ) − h(σᵢ)) = εA rᵢ + ε²A ρ₀ᵢ. Since b is minimised over,
   L\*_ε = min_b mean ℓ(zᵢ(b) + δᵢ) with δᵢ = Δᵢ − εAc, for any c. Also mean eᵢδᵢ = mean eᵢΔᵢ = εL₁ + ε²A·mean eᵢρ₀ᵢ,
   and |eᵢ| ≤ 1.
2. **Upper bound.** Take b = b₀. Then ℓ(z + δ) ≤ ℓ(z) + ℓ′(z)δ + δ²/8, since ℓ″ ≤ ¼.
3. **Lower bound.**
   - Convexity gives ℓ(z + δ) ≥ ℓ(z) + ℓ′(z)δ. So L_ε(b) ≥ L_0(b) + mean ℓ′ᵢ(b)δᵢ ≥ L\*_0 + mean eᵢδᵢ − ¼|b − b₀|·mean|δᵢ|.
     This uses L_0(b) ≥ L\*_0 and |ℓ′ᵢ(b) − ℓ′ᵢ(b₀)| ≤ ¼|b − b₀|.
   - The minimiser b_ε of the shifted problem is the root of the monotone F(b) = mean s(zᵢ(b) + δᵢ) − ȳ. Shifting
     every argument by at most ‖δ‖_∞ moves that root by at most ‖δ‖_∞, so |b_ε − b₀| ≤ ‖δ‖_∞.
4. **Collect.** |δᵢ| ≤ εA·tᵢ, and mean δ² ≤ ‖δ‖_∞·mean|δ|. ∎

*Checks.*
- At θ_c, 12 random θ in [0, 6] × [−8, 8], A ∈ {0.66, 0.6854, 0.71} and ε ∈ {10⁻⁴, 10⁻³, 10⁻²}, the exact L\*_ε satisfies the
  bound.
- **Constructed fail:** omitting L₁, i.e. |L\*_ε − L\*_0| ≤ ε²C, fails at θ_c for ε = 10⁻³, because L₁ = 0.083 there.

**Constants** (`constants.json`; floats with the 10⁻⁹ allowance, the producer standard).
- **At the branch centre θ_c**, A ∈ [0.68375, 0.68625]: L₁ = 0.08254 to 0.08292, and C ≤ 2.744 for ε ≤ 10⁻³. The branch
  therefore rises with ε at first order.
- **Uniform on each set**: S = sup|σ|; C here takes c = 0 and |rᵢ| ≤ r_max(S), with ε ≤ 10⁻³.

| set | S | m₂(S) | r_max(S) = max_{\|σ\|≤S}\|r\| | C (uniform) |
|---|---|---|---|---|
| θ_c | 4.707 | 29.4 | 2.771 | 23.8 |
| 0.05 box U | 4.857 | 35.2 | 3.427 | 29.7 |
| 0.15 box | 5.157 | 49.6 | 7.535 | 61.0 |
| K(24) | 98.83 | 1.84·10¹⁰ | 7.84·10⁷ | 3.6·10¹⁵ |

- **The closed form for r_max.** |r| = |σ|³|20 − σ²|/120. It has an interior maximum 12^{3/2}·8/120 = 2.7713 at σ² = 12,
  vanishes at σ² = 20, and increases beyond. A test checks this on a grid.

### 20.2 Localisation, uniformly in ε, and where a sharper localisation set fails

**Lemma G2 holds uniformly for ε ≤ 0.0462** (`G2_range`).
- Its condition (a) is 4W(s, a) + u₀ < 2π − β, with s ≥ 0.66ε^{−3/2} and u₀ < 2√2√ε. Both sides are monotone in ε: W
  increases (s falls and a rises) and 2π − β decreases. So the condition holds on all of (0, ε] once it holds at ε.
- On a grid of step 10⁻⁴ the last pass is ε = 0.0462: 4W + u₀ = 5.98441 < 5.98489. W is the §5.0 formula, evaluated
  in floats.
- Condition (b) at ε₀ is §20.5's localisation slack.
- **So for every such ε, every global minimiser lies in K(24) relative to the fold centre its t-range meets.**
- **The isotonic bounds are themselves exact for every such ε.** The dropped points have |σ| < u₀/√ε < 2√2 at every ε
  (G2 step 4), so B(P), B_full and the per-cell versions below need no expansion.

**A sharper localisation set does not follow, and here is exactly where it fails.**
- Exact PAVA (`B_table`) gives:

| P | 18 | 18.25 | 18.5 | 19 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|
| B(P) | 0.35296 | 0.35478 | 0.35660 | 0.36020 | 0.36731 | 0.37949 | 0.38797 |

- The branch upper bound near the switch is 0.35287 on k = 19 and 0.35260 on k = 20. So B(P) excludes |p| ≥ P only for
  P ≥ 18.0, and only barely: by 9·10⁻⁵ on k = 19. (Earlier exploration: B(16) = 0.336 and B(12) = 0.275 lie below the
  branch.)
- The set left, {|p| < 18, some |σᵢ| < 2√2}, has |q| < 2√2 + 36 and so **|σ| up to 4·18 + 2√2 = 74.8**. That is
  m₂ ≈ 2.6·10⁹ and r_max ≈ 1.9·10⁷.
- A q-dependent version, the drop window at its actual position (the per-cell PAVA of §20.5), does not give a smaller
  SET either. At (p, q) = (17.87, 32.90) and (17.99, 38.83), the isotonic bound on the retained points is below the
  branch value, while σ runs up to 69 and 75.
- The true limit loss there is 1,877 and 2,800, so these points are far from competitive. No monotone-fit bound sees
  this, because a monotone fit discards the size of the logits.
- **So the large-|σ| part of K(24) cannot be removed by a localisation set of this type.** It has to be handled region
  by region, with bounds that do not pay ε·|σ|⁵.

### 20.3 Task 2: the certified loss margin near the switch

The switch box Z of §16.3 has A-range [0.684945, 0.685945]. It meets the annulus sub-intervals k = 19 and 20, so
J_Z = [0.68375, 0.68625]. On these two sub-intervals the certified outer-exclusion margin is (`margin`):

| k | A | producer margin (lower end) | Arb re-check | upper end |
|---|---|---|---|---|
| 19 | [0.68375, 0.685] | 1.1035·10⁻⁴ | 2.755·10⁻⁴ | 4.7294·10⁻³ |
| 20 | [0.685, 0.68625] | **1.0301·10⁻⁴** | 2.706·10⁻⁴ | **4.7027·10⁻³** |

- **m ∈ [1.0301·10⁻⁴, 4.7027·10⁻³]**, with min over k = 19, 20 at each end. The margin is
  m = inf over θ ∈ K(24) ∖ (0.15 box) and A ∈ J of [L\*_0(θ; A) − L\*_0(θ_c; A)].
- **Source**: `certv2_annulus_parts.csv` rows 19–20 (`certificates_v2.outer_exclusion`), checked independently by
  `certificate_checks/outer_glob_annulus.json`, whose fresh Arb branch and bound gives ≥ 2.706·10⁻⁴.
- **The lower end is not a property of the problem.** It is the producer's stopping target (MARGIN = 10⁻⁴) plus slack:
  the search stops refining once every cell clears 10⁻⁴.
- **The upper end is attained.** It is a competitor the producer evaluated (outer_attained at A_c) minus the branch
  upper bound, plus 10⁻⁹. So the true margin is at least 1.0301·10⁻⁴ and at most 4.70·10⁻³.

### 20.4 Task 3 as posed: one margin against a uniform perturbation of the difference

**Decision rule** (`decide`, tests on constructed pass / fail / undecided inputs). The difference between a competitor
θ′ and the branch θ_c is

  Δ(ε) = L\*_ε(θ′) − L\*_ε(θ_c) ∈ [m_lo + ε d₁,lo − ε²c₂, m_hi + ε d₁,hi + ε²c₂],

where d₁ encloses L₁(θ′) − L₁(θ_c), the first-order shift of the DIFFERENCE, and c₂ = C(θ′) + C(θ_c). The verdict is:
- **pass** if the lower end is > 0;
- **fail** if the upper end is < 0;
- **undecided** otherwise.

**Uniform version.** Bound the competitor's L₁ over all of K(24) ∖ box and keep the branch's L₁ exact:
- L₁(θ′) ≥ −A·r_max(98.83) = −0.68625 × 7.840·10⁷ = −5.38·10⁷;
- L₁(θ_c) ≤ 0.08292;
- C_K ≤ A·m₂(S_K) + ¼A²(2r_max(S_K) + εm₂(S_K))² (conservative), which is 2.9·10¹⁵ at ε₀;
- C(θ_c) = 2.744.

Then D(ε) = 1.0301·10⁻⁴ − 5.38·10⁷·ε − (2.9·10¹⁵ + 2.7)·ε².
- **Root 1.9143·10⁻¹², so ε₀ = 1.5·10⁻¹²** (0.8 × root, 2 significant figures; decision "pass").
- The first-order term is 99.99% of the perturbation at the root. The ε² remainder is irrelevant here.
- With the Arb margin 2.706·10⁻⁴, the root is 5.03·10⁻¹² and ε₀ = 4·10⁻¹².

**Compared with §16's 1.4·10⁻¹².** The order is the same because the binding quantity is the same: a margin of order 10⁻⁴
against A·|σ|⁵/120 at |σ| = 98.8. The small differences are:
- (i) the margin near the switch, 1.0301·10⁻⁴, rather than the global minimum 1.0024·10⁻⁴;
- (ii) A ≤ 0.68625 rather than 0.71;
- (iii) r_max(S) = S³(S² − 20)/120 = 7.840·10⁷, where §16 bounds the perturbation of the loss by m₀(S) = S⁵/120 + S³/6
  = 7.934·10⁷. The two terms of r partly cancel;
- (iv) the branch's own first-order shift enters exactly (+0.083) rather than as 0.71·ε·m₀(4.707);
- (v) §16's 0.8 safety factor, which is used here too.

**Why including L₁ does not help in this form.** Including the first-order shift of the difference is necessary, and it
is included. But with one uniform margin, the competitor's L₁ can only be bounded by sup|r| over K(24). Only a
region-by-region bound can use the fact that cells with large |σ| also have large margins. That is §20.5.

### 20.5 Theorem U1: the cell-by-cell version, ε₀ = 4.1·10⁻⁶

**Theorem U1.** Let 0 < ε ≤ ε₀ = 4.1·10⁻⁶, a = 1 + ε, and the 800-point quadrature objective. Then for every
A ∈ [0.66, 0.71]:
- **(i)** L\*_ε(·; A) has a unique global minimiser, up to p ↦ −p and the windings b₁ ↦ b₁ + 2πk. It lies in the 0.05 box U
  around θ_c and is the continued branch.
- **(ii)** The branch's gap changes sign exactly once in [0.66, 0.71], at A_ε in Z's A-range [0.684945, 0.685945].
- **(iii)** Hence **s_glob(ε) = s_br(ε) = A_ε ε^{−3/2}**, with A_ε = A\*(1 + c_s ε + O(ε²)), A\* = 0.685445237… and
  c_s ∈ [0.6621547, 0.6621550]. Eq. (5)'s s-form therefore holds for the global threshold on (0, 4.1·10⁻⁶].
- The R-form is still not covered, for the reason given in §16.5.

**Proof.** The four-link chain of §16.2, with link 2 replaced and the §16.3 Krawczyk part re-run at ε₀.

**Link 1, localisation.**
- Lemma G2 holds at ε₀, since ε₀ ≤ 0.0462.
- The branch upper bound at ε₀ is max over the 40 sub-intervals of BU = bu₀ + ε₀·max(L₁,0) + ε₀²C. It sits below
  min(B(24), B_full, log 2) by **0.02993**.

**Link 2, ε-uniform outer exclusion (new).** We prove L\*_ε(θ; A) > L\*_ε(θ_c; A) for every θ ∈ K(24) ∩ {p ≥ 0} outside the
0.15 box, every A in the sub-interval, and every ε ∈ [0, ε₀].
- **Branch side.** L\*_ε(θ_c; ·) is convex in A for every ε (z is jointly linear in (A, b)). So its sup over the
  sub-interval is at an endpoint, where Lemma U2 bounds it by L\*_0 + εL₁ + ε²C.
- **Competitor side.** The search is the producer's quadtree from 0.25 cells. A cell is excluded if ANY of the following
  holds. Each test holds for every ε ≤ ε₀ (BU is the branch bound above, and τ = 10⁻⁷ sits on top of the 10⁻⁹ rounding
  allowance).
  - **(iso)** p_lo ≥ P_k with B(P_k) > BU + τ. Exact, by Lemma G2. Here P_iso = 18.0.
  - **(full)** every point has |σᵢ| ≥ 2√2 on the whole cell, and B_full > BU + τ. Exact: no point is dropped and the
    logit is monotone in x (G2 step 3).
  - **(pert)** D(0) > τ and D(ε₀) > τ, where D(ε) = lb₀ − bu₀ + ε(L₁,lo − L₁,bu) − ε²(C_cell + C_bu).
    - lb₀ is the producer's limit cell bound (§3(c) link 2, the same formula).
    - L₁,lo = max(centre value − Lipschitz terms, −A·mean|r − c|). The Lipschitz constants are
      |∂_θL₁| ≤ ½A²·maxⱼ|h′ⱼ∂σⱼ|·mean|rᵢ − c| + A·mean|r′ᵢ∂σᵢ| and |∂_A L₁| ≤ mean|r − c|·(1 + A·maxᵢ|hᵢ − c_h|).
      Both follow from ∂b\* being a weighted mean of −∂zᵢ, and mean s′(∂zᵢ + ∂b\*) = 0.
    - C_cell uses the cell's σ-ranges.
    - D is concave in ε, because C is evaluated at ε₀ and is nondecreasing. So its minimum on [0, ε₀] is at an end.
    - This is where **the first-order shift of the difference** enters. Near the box the competitor's L₁ is close to the
      branch's, and the centre-plus-Lipschitz bound sees that.
  - **(sep)** a class-1 outer window lies entirely below the inner window on the φ_ε scale, by a gap g > 0. Each point's
    φ_ε is enclosed by [min h − ε₀m₀, max h + ε₀m₀] over its σ-interval (Lemma G1). Then L\* ≥ π·softplus(A_lo g/2), which
    is §5.0's separation argument.
  - **(fav)** the favourable-logit profile: zᵢ ≤ (class 1) or ≥ (class 0) the corner of A_lo, A_hi times that φ_ε
    enclosure.
    - ℓ(·, 1) decreases and ℓ(·, 0) increases, so L\* ≥ min_b mean ℓ(z_fav + b).
    - The minimum is bounded below by the tangent at the left end of a bisection bracket (convexity).
    - This is exact in ε and needs no expansion. It is the test that disposes of cells where |σ| is large but the loss is
      far above the branch. Examples: (9.2, 10.8), where L\*_0 ≈ 92; or the band p ≈ 17.9, q ≈ 33.
  - **(iso-cell)** the cell's own dropped set (points that can reach |σ| < 2√2 on the cell), with exact PAVA on the
    retained points.
- **Result at ε₀ = 4.1·10⁻⁶** (`summary.json`): **all 40 sub-intervals are certified**.
  - At most 4 rounds and a peak of 39,072 cells, against a cap of 1.5M; 1,567,287 leaves in total.
  - Leaves by test: iso 375,661, full 690,828, pert 499,931, fav 867, sep 0, iso-cell 0.
  - The smallest D over pert-excluded cells is 8.4·10⁻⁷ (k = 34). On the switch sub-intervals it is 5.41·10⁻⁵ (k = 19)
    and 9.89·10⁻⁵ (k = 20).
  - These are stopping values, not margins: a cell is accepted once D > τ.

**Link 3, ring.** §16.2's gradient bound (Lemma G1, first order) against the certified ring margin 3.2786·10⁻³ (the
minimum over the 40 sub-intervals). The perturbation at ε₀ is 1.822·10⁻³, so the **slack is 1.456·10⁻³**. The bound
holds up to ε = 7.38·10⁻⁶.

**Link 4, PD.** §16.2's Hessian bound against λ_min ≥ 0.04735. The perturbation at ε₀ is 0.03753, so the **slack is
0.00981**. The bound holds up to **ε = 5.17·10⁻⁶. This is the binding link.**

**ε₀ rule.** ε₀ = 0.8 × min(ring limit, PD limit) = 0.8 × 5.17·10⁻⁶, rounded to 2 significant figures: **4.1·10⁻⁶**
(`eps0_matches_rule`).

**Part (ii)** (`links_eps_4.1e-6.json`, `switch`). This is §16.3 re-run at ε₀ with the **second-order inflation** of
Lemmas U1/U1′: φ ∈ h + [0, ε₀]·r + [−1, 1]ε₀²m₂, and likewise for φ′ and φ″ (`Inflated2`). It encloses φ_ε for every
ε ∈ [0, ε₀]. The gap's extrema allowance keeps Lemma G1's first-order term.
- The switch box Z (radii (6, 6, 2, 5)·10⁻⁴, as in §16.3) passes the 4-variable Krawczyk test.
- The active pair (inner 0.8, outer −1.2) is valid on Z. The critical-point clearance is 0.01 + 2√2ε₀.
- 110 A-boxes cover [0.66, 0.71]: 31 left of Z, 50 inside and 29 right. Every 3-variable Krawczyk test passes with its
  box in U.
  - The largest gap upper bound on the left is −2.30·10⁻⁴.
  - The smallest gap lower bound on the right is +7.36·10⁻⁴.
  - Every inside box lies in Z's θ-range.
- So the sign changes exactly once, as in §16.3.
- For comparison, the first-order inflation also passes the Z test at 4.5·10⁻⁶, and the second-order one up to at least
  3·10⁻⁵ (exploration). So this part does not bind.

**Part (iii)** is as in §16.4. ∎

### 20.6 Why ε₀ grew by 2.9·10⁶, and what now limits it

**Why ε₀ grew.**
- In §16 and §20.4, ONE margin (10⁻⁴, a stopping target) is set against the WORST perturbation over K(24)
  (A|σ|⁵/120 at |σ| = 98.8).
- Cell by cell, the cells with large |σ| have large margins, or are excluded exactly and uniformly in ε. That uses
  B(P), B_full and the per-cell PAVA, or the favourable-logit profile, none of which pays ε|σ|⁵.
- The expansion is used only where |σ| is moderate. There the first-order shift of the difference is small, because the
  competitor's L₁ is close to the branch's.
- Once the outer link stops binding, the next limits are the ring and PD links. Those still use §16's crude first-order
  bounds on S_R = 5.16 and S_U = 4.86.

**What the outer link alone reaches** (`outer_scan_19.json`, sub-interval k = 19; EXPLORATORY, not part of the theorem).

| ε | 10⁻⁴ | 10⁻³ | 3·10⁻³ | 6·10⁻³ |
|---|---|---|---|---|
| certified | yes | yes | yes | yes |
| seconds | 1.7 | 2.2 | 3.8 | 5.4 |
| cells by sep / fav / iso-cell | 2,764 / 67 / 0 | 8,508 / 463 / 0 | 8,767 / 1,947 / 0 | 7,912 / 3,253 / 127 |

- A run at 10⁻² was stopped after about 20 minutes without a verdict. The per-cell PAVA loop dominated.
- **Before** the sep and fav tests were added, ε = 10⁻⁴ FAILED. The live cells were at p ≈ 17.9, q ≈ 33 and p ≈ 18, q ≈ 39,
  just below P_iso. There L\*_0 ≈ 1,900–2,800, but the second-order term ε²C ~ ε²·4·10¹² is meaningless, because
  u = √ε σ ~ 0.7.
- That band is the large-|σ| obstruction of §20.2. It is closed by the exact favourable-logit bound, not by a sharper
  set.

**What would enlarge ε₀ further (U2).**
- **PD and ring at second order.** Re-certify the Hessian PD on U and the ring gradient with Inflated2-type enclosures,
  in interval arithmetic over sub-boxes, rather than §16.2's entry bounds. The ε-shifts are then ε·(∂_ε of the Hessian
  and gradient) ± ε²(…). That should lift both links toward the outer link's ≥ 6·10⁻³.
- **The hard cap of this route is Lemma G2's ε ≤ 0.0462.** Above it, the t-range of |w₁| ≤ W can meet two fold windows,
  and the single-fold localisation fails at its step 2. So ε_max = 0.6 cannot be reached by perturbing the limit
  certificate, even cell by cell. U2 needs a multi-fold finite-ε localisation, and per-a or ε-interval certificates of
  the §5 kind.

### 20.7 What is proved, certified or argued

| item | status |
|---|---|
| Lemma U1, U1′ (pointwise remainders, O(ε²), all σ, all ε > 0) | **proved**; checked at 80 digits (max ratio 0.9999991) |
| Lemma U2 (profiled remainder, C explicit; \|L₁\| ≤ A·mean\|r − c\|) | **proved**; checked on constructed cases |
| r_max closed form | **proved** (elementary); checked on a grid |
| L₁, C at θ_c | float with the 10⁻⁹ allowance (producer standard) |
| margin m ∈ [1.0301·10⁻⁴, 4.7027·10⁻³] near the switch | **certified** (producer float B&B + allowance); lower end ≥ 2.706·10⁻⁴ **checked in Arb** |
| ε₀ = 1.5·10⁻¹² for one margin vs the uniform perturbation of the difference | **proved** given m |
| Lemma G2 for ε ≤ 0.0462 | proved (§16.1), with W(s, a) evaluated in floats on a 10⁻⁴ grid; monotonicity in ε argued |
| B(P) table, B_full, per-cell PAVA | **exact** (integer block sums, 50-digit logs) |
| sep and fav lower bounds | **proved** (Lemma G1 + §5.0's argument + convexity) |
| ε-uniform outer exclusion, 40 sub-intervals at ε₀ = 4.1·10⁻⁶ | **certified, float with allowance** (τ = 10⁻⁷ on top of 10⁻⁹); **not yet Arb-checked** (U3) |
| ring and PD links at ε₀ | **proved** from certified margins (§16.2's bounds) |
| switch box and gap sign over [0.66, 0.71] at ε₀ | **certified** (mpmath interval arithmetic, 30 digits; second-order inflation) |
| φ_ε's critical points within 0.01 + 2√2ε₀ of ±√2 | argued (as in §16.3: ±arccos(1/a)/√ε = ±√2(1 − 5ε/12 + …)) |
| Theorem U1 | holds given the rows above; fully certified once U3 confirms the outer exclusion in Arb |
| outer link alone up to ≥ 6·10⁻³ | **exploratory** (k = 19 only) |

### 20.8 Export needed for the independent Arb checker (U3; not run here)

- **The claim per sub-interval k**:
  - the A-interval, θ_c, ε₀, P = 24, ρ = 0.15, τ;
  - the branch-side constants: bu₀, L₁,bu, C_bu, BU. These are to be re-derived in Arb from L\*_0, L₁ and C at θ_c at
    both endpoints;
  - the B(P) table and B_full;
  - the six test formulas exactly as in §20.5.
  - Each `outer_eps_4.1e-6_kNN.json` carries these. It also records the SHA-256 of `certv2_annulus_parts.csv`,
    `mn2_bounds.csv` and `constants.json`.
- **Leaves**:
  - k = 19 and 20 have `…_leaves.npz` (p, q, hp, hq, test, D), with a file SHA-256 and an array SHA-256;
  - the other 38 record the array SHA-256 of the in-memory leaves, so a re-run can be compared bit for bit.
- **The checker should**:
  - recompute BU in Arb;
  - check each leaf's test in Arb, either replaying the leaves or by a fresh search as Track 4 did;
  - check that the leaves tile K(24) ∩ {p ≥ 0} minus the 0.15 box (quadtree coverage);
  - re-evaluate the B(P) table.
- **The Krawczyk part** is already in interval arithmetic, so an Arb re-run of it is optional.
- **Not exported**: the exploratory scan.

## 21. Track K: small-scale compactness for Theorem 1, sine family, width 1, on all three objectives (2026-10-15)

**Status, stated per part.** Theory only; no training, no seeds. Every number below is from `src/small_scale_k.py` →
`results/small_scale_k.json` (the `ledger` block); tests in `tests/test_small_scale_k.py`. Labels: **proved** (by hand,
every step written here), **certified** (a finite computation in ball arithmetic, python-flint `arb` at 128 bits, or in
exact rationals), **argued** (not used for any conclusion).
- **PROVED, on all three objectives** (continuous windows; the 800-point quadrature as the code evaluates it, in
  float64; the exact linspace population of §17): for every a in the tables, there are s₁ > 0 and W such that for
  every s ∈ (0, s₁) conditional minimisers exist and every one has |w₁| ≤ W. This is Theorem 1's small-scale
  compactness hypothesis (§10 Corollary 1), in the sense fixed in 21.0.
- **New relative to §17.**
  - Continuous windows: upgraded from CHECKED (conditional on a float scan) to **proved with certified constants**,
    and also with a cruder set that needs no scan at all.
  - Float quadrature: §17's "float caveat" is **closed**. The constants are astronomically weak (W ≈ 1.4·10¹⁹,
    s₁ ≈ 3.7·10⁻¹⁹), and they are what the aliasing forces, not slack.
  - Exact population: the alias is **forced**. Every admissible W satisfies W ≥ π/q = 623,613.996, and the minimisers
    tend to |w₁| ≥ π/q as s → 0 (proved). So on a finite quadrature the compact set must contain the aliased maximiser.
  - Corollary 1(i) is re-proved **without the remainder lemma** (Lemma K1). Nothing in this section uses Lemma 1's
    remainder constant K (§10 Lemma 1; in this note K is given only with a proof sketch, `theorem1_hypotheses.md` row R).
- **Not proved:** width 2. The step that fails is stated in 21.6, with a partial result (proved).
- **Unchanged:** training samples fail (§17, Proposition C′). Corollary 1(ii) on the finite objectives is not examined.

### 21.0 The statement being proved

- **Objectives P.** All three are balanced (class masses ½) and class-symmetric about x = 0.
  - **P_c, the continuous windows:** ½ uniform on I = [−0.8, 0.8] plus ½ uniform on O = ±[1.2, 2].
  - **P_fl, the 800-point quadrature as evaluated:** the float64 points of `blockB_landscape.population_data`. The
    objective is the exact mean over these 800 binary64 numbers, read as exact rationals.
  - **P_ex, the exact linspace population:** the same 800 points in exact arithmetic, x = n·q, q = 0.4/79,401 (§17.4).
- **Parameters.** θ = (w₁, b₁, σ₂) with σ₂ = sign w₂; s = |w₂| > 0; the output bias b is profiled. So
  φ_θ = σ₂ f_a(w₁x + b₁), f_a(t) = t + a sin t, and L*_P(θ; s) = min_b E_P ℓ(sφ_θ + b, y).
- **Symmetries (proved; each is exact).**
  - (W) b₁ ↦ b₁ + 2πk. Since f_a(t + 2πk) = f_a(t) + 2πk, φ shifts by the constant σ₂·2πk, which b absorbs. So L*
    is invariant, and so are Δμ and Var(φ).
  - (R1) (w₁, b₁, σ₂) ↦ (−w₁, −b₁, −σ₂). f_a is odd, so φ is unchanged pointwise.
  - (R2) w₁ ↦ −w₁ at fixed b₁ (the reflection p ↦ −p of the rescaled coordinates). This maps φ(x) to φ(−x), so L*
    is invariant on class-symmetric data. All three objectives are class-symmetric; for P_fl this is checked exactly
    (21.4).
- **The parameter space** is the quotient Θ̄ = ℝ × (ℝ/2πℤ) × {±1}. L*(·; s) is a continuous function on Θ̄: the
  loss is jointly continuous in (θ, b), strictly convex and coercive in b because both classes are present and φ is
  bounded on the support, so the profiled minimum is continuous (Berge).
- **"A common compact set"** means K_W := {|w₁| ≤ W} ⊂ Θ̄. K_W is compact and invariant under R1 and R2, so no
  further quotient is needed. In raw coordinates: every conditional minimiser has |w₁| ≤ W, and its winding
  representative b₁ mod 2π lies in [0, 2π).
- **Statement (Theorem K).** For P ∈ {P_c, P_fl, P_ex} and each tabulated a, with S_P = sup Δμ < ∞:
  - (a) for every s ∈ (0, s₁), minimisers of L*_P(·; s) exist. Every θ ∉ K_W has L*_P(θ; s) > min L*_P(·; s)
    (strictly), so every minimiser lies in K_W.
  - (b) every limit point in Θ̄ of minimisers θ_s, as s → 0, maximises Δμ. In fact Δμ(θ_s) ≥ S_P − sV/2.
  - (c) the limit's location:
    - P_c: |w₁| ∈ [1.784407, 1.800158] and |sin b₁| = 1;
    - P_ex: |w₁| ≥ π/q, and no W < π/q is admissible;
    - P_fl: not located, but a maximiser exists with |w₁| ≤ π·2⁶⁰.

### 21.1 What §17 proved, re-checked; and what it left open

- **Re-checked, every step.** Lemma C1 (exact decomposition), Lemma C2 (the three ψ bounds, including
  f″ = u² − tanh²u ≥ 0), Lemma C3 (spreading), Theorem C (steps 1–5) and Proposition C′ are correct as written.
  Lemma C3 and Theorem C hold verbatim for a probability measure in place of n points: "at least F·n points"
  becomes "mass at least F".
- **Two facts §17 used without stating them; both are needed below.**
  - **Remark K0 (monotonicity in V; proved).** In Theorem C, V₁ = Var(φ_{θ₁}) enters only through step 1's upper
    bound L*(θ₁; s) ≤ log 2 − (s/4)S + (s²/8)V₁. So the theorem holds with V₁ replaced by any V ≥ V₁. Then
    s₁ = √(8Fψ(2)/V) decreases and W_c = (a + √(1.2V/F))/τ increases, so the statement stays valid, only weaker.
  - **Strict exclusion (proved).** For s < s₁ and |w₁| > W_c, steps 2–5 give
    L*(θ; s) − L*(θ₁; s) ≥ (s/4)(S − Δμ(θ)) + V_s(θ) − s²V/8 > 0. So every θ outside K_{W_c} is strictly beaten by θ₁.
- **Left open by §17:**
  - (i) on P_c, the input "S is attained at α\*_c with Var 5.66–6.23" rested on a float Lipschitz scan;
  - (ii) on P_fl, the float caveat (§17.4);
  - (iii) whether the alias really has to be inside the compact set, or is just inside the one Theorem C produces;
  - (iv) width 2.
  §21 closes (i)–(iii). For (iv) it identifies the failing step and proves a partial result.

### 21.2 Two lemmas valid on any balanced objective

**Lemma K1 (first-order selection, no remainder; proved).** For every θ′, every s > 0 and every conditional minimiser
θ_s:  Δμ(θ_s) ≥ Δμ(θ′) − (s/2)·Var(φ_{θ′}).
- *Proof.* By C1, L*(θ_s; s) = log 2 − (s/4)Δμ(θ_s) + V_s(θ_s) ≥ log 2 − (s/4)Δμ(θ_s), since V_s ≥ 0.
- By C1 and ψ(z) ≤ z²/8 with c = mean φ_{θ′}: L*(θ′; s) ≤ log 2 − (s/4)Δμ(θ′) + (s²/8)Var(φ_{θ′}).
- Minimality gives L*(θ_s; s) ≤ L*(θ′; s). Combine the three and divide by s/4. ∎
- **Corollary 1(i) of §10, re-proved.** Suppose the θ_s lie in a fixed K_W. Δμ is continuous on Θ̄ (it is invariant
  under (W)). Take θ′ = θ₁, an attaining point. Then any limit point θ̄ has Δμ(θ̄) = lim Δμ(θ_{s_n}) ≥ S, so it
  maximises Δμ.
  - This replaces the "O(s³)" step of §10's proof, which needed Lemma 1's remainder to be uniform. That remainder's
    constant K is still a sketch (`theorem1_hypotheses.md`, row R). It is not used here.

**Lemma K2 (uniform coercivity in w₁, and existence; proved).** Let F = 1 − N_τ/n, with N_τ the largest mass in a
closed x-interval of length 2τ = 0.8. Then for every θ with τ|w₁| > a and every s > 0:

  L*(θ; s) ≥ log 2 − (s/4)S + F·ψ(s(τ|w₁| − a)).

- *Proof.* This is C1 with Δμ ≤ S, plus §17 step 3 (C3 and the monotonicity of ψ). ∎
- ψ(z) → ∞ as |z| → ∞. So for each s > 0, L*(·; s) → ∞ as |w₁| → ∞, uniformly in (b₁, σ₂).
- A sublevel set of L*(·; s) is therefore closed and contained in some K_W, hence compact. L* is continuous, so it
  attains its minimum. **Conditional minimisers exist at every s > 0** on all three objectives, since S < ∞ there.
  - S < ∞ because Δμ = σ₂ a sin b₁ D(w₁) with |D| ≤ 2 (§11 Step 1; it needs class symmetry).
- Lemma K2 is the author's suggested "uniformly coercive outside a fixed ball" route. With Remark K0 it is Theorem C.

### 21.3 The continuous windows P_c (proved, with certified constants)

**Lemma K3 (attainment and localisation of the class-mean maximiser).** On P_c,
D_c(α) = [sin 2α − sin 1.2α − sin 0.8α]/(0.8α) (§11 Step 1), and D_c is even.
1. **D_c → 0 at both ends (proved).**
   - The numerator's linear terms cancel (2 − 1.2 − 0.8 = 0), and |sin u − u| ≤ |u|³/6. So
     |D_c(α)| ≤ (8 + 1.728 + 0.512)α³/(6·0.8α) = (32/15)α².
   - Also |D_c(α)| ≤ 3/(0.8α).
   - So |D_c| extends continuously to α = 0 with value 0, and tends to 0 as α → ∞. Hence sup|D_c| > 0 is
     **attained** at some α₁ > 0.
   - Δμ = σ₂ a sin b₁ D_c(w₁), so S = a·max|D_c|. It is attained exactly when |w₁| = α₁ maximises |D_c| and
     σ₂ sin b₁ = sign D_c(α₁); in particular cos b₁ = 0.
2. **A certified lower bound (certified).** D_lo := 1.5732418725 ≤ |D_c(1.7922917)| (ball enclosure at the exact
   rational 17,922,917/10⁷). So max|D_c| ≥ D_lo.
3. **Every maximiser has |α₁| ≤ A₀ := 3/(0.8·D_lo) ≤ 2.3836131 (proved).** For α > A₀, |D_c| ≤ 3/(0.8α) < D_lo.
4. **No maximiser in (0, ½] (proved).** There |D_c| ≤ (32/15)/4 = 8/15 < D_lo.
5. **Localisation on [½, A₀] (certified).**
   - Adaptive bisection. A cell is cleared when the ball enclosure of |D_c| has upper end < D_lo; it is kept only if
     it lies inside [α₀ − 0.01, α₀ + 0.01], α₀ = 1.7922917, and has width ≤ 10⁻⁴. Otherwise it is bisected
     (depth ≤ 40).
   - Result: 385 cells cleared, 274 kept, none unresolved (maximum depth 15).
   - **Every maximiser lies in [1.784407, 1.800158]** (the span of the kept cells).
   - The test point's own cell is among the kept ones. Requiring kept cells to be narrow means a D_lo above the true
     maximum would clear everything; this is a tested fail case.
6. **The variance at any maximiser (proved formula; certified bound).**
   - With cos b₁ = 0, φ = σ₂[α₁x + b₁ + a sin b₁·cos(α₁x)].
   - Cov(x, cos α₁x) = E[x cos α₁x] = 0, because the integrand is odd and P_c is symmetric with E x = 0. So
     V(α₁) = α₁²·Var(x) + a²·Var(cos α₁x), with Var(x) = ½(0.64/3 + 6.272/2.4) = 106/75 exactly.
   - Var(cos αx) = ½ + ½C(2α) − C(α)², with C(β) = ½·sin(0.8β)/(0.8β) + ½·(sin 2β − sin 1.2β)/(0.8β).
   - V is enclosed over the whole box α₀ ± 0.01 on 400 sub-cells, giving V̄(a) ≥ Var(φ_{θ₁}) for every maximiser θ₁.
   - **Crude alternative (no scan; proved plus the one evaluation in step 2):** V ≤ A₀²·106/75 + a².
7. **F = 3/4 exactly (proved).** P_c has density ½/1.6 = 0.3125 on I and on O, so a closed interval of length 0.8 has
   mass ≤ 0.25.

**Theorem K-c (proved; constants certified).** For s < s₁, every conditional minimiser on P_c has |w₁| ≤ W
(Theorem C with Remark K0, τ = 0.4).

| a | V̄ (box) | s₁ ≥ (box) | W ≤ (box) | V (crude, no scan) | s₁ ≥ (crude) | W ≤ (crude) |
|---|---|---|---|---|---|---|
| 1.30 | 5.7096 | 0.675 | 10.81 | 9.720 | 0.517 | 13.11 |
| 1.60 | 6.2855 | 0.643 | 11.93 | 10.590 | 0.496 | 14.29 |

- All nine a (1.02–1.60) are in the JSON. The box W ranges over 9.82–11.93, and s₁ ≥ 0.643 at every a.
- **Against §17's float-scan values** (10.77 and 0.678 at a = 1.30; 11.90 and 0.646 at 1.60): the rigorous ones
  are slightly weaker, because V̄ covers the whole box rather than the float maximiser. They agree to within 0.5%.
- **Consequence (proved).** By Lemma K1 and Lemma K3(5), every s → 0 limit point has |w₁| ∈ [1.784407, 1.800158].
  By §11 Step 2 it is unplaced (G ≤ 2a − 2.8|w₁| < 0) for every a < 1.4·1.784407 = 2.498.
- **Not certified here:** uniqueness of the maximiser inside the box, and quadratic growth (row Q of
  `theorem1_hypotheses.md`, still float-checked). Neither is needed for compactness.

### 21.4 The float quadrature P_fl, as evaluated (proved; exact arithmetic)

**Lemma K5 (exact facts about the 800 binary64 points; certified in exact rationals).**
- Every x_i = k_i/2⁶⁰ with k_i ∈ ℤ, and gcd(k_i) = 1.
- Each class is exactly symmetric about 0: the sorted values satisfy x_(j) = −x_(n+1−j). Hence m = E_O x − E_I x = 0
  exactly.
- N_τ = 200 (exact Fraction sliding count, closed interval of length exactly 4/5), so F = 3/4.
- Var(x) = 1.414136010… (exact rational).

**Theorem K-fl (proved).**
1. By class symmetry, §11 Step 1 holds exactly on P_fl: Δμ = σ₂ a sin b₁ D_fl(w₁), and Cov(x, cos αx) = 0.
2. cos(α k_i/2⁶⁰) is 2π·2⁶⁰-periodic in α, and D_fl is even and continuous. So sup|D_fl| is attained at some α₁ with
   0 < α₁ ≤ π·2⁶⁰ ≤ 3.6220097·10¹⁸.
3. Take θ₁ = (α₁, b₁, σ₂) with cos b₁ = 0. Then Var(φ_{θ₁}) = α₁²Var(x) + a²Var(cos) ≤ (π·2⁶⁰)²·Var(x) + a² ≤ 1.8552·10³⁷.
4. Theorem C with Remark K0 then gives, for every tabulated a (the a-dependence is below the printed digits):
   **s₁ ≥ 3.7455·10⁻¹⁹ and W ≤ 1.3621·10¹⁹.** ∎

- **What this is, and what it is not.**
  - This is a proof of the hypothesis as stated, for the objective the code actually evaluates.
  - It says nothing at any practically relevant scale. The constants are this weak because the true maximiser of
    |D_fl| is not located: a lattice of mesh 2⁻⁶⁰ has only the trivial period bound.
- **Diagnostic (certified).** The alias π/q of the exact lattice is still a near-maximiser on the float data:
  2 − |D_fl(π/q)| ≤ 1.5·10⁻²¹ (ball arithmetic over the 800 binary64 points; the floats sit within 1.40·10⁻¹⁶ of n·q).
  - So S_fl ≥ 2a(1 − 7.5·10⁻²²).
  - The float objective's s → 0 limit is therefore an aliased weight as well: every maximiser has
    |D_fl| ≥ 2 − 1.5·10⁻²¹, far above |D_fl| ≈ 1.571 at α\*.
  - Its location, which may lie beyond π/q, is not computed.

### 21.5 The exact linspace population P_ex: the alias is forced (proved)

- **§17's constants, recomputed** in ball arithmetic from exact rationals, agree: W ≤ 2,345,103.67 and
  s₁ ≥ 2.17545·10⁻⁶ at a = 1.30. Here V = (π/q)²·Var(x) + a² is exact at the alias, and Var(cos) = 1 there.
  N_τ = 200 exactly.

**Lemma K4 (the alias is the smallest class-mean maximiser; proved).** On P_ex, |D(α)| = 2 implies α = 0 or
|α| ≥ π/q. Since D(0) = 0, every maximiser of Δμ has |w₁| ≥ π/q, and |w₁| = π/q is one (§17.4).
- *Proof.* Both class means of cos lie in [−1, 1]. So |D| = 2 forces cos(αx) = c on every inner point and −c on
  every outer point, with c = ±1.
- If cos u = cos v = ±1, then u, v ∈ πℤ with the same parity, so u − v ∈ 2πℤ.
- Consecutive inner points differ by exactly 796q, and consecutive outer points on the same side by exactly 798q
  (certified in integers: the only gaps are 796 and 798).
- Hence α·796q ∈ 2πℤ and α·798q ∈ 2πℤ. Subtracting, α·2q ∈ 2πℤ, i.e. αq ∈ πℤ. ∎

**Corollary K4′ (proved; not quantitative).** For every δ > 0 there is s_δ > 0 such that every conditional minimiser
at s < s_δ has |w₁| > π/q − δ. Consequently every admissible W (one for which, for some s′ > 0, all minimisers at all
s < s′ lie in K_W) satisfies **W ≥ π/q = 623,613.996**. Together with Theorem C: **the best W lies in
[623,614, 2,345,104].**
- *Proof.* Suppose not: then s_n → 0 with minimisers θ_n (they exist by K2) satisfying |w₁| ≤ π/q − δ.
- By Theorem C they lie in K_{W_c}, so a subsequence converges in Θ̄ to some θ̄ with |w̄₁| ≤ π/q − δ.
- By Lemma K1 and continuity, θ̄ maximises Δμ. That contradicts Lemma K4.
- The bound on W follows because minimisers exist at every s. ∎
- **Consistency with the earlier numbers.**
  - The localisation lemma (§5.0) excludes |w₁| ≥ π/q for s > 7.24·10⁻⁶.
  - The 40-digit crossover puts the alias ahead of θ\* only below s ≈ 2·10⁻¹² (`theorem1_checks.json`).
  - So s_δ is somewhere at or below that scale. K4′ does not compute it.

### 21.6 Width 2: the step that fails, and what still holds

- **Setting.** φ = Σᵢ ṽᵢ f_a(αᵢx + βᵢ), ‖ṽ‖₁ = 1 (§10), on P_c.
- **The step that fails is Lemma C3.** Spreading bounds the slope of the linear part, c = Σᵢ ṽᵢαᵢ, and not the αᵢ.
  A unit with ṽᵢ → 0 and αᵢ → ∞ at bounded ṽᵢαᵢ keeps φ bounded and V_s small. So V_s gives no bound on the αᵢ, and
  K_W has no width-2 analogue from this argument.
- Whether width-2 minimisers actually escape is **neither proved nor disproved**.

**Lemma K6 (partial; proved).** Let s < √(8Fψ(2))/a (= 1.24 at a = 1.30), and let θ_s be a width-2 conditional
minimiser on P_c. Then:
- (i) |c| ≤ (a + √(1.2a²/F))/τ = a(1 + √1.6)/0.4 ≤ 5.67a;
- (ii) every unit with |αᵢ| ≥ 2A₀ = 4.767 carries weight |ṽᵢ| ≤ s·a/D_lo.

*Proof.*
- **The competitor.** The cancelling pair (ṽ = (½, ½), α = (α₁, −α₁), equal βᵢ with sin βᵢ = ±1, α₁ a maximiser
  of |D_c|) has φ = β + a sin β·cos(α₁x). So it attains S = a·max|D_c| with Var ≤ a².
- **(i)** As in Theorem C, V_s(θ_s) ≤ s²a²/8. Here |φ − c₀| ≥ |cx + const − c₀| − a, since the oscillatory part is
  bounded by a·‖ṽ‖₁ = a. Steps 3–5 of Theorem C then apply with |w₁| replaced by |c|.
- **(ii)**
  - The linear parts drop out of Δμ by symmetry, so Δμ = a Σᵢ ṽᵢ sin βᵢ D_c(αᵢ) ≤ a Σᵢ|ṽᵢ||D_c(αᵢ)|.
  - Lemma K1 with the pair gives a Σᵢ|ṽᵢ|(max|D_c| − |D_c(αᵢ)|) ≤ s a²/2.
  - For |αᵢ| ≥ 2A₀: |D_c(αᵢ)| ≤ D_lo/2 ≤ max|D_c|/2. So |ṽᵢ|·max|D_c|/2 ≤ s a/2. ∎
- So at width 2 at most an O(s) weight can sit on large input weights. Compactness in parameter space would need a
  second-order argument showing that such a unit never lowers L*. That is not done.

### 21.7 What this means for Theorem 1 (width 1)

- **The small-scale compactness hypothesis now holds, proved, on every objective the paper uses for the population.**
  Corollary 1(i) follows without the remainder lemma (K1).
- **On the continuous windows** the compact set is modest: |w₁| ≤ 12 for s < 0.64 at every tabulated a. The limit is
  the ramp-plus-cosine at |w₁| ∈ [1.7844, 1.8002], which is unplaced.
- **On either 800-point quadrature** the hypothesis holds only with the alias inside the compact set, and this is
  forced on P_ex (W ≥ 623,614). The literal s → 0 limit is an aliased weight, not α\*.
  - The alias is unplaced too, so the width-1 verdict is unchanged.
  - The paper's limit statement "the small-scale minimiser is the cosine at α\*" needs "on the continuous windows",
    or "for s above ≈ 10⁻¹¹" (unchanged from `theorem1_hypotheses.md`).
- **Training samples** (m ≠ 0): compactness fails (§17, C′). The unplaced conclusion still holds (C3).
- **Producer and tests.**
  - The run is one process at nice 15, under a second of CPU and well under 1 GB.
  - Tests cover: the D_c enclosure against the closed form and against quadrature; the small-α and decay bounds; the
    Var(cos) enclosure; the localisation pass and three constructed fail cases (D_lo above the maximum, a box too
    small, a wrong centre); agreement of the constants with §17's formulas, and their monotonicity; the exact
    spreading count, dyadic lattice, symmetry (with a one-ulp fail case) and gap facts on constructed data; and the
    committed numbers.

## 22. Track K follow-up: width 2, the escaping unit's ramp limit; compactness on the symmetric windows, non-attainment on the asymmetric windows (2026-10-15)

**Status, stated per part.** Theory only; no training, no seeds. Every number below is from `src/small_scale_w2.py` →
`results/small_scale_w2.json` (the `ledger` block); tests in `tests/test_small_scale_w2.py`. Labels as in §21:
**proved** (by hand, every step written here), **certified** (a finite computation in `arb` ball arithmetic at 128 bits,
rigorous integrals by `acb.integral`, or exact rationals), **argued** (not used for any conclusion), **exploratory**
(float, not part of any proof). When this work started §20 (Track U1) was not yet in the note; U1 committed it
concurrently (042691a), so §22 is appended after §21, the last section.
- **The ramp limit (Lemma K7; proved).** A unit with ṽ₂ → 0, |α₂| → ∞, ṽ₂α₂ → γ converges, uniformly in x, to the
  linear term γx plus a constant. Its bias only adds a constant (absorbed by b). Its sine part is bounded by a|ṽ₂| and
  vanishes in the limit **on every objective, finite or continuous; aliasing does not change the limit**. The limit
  configuration is **φ̄ = γx + σf_a(αx + β)**: one unit carrying all the ℓ₁ weight, plus a free linear term.
  - Aliasing changes the **approach**, not the limit. On the continuous windows the escaping sine part changes the loss
    by O(sε/|α₂|) (Lemma K9, explicit), where ε = |ṽ₂|. On the quadratures it does not average out, and it can itself
    sit on an alias with |D| = 2.
- **(1) Symmetric continuous windows P_c: width-2 compactness PROVED (Theorem K11), constants certified.**
  - For every tabulated a and every s ≤ s₂(a), with s₂ ≥ 0.0066 at every a and s₂ = 0.01 at a = 1.30: conditional
    minimisers exist, both weights are nonzero, and both |αᵢ| ≤ W₂(a) (≤ 8.84 at every a; 6.37 at a = 1.30).
  - **The ramp limit itself is not strictly worse.** At a class-mean maximiser, with γ cancelling the slope, it is
    exactly the cancelling pair's function, so it ties at every s.
  - **What is strictly worse (proved):**
    - every finite approach to a ramp limit. The first-order cost is sε·aD\*/4: the ℓ₁ weight the escaping unit uses.
    - every ramp limit whose slope lies outside the "mirror range" |c̄| ≤ |α|. Inside that range the ramp limit is
      realised by a compact configuration (Lemma K8).
  - **Consequences (proved; uniqueness of the maximiser is now certified).**
    - The s → 0 limit is the cancelling-pair function const − a·cos(α̂x), where α̂ is the unique maximiser of |D_c|.
      This is §10.1's second-order selection, obtained without the quadratic-growth hypothesis (row Q) and without
      Lemma 1's remainder constant K.
    - It is placed: G₊ ≥ 0.887 at a = 1.30 (registered value 0.890).
    - So every width-2 conditional minimiser on P_c is placed for all s below some s\* > 0. s\* is not computed.
- **Symmetric quadratures P_ex, P_fl: NOT proved.** Two steps fail (22.3):
  - Lemma K9 has no analogue there: aliasing.
  - There is an exact unbounded symmetry, the alias translations (Proposition K10). It moves every minimiser whose
    weight ratio is rational along an unbounded orbit.
- **(2) Asymmetric continuous windows P_a (Δ = 0.4): YES, PROVED (Theorem K12), constants certified.**
  - Take every tabulated a and every s ≤ s_a(a), with s_a ≥ 0.0071 at every a and s_a = 0.0076 at a = 1.30. Then the
    infimum of the width-2 conditional loss equals the minimum over ramp limits and is **not attained** by any finite
    configuration.
  - Every minimising sequence escapes: one weight → 0, its |α| → ∞, and a subsequence converges to a ramp-limit
    minimiser.
  - Theorem 1 is restated on the extended space Θ̂ = Θ₂ ∪ R in 22.5. Which conclusions survive:
    - **survive:** existence; the C1, Lemma 1 and K1 identities; "unplaced at small scale" (Proposition K13, explicit
      s_u = 0.0202 at a = 1.30);
    - **fails even on Θ̂:** Corollary 1(i) in raw coordinates, because the slope still diverges like κ\*/s;
    - **holds in rescaled coordinates** (proved): κ = s·c̄ → κ\* = 0.2825 (the logistic-regression slope), and the sine
      part maximises a residual-weighted class gap Γ;
    - **not examined:** Corollary 1(ii) and the large-s side (Lemma 2).
- **Exploratory (float; Track 2's registered 800-point landscape).**
  - At both T2-1 bracket ends, and at every scanned s ≥ 0.178, the ramp-limit minimum equals T2-1's registered
    retained loss to ≤ 1.7·10⁻¹⁵, with slope inside the mirror range.
  - At s = 0.1 and 0.133 it lies below the registered retained loss, by 1.14·10⁻³ and 3.4·10⁻⁴.
  - Every such point is unplaced, so **no registered status or verdict changes** (22.6).

### 22.0 Setting

- **Width 2 as in §10.**
  - φ_θ = ṽ₁f_a(α₁x + β₁) + ṽ₂f_a(α₂x + β₂), with |ṽ₁| + |ṽ₂| = 1. The logits are sφ + b, with s = ‖v‖₁ and b profiled.
  - Θ₂ = {ṽ} × ℝ² × (ℝ/2πℤ)². βᵢ is taken mod 2π by (W) of §21.0: shifting βᵢ by 2π adds the constant 2πṽᵢ to φ.
  - K_W := {|α₁|, |α₂| ≤ W} ⊂ Θ₂ is compact.
- **Convention.** f_a(t) = t + a sin t, as in §§1, 11 and 21. The author's t − a sin t equals f_a(t + π) − π. That is the
  same family with each β shifted by π, so nothing below depends on the choice.
- **Decomposition.** φ = cx + κ₀ + o(x), with:
  - c = ṽ₁α₁ + ṽ₂α₂, the slope;
  - κ₀ a constant;
  - o = aΣᵢṽᵢ sin(αᵢx + βᵢ), so |o| ≤ a.
- **Class-mean gap.** On a balanced objective, Δμ = c·m + aΣᵢṽᵢA(αᵢ, βᵢ), where m = E_O x − E_I x and
  A(α, β) := E_O sin(αx + β) − E_I sin(αx + β), so |A| ≤ 2. On class-symmetric data, m = 0 and A = sin β·D(α) (§11 Step 1).
- **Objectives.**
  - P_c, P_ex and P_fl are as in §21.0.
  - **P_a**, the continuous version of Track 2's windows (`asym_registration.md`): ½U(I) + ½U(O_Δ), with Δ = 0.4 and
    O_Δ = [−2, −1.2] ∪ [1.2, 2.4]. The densities are 5/16 on I and 1/4 on O_Δ.
    - Exact (certified in rationals): m = 11/25, Var x = 11917/7500 = 1.588933, E|x| = 53/50.
  - **P_an**, Track 2's 800-point population (`asym_pilot.population(0.4)`): 400 + 400 points. m = 0.44 to float
    rounding, and Var x = 1.589903 (exact rationals of the binary64 points).
- **Five facts used throughout (proved).**
  - **(F1)** L*(φ; s) does not change when a constant is added to φ, and |L*(φ; s) − L*(ψ; s)| ≤ s·sup|φ − ψ| on the
    support. This holds because ℓ is 1-Lipschitz in z and the minimum over b preserves that.
  - **(F2)** L* ≥ log 2 − (s/4)Δμ, with no remainder (§11 Proposition (i)).
  - **(F3)** L* ≤ log 2 − (s/4)Δμ + (s²/8)Var φ (Lemma C1 with ψ ≤ z²/8).
  - **(F4) (new)** V_s(θ) ≥ (s²/8)Var φ − (s⁴/192)·R(φ)⁴, where R(φ) = sup φ − inf φ on the support.
    - Lemma C2's inequality log cosh u ≥ u²/2 − u⁴/12 holds for every u, so ψ(z) ≥ z²/8 − z⁴/192 for every z.
    - The minimising c lies in [inf φ, sup φ], since ψ is even and increasing in |z|. So E(φ − c)⁴ ≤ R⁴.
    - Also E(φ − c)² ≥ Var φ.
  - **(F5) (scale derivative)** For fixed φ, g(t) := L*(φ; t) has g′(0) = −Δμ/4 and 0 ≤ g″ ≤ Var(φ)/4. Hence
    g′(t) ≤ −Δμ/4 + t·Var(φ)/4.
    - Here g″ = E σ′(z)(φ − m_w)², with m_w the σ′-weighted mean (§10.1). This is ≤ ¼E(φ − Eφ)².

### 22.1 The escaping unit's limit, the mirror fold, and averaging

**Lemma K7 (the ramp limit; proved).** Let θⁿ ∈ Θ₂ satisfy:
- ṽ₂ⁿ → 0, |α₂ⁿ| → ∞ and ṽ₂ⁿα₂ⁿ → γ ∈ ℝ;
- (ṽ₁ⁿ, α₁ⁿ, β₁ⁿ) → (σ, α, β), with σ = ±1.

Then, uniformly on every bounded x-set,

  **φ_{θⁿ} − ṽ₂ⁿβ₂ⁿ → φ̄ := γx + σf_a(αx + β),**

and L*_P(θⁿ; s) → L*_P(φ̄; s) for every objective P (continuous or finite) and every s.
- *Proof.* Expand the escaping unit: ṽ₂f_a(α₂x + β₂) = (ṽ₂α₂)x + ṽ₂β₂ + aṽ₂ sin(α₂x + β₂).
  - The sine term is bounded by a|ṽ₂| uniformly on ℝ, whatever α₂ and β₂ are.
  - ṽ₂β₂ is a constant.
  - The linear term converges uniformly on bounded sets.
  - Unit 1 converges uniformly on bounded sets by continuity. Then apply (F1). ∎
- **The ramp-limit set** is R := {γx + σf_a(αx + β)} ≅ ℝ × {±1} × ℝ × ℝ/2πℤ. Equivalently, φ̄ = c̄x + const +
  σa sin(αx + β), with a free total slope c̄ = γ + σα.
- **The approach (the correction at first order in ε = |ṽ₂|).**
  - *Continuous windows* (P_c, P_a): the sine part averages out. By Lemma K9 its effect on the loss is at most
    sεaK₁/|α₂|.
  - *Quadratures:* it does not average out. On P_ex every x is a multiple of q, so sin(α₂xᵢ + β₂) is 2π/q-periodic in
    α₂. The escaping unit's class-mean contribution εa·sin β₂·D_ex(α₂) does not tend to 0. It equals ±2εa on the
    aliases α₂ ∈ (2k + 1)π/q, so the escaping unit can itself be a class-mean maximiser. On P_fl the same holds with
    period 2π·2⁶⁰.
- **Configurations that do not converge uniformly** (both weights bounded away from 0, both |αᵢ| → ∞, slopes
  cancelling) are not needed:
  - Theorem K11 excludes them at small s by Lemma K6(ii);
  - Theorem K12 excludes them by the bound in its proof of (c).

**Lemma K8 (the mirror fold; proved).** For every (α, β), the units (α, β) and (−α, π − β) have the same sine part,
because sin(π − (αx + β)) = sin(αx + β).
- With weights pσ and qσ (p, q ≥ 0, p + q = 1): φ = σ(p − q)αx + const + σa sin(αx + β).
- So every element of R with |c̄| ≤ |α| is the function of a configuration in K_{|α|}.
- The cancelling pair of §10.1 is the case p = q, β = ±π/2.
- Units (α, β) and (α, β + π) with weights of opposite signs give the same family. Track 2's registered retained
  points have this form.

**Lemma K9 (fast oscillation; proved).** Let P be piecewise uniform: density ρ_J on finitely many intervals J, with
class label y_J on J. Let z be absolutely continuous on each J. Then for every α ≠ 0 and every β,

  |E_P[(σ(z) − y)·sin(αx + β)]| ≤ Σ_J ρ_J(2 + ¼∫_J|z′|)/|α|.

- *Proof.* Integrate by parts on each J: ∫_J G sin(αx + β) = [−G cos(αx + β)/α]_J + (1/α)∫_J G′cos(αx + β), with
  G = σ(z) − y_J ∈ (−1, 1) and |G′| ≤ ¼|z′|. ∎
- The tests check it on smooth G (pass), and on a square wave locked to the sine. There the bound without its
  variation term fails.

**Proposition K10 (alias translations on the quadratures; proved).** On P_ex, the map
(α₁, α₂) ↦ (α₁ + 2πk₁/q, α₂ + 2πk₂/q), with kᵢ ∈ ℤ, changes φ on the data by the ramp 2π(ṽ₁k₁ + ṽ₂k₂)x/q and by
nothing else.
- *Proof.* xᵢ = nᵢq, so the sines are unchanged. Each linear term ṽⱼαⱼxᵢ changes by 2πṽⱼkⱼnᵢ. ∎
- **Consequence.** Suppose ṽ₁k₁ + ṽ₂k₂ = 0 for some (k₁, k₂) ≠ 0, i.e. ṽ₁/ṽ₂ ∈ ℚ. An example is the cancelling pair
  ṽ = (½, ½) with k₂ = −k₁. Then L*_{P_ex} is exactly constant along an unbounded orbit.
  - So if any width-2 minimiser on P_ex has a rational weight ratio, the minimiser set is unbounded in Θ₂, and no
    K_W contains it.
  - On P_fl the same holds with 2π/q replaced by 2π·2⁶⁰.
  - Whether the minimisers have rational ratios is not determined here.

### 22.2 The symmetric continuous windows P_c (proved; constants certified)

**Constants (certified).**
- D_lo = 1.5732418725 and 2A₀ ≤ 4.767226 (§21).
- On the maximiser box α₀ ± 0.01, which contains every maximiser of |D_c| (Lemma K3), Var(cos αx) ≤ V̄c = 0.661956.
  So V_p := a²V̄c bounds the variance of the cancelling pair at any maximiser.
- **Localisation threshold.** Every α > 0 with |D_c(α)| ≥ D_lo − 0.05 has α ≥ α_L = 1.589473. This is certified by
  bisection: 59 cells cleared, 6,748 kept, none unresolved; (0, ½] is cleared by the bound (32/15)α².
- **Uniqueness (certified, new).** D_c″ > 0 on the whole box: its lower end is 2.47. With D_c < 0 there, D_c is
  strictly convex on the box. So |D_c| has **exactly one maximiser α̂ > 0** (up to the sign of α). This upgrades row A2's
  "unique, computer-checked" to certified.

**Theorem K11 (width 2 on P_c; proved, constants certified).** For each tabulated a and every s ∈ (0, s₂(a)]:
- **(a) Compactness.** Width-2 conditional minimisers exist. Every one has both weights nonzero and |α₁|, |α₂| ≤ W₂(a).
  Every θ outside K_{W₂}, and every θ with a zero weight, has L* strictly above the minimum.
- **(b) The limit.** Every s → 0 limit point has slope c = 0, and every unit of nonzero weight has |αᵢ| = α̂ with
  ṽᵢ sin βᵢ·D_c(α̂) = |ṽᵢ||D_c(α̂)|. So the limit function is **const − a·cos(α̂x)**, the cancelling pair's function.
- **(c) Placement.** That limit is placed: G₊ ≥ a·g_lo with g_lo = 0.6823. Hence there is s\* > 0 (not computed) such
  that every width-2 conditional minimiser at s < s\* is placed.

| a | s₂ (verified) | W₂ | G₊ of the limit ≥ |
|---|---|---|---|
| 1.02 | 0.016 | 8.84 | 0.696 |
| 1.30 | 0.01 | 6.37 | 0.887 |
| 1.40 | 0.0087 | 6.04 | 0.955 |
| 1.50 | 0.0075 | 5.78 | 1.023 |
| 1.60 | 0.0066 | 5.61 | 1.092 |

- All nine a are in the JSON. s₂ ranges over 0.0066–0.016, and W₂ ≤ 8.84 at every a.

**Proof.**
- **Notation.** θ_pair is the cancelling pair at the maximiser α̂. By (F3),
  L*(θ_pair) ≤ log 2 − (s/4)aD\* + (s²/8)V_p, with D\* = |D_c(α̂)|. Let Λ_s := {θ : L*(θ; s) ≤ L*(θ_pair; s)}.
- **Step 0 (facts on Λ_s, for s ≤ s₂).**
  - (i) Lemma K1 with θ′ = θ_pair gives Σᵢ(|ṽᵢ|D\* − ṽᵢ sin βᵢ D_c(αᵢ)) ≤ η₀ := sV_p/(2a). This uses only
    L*(θ) ≤ L*(θ′). Every term of the sum is ≥ 0.
  - (ii) A unit with |αᵢ| ≥ 2A₀ has |D_c(αᵢ)| ≤ D_lo/2 by the decay bound. So its weight is ≤ ε̄ := sV_p/(aD_lo)
    (Lemma K6(ii) with V_p). At s₂, 2ε̄ < 1, so at most one unit has |αᵢ| ≥ 2A₀.
  - (iii) |c| ≤ c̄ := (a + √(1.2V_p/F))/τ, by Lemma K6(i) with V_p (this needs s < √(8Fψ(2)/V_p)). Hence
    R(φ) ≤ 4c̄ + 2a.
  - (iv) Label the units so that either ṽ₂ = 0, or both weights are nonzero and |α₂| ≥ 2A₀ (then, by (ii), |α₂| is
    the larger). Put ε := |ṽ₂| and σ := sign ṽ₁.
    - Then u₁ := σ sin β₁·D_c(α₁) ≥ D\* − η₁, with η₁ := η₀/(1 − ε̄).
    - So |D_c(α₁)| ≥ D_lo − η₁ > D_lo − 0.05, hence |α₁| ≥ α_L.
    - |α₁| < 2A₀ when ε > 0, by (ii).
    - cos²β₁ ≤ k_c := 1 − (1 − η₁/D_lo)².
- **Step 1 (Case 2: |c| ≥ (1 − ε)|α₁|).** This case includes every configuration with a zero weight, and every ramp
  limit with |c̄| ≥ |α|.
  - *Competitor θ″:* the cancelling pair at the same |α₁|, with sin β = sign D_c(α₁). Then Δμ(θ″) = a|D_c(α₁)| and
    Var φ_{θ″} = a²Var(cos α₁x) =: T ≤ a².
  - Δμ(θ) ≤ a[(1 − ε)|D_c(α₁)| + ε|D_c(α₂)|] ≤ a|D_c(α₁)|, since |D_c(α₂)| ≤ D_lo/2 < |D_c(α₁)|.
  - Write φ − const = X + Y + r, with:
    - X = cx + B sin α₁x, which is odd, and B = (1 − ε)aσ cos β₁;
    - Y = A cos α₁x, which is even, and A = (1 − ε)aσ sin β₁;
    - r = ṽ₂a sin(α₂x + β₂).
  - Cov(X, Y) = 0, because P_c is symmetric. Put M := (c̄√Var x + a)² + a² ≥ Var X + Var Y. Then:
    - Var φ ≥ Var X + Var Y − 2εa√M;
    - Var X = E X² ≥ P := ((1 − ε̄)α_L√Var x − a√k_c)₊²;
    - Var Y − T ≥ −a²(2ε̄ + k_c).
  - By C1 with (F4) for θ, and (F3) for θ″:
    L*(θ) − L*(θ″) ≥ (s²/8)[P − a²(2ε̄ + k_c) − 2ε̄a√M − s²R⁴/24].
  - The bracket is certified positive at s₂ for every a (for example 3.020 − 2.938 at a = 1.30). θ″ ∈ K_{2A₀}.
- **Step 2 (Case 1: |c| < (1 − ε)|α₁|, so ε > 0; and |α₂| > W₁).**
  - *Competitor θ′:* the mirror fold (Lemma K8) at (α₁, β₁) with slope c′ := c/(1 − ε), where |c′| < |α₁|. So
    φ′ = c′x + σa sin(α₁x + β₁) + const, and **exactly φ_θ = (1 − ε)φ′ + r + const**.
  - ℓ is convex in z, so at z₀ = s(1 − ε)φ′ + b_θ: L*(θ; s) ≥ g(s(1 − ε)) + s·E[(σ(z₀) − y)r], with g(t) := L*(φ′; t).
  - By Lemma K9 with |z₀′| ≤ 2sA₀(1 + a): |E[(σ(z₀) − y)r]| ≤ εaK₁/|α₂|, with K₁ := 15/8 + sA₀(1 + a)/2.
  - By (F5): g(s(1 − ε)) − g(s) ≥ sε(Δμ(φ′) − s·Var φ′)/4, with Δμ(φ′) = au₁/(1 − ε) ≥ a(D_lo − η₁) and
    Var φ′ ≤ (2A₀√Var x + a)².
  - Hence L*(θ) − L*(θ′) ≥ sε[(a(D_lo − η₁) − s(2A₀√Var x + a)²)/4 − aK₁/|α₂|]. This is > 0 for |α₂| > W₁, where
    W₁ := 4aK₁/(a(D_lo − η₁) − s(2A₀√Var x + a)²). Also θ′ ∈ K_{2A₀}.
  - **This is the precise sense in which escape loses.** The escaping unit uses ε of the ℓ₁ budget, which costs
    sε·aD\*/4 at first order in Δμ. Its ramp can be produced for nothing by the mirror fold. The averaging of its sine
    part gives back only O(sε/|α₂|).
- **Step 3 (existence; proof of (a)).** Put W₂ := max(2A₀, W₁(s₂)).
  - L* is continuous on the compact K_{W₂}, so it attains its minimum there, at some θ_m.
  - Every θ ∉ Λ_s has L*(θ) > L*(θ_pair) ≥ L*(θ_m).
  - Every θ ∈ Λ_s that lies outside K_{W₂}, or has a zero weight, is strictly beaten by a point of K_{W₂} (Steps 1–2).
    ∎(a)
- **Monotonicity in s.** Every quantity in Steps 0–2 (η₀, ε̄, η₁, k_c, the R-term, K₁) increases with s, and every
  margin decreases. So the verification at s₂ covers all of (0, s₂], and W₁(s) ≤ W₁(s₂). The tests re-verify at s₂/10,
  and check that Step 1 fails at 3s₂.
- **(b)** Take minimisers θ_n at s_n → 0. They lie in K_{W₂}, so a subsequence converges in Θ₂ to some θ̄.
  - Step 0(i) gives Δμ(θ̄) = aD\*. With uniqueness, every unit of nonzero limit weight has |αᵢ| = α̂ and cos βᵢ = 0,
    with the stated alignment. So φ̄ = c̄x + const − a cos(α̂x), since D_c(α̂) < 0.
  - Comparing θ_n with θ_pair through C1 and (F4): (s²/8)Var φ_n − s⁴R⁴/192 ≤ (s²/8)a²Var(cos α̂x), since
    Δμ(θ_n) ≤ aD\*. So Var φ_n ≤ a²Var(cos α̂x) + s_n²R⁴/24.
  - Cov(x, cos α̂x) = 0, so Var φ̄ = c̄²Var x + a²Var(cos α̂x). In the limit this forces c̄ = 0. ∎(b)
- **(c)** For α in the box, 0.8α < π and 1.2α < π < 2α < 2π (checked). So min_I cos(αx) = cos(0.8α), and max_O cos(αx)
  is attained at |x| = 1.2 or 2.
  - Hence G₊(−a cos αx) = a[cos 0.8α − max(cos 1.2α, cos 2α)] ≥ a·g_lo. This is enclosed over 400 cells of the box,
    which also certify D_c < 0.
  - G is continuous on Θ₂. If unplaced minimisers existed along some s_n → 0, they would have an unplaced limit point.
    That contradicts (b). ∎

**Answer to question (1).**
- **The ramp limit is not strictly worse as a limit object.**
  - The point (σ, α̂, β = ∓π/2, γ = −σα̂) of R is exactly the cancelling pair's function, so it ties at every s.
  - More generally, R ∩ {|c̄| ≤ |α|} is realised inside K_{|α|} (Lemma K8).
- **What is strictly worse at small s** is everything else that escapes:
  - every finite configuration with an escaping unit (Step 2);
  - every ramp limit with |c̄| ≥ |α| (Step 1, with ε = 0).
- That is what compactness needs.
- **Float illustration (exploratory; not part of the proof).** At s = 0.005 and a = 1.30, take the escaping sequence:
  unit 1 at α\* with weight 1 − ε, and unit 2 carrying the cancelling slope. Its loss exceeds the pair's by
  2.55·10⁻⁴, 2.55·10⁻⁵ and 2.55·10⁻⁶ at ε = 0.1, 0.01 and 0.001. The first-order prediction sεaD\*/4 gives 2.56·10⁻⁴,
  2.56·10⁻⁵ and 2.56·10⁻⁶.

### 22.3 The symmetric quadratures P_ex and P_fl: where the argument fails

- **Step 0(ii) fails.** Lemma K6(ii) uses the decay of D_c. On P_ex, D_ex is 2π/q-periodic and equals −2 on every odd
  alias. So a large α carries no penalty at all, and units at large |α| need not have small weight.
- **Step 2 fails.** Lemma K9 has no analogue. The escaping unit's sine part does not average out, and its first-order
  cost of escape can be zero: it can sit on an alias.
- **There is an exact non-compact symmetry** (Proposition K10).
- So **width-2 compactness on P_ex and P_fl is not proved**. The natural statement would be compactness modulo the alias
  lattice; it is not attempted. §11's correction stands: state the width-2 small-scale selection on the continuous
  windows (Theorem K11), or for s above about 10⁻¹¹.

### 22.4 The asymmetric continuous windows P_a: the infimum is a ramp limit and is not attained (proved; constants certified)

**Constants (certified).**
- m, Var x and E|x| are exact (22.0). F_τ := 1 − (largest mass of a closed interval of length 2τ) = ½ at τ = 1. This is
  exact: the mass is piecewise linear in the interval's start, so it is maximised at a breakpoint.
- **Logistic regression.** L₀ := min_{κ,b} E ℓ(κx + b, y). Its unique minimiser (κ\*, b\*) lies within 5.7·10⁻¹⁶ of
  (0.2825488448, −0.0621292115).
  - Certified: the gradient is enclosed by rigorous integration (‖∇‖ ≤ 1.2·10⁻¹⁶), and the objective is λ-strongly
    convex with λ ≥ 0.2036 on a ball of radius 10⁻⁶.
  - Also L₀ ≤ log 2 − m²/(8Var x) = 0.677917. This is exact, from ℓ ≤ log 2 + (½ − y)z + z²/8.
- **The residual-weighted sine gap** is Γ(ω) := E[(y − σ(κ\*x + b\*))·ω(x)].
  - For ω₀ = a sin(α₀x + β₀), with α₀ = 1.687111 and β₀ = −1.689795 (the float maximiser), Γ(ω₀) ≥ aγ₀ with
    **γ₀ = 0.38020445** (certified).
  - By Lemma K9, Γ(a sin(αx + β)) → 0 as |α| → ∞.

**Lemma K12a (the scale profile of a sine part; proved).** For ω = σa sin(αx + β), let
h_ω(t) := min_{κ,b} E ℓ(κx + tω + b, y).
- h_ω is convex on ℝ, because it is a partial minimisation of a jointly convex function.
- It is differentiable, with h_ω′(t) = E[(σ(z_t) − y)ω], by Danskin's theorem (the minimiser is unique).
- h_ω(0) = L₀ and h_ω′(0) = −Γ(ω).
- h_ω′ is (a²/4)-Lipschitz. For all t and δ, h(t + δ) ≤ h(t) + δh′(t) + δ²Eω²/8; for a convex differentiable function,
  that uniform quadratic upper bound implies a Lipschitz derivative.
- Hence, for all t ≥ 0:

  **L₀ − tΓ(ω) ≤ h_ω(t) ≤ L₀ − tΓ(ω) + t²a²/8.** (22.4a)

- For φ̄ ∈ R with sine part ω, L*(φ̄; s) ≥ h_ω(s), with equality at the optimal slope. So L̂(s) := inf_R L* = inf_ω h_ω(s).

**Lemma K12b (the slope bound; proved).** If L*(θ; s) ≤ log 2, for θ ∈ Θ₂ or θ ∈ R, then
s|c| ≤ κ̄(s) := max(sa/τ, [F log 2 + sa(1 + F)/2]/(Fτ/2 − m/4)), with τ = 1 and F = ½.
- *Proof.* Use C1; Δμ ≤ |c|m + 2a; C3's spreading (mass at least F lies outside an interval of length 2τ, and has
  |φ − c₀| ≥ τ|c| − a there); and ψ(z) ≥ |z|/2 − log 2. ∎
- κ̄ = 2.52–2.54 at s_a.

**Theorem K12 (non-attainment; proved, constants certified).** For each tabulated a and every s ∈ (0, s_a(a)], put
K₁ := 13/8 + κ̄/4 and W_a := 2K₁/(γ₀ − 7sa/8). Then:
- **(a)** inf_{Θ₂} L*_{P_a}(·; s) = L̂(s), and L̂(s) is **attained on R**, at points with |α| ≤ W_R := K₁/(γ₀ − sa/8)
  and s|c̄| ≤ κ̄.
- **(b)** **No θ ∈ Θ₂ attains it**: L*(θ; s) > L̂(s) for every θ ∈ Θ₂.
- **(c)** Every minimising sequence θⁿ in Θ₂ escapes to the ramp limit. After relabelling the units:
  - ṽ₂ⁿ → 0 and |α₂ⁿ| → ∞;
  - a subsequence converges in the sense of Lemma K7 to a minimiser on R.

| a | s_a (verified) | W_a | W_R | s_u (P_a) | s_u (P_an) |
|---|---|---|---|---|---|
| 1.02 | 0.0082 | 12.10 | 5.95 | 0.0258 | 0.0258 |
| 1.30 | 0.0076 | 12.15 | 5.96 | 0.0202 | 0.0202 |
| 1.40 | 0.0074 | 12.17 | 5.96 | 0.0188 | 0.0188 |
| 1.50 | 0.0072 | 12.18 | 5.96 | 0.0175 | 0.0175 |
| 1.60 | 0.0071 | 12.20 | 5.96 | 0.0165 | 0.0164 |

**Proof.**
- **The comparison point.** Let U(s) := h_{ω₀}(s). Then U(s) ≥ L̂(s), and by (22.4a), U(s) ≤ L₀ − saγ₀ + s²a²/8.
- **The assumption.** Let θ ∈ Θ₂ with L*(θ; s) ≤ L̂(s); a minimiser would satisfy this.
  - Then L*(θ) ≤ L₀ ≤ log 2, so s|c| ≤ κ̄ (Lemma K12b).
  - Units of zero weight do not affect φ. Put W := W_a(s).
- **Case A: every unit of nonzero weight has |αᵢ| ≤ W.**
  - Then |c| ≤ W and Δμ ≤ Wm + 2a.
  - By (F2), L*(θ) ≥ log 2 − (s/4)(Wm + 2a) > log 2 − m²/(8Var x) ≥ L₀ ≥ U(s) ≥ L̂(s). Contradiction.
  - The strict inequality is the certified condition at s_a, and it is monotone in s.
- **Case B: the nonzero-weight unit with the largest |α| (call it unit 2) has |α₂| > W.** Put ε := |ṽ₂| ∈ (0, 1].
  - **(B1) ε = 1.** Then φ = cx + r + const, with r = ±a sin(α₂x + β₂).
    - By convexity at z₀ = scx + b_θ, and Lemma K9 with |z₀′| ≤ κ̄: L*(θ) ≥ L₀ − saK₁/|α₂|.
    - This is > L₀ − saγ₀ + s²a²/8 ≥ U(s), since |α₂| > W ≥ W_R. Contradiction.
  - **(B2) ε < 1.** Unit 1 has weight 1 − ε and sine part ω = σa sin(α₁x + β₁), with σ = sign ṽ₁. So
    φ = cx + (1 − ε)ω + r + const.
    - By convexity at z₀ = scx + s(1 − ε)ω + b_θ (with |z₀′| ≤ κ̄ + sa|α₁| ≤ κ̄ + sa|α₂|), and Lemma K9:

      L*(θ) ≥ h_ω(s(1 − ε)) − e, with e := sεa(K₁/|α₂| + sa/4). (22.4b)

    - From L*(θ) ≤ U(s): h_ω(s(1 − ε)) ≤ U(s) + e.
    - By convexity, h_ω′(0) ≤ (h_ω(s(1 − ε)) − L₀)/(s(1 − ε)) ≤ (U(s) + e − L₀)/s. The numerator is negative, by the
      same condition as below.
    - With Lemma K12a: −h_ω′(s) ≥ (L₀ − U(s) − e)/s − sa²/4 ≥ aγ₀ − sa²/8 − a(K₁/|α₂| + sa/4) − sa²/4.
    - Convexity once more: h_ω(s(1 − ε)) ≥ h_ω(s) − sε·h_ω′(s). With (22.4b):

      **L*(θ) − h_ω(s) ≥ sε[−h_ω′(s) − a(K₁/|α₂| + sa/4)] ≥ sεa[γ₀ − 7sa/8 − 2K₁/|α₂|] > 0** for |α₂| > W_a.

    - h_ω(s) ≥ L̂(s), because it is the loss of the ramp-limit point with sine part ω and the optimal slope. So
      L*(θ) > L̂(s). Contradiction.
  - So no θ ∈ Θ₂ has L*(θ) ≤ L̂(s). ∎(b)
- **(a)**
  - *inf over Θ₂ ≤ L̂(s).* Every φ̄ ∈ R is the limit of a sequence as in Lemma K7: ṽ₂ = 1/n, α₂ = nγ, ṽ₁ = σ(1 − 1/n).
    L* converges along it by (F1). With (b), inf_{Θ₂} = L̂(s).
  - *Attainment on R.* Take φ̄ ∈ R with sine part ω and |α| > W_R. Then L*(φ̄) ≥ h_ω(s) ≥ L₀ − sΓ(ω) ≥ L₀ − saK₁/|α|,
    which is > U(s). This uses Lemma K9 at z = κ\*x + b\*, with |κ\*| ≤ κ̄.
  - Also s|c̄| ≤ κ̄ on {L* ≤ log 2}. So L̂(s) is an infimum over the compact set {s|c̄| ≤ κ̄, |α| ≤ W_R, β mod 2π, σ},
    on which L* is continuous. It is attained. ∎(a)
- **(c)** Let L*(θⁿ) ≤ L̂(s) + δ_n with δ_n → 0.
  - If every unit of nonzero weight had |αᵢⁿ| ≤ W_a along a subsequence, Case A would give
    L* ≥ log 2 − (s/4)(W_a m + 2a), a fixed margin above L̂. So eventually |α₂ⁿ| > W_a.
  - Repeating (B2) with δ_n: sε_n·a[γ₀ − 7sa/8 − 2K₁/|α₂ⁿ|] ≤ 2δ_n. So ε_n → 0.
  - |α₁ⁿ| stays bounded. Otherwise both sine parts are fast, and by Lemma K9 twice with (22.4a),
    L*(θⁿ) ≥ L₀ − sa(2K₁/|α₁ⁿ| + sa/4) → L₀ − s²a²/4. That is > U(s), since γ₀ > 3sa/8.
  - Then ṽ₂ⁿα₂ⁿ = cⁿ − ṽ₁ⁿα₁ⁿ is bounded, because s|cⁿ| ≤ κ̄. So a subsequence converges, and Lemma K7 applies.
  - By (F1) the limit φ̄ ∈ R has L*(φ̄) = L̂(s). ∎
- **Monotonicity.** κ̄, K₁ and W_a increase with s, and Case A's margin decreases. So the verification at s_a covers
  (0, s_a]. The tests re-verify at s_a/10 and check failure at 2s_a.
- **Float illustration (exploratory).** At s = 0.005 and a = 1.30, take unit 1 at the float ramp-limit optimum
  (α = 1.70) and unit 2 forced by the optimal slope. Then L* − L̂ = 2.46·10⁻⁴, 2.46·10⁻⁵, 2.47·10⁻⁶ and 7.5·10⁻⁷ at
  ε = 0.1, 0.01, 0.001 and 0.0003. The first-order prediction sεaγ gives 2.47·10⁻⁴·(ε/0.1). The loss decreases
  linearly to the infimum and never reaches it.

**Proposition K13 (unplaced at small scale on asymmetric data; proved, explicit).** Take any balanced objective with
m ≠ 0 whose windows have −2, 2 ∈ O and ±0.8 ∈ I: P_a, P_an, or a Track 2 training set. At width 1 or 2, for points of
Θ₂ ∪ R:
- **(i) (the domain lemma at width 2.)** G₊ and G₋ are both ≤ 2a − 2.8|c|. So |c| ≥ a/1.4 implies unplaced.
  - *Proof:* for c > 0, G₊ ≤ φ(−2) − φ(0.8) and G₋ ≤ φ(−0.8) − φ(2), and |o(u) − o(v)| ≤ 2a. The case c < 0 is the
    mirror image. ∎
- **(ii)** inf L* ≤ log 2 − m²/(8Var x) at every s > 0. Use the pure ramp: units (c, β) and (c, β + π) with weights ½
  and ½ give φ = cx + const. Take c = m/(s·Var x) and apply (F3).
- **(iii)** Every θ with |c| < a/1.4, in particular every placed θ, has L* ≥ log 2 − (s/4)(a|m|/1.4 + 2a), by (F2).
  - So for s < **s_u := m²/(2Var x·(a|m|/1.4 + 2a))**, every placed point lies above the infimum by at least
    m²/(8Var x) − (s/4)(a|m|/1.4 + 2a) > 0.
- s_u is given in the table above (0.0202 at a = 1.30, on both P_a and P_an).
- This is the width-2 analogue of §11's s₀, and it needs neither attainment nor compactness.
- It is consistent with T2-1, whose registered switch lies at s ∈ [0.437, 0.453], above s_u.

### 22.5 Theorem 1 on the extended space Θ̂ := Θ₂ ∪ R (P_a)

- **The topology.** Θ₂ keeps its own topology, and θⁿ → φ̄ ∈ R as in Lemma K7. L* extends continuously, by (F1).
- **What Theorem K12 gives.** At each s ≤ s_a:
  - the minimum over Θ̂ exists and lies on R;
  - every minimising sequence in Θ₂ has a subsequence converging in Θ̂ to a minimiser. The proof of (c) shows that
    {L* ≤ L̂ + δ} is relatively sequentially compact in Θ̂ for small δ.

| ingredient of Theorem 1 | on Θ₂ (P_a) | on Θ̂ (P_a) |
|---|---|---|
| a conditional minimiser exists at fixed small s | **fails** (K12(b), s ≤ s_a) | **holds** (K12(a)) |
| C1 decomposition; Lemma 1's expansion; Lemma K1 | hold | hold: they are statements about φ |
| Corollary 1(i): limit points maximise Δμ (needs S < ∞ and compactness) | **fails**: S = +∞ (c·m is unbounded), and minimisers leave every K_W (Proposition C′) | **still fails** in raw coordinates: s·c̄ → κ\* > 0, so \|c̄\| → ∞ |
| Corollary 1(i), rescaled (new) | — | **holds** (proved below): in the coordinates (κ = s·c̄, σ, α, β) on R, limit points have κ = κ\* = 0.2825 and (σ, α, β) maximising Γ |
| Corollary 1(ii) (variance selection) | not examined | not examined: the first-order selection is Γ, and its tie structure is not studied |
| Corollary 2 / "unplaced at small scale" | **holds** (K13; s_u explicit) | **holds** (K13 applies on R) |
| Lemma 2 (large s: Γ_n attained, compactness) | not examined here; T2-1 found placed minimisers for s ≥ 0.453 (registered, numerical) | **open**: whether ramp limits raise the large-s gap supremum is not examined |
| the criterion (gating iff unplaced at small scale) | small-scale side holds (K13); large-scale side is T2-1's registered result | small-scale side holds; large-scale side open |

**Rescaled Corollary 1(i) on R (proved).** Let φ_s ∈ R be minimisers, with sine parts ω_s.
- By (22.4a), h_{ω_s}(s) ≥ L₀ − sΓ(ω_s), and L̂(s) ≤ L₀ − sΓ\* + s²a²/8, where Γ\* := sup Γ.
- Γ\* is attained: Γ(σa sin(αx + β)) is continuous in (α, β), and tends to 0 as |α| → ∞ by Lemma K9.
- So Γ(ω_s) ≥ Γ\* − sa²/8. Hence |α_s| ≤ K₁/(Γ\*/a − sa/8) stays bounded, and limit points maximise Γ.
- κ_s = s·c̄_s minimises the strictly convex E ℓ(κx + sω_s + b). That differs from the logistic objective by at most
  sa in sup norm. By uniform convergence on compacts and uniqueness of (κ\*, b\*), κ_s → κ\*. ∎
- **So on asymmetric windows the small-scale limit is a statement in rescaled coordinates:** the logistic-regression
  ramp κ\*x/s, plus the full-weight sine selected by the residual-weighted gap Γ.
  - The float maximiser is α₀ = 1.687, against α\* = 1.792 for D_c on the symmetric windows.
  - In raw coordinates nothing converges.

### 22.6 Exploratory: Track 2's registered landscape P_an (float; not part of any proof)

- **Method.** The ramp-limit minimum on P_an at a = 1.30: a 2-D Newton solve in (κ, b) for each (α, β) on a 150 × 72
  grid, then pattern refinement. This is a scratch computation; it is not committed.
- **At the registered scales s ≥ 0.178** (0.178, 0.237, 0.316, 0.422 and 0.562, and both T2-1 bracket ends 0.43714 and
  0.45316):
  - the ramp-limit minimum equals T2-1's registered retained loss (`asym_parts/scan.csv`, `validate.csv`) to
    ≤ 1.7·10⁻¹⁵;
  - its slope is inside the mirror range (|c̄| = 1.48 down to 0.42, against |α| = 1.686).
  - So there the registered minimiser **is** the ramp-limit minimiser, realised compactly (Lemma K8).
- **At s = 0.1 and 0.1334** the ramp-limit minimum is lower than the registered retained loss, by 1.14·10⁻³ and
  3.4·10⁻⁴. Its slope is |c̄| = 2.71 and 2.01, both above |α|.
  - So there the registered "conditional minimiser" is not the infimum.
  - By analogy with Theorem K12 the infimum is presumably not attained there (**argued**). P_an is a finite set, where
    K12 is not proved.
  - Both ramp-limit points have |c̄| > a/1.4 = 0.93, so they are unplaced (K13(i)).
- **T2-1's status at every scanned scale, and its verdict (PASS), are unchanged.** No registered number is altered.

### 22.7 Summary: what is proved, and what fails

- **Proved:**
  - the ramp limit (K7);
  - the mirror fold (K8);
  - averaging on piecewise-uniform measures (K9);
  - the exact alias translations (K10);
  - width-2 compactness on P_c, with certified constants, the cancelling-pair limit and its placement (K11);
  - non-attainment and escape on P_a, with certified constants (K12);
  - unplaced at small scale on asymmetric data (K13);
  - the rescaled Corollary 1(i) on Θ̂.
- **Certified (new):** uniqueness of the class-mean maximiser on P_c (D_c″ > 0 on the maximiser box), and the
  logistic-regression optimum and residual gap on P_a.
- **Fails, or is not proved:**
  - width-2 compactness on the quadratures P_ex and P_fl (22.3: aliasing and K10);
  - non-attainment on the finite P_an (K9 fails on a finite set, so it is only argued, 22.6);
  - Corollary 1(i) on asymmetric windows in raw coordinates, on Θ₂ and on Θ̂ alike;
  - the large-s side of Theorem 1 on Θ̂;
  - Corollary 1(ii) on P_a.
- s\* in Theorem K11(c) is not quantified.
- **Producer and tests.**
  - The run is one process at nice 15: about 2 s of CPU and 0.27 GB peak RSS.
  - Tests cover:
    - the mirror-fold identity, with a wrong-mirror fail case;
    - the ramp limit's uniform error a·ε;
    - Lemma K9 on smooth G, and a square-wave fail case without the variation term;
    - the localisation pass, with two fail cases (a threshold above the maximum, and one too low for the small-α
      clearance);
    - Theorem K11's conditions passing at s₂ and s₂/10 and failing at 3s₂, plus an α_L fail case;
    - the closed form of D_c″ against finite differences, and the uniqueness certificate with a wide-box fail case;
    - the pair-variance box against floats;
    - placement on the box, with a shifted-box fail case;
    - exact moments and interval masses on constructed measures;
    - the logistic certificate, with a perturbed-start fail case;
    - γ₀, with an opposite-phase fail case;
    - Theorem K12's conditions passing at s_a and s_a/10 and failing at 2s_a, plus no-gap and symmetric-window fail
      cases;
    - the s_u formula;
    - the committed numbers.
