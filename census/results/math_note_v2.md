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
  **R_glob^∞ = KA*/2 ∈ [0.19738, 0.19920]** (`limit_K_base.csv`).
- (c) **Single branch**:
  - for A ∈ [0.66, 0.71], the global minimiser lies in the box U = [p₀ ± 0.05] × [q₀ ± 0.05], with
    (p₀, q₀) = (1.66858, 1.36972);
  - the (p, q, b) Hessian is positive definite on U × [0.66, 0.71], with λ_min ≥ 0.0473;
  - so the global minimiser is the unique critical point there, and it is C^∞ in A.
  - *Competitor bound outside U, uniform in A: PENDING (annulus).*
- (d) **Transversality (H2′)**: dG_0/dA along the branch at A* ≠ 0. *Interval: PENDING (h2).* The
  float value is 1.304.

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
- **Competitor exclusion**: `annulus_job` bounds L*_0 from below on K(24) \ U, uniformly over eight
  A-subintervals, via convexity in A. It is compared with the certified branch upper bound on the
  same subinterval. *PENDING.*

**(d) Transversality.**
- Along the branch, G_0 is differentiable wherever its active set is unique: one extremal point of I
  and one of O.
- dG_0/dA = ∇G_0 · dθ/dA, with dθ/dA = −H⁻¹∂_A∇L_0 (implicit-function theorem on the full
  (p, q, b) gradient system).
- Evaluated in interval arithmetic over the certified argmin enclosure (`ift_limit`, `h2`).
  *PENDING.*

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
| No competitor outside U, uniform in A | PENDING (annulus) | `mn2_annulus.csv` | — |
| Transversal crossing (H2′) at A* | PENDING (interval); float 1.304 | `mn2_h2prime.csv` | — |
| Local branch or global minimiser? | **Global** in the limit on K(24) (localisation, B&B, annulus); **global at finite a** only at a ∈ {1.30, …, 1.60} (Block 1c); **local** otherwise | as above | — |
| Solve threshold: nondegeneracy and transversality | PENDING (limit, `mn2_solve_limit.csv`; finite a, `mn2_solve_finite.csv`) | — | — |
| Other window geometries | **Not done**: A* is certified for the base window only | — | rerun `limit_bnb switch` per window |

## 5. Finite-a certificates (Block 1c)

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
| (c) competitors outside U | annulus | `math_note_v2_checks annulus` → `mn2_annulus.csv` | PENDING |
| (d) transversality | H2′ | `math_note_v2_checks h2` → `mn2_h2prime.csv` | interval arithmetic (PENDING) |
| C² convergence | derivative control | `taylor_constants` → `mn2_neighbourhood.csv` (M₀–M₃) | closed-form bounds, float max over 200,001 points of polynomials |
| finite-a global preference | Block 1c | `conditional_certified brackets` → `cond_certified_brackets.csv` | `mn2_rounding.csv`, a = 1.30 rows: ≤ 1.1e−16 against certified gaps ≥ 1.8e−8 |
| solve threshold | nondegeneracy, transversality | `math_note_v2_checks solve_limit`, `solve_finite` | PENDING |

## 7. Limits

- The corollary is asymptotic: "for all sufficiently small ε". No explicit ε₀ is claimed. The measured
  finite-a thresholds (ε = 0.30–0.60) sit 7–14% above the limit.
- Global preference at finite a rests on Block 1c at the six reported a, not on the limit.
- Everything is exact for the stated finite objective. For another sample (for example a run's own 400
  training points) the threshold differs; see `own_threshold_prediction.md`.
