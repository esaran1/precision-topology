# Registration: the first-order coefficient c₁ at small ε

**Written before any certified Ĝ(a) or R_glob(a) at a = 1.01, 1.02, 1.03 or 1.04 is computed for this test.**
Date: 2026-09-23 13:55 EDT. Producer: `src/first_order.py` (`finite`, `score`), committed with this file.

## Prediction (computed independently of unconstrained training trajectories)

`first_order_c1.csv` and `math_note_v2.md` §8 give the complete first-order calculation: the switch shift by
the implicit-function theorem plus the correction to K.

  R_glob(ε) = R_glob^∞ (1 + c₁ε + O(ε²)),   **c₁ ∈ [0.2852300, 0.2852303]**,
  with A′(0)/A* ∈ [0.6621547, 0.6621550] and k₁ ∈ [−0.37692472, −0.37692472].

R_glob^∞ ∈ [0.1985926, 0.1985927] (Krawczyk A* × the K enclosure). This is conditional on the annulus
certificate (math_note_v2 (c)).

## Procedure (a = 1.01, 1.02, 1.03, 1.04; ε = a − 1; 800-point quadrature objective, as in Block 1c)

- **Ĝ(a)**: `ghat_bnb.certify(a, target_rel = 1e−6)`, a global branch and bound.
- **R_glob(a)**: the Block 1c bracket procedure (`conditional_certified.evaluate`: certified m₋ against m₊,
  tolerance 1e−7, tightened to 1e−9).
  - Start at s₀ = A*/ε^{3/2}, the limit value, not the prediction.
  - Step outward by 1% until the certified status flips.
  - Bisect until the relative width in s is ≤ 2e−4.
  - If a midpoint is unresolved, stop and use the last resolved bracket. Its width is reported, and the
    tolerance below adapts to it.
- **R interval**: [s_lo·Ĝ_lo/2, s_hi·Ĝ_hi/2].

## Registered test (primary)

- **Feasible set** C: the set of c₁ for which some R∞ in its interval and some c₂ ∈ ℝ make
  R∞(1 + c₁ε + c₂ε²) pass through all four certified R_glob intervals.
  - Each interval is widened by ±ε³·R∞, which allows an O(ε³) term with |c₃| ≤ 1.
  - C is computed exactly (a 2D linear program per R∞).
  - The tolerance is therefore set by the bracket widths actually attained.
- **PASS** iff the predicted c₁ interval meets C.
- **FAIL** iff it does not, or if C is empty (no law of that form fits). Either way, the perturbation
  calculation or its hypotheses contain an error, and it is reported.
- **INCONCLUSIVE** iff C is wider than 0.1 (brackets too coarse to test). It is reported as such, not as
  a pass.
- **Competing values**, each reported as excluded or not by C:
  - 0.49, the earlier incomplete prediction;
  - −0.377, the K correction alone;
  - 0.662, the switch shift alone;
  - 0, no first-order term.
- **If PASS**: the large-ε mismatch (the one-term law fails, §5) is higher order, and the note says so.
- **Not discriminated by design**: the large-ε two-term range [0.243, 0.321] contains 0.285. This test
  checks the derivation; it does not separate it from that fit.

## Registered component tests (secondary)

The same feasible-set construction is applied to each component separately:
- **k₁**: on K(ε) = Ĝ(a)/ε^{3/2}, against K ∈ the branch-and-bound enclosure.
- **A′(0)/A***: on A_ε = s·ε^{3/2}, against the Krawczyk A*.

## Disclosed prior exposure

- **Y1 grid R_glob at a = 1.01–1.04** (`crossfamily_followup_results.md`) was seen, at resolution
  ΔR = 0.00215. That is about 1% of R, the same size as the whole first-order effect at ε = 0.04 (0.23%
  of R∞ per 0.01 in ε), so it cannot resolve c₁.
- **Certified Ĝ(1.02)** (`ghat_certified_all.csv`, relative width 6e−4) was seen. The new run recomputes
  it at 1e−6.
- No other quantity at these a was computed.

## Supplementary analysis — 2026-09-23 19:19 EDT (written and committed before the registered result exists or is scored)

- **Reported beside the registered verdict, which stands whatever this shows.**
- **Why**: the registered bracket rule compares the certified regional minima on either side of G = 0. At
  a continuous crossing those minima converge at the switch, so the comparison loses resolution where
  it is needed, and the rule stops at "unresolved".
- **Supplementary measurement** (`first_order.supplementary`, `score_supplementary`): the same
  quadrature objective, the same branch and the same quantity, measured along the branch minimiser.
  - **Switch point**: a Krawczyk test certifies a unique zero of Φ(w₁, b₁, b₂, s) = (∇_{w₁,b₁,b₂} L, G)
    in a small box, with G on the active pair (outer x = −1.2, inner x = 0.8), in interval arithmetic
    at 30 digits. That gives a bracket for the switch scale s.
  - **Branch**: the (w₁, b₁, b₂) Hessian is positive definite over the box, so the zero is the branch
    minimiser and locally unique.
  - **Active pair**: it is unique over the box; the other edges and the interior critical points of f_a
    are strictly dominated.
  - **Global minimiser**: a certified competitor bound (`profiled_bnb.competitor_gap`, balls of
    radius 0.25√ε around the branch and its mirror) shows it is the global minimiser at the switch.
- **R brackets**: [s_lo·Ĝ_lo/2, s_hi·Ĝ_hi/2], with Ĝ(a) from the registered run.
- **Feasible c₁ set**: the registered construction (`_feasible_c1`: same limit interval, same O(ε³)
  allowance), reported with the prediction's position relative to it. **No verdict attaches to it.**
- **Tried beforehand only at a = 1.05**, outside the registered set: certified, with a bracket width of
  1e−9 in s and a competitor margin of +0.0135.

## Status and a second supplementary cross-check — 2026-09-23 22:58 EDT (before the registered a = 1.01 result exists)

- **The registered computation at a = 1.01 continues unchanged.**
  - Its first step is the certified Ĝ(1.01) (`ghat_bnb.certify`, target 1e−6). After six hours it had
    grown to 15 GB, which threatened the machine.
  - It is **paused** (SIGSTOP), not killed. Nothing is lost, including the in-memory results for
    a = 1.02–1.04.
  - It resumes **alone** once the heavier jobs finish. **The registered result is the verdict.**
- **Supplementary cross-check of Ĝ(a), a = 1.01–1.04** (`first_order.ghat_rescaled` →
  `first_order_ghat_rescaled.csv`, committed before it runs). It is reported beside the registered Ĝ.
  - It certifies the same global supremum to the same relative target, split into two parts:
    - **the rescaled box** w₁ = √ε u, b₁ = π + √ε v, (u, v) ∈ [0, 8] × [−12, 12]: a branch and bound
      with exact extrema of φ_ε and step |φ_ε′| ≤ 1 + aσ²/2, relative target 1e−6;
    - **an exclusion branch and bound** over the rest of `ghat_bnb`'s domain (w₁ ∈ (0, a], b₁ ∈ [0, 2π),
      its step (1 + a)(2.8h_w + 2h_b)), proving that every point outside the box has G below the box's
      attained value.
  - Ĝ ∈ ε^{3/2}[K_lo, K_hi] iff the exclusion closes.
  - **Tried beforehand only at a = 1.05**, outside the test set: [0.00636009, 0.00636009], consistent
    with the registered certified value there (0.006360–0.006361), in 1.6 s at 0.6 GB.

## Amendment 2 — 2026-09-24 00:51 EDT (before any a = 1.01 verdict exists)

- **What changes**: at **a = 1.01 only**, the registered Ĝ step (`ghat_bnb.certify`, relative target 1e−6)
  is replaced by the **rescaled certified search** (`first_order.ghat_rescaled`: the box branch and bound
  plus the exclusion branch and bound over the rest of the domain).
- **Reason**: the registered Ĝ step at a = 1.01 **exceeded the available memory**. It reached 15 GB after six
  hours; the job was paused, and it was then lost in a machine crash. At ε = 0.01 the supremum is about 6e−4,
  while that routine's Lipschitz step does not shrink with ε.
- **The quantity and the tolerance are unchanged**: the same global supremum of the class gap over placements,
  certified to the same 1e−6 relative target. It was already computed and committed, before any a = 1.01
  verdict existed: **Ĝ(1.01) ∈ [0.00057731, 0.00057731]**, exclusion closed (`first_order_ghat_rescaled.csv`).
- **Unchanged**: the switch bracket at a = 1.01 uses the **registered procedure**, run under the memory
  watchdog. **a = 1.02–1.04 stay exactly as registered.**
- The c₁ runs lost in the crash (a = 1.02–1.04, and a = 1.01's unfinished Ĝ step) rerun from this procedure.
