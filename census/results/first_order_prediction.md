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

## Result — scored 2026-09-24 03:20 EDT with the committed scorers (`first_order score`, `score_supplementary`)

**Registered primary verdict: INCONCLUSIVE.**

- **Why**: the feasible set C = [0.22521, 0.33020] has width 0.105, just over the registered 0.1 limit.
- **The prediction** c₁ ∈ [0.2852300, 0.2852303] lies inside C.
  - By the registered rule that is **not a pass**, and it is not reported as one.
- **Brackets**: all four a stopped at an unresolved midpoint, as the procedure allows, so the last resolved
  bracket was used. Relative widths in s:

  | a | relative width in s |
  |---|---:|
  | 1.01 | 6.2e−4 |
  | 1.02 | 1.2e−3 |
  | 1.03 | 6.2e−4 |
  | 1.04 | 2.5e−3 |

  All four are above the 2e−4 bisection target. The registered tolerance adapts to them, and the
  resulting C is what is too wide.
- **Registered R_glob intervals**: 1.01 [0.199085, 0.199209]; 1.02 [0.199583, 0.199832];
  1.03 [0.200212, 0.200336]; 1.04 [0.200602, 0.201101].
  - Ĝ is converged at every a, with relative width ≤ 7.9e−7.
- **Competing values**, reported as registered:
  - 0.49 (earlier): excluded by C;
  - −0.377 (K correction alone): excluded;
  - 0.662 (switch shift alone): excluded;
  - 0 (no first-order term): excluded.
- **Secondary component tests** (same construction):
  - **k₁: PASS**. The prediction −0.376925 lies in C_k = [−0.377494, −0.376195], width 0.0013.
  - **A′(0)/A\***: INCONCLUSIVE. C_A = [0.6019, 0.7073], width 0.105; the prediction 0.662155 lies inside.
- **R_glob^∞ is no longer conditional.** The registration called the limit interval conditional on the
  annulus certificate, and that certificate now holds (`certv2_annulus_summary.csv`, commit `133cd83`). So
  the limit interval used by C is certified.

**Supplementary branch-root analysis** (committed before the registered result existed; no verdict
attaches to it).
- **Switch scale**: certified at every a by the Krawczyk test.
  - Hessian PD, λ_min ≥ 0.1525.
  - Active pair unique.
  - Global at the switch: competitor margin ≥ +0.0131.
- **s-brackets**: width below 1e−9 relative. Each lies inside the registered bracket at the same a.

  | a | s |
  |---|---:|
  | 1.01 | 689.975926 |
  | 1.02 | 245.539519 |
  | 1.03 | 134.520585 |
  | 1.04 | 87.934061 |

- **Implied feasible c₁ set** (registered construction, registered Ĝ): [0.284596, 0.285895], width 0.0013.
  - The prediction lies inside.
  - All four competing values lie outside it: 0.49 and 0.662 above; −0.377 and 0 below.

**Ĝ cross-check** (`first_order_ghat_rescaled.csv`):
- At a = 1.02–1.04 it is independent of the registered `ghat_bnb` run and agrees with it. The lower bounds
  coincide, and the rescaled upper bounds are nested inside the registered ones. Both exclusions closed.
- At a = 1.01 the two are identical by construction (Amendment 2), so there it is not a cross-check.
