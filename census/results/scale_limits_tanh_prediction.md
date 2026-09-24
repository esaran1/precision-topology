# Registration: the small-scale limit for tanh at width 2 (the third registered case)

**For the rebuttal revision** (not the submission). **Written and committed before any tanh Δμ computation.** Theory:
`math_note_v2.md` §10 and §10.1. Producer: `src/scale_limits_tanh.py`, committed with this file. Requested by the
author on 2026-09-24 ("make the tanh case registered"), with the expected outcome stated first.

## Expected outcome (stated first)

**A supremum approached by placed configurations and not attained.**
- sup Δμ = 1 over width-2 tanh networks with ‖ṽ‖₁ = 1, and no finite parameter attains it.
- Along every box ladder, the Var-selected near-maximiser is the **symmetric step pair**. It is placed (G₊ > 0), and
  its G₊ rises toward 1.
- At first order it **ties** with the single steep units, which are unplaced (G₊ < 0) at every finite α. This is the
  same kind of tie as for f_a, and it is again decided by the second-order (Var) selection.

The author's competing alternative is **an attained placed maximiser**. The argument below rules it out analytically,
so observing it would mean the search or the argument is wrong (see the stop conditions).

## The analysis behind the expectation (done before computing)

- **Data.** On the 800-point population, O₋ = −O₊ and I = −I, the classes are balanced, and every O₋ point lies
  below every I point, which lies below every O₊ point.
- **One unit (upper bound).** u = tanh(αx + β) is monotone. Take α > 0; α < 0 is the mirror image. Write m₋, m_I,
  m₊ for its means on O₋, I and O₊. Then
  - D = mean_O u − mean_I u = (m₊ − m_I)/2 + (m₋ − m_I)/2;
  - monotonicity gives m₋ ≤ m_I;
  - |tanh| < 1 gives m₊ − m_I < 2.
  - So **D < 1 strictly at every finite (α, β)**, and likewise D > −1.
- **Width 2.** Δμ = Σṽᵢ Dᵢ ≤ maxᵢ |Dᵢ| < 1, so **the supremum is not attained**.
- **The supremum is 1.** Take the step tanh(α(x − c)) with c ∈ (0.8, 1.2) and α → ∞. Its values tend to +1 on O₊ and
  −1 on O₋ ∪ I, so D → 1.
- **The first-order tie.** Let s₊ = lim tanh(α(x − c)) and s₋ = lim tanh(α(x + c)). Every mixture
  φ_t = t·s₊ − (1 − t)·s₋ (t ∈ [0, 1], ‖ṽ‖₁ = 1) has Δμ → 1. In the limit its values are:

  | points | value of φ_t |
  |---|---|
  | O₊ | 2t − 1 |
  | I | −1 |
  | O₋ | 1 − 2t |

  - So Var(φ_t) = 0.25 + 0.5·(2t − 1)² and G₊(φ_t) = 1 − |2t − 1|.
  - The **second-order selection picks t = ½** (the symmetric step pair, a "bump"), with Var → 0.25 and G₊ → 1.
  - The single units (t = 0 or 1) have Var → 0.75. At every finite α their G₊ < 0: the far outer class sits below
    the inner class.
- **Where tanh sits relative to the criterion.** The criterion in §10.1 assumes that Γ_n is attained (Lemma 2). For
  tanh at width 2 it is not: G_n < 1 at every finite α. So tanh lies **outside the criterion's hypotheses**. What is
  registered here is the small-scale limit itself, i.e. whether the configurations approaching the supremum, under
  the Var selection, are placed.

## What is computed

- **Population objective** (800 points), width 2, tanh, ‖ṽ‖₁ = 1, with the same parametrisation as the f_a width-2
  computation: (α₁, β₁, α₂, β₂, t, σ).
- **Box ladder A ∈ {5, 10, 20, 40}** with the constraint |αᵢ| ≤ A, imposed by αᵢ = A·sin qᵢ; β is unconstrained.
  - The ladder stops at 40 because 1 − Δμ decays like e^{−0.4A} (the data reach the gap edges ±0.8 and ±1.2
    exactly). By A = 80 the difference would fall below double-precision resolution.
- **Per box**:
  - 2,000 BFGS restarts, every one retained;
  - a restart ladder 500 → 2,000 (the maximum unchanged to 1e−9);
  - an independent CMA-ES search in the alternative parametrisation (v unconstrained and normalised, α clipped to
    the box), which must not exceed the retained maximum by more than 1e−9.
- **Analytic candidates per box**, with c optimised by a dense grid on [0.8, 1.2] (step 1e−4) plus golden section:
  - the single step at |α| = A, tanh(A(x − c));
  - the symmetric pair ½tanh(A(x − c)) − ½tanh(A(x + c)).
  - Both must attain the box maximum to 1e−9. This is the first-order tie.
- **Var selection**: among the retained candidates within 1e−9 of the box maximum, plus the two analytic
  candidates, take the one with the smallest Var(φ).
- **Reported per box**:
  - the box maximum and 1 − max;
  - max|αᵢ|/A of the maximisers;
  - the selected member's G₊ (exact extrema, `width2_geometry.gaps`), G_n and Var;
  - the G₊ range of each type found (single unit, pair).

## Registered outcomes and stop conditions

- **T1 (expected): supremum approached by placed configurations, not attained.** All of the following hold:
  - at every rung, the box maximum is < 1 and exceeds the previous rung's by more than 1e−9;
  - the maximisers sit on the box boundary (max|αᵢ| ≥ 0.999·A);
  - the Var-selected member has **G₊ > 0 at every A**.
  - Recorded as the third case: *width 2, tanh: small-scale limit placed (not attained); outside the criterion's
    attainment hypothesis*.
- **T2 (the author's alternative): an attained placed maximiser.** The rung-to-rung increase is ≤ 1e−9 from A = 20
  to 40, with an interior maximiser (max|αᵢ| < 0.999·A) and G₊ > 0. This contradicts the bound Δμ < 1 = sup Δμ
  above: **stop and report**. Either the search failed to reach the boundary or the argument has an error; neither
  is resolved after the fact.
- **Stop if the Var-selected member has G₊ ≤ 0 at any A.** That would contradict the limit analysis. Stop and report.
- **Stop if any validation fails**: the ladder, the independent search, or the analytic candidates not attaining
  the maximum.
- Any other pattern is reported as "neither", with every number.

## Disclosed prior exposure (not blind)

- The exploratory pilot scan (`width2_pilot_scan.csv`, design revision 2) found tanh conditional minimisers with
  G₊ = 1 and α → ∞ at every scanned scale.
- The analysis above was done before this registration.
- The outcome is therefore **largely predictable**. Registering fixes the ladder, the validation, the selection and
  the stop conditions before the numbers exist.
- **Exposure from the producer's tests (disclosed, 2026-09-24).** The unit tests (`tests/test_scale_limits_tanh.py`)
  were run before this file was committed. They report pass or fail only, but in doing so they evaluated:
  - the two analytic candidates at A = 5 and 20: their Δμ tie to 1e−12, the single step's G₊ < 0 < the pair's G₊,
    and the pair's smaller Var;
  - the single step's Δμ at A = 5, 10, 20 and 40: rising, with 1 − Δμ(40) < 1e−5.

  The test run also caught a bug in `analytic`, which passed a (p, σ) tuple as p. It was fixed before this commit.
  The search, its validation, the Var selection over the retained maximisers and the verdict have **not** been
  computed.

## Result — computed 2026-09-24 with the committed producer (`python -m src.scale_limits_tanh`, 807 s, 0.2 GB)

**Registered outcome: "neither".** One of T1's conditions failed at the last rung. No stop condition fired: all
validation passed, the Var-selected member is placed at every A, and T2 is not indicated.

| A | box maximum | 1 − max | ladder / independent / analytic tie | maximisers on the boundary (min max\|αᵢ\|/A) | single-unit G₊ | selected member | selected G₊ | selected Var |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.9677930 | 3.22e−2 | pass / pass / pass | yes (1.0000) | −0.2381 | symmetric pair (t = ½) | +0.7616 | 0.2349 |
| 10 | 0.9976487 | 2.35e−3 | pass / pass / pass | yes (1.0000) | −0.0359 | symmetric pair | +0.9640 | 0.2488 |
| 20 | 0.9999774 | 2.26e−5 | pass / pass / pass | yes (1.0000) | −6.70e−4 | symmetric pair | +0.99933 | 0.249989 |
| 40 | 0.999999996 | 4.10e−9 | pass / pass / pass | **no (0.9908)** | −2.9e−7 to −2.1e−7 | symmetric pair | +0.9999998 | 0.250000 |

- **T1 conditions.**
  - Box maximum < 1 and rising by more than 1e−9 at every rung: **yes**. The increments are 3.0e−2, 2.3e−3 and
    2.26e−5.
  - Var-selected member G₊ > 0 at every A: **yes**.
  - All retained maximisers on the boundary (max|αᵢ| ≥ 0.999·A): **yes at A = 5, 10, 20; no at A = 40**.
- **The first-order tie, as predicted.** At every A, the analytic single step and the analytic symmetric pair both
  attain the box maximum to 1e−9. The retained maximisers are single units (1,057 / 948 / 806 / 741) and pairs
  (509 / 452 / 432 / 366). The single units are unplaced at every A. The Var selection picks the symmetric pair,
  whose Var → 0.25 and G₊ → 1, the limits derived above.
- **Why A = 40 failed the boundary condition** (post hoc, labelled).
  - At A = 40, 1 − max = 4.1e−9, the same order as the registered tie tolerance of 1e−9.
  - 9 of the 1,107 retained near-maximisers have max|αᵢ| between 0.9908·A and 0.999·A. All are within 8.0e−10 of
    the maximum: Δμ is flat to within the tolerance that close to saturation.
  - The best candidate itself is on the boundary (max|αᵢ| = A) at every rung.
  - I did not foresee this interaction when I chose A = 40 (only the double-precision floor at A = 80). It is a flaw
    in the registered criterion, not in the search.
- **Post hoc reading (not a registered verdict).** Every number is what T1 predicted. With the boundary condition
  applied to the best candidate instead of to every member of the tie set, T1 would hold at all four rungs. That
  reading was formed after seeing the outcome, so it is not scored. The registered outcome stays "neither".
- **Analytic status, independent of this computation.** The bound Δμ < 1 = sup Δμ (above) proves non-attainment.
  The limit analysis gives the Var-selected configurations G₊ → 1. The computation confirms both numerically at
  every rung but did not meet its own registered boundary criterion at A = 40.
