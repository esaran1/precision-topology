# WP-36. Transfer to simplicity bias, redesigned and gated (Track T v2; final night)

A separate file: the main patch (`WRITER_INPUTS_v4_patch.md`) is unchanged. There, WP-29 keeps its one-sentence scope
note about the first pilot. Producer: `src/simplicity_bias*.py` → `results/simplicity_bias*/`.

This attempt:
- adds weight decay on the hidden weights (the same λ in the fixed-scale objective and in training);
- replaces the switch with the scale s_q at which the minimizer's slab usage reaches a level q fixed in advance;
- predicts the training crossing as s_q(1 + κ_q·χ).

The gate, and the registration if the gate passed, were committed before the corresponding computation. The text below
is the Track T writer input.

### Frozen before any comparison

- The design and gate were committed at `af51125` before any v2 computation:
  - λ rule, attainment check, q rule, gate;
  - κ_q, χ and the P rules;
  - seeds 2,710,000–2,710,039;
  - crossing definition;
  - T1, T3 and the validity condition.
- **Amendment 1**, committed at `f565591` before the freeze and before any registered training: the ṡ window is
  min(100, t₀). It is disclosed in the design.
- **No comparison exists.** No registered seed was trained, and no training ρ₂ was ever computed.

### Numbers (from the committed producers)

| Quantity | Value |
|---|---|
| λ (smallest grid value passing the attainment check) | 1e−4 |
| A-priori bound B(λ) | 117.7 |
| Largest retained ‖(W, c)‖ on the path | 16.5 |
| Path scales computed / attained | 54 / 54 |
| Sets A and B retained-loss agreement at every grid scale | ~1e−15 |
| ρ₂ at s = 0.5 / 43.4 | 3.2e−9 / 0.783 |
| q (midway) | 0.3914 |
| s_q (both sets: bracket [3.58512, 3.597642]) | 3.5914 (agreement 0.0%) |
| CMA-ES at bracket ends | nothing lower |
| ρ₂ across the s_q bracket | 0.182 → 0.448 (jump; G₊ −0.153 → +0.071) |
| ∇ρ₂ check (autograd vs central / central vs forward; limit 1e−3) | 0.033 / 0.055, **FAIL** |

**Path shape:**
- ρ₂ = 0 up to s = 1.22. The minimiser is linear-only there: x₁ alone, with the overlap band hedged.
- ρ₂ then rises continuously on a mixed branch, from 0.085 at s = 1.53 to 0.182 at s = 3.585.
- At s ≈ 3.59 it switches discontinuously to a slab-dominant branch (ρ₂ = 0.448, every point separated).
- It then rises continuously to 0.783 at s = 43.4.

### Say / Do not say

**Outcome that occurred: gate passed, registration stopped at the κ_q check.**
- **Say:**
  - "With weight decay (λ = 10⁻⁴, the smallest grid value at which the fixed-scale minimiser is attained), the
    linear-plus-slab benchmark's fixed-scale minimiser is attained at every scale."
  - "Two independent restart sets give identical paths."
  - "The minimiser uses only the linear coordinate at small output scale (ρ₂ ≈ 0 up to s ≈ 1.2) and becomes
    slab-dominant at large scale (ρ₂ = 0.78 at s = 43)."
  - "The transition to slab dominance is a discontinuous switch at s ≈ 3.59, where the minimiser also starts
    separating every point."
  - "A registered training test of the lag was not run. At the switch the slab-usage measure is not differentiable,
    because the slab units are mirror-symmetric, so the registered lag constant κ_q is undefined. The branch jump
    means the continuous linear-response lag would not apply in any case."
  - "The redesign (weight decay, slab-usage path) was one gated attempt, made after the first design's gate failed."
- **Do not say:**
  - that the paper's account predicts when networks abandon the simple feature in the simplicity-bias benchmark;
  - that the lag law transfers to this setting;
  - anything about training crossings on this benchmark, since there are none;
  - "validated/certified switch": the search is validated (ladder, audit, CMA-ES), not certified;
  - that the switch location 3.59 is a prediction confirmed by training.

**Outcome "gate failed"** (did not occur for v2; it did occur for v1):
- **Say:** the v1 statement. Without weight decay the minimiser is not attained, the hidden weights diverge, and x₂ is
  added continuously rather than switched in.
- **Do not say:** that v2 failed its gate.

**Outcome "registered, not scored by 04:15":** did not occur. Nothing was registered.

**Outcome "PASS" / "FAIL":** did not occur. There is no registered test and no verdict. Do not report T1 or T3 in any
form.
