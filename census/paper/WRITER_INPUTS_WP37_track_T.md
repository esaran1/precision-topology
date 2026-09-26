# WP-37. Does training acquire the slab feature just above the attained fixed-scale switch? (Track T v3; registered)

A separate file: the main patch (`WRITER_INPUTS_v4_patch.md`) and WP-36 are unchanged. Setting: the weight-decayed
linear-plus-slab task of WP-36 (λ = 1e−4), with the attained, reproducible fixed-scale switch at s = 3.5914. The
registered prediction is that training's slab share first reaches q = 0.3914 just above the switch. It uses no κ.
Everything was frozen before any training. The text below is the Track T writer input.

### Frozen before training

- **s_switch = 3.5914**, the committed v2 s_q (v2 `frozen.json`, deb8853e…).
  - The attained, weight-decayed fixed-scale path jumps across it: ρ₂ 0.182 → 0.448.
- λ = 1e−4, q = 0.3914, the same data and the same ρ₂ measure.
- The prediction: training crosses q just above the switch.
- Seeds 2,720,000–2,720,039.
- Initialisation: PyTorch default, with the output scale capped at 1.7957. No seed needed the cap; the largest raw
  scale was 1.545.
- Adam lr 0.01 with the same decay. Stop at s ≥ 10.77 or 40,000 steps.
- Every-step ρ₂. Crossing = first upward passage of q.
- The χ definition, C1, C2 and the validity rule.
- A note, before training, that the committed H has smallest eigenvalue λ = 1e−4 (an idle unit), so χ could be large.

### Results (40 runs)

**Run length and crossings**
- Every run reached s ≥ 10.77, in a median of 595 steps (412 to 1,083).
- 31 of 40 runs start with ρ₂(0) ≥ q.
- **22 of 40 runs have an upward passage.** Validity requires 30.
- Of these 22:
  - **6 are early transients, at s = 0.20 to 0.95**, below the switch. All 6 start with ρ₂(0) < q. Their χ is negative,
    because s is still falling.
  - **16 cross late, at s = 8.9 to 10.7** (median 10.32 = 2.87 × switch).
- Fraction of crossing runs at or above the switch: **0.727**. C1 requires 0.90, so the would-be verdict is FAIL.
- Median s_cross/switch: **2.86**. C2 requires [1.00, 1.25], so the would-be verdict is FAIL.

**χ**
- **Median χ at crossing: 7.43.** Validity requires ≤ 0.06.
- Quartiles −108, 7.4 and 10.1. Among the late crossings the median is 9.2.
- Training is far from quasi-static: s grows about ten-fold in about 600 Adam steps.

**Comparison with the path.** At the end of each run (s ≈ 10.8), training's ρ₂ has median 0.39 (quartiles 0.34 and
0.41). The fixed-scale minimiser at s ≈ 9 to 11 has ρ₂ ≈ 0.74. So training lags the path's slab usage far behind the
switch.

### Say / Do not say

**Outcome that occurred: UNRESOLVED, by crossings (22 < 30) and by χ (median 7.4 > 0.06).**
- **Say:**
  - "In a registered test at the attained fixed-scale switch (s = 3.59, weight decay 10⁻⁴), Adam training was far
    outside the slow regime: the median χ at crossing was 7.4, more than 100 times the validity bound of 0.06. Only
    22 of 40 runs crossed the slab-usage level from below."
  - "The test is therefore unresolved. Descriptively, most crossings occurred at about 2.9 times the switch scale, and
    6 were early transients below it."
  - "Training's slab share at s ≈ 10.8 (median 0.39) was well below the fixed-scale minimiser's (≈ 0.74)."
  - "The transfer to the simplicity-bias benchmark is not established. The fixed-scale landscape shows the predicted
    linear-to-slab switch, but at this learning rate training does not follow it closely enough for the test to
    apply."
- **Do not say:**
  - that the account predicts when networks abandon the simple feature on this benchmark;
  - that the test failed or passed (the would-be FAILs are descriptive only);
  - that the lag law was tested here (no κ was used);
  - that a PASS might have occurred at a slower rate; this was not run.

**Other outcomes (none occurred):**
- **PASS/PASS.**
  - *Would have said:* training acquires the slab feature just above the attained switch, in at least 90% of crossing
    runs and with median ratio in [1.00, 1.25], in the slow regime.
  - *Do not say:* that this is a κ-based lag prediction.
- **Any FAIL (valid).**
  - *Would have said:* in the slow regime, training's slab acquisition did not occur just above the switch, and would
    have reported which criterion failed.
  - *Do not say:* that the landscape switch is wrong. The landscape result stands separately.
- **UNRESOLVED by crossings only, or by χ only.**
  - *Would have said:* the corresponding single condition.
  - *Do not say:* a C1 or C2 verdict.
- **Not scored by 04:15.** Did not occur: the test was scored at 03:03 EDT.

## Paper sentence (main.tex line 185), checked against the Say / Do not say lines above

The author proposed a sentence, and it was corrected against WP-36 and WP-37 in three places:
1. Without weight decay the minimizer is not attained, so the sentence cannot speak of "the minimizer" there.
2. With weight decay the minimizer becomes slab-dominant through one discontinuous switch after a mixed branch, not a
   direct linear-to-slab switch.
3. The Say line "the transfer is not established" is added, with both validity thresholds.

Use as written:

> In an exploratory extension to a linear-plus-slab benchmark \citep{shah2020pitfalls} with a width-four tanh network, the
> fixed-scale minimizer was not attained without weight decay (its hidden weights diverged and the slab feature entered
> gradually); with weight decay it was attained and became slab-dominant through a single discontinuous switch at a
> reproducible output scale. A registered training test of that switch was unresolved on its validity conditions: only
> 22 of 40 runs crossed (30 were required), and the growth-to-relaxation ratio at crossing (median 7.4) lay far outside
> the validity bound of 0.06, so the transfer to this benchmark is not established.

## Census rows for this registration (folded into the main census on 2026-09-26, at the author's request; WP-14)

| id | block | registration | verdict | scored |
|---|---|---|---|---|
| trackT-C1 | simplicity-bias transfer (Track T v3) | results/simplicity_bias_v3_registration.md (91ef9cf) | UNRESOLVED | fraction of crossing runs at s ≥ 3.5914: 0.727 (would be FAIL); validity failed: 22 crossings < 30 and median χ 7.43 > 0.06 |
| trackT-C2 | simplicity-bias transfer (Track T v3) | results/simplicity_bias_v3_registration.md (91ef9cf) | UNRESOLVED | median crossing/switch 2.86 (would be FAIL); same validity failure |

With these two rows the census headline is 234 predictions: 212 scored by a registered rule (100 / 66 / 8 / 38) and 22
assigned post hoc (WP-14).
