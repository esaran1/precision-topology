# WP-38. The validity boundary (registered) and the Adam ordering diagnostic (post hoc) (final night; for the submission)

This is a separate file. The main patch is unchanged, except that `collapse.pdf` and its captions.md entry were
regenerated to include the boundary test's points, at the author's request.

## 1. Registered boundary test: width-1 SGD forced ramps at χ = 0.03–0.4

- **Producer and registration:** `src/ramp_boundary.py` → `results/ramp_boundary/`; registration
  `results/ramp_boundary_registration.md` (a22e1aa).
- **Frozen before any run:** each seed's forced-branch switch and χ_own (SHA-256 in `frozen.sha256`).
- **Design:** SGD at χ = 0.03, 0.06, 0.1, 0.2 and 0.4 (γ = χ·η·λ_min(H), set a priori), at a = 1.30 and 1.50, 40 fresh
  seeds per cell.

**Registered verdicts: B1 UNRESOLVED, B2 UNRESOLVED.** Only 3 of 10 cells met validity (at least 30 crossings, and χ_own
within 30% of target). In every cell χ_own was within 3% of target; the invalid cells lacked crossings.
- B1 depends on the χ = 0.06 cell at a = 1.30, which had 28 crossings.
- B1 could not have passed in any case: its valid cell χ = 0.03 at a = 1.30 lies outside the band, at 1.67.
- B2 depends on the χ = 0.4 cells, which had 15 and 6 crossings.
- At χ ≥ 0.1, most runs did not cross within the ramp (end scale s\*·max(3, 1 + 4κχ)). Their lag exceeded about four
  times the prediction, or they never placed.

| a | χ target | crossings / 40 | median obs/pred | in [0.75, 1.25] | valid |
|---|---|---|---|---|---|
| 1.30 | 0.03 | 40 | 1.67 | no | yes |
| 1.30 | 0.06 | 28 | 3.89 | no | no |
| 1.30 | 0.1 | 27 | 2.95 | no | no |
| 1.30 | 0.2 | 20 | −2.03 | no | no |
| 1.30 | 0.4 | 15 | 1.54 | no | no |
| 1.50 | 0.03 | 40 | 1.22 | yes | yes |
| 1.50 | 0.06 | 40 | 1.67 | no | yes |
| 1.50 | 0.1 | 3 | 2.59 | no | no |
| 1.50 | 0.2 | 0 | — | — | no |
| 1.50 | 0.4 | 6 | 3.25 | no | no |

- **χ\* (descriptive):** 0.03 at a = 1.30 and 0.06 at a = 1.50.
- **Descriptive (post hoc, not registered):** in the three valid cells the ratio follows the predicted lag κχ, not χ.
  It is 1.22 at κχ ≈ 0.12 and 1.67 at κχ ≈ 0.23–0.24.
  - The earlier ramp cells (WP-24, κχ ≤ 0.1) were 0.87–1.26.
  - So the linear law's error grows with the predicted lag itself. Beyond κχ ≈ 0.1–0.2 it under-predicts, and beyond
    κχ ≈ 0.4 the forced ramp outruns the branch.

**Replacement sentence for the paper's boundary statement:**
> "In a registered test with forced SGD ramps at five preset rates (χ = 0.03–0.4, a = 1.30 and 1.50), the verdicts were
> unresolved because most fast-ramp runs outran the tracked branch; descriptively, the lag stayed within 25% of κ(a)χ only
> where the predicted lag κχ was about 0.1 and exceeded it by a factor of 1.7 at κχ ≈ 0.23, so the account's range of
> validity is set by the predicted lag, κχ ≲ 0.1, rather than by χ alone."

**Say:**
- the sentence above;
- that the boundary verdicts are unresolved;
- that κχ ≲ 0.1 is a descriptive reading.

**Do not say:**
- "the boundary test passed or failed";
- "χ ≲ 0.06 is the boundary" without qualification. At a = 1.30, χ = 0.03 is already outside ±25%, because κ(1.30) is
  large.

## 2. Adam ordering diagnostic (POST HOC; the registered L3 FAIL stands)

- **Producer:** `src/track_a_diag.py` → `results/track_a/diag_p_at_crossing.{csv,json}` (7896d6b).
- **Runs:** the existing a = 1.65 Adam runs, 74 with a prediction.
- **Method:** the same pipeline with the preconditioner frozen at a different point. The registered pipeline is
  reproduced exactly.

| P frozen at | per-run Spearman (trajectory) | within 10% | median obs/pred | obs/pred q10–q90 | predicted lag q10–q90 |
|---|---|---|---|---|---|
| rule point, last passage of 0.5·s\* (registered) | 0.25 | 22% | 1.07 | 0.74–1.75 | 0.054–0.129 |
| the first step |w₂| reaches the occupied branch's switch (candidate rule, chosen after seeing these runs; before crossing in 74/74) | 0.95 | 91% | 1.04 | 1.02–1.10 | 0.074–0.102 |
| the observed crossing (uses crossing-time information) | 0.99 | 100% | 1.06 | 1.02–1.07 | 0.074–0.102 |

The observed lags span 0.079–0.106 (q10–q90).

**Outcome: a measurement-point artifact.**
- Freezing P at the run's own branch switch uses only pre-crossing information and recovers the per-run ordering.
- The registered rule point, at half the switch, is too early: Adam's preconditioner is still changing there.

**Sentence for the paper:**
> "Adam's failed per-run ranking at a = 1.65 is a measurement-point artifact: post hoc, freezing the preconditioner at
> the first step at which the output scale reaches the occupied branch's switch, which is still before the crossing in
> all 74 runs, gives a per-run rank correlation of 0.95 and 91% of runs within 10% of the prediction; we name this
> rule, chosen after seeing these runs, for a future registered test."

**Do not say:**
- that L3 passed;
- that the corrected rule was registered, validated or prospectively tested. It was chosen after seeing these runs and
  is a candidate for a new registration.
- that P at the crossing is a prediction (it uses crossing-time information).

## 3. Census rows (not folded into the main census; the main patch is unchanged)

| id | block | verdict |
|---|---|---|
| boundary-B1 | boundary test (final night) | UNRESOLVED |
| boundary-B2 | boundary test (final night) | UNRESOLVED |

Folding these in would make the headline 236 predictions: 214 scored by a registered rule (100 / 66 / 8 / 40) and 22
post hoc. That would change WP-14 in the main patch, so it was not done.
