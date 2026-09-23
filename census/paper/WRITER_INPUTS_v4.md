# Writer inputs, v4 (supersedes WRITER_INPUTS.md where they differ)

Written for the paper's writer. Every item names its registration, producing script and artifact. The v3
handoff (`WRITER_INPUTS.md`, tag `paper-submitted-v3`) remains correct except where this file changes it.

## Status

| block | status | for the 25 September submission? |
|---|---|---|
| 1. Audit of the conditional threshold | **complete**; registered decision rule met; 1d (objective choice) running | yes (1d: added when done) |
| 2. Proposition 2 made precise | certified localisation, rounding and bounds done; H2′, neighbourhood and solve checks running | yes, once the checks close |
| 3. Prospective held-out prediction | **complete; primary criterion PASSED** | yes |
| 4. Fixed-scale dynamics | running under an amended validity check (see below) | if scored and verified in time; otherwise rebuttal |
| 5. Retention curve | queued after Block 4 | if scored and verified in time; otherwise rebuttal |
| 6. Width 2 | design committed for review (`block6_width2_design.md`); nothing run | rebuttal |

## Terminology (all paper text)

Replace "training-free" with **"computed independently of unconstrained training trajectories"**. The
conditional search is itself an optimisation. Done in every paper-facing file of the repository; four
frozen registration documents keep their original wording.

## Block 1 — the conditional threshold is not an artifact of the search

Registered `conditional_audit_prediction.md` (`6b17508`, before any audit computation). Results
`conditional_audit.md`; producers `src/conditional_audit.py`, `src/profiled_bnb.py`,
`src/conditional_certified.py`.

- **Decision rule met; no stop condition fired.**
- **Every candidate retained (1a).** The frozen search was replayed at all 358 scales it evaluates:
  16,468 candidates, and the frozen result reproduced at 358 of 358.
  - No discarded candidate — degenerate, screened out, not lowest, or the exact constant predictor —
    has lower loss than the retained branch, anywhere.
  - Retained minimisers are stationary: gradient norm at most 2.3e-6.
  - The candidate-loss plots (`results/figures/cond_audit_candidates.pdf`) show one continuous branch
    at every a.
- **Stricter optimisation (1e)**: 10× steps and a gradient-norm stop at 1e-8. It agrees with the
  frozen search at all 12 thresholds, and losses agree to 2e-16.
- **Certified thresholds (1c).**
  - **Method**: exhaustive branch and bound on the profiled conditional loss. The output bias is
    solved exactly (a strictly convex scalar problem). The domain is the full placement domain: b₁ is
    2π-periodic, and |w₁| is bounded analytically.
  - **Every R_glob sign decision is rigorous.** The two certified intervals are separated by at least
    1.3e-8. That is 100× the stated rounding margin, with attained values re-checked in interval
    arithmetic to 1.1e-16.
  - **The switch is a continuous crossing of one basin**, not an exchange: the minimisers on either
    side of G = 0 lie 0.00015–0.0011 apart.

**Paper-facing thresholds.** Quote the certified intervals. The frozen values are the first grid point
at or above each certified switch, so they overstate it by less than one grid step (0.05 in |w₂|).

| a | Ĝ (certified) | **R_glob, certified** | frozen R_glob | **R_solve, certified bracket** | frozen R_solve | \|w₂\| intervals (glob / solve) |
|---|---:|---|---:|---|---:|---|
| 1.30 | 0.086102 | (0.21310, 0.21364] | 0.21525 | (0.30566, 0.30620] | 0.30781 | (4.9500, 4.9625] / (7.1000, 7.1125] |
| 1.35 | 0.106914 | (0.21516, 0.21583] | 0.21650 | (0.30537, 0.30604] | 0.30738 | (4.0250, 4.0375] / (5.7125, 5.7250] |
| 1.40 | 0.128772 | (0.21730, 0.21811] | 0.21891 | (0.30503, 0.30583] | 0.30583 | (3.3750, 3.3875] / (4.7375, 4.7500] |
| 1.45 | 0.151545 | (0.21879, 0.21974] | 0.21974 | (0.30404, 0.30498] | 0.30688 | (2.8875, 2.9000] / (4.0125, 4.0250] |
| 1.50 | 0.175126 | (0.22110, 0.22219] | 0.22329 | (0.30428, 0.30538] | 0.30647 | (2.5250, 2.5375] / (3.4750, 3.4875] |
| 1.60 | 0.224360 | (0.22436, 0.22576] | 0.22997 | (0.30289, 0.30429] | 0.30850 | (2.0000, 2.0125] / (2.7000, 2.7125] |

(R_solve is the global minimiser's solve margin changing sign, evaluated at the certified minimiser.
Its certification over the minimiser's enclosure is in Block 2 below.)

## Block 3 — prospective prediction on held-out geometries: primary criterion PASSED

Registered `prospective_prediction.md` (`8d8dec6`, before any registered training; predictions frozen
in `prospective_predictions.csv`, SHA-256 in the registration). Results `prospective_scores.csv`,
`prospective_comparisons.csv`; producer `src/prospective.py`.

**Design.**
- **Held-out windows**: four windows chosen by a structural rule with no training — the certified
  limiting-cubic κ₀ at quantiles 0.1, 0.35, 0.65 and 0.9 of an 88-window family that excludes every
  window used before. Crossed with a = 1.30 and 1.50, that gives 8 settings, each with 86–88 crossings
  out of 90 runs.
- **Change from the plan, registered with its reason**: a = 1.30 and 1.50 instead of 1.35 and 1.45.
  Only those values have base runs under the identical protocol, which the lag calibration needs.
- **Models**:
  - **U**: the certified conditional threshold R_glob of the new window.
  - **C** = λ(a)·R_glob. **λ is a lag factor fitted on the base window at the same a** (1.115 at 1.30,
    1.164 at 1.50) and must be labelled fitted.
  - **B1**: the base window's crossing |w₂|, converted to R with the new window's Ĝ (output weight
    without geometry).
  - **B2**: one pooled crossing R over all existing settings (no geometry-specific minimisation).
- **Disclosed**: one timing run (seed 999999) was made in H35 before registration. It is excluded
  from scoring, and no prediction depends on it.

**Result.** Metric: |log(pred/obs)| of the per-setting median crossing R; setting-level paired bootstrap.

| comparison | mean difference | 95% interval | |
|---|---:|---|---|
| **C − B1** (registered) | −0.257 | [−0.370, −0.131] | **C better by a wide margin** |
| **C − B2** (registered) | −0.055 | [−0.103, −0.010] | **C better by a modest margin** |
| C − U | −0.086 | [−0.111, −0.063] | C better |
| secondary target, C − B3 (fraction placed by budget) | −0.044 | [−0.068, −0.018] | C better |

**Every setting** (absolute log errors; "best" is the lowest):

| setting | crossed | observed median crossing R | U | C | B1 | B2 | best | C signed log error |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| H10, a = 1.30 | 88 | 0.2119 | 0.077 | 0.032 | 0.358 | 0.073 | C | +0.0320 |
| H10, a = 1.50 | 88 | 0.2259 | 0.116 | 0.037 | 0.339 | 0.009 | B2 | +0.0366 |
| H35, a = 1.30 | 86 | 0.2296 | 0.080 | 0.028 | 0.013 | 0.007 | B2 | +0.0284 |
| H35, a = 1.50 | 87 | 0.2476 | 0.124 | 0.028 | 0.019 | 0.082 | B1 | +0.0284 |
| H65, a = 1.30 | 87 | 0.2415 | 0.088 | 0.021 | 0.291 | 0.057 | C | +0.0212 |
| H65, a = 1.50 | 87 | 0.2679 | 0.142 | 0.010 | 0.267 | 0.161 | C | +0.0104 |
| H90, a = 1.30 | 88 | 0.2417 | 0.088 | 0.021 | 0.492 | 0.058 | C | +0.0205 |
| H90, a = 1.50 | 88 | 0.2704 | 0.151 | 0.001 | 0.458 | 0.170 | C | +0.0013 |

- **C's error is at most 3.7% in every setting.**
- **Systematic limitation of the calibration: C over-predicts in all 8 settings**, with a mean signed
  log error of **+0.022** (setting-level 95% interval [0.015, 0.030]). The reason is that the lag factor
  fitted on the base window exceeds the held-out windows' lag by about 2%. It was **not recalibrated**
  on the held-out data. State it beside the 3.7% maximum error.
- **C loses in four pairwise comparisons**:
  - to B2 at H10, a = 1.50 (0.037 against 0.009);
  - to B2 at H35, a = 1.30 (0.028 against 0.007);
  - to B1 at H35, a = 1.30 (0.028 against 0.013);
  - to B1 at H35, a = 1.50 (0.028 against 0.019).

  These are the windows whose R_glob sits nearest the pooled value, or whose Ĝ is nearest the base
  window's.
- **U under-predicts in all 8 settings**, by log(obs/U) = 0.077–0.151 (U is 7.4–14.0% below the
  observed value).
  - U's registered range [0.080, 0.182] holds at 7 of 8 settings. H10 at a = 1.30 is outside at
    0.0767.
  - Under the registered per-setting rule this is 7 of 8. Read as a joint prediction, it **fails
    narrowly**. The primary result is unaffected.

**Secondary analyses** (labelled secondary; the registered analysis above is primary):
- **Window-clustered bootstrap**: the 8 settings are 4 windows × 2 a. Enumerating all 256 window
  resamples gives C − B1 in [−0.414, −0.069] and C − B2 in [−0.098, −0.012]. Both still exclude zero.
- **Leave one window out**: C beats both baselines with each of the four windows removed. The C − B2
  mean ranges from −0.039 (without H90) to −0.071 (without H10).
- **Residual bias** (the systematic limitation above): the window-level 95% interval for C's mean
  signed log error is [0.013, 0.031].
- **Threshold definition is consistent**: λ and the held-out predictions both use the certified
  R_glob bracket midpoint from the same procedure, so there is no definition bias. Bracket
  resolution adds at most ±0.6%.

**Suggested wording.** "On four held-out window geometries chosen by a structural rule before any
training, the conditional threshold, scaled by a lag factor fitted on the base window at the same a,
predicted the median crossing within 3.7% in all eight settings, with a systematic over-prediction
of 2.2% on average (95% interval 1.5–3.0%), because the base-window lag exceeds the held-out
windows'. It outperformed an output-weight baseline (mean absolute log error 0.257 lower, 95% interval [0.131, 0.370]) and a common-R baseline
(0.055 lower, [0.010, 0.103])."

## Block 2 — Proposition 2 made precise

MATH_NOTE_SECTION

## Blocks 4 and 5 — fixed-scale dynamics and retention

**A registered validity check fired, and was amended before any outcome was computed.**
- **What fired**: the registration (`fixed_scale_prediction.md`, `888344d`) required the k = 1 replay
  to reproduce an independent frozen-w₂ continuation to 1e-10. On launch the check gave 0.105.
- **Why the check was invalid**: its reference zeroed w₂'s gradient under the full Adam optimiser.
  That does not freeze w₂ — Adam's stored momentum kept moving it by up to 0.033 in 25 steps.
- **The replay itself was correct**: a reference that truly freezes w₂ (reset after every step)
  reproduces it exactly.
- **The amendment** (`10c1ea5`) replaced only that reference. No outcome prediction, horizon, grid,
  sample size or criterion changed.
- **Tests before any replay**: 11 tests covering the reference, the replay, and every stop condition
  in a should-fire and a should-not-fire case (`tests/test_fixed_scale.py`). All pass.
- **Record**: this was the second error in that registration's stop conditions; both concerned the
  implementation check, not an outcome.
- **E-2 stays failed** whatever these blocks show.

BLOCKS45_SECTION
