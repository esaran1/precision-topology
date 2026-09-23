# Design for review: prospective test of the own-seed predictor (U_own)

**Design only. Nothing here has been computed on the new settings, and nothing runs until it is approved.**
On approval, this file becomes the registration. It is committed, with its predictions file and SHA-256,
before any training in the new settings.

## Question

S3 found three things:
- each run's own conditional threshold sits about 6% above the population threshold;
- free-training crossings track it (Spearman ρ ≈ 0.88);
- measured against it, the offset shrinks to 3.1% (a = 1.30) and 6.4% (a = 1.50).

This design tests, prospectively on settings nobody has trained, whether **a run's own threshold,
computed from its own training set before training, predicts where that run becomes correct**. It covers
both a. It does not restrict to settings that resemble a = 1.30, because that would select on outcome.

## Settings (structural rule, no training)

- **Candidate family**: the 88 symmetric windows with K > 0 of `prospective_candidates.csv` (Block 3's
  family). Block 3's four windows are removed, leaving 84. Block G's windows were never in the family.
- **Selection**: the windows nearest the κ₀ quantiles 0.2, 0.4, 0.6 and 0.8 of the 84 (certified κ₀,
  distinct picks). Computed from the certified κ₀ table only:

  | name | I | O | κ₀ (certified) |
  |---|---|---|---|
  | W20 | ±0.8 | ±[1.1, 1.9] | [0.24159, 0.24161] |
  | W40 | ±0.9 | ±[1.4, 2.0] | [0.33443, 0.33445] |
  | W60 | ±0.6 | ±[1.1, 2.3] | [0.39845, 0.39847] |
  | W80 | ±0.9 | ±[1.7, 2.3] | [0.47250, 0.47254] |

- **Settings**: 4 windows × a ∈ {1.30, 1.50} = 8. None overlaps any setting already trained (base, G1–G4,
  H10–H90).
- **Protocol**: Block 3's, so that its U and C apply unchanged.
  - Training sets `Window.data(200, seed)` (400 points), float64, Adam lr 1e−2, torch.manual_seed(seed)
    then U(−1, 1)⁴, 12,000 steps.
  - Crossing is the first step at which the dense-grid oriented gap is > 0. R_cross = |w₂|·Ĝ_cert(window,
    a)/2.
- **Seeds**: 90 per setting, seeds 400,000–400,089 (fresh).
- **Crossings**: at least 80 per setting are required; 20 seeds at a time are added otherwise, up to 150.

## Predictors (all computed before any training in the new settings)

- **U_own (primary, nothing fitted)**: for each run, R_own = that run's own R_glob, computed from its own
  training set by the validated own-seed search (`own_threshold.global_min`, grid 0.02, 8 refinements,
  bisection to 0.01 in |w₂|; the S1–S3 method) and converted with the window's certified Ĝ. The
  prediction for the run's crossing R is R_own.
- **C_own (secondary, fitted, labelled)**: ρ_res(a)·R_own. The residual factor is frozen from existing
  base-window data: exp(median log(R_cross/R_own)) over the phase 2b seeds 0–39 that crossed within
  12,000 steps.
  - **ρ_res(1.30) = 1.0311, ρ_res(1.50) = 1.0660.**
- **Comparators (Block 3's models, computed for the same settings)**:
  - **U** = the certified population R_glob of the window at a (quadrature objective, bracket midpoint).
  - **C** = λ(a)·U with Block 3's frozen λ(1.30) = 1.11487 and λ(1.50) = 1.16440.
  - U and C are the same for every run in a setting.
- **Frozen before training**: all per-run R_own, the per-setting U and C, and ρ_res are written to
  `prospective_own_predictions.csv`. It is committed with its SHA-256 before the first training step.

## Targets

1. **Per run (primary)**: |log(R_cross/pred)| for each run that crosses, for each predictor.
2. **Per setting (secondary)**: |log(median R_cross / median pred)|. For U_own the setting prediction is
   the median R_own over the setting's runs, and for C_own it is ρ_res times that.

## Expected per-run error of U_own (registered from existing base-window data, crossings ≤ 12,000)

| a | median per-run \|log(R_cross/R_own)\| | bootstrap 95% | 10th–90th percentile | **registered range for the new settings** |
|---|---:|---|---|---|
| 1.30 | 0.031 | [0.030, 0.032] | [0.025, 0.039] | **[0.001, 0.061]** |
| 1.50 | 0.064 | [0.062, 0.066] | [0.050, 0.077] | **[0.034, 0.094]** |

The registered range is the base median ± 0.03. The 0.03 allows for window-to-window variation in the lag:
Block 3's C, calibrated on the base window, had per-setting signed errors from +0.001 to +0.037 on
held-out windows.

## Registered comparisons (per a; each a scored separately; pooled reported as well)

Uncertainty is at the setting level. There are 4 windows per a, so the window-clustered bootstrap
enumerates all 4⁴ = 256 resamples. Each statistic is the mean over the a's settings of the per-setting
mean per-run |log error| difference.

- **P1 (primary)**: U_own beats U at each a. The mean difference (U_own − U) is < 0, and its 95%
  window-level interval lies below 0.
  - Base-window expectation: 0.039 against 0.095 (1.30), 0.071 against 0.120 (1.50).
- **P2**: U_own against C. The expectations from existing data are **split by a**, and registered as
  such:
  - **P2a (a = 1.30)**: U_own beats C, with the interval below 0 (base: 0.039 against 0.057).
  - **P2b (a = 1.50)**: U_own does **not** beat C (base: 0.071 against 0.057). Registered outcome: the
    interval for U_own − C is not entirely below 0.
  - A pass of P2b means the expected ordering held. It is not a success for U_own.
- **P3 (secondary, fitted)**: C_own beats C at each a, with the interval below 0.
  - Base, in sample: 0.012 against 0.057 (1.30), 0.015 against 0.057 (1.50). That is optimistic, because
    ρ_res was fitted there.
- **P4 (range)**: at each a, the median per-run |log error| of U_own, pooled over that a's 4 settings,
  lies in the registered range above.
- **Per-setting target (secondary)**: U_own and C_own against U and C on the per-setting median, with
  the same window-level bootstrap. Reported, no pass criterion.
- **Tracking**: Spearman ρ between R_cross and R_own within each setting. Reported against S3's
  ρ ≈ 0.88, no criterion.

## Failure criteria

- **P1 fails** at an a if the interval for U_own − U is not entirely below 0. Then the own-seed threshold
  does not improve on the population threshold as a per-run predictor on new windows at that a.
- **P2a fails** if the interval for U_own − C at 1.30 is not entirely below 0. **P2b fails** if U_own beats
  C at 1.50 with the interval below 0. That would be an unexpected success, reported as a failure of the
  registered expectation.
- **P4 fails** at an a if the median per-run error falls outside its range.
- **Stop and report**:
  - any validation failure of the own-seed search on the new windows (below);
  - fewer than 80 crossings in a setting after 150 seeds;
  - any change to the predictions file after its hash is committed.

## Validation of the own-seed search on the new windows (before training, reported)

- Certified branch and bound (`conditional_certified.evaluate`) at both bracket ends for 3 seeds per
  setting (seeds 400,000–400,002).
- Stop if any certified status **contradicts** a bracket end. Unresolved ends are reported, as in S1–S3.

## Cost (estimate)

- **Own-seed search**: about 65 s per run, 720 runs, about 13 CPU-hours (about 4.5 h at 3 workers).
- **Training**: about 720 runs × ≤ 15 s.
- **Certified validation**: 24 runs × 2 ends.
- To halve the search cost, the seeds per setting can drop to 60 (requirement ≥ 50 crossings), at the
  price of wider per-setting medians.

## Order

It runs after the queue already agreed (S1/S2 scoring, the c₁ test, the diagnosis, the solve checks, the
size test).
