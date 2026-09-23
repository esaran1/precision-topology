# Registration: prospective test of the own-seed predictor U_own

**Registered 2026-09-23 16:38 EDT, before any computation on the new settings.**
- No certified Ĝ, R_glob or own-seed threshold has been computed for them, and no run trained.
- This file supersedes the design sent for review (commit `6bf3298`), with the four approved changes.
- Producer: `src/prospective_own.py`.
- Before the first training step, the predictions file is committed with its SHA-256.

## Question

S3 found that each run's own conditional threshold sits about 6% above the population threshold, and
that free-training crossings track it (ρ ≈ 0.88). This test asks, on settings nobody has trained,
whether **a run's own threshold, computed from its own training set before training, predicts where
that run becomes correct**. It covers both a = 1.30 and a = 1.50, with results reported per a.

## Settings (structural rule, no training)

- **Candidate family**: the 88 symmetric windows with K > 0 in `prospective_candidates.csv`.
- **Exclusions**: every window ever trained is excluded.
  - The base window and Block G's G1–G3 were never in the family. The family excluded them by
    construction.
  - G4 is asymmetric and cannot be in it.
  - Block 3's H10, H35, H65 and H90 are removed.
  - That leaves 84 eligible windows. `select()` asserts that no selected window equals a trained one.
- **Selection**: the windows nearest the κ₀ quantiles 0.1, 0.2, …, 0.8 of the 84 (certified κ₀,
  distinct picks; `prospective_own_windows.csv`):

  | name | I | O | κ₀ (certified) |
  |---|---|---|---|
  | V10 | ±0.7 | ±[0.9, 1.5] | [0.19030, 0.19031] |
  | V20 | ±0.8 | ±[1.1, 1.9] | [0.24159, 0.24161] |
  | V30 | ±0.9 | ±[1.3, 2.1] | [0.27897, 0.27899] |
  | V40 | ±0.9 | ±[1.4, 2.0] | [0.33443, 0.33445] |
  | V50 | ±0.5 | ±[0.9, 2.1] | [0.35854, 0.35857] |
  | V60 | ±0.6 | ±[1.1, 2.3] | [0.39845, 0.39847] |
  | V70 | ±0.5 | ±[0.9, 1.5] | [0.43300, 0.43304] |
  | V80 | ±0.9 | ±[1.7, 2.3] | [0.47250, 0.47254] |

- **Settings**: 8 windows × a ∈ {1.30, 1.50} = **16**. The window is the unit of generalisation.
- **Protocol**: Block 3's (`blockG_windows.train`).
  - `Window.data(200, seed)`, float64, Adam lr 1e−2, torch.manual_seed(seed) then U(−1, 1)⁴.
  - **60 seeds per setting**: 400,000–400,059.
  - Crossing is checked as in Block 3, and R_cross = |w₂|·Ĝ_cert(window, a)/2.

## Budget and non-crossing runs (registered before training)

- **Budget 12,000 steps.** Judged from Block 3's crossing distributions under this protocol: 95.6–97.8%
  of runs crossed by 12,000 in every setting, and the 95th percentile of the crossing step was
  6,235–8,598.
- A longer budget would add few crossings. On the base window, phase 2b crossed 37 of 40 runs by 12,000
  and 38 of 40 by 32,000. It would also depart from the protocol under which Block 3's λ was fitted.
- **Reported**: the number of non-crossers in every setting.
- **Primary analysis**: per-run error on crossers only.
- **Registered sensitivity analysis**: each non-crosser's crossing R is at least its final R. It is
  included with R_cross set to its final R (the bound), and every comparison is recomputed.

## Predictors (all computed and frozen before any training in the new settings)

- **U_own (primary, nothing fitted)**:
  - R_own is each run's own R_glob, from its own training set.
  - It uses the validated own-seed search (`own_threshold.global_min`: grid 0.02, 8 refinements, the
    window's own w-bound and gap).
  - It is bracketed from the window's population |w₂| in steps of 0.1, bisected to 0.01, and converted
    with the window's certified Ĝ.
- **C_own (secondary, fitted, labelled)**: ρ_res(a)·R_own, with ρ_res frozen from existing base-window
  data. That is exp(median log(R_cross/R_own)) over phase 2b seeds 0–39 crossing within 12,000 steps:
  **ρ_res(1.30) = 1.0311, ρ_res(1.50) = 1.0660**.
- **Comparators (Block 3's models)**:
  - **U** = the certified population R_glob (Block 3's bracket procedure on the window's quadrature
    population).
  - **C** = λ(a)·U with Block 3's frozen λ(1.30) = 1.11487 and λ(1.50) = 1.16440.
- **Validation before training**: the certified branch and bound at both bracket ends for seeds
  400,000–400,002 in every setting. **Stop if any certified status contradicts a bracket end.**
  Unresolved ends are reported.

## Targets

1. **Per run (primary)**: |log(R_cross/pred)|.
2. **Per setting (secondary)**: |log(median R_cross / median pred)|. Reported, with no criterion.

## Registered comparisons (scored separately at each a; each a has 8 windows)

- **Setting-level uncertainty**: the window-level bootstrap of the mean over the a's 8 windows of a
  per-window statistic (10,000 resamples of windows with replacement, seed 0).

| | statistic (per window, then mean over windows) | pass iff |
|---|---|---|
| **P1** (primary) | mean per-run \|err\| of U_own − that of U | 95% interval entirely below 0 |
| **P2a** (a = 1.30), non-inferiority | median per-run \|err\| of U_own − that of C | 95% interval upper end < **0.01** |
| **P2b** (a = 1.50) | mean per-run \|err\| of U_own − that of C | interval **not** entirely below 0 (U_own does not beat C). Whether it lies entirely above 0 (C better) is also reported. |
| **P3** (fitted) | mean per-run \|err\| of C_own − that of C | interval entirely below 0 |
| **P4** | **the median, over all crossing runs at that a (pooled over its 8 settings), of the per-run absolute log error of U_own** | it lies in **[0.001, 0.061] at a = 1.30** and **[0.034, 0.094] at a = 1.50** |

- **P3 caveat**: ρ_res was fitted on the base window, so the base-window comparison behind P3 (0.012
  against 0.057 at 1.30; 0.015 against 0.057 at 1.50) is **in-sample** and optimistic.
- **P4's range**: the base-window median ± 0.03, where the base median is 0.031 at 1.30 and 0.064 at
  1.50. The 0.03 allows for window-to-window lag variation of the size Block 3 saw (C's per-setting
  signed errors ran from +0.001 to +0.037).
- **Failure criteria**:
  - Each of P1, P2a, P3 and P4 fails at its a if its pass condition is not met.
  - P2b fails if U_own beats C at 1.50 with the interval entirely below 0. That would be an unexpected
    success, recorded as a failure of the registered expectation.
  - Results are reported per a. Pooled numbers are given as well, but they carry no verdict.
- **Also reported**:
  - Spearman ρ between R_cross and R_own in each setting, against S3's ≈ 0.88;
  - per-setting median errors for all four predictors.

## Power (simulated from existing data; `prospective_own_power.csv`)

**Model**: 1,000 simulated experiments of 8 windows × 60 runs per a, each scored exactly as above
(2,000 window resamples).
- Per-run signed errors (U_own, U) are resampled jointly from the base window's 37 runs crossing within
  12,000 steps.
- A window effect is drawn N(0, sd_a), with sd_a the spread of Block 3's per-setting log(obs/U) at that
  a (0.0056 at 1.30, 0.0162 at 1.50). C's and C_own's errors follow from λ and ρ_res.

| scenario | a | P1 | P2a / P2b | P3 | P4 |
|---|---|---:|---:|---:|---:|
| as observed (common window shift) | 1.30 | 1.00 | 1.00 | 1.00 | 1.00 |
| as observed (common window shift) | 1.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| stress: window sd × 3, independent shifts for U_own and U | 1.30 | 1.00 | 1.00 | 1.00 | 1.00 |
| stress: window sd × 3, independent shifts for U_own and U | 1.50 | **0.71** | 0.98 | **0.61** | 0.88 |

- For P2b the figure is the probability that the expectation holds.
- **Reading**:
  - Under the variation actually seen, every comparison is near-certain to resolve.
  - At a = 1.50, P1 and P3 lose power if window effects are three times larger than Block 3's and act
    separately on the two predictors.
  - The estimates rest on one base window's per-run errors. That is the main limitation of this power
    check.

## Cost (estimate)

- **Own-seed search**: about 65 s per run × 960 runs ≈ 17 CPU-hours (about 6 h at 3 workers).
- **Certified validation**: 48 runs × 2 ends.
- **Training**: 960 runs × ≤ 15 s.

## Order

It runs after the currently queued items, using free slots as they open. The sequence is `predict`
(hash committed), `validate`, `train`, then `score`.
