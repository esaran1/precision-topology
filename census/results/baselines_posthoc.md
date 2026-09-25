**POST HOC** (Track 6, specified 2026-09-25 after Block 3 and the own-seed test were scored). No registered verdict changes. Nothing here was registered.

# Stronger baselines for the held-out test (Block 3) and the own-seed test

Producer: `src/baselines_posthoc.py` (`python -m src.baselines_posthoc`). Tests: `tests/test_baselines_posthoc.py`.
Outputs:
- `baselines_posthoc_fits.csv`: fitted parameters.
- `baselines_posthoc_training_fit.csv`: in-sample errors on Block G.
- `baselines_posthoc_block3.csv`: per-setting predictions and errors.
- `baselines_posthoc_comparisons.csv`: every interval.
- `baselines_posthoc_own_errors.csv` and `baselines_posthoc_own_settings.csv`: own-seed errors.
- `baselines_posthoc_validation.csv`: the reproduction checks.

## Training data (the same for every baseline)

Every baseline is fitted only on the ten Block G settings: five windows × a ∈ {1.30, 1.50}.
- **Crossing R**: the crossers in `blockG_crossings.csv` (379 runs), with R = |w₂|·Ĝ/2 and certified Ĝ. This is exactly `prospective.calibrate`.
- **Conditional threshold**: the certified R_glob of these ten settings (`prospective_rglob_all.csv`).
- **Gap size**: the certified Ĝ of these settings (`prospective_ghat.csv`).
- **Limit constant**: the K of each Block G window (`limit_windows.csv`, K_lo), with κ₀ = K/D∞ and D∞ = 4√2/3.

No Block 3 run and no own-seed run enters any fit. `fit_baselines` raises an error if the training table contains any window outside Block G. A test checks this.

Held-out inputs used at prediction time:
- **Block 3**: U, κ₀ from the registration file `prospective_windows.csv`, and certified Ĝ.
- **Own-seed test**: U, κ₀ from `prospective_own_windows.csv`, and Ĝ.

All of these are training-free and existed before the runs they are scored against.

**Timing caveat.** The Block G K values in `limit_windows.csv` were computed on 2026-09-24, after Block 3 was scored. K depends only on window geometry: it is a branch and bound on the limiting cubic, with no training data and no held-out quantity. The code asserts that its K for H10–H90 equals the pre-registration `prospective_windows.csv` values exactly.

## Baselines (all fixed in code before any was scored)

| model | formula | fitted on | uses the conditional threshold U? |
|---|---|---|---|
| C (registered) | λ(a)·U, λ = base median crossing R / R_glob(base) | base window only, 2 numbers | yes |
| B1 (registered) | base median crossing \|w₂\|(a) · Ĝ(target)/2 | base window only | no |
| B2 (registered) | pooled median crossing R | all Block G crossers | no |
| **PL** | λ_pool(a)·U. λ_pool(a) = median over every Block G crosser at a (5 windows pooled) of R_cross / R_glob(window, a) | 10 settings, 2 numbers | yes |
| **PL5** (variant) | λ₅(a)·U. λ₅(a) = median over the 5 windows of (median crossing R / R_glob) | 10 settings, 2 numbers | yes |
| **RK** | exp(β₀ + β₁ log κ₀ + β₂·1{a = 1.50}), OLS on log median crossing R | 10 settings, 3 numbers | **no** |
| **RKa** (variant) | per a: exp(α_a + γ_a log κ₀), OLS | 5 settings per a, 4 numbers | **no** |
| **RG** | per a: exp(δ_a + η_a log Ĝ(window, a)), OLS | 5 settings per a, 4 numbers | **no** |

Fitted values:
- **PL**: λ_pool = 1.10369 (1.30) and 1.17787 (1.50).
- **PL5**: λ₅ = 1.11024 and 1.18115. For comparison, the registered λ is 1.11487 and 1.16440.
- **RK**: β = (−1.3707, 0.1920, 0.1003).
- **RKa**: (α, γ) = (−1.3989, 0.1694) at 1.30 and (−1.2422, 0.2145) at 1.50.
- **RG**: (δ, η) = (−1.1572, 0.1789) at 1.30 and (−1.0886, 0.2302) at 1.50.

In-sample mean |log error| on the 10 training settings:

| PL | PL5 | RK | RKa | RG |
|---:|---:|---:|---:|---:|
| 0.0109 | 0.0100 | 0.1463 | 0.1463 | 0.1391 |

## Reproduction check (the validation, run before anything new is reported)

163 checks all reproduce to within 1e-12. The largest |difference| is 2.2e-16. They cover:
- λ(a), B1 and B2 in `prospective_calibration.csv`.
- The ten Block G median crossing R and R_glob values.
- Every Block 3 observed median, the U/C/B1/B2 predictions and their |log errors| (`prospective_scores.csv`).
- The committed C − B1, C − B2 and C − U mean and 95% interval (`prospective_comparisons.csv`).
- The own-seed primary P1–P4 statistics and intervals (`prospective_own_scores.csv`).
- The own-seed setting-level |log error| of U and C (`prospective_own_settings_scored.csv`).

## Block 3: registered metric and bootstrap

The metric is |log(pred / median crossing R)| per setting, averaged over the 8 held-out settings. The interval comes from `prospective._boot_diff`: setting-level paired bootstrap, 10,000 resamples, seed 0, 95% percentile interval of mean(err_C − err_X).

| model | mean \|log error\| (8 settings) | at a = 1.30 | at a = 1.50 | C − model: mean [95% interval] | verdict |
|---|---:|---:|---:|---|---|
| C (registered) | **0.0223** | 0.0255 | 0.0192 | — | — |
| **PL** | **0.0231** | 0.0154 | 0.0307 | −0.0007 [−0.0088, +0.0074] | **matches C** (interval contains 0) |
| **PL5** | 0.0274 | 0.0214 | 0.0335 | −0.0051 [−0.0120, +0.0019] | not separated from C (interval contains 0) |
| B2 (registered) | 0.0773 | 0.0489 | 0.1058 | −0.0550 [−0.1027, −0.0105] | C better |
| RG | 0.1047 | 0.1153 | 0.0940 | −0.0823 [−0.0926, −0.0722] | C better |
| U (registered) | 0.1081 | 0.0832 | 0.1330 | −0.0858 [−0.1111, −0.0630] | C better |
| RK | 0.1099 | 0.1156 | 0.1041 | −0.0875 [−0.0993, −0.0746] | C better |
| RKa | 0.1099 | 0.1189 | 0.1008 | −0.0875 [−0.0972, −0.0778] | C better |
| B1 (registered) | 0.2796 | 0.2885 | 0.2708 | −0.2573 [−0.3704, −0.1312] | C better |

- **The pooled-lag baseline PL matches C on Block 3.** The mean difference is −0.0007 and the interval contains 0. PL is better than C at a = 1.30 and worse at 1.50. PL5 is also not separated from C.
- **Both PL and PL5 use the certified conditional threshold U.** They differ from C only in the lag factor: fitted on five windows instead of the base window alone.
- **Every baseline that does not use U loses to C, with intervals entirely below 0.** These are RK, RKa (log κ₀ and a) and RG (log Ĝ per a). They also lose to the registered B2: their mean errors are 0.105–0.110, against B2's 0.077.
- All three regressions predict below the observed median at all 8 settings.
- **RK and RKa have identical mean errors.** This is algebra, not a bug: the κ₀ values are the same at both a, and every residual keeps its sign within a window. Their per-setting errors and intervals differ.

## Own-seed test: per-run residuals, registered comparison machinery

**Setup.**
- Crossers only (the registered primary analysis): 460 runs per a, 8 windows.
- Signed residual e_X = log(R_cross / X) per run.
- Statistic: mean over windows of (mean |e_C| − mean |e_X|).
- Interval: `prospective_own._win_boot` (window-level bootstrap, 10,000 resamples, seed 0), per a, as in the registered P2b/P3.
- C here is the registered own-seed C: U·λ with λ rounded to 1.11487 / 1.16440.

| model | mean \|e\| a = 1.30 | mean \|e\| a = 1.50 | C − model, a = 1.30 | C − model, a = 1.50 |
|---|---:|---:|---|---|
| C | 0.0711 | 0.0739 | — | — |
| PL | 0.0721 | 0.0727 | −0.0010 [−0.0023, −0.00002] (C better, by 0.001) | +0.0012 [−0.0007, +0.0032] (not separated) |
| PL5 | 0.0714 | 0.0726 | −0.0003 [−0.0009, +0.0001] (not separated) | +0.0012 [−0.0012, +0.0038] (not separated) |
| RK | 0.1435 | 0.1354 | −0.0724 [−0.0836, −0.0608] (C better) | −0.0615 [−0.0769, −0.0455] (C better) |
| RKa | 0.1462 | 0.1326 | −0.0751 [−0.0879, −0.0619] (C better) | −0.0587 [−0.0724, −0.0446] (C better) |
| RG | 0.1432 | 0.1273 | −0.0721 [−0.0841, −0.0598] (C better) | −0.0534 [−0.0660, −0.0400] (C better) |
| U (reference) | 0.1220 | 0.1660 | | |

The mean |e| columns are means over windows of the window-mean |e|.

Setting-level check: |log error| of the median crossing R per window (8 windows per a), with `_boot_diff` over windows (`baselines_posthoc_comparisons.csv`).
- **C − PL**: −0.0047 [−0.0084, +0.0003] at 1.30 and +0.0007 [−0.0073, +0.0086] at 1.50. Not separated.
- **C − PL5**: not separated at either a.
- **C − RK, RKa and RG**: between −0.099 and −0.132, every interval below 0.

## Summary (numbers only)

- **The pooled lag factor matches C.** PL (λ from all five Block G windows × U) is statistically indistinguishable from C on Block 3 and on the own-seed test at a = 1.50. At a = 1.30 in the own-seed test, C is better by 0.001, with the interval just below 0. PL5 is indistinguishable from C everywhere.
- **Baselines without the conditional threshold do not match C.** Every regression that avoids U (log κ₀ with a, log κ₀ per a, log Ĝ per a) has about 4–5× C's error on Block 3. It is worse than C, and than the registered B2, with intervals entirely below 0. It is also worse than C on the own-seed test at both a.

## Caveats

- These analyses are post hoc. The baseline list was fixed in code before scoring, but it was written after both tests had been scored and their headline results were known.
- **No baseline uses held-out information.** Fits read only Block G files, and the fit function rejects any other window. Held-out κ₀ and Ĝ are pre-registration, training-free quantities. The only timing caveat is the Block G K (computed after Block 3 was scored, but training-free geometry).
- The regressions are fitted on 5 or 10 settings, and their in-sample mean |log error| is large (0.14–0.15). The largest in-sample residuals are G4_shifted (0.25–0.29), then base and G3_far_outer (0.13–0.16).
