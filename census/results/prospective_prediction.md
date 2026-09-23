# Registration: prospective prediction on held-out window geometries (Block 3)

**Written before any training in the held-out settings (one disclosed timing run, below).**
Date: 2026-09-23. The procedure was fixed in code at `a496a1f` before any prediction value was computed.
Producer: `src/prospective.py`, `src/prospective_windows.py`.

## Change from the plan, recorded before any training

The plan named a = 1.35 and 1.45. **This registration uses a = 1.30 and 1.50.**

- **Reason:** the calibration λ(a) and baselines B1–B3 must come from base-window runs under exactly
  the protocol the new runs use. Those are Block G's runs: `Window.data`, Adam lr 1e-2, 12,000 steps,
  crossing checked every 50 steps. They exist only at a = 1.30 and 1.50 (`blockG_crossings.csv`).
- The phase 2b base crossings at 1.35 and 1.45 use a different sampler and checking cadence. Using
  them would mix protocols.
- With 1.30 and 1.50, the lag correction needs no interpolation.

## Held-out settings (chosen by a structural rule, no training)

- **Candidate family**: symmetric windows with inner half-width i ∈ {0.5, …, 0.9}, outer inner edge
  i + g (g ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8}), and outer outer edge o₁ + d (d ∈ {0.6, 0.8, 1.2}). The five
  windows already used are excluded, leaving 88 candidates.
- **Selection**: κ₀ = K/D∞ for each candidate, from the limiting cubic alone (certified branch and
  bound; `prospective_candidates.csv`). The four candidates nearest the κ₀ quantiles 0.1, 0.35, 0.65
  and 0.9 were chosen (`prospective_windows.csv`).

| setting | inner | outer | κ₀ (certified) |
|---|---|---|---:|
| H10 | [−0.6, 0.6] | ±[0.8, 2.0] | 0.19101 |
| H35 | [−0.6, 0.6] | ±[0.9, 1.7] | 0.30073 |
| H65 | [−0.8, 0.8] | ±[1.4, 2.2] | 0.41889 |
| H90 | [−0.7, 0.7] | ±[1.5, 2.7] | 0.51326 |

Held-out settings: these 4 windows × a ∈ {1.30, 1.50} = **8 settings**.

**No prior training in these windows.**
- Every window ever defined for training, across the whole git history, is one of Block G's five.
- The only results files with a window column are Block G's and this block's certified-Ĝ table.

**Disclosed exception.** Before this registration, **one timing run** was executed: H35, a = 1.30,
seed 999999, to measure runtime (3.2 s). It printed a crossing |w₂| of 4.84.
- Seed 999999 is outside the registered seeds, and the run is excluded from scoring.
- No prediction below depends on it. They come from certified R_glob and frozen existing data, by the
  procedure already committed at `a496a1f`.

## Protocol for the new runs

- `blockG_windows.train`: `Window.data(200 per class, seed)`, float64, Adam lr 1e-2,
  `torch.manual_seed(seed); U(−1, 1)⁴`, 12,000 steps.
- **Crossing**: the first 50-step check at which the oriented dense-grid gap is > 0. This is the same
  definition as the calibration data.
- **Seeds**: 100,000–100,089, i.e. **90 per setting**. If a setting has fewer than 60 crossings, 30
  more seeds are added at a time (100,090 upward), up to 180 per setting.
- **R at crossing** = |w₂|·Ĝ(window, a)/2, with the certified Ĝ (`prospective_ghat.csv`, lower end).

## Predictions (frozen below; computed by `python -m src.prospective predict`)

Primary target: **per-setting median crossing R**.
- **U**, uncalibrated: R_glob(setting). This is the certified conditional threshold from the profiled
  branch and bound on the window's quadrature population (`prospective_rglob_all.csv`), bracketed to
  below one |w₂| grid step (0.05); the prediction is the bracket midpoint in R. The known lag means U
  is expected to miss by a consistent amount.
  **U's expected error range** is the range of log(median crossing R / R_glob) over the 10 existing
  settings, given below.
- **C**, calibrated: λ(a)·R_glob(setting). λ(a) = the base window's median crossing R / its certified
  R_glob, from frozen Block G data. **λ is fitted.** It is separate from the no-fitting claim about
  geometric constants.
- **B1**, output weight without geometry: the base window's median crossing |w₂| at a, converted to R
  with the new window's Ĝ.
- **B2**, a common R without geometry-specific minimisation: the pooled median crossing R over all
  existing settings (5 windows × 2 a, certified Ĝ per window).

Secondary target: **fraction of runs placed by budget B ∈ {1,000, 2,000, 4,000, 8,000, 12,000}**.
- **C**: the fraction of base-task |w₂| trajectories at the same a whose |w₂| at step B reaches C's
  predicted crossing |w₂| = 2R_C/Ĝ(new). Trajectories are the frozen phase 2b checkpoints, budget
  32,000, 40 seeds, linear interpolation in step.
- **B3**: the base window's fraction placed by B at the same a (frozen Block G data).

| setting | a | certified R_glob (bracket in R) | **U** | **C** = λ·R_glob | **B1** | **B2** |
|---|---:|---|---:|---:|---:|---:|
| H10 | 1.30 | [0.19500, 0.19750] (tie-limited) | 0.19625 | 0.21879 | 0.14818 | 0.22803 |
| H10 | 1.50 | [0.20000, 0.20250] | 0.20125 | 0.23434 | 0.16090 | 0.22803 |
| H35 | 1.30 | [0.21125, 0.21250] | 0.21187 | 0.23621 | 0.23266 | 0.22803 |
| H35 | 1.50 | [0.21750, 0.22000] | 0.21875 | 0.25471 | 0.25231 | 0.22803 |
| H65 | 1.30 | [0.22000, 0.22250] | 0.22125 | 0.24666 | 0.32314 | 0.22803 |
| H65 | 1.50 | [0.23000, 0.23500] | 0.23250 | 0.27072 | 0.34990 | 0.22803 |
| H90 | 1.30 | [0.22000, 0.22250] | 0.22125 | 0.24666 | 0.39516 | 0.22803 |
| H90 | 1.50 | [0.23000, 0.23500] | 0.23250 | 0.27072 | 0.42738 | 0.22803 |

Calibration (frozen Block G data): **λ(1.30) = 1.11487, λ(1.50) = 1.16440 (fitted)**; B1 base median crossing |w₂| = 5.5191 (1.30), 2.9422 (1.50); B2 pooled median crossing R = 0.22803. **U's expected error range**: log(obs/U) ∈ [0.0801, 0.1824] (over the 10 existing settings, `prospective_existing_settings.csv`).

*Tie-limited brackets:* at three settings (G2 at both a, H10 at 1.30) the two constrained minima m₊ and m₋ agree to within the certificate's 1e-9 tolerance over a small range of s — expected at a continuous crossing, where m₋ − m₊ is quadratic in G, and G is small in narrow-gap windows. There the bracket (≈ 0.0025 in R) is set by that tolerance, not by the 0.05 grid step.

Secondary (fraction placed by budget):

| setting | a | model | 1k | 2k | 4k | 8k | 12k |
|---|---:|---|---:|---:|---:|---:|---:|
| H10 | 1.30 | B3 | 0.000 | 0.100 | 0.750 | 0.900 | 0.925 |
| H10 | 1.30 | C | 0.000 | 0.000 | 0.425 | 0.900 | 0.925 |
| H10 | 1.50 | B3 | 0.175 | 0.650 | 0.850 | 0.925 | 0.925 |
| H10 | 1.50 | C | 0.000 | 0.375 | 0.825 | 0.900 | 0.925 |
| H35 | 1.30 | B3 | 0.000 | 0.100 | 0.750 | 0.900 | 0.925 |
| H35 | 1.30 | C | 0.000 | 0.025 | 0.750 | 0.900 | 0.925 |
| H35 | 1.50 | B3 | 0.175 | 0.650 | 0.850 | 0.925 | 0.925 |
| H35 | 1.50 | C | 0.125 | 0.650 | 0.850 | 0.925 | 0.925 |
| H65 | 1.30 | B3 | 0.000 | 0.100 | 0.750 | 0.900 | 0.925 |
| H65 | 1.30 | C | 0.000 | 0.275 | 0.825 | 0.925 | 0.925 |
| H65 | 1.50 | B3 | 0.175 | 0.650 | 0.850 | 0.925 | 0.925 |
| H65 | 1.50 | C | 0.350 | 0.650 | 0.875 | 0.925 | 0.925 |
| H90 | 1.30 | B3 | 0.000 | 0.100 | 0.750 | 0.900 | 0.925 |
| H90 | 1.30 | C | 0.025 | 0.500 | 0.850 | 0.925 | 0.925 |
| H90 | 1.50 | B3 | 0.175 | 0.650 | 0.850 | 0.925 | 0.925 |
| H90 | 1.50 | C | 0.450 | 0.675 | 0.875 | 0.925 | 0.925 |

Frozen file: `prospective_predictions.csv`, SHA-256 `6b675fd523d930c99a49fe3f952828f38a8ee00ac01ac87c15ecd7e1047ddc64`.

## Metric, summary and criteria

- **Metric**: absolute log error |log(pred/obs)| per setting, primary target.
- **Summary**: the mean over the 8 held-out settings.
- **Model comparison**: setting-level paired bootstrap. Resample the 8 settings with replacement,
  10,000 times, seed 0. Form a 95% percentile interval of mean(err_C − err_X).
- **Primary success**: **C beats B1 and B2**, i.e. both intervals, mean(err_C − err_B1) and
  mean(err_C − err_B2), lie entirely below 0.
- **Primary failure**: either interval includes or exceeds 0. Reported as a failure, with every setting
  shown in the figure.
- **U**: its per-setting log error, log(U/obs), is expected to lie in the range above. Reported as
  inside or outside that range, setting by setting.
- **Secondary**: the mean absolute error of the placed fraction over the 8 settings × 5 budgets,
  C against B3, with the same setting-level paired bootstrap. Registered expectation: C's is lower
  (interval of the difference below 0). Secondary, reported either way.

If the held-out test fails, it is reported as a failure. **No new settings are chosen afterwards to
find one that passes.**
