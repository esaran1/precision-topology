# Phase 1A: causal forecaster, extrapolation-error estimate on pilot seeds

**PHASE 1A PILOT, 2026-09-30.** Not a registration. Code, tests and an error estimate on FRESH PILOT SEEDS only. No registered seed was used, no registered run was read or re-scored (that is Phase 1B, after the forecaster is frozen). The crossings below are of pilot runs, measured with the existing exact detectors, for the error estimate only. Every number here is generated from `results/phase1a/summary.json` by `src/phase1a_pilot.py` (`render_md`); the ledger regenerates both.

## 1. The forecaster (`src/causal_forecast.py`)

- **Cutoff (a stopping time).** t_c = the first step t ≥ 1 with s_t ≥ f·s_ref, s_ref the run's FROZEN own-branch switch (width 1: s\*_frozen; GELU-T and W2-A: the occupied copy's frozen s_switch). The harness locates t_c (it compares s_{t_c} with a frozen level and passes on only the integer); the forecaster reads rows 0 … t_c − 1 and nothing else of the run's path. **Width 1 has two variants:** `w1_sgd` (the spec literally: s_ref = s\*_frozen) and `w1_sgd_run` (a nested stopping time: t_c = the first t with s_t ≥ f·s\*_run(t), s\*_run(t) the switch of the branch occupied at the causal rule point of s_0 … s_{t−1}; the harness reads the hidden state at that rule point only; `w1_cutoff_on_s_run`). The run's own branch at a = 1.58 often differs from the landscape branch (s\*_run/s\*_frozen 0.82–1.25 on the dev runs), so the spec-literal cutoff can land after the crossing.
- **(i) Extrapolation.** Window = the last ⌈w·(t_c − t_start)⌉ visible steps (at least 50; t_start = the release, or the width-1 rule point). y = log s. Families: `lin` (y linear in t), `quad` (quadratic in t), `rate` (one-step growth rate of y linear in y: an autonomous local model of the slaved output dynamics). Anchored at the last visible point. Width 2: s and the share u = |v₁|/s (lin/quad: the same family in t; rate: u linear in log s); signs of v_{t_c−1}. Extrapolated until ŝ ≥ 1.6·s_ref or the run's budget.
- **(ii) Integration.** Track A's R4 recursion (`linear_response.simulate`; W2-A `width2_asym.r4_recursion`, unchanged) along [visible path, extrapolated path], from the release state (GELU-T, W2-A) or the causal rule point (width 1: the first step ≥ 1 with s ≥ 0.5·s\*_frozen at every step up to t_c − 1), with the frozen branch θ\*(s), H(s) of the occupied copy on the own sample. Lag-free switch: W2-A T and T′, `width2_asym.own_path_switch` along the EXTRAPOLATED v path; otherwise the first extrapolated step with ŝ ≥ the frozen switch (width 1: s\*_run of the rule-point branch).
- **(iii) Output.** t_fc, s_fc, t_sw,fc, s_sw,fc, signed lag in steps (t_fc − t_sw,fc) and scale (r_fc = s_fc/s_sw,fc − 1), and the last row each guard saw.
- **Width-1 Adam (implemented and tested, not piloted).** P = 1/(√v̂ + ε) frozen at the rule point (Track A) or at t_c − 1 (`adam_P = "cutoff"`, the causal analogue of 2A's P at t_sw); m₀ at the rule point; moments guarded with the cutoff.

## 2. Enforcement (`tests/test_causal_forecast.py`)

- The forecasters receive `GuardedArray` views, never arrays. A read of a row ≥ t_c (int, slice bound, index list, mask) raises `CausalityViolation`; the hidden-state view admits only the release row (GELU-T, W2-A) or ONE row before t_c (width 1: the rule point); Adam moments are cut at t_c. `run_forecast` re-asserts every guard's last row < t_c.
- (a) leaky forecasters (reading s_{t_c}; a second hidden row; a hidden state after release) are caught; a guard BYPASS (reading the private array) is caught by poisoning; (b) the real forecasters pass (width-1 SGD and Adam on fresh runs; GELU-T and W2-A T on committed fixtures); (c) NaN and garbage in every row ≥ t_c leave the output identical (guarded and unguarded); (d) Adam moments poisoned at rows ≥ t_c, or NaN except the rows the rule names, and the hidden state NaN except the allowed row: identical.
- Constructed cases: exact exponential growth is extrapolated exactly by every family and gives the analytic crossing and the known lag 1/(ηh) = 40 steps (PASS); a growth rate that changes after the cutoff is an extrapolation error the comparison catches (FAIL); a non-growing path or a switch beyond the budget gives no forecast (UNRESOLVED).

## 3. Settings, seeds and the candidate a

| setting | dev seeds | est seeds | est: eligible (on-branch) / crossed | s_obs/s_ref min–max (est) |
|---|---|---|---|---|
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_frozen (spec-literal) | 9,310,000–9,310,119 (25) | 9,310,200–9,310,239 (40) | 40 / 32 | 0.8136–1.3046 |
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time) | 9,310,000–9,310,119 (25) | 9,310,200–9,310,239 (40) | 40 / 32 | 0.8136–1.3046 |
| GELU-T random start (ρ = 1, η = 0.03) | 9,320,000–9,320,004 (5) | 9,320,100–9,320,119 (20) | 19 / 19 | 0.9985–1.0045 |
| W2-A T (ρ = 2⁻¹⁰, η = 0.03) | 9,330,000–9,330,009 (10) | 9,330,100–9,330,159 (60) | 26 / 26 | 0.9997–1.0026 |
| W2-A T′ (ρ = 2⁻¹², η = 0.03) | 9,340,000–9,340,004 (5) | 9,340,100–9,340,119 (20) | 20 / 20 | 0.9983–0.9998 |

- **Seeds.** Fresh: no 7-digit seed with these prefixes occurs in src/, tests/, results/ or paper/ (`seed_scan.json`), and none is a registered or pilot seed of any test module. Why 20 (T: 60): q90 needs ≥ 10 points to be more than the maximum; with ~20 per setting the q90 is the 18th–19th order statistic and the median's 95% order-statistic interval spans ranks 6–15. T has about half its runs on T at release (98/200 registered), hence 40 seeds, extended to 60 when only 15 of the first 40 were on T (the 20 added seeds were run after the first 40 were forecast; nothing else changed). Width 1: the first est range (9,310,100–119) was inspected, the width-1 cutoff variant and the window start were revised, and that range was RELABELLED dev; the width-1 est range is 9,310,200–239 (40 seeds; about 15% of width-1 runs do not cross within 32,000 steps). Dev seeds (5; T: 10) chose the extrapolation family and window and are not in the error estimate.
- **Candidate a = 1.58** (width-1 SGD). `a_candidate_scan.json`: no a-column of any results CSV equals 1.58; the only value in [1.575, 1.585] is 1.575 in `corner_tracking.csv` (a landscape table, no training; the same file Track A cited for 1.65). The token 1.58 occurs only as a scale value in `sb_fold/branch_losses.csv` (an s-grid at a = 1.50) and in this phase's own ledger check. Landscape (`landscape_w1.json`, Track A's construction): s\*_pop = 2.090122, validated by the conditional search (unplaced at 0.995·s\*, placed at 1.005·s\*). The pilot crossings at 1.58 exist only on the pilot seeds (disclosed).
- W2-A uses its registered pipeline (hold, ρ from its pilot.json, budgets, copies); GELU-T its registered pipeline (ρ = 1). The follow check (a hidden-state read at t₀.₈) is NOT a forecaster input; it is recorded for the pilot runs only.

## 4. Choice of the extrapolation family and window (dev seeds only)

- **First rule** (written before the dev forecasts were inspected): fewest missing dev forecasts, ties by the pooled median (over settings and f) of the per-setting median \|t_fc − t_obs\|/max(\|lag\|, 1). On the first dev pass it picked lin/0.25, among the least accurate configurations, because one long-stall width-1 dev run (seed 9,310,000: s fell to 0.21·s_ref, then rose to the switch in ~500 steps) had missing forecasts in most other configurations. With the width-1 window counted from the rule point (§1) those forecasts exist, and on the final dev data the first rule picks `quad/0.05`.
- **Revised rule (DISCLOSED DEVIATION; after inspecting the first dev pass, before any est forecast of the final est ranges):** the same pooled median with a missing forecast counted as +∞ in its per-setting median. The window 0.05 was added to the dev grid after the first dev pass. **Primary: `quad/0.05`** (both rules agree on the final dev data). The est seeds played no part in either choice. "Missing" includes the dev runs that never reach the cutoff (never cross), the same in every configuration.
- **Horizon (disclosed change during the dev pass):** the extrapolation horizon was raised from 1.3·s_ref to 1.6·s_ref after the first, aborted dev pass: at 1.3 the width-1 extrapolation ended before the predicted crossing in a dev run with s\*_run/s\*_frozen = 1.18 and r ≈ 0.08 (1.6·s\*_frozen is within the branch grid, 1.7·s\*_frozen). The R4 integration was made lazy (θ\*, H evaluated only up to the hit); its output is identical (checked on a logged GELU-T forecast).

| family / window | dev cells | missing dev forecasts | pooled median rel. error (missing ignored) | pooled median rel. error (missing = +∞) |
|---|---|---|---|---|
| lin/0.05 | 16 | 12 | 1.868 | 1.868 |
| lin/0.1 | 16 | 12 | 2.365 | 2.365 |
| lin/0.25 | 16 | 12 | 3.865 | 3.865 |
| quad/0.05 **(primary)** (first rule) | 16 | 12 | 0.225 | 0.227 |
| quad/0.1 | 16 | 12 | 0.270 | 0.273 |
| quad/0.25 | 16 | 15 | 0.474 | 0.610 |
| rate/0.05 | 16 | 12 | 0.412 | 0.412 |
| rate/0.1 | 16 | 12 | 0.614 | 0.614 |
| rate/0.25 | 16 | 12 | 1.219 | 1.219 |

## 5. Error estimate (est seeds; primary `quad/0.05`)

Errors are forecast − actual, over the runs with a forecast and a crossing. lag = t_obs − t_sw,act (steps). "/|lag|" = in units of the run's actual lag (denominator at least 1 step). ratio = r_obs/r_fc (signed scale lags). baseline = the lag-free forecast t_sw,fc. ref = the same integration along the ACTUAL path (path-conditioned, POST HOC; not causal; shows how much of the error is extrapolation).

### width 1, SGD lr 0.3, a = 1.58, cutoff on s*_frozen (spec-literal)

| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \|err\| median / q90 (steps) | \|err\|/\|lag\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | \|switch err\| median / q90 | baseline \|err\| median / q90; beats | ref \|err\| median / q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 26 / 40 | 32/32; 23/26; 23/26 | 14 / 294.5 | 16 / 30 | 0.19 / 0.36 | 8 / 12.5 | 1.283 / 1.789 / 2.700 | 24 / 41 | 60 / 68.5; 23/26 | 4 / 4 |
| 0.9 | 32 / 40 | 29/32; 29/32; 27/32 | -97 / 194 | 3 / 20.9 | 0.04 / 0.24 | 6 / 8 | 1.054 / 1.172 / 1.600 | 3 / 18 | 78 / 88; 29/32 | 4 / 4 |
| 0.95 | 32 / 40 | 29/32; 29/32; 8/32 | -156 / 141 | 4 / 5 | 0.05 / 0.06 | 5 / 6 | 1.080 / 1.097 / 1.424 | 0 / 11.4 | 80.5 / 90.7; 32/32 | 4 / 4 |
| 0.98 | 32 / 40 | 29/32; 28/32; 7/32 | -192 / 107.5 | 4 / 5 | 0.05 / 0.06 | 4 / 5.9 | 1.047 / 1.075 / 1.291 | 0 / 6.6 | 82 / 90.8; 32/32 | 4 / 4 |

Actual lag (steps, est runs with a forecast at f = 0.8): min 66, median 84, max 96. Missing forecasts by status (f = 0.98): no cutoff: s never reaches f·s_ref in the saved path: 8; ok: 32.

### width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time)

| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \|err\| median / q90 (steps) | \|err\|/\|lag\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | \|switch err\| median / q90 | baseline \|err\| median / q90; beats | ref \|err\| median / q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 32 / 40 | 32/32; 32/32; 32/32 | 216 / 287.5 | 16 / 19.9 | 0.19 / 0.25 | 8 / 10 | 1.688 / 1.829 / 1.970 | 24 / 26.9 | 59 / 64; 32/32 | 4 / 4 |
| 0.9 | 32 / 40 | 32/32; 32/32; 30/32 | 143 / 186.5 | 3 / 4 | 0.03 / 0.05 | 5 / 6.9 | 1.153 / 1.172 / 1.203 | 2 / 3 | 81.5 / 87.8; 32/32 | 4 / 4 |
| 0.95 | 32 / 40 | 32/32; 32/32; 0/32 | 105 / 134.5 | 4 / 5 | 0.05 / 0.06 | 4.5 / 6 | 1.082 / 1.094 / 1.117 | 0 / 1 | 83.5 / 90.7; 32/32 | 4 / 4 |
| 0.98 | 32 / 40 | 32/32; 32/32; 0/32 | 82 / 104.5 | 4 / 5 | 0.05 / 0.06 | 4 / 5 | 1.056 / 1.071 / 1.086 | 0 / 0 | 84 / 90.8; 32/32 | 4 / 4 |

Actual lag (steps, est runs with a forecast at f = 0.8): min 66, median 84, max 96. Missing forecasts by status (f = 0.98): no cutoff: s never reaches f·s_ref in the saved path: 8; ok: 32.

### GELU-T random start (ρ = 1, η = 0.03)

| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \|err\| median / q90 (steps) | \|err\|/\|lag\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | \|switch err\| median / q90 | baseline \|err\| median / q90; beats | ref \|err\| median / q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 19 / 19 | 19/19; 19/19; 19/19 | 1789 / 2228 | 80 / 165.4 | 4.38 / 12.98 | 1 / 3.2 | 0.859 / 1.090 / 1.320 | 78 / 164.4 | 65 / 117; 2/19 | 2 / 4.2 |
| 0.9 | 19 / 19 | 19/19; 19/19; 19/19 | 903 / 1151 | 12 / 16.8 | 0.63 / 1.83 | 1 / 3.4 | 0.739 / 0.977 / 1.172 | 10 / 16 | 11 / 33.2; 10/19 | 2 / 4.2 |
| 0.95 | 19 / 19 | 19/19; 19/19; 19/19 | 451 / 594 | 3 / 5.2 | 0.17 / 0.89 | 2 / 4.2 | 0.682 / 0.942 / 1.145 | 1 / 2.2 | 17 / 50.4; 17/19 | 2 / 4.2 |
| 0.98 | 19 / 19 | 19/19; 19/19; 19/19 | 177 / 249 | 2 / 4.2 | 0.11 / 0.72 | 2 / 4.2 | 0.678 / 0.917 / 1.175 | 0 / 1 | 18 / 53.2; 18/19 | 2 / 4.2 |

Actual lag (steps, est runs with a forecast at f = 0.8): min -16, median 18, max 56. Missing forecasts by status (f = 0.98): ok: 19.

### W2-A T (ρ = 2⁻¹⁰, η = 0.03)

| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \|err\| median / q90 (steps) | \|err\|/\|lag\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | \|switch err\| median / q90 | baseline \|err\| median / q90; beats | ref \|err\| median / q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 26 / 26 | 26/26; 26/26; 26/26 | 2947 / 3939 | 732 / 850.5 | 31.19 / 67.78 | 46.5 / 52.5 | 0.229 / 0.424 / 0.609 | 687.5 / 817.5 | 664 / 765.5; 0/26 | 0 / 0 |
| 0.9 | 26 / 26 | 26/26; 26/26; 26/26 | 1492 / 2008.5 | 51 / 65.5 | 2.28 / 4.47 | 4 / 4 | 0.806 / 0.890 / 0.943 | 47.5 / 61.5 | 26.5 / 34; 0/26 | 0 / 0 |
| 0.95 | 26 / 26 | 26/26; 26/26; 26/26 | 752 / 1022 | 7 / 9.5 | 0.32 / 0.62 | 1 / 1 | 0.947 / 0.975 / 1.000 | 7 / 8.5 | 17 / 31; 21/26 | 0 / 0 |
| 0.98 | 26 / 26 | 26/26; 26/26; 26/26 | 303 / 423.5 | 1 / 1 | 0.04 / 0.10 | 0 / 1 | 0.971 / 0.994 / 1.007 | 1 / 1 | 23.5 / 38; 26/26 | 0 / 0 |

Actual lag (steps, est runs with a forecast at f = 0.8): min 4, median 23.5, max 77. Missing forecasts by status (f = 0.98): ok: 26.
Switch along the extrapolated vs the actual v path: \|s_sw,fc/s_sw,act − 1\| median / q90 = f 0.8: 0.020680 / 0.023776, f 0.9: 0.001415 / 0.001537, f 0.95: 0.000188 / 0.000198, f 0.98: 0.000021 / 0.000022.

### W2-A T′ (ρ = 2⁻¹², η = 0.03)

| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \|err\| median / q90 (steps) | \|err\|/\|lag\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | \|switch err\| median / q90 | baseline \|err\| median / q90; beats | ref \|err\| median / q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 20 / 20 | 20/20; 20/20; 20/20 | 7997 / 9790 | 2192 / 2684.3 | 47.28 / 104.27 | 54 / 71.2 | -11.048 / -2.536 / 22.095 | 2138.5 / 2613.1 | 2177.5 / 2659.1; 5/20 | 0 / 0 |
| 0.9 | 20 / 20 | 20/20; 20/20; 20/20 | 4025 / 4940 | 129 / 152 | 2.77 / 6.01 | 3 / 4 | 1.064 / 1.096 / 1.123 | 126.5 / 149.8 | 171.5 / 190; 19/20 | 0 / 0 |
| 0.95 | 20 / 20 | 20/20; 20/20; 20/20 | 1993 / 2463 | 16 / 19.2 | 0.35 / 0.78 | 1 / 1 | 1.001 / 1.019 / 1.033 | 15 / 18.2 | 63.5 / 72.5; 19/20 | 0 / 0 |
| 0.98 | 20 / 20 | 20/20; 20/20; 20/20 | 759 / 959 | 1 / 2 | 0.03 / 0.11 | 0 / 1 | 0.990 / 0.997 / 1.015 | 2 / 2 | 48.5 / 60.3; 19/20 | 0 / 0 |

Actual lag (steps, est runs with a forecast at f = 0.8): min -69, median -46.5, max 1. Missing forecasts by status (f = 0.98): ok: 20.
Switch along the extrapolated vs the actual v path: \|s_sw,fc/s_sw,act − 1\| median / q90 = f 0.8: 0.029561 / 0.032790, f 0.9: 0.001680 / 0.001813, f 0.95: 0.000203 / 0.000224, f 0.98: 0.000019 / 0.000022.

### Sensitivity: every family and window (est seeds; \|err\| median / q90 in steps, missing)

| setting | config | f = 0.8 | f = 0.9 | f = 0.95 | f = 0.98 |
|---|---|---|---|---|---|
| w1_sgd | lin/0.05 | 44 / 88.7 (8) | 14.5 / 47 (8) | 6 / 32.1 (8) | 3 / 24.2 (8) |
| w1_sgd | lin/0.1 | 44 / 88.7 (8) | 14.5 / 47 (8) | 6 / 32.1 (8) | 3 / 24.2 (8) |
| w1_sgd | lin/0.25 | 47 / 90.6 (8) | 18 / 52 (8) | 8 / 37.1 (8) | 4 / 29.2 (8) |
| w1_sgd | quad/0.05 | 16 / 30 (14) | 3 / 20.9 (8) | 4 / 5 (8) | 4 / 5 (8) |
| w1_sgd | quad/0.1 | 16 / 30 (14) | 3 / 20.9 (8) | 4 / 5 (8) | 4 / 5 (8) |
| w1_sgd | quad/0.25 | 20.5 / 37.5 (14) | 2 / 14.2 (10) | 4 / 9.5 (8) | 4 / 5 (8) |
| w1_sgd | rate/0.05 | 4 / 28.2 (8) | 3 / 4 (8) | 4 / 4.9 (8) | 4 / 4 (8) |
| w1_sgd | rate/0.1 | 4 / 28.2 (8) | 3 / 4 (8) | 4 / 4.9 (8) | 4 / 4 (8) |
| w1_sgd | rate/0.25 | 5 / 30.1 (8) | 3 / 5.8 (8) | 4 / 4.9 (8) | 4 / 4.9 (8) |
| w1_sgd_run | lin/0.05 | 42.5 / 49.8 (8) | 14 / 16 (8) | 6 / 7 (8) | 3 / 4 (8) |
| w1_sgd_run | lin/0.1 | 42.5 / 49.8 (8) | 14 / 16 (8) | 6 / 7 (8) | 3 / 4 (8) |
| w1_sgd_run | lin/0.25 | 45.5 / 54.7 (8) | 17.5 / 20.9 (8) | 8 / 10 (8) | 4 / 5 (8) |
| w1_sgd_run | quad/0.05 | 16 / 19.9 (8) | 3 / 4 (8) | 4 / 5 (8) | 4 / 5 (8) |
| w1_sgd_run | quad/0.1 | 16 / 19.9 (8) | 3 / 4 (8) | 4 / 5 (8) | 4 / 5 (8) |
| w1_sgd_run | quad/0.25 | 21 / 26.9 (8) | 2 / 3 (8) | 4 / 5 (8) | 4 / 5 (8) |
| w1_sgd_run | rate/0.05 | 3 / 4 (8) | 3 / 4 (8) | 4 / 4 (8) | 4 / 4 (8) |
| w1_sgd_run | rate/0.1 | 3 / 4 (8) | 3 / 4 (8) | 4 / 4 (8) | 4 / 4 (8) |
| w1_sgd_run | rate/0.25 | 4 / 6.9 (8) | 3 / 3 (8) | 4 / 4 (8) | 4 / 4.9 (8) |
| gelu_random | lin/0.05 | 325 / 429 (0) | 90 / 119.4 (0) | 25 / 36 (0) | 4 / 8.6 (0) |
| gelu_random | quad/0.05 | 80 / 165.4 (0) | 12 / 16.8 (0) | 3 / 5.2 (0) | 2 / 4.2 (0) |
| gelu_random | quad/0.1 | 88 / 192 (0) | 14 / 21.4 (0) | 4 / 5.2 (0) | 2 / 4.2 (0) |
| gelu_random | rate/0.05 | 25 / 59 (0) | 6 / 8.2 (0) | 2 / 4.2 (0) | 2 / 4.2 (0) |
| w2a_T | quad/0.05 | 732 / 850.5 (0) | 51 / 65.5 (0) | 7 / 9.5 (0) | 1 / 1 (0) |
| w2a_T | quad/0.1 | 915.5 / 1088.5 (0) | 68 / 91.5 (0) | 12 / 16.5 (0) | 2 / 3 (0) |
| w2a_T | rate/0.05 | 416.5 / 440 (0) | 116 / 123 (0) | 36 / 39 (0) | 9 / 11 (0) |
| w2a_Tp | quad/0.05 | 2192 / 2684.3 (0) | 129 / 152 (0) | 16 / 19.2 (0) | 1 / 2 (0) |
| w2a_Tp | quad/0.1 | 2447 / 3161.2 (0) | 153.5 / 187.4 (0) | 22 / 28.3 (0) | 3 / 4.1 (0) |
| w2a_Tp | rate/0.05 | 1239 / 1450.7 (0) | 336 / 397.2 (0) | 97.5 / 116.7 (0) | 22 / 27.1 (0) |

## 6. Proposal for the author (checkpoint 1)

Rules (fixed before any est forecast; the f rule REVISED after the dev pass, disclosed): **f** = the largest f in the grid such that, in every setting and every crossed est run with a forecast, the cutoff precedes both the crossing and the actual lag-free switch by at least that run's \|lag\| (the first rule, ≥ 2·max\|lag\| steps before the crossing, admitted only f = 0.8 on the dev runs, where the W2-A crossing forecasts are off by hundreds to thousands of steps). **Step tolerances** τ_cross = 1.5·q90 \|t_fc − t_obs\| and τ_lag = 1.5·q90 \|lag_fc − lag_act\|, rounded up to 5 steps (1C criterion: ≥ 80% of scored runs within τ). **Ratio band** b = max(0.10, \|m − 1\| + 2·SE) rounded up to 0.05 (m the pilot median of r_obs/r_fc, SE = 1.2533·(IQR/1.349)/√n; 1C criterion: median r_obs/r_fc in [1 − b, 1 + b]). **Baseline**: the forecast beats the lag-free forecast t_sw,fc in at least half the runs.

- **One-lag rule, one common f** (width 1 with the s\*_run cutoff): 0.8 (with the spec-literal width-1 cutoff: none; first rule: 0.8): largest 0.8.
- **One-lag rule, per setting:** w1_sgd None; w1_sgd_run 0.8; gelu_random 0.98; w2a_T 0.98; w2a_Tp 0.98.
- **Minimal rule** (the cutoff strictly before both the crossing and the actual lag-free switch in every crossed est run, no horizon requirement), one common f: 0.8, 0.9, 0.95, 0.98 (with the spec-literal width-1 cutoff: none).
- **Why they differ:** width-1 SGD (lr 0.3, ρ = 1) reaches its switch within about one lag (~80 steps) of a 0.9–0.95 cutoff, so the one-lag rule leaves only f = 0.8 there; at f = 0.8 the slow W2-A paths are extrapolated over thousands of steps and their crossing forecasts are off by hundreds to thousands of steps (their lag forecasts much less: §5).
- **Recommendation (FLAGGED: made after the est results were seen; the rules above are unchanged): one common f = 0.95, family `quad`, window 0.05·(t_c − t_start), at least 50 steps, and the width-1 cutoff on s\*_run.** At f = 0.95 the cutoff precedes both events in every crossed est run of every setting with a forecast (§5, third column), by hundreds of steps on the GELU-T and W2-A paths (several lags; s margin ≈ 5% against \|r\| ≲ 0.2%); on the fast width-1 path it precedes the crossing by 1.5–1.9 lags but the lag-free switch by only 37–76 steps (0.4–0.9 lag; f = 0.9: 73–151 steps, 1.9–2.7 lags to the crossing), so for width 1 the author may prefer f = 0.9 (errors: §5). The crossing and lag errors at 0.95 are a few steps. f = 0.98 is more accurate still but its W2-A and GELU-T horizon is a few hundred steps; f = 0.8 extrapolates the slow paths too far. The final f is the author's decision.

**Tolerances at the recommended f = 0.95 (proposal):**

| setting | f | τ_cross (steps) | within | τ_lag (steps) | within | ratio band ±b | pilot median ratio | beats lag-free |
|---|---|---|---|---|---|---|---|---|
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_frozen (spec-literal) | 0.95 | 10 | 29/32 | 10 | 32/32 | 0.15 | 1.097 | 32/32 |
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time) | 0.95 | 10 | 32/32 | 10 | 32/32 | 0.10 | 1.094 | 32/32 |
| GELU-T random start (ρ = 1, η = 0.03) | 0.95 | 10 | 19/19 | 10 | 19/19 | 0.15 | 0.942 | 17/19 |
| W2-A T (ρ = 2⁻¹⁰, η = 0.03) | 0.95 | 15 | 26/26 | 5 | 26/26 | 0.10 | 0.975 | 21/26 |
| W2-A T′ (ρ = 2⁻¹², η = 0.03) | 0.95 | 30 | 20/20 | 5 | 20/20 | 0.10 | 1.019 | 19/20 |

**Tolerances at the per-setting one-lag f:**

| setting | f | τ_cross (steps) | within | τ_lag (steps) | within | ratio band ±b | pilot median ratio | beats lag-free |
|---|---|---|---|---|---|---|---|---|
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time) | 0.8 | 30 | 32/32 | 15 | 32/32 | 0.90 | 1.829 | 32/32 |
| GELU-T random start (ρ = 1, η = 0.03) | 0.98 | 10 | 19/19 | 10 | 19/19 | 0.15 | 0.917 | 18/19 |
| W2-A T (ρ = 2⁻¹⁰, η = 0.03) | 0.98 | 5 | 26/26 | 5 | 26/26 | 0.10 | 0.994 | 26/26 |
| W2-A T′ (ρ = 2⁻¹², η = 0.03) | 0.98 | 5 | 20/20 | 5 | 20/20 | 0.10 | 0.997 | 19/20 |

**Tolerances at the common one-lag f = 0.8:**

| setting | f | τ_cross (steps) | within | τ_lag (steps) | within | ratio band ±b | pilot median ratio | beats lag-free |
|---|---|---|---|---|---|---|---|---|
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_frozen (spec-literal) | 0.8 | 45 | 24/26 | 20 | 25/26 | 0.90 | 1.789 | 23/26 |
| width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time) | 0.8 | 30 | 32/32 | 15 | 32/32 | 0.90 | 1.829 | 32/32 |
| GELU-T random start (ρ = 1, η = 0.03) | 0.8 | 250 | 19/19 | 5 | 18/19 | 0.20 | 1.090 | 2/19 |
| W2-A T (ρ = 2⁻¹⁰, η = 0.03) | 0.8 | 1280 | 26/26 | 80 | 26/26 | 0.70 | 0.424 | 0/26 |
| W2-A T′ (ρ = 2⁻¹², η = 0.03) | 0.8 | 4030 | 20/20 | 110 | 20/20 | 6.80 | -2.536 | 5/20 |

The final f, window and tolerances are the author's decision. Compute (one process, one thread, nice 15; runs + forecasts, all seeds, summed per-row wall clock): 10.77 h, of which 45 forecast rows ran during the machine's overnight idle period and are inflated (single forecasts of 6–34 min that take ~4 s by day); with those rows at their setting's median: 2.48 h. Peak RSS of any job < 0.5 GB.

## 7. Reproduce

```
python -m src.phase1a_pilot scan; python -m src.phase1a_pilot landscape
python -m src.phase1a_pilot run SETTING dev|est        # resumable; paths/ untracked, hashed in the rows
python -m src.phase1a_pilot forecast SETTING dev|est   # every family, window and f
python -m src.phase1a_pilot summarize                  # summary.json, error_estimate.md
```
