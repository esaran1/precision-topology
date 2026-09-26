# Writer inputs v4 — completion patch (2026-09-24)

Written for the paper's writer. Each item is tied to a ledger ID in `src/verify_ledger.py` (group "V4 patch").
Producer: `src/writer_patch.py` (`python -m src.writer_patch`), which also renders this file. Every item here is
**ready for the submission**.

## WP-1. Prospective own-seed test, per setting

Per setting: the number of runs that crossed, and the mean per-run |log(R_cross / prediction)| for five
predictors:
- **U_own**: the run's own threshold, nothing fitted;
- **U**: the population R_glob;
- **C**: λ(a)·R_glob, fitted;
- **C_own**: ρ_res(a)·R_own, fitted;
- **S-early**: the early branch's own threshold, unfitted. No run was undefined for S-early.

**Intervals**:
- **Per-setting rows**: a run-level bootstrap 95% (10,000 resamples, seed 0). This is **descriptive**. The
  registered unit of uncertainty is the window, so no per-setting interval was registered.
- **The "all 8" rows**: the registered window-level bootstrap (10,000 resamples of windows, seed 0).

Source: `writer_patch_prospective_own_per_setting.csv`, built from `prospective_own_predictions.csv` and
`prospective_own_runs.csv` with the registered definitions.

| setting | crossed | U_own | U | C | C_own | S_early |
|---|---|---|---|---|---|---|
| V10, a = 1.30 | 57/60 | 0.046 [0.034, 0.061] | 0.121 [0.100, 0.143] | 0.086 [0.070, 0.104] | 0.022 [0.008, 0.039] | 0.075 [0.050, 0.104] |
| V10, a = 1.50 | 57/60 | 0.079 [0.066, 0.096] | 0.143 [0.120, 0.167] | 0.088 [0.070, 0.107] | 0.033 [0.016, 0.054] | 0.102 [0.079, 0.128] |
| V20, a = 1.30 | 57/60 | 0.053 [0.039, 0.070] | 0.115 [0.095, 0.136] | 0.082 [0.065, 0.100] | 0.028 [0.012, 0.048] | 0.072 [0.047, 0.104] |
| V20, a = 1.50 | 57/60 | 0.085 [0.074, 0.099] | 0.142 [0.121, 0.164] | 0.079 [0.061, 0.097] | 0.033 [0.017, 0.053] | 0.098 [0.080, 0.122] |
| V30, a = 1.30 | 57/60 | 0.052 [0.039, 0.067] | 0.109 [0.091, 0.128] | 0.073 [0.058, 0.090] | 0.027 [0.012, 0.045] | 0.070 [0.047, 0.098] |
| V30, a = 1.50 | 57/60 | 0.087 [0.078, 0.097] | 0.139 [0.119, 0.159] | 0.069 [0.054, 0.084] | 0.033 [0.020, 0.050] | 0.113 [0.090, 0.141] |
| V40, a = 1.30 | 57/60 | 0.053 [0.042, 0.067] | 0.100 [0.083, 0.117] | 0.065 [0.052, 0.079] | 0.026 [0.013, 0.043] | 0.070 [0.050, 0.095] |
| V40, a = 1.50 | 57/60 | 0.092 [0.084, 0.101] | 0.143 [0.127, 0.160] | 0.058 [0.047, 0.070] | 0.037 [0.025, 0.051] | 0.118 [0.097, 0.142] |
| V50, a = 1.30 | 58/60 | 0.056 [0.047, 0.068] | 0.166 [0.143, 0.190] | 0.091 [0.074, 0.111] | 0.029 [0.017, 0.045] | 0.070 [0.054, 0.090] |
| V50, a = 1.50 | 58/60 | 0.103 [0.095, 0.112] | 0.219 [0.193, 0.246] | 0.098 [0.080, 0.117] | 0.046 [0.035, 0.060] | 0.125 [0.108, 0.146] |
| V60, a = 1.30 | 58/60 | 0.052 [0.044, 0.063] | 0.146 [0.126, 0.167] | 0.075 [0.060, 0.091] | 0.025 [0.015, 0.039] | 0.074 [0.055, 0.098] |
| V60, a = 1.50 | 58/60 | 0.108 [0.099, 0.117] | 0.204 [0.182, 0.227] | 0.084 [0.070, 0.099] | 0.051 [0.040, 0.063] | 0.136 [0.117, 0.159] |
| V70, a = 1.30 | 58/60 | 0.056 [0.050, 0.064] | 0.116 [0.100, 0.131] | 0.053 [0.043, 0.065] | 0.028 [0.020, 0.040] | 0.079 [0.062, 0.097] |
| V70, a = 1.50 | 58/60 | 0.108 [0.099, 0.118] | 0.170 [0.152, 0.187] | 0.061 [0.050, 0.073] | 0.051 [0.040, 0.063] | 0.138 [0.121, 0.159] |
| V80, a = 1.30 | 58/60 | 0.053 [0.046, 0.061] | 0.104 [0.092, 0.115] | 0.042 [0.033, 0.053] | 0.026 [0.017, 0.037] | 0.070 [0.054, 0.089] |
| V80, a = 1.50 | 58/60 | 0.109 [0.098, 0.120] | 0.167 [0.151, 0.183] | 0.055 [0.046, 0.065] | 0.052 [0.041, 0.065] | 0.133 [0.115, 0.154] |
| all 8 (window-level), a = 1.30 | 460/480 | 0.052 [0.050, 0.054] | 0.122 [0.109, 0.138] | 0.071 [0.060, 0.081] | 0.026 [0.025, 0.028] | 0.072 [0.070, 0.075] |
| all 8 (window-level), a = 1.50 | 460/480 | 0.096 [0.089, 0.104] | 0.166 [0.148, 0.188] | 0.074 [0.064, 0.084] | 0.042 [0.036, 0.048] | 0.120 [0.111, 0.130] |

## WP-2. Window definitions and the calibration rules

Class 0 on I, class 1 on O. Every window is x-symmetric. Source: `writer_patch_windows.csv`, built from
`prospective_windows.csv` and `prospective_own_windows.csv`.

| window | test | I | O | kappa0_lo | kappa0_hi |
|---|---|---|---|---|---|
| H10 | Block 3 | [-0.6, 0.6] | [-2.0, -0.8] U [0.8, 2.0] | 0.19101 | 0.19102 |
| H35 | Block 3 | [-0.6, 0.6] | [-1.7, -0.9] U [0.9, 1.7] | 0.30072 | 0.30074 |
| H65 | Block 3 | [-0.8, 0.8] | [-2.2, -1.4] U [1.4, 2.2] | 0.41889 | 0.41891 |
| H90 | Block 3 | [-0.7, 0.7] | [-2.7, -1.5] U [1.5, 2.7] | 0.51326 | 0.51329 |
| V10 | own-seed | [-0.7, 0.7] | [-1.5, -0.9] U [0.9, 1.5] | 0.19030 | 0.19031 |
| V20 | own-seed | [-0.8, 0.8] | [-1.9, -1.1] U [1.1, 1.9] | 0.24159 | 0.24161 |
| V30 | own-seed | [-0.9, 0.9] | [-2.1, -1.3] U [1.3, 2.1] | 0.27897 | 0.27899 |
| V40 | own-seed | [-0.9, 0.9] | [-2.0, -1.4] U [1.4, 2.0] | 0.33443 | 0.33445 |
| V50 | own-seed | [-0.5, 0.5] | [-2.1, -0.9] U [0.9, 2.1] | 0.35854 | 0.35856 |
| V60 | own-seed | [-0.6, 0.6] | [-2.3, -1.1] U [1.1, 2.3] | 0.39845 | 0.39847 |
| V70 | own-seed | [-0.5, 0.5] | [-1.5, -0.9] U [0.9, 1.5] | 0.43300 | 0.43304 |
| V80 | own-seed | [-0.9, 0.9] | [-2.3, -1.7] U [1.7, 2.3] | 0.47250 | 0.47254 |

**Calibration rules as registered** (`writer_patch_calibration.csv`):
- **C (Block 3)**: C = λ(a)·R_glob, with λ(1.30) = 1.11487 and λ(1.50) = 1.1644. λ is the base
  window's median crossing R divided by its certified R_glob, frozen before Block 3.
- **C_own (own-seed test)**: C_own = ρ_res(a)·R_own, with ρ_res(1.30) = 1.0311 and
  ρ_res(1.50) = 1.066.
  - ρ_res = exp(median log(R_cross/R_own)) over the base window's phase 2b seeds 0–39 that crossed within
    12,000 steps.
  - It was frozen before any computation on the new settings (`prospective_own_prediction.md`).

## WP-3. The localisation bound W(s, a) for the finite-a certificates

**Lemma (localisation of the finite-a conditional search).**

- **Setting**:
  - z(x) = s·f_a(w₁x + b₁) + b₂, with s = |w₂| > 0 in the orientation w₂ > 0, and f_a(t) = t + a sin t, so
    t − a ≤ f_a(t) ≤ t + a.
  - Class 1 lies on O = [−2, −1.2] ∪ [1.2, 2] and class 0 on I = [−0.8, 0.8], with n points in total and balanced
    classes (ȳ = ½; true of the 800-point population and of every 400-point training set).
  - L* is the profiled loss, minimised over b₂.
- **Data**: the population objective has n = 800 points:
  - 400 inner points x = linspace(−0.8, 0.8, 400), class 0;
  - 200 outer points x = linspace(1.2, 2.0, 200), class 1, and their 200 negatives, class 1.
  (`blockB_landscape.population_data`; a training set is `fold1d.make_data(200, seed)`.)
- **Notation**: for a cut c ∈ [0, 0.8), let n_O⁻ = #{class 1: x ≤ −1.2} and n_I^{≥c} = #{class 0: x ≥ c}. Set
  π(c) = min(n_O⁻, n_I^{≥c})/n and δ(π) = 2 log(2^{1/π} − 1).
- **The cut grid**: C = {0.799·k/79 : k = 0, 1, …, 79}, the 80 cuts the certified code evaluates
  (`np.linspace(0, 0.799, 80)`).
- **Claim**: if w₁ > W₊(s, a) = min over c ∈ C of (2a + δ(π(c))/s)/(1.2 + c), then L*(w₁, b₁; s) > log 2 for every b₁.
  (The argument holds for every single c; minimising over the grid C reproduces the code exactly.)
  The case w₁ < 0 is the mirror image: use the right-outer class-1 points (x ≥ 1.2) against the class-0 points
  with x ≤ −c, which gives W₋.
- **Conclusion**: with W(s, a) = max(W₊, W₋), every global conditional minimiser has |w₁| ≤ W. The constant
  predictor attains exactly log 2, so the global minimum is at most log 2.
- **b₁**: it ranges over [0, 2π). A shift of b₁ by 2π adds 2πs to every logit, which the profiled b₂ absorbs.

**Proof** (w₁ > 0).
1. For x ≤ −1.2, w₁x + b₁ ≤ −1.2w₁ + b₁, so z ≤ A := s(−1.2w₁ + b₁ + a) + b₂.
2. For x ≥ c, z ≥ s(cw₁ + b₁ − a) + b₂ = A + Δ, with Δ = s((1.2 + c)w₁ − 2a).
3. Take Δ > 0 and let M = A + Δ/2.
   - If M ≥ 0, every class-0 point with x ≥ c has z ≥ Δ/2, so its loss softplus(z) is at least softplus(Δ/2).
   - If M < 0, every class-1 point with x ≤ −1.2 has z < −Δ/2, so its loss softplus(−z) exceeds softplus(Δ/2).
4. All other losses are non-negative. So for every b₂, L ≥ π(c)·softplus(Δ/2), and hence
   L* ≥ π(c)·softplus(Δ/2).
5. π·softplus(Δ/2) > log 2 ⟺ Δ > δ(π) ⟺ w₁ > (2a + δ(π)/s)/(1.2 + c).
6. Every c gives a valid bound, so the minimum over the grid C does too.
   (For 1/π ≥ 1000 the code replaces δ by the larger 2 log 2/π, which is conservative; this never occurs here.) ∎

**Illustration** (a = 1.30, s = 4.95, the lower end of the certified R_glob bracket; population data):
- **With the cut c = 0.4**: n_O⁻ = 200, n_I^{≥0.4} = 100, π = 1/8, δ = 2 log 255 = 11.08. This gives
  W₊ = (2.6 + 11.08/4.95)/1.6 = 3.024.
- **With the grid-optimal cut c = 0.799·22/79 = 0.22251**: n_I^{≥c} = 145, π = 145/800 = 0.18125,
  δ = 2 log(2^{1/0.18125} − 1) = 7.6045, W₊ = (2.6 + 7.6045/4.95)/(1.2 + 0.22251) = 2.9077.
  W₋ is the same, because the data are x-symmetric, so **W = 2.908**: the value the certified search used.

**Checked at every certified a** (`writer_patch_w_bound.csv`: population objective, both ends of every
certified R_glob and R_solve bracket):
- The W computed here equals the W used by the certified search (`profiled_bnb.w_bound`) to 1e−12.
- The profiled loss at |w₁| = W, 1.5W and 3W, on 720 values of b₁ and both signs of w₁, is at least 4.34 > log 2
  everywhere.
- The table shows the lower end of each bracket; both ends are in the CSV. The two sides (w₁ > 0 and w₁ < 0)
  give equal W because the population is x-symmetric.

| a | kind | s | W | c_> | n_inner_> | pi_> | min_profiled_loss_at_and_beyond_W |
|---|---|---|---|---|---|---|---|
| 1.30 | glob | 4.9500 | 2.908 | 0.2225 | 145 | 0.18125 | 6.652 |
| 1.30 | solve | 7.1000 | 2.562 | 0.3135 | 122 | 0.15250 | 7.040 |
| 1.35 | glob | 4.0250 | 3.226 | 0.2225 | 145 | 0.18125 | 6.397 |
| 1.35 | solve | 5.7125 | 2.828 | 0.2731 | 132 | 0.16500 | 7.170 |
| 1.40 | glob | 3.3750 | 3.548 | 0.1821 | 155 | 0.19375 | 5.490 |
| 1.40 | solve | 4.7375 | 3.097 | 0.2225 | 145 | 0.18125 | 7.256 |
| 1.45 | glob | 2.8875 | 3.876 | 0.1618 | 160 | 0.20000 | 4.885 |
| 1.45 | solve | 4.0125 | 3.371 | 0.2225 | 145 | 0.18125 | 6.416 |
| 1.50 | glob | 2.5250 | 4.199 | 0.1416 | 165 | 0.20625 | 4.543 |
| 1.50 | solve | 3.4750 | 3.646 | 0.2023 | 150 | 0.18750 | 5.667 |
| 1.60 | glob | 2.0000 | 4.861 | 0.1214 | 170 | 0.21250 | 4.337 |
| 1.60 | solve | 2.7000 | 4.218 | 0.1618 | 160 | 0.20000 | 4.822 |

## WP-4. Registration census by block and by registration file

**Headline convention** (as in the 2026-09-23 census): each registered prediction is counted once across a.
- **Headline**: 206 scored by their registered rules: 97 PASS, 65 FAIL, 8 PARTIAL,
  36 UNRESOLVED.
- **Post hoc**: 22 assigned post hoc: 3 / 6 / 13 / 0.
- **Total**: 228 registered predictions.

**Added in the 2026-09-25 round** (registrations through the current commit):
- Block 4b (`residual_mechanism_design.md`): the two competing hypotheses, inherited displacement and optimiser memory,
  are FAIL at both a (neither intervention removes half the residual; the registered competing outcome holds). Its
  four validity gates pass; teleport + reset is reported with no criterion.
- Width-1 consistency check (`scale_limits_prediction.md`): PASS.
- tanh (`scale_limits_tanh_prediction.md`): FAIL. The registered expectation was not met as written ("neither", by
  the author's decision; the E-stall convention).

**Not in the headline.**
- *Registered decision rule (outcome, not a prediction)*: the width-2 verdict (`scale_limits_prediction.md`). Its
  registration stated a two-outcome rule (G > 0: no placement threshold; G <= 0: a threshold exists) rather than a
  predicted outcome. **Outcome: no placement threshold for f_a at width 2.** The outcome is decided by the
  second-order (Var) selection, which the registration added; its direct check confirms it (below).
- *Registered check, scored*: the width-2 direct check. All 8 scales are placed (the cancelling pair), so the decision
  rule's outcome is confirmed directly (2026-09-25).
- *Pending (registered, not yet scored)*: the width-2 no-gating test (not run).
- *Designed but never registered*: W0–W4 and the tanh criterion of `width2_design.md`. The design registers them
  with frozen thresholds at its step 5, which the small-scale verdict made moot. They are not applicable at width 2.

**Row-level tables shipped in the supplementary**:
- `results/registration_census.csv`: the headline, one row per prediction;
- `results/registration_census_by_unit.csv`: the appendix view, per a;
- `results/registration_census_v4_gates_and_reported.csv`: validity gates, no-criterion items, the registered
  decision rule, pending registrations and designed-but-unregistered predictions;
- `results/registration_tally.csv`: the tallies.

Source for the tables below: `writer_patch_census_by_block.csv`.

### By block
| group | scoring | PASS | FAIL | PARTIAL | UNRESOLVED | n |
|---|---|---|---|---|---|---|
| A5d k=1 | registered rule | 2 | 1 | 0 | 2 | 5 |
| A5d k=10 | registered rule | 1 | 2 | 0 | 2 | 5 |
| Arrhenius 08-27 | registered rule | 15 | 11 | 1 | 15 | 42 |
| B | post hoc (census) | 1 | 2 | 0 | 0 | 3 |
| Block 3 (held-out windows) | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| Block 3 (held-out windows) | registered rule | 2 | 0 | 0 | 0 | 2 |
| Block 4 (fixed scale) | registered rule | 2 | 1 | 0 | 0 | 3 |
| Block 4 horizon extension | registered rule | 0 | 2 | 0 | 0 | 2 |
| Block 4b (residual mechanism) | registered rule | 0 | 2 | 0 | 0 | 2 |
| Block 5 (retention) | registered rule | 2 | 0 | 0 | 0 | 2 |
| C | registered rule | 1 | 2 | 0 | 0 | 3 |
| E | post hoc (census) | 1 | 0 | 0 | 0 | 1 |
| E | registered rule | 2 | 3 | 0 | 0 | 5 |
| E-stall | registered rule | 0 | 2 | 0 | 0 | 2 |
| F | registered rule | 1 | 1 | 0 | 2 | 4 |
| G | registered rule | 3 | 2 | 0 | 0 | 5 |
| H | registered rule | 0 | 1 | 0 | 2 | 3 |
| K | registered rule | 3 | 2 | 0 | 0 | 5 |
| MNIST budget law 09-11 | registered rule | 0 | 1 | 0 | 2 | 3 |
| R-collapse 09-12 | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| R-collapse 09-12 | registered rule | 0 | 1 | 1 | 0 | 2 |
| S2 (withdrawn) | registered rule | 1 | 1 | 0 | 0 | 2 |
| SGD own thresholds (Track 4) | registered rule | 3 | 0 | 0 | 0 | 3 |
| asymmetric windows (Track 2) | registered rule | 2 | 2 | 0 | 2 | 6 |
| band task in R^d (Track 3B) | post hoc (census) | 0 | 0 | 3 | 0 | 3 |
| band task in R^d (Track 3B) | registered rule | 1 | 1 | 0 | 0 | 2 |
| c1 first order | registered rule | 2 | 0 | 0 | 2 | 4 |
| collapse 09-12 | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| collapse 09-12 | registered rule | 1 | 2 | 0 | 0 | 3 |
| corrugation 08-06 | registered rule | 0 | 2 | 0 | 0 | 2 |
| corrugation readings 08-22 | registered rule | 2 | 0 | 0 | 0 | 2 |
| cross-family | registered rule | 3 | 1 | 0 | 0 | 4 |
| early census 08-23 | post hoc (census) | 0 | 2 | 0 | 0 | 2 |
| early census 08-23 | registered rule | 7 | 1 | 2 | 1 | 11 |
| early census 08-23 (basin) | registered rule | 1 | 4 | 0 | 0 | 5 |
| early census 08-23 (width) | registered rule | 3 | 1 | 0 | 0 | 4 |
| interleaved 08-05 | registered rule | 0 | 0 | 0 | 2 | 2 |
| kappa | registered rule | 3 | 1 | 0 | 0 | 4 |
| lag test | registered rule | 1 | 1 | 0 | 0 | 2 |
| lag test 2 (deconfounded) | registered rule | 0 | 2 | 0 | 0 | 2 |
| localization 08-22 | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| localization 08-22 | registered rule | 1 | 0 | 1 | 0 | 2 |
| metric artifact 09-12 | post hoc (census) | 0 | 0 | 2 | 0 | 2 |
| non-sine activations (Track 3A) | registered rule | 0 | 2 | 0 | 2 | 4 |
| nu | registered rule | 0 | 1 | 0 | 0 | 1 |
| own-seed thresholds | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| own-seed thresholds | registered rule | 1 | 1 | 0 | 0 | 2 |
| phase1 | registered rule | 1 | 1 | 0 | 0 | 2 |
| phase2b | registered rule | 0 | 1 | 0 | 0 | 1 |
| phase2b across a | registered rule | 2 | 1 | 0 | 1 | 4 |
| precision | registered rule | 0 | 0 | 0 | 1 | 1 |
| prospective own-seed | registered rule | 8 | 0 | 0 | 0 | 8 |
| ramp (Track 1B) | post hoc (census) | 0 | 0 | 3 | 0 | 3 |
| ramp (Track 1B) | registered rule | 3 | 1 | 0 | 0 | 4 |
| sample size | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| sample size | registered rule | 1 | 1 | 0 | 0 | 2 |
| scale limits (width 2) | registered rule | 1 | 1 | 0 | 0 | 2 |
| scaling limit | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| scaling limit | registered rule | 3 | 0 | 0 | 0 | 3 |
| search 08-22 | registered rule | 2 | 0 | 2 | 0 | 4 |
| third optimizer 09-14 | registered rule | 2 | 0 | 0 | 0 | 2 |
| threshold 08-22 | post hoc (census) | 1 | 0 | 0 | 0 | 1 |
| threshold 08-22 | registered rule | 2 | 1 | 1 | 0 | 4 |
| timescale prospective test (Task B) | registered rule | 2 | 0 | 0 | 0 | 2 |
| width 2 no-gating test (Track 7) | registered rule | 0 | 1 | 0 | 0 | 1 |
| winding 08-22 | registered rule | 4 | 0 | 0 | 0 | 4 |

### By registration file
| group | scoring | PASS | FAIL | PARTIAL | UNRESOLVED | n |
|---|---|---|---|---|---|---|
| results/act_training_registration.md | registered rule | 0 | 2 | 0 | 2 | 4 |
| results/amplification_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/amplification_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/arrhenius_prediction.md | registered rule | 15 | 11 | 1 | 15 | 42 |
| results/asym_registration.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/asym_registration.md (amendment 1) | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/asym_registration.md (amendment 2) | registered rule | 0 | 0 | 0 | 1 | 1 |
| results/asym_registration.md (amendment 3) | registered rule | 0 | 0 | 0 | 1 | 1 |
| results/asym_registration.md (amendment 4) | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/band_rd_registration.md | post hoc (census) | 0 | 0 | 3 | 0 | 3 |
| results/band_rd_registration.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/basin_prediction.md | registered rule | 1 | 4 | 0 | 0 | 5 |
| results/blockA5d_k1_prediction.md | registered rule | 2 | 1 | 0 | 2 | 5 |
| results/blockA5d_prediction.md | registered rule | 1 | 2 | 0 | 2 | 5 |
| results/blockB_prediction.md | post hoc (census) | 1 | 2 | 0 | 0 | 3 |
| results/blockC_equivalence_prediction.md | registered rule | 1 | 2 | 0 | 0 | 3 |
| results/blockE_redesign.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/blockE_stall_prediction.md | registered rule | 0 | 2 | 0 | 0 | 2 |
| results/blockF_lag_prediction.md | post hoc (census) | 1 | 0 | 0 | 0 | 1 |
| results/blockF_lag_prediction.md | registered rule | 2 | 3 | 0 | 2 | 7 |
| results/blockG_g3_correction.md | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/blockG_prediction.md | registered rule | 3 | 1 | 0 | 0 | 4 |
| results/blockH_rate_prediction.md | registered rule | 0 | 1 | 0 | 2 | 3 |
| results/blockK_prediction.md | registered rule | 3 | 2 | 0 | 0 | 5 |
| results/blockS2_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/bottleneck_prediction.md | registered rule | 0 | 1 | 0 | 1 | 2 |
| results/c1_followup_registration.md | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/collapse_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/collapse_prediction.md | registered rule | 1 | 2 | 0 | 0 | 3 |
| results/corrugation_prediction.md | registered rule | 0 | 2 | 0 | 0 | 2 |
| results/corrugation_readings_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/crossfamily_followup_prediction.md | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/crossfamily_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/crossfamily_q1control_prediction.md | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/first_order_prediction.md | registered rule | 1 | 0 | 0 | 2 | 3 |
| results/fixed_scale_horizon_prediction.md | registered rule | 0 | 2 | 0 | 0 | 2 |
| results/fixed_scale_prediction.md | registered rule | 4 | 1 | 0 | 0 | 5 |
| results/fold1d_prediction.md | registered rule | 2 | 0 | 2 | 0 | 4 |
| results/gelu_scale_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/gelu_scale_prediction.md | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/interleaved_predictions.md | registered rule | 0 | 0 | 0 | 2 | 2 |
| results/kappa_certification_prediction.md | registered rule | 3 | 1 | 0 | 0 | 4 |
| results/lag_test2_prediction.md | registered rule | 0 | 2 | 0 | 0 | 2 |
| results/lag_test_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/localization_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/localization_prediction.md | registered rule | 1 | 0 | 1 | 0 | 2 |
| results/metric_artifact_prediction.md | post hoc (census) | 0 | 0 | 2 | 0 | 2 |
| results/mnist_budget_law_prediction.md | registered rule | 0 | 1 | 0 | 2 | 3 |
| results/nu_prediction.md | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/offset_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/own_threshold_prediction.md | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| results/own_threshold_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/phase1_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/phase2b_across_a_prediction.md | registered rule | 2 | 1 | 0 | 1 | 4 |
| results/phase2b_prediction.md | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/precision_prediction.md | registered rule | 0 | 0 | 0 | 1 | 1 |
| results/prospective_own_prediction.md | registered rule | 8 | 0 | 0 | 0 | 8 |
| results/prospective_prediction.md | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| results/prospective_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/r_collapse_prediction.md | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| results/r_collapse_prediction.md | registered rule | 0 | 1 | 1 | 0 | 2 |
| results/ramp_registration.md | post hoc (census) | 0 | 0 | 3 | 0 | 3 |
| results/ramp_registration.md | registered rule | 3 | 1 | 0 | 0 | 4 |
| results/residual_mechanism_design.md | registered rule | 0 | 2 | 0 | 0 | 2 |
| results/sample_size_prediction.md | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| results/sample_size_prediction.md | registered rule | 1 | 1 | 0 | 0 | 2 |
| results/scale_limits_prediction.md | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/scale_limits_tanh_prediction.md | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/scaling_limit_prediction.md | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| results/scaling_limit_prediction.md | registered rule | 3 | 0 | 0 | 0 | 3 |
| results/search_prediction.md | registered rule | 2 | 0 | 2 | 0 | 4 |
| results/sgd_own_registration.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/sgd_own_registration.md (amendment 1) | registered rule | 1 | 0 | 0 | 0 | 1 |
| results/third_optimizer_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/threshold_prediction.md | post hoc (census) | 1 | 0 | 0 | 0 | 1 |
| results/threshold_prediction.md | registered rule | 2 | 1 | 1 | 0 | 4 |
| results/ts_test_registration.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/width2_nogating_design.md | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/width_prediction.md | registered rule | 3 | 1 | 0 | 0 | 4 |
| results/winding_prediction.md | registered rule | 4 | 0 | 0 | 0 | 4 |

## WP-5. Figure manifest after the page-width rebuild

All figures are built at ICLR's text width (5.5 in) and placed at their built size. Every glyph is at least 8 pt.
Producer: `src/figures_v4.py`. Sizes are read from the PDFs (`writer_patch_figure_sizes.csv`).

| file | width_in | height_in | description |
|---|---|---|---|
| results/figures/v4/v4_cond_candidates.pdf | 5.366 | 2.986 | Block 1 at a = 1.30: (a) every candidate of the frozen search above the single retained branch; (b) the branch's gap crossing G = 0 once, with the certified R_glob and R_solve brackets. |
| results/figures/v4/v4_decomposition.pdf | 5.287 | 2.349 | Phase 1: fraction solved, failed on placement (G ≤ 0) or failed on bias (G > 0, b₂ outside its interval) against ε = a − 1; n = 400 runs per a; Clopper–Pearson 95%. |
| results/figures/v4/v4_fixed_scale.pdf | 5.366 | 2.991 | (a) Block 4 placement, (b) Block 5 retention with E-2's criterion and Block E's 33/37 (FAIL), (c) horizon extension (Q1, Q2 FAIL), against held R/R_glob; Clopper–Pearson 95%. |
| results/figures/v4/v4_mirror_branches.pdf | 5.407 | 2.771 | Post hoc: crossing R against the own threshold of the mirror branch occupied at the crossing, marked by branch; Spearman ρ = 0.997 (a = 1.30), 0.995 (a = 1.50), bootstrap 95%; n = 38 per a. |
| results/figures/v4/v4_prospective.pdf | 5.288 | 2.699 | Block 3: predicted against observed median crossing R in 8 held-out settings; C (λ fitted) filled, U (nothing fitted) open, B1, B2; bootstrap 95%. |
| results/figures/v4/v4_prospective_own.pdf | 5.379 | 3.096 | Prospective own-seed test: (a) per-run crossing R against the frozen own threshold, with the fitted C_own lines; (b) mean per-run |log error| of U, C, U_own, C_own at each a with the registered verdicts. |
| results/figures/v4/v4_thresholds.pdf | 5.338 | 3.123 | (a) certified R_glob and R_solve at a = 1.30–1.60 against free-training crossing violins, with the certified ε → 0 limits; (b) small ε: certified brackets at a = 1.01–1.04, the sharp limit and the first-order line; the registered c₁ test INCONCLUSIVE. |

## WP-6. Crossing detection and the residual (post hoc audit; no registered verdict changes)

Sources: `crossing_audit.md`, `crossing_audit_summary.csv`, `cadence_sensitivity.csv` (producers `src/crossing_audit.py`,
`src/cadence_sensitivity.py`). Every rerun reproduced its stored crossing step and |w₂| bit for bit.

**Check interval, stated beside each number.** Phase 2b (S3), the size test, both lag tests (every φ arm; the interval
did not scale with the budget, which was 32,000/φ) and Block 4b check placement **every step**. Block 3, the prospective
own-seed test and the Block G runs behind λ, B1 and B2 check **every 50 steps**.

**Every-step experiments: the residual is unaffected by detection.** The upward bias of the recorded crossing is at most
one step's growth of R: median 0.05%–0.15%
of the threshold, at most 0.44%. The residual against the run's own threshold
moves by at most 0.09 points under interpolation: S3 3.1% → 3.1% (a = 1.30) and
6.6% → 6.6% (a = 1.50); the size test (the lag tests' φ = 1 arms and 4b's control)
3.2% → 3.1% and 6.7% → 6.6%. Detection spacing is ruled out as the
cause of the every-step residual.

**50-step experiments: detection overstates the residual; interpolated values as a labelled sensitivity analysis.**
Crossing located by linear interpolation of G between the last negative and the first positive 50-step check (all
crossing runs):

| quantity (check interval 50 steps) | check-based | interpolated (post hoc) |
|---|---|---|
| own-seed: median R_cross/U_own − 1, a = 1.30 (n = 460) | 4.1% | 2.9% |
| own-seed: median R_cross/U_own − 1, a = 1.50 (n = 460) | 8.8% | 6.2% |
| Block 3: median over settings of (median R/U) − 1, a = 1.30 | 8.8% | 7.3% |
| Block 3: median over settings of (median R/U) − 1, a = 1.50 | 14.2% | 11.5% |
| λ(1.30) (Block G base) | 1.1149 | 1.0979 |
| λ(1.50) (Block G base) | 1.1644 | 1.1298 |
| B2 pooled median crossing R (five Block G windows) | 0.2280 | 0.2260 |

**The residual statement (narrower form, finalised on the full rerun).** Against the run's own threshold, the residual
is about 3% at a = 1.30 and about 6–6.5% at a = 1.50 whether placement is checked every step (S3 3.1% and
6.6%) or every 50 steps with the crossing interpolated (own-seed 2.9% and 6.2%).
Against a population threshold it stays larger after interpolation (Block 3 against U: 7.3% and
11.5%), so that excess is not a detection effect. The check-spacing explanation is ruled out for the every-step residual; in the 50-step
experiments detection adds about 1.2 (a = 1.30) and 2.6
(a = 1.50) points on top of it.

**Registered comparisons under interpolation (sensitivity; the registered verdicts stand as scored).** Own-seed
criteria as registered (statistic [95% interval]; pass/fail check-based / interpolated):

| criterion | check-based | interpolated | verdict check / interpolated |
|---|---|---|---|
| own-seed P1, a = 1.30 | -0.0696 [-0.0852, -0.0565] | -0.0712 [-0.0863, -0.0585] | pass / pass |
| own-seed P2a, a = 1.30 | -0.0136 [-0.0256, -0.0016] | -0.0302 [-0.0372, -0.0226] | pass / pass |
| own-seed P3, a = 1.30 | -0.0448 [-0.0553, -0.0334] | -0.0523 [-0.0593, -0.0447] | pass / pass |
| own-seed P4, a = 1.30 | 0.0419 [0.0010, 0.0610] | 0.0293 [0.0010, 0.0610] | pass / pass |
| own-seed P1, a = 1.50 | -0.0695 [-0.0864, -0.0562] | -0.0698 [-0.0861, -0.0568] | pass / pass |
| own-seed P2b, a = 1.50 | 0.0226 [0.0091, 0.0363] | -0.0025 [-0.0131, 0.0084] | pass / pass |
| own-seed P3, a = 1.50 | -0.0318 [-0.0436, -0.0196] | -0.0468 [-0.0530, -0.0397] | pass / pass |
| own-seed P4, a = 1.50 | 0.0867 [0.0340, 0.0940] | 0.0615 [0.0340, 0.0940] | pass / pass |

For P4 the bracket is the registered acceptance range, not an interval.

**Own-seed at a = 1.50.** P2b passes under both definitions. The stronger reading, that C beats U_own at a = 1.50
(interval above 0), holds check-based (+0.023 [+0.009, +0.036]) but does **not** survive interpolation (-0.003 [-0.013, +0.008]).
**Flag for softening:** the draft's "C is better, as registered" (`WRITER_INPUTS_v4.md` line 13, "P2b: C better at 1.50,
as registered", and line 662, "C is better, +0.023 [+0.009, +0.036]") should say that P2b passes, and that C's advantage
at a = 1.50 depends on the detection rule (check-based +0.023 [+0.009, +0.036]; interpolated -0.003 [-0.013, +0.008]).

**Block 3.** The sensitivity analysis is the **fully interpolated** version: calibration (λ, hence C; B1; B2, all from
the Block G runs) and observations both use interpolated crossings, so one detection rule applies throughout. The
observations-only version is reported beside it, **labelled as mixing detection rules** (predictions calibrated on
50-step checks, observations interpolated). Upper end of the 95% interval of the |log error| difference (C better if
< 0):

| comparison | check-based (as registered) | fully interpolated (sensitivity) | observations only (mixes detection rules) |
|---|---|---|---|
| C - B1 | -0.1312 (excludes 0) | -0.1349 (excludes 0) | -0.1270 (excludes 0) |
| C - B2 | -0.0105 (excludes 0) | -0.0169 (excludes 0) | +0.0053 (includes 0) |
| C - U | -0.0630 (excludes 0) | -0.0544 (excludes 0) | -0.0363 (excludes 0) |

In the fully interpolated version every registered comparison keeps its sign and excludes 0. Mixing the detection
rules makes C − B2 include 0; that version is shown for completeness, not as the analysis.

## WP-7. Ĝ(a) as rigorous enclosures (replaces the float values; Block 2)

Source: `ghat_rigorous.csv` (producer `src/ghat_rigorous.py`, from the independent Arb checker's results in
`certificate_checks/`). **Every R now uses the rigorous lower end** `Ĝ_cert` below: the checker's lower bound on G at
the same witness point as the published float value, rounded down to a double. The upper end is the checker's bound
over every leaf of the branch and bound, rounded up. The domain reduction to w₁ ∈ (0, a/1.4] is analytic.

**(i) Ĝ_cert, the value every R uses** (the rigorous value at the same witness as the old float value):

| a | Ĝ_cert (rigorous, rounded down) | old float Ĝ_cert | change | leaves checked |
|---|---|---|---|---|
| 1.02 | 0.001626744762 | 0.001626744762 | -1.4e-16 | 110,724,630 |
| 1.05 | 0.006360085994 | 0.006360085994 | -1.6e-16 | 18,041,202 |
| 1.10 | 0.017671950853 | 0.017671950853 | -3.8e-17 | 4,086,270 |
| 1.15 | 0.031918407628 | 0.031918407628 | -4.8e-16 | 1,814,754 |
| 1.25 | 0.066505622211 | 0.066505622211 | -1.8e-16 | 722,136 |
| 1.30 | 0.086101793878 | 0.086101793878 | +4.3e-16 | 580,392 |
| 1.35 | 0.106913550661 | 0.106913550661 | -6.9e-17 | 484,512 |
| 1.40 | 0.128771895556 | 0.128771895556 | +1.4e-16 | 415,350 |
| 1.45 | 0.151545318737 | 0.151545318737 | -3.9e-16 | 272,292 |
| 1.50 | 0.175126200489 | 0.175126200489 | -1.1e-15 | 243,594 |
| 1.60 | 0.224360202126 | 0.224360202126 | -1.1e-16 | 200,394 |
| 2.00 | 0.441033924063 | 0.441033924063 | +8.3e-16 | 91,668 |
| 3.00 | 1.052297757851 | 1.052297757851 | -6.7e-16 | 41,988 |

**(ii) The enclosure of the supremum G* = Ĝ(a), reported separately.** Lower end: the best certified lower bound at each
a, the larger of the rigorous values at the two recorded attained points (Ĝ_cert's witness and the branch and bound's
own point). Upper end: the rigorous leaf bound, rounded up. The difference between the best certified lower bound and
Ĝ_cert is given **both absolute and relative**. The largest relative difference is 8.36e-05 (at
a = 1.10, where it is 1.48e-06 absolute); the "up to 8.4e−5" is this relative figure. The largest
absolute difference is 7.09e-06 (at a = 1.60, 3.16e-05 relative).

| a | G* ∈ [best certified lower, upper] | lower end from | best lower − Ĝ_cert (absolute) | (relative) | change of upper end vs old float |
|---|---|---|---|---|---|
| 1.02 | [0.001626745002, 0.001627653245] | branch and bound point | 2.40e-10 | 1.48e-07 | +1.5e-17 |
| 1.05 | [0.006360085994, 0.006363833420] | Ĝ_cert witness | 0.00e+00 | 0.00e+00 | -7.7e-17 |
| 1.10 | [0.017673427712, 0.017688791467] | branch and bound point | 1.48e-06 | 8.36e-05 | -2.9e-16 |
| 1.15 | [0.031920435133, 0.031951889476] | branch and bound point | 2.03e-06 | 6.35e-05 | -1.2e-16 |
| 1.25 | [0.066507277337, 0.066573122000] | branch and bound point | 1.66e-06 | 2.49e-05 | -5.6e-17 |
| 1.30 | [0.086103036064, 0.086170343941] | branch and bound point | 1.24e-06 | 1.44e-05 | +3.1e-16 |
| 1.35 | [0.106914064287, 0.106982835379] | branch and bound point | 5.14e-07 | 4.80e-06 | +1.8e-16 |
| 1.40 | [0.128772506040, 0.128842740347] | branch and bound point | 6.10e-07 | 4.74e-06 | -1.9e-16 |
| 1.45 | [0.151545622037, 0.151689017080] | branch and bound point | 3.03e-07 | 2.00e-06 | -1.1e-16 |
| 1.50 | [0.175126200489, 0.175272043337] | Ĝ_cert witness | 0.00e+00 | 0.00e+00 | -1.7e-16 |
| 1.60 | [0.224367291384, 0.224519465715] | branch and bound point | 7.09e-06 | 3.16e-05 | +3.3e-16 |
| 2.00 | [0.441033924063, 0.441384755802] | Ĝ_cert witness | 0.00e+00 | 0.00e+00 | -3.3e-16 |
| 3.00 | [1.052297757851, 1.053233148730] | Ĝ_cert witness | 0.00e+00 | 0.00e+00 | +4.4e-16 |

**No printed digit of Ĝ or R changes.** The largest relative change of Ĝ is δ = 8.4e-14. Every R is linear in Ĝ.
All 705 printed-number checks of the ledger (`src/verify_ledger.py`, which verifies every
printed number against its artifact) still hold, and round to the same printed digits, with their artifact value
scaled by 1 ± δ (conservatively applied to every check, Ĝ-dependent or not); 0 are unstable
(`ghat_digit_stability.csv`). A further 13 checks compare two artifacts to 1e−12; they are
not printed numbers. 6 of them move in their last digits under the blanket δ, and none of
those depends on the replaced Ĝ(a) (`ghat_digit_stability_machine_precision.csv`: A* is a limit constant; WP-1's P1 and
P3 use the own-seed windows' Ĝ, which is not replaced). That separation was set after the first run flagged them. The old float endpoints are kept in the table; the strict
checks on them keep reporting that they are not rigorous in the last one or two ulps.

Artifacts computed before the switch (stored R columns) keep the old float Ĝ; they differ by at most δ relative, which
the check above covers. Every code path that computes R now reads `ghat_rigorous.ghat_R`.

R keeps Ĝ_cert (table (i)); the best certified lower bound in table (ii) is a valid bound on G* but a different number
from the one R has always used (author's decision 2026-09-24).

## WP-8. The small-scale selection, resolved against the unplaced region (direct check; rebuttal revision)

Sources: `width2_unplaced.csv`, `width2_unplaced_localmin.csv` (producer `src/width2_unplaced.py`), the direct check
`width2_w0 smallscale` (validated W0 search, 4,000 restarts per scale).

At these scales the unplaced region contains local minima: single units at α* with an idle second unit (G₊ ≈ −3.3 to
−3.7), whose loss exceeds the placed cancelling pair's by the predicted single-unit second-order gap (s²/8)·α*²·Var(x)
to within 0.1–2%. The infimum over the unplaced region is lower: it lies on the placement boundary G₊ = 0, at a
partially cancelling pair (|c| ≈ 0.43–0.51). Descent from there reaches the placed cancelling pair, and the loss drop
matches the second-order prediction (s²/8)·c²·Var(x) to within 0.2–3%. In every search the placed minimiser lies below
every unplaced configuration, by at least 9.9e−8 at R₂ = 0.001 (about 99× the 1e−9 tie tolerance).

| scale | best placed: the direct check's retained minimiser | best unplaced (infimum over G₊ ≤ 0) | difference (× tie tolerance) | lowest unplaced local minimum, above the placed pair |
|---|---|---|---|---|
| a = 1.30, R₂ = 0.001 | 0.6922871697904 (placed, cancelling pair) | 0.6922872690039 (boundary, G₊ = -2.0e-10) | 9.92e-08 (99× 1e−9) | 1.61e-06 (single unit, G₊ = -3.65) |
| a = 1.30, R₂ = 0.003 | 0.6905695256248 (placed, cancelling pair) | 0.6905704168112 (boundary, G₊ = -1.2e-11) | 8.91e-07 (891× 1e−9) | 1.45e-05 (single unit, G₊ = -3.64) |
| a = 1.30, R₂ = 0.01 | 0.6845827328908 (placed, cancelling pair) | 0.6845925670975 (boundary, G₊ = -2.0e-12) | 9.83e-06 (9,834× 1e−9) | 1.60e-04 (single unit, G₊ = -3.59) |
| a = 1.30, R₂ = 0.02 | 0.6760975221489 (placed, cancelling pair) | 0.6761364675950 (boundary, G₊ = -1.6e-13) | 3.89e-05 (38,945× 1e−9) | 6.33e-04 (single unit, G₊ = -3.52) |
| a = 1.50, R₂ = 0.001 | 0.6922871675820 (placed, cancelling pair) | 0.6922872667961 (boundary, G₊ = -1.7e-10) | 9.92e-08 (99× 1e−9) | 1.21e-06 (single unit, G₊ = -3.45) |
| a = 1.50, R₂ = 0.003 | 0.6905695190120 (placed, cancelling pair) | 0.6905704102030 (boundary, G₊ = -1.9e-12) | 8.91e-07 (891× 1e−9) | 1.09e-05 (single unit, G₊ = -3.44) |
| a = 1.50, R₂ = 0.01 | 0.6845827109908 (placed, cancelling pair) | 0.6845925452478 (boundary, G₊ = -4.4e-13) | 9.83e-06 (9,834× 1e−9) | 1.20e-04 (single unit, G₊ = -3.40) |
| a = 1.50, R₂ = 0.02 | 0.6760974787560 (placed, cancelling pair) | 0.6761364244002 (boundary, G₊ = -2.9e-13) | 3.89e-05 (38,946× 1e−9) | 4.77e-04 (single unit, G₊ = -3.34) |

## WP-9. Width 2: the small-scale criterion, the registered verdict and the tanh case (rebuttal revision)

Sources: `math_note_v2.md` §10 and §10.1 (at `census/results/math_note_v2.md`), `scale_limits_prediction.md`,
`scale_limits_tanh_prediction.md`, WP-8. IDs are `src/verify_ledger.py` check labels.

**The small-scale criterion (math note §10.1, summary).** For fixed hidden parameters θ, the profiled loss at output
scale s is L*(θ; s) = log 2 − (s/4)·Δμ(θ) + (s²/8)·Var(φ_θ) + O(s⁴) (g‴(0) = 0; remainder checked numerically). So the
small-scale conditional minimiser maximises the class-mean gap Δμ, and among tied maximisers the s² term selects the
one with the smallest Var(φ). If the data gap Γ_n is attained (Lemma 2), output scale gates placement if and only if
this small-scale minimiser is unplaced.

**Width-1 consistency check (registered, passed).** The selected Δμ-maximiser has G ≤ 0 at all six a
(G = -3.349 … -3.662), as required by the certified width-1 thresholds.
IDs: `alpha*`; `width 1: selected dmu-maximiser G <= 0 at all six a`; `width 1: G range lo (a = 1.30)`; `width 1: G range hi (a = 1.60)`.

**The registered width-2 verdict: no placement threshold for f_a.** The Var-selected Δμ-maximiser is the cancelling
cosine pair, placed with G = +0.890 (a = 1.30) and +1.026 (a = 1.50);
W1 and W4 were not run. IDs: `width 2: validation (ladder, independent, pair attains max, pair Var minimal)`; `width 2: selected G at a=1.30`; `width 2: selected G at a=1.50`; `verdict: no placement threshold for f_a at width 2`.

**Its dependency on the variance selection.** At first order the Δμ-maximisers tie: single units (unplaced) and
pairs (placed). Δμ alone does not decide the verdict; the s² (Var) term does. That selection rule was added in the
registration (committed before any maximiser was computed) and was not in the author's original rule. The direct
check (WP-8) tests the verdict without the expansion: at each scale the validated W0 search finds the conditional
minimiser itself. Complete: all 8 scales (placed cancelling pair at each,
every validation passed, the best unplaced configuration on the G = 0 boundary, at least 99× the tie tolerance above)
— the verdict is confirmed directly (table in WP-8, not repeated
here). IDs: `direct check: every landed scale placed, cancelling pair`; `direct check: no finished restart below the retained minimiser (landed scales)`; `min margin at R2 = 0.001`; `all best-unplaced on the boundary`.

**Coverage of the width-2 threshold scan (W0; never registered).** At a = 1.30 no R₂ value was completed: the scan
never started, because its Γ̂₂ stop fired first. At a = 1.50 the scan completed R₂ = 0.02, 0.03, …, 0.10 (9 of 99
planned points; 2,000 restarts each, audit passed, placed at every point), not validated in the design's sense
(validation runs only near a threshold, and there was none), and was terminated. The validated small-scale results
are the direct check's 8 scales (WP-8). Source: `width2_design.md`, "W0 threshold-scan coverage, as run".

**tanh: registered outcome "neither".** Every registered expectation held except one: validation passed at every box
size, the maxima are < 1 and rising, the first-order tie is present, the Var-selected symmetric pair is placed at every
box (G₊ → 1) and the single units are unplaced. The boundary criterion as written fails at A = 40, because the 1e−9 tie
tolerance is of the order of 1 − max (4.1e−9) there. Non-attainment of the supremum is proved analytically
(Δμ < 1 = sup Δμ). No re-registration. tanh lies outside the criterion's attainment hypothesis.
IDs: `tanh: registered outcome 'neither'`; `tanh: validation passed at every A`; `tanh: box maxima < 1 and rising`; `tanh: boundary condition fails only at A = 40`; `tanh: selected member is the symmetric pair at every A`; `tanh: single units unplaced at every A`.

**Say**
- "At width 2, output scale does not gate the placement of the conditional minimiser for f_a: the registered
  small-scale prediction selects the placed cancelling pair, and a direct search at small scales finds that pair as the
  conditional minimiser. Training at small held scale is nevertheless gated (WP-20)."
- "The prediction rests on a second-order selection: at first order, unplaced single units and placed pairs tie."
- "For tanh the supremum of Δμ is not attained; the registered test returned 'neither' because its boundary criterion
  failed at the largest box, although every other expectation held."

**Do not say**
- "proved" or "certified" for the width-2 verdict (it is a registered, validated computation plus a lemma, not a
  certificate); or that Δμ alone decides it.
- that the direct check is complete while any scale is PENDING (it completed on 2026-09-25: all 8 scales placed).
- that the unplaced region has no local minima, or that the boundary point is a competing minimum (WP-8 wording).
- that tanh "confirmed" or "passed" the registered expectation.
- that output scale does not gate width-2 *training*: the registered training test found gating (WP-20).

## WP-10. The residual mechanism: Block 4a (post hoc) and Block 4b (registered)

Sources: `residual_posthoc_correlations.csv`, `residual_mechanism_design.md`, `residual_mechanism_scores.csv`.

**Block 4a (POST HOC correlations).** At the deconfounded switch point (0.7× the own threshold), the final residual
correlates with the displacement from the occupied branch's conditional minimiser (Spearman ρ = 0.68 at
a = 1.30, 0.67 at a = 1.50; n = 48 each; permutation p ≈ 1e−4) and anti-correlates with Adam's
√v̂(w₂) (ρ = -0.47, -0.48).
IDs: `4a a=1.3 displacement rho (post hoc)`; `4a a=1.5 displacement rho (post hoc)`; `4a a=1.3 sqrt vhat(w2) rho (post hoc)`; `4a a=1.5 sqrt vhat(w2) rho (post hoc)`.

**Block 4b (registered): competing outcome at both a — neither teleporting to the branch minimiser nor resetting
Adam's moments removes half the residual; the mechanism is unresolved.** Every validity check passed (control
bit-identical to the lag test's φ = 1 continuation, branch preserved, teleports on the minimiser, 0 excluded, 48/48
crossings per arm). Median residual against the own threshold [95% interval of control − arm]:

| a | control | teleport | reset | teleport + reset (reported, no criterion) |
|---|---|---|---|---|
| 1.30 | 3.105% | 3.079% [-0.05, +0.05] | 3.148% [-0.16, +0.02] | 3.122% [-0.14, +0.04] |
| 1.50 | 6.559% | 6.547% [-0.10, +0.17] | 6.722% [-0.50, +0.02] | 6.601% [-0.37, +0.16] |

IDs: `4b: every validity check passed`; `4b a=1.3: verdict competing (neither removes half)`; `4b a=1.5: verdict competing (neither removes half)`; `4b a=1.3 control residual`; `4b a=1.3 teleport residual`; `4b a=1.5 reset residual`; `4b a=1.3 teleport+reset residual`; `4b a=1.5 teleport+reset residual`.
Detection does not explain the residual either (WP-6).

**Say**
- "Post hoc, the residual correlates with the run's displacement from its branch minimiser at the switch point; a
  registered intervention that removes the displacement does not remove the residual, so the correlation is not causal
  there. Neither inherited displacement nor optimiser memory alone accounts for it; the mechanism is open."

**Do not say**
- that displacement, optimiser memory or adiabatic lag causes or explains the residual;
- that Block 4a's correlations are registered, or that 4b found a small effect of any arm (every interval includes 0).

**Validity note (found 2026-09-25 while replaying these runs; no verdict changed).** In the registered Block 4b run, the
arms shared optimiser-state tensors, and PyTorch's `load_state_dict` does not copy them.
- The control arm's continuation therefore mutated the stored Adam state in place. The **teleport** arm (registered as
  "Adam state kept") started from the control run's *end-of-run* Adam state.
- Likewise, the **teleport_reset** arm started from the reset run's end state rather than zeroed moments.
- Control and reset are unaffected. Replaying in the original order reproduces every recorded crossing (WP-23).
- **Corrected rerun (author's instruction; `residual_mechanism_corrected`, 1838f21).** Both teleport arms were rerun
  from fresh deep copies of each checkpoint, with each start state verified bit for bit.
  - Under the registered criteria, **4b-ID and 4b-OM still FAIL at both a.** No verdict changes.
  - Teleport median residual: 0.03066 (a = 1.30) and 0.06504 (a = 1.50), against as-run 0.03079 and 0.06547.
  - Teleport_reset: 0.03130 and 0.06542, against as-run 0.03122 and 0.06601.
- **Audit of every multi-arm experiment that restarts from saved optimiser state** (`state_audit`). Each arm's actual
  starting parameters and Adam state were compared bit for bit with a fresh load of its checkpoint, with the arms run
  in their original order.
  - Clean: Blocks 4 and 5 (128 arms), the horizon extension (40), both lag-test-2 rules (32) and the no-gating
    replays (48).
  - Only Block 4b's registered teleport arms are flagged. This was the audit's positive control, and the corrected
    arms are clean.
- **Say:** "A shared-state bug affected two arms of one block. A corrected rerun leaves its registered verdicts
  unchanged, and an audit of every other multi-arm experiment found no contamination."
IDs: `Block 4b corrected ID FAIL`; `Block 4b corrected OM FAIL`; `State audit clean arms`; `State audit positive control`.

## WP-11. The independent certificate checker: what is verified, what is pending, what ships

Sources: `certificate_audit.md`, `verify_certificates_finite.log`, `verify_certificates_ghat.log`,
`certificate_checks/`, `certificates_manifest.csv`. Checker: `src/verify_certificates.py` (python-flint / Arb, 80-bit
ball arithmetic; imports nothing from `src/`; every decision an Arb comparison).

**Verified so far.**
- **Finite-a status certificates**, both ends of the global bracket at a = 1.30, 1.35, 1.40, 1.45, 1.50, 1.60 (12
  certificates): every leaf's claim verified, exact tiling, the localisation lemma W recomputed, hashes match. All pass.
  ID: `Block 2: 12 finite-a certificates pass`.
- **Ĝ(a) enclosures**, a = 1.02, 1.05, 1.10, 1.15, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60, 2.00, 3.00 (13 certificates, 42 thousand to 111 million leaves):
  hashes, exact tiling of the domain, and the analytic domain reduction pass at every a. The float endpoints of the
  original searches miss the rigorous enclosure by at most 1.1e−15, so the rigorous enclosures replace them (WP-7).
  IDs: `Ghat certificates checked (a = 1.02-3.0)`; `Ghat: structure passes at every a (hashes, tiling, domain, Ghat_cert <= hi)`.

- **Limit switch**: the status certificates at both ends of the search-certified bracket, A = 0.68125 ("minus") and
  A = 0.6875 ("plus"), both pass. So **A* ∈ [0.68125, 0.6875] is independently verified**, the same bracket as the
  search's. The checks cover every losing-region leaf (outside its region, or loss above U), exact tiling, data
  symmetry, and the localisation bounds B(24) and B_full, recomputed by exact PAVA (so the localisation is verified
  too). IDs: `limit check A_lo passes`; `limit check A_hi passes`.

**Pending (not yet exported or checked):** K = sup G₀ (its domain lemma is written, math note §8), the Krawczyk boxes
for c₁, the solve brackets (finite and limit), outer exclusion and ring, and the PD boxes.

**K's domain (lemma written and checked, 2026-09-24).** K's branch and bound searches (u, v) ∈ [0, 8] × [−12, 12].
The domain lemma (math note §8, "Domain lemma for K") proves G₀(u, v) ≤ 0 whenever |u| ≥ √(50/3) ≈ 4.08 or
|v| ≥ √2 + 0.6|u|. So the supremum lies inside the box, and K is the supremum over the whole plane. The lemma is an exact
algebraic argument, checked numerically: the identity holds to 1.4e−12, and G₀ ≤ 0 at 4.0 million excluded points
(`k_domain_check.csv`). The Arb check of K's branch and bound over the box is still pending, as listed above.
IDs: `K domain: G0 <= 0 on the excluded region (u > 0 max)`; `K domain: region inside the certified box`;
`K domain: K argmax inside the region`.

**What the supplementary will ship** (not rebuilt yet; built once every claim is final): the checker
(`src/verify_certificates.py`) and its tests; the manifest of SHA-256 hashes, each with the command that regenerates
the file bit-identically (`certificates_manifest.csv`); the exporting searches and `src/cert_export.py`. The certificate
data files are not shipped (regenerated from the commands and verified against the hashes); a single command runs the
checker, with its expected output.

**Say**
- "An independent checker in ball arithmetic, sharing no code with the searches, verifies the finite-a certificates,
  the Ĝ enclosures and the limit-switch bracket; the remaining certificate families are being exported."

**Say** (the sharp limit threshold)
- "K = sup G₀ is certified by branch and bound over a box, and a domain lemma shows the supremum lies inside it."

**Do not say**
- that K (or the sharp limit threshold built on it) has been independently checked in Arb, until that check is done;
- that every certificate has been independently checked, until the pending families are;
- that the checker verified the original float endpoints of Ĝ (it verified rigorous enclosures that differ from them
  by at most 1.1e−15).

## WP-12. The mechanism, stated analytically; the scaling law; why Adam is delayed (for the submission)

Sources: math note §11 and §11.1 (`math_note_for_writer.md`); producer `src/harsh_review_a.py` → `wp12_identity.csv`,
`wp12_mechanism.csv`, `wp12_scaling.csv`, `wp12_onset_bound.csv`, `wp12_adam_bound.csv`. This answers the review's
point that the mechanism is in the appendix but not the main text. The recommended placement is a short main-text
proposition (Step 1, Step 2 and the Proposition) with the proof in the appendix.

**What the loss rewards (the mechanism in one paragraph).** At small output scale the profiled loss is
log 2 − (s/4)Δμ + O(s²), so it rewards the **class-mean gap** Δμ. At large scale it is controlled by e^(−sG/2), so it
rewards the **worst-case gap** G. The threshold R_glob is where the conditional minimiser switches from the first
kind of solution to the second.

**Step 1 (identity).** On windows symmetric about 0, the linear part of f_a contributes nothing to Δμ, and
Δμ = sign(w₂)·a·sin b₁·D(w₁) with D(α) = E_O cos αx − E_I cos αx. The identity is checked to
3.3e-15 on the data points and
2.3e-14 on the continuous windows. The maximiser is
|w₁| = α* = 1.7913244 (data points; 1.7922917 on the continuous windows), with
sin b₁ = ±1, and it does not depend on a.
IDs: `A1 identity: max abs err, data points`; `A1 identity: max abs err, continuous windows`; `A1 alpha* (data)`; `A1 alpha* (continuous)`.

**Step 2 (placement needs a small first-layer weight).** G ≤ 2a − 2.8|w₁|, using the endpoint pairs (−2.0, 0.8) and
(−0.8, 2.0). These four points are window endpoints and data points, so the bound holds for the continuous gap G and
the data gap G_n alike. Placement therefore needs |w₁| < a/1.4. Since α* > a/1.4 for every a < 1.4α* =
2.50785, the class-mean maximiser is unplaced there: G = -3.66
at a = 1.30 and -3.45 at a = 1.50.
IDs: `A1 1.4 alpha*`; `A1 G at theta* (1.30)`; `A1 G at theta* (1.50)`.

**Proposition (an explicit analytic bracket for the switch, width 1).** For 1 < a < 1.4α* and s < s₀(a), every placed
parameter has a strictly higher profiled loss than the unplaced class-mean maximiser. For s > s₁(a), every unplaced
parameter has a strictly higher loss than Ĝ's placed witness. So the switch exists and lies in [s₀, s₁]. The proof
uses only convexity of the logistic loss, ℓ″ ≤ 1/4 and Step 2, plus four finite evaluations. No compactness and no
attainment are needed.

| a | s₀ | s₁ (data gap) | certified \|w₂\|_glob |
|---|---|---|---|
| 1.30 | 0.342 | 163.8 | [4.95, 4.963] |
| 1.35 | 0.327 | 131.9 | [4.025, 4.037] |
| 1.40 | 0.310 | 109.5 | [3.375, 3.388] |
| 1.45 | 0.293 | 93.1 | [2.888, 2.9] |
| 1.50 | 0.274 | 80.5 | [2.525, 2.538] |
| 1.60 | 0.236 | 62.9 | [2, 2.013] |

Every certified threshold lies inside the bracket. The bracket is loose, by roughly 15× below and 30× above: it
proves the switch **exists**, and the certificates **locate** it. In R units the upper end is R₁ = log(n/log 2) =
7.05 for every a.
IDs: `A1 s0(1.30)`; `A1 s0(1.50)`; `A1 s1_data(1.30)`; `A1 s1_data(1.50)`; `A1 R1 = log(n/log 2)`; `A1 every certified bracket inside [s0, s1]`; `A1 bound check at s0 (1.30): L*(theta*) below placed lower bound`.
The window-gap version of s₁ needs Ĝ > η (a Lipschitz correction). That holds at every tabulated a except 1.02
(`A1 window s1 defined for a >= 1.05`).

**Width 2, in the same terms.** On symmetric windows, the second-order term selects the pair ṽ₁α₁ = −ṽ₂α₂. It
cancels the ramp and leaves a pure cosine, which is placed. So the class-mean maximiser is already placed and there
is no switch (WP-9). The same criterion predicts a switch on **asymmetric** windows, where the ramp enters Δμ. That
prediction was registered and passed: a validated width-2 threshold exists there. The registered training prediction
failed: training crosses at about 3× the threshold scale (WP-15).

**The scaling law (A2).** |w₂|_glob(a) = A*·ε^(−3/2)·(1 + 0.66215ε + O(ε²)), with ε = a − 1. A* ∈ [0.68125, 0.6875]
is certified and independently checked (Krawczyk value 0.6854452); the correction A′(0)/A* comes from math note §8.

| a | limit A*ε^(−3/2) | with first-order term | certified \|w₂\|_glob | first order vs certified |
|---|---|---|---|---|
| 1.02 | 242.3 | 245.6 | not certified | — |
| 1.05 | 61.31 | 63.34 | not certified | — |
| 1.10 | 21.68 | 23.11 | not certified | — |
| 1.15 | 11.8 | 12.97 | not certified | — |
| 1.25 | 5.484 | 6.391 | not certified | — |
| 1.30 | 4.171 | 5 | [4.95, 4.963] | +0.9% |
| 1.35 | 3.31 | 4.078 | [4.025, 4.037] | +1.1% |
| 1.40 | 2.709 | 3.427 | [3.375, 3.388] | +1.4% |
| 1.45 | 2.271 | 2.947 | [2.888, 2.9] | +1.8% |
| 1.50 | 1.939 | 2.581 | [2.525, 2.538] | +1.9% |
| 1.60 | 1.475 | 2.061 | [2, 2.013] | +2.7% |

IDs: `A2 w2 first order (1.02)`; `A2 w2 limit (1.02)`; `A2 first order vs certified, max rel`; `A2 first order vs certified, min rel`.

**Adam's per-step bound (rigorous, for every gradient sequence).** Take torch Adam with lr = 0.01 and
(β₁, β₂) = (0.9, 0.999), bias-corrected. By Cauchy–Schwarz, every coordinate moves at most lr·B_t per step, with
B_t ≤ B_∞ = 7.2703. The bound is attained by a geometric gradient sequence. Summed, |w₂| ≤ 106.93
after 2,000 steps from |w₂(0)| ≤ 1. At a = 1.02 the required scale is 245.6 (≥ 240.9
even at A*'s certified lower end). So any Adam run needs **at least 3,966 steps**, and
about 24,456 at the typical rate of lr per step. Observed at 2,000 steps: 0/200
solves, with terminal |w₂| median 1.85 and maximum 3.92.
IDs: `A2 B_inf`; `A2 reachable |w2| at 2000 steps`; `A2 bound attained (t = 400)`; `A2 N_min at 1.02`; `A2 N at lr per step, 1.02`; `A2 a=1.02: solved`; `A2 a=1.02: median terminal |w2|`; `A2 a=1.02: max terminal |w2|`.

**What the law explains and what it does not.**
- **It explains** why the required scale diverges as ε^(−3/2), with a certified constant and first-order term (within
  3% of every certified finite-a threshold). It also shows that, with a bounded per-step move, the number of steps must
  diverge at least as fast.
- **It explains** the 0/200 at a = 1.02 in 2,000 steps. The threshold scale cannot be reached, whatever the gradients.
  This makes the 0/200 *consistent with and forced by* the threshold-crossing picture; it is not independent evidence
  for that picture.
- **It does not explain** Adam's actual growth. The worst-case bound is loose by one to two orders of magnitude. At
  2,000 steps it permits every a ≥ 1.035, but the observed onset is a = 1.60. At 128,000
  steps it permits a ≥ 1.0018; observed 1.03. The observed onset follows the *measured*
  growth |w₂| ∝ B^α, α = 1.1172, and α's derivation failed (SGD). So −0.745 (registered) against −0.734 (measured)
  tests the derived ε^(−3/2) combined with a measured α; it is not a derivation of α.
- **It does not explain** the residual (why crossings sit above the conditional threshold).
IDs: `A2 worst-case a_min at 2000`; `A2 lr-rate a_min at 2000`; `A2 worst-case a_min at 128000`.

**Say:** "At small output scale the loss rewards the class-mean gap, whose maximiser is provably unplaced for
a < 2.51; at large scale it rewards the worst-case gap, whose maximiser is placed. The conditional threshold is where
the minimiser switches, and its existence follows analytically; its location is certified." / "Adam's per-step move
is bounded (attainably) by 7.27·lr, so at a = 1.02 the conditional threshold scale cannot be reached within 2,000
steps."

**Do not say:** "we derive Adam's growth rate" or "the scaling law predicts the onset" (α is measured). Do not say
"the per-step bound explains the delay quantitatively" (it is loose by one to two orders of magnitude). Do not say
"the proposition locates the threshold" (the bracket spans more than two decades). Do not say "the 0/200 at a = 1.02
confirms the mechanism" (it is forced by it).

## WP-13. Related work: the reviewer's comparators (for the submission)

How these were verified (2026-09-25): each field was read from the proceedings page (NeurIPS proceedings, PMLR,
proceedings.iclr.cc) and from arXiv, using the pages' own citation metadata and official BibTeX. **Cite the
proceedings metadata, not arXiv's.** They differ as follows:
- Ahn et al.: the title differs.
- Kumar et al., Barak et al., Prieto et al.: middle initials are dropped in the proceedings.
- Pesme and Flammarion, Refinetti et al., Prieto et al.: the abstracts differ.

openreview.net refused every automated request, so no OpenReview decision string is verified.

**Six verified on proceedings pages.** Each has one sentence on what it shares with this paper and how it differs, written from the
paper's own abstract.

- **Ahn et al. (NeurIPS 2023), `ahn2023threshold`.** They prove, for gradient descent on simplified two-layer models,
  a sharp step-size transition below which the network fails to learn threshold neurons (non-zero first-layer bias).
  Shared: a sharp transition decides whether a unit acquires a useful bias/placement. Differs: their control
  parameter is the learning rate, acting through edge-of-stability dynamics. Ours is output scale, and our threshold
  is a property of the loss at fixed scale, certified independently of any trajectory.
- **Kumar et al. (ICLR 2024), `kumar2024grokking`.** Grokking arises from a transition from lazy to rich
  (feature-learning) dynamics, whose rate is controlled by the parameters that scale the network output. Shared:
  output scale governs when features are learned. Differs: they study the rate of feature learning along gradient
  descent on polynomial regression. We locate a scale at which the preferred solution of the conditional loss changes,
  a static property of the loss.
- **Pesme and Flammarion (NeurIPS 2023), `pesme2023saddle`.** They prove that gradient flow on diagonal linear networks
  from vanishing initialisation jumps from saddle to saddle, with the saddles and jump times given exactly. Shared: an
  exact account of discrete transitions in a small model. Differs: their transitions happen in time along the flow;
  ours are in output scale on the conditional loss, and training's relation to them is measured, not derived.
- **Refinetti et al. (ICML 2023), `refinetti2023neural`.** Networks trained with SGD first classify with lower-order
  input statistics (mean, covariance) and use higher-order statistics later. Shared: a low-order statistic comes
  first. Our small-scale lemma (WP-12) shows the conditional minimiser at small scale maximises the class-mean gap, and
  at large scale the worst-case gap. Differs: their ordering is in training time and over input statistics; ours is in
  output scale, for a fixed finite objective, and is proved.
- **Barak et al. (NeurIPS 2022), `barak2022hidden`.** Learning sparse parities shows abrupt transitions at about
  n^O(k) iterations. SGD makes continual progress through a Fourier gap that loss and error do not show. Shared: an
  abrupt change in training preceded by steady growth in a hidden variable (for us, the output scale grows before
  placement). Differs: their delay is computational; ours is a scale threshold of the loss, with a certified location.
- **Prieto et al. (ICLR 2025), `prieto2025grokking`.** Without regularisation, gradients align with a direction that
  scales the logits without changing predictions. That scaling delays generalisation and ends in Softmax Collapse.
  Shared: logit-scale growth as the variable behind a delayed transition. Differs: in their account the growth
  delays generalisation and ends in floating-point failure. In ours, output-scale growth is what makes placement the
  preferred solution, and the transition occurs when the scale crosses a certified threshold.

**Two verified against the official ICLR 2023 conference pages (iclr.cc).** The author accepted iclr.cc as the source
on 2026-09-25; OpenReview was unreachable, and proceedings.iclr.cc has no 2023 volume. Every field below matches iclr.cc.
- **Liu, Michaud and Tegmark (ICLR 2023), `liu2023omnigrok`.** Grokking is explained by the "LU mechanism": training
  and test losses plotted against weight norm look like "L" and "U". Weight norm, controlled by initialisation and
  weight decay, then decides whether grokking occurs. Shared: a norm/scale variable organises a qualitative change.
  Differs: their variable indexes a train–test mismatch in generalisation. Ours indexes which minimiser of the
  training loss itself is placed at fixed scale.
- **Chiang et al. (ICLR 2023), `chiang2023loss`.** Gradient-free optimisers, including guess-and-check (sample random
  parameters until training accuracy is perfect), reach test accuracy comparable to SGD in low-sample regimes. So the
  implicit-regularisation behaviour is largely independent of the optimiser. Shared: attributing an outcome to the
  loss landscape rather than to the optimiser; our threshold is a property of the loss. Differs: they study
  generalisation; we study which training-loss minimiser is placed at a given output scale.

**Do not say** that any of these papers studies a conditional threshold in output scale. Do not cite an OpenReview
decision ("oral", "poster") from these notes; iclr.cc lists the two 2023 papers as "Oral presentation / top 25%", and
that is the only source checked.

BibTeX (all eight; the two 2023 ICLR entries are verified on iclr.cc):
```bibtex
% verified: https://proceedings.neurips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html
@inproceedings{ahn2023threshold,
  title     = {Learning threshold neurons via edge of stability},
  author    = {Ahn, Kwangjun and Bubeck, Sebastien and Chewi, Sinho and Lee, Yin Tat and Suarez, Felipe and Zhang, Yi},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {19540--19569},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/075280-0858},
  year      = {2023}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/63ed15a46a143ff57484b38cd6b85d91-Abstract-Conference.html
@inproceedings{kumar2024grokking,
  title     = {Grokking as the transition from lazy to rich training dynamics},
  author    = {Kumar, Tanishq and Bordelon, Blake and Gershman, Samuel and Pehlevan, Cengiz},
  booktitle = {International Conference on Learning Representations},
  pages     = {23010--23035},
  year      = {2024}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html
@inproceedings{pesme2023saddle,
  title     = {Saddle-to-Saddle Dynamics in Diagonal Linear Networks},
  author    = {Pesme, Scott and Flammarion, Nicolas},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {7475--7505},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/075280-0329},
  year      = {2023}
}

% verified: https://proceedings.mlr.press/v202/refinetti23a.html
@inproceedings{refinetti2023neural,
  title     = {Neural networks trained with {SGD} learn distributions of increasing complexity},
  author    = {Refinetti, Maria and Ingrosso, Alessandro and Goldt, Sebastian},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {28843--28863},
  publisher = {PMLR},
  year      = {2023}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html
@inproceedings{barak2022hidden,
  title     = {Hidden Progress in Deep Learning: {SGD} Learns Parities Near the Computational Limit},
  author    = {Barak, Boaz and Edelman, Benjamin and Goel, Surbhi and Kakade, Sham and Malach, Eran and Zhang, Cyril},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {21750--21764},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/068431-1581},
  year      = {2022}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9e6ac15e689e06139d7b39e1667b165-Abstract-Conference.html
@inproceedings{prieto2025grokking,
  title     = {Grokking at the Edge of Numerical Stability},
  author    = {Prieto, Lucas and Barsbey, Melih and Mediano, Pedro and Birdal, Tolga},
  booktitle = {International Conference on Learning Representations},
  pages     = {81151--81168},
  year      = {2025}
}

% verified: https://iclr.cc/virtual/2023/oral/12716 (OpenReview unreachable; arXiv 2210.01117 lists "Eric J. Michaud")
@inproceedings{liu2023omnigrok,
  title     = {Omnigrok: Grokking Beyond Algorithmic Data},
  author    = {Liu, Ziming and Michaud, Eric and Tegmark, Max},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

% verified: https://iclr.cc/virtual/2023/oral/12746 (OpenReview forum QC10RmRbZy9 unreachable; no arXiv version)
@inproceedings{chiang2023loss,
  title     = {Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent},
  author    = {Chiang, Ping-yeh and Ni, Renkun and Miller, David Y. and Bansal, Arpit and Geiping, Jonas and Goldblum, Micah and Goldstein, Tom},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

## WP-14. The census sorted by relevance (for the submission; POST HOC classification)

Producer: `src/census_relevance.py` → `census_relevance.csv`, `census_relevance_counts.csv`. The rule was fixed by
**block topic, never by outcome**, and ties were resolved toward *central*, so that no failure is hidden as
peripheral.
- **Central:** the block's predictions concern the conditional threshold (its value, what it predicts about training
  crossings, the residual), the scaling reduction (κ, K, A*, c₁, the scaling limit, cross-family), the prospective
  held-out predictions (Block G, Block 3, own-seed), or width 2.
- **Peripheral:** budget laws, barriers and sharpness, trapping, optimiser equivalence, and the early exploratory
  probes.
- The classification is post hoc: it was made on 2026-09-25, after every verdict was known.

| scoring | relevance | PASS | FAIL | PARTIAL | UNRESOLVED |
|---|---|---|---|---|---|
| post hoc (census) | central | 1 | 2 | 13 | 0 |
| post hoc (census) | peripheral | 2 | 4 | 0 | 0 |
| registered rule | central | 50 | 30 | 1 | 11 |
| registered rule | peripheral | 47 | 35 | 7 | 25 |

IDs: `A4 FAIL registered central`; `A4 FAIL registered peripheral`; `A4 FAIL post hoc central`; `A4 FAIL post hoc peripheral`; `A4 PASS registered central`; `A4 PASS registered peripheral`; `A4 rows`.

**Failed central predictions (32: 30 under the registered rule, 2 by post hoc scoring), with one line each:**

*threshold* (22):
- `H-3`: growth-rate dose-response validity gate failed (m = 1 placed 0.500 vs control 0.925); block inconclusive.
- `F-2`: offsets scaling with the optimiser's rate ratio: SGD misses by 6.19 pp.
- `rc-H`: 'R is the controlling variable' fails across activation families (2 of 4 failure conditions): R is a correlate there.
- `P1-b`: median solved rho predicted in [0.5, 0.95]; failed as registered.
- `p2b-Hreverse`: median R at crossing 0.2330 against the registered < 0.20.
- `p2bA-Rrho`: coefficient-of-variation comparison across a (R 0.083 vs R*rho 0.193) went against the prediction.
- `B-spin`: (post hoc scoring) the registered falsifier of the spinodal / hysteresis picture fired at a = 1.30; the framing was dropped.
- `B-solve`: (post hoc scoring) R at solve exceeds R_solve by 16-29%, against a 15% tolerance.
- `B4-D2`: the competing saturation outcome at fixed scale failed (favourable to D1).
- `B4h-Q1`: x50 did not move toward 1.0 with a longer horizon (1.046 / 1.045 / 1.045); the registered competing outcome, a persistent offset, held.
- `B4h-Q2`: the placed fraction below threshold did not vanish at 64k steps (0.015 at 0.9, 0.094 at 0.95).
- `own-S1`: own-threshold rule agreement 0.894 < 0.95 (its +13.4 pp over the population rule passes).
- `size-N3`: the own-sample excess is below half the free offset at some n (a = 1.30 at n = 6,400; a = 1.50 at all n).
- `lag-L2`: residual proportional to rate: ratio 0.858 (1.30) and 0.815 (1.50), outside [0.30, 0.70].
- `lag2-L1'`: deconfounded lag test: the residual depends on rate too weakly (ratio 0.894 / 0.859 > 0.5).
- `lag2-L2'`: deconfounded proportionality fails (0.974 / 0.958).
- `4b-ID`: inherited displacement: neither teleport nor reset removes half the residual (the competing outcome held).
- `4b-OM`: optimiser memory: likewise, neither arm removes half the residual.
- `ramp-R5-R1`: Track 1B ramp, winding sign test (a = 1.30, k = 0): the pooled through-origin slope of r on kappa*chi is 1.3003 (SGD; bound 1.3) and 9.1 (Adam); R2 and R3 pass and the crossings do come early as predicted.
- `act-Ta-200`: GELU/SiLU/Mish, 200 seeds: fewer than 90% of crossings at or above the validated switch (0.48 / 0.62 / 0.70).
- `act-Tb-200`: GELU/SiLU/Mish, 200 seeds: the median residual against the population threshold is not within tolerance of kappa*chi (wrong sign or size).
- `band-P2a-120`: band task in R^d, 120 seeds: the median residual against the population threshold lies above the width-1 range at d = 2 and d = 4 (0.19-0.55).

*scaling* (4):
- `X1`: cross-family: q2 matches family A on R_glob at 6/6 a but on R_solve at 0/6, so the joint prediction fails (the solve threshold does not transfer).
- `K-2b`: numeric sub-claim 'roughly 115-145x' was an arithmetic error (actual 11.8-145x).
- `K-3`: registered margin >= 0.10 not met (0.0089); the claim was removed.
- `C-2 (kappa)`: kappa's magnitude clause passes (0.69-0.76%), its containment clause fails; FAIL under the conjunctive falsifier.

*prospective* (2):
- `G-1`: training-free out-of-distribution windows: 6 of 10 within 15%, short of the registered count.
- `G-3b`: sub-claim (far-outer window threshold >= base) was wrong by a provable inequality: a registration error.

*width2* (4):
- `sl-tanh`: tanh at width 2: the boundary condition failed at A = 40 (criterion flaw); recorded 'neither'.
- `T2-3`: asymmetric windows, width 2: training crosses above the validated threshold (93.7%) but at a median 3.29x the threshold scale, above the registered bound of 1.25.
- `nogating`: width 2, symmetric windows, fixed-scale training: placement is gated at small held scale (placed 0.46 at R2 = 0.003, rising to 0.98-0.99 at 0.1, both a), against the predicted no gating.
- `T2-3d`: slowed regime, fresh seeds: the width-1 own-sample threshold has median |log err| 0.019 but does not beat the population width-2 threshold (paired interval [-1.04, +0.006]); crossings bimodal.

**The pattern, stated plainly.** The central failures cluster in the **training-side** claims, and most of them
concern the **residual**, i.e. where crossings sit relative to the threshold and why:
- the lag tests;
- Block 4b;
- the horizon extension;
- own-S1;
- sample size.

The rest are:
- sub-claims of the scaling reduction (two of them arithmetic or registration errors);
- the solve-threshold transfer (X1);
- tanh at width 2.

Tonight's program (2026-09-25) added four central failures, all training-side:
- the ramp's winding sign test (R5-R1; pooled slope 1.3003 against a bound of 1.3);
- the two 200-seed training tests for GELU, SiLU and Mish;
- the R^d band task's width-1 lag range (120 seeds).

The final round added one central PASS: the registered c₁ follow-up (WP-30), whose interval of width 0.042 contains
the derived c₁. The original c₁ test stays INCONCLUSIVE.

It also added seven PARTIAL rows, assigned post hoc because verdicts differ across units: the ramp's R1–R3 (across a
and optimiser) and the band task's primary P1, P2a and P2b (across d). Designs that failed their own rules before
registration (2C, the GELU prospective test, the Adam ramp from initialisation) are not registrations and are not in the
census.

None is a failure of the certified threshold values themselves, and the primary prospective comparisons (Block 3,
own-seed primary) passed.

**Say:** "Of 71 failed predictions (65 under the registered rule, 6 by post hoc scoring), 32 bear on a central
claim. Most of these concern the residual and training's tracking of the threshold (lag tests, Block 4b, the ramp's sign
test, training outside the sine family, the lag in R^d); others are the width-2 training predictions (WP-15, WP-20)." **Do not say** "the central claims never failed" or "the failures are
peripheral". Do not present the relevance classification as registered;
it is post hoc.

## WP-15. Width 2 on asymmetric windows: the landscape threshold survives; training crosses far above it (Track 2, registered; for the submission)

Exploratory pilot (Δ = 0.8, a = 1.30): `asym_pilot_design.md`, with GO/no-go fixed before running (374b86a, GO
129a487). Registration: `asym_registration.md` (53dce22), with amendment 1 (2505b2d, matched initialisation, the
author's instruction). Producer: `src/asym_register.py` → `asym_parts/`, `asym_scores.json`. Windows:
I = [−0.8, 0.8], O = [−2.0, −1.2] ∪ [1.2, 2.4], so Δ = 0.4. This answers "the width-2 analysis contains no threshold
test where one is predicted".

**Why a threshold was predicted.** On asymmetric windows the linear ramp enters the class-mean gap (m = E_O x − E_I x
= 0.44). So the small-scale conditional minimiser uses the ramp and is unplaced, and the criterion (WP-12, math note
§10.1) predicts a switch. On symmetric windows it predicts none, and none was found (WP-9).

| prediction | criterion | result |
|---|---|---|
| **T2-1** (landscape) | a validated width-2 search finds exactly one switch (competing: placed at every scale) | **PASS**: unplaced at s ≤ 0.4217, placed at s ≥ 0.5623 on a 17-scale grid; bisection bracket **s ∈ [0.4371, 0.4532]**, validated at both ends (restart ladder 500–4,000 unchanged; independent CMA-ES agrees; audit clean) |
| **T2-2** (secondary) | s_hi(Δ = 0.4) < 0.5623 (the pilot's lower end at Δ = 0.8) | **PASS** (0.4532) |
| **T2-3** (training, W1's criterion) | ≥ 90% of crossings at or above s_hi; bootstrap CI of the median ratio − 1 above 0; **median(s_cross/s_lo) ≤ 1.25** | **FAIL**: 93.7% at or above s_hi; CI [1.89, 2.56]; **median ratio 3.29 > 1.25** |

- T2-3 used matched initialisation, seeds 600,000–600,079: 1 run placed at initialisation
  (excluded; under the 20% stop), and 79 crossed.
- **Disclosed:** a first T2-3 arm with standard initialisation ran before amendment 1. Its outputs were sealed unread
  (hashes committed), and it is withdrawn and unscored.
IDs: `Track 2 T2-1 PASS`; `Track 2 s_lo`; `Track 2 s_hi`; `Track 2 T2-2 PASS`; `Track 2 T2-3 FAIL`; `Track 2 T2-3 frac above`; `Track 2 T2-3 median ratio`.

**Reading.**
- The criterion's landscape prediction holds at width 2: where the small-scale class-mean maximiser is unplaced, a
  placement threshold exists, validated like W0.
- Training does cross above that threshold (94% of runs), but at a median of about **3.3×** the threshold scale, not
  within the registered 25%. So at width 2 the landscape threshold does not predict *where* training crosses, as the
  width-1 threshold does (within 3–7%).


**POST HOC, EXPLORATORY (the author's request, after T2-3 failed): what the runs cross on.** Producer:
`src/asym_posthoc.py` → `asym_posthoc/`. The definitions were fixed before running: knockout class at the crossing;
weight shares along the trajectory; each run's branch switch scale by continuation from its crossing configuration;
the explicit single-unit branch; and the width-1 threshold for these windows. All 79 replays reproduce their
crossings exactly.
- **No run crosses on a single-unit branch.** At the crossing, 76 of 79 are a cooperating pair (neither unit
  alone is placed) and 3 are redundant; **0 are single-unit**. Before the crossing, the median run
  has a dominant unit (share ≥ 0.75) at only 19% of its logged steps.
- **The single-unit branch** (the width-1 minimiser embedded in width 2, with 99% of the weight on one unit) switches
  at s ∈ (4.87, 5.62]. The width-1 threshold for these windows
  is s = 5.09. **97.5% of crossings lie below it**: median crossing
  s = 1.44 (10–90%: 0.77–2.42), i.e. 0.29× the width-1
  threshold.
- **The branch the runs actually cross on** (continuation from each crossing configuration) switches at a median
  s = 0.439, essentially the width-2 global threshold (0.445). The runs cross at
  3.2× their own branch's switch.
- **No reference tracks the crossings.** Mean |log error|: branch 1.11, width-2 global
  1.19, width-1 1.27. Branch − global is
  -0.077 [-0.196, +0.015] and width-1 − global is
  +0.081 [-0.140, +0.301]; both intervals contain 0.
  Spearman(crossing, branch switch) = 0.30.
IDs: `Track 2 posthoc single-unit at crossing`; `Track 2 posthoc pair at crossing`; `Track 2 posthoc single branch switch lo`; `Track 2 posthoc single branch switch hi`; `Track 2 posthoc width-1 threshold`; `Track 2 posthoc frac below single branch`; `Track 2 posthoc median branch switch`; `Track 2 posthoc branch minus glob hi`; `Track 2 posthoc Spearman branch`.

**Say (post hoc, beside the failure):** "The registered training prediction failed. Post hoc, the runs cross as
cooperating pairs, on the branch whose switch is the width-2 threshold, but about three times above it, and below the
single-unit (width-1) threshold. Neither threshold, nor the branch's own switch, predicts the crossing scale."

**Do not say:** that the failure is explained by a single-unit branch (no run is single-unit at its crossing), that
crossings "track" any of these thresholds, or that the post hoc analysis rescues T2-3. It is exploratory, and the
registered verdict is FAIL.


**T2-3b (registered after T2-3's failure; amendment 2, 61f3353): T2-3 plus the approved sweep-rate matching.**
- When T2-3 was amended, only matched initialisation was specified. The approved design's sweep-rate matching
  (Revision 3) was left out, and T2-3b adds it.
- The output learning-rate factor φ₂ = 0.0093 came from the steps-matching pilot (calibration seeds,
  ‖w₂‖₁ only). The validity check requires the achieved median timescale ratio at crossing to lie in the width-1 Adam
  range [0.0014, 0.023].
- **Result: UNRESOLVED.** The achieved median ratio is 6.8e-06, far *below* the range. Steps
  matching over-slows the scale's growth at the crossing, so the two notions of rate matching disagree by about three
  orders of magnitude.
- For information only, the criteria alone would give FAIL: 71/80 crossed,
  56.3% at or above s_hi, median ratio 10.4, and a
  bootstrap interval [-0.41, 9.52] that contains 0.
- **T2-3 stays FAIL.**
IDs: `Track 2 T2-3b UNRESOLVED`; `Track 2 T2-3b phi2`; `Track 2 T2-3b median ratio at crossing`; `Track 2 T2-3b frac above`; `Track 2 T2-3b median crossing ratio`.

**EXPLORATORY note (author's request; changes nothing above).**
- *Training data.* The T2-3 runs trained on their own sampled sets, not on the population objective.
- *Own-sample thresholds.* Own-sample width-2 thresholds for 10 seeds (200 restarts) have median
  s = 0.445 (range 0.346–0.573), the population value.
  - These seeds cross at 4.1× their own threshold, against 4.3× the
    population's.
  - Mean |log error| is 1.52 against own and 1.51 against the population;
    5/10 runs are closer to their own; Spearman(crossing, own) = -0.74 (n = 10).
  - So own-sample thresholds do not explain the width-2 residual.
- *Timescale ratio.* At the T2-3 crossings the median ratio is 2.73, about 100× beyond the width-1
  range. The width-1 fit, extrapolated, predicts a residual of 7.3 against
  2.23 observed. Within T2-3, Spearman(residual, ratio) = -0.24.
IDs: `Track 2 expl own median`; `Track 2 expl cross over own`; `Track 2 expl Spearman own`; `Track 2 expl ratio median`; `Track 2 expl ratio Spearman within`.

**T2-3c (registered after T2-3's FAIL and T2-3b's UNRESOLVED; amendment 3, 9ba85f5): timescale-ratio matching.**
- φ₂ = 0.01778 was chosen by a pilot on fresh calibration seeds. The pilot recorded only the timescale ratio
  at the step where ‖v‖₁ first reaches the threshold scale (median 0.0049, within 20% of 0.0057),
  and never evaluated placement.
- **Result: UNRESOLVED.** The achieved median ratio **at crossing** is 1.3e-05, far below the
  width-1 range. The risk stated in the registration materialised: the runs cross only much later (median
  10.8× s_lo), after the scale has nearly stopped growing.
- For information only, the criteria alone would give FAIL: 80/80 crossed,
  47/80 at or above s_hi, median ratio 10.8.
IDs: `Track 2 T2-3c UNRESOLVED`; `Track 2 T2-3c phi2`; `Track 2 T2-3c median ratio at crossing`; `Track 2 T2-3c frac above`; `Track 2 T2-3c median crossing ratio`.

**EXPLORATORY diagnosis (item 2; POST HOC; gate STOP, so no further test and no change to this section's claims).**
The rules and the gate were committed before any result (`width2_diagnosis_gate.md`, 4b7826e). Producer:
`src/width2_diagnosis.py` → `width2_diagnosis/`.
- **Predictors of each run's crossing scale:** the population width-2 threshold; the own-sample width-2 threshold; the
  own-sample width-1 threshold; the population width-1 threshold (s = 5.09); and the switch of the branch reached from
  the run's own crossing configuration.
- **Selected (lowest pooled error): the branch switch.** Median |log error| is 0.010 (T2-3b) and
  0.006 (T2-3c), but **1.17 (T2-3)**, so the gate (≤ 0.10 in every φ₂ setting) STOPS.
- In the two slowed arms, the **width-1 own-sample threshold on the same training set** also matches the crossings closely:
  median |log error| 0.015 and 0.011 (10 seeds each). Nothing tracks the full-speed T2-3
  crossings.
- Caveats:
  - the branch switch is computed from the crossing configuration itself, so it is close to circular;
  - it is undefined for 41 slowed-arm runs;
  - the own-sample predictors rest on 10 seeds per arm.
IDs: `Item 2 gate STOP`; `Item 2 selected P5`; `Item 2 P5 T2-3 error`; `Item 2 P3 T2-3b error`; `Item 2 P3 T2-3c error`.
**Do not say** that any predictor explains where width-2 training crosses, or that width-2 crossings follow the
width-1 threshold. This is a post hoc, 10-seed observation in the slowed arms only, and it failed the gate.

**T2-3d (registered as designed after the item 2 diagnosis; amendment 4, 394ebcc): the slowed regime only.**
- φ₂ = 0.01778 (T2-3c's), fresh seeds 600,480–600,559.
- Each seed's width-1 own-sample threshold was frozen with a hash before any training (68323b2).
- Criteria: (i) median |log error| ≤ 0.10, and (ii) it beats the population width-2 threshold (paired interval below 0).
- **Result: FAIL.**
  - (i) holds: 0.019. The median crossing is 0.98× the width-1 own
    threshold.
  - (ii) fails: [-1.04, +0.0064] contains 0.
- The crossings are bimodal: 47 of 79 runs cross at their width-1 own threshold, and
  31 cross early at small scale (s = 0.016–0.42).
- **Scope:** T2-3d covers the slowed regime only. The full-speed regime (T2-3) remains unexplained.
IDs: `Track 2 T2-3d FAIL`; `Track 2 T2-3d median err`; `Track 2 T2-3d diff hi`; `Track 2 T2-3d bimodal`.
**Say (T2-3d):** "A registered test designed after the diagnosis found that, in the slowed regime, the median run crosses at
its own width-1 threshold, but about 40% of runs cross early at small scale, so the registered comparison with the
width-2 threshold fails. The full-speed regime remains unexplained."
**Do not say** that the width-1 threshold predicts width-2 crossings, or that T2-3d rescues T2-3, T2-3b or T2-3c.

**Say (T2-3, T2-3b and T2-3c, in order):** "The registered width-2 training prediction on asymmetric windows, T2-3,
failed: runs crossed at a median 3.3× the threshold. Two follow-ups were registered after that failure, each labelled as
such, to put width-2 training in width 1's timescale regime. T2-3b (steps-matched output rate) is unresolved: its runs
crossed at a timescale ratio far below width 1's range. T2-3c (ratio-matched at the threshold scale) is unresolved for
the same reason: the ratio at crossing fell far below the range. Neither follow-up tested the prediction, and T2-3 stands
as a failure."

**Do not say:** that T2-3c (or T2-3b) is a retry of a failed test, that either rescues or qualifies T2-3's failure,
that width-2 training "would" track the threshold in width 1's regime (never achieved), or that own-sample thresholds
or the timescale ratio explain the width-2 residual.

**Say:** "On asymmetric windows, where the criterion predicts a switch, a validated width-2 placement threshold exists
(registered). Width-2 training crosses above it, but at about three times the threshold scale, so the registered
training prediction fails."

**Do not say** that the width-2 threshold predicts training crossings. Do not cite the Δ = 0.8 pilot as evidence:
it is exploratory.

## WP-16. Does the conditional threshold predict SGD crossings? (Track 4, registered; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Registration: `sgd_own_registration.md` (0506650); budget fixed before any registered run (988c1d7); amendment 1,
the timescale extension (1df48d1, 13:13 EDT), registered before any SGD run was scored. Producer: `src/sgd_own.py` →
`sgd_own_runs.csv`, `sgd_own_scores.csv`, `sgd_own_ratios.csv`, `sgd_own_extension_scores.csv`. This answers "does the
conditional threshold predict SGD crossings, or only the a = 1.25 midpoint?"

**Design.**
- SGD at lr 0.3 (the only rate that solves in this setting), full batch.
- Exactly the phase-2b training sets (seeds 0–39, a = 1.30 and 1.50, n = 400), whose own-sample global thresholds were
  already computed. So there is no new threshold computation.
- The same initialisation as the Adam runs of each seed, and every-step crossing detection.
- Budget 32,000 steps, fixed by a rule on calibration seeds that recorded only |w₂| growth.

| a | crossed | G1: mean(e_own − e_pop) [95% CI] | G2: Spearman(cross, own) | residual vs own | vs population |
|---|---|---|---|---|---|
| 1.30 | 30/40 | -0.0459 [-0.0738, -0.0172] **PASS** | 0.674 **PASS** | +2.95% | +9.45% |
| 1.50 | 30/40 | -0.0439 [-0.0727, -0.0148] **PASS** | 0.684 **PASS** | +6.43% | +12.74% |

- **G1 passes at both a.** Each run's own threshold predicts its SGD crossing better than the population threshold.
- **G2 passes at both a.**
- The SGD residuals against the own threshold, +2.95% and +6.43%, are close to Adam's in the deconfounded lag test
  (+3.1% and +6.6% at φ = 1).
- **Caveat.** The budget rule aimed at ≥ 90% crossing; 75% crossed at each a (30/40), which is exactly the registered
  minimum of 30. Report the crossing fraction.
IDs: `Track 4 G1 a=1.30`; `Track 4 G1 a=1.50`; `Track 4 G2 a=1.30`; `Track 4 G2 a=1.50`; `Track 4 crossed a=1.30`; `Track 4 crossed a=1.50`; `Track 4 residual vs own a=1.30`; `Track 4 residual vs own a=1.50`.

**The residual's timescale (Track 3, POST HOC, exploratory).**
- For each of the 384 crossings of the deconfounded lag test (both a, four learning-rate arms), the run was replayed
  to its crossing. Every replay reproduces its crossing exactly.
- Two rates were measured at the crossing: the output scale's growth rate d log s/dt, and the branch's
  Adam-preconditioned relaxation rate, lr·λ_min(D^(−1/2) H D^(−1/2)).
- Spearman(residual, growth/relaxation) is **0.634 [0.542, 0.721]**
  pooled. Within each a it is **0.454** and **0.507**; much
  of the pooled value is the difference between the two a. Across the 8 cells it is 1.0.
- The growth rate varies only about 25% across an 8× range of w₂'s learning rate, because Adam's normalisation
  compensates. This matches L1′'s registered failure.
IDs: `Track 3 Spearman pooled`; `Track 3 Spearman CI lo`; `Track 3 Spearman CI hi`; `Track 3 Spearman within a=1.30`; `Track 3 Spearman within a=1.50`; `Track 3 replays reproduce`.

**The timescale account, tested prospectively on SGD (EXT, registered amendment).**
- The residual = α + β·ratio fit on Adam (α = 0.0157, β = 2.658) was frozen before scoring.
- Each SGD run's own ratio at its crossing was measured with the identity preconditioner.

| a | median SGD ratio | predicted residual | observed | tolerance | verdict |
|---|---|---|---|---|---|
| 1.30 | 0.00354 | 0.0251 | 0.0295 | ±0.0100 | **PASS** |
| 1.50 | 0.01404 | 0.0530 | 0.0643 | ±0.0133 | **PASS** |

- It passes at both a. At a = 1.50 the miss (0.0113) is close to the tolerance (0.0133).
- The SGD ratios lie inside the fitted Adam range, so this is interpolation across optimisers, not extrapolation.
IDs: `Track 4 EXT a=1.30`; `Track 4 EXT a=1.50`; `Track 4 EXT pred a=1.30`; `Track 4 EXT pred a=1.50`.

**Sensitivity analysis (POST HOC, author's request): non-crossers treated as crossing at their final |w₂|.**
- 25% of runs did not cross (10 per a). Here each is scored as crossing at (or above) its final |w₂|, a lower bound on
  its unobserved crossing. A replay confirms that none of them crosses within the budget.

| a | G1 [95% CI] | G2: Spearman | EXT: observed median vs predicted |
|---|---|---|---|
| 1.30 | [-0.056, -0.004] **PASS** | 0.502 **FAIL** | 0.0277 vs 0.0251 (±0.0100) **PASS** |
| 1.50 | [-0.056, -0.003] **PASS** | 0.507 **FAIL** | 0.0586 vs 0.0530 (±0.0133) **PASS** |

- **One verdict changes: G2 fails at both a under this imputation.** G1 and EXT are unchanged.
- The non-crossers ended at a median **0.15×** (a = 1.30) and
  **0.30×** (a = 1.50) of their own threshold. They stalled well
  below the threshold scale, so the imputed values are very loose lower bounds, not crossings.
IDs: `Track 4 sens G1 a=1.30`; `Track 4 sens G1 a=1.50`; `Track 4 sens G2 a=1.30`; `Track 4 sens G2 a=1.50`; `Track 4 sens EXT a=1.30`; `Track 4 sens EXT a=1.50`; `Track 4 sens noncrosser final/own a=1.30`; `Track 4 sens noncrosser final/own a=1.50`.

**How to describe the timescale account.**
- It is **supported by two prospective tests, not established**: the SGD extension (this section) and Task B (WP-21:
  Adam at a = 1.45 and 1.60, which the fit never saw, on fresh training sets).
- The Adam relationship is a post hoc correlation. Always give the within-a values beside the pooled one: pooled
  Spearman 0.63, but only 0.45 (a = 1.30) and
  0.51 (a = 1.50) within each a.
- Two registered prospective checks passed: EXT on a different optimiser (at a = 1.50 near the edge of its tolerance),
  and TS-1 at two unseen activation values (WP-21).

**Say:** "Trained with SGD on the same samples, each run's own conditional threshold predicted its crossing better
than the population threshold (registered; both a; 75% of runs crossed). A post hoc timescale ratio, correlated with
the residual on Adam runs (Spearman 0.63 pooled; 0.45 and 0.51 within each a), predicted the median SGD residual
within the registered tolerance, and a second registered test at unseen activation values also passed (WP-21). The
timescale account is supported by these two prospective tests; it is not established."

**Do not say:** that the timescale ratio is the residual's mechanism, or that the account is established (it did not
carry over to width 2: WP-15's exploratory note). Do not quote
the pooled Spearman without the within-a values. Do not say that SGD crossings were universal. Do not report G2 without
noting that it fails when the non-crossers are imputed. Do not merge these runs with Block F's a = 1.25 SGD arm.

## WP-17. Independent certificate checks: status after Track 5 (for the submission; replaces WP-11's "pending" list)

Checker: `src/verify_certificates.py` (python-flint / Arb, 80-bit balls; imports nothing from `src/`). Each item is
verified, failed, or not run. Tests exercise every new check on constructed pass and fail cases.

| certificate | status | what the checker did |
|---|---|---|
| K = sup G₀ over [0, 8] × [−12, 12] | **verified** | a fresh Arb branch and bound over the whole box: sup G₀ ≤ 0.5794951 (the published hi) and ≤ 0.5794559217 (the tight hi); the lower end is attained at the published argmax; the domain lemma is checked in exact rational arithmetic |
| c₁ Krawczyk boxes (switch and K vertex) | **verified** | switch: unique zero in the 1e−9 box, A* ∈ [0.6854452375756532, 0.6854452375756537], A′(0)/A* ∈ [0.6621547824344876, 0.6621549498240902], active set unique; K vertex unique, K = 0.579455883427557 |
| PD boxes, glob chain (500 boxes, U × [0.66, 0.71]) | **verified** | b* bracketed; the Hessian enclosure minus 0.0473·I is PD by Sylvester on every box |
| PD boxes, solve chain (100 boxes, × [1.05875, 1.06]) | **verified** | as above, with margin 0.0209 |
| ring (no critical point, 0.05–0.15), glob chain, 40 A-sub-intervals | **verified** | the envelope gradient excludes 0 in p or q on every ring box, with b* bracketed, uniformly over each A-sub-interval |
| ring, solve chain | **verified** | as above, over [1.05875, 1.06] |
| outer exclusion (K(24) minus the 0.15 box), both chains | not run | projected > 10 CPU-hours in Arb (about 39,000 cells per A-sub-interval before refinement, × 40 sub-intervals) |
| limit solve bracket (A_solve ∈ (1.05875, 1.06]) | not run | needs the solve-chain outer exclusion above |
| finite-a solve brackets (12 certificates) | not run | timing sample: one bracket end's search at the published 1e−11 tolerance took 19 min and produced 25.2 million losing-region leaves; at about 20 ms per leaf in Arb, that is several CPU-days per certificate |

**What is now fully independently verified.**
- **R_glob^∞ ∈ [0.19738, 0.19920]:** A*'s bracket (WP-11) and K's enclosure (above) are both independently checked.
- **A* and A′(0)/A* on the branch (Krawczyk), and K at the vertex.** The sharp value R_glob^∞ ∈ [0.1985926, 0.1985927]
  additionally needs the Krawczyk switch to be the *global* switch. That needs the uniqueness chain: localisation
  (verified, WP-11), outer exclusion (not run), ring (above) and PD (above). So the sharp value is **not yet fully
  independent**; the bracket is.

**Say (certificate status paragraph):** "An independent checker in ball arithmetic (Arb) re-verifies the finite-a
placement brackets, the Ĝ(a) enclosures, the limit switch bracket and its localisation, K = sup G₀ with its domain
lemma, the Krawczyk boxes of the first-order calculation, and the positive-definite and ring certificates of the
uniqueness chain. The outer-exclusion certificates and the solve brackets are verified by the original searches only."

**Do not say:** "all certificates are independently verified", or that the sharp R_glob^∞ = 0.19859 is independently
verified.

## WP-18. Stronger baselines (Track 6; POST HOC; for the submission)

Producer: `src/baselines_posthoc.py` → `baselines_posthoc*.csv`, `baselines_posthoc.md` (d7d88d8). Every baseline is fitted
only on the ten Block G settings, the data available before the held-out runs. The re-implementation first reproduces
the committed C, B1, B2, U and own-seed numbers (163 checks, largest difference 2.2e−16). This answers "B1 and B2 are
weak".

| model | what it uses | Block 3 mean \|log err\| | C − model [95% CI] |
|---|---|---|---|
| C (registered) | U × λ fitted on the base window | 0.0223 | — |
| PL | U × λ pooled over all five Block G windows | 0.0231 | −0.0007 [−0.0088, +0.0074]: **matches C** |
| PL5 | U × median of the five windows' λ | 0.0274 | −0.0051 [−0.0120, +0.0019]: not separated |
| B2 (registered) | pooled median crossing R | 0.0773 | −0.0550 [−0.1027, −0.0105] |
| RG | per-a regression on log Ĝ | 0.1047 | −0.0823 [−0.0926, −0.0722] |
| RK | regression on log κ₀ and a (no conditional threshold) | 0.1099 | −0.0875 [−0.0993, −0.0746] |
| B1 (registered) | base-window median \|w₂\| | 0.2796 | −0.2573 [−0.3704, −0.1312] |

- On the own-seed test, PL is not separated from C at a = 1.50. At a = 1.30, C is better by 0.001 (the interval
  just excludes 0).
- **Stated plainly: a stronger baseline (PL) matches C.** PL still uses the conditional threshold U; it differs from C
  only in fitting the lag factor on five windows instead of one.
- **Every baseline that does not use the conditional threshold loses to C by a factor of 4–5 in error, and also to
  B2.**
IDs: `Track 6 reproduction checks`; `Track 6 PL mean abs log err`; `Track 6 C - PL lo`; `Track 6 C - PL hi`; `Track 6 C - RK hi`.

**Say:** "Baselines that do not use the conditional threshold do 4–5 times worse than the registered predictor. A
variant that pools the lag factor over all calibration windows matches it (post hoc)."

**Do not say:** "C beats every baseline". Do not present PL as registered: it is post hoc.

## WP-19. H1 is proved for small ε (Track 8; for the submission)

Source: math note §12 (`math_note_for_writer.md`), every step checked by hand. The numerical sanity checks
(`src/h1_checks.py` → `h1_checks.log`) are separate from the proof. This answers "Proposition 2 also retains
hypothesis H1".

**Statement.** For 0 < ε ≤ 0.029 (a ≤ 1.029), Ĝ(a) > 0 is attained, and every maximiser has, up to the symmetries,
rescaled coordinates with 1.0155 ≤ p < 1.74964 and |q| < 1.7956, inside the compact set C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}.
Moreover |K(ε) − K| ≤ 23.3ε.

**Proof idea, for the text.**
- An exact identity: f_a(t_o) − f_a(t_i) = 2a cos(m) sin(d/2) − d. Applied to three pairs of window endpoints it
  gives G ≤ 0.4εw₁, that G > 0 forces 1.4w₁ below the root of sin x = x/a, and an ellipse bound on the bias.
- An exact rational evaluation, G₀(1.6, 1.2) = 27088/46875, bounds K(ε) from below.
- No numerical certificate is used.

**Consequence.** Proposition 2 is asymptotic ("for all sufficiently small ε"), so **H1 is no longer a hypothesis: it
is proved.**

**What remains certified rather than proved.**
- That the maximiser is the tied corner (the K(ε) = K_loc(ε) step of the c₁ calculation). This needs uniqueness of
  G₀'s maximiser, which K's branch and bound certifies; that check is now independently verified in Arb (WP-17).
- The argument does not cover a ≥ 1.30: there the q-bound gives 2.0008 > 2. At a = 1.02–1.25 the location in C
  follows from the proof's lemmas together with the certified Ĝ values.

**Say:** "Hypothesis H1 of Proposition 2 is proved for ε ≤ 0.029 (Appendix …)."
**Do not say:** "H1 is proved for all tested a", or "the corner is proved to be the global maximiser". The latter is
certified, not proved.

## WP-20. Does output scale gate placement in width-2 training? (Track 7, registered; for the submission)

Design and registration: `width2_nogating_design.md` (approved 2026-09-24, with amendments before any run). Horizon
H = 16,000, frozen with a hash from a pilot that cannot see the outcome (5922e24). Producer:
`src/width2_nogating.py` → `width2_nogating_scores.csv`. This answers "the width-2 analysis contains no width-2 training
test".

**Design.**
- Width 2, symmetric windows, a = 1.30 and 1.50.
- The latest pre-placement checkpoints of 80 training runs per a (matched initialisation) are replayed with ‖w₂‖₁
  held at R₂ ∈ {0.003, 0.01, 0.03, 0.1}, far below width 1's switch. The primary arm preserves the optimiser state.
- The outcome is placement at H, from exact extrema.
- The landscape verdict (WP-9) says the conditional minimiser is placed at every scale, so the prediction was
  **no gating**.

| arm | placed fraction at R₂ = 0.003 / 0.01 / 0.03 / 0.1 |
|---|---|
| a = 1.30, preserved (primary) | 0.46 / 0.50 / 0.68 / 0.99 |
| a = 1.50, preserved (primary) | 0.46 / 0.49 / 0.64 / 0.97 |
| a = 1.30, reset | 0.44 / 0.45 / 0.64 / 1.00 |
| a = 1.50, reset | 0.42 / 0.46 / 0.62 / 0.97 |
| tanh, preserved (descriptive) | 0.59 / 0.61 / 0.68 / 0.75 |

- **Registered verdict: the predicted "no gating" FAILS at both a. The registered competing outcome, gating at small
  scale, holds.**
- Validity: the positive control passed (width 1: 0.0 placed at R/R_glob = 0.1, both a). The validity checks passed
  on constructed cases before the run.
- **Where the unplaced runs sit (descriptive, reported beside the verdict).**
  - At H they are stationary two-unit configurations: median scale-relative gradient 1.4e−7, and 59% below the 1e−6
    tolerance.
  - None is a single-unit local minimum, so the registered longer-horizon extension applied to 0 runs.
  - Only 5 of 640 endpoints changed status between H/4 and H, all to placed.
  - Placed endpoints are almost all the cancelling pair.
IDs: `Track 7 no-gating FAIL a=1.30`; `Track 7 no-gating FAIL a=1.50`; `Track 7 placed 0.003 a=1.30`; `Track 7 placed 0.1 a=1.30`; `Track 7 positive control`; `Track 7 H`.

**Reading.**
- At width 2 on symmetric windows, the *conditional minimiser* is placed at every scale (WP-9, registered).
- *Training* at a fixed small output scale nevertheless stays unplaced about half the time, at stationary unplaced
  two-unit configurations.
- So gating in training does not require a threshold of the conditional minimiser. Here it comes from where training
  gets stuck, not from what the loss prefers globally.

**Say:** "The landscape predicts no gating at width 2 on symmetric windows, and none exists in the conditional
minimiser. The registered training test nevertheless found gating: at small held scale about half the runs remain
unplaced, at stationary two-unit configurations. So the conditional-threshold account does not by itself predict
width-2 training."

**Do not say:** "width 2 confirms no gating", or that the width-2 result is a training-level confirmation. The training
prediction failed.

## WP-21. A second prospective test of the timescale account (Task B, registered; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Registration: `ts_test_registration.md` (db2a915), written before any run. The own thresholds for the fresh training
sets were frozen with a hash before any Adam run (0da303c). Producer: `src/ts_test.py` → `ts_test/runs.csv`,
`ts_test/scores.csv`.

**Design.**
- Adam, standard protocol, every-step crossing detection.
- **Fresh training sets:** seeds 830,000–830,079, never used anywhere, at **a = 1.45 and 1.60**. The timescale fit
  used only a = 1.30 and 1.50.
- The residual–ratio relationship is the one frozen for the SGD extension (α = 0.0157, β = 2.658).

| a | crossed | median ratio at crossing | TS-1 predicted | observed | tolerance | TS-1 | TS-2: own − population [95% CI] | TS-2 |
|---|---|---|---|---|---|---|---|---|
| 1.45 | 80/80 | 0.0112 | 0.0456 | 0.0532 | ±0.0114 | **PASS** | [-0.059, -0.022] **PASS** |
| 1.60 | 80/80 | 0.0249 | 0.0818 | 0.0809 | ±0.0205 | **PASS** | [-0.059, -0.022] **PASS** |

- Every run crossed, so the registered non-crosser sensitivity analysis is identical to the primary scores.
- At a = 1.60 the median ratio (0.0249) is slightly above the fit's range (maximum
  0.0229), so that prediction is a slight extrapolation.
IDs: `Task B TS-1 a=1.45`; `Task B TS-1 a=1.60`; `Task B TS-2 a=1.45`; `Task B TS-2 a=1.60`; `Task B pred a=1.45`; `Task B obs a=1.45`; `Task B pred a=1.60`; `Task B obs a=1.60`; `Task B crossed`.

**Say:** "A second registered test, at activation values the fit never saw and on fresh training sets, predicted the
median residual from the Adam timescale relationship within tolerance at both values, and each run's own threshold
again beat the population threshold."

**Do not say:** that this establishes the timescale account as the residual's mechanism (it rests on a post hoc fit
whose within-a correlations are weak, and it did not carry over to width 2), or that the a = 1.60 prediction is an
interpolation.

## WP-22. The mechanism figure (Task C; for the submission, main text)

File: `results/figures/v5/mechanism_w1.pdf`. Caption entry in `results/figures/v5/captions.md` ("mechanism_w1"), with
its one-sentence message, population and every number. Producer: `src/mechanism_w1_figure.py`. It passes the v5 audit
(5.49 × 2.38 in, smallest glyph 8 pt).

**What it shows.** For width 1 at a = 1.30 and 1.50, the conditional minimiser's |w₁| against R/R_glob (log scale).
- It starts near the class-mean optimum α* = 1.79 at small scale: 1.751 and
  1.756 at the smallest scale shown.
- It falls below the placement bound a/1.4 before R_glob. The bound is necessary, not sufficient.
- Placement switches on at R_glob, where all 38 + 38 free-training
  crossings sit, every one with |w₁| < a/1.4.
- It is the picture of WP-12: the loss moves from rewarding the class-mean gap to rewarding the worst-case gap.
IDs: `Task C figure audit`; `Task C w1 at smallest a=1.30`; `Task C crossings below bound`.

**Say:** "As output scale grows, the conditional minimiser's first-layer weight leaves the class-mean optimum and
crosses below the placement bound; placement switches on at R_glob, where training crosses."

**Do not say:** that the minimiser path is certified (it is a validated search; only R_glob is certified), or that
crossing the bound a/1.4 is the switch (the bound is necessary, not sufficient).

## WP-23. Is the timescale relationship consistent with the within-a interventions? (Item 1; POST HOC; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Producer: `src/timescale_consistency.py` → `timescale_consistency/runs.csv`, `arms.csv`, `summary.json`.
- **Frozen relationship:** residual = α + β·ratio, with α = 0.0157 and
  β = 2.658. This is the file used by the SGD extension and Task B (SHA-256 16792946…).
- **Runs:** every crossing run of the first lag test (all φ arms), the deconfounded lag test (primary and t\* rules,
  all φ arms) and Block 4b (all arms), 1530 in total.
- Each run was replayed to its recorded crossing, and every replay reproduces its crossing exactly.
- For each arm, the predicted median residual (from each run's own ratio at crossing) is compared with the observed
  one, at the registered tests' tolerance, max(0.01, 0.25·|pred|).

| test | a | arm (φ or intervention) | n | median ratio | predicted | observed | tolerance | within |
|---|---|---|---|---|---|---|---|---|
| 4b | 1.30 | control | 48 | 0.0040 | 0.0263 | 0.0311 | ±0.0100 | yes |
| 4b | 1.30 | reset | 48 | 0.0041 | 0.0266 | 0.0315 | ±0.0100 | yes |
| 4b | 1.30 | teleport | 48 | 0.0040 | 0.0264 | 0.0308 | ±0.0100 | yes |
| 4b | 1.30 | teleport_reset | 48 | 0.0041 | 0.0266 | 0.0312 | ±0.0100 | yes |
| 4b | 1.50 | control | 48 | 0.0157 | 0.0574 | 0.0656 | ±0.0143 | yes |
| 4b | 1.50 | reset | 48 | 0.0163 | 0.0591 | 0.0672 | ±0.0148 | yes |
| 4b | 1.50 | teleport | 48 | 0.0158 | 0.0576 | 0.0655 | ±0.0144 | yes |
| 4b | 1.50 | teleport_reset | 48 | 0.0161 | 0.0585 | 0.0660 | ±0.0146 | yes |
| lag1 | 1.30 | 0.25 | 49 | 0.0009 | 0.0181 | 0.0047 | ±0.0100 | **no** |
| lag1 | 1.30 | 0.5 | 48 | 0.0035 | 0.0250 | 0.0266 | ±0.0100 | yes |
| lag1 | 1.30 | 1.0 | 48 | 0.0040 | 0.0263 | 0.0311 | ±0.0100 | yes |
| lag1 | 1.30 | 2.0 | 44 | 0.0040 | 0.0264 | 0.0312 | ±0.0100 | yes |
| lag1 | 1.50 | 0.25 | 49 | 0.0035 | 0.0249 | 0.0107 | ±0.0100 | **no** |
| lag1 | 1.50 | 0.5 | 48 | 0.0146 | 0.0545 | 0.0534 | ±0.0136 | yes |
| lag1 | 1.50 | 1.0 | 48 | 0.0157 | 0.0574 | 0.0656 | ±0.0143 | yes |
| lag1 | 1.50 | 2.0 | 44 | 0.0158 | 0.0576 | 0.0664 | ±0.0144 | yes |
| lag2-prim | 1.30 | 0.25 | 48 | 0.0034 | 0.0248 | 0.0278 | ±0.0100 | yes |
| lag2-prim | 1.30 | 0.5 | 48 | 0.0038 | 0.0259 | 0.0302 | ±0.0100 | yes |
| lag2-prim | 1.30 | 1.0 | 48 | 0.0040 | 0.0263 | 0.0311 | ±0.0100 | yes |
| lag2-prim | 1.30 | 2.0 | 48 | 0.0041 | 0.0265 | 0.0316 | ±0.0100 | yes |
| lag2-prim | 1.50 | 0.25 | 48 | 0.0135 | 0.0515 | 0.0563 | ±0.0129 | yes |
| lag2-prim | 1.50 | 0.5 | 48 | 0.0150 | 0.0556 | 0.0629 | ±0.0139 | yes |
| lag2-prim | 1.50 | 1.0 | 48 | 0.0157 | 0.0574 | 0.0656 | ±0.0143 | yes |
| lag2-prim | 1.50 | 2.0 | 48 | 0.0162 | 0.0587 | 0.0667 | ±0.0147 | yes |
| lag2-tstar | 1.30 | 0.25 | 48 | 0.0013 | 0.0192 | 0.0071 | ±0.0100 | **no** |
| lag2-tstar | 1.30 | 0.5 | 48 | 0.0035 | 0.0251 | 0.0271 | ±0.0100 | yes |
| lag2-tstar | 1.30 | 1.0 | 48 | 0.0040 | 0.0263 | 0.0311 | ±0.0100 | yes |
| lag2-tstar | 1.30 | 2.0 | 48 | 0.0041 | 0.0265 | 0.0315 | ±0.0100 | yes |
| lag2-tstar | 1.50 | 0.25 | 48 | 0.0054 | 0.0300 | 0.0164 | ±0.0100 | **no** |
| lag2-tstar | 1.50 | 0.5 | 48 | 0.0146 | 0.0544 | 0.0557 | ±0.0136 | yes |
| lag2-tstar | 1.50 | 1.0 | 48 | 0.0157 | 0.0574 | 0.0656 | ±0.0143 | yes |
| lag2-tstar | 1.50 | 2.0 | 48 | 0.0162 | 0.0587 | 0.0674 | ±0.0147 | yes |

- **28 of 32 arms are within tolerance.** The 4 misses are exactly the φ = 0.25 arms of
  the first lag test and of the t\* rule, at both a.
- There, slowing w₂'s learning rate from early on lowers the ratio at crossing 3–4.5×. The residual falls in the
  predicted direction but by more than predicted: observed 0.005–0.016 against predicted
  0.018–0.030.
- **Where the interventions barely move the ratio,** the relationship predicts every arm. That covers the deconfounded
  primary rule (φ applied from 0.7 of the threshold), where Adam's normalisation absorbs the change, so the ratio moves
  by at most 20% and the residual by 11–14%. It also covers all four Block 4b arms, where the ratio does not move.
- **Slopes:** within a, over all arms, the residual–ratio slope is 4.69 (a = 1.30) and
  3.09 (a = 1.50), against the pooled fit's 2.66.
- **Partial correlation:** the Spearman partial correlation of residual and ratio, controlling for a, is
  0.51 over all arms and 0.48 on the fit's own data.
- **Validity note (Block 4b; disclosed, no verdict changed).** In the registered Block 4b run, the arms shared optimiser-state
  tensors, which the preceding arm's continuation mutated in place (`load_state_dict` shares them).
  - The **teleport** arm therefore started from the control run's end-of-run Adam state, not the state at the switch.
  - The **teleport_reset** arm started from the reset run's end state, not zeroed moments.
  - Control and reset are unaffected.
  - Here the arms were replayed *as run*, which reproduces them exactly. The corrected rerun leaves 4b-ID and 4b-OM
    FAIL (WP-10).
IDs: `Item 1 arms within tolerance`; `Item 1 misses are the phi=0.25 arms`; `Item 1 within-a slope a=1.30`; `Item 1 within-a slope a=1.50`; `Item 1 partial Spearman`; `Item 1 replays reproduce`.

**Statement (in between, stated exactly).**
- The frozen relationship predicts the residual across activation values and optimisers (two registered prospective
  tests). Within a, it predicts every intervention arm that leaves the timescale ratio near its unperturbed value.
- It gets the *direction* of a large within-a slowdown right but not its *size*. When the output rate is slowed from
  early on, the residual falls further than the linear relationship predicts; the within-a dependence is steeper.
- So the account is quantitatively supported across a and optimisers and for small within-a changes, not for large
  within-a changes of the ratio.

**Say:** "A single linear relation between the residual and the ratio of output growth to branch relaxation predicts
the residual across activation values and optimisers (two registered prospective tests) and for within-a interventions
that leave the ratio nearly unchanged. It captures the direction but underestimates the size of the effect when the
output rate is slowed substantially (post hoc)."

**Do not say:** that the relation holds quantitatively within a for arbitrary interventions, that it is the residual's
mechanism, or that the lag tests confirm it. The registered lag-test predictions L2, L1′ and L2′ failed, and those
verdicts stand.

## WP-24. The lag law r = κ(a)·χ: a no-fit constant, and a registered ramp test (Tracks 1A and 1B; for the submission)

Producers: `src/lag_law.py` → `lag_law/` (κ, predictions, comparison, winding check, relaxation times);
`src/ramp.py` → `ramp/` (design, runs, scores, post hoc). Math note §13 has the derivation.

**Label for everything in Track 1A: derived after the fitted relationship (residual = α + β·ratio, WP-16/21/23) was known.**

### 1A. What was committed before any comparison (b6ae433)

- **The law.** Hidden coordinates θ = (w₁, b₁, b₂) track the conditional stationary branch θ\*(s) of the loss at fixed
  output scale s. To first order the steady lag gives

  r = (s_c − s\*)/s\* = κ(a)·χ, with χ = (ṡ/s\*)/(ηλ_min) and κ(a) = λ_min·[∇G·(PH)⁻¹θ\*′]/[∇G·θ\*′].

  - H is the Hessian at the switch s\* and θ\*′ the branch tangent.
  - P = I for SGD. For Adam, P = diag(1/(√v̂ + ε)) is the median preconditioner at crossing.
  - λ_min = λ_min(P^{1/2}HP^{1/2}).
  - κ has no free parameter. χ is the paper's timescale ratio, up to s\* against s_c (a factor 1 + r).
- **Winding.** f_a(t + 2π) = f_a(t) + 2π, so b₁ has 2π copies with the same loss and threshold but a different output-bias
  drift. κ depends on the copy k. Every existing run sits on k = −1 at a = 1.30 and on k = 0 elsewhere.
  - This was found, and checked with controlled population ramps, before any comparison (`winding_check.csv`):

| a | k | χ | predicted r | measured r |
|---|---|---|---|---|
| 1.30 | +0 | 0.003 | -0.0223 | -0.0215 |
| 1.30 | -1 | 0.003 | +0.0229 | +0.0236 |
| 1.30 | +1 | 0.003 | -0.0675 | -0.0607 |
| 1.50 | +0 | 0.002 | +0.0081 | +0.0082 |
| 1.50 | +0 | 0.005 | +0.0204 | +0.0207 |

- **Values and predictions.** Per-run predictions were committed with SHA-256 af6bb137… for 1,750 runs: 32
  intervention arms, SGD at two a, and Task B at two a. 875 of these runs sit on the mirror branch, where κ is
  unchanged.

| a | k | κ_Adam | κ_SGD | observed/predicted slope (post hoc, through the origin) |
|---|---|---|---|---|
| 1.30 | -1 | 7.61 | 7.62 | 0.89 |
| 1.45 | +0 | 4.59 | 4.61 | 0.83 |
| 1.50 | +0 | 4.05 | 4.07 | 0.91 |
| 1.60 | +0 | 3.29 | 3.32 | 0.82 |

### 1A. Result (899850d; no refit)

| set | arms | within max(0.01, 0.25·pred) | observed/predicted per arm |
|---|---|---|---|
| 4b | 8 | 8 | 1.00–1.03 |
| SGD | 2 | 2 | 1.09–1.12 |
| TaskB | 2 | 2 | 0.99–1.03 |
| lag1 | 8 | 8 | 0.70–1.04 |
| lag2-prim | 8 | 8 | 1.02–1.07 |
| lag2-tstar | 8 | 8 | 0.71–1.03 |

- **36 of 36 arms are within tolerance.**
  - The four φ = 0.25 arms of the first lag test and the t\* rule (w₂ slowed from early on) are at
    0.70–0.76. They are within tolerance only through the 0.01 floor.
  - The other 32 arms are at 0.91–1.12. This corrects the 899850d
    commit message, which said 0.99–1.13 and called the four low arms "the φ = 0.25 arms"; the deconfounded primary
    rule's φ = 0.25 arms are at 1.03–1.07.
- **Per a (post hoc through-origin fit):** observed slope / κ = 0.89 (1.30), 0.83 (1.45), 0.91 (1.50), 0.82 (1.60).
  That is within 30% at every a. The law over-predicts by 9–18%.
- **Timescales.** The relaxation time 1/(ηλ_min) is 7.8–16.2
  steps; Adam's momentum time is 10 steps. Momentum does not change the steady lag (math note §13.3).

### 1B. The ramp experiment (registered c4b4c6d; own thresholds frozen 83f66f5 and 15b4744 before any crossing was read)

**Design.**
- Output scale forced to s = s₀e^{γt} from 0.5·s\* after a 4,000-step warm-up. Hidden coordinates and output bias
  train normally (Adam lr 0.01 or SGD lr 0.3).
- 40 fresh seeds per cell, 6 γ cells per setting.
- Every run starts on the population branch at winding k:
  - (1.30, k = −1) and (1.50, k = 0) are the natural windings (R1–R3);
  - (1.30, k = 0) has κ < 0, so the law predicts crossing *before* the threshold (R5).
- Predictions: r_pred = κ_k·χ. SGD's χ is fully a priori from γ, η and the landscape. Adam's χ uses each run's measured
  v̂ at crossing.
- Observed r uses each seed's own-sample global threshold.
- **Rules:**
  - R1: through-origin slope in [0.7, 1.3];
  - R2: sign(κ)·Spearman(γ, cell median) ≥ 0.9;
  - R3: |slowest-cell median| < 0.005.
- **Adam's γ grid.** It was calibrated on pilot seeds from predictions only, and targets predicted lags of only
  0.0005–0.006. Adam cannot follow faster ramps: its step is capped near η per coordinate, and the pilot runs stopped
  crossing.

**Registered verdicts (all 1,440 runs crossed; none placed during warm-up; no run changed winding).**

| a | k | optimiser | rule set | R1: slope | R2: signed Spearman | R3: slowest median |
|---|---|---|---|---|---|---|
| 1.30 | -1 | adam | R1–R3 | -7.12 **FAIL** | 1.00 **PASS** | -0.0000 **PASS** |
| 1.30 | -1 | sgd | R1–R3 | 0.78 **PASS** | 1.00 **PASS** | +0.0009 **PASS** |
| 1.30 | +0 | adam | R5 (sign test) | 9.10 **FAIL** | 1.00 **PASS** | -0.0012 **PASS** |
| 1.30 | +0 | sgd | R5 (sign test) | 1.30 **FAIL** | 1.00 **PASS** | -0.0014 **PASS** |
| 1.50 | +0 | adam | R1–R3 | -9.99 **FAIL** | 0.71 **FAIL** | -0.0250 **FAIL** |
| 1.50 | +0 | sgd | R1–R3 | -0.10 **FAIL** | 1.00 **PASS** | -0.0233 **FAIL** |

**Diagnosis (after scoring).**
- Every γ cell contains the same subpopulation of seeds at a large static offset: 5% quantiles ≈ −0.25 and 95% ≈ +0.09
  in every cell.
- For these seeds the own-sample *global* threshold is not the switch of the branch the ramp forces. For 35% of seeds at
  a = 1.30 and 68% at a = 1.50, that branch's own-sample switch differs from s_own by more than 1%.
- Free training is not forced onto a branch; the ramp is. The offsets pull the pooled slopes (R1) and, at a = 1.50, the
  medians (R3).

**POST HOC (labelled): the same rules against each seed's tracked-branch own-sample switch** (`ramp.posthoc_branch`;
the last column is the fraction of seeds whose branch switch is > 1% from s_own).

| a | k | optimiser | rule set | R1 | R2 | R3 | branch ≠ own |
|---|---|---|---|---|---|---|---|
| 1.30 | -1 | adam | R1–R3 | 0.52 **FAIL** | 1.00 **PASS** | +0.0002 **PASS** | 0.350 |
| 1.30 | -1 | sgd | R1–R3 | 1.24 **PASS** | 1.00 **PASS** | +0.0010 **PASS** | 0.350 |
| 1.30 | +0 | adam | R5 (sign test) | 1.49 **FAIL** | 1.00 **PASS** | -0.0009 **PASS** | 0.350 |
| 1.30 | +0 | sgd | R5 (sign test) | 0.90 **PASS** | 1.00 **PASS** | -0.0010 **PASS** | 0.350 |
| 1.50 | +0 | adam | R1–R3 | 0.90 **PASS** | 0.94 **PASS** | -0.0002 **PASS** | 0.675 |
| 1.50 | +0 | sgd | R1–R3 | 1.14 **PASS** | 1.00 **PASS** | +0.0010 **PASS** | 0.675 |

Cell medians, post hoc (observed vs predicted, slowest to fastest γ):

| a | k | optimiser | observed | predicted |
|---|---|---|---|---|
| 1.30 | -1 | adam | +0.0002 / +0.0003 / +0.0007 / +0.0013 / +0.0030 / +0.0036 | +0.0005 / +0.0008 / +0.0013 / +0.0021 / +0.0032 / +0.0049 |
| 1.30 | -1 | sgd | +0.0010 / +0.0026 / +0.0066 / +0.0168 / +0.0442 / +0.1262 | +0.0010 / +0.0025 / +0.0063 / +0.0158 / +0.0398 / +0.1000 |
| 1.30 | +0 | adam | -0.0009 / -0.0013 / -0.0019 / -0.0034 / -0.0047 / -0.0072 | -0.0005 / -0.0008 / -0.0013 / -0.0021 / -0.0032 / -0.0052 |
| 1.30 | +0 | sgd | -0.0010 / -0.0026 / -0.0065 / -0.0161 / -0.0386 / -0.0875 | -0.0010 / -0.0025 / -0.0063 / -0.0158 / -0.0398 / -0.1000 |
| 1.50 | +0 | adam | -0.0002 / -0.0003 / +0.0001 / +0.0010 / +0.0014 / +0.0043 | +0.0005 / +0.0008 / +0.0014 / +0.0023 / +0.0038 / +0.0065 |
| 1.50 | +0 | sgd | +0.0010 / +0.0025 / +0.0063 / +0.0160 / +0.0414 / +0.1136 | +0.0010 / +0.0025 / +0.0063 / +0.0158 / +0.0398 / +0.1000 |

- **SGD:** observed/predicted cell medians are 0.99–1.06 in the four slowest cells,
  0.97–1.11 in the fifth (κχ = 0.04) and 0.87–1.26 in the fastest (κχ = 0.1, where the
  linearisation starts to fail; math note §13.3(iv)).
  - The sign test holds: on the k = 0 copy at a = 1.30 the crossings come *early*, as predicted.
- **Adam:** the predicted lags are ≤ 0.006, and the cell medians do not follow them. Observed/predicted is
  0.35–0.94 (a = 1.30, k = −1), 1.40–1.85 (k = 0) and
  -0.39 to 0.67 (a = 1.50).
  - The post hoc R1 pass at a = 1.50 comes from the pooled per-run slope, not from agreement at the cell level.
  - Adam's frozen-preconditioner linearisation is not supported at these small lags. Its v̂ adapts during the ramp
    (math note §13.3(iv)(c)).

### Why Adam differs in the ramp (POST HOC; `ramp.adam_contrast` → `ramp/adam_contrast.csv`)

Three things could differ between the Adam ramp and free Adam training, where the law held (Track 1A): the measured preconditioner, the growth rate and the branch. The data single out the **preconditioner**.
- **Preconditioner.** At crossing, Adam's √v̂ is 8–88× smaller in the ramp than in free training, per coordinate and at both a. The 4,000-step warm-up holds the hidden coordinates at a stationary point, where the gradient vanishes, so v̂ decays; in free training v̂ still carries the larger gradients of the approach.
  - With P = 1/(√v̂ + ε), the frozen-P relaxation time 1/(ηλ_min(P^{1/2}HP^{1/2})) is 0.17–0.19 steps in the ramp, against 7.8–15.0 steps in free training.
  - A relaxation time below one step means ηλ_min > 1. The linear update with that P would overshoot, so the linearisation behind the law (small ηλ, P fixed over the relaxation) does not describe the ramp. Adam there is in its self-normalising regime: steps of about η per coordinate, with v̂ adapting to the ramp's own gradients.
  - Consistently, the ramp reaches its crossing 238–624 steps after growth starts (median), within v̂'s 1,000-step memory. Free training takes 1680–3266 steps.
- **Growth rate.** The ramp's rate is 2.3–2.7× the free-training rate at crossing (medians), the same order. SGD in the ramp covers far wider rates and follows the law, so the rate alone does not explain Adam's failure.
- **Branch.** The ramp runs sit on the same windings as free training (k = −1 at a = 1.30, k = 0 at a = 1.50). They are all on the non-mirror branch, where κ is the same. The post hoc scoring already uses each run's tracked branch, and SGD passes on the same branches.
- **So:** the law's Adam form needs a preconditioner that is stationary and small enough that ηλ_min ≪ 1 over the relaxation. Free training satisfies this; the ramp, started at a stationary point, does not. This is a limit of the frozen-preconditioner linearisation (math note §13.3(iv)(c)), identified after scoring.

### Follow-up requested after scoring: the Adam ramp from initialisation (NOT registered: infeasible by its design rules)

- **Design, fixed before any registered run** (`src/ramp2.py`; `ramp2/design_sweep.csv`; pilot seeds 869,100–869,107,
  predictions only):
  - no warm-up; standard initialisation; s = 0.5·e^{γt};
  - each run to be scored against its tracked-branch switch;
  - γ admissible only if ≥ 6 of 8 pilot runs cross **and** the median ηλ_min(P^{1/2}HP^{1/2}) at crossing is ≤ 0.5,
    so that the frozen-preconditioner law can apply.
- **Outcome of the design sweep:** no admissible γ at either a.
  - Where most runs cross (γ ≤ 1.8e-4), v̂ collapses during the slow ramp and ηλ_min is 4.4–11.5.
  - Where ηλ_min ≤ 0.5 (γ ≈ 1e-3), at most 3 of 8 runs cross.
- So no forced exponential ramp at these rates gives a test in the law's Adam regime, and nothing was registered or
  run. Free training meets the condition because its growth is not imposed (relaxation 7.8–15 steps; R4, 1A).
- **Say:** "Adam's version of the law needs a stationary preconditioner. A forced ramp either lets it collapse (slow ramps)
  or outruns the unit (fast ramps), so the law's Adam form is supported by free training only."

### R4. Free Adam training at three learning rates (registered with 1B; seeds 860,100–860,139)

| a | crossed (η = 0.01 / 0.005 / 0.0025) | median residual | tolerance | R4 |
|---|---|---|---|---|
| 1.30 | 39 / 39 / 39 | +0.0296 / +0.0304 / +0.0308 | ±0.0100 | **PASS** |
| 1.50 | 39 / 39 / 39 | +0.0619 / +0.0625 / +0.0647 | ±0.0155 | **PASS** |

- The law predicts η-invariance. Growth ṡ and relaxation ηλ both scale with η, so χ does not change.
- A pass means that the lag does not vanish in the gradient-flow limit of free training.

### Notation for the main text

- κ(a) is the lag constant. The plan's notation table uses κ(a). The existing κ₀ (the limiting-cubic constant in Block
  3's window design) must then get another symbol in the main text (suggestion: ν₀), and Ĝ/D is not called κ anywhere
  in the main text.
- χ = (ṡ/s\*)/(ηλ_min). The tests used each run's measured ratio at crossing, which divides by s_c instead of s\*.
  State this once.
- The fitted line residual = α + β·ratio (WP-16, WP-21, WP-23) stays as the registered basis of those verdicts
  (EXT, TS-1). Where the paper states the account, use r = κ(a)χ, labelled "derived after the fitted relationship was
  known". It has no intercept and no fitted parameter.

### Say / Do not say

**The κ outcome (1A) is "within 30%"** (post hoc slope/κ = 0.82–0.91 at every a; 36/36 arms within tolerance).

- **Say (observed outcome, within 30%):** "A first-order tracking analysis gives the lag constant κ(a) with no fitted
  parameter. Computed before comparison, it predicts the median residual of all 36 existing arms within tolerance, and
  the observed slope is within 30% of κ(a) at every a (the analysis was derived after a fitted relationship was
  known)."
- **If it had been within a factor of two (not observed),** the sentence would be: "predicts the residual's scale within
  a factor of two".
- **If worse (not observed),** it would be: "does not predict the residual quantitatively".
- **Say (ramp, registered):** "In a registered test that forces the output scale to grow at set rates, the lag increases
  monotonically with the rate in every setting (R2 passes in 5 of 6) and vanishes at the slowest rate at a = 1.30. The
  pooled magnitude criterion (R1) passes only for SGD at a = 1.30 on the natural winding. The failures trace to seeds
  whose own-sample global threshold is not the switch of the branch the ramp forces."
- **Say (ramp, post hoc, labelled):** "Measured against the switch of the branch each run actually tracks, SGD's
  crossing lag matches κ(a)χ with no fitted parameter across two decades of rate: within 6% in the four slowest rate
  cells and within 30% in every cell, including the predicted *early* crossing on the winding copy with negative κ.
  Adam's lags in the ramp do not follow the law."
- **Do not say:**
  - that the ramp test passed as registered;
  - that κ was predicted before the fitted relationship was known;
  - that the law holds for Adam in the ramp (its cell medians do not follow κχ, even post hoc);
  - that the law holds in the nonlinear regime κχ ≳ 0.1;
  - that the lag law explains width 2 (WP-15; Track 2 below).
IDs: `1A arms within tolerance`; `1A slope/kappa a=1.30`; `1A slope/kappa a=1.50`; `1B registered R1 SGD 1.30 k=-1`; `1B post hoc R1 SGD 1.50`; `1B branch off own a=1.50`.

## WP-25. Outside the sine family: GELU, SiLU and Mish at width 1 (Track 3A; for the submission)

Producers: `src/act_general.py` (steps 0–4) and `src/act_posthoc.py` → `act_general/`. Summary: `results/act_summary.md`.
Registrations: criterion `act_criterion_registration.md` (b707e86, amendments a251e80 and cf2d68b, both implementation
only and made before any criterion result); training `act_training_registration.md` (ad99053; κ frozen with SHA-256
before any run). Labels: registered, validated (not certified), post hoc.

**Step 0 (validated).** The generalised machinery reproduces the certified f_a bracket at a = 1.30: the switch is in
[4.95796, 4.95835], inside [4.95, 4.9625].

**Steps 1–3 (validated; the criterion registered).**
- A single unit places and solves for all three activations. The dip sits under the inner window (σ = +1). A midpoint
  output bias is sign-correct and a wrong bias is not.
- Differences from the sine case:
  - the dip has a fixed size, so there is no winding;
  - the ramp is not odd, so the class-mean maximiser is not attained.
- The registered criterion: a switch is predicted iff the validated conditional minimiser is unplaced at s = 0.05 and
  at s = 0.1.

| activation | Ĝ | maximiser (w₁, b₁) | status at s = 0.05 / 0.1 | criterion (registered) | validated switch bracket in s | s\* (tracked branch) | κ_Adam |
|---|---|---|---|---|---|---|---|
| GELU | 0.037547 | (0.864, -0.904) | unplaced / unplaced | **switch predicted** | [6.6117, 6.6714] | 6.6456 | +0.1473 |
| SiLU | 0.050912 | (1.583, -1.636) | unplaced / undecided | **undetermined** | [3.6190, 3.6517] | 3.6392 | -0.1444 |
| Mish | 0.054772 | (1.526, -1.561) | unplaced / undecided | **undetermined** | [3.1908, 3.2197] | 3.2147 | -0.1539 |

- **Post hoc, disclosed:** every small-scale minimiser is a one-sided ramp. It is "placed" in exact arithmetic by a
  margin of 10⁻¹⁵³ or smaller, and its gap is 0 in double precision. SiLU and Mish are "undetermined" because of this.
- Brackets are validated (restart ladder, independent CMA-ES, audit), not certified.

**Step 4: training (registered).**
- Adam lr 0.01, budget 32,000, fresh seeds.
- T-a: ≥ 90% of crossing runs at s ≥ s_lo.
- T-b: median residual within max(0.01, 0.25|pred|) of median κχ.
- A cell with < 30 crossings is UNRESOLVED.
- The 200-seed arm was registered before any run, after calibration showed crossing rates near 50%.

| activation | arm | runs | crossing | T-a | T-b |
|---|---|---|---|---|---|
| GELU | primary | 40 | 23 | **UNRESOLVED** | **UNRESOLVED** |
| GELU | secondary pooled 200 | 200 | 132 | **FAIL** (0.477) | **FAIL** (obs -0.0183, pred +0.0023, median χ 0.015) |
| SiLU | primary | 40 | 25 | **UNRESOLVED** | **UNRESOLVED** |
| SiLU | secondary pooled 200 | 200 | 135 | **FAIL** (0.622) | **FAIL** (obs +0.0353, pred -0.0324, median χ 0.225) |
| Mish | primary | 40 | 25 | **UNRESOLVED** | **UNRESOLVED** |
| Mish | secondary pooled 200 | 200 | 135 | **FAIL** (0.704) | **FAIL** (obs +0.0658, pred -0.0393, median χ 0.256) |

**Post hoc** (`posthoc_training.csv`). The residual is measured against the *population* threshold.

| activation | IQR of r | fraction with \|r\| ≤ 0.10 | crossings below ½·s_glob | Spearman(r, χ) | final \|w₂\| of non-crossing runs (median) |
|---|---|---|---|---|---|
| GELU | [-0.110, +0.079] | 0.52 | 18 | -0.35 | 0.16 |
| SiLU | [-0.044, +0.144] | 0.53 | 1 | -0.29 | 0.07 |
| Mish | [-0.022, +0.162] | 0.50 | 1 | -0.23 | 0.11 |

- About a third of runs never cross (60 (GELU), 65 (SiLU), 65 (Mish), of 200). They stall near a trivial point with small |w₂|.
- SiLU's and Mish's crossings happen at χ ≈ 0.2, which is outside the regime where the linear lag law applies (math note
  §13.3(iv)).
- Own-sample thresholds were not computed. Each training set has 400 points (200 per class); `act_summary.md` says
  "200 points per run", which means 200 per class.

**POST HOC checks on the same runs (author's request; `src/act_posthoc2.py` → `act_general/posthoc2_*`; no new runs;
the registered verdicts above stand as scored).**

*(1) Validity: χ at crossing against the width-1 sine range.*
- The quantity is the same growth-to-relaxation ratio as for the sine runs, with each run's own Adam preconditioner.
- The width-1 lag law was verified (Track 1A) on runs with χ = 0.0001–0.064 (95% of runs
  ≤ 0.021). The largest arm-median χ at which it held is 0.0249.
- **The validity condition used here:** the median χ at crossing is ≤ 0.0249.

| activation | median χ [IQR] | runs with χ ≤ 0.0249 | runs with χ ≤ 0.064 | condition met |
|---|---|---|---|---|
| GELU | 0.0137 [0.0079, 0.0215] | 81% | 94% | yes |
| SiLU | 0.2242 [0.1422, 0.3475] | 1% | 5% | **no** |
| Mish | 0.2556 [0.1721, 0.4209] | 0% | 2% | **no** |

*(2) Crossings scored against each run's tracked-branch switch* (Newton continuation on the run's own 400-point sample
from its crossing state, as for the ramp).

| activation | runs with a branch switch | median r vs branch | median κχ | tolerance | within | at or above the branch switch | \|r\| ≤ 0.01 | branch switch / population threshold, 10–90% |
|---|---|---|---|---|---|---|---|---|
| GELU | 117 of 132 | +0.0017 | +0.0021 | ±0.01 | yes | 93% | 96% | 0.88–1.17 |
| SiLU | 134 of 135 | +0.0142 | -0.0324 | ±0.01 | **no** | 66% | 31% | 0.83–1.19 |
| Mish | 134 of 135 | +0.0342 | -0.0393 | ±0.01 | **no** | 75% | 28% | 0.84–1.20 |

- **GELU (condition met):**
  - Against its tracked branch, the crossing follows the lag law: median r +0.0017 against
    κχ +0.0021.
  - 93% of runs cross at or above their branch switch, and
    96% within 1% of it.
  - The registered tests compared crossings with the population threshold. Each run's own branch switch (on its own
    400-point sample) lies 0.88–1.17× the population threshold (10–90%).
    Since the crossings sit within 1% of those switches, that spread is what the registered tests measured.
  - The other 15 GELU crossings all happen early (s ≤ 0.45, against a population threshold of
    6.64), from states far from the retained branch. Continuing their branch from the crossing finds no switch:
    no sign change within 400 continuation steps in 3, continuation lost in 11, gap undecided at the cell cap in 1.
- **Requested follow-up, a GELU prospective test with an early-scale branch rule: NOT registered (infeasible).**
  - The rule scale had to lie below the earliest crossing observed, s = 0.0043. None of the
    200 existing runs starts below it (median initial |w₂| is 0.54), so the only
    admissible "early scale" is initialisation.
  - There, the branch classification is final for only 34 of 132 crossing runs
    (26%), against the required 95% (`src/gelu_rule_feasibility.py`).
  - Nothing was registered or trained.
- **SiLU and Mish (condition not met):**
  - χ at crossing is about 10× beyond the largest χ at which the sine law was verified.
  - Against the tracked branch, the residual has the wrong sign for κχ, and only 31% and
    28% of runs cross within 1% of their branch switch.
  - This is outside the regime of the linear lag law (math note §13.3(iv)(a)), so these runs neither test nor
    contradict it.

**Say:**
- "For GELU, SiLU and Mish a single unit can solve the task, and the loss at fixed output scale has a validated
  unplaced-to-placed switch."
- "(Post hoc) For GELU, whose runs cross in the timescale regime where the sine lag law was verified, crossings track
  within 1% of each run's own branch switch and follow the lag law; the registered tests, which used the population
  threshold, measured the spread of those switches. For SiLU and Mish, output growth at crossing is about ten times too fast for the law to apply."
- "Free Adam training does not track it. Registered tests on 200 seeds per activation fail: 48–70% of crossings lie
  above the switch, and the median residual has the wrong sign or size for the lag law. About a third of runs never
  cross."
- "The scale-gating account is therefore specific to activations whose non-monotone part scales with the pre-activation
  (the sine family here). Its extension to practical activations is not supported by the registered tests."

**Do not say:**
- that the switch for these activations is certified;
- that the criterion predicted the switch for SiLU or Mish (it was undetermined);
- that training confirms scale gating outside the sine family;
- that the registered tests passed, or that the post hoc branch scoring was registered;
- that the lag law holds for SiLU or Mish (their runs are outside its validity condition).
IDs: `3A GELU T-a`; `3A SiLU T-b`; `3A Mish crossing`; `3A GELU bracket lo`.

## WP-26. Citations, main-text notation, the checker paragraph, and run populations (Track 4; for the submission)

Sources: `results/track4_writer_inputs.md` (the full Track 4 report, including the appendix-only symbol list and the
list of symbol clashes) and `src/track4_populations.py` → `track4_populations.json` (every population number below).

### 1. Six citations: verified against the proceedings or iclr.cc pages

**Result: all six are INCLUDED. None is excluded.**

| key | venue | official page | arXiv | differences to note |
|---|---|---|---|---|
| `lyu2024dichotomy` | ICLR 2024 | proceedings.iclr.cc (pp. 33897–33936) | 2311.18817 ("Published as a conference paper at ICLR 2024") | The proceedings give "Du, Simon" and "Lee, Jason"; arXiv gives "Du, Simon S." and "Lee, Jason D.". The entry uses the proceedings form, following WP-13's rule. Note that `references.bib` (`jin…`, line 203) spells the same authors "Simon Shaolei Du" and "Jason D. Lee", so choose one form for the whole bibliography. |
| `kunin2024getrich` | NeurIPS 2024 (vol. 37) | proceedings.neurips.cc (pp. 81157–81203, DOI 10.52202/079017-2580) | 2406.06158 ("NeurIPS 2024") | none |
| `glasgow2024sgd` | ICLR 2024 | proceedings.iclr.cc (pp. 52419–52430) | 2309.15111 | The title's "near-Optimal" and "the XOR problem" are lower-case on both pages, exactly as written. |
| `zhu2023minimalist` | ICLR 2023 | iclr.cc/virtual/2023/poster/11908 | 2210.03294 | No proceedings volume, so there are no pages. |
| `rubin2024grokking` | ICLR 2024 | proceedings.iclr.cc (pp. 23881–23904) | 2310.03789 | none |
| `nanda2023progress` | ICLR 2023 | iclr.cc/virtual/2023/poster/11385 (also listed as oral 12572) | 2301.05217 | The current arXiv abstract differs from the one on iclr.cc. The sentence uses only content that appears in both. iclr.cc labels it "top 25% paper"; per WP-13, do not cite any presentation label. |

**BibTeX conventions (the same as WP-13).**
- Fields are read from the proceedings page.
- `editor` and `url` are left out.
- The ICLR proceedings BibTeX has `volume = {2024}`, which is only the year repeated. WP-13's ICLR entries leave it
  out, so these do too.
- ICLR 2023 entries have no pages because there is no volume.

```bibtex
% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/909c8fef63e1cede406ce9e6794f99a2-Abstract-Conference.html
@inproceedings{lyu2024dichotomy,
  title     = {Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking},
  author    = {Lyu, Kaifeng and Jin, Jikai and Li, Zhiyuan and Du, Simon and Lee, Jason and Hu, Wei},
  booktitle = {International Conference on Learning Representations},
  pages     = {33897--33936},
  year      = {2024}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2024/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html
@inproceedings{kunin2024getrich,
  title     = {Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning},
  author    = {Kunin, Daniel and Ravent{\'o}s, Allan and Domin{\'e}, Cl{\'e}mentine and Chen, Feng and Klindt, David and Saxe, Andrew and Ganguli, Surya},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {81157--81203},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/079017-2580},
  year      = {2024}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html
@inproceedings{glasgow2024sgd,
  title     = {{SGD} Finds then Tunes Features in Two-Layer Neural Networks with near-Optimal Sample Complexity: A Case Study in the {XOR} problem},
  author    = {Glasgow, Margalit},
  booktitle = {International Conference on Learning Representations},
  pages     = {52419--52430},
  year      = {2024}
}

% verified: https://iclr.cc/virtual/2023/poster/11908 (proceedings.iclr.cc has no 2023 volume; arXiv 2210.03294 agrees)
@inproceedings{zhu2023minimalist,
  title     = {Understanding Edge-of-Stability Training Dynamics with a Minimalist Example},
  author    = {Zhu, Xingyu and Wang, Zixuan and Wang, Xiang and Zhou, Mo and Ge, Rong},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/682f87a8c306098ec8be29019bd76aa4-Abstract-Conference.html
@inproceedings{rubin2024grokking,
  title     = {Grokking as a First Order Phase Transition in Two Layer Networks},
  author    = {Rubin, Noa and Seroussi, Inbar and Ringel, Zohar},
  booktitle = {International Conference on Learning Representations},
  pages     = {23881--23904},
  year      = {2024}
}

% verified: https://iclr.cc/virtual/2023/poster/11385 (proceedings.iclr.cc has no 2023 volume; arXiv 2301.05217 agrees)
@inproceedings{nanda2023progress,
  title     = {Progress measures for grokking via mechanistic interpretability},
  author    = {Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

**One sentence each, in writer wording.** Each is based on the abstract on the official page.

- **Lyu et al. (2024).** They prove that homogeneous networks trained from large initialisation with small weight
  decay stay near a kernel predictor for a long time, then move sharply to a min-norm or max-margin predictor, which
  changes test accuracy abruptly. Their transition happens along a training trajectory. Ours is a static property of
  the training loss minimised at fixed output scale, certified by interval arithmetic, and training enters it only
  through a computable lag.
- **Kunin et al. (2024).** They derive exact solutions for a minimal model that moves between lazy and rich learning.
  These show that unbalanced layer-wise initialisation variances and learning rates set the degree of feature
  learning, through conserved quantities that shape the learning trajectory. Their solutions describe a training
  trajectory. Our threshold is a static, certified property of the loss at fixed output scale, and training enters it
  only through a computable lag.
- **Glasgow (2024).** For minibatch SGD on a two-layer ReLU network learning XOR, they prove two phases. First, a small
  network's neurons find features independently. Then SGD maintains and balances those features, and the few neurons
  that found them are amplified as their second-layer weights grow. We share the role of output-weight growth in making
  a hidden unit's feature pay off. We differ in locating the output scale at which correct placement becomes the loss's
  preferred solution, as a certified property of the loss rather than through an analysis of SGD's trajectory.
- **Zhu et al. (2023).** They build a simple objective that shows edge-of-stability behaviour and analyse its training
  dynamics rigorously in a large local region. As they do, we use a minimal model to make a training phenomenon exactly
  analysable. Our object, though, is a threshold of the training loss at fixed output scale, not the dynamics of
  gradient descent at large step size.
- **Rubin et al. (2024).** Using the adaptive-kernel theory of feature learning on teacher–student models, they show
  that after grokking the network is analogous to the mixed phase that follows a first-order phase transition, with
  internal representations sharply different from those before it. Our threshold is also a switch between two kinds
  of solution. Here it is between minimisers of a fixed-scale training loss for a single hidden unit, located by
  certified computation rather than by a kernel-limit theory.
- **Nanda et al. (2023).** They reverse-engineer the algorithm that small transformers learn for modular addition, and
  use it to define progress measures that change continuously before the grokking transition. We share the aim of
  explaining an abrupt change by a quantity that moves continuously before it: here, output scale relative to a
  certified threshold. In our case that quantity is computed from the loss, not read off trained weights.

**Do not say** that any of these papers studies a conditional threshold in output scale (the same rule as WP-13).

### 2. Main-text notation

| symbol | meaning | where defined |
|---|---|---|
| s | output scale: \|w₂\| at width 1; ‖w₂‖₁ at width 2 | math note §1, §10 |
| R | output scale in gap units, R = sĜ(a)/2 (certified Ĝ_cert) | math note §1; WP-7 |
| Ĝ(a) | certified maximum class gap over first-layer parameters | math note "Three objects"; WP-7 |
| G | worst-case class gap of the hidden unit on the continuous windows (G > 0: placed) | math note §10–§11; WP-12 |
| R\* ≡ R_glob | the conditional placement threshold: where the global minimiser of the loss at fixed output scale becomes placed (certified brackets) | math note "Three objects"; v4 Block 1 |
| R_solve | where that minimiser becomes sign-correct everywhere (certified brackets) | v4 Block 1; math note §2(b) |
| R_own | R_glob computed on a run's own training set (400 points, 200 per class), before training | v4 Block 1 (1d) |
| χ | timescale ratio (ṡ/s\*)/(ηλ_min(P^{1/2}HP^{1/2})); the tests use each run's measured ratio at crossing (ṡ/s_c), a factor 1 + r apart | math note §13.1, §13.3(i); WP-24 |
| κ(a) | lag constant in r = κ(a)·χ, with no free parameter; depends on the winding of b₁ | math note §13; WP-24 |

- **Clashes to resolve in the main text.** κ₀ (the limiting-cubic constant in Block 3's window design) needs another
  symbol there, for example ν₀. "A\*" is the limit switch in rescaled units, not R\*.
- Every other symbol is appendix only. The full list and all clashes (α, β, U, C, K, D, φ, σ, λ, ε) are in the Track 4
  report, §2.
- That report's notes 1–3 predate Track 1A's commit: κ(a) and χ are now defined in math note §13 and WP-24.

### 3. AI Use Statement: the independent checker (one paragraph, under 150 words)

> An independent checker, `verify_certificates.py`, re-verifies the certificates and shares no code with the
> certifying searches. It imports nothing from the project's source tree. It re-implements every objective
> (the profiled loss, the class gap and the limit problem) from its mathematical definition in ball arithmetic
> (python-flint/Arb, 80-bit), whereas the searches used float Lipschitz bounds, numpy outward rounding or mpmath
> intervals. Every decision is an Arb comparison, so rounding is enclosed. For exported certificates it reads only
> the certificate data files, each checked against a committed SHA-256 hash. Its tests include constructed cases it
> must reject. It verifies the finite-a placement brackets, the Ĝ(a) enclosures, the limit-switch bracket and its
> localisation, K = sup G₀ with its domain lemma, the Krawczyk boxes of the first-order calculation, and the
> positive-definite and ring certificates. The outer-exclusion certificates and the solve brackets (finite-a and
> limit) were not run; only the original searches support them.

- Every statement comes from the checker's docstring, `PREC = 80`, the manifest hash check, `certificate_audit.md` and
  WP-17's table.
- Keep "for exported certificates": the K, Krawczyk, positive-definite and ring checks take their inputs differently
  (Track 4 report, §3).
- The files do not say who wrote the checker or whether AI tools were used. The author adds that.

### 4. Run populations (a report, not a choice)

| population | runs | solved | placement failures | bias failures |
|---|---|---|---|---|
| 2,400 float32 | 2,400 | 430 | 1,664 | 306 |
| 2,400 float64 | 2,400 | 440 | 1,633 | 327 |
| 4,800 pooled | 4,800 | 870 | 3,297 | 633 |

- The float32 half is the width-1 sweep plus the refinement, run for run: 2,400
  matched, solve outcomes identical, and |w₂| identical (maximum difference 0.0).
- **The float64 half is not a re-run of the float32 runs at higher precision.** At the same (a, seed):
  - the sign of w₁ agrees in 48.6% of 2,400 pairs;
  - the failure class agrees in 72.8%;
  - the solve outcome agrees in 85.25%;
  - the median relative difference in |w₂| is 0.44 (relative to float32; the Track 4
    report's 0.50 used another denominator).

  These are two independent initialisation draws labelled by precision (the per-dtype RNG problem recorded in the
  retraction, commit 1abaf09).
- **Which main-text number uses which population:**
  - The decomposition figure and its caption use the 4,800 pooled, labelled "(200 seeds × 2 precisions)". That label
    describes the tags, not two precisions of the same runs.
  - The metric check (`metric_check.py`; caption and VERIFIED_NUMBERS §6) uses the 2,400 float32.
  - The a = 1.02 numbers in WP-12 (A2) and math note §11.1 use the 2,400 float32: 0/200 solved,
    median terminal |w₂| 1.85, maximum 3.92.
    - The float64 half gives 0/200, median 1.92, maximum
      3.94.
  - WP-12's onset values (1.60 at 2,000 steps; 1.18, 1.06 and 1.03 at 8k, 32k and 128k) are constants in
    `harsh_review_a.py` taken from the onset analyses, not from either population.
- **So the main text currently mixes the two populations.** Whichever is chosen, the decomposition caption should not
  say "2 precisions" as if the runs were paired.

**The population the main text should use (final round): the 2,400 float32 runs.**
- **Why:**
  - They are one initialisation draw per seed at one precision, so they have no hidden pairing.
  - They are exactly the width-1 sweep plus refinement.
  - The metric check, the a = 1.02 numbers and VERIFIED_NUMBERS §6 already use them.
- **What changes:** only the decomposition figure and caption. Use `results/figures/v5/decomposition_float32.pdf`
  (captions.md entry "decomposition_float32") in place of `decomposition.pdf`.
- The pooled 4,800 can appear once, as a robustness line: "an independent second draw of 2,400 runs gives 440 / 1,633 / 327".
- **Decomposition** (`track4_float32_decomposition.csv`): 430 solved, 1,664 placement failures and 306
  bias failures in 2,400 runs (200 per a):

| a | solved | placement failure | bias failure |
|---|---|---|---|
| 1.02 | 0 | 200 | 0 |
| 1.05 | 0 | 200 | 0 |
| 1.10 | 0 | 200 | 0 |
| 1.15 | 0 | 200 | 0 |
| 1.25 | 0 | 199 | 1 |
| 1.30 | 0 | 175 | 25 |
| 1.35 | 2 | 135 | 63 |
| 1.40 | 15 | 98 | 87 |
| 1.45 | 50 | 77 | 73 |
| 1.50 | 81 | 66 | 53 |
| 2.00 | 143 | 54 | 3 |
| 3.00 | 139 | 60 | 1 |
IDs: `T4 pooled counts`; `T4 sign agreement`; `T4 a=1.02 float32`.

## WP-27. Width 2: the lag law where the account applies, the stuck states, and the validity condition (Track 2; POST HOC; for the submission)

Producers: `src/width2_lag.py` → `width2_lag/`; `src/width2_basins.py` → `width2_basins/`; `src/t2c.py` → `t2c/`
(commit 764c9dc). **Everything here is POST HOC, and no verdict changes: T2-3 FAIL, T2-3b UNRESOLVED, T2-3c UNRESOLVED,
T2-3d FAIL.** 2C was not registered: the author's calibration rule found no admissible rule scale.

**Coordinator note (checked against `width2_lag/summary.csv`).** The three full-speed runs called "tracking" below have
crossing states 0.066 from their branch (median). That is just above the module's 0.05 on-branch
threshold, so the summary files list them as "off branch". Describe them as "the three full-speed runs with χ in width 1's
range", not as on-branch.

**Reconciliation of the two loss statements (coordinator, final round; `src/width2_reconcile.py` →
`width2_basins/reconcile.csv`, `reconcile_summary.json`; POST HOC).** Both statements below are exactly true. They are about
different objectives.
- **Population objective (symmetric windows, 800 points).** At R₂ = 0.003 and 0.01 the validated global conditional
  minimiser is placed (a cancelling pair; direct check). Every f_a stuck configuration, evaluated on the population
  objective at the same scale, lies above it: 190 of 190, by
  1.1e-05 to 2.8e-04 (median 3.5e-05).
- **Each run's own 400-point training set (search, not validated).** A placed minimiser was found for 58 of 265 stuck
  states, and in 37 of those 58 the stuck state has the lower loss.
- There is no validated population minimiser at R₂ = 0.03 and 0.1, so 75 stuck states have no
  population comparison.
- **Wording:** do not write "the stuck states are often lower in loss than any placed state" without "on the run's own
  training set". Write: "On the population objective the global minimiser at the same scale is placed and every stuck
  state lies above it (by 10⁻⁵–10⁻⁴); on the finite training sets the placed and stuck minima are nearly degenerate, and
  the stuck one is often lower."

The text below is the Track 2 writer input.

### 2B. Does the lag law hold at width 2 where the account applies? (POST HOC)

#### Method (fixed before any comparison was computed; in the module docstring)

- **Which runs.** Every crossing run of T2-3 (full speed, φ₂ = 1), T2-3b (φ₂ = 0.0093) and T2-3c / T2-3d
  (φ₂ = 0.01778). Asymmetric windows, a = 1.30, Δ = 0.4, each run on its own training set.
- **Replay.** Each run was replayed to its recorded crossing step.
- **Width-2 analogue of 1A's coordinates.**
  - Fast coordinates are z = (α₁, β₁, α₂, β₂, b), trained at η = 0.01.
  - The slow variables are the output weights v = (v₁, v₂). In the slowed arms both components move at φ₂η.
- **The tracked branch.** z*(v) is the stationary point in z at fixed v, found by damped Newton from the crossing
  state, with a positive-definite Hessian.
- **Following it.** Newton continuation (predictor–corrector, steps ≤ 2% of ‖v‖₁) along the run's own recorded
  v-path:
  - backward from the crossing if the branch is placed there;
  - forward along the crossing velocity if it is not.
- **The own-branch switch s\*.** The scale at which the branch's exact placement gap G₊ changes sign, bisected.
- **κ₂.** Computed as in 1A at the switch, with P = diag(1/(√v̂+ε)) from the run's own Adam state at the crossing.
  - Along the path, the gap's rate also includes the rotation of v's direction:
    ġ = ∇_zG·θ̇* + D_ṽG[ṽ̇].
  - With no rotation this is exactly 1A's formula.
  - The run's actual coordinates are used, so the 2π winding enters as κ_k does in 1A.
- **χ and the comparison.** χ = (ṡ/s\*)/(ηλ_min(P^{1/2}HP^{1/2})). The observed r_obs = s_c/s\* − 1 is compared
  with r_pred = κ₂χ, with no fit.
- **Secondary check.** The ray version (v = s·ṽ_c, 1A's formula literally) is kept in the parts files as
  `ray_*`.
- **Disclosed change.** The path version was made primary after the ray version was tried on three T2-3c runs. One
  of them had v's direction rotating at 0.88 of the scale's growth rate.

#### Results (`results/width2_lag/summary.csv`, `summary.json`, `chi_bins.csv`)

**Branch defined.**
- **Slowed arms: 191 of 230 crossers.**
- The 39 without a branch are all but one early crossers:
  - s_cross ≤ 0.18, median step 48, about 0.5 relaxation times after initialisation;
  - the statuses are placed-back-to-init 12, no PD branch at the crossing 8, jump 8, branch lost 6, no switch in
    range 5;
  - these crossings happen during the initial relaxation and are not tracking any branch.
- **T2-3: 75 of 79.**

**Width-1 reference** (`lag_law/compare_arms.csv`, `predictions.csv`). The lag law held in all 36 of 36 arms,
with per-run χ up to **0.064** (arm medians up to 0.025).

**Slowed arms, branch-defined runs with χ within the width-1 range (χ ≤ 0.064; n = 53 / 54 / 58 for b / c / d):**

| arm | median r_obs | median r_pred | per-run within max(0.01, 0.25\|pred\|) | median \|log(s_c/s\*)\| |
|---|---|---|---|---|
| T2-3b | −0.0049 | +0.00003 | 66% | 0.0059 |
| T2-3c | +0.00005 | +0.00005 | 83% | 0.0015 |
| T2-3d | −0.0012 | +0.00005 | 81% | 0.0024 |

- Pooled over the slowed arms: 165 runs, 77% within, median |log(s_c/s\*)| = 0.0033.
- The per-arm medians are within tolerance in every arm, but only through the 0.01 floor.

**How weak this test is (say this).**
- In the slowed arms the predicted lag is tiny, |r_pred| ≤ 0.005. The late crossers (s ≈ 4.2–6.6, 133 of 191, the majority)
  have χ ≈ 1e−5.
- The mid crossers (s ≈ 0.05–0.49) have κ₂ ≈ 0.01–0.1: their branch's gap is driven mainly by the rotation of v, so a
  lag in time is a small lag in s.
- **So the slowed arms test the zero-lag limit:** the run crosses at its own tracked branch's switch to within about
  1%. They do not test κ quantitatively.
- The mid crossers on the branch (n = 9 / 9 / 11) have median obs 0.0011 / 0.0016 / 0.0022 against pred
  0.0002 / 0.0010 / 0.0009. That is within the floor, but obs/pred is 1.6–6.6, not about 1.

**A systematic departure not predicted by the lag law.**
- The late slowed crossers cross slightly **before** their branch's switch: median r_obs −0.0066 (b), −0.0005 (c),
  −0.0038 (d), against pred +0.00005.
- Their crossing states sit about 1e−3 from the branch (dist_c median 0.0013 / 0.0002 / 0.0008).
- A plausible reading, **not tested**: Adam's step-scale fluctuation about a nearly flat branch lets the gap first
  turn positive early.
- It is largest in T2-3b, the slowest arm, where the gap grows most slowly per step.
- This is what drives the within-tolerance fraction below 100% (66–83%).

**Full speed (T2-3).**
- **χ at the crossing:** χ_c = (ṡ/s_c)/(ηλ_min,c) ranges 0.0031–3.05, median 0.99 (77 runs with a PD branch at the
  crossing).
- With s\* in place of s_c: χ = 0.0032–1.0e5, median 1.73 (75 branch-defined runs).
- 72 of the 75 branch-defined runs have χ ≥ 0.41. They cross about 1 relaxation time after initialisation (median
  step·ηλ_min = 1.0), far from the branch they are assigned (median ‖z_c − z\*‖∞ = 1.75). They do not track it:
  median r_obs 1.22 against pred 0.27, and 4% within.
- **3 T2-3 runs do track** (seeds 600002, 600058, 600073):
  - late crossings (s_c = 4.8–6.2, 2,100–2,600 steps, about 250 relaxation times);
  - χ = 0.0032–0.0045;
  - two units of nearly equal weight share (0.47–0.49);
  - κ₂ = 7.47–8.25, close to the width-1 κ(1.30) = 7.61;
  - **r_obs / r_pred = 0.0359/0.0345, 0.0356/0.0342, 0.0245/0.0236, i.e. obs/pred = 1.04, 1.04, 1.04.**
  - This is the only width-2 case where the predicted lag exceeds the 0.01 floor. There the no-fit prediction agrees
    to 4%.

**χ bins, all arms, branch-defined runs** (`chi_bins.csv`; per-run fraction within tolerance):

| χ bin | n | fraction within |
|---|---|---|
| (0, 1e−4] | 133 | 77% |
| (1e−4, 0.01] | 5 | 100% |
| (0.01, 0.064] | 30 | 73% |
| (0.064, 0.3] | 19 | 32% |
| (0.3, 3] | 51 | 6% |
| > 3 | 28 | 0% |

**Verdict (POST HOC, descriptive).**
- Where the account applies (χ in width 1's range and the run relaxed onto a branch), width-2 crossings occur at the
  run's own tracked branch's switch, to about 1% (77% of runs within max(0.01, 0.25|pred|)).
- In the 3 runs where the lag is large enough to measure, the no-fit lag law holds (obs/pred = 1.04).
- The slowed arms confirm only the zero-lag limit.
- Above χ ≈ 0.06–0.3 the account fails, and T2-3 sits at χ ≈ 1.

---------------------------------------------------------------------------------------------------------------------

### 2A. Are the stuck states basins? (POST HOC)

#### Method (fixed before any endpoint was evaluated; in the `src/width2_basins.py` docstring)

- **Endpoints.** Every Track 7 no-gating replay endpoint at the scored step H = 16,000 that is **unplaced** (exact
  G₊ ≤ 0) and **stationary** by Track 7's criterion (max|∇L|/‖v‖₁ ≤ 1e−6).
  - That is 281 endpoints: f1.30 125, f1.50 140, tanh 16.
  - They span all four R₂ levels and both variants.
  - Each was re-created by deterministic replay; all 281 reproduced to 1e−9.
- **Polish.** Newton on the fixed-scale conditional loss.
- **Reduced Hessian.** The Hessian in (θ, v, b), restricted to the 6 free directions of the held ℓ₁ sphere.
- **Classes:**
  - strict minimum (PD);
  - saddle;
  - duplicate-unit (Morse–Bott) minimum: the two units are copies up to orientation, the only zero mode is the weight
    transfer between them (an exact symmetry), and the rest is PD;
  - undetermined.
- **Loss gap.** The endpoint's loss minus that of the best **placed** conditional minimiser found at the same s on the
  run's own training set.
  - Starts: the population direct-check minimiser and its images, plus a 100-restart search with the 30 lowest
    candidates Newton-polished.
  - This is the best placed minimiser found, not a certified global one.
  - Convergence check, added after the first 136 endpoints and disclosed: all 50 (act, seed, R₂) groups with a placed
    minimiser were re-run and polished to 2,000 Newton iterations. Every best placed point had max|∇| ≤ 2e−9, stayed
    placed and did not move (`placed_check.csv`).

#### Results (`results/width2_basins/summary.csv`, `by_scale.csv`, `endpoints.csv`, `placed_check.csv`)

**f_a (f1.30 and f1.50), 265 endpoints.**
- **All 265 are duplicate-unit configurations.** The two units are copies up to orientation, each with weight share
  0.5 and cancellation index 1. The function is a single width-1 unit with output weight s, embedded in width 2.
- **The reduced Hessian is PD in 0 of 265.** Each has exactly one zero eigenvalue (|λ|/λ_max ≤ 2e−12). Its
  eigenvector is the weight transfer between the copies, along which the function is exactly unchanged.
- **On the complement the Hessian is PD in 265 of 265** (λ_min/λ_max median 0.017). No endpoint is a saddle.
- **So 100% are conditional local minima of the fixed-scale loss, of Morse–Bott type:** a line of equal-loss minima,
  not isolated ones. They are basins. The weight-transfer direction is flat, but it leads only to other points of the
  same line.

**Loss gap to the best placed minimiser (same s, own training set).**
- A placed conditional minimiser was found for only **58 of 265** f_a endpoints:
  - R₂ = 0.003: 4 of 101;
  - R₂ = 0.01: 13 of 89;
  - R₂ = 0.03: 36 of 70;
  - R₂ = 0.1: 5 of 5.
- Where one was found, the stuck state is **lower** in 37 of 58 (median gap −0.0027; range −0.019 to +0.0018).
- By scale:
  - R₂ = 0.003: +1e−5 to +2e−5 (4 cases);
  - R₂ = 0.01: −0.0018 (f1.30) and −0.0030 (f1.50);
  - R₂ = 0.03: −0.0049 (f1.30) and +0.0006 (f1.50);
  - R₂ = 0.1: −0.015 and −0.016.
- The stuck state itself is the lowest loss found in **108 of 265**. A placed configuration is the lowest found in
  only **18 of 265**.
- **So on the runs' own training sets the stuck states are not high-loss traps.** They are usually as low as, or lower
  than, any placed configuration found at that scale.
- **Contrast with the population** (symmetric), where the validated direct check found the placed cancelling pair
  as the global minimiser at R₂ = 0.003 and 0.01.
- **Not tested:** that finite-sample asymmetry of the 400-point training sets is the cause.

**tanh, 16 endpoints (descriptive).**
- None is a duplicate.
- Polishing moved them far (median max-coordinate move 2.8), so they were not near stationary points.
- Afterwards: 9 saddles and 7 undetermined, **none a local minimum**.

#### Writer section, 2A

**Say:**
- (POST HOC) Every stationary unplaced endpoint of the f_a no-gating test (265 of 265) is a conditional local minimum
  of the fixed-scale loss.
  - Each is a width-1 unit duplicated across the two hidden units.
  - Its reduced Hessian has one exact zero mode, the weight transfer between the copies, and is positive definite on
    the rest (a Morse–Bott line of minima).
  - So the stuck states are basins, which is why training at fixed scale does not leave them.
- (POST HOC) At the same scale on the run's own training set, a placed minimiser was found for only 58 of these
  endpoints. In 37 of those 58 the stuck state has the lower loss. The stuck state is the lowest found in 108 of 265.
- The reduced Hessian is never strictly positive definite: 0 of 265, because of the exact symmetry.

**Do not say:**
- "the stuck states are strict (isolated) local minima";
- "the stuck states are suboptimal / higher-loss traps" or "the placed minimiser is the global minimiser on the
  training set". On the own training sets the opposite is more common, and the placed search is not certified;
- anything that changes Track 7's registered outcome;
- anything about tanh beyond "no local minima among its 16 stationary-by-criterion endpoints".

---------------------------------------------------------------------------------------------------------------------

### 2C. The validity condition, and the registered test (NOT registered: the calibration rule stopped it)

#### Validity condition, from the data (`width2_lag/summary.json`, `chi_bins.csv`; `lag_law/compare_arms.csv`)

- **Width 1:** tracking held in all 36 arms, at per-run χ up to 0.064 (arm medians ≤ 0.025).
- **Width 2, where it held:**
  - branch-defined runs with χ ≤ 0.064 are 73–100% within tolerance per bin (overall 77%);
  - with 0.064 < χ ≤ 0.3, 32%;
  - with 0.3 < χ ≤ 3, 6%;
  - above 3, 0%.
- **Width 2, where it failed:** at full speed (T2-3), χ at the crossing has median 0.99 (range 0.0031–3.05), and
  72 of 75 runs are at χ ≥ 0.41.
- **The condition as stated:** χ ≲ 0.06, and the crossing comes after the run has relaxed onto a branch.
  - Branch-defined tracking runs had ≥ 6.8 relaxation times before the crossing.
  - The runs with no branch crossed at a median 0.5 relaxation times.
  - The early slowed crossers fail this second part even when their χ is small. Their crossing is part of the initial
    relaxation into a basin that is already placed (12 are placed along their whole path back to initialisation).

#### Calibration of the rule scale (author's binding amendment), BEFORE any registration

**Rule and definitions** (fixed in the `src/t2c.py` docstring and tests before computing):
- The candidate rule scale s_r lies on the grid 10^(−2.5 + k/8).
- A run is classified at s_r at the first passage of its ‖v‖₁ through s_r, and only if that passage is strictly before
  its crossing.
- **"Final":** the basin found there (the fixed-v Newton branch point), continued along the run's own path, ends at
  its crossing branch.
- **Choice:** the earliest s_r with the final fraction ≥ 0.95 **and** s_r < the earliest crossing observed.

**Result** (`results/t2c/calibration.json`): 239 runs (T2-3b, T2-3c, T2-3d, not placed at step 0).
- The earliest crossing is **s = 0.0164**.
- Under matched initialisation the runs start at ‖v‖₁ ≈ 0.001–0.081 (median 0.041).
- So below 0.0164, at most **7.5%** of runs ever pass through s_r before crossing.
- At larger s_r, runs that cross first have no classification. The fraction with any classification peaks at **0.778
  (s_r = 0.075)** and falls to 0.60 by s_r = 0.56, as the early crossers (40%) drop out.
- The available fraction is an upper bound on the final fraction. **No scale can be final in ≥ 95% of runs, and
  none below the earliest crossing comes close.**
- **Finality where it was computed** (`results/t2c/finality_runs.csv`):
  - at s_r = 0.075 (the peak of availability), 131 of 239 runs are final (0.548). Of the 177 classifiable crossers,
    28 continued to their crossing on a different branch and 18 hit a jump or had no basin at the rule scale;
  - at s_r = 0.0133 (below the earliest crossing), 4 of 239 are final (0.017).
- **Decision: STOP.** Nothing was registered, no fresh seed was trained, and no seed range was consumed.

**Possible redesign, for the author only** (not done, not registered):
- Classify each run's basin at step 0 (the fixed-v Newton branch point from the initial state), which is before any
  crossing for every run not placed at step 0.
- Predict the crossing from that basin's path-continued switch.
- The 2B data suggest this would separate the relaxation crossers (placed basins) from the trackers. It is a different
  rule from the one the author fixed.

---------------------------------------------------------------------------------------------------------------------

### Writer section

#### 2B

**Say:**
- (POST HOC) At width 2, in the regime where the account applies, a run crosses at the switch of the branch it
  tracks:
  - the regime is χ within width 1's range (≤ 0.06) and the run relaxed onto a branch;
  - the branch is followed by Newton continuation along the run's own output-weight path;
  - 77% of 165 slowed runs are within max(0.01, 0.25|pred|), with median |log(s_c/s\*)| = 0.003.
- (POST HOC) The three full-speed runs that track a branch have a measurable lag. There the no-fit width-1 formula
  (κ₂ ≈ 7.5–8.2, computed on the run's own branch) predicts it to 4% (obs/pred 1.04).
- The slowed arms test only the zero-lag limit: the predicted lags there are ≤ 0.005, below the 0.01 floor.
- The full-speed arm T2-3 sits at χ ≈ 1 (0.003–3), about 15–30× beyond width 1's range. There runs cross within about one
  relaxation time, far from any branch. That is why T2-3 FAILED; the verdict stands.

**Do not say:**
- that the lag law is "confirmed at width 2" in general, or that κ is validated by the slowed arms (their predicted
  lags are below resolution);
- that T2-3, T2-3b, T2-3c or T2-3d change verdict;
- that the late slowed crossers' early crossing (median r_obs −0.0005 to −0.0066) is explained; it is not tested;
- "own-branch switch" as if it were the population threshold, or the width-1 own-sample threshold. It is a per-run,
  per-path quantity computed after the fact.

#### 2A (see the 2A section above)

#### 2C

**Say:**
- The validity condition, from the data: tracking holds up to χ ≈ 0.06 at both widths. At width 2 it also needs the
  run to have relaxed onto a branch before crossing.
- About 17% of slowed runs (39/230) cross during the initial relaxation (≈ 0.5 relaxation times) into a basin that
  is already placed. The account does not apply to them.
- The registered prospective test was not run: the pre-set calibration rule found no admissible rule scale. At most
  78% of runs are ever classifiable before crossing at any scale, and at most 7.5% below the earliest crossing
  (0.016).

**Do not say:**
- that a prospective width-2 basin test was registered or passed;
- that the early-scale rule "fails". It was never applied.

#### If asked about the other possible outcomes (so wording is ready)
- Had 2C been registered and PASSED: "a fresh-seed test predicted each run's crossing from its early basin". This
  **did not happen**, so do not use it.
- Had it FAILED: the FAIL would have been reported beside T2-3's FAIL. **Not applicable.**

---------------------------------------------------------------------------------------------------------------------

### Track 2, final round: two-class prospective predictor (GATED). The gate FAILED, so nothing was registered.

**Label:** a new predictor, designed from the diagnosis of earlier failures. T2-3 stays FAIL, T2-3b and T2-3c stay
UNRESOLVED, and T2-3d stays FAIL.

**Producer:** `src/t2g.py`, with tests in `tests/test_t2g.py`. The tests exercise every gate and criterion on
constructed pass and fail cases.

**Outputs:**
- `results/t2g/classify_<arm>.csv`, `own_<arm>.csv`, `gate_runs.csv`;
- `results/t2g/gate.json` (the gate);
- `results/t2g/gate_misses.json` (descriptive only, written after the gate).

#### Committed before the classifier was evaluated

Commit `7b8195a`, with `results/t2g_registration.md`.

**Classifier.** It uses only the initial state and a pre-computation, never training.
- A 2,000-step fixed-v Adam relaxation of the hidden coordinates from the initial state.
- If it is placed at any step, the class is EARLY and the predicted crossing is s₀, the initial ‖v‖₁.
- Otherwise the relaxed basin is followed by Newton continuation along the initial output-weight ray up to s = 1.0.
  If it turns placed at s_sw, the class is EARLY and the predicted crossing is s_sw.
- Otherwise the class is LATE and the predicted crossing is the width-1 own-sample threshold (T2-3d's method).

**Ground truth.** A run is EARLY iff it crossed at s_cross < 1.0.

**Gate.** Accuracy ≥ 0.95, and median |log error| ≤ 0.10 over the predicted-LATE runs.

**Step 2 was fully specified in advance:** fresh seeds 870,000–870,079, criteria (i)–(iii) and validity.

#### Gate result (POST HOC, 239 existing slowed runs; `results/t2g/gate.json`): **FAIL**

**Accuracy is 0.812, against ≥ 0.95 required.** Per arm: T2-3b 0.825 (66 of 80), T2-3c 0.8625 (69 of 80), T2-3d 0.747 (59 of 79).

| | true EARLY | true LATE |
|---|---|---|
| predicted EARLY | 54 | 3 |
| predicted LATE | 42 | 140 |

**The late-class criterion passes on its own.** Over the 174 predicted-LATE crossing runs, the median
|log(s_cross/s_w1,own)| is **0.014**, against ≤ 0.10 required.

**The gate fails on accuracy alone.** Following the rule, step 2 was not registered, the fresh seeds were not
touched, and no fresh-seed run was trained.

#### Descriptive, after the gate (`results/t2g/gate_misses.json`; no rule was changed)

**The 42 missed early crossers:**
- They crossed at s = 0.016–0.49 (median 0.13), at step 17–1,904 (median 347).
- The classifier found each one's initial basin unplaced all the way up to s = 1 along the initial ray.
- In Track 2B, 35 of them have a branch switch along their own recorded v-path, and 24 were on that branch at the
  crossing (dist_c ≤ 0.05).
- So most are branch-trackers whose switch comes from how v grows and rotates during training. The initial ray does
  not see this.

**The 54 caught early crossers:**
- 42 were placed during the fixed-v relaxation, and 12 had a ray switch below 1.
- They crossed at s = 0.018–0.32 (median 0.061).
- Their predicted early scale (s₀ or s_sw) has median |log error| 0.37.

**False EARLY: 3 runs.** One never crossed; two crossed late, at 5.84 and 5.70.

#### Writer section

**Say:**
- (POST HOC gate) A two-class predictor computed entirely before training was designed from the diagnosis of earlier
  failures.
  - Its late class predicts the late crossers to a median |log error| of 0.014.
  - But it identifies the early crossers with only 81% accuracy (54 of 96 found), below the pre-set 95% gate.
  - So no prospective width-2 test was registered.
- Whether a width-2 run crosses early is only partly set by its initial basin:
  - about 44% of early crossers (42 of 96) are missed from the initial state;
  - most of those track a branch whose switch is driven by how the output weights grow and rotate during training.

**Do not say:**
- that a prospective width-2 prediction was registered, run or passed;
- that the early crossers "are identifiable before training";
- that the late-class accuracy (0.014) is a prospective result. It is post hoc on existing runs, and the gate failed;
- anything that changes T2-3 (FAIL), T2-3b / T2-3c (UNRESOLVED) or T2-3d (FAIL).

**If the gate had passed** (it did not), step 2's registered PASS or FAIL would have been reported beside T2-3d's FAIL.
**Not applicable.**

IDs: `T2 2B slowed within`; `T2 2B T2-3 three runs obs/pred`; `T2 2A Morse-Bott`; `T2 2C STOP`.

## WP-28. Higher input dimension: the band task in R^d, d = 2 and 4 (Track 3B; for the submission)

Producer: `src/band_rd.py` → `band_rd/`. Registration: `results/band_rd_registration.md` (f19535b). Tests:
`tests/test_band_rd.py`. The per-step traces (`band_rd/traces_*.npz`, 135 MB) are regenerated by
`python -m src.band_rd run`; their SHA-256 are in `band_rd/traces_sha256.txt`. The text below is the Track 3B writer
input, sections 1–6.

### 1. What was committed before any comparison

| commit | content |
|---|---|
| f19535b | REGISTRATION: design, proof that the threshold carries over, population check (validated, not certified), seeds, pilot disclosure, budget 64,000, rules P1 / P2a / P2b, validity checks, descriptives, tests. Frozen inputs: `width1_reference.json` (SHA-256 da0f85fd…), `own_x1_frozen.csv` (SHA-256 b4c9abcb…, 80/80 defined), batching validation (max difference 7e−15). |
| ca62e36 | RESULT: runs, scoring, registered verdicts and descriptives. |
| e51c0bb | POST HOC (labelled): residual against each run's own R^d branch switch. |

Disclosed ordering caveat. The design was fixed in code before any d > 1 training, but the registration document was
written after a 12-seed pilot, which recorded crossing counts and steps but no crossing scales. The pilot changed only
the budget (64,000) and led to adding the secondary arm, because ≥ 90% crossing turned out not to be reachable at
d = 4.

### 2. Design (short)

- **Features.** x₁ is exactly the width-1 task: `fold1d.make_data(200, seed)`, 400 points. The coordinates
  x₂…x_d are i.i.d. U(−2, 2) and independent of the label.
- **Model and training.** A single unit z = w₂ f_a(w·x + b₁) + b₂ with w ∈ R^d. Initialisation is U(−1, 1) per
  coordinate, and its first four draws equal the width-1 initialisation of the same seed. Training is Adam at lr 0.01,
  full batch.
- **Placement gap.** The gap is taken on the continuous support. Every class window is widened by N = 2‖w_noise‖₁,
  and extrema are exact. When w_noise = 0 it equals the width-1 gap bit for bit (tested).
- **The threshold carries over (proof).** The R^d population loss equals E_ξ L₁(w₁, b₁ + ξ, b₂), where
  ξ = w_noise·x_noise is independent of the label. This is at least L₁*, with equality only when w_noise = 0. So the
  conditional minimiser has w_noise = 0, and the certified width-1 bracket carries over exactly: a = 1.30 in
  (4.95, 4.9625], a = 1.50 in (2.525, 2.5375].
- **Numeric check** (validated, not certified; `popcheck_summary.csv`). At both bracket ends, for d = 2 and d = 4:
  - no start ends below L₁*;
  - every start that reaches L₁* has ‖w_noise‖ ≤ 5e−12;
  - the noise Hessian block equals (4/3)·∂²L/∂b₁² to within 5e−15;
  - the profile in ‖w_noise‖ increases strictly;
  - there is a higher stationary point carrying a noise weight: ‖w_noise‖ ≈ 0.64–0.84, loss 0.37 above L₁*.

### 3. Registered results (primary: 40 seeds 880,000–880,039 per cell; `verdicts.csv`, `verdicts.json`)

| d | a | crossings | P1: fraction s_c ≥ s_lo (≥ 0.90) | P2a: median r_pop, width-1 IQR | P2b: median r_own vs pred κχ (tol) |
|---|---|---|---|---|---|
| 2 | 1.30 | 36/40 | 0.972 **PASS** | 0.182 vs [0.035, 0.154] **FAIL (above)** | 0.147 vs 0.024 (0.010) **FAIL** |
| 2 | 1.50 | 36/40 | 0.972 **PASS** | 0.207 vs [0.072, 0.185] **FAIL (above)** | 0.187 vs 0.049 (0.012) **FAIL** |
| 4 | 1.30 | 29/40 | **UNRESOLVED** (< 30) | **UNRESOLVED** | **UNRESOLVED** |
| 4 | 1.50 | 29/40 | **UNRESOLVED** | **UNRESOLVED** | **UNRESOLVED** |

**Secondary arm** (registered; 120 seeds pooled; P1 and P2a only; reported beside the primary, not in place of it):

| d | a | crossings | P1 | P2a |
|---|---|---|---|---|
| 2 | 1.30 | 112/120 | 0.946 PASS | 0.189 FAIL (above) |
| 2 | 1.50 | 112/120 | 0.964 PASS | 0.226 FAIL (above) |
| 4 | 1.30 | 90/120 | 1.000 PASS | 0.528 FAIL (above) |
| 4 | 1.50 | 93/120 | 1.000 PASS | 0.553 FAIL (above) |

**Validity.** The recomputed R^d gap was > 0 at every recorded crossing. Every branch converged, so every d = 2
crossing run was usable for P2b. No run was placed at initialisation.

**Width-1 control** (descriptive, not scored; d = 1, same seeds, same code). All 40 runs crossed in each cell.
- P1 fractions: 0.875 at a = 1.30 (it would fail P1, as the width-1 reference 0.842 anticipated) and 0.975 at a = 1.50.
- Median r_pop: 0.055 and 0.088, both inside the IQR.
- Lag law: r_own 0.0298 vs pred 0.0289, and 0.0622 vs 0.0626, both within tolerance.

So on these fresh seeds the width-1 lag law reproduces, and the R^d excess is not a seed effect.

### 4. Registered descriptives: how the unit finds the direction (`descriptives.csv`; primary cells unless noted)

- **The noise weight collapses early.** ρ = ‖w_noise‖₂/|w₁|:
  - at initialisation, median 1.23 (d = 2) and 2.29 (d = 4);
  - at |w₂| = 0.5·s_mid, median 0.014 (d = 2) and 0.028–0.030 (d = 4);
  - at the crossing, median 0.012 (d = 2) and 0.020–0.021 (d = 4).

  In every crossing run ρ falls below 0.1 before the crossing, at a median 3–9% of the crossing time.
- **No run crosses with a non-negligible noise weight.** Zero runs have ρ_c ≥ 0.05; the largest ρ_c is 0.034 in the
  primary cells and 0.048 in the pooled cells.
- **The residual noise weight still matters.** Its widening at the crossing is a median 3% (d = 2) and 7–8% (d = 4) of
  the projected inner half-width. Every crossing comes after the x₁ direction alone is placed (G1 > 0), never on the
  same step:
  - d = 2: median delay 295 / 143 steps, +13% in scale;
  - d = 4: 770 / 361 steps, +36–39%.
- **Non-crossing runs are noise-dominated.** Every one ends with ρ > 1: 4 of 40 at d = 2 and 11 of 40 at d = 4. Their
  median final values are ‖w_noise‖ ≈ 2.1–2.3, |w₁| ≈ 0.5–1.4 and |w₂| ≈ 0.06–0.08. This matches the noise-carrying
  stationary point found in the population check. Pooled: 8/120 at d = 2 and 30 and 27 of 120 at d = 4.
- **Windings and branch geometry match width 1.**
  - Windings as in width 1: k = −1 at a = 1.30 and k = 0 at a = 1.50, with one exception per d = 2 cell.
  - The branch's own w_noise has median norm 0.010–0.021.
  - λ_min(full)/λ_min(signal block) = 0.999, so the noise block does not set the relaxation rate.

### 5. POST HOC (labelled; after the registered result; `posthoc_summary.csv`, `posthoc_scored_runs.csv`)

For each crossing run, the switch of its own-sample R^d branch was computed: continuation in all hidden coordinates,
using the R^d gap.

- **The R^d own-sample switch sits above the x₁-sample own threshold.** Median ratio 1.118–1.120 at d = 2 and
  1.401–1.412 at d = 4. The x₁-direction (G1) switch of the same branch equals the x₁ own threshold (ratio
  0.9996–0.9997).
- **Measured against that switch, the lag law holds in every cell,** within the registered tolerance (obs/pred in
  brackets):

  | d | a = 1.30: median r_branch vs pred | a = 1.50: median r_branch vs pred |
  |---|---|---|
  | 2 | 0.025 vs 0.024 (1.06) | 0.054 vs 0.049 (1.10) |
  | 4 | 0.019 vs 0.017 (1.09) | 0.038 vs 0.036 (1.07) |
  | 4, pooled | 0.014 vs 0.017 (0.84) | 0.033 vs 0.034 (0.96) |
  | 1, control | 0.030 vs 0.029 | 0.063 vs 0.063 |

- **Interpretation (post hoc).** The excess residual in R^d is a static finite-sample offset. On 400 points the loss
  minimiser keeps a small noise weight, and on the continuous support that weight widens the class windows and raises
  the sample's own threshold. The dynamic lag behind that threshold is width-1-sized and follows κχ. This decomposition
  was not registered.

### 6. Writer section

**Outcome: P1 at d = 2 (PASS, both a; secondary PASS in all four cells)**
- Say: "In R² with a label-independent noise coordinate, 97% of crossing runs placed the unit at or above the certified
  width-1 threshold (35/36 at both a; the pooled 120 seeds: 95% and 96%; d = 4 pooled: 100%)."
- Do not say: that P1 passed at d = 4 in the registered primary arm. It was UNRESOLVED with 29 crossings; the 100%
  figure is the secondary arm.

**Outcome: P2a and P2b at d = 2 (FAIL, above), P2a in the secondary arm (FAIL, above, all four cells)**
- Say: "The registered prediction that the crossing lag stays in the width-1 range failed. Crossings in R^d came later.
  The median residual against the population threshold was 0.18–0.21 at d = 2 (width-1 upper quartile 0.15 / 0.18)
  and 0.53–0.55 at d = 4 (pooled). Against the x₁-sample own threshold, the median residual was 0.15 / 0.19, against a
  predicted lag of 0.024 / 0.049."
- Say (clearly labelled post hoc): "A post hoc decomposition attributes the excess to the finite sample's own R^d
  threshold, not to a longer lag. The sample minimiser keeps a noise weight of about 1–2% of |w₁|, which widens the
  class windows on the continuous support and raises the sample threshold by 12% (d = 2) and 40% (d = 4). Measured from
  that threshold, the lag matches the width-1 lag law (obs/pred 0.84–1.10 across cells)."
- Do not say: "training crosses above the threshold with a lag in the width-1 range". That was the registered claim, and
  it failed.
- Do not say: that the lag law was confirmed in R^d as a registered result. The branch-switch comparison is post hoc.
- Do not say: that the conditional threshold moves in R^d. The population threshold provably does not (§2). What moves
  is the finite-sample own threshold.
- Do not present the post hoc offset as a law in d or n. Only d = 2, d = 4 and n = 400 were measured.

**Outcome: d = 4 primary UNRESOLVED**
- Say: "At d = 4 the registered primary cells were UNRESOLVED (29 of 40 crossings, below the registered 30). A
  registered secondary arm of 120 seeds crossed in 90 and 93 runs."
- Say: "About a quarter of d = 4 runs (11/40; 30 and 27 of 120) never place the unit. They converge to a noise-dominated
  state (‖w_noise‖ > |w₁|) that matches a noise-direction stationary point of the population loss."
- Do not say: that a longer budget would have resolved d = 4. The pilot ran to 128,000 steps without further crossings.

**Outcome: how the unit finds the direction (descriptive)**
- Say: "The unit finds the x₁ direction early. The noise-to-signal weight ratio falls from about 1–2 at initialisation
  to about 0.01–0.03 by half the threshold scale, and below 0.1 within the first 3–9% of the time to crossing. No run
  crossed with a noise ratio of 5% or more."
- Say: "Runs that fail to find the direction never cross: 4/40 at d = 2 and 11/40 at d = 4 end noise-dominated."
- Do not say: that the noise weight goes to zero before the crossing. It settles at the finite-sample value (about
  0.01–0.02), and that residual is what delays the crossing.

**On the reviewer's inner-ball-versus-shell task.**
- Do not say: that this track ran or tested it. It replaces that task with the band task in R^d, because a single unit
  cannot solve inner-ball-versus-shell: projections of the shell cover the ball's interval.

IDs: `3B P1 d=2`; `3B P2a d=2`; `3B P2b d=2`; `3B d=4 UNRESOLVED`; `3B post hoc obs/pred range`.

## WP-30. A decisive test of the first-order coefficient c₁ (Track 3, final round; registered follow-up; for the submission)

Producer: `src/c1_followup.py` → `c1_followup_*`. Registration: `results/c1_followup_registration.md` (e233ef5), committed
before any added certificate was computed. The original test stays INCONCLUSIVE as registered. Independent Arb check: 3
of the 10 added bracket-end certificates (a = 1.12 both ends, a = 1.11 upper) pass; the other 7, and the rescaled Ĝ, are
not independently checked. The decisive points are at ε = 0.08–0.12, not ε ≤ 0.04. The text below is the Track 3 writer
input.

### 1. What was committed before computing (`e233ef5`)

- **Added a**: 1.08, 1.09, 1.10, 1.11, 1.12.
- **Bracket procedure**: the registered one, unchanged.
  - Start at A*/ε^{3/2}, step outward by 1%, bisect to 2e−4, stop at the first unresolved midpoint.
  - `conditional_certified.evaluate`, tolerance 1e−7 tightened to 1e−9.
- **Ĝ(a)**: the rescaled certified search, Amendment 2's procedure.
- **Estimator**: `first_order._feasible_c1`, unchanged. Its inputs are the four original registered brackets
  (frozen, SHA-256) plus the added ones.
- **Validity and criterion**:
  - valid iff the feasible width is ≤ 0.1, otherwise INVALID/UNRESOLVED;
  - PASS iff C meets the derived c₁ ∈ [0.2852300, 0.2852303];
  - otherwise FAIL, including when C is empty.
- **Drop rule**: an added a that does not certify is dropped.
- **Checker rule**: an a whose checked endpoint certificate fails the independent checker is removed, and the
  verdict is recomputed without it.
- **Tests**: the scoring rule and the validity check were exercised on constructed pass, fail, no-law and
  invalid cases, and the original four alone reproduce width 0.10498. The tests passed before any added
  certificate was computed.

### 2. Why these ε (Step 1; `c1_followup_model.json`, `c1_followup_timing_a1.0550.json`)

- **The bracket's resolution is bounded below, whatever ε is.**
  - At the switch, the certified minima m₋ and m₊ meet tangentially: m₋ − m₊ ≈ 0.0215·d|d|, with
    d = s/s* − 1.
  - At tolerance 1e−9 a status therefore resolves only for |d| ≳ 1.75e−4.
  - So the smallest bracket the registered bisection can reach is 0.01/16 = 6.25e−4 relative in s. The
    registered 2e−4 target is unreachable.
- **Small ε cannot reach the design goal.** With that floor, added points at ε ≤ 0.04 cannot bring the
  feasible width below 0.05: Monte Carlo median 0.069, P(< 0.05) = 0.15.
- **ε ≈ 0.1 can.** There the ±ε³ allowance is still small relative to the signal: the projection for
  ε = 0.08–0.12 had median 0.042 and max 0.045.
- **The law holds at these ε.** The two-term law with the small-ε c₂ (−0.13) predicts R(1.30) = 0.21319,
  inside the certified (0.21310, 0.21364]. So ε ≤ 0.12 lies well inside the |c₃| ≤ 1 allowance.
- **Timing**: one full certified threshold at a = 1.055 (not in the set, not used) took 78 s at 0.75 GB.

### 3. The certificates

All five added a certified, so none was dropped.

| a | s bracket | rel. width | Ĝ(a) | R_glob(a) interval | evaluations | wall s |
|---|---|---:|---|---|---:|---:|
| 1.08 | [31.837918, 31.91751275] | 2.50e−3 | [0.01273485, 0.01273485] | [0.202725, 0.203232] | 10 | 69 |
| 1.09 | [26.865283375, 26.8819595] | 6.21e−4 | [0.01514250, 0.01514251] | [0.203404, 0.203530] | 12 | 60 |
| 1.10 | [23.08107775, 23.0954585] | 6.23e−4 | [0.01767344, 0.01767345] | [0.203961, 0.204088] | 13 | 65 |
| 1.11 | [20.118496, 20.143426] | 1.24e−3 | [0.02031933, 0.02031935] | [0.204397, 0.204651] | 12 | 61 |
| 1.12 | [17.7671015, 17.778150688] | 6.22e−4 | [0.02307296, 0.02307297] | [0.204970, 0.205097] | 14 | 61 |

- **Brackets**:
  - Every bracket stopped at an unresolved midpoint, as the error model predicts.
  - Three reached the model's minimum width, 6.2e−4.
  - Every evaluation converged.
- **Ĝ**: the relative width is ≤ 5.7e−7 at every a, and every exclusion closed.
- **Compute**: one process at nice 15, peak memory below 1 GB, about 5 min in total.

**Independent check** (`verify_certificates.check_finite`, Arb, one worker, largest a first; about 23 min per
certificate): **3 of the 10 bracket-end certificates were independently checked, and all 3 pass.**
- **Passed**: `c1f_a1.12_hi` ('plus'), `c1f_a1.12_lo` ('minus') and `c1f_a1.11_hi` ('plus').
  - Every sub-check is true: hashes, lemma W, tiling/coverage, the winning point in its region, the losing
    claims above U, and every leaf claim verified.
  - Check times were 1375 s, 1305 s and 1720 s.
- **Not checked** (stopped at 23:59 EDT on the coordinator's wrap-up instruction):
  - `c1f_a1.11_lo` was exported and hashed in the manifest; its check was interrupted.
  - `c1f_a1.10_hi/lo`, `c1f_a1.09_hi/lo` and `c1f_a1.08_hi/lo` were not exported.
  - These are search-certified only.
- **No a was excluded by the checker rule.**

**Ĝ(a)** from the rescaled search has no independent checker. It is search-certified only.

### 4. Fitted interval and verdict

- **Feasible set** (original four + five added, registered estimator): **C = [0.26531, 0.30687], width 0.0416.**
  - The width is valid (≤ 0.1), and the design goal of < 0.05 is met.
- **The derived c₁** ∈ [0.2852300, 0.2852303] lies inside C.
- **Registered follow-up verdict: PASS.**
- **Competing values**: 0.49 (the earlier incomplete prediction) and 0.662 (the switch shift alone) are
  excluded above; −0.377 (the K correction alone) and 0 (no first-order term) are excluded below.
- **The original test** (`first_order_prediction.md`) stays **INCONCLUSIVE** as registered (width 0.105).
  This is a follow-up registered after it.
- **For comparison only** (no verdict attaches): the supplementary branch-root set is [0.2846, 0.2859],
  inside C.

### 5. Say / Do not say

**PASS (this outcome)**
- Say: "In a follow-up registered after the original test was inconclusive, we added certified thresholds at
  a = 1.08–1.12 with the registered bracket procedure. The unchanged feasible-set estimator then gives
  c₁ ∈ [0.265, 0.307] (width 0.042 ≤ 0.1), which contains the derived c₁ = 0.28523 and excludes 0, 0.49,
  −0.377 and 0.662."
- Say: "The original registered test remains inconclusive (width 0.105 > 0.1). The follow-up needed ε up to
  0.12, because at tolerance 1e−9 the bracket procedure cannot resolve the switch below a relative width of
  6e−4."
- Do not say: "the original c₁ test passed"; "c₁ is measured to six digits" (the fitted interval is 0.042 wide;
  six digits is the derivation); "confirmed at small ε ≤ 0.04" (the decisive points are at ε = 0.08–0.12,
  within an O(ε³) allowance |c₃| ≤ 1).
- Do not say "all certificates independently checked" unless §3 lists all ten as checked. Ĝ from the rescaled
  search is not independently checked.

**FAIL** (not this outcome)
- It would have read: "a certified follow-up excludes the derived c₁, or no two-term law with |c₃| ≤ 1 fits". At
  ε ≤ 0.12 that would implicate either c₁ or the higher-order allowance.

**INVALID/UNRESOLVED** (width > 0.1; not this outcome)
- It would have read: "the follow-up was too coarse to test c₁". It would not be reported as a pass.

**Not run** (not this outcome)
- It would have read: "the follow-up was not run; the c₁ test remains inconclusive".


IDs: `c1 follow-up PASS`; `c1 follow-up width`; `c1 follow-up contains derived`.

## WP-31. Exact linear response along the trajectory: what accounts for the 0.82–0.91 shortfall (Track 1, final round; POST HOC; for the submission)

Producer: `src/linear_response.py` → `linear_response/`. Every prediction was committed and hashed before any observed
crossing was read. Everything here is POST HOC and changes no registered verdict. The text below is the Track 1 writer
input.

### 1. What was committed before the comparison

| item | value |
|---|---|
| predictions commit | `60cb084` (code, tests, `predictions.csv`, `predictions.sha256`, `predict.log`) |
| `predictions.csv` SHA-256 | `e86ca102bcf87f6ec44565d189d4c50fd0e4204db338309d6240527b1c39faa2` |
| contents | all 1,750 runs of the 1A set (36 arms), status ok for 1,750 of 1,750. Full model from 0.7·s\* and 0.5·s\*; ablations a, b, c, c_path, d, a_nomom, full_m0H (each from both start points); slaved own-coefficient formula |
| gate | `tests/test_linear_response.py` (8 tests: constant coefficients plus a linear θ\* path reproduce the analytic slaved lag κχ; a start at the slaved δ keeps δ constant; momentum with β₁ = 0 equals the no-momentum recursion; momentum leaves the steady lag unchanged; events; analytic derivatives against autograd; the exact gap against `lag_law.gap`; branch interpolation). These ran before any prediction. Suite at the predictions commit: 469 passed (tests minus certificates) and 27 passed (certificates) |
| compare | `python -m src.linear_response compare` first asserts that the hash matches and that the file is committed and unmodified. Observed crossings are read only inside `observed()` / `observed_1A()` |

**Disclosures about the pre-commit phase.**
- While developing, before the commit, the exact-gradient reconstruction (§4.4) was run on two runs: a = 1.30, seed
  300000, the φ = 1 trajectory and the 4b teleport arm. That reconstruction reproduces the run's own crossing, so these
  two crossing scales were effectively seen. The reconstruction was then removed from the committed variants. No setting
  was tuned on it.
- The stability rule (§2) and the reading of the ablations (momentum kept in a–c) were fixed after prediction-side
  development output on a handful of runs. No observed data was involved.

### 2. Method (the `src/linear_response.py` docstring is the authoritative statement)

- **Runs.** The 1,750 runs of `results/lag_law/predictions.csv`. Only their identifiers and the committed κχ were read.
  All runs were computed, with no subsample, in one worker process at nice 15 with one thread (41 min).
- **Replay.**
  - Each run is replayed deterministically from its initialisation under its own protocol:
    - Adam: torch defaults, lr 0.01.
    - SGD: plain torch SGD, lr 0.3, no momentum.
    - The φ rule on w₂.
    - The Block 4b interventions at t\*, exactly as run, including the AS-RUN sharing of optimiser state.
  - At every step the replay records θ, |w₂|, Adam's first moment, v̂ and the step count.
  - It stops at a fixed horizon: |w₂| ≥ 1.6·s\*_pop(a) or the run's budget.
  - The gap of the actual trajectory is never evaluated.
  - For the lag2 and 4b arms, the pre-t\* segment is replayed from initialisation and checked bitwise against the saved
    t\* checkpoint (θ and both Adam moments).
- **Branch and s\*.**
  - Objective: the run's own sample (6,400 points for lag1, lag2 and 4b; 400 for SGD and Task B) with mean BCE.
  - Orientation: canonical w₂ > 0. When w₂ < 0, (w₁, b₁) → (−w₁, −b₁), and Adam's first moment likewise.
  - θ = (w₁, b₁, b₂), with b₂ trained, as in 1A.
  - Tracked branch: Newton at fixed s from the run's state at the first step with s ≥ s\*_pop(a). Continuation then
    uses a tangent predictor on the grid s ∈ [0.3, 1.7]·s\*_pop with spacing 0.002·s\*_pop.
  - s\* is the root of G(θ\*(s)) nearest the anchor, found by bisection on the Newton-refined branch. G uses exact extrema
    on the continuous windows.
  - Refinement: Newton from the state at the first step with s ≥ s\*. This never changed the branch (0 of 1,750).
  - Interpolation:
    - θ\*(s) uses cubic Hermite interpolation with the exact tangents. The maximum error at the switch is 8.3e-13.
    - H(s) is interpolated linearly. The maximum error at the switch is 6.8e-7.
- **Full model.** It starts at t₀, the step at which the run last passes c·s\* before the switch (c = 0.7 and 0.5).
  - Initial state: δ₀ = θ_{t₀} − θ\*(s_{t₀}), and m₀ = the run's actual Adam first moment at t₀. The variant full_m0H
    uses H δ₀ instead.
  - Recursion:
    - m_{t+1} = β₁m_t + (1 − β₁)H(s_t)δ_t
    - δ_{t+1} = δ_t − η P_{t+1} m_{t+1}/(1 − β₁^{k_{t+1}}) − [θ\*(s_{t+1}) − θ\*(s_t)]
    - P is the run's own 1/(√v̂ + ε), using torch's ordering and bias corrections.
    - SGD: δ_{t+1} = (I − ηH)δ_t − Δθ\*.
  - Prediction: the first step with G(θ\*(s_t) + δ_t) > 0. The prediction is the actual s_t at that step.
  - 4b interventions inside the window: the actual θ jump is added to δ, and m is replaced by the loaded first moment,
    which is zero for reset.
- **Ablations.** The author's (a)–(c) change the coefficients of the full model and keep its momentum state; (d) removes
  that state.
  - (a) All coefficients are frozen at the switch: H, P, θ\*′ and ∇G at s\*, with the linear branch and linear G exactly
    as in 1A. This is the 1A model iterated along the actual s path, without the steady-state assumption.
  - (b) Only P varies in time.
  - (c) Only H and θ\*′ vary: the exact θ\*(s) path with exact G; P frozen.
  - c_path: the exact path, with H and P frozen.
  - (d) The full model without the momentum state.
  - a_nomom: the literal 1A dynamics in discrete time, i.e. (a) without momentum.
  - slaved: the closed-form 1A formula with each run's own coefficients at its own switch, and ṡ averaged over the 100
    steps before the switch.
- **Stability rule** (committed). A variant gets no prediction if its iterated one-step linear map has spectral radius
  > 1 (checked every 10th step), or if sup|δ| > 1.
- **Different branch at the start.** Newton from the actual state at s_{t₀} converges to a point at sup distance
  > 1e-6 from θ\*(s_{t₀}), or does not converge. The model is run for these runs anyway: none is excluded.
- **Lag.** r_obs = s_obs/s\* − 1 and r_pred = s_pred/s\* − 1, both against the run's tracked-branch switch s\*. The
  observed crossing uses the runs' own dense-grid gap (4,001 points); the prediction uses the exact gap. §4.4 shows the
  difference never changes the crossing step.

### 3. Results

#### 3.1 Branch mismatch (the author's addition)

- **0 of 1,750** runs are on a different branch at 0.7·s\*, and **0 of 1,750** at 0.5·s\*. No runs were excluded.
  - The maximum distance from the start state's Newton point to the tracked branch is 6.5e-12 at 0.7·s\* and 8.7e-12 at
    0.5·s\*.
  - No run changes the sign of w₂ inside the window.
  - Per-arm counts are 0 / 0 in every arm (`tables.md`, last column).

#### 3.2 The full model, per a (pooled over optimisers, the statistic behind 1A's 0.82–0.91)

Through-origin slope of r_obs on r_pred, with the median per-run ratio r_obs/r_pred in brackets (`attribution.csv`):

| a | n | 1A as published: residual vs global own threshold, on κχ | **full, from 0.7·s\*** | **full, from 0.5·s\*** |
|---|---|---|---|---|
| 1.30 | 795 | 0.89 (1.02) | **1.02 (1.02)** | **1.02 (1.02)** |
| 1.45 | 80 | 0.83 (1.01) | **1.03 (1.03)** | **1.03 (1.03)** |
| 1.50 | 795 | 0.91 (1.00) | **1.04 (1.04)** | **1.04 (1.04)** |
| 1.60 | 80 | 0.82 (1.01) | **1.05 (1.05)** | **1.05 (1.05)** |

- The 1A column reproduces the published 0.89 / 0.83 / 0.91 / 0.82.
- Per arm, the full model's observed/predicted ratio of medians is **1.00–1.05 in all 36 arms**, from either start
  point.
- By κχ tercile within each a, the full model's median per-run ratio is 1.015–1.049. It is flat in χ, whereas κχ falls
  to 0.79–0.93 in the top tercile (§3.4).
- **The two starting points give the identical predicted crossing in every run** (1,706 of 1,706 runs with both
  defined). The transient from δ₀ has died out long before the crossing: the median Adam relaxation time at the switch
  is 8.7 steps, and the start points lie 428 and 672 steps before the switch (medians). **The conclusion does not
  change between 0.7·s\* and 0.5·s\*.**
- 42 runs have no full-model prediction from 0.7·s\*, and 23 from 0.5·s\*. The stability rule removed them.
  - 24 of the 42 are 4b reset runs. Right after the moment reset, Adam's bias-corrected v̂ is tiny, so P is huge and the
    iterated map transiently has radius > 1.
  - The rest are a few recurring seeds spread across arms.

#### 3.3 Which simplification accounts for the 0.82–0.91

The table removes one approximation at a time. Entries are the per-a through-origin slope of r_obs on r_pred. The 96
4b-reset runs are left out of the ablation rows only; the slope of every row including them is in `attribution.csv`.

| step | 1.30 | 1.45 | 1.50 | 1.60 |
|---|---|---|---|---|
| 1A: residual vs **global own-sample threshold**, on κχ | 0.89 | 0.83 | 0.91 | 0.82 |
| κχ, residual vs the run's **tracked-branch switch** | 1.02 | 1.01 | 0.98 | 0.90 |
| slaved formula with **own coefficients** at own switch | 1.01 | 0.99 | 0.96 | 0.90 |
| (a) frozen coefficients, **no steady-state assumption** (recursion along the actual s path) | 1.01 | 1.02 | 1.03 | 1.04 |
| (b) + time-varying P | 1.02 | 1.03 | 1.04 | 1.04 |
| (c) + time-varying H and θ\* path (P frozen) | 1.01 | 1.02 | 1.03 | 1.04 |
| (d) all time-varying, no momentum (only where stable) | 1.02 | 1.03 | 1.03 | 1.03 |
| full (all time-varying, momentum) | 1.02 | 1.03 | 1.04 | 1.05 |

Median per-run ratios of observed to predicted lag, by κχ tercile at the two largest a (`attribution_tercile.csv`):

| a, tercile (median κχ) | vs global threshold, κχ | vs branch switch, κχ | slaved own | (a) | full |
|---|---|---|---|---|---|
| 1.50, low (0.046) | 1.03 | 1.03 | 1.01 | 1.00 | 1.03 |
| 1.50, high (0.072) | 0.91 | 0.93 | 0.92 | 1.06 | 1.04 |
| 1.60, low (0.070) | 1.07 | 1.06 | 1.01 | 1.02 | 1.05 |
| 1.60, high (0.099) | 0.79 | 0.80 | 0.83 | 1.07 | 1.05 |

**Attribution.**
- **The reference against which 1A measured the lag.** 1A used each run's global own-sample threshold. Here the lag is
  measured against the switch of the branch the run actually tracks. In 22.8% of runs the tracked switch lies more than
  1% from the global threshold (17.5–24.2% per a and optimiser).
  - At a = 1.30 and 1.45 this change alone takes the slope from 0.89 to 1.02 and from 0.83 to 1.01.
  - This is the same effect §13.6 of the math note found for the ramps.
- **The steady-state (slaved) displacement.** This part matters at a = 1.50 and 1.60, where χ is largest.
  - With own coefficients the slaved formula still gives 0.96 and 0.90, and 0.83–0.92 in the top κχ tercile.
  - Iterating the same frozen-coefficient linear model along the actual s path, which is ablation (a), gives 1.03 and
    1.04, and 1.06–1.07 in the top tercile.
  - The start point does not matter (§3.2), so this is not the initial transient. It is the variation of ṡ within the
    relaxation memory, which the slaved formula replaces by ṡ at the switch. That is an interpretation; the tercile
    pattern is the measured fact.
- **Coefficients frozen at the switch.** Not responsible. Adding time-varying P, or time-varying H and θ\*(s), moves the
  slope by at most 0.01 from (a).
  - c_path, which uses the exact θ\*(s) path with H and P frozen, alone gives 0.98 at every a. The exact path and the
    time variation of H partly cancel.
- **Momentum.** Not responsible for the lag. Where the no-momentum model is defined, full and (d) give the same
  prediction: median relative difference 0, maximum 9%, over 510 runs.
  - The momentum state is, however, needed for the linearised Adam dynamics to exist at all. At the switch the
    no-momentum map I − ηPH has spectral radius > 1 in 71% of Adam runs (median 2.9, maximum 37.8).
  - Along the path it is unstable in 1,240 of 1,690 Adam runs. So (d) and a_nomom have no prediction for most Adam runs
    (18 of 36 arms have none).
  - With the momentum state the map's radius is √β₁ = 0.949 at the switch in every Adam run but one.
- **Ablation (a) reproduces κχ, as the machinery check requires.**
  - Median (a) prediction / median κχ is 0.88–1.05 per arm, excluding the two 4b-reset arms. The lower end is the
    high-χ arms, where the non-steady recursion predicts less than the slaved formula.
  - Own κ over the committed κ has median 0.995. Own χ at the switch over 1A's χ at the crossing has median 1.016.
  - In the reset arms, (a) and (b) from 0.7·s\* give a spurious early crossing: arm ratios −0.11 and −0.23. The moment
    reset lies inside the window, and Adam's post-reset step (about η per coordinate on w₂) drives the frozen linear
    branch.
- **What remains, +2% to +5%.** The full model under-predicts the observed lag by 2% (a = 1.30) to 5% (a = 1.60). The
  excess grows with χ. See §4.4.

#### 3.4 Per-arm table

`results/linear_response/tables.md` gives:
- all 36 arms with observed/predicted for κχ, slaved, full from both start points, and ablations (a)–(d) from both;
- the per-a table split by optimiser;
- the prediction-only table of each variant over κχ.

Arm-level ranges of the observed/predicted ratio of medians:

| model | range |
|---|---|
| κχ vs tracked-branch switch | 0.91–1.09 |
| slaved own | 0.94–1.03 |
| full, 0.7·s\* | 1.00–1.05 |
| full, 0.5·s\* | 1.00–1.05 |
| (a), excluding reset | 0.98–1.05 |
| (b), excluding reset | 1.00–1.05 |
| (c) | 0.98–1.05 |
| (d), 18 arms with predictions | 1.00–1.05 |

#### 3.5 SGD separately (P = I, no momentum; the full model is (c) and (d))

Full model / observed, with slope and median ratio:

| a | slope | median ratio |
|---|---|---|
| 1.30 | 1.02 | 1.02 |
| 1.50 | 1.03 | 1.04 |

κχ against the tracked-branch switch gives 1.05 and 1.08 (medians 1.07, 1.08). The slaved own formula gives 1.00 and
1.00.

### 4. Checks

1. **Interpolation.**
   - θ\*(s) at the switch: maximum error 8.3e-13, Hermite against Newton.
   - H at the switch: maximum error 6.8e-7.
   - Tests check θ to 1e-7 and H to 1e-3 at interior points.
2. **Replays.**
   - Every lag2 and 4b replay reproduced its t\* checkpoint bitwise.
   - Every run completed: status ok for 1,750 of 1,750.
3. **Start points.**
   - s₀/s\* is at most 0.716 for the 0.7 start and at most 0.511 for the 0.5 start.
   - Interventions fall inside the window in 90 runs (0.7 start) and 288 runs (0.5 start). They are handled as events.
4. **Exact-gradient reconstruction** (POST HOC, **run after the comparison**; `diagnose`, the first 12 seeds of every
   arm; `diagnostic_exact_grad*.csv`).
   - Replacing Hδ by the exact gradient ∇L(θ\*(s_t) + δ_t; s_t) in the full model, with everything else unchanged,
     reproduces the **observed crossing step exactly in 426 of 426 runs (0.7 start) and 430 of 430 (0.5 start)**. The
     other 6 and 2 of the 432 fall under the committed stability rule.
   - So the machinery is exact, including P_t, the bias corrections, the event handling and the continuous-vs-grid gap.
     The **only** difference between the full model and the actual run is the linearisation of the gradient in δ.
   - On these runs the observed-to-full slope is 1.034 overall. Per a it is 1.02 (1.30 Adam), 1.01 (1.30 SGD),
     1.03 (1.45), 1.04 (1.50 Adam), 1.03 (1.50 SGD) and 1.05 (1.60).
   - The +2–5% excess is therefore the second-order (nonlinear-in-δ) response.

### 5. Writer section

**Outcome: the full model matches closely.**
- The observed/predicted lag ratio is **1.02–1.05 per a** by both the through-origin slope and the median per-run
  ratio, and **1.00–1.05 in each of the 36 arms**.
- This holds from either start point, with predictions identical in every run, and for Adam and SGD alike.

**Say:**
- "Post hoc, we integrated the linearised tracking dynamics along each run's own trajectory. The inputs were the
  actual output-scale path, the run's own Adam preconditioner and momentum, and the exact branch path and gap. The
  crossing predicted this way was committed before comparison. It gives observed/predicted lag ratios of 1.02–1.05 at
  every a (1.00–1.05 in each of the 36 arms). The lag is therefore quantitatively first-order tracking."
- "The 0.82–0.91 shortfall of the closed-form law κχ has two sources, neither of them frozen coefficients or the
  omitted momentum.
  (i) The reference. Measured against the switch of the branch each run actually tracks, rather than its sample's
  global threshold, the ratio becomes 1.02 and 1.01 at a = 1.30 and 1.45. About one run in five tracks a branch whose
  own switch lies more than 1% from the global threshold.
  (ii) The steady-state (slaved) approximation, which over-predicts the largest lags. At a = 1.50 and 1.60 the slaved
  formula gives 0.96 and 0.90, and 0.83–0.92 in the top third of κχ. Iterating the same frozen-coefficient model along
  the actual path gives 1.03 and 1.04."
- "Time variation of the Hessian, branch tangent and preconditioner changes the predicted lag by at most about 1%.
  Momentum leaves it unchanged, but it is required for the linearised Adam dynamics to be stable at all."
- "The remaining 2–5% under-prediction is second order in the displacement. Replacing the linearised gradient by the
  exact one reproduces every observed crossing step exactly (856 run-starts)."
- "No run was on a different branch at the starting point; none was excluded."
- Always with the label: post hoc; derived and compared after the 1A shortfall was known.

**Do not say:**
- "The lag law r = κχ is exact" or "κχ matches to 2%". The closed form, which is steady-state, over-predicts the
  largest lags by up to 20% (top κχ tercile at a = 1.60: 0.80).
- "The shortfall was caused by frozen coefficients" or "by momentum". Ablations (b)/(c) move the slope by ≤ 0.01, and
  (d) equals full where it exists.
- "The linear model is exact". It under-predicts by 2–5%, growing with χ.
- That the 0.82–0.91 was a uniform per-run shortfall. The median per-run ratio against the 1A reference was already
  1.00–1.02. The through-origin slope was pulled down by the largest-κχ runs and by runs whose tracked switch is off the
  global threshold.
- That this analysis is prospective or registered. It is post hoc on existing runs, with predictions committed before
  comparison.
- Any statement about widths other than 1, or about activations other than f_a.

**If the full model had not matched** (for completeness, not the outcome): "the linear-tracking picture is incomplete
by X%, and the paper says so." Not applicable: X is 2–5%, and §4.4 locates it in the second-order response.

### 6. Files

- `src/linear_response.py`: replay, branch, recursion, predict/finalize/compare/attribution/extras/tables/diagnose.
- `tests/test_linear_response.py`
- `results/linear_response/`:
  - `predictions.csv` and `predictions.sha256` (committed at 60cb084)
  - `compare_runs.csv`, `compare_arms.csv`, `compare_per_a.csv`
  - `attribution.csv`, `attribution_tercile.csv`
  - `diagnostic_exact_grad.csv`, `diagnostic_exact_grad_per_a.csv`, `diagnostic_exact_grad.jsonl`
  - `tables.md`, `summary.json`, `predict.log`, `diagnose.log`
- Not committed: `predictions_parts.jsonl`, the 10 MB resumable raw form of `predictions.csv`; same content.


IDs: `LR predictions hash`; `LR full model ratio`.

## WP-29. Open items for the rebuttal (final round; not run, not in the paper's results)

These were out of scope for the final round. The paper may list them as open; it must not state results for them.
- **The outer-exclusion certificate check** by the independent checker (over 10 CPU-hours). Only the original search
  supports the outer-exclusion certificates (WP-17).
- **Further activation families** beyond GELU, SiLU and Mish (WP-25).
- **Any other width-2 variant** (WP-15, WP-20, WP-27).
- **Designs that failed their own rules before registration** (tonight; recorded, not run):
  - the width-2 early-basin prospective test (2C, WP-27);
  - the GELU early-scale prospective test (WP-25);
  - the Adam ramp from initialisation (WP-24).

**Do not say** that any of these was tested.
