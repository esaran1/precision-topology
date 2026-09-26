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
- **Headline**: 195 scored by their registered rules: 92 PASS, 61 FAIL, 8 PARTIAL,
  34 UNRESOLVED.
- **Post hoc**: 16 assigned post hoc: 3 / 6 / 7 / 0.
- **Total**: 211 registered predictions.

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
| c1 first order | registered rule | 1 | 0 | 0 | 2 | 3 |
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
| nu | registered rule | 0 | 1 | 0 | 0 | 1 |
| own-seed thresholds | post hoc (census) | 0 | 0 | 1 | 0 | 1 |
| own-seed thresholds | registered rule | 1 | 1 | 0 | 0 | 2 |
| phase1 | registered rule | 1 | 1 | 0 | 0 | 2 |
| phase2b | registered rule | 0 | 1 | 0 | 0 | 1 |
| phase2b across a | registered rule | 2 | 1 | 0 | 1 | 4 |
| precision | registered rule | 0 | 0 | 0 | 1 | 1 |
| prospective own-seed | registered rule | 8 | 0 | 0 | 0 | 8 |
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
| results/amplification_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/amplification_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/arrhenius_prediction.md | registered rule | 15 | 11 | 1 | 15 | 42 |
| results/asym_registration.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/asym_registration.md (amendment 1) | registered rule | 0 | 1 | 0 | 0 | 1 |
| results/asym_registration.md (amendment 2) | registered rule | 0 | 0 | 0 | 1 | 1 |
| results/asym_registration.md (amendment 3) | registered rule | 0 | 0 | 0 | 1 | 1 |
| results/asym_registration.md (amendment 4) | registered rule | 0 | 1 | 0 | 0 | 1 |
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
All 547 printed-number checks of the ledger (`src/verify_ledger.py`, which verifies every
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
| post hoc (census) | central | 1 | 2 | 7 | 0 |
| post hoc (census) | peripheral | 2 | 4 | 0 | 0 |
| registered rule | central | 45 | 26 | 1 | 9 |
| registered rule | peripheral | 47 | 35 | 7 | 25 |

IDs: `A4 FAIL registered central`; `A4 FAIL registered peripheral`; `A4 FAIL post hoc central`; `A4 FAIL post hoc peripheral`; `A4 PASS registered central`; `A4 PASS registered peripheral`; `A4 rows`.

**Failed central predictions (28: 26 under the registered rule, 2 by post hoc scoring), with one line each:**

*threshold* (18):
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

None is a failure of the certified threshold values themselves, and the primary prospective comparisons (Block 3,
own-seed primary) passed.

**Say:** "Of 67 failed predictions (61 under the registered rule, 6 by post hoc scoring), 28 bear on a central
claim; most of these concern the residual's mechanism, which the paper reports as open, and one is the width-2 training
prediction on asymmetric windows (WP-15)." **Do not say** "the central claims never failed" or "the failures are
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

- **SGD:** the law predicts the cell medians to within a few percent in every cell except the fastest (κχ = 0.1), where
  observed/predicted is 1.14–1.26 (the nonlinear regime, math note §13.3(iv)).
  - The sign test holds: on the k = 0 copy at a = 1.30 the crossings come *early*, as predicted.
- **Adam:** predicted lags are ≤ 0.006, and there the medians carry a small offset of order 0.001. R1 fails at a = 1.30
  and passes at a = 1.50.

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
  crossing lag matches κ(a)χ with no fitted parameter across two decades of rate, including the predicted *early*
  crossing on the winding copy with negative κ."
- **Do not say:**
  - that the ramp test passed as registered;
  - that κ was predicted before the fitted relationship was known;
  - that the law holds for Adam at small predicted lags (R1 fails at a = 1.30 post hoc too);
  - that the law holds in the nonlinear regime κχ ≳ 0.1;
  - that the lag law explains width 2 (WP-15; Track 2 below).
IDs: `1A arms within tolerance`; `1A slope/kappa a=1.30`; `1A slope/kappa a=1.50`; `1B registered R1 SGD 1.30 k=-1`; `1B post hoc R1 SGD 1.50`; `1B branch off own a=1.50`.
