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
- **Headline**: 183 scored by their registered rules: 85 PASS, 58 FAIL, 8 PARTIAL,
  32 UNRESOLVED.
- **Post hoc**: 16 assigned post hoc: 3 / 6 / 7 / 0.
- **Total**: 199 registered predictions.

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
  second-order (Var) selection, which the registration added; its direct check is pending (below).
- *Pending (registered, not yet scored)*: the width-2 direct check (2 of 8 scales scored so far, both placed), and
  the width-2 no-gating test (not run).
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
| winding 08-22 | registered rule | 4 | 0 | 0 | 0 | 4 |

### By registration file
| group | scoring | PASS | FAIL | PARTIAL | UNRESOLVED | n |
|---|---|---|---|---|---|---|
| results/amplification_prediction.md | post hoc (census) | 0 | 1 | 0 | 0 | 1 |
| results/amplification_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/arrhenius_prediction.md | registered rule | 15 | 11 | 1 | 15 | 42 |
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
| results/third_optimizer_prediction.md | registered rule | 2 | 0 | 0 | 0 | 2 |
| results/threshold_prediction.md | post hoc (census) | 1 | 0 | 0 | 0 | 1 |
| results/threshold_prediction.md | registered rule | 2 | 1 | 1 | 0 | 4 |
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
All 393 printed-number checks of the ledger (`src/verify_ledger.py`, which verifies every
printed number against its artifact) still hold, and round to the same printed digits, with their artifact value
scaled by 1 ± δ (conservatively applied to every check, Ĝ-dependent or not); 0 are unstable
(`ghat_digit_stability.csv`). A further 12 checks compare two artifacts to 1e−12; they are
not printed numbers. 5 of them move in their last digits under the blanket δ, and none of
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
| a = 1.30, R₂ = 0.003 | PENDING | 0.6905704168112 (boundary, G₊ = -1.2e-11) | PENDING | 1.45e-05 (single unit, G₊ = -3.64) |
| a = 1.30, R₂ = 0.01 | PENDING | 0.6845925670975 (boundary, G₊ = -2.0e-12) | PENDING | 1.60e-04 (single unit, G₊ = -3.59) |
| a = 1.30, R₂ = 0.02 | PENDING | 0.6761364675950 (boundary, G₊ = -1.6e-13) | PENDING | 6.33e-04 (single unit, G₊ = -3.52) |
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
minimiser itself. Landed so far: a = 1.30 at R₂ = 0.001, a = 1.50 at R₂ = 0.001, a = 1.50 at R₂ = 0.003 (placed cancelling pair at each, every validation passed, the best unplaced
configuration on the G = 0 boundary, at least 99× the tie tolerance above); **5 scale(s) PENDING** (table in WP-8, not
repeated here). IDs: `direct check: every landed scale placed, cancelling pair`; `direct check: no finished restart below the retained minimiser (landed scales)`; `min margin at R2 = 0.001`; `all best-unplaced on the boundary`.

**tanh: registered outcome "neither".** Every registered expectation held except one: validation passed at every box
size, the maxima are < 1 and rising, the first-order tie is present, the Var-selected symmetric pair is placed at every
box (G₊ → 1) and the single units are unplaced. The boundary criterion as written fails at A = 40, because the 1e−9 tie
tolerance is of the order of 1 − max (4.1e−9) there. Non-attainment of the supremum is proved analytically
(Δμ < 1 = sup Δμ). No re-registration. tanh lies outside the criterion's attainment hypothesis.
IDs: `tanh: registered outcome 'neither'`; `tanh: validation passed at every A`; `tanh: box maxima < 1 and rising`; `tanh: boundary condition fails only at A = 40`; `tanh: selected member is the symmetric pair at every A`; `tanh: single units unplaced at every A`.

**Say**
- "At width 2, output scale does not gate placement for f_a: the registered small-scale prediction selects the placed
  cancelling pair, and a direct search at small scales finds that pair as the conditional minimiser."
- "The prediction rests on a second-order selection: at first order, unplaced single units and placed pairs tie."
- "For tanh the supremum of Δμ is not attained; the registered test returned 'neither' because its boundary criterion
  failed at the largest box, although every other expectation held."

**Do not say**
- "proved" or "certified" for the width-2 verdict (it is a registered, validated computation plus a lemma, not a
  certificate); or that Δμ alone decides it.
- that the direct check is complete while any scale is PENDING.
- that the unplaced region has no local minima, or that the boundary point is a competing minimum (WP-8 wording).
- that tanh "confirmed" or "passed" the registered expectation.

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

**Pending (not yet exported or checked).** In priority order (the main theorem rests on the first two):
1. the limit-problem chain: the limit-switch status certificates at both ends of the A* bracket (exported; the Arb
   check is running); K = sup G₀; the Krawczyk boxes for c₁;
2. the solve brackets;
3. outer exclusion and ring, the PD boxes, localisation (B(24) and B_full).

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
- "An independent checker in ball arithmetic, sharing no code with the searches, verifies the finite-a certificates
  and the Ĝ enclosures; the remaining certificate families are being exported."

**Say** (the sharp limit threshold)
- "K = sup G₀ is certified by branch and bound over a box, and a domain lemma shows the supremum lies inside it."

**Do not say**
- that K (or the sharp limit threshold built on it) has been independently checked in Arb, until that check is done;
- that every certificate has been independently checked, until the pending families are;
- that the checker verified the original float endpoints of Ĝ (it verified rigorous enclosures that differ from them
  by at most 1.1e−15).
