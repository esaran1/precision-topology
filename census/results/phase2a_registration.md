# Phase 2A registration: slow tracking on the simplicity-bias benchmark

**REGISTERED.** This file, the approved design page, `src/phase2a.py`, `src/causal_forecast_fold.py`,
`tests/test_phase2a.py`, the frozen forecaster and the code it reaches, and the frozen inputs (`frozen.json`,
`pilot.json`, `seed_scan.json` and Track 1A's branch files) are committed in one commit before any registered run.
Their SHA-256 hashes are in `results/phase2a/registration.sha256` (§11).
- **Not yet done:** no registered seed has been drawn, held or trained.
- **The open details:** the author decided them on 2026-10-04 (§9). S1: the middle pilot rate moved off the ladder,
  to 2⁻¹⁶·⁵⁶²⁵. D1–D15 were approved as proposed, with ONE addition: **every verdict is computed over the 22 ladder
  rates with no prior outcome.** The 5 rates already run in the exploration are still run and are reported as
  descriptive (§2).
- **Pilot:** pilot seeds only, at the three pilot rates. τ₁ = 0.15 and τ₂ = 0.25; the STOP rule did not fire (§12).
- **Before any registered training:** the registration commit is pushed and its OpenTimestamps proof is obtained.

## 1. Design reference

- **Approved page:** `results/designs/phase2a_slow_sb_design.md` (approved by the author 2026-10-04, commit ca38ff5;
  revised draft 053dca5). Everything on it is binding here. The approval:
  1. f = 0.95 (not 0.90). The tightening cutoff (1 − f ∝ ρ^{2/3}, equal to 0.05 at 2⁻¹⁵) and each rate's forecast
     horizon are reported descriptively only.
  2. The worker's item-2 choices: 27 distinct rates (anchors 2⁻¹³, 2⁻¹⁴, 2⁻¹⁵; new 2^−(15+k/8), k = 1…24), assigned
     in seed order to the seeds landing on M, fastest first; three pilot rates 2⁻¹⁵·⁰⁶²⁵, 2⁻¹⁶·⁵, 2⁻¹⁷·⁹³⁷⁵ (the middle one later moved to 2⁻¹⁶·⁵⁶²⁵, item 5); gate ≥ 27
     of 120 seeds on M; no forecast fails C1, C2 and C4; τ₁, τ₂ = 1.5 × the largest pilot error at f = 0.95, rounded
     up to 0.05; training stops at t_c, the state is hashed, observation resumes from it to 1.25·s_F.
  3. The fixed dataset. 2A tests ONE landscape across rates (§2), with the resampling exploration as a disclosed limit
     (§8).
  4. Everything else as revised.
  5. **Second decision (author, 2026-10-04, after the stop of c3bf012/0c4a72f).**
     - S1 (b): the middle pilot rate is 2⁻¹⁶·⁵⁶²⁵ (k 12.5), off the ladder like the other two.
     - D1–D15 are approved as proposed.
     - ADDITION: the 5 ladder rates with an exploratory outcome (2⁻¹³, 2⁻¹⁴, 2⁻¹⁵, 2⁻¹⁶, 2⁻¹⁷) are excluded from every
       scored set. Every verdict is computed over the 22 others. All 27 rates are still run, and the 5 are reported
       as descriptive.
- **Inputs:** the v2 frozen inputs (`results/simplicity_bias_v2/frozen.json`: q, s\*, λ); Track 1A's branches
  (`src/sb_fold.py`, `results/sb_fold/`: M, S, L0, S2 continuation points and M's Δs = 0.001 grid; s_F = 4.767689);
  the frozen Phase 1A forecaster `src/causal_forecast.py` (25aac9e, unchanged: `extrapolate_scale`, `GuardedArray`,
  `cutoff_step`); the exploration `results/designs/phase2a_explore/` (3fe8633 and earlier).

## 2. Setting, rates and seeds

- **One landscape.** The fixed v3 data (800 points), tanh, width 4, λ = 1e−4, BCE, float64, full batch, no momentum,
  η = 1. s = ‖v‖₁; q = 0.3914103370; s\* = 3.5913755425 (v2 `frozen.json`). Every M release is the same point, so every
  run is deterministic given ρ: **the rates, not the seeds, are the replicates.** A seed only decides whether it lands
  on M and which rate it takes.
- **The scoring set: 22 rates** (`phase2a.SCORED_LOG2`, `score_tables`).
  - It is the ladder minus the 5 rates with a prior outcome (`PREOBSERVED_LOG2` = 2⁻¹³, 2⁻¹⁴, 2⁻¹⁵, 2⁻¹⁶, 2⁻¹⁷;
    `p2a_explore_f` and `_g`): 2⁻¹⁵·¹²⁵ … 2⁻¹⁸ without 2⁻¹⁶ and 2⁻¹⁷.
  - Every fraction, bootstrap and validity rule runs over these 22 distinct rates (§5, §6).
  - The 5 pre-observed rates are RUN like the others (all 27 run, one per M seed; the gate stays ≥ 27 of 120).
    The same statistics are reported for them as DESCRIPTIVE, with denominator 5, and never enter a verdict.
- **Output rule (scale only).** (W, c, b) ← · − η∇; v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√3 on the three
  active units (the idle unit's component is 0). **Not the practitioner's knob** (the plain output learning rate leaves
  χ ≥ 3.5 at every ρ; Phase 2B tests that knob).
- **Hold.** v3's init (`simplicity_bias_v3.init_net`'s draw, from a LOCAL `torch.Generator(seed)`, bit-identical to
  `torch.manual_seed(seed)` + the two Linear layers; tested) with v rescaled to ‖v‖₁ = s₀ = 1.7957; fixed-s BFGS
  (`sb_fold.local_min_batch`, gtol 1e−8; the 120 seeds in one batch, in seed order). Label: the branch (L0, M, S, S2)
  whose frozen point at s₀ lies within 1e−3 (function space, `sb_fold.fdist`), the nearest if several, else "other".
- **Release.** A seed labelled M is released at the exact branch point θ_M(s₀) (Newton, `sb_fold.branch_point`), idle
  unit exactly 0: **a three-unit test.** Off-M seeds are counted, not trained.
- **Plan.** The i-th seed (in seed order) labelled M takes the i-th ladder rate, fastest first. M seeds beyond the 27th
  are counted, not trained.
- **Ladder (27 rates):** 2⁻¹³, 2⁻¹⁴, 2⁻¹⁵, then 2^−(15+k/8) for k = 1…24 (2⁻¹⁵·¹²⁵ … 2⁻¹⁸).
- **Seeds:** registered 2,937,000–2,937,119 (120); pilot 2,937,900–2,937,959 (60). Re-verified unused
  (`results/phase2a/seed_scan.json`, `phase2a.scan`): no number in either range, also written with `_` or `,`, in any
  text file under src/, tests/, results/ or paper/ (2A's own files and the design page excepted); no overlap with the
  registered or pilot seeds of any test module; disjoint from the exploration seeds 2,930,000–2,930,199.

## 3. Order of the steps (`src/phase2a.py`)

1. `scan`, `freeze`, `pilot`, `manifest`: before the registration commit. No registered seed is drawn.
2. `run`, after the registration commit and its timestamp:
   - the holds of the 120 seeds, the gate, the plan (`holds.jsonl`);
   - per planned rate: training from θ_M(s₀) to t_c, with NO ρ₂ and no branch check; the state at t_c (17 float64)
     stored and hashed (SHA-256 of its little-endian bytes); the causal forecast at f = 0.95; the STOP-rule quantities
     (`runs.jsonl`).
3. `finalize`: `forecasts.sha256` over `holds.jsonl` and `runs.jsonl`, with every planned rate exactly once and no
   observed key in any row. These three files are **committed before observation.**
4. `observe` asserts the registration and the committed forecasts, then per rate:
   - retrains from the release with ρ₂ at **every** step;
   - at t_c asserts the state equals the hashed one bit for bit, and continues from the hashed state;
   - stops once the crossing and s_F have both been reached (and t ≥ t_c), or at s ≥ 1.25·s_F, or at the budget;
   - records t_obs, s_obs, t_F, the follow check at 0.8·s_F, the idle unit and the signs, and the descriptive
     tightening-cutoff forecast (`observed.jsonl`).
5. `score` writes `scores.json`.

## 4. The forecast (causal) and its enforcement

- **Cutoff (a stopping time; harness side).** t_c = the first step t ≥ 1 with s_t ≥ 0.95·s_F
  (`train_to_cutoff`, equal to `causal_forecast.cutoff_step`). Training stops at t_c.
- **Forecaster** (`src/causal_forecast_fold.py`). It reads s only, rows 0 … t_c − 1, through a `GuardedArray` with
  cutoff t_c, and uses the frozen `extrapolate_scale` (quad, window ⌈0.05·t_c⌉ ≥ 50, horizon 1.6·s_F) unchanged:
  - ŝ_k forecasts s at step t_c + k (anchored at s_{t_c−1});
  - t̂_F = t_c + (the first k with ŝ_k ≥ s_F): the no-delay forecast;
  - ṡ̂_F = ŝ_{k_F} − ŝ_{k_F−1} (ŝ_{−1} := s_{t_c−1});
  - ε̂_F = (ṡ̂_F/s_F)/(ηΛ_F), r_fc = Ω₀ε̂_F^{2/3}, ŝ_c = s_F(1 + r_fc), Ω₀ = 2.338107;
  - t_fc = t_c + (the first k with ŝ_k ≥ ŝ_c); delay_fc = t_fc − t̂_F.
  - **No forecast** when the extrapolation returns no path, or no ŝ reaches s_F, or none reaches ŝ_c, within 1.6·s_F.
    No forecast fails C1, C2 and C4 (§5).
  - This is the rule of `p2a_explore_f`/`g`, with the exploration's silent `argmax` fallback (t_fc = t_c when ŝ_c is
    never reached) replaced by "no forecast".
- **Enforcement** (`tests/test_phase2a.py`):
  - a forecaster that reads the cutoff row, or slices past it, raises `CausalityViolation`;
  - a forecaster that bypasses the guard is caught by the NaN recomputation;
  - on a real path the last row read is t_c − 1, and the output is identical with every row ≥ t_c NaN, 1e9 or −3.
- **Per run.** `run_forecast` asserts the guard's last row < t_c and recomputes with every row ≥ t_c NaN. A
  differing recomputation aborts the run (`CausalityViolation`). It is not a per-rate miss.

## 5. Criteria (over the 22 scored rates; `phase2a.score_tables` → `criteria`)

r_obs = s_obs/s_F − 1; delay_obs = t_obs − t_F, where t_F is the first step with s ≥ s_F on the observed path.
Numbers are compared in float64, without rounding.

| | statistic | PASS if |
|---|---|---|
| **F** | s_F ≤ s_obs ≤ 1.25·s_F (inclusive) | at ≥ 90% of the 22 scored rates (≥ 20), AND at no scored rate s_obs ≤ 1.25·s\* (the falsifier, inclusive) |
| **C1** | \|t_fc − t_obs\| ≤ τ₁·delay_fc (inclusive), τ₁ = 0.15 | at ≥ 80% of the 22 scored rates (≥ 18) |
| **C2** | \|delay_fc − delay_obs\| ≤ τ₂·delay_fc (inclusive), τ₂ = 0.25 | at ≥ 80% of the 22 scored rates (≥ 18) |
| **C3** | Spearman(ρ, r_obs/r_fc) (average ranks) and min r_obs/r_fc | Spearman ≥ 0.8 AND r_obs/r_fc > 1 at every scored rate with a ratio |
| **C4** | D = \|t_fc − t_obs\| − \|t̂_F − t_obs\|, paired per rate | upper end of the 95% percentile bootstrap interval of mean D < 0 |
| **E** | OLS slope of ln r_obs on ln ε̂_F | 95% CI inside [0.55, 0.80] (closed) |

- **Denominators (D1).** F, C1 and C2 count all 22 scored rates. A scored rate with no forecast, or with no crossing,
  is not within.
- **C3 (D2).** Over the scored rates with a forecast and a crossing (finite r_obs/r_fc, r_fc > 0). Fewer than 3 such
  rates, or Spearman NaN: UNRESOLVED.
- **C4 (D3).**
  - Any scored rate with no forecast: FAIL (the page).
  - Otherwise D runs over the scored rates with a crossing: 10,000 resamples with replacement, from a local
    `numpy.random.default_rng(2937000)`; the interval is the 2.5th and 97.5th percentiles.
  - PASS iff the upper end is < 0. An interval above 0, or containing 0, is FAIL. Fewer than 2 rates: UNRESOLVED.
- **E (D4).**
  - Over the scored rates with a forecast and a crossing above s_F (r_obs > 0).
  - The CI is the 2.5–97.5 percentile interval of the OLS slope over 10,000 rate resamples (local
    `default_rng(2937000)`). Resamples with no spread in ε̂ are dropped and counted.
  - Fewer than 3 rates: UNRESOLVED. The OLS standard error is descriptive.
- **Outcome (D10).**
  - The gate failing gives "UNRESOLVED (gate)"; validity failing gives "UNRESOLVED (validity)".
  - Otherwise PASS iff every criterion passes; UNRESOLVED if any criterion is UNRESOLVED; otherwise FAIL, naming each
    failing criterion.
  - This is the GELU-T / W2-A / Phase 1C rule. F's falsifier is reported as a flag in every case.

## 6. Gate and validity (`phase2a.gate`, `phase2a.validity`)

1. **Gate:** ≥ 27 of the 120 registered seeds land on M (P = 0.997 at the Wilson low 0.335 of the exploration's 80/200
   at s₀). If it fails, no rate is trained and the outcome is "UNRESOLVED (gate)".
2. **Validity (D5, D6).** If any condition fails, the outcome is "UNRESOLVED (validity)". The conditions:
   - **Follow:** at ≥ 90% of the 22 scored rates (≥ 20), the state at the first step with s ≥ 0.8·s_F, minimised at its own s
     (`sb_fold.local_min_batch`), lies within 1e−3 of M's exact point at that s. A rate that never reaches 0.8·s_F
     fails this.
   - **Cutoff before the crossing:** at ≥ 90% of the 22 scored rates (≥ 20), t_c < t_obs (strict). A rate with no cutoff fails
     this; a rate with a cutoff and no crossing passes it.
   - The two 90% rules are evaluated separately.
   - **Idle unit and signs:** at every scored rate, the idle unit's W, c and v are exactly 0.0, and sign(v_k) is unchanged
     for the three active units, at every step up to and including the crossing (to the end of observation if there
     is no crossing).
3. Scored rates whose cutoff is not before the crossing stay in every criterion's denominator (D14).
4. The constructed tests (`tests/test_phase2a.py`, "the 22 scored rates") confirm each rule with 22 scored rates
   among 27 run:
   - **F:** 20/22 PASS, 19/22 FAIL, a scored falsifier FAILs.
   - **C1 and C2:** 18/22 PASS, 17/22 FAIL.
   - **No forecast:** at a scored rate it fails C1, C2 and C4; at a pre-observed rate it does not.
   - **C3:** 3 finite ratios is computable, 2 is UNRESOLVED; a scored ratio below 1 FAILs.
   - **C4 and E:** both bootstraps have n = 22, and extreme values at the pre-observed rates change nothing.
   - **Validity:** follow and cutoff-before at 20/22 are valid and at 19/22 are not; an idle or sign violation at a
     scored rate makes the outcome UNRESOLVED (validity).
   - **Pre-observed rates:** breaking every rule at all 5, the falsifier included, leaves a PASS, and their
     descriptive block shows the failure.

## 7. Descriptive (never a verdict)

- **Each rate's forecast horizon**, t_obs − t_c, in steps and in delays ((t_obs − t_c)/delay_obs).
- **The tightening cutoff.**
  - 1 − f(ρ) = 0.05·(ρ/2⁻¹⁵)^{2/3}, calibrated so that it equals 0.05 at 2⁻¹⁵: f = 0.874 at 2⁻¹³, 0.95 at 2⁻¹⁵ and
    0.9875 at 2⁻¹⁸.
  - The same guarded forecaster runs at f(ρ), with its NaN check reported. The F–E statistics are computed at those
    cutoffs.
  - These are computed in `observe` (D15): for ρ < 2⁻¹⁵ the cutoff lies after the registered t_c.
- Per rate: ε̂_F, r_fc, r_obs/r_fc, delay_obs/delay_fc, max χ on [s\*, t_c), and the follow-check distance.
- **The 5 pre-observed rates** (2⁻¹³ … 2⁻¹⁷): F–E and validity over these 5 alone (denominator 5).
- The tightening-cutoff statistics are computed over the 22 scored rates.
- The landing counts (L0, M, S, S2, other) and the surplus M seeds.
- C4's fraction of rates with the forecast closer than t̂_F; E's OLS standard error.

## 8. Disclosures (‡ on the page, and others)

- **Set after exploratory data (‡):**
  - the rate ladder and its spacing; the pilot rates; C3's asymptotic form (it replaced the draft's band
    b ≈ 0.40); the E band [0.55, 0.80], set knowing the exploratory slope 0.683; f = 0.95, chosen after the exploration
    showed that C1 and C4 would fail at 0.90 on the new ladder.
- **Ladder rates already observed in the exploration.** Every run is deterministic given ρ, and the exploration used the
  same release, rule and data. So the exploratory outcomes of the ladder rates it ran are the registered outcomes
  (reproduced bit for bit at 2⁻¹⁰ by `test_real_run_matches_the_committed_exploration_at_2_to_minus_10`):
  - 2⁻¹³ and 2⁻¹⁴ (`p2a_explore_f`), and 2⁻¹⁵ (`p2a_explore_g`): the page's three anchors;
  - **not named on the page:** 2⁻¹⁶ (k = 8) and 2⁻¹⁷ (k = 16), also run in `p2a_explore_g`.
  - So 5 of the 27 rates have known outcomes before registration. The 2⁻¹⁵, 2⁻¹⁶ and 2⁻¹⁷ values at f = 0.95 are the
    page's expected-outcome table.
  - **The author's decision (2026-10-04):** these 5 are excluded from every scored set. They are run and reported as
    descriptive (§2).
- **The fixed dataset (disclosed limit; `p2a_explore_g`).**
  - 2A tests one landscape across rates. On resampled data from the same generator, the M branch moves: its stable
    ranges were 2.69–5.62 (seed 2,930,000) and 1.96–5.86 (seed 2,930,001).
  - s₀ = 1.7957 lies off M on both traced seeds.
  - One seed's branch (2,930,002) could not be traced: the data homotopy's Newton failed.
  - One run, seed 2,930,001 at 2⁻¹⁵ (released at 2.057 on its own M; |m′c′| = 6.9e−8; s_F/s\* = 1.09), did not cross
    by 1.25·s_F.
  - 2A therefore says nothing about sample-to-sample variation.
- **Scale only.** The rule slows the output scale, not the output vector. It is not the practitioner's knob.
- **Three-unit test.** The idle unit is exactly 0 and stays 0. Its free-training block is a saddle (coupling
  ≈ −0.027 at s\*): a 1e−6 seed is recruited in ~300 steps (exploration). So 2A tests the invariant three-unit
  subnetwork.
- **E band.** Set knowing the exploratory slope (0.683; local slope 0.70).
- **Λ_F metric.** |m′c′| and Λ_F are computed in training coordinates with P = I on the fixed-s tangent (the
  scale-only rule's metric there), not Track 1A's reduced η² coordinates.
- **ρ₂ every step.** The exploration evaluated ρ₂ every 20 steps and refined to the step. The registered observation
  evaluates it at every step (v3's convention), with a grid form of ρ₂ equal to `v2.rho2_batch` to 1e−13 (tested). It
  gives the identical t_obs at 2⁻¹⁰ (tested).
- **Set by me while preparing this file (not on the page; §9):** the bootstrap generator seed 2,937,000; the step
  budget; the observation end; the readings D1–D15.

## 9. The details the approved page left open (decided by the author 2026-10-04)

**S1 (decided: option (b)).**
- The page called the three pilot rates "off the ladder", but 2⁻¹⁶·⁵ = 2^−(15 + 12/8) is ladder rate k = 12. Each run is
  deterministic given ρ, so piloting it would have observed a registered rate.
- It was never run: `phase2a.pilot` refuses a ladder rate (`pilot_rate_allowed`, tested).
- The author moved it to **2⁻¹⁶·⁵⁶²⁵ (k 12.5)**, a half-step like 2⁻¹⁵·⁰⁶²⁵ (k 0.5) and 2⁻¹⁷·⁹³⁷⁵ (k 23.5).
- No substitute rate had been run before this decision.

**Details that touch a criterion, the gate, validity or the scoring set.** Approved as proposed (author, 2026-10-04),
with the scoring set narrowed to the 22 rates with no prior outcome (row D16).

| # | detail | proposed resolution (implemented, tested) | touches |
|---|---|---|---|
| D1 | Denominators of F, C1, C2 | All 22 scored rates. A scored rate with no crossing (by 1.25·s_F or the budget) is outside F's window and not within C1/C2; a rate with no forecast is not within C1/C2. | F, C1, C2 |
| D2 | C3's rate set | Scored rates with a forecast and a crossing (finite ratio, r_fc > 0); "> 1 at every rate" read over these; < 3 rates or Spearman NaN: UNRESOLVED; average ranks. | C3 |
| D3 | C4 details | Any scored no-forecast rate: C4 FAIL (literal reading of "fails C4"). Otherwise scored rates with a crossing; mean D; 10,000 resamples; default_rng(2937000); 2.5–97.5 percentiles; < 2 rates: UNRESOLVED. | C4 |
| D4 | E's 95% CI | Percentile bootstrap of the OLS slope over scored-rate resamples (10,000; default_rng(2937000)), closed band; scored rates with a forecast and r_obs > 0; < 3: UNRESOLVED. (Alternative: OLS t-interval, reported descriptively.) | E |
| D5 | The two 90% validity rules | Evaluated separately over the 22 scored rates (≥ 20 each); follow check = BFGS at own s within 1e−3 of M's exact point; never reaching 0.8·s_F fails; no cutoff = not before; cutoff and no crossing = before; t_c = t_obs is not before. | validity |
| D6 | Idle unit and signs | Idle W, c, v exactly 0.0 and active sign(v_k) unchanged at every step through the crossing step (to the end of observation if none). | validity |
| D7 | Step budget | ⌈64/ρ⌉ steps (≈ 2× the exploratory steps to 1.25·s_F). No cutoff within it: no forecast. Observation ending at the budget: no crossing. | F, C1, C2, C4, validity |
| D8 | Observation end | Stops once the crossing and s_F are both reached (t ≥ t_c), or at s ≥ 1.25·s_F. The page says "to 1.25·s_F"; nothing after the crossing enters any rule. | none (compute) |
| D9 | The τ errors and the pilot STOP rule | e₁ = \|t_fc − t_obs\|/delay_fc, e₂ = \|delay_fc − delay_obs\|/delay_fc at f = 0.95 (C1's and C2's statistics); τᵢ = ⌈1.5·max eᵢ/0.05⌉·0.05. A pilot run without a forecast or a crossing, or fewer than 3 pilot seeds on M: STOP. χ_t = (Δ log s over the last min(100, t) steps)/(ηλ_min(s_t)) (v3's `chi_at`), λ_min interpolated linearly in the frozen M table, over the steps with s ≥ s\* and t < t_c. | C1, C2 (τ) |
| D10 | Outcome rule | GELU-T / W2-A / 1C: gate, then validity, then UNRESOLVED / PASS / FAIL naming each; the falsifier is a flag on F. | outcome |
| D11 | Labels at s₀ | L0, M, S and S2 points at exactly s₀ (frozen; those inside a stable range), nearest within 1e−3, else "other"; one BFGS batch in seed order. | gate |
| D12 | Bootstrap seed | 2,937,000 (the first registered seed), as 1C used its first seed. | C4, E |
| D13 | Init generator | A local torch.Generator reproducing v3's init_net bit for bit. The exploration's landing fractions used numpy draws of the same distributions. | gate |
| D14 | Rates whose cutoff is not before the crossing | Kept in every criterion (every fraction over the 22 scored rates). 1C instead left such runs unscored. | scoring set |
| D15 | Tightening forecasts | Computed in `observe` (descriptive; guarded; NaN check reported). | none |
| D16 | **The scoring set (the author's addition)** | Verdicts over the 22 rates with no prior outcome. 2⁻¹³, 2⁻¹⁴, 2⁻¹⁵, 2⁻¹⁶ and 2⁻¹⁷ are run, and reported as descriptive with denominator 5. The gate is unchanged (≥ 27 of 120; all 27 run). | all criteria, validity, scoring set |

## 10. Competing outcomes and falsifiers

| outcome | reading |
|---|---|
| PASS (F, C1–C4, E, over the 22 scored rates) | The crossing comes at the fold, with a ρ^{2/3} delay forecast in advance from s before the cutoff only. |
| **F falsifier: a crossing at or below 1.25·s\*** | The switch, not the fold: the account is falsified. |
| F fails, crossings between 1.25·s\* and s_F | An early exit, as in v3. |
| F fails, no crossing by 1.25·s_F | H-F5 fails (the run does not escape into S past the fold). |
| E fails, slope ≈ 1 | A linear lag, not the fold delay. |
| E fails, slope ≈ 0, or C4 fails together with C1 | No rate-dependent delay, or the forecast horizon is too long. |
| C3 fails | The leading-order fold asymptotics fail: the ratio does not fall toward 1 as ρ → 0, or the delay is overestimated. |
| C1 or C2 fails alone | The fold arrival or the delay is mis-forecast beyond τ₁ = 0.15 or τ₂ = 0.25 of the forecast delay. |
| UNRESOLVED (gate) | Fewer than 27 seeds land on M. This says nothing about the fold. |
| UNRESOLVED (validity) | The runs leave M before 0.8·s_F, cutoffs come too late, or the idle unit or a sign moves. This says nothing about the fold. |

## 11. Frozen files and hashes

`results/phase2a/registration.sha256` (`phase2a.manifest`) is written by the registration commit. It lists:
- this file and the design page;
- the code closure of `src/phase2a.py` (`causal_forecast_fold`, `causal_forecast`, `sb_fold`, `simplicity_bias_v2`
  and their dependencies);
- `tests/test_phase2a.py` and `tests/test_causal_forecast.py`;
- `results/phase2a/{seed_scan,frozen,pilot}.json`, `results/simplicity_bias_v2/frozen.json`, and Track 1A's branch
  files (`results/sb_fold/branch_{M,L0,S,S2}_{fwd,bwd}.npz`, `branches.json`, `grid_M.npz`).

`run`, `observe` and `score` assert every hash, and that the manifest and `registration_stamp.txt` are committed and
unmodified.

**Freeze summary:** (§12).

## 12. Pilot and freeze (2026-10-04; pilot seeds only)

**Seed scan** (`seed_scan.json`): 0 files match either range; 0 overlaps with any module's registered or pilot seeds;
disjoint from the exploration seeds.

**Freeze** (`frozen.json`). No seed was drawn. It was re-run after the decision, to record the new pilot rate and the
scored and pre-observed sets; every other field is unchanged.
- s_F = 4.767689442106793 = 1.3275·s\* (`sb_fold.fold_of("M")`).
- M table: λ_min at 2,973 points of M's Δs = 0.001 grid, s = 1.795 … 4.767; smallest 3.64e−5 at the last point.
- |m′c′| = 4.782108e−7 (50 points in the last 0.05 below s_F); 4.821917e−7 over the last 0.025 (25 points).
- Λ_F = 1.509954e−3 (exploration 1.5099538e−3).
- Release θ_M(s₀): active units 0, 1, 2; ρ₂ = 0.08406; loss 0.291498; largest gradient on (W, c, b) of the active
  units 1.2e−12. SHA-256 95faef28….
- Branch points at s₀ for the labels: L0, M and S2 (S has no stable point at s₀).
- Budgets ⌈64/ρ⌉: 524,288 steps at 2⁻¹³ … 16,777,216 at 2⁻¹⁸.

**Pilot holds** (60 pilot seeds, one batch, 1.0 s): M 23 (38%), L0 14, other 23. The exploration had M 40% at s₀.
Assignment: 2,937,901 → 2⁻¹⁵·⁰⁶²⁵; 2,937,906 → 2⁻¹⁶·⁵⁶²⁵ (after S1; the superseded 2⁻¹⁶·⁵ entry, never run, is
kept in `pilot.json` under `superseded_runs`); 2,937,910 → 2⁻¹⁷·⁹³⁷⁵.

**Pilot runs** (f = 0.95; training to t_c, forecast, then observation resumed from the hashed t_c state):

| ρ | t_c | t̂_F | t_fc | delay_fc | t_F | t_obs | delay_obs | s_obs/s_F | r_obs/r_fc | e₁ (C1) | e₂ (C2) | horizon (steps, delays) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2⁻¹⁵·⁰⁶²⁵ | 513,274 | 587,302 | 608,285 | 20,983 | 586,508 | 610,369 | 23,861 | 1.01574 | 1.1937 | 0.0993 | 0.1372 | 97,095; 4.07 |
| 2⁻¹⁶·⁵⁶²⁵ | 1,451,331 | 1,660,629 | 1,690,036 | 29,407 | 1,658,346 | 1,690,744 | 32,398 | 1.00760 | 1.1526 | 0.0241 | 0.1017 | 239,413; 7.39 |
| 2⁻¹⁷·⁹³⁷⁵ | 3,763,915 | 4,306,693 | 4,346,931 | 40,238 | 4,300,721 | 4,343,925 | 43,204 | 1.00392 | 1.1208 | 0.0747 | 0.0737 | 580,010; 13.4 |

- In all three runs:
  - the guard's last row is t_c − 1, and the NaN recomputation is identical;
  - the state at t_c matched its hash bit for bit;
  - the follow check at 0.8·s_F is on M (distances 4e−9, 1e−9, 4e−8);
  - the idle unit stayed exactly 0 and the signs stayed fixed to the crossing.
- **STOP rule at 2⁻¹⁵·⁰⁶²⁵:** ε̂_F = 4.23e−4 ≤ 0.01; max χ_t on [s\*, t_c) = 0.0039 ≤ 0.1. Neither condition fires.
  The pilot STOP rule did not fire (`pilot.json`: stop false).
- **τ₁ = 0.15** = ⌈1.5 × 0.0993/0.05⌉·0.05: the largest e₁ is 0.0993, at 2⁻¹⁵·⁰⁶²⁵.
- **τ₂ = 0.25** = ⌈1.5 × 0.1372/0.05⌉·0.05: the largest e₂ is 0.1372, at 2⁻¹⁵·⁰⁶²⁵.
- **Machine.** Memory gate before every job (`memory_gate.log`). Peak RSS 1.01 GB (the 2⁻¹⁷·⁹³⁷⁵ observation). Times:
  2⁻¹⁵·⁰⁶²⁵ 31 s + 102 s; 2⁻¹⁶·⁵⁶²⁵ 87 s + 279 s; 2⁻¹⁷·⁹³⁷⁵ 228 s + 719 s.

## 13. Compute and machine rules

- **Compute:**
  - run: ≈ 43 M training steps at 55 µs (≈ 0.7 h);
  - observe: ≈ 50 M steps with ρ₂ every step at ≈ 165 µs (≈ 2.3 h);
  - one process.
- **Machine rules:**
  - one process, nice 15, one thread (`OMP_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1`, `torch.set_num_threads(1)`);
  - before every job, a memory gate (free ≥ 25%, swap free ≥ 500 MB; it waits) and a disk check (free ≥ 20 GB; else
    STOP), logged to `results/phase2a/memory_gate.log`;
  - stop above 3 GB RSS;
  - only small summaries and the hashed t_c states are saved, no paths.

## 14. Reproduce

```
python -m src.phase2a scan; python -m src.phase2a freeze; python -m src.phase2a pilot
python -m src.phase2a manifest             # the registration commit
python -m src.phase2a run                  # after the push and the OpenTimestamps proof
python -m src.phase2a finalize             # commit holds.jsonl, runs.jsonl, forecasts.sha256
python -m src.phase2a observe; python -m src.phase2a score
```
