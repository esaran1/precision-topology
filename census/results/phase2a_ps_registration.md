# Phase 2A-PS registration: the fold prediction with a sample per seed

**DRAFT — NOT REGISTERED.** Stopped before the registration commit under the standing rule: the details in §9 touch a
criterion, the gate, validity or the scoring set and are not on the approved page. Each has a proposed resolution,
implemented in `src/phase2a_ps.py` and tested in `tests/test_phase2a_ps.py`. **The author decides them first.**
- **Not yet done:** no registered seed has been trained. The 400 registered seeds are FROZEN only (no training; §12).
- **Before any registered training:** the registration commit is pushed and its OpenTimestamps proof is obtained.
- **The registration commit** will hold this file, the approved page, `src/phase2a_ps.py` and its import closure,
  `tests/test_phase2a_ps.py`, and the frozen inputs (`seed_scan.json`, `pilot*.json*`, `frozen_parts.jsonl`,
  `frozen.json`: EVERY seed's status and class, and the class listing's SHA-256). Their SHA-256 hashes go in
  `results/phase2a_ps/registration.sha256` (§11).

## 1. Design reference

- **Approved page:** `results/designs/phase2a_perseed_design.md` (approved by the author 2026-10-05, commit cec2726;
  decisions 1–9 as recommended; 400 seeds). Everything on it is binding here.
- **Seeds (author-approved 2026-10-05):** registered 2,975,000–2,975,399 (400); pilot 2,976,000–2,976,059 (60).
- **Phase 2A machinery reused unchanged** (registration 569b836, result 7b15d9f: PASS):
  - the scale-only rule (`phase2a.make_step`'s expression, on the seed's sample; bit for bit on the fixed data, tested);
  - the cutoff (`train_to_cutoff`), the hashed t_c state and the resume from it;
  - ρ₂ every step (Phase 2A's grid form, built from the seed's sample; equal to `v2.rho2_batch` to 1e−13, tested);
  - the crossing (first upward passage of q); the causal fold forecaster `src/causal_forecast_fold.py` UNCHANGED
    (leading order, r_fc = Ω₀ε̂_F^{2/3}), with its guard and NaN recomputation;
  - `phase2a.criterion_within` (C1, C2), `phase2a.outcome`, `phase2a.fold_constant_fit`, `phase2a.theta_from_u`.
- **Inputs:** the v2 frozen q = 0.3914103370; the fixed dataset's fold s_F = 4.767689442106793 and
  Λ_F = 1.509953872657247e−3 (2A `frozen.json`; asserted equal); Track 1A's stored M and S branches (the homotopy start);
  Phase 2A's POST HOC fit (`phase2a_posthoc.json`; descriptive only).
- **Exploration** (seeds 2,930,000–2,930,199 only; never registered): `results/designs/phase2a_explore/p2a_explore_g`,
  `_h` (the page's numbers), `_i` (the |m′c′| window check of §9 D1; run while preparing this file).

## 2. Setting, seeds and replicates

- **Each seed is its own landscape.** Its 800-point sample, M branch, fold, constants and causal forecast are its own:
  **the seeds are the replicates.** Otherwise as 2A: tanh, width 4, λ = 1e−4, BCE, float64, full batch, η = 1,
  s = ‖v‖₁, the scale-only rule, three active units (the idle unit exactly 0).
- **Sample (rule 1).** The benchmark generator with i.i.d. levels on the product grid (`p2a_explore_g.sample`, equal bit
  for bit, tested), numpy `default_rng([seed, 7])`: per class 20 × 20 points; x₁ levels 4 from U[−0.1, 0.1] and 16
  from U[0.1, 1]; x₂ 20 middle-slab levels (class 0) and 10 outer levels mirrored (class 1). Its SHA-256 is frozen.
- **Mechanism test (rule 2).** Each run is released at its seed's EXACT M(s₀) (Newton on the seed's own landscape), not
  trained from initialisation, and there is no BFGS hold.

## 3. Order of the steps (`src/phase2a_ps.py`)

1. **Before the registration commit:** `scan`, `pilot` (pilot seeds only), `freeze` (all 400 registered seeds, no
   training; → `frozen_parts.jsonl`, `frozen.json`), `manifest`. Then `stamp` (after the registration commit).
2. **`run`**, after the registration commit is pushed and OpenTimestamped: per planned (seed, rate), training from the
   seed's frozen release to t_c (first s ≥ 0.95·s_F(seed); NO ρ₂, no branch check), the state at t_c stored and hashed,
   the causal forecast with the seed's own (s_F, Λ_F), and the competitor: the same forecaster at the same t_c with the
   fixed-dataset fold (4.7677, 1.50995e−3) (`runs.jsonl`). Each run first re-derives the sample and the release and
   asserts their frozen hashes.
3. **`finalize`:** `forecasts.sha256` over `runs.jsonl` (every planned run exactly once, no observed key); both files are
   **committed before observation.**
4. **`observe`** asserts the registration and the committed forecasts, then per run: retrains with ρ₂ at EVERY step,
   asserts the t_c state equals the hashed one bit for bit and continues from it, stops once the crossing and s_F are
   both reached (t ≥ t_c), or at s ≥ 1.25·s_F, or at the budget; records t_obs, s_obs, t_F, the follow check at
   0.8·s_F, the idle unit and the signs (`observed.jsonl`).
5. **`score`** writes `scores.json`, with the counts of §6 at its top.

## 4. The per-seed freeze (rules 2–5; `freeze_raw`, `evaluate`)

`freeze_raw` records the raw quantities; `evaluate` applies the rules below. `summary` asserts every stored evaluation
equals a fresh one.

- **Branch identity (rule 3).** A data homotopy at fixed s = 4.49485 (Track 1A's stored M point nearest 4.5):
  X(λ) = (1 − λ)X_fixed + λX_seed, Newton at λ = 1/n, 2/n, …, 1 from the previous solution, n = 50; n = 200 if the
  50-step homotopy fails (Newton, or reduced λ_min ≤ 0 at some λ). **Accepted** iff:
  - Newton converges and the reduced λ_min > 0 at every λ;
  - the forward continuation (pseudo-arclength, h = 0.05, ≤ 800 points) finds an upper fold, and the contiguous stable
    run of continuation points containing the homotopy point reaches it (every forward point before the fold is
    stable);
  - ρ₂ < q at every point of that stable run and at M(0.9999·s_F) (`sb.feature_usage` on the seed's sample).
- **s_F (rule 4)** = the refined upper fold at h = 0.05. **Validated** iff:
  - the fold at h/2 = 0.025 agrees within 1e−6·s_F;
  - λ_min → 0: the reduced λ_min at the refined fold is within 1e−6 of 0, and the training-coordinate tangent λ_min
    falls toward the fold (λ_min² fit's linear coefficient > 0; λ_min at 0.001 below s_F < λ_min at 0.05 below);
  - no perturbed minimum on M above it: of the 30 perturbed minima at 1.001·s_F and of the 30 at 1.01·s_F, none lies
    within 1e−3 (function space) of M(0.9999·s_F).
- **|m′c′| and Λ_F (rule 4).** λ_min of the training-coordinate Hessian on the fixed-s tangent of the active units
  (P = I; 2A's metric) on the Δs = 0.001 grid over the last 0.05 below s_F. 2A's estimator, λ_min² = a·d + b·d²,
  d = s_F − s, |m′c′| = |a|/4: the value over the last 0.05, the check over the last 0.025. **Validated** iff they agree
  within 5%. Λ_F = √(|m′c′|·s_F). **(§9 D1: a STOP item.)**
- **s\*** = the scale where the seed's own M and S have equal loss. S is found by the same homotopy (from Track 1A's
  stored S point nearest 4.5, same acceptance) and continuation. s\* is the bisection root (to 1e−8) on
  [max(lo_M, lo_S) + 1e−3, min(hi_M, hi_S) − 1e−3], when the loss difference changes sign there; otherwise "no s\*".
- **Release and eligibility (rule 2).** lo = the smallest s of M's stable run (the exploration's lower end).
  s₀ = lo + FRAC0·(s_F − lo), FRAC0 = (1.7957 − 1.154)/(4.767689 − 1.154) = 0.177575 (the page's 0.1776). It maps 2A's
  1.7957. **Eligible** iff s₀ ≤ 0.6·s_F (inclusive). The release is θ_M(s₀): Newton from the nearest stable point,
  training coordinates, idle unit exactly 0. Its SHA-256 is frozen, as is M's point at 0.8·s_F (the follow-check anchor).
- **Class (rule 5).** From the 30 perturbed minima at 1.01·s_F: n = #(ρ₂ ≥ q). **clean** n ≥ 27, **none** n = 0,
  **mixed** 1 ≤ n ≤ 26.
  - Perturbations: one `default_rng(0)` per seed, drawn in the order 0.99, 0.999, 1.001, 1.01·s_F (30 × 16 normals
    each; the class reads the fourth block). δ = N(0, 1)·(0.05|P| + 0.02) around M(0.9999·s_F), with |η| kept.
  - Each perturbed point is minimised by fixed-s BFGS (`sb_fold.local_min_batch`, gtol 1e−8).
  - ρ₂ is `sb.feature_usage` on the seed's sample. This is exactly `p2a_explore_h`.
- **Status** = the FIRST failing reason, in this order:
  1. untraceable (M homotopy failed / no upper fold / homotopy point not stable / stable part does not reach the fold /
     ρ₂ ≥ q on the stable part);
  2. s_F not validated;
  3. Λ_F not validated;
  4. no s\*;
  5. release Newton failed;
  6. ineligible;
  7. otherwise **scoreable**.
  
  Only scoreable seeds are trained or scored. The others are counted and reported prominently.
- **Every seed's class and status are frozen and hashed** before the registration. The listing is "seed status class"
  per seed in seed order; its SHA-256 is in `frozen.json`, which is in the manifest.

## 5. Rates, plan and forecast (rules 6, 7)

- **Rates.** Clean seeds at 2⁻¹⁶ (scored) and 2⁻¹⁴ (per-seed exponent); none and mixed at 2⁻¹⁴.
- **Plan (frozen in `frozen.json`).** The first 40 clean, 30 none and 20 mixed SCOREABLE seeds in seed order. The rest
  are counted, not trained.
- **Budget.** ⌈64/ρ⌉ steps per run: 1,048,576 at 2⁻¹⁴ and 4,194,304 at 2⁻¹⁶. No cutoff within it: no forecast.
  Observation ending at the budget: no crossing.
- **Forecast.** f = 0.95·s_F(seed). `causal_forecast_fold.run_forecast` is used unchanged, with the seed's own
  (s_F, Λ_F), η = 1, and forecast budget ⌈64/ρ⌉ − t_c. It is leading order: r_fc = Ω₀ε̂_F^{2/3}, ŝ_c = s_F(1 + r_fc).
  - It reads s only, rows 0 … t_c − 1, through a `GuardedArray`. The guard's last row is asserted < t_c, and a
    recomputation with every row ≥ t_c set to NaN must be identical. Otherwise the run aborts (`CausalityViolation`).
  - **Competitor:** the same call at the same t_c with (4.767689442106793, 1.509953872657247e−3).
- **2A's fitted correction** (r = Aε^{2/3} + Cε ln(1/ε), A = 2.3194, C = 0.8030; `phase2a_posthoc.json`) is computed
  per seed and **shown descriptively only, labelled POST HOC.**

## 6. Criteria (`phase2a_ps.score_tables`)

Clean seeds at 2⁻¹⁶ unless stated. 10,000 bootstrap resamples over seeds, from a fresh local
`default_rng(2,975,000)`, with 2.5–97.5 percentiles. r_obs = s_obs/s_F − 1 (own s_F); t_F = the first step with
s ≥ s_F. **PASS needs F, H, E_seed, C3, C4 and P; C1 and C2 are secondary.**

| | statistic | PASS if |
|---|---|---|
| **F** | s_F ≤ s_obs ≤ 1.25·s_F (inclusive) | at ≥ 90% of the trained clean seeds, AND no falsifier: a crossing BELOW s_F (strict) and at or below 1.25·s\* (own s\*) |
| **H** | none seeds (2⁻¹⁴): reached s ≥ 1.25·s_F with no crossing | at ≥ 80% of the trained none seeds (≥ 10 needed) |
| **E_seed** | per clean seed Δln r_obs/Δln ε̂_F between 2⁻¹⁴ and 2⁻¹⁶; their median | 95% bootstrap CI of the median inside [0.55, 0.80] (closed) |
| **C3** | r_obs/r_fc | > 1 at ≥ 90% of the trained clean seeds |
| **C4** | D = \|t_fc − t_obs\| − \|t̂_F − t_obs\| | upper end of the 95% CI of mean D < 0 |
| **P** | D = \|ln ŝ_c/s_obs\| − \|ln ŝ_c^fixed/s_obs\| | upper end of the 95% CI of mean D < 0 |
| C1 | \|t_fc − t_obs\| ≤ 0.15·delay_fc | at ≥ 80% (SECONDARY; **registered expected FAIL**, 1/5 in exploration) |
| C2 | \|delay_fc − delay_obs\| ≤ 0.25·delay_fc | at ≥ 80% (SECONDARY; **registered expected FAIL**) |

- **Outcome** (2A's rule over F, H, E_seed, C3, C4, P):
  - the gate failing gives "UNRESOLVED (gate)"; validity failing gives "UNRESOLVED (validity)";
  - otherwise PASS iff all six pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each.
  - C1 and C2 are reported beside it and never change it. F's falsifier is reported in every case.
- **Mixed seeds** enter no criterion (descriptive).

## 7. Gate and validity

- **Gate (frozen before the registration):** ≥ 24 clean scoreable seeds of the 400. If it fails, nothing is trained;
  the outcome is "UNRESOLVED (gate)". The page's power: P(gate) = 0.987 at the exploration's Wilson low 0.089 (5/25),
  and 1.00 at 0.12 and above. With §9 D1, the exploration's clean scoreable fraction is 3/25 (§9).
- **Validity.** If any condition fails, the outcome is "UNRESOLVED (validity)":
  - at ≥ 90% of the trained clean seeds (2⁻¹⁶), the follow check lands on the seed's own M. The state at the first
    step with s ≥ 0.8·s_F is minimised at its own s and must lie within 1e−3 of the seed's M point at that s (Newton
    from the frozen M(0.8·s_F)). Never reaching 0.8·s_F fails.
  - at ≥ 90% of the trained clean seeds (2⁻¹⁶), t_c < t_obs (strict). No cutoff fails; a cutoff and no crossing passes.
    The two 90% rules are evaluated separately.
  - at EVERY clean run (2⁻¹⁶ and 2⁻¹⁴), the idle unit's W, c and v stay exactly 0.0, and sign(v_k) is unchanged for
    the active units, at every step up to the crossing (to the end of observation if none).

## 8. Disclosures

- **Grid levels over i.i.d. points (‡).** With i.i.d.-point resampling, S was untraceable on 7 of 9 seeds (no s\*).
- **A MECHANISM test.** Each run is released at its seed's exact M point. This is not training from initialisation.
- **Classes defined after exploration (‡).** The classification rule and EVERY seed's class are frozen and hashed (with
  the OpenTimestamped registration) before any training.
- **Untraceable fraction.** In exploration, M could not be traced on 9 of 25 grid seeds (36%). In the results it is
  reported prominently, with the counts of seeds lacking validated constants, lacking s\*, and ineligible (§12 for the
  registered seeds).
- **Falsifier.** Read as stated in F: it fires only on a crossing below s_F and at or below 1.25·s\*. It does not fire
  on a crossing at or above s_F (seeds with s_F/s\* < 1.25).
- **C1 and C2: registered expected FAIL (‡).** They passed on 1/5 clean seeds in exploration.
- **Leading-order forecast.** The forecaster is 2A's, unchanged. 2A's fitted correction is shown descriptively only,
  labelled POST HOC (it was fitted on 2A's results).
- **Set after exploratory data (‡ on the page):** the sample rule, the release fraction, the class rule and its
  thresholds, the rates, the gate, and C1/C2's expectation.
- **P's power** (≥ 0.98 at n = 10) rests on 11 exploratory values and is optimistic.
- **Exploratory outcomes on the registered design.** None: the exploration seeds are disjoint, and every seed has its
  own landscape. The exploration's 2⁻¹³ and 2⁻¹⁵ runs were not at the registered rates.
- **Set while preparing this file (not on the page): §9.**

## 9. Details the approved page left open

### Touching a criterion, the gate, validity or the scoring set: FOR THE AUTHOR (STOP)

| # | detail | proposed resolution (implemented, tested) | touches |
|---|---|---|---|
| **D1** | **Rule 4's two |m′c′| windows and estimator** (the page says "two windows agreeing ≤ 5%" and names neither) | 2A's estimator and windows: quadratic fit λ_min² = a·d + b·d², value over the last 0.05 (= the exploration's Λ_F, on which the page's numbers rest), check over the last 0.025, agreement ≤ 5% (inclusive). **Consequence** (`p2a_explore_i`, 16 traced exploration seeds): 9/16 agree. 7 disagree by more than 5%: 2,930,004 (84%), 006 (9.7%), 009 (18%), 010 (14%), 018 (32%), 020 (9.5%), 022 (26%). **2 of the 5 clean seeds fail (006, 010).** So the clean scoreable fraction is 3/25. Its Wilson low is 0.042, and P(gate ≥ 24 of 400) = 0.05 there (0.53 at 0.06, 0.95 at 0.08, 1.00 at 0.12). The disagreement is systematic, not noise: near a fold λ_min = √(4\|m′c′\|d)(1 + O(√d)), so λ²/d carries a √d term that the quadratic form lacks. A fold-form fit (a·d + c·d^{3/2} + b·d²) does not rescue the rule: still 9/16 and clean 3/5 (006 at 5.7%, 010 at 9.7%), and it changes Λ_F by ×0.84–2.75 (×1.03–1.23 on the 5 clean seeds). **Alternatives for the author:** (a) as proposed; (b) the fold-form estimator; (c) the agreement check reported descriptively, with Λ_F from the 0.05 window as in the exploration. Under (c), every traced seed with validated s_F keeps its exploration Λ_F. | scoring set, gate, every criterion (via Λ_F) |
| D2 | s_F validation thresholds (rule 4) | h vs h/2 within 1e−6·s_F (exploration max 4.4e−9 absolute). λ_min → 0: \|reduced λ_min at the refined fold\| ≤ 1e−6 (exploration ~1e−16), and the tangent λ_min falls toward the fold (fit coefficient > 0; last < first). "No perturbed minimum on M above it": 0 of 30 within 1e−3 of M(0.9999·s_F) at 1.001·s_F AND at 1.01·s_F (exploration: 0 on every traced seed). | scoring set |
| D3 | Branch identity details (rule 3) | Retry with 200 steps iff the 50-step homotopy fails Newton OR λ_min ≤ 0 (the exploration retried on Newton failure only; no exploration seed differs). "Upper fold" = a fold on the forward continuation (h 0.05, ≤ 800 points, s ≤ 60), with the contiguous stable run reaching it. "ρ₂ < q on its stable part" = at every stable continuation point (Δ arclength 0.05) and at M(0.9999·s_F) (the exploration used 12 points). lo = the smallest s in the stable run. S gets the same homotopy acceptance; s\* needs a sign change of the loss difference on the common stable range. | scoring set, eligibility |
| D4 | Seeds failing constant validation | Counted (as "s_F not validated" or "Λ_F not validated") and never scored, like untraceable and no-s\* seeds. Status = the first failing reason in §4's order. The gate and the caps count scoreable seeds only. | scoring set, gate |
| D5 | Class details (rule 5) | The perturbation recipe of `p2a_explore_h` exactly: default_rng(0) per seed, fourth block of 30; ρ₂ by `sb.feature_usage` on the seed's sample; q = 0.39141. The class is recorded for every traced seed, but counted only when the seed is scoreable. | scoring set, gate |
| D6 | Caps | Over SCOREABLE seeds in seed order (clean 40, none 30, mixed 20). Fewer than a cap: all are trained. The plan is frozen in `frozen.json` before the registration. | scoring set |
| D7 | FRAC0 | The exploration's exact (1.7957 − 1.154)/(4.767689 − 1.154) = 0.177575 (the page prints 0.1776). It shifts s₀ by ≈ 1e−4. | eligibility |
| D8 | F's denominator and the falsifier's scope | All trained clean seeds at 2⁻¹⁶; no crossing = not within. Falsifier: s_obs < s_F (strict) AND s_obs ≤ 1.25·s\* (inclusive), over the clean 2⁻¹⁶ runs. At every other run it is a descriptive flag. | F |
| D9 | H | Over the trained none seeds at 2⁻¹⁴. A seed counts iff observation reached s ≥ 1.25·s_F with no crossing. A run that ends at the budget below 1.25·s_F without crossing does NOT count (conservative). Fewer than 10 none seeds: UNRESOLVED. | H |
| D10 | E_seed | A seed's exponent needs a forecast (status ok) and r_obs > 0 at both rates; otherwise it is excluded and counted. The median; the percentile bootstrap of the median. Fewer than 3: UNRESOLVED. | E_seed |
| D11 | C3's denominator | All trained clean seeds at 2⁻¹⁶. No forecast or no crossing is not > 1. Ratio = r_obs/r_fc with r_fc > 0. | C3 |
| D12 | C4 | 2A's D3, per seed. Any trained clean seed with no forecast: FAIL. Otherwise seeds with a crossing; mean D; fewer than 2: UNRESOLVED. | C4 |
| D13 | P | ŝ_c is the forecaster's s_c_fc (defined whenever ε̂_F is, even if no ŝ reaches it). A crossing seed without its own ŝ_c: FAIL. A crossing seed without ŝ_c^fixed: excluded, counted. Seeds without a crossing: excluded, counted. Fewer than 2: UNRESOLVED. | P |
| D14 | C1, C2 denominators | All trained clean seeds at 2⁻¹⁶. No forecast, or no crossing, is not within (2A's D1). | C1, C2 |
| D15 | Validity scope | The follow and cutoff-before rules over the clean 2⁻¹⁶ runs. The idle and sign rule over EVERY clean run (2⁻¹⁶ and 2⁻¹⁴). None and mixed runs are not in validity (exploration: signs moved on a none seed). | validity |
| D16 | Bootstrap generator | default_rng(2,975,000) (the first registered seed), fresh for each criterion, as 2A used its first seed. | E_seed, C4, P |
| D17 | Forecast budget | The page's ⌈64/ρ⌉ − t_c (2A passed ⌈64/ρ⌉ − t_c + 1). Run budget ⌈64/ρ⌉; observation end as 2A's D8. | every forecast |

### Implementation only (not touching a criterion, the gate, validity or the scoring set)

- **Files.** Per-seed rows are appended to `frozen_parts.jsonl` (resumable); `frozen.json` holds the summary. The raw
  λ_min grid (50 values) is stored per seed, so a change to D1 or D2 re-evaluates without refreezing.
- **Hashes.** The sample's SHA-256 (little-endian float64 X then Y) and the release's SHA-256 are frozen and re-asserted
  by `run_one`/`observe_one`. The class listing's SHA-256 is in `frozen.json`.
- **Follow anchor.** The seed's M point at exactly 0.8·s_F (reduced coordinates) is frozen. The follow check runs Newton
  from it to the state's own s.
- **ρ₂.** During observation, ρ₂ is Phase 2A's grid form built from the seed's sample. For the class and identity it is
  `sb.feature_usage` (the exploration's).
- **POST HOC correction.** 2A's theory-backed free-A fit (ε̂_F source, all 27 rates) is read from
  `phase2a_posthoc.json`. It is descriptive only.
- **Machine rules.** One process, nice 15, one thread. A memory gate (free ≥ 25%, swap free ≥ 500 MB; waits) and a
  disk check (≥ 20 GB, else STOP) run before every job and between seeds, as a separate logged step before each job
  (`python -m src.phase2a_ps gate TAG`) and inside the job. Stop above 3 GB RSS. Every training run and every forecast
  has a hard budget.

## 10. Competing outcomes and falsifiers

| outcome | reading |
|---|---|
| PASS (F, H, E_seed, C3, C4, P) | Seed by seed, on landscapes that differ, the crossing comes at the seed's own fold. Its ρ^{2/3} delay is forecast from s before the cutoff, and the seed's own fold forecasts better than the fixed dataset's. |
| **F falsifier: a crossing below s_F and at or below 1.25·s\*** | The switch, not the fold: the account is falsified. |
| F fails: crossings between 1.25·s\* and s_F, or none by 1.25·s_F | An early exit (as in v3), or the escape into S fails past the fold. |
| H fails | Seeds whose static picture has no S-escape above the fold still cross: the class rule (H-F5) does not predict the dynamics. |
| E_seed fails, exponent ≈ 1 or ≈ 0 | A linear lag, or no rate-dependent delay. |
| C3 fails | The leading-order delay is not an under-estimate seed by seed. |
| C4 fails | The fold-delay forecast is no better than the no-delay forecast. |
| P fails | The seed's own fold carries no information beyond the fixed dataset's. |
| C1 or C2 fails (expected) | Secondary: the leading-order timing is not within τ. |
| UNRESOLVED (gate) | Fewer than 24 clean scoreable seeds. This says nothing about the fold. |
| UNRESOLVED (validity) | The runs leave M before 0.8·s_F, cutoffs come too late, or the idle unit or a sign moves. |

## 11. Frozen files and hashes

`results/phase2a_ps/registration.sha256` (`phase2a_ps.manifest`) will list:
- this file and the design page;
- the import closure of `src/phase2a_ps.py` (`phase2a`, `causal_forecast_fold`, `causal_forecast`, `sb_fold`,
  `simplicity_bias`, `simplicity_bias_v2`, …);
- `tests/test_phase2a_ps.py` and `tests/test_causal_forecast.py`;
- `results/phase2a_ps/{seed_scan.json, pilot.json, pilot_parts.jsonl, frozen_parts.jsonl, frozen.json}`;
- v2's and 2A's `frozen.json`, `phase2a_posthoc.json`, and Track 1A's M and S branch files;
- the exploration code `p2a_explore_g.py`, `p2a_explore_h.py` and `p2a_explore_h.jsonl`.

`run`, `observe` and `score` assert every hash, and that the manifest and `registration_stamp.txt` are committed and
unmodified.

## 12. Scan, pilot and freeze

(Filled in below as each step completes.)

## 13. Compute and machine rules

- **Freeze:** 400 seeds, ≈ 9–10 s per traced seed and < 1 s per untraceable one (≈ 1 h).
- **Runs (after registration):** 40 clean × (2⁻¹⁴ + 2⁻¹⁶), plus 30 none and 20 mixed at 2⁻¹⁴: ≈ 5.5 h (page).
- **Machine:** one process, nice 15, one thread; memory gate and disk check (§9); stop above 3 GB RSS; small files only.

## 14. Reproduce

```
python -m src.phase2a_ps gate scan; python -m src.phase2a_ps scan
python -m src.phase2a_ps gate pilot; python -m src.phase2a_ps pilot
python -m src.phase2a_ps gate freeze; python -m src.phase2a_ps freeze        # -> frozen.json (summary)
python -m src.phase2a_ps manifest       # the registration commit; then: stamp, commit, push, OpenTimestamps
python -m src.phase2a_ps run; python -m src.phase2a_ps finalize               # commit runs.jsonl, forecasts.sha256
python -m src.phase2a_ps observe; python -m src.phase2a_ps score
```
