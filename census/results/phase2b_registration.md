# Phase 2B registration: the lever (output learning rate) on the simplicity-bias benchmark

**REGISTERED.** This file, the approved design page, `src/phase2b.py` and the code it reaches, `tests/test_phase2b.py`,
the frozen inputs (`seed_scan.json`, `frozen.json`) and the exploration files they read are committed in one commit
before any registered run. Their SHA-256 hashes are in `results/phase2b/registration.sha256` (§13).
- **The open details (§11):** the author decided them on 2026-10-04/05. D1–D4 and D6–D9 were approved as proposed.
  **D5 was changed:** a non-finite run makes UNRESOLVED only the criteria that use that run's arm. Every criterion
  compares against arm 1, so a non-finite arm-1 run voids every criterion. The draft was 1531099.
- **Precondition met:** 2B registers only after Phase 2A's result. That result is in: registration 569b836,
  OpenTimestamps 45fcaed, result 7b15d9f (OUTCOME PASS).
- **Not yet done:** no registered seed (2,962,000–2,962,079) has been drawn, initialised or trained. There is no pilot
  (the page has none).
- **Before any registered run:** the registration commit is pushed and its OpenTimestamps proof is obtained. Then
  `run`, then `score`.

## 1. Design reference

- **Approved page:** `results/designs/phase2b_lever_design.md`, approved by the author on 2026-10-02. It was revised in
  5657286 and fb7b41a, first drafted in dbb7cdb, and marked approved in 8399b31. Everything on it is binding here.
- **The author's decisions (2026-10-02):**
  1. Arm 2 (v at lr/16) is primary. Arm 2b (v and b at lr/16) is secondary.
  2. Both global arms, ÷12.18 and ÷16.14.
  3. The matched scales {1.25, 2, 3}·s\* and BCE levels {0.1, 0.03, 0.01}.
  4. N is one-sided, with δ = 0.03 (ρ₂, reversed accuracy) and 0.02 (shuffled accuracy).
  5. O at 75%, with the onset "lasting" to T_C.
  6. The median paired bootstrap.
  7. 80 seeds.
  8. R_ℓ-acc is a named criterion.
  9. The T_C comparison is descriptive.
  10. Arm 2b keeps R_ℓ-acc as a registered criterion: its expected failure (power 0.18) is informative.
- **Exploration inputs:** commits e7c53fa, 68bdf76, a8f88c9 and bbf5406, plus
  `results/designs/phase2b_explore/README_lever.md` and its second addendum (5657286). They used the non-registered seeds
  2,953,000–2,953,019; the range 2,953,000–2,953,099 is never to be registered.

## 2. Setting, arms, seeds

- **Setting:** v3, unchanged.
  - The 800 points of `src/simplicity_bias_v2.data()`. Tanh width 4. The objective is BCE (mean) +
    (λ/2)(‖W‖² + ‖c‖²) with λ = 1e−4. Float64, full batch.
  - s = ‖v‖₁; q = 0.3914103370353161 and s\* = 3.5913755424683727 (both from v2's `frozen.json`).
- **Initialisation:** `src/simplicity_bias_v3.init_net(seed, cap = 0.5·s*)`.
  - It is called inside `torch.random.fork_rng`, so the global torch RNG state is restored (tested).
  - No warm start and no idle-unit zeroing.
- **Optimiser:** Adam (β 0.9, 0.999; ε 1e−8) with per-parameter learning rates. It is implemented in numpy with
  `torch.optim.Adam`'s formula. Tested against torch with parameter groups: relative difference < 1e−12 over 200 steps.
- **Budget:** every run trains **T_C = 40,000 steps** (states 0 … 40,000), with ρ₂ and BCE at every step.

| arm | code | learning rates | verdict |
|---|---|---|---|
| 1 standard | `std` | 0.01 on all 17 parameters | reference |
| 2 output, **primary** | `out16` | 0.01/16 on v (the 4 output weights); 0.01 on W, c, b | R_s, R_ℓ-ρ₂, R_ℓ-acc, O |
| 2b output, secondary | `out16b` | 0.01/16 on v and b; 0.01 on W, c | its own verdict: R_s, R_ℓ-ρ₂, R_ℓ-acc, O |
| 3 global | `glob` | 0.01/12.18 on all‡ | N |
| 3cm global, cost-matched | `globcm` | 0.01/16.14 on all‡ | N |

**Seeds:** 80 registered seeds, **2,962,000–2,962,079**, shared by all five arms (400 runs). Each seed is run in order,
arm 1 first, because the descriptive matched step reads arm 1's step to 3 s\*. Then arms 2, 2b, 3 and 3cm.

## 3. Order of the steps (`src/phase2b.py`)

| step | when | output |
|---|---|---|
| `scan` | before registration | `results/phase2b/seed_scan.json`: the 80 seeds are unused in src/, tests/, results/, paper/, notes/, independent/, data/ and dist/, and no registered test module's `SEEDS*` constant overlaps them |
| `freeze` | before registration; exploration seeds only | `results/phase2b/frozen.json`: the constants, the arms' learning-rate vectors, the cells and thresholds, the global factors recomputed from the committed exploration, and the exact reproduction of every arm's committed exploration run on seed 2,953,000 |
| `manifest` | the registration commit | `results/phase2b/registration.sha256` |
| `run` | after the registration commit and its OpenTimestamps proof | `results/phase2b/runs.jsonl` (resumable per (arm, seed); a memory gate before each run) |
| `score` | after all 400 runs | `results/phase2b/scores.json` (refuses an incomplete or duplicated set of runs) |

## 4. Measurements and matching (`phase2b.run_one`, `metrics`)

- **First passage, no interpolation.** The overshoot at the matched scales is reported.
  - Scale: the first step with s ≥ s_m, for s_m ∈ {1.25, 2, 3}·s\*.
  - Loss: the first step with BCE ≤ ℓ, for ℓ ∈ {0.1, 0.03, 0.01}‡. BCE is the mean data loss: the objective minus
    (λ/2)(‖W‖² + ‖c‖²).
- **At each matched point:** ρ₂ (the v2/v3 definition, evaluated exactly on the 40 × 40 product grid of the data's
  coordinate values; equal to `v2.rho2_batch`, tested) and two shifted accuracies.
  - Shuffled accuracy: x₁ is replaced by every x₁ of the 800 points, which is the exact expectation over a uniform
    permutation.
  - Reversed accuracy: x₁ → −x₁, which swaps the classes' x₁ distributions exactly (x₂ unchanged).
  - Prediction: class 1 iff z + b > 0.
- **Onset:** t_on is the first step such that ρ₂ ≥ q at every step from t_on to T_C. If ρ₂(T_C) < q there is no such
  step. If ρ₂ ≥ q from initialisation on, t_on = 0, at s(0) ≤ 0.5·s\*.
- **Also recorded per run:**
  - A non-finite check at every step.
  - The state at T_C, and the state at arm 1's step to 3 s\* (the matched step).
  - The SHA-256 of the (s, BCE, ρ₂) paths.
  - A 64-point trajectory sample.

## 5. Criteria (`phase2b.arm_verdicts`, `criterion_lower`, `criterion_N`, `criterion_O`)

**Paired differences.** Δ = arm − arm 1 on the same seed, per cell. There are 18 cells: the 6 matched points
× (ρ₂, shuffled, reversed). Every cell uses a 95% percentile bootstrap interval of the median Δ: 10,000 resamples,
seeds resampled jointly, `default_rng(20261002)` (details in §11).

| criterion | arms | cells | PASS |
|---|---|---|---|
| **R_s** | 2, 2b | ρ₂, shuffled and reversed accuracy at 1.25, 2, 3·s\* (9) | every lower end > 0 |
| **R_ℓ-ρ₂** | 2, 2b | ρ₂ at BCE 0.1, 0.03, 0.01 (3) | every lower end > 0 |
| **R_ℓ-acc** | 2, 2b | shuffled and reversed accuracy at BCE 0.1, 0.03, 0.01 (6) | every lower end > 0 |
| **O** | 2, 2b | — | ≥ 75%‡ of the arm's 80 runs have s(t_on) > s\*; a run with no t_on counts against |
| **N** | 3, 3cm (each its own) | all 18 | every upper end < δ: 0.03 for ρ₂ and reversed accuracy, 0.02 for shuffled‡ |

- **FAIL:** any criterion that does not pass, unless it is UNRESOLVED (§6).
- **Strict inequalities:** a lower end of exactly 0 does not pass, and an onset at exactly s\* is not above it.
- **N is one-sided:** a global arm that is worse than arm 1 passes N.

## 6. Validity: UNRESOLVED (`phase2b.reach_ok`, `score_tables`)

A criterion is **UNRESOLVED**, overriding both PASS and FAIL, in either case:
- **Reach:** fewer than 90% of the seeds reach one of its cells in both arms, i.e. fewer than 72 of 80. Applies to
  R_s, R_ℓ-ρ₂, R_ℓ-acc and N.
- **Non-finite:** a run of an arm the criterion uses is non-finite (§11, D5 and D6). Arm a's criteria use arm a and
  arm 1. So a non-finite arm-3 run voids N(3) only, a non-finite arm-2b run voids 2b's four criteria only, and a
  non-finite arm-1 run voids every criterion.

Every constructed case is tested (`tests/test_phase2b.py`):
- PASS and FAIL for each criterion and each cell family. N for each global arm, at its margins and one-sided.
- O at 60/80 (PASS) and 59/80 (FAIL); runs without an onset count against.
- Reach at 72/80 (resolved) and 71/80 (UNRESOLVED), with reach counted in both arms. Reach overrides a FAIL.
- Non-finite runs: in arm 3 only N(3) is UNRESOLVED, in arm 3cm only N(3cm), in arm 2b only 2b's criteria, in arm 2
  only arm 2's, and in arm 1 every criterion. All other verdicts are unchanged, including a FAIL elsewhere.
- Incomplete or duplicated runs are refused.

## 7. Outcome rule (`phase2b.outcome`)

**Primary outcome:** arm 2's R_s, R_ℓ-ρ₂, R_ℓ-acc and O, together with N for arm 3 and for arm 3cm.

| verdicts | primary outcome | statement |
|---|---|---|
| all six PASS | **PASS** | "All pass: the lever is the rate ratio, not the step count, and slab use still starts above s\*." |
| any UNRESOLVED | **UNRESOLVED** | lists the unresolved and failed criteria |
| N FAIL (arm 3 or 3cm) | **FAIL** | "N fails: global slowing helps too", per arm |
| only R_ℓ-ρ₂ and/or R_ℓ-acc FAIL | **FAIL** | "Only R_ℓ fails: a scale-only effect." |
| O FAIL | **FAIL** | "O fails: the onset moves to or below s\*." |
| other failures (e.g. R_s) | **FAIL** | listed, with no interpretation (the page names none) |

- **Secondary:** arm 2b's four verdicts are reported on their own and never change the primary outcome.
- **Expected for arm 2b:** it is expected to FAIL R_ℓ-acc (power 0.18; §10). That failure is informative, and it is a
  registered result.

## 8. Strict causal rule

**No criterion is a forecast.**
- Every criterion compares observed outcomes after every run has ended, so no cutoff applies.
  `score_tables` refuses to score anything but the complete set of 400 runs.
- `src/phase2b.py` reaches no forecaster. Its import closure contains no `causal_forecast*` module.
- A test asserts that no criterion function references a forecast. It also asserts that a forecast key added to every
  run row changes no verdict.

## 9. Descriptive, registered as such (`phase2b.descriptive`; never a verdict)

- **The end-of-training comparison at T_C:** ρ₂, both accuracies and s, as paired medians, seeds up/down and a
  descriptive bootstrap interval. **No endpoint advantage is claimed.** In the exploration, arm 1 was higher at T_C in
  17/20 seeds.
- **The matched step:** each arm's values at arm 1's step to 3 s\*, against arm 1 at 3 s\*.
- **Steps to each matched point, and step cost:** the ratio to arm 1 per seed.
- **Onset location:** s/s\* and s/s_F (s_F = 4.767689442106793, M's fold from `results/phase2a/frozen.json`). It is
  not tested against the fold. Also the counts in [1, 1.25]·s\*, from initialisation, and at or below s\*.
- **Two-sided equivalence for the global arms:** the same 95% intervals inside (−δ, δ).
- **The maximum overshoot** at the matched scales.

## 10. Disclosures

**Set from exploratory data (‡ on the page)**, all on the 20 exploration seeds 2,953,000–2,953,019:

- **Global factor 12.18.** It is the median over seeds of arm-2 ÷ arm-1 steps to 3 s\* (12.177, rounded).
  - It realises only ×9.454 steps to 3 s\* (r_g), because Adam's s-growth per step is not proportional to lr.
- **Cost-matched factor 16.14.** The cost ratio is modelled as G^a through (1, 1) and (12.18, r_g), so
  G_cm = 12.18^(ln 12.177/ln 9.454) = 16.1426 → 16.14. A linear rescaling would have given 15.69.
  - It was confirmed on the exploration seeds after it was fixed: ×11.78, i.e. 0.97 [0.92–1.01] of arm 2's steps to
    3 s\* per seed.
  - `freeze` recomputes both factors from the committed runs (tested).
- **δ margins:** 0.03 for ρ₂ and reversed accuracy, 0.02 for shuffled. They were set from the exploration, where the
  binding cell (ρ₂ at 1.25 s\*, arm 3) had an upper end of 0.028.
- **Matched levels:** the scales {1.25, 2, 3}·s\* and the BCE levels {0.1, 0.03, 0.01} were chosen from the explored
  grid.
- **The 75% O threshold.** On the exploration seeds, arm 2's lasting onset was above s\* in 18/20 (0.90; Wilson 95%
  lower bound 0.70). P(PASS) at n = 60 is 0.9998 at 0.90, but 0.24 at 0.70.
- **T_C = 40,000:** v3's cap.
  - In the exploration, "lasting" depends on the end of the run. With a 100,000-step end, 10/20 standard runs lost and
    regained ρ₂ ≥ q at large s.
  - Arm 2's onsets were identical at 3 s\*, 40,000 and 100,000.

**Other disclosures:**
- **v only versus v and b.**
  - Arm 2 slows the 4 output weights only; the output bias b stays at 0.01. Arm 2b slows v and b.
  - The exploration showed both similar at matched scale. Arm 2b was weak at matched loss on the shifted accuracies
    (shuffled at BCE 0.05 / 0.03: 11/9 seeds up/down).
  - Arm 2 is primary by the author's decision.
- **Arm 2b's expected R_ℓ-acc failure.**
  - On the exploration seeds its lower ends at BCE 0.1 and 0.03 are below 0: shuffled −0.003 and −0.010, reversed
    −0.010 and −0.019.
  - P(all pass) at n = 80 is 0.18.
  - The author kept R_ℓ-acc for 2b because its expected failure is informative (decision 10).
- **Power at n = 80** (resampling the exploration's paired differences; 300 simulations, inner bootstrap 1,000):
  arm 2 1.00, arm 2b 0.18, arm 3 0.96, arm 3cm 0.99.
- **The global arms' theoretical basis.**
  - Corollary L3 (math note §15.4): for **fixed-P** descent the lag is η-invariant, so a globally slowed run should
    match arm 1 at matched s. **It is not proved for Adam**, whose preconditioner moves (§15.5).
  - R4 (§13.6) is registered Adam evidence **in the sine family only**, not on this benchmark.
  - So N is an empirical test here, not a corollary. Arm 2 changes the output/hidden rate ratio, which L3 does not fix.
- **What the exploration showed:**
  - Global slowing gave no gain: |median Δρ₂| ≤ 0.013 up to 2 s\*, and slightly lower than arm 1 at 3 s\* (−0.043 for
    ÷12.18, −0.053 for ÷16.14).
  - Arm 2: Δρ₂ +0.25 to +0.31 at matched scale and +0.06 to +0.19 at matched loss (20/0 seeds).
  - Arm 1 is better at the matched step (Δρ₂ −0.24 for arm 2).
- **Exact reproduction of the exploration.** `run_one` reproduces the committed exploration runs bit for bit.
  - All five arms on seed 2,953,000 over 40,000 steps (`freeze`, §13).
  - The primary arm in the test suite, plus every arm's path over 3,000 steps on seed 2,953,001.
  - The bootstrap reproduces the committed exploration interval exactly. The criteria reproduce the committed
    exploration verdicts: arm 2 all pass, arm 2b R_ℓ-acc FAIL, N passes for arms 3 and 3cm.
- **One fixed dataset.** As in v3 and 2A, the data are the fixed 800 points; the seeds vary only the initialisation.

## 11. The details the approved page left open (decided by the author 2026-10-04/05)

Each detail below is not on the page and touches a criterion, a validity condition or the scoring set. The worker
stopped before the registration commit (draft 1531099) and proposed a resolution for each.
- **The author's decision:** D1–D4 and D6–D9 approved as proposed. **D5 changed** to the per-arm rule below.
- Every resolution is implemented in `src/phase2b.py` and tested.

| # | detail | registered resolution | source |
|---|---|---|---|
| D1 | The bootstrap generator across arms | A **fresh** `default_rng(20261002)` for each compared arm (2, 2b, 3, 3cm). One 10,000 × 80 index matrix is drawn from it and used for all 18 cells of that arm, so every arm and every cell see the same resamples. | `p2b_lever_power2.py` |
| D2 | Seeds that do not reach a cell in both arms | Excluded from that cell only: in each resample, the median is over the resampled seeds that reach it (`nanmedian`), and the point median likewise. Resampling stays over all 80 seeds. | `p2b_lever_power.ci_median` |
| D3 | The percentile rule | numpy's default (linear interpolation) at 2.5 and 97.5. | as explored |
| D4 | UNRESOLVED precedence | UNRESOLVED overrides PASS as well as FAIL. A criterion with < 90% reach, or with a non-finite run, is never PASS. | page: "It is UNRESOLVED if …" |
| D5 | Scope of "if any run is non-finite" | **Author's decision (changed from the literal reading proposed in the draft):** a non-finite run makes UNRESOLVED only the criteria that use that run's arm. Arm a's criteria (R_s, R_ℓ-ρ₂, R_ℓ-acc and O for an output arm; N for a global arm) use arm a and arm 1. So a non-finite arm-3 run voids N(3) only, a non-finite arm-2b run voids 2b's criteria only, and a non-finite arm-1 run voids every criterion. | author, 2026-10-04/05 |
| D6 | What "non-finite" means | At any step 0 … T_C, a parameter, the objective or its gradient, or ρ₂ is not finite; or a metric at a recorded point is not finite. The run stops at that step. | — |
| D7 | Reach, and the 90% count | A seed reaches a point if its first passage falls at a step in 0 … 40,000. 90% of 80 = 72, so fewer than 72 seeds reaching in both arms is UNRESOLVED. The rule has no cell for O, so O's only UNRESOLVED condition is a non-finite run of its arm or of arm 1 (D5). | — |
| D8 | The accuracy decision at a tie | Class 1 iff z + b > 0, so z + b = 0 is class 0 (measure zero). | as explored |
| D9 | An onset from initialisation | If ρ₂ ≥ q at every step from step 0, then t_on = 0 at s(0) ≤ 0.5·s\* < s\*. It counts against O, by the page's definition read literally. | the page's words |

**Not touching a criterion** (recorded; no decision was needed):
- The run order: per seed, arm 1 first.
- The primary-outcome grouping (arm 2 with both N; §7).
- The file formats.
- The descriptive matched step.
- s_F taken from 2A's `frozen.json`, for descriptive onset location only.
- One memory gate before every run.

## 12. Competing outcomes and falsifiers

- **All pass.** The lever is the output/hidden rate ratio, not the step count, and slab use still starts above s\*.
- **N fails** (arm 3 or 3cm). Global slowing helps too, so the step count, not the ratio, may be the lever.
- **Only R_ℓ fails.** The gain is a scale-only effect: at matched training loss arm 2 is not better.
- **O fails.** The onset moves to or below s\*: slowing the output rate changes when slab use starts, not only how much
  of it there is.
- **R_s fails.** The exploration's main effect does not replicate. The page names no interpretation for this.

## 13. Frozen files and hashes

**Scan and freeze, 2026-10-04.** No registered seed was drawn; the freeze used exploration seed 2,953,000 only.
- **Seed scan** (`results/phase2b/seed_scan.json`). No number in 2,962,000–2,962,079, written plain, with `_` or with
  `,`, appears in any text file under src/, tests/, results/, paper/, notes/, independent/, data/ or dist/ (2B's own
  files and the design page excepted). The seeds overlap no `SEEDS*` or `PILOT_SEEDS*` constant of track_a, track2a,
  track2b, track2c, gelu_transfer, width2_asym, phase1a_pilot, phase1c, phase2a, simplicity_bias_v2 or
  simplicity_bias_v3. They are disjoint from the exploration range 2,953,000–2,953,099. **unused: true.**
- **Freeze** (`results/phase2b/frozen.json`):
  - q, s\* and λ equal v2's `frozen.json`, and s_F equals 2A's `frozen.json`.
  - The global factors, recomputed from the committed exploration runs: r_o = 12.1773 → G = 12.18; r_g = 9.4541;
    a = 0.8987; G_cm = 16.14. Both reproduce.
  - **Every arm's committed exploration run on seed 2,953,000 is reproduced bit for bit** by `run_one` over 40,000
    steps. Every difference is exactly 0: the steps to every matched point and the matched step, 56–64 metrics, the
    lasting onset, and 217–233 trajectory samples. About 6.3 s per run.
- **SHA-256:**
  - `seed_scan.json` b298620f…
  - `frozen.json` 2d302633…
- **The manifest** `results/phase2b/registration.sha256` was written for the registration commit, after the author's
  decision on §11. It covers the import closure of `src/phase2b.py`, this file, the design page, the tests,
  `seed_scan.json`, `frozen.json`, v2's and 2A's `frozen.json`, and the exploration code, the exploration JSON and the
  100 exploration run files that the tests and the freeze read.

## 14. Compute and machine rules

- **Scale:** 400 runs × 40,000 steps, about 6.5 s per run unloaded (measured: 6.4–6.7 s), so about 45 min, and up to
  about 3 h under other load.
- **One process:** nice 15, one thread (`OMP/MKL/OPENBLAS/VECLIB_MAXIMUM_THREADS=1`; torch 1 thread). Peak RSS
  0.38 GB; stop above 3 GB.
- **Memory gate before every run:** free ≥ 25% and swap free ≥ 500 MB, else wait. Disk free ≥ 20 GB, else STOP.
  Logged to `results/phase2b/memory_gate.log`.

## 15. Reproduce

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 15 .venv/bin/python -m src.phase2b scan | freeze | manifest | run | score
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_phase2b.py
```
