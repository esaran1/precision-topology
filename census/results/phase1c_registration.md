# Phase 1C registration: the registered causal replication on fresh seeds

**STATUS: PREPARED, NOT YET REGISTERED (2026-10-01).** §9 lists details the approved page leaves open that touch a
criterion, the validity conditions or the scoring set. By the author's standing rule they come to the author before the
registration commit. Each has a proposed resolution, implemented in `src/phase1c.py` and tested in
`tests/test_phase1c.py`. Once the author approves them (or changes them), this status line is replaced by the
registration statement, the manifest is rewritten, and the registration commit is made. No registered seed has been
trained. No gap, placement or crossing of any training run of a 1C seed has been evaluated.

## 1. Design reference

- **Approved page:** `results/designs/phase1c_causal_design.md` (approved 2026-10-01 with changes 1–4; commit b08ae80;
  draft 6c01389). Its hash is in the manifest (§11). Everything on it is binding here. The decisions:
  1. W1: f = 0.90, τ_cross 10, τ_lag 15, b = 0.20. These were set from the est seeds after seeing their bias (§8).
  2. G, T and T′: f = 0.95, with the draft's tolerances.
  3. Descriptive only: horizons, the errors at the other f, and the run fraction closer than the no-lag forecast.
  4. C4 is replaced by the paired bootstrap interval (§5).
  5. Everything else is as drafted.
- **Inputs:**
  - Phase 1A, commit 25aac9e: `src/causal_forecast.py` (the frozen forecaster); `tests/test_causal_forecast.py`;
    `src/phase1a_pilot.py` (the driver whose `make_inputs` and `context` build the forecaster's guarded inputs; also
    W1's training, detection and s\*_frozen); `results/phase1a/landscape_w1.json`; `summary.json`.
  - The replicated registrations: Track A (`results/track_a_registration.md`), GELU-T
    (`results/gelu_transfer_registration.md`, 736b6bf) and W2-A (`results/width2_asym_registration.md`, 28b2432).
- **Gate to Phase 2:** W1 PASS, and G PASS or T PASS (`phase1c.phase2_gate`).

## 2. Arms, seeds and settings

| arm | replicates (pipeline called unchanged) | seeds | f (other f, descriptive) | τ_cross, τ_lag | b | gate |
|---|---|---|---|---|---|---|
| **W1** (required) | Track A, SGD lr 0.3, a = 1.58, s\*_pop = 2.090122 | 9,350,000–9,350,099 (100) | 0.90 (0.95) | 10, 15 | 0.20 | none (no hold); validity ≥ 60 crossings and ≥ 60 scored (Track A) |
| **G** | GELU-T random arm (ρ = 1, η = 0.03, 40,000 steps) | 9,360,000–9,360,079 (80) | 0.95 (0.90) | 10, 10 | 0.15 | ≥ 80% on a copy at release, regardless of G in the hold (GELU-T §3) |
| **T** | W2-A arm T (ρ = 2⁻¹⁰, η = 0.03, 100,000 steps) | 9,370,000–9,370,199 (200) | 0.95 (0.90) | 15, 5 | 0.10 | ≥ 60 runs on T, regardless of G in the hold (W2-A §4) |
| **T′** (own verdict) | W2-A arm T′ (ρ = 2⁻¹², η = 0.03, 400,000 steps) | 9,380,000–9,380,119 (120) | 0.95 (0.90) | 30, 5 | 0.10 | ≥ 108/120 on T′ and no run with G > 0 at the start or any hold step (W2-A §4) |

- **Seeds are fresh.** `results/phase1c/seed_scan.json` (`phase1c.scan`) records two checks:
  - No 7-digit number with an arm's 4-digit prefix (9350, 9360, 9370 or 9380; also written 9_350_xxx or 9,350,xxx)
    occurs in any text file under src/, tests/, results/ or paper/. 1C's own files are excepted.
  - No 1C seed is a registered or pilot seed of any test module (`registered_overlap`), Phase 1A included.
- **Per-seed draws, as in each registration:**
  - W1: `fold1d.make_data(200, seed)`, and U(−1, 1)⁴ in float32 from a local torch Generator, cast to float64.
  - G: the sample and the Generator draw of GELU-T §3.
  - T: the sample and `width2_train`'s draw, from a local generator.
  - T′: θ\*_T′,pop(s₀).
  - No global RNG state is used anywhere.

## 3. Carried over unchanged (references)

| item | W1 | G | T, T′ |
|---|---|---|---|
| landscape | Track A §3 construction at 1.58 (`phase1a/landscape_w1.json`, validated: global_min unplaced at 0.995·s\*, placed at 1.005·s\*) | GELU-T §2, `gelu_transfer/landscape.json` | W2-A §6, `width2_asym/landscape.json` |
| per-seed frozen inputs (before training) | Track A R0: s\*_frozen (`phase1a_pilot.w1_frozen`); the own-sample threshold, descriptive (`own_threshold`) | GELU-T §6 (`gelu_transfer.freeze_one`: both copies, switches, κ, follow points, W) | W2-A §6 (`width2_asym.freeze_copy`): the arm's scored copy in full, the other classified copy as a point; `arm_hold_steps` |
| start, hold, release, classification, windings | none (free training from the draw, Track A §4) | GELU-T §3 (`run_start`) | W2-A §4 (`run_start`; the winding pair fixed at release) |
| ρ, pilot median χ (V6) | none | `gelu_transfer/pilot.json` (ρ = 1; χ 0.022408) | `width2_asym/pilot.json` (T 2⁻¹⁰, χ 0.025867; T′ 2⁻¹², χ 0.016168) |
| training | `phase1a_pilot.w1_train` (Track A's SGD, 32,000 steps) | `release_train` (40,000) | `train_path` (`budget(arm, ρ)`) |
| detection (every step, full budget) | `phase2b_ordering.state` (`w1_observe`) | `phase2b_ordering.state` (GELU-T `observe`) | `observe_path` (enclosure lower end > 0) |
| actual lag-free switch | first t with s ≥ s\*_run (the cutoff's rule-point branch, §4) | the copy's frozen switch, first t ≥ | `own_path_switch` on the actual v path (W2-A §5, change C) |
| follow check | none (Track A has none) | GELU-T §5, §13 at t₀.₈ | W2-A §9 at t₀.₈ |
| gate | none | GELU-T `gate` (random) | W2-A `gate` (T, Tp) |
| validity | Track A §6: ≥ 60 crossings; each criterion ≥ 60 runs | GELU-T §9, V1–V7 (`gelu_transfer.score_arm`) | W2-A §10, V1–V7 with changes 3 (`width2_asym.score_arm`) |
| validity inputs (κ, ηλ, κ/(ηλ), χ at t_sw, window χ) | — | registered `predict_one(with_traj=False)` | registered `predict_one(with_traj=False)` |
| outcome rule | — | PASS if all pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL naming each failure | same |

Only the predictions change. The registered predictions (r_traj, r_cf as a prediction) are not computed or scored in
1C. The registered r_cf (κχ at the actual t_sw) is computed because V5 reads it.

## 4. The 1C prediction (causal) and the order of the steps

- **Cutoff (a stopping time; harness side, `phase1c.harness_cutoff`).**
  - G, T, T′: t_c = the first t ≥ 1 with s_t ≥ f·s_ref, where s_ref is the occupied copy's frozen switch.
  - W1: the nested stopping time on s\*_run (`causal_forecast.w1_cutoff_on_s_run`). s\*_run is the switch of the branch
    at the causal rule point, the first t with s ≥ 0.5·s\*_frozen at every step up to t_c − 1. This **replaces
    Track A's R1**, whose rule point used t_top, a step after the cutoff.
- **Forecaster.** `causal_forecast.run_forecast` (25aac9e), unchanged:
  - `quad` extrapolation of log s over the last ⌈0.05·(t_c − t_start)⌉ steps (at least 50). W2-A also extrapolates
    the share. The extrapolation runs to 1.6·s_ref.
  - R4 along [visible path, extrapolated path].
  - The lag-free switch comes from the extrapolated path: W2-A uses `own_path_switch` along the extrapolated v; G and
    W1 use the first extrapolated ŝ at or above the frozen switch (W1: s\*_run).
  - Its inputs are the Phase 1A driver's `make_inputs` with `guarded=True`. Every array is a `GuardedArray` view of
    rows < t_c. The hidden state allows one row: release (G, T, T′) or the rule point (W1). `run_forecast`
    re-asserts every guard's last row < t_c.
- **Per-run NaN recomputation.** The same forecaster runs unguarded on copies in which every output row ≥ t_c and
  every hidden row ≥ t_c is NaN. `nan_recompute_identical` records whether the output is identical; validity needs it
  (§6).
- **Order** (`src/phase1c.py`):
  1. `run`, per seed after the registration commit and its timestamp:
     - start, hold and release, then training to the full budget, with no gap or placement evaluated after release;
     - the forecasts at f and at the other f;
     - the output-path prefix (to 1.6·s_ref + 500 steps; it never uses a crossing) and the hidden rows the forecaster
       may read, saved untracked with their SHA-256 in the row.
  2. `finalize`: `forecasts.sha256`, with every seed exactly once and no observed key in any row. The runs files and
     the hash file are **committed before observation**.
  3. `observe` asserts the registration and the committed forecasts, then for each scored-copy run:
     - retrains it deterministically and asserts the saved prefix bit for bit;
     - runs every-step detection over the full budget;
     - computes the actual switch, the follow check and the registered validity inputs.
  4. `score` writes `scores.json`.
- **Enforcement** (`tests/test_phase1c.py`). The whole 1C prediction is checked on real W1, G and T inputs:
  - its last row read is < t_c, and its NaN recomputation is identical;
  - poisoning every row after t_c gives an identical record (the harness reads s at t_c, a stopping time);
  - a forecaster that reads the cutoff row raises `CausalityViolation`;
  - a forecaster that bypasses the guard (reads the private array) is caught by the NaN recomputation, and the arm
    becomes not valid.

## 5. Criteria (per arm; over the scored runs, §6; `phase1c.criteria`)

lag_fc = t_fc − t_sw,fc and lag_obs = t_obs − t_sw, both signed and in steps. r_fc = s_fc/s_sw,fc − 1 and r_obs =
s_obs/s_sw − 1.

| | statistic | PASS if |
|---|---|---|
| **C1** | \|t_fc − t_obs\| ≤ τ_cross (inclusive) | in ≥ 80% of the scored runs (a miss is not within) |
| **C2** | \|lag_fc − lag_obs\| ≤ τ_lag (inclusive) | in ≥ 80% of the scored runs (a miss is not within) |
| **C3** | median of r_obs/r_fc over the scored runs with a forecast | ∈ [1 − b, 1 + b] (closed) |
| **C4** | D = \|t_fc − t_obs\| − \|t_nolag − t_obs\|, t_nolag = t_sw,fc, paired per run over the scored runs with a forecast | the upper end of the 95% percentile bootstrap interval of mean D is < 0 |
| **S** (T′) | lag_fc < 0 and lag_obs < 0 | in ≥ 80% of the scored runs with a forecast and \|lag_fc\| ≥ 6 |

- **C4, as implemented** (`criterion_c4`, `bootstrap_mean_ci`):
  - 10,000 resamples of the runs, with replacement, from `numpy.random.default_rng(9350000)` (a local generator);
  - the interval is the 2.5th and 97.5th percentiles of the resampled means; this is GELU-T's and W2-A's procedure
    (tested equal);
  - **reading: PASS iff the upper end is < 0. An interval entirely above 0, or containing 0, is FAIL;**
  - fewer than 2 runs, or a non-finite interval, is UNRESOLVED;
  - the run fraction with D < 0 (the draft's C4) is reported as DESCRIPTIVE.
- **Not computable:** C1 or C2 with no scored run, C3 with no finite ratio, and S with no eligible run are UNRESOLVED.
- **Outcome per arm:** PASS iff every criterion passes. UNRESOLVED if any is UNRESOLVED. Otherwise FAIL, naming each
  failing criterion. Numbers are compared in float64 without rounding.

## 6. Gate, scored runs and validity (`phase1c.score_arm_1c`)

1. **Gate** (G, T, T′): the registered gate, unchanged. If it fails, every criterion is UNRESOLVED and the outcome is
   "UNRESOLVED (gate)".
2. **Crossing runs (the base).**
   - W1: a frozen s\*_frozen and a crossing within the budget.
   - G, T, T′: the registered scored-run conditions without the registered predictions: on a scored copy at release
     (valid copy, winding in the table), follows its branch, crosses within the budget, with finite r_cf and r_obs.
3. **Scored runs:** the base with the cutoff **strictly** before the crossing (t_c < t_obs). W1 also needs the actual
   switch t_sw to be defined.
   - Later cutoffs, and crossing runs with no cutoff, are unscored and counted.
   - A scored run without a forecast (t_fc or t_sw,fc undefined) is a **miss**.
4. **Validity.** If any condition fails, every criterion is UNRESOLVED and the outcome is "UNRESOLVED (validity)". The
   conditions:
   - **W1:** ≥ 60 crossings and ≥ 60 scored runs (Track A's counts).
   - **G, T, T′:** the registered V1–V7 over the scored runs. They come from the registered function, which is given a
     finite placeholder for the registered predictions exactly on the scored runs, so that its scored set is the 1C
     scored set, misses included. V2 keeps its registered denominator; V1 counts misses.
   - **Every arm:** the cutoff is strictly before the crossing in ≥ 90% of the base.
   - **Every arm:** the NaN recomputation is identical in every run with a cutoff at the arm's f.
5. **Follow check:** it reads one state at t₀.₈ < t_c (0.8 < 0.95, against the same frozen switch). Validity reads the
   full path only after the forecasts are committed.

## 7. Descriptive (never a verdict)

- Per arm, over the scored runs: the forecast horizon t_obs − t_c and t_sw − t_c, in steps and in lags (÷ max(|lag_obs|,
  1)).
- The criteria statistics at the other f (0.95 for W1; 0.9 for G, T and T′), over the base runs whose other-f cutoff
  is before the crossing.
- The run fraction with the forecast closer than the no-lag forecast.
- **L1–L5 on r_fc.** These come from the registered function with r_traj := r_fc, s_traj := s_fc and t_traj := t_fc,
  over the scored runs with a forecast:
  - W1 uses Track A's L1–L3; its L2 is Track A's R5 closed form, κ_SGD(1.58)·χ, with ṡ at the actual t_sw.
  - G, T and T′ use L2 with the registered r_cf.
  - Neither L2 is a 1C forecast.
- G and T: the criteria verdicts without the runs with G > 0 in the hold (the registered sensitivity analysis).
- The number of runs whose cutoff is not before the crossing, the misses, and the follow-check failures.

## 8. Disclosures: what was decided after seeing pilot data (‡ on the page)

All pilot data are Phase 1A data: the dev seeds (family and window) and the est seeds (error estimate). No 1C seed has
been drawn.

- **f.**
  - The Phase 1A rule gave f = 0.8 (one-lag rule) or the grid {0.8, …, 0.98} (minimal rule). The recommendation 0.95
    was flagged as made after the est results. The author chose 0.95 for G, T and T′.
  - For W1 the author chose **0.90 after seeing the est-seed bias** (median r_obs/r_fc 1.094 at 0.95, 0.006 inside
    the draft's band; 1.172 at 0.9). At 0.9 the cutoff lies 73–151 steps before the actual switch (1.9–2.7 lags to the
    crossing); at 0.95 it lies 37–76 steps before.
- **W1 tolerances (change 1).**
  - The rule: 1.5 × est-seed q90, rounded up to the next multiple of 5 steps (`phase1a_pilot._tolerances`, the
    Phase 1A rule), applied to the est seeds at f = 0.9 after they were seen:
    - q90 \|t_fc − t_obs\| = 4.0 steps → 6.0 → **τ_cross = 10**;
    - q90 \|lag_fc − lag_obs\| = 6.9 → 10.35 → **τ_lag = 15**.
  - **b = 0.20**, set by the author. Phase 1A's band rule gives the same value: \|1.172 − 1\| + 2·SE = 0.181 → 0.20.
  - The est seeds pass at these settings: C1 32/32, C2 32/32, C3 1.172, C4 interval [−79.5, −74.9].
- **G, T, T′ tolerances:** the Phase 1A proposal at f = 0.95, set from the est seeds.
- **The draft's thresholds:** 80% (C1, C2) and the ≥ 50% run fraction were set by the author on the est seeds.
- **Forecaster choices:**
  - family `quad`, window 0.05: chosen on the dev seeds by a rule revised after the first dev pass (Phase 1A §4);
  - horizon 1.6: raised from 1.3 during the dev pass;
  - the W1 cutoff on s\*_run: adopted after the est range 9,310,100–119 was inspected and relabelled dev.
- **a = 1.58:** chosen in Phase 1A. No a-column of any results CSV equals 1.58. The pilot crossings at 1.58 are on
  the Phase 1A seeds only.
- **Seed counts:** chosen with binomial gate probabilities at the pilot rates (the page's table).
- **Set by me while preparing this file (not on the page; §9):**
  - the C4 bootstrap generator seed 9,350,000;
  - the four seed ranges.
- **Pilot check.** `results/phase1c/pilot_check.json` (`phase1c.pilot_check`) applies the 1C criteria code to the
  committed Phase 1A est forecasts. At each arm's f every criterion passes:

  | arm | C1 | C2 | C3 | C4 interval | S |
  |---|---|---|---|---|---|
  | W1 | 32/32 | 32/32 | 1.172 | [−79.5, −74.9] | — |
  | G | 19/19 | 19/19 | 0.942 | [−28.4, −12.5] | — |
  | T | 26/26 | 26/26 | 0.975 | [−15.3, −6.4] | — |
  | T′ | 20/20 | 20/20 | 1.019 | [−49.8, −36.0] | pass |

  At the other f, C1 fails for G, T and T′; C3 and C4 also fail for T. This is descriptive, and it is why 0.95 was
  chosen.

## 9. Details the approved page leaves open

**STOP items.** Each of these touches a criterion, a validity condition or the scoring set. The author decides them
before the registration commit. The proposed resolution is implemented and tested.

| # | detail | proposed resolution (implemented) | touches |
|---|---|---|---|
| D1 | **The 1C scored set.** | The base (§6.2) with t_c < t_obs. G, T and T′ keep the registered conditions except the registered predictions: finite r_traj and s_traj are dropped; finite r_cf (V5's input) and r_obs are kept. W1 also needs the actual t_sw to be defined (the analogue of finite r_cf). | scoring set |
| D2 | **Misses in each criterion.** | C1 and C2: a miss is in the denominator and not within. C3, C4 and S: misses are excluded, because they have no r_fc, t_fc or lag_fc; they already fail C1 and C2. A forecast with t_fc but no t_sw,fc is a miss. | C1–C4, S |
| D3 | **V1–V7 with the new predictions.** | The registered V1–V7, computed by the registered function over the 1C scored set (D1), misses included, so V1 counts misses. The registered predictions are replaced by a finite placeholder only to select that set. Every V reads only frozen and observed quantities, so their values do not depend on the placeholder (tested equal to the registered call). | validity |
| D4 | **W1 validity.** | Track A's counts carried over: ≥ 60 crossings within the budget, and ≥ 60 scored runs, misses included ("≥ 60 scored" on the page). | validity |
| D5 | **The 90% cutoff rule.** | Its denominator is the base (§6.2): crossing runs on a scored copy that follow their branch, with no cutoff counted as not before. A failure makes the arm UNRESOLVED (validity). It is computed at the arm's f only. | validity |
| D6 | **A NaN recomputation that differs.** | Any run with a cutoff at the arm's f whose recomputation differs makes the arm UNRESOLVED (validity). It is not treated as a per-run miss: a differing recomputation means the forecaster read past the cutoff. | validity |
| D7 | **C4 details.** | The statistic is the MEAN of D (as in the registered L4 and L5). The generator is default_rng(9350000). The interval is the 2.5–97.5 percentiles. Fewer than 2 runs is UNRESOLVED. | C4 |
| D8 | **C3 and "not computable".** | A run with r_fc = 0 has no ratio and is excluded. C3 with no finite ratio is UNRESOLVED. C1 and C2 with no scored run are UNRESOLVED. S with no run having \|lag_fc\| ≥ 6 is UNRESOLVED. | C1–C3, S |
| D9 | **S with lag_obs undefined.** | It counts as not both negative. For T′ this cannot occur in a scored run, because finite r_obs needs the own-path switch. | S |
| D10 | **W1's observed switch.** | The actual t_sw is the first step with s ≥ s\*_run, where s\*_run is the branch at the arm's-f cutoff rule point (the forecaster's own). Track A's t_top rule point is not used (the page replaces R1). r_obs = s_obs/s\*_run − 1. | C2, C3 |

**Implementation only.** These do not touch a criterion, a gate, validity or the scoring set.

1. **Disk.**
   - The output path is saved only up to the first step with s ≥ 1.6·s_ref, plus 500 steps (compressed, untracked,
     hashed).
   - `observe` retrains each scored-copy run deterministically to the full budget. It asserts the saved prefix bit for
     bit and detects on the full path, so the budgets are unchanged.
   - W2-A saved full paths (3.8 GB). Here the disk is at 91%.
2. **Runs not on a scored copy** (or on an invalid copy, or with a winding outside the table) get no forecast and no
   observation. They are counted from the runs file, and their release record enters the gate.
3. **The forecasts at the other f** are computed in the same `run` step, before observation. They are descriptive, and
   their NaN check is reported but is not a validity condition.
4. **W1 per-seed frozen inputs:** s\*_frozen and, as in Track A R0, the own-sample threshold (descriptive).
5. **The descriptive L1–L5 definitions** (§7), and the G and T sensitivity analyses.
6. **The four seed ranges** (prefixes 9350, 9360, 9370 and 9380), with the scan pattern of Phase 1A widened to the
   comma form.
7. **The ledger** lists every 1C artifact as a producer (`verify_ledger.PRODUCERS`).

## 10. Competing outcomes and falsifiers (per arm)

| outcome | reading |
|---|---|
| PASS (C1–C4, + S for T′) | The crossing and the signed lag are forecast before they happen, from data before the cutoff only. |
| C1 fails, C2 passes | The switch is mis-extrapolated, not the lag. |
| C2 fails | The lag itself is not forecast. |
| C3 fails | The scale of the lag is off by more than b. |
| **C4 fails (falsifier)** | The forecast is no better than ignoring the lag (an interval above 0 or containing 0). |
| S fails (T′) | The sign of the lag (crossing before the switch) is not forecast. |
| UNRESOLVED (gate) | The start does not reach the branch, as in the replicated test's gate. This says nothing about the lag. |
| UNRESOLVED (validity) | Too few scored runs; cutoffs too late (< 90% before the crossing); the replicated test's V1–V7; or a forecast that read past its cutoff (NaN check). This says nothing about the lag. |
| Phase 2 gate fails | W1 does not PASS, or neither G nor T passes. |

## 11. Frozen files and hashes

`results/phase1c/registration.sha256` (`phase1c.manifest`) is written by the registration commit and lists:

- this file and the design page;
- the code closure of `src/phase1c.py` (every src module reached by relative imports, including `causal_forecast`,
  `phase1a_pilot`, `track_a`, `gelu_transfer`, `width2_asym` and their dependencies);
- `tests/test_phase1c.py` and `tests/test_causal_forecast.py`;
- the frozen per-seed files `results/phase1c/frozen_{W1,G,T,Tp}.jsonl` and `results/phase1c/seed_scan.json`;
- every frozen input the replicated pipelines read: `phase1a/landscape_w1.json`, `phase1a/summary.json`, the Phase 1A
  fixtures, `gelu_transfer/{landscape,pilot}.json`, `act_general/kappa_gelu_frozen.json`,
  `width2_asym/{landscape,pilot}.json`, `asym_scores.json` and `lag_law/kappa.csv`.

`run`, `observe` and `score` assert every hash, and that the manifest is committed and unmodified.
`results/phase1c/registration_stamp.txt` records the registration commit and the SHA-256 of this file and of the
manifest. Its OpenTimestamps proof is added after the push. No registered seed runs before the push and the proof.

## 12. Compute and machine rules

- **Compute:** from Phase 1A's per-run times, about 1.2 CPU-hours for training and forecasts, plus the same again for
  the retraining at observation.
- **Machine rules:**
  - one process, nice 15, one thread (`OMP_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1`, `torch.set_num_threads(1)`);
  - a memory gate before every job and between seeds: free ≥ 25% and swap free ≥ 500 MB, waiting while it fails,
    logged to `results/phase1c/memory_gate.log`;
  - stop above 3 GB RSS.

## 13. Reproduce

```
python -m src.phase1c scan; python -m src.phase1c pilot_check
python -m src.phase1c freeze W1|G|T|Tp      # resumable; before the registration commit
python -m src.phase1c manifest             # the registration commit
python -m src.phase1c run W1|G|T|Tp        # after the push and the OpenTimestamps proof
python -m src.phase1c finalize             # commit runs_*.jsonl + forecasts.sha256
python -m src.phase1c observe W1|G|T|Tp; python -m src.phase1c score
```
