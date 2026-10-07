# Phase 2A-PS2 registration: the per-seed fold test, with the pipeline repaired

**NOT YET REGISTERED.** This file is prepared for the registration commit. It stops before that commit for the
details in §9 that touch a criterion, the gate, validity or the scoring set: they need the author's approval.

Status:
- **Registered after 2A-PS's UNRESOLVED (validity) outcome.** 2A-PS's registered verdict STAYS UNRESOLVED
  (validity). This is a new registration, not a re-scoring of 2A-PS.
- **Not yet done:** no registered seed has been trained.
- **Before any registered training:** the registration commit is pushed and its OpenTimestamps proof is obtained.
- **The registration commit** will hold this file, the approved page, `src/phase2a_ps2.py` and its import closure
  (which includes `src/phase2a_ps.py`, unchanged), `tests/test_phase2a_ps2.py`, and the frozen inputs. Their SHA-256
  hashes go in `results/phase2a_ps2/registration.sha256` (§11).

## 1. Design reference

- **Approved page:** `results/designs/phase2a_ps2_design.md` (commit 820ef52; approved by the author 2026-10-07, "as
  written, including the worker's edits"). Everything on it is binding here.
- **‡ = set after seeing 2A-PS data. Every change below is ‡.** Each is justified from 2A-PS's POST HOC diagnosis
  (2f0fb33).
- **2A-PS's machinery is reused unchanged** (`src/phase2a_ps.py`, registration c819290; hashed in 2A-PS's
  manifest, imported and never modified; a test asserts its hash). This covers:
  - the per-seed sample, the branch identity, s_F, s\* and Λ_F (0.05 window; the agreement is descriptive);
  - the exact-M release at s₀, the scale-only rule with η = 1, and the rates 2⁻¹⁶ (scored) and 2⁻¹⁴;
  - the f = 0.95 leading-order causal forecast (`causal_forecast_fold.run_forecast`, guarded, NaN recomputation);
  - the criteria F, H, E_seed, C3, C4 and P (C1 and C2 secondary), validity, the outcome rule, and the class rule.
- **What the new module adds** (`src/phase2a_ps2.py`): the activity test, the per-seed budgets and caps, the new
  statuses, the descriptive wider search, its own seeds, freeze, plan and files.
- **Exploration** (seeds 2,930,112–2,930,199 only; never registered): `results/designs/phase2a_ps2_explore/`.

## 2. The three changes (‡)

### (a) The activity test (Phase 3's)

- **Rule.** Unit k is active iff |v_k| ≥ 10⁻⁸·s, with s = ‖v‖₁ (inclusive).
- **Why.** 2A-PS counted v₂ = 1.2e−93 as active (2,975,052). So â leaked, a sign flipped at step 1, and there was no
  forecast (also 2,975,127).
- **Handling.** A release where 2A-PS's rule (v ≠ 0.0) counts a unit active but this test does not gets the status
  **"inactive unit at release"**. It comes right after "release Newton failed". Such a seed is counted, never trained
  or scored.
- **Training is unchanged.** On every seed that passes the test, the active set is 2A-PS's (asserted). So the step,
  â, the idle unit and the signs are exactly 2A-PS's.
- **On 2A-PS's committed releases** (tested): the test drops a unit on exactly five releases, 2,975,052, 127, 226,
  260 and 345. Their shares are 4e−94 to 3e−88. All five were scoreable in 2A-PS. Every other release has a smallest
  share > 0.04.

### (b) Per-seed budget (Phase 3's)

- **t\*** = (1/ρ)∫ds/(3·(−dL_M/ds)), from s₀ to 0.9999·s_F, on the seed's frozen M. If dL_M/ds ≥ 0, the status is
  "stall" (counted).
- **B** = min(⌈k·t\*⌉ + 3000, ⌈192/ρ⌉). **k = 1.5 for clean seeds and 2.5 for none and mixed seeds** (their
  observation must reach 1.25·s_F).
- **Budgets.** B is the run budget, the observation budget and, minus t_c, the forecast budget.
- **Over cap.** If ⌈k·t\*⌉ + 3000 > ⌈192/ρ⌉, the status is "over cap": counted, not trained.
- **Hard per-run caps:**
  - ⌈192/ρ⌉ steps: 3,145,728 at 2⁻¹⁴ and 12,582,912 at 2⁻¹⁶;
  - 1 GB current RSS: the run is aborted, counted and not scored.
- **Author's pre-registration check.** STOP before registering if more than 5 scoreable clean seeds, or more than 5
  scoreable none seeds, are over cap (§12).

### (c) H on the REGISTERED 30-minimum class, unchanged

- **The class.** n = #(ρ₂ ≥ q) among 30 perturbed minima at 1.01·s_F (2A-PS's rule 5): clean n ≥ 27, none n = 0,
  mixed 1–26. The rule measures **whether slab basins lie near the fold.**
- **H re-tests the same rule that failed in 2A-PS** (20/30), now with the pipeline repairs (activity test, per-seed
  budgets).
- **The wider search is DESCRIPTIVE ONLY.** It uses 360 minima: s ∈ {1.01, 1.02, 1.05, 1.10}·s_F × perturbations
  ×1, ×2, ×4 × 30, from `default_rng([seed, 20261005])`, classed by the posthoc `wider_class`.
  - It is frozen per seed and reported as crossing rates by wider-search class.
  - It is never a criterion and never changes a verdict (tested).

## 3. Order of the steps (`src/phase2a_ps2.py`)

1. **Before the registration commit:** `scan`, `pilot` (pilot seeds only), `freeze` (all 600 registered seeds, no
   training; → `frozen_parts.jsonl`, `frozen.json`, with the author's over-cap check), `manifest`. Then `stamp`.
2. **`run`**, after the registration commit is pushed and OpenTimestamped. For each planned (seed, rate), with its
   frozen budget B:
   - training from the frozen release to t_c (the first s ≥ 0.95·s_F; NO ρ₂);
   - the t_c state hashed;
   - the causal forecast with the seed's own (s_F, Λ_F), and the fixed-dataset competitor;
   - forecast budget B − t_c.
3. **`finalize`:** `forecasts.sha256` over `runs.jsonl`. Both are **committed before observation.**
4. **`observe`:** ρ₂ at every step; the t_c state asserted equal bit for bit. It stops at the crossing and s_F, at
   1.25·s_F, or at B.
5. **`score`:** `scores.json`, with the counts at the top (untraceable first).

## 4. The per-seed freeze

- **2A-PS's raw freeze, unchanged** (`phase2a_ps.freeze_raw`). It records the sample hash, the M homotopy, s_F and its
  validation, the 30-minimum blocks, the λ_min grid and Λ_F, S and s\*, s₀, the release and the follow anchor.
- **Added per seed** (`ps2`):
  - the activity test at the release (every seed with a release);
  - for seeds whose 2A-PS status is ineligible or scoreable:
    - the seed's M rebuilt by the registered steps (s_F and the release hash asserted equal);
    - dL_M/ds on t\*'s grid, and ρ·t\*;
    - the wider search.
- **Status** = the FIRST failing reason, in this order:
  1. untraceable (M homotopy failed / no upper fold / homotopy point not stable / stable part does not reach the fold /
     ρ₂ ≥ q on the stable part);
  2. s_F not validated;
  3. Λ_F not validated;
  4. no s\*;
  5. release Newton failed;
  6. **inactive unit at release** (a);
  7. ineligible (s₀ > 0.6·s_F);
  8. **t\* not computable** (Newton failed on t\*'s grid; §9 D3);
  9. **stall** (dL_M/ds ≥ 0 on t\*'s grid);
  10. **over cap** (at any of the class's rates; §9 D4);
  11. otherwise **scoreable**.

  Only scoreable seeds are trained or scored. Every other seed is counted and reported prominently, with the
  untraceable count first.
- **Every seed's status and class are frozen.** Their listing ("seed status class" in seed order) and its SHA-256 go
  in `frozen.json`. `summary` asserts every stored evaluation equals a fresh one. `reevaluate` re-applies the rules to
  the stored raw records without refreezing.

## 5. Rates, plan, budget and forecast

- **Rates.** Clean seeds at 2⁻¹⁶ (scored) and 2⁻¹⁴ (per-seed exponent); none and mixed seeds at 2⁻¹⁴.
- **Plan** (frozen in `frozen.json`): the first 40 clean, 30 none and 20 mixed SCOREABLE seeds in seed order, each run
  with its frozen B. The rest are counted, not trained.
- **Forecast.** Unchanged from 2A-PS (§5 of its registration): the seed's own (s_F, Λ_F), η = 1, leading order
  r_fc = Ω₀ε̂_F^{2/3}. It reads s rows 0 … t_c − 1 through a `GuardedArray`, and the NaN recomputation must be
  identical (else `CausalityViolation`). The competitor uses the fixed-dataset fold (4.767689442106793,
  1.509953872657247e−3). The only change is the forecast budget: B − t_c.
- **2A's fitted correction** is shown descriptively only, labelled POST HOC (as in 2A-PS).

## 6. Criteria (unchanged from 2A-PS; `phase2a_ps.score_tables`)

Clean seeds at 2⁻¹⁶ unless stated. 10,000 bootstrap resamples over seeds, with 2.5–97.5 percentiles. **PASS needs
F, H, E_seed, C3, C4 and P. C1 and C2 are secondary.**

| | statistic | PASS if |
|---|---|---|
| **F** | s_F ≤ s_obs ≤ 1.25·s_F | at ≥ 90% of the trained clean seeds, AND no falsifier (a crossing below s_F and at or below 1.25·s\*) |
| **H** | none seeds (2⁻¹⁴): reached s ≥ 1.25·s_F with no crossing | at ≥ 80% of the trained none seeds (≥ 10 needed, else UNRESOLVED) |
| **E_seed** | per clean seed Δln r_obs/Δln ε̂_F between 2⁻¹⁴ and 2⁻¹⁶; their median | 95% bootstrap CI of the median inside [0.55, 0.80] |
| **C3** | r_obs/r_fc | > 1 at ≥ 90% of the trained clean seeds |
| **C4** | D = \|t_fc − t_obs\| − \|t̂_F − t_obs\| | upper end of the 95% CI of mean D < 0; any trained clean seed without a forecast: FAIL |
| **P** | D = \|ln ŝ_c/s_obs\| − \|ln ŝ_c^fixed/s_obs\| | upper end of the 95% CI of mean D < 0 |
| C1 | \|t_fc − t_obs\| ≤ 0.15·delay_fc | at ≥ 80% (secondary; registered expected FAIL) |
| C2 | \|delay_fc − delay_obs\| ≤ 0.25·delay_fc | at ≥ 80% (secondary; registered expected FAIL) |

- **Outcome.** 2A's rule over F, H, E_seed, C3, C4 and P:
  - "UNRESOLVED (gate)" if the gate fails, "UNRESOLVED (validity)" if validity fails;
  - otherwise PASS iff all six pass, UNRESOLVED if any is UNRESOLVED, else FAIL, naming each.
- **Every denominator, threshold and the falsifier's scope** are 2A-PS's D8–D16.
- **Aborted runs** (1 GB RSS) are excluded and counted (§9 D6).
- **Mixed seeds** enter no criterion.
- **DESCRIPTIVE:**
  - crossing rates by wider-search class;
  - every criterion split by the |m′c′| agreement check (as in 2A-PS);
  - 2A's POST HOC correction.

## 7. Gate and validity

- **Gate:** ≥ 24 clean scoreable seeds of the 600 (P ≈ 1). If it fails, nothing is trained; the outcome is
  "UNRESOLVED (gate)".
- **H:** fewer than 10 trained none seeds make H UNRESOLVED (about 100 expected).
- **Validity** (unchanged from 2A-PS §7; any failure gives "UNRESOLVED (validity)"):
  - the follow check at 0.8·s_F lands on the seed's own M at ≥ 90% of the trained clean seeds (2⁻¹⁶);
  - t_c < t_obs at ≥ 90% (a cutoff and no crossing passes);
  - at EVERY clean run (2⁻¹⁶ and 2⁻¹⁴), the idle unit stays exactly 0 and the active signs stay fixed up to the
    crossing.

## 8. Disclosures

- **Every change is ‡.** Each was set after 2A-PS's results, from its POST HOC diagnosis (2f0fb33): the activity test,
  the per-seed budget and its k, and the decision to keep H on the registered class.
- **k was set from 2A-PS's COMMITTED OBSERVATIONS.**
  - 6 of 27 none runs ended after 1.5·t_F (max 1.94).
  - Clean runs ended by 1.15·t_F.
  - k = 1.5 (clean) and 2.5 (none, mixed) cover these with margin.
- **H re-tests the same rule that failed in 2A-PS.** In 2A-PS, 7/30 none seeds crossed and 3 more ended at the budget
  below 1.25·s_F (23/30 = 0.77, near the 0.8 threshold).
  - **What the rule measures:** whether slab basins lie near the fold.
  - **Exploration** (2⁻¹², not a registered rate): none 1/13 crossed, mixed 7/7, clean 14/14.
- **H's power at n = 30:** 0.93 if 87.5% of none seeds truly do not cross; 0.61 at 80%; 0.16 at 70%.
- **The wider search is descriptive.**
  - It did not separate 2A-PS's crossers: it reclassified 6/7 crossers vs 15/23 non-crossers to mixed.
  - In exploration it left 0/36 seeds none, and wide-mixed seeds crossed 8/20.
  - A none class defined by S's ρ₂ < q was empty (0/36), so it was not adopted.
- **C3 is a KNOWN RISK.** In 2A-PS the ratio was > 1 on only 32/36 forecasting seeds (0.89 < 0.90). This
  registration does not change C3 or the forecaster. If C3 fails, the leading-order delay is not an under-estimate
  seed by seed.
- **Over-cap and stall seeds are slow, large-s_F seeds.** Counting them and not training them removes the slowest
  seeds from F and H. In exploration, 1/36 was over cap (a mixed seed, ρ·t\* 90.6). The counts are in §12.
- **Untraceable fraction (headline).** 35.5% in 2A-PS and 38.6% in exploration. It is kept as a headline count (§12).
- **A MECHANISM test.** Each run is released at its seed's exact M point. This is not training from initialisation.
- **Set while preparing this file (not on the page): §9.**

## 9. Details the approved page left open

### Touching a criterion, the gate, validity or the scoring set: FOR THE AUTHOR'S APPROVAL

Each is implemented and tested as proposed. The freeze stores the raw inputs, so another choice re-evaluates without
refreezing (`reevaluate`).

| # | detail | proposed resolution (implemented, tested) | touches |
|---|---|---|---|
| **D1** | **Where the new statuses sit** (the page: "inactive unit at release" after "release Newton failed"; "stall" and "over cap" left open) | … → release Newton failed → **inactive unit** → ineligible → **t\* not computable** → **stall** → **over cap** → scoreable. Inactive comes before ineligible, as the page says, so an inactive ineligible seed counts as inactive. The budget statuses come last: they are evaluated only for seeds that pass every 2A-PS check. | scoring set, gate |
| **D2** | **t\*'s grid** | Exactly the exploration's (on which the calibration t_F/t\* = 1.002–1.011 rests): **121 points, linspace(s₀, 0.9999·s_F)**. dL_M/ds by central differences with half-width 10⁻⁶·s_F (the upper point capped at 0.99995·s_F), Newton on the seed's M at each point. Trapezoid (`quasi_static_rho_time`). n = 3, the page's 3. **Stall** = dL_M/ds ≥ 0 at any grid point. A sign change strictly between two points, which is less than 1% of [s₀, s_F] wide, would be missed. | scoring set, every budget |
| **D3** | **Newton failing on t\*'s grid** (not on the page) | A new status, **"t\* not computable"**: counted, not trained. It comes before stall. Count: §12. | scoring set |
| **D4** | **A clean seed has two rates** (the cap differs: 3000 steps is 0.18 of ρ·t at 2⁻¹⁴ and 0.05 at 2⁻¹⁶) | A clean seed is over cap if it is over cap at EITHER rate, so it is trained at both rates or at neither. In practice 2⁻¹⁴ binds: 1.5·ρt\* > 191.82. | scoring set, gate, E_seed |
| **D5** | **What "counted, not trained" means for the gate, the caps and the author's check** | Inactive, t\* not computable, stall and over-cap seeds are NOT scoreable. They are excluded from the gate count (≥ 24 clean scoreable), from the caps (the plan takes the first 40/30/20 scoreable) and from every criterion, and counted prominently. "Scoreable clean (none) seeds over cap" in the author's check = seeds whose status is "over cap", that is, which passed every earlier check. | gate, scoring set |
| **D6** | **Aborted runs** (1 GB current RSS, checked every 20,000 steps, as in Phase 3) | Counted, not scored, as in Phase 3: the run is excluded from every criterion and from validity, and not observed. An aborted 2⁻¹⁴ clean run leaves the seed without an exponent (E_seed excludes it, counted). An aborted 2⁻¹⁶ clean run removes the seed from F, C3, C4, P and validity. Expected 0: 2A-PS's peak was 0.73 GB. | every criterion, validity |
| **D7** | **H's run end with per-seed budgets** | Unchanged from 2A-PS D9: a none seed counts only if observation reached 1.25·s_F with no crossing. A run that ends at B below 1.25·s_F without crossing does NOT count (conservative). | H |
| **D8** | **The bootstrap generator** | Unchanged: `default_rng(2,975,000)`, because the criteria code is 2A-PS's, imported unchanged. The alternative is 2,987,000, following the convention "the first registered seed". | E_seed, C4, P |
| **D9** | **The activity test and the class rule** | Independent. The class is computed from M(0.9999·s_F), as in 2A-PS, and the activity test reads only the release. The class is recorded for every traced seed and counted only when scoreable. The test is applied at the release only. During training, the idle-unit and sign validity rules are 2A-PS's, unchanged. | scoring set |

### Implementation only (not touching a criterion, the gate, validity or the scoring set)

- **Where the wider search runs.** On every seed whose 2A-PS status is ineligible or scoreable (a superset), and it is
  reported on the scoreable seeds. The stream is `default_rng([seed, 20261005])`, the posthoc's and the
  exploration's. The exploration seed 2,930,143's class, ρ·t\* and wider search are reproduced exactly (tested).
- **M for t\*.** Rebuilt by 2A-PS's registered homotopy and continuation (posthoc `seed_M`), with s_F and the release
  hash asserted equal to the frozen ones.
- **The RSS check.** The current RSS comes from `ps` every 20,000 steps. The process also stops above 3 GB peak RSS
  (2A-PS's machine rule).
- **Files.** `frozen_parts.jsonl` (one row per seed: 2A-PS's raw record + `ps2` + `evaluation`) and `frozen.json`
  (counts, gate, over-cap check, plan with budgets, class listing, budget distributions, activity drops, stalls,
  wider-search table).
- **Reproduction with the repairs switched off** (tested):
  - 2A-PS's committed evaluations of all 400 seeds;
  - its freeze of 2,975,052;
  - its runs and observations of 2,975,052 (both rates) and 2,975,015 (2⁻¹⁴), bit for bit.
- **Machine rules.** One process, nice 15, one thread. A memory gate (free ≥ 25%, swap free ≥ 500 MB) runs as a
  separate logged step before each job and between seeds, plus a disk check (≥ 20 GB). Long jobs run fully detached
  and are resumable.

## 10. Competing outcomes

| outcome | reading |
|---|---|
| PASS | 2A-PS's failures were the pipeline (release, budget). |
| **C3 FAIL (known risk)** | The leading-order delay is not an under-estimate seed by seed (2A-PS: 32/36). |
| H FAIL | With the pipeline repaired, slab basins near the fold do not predict crossing. |
| H UNRESOLVED | Fewer than 10 none seeds (unlikely). |
| F or C4 FAIL | The no-forecast seeds were not only a budget artefact. |
| UNRESOLVED (validity) | A sign or release problem survives the activity test. |
| UNRESOLVED (gate) | Fewer than 24 clean scoreable seeds. |

## 11. Frozen files and hashes

`results/phase2a_ps2/registration.sha256` (`phase2a_ps2.manifest`) will list:
- this file and the approved page;
- the import closure of `src/phase2a_ps2.py` (`phase2a_ps`, `phase2a_ps_posthoc`, `phase2a`, `causal_forecast_fold`,
  `causal_forecast`, `sb_fold`, …);
- the tests `tests/test_phase2a_ps2.py`, `tests/test_phase2a_ps.py` and `tests/test_causal_forecast.py`;
- `results/phase2a_ps2/{seed_scan.json, pilot.json, pilot_parts.jsonl, frozen_parts.jsonl, frozen.json}`;
- 2A-PS's registration, manifest, frozen parts, runs, observations and scores, and the posthoc JSON;
- v2's and 2A's `frozen.json`, `phase2a_posthoc.json`, and Track 1A's M and S branch files;
- the exploration files (`p2_explore.py`/`.jsonl`, `p2_summary.py`/`.log`, `p2_seedscan.py`/`.json`, README) and
  Phase 3's `p3_activity.py`.

`run`, `observe` and `score` assert every hash, and that the manifest and `registration_stamp.txt` are committed and
unmodified.

## 12. Scan, pilot and freeze (no registered seed trained)

(Filled in below as the steps complete.)

## 13. Reproduce

```
python -m src.phase2a_ps2 gate scan;   python -m src.phase2a_ps2 scan
python -m src.phase2a_ps2 gate pilot;  python -m src.phase2a_ps2 pilot
python -m src.phase2a_ps2 gate freeze; python -m src.phase2a_ps2 freeze   # -> frozen.json, the author's over-cap check
python -m src.phase2a_ps2 manifest     # the registration commit; then: stamp, commit, push, OpenTimestamps
python -m src.phase2a_ps2 run; python -m src.phase2a_ps2 finalize        # commit runs.jsonl, forecasts.sha256
python -m src.phase2a_ps2 observe; python -m src.phase2a_ps2 score
```
