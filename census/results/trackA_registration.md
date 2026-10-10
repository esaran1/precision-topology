# Track A registration: the causal Adam per-run forecast at a = 1.77

**REGISTERED.** This file, the approved design page, the code, the tests, the frozen rules, the landscape, the seed scan,
the frozen per-seed files and the pilot are committed in one commit before any registered run. Their SHA-256 hashes are
in `results/trackA/registration.sha256` (§10).
- **Author's approval (2026-10-09):**
  - The pilot tolerances τ_cross 10, τ_lag 10 and b 0.10 (§7), with the thin b margin recorded as a known risk (§7.1).
  - The "1.77 unused" rule (`A_HITS_REVIEWED`, §2).
  - D1–D8 (§9). D8 is approved on condition that it measures the observed lag exactly as 1C's width-1 arm did (§9.1).
- **Not yet done:** no registered seed has been trained or frozen for training, and no gap, placement or crossing of any
  run of a registered seed has been evaluated.
- **Before any registered training:** the registration commit is pushed and its OpenTimestamps proof
  (`results/trackA/registration_stamp.txt.ots`) exists. `run` refuses to start without it (tested).

## 1. Design reference

- **Approved page:** `results/designs/trackA_causal_adam_design.md` (draft 33de250; the author's decisions of
  2026-10-09). Its hash is in the manifest (§10). Everything on it is binding here. The decisions:
  1. The extrapolation is the Adam-specific **`slin_20`** (§4). It was **chosen after exploration**, out of 7
     families, on the same 120 exploration seeds (9,771,000–119 at a = 1.85; `results/designs/trackA_explore/`).
  2. The preconditioner rule is the **coupled `coupled_sw`** (§4), also chosen after that exploration.
  3. a = 1.77; f = 0.90 (0.95 descriptive); 100 registered seeds 9,776,000–099; a 40-seed pilot 9,774,000–039.
  4. R4-unstable runs count as misses.
  5. **Author's addition:** the two rules are FROZEN AND HASHED before registration (§4.3), and their forecast error is
     estimated on PILOT SEEDS ONLY (§7).
- **Code:** `src/causal_adam.py` (the forecaster), `src/trackA_causal.py` (the driver), `tests/test_trackA_causal.py`.
  `src/causal_forecast.py` (25aac9e) and every registered module are used unchanged.

## 2. Setting

| item | value |
|---|---|
| network, data | width 1, sin family at **a = 1.77**; `fold1d.make_data(200, seed)` |
| optimiser | free Adam, lr 0.01 (torch defaults β₁ 0.9, β₂ 0.999, ε 1e-8) |
| training | `phase1a_pilot.w1_train(seed, "adam", 32000, a=1.77)`: init U(−1, 1)⁴ in float32 from a local torch Generator, cast to float64; no global RNG state |
| budget, detection | 32,000 steps; every-step detection `phase1a_pilot.w1_observe` (phase2b_ordering.state, the first placed step t ≥ 1) |
| landscape | Track A's construction (`src/track_a.py`, byte-identical to its registered version, through `track2a.load_track_a` with a, the seeds and OUT rebound): continuation in a from the certified 1.60 switch point, principal copy, switch s\*_pop; validated by `own_threshold.global_min` (unplaced at 0.995·s\*_pop, placed at 1.005·s\*_pop, same branch up to the mirror and 2π). A failure stops before the pilot. Result: §8 |
| s\*_frozen | per seed, before training: Newton from the landscape switch point onto the own sample, continued (grid [0.3, 1.7]·s\*_pop, spacing 0.002·s\*_pop); its switch (Track A's R0). `results/trackA/frozen_registered.jsonl` |

**a = 1.77 is unused.** `results/trackA/seed_scan.json` (`trackA_causal.scan`): no occurrence of `1.77`, `1.770…` or
`1.769999…` in any text file under src/, tests/, results/, paper/, notes/, independent/ or data/ (Track A's own files
excepted) is an activation value.
- **Disclosed: the page's claim needed a reviewed list.** The page says "1.77" occurs nowhere. The token scan in fact
  matches 11 files, and every hit was reviewed (`A_HITS_REVIEWED`; contexts in `seed_scan.json`). None is an activation
  value. The hits are Track A's own ledger comment, a step size 1.77e−3, an interval bound, a ratio quantile, gap values
  in an exploration log, a scale grid point of the simplicity-bias landscape, timings ("secs": 1.77), and an R-column
  ratio 1.769999… in the gitignored `phase2b_checkpoints.csv`, whose a column holds only 1.3–1.6.
- **The rule:** a is unused iff every file with the token is on the reviewed list, with the same count of matching lines.
- The first pattern (`1.769999` without a boundary) also matched `131.769999…` in that CSV. It was given a boundary
  before the recorded scan.
- The page's scan also lists the a values with training or crossing data (1.5–2.2), and notes that 1.75 and 1.775
  appear only in landscape tables.

## 3. Seeds

- **Registered:** 9,776,000–9,776,099 (100). **Pilot:** 9,774,000–9,774,039 (40). Disjoint.
- **Fresh:** the scan finds no 7-digit number with prefix 9774 or 9776 (also written 9_774_xxx or 9,774,xxx) in any text
  file under the same directories, and no overlap with the SEEDS\* / PILOT_SEEDS\* constants of the registered modules.

## 4. The forecast (frozen)

### 4.1 Cutoff and rule point (1C's W1, unchanged)

- **Cutoff t_c** (harness side, a nested stopping time; `causal_forecast.w1_cutoff_on_s_run`): the first step t with
  s_t ≥ f·s\*_run(t), where s\*_run(t) is the switch of the branch occupied at the causal rule point t_R(t). f = 0.90.
- **Rule point t_R:** the first step ≥ 1 with s_t ≥ 0.5·s\*_frozen at every t_R ≤ t < t_c. The forecaster recomputes t_R
  and s\*_run from rows < t_c; branch rule = Track A's (Newton at s_{t_R} from the rule-point state, continued on the own
  sample).

### 4.2 The two rules

1. **Extrapolation `slin_20`.** s is linear in t, by least squares over the last n = max(3, min(20, t_c − t_R)) visible
   steps (t_c − n … t_c − 1). The path is anchored at s_{t_c−1} and runs until ŝ ≥ 1.6·s\*_frozen (inclusive) or the
   step budget. A fitted slope ≤ 0 means no forecast.
2. **Preconditioner `coupled_sw`.** R4 runs from t_R with δ₀ = θ_{t_R} − θ\*(s_{t_R}), m₀ = m_{t_R} and the visible
   P_t = 1/(√v̂_t + ε) for steps < t_c.
   - After t_c it carries the raw second moment, v ← β₂v + (1 − β₂)(Hδ)², starting from v̂_{t_c−1}(1 − β₂^{t_c−1});
     v̂_t = v_t/(1 − β₂^t) and P_t = 1/(√v̂_t + ε).
   - v̂(t_sw,fc), at the forecast switch t_sw,fc (the first extrapolated step with ŝ ≥ s\*_run), is taken from that carry.
     If t_sw,fc < t_c it is the visible row. If the coupled recursion stops before t_sw,fc (a crossing, divergence or
     the branch-grid end), it is the decay-only carry v̂_{t_c−1}(1 − β₂^{t_c−1})·β₂^{t_sw,fc − t_c + 1}/(1 − β₂^{t_sw,fc})
     (the exploration's code, unchanged; open detail D2).
   - P̂ = 1/(√v̂(t_sw,fc) + ε) is frozen, and R4 (`causal_forecast.lag_forecast_1d`, unchanged; Adam's bias correction
     1 − β₁^t) is rerun with it. Its first step with G(θ\*(ŝ_t) + δ_t) > 0 is the forecast crossing t_fc.
3. **Output:** t_fc, s_fc, r_fc = s_fc/s\*_run − 1, t_sw,fc and lag_fc = t_fc − t_sw,fc. **No forecast (a miss):** no
   rule point, no branch minimum or switch, a slope ≤ 0, no forecast switch, no predicted crossing, or R4 unstable
   (spectral radius > 1 or sup|δ| > 1, the Track 1 rule).

### 4.3 Frozen and hashed

`results/trackA/frozen_rules.json` (`trackA_causal.freeze_rules`, committed before the pilot) holds the SHA-256 of the
source of `slin_extrapolate`, `coupled_vhat` and `coupled_sw_vhat`, and of all three together with their constants
(K = 20, at least 3 points, horizon 1.6, β₁, β₂, ε). `all_with_constants` =
`bc03480a0013d6c7011b7c20662d4a165b47c2a6765f9bbbd9bb4b05db189683`. `test_frozen_rule_hash_recomputes` recomputes it;
the pilot, `run` and `observe` assert it.

### 4.4 Guards

- The forecaster receives only GuardedArray views with cutoff t_c of w₂, M and v̂ (4 columns), and of the hidden state
  (w₁, b₁, b₂) with at most one readable row (t_R < t_c). Reading v̂, M or w₂ at row t_c raises CausalityViolation, and so
  does a second hidden row (tested). Every guard's last row read is asserted < t_c.
- **NaN recomputation, per run:** the same forecaster, unguarded, on copies with every row ≥ t_c NaN. The outputs must
  be identical. An exception during the recomputation counts as not identical (D6). A guard bypass fails it (tested).

## 5. Order

`run` (after the push and the OpenTimestamps proof): for each registered seed, training to the full budget **without
any gap or placement evaluation**, then the forecasts. The runs file holds no observed quantity, and the training hash
(W, M, v̂) is stored. `finalize`: forecasts.sha256, committed before `observe`. `observe`: retrains (hash asserted),
every-step detection, the actual switch t_sw (the first step with s ≥ the cutoff's s\*_run), and P at t_sw. `score`.

## 6. Scoring

- **Base:** runs with s\*_frozen and a crossing within the budget.
- **Scored:** base runs with t_c < t_obs, t_sw defined and finite r_obs = s_obs/s\*_run − 1.
- **Miss:** a scored run without a forecast (t_fc or t_sw,fc undefined; R4-unstable included). It counts in C1 and C2's
  denominator as not within. It is excluded from C3 and C4.

| criterion | rule | PASS iff |
|---|---|---|
| **C1** | \|t_fc − t_obs\| ≤ τ_cross | in ≥ 80% of the scored runs |
| **C2** | \|lag_fc − lag_obs\| ≤ τ_lag, lag_obs = t_obs − t_sw | in ≥ 80% of the scored runs |
| **C3** | median r_obs/r_fc (runs with a forecast) | in [1 − b, 1 + b] (closed) |
| **C4** | D = \|t_fc − t_obs\| − \|t_sw,fc − t_obs\|, paired per run (runs with a forecast) | the upper end of the 95% percentile bootstrap interval of mean D is < 0 (10,000 resamples, numpy default_rng(9,776,000)) |

**Outcome:** PASS iff C1–C4 all PASS; otherwise FAIL, naming each failed criterion (`phase1c.outcome`). **Falsifier:**
C4 fails.

**Validity** (otherwise UNRESOLVED, every criterion):
- ≥ 60 base runs and ≥ 60 scored runs;
- t_c < t_obs in ≥ 90% of the base runs (a run with no cutoff counts as not before);
- the actual switch t_sw is defined in every base run with t_c < t_obs (D1);
- the NaN recomputation is identical in every run with a cutoff.

**Descriptive (never a verdict):** P̂'s error against P(t_sw), per coordinate; 1C's extrapolation (quad in log s,
5% / ≥ 50 window, with coupled_sw); P at t_c − 1 (slin_20 with v̂_{t_c−1}); the primary rules at f = 0.95; the horizon
t_obs − t_c.

## 7. Tolerances (pilot rule; pilot seeds only)

**Registered tolerances (approved by the author 2026-10-09): τ_cross = 10, τ_lag = 10, b = 0.10.** They are
`trackA_causal.TOLERANCES`, asserted equal to the pilot rule's output in the hashed `pilot.json` (tested).

- **Rule:** τ_cross = 1.5 × the pilot q90 of |t_fc − t_obs|, and τ_lag = 1.5 × the pilot q90 of |lag_fc − lag_obs|,
  each rounded up to a multiple of 5. b = |median r_obs/r_fc − 1| + 2SE, rounded up to 0.05, with
  SE = 1.2533·(IQR/1.349)/√n (Phase 1A's pilot rule; D3).
- **Pilot set:** the pilot runs with s\*_frozen, a crossing, t_c < t_obs and t_sw defined. The errors are taken over those
  with a forecast. Percentiles use numpy's linear interpolation.
- **Source:** `results/trackA/pilot.json` (`trackA_causal.pilot_summary`), from `pilot_runs.jsonl`. `score` uses
  `TOLERANCES` after asserting that they equal that hashed file's values.

**Pilot result (40/40 pilot seeds; approved by the author 2026-10-09):**

| τ_cross | τ_lag | b |
|---|---|---|
| **10** (1.5 × q90 6 = 9 → 10) | **10** (1.5 × q90 4 = 6 → 10) | **0.10** (\|0.9515 − 1\| + 2 × 0.0206 = 0.090 → 0.10) |

- **Pilot set:** 40 of 40 pilot seeds have s\*_frozen and cross. All 40 have t_c < t_obs and t_sw defined, and all 40
  have a forecast (0 misses). The NaN recomputation is identical in all 40.
- **Pilot errors:** \|t_fc − t_obs\| has median 3.5, q90 6 and max 13. The lag error has median 2.5, q90 4 and max 12.
  r_obs/r_fc has median 0.9515, q10 0.774 and q90 1.018 (IQR 0.865–1.005).
- **Robustness:** the alternatives give the same three numbers. These are the exploration's SE (1.2533·sd/√n = 0.0244,
  giving b = 0.10) and Phase 1A's floors (τ_lag 10, b 0.10).
- **In sample (descriptive, not a test):** at these tolerances the pilot has C1 39/40, C2 39/40 and C3 median 0.9515.
  C4's interval is [−63.7, −47.3].
- **P̂ error against P(t_sw):** median 0.13%, 0.13% and 0.22% (w₁, b₁, b₂); q90 0.33%, 0.40% and 0.62%. coupled_sw
  took v̂(t_sw,fc) from the coupled carry in 40 of 40 runs (D2's fallback was never used).
- **Descriptive variants on the pilot:**
  - P at t_c − 1 (slin_20): 0 misses, q90 8 / 5, median 0.948. P error is about 6.7%.
  - 1C's extrapolation: 11 misses (6 no forecast switch, 5 not growing), C1 0.725, median 1.169.
  - f = 0.95: 0 misses, q90 4 / 3, median 0.966.
- **Horizon:** t_obs − t_c has q10 34, median 115 and q90 167 steps.

### 7.1 Known risk: the thin b margin (disclosed before any registered run)

- The pilot median r_obs/r_fc, 0.9515, lies inside [0.90, 1.10] by only about 0.05.
- The exploration's median at a = 1.85 was 0.94 (114 runs with a forecast), inside the band by 0.04. Its split-rule b
  was 0.20.
- A registered median below 0.90 fails C3, and with it the outcome, even if C1, C2 and C4 pass.
- The author approved b = 0.10 with this risk recorded.

## 8. Landscape at a = 1.77

- `results/trackA/landscape.json` (`trackA_causal.landscape`, Track A's function unchanged): continuation from the 1.60
  switch point gives winding shift 0. Principal copy: b₁ = −2.14356.
- **s\*_pop = 1.465380.** θ\* = (−1.19343, −2.14356, 4.02233). Gradient residual 1.5e−13; H is positive definite;
  κ_SGD = 2.562.
- **VALIDATED** by `own_threshold.global_min`. At 0.995·s\*_pop the global minimum has G = −0.00284 (unplaced); at
  1.005·s\*_pop it has G = +0.00281 (placed). It lies on the same branch, up to the mirror and 2π, at both levels.
- **Frozen s\*_frozen:** registered 100/100 (range 1.054–2.014), `frozen_registered.jsonl`; pilot 40/40 (1.228–1.704),
  `frozen_pilot.jsonl`.

## 9. Open details (implementation choices not fixed by the page; D1–D8 APPROVED by the author 2026-10-09)

| # | detail | implemented as | touches |
|---|---|---|---|
| D1 | "the actual switch is defined" | A validity condition: t_sw is defined in every base run with t_c < t_obs. t_sw is also required for a scored run, as in 1C's D1. | validity, scored set |
| D2 | coupled_sw when the coupled recursion stops before t_sw,fc | Decay-only carry from v̂_{t_c−1} (the exploration's code). The visible row if t_sw,fc < t_c. | forecast |
| D3 | SE of the median in b | 1.2533·(IQR/1.349)/√n (Phase 1A). No floors: Phase 1A's τ_lag ≥ max(1.5·q90, 1) and b ≥ 0.10 are not used, because the page states none. | tolerances |
| D4 | the pilot error set | The scored-run definition on the pilot runs. Errors are over the runs with a forecast. | tolerances |
| D5 | C4 bootstrap seed | 9,776,000, the first registered seed (1C used its first W1 seed). | C4 |
| D6 | NaN recomputation raising | Counts as not identical. An exception in the guarded forecast itself stops the job; it is not scored as a miss. | validity |
| D7 | window when t_c − t_R < 3 | 3 points, reaching before t_R (the exploration's max(3, …)). | forecast |
| D8 | r_obs | s_obs/s\*_run − 1, with s\*_run the cutoff's (1C's D10). | C3 |

### 9.1 D8: the observed lag is measured exactly as 1C's width-1 arm did (the author's condition)

| quantity | Track A (`src/trackA_causal.py`) | 1C W1 (`src/phase1c.py`) |
|---|---|---|
| cutoff and s\*_run | `causal_forecast.w1_cutoff_on_s_run` (`harness_cutoff`) | the same function (`harness_cutoff`) |
| t_sw | `LR._first_ge(s, s_run)`, s_run = the cutoff's `cutoff_s_run` (`observe_one`) | the same, lines 873–880 |
| s_sw | s_run | s_run, lines 873–880 |
| r_obs | s_obs/s_sw − 1 (`score_track_a`) | the same, line 297 |
| lag_obs | t_obs − t_sw (`criteria`) | the same, line 228 |

`test_observed_lag_measured_as_1c_width1` checks this.

## 10. Manifest

`results/trackA/registration.sha256` (`trackA_causal.manifest`): this file, the design page, the code closure of
`src/trackA_causal.py`, the tests, the seed scan, the landscape, the frozen rules, the frozen per-seed files, the pilot
files and the exploration files.

## 11. Machine

One process, nice 15, one thread. A memory gate (free ≥ 25%, swap free ≥ 500 MB) runs as a separate logged step before
each job and between seeds. The job stops above 1 GB RSS. About 4 s per seed.
