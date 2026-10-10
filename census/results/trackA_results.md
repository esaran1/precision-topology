# Track A results: causal Adam per-run forecast at a = 1.77

**Registered outcome: PASS (C1, C2, C3 and C4 all PASS; validity met).**

- **Registration:** `results/trackA_registration.md`, commit 173a612. Stamp commit d7af2d1. OpenTimestamps proof commit
  f1809a7, made and pushed before any registered run.
- **Forecasts:** `results/trackA/runs.jsonl` and `forecasts.sha256` (148839e7…d5b5), committed and pushed (1b89a73)
  before observation.
- **Observations:** `results/trackA/observed.jsonl`, committed and pushed (3f8a2fe) before any verdict.
- **Scores:** `results/trackA/scores.json`, produced by `python -m src.trackA_score score` (the scoring fix, §1).
- **Scope:** 100 registered seeds (9,776,000–9,776,099); f = 0.90; frozen rules `slin_20` and `coupled_sw`
  (hash bc03480a…9683); tolerances τ_cross = 10, τ_lag = 10, b = 0.10 (pilot rule, approved 2026-10-09).

Every number below is copied from `results/trackA/scores.json` unless another producer is named. Numbers are rounded
for display only; the verdicts were computed on unrounded values.

## 1. The scoring bug and its fix (POST-REGISTRATION; approved by the author 2026-10-10, before any verdict)

The registered `python -m src.trackA_causal score` crashed before computing anything
(`results/trackA/score.log`, committed in 3f8a2fe):

```
  File ".../src/trackA_causal.py", line 806, in score
    res = score_track_a(table(rows, obs, F, "primary"), tol)
  File ".../src/trackA_causal.py", line 456, in score_track_a
    R = {k: (P1._arr(v, bool) if k in bool_keys else P1._arr(v)) for k, v in R.items()}
  File ".../src/phase1c.py", line 98, in _arr
ValueError: could not convert string to float: 'ok'
```

- **Cause.** `table()` returns two columns that `score_track_a`'s docstring does not list: `status` (a string) and
  `P_relerr` (used only for a descriptive statistic). `score_track_a` converts every column it is given to float. The
  registered tests called `score_track_a` only on constructed arrays without those columns, so the path from `table()`
  to the verdict was never exercised. The pilot used a different function (`pilot_errors`) and did not reach it.
- **Fix.** `src/trackA_score.py` (commit b500a50) calls the registered functions unchanged and passes `score_track_a`
  only the columns its docstring documents. `src/trackA_causal.py` and the registration manifest are unchanged.
  Tolerances, criteria, the miss rule, the bootstrap seed (9,776,000) and the descriptive statistics are the
  registered ones.
- **Tests.** `tests/test_trackA_score.py` (8 tests): the registered path crashes on constructed rows; the columns
  passed are exactly the documented ones; the full path `table()` → verdict gives PASS, FAIL (C1), FAIL (C4) and
  UNRESOLVED on constructed cases; misses count against C1 and C2; the result equals `score_track_a` on the same arrays.
- **Pilot check (author's condition).** Before any registered verdict, the entry point was run on the pilot data
  (`python -m src.trackA_score pilot_check`, `results/trackA/pilot_check.log`). It reproduces every in-sample pilot
  number in `pilot.json` exactly (Python equality): C1 39/40, C2 39/40, median ratio 0.9515434836200464, C4 interval
  [−63.725, −47.325]. (On the pilot, validity shows "not met" only because 40 < 60 runs; the check compares the
  criteria.)

## 2. Registered verdicts

### Validity (all met)

| condition | value | verdict |
|---|---|---|
| ≥ 60 crossing runs | 99 | met |
| ≥ 60 scored runs | 98 | met |
| cutoff before the crossing in ≥ 90% of crossing runs | 98 / 99 = 0.990 | met |
| actual switch defined in every crossing run with t_c < t_obs | yes | met |
| NaN recomputation identical in every run with a cutoff | 0 differ | met |

- Seed 9,776,091 did not cross within the budget (not a crossing run).
- Seed 9,776,062 crossed (t_obs = 1117) but had no cutoff; it counts as "cutoff not before the crossing" and is not
  scored.

### Criteria

| criterion | rule | value | verdict |
|---|---|---|---|
| C1 | ≥ 80% of scored runs with \|t_fc − t_obs\| ≤ 10 (miss = not within) | 93 / 98 = 0.949 | **PASS** |
| C2 | ≥ 80% of scored runs with lag error ≤ 10 (miss = not within) | 93 / 98 = 0.949 | **PASS** |
| C3 | median r_obs / r_fc in [0.90, 1.10] | 0.940 (n = 95) | **PASS** |
| C4 | upper end of the 95% bootstrap interval of mean D < 0 | mean D = −53.5, [−59.3, −47.4] (n = 95) | **PASS** |

**Known risk recorded at registration (§7.1): the thin b margin.** The median ratio 0.940 lies 0.040 inside the lower
edge of [0.90, 1.10]. The pilot median was 0.952 and the exploration's 0.94.

### The three misses (no forecast; D4: in C1 and C2's denominator, not within; excluded from C3 and C4)

All three are caused by the registered extrapolation horizon (1.6 × s*_frozen): the observed crossing lies beyond it.
None is a guard event or an error; the NaN recomputation is identical in all three. Values from `runs.jsonl`,
`observed.jsonl` and `frozen_registered.jsonl`.

| seed | horizon 1.6·s*_frozen | s_obs | status | note |
|---|---|---|---|---|
| 9,776,015 | 1.888 | 2.084 | no predicted crossing on the extrapolated path | forecast switch at step 767 (actual 769); the crossing is beyond the horizon |
| 9,776,054 | 2.021 | 2.030 | no predicted crossing on the extrapolated path | crossing 0.009 beyond the horizon; 1C's extrapolation (descriptive) forecast 538 vs 542 observed |
| 9,776,094 | 1.873 | 2.064 | no forecast switch on the extrapolated path | s*_run = 1.896 is itself beyond the horizon |

### The two forecasts outside the tolerances (DESCRIPTIVE)

| seed | t_c | t_fc | t_obs | crossing error | lag error | note |
|---|---|---|---|---|---|---|
| 9,776,017 | 871 | 814 | 916 | −102 | −101 | the forecast crossing (814) lies before the cutoff (871); the forecast switch (892) is close to the actual one (893) |
| 9,776,096 | 15,236 | 15,247 | 15,268 | −21 | −16 | a late crossing (step 15,268) |

## 3. DESCRIPTIVE (no verdict)

From `scores.json` → `DESCRIPTIVE`. "Pilot-set" runs = the scored-run definition; errors over runs with a forecast.

| variant | forecasts / misses | \|t_fc − t_obs\| q50 / q90 | lag error q50 / q90 | median r_obs/r_fc | C1 / C2 frac within (in sample) |
|---|---|---|---|---|---|
| primary (slin_20, coupled_sw), f = 0.90 | 95 / 3 | 4 / 5 | 2 / 4 | 0.940 | 0.949 / 0.949 |
| P frozen at t_c − 1 (P_cut) | 95 / 3 | 4 / 7 | 2 / 5 | 0.941 | 0.929 / 0.929 |
| 1C's extrapolation (ext_1c) | 77 / 21 | 3 / 6 | 5 / 12.8 | 1.252 | 0.786 / 0.684 |
| primary rule at f = 0.95 | 95 / 3 | 3 / 4 | 2 / 4 | 0.970 | 0.939 / 0.939 |

- **P̂ error (primary) vs P at the actual switch:** median |relative error| 0.14%, 0.14%, 0.24% (w1, b1, b2); q90 0.37%,
  0.38%, 0.76%. With P frozen at t_c − 1 the median is 7.1%, 7.1%, 3.8%.
- **1C's extrapolation** misses 21 runs (15 with no forecast switch, 4 with the path not growing at the cutoff, 2 with
  no predicted crossing), as in the exploration, where it motivated `slin_20` (disclosed at registration).

## 4. Provenance

| numbers | producer | artifact (commit) |
|---|---|---|
| verdicts, validity, counts, DESCRIPTIVE | `src/trackA_score.py score` | `results/trackA/scores.json` (this commit) |
| crash traceback | `src/trackA_causal.py score` | `results/trackA/score.log` (3f8a2fe) |
| pilot reproduction | `src/trackA_score.py pilot_check` | `results/trackA/pilot_check.log` (b500a50) |
| pilot reference numbers | `src/trackA_causal.py pilot_summary` | `results/trackA/pilot.json` (ed12b22) |
| per-run values in §2 tables | `src/trackA_causal.py run` / `observe` / `freeze` | `runs.jsonl` (1b89a73), `observed.jsonl` (3f8a2fe), `frozen_registered.jsonl` (ed12b22) |
