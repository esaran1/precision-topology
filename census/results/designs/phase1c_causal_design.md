Approved by the author 2026-10-01 with changes 1–4 (draft: 6c01389). Not a registration; no compute, no seeds drawn.
Inputs: Phase 1A (25aac9e). **Standing rule: any criterion or validity detail not on this page comes to the author
before registration.** ‡ = decided after seeing pilot (Phase 1A estimation-seed) data.

# Design 1C: registered causal replication on fresh seeds

**Arms.** Settings, holds, ρ and pilot χ (registered `pilot.json`), budgets, windings, release classification, gates,
V1–V7 and the follow check carried over unchanged; only the predictions change. **Gate to Phase 2: W1 PASS and
(G or T PASS).** Fresh seed ranges, verified unused (src/, tests/, results/, paper/) at registration.

| arm | replicates | seeds | f | gate: P(met) at pilot rate (Wilson 95% low) |
|---|---|---|---|---|
| **W1** (required) | Track A SGD at **a = 1.58**, s\*_pop = 2.090122 | 100 | **0.90‡** | no hold; ≥ 60 scored: > 0.999 at 32/40 (0.885) |
| **G** | GELU-T random start | 80 | 0.95‡ | ≥ 80% on-branch: ≈ 1 at 19/20 (0.27); registered 80/80 |
| **T** | W2-A T | 200 | 0.95‡ | ≥ 60 on T: 0.99996 at 26/60 (0.71); registered 98/200 |
| **T′** (sign test, own verdict) | W2-A T′ | 120 | 0.95‡ | ≥ 108 on T′, no hold G > 0: ≈ 1 at 25/25 (0.18; 0.993 at 0.95) |

**Cutoff.** t_c = first step with s_t ≥ f·s_ref (change 1: f = 0.90 for W1; change 2: f = 0.95 for G, T, T′). G, T,
T′: s_ref = the copy's frozen switch. W1‡: s_ref = s\*_run of the branch at the causal rule point (first t with s ≥
0.5·s\*_frozen at every step up to t − 1); this replaces Track A's R1, which used t_top (after the cutoff).

**Forecaster** (`src/causal_forecast.py`, 25aac9e, hashed): `quad`‡ extrapolation of log s (W2-A: and share) over the
last 5%‡ of the visible path, horizon 1.6·s_ref‡; R4 along visible + extrapolated path; lag-free switch from the
extrapolated path (T′'s switch and lag now forecast before the cutoff). It sees `GuardedArray` views only;
`tests/test_causal_forecast.py` fails on any read at or after t_c. Per run, last rows read < t_c are asserted and the
forecast recomputed with rows ≥ t_c NaN must be identical.

**Frozen before training** (one commit, OpenTimestamps): page, registration, forecaster, scoring code and tests,
per-seed files, f, tolerances. Forecasts committed and hashed before observation.

**Criteria‡** (per arm; scored runs; lag_fc = t_fc − t_sw,fc, lag_obs = t_obs − t_sw, signed): C1 |t_fc − t_obs| ≤
τ_cross in ≥ 80%; C2 |lag_fc − lag_obs| ≤ τ_lag in ≥ 80%; C3 median r_obs/r_fc ∈ [1 ± b]; **C4 (change 4)**: D =
|t_fc − t_obs| − |t_nolag − t_obs|, t_nolag = t_sw,fc (the no-lag forecast: crossing at the forecast switch); paired
bootstrap 95% percentile interval of D (10,000 resamples, fixed RNG seed); PASS iff its upper end < 0 (an interval
entirely above 0, or containing 0, is FAIL). T′ adds S: lag_fc < 0 and lag_obs < 0 in ≥ 80% of runs with |lag_fc| ≥ 6
(pilot 19/19). PASS = all; L1–L5 on r_fc reported descriptively.

| arm | f | τ_cross, τ_lag | b | pilot C1, C2, median ratio; forecast closer than no-lag |
|---|---|---|---|---|
| W1 | 0.90 | 10, 15 | 0.20 | 32/32, 32/32, 1.172; 32/32 |
| G | 0.95 | 10, 10 | 0.15 | 19/19, 19/19, 0.942; 17/19 |
| T | 0.95 | 15, 5 | 0.10 | 26/26, 26/26, 0.975; 21/26 |
| T′ | 0.95 | 30, 5 | 0.10 | 20/20, 20/20, 1.019; 19/20 |

W1 (change 1‡, set from the est seeds after seeing their bias): τ = 1.5 × est-seed q90, rounded up to the next
multiple of 5 steps: q90 |t_fc − t_obs| = 4.0 → 6.0 → **10**; q90 |lag_fc − lag_obs| = 6.9 → 10.35 → **15**; b = 0.20
(est median ratio 1.172). G, T, T′: the draft's tolerances (change 2).

**Descriptive only (change 3), every arm:** forecast horizon t_obs − t_c and t_sw − t_c, in steps and in lags; errors
at the other f (0.9 for G, T, T′; 0.95 for W1); the run fraction with the forecast closer than the no-lag forecast
(the draft's C4, change 4).

**Validity.** t_c strictly before the crossing in ≥ 90% of crossing runs (pilot 100%); later cutoffs unscored, counted.
Cutoff but no forecast: a miss. The follow check reads one state at t₀.₈ < t_c. Validity reads the full path after
forecasts are committed.

**Compute.** From Phase 1A's per-run times: ≈ 1.2 CPU-hours; one worker, nice 15, < 0.5 GB; ≈ 2 h wall clock.

**Outcomes.** PASS: the lag is forecast before it happens. C1 fails, C2 passes: the switch is mis-extrapolated, not the
lag. C4 fails: no better than ignoring the lag (falsifier). S fails: sign not forecast. UNRESOLVED: says nothing.

**Decisions (author, 2026-10-01).** 1. f = 0.90 for W1, b = 0.20, τ by the rule above‡. 2. f = 0.95 for G, T, T′ with the
draft's tolerances. 3. Horizons and other-f errors descriptive. 4. C4 = the paired bootstrap above; the run fraction
descriptive. 5. Everything else as drafted: W1 cutoff on s\*_run; quad / 5% / 1.6; seeds W1 100, G 80, T 200, T′ 120;
a = 1.58; verdict on C1–C4 (+ S for T′); L1–L5 descriptive; the 90% cutoff rule, no forecast = miss and the per-run NaN
recomputation as stated.
