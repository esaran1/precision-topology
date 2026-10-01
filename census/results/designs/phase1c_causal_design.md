DRAFT for the author's approval, 2026-10-01. Not a registration; no compute, no seeds drawn. Inputs: Phase 1A
(25aac9e). **Standing rule: any criterion or validity detail not on the approved page comes to the author before
registration.** ‡ = decided after seeing pilot data.

# Design 1C: registered causal replication on fresh seeds

**Arms.** Settings, holds, ρ and pilot χ (registered `pilot.json`), budgets, windings, release classification, gates,
V1–V7 and the follow check carried over unchanged; only the predictions change. **Gate to Phase 2: W1 PASS and
(G or T PASS).** Fresh seed ranges, verified unused at registration.

| arm | replicates | seeds | gate: P(met) at pilot rate (Wilson 95% low) |
|---|---|---|---|
| **W1** (required) | Track A SGD at **a = 1.58**, s\*_pop = 2.090122 | 100 | no hold; ≥ 60 scored: > 0.999 at 32/40 (0.885) |
| **G** | GELU-T random start | 80 | ≥ 80% on-branch: ≈ 1 at 19/20 (0.27); registered 80/80 |
| **T** | W2-A T | 200 | ≥ 60 on T: 0.99996 at 26/60 (0.71); registered 98/200 |
| **T′** (sign test, own verdict) | W2-A T′ | 120 | ≥ 108 on T′, no hold G > 0: ≈ 1 at 25/25 (0.18; 0.993 at 0.95) |

**Cutoff.** t_c = first step with s_t ≥ f·s_ref, **f = 0.95‡**. G, T, T′: s_ref = the copy's frozen switch. W1‡:
s_ref = s\*_run of the branch at the causal rule point (first t with s ≥ 0.5·s\*_frozen at every step up to t − 1; Track
A's R1 used t_top, after the cutoff).

**Forecaster** (`src/causal_forecast.py`, 25aac9e, hashed): `quad`‡ extrapolation of log s (W2-A: and share) over the
last 5%‡ of the visible path, horizon 1.6·s_ref‡; R4 along visible + extrapolated path; lag-free switch from the
extrapolated path (T′'s switch and lag now forecast before the cutoff). It sees `GuardedArray` views only;
`tests/test_causal_forecast.py` fails on any read at or after t_c. Per run, last rows read < t_c are asserted and the
forecast recomputed with rows ≥ t_c NaN must be identical.

**Frozen before training** (one commit, OpenTimestamps): page, registration, forecaster, scoring code and tests,
per-seed files, f, tolerances. Forecasts committed and hashed before observation.

**Criteria‡** (per arm; scored runs; lag_fc = t_fc − t_sw,fc, lag_obs = t_obs − t_sw, signed):
C1 |t_fc − t_obs| ≤ τ_cross in ≥ 80%; C2 |lag_fc − lag_obs| ≤ τ_lag in ≥ 80%; C3 median r_obs/r_fc ∈ [1 ± b];
C4 |t_fc − t_obs| < |t_sw,fc − t_obs| in ≥ 50%. T′ adds S: lag_fc < 0 and lag_obs < 0 in ≥ 80% of runs with
|lag_fc| ≥ 6 (pilot 19/19). PASS = all; L1–L5 on r_fc reported.

| arm | τ_cross, τ_lag | b | pilot C1, C2, median ratio, C4 |
|---|---|---|---|
| W1 | 10, 10 | 0.10 | 32/32, 32/32, 1.094, 32/32 |
| G | 10, 10 | 0.15 | 19/19, 19/19, 0.942, 17/19 |
| T | 15, 5 | 0.10 | 26/26, 26/26, 0.975, 21/26 |
| T′ | 30, 5 | 0.10 | 20/20, 20/20, 1.019, 19/20 |

**Validity.** t_c strictly before the crossing in ≥ 90% of crossing runs (pilot 100%); later cutoffs unscored, counted.
Cutoff but no forecast: a miss. The follow check reads one state at t₀.₈ < t_c. Validity reads the full path after
forecasts are committed.

**Compute.** From Phase 1A's per-run times: ≈ 1.2 CPU-hours; one worker, nice 15, < 0.5 GB; ≈ 2 h wall clock.

**Outcomes.** PASS: the lag is forecast before it happens. C1 fails, C2 passes: the switch is mis-extrapolated, not the
lag. C4 fails: no better than ignoring the lag (falsifier). S fails: sign not forecast. UNRESOLVED: says nothing.

**Needs your decision**
1. f‡: 0.95; or 0.9 for W1 (cutoff 73–151 steps before the switch, not 37–76; ratio 1.17, b 0.20). The one-lag rule
   forces 0.8: crossing error q90 850 (T), 2,684 (T′) steps.
2. W1 cutoff on s\*_run‡ or s\*_frozen (after the crossing in 3/32 pilot runs).
3. Every τ, b and the 80%/50% thresholds‡ (set on est seeds after seeing them). W1's 1.094 lies 0.006 inside its band
   (≈ 3% chance of leaving it by sampling; R4's own 4-step bias at lr 0.3).
4. Seed counts (Track A's 80 give 0.89 for W1).
5. a = 1.58 (no a-column equals it).
6. Verdict on C1–C4 (+ S) only, or L1–L5 too (pilot Spearman 0.90–0.999).
7. New here: S, the 90% cutoff rule, no forecast = miss, the per-run NaN recomputation.
