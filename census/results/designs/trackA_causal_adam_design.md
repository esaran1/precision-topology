DRAFT for approval (2026-10-09). Not a registration; no registered or pilot seed drawn. ‡ = set after exploratory
data (`trackA_explore/`: 120 exploration seeds at a = 1.85, which already has data).

# Design: causal Adam per-run forecast at an unseen activation value

**Why, and what the exploration found.** 2A's PASS froze P at t_sw. That step lies after any causal cutoff.
- **P at the switch is predictable from rows < t_c.** At f = 0.90 the coupled v̂ model below is off by a median 0.3%
  (q90 1%). P at t_c − 1 is off by 7.7%.
- **But P was not the main problem.** With 1C's extrapolation (quad in log s, 5% window), even the true P(t_sw) gives
  32/119 misses at f = 0.90 (C1 0.72, median r_obs/r_fc 1.30).
  - Adam reaches the cutoff 30–70 steps after the rule point, so the 50-step window includes the burst before it.
  - Adam's s grows almost linearly, so the log-quadratic fit bends over below s\*_run.
- **On the actual s path (not causal), every P rule passes**, including P at t_c − 1.

**So the v̂ forecaster alone cannot pass.** The smallest honest fix‡ changes the extrapolation for Adam, stated openly.

**Setting.** Width 1, free Adam (lr 0.01), budget 32,000, every-step detection; otherwise Track A's pipeline.
**a = 1.77.**
- **Scan** (results CSV a-columns and module constants): training or crossing data at 1.5, 1.55, 1.58, 1.6, 1.65, 1.7,
  1.8, 1.85, 1.9, 2.0 and 2.2; 1.75 and 1.775 appear only in landscape tables; "1.77" occurs nowhere in src/, tests/,
  results/, paper/ or notes/.
- **Landscape:** Track A's continuation from 1.60 and validation; a failure stops before the pilot.

**Cutoff.** 1C's W1 nested stopping time on s\*_run. f = 0.90.

**Forecaster** (new `src/causal_adam.py`; `causal_forecast.py` 25aac9e unchanged):
1. **Extrapolation‡ (`slin_20`).** s is linear in t, least squares over the last min(20, t_c − t_R) steps (at least 3).
   It is anchored at s_{t_c−1} and runs to 1.6·s\*_frozen. A slope ≤ 0 means no forecast.
2. **P at the forecast switch‡ (`coupled_sw`).** R4 runs from t_R with the visible P_t before t_c.
   - After t_c it carries the raw second moment, v ← β₂v + (1 − β₂)(Hδ)², starting from v̂_{t_c−1}(1 − β₂^{t_c−1}).
   - v̂ = v/(1 − β₂^t), so β₂'s memory and the bias correction are exact.
   - P̂ = 1/(√v̂(t_sw,fc) + ε) is frozen, and R4 is rerun. This is 2A's rule, with t_sw forecast.
3. **Guards.** GuardedArray views (cutoff t_c) of w₂, M and v̂, plus one hidden row at t_R; a NaN recomputation per run.

**Exploratory result‡** (a = 1.85, f = 0.90, 119 crossing runs):
- 5 misses (4 R4-unstable).
- C1 0.92 and C2 0.92 at q90 errors of 6 and 4 steps.
- Median ratio 0.94; C4 upper end −47.
- Pilot-rule tolerances from 60 seeds (10 / 10 / 0.20) pass the other 59.
- P at t_c − 1 also passes (q90 8 / 6.7), so predicting P is only a refinement.

**Criteria (1C's).**
- **C1:** |t_fc − t_obs| ≤ τ_cross in ≥ 80% of scored runs; a miss counts as not within.
- **C2:** |lag_fc − lag_obs| ≤ τ_lag in ≥ 80%.
- **C3:** median r_obs/r_fc ∈ [1 ± b].
- **C4:** paired bootstrap of D = |t_fc − t_obs| − |t_sw,fc − t_obs|; PASS iff the upper end of the 95% interval is < 0.
- PASS = all four.

**Tolerances (pilot rule).** τ = 1.5 × pilot q90, rounded up to a multiple of 5; b = |median − 1| + 2SE, rounded up to
0.05. Source: 40 pilot seeds (9,774,000–039) at a = 1.77 after approval; the numbers come to you before registration.

**Validity** (otherwise UNRESOLVED):
- ≥ 60 crossings and ≥ 60 scored runs;
- t_c < t_obs in ≥ 90% of crossing runs (exploration: 119/119);
- the actual switch is defined;
- the NaN recomputation is identical in every run.

**Seeds.** 100 (9,776,000–099; prefixes 9774 and 9776 are unused). P(C1 ≥ 80%) is 0.94 at the Wilson lower bound
(0.85) of the exploratory rate.

**Frozen before training** (one commit): page, module, tests, landscape, per-seed s\*_frozen, tolerances, manifest.
- **OTS guard:** `run` refuses to start until `registration_stamp.txt.ots` exists, and a test checks this.
- **Causality test:** reading v̂ or M at row t_c raises CausalityViolation; a bypass fails the NaN recomputation.

**Descriptive:** P̂ error; 1C's extrapolation; P at t_c − 1; f = 0.95. **Compute:** about 3.5 s per seed; under 1 h in
total (one process, nice 15). **Falsifier:** C4 fails.

**Needs your decision.**
1. The extrapolation changes for Adam (`slin_20`‡). The alternative is to keep 1C's extrapolation and expect FAIL.
2. P rule: `coupled_sw`‡, or g² held at its window mean (`gwin`, similar accuracy).
3. a = 1.77, f = 0.90 (0.95 descriptive), 100 seeds, a 40-seed pilot.
4. R4-unstable runs stay misses.
