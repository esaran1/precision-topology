Approved by the author 2026-10-04 (f = 0.95; the worker's item-2 choices; the fixed dataset, one landscape across rates,
with the resampling exploration as a disclosed limit; everything else as revised in 053dca5). Not yet a registration:
it registers with `results/phase2a_registration.md`, and its OpenTimestamps proof comes before any registered run.
Exploratory numbers: `phase2a_explore/` (`p2a_explore_f`, `p2a_explore_g`). ‡ = set after exploratory data.

# Design 2A: slow tracking on the simplicity-bias benchmark

**Setting (v3, unchanged).** The same 800 points; tanh, width 4; λ = 1e−4; s = ‖v‖₁; q = 0.3914; s\* = 3.5914;
float64, full batch, no momentum, η = 1.

**One landscape.** 2A runs on the FIXED v3 dataset. It tests ONE landscape across rates: the rates, not the seeds, are
the replicates, and every fraction and bootstrap runs over the 27 distinct rates. Whether the result holds on other
samples of the same generator is not tested (disclosed limit, below).

**Prediction: the fold, not the switch.**
- Fixed-P descent never leaves a strict minimum while it exists (§14.4), so the global switch is no event on M. A run
  crosses at its own branch's first event: its switch (§13) or its fold.
- On M (stable 1.154–4.7677), ρ₂ ≤ 0.279 < q and G₊ < 0: no switch. The event is M's fold, s_F = 4.7677 = 1.33·s\*.
- Past s_F the run escapes into S, where ρ₂ ≈ 0.58 (H-F5; Track 1A: 30 of 30 starts above the fold land on S).
- **s_c = s_F(1 + Ω₀ε_F^{2/3})**, ε_F = (ṡ_F/s_F)/(ηΛ_F), Λ_F = (|m′c′|s_F)^{1/2}: the delay grows as ρ^{2/3}, not ρ.

**Crossing** = the first upward passage of q by ρ₂ (v3's event).

**Output rule.** v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√3 on the active units: only the output scale is
slowed. This is not the practitioner's knob: the plain output learning rate leaves χ ≥ 3.5 at every ρ (Phase 2B tests
that knob).

**Hold, release.** v3 init rescaled to s₀ = 1.7957, fixed-s BFGS; label = branch within 1e−3. Runs that land off M are
counted, not trained. M runs: Newton to M(s₀), idle unit exactly 0, so **this is a three-unit test**. All M releases
are one point; runs differ only in ρ.

**Rate ladder ‡** (27 distinct rates). Anchors, already explored: 2⁻¹³, 2⁻¹⁴, 2⁻¹⁵. New:
2^−(15 + k/8), k = 1…24 (2⁻¹⁵·¹²⁵ … 2⁻¹⁸). The i-th seed that lands on M takes the i-th rate, fastest first.

**Pilot ‡.** Three pilot rates off the ladder: 2⁻¹⁵·⁰⁶²⁵, 2⁻¹⁶·⁵, 2⁻¹⁷·⁹³⁷⁵. Each runs from a pilot seed that lands on M.
STOP if ε̂_F > 0.01 or max χ_t > 0.1 on [s\*, t_c) at 2⁻¹⁵·⁰⁶²⁵ (λ_min from the frozen M table). τ₁ and
τ₂ = 1.5 × the largest pilot error (at f = 0.95), rounded up to 0.05.

**Cutoff, forecaster.**
- t_c = the first step with s ≥ 0.95·s_F.
- `src/causal_forecast_fold.py` (new) reads s only, through a GuardedArray, using the frozen `extrapolate_scale`
  (quad, 5% window). It gives t̂_F, ṡ̂_F → ε̂_F, ŝ_c, t_fc; the no-delay forecast is t̂_F.
- No ŝ ≥ ŝ_c within 1.6·s_F means no forecast, which fails C1, C2 and C4.
- Its test fails on any read ≥ t_c; the last row read is < t_c, and recomputation with NaN after t_c is identical.
- Training stops at t_c and the state is hashed; observation resumes from it to 1.25·s_F.

**Descriptive only (never a verdict).**
- **Each rate's forecast horizon** t_obs − t_c, in steps and in delays ((t_obs − t_c)/delay_obs).
- **The tightening cutoff.** 1 − f ∝ ρ^{2/3}, calibrated so that it equals 0.05 at 2⁻¹⁵: 1 − f(ρ) = 0.05·(ρ/2⁻¹⁵)^{2/3}
  (f = 0.874 at 2⁻¹³, 0.95 at 2⁻¹⁵, 0.9875 at 2⁻¹⁸). The same forecaster runs at f(ρ) and the C1–C4 statistics are
  reported beside the registered ones.
- Horizon at f = 0.95 (exploratory):

| ρ | 2⁻¹¹ | 2⁻¹⁴ | 2⁻¹⁵ | 2⁻¹⁶ | 2⁻¹⁷ | 2⁻¹⁸ |
|---|---|---|---|---|---|---|
| steps | 15,190 | 54,408 | 93,696 | 169,057 | 315,803 | ≈0.6 M (extrap.) |
| delays | 1.4 | 2.8 | 4.0 | 5.9 | 8.9 | ≈13 |

**Criteria.** Each run is deterministic given ρ, so every fraction and bootstrap runs over the 27 distinct rates
(10,000 resamples). PASS requires all of these:
- **F** — s_F ≤ s_obs ≤ 1.25·s_F at ≥ 90% of rates, and at no rate s_obs ≤ 1.25·s\*.
- **C1** — |t_fc − t_obs| ≤ τ₁·delay_fc at ≥ 80% of rates.
- **C2** — |delay_fc − delay_obs| ≤ τ₂·delay_fc at ≥ 80% of rates.
- **C3 (asymptotic) ‡** — Spearman(ρ, r_obs/r_fc) ≥ 0.8 (positive: the ratio rises with ρ, so it falls toward 1 as
  ρ → 0), and r_obs/r_fc > 1 at every rate.
- **C4** — the upper end of the paired bootstrap of mean(|t_fc − t_obs| − |t̂_F − t_obs|) < 0.
- **E** — the slope of ln r_obs on ln ε̂_F has its 95% CI inside [0.55, 0.80]. A linear lag would give 1. The band was
  set knowing the exploratory slope, 0.683.

**Validity.**
- Gate: ≥ 27 of 120 seeds on M (P = 0.997 at the Wilson low, 0.335).
- At ≥ 90% of rates: the state at 0.8·s_F lies in M's basin, and t_c comes before the crossing.
- At every rate, the idle unit stays 0 and the active signs stay fixed up to the crossing.

**Expected outcome on the new ladder at f = 0.95 (exploratory).**

| ρ | 2⁻¹⁵ | 2⁻¹⁶ | 2⁻¹⁷ |
|---|---|---|---|
| s_obs/s_F | 1.016 | 1.010 | 1.006 |
| C1 error/delay_fc | 0.10 | 0.06 | 0.003 |
| C2 error/delay_fc | 0.14 | 0.11 | 0.09 |
| C4 difference (steps) | −20,694 | −25,892 | −32,271 |
| r_obs/r_fc | 1.196 | 1.167 | 1.142 |

- The ratio continues the explored 1.266 (2⁻¹¹) → 1.224 (2⁻¹⁴) at f = 0.95.
- F, C1, C2, C3, C4 and E: expected to pass. (At f = 0.90, C1 and C4 were expected to fail: the error in t̂_F doubles
  each octave while the delay grows only as ρ^{−1/3}; hence f = 0.95.)

**Outcomes.**

| Observed | Reading |
|---|---|
| crossing ≤ 1.25·s\* | the switch: the account is falsified |
| between 1.25·s\* and s_F | an early exit, as in v3 |
| E ≈ 1 | a linear lag, not the fold delay |
| E ≈ 0, or C4 fails with C1 | no rate-dependent delay, or the forecast horizon is too long |
| ratio not falling toward 1 | the leading-order fold asymptotics fail |
| no crossing by 1.25·s_F | H-F5 fails |
| all pass | the crossing comes at the fold, with a ρ^{2/3} delay forecast in advance |

**Seeds** (verified unused): 2,937,000–2,937,119; pilot 2,937,900–2,937,959.
**Compute** (one worker, nice 15, peak RSS 0.83 GB at 2⁻¹⁷, measured): each run takes 124 s at 2⁻¹⁵ and doubles each
octave. Ladder 2.97 h; pilot 24 min; holds and freeze < 5 min; total ≈ 3.4 h.

**Disclosed limit: per-seed samples (explored, not adopted; `p2a_explore_g`).**
- Freeze: 13.5 s per seed, measured (continuation 7.1, validation 2.1, |m′c′| 0.4, S and s\* 3.9). One rate per M seed
  would cost 1.15× the ladder alone; every seed on the whole ladder ≈ 30 h.
- On resampled data the M branch moves: stable ranges 2.69–5.62 (seed 2,930,000) and 1.96–5.86 (seed 2,930,001).
- s₀ = 1.7957 lies off M on both traced seeds.
- One seed's branch (2,930,002) could not be traced (the data homotopy failed).
- On seed 2,930,001: |m′c′| = 6.9e−8 against 4.78e−7, and s_F/s\* = 1.09. Its run at 2⁻¹⁵ (released at 2.057 on its
  own M) did not cross by 1.25·s_F.
- So 2A says nothing about sample-to-sample variation; a per-seed design would need new choices that bear on criteria.

**Decided (author, 2026-10-04)**

| # | Decision |
|---|---|
| 1 | Output scale only; disclosed as not the practitioner's knob |
| 2 | Idle unit exactly 0; a three-unit test |
| 3 | Ladder 2⁻¹⁵…2⁻¹⁸ with explored anchors; cost of per-seed samples estimated |
| 4 | Crossing = the first upward passage of q by ρ₂ |
| 5, 6 | s₀ = 1.7957; BFGS hold |
| 7 | **f = 0.95** (approval); the tightening cutoff (1 − f ∝ ρ^{2/3}) and each rate's horizon are descriptive only |
| 8 | C3 → asymptotic ratio test |
| 9 | Off-M runs counted, not trained |
| 10 | τ rule; E band [0.55, 0.80] (disclosed) |
| 11 | 27 distinct rates at ⅛-octave spacing, assigned in seed order to M landings, fastest first; three pilot rates off the ladder; gate 27 of 120; no forecast fails C1, C2, C4; τ₁, τ₂ = 1.5 × the largest pilot error at f = 0.95, rounded up to 0.05; training stops at t_c, state hashed, observation resumes from it |
| 12 | Fixed dataset: one landscape across rates; the per-seed resampling exploration is a disclosed limit |
