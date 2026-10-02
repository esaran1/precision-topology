Draft for checkpoint 3. Not a registration. No registered or pilot seed has been drawn.
Exploratory numbers are in `phase2a_explore/`. ‡ marks a choice made after the exploratory data.

# Design 2A: slow tracking on the simplicity-bias benchmark

**Setting (v3, unchanged).** The same 800 points; tanh, width 4; λ = 1e−4; s = ‖v‖₁; q = 0.3914; s\* = 3.5914 (M/S
equal loss at 3.5891); float64, full batch, no momentum.

**Prediction: the fold, not the switch.**
- Fixed-P descent never leaves a strict minimum while it exists (§14.4), so the global switch is no event on M. A run
  crosses at its own branch's first event: its switch (§13) or its fold.
- On M (stable 1.154–4.7677), ρ₂ ≤ 0.279 < q and G₊ < 0: no switch (κ undefined). The event is M's fold,
  s_F = 4.7677 = 1.33·s\*.
- Past s_F the run escapes into S, where ρ₂ ≈ 0.58 (H-F5; Track 1A: 30 of 30 starts above the fold land on S).
- **s_c = s_F(1 + Ω₀ε_F^{2/3})**, with ε_F = (ṡ_F/s_F)/(ηΛ_F) and Λ_F = (|m′c′|s_F)^{1/2}.
- So the crossing comes after v3's band (1.25·s\*), and the delay grows as ρ^{2/3}, not as ρ.
- Other branches: L0 (no fold below 37.5) never crosses; Mp (mirror pair only) has no prediction.

**Crossing** = the first upward passage of q by ρ₂ (v3's event). It can only occur in the jump from M to S. G₊ = 0
and the basin change are descriptive.

**Output rate ‡.**
- ρ on all of v slows the shares too: χ ≥ 3.5 for every ρ (exploratory crossings 1.33/0.99/0.88·s_F at ρ =
  2⁻⁸/2⁻¹¹/2⁻¹⁴, state 0.1–0.2 off M).
- Proposed: v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√3 on the active units (fixed P, §15). Then χ = 133ρ at
  s\*, ε_F = 15ρ (|m′c′| = 4.78e−7). η = 1 (ηλ_max ≤ 0.29; s_c η-free to 2e−5).

**Hold, release.**
- s₀ = 0.5·s\*: v3 init rescaled to s₀, fixed-s BFGS; label = branch within 1e−3 (exploratory, 200 seeds: M 40%,
  L0 32%, Mp 28%). Off-M runs are counted, not trained.
- M: Newton to M(s₀), idle unit exactly 0 ‡ (stays 0; a 1e−6 seed is recruited in ~300 steps).
- All M releases are one point, so ρ is the replication axis ‡: ρ_i = ρ_max·2^(−3u_i), u_i = frac(0.618034(i + 1)).

**Pilot rule.**
- Train the pilot M release to t_c only. Start from ρ = 2⁻¹⁰ and halve, at most three times, until ε̂_F ≤ 0.01 and
  max χ_t ≤ 0.1 on [s\*, t_c), with λ_min taken from the frozen table for M. If that fails, STOP.
- The exploratory value is 2⁻¹¹ (ε̂_F = 0.0070, χ ≤ 0.065).

**Cutoff, forecaster.**
- The cutoff t_c is the first step with s ≥ 0.95·s_F.
- `src/causal_forecast_fold.py` (new) reads s only, through a GuardedArray, using the frozen `extrapolate_scale`.
- It gives t̂_F, ṡ̂_F → ε̂_F, ŝ_c, t_fc; the no-delay forecast is t̂_F. Its test fails on any read ≥ t_c; per run,
  the last row read is < t_c and the NaN-after-t_c recomputation is identical.

**Criteria** (PASS requires all)
- **F**: s_F ≤ s_obs ≤ 1.25·s_F in ≥ 90% of runs, and no run at or below 1.25·s\*.
- **C1** |t_fc − t_obs| ≤ τ₁·delay_fc, and **C2** |delay error| ≤ τ₂·delay_fc, each in ≥ 80% of runs.
- **C3** median r_obs/r_fc ∈ [1 ± b].
- **C4** paired-bootstrap upper end of |t_fc − t_obs| − |t̂_F − t_obs| < 0.
- **E** slope of ln r_obs on ln ε̂_F, with its 95% CI inside [0.55, 0.80]. A linear lag would give 1.
- τ and b are set ‡ at 1.5 × the pilot error, rounded up to 0.05.
- Exploratory (7 rates): F 7/7 (1.03–1.11·s_F); errors ≤ 0.18 (τ ≈ 0.30); ratio 1.22–1.27 (b ≈ 0.40); C4 7/7;
  slope 0.68.

**Validity.**
- Gate: ≥ 25 of 120 on M (P 0.999 at the Wilson low 0.335).
- Follow check: the state at 0.8·s_F (past s\*) in M's basin in ≥ 90% (exploratory 4/4).
- Idle unit 0 and active signs fixed to the crossing; t_c before the crossing in ≥ 90% (earlier ones still count in F).

**Outcomes.**

| Observed crossing | Reading |
|---|---|
| ≤ 1.25·s\* | the switch: account falsified |
| between 1.25·s\* and s_F | an early exit, as in v3 |
| E ≈ 1 | a linear lag, not the fold delay |
| E ≈ 0, or C4 fails | no rate-dependent delay |
| no crossing by 1.25·s_F | H-F5 fails |
| all pass | crossing at the fold, ρ^{2/3} delay forecast in advance |

**Seeds:** 2,936,000–2,936,119; pilot 2,936,900–2,936,959 (both verified unused).
**Compute:** ≈ 4 h, one worker, nice 15, < 0.5 GB: ~48 M runs (forecast, then observation) and ~24 pilot runs at
0.5–3 min each.

**Needs your decision.**
1. ρ on the output scale only.
2. Idle unit 0 (tests the 3-unit invariant subnetwork).
3. ρ ladder (ρ_max/8 to ρ_max) as the replication axis.
4. ρ₂, not G₊, as the event.
5. s₀ = 1.7957 (at 2.5: M 49.5%, same s_c).
6. BFGS for the hold.
7. f = 0.95 (at 0.90 the crossing errors are ≤ 0.08).
8. b ≈ 0.40 (leading-order bias), or lower ρ.
9. Whether to train the runs that land off M.
10. The tolerance rule and the band for E.
