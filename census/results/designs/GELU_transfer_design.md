Approved by the author 2026-09-29 (changes: Adam arm dropped; random start primary, branch-point start a mechanism
control; L4 against s_glob; L5 and a follows-branch condition added). The draft with its exploratory numbers is 5e6323f.

# Design GELU-T: the lag law at GELU under free SGD

**Setting (Track 3A).** Width 1, z = w₂·GELU(w₁x + b₁) + b₂; own sample `make_data(200, seed)`; float64; every-step
detection; s = |w₂|; s₀ = 0.5·s_pop, s_pop = 6.645633 (tracked branch). Seeds 876,000–876,079 for both arms; pilot
876,900–876,909.

**Both arms.** w₂ held at +s₀ while (w₁, b₁, b₂) relax by GD (lr 0.3) for W_seed = max(4000, ⌈25/(0.3·λ_min,own(s₀))⌉)
steps, G checked each step. Then full-batch SGD: η = 0.03 on (w₁, b₁, b₂), ρη on w₂, 40,000 steps. At release,
Newton at s₀ puts the run on a copy of its own-sample branch (within 1e−6 of the frozen θ\*_own(s₀), state within 1e−3).
- **Primary (random start):** hidden ~ U(−1, 1)³. A run on the target (w₁ > 0) or mirror copy gets that copy's
  prediction; a run on neither gets none and is counted, never replaced. Gate: ≥ 80% on a copy, no G > 0 in the hold.
- **Control (branch-point start):** θ\*_pop(s₀), odd seeds mirrored. Gate: ≥ 90% on target, no G > 0 in the hold.
- Failed gate: UNRESOLVED, stop. The headline verdict is the primary arm's.

**Frozen (hashed, timestamped):** per seed and copy θ\*_own(s₀), λ_min, W_seed, s_switch,branch (validated
continuation), κ_seed (P = I); ρ; code.

**Predictions.** r_traj: Track A's R4 recursion from release on the occupied branch. r_cf = κ_seed·χ at t_sw (first
s ≥ s_switch,branch). Lags are against s_switch,branch.

**Why η = 0.03.** In free training the lag in s does not depend on the learning rate. This is proved for fixed-P
gradient descent, SGD included (Corollary L3, §15.4), and was observed in the registered R4 (§13.6: free Adam,
η = 0.01/0.005/0.0025, sine family, PASS at both a). No registered result covers SGD or GELU. η = 0.03 only makes the
lag span enough steps to resolve.

**Pilot rule** (pilot seeds, to t_sw): ρ = min(1, 2^⌊log₂(0.1/q90)⌋), q90 of κ_seed·χ at t_sw; if ρ < 1, rerun, keep ρ
if q90 ≤ 0.1, else halve (at most three times), then STOP.

**Criteria, per arm** (scored: on a copy, crossed, prediction finite)
- L1 median r_obs/r_traj ∈ [0.90, 1.10]; L2 median r_obs/r_cf ∈ [0.80, 1.20]; L3 Spearman(r_traj, r_obs) ≥ 0.5.
- L4 D = |log(s_obs/s_pred)| − |log(s_obs/s_glob)|, s_glob = 6.641492; L5 the same with s_switch,branch (tests the lag
  itself). PASS if the 95% percentile bootstrap upper end of mean D < 0 (10,000 resamples, fixed seed).

**Validity** (else L1–L5 UNRESOLVED): ≥ 60/80 cross with a prediction; t_sw before the crossing in ≥ 90% of crossing
runs with κ_seed > 0; ηλ_min ≤ 0.5 in ≥ 80% of scored runs; median predicted lag ≥ 10 steps; q90 κχ at t_sw ≤ 0.1;
median χ within 30% of the pilot's; q90 of max_{t ≤ t_sw} χ_t ≤ 0.25.

**Readings.** All pass: the law transfers to GELU with no fit. L2 alone: the slaved form misses the transient. L3:
ordering. L4: the own-branch switch misses the sample variation. L5: the lag is unresolved.
