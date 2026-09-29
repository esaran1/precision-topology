# Design GELU-T: the lag law at GELU from a declared warm start (for approval; not registered)

**Setting (as Track 3A)**
- Width 1, z = w₂·GELU(w₁x + b₁) + b₂ (exact erf form); I = [−0.8, 0.8] class 0, O = ±[1.2, 2.0] class 1; each seed's
  own 400-point sample `fold1d.make_data(200, seed)`; float64; every-step detection (`phase2b_ordering.state`, dense
  windows); s = |w₂|, w₂ > 0 (placement needs σ = +1).
- **s_pop = 6.645633**, the population switch on the tracked branch (`kappa_gelu_frozen.json`; inside the validated
  bracket [6.6117, 6.6714]). κ_SGD(s_pop) = 0.132; κ_Adam = 0.147 with 3A's calibration P.
- Known from free Adam (3A, 200 seeds): χ at crossing median 0.0137 (IQR 0.008–0.022); crossings sit at their branch's
  switch (median 1.0017, 0/114 folds, POST HOC); branch switch / s_pop has q10–q90 0.88–1.17.
- **Seeds:** registered 876,000–876,079 (80, shared by both arms); pilot 876,900–876,909. No seed in 876,000–876,999
  appears in `src/`, `tests/`, `results/` (text, parquet) or `paper/`; Track 2A's WIP uses 1,850,000–1,850,079.

**Exploratory feasibility (population, or landscape only on 12 already-used 3A seeds 850,000–850,011; recomputed and
frozen at registration)**
- The population branch continues from s_pop down to s = 0.3 with no fold or branch point. At s = 0.3 it is the
  one-sided ramp of the small-scale minimiser (w₁ = 65). So the branch at s₀ is the small-scale branch.
- At s₀ = 0.5·s_pop = 3.323: G = −0.100 and λ_min(H) = 0.035. dL/ds < 0 at every sampled s in [0.3, 13], so w₂ grows
  after release.
- Own samples: both copies (w₁ ≷ 0) continue from s₀ to a switch with no fold. For the 8 seeds act_fold analysed,
  the occupied copy's switch equals act_fold's to 6 digits. Switch / s_pop: 0.70–1.43. Own κ_SGD: 0.076–0.22, and
  one copy is −0.19.
- Free SGD after release (ρ = 1): χ at the switch = 0.022, so κχ = 0.003. Closed form obs/pred: 0.957 at η = 0.03
  (lag 35 steps) and 0.99 at η = 0.1 (10 steps). With a faster output (ρ = 3, 10), obs/pred falls to 0.66 and 0.01.
- **Adam after release:** from the relaxed branch with a fresh state, ηλ_min(P_tsw) = 4.6–5.7 (lr 0.01 to 0.001). That
  is outside the law's regime (§13.3(iv)(c)), like the 1B ramp. The Adam lag is 0–7 steps at both lr 0.01 and 0.001,
  so it does not grow as lr is lowered.

**Warm start and release**
- Start: the population branch point θ\*_pop(s₀) (1B's start), mirrored (w₁ → −w₁) for odd seeds. The hold relaxes it
  onto the seed's own branch; the target is that copy's own-sample point θ\*_own(s₀) (Newton from θ\*_pop(s₀)).
- Hold: w₂ = +s₀ fixed. (w₁, b₁, b₂) relax by full-batch GD (lr 0.3) for W_seed = max(4000, ⌈25/(0.3·λ_min,own(s₀))⌉)
  steps. G is checked at every hold step.
- Release: all four parameters train freely, with the arm's optimiser and a fresh optimiser state.
- Budget: 40,000 steps after release.

**Branch classification at release**
- Damped Newton at s₀ from the release state, accepted if max|∇| < 1e−8 and H is positive definite.
- The run is **on target** if the result is within 1e−6 (sup norm) of the frozen θ\*_own(s₀) of its copy and the
  release state is within 1e−3 of it.
- Nothing after release enters the branch identity.

**Frozen before any training** (one commit, with a SHA-256 manifest and an OpenTimestamps proof of the commit)
- The population landscape: s_pop, θ\*_pop(s₀), H, θ\*′ and ∇G.
- Per seed (pilot and registered) and per copy: θ\*_own(s₀), λ_min(s₀), W_seed, and **s_switch,branch**.
  - It is found by pseudo-arclength continuation (act_fold) from s₀ to 1.6·s_pop.
  - Validation: step halving agrees to 1e−6; λ_min > 0 with no turning point on [s₀, s_switch]; the switch is the
    first G sign change and is bisected to 1e−10; the exact-extrema G is decided at (1 ± 1e−3)·s_switch.
- Also per seed: **κ_seed** (own H, θ\*′ and ∇G at s_switch; P = I; ∇G one-sided agreement ≤ 1e−5).
- ρ from the pilot, s₀, the rules above, the code and tests.

**Prediction.** s_cross = s_switch,branch·(1 + κχ). The lag law r = κχ comes from §13.1. Theorem L2 (§15) proves it
for fixed-P descent, so it covers the SGD arm but not Adam (§15.5). Release starts on the branch (δ₀ ≈ 0), so the
tracking transient starts from a known state, and ηΛ ≤ 1 holds at η = 0.03.
- **Trajectory-integrated (r_traj):** Track A's R4 recursion is started at release. It uses the own branch along the
  run's s_t, and the first step with G(θ\*(s_t) + δ_t) > 0.
- **Closed form (r_cf):** κχ, with χ = (ṡ/s_switch)/(ηλ_min(P^{1/2}HP^{1/2})) and ṡ over the 100 steps before t_sw.
  Here t_sw is the first step with s ≥ s_switch,branch.

**Arms**
- **(i) SGD:**
  - η = 0.03 for (w₁, b₁, b₂) and ρη for w₂, no momentum.
  - η is set a priori so that the median lag κ/(ηλ_min) is at least about 25 steps. At Track A's η = 0.3 it is 3–4
    steps, which is at the step resolution.
- **(ii) Adam:**
  - lr 0.01, with the 2A rule: P = 1/(√v̂ + ε) frozen at t_sw, from the occupied branch's s_switch.
  - κ is taken per run from P_tsw's shape and the own-branch H, θ\*′ and ∇G.

**Pilot rule (SGD arm; before registered training; pilot seeds only)**
- Run the arm at ρ = 1 on 876,900–876,909 only up to t_sw. No gap is evaluated after release, and no crossing or
  residual is computed.
- Measure each pilot run's κ_seed·χ at t_sw.
- ρ = min(1, 2^⌊log₂(0.1/q90)⌋), where q90 is the 90th percentile of the pilot κχ values.
- If ρ < 1, rerun the pilot at ρ. Keep ρ only if q90 ≤ 0.1; otherwise halve ρ, up to three times, and then STOP.
- Exploratory expectation: q90 ≈ 0.003, so ρ = 1 (the rule is inactive).

**Gate (registered; checked per arm before any scoring):** at least 90% of the 80 warm-started runs are on target at
release, and no run has G > 0 at any hold step. Otherwise the arm is UNRESOLVED, and it stops.

**Criteria, registered, per arm** (scored runs: on target, crossed, prediction finite; the lag is measured against
s_switch,branch)
- L1: median r_obs/r_traj ∈ [0.90, 1.10].
- L2: median r_obs/r_cf ∈ [0.80, 1.20].
- L3: per-run Spearman(r_traj, r_obs) ≥ 0.5.
- L4 (beats the population switch):
  - For each run, D = |log(s_obs/s_pred)| − |log(s_obs/s_pop)|.
  - PASS if the upper end of the 95% percentile bootstrap interval of mean D (10,000 run resamples, fixed RNG seed)
    is < 0.

**Validity, per arm** (if any fails, L1–L4 are UNRESOLVED)
- At least 60 of 80 runs cross with a prediction.
- t_sw precedes the crossing in at least 90% of crossing runs with κ_seed > 0. Runs with κ_seed < 0 are predicted to
  cross early, are scored for SGD, and are unscored for Adam (their P would come after the crossing).
- Regime: ηλ_min(P_tsw) ≤ 0.5 in at least 80% of scored runs.
- Resolution: the median predicted lag is at least 10 steps.
- SGD only: the realised q90 of κχ at t_sw is ≤ 0.1, and the realised median χ is within 30% of the pilot median.

**Competing outcomes and falsifiers**
- **All pass:** the law transfers to GELU from a declared start with no fit, and the lag is set by the occupied
  branch's switch.
- **L1 passes and L2 fails:** the slaved closed form misses the early transient (exploratory 0.957). The recursion
  does not.
- **L3 fails:** the magnitude is right but the per-run ordering is not.
- **L4 fails:** the own-branch switch does not carry the sample variation. This would contradict 3A's post hoc result.
- **Gate fails:** the warm start does not hold runs on the branch.
- **Adam regime or resolution UNRESOLVED (expected):** the law's Adam form has no support after a warm start (as in 1B).

**Compute (one worker, nice 15, about 0.5 GB)**
- Freezing 90 seeds × 2 copies, with step halving: about 10 minutes.
- Pilot: 1 minute.
- 160 runs: about 15 minutes.
- r_traj replays: about 30 minutes.
- Total: under 1.5 hours.

**Needs your decision**
1. **Adam arm.** Exploration shows it will be UNRESOLVED on the regime condition, the resolution condition or both.
   The options are (a) keep it as written, as a prospective check of that regime; (b) drop it; (c) replace it with free
   Adam from U(−1, 1)⁴, which has no warm start and a gate at 2A's rule point. Option (c) still has lags of 0–7 steps.
2. **η = 0.03 for SGD**, for resolution, or Track A's 0.3.
3. **Pilot rule.** As specified, it is inactive (κχ ≈ 0.003). Should it also require the slaving condition along the
   path (q90 of max over t ≤ t_sw of χ_t ≤ 0.25)? The option ρ > 1 is not offered, because exploratory ρ = 3 already
   fails.
4. **Start.** The population branch point (1B, recommended) or a random hidden init. On the population, a random
   hidden init put 54/60 runs on the target or mirror branch (5 at w₁ = 0), which makes the gate borderline.
5. s₀ = 0.5·s_pop (1B) or 0.6·s_pop, where λ_min is 0.063 instead of 0.035.
6. Copy by seed parity, or all w₁ > 0.
7. s_pop = 6.645633, or 3A's s_glob = 6.641492.
8. **Add L5 (lag against no lag).** L4 tests the sample term (about 10%), not the lag (about 0.3%). L5 would be L4's
   paired rule with s_switch,branch in place of s_pop.
9. L4 uses the mean of D. Mean or median?
10. 80 seeds per arm, shared between the arms.
