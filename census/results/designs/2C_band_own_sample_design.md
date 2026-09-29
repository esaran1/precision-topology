# Design 2C: band task in R^d against the own-sample switch, prospectively (for approval; not registered)

**Setting.** As Track 3B:
- the x₁ band task, with x₂…x_d i.i.d. U(−2, 2), at d = 2 and 4 and a = 1.30 and 1.50;
- a single f_a unit, free Adam (lr 0.01), every-step detection;
- **60 fresh seeds per cell**, because d = 4 crossed only about 73% of the time in 3B.

**Prediction**
- s_cross = s_own,d·(1 + κχ).
- s_own,d is each seed's own-sample R^d switch, computed BEFORE training: the placement switch of the own-sample
  conditional minimiser in all coordinates, noise weights included, by the validated conditional search (restart
  ladder, independent search, audit).
- κ is the width-1 κ_k.
- χ comes from pre-crossing information only, with the Track A rule point (last upward passage of 0.5·s_own,d) and P
  frozen at t_sw (the 2A rule, if 2A passes first; otherwise at the rule point, stated in the registration).

**Frozen before training** (hash plus OpenTimestamps): every seed's s_own,d, κ, the P and χ rules, the pipeline.

**Criteria, registered, per cell**
- C1: median |log(s_obs/s_pred)| ≤ 0.05.
- C2: median lag ratio obs/pred in [0.75, 1.25].
- C3: the prediction beats the x₁-only own threshold (paired interval of mean |log err| difference below 0). That
  predictor ignores the noise coordinates, which were the post hoc 12% (d = 2) and 40% (d = 4) static offset.

**Validity**
- At least 30 crossings per cell.
- The noise ratio ρ at crossing is < 0.05 in at least 90% of crossing runs.

**Competing outcomes**
- PASS: the R^d excess is a static own-sample threshold shift plus a width-1 lag, confirmed prospectively.
- C1 or C2 fails: the post hoc decomposition does not hold prospectively.
- C3 fails: the noise-coordinate shift is not predictable from the sample before training.

**Compute**
- The own-sample R^d switch per seed is the costly part: validated search in d + 3 coordinates, about 1–3 min per
  seed, so 240 seed-cells take about 4–10 hours at one worker.
- Training: under 1 hour.
