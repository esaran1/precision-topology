# Item 2: what sets where width-2 training crosses? Diagnosis rules and gate (written before any per-predictor result)

Committed before any predictor below is computed or compared with crossing scales. POST HOC diagnosis on existing runs
(T2-3: φ₂ = 1; T2-3b: φ₂ = 0.009306; T2-3c: φ₂ = 0.01778; a = 1.30, Δ = 0.4). T2-3 stays FAIL, and T2-3b and T2-3c
stay UNRESOLVED, whatever this shows.

**Candidate predictors of each run's crossing scale s_cross.**
- **P1, population width-2 threshold:** s = 0.44507 (the T2-1 bracket midpoint). Constant.
- **P2, own-sample width-2 threshold** on the run's own 400-point training set. Method: the earlier 10-seed check
  (`asym_posthoc2.own_threshold`: 200 restarts per scale, scan, bisection to 5%). Evaluated on the **first 10 crossing
  seeds by seed order in each arm**; T2-3's 10 already exist.
- **P3, own-sample width-1 threshold** on the same training set and windows. Method: the width-1 conditional search
  (`asym_pilot._width1_batch`, 100 restarts, polished) at a scan s = 10^(k/8), k = 0..13, then bisection to 5%. Same
  10 seeds per arm.
- **P4, population width-1 threshold** for these windows: s = 5.0937 (`asym_posthoc/width1.json`). Constant.
- **P5, branch switch scale** from the run's own crossing configuration: the smallest s at which damped Newton from that
  configuration lands placed (`asym_posthoc.branch_switch`). Evaluated for every crossing run.

**Metrics,** per φ₂ setting and pooled over the three: the median of |log(s_cross / predictor)|, and
Spearman(s_cross, predictor). A constant predictor has no Spearman correlation (reported as undefined).

**Selection rule, fixed now.** Take the predictor with the lowest pooled median absolute log error, computed on the
**common subset** (the 30 seeds that have P2 and P3), so every predictor is compared on the same runs.

**Gate, fixed now.** GO only if the selected predictor has:
- a median absolute log error ≤ 0.10 in **every** φ₂ setting (on its evaluated runs), **and**
- a pooled Spearman ≥ 0.6. A constant predictor therefore cannot pass.

Otherwise STOP: the width-2 section of the paper stays as it is, and the diagnosis is reported as exploratory only.
A registered prospective test (step 3) runs only on GO, only if a timed sample shows it can be scored by 01:30 EDT,
and is registered before any run.
