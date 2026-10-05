# Revision notes (record only; author-requested 2026-10-02). Not writer inputs; the revision is held until reviews arrive.

## From Phase 1B (POST HOC causal re-scoring, commit 221c8ce; results/phase1b_results.md)

1. **Track A's per-run claim does not hold as a causal forecast.** With the frozen causal forecaster (f = 0.90, cutoff on
   s*_run), Track A SGD (a = 1.65) fails C3 (median r_obs/r_fc = 1.216, outside ±0.20; C1, C2, C4 pass) and Track A Adam
   fails C1 and C2 (39/74 and 43/74 within tolerance; 8 misses). The registered verdicts stand (SGD L1–L3 PASS; Adam
   L1, L2 PASS, L3 FAIL), but the per-run agreement relied on the run's own path up to the predicted step
   (results/prediction_inputs_audit.md, 432a5a2).
2. **Test 2A's PASS does not hold as a causal forecast.** Re-scored causally (P at t_c − 1 standing in for the
   registered P at t_sw, which lies after the cutoff), 2A fails C1, C2 and C3 (55/75, 55/75, median ratio 1.354;
   19 misses). The registered PASS (A1 0.932, A2 0.773, A3 1.058) stands as registered but used P at t_sw.

Descriptive context (same file): with P at t_c − 1, Track A Adam passes C1–C4 at both f; Track A SGD passes at
f = 0.95. The slow, branch-tracking settings (GELU-T both arms, W2-A T and T′) pass as causal forecasts, and Phase 1C
(registration 6433edc) replicated them prospectively.

## From Phase 2B (registered; registration 9239902, results 938ddd7)

3. **Arm 2b's R_ℓ-acc pass is narrow** (author-requested note, 2026-10-05). Arm 2b (output weights and bias ÷16) was
   registered with an EXPECTED FAIL on shifted-test accuracy at matched loss (power 0.18 from exploration). It passed,
   but narrowly: the smallest lower end of the 95% interval was +0.0038 (BCE = 0.1, reversed accuracy). Report it as a
   narrow pass against a registered expectation of failure, not as a robust effect. The primary arm (output weights
   only) passed R_ℓ-acc with a smallest lower end of +0.031.
