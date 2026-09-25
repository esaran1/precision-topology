# Registration: a second prospective test of the timescale account (Task B, TS-1 and TS-2)

**Written and committed before any Adam run on these training sets.** Date: 2026-09-25, about 15:10 EDT. Code:
`src/ts_test.py`. Tests: `tests/test_ts_test.py`, which exercise TS-1, TS-2 and UNRESOLVED on constructed cases.

**Why.** The timescale account has one prospective test (Track 4's SGD extension, PASS at both a). This is a second
one:
- at activation values the Adam fit never saw (a = 1.45 and 1.60; the fit used a = 1.30 and 1.50);
- on training sets never used anywhere: `fold1d.make_data(200, seed)`, **seeds 830,000–830,079**. Every result file
  was checked for these seeds, and none contains them.

(The author chose option (b). The originally proposed seeds 0–49 were rejected because phase 2b had already recorded
Adam crossings for seeds 0–39 at these a.)

**Frozen before any Adam run.**
- **The relationship:** `residual_timescale_fit.json` (committed in 1df48d1). Residual = α + β·ratio, α = 0.0157178,
  β = 2.65842, fitted post hoc on the 384 deconfounded lag-test Adam runs (a = 1.30, 1.50). This is the same file the
  SGD extension used.
- **Own thresholds:** own global thresholds for all 160 (seed, a) sets, by the grid method used for the phase-2b own
  thresholds (`own_threshold.own_threshold`, started from the population bracket midpoint, bisected to 0.01 in |w₂|).
  They are written to `ts_test_own_frozen.csv`, and its SHA-256 is committed before `ts_test train` runs. The
  population thresholds are the certified glob bracket midpoints: 2.89375 at a = 1.45 and 2.00625 at a = 1.60.

**Protocol.**
- Adam, the standard protocol (as `phase2b_ordering.run`): U(−1, 1)⁴ drawn in float32 and cast, lr 0.01, full batch,
  budget 32,000 steps.
- Every-step crossing detection with `phase2b_ordering.state`.
- The timescale ratio at the crossing is defined exactly as in Track 3.

**Predictions (scored separately at a = 1.45 and a = 1.60, on the runs that cross).**
- **TS-1.** The median residual, w₂,cross/w₂,own − 1, equals median(α + β·ratioᵢ) over the crossing runs, within
  max(0.01, 0.25·|pred|) (the SGD extension's tolerance). UNRESOLVED if fewer than 30 runs have a finite, positive
  ratio.
- **TS-2.** Each run's own threshold predicts its crossing better than the population threshold. PASS iff the paired
  run-level bootstrap 95% interval (10,000 resamples) of mean(|log(cross/own)| − |log(cross/pop)|) lies entirely
  below 0.
- **Coverage.** UNRESOLVED at an a with fewer than 30 crossings.
- **Sensitivity, registered now:** each non-crossing run is re-scored as crossing at (or above) its final |w₂|. TS-1
  uses the crossers' ratios; TS-2 uses all runs. This is reported beside the registered verdicts and does not replace
  them.

**Competing outcome.** The residual at these a is not predicted by the Adam-fitted timescale relationship (TS-1
fails), or the own threshold does not beat the population (TS-2 fails).
