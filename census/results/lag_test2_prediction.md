# Registration: deconfounded adiabatic-lag test (L1′, L2′)

**Registered 2026-09-23 23:25 EDT, before any continuation at φ ≠ 1.** Producer: `src/lag_test2.py`. Checks:
`tests/test_registered_checks.py` (the `lag2` tests; all pass).

## Why

- The first lag test (`lag_test_prediction.md`) passed L1 and failed L2.
- But slowing w₂ from the start also lengthened the time on the constant-predictor plateau, a possible
  confound.
- Here φ is switched only **after** the run has left the plateau for the last time before its crossing.

## Design

- **Runs**: the size test's n = 6,400 training sets, a ∈ {1.30, 1.50}, seeds 300,000–300,049, under the
  standard protocol (φ = 1). The lag test reproduced these bit for bit.
- **Switch point t\***: the first step after the run's **last** plateau step before its crossing.
  - Plateau: |w₁| < 0.05 **or** |w₂| < 0.05, the combined definition.
  - A run never on the plateau has t\* = 1.
  - **Exclusion rule** (counted and reported): R(t\*) > 0.8·R_own (it left the plateau too close to its
    own threshold), or t\* ≥ its crossing step.
- **Checkpoint**: the full parameter and Adam state after step t\* at φ = 1, regenerated deterministically
  (two passes: find t\*, then rerun to it) and saved (`results/lag_test2_states/`, not committed).
- **Continuation**: from the restored checkpoint, w₂'s learning rate × φ ∈ {0.25, 0.5, 1, 2}
  (`lag_test.scale_w2_step`), everything else unchanged, for ⌈(32,000 − t\*)/φ⌉ further steps.
- **Residual(a, φ)** = median over crossing runs of R_cross/R_own − 1, against the **global own threshold**
  (`sample_size_own.csv`, n = 6,400). Bootstrap 95% intervals: 10,000 resamples, seed 0.
- **Crossings**: at least 40 per cell. A cell with fewer is reported as insufficient and not extended.

## Registered predictions (the same criteria as L1 and L2, scored separately at each a)

- **L1′ (primary)**: all three must hold:
  - residual(0.25) < residual(0.5) < residual(1) < residual(2);
  - the 95% interval of residual(1) − residual(0.25) lies above 0;
  - residual(0.25) ≤ ½·residual(1).
- **L2′ (secondary)**: residual(0.5)/residual(1) ∈ [0.30, 0.70] and residual(0.25)/residual(1) ∈
  [0.05, 0.45].
- **Competing**: the 95% interval of residual(2) − residual(0.25) contains 0.
- **Failed ordering steps** with overlapping intervals are recorded as failures, with a separate note that
  the values are indistinguishable.

## Reported beside the verdicts (no criterion)

- **Plateau re-entry after the switch**, per φ: the median number of steps with |w₁| < 0.05 or |w₂| < 0.05
  after t\*, and the fraction of runs with any. It should be near zero.
- **The mirror share at the crossing**, per φ.
- **The direct lag diagnostic**: the median distance at the crossing from the hidden parameters to the
  branch minimiser at the current scale (`lag_test._diag_job`).

## Validity and stop conditions (exercised on constructed pass and fail cases before registering)

- **φ = 1 continuations must reproduce the original runs bit for bit**: crossing step and |w₂| at the
  crossing, exact round-trip parsing on both sides, every continued run
  (`lag_test2.check_continuation`). **Stop if not.**
  - Constructed cases: it passes on identical data, and fails on a one-ulp change in |w₂| and on an
    off-by-one crossing step.
  - **Real runs, before registration**: a = 1.30 seed 300,000 (t\* = 16) and a = 1.50 seed 300,001
    (t\* = 1), φ = 1 only. The restored-state continuation reproduces the stored crossings bit for bit.
    Those two checkpoints are regenerated identically by the registered run.
- **The t\* rule**: tested for never on the plateau, the last exit, and both exclusions.
- **The scorer**: a proportional lag passes L1′ and L2′; a flat residual fails both and meets the competing
  outcome; the indistinguishable note appears on overlapping failed steps.

## Compute (estimated before registering)

- **Checkpoints**: about 1–2 s per run, 100 runs.
- **Continuations**: under 1 s at φ = 1 and up to about 4× that at φ = 0.25; 400 continuations.
- **Diagnostics**: about 8 s per crossing, about 400.
- **Total**: about 1.3 CPU-hours, about 30 minutes at 3 workers, under 0.5 GB per worker.

## Order

It runs after the prospective own-seed test and before any width-2 runs. The sequence is `checkpoints`,
`continue` (which includes the φ = 1 check), `diagnostics`, then `score`.

## Amendment 1 — 2026-09-23 23:31 EDT (before any continuation at φ ≠ 1)

**The primary switch rule is replaced; the t\* rule is kept as a secondary analysis.**

- **Primary switch rule**: switch at the first step where **R ≥ 0.7 × the run's own threshold**, provided the
  run has already left the plateau for the last time before its crossing. Plateau is the combined
  definition, as before.
  - **Exclusions** (counted): runs whose last plateau visit comes at or after that point, and runs that
    cross before reaching it.
  - **Branch commitment**: the switch must also lie past the branch-commitment level observed on the base
    window (R > 0.10 at a = 1.30, R > 0.15 at a = 1.50).
    - Checked on the own thresholds (n = 6,400): 0.7 × R_own exceeds 0.10 for all 50 runs at a = 1.30. At
      a = 1.50 it is at or below 0.15 for **2 of 50 runs** (min 0.1500).
    - For those two, the switch level is **the larger of the two, 0.15**. This is recorded per run
      (`used_commit_level`).
  - **Why**: every arm then shares identical training up to a point where the plateau phase is over and
    the branch is committed. So φ can only affect the approach to the threshold.
- **Verdict**: **L1′ and L2′ as registered are scored on the primary rule over all included runs.**
- **Reported beside it**, with the same L1′ and L2′ criteria, no verdict:
  - the **t\* rule** (secondary);
  - both rules **split by stratum**: runs with a plateau visit before their crossing, and runs never on
    the plateau.
- **Unchanged**: the criteria, the bit-for-bit φ = 1 check (now applied to every rule's φ = 1
  continuations), and the diagnostics (plateau re-entry, mirror share, lag diagnostic), now per rule.
- **Check tests re-run with the new rule, all pass (24)**:
  - the 0.7 × R_own switch;
  - the commitment level taking over;
  - **a run that re-enters the plateau after the switch point** (excluded);
  - a run that crosses before the switch level (excluded);
  - a run never on the plateau;
  - the scorer with both rules and the strata.
  - Real runs, φ = 1 only (a = 1.30 seed 300,000 and a = 1.50 seed 300,001, the latter at the 0.15
    level): the continuations under both rules reproduce the stored crossings bit for bit.
- **Compute**, about doubled by the second rule's continuations and diagnostics: about 2.6 CPU-hours,
  about 1 hour at 3 workers.
