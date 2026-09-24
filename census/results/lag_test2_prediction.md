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
