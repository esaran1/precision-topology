# Track 2 (final round): a prospective two-class width-2 prediction, with a gate

**This is a new predictor, designed from the diagnosis of earlier failures** (T2-3d, and the Track 2B results in
`track2_writer_inputs.md`).

**T2-3 stays FAIL, T2-3b and T2-3c stay UNRESOLVED, and T2-3d stays FAIL, whatever this shows.**

## Committed before the classifier was evaluated on any run

This file, `src/t2g.py` and `tests/test_t2g.py` were committed before the classifier was evaluated on any run. The
tests exercise every gate and criterion on constructed pass and fail cases.

**What had already been seen**, all disclosed:
- T2-3d's outcome;
- the Track 2B per-run analysis of T2-3b, T2-3c and T2-3d. It showed that early crossers cross during the initial
  relaxation, often into a basin that is already placed, and that crossing scales have a gap between 0.52 and 4.16;
- timing runs of the classifier on seeds 999,996–999,998. These seeds are not part of any arm.

### Setting

- a = 1.30, Δ = 0.4 (asymmetric windows).
- Matched initialisation, k = 0.04209.
- Each run uses its own training set, `asym_register.training_set(seed)`.

### Classifier

The classifier uses only the initial state and a pre-computation. It never uses training.

**A. Fixed-v relaxation.**
- Start from q₀ (`init_params(seed)`, with v scaled by k).
- Run Adam (lr 0.01, fresh state) on z = (α₁, β₁, α₂, β₂, b) for 2,000 steps, holding v at v₀.
- Check exact placement at every step.
- If the state is placed at any step, the class is **EARLY** and the predicted crossing is s₀ = ‖v₀‖₁.

**B. Otherwise, follow the basin it relaxed into.**
- The basin is z\*(v₀): damped Newton at fixed v₀, started from the relaxed state. It must be a minimum.
- Follow it by Newton continuation (predictor–corrector, steps ≤ 2%) along the ray v = s·v₀/s₀, from s₀ up to
  S_CUT = 1.0.
- If its exact gap G₊ turns positive at s_sw ≤ 1.0, the class is **EARLY** and the predicted crossing is s_sw.
- Otherwise the class is **LATE** and the predicted crossing is the run's width-1 own-sample threshold. This covers
  three cases: no basin, never placed up to 1.0, or the branch is lost or jumps. The threshold uses
  `width2_diagnosis.own_w1`, as in T2-3d.

### Ground truth, from observed training

- A run is EARLY iff it crossed and s_cross < 1.0. Otherwise it is LATE, and a non-crossing run is LATE.
- Runs placed at step 0 are excluded and counted.

## Step 1: gate (POST HOC, existing slowed runs T2-3b, T2-3c, T2-3d)

The gate covers the 239 runs that were not placed at step 0.

**PASS iff both hold:**
- classification accuracy ≥ 0.95;
- median |log(s_cross / s_w1,own)| ≤ 0.10 over the predicted-LATE crossing runs, with at least 20 such runs having a
  defined own threshold.

**Own thresholds** are computed only for predicted-LATE runs. For T2-3d the frozen values
(`asym_t23d_own_frozen.csv`) are reused.

**If the gate fails:** stop and report. Nothing is registered.

## Step 2: registered test on fresh seeds (only if the gate passes)

**Seeds:** 870,000–870,079. I verified this range is unused in `src/` and `results/`. The extension, used only if
fewer than 40 runs cross, is 870,080–870,159.

**Protocol:**
- T2-3c's φ₂ = 0.01778, budget 32,000.
- Placement is checked from step 0; if more than 20% of runs are placed at step 0, stop.
- Training uses `asym_t23c.train_one`.

**Before any training**, every seed's class and prediction (s₀ or s_sw for EARLY, the own threshold for LATE) is
frozen in `results/t2g/fresh_frozen.csv`, and its SHA-256 is committed.

**Criteria:**
- (i) classification accuracy ≥ 0.90;
- (ii) median |log(s_cross/s_pred)| ≤ 0.10 over crossing runs;
- (iii) the paired run-level bootstrap 95% interval (10,000 resamples) of
  mean(|log(s_cross/s_pred)| − |log(s_cross/0.44508)|) lies entirely below 0.

**Validity:** at least 40 crossings; otherwise UNRESOLVED.

**PASS iff (i), (ii) and (iii) all hold.**

## Competing predictions and falsifiers

**Competing prediction.** The early crossers are not determined by the initial basin: the fixed-v relaxation misses
the runs whose crossing depends on how v grows or rotates.
- That would show as accuracy below 0.95 at the gate, or below 0.90 in step 2.
- The late class could also be mispredicted, which would show as an error above 0.10.

**Falsifier.** Either criterion failing.
