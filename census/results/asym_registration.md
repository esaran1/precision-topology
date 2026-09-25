# Registration: does a placement threshold survive at width 2 on asymmetric windows? (Track 2, Step 2)

**Written and committed before any registered evaluation.** Date: 2026-09-25, about 13:00 EDT. Code:
`src/asym_register.py` (tests: `tests/test_asym_register.py`, which exercise every decision rule below on constructed
pass and fail cases). Nothing at Δ = 0.4 has been evaluated.

What has been seen before this registration:
- the exploratory pilot at Δ = 0.8 (`asym_pilot_design.md`, `asym_pilot.csv`). It showed a width-2 switch at
  a = 1.30 in s ∈ (0.562, 1.0], and GO under its pre-committed rule;
- two timings: one scale at Δ = 0.8 (36 s), and one training run on calibration seed 10,000 at Δ = 0.4. Only its
  elapsed time was printed.

For the submission if complete and validated by the author's cutoff. Otherwise it is for the rebuttal.

## Setting

- f_a with a = 1.30, width 2: N(x) = v₁u(α₁x + β₁) + v₂u(α₂x + β₂) + b, and s = ‖v‖₁.
- Windows: I = [−0.8, 0.8] (class 0) and O = [−2.0, −1.2] ∪ [1.2, 2.0 + Δ] (class 1), with **Δ = 0.4**.
- **Population** (landscape): `asym_pilot.population(0.4)`. It has 400 inner points and 400 outer points split by
  length, 160 left and 240 right.
- **Training sets** (seed k): `asym_register.training_set(k)`, with 200 inner points ~ U(I) and 200 outer points,
  uniform on O. The side is chosen with probability proportional to its length. Values are rounded through float32,
  as in `fold1d.make_data`.
- **Placement**: the continuous-window G₊ = min_O φ − max_I φ on these windows, from exact extrema. Placed iff the
  enclosure's lower end is > 0; unplaced iff its upper end is ≤ 0; otherwise undecided.

## Why the criterion predicts a threshold (stated before measuring)

- On asymmetric windows the linear part enters the class-mean gap: Δμ ∋ m·(ṽ₁α₁ + ṽ₂α₂), with m = E_O x − E_I x.
  In the population m = 0.44.
- So the small-scale conditional minimiser uses the ramp instead of cancelling it, and a ramp is unplaced. At large
  scale the worst-case gap wins, and it is placed.
- Criterion (math note §10.1): gating iff the small-scale minimiser is unplaced. So a switch is predicted at width 2.
  On symmetric windows no switch exists (registered, WP-9).

## Predictions

**T2-1 (primary, landscape).** The validated conditional-minimiser search finds exactly one placement switch:
unplaced at every scanned scale below it and placed at every scanned scale above it.
- *Search.* Scale grid s = 10^(−1 + k/8), k = 0..16 (0.1 to 10). Each scale gets 1,000 batched-BFGS restarts
  (`width2_conditional.search_batch`, cap 3,000, seed 100 + k). The five lowest distinct candidates are polished by
  damped Newton. The constant predictor is included as a candidate.
  - Registered extensions, applied once: if every scale is placed, add s = 10^(−2 + k/8) for k = 0..7. If every
    scale is unplaced, add s = 10^(1 + k/8) for k = 1..8.
- *Bisection* in log s between the bracketing grid points, to a relative width of 5%, with the same per-scale
  procedure (seed 300 + step).
- *Validation* at both final bracket ends (s_lo unplaced, s_hi placed):
  - (i) restart ladder: 500/1,000/2,000/4,000 nested restarts (seed 500). The retained loss must be unchanged within
    1e−9, with the same status;
  - (ii) independent search: CMA-ES with v unconstrained, 40 starts (`independent_search`). It must not be below the
    retained loss by more than 1e−9;
  - (iii) audit: no polished candidate of the other status within 1e−9 of the retained loss.
- *Verdicts:*
  - PASS: switch, and the bracket is validated.
  - **FAIL**: every scale placed (the competing prediction, which is what symmetric windows give); or every scale
    unplaced after the extension; or more than one status change, or a placed-to-unplaced change.
  - UNRESOLVED: an undecided or tied scale in the scan, or validation fails at either end. This is a stop condition,
    reported as such.

**T2-2 (secondary, directional).** The switch lies below the pilot's lower bracket end: s_hi(Δ = 0.4) < 0.5623.
- Reason: a smaller asymmetry gives the ramp a smaller first-order reward, so the placed cancelling pair wins
  earlier.
- The competing outcome is s_hi ≥ 0.5623, which is a FAIL. Scored only if T2-1 passes.

**T2-3 (primary, training).** This is the width-2 W1 criterion (`width2_design.md`, `width2_train.score_w1`) with
its registered constants unchanged, written in s units. Ratios of s need no Γ̂₂.
- *Protocol:* width 2, float64, full batch, Adam lr 0.01, θ ~ U(−1, 1)⁷ (`width2_train.init_params`), budget 32,000
  steps. The crossing is the first step at which G₊ > 0, from exact extrema, checked every step
  (`width2_train.train`). Seeds 0–79.
  - Registered extension: if fewer than 40 of them cross, add seeds 80–159.
  - Fewer than 40 crossings after that is UNRESOLVED.
- *PASS* requires all three of:
  - at least 90% of crossing runs have s_cross ≥ s_hi;
  - the bootstrap 95% interval (10,000 resamples) of median(s_cross/s_hi) − 1 lies above 0;
  - **median(s_cross/s_lo) ≤ 1.25**, the stated upper bound on the median ratio.
- Otherwise FAIL. It is scored only if T2-1 passes; if T2-1 fails there is no threshold, and T2-3 is "not applicable".
- s_hi is used for the "above" clauses and s_lo for the upper bound, so the bracket's width counts against the
  prediction in both directions.

## Reporting

- Every scale, bisection step, validation end and training run is appended to `results/asym_parts/`. Scores go to
  `results/asym_scores.json` (`python -m src.asym_register score`).
- Nothing is re-run to change a verdict. A stop condition is reported, not repaired.
- A Δ = 0.8 or other-a result may be added only as labelled exploratory context.

## Resources

- One worker for the landscape stages and one for training, at nice 15. Each job is under 200 MB.
- Hard cutoff 23:00 EDT. Whatever is not complete and validated then is reported as not done.

## Amendment 1 (2026-09-25 12:59 EDT, author's instruction): T2-3 uses matched initialisation

**What happened before this amendment (disclosed).** At 12:54 EDT, right after this registration was committed
(53dce22), T2-3's training was launched with the **standard** initialisation, θ ~ U(−1, 1)⁷ (median ‖v‖₁ ≈ 0.98).
- All 80 runs (seeds 0–79) finished within about a minute.
- The author then pointed out that runs from standard initialisation may start above the switch. The Δ = 0.8
  pilot's switch is at s ≈ 0.56–1.0.
- The job's outputs were **moved aside unread**, with SHA-256 hashes committed:
  - `results/asym_sealed/train_uniform_init_seeds0-79.csv` b212646a7435a1bbbfc8b4af84876e2ee03e66a8bfbce7acdd4de1b4ee0be2a8
  - `results/asym_sealed/train_uniform_init_seeds0-79.log` bc595073185d999fd49273304a1642fafa9d7542f57ad6070d750849ceec7633
- No crossing value, crossing count or step has been read. The standard-initialisation T2-3 is **withdrawn,
  unscored**. Its sealed runs may be opened later only as a descriptive standard-initialisation arm with no criterion,
  as in the approved width-2 design, revision 2 (`width2_design.md`, "Secondary arm").

**T2-3 now uses the matched-initialisation rule of the approved width-2 design (revision 2), scaled to this geometry's
bracketed threshold.**
- Only the output weights are scaled: v₀ ← k·v₀. The hidden layer (α, β) and b are unchanged.
- k = r₁(1.30)·s_glob/0.97946, where:
  - r₁(1.30) = 0.09261 is the width-1 ratio of median initial |w₂| to |w₂|_glob;
  - 0.97946 is the width-2 standard median initial ‖v‖₁ over seeds 600,000–600,079;
  - s_glob is the **midpoint of T2-1's validated bracket at Δ = 0.4**, (s_lo + s_hi)/2.
- So the median matched initial ‖v‖₁/s_glob equals r₁(1.30) on these seeds.
- k is computed only after T2-1's bracket is validated. It is written to `results/asym_frozen.json`, and that file's
  SHA-256 is committed before any matched run. If T2-1 does not pass, T2-3 is not run.

**Seeds** are 600,000–600,079, the seeds on which 0.97946 was computed. The registered extension, if fewer than 40
cross, is 600,080–600,159. Training sets come from `asym_register.training_set(seed)` (asymmetric windows, as
registered).

**Placement checked from step 0** (as in the approved no-gating design). G₊ ≥ 0 is invariant to rescaling v, so
matching cannot move a hidden layer that is already placed.
- A run placed at step 0 has no crossing through the threshold. It is excluded from T2-3 and counted.
- **If more than 20% of runs are placed at step 0, stop and report**: the test would sample a selected subset.
- All else is unchanged: budget 32,000 steps, Adam lr 0.01, the crossing is the first step with G₊ > 0 (exact
  extrema), and the T2-3 criterion and constants (≥ 90% at or above s_hi; bootstrap CI above 0; median/s_lo ≤ 1.25;
  at least 40 crossings) stay as registered.

## Amendment 2 (2026-09-25 14:28 EDT): T2-3b, REGISTERED AFTER T2-3's FAILURE

**Status of T2-3: FAIL as registered. This does not change it.** When the author instructed amendment 1, only
matched initialisation was specified. The approved width-2 design's sweep-rate matching (Revision 3, primary W1 arm)
was left out. The exploratory timescale analysis found T2-3's crossings at a median ratio of 2.73, about 100× beyond the
width-1 Adam range. So the author registers a follow-up with sweep-rate matching. It is labelled as registered after
T2-3's failure, and T2-3's outcome was known when it was written.

**T2-3b.**
- **Same as T2-3:** a = 1.30, Δ = 0.4, the population threshold bracket [0.4371, 0.4532] (T2-1), the criteria
  (≥ 90% at or above s_hi; bootstrap CI of median(s_cross/s_hi) − 1 above 0; median(s_cross/s_lo) ≤ 1.25; at least 40
  crossings), matched initialisation (k = 0.04209), budget 32,000, and placement from step 0 with the 20% stop.
- **New: the output weights' learning-rate factor φ₂**, applied as v ← v_before + φ₂(v_after − v_before) after each
  Adam step.
- **φ₂ comes from Revision 3's steps-matching rule.** The median steps for ‖w₂‖₁ to reach s₂,glob (0.4451) at width 2
  must equal, within 2%, the median steps for |w₂| to reach |w₂|_glob(1.30) = 4.95625 at width 1 (width-1 protocol).
  - The pilot used calibration seeds 510,000–510,039 and recorded ‖w₂‖₁ and |w₂| only; no placement was evaluated.
  - Bisection in log φ₂ on [1e−4, 1].
  - Result: width-1 median 2,499 steps; **φ₂ = 0.009306**, width-2 median 2,530 steps (+1.2%). Record:
    `asym_t23b/pilot.json`.
  - Frozen in `asym_t23b_frozen.json` (SHA-256 ab3b39e110072851…).
- **Seeds:** 600,160–600,239. Registered extension if fewer than 40 cross: 600,240–600,319.
- **Validity check.** The achieved median timescale ratio at crossing must lie in the width-1 Adam range
  [0.0014, 0.023]. Otherwise **T2-3b is UNRESOLVED**, whatever the criteria give. The ratio is defined as in the
  exploratory analysis (`asym_posthoc2`), with growth measured under φ₂.
- Code: `src/asym_t23b.py`. Tests: `tests/test_asym_t23b.py` (the φ₂ bisection and the validity check, on pass
  and fail cases).
- **Competing outcome:** even with a matched sweep rate, crossings sit far above the width-2 threshold, and T2-3b
  fails.
