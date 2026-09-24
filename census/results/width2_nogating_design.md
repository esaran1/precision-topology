# Design (APPROVED 2026-09-24 with additions, revision 1; nothing run): a training confirmation that output scale does not gate placement at width 2

**For the rebuttal revision** (not the submission). **Runs only after the direct small-scale check
(`scale_limits_prediction.md`, last section) confirms the registered verdict.** If that check finds an unplaced
minimiser at any scale, this test is not run as designed and the author is told first.

## Why

The registered small-scale test found no placement threshold for f_a at width 2 (`scale_limits_prediction.md`;
`math_note_v2.md` §10.1). That is a statement about the **conditional minimiser**. Whether **training** at a fixed
small output scale actually reaches placement is a separate question. This test asks it with the fixed-scale
machinery that tested gating at width 1 (Blocks 4 and 5, and the approved W4 machinery).

## Initialisation (approved by the author, 2026-09-24)

- The matched-initialisation rule of revision 2, k(a) = r₁(a)·s₂,glob(a)/0.97946, scales to the width-2 threshold,
  which does not exist.
- **Approved: absolute-scale matching.** v is scaled so that the median initial ‖v‖₁ equals the width-1 median
  initial |w₂|, 0.45899 (seeds 600,000–600,079, width-1 protocol, float32 draw as in training). That gives
  **k = 0.45899/0.97946 = 0.46862**, the same at both a.
- The hidden layer and the output bias are unchanged, as in revision 2. k is written to
  `width2_nogating_frozen.json` and its SHA-256 committed before any run.

## Checkpoints (definition approved, 2026-09-24)

- **Training**: seeds **630,000–630,079**, fresh and disjoint from every other seed set. Width 2 at a = 1.30 and
  1.50, matched initialisation (k above), the registered protocol (Adam lr 1e−2, float64, full batch,
  `fold1d.make_data(200, seed)`), budget 32,000 steps.
- **Placement is checked at every step** (G₊(φ_v), exact extrema), from step 0 (the initial state) up to the first
  step with G₊ > 0.
- **Pre-placement checkpoint**: any saved state with **G₊ ≤ 0**. Full states (parameters and Adam state) are saved
  at step 0 and at log-spaced steps {1, 2, 4, 8, …} while G₊ ≤ 0.
- **Used**: per run, the **latest pre-placement checkpoint**, i.e. the last saved state with G₊ ≤ 0 before the
  first placement.
  - **Step 0 is allowed.** A run that places within its first check interval (before its first saved step after
    0) uses its initial state, whose Adam state is empty. These runs are counted and reported separately.
  - A run with G₊ > 0 **at** step 0 has no pre-placement checkpoint. It is excluded and counted. **If more than 20%
    of runs at an a are excluded this way, stop and report**: the test would then sample a selected subset.
  - A run that never places within 32,000 steps uses its last saved state and is counted separately.

## Intervention and replay

- **Held scales**: ‖w₂‖₁ = 2R₂/Γ̂₂(a) at **R₂ ∈ {0.003, 0.01, 0.03, 0.1}**. These are the small scales the criterion
  is about. At a = 1.30, ‖w₂‖₁ ≈ 0.005–0.17, far below width 1's switch at |w₂| ≈ 5.
- **Rescale**: (w₂, b) → k′(w₂, b) with k′ = held/‖v‖₁, which preserves every decision at the checkpoint.
- **Replay**: (θ, v, b) train with ‖v‖₁ held by projection after every Adam step.
  - Optimiser state: "preserved" (primary) and "reset" (secondary).
  - The replay code is width2_train.replay; its validity checks are already tested on constructed cases.
- **Horizon rule (stated in full).** H is fixed from a pilot that cannot see the outcome.
  - Pilot: width 2, calibration seeds 500,000–500,019, both a, their latest pre-placement checkpoints (as above),
    replayed with the preserved optimiser state at held **R₂ = 0.5**, where placement is expected under both
    hypotheses. Each pilot replay runs 128,000 steps, with placement checked every 100 steps.
  - Per pilot replay, the **settling time** T is the first check after which the placement indicator never changes
    again up to 128,000. A replay whose indicator still changes in the last 12,800 steps (the final 10%) gets
    T = 128,000.
  - **H = the smallest of {16,000, 32,000, 64,000, 128,000} that is at least 4 × the 95th percentile of T**, pooled
    over both a (40 replays).
  - If 4 × p95(T) > 128,000, no H is chosen: **stop and report** before any scored replay.
  - H is written to `width2_nogating_frozen.json` with k, and hashed, before the scored replays.
  - Outcomes are recorded at H/16, H/4 and H. Only H is scored.
- **Outcome**: **placed**, meaning G₊(φ_v) > 0 at H, from exact extrema. This is placement, not sign-correctness,
  because the claim is about placement gating. Sign-correctness is reported beside it.

## Registered predictions (scored per a; 80 replays per held scale and variant)

| outcome | criterion (primary variant, "preserved") |
|---|---|
| **No gating (predicted)** | at **every** held scale the placed fraction at H is **≥ 0.90**, and its Clopper–Pearson 95% lower bound is ≥ 0.80 |
| **Gating at small scale (competing)** | at the **smallest** held scale (R₂ = 0.003) the placed fraction is **≤ 0.50**, and the fractions increase with scale (no significant decrease between levels; paired exact McNemar, one-sided p < 0.05) |
| neither | reported as such, with every fraction |

- **Positive control (the test can detect gating when it exists), scales confirmed in advance.** Width 1, seeds
  630,000–630,019, both a, Block 4's replay machinery (held |w₂|, preserved optimiser state, the same H), with
  latest pre-placement checkpoints defined as above.
  - **Held scales: R/R_glob(a) ∈ {0.1, 0.3, 0.5}**, i.e. |w₂| = 0.1, 0.3 and 0.5 × w₂,glob(a) from the certified
    brackets (midpoints). All three are below the certified threshold, where width 1 cannot place.
  - Expected placed fraction ≈ 0 at every level.
  - **A control placed fraction above 0.2 at R/R_glob = 0.1 means the machinery cannot detect gating: stop and
    report.**
- **tanh (secondary, descriptive)**: the same replays, with the stated expectation of no gating (the minimiser is
  placed at every scale). The conditional infimum is not attained, so α may grow without bound during replays; the
  growth is reported.
- **Per-unit realisation at every endpoint (H/16, H/4 and H), and whether it is the cancelling pair.**
  - Recorded per endpoint: each unit's weight share |vᵢ|/‖v‖₁; its α and β; the cancellation index
    |Σvᵢαᵢ|/Σ|vᵢαᵢ|; and the knockout class (single-unit if one unit alone keeps G₊ > 0, shared if neither alone
    does but both do, redundant if each alone does).
  - **Realises the cancelling pair** (rule fixed now): both weight shares in [0.4, 0.6], |α₁| and |α₂| within 10% of
    each other, cancellation index ≤ 0.1, and knockout class shared. The strict W0 rule (tolerances 1e−3,
    `width2_w0._is_pair`) is reported beside it.
  - Reported per a, held scale and variant: the fraction of endpoints realising the pair, among placed and among
    unplaced endpoints separately.
  - **Expected: the cancelling pair at small scales.** Descriptive, not scored.

## Validity checks and stop conditions (each already exercised on constructed cases in `tests/test_width2.py`)

- k = 1 reproduction against the independent true-freeze reference, to 1e−10 on 20 checkpoints (fail case: a
  gradient-only freeze).
- Decisions preserved at rescaling (fail case: k < 0).
- ‖v‖₁ drift ≤ 1e−12 at every step (fail case: no projection).
- Training determinism (fail case: a perturbed initialisation).
- Crossing and placement detection against an independent dense check.
- **Any failure stops.**

## Compute (from the measured per-step costs: training 0.15 ms per step, replay 0.09 ms per step)

| part | size | cost |
|---|---|---|
| training for checkpoints | 80 seeds × 2 a × ≤ 32k steps | ≤ 15 min |
| horizon pilot | 20 × 2 a × ≤ 128k steps | ≤ 10 min |
| replays | 80 × 2 a × 4 scales × 2 variants = 1,280, at H ≤ 64k | ≤ 2 CPU-hours |
| positive control | 20 × 3 levels × 2 variants at width 1 | minutes |
| tanh secondary | about the same as f_a | ≤ 2 CPU-hours |
| **total** | | **about 4.5 CPU-hours** (under 2 hours at three workers), < 0.5 GB per worker |

## Implementation note (2026-09-24, before any run)

- **Producer**: `src/width2_nogating.py`; tests: `tests/test_width2_nogating.py`.
- **Validity checks**: all ten pass on constructed pass and fail cases (`width2_nogating_validity.csv`, 3 s).
  - k = 1 reproduction: 20 checkpoints, maximum difference 2.2e−16.
  - Drift with projection: 6.9e−18; without projection (the fail case): 2.23.
  - The width-1 control replay equals `fixed_scale.replay` at a = 1.30 exactly.
- **Arithmetic of the primary criterion.** At n = 80 the 0.90 fraction is the binding condition. 72/80 already has a
  Clopper–Pearson lower bound of 0.812 ≥ 0.80, so the CP condition binds only if runs are excluded (n < 80). This is
  stated, not changed.
- **Storage.** Only each run's latest pre-placement state is persisted (`width2_nogating_parts/ck_*.pkl`), since it
  is the only state used. The count of states saved in memory is recorded per run.
- **Not run**: the pilot, training, control and replays wait for the direct small-scale check.
