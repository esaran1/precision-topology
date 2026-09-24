# Design (for review; nothing run): a training confirmation that output scale does not gate placement at width 2

## Why

The registered small-scale test found no placement threshold for f_a at width 2 (`scale_limits_prediction.md`;
`math_note_v2.md` §10.1). That is a statement about the **conditional minimiser**. Whether **training** at a fixed
small output scale actually reaches placement is a separate question. This test asks it with the fixed-scale
machinery that tested gating at width 1 (Blocks 4 and 5, and the approved W4 machinery).

## Initialisation (needs the author's decision)

- The approved matched-initialisation rule, k(a) = r₁(a)·s₂,glob(a)/0.97946, scales to the width-2 threshold. That
  threshold does not exist.
- **Proposed**: match the **absolute** initial output scale to width 1. Scale v so that the median initial ‖v‖₁
  equals the width-1 median initial |w₂|, 0.45899 (seeds 600,000–600,079, width-1 protocol). That gives
  **k = 0.45899/0.97946 = 0.46862**.
- The hidden layer and the output bias are unchanged, as in revision 2. k is frozen, with a hash, before any run.

## Checkpoints

- **Training**: seeds **630,000–630,079**, fresh and disjoint from every other seed set. Width 2 at a = 1.30 and
  1.50, matched initialisation, the registered protocol (Adam lr 1e−2, float64, full batch,
  `fold1d.make_data(200, seed)`), budget 32,000 steps.
- **Saved**: full states (parameters and Adam state) at log-spaced steps while G₊(φ_v) ≤ 0.
- **Used**: per run, the **latest pre-placement checkpoint** (G₊ ≤ 0, exact extrema).
- A run already placed at initialisation contributes its initial state, **unplaced by construction**: v is rescaled
  as below, then G₊ is checked ≤ 0, and the run is excluded and counted if not.

## Intervention and replay

- **Held scales**: ‖w₂‖₁ = 2R₂/Γ̂₂(a) at **R₂ ∈ {0.003, 0.01, 0.03, 0.1}**. These are the small scales the criterion
  is about. At a = 1.30, ‖w₂‖₁ ≈ 0.005–0.17, far below width 1's switch at |w₂| ≈ 5.
- **Rescale**: (w₂, b) → k′(w₂, b) with k′ = held/‖v‖₁, which preserves every decision at the checkpoint.
- **Replay**: (θ, v, b) train with ‖v‖₁ held by projection after every Adam step.
  - Optimiser state: "preserved" (primary) and "reset" (secondary).
  - The replay code is width2_train.replay; its validity checks are already tested on constructed cases.
- **Horizon H** is chosen from pilot convergence times that do not test the outcome.
  - The pilot uses calibration seeds 500,000–500,019, held at **R₂ = 0.5**, where placement is expected under both
    hypotheses. It measures the step after which sign-correctness no longer changes.
  - H is the smallest of {16k, 32k, 64k, 128k} steps that is at least 4× the 95th percentile of that time.
  - Outcomes are recorded at H/16, H/4 and H.
- **Outcome**: **placed**, meaning G₊(φ_v) > 0 at H, from exact extrema. This is placement, not sign-correctness,
  because the claim is about placement gating. Sign-correctness is reported beside it.

## Registered predictions (scored per a; 80 replays per held scale and variant)

| outcome | criterion (primary variant, "preserved") |
|---|---|
| **No gating (predicted)** | at **every** held scale the placed fraction at H is **≥ 0.90**, and its Clopper–Pearson 95% lower bound is ≥ 0.80 |
| **Gating at small scale (competing)** | at the **smallest** held scale (R₂ = 0.003) the placed fraction is **≤ 0.50**, and the fractions increase with scale (no significant decrease between levels; paired exact McNemar, one-sided p < 0.05) |
| neither | reported as such, with every fraction |

- **Positive control (the test can detect gating when it exists).** Width 1, seeds 630,000–630,019, Block 4's
  replay at held R/R_glob ∈ {0.1, 0.3, 0.5}. Expected placed fraction ≈ 0 (below threshold). A control placed
  fraction above 0.2 at R/R_glob = 0.1 means the machinery cannot detect gating: stop and report.
- **tanh (secondary, descriptive)**: the same replays, with the stated expectation of no gating (the minimiser is
  placed at every scale). The conditional infimum is not attained, so α may grow without bound during replays; the
  growth is reported.
- **Per-unit realisation at every endpoint**: weight share, the cancellation index |Σvᵢαᵢ|/Σ|vᵢαᵢ|, and the
  knockout class (single-unit, shared or redundant). **Expected: shared placement (the cancelling pair)** at small
  scales. This is descriptive.

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
