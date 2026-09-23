# Registration: equilibrium against relaxation at fixed scale (Block 4 extension)

**Written before any replay beyond the registered 4,000-step horizon.** Date: 2026-09-23 13:02 EDT.
Builds on `fixed_scale_prediction.md` (Block 4: D1 passed, 50% point 1.049 at 4,000 steps; D2 failed).

## Design

- **Replays**: the same 543 before-placement checkpoints and the same deterministic replay
  (`fixed_scale.replay`: rescale (w₂, b₂), hold |w₂|, train (w₁, b₁, b₂), a = 1.30, base window,
  Adam lr 1e-2).
- **Optimiser state**: both variants; "preserved" is primary, "reset" is reported alongside.
- **Levels**: R/R_glob ∈ {0.9, 0.95, 1.0, 1.05, 1.1}. R_glob is Block 1c's certified midpoint.
- **Horizons**: **4,000, 16,000 and 64,000 steps**, i.e. 1×, 4× and 16×, recorded within one replay.
  The replay is deterministic, so the 4,000-step outcome must reproduce Block 4's exactly; that is a
  validity check. **Stop and report if it doesn't.**
- **Outcome**: placed at the horizon (G > 0).
- **50% point x₅₀(H)**: linear interpolation over the five levels, first crossing of 0.5.

## Registered predictions

- **Q1 (equilibrium is the conditional threshold)**: x₅₀ moves toward 1.0 as the horizon grows.
  Precisely, all three must hold:
  - x₅₀(4k) ≥ x₅₀(16k) ≥ x₅₀(64k);
  - |x₅₀(64k) − 1| < |x₅₀(4k) − 1|;
  - **x₅₀(64k) ∈ [0.98, 1.02]**.
- **Q2 (no placement below the threshold at long horizons)**: the placed fraction at 0.9 and at 0.95 is
  non-increasing in horizon, and at 64,000 steps is **0 at level 0.9 and ≤ 0.01 at level 0.95**.
- **Competing outcome (the fixed-scale switch sits above the conditional one)**: x₅₀(64k) ∈
  [1.03, 1.07]. If this occurs, Q1 fails and the paper states that the fixed-scale switch sits
  about 5% above the conditional threshold at this setting.

Q1 and Q2 are scored independently.
- An x₅₀(64k) between 1.02 and 1.03 fails Q1 without meeting the competing outcome, and is reported
  as intermediate.
- **Per-seed variation of the threshold** (the 1d objective check) may itself place some runs below
  the population threshold at equilibrium. If Q2 fails, the exploratory diagnosis below decides
  whether that is the cause. The verdict stands either way.

## Exploratory, no prediction (labelled)

- **The 7 checkpoints placed at 0.9 after 4,000 steps**:
  - whether they stay placed at 64,000 steps;
  - their position relative to the certified branch at that scale, modulo symmetries;
  - their own per-seed certified threshold (the certified procedure applied to each run's
    400-point training set).

  The diagnosis will be one of: transient overshoot, a second basin, per-seed threshold, or bracket
  resolution.
- **Critical slowing**:
  - the time to first placement against (R/R_glob − 1) above the threshold, from the replays;
  - integrating that relaxation rate along free-training |w₂| trajectories, compared with the
    observed free-training offset.

  This is post hoc, since the free-training crossings have been seen. If it works, a prospective
  design is written and **not run** before review.
