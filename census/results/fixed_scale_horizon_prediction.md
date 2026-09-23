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

## Stop report — 2026-09-23 17:49 EDT (validity check fired; nothing scored)

- **The registered 4,000-step reproduction check returned False** (`run_horizons`: "Block 4 reproduced at
  4,000 steps: False on 3258 rows").
- **Diagnosis**: the check compares in-memory G at 4,000 steps **exactly** (rtol = atol = 0) against Block
  4's G_end as read from `fixed_scale_block4.csv` with pandas' default CSV float parser. That parser is
  not round-trip exact: it differs from exact parsing on 7,959 of 8,688 stored values, by at most
  9.95e−17.
- **Re-check with exact (round-trip) parsing on both sides**, labelled as a diagnosis, not an amendment:
  on all 3,258 matched rows, placement at 4,000 steps equals Block 4's and **G at 4,000 steps is
  bit-identical** to Block 4's G_end. The replays reproduce Block 4 exactly; the check's
  implementation did not read the stored values exactly.
- **Status**: per the registration, Q1, Q2, S1 and S2 are not scored. The 0.9× endpoint characterisation
  is not run either, since it reads the same long-horizon outputs. Awaiting a decision on amending the
  check's implementation to exact parsing.
- If amended, this is the third implementation error in this registration family's validity checks,
  after amendments 1 and 2 of `fixed_scale_prediction.md`. None of the three concerned an outcome
  criterion.

## Amendment 1 — 2026-09-23 17:55 EDT (after the stop report above; before any Q1, Q2, S1 or S2 outcome is scored)

- **Fired check**: the registered 4,000-step reproduction check (stop report above, commit `856327b`).
- **Diagnosis**: it compared exact in-memory values with Block 4's stored values parsed by pandas'
  default CSV float parser. That parser is not round-trip exact: 7,959 of 8,688 stored values are off by
  at most 9.95e−17.
- **What changes, and only this**: how stored values are read. Both sides of the comparison are now read
  from their files with exact round-trip float parsing (`fixed_scale.check_horizons`).
- **What does not change**:
  - the tolerance stays at **zero**, for both placement and G;
  - it still compares **all 3,258 overlapping rows**;
  - **no prediction, level, horizon or criterion changes**: Q1, Q2 and the competing outcome, and S1 and
    S2 in `own_threshold_prediction.md`, are untouched.
- **Result of the amended check**: 3,258 rows compared, 0 placement mismatches, 0 values of G that are not
  bit-identical. The check passes (`fixed_scale_horizons_check.csv`).
- **Tests of the amended check** (`tests/test_registered_checks.py`):
  - it passes on identical awkward floats written and read back;
  - it fails on a one-ulp change in G;
  - it fails on a flipped placement;
  - the default parser demonstrably does not round-trip.
- **Scoring**: Q1 and Q2 use `fixed_scale.score_horizons`, and S1 and S2 use the scorer committed
  before the data (`a015ace`), unchanged.
