# Registration: fixed-scale dynamics (Block 4) and the retention curve (Block 5)

**Written before any run of these blocks.** Date: 2026-09-23, 12:10 EDT (16:10 UTC).
Infrastructure: `src/fixed_scale.py`. E-2 stays failed whatever these show.

## Shared runs

- **180 fresh runs** at a = 1.30, seeds 200,000–200,179 (disjoint from every earlier run).
  Protocol: `fold1d.make_data(200, seed)`, float64, Adam lr 1e-2, initialisation
  `torch.manual_seed(seed); U(−1, 1)⁴`, budget 12,000 steps. This is the phase 2b protocol.
- **Full state saved** (θ, Adam `exp_avg`, `exp_avg_sq`, step count) at 60 log-spaced steps and at the
  first step whose oriented dense-grid gap is > 0 (first placement).
- **Scale variable**: `R = |w₂|Ĝ/2` with the certified `Ĝ(1.30) = 0.0861018`, measured in units of the
  certified `R_glob(1.30)` from Block 1c (`cond_certified_thresholds.csv`, midpoint of the certified
  `|w₂|` interval).
- **Levels**: R/R_glob ∈ {0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}. Paired design: every checkpoint goes
  through every level.
- **Intervention**: jointly rescale `(w₂, b₂)` by `k`. This preserves every decision at the moment of
  intervention. Then train `(w₁, b₁, b₂)` with `|w₂|` held fixed.
- **Optimiser-state variants, each with its own k = 1 control**:
  - **preserved**: the checkpoint's Adam moments and step count for `w₁, b₁, b₂`. The `b₂` moments
    are kept as saved (not rescaled); `w₂`'s moments are dropped because `w₂` is frozen.
  - **reset**: fresh Adam.
- **Recorded**:
  - the first update's ΔG, and G every 25 steps;
  - the first step with G > 0 (Block 4) or G ≤ 0 (Block 5);
  - gradient norms at the start and end;
  - the fraction of points with saturated logits (|σ(z) − y| < 1e-3) at the start;
  - the end distance in `(w₁, b₁)` to the certified conditional-branch argmin at the held scale.

## Block 4: before placement

- **Checkpoints**: G < 0, one per run per stratum of current R/R_glob:
  [0.2, 0.5), [0.5, 0.8), [0.8, 1.0), [1.0, 1.3). Within a stratum, the latest qualifying checkpoint of
  each run is used. No other selection.
- **Horizon**: 4,000 steps. Outcome: placed at the horizon (G > 0).
- **D1, equilibrium (primary)**, for the "preserved" variant:
  - the placement fraction at the horizon rises with R/R_glob;
  - its 50% point (linear interpolation over the levels, first crossing of 0.5) lies in **[0.9, 1.25]**;
  - no level's fraction is significantly below that of a lower level (paired exact McNemar test,
    p < 0.05, one-sided).
  - The same criteria are reported for "reset" as a secondary test.
- **D2, saturation (competing)**: the placement fraction at R/R_glob = 2.0 is significantly below the
  maximum over levels (paired exact McNemar, p < 0.05, one-sided).
- **Scoring**: D1 and D2 are scored independently. Both can pass (rise, then fall) and both can fail.
- **D3, instantaneous direction — exploratory only, no prediction**: the sign of the first ΔG against
  level, by starting stratum.

## Block 5: just after placement (retention)

- **Checkpoints**: the first-placement checkpoint of every run that places within 12,000 steps.
- **Sample size**, from a precision target fixed now: a 95% interval half-width ≤ 0.08 on retention at
  every level needs n ≥ 1.96²·0.25/0.08² = **150** placed checkpoints. 180 runs are expected to give
  about 160 (≈ 90% place by 12k at a = 1.30). If fewer than 150 place, 30 more seeds are added at a
  time, up to 270.
- **Horizon**: 12,000 steps, as in Block E.
- **Outcome**: **retained** = G > 0 at every 25-step check through the horizon, i.e. never lost. This
  matches Block E's "kept". Placed at the horizon is secondary.
- **Registered predictions**:
  - **Monotone**: retention is non-decreasing in R/R_glob. No level is significantly below a lower
    level (paired exact McNemar, p < 0.05, one-sided).
  - **Location**: the 50% point lies in **[0.9, 1.1]**.
  - **Reported without a prediction**: the 10–90% transition width, and its dependence on starting
    state (runs split at the median first-placement step; and at the median G at placement).
- Block E's pilot data (0/37 kept at 0.85, 33/37 at 1.15) were seen before this registration. The band
  [0.9, 1.1] is centred on the threshold itself, not fitted to them.

## Stop and report if

- the rescaling changes any decision at the moment of intervention (checked on every replay);
- any replay at k = 1 in the "preserved" variant does not reproduce the uninterrupted training
  trajectory for its first 25 steps (to 1e-10).
