# Registration: sample-specific conditional thresholds (own-seed R_glob)

**Written before any own-seed threshold is computed, before any 1d output is inspected, and before any
long-horizon replay has run.** Date: 2026-09-23 13:14 EDT. 1d is still running and its outputs have not been read. The
long-horizon replays (`fixed_scale_horizon_prediction.md`) are queued and not started. The per-seed
diagnosis of the 0.9 placements is queued and not started.

## Hypothesis

Each training run optimises its own 400-point training set (`fold1d.make_data(200, seed)`), so it has
its own conditional switch. That switch may differ from the population threshold computed on the 800
quadrature points.

**Confirmed before registering:**
- Block 4's replays train on each run's own training set: `fixed_scale.replay` uses
  `make_data(200, ck["seed"])`, the same data as the run.
- The free-training crossing runs (phase 2b, seeds 0–39) use `make_data(200, seed)`, whose data
  depend only on the seed.

## Computation (`src/own_threshold.py`)

- **Own-seed R_glob**: the conditional threshold of the run's own training set, found by the
  validated conditional search in profiled form.
  - The output bias is profiled exactly (Block 1b).
  - A dense grid of the profiled loss covers the certified domain (|w₁| ≤ W(s, a), b₁ ∈ [0, 2π),
    step 0.02), followed by local refinement of the global grid minimum.
  - The switch is the scale where the refined global minimiser's G changes sign, bracketed from the
    population value and bisected to width 0.01 in |w₂|.
  - R = |w₂|·Ĝ_cert(a)/2 with the population's certified Ĝ.
- **Validation**:
  - the certified branch and bound (Block 1c) at both ends of the own bracket, for 20 randomly
    chosen Block 4 seeds (seed 0 of `numpy.random.default_rng`);
  - 1d's certified per-seed thresholds, where seeds overlap (a = 1.30, seeds 0–39).
  - Reported as the fraction of brackets that agree.
- **Seeds**:
  - (i) every seed behind Block 4's 543 checkpoints (a = 1.30);
  - (ii) the free-training crossing runs at a = 1.30 and 1.50: phase 2b seeds 0–39.

## Registered predictions

- **S1 (per replay)**: at the 64,000-step horizon of the registered long-horizon replays ("preserved",
  levels 0.9–1.1), a replay is placed **if and only if its held R exceeds its own seed's R_glob**.
  - **Agreement ≥ 0.95** of replays.
  - It **beats the population rule** (placed iff level > 1), i.e. the own-seed rule's agreement exceeds
    the population rule's by **≥ 5 percentage points**.
- **S2 (placement curve)**: x₅₀ at 64,000 steps equals median(own R_glob)/R_glob_pop **within ±0.02**.
  - If own-seed thresholds sit above the population value, Q1 (horizon registration) fails and its
    competing outcome holds. S2 says by how much.
- **S3 (offset)**, at a = 1.30 and at a = 1.50, on the budget-32,000 free-training runs:
  - (a) the median own-seed threshold exceeds the population threshold, and
    (median own / population − 1) is **at least half the observed free-training offset** (≥ 4.8% at
    1.30, where the offset is 9.6%; ≥ 6.3% at 1.50, where it is 12.6%);
  - (b) the free-training crossing R **correlates positively with own-seed R_glob** (Spearman, one-sided
    p < 0.05);
  - (c) the offset measured against each run's own threshold, |median log(cross/own)|, is **smaller**
    than against the population threshold, |median log(cross/pop)|.
- **Competing outcome**: own-seed thresholds centred on the population value (median ratio within
  ±1%). Then heterogeneity explains the placement curve's width but not the offset.

S1, S2 and S3 (a–c at each a) are scored separately and reported with Q1/Q2 and the diagnosis of the
0.9 placements.
