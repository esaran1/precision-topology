# Track 1A (items 1–2): the fold of the linear-dominant branch and the v3 runs, POST HOC

**POST HOC:** existing data only, no new training.
- Producer: `src/sb_fold.py` → `results/sb_fold/`, with summary `summary.json`.
- Tests: `tests/test_sb_fold.py`. The fold detector was tested on constructed saddle-nodes (1-D, and 2-D coupled) at
  three step sizes, with no false fold on a regular branch, before it was applied.
- Landscape: the v2/v3 weight-decayed fixed-scale objective (λ = 1e−4, the same 800 points), with registered
  s* = 3.5914.

## Verdict

- **Gate to 1B: FAIL.**
- The fold exists and is validated: **s_fold = 4.76769, and s_fold/s* = 1.3275.**
- The v3 late crossings are at median s_cross/s_fold = **2.17** (s_cross/s* = 2.86), far outside the 15% band.
- The runs also do not leave the linear-dominant branch's basin near the fold. They leave it earlier, at median
  0.66·s_fold for the late crossers, and only 1 of 16 does so within 15% of the fold.
- **Training never tracks any branch quasi-statically.** No training state at any step comes within 10⁻³ (function
  space) of any branch point.
- The late crossings are therefore not explained by the fold. They reflect the network's slab share lagging far
  behind the minimiser of the basin it is in.

## 1. Branch structure (continuation)

**Method.**
- The minimiser at a v2 retained point is taken to its **active units** only. Idle units (v = 0, W = c = 0) are exact
  spectators: their Hessian block is diagonal and positive (λ on W and c, and a positive η entry), and every coupling
  to the active block carries a factor vₖ = 0.
- One η is gauge-fixed to 1, which removes the exactly flat η-radial direction.
- b is joint rather than profiled. The joint Hessian is positive definite exactly when the profiled one is (Schur
  complement, since ∂²L/∂b² > 0).
- **Pseudo-arclength continuation** is run in (u, s):
  - tangent from the null vector of [F_u F_s];
  - Euler predictor and Newton corrector, with residual ≤ 1e−11;
  - a fold is where the tangent's s-component changes sign, refined by bisecting the step;
  - the reduced Hessian's smallest eigenvalue is recorded at every point;
  - step sizes h = 0.05, 0.025 and 0.0125.

**Four branches.** "Stable" means the reduced Hessian is positive definite.

| Branch | Active units | Stable range in s | ρ₂ on the stable part | Ends |
|---|---|---|---|---|
| L0: one pure-x₁ unit | 1 | 0.05 to beyond 37.5 | 0 | no fold |
| **M: linear-dominant.** A mirror pair (≈(9, ±4), c ≈ −1.8) plus an x₁ unit. | 3 | **1.15397 to 4.76769** | 0.032 → 0.280 | **fold at both ends** |
| S: slab branch. A slab pair (≈(2, ±8), c ≈ −3.1) plus an x₁ unit. | 3 | 1.91638 to 9.39777 | 0.044 → 0.722 | fold at both ends |
| S2: slab pair only | 2 | 0.3163 to beyond 44.6 | 0.531 → 0.784 | lower fold at 0.31593 |

**Global minimiser, by equal-loss scales** (bisection to 1e−10):
- L0 → M at **1.48955**;
- M → S at **3.58906**, which is 0.07% below the registered s* = 3.5914 (the geometric mean of v2's 0.35% bracket);
- S → S2 at **8.61658**.

**The "linear branch" is not a single branch.** The small-scale minimiser L0 does not continue into the mixed segment.
M is born at its own fold at s = 1.154 and becomes the global minimiser at 1.490. The fold asked about is therefore
**M's upper fold**. L0 has no fold at all up to s = 37.5.

## 2. The fold of M, and its validation

- **s_fold = 4.767689442106.**
  - It is the same to 6e−13 across h = 0.05, 0.025 and 0.0125.
  - At the fold, the tangent's s-component is ~1e−14 and the reduced Hessian's smallest eigenvalue is ~1e−17.
  - Along M, the smallest eigenvalue is about 3.3e−4 for s ≈ 2 to 4.5. It falls to 1.2e−4 at s = 4.76 and to 0 at the
    fold.
  - ρ₂ at the fold is 0.280. G₊ = −0.035, so M never separates every point.
- **s_fold/s* = 1.3275.** Against the exact M/S equal-loss scale the ratio is 1.3284.
- **Independent search near the fold** (full 16-coordinate BFGS, b profiled, 30 perturbed starts each):

  | Scale | Starts that land back on M (within 1e−3) | Where the rest go |
  |---|---|---|
  | s_fold·0.99 | 27 of 30 (median distance 4e−7) | S, ρ₂ 0.555 |
  | s_fold·0.999 | 18 of 30 | S |
  | s_fold·1.001 | 0 of 30 | all to S, ρ₂ 0.559 |
  | s_fold·1.01 | 0 of 30 | all to S, ρ₂ 0.562 |

  - Newton at fixed s from the fold point does not converge above the fold.
- **Full-space Hessian check.** Over all 16 coordinates, with the η-radial direction projected out, the smallest
  eigenvalue along M is 1.0e−4 at every checked s. That value is the idle unit (λ); the active block is positive.
  - Checked at s = 1.2, 2, 3, 3.5914, 4, 4.5, 4.7 and 4.76.
- **Eigenvalue path data:**
  - `branch_points.csv`: every continuation point, with loss, ρ₂, G₊ and eig_min;
  - `fold_constants.json`: λ_min on the Δs = 0.001 grid over the last 0.05 in s.
- **Fold constant** (math note §14: |m′c′| = ¼·|d(λ_min²)/ds| at the fold, from a quadratic fit in s_F − s over the
  last 0.05, checked over the last 0.025):

  | Preconditioner | Coordinates | abs(m′c′), window 0.05 | Window 0.025 |
  |---|---|---|---|
  | P = I | reduced | 5.00e−7 | 5.06e−7 |
  | P = I | training | 4.95e−7 | — |
  | Adam P | training | **3.57e−3** | 3.63e−3 |

  - The Adam P uses the median over runs of the block medians at the step where s first reaches s_fold:
    P_W = 85.7, P_c = 185.7, P_v = 19.4, P_b = 51.0.
  - v̂ for the output weights was reconstructed from the recorded parameters. The reconstruction reproduces the
    stored v̂ of W, c and b to 5e−14 relative.
  - With the median ṡ at the fold (0.0347 per step) and lr 0.01: Λ_F = 0.130, **ε_F = 5.6**, and r_F = 2.338·ε^{2/3}
    = 7.4.
  - ε_F ≫ 1 is outside the law's small-ε regime, so these are read-offs, not predictions.
  - The same law gives, for S's upper fold, s_F = 9.39777 and |m′c′| = 5.1e−8 with P = I.

## 3. The v3 runs along their trajectories

**Classifier.** It was defined before any leave scale was computed; I had already seen the v3 crossing scales and
the label sequences of 6 runs.
- At every recorded step (all steps, all 40 runs), the run's state is mapped to the landscape parametrisation: signs
  absorbed, η = √(|v|/‖v‖₁).
- **Basin:** a full 16-coordinate local minimisation at the run's own s (BFGS, gradient ≤ 1e−8). The result is
  labelled with the branch (L0, M, S, S2) whose point at that exact s lies within 10⁻³ in function-space RMS (unit-ℓ₁
  φ over the 800 points, invariant to permutations). Otherwise it is "other".
- **On-branch:** the same test applied to the raw state, without minimisation.
- **Leaving M:** the scale at the first step after the last step in the M basin.

**Results:**

| Group | n | In M's basin at some step | Median leave/s_fold (range) | Leave within 15% of s_fold | Median s_cross/s_fold | Median ρ₂ of the run at s_fold | Basin at s_fold |
|---|---|---|---|---|---|---|---|
| late crossers | 16 | 16 | **0.66** (0.50–0.88) | 1/16 | **2.17** | 0.13 | S in 14, L0 1, other 1 |
| early crossers | 6 | 4 | 0.77 (0.55–0.82) | 0/4 | 0.14 | 0.11 | S in 5, L0 1 |
| non-crossers | 18 | 18 | 0.78 (0.57–0.99) | 6/18 | — | 0.02 | S 11, other 4, L0 2, S2 1 |
| all | 40 | 38 | 0.71 | 7/38 | 2.15 | 0.06 | S in 30 |

- **No run is ever on a branch.** The raw state is within 10⁻³ of a branch point at 0% of steps for crossers and at
  most 2.4% of steps for any run. The median minimum distance to any branch point over a run is 0.18.
- The basin sequence is L0 → M from about s = 1.8–2.2, with flicker between M and "other". It then goes to S from
  about s = 3.1–4.2, and to S2 later.
- At s_fold, 30 of 40 runs are in S's basin, whose minimiser has ρ₂ = 0.56, while the runs' own ρ₂ is 0.06 (median).
- The late crossers reach ρ₂ = q = 0.391 only at s ≈ 9–11 (2.17·s_fold). By then they are in the S or S2 basin, with
  minimiser ρ₂ 0.67–0.74.
- The early crossers are transients at s < 1 on the way down from a high initial ρ₂.
- The non-crossers end with ρ₂ below q, still in the S/S2 basins.

## What the branch structure shows instead

- The landscape has a clean sequence of first-order switches: L0 → M (1.490), M → S (3.589), S → S2 (8.617).
- M has a validated fold at 1.3275·s*.
- The v3 runs, at χ ≈ 7, are far from quasi-static.
  - Their states sit about 0.2 away, in function space, from every branch.
  - They move from M's basin to S's basin **before** M's fold, mostly between s* and the fold.
  - Their slab share then relaxes slowly toward the basin minimiser and reaches q only at about 2.2·s_fold.
- The late crossing scale is therefore a **relaxation lag in ρ₂ within the slab basin**, not a delayed exit from a
  metastable linear branch at its fold.
- The fold-delay law's small-ε condition fails by a wide margin (ε_F ≈ 5.6), so the law is not expected to apply to
  these runs.
