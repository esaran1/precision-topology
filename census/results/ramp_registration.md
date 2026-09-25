# Track 1B registration: the ramp experiment (R1–R5), 2026-09-25

Committed before any registered ramp or free-training run. Code: `src/ramp.py`; tests: `tests/test_ramp.py`
(constructed pass and fail cases for every rule, including a negative κ, an unresolved cell count and a warm-up STOP).

## Question

Track 1A derived the lag law r = κ_k(a)·χ and compared it with existing runs, with no refit. Those runs were not designed
to test it: ṡ was not controlled, and χ spanned only what free training produced. Here the output scale is forced to grow
at a chosen rate while the hidden coordinates (w₁, b₁) and the output bias b₂ train normally. The law then predicts the
crossing residual from the imposed rate, with no free parameter.

## Protocol

- Width 1, f_a(t) = t + a sin t, a ∈ {1.30, 1.50}. Each seed has its own training sample of 400 points, 200 per class (`fold1d.make_data(200, seed)`; the registered text said "200-point", corrected 2026-09-25 before any crossing was read; the code is unchanged).
- Registered seeds: 860,000–860,039 (the same 40 in every cell).
- w₂ = +s(t), set every step. s = s₀ = 0.5·s\* during a warm-up of 4,000 steps, then s = s₀·exp(γ(t − 4000)) up to 3·s\*.
  Here s\* is the population switch (certified glob-bracket midpoint).
- (w₁, b₁, b₂) train with full batch: Adam at lr 0.01 or SGD at lr 0.3.
- **Initialisation (a design choice fixed by the pilot below).** Every seed starts on the population branch point θ\*(s₀), obtained by Newton continuation from θ\*(s\*) (`ramp.branch_init`), on winding copy k.
  - A copy (w₁, b₁ + 2πk, b₂ − 2πk·s₀) has the same logits, loss and Hessian.
  - The warm-up relaxes each seed to its own-sample branch.
- Crossing: the first step with G > 0 on the dense windows (4,001 inner and 2 × 2,000 outer points).
  - Observed residual: r = s_cross / s_own − 1.
  - s_own is the seed's own-sample threshold (the frozen global conditional search; `own_threshold.own_threshold`; bracket width 0.12%).
- **Settings** (a, winding k, optimiser), 6 γ cells each:
  - (1.30, −1) and (1.50, 0): the windings every existing run sits on. These are primary (R1–R3).
  - (1.30, 0): the other winding copy, where κ is negative, so the law predicts crossing *before* s_own. This is the sign test (R5).
- The seeds are trained in one vectorised batch.
  - SGD batch and single-seed runs agree exactly (≤ 1e-12) at every step.
  - Adam agrees exactly through 1,000 warm-up steps. After that the branch start leaves gradients at roundoff scale, and m/√v̂ amplifies one-ulp reduction differences to about 1e-5 in the parameters (tested; recorded here). That is about 1e-5 relative in s_cross, well below the own-threshold resolution.

## γ grids (`results/ramp/design.csv`, fixed before any registered run)

- **SGD:** χ = γ/(η λ_min(H)) is known a priori, so γ = target·η·λ_min(H)/|κ_k,SGD|. There are 6 log-spaced targets for |r| from 0.001 to 0.1. The slowest cell has 302,036 ramp steps.
- **Adam:** v̂ adapts to the ramp, so χ is not known a priori.
  - γ was calibrated on 8 pilot seeds (869,000–869,007, outside every registered range): an 11-point γ sweep, the median *predicted* |r| per γ (no observed residual was computed), and log-log interpolation. `results/ramp/design_adam_sweep.csv` holds the sweep.
  - Targets are 6 log-spaced values from 0.0005 to 0.006.
  - Pilot finding (disclosed): Adam runs stop crossing at all once γ ≳ 3e-3 (a = 1.30) or ≳ 1e-2 (a = 1.50). Adam's update is capped near η per coordinate per step, so it cannot follow faster ramps. That regime is outside the linear law, so Adam's targets stop where every pilot run crossed.
- Pilot history (disclosed): a first pilot from the standard random initialisation had few crossings in fast cells (0/8 at a = 1.30). Those runs were off the tracked branch. This is why the branch start is used. The pilot also showed that from the branch point, a = 1.30 sits on winding 0, which led to the explicit winding settings. `results/ramp/pilot_design*.csv` record predictions and crossing counts only.

## Prediction per run (no free parameter)

r_pred = κ_k·χ, with χ = γ/(η λ_min(P^{1/2}HP^{1/2})).

- H, θ\*′ and ∇G are taken at the population switch (`results/lag_law/kappa.csv`, committed b6ae433).
- k = round((b₁ − b₁\*)/2π) is measured at the run's crossing.
- P = I for SGD, so χ is fully a priori.
- For Adam, P = diag(1/(√v̂ + 1e-8)) is the run's own at the crossing. Adam's χ therefore uses a measured quantity; SGD's does not.

## Rules (per setting; `ramp.score_setting`)

A γ cell is resolved if ≥ 30 of its 40 runs cross. A setting needs ≥ 4 resolved cells, or it is UNRESOLVED. A setting STOPs if more than 20% of its runs place during warm-up.

- **R1 (magnitude):** the through-origin slope of observed on predicted r, over all crossing runs in resolved cells, lies in [0.7, 1.3], i.e. within ±30% of the derived κ. (The slope weights the fast cells most; per-cell medians are reported beside it.)
- **R2 (monotone in rate):** sign(κ)·Spearman(γ, cell median r) ≥ 0.9 over the resolved cells.
- **R3 (the lag vanishes when the ramp is slow):** |median r| in the slowest resolved cell < 0.005.
- **R5 (winding sign test):** R1–R3 applied to setting (1.30, k = 0), where κ < 0.
- **R4 (free training; `ramp.score_r4`):**
  - Setup: standard free Adam training (all four parameters, U(−1, 1) initialisation) at η = 0.01, 0.005 and 0.0025, with budget 32,000·(0.01/η).
  - Seeds: 860,100–860,139, each with its own threshold (`own_free_frozen.csv`).
  - Rule: at each a, the median residual at every η lies within max(0.01, 0.25·|median at η = 0.01|) of the η = 0.01 median. Each η needs ≥ 30 crossings, or it is UNRESOLVED.
  - The law predicts η-invariance, because ṡ and ηλ both scale with η.

## Competing predictions and falsifiers

- **Lag law holds:** R1–R3 pass; R5 passes (early crossing, |r| growing with γ); R4 passes.
- **The residual is a static offset, not a lag** (for example sample or branch mismatch): R3 fails (r does not vanish at slow γ) and R2 fails (r flat in γ).
- **The linear law has the right shape but the wrong magnitude:** R2 and R3 pass, R1 fails. This is reported as "within a factor of two" or worse, from the slope.
- **The winding dependence is wrong:** R5 fails (the k = 0 copy at a = 1.30 crosses late).
- **The residual is a finite-step artifact:** R4 fails, with the residual shrinking with η.
- **Adam's frozen-P linearisation fails because P is non-stationary:** R1 fails for Adam but passes for SGD. It is reported as such; it does not cast doubt on the SGD result.

## Own thresholds

They are a deterministic function of each seed's training sample (the frozen search). They are computed in parallel with the runs, written to `results/ramp/own_frozen.csv` and `own_free_frozen.csv`, and hashed (`.sha256`) before any crossing is read. `ramp.score` refuses to run on a hash mismatch.
