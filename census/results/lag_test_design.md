# Design for review: is the free-training residual adiabatic lag of w₂?

**Design only: not registered, not run.** On approval it becomes the registration, committed before any
run at φ ≠ 1. Producer: `src/lag_test.py`. Check tests: `tests/test_registered_checks.py`, the `lag` tests.

## Question

- The size test showed that the free-training offset has two parts:
  - a finite-sample part, which vanishes with n (the own excess is about 1% at n = 6,400);
  - a residual, which persists at about 3% (a = 1.30) and 6% (1.50) against the own threshold at
    n = 6,400.
- **Hypothesis (adiabatic lag)**: during free training, |w₂| is still growing when the placement switch
  becomes available. The crossing is observed late, at a larger R, by an amount set by how fast w₂ moves
  relative to (w₁, b₁).
- **Test**: slow w₂ alone. If the residual is lag, it should shrink toward zero. If it is a property of
  the landscape, it should not depend on w₂'s speed.

## Design

- **Cells**: a ∈ {1.30, 1.50} × w₂ learning-rate factor φ ∈ {0.25, 0.5, 1, 2}.
- **Training sets and seeds**: n = 6,400 total points (`fold1d.make_data(3,200, seed)`), with the size
  test's 50 seeds (300,000–300,049). These are the same training sets, so each run's own threshold is
  already computed (`sample_size_own.csv`, n = 6,400). The design is paired across φ.
- **Everything else unchanged**: the phase 2b protocol.
  - float64, full batch, Adam lr 1e−2 for all parameters, torch.manual_seed(seed) then U(−1, 1)⁴;
  - crossing is the first step at which the dense-grid oriented gap is > 0, with R_cross = |w₂|·Ĝ_cert/2.
- **How w₂'s learning rate is scaled**: after each standard Adam step, w₂'s update is rescaled by φ. That
  is identical to a per-parameter learning rate of 1e−2·φ for w₂. Adam's moment estimates are
  unaffected.
  - At φ = 1 the rescaling is not applied at all, so φ = 1 is exactly the reference runs' code path.
  - A first implementation used separate Adam parameter groups. On one of two seeds tried, it differed
    from the reference in the last few bits of |w₂|, so it was replaced. That is disclosed here.
- **Budget**: 32,000/φ steps (128,000 at φ = 0.25; 16,000 at φ = 2). A slower w₂ needs proportionally
  more steps to reach the same scale.
- **Crossings**: at least 40 of 50 per cell are required. A cell with fewer is reported as "insufficient
  crossings", and it is not extended.

## Validity check (a stop condition)

- **φ = 1 must reproduce the size test's n = 6,400 free-training runs exactly**, for all 100 runs (both
  a): the crossing step and |w₂| at crossing are bit-identical, read with exact round-trip parsing on
  both sides, and non-crossers are non-crossers on both sides.
- **Stop and report if it does not** (`lag_test.check_reproduce`, called before scoring).
- **Exercised before this design**:
  - on constructed files: it passes on identical data, fails on a one-ulp change in |w₂|, and fails on a
    crossing where the reference had none;
  - on real runs, with the final code, at a = 1.30 seeds 300,000 and 300,001 and a = 1.50 seed 300,000
    (data already seen): crossing step and |w₂| are bit-identical to the size test's runs.
- **The rescaling itself**: tested against a hand-written Adam with a per-parameter learning rate. It
  agrees to 1e−12 over 50 steps at φ ∈ {0.25, 1, 2}, and at φ = 1 it is a no-op.

## Registered quantity

- **residual(a, φ)** = median over crossing runs of (R_cross / R_own) − 1, with R_own each run's own
  threshold at n = 6,400.
- **Threshold used**: the own global threshold (`sample_size_own.csv`), or the branch-matched one.
  **This is decided before registration**, from the mirror-branch analysis:
  - if that analysis shows the branch a run ends on is predictable from its initialisation (criterion to
    be set with you once the analysis is in), the branch-matched threshold selected by initialisation is
    used;
  - otherwise, the global own threshold.
- **Uncertainty**: bootstrap 95% intervals (10,000 resamples, seed 0) for each residual, and for the
  differences between residuals.

## Registered predictions (scored separately at a = 1.30 and 1.50)

- **L1 (lag)**: the residual decreases as w₂'s learning rate decreases, and tends toward zero. All three
  must hold:
  - residual(0.25) < residual(0.5) < residual(1) < residual(2);
  - the 95% interval of residual(1) − residual(0.25) lies above 0;
  - residual(0.25) ≤ ½·residual(1).
- **Competing (no dependence)**: the 95% interval of residual(2) − residual(0.25) contains 0. Then the
  residual is not adiabatic lag of w₂.
- **Scorer tests** (constructed data): a planted lag passes L1 and fails the competing outcome; a flat
  residual fails L1 and meets the competing outcome.

## Cost (estimate)

- 400 runs at n = 6,400, including the 100 φ = 1 runs that reproduce existing data.
- Slower φ runs cross later: up to about 4× the steps at φ = 0.25.
- Roughly 1–2 hours at 3 workers.
