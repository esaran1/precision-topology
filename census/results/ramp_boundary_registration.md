# Registration: the boundary test (width-1 SGD forced ramps at χ = 0.03–0.4), 2026-09-26

Committed with the frozen inputs (`results/ramp_boundary/design.csv`, `frozen_switches.csv`, SHA-256 in
`frozen.sha256`) **before any registered run**. Code: `src/ramp_boundary.py`. Tests: `tests/test_ramp_boundary.py`
(constructed PASS, FAIL and UNRESOLVED cases for every rule).

## Question

The lag law r = κχ was verified up to χ ≈ 0.06 (WP-24, WP-31, WP-33). Where does it stop holding? This test forces the
output scale to grow at five rates chosen a priori so that χ = 0.03, 0.06, 0.1, 0.2 and 0.4, and scores each run
against the switch of the branch the ramp forces. That switch is computed from the seed's sample before any run.

## Protocol

- Width 1, f_a(t) = t + a sin t, at a = 1.30 (winding −1) and a = 1.50 (winding 0), the natural windings.
- 40 fresh seeds (862,000–862,039, unused elsewhere), each with its own 400-point training sample.
- SGD with lr 0.3 and full batch, via `ramp.batch_ramp`. There is a 4,000-step warm-up at the population branch point
  θ\*(s₀), with s₀ = 0.5·s\*. After it, s = s₀·e^{γ(t−4000)}.
- **γ is set a priori:** γ = χ_target·η·λ_min(H_pop), with H_pop the committed population Hessian at the switch. For SGD,
  χ = γ/(ηλ_min) exactly.
- The ramp ends at s\*·max(3, 1 + 4κχ_target), so the predicted crossing lies inside the ramp.
- **Forced-branch switch s_branch (frozen before any run):** the forced branch is continued on the seed's own sample from
  θ\*(s₀) to where its gap turns positive, then bisected to 1e−6 relative.
- **χ_own = γ/(η λ_min(H_own)):** H_own is the Hessian at the forced branch point just below s_branch. It is frozen, and
  is used only for validity.
- **Observed:** r = s_cross/s_branch − 1, with crossing detected at every step by the dense-window gap.
- **Predicted:** r = κ_k·χ, with κ_k the committed SGD κ (lag_law) for the run's winding at crossing. The per-run ratio
  is obs/pred, and the cell statistic is its median.

## Rules

- **B1:** the median ratio lies in [0.75, 1.25] in the χ = 0.03 and 0.06 cells at both a. PASS only if all four are in
  the band.
- **B2:** the median ratio lies outside [0.75, 1.25] in the χ = 0.4 cell at both a. PASS only if both are outside.
- **χ\* (descriptive, per a):** the smallest target whose median ratio leaves the band.
- **Validity, per cell:** the median χ_own is within 30% of the target, and at least 30 of 40 runs cross. A rule that
  depends on an invalid cell is UNRESOLVED.

## Competing predictions

- **Law holds up to a boundary near χ ≈ 0.06–0.1 (expected):** B1 PASS and B2 PASS, with χ\* at 0.1 or 0.2.
- **Law already fails at small χ:** B1 FAIL.
- **Law holds even at χ = 0.4:** B2 FAIL, so the boundary lies above the tested range.
- **Nonlinearity makes the forced ramp outrun the tracking:** the fast cells fail to cross, and B2 is UNRESOLVED.
