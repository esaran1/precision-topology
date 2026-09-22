# Registration (FOLLOW-UP): the cross-family test inside the q2 reduction's domain

**New registration, made AFTER seeing the original test fail.** Timestamp:
2026-09-22T22:56:38Z. Written before any threshold of this follow-up
has been computed.

## What came before, and what does not change

- The original registration (`crossfamily_prediction.md`, commit `a649e31`)
  predicted **X1**: q2's `R_glob` and `R_solve` match family A's within the
  finite-ε correction at `a` = 1.30–1.60. **X1 FAILED as registered**: `R_glob`
  matched at 6 of 6, `R_solve` at 0 of 6 (`crossfamily_results.md`).
- **Post-hoc diagnosis:** the q-families are polynomial only for `|t| ≤ 1`, so their
  reduction is exact only while `ε^{1/2}·max|σ| ≤ 1`. At the tested `a`, 25–34% of
  window points lay on the linear continuation.
- **X1 stays failed in every table, whatever this follow-up shows.** This is a new
  test with its own verdict, not a re-scoring of X1.

## Choice of a, and the domain condition verified before computing

The T70 compact set (`|u| ≤ 2.2`, `|v| ≤ 2`, `|x| ≤ 2`) gives `|σ_A| ≤ 6.4`. The
q2 ↔ A identity rescales `σ` by `1/√2`, so `|σ_q2| ≤ 4.5255` for every
placement in it. The condition holds for all such placements when `ε ≤ 0.0488`.
Chosen: **`a` ∈ {1.01, 1.02, 1.03, 1.04}**.

Verified numerically before any threshold (`crossfamily_followup_domain.csv`):

| `a` | `√ε·max|σ_q2|` | `t` range at q2's certified `Ĝ` argmax | share of window with `|t| > 1` |
|---:|---:|---|---:|
| 1.01 | 0.4525 | [−0.3107, +0.1412] | **0.0000** |
| 1.02 | 0.6400 | [−0.4372, +0.1987] | **0.0000** |
| 1.03 | 0.7838 | [−0.5328, +0.2422] | **0.0000** |
| 1.04 | 0.9051 | [−0.6123, +0.2783] | **0.0000** |

The same share is recorded at each threshold's conditional minimiser, and
reported.

## Procedure

For **family A and q2** at each `a`: certified `Ĝ` (`kappa_certify.certify_exact`
for A; the same exact-extrema certificate for q2); `R_glob` and `R_solve` by the
**frozen Block B procedure** (`best_conditional`, 50 / 24 restarts, 600-step
screen, 8 kept, 3,000 inner steps, degeneracy exclusion, 800-point population
loss, coarse-then-fine bracketing). For q2 only the activation is substituted.
**The grid is fixed in R** at family A's frozen resolution at `a = 1.30`
(`ΔR_coarse = 0.0215`, `ΔR_fine = 0.00215`), for both families at every `a`;
range `R ∈ [0.05, 0.60]`. At these `ε`, `|w₂|` is in the hundreds, so a fixed
`|w₂|` grid would be meaningless.

## Registered prediction

**Y1.** At all four `a`, for **both** `R_glob` and `R_solve`:

    |R_x^{q2}(a) − R_x^{A}(a)| ≤ tol_x(a) = max( |R_x^A(a) − R_x^∞|, 2·ΔR_fine )

with `R_glob^∞ = 0.19991`, `R_solve^∞ = 0.30711` (T58).

## Reading, fixed now

- **If Y1 holds for R_solve**, the paper states that the reduction predicts both
  thresholds across equivalent families, and that X1's failure is explained by the
  domain violation.
- **If R_solve still disagrees**, the reduction covers the placement threshold only,
  and the math note says so.
