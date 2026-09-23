# Cross-family follow-up inside the q2 domain: results

Scored against `crossfamily_followup_prediction.md` (registered 2026-09-22T22:56:38Z,
after X1 failed, before any follow-up threshold was computed). Artifacts:
`crossfamily_followup_domain.csv`, `crossfamily_followup_thresholds.csv`,
`crossfamily_followup_scores.csv`; producer `src/crossfamily_followup.py`.
Procedure: the frozen Block B minimisation run in limit coordinates
(`src/blockB_scaled.py`). That is a **recorded deviation**, justified by the frozen
procedure's validity-gate failure at these `a` alone, and it passed its gate at
`a = 1.30` before use (`blockB_scaled_validity.md`).

**X1 stays failed.** This is a separate test with its own verdict.

## Domain condition (verified before computing)

`√ε·max|σ_q2|` = 0.453 / 0.640 / 0.784 / 0.905 at `a` = 1.01 / 1.02 / 1.03 / 1.04.
The share of window points on q2's linear continuation is **0.0000 at every `a`**,
both at the certified `Ĝ` argmax and at every conditional minimiser.

## Verdict

| prediction | result |
|---|---|
| **Y1**: q2 within `tol` of family A at all four `a`, for both `R_glob` and `R_solve` | **PASS**: `R_glob` 4 of 4, `R_solve` 4 of 4 |

| `a` | `R_glob` A | `R_glob` q2 | `R_solve` A | `R_solve` q2 | tol |
|---:|---:|---:|---:|---:|---:|
| 1.01 | 0.2007 | 0.2007 | 0.3083 | 0.3083 | 0.0043 |
| 1.02 | 0.2007 | 0.2007 | 0.3083 | 0.3083 | 0.0043 |
| 1.03 | 0.2007 | 0.2007 | 0.3083 | 0.3083 | 0.0043 |
| 1.04 | 0.2028 | 0.2007 | 0.3083 | 0.3083 | 0.0043 |

## Reading (as fixed in the registration)

Y1 holds for `R_solve`. **Inside its domain, the reduction predicts both
thresholds across equivalent families**, and X1's `R_solve` failure at
`a = 1.30–1.60` is consistent with the domain violation diagnosed there.

## Limits, stated with the result

- **Resolution.** The comparison is at the registered fixed `R`-grid resolution
  (`ΔR_fine = 0.00215`). At these `ε`, both families sit within one grid step of
  the scaling limit (`R_glob^∞ = 0.19991`, `R_solve^∞ = 0.30711`), and seven of
  the eight comparisons are identical on the grid. Differences smaller than about
  0.002 are not resolved. The test shows that **one limit governs both
  families**. It does not measure a finite-`ε` difference between them.
- **No negative control in this follow-up.** q1, which separated from A in the
  original test (X2), was not rerun at these `a`. So this test's power to
  distinguish families with a *different* limit is shown only by X2, at larger
  `ε`.
- The "domain violation explains X1" reading is the registration's fixed reading.
  The mechanism (why `R_solve` but not `R_glob` was affected outside the domain)
  remains untested (`crossfamily_results.md`).
