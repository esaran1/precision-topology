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
- **Negative control added afterwards (Z1, below).** At registration Y1 had no
  negative control. The q1 control, registered after Y1 passed, shows that the
  procedure separates a family with a different limit at these `ε`.
- The "domain violation explains X1" reading is the registration's fixed reading.
  The mechanism (why `R_solve` but not `R_glob` was affected outside the domain)
  remains untested (`crossfamily_results.md`).

## Negative control (Z1): q1 at the same a — added after Y1

Registered in `crossfamily_q1control_prediction.md` (`eb66cb4`, after Y1 passed,
before any q1 threshold at these `a`). The q1 domain was verified first
(`ε·max|σ| ≤ 0.72` over the whole scan box; 0% of the window outside at every
argmax and every minimiser). The q1 scaled procedure passed its gate at
`a = 1.30` (frozen vs scaled: `R_glob` 0.27817 vs 0.27817, `R_solve` 0.31907 vs
0.31907, differences 2e-8). Artifacts: `crossfamily_q1control_*.csv`, producer
`src/crossfamily_q1control.py`.

| `a` | `R_glob` A | `R_glob` q1 | steps | `R_solve` A | `R_solve` q1 | steps |
|---:|---:|---:|---:|---:|---:|---:|
| 1.01 | 0.2007 | 0.2782 | +36 | 0.3083 | 0.3191 | +5 |
| 1.02 | 0.2007 | 0.2782 | +36 | 0.3083 | 0.3191 | +5 |
| 1.03 | 0.2007 | 0.2782 | +36 | 0.3083 | 0.3191 | +5 |
| 1.04 | 0.2028 | 0.2782 | +35 | 0.3083 | 0.3191 | +5 |

**Z1 PASS for both thresholds** (beyond two steps at 4 of 4 `a`; within one step
at 0 of 4). As registered, **Y1's agreement is discriminating for both
thresholds**. The procedure resolves a family with a different limit at these
`ε`, so q2's agreement with A reflects the reduction, not the grid.

- The `R_solve` separation is 5 grid steps (0.0108), a clear but modest margin.
  The `R_glob` separation is 35–36 steps.
- Identical `R` across `a` is grid quantisation: every threshold is reported at a
  fixed `R`-grid point, and q1's thresholds at 1.01–1.04 fall in the same cell as
  at `a = 1.30`.
