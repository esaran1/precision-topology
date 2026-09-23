# Registration: q1 negative control for the cross-family follow-up (Y1)

**New registration, made AFTER Y1 passed** (`crossfamily_followup_results.md`).
Timestamp: 2026-09-23T01:30Z. Written after the domain check below and before
any q1 threshold at `a` = 1.01–1.04 was computed, and before the q1 validity
gate was run.

## Why

Y1 found q2 = family A on `R_glob` and `R_solve` at `a` = 1.01–1.04, but at grid
resolution (`ΔR_fine = 0.00215`), with both families within one step of the
scaling limit. Agreement at that resolution means something only if a family
with a **different** limit separates. q1 is that family: its limit
`h₁(σ) = −σ + sgn(σ)σ²/2` is not a rescaling of `h`, and it separated from A in
the original test (X2). Y1 and X1 stand as recorded, whatever this shows.

## Domain condition, verified before registering

For q1 the reduction `f = ε²[h₁ + ε g₁]` is exact while `ε·|σ| ≤ 1`. Every
placement in the certification scan box (`|u|, |v| ≤ 6`, `|x| ≤ 2`) has
`|σ| ≤ 18`. `crossfamily_q1control_domain.csv`:

| `a` | `ε·18` | `t` range at q1's certified `Ĝ` argmax | share of window with `|t| > 1` |
|---:|---:|---|---:|
| 1.01 | 0.18 | [−0.0327, +0.0129] | **0.0000** |
| 1.02 | 0.36 | [−0.0649, +0.0256] | **0.0000** |
| 1.03 | 0.54 | [−0.0963, +0.0381] | **0.0000** |
| 1.04 | 0.72 | [−0.1272, +0.0503] | **0.0000** |

The share is also recorded at every conditional minimiser and reported.

## Procedure

Exactly Y1's: the scale-equivariant Block B minimisation (`src/blockB_scaled.py`,
generalised to `s = ε^{1/q}`, amplitude `ε^{1+1/q}`; A and q2 are unchanged), the
same restarts, screen, keep, steps and population data, and the same fixed `R`
grid (`ΔR_coarse = 0.0215`, `ΔR_fine = 0.00215`, `R ∈ [0.05, 0.60]`). q1's `Ĝ` is
certified by the same exact-extrema scan in q1 limit coordinates.
Family A's thresholds are **the committed Y1 values**, not recomputed.

**Gate, before use:** at `a = 1.30`, q1's scaled thresholds must reproduce its
frozen ones (`crossfamily_thresholds.csv`: `R_glob` 0.2782, `R_solve` 0.3191)
within one grid step, as A and q2 did. **If the gate fails, stop and report; no
q1 threshold at 1.01–1.04 is used.**

## Registered prediction

**Z1.** At all four `a`, for **both** `R_glob` and `R_solve`:
`|R_x^{q1}(a) − R_x^{A}(a)| > 2·ΔR_fine = 0.0043`.

## Reading, fixed now (per threshold)

- **q1 beyond two steps at all four `a`**: Z1 holds for that threshold, and the
  procedure resolves a different limit at these `ε`. Y1's agreement for that
  threshold is **discriminating**, so it reflects the reduction.
- **q1 within one step of A at all four `a`**: Y1's agreement for that threshold
  **reflects resolution, not the reduction**, and the math note says so.
- **Mixed**: Y1's agreement for that threshold counts as discriminating only at
  the `a` where q1 is beyond two steps. Where q1 is within one step it reflects
  resolution. Between one and two steps it is inconclusive. Each is stated per `a`.
