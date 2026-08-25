# Claim-invalidating correction: T30's "solution set empty in the box" is false

Found 2026-08-25 during the Part 5 theorem work (hardening brief).
Reported before continuing the audit, per the standing rule.

## How it was found (the methodological point)

**Deriving theory exposed an empirical error.** The Part 5 lower bound
says a solving network needs |w₂| ≥ c/D(a) with c set by task geometry
(c ≈ 1.2 as measured). At a = 1.02 that permits |w₂| ≈ 1 — flatly
contradicting the committed claim that solutions there require |w₂| ≈ 58
and therefore cannot fit in the box. One of the two had to be wrong. The
theorem was right: an explicit |w₂| = 1 solution exists, verified below.

Independently and a few hours earlier, the Part 3 overclaim sweep had
flagged this same sentence — "empty within both boxes **because**
required |w₂| ≈ 58 exceeds the box" — purely on the causal connective,
with no knowledge of the mathematics, and deferred it to the theorem
work rather than rewording it. Two unrelated detectors, a derivation and
a language audit, converged on the same false sentence. The general
lesson is recorded in `notes/instrument_artifacts.md`.

## The claim, as committed

`fold1d_results.md` (Part 2a–2b table) and ledger row T30 stated:

> At a = 1.05/1.10 the solution set is **empty within both boxes** —
> because required |w₂| ≈ 58 exceeds the box. The box-dependence the
> task brief asked us to justify *is the mechanism made visible*: the
> solution set migrates to large ‖θ‖ as a → 1⁺.

## The counterexample

Solutions exist inside **[−5, 5]⁴** at every value tested, including
a = 1.02, with |w₂| = 1. Construction (fold the data window into the
local **minimum** m₀ = π + arccos(1/a) of f_a, then centre b₂ in the
resulting gap):

| a | w₁ | b₁ | w₂ | b₂ | class gap | max inner logit | min outer logit |
|---|---|---|---|---|---|---|---|
| 1.02 | 0.142815 | 3.339947 | 1.0 | −3.140782 | 5.48e−4 | −2.74e−4 | +2.74e−4 |
| 1.05 | 0.226187 | 3.451437 | 1.0 | −3.138588 | 2.19e−3 | −1.09e−3 | +1.09e−3 |
| 1.10 | 0.317978 | 3.571292 | 1.0 | −3.133659 | 6.28e−3 | −3.14e−3 | +3.14e−3 |
| 1.25 | 0.501931 | 3.785094 | 1.0 | −3.118383 | 2.59e−2 | −1.29e−2 | +1.29e−2 |
| 1.50 | 0.706498 | 3.982661 | 1.0 | −3.106537 | 7.68e−2 | −3.84e−2 | +3.84e−2 |

Verified in float64 on 300,001 inner and 200,002 outer points, strict
inequalities on both sides, every parameter inside [−5, 5]. (A
4-decimal rounding of these parameters does *not* solve — the margins
are ~1e−4 — which is itself part of the story below.)

## Why the grid said "empty"

`solution_mask` scans a 41-point grid per axis: step 0.25 on [−5, 5].
A solution needs b₂ inside an interval of width |w₂|·gap(a). For a grid
node to land inside that interval requires

  |w₂| > step / gap(a) = 0.25 / gap. At the widest-gap w₁ this is
  **456** (a = 1.02), **114** (a = 1.05), **39.8** (a = 1.10) for box 5,
  and double each for box 10; at other w₁ within the admissible band the
  requirement runs higher still (e.g. ≈ 170 at a = 1.05, w₁ = 0.45c;
  ≈ 60.7 at a = 1.10).

**That range brackets the reported "≈ 58".** It was never a property of the
solution set — it is the grid's own resolution requirement. The
solution set at a ≤ 1.10 is not absent from the box; it is a sheet
thinner than one grid cell in the b₂ direction, which the scan cannot
see.

## What this does and does not overturn

**Overturned:**
- "The solution set is empty within both boxes at a = 1.05/1.10."
- "Required |w₂| ≈ 58" as a statement about solutions.
- "The box-dependence is the mechanism made visible" — the emptiness
  was an artifact, so it visualizes nothing.

**Not overturned (checked, not assumed):**
- The **findability** results are untouched: 0/200 solves at a ≤ 1.30
  is a training measurement, independent of any grid.
- **Basin volume jumping at the onset** stands: basins were measured by
  SGD from 6,561 initializations, not from the solution grid, and the
  a = 1.25 cell (basin 0 with solution measure > 0) is unaffected.
- **Solutions-with-zero-basins at a = 1.25–1.35** stands, and this
  correction *strengthens* it: solutions exist even lower (a = 1.02)
  with no basin at all.
- The **monotonic zero** (P1-zero, 0/2,000) is a provable impossibility
  and is untouched.
- The **thin-sheet** finding is confirmed, not weakened: the sheets are
  thinner than the audit's own grid could resolve.

## Consequences for the reported geometry table

Every "solution_fraction" in `fold1d_geometry.csv` is a **lower bound
at grid resolution**, not a measure. The reported growth of solution
measure with a is real in direction but the small-a values (0 at
1.05/1.10, "below one cell" at 1.25/1.35) must be read as
"unresolvable at 41⁴", not as "zero". Ordering versus basin volume —
the actual registered prediction — survives, because basins were
measured independently.

## Corrections applied

- `fold1d_results.md`: dated correction block, original preserved.
- `CLAIMS.md` T30: box-emptiness clause withdrawn, thin-sheet and
  basin-volume clauses retained with the grid-resolution caveat.
- `notes/overclaim_sweep.md`: the item flagged there ("empty within
  both boxes *because* required |w₂| ≈ 58") is resolved — it was
  indeed wrong, and causal wording was the tell.

Generator for the counterexamples: `src/box_counterexample.py`.
