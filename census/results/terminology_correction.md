# Terminology correction (2026-08-28): "solution" conflated two things

`criticality_results.md` establishes that the objects we called "zero-basin
solutions" are **not critical points of the loss**: gradient norm 0.02-0.27
(20x-227x training's own terminal gradient norm), loss 0.22-0.69, and
lambda_min < 0 at six of eight values.

Throughout the repository, **"solution" has meant "parameter vector that
classifies every point of both class regions correctly"** -- an exact region
check (`fold1d.solves`). It has *not* meant "minimum of the training loss".
In most of this project the two coincide closely enough that the distinction
did not bite. For the constructed |w2| = 1 objects it does not hold at all.

## The corrected vocabulary

| old phrasing | corrected phrasing |
|---|---|
| "solutions GD does not reach" | **"correctly-classifying regions GD does not terminate in"** |
| "zero-basin solutions" | **"non-critical correctly-classifying points"** (they have no basin because they are not attractors) |
| "solutions exist with no basin" | "correct classification is available at parameters where loss minimization does not stop" |
| "the solution manifold" | "the correctly-classifying region" (it is a region of parameter space, not a set of minima) |

The **found** solutions are unaffected: gradient norm 0.006-0.014, loss
0.02-0.24, lambda_min >= 0. Those are at or near genuine minima and calling
them solutions in either sense is correct.

## What changes and what does not

**Unchanged** (these were always about classification, and the exact-region
check is exactly the right instrument for them):
- the monotonic zero, T1/T2, and its 0-in-5,580 count;
- the impossibility result for monotone f (a provable statement about sign
  changes, nothing to do with critical points);
- the lower bound theorem T37 (a necessary condition on any correctly-
  classifying parameter vector, stated for a given margin);
- the measured onset and every solve rate (all measured by classification);
- the exclusion table's barrier, distance and margin columns.

**Changed in meaning, not in number**:
- T30's "solutions with zero basins" -> correct classification available at
  non-critical points;
- T40's "basin volume" framing -> the basin measurements stand as
  measurements, but a zero basin around a non-critical point is expected,
  not anomalous;
- T41's headline "unreachable minima" -> unreachable *correctly-classifying
  points*, which are not minima.

**Corrected outright**: T41's claim that Ahn-Zhang-Sra "does not apply" was
tested on one of its two conditions (see `criticality_results.md` 1b).

## Why this is the better result

The phenomenon was: correct classification exists at parameters GD never
visits, for no reason we could find. It now is: **loss minimization terminates
in a different region of parameter space than the one where correct
classification first becomes available**, with the offset controlled by an
analytic expressivity threshold. That is derivable rather than merely
measurable, and Parts 2-3 test the derivation.
