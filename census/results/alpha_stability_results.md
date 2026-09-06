# Part 3: alpha stability and error propagation. Objection upheld on 3a.

**Provenance: raised in external expert review** -- "the fitted weight-growth
exponent changes substantially with the budget range, so its stability and the
uncertainty in the onset exponents need closer examination."

## 3a: alpha is NOT stable. It is still drifting at the largest budget.

Alpha fitted over every contiguous sub-range of >=3 cells (8 budgets,
1k-160k):

| range | n | alpha | R^2 |
|---|---|---|---|
| 1k-4k | 3 | **1.5123** | 0.991 |
| 1k-16k | 5 | 1.3456 | 0.993 |
| 1k-160k | 8 | **1.1173** | 0.983 |
| 2k-160k | 7 | 1.0387 | 0.988 |
| 8k-160k | 5 | 0.9160 | 0.989 |
| 16k-160k | 4 | 0.8464 | 0.988 |
| 40k-160k | 3 | **0.7193** | 0.995 |

**Spread across sub-ranges: 0.7193 to 1.5123, a range of 0.79.** First three
cells give 1.5123; last three give 0.7193. **Alpha is monotonically decreasing
with the fitting window and has not stabilised at 160k.**

Every sub-range fits with R^2 >= 0.983, so the *local* behaviour is
power-law-like at every scale -- but the exponent itself keeps falling. That
is the signature of a curve that is not a single power law over this range.

**The honest statement, which replaces "alpha = 1.1172":** terminal weight
scale grows approximately as a power law over any restricted budget window,
with an exponent that **decreases monotonically from ~1.5 at short budgets to
~0.72 at long ones**. The value 1.1172 is the full-range (1k-160k) fit and
must be quoted with that range attached. It is not an asymptotic constant, and
extrapolating beyond 160k is unsupported.

## 3b: propagated uncertainty. All four families still agree.

Full-range alpha = 1.1173, SE = 0.0606, **95% CI [0.9985, 1.2360]**.

| family | beta | predicted interval (-alpha/beta) | measured | overlap |
|---|---|---|---|---|
| q4 | 1.25 | [-0.9888, -0.7988] | -0.8305 | **yes** |
| q2 / A | 1.50 | [-0.8240, -0.6657] | -0.7340 | **yes** |
| q1 | 2.00 | [-0.6180, -0.4993] | -0.6521 | **yes** |
| q0.667 | 2.50 | [-0.4944, -0.3994] | -0.5000 | **yes** |

All four overlap once both sides carry uncertainty (measured values also carry
+-0.114 to +-0.171 of grid resolution).

**The registered +-0.15 band was WIDER than the propagated uncertainty**
(+-0.079 at beta = 1.5 from alpha's CI alone), so the original test was
**looser** than the data supported -- not tighter. Passing it was therefore a
weaker result than the numbers allow; the tighter propagated test also passes.

## 3c: the through-origin slope, as intervals

| quantity | estimate | 95% CI |
|---|---|---|
| through-origin slope (4 families) | 1.1240 | [1.0084, 1.2397] |
| independently measured alpha | 1.1173 | [0.9985, 1.2360] |

**The intervals overlap on [1.0084, 1.2360].**

The point estimates differ by 0.6%, but the confidence intervals are +-10% and
+-11%. **This is "consistent within uncertainty", NOT a 0.6% agreement**, and
the earlier report of "1.8% apart" (and this run's 0.6%) overstated the
precision by treating two uncertain quantities as exact. Corrected here.

## Comparison of the three objections

The three review objections had **three different outcomes**, and the
distinction matters:

- **Family B (objection 1): a real correction.** T28's "A_req undefined" was
  false, the dissociation framing was withdrawn, and family B turned out to be
  a counterexample with a *reversed* exponent (+0.25 over 64x). The objection
  cost us a committed claim -- and the surviving law fits better in the
  corrected measure (mean error 0.049 vs 0.070).
- **kappa (objection 2b): a real question with a clean answer, costing
  nothing.** kappa is indeed numerically obtained rather than derived, and we
  now say so; but it cancels from every exponent, and all conclusions hold
  over kappa in [0.1, 1.0], a 10x range. The objection improved the
  presentation and changed no result.
- **alpha stability (objection 3a): a real qualification.** Alpha is not a
  constant over this range; it drifts from 1.51 to 0.72 depending on the
  fitting window. The budget law holds over any restricted window but the
  exponent must always be quoted with its range, and extrapolation is
  unsupported.

Two of three objections changed what we claim. One changed only how we say it.
