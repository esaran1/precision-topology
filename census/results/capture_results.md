# The derived capture cross-section: orders rates across a, fails the intervention test

Registered idea: capture probability scales with the admissible b2 interval
width `|w2| * G*(a)` (closed form from `fold1d_theorem.md`) relative to the
realized step in b2. No fitted parameters -- both quantities measured
independently. Data: `capture_cross_section.csv`, `capture_intervention.csv`.

## Across a: orders the rates, but that is weak evidence

| a | rate | G*(a) | \|w2\| reached | width | step_b2 | ratio |
|---|---|---|---|---|---|---|
| 1.02 | 0.000 | 0.0016 | 1.73 | 0.0028 | 0.0021 | 1.3 |
| 1.10 | 0.000 | 0.0175 | 2.27 | 0.0397 | 0.0037 | 10.7 |
| 1.25 | 0.000 | 0.0662 | 3.01 | 0.1991 | 0.0057 | 34.7 |
| 1.35 | 0.010 | 0.1067 | 3.29 | 0.3514 | 0.0063 | 55.6 |
| 1.45 | 0.250 | 0.1513 | 3.53 | 0.5339 | 0.0065 | 82.7 |
| 1.50 | 0.405 | 0.1746 | 3.61 | 0.6304 | 0.0064 | 98.3 |
| 2.00 | 0.715 | 0.4406 | 3.63 | 1.6011 | 0.0047 | 342.8 |
| 3.00 | 0.695 | 1.0492 | 3.03 | 3.1747 | 0.0025 | 1283.5 |

Spearman(ratio, rate) = 0.983, and the zero cells (ratio 1.3-34.7) separate
from the nonzero ones (55.6-1283.5). **But this is the same weak inference as
P-ratio**: G*(a) rises monotonically in a, so any monotone function of a
would order the rates equally well. Consistency, not evidence.

## Holding a fixed and varying the step: the account fails

The test the across-a table cannot do. At fixed a, vary lr only:

| a | lr | rate | \|w2\| reached | width | step_b2 | **ratio** |
|---|---|---|---|---|---|---|
| 1.25 | 3e-2 | **0.500** | 10.25 | 0.678 | 0.0172 | **39.4** |
| 1.25 | 1e-2 | **0.000** | 3.15 | 0.209 | 0.0057 | **36.3** |
| 1.25 | 3e-3 | **0.000** | 0.95 | 0.063 | 0.0016 | **40.1** |
| 1.25 | 1e-3 | **0.000** | 0.56 | 0.037 | 0.00003 | **1293.3** |
| 1.45 | 3e-2 | 0.875 | 10.43 | 1.578 | 0.0135 | 117.1 |
| 1.45 | 1e-2 | 0.175 | 3.66 | 0.554 | 0.0065 | 85.7 |
| 1.45 | 3e-3 | 0.000 | 0.97 | 0.147 | 0.0018 | 80.3 |
| 1.45 | 1e-3 | 0.000 | 0.58 | 0.088 | 0.00003 | 3510.9 |

**The ratio is flat (36-40) while the rate goes 0.000 -> 0.500**, and at
lr = 1e-3 the ratio is the *largest in the table* (1293) with rate exactly 0.
Width and step scale together as lr changes, so their quotient is nearly
invariant -- precisely the wrong behaviour for a quantity meant to predict a
rate that varies by 0.5 across those cells.

**The derived capture cross-section is falsified as a predictor.** It does not
replace the failed Arrhenius form; it fails in a new way, by being invariant
where the phenomenon varies.

## What the same table does establish

The column that tracks the rate across every cell, in both tables, is
**`|w2|` reached**: 0.56 / 0.95 / 3.15 / 10.25 at a = 1.25, with rates
0.000 / 0.000 / 0.000 / 0.500. Solutions at a = 1.25 need |w2| large enough
that the fold's output crosses zero with usable margin; small-step runs stop
at |w2| < 1 and never get there. This is `reach` (T40) again, now measured
under step-size intervention as well as across a.

So the mechanism remains **where the optimizer travels in |w2|**, and it is so
far a measured regularity rather than a derived law: we can predict findability
from reached-|w2| plus the theorem's requirement, but we cannot yet derive
reached-|w2| from the optimizer's parameters.
