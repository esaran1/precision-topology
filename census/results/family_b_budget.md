# 2b: the budget law does NOT generalize to family B. P-2b falsified.

Registered: "the pwl family's onset also moves with budget, in the same
direction (lower onset at larger budget). Falsified if flat or reversed."
Data: `onset_family_b.csv`, `onset_family_b_curves.csv`.

## Result: family B's onset does not move, and its rates fall with budget

| budget | onset (alpha) | bracketed |
|---|---|---|
| 2,000 | **-0.05** | yes (-0.05: 0.575 -> -0.03: 0.475) |
| 16,000 | **-0.05** | yes (-0.05: 0.500 -> -0.02: 0.250) |
| 64,000 | **none located** | no -- max rate 0.150 at any alpha tested |

**Measured exponent: 0.0000** (n = 2 bracketed cells), against family A's
**-0.7340**.

Rates at matched alpha *decline* with budget:

| alpha | B = 2,000 | B = 16,000 | B = 64,000 |
|---|---|---|---|
| -0.05 | 0.575 | 0.500 | -- |
| -0.02 / -0.03 | 0.475 | 0.250 | -- |
| -0.01 | -- | -- | 0.150 |

At B = 64,000 no alpha reaches 50%, so the cell is unbracketed and excluded
from the fit under the standing criteria. It is reported as a bound, and it
is a bound in the *unfavourable* direction: more budget did not lower the
onset, it failed to locate one at all.

**P-2b is falsified.** The budget law is not a property of the mechanism in
general; it is a property of family A.

## The scope this forces, and the hypothesis it suggests

The defensible claim narrows from "onsets are budget-dependent" to:

> **Budget-dependent onsets occur where the required weight scale diverges.**

Family A's fold depth vanishes as D(a) ~ (8/3)(a-1)^{3/2}, so required
|w2| >= 2m/(kappa*D(a)) diverges as a -> 1+, and budget buys the travel that
closes the gap. Family B degenerates in **shear rather than depth**, and
`A_req` was already found **undefined** for family B (T28) -- there is no
diverging scale requirement for budget to overcome.

**This is a hypothesis consistent with both measurements, not a result.** It
was not tested here: testing it would require a family whose required scale
diverges at a different rate and checking that its budget exponent tracks
that rate. Stated as an open prediction rather than a finding.

## Why the negative strengthens rather than weakens the account

A budget law that applied to every activation family would be a statement
about optimizers. One that applies exactly where the theorem says required
scale diverges, and not where it says the requirement is undefined, is a
statement about the **interaction** between an expressivity boundary and a
finite optimization budget -- which is the paper's actual claim.
