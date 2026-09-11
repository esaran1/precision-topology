# Rounded-value and point-set audit of committed derived quantities

Prompted by finding that `alpha_stability_results.md`'s 1.5123 reproduces only
from 2-decimal display values. Sixteen derived quantities recomputed from
full-precision artifacts.

## Result: 14 of 16 clean, 2 flagged

| quantity | committed | artifact | diff | status |
|---|---|---|---|---|
| alpha 1k-16k | 1.3456 | 1.3452 | 0.0004 | ok |
| alpha 1k-160k | 1.1173 | 1.1172 | 0.0001 | ok |
| alpha 2k-160k | 1.0387 | 1.0389 | 0.0002 | ok |
| alpha 8k-160k | 0.9160 | 0.9160 | 0.00004 | ok |
| alpha 16k-160k | 0.8464 | 0.8464 | 0.00004 | ok |
| alpha 40k-160k | 0.7193 | 0.7193 | 0.00001 | ok |
| effective-beta slope | 1.0547 | 1.0548 | 0.0001 | ok |
| CIFAR 2ep advantage | 750.4 | 750.375 | 0.025 | ok (display rounding) |
| CIFAR 12ep advantage | 150.6 | 150.625 | 0.025 | ok (display rounding) |
| CIFAR ratio 2->12 | 4.98 | 4.9817 | 0.002 | ok |
| SGD/Adam exponent ratio | 0.447 | 0.4473 | 0.0003 | ok |
| alpha ratio SGD/Adam | 0.643 | 0.6433 | 0.0003 | ok |
| kappa min / max | 0.305 / 0.328 | 0.3046 / 0.3284 | 0.0004 | ok |
| **alpha 1k-4k** | **1.5123** | **1.5109** | **0.0014** | **rounded source** |
| **through-origin slope** | **1.1240** | **1.0984** | **0.0256** | **POINT-SET ERROR** |

## Flag 1: alpha over 1k-4k -- rounded source, cosmetic

1.5123 reproduces only from the 2-dp printed values (0.87/2.95/7.08); 3-dp
gives 1.5113, full precision **1.5109**. Difference 0.0014 against a
sub-range spread of 0.79. **No conclusion moves. Quote 1.5109.**

## Flag 2: the through-origin slope -- NOT rounding, a POINT-SET ERROR

This is the paper's headline agreement and it is quoted three different ways
across nine documents.

**The two values are different point sets, both computed correctly:**

| point set | slope |
|---|---|
| **5 points** (q4, q2, family A, q1, q0.667) | **1.0984** |
| 4 points (q4, family A, q1, q0.667 -- q2 omitted) | **1.1240** |

`beta_law_points.csv`, which the figure generator uses, contains **all five**
and yields **1.0984**. The value **1.1240 comes from a four-point set that
drops q2** -- and q2 and family A share `beta = 1.5`, so dropping q2 removes
the duplicate x-position. That is a defensible modelling choice but it was
never stated, and the two values then propagated independently.

**Current state of the record:**

| document | value | point set |
|---|---|---|
| `beta_law_results.md` | 1.0969 | earlier, pre-merge |
| `asymptotic_finding.md` | **1.0984** | 5 points, correct |
| `family_b_correction.md` | 1.0969 | earlier |
| `alpha_stability_results.md` | 1.1240 | 4 points |
| `cross_optimizer.md` | 1.1240 | 4 points |
| `geometric_transfer.md` | 1.1240 | 4 points |
| `sgd_law_results.md` | 1.1240 | 4 points |
| `interpretive_audit.md` | 1.1240 | 4 points |
| **`paper/results_draft.md`** | **1.1240** | **4 points** |

**The draft quotes 1.1240 while the figure it accompanies plots 1.0984.** A
reviewer comparing text to figure finds a discrepancy in the headline number.

**This is exactly the failure Part 2 was meant to catch, and the
cross-sentence check missed it** because the check only looked for two values
*within the draft* -- the draft is internally consistent and externally wrong.

## Which value should the paper use

**1.0984, the five-point fit**, because:

- it uses every measured family, including q2, whose omission was never
  justified;
- it is what `beta_law_points.csv` and the committed figure produce;
- the duplicate x-position at `beta = 1.5` is a feature -- q2 and family A are
  **different activations at the same analytic beta**, which is the
  construction's cross-route control, not a redundancy to be dropped.

Consequence for the claim: the slope becomes **1.0984 against alpha 1.1173**,
**1.7% apart** rather than 0.6%. The intervals still overlap. **The agreement
is slightly weaker than the draft states and must be restated.**

## Action list

1. Draft: 1.1240 -> **1.0984**, and recompute its interval on the 5-point set.
2. Eight committed documents carry 1.1240 or 1.0969: add a dated correction
   pointing to the five-point value.
3. `alpha_stability_results.md`: 1.5123 -> **1.5109**.
4. Figure caption must state the point set explicitly.
