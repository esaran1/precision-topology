# Part 2: sentence verification

Every number in `results_draft.md` traced to its **raw artifact**, not to the
ledger row quoting it. 36 numbers checked; **34 verified, 2 mismatches**, both
diagnosed below. Then a cross-sentence check for the same quantity appearing
with two values.

## Result: 34 of 36 verified against artifacts

Representative traces (full check in the verification script):

| sentence | number | artifact | verified |
|---|---|---|---|
| impossibility, monotone f | -- | analytic | n/a |
| D overstated at a=1.02 | 42.7% | `dip_depth` | yes |
| D overstated at a=3.0 | 136.1% | `dip_depth` | yes |
| theorem verification | 66, 0 violations, slack 1.0394 | `theorem_verification.csv` | yes |
| gradient norms, unreached | 0.019-0.272 | `criticality.csv` | yes |
| gradient norms, found | 0.006-0.014 | `criticality.csv` | yes |
| lambda_min negative | 6 of 8 | `criticality.csv` | yes |
| MEP barrier | exactly 0.000 | `exclusion_table.csv` | yes |
| distance, unreached / found | 4.7-5.5 / 10.2-17.7 | `exclusion_table.csv` | yes |
| sharpness product | 0.025-0.030 | `exclusion_table.csv` | yes |
| onsets, 6 bracketed | 1.60 -> 1.03 | `onset_law_extended.csv` | yes |
| onset exponent | -0.7340 | `onset_law_extended.csv` | yes |
| alpha (headline) | 1.1173 | `budget_alpha.csv` | yes |
| solve rate at 80k, 160k | 1.000 | `budget_alpha.csv` | yes |
| alpha sub-range 40k-160k | 0.7193 | `budget_alpha.csv` | yes |
| cost-axis check | 5 bracketed, -1.0242 | `cost_law.csv` | yes |
| four family exponents | -0.8305 / -0.6749 / -0.6521 / -0.5000 | `family_onsets.csv` | yes |
| family A under SGD | -0.3255 | `sgd_onsets.csv` | yes |
| q4 under SGD | +0.0056, 4 bracketed | `sgd_q4_refined.csv` | yes |

## Mismatch 1: the local slope of D, 1.4364 vs 1.4326

**Not an error in either place -- two estimators of the same quantity.**

- **1.4364** = least-squares slope over sampled `eps` values in [0.03, 0.60].
- **1.4326** = two-endpoint slope, `log(D(1.60)/D(1.03)) / log(0.60/0.03)`.

The draft inherited 1.4364 from `asymptotic_finding.md` without stating which
estimator produced it. **Action: the draft must say "least-squares slope over
the sampled range", or quote the endpoint value.** Either is defensible; an
unlabelled number is not, because a reader recomputing by the obvious
two-point method gets a different answer.

## Mismatch 2: alpha over 1k-4k, 1.5123 vs 1.5109 -- a real finding

The committed `alpha_stability_results.md` quotes **1.5123**. Recomputing from
`budget_alpha.csv` gives **1.5109**.

**Cause, identified exactly**: 1.5123 is reproducible *only* from the
**2-decimal rounded values** printed in the log table (0.87 / 2.95 / 7.08).
From 3-decimal values it is 1.5113; from the CSV's full precision, 1.5109.

| source | value |
|---|---|
| 2-dp printed table | **1.5123** (what the document quotes) |
| 3-dp values | 1.5113 |
| **artifact, full precision** | **1.5109** |

The difference is 0.0014 and changes nothing -- the sub-range spread is 0.79.
But **a committed document quotes a number computed from rounded display
values rather than from the artifact**, which is precisely the class of error
this task exists to find. **Action: quote 1.5109; correct
`alpha_stability_results.md` with a dated note.**

## Cross-sentence consistency: no quantity appears with two values

The historically two-valued quantities were checked explicitly:

| quantity | variants in the record | in the draft |
|---|---|---|
| alpha | 1.1172 (1k-160k), 1.1084 (2k-128k), 0.72-1.51 (sub-ranges) | **1.1173 only**, with the sub-range drift stated as a separate, labelled fact |
| onset exponent | -0.7340 (6 cells), -0.8305 (3 cells, 2k-32k) | **-0.7340 only**; -0.8305 appears only as q4's four-family exponent, a different quantity |
| through-origin slope | 1.0878, 1.0969 (pre-correction), 1.1240 (with interval) | **1.1240 only**, with interval |
| effective-beta slope | 1.0547 | present, **at the same place** as the headline |
| CIFAR ratio | 4.98x (single experiment), 5.65x (two) | **neither** -- see gap 1 |
| monotonic zero | 5,540 (superseded), 5,580 (corrected) | **neither** -- see gap 2 |

**No quantity appears in the draft with two different values.**

## Two gaps the check exposed

**Gap 1: the CIFAR result is absent from the draft entirely.** §3-§6 as
briefed cover the toy setting; the real-data experiment has no section. It
must appear -- with **4.98x**, the single-experiment value -- or the paper
loses its only non-toy measurement. *This is a structural gap, not a wording
one.*

**Gap 2: the monotonic zero (0 in 5,580) is absent.** §3 states the
impossibility analytically ("no monotone activation can solve this task at any
parameter setting"), which is the stronger statement for the 1D task and does
not need the empirical count. But the 5,580-run zero is the project's largest
single body of evidence and belongs in the paper, in the link-setting section
not yet drafted. **Recorded so it is not lost.**

## One sentence that cannot be written cleanly

The §5 sentence on what the budget law asserts about individual runs is
carrying too much:

> "Claims in this section are about where the population threshold sits, not
> about how far any individual run travels."

This is correct but it under-reports what we know: median terminal `|w2|` does
grow 0.87 -> 264 per-run, so individual runs *do* travel further; the
population framing is needed because the **onset** is a quantile and the
staller fraction also falls. Both mechanisms are present and the measurement
does not separate them over 1k-16k, where the fit has most leverage.
**Flagged: this needs two sentences, not one qualifier**, and the honest
version says the threshold moves for two reasons we cannot disentangle.
