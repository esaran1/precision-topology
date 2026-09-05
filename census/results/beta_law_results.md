# 2c: the onset exponent tracks 1/beta. The dissociation is now a quantitative law.

Registered in `arrhenius_prediction.md` after beta verification and **before
any training**. Data: `family_onsets.csv`, `family_onset_curves.csv`,
`beta_law_points.csv`. Figure: `figures/beta_law.png`.

## PRIMARY TEST (registered as the test): the extreme-pair ratio

| | value |
|---|---|
| measured ratio (q4 / q0.667) | **1.661** |
| predicted ratio (beta 2.5 / beta 1.25) | **2.000** |
| deviation | **16.9%** |
| registered band | **+-25%** |
| **VERDICT** | **PASS** |

The registration stated that four exponents each inside +-0.15 is achievable
by accident, and that the real test is the exponents **ordered by beta with
the extreme ratio near 2**. That test passes.

## Supporting detail: individual exponents

| family | beta | measured | predicted (-alpha/beta) | \|diff\| | in +-0.15 band |
|---|---|---|---|---|---|
| q4 | 1.25 | **-0.8305** | -0.8938 | 0.063 | yes |
| familyA (sin) | 1.50 | **-0.7340** | -0.7448 | 0.011 | yes (measured before the constructed families existed) |
| q0.667 | 2.50 | **-0.5000** | -0.4469 | 0.053 | yes |

**Ordering by beta: correct.** Steeper onset exponent for smaller beta,
monotone across all three.

## The law in one object

Plotting measured exponent against 1/beta, the prediction is a line through
the origin of slope -alpha, where **alpha = 1.1172 was measured on the budget
sweep and never fitted to these families**:

- through-origin slope of the three points: **1.0878**
- independently measured alpha: **1.1172** (**2.6% apart**)
- R^2 about the *predicted* line (not a fit): **0.898**

The x-axis is analytic (beta = 1 + 1/q, verified to <=0.7%), so the only
empirical input to the whole relationship is alpha, measured elsewhere.

## Bracketing, reported as always

| family | budget | onset eps | bracketed |
|---|---|---|---|
| q4 | 2,000 / 8,000 / 32,000 | 0.25 / 0.06 / 0.025 | yes (3 cells) |
| q4 | 128,000 | 0.006 | **no -- bound only, excluded from fit** |
| q0.667 | 2,000 | -- | **no -- no eps reached 50%, excluded** |
| q0.667 | 8,000 / 32,000 / 128,000 | 0.4 / 0.25 / 0.1 | yes (3 cells) |

Both fits use **3 bracketed cells**; unbracketed cells are excluded under the
standing criteria, not silently included.

## What this establishes, and what it does not

**Establishes**: the budget exponent tracks the rate at which required scale
diverges. Family B's flat onset (exponent 0.0000, `family_b_budget.md`) is
the beta-undefined end of the same relationship, so the dissociation is now
one quantitative law rather than a presence/absence contrast:
**eps_onset ~ B^{-alpha/beta}**.

**Does not establish**: an end-to-end derivation. **alpha remains measured,
not derived** (P-1a and P-1c both falsified, `alpha_derivation_results.md`).
This is a law with one empirical input.

**Reader note.** q4's predicted -0.8938 sits essentially at the upper edge of
the band registered for **family A** ([-0.895, -0.595]). That is coincidence:
-0.8938 = -1.1172/1.25 is an independent prediction for a different family,
not a restatement of family A's band.

Remaining: q2 (beta 1.5) and q1 (beta 2.0). q2 is the construction's own
re-test of family A -- it predicts -0.7448, the value family A measured at
-0.7340 through an entirely separate activation.
