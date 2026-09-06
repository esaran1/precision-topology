# Part 2b: kappa. Derivation partial; and no conclusion depends on its value.

**Provenance: raised in external expert review** -- "the numerical kappa bound
needs justification beyond the grid search."

## The derivation: partial success, reported as such

**Proved (numerically, to 0.2%): kappa depends only on the window RATIOS, not
on their absolute scale.** Scaling all three window edges by 2x
(0.8/1.2/2.0 -> 1.6/2.4/4.0) leaves kappa at 0.3147 against 0.3152 --
invariant. Changing the ratio moves it substantially (r1 = 2.0 gives 0.5166).
So kappa = kappa(r1, r2) with r1 = OUTER_MIN/INNER_MAX = 1.5 and
r2 = OUTER_MAX/INNER_MAX = 2.5. That is a real structural result: kappa is a
property of the task's window geometry, not of the activation or of eps.

**Not obtained: a closed form.** The natural quadratic-dip model
(gap/D = (1 - 1/r1^2)/2) predicts 0.278 against the measured 0.315, and the
discrepancy widens with r1 (0.375 vs 0.517 at r1 = 2.0). The dip is not
quadratic over the traversed range, and no closed form was found.

**kappa is therefore a NUMERICALLY OBTAINED constant and is reported as one
everywhere.** Grid used: 33 values of b1 x 400 values of w1 (`maximum_gap` in
`fold1d_theorem.py`), giving kappa in [0.305, 0.328] over a in [1.02, 3.0].

## Sensitivity: kappa cancels from every exponent

Where kappa appears:

| quantity | kappa-dependent? |
|---|---|
| theorem's bound \|w2\| >= 2m/(kappa*D(a)) | **yes** -- it is the prefactor |
| beta (fold-depth exponent), D(a) ~ eps^beta | **no** |
| alpha (budget growth exponent) | **no** |
| onset exponent -alpha/beta | **no** |
| four-family relationship | **no** |

**kappa scales the required-|w2| prefactor and cancels from every exponent.**
Downstream exponent conclusions are exactly kappa-independent -- not
approximately, but algebraically.

Sensitivity of the one quantity that does depend on it (required |w2| at
m = 0.01):

| kappa | req \|w2\| at a = 1.02 | at a = 1.5 | vs kappa = 0.315 |
|---|---|---|---|
| 0.10 | 37.8 | 0.361 | 3.15x |
| 0.20 | 18.9 | 0.181 | 1.57x |
| **0.315** | **12.0** | **0.115** | 1.00x |
| 0.50 | 7.6 | 0.072 | 0.63x |
| 1.00 | 3.8 | 0.036 | 0.32x |

**Range over which conclusions hold: kappa in [0.1, 1.0]** -- a tenfold range,
far beyond the measured [0.305, 0.328]. Over that entire range no exponent
changes, no onset moves, and no law is affected; only the required-|w2|
prefactor rescales proportionally.

The theorem's empirical verification (0 violations in 66 solvers, minimum
slack 1.04x, `fold1d_theorem.md`) uses **G\*(a) measured directly**, not
kappa, so it is unaffected by the constant's provenance.

**Conclusion: kappa being numerically obtained is a stated limitation, not a
load-bearing weakness.** It is presented as a measured constant with its grid
resolution recorded, never as a derived one.
