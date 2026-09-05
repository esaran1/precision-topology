# Part 1: alpha is NOT derived. P-1a and P-1c both falsified; the mechanism is measured, not derived.

Registered in `arrhenius_prediction.md` before each measurement. Data:
`alpha_trace.csv`, `alpha_composition.csv`, fits in
`alpha_derivation_fit.txt`, `alpha_composition_fit.txt`.

## 1a: the gradient-flow route fails for Adam. FALSIFIED.

| quantity | measured | implied alpha |
|---|---|---|
| gradient exponent, \|dL/dw2\| ~ \|w2\|^{-p} | **p = 1.085** | 1/(1+p) = **0.480** |
| Adam realized step, ~ \|w2\|^{-p_step} | **p_step = 0.005** | **0.995** |
| direct \|w2\| ~ t^alpha within traces | -- | **1.081** (median, n=11, IQR 1.02-1.25) |

Registered P-1a required p = -0.105. Measured p = **1.085**. Falsified by an
order of magnitude in the wrong direction.

**What replaced it, as three measurements rather than an argument:** the loss
gradient decays as |w2|^{-1.085}; Adam divides by the gradient's running
magnitude, which cancels that decay almost exactly (realized step exponent
0.005, i.e. scale-free); so |w2| grows linearly and **alpha ~ 1 for Adam**.

## 1c: the decisive optimizer-swap test. FALSIFIED.

Plain SGD does not normalize, so it should inherit p = 1.085 directly:
predicted **alpha_SGD = 1/(1+1.085) = 0.479**, band [0.38, 0.58], registered
before any SGD run.

| optimizer | alpha (population) | alpha (escapers only) |
|---|---|---|
| Adam | 1.2627 | 1.1416 |
| **SGD** | **0.7188** | **0.6678** |

**VERDICT: OUTSIDE BAND.** Measured 0.719 against predicted 0.479 -- 50%
high, and outside on the escaper fit (0.668) as well, so it is not a
composition artifact.

The registration stated in advance that a miss here falsifies **the
mechanism** (that alpha follows from the gradient's |w2| scaling), not the
optimizer choice. That is the outcome. **alpha remains a measured quantity.**

What survives, because it does not depend on the SGD prediction:
- the Adam normalization account (three measurements above);
- the qualitative separation: SGD's terminal scale grows substantially more
  slowly than Adam's (at B = 40,000: 19.4 vs 97.5), consistent with SGD
  inheriting a decaying gradient while Adam cancels it;
- the registered onset prediction -0.745 and its measurement -0.734;
- the family-B double dissociation.

Section 6 is unchanged. The optimizer-dependence claim drafted in
`section7_optimizer_draft.md` **Branch B applies**: §7 drops it, and the
derivation of alpha is stated as open.

## P-composition: CONFIRMED in sign and location

Stallers at short budgets drag the median terminal |w2| down, steepening the
cross-budget slope upward. Predicted: excluding them lowers alpha.

| optimizer | alpha all | alpha escapers | change |
|---|---|---|---|
| Adam | 1.2627 | 1.1416 | **-0.121** |
| SGD | 0.7188 | 0.6678 | **-0.051** |

Both fall, as predicted. The effect is concentrated where stalling is common
(Adam B = 1,000: population median 0.871 vs escaper median 1.561, 8/30
stalled; by B = 8,000 they agree within 1%, 3/30 stalled).

## Discrepancy to reconcile: two Adam alphas

This sweep's Adam population fit is **1.2627**; the committed value (T43) is
**1.1172**. Cause: different budget grids. The committed fit spans **eight**
budgets 1k-160k; this one spans **six**, 1k-40k, omitting the 80k/160k cells
where growth flattens -- which steepens the slope.

**The committed 1.1172 is the better-supported estimate** (wider range, more
points) and stands. This sweep's 1.2627 should not be quoted as a competing
measurement of the same quantity; it is the same quantity measured over a
narrower range. Noted here so §6 quotes one number with its range stated.

## Consistency resolution (decided before writing, by inspection)

`onset_more.rate_at` -> `run_full` runs all seeds and counts stallers as
failures, so **the onset rate is a population quantity**. The internally
consistent alpha for the onset law is therefore the population alpha
(1.1172, stallers included) -- the one that produced the registered -0.745
and the measured -0.734. The escaper alpha (~1.14 here, ~1.0 from within-run
traces) describes the individual growth trajectory. Two quantities, two
purposes, both correct.
