# What the phenomenon is, restated after the criticality measurements

Supersedes the framing of `thin_target.md` (kept for the record). Part 4 of
the criticality brief. The case we are in: **the non-critical case.**

## The corrected account

The objects we called zero-basin solutions are **not minima**. They are
correctly-classifying parameter vectors with gradient norm 0.02-0.27
(20x-227x training's own terminal gradient norm), loss 0.22-0.69, and
lambda_min < 0 at six of eight values. There is no basin because there is no
attractor.

What gradient descent is doing is climbing in |w2| -- larger |w2| gives larger
margin gives lower BCE -- and it is **stopped by the step budget, not by an
attractor**: median terminal |w2| grows 3.15 -> 48.48 across budgets
2k -> 20k without saturating, and the solve rate follows, 0.000 -> 0.925 at
a = 1.25.

So the phenomenon is:

> **Correct classification becomes available along a direction the loss
> rewards only weakly, so whether it is reached is set by how far the
> optimizer travels before it is stopped -- and the distance required
> diverges as the expressivity threshold is approached.**

The divergence is analytic: the theorem gives |w2| >= 2m/(kappa*D(a)) with
D(a) ~ (8/3)(a-1)^{3/2}, so the required travel grows like (a-1)^{-3/2}.

## What is still ours, stated strictly

1. **A controlled family across an analytic expressivity boundary**, with the
   required weight scale known in closed form and diverging at the threshold.
2. **A proven necessary condition** (T37) verified on 66 solvers found weeks
   before the theorem existed: 0 violations, minimum slack 1.04x.
3. **The impossibility side**: monotone f cannot solve the task at any scale,
   and 0 separations in 5,580 monotonic runs.
4. **Four manipulations of the gap**, all measured: initialization scale (up
   and down), step size, optimizer, and **budget**.
5. **The exclusion measurements**, now correctly interpreted: barriers exactly
   0, sharpness 70x inside the stable regime, distance reversed, margin
   overlapping. These remain measurements that could have come out otherwise;
   what changed is that they now have an explanation rather than being a
   mystery.

## What is NOT ours, and what we got wrong

- GD avoiding sharp minima, the 2/eta threshold, edge of stability:
  Ahn-Zhang-Sra, Lee et al., Cohen et al.
- **Our error, recorded**: we reported Ahn-Zhang-Sra as "not applying" on a
  test of one of its two conditions (`criticality_results.md` 1b). With
  lambda_min < 0 the first condition is live, though the cleaner statement is
  that these points are not stationary at all and so lie outside the
  theorem's scope entirely.
- **Our imprecision, recorded**: "solution" meant "classifies correctly", not
  "minimum of the loss", and the repository conflated them
  (`terminology_correction.md`).
- The framing "unreachable minima" was **wrong** and is withdrawn.

## What we still cannot derive

The margin threshold from the theorem is **necessary but not sufficient**:
predicting rates from the fraction of runs terminating above 2m/G*(a)
over-predicts everywhere it fails (mean abs error 0.283, all errors
positive). |w2| alone does not fix (w1, b1) into the fold's usable window, so
a one-dimensional criterion cannot work. **We do not have a quantitative law
for the rate**, and that is stated as open.

A second obstruction was found and is not explained either: **32% of recorded
trajectory points sit within 1e-4 of loss = log 2**, the constant-predictor
saddle. That is a genuine critical point and a separate failure mode from the
budget cutoff.

## Honest one-sentence version

Correct classification in this family requires a weight scale that diverges
as (a-1)^{-3/2} at an analytic expressivity threshold; gradient descent
climbs toward it but is stopped by its step budget, so the boundary between
"expressible" and "found" is set by optimizer travel rather than by any
property of the loss landscape at the target -- and we can measure this
precisely while being unable, so far, to predict the rate from it.
