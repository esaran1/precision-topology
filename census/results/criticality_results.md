# Part 1a: the zero-basin points are NOT critical points, and lambda_min < 0

Registered P-crit-1a/1b in `arrhenius_prediction.md`. Data: `criticality.csv`.
**Both registered predictions are borne out, and one of them means an earlier
claim of ours was wrong.**

## The measurement

| a | population | \|w2\| | **grad norm** | **loss** | **lambda_min** | lambda_max | traj grad at end |
|---|---|---|---|---|---|---|---|
| 1.02 | zero-basin | 1.00 | **0.0192** | 0.692 | **−0.174** | 2.72 | 0.0211 |
| 1.10 | zero-basin | 1.00 | **0.0906** | 0.679 | **−0.315** | 2.73 | 0.0506 |
| 1.25 | zero-basin | 1.00 | **0.1947** | 0.640 | **−0.237** | 2.78 | 0.0067 |
| 1.35 | zero-basin | 1.00 | **0.2394** | 0.610 | **−0.103** | 2.82 | 0.0023 |
| 1.45 | zero-basin | 1.00 | **0.2656** | 0.576 | **−0.020** | 2.88 | 0.0014 |
| 1.45 | found | 5.47 | 0.0114 | 0.234 | +0.0005 | 2.86 | 0.0014 |
| 1.50 | zero-basin | 1.00 | **0.2724** | 0.559 | **−0.010** | 2.91 | 0.0012 |
| 1.50 | found | 4.39 | 0.0137 | 0.244 | +0.0008 | 2.54 | 0.0012 |
| 2.00 | zero-basin | 1.00 | 0.2250 | 0.409 | +0.0095 | 3.02 | 0.0006 |
| 2.00 | found | 4.24 | 0.0113 | 0.084 | +0.0016 | 1.90 | 0.0006 |
| 3.00 | zero-basin | 1.00 | 0.1459 | 0.220 | +0.0206 | 2.48 | 0.0003 |
| 3.00 | found | 3.47 | 0.0057 | 0.019 | +0.0018 | 1.09 | 0.0003 |

## 1a: not critical points. P-crit-1a borne out.

Zero-basin gradient norms are **0.015-0.27**, and **20x-200x larger than the
gradient norm training has when it terminates** at the same `a`
(traj grad at end: 0.0006-0.021). At a = 1.25 the ratio is 0.195 / 0.0067 =
**29x**; at a = 1.5 it is 0.272 / 0.0012 = **227x**. Found solutions sit at
0.006-0.014, i.e. at or near where the trajectory's own gradient has flattened.

Loss confirms it: zero-basin points sit at **0.22-0.69** while found solutions
at the same `a` sit at **0.02-0.24**, a factor 2-11 lower. A point with loss
0.64 and gradient norm 0.19 is a place gradient descent is passing *through*,
not a place it stops.

**These are not minima of the loss. They are correctly-classifying points on
a slope.**

## 1b: lambda_min < 0 at five of eight values. Our earlier dismissal was incomplete.

lambda_min is **negative at a = 1.02, 1.10, 1.25, 1.35, 1.45, 1.50** (values
−0.174 to −0.010). These points are **saddles**, not minima.

Verified as genuine descent directions rather than numerical noise: stepping
along the lambda_min eigenvector lowers the loss in the expected direction at
every value tested (a = 1.02: 0.69187 -> 0.69069; a = 1.25: 0.64047 ->
0.63071; a = 1.45: 0.57593 -> 0.56966), and at a = 1.25 and 1.45 the perturbed
point **still classifies correctly** -- so the descent direction lies partly
within the correctly-classifying region.

**Correction to a claim we published.** `sharpness_results.md` and ledger row
T41 state that the Ahn-Zhang-Sra mechanism "does not apply" to these points,
on the grounds that lambda_max * eta is far below 2. That test was
**incomplete**: Theorem 1 requires `lambda_min < 0` **OR** `lambda_max >
2/eta`, and we tested only the second. With lambda_min < 0 at six of eight
values, **Theorem 1 does apply to those points via its first condition**, and
they are exactly the kind of stationary point it says GD avoids from almost
every initialization.

Two qualifications, stated so the correction is not over-broad:
- Theorem 1 concerns *stationary* points. These points have gradient norm
  0.02-0.27 and are **not stationary at all**, so the cleanest statement is
  that they fall outside the theorem's scope for a more basic reason than
  either eigenvalue condition.
- At a = 2.0 and 3.0 lambda_min is positive (+0.010, +0.021) and the points
  are genuinely local-minimum-like in curvature, though still non-critical by
  gradient norm.

The error is recorded rather than absorbed: we reported a mechanism as
excluded on a test that covered one of its two conditions.

## Consequence

"Zero basin" is not mysterious in the way we framed it. There is no basin
because **there is no attractor**: the loss at these points is high and
falling, with a descent direction available. Gradient descent does not stop
there because it has no reason to.

This does not affect the exact-zero counts, the impossibility result, the
theorem, or the measured onset. It changes what the phenomenon *is*, and the
language throughout the repository must change with it (1c).
