# Linking-number estimator validation

This is the blocking gate on Half B of the width sweep. Until the estimator
recovers known integers, has a calibrated noise floor, and refuses the undefined
case, no linking measurement on trained representations is interpretable.

**Outcome: the gate passes.** All seven reference configurations recover their
exact integers, convergence is second-order in sample count, the estimator
tolerates sampling noise up to 20% of curve radius, and it reports the
intersecting case as undefined rather than returning a fraction.

## Method

Direct Gauss double integral on ordered polygonal cycles, following Appendix
G.8 rather than the Appendix H point-cloud detector. Our class cores are known
parametrised curves, so ordered samples can be propagated and the integral
evaluated on the resulting polygon with no PCA projection and no cycle
reconstruction from a k-NN graph. Midpoint quadrature over all segment pairs,
float64, both cycles treated as closed.

Reported per estimate: the raw integral, the nearest integer, the residual
(distance from that integer), the minimum inter-curve vertex distance, and two
flags — `defined` (curves disjoint) and `converged` (residual below 0.25).

## Reference configurations

| Configuration | Expected | Raw | Rounded | Residual | Min distance |
|---|---:|---:|---:|---:|---:|
| Unlink | 0 | 0.000000 | **0** | 0.00e+00 | 4.0000 |
| Hopf link | −1 | −1.000165 | **−1** | 1.65e−04 | 1.0000 |
| Hopf, reversed orientation | +1 | +1.000165 | **+1** | 1.65e−04 | 1.0000 |
| (2,4) torus link | −2 | −2.000255 | **−2** | 2.55e−04 | 1.2000 |
| Chain, adjacent pair 1–2 | −1 | −1.000165 | **−1** | 1.65e−04 | 0.8000 |
| Chain, adjacent pair 2–3 | +1 | +1.000165 | **+1** | 1.65e−04 | 0.8000 |
| Chain, end pair 1–3 | 0 | 0.000000 | **0** | 0.00e+00 | 1.2000 |

All seven recover the exact integer. Orientation reversal flips the sign
exactly (`raw` values are equal and opposite to 1e−9). Argument order does not
change the result. The chain behaves as a chain: adjacent components link with
magnitude 1 and opposite signs, the two ends do not link.

One geometric note. An earlier version of the chain fixture used unit circles
at centre distance 2, which are **tangent** — minimum distance exactly 0 — so
the end pair was a degenerate rather than a clean unlink and the integral was
undefined there. The fixture now enlarges the middle circle so the ends are
separated by 1.2. This is a real trap for any linking analysis on sampled data
and is why minimum distance is reported alongside every estimate.

## Noise floor 1: sampling density

Hopf link, exact value −1, no jitter:

| Points per curve | Raw | Residual |
|---:|---:|---:|
| 20 | −1.016679 | 1.67e−02 |
| 50 | −1.002638 | 2.64e−03 |
| 100 | −1.000658 | 6.58e−04 |
| 200 | −1.000165 | 1.65e−04 |
| 400 | −1.000041 | 4.11e−05 |
| 800 | −1.000010 | 1.03e−05 |
| 1600 | −1.000003 | 2.57e−06 |

The residual falls by a factor of ~4 per doubling, which is second-order
convergence and is pinned by a regression test. Even 20 points per curve
rounds to the correct integer. At the 200 points per class used in the paper's
Table 8, the discretisation error is 1.65e−04, four orders of magnitude below
the 0.5 needed to change the rounded value.

## Noise floor 2: quadrature subdivision

Hopf link at 100 points per curve, refining edges rather than resampling:

| Subdivisions | Raw | Residual |
|---:|---:|---:|
| 1 | −1.000658 | 6.58e−04 |
| 2 | −1.000164 | 1.64e−04 |
| 4 | −1.000041 | 4.11e−05 |
| 8 | −1.000010 | 1.03e−05 |

Subdivision recovers the same convergence rate as resampling, so a coarse
propagated cycle can be refined after the fact.

## Noise floor 3: dispersion across seeds

Hopf link, 200 points, Gaussian jitter applied to every vertex, 10 seeds per
level. Jitter is in absolute units against a curve radius of 1.0.

| Jitter | Mean raw | SD across seeds | Max residual | All round to −1 |
|---:|---:|---:|---:|---|
| 0.00 | −1.000165 | 2.34e−16 | 1.65e−04 | yes |
| 0.01 | −1.000161 | 1.82e−05 | 1.81e−04 | yes |
| 0.02 | −1.000151 | 7.83e−05 | 2.53e−04 | yes |
| 0.05 | −1.000066 | 6.73e−04 | 1.07e−03 | yes |
| 0.10 | −0.999650 | 4.36e−03 | 6.91e−03 | yes |
| 0.20 | −1.010429 | 5.95e−02 | 9.73e−02 | yes |

**The estimator tolerates noise up to 20% of curve radius** without changing
the integer. This is the seed-level uncertainty the paper does not report for
its Table 8.

## Failure boundary

Where it breaks, 20 seeds per level:

| Jitter | Mean raw | SD | Wrong integer | Mean min distance |
|---:|---:|---:|---:|---:|
| 0.30 | −0.489934 | 1.5586 | 12 / 20 | 0.0710 |
| 0.40 | −3.898837 | 19.4537 | 19 / 20 | 0.0549 |
| 0.50 | +3.262550 | 22.1344 | 20 / 20 | 0.0450 |
| 0.70 | −8.086306 | 89.7725 | 20 / 20 | 0.0676 |

The mechanism is visible in the last column: heavy jitter perturbs the two
curves into near-intersection, the integrand blows up like `1/|x-y|^2`, and the
estimate becomes meaningless. This is the same failure Ren and Lim report in
Table 8, where ReLU drives minimum distance to 0.00 and the accompanying link
values 0.50 and 0.18 are starred as artifacts. Our failure region reproduces
that behaviour, which is reassuring about the diagnosis: fractional values near
an intersection are numerical artifacts, not partial linking.

Crucially the failure is **detectable rather than silent** — a regression test
requires that at jitter 0.5 at least 8 of 10 estimates either fail to round to
the truth or fail the convergence check. The estimator does not return a
clean-looking wrong integer.

## The undefined case

Two unit circles in the xy- and xz-planes sharing a centre intersect
transversally at (±1, 0, 0):

| Quantity | Value |
|---|---|
| Minimum distance | 0.000000 |
| `defined` | **False** |
| `converged` | **False** |
| Residual | NaN |

The estimator reports the case as undefined rather than emitting a number. This
matters for Half B specifically: Table 8 shows ReLU collapsing minimum distance
to zero by layer 1, so any layerwise tracking through a monotonic network is
expected to hit this condition, and it must be reported as "linking undefined
here" rather than as a link value that changed.

Degradation is gradual on approach rather than sudden. With curves offset by
1.00, 0.50, 0.20, 0.10, and 0.05 the estimate stays at −1.00004; at offset 0.02
it is −0.99874; at offset 0 it is undefined. So a small positive gap is still
usable, and the reported minimum distance is the quantity that tells a caller
which regime they are in.

## What this does not establish

- The estimator is validated on **closed curves sampled densely and in order**.
  Our class supports are sampled *solid tori*, not curves. Half B must
  propagate ordered core-circle samples separately; feeding it an unordered
  point cloud from the tube interior is not a defined operation.
- Validation is in `R^3`. At hidden widths above 3, two 1-D curves are not
  complementary-dimensional under `m + n + 1 = d`, so this invariant is not the
  Theorem 4.7 invariant there, and any projection back to `R^3` is a stated
  convention rather than a measurement of the layer.
- Nothing here has touched a trained network.

## Status

Gate **passed**. Half B may proceed to propagating ordered core-circle samples
through trained networks, with minimum distance reported at every layer and the
undefined case handled explicitly.
