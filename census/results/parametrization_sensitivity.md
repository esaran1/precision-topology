# Parametrization sensitivity of the width-3 result

Status: **partial — 6 of 12 configurations complete** (2,000 of 3,840 runs). The
three non-axis-aligned configurations, which are the furthest from the
handcrafted case the objection describes, are still running and are reported
when they land. Everything below is held-out evaluation accuracy on 2,000
points, seeds recorded per run in `parametrization_sweep.csv`.

## Headline: minimum errors, which needs no threshold

**GELU reaches 0 misclassified points at width 3. No monotonic run gets within
10 points of separation, on a 2,000-point evaluation set.**

| Activation | Monotonic | Best run (min errors) | Median errors |
|---|---|---:|---:|
| **GELU** | **no** | **0** | 56.0 |
| tanh | yes | 10 | 67.5 |
| ReLU | yes | 11 | 1000.0 |
| leaky-ReLU | yes | 12 | 68.5 |

This is the criterion-free form of the claim. It does not depend on where a
threshold is placed, and it survives the disagreement between accuracy criteria
documented below, which would otherwise undercut any statement built on
"fraction perfect".

**All 16 runs that come within 5 errors of separation are GELU.** Not one
monotonic run appears in that set, across five completed parametrizations, four
depths, and ten seeds each.

## The near-misses are sampling artifacts on runs that essentially succeeded

Seven width-3 runs miss separation by 1 to 5 points. All seven are GELU. Their
misclassified points sit at the tube surface:

| Parametrization | Depth | Seed | Errors | Median relative radius | Fraction in outer decile |
|---|---:|---:|---:|---:|---:|
| baseline | 3 | 5 | 1 | 0.945 | 1.00 |
| baseline | 8 | 2 | 3 | 0.833 | 0.00 |
| thin_tube | 3 | 5 | 1 | 0.844 | 0.00 |
| thick_tube | 5 | 0 | 1 | 0.993 | 1.00 |
| thick_tube | 5 | 1 | 1 | 0.958 | 1.00 |
| thick_tube | 5 | 5 | 4 | 0.917 | 0.75 |
| asymmetric_tube | 12 | 3 | 3 | 0.945 | 1.00 |

Relative radius 1.0 means the point lies exactly on the tube skin. These
networks separated the geometry and lost points that happened to be sampled at
the surface, where any decision boundary threading between two tubes must pass
closest to the data.

So the comparison is not "GELU clears a threshold more often". It is that **the
monotonic activations are 10 points away from separation at their best, while
GELU's failures are sampling artifacts on runs that essentially succeeded.**

## Failure modes differ, though not absolutely

Extending the radius analysis to the ten best monotonic runs — the ones closest
to separation anywhere in the completed grid:

| | GELU near-misses (n=7) | Best monotonic runs (n=10) |
|---|---:|---:|
| Median relative radius | 0.945 | 0.859 |
| Mean fraction in outer decile | 0.679 | 0.327 |
| Median fraction in outer decile | 1.000 | 0.300 |
| Runs with **all** errors in outer decile | **4 of 7** | **0 of 10** |

Both fail near the surface, and that shared concentration is **geometry rather
than a shared failure mode**. These are linked tubes: the two components pass
closest to each other near their surfaces, so any decision boundary threading
between them runs nearest the data there, and errors from any cause will
cluster toward the skin. The shared boundary concentration is therefore what the
configuration forces and carries no information about mechanism. Reading it as
"both activations fail the same way" would be the obvious objection and would
be mistaken.

What distinguishes them is how far the errors **reach inward from** that forced
concentration. No monotonic run has all its errors in the outer decile, while
four of seven GELU runs do, and the typical monotonic run has about 30% of its
errors there against GELU's 100%.

This is a difference of degree, not of kind, and is reported as such. The
monotonic failures are not confined to the boundary; neither are they uniformly
distributed through the interior. The honest statement is that monotonic errors
penetrate the tube interior in a way GELU's near-miss errors do not, against a
shared baseline concentration that the link geometry imposes on both.

## The accuracy criterion and the median disagree everywhere

**In all five completed parametrizations, ranking the activations by fraction
perfect gives a different order than ranking them by median accuracy.** Not one
agrees:

| Parametrization | By fraction perfect | By median accuracy | Agree |
|---|---|---|---|
| baseline | gelu > leaky > relu > tanh | **leaky** > gelu > tanh > relu | no |
| thin_tube | gelu > leaky > relu > tanh | **leaky** > tanh > gelu > relu | no |
| thick_tube | gelu > leaky > relu > tanh | gelu > leaky > tanh > relu | no |
| asymmetric_tube | gelu > leaky > relu > tanh | gelu > tanh > leaky > relu | no |
| unequal_major | gelu > leaky > relu > tanh | gelu > tanh > leaky > relu | no |

The sharpest case is `thin_tube`, where **median accuracy ranks leaky-ReLU above
GELU by 0.0003 while GELU is the only activation that ever separates the
classes at all.**

This is a finding about the metric, not a caveat about these runs. Mean or
median accuracy is the wrong summary for a question about separability: it
averages over a mixture of runs that separated, runs that missed by one surface
point, and runs that collapsed to chance. The field's default of reporting mean
accuracy would, on this data, produce the wrong ordering. Error counts and
distributions do not have that failure mode.

## Why exact separation is tube-radius sensitive

Exact separation on a finite sample is not a property of the learned map. It is
the event that no sampled point falls on the wrong side, which depends on how
many points were drawn and where they landed. Thinner tubes concentrate points
near the decision surface, so a configuration can be *easier* on average while
producing *fewer* exactly-perfect runs.

`thin_tube` shows this directly: its GELU accuracies are higher than baseline's
(median 0.9880 against 0.9632) yet it has **fewer** perfect runs (0 against 2).

## Width 3: distributions, not means

Perfect runs out of 40 per cell, with maxima attached so the distance from
separation is visible:

| Parametrization | GELU perfect | mono perfect | GELU max | tanh max | ReLU max | leaky max |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2 | **0 / 120** | 1.0000 | 0.9730 | 0.9735 | 0.9740 |
| thick_tube | 2 | **0 / 120** | 1.0000 | 0.9660 | 0.9480 | 0.9930 |
| thin_tube | 0 | **0 / 120** | 0.9995 | 0.9950 | 0.9945 | 0.9940 |
| unequal_major | 3 | **0 / 120** | 1.0000 | 0.9830 | 0.9825 | 0.9820 |
| asymmetric_tube | 2 | **0 / 120** | 1.0000 | 0.9840 | 0.9860 | 0.9910 |

**The monotonic activations are at 0 perfect in every completed parametrization,
0 of 600 runs.** Their best evaluation accuracy anywhere is 0.9950.

## The gap under two criteria

| Parametrization | Gap (perfect) | Gap (≥ 0.999) | Monotonic min errors |
|---|---:|---:|---:|
| baseline | +0.050 | +0.075 | 52 |
| thick_tube | +0.050 | +0.100 | 14 |
| asymmetric_tube | +0.050 | +0.050 | 18 |
| unequal_major | +0.075 | +0.075 | 34 |
| **thin_tube** | **0.000** | **+0.025** | **10** |

### The thin_tube exception, stated precisely

Under `thin_tube` the gap **vanishes under the strict criterion and persists
under the tolerant one**. The strict criterion is what breaks there, not the
separation: GELU's best `thin_tube` run misses by exactly 1 point, and that
point sits at relative radius 0.844 inside the tube.

`thin_tube` also has the **smallest monotonic minimum errors in the table, at
10** — the closest any monotonic configuration comes to separation anywhere in
the completed grid. So if the effect were going to disappear under any
configuration tested, this is where it would, and under the tolerant criterion
it does not.

## What the main sweep never varied

`data.linked_tori` exposes only four parameters: `n_per_class`, `tube_radius`,
`major_radius`, and `seed`. Both `tube_radius` and `major_radius` are **single
shared scalars applied to both tori**, so aspect-ratio asymmetry between the two
components was not merely unvaried in the main sweep — it was **inexpressible**.
The author named parametrization as a confound, and this is a property of the
parametrization that the original generator could not represent at all.

Also hardcoded: the offset of the second core (exactly one major radius along
`x`), and the orientation of both cores. Core 0 lies in the plane `z = 0` and
core 1 in `y = 0`, each confined to a coordinate plane, so a single input
coordinate distinguishes the components and the fold direction is available to
the first affine layer. That is the specific property the objection concerns,
and the three configurations that break it are still running.

## Validation

Every configuration was checked before any training: linking number ±1 by the
validated Gauss estimator, tubes embedded, tubes disjoint. The check rejected
two draft configurations whose tubes overlapped once the second major radius was
raised, with gaps −0.0000 and −0.0500; both were corrected by increasing the
offset. All 12 configurations in the final grid are valid links.

## Minimum errors per configuration

The pooled headline could hide a configuration where the gap closes, so it is
also reported per configuration. Minimum misclassified points out of 2,000, at
width 3, n = 40 per activation:

| Parametrization | GELU | tanh | ReLU | leaky-ReLU | Best monotonic | **Gap** |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0** | 54 | 53 | 52 | 52 | **52** |
| thick_tube | **0** | 68 | 104 | 14 | 14 | **14** |
| unequal_major | **0** | 34 | 35 | 36 | 34 | **34** |
| asymmetric_tube | **0** | 32 | 28 | 18 | 18 | **18** |
| asymmetric_both | **0** | 15 | 14 | 14 | 14 | **14** |
| thin_tube | 1 | 10 | 11 | 12 | 10 | **9** |

**The gap holds in every completed configuration.** GELU reaches exact
separation in five of six; the smallest gap anywhere is 9 points, under
`thin_tube`, which is also where GELU itself misses by 1. No configuration
closes it.

The gap varies widely in size — from 9 to 52 points — so its magnitude is
parametrization-dependent even though its sign is not. `baseline` gives the
largest gap of any configuration tested, which is worth noting: the original
parametrization flatters the effect relative to the alternatives, and a reader
should not take 52 as typical.

## Fold layer: a prediction stated before the test

The author's account is that an untweaked Hopf parametrization exposes the fold
direction immediately after one affine map, which would make our layer-1 folding
a property of the problem rather than of the network. The non-axis-aligned
configurations — `rotated_30`, `rotated_generic`, and `generic` — are the
tweaked case: their second core is rotated out of the coordinate planes, so no
single input coordinate distinguishes the components.

**Prediction under that account: fold layer should move later in the
non-axis-aligned configurations.** If it stays at layer 1 across all three, the
account does not explain the observation and something else is producing the
immediacy.

This is recorded before those configurations have finished training, so the
result below is a test rather than a reading.

**Measurability caveat.** Fold layer is only defined for runs that separate,
which so far means GELU only — no monotonic run has separated at width 3 in any
configuration. If a non-axis-aligned configuration produces no GELU
separations, its fold layer is **unmeasurable**, which is not the same as absent
and will be reported as such.

Results are reported as a distribution across runs per configuration, not a
single value.

## Still outstanding

`near_offset`, `far_offset`, `oblique_offset`, `rotated_30`, `rotated_generic`,
and `generic`. The last three break the coordinate-plane alignment and are the
configurations furthest from the handcrafted case. Fold-layer results for
successful width-3 runs are also pending.
