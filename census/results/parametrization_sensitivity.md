# The width-3 result across twelve torus parametrizations

**This supersedes the single-parametrization version of the width-3 result.**
That earlier statement rested on one hardcoded torus configuration, which the
paper's author identified as a confound: training difficulty depends on the
parametrization, and the original one is handcrafted and simple. The result is
now measured across twelve configurations spanning tube radius, aspect ratio
including asymmetric cases the original generator could not express, offset, and
rotations that break the coordinate-plane alignment.

Complete: 12 configurations, 3,840 runs, held-out evaluation accuracy on 2,000
points, seeds recorded per run in `parametrization_sweep.csv`. Every
configuration was verified as a genuine embedded link before training.

## 1. No monotonic activation separates, in any parametrization

**0 separations out of 1,440 monotonic runs at width 3.**

| Activation | Monotonic | Runs | Separations (0 errors) | Best run |
|---|---|---:|---:|---:|
| tanh | yes | 480 | **0** | 9 errors |
| ReLU | yes | 480 | **0** | 11 errors |
| leaky-ReLU | yes | 480 | **0** | 11 errors |
| **All monotonic** | yes | **1,440** | **0** | **9 errors** |
| **GELU** | **no** | 480 | **33** | **0 errors** |

This is the load-bearing result and it is criterion-free: it counts
misclassified points, so it does not depend on where an accuracy threshold is
placed. The best monotonic run anywhere in the grid — across twelve
parametrizations, four depths, ten seeds, and three monotonic activations —
misses separation by 9 points out of 2,000. **All 46 runs that come within 5
errors of separation are GELU.**

The width 3→4 boundary also holds throughout, in the sense that every
activation reaches separation *in at least one run* at width 4 where none of the
monotonic ones ever does at width 3. That is not the same as every activation
succeeding at width 4: of 480 runs each, GELU separates 315 times, tanh 209,
leaky-ReLU 55, and **ReLU only 10 (2.08%)**. ReLU is not reliably trainable to
perfect accuracy at width 4 here.

## 2. GELU separates in ten of twelve, including all three non-axis-aligned

Separations per configuration:

| Parametrization | Axis-aligned | GELU separations |
|---|---|---:|
| **generic** | **no** | **7** |
| near_offset | yes | 5 |
| **rotated_30** | **no** | **4** |
| asymmetric_both | yes | 3 |
| oblique_offset | yes | 3 |
| unequal_major | yes | 3 |
| asymmetric_tube | yes | 2 |
| baseline | yes | 2 |
| **rotated_generic** | **no** | **2** |
| thick_tube | yes | 2 |
| far_offset | yes | 0 |
| thin_tube | yes | 0 |

**The non-axis-aligned configurations separate more readily than the baseline,
not less.** They average 4.33 separations against 2.22 for the axis-aligned
ones, and `generic` — which differs from the original in tube radius, major
radius, offset, and rotation simultaneously, making it the furthest
configuration from the handcrafted case — **produced 7 separations, more than
any other configuration in the grid**. The baseline produced 2.

This is the opposite of what the handcrafted-and-simple account would suggest.
If the baseline's separability were an artifact of its simple, axis-aligned
geometry, the configurations that remove that simplicity should separate less
often. They separate more.

## 3. Fold layer: a prediction stated before the test

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

### Result: the prediction is not borne out

The prediction above is unchanged from when it was written, before these
configurations finished training.

| Parametrization | Axis-aligned | Separating runs | Fold layers |
|---|---|---:|---|
| baseline | yes | 2 | L1 ×2 |
| thick_tube | yes | 2 | L1 ×2 |
| thin_tube | yes | 0 | **unmeasurable** |
| unequal_major | yes | 3 | L1 ×3 |
| asymmetric_tube | yes | 2 | L1 ×2 |
| asymmetric_both | yes | 3 | L1 ×3 |
| near_offset | yes | 5 | L1 ×5 |
| far_offset | yes | 0 | **unmeasurable** |
| oblique_offset | yes | 3 | L1 ×3 |
| **rotated_30** | **no** | 4 | **L1 ×4** |
| **rotated_generic** | **no** | 2 | **L1 ×2** |
| **generic** | **no** | 7 | **L1 ×6, L2 ×1** |

**Fold occurs at layer 1 in 32 of 33 separating runs (97%).** Splitting by
alignment: all 20 axis-aligned separations fold at layer 1, and 12 of 13
non-axis-aligned separations do, the exception being one `generic` run at depth
12 that folds at layer 2.

**The prediction is not borne out.** Under the author's account, rotating the
second core out of the coordinate planes should push the fold later, because no
single input coordinate then exposes the direction. It does not: the
non-axis-aligned configurations fold at layer 1 essentially as often as the
axis-aligned ones (mean fold layer 1.077 against 1.000).

So the account does not explain the layer-1 immediacy we observe. Removing the
coordinate-plane alignment — the specific property that makes the baseline
"handcrafted and simple" — leaves the immediacy intact.

**This measurement cannot distinguish two remaining explanations, and that
limitation is as important as the result.** Either the fold is cheap for a
width-3 affine map to find under *any* parametrization of this link, in which
case the immediacy is a property of the problem but not of the specific
axis-alignment the objection named; or the immediacy is a property of the
optimiser rather than of the geometry at all. Nothing measured here separates
those. What can be said is narrower than either: the particular mechanism
proposed — that the input parametrization exposes the fold direction to the
first affine map — does not survive removing the alignment that would expose it.

#### The existence proof that later folding is possible

One run folds at layer 2. It is reported in full because it is an existence case:
layer-1 folding is *rare rather than forced* — a two-layer fold is available and
simply seldom taken.

`generic`, depth 12, seed 7:

| Layer | Linking | Min distance | Reportable |
|---:|---:|---:|---|
| 0 | **−1** | 0.661369 | yes |
| 1 | **−1** | 0.300112 | yes |
| 2 | **0** | 0.151286 | yes |
| 3 | 0 | 0.265842 | yes |
| 4 | 0 | 0.188869 | yes |
| 5 | 0 | 0.554790 | yes |
| 6 | 0 | 0.635199 | yes |
| 7 | 0 | 0.936105 | yes |
| 8 | 0 | 0.929120 | yes |
| 9 | 0 | 0.796679 | yes |
| 10 | 0 | 0.556137 | yes |
| 11 | 0 | 0.907599 | yes |
| 12 | 0 | **2.247715** | yes |

Linking holds at −1 through layer 1 and reaches 0 at layer 2. The curves stay
disjoint at every layer — minimum distance never touches zero — and every layer
is reportable, so no value here comes from the artifact regime. Minimum distance
then grows steadily to 2.248 by the output, the same post-fold separation
signature seen in the width-3 traces. This is a genuine two-layer unlinking, not
a measurement failure.

**Two configurations have unmeasurable fold layers**, not absent ones:
`thin_tube` and `far_offset` produced no width-3 separations by any activation,
so there is no separating run to trace. This is reported as unmeasurable
because the quantity is undefined there, not because folding failed to occur.

## 4. The two configurations where width 3 defeats everything

`far_offset` and `thin_tube` produced no separations by any activation. They are
the two places the grid finds width 3 insufficient outright, and each needs
stating precisely because each could be misread.

### far_offset: the only negative gap, and not a reversal

| Activation | Min errors | Separations | Median errors |
|---|---:|---:|---:|
| ReLU | **25** | 0 | 1000 |
| GELU | 33 | **0** | 71 |
| leaky-ReLU | 47 | 0 | 77 |
| tanh | 56 | 0 | 74 |

ReLU's best run misses by 25 points and GELU's by 33, so the gap is −8, the only
negative value in the grid.

**This is not a case where monotonic activations succeed and GELU does not.
Neither succeeds.** No activation separates, so the −8 compares two failures. A
25-point miss and a 33-point miss are both failures to separate, and ordering
them says nothing about which activation can separate the configuration —
because on this evidence none can at width 3. ReLU's median of 1000 errors also
shows its distribution is dominated by dying-ReLU collapse, so its minimum comes
from a small number of surviving runs.

It should not be read as a counterexample to the monotonic zero: the monotonic
activations separate here exactly as often as everywhere else in the grid, which
is never. Nor should it be omitted — it is the configuration where the *gap*
statistic breaks down, and the gap is only meaningful when at least one
activation separates.

### thin_tube: GELU misses by a single point

| Activation | Min errors | Separations |
|---|---:|---:|
| **GELU** | **1** | 0 |
| tanh | 10 | 0 |
| ReLU | 11 | 0 |
| leaky-ReLU | 12 | 0 |

GELU's best `thin_tube` run misclassifies exactly **one point out of 2,000**, at
relative radius 0.844 inside its tube. The gap remains positive at 9 points.

This is the closest any configuration comes to closing the gap, and it is also
where the strict criterion is least informative: a single surface point decides
whether the run counts as a separation. `thin_tube` has the smallest monotonic
minimum in the grid at 10 errors, so if the effect were going to vanish under
any tested configuration it would vanish here. It does not, but the margin is
one point of evidence rather than many.


---

## Supporting analyses

### Minimum errors: the criterion-free comparison

**GELU reaches 0 misclassified points at width 3. No monotonic run gets within
10 points of separation, on a 2,000-point evaluation set.**

Pooled over all 12 configurations, 480 runs per activation at width 3. The
median is given first, deliberately, because it is unimpressive and a reader
should see it before the tail statistic:

| Activation | Monotonic | **Median errors** | Best run | **Separations (0 errors)** |
|---|---|---:|---:|---:|
| **GELU** | **no** | **61** | 0 | **33 / 480** |
| tanh | yes | 73 | 9 | **0 / 480** |
| leaky-ReLU | yes | 73 | 11 | **0 / 480** |
| ReLU | yes | 1000 | 11 | **0 / 480** |

**On typical performance the activations are close.** GELU's median of 61
misclassified points against tanh's 73, on a 2,000-point evaluation set, is a
real difference (permutation p = 0.0000) but a small one. ReLU's median of 1000
is dead-ReLU collapse rather than a fitting difference, and is discussed
separately.

The categorical difference is confined to the extreme: **GELU reaches exact
separation 33 times and no monotonic activation reaches it once.** Why that
shape is the informative one, rather than a broad advantage, is argued below.

**All 46 runs within 5 errors of separation are GELU.** The width 3→4 boundary
also holds across the full grid, in the sense that each activation separates in
at least one width-4 run where no monotonic activation ever separates at width
3. Success rates at width 4 differ sharply and should not be read as uniform:
of 480 runs each, GELU separates 315 times, tanh 209, leaky-ReLU 55, and **ReLU
only 10 (2.08%)**.

This is the criterion-free form of the claim. It does not depend on where a
threshold is placed, and it survives the disagreement between accuracy criteria
documented below, which would otherwise undercut any statement built on
"fraction perfect".

> **The load-bearing claim is the zero, not the minimum.** Per
> `notes/reporting_rules.md`, a count of exactly zero over a whole population is
> not an extreme-value statistic: it does not move with `n` the way a minimum
> does, and the only way more runs can overturn it is by producing a separation.
> **0 monotonic separations in 1,440 runs** under our protocol, and 0 in 360
> under the author's, is the claim to lean on. The monotonic *minimum* — 9, or
> 26, or 42 depending on cell and protocol — is a fragile summary and is
> reported below only alongside its distribution.

Across the full grid the gap is positive in 11 of 12 configurations, ranging
from 9 to 56 points with a median of 18 among the ten where any activation
separates. **`baseline`, the original parametrization, sits near the top of that
range at 52 and should not be quoted as typical.** The twelfth configuration,
`far_offset`, defeats every activation at width 3 and is discussed below. See
the per-configuration table.

**All 16 runs that come within 5 errors of separation are GELU.** Not one
monotonic run appears in that set, across five completed parametrizations, four
depths, and ten seeds each.

### The near-misses are sampling artifacts on runs that essentially succeeded

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

### Failure modes differ, though not absolutely

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

### The accuracy criterion and the median disagree everywhere

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

### Why exact separation is tube-radius sensitive

Exact separation on a finite sample is not a property of the learned map. It is
the event that no sampled point falls on the wrong side, which depends on how
many points were drawn and where they landed. Thinner tubes concentrate points
near the decision surface, so a configuration can be *easier* on average while
producing *fewer* exactly-perfect runs.

`thin_tube` shows this directly: its GELU accuracies are higher than baseline's
(median 0.9880 against 0.9632) yet it has **fewer** perfect runs (0 against 2).

### Width 3: distributions, not means

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

### The gap under two criteria

| Parametrization | Gap (perfect) | Gap (≥ 0.999) | Monotonic min errors |
|---|---:|---:|---:|
| baseline | +0.050 | +0.075 | 52 |
| thick_tube | +0.050 | +0.100 | 14 |
| asymmetric_tube | +0.050 | +0.050 | 18 |
| unequal_major | +0.075 | +0.075 | 34 |
| **thin_tube** | **0.000** | **+0.025** | **10** |

#### The thin_tube exception, stated precisely

Under `thin_tube` the gap **vanishes under the strict criterion and persists
under the tolerant one**. The strict criterion is what breaks there, not the
separation: GELU's best `thin_tube` run misses by exactly 1 point, and that
point sits at relative radius 0.844 inside the tube.

`thin_tube` also has the **smallest monotonic minimum errors in the table, at
10** — the closest any monotonic configuration comes to separation anywhere in
the completed grid. So if the effect were going to disappear under any
configuration tested, this is where it would, and under the tolerant criterion
it does not.

### What the main sweep never varied

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

### Validation

Every configuration was checked before any training: linking number ±1 by the
validated Gauss estimator, tubes embedded, tubes disjoint. The check rejected
two draft configurations whose tubes overlapped once the second major radius was
raised, with gaps −0.0000 and −0.0500; both were corrected by increasing the
offset. All 12 configurations in the final grid are valid links.

### Minimum errors per configuration

The pooled headline could hide a configuration where the gap closes, so it is
also reported per configuration. Minimum misclassified points out of 2,000, at
width 3, n = 40 per activation:

**The gap and the monotonic minimum are reported as one column, not two.**
Wherever GELU separates exactly its minimum is 0, so the gap
(`monotonic_min − GELU_min`) and the monotonic minimum are *the same number*.
Presenting them side by side would invite reading a correlation between them as
a finding when it is an identity. The single column below is the monotonic
minimum, which equals the gap in every row where GELU's minimum is 0; the one
row where it does not is marked.

| Parametrization | Aligned | GELU min | tanh | ReLU | leaky-ReLU | **Monotonic min (= gap)** |
|---|---|---:|---:|---:|---:|---:|
| baseline | yes | **0** | 54 | 53 | 52 | **52** |
| oblique_offset | yes | **0** | 35 | 55 | 63 | **35** |
| unequal_major | yes | **0** | 34 | 35 | 36 | **34** |
| asymmetric_tube | yes | **0** | 32 | 28 | 18 | **18** |
| thick_tube | yes | **0** | 68 | 104 | 14 | **14** |
| asymmetric_both | yes | **0** | 15 | 14 | 14 | **14** |
| near_offset | yes | **0** | 47 | 56 | 11 | **11** |
| **rotated_30** | **no** | **0** | 60 | 56 | 59 | **56** |
| **rotated_generic** | **no** | **0** | 15 | 58 | 54 | **15** |
| **generic** | **no** | **0** | 9 | 41 | 40 | **9** |
| thin_tube | yes | 1 | 10 | 11 | 12 | 10 → gap 9 |
| far_offset | yes | 33 | 56 | 25 | 47 | 25 → **gap −8** |

**GELU reaches exact separation in 10 of 12 configurations. No monotonic
activation reaches it in any configuration**, across 1,440 monotonic runs at
width 3.

The gap is positive in 11 of 12. The exception is `far_offset`, and it is not a
case where monotonic activations separate: **no activation separates there**.
GELU's best run misses by 33 points and ReLU's by 25, so the −8 is a comparison
between two failures. It should not be read as a configuration where the
monotonic activations succeed and GELU does not — neither does. `far_offset` and
`thin_tube` are the two configurations where width 3 defeats every activation
tested.

Restricting to the 10 configurations where separation occurs at all, the gap
ranges from **9 to 56** with a median of 18.

The three non-axis-aligned configurations behave like the rest: GELU separates
in all three, with gaps of 9, 15, and 56. `generic`, the configuration furthest
from the baseline in every parameter simultaneously, produced **7 GELU
separations — more than any other configuration in the grid**.

The gap varies widely in size — from 9 to 52 points — so its magnitude is
parametrization-dependent even though its sign is not. **`baseline` gives the
largest gap of any configuration tested**, so the original parametrization is
the most favorable one in the grid and 52 should not be quoted as typical.

**The representative figure is 14 points, with a range of 9 to 52.** Two of the
six configurations sit at exactly 14 and it is close to the median of 16; the
honest summary is that the best monotonic run misses separation by of order ten
points where GELU reaches zero.

#### How the single quantity varies across configurations

Because the gap and the monotonic minimum are one quantity here, the question is
not whether they correlate — they are identical wherever GELU separates — but
whether that **one** quantity tracks how hard the instance is, measured
independently.

Sorted by that quantity:

| Parametrization | Monotonic min (= gap) | Width-4 monotonic perfect rate | GELU median errors |
|---|---:|---:|---:|
| thin_tube | 10 | 0.083 | 24.0 |
| asymmetric_both | 14 | 0.150 | 27.5 |
| thick_tube | 14 | 0.217 | 113.5 |
| asymmetric_tube | 18 | 0.175 | 57.5 |
| unequal_major | 34 | 0.233 | 48.0 |
| baseline | 52 | 0.183 | 73.5 |

Neither proxy enters the gap's definition. Against them the relationship is real
but weak: **+0.45** with the width-4 monotonic perfect rate and **+0.21** with
GELU's median error count at width 3, both on six points.

For the record of what was *not* found: correlating the gap against the
monotonic minimum directly gives 0.9997, and that number is an artifact of the
two being the same quantity. It is recorded here only so that a reader who
computes it is not misled into treating it as evidence.

So the observation is: **configurations where monotonic activations come closer
to separation are the same configurations where the gap narrows, and both appear
to move with how hard the particular instance is, though the evidence that they
track a single difficulty parameter is suggestive rather than settled.** No
mechanism is offered here, and six configurations cannot support one.

#### Consequence for how the number should be quoted

If this pattern holds through the remaining configurations, it follows that
**the gap is not a fixed property of the activation class but a function of how
hard the specific instance is.** Any single number quoted for it — 14, 52, or
otherwise — is then a statement about our parametrization choices as much as
about monotonic versus non-monotonic activations. The sign of the gap has been
stable across everything tested; its magnitude has not, and should always be
reported with the configuration that produced it.

## Audit against the reporting rules

`notes/reporting_rules.md` requires that any minimum or gap appear with its
distribution, and that any comparison of minima across conditions be
resampling-checked. Both quantitative claims in this document were re-examined.
One survives unchanged, one needs qualifying.

### The 9–56 range survives

The per-configuration gap is a minimum over a 40-run cell, so it is exposed to
the extreme-value problem that invalidated the tanh protocol comparison. It is
not affected here.

Monotonic error distributions at width 3, n = 120 per configuration:

| Configuration | min | p10 | p25 | median | p75 |
|---|---:|---:|---:|---:|---:|
| generic | 9 | 43 | 52 | 76 | 667 |
| thin_tube | 10 | 14 | 17 | **24** | 683 |
| near_offset | 11 | 61 | 72 | 94 | 1000 |
| asymmetric_both | 14 | 19 | 25 | 44 | 578 |
| thick_tube | 14 | 109 | 116 | **144** | 586 |
| rotated_generic | 15 | 63 | 70 | 86 | 684 |
| asymmetric_tube | 18 | 34 | 56 | 82 | 685 |
| far_offset | 25 | 61 | 69 | 78 | 578 |
| unequal_major | 34 | 38 | 46 | 60 | 590 |
| oblique_offset | 35 | 64 | 70 | 82 | 504 |
| baseline | 52 | 61 | 68 | 80 | 657 |
| rotated_30 | 56 | 67 | 80 | 94 | 413 |

**The minima are not outliers within their cells.** The gap between best and
second-best, normalised by the interquartile range, is at most 0.064 and is
below 0.01 for five configurations — nothing like the tanh case, where the best
run stood 23 points clear.

The medians span 24 to 144, a **6.1× ratio**, against 6.2× for the minima. The
two summaries agree on the spread. A permutation test on the extremes of the
range, `thin_tube` against `rotated_30`, gives **p = 0.0000 on medians and
p = 0.0000 on minima** over 5,000 permutations.

The range is a real parametrization effect. The caveat already recorded — that
its magnitude depends on parametrization and no single number should be quoted
without its configuration — stands, and is now supported by distributions rather
than by minima alone.

### The "46 runs within 5 errors" claim survives, but the bulk difference is modest

This is an extreme-tail statement and was checked directly. Full error
distributions at width 3, n = 480 per activation:

| Activation | min | p5 | p25 | median | p75 | max | runs ≤5 errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GELU** | **0** | 0 | 34 | **61** | 80 | 208 | **46** |
| tanh | 9 | 21 | 54 | **73** | 94 | 667 | **0** |
| leaky-ReLU | 11 | 19 | 56 | 73 | 103 | 1000 | **0** |
| ReLU | 11 | 35 | 94 | 1000 | 1000 | 1000 | **0** |

**The tail claim holds and is not a lucky draw.** Resampling 480 runs from
tanh's own distribution 5,000 times produces a maximum count of **0** runs at or
below 5 errors, against GELU's observed 46. The separation in the tail is
categorical rather than marginal.

**But the bulk difference is much smaller than the tail suggests, and this
qualifies how the result should be described.** GELU's median is 61 against
tanh's 73 — a 12-point difference on 2,000 evaluation points, real
(permutation p = 0.0000) but modest. GELU is not broadly better at this task;
it is comparable in the bulk and categorically different only at the extreme
where separation actually happens.

That distinction matters for interpretation. The claim this project supports is
**not** "GELU classifies linked tori better than monotonic activations at width
3" — the medians barely differ, 61 against 73. It is the narrower and stranger
claim that **GELU reaches exact separation while monotonic activations never
do**, despite performing similarly on average.

### Why a tail-confined difference is the interesting shape

This is the argument connecting the measurement to the theory, so it is stated
explicitly rather than left implicit.

An **optimisation advantage** — one activation being easier to train, better
conditioned, less prone to bad minima — should shift the whole distribution. If
GELU were simply easier to optimise here, its runs would be broadly better:
lower median, lower quartiles, a distribution translated toward zero. Some of
that is present, but only slightly, at 12 points of median on a 2,000-point
evaluation set.

An **expressivity barrier** predicts something different and more specific. If a
width-3 monotonic network *cannot represent* a separating map, then no amount of
optimisation reaches zero, however well the run goes otherwise. The bulk of the
distribution is governed by how well each run fits the data, which the barrier
does not constrain — a monotonic network can fit almost all of it. Only the
extreme is constrained, because only the extreme requires the map that does not
exist. **The signature is a distribution that looks ordinary in the bulk and
terminates before zero.**

That is what is observed. The measurement matches the shape a representational
limit predicts and does not match the shape an optimisation advantage predicts.

### The approach to zero: a sharp floor, not a thinning tail

The argument above is stronger if the monotonic distributions stop abruptly
rather than petering out. Run counts by error band at width 3, n = 480 per
activation and n = 1,440 monotonic in total:

| Activation | 0 | 1–5 | 6–8 | 9–15 | 16–25 | 26–50 |
|---|---:|---:|---:|---:|---:|---:|
| **GELU** | **33** | **13** | **5** | 4 | 34 | 95 |
| tanh | 0 | 0 | 0 | 7 | 35 | 61 |
| leaky-ReLU | 0 | 0 | 0 | 9 | 39 | 55 |
| ReLU | 0 | 0 | 0 | 6 | 12 | 22 |
| **All monotonic** | **0** | **0** | **0** | **22** | **86** | **138** |

**GELU populates every band down to zero: 33 runs at 0, 13 at 1–5, 5 at 6–8. No
monotonic run lands anywhere below 9, in 1,440 runs.**

The monotonic tail is not thin near the boundary — 22 runs sit at 9–15 errors,
so runs are arriving close to separation in quantity. They simply stop. The
smallest monotonic errors anywhere are 9, 10, 11, 11, 12, 12, 13, 13, 13, 14,
14, …: a populated band with a hard edge below it.

**The observed facts carry this on their own: 22 monotonic runs in the 9–15
band, zero below 9, across 1,440 runs.** That is a floor regardless of what
shape either distribution has.

> **Retraction (2026-08-22).** The floor argument in this section is
> **retracted, not adjusted**. The threshold sweep found a monotonic
> network at **2 errors** and five more below 9; derivative-free search
> found monotonic networks at **0 train errors**. The populated 9–15 band
> with a hard edge below it was an artifact of which activations SGD was
> searching with — tanh, ReLU and leaky-ReLU — not a property of the
> monotonic category. The replacement claim: **monotonic networks can
> shatter the sample and cannot separate the region**; the barrier is at
> exactly zero eval errors, and monotonic networks approach it arbitrarily
> closely without reaching it. The band counts above remain correct as
> measurements of these three activations under SGD; the floor reading
> built on them does not survive. `CLAIMS.md` T6; `threshold_results.md`
> Prediction 3; `search_results.md` train-zero finding.

> *Illustration of scale only, not evidence.* GELU has 51 runs at ≤8 errors
> against 4 at 9–15, roughly 12.75 to 1. Applying that ratio to the monotonic
> 9–15 count would suggest about 280 monotonic runs at ≤8. This is **not** an
> argument: it assumes the two distributions have the same shape near the
> boundary, which is close to assuming what the section is arguing for. It is
> offered only to give a sense of how large the absence is, and should not be
> quoted as a result.

A tail that thins gradually and happens to terminate at 9 would be weak
evidence, consistent with monotonic activations simply being somewhat worse. A
tail that is well populated at 9–15 and then stops completely is the shape of a
floor.

---

> **Status note (2026-08-22).** Dense verification (`dense_check.md`) found
> that 15 of the 33 width-3 GELU separations reported here fail on 100,000
> fresh points from their own configuration (worst 97 errors, margins to
> −25). Dense-verified survivors by configuration: thick_tube 2/2,
> unequal_major 3/3, rotated_30 3/4, generic 3/7, oblique_offset 2/2,
> asymmetric_both 2/3, near_offset 2/5, rotated_generic 1/2, baseline 0/2,
> asymmetric_tube 0/2. Separation counts in this document are sample-level
> (0 errors on 2,000 eval points); no number above is revised, and the
> monotonic zero is unaffected (a sample-level zero already implies a
> regional zero).
