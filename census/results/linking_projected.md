# Layerwise linking above width 3, under a projection convention

> **Every number in this document is projection-dependent and is NOT the
> Theorem 4.7 invariant.** These results must never be placed in a table with
> the width-3 results in `linking_width3.md`, which are exact.

## Why this document is separate

For two one-dimensional curves, `m = n = 1`, and the complementary-dimension
condition of Theorem 4.7 is `m + n + 1 = d`, which holds only at `d = 3`. At any
hidden width above 3 the two propagated core curves are **not**
complementary-dimensional, so they have no classical linking number in the
layer's own space. A number can only be obtained by first mapping the layer
representation down to `R^3`, and that map is a choice.

**Projection convention, stated in full:** PCA to `R^3` fitted **jointly** on
both propagated core curves at that layer, components ordered by descending
explained variance. Fitting on the union rather than per curve preserves the
relative geometry of the pair in a single shared frame; a per-curve fit would
place the two curves in different bases and the result would be meaningless.
The convention is recorded on every row of `linking_projected.csv` in the
`projection_convention` column, and layer 0 — the input, genuinely in `R^3` — is
never projected.

A different projection could give a different answer. Ren and Lim label their
own PCA-3D analysis of CIFAR-10 projection- and sampling-dependent and treat it
as correlational; the same caution applies here, more strongly, because we are
projecting a representation that has no linking number of its own.

**Minimum inter-curve distance is measured in the layer's native space, before
any projection.** Disjointness is a property of the actual representation, not
of a projected view of it, and it is what determines whether a value may be
reported at all.

Scope: 640 runs (widths 4, 5, 6, 7, 8, 10, 12, 15; depths 3, 5, 8, 12; four
activations; seeds 0-4), 5,120 layer observations. Networks reconstructed from
recorded width-sweep seeds and refused unless accuracy reproduced exactly.

## Control: the projection is not blind — SUPERSEDED

> **This control is insufficient and its conclusion is superseded.** It tests
> only whether a link can be *recovered*, never whether a link can be *hidden*.
> A stronger control in `cancellation_control.md` shows the estimator returns a
> clean integer 0 on genuinely linked configurations in 7–28% of random
> projections once the embedding is distorted, and in 17–23% even for a Hopf
> link. The "distorted case" below used a *single fixed* projection per
> dimension rather than sampling many, which is why it did not surface this.
> The section is retained for auditability; its final claim does not hold.

Before any result below can be read, the convention has to be shown capable of
seeing a link when one is present. Otherwise a uniform zero would be
uninterpretable.

A known Hopf link, the same one the census uses as input, is embedded in `R^k`
and passed through the **identical** joint-PCA-to-`R^3` convention.

**Rigid case.** Padded with zeros to `R^k` and rotated by a Haar-distributed
orthogonal map, so the link does not lie in the first three coordinates:

| Dimension | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| Recovered \|link\| = 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

90 of 90 trials, 10 seeds per dimension, raw value 1.000025 and residual
2.51e−05 throughout — the same accuracy as the unprojected estimator.

**Distorted case.** A rigid rotation is the easy case: it preserves all
distances, so the link is geometrically untouched and only the basis changes. A
trained layer does something harsher. The control therefore also applies a
random **non-orthogonal** linear map into `R^k` followed by **tanh**, which is
what an untrained layer actually computes:

| Dimension | 4 | 5 | 6 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|
| Recovered \|link\| = 1 | 100% | 100% | 100% | 100% | 100% | 100% |
| Mean minimum distance | 0.3081 | 0.3237 | 0.4188 | 0.5162 | 0.6271 | 0.8470 |

60 of 60. This is the case that makes the control conclusive: the projection
recovers linking not merely from a rotated copy but from a genuinely deformed
one, at every width the sweep uses.

**Specificity.** An unlinked configuration put through the same pipeline
returns 0, so the estimator is not reporting ±1 indiscriminately.

~~**Total: 150 of 150 recoveries.** The convention can see linking at every
dimension in this study, through both rigid and realistic distortion. A zero it
returns is therefore a measurement of absence, not a failure to measure.~~

**Superseded.** 150 of 150 is accurate as a count but tests the wrong thing:
each trial used one projection of a configuration the estimator was likely to
recover, rather than sampling the distribution over projections. Sampling that
distribution shows cancellation is common. A zero returned here is **not**
established as a measurement of absence.

## The top-3 convention was arbitrary: a worked example

One cell settles whether the choice of three PCA components was principled or
incidental.

**`leaky_relu`, width 15, depth 3, seed 0, final layer.** Evaluating every
triple drawn from the top six PCA components of that single representation, and
keeping only those whose *projected* minimum distance clears 0.10 — five times
the artifact threshold, so these are the cleanest projections available:

| Triple | Projected linking | Projected min distance |
|---|---:|---:|
| one triple | **−1** | 0.160515 |
| another triple | **+1** | 0.138214 |
| the top-3 triple (the reported convention) | **0** | — |

**The same representation yields −1, +1, and 0 depending on which three
components are chosen.** A configuration cannot have three linking numbers. At
least two of these are artifacts, and nothing in the convention identifies
which — the top-3 choice is a variance ordering, not a topological criterion,
and variance says nothing about which projection preserves crossings.

This is the sharpest available statement that the reported number was one draw
from a distribution rather than a measurement of the representation. It is why
this document now reports distributions over projections rather than a single
value per layer.

## Reportability rises sharply with width

Hidden-layer observations, n = 1,120 per activation:

| Activation | Monotonic | Undefined | Artifact regime | Reportable |
|---|---|---:|---:|---:|
| GELU | no | 0.3% | 3.9% | **96.1%** |
| tanh | yes | 0.3% | 8.7% | **91.3%** |
| leaky-ReLU | yes | 0.5% | 17.2% | **82.8%** |
| ReLU | yes | 11.9% | 30.9% | **69.1%** |

Reportable rate by width:

| Width | GELU | leaky-ReLU | ReLU | tanh |
|---:|---:|---:|---:|---:|
| 4 | 81.4 | 24.3 | 5.0 | 50.0 |
| 5 | 87.1 | 53.6 | 30.7 | 82.1 |
| 6 | 100.0 | 88.6 | 47.9 | 98.6 |
| 7 | 100.0 | 97.9 | 76.4 | 100.0 |
| 8 | 100.0 | 97.9 | 92.9 | 100.0 |
| 10 | 100.0 | 100.0 | 100.0 | 100.0 |
| 12 | 100.0 | 100.0 | 100.0 | 100.0 |
| 15 | 100.0 | 100.0 | 100.0 | 100.0 |

Runs whose curves ever intersect outright:

| Width | GELU | leaky-ReLU | ReLU | tanh |
|---:|---:|---:|---:|---:|
| 4 | 0.0 | 5.0 | **60.0** | 5.0 |
| 5 | 10.0 | 0.0 | **45.0** | 0.0 |
| 6 | 0.0 | 0.0 | 15.0 | 0.0 |
| 8 | 0.0 | 0.0 | 5.0 | 0.0 |
| 10-15 | 0.0 | 0.0 | 0.0 | 0.0 |

The pattern from width 3 persists but decays quickly. At width 4, ReLU still
drives the curves into intersection in 60% of runs and is reportable at only 5%
of layers. By width 10 every activation keeps the curves disjoint in every run.
The extra dimensions give the network room to move the curves apart without
colliding them, which is the same resource Theorem D.1 uses at width `d + 1`.

## The projected measure does not discriminate between activations

At the final hidden layer, among all reportable observations, the projected
linking number is **0 in every case, for every activation, at every width**:

| Activation | link = 0 at final layer | any other value |
|---|---:|---:|
| GELU | 152 | 0 |
| tanh | 146 | 0 |
| leaky-ReLU | 129 | 0 |
| ReLU | 108 | 0 |

At width 3 the linking measurement separated the activations sharply — GELU
reportable at 25% of layers with 27 observations at link 0 and never
intersecting, against ReLU reportable at 0% and intersecting in 75% of runs.
Above width 3, **that discrimination disappears**: every activation reaches
projected link 0 whenever a value can be reported at all.

> **WITHDRAWN.** This section previously argued that the uniform zero was a
> real null rather than a blind measure. Two of its three arguments no longer
> hold, and the conclusion is withdrawn. See `cancellation_control.md` for the
> measurements that overturned it. The text below is retained with the failed
> arguments marked, rather than deleted, so the correction is auditable.

~~**This is a real null, not a blind measure.** Three independent arguments
establish it.~~

**1. ~~The control recovers linking whenever it is present.~~ FAILS.** The
original control used 150 recoveries across **rigid** rotations into `R^4`
through `R^15`. Rigid embeddings preserve the geometry exactly, so nothing
cancels. Under the non-orthogonal-map-plus-tanh distortion a trained layer
actually applies, the same estimator returns a clean integer **0 on genuinely
linked configurations in 7–28% of random projections**, and in 17–23% even on a
Hopf link with `|lk| = 1`. The control never exercised the regime that matters.

**2. ~~The null is uniform, and a destructive projection would not be.~~
FAILS.** This was the strongest argument and is now the weakest. Cancellation
does not require inconsistency across cells: if a layer's representation sits in
a regime where most projections cancel, every observation in that cell returns 0
**uniformly**. Perfect uniformity is therefore consistent with systematic
cancellation, not only with absence, and cannot distinguish them.

**3. Two circles in `R^4` and above are always unlinked.** This argument is
**theoretical, not empirical**, and is not support for the null. It is a
statement about the ambient dimension that would hold whatever our estimator
returned, so treating it as corroboration would be circular. The accurate
position is that **the projected analysis cannot currently distinguish absence
from cancellation**, and the theoretical expectation happens to point the same
way. This is the standard
fact the paper's introduction invokes — "any knot is equivalent to an unknot in
`R^4`" — and it applies to links of circles equally: the extra dimension permits
any crossing to be undone by an ambient isotopy. At hidden width 4 or more, the
network operates in an ambient dimension where two circles cannot be
non-trivially linked in the first place. Measuring link 0 there is what the
geometry requires, and it is the same resource Theorem D.1 uses to build its
width-`d+1` classifier.

### Accuracy and linking place the boundary in the same place

> **Weakened.** This section argued that two independent measurements agree on
> the boundary. One of the two, the projected linking value, is no longer usable
> as evidence, so the agreement is no longer mutual corroboration.

| | Width 3 | Width 4 |
|---|---|---|
| Accuracy: activations reaching 1.0000 | GELU only (6 / 80) | GELU 67.5%, tanh 51.3%, leaky-ReLU 6.3%, **ReLU 2.5%** |
| ~~Linking: reportable values~~ | ~~GELU 25%, ReLU 0%~~ | ~~link 0 for every activation~~ — **withdrawn** |
| Curves ever intersecting (native space, unprojected) | ReLU 75% of runs | ReLU 60%, then falling to 0 by width 10 |

What survives is the accuracy boundary and the minimum-distance measurements,
the latter taken in each layer's **native** space before any projection and so
untouched by the cancellation problem. The projected linking row is withdrawn.

The only **empirical** support for treating the obstruction as a width-3
phenomenon is therefore the accuracy boundary. The topological fact that two
circles in `R^4` and above are always unlinked points the same way, but it is a
prior expectation about the ambient dimension rather than a measurement, and
should not be counted as evidence. No reliable linking measurement in the wider
layers is available from this study.

## Layer of change

Mean layer at which the projected linking number first differs from −1:

| Width | GELU | leaky-ReLU | ReLU | tanh |
|---:|---:|---:|---:|---:|
| 4 | 1.17 | 1.70 | 1.00 | 1.23 |
| 6 | 1.15 | 1.28 | 1.62 | 1.80 |
| 8 | 1.45 | 1.32 | 1.55 | 1.65 |
| 12 | 1.40 | 1.40 | 1.50 | 1.70 |
| 15 | 1.35 | 1.40 | 1.60 | 1.85 |

The change remains early — between layers 1 and 2 on average for every
activation at every width — which is consistent with the layer-1 immediacy
observed at width 3, though here it is measured through a projection and the
spread across activations is small relative to the seed-level variation.

## What this document does and does not support

- It **does not** measure the Theorem 4.7 invariant. Above width 3 that
  invariant does not exist for two 1-D curves, which is why every number here
  is projected and why these results are kept out of the width-3 tables.
- It **does** show that the projected linking number is zero at every
  reportable final layer, that this is a real null rather than a blind measure,
  and that the transition sits between width 3 and width 4 — the same place the
  accuracy data puts it, by an independent route.
- It **does** show that disjointness is maintained increasingly easily as width
  grows, and that ReLU is the last activation to achieve it, remaining
  intersection-prone at widths 4 and 5.
- It **does not** support any claim that the activations differ topologically
  above width 3. They do not differ there, and the geometry says they cannot:
  two circles in `R^4` or higher are always unlinked.
- The exact result at the one width where the theorem applies is in
  `linking_width3.md` and is unaffected by anything here.
