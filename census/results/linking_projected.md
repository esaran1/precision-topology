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

This is the central negative result of this document. At width 3 the linking
measurement separated the activations sharply — GELU reportable at 25% of layers
with 27 observations at link 0 and never intersecting, against ReLU reportable
at 0% and intersecting in 75% of runs. Above width 3, under this projection,
**that discrimination disappears**: every activation reaches projected link 0
whenever a value can be reported at all.

Two readings are available and this measurement does not choose between them.

1. The obstruction genuinely is gone above width 3, exactly as Theorem D.1
   says it should be, so all activations succeed and there is nothing left to
   discriminate. Width 4 is already theoretically sufficient.
2. The PCA projection destroys whatever structure remains, so the measure is
   uninformative rather than the networks being alike.

The accuracy data favours the first reading — at width 4 every activation
reaches 1.0000 in some runs, whereas at width 3 only GELU does — but the
projected linking number cannot adjudicate this on its own, and should not be
quoted as if it could.

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
  invariant does not exist for two 1-D curves.
- It **does** show that disjointness is maintained increasingly easily as width
  grows, and that ReLU is the last activation to achieve it, remaining
  intersection-prone at widths 4 and 5.
- It **does not** support any claim that the activations differ topologically
  above width 3, because the projected measure returns the same value for all
  of them.
- The exact result at the one width where the theorem applies is in
  `linking_width3.md` and is unaffected by anything here.
