# Task F Part 2: settling the width axis before registering

Written before the Part 2 registration and before any bottleneck run,
per direction. The question this document answers: **is there a width
variable in the CIFAR CNN that defensibly corresponds to the account's
"width", or must we fall back to an MLP arm?**

## Why "width" is ambiguous in a conv net

The account's width is the dimension of a *flat representation* passing
through a layer — width 3 at d = 3 in the link setting, bottleneck
width w in the MNIST experiment. A conv feature map has no single such
number. Traced through the restored depth-8 architecture:

| after | channels | spatial | product |
|---|---:|---:|---:|
| conv1 | 32 | 32×32 | 32,768 |
| conv4 | 64 | 8×8 | 4,096 |
| conv6 | 128 | 4×4 | 2,048 |
| conv8 | 128 | 2×2 | **512** |
| flatten | — | — | 512 |
| fc | 128 | — | 128 |

At conv8, is the width 128 (channels), 4 (spatial positions), or 512
(product)? Each has an argument. A registration that picked one would
rest on that choice, and a null result would be attributable to the
choice rather than to the account — the MNIST failure mode (T34) in a
new setting.

**Channel count is the worst of the three despite being the intuitive
answer**: a conv layer applies the same 128 filters at every spatial
position, so the representation reaching the next layer is not a
128-dimensional vector but 128 numbers replicated across a 2×2 grid
with different content at each. Whatever the fold account says about a
128-wide flat representation does not transfer to that object without
an argument nobody has made.

## The resolution: bottleneck at the flatten point

The architecture already funnels through a flat 512-vector:

    conv1..conv8 → flatten(512) → fc(128) → head(10)

Insert a linear bottleneck of width w between flatten and fc:

    conv1..conv8 → flatten(512) → **W_bottleneck(w)** → fc(128) → head(10)

with the activation applied at the bottleneck as at every other hidden
layer. Then:

- **w is a single unambiguous number** — the dimension of a flat vector
  representation, the same kind of object as the MNIST bottleneck and
  the account's width-d layer. No channel/spatial/product choice.
- **The comparison to intrinsic dimension is well posed**: the ID of
  the 512-d flattened conv output is estimable by the same estimators
  used on flat representations everywhere else in this project.
- **It sits in the setting whose positive control is restored** (T35),
  which the MNIST bottleneck never had.

## Scope limit, stated in both directions

The bottleneck sits **after** the conv stack. Therefore:

- A **null result** at this location would not rule out fold effects
  operating inside the conv layers — it would bound the claim to this
  one flat representation.
- A **positive result** at this location would **not establish** that
  fold effects operate inside the conv stack either. The claim is about
  one flat representation, in both directions.

This symmetry is registered now so neither direction can be quietly
widened afterwards.

## The two control arms that make the gate work

The restored control (T35) was measured on the **unmodified** depth-8
net. Inserting a bottleneck changes the architecture, so the advantage
must be re-established *in the modified architecture* before any width
sweep means anything. Two arms, both required:

1. **`w = 512` (narrowing-free bottleneck)** — a bottleneck at full
   flatten width. This is the original architecture **plus an extra
   linear layer**.
2. **`identity-insert` control** — the same extra linear layer at full
   width, i.e. architecturally identical to arm 1, run so that
   layer-insertion and narrowing are separable.

Without arm 2, a shrunken advantage at wide w is uninterpretable: it
could be the added layer or the narrowing, and the gate could not do
its job. With both, the decomposition is explicit — arm 2 isolates the
cost of inserting a layer, and arms 1-vs-narrow isolate the cost of
narrowing.

**Gate:** if the GELU-over-tanh advantage is absent in the wide arms,
the sweep has no effect to track and Part 2 stops there and reports,
rather than sweeping width against an advantage that does not exist in
the modified architecture.

## Intrinsic dimension and the disagreement rule

ID is estimated on the 512-d flattened conv output of reference nets
trained **without** a bottleneck (GELU and tanh, 2 seeds each, two
disjoint 5,000-point subsamples), by three nonlinear estimators —
TwoNN, Levina-Bickel MLE at k = 10 and k = 20 — with PCA-95% as a
linear reference only. Artifact: `cifar_intrinsic.csv`.

**Decision rule, fixed before the numbers are read:**

- The sweep's width grid is geometric (see registration), so its
  resolution is a factor ~2 per step.
- If the three nonlinear estimators agree within a factor of 2, the
  prediction's test regions (at/below ID, and at/above 2×ID) are
  well separated and the CNN arm proceeds.
- **If they disagree beyond a factor of 2** — i.e. beyond the sweep's
  own resolution — the CNN arm is **not** rescued by narrowing the
  sweep or by picking a central estimate. We **propose the CIFAR MLP
  arm instead**, where width is unambiguous by construction, and accept
  the weaker positive control that comes with it. A weaker control on
  an unambiguous axis is worth more than a strong control on an axis we
  had to argue for.

## Recommendation

The FC-bottleneck-at-flatten is defensible and is the recommended axis,
subject to (a) the ID disagreement rule above and (b) both control arms
showing the advantage survives architecture modification. If either
fails, the MLP arm is the fallback, and this document is the record of
why.
