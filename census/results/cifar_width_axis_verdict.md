# Width-axis verdict: the CNN arm fails its own pre-fixed rule — propose the MLP arm

Applying the decision rule fixed in `cifar_width_axis.md` **before** the
intrinsic-dimension numbers were read. Data: `cifar_intrinsic.csv`
(4 reference nets — GELU and tanh, 2 seeds each — 2 disjoint 5,000-point
subsamples each, 8 measurements).

## The measurement

ID of the 512-d flattened conv output (the bottleneck's input):

| estimator | range over 8 measurements |
|---|---|
| TwoNN | 45.2 – 51.0 |
| Levina-Bickel MLE k = 10 | 25.6 – 29.9 |
| Levina-Bickel MLE k = 20 | 25.8 – 30.3 |
| PCA-95% (linear reference, excluded) | 364 – 398 |

Overall nonlinear spread **[25.6, 51.0], ratio 1.99×**.

## The rule, and why 1.99 ≤ 2.0 is not a pass

The registered rule: proceed with the CNN arm if the three nonlinear
estimators agree within a factor of 2 (the sweep's own resolution,
~2× per geometric step); otherwise propose the MLP arm and do **not**
rescue the CNN arm by narrowing the sweep or picking a central
estimate.

The measured ratio is **1.99×** — inside the threshold by 0.5%. Treating
that as a pass would be the precise failure the rule was written to
prevent: a threshold cleared by a margin far smaller than the
measurement's own variability is not evidence of agreement. Two
supporting facts make this decisive rather than a judgement call:

1. **The disagreement is systematic, not noise.** TwoNN exceeds MLE by
   1.69–1.75× in *every* one of the four representations, while
   seed/activation variation within an estimator is only 1.13–1.17×.
   This is a stable method-dependent gap, so more seeds would not
   shrink it — it would persist at any sample size.
2. **The two estimators imply non-overlapping test regions.** Under
   MLE, "at/below ID" means w ≤ 28 and "at/above 2×ID" means w ≥ 55.
   Under TwoNN, "at/below ID" means w ≤ 48 and "at/above 2×ID" means
   w ≥ 95. **The MLE "gone" region (w ≥ 55) sits inside the TwoNN
   "advantage present" region (w ≤ 48–95.)** A width of, say, 64 is
   simultaneously "well above 2×ID" and "near ID" depending on
   estimator. The registration would be untestable exactly as the
   MNIST bottleneck registration was.

A third, independent problem compounds it: 2×ID under TwoNN is ~95,
which is 19% of the 512-d input. The "wide" end of the sweep would sit
uncomfortably close to the representation's own dimension, leaving
little room to separate "wide relative to ID" from "wide relative to
the layer".

## Verdict

**The CNN arm does not proceed.** Per the rule, we propose the **CIFAR
MLP arm**, where width is unambiguous by construction, and accept the
weaker positive control that comes with it — a weaker control on an
unambiguous axis being worth more than a strong control on an axis we
had to argue for.

## What the MLP arm costs and what it buys

- **Cost**: the restored positive control (T35) is specific to the
  depth-8 CNN. An MLP on CIFAR-10 is a different architecture, so the
  control must be **re-established there before any width sweep** —
  same four criteria, same tanh-primary comparison. It may fail: the
  MNIST MLP arm of the Task F map showed no reproducible GELU
  advantage at any depth (T35), and CIFAR MLPs may behave the same
  way. If it does fail, Part 2 stops and we report that the width
  prediction could not be tested on real data in either setting we
  could make defensible.
- **Buys**: width is the layer's dimension, full stop — no
  channel/spatial/product choice, no conv-feature-map ID estimate, and
  the ID of a flat MLP hidden representation is the same kind of
  quantity this project has estimated throughout.

## Recommendation, deferred to the user

Two options, and the choice is a scope decision rather than a technical
one:

1. **Run the MLP arm**: first re-establish the control (GELU vs tanh,
   n ≥ 10, plateau criterion), then register the width prediction only
   if it holds. Cost roughly a day of CPU for the control alone.
2. **Stop Part 2 here** and record that the width prediction is
   untested on real data: the CNN setting has a restored control but no
   defensible width axis; the MNIST setting had a defensible axis but no
   control (T34). Neither setting supplies both.

Option 2 is a legitimate stopping point and costs nothing further.
Option 1 is worth it only if the MLP control holds, which the MNIST
evidence makes genuinely uncertain.

**No prediction is registered.** Registering against an axis whose
estimators disagree beyond the sweep resolution would repeat the exact
failure this document exists to prevent.
