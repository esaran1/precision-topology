# Registered predictions: the two G.1 readings

Written **before** the corrugation sweep is run and before any result exists.
The predictions themselves are never revised; outcomes are appended in marked
sections beneath them.

This supplements `corrugation_prediction.md`, whose Prediction 1 stands
unchanged. What is added here is the split into two readings of Appendix G.1,
which was not anticipated when that file was written.

## The ambiguity in Appendix G.1

The paper states the thickening as:

> Sample points as `γ(t) + ε·n(t)` where `n(t)` is a unit normal to the curve,
> `ε ~ U(0, r)` with `r = 0.15`, and high-frequency oscillations `0.3 sin(100t)`
> are added to preserve topology.

Two readings are physically different and cannot both be implemented as one
condition.

**Reading A — the oscillation displaces the core.** The literal reading, and the
one matching the author's follow-up description of a periodic oscillation
orthogonal to the radial direction. At the published values this produces a
**self-overlapping swept tube**: at frequency 100 on a unit circle, one
oscillation period spans arc 0.063 while the tube is 0.30 wide, so successive
crests are about five times closer together than the tube diameter. The core
curve remains embedded with `|lk| = 1`; the thickened region around it does not.

**Reading B — the oscillation modulates the sampled offsets.** The oscillation
perturbs where points are drawn within the tube rather than moving the
centreline. This matches the stated purpose, "to preserve topology", reading as
a perturbation that keeps the thickened cloud from degenerating. It stays
embedded at the published values.

Both are implemented and reported separately. Neither is presented as the
author's intent; the ambiguity is in the source text and is recorded as such.

### Reading A has no sensible operating point

This is the more useful observation than any ratio, so it is stated directly.

**Under Reading A there is no amplitude that is simultaneously embedded and
meaningfully corrugating.** At `r = 0.15` and frequency 100 the largest
amplitude keeping the swept solid embedded is **0.01657** — about 11% of the
tube radius, a perturbation small enough that the tube is barely deformed. Any
amplitude large enough to corrugate the tube, so that no single planar fold
separates the components, exceeds that limit. The two requirements do not
overlap usefully on the axis.

The same holds on the frequency axis: at the published amplitude of 0.3 the
largest embedded frequency is **5.18**, about five oscillations around the whole
curve, which is not a high-frequency oscillation in the sense the phrase
implies.

The ratios against published values are 18× in amplitude and 19× in frequency,
but the ratios are the less informative way to put it. The point is not that the
published numbers are past a threshold; it is that **the embedded side of the
boundary does not contain values that do the job the oscillation is described as
doing.**

> *Correction.* An earlier version of this file reported these boundaries as
> 0.00108 and 0.5, giving ratios of 277× and 200×. Those figures came from a
> self-intersection test that compared arc-length exclusion against chord
> distance; since a chord is always shorter than its arc, the test flagged every
> smooth closed curve, including an uncorrugated circle. The corrected test uses
> a 1.5-diameter arc margin. The qualitative conclusion is unchanged but the
> magnitudes are roughly fifteen times smaller, and the corrected numbers are
> the ones to use.

### The counter-position, stated fairly

It is entirely coherent that Reading A is what was intended and that the swept
solid was simply never required to be embedded.

Under Reading A the **core remains a valid knot with linking number −1**, which
we verify: residual 3.92e−07, core separation 0.6955. If the analysis in the
paper operates only on the core — and the linking number is defined on the core,
not on the solid — then whether the thickened region self-overlaps has no
bearing on any quantity they compute. The thickening would then be a sampling
device for producing training points near the curve, not an object whose
topology matters.

On that reading the geometry is not an error. It is a choice about what the
thickened region is for, and the answer may simply be "somewhere to draw points
from". **This is a question about what the thickened region is meant to be, not
a defect in the paper**, and it is recorded here in those terms.

What the sweep decides, and the argument does not, is whether the two readings
produce different results. If they agree, the ambiguity is immaterial to our
conclusions and can be reported as resolved-by-irrelevance. If they disagree,
the question of intent becomes load-bearing and should go to the author.

## Gating

- **Reading A**: embeddedness is relaxed to a **reported diagnostic**. Minimum
  self-gap is recorded for every configuration, and the writeup states that at
  published values the thickened region self-overlaps. Linking of the cores is
  still gated hard at `|lk| = 1`, since the invariant is defined on the cores
  and that is what the fold measurement uses.
- **Reading B**: the embeddedness gate stays **hard**. Any configuration whose
  tubes overlap or self-intersect is rejected before training.
- **Low-amplitude embedded arm under Reading A**, to separate corrugation
  effects from self-overlapping-tube effects. If the two arms agree, the
  self-overlap is not driving the result; if they disagree, it is.
- **Uniform noise** as a separately swept parameter under both readings, with
  between-class collisions verified absent and minimum between-class distance
  after noise reported.

## Prediction 1 (restated, unchanged): fold layer moves later

**Corrugation should push the fold layer later than layer 1.** With a corrugated
tube, no single direction breaks the link without creating other problems, so a
network should not be able to resolve the topology with one coordinate fold
immediately after the first affine map.

This is the prediction the earlier axis-alignment test failed to examine,
because rigid rotations are undone by the first affine layer at no cost. This
test varies the property the account concerns.

- Fold layer moves later as amplitude or frequency rises: the account is
  supported, and the layer-1 immediacy observed so far is a property of the
  smooth link rather than of the network.
- Fold layer stays at 1 across corrugation: the same negative as before, but
  now against a test that varies the relevant property, so it carries weight the
  earlier negative did not.

Fold layer is measurable only on runs that separate. A configuration producing
no separations has an **unmeasurable** fold layer, not an absent one.

## Prediction 2: the monotonic zero under corrugation

**The monotonic zero should survive.** It has held across 12 parametrizations,
two protocols, and 1,800 runs, and corrugation changes the geometry of the link
rather than the representational capacity of a width-3 monotonic network.

If any monotonic run separates on a corrugated link, that is the most important
result the project would have produced and must be reported immediately and
without softening. Per `notes/reporting_rules.md`, this is a count rather than a
minimum, so it is the claim to lean on and the one a new condition can genuinely
overturn.

## Prediction 3: the tail-versus-bulk shape

**GELU's advantage should stay confined to the tail.** The current shape is
medians of 61 against 73 with 33 separations against 0. If corrugation makes the
task harder, medians should rise for all activations together while the
categorical tail difference persists.

The outcome that would count against the barrier reading is a **broad shift**:
GELU's whole distribution pulling away from the monotonic ones, or the two
converging in the tail while staying apart in the bulk. Either would suggest
what is being measured is optimisation difficulty rather than a representational
limit.

## What is measured

Widths 3 and 4, all four activations, depths 3/5/8/12, at least 10 seeds, on the
corrugated grids for both readings, with the zero-amplitude case required to
reproduce the existing baseline exactly as a generator correctness check.

Primary: **count of monotonic separations**, then the error distribution by
activation with medians reported alongside tail counts, then fold-layer
distribution on separating runs. Minima appear only with their distributions,
per the standing rules.
