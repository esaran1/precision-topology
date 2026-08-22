# Link tracer: design document

**Status: design only. Nothing here is implemented, and nothing should be
implemented from it until the open problems in the final section are resolved.**

## The proposal

Take a network **not** trained on links. Draw an imaginary link in its input
feature space. Trace the linking number through intermediate layers and ask
whether ordinary trained networks preserve or destroy topological structure
incidentally, as a byproduct of doing something else.

This differs from everything the project has measured so far. Every previous
experiment trained a network *on* a linked dataset and asked whether it could
separate the classes. Here the network has no relationship to the link at all;
the link is an instrument inserted into a representation built for another
purpose.

---

## 1. Which network

**Recommendation: a small MLP or CNN trained on MNIST, with CIFAR-10 as a
secondary target. Not a transformer, at least not first.**

The argument is about what can be measured, not about which model is most
interesting.

**Against a small GPT checkpoint.** The residual stream is high-dimensional
(768 at the smallest useful scale), the input space is discrete token IDs rather
than a continuous space a curve can be drawn in, and the natural "input" is an
embedding lookup. Drawing a closed curve in embedding space is possible but its
relationship to anything the model was trained on is unclear — no continuous path
between two token embeddings corresponds to inputs the model ever sees. That
makes the placement question below unanswerable.

**For MNIST.** Input space is `R^784`, continuous, and the data occupies a
well-characterised low-dimensional manifold within it. A small MLP has hidden
widths we can choose, so the tracer can operate at widths where the invariant is
defined or close to it. Training is fast enough to sweep architectures and
seeds, which every result in this project has needed.

**CIFAR-10 second**, because convolutional structure makes the input-space
geometry harder to reason about — a curve in pixel space passes through images
that are not photographs of anything — but it is closer to the setting the paper's
own CIFAR analysis uses, so comparison is possible.

### The dimensionality problem, stated plainly

MNIST inputs are 784-dimensional. **A one-dimensional imaginary link is a very
thin object in a very large space.** Two consequences:

1. The link occupies a measure-zero subset of a space the network has strong
   opinions about only near the data manifold. Almost all of `R^784` is off-data,
   and network behaviour there is close to arbitrary.
2. Any linking measurement in a hidden layer of width > 3 requires projection,
   and **Item 2 of the current work established that projection can cancel a
   nonzero linking number to zero in 7–28% of random projections once the
   embedding is distorted** — 17–23% even for a Hopf link. That failure applies
   here with full force and is the single largest threat to this experiment.

---

## 2. Scale of the imaginary link

**Recommendation: set scale relative to the data's own scale, and sweep it.**

The failure modes bracket the useful range:

- **Too small.** A link of radius much less than the typical distance between
  nearby training points sits inside the network's effective linear regime. A
  locally affine map preserves linking exactly, so the answer is known in
  advance and the measurement is vacuous.
- **Too large.** A link extending well beyond the data's support lies where the
  network was never constrained. Its behaviour there is an extrapolation
  artifact, and "the network destroyed the link" would mean only "the network
  does arbitrary things far from its training data".

**Proposed principled choice.** Let `σ` be the median pairwise distance between
training points, and `r` the radius of the data manifold's bounding region.
Sweep link radius over `[0.1σ, 0.3σ, σ, 3σ, 0.3r]`, reporting each separately.
The interesting regime is where the link is large enough to span several data
points — so the network's local behaviour varies across it — and small enough to
stay within the support.

A second scale parameter matters: the **separation between the two components**
relative to their radius. Held fixed at the Hopf-link proportions used
throughout this project, so results are comparable.

---

## 3. Where the link lies

The author asks whether this matters. **Prediction: it matters more than any
other design choice**, and it is the axis most likely to produce an interpretable
result.

Three placements, all measurable:

**(a) On the data manifold.** Both components drawn so every point is close to
some training example — for MNIST, interpolations between digit images. The
network has strong, trained opinions here.

*Prediction:* linking is destroyed, and early. Building class-relevant
invariances means collapsing directions that distinguish nothing about the
label, and a link is a structure such collapse would tear.

**(b) Off the data manifold.** Both components in a region of input space the
data does not occupy — for MNIST, images of pure noise at matched norm.

*Prediction:* linking is preserved further, possibly indefinitely. The network
has no reason to have learned any particular map there, and an untrained
direction acts closer to a random projection, which preserves linking generically.

**(c) Spanning both.** One component on-manifold, one off.

*Prediction:* the most informative case. If linking is destroyed here at the
layer where the on-manifold component enters a class-relevant subspace, that
localises the destruction to the trained part of the map.

### What would distinguish them

The three placements make different predictions about **the layer at which
linking is lost**. If (a) loses it early, (b) late or never, and (c) at an
intermediate layer traceable to the on-manifold component, that is a coherent
picture: topological destruction is a byproduct of learned invariance, and it
happens where the learning is.

If all three behave identically, the placement does not matter and the result is
about the architecture rather than the training — which would be a weaker but
still reportable finding.

---

## 4. Which projection

**This is the design's most serious vulnerability, and Item 2's findings
constrain it severely.**

Whatever is concluded about projection in the current work constrains what is
measurable here. As it stands:

- A single projection convention, PCA-to-`R^3` or otherwise, is **not usable**.
  The worked example in `results/linking_projected.md` shows one representation
  returning link 0, −1, and +1 from different triples of its own leading PCA
  components.
- The minimum viable approach is to report the **distribution over many random
  projections**, plus the fraction returning zero, plus the projected minimum
  distance for each so artifact-regime values can be excluded.
- Even then, a uniform zero cannot distinguish absence from systematic
  cancellation. That limitation is not fixable by more projections.

**For a transformer specifically**, fixed residual-stream coordinates would be a
natural choice, but they are a basis chosen for interpretability convenience, not
for preserving crossings, and there is no reason to expect them to behave better
than PCA. The same self-inconsistency check should be run before trusting any
such convention.

**Recommendation:** treat the projection as the primary source of uncertainty,
run the cancellation control from `results/cancellation_control.md` on this
setting before collecting any results, and report distributions throughout.

---

## 5. What object to trace

**This may invalidate the circles-plus-projection approach entirely, and it is
flagged as such.**

Above three dimensions, **two circles are always unlinked**. Any two disjoint
closed curves in `R^4` or higher are ambient-isotopic to a trivial pair. So
tracing two circles through a width-768 residual stream is tracing an object
that is, in that space, definitionally unlinked. Whatever the projection returns
is a fact about the projection.

The correct object in higher dimensions is a **generalised Hopf link**: for
`R^4`, a 2-sphere and a circle, satisfying `m + n + 1 = d`. That is the pairing
for which linking number is defined and for which Theorem 4.7 is stated.

Two paths:

**(i) Circles plus projection.** Cheap, and comparable to the paper's own
CIFAR-10 analysis, which also projects to `R^3`. But it measures a
projection-dependent quantity with no invariant meaning in the layer's own
space, and Item 2 shows that quantity is unreliable. **This path can produce
numbers but probably cannot produce conclusions.**

**(ii) The correct generalised object.** Trace an `S^m` and an `S^n` with
`m + n + 1 = d` for the layer's actual width. For width 768 that means, for
instance, a 766-sphere and a circle. This is correct but faces immediate
practical problems: sampling a 766-sphere densely enough to compute a degree
integral is infeasible, and the Gauss-map degree formulation does not reduce to
anything cheap in high dimension.

**Honest assessment:** neither path is currently viable at transformer widths.
Path (i) is measurable but not meaningful; path (ii) is meaningful but not
measurable. **This is the strongest argument for choosing a small network with
narrow hidden layers**, where width 3 or 4 makes the ordinary Hopf link the
correct object and no projection is needed.

A narrow-bottleneck MNIST autoencoder or classifier with a width-3 layer would
let the tracer operate in the regime where every measurement in this project has
been reliable. That is a real experiment. Tracing circles through a 768-wide
residual stream is not, on current evidence.

---

## 6. Hypotheses and what would bear on them

Two competing accounts, plus the outcome that would fit both.

**H1 — Preservation.** Trained networks preserve topological structure they were
never asked to change. Nothing in the loss rewards altering the linking of
arbitrary curves, and generic smooth maps preserve linking where they are
locally invertible.

*Favoured by:* linking preserved through most layers, with loss occurring only
where the network provably reduces rank; preservation stronger off-manifold than
on; preservation stronger in wider layers.

**H2 — Incidental destruction.** Networks destroy topological structure as a
byproduct of building task-relevant invariances. A classifier's job is to
collapse variation that does not affect the label, and such collapse is exactly
the rank-deficiency Lemma 3.5 shows forces intersections.

*Favoured by:* linking destroyed early and consistently; destruction stronger
on-manifold than off; destruction correlated with layer-wise rank collapse or
with the layer where class information becomes linearly decodable.

**The outcome consistent with both, which is where this kind of experiment
usually lands:** linking is destroyed at layer 1 in every configuration, at every
scale and placement. That is compatible with H2 (the first layer already builds
invariances) and with H1 plus an artifact (the first affine map is rank-deficient
in a way that has nothing to do with training). **Distinguishing them requires a
control: the same measurement on an untrained network at initialisation.** If
trained and untrained networks destroy linking identically, the result is about
architecture, not learning, and neither hypothesis is supported.

That control should be built in from the start, not added later.

---

## 7. What would make the result uninteresting

Stated in advance, so it cannot be rationalised afterwards.

**The most likely uninteresting outcome: linking is destroyed at layer 1 in
every configuration, regardless of scale and placement, and the untrained
control behaves identically.** In that case we have learned that a rank-reducing
affine map destroys a one-dimensional link, which is Lemma 3.5 and was known
before the experiment. The finding would be "networks have layers that reduce
rank", which is not about topology.

Given this project's own results — fold layer 1 in **all 34** separating runs,
across every parametrization, corrugation amplitude, and both G.1 readings —
**this outcome should be considered the default expectation rather than a
surprise.** Layer-1 immediacy has been the single most robust observation in the
entire project, and there is no reason to expect an imaginary link in an
unrelated network to behave differently.

Other uninteresting outcomes:

- **Projection-dominated results.** If the answer changes with the projection,
  as Item 2 suggests it will above width 3, then no conclusion about the network
  is available and the result is about the measurement.
- **Off-manifold arbitrariness.** If linking is preserved only where the network
  was never trained, the finding reduces to "untrained regions act randomly",
  which is expected and uninformative.
- **Scale-dependence with no interpretable threshold.** If the answer varies with
  link radius but the transition does not align with any property of the data or
  the network, the scale sweep has found a tuning parameter, not a phenomenon.

---

## 8. Open problems, to resolve before implementing

1. **The correct object above `R^3`.** Path (ii) is not computationally viable at
   realistic widths. Either the experiment is confined to narrow layers where
   circles are correct, or a tractable higher-dimensional invariant is needed.
   This is unresolved and is the primary blocker.
2. **Projection reliability.** The cancellation result must be extended to this
   setting before any width > 3 measurement is trusted.
3. **The untrained control.** Required, per section 6, and cheap. Should be
   specified before the main runs rather than after.
4. **What "on the data manifold" means operationally for MNIST.** Interpolating
   between digit images produces images that are not digits. Whether that counts
   as on-manifold needs a stated definition, since the entire placement axis
   depends on it.

**Recommendation:** if this is pursued, start with a narrow-bottleneck MNIST
classifier at width 3 or 4, where the invariant is defined, no projection is
needed, and the measurement machinery in `src/linking.py` applies unchanged.
That version is implementable now. The transformer version is not.
