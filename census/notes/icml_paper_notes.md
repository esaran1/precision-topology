# Notes on Ren and Lim (ICML 2026)

Source: Junyu Ren and Lek-Heng Lim, ["Low-dimensional topology of deep
neural networks"](https://arxiv.org/abs/2606.31856), arXiv:2606.31856v1,
30 June 2026; Proceedings of the 43rd International Conference on Machine
Learning, PMLR 306. These notes cover the full paper and appendices.

## Scope of the paper

The paper studies *extrinsic* topology: how two class manifolds are embedded
relative to one another in an ambient representation space. Its main invariant
is linking number. This differs from the intrinsic, per-class Betti numbers in
Naitzat, Zhitnikov, and Lim (JMLR 2020). The paper deliberately fixes the
representation width so that increasing ambient dimension cannot erase the
topological obstruction by itself.

The theory is about continuous class manifolds and continuous network maps,
not merely a finite training or evaluation sample. Its classification result
assumes a linear readout. For binary classification, Appendix A.2 proves that a
two-logit affine head perfectly classifies the class supports exactly when their
feature representations are linearly separable.

## Architecture ordering and its conditions

Table 1 gives the following ordering *only for the topological transformations
studied in the paper and under a fixed width constraint*:

1. Ambient homeomorphisms (invertible/flow-based models and Neural ODEs) can
   deform shapes but cannot merge components, fill holes, or unlink a
   two-component link.
2. Width-`d` feedforward networks with continuous coordinate-wise monotonic
   activations, and autoencoders with a width-`d` monotonic bottleneck, can
   merge components and fill holes but cannot unlink the linked manifolds in
   the theorem.
3. Width-`d` feedforward networks with a suitable nonmonotonic activation,
   discrete ResNets, and the paper's pure-transformer construction can perform
   the coordinate-fold operation used to unlink the examples.

Thus the table supports the scoped preorder

`ambient homeomorphism < monotonic feedforward/bottleneck < nonmonotonic
feedforward = ResNet = pure transformer`

for these witness transformations. It is not a general ordering of all
functions representable by these architecture families.

The hypotheses differ by architecture:

- **Monotonic feedforward impossibility (Theorem 4.7).** The class supports are
  disjoint, closed, oriented submanifolds `M^m, N^n` of `R^d`, with
  `m + n + 1 = d` and nonzero linking number. Every intermediate representation
  has dimension at most `d`; affine maps are followed by continuous
  coordinate-wise monotonic activations. The output classes are tested by a
  linear readout. Under these assumptions, perfect classification is
  impossible at width `d`, regardless of depth. Theorem 3.7 is the
  `m = n = 1, d = 3` ReLU/Hopf-link case.
- **Autoencoder result (Corollary A.5).** The data lie in a `d`-dimensional
  affine subspace of the input space, the bottleneck is width `d`, and the
  encoder/decoder layers through the pre-final feature map are monotonic
  feedforward layers. A wider input or final decoder does not remove the
  bottleneck obstruction.
- **Nonmonotonic feedforward construction (Section 5.1 and Appendix F.2).** The
  data are compact and the scalar activation has a strict local extremum on an
  open interval. Affine shifts and rescalings move the relevant coordinate into
  that interval, after which the nonmonotonicity supplies a coordinate fold.
- **ResNet construction (Theorem 5.2).** A discrete residual block synthesizes
  `|x| = x + 2 ReLU(-x)`. This is a constructive escape from the monotonic
  feedforward hypothesis. It does not apply to Neural ODE flows, whose maps are
  ambient homeomorphisms.
- **Transformer construction (Theorem 5.3).** A pure transformer without
  residual connections or layer normalization uses two copied tokens,
  distinct positional encodings, and a single attention head to produce a
  local V-shaped coordinate map after affine rescaling. The claim is a local
  fold on the compact data domain, not a global approximation of `|x|`.

## The obstruction and the width at which it exists

For two disjoint oriented curves `X, Y` in `R^3`, the linking number is the
degree of the Gauss map `(x, y) -> (x-y)/||x-y||`, equivalently the Gauss
double integral or half the signed crossing count in a regular projection.
The Hopf link has linking number `+1` or `-1`, depending on orientation.

Proposition 3.6 proves that linearly separable class curves must have linking
number zero: each class contracts inside its own convex half-space, giving a
link homotopy to two points. Theorem 3.7 then proves that a width-3 ReLU
feedforward network cannot linearly separate a nontrivially linked pair. If all
intermediate class images remain disjoint, invertible affine maps preserve link
up to sign and ReLU preserves it by a disjoint straight-line homotopy. A
rank-deficient affine map is the only remaining route, but Lemma 3.5 shows that
it forces the two linked class images to intersect. An intersection propagates
through all deterministic downstream layers and prevents perfect separation.

The higher-dimensional theorem replaces the curves with complementary
manifolds `M^m, N^n` in `R^d`, where `m + n + 1 = d`, and defines link as the
degree of the higher-dimensional Gauss map. The same argument applies to any
continuous coordinate-wise monotonic activation, including ReLU, leaky-ReLU,
sigmoid, and tanh. The operative property in this ICML theory is monotonicity,
not boundedness, differentiability, or injectivity.

The exact theoretical threshold is:

- width `d`: the obstruction holds under Theorem 4.7's hypotheses;
- width `d + 1`: the classification obstruction can be avoided. Theorem D.1
  uses width-`d+1` universal approximation to map any finite collection of
  disjoint compact class supports into disjoint scalar intervals.

The width-`d+1` construction need not perform an ambient unlinking in `R^d`;
it produces the continuous class-label function required for classification.
For two one-dimensional linked curves, the classical linking-number
obstruction is specifically a three-dimensional ambient-space phenomenon.
Once their representation lies in `R^4` or higher, this particular invariant
is no longer the complementary-dimension linking invariant of Theorem 4.7.

## What `d + c` means in this source

The paper does **not** define a quantity `c`, state a `d + c` theorem, or bound
an additive empirical range by `c < 5`. I cannot recover the requested meaning
or bound from arXiv:2606.31856v1 without importing information external to the
paper.

The closest paper results are different:

- The exact theorem is the tight `d` versus `d + 1` classification threshold
  above.
- Appendix G.7 studies width expansion empirically for ten copies of linked
  3-spheres in `R^7`. It tests widths 7, 8, 10, 14, 21, 28, 35, and 49, reports
  small finite-seed nonmonotonicities, and observes roughly 99--100% accuracy
  near width `5d`. It also reports 98.5% mean accuracy at width 20 (`4d`) for a
  related `R^5` experiment. These results do not establish an additive constant
  below five.
- The primary Hopf-link experiments compare architectures at width 3; they do
  not sweep widths 3 through 8.

Accordingly, treating widths 3--8 as an empirical `d + c` regime is motivated
by the reviewer feedback, not documented by this paper. It should be labeled
that way. The paper directly supports width 3 as the obstruction regime and
width 4 as theoretically sufficient for classification.

## Definition and estimation of linking number

### Exact/parametrized definition

For disjoint oriented closed curves in `R^3`, Definition 3.2/C.1 uses

`link(X,Y) = (1/(4*pi)) integral_X integral_Y ((x-y) dot (dx cross dy)) /
||x-y||^3`.

It is an integer and equals the degree of the Gauss map. In higher dimensions,
Definition 4.1/C.2 uses the degree of the Gauss map for complementary closed
oriented manifolds. Remark C.4 is important for network layers: a transformed
component may self-intersect and cease to be an embedded manifold, but the link
of the two *parametrizing maps* remains defined as long as the images of the two
different components are disjoint.

### Synthetic layer tracking

Appendix G.8 tracks the Gauss integral and minimum cross-class distance through
a width-3, depth-5 network using the best seed and 200 samples per class. For
ReLU, the minimum distance reaches zero; the reported fractional Gauss values
near that event are explicitly called numerical artifacts because linking
number is undefined once the components intersect and the integral is
ill-conditioned as `||x-y||` approaches zero. GELU and ReLU+skip reach link
zero while keeping a positive minimum distance. The paper does not report a
seed-level uncertainty or a calibrated numerical noise floor for Table 8.

### General point-cloud detector

Appendix H's extrinsic-TDA pipeline for unlabeled manifold parametrizations is:

1. Project the union of the two point clouds to `R^3` with PCA.
2. Build an epsilon-filtered per-class `k`-nearest-neighbor graph (mutual k-NN
   is recommended for heterogeneous density).
3. Construct a spanning forest and the associated fundamental cycle basis for
   each class graph.
4. Evaluate every cross-class pair of basis cycles with midpoint quadrature of
   the polygonal Gauss integral, subdividing edges, and round the result to the
   nearest integer.
5. Return a nonzero basis-pair witness if one exists.

The paper suggests `k` between 6 and 15, epsilon at the 70th percentile of
nearest-neighbor distances, 4 or 8 subdivisions per edge, and minimum cycle
length at least 4. Its CIFAR analysis uses much denser sampling and stricter
parameters, repeats augmentation/detection 11 times, and calls the fraction of
runs with a nonzero witness "linking consistency." It explicitly treats that
result as projection- and sampling-dependent, correlational evidence.

For the census's known Hopf-link core parametrizations, direct polygonal Gauss
quadrature is better matched to Appendix G.8 than the PCA/k-NN cycle detector.
For the sampled solid tori or arbitrary hidden representations, a cycle must be
specified or reconstructed; applying the Gauss integral to an unordered point
cloud is not defined.

## Proven results versus empirical observations

### Proven or constructively established under stated hypotheses

- Proposition 3.6 / Lemma C.6: linear separation implies linking number zero.
- Theorems 3.7, 4.5, and 4.7: width-`d` monotonic feedforward networks cannot
  perfectly separate nonzero-linked complementary class manifolds.
- Lemmas C.7 and C.8: monotonic activations preserve link up to sign while
  cross-class images stay disjoint; rank-deficient affine maps force a
  cross-class intersection.
- Corollary A.5: the corresponding monotonic autoencoder bottleneck result.
- Theorem 5.2: a discrete ReLU ResNet can synthesize the coordinate fold.
- Theorem 5.3: the stated two-token attention construction has a local
  V-shaped fold usable on a rescaled compact domain.
- Theorem D.1: width `d + 1` is sufficient to classify arbitrary disjoint
  compact class supports; Theorem E.1 gives the matching universal-
  approximation lower bound for continuous coordinate-wise monotonic
  activations.

### Empirically observed, not implied as an optimization guarantee

- At width 3, ReLU feedforward models remain below the Hopf-link accuracy
  ceiling in the reported runs, while some moderate-depth GELU and ReLU-ResNet
  runs reach perfect accuracy.
- Accuracy differences between monotonic and folding-capable architectures on
  higher-dimensional, multi-copy links vary with dimension and copy count.
- Appendix G.7's finite-seed width-expansion curve is not monotone at every
  tested width even though performance improves overall.
- Appendix G.8's layerwise link trajectory is one best-seed numerical study,
  not an uncertainty-calibrated estimator study.
- CIFAR-10 linking is detected only after projection to PCA-3D. Its association
  with confusion and activation choice is explicitly correlational and can be
  confounded by projection, sampling, local geometry, and semantic similarity.
- The theory concerns expressivity. It does not predict which representable
  transformation gradient-based optimization will find.

## Contradictions, refinements, and reframings for the current census

This section is the main design consequence of the paper.

1. **The current widths do not test the theorem's obstruction.** Width 3 is the
   theorem-aligned setting for linked curves in `R^3`. Width 4 is already
   theoretically sufficient for classification, and widths 5, 15, 30, and 50
   are outside the exact obstruction. Widths 3--8 can still be an externally
   motivated empirical transition study, but only width 3 directly tests
   Theorem 3.7. The current pooled headline is dominated by out-of-regime
   widths and cannot be interpreted as evidence about the link obstruction.

2. **The current solid-torus samples are not the theorem's class manifolds.**
   `census/src/data.py` samples volume from two solid tubes around linked core
   circles. The core circles have a valid linking number, but the sampled class
   supports are three-dimensional solids, not the disjoint one-dimensional
   closed oriented curves required by Theorem 3.7. The ICML experiments also
   train on thickened samples, but the theorem and the layerwise link estimator
   concern the underlying parametrized cores. A redesigned topology analysis
   must separately propagate ordered core-circle samples through the trained
   feature map.

3. **The ICML operative variable is monotonicity, not boundedness.** The current
   FINDINGS.md reframes tanh versus leaky-ReLU around boundedness. Under Theorem
   4.7, tanh, leaky-ReLU, and ReLU are all monotonic and all face the same
   width-3 link obstruction while cross-class images remain disjoint. The
   paper's predicted escape activations are nonmonotonic ones such as GELU,
   Swish/SiLU, and Mish. Finite quantization is discontinuous and therefore
   falls outside the theorem, but boundedness by itself is not the ICML
   mechanism.

4. **Within-class collision is not a linking-number measurement.** The JMLR
   paper studies intrinsic, per-class Betti numbers; within-class collapse is
   adjacent to that question. The ICML invariant is relational and
   between-class. The current vector-collision excess is almost entirely
   within-class and cannot be presented as link simplification or unlinking.
   Between-class collision and separation metrics are the relevant collision-
   based diagnostics, while linking number must be estimated separately.

5. **The class-purity result is selection-constrained, but the claimed
   tautology needs two qualifications.** For representations actually consumed
   by a deterministic classifier, an exact between-class collision forces at
   least one error; therefore perfect accuracy forces collision groups to be
   class-pure on that same evaluated set. However, the implemented gate is
   exactly 100% training accuracy and at least 99% evaluation accuracy, not
   exactly 100% on both. In the saved status artifact, 442 of 447 accepted runs
   have 100% evaluation accuracy; five have 99.90% or 99.95%. More importantly,
   the reported bfloat16/fixed-point groups are produced by *post-hoc*
   quantization of a hidden activation, while the classifier consumed the
   original float32 activation. Full-precision accuracy therefore does not
   logically force those post-hoc quantized groups to be pure. The reported
   purity should still be demoted: it is strongly conditioned by the gate and
   is not evidence of topology change, but calling every reduced-precision
   purity value a strict tautology would misdescribe the implementation.

6. **Post-hoc and interleaved quantization are different maps.** The current
   census measures `Q` applied independently to a saved layer output. It does
   not evaluate a network that consumes the quantized representation, and it
   does not interleave quantization between layers. The deterministic inclusion
   `collisions(Q∘F) ⊇ collisions(F)` must hold for exact equality pairs and is
   a necessary regression test. It says post-hoc quantization cannot separate
   a pair that `F` already merged. The paper itself does not discuss
   quantization or floating-point precision. Interleaved quantization is a
   discontinuous perturbation outside Theorem 4.7's continuity hypotheses, so
   its collision-set relation has to be checked empirically as proposed in C3.

7. **A sampled collision test is not a complete unlinking test.** The theorem
   concerns intersections and linking of continuous class supports. A finite
   evaluation set can have no between-class duplicate vectors while the images
   of the underlying continuous cores intersect between sample locations or
   remain linked. C3 can test the stated pair-set inclusion on sampled inputs;
   it cannot by itself establish the presence or absence of a topological
   unlinking.

8. **Linking number is well-founded only at compatible layer dimensions.** For
   two one-dimensional cores, the paper's linking number applies in a
   three-dimensional representation. At hidden widths 4--8, two curves are not
   complementary-dimensional manifolds under `m + n + 1 = d`, so their
   classical linking number is not the invariant in Theorem 4.7. A PCA or other
   projection back to `R^3` would produce a projection-dependent diagnostic,
   not the layer-space linking invariant. Therefore C4 is mathematically direct
   at width 3 only. Extending it to widths 4--8 requires an explicitly labeled
   projection convention or a different invariant; the paper does not supply
   one.

9. **The proposed quantization-unit margin is underspecified for IEEE formats.**
   Fixed-point quantizers have a uniform step, but float16, bfloat16, float32,
   and float64 have exponent-dependent ULP spacing. A vector also has one local
   spacing per coordinate. Before C2, "quantization step" must be defined, for
   example through coordinatewise local ULPs and a stated aggregation or by a
   direct count of representable bins along the cross-class displacement. The
   paper does not define this metric.

10. **The existing success-gated census cannot estimate an obstruction-induced
    failure rate.** Failed runs were excluded by design, and only accepted runs
    entered saturation/collision summaries. The corrected-width study must keep
    all width-level pass/fail records primary. An accepted width-3 run on finite
    samples would not refute the theorem, because the gate tests samples from
    thickened volumes rather than perfect separation of the continuous core
    curves.

## Items that cannot be determined from the paper

- No `d + c` definition, bound on `c`, or additive empirical range appears in
  arXiv:2606.31856v1.
- Appendix G.8 does not state enough estimator detail to reproduce its
  layerwise Gauss-integral discretization exactly, nor does it provide an
  uncertainty estimate across seeds or discretizations.
- The paper does not specify how to assign a classical linking number directly
  to the same pair of one-dimensional curves after embedding them in hidden
  widths greater than three.
- The paper contains no floating-point or quantization analysis and therefore
  does not resolve how an IEEE-format "quantization-unit margin" should be
  defined.
