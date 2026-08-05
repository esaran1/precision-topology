# Findings

In trained tanh networks on linked-tori data, reducing activation precision to
bfloat16 was associated with 56.8787% ± 4.9153% final-layer vector
non-injectivity, versus 0.3455% ± 0.3488% at float32.

The effect was specific to the only bounded activation tested: pooled bfloat16
excess over initialization was +43.0156% ± 3.0951% for tanh, versus −25.7277%
± 3.6525% for leaky-ReLU and −15.7512% ± 4.2949% for ReLU. The pooled tanh
figure is secondary to the width-resolved estimates below.

The conjecture's operational precondition is therefore present at practical
precision in these trained networks: reducing activation-output precision can
make the observed layer map genuinely non-injective. Width-matched
initialization baselines indicate that much of the effect develops during
training rather than being fixed entirely by initialization.

Control blobs nevertheless collapsed more strongly than linked tori, so the
effect tracks class separability more than topological complexity.

Unless stated otherwise, each mean and sample standard deviation is computed
across five seed-level averages. Within a seed, accepted architecture-layer
rows are averaged with equal weight. Failed runs never enter these summaries.

## Activation specificity

The JMLR construction is specifically a clipping mechanism: outputs accumulate
against the finite asymptotes of tanh and round to the same endpoint. It
therefore appears to require boundedness as well as exact-arithmetic
injectivity. The three tested activations form a partial factorial comparison:

| Activation | Injective in exact arithmetic | Bounded | Observed bfloat16 excess |
|---|:---:|:---:|---:|
| tanh | yes | yes | +43.0156% ± 3.0951% |
| leaky-ReLU | yes | no | −25.7277% ± 3.6525% |
| ReLU | no | no | −15.7512% ± 4.2949% |

Only the bounded, injective activation had positive excess. Leaky-ReLU is the
control that separates boundedness from injectivity: it is injective but has no
asymptote or representational ceiling against which outputs can accumulate.
Boundedness, not injectivity alone, is therefore the property this mechanism
appears to require, as the JMLR clipping construction predicts.
This role was not part of a preregistered factorial design; it emerged from the
results. A paired, matched-architecture tanh-minus-ReLU comparison was
+63.3474% ± 1.9422%. The pattern is consistent with the clipping construction's
boundedness account, but it does not establish that account causally.

ReLU also behaves differently at float32. Its linked-tori final-layer vector
collision was 2.3640% ± 3.4649%, compared with 0.2385% ± 0.3273% at
initialization and 2.1256% ± 3.3701% excess. Tanh's float32 excess was 0.3455%
± 0.3488%; leaky-ReLU's was −0.9068% ± 1.3220% because training reduced its
initialization collisions. These float32 excess directions do not clear the
seed-variance rule defined below.

Simulated fixed-point collision metrics are reported only for tanh. ReLU and
leaky-ReLU outputs are unbounded, so fixed-point quantization would require a
free per-tensor scale; those rows are omitted rather than made incomparable by
an improvised scale.

### Untested prediction

The boundedness account generates a specific falsifiable prediction: other
bounded, injective activations should show positive precision-induced excess
like tanh. Sigmoid and softsign are direct tests because both are bounded and
strictly monotonic. A bounded, injective activation with null or negative excess
would count against this account. Hard-tanh would be a useful bounded control,
but not a direct test of this prediction: its flat tails make it non-injective
in exact arithmetic. This prediction is post hoc and untested here; testing it
would require one additional activation sweep but no new methodology.

## Width-resolved initialization baseline

The initialization baseline is strongly width-dependent, as expected for
vectors with different dimensions:

| Width | All-init baseline | Trained collision | Matched baseline | Matched excess | Excess seed n |
|---:|---:|---:|---:|---:|---:|
| 5 | 56.0400% ± 8.4149% | 80.7583% ± 2.2480% | 55.5854% ± 9.6393% | 25.1729% ± 11.1080% | 4 |
| 15 | 12.2000% ± 8.6410% | 60.1625% ± 3.5662% | 12.2000% ± 8.6410% | 47.9625% ± 11.9408% | 5 |
| 30 | 0.0700% ± 0.0727% | 50.6625% ± 3.2947% | 0.0700% ± 0.0727% | 50.5925% ± 3.2979% | 5 |
| 50 | 0.0050% ± 0.0112% | 43.4225% ± 2.8252% | 0.0050% ± 0.0112% | 43.4175% ± 2.8248% | 5 |

The all-initialization column uses deterministic step-zero recomputation for
all five seeds and all four depths; no training was rerun. Trained and matched
columns remain restricted to accepted runs. Width 5 consequently has four
seed-level excess estimates because seed 2 failed the training gate at every
depth; no value is imputed. The previously reported pooled initialization
baseline of 13.8630% ± 5.9025% mixes width regimes and is secondary rather than
the primary comparator.

Excess is non-monotonic in width and peaks at width 30. Width 5 is the least
reliable point: it has only four seed-level excess estimates, and its 55.5854% ±
9.6393% matched initialization baseline leaves limited headroom for measurable
excess, so its lower value may be a ceiling artifact rather than a genuine
architectural effect. Widths 30 and 50 have near-zero matched baselines of
0.0700% ± 0.0727% and 0.0050% ± 0.0112%, respectively, and therefore give the
cleanest excess estimates. Four widths with these error bars do not support a
trend claim.

## Vector-collision structure

Every bfloat16 collision group in the full accepted sweep was class-pure:
100.0000% ± 0.0000% for each activation and each dataset. For linked-tori tanh
final layers, 63.7010% ± 4.5352% of inputs belonged to a collision group. Across
architecture rows, group-size mean, median, and maximum were respectively
12.5303 ± 2.5980, 3.7704 ± 0.1795, and 225.2821 ± 82.7277.

The purity value is not an independent finding. It is near-forced by margin, and
the earlier reading of this section -- that reduced precision collapses
within-class representations while preserving class separation -- is withdrawn.
The acceptance gate required exactly 100% training accuracy and at least 99%
evaluation accuracy. A between-class collision means two inputs of different
classes map to identical representations, which makes them indistinguishable to
any deterministic downstream classifier and forces at least one of them to be
misclassified. The gate therefore entails a strictly positive between-class
separation in float32 at every layer. Class purity under quantization follows
wherever that separation exceeds the local quantization step, so observing it
mostly restates the selection rule together with a resolution comparison. It is
not evidence about a mechanism.

The margin analysis in `results/between_class_margin.md` makes the "wherever"
precise across all 75 accepted linked-tori tanh runs and 3,682 layer-quantizer
rows. Expressing the minimum between-class Chebyshev separation in units of the
local quantization step, 3,627 of 3,682 rows have margin of at least one step.
All 55 rows below one step are fixed-4, the coarsest tested quantizer, at widths
5 and 15. Margin below one is a necessary condition for impurity with zero
exceptions: all 15 impure rows have margin below one, and none of the 1,650 rows
at or above one step is impure. It is not sufficient: 40 rows fall below one step
yet stay fully pure, because sharing a cell requires the pair to fall within the
same cell boundaries rather than merely to be closer together than the cell is
wide. The margin criterion consequently over-predicts impurity, flagging 55 rows
at risk where 15 contain an actual between-class collision. Counting
between-class pairs that quantize to identical vectors separates the two groups
exactly, at 1,690 pure versus 15 impure with no errors in either direction. The
exceptions therefore locate exactly where margin fails to exceed resolution,
rather than showing that purity was vacuous.

Final-layer margin is close to invariant in width. At fixed-4 the final-layer
margin is 12.2968, 12.5523, 12.6345, and 13.5179 steps at widths 5, 15, 30, and
50: width varies by a factor of ten while the margin changes by a factor of
1.0993. The invariance is in the raw post-activation Chebyshev distance
(1.5371 ± 0.2848, 1.5690 ± 0.2379, 1.5793 ± 0.3139, and 1.6897 ± 0.3735 at those
widths), measured before any quantizer, so it is not an artifact of the constant
fixed-point step; bfloat16, whose step is data-dependent, gives an independent
ratio of 1.097. This is an open observation with no mechanism offered here. It
suggests training targets a characteristic separation relative to representation
scale rather than exploiting the additional width available, but this analysis
does not establish that, and four widths all at or above the theoretically
sufficient width cannot support the claim. It should be rechecked at widths 3
to 8.

This sits alongside a width-dependence that runs the other way. Between-class
margin is close to width-invariant, while within-class vector-collision excess
is not: it is non-monotonic in width and peaks at width 30 (50.5925% ± 3.2979%).
These are different quantities -- margin is a between-class minimum over pairs,
collision excess is a within-class count -- so there is no contradiction between
them. The implication is nonetheless worth stating: if the minimum between-class
separation barely moves across a tenfold change in width, the width-dependence
of collision excess does not originate in between-class geometry. Within-class
collision is additionally subject to pigeonhole pressure that varies strongly
with output dimension, which the matched initialization baselines show directly
at 55.5854% ± 9.6393% for width 5 against 0.0050% ± 0.0112% for width 50. That
pressure is a property of how many distinct representable vectors the layer
affords, not of how the two classes sit relative to one another. Reconciling the
two measurements is owed to the redesign and is not resolved here; no collision
value in this document is changed by the margin analysis.

Layer position governs how far the gate's entailment reaches. Post-hoc
quantization at a hidden layer is counterfactual: the trained classifier
consumed the unquantized float32 activation, so no hidden-layer quantized
representation was ever required by the gate to be class-separable. Hidden-layer
purity was never constrained by the acceptance criterion, which is why all 15
impure rows occur there (layers 1-3, width 5, depths 6, 8, and 10, seeds 0, 1,
and 3; lowest observed purity 97.1631%). The final hidden layer is different: it
is what the linear readout consumes, and its minimum margin never falls below
4.9174 steps at any tested precision, with the smallest bfloat16 final-layer
margin at 157.3581 steps. Final-layer purity reflects margin exceeding
resolution. The bfloat16 figure quoted above is accurate as a measurement, and
no bfloat16 row anywhere falls below one step.

The within-class collision measurements elsewhere in this document are
unaffected. Collision rates, group counts, group sizes, and excess over
initialization remain valid measurements; only the standalone purity claim is
withdrawn.

## Saturation census

Across all accepted linked-tori tanh configuration-layers, paper-criterion
saturation increased from 0.0000% ± 0.0000% at float64 and 0.000309% ± 0.000427%
at float32 to 0.4257% ± 0.1294% at IEEE float16, 1.6063% ± 0.4596% at the
paper's `2^-9` half threshold, and 3.0363% ± 0.5482% at bfloat16. The paired
bfloat16-minus-float32 difference was 3.0360% ± 0.5481%. Simulated fixed-8
matched bfloat16 at 3.0363% ± 0.5482%; fixed-6 and fixed-4 rose to 9.2964% ±
0.8680% and 18.9002% ± 1.3129%.

Saturation is concentrated toward the classifier. In linked-tori tanh networks,
the final hidden layer averaged 16.5752% ± 3.2417% paper saturation, compared
with 0.7851% ± 0.2036% over non-final layers. The final-layer per-unit median
was 12.5280% ± 5.2741%, the per-unit maximum was 48.0328% ± 7.9901%, and
13.1300% ± 5.2129% of units saturated on more than half their inputs. Thus the
pooled result is neither uniform across units nor solely a handful of permanently
pinned units.

Distance from the output is more explanatory than a simple monotone layer
story. At distances zero, one, and two from the classifier, linked-tori
bfloat16 vector collision was respectively 56.8787% ± 4.9153%, 16.8505% ±
4.7186%, and 7.3015% ± 3.7191%, while paper saturation was 16.5752% ± 3.2417%,
1.2212% ± 0.4260%, and 1.0014% ± 0.4528%. The strong final-layer concentration
is consistent with a classification-confidence explanation. Vector collision
also remains larger than exact clipping alone would predict in earlier layers,
so general quantization coarseness is a second mechanism.

The pilot's layer-2 blip was not purely single-seed noise. Absolute layer 2 had
1.3740% ± 0.1950% paper saturation, versus 0.5352% ± 0.1538% at layer 3 and
0.9075% ± 1.0015% at layer 5. Its non-monotonicity persists, but the layer-5
direction does not clear the seed-variance rule and the dominant effect remains
position near the output.

## Paper proxy versus exact rounding

The criterion ambiguity is load-bearing. Across linked-tori tanh layers, the
paper proxy gave 3.0363% ± 0.5482% at bfloat16 while the closer exact-rounding
criterion gave 1.6063% ± 0.4596%. In the final hidden layer the values were
16.5752% ± 3.2417% and 8.5670% ± 2.7929%. The much larger ratio seen in the
single-seed pilot did not generalize, but both definitions remain primary because
the sweep-level difference is still roughly twofold.

The source's format ambiguity matters as well. IEEE float16 (`delta=2^-11`)
gave 0.4257% ± 0.1294% saturation, the paper's stated half value (`delta=2^-9`)
gave 1.6063% ± 0.4596%, and bfloat16 (`delta=2^-8`) gave 3.0363% ± 0.5482%.
The paper-half paper threshold equals the bfloat16 exact threshold by construction.

## Training dynamics

The representative subset was depth 6, width 30, all three activations, both
datasets, and five seeds. For linked-tori tanh final layers, initialization had
0.0000% ± 0.0000% paper saturation, 0.0000% ± 0.0000% exact saturation, and
0.0000% ± 0.0000% bfloat16 vector collision. The first observed all-seed
accuracy plateau was step 200, the 10% checkpoint: training and evaluation
accuracy were already 100.0000% ± 0.0000%. At that checkpoint, vector collision
was 26.7400% ± 10.6906% and paper saturation was 15.1067% ± 15.2022%.

Vector collision continued to 43.4400% ± 10.8292% at the final checkpoint;
40.5176% ± 11.5939% of each seed's total observed collision increase occurred
after the accuracy plateau. Paper and exact saturation reached 27.4227% ±
19.1404% and 15.9403% ± 16.8093% at the final checkpoint. The trajectory
supports continued post-plateau collision growth, but the 10% paper-saturation
direction does not clear the seed-variance rule.

Onset is unresolved. Accuracy was already 100.0000% ± 0.0000% at the first
post-initialization checkpoint, so everything needed to locate the transition
falls between steps 0 and 200. The existing checkpoints cannot distinguish an
early jump from gradual growth within that interval. Resolving onset would
require checkpoints about every 20 steps or finer through step 200; no denser
training was run for this follow-up.

## Control dataset: separating two mechanisms

Blob final layers at float32 had 53.7006% ± 2.1486% tanh vector collision with
0.0000% ± 0.0000% paper saturation and 0.0000% ± 0.0000% exact saturation.
Linked-tori tanh at float32 had 0.3455% ± 0.3488% vector collision. Thus blobs
show bit-identical float32 vectors without clipping to ±1 and without reducing
precision below the training format.

There appear to be two mechanisms. General representational collapse dominates
on easily separable data and does not require reduced precision; precision-
induced clipping is the mechanism described by the conjecture. The linked-tori
setting largely isolates the second because the first is small there, although
it is not completely absent. The effect is not specific to topologically
entangled data and may reflect within-class compression in any easy classifier.

The paired blob-minus-tori analysis uses the same seeds and only architecture-
layer rows accepted for both datasets. Under that matched estimand, bfloat16
paper saturation was 23.5296% ± 3.4718% on blobs and 3.0363% ± 0.5482% on tori,
a gap of 20.4932% ± 3.7752%. Final-layer bfloat16 vector-collision excess was
82.4909% ± 6.5347% versus 43.0156% ± 3.0951%, a gap of 39.4753% ± 5.6362%.
Float32 final-layer vector collision was 54.1285% ± 2.0079% versus 0.3455% ±
0.3488%, a gap of 53.7830% ± 2.0083%. All three gaps clear the seed-variance
rule.

These matched blob summaries differ from the earlier separately aggregated
values because the estimand changed, not because any underlying number changed.
With all accepted blob architectures included independently, bfloat16 paper
saturation was 21.8023% ± 0.4641%, bfloat16 final-layer excess was 78.6675% ±
2.5659%, and float32 final-layer collision was 53.7006% ± 2.1486%. Restricting
blobs to rows with an accepted linked-tori partner gives 23.5296% ± 3.4718%,
82.4909% ± 6.5347%, and 54.1285% ± 2.0079% respectively.

## Training failures

Of 480 exact grid runs, 447 passed the strict gate and 33 were excluded. Every
failure had width 5. Exact failure counts were: tanh linked tori 5 of 80, tanh
blobs 0 of 80, ReLU linked tori 18 of 80, ReLU blobs 3 of 80, leaky-ReLU linked
tori 7 of 80, and leaky-ReLU blobs 0 of 80. The status artifacts retain every
accuracy and failure reason.

## Specification corrections

Thresholds use the exact identity
`atanh(1-delta) = 0.5 * (ln(2-delta) - ln(delta))`. The asymptotic form
`0.5 * ln(2/delta)` was rejected as the primary calculation; its maximum
configured deviation is 0.0159 at fixed-4. Direct `atanh`, where representable,
and the asymptotic value are cross-checks only.

The initially specified pooled scalar collision rate was rejected because it
measured format cardinality. The replacement reports within-unit distributions,
complete-vector collisions, collision-group purity and size, and excess relative
to the untrained network on identical evaluation inputs.

Known mismatch between the metric and the framing. The collision metric counts
inputs whose representations coincide, which is a *within-class* quantity in
practice: it is closest in spirit to the per-class Betti numbers tracked by the
JMLR line of work, which are intrinsic descriptors of one class at a time.
Linking number, which this project's framing invokes, is a *relational,
between-class* invariant describing how two class manifolds are embedded
relative to one another. These are different quantities, and a change in the
former does not imply a change in the latter. The collision measurements
reported here should not be read as measurements of linking or unlinking. This
mismatch is recorded here as known and is to be corrected in the redesign, which
will add primary between-class metrics and a separate linking-number estimator.

Pre-activations remain genuine float32 training outputs, but saturation
comparisons promote them to float64 before applying the float64 threshold. A
regression test pins the boundary where PyTorch would otherwise round the scalar
threshold to float32.

## Statistical adequacy

This subsection uses five seed-level values for every comparison. As a simple,
reproducible descriptive rule, a directional comparison is called clear only
when its sample SD is no more than half the absolute mean. This is not a
hypothesis test or confidence interval.

The following do not support directional claims at the present seed count:
10%-checkpoint paper saturation was 15.1067% ± 15.2022%, ReLU float32 excess
was 2.1256% ± 3.3701%, leaky-ReLU float32 excess was −0.9068% ± 1.3220%, and
absolute-layer-5 saturation was 0.9075% ± 1.0015%.

The central paired comparisons clear the rule comfortably. The matched
bfloat16 tanh-minus-ReLU excess differential was 63.3474% ± 1.9422%, and the
linked-tori bfloat16-minus-float32 paper-saturation difference was 3.0360% ±
0.5481%. The paired blob-minus-tori gaps were 20.4932% ± 3.7752% for bfloat16
paper saturation, 39.4753% ± 5.6362% for bfloat16 final-layer excess, and
53.7830% ± 2.0083% for float32 final-layer collision. The pooled linked-tori
tanh bfloat16 excess of 43.0156% ± 3.0951% also clears the rule, though the
width-resolved values are the primary summaries.

## Reasons this result might be wrong

- The boundedness interpretation is post hoc. Only one bounded activation was
  tested, and the activation set was not designed as a factorial test of
  boundedness and exact-arithmetic injectivity. The sigmoid/softsign prediction
  needs a new sweep before this interpretation can be distinguished from a
  tanh-specific effect.
- Initialization is a first-order confound. Main runs retain PyTorch's default
  `nn.Linear` Kaiming-uniform initialization with `a=sqrt(5)`, gain `1/sqrt(3)`,
  and default uniform bias. No alternative initialization scale was tested.
- The strict success gate selects activation regimes. All 33 failed runs were
  width 5; excluding them is required by the scientific question but changes
  the architecture mixture across seeds.
- Full-batch Adam at a fixed learning rate and 2,000 steps may encourage
  class-cluster compression differently from the paper's training procedure.
- The blob result suggests the measured collapse may be ordinary classifier
  confidence rather than a response to linked geometry.
- The linked solid-tori construction has Gauss linking integral
  `-1.000010280928`, but the paper does not specify enough geometry for
  point-for-point D-II reproduction. Tube radius and sampling density may alter
  activation scale.
- Threshold counts operate on stored float32 pre-activations. Promotion makes
  the comparison itself precise, but information already rounded during the
  forward pass cannot be recovered.
- Collision rates depend on evaluation-set size and width. The initialization
  baseline controls the main pigeonhole effect but cannot make different vector
  dimensions intrinsically identical experiments. The A1 breakdown shows that
  this dependence is large, and the width-5 trained excess has only four
  seed-level estimates.
- The collision-group purity result is withdrawn as a standalone finding
  because it is near-forced by margin. Every accepted run was gated on 100%
  training and at least 99% evaluation accuracy. Between-class collision implies
  indistinguishability to any deterministic downstream classifier and therefore
  misclassification, so the gate entails strictly positive float32 between-class
  separation at every layer; purity under quantization then follows wherever
  that separation exceeds the local quantization step. Reporting purity alone
  restates the selection rule plus a resolution comparison and carries no
  information about the mechanism. Two qualifications bound the entailment:
  hidden-layer quantization is counterfactual, since the classifier consumed
  float32 activations, so hidden-layer purity was never gate-constrained; and
  the gate required >=99% rather than exactly 100% evaluation accuracy, with 442
  of 447 accepted runs at exactly 100% and five at 99.90% or 99.95%. The
  associated within-class collision numbers remain valid as measurements. See
  `results/between_class_margin.md` for the margin table and the 15 fixed-4 rows
  that are not fully pure.
- Labels provide only a coarse two-class partition. Purity does not identify
  what within-class geometry was
  discarded.
- Dynamics are too sparse before step 200 to identify onset or its relation to
  the moment accuracy first reaches its plateau.
