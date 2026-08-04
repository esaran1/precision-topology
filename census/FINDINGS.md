# Findings

In trained tanh networks on linked-tori data, reducing activation precision to bfloat16 was associated with 56.8787% ± 4.9153% final-layer vector non-injectivity; width-matched excess over initialization was 25.1729% ± 11.1080% at width 5, 47.9625% ± 11.9408% at width 15, 50.5925% ± 3.2979% at width 30, and 43.4175% ± 2.8248% at width 50, with the pooled 43.0156% ± 3.0951% retained only as a secondary summary, versus 0.3455% ± 0.3488% at float32. The effect is activation-specific: ReLU and leaky-ReLU showed negative bfloat16 excess of −15.7512% ± 4.2949% and −25.7277% ± 3.6525%. The tanh–ReLU sign reversal is consistent with a mechanism that supplies non-injectivity to an activation that is injective in exact arithmetic but is unnecessary for one already non-injective; leaky-ReLU is itself injective in exact arithmetic, so injectivity alone is not a sufficient explanation. The conjecture's operational precondition is present at practical precision, but control blobs collapsed more strongly than linked tori, so the effect tracks class separability more than topological complexity.

Unless stated otherwise, each mean and sample standard deviation is computed
across five seed-level averages. Within a seed, accepted architecture-layer
rows are averaged with equal weight. Failed runs never enter these summaries.

## Activation specificity

The mechanism predicts that precision loss should confer non-injectivity on an
activation that is injective in exact arithmetic, such as tanh, while it should
not be needed by ReLU, which is already non-injective. On linked-tori final
layers at bfloat16, tanh had +43.0156% ± 3.0951% excess vector collision, while
ReLU and leaky-ReLU had −15.7512% ± 4.2949% and −25.7277% ± 3.6525%. A paired,
matched-architecture tanh-minus-ReLU comparison was +63.3474% ± 1.9422%. The
sign reversal is consistent with the prediction; it does not establish the
causal story, and leaky-ReLU's negative result shows that exact-arithmetic
injectivity is not by itself enough.

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

Every bfloat16 collision group in the full accepted sweep was class-pure:
100.0000% ± 0.0000% for each activation and each dataset. For linked-tori tanh
final layers, 63.7010% ± 4.5352% of inputs belonged to a collision group. Across
architecture rows, group-size mean, median, and maximum were respectively
12.5303 ± 2.5980, 3.7704 ± 0.1795, and 225.2821 ± 82.7277. This pattern is
consistent with reduced precision collapsing within-class representations while
preserving class separation. It does not by itself support a claim about a
topological change.

The operational precondition therefore occurs in these trained networks:
reducing activation-output precision can make a layer map genuinely
non-injective on the observed evaluation set. The width-matched initialization
baselines show that much of the linked-tori tanh effect develops during training
rather than being fixed entirely by initialization.

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
- All collision groups were class-pure, but labels provide only a coarse
  two-class partition. Purity does not identify what within-class geometry was
  discarded.
- Dynamics are too sparse before step 200 to identify onset or its relation to
  the moment accuracy first reaches its plateau.
