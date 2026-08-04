# Findings

In the preregistered pilot, bfloat16 retained 1,225 of 2,000 trained final-layer tanh vectors versus 2,000 of 2,000 at float32 and initialization; across the accepted linked-tori sweep, paper-criterion tanh saturation was 3.0363% ± 0.5482% at bfloat16 versus 0.000309% ± 0.000427% at float32.

Unless stated otherwise, each mean and sample standard deviation is computed
across five seed-level averages. Within a seed, accepted architecture-layer
rows are averaged with equal weight. Failed runs never enter these summaries.

## Headline: vector-level non-injectivity

For the final hidden layer of trained linked-tori tanh networks, bfloat16 vector
collision was 56.8787% ± 4.9153%, compared with an untrained-network baseline
of 13.8630% ± 5.9025%; the excess was 43.0156% ± 3.0951%. At float32, the
corresponding trained rate was 0.3455% ± 0.3488%, the baseline was 0.0000% ±
0.0000%, and the excess was 0.3455% ± 0.3488%.

Every bfloat16 collision group in the full accepted sweep was class-pure:
100.0000% ± 0.0000% for each activation and each dataset. For linked-tori tanh
final layers, 63.7010% ± 4.5352% of inputs belonged to a collision group. Across
architecture rows, group-size mean, median, and maximum were respectively
12.5303 ± 2.5980, 3.7704 ± 0.1795, and 225.2821 ± 82.7277. This pattern is
consistent with reduced precision collapsing within-class representations while
preserving class separation. It does not by itself support a claim about a
topological change.

The conjecture's operational precondition therefore occurs in these trained
networks: reducing activation-output precision can make a layer map genuinely
non-injective on the observed evaluation set. The initialization baseline shows
that much of the linked-tori tanh effect develops during training rather than
being fixed entirely by initialization.

## Saturation census

Across all accepted linked-tori tanh configuration-layers, paper-criterion
saturation increased from 0.0000% ± 0.0000% at float64 and 0.000309% ± 0.000427%
at float32 to 0.4257% ± 0.1294% at IEEE float16, 1.6063% ± 0.4596% at the
paper's `2^-9` half threshold, and 3.0363% ± 0.5482% at bfloat16. Simulated
fixed-8 matched bfloat16 at 3.0363% ± 0.5482%; fixed-6 and fixed-4 rose to
9.2964% ± 0.8680% and 18.9002% ± 1.3129%.

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
0.9075% ± 1.0015% at layer 5. Its non-monotonicity persists, but the dominant
effect is still position near the output.

## Paper proxy versus exact rounding

The criterion ambiguity is load-bearing. Across linked-tori tanh layers, the
paper proxy gave 3.0363% ± 0.5482% at bfloat16 while the closer exact-rounding
criterion gave 1.6063% ± 0.4596%. In the final hidden layer the values were
16.5752% ± 3.2417% and 8.5670% ± 2.7929%. The pilot's larger gap—10.3550%
paper versus 1.1050% exact—did not generalize at that ratio, but both definitions
remain primary because the sweep-level difference is still roughly twofold.

The source's format ambiguity matters as well. IEEE float16 (`delta=2^-11`)
gave 0.4257% ± 0.1294% saturation, the paper's stated half value (`delta=2^-9`)
gave 1.6063% ± 0.4596%, and bfloat16 (`delta=2^-8`) gave 3.0363% ± 0.5482%.
The paper-half paper threshold equals the bfloat16 exact threshold by construction.

## Training dynamics

The representative subset was depth 6, width 30, all three activations, both
datasets, and five seeds. For linked-tori tanh final layers, initialization had
0.0000% ± 0.0000% paper saturation, 0.0000% ± 0.0000% exact saturation, and
0.0000% ± 0.0000% bfloat16 vector collision. At 10% of updates, these had risen
to 15.1067% ± 15.2022%, 6.0037% ± 7.6832%, and 26.7400% ± 10.6906%. Training
and evaluation accuracy were already 100.0000% ± 0.0000% at that checkpoint.

The same three metrics continued upward after the accuracy plateau, reaching
27.4227% ± 19.1404%, 15.9403% ± 16.8093%, and 43.4400% ± 10.8292% at the final
checkpoint. The effect therefore emerges within the first 10% of sampled
training and continues strengthening after classification accuracy saturates;
the present checkpoint spacing cannot locate its onset more precisely.

## Control dataset

The control blobs show a stronger effect, not a weaker one. Across tanh layers,
bfloat16 paper saturation was 21.8023% ± 0.4641% on blobs versus 3.0363% ±
0.5482% on linked tori; exact saturation was 14.1220% ± 0.6705% versus 1.6063%
± 0.4596%. Final-layer bfloat16 vector collision on blobs was 99.8088% ±
0.0578%, with a 21.1413% ± 2.5252% baseline and 78.6675% ± 2.5659% excess.

Even float32 blob final layers had 53.7006% ± 2.1486% tanh vector collision
while their exact and paper saturation fractions were both 0.0000% ± 0.0000%.
This separates general finite-precision coarsening from clipping to ±1. The
observed mechanism is not specific to topologically entangled data and may
largely reflect within-class representation compression in any easy classifier.

## ReLU comparison

ReLU behaves differently because it is non-injective without precision loss.
On linked-tori final layers at float32, ReLU vector collision was 2.3640% ±
3.4649%, compared with 0.2385% ± 0.3273% at initialization and 2.1256% ±
3.3701% excess. Tanh's float32 excess was 0.3455% ± 0.3488%; leaky-ReLU's was
-0.9068% ± 1.3220% because training reduced its initialization collisions.

At bfloat16, linked-tori ReLU collision was 19.6731% ± 5.2686%, below its
35.4243% ± 5.2786% baseline, for -15.7512% ± 4.2949% excess. Leaky-ReLU was
9.3988% ± 2.6008% versus a 35.1265% ± 5.6856% baseline, or -25.7277% ±
3.6525% excess. Tanh instead had +43.0156% ± 3.0951% excess. This is consistent
with tanh training moving representations toward coarser output regions while
ReLU-family training often separates representations that were already colliding
at initialization.

Simulated fixed-point collision metrics are reported only for tanh. ReLU and
leaky-ReLU outputs are unbounded, so fixed-point quantization would require a
free per-tensor scale; those rows are omitted rather than made incomparable by
an improvised scale.

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
  dimensions intrinsically identical experiments.
- All collision groups were class-pure, but labels provide only a coarse
  two-class partition. Purity does not identify what within-class geometry was
  discarded.
