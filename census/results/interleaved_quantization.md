# Post-hoc versus interleaved quantization: collision set relations

Subset: the 18 accepted linked-tori tanh runs at **depth 6** (widths 5, 15, 30,
50; seeds 0-4; width 5 has three accepted seeds because seeds 2 and 4 failed the
training gate). All seeds are recorded per row in
`interleaved_quantization.csv`. This is one depth, deliberately, and is not a
sweep-wide result.

No new training was performed. Every run was reconstructed under its recorded
seeds and verified before use; the maximum deviation of reconstructed accuracy
and saturation from recorded values was **0.0 exactly** across all 18 runs.

## Regimes

- **F** — full precision throughout, no quantization.
- **QF** — full precision forward pass, quantized once at the observation point.
- **G** — quantization applied to the activations at every layer, so each layer
  consumes a quantized representation.

## The detector fires (positive control)

Before running on trained weights, the detector was required to find a pair that
provably collides in F and separates in G. In a hand-built network where layer 2
computes `ReLU(8*u0 - 3.2)`, inputs 0.36 and 0.39 are both below the 0.4
threshold and clamp to exactly 0.0 in full precision. Under fixed-4 the cell
boundary at 0.375 falls between them, sending them to 0.3125 and 0.4375, on
opposite sides of the threshold; the gain of 8 makes the resulting divergence
exceed one cell so it survives requantization. The detector reports exactly one
F-not-G violation, between-class, diverging at layer 1, with QF containment
intact.

The construction required three failed attempts, and the failures are
informative. The first discarded the diverging unit at layer 2, erasing the
separation. The second propagated that unit but thereby destroyed the
F-collision. The third produced a divergence smaller than one quantization cell,
which was rounded back together. The effect needs a specific conjunction: an
F-collision arising from a **clamping** mechanism that quantization can undo, a
cell boundary falling **between** the two inputs while they are still distinct,
and enough downstream **gain** for the divergence to survive requantization.
Nothing in a trained network arranges for these to coincide, which is itself a
reason to expect the phenomenon to be rare.

## Result 1: at the network output, the test is vacuous

As originally specified, the comparison is made at the network output. Both
containments hold in all 126 configuration-quantizer rows:

- `collisions(QF) ⊇ collisions(F)` — **holds everywhere** (correctness check).
- `collisions(G) ⊇ collisions(F)` — **holds everywhere**, with **0** violating
  pairs.

However **`collisions(F)` is empty**: across all 18 runs, all 2,000 evaluation
inputs produce 2,000 distinct full-precision outputs. Both containments are
therefore satisfied vacuously over an empty set, and this is not evidence about
interleaved quantization.

The reason is structural. The output is two real-valued logits in float64, and
exact equality of a continuous real vector essentially never occurs. The same
holds at the final hidden layer in float64 and at the float32 training
precision: 2,000 unique of 2,000 in every case. In these networks
`collisions(F)` is empty at every observation point, so **the containment
question as literally posed cannot be tested on trained weights.** A zero count
here means the population being tested is empty, not that the relation was
checked and survived.

## Result 2: at a fixed quantization level, containment fails in both directions

The comparison becomes non-trivial when both sides are taken at the same
quantization level, at the final hidden layer, where quantized collisions are
dense (for one bfloat16 run: 207 unique vectors of 2,000). QF is the census's
existing post-hoc metric; G reaches the same layer through a network that
quantized every preceding activation.

Pooled over the 18 runs, pairs colliding under one regime but not the other:

| Quantizer | QF pairs | G pairs | QF-not-G | G-not-QF |
|---|---:|---:|---:|---:|
| float64 | 0 | 0 | 0 | 0 |
| float32 | 85 | 85 | 0 | 0 |
| float16 | 176,687 | 174,760 | 10,426 | 8,499 |
| bfloat16 | 791,414 | 791,057 | 39,678 | 39,321 |
| fixed-8 | 2,015,383 | 2,016,190 | 100,420 | 101,227 |
| fixed-6 | 4,989,228 | 4,964,034 | 506,550 | 481,356 |
| fixed-4 | 10,625,464 | 9,890,405 | 2,155,243 | 1,420,184 |

Containment holds in only 37 of 126 rows, and the 37 are float64, float32, and
a few float16 rows. **Neither direction of containment holds in general.** Two
inputs merged by post-hoc quantization can be kept apart by interleaving, and
inputs merged under interleaving can be distinct post-hoc. Interleaved
quantization is not a refinement or coarsening of post-hoc quantization; it is a
different map, as the theoretical argument anticipated.

Seed-level variation, pairs per run (mean ± SD over seeds):

| Quantizer | Width | QF-not-G | G-not-QF |
|---|---:|---:|---:|
| bfloat16 | 5 | 6,020.33 ± 974.82 | 5,673.67 ± 435.43 |
| bfloat16 | 15 | 2,730.80 ± 376.74 | 3,024.60 ± 417.75 |
| bfloat16 | 30 | 1,183.40 ± 643.61 | 1,066.20 ± 483.23 |
| bfloat16 | 50 | 409.20 ± 207.51 | 369.20 ± 187.49 |
| fixed-4 | 5 | 195,918.67 ± 84,334.23 | 60,108.00 ± 42,254.80 |
| fixed-4 | 15 | 145,569.20 ± 29,450.02 | 74,287.80 ± 23,760.91 |
| fixed-4 | 30 | 100,538.80 ± 20,976.33 | 90,155.60 ± 19,558.87 |
| fixed-4 | 50 | 67,389.40 ± 14,250.73 | 83,528.60 ± 13,202.13 |

Divergence grows monotonically as precision coarsens and as width narrows.

## Result 3: only fixed-4 produces between-class divergence, and it breaks accuracy

Class composition of the divergent pairs is the part that bears on separation.

**Every** QF-not-G pair at every precision is within-class: 0 between-class out
of 2,155,243 at fixed-4 and 0 out of 39,678 at bfloat16. For G-not-QF, all
precisions except fixed-4 are also entirely within-class. The single exception
is fixed-4, with 99,142 between-class pairs, concentrated at narrow widths:

| Width | G-not-QF within-class | G-not-QF between-class |
|---:|---:|---:|
| 5 | 90,304 | 90,020 |
| 15 | 362,325 | 9,114 |
| 30 | 450,770 | 8 |
| 50 | 417,643 | 0 |

These between-class collisions are genuine, and they correspond to the network
failing the accuracy gate when it actually consumes quantized activations.
Evaluation accuracy of the network run in regime G:

| Width | Seed | Accuracy (F) | Accuracy (G, fixed-4) |
|---:|---:|---:|---:|
| 5 | 0 | 1.0000 | 0.9900 |
| 5 | 1 | 1.0000 | 0.9610 |
| 5 | 3 | 1.0000 | 0.8970 |
| 15 | 3 | 1.0000 | 0.9805 |
| 15 | 4 | 1.0000 | 0.9650 |
| 50 | 0-4 | 1.0000 | 0.9990-1.0000 |

At bfloat16 every run retains 1.0000 accuracy under G. For the width-5 seed-3
run at fixed-4, 6 of 15 final-layer G groups are class-impure while 0 of 14 QF
groups are; accuracy falls to 89.70%.

So the between-class result is not a counterexample to the separation argument.
It is the expected consequence of running a network at a precision it was not
trained for: fixed-4 interleaving degrades the network below the acceptance
gate, and the between-class collisions are the mechanism of that degradation. A
network in that state would never have entered the census.

## What this does and does not support

Supporting the argument: at every precision the network actually tolerates —
float32, float16, bfloat16, fixed-8, fixed-6 — interleaved quantization produces
**zero** between-class collisions that post-hoc quantization does not also
produce. Interleaving does not supply separation between classes that
full-precision weights had not already established.

Refining the argument: the superset relation `collisions(G) ⊇ collisions(F)`
does not hold, and neither does its converse, once both sides are evaluated at a
common quantization level. The failure is real and large (39,678 and 39,321
pairs at bfloat16), but it is **entirely within-class** except where fixed-4
breaks the classifier outright. The relation fails technically without affecting
class separation.

Not established here: this is depth 6 only, one dataset, one activation, and a
finite evaluation sample. A sampled collision test cannot establish anything
about the continuous class supports, and no claim about linking or unlinking
follows from these counts.

## One-sentence summary

> Across 18 accepted linked-tori tanh runs at depth 6 (seeds recorded, exact
> recovery verified), interleaved quantization produced no between-class
> collisions that post-hoc quantization did not also produce at any precision
> the network tolerates — float32, float16, bfloat16, fixed-8, and fixed-6 — with
> the sole exception of fixed-4, where 99,142 between-class pairs appear only
> because interleaving degrades evaluation accuracy from 100% to as low as
> 89.70% and would have failed the acceptance gate outright; the strict superset
> relation `collisions(G) ⊇ collisions(F)` is nonetheless untestable as posed,
> since exact full-precision collisions never occur in these networks, and when
> both regimes are compared at a common quantization level containment fails in
> both directions but entirely within-class.
