# Task F 1b: what counts as the positive control being restored

Fixed 2026-08-23, before any Task F search run.

The 2a bottleneck experiment was uninformative because its positive
control failed: GELU never beat ReLU anywhere in it. Task F searches for
a setting where the folk GELU-over-ReLU advantage reliably reproduces;
only such a setting can host a retest of the width prediction.

## Restoration criterion (all four required)

1. **Effect**: GELU's mean held-out accuracy exceeds ReLU's.
2. **Statistics**: two-sided permutation test p < 0.05 at **n ≥ 10
   seeds per activation** in the candidate configuration.
3. **Replication**: the same-signed, nominally significant effect in at
   least **two nearby configurations** (adjacent depth, adjacent
   budget, or warmup on/off) — a single significant cell in a searched
   grid is treated as a multiple-comparisons artifact, full stop.
4. **Not budget-censoring**: training error must have plateaued (final-
   epoch training-loss improvement < 1% relative) in both activations,
   so the effect is not an artifact of stopping mid-descent.

## The 1c kill switch, run early

In every configuration where GELU beats ReLU, tanh is run at equal n.
**If tanh beats ReLU by a statistically indistinguishable margin
wherever the GELU advantage appears** (two-sided test of GELU vs tanh
n.s. and sign of tanh−ReLU matching GELU−ReLU), the folk advantage is a
smoothness effect, the fold account has nothing to explain, and Task F
stops there and reports — no width sweep, bridge recorded as
inapplicable rather than untested.

## Search grid (the 1a map; absences recorded as findings)

Cheap arm — MNIST MLPs, trained past plateau (15 epochs, Adam 1e-3,
batch 256): hidden depth 4, 8, 12 × {gelu, relu, tanh} × 3 pilot seeds,
width 256.

Main arm — CIFAR-10, small VGG-style CNN (3×3 convs, max-pool every
second conv, channels 32→64→128, two-layer head), Adam 1e-3, batch 128,
12 epochs: conv depth 4, 8 × {gelu, relu, tanh} × 3 pilot seeds; a
500-step linear-warmup variant on the most promising depth. No
augmentation (budget; recorded as a scope limit of the map).

Pilots at 3 seeds only locate candidates; nothing is declared from a
pilot. Candidates go to the n ≥ 10 replication of the criterion above.

## Budget

Hard limit one to two days. If no configuration passes all four
criteria in that time, Task F ends with the negative map and the
pre-agreed scope sentence (Part 3 of the brief); the budget is not
extended to rescue the bridge.
