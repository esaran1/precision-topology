# Corrugation sweep: outcomes against the registered predictions

5,040 runs across 21 configurations spanning both readings of Appendix G.1,
widths 3 and 4, depths 3/5/12, four activations, 10 seeds. Every configuration
verified at core linking `|lk| = 1` with zero between-class collisions before
training. Predictions were registered in
`results/corrugation_readings_prediction.md` before any run executed and are
reported below in the order registered.

**Reading A** displaces the core (9 non-embedded configurations, 3 embedded
including the two low-amplitude arms). **Reading B** modulates the sampled
radius (9 configurations, all embedded). Under Reading A the swept solid
self-overlaps at published values; this is reported as a diagnostic, and the
cores remain a genuine link.

---

## Prediction 1: fold layer should move later — NOT BORNE OUT

**Fold layer is 1 in all 34 separating runs.** Every configuration, both
readings, every amplitude and frequency tested.

| Grouping | n | mean fold layer | median | min | max |
|---|---:|---:|---:|---:|---:|
| Reading A (core displaced) | 23 | 1.000 | 1 | 1 | 1 |
| Reading B (offsets modulated) | 11 | 1.000 | 1 | 1 | 1 |
| Uncorrugated (`flat`) | 2 | 1.000 | 1 | 1 | 1 |
| Corrugated | 32 | 1.000 | 1 | 1 | 1 |

By amplitude under Reading A, spanning the embedded limit at 0.01657 and the
published value at 0.3:

| Amplitude | 0.000 | 0.001 | 0.05 | 0.15 | 0.30 | 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| Separating runs | 2 | 1 | 3 | 3 | 12 | 2 |
| Mean fold layer | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

The input configuration is `link = −1` on every traced run, so the fold is a
genuine change from −1 to 0 rather than a measurement that started at zero.

**This negative carries weight the earlier one did not.** The previous
fold-layer test varied axis alignment, which the first affine layer undoes at no
cost, so it never examined the property the account concerns. This test varies
the corrugation itself, at amplitudes up to 18× past the embeddedness limit and
frequencies up to 200, and the immediacy is unchanged.

The account under test was that a corrugated tube leaves no single direction
that breaks the link without creating other problems, so a network should not
resolve the topology in one coordinate fold. **That does not describe what these
networks do.** Something other than the availability of a single clean fold
direction is producing the layer-1 immediacy, and this measurement does not
identify what. It remains unable to separate "the fold is cheap for a width-3
affine map under any parametrization of this link" from "the immediacy is a
property of the optimiser".

One caveat on power: fold layer is measurable only on separating runs, and only
GELU separates at width 3, so all 34 traces are GELU. The prediction concerned
folding in general; what is tested is folding by the one activation that reaches
separation.

---

## Prediction 2: the monotonic zero should survive — BORNE OUT

**0 monotonic separations in 1,890 width-3 runs on corrugated links.**

| Reading | Runs | Separations | Min errors |
|---|---:|---:|---:|
| A (core displaced) | 1,080 | **0** | 24 |
| B (offsets modulated) | 810 | **0** | 23 |
| **Total** | **1,890** | **0** | **23** |

This extends the count to 3,330 monotonic width-3 runs with zero separations
across 33 configurations, two protocols, and both G.1 readings. Per
`notes/reporting_rules.md` this is a count rather than an extreme-value
statistic, and it is the claim this project leans on.

GELU separates 34 times in 630 width-3 runs on the same links, in 18 of the 21
configurations.

---

## Prediction 3: the advantage should stay tail-confined — BORNE OUT, more sharply

| Activation | n | **median errors** | p25 | p75 | runs ≤5 | **separations** |
|---|---:|---:|---:|---:|---:|---:|
| **GELU** | 630 | **65** | 51 | 92 | **60** | **34** |
| tanh | 630 | 70 | 56 | 115 | **0** | **0** |
| leaky-ReLU | 630 | 80 | 55 | 162 | **0** | **0** |
| ReLU | 630 | 1000 | 88 | 1000 | **0** | **0** |

**The bulk difference narrowed to 4 points.** GELU's median is 65 against tanh's
70, down from 61 against 73 on the uncorrugated grid. It remains statistically
real (permutation p = 0.0006 over 5,000 permutations) and is now smaller than
before.

**The tail difference did not narrow at all.** 60 GELU runs at ≤5 errors against
0 for every monotonic activation. Resampling 630 runs from tanh's own
distribution 5,000 times produces a maximum of **0** runs at ≤5 errors.

Corrugation therefore moved the two quantities in opposite directions: the bulk
gap shrank while the categorical tail gap held. That is the shape the
tail-versus-bulk argument predicts and the opposite of what a broad optimisation
advantage would produce.

### The floor is sharper than on the uncorrugated grid

Run counts by error band at width 3:

| Activation | 0 | 1–5 | 6–8 | 9–15 | 16–25 | 26–50 |
|---|---:|---:|---:|---:|---:|---:|
| **GELU** | **34** | **26** | **2** | **1** | 3 | 80 |
| tanh | 0 | 0 | 0 | 0 | 1 | 97 |
| leaky-ReLU | 0 | 0 | 0 | 0 | 0 | 76 |
| ReLU | 0 | 0 | 0 | 0 | 1 | 23 |
| **All monotonic** | **0** | **0** | **0** | **0** | **2** | **196** |

**No monotonic run lands below 16 errors in 1,890 runs**, against 9 on the
uncorrugated grid. GELU populates 0, 1–5, 6–8, and 9–15. The gap between the
lowest monotonic result and GELU's separations widened under corrugation.

---

## Do the two readings differ?

Not materially, on any measured quantity.

| Quantity | Reading A | Reading B |
|---|---|---|
| Monotonic separations at width 3 | 0 / 1,080 | 0 / 810 |
| Monotonic minimum errors | 24 | 23 |
| Fold layer, separating runs | 1 in all 23 | 1 in all 11 |
| GELU separations | 21 / 360 | 13 / 270 |

The ambiguity in Appendix G.1 is **immaterial to every conclusion drawn here**.
Whether the oscillation displaces the core or modulates the sampled offsets, the
monotonic zero holds, the fold stays at layer 1, and the tail-versus-bulk shape
is the same. The question of intent can be put to the author as a point of
accuracy about the parametrization, but nothing in this analysis depends on the
answer.

The low-amplitude embedded arm under Reading A behaves like the rest:
`A_embedded_a0.001` and `A_embedded_f0.5` each produce separations with fold
layer 1 and no monotonic separations, so the self-overlapping swept tube at
published values is not driving any result.

---

## Width 4

| Activation | Separations / 630 |
|---|---:|
| GELU | 433 (68.7%) |
| tanh | 346 (54.9%) |
| leaky-ReLU | 37 (5.9%) |
| ReLU | 36 (5.7%) |

The width 3→4 boundary holds on corrugated links, with the same caveat recorded
elsewhere: every activation separates in at least one width-4 run, which is not
the same as every activation succeeding at width 4. ReLU and leaky-ReLU remain
below 6%.
