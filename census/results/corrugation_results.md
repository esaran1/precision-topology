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

### What this test can and cannot speak to

Stated bluntly, because the limitation is structural rather than incidental.

Fold layer is measurable only on runs that separate. **Only GELU separates at
width 3. All 34 traces are therefore GELU traces, and what was tested is GELU's
folding.**

The prediction concerned folding in general. The test covers folding by one
activation — the only one that reaches separation on any of these links.

**The negative is real within that scope.** Corrugation is not undone by the
first affine layer the way a rigid rotation is, and it was tested at amplitudes
18× past the embeddedness limit and frequencies to 200. GELU's fold does not
move later under any of it.

**It cannot speak to whether monotonic activations fold later, because they
never fold at all.** A monotonic network that never reaches separation has no
fold layer to measure — the quantity is undefined for it, not large. Any claim
about where monotonic activations would fold, if corrugation forced them to fold
later, is outside what this measurement can reach.

> **Restatement (2026-08-22, dense verification).** Of the 34 traced runs,
> **only 5 are dense-verified separations** (`dense_check.md`, addendum),
> spanning four corrugated configurations plus `flat` — none at the
> published amplitude 0.3, none at frequencies above 50. Fold layer is 1 in
> all 5 (and in all 29 sample-level-only traces, so no evidence points the
> other way). But the claim this section makes — corrugation up to 18× the
> embeddedness limit and frequency 200 does not move the fold — was
> supported at that strength by runs that were not regional separations.
> **The dense-verified version of the negative covers only mild
> corrugation.** At the published corrugation values and beyond, no
> dense-verified separating run exists to measure a fold layer on, so
> whether strong corrugation moves the fold is now an open question rather
> than a settled negative. The n=34 statements above stand as written about
> sample-level separations.

---

## Prediction 2: the monotonic zero should survive — BORNE OUT

**0 monotonic separations in 1,890 width-3 runs on corrugated links.**

| Reading | Runs | Separations | Min errors |
|---|---:|---:|---:|
| A (core displaced) | 1,080 | **0** | 24 |
| B (offsets modulated) | 810 | **0** | 23 |
| **Total** | **1,890** | **0** | **23** |

This extends the count to 3,330 monotonic width-3 runs with zero separations
across 33 configurations, two protocols, and both G.1 readings. *(Ledger
audit 2026-08-22: this total covers the corrugation and parametrization
sweeps only; the audited all-strata total is 5,570 — `CLAIMS.md`, T1.)* Per
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
uncorrugated grid. GELU populates 0, 1–5, 6–8, and 9–15.

### The floor rose from 9 to 16 while the bulk gap narrowed

This is the sharpest single number in this round, so it is stated on its own.

Corrugation made the task harder for every activation. On the bulk that shows up
as convergence: the GELU-to-tanh median gap fell from **12 points to 4**. If what
separates the activations were an optimisation advantage, added difficulty
pressing on everyone should close the gap — and in the bulk, it did.

The floor moved the other way. The lowest error any monotonic run achieved rose
from **9 to 16**, while GELU continued to reach exactly 0. Added difficulty
pushed the monotonic floor *up* rather than pulling GELU's separations *down*.

**That is what a fixed barrier looks like under added difficulty, and it is the
opposite of what a closing optimisation gap would do.** A barrier does not move
when the task gets harder; the distributions retreat from it. An optimisation
advantage would erode, and the bulk shows exactly that erosion happening at the
same time the floor holds.

> **Retraction (2026-08-22), superseding the scope-narrowing note that
> previously stood here.** The floor argument in this section — a populated
> band above 9–16 errors with a hard edge, read as the shape of a barrier —
> **is retracted, not narrowed**. A monotonic network reached 2 errors
> (threshold sweep), and derivative-free search produced monotonic networks
> at 0 *train* errors: the 9–15 band with its apparent hard edge was an
> artifact of which activations SGD was searching with, not a property of
> the monotonic category. The replacement claim: **monotonic networks can
> shatter the sample and cannot separate the region.** The barrier is at
> exactly zero eval errors; monotonic networks approach it arbitrarily
> closely without reaching it, and no populated-band structure above zero
> carries evidential weight. The numbers in this section remain correct as
> measurements of tanh/ReLU/leaky-ReLU under SGD; the argument built on
> them does not survive. See `threshold_results.md` (Prediction 3) and
> `search_results.md` (the train-zero finding); `CLAIMS.md` T6 is
> authoritative.

> **Status note (2026-08-22, dense verification).** `dense_check.md`: of the
> 34 width-3 GELU separations in this sweep, only **5 survive 100,000 fresh
> points** from their own configuration, and 16 of the 21 configurations are
> left with zero dense-verified separations. Every separation count in this
> document is sample-level (0 errors on 2,000 eval points). The monotonic
> zero is unaffected — a sample-level zero already implies a regional zero —
> and the tail-versus-bulk contrast survives in direction, but the corrugated
> GELU separation counts should not be quoted as regional separations. The
> fold-layer-1 observation keeps its full support on the baseline link, where
> the dense survivors concentrate.

---

## Do the two readings differ?

Not materially, on any measured quantity.

| Quantity | Reading A | Reading B |
|---|---|---|
| Monotonic separations at width 3 | 0 / 1,080 | 0 / 810 |
| Monotonic minimum errors | 24 | 23 |
| Fold layer, separating runs | 1 in all 23 | 1 in all 11 |
| GELU separations | 21 / 360 | 13 / 270 |

**The ambiguity in Appendix G.1 is immaterial to every result in this
project.** Both readings give 0 monotonic separations, monotonic minima of 24
and 23, and fold layer 1 on every traced run. The embedded low-amplitude arms
under Reading A behave like the rest, so the self-overlapping swept tube at
published values drives nothing.

**We are not blocked on this and no result is contingent on it.** The question
for the author is one of accuracy about their published parametrization — which
of the two readings Appendix G.1 intends — and it is worth asking for that
reason alone. It is not a dependency on our side, and nothing in this analysis
waits on the answer.

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

---

> **Duplicate-condition correction (2026-08-23).** Reading B's sampler
> ignores the amplitude parameter (modulation depth hardcoded at 0.5), so
> `B_a0.05`, `B_a0.15`, `B_a0.5`, and `B_paper` are **one condition run
> four times** — their per-run accuracies are bit-identical. Corrected
> distinct counts: 21 configurations → **18**; 5,040 runs → **4,320**
> distinct; corrugated monotonic width-3 1,890 → **1,620** (still 0
> separations); GELU width-3 separations 34 → **28**; runs ≤5 errors 60 →
> **48**; configurations with GELU separations 18 → **15**. Medians and
> permutation results are essentially unchanged (duplicates were unbiased
> copies). Full analysis and mechanism: `reading_b_anomaly.md`;
> authoritative counts: `CLAIMS.md`.
