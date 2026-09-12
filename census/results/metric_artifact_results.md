# Item 1: is the R threshold a metric artifact? — partly yes

Registered in `results/metric_artifact_prediction.md` before any continuous
metric was computed against R, and after reading Schaeffer, Miranda & Koyejo
from the source PDF (`paper/sources/schaeffer_mirage.txt`).

Date: 2026-09-12.

---

## Headline, reported unsoftened

**Registered outcome 3 (intermediate), and close to outcome 2.** The ratio of
transition widths is `W_continuous / W_binary = 2.09`, just past the boundary
of 2.0 fixed in advance for "genuine". But the ratio understates the problem,
and the substantive finding is in the *locations* rather than the widths:

> **The continuous error count completes 76% of its total improvement in a
> region where not a single run solves.** Mean eval errors fall from 153.6 to
> 36.3 — a 77% reduction over 1,817 runs — with **0 solves**. The binary rate
> does not leave zero until R = 0.30, by which point most of the underlying
> improvement has already happened.

The 10%-to-90% crossings do not even overlap: the continuous metric moves over
`R in [0.055, 0.307]`, the binary rate over `R in [0.332, 0.452]`.

**Consequence, applied immediately**: the claim "R identifies a sharp
capability threshold" is **not supported** and is withdrawn. The supported
claim is **"R predicts where the binary solve criterion flips"**, which is
smaller. The R result stands as a predictor of the criterion we defined; it
does not establish a discontinuity in the underlying system.

## What Schaeffer et al. actually claim

Read from the source, not summarized from memory. For a **fixed task, fixed
model family, and fixed model outputs**, apparent emergence arises from the
researcher's metric: **nonlinear or discontinuous metrics produce apparent
emergent abilities, linear or continuous metrics produce smooth predictable
change**. Evidence: three confirmed predictions on InstructGPT/GPT-3; a
BIG-Bench meta-analysis where emergence appears under at most 5 of 39 metrics
with **>92% under two** — Multiple Choice Grade (discontinuous) and Exact
String Match (nonlinear); and a constructive demonstration manufacturing
apparent emergence in vision networks by metric choice. They state explicitly
that **"nothing in this paper should be interpreted as claiming that large
language models cannot display emergent abilities."**

Our binary solve criterion is an indicator on perfect performance — the same
shape as Exact String Match. The objection applies to us at full strength.

## The measurement (1a)

2,400 runs carrying both a continuous eval error count (0-2,000) and a
computable R.

| R bin | n | solve rate | mean errors | median | q10 | q90 |
|---|---:|---:|---:|---:|---:|---:|
| (0, 0.05] | 995 | 0.000 | 153.6 | 132 | 109 | 203 |
| (0.05, 0.1] | 214 | 0.000 | 133.2 | 111 | 97 | 201 |
| (0.1, 0.15] | 151 | 0.000 | 116.0 | 102 | 92 | 193 |
| (0.15, 0.2] | 178 | 0.000 | 110.9 | 102 | 91 | 173 |
| (0.2, 0.25] | 151 | 0.000 | 90.8 | 95 | 41 | 115 |
| (0.25, 0.3] | 128 | **0.000** | **36.3** | 26 | 12 | 54 |
| (0.3, 0.32] | 38 | 0.026 | 12.3 | 10 | 4 | 25 |
| (0.34, 0.36] | 45 | 0.222 | 10.6 | 4 | 0 | 15 |
| (0.36, 0.38] | 33 | 0.515 | 14.0 | 0 | 0 | 8 |
| (0.4, 0.42] | 36 | 0.806 | 11.2 | 0 | 0 | 3 |
| (0.46, 0.48] | 22 | 1.000 | 0.0 | 0 | 0 | 0 |
| > 0.6 | 280 | 1.000 | 0.0 | 0 | 0 | 0 |

## The registered criterion, scored

`W` = range of R over which a metric moves from 10% to 90% of its total
observed swing, each metric normalized by its own range.

| metric | `W` | 10% at | 90% at |
|---|---:|---:|---:|
| binary solve rate | 0.1201 | R = 0.332 | R = 0.452 |
| continuous error count | 0.2513 | R = 0.055 | R = 0.307 |

**Ratio = 2.09.** Registered bands: `<= 2` genuine, `>= 5` artifact, between
intermediate. **Verdict: intermediate**, marginally outside "genuine".

**But the width ratio is the wrong summary here and we say so rather than
hiding behind a number that flatters us.** The two transitions occupy
*disjoint* ranges of R. A metric artifact in Schaeffer's sense is a binary
metric jumping while the underlying quantity improves smoothly *through the
same region*. That is close to what we observe: the underlying quantity does
most of its improving first, and the binary metric then jumps later, at the
point where the residual error finally reaches exactly zero.

## Is the underlying decline itself sharp?

If the error decline had its own sharp feature at R ~ 0.37, the binary metric
would merely be relocating a real discontinuity. It does not. Local slope
`d(errors)/dR` by bin:

| R | mean errors | slope |
|---:|---:|---:|
| 0.213 | 102.3 | −222 |
| 0.237 | 80.3 | −882 |
| **0.263** | **39.1** | **−1649** |
| 0.287 | 34.3 | −190 |
| 0.312 | 16.3 | −720 |
| 0.338 | 15.8 | −20 |
| **0.362** | **12.4** | **−136** |
| 0.388 | 5.4 | −280 |

**The steepest decline is at R ~ 0.263, not at the binary threshold of 0.37.**
By the time the binary rate is transitioning, the continuous slope has
flattened to a fifth of its peak. The sharp feature in the underlying quantity
and the sharp feature in the binary metric are **in different places**.

## A metric we could not use, and why

Achieved logit margin is continuous and signed, and we computed it. It cannot
serve as an independent test: **margin > 0 is definitionally equivalent to
separating**, so its zero-crossing must coincide with the binary transition by
construction. Reporting it as corroboration would be circular. We note it
(margin rises monotonically through the band, crossing 0 at R ~ 0.40) and do
not use it as evidence.

This is worth stating because it is the trap a reader might fall into: of our
three continuous metrics, one is independent (error count), one is definitional
(margin), and dense-verification margin is only stored for already-separating
runs, so it is conditioned on the outcome.

## The impossibility argument, scoped (1c)

Below `a = 1`, no width-1 network solves at any R. This is a **theorem**: a
monotone activation composed with affine maps yields at most one sign change,
while `sign(|x| - 1)` requires two. No metric choice alters it.

**What this establishes**: at least one boundary in this system is genuine and
not metric-induced — the analytic threshold at `a = 1`. That is a real
disanalogy with the LLM setting, where no such impossibility result is
available, and it is worth stating.

**What it does not establish**: that the R threshold at ~0.37 is equally
genuine. That threshold sits well inside the non-monotonic regime where
solutions provably exist, and the measurement above shows the underlying
quantity moving smoothly through it. **The first argument does not carry the
second, and we do not use it to.**

## What we now claim, and what we withdraw

**Withdrawn**: "R identifies a sharp capability threshold." The continuous
error count does not have a sharp feature at R = 0.37, and 76% of its total
improvement occurs where the solve rate is identically zero.

**Retained, and still substantial**:

1. **R predicts where the binary criterion flips**, with AUC 0.9975 against
   0.8393 for raw `|w2|`, pooled across budgets and both optimizers. That is a
   statement about our criterion, and our criterion is the one the theorem
   constrains: the theorem is about exact separation, not about error counts.
2. **The exact-separation criterion is appropriate for this task — which is a
   separate question from whether its transition is sharp, and a reviewer will
   separate them.** Unlike Exact String Match on a language benchmark, 0 errors
   is the quantity the impossibility theorem speaks about: a run with 36 mean
   errors has not "nearly" realized `sign(|x| - 1)`, it has failed to, and the
   fold-depth bound is a statement about the networks that succeed. **That
   argues the binary criterion is the right criterion. It does not argue that
   its transition is sharp, and the measurement above shows it is not.** The
   honest joint statement is: *the binary criterion is appropriate for this
   task, and its transition is nonetheless sharper than the underlying quantity
   warrants.*
3. **The transition in the underlying quantity is real but earlier**, peaking
   at R ~ 0.26. A future version of this analysis should report the error-count
   curve alongside the solve rate rather than the solve rate alone.

**The honest framing for the paper**: our results describe a threshold in
*exact realizability*, which is the property the theorem constrains, while the
underlying approximation quality improves smoothly and mostly earlier. Both
facts belong in the text. Presenting only the binary curve would invite exactly
Schaeffer's critique, and the critique would be substantially correct.

---

# Item 2: does the product structure survive the family failure? — yes

Registered criterion (`metric_artifact_prediction.md`): R dominates if its AUC
exceeds the best competitor's by **>= 0.05 with non-overlapping 95% bootstrap
intervals**, in a family with **>= 20 runs inside its transition band**.

## Qualification

Transition band = the R range where the family's binned solve rate is strictly
between 5% and 95%.

| family | n | solved | band | runs inside | qualifies |
|---|---:|---:|---|---:|---|
| A | 2,910 | 579 | (0.277, 0.476) | 415 | **yes** |
| q2 | 360 | 213 | (0.122, 0.328) | 26 | **yes** |
| q1 | 360 | 213 | (0.149, 0.268) | 25 | **yes** |

All three qualify. q4 and q0.667 were not regenerated and are absent.

## Result: R dominates in every qualifying family

| family | R | `|w2|` alone | `G*(a)` alone | margin | non-overlap | dominates |
|---|---:|---:|---:|---:|---|---|
| A | **0.9975** [0.9966, 0.9984] | 0.8393 [0.8240, 0.8544] | 0.8009 [0.7798, 0.8220] | **+0.1583** | yes | **YES** |
| q2 | **0.9998** [0.9994, 1.0000] | 0.9439 [0.9204, 0.9640] | 0.6281 [0.5670, 0.6873] | **+0.0559** | yes | **YES** |
| q1 | **0.9995** [0.9982, 1.0000] | 0.9250 [0.8960, 0.9490] | 0.6330 [0.5731, 0.6916] | **+0.0745** | yes | **YES** |

Every margin clears the registered 0.05 with non-overlapping intervals.

### The product form is not an assumption — it is what the data selects

This is the strongest result in the analysis, and it is independent
confirmation of the theorem's structure arriving from a different direction.

The theorem *asserts* that `|w2|` and `G*` enter as a product. That could be an
assumption we imposed on the data by constructing R that way. So we scanned the
whole family of power combinations `|w2|^p * G*^q` over `p, q` in [0.3, 2.0] and
asked which separates best. Because ranking by `p log|w2| + q log G*` depends
only on the ratio `p/q`, the theorem's product form is exactly `p/q = 1`, and
the scan asks whether any other exponent ratio would do better:

| family | R (`p/q = 1`) | best over the scan | best `p/q` | achievable gain |
|---|---:|---:|---:|---:|
| A | 0.9975 | 0.9984 | 1.18 | **+0.0008** |
| q2 | 0.9998 | 0.9998 | 1.00 | **+0.0000** |
| q1 | 0.9995 | 0.9999 | 1.17 | **+0.0004** |

**The optimal ratio is 1.00 to 1.18 in every family, and the maximum gain
available from any other exponent pair is 0.0008 AUC.** Among all power
combinations of the two quantities, the data picks the theorem's. The product
structure is therefore not something the theorem imposes on the analysis; it is
what the measurement selects.

### The check that validates the pipeline

At fixed `a`, `G*(a)` is a constant, so R is a monotone rescaling of `|w2|` and
their AUCs **must** agree by construction. They do, to four decimals, in all 19
sufficiently populated `(family, a)` cells.

This is not a formality. Had the R computation carried a bug — a misaligned
`G*` lookup, a wrong orientation, a stale join — the fixed-`a` AUCs would have
diverged and the entire result above would have been a plausible artifact. The
agreement is what separates this from analysis that merely looks right.

### What R contributes, in one sentence

**`|w2|` predicts solving well within a fixed activation; what it cannot do is
compare across `a`. `G*(a)` is the exchange rate the theorem supplies.**

The same `|w2|` means different things when the fold is deeper or shallower,
and `G*` is exactly the conversion factor between weight scale and achievable
margin. That is the whole of R's contribution, and it is why R and `|w2|` are
indistinguishable within a family and far apart across families.

## What Item 2 establishes, against what the family test rejected

The family test (T52) rejected **a universal threshold value**: `R50` is 0.373,
0.295, 0.247 across A, q2, q1, shifts three to five times the materiality
threshold, surviving the largest defensible `G*` correction.

Item 2 establishes the weaker and different claim that **the product structure
is universal**: in every qualifying family, solvability is governed by
`|w2| x G*(a)` rather than by either factor alone, with the theorem's exponents,
and no better exponent pair exists in the data.

**The defensible statement is therefore**: solvability is governed by the
product of an architectural quantity fixed before training (`G*`) and a
training quantity set by budget and optimizer (`|w2|`), **with a
family-specific threshold**. That is materially weaker than "R is the
controlling variable" and materially stronger than "R is a family-A result".

**Scope, stated plainly**: three families, all with `beta` between 1.5 and 2.0,
all trained with Adam except family A's optimizer arm. q4, q0.667 and family B
are untested. Family B is positively homogeneous with `beta = 1` and is a known
counterexample to the budget law; nothing here addresses it.

---

# Item 2, extended: family B — the structure claim has a boundary

The three families tested above all sit in `beta` in [1.5, 2.0]. **Family B**
(`f(t) = max(t, alpha*t)`, non-monotone iff `alpha < 0`) is the known
`beta = 1` counterexample to the budget law (T44). Its terminal weights **were
recoverable** from `fold1d_sweep.csv` — 1,600 stored rows, 1,000 at
`alpha < 0`, 578 solved — so no retraining was needed.

`G*(alpha)` computed on a wide unrestricted scan, scored in both orientations:

| `alpha` | −1.0 | −0.5 | −0.25 | −0.1 | −0.05 |
|---|---:|---:|---:|---:|---:|
| `G*` | 3.1196 | 1.6000 | 0.8000 | 0.3200 | 0.1600 |

`G* ~ |alpha|^{0.9935}`, confirming `beta = 1` numerically for a positively
homogeneous family, as the analytic argument requires.

## Result: R does NOT dominate in family B

Transition band (0.598, 1.463) contains 286 runs, so the family qualifies.

| variable | AUC | 95% CI |
|---|---:|---|
| `R = |w2| G*/2` | 0.9804 | [0.9729, 0.9872] |
| **`|w2|` alone** | **0.9901** | [0.9819, 0.9967] |
| `G*(alpha)` alone | 0.3837 | [0.3488, 0.4180] |

**R is *worse* than raw `|w2|`** (margin −0.0097, intervals overlapping), and
the exponent scan puts the best ratio at `p/q = 1.54` rather than 1, gaining
+0.0183. **Registered dominance: NO.**

## Why, and the diagnosis strengthens the interpretation rather than weakening it

The solve rate in family B is **nearly constant in `alpha`** — 0.565, 0.555,
0.580, 0.595, 0.595 across a 20x range of `G*`. The fold parameter barely
affects solvability at all.

The reason is **compensation**. Median terminal `|w2|` rises as the fold
shallows: 2.33, 2.96, 4.25, 6.77, 9.21, i.e. `|w2| ~ G*^{-0.475}`. Training
reaches proportionally larger weights exactly as `G*` shrinks, so the product
stays far flatter (4.9x) than either factor alone (19x for `G*`, 4.0x for
`|w2|`).

So in family B, `G*` carries **no signal about solvability** — its own AUC is
0.384, below chance — and multiplying `|w2|` by it therefore *adds noise to a
variable that already worked*. Within each `alpha`, `|w2|` alone separates at
AUC 1.0000, 1.0000, 1.0000, 1.0000, 0.9777.

**One mechanism, two symptoms.** Positive homogeneity gives `beta = 1` and full
compensation. That is why family B breaks the **budget law** (the onset does
not move as predicted, T44) *and* why it breaks the **product structure** (the
exchange rate is constant in the relevant sense, so there is nothing to
commensurate). The two failures are not independent problems; they are the same
property of the family seen twice.

## What this does to the Item 2 claim

**It bounds it, honestly, rather than extending it.** The registered hope was
that testing family B would either extend the structure claim to a family where
the law fails — separating structure from scaling law — or bound the claim. The
second happened.

The defensible statement becomes:

> Solvability is governed by the product of an architectural quantity fixed
> before training (`G*`) and a training quantity set by budget and optimizer
> (`|w2|`), **in families where the fold depth is not fully compensated by
> weight growth** — verified in family A, q2 and q1 (`beta` in [1.5, 2.0]),
> **and failing in the positively homogeneous family B (`beta = 1`)**, where
> training compensates exactly and `G*` carries no signal.

That is narrower than the claim written before family B was tested, and it is
the claim the data supports. The condition is not ad hoc: homogeneity is
checkable in advance from the activation's functional form, and it is the same
condition that already scopes the budget law.

**Remaining untested**: q4 (`beta = 1.25`) and q0.667 (`beta = 2.5`), whose
per-run weights were not stored and were not regenerated. They sit at the
extremes of the `beta` range rather than outside the regime, so they would
tighten the interval rather than test the boundary; family B was the
informative case and it has been run.
