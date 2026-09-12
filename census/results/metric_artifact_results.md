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
2. **The exact-separation criterion is not arbitrary here.** Unlike Exact
   String Match on a language benchmark, 0 errors is the quantity the
   impossibility theorem speaks about. A run with 36 mean errors has not
   "nearly" realized `sign(|x| - 1)`; it has failed to, and the fold-depth
   bound is a statement about the networks that succeed.
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

## The product form is what the data picks, not only what the theorem asserts

An adversarial check: scan `|w2|^p * G*^q` over `p, q` in [0.3, 2.0] and ask
whether some other exponent pair separates better. Since ranking by
`p log|w2| + q log G*` depends only on the ratio `p/q`, the theorem's product
form is exactly `p/q = 1`.

| family | R (`p/q = 1`) | best grid AUC | best `p/q` | gain |
|---|---:|---:|---:|---:|
| A | 0.9975 | 0.9984 | 1.18 | +0.0008 |
| q2 | 0.9998 | 0.9998 | 1.00 | +0.0000 |
| q1 | 0.9995 | 0.9999 | 1.17 | +0.0004 |

**The optimal ratio is within 18% of 1 in every family, and the achievable gain
is at most 0.0008 AUC.** The exponent the theorem gives is the one the data
selects.

## Where R's contribution actually lies

At fixed `a`, `G*(a)` is a constant, so R is a monotone rescaling of `|w2|` and
the two must have identical AUC. They do, to four decimals, in all 19
sufficiently populated `(family, a)` cells — an internal consistency check that
would have caught a bug in the R computation.

**So R's entire contribution is commensuration across `a`.** `|w2|` predicts
solving perfectly well within a fixed activation; what it cannot do is compare
runs at different `a`, because the same `|w2|` means different things when the
fold is deeper or shallower. `G*(a)` is the exchange rate, and the theorem
supplies it.

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
