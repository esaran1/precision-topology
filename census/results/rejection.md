# Part 3: the rejection

The most damaging honest review a competent reviewer could write from the
paper and code as they stand. Every criticism is one they could actually make;
none is answered elsewhere in the text. Answers follow, and **where we cannot
answer, it says so.**

---

# THE REVIEW

## R1. The central result is a consequence of the setup, not a finding

The paper's headline is that the onset exponent scales as 1/beta. But beta is
*defined* as the exponent with which fold depth vanishes, and the theorem
states that required weight scale goes as 1/D(a) ~ eps^{-beta}. If terminal
weight scale grows as B^alpha, then setting B^alpha = eps^{-beta} gives
eps ~ B^{-alpha/beta} **in one line of algebra**. The "law" is that
substitution. The four families do not test a hypothesis; they check that
arithmetic holds when both inputs are measured, on families the authors built
so that beta takes chosen values.

The paper would be stronger if it said this plainly: it is a quantitative
verification of a two-step chain, not a discovery of a relationship. As
written, the framing ("we derived the power law and it holds") oversells a
substitution.

## R2. A single-optimizer scaling relationship does not support the conclusions

The abstract claims architecture comparisons at fixed budget measure
reachability. The evidence is a scaling relationship that **the authors
themselves show does not survive an optimizer change**: SGD's exponent ratio
misses the alpha ratio by 30%, and for the one family measurable under both
optimizers the exponent is -0.83 under Adam and statistically zero under SGD.

If the relationship fails under the second optimizer tried, the natural
inference is that it is a property of Adam's dynamics rather than of the
architecture-capability boundary. The authors report this honestly, which is
creditable, but then continue to frame the result as being about expressivity
thresholds. A relationship holding for one optimizer on four families built by
the authors is thin support for a claim about how the field should conduct
architecture comparisons.

## R3. "Train longer" is the whole phenomenon

The paper shows a = 1.25 goes from 0/200 to 40/40 by increasing budget 40x.
That is a statement that undertrained models underperform. The interesting
version would be that some capability is unreachable at *any* budget, and the
paper explicitly disclaims it: the gap closes as B -> infinity.

So the finding reduces to: near a capability boundary, you need more compute,
and the amount grows as you approach it. That is what practitioners already
assume. The exponent -1.36 quantifies it, but the paper does not show that
this exponent transfers to any setting anyone cares about -- the CIFAR
experiment measures a *magnitude* change in an activation comparison, not a
threshold exponent, and the two are connected only by analogy.

## R4. The families are constructed to produce the result

Four of five families are built from one template, f'(x) = (1-a) + a|x|^q,
chosen so beta = 1 + 1/q takes specified values. The fifth (sin) is not, and
it lands on the line -- but it is one point. A relationship fitted across
points the authors placed on the x-axis, using a template designed to vary
exactly the quantity in question, is close to unfalsifiable by construction.

What would convince: a family from a structurally different route, or a
naturally occurring activation with a computable beta that was not selected
for. Neither appears.

## R5. The exclusion table excludes explanations nobody offered

Five mechanisms are ruled out: barriers, sharpness, distance, margin,
criticality. But no prior work proposed any of these as the explanation for a
capability threshold in a four-parameter network on a synthetic 1-D task,
because no prior work considered this object. Excluding explanations for a
phenomenon you introduced is not the same as advancing a debate.

The genuinely interesting exclusion -- edge of stability -- turns out not to
apply because the points are not stationary, which the authors discovered only
after building an experimental program around the wrong test.

## R6. The setting cannot support conclusions about real training

A four-parameter network on 400 points is not a model of anything. The paper's
one real-data experiment (CIFAR-10) shows a 5x magnitude change in an
activation advantage across budgets, which is a different phenomenon from the
threshold displacement the theory describes, in a setting where beta is not
computable and no onset exists. The bridge between the toy result and the real
one is stated as analogy.

---

# THE ANSWERS

## To R1 -- the strongest criticism, and it is partly right

**Conceded**: the algebra is one line, and the paper must say so. What is not
conceded is that this makes it contentless. The substitution requires **both
inputs to be independently measurable and to actually take the values the
substitution needs** -- and neither was guaranteed:

- alpha is measured on a **different experiment** (terminal weight growth) and
  never fitted to the families. It could have taken any value; the predicted
  line's slope came out at 1.1240 [1.008, 1.240] against alpha's 1.1173
  [0.999, 1.236].
- The chain assumes training reaches the required scale by *growing* |w2|
  along a fold-amplifying direction. **That assumption is falsifiable and we
  falsified two versions of it**: family B has a diverging requirement, meets
  it by 900-1200x, and still fails; and the alpha-dependence fails
  quantitatively under SGD.

So the substitution is not automatic: it holds where the mechanism holds and
demonstrably fails where it does not. **But R1's core point stands and the
framing must change** -- from "we derived a law" to "we verified that a
two-step chain closes quantitatively, under one optimizer, and identified two
regimes where it does not."

## To R2 -- conceded, and it is already in the abstract

The single-optimizer scope is stated in the abstract, not in limitations,
precisely because of this objection. **We tested transfer three ways and
report all three failures** (`cross_optimizer.md`): the dynamical form fails
quantitatively, the ratio test could not execute, and the same-family
comparison fails decisively.

What we dispute is the inference. **The relationship failing under SGD is
itself a finding**: whatever governs the geometry-to-onset relationship is not
optimizer-independent. That is a constraint on any future account. We do not
claim more than one optimizer's worth of evidence, and the paper's scope
sentence says so.

**Cannot answer**: whether the relationship would hold for a third optimizer,
or for SGD on families it can reach. Untested.

## To R3 -- partly answered, partly not

**Answered**: "train longer" does not predict *how much* longer, and the
exponent is not free -- it is predicted from an analytically-derived beta and
an independently-measured alpha, then verified. Nor does "train longer"
predict that the requirement diverges at an **analytic expressivity
threshold** rather than at an arbitrary point, or that families with different
fold geometry diverge at different rates.

**Not answered**: R3 is right that the asymptotic disclaimer weakens the
result, and right that the CIFAR experiment measures a different quantity.
**We have no measurement of a threshold exponent in a real setting**, and the
connection is an analogy. Stated as such.

## To R4 -- largely conceded

**Conceded**: four of five families are ours, from one template. The sin
family is independent and lands on the line, and q2 reproduces its beta
through a different functional form with a constant sqrt(2) prefactor ratio --
so the template is faithful where it can be checked. But **a structurally
different construction route has not been tried**, and R4 is right that this
is the strongest available check we did not run.

**Cannot answer.** This is listed as an open exposure in the interpretive
audit and should appear in the paper's limitations in the same words.

## To R5 -- disputed, with a concession

**Disputed**: the exclusions are not of explanations nobody offered. Barriers
and sharpness are the two standard accounts of why gradient descent fails to
reach a minimum (Ahn-Zhang-Sra; Cohen et al.), and both are routinely invoked
for exactly this kind of failure. Ruling them out **by measurement on the same
objects** is what licenses looking elsewhere.

**Conceded**: the EoS program was built on a test that examined one of the
theorem's two conditions, and we corrected it ourselves. That is in the record
(`criticality_results.md`). A reviewer is entitled to note that the discovery
came late.

**Also conceded (new, 2026-09-08)**: the barrier exclusion is narrower than
first stated -- the string method shows the *landscape* admits a barrier-free
path, not that SGD's trajectory is barrier-free. The exclusion table now says
this.

## To R6 -- conceded on scope, disputed on value

**Conceded**: the toy setting is not a model of real training, beta is not
computable on CIFAR, and the CIFAR result measures magnitude change rather
than threshold displacement. The bridge is an analogy and the paper says so.

**Disputed**: the four-parameter setting is what makes the exclusions
possible. Exact Hessians, exact solution-set geometry, and analytic beta are
available *because* the model is small. A larger setting would produce a less
interpretable version of the same claim. The tradeoff is explicit.

---

# What this review would cost us

If written by a determined reviewer, **R1 and R4 together are close to
sufficient for rejection at a top venue**: a one-line substitution verified on
self-constructed families, under one optimizer. R2 and R6 are scope
limitations we have already conceded in the text, which blunts them. R3 is
answerable but the answer is weaker than we would like. R5 is the one we would
win.

**The two changes that would most reduce the damage**, neither of which we can
make now:

1. A family from a structurally different construction route (answers R4).
2. A threshold-exponent measurement in any non-toy setting (answers R3 and R6).
