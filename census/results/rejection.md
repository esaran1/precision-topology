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

## To R1 -- the objection makes a false prediction

R1 says the law is one line of algebra: set B^alpha = eps^{-beta}, get
eps ~ B^{-alpha/beta}. The algebra is indeed trivial. **The objection
nonetheless fails, because it treats a measured dynamical fact as an a priori
one.**

The substitution requires that terminal weight scale follow a power law in
budget. **That is not given; it is measured, and it is not universal:**

- **alpha is not range-stable.** Fitted over every contiguous sub-range it
  falls monotonically from **1.5109** (1k-4k) to **0.7193** (40k-160k), with
  R^2 >= 0.983 in every window. Growth is sub-power-law -- locally
  power-law-like at every scale with a drifting exponent, the signature of a
  logarithmic or saturating process. A "trivial substitution" presumes a
  quantity that does not exist as a range-independent constant.
- **The chain does not close under SGD.** If the relationship were mere
  algebra, it would close for *any* optimizer with any alpha: measure alpha,
  divide by beta, done. It does not. The exponent ratio is **0.447** against
  the alpha ratio **0.643** (30% gap, on the quantity where grid resolution
  partially cancels), and for q4 the exponent is **-0.8305** under Adam and
  **+0.0056** (bounded |exp| < 0.0185) under SGD -- a factor of at least 45 at
  fixed beta.

**So the objection makes a prediction, and the prediction is false.** If the
law were a definitional consequence, it could not fail under an optimizer
change; it would follow from the definitions regardless of dynamics. Two of
the two optimizers tested were required to satisfy it, and one does not.

The same point holds for family B, which has a **diverging** requirement
(beta = 1, derived analytically after external review corrected our error),
**exceeds** it by 900-1200x in training, and **still fails** -- its onset
exponent is **+0.25** over a matched 64x range, the wrong sign entirely.
Algebra cannot produce a counterexample to itself.

**What the algebra buys, precisely**: given that (i) terminal scale grows as a
power law over a stated range, (ii) required scale diverges with a derivable
exponent, and (iii) training reaches solutions by growing |w2| along a
fold-amplifying direction, the onset exponent follows. **Each of (i)-(iii) is
an empirical claim that can fail. (i) holds only over restricted windows,
(iii) fails for family B, and the conjunction fails under SGD.** The
contribution is establishing where the conjunction holds and exhibiting two
regimes where it does not -- not the division.

**Conceded on wording only**: the paper should not say "we derived the power
law and it holds", which oversells. It should say the chain closes
quantitatively under one optimizer for families where the mechanism applies,
and identify the two regimes where it demonstrably does not.

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

*Attempted 2026-09-08*: a second construction route was derived and verified
-- fixing the activation and varying the **traversal scale** rather than the
functional form. It gives **beta = 2 exactly** (verified 1.9945 over three
decades; the derivation beta = order of the fold minimum's vanishing is exact
across five test folds). But **beta = 2 collides with q1**, and obtaining a
different beta from that route requires varying the fold's vanishing order --
**the same knob the q-families turn**. It is a reparametrization, not an
independent mechanism, and was **not run** (`traversal_route.md`). R4 stands.

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
