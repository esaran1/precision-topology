# Review objection 1: the amplification measure. Upheld in part, refuted in part.

**Provenance: raised in external expert review** by an author of the paper
this work builds on, after reading the released code and results. Recorded
here because the objection changed a committed claim.

> "For the piecewise family, bounded output weight does not imply bounded
> amplification: the required scale can diverge through the first-layer
> weight."

## Upheld: family B's requirement diverges, and T28 was wrong

Family B's `f_alpha(t) = max(t, alpha*t)` is **positively homogeneous of
degree 1**, so the network depends on the weights only through the
scale-invariant direction and the **product |w1*w2|**. Reasoning from |w2|
alone is meaningless for this family -- exactly as the objection states.

Maximum class gap at |w1| = 1 is **0.4*|alpha| exactly** (fitted exponent
1.000000, constant ratio over three decades), so required amplification
**diverges as 1/|alpha|** and family B's exponent is **beta_B = 1**, not
undefined.

**T28's "A_req is undefined for Family B" is false and is corrected.** The
original analysis computed the requirement in the wrong variable.

## Refuted: the product measure cannot be the discriminating variable

For the four constructed (non-homogeneous) families, the optimal traversal
scales as w1* ~ eps^{1/q} while req|w2| ~ eps^{-(1+1/q)}. Their product is
therefore **eps^{-1} for every q**: an algebraic identity, not a measurement.

| family | beta(\|w1*w2\|) | beta(\|w2\|) | w1* exponent | 1/q |
|---|---|---|---|---|
| q4 | **1.001** | 1.239 | 0.238 | 0.250 |
| q2 | **1.011** | 1.492 | 0.481 | 0.500 |
| q1 | **1.004** | 1.797 | 0.793 | 1.000 |
| q0.667 | **1.027** | 2.152 | 1.125 | 1.500 |

**beta(product) = 1.00 +- 0.03 across families whose beta(|w2|) spans
1.24-2.15.** A measure taking the same value for every family **cannot**
predict the observed spread of onset exponents (-0.50 to -0.83). The product
is ruled out by an identity, not by preference.

The discriminating quantity is **req|w2| evaluated at the optimal traversal**,
whose exponent recovers the fold-depth beta to within 1% for the shallow
families (1.239 vs 1.250; 1.492 vs 1.500) and less well for the steep ones
(1.797 vs 2.000; 2.152 vs 2.500). Using the measured rather than idealised
beta slightly **improves** the law's fit: mean |error| 0.049 vs 0.070.

**So the four-family law is not an artifact of a shared error.** It survives
the objection, restated in the corrected measure.

## Homogeneity checked for every family (not assumed)

`f(c*x)` picks up a `c^q` factor on the nonlinear term, so homogeneity holds
only at q = 0. Verified numerically: family B deviates by 1.1e-16
(homogeneous to machine precision); q0.667/q1/q2/q4 and family A deviate by
2.4-7.8 (O(1)). **Family B is the only homogeneous family**, so the error
does not propagate to any other point on the line.

## The scope condition, derived from homogeneity alone

Stated **independently of family B's measured behaviour**, which is what
makes it a scope condition rather than a post-hoc exclusion:

> A homogeneous activation family has no fold *width* for the traversal to
> match. Its requirement therefore collapses into the product |w1*w2|, where
> the exponent takes the identity value 1 and carries no family-
> discriminating information. **The law's class is the non-homogeneous
> families, whose requirement diverges through a fold depth with the optimal
> traversal at a family-specific scale.**

Nothing in that chain references family B's flat onset.

## What remains open

Training over-shoots family B's requirement by 900-1,200x and still mostly
fails (at alpha = -0.002, required |w1*w2| = 25, reached 23,051, solve rate
0.033), and the solve rate *falls* as budget rises. In the product measure
alpha_product = **1.049**, so if beta = 1 the predicted exponent is **-1.05**
against family B's measured **0.000** -- a gap of 1.05.

**Family B lies outside the law's class by homogeneity, and its flat exponent
is not accounted for by any amplification measure we tested.** That is an
open question, not a resolved boundary case.

(Bookkeeping note: alpha(product) != alpha(w1) + alpha(w2) because these are
medians, and the median of a product is not the product of medians. The
identity holds per-run. alpha_product = 1.049 excludes the B=1000 cell, where
8/30 runs stalled and the product median was a 40x outlier; including it
inflates the estimate to 1.467.)
