# CORRECTION (2026-09-06): family B's requirement DOES diverge. T28 and T44's
# limiting case are wrong.

Raised in expert review: "For the piecewise family, bounded output weight does
not imply bounded amplification: the required scale can diverge through the
first-layer weight." **The objection is correct.** We argued family B's
requirement from |w2| alone and never checked w1.

## 1a: the derivation, exact

Family B's f_alpha(t) = max(t, alpha*t) is **positively homogeneous of degree
1**: f(c*t) = c*f(t) for c > 0. Hence

    w2 * f(w1*x + b1) = (w1*w2) * f(x + b1/w1) * sign-corrected,

so the network depends on the weights only through the scale-invariant
direction plus the **product |w1*w2|**. The amplification measure is the
product, never |w2| alone -- exactly as the reviewer states.

Maximum achievable class gap at |w1| = 1:

| alpha | max gap | required \|w1*w2\| at m = 0.01 |
|---|---|---|
| -0.5 | 2.00e-1 | 0.10 |
| -0.05 | 2.00e-2 | 1.00 |
| -0.005 | 2.00e-3 | 10.0 |
| -0.001 | 4.00e-4 | 50.0 |

**gap = 0.4*|alpha| exactly** (fitted exponent 1.000000; ratio constant at
0.4 over three decades). So required amplification **diverges as 1/|alpha|**,
and family B's depth exponent is **beta_B = 1 exactly** -- not undefined.

## What this overturns

1. **T28's "A_req is undefined for family B" is WRONG.** A_req diverges as
   1/|alpha|. The earlier analysis computed the requirement in the wrong
   variable (|w2|, which homogeneity makes meaningless on its own).
2. **Family B is not the beta-undefined endpoint of the four-family law.**
   It has beta = 1, the *steepest* of every family tested (q4 1.25, q2/A 1.5,
   q1 2.0, q0.667 2.5), so the law predicts the LARGEST onset exponent in
   magnitude: **-alpha_budget/beta_B = -1.1172**.
3. **Measured family B onset exponent: 0.0000.** Family B is therefore a
   **counterexample to the law, not its limiting case.** The published
   "double dissociation" does not hold: family B has a diverging requirement
   AND a flat onset.

## 1b/1c: which outcome class -- (ii), with a twist that makes it worse

Registered classes: (i) nothing diverges; (ii) diverges but training reaches
it; (iii) diverges and training does not reach it.

**Outcome (ii) holds, by enormous margins:**

| alpha | budget | solve rate | required \|w1*w2\| | reached \|w1*w2\| (median) | over-shoot |
|---|---|---|---|---|---|
| -0.05 | 2,000 | 0.567 | 1.00 | 94.1 | 94x |
| -0.05 | 32,000 | 0.433 | 1.00 | 1,159.7 | 1,160x |
| -0.01 | 32,000 | 0.133 | 5.00 | 5,742.3 | 1,148x |
| -0.005 | 32,000 | 0.133 | 10.0 | 11,141.5 | 1,114x |
| -0.002 | 32,000 | 0.033 | 25.0 | 23,051.5 | **922x** |

Training reaches 900-1,200x the required amplification and **still mostly
fails**. Worse, at every alpha the solve rate **falls** when the budget rises
from 2,000 to 32,000 (0.567 -> 0.433, 0.300 -> 0.133), while |w1*w2| grows
~50-1000x. More amplification, less success.

So the correct statement is not "requirement within reach, therefore no gap".
It is: **family B's requirement diverges, training over-shoots it by three
orders of magnitude, and findability still fails -- so for family B the
required-scale account does not govern findability at all.**

## Status of the law

The four-family relationship eps_onset ~ B^{-alpha/beta} (T44) stands **on its
own four families**, whose betas were verified analytically and whose measured
exponents track 1/beta (through-origin slope 1.0969 vs measured alpha 1.1172).
What it loses is its claimed limiting case and the dissociation framing:

- family B is **not** evidence that the law "predicts an absence";
- family B is an **unexplained counterexample** with beta = 1 and a flat
  onset, and must be reported as such;
- the law's scope is therefore narrower: it holds across the four constructed
  fold-depth families, and there exists at least one family (B) where a
  diverging requirement does not produce a budget-dependent onset.

Why family B might differ, stated as hypothesis and NOT as result:
homogeneity means B can supply unlimited amplification by trading w1 against
w2 at no cost, so amplification is never the binding constraint; whatever
limits family B's findability is something else, and we have not identified
it. That is an open question, not a resolved one.
