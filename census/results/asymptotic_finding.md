# Finding (independent reimplementation, 2026-09-08): the (8/3)eps^{3/2} form
# is an a->1+ asymptotic and must be labelled as one

Found by reimplementing D(a) from its definition (`independent/reimpl.py`,
imports only numpy). **Both implementations agree to 1.6e-11 relative**, so
no computed number is affected. The finding is about the **stated form**.

## The approximation is badly inaccurate over the measured range

| a | exact D(a) | (8/3)eps^{3/2} | error |
|---|---|---|---|
| 1.02 | 0.005286 | 0.007542 | **+42.7%** |
| 1.10 | 0.057116 | 0.084327 | **+47.6%** |
| 1.50 | 0.553931 | 0.942809 | **+70.2%** |
| 3.00 | 3.194935 | 7.542472 | **+136.1%** |

Every *use* of D in the project calls `dip_depth()` (exact), so no result
moves. But a reader substituting the closed form gets a materially wrong
number, and a reviewer checking our arithmetic hits this immediately.

## Consequence 1: state the theorem in D(a), not in the asymptotic

T37's bound is |w2| >= 2m/(kappa*D(a)). It is **verified against 66 solvers
using measured G\*(a)**, so the verification is sound. But if the paper states
the theorem with (8/3)eps^{3/2} substituted, **the stated bound is wrong by up
to 136% while the verified one is not.**

**Action**: state the theorem in terms of D(a), and give the asymptotic
separately, explicitly labelled as the a -> 1+ limit.

## Consequence 2: the exponent chain -- checked, and it survives

The onset exponent traces to required scale ~ eps^{-beta} with beta = 3/2 from
the asymptotic. If the true D deviates by 43-136% across the measured range,
is the exponent attributed to a regime it does not describe?

**Local logarithmic slope of the exact D(a):**

| eps | local slope |
|---|---|
| 0.02 | 1.4936 |
| 0.10 | 1.4696 |
| 0.30 | 1.4097 |
| 0.60 | 1.3475 |
| 2.00 | 1.2219 |

| range | slope |
|---|---|
| **onset range (eps 0.03-0.60)** | **1.4364** |
| onsets excluding the 2k cell | 1.4549 |
| asymptotic limit (eps -> 0.01) | 1.4982 |

**The slope over the range the onsets actually span is 1.4364 against the
claimed 1.5 -- a 4.2% deviation.** So the asymptotic has very nearly the right
*slope* over the measured range even though its *prefactor* is badly wrong.
The prefactor cancels in an exponent; the slope is what matters.

Downstream effect of using 1.4364 instead of 1.5:

| beta | onset exponent | cost exponent |
|---|---|---|
| 1.5000 (claimed) | -0.7448 | -1.343 |
| **1.4364 (true, over range)** | **-0.7778** | **-1.286** |

The shift is **0.0330**, which is **24% of the registered band width**
(+-0.135) and well inside it. Measured onset exponent -0.7340 remains
consistent with both.

**Conclusion**: the chain is sound. The exponent 3/2 is a good description of
the measured range (within 4.2%), not merely of the limit. But the paper
should quote beta = 3/2 as the analytic limit **and** note that the effective
local exponent over the measured range is ~1.44, so the derivation is not
being applied outside where it holds.

---

# Effective beta across all five families: the substitution is not neutral

Prompted by review: each family's beta is an **asymptotic** (1 + 1/q as
eps -> 0), and each has an **effective local value** over the eps range its own
onsets actually span. If the through-origin slope shifts materially when
effective beta replaces asymptotic beta, that must be stated.

| family | onset eps range | beta_asym | beta_eff | pred(asym) | pred(eff) | measured |
|---|---|---|---|---|---|---|
| q4 | 0.025-0.250 | 1.2500 | **1.2285** | -0.8938 | -0.9095 | -0.8305 |
| q2 | 0.015-0.250 | 1.5000 | **1.4630** | -0.7449 | -0.7637 | -0.6749 |
| q1 | 0.040-0.600 | 2.0000 | **1.8409** | -0.5586 | -0.6069 | -0.6521 |
| q0.667 | 0.100-0.400 | 2.4993 | **2.2391** | -0.4470 | -0.4990 | -0.5000 |
| family A | 0.030-0.600 | 1.5000 | **1.4326** | -0.7449 | -0.7799 | -0.7340 |

Effective beta is **below** asymptotic beta in every family (D's local slope
falls as eps grows), by 1.7% (q4) to 10.4% (q0.667).

## The two diagnostics disagree about which beta is right

| | asymptotic beta | effective beta |
|---|---|---|
| mean per-family \|error\| | 0.0581 | **0.0520** (better) |
| through-origin slope | **1.0984** | 1.0547 |
| slope vs measured alpha = 1.1173 | **1.7% apart** (better) | 5.6% apart |

Per family: effective beta **improves** q1 (0.094 -> 0.045) and q0.667
(0.053 -> 0.001) but **worsens** q4, q2 and family A -- including family A,
whose agreement degrades from 0.011 to 0.046.

**So using effective beta improves the average per-family fit while worsening
the slope's agreement with the independently measured alpha.** The two
diagnostics point opposite ways, and that disagreement is itself the finding:
the relationship is not resolved finely enough to distinguish a 3-10% change
in beta.

## What the paper must do

**Report both, and say which is used.** The headline figure and the through-
origin slope use **asymptotic beta**, because beta is defined analytically
there and the x-axis (1/beta) is then a derived quantity with no measurement
error. Using effective beta would put a measured quantity on both axes and
make the slope a regression of two noisy variables.

**State the sensitivity**: the through-origin slope is 1.0984 (asymptotic) or
1.0547 (effective) against alpha = 1.1173 [0.999, 1.236]; **both lie inside
alpha's confidence interval**, so the conclusion -- slope consistent with
independently measured alpha -- is unchanged. The choice affects the headline
number by 4%, well inside the interval, and it must not be presented as a
1.7% agreement without noting that the alternative gives 5.6%.

This is the same class as the alpha window question (1.1172 vs 1.1084): a
quantity with two defensible values where the paper states which it uses and
why, rather than quoting the more favourable one.
