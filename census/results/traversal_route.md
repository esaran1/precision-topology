# The traversal route: derived, verified, and what it can and cannot buy

Prompted by R4 (four of five families come from one template). **Derivation
and numerical verification only -- no sweep run.** The question is whether
this route supplies a second construction mechanism worth a night of compute.

## Derivation

Fix a non-monotonic activation at a **fixed** a0 > 1, so the fold's *shape*
never changes, and vary the **input scale**: the data window traverses only
|u| <= s of the fold. Since c is absorbed into w1, this is not a new
activation family at all -- it is the same fold, traversed to a different
depth.

Near a fold minimum with f(t) - f(t*) ~ C|t - t*|^p, a window of half-width s
placed optimally spans a class gap

    G(s) ~ C * s^p * k(window ratios)

so **beta_traversal = p, the order to which the fold's minimum vanishes.**
For family A's sin fold the minimum is **quadratic** (f''(t*) = 1.118 at
a0 = 1.5, nonzero), giving **beta = 2**.

## Numerical verification, three decades

Max class gap vs traversal scale s, w1 = s/OUTER_MAX:

| s | max gap | local slope |
|---|---|---|
| 3e-4 | 1.006e-08 | -- |
| 1e-3 | 1.118e-07 | 1.9998 |
| 1e-2 | 1.113e-05 | 1.9976 |
| 1e-1 | 1.103e-03 | 1.9992 |
| 3e-1 | 9.598e-03 | 1.9694 |

**Fitted beta over three decades: 1.9945**, against the derived 2. Clean.

Other fold orders, to check the derivation is about vanishing order and not
about sin specifically:

| fold | derived beta | measured |
|---|---|---|
| \|t\| (kink) | 1 | **1.0000** |
| \|t\|^1.5 | 1.5 | **1.5000** |
| sin fold (quadratic) | 2 | **1.9967** |
| \|t\|^3 | 3 | **3.0000** |
| t^4 | 4 | **4.0000** |

The derivation is exact.

## Verdict: NOT worth the night, and here is the honest reason

**beta_traversal = 2.00 collides with q1's beta = 2.00.** So the route does
not supply a *new* beta -- it supplies a **second route to a beta we already
have**.

Worse for R4's purposes: obtaining a *different* beta from this route requires
choosing a fold whose minimum vanishes to a different order, which is **the
same knob the q-families already turn** (their beta = 1 + 1/q comes from
f' vanishing to order q). Traversal scaling with a **fixed** activation gives
only beta = 2. So this is not a structurally different mechanism; it is the
same mechanism reached by rescaling the input instead of reshaping the
function.

## What it would nonetheless buy, if run

A sweep at beta_traversal = 2 would be a **cross-route replication at matched
beta** -- the same test q2-vs-family-A already provides at beta = 1.5, where
two different functional forms gave -0.6749 and -0.7340 (agreeing within grid
resolution). A second such check at beta = 2 would strengthen the
"construction is faithful" argument, but it **does not answer R4**, whose
complaint is that the *x-axis positions* come from one template.

**Recommendation: do not run it.** It costs a night to add a fifth point at a
beta we have already measured, via a mechanism that is a reparametrization of
the one we used. R4 remains open and should be stated as open. The honest
answer to R4 is that the strongest available check -- a genuinely independent
construction route spanning several beta -- was not performed, and that family
A (independent, not ours) landing on the line at beta = 1.5 is the only
cross-route evidence we have.
