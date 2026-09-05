# §7 draft: the optimizer-dependence sentence

**Written 2026-09-05, BEFORE the SGD terminal-scale cells reported.** Both
branches drafted in advance so that whichever lands is reported as
anticipated rather than constructed after the fact. Provenance is verifiable
from this file's commit timestamp, which precedes the SGD result.

## Branch A -- if alpha_SGD lands in the registered band [0.38, 0.58]

Draft sentence for §7:

> The effective expressivity boundary depends on **which optimizer** is used,
> not only on how long it is run. Terminal weight scale grows as B^alpha with
> alpha ~ 1 under Adam, whose normalization cancels the loss gradient's
> |w2|^{-1.085} decay, but as B^0.48 under plain SGD, which inherits that
> decay directly. Through epsilon_onset ~ B^{-2*alpha/3} this predicts onset
> exponents of **-0.745 for Adam against -0.319 for SGD** -- a factor of 2.3
> in how fast the effective boundary retreats with budget. An architecture
> comparison is therefore confounded by optimizer choice through a mechanism
> that can be derived and quantified, not merely by budget.

Why this is stronger than the budget result alone: budget-dependence can be
read as "train longer, get closer", which is implicit in any convergence
analysis. **Optimizer-dependence of the exponent is not**, because it says
two practitioners running the same architecture to the same loss with
different optimizers face different effective capability boundaries. The
derivation locates the cause precisely -- in whether the update normalizes by
gradient magnitude -- rather than attributing it to optimizer quality.

Status if this branch holds: the chain is derived end to end. Gradient
exponent measured (1.085), Adam's normalization measured (step exponent
0.005), alpha derived for both optimizers, onset exponent predicted from
alpha and the analytic depth exponent, and the SGD value predicted before
measurement.

## Branch B -- if alpha_SGD misses the band

Report plainly: the gradient-flow route, applied to the optimizer where it
should hold, does not reproduce the measured SGD exponent. That **falsifies
the mechanism** (the claim that alpha follows from the gradient's |w2|
scaling), not the optimizer choice -- as the registration states in advance.

The paper then keeps alpha as a **measured** quantity. What survives intact:
the Adam normalization story (gradient decays |w2|^{-1.085}, realized step is
scale-free at |w2|^{-0.005}, so alpha ~ 1 for Adam), which is three
independent measurements and does not depend on the SGD prediction; the
registered onset prediction -0.745 and its measurement -0.734; and the
family-B double dissociation. Section 6 is unchanged; §7 drops the
optimizer-dependence claim and says the derivation of alpha remains open.

## Fixed in advance either way

- The registered band is [0.38, 0.58] around alpha_SGD = 0.479, and the
  dependent onset prediction is -0.319 (band [-0.42, -0.22]).
- SGD's alpha is fitted on the **population** median (stallers included), the
  same quantity as Adam's 1.1172, since the onset measurement is a population
  quantity (`onset_more.rate_at` counts stallers as failures).
- If SGD's stall fraction differs greatly from Adam's, the escaper-only fit
  is reported alongside, but the population fit is the registered test.
