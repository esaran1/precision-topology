# Geometric transfer under SGD: ATTEMPTED, NOT EXECUTABLE

Registered 2026-09-07 (P-abs, P-ratio, P-null, plus a refined q0.667 grid).
**The test did not execute.** This is an instrument limitation, not a finding
about the law, and is reported as such.

## What was attempted

Whether the **geometric** part of the law (onset exponent ~ 1/beta, beta
derived analytically with no training) transfers across optimizers, using the
ratio e(q4)/e(q0.667), in which **alpha cancels exactly**. The ratio was the
test precisely because the dynamical part (alpha) had already failed the
cross-optimizer comparison (T45).

## Why it did not execute

**q0.667 (beta = 2.5) never solves under SGD.** Rate 0.000 at every grid point
from eps = 0.6 down through the refined 1.25x grid, at all four budgets
including 128,000. **Zero bracketed cells.** Plain SGD at lr = 0.3 -- the only
learning rate that solves family A at all (`step_size_results.md`) -- cannot
reach this family's solutions within the budgets tested.

**q4 (beta = 1.25) was unresolved, not flat.** Its onset pinned at eps = 0.25
for all four budgets, because the next grid step down (0.15) failed at every
budget while 0.25 succeeded at every budget. The true onset lies in
(0.15, 0.25] at all four budgets and the ~1.6x grid could not locate it. The
apparent exponent of 0.0000 is a grid artifact.

**Diagnosis.** Both eps grids were chosen from the **Adam** sweeps, where
onsets sit at 1.03-1.60. Under SGD onsets sit **higher and compress** (family
A moves only 1.60 -> 1.16 over 64x against Adam's 1.60 -> 1.03), so for q4 the
entire budget range falls inside one grid interval, and for q0.667 the onset
is beyond eps = 0.6 or unreachable.

## What is NOT claimed

**This is not evidence that the geometric relationship fails under SGD.** The
registered null (ratio ~ 1.0) concerned a *measured* ratio; no ratio was
measured. Reporting a failure here would be reporting the grid, not the law.

**q0.667 is deliberately not chased to larger eps.** If it requires eps > 0.6
to solve at all, any onset found there would sit far from the threshold, where
beta = 2.5 -- a near-threshold expansion in (a-1)^{5/2} -- may not apply. A
number obtained in that regime would not measure what beta describes.

## What is being measured instead

**q4 alone, on a refined 1.08x grid across (0.15, 0.25].** One family, one
interval. It answers whether q4's onset is genuinely budget-independent under
SGD or merely unresolved, and if it moves it yields a **same-family
cross-optimizer comparison at fixed beta = 1.25** against Adam's measured
-0.8305 -- isolating the dynamical difference without needing the ratio.

## Status of the geometric claim

**Open.** Tested under Adam across four families (T44, through-origin slope
1.1240 [1.008, 1.240] against an independently measured alpha 1.1173
[0.999, 1.236]); **untested across optimizers**, because the second optimizer
cannot reach the families the test requires at feasible budgets. The
four-family result remains a single-optimizer result and the paper should say
so.

---

# q4 under SGD, refined: the onset is FLAT, and the bound is tight

Refined 1.08x grid across (0.15, 0.25], **4 of 4 cells bracketed**:

| budget | q4 onset (eps) |
|---|---|
| 2,000 | 0.18376 |
| 8,000 | 0.17015 |
| 32,000 | 0.18376 |
| 128,000 | 0.18376 |

Three of four budgets give the **identical** grid value; the fourth differs by
one 1.08x step (non-monotonically -- it is *lower* at 8k than at 32k and 128k,
so the variation is grid noise, not a trend).

**Fitted exponent: +0.0056.** Total variation 1.08x across a **64x** budget
range.

## Stated as a bound, not a null

Flat at 1.08x resolution over 64x bounds the exponent:

    |exponent| < ln(1.08) / ln(64) = **0.0185**

and the fitted value (0.0056) is well inside that. So:

| family beta = 1.25 | exponent |
|---|---|
| **Adam** | **-0.8305** |
| **SGD** | **+0.0056**, \|value\| < **0.0185** |

**SGD's onset exponent for q4 is at least 45x smaller in magnitude than
Adam's, at fixed beta.** The coarse grid's apparent flatness was an artifact
(it reported 0.25 everywhere; the refined onset is 0.184), but the refined
measurement finds the onset genuinely does not move.

**The registered absolute prediction is missed badly**: -alpha_SGD/beta =
-0.5750 against a measured +0.0056, a miss of 0.58 -- far outside any
resolution argument.

## Reading, with both possibilities stated

Family A **does** move under SGD (1.60 -> 1.16 over 64x, exponent -0.3255), so
SGD onsets are not generally static. Two readings of q4's flatness:

1. **A beta-dependent difference between optimizers**: SGD's onset moves for
   beta = 1.5 (family A) but not for beta = 1.25 (q4). If real, this is
   interesting in its own right and is **not** what the law predicts -- the
   law says smaller beta gives a *steeper* exponent, and q4 has the smallest
   beta of any family tested.
2. **Movement below resolution**: q4's onset moves by under 8% across 64x.
   Even so, that is bounded at |exponent| < 0.0185 against Adam's 0.8305.

Both readings agree on the quantitative content: **at fixed beta = 1.25, the
two optimizers differ by at least a factor of 45 in onset exponent.** That is
a same-family cross-optimizer result and it does not depend on the ratio test
that could not run.

## Consequence for the law

The law predicts the exponent should scale as -alpha/beta, so q4 (the smallest
beta) should have the **steepest** exponent under either optimizer. Under Adam
it does (-0.8305, steepest of four families). **Under SGD it is flat.** So the
1/beta scaling, which holds across four families under Adam, **does not
reproduce under SGD for the one family that could be measured.**

Combined with T45 (the alpha-dependence holds directionally but the exponent
ratio 0.447 misses the alpha ratio 0.643), the honest position is:

> **The four-family geometric relationship is an Adam result.** Its transfer
> to other optimizers is untested for three of four families (unreachable at
> feasible budgets) and **fails for the one family that could be measured**.
