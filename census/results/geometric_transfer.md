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
