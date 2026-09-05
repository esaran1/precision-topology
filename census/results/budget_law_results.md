# The budget law: an effective threshold displaced from the analytic one by a power law

Registered in `arrhenius_prediction.md`: alpha measured first (1k-160k
budgets), the onset exponent -2*alpha/3 and its band fixed **before any onset
was measured**, and the floor-vs-failure criteria fixed before the
largest-budget cell resolved. Data: `budget_alpha.csv`, `onset_law.csv`,
`onset_curves.csv`, `joint_criterion.csv`.

## The headline, which needs no fit

**At a = 1.25 the solve rate is 0/200 at the committed 2,000-step budget and
40/40 = 100% at 80,000 and 160,000 steps.** Same task, same architecture, same
analytic threshold, one hyperparameter changed. The value the project reported
as unreachable is unanimous under a longer budget.

## Reachable scale grows as a power law in budget

| budget | 1k | 2k | 4k | 8k | 16k | 40k | 80k | 160k |
|---|---|---|---|---|---|---|---|---|
| median terminal \|w2\| | 0.87 | 2.95 | 7.08 | 17.45 | 37.92 | 97.48 | 170.7 | 264.2 |
| solve rate | 0.000 | 0.000 | 0.067 | 0.833 | 0.900 | 0.933 | **1.000** | **1.000** |
| stall fraction | 0.267 | 0.200 | 0.100 | 0.100 | 0.100 | 0.033 | 0.000 | 0.000 |

**alpha = 1.1172** (R^2 = 0.983 over more than two decades).

## The onset moves as predicted

Required scale diverges as |w2| ~ c/D(a) ~ (a-1)^{-3/2}; equating to
reachable |w2| ~ B^alpha gives (a_onset - 1) ~ B^{-2*alpha/3}.

**BRACKETED CELLS: 4 of 4.** Every onset is located, not bounded: each has a
grid value at rate >= 50% and a strictly smaller one below it.

| budget | onset | bracketing evidence |
|---|---|---|
| 2,000 | **1.60** | 1.60: 0.550 -> 1.55: 0.450 |
| 8,000 | **1.18** | 1.18: 0.850 -> 1.16: 0.475 |
| 32,000 | **1.06** | 1.06: 0.650 -> 1.05: 0.000 |
| 128,000 | **1.03** | 1.035: 1.000 -> 1.03: 0.700 -> 1.025: 0.000 |

**Measured exponent -0.7275** (R^2 = 0.9865, n = 4) against the registered
**-0.7448**, band [-0.895, -0.595]: **within band**, and within 2.3% of the
point prediction.

**Power of this test, stated before the fit quality**: n = 4. A power law on
four points is weak evidence however well it fits. The result is reported as
**consistent with** the predicted exponent, not as confirming it. What
carries more weight than the exponent is that the onset moved from 1.60 to
1.03 -- a factor of 20 in (a_onset - 1) -- across a 64x budget range, in the
predicted direction, with the prediction fixed in advance.

The floor-vs-failure criteria were not needed: the 128k cell bracketed
genuinely (1.000 at 1.035, 0.700 at 1.03, 0.000 at 1.025) rather than running
out of grid.

## The claim

Approaching an expressivity threshold, the optimization budget required for
correct classification **diverges as a power law**, so any finite budget
produces an **effective threshold displaced from the analytic one by a
predictable amount**. The four manipulations -- initialization scale, step
size, optimizer, budget -- are one mechanism: they all change how far training
travels before it stops.

## Part 3 fixed: the joint criterion, with FEWER fitted parameters

The 1-D criterion (terminal |w2| >= 2m*/G*(a)) carries **one fitted constant**
(m* = 0.1, chosen to minimise error). The joint criterion carries **zero**: it
asks whether the terminal (w1, b1, w2) admits *any* b2 that classifies, with
b2 set analytically to the interval midpoint. So the improvement is
correctness, not capacity.

| criterion | fitted parameters | mean \|err\| | max \|err\| | signed mean | exact cells |
|---|---|---|---|---|---|
| 1-D \|w2\| threshold | **1** | 0.283 | 0.925 | +0.283 | 9/21 |
| **joint (w1,b1,w2)** | **0** | **0.100** | 0.500 | +0.100 | **15/21** |

Correlation with observed rates 0.91. Every catastrophic miss of the 1-D
version is removed: init scale 0.3 at a = 1.25 (0.925 -> **0.000**, observed
0.000), a = 1.25 standard (0.550 -> **0.000**), a = 1.45 lr 3e-3
(0.400 -> **0.000**).

**Still not sufficient**: the signed mean is +0.100 and never negative, so an
adequate geometric configuration remains necessary-but-not-quite-sufficient.
The residual is b2 failing to land, which this criterion deliberately factors
out. The three worst cells are all near-onset, where rates are steepest.

## P-stall: falsified as registered

Registered as budget-independent. **Observed: the stall fraction falls to
0.000** (0.267 at 1k -> 0.000 at 80k). Both stalling and solving reach their
extremes over the same range; only the schedules differ -- between 2k and 8k
solving moves 0.000 -> 0.833 while stalling moves only 0.200 -> 0.100.

The supported claim is the weaker one: **two mechanisms with different time
constants**, not two distinct mechanisms. The log-2 saddle (constant
predictor) is a genuine critical point and a real attractor, unlike the
correctly-classifying points, but it is escaped given enough budget.
