# The budget law in a standard setting: control 2a fails, experiment stops

**Outcome 4 of the four registered in `mnist_budget_law_prediction.md`: the
setting does not exhibit the phenomenon.** Reported with the prominence the
registration requires. The paper stays scoped to the toy setting.

Date: 2026-09-11. Wall time: about 6 minutes of CPU across pilot, ID
estimation, and controls — the experiment stopped early by design, not by
budget.

---

## Summary

At no bottleneck width from 1 to 8, and at no capability threshold from 0.80
to 0.97, do monotonic activations fail where non-monotonic ones succeed on
MNIST. **At the narrowest bottleneck the ordering is reversed and
significantly so**: monotonic `a <= 1` beats non-monotonic `a > 1` by 9.4
accuracy points (`p = 0.0002`, permutation, 20,000 resamples).

Per the registered failure condition — *"if monotonic `a` reaches the
criterion at >= 50% at the narrow bottleneck, the setting does not exhibit the
phenomenon and the experiment stops"* — Parts 3, 4 and 5 were not run.

## Intrinsic dimension of the pre-bottleneck representation

Measured on the 256-unit ReLU layer of trained networks, 3 seeds, 5,000 test
points, three nonlinear estimators (`results/mnist_fold_id.csv`):

| estimator | mean | min | max |
|---|---:|---:|---:|
| MLE `k = 10` | 8.04 | 8.03 | 8.05 |
| MLE `k = 20` | 8.10 | 8.05 | 8.14 |
| TwoNN | 13.67 | 13.58 | 13.83 |

Spread **8.03 to 13.83, a factor of 1.72** — below the factor-2 threshold the
registration set for proceeding, and far tighter than the 1.99x systematic
disagreement that closed the earlier CIFAR width-axis attempt (T39). The
narrow bottleneck was therefore placed at `w = 8` and probed down to `w = 1`;
the wide comparison would have been `w = 32`, above `2 x ID ~ 28`.

## The pilot, and why no capability threshold exists

`results/mnist_fold_pilot.csv` (widths 4/6/8) and
`mnist_fold_pilot_narrow.csv` (widths 1/2/3), `a` in {0, 0.5, 1, 1.5, 2.5, 4},
5 seeds per cell, 2,000 steps.

Best mean test accuracy by width, monotonic against non-monotonic:

| width | best `a <= 1` | best `a > 1` | gap |
|---:|---:|---:|---:|
| 1 | **0.8696** | 0.7922 | **+0.0775** |
| 2 | 0.9473 | 0.9491 | −0.0018 |
| 3 | **0.9641** | 0.9634 | +0.0007 |
| 4 | 0.9700 | 0.9711 | −0.0012 |
| 6 | 0.9730 | 0.9742 | −0.0012 |
| 8 | **0.9741** | 0.9739 | +0.0002 |

From width 2 up, the two classes are indistinguishable — every gap is under
0.2 accuracy points, smaller than the seed-to-seed spread within a single
cell. At width 1, where a gap finally appears, **it has the wrong sign.**

The capability criterion was to be chosen so that monotonic fails and
non-monotonic succeeds. We swept every threshold from 0.80 to 0.97 at each
width to find one:

| threshold | width | monotonic success | non-monotonic success |
|---:|---:|---:|---:|
| 0.80 | 1 | 1.00 | 0.33 |
| 0.85 | 1 | 0.90 | 0.07 |
| 0.95 | 2 | 0.10 | 0.47 |
| 0.97 | 4 | 0.33 | 0.40 |
| 0.97 | 8 | 1.00 | 0.80 |

**No threshold produces the registered pattern** (monotonic < 50%,
non-monotonic >= 50%). The closest cell — width 2 at 0.95 — has non-monotonic
at 0.47, itself below 50%, and the two thresholds where monotonic genuinely
fails are thresholds where non-monotonic fails worse.

## The reversal at width 1, tested

Pooling all `a <= 1` against all `a > 1` at width 1 (n = 10 and 15 runs):

> monotonic **0.8658** vs non-monotonic **0.7715**, difference **+0.0943 in
> favour of monotonic**, two-sided permutation **p = 0.0002** (20,000
> resamples, seed 0).

The fold does not merely fail to help at the tightest bottleneck. **It hurts,
significantly.**

## Three checks that the failure is real and not an artifact

**1. The fold is active.** If bottleneck pre-activations were small, `f_a`
would never reach its non-monotone region and the test would be vacuous. It is
not: `f_a` is non-monotone where `|x| > acos(-1/a)`, and the fraction of
bottleneck units past that point is

| width, `a` | fold onset `|x| >` | units in fold region |
|---|---:|---:|
| 1, 2.5 | 1.98 | 11.7% |
| 2, 2.5 | 1.98 | 13.0% |
| 8, 2.5 | 1.98 | 26.8% |
| 1, 4.0 | 1.82 | 10.2% |
| 8, 1.5 | 2.30 | 34.7% |

Between a tenth and a third of units sit in the folding region. The activation
is doing what it is supposed to do; the task does not benefit.

**2. Not a budget artifact.** Quadrupling the budget to 8,000 steps
(`mnist_fold_pilot_longbudget.csv`) leaves the picture unchanged and the
width-1 reversal intact: `a = 0` reaches 0.9041 against `a = 2.5` at 0.8487.
Both classes improve; the ordering does not move.

**3. Not an ID-estimate error.** The bottleneck was probed from `w = 1`, far
below every estimate, to `w = 8`, at the MLE estimate. The phenomenon is
absent across the whole range, so no placement of the narrow bottleneck inside
the estimators' spread would have found it.

## What this means

The registration fixed this reading in advance, and we hold to it: **MNIST
does not exhibit the capability boundary the toy task exhibits**, so it cannot
be used to test whether the boundary moves with budget. The budget law remains
scoped to the four-parameter task.

The informative part is *why*, and the registration anticipated this too: a
control failure here "would mean real-data structure provides what the fold
provides in the toy task, which is itself informative." That is what the width-1
reversal suggests. In the 1D task the fold is **necessary** — no monotone map
can produce two sign changes, so `|w2|` must diverge as the fold shallows. On
MNIST at a 1-unit bottleneck the network is not being asked to fold a linked
or nested structure; it is being asked to compress a 10-class problem through
one scalar, and a monotone map does that better than one that folds distinct
inputs on top of each other. **The fold is not a free capability: where it is
not needed to separate the classes, identifying distant inputs is a cost.**

This is consistent with §2.5's width result, where the monotonicity-specific
advantage vanishes by width 6, and it sharpens it: the advantage does not
merely vanish off the obstruction, it can reverse.

## What this does not establish

- It does not show the budget law is false outside the toy task. It shows
  MNIST-MLP is not a setting where the question can be posed, because the
  precondition — a capability boundary that depends on fold depth — is absent.
- It does not test CIFAR (Part 5), which was gated on MNIST resolving
  positively and was not run.
- It does not rule out some other standard setting exhibiting the boundary.
  What a qualifying setting needs is now sharper than the T39 list: not only
  an unambiguous width axis, but a task whose classes are **not separable by a
  monotone map through the bottleneck**. MNIST's are.

## Artifacts

`mnist_fold_id.csv`, `mnist_fold_pilot.csv`, `mnist_fold_pilot_narrow.csv`,
`mnist_fold_pilot_longbudget.csv`; module `src/mnist_fold.py`; registration
`results/mnist_budget_law_prediction.md`.
