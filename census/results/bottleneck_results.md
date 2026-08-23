# Task E Part 2a: MNIST bottleneck sweep — registered test failed, reversed

Registration: `bottleneck_prediction.md` (committed before the sweep).
Data: `bottleneck_sweep.csv` (300 runs). Analysis inline below
(permutation tests: 100,000 resamples, seed 0).

## The result, unsoftened

**Test A failed, with the sign reversed. tanh beats GELU at every
bottleneck width, significantly, from width 2 to width 48.** Mean test
errors per 10,000 (10 seeds each), GELU − tanh difference and two-sided
permutation p:

| width | gelu | tanh | diff | p (two-sided) |
|---:|---:|---:|---:|---:|
| 2 | 1088.8 | 868.8 | +220.0 | 0.009 |
| 4 | 610.2 | 520.2 | +90.0 | 0.0002 |
| 6 | 483.2 | 422.0 | +61.2 | 0.0002 |
| 8 | 469.0 | 421.8 | +47.2 | 0.002 |
| 10–24 | 436–459 | 376–392 | +61 to +78 | < 0.0001 |
| 32 | 441.0 | 371.6 | +69.4 | < 0.0001 |
| 48 | 405.2 | 365.8 | +39.4 | 0.0006 |

The registered one-sided test (GELU better at widths 4, 6, 8) returns
p ≈ 1.0 at all three widths. Test B is moot — there is no advantage
whose disappearance could be tested. **Under the registered protocol
the bridge fails at its first step**, and per the brief, the paper's
claim stays internal to the controlled settings.

## Two structural facts that shape the interpretation

**1. The difference has no width structure at all.** The tanh advantage
is a roughly constant offset (+40 to +90 errors) from width 4 through
width 48, indifferent to the intrinsic-dimension band [7, 13] and to
2·ID. Whatever produces it is not the fold mechanism *and* not any
mechanism concentrated at narrow bottlenecks. A fold-account failure
with width structure (advantage in the wrong place) would have been
more damaging than this: the pair shows no bottleneck-linked behaviour
of any kind.

**2. The setting fails its own positive control.** The phenomenon the
bridge was meant to explain — GELU/Swish outperforming ReLU — is absent
here: GELU never beats ReLU at any width either (ReLU's means sit at or
below GELU's at 9 of 10 widths). A 784→128→w→128→10 MLP trained 3
epochs on MNIST does not exhibit the folk advantage at all. So this
experiment cannot distinguish "the fold account's width prediction is
wrong on real data" from "this setting lacks the phenomenon the
prediction is about." Both readings are consistent with the data; the
registered prediction named this setting and failed in it, and that
stands as recorded.

## Diagnosis of the reversal (descriptive, not registered)

The tanh advantage is already present in **training** error at every
width (e.g. width 32: 2.79% vs 3.70%), and every one of the 300 runs
was censored on the 99%-train-accuracy criterion — 3 epochs reaches
~96–97% train accuracy. This is the short-budget optimization regime,
and in it tanh simply optimizes this architecture faster (consistent
with Part 1c, where tanh's *speed* ordering was mid-pack but its
final-separation ordering above the ReLU family). Whether the reversal
survives training to convergence is untested and outside the registered
protocol; testing it would be a new registration, not a reanalysis.

## What this does and does not change

- The Part 1 in-setting result stands: width-dependence of the
  monotonicity-specific advantage, categorical at width d, gone by 2d.
- The registered real-data extension of that result is **falsified as
  registered**: no GELU-over-tanh advantage exists in this setting at
  any width, so no width-dependence of it could be observed.
- 2b's controls and 2c's scale intervention presuppose a narrow-width
  advantage to dissect; with none present (and the positive control
  absent), they have nothing to operate on in this setting. Not run;
  stopping at the ordered stop-point for direction.
- Honest scope sentence: "In the one standard-architecture setting we
  tested — a short-budget MNIST bottleneck MLP — the predicted
  activation effect did not appear, and neither did the folk
  GELU-over-ReLU advantage the account was meant to explain; the width
  prediction is falsified in that setting and untested in settings that
  actually exhibit the phenomenon."
