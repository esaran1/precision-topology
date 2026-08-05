# Between-class margin in units of quantization resolution

Scope: all 75 accepted linked-tori tanh runs (depths 4, 6, 8, 10; widths 5, 15,
30, 50; seeds 0-4), every hidden layer, every quantizer with a real
implementation. 3,682 layer-quantizer rows. Seeds are recorded per row in
`between_class_margin.csv`.

No new experimental condition was introduced. Each run was reconstructed under
its recorded seeds and verified against its recorded values before use; the
maximum deviation of reconstructed accuracy and saturation from the recorded
values was **0.0 exactly** across all 75 runs and all 13,500 compared saturation
values. Any nonzero deviation would have aborted the analysis.

## Definition of the quantization step

"Quantization step" is not a single number for IEEE formats: ULP spacing depends
on the exponent, so it varies by coordinate and across the range of each
coordinate. Two conventions are therefore stated explicitly.

- **Step.** Evaluated *locally* at the two points being compared, taking the
  widest local ULP across the coordinates of the closest between-class pair.
  For fixed-point quantizers this reduces to the constant grid spacing
  (fixed-N has spacing `2/2^N`; fixed-4 = 0.125).
- **Distance.** Reported both as Euclidean and as Chebyshev (max-coordinate)
  separation. Chebyshev is the decision-relevant one: coordinate-wise
  quantization keeps a pair distinct if *any* single coordinate falls in a
  different cell, which is governed by the largest coordinate gap. The ratio
  below uses Chebyshev distance over local step.

Distances are measured on post-activation values, which is what the quantizer
consumes, matching the existing collision metric.

## Minimum margin in steps, by width and precision

Minimum over all runs and all layers (lower is closer to a collision):

| Width | fixed-4 | fixed-6 | fixed-8 | bfloat16 | float16 | float32 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.2891 | 1.1565 | 4.6262 | 10.4025 | 83.2202 | 6.8174e+05 |
| 15 | 0.4949 | 1.9798 | 7.9191 | 15.8383 | 126.7063 | 1.0380e+06 |
| 30 | 1.0127 | 4.0508 | 16.2034 | 32.4067 | 259.2537 | 2.1238e+06 |
| 50 | 1.2910 | 5.1640 | 20.6560 | 28.4468 | 330.4959 | 2.7074e+06 |

Margin in steps grows with width and with precision. Only fixed-4 ever falls
below one, and only at widths 5 and 15.

## Final-layer margin, mean over seeds with seed-level SD

Final hidden layer only (the representation nearest the classifier), minimum
margin in steps within each run:

| Width | fixed-4 | fixed-6 | bfloat16 | float32 |
|---:|---:|---:|---:|---:|
| 5 | 12.2968 ± 2.2788 | 49.1872 ± 9.1150 | 393.4977 ± 72.9201 | 2.5788e+07 ± 4.7789e+06 |
| 15 | 12.5523 ± 1.9029 | 50.2093 ± 7.6116 | 401.6740 ± 60.8926 | 2.6324e+07 ± 3.9907e+06 |
| 30 | 12.6345 ± 2.5113 | 50.5379 ± 10.0451 | 404.3029 ± 80.3605 | 2.6496e+07 ± 5.2665e+06 |
| 50 | 13.5179 ± 2.9879 | 54.0717 ± 11.9514 | 368.6527 ± 106.9619 | 2.8349e+07 ± 6.2660e+06 |

The final-layer minimum never falls below **4.9174 steps** in any run at any
precision, including fixed-4. At bfloat16 the smallest final-layer margin over
all 75 runs is 157.3581 steps.

## Width-invariance of final-layer margin

Final-layer margin in quantization steps is nearly constant in width. At
fixed-4 the means are 12.2968, 12.5523, 12.6345, and 13.5179 at widths 5, 15,
30, and 50. Width varies by a factor of ten while the margin changes by a factor
of 1.0993.

This is not an artifact of the step convention. Fixed-point quantizers have a
constant step, so the fixed-4, fixed-6, and fixed-8 ratios are the same number
rescaled. The underlying quantity is the raw post-activation Chebyshev distance,
which is measured before any quantizer is applied:

| Width | Final-layer raw Chebyshev distance |
|---:|---:|
| 5 | 1.5371 ± 0.2848 |
| 15 | 1.5690 ± 0.2379 |
| 30 | 1.5793 ± 0.3139 |
| 50 | 1.6897 ± 0.3735 |

bfloat16, whose step is data-dependent rather than constant, gives an
independent ratio of 1.097 across the same widths.

This is an open observation, not an explained effect. It suggests that training
targets a characteristic separation relative to representation scale rather than
exploiting the additional width available to it, but this analysis does not
establish that, and no mechanism is offered here. Four widths, all at or above
the width at which the classification obstruction is theoretically resolved, are
too few and too narrowly placed to support a claim. It should be rechecked at
widths 3 to 8, where the relevant transition is expected to sit.

## Where the ratio falls below one

55 of 3,682 rows have Chebyshev margin below one step. All 55 are **fixed-4**;
39 at width 5 and 16 at width 15; **none is a final layer**. No bfloat16,
float16, float32, float64, fixed-6, or fixed-8 row falls below one anywhere.

| Layer position | margin >= 1 | margin < 1 |
|---|---:|---:|
| Hidden (distance from output > 0) | 3,102 | 55 |
| Final (distance from output = 0) | 525 | 0 |

## Does the ratio predict the impure rows?

Among the 1,705 rows that contain at least one collision group:

| | pure | impure |
|---|---:|---:|
| margin >= 1 | 1,650 | 0 |
| margin < 1 | 40 | 15 |

Margin below one step is a **necessary** condition for impurity with zero
exceptions: all 15 impure rows have margin below one, and none of the 1,650 rows
at or above one step is impure. It is **not sufficient**: 40 rows fall below one
step yet remain fully pure. Two points separated by less than one step usually
still land in different cells, because sharing a cell requires the pair to fall
within the same cell boundaries, not merely to be closer together than the cell
is wide. The margin criterion therefore over-predicts impurity, flagging 55 rows
as at risk where 15 actually contain a between-class collision.

The sufficient condition is actual cell coincidence. Counting between-class
pairs that quantize to identical vectors predicts impurity **exactly**:

| | pure | impure |
|---|---:|---:|
| zero between-class collision pairs | 1,690 | 0 |
| at least one such pair | 0 | 15 |

All 40 margin<1-but-pure rows have exactly zero between-class collision pairs.
This is a consistency check on the metric rather than an independent finding:
an impure group is by definition one containing inputs of both classes.

## The 15 impure rows

All are fixed-4, linked-tori, tanh, width 5, hidden layers only (layers 1-3,
never final), across depths 6, 8, 10 and seeds 0, 1, 3. Total between-class
pairs: 1,875. Lowest purity observed: 97.1631%.

| Depth | Width | Seed | Layer | Chebyshev dist | Step | Margin | BC pairs | Purity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 5 | 1 | 2 | 0.0729 | 0.125 | 0.5829 | 2 | 99.6124% |
| 6 | 5 | 3 | 2 | 0.0412 | 0.125 | 0.3295 | 186 | 99.2278% |
| 6 | 5 | 3 | 3 | 0.0562 | 0.125 | 0.4498 | 48 | 99.2958% |
| 8 | 5 | 0 | 2 | 0.0485 | 0.125 | 0.3880 | 45 | 99.0385% |
| 8 | 5 | 0 | 3 | 0.0538 | 0.125 | 0.4307 | 1 | 99.5475% |
| 8 | 5 | 1 | 2 | 0.0450 | 0.125 | 0.3602 | 10 | 99.6109% |
| 8 | 5 | 3 | 2 | 0.0482 | 0.125 | 0.3856 | 180 | 98.8235% |
| 8 | 5 | 3 | 3 | 0.0481 | 0.125 | 0.3844 | 364 | 98.9011% |
| 10 | 5 | 0 | 1 | 0.0656 | 0.125 | 0.5251 | 2 | 99.7753% |
| 10 | 5 | 0 | 2 | 0.0480 | 0.125 | 0.3843 | 28 | 99.2806% |
| 10 | 5 | 0 | 3 | 0.0483 | 0.125 | 0.3867 | 29 | 97.1631% |
| 10 | 5 | 1 | 2 | 0.0838 | 0.125 | 0.6702 | 6 | 99.7126% |
| 10 | 5 | 1 | 3 | 0.0860 | 0.125 | 0.6879 | 8 | 99.5122% |
| 10 | 5 | 3 | 2 | 0.0406 | 0.125 | 0.3251 | 888 | 98.6239% |
| 10 | 5 | 3 | 3 | 0.0361 | 0.125 | 0.2891 | 78 | 99.3421% |

Width-5 fixed-4 hidden-layer minimum margin, per seed: 0.3769, 0.3602, 0.2891,
0.6991 (seeds 0, 1, 3, 4), mean 0.4313 ± 0.1825. Seed 2 failed the training gate
at every depth and contributes no accepted run.

## Interpretation

Purity is near-forced by margin rather than circular. The acceptance gate
requires the classifier to separate the classes in float32, which entails a
strictly positive between-class separation at every layer. Purity under
quantization then follows wherever that separation exceeds the local
quantization step, which is the case in 3,627 of 3,682 measured rows.

The exceptions locate precisely where it does not. They are confined to the
coarsest quantizer (fixed-4, step 0.125) at the narrowest width, in hidden
layers where representations are least spread out. Margin < 1 is necessary for
impurity and admits no counterexample; cell coincidence is what converts the
possibility into an observed impure group.

A separate point governs how far the gate's entailment reaches. Post-hoc
quantization at a hidden layer is **counterfactual**: the trained classifier
consumed the unquantized float32 activation, so no hidden-layer quantized
representation was ever required by the gate to be class-separable. Hidden-layer
purity was therefore never constrained by the acceptance criterion, which is why
the only impure rows appear there. Final-layer purity is a different matter: the
final hidden representation is what the linear readout consumes, and its margin
exceeds the resolution of every tested format by at least 4.9174 steps, so
final-layer purity reflects margin exceeding resolution rather than selection
alone.
