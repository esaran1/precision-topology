# Task E Part 2a: registered prediction — MNIST bottleneck sweep

Registered 2026-08-23, after intrinsic-dimension estimation
(`intrinsic_dimension.csv`, committed with this file) and BEFORE any
bottleneck training run.

## The claim and who carries it

**The GELU-over-tanh advantage is present at bottleneck widths near the
intrinsic dimension of the data arriving at the bottleneck, and gone by
roughly twice it.** Both activations are smooth; only GELU is
non-monotonic — this pair isolates the fold account from the smoothness
account. ReLU and leaky-ReLU are run and reported alongside but carry
no prediction: Part 1c isolated a ReLU-family optimization deficit
(persisting to width 8 at d = 3, shared by monotonic tanh comparisons,
not explained by dead units), so any GELU-beats-ReLU result at narrow
bottlenecks is contaminated by that deficit and does not test the fold
account.

## The quantitative boundary, derived not guessed

In our own setting (Part 1b) the monotonicity-specific advantage —
non-monotonic vs tanh — is significant at widths ≤ 5 and statistically
zero from width 6 = 2d on. The registered mapping to MNIST: advantage
present at bottleneck width ≲ ID, gone at width ≥ 2·ID.

## Intrinsic dimension: measured spread, decided in advance

The registered width axis is the ID of the **bottleneck input** — the
128-unit first-hidden-layer representation of reference networks
trained without a bottleneck (784→128→128→10; tanh and GELU, 3 seeds,
2 disjoint 5,000-point subsamples each). Estimates:

| estimator | bottleneck input | raw pixels |
|---|---|---|
| TwoNN | 11.9 – 12.7 | 18.1 – 18.6 |
| MLE k=10 | 7.0 – 7.4 | 10.8 – 10.9 |
| MLE k=20 | 7.1 – 7.4 | 11.1 |
| PCA-95% (linear reference, excluded from spread) | 23 – 35 | ~150 |

Nonlinear spread at the bottleneck: **[7, 13]**; central estimate
(median of the 12 nonlinear values) **≈ 9.7 ≈ 10**. Estimator
disagreement (TwoNN vs MLE, a factor ~1.7) is the known method
dependence; the raw-pixel MLE ≈ 11 matching the literature's 10–15 is
the sanity anchor. The representation ID is stable across activation,
seed, and subsample to ±0.4.

**Decision rule, fixed now: the conclusion must survive the whole
nonlinear spread, not just the central estimate.** The spread permits
this — the two test regions below are disjoint for every estimator:

- **Advantage region (test A): widths 4, 6, 8** — at or below ID for
  every estimator (max lower edge 7).
- **Gone region (test B): widths 32, 48** — at or above 2·ID for every
  estimator (2 × 12.7 ≈ 25.4 < 32).
- **Transition zone: widths 10–24** — estimator-sensitive by
  construction; reported descriptively, never adjudicated. If the
  advantage's observed vanishing point lands here, the *location* claim
  is only as sharp as the spread and will be stated with that caveat;
  presence-in-A and absence-in-B are the registered tests and remain
  decidable regardless.

Underpowered escape, registered: if test A fails but the transition
zone shows an advantage (i.e. the advantage exists but sits above every
ID estimate), the boundary prediction failed. If results in A and B are
individually ambiguous (criteria below), the test is reported as
underpowered, not read in whichever direction fits.

## Success criteria (fixed now)

Primary outcome: **test-set error on the 10,000-image MNIST test set**
(graded, not thresholded). Secondary: batches-to-99%-train-accuracy
(convergence, reported separately per Part 1c discipline), dead-unit
rates at the bottleneck (ReLU control).

- **Test A passes** if GELU's mean test error is below tanh's at each
  of widths 4, 6, 8, with one-sided permutation p < 0.05 (100,000
  resamples, seed 0, on the difference of means) at ≥ 2 of the 3
  widths.
- **Test B passes** if at widths 32 and 48 the GELU−tanh difference is
  n.s. (two-sided p > 0.05) **and** its point estimate is smaller than
  one third of the mean advantage measured in region A — the second
  clause prevents claiming "gone" from an underpowered null.
- **The account's prediction holds** only if both pass. A persists +
  B fails (advantage survives at 2·ID) falsifies the width dependence
  on real data; A fails kills the bridge at the first step.

## Grid

Bottleneck widths 2, 4, 6, 8, 10, 12, 16, 24, 32, 48. Architecture
784 → 128 → w → 128 → 10, activation applied at every hidden layer,
Adam 1e-3, batch 256, 3 epochs over the 60,000-image training set,
float32, CPU. Seeds 0–9 for tanh and GELU (the prediction pair), 0–4
for ReLU and leaky-ReLU (context only). 300 runs. Width 2 is below
every ID estimate and included to see the saturation/underfitting side;
it carries no registered claim.
