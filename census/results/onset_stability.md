# Onset-estimator stability: bootstrapped, and the instability is localised

Prompted by a **disagreement in the independent reimplementation (1b)**: an
independently written pipeline (numpy Adam from the update rule, independent
data RNG) located family A's B = 2,000 onset at **1.55** against production's
**1.60** -- one grid step apart.

## The disagreement is sampling, not implementation

Per-cell rates, n = 40 each, two pipelines:

| a | production | independent |
|---|---|---|
| 1.70 | 0.650 | 0.750 |
| 1.60 | 0.550 | 0.675 |
| 1.55 | 0.450 | 0.500 |
| 1.50 | 0.375 | 0.425 |

The independent rates are uniformly higher by 0.05-0.125 -- a consistent
sampling offset from different random draws, not a structural difference.
Production's two deciding cells are 0.550 and 0.450, and **both 95% binomial
CIs straddle 0.5**, so which grid value is "the onset" turns on about 2 runs
in 40.

## Full-pipeline bootstrap, not a propagated per-cell term

A per-cell uncertainty would assume independence across cells; the observed
consistent offset argues against that. Instead the **entire pipeline** was
resampled: draw 40 seeds with replacement within each cell, re-apply the
bracketing rule to redetermine every onset, refit the exponent. 600 replicates,
per-seed outcomes cached in `onset_bootstrap_outcomes.csv`.

| quantity | value |
|---|---|
| point estimate | -0.7275 |
| bootstrap mean | -0.7285 |
| bootstrap sd | 0.0191 |
| **95% interval** | **[-0.7609, -0.7065]**, half-width **0.0272** |
| grid-resolution term previously propagated | +-0.1140 |
| **ratio** | **0.24x** |

**The bootstrapped uncertainty is four times SMALLER than the grid-resolution
term already being propagated.** My expectation before running it was the
opposite, and that expectation was wrong.

## Why it is small: the instability is localised to one cell

The bracket's sharpness differs enormously by budget:

| budget | onset rate -> next rate | drop | stable? |
|---|---|---|---|
| 2,000 | 0.550 -> 0.450 | **0.100** | **shallow, unstable** |
| 8,000 | 0.600 -> 0.200 | 0.400 | sharp |
| 32,000 | 0.650 -> 0.000 | 0.650 | sharp |
| 128,000 | 0.700 -> 0.000 | 0.700 | sharp |

At three of four budgets the cell below the onset is far from 0.5, so
resampling **cannot** move the bracket. Only B = 2,000 is genuinely a coin
flip -- and that is exactly the cell where the independent implementation
disagreed. The exponent is fitted across all four, so one unstable onset moves
it little.

## Correction to my own earlier framing

I reported the 1b disagreement as implying "the onset estimator is unstable at
n = 40 whenever a cell lands near the threshold", and suggested individual
onsets carry ~+-1 grid step of sampling uncertainty **on top of** grid
resolution. The bootstrap shows that is **overstated**: the deciding cell is
near 0.5 by construction, but its *neighbour* usually is not, and the bracket
is what determines the onset. **Sampling uncertainty on the fitted exponent is
+-0.027, well inside the +-0.114 already carried.**

## Which registered verdicts change

**None.** Adding the bootstrap term in quadrature to the grid term gives
sqrt(0.1140^2 + 0.0272^2) = **0.1172**, a 2.8% widening:

| verdict | margin | affected? |
|---|---|---|
| SGD onset test (missed band edge by 0.0356) | band would widen 0.1181 -> 0.1213 | **no** -- still outside, miss unchanged in kind |
| 3-cell margin comparison (0.0008 inside) | already reported as decided by noise; the m=0 / m=0.01 bit-identity was and remains the load-bearing evidence | **no change in conclusion**, but the marginality is now quantified |
| four-family exponents (all overlap) | overlaps widen slightly | **no** |
| q4 SGD flat (\|exp\| < 0.0185 vs -0.8305) | 45x gap | **no** |
| 0/200 -> 100% at a = 1.25 | not a threshold estimate | **no** |
| monotonic zero (0 in 5,580) | no threshold involved | **no** |

**The honest statement**: the deciding cell is a coin flip by construction, but
the bracket usually is not, and the fitted exponents carry +-0.027 of sampling
uncertainty against the +-0.114 of grid resolution that dominates them.
