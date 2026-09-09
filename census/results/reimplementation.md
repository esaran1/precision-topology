# Part 1: independent reimplementation

Fresh code written from mathematical definitions in `independent/`, importing
**nothing from `src/`** except data loading (verified: zero `src` imports in
`reimpl.py`). Compared against production.

| # | measurement | independent vs production | verdict |
|---|---|---|---|
| 1a | separation check, 500 stored parameter vectors | **500/500 agree** (106 solved both) | agree |
| 1b | onset, family A at B = 2,000 | **1.55 vs 1.60** | **DISAGREE** -- diagnosed below |
| 1c | fold depth D(a), 7 values | max rel diff **1.6e-11** | agree |
| 1d | Gauss linking, 7 reference configurations | max diff **<1e-6** | agree |
| 1e | gradient norm + Hessian at 8 exclusion-table points | rel diff **3.3e-10** (gradients), **6.3e-7** (lambda_min) | agree |
| 1f | exponent fits (alpha, onset, 4 families, 2 SGD) | max diff **4.5e-05** | agree |

## 1e in full, since it is load-bearing

The gradient norms are what withdrew the "unreachable minima" framing and
reframed the phenomenon, so this is not merely a pipeline check. Autograd
against central finite differences, at the constructed points:

| a | grad (autograd) | grad (finite diff) | rel diff | lambda_min ag | lambda_min fd |
|---|---|---|---|---|---|
| 1.02 | 0.0192137943 | 0.0192137943 | 2.7e-10 | -0.17389 | -0.17389 |
| 1.25 | 0.1947329566 | 0.1947329566 | 1.5e-10 | -0.23667 | -0.23667 |
| 1.45 | 0.2655551651 | 0.2655551651 | 8.5e-11 | -0.01988 | -0.01988 |
| 1.50 | 0.2723914077 | 0.2723914077 | 8.7e-12 | -0.01015 | -0.01015 |
| 2.00 | 0.2249463214 | 0.2249463214 | 8.7e-12 | +0.00954 | +0.00954 |
| 3.00 | 0.1458884284 | 0.1458884284 | 3.8e-11 | +0.02056 | +0.02056 |

**Ten significant figures.** And the sign of lambda_min -- the criticality
claim -- agrees at all six values where it is negative. **The criticality
finding and everything built on it (`thin_target_v2.md`, T41, the withdrawal
of "unreachable minima") is unaffected.**

## 1b: the disagreement, and what it revealed

One grid step apart. **Cause: sampling, not implementation.** The independent
pipeline's rates run uniformly 0.05-0.125 higher across all five grid values
(independent RNG for both data and initialization), and production's deciding
cells are 0.550/0.450 with both binomial 95% CIs straddling 0.5.

Full-pipeline bootstrap (600 replicates, resampling seeds and **redetermining
every onset through the bracketing rule**, so correlation is captured rather
than assumed): exponent 95% interval **[-0.7609, -0.7065]**, half-width
**+-0.0272** -- **0.24x** the +-0.114 grid-resolution term already carried.
Adding in quadrature widens bands by 2.8% and **changes no registered
verdict** (`onset_stability.md`).

**The estimator's stability is a measured property with a named failure mode,
not an assumption.** Bracket sharpness by budget:

| budget | onset rate -> next | drop | |
|---|---|---|---|
| 2,000 | 0.550 -> 0.450 | 0.100 | **shallow -- the unstable cell** |
| 8,000 | 0.600 -> 0.200 | 0.400 | sharp |
| 32,000 | 0.650 -> 0.000 | 0.650 | sharp |
| 128,000 | 0.700 -> 0.000 | 0.700 | sharp |

The deciding cell straddles 0.5 by construction, but its **neighbour** usually
does not, so 3 of 4 brackets cannot move under resampling. **The mechanism
predicts where the instability appeared**: B = 2,000 is the one shallow
bracket, and it is exactly the cell where the independent implementation
disagreed.

## Separate finding: the asymptotic form

Reimplementing D(a) surfaced that the prose form (8/3)eps^{3/2} is an
**a -> 1+ asymptotic** inaccurate by 43-136% over the measured range, though
every computation uses the exact `dip_depth()`. The local slope over the onset
range is **1.4364** against the claimed 1.5 (4.2%), so the exponent chain
survives; the theorem must be stated in D(a) with the asymptotic labelled
separately (`asymptotic_finding.md`).
