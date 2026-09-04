# Parts 2-3: termination is a budget cutoff, not an attractor; the overlap prediction fails

Registered P-2c and P-3b in `arrhenius_prediction.md`. Data:
`termination.csv`, `termination_traces.csv`, `overlap.csv`,
`overlap_prediction.csv`.

## 2c: reached-|w2| does not saturate. Findability is budget-limited.

| a | budget | rate | median terminal \|w2\| |
|---|---|---|---|
| 1.25 | 2,000 | **0.000** | 3.15 |
| 1.25 | 4,000 | 0.050 | 7.75 |
| 1.25 | 8,000 | 0.875 | 17.90 |
| 1.25 | 20,000 | **0.925** | **48.48** |
| 1.35 | 2,000 | 0.000 | 3.51 |
| 1.35 | 20,000 | 0.925 | 45.12 |
| 1.45 | 2,000 | 0.175 | 3.66 |
| 1.45 | 20,000 | 0.925 | 41.65 |

**|w2| grows without bound with budget (3.2 -> 48) and never saturates**, and
findability rises with it, 0.000 -> 0.925 at a = 1.25. P-2c is answered
decisively in the **cutoff** direction: there is no attractor at the terminal
|w2|; training is simply stopped mid-ascent.

This is a fourth manipulation of the gap, and the cheapest: **the committed
onset (1.30, 1.35] is a 2,000-step artifact.** At 20,000 steps a = 1.25 solves
at 92.5%.

## 2b: dL/dw2 is not zero at termination

Median |dL/dw2| at step 2,000 is 4.4e-4 to 7.0e-4 across a = 1.25-2.00, with
median loss still 0.12-0.44. The gradient is small but not vanishing, and the
loss is far above 0. Consistent with 2c: the run is cut off on a slope.

## 2a: trajectories split into escapers and stallers

Recorded traces (a = 1.25) show two behaviours:

- **Escapers**: |w2| rises steadily, loss falls (seed 1: |w2| 0.19 -> 3.71,
  loss 0.758 -> 0.417; seed 2: 0.27 -> 5.27, 0.701 -> 0.377).
- **Stallers**: loss pinned at **0.6931 = log 2** -- the constant-predictor
  saddle -- with |w2| drifting *down* (seed 0: 0.82 -> 0.13; seed 4: stuck at
  0.6931 for the whole run). **32% of all recorded trace points sit within
  1e-4 of log 2.**

So the population is a mixture: runs that escape the log-2 saddle and climb in
|w2|, and runs that do not escape within budget. |w2| is not monotone in
either group.

## 3b: the overlap prediction FAILS

Derived threshold from the theorem: a network with margin m needs
|w2| >= 2m/G*(a). Predicting the rate as the fraction of terminal |w2| above
that threshold, with **one global constant m\*** fitted across all 18
conditions (not per-condition):

Best fit m\* = 0.1: **mean absolute error 0.283, max 0.925**, correlation 0.65.

| condition | predicted | observed | error |
|---|---|---|---|
| a = 1.25, standard | 0.550 | **0.000** | +0.550 |
| a = 1.35, standard | 0.650 | **0.000** | +0.650 |
| init scale 0.3, a = 1.25 | 0.925 | **0.000** | +0.925 |
| init scale 3.0, a = 1.50 | 0.875 | **0.000** | +0.875 |
| a = 1.45, lr 3e-2 | 0.875 | 0.875 | 0.000 |

The errors are **all positive**: the criterion over-predicts everywhere it
fails. Terminal |w2| above the margin threshold is **necessary but not
sufficient** -- many runs terminate at adequate |w2| and still classify
incorrectly, because |w2| alone does not fix (w1, b1) into the fold's
usable window.

**P-3b is falsified.** The overlap of the two regions does not predict the
rate; the mechanism is not derived by this route.

## What Parts 2-3 establish

1. The phenomenon is **budget-limited ascent**, not an attractor mismatch:
   terminal |w2| is wherever the run was stopped, and grows indefinitely.
2. A **log-2 saddle** (the constant predictor) holds a substantial fraction of
   runs, which is a second, separate obstruction and is a genuine critical
   point unlike the correctly-classifying points.
3. The **onset is budget-dependent**, joining step size, initialization scale
   and optimizer as the fourth thing that moves it. It is not a property of
   the task.
4. The margin threshold from the theorem is **necessary but not sufficient**;
   a one-dimensional |w2| criterion cannot predict rates.
