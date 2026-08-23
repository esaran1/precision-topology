# Registered predictions: the harder search for a monotonic solution (Part 2b)

Written **before** any of the four searches is implemented or run. The point
of this exercise is that it can overturn the central claim; each prediction
below names the outcome that would do so. Any monotonic width-3 network
reaching exactly 0 eval errors, found by any method, overturns the
monotonic zero and is reported immediately, prominently, and unsoftened.

The claim under attack: 0 monotonic separations in 4,610 width-3 runs — all
of them SGD (full-batch Adam) from Kaiming-uniform initialization. That is
evidence about what SGD finds, not about what the architecture can
represent. These four searches attack the gap.

## Search 1: mass random restarts

**What:** 200 seeds (10× the standard 20) at the two best-performing
monotonic width-3 configurations: `sin_family(a=0.95), depth 8` (best
observed: 2 errors) and `tanh, depth 8` (best fixed-activation: 26 errors).
Same protocol as every sweep.

**Prediction:** zero separations in 400 runs. Best error: single digits for
`sin(0.95)` (its 20-seed best was 2, so a 200-seed best of 1–2 is likely);
tanh stays ≥ 20. Per the standing rules the best is an extreme-value
statistic and will be reported with its full distribution.

**Overturn condition:** any run at exactly 0.

## Search 2: initialize from a separating GELU solution, fine-tune monotonic

**What:** reconstruct the six separating width-3 GELU networks from the
threshold sweep (depths 3,3,5,5,8,12 — deterministic retrain), swap the
activation to each of tanh, `sin(1.0)`, `sin(0.95)`, leaky-ReLU with weights
frozen at the GELU solution, measure errors immediately after the swap, then
fine-tune (Adam, 2,000 steps, lr 1e-3 and 1e-2 both) and measure again.

**Prediction:** the swap breaks separation immediately (errors > 0 at step
0 post-swap) for every source × target, and fine-tuning does not recover 0.
Reasoning: the fold that separates the link exploits GELU's negative-slope
region; a monotonic activation has no such region, so the swapped network
computes a different, non-separating map, and fine-tuning faces the same
barrier as training from scratch.

**Overturn condition:** any swapped-and-tuned network at exactly 0. If the
swap alone preserves 0, the solution exists in monotonic parameter space and
SGD merely failed to find it — the strongest possible overturn.

## Search 3: activation annealing through the threshold

**What:** train `sin_family(a=3)` width-3 networks to separation (23/80
seeds do), then anneal `a` down in steps of 0.05 from 3.0 to 0.8, training
200 further steps at each value, recording eval errors at every step. At
least 10 independent traces from different separating seeds. The network is
free to adapt its weights the whole way down; the question is the largest
`a` at which it can no longer hold 0 errors.

**Prediction:** every trace loses separation at some `a* > 1.0` and never
regains it below; errors at `a ≤ 1.0` are positive in all traces. The
sharpest version: `a*` clusters just above 1 (the network adapts until the
dip is too shallow to fold with), which would localize the practical barrier
at the analytic threshold far more precisely than the Part 1 grid.

**Overturn condition:** any trace holding 0 errors at any `a ≤ 1.0`.
**Against-the-reading condition:** traces losing separation far above 1
(say `a* > 2`) would suggest the annealing path, not the threshold, is the
obstacle, and the test loses its localizing power.

## Search 4: derivative-free direct optimization

**What:** CMA-ES (implemented and unit-tested in-repo, no SGD anywhere) on
the flattened weights of width-3 networks at depth 3 (44 parameters) and
depth 5, objective = training-set error count, with eval-set verification of
any candidate reaching 0. Activations: tanh, `sin(0.95)`, `sin(1.0)`
(monotonic targets) and `sin(1.5)` (**positive control** — the search
machinery must find a separating network where SGD can, or its monotonic
failures mean nothing). Budget: ≥ 20 restarts × enough generations that the
positive control succeeds reliably; identical budget for monotonic targets.
Nelder-Mead from multiple simplex initializations as a second, independent
method if time permits.

**Prediction:** the positive control reaches 0 errors in most restarts; no
monotonic target reaches 0 in any restart. Best monotonic CMA-ES result
lands near the SGD floor (single digits), which would itself be informative:
two unrelated search methods hitting the same nonzero floor is what a
representational barrier looks like, and what a shared optimization
pathology would have to mimic.

**Overturn condition:** any monotonic candidate at 0 train errors that also
reaches 0 eval errors. (A 0-train / >0-eval candidate is overfitting on
2,000 points, not separation, and is reported as such — this is the Part 2c
distinction.)

## If all four fail

The claim upgrades from "SGD from default initialization never separates"
to: no separating monotonic width-3 network was found by SGD from default
initialization (4,610 runs), by mass restarts at the best configuration
(400), by descent from a known-good non-monotonic solution (48
combinations), by annealing across the threshold (≥10 traces), or by
derivative-free direct search (CMA-ES, matched budget to a passing positive
control). Stated with all N's, that is the strongest form of the claim this
project can produce without a nonexistence proof.
