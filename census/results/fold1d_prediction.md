# Registered predictions: the minimal fold task (Task B, Part 1)

Written before any 1D experiment runs.

## The task, fixed in advance

Classify `y = sign(|x| − 1)` with a width-1 network
`N(x) = w₂·f(w₁x + b₁) + b₂`, single logit, sign readout. Data: inner
class uniform on |x| ≤ 0.8, outer class uniform on 1.2 ≤ |x| ≤ 2.0
(margin 0.4, two-sided), 200 points per class train and eval,
BCE-with-logits, Adam lr 1e-2, 2,000 steps, 200 seeds per activation
value. Dense verification: 10,000 fresh points, errors must be 0.

**Impossibility is one line**: monotonic `f` makes `N` monotonic in `x`,
so `N` has at most one sign change; the task needs two (at ±1). No
monotonic width-1 network solves it for any weights.

## Predictions

**P1-zero (provable).** Exact zero solve rate for every monotonic value:
Family A `a ≤ 1`, Family B `α ≥ 0`, tanh, ReLU, leaky-ReLU. Any nonzero
is a bug in the harness, to be found and fixed, not a finding.

**P2-onset.** The findability onset sits strictly past the analytic
threshold, mirroring the link setting: zero or near-zero rates at
`a = 1.01–1.05`, rising later.

**P3-calibration (family A only, as the failed 2a-match requires).** The
weight-scale account predicts the onset location by the same procedure
used in the link setting: the onset is where the required amplification
`A_req(a) ∝ 1/fold-depth(a)` crosses the scale that training actually
reaches (here the distribution of trained `|w₂|·|slope|`-type products,
measured from the sweep's own runs). Because a width-1 network has a
single amplifying weight where the depth-5 link networks had a
five-layer product, the reachable scale here is far smaller, and the
account therefore predicts the 1D onset lands **later** (deeper into the
non-monotonic side) than the link setting's 1.085. Coarse registered
point prediction from that reasoning: onset in **[1.10, 1.30]**. The
committed test is the procedure (crossing predicts onset), not the point
value; but the point value is recorded so that hindsight cannot move it.

**P4-GELU.** GELU solves the 1D task at a substantial rate (its fold
depth 0.17 needs only ~20× amplification, within reach of even a single
weight), i.e. no deep rate suppression like the link setting's 8% —
registered expectation ≥ 30%.

## What would count against what

- Nonzero monotonic rate → bug, full stop.
- Onset exactly at the threshold (positive rate at 1.01–1.02) → **the
  gap requires the harder setting** — a major result in the other
  direction, to be reported prominently per the task brief.
- Onset far outside [1.10, 1.30] with the crossing procedure also
  failing → the weight-scale account is not setting-independent.
- Crossing procedure predicts the onset but the point guess was wrong →
  procedure vindicated, guess corrected, stated as such.
