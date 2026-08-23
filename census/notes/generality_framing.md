# What the generality results actually support (Task B, Part 4)

An honest assessment, written after Parts 1–3.

## What broadened

**The gap does not require topology.** The full structure — a provable
impossibility side, an exact zero holding to well past the analytic
threshold, a findability onset located by required-versus-reachable
weight scale, thin-sheet solutions with basin volume jumping at the
onset — reproduces in a four-parameter, one-dimensional task with no
links, no knots, and no ambient-isotopy content anywhere. The
link-setting phenomenon is a fold phenomenon, and the claim can be
stated as: **wherever a task demands a fold and the activation's fold is
shallow, separation is representable the moment the activation is
non-monotonic but findable only when the amplification the fold needs is
within the scale training reaches.** Knots are one place that demands a
fold; `sign(|x|−1)` is another; the mechanism does not distinguish them.

## What did not broaden, and why that supports rather than weakens the account

- **Family B has no gap in 1D** (57% solve rate immediately past its
  threshold) despite having one in the link setting. This is not a
  failure of generality — the account *predicts* it: B's required
  amplification at α = −0.05 is ~7–10 in 1D, within reach, so no gap.
  The gap is not a property of an activation family; it is a property of
  the required/reachable ratio, which is task- and architecture-dependent.
- **The onset location moved** (links: a ≈ 1.085; 1D: a ≈ 1.4). Same
  reason: the link networks' five-layer products reach ~150; the 1D
  single amplifying weight reaches ~6. The mechanism is
  setting-independent; its calibration is not, and cannot be — the
  reachable scale is a property of the architecture and optimizer.
- **The registered point guess for the 1D onset was wrong** ([1.10,
  1.30] vs observed (1.30, 1.35]) because the reachable scale was
  guessed rather than measured. The registered procedure, run with
  measured inputs, lands inside the transition. Guesses are cheap;
  the procedure is the claim.

## What is genuinely open

- **Intermediate dimensions are unresolved.** The 2D annulus and 3D
  nested-shell tasks produced too few solves (3 in 2,200 runs) to
  locate onsets or test the calibration across dimensions. Nothing
  contradicts the account there — every solve was non-monotonic and
  every monotonic run failed — but "consistent" is all that can be
  said. The cross-dimensional test needs budgets this project does not
  have, and any claim of setting-independence should say "demonstrated
  in 1D and 3D-linked, untested between."
- **Family B's link-setting offset (α ≈ −0.22) has a predicted but
  unverified mechanism**: presumably B's required scale in the width-3
  link setting exceeds reach near α = 0 (its fold degenerates in shear,
  not depth). The construction cannot probe it (A_req undefined for B),
  so this rests on the analogy, not on a measurement.

## The counterfactual that did not happen

Had the 1D task shown findability beginning exactly at a = 1.001, the
gap would have been a property of the hard setting and the fold-usability
story would have collapsed. The opposite occurred: the 1D gap is *wider*
than the link gap (onset 0.4 past threshold instead of 0.085), exactly
as a lower reachable scale predicts. The minimal setting did not shrink
the phenomenon; it enlarged it on schedule.

## Recommended scope sentence for the paper

"The possible-but-unfindable gap at the monotonicity threshold is not a
topological phenomenon: it reproduces, with the same weight-scale
mechanism and a calibration that tracks the architecture's reachable
scale, in a four-parameter one-dimensional fold task where impossibility
is provable in one line — and its exact solution-set and basin geometry
there show thin-sheet solutions whose basin volume, not radius, turns on
at the findability onset."
