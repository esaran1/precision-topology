# Registered predictions: corrugated parametrization and author protocol

Written **before** the corrugation generator was implemented and before any run
executed, so the results are a test rather than a reading. **The predictions
themselves are never revised.** Outcomes are appended in clearly marked sections
below each prediction, leaving the original wording intact so the comparison is
auditable.

## Why this supersedes the earlier fold-layer test

Our previously registered fold-layer prediction concerned axis alignment. The
author has since corrected that framing: **an affine layer precedes the first
activation and can perform arbitrary rotations, so axis alignment is irrelevant
to the network.** Our "non-axis-aligned" configurations were rigid rotations,
which the first affine map can undo at no cost. They were no-ops from the
network's perspective, and the earlier prediction was therefore **never actually
tested** — the negative result it produced says nothing, because the property it
varied is not a property the network sees.

What the author meant is a link deformed such that a fold along any *single*
direction cannot break the link without creating other problems. His concrete
suggestion is a high-frequency periodic oscillation orthogonal to the radial
direction, which corrugates the tube so no single planar fold separates the
components cleanly.

## Our replication has been incomplete from the start

Appendix G.1 of arXiv:2606.31856v1 gives their thickening procedure verbatim:

> Sample points as `γ(t) + ε·n(t)` where `n(t)` is a unit normal to the curve,
> `ε ~ U(0, r)` with `r = 0.15`, and high-frequency oscillations `0.3 sin(100t)`
> are added to preserve topology.

**Our `linked_tori` has never had the oscillation term.** Corrugation is not a
newly proposed condition — it is part of their published parametrization, and
every result this project has produced is on a smoother link than theirs. That
includes the width-3 separation result and the earlier fold-layer measurement,
both of which were measured on a link without the oscillation.

This is stated as a limitation of the replication, not as a defence of it.

## Prediction 1: fold layer under corrugation

**Corrugation should move the fold layer later.** The mechanism the author
describes is that with a corrugated tube, no single direction breaks the link
without creating other problems, so a network cannot resolve the topology with
one coordinate fold immediately after the first affine map.

- If fold layer moves later as amplitude or frequency rises, the account is
  supported and the layer-1 immediacy we have observed is a property of the
  smooth link rather than of the network.
- If fold layer stays at 1 across corrugation amplitudes, that is the same
  negative as before — but this time against a test that varies the property the
  account concerns, so the negative carries weight the earlier one did not.

## Prediction 2: the gap under the author's protocol

Three differences between our training and theirs all push toward more
dead-ReLU collapse in our runs: our learning rate is 10× theirs (`1e-2` against
`1e-3`), we use full-batch against their batch size 128, and we have no early
stopping. 45% of our width-4 ReLU runs sit at exactly chance.

**Prediction: ReLU's at-chance fraction should fall substantially under their
protocol.** Minibatch noise is a known escape from the dead-ReLU basin, a
smaller step reduces the chance of driving units permanently negative, and early
stopping can recover a better checkpoint.

The consequential question is what happens to the monotonic-versus-GELU gap:

- If the gap **survives** their protocol, it is a property of the activation
  class rather than of our optimisation choices, and is safe to quote.
- If the gap **narrows or closes**, then part of what we have been attributing
  to monotonicity is an artifact of our own training settings. That must be
  known before the gap is quoted anywhere.

Both outcomes are reportable. The gap will be reported under both protocols
regardless of which way it falls.

## Outcome of prediction 2: not borne out

Recorded after the fact, without altering the prediction above.

**The predicted substantial drop in dead-ReLU did not occur.** The at-chance
fraction fell by roughly five percentage points, not the large drop the
reasoning implied:

| Width | Ours: at chance | Theirs: at chance | Change |
|---:|---:|---:|---:|
| 3 | 61.3% | 55.8% | −5.5 pts |
| 4 | 45.0% | 40.8% | −4.2 pts |
| 5 | 27.5% | 20.0% | −7.5 pts |
| 6 | 20.0% | 12.5% | −7.5 pts |

Minibatch noise, a learning rate 10× smaller, and early stopping with
best-checkpoint restoration together move the collapse rate only slightly.

**This makes the dead-ReLU numbers more robust than if the prediction had
held.** Had the rate collapsed under their settings, our reported figures would
have been an artifact of our optimiser and would have needed correcting
wherever quoted. Instead the collapse survives three simultaneous changes, each
of which independently works against it, which points to it being closer to a
property of narrow ReLU networks on this problem than to our training choices.
The existing figures stand as measured; what changes is that we can now say they
are not protocol-specific.

The reasoning behind the prediction was not unsound — minibatch noise *is* a
documented escape from the dead-ReLU basin — it simply does not dominate at
these widths. That is worth knowing and could not have been established without
running it.

## What will be measured

Widths 3 and 4, all four activations, depths 3/5/8/12, at least 10 seeds, under
two protocols (ours and theirs), on the corrugated parametrization anchored at
their exact values with amplitude and frequency swept around it.

Primary: minimum errors per cell, which is criterion-free. Also: fraction
reaching 1.0000 on held-out accuracy, fold-layer distribution, and the
monotonic-versus-GELU gap.

Every configuration is verified embedded, non-self-intersecting, and of linking
number ±1 before training. The zero-amplitude case must reproduce the existing
baseline exactly, as a correctness check on the generator.
