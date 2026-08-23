# Registered predictions: localizing the failure (Part 3)

Written before implementation. Three probes: layerwise distillation of a
separating GELU network into monotonic students, linking traces through the
best monotonic runs, and the spatial distribution of near-miss errors.

## Probe 1: layerwise distillation

For a separating width-3 GELU network, train monotonic width-3 networks to
regress each layer-k intermediate output (standard MSE on the training
inputs), and measure both the regression error and — for the full pipeline
with the GELU head reattached — whether the distilled stack classifies.

**Prediction:** the failure localizes at **layer 1**, the fold layer in all
34 traced runs and in the constructed witness. Monotonic students should
match the GELU network's layer-1 *pointwise values* only by sacrificing the
fold (a monotonic map cannot be 2-to-1 along a line), so the distillation
error at layer 1 should be concentrated exactly where the fold merges
points — pairs mapped together by GELU that a monotonic student must keep
apart.

**Named alternative:** if monotonic students match layer 1 well and the
failure appears only at depth, the fold story is wrong and the obstruction
is distributed, which would count against the single-fold account the
witness supports.

## Probe 2: linking traces through the best monotonic runs

The 34 existing traces are all GELU (only GELU separates). The direct
comparison — what the closest monotonic runs do to the linking number — is
missing. Trace the best monotonic width-3 runs (the 2/6/6/8-error
`sin(≤1)` runs and the best tanh runs) with the same width-3 no-projection
estimator.

**Prediction:** linking stays at −1 through **every** layer in every
monotonic trace — they do not get partway (there is no partway: the
invariant is integer-valued and a homeomorphism preserves it). The
interesting measurable is the **minimum inter-class distance profile**: the
near-miss networks should crush the two components very close together
(small min distance at some layer) while never crossing, whereas GELU
traces show the fold (lk −1 → 0) at layer 1 followed by growing distance.
If a monotonic trace shows lk ≠ −1 at any layer with a clean residual and
healthy min distance, something is wrong with either the trace or the
claim, and it must be chased before anything else is reported.

## Probe 3: where the errors are

For near-miss monotonic runs, locate the misclassified eval points
relative to the link geometry.

**Prediction:** errors cluster where the two tubes approach each other —
the region a fold would have to resolve and a monotonic map cannot. The
natural coordinates: distance to the *other* component's core, and angle
around the own core. Errors should concentrate at the closest-approach
angles, and be overwhelmingly on one class (whichever component the
network's decision boundary fails to wrap). A uniform error scatter would
count against the incomplete-fold reading and suggest ordinary
capacity-limited noise instead.
