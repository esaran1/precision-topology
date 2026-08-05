# Predictions recorded before running the detector on trained weights

Written and committed before the detector was run on any recovered run, so the
interpretation below is stated in advance rather than fitted to the counts.

## Prediction 1: where violations would appear, if any

The positive control's F-collision was produced by ReLU clamping two distinct
preactivations to exactly 0.0. The census networks are tanh, whose analogous
many-to-one mechanism is saturation: once `|x|` is large enough, `tanh(x)`
rounds to the same representable value for a range of inputs.

Saturation is measured and is strongly concentrated by layer. In accepted
linked-tori tanh runs, paper-criterion saturation is 16.5752% ± 3.2417% on final
hidden layers against 0.7851% ± 0.2036% over non-final layers. The ingredient
the control needed therefore exists in these networks but is concentrated in one
layer.

If any F-not-G violations appear, they are expected at or near the final layer
and at coarse precision (fixed-4 first, then fixed-6 and bfloat16), because that
is where saturation is dense enough to produce exact full-precision collisions
in the first place. Violations at early layers or at float32/float64 would not
match this reasoning and would call for a separate explanation.

## Prediction 2: why a zero count is the structurally expected outcome

The positive control's violating pair was **between-class**. That is available
in a hand-built network but is not available in an accepted census run.

A between-class F-collision means two inputs of different classes produce
identical full-precision outputs, which makes them indistinguishable to the
deterministic readout and forces at least one to be misclassified. Every
accepted run was gated on exactly 100% training accuracy and at least 99%
evaluation accuracy, so accepted runs cannot contain between-class F-collisions
at the output in any quantity the gate would have caught.

The real sweep therefore differs from the control **structurally, not
incidentally**. The control demonstrates that the detector fires when a
qualifying pair exists; the gate largely removes the between-class version of
that pair from the population being measured. A zero between-class count should
be read as the expected consequence of the acceptance criterion, not as evidence
that the detector failed. The positive control is what distinguishes those two
readings.

Within-class F-collisions carry no such restriction: the gate is indifferent to
two same-class inputs sharing an output vector. If violations occur at all, the
within-class variety is where they are structurally permitted.

## What the control's difficulty implies about the expected rate

The control required a specific conjunction, found only after three failed
constructions:

1. An F-collision arising from a **clamping** mechanism that quantization can
   undo. A collision produced any other way tends to survive quantization.
2. A quantization cell boundary falling **between** the two inputs at the
   layer where they are still distinct.
3. Enough **gain** downstream for the resulting divergence to exceed one
   quantization cell, so it is not rounded back together at the next
   quantization step.

Each failed attempt broke one of these: the first discarded the diverging unit
so the separation was erased, the second propagated that unit but destroyed the
F-collision, and the third produced a divergence smaller than one cell that was
rounded away again. That all three conditions must hold simultaneously is
itself a reason to expect the phenomenon to be rare in trained networks, where
nothing arranges for them to coincide.
