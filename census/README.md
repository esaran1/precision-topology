# Finite-Precision Saturation Census

This directory contains an isolated PyTorch/CPU experiment measuring finite-precision
saturation and output collisions in small multilayer perceptrons. It does not depend
on or execute the repository's legacy TensorFlow/Julia code.

Create the isolated environment with:

```bash
python3 -m venv census/.venv
census/.venv/bin/python -m pip install -r census/requirements.txt
```

Run instructions and measured findings will be completed after the experiment runs.

## Dataset geometry

The linked dataset is a documented, reasonable realization of D-II rather than
a claim of point-for-point reproduction: the paper does not fully specify its
sampling geometry. For major radius `R`, class 0 has core
`(R cos(t), R sin(t), 0)` and class 1 has core
`(R + R cos(t), 0, R sin(t))`. Uniform-by-volume disk offsets create solid
tubes around the cores. The core curves have Gauss linking number of magnitude
one. The configurable tube-radius guard keeps the two solid tori disjoint.

The control is two isotropic Gaussian blobs in R3, centered on the x-axis.

## Model initialization

Main-run MLPs retain PyTorch's unmodified `nn.Linear.reset_parameters` behavior:
Kaiming-uniform weights with `a=sqrt(5)` (gain `1/sqrt(3)`, equivalent bound
`1/sqrt(fan_in)`) and uniform biases with the same bound. Inputs are not
standardized. Initialization metadata is recorded with census results because
it directly controls initial pre-activation scale.

## Training criterion

Models train in float32 on CPU with deterministic, full-batch Adam
(`learning_rate=0.01`) for 2,000 fixed steps. A run enters the census only if
final training accuracy is exactly 100% and evaluation accuracy is at least
99% (evaluation error at most 1%). Failures are recorded and excluded.
Checkpoints are captured before training at step 0 and after 10%, 25%, 50%, and
100% of updates. Each checkpoint stores evaluation-set hidden pre-activations,
not post-activation values.

## Collision metrics

Scalar output values are never pooled across units. For each real quantizer,
the census reports the distribution of within-unit collision rates and, as the
headline injectivity measure, the fraction of duplicate complete activation
vectors across evaluation inputs. The identical inputs passed through the
untrained step-0 network provide the pigeonhole baseline. Reported excess is
`trained collision rate - initialization collision rate`. The paper's `2^-9`
half row has no real quantizer, so all collision fields are null.
