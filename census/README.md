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
