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
