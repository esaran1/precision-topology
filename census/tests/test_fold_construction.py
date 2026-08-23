"""The pipeline's precondition gate must match the analytic monotonicity facts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fold_construction import PipelineFailure, fold_point


def _gelu(x):
    return torch.nn.functional.gelu(torch.tensor(x, dtype=torch.float64)).numpy()


@pytest.mark.parametrize("fn", [
    np.tanh,
    lambda x: np.maximum(x, 0.0),
    lambda x: np.where(x >= 0, x, 0.01 * x),
    lambda x: x + 0.98 * np.sin(x),
    lambda x: np.where(x >= 0, x, 0.05 * x),
])
def test_monotonic_activations_fail_at_step_one(fn):
    with pytest.raises(PipelineFailure):
        fold_point(fn)


def test_threshold_boundary_is_exact():
    with pytest.raises(PipelineFailure):
        fold_point(lambda x: x + 0.999 * np.sin(x))
    point = fold_point(lambda x: x + 1.001 * np.sin(x))
    assert point.kind == "max"


def test_gelu_fold_point_is_the_true_minimum():
    point = fold_point(_gelu)
    assert point.kind == "min"
    assert abs(point.location - (-0.7518)) < 1e-2


def test_underflow_noise_is_not_a_fold():
    """A function whose derivative flickers at 1e-13 must not pass."""

    def noisy_monotone(x):
        return x + 1e-13 * np.sin(50.0 * x)

    with pytest.raises(PipelineFailure):
        fold_point(noisy_monotone)


def test_pwl_fold_at_kink():
    point = fold_point(lambda x: np.where(x >= 0, x, -0.22 * x))
    assert point.kind == "min"
    assert abs(point.location) < 1e-3
