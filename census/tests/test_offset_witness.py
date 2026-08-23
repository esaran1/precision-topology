"""The a=1.02 exhibit must be reproducible, separating, and genuinely folded."""

from __future__ import annotations

import numpy as np
import torch

from src.linking import linking_number
from src.offset_witness import (
    SHRINK,
    T_STAR,
    TARGET_A,
    OffsetWitness,
    dense_errors,
    f_a,
    train_offset_witness,
)


def test_exhibit_separates_dense_sample():
    model = train_offset_witness()
    errors, margin = dense_errors(model, 10_000, 930_001)
    assert errors == 0
    assert margin > 0.0


def test_exhibit_deterministic():
    first = train_offset_witness(steps=50)
    second = train_offset_witness(steps=50)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_fold_coordinate_straddles_local_max():
    """The data's fold coordinate must cross t*, or there is no fold."""

    left = T_STAR + SHRINK * (-2.2)
    right = T_STAR + SHRINK * (1.2)
    assert left < T_STAR < right
    # and the right extent stays inside the decreasing branch
    window = 2 * np.arccos(1.0 / TARGET_A)
    assert right < T_STAR + window


def test_frozen_fold_kills_linking():
    """lk goes -1 -> 0 at the f_{1.02} layer itself, before any training."""

    model = OffsetWitness()  # untrained continuation; frozen part is fixed
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    a_curve = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    b_curve = np.stack([1 + np.cos(t), np.zeros_like(t), np.sin(t)], 1)
    inp = linking_number(
        torch.tensor(a_curve, dtype=torch.float64),
        torch.tensor(b_curve, dtype=torch.float64),
    )
    assert inp.rounded == -1
    with torch.no_grad():
        ra = model.frozen_representation(torch.tensor(a_curve, dtype=torch.float32)).to(torch.float64)
        rb = model.frozen_representation(torch.tensor(b_curve, dtype=torch.float32)).to(torch.float64)
    folded = linking_number(ra, rb)
    assert folded.defined
    assert folded.rounded == 0
    assert folded.min_distance > 0.02


def test_f_a_is_nonmonotonic_but_barely():
    values = torch.linspace(T_STAR - 0.1, T_STAR + 0.3, 10_001, dtype=torch.float64)
    outputs = f_a(values)
    differences = outputs[1:] - outputs[:-1]
    assert (differences < 0).any()  # the dip exists
    assert float(outputs.max() - outputs[-1]) < 0.01  # and is shallow
