"""Winding-link geometry must validate before any training uses it."""

from __future__ import annotations

import numpy as np
import torch

from src.linking import linking_number
from src.winding import GRID, WindingLink, core_a, core_b, sample, validate


def test_linking_number_equals_winding():
    for link in GRID:
        estimate = linking_number(
            torch.tensor(core_a(2048), dtype=torch.float64),
            torch.tensor(core_b(link, 2048), dtype=torch.float64),
        )
        assert estimate.defined
        assert abs(estimate.rounded) == link.q


def test_all_grid_configurations_valid():
    for link in GRID:
        assert validate(link).valid, link.name


def test_q1_is_a_hopf_style_link():
    """q = 1 must carry |lk| = 1, the same class as the baseline Hopf link."""

    report = validate(WindingLink(q=1))
    assert abs(report.linking_number) == 1


def test_sample_shapes_and_determinism():
    link = GRID[1]
    first = sample(link, 200, seed=5)
    second = sample(link, 200, seed=5)
    assert first.features.shape == (400, 3)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


def test_samples_stay_near_their_tubes():
    """Class-A samples lie within tube_radius_a of A's core (and B's of B's)."""

    link = GRID[3]  # q = 4, the tightest geometry
    data = sample(link, 500, seed=1)
    a_points = torch.tensor(data.features[data.labels == 0], dtype=torch.float64)
    b_points = torch.tensor(data.features[data.labels == 1], dtype=torch.float64)
    a_core = torch.tensor(core_a(8192), dtype=torch.float64)
    b_core = torch.tensor(core_b(link, 8192), dtype=torch.float64)
    assert float(torch.cdist(a_points, a_core).min(dim=1).values.max()) <= link.tube_radius_a + 1e-6
    assert float(torch.cdist(b_points, b_core).min(dim=1).values.max()) <= link.tube_radius_b + 1e-6
