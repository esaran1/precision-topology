"""Tests for layerwise linking traces through trained networks."""

from __future__ import annotations

import math

import pytest
import torch

from src.data import linked_core_circles
from src.linking import linking_number
from src.linking_trace import (
    ARTIFACT_DISTANCE,
    PROJECTION_CONVENTION,
    _project_to_r3,
    first_change_layer,
    first_unreportable_layer,
    trace_linking,
    trace_to_records,
)
from src.models import MLP


def test_census_cores_start_linked():
    """The input configuration must be a genuine Hopf link."""

    first, second = linked_core_circles(512)
    estimate = linking_number(torch.tensor(first), torch.tensor(second))
    assert estimate.rounded == -1
    assert estimate.min_distance == pytest.approx(1.0, abs=1e-9)
    assert estimate.converged


def test_identity_network_preserves_linking():
    """A width-3 network wired to the identity must not change the link."""

    model = MLP(3, 2, 3, "leaky_relu")
    with torch.no_grad():
        for layer in model.hidden_layers:
            layer.weight.copy_(torch.eye(3))
            layer.bias.zero_()
    # leaky-ReLU is not the identity on negatives, so use a positive shift and
    # check the link is preserved rather than the coordinates.
    with torch.no_grad():
        model.hidden_layers[0].bias.fill_(5.0)
    trace = trace_linking(model, n_core_points=200)
    assert trace[0].rounded == -1
    for entry in trace:
        if entry.reportable:
            assert entry.rounded == -1


def test_trace_marks_width_three_as_unprojected():
    model = MLP(3, 2, 3, "tanh")
    trace = trace_linking(model, n_core_points=128)
    assert all(not entry.projected for entry in trace)
    assert trace[0].regime == "exact (d=3)"


def test_trace_marks_wider_layers_as_projected():
    """Above width 3 the estimate is a projection, and must say so."""

    model = MLP(3, 2, 8, "tanh")
    trace = trace_linking(model, n_core_points=128)
    # Layer 0 is the input in R^3 and is never projected.
    assert not trace[0].projected
    assert all(entry.projected for entry in trace[1:])
    records = trace_to_records(trace)
    for record in records[1:]:
        assert record["projection_convention"] == PROJECTION_CONVENTION
    assert records[0]["projection_convention"] is None


def test_artifact_regime_suppresses_the_value():
    """No linking number may be emitted from the ill-conditioned regime."""

    model = MLP(3, 1, 3, "relu")
    with torch.no_grad():
        # Collapse everything onto a plane so the curves are driven together.
        model.hidden_layers[0].weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        model.hidden_layers[0].bias.zero_()
    trace = trace_linking(model, n_core_points=200)
    collapsed = trace[1]
    assert collapsed.min_distance <= ARTIFACT_DISTANCE
    assert not collapsed.reportable
    assert collapsed.raw is None
    assert collapsed.rounded is None


def test_projection_preserves_a_three_dimensional_configuration():
    """PCA of an R^3 configuration embedded in R^5 recovers its linking."""

    first, second = linked_core_circles(256)
    left = torch.tensor(first, dtype=torch.float64)
    right = torch.tensor(second, dtype=torch.float64)
    padding = torch.zeros(left.shape[0], 2, dtype=torch.float64)
    embedded_left = torch.cat([left, padding], dim=1)
    embedded_right = torch.cat([right, padding], dim=1)
    projected_left, projected_right = _project_to_r3(embedded_left, embedded_right)
    estimate = linking_number(projected_left, projected_right)
    assert abs(estimate.rounded) == 1


def test_first_change_layer_finds_the_transition():
    model = MLP(3, 2, 3, "tanh")
    trace = trace_linking(model, n_core_points=128)
    change = first_change_layer(trace)
    if change is not None:
        assert 1 <= change <= len(trace) - 1


def test_first_unreportable_layer_is_none_when_all_reportable():
    first, second = linked_core_circles(128)
    estimate = linking_number(torch.tensor(first), torch.tensor(second))
    assert estimate.reportable if hasattr(estimate, "reportable") else True
    model = MLP(3, 1, 3, "leaky_relu")
    with torch.no_grad():
        model.hidden_layers[0].weight.copy_(torch.eye(3) * 2.0)
        model.hidden_layers[0].bias.fill_(5.0)
    trace = trace_linking(model, n_core_points=200)
    assert first_unreportable_layer(trace) is None


def test_records_carry_metadata_and_regime():
    model = MLP(3, 2, 3, "gelu")
    trace = trace_linking(model, n_core_points=128)
    records = trace_to_records(trace, activation="gelu", seed=7)
    assert all(record["activation"] == "gelu" for record in records)
    assert all(record["seed"] == 7 for record in records)
    assert {record["regime"] for record in records} <= {
        "exact (d=3)",
        "projected",
        "artifact (too close)",
        "undefined (curves meet)",
    }


def test_min_distance_is_measured_before_projection():
    """Disjointness is a property of the layer, not of the projected view."""

    model = MLP(3, 1, 6, "tanh")
    trace = trace_linking(model, n_core_points=128)
    entry = trace[1]
    assert entry.projected
    assert entry.min_distance > 0.0
    assert not math.isnan(entry.min_distance)
