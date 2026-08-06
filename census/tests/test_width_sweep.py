"""Tests for Half A of the d=3 width sweep."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models import MONOTONIC_ACTIVATIONS, NONMONOTONIC_ACTIVATIONS, MLP
from src.width_sweep import CHANCE_ACCURACY, WidthSweepConfig, run_width_sweep


def test_gelu_is_non_monotonic():
    """The whole reason GELU is in the grid: it is not monotonic.

    Every other activation in the sweep is monotonic and therefore in one
    expressivity class under Ren and Lim.
    """

    values = torch.linspace(-3.0, 0.0, 200, dtype=torch.float64).requires_grad_(True)
    outputs = torch.nn.functional.gelu(values)
    gradients = torch.autograd.grad(outputs.sum(), values)[0]
    assert gradients.min() < 0.0


def test_monotonic_activations_are_monotonic():
    values = torch.linspace(-5.0, 5.0, 400, dtype=torch.float64).requires_grad_(True)
    for name in MONOTONIC_ACTIVATIONS:
        model = MLP(1, 1, 1, name)  # type: ignore[arg-type]
        outputs = model._activate(values)
        gradients = torch.autograd.grad(outputs.sum(), values, retain_graph=True)[0]
        assert gradients.min() >= 0.0, f"{name} is not monotonic"


def test_activation_classes_are_disjoint_and_cover_the_grid():
    config = WidthSweepConfig()
    assert set(config.activations) == set(
        MONOTONIC_ACTIVATIONS + NONMONOTONIC_ACTIVATIONS
    )
    assert not set(MONOTONIC_ACTIVATIONS) & set(NONMONOTONIC_ACTIVATIONS)


def test_mlp_supports_gelu():
    model = MLP(3, 2, 4, "gelu")
    assert model(torch.randn(6, 3)).shape == (6, 2)


def test_unknown_activation_still_rejected():
    with pytest.raises(ValueError):
        MLP(3, 2, 4, "swish")  # type: ignore[arg-type]


def test_grid_matches_the_approved_design():
    config = WidthSweepConfig()
    assert tuple(config.widths) == (3, 4, 5, 6, 7, 8, 10, 12, 15)
    assert tuple(config.depths) == (3, 5, 8, 12)
    assert tuple(config.seeds) == tuple(range(20))
    assert config.total_runs() == 2880


def test_sweep_records_failed_runs(tmp_path: Path):
    """Pass rate is a primary outcome, so failures must not be dropped."""

    # Width 3 depth 3 with ReLU reliably fails the gate; the row must survive.
    config = WidthSweepConfig(
        widths=(3,), depths=(3,), activations=("relu",), seeds=(0,), max_steps=50
    )
    frame = run_width_sweep(config, tmp_path, verbose=False)
    assert len(frame) == 1
    assert not frame.passed.all()
    assert frame.failure_reason.notna().all()


def test_sweep_records_seeds_and_flags(tmp_path: Path):
    config = WidthSweepConfig(
        widths=(15,), depths=(3,), activations=("tanh",), seeds=(0, 1), max_steps=200
    )
    frame = run_width_sweep(config, tmp_path, verbose=False)
    assert set(frame.seed) == {0, 1}
    assert frame.train_data_seed.tolist() == [10_000, 10_001]
    assert frame.eval_data_seed.tolist() == [20_000, 20_001]
    for column in ("at_chance", "perfect_eval", "monotonic"):
        assert column in frame.columns
    assert frame.monotonic.all()


def test_chance_and_perfect_flags_are_consistent(tmp_path: Path):
    config = WidthSweepConfig(
        widths=(3,), depths=(3,), activations=("relu",), seeds=(0,), max_steps=50
    )
    frame = run_width_sweep(config, tmp_path, verbose=False)
    row = frame.iloc[0]
    assert row.at_chance == (row.final_eval_accuracy <= CHANCE_ACCURACY)
    assert row.perfect_eval == (row.final_eval_accuracy >= 1.0)


def test_sweep_writes_both_artifacts(tmp_path: Path):
    config = WidthSweepConfig(
        widths=(15,), depths=(3,), activations=("tanh",), seeds=(0,), max_steps=100
    )
    run_width_sweep(config, tmp_path, verbose=False)
    assert (tmp_path / "width_sweep.csv").exists()
    assert (tmp_path / "width_sweep.parquet").exists()
