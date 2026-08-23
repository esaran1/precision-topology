"""Determinism is the foundation every recovery check stands on."""

from __future__ import annotations

import torch

from src.census import SweepConfig, _make_data
from src.reproduce import ARTIFACTS
from src.train import TrainingConfig, train_mlp


def test_training_is_bit_deterministic():
    config = SweepConfig(
        n_train_per_class=200,
        n_eval_per_class=200,
        max_steps=100,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    train_data, eval_data, *_ = _make_data("linked_tori", 3, config)
    runs = [
        train_mlp(
            train_data,
            eval_data,
            hidden_depth=2,
            hidden_width=3,
            activation="gelu",
            config=TrainingConfig(seed=3, max_steps=100),
        )
        for _ in range(2)
    ]
    for left, right in zip(
        runs[0].model.state_dict().values(), runs[1].model.state_dict().values()
    ):
        assert torch.equal(left, right)
    assert runs[0].final_eval_accuracy == runs[1].final_eval_accuracy


def test_data_generation_is_deterministic():
    config = SweepConfig(
        n_train_per_class=100,
        n_eval_per_class=100,
        max_steps=1,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    first, _, *_ = _make_data("linked_tori", 7, config)
    second, _, *_ = _make_data("linked_tori", 7, config)
    assert (first.features == second.features).all()


def test_artifact_registry_names_real_modules():
    import importlib

    for stem, (module, _) in ARTIFACTS.items():
        loaded = importlib.import_module(module)
        assert hasattr(loaded, "main"), stem
