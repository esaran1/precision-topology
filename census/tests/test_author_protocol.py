"""Tests for the Appendix G.2 training path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.author_protocol import (
    UNSPECIFIED,
    AuthorTrainingConfig,
    train_mlp_author,
)
from src.census import _make_data
from src.width_sweep import WidthSweepConfig


def _data(seed: int = 0):
    return _make_data("linked_tori", seed, WidthSweepConfig().as_census_config())[:2]


def test_config_matches_the_published_settings():
    config = AuthorTrainingConfig(seed=0)
    assert config.learning_rate == 1e-3
    assert config.batch_size == 128
    assert config.max_epochs == 800
    assert 100 <= config.patience <= 200


def test_every_unspecified_choice_is_recorded():
    names = {name for name, _ in UNSPECIFIED}
    assert {"patience", "early-stopping monitor", "checkpoint restored"} <= names
    for _, choice in UNSPECIFIED:
        assert choice


def test_invalid_configs_are_rejected():
    for kwargs in (
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"max_epochs": 0},
        {"patience": 0},
    ):
        with pytest.raises(ValueError):
            AuthorTrainingConfig(seed=0, **kwargs).validate()  # type: ignore[arg-type]


def test_training_is_deterministic_for_a_seed():
    train_data, eval_data = _data()
    config = AuthorTrainingConfig(seed=3, max_epochs=12, patience=100)
    first = train_mlp_author(train_data, eval_data, 3, 6, "tanh", config)
    second = train_mlp_author(train_data, eval_data, 3, 6, "tanh", config)
    assert first.final_eval_accuracy == second.final_eval_accuracy
    assert first.best_epoch == second.best_epoch


def test_restores_the_best_checkpoint_not_the_last():
    """Early stopping is pointless if the final epoch is reported instead."""

    train_data, eval_data = _data()
    config = AuthorTrainingConfig(seed=1, max_epochs=40, patience=5)
    result = train_mlp_author(train_data, eval_data, 5, 6, "tanh", config)
    # The restored model must score exactly what the best epoch scored, so the
    # reported accuracy cannot be below any epoch the run passed through.
    assert result.best_epoch <= result.epochs_run
    features = torch.as_tensor(eval_data.features, dtype=torch.float32)
    labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)
    with torch.no_grad():
        recomputed = float(
            (result.model(features).argmax(dim=1) == labels).to(torch.float64).mean()
        )
    assert recomputed == pytest.approx(result.final_eval_accuracy, abs=1e-12)


def test_early_stopping_triggers_on_a_short_patience():
    train_data, eval_data = _data()
    result = train_mlp_author(
        train_data,
        eval_data,
        3,
        3,
        "relu",
        AuthorTrainingConfig(seed=0, max_epochs=500, patience=3),
    )
    assert result.stopped_early
    assert result.epochs_run < 500


def test_minibatching_uses_the_configured_size():
    """A batch size above the dataset makes it full-batch; below it does not."""

    train_data, eval_data = _data()
    small = train_mlp_author(
        train_data, eval_data, 3, 6, "tanh",
        AuthorTrainingConfig(seed=2, batch_size=128, max_epochs=6, patience=100),
    )
    full = train_mlp_author(
        train_data, eval_data, 3, 6, "tanh",
        AuthorTrainingConfig(seed=2, batch_size=4000, max_epochs=6, patience=100),
    )
    # Different gradient paths must give different results after equal epochs.
    assert small.final_eval_accuracy != full.final_eval_accuracy


def test_accuracy_is_a_valid_fraction():
    train_data, eval_data = _data()
    result = train_mlp_author(
        train_data, eval_data, 3, 5, "gelu",
        AuthorTrainingConfig(seed=0, max_epochs=10, patience=100),
    )
    for value in (result.final_train_accuracy, result.final_eval_accuracy):
        assert 0.0 <= value <= 1.0
    # 2,000 evaluation points, so accuracy is a multiple of 1/2000.
    assert abs(result.final_eval_accuracy * 2000 - round(result.final_eval_accuracy * 2000)) < 1e-9
