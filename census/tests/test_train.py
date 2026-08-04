import numpy as np
import pytest
import torch

from src.data import gaussian_blobs
from src.models import MLP
from src.train import TrainingConfig, checkpoint_steps, meets_training_criterion, train_mlp


def _small_data():
    return (
        gaussian_blobs(24, seed=10),
        gaussian_blobs(24, seed=11),
    )


def test_checkpoint_schedule_includes_initialization_and_required_progress():
    assert checkpoint_steps(100) == (0, 10, 25, 50, 100)
    assert checkpoint_steps(10) == (0, 1, 2, 5, 10)


def test_step_zero_checkpoint_precedes_updates_and_stores_preactivations():
    train_data, eval_data = _small_data()
    config = TrainingConfig(seed=123, max_steps=10)
    result = train_mlp(train_data, eval_data, 2, 5, "tanh", config)
    initial = result.checkpoints[0]

    torch.manual_seed(123)
    fresh = MLP(3, 2, 5, "tanh")
    assert initial.step == 0
    assert initial.progress == 0.0
    assert all(torch.equal(initial.state_dict[name], value) for name, value in fresh.state_dict().items())
    assert len(initial.eval_preactivations) == 2
    assert all(values.shape == (48, 5) for values in initial.eval_preactivations)


def test_training_is_bit_identical_for_same_seed():
    train_data, eval_data = _small_data()
    config = TrainingConfig(seed=88, max_steps=10)
    first = train_mlp(train_data, eval_data, 2, 5, "relu", config)
    second = train_mlp(train_data, eval_data, 2, 5, "relu", config)
    assert first.final_train_accuracy == second.final_train_accuracy
    assert first.final_eval_accuracy == second.final_eval_accuracy
    assert all(torch.equal(a, b) for a, b in zip(first.model.parameters(), second.model.parameters()))
    for first_checkpoint, second_checkpoint in zip(first.checkpoints, second.checkpoints):
        assert first_checkpoint.step == second_checkpoint.step
        assert first_checkpoint.train_loss == second_checkpoint.train_loss
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                first_checkpoint.eval_preactivations,
                second_checkpoint.eval_preactivations,
            )
        )


def test_training_criterion_is_strict_and_explicit():
    assert meets_training_criterion(1.0, 0.99)
    assert not meets_training_criterion(np.nextafter(1.0, 0.0), 1.0)
    assert not meets_training_criterion(1.0, np.nextafter(0.99, 0.0))


def test_easy_blobs_pass_gate_after_training():
    train_data, eval_data = _small_data()
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=2,
        hidden_width=5,
        activation="tanh",
        config=TrainingConfig(seed=7, max_steps=100),
    )
    assert result.passed, result.failure_reason
    assert result.final_train_accuracy == 1.0
    assert result.final_eval_accuracy >= 0.99
