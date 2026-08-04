"""Deterministic full-batch CPU training with mandatory step-zero checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import Dataset
from .models import ActivationName, MLP


CHECKPOINT_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 1.0)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    max_steps: int = 2_000
    learning_rate: float = 1e-2
    train_accuracy_required: float = 1.0
    eval_accuracy_required: float = 0.99
    cpu_threads: int = 1

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.train_accuracy_required <= 1.0:
            raise ValueError("train_accuracy_required must be in [0, 1]")
        if not 0.0 <= self.eval_accuracy_required <= 1.0:
            raise ValueError("eval_accuracy_required must be in [0, 1]")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")


@dataclass(frozen=True)
class TrainingCheckpoint:
    step: int
    progress: float
    train_loss: float
    train_accuracy: float
    eval_accuracy: float
    state_dict: dict[str, torch.Tensor]
    eval_preactivations: tuple[torch.Tensor, ...]


@dataclass
class TrainingResult:
    model: MLP
    checkpoints: tuple[TrainingCheckpoint, ...]
    passed: bool
    failure_reason: str | None
    final_train_accuracy: float
    final_eval_accuracy: float
    config: TrainingConfig


def seed_everything(seed: int, cpu_threads: int = 1) -> None:
    """Set every RNG used by this experiment and deterministic CPU execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cpu_threads)


def checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return tuple(sorted({int(round(max_steps * fraction)) for fraction in CHECKPOINT_FRACTIONS}))


def meets_training_criterion(
    train_accuracy: float,
    eval_accuracy: float,
    train_accuracy_required: float = 1.0,
    eval_accuracy_required: float = 0.99,
) -> bool:
    """Strict gate for inclusion in the saturation census."""

    return train_accuracy >= train_accuracy_required and eval_accuracy >= eval_accuracy_required


def _tensor_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(dataset.features, dtype=torch.float32, device="cpu")
    labels = torch.as_tensor(dataset.labels, dtype=torch.int64, device="cpu")
    return features, labels


@torch.no_grad()
def _accuracy(model: MLP, features: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(features)
    return float((logits.argmax(dim=1) == labels).to(torch.float64).mean().item())


def _capture_checkpoint(
    model: MLP,
    step: int,
    max_steps: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
) -> TrainingCheckpoint:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_logits = model(train_features)
        train_loss = float(F.cross_entropy(train_logits, train_labels).item())
        train_accuracy = float(
            (train_logits.argmax(dim=1) == train_labels).to(torch.float64).mean().item()
        )
        eval_accuracy = _accuracy(model, eval_features, eval_labels)
        preactivations = tuple(model.collect_preactivations(eval_features))
    if was_training:
        model.train()
    state = {name: value.detach().to(device="cpu").clone() for name, value in model.state_dict().items()}
    return TrainingCheckpoint(
        step=step,
        progress=step / max_steps,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        state_dict=state,
        eval_preactivations=preactivations,
    )


def train_mlp(
    train_data: Dataset,
    eval_data: Dataset,
    hidden_depth: int,
    hidden_width: int,
    activation: ActivationName,
    config: TrainingConfig,
) -> TrainingResult:
    """Train one MLP in float32 on CPU and capture all mandated checkpoints."""

    config.validate()
    seed_everything(config.seed, config.cpu_threads)
    model = MLP(3, hidden_depth, hidden_width, activation).to(device="cpu", dtype=torch.float32)
    train_features, train_labels = _tensor_dataset(train_data)
    eval_features, eval_labels = _tensor_dataset(eval_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    required_steps = set(checkpoint_steps(config.max_steps))
    checkpoints: list[TrainingCheckpoint] = []

    # This happens before optimizer construction can perform any update and is
    # deliberately outside the training loop so step zero cannot be skipped.
    checkpoints.append(
        _capture_checkpoint(
            model,
            0,
            config.max_steps,
            train_features,
            train_labels,
            eval_features,
            eval_labels,
        )
    )

    model.train()
    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

        if step in required_steps:
            checkpoints.append(
                _capture_checkpoint(
                    model,
                    step,
                    config.max_steps,
                    train_features,
                    train_labels,
                    eval_features,
                    eval_labels,
                )
            )

    final = checkpoints[-1]
    passed = meets_training_criterion(
        final.train_accuracy,
        final.eval_accuracy,
        config.train_accuracy_required,
        config.eval_accuracy_required,
    )
    failure_reason = None
    if not passed:
        failure_reason = (
            f"required train_accuracy>={config.train_accuracy_required:.6f} and "
            f"eval_accuracy>={config.eval_accuracy_required:.6f}; got "
            f"train_accuracy={final.train_accuracy:.6f}, "
            f"eval_accuracy={final.eval_accuracy:.6f}"
        )
    return TrainingResult(
        model=model,
        checkpoints=tuple(checkpoints),
        passed=passed,
        failure_reason=failure_reason,
        final_train_accuracy=final.train_accuracy,
        final_eval_accuracy=final.eval_accuracy,
        config=copy.copy(config),
    )
