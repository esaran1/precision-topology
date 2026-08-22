"""Training under the protocol stated in Appendix G.2 of arXiv:2606.31856v1.

Their stated settings:

    Adam (FFN) at learning rate 1e-3, batch size 128, up to 800 epochs,
    early stopping with patience 100-200, cross-entropy loss, on 6000 points
    (3000 per class) split 80/20 train/validation.

This exists as a separate path rather than as options on ``train_mlp`` so our
own protocol is untouched and the two can never be silently mixed.  Everything
the paper does not specify is listed in ``UNSPECIFIED`` below and the choice
made here is stated with it.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch.nn import functional as F

from .data import Dataset
from .models import ActivationName, MLP
from .train import seed_everything


# Decisions the paper does not pin down, and what was chosen.  These are
# reported alongside every result produced by this module.
UNSPECIFIED: tuple[tuple[str, str], ...] = (
    ("patience", "150 epochs, the midpoint of the stated 100-200 range"),
    ("early-stopping monitor", "evaluation accuracy, ties broken by evaluation loss"),
    ("checkpoint restored", "best evaluation accuracy, not the final epoch"),
    ("initialization", "PyTorch nn.Linear default, unchanged from our protocol"),
    ("shuffling", "reshuffled each epoch with a per-run generator"),
    ("last batch", "kept even when smaller than 128"),
    ("weight decay", "none; plain Adam, since AdamW is used only for their ResNets"),
)


@dataclass(frozen=True)
class AuthorTrainingConfig:
    """Appendix G.2 settings."""

    seed: int
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 800
    patience: int = 150
    cpu_threads: int = 1

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


@dataclass
class AuthorTrainingResult:
    model: MLP
    final_train_accuracy: float
    final_eval_accuracy: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    config: AuthorTrainingConfig


def _tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(dataset.features, dtype=torch.float32, device="cpu"),
        torch.as_tensor(dataset.labels, dtype=torch.int64, device="cpu"),
    )


@torch.no_grad()
def _accuracy_and_loss(
    model: MLP, features: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    logits = model(features)
    accuracy = float((logits.argmax(dim=1) == labels).to(torch.float64).mean().item())
    loss = float(F.cross_entropy(logits, labels).item())
    if was_training:
        model.train()
    return accuracy, loss


def train_mlp_author(
    train_data: Dataset,
    eval_data: Dataset,
    hidden_depth: int,
    hidden_width: int,
    activation: ActivationName,
    config: AuthorTrainingConfig,
) -> AuthorTrainingResult:
    """Minibatch Adam with early stopping and best-checkpoint restoration."""

    config.validate()
    seed_everything(config.seed, config.cpu_threads)
    model = MLP(3, hidden_depth, hidden_width, activation).to(
        device="cpu", dtype=torch.float32
    )
    train_features, train_labels = _tensors(train_data)
    eval_features, eval_labels = _tensors(eval_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)

    n_train = train_features.shape[0]
    best_accuracy = -1.0
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improvement = 0
    epochs_run = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        epochs_run = epoch
        model.train()
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_features[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()

        accuracy, loss_value = _accuracy_and_loss(model, eval_features, eval_labels)
        improved = accuracy > best_accuracy or (
            accuracy == best_accuracy and loss_value < best_loss
        )
        if improved:
            best_accuracy = accuracy
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.patience:
                stopped_early = True
                break

    # Restore the best checkpoint rather than the last, which is what early
    # stopping means; reporting the final epoch would discard the point of it.
    model.load_state_dict(best_state)
    train_accuracy, _ = _accuracy_and_loss(model, train_features, train_labels)
    eval_accuracy, _ = _accuracy_and_loss(model, eval_features, eval_labels)
    return AuthorTrainingResult(
        model=model,
        final_train_accuracy=train_accuracy,
        final_eval_accuracy=eval_accuracy,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        config=config,
    )
