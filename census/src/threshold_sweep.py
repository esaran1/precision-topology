"""The monotonicity-threshold sweep: two parametric activation families.

Predictions were registered in ``results/threshold_prediction.md`` before this
module existed.  Family A is ``x + a*sin(x)`` with analytic threshold ``a = 1``;
Family B is negative-slope leaky-ReLU with analytic threshold ``alpha = 0``.
The four fixed activations run in the same sweep, same seeds, so the
comparison is within-run.

Confound measurements ride along with every run: final train loss, the
gradient norm of the full-batch loss at the final weights, and the fraction of
hidden units whose activation output is constant on the eval set.  These are
what let a transition in separation rate be distinguished from a transition in
trainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .models import (
    DEFAULT_INITIALIZATION_GAIN,
    DEFAULT_INITIALIZATION_SCHEME,
    MLP,
    parametric_monotonic,
)
from .train import TrainingConfig, TrainingResult, train_mlp


CHANCE_ACCURACY = 0.5

# Grids exactly as registered.  ``None`` parameter marks a fixed activation.
FAMILY_A = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0)
FAMILY_B = (-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
BASELINES = ("tanh", "relu", "leaky_relu", "gelu")


def conditions() -> tuple[tuple[str, float | None], ...]:
    """(activation, parameter) pairs for the whole sweep."""

    rows: list[tuple[str, float | None]] = []
    rows.extend(("sin_family", value) for value in FAMILY_A)
    rows.extend(("pwl_family", value) for value in FAMILY_B)
    rows.extend((name, None) for name in BASELINES)
    return tuple(rows)


def condition_monotonic(activation: str, parameter: float | None) -> bool:
    if parameter is None:
        return activation != "gelu"
    return parametric_monotonic(activation, parameter)


@dataclass(frozen=True)
class ThresholdSweepConfig:
    widths: Sequence[int] = (3, 4)
    depths: Sequence[int] = (3, 5, 8, 12)
    seeds: Sequence[int] = tuple(range(20))
    dataset: str = "linked_tori"
    n_per_class: int = 1_000
    max_steps: int = 2_000
    learning_rate: float = 1e-2
    tube_radius: float = 0.2

    def total_runs(self) -> int:
        return (
            len(conditions()) * len(self.widths) * len(self.depths) * len(self.seeds)
        )

    def as_census_config(self) -> SweepConfig:
        return SweepConfig(
            n_train_per_class=self.n_per_class,
            n_eval_per_class=self.n_per_class,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            tube_radius=self.tube_radius,
        )


def final_gradient_norm(
    model: MLP, features: torch.Tensor, labels: torch.Tensor
) -> float:
    """Norm of the full-batch loss gradient at the final weights."""

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(features), labels)
    loss.backward()
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.to(torch.float64).pow(2).sum()
    model.zero_grad(set_to_none=True)
    return float(total.sqrt().item())


def inactive_unit_fraction(result: TrainingResult) -> float:
    """Fraction of hidden units whose activation output is constant on eval.

    A dead ReLU-family unit (always in one linear piece) and a saturated
    tanh-family unit both register here.  Computed from the final checkpoint's
    stored eval preactivations, so no extra forward pass is needed.
    """

    final = result.checkpoints[-1]
    inactive = 0
    total = 0
    for preactivation in final.eval_preactivations:
        output = result.model._activate(preactivation.to(torch.float64))
        variance = output.var(dim=0, correction=0)
        inactive += int((variance < 1e-12).sum().item())
        total += int(variance.numel())
    return inactive / total if total else 0.0


def run(
    config: ThresholdSweepConfig, output_directory: Path, verbose: bool = True
) -> pd.DataFrame:
    data_config = config.as_census_config()
    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0

    for activation, parameter in conditions():
        for depth in config.depths:
            for width in config.widths:
                for seed in config.seeds:
                    started = time.monotonic()
                    train_data, eval_data, train_seed, eval_seed = _make_data(
                        config.dataset, seed, data_config
                    )
                    result = train_mlp(
                        train_data,
                        eval_data,
                        hidden_depth=depth,
                        hidden_width=width,
                        activation=activation,  # type: ignore[arg-type]
                        config=TrainingConfig(
                            seed=seed,
                            max_steps=config.max_steps,
                            learning_rate=config.learning_rate,
                        ),
                        activation_parameter=parameter,
                    )
                    train_features = torch.as_tensor(
                        train_data.features, dtype=torch.float32
                    )
                    train_labels = torch.as_tensor(
                        train_data.labels, dtype=torch.int64
                    )
                    gradient_norm = final_gradient_norm(
                        result.model, train_features, train_labels
                    )
                    duration = time.monotonic() - started
                    completed += 1
                    rows.append(
                        {
                            "family": (
                                "A" if activation == "sin_family"
                                else "B" if activation == "pwl_family"
                                else "fixed"
                            ),
                            "activation": activation,
                            "parameter": parameter,
                            "monotonic": condition_monotonic(activation, parameter),
                            "depth": depth,
                            "width": width,
                            "seed": seed,
                            "train_data_seed": train_seed,
                            "eval_data_seed": eval_seed,
                            "passed": result.passed,
                            "final_train_accuracy": result.final_train_accuracy,
                            "final_eval_accuracy": result.final_eval_accuracy,
                            "perfect_eval": result.final_eval_accuracy >= 1.0,
                            "at_chance": result.final_eval_accuracy <= CHANCE_ACCURACY,
                            "eval_errors": int(
                                round(
                                    (1.0 - result.final_eval_accuracy)
                                    * 2
                                    * config.n_per_class
                                )
                            ),
                            "final_train_loss": result.checkpoints[-1].train_loss,
                            "final_gradient_norm": gradient_norm,
                            "inactive_unit_fraction": inactive_unit_fraction(result),
                            "max_steps": config.max_steps,
                            "learning_rate": config.learning_rate,
                            "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                            "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                            "duration_seconds": duration,
                        }
                    )
                    if verbose and completed % 200 == 0:
                        label = (
                            f"{activation}({parameter:g})"
                            if parameter is not None
                            else activation
                        )
                        print(
                            f"[{completed}/{total}] {label} d={depth} w={width} "
                            f"s={seed} eval={result.final_eval_accuracy:.4f}",
                            flush=True,
                        )
                    if completed % 400 == 0 or completed == total:
                        _write(pd.DataFrame(rows), output_directory)

    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "threshold_sweep"
    with artifact_lock(stem, "monotonicity threshold sweep"):
        csv_temp = stem.with_suffix(".csv.tmp")
        parquet_temp = stem.with_suffix(".parquet.tmp")
        frame.to_csv(csv_temp, index=False)
        frame.to_parquet(parquet_temp, index=False)
        csv_temp.replace(stem.with_suffix(".csv"))
        parquet_temp.replace(stem.with_suffix(".parquet"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = ThresholdSweepConfig()
    print(f"threshold sweep: {config.total_runs()} runs", flush=True)
    run(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
