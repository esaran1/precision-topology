"""Tidy saturation and collision measurements for trained MLP checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .precision import (
    DELTA_TABLE,
    collision_metrics,
    per_unit_collision_rates,
    saturation_metrics,
)
from .train import TrainingCheckpoint, TrainingResult


@dataclass(frozen=True)
class RunMetadata:
    dataset: str
    depth: int
    width: int
    activation: str
    seed: int
    train_data_seed: int
    eval_data_seed: int
    n_train_per_class: int
    n_eval_per_class: int
    tube_radius: float | None


def _collision_excess(
    current: torch.Tensor,
    baseline: torch.Tensor,
    quantizer: str | None,
    current_vector_rate: float | None,
    baseline_vector_rate: float | None,
) -> dict[str, float | None]:
    if quantizer is None:
        return {
            "excess_per_unit_collision_mean": None,
            "excess_per_unit_collision_std": None,
            "excess_per_unit_collision_min": None,
            "excess_per_unit_collision_median": None,
            "excess_per_unit_collision_max": None,
            "excess_vector_collision_rate": None,
        }
    current_rates = per_unit_collision_rates(current, quantizer)
    baseline_rates = per_unit_collision_rates(baseline, quantizer)
    assert current_rates is not None and baseline_rates is not None
    excess = current_rates - baseline_rates
    assert current_vector_rate is not None and baseline_vector_rate is not None
    return {
        "excess_per_unit_collision_mean": float(excess.mean().item()),
        "excess_per_unit_collision_std": float(excess.std(correction=0).item()),
        "excess_per_unit_collision_min": float(excess.min().item()),
        "excess_per_unit_collision_median": float(torch.quantile(excess, 0.5).item()),
        "excess_per_unit_collision_max": float(excess.max().item()),
        "excess_vector_collision_rate": current_vector_rate - baseline_vector_rate,
    }


def measure_checkpoint(
    result: TrainingResult,
    checkpoint: TrainingCheckpoint,
    initialization: TrainingCheckpoint,
    metadata: RunMetadata,
) -> pd.DataFrame:
    """Measure one checkpoint relative to its step-zero untrained baseline."""

    if not result.passed:
        raise ValueError("failed training runs must be excluded from the census")
    if initialization.step != 0:
        raise ValueError("collision baseline must be the step-zero checkpoint")
    if len(checkpoint.eval_preactivations) != metadata.depth:
        raise ValueError("checkpoint layer count does not match metadata depth")
    if len(initialization.eval_preactivations) != metadata.depth:
        raise ValueError("initialization layer count does not match metadata depth")

    rows: list[dict[str, object]] = []
    for layer_index, (values, baseline_values) in enumerate(
        zip(checkpoint.eval_preactivations, initialization.eval_preactivations),
        start=1,
    ):
        collision_cache: dict[str | None, tuple[dict[str, float | None], dict[str, float | None]]] = {}
        for spec in DELTA_TABLE:
            paper = saturation_metrics(values, spec.delta, "paper")
            exact = saturation_metrics(values, spec.delta, "exact")
            if spec.quantizer not in collision_cache:
                collision_cache[spec.quantizer] = (
                    collision_metrics(values, spec.quantizer),
                    collision_metrics(baseline_values, spec.quantizer),
                )
            collisions, baseline_collisions = collision_cache[spec.quantizer]
            excess = _collision_excess(
                values,
                baseline_values,
                spec.quantizer,
                collisions["vector_collision_rate"],
                baseline_collisions["vector_collision_rate"],
            )
            row: dict[str, object] = {
                **metadata.__dict__,
                "layer": layer_index,
                "distance_from_output": metadata.depth - layer_index,
                "relative_layer_position": layer_index / metadata.depth,
                "format": spec.format,
                "mantissa_bits": spec.mantissa_bits,
                "delta_value": spec.delta,
                "convention": spec.convention,
                "quantizer": spec.quantizer,
                "precision_note": spec.note,
                "training_step": checkpoint.step,
                "training_progress": checkpoint.progress,
                "train_accuracy": checkpoint.train_accuracy,
                "eval_accuracy": checkpoint.eval_accuracy,
                "training_steps_total": result.config.max_steps,
                "learning_rate": result.config.learning_rate,
                "train_accuracy_required": result.config.train_accuracy_required,
                "eval_accuracy_required": result.config.eval_accuracy_required,
                "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
            }
            row.update({f"paper_{name}": value for name, value in paper.items()})
            row.update({f"exact_{name}": value for name, value in exact.items()})
            row.update(collisions)
            row.update({f"baseline_{name}": value for name, value in baseline_collisions.items()})
            row.update(excess)
            rows.append(row)
    return pd.DataFrame(rows)
