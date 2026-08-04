"""Tidy saturation and collision measurements for trained MLP checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd
import torch

from .data import Dataset, gaussian_blobs, linked_tori
from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .precision import (
    DELTA_TABLE,
    collision_metrics,
    per_unit_collision_rates,
    saturation_metrics,
)
from .train import TrainingCheckpoint, TrainingResult
from .train import TrainingConfig, train_mlp


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
    activation: str,
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
    current_rates = per_unit_collision_rates(current, quantizer, activation)  # type: ignore[arg-type]
    baseline_rates = per_unit_collision_rates(baseline, quantizer, activation)  # type: ignore[arg-type]
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
    eval_labels: torch.Tensor,
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
        specs = (
            DELTA_TABLE
            if metadata.activation == "tanh"
            else tuple(spec for spec in DELTA_TABLE if not spec.format.startswith("fixed-"))
        )
        for spec in specs:
            paper = saturation_metrics(values, spec.delta, "paper")
            exact = saturation_metrics(values, spec.delta, "exact")
            if spec.quantizer not in collision_cache:
                collision_cache[spec.quantizer] = (
                    collision_metrics(values, spec.quantizer, metadata.activation, eval_labels),
                    collision_metrics(baseline_values, spec.quantizer, metadata.activation, eval_labels),
                )
            collisions, baseline_collisions = collision_cache[spec.quantizer]
            excess = _collision_excess(
                values,
                baseline_values,
                spec.quantizer,
                collisions["vector_collision_rate"],
                baseline_collisions["vector_collision_rate"],
                metadata.activation,
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


@dataclass(frozen=True)
class SweepConfig:
    datasets: Sequence[str] = ("linked_tori", "blobs")
    depths: Sequence[int] = (4, 6, 8, 10)
    widths: Sequence[int] = (5, 15, 30, 50)
    # Tanh runs first so the primary result is persisted before controls.
    activations: Sequence[str] = ("tanh", "relu", "leaky_relu")
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    n_train_per_class: int = 1_000
    n_eval_per_class: int = 1_000
    max_steps: int = 2_000
    learning_rate: float = 1e-2
    tube_radius: float = 0.2
    blob_standard_deviation: float = 0.2
    blob_separation: float = 4.0
    dynamics_depth: int = 6
    dynamics_width: int = 30


def _make_data(dataset: str, seed: int, config: SweepConfig) -> tuple[Dataset, Dataset, int, int]:
    train_seed = 10_000 + seed
    eval_seed = 20_000 + seed
    if dataset == "linked_tori":
        train_data = linked_tori(
            config.n_train_per_class,
            tube_radius=config.tube_radius,
            seed=train_seed,
        )
        eval_data = linked_tori(
            config.n_eval_per_class,
            tube_radius=config.tube_radius,
            seed=eval_seed,
        )
    elif dataset == "blobs":
        train_data = gaussian_blobs(
            config.n_train_per_class,
            standard_deviation=config.blob_standard_deviation,
            separation=config.blob_separation,
            seed=train_seed,
        )
        eval_data = gaussian_blobs(
            config.n_eval_per_class,
            standard_deviation=config.blob_standard_deviation,
            separation=config.blob_separation,
            seed=eval_seed,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return train_data, eval_data, train_seed, eval_seed


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = stem.with_suffix(".csv.tmp")
    parquet_temp = stem.with_suffix(".parquet.tmp")
    frame.to_csv(csv_temp, index=False)
    frame.to_parquet(parquet_temp, index=False)
    csv_temp.replace(stem.with_suffix(".csv"))
    parquet_temp.replace(stem.with_suffix(".parquet"))


def run_sweep(config: SweepConfig, output_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the configured grid, persisting accepted census rows after every run."""

    total = (
        len(config.datasets)
        * len(config.depths)
        * len(config.widths)
        * len(config.activations)
        * len(config.seeds)
    )
    census_frames: list[pd.DataFrame] = []
    dynamics_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    completed = 0

    for activation in config.activations:
        for dataset_name in config.datasets:
            for depth in config.depths:
                for width in config.widths:
                    for seed in config.seeds:
                        started = time.monotonic()
                        train_data, eval_data, train_seed, eval_seed = _make_data(
                            dataset_name, seed, config
                        )
                        training_config = TrainingConfig(
                            seed=seed,
                            max_steps=config.max_steps,
                            learning_rate=config.learning_rate,
                        )
                        result = train_mlp(
                            train_data,
                            eval_data,
                            hidden_depth=depth,
                            hidden_width=width,
                            activation=activation,  # type: ignore[arg-type]
                            config=training_config,
                        )
                        duration = time.monotonic() - started
                        completed += 1
                        metadata = RunMetadata(
                            dataset=dataset_name,
                            depth=depth,
                            width=width,
                            activation=activation,
                            seed=seed,
                            train_data_seed=train_seed,
                            eval_data_seed=eval_seed,
                            n_train_per_class=config.n_train_per_class,
                            n_eval_per_class=config.n_eval_per_class,
                            tube_radius=config.tube_radius if dataset_name == "linked_tori" else None,
                        )
                        status_rows.append(
                            {
                                **metadata.__dict__,
                                "passed": result.passed,
                                "failure_reason": result.failure_reason,
                                "final_train_accuracy": result.final_train_accuracy,
                                "final_eval_accuracy": result.final_eval_accuracy,
                                "max_steps": config.max_steps,
                                "learning_rate": config.learning_rate,
                                "train_accuracy_required": training_config.train_accuracy_required,
                                "eval_accuracy_required": training_config.eval_accuracy_required,
                                "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                                "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                                "duration_seconds": duration,
                            }
                        )
                        if result.passed:
                            labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)
                            final_frame = measure_checkpoint(
                                result,
                                result.checkpoints[-1],
                                result.checkpoints[0],
                                metadata,
                                labels,
                            )
                            census_frames.append(final_frame)
                            if depth == config.dynamics_depth and width == config.dynamics_width:
                                dynamics_frames.extend(
                                    measure_checkpoint(
                                        result,
                                        checkpoint,
                                        result.checkpoints[0],
                                        metadata,
                                        labels,
                                    )
                                    for checkpoint in result.checkpoints
                                )

                        census_frame = pd.concat(census_frames, ignore_index=True) if census_frames else pd.DataFrame()
                        dynamics_frame = (
                            pd.concat(dynamics_frames, ignore_index=True)
                            if dynamics_frames
                            else pd.DataFrame()
                        )
                        status_frame = pd.DataFrame(status_rows)
                        if not census_frame.empty:
                            _write_frame(census_frame, output_directory / "saturation")
                        _write_frame(status_frame, output_directory / "training_status")
                        if not dynamics_frame.empty:
                            _write_frame(dynamics_frame, output_directory / "training_dynamics")
                        print(
                            f"[{completed}/{total}] {activation} {dataset_name} depth={depth} "
                            f"width={width} seed={seed} passed={result.passed} "
                            f"train={result.final_train_accuracy:.4f} eval={result.final_eval_accuracy:.4f} "
                            f"seconds={duration:.2f}",
                            flush=True,
                        )

    return census_frame, status_frame, dynamics_frame


def main() -> None:
    run_sweep(SweepConfig(), Path(__file__).resolve().parents[1] / "results")


if __name__ == "__main__":
    main()
