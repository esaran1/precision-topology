"""Train across torus parametrizations at widths 3 and 4.

Every configuration is validated as a genuine embedded link before any training
happens; an invalid one aborts rather than silently producing a cell whose
class supports are unlinked or self-intersecting.

Data seeds are offset from the main census range so a parametrization run can
never be confused with a baseline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from .artifact_lock import artifact_lock
from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .parametrization import GRID, TorusLink, axis_alignment, sample_link, validate
from .train import TrainingConfig, train_mlp


TRAIN_SEED_BASE = 30_000
EVAL_SEED_BASE = 40_000
CHANCE_ACCURACY = 0.5


@dataclass(frozen=True)
class ParametrizationSweepConfig:
    widths: Sequence[int] = (3, 4)
    depths: Sequence[int] = (3, 5, 8, 12)
    activations: Sequence[str] = ("tanh", "relu", "leaky_relu", "gelu")
    seeds: Sequence[int] = tuple(range(10))
    links: Sequence[TorusLink] = GRID
    n_per_class: int = 1_000
    max_steps: int = 2_000
    learning_rate: float = 1e-2

    def total_runs(self) -> int:
        return (
            len(self.links)
            * len(self.widths)
            * len(self.depths)
            * len(self.activations)
            * len(self.seeds)
        )


def validate_grid(links: Sequence[TorusLink]) -> pd.DataFrame:
    """Validate every configuration, raising if any is not a genuine link."""

    rows = []
    for link in links:
        report = validate(link)
        alignment_a, alignment_b = axis_alignment(link)
        rows.append(
            {
                "name": report.name,
                "linking_number": report.linking_number,
                "linking_residual": report.linking_residual,
                "core_separation": report.core_separation,
                "tube_gap": report.tube_gap,
                "axis_alignment_a": alignment_a,
                "axis_alignment_b": alignment_b,
                "axis_aligned": alignment_b < 1e-9,
                "major_radius_a": link.major_radius_a,
                "major_radius_b": link.major_radius_b,
                "tube_radius_a": link.tube_radius_a,
                "tube_radius_b": link.tube_radius_b,
                "aspect_a": link.aspect_a(),
                "aspect_b": link.aspect_b(),
                "offset": str(link.offset),
                "rotation_degrees": str(link.rotation_degrees),
                "valid": report.valid,
                "reason": report.reason(),
            }
        )
    frame = pd.DataFrame(rows)
    invalid = frame[~frame.valid]
    if not invalid.empty:
        raise ValueError(
            "invalid link configurations: "
            + "; ".join(f"{row['name']}: {row['reason']}" for _, row in invalid.iterrows())
        )
    return frame


def run(
    config: ParametrizationSweepConfig,
    output_directory: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    validation = validate_grid(config.links)
    validation.to_csv(output_directory / "parametrization_validation.csv", index=False)

    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0
    for link in config.links:
        alignment_a, alignment_b = axis_alignment(link)
        for activation in config.activations:
            for depth in config.depths:
                for width in config.widths:
                    for seed in config.seeds:
                        train_seed = TRAIN_SEED_BASE + seed
                        eval_seed = EVAL_SEED_BASE + seed
                        train_data = sample_link(link, config.n_per_class, train_seed)
                        eval_data = sample_link(link, config.n_per_class, eval_seed)
                        started = time.monotonic()
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
                        )
                        duration = time.monotonic() - started
                        completed += 1
                        rows.append(
                            {
                                "parametrization": link.name,
                                "axis_aligned": alignment_b < 1e-9,
                                "axis_alignment_b": alignment_b,
                                "major_radius_a": link.major_radius_a,
                                "major_radius_b": link.major_radius_b,
                                "tube_radius_a": link.tube_radius_a,
                                "tube_radius_b": link.tube_radius_b,
                                "aspect_a": link.aspect_a(),
                                "aspect_b": link.aspect_b(),
                                "offset": str(link.offset),
                                "rotation_degrees": str(link.rotation_degrees),
                                "n_per_class": config.n_per_class,
                                "activation": activation,
                                "monotonic": activation != "gelu",
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
                                "max_steps": config.max_steps,
                                "learning_rate": config.learning_rate,
                                "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                                "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                                "duration_seconds": duration,
                            }
                        )
                        if verbose and completed % 100 == 0:
                            print(f"[{completed}/{total}] {link.name} {activation} "
                                  f"d={depth} w={width} s={seed} "
                                  f"eval={result.final_eval_accuracy:.4f}", flush=True)
                        if completed % 200 == 0 or completed == total:
                            _write(pd.DataFrame(rows), output_directory)
    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "parametrization_sweep"
    with artifact_lock(stem, "parametrization sweep"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = ParametrizationSweepConfig()
    print(f"parametrization sweep: {config.total_runs()} runs", flush=True)
    run(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
