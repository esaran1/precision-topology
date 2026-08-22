"""Train across corrugated parametrizations, both G.1 readings.

Reading A relaxes embeddedness of the swept solid to a reported diagnostic;
Reading B gates it hard.  Core linking is gated hard under both, since the
linking number is defined on the cores and that is what the fold measurement
uses.

Data seeds are offset again so a corrugation run cannot be confused with a
baseline or parametrization run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from .artifact_lock import artifact_lock
from .corrugation import GRID, CorrugatedLink, sample, validate
from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .train import TrainingConfig, train_mlp


TRAIN_SEED_BASE = 50_000
EVAL_SEED_BASE = 60_000
CHANCE_ACCURACY = 0.5


@dataclass(frozen=True)
class CorrugationSweepConfig:
    widths: Sequence[int] = (3, 4)
    depths: Sequence[int] = (3, 5, 8, 12)
    activations: Sequence[str] = ("tanh", "relu", "leaky_relu", "gelu")
    seeds: Sequence[int] = tuple(range(10))
    links: Sequence[CorrugatedLink] = GRID
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


def validate_grid(links: Sequence[CorrugatedLink]) -> pd.DataFrame:
    """Validate every configuration; core linking is a hard gate for both readings."""

    rows = []
    for link in links:
        report = validate(link, n_points=4096)
        embedded = not report.self_intersects and report.tube_gap > 0.0
        rows.append(
            {
                "name": report.name,
                "reading": link.reading,
                "amplitude": link.amplitude,
                "frequency": link.frequency,
                "noise": link.noise,
                "tube_radius": link.base.tube_radius_a,
                "linking_number": report.linking_number,
                "linking_residual": report.linking_residual,
                "core_separation": report.core_separation,
                "tube_gap": report.tube_gap,
                "self_intersects": report.self_intersects,
                "embedded": embedded,
                "min_between_class_distance": report.min_between_class_distance,
                "between_class_collisions": report.between_class_collisions,
            }
        )
    frame = pd.DataFrame(rows)

    # Hard gate for both readings: cores must be a genuine link.
    bad_link = frame[
        frame.linking_number.isna() | (frame.linking_number.abs() != 1)
    ]
    if not bad_link.empty:
        raise ValueError(
            "core linking is not +/-1 for: " + ", ".join(bad_link.name)
        )
    # Hard gate for Reading B only: the swept solid must be embedded.
    bad_embed = frame[(frame.reading == "offset") & (~frame.embedded)]
    if not bad_embed.empty:
        raise ValueError(
            "Reading B requires embedded configurations; failed: "
            + ", ".join(bad_embed.name)
        )
    # Between-class collisions are never acceptable under either reading.
    bad_collide = frame[frame.between_class_collisions.fillna(0) > 0]
    if not bad_collide.empty:
        raise ValueError(
            "between-class collisions present in: " + ", ".join(bad_collide.name)
        )
    return frame


def run(
    config: CorrugationSweepConfig, output_directory: Path, verbose: bool = True
) -> pd.DataFrame:
    validation = validate_grid(config.links)
    validation.to_csv(output_directory / "corrugation_validation.csv", index=False)

    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0
    for link in config.links:
        report = validation[validation.name == link.name].iloc[0]
        for activation in config.activations:
            for depth in config.depths:
                for width in config.widths:
                    for seed in config.seeds:
                        train_seed = TRAIN_SEED_BASE + seed
                        eval_seed = EVAL_SEED_BASE + seed
                        train_data = sample(link, config.n_per_class, train_seed)
                        eval_data = sample(link, config.n_per_class, eval_seed)
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
                                "configuration": link.name,
                                "reading": link.reading,
                                "amplitude": link.amplitude,
                                "frequency": link.frequency,
                                "noise": link.noise,
                                "embedded": bool(report.embedded),
                                "linking_number": int(report.linking_number),
                                "min_between_class_distance": report.min_between_class_distance,
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
                                "eval_errors": int(
                                    round((1.0 - result.final_eval_accuracy) * 2 * config.n_per_class)
                                ),
                                "max_steps": config.max_steps,
                                "learning_rate": config.learning_rate,
                                "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                                "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                                "duration_seconds": duration,
                            }
                        )
                        if verbose and completed % 200 == 0:
                            print(
                                f"[{completed}/{total}] {link.name} {activation} "
                                f"d={depth} w={width} s={seed} "
                                f"eval={result.final_eval_accuracy:.4f}",
                                flush=True,
                            )
                        if completed % 400 == 0 or completed == total:
                            _write(pd.DataFrame(rows), output_directory)
    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "corrugation_sweep"
    with artifact_lock(stem, "corrugation sweep"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = CorrugationSweepConfig()
    print(f"corrugation sweep: {config.total_runs()} runs", flush=True)
    run(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
