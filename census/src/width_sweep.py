"""Half A of the d=3 width sweep: accuracy and pass rate versus width.

This is the first width evidence at ``d = 3`` in this project or in Ren and
Lim, whose ``R^3`` experiments fix width at 3 and vary depth instead.  It is
not a replication, and nothing in that paper predicts its outcome at widths 4
through 15.

No run is excluded.  Pass rate is a primary outcome here rather than a filter,
so failed runs are recorded alongside successful ones, and accuracy is reported
as a distribution rather than a mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from .census import _make_data, SweepConfig
from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .train import TrainingConfig, train_mlp


# Chance accuracy for the balanced two-class evaluation set.
CHANCE_ACCURACY = 0.5


@dataclass(frozen=True)
class WidthSweepConfig:
    """Grid for Half A, as approved in notes/width_sweep_design.md."""

    # 3 is the theorem-obstructed width; 4 is theoretically sufficient; 5-8 is
    # the additive transition region; 10, 12, 15 test whether anything is still
    # changing out to 5d, which is where the two accounts disagree.
    widths: Sequence[int] = (3, 4, 5, 6, 7, 8, 10, 12, 15)
    # Overlaps the paper's Table 2 grid so the depth trend is comparable.
    depths: Sequence[int] = (3, 5, 8, 12)
    # The three monotonic activations are a robustness check within one
    # expressivity class; GELU is the only one that can bear on the ordering.
    activations: Sequence[str] = ("tanh", "relu", "leaky_relu", "gelu")
    seeds: Sequence[int] = tuple(range(20))
    dataset: str = "linked_tori"
    n_train_per_class: int = 1_000
    n_eval_per_class: int = 1_000
    max_steps: int = 2_000
    learning_rate: float = 1e-2
    tube_radius: float = 0.2

    def total_runs(self) -> int:
        return (
            len(self.widths)
            * len(self.depths)
            * len(self.activations)
            * len(self.seeds)
        )

    def as_census_config(self) -> SweepConfig:
        """Data construction settings, shared with the original census."""

        return SweepConfig(
            n_train_per_class=self.n_train_per_class,
            n_eval_per_class=self.n_eval_per_class,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            tube_radius=self.tube_radius,
        )


def run_width_sweep(
    config: WidthSweepConfig,
    output_directory: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every cell of the grid, recording all runs including failures."""

    data_config = config.as_census_config()
    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0

    for activation in config.activations:
        for depth in config.depths:
            for width in config.widths:
                for seed in config.seeds:
                    started = time.monotonic()
                    train_data, eval_data, train_seed, eval_seed = _make_data(
                        config.dataset, seed, data_config
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
                    rows.append(
                        {
                            "dataset": config.dataset,
                            "depth": depth,
                            "width": width,
                            "activation": activation,
                            "monotonic": activation != "gelu",
                            "seed": seed,
                            "train_data_seed": train_seed,
                            "eval_data_seed": eval_seed,
                            "n_train_per_class": config.n_train_per_class,
                            "n_eval_per_class": config.n_eval_per_class,
                            "tube_radius": config.tube_radius,
                            "passed": result.passed,
                            "failure_reason": result.failure_reason,
                            "final_train_accuracy": result.final_train_accuracy,
                            "final_eval_accuracy": result.final_eval_accuracy,
                            "at_chance": result.final_eval_accuracy <= CHANCE_ACCURACY,
                            "perfect_eval": result.final_eval_accuracy >= 1.0,
                            "max_steps": config.max_steps,
                            "learning_rate": config.learning_rate,
                            "train_accuracy_required": training_config.train_accuracy_required,
                            "eval_accuracy_required": training_config.eval_accuracy_required,
                            "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                            "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                            "duration_seconds": duration,
                        }
                    )
                    if verbose and completed % 25 == 0:
                        print(
                            f"[{completed}/{total}] {activation} depth={depth} "
                            f"width={width} seed={seed} passed={result.passed} "
                            f"eval={result.final_eval_accuracy:.4f}",
                            flush=True,
                        )
                    # Persist after every cell so a long run is resumable and
                    # partial results are never lost.
                    if completed % 100 == 0 or completed == total:
                        _write(pd.DataFrame(rows), output_directory)

    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "width_sweep"
    csv_temp = stem.with_suffix(".csv.tmp")
    parquet_temp = stem.with_suffix(".parquet.tmp")
    frame.to_csv(csv_temp, index=False)
    frame.to_parquet(parquet_temp, index=False)
    csv_temp.replace(stem.with_suffix(".csv"))
    parquet_temp.replace(stem.with_suffix(".parquet"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = WidthSweepConfig()
    print(f"width sweep Half A: {config.total_runs()} runs", flush=True)
    run_width_sweep(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
