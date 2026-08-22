"""Width sweep under the author's protocol, for comparison with ours.

The parametrization is held fixed at our baseline linked tori so that
*protocol* is the only variable between this sweep and ``width_sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from .author_protocol import UNSPECIFIED, AuthorTrainingConfig, train_mlp_author
from .census import _make_data
from .models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from .width_sweep import CHANCE_ACCURACY, WidthSweepConfig


@dataclass(frozen=True)
class ProtocolSweepConfig:
    widths: Sequence[int] = (3, 4, 5, 6)
    depths: Sequence[int] = (3, 5, 8, 12)
    activations: Sequence[str] = ("tanh", "relu", "leaky_relu", "gelu")
    seeds: Sequence[int] = tuple(range(30))
    dataset: str = "linked_tori"

    def total_runs(self) -> int:
        return (
            len(self.widths) * len(self.depths) * len(self.activations) * len(self.seeds)
        )


def run(
    config: ProtocolSweepConfig, output_directory: Path, verbose: bool = True
) -> pd.DataFrame:
    data_config = WidthSweepConfig().as_census_config()
    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0

    for activation in config.activations:
        for depth in config.depths:
            for width in config.widths:
                for seed in config.seeds:
                    train_data, eval_data, train_seed, eval_seed = _make_data(
                        config.dataset, seed, data_config
                    )
                    started = time.monotonic()
                    training_config = AuthorTrainingConfig(seed=seed)
                    result = train_mlp_author(
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
                            "protocol": "author",
                            "dataset": config.dataset,
                            "depth": depth,
                            "width": width,
                            "activation": activation,
                            "monotonic": activation != "gelu",
                            "seed": seed,
                            "train_data_seed": train_seed,
                            "eval_data_seed": eval_seed,
                            "final_train_accuracy": result.final_train_accuracy,
                            "final_eval_accuracy": result.final_eval_accuracy,
                            "perfect_eval": result.final_eval_accuracy >= 1.0,
                            "at_chance": result.final_eval_accuracy <= CHANCE_ACCURACY,
                            "best_epoch": result.best_epoch,
                            "epochs_run": result.epochs_run,
                            "stopped_early": result.stopped_early,
                            "learning_rate": training_config.learning_rate,
                            "batch_size": training_config.batch_size,
                            "max_epochs": training_config.max_epochs,
                            "patience": training_config.patience,
                            "initialization_scheme": DEFAULT_INITIALIZATION_SCHEME,
                            "initialization_gain": DEFAULT_INITIALIZATION_GAIN,
                            "duration_seconds": duration,
                        }
                    )
                    if verbose and completed % 100 == 0:
                        print(
                            f"[{completed}/{total}] {activation} d={depth} w={width} "
                            f"s={seed} eval={result.final_eval_accuracy:.4f}",
                            flush=True,
                        )
                    if completed % 200 == 0 or completed == total:
                        _write(pd.DataFrame(rows), output_directory)
    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "protocol_sweep"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = ProtocolSweepConfig()
    print(f"author-protocol sweep: {config.total_runs()} runs", flush=True)
    for name, choice in UNSPECIFIED:
        print(f"  unspecified: {name} -> {choice}", flush=True)
    run(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
