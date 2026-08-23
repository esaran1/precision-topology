"""Search 1 of Part 2b: mass random restarts at the best monotonic configs.

200 seeds (10x the standard 20) at the two best-performing monotonic width-3
configurations.  Seeds 0-19 duplicate the threshold sweep's runs by design:
they land on identical results (same deterministic pipeline), which is a
recovery check, and the remaining 180 are new attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .train import TrainingConfig, train_mlp


@dataclass(frozen=True)
class RestartTarget:
    activation: str
    parameter: float | None
    depth: int


TARGETS: tuple[RestartTarget, ...] = (
    RestartTarget("sin_family", 0.95, 8),  # best observed monotonic: 2 errors
    RestartTarget("tanh", None, 8),        # best fixed monotonic: 26 errors
)
WIDTH = 3
SEEDS: Sequence[int] = tuple(range(200))


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    data_config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    rows: list[dict[str, object]] = []
    total = len(TARGETS) * len(SEEDS)
    completed = 0
    for target in TARGETS:
        for seed in SEEDS:
            started = time.monotonic()
            train_data, eval_data, train_seed, eval_seed = _make_data(
                "linked_tori", seed, data_config
            )
            result = train_mlp(
                train_data,
                eval_data,
                hidden_depth=target.depth,
                hidden_width=WIDTH,
                activation=target.activation,  # type: ignore[arg-type]
                config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
                activation_parameter=target.parameter,
            )
            completed += 1
            rows.append(
                {
                    "activation": target.activation,
                    "parameter": target.parameter,
                    "depth": target.depth,
                    "width": WIDTH,
                    "seed": seed,
                    "train_data_seed": train_seed,
                    "eval_data_seed": eval_seed,
                    "final_eval_accuracy": result.final_eval_accuracy,
                    "perfect_eval": result.final_eval_accuracy >= 1.0,
                    "eval_errors": int(round((1.0 - result.final_eval_accuracy) * 2_000)),
                    "duration_seconds": time.monotonic() - started,
                }
            )
            if verbose and completed % 50 == 0:
                print(f"[{completed}/{total}]", flush=True)
            if completed % 100 == 0 or completed == total:
                _write(pd.DataFrame(rows), output_directory)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "search_restarts"
    with artifact_lock(stem, "mass restarts"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    print(f"mass restarts: {len(TARGETS) * len(SEEDS)} runs", flush=True)
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
