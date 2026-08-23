"""Part 4 sweep: separation rate as a function of (|lk|, width).

Predictions in ``results/winding_prediction.md``.  Geometry is validated
before training and the run aborts on any invalid configuration.  Every
separating run is dense-verified on 100,000 fresh points immediately, per
the Part 2c protocol; ``regionally_separating`` is the primary outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import zlib
from typing import Sequence

import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .train import TrainingConfig, train_mlp
from .winding import GRID, WindingLink, sample, validate


TRAIN_SEED_BASE = 70_000
EVAL_SEED_BASE = 80_000
DENSE_SEED_BASE = 910_000
DENSE_PER_CLASS = 50_000


@dataclass(frozen=True)
class WindingSweepConfig:
    widths: Sequence[int] = (3, 4, 5, 6)
    depth: int = 5
    activations: Sequence[tuple[str, float | None]] = (
        ("tanh", None),
        ("relu", None),
        ("leaky_relu", None),
        ("gelu", None),
        ("sin_family", 2.0),
    )
    seeds: Sequence[int] = tuple(range(20))
    links: Sequence[WindingLink] = GRID
    n_per_class: int = 1_000
    max_steps: int = 2_000
    learning_rate: float = 1e-2

    def total_runs(self) -> int:
        return (
            len(self.links) * len(self.widths) * len(self.activations) * len(self.seeds)
        )


def run(config: WindingSweepConfig, output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    validations = []
    for link in config.links:
        report = validate(link)
        if not report.valid:
            raise ValueError(f"{link.name} failed validation: {report}")
        validations.append(report.__dict__)
    pd.DataFrame(validations).to_csv(
        output_directory / "winding_validation.csv", index=False
    )

    rows: list[dict[str, object]] = []
    total = config.total_runs()
    completed = 0
    for link in config.links:
        for activation, parameter in config.activations:
            for width in config.widths:
                for seed in config.seeds:
                    started = time.monotonic()
                    train_data = sample(link, config.n_per_class, TRAIN_SEED_BASE + seed)
                    eval_data = sample(link, config.n_per_class, EVAL_SEED_BASE + seed)
                    result = train_mlp(
                        train_data,
                        eval_data,
                        hidden_depth=config.depth,
                        hidden_width=width,
                        activation=activation,  # type: ignore[arg-type]
                        config=TrainingConfig(
                            seed=seed,
                            max_steps=config.max_steps,
                            learning_rate=config.learning_rate,
                        ),
                        activation_parameter=parameter,
                    )
                    perfect = result.final_eval_accuracy >= 1.0
                    dense_errors: int | None = None
                    dense_margin: float | None = None
                    if perfect:
                        key = f"{link.name}|{activation}|{parameter}|{width}|{seed}"
                        dense_seed = DENSE_SEED_BASE + zlib.crc32(key.encode()) % 50_000
                        dense = sample(link, DENSE_PER_CLASS, dense_seed)
                        features = torch.as_tensor(dense.features, dtype=torch.float32)
                        labels = torch.as_tensor(dense.labels, dtype=torch.int64)
                        with torch.no_grad():
                            logits = result.model(features)
                            correct = logits.gather(1, labels[:, None]).squeeze(1)
                            wrong = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
                            margin = correct - wrong
                        dense_errors = int((margin <= 0.0).sum().item())
                        dense_margin = float(margin.min().item())
                    completed += 1
                    rows.append(
                        {
                            "configuration": link.name,
                            "q": link.q,
                            "activation": activation,
                            "parameter": parameter,
                            "monotonic": activation != "gelu"
                            and not (activation == "sin_family" and (parameter or 0) > 1),
                            "depth": config.depth,
                            "width": width,
                            "seed": seed,
                            "final_eval_accuracy": result.final_eval_accuracy,
                            "perfect_eval": perfect,
                            "eval_errors": int(
                                round((1.0 - result.final_eval_accuracy) * 2 * config.n_per_class)
                            ),
                            "dense_errors": dense_errors,
                            "dense_min_margin": dense_margin,
                            "regionally_separating": perfect and dense_errors == 0,
                            "duration_seconds": time.monotonic() - started,
                        }
                    )
                    if verbose and completed % 100 == 0:
                        print(f"[{completed}/{total}]", flush=True)
                    if completed % 200 == 0 or completed == total:
                        _write(pd.DataFrame(rows), output_directory)
    frame = pd.DataFrame(rows)
    _write(frame, output_directory)
    return frame


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "winding_sweep"
    with artifact_lock(stem, "winding sweep"):
        csv_temp = stem.with_suffix(".csv.tmp")
        parquet_temp = stem.with_suffix(".parquet.tmp")
        frame.to_csv(csv_temp, index=False)
        frame.to_parquet(parquet_temp, index=False)
        csv_temp.replace(stem.with_suffix(".csv"))
        parquet_temp.replace(stem.with_suffix(".parquet"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    config = WindingSweepConfig()
    print(f"winding sweep: {config.total_runs()} runs", flush=True)
    run(config, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
