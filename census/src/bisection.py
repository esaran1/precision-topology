"""A2: bisect the two transition intervals, with near-threshold probes.

Family A over (1.00, 1.10], family B over [-0.25, 0), 40 runs per value
(4 depths x 10 seeds, matching the pooled structure of the original 80-run
cells), dense verification of every eval-0 run built in, and the A5
confound fields (final loss, gradient norm, inactive-unit fraction)
recorded per run.
"""

from __future__ import annotations

from pathlib import Path
import time
import zlib

import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .data import linked_tori
from .threshold_sweep import final_gradient_norm, inactive_unit_fraction
from .train import TrainingConfig, train_mlp


FAMILY_A_VALUES = (1.001, 1.005, 1.01, 1.02, 1.04, 1.06, 1.07, 1.08, 1.09, 1.10)
FAMILY_B_VALUES = (-0.005, -0.02, -0.05, -0.11, -0.13, -0.16, -0.20, -0.25)
DEPTHS = (3, 5, 8, 12)
SEEDS = tuple(range(10))
DENSE_PER_CLASS = 50_000


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    conditions = [("sin_family", value) for value in FAMILY_A_VALUES]
    conditions += [("pwl_family", value) for value in FAMILY_B_VALUES]
    rows: list[dict] = []
    total = len(conditions) * len(DEPTHS) * len(SEEDS)
    completed = 0
    for activation, parameter in conditions:
        for depth in DEPTHS:
            for seed in SEEDS:
                started = time.monotonic()
                train_data, eval_data, *_ = _make_data("linked_tori", seed, config)
                result = train_mlp(
                    train_data, eval_data, hidden_depth=depth, hidden_width=3,
                    activation=activation,  # type: ignore[arg-type]
                    config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
                    activation_parameter=parameter,
                )
                errors = int(round((1 - result.final_eval_accuracy) * 2_000))
                dense_err = None
                if errors == 0:
                    tag = f"bisect|{activation}|{parameter}|{depth}|{seed}"
                    dense_seed = 940_000 + zlib.crc32(tag.encode()) % 50_000
                    dense = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=dense_seed)
                    features = torch.as_tensor(dense.features, dtype=torch.float32)
                    labels = torch.as_tensor(dense.labels, dtype=torch.int64)
                    with torch.no_grad():
                        dense_err = int(
                            (result.model(features).argmax(1) != labels).sum().item()
                        )
                train_features = torch.as_tensor(train_data.features, dtype=torch.float32)
                train_labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
                completed += 1
                rows.append({
                    "activation": activation,
                    "parameter": parameter,
                    "depth": depth,
                    "seed": seed,
                    "eval_errors": errors,
                    "dense_errors": dense_err,
                    "regionally_separating": errors == 0 and dense_err == 0,
                    "final_train_loss": result.checkpoints[-1].train_loss,
                    "final_gradient_norm": final_gradient_norm(
                        result.model, train_features, train_labels
                    ),
                    "inactive_unit_fraction": inactive_unit_fraction(result),
                    "duration_seconds": time.monotonic() - started,
                })
                if verbose and completed % 80 == 0:
                    print(f"[{completed}/{total}]", flush=True)
                if completed % 80 == 0 or completed == total:
                    _write(pd.DataFrame(rows), output_directory)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    stem = output_directory / "bisection"
    with artifact_lock(stem, "transition bisection"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    print(f"bisection: {(len(FAMILY_A_VALUES)+len(FAMILY_B_VALUES))*len(DEPTHS)*len(SEEDS)} runs", flush=True)
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
