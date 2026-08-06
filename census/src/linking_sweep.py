"""Half B: layerwise linking traces through trained width-sweep networks.

Networks are reconstructed from the recorded width-sweep seeds rather than
reloaded, since the sweep persisted measurements and not weights.  Training is
deterministic, so a reconstruction reproduces the recorded run exactly; the
accuracy check below refuses any run that does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import torch

from .census import _make_data
from .linking_trace import trace_linking, trace_to_records
from .train import TrainingConfig, train_mlp
from .width_sweep import WidthSweepConfig


class ReconstructionDivergence(RuntimeError):
    """Raised when a rebuilt network does not reproduce its recorded accuracy."""


def reconstruct(row: pd.Series, config: WidthSweepConfig | None = None):
    """Retrain one recorded width-sweep run under its recorded seeds."""

    config = config or WidthSweepConfig()
    data_config = config.as_census_config()
    train_data, eval_data, train_seed, eval_seed = _make_data(
        str(row.dataset), int(row.seed), data_config
    )
    if int(row.train_data_seed) != train_seed or int(row.eval_data_seed) != eval_seed:
        raise ReconstructionDivergence(
            f"data seeds differ for {tuple(row[['activation','depth','width','seed']])}"
        )
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=int(row.depth),
        hidden_width=int(row.width),
        activation=str(row.activation),  # type: ignore[arg-type]
        config=TrainingConfig(
            seed=int(row.seed),
            max_steps=int(row.max_steps),
            learning_rate=float(row.learning_rate),
        ),
    )
    recorded_eval = float(row.final_eval_accuracy)
    recorded_train = float(row.final_train_accuracy)
    if (
        result.final_eval_accuracy != recorded_eval
        or result.final_train_accuracy != recorded_train
    ):
        raise ReconstructionDivergence(
            f"accuracy differs for {tuple(row[['activation','depth','width','seed']])}: "
            f"recorded train={recorded_train} eval={recorded_eval}, "
            f"rebuilt train={result.final_train_accuracy} eval={result.final_eval_accuracy}"
        )
    return result


def trace_rows(
    rows: pd.DataFrame,
    n_core_points: int = 512,
    config: WidthSweepConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reconstruct and trace every run in ``rows``."""

    records: list[dict[str, object]] = []
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        result = reconstruct(row, config)
        trace = trace_linking(result.model, n_core_points=n_core_points)
        records.extend(
            trace_to_records(
                trace,
                dataset=str(row.dataset),
                activation=str(row.activation),
                monotonic=bool(row.monotonic),
                depth=int(row.depth),
                width=int(row.width),
                seed=int(row.seed),
                final_eval_accuracy=float(row.final_eval_accuracy),
                passed=bool(row.passed),
                perfect_eval=bool(row.perfect_eval),
                at_chance=bool(row.at_chance),
            )
        )
        if verbose:
            print(
                f"[{position}/{len(rows)}] {row.activation} depth={row.depth} "
                f"width={row.width} seed={row.seed} eval={row.final_eval_accuracy:.4f}",
                flush=True,
            )
    return pd.DataFrame(records)


def select(
    sweep: pd.DataFrame,
    width: int | None = None,
    activation: str | None = None,
    perfect_only: bool = False,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    selected = sweep
    if width is not None:
        selected = selected[selected.width == width]
    if activation is not None:
        selected = selected[selected.activation == activation]
    if perfect_only:
        selected = selected[selected.perfect_eval]
    if seeds is not None:
        selected = selected[selected.seed.isin(list(seeds))]
    return selected.sort_values(["activation", "depth", "seed"]).reset_index(drop=True)


def load_sweep(results_directory: Path) -> pd.DataFrame:
    return pd.read_parquet(results_directory / "width_sweep.parquet")


def write(frame: pd.DataFrame, results_directory: Path, stem: str) -> None:
    target = results_directory / stem
    frame.to_csv(target.with_suffix(".csv"), index=False)
    frame.to_parquet(target.with_suffix(".parquet"), index=False)
