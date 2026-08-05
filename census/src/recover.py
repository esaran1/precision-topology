"""Deterministic reconstruction of already-measured census runs.

No new experimental condition is introduced here.  Every run reconstructed by
this module is one that ``census.run_sweep`` already executed and recorded; the
seeds, data construction, and training configuration are taken from the saved
status table so the reconstruction is bit-for-bit the same computation.  The
recovery is only trustworthy if it reproduces the recorded measurements
exactly, so :func:`verify_recovery` re-derives the recorded accuracy and
saturation values and refuses to return a divergent run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import torch

from .census import RunMetadata, SweepConfig, _make_data
from .models import MLP
from .precision import DELTA_TABLE, saturation_metrics
from .train import TrainingConfig, TrainingResult, train_mlp


# Recorded accuracies are float64 round-trips of exact rational counts and
# saturation fractions are means of exact boolean counts, so an exactly
# reproduced run matches to the last bit.  Any nonzero difference means the
# reconstruction is not the recorded computation.
EXACT = 0.0


@dataclass(frozen=True)
class RecoveredRun:
    """One reconstructed run together with the status row it reproduces."""

    metadata: RunMetadata
    result: TrainingResult
    model: MLP
    eval_features: torch.Tensor
    eval_labels: torch.Tensor
    recorded_train_accuracy: float
    recorded_eval_accuracy: float

    @property
    def key(self) -> tuple[str, int, int, str, int]:
        return (
            self.metadata.dataset,
            self.metadata.depth,
            self.metadata.width,
            self.metadata.activation,
            self.metadata.seed,
        )


class RecoveryDivergence(RuntimeError):
    """Raised when a reconstructed run does not reproduce its recorded values."""


def accepted_runs(
    status: pd.DataFrame,
    dataset: str | None = None,
    activation: str | None = None,
    depth: int | None = None,
    widths: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Select accepted status rows matching the requested subset."""

    selected = status[status.passed]
    if dataset is not None:
        selected = selected[selected.dataset == dataset]
    if activation is not None:
        selected = selected[selected.activation == activation]
    if depth is not None:
        selected = selected[selected.depth == depth]
    if widths is not None:
        selected = selected[selected.width.isin(list(widths))]
    return selected.sort_values(["depth", "width", "seed"]).reset_index(drop=True)


def reconstruct_run(row: pd.Series, config: SweepConfig | None = None) -> RecoveredRun:
    """Re-run one recorded configuration under its recorded seeds."""

    config = config or SweepConfig()
    dataset = str(row.dataset)
    depth = int(row.depth)
    width = int(row.width)
    activation = str(row.activation)
    seed = int(row.seed)

    train_data, eval_data, train_seed, eval_seed = _make_data(dataset, seed, config)
    # The saved status row carries the seeds actually used; if the current
    # construction disagrees the reconstruction is not the recorded run.
    if int(row.train_data_seed) != train_seed or int(row.eval_data_seed) != eval_seed:
        raise RecoveryDivergence(
            f"data seeds differ for {dataset} depth={depth} width={width} seed={seed}: "
            f"recorded ({int(row.train_data_seed)}, {int(row.eval_data_seed)}), "
            f"reconstructed ({train_seed}, {eval_seed})"
        )

    training_config = TrainingConfig(
        seed=seed,
        max_steps=int(row.max_steps),
        learning_rate=float(row.learning_rate),
        train_accuracy_required=float(row.train_accuracy_required),
        eval_accuracy_required=float(row.eval_accuracy_required),
    )
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=depth,
        hidden_width=width,
        activation=activation,  # type: ignore[arg-type]
        config=training_config,
    )
    metadata = RunMetadata(
        dataset=dataset,
        depth=depth,
        width=width,
        activation=activation,
        seed=seed,
        train_data_seed=train_seed,
        eval_data_seed=eval_seed,
        n_train_per_class=int(row.n_train_per_class),
        n_eval_per_class=int(row.n_eval_per_class),
        tube_radius=config.tube_radius if dataset == "linked_tori" else None,
    )
    return RecoveredRun(
        metadata=metadata,
        result=result,
        model=result.model,
        eval_features=torch.as_tensor(eval_data.features, dtype=torch.float32, device="cpu"),
        eval_labels=torch.as_tensor(eval_data.labels, dtype=torch.int64, device="cpu"),
        recorded_train_accuracy=float(row.final_train_accuracy),
        recorded_eval_accuracy=float(row.final_eval_accuracy),
    )


def verify_recovery(run: RecoveredRun, saturation: pd.DataFrame) -> dict[str, float]:
    """Confirm a reconstruction reproduces recorded accuracy and saturation exactly.

    Raises :class:`RecoveryDivergence` on any difference.  Returns the observed
    maximum absolute deviations so callers can record that they were zero.
    """

    result = run.result
    if result.passed != True:  # noqa: E712 - explicit: recorded rows are accepted runs
        raise RecoveryDivergence(f"{run.key}: reconstruction did not pass the training gate")

    train_delta = abs(result.final_train_accuracy - run.recorded_train_accuracy)
    eval_delta = abs(result.final_eval_accuracy - run.recorded_eval_accuracy)
    if train_delta > EXACT or eval_delta > EXACT:
        raise RecoveryDivergence(
            f"{run.key}: accuracy diverged (train {train_delta:.3e}, eval {eval_delta:.3e})"
        )

    dataset, depth, width, activation, seed = run.key
    recorded = saturation[
        (saturation.dataset == dataset)
        & (saturation.depth == depth)
        & (saturation.width == width)
        & (saturation.activation == activation)
        & (saturation.seed == seed)
    ]
    if recorded.empty:
        raise RecoveryDivergence(f"{run.key}: no recorded saturation rows to verify against")

    preactivations = run.result.checkpoints[-1].eval_preactivations
    worst = 0.0
    compared = 0
    by_format = {spec.format: spec for spec in DELTA_TABLE}
    for _, recorded_row in recorded.iterrows():
        layer_index = int(recorded_row.layer)
        spec = by_format[str(recorded_row.format)]
        # Duplicate format names differ only by convention/delta; match on both.
        if float(recorded_row.delta_value) != spec.delta:
            spec = next(
                candidate
                for candidate in DELTA_TABLE
                if candidate.format == str(recorded_row.format)
                and candidate.delta == float(recorded_row.delta_value)
            )
        values = preactivations[layer_index - 1]
        for criterion in ("paper", "exact"):
            metrics = saturation_metrics(values, spec.delta, criterion)  # type: ignore[arg-type]
            recomputed = metrics["total_saturation_fraction"]
            reference = float(recorded_row[f"{criterion}_total_saturation_fraction"])
            deviation = abs(recomputed - reference)
            worst = max(worst, deviation)
            compared += 1
            if deviation > EXACT:
                raise RecoveryDivergence(
                    f"{run.key}: layer {layer_index} {spec.format} {criterion} saturation "
                    f"diverged by {deviation:.3e} "
                    f"(recorded {reference!r}, reconstructed {recomputed!r})"
                )

    return {
        "train_accuracy_deviation": train_delta,
        "eval_accuracy_deviation": eval_delta,
        "max_saturation_deviation": worst,
        "saturation_values_compared": float(compared),
    }


def recover_subset(
    status: pd.DataFrame,
    saturation: pd.DataFrame,
    dataset: str | None = None,
    activation: str | None = None,
    depth: int | None = None,
    widths: Sequence[int] | None = None,
    config: SweepConfig | None = None,
    verbose: bool = False,
) -> Iterator[tuple[RecoveredRun, dict[str, float]]]:
    """Reconstruct and verify each accepted run in the requested subset."""

    rows = accepted_runs(status, dataset, activation, depth, widths)
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        run = reconstruct_run(row, config)
        deviations = verify_recovery(run, saturation)
        if verbose:
            print(
                f"[{position}/{len(rows)}] recovered {run.key} "
                f"max_saturation_deviation={deviations['max_saturation_deviation']:.1e} "
                f"values_compared={int(deviations['saturation_values_compared'])}",
                flush=True,
            )
        yield run, deviations


def load_results(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(directory / "training_status.parquet"),
        pd.read_parquet(directory / "saturation.parquet"),
    )
