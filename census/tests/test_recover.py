"""Tests for deterministic reconstruction of recorded runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.recover import (
    RecoveryDivergence,
    accepted_runs,
    load_results,
    reconstruct_run,
    verify_recovery,
)


RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def recorded() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_results(RESULTS)


def test_accepted_runs_filters_to_passing_subset(recorded):
    status, _ = recorded
    rows = accepted_runs(status, "linked_tori", "tanh", 6)
    assert len(rows) > 0
    assert rows.passed.all()
    assert set(rows.dataset) == {"linked_tori"}
    assert set(rows.activation) == {"tanh"}
    assert set(rows.depth) == {6}


def test_reconstruction_reproduces_recorded_values_exactly(recorded):
    """The recovery is only usable if it is bit-for-bit the recorded run."""

    status, saturation = recorded
    row = accepted_runs(status, "linked_tori", "tanh", 4).iloc[0]
    run = reconstruct_run(row)
    deviations = verify_recovery(run, saturation)
    assert deviations["train_accuracy_deviation"] == 0.0
    assert deviations["eval_accuracy_deviation"] == 0.0
    assert deviations["max_saturation_deviation"] == 0.0
    assert deviations["saturation_values_compared"] > 0


def test_reconstruction_is_repeatable(recorded):
    status, _ = recorded
    row = accepted_runs(status, "linked_tori", "tanh", 4).iloc[0]
    first = reconstruct_run(row)
    second = reconstruct_run(row)
    assert first.result.final_eval_accuracy == second.result.final_eval_accuracy
    for left, right in zip(
        first.result.checkpoints[-1].eval_preactivations,
        second.result.checkpoints[-1].eval_preactivations,
    ):
        assert (left == right).all()


def test_verify_recovery_rejects_mismatched_accuracy(recorded):
    """A divergent reconstruction must abort rather than be silently accepted."""

    status, saturation = recorded
    row = accepted_runs(status, "linked_tori", "tanh", 4).iloc[0]
    run = reconstruct_run(row)
    tampered = type(run)(
        metadata=run.metadata,
        result=run.result,
        model=run.model,
        eval_features=run.eval_features,
        eval_labels=run.eval_labels,
        recorded_train_accuracy=run.recorded_train_accuracy,
        recorded_eval_accuracy=run.recorded_eval_accuracy - 0.01,
    )
    with pytest.raises(RecoveryDivergence):
        verify_recovery(tampered, saturation)


def test_verify_recovery_requires_recorded_rows(recorded):
    status, saturation = recorded
    row = accepted_runs(status, "linked_tori", "tanh", 4).iloc[0]
    run = reconstruct_run(row)
    with pytest.raises(RecoveryDivergence):
        verify_recovery(run, saturation.iloc[0:0])
