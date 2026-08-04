import numpy as np
import pytest

from src.census import RunMetadata, measure_checkpoint
from src.data import gaussian_blobs
from src.models import DEFAULT_INITIALIZATION_GAIN, DEFAULT_INITIALIZATION_SCHEME
from src.train import TrainingConfig, train_mlp


@pytest.fixture(scope="module")
def measured():
    train_data = gaussian_blobs(24, seed=31)
    eval_data = gaussian_blobs(24, seed=32)
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=2,
        hidden_width=5,
        activation="tanh",
        config=TrainingConfig(seed=4, max_steps=100),
    )
    assert result.passed
    metadata = RunMetadata("blobs", 2, 5, "tanh", 4, 31, 32, 24, 24, None)
    return result, metadata


def test_census_rows_include_layer_position_and_initialization_metadata(measured):
    result, metadata = measured
    frame = measure_checkpoint(result, result.checkpoints[-1], result.checkpoints[0], metadata)
    assert len(frame) == 2 * 9
    assert set(frame["distance_from_output"]) == {0, 1}
    assert set(frame["relative_layer_position"]) == {0.5, 1.0}
    assert set(frame["initialization_scheme"]) == {DEFAULT_INITIALIZATION_SCHEME}
    assert set(frame["initialization_gain"]) == {DEFAULT_INITIALIZATION_GAIN}


def test_vector_collision_is_headline_and_paper_half_is_null(measured):
    result, metadata = measured
    frame = measure_checkpoint(result, result.checkpoints[-1], result.checkpoints[0], metadata)
    assert "vector_collision_rate" in frame
    assert "excess_vector_collision_rate" in frame
    paper_half = frame[(frame["format"] == "half") & (frame["convention"] == "paper")]
    assert paper_half["vector_collision_rate"].isna().all()
    assert paper_half["excess_vector_collision_rate"].isna().all()


def test_initialization_is_its_own_zero_excess_baseline(measured):
    result, metadata = measured
    initial = measure_checkpoint(result, result.checkpoints[0], result.checkpoints[0], metadata)
    available = initial[initial["quantizer"].notna()]
    assert np.array_equal(
        available["excess_per_unit_collision_mean"].to_numpy(),
        np.zeros(len(available)),
    )
    assert (available["excess_vector_collision_rate"] == 0.0).all()


def test_failed_run_is_rejected_from_census(measured):
    result, metadata = measured
    result.passed = False
    with pytest.raises(ValueError, match="excluded"):
        measure_checkpoint(result, result.checkpoints[-1], result.checkpoints[0], metadata)
    result.passed = True
