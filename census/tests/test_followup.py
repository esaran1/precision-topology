from pathlib import Path

import pandas as pd
import pytest

from src.followup import (
    dataset_gap_analysis,
    dynamics_analysis,
    recompute_initialization_baselines,
    width_baseline_analysis,
)


RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def saturation():
    return pd.read_parquet(RESULTS / "saturation.parquet")


@pytest.fixture(scope="module")
def dynamics():
    return pd.read_parquet(RESULTS / "training_dynamics.parquet")


@pytest.fixture(scope="module")
def initialization():
    return recompute_initialization_baselines()


def test_recomputed_initialization_matches_saved_accepted_baselines(saturation, initialization):
    saved = saturation[
        (saturation["dataset"] == "linked_tori")
        & (saturation["activation"] == "tanh")
        & (saturation["format"] == "bfloat16")
        & (saturation["convention"] == "IEEE")
        & (saturation["distance_from_output"] == 0)
    ][["depth", "width", "seed", "baseline_vector_collision_rate"]]
    merged = saved.merge(
        initialization,
        on=["depth", "width", "seed"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    assert (
        merged["baseline_vector_collision_rate_saved"]
        == merged["baseline_vector_collision_rate_recomputed"]
    ).all()


def test_width_analysis_has_full_init_seeds_and_honest_width5_excess_count(
    saturation, initialization
):
    result = width_baseline_analysis(saturation, initialization).set_index("width")
    assert (result["all_init_seed_count"] == 5).all()
    assert result.loc[5, "matched_excess_seed_count"] == 4
    assert (result.loc[[15, 30, 50], "matched_excess_seed_count"] == 5).all()
    assert result.loc[5, "all_init_baseline_mean"] == pytest.approx(0.5604)
    assert result.loc[30, "matched_excess_mean"] == pytest.approx(0.505925)


def test_dynamics_plateau_and_post_plateau_fraction_are_pinned(dynamics):
    trajectory, summary = dynamics_analysis(dynamics)
    assert list(trajectory["training_step"]) == [0, 200, 500, 1000, 2000]
    row = summary.iloc[0]
    assert row["plateau_step"] == 200
    assert row["collision_at_plateau_mean"] == pytest.approx(0.2674)
    assert row["collision_at_final_mean"] == pytest.approx(0.4344)
    assert row["fraction_collision_increase_after_plateau_mean"] == pytest.approx(
        0.4051762340210721
    )


def test_dataset_gaps_are_paired_and_clear_seed_variance(saturation):
    gaps = dataset_gap_analysis(saturation).set_index("analysis")
    assert (gaps["seed_count"] == 5).all()
    assert gaps["clears_half_mean_rule"].all()
    assert gaps.loc["all-layer paper saturation", "blobs_minus_tori_mean"] == pytest.approx(
        0.20493225303234502
    )
    assert gaps.loc[
        "final-layer vector collision excess", "blobs_minus_tori_mean"
    ] == pytest.approx(0.3947525)
    assert gaps.loc["final-layer vector collision", "blobs_minus_tori_mean"] == pytest.approx(
        0.53783, abs=5e-7
    )
