"""Compute between-class margins for recovered runs and test the purity prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .margin import between_class_margin
from .precision import DELTA_TABLE
from .recover import load_results, recover_subset


def margin_frame(
    results_directory: Path,
    dataset: str = "linked_tori",
    activation: str = "tanh",
    verbose: bool = True,
) -> pd.DataFrame:
    """Measure between-class margin at every layer and precision of a subset."""

    status, saturation = load_results(results_directory)
    quantizers = []
    for spec in DELTA_TABLE:
        if spec.quantizer is None or spec.quantizer in quantizers:
            continue
        if spec.quantizer.startswith("fixed-") and activation != "tanh":
            continue
        quantizers.append(spec.quantizer)

    rows: list[dict[str, object]] = []
    for run, deviations in recover_subset(
        status, saturation, dataset, activation, verbose=verbose
    ):
        preactivations = run.result.checkpoints[-1].eval_preactivations
        for layer_index, values in enumerate(preactivations, start=1):
            for quantizer in quantizers:
                measurement = between_class_margin(
                    values, run.eval_labels, quantizer, activation  # type: ignore[arg-type]
                )
                rows.append(
                    {
                        "dataset": run.metadata.dataset,
                        "depth": run.metadata.depth,
                        "width": run.metadata.width,
                        "activation": run.metadata.activation,
                        "seed": run.metadata.seed,
                        "layer": layer_index,
                        "distance_from_output": run.metadata.depth - layer_index,
                        "quantizer": quantizer,
                        "min_between_class_distance": measurement.min_between_class_distance,
                        "min_between_class_chebyshev": measurement.min_between_class_chebyshev,
                        "quantization_step": measurement.quantization_step,
                        "margin_in_steps": measurement.margin_in_steps,
                        "margin_in_steps_chebyshev": measurement.margin_in_steps_chebyshev,
                        "margin_below_one": measurement.below_one,
                        "between_class_collision_pairs": measurement.between_class_collision_pairs,
                        "recovery_max_saturation_deviation": deviations[
                            "max_saturation_deviation"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def join_purity(margins: pd.DataFrame, saturation: pd.DataFrame) -> pd.DataFrame:
    """Attach recorded collision-group purity to each margin row."""

    keys = ["dataset", "depth", "width", "activation", "seed", "layer", "quantizer"]
    recorded = (
        saturation[keys + ["collision_group_pure_fraction", "collision_group_count"]]
        .drop_duplicates(subset=keys)
        .copy()
    )
    return margins.merge(recorded, on=keys, how="left")


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    frame = margin_frame(directory)
    _, saturation = load_results(directory)
    joined = join_purity(frame, saturation)
    joined.to_csv(directory / "between_class_margin.csv", index=False)
    print(f"wrote {len(joined)} margin rows")


if __name__ == "__main__":
    main()
