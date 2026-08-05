"""Run the F / QF / G comparison over recovered census runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interleaved import compare_final_layer, compare_regimes
from .precision import DELTA_TABLE
from .recover import load_results, recover_subset


def interleaved_frame(
    results_directory: Path,
    dataset: str = "linked_tori",
    activation: str = "tanh",
    depth: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare collision regimes for each accepted run in the subset."""

    status, saturation = load_results(results_directory)
    quantizers: list[str] = []
    for spec in DELTA_TABLE:
        if spec.quantizer is None or spec.quantizer in quantizers:
            continue
        if spec.quantizer.startswith("fixed-") and activation != "tanh":
            continue
        quantizers.append(spec.quantizer)

    rows: list[dict[str, object]] = []
    for run, deviations in recover_subset(
        status, saturation, dataset, activation, depth, verbose=verbose
    ):
        for quantizer in quantizers:
            comparison = compare_regimes(
                run.model, run.eval_features, run.eval_labels, quantizer
            )
            if not comparison.post_hoc_superset_holds:
                raise RuntimeError(
                    "collisions(QF) >= collisions(F) failed for "
                    f"{run.key} at {quantizer}; this is a bug, not a finding. "
                    f"violating pairs: {comparison.post_hoc_violations[:8]}"
                )
            layer = compare_final_layer(
                run.model, run.eval_features, run.eval_labels, quantizer
            )
            rows.append(
                {
                    "dataset": run.metadata.dataset,
                    "depth": run.metadata.depth,
                    "width": run.metadata.width,
                    "activation": run.metadata.activation,
                    "seed": run.metadata.seed,
                    "quantizer": quantizer,
                    "n_inputs": comparison.n_inputs,
                    # Output-level regimes as originally specified.
                    "output_full_precision_pairs": comparison.full_precision_pairs,
                    "output_post_hoc_pairs": comparison.post_hoc_pairs,
                    "output_interleaved_pairs": comparison.interleaved_pairs,
                    "post_hoc_superset_holds": comparison.post_hoc_superset_holds,
                    "interleaved_superset_holds": comparison.interleaved_superset_holds,
                    "f_not_g_pairs": len(comparison.interleaved_violations),
                    "f_not_g_within_class": comparison.within_class_violations,
                    "f_not_g_between_class": comparison.between_class_violations,
                    # Final hidden layer, where quantized collisions are dense.
                    "final_post_hoc_pairs": layer.post_hoc_pairs,
                    "final_interleaved_pairs": layer.interleaved_pairs,
                    "final_qf_not_g": layer.post_hoc_not_interleaved,
                    "final_qf_not_g_within": layer.post_hoc_not_interleaved_within,
                    "final_qf_not_g_between": layer.post_hoc_not_interleaved_between,
                    "final_g_not_qf": layer.interleaved_not_post_hoc,
                    "final_g_not_qf_within": layer.interleaved_not_post_hoc_within,
                    "final_g_not_qf_between": layer.interleaved_not_post_hoc_between,
                    "final_containment_holds": layer.containment_holds,
                    "recovery_max_saturation_deviation": deviations[
                        "max_saturation_deviation"
                    ],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    # Subset: accepted linked-tori tanh runs at depth 6, all seeds, all widths.
    frame = interleaved_frame(directory, depth=6)
    frame.to_csv(directory / "interleaved_quantization.csv", index=False)
    print(f"wrote {len(frame)} comparison rows")


if __name__ == "__main__":
    main()
