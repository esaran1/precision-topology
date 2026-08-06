"""Layer at which linking changes, per parametrization.

The specific claim under test is that with an untweaked parametrization the
fold direction is exposed immediately after the first affine map, so that
folding at layer 1 would be a property of the problem rather than of the
network.  If some parametrizations fold later, that supports the account and
identifies which properties expose the direction early.

Only width 3 is traced: there the estimate is the Theorem 4.7 invariant in the
layer's own space and no projection is involved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .linking import linking_number
from .linking_trace import ARTIFACT_DISTANCE
from .models import MLP
from .parametrization import GRID, TorusLink, core_curves, sample_link
from .parametrization_sweep import EVAL_SEED_BASE, TRAIN_SEED_BASE
from .train import TrainingConfig, train_mlp


@torch.no_grad()
def trace(model: MLP, link: TorusLink, n_core_points: int = 512) -> list[dict[str, object]]:
    """Propagate this parametrization's own cores through the network."""

    first_array, second_array = core_curves(link, n_core_points)
    first = torch.as_tensor(first_array, dtype=torch.float64)
    second = torch.as_tensor(second_array, dtype=torch.float64)

    rows: list[dict[str, object]] = []
    for layer_index in range(model.hidden_depth + 1):
        if layer_index > 0:
            hidden = model.hidden_layers[layer_index - 1]
            weight = hidden.weight.to(torch.float64)
            bias = hidden.bias.to(torch.float64)
            first = model._activate(torch.nn.functional.linear(first, weight, bias))
            second = model._activate(torch.nn.functional.linear(second, weight, bias))
        distance = float(torch.cdist(first, second).min().item())
        estimate = linking_number(first, second)
        usable = estimate.defined and distance > ARTIFACT_DISTANCE
        rows.append(
            {
                "layer": layer_index,
                "linking_rounded": estimate.rounded if usable else None,
                "linking_raw": estimate.raw if usable else None,
                "min_distance": distance,
                "defined": estimate.defined,
                "reportable": usable,
            }
        )
    return rows


def fold_layer(rows: list[dict[str, object]]) -> int | None:
    """First layer whose reportable linking number differs from the input."""

    baseline = rows[0]["linking_rounded"]
    for row in rows[1:]:
        if row["reportable"] and row["linking_rounded"] != baseline:
            return int(row["layer"])
    return None


def trace_successful_runs(
    sweep: pd.DataFrame,
    config_by_name: dict[str, TorusLink],
    n_per_class: int = 1_000,
    max_steps: int = 2_000,
    learning_rate: float = 1e-2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Retrain and trace every width-3 run that reached perfect evaluation."""

    selected = sweep[(sweep.width == 3) & (sweep.perfect_eval)]
    records: list[dict[str, object]] = []
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        link = config_by_name[str(row.parametrization)]
        seed = int(row.seed)
        train_data = sample_link(link, n_per_class, TRAIN_SEED_BASE + seed)
        eval_data = sample_link(link, n_per_class, EVAL_SEED_BASE + seed)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=int(row.depth),
            hidden_width=3,
            activation=str(row.activation),  # type: ignore[arg-type]
            config=TrainingConfig(seed=seed, max_steps=max_steps, learning_rate=learning_rate),
        )
        if result.final_eval_accuracy != float(row.final_eval_accuracy):
            raise RuntimeError(
                f"reconstruction diverged for {row.parametrization} "
                f"{row.activation} d={row.depth} s={seed}"
            )
        rows = trace(result.model, link)
        change = fold_layer(rows)
        for entry in rows:
            records.append(
                {
                    "parametrization": str(row.parametrization),
                    "axis_aligned": bool(row.axis_aligned),
                    "activation": str(row.activation),
                    "monotonic": bool(row.monotonic),
                    "depth": int(row.depth),
                    "seed": seed,
                    "fold_layer": change,
                    **entry,
                }
            )
        if verbose:
            print(
                f"[{position}/{len(selected)}] {row.parametrization} {row.activation} "
                f"d={row.depth} s={seed} fold_layer={change}",
                flush=True,
            )
    return pd.DataFrame(records)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    sweep = pd.read_parquet(directory / "parametrization_sweep.parquet")
    frame = trace_successful_runs(sweep, {link.name: link for link in GRID})
    frame.to_csv(directory / "parametrization_linking.csv", index=False)
    frame.to_parquet(directory / "parametrization_linking.parquet", index=False)
    print(f"wrote {len(frame)} rows", flush=True)


if __name__ == "__main__":
    main()
