"""Search 3 of Part 2b: anneal Family A through the monotonicity threshold.

Train ``sin_family(a=3)`` width-3 networks to separation, then lower ``a`` in
steps of 0.05 from 3.0 to 0.8, training 200 further steps at each value with
the network free to adapt.  Recorded per step: eval errors before and after
the adaptation window, so "could not hold separation" is distinguishable
from "was not given the chance to adapt".

The registered question is the largest ``a`` at which separation is lost.
Holding 0 errors at any ``a <= 1.0`` overturns the monotonic zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .train import TrainingConfig, train_mlp


START_PARAMETER = 3.0
END_PARAMETER = 0.8
STEP = 0.05
ADAPT_STEPS = 200
ANNEAL_LEARNING_RATE = 1e-2


def anneal_schedule() -> tuple[float, ...]:
    count = int(round((START_PARAMETER - END_PARAMETER) / STEP))
    return tuple(round(START_PARAMETER - index * STEP, 10) for index in range(count + 1))


def _errors(model, features: torch.Tensor, labels: torch.Tensor) -> int:
    model.eval()
    with torch.no_grad():
        return int((model(features).argmax(dim=1) != labels).sum().item())


def anneal_one(depth: int, seed: int, verbose: bool = True) -> pd.DataFrame | None:
    """Train at a=3; if separating, anneal down.  None if the seed never separates."""

    data_config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    train_data, eval_data, *_ = _make_data("linked_tori", seed, data_config)
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=depth,
        hidden_width=3,
        activation="sin_family",
        config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        activation_parameter=START_PARAMETER,
    )
    if result.final_eval_accuracy < 1.0:
        return None

    model = result.model
    train_features = torch.as_tensor(train_data.features, dtype=torch.float32)
    train_labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
    eval_features = torch.as_tensor(eval_data.features, dtype=torch.float32)
    eval_labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)
    optimizer = torch.optim.Adam(model.parameters(), lr=ANNEAL_LEARNING_RATE)

    rows: list[dict[str, object]] = []
    for parameter in anneal_schedule():
        model.activation_parameter = parameter
        before = _errors(model, eval_features, eval_labels)
        model.train()
        for _ in range(ADAPT_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_features), train_labels)
            loss.backward()
            optimizer.step()
        after = _errors(model, eval_features, eval_labels)
        rows.append(
            {
                "depth": depth,
                "seed": seed,
                "parameter": parameter,
                "monotonic": parameter <= 1.0,
                "errors_before_adapt": before,
                "errors_after_adapt": after,
                "train_errors_after_adapt": _errors(model, train_features, train_labels),
            }
        )
        if verbose and abs(parameter - round(parameter * 4) / 4) < 1e-9:
            print(
                f"d={depth} s={seed} a={parameter:.2f} "
                f"before={before} after={after}",
                flush=True,
            )
    return pd.DataFrame(rows)


def separating_seeds(depth_seed_limit: int = 12) -> list[tuple[int, int]]:
    """(depth, seed) pairs whose a=3 run separated in the threshold sweep."""

    directory = Path(__file__).resolve().parents[1] / "results"
    frame = pd.read_parquet(directory / "threshold_sweep.parquet")
    subset = frame[
        (frame.activation == "sin_family")
        & (frame.parameter == 3.0)
        & (frame.width == 3)
        & frame.perfect_eval
    ]
    pairs = [(int(row.depth), int(row.seed)) for row in subset.itertuples()]
    return pairs[:depth_seed_limit]


def run(output_directory: Path, n_traces: int = 12) -> pd.DataFrame:
    frames = []
    for depth, seed in separating_seeds(n_traces):
        trace = anneal_one(depth, seed)
        if trace is None:
            raise RuntimeError(
                f"d={depth} s={seed} was recorded separating at a=3 but did not "
                "reproduce; stopping rather than substituting"
            )
        frames.append(trace)
    frame = pd.concat(frames, ignore_index=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "search_anneal"
    with artifact_lock(stem, "activation annealing"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
    return frame


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
