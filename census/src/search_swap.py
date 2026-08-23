"""Search 2 of Part 2b: descend from a separating GELU solution.

Reconstruct each separating width-3 GELU run from the threshold sweep
(deterministic retrain, verified against the recorded accuracy), swap the
activation to a monotonic one with the weights frozen at the GELU solution,
measure immediately, then fine-tune and measure again.

If the swap alone preserves separation, the solution exists in monotonic
parameter space and SGD merely failed to find it from scratch — the
strongest possible overturn of the monotonic zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .models import MLP
from .train import TrainingConfig, train_mlp


# Separating width-3 GELU runs in the threshold sweep (depth, seed).
GELU_SOURCES: tuple[tuple[int, int], ...] = (
    (3, 10),
    (3, 17),
    (5, 0),
    (5, 1),
    (8, 14),
    (12, 2),
)

# (activation, parameter): all monotonic; sin(1.0) is the threshold itself.
TARGETS: tuple[tuple[str, float | None], ...] = (
    ("tanh", None),
    ("leaky_relu", None),
    ("sin_family", 1.0),
    ("sin_family", 0.95),
)

FINE_TUNE_RATES = (1e-3, 1e-2)
FINE_TUNE_STEPS = 2_000


class ReconstructionDivergence(RuntimeError):
    """A reconstructed GELU source did not reproduce separation."""


@dataclass
class SwapOutcome:
    depth: int
    seed: int
    target: str
    parameter: float | None
    errors_after_swap: int
    errors_after_tune: dict[float, int]


def _errors(model: MLP, features: torch.Tensor, labels: torch.Tensor) -> int:
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return int((predictions != labels).sum().item())


def _fine_tune(
    model: MLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    steps: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)  # Adam itself is deterministic; seed for hygiene.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    data_config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    rows: list[dict[str, object]] = []
    for depth, seed in GELU_SOURCES:
        train_data, eval_data, *_ = _make_data("linked_tori", seed, data_config)
        source = train_mlp(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=3,
            activation="gelu",
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        )
        if source.final_eval_accuracy < 1.0:
            raise ReconstructionDivergence(
                f"gelu d={depth} s={seed} reconstructed at "
                f"{source.final_eval_accuracy:.6f}, expected 1.0"
            )
        train_features = torch.as_tensor(train_data.features, dtype=torch.float32)
        train_labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
        eval_features = torch.as_tensor(eval_data.features, dtype=torch.float32)
        eval_labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)
        state = source.model.state_dict()

        for target, parameter in TARGETS:
            swapped = MLP(3, depth, 3, target, activation_parameter=parameter)  # type: ignore[arg-type]
            swapped.load_state_dict(state)
            after_swap = _errors(swapped, eval_features, eval_labels)
            row: dict[str, object] = {
                "source_depth": depth,
                "source_seed": seed,
                "target": target,
                "parameter": parameter,
                "errors_after_swap": after_swap,
                "train_errors_after_swap": _errors(
                    swapped, train_features, train_labels
                ),
            }
            for rate in FINE_TUNE_RATES:
                tuned = MLP(3, depth, 3, target, activation_parameter=parameter)  # type: ignore[arg-type]
                tuned.load_state_dict(state)
                started = time.monotonic()
                _fine_tune(
                    tuned, train_features, train_labels, rate, FINE_TUNE_STEPS, seed
                )
                row[f"errors_after_tune_lr{rate:g}"] = _errors(
                    tuned, eval_features, eval_labels
                )
                row[f"tune_seconds_lr{rate:g}"] = time.monotonic() - started
            rows.append(row)
            if verbose:
                print(
                    f"gelu d={depth} s={seed} -> {target}"
                    f"{'' if parameter is None else f'({parameter:g})'}: "
                    f"swap={after_swap} "
                    + " ".join(
                        f"lr{rate:g}={row[f'errors_after_tune_lr{rate:g}']}"
                        for rate in FINE_TUNE_RATES
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "search_swap"
    with artifact_lock(stem, "activation swap"):
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
