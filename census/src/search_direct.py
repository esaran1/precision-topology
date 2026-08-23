"""Search 4 of Part 2b: derivative-free direct optimization of the weights.

CMA-ES on the flattened parameters of width-3 networks.  No SGD anywhere.

**Objective change from the registration, forced by the positive control.**
The registered objective was the training error count with a bounded
tie-breaker.  That objective failed its own positive control: on
``sin_family(1.5)``, where SGD separates 20% of the time, count-driven
CMA-ES reached best counts of 11-57 across budgets to 6,000 generations and
never 0 -- the count landscape is a staircase of plateaus and the search
stalls on it.  The objective was therefore changed to the smooth full-batch
cross-entropy, with the error count used for verification only; under it the
positive control reaches 0 train errors in 2 of 3 probe restarts.  The
registered *predictions* are unchanged; this is a machinery fix documented in
``results/search_results.md``.

``sin_family(1.5)`` is the positive control: the machinery must find
separating networks where SGD can, or its failures on monotonic targets
mean nothing.  Any candidate reaching 0 train errors is verified on the
held-out eval set; only 0 eval errors counts as separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .cmaes import cma_es
from .models import MLP


@dataclass(frozen=True)
class DirectTarget:
    activation: str
    parameter: float | None
    monotonic: bool


TARGETS: tuple[DirectTarget, ...] = (
    DirectTarget("sin_family", 1.5, False),  # positive control
    DirectTarget("tanh", None, True),
    DirectTarget("sin_family", 1.0, True),
    DirectTarget("sin_family", 0.95, True),
)
DEPTHS: Sequence[int] = (3, 5)
RESTARTS: Sequence[int] = tuple(range(20))
MAX_GENERATIONS = 4_000
POPULATION = 15
SIGMA0 = 0.5
LOSS_TARGET = 1e-4


def flatten(model: MLP) -> np.ndarray:
    return np.concatenate(
        [parameter.detach().numpy().ravel() for parameter in model.parameters()]
    ).astype(np.float64)


def unflatten(model: MLP, vector: np.ndarray) -> None:
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            values = vector[cursor : cursor + size].reshape(parameter.shape)
            parameter.copy_(torch.as_tensor(values, dtype=parameter.dtype))
            cursor += size
    assert cursor == vector.size


def make_objective(
    model: MLP, features: torch.Tensor, labels: torch.Tensor
):
    """Smooth full-batch cross-entropy; error counts are verification only."""

    def objective(vector: np.ndarray) -> float:
        unflatten(model, vector)
        with torch.no_grad():
            return float(F.cross_entropy(model(features), labels).item())

    return objective


def count_errors(
    model: MLP, vector: np.ndarray, features: torch.Tensor, labels: torch.Tensor
) -> int:
    unflatten(model, vector)
    with torch.no_grad():
        return int((model(features).argmax(dim=1) != labels).sum().item())


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    data_config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        for target in TARGETS:
            for restart in RESTARTS:
                # Restart-specific data seed keeps parity with the sweeps'
                # per-seed datasets; the search seed drives CMA-ES draws.
                train_data, eval_data, *_ = _make_data(
                    "linked_tori", restart, data_config
                )
                features = torch.as_tensor(train_data.features, dtype=torch.float32)
                labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
                eval_features = torch.as_tensor(
                    eval_data.features, dtype=torch.float32
                )
                eval_labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)

                torch.manual_seed(restart)
                model = MLP(
                    3,
                    depth,
                    3,
                    target.activation,  # type: ignore[arg-type]
                    activation_parameter=target.parameter,
                )
                x0 = flatten(model)
                objective = make_objective(model, features, labels)
                started = time.monotonic()
                result = cma_es(
                    objective,
                    x0=x0,
                    sigma0=SIGMA0,
                    max_generations=MAX_GENERATIONS,
                    population=POPULATION,
                    seed=restart,
                    target_f=LOSS_TARGET,
                )
                best_train_errors = count_errors(
                    model, result.best_x, features, labels
                )
                eval_errors = count_errors(
                    model, result.best_x, eval_features, eval_labels
                )
                rows.append(
                    {
                        "activation": target.activation,
                        "parameter": target.parameter,
                        "monotonic": target.monotonic,
                        "depth": depth,
                        "restart": restart,
                        "best_loss": result.best_f,
                        "best_train_errors": best_train_errors,
                        "eval_errors_of_best": eval_errors,
                        "separated": best_train_errors == 0 and eval_errors == 0,
                        "evaluations": result.evaluations,
                        "generations": result.generations,
                        "duration_seconds": time.monotonic() - started,
                    }
                )
                if verbose:
                    label = (
                        f"{target.activation}({target.parameter:g})"
                        if target.parameter is not None
                        else target.activation
                    )
                    print(
                        f"d={depth} {label} r={restart}: train={best_train_errors} "
                        f"eval={eval_errors} gens={result.generations}",
                        flush=True,
                    )
                _write(pd.DataFrame(rows), output_directory)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "search_direct"
    with artifact_lock(stem, "direct CMA-ES search"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
