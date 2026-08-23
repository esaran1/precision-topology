"""Part 3: localize the monotonic failure.

Three probes, predictions registered in ``results/localization_prediction.md``:

1. Layerwise distillation of a separating GELU network into monotonic
   students (prefix replacement: student computes layers 1..k, the GELU
   network's remaining layers and head are reattached unchanged).
2. Linking traces through the best monotonic width-3 runs -- the direct
   comparison to the 34 GELU traces that has been missing.
3. The spatial distribution of near-miss errors relative to the link
   geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .census import SweepConfig, _make_data
from .linking_trace import trace_linking
from .models import MLP
from .train import TrainingConfig, seed_everything, train_mlp


def _standard_config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )


def reconstruct(
    activation: str, parameter: float | None, depth: int, seed: int
) -> tuple[MLP, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically retrain one recorded width-3 run."""

    train_data, eval_data, *_ = _make_data("linked_tori", seed, _standard_config())
    result = train_mlp(
        train_data,
        eval_data,
        hidden_depth=depth,
        hidden_width=3,
        activation=activation,  # type: ignore[arg-type]
        config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        activation_parameter=parameter,
    )
    return (
        result.model,
        torch.as_tensor(train_data.features, dtype=torch.float32),
        torch.as_tensor(train_data.labels, dtype=torch.int64),
        torch.as_tensor(eval_data.features, dtype=torch.float32),
        torch.as_tensor(eval_data.labels, dtype=torch.int64),
    )


# ---------------------------------------------------------------- probe 1


def _prefix_output(model: MLP, inputs: torch.Tensor, k: int) -> torch.Tensor:
    hidden = inputs
    for layer in model.hidden_layers[:k]:
        hidden = model._activate(layer(hidden))
    return hidden


def _suffix_logits(model: MLP, hidden: torch.Tensor, k: int) -> torch.Tensor:
    for layer in model.hidden_layers[k:]:
        hidden = model._activate(layer(hidden))
    return model.output_layer(hidden)


def distill_layer(
    teacher: MLP,
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    k: int,
    student_activation: str = "tanh",
    student_depth: int | None = None,
    steps: int = 4_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Train a monotonic student to regress the teacher's layer-k output.

    The student's depth defaults to k (the same architectural prefix).  After
    training, the teacher's remaining layers and head are reattached to the
    student's output and classification errors are measured.
    """

    with torch.no_grad():
        train_target = _prefix_output(teacher, train_features, k)
        eval_target = _prefix_output(teacher, eval_features, k)
    seed_everything(seed)
    student = MLP(3, student_depth or k, 3, student_activation)  # type: ignore[arg-type]
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    student.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _prefix_output(student, train_features, len(student.hidden_layers))
        F.mse_loss(prediction, train_target).backward()
        optimizer.step()
    student.eval()
    with torch.no_grad():
        student_eval = _prefix_output(student, eval_features, len(student.hidden_layers))
        mse = float(F.mse_loss(student_eval, eval_target).item())
        worst = float((student_eval - eval_target).norm(dim=1).max().item())
        logits = _suffix_logits(teacher, student_eval, k)
        errors = int((logits.argmax(dim=1) != eval_labels).sum().item())
        teacher_logits = _suffix_logits(teacher, eval_target, k)
        teacher_errors = int((teacher_logits.argmax(dim=1) != eval_labels).sum().item())
    return {
        "layer": k,
        "student_activation": student_activation,
        "eval_mse": mse,
        "worst_point_distance": worst,
        "errors_with_teacher_suffix": errors,
        "teacher_errors": teacher_errors,
    }


# ---------------------------------------------------------------- probe 3


@dataclass(frozen=True)
class ErrorGeography:
    """Misclassified eval points located relative to the link geometry."""

    n_errors: int
    fraction_class_a: float
    median_distance_to_other_core: float
    p10_distance_to_other_core: float
    baseline_median_distance: float
    angular_concentration: float
    concentration_null_p95: float


def _distance_to_cores(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distance to core A, distance to core B) for each point.

    Core A: unit circle in z = 0 centred at origin.  Core B: unit circle in
    the xz-plane centred at (1, 0, 0) -- the data.linked_tori geometry.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radial_a = np.sqrt(x**2 + y**2)
    dist_a = np.sqrt((radial_a - 1.0) ** 2 + z**2)
    xb = x - 1.0
    radial_b = np.sqrt(xb**2 + z**2)
    dist_b = np.sqrt((radial_b - 1.0) ** 2 + y**2)
    return dist_a, dist_b


def error_geography(
    model: MLP,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    null_draws: int = 5_000,
    null_seed: int = 0,
) -> ErrorGeography:
    with torch.no_grad():
        wrong = (model(eval_features).argmax(dim=1) != eval_labels).numpy()
    points = eval_features.numpy()
    labels = eval_labels.numpy()
    dist_a, dist_b = _distance_to_cores(points)
    # Distance to the *other* component's core.
    other = np.where(labels == 0, dist_b, dist_a)
    errors = wrong.nonzero()[0]
    if errors.size == 0:
        raise ValueError("no errors to locate")

    # Angle around the own core, for angular concentration.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    angle = np.where(labels == 0, np.arctan2(y, x), np.arctan2(z, x - 1.0))
    vectors = np.stack([np.cos(angle[errors]), np.sin(angle[errors])])
    concentration = float(np.linalg.norm(vectors.mean(axis=1)))

    # Null: same number of points drawn from the same class mix at random.
    rng = np.random.default_rng(null_seed)
    null_values = []
    for _ in range(null_draws):
        chosen = rng.choice(points.shape[0], size=errors.size, replace=False)
        null_vectors = np.stack([np.cos(angle[chosen]), np.sin(angle[chosen])])
        null_values.append(float(np.linalg.norm(null_vectors.mean(axis=1))))
    null_p95 = float(np.quantile(null_values, 0.95))

    return ErrorGeography(
        n_errors=int(errors.size),
        fraction_class_a=float((labels[errors] == 0).mean()),
        median_distance_to_other_core=float(np.median(other[errors])),
        p10_distance_to_other_core=float(np.quantile(other[errors], 0.10)),
        baseline_median_distance=float(np.median(other)),
        angular_concentration=concentration,
        concentration_null_p95=null_p95,
    )


# ---------------------------------------------------------------- runner


NEAR_MISS_RUNS: tuple[tuple[str, float | None, int, int], ...] = (
    ("sin_family", 0.95, 8, 2),   # 2 errors
    ("sin_family", 0.90, 12, 0),  # 6
    ("sin_family", 0.95, 12, 19),  # 6
    ("sin_family", 1.00, 3, 4),   # 6
    ("sin_family", 0.95, 8, 1),   # 8
    ("sin_family", 1.00, 5, 14),  # 8
    ("tanh", None, 8, 8),         # 26, best fixed monotonic
    ("tanh", None, 5, 12),        # 49
)

GELU_TRACE_SOURCE = ("gelu", None, 3, 10)  # separating, for side-by-side


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"

    # Probe 2 + 3 on the near-miss runs.  Traces run at two resolutions and a
    # value is convergent only if the reportable rounded value agrees; the
    # 512-point default proved under-resolved deep in distorting networks.
    # Determinant signs are recorded because an orientation-reversing affine
    # layer legitimately flips the sign of lk; |lk| is the invariant.
    trace_rows: list[dict[str, object]] = []
    geo_rows: list[dict[str, object]] = []
    for activation, parameter, depth, seed in NEAR_MISS_RUNS + (GELU_TRACE_SOURCE,):
        model, _, _, eval_features, eval_labels = reconstruct(
            activation, parameter, depth, seed
        )
        determinants = [
            float(np.linalg.det(layer.weight.detach().numpy().astype(np.float64)))
            for layer in model.hidden_layers
        ]
        coarse = trace_linking(model, n_core_points=2048)
        fine = trace_linking(model, n_core_points=8192)
        for low, measurement in zip(coarse, fine):
            low_value = low.rounded if low.reportable else None
            fine_value = measurement.rounded if measurement.reportable else None
            trace_rows.append(
                {
                    "activation": activation,
                    "parameter": parameter,
                    "depth": depth,
                    "seed": seed,
                    "layer": measurement.layer,
                    "layer_determinant": (
                        determinants[measurement.layer - 1]
                        if measurement.layer > 0
                        else None
                    ),
                    "linking_number": fine_value,
                    "converged": fine_value == low_value,
                    "residual": measurement.residual,
                    "min_distance": measurement.min_distance,
                    "regime": measurement.regime,
                }
            )
        with torch.no_grad():
            errors = int((model(eval_features).argmax(1) != eval_labels).sum().item())
        if 0 < errors:
            geography = error_geography(model, eval_features, eval_labels)
            geo_rows.append(
                {
                    "activation": activation,
                    "parameter": parameter,
                    "depth": depth,
                    "seed": seed,
                    "errors": geography.n_errors,
                    "fraction_class_a": geography.fraction_class_a,
                    "median_dist_other_core": geography.median_distance_to_other_core,
                    "p10_dist_other_core": geography.p10_distance_to_other_core,
                    "baseline_median_dist": geography.baseline_median_distance,
                    "angular_concentration": geography.angular_concentration,
                    "concentration_null_p95": geography.concentration_null_p95,
                }
            )
        print(f"{activation}({parameter}) d={depth} s={seed}: errors={errors}", flush=True)

    pd.DataFrame(trace_rows).to_csv(directory / "localization_traces.csv", index=False)
    pd.DataFrame(geo_rows).to_csv(directory / "localization_errors.csv", index=False)

    # Probe 1: distillation of the separating GELU d=3 network.
    activation, parameter, depth, seed = GELU_TRACE_SOURCE
    teacher, train_features, _, eval_features, eval_labels = reconstruct(
        activation, parameter, depth, seed
    )
    distill_rows = []
    for k in range(1, depth + 1):
        for student_seed in range(3):
            outcome = distill_layer(
                teacher, train_features, eval_features, eval_labels, k,
                seed=student_seed,
            )
            outcome["student_seed"] = student_seed
            distill_rows.append(outcome)
            print(
                f"distill layer {k} seed {student_seed}: mse={outcome['eval_mse']:.5f} "
                f"errors={outcome['errors_with_teacher_suffix']}",
                flush=True,
            )
    pd.DataFrame(distill_rows).to_csv(directory / "localization_distill.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
