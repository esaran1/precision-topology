"""Task E Part 1b: width dependence of the activation advantage at d = 3.

Registered in ``results/width_prediction.md`` (P-W1..P-W4 plus the dated
amendment) BEFORE this module ran.  Fresh seeds 100-199 keep every cell
out-of-sample relative to ``width_sweep.csv`` (seeds 0-19); depth 3 is the
primary grid and depth 6 a 40-seed interaction slice.

Measures, fixed at registration:
- dense-verified separation: 0 errors on the 2,000-point held-out eval AND
  0 errors on 100,000 fresh points (crc32-derived seed, dense_check style);
- minimum errors: eval error count per run;
- convergence speed: first step with train accuracy 1.0, probed every 50
  steps, censored at 2,000 (recorded as -1, never imputed).
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .data import linked_tori
from .models import MLP
from .train import seed_everything

TUBE_RADIUS = 0.2
N_PER_CLASS = 1_000
STEPS = 2_000
LR = 1e-2
PROBE_EVERY = 50
DENSE_PER_CLASS = 50_000
DENSE_SEED_BASE = 900_000

WIDTHS_PRIMARY = (3, 4, 5, 6, 8, 12, 16, 24, 32)
SEEDS_PRIMARY = tuple(range(100, 200))
WIDTHS_DEPTH6 = (3, 6, 16, 32)
SEEDS_DEPTH6 = tuple(range(100, 140))

ACTIVATIONS: tuple[tuple[str, float | None], ...] = (
    ("gelu", None),
    ("sin_family", 1.5),
    ("tanh", None),
    ("relu", None),
    ("leaky_relu", None),
)


def _tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(dataset.features, dtype=torch.float32),
        torch.tensor(dataset.labels, dtype=torch.int64),
    )


def _errors(model: MLP, features: torch.Tensor, labels: torch.Tensor) -> int:
    with torch.no_grad():
        return int((model(features).argmax(dim=1) != labels).sum().item())


def _inactive_fraction(model: MLP, features: torch.Tensor) -> float:
    """Fraction of hidden units with preactivation <= 0 on every input.

    Meaningful as "dead" only for the ReLU family; recorded for all runs.
    """

    with torch.no_grad():
        layers = model.collect_preactivations(features)
    dead = 0
    total = 0
    for pre in layers:
        dead += int((pre <= 0.0).all(dim=0).sum().item())
        total += pre.shape[1]
    return dead / total if total else 0.0


def run_one(activation: str, parameter: float | None, depth: int, width: int,
            seed: int) -> dict:
    started = time.monotonic()
    train = linked_tori(N_PER_CLASS, tube_radius=TUBE_RADIUS, seed=10_000 + seed)
    evaluation = linked_tori(N_PER_CLASS, tube_radius=TUBE_RADIUS, seed=20_000 + seed)
    x, y = _tensors(train)
    xe, ye = _tensors(evaluation)

    seed_everything(seed)
    model = MLP(3, depth, width, activation,  # type: ignore[arg-type]
                activation_parameter=parameter).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    steps_to_criterion = -1  # -1 = censored (never reached 1.0 train accuracy)
    model.train()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        if steps_to_criterion < 0 and step % PROBE_EVERY == 0:
            model.eval()
            if _errors(model, x, y) == 0:
                steps_to_criterion = step
            model.train()

    model.eval()
    train_errors = _errors(model, x, y)
    eval_errors = _errors(model, xe, ye)
    dense_errors = -1  # -1 = not checked (eval errors nonzero)
    if eval_errors == 0:
        key = f"width_effect|{activation}|{parameter}|{depth}|{width}|{seed}"
        dense_seed = DENSE_SEED_BASE + zlib.crc32(key.encode()) % 50_000
        dense = linked_tori(DENSE_PER_CLASS, tube_radius=TUBE_RADIUS, seed=dense_seed)
        xd, yd = _tensors(dense)
        dense_errors = _errors(model, xd, yd)

    return {
        "activation": activation, "parameter": parameter, "depth": depth,
        "width": width, "seed": seed,
        "train_errors": train_errors, "eval_errors": eval_errors,
        "dense_errors": dense_errors,
        "solved": eval_errors == 0 and dense_errors == 0,
        "steps_to_criterion": steps_to_criterion,
        "final_eval_accuracy": 1.0 - eval_errors / (2 * N_PER_CLASS),
        "at_chance": eval_errors >= N_PER_CLASS,
        "inactive_unit_fraction": _inactive_fraction(model, x),
        "duration_seconds": time.monotonic() - started,
    }


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "width_effect"
    rows: list[dict] = []

    def persist() -> None:
        frame = pd.DataFrame(rows)
        with artifact_lock(stem, "task E width sweep"):
            temp = stem.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))

    grids = (
        [(3, w, s) for w in WIDTHS_PRIMARY for s in SEEDS_PRIMARY]
        + [(6, w, s) for w in WIDTHS_DEPTH6 for s in SEEDS_DEPTH6]
    )
    # cell-major order: finish (depth, width, activation) cells one at a time
    for depth, width in sorted({(d, w) for d, w, _ in grids}):
        seeds = SEEDS_PRIMARY if depth == 3 else SEEDS_DEPTH6
        for activation, parameter in ACTIVATIONS:
            for seed in seeds:
                rows.append(run_one(activation, parameter, depth, width, seed))
            cell = [r for r in rows if r["depth"] == depth and r["width"] == width
                    and r["activation"] == activation]
            solved = sum(r["solved"] for r in cell)
            print(f"depth={depth} width={width} {activation}({parameter}): "
                  f"{solved}/{len(cell)} dense-verified", flush=True)
            persist()
    print("done", flush=True)


if __name__ == "__main__":
    main()
