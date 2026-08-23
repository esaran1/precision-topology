"""One entry point for regenerating and verifying the project's numbers.

    python -m src.reproduce --list            # every artifact and its command
    python -m src.reproduce --verify          # cheap determinism spot-checks
    python -m src.reproduce --run <name>      # regenerate one sweep (long)

``--verify`` retrains one recorded run per sweep family and requires
bit-identical agreement with the stored artifact row, plus a same-process
double-training determinism check.  It runs in about a minute and is the
thing to run first on a new machine; if it passes, the long sweeps are
expected to reproduce exactly, because every sweep uses the same
deterministic pipeline it exercises.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"

# artifact stem -> (module to run, what it produces)
ARTIFACTS: dict[str, tuple[str, str]] = {
    "width_sweep": ("src.width_sweep", "width x depth x activation grid, d=3 link"),
    "threshold_sweep": ("src.threshold_sweep", "parametric activation families"),
    "parametrization_sweep": ("src.parametrization_sweep", "12 torus parametrizations"),
    "corrugation_sweep": ("src.corrugation_sweep", "both G.1 readings"),
    "protocol_sweep": ("src.protocol_sweep", "author-protocol replication"),
    "winding_sweep": ("src.winding_sweep", "linking-number scaling, q=1..4"),
    "search_restarts": ("src.search_restarts", "200-seed monotonic restarts"),
    "search_swap": ("src.search_swap", "GELU-solution activation swaps"),
    "search_anneal": ("src.search_anneal", "activation annealing traces"),
    "search_direct": ("src.search_direct", "CMA-ES direct weight search"),
    "dense_check": ("src.dense_check", "dense verification of separating runs"),
    "localization": ("src.localization", "traces, distillation, error geography"),
}


def list_artifacts() -> None:
    for stem, (module, description) in ARTIFACTS.items():
        exists = (
            "present"
            if (RESULTS / f"{stem}.csv").exists()
            or (RESULTS / f"{stem}.parquet").exists()
            or (RESULTS / f"{stem}_traces.csv").exists()
            else "MISSING"
        )
        print(f"{stem:24s} python -m {module:28s} [{exists}] {description}")


def _verify_one(
    description: str, stored: float, recomputed: float
) -> bool:
    exact = stored == recomputed
    print(
        f"  {'ok ' if exact else 'FAIL'} {description}: stored={stored!r} "
        f"recomputed={recomputed!r}"
    )
    return exact


def verify() -> bool:
    from .census import SweepConfig, _make_data
    from .train import TrainingConfig, train_mlp

    passed = True
    config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )

    print("determinism: same config trained twice in-process")
    train_data, eval_data, *_ = _make_data("linked_tori", 0, config)
    results = [
        train_mlp(
            train_data,
            eval_data,
            hidden_depth=3,
            hidden_width=3,
            activation="tanh",
            config=TrainingConfig(seed=0, max_steps=200),
        )
        for _ in range(2)
    ]
    identical = all(
        torch.equal(a, b)
        for a, b in zip(
            results[0].model.state_dict().values(),
            results[1].model.state_dict().values(),
        )
    )
    print(f"  {'ok ' if identical else 'FAIL'} state dicts bit-identical")
    passed &= identical

    print("recovery: one recorded run per sweep family, exact accuracy match")
    checks = [
        ("width_sweep.parquet", dict(activation="gelu", depth=3, width=3, seed=10), None),
        (
            "threshold_sweep.parquet",
            dict(activation="sin_family", parameter=0.95, depth=8, width=3, seed=2),
            0.95,
        ),
    ]
    for artifact, selector, parameter in checks:
        frame = pd.read_parquet(RESULTS / artifact)
        for key, value in selector.items():
            frame = frame[frame[key] == value]
        row = frame.iloc[0]
        train_data, eval_data, *_ = _make_data("linked_tori", int(row.seed), config)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=int(row.depth),
            hidden_width=int(row.width),
            activation=row.activation,
            config=TrainingConfig(seed=int(row.seed), max_steps=2_000, learning_rate=1e-2),
            activation_parameter=parameter,
        )
        passed &= _verify_one(
            f"{artifact} {selector}",
            float(row.final_eval_accuracy),
            result.final_eval_accuracy,
        )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--run", metavar="ARTIFACT")
    arguments = parser.parse_args()
    if arguments.list:
        list_artifacts()
    elif arguments.verify:
        sys.exit(0 if verify() else 1)
    else:
        if arguments.run not in ARTIFACTS:
            parser.error(f"unknown artifact: {arguments.run}; use --list")
        module, _ = ARTIFACTS[arguments.run]
        import importlib

        importlib.import_module(module).main()


if __name__ == "__main__":
    main()
