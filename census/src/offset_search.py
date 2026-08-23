"""A4: the four reachability searches at a = 1.02.

Registered in ``results/offset_prediction.md``.  Every candidate reaching
eval-0 is dense-verified on 100,000 fresh points immediately; only
dense-verified separations count.
"""

from __future__ import annotations

from pathlib import Path
import time
import zlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .cmaes import cma_es
from .data import linked_tori
from .models import MLP
from .search_direct import count_errors, flatten, make_objective
from .train import TrainingConfig, train_mlp

TARGET_A = 1.02
DENSE_PER_CLASS = 50_000


def _config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )


def _dense_errors(model, tag: str) -> int:
    seed = 920_000 + zlib.crc32(tag.encode()) % 50_000
    dense = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=seed)
    features = torch.as_tensor(dense.features, dtype=torch.float32)
    labels = torch.as_tensor(dense.labels, dtype=torch.int64)
    with torch.no_grad():
        return int((model(features).argmax(dim=1) != labels).sum().item())


def _eval_errors(model, eval_features, eval_labels) -> int:
    with torch.no_grad():
        return int((model(eval_features).argmax(dim=1) != eval_labels).sum().item())


def search_restarts(rows: list) -> None:
    for depth in (5, 8):
        for seed in range(200):
            train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
            result = train_mlp(
                train_data, eval_data, hidden_depth=depth, hidden_width=3,
                activation="sin_family",
                config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
                activation_parameter=TARGET_A,
            )
            errors = int(round((1 - result.final_eval_accuracy) * 2_000))
            dense = (
                _dense_errors(result.model, f"restart|{depth}|{seed}")
                if errors == 0 else None
            )
            rows.append({"search": "restarts", "depth": depth, "seed": seed,
                         "eval_errors": errors, "dense_errors": dense,
                         "separated": errors == 0 and dense == 0})
            if seed % 50 == 49:
                print(f"restarts d={depth} [{seed+1}/200]", flush=True)


def _adam_steps(model, features, labels, steps, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), labels).backward()
        optimizer.step()
    model.eval()


def _a3_separating_sources(limit: int = 12) -> list[tuple[int, int]]:
    frame = pd.read_parquet(Path(__file__).resolve().parents[1] / "results" / "threshold_sweep.parquet")
    subset = frame[(frame.activation == "sin_family") & (frame.parameter == 3.0)
                   & (frame.width == 3) & frame.perfect_eval]
    return [(int(r.depth), int(r.seed)) for r in subset.itertuples()][:limit]


def search_finetune(rows: list) -> None:
    for depth, seed in _a3_separating_sources():
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
        source = train_mlp(
            train_data, eval_data, hidden_depth=depth, hidden_width=3,
            activation="sin_family",
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
            activation_parameter=3.0,
        )
        assert source.final_eval_accuracy >= 1.0
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        for lr in (1e-3, 1e-2):
            model = MLP(3, depth, 3, "sin_family", activation_parameter=TARGET_A)
            model.load_state_dict(source.model.state_dict())
            after_swap = _eval_errors(model, ef, el)
            _adam_steps(model, tf, tl, 2_000, lr)
            errors = _eval_errors(model, ef, el)
            dense = _dense_errors(model, f"finetune|{depth}|{seed}|{lr}") if errors == 0 else None
            rows.append({"search": "finetune", "depth": depth, "seed": seed, "lr": lr,
                         "errors_after_swap": after_swap, "eval_errors": errors,
                         "dense_errors": dense, "separated": errors == 0 and dense == 0})
        print(f"finetune d={depth} s={seed} done", flush=True)


def search_anneal(rows: list) -> None:
    schedule = [round(3.0 - 0.05 * i, 10) for i in range(int(round((3.0 - TARGET_A) / 0.05)) + 1)]
    schedule[-1] = TARGET_A
    for depth, seed in _a3_separating_sources():
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
        result = train_mlp(
            train_data, eval_data, hidden_depth=depth, hidden_width=3,
            activation="sin_family",
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
            activation_parameter=3.0,
        )
        model = result.model
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for value in schedule:
            model.activation_parameter = value
            model.train()
            for _ in range(200):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(tf), tl).backward()
                optimizer.step()
        model.activation_parameter = TARGET_A
        _adam_steps(model, tf, tl, 2_000, 1e-2)
        model.eval()
        errors = _eval_errors(model, ef, el)
        dense = _dense_errors(model, f"anneal|{depth}|{seed}") if errors == 0 else None
        rows.append({"search": "anneal", "depth": depth, "seed": seed,
                     "eval_errors": errors, "dense_errors": dense,
                     "separated": errors == 0 and dense == 0})
        print(f"anneal d={depth} s={seed}: errors at 1.02 = {errors}", flush=True)


def search_cma(rows: list) -> None:
    for depth in (3, 5):
        for restart in range(20):
            train_data, eval_data, *_ = _make_data("linked_tori", restart, _config())
            features = torch.as_tensor(train_data.features); labels = torch.as_tensor(train_data.labels)
            ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
            torch.manual_seed(restart)
            model = MLP(3, depth, 3, "sin_family", activation_parameter=TARGET_A)
            result = cma_es(
                make_objective(model, features, labels), flatten(model), 0.5,
                max_generations=4_000, population=15, seed=restart, target_f=1e-4,
            )
            train_errors = count_errors(model, result.best_x, features, labels)
            eval_errors = count_errors(model, result.best_x, ef, el)
            dense = _dense_errors(model, f"cma|{depth}|{restart}") if eval_errors == 0 else None
            rows.append({"search": "cma", "depth": depth, "seed": restart,
                         "train_errors": train_errors, "eval_errors": eval_errors,
                         "dense_errors": dense, "separated": eval_errors == 0 and dense == 0})
            print(f"cma d={depth} r={restart}: train={train_errors} eval={eval_errors}", flush=True)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    rows: list[dict] = []
    started = time.monotonic()
    for stage in (search_restarts, search_finetune, search_anneal, search_cma):
        stage(rows)
        frame = pd.DataFrame(rows)
        stem = directory / "offset_search"
        with artifact_lock(stem, "offset searches"):
            temp = stem.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))
    print(f"done in {time.monotonic()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
