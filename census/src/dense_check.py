"""Part 2c: dense-sample verification of every separating width-3 network.

The census criterion -- 0 errors on 2,000 eval points -- is sample-level
separation.  Part 2b produced monotonic networks with 0 *train* errors and
nonzero eval errors, and the witness selection produced an eval-perfect
network that failed at 50,000 points.  So every separating width-3 run in
the project is reconstructed here (deterministically, with a hard recovery
check against its recorded accuracy) and evaluated on 100,000 fresh points
from its own link, with the minimum logit margin recorded.  A stratified
random sample of width-4 separating runs is checked the same way.
"""

from __future__ import annotations

from pathlib import Path
import time
import zlib

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .author_protocol import AuthorTrainingConfig, train_mlp_author
from .census import SweepConfig, _make_data
from .corrugation import GRID as CORRUGATION_GRID
from .corrugation import sample as corrugation_sample
from .data import linked_tori
from .parametrization import GRID as PARAM_GRID
from .parametrization import sample_link
from .winding import GRID as WINDING_GRID
from .winding import sample as winding_sample
from .train import TrainingConfig, train_mlp


DENSE_PER_CLASS = 50_000
DENSE_SEED_BASE = 900_000
WIDTH4_SAMPLE = 200
WIDTH4_SAMPLE_SEED = 424_242

_PARAM_BY_NAME = {link.name: link for link in PARAM_GRID}
_CORR_BY_NAME = {link.name: link for link in CORRUGATION_GRID}
_WINDING_BY_NAME = {link.name: link for link in WINDING_GRID}
# winding_sweep's own data seed bases (src/winding_sweep.py)
_WINDING_TRAIN_SEED_BASE = 70_000
_WINDING_EVAL_SEED_BASE = 80_000


class RecoveryDivergence(RuntimeError):
    """A reconstructed run did not reproduce its recorded accuracy."""


def _margin_errors(model, features: torch.Tensor, labels: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(features)
        correct = logits.gather(1, labels[:, None]).squeeze(1)
        wrong = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
        margin = correct - wrong
    return int((margin <= 0.0).sum().item()), float(margin.min().item())


def _standard_config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )


def _reconstruct(row: pd.Series, sweep: str):
    """Retrain one recorded run; returns (model, dense_features, dense_labels)."""

    depth, width, seed = int(row.depth), int(row.width), int(row.seed)
    # zlib.crc32 is deterministic across processes; hash() is not.
    configuration = getattr(row, "configuration", getattr(row, "parametrization", ""))
    key = f"{sweep}|{configuration}|{row.activation}|{depth}|{width}|{seed}"
    dense_seed = DENSE_SEED_BASE + zlib.crc32(key.encode()) % 50_000

    if sweep in ("width_sweep", "threshold_sweep"):
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _standard_config())
        parameter = None
        if sweep == "threshold_sweep" and not pd.isna(row.parameter):
            parameter = float(row.parameter)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=width,
            activation=row.activation,
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
            activation_parameter=parameter,
        )
        dense = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=dense_seed)
    elif sweep == "protocol_sweep":
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _standard_config())
        result = train_mlp_author(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=width,
            activation=row.activation,
            config=AuthorTrainingConfig(seed=seed),
        )
        dense = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=dense_seed)
    elif sweep == "parametrization_sweep":
        link = _PARAM_BY_NAME[row.parametrization]
        train_data = sample_link(link, 1_000, 30_000 + seed)
        eval_data = sample_link(link, 1_000, 40_000 + seed)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=width,
            activation=row.activation,
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        )
        dense = sample_link(link, DENSE_PER_CLASS, dense_seed)
    elif sweep == "corrugation_sweep":
        link = _CORR_BY_NAME[row.configuration]
        train_data = corrugation_sample(link, 1_000, 50_000 + seed)
        eval_data = corrugation_sample(link, 1_000, 60_000 + seed)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=width,
            activation=row.activation,
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        )
        dense = corrugation_sample(link, DENSE_PER_CLASS, dense_seed)
    elif sweep == "winding_sweep":
        link = _WINDING_BY_NAME[row.configuration]
        train_data = winding_sample(link, 1_000, _WINDING_TRAIN_SEED_BASE + seed)
        eval_data = winding_sample(link, 1_000, _WINDING_EVAL_SEED_BASE + seed)
        parameter = None
        if "parameter" in row and not pd.isna(row.parameter):
            parameter = float(row.parameter)
        result = train_mlp(
            train_data,
            eval_data,
            hidden_depth=depth,
            hidden_width=width,
            activation=row.activation,
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
            activation_parameter=parameter,
        )
        dense = winding_sample(link, DENSE_PER_CLASS, dense_seed)
    else:
        raise ValueError(f"unknown sweep: {sweep}")

    recorded = float(row.final_eval_accuracy)
    if result.final_eval_accuracy != recorded:
        raise RecoveryDivergence(
            f"{sweep} {row.activation} d={depth} w={width} s={seed}: "
            f"reconstructed {result.final_eval_accuracy!r}, recorded {recorded!r}"
        )
    features = torch.as_tensor(dense.features, dtype=torch.float32)
    labels = torch.as_tensor(dense.labels, dtype=torch.int64)
    return result.model, features, labels, dense_seed


def separating_rows() -> list[tuple[str, pd.Series]]:
    directory = Path(__file__).resolve().parents[1] / "results"
    out: list[tuple[str, pd.Series]] = []
    for sweep in (
        "width_sweep",
        "threshold_sweep",
        "parametrization_sweep",
        "corrugation_sweep",
        "protocol_sweep",
    ):
        frame = pd.read_parquet(directory / f"{sweep}.parquet")
        subset = frame[(frame.width == 3) & frame.perfect_eval]
        out.extend((sweep, row) for _, row in subset.iterrows())
    return out


def width4_sample_rows() -> list[tuple[str, pd.Series]]:
    directory = Path(__file__).resolve().parents[1] / "results"
    pools: list[tuple[str, pd.DataFrame]] = []
    for sweep in (
        "width_sweep",
        "threshold_sweep",
        "parametrization_sweep",
        "corrugation_sweep",
        "protocol_sweep",
    ):
        frame = pd.read_parquet(directory / f"{sweep}.parquet")
        pools.append((sweep, frame[(frame.width == 4) & frame.perfect_eval]))
    total = sum(len(pool) for _, pool in pools)
    rng = np.random.default_rng(WIDTH4_SAMPLE_SEED)
    out: list[tuple[str, pd.Series]] = []
    for sweep, pool in pools:
        share = max(1, int(round(WIDTH4_SAMPLE * len(pool) / total)))
        chosen = rng.choice(len(pool), size=min(share, len(pool)), replace=False)
        out.extend((sweep, pool.iloc[int(index)]) for index in chosen)
    return out


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = [("w3", sweep, row) for sweep, row in separating_rows()]
    work += [("w4_sample", sweep, row) for sweep, row in width4_sample_rows()]
    total = len(work)
    for index, (group, sweep, row) in enumerate(work, start=1):
        started = time.monotonic()
        model, features, labels, dense_seed = _reconstruct(row, sweep)
        errors, min_margin = _margin_errors(model, features, labels)
        rows.append(
            {
                "group": group,
                "sweep": sweep,
                "configuration": getattr(
                    row, "configuration", getattr(row, "parametrization", "baseline")
                ),
                "activation": row.activation,
                "parameter": getattr(row, "parameter", None),
                "depth": int(row.depth),
                "width": int(row.width),
                "seed": int(row.seed),
                "dense_seed": dense_seed,
                "dense_n": 2 * DENSE_PER_CLASS,
                "dense_errors": errors,
                "dense_min_margin": min_margin,
                "regionally_separating": errors == 0,
                "duration_seconds": time.monotonic() - started,
            }
        )
        if verbose and (index % 25 == 0 or errors > 0):
            print(
                f"[{index}/{total}] {sweep} {row.activation} d={int(row.depth)} "
                f"w={int(row.width)} s={int(row.seed)}: dense_errors={errors}",
                flush=True,
            )
        if index % 50 == 0 or index == total:
            _write(pd.DataFrame(rows), output_directory)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "dense_check"
    with artifact_lock(stem, "dense verification"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
