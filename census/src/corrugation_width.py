"""B3 + B4: corrugation-strength gradient at widths 3, 4, and 6.

Reading A + flat only (the Reading-B axis is vacuous per
results/reading_b_anomaly.md), zero noise.  GELU at widths 3/4/6 carries
the gradient; tanh at widths 4/6 is the monotonic comparator that decides
generic-difficulty versus width-specificity (B4).  Every eval-0 run is
dense-verified on 100,000 fresh points from its own configuration.
"""

from __future__ import annotations

from pathlib import Path
import time
import zlib

import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .corrugation import GRID, sample
from .train import TrainingConfig, train_mlp


CONFIG_NAMES = (
    "flat", "A_a0.05", "A_a0.15", "A_paper", "A_a0.5",
    "A_embedded_f0.5", "A_f10", "A_f50", "A_f200",
)
CELLS = (
    ("gelu", 3), ("gelu", 4), ("gelu", 6),
    ("tanh", 4), ("tanh", 6),
)
SEEDS = tuple(range(30))
DEPTH = 5
TRAIN_SEED_BASE = 50_000   # same offsets as the corrugation sweep
EVAL_SEED_BASE = 60_000
DENSE_PER_CLASS = 50_000

_BY_NAME = {link.name: link for link in GRID}


def run(output_directory: Path, verbose: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(CONFIG_NAMES) * len(CELLS) * len(SEEDS)
    completed = 0
    for name in CONFIG_NAMES:
        link = _BY_NAME[name]
        for activation, width in CELLS:
            for seed in SEEDS:
                started = time.monotonic()
                train_data = sample(link, 1_000, TRAIN_SEED_BASE + seed)
                eval_data = sample(link, 1_000, EVAL_SEED_BASE + seed)
                result = train_mlp(
                    train_data, eval_data, hidden_depth=DEPTH, hidden_width=width,
                    activation=activation,  # type: ignore[arg-type]
                    config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
                )
                errors = int(round((1 - result.final_eval_accuracy) * 2_000))
                dense_err = None
                if errors == 0:
                    tag = f"cw|{name}|{activation}|{width}|{seed}"
                    dense_seed = 950_000 + zlib.crc32(tag.encode()) % 50_000
                    dense = sample(link, DENSE_PER_CLASS, dense_seed)
                    features = torch.as_tensor(dense.features, dtype=torch.float32)
                    labels = torch.as_tensor(dense.labels, dtype=torch.int64)
                    with torch.no_grad():
                        dense_err = int(
                            (result.model(features).argmax(1) != labels).sum().item()
                        )
                completed += 1
                rows.append({
                    "configuration": name,
                    "amplitude": link.amplitude,
                    "frequency": link.frequency,
                    "activation": activation,
                    "width": width,
                    "depth": DEPTH,
                    "seed": seed,
                    "eval_errors": errors,
                    "dense_errors": dense_err,
                    "regionally_separating": errors == 0 and dense_err == 0,
                    "duration_seconds": time.monotonic() - started,
                })
                if verbose and completed % 100 == 0:
                    print(f"[{completed}/{total}]", flush=True)
                if completed % 150 == 0 or completed == total:
                    _write(pd.DataFrame(rows), output_directory)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, output_directory: Path) -> None:
    stem = output_directory / "corrugation_width"
    with artifact_lock(stem, "corrugation width gradient"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    print(f"corrugation width gradient: {len(CONFIG_NAMES)*len(CELLS)*len(SEEDS)} runs", flush=True)
    run(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
