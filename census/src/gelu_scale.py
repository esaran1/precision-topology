"""Task A: does the weight-scale mechanism reach GELU?

Three arms at width 3, depth 5, baseline link, 200 seeds each:

  standard   ordinary Kaiming initialization (also supplies the standard
             scale distribution and the baseline rate at matched n)
  scaled_up  each layer's weight tensor rescaled at initialization to the
             median per-layer scale of *found* dense-verified GELU
             separators (trained endpoints), directions untouched
  scaled_down each layer rescaled to 0.3x standard (the reverse test)

Every eval-0 run is dense-verified on 100k points.  Per-layer spectral
norms are recorded for every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .data import linked_tori
from .models import MLP
from .train import TrainingConfig, seed_everything, train_mlp


DEPTH = 5
N_SEEDS = 200
DENSE_PER_CLASS = 50_000


def _config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000, n_eval_per_class=1_000,
        max_steps=2_000, learning_rate=1e-2, tube_radius=0.2,
    )


def layer_norms(model: MLP) -> list[float]:
    norms = [
        float(torch.linalg.matrix_norm(layer.weight.detach(), ord=2))
        for layer in model.hidden_layers
    ]
    norms.append(float(torch.linalg.matrix_norm(model.output_layer.weight.detach(), ord=2)))
    return norms


def found_separator_pattern() -> list[float]:
    """Median per-layer spectral norms of the found dense-verified GELU
    width-3 depth-5 baseline-link solutions (threshold sweep seeds 0, 1)."""

    patterns = []
    for seed in (0, 1):
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
        result = train_mlp(
            train_data, eval_data, hidden_depth=DEPTH, hidden_width=3,
            activation="gelu",
            config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        )
        assert result.final_eval_accuracy >= 1.0
        patterns.append(layer_norms(result.model))
    frame = pd.DataFrame(patterns)
    return [float(v) for v in frame.median(axis=0)]


def run_arm(arm: str, pattern: list[float] | None, seed_base: int,
            directory: Path) -> pd.DataFrame:
    rows = []
    for index in range(N_SEEDS):
        seed = seed_base + index
        data_seed = index  # data varies with index; init seed = seed
        train_data, eval_data, *_ = _make_data("linked_tori", data_seed, _config())
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        seed_everything(seed)
        model = MLP(3, DEPTH, 3, "gelu")
        if pattern is not None:
            with torch.no_grad():
                layers = list(model.hidden_layers) + [model.output_layer]
                for layer, target in zip(layers, pattern):
                    current = float(torch.linalg.matrix_norm(layer.weight, ord=2))
                    layer.weight.mul_(target / max(current, 1e-12))
        init_norms = layer_norms(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(2_000):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(tf), tl).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = int((model(ef).argmax(1) != el).sum().item())
        dense = None
        if errors == 0:
            sample = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=968_000 + seed)
            df = torch.as_tensor(sample.features); dl = torch.as_tensor(sample.labels)
            with torch.no_grad():
                dense = int((model(df).argmax(1) != dl).sum().item())
        final_norms = layer_norms(model)
        rows.append({
            "arm": arm, "seed": seed, "data_seed": data_seed,
            "eval_errors": errors, "dense_errors": dense,
            "separated": errors == 0 and dense == 0,
            "init_spec_product": float(pd.Series(init_norms).prod()),
            "final_spec_product": float(pd.Series(final_norms).prod()),
            "final_norms": json.dumps([round(v, 4) for v in final_norms]),
        })
        if errors == 0 or (index + 1) % 50 == 0:
            print(f"{arm} [{index+1}/{N_SEEDS}] eval={errors} dense={dense}", flush=True)
        if (index + 1) % 50 == 0 or index + 1 == N_SEEDS:
            _write(pd.DataFrame(rows), directory, f"gelu_scale_{arm}")
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, directory: Path, stem_name: str) -> None:
    stem = directory / stem_name
    with artifact_lock(stem, stem_name):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    import sys
    directory = Path(__file__).resolve().parents[1] / "results"
    arm = sys.argv[1]
    if arm == "standard":
        run_arm("standard", None, 0, directory)
    elif arm == "scaled_up":
        pattern = found_separator_pattern()
        print(f"separator pattern (per-layer spectral norms): {[round(v,3) for v in pattern]}", flush=True)
        run_arm("scaled_up", pattern, 1_000, directory)
    elif arm == "scaled_down":
        # 0.3x the standard Kaiming per-layer norms, measured on a fresh init
        seed_everything(0)
        reference = MLP(3, DEPTH, 3, "gelu")
        pattern = [0.3 * v for v in layer_norms(reference)]
        run_arm("scaled_down", pattern, 2_000, directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
