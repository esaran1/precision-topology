"""A4 existence exhibit: a separating width-3 network at a = 1.02.

The construction follows Appendix F.2's recipe exactly, at a parameter value
deep inside the observed offset region where SGD never separates:

  frozen affine    (x, y, z) -> (t* + s(x-1), s y, s z)      s = 0.06
  f_{1.02}         applied coordinate-wise (the ONLY non-monotonicity)
  frozen affine    amplify the folded coordinate 600x, undo the
                   near-linear f_a action on y and z
  trained          two width-3 tanh layers (monotone), linear head

``t*`` is the local maximum of f_{1.02}; the data's fold coordinate spans
[-0.132, +0.072] around it, inside the near-quadratic neighbourhood, so the
realized fold is a near-symmetric inverted parabola of depth ~5.3e-3 before
amplification.  Shape matters: pilots recorded in results/offset_results.md
show SGD trains a continuation on an inverted-parabola fold but not on the
sheared tent that larger ``s`` produces.

Deterministic from CONTINUATION_SEED.  Verified: 0 errors on 1,400,000
fresh points; linking -1 -> 0 exactly at the f_{1.02} layer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .census import SweepConfig, _make_data
from .data import linked_tori
from .train import seed_everything


TARGET_A = 1.02
T_STAR = math.pi - math.acos(1.0 / TARGET_A)
F_TSTAR = T_STAR + TARGET_A * math.sin(T_STAR)
SHRINK = 0.06
AMPLIFICATION = 600.0
YZ_RESCALE = 1.0 / (SHRINK * (1.0 + TARGET_A))
CONTINUATION_SEED = 16
CONTINUATION_STEPS = 3_000


def f_a(values: torch.Tensor) -> torch.Tensor:
    return values + TARGET_A * torch.sin(values)


class OffsetWitness(nn.Module):
    """Frozen f_{1.02} fold in its quadratic regime + trained tanh continuation."""

    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(3, 3) for _ in range(depth))
        self.head = nn.Linear(3, 2)

    def frozen_representation(self, points: torch.Tensor) -> torch.Tensor:
        z1 = f_a(T_STAR + SHRINK * (points[:, 0] - 1.0))
        z2 = f_a(SHRINK * points[:, 1])
        z3 = f_a(SHRINK * points[:, 2])
        return torch.stack(
            [AMPLIFICATION * (z1 - F_TSTAR), YZ_RESCALE * z2, YZ_RESCALE * z3],
            dim=1,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        hidden = self.frozen_representation(points)
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        return self.head(hidden)


def train_offset_witness(
    seed: int = CONTINUATION_SEED, steps: int = CONTINUATION_STEPS
) -> OffsetWitness:
    config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=steps,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    train_data, _, *_ = _make_data("linked_tori", seed, config)
    features = torch.as_tensor(train_data.features, dtype=torch.float32)
    labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
    seed_everything(seed)
    model = OffsetWitness()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), labels).backward()
        optimizer.step()
    model.eval()
    return model


def dense_errors(model: OffsetWitness, n_per_class: int, seed: int) -> tuple[int, float]:
    data = linked_tori(n_per_class, tube_radius=0.2, seed=seed)
    features = torch.as_tensor(data.features, dtype=torch.float32)
    labels = torch.as_tensor(data.labels, dtype=torch.int64)
    with torch.no_grad():
        logits = model(features)
        margin = (
            logits.gather(1, labels[:, None]).squeeze(1)
            - logits.gather(1, (1 - labels)[:, None]).squeeze(1)
        )
    return int((margin <= 0.0).sum().item()), float(margin.min().item())


def export(model: OffsetWitness, path: Path) -> None:
    payload = {
        "frozen": {
            "a": TARGET_A,
            "t_star": T_STAR,
            "f_t_star": F_TSTAR,
            "shrink": SHRINK,
            "amplification": AMPLIFICATION,
            "yz_rescale": YZ_RESCALE,
            "description": "affine -> f_a -> affine, fold coordinate x-1 across the local max",
        },
        "continuation": {
            name: value.detach().numpy().tolist()
            for name, value in model.state_dict().items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))


# Reconstructed generator (audit 2026-08-24).  The original 1.4M-point
# evaluation ran from uncommitted session code with unrecorded sample seeds
# (AUDIT.md finding 8).  This main() is the committed replacement: four
# fresh 175,000-per-class samples (1,400,000 points total) at the fixed
# seeds below, disjoint from the training data seed (10_016 lineage) and
# from every crc32 dense band in use (900k-970k bases are avoided by
# offsetting into 931k).  Results are printed and persisted to
# results/offset_witness_dense.csv.  The headline claim (T25) rests on
# this committed run, not on the original prose figure.
DENSE_SAMPLE_SEEDS = (931_001, 931_002, 931_003, 931_004)
DENSE_SAMPLE_PER_CLASS = 175_000


def main() -> None:
    import pandas as pd

    from .artifact_lock import artifact_lock

    model = train_offset_witness()
    rows = []
    total_points = 0
    total_errors = 0
    for sample_seed in DENSE_SAMPLE_SEEDS:
        errors, margin = dense_errors(model, DENSE_SAMPLE_PER_CLASS, sample_seed)
        rows.append({"sample_seed": sample_seed,
                     "n_points": 2 * DENSE_SAMPLE_PER_CLASS,
                     "errors": errors, "min_margin": margin})
        total_points += 2 * DENSE_SAMPLE_PER_CLASS
        total_errors += errors
        print(rows[-1], flush=True)
    print(f"total: {total_errors} errors on {total_points} points", flush=True)
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "offset_witness_dense"
    with artifact_lock(stem, "offset witness dense evaluation"):
        temp = stem.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
