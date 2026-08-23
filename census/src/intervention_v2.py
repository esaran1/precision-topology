"""2b intervention, corrected implementation.

v1 initialized all three first-layer units at the fold point, folding every
coordinate and destroying the pass-through information; 0/40 (recorded).
The construction's scale *pattern* is one fold unit plus two pass units.
v2 implements exactly that pattern with random directions within it:

  layer 1: unit 0 = shrink-scale random weights, bias ~ N(t*, 0.1 shrink)
           units 1,2 = shrink-scale random weights, bias ~ N(0, shrink)
  layer 2: column 0 (from the fold unit) scaled to the required
           amplification; other columns at pass-through scale; biases
           cancel the amplified f(t*) offset
  layers 3-4, head: standard initialization

Everything is then trained by standard Adam — the intervention touches
only the initialization.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .amplification import _write, sin_constants
from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .data import linked_tori
from .models import MLP
from .train import seed_everything


def run(a: float = 1.02, seeds: int = 40) -> pd.DataFrame:
    t_star, shrink, yz_slope = sin_constants(a)
    f_t_star = t_star + a * math.sin(t_star)
    xs = np.linspace(t_star, t_star + shrink * 1.2, 20_001)
    fold = xs + a * np.sin(xs)
    amplification = 3.2 / float(fold.max() - fold.min())
    config = SweepConfig(
        n_train_per_class=1_000, n_eval_per_class=1_000,
        max_steps=2_000, learning_rate=1e-2, tube_radius=0.2,
    )
    rows = []
    for seed in range(seeds):
        train_data, eval_data, *_ = _make_data("linked_tori", seed, config)
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        seed_everything(seed)
        model = MLP(3, 4, 3, "sin_family", activation_parameter=a)
        with torch.no_grad():
            w1 = model.hidden_layers[0].weight
            w1.mul_(shrink / w1.abs().mean())
            model.hidden_layers[0].bias[0] = t_star + 0.1 * shrink * torch.randn(())
            model.hidden_layers[0].bias[1] = shrink * torch.randn(())
            model.hidden_layers[0].bias[2] = shrink * torch.randn(())
            w2 = model.hidden_layers[1].weight
            base = w2.abs().mean()
            w2[:, 0].mul_(amplification / base)
            w2[:, 1:].mul_((1.0 / (shrink * yz_slope)) / base)
            model.hidden_layers[1].bias.copy_(-w2[:, 0] * f_t_star)
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
            sample = linked_tori(50_000, tube_radius=0.2, seed=965_000 + seed)
            df = torch.as_tensor(sample.features); dl = torch.as_tensor(sample.labels)
            with torch.no_grad():
                dense = int((model(df).argmax(1) != dl).sum().item())
        rows.append({"seed": seed, "eval_errors": errors, "dense_errors": dense,
                     "separated": errors == 0 and dense == 0})
        if errors == 0 or seed % 10 == 9:
            print(f"v2 seed {seed}: eval={errors} dense={dense}", flush=True)
    frame = pd.DataFrame(rows)
    _write(frame, Path(__file__).resolve().parents[1] / "results", "scaled_init_v2")
    return frame


def main() -> None:
    frame = run()
    print(f"separations: {int(frame.separated.sum())}/{len(frame)}", flush=True)


if __name__ == "__main__":
    main()
