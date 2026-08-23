"""Task B Part 3: intermediate settings between 1D folds and 3D links.

2D annulus (width 2): class 1 = annulus 1.2 <= r <= 2.0; class 0 = disk
r <= 0.8 union ring 2.4 <= r <= 3.0.  The three-region structure forces two
sign changes along every ray — a fold in radius — where a plain
annulus-versus-disk task would need only one and no fold at all.

3D nested shells (width 3): class 1 = shell 1.0 <= r <= 1.4; class 0 =
ball r <= 0.6 union shell 1.8 <= r <= 2.2.  Nested, not linked — isolates
"requires a fold" from "requires unlinking" (JMLR concentric-shell
precedent).

Monotonic activations are expected to fail (radial profile of a
width-d monotone network cannot make two sign changes); this is measured
rather than asserted, and any monotonic solve is reported as a surprise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .models import MLP
from .train import seed_everything


def sample_radial(dimension: int, bands_zero: list[tuple[float, float]],
                  band_one: tuple[float, float], n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)

    def shell(lo: float, hi: float, n: int) -> np.ndarray:
        direction = rng.normal(size=(n, dimension))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = (rng.uniform(lo**dimension, hi**dimension, n)) ** (1.0 / dimension)
        return direction * radius[:, None]

    per = n_per_class // len(bands_zero)
    zero = np.concatenate([shell(lo, hi, per) for lo, hi in bands_zero])
    one = shell(*band_one, n_per_class)
    x = np.concatenate([zero, one]).astype(np.float32)
    y = np.concatenate([np.zeros(len(zero)), np.ones(len(one))]).astype(np.int64)
    return torch.tensor(x), torch.tensor(y)


# Margins widened and depth/steps raised after a first attempt at depth 2 /
# 2,000 steps produced zero solves for every activation including GELU and
# sin(3.0) (best 19/600) — a design too hard to resolve threshold structure;
# the piloted design below solves for GELU and clearly separates tanh.
SETTINGS = {
    "annulus2d": dict(dimension=2, width=2,
                      bands_zero=[(0.0, 0.7), (2.6, 3.2)], band_one=(1.3, 2.0)),
    "shells3d": dict(dimension=3, width=3,
                     bands_zero=[(0.0, 0.55), (1.9, 2.4)], band_one=(0.95, 1.5)),
}

GRID = (
    [("sin_family", a) for a in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)]
    + [("pwl_family", -0.05), ("pwl_family", -0.25)]
    + [("tanh", None), ("relu", None), ("gelu", None)]
)


def run_one(setting: str, name: str, parameter: float | None, seed: int,
            depth: int = 4, steps: int = 4_000) -> dict:
    spec = SETTINGS[setting]
    x, y = sample_radial(spec["dimension"], spec["bands_zero"], spec["band_one"], 300, seed)
    xe, ye = sample_radial(spec["dimension"], spec["bands_zero"], spec["band_one"], 300, 700_000 + seed)
    seed_everything(seed)
    model = MLP(spec["dimension"], depth, spec["width"], name,  # type: ignore[arg-type]
                activation_parameter=parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        errors = int((model(xe).argmax(1) != ye).sum().item())
    dense_ok = False
    if errors == 0:
        xd, yd = sample_radial(spec["dimension"], spec["bands_zero"], spec["band_one"], 5_000, 970_000 + seed)
        with torch.no_grad():
            dense_ok = int((model(xd).argmax(1) != yd).sum().item()) == 0
    return {"setting": setting, "activation": name, "parameter": parameter,
            "seed": seed, "eval_errors": errors, "solved": errors == 0 and dense_ok}


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    rows = []
    for setting in SETTINGS:
        for name, parameter in GRID:
            for seed in range(100):
                rows.append(run_one(setting, name, parameter, seed))
            solved = sum(r["solved"] for r in rows
                         if r["setting"] == setting and r["activation"] == name
                         and r["parameter"] == parameter)
            print(f"{setting} {name}({parameter}): {solved}/100", flush=True)
            frame = pd.DataFrame(rows)
            stem = directory / "fold_settings"
            with artifact_lock(stem, "intermediate settings"):
                temp = stem.with_suffix(".csv.tmp")
                frame.to_csv(temp, index=False)
                temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
