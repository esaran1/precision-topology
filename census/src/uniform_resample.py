"""1f: volume-uniform re-verification of dense-verified separations.

``linked_tori`` samples each solid torus uniformly in (theta, cross-
section area), which is not volume-uniform: the volume element carries
(R + rho*cos(phi)), so sampling density per unit volume is
R/(R + rho*cos(phi)) times uniform — +25% at the inner equator, -17% at
the outer (r/R = 0.2), max/min ratio exactly 1.5.  Coverage is complete
(density bounded below at 0.833x uniform), so zero-error checks remain
evidence of regional separation; this module removes even that caveat
for the exposed cells by re-verifying under an exactly volume-uniform
sampler (rejection: accept with probability (R + rho*cos(phi))/(R + r)).

Cells re-verified: every width-3 dense survivor from the BASELINE-geometry
sweeps (width, threshold, protocol), the Part 2a witness, and the a = 1.02
offset witness.  Parametrization- and corrugation-sweep survivors are
excluded by design: their links are deformed geometries and this module's
sampler generates baseline tori, so evaluating those models here would
test them on the wrong dataset (a first draft did exactly that; those
rows were discarded).  A volume-uniform check for deformed links needs
per-family samplers with their own Jacobians and is recorded as future
work, not silently skipped.

For any run with nonzero uniform errors, a fresh area-uniform 100k
control sample is also evaluated: matching error counts there indicate
the documented flip-prone band (sample noise near 1-in-100k margins,
dense_check.md), not a sampler-shape effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .data import Dataset

RESULTS = Path(__file__).resolve().parents[1] / "results"
UNIFORM_SEED_BASE = 975_000


def linked_tori_uniform(n_per_class: int, tube_radius: float = 0.2,
                        seed: int = 0, major_radius: float = 1.0) -> Dataset:
    """Volume-uniform version of data.linked_tori (same tori, same labels)."""

    rng = np.random.default_rng(seed)
    features = []
    for class_index in range(2):
        needed = n_per_class
        chunks = []
        while needed > 0:
            m = int(needed * 1.6) + 16
            theta = rng.uniform(0.0, 2.0 * np.pi, m)
            phi = rng.uniform(0.0, 2.0 * np.pi, m)
            rho = tube_radius * np.sqrt(rng.uniform(0.0, 1.0, m))
            accept = rng.uniform(0.0, 1.0, m) < (
                (major_radius + rho * np.cos(phi)) / (major_radius + tube_radius))
            theta, phi, rho = theta[accept][:needed], phi[accept][:needed], rho[accept][:needed]
            cosine, sine = np.cos(theta), np.sin(theta)
            if class_index == 0:
                core = major_radius * np.column_stack((cosine, sine, np.zeros_like(theta)))
                normal = np.column_stack((cosine, sine, np.zeros_like(theta)))
                binormal = np.broadcast_to(np.array([0.0, 0.0, 1.0]), core.shape)
            else:
                core = major_radius * np.column_stack((1.0 + cosine, np.zeros_like(theta), sine))
                normal = np.column_stack((cosine, np.zeros_like(theta), sine))
                binormal = np.broadcast_to(np.array([0.0, 1.0, 0.0]), core.shape)
            offset = (rho * np.cos(phi))[:, None] * normal + (rho * np.sin(phi))[:, None] * binormal
            chunks.append(core + offset)
            needed -= len(theta)
        features.append(np.concatenate(chunks))
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    return Dataset(features=x, labels=y)


def _errors(model, features: np.ndarray, labels: np.ndarray) -> int:
    with torch.no_grad():
        logits = model(torch.tensor(features))
        return int((logits.argmax(dim=1) != torch.tensor(labels)).sum().item())


def main() -> None:
    import zlib

    from .dense_check import _reconstruct, separating_rows

    def norm_param(value) -> str:
        return "" if value is None or (isinstance(value, float) and pd.isna(value)) else f"{float(value):.6g}"

    frame = pd.read_csv(RESULTS / "dense_check.csv")
    survivors = frame[(frame.group == "w3") & frame.regionally_separating]
    keys = {(r.sweep, str(r.configuration), r.activation, norm_param(r.parameter),
             int(r.depth), int(r.seed)) for r in survivors.itertuples()}
    rows = []
    for sweep, row in separating_rows():
        if sweep not in ("width_sweep", "threshold_sweep", "protocol_sweep"):
            continue  # deformed-link geometries: see module docstring
        configuration = str(getattr(row, "configuration",
                                    getattr(row, "parametrization", "baseline")))
        parameter = norm_param(getattr(row, "parameter", None))
        if (sweep, configuration, row.activation, parameter,
                int(row.depth), int(row.seed)) not in keys:
            continue
        model, *_ = _reconstruct(row, sweep)
        key = f"uniform|{sweep}|{configuration}|{row.activation}|{row.depth}|{row.width}|{row.seed}"
        seed = UNIFORM_SEED_BASE + zlib.crc32(key.encode()) % 20_000
        data = linked_tori_uniform(50_000, seed=seed)
        errors = _errors(model, data.features, data.labels)
        control_errors = -1
        if errors > 0:
            from .data import linked_tori
            control = linked_tori(50_000, seed=seed + 60_000)
            control_errors = _errors(model, control.features, control.labels)
        rows.append({"sweep": sweep, "configuration": configuration,
                     "activation": row.activation, "parameter": parameter,
                     "depth": row.depth, "width": row.width, "seed": row.seed,
                     "uniform_seed": seed, "uniform_errors": errors,
                     "area_uniform_control_errors": control_errors})
        print(rows[-1], flush=True)
        stem = RESULTS / "uniform_resample"
        with artifact_lock(stem, "volume-uniform resample"):
            temp = stem.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))

    from .offset_witness import train_offset_witness
    from .witness import train_witness

    witness = train_witness()
    data = linked_tori_uniform(500_000, seed=UNIFORM_SEED_BASE)
    print("witness (2a) uniform 1M:", _errors(witness, data.features, data.labels), flush=True)
    offset = train_offset_witness()
    data = linked_tori_uniform(500_000, seed=UNIFORM_SEED_BASE + 1)
    errors = _errors(offset, data.features, data.labels)
    print("offset witness uniform 1M:", errors, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
