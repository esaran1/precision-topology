"""2a: required amplification by bisection; 2b: achievable amplification
and the scaled-initialization intervention.

Registered in ``results/amplification_prediction.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .census import SweepConfig, _make_data
from .data import linked_tori
from .models import MLP
from .train import seed_everything


DENSE_PER_CLASS = 50_000
CONTINUATION_SEEDS = range(10)


def _config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000, n_eval_per_class=1_000,
        max_steps=2_000, learning_rate=1e-2, tube_radius=0.2,
    )


# ------------------------------------------------------------------ recipes

def sin_constants(a: float) -> tuple[float, float, float]:
    """(t_star, shrink, yz_slope) for family A."""

    t_star = math.pi - math.acos(1.0 / a)
    window = 2.0 * math.acos(1.0 / a)
    shrink = 0.33 * window / 2.2
    return t_star, shrink, 1.0 + a


def pwl_constants(alpha: float) -> tuple[float, float, float]:
    """(t_star, shrink, yz_slope) for family B: fold at the kink, x>0 slope 1."""

    return 0.0, 0.3, 1.0


def activation_fn(family: str, parameter: float):
    if family == "sin":
        return lambda v: v + parameter * torch.sin(v)
    return lambda v: torch.where(v >= 0.0, v, parameter * v)


def build_and_train(
    family: str, parameter: float, amplification: float, seed: int,
    steps: int = 3_000,
) -> tuple[int, int | None]:
    """One construction attempt; returns (eval_errors, dense_errors|None)."""

    f = activation_fn(family, parameter)
    t_star, shrink, yz_slope = (
        sin_constants(parameter) if family == "sin" else pwl_constants(parameter)
    )
    f_t_star = float(f(torch.tensor(t_star)))
    f_zero = float(f(torch.tensor(0.0)))
    sign = 1.0 if family == "sin" else -1.0  # max fold vs min fold

    class Constructed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.ModuleList(torch.nn.Linear(3, 3) for _ in range(2))
            self.head = torch.nn.Linear(3, 2)

        def forward(self, p):
            z1 = f(t_star + shrink * (p[:, 0] - 1.0))
            z2 = f(shrink * p[:, 1])
            z3 = f(shrink * p[:, 2])
            h = torch.stack([
                sign * amplification * (z1 - f_t_star),
                (z2 - f_zero) / (shrink * yz_slope),
                (z3 - f_zero) / (shrink * yz_slope),
            ], dim=1)
            for layer in self.hidden:
                h = torch.tanh(layer(h))
            return self.head(h)

    train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
    tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
    ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
    seed_everything(seed)
    model = Constructed()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(tf), tl).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        errors = int((model(ef).argmax(1) != el).sum().item())
    dense = None
    if errors == 0:
        sample = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=963_000 + seed)
        df = torch.as_tensor(sample.features); dl = torch.as_tensor(sample.labels)
        with torch.no_grad():
            dense = int((model(df).argmax(1) != dl).sum().item())
    return errors, dense


def succeeds(family: str, parameter: float, amplification: float) -> bool:
    for seed in CONTINUATION_SEEDS:
        errors, dense = build_and_train(family, parameter, amplification, seed)
        if errors == 0 and dense == 0:
            return True
    return False


def required_amplification(
    family: str, parameter: float, low: float = 1.0, high: float = 20_000.0,
    iterations: int = 7,
) -> tuple[float | None, float | None]:
    """Log-bisection for the smallest working amplification.

    Returns (last failing, first succeeding); (None, None) if even ``high``
    fails, (low band) if even ``low`` works.
    """

    if not succeeds(family, parameter, high):
        return high, None
    if succeeds(family, parameter, low):
        return None, low
    log_low, log_high = math.log10(low), math.log10(high)
    for _ in range(iterations):
        mid = 10 ** ((log_low + log_high) / 2)
        if succeeds(family, parameter, mid):
            log_high = math.log10(mid)
        else:
            log_low = math.log10(mid)
    return 10 ** log_low, 10 ** log_high


def run_2a(directory: Path) -> None:
    targets = [
        ("sin", 1.02), ("sin", 1.05), ("sin", 1.09), ("sin", 1.25),
        ("pwl", -0.05), ("pwl", -0.11), ("pwl", -0.22), ("pwl", -0.25),
    ]
    rows = []
    for family, parameter in targets:
        failing, succeeding = required_amplification(family, parameter)
        rows.append({
            "family": family, "parameter": parameter,
            "last_failing_amp": failing, "first_succeeding_amp": succeeding,
        })
        print(f"{family} {parameter}: A_req in ({failing}, {succeeding}]", flush=True)
        _write(pd.DataFrame(rows), directory, "required_amplification")


def run_2b_intervention(directory: Path, a: float = 1.02, seeds: int = 40) -> None:
    """Scaled initialization at a=1.02: standard MLP, standard training, but
    layers 1-2 initialized at the construction's scale pattern with random
    directions."""

    t_star, shrink, _ = sin_constants(a)
    xs = np.linspace(t_star, t_star + shrink * 1.2, 20_001)
    f = xs + a * np.sin(xs)
    depth = float(f.max() - f.min())
    amplification = 3.2 / depth
    rows = []
    for seed in range(seeds):
        train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        seed_everything(seed)
        model = MLP(3, 4, 3, "sin_family", activation_parameter=a)
        with torch.no_grad():
            # layer 1: shrink-scale weights, bias placing coordinates near t*
            model.hidden_layers[0].weight.mul_(shrink / model.hidden_layers[0].weight.abs().mean())
            model.hidden_layers[0].bias.normal_(mean=t_star, std=0.1 * shrink)
            # layer 2: amplification-scale weights, random directions
            model.hidden_layers[1].weight.mul_(amplification / model.hidden_layers[1].weight.abs().mean())
            model.hidden_layers[1].bias.normal_(mean=0.0, std=1.0)
            model.hidden_layers[1].bias.sub_(model.hidden_layers[1].weight.sum(dim=1) * (t_star + a * math.sin(t_star)) / 3.0)
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
            sample = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=964_000 + seed)
            df = torch.as_tensor(sample.features); dl = torch.as_tensor(sample.labels)
            with torch.no_grad():
                dense = int((model(df).argmax(1) != dl).sum().item())
        rows.append({"seed": seed, "eval_errors": errors, "dense_errors": dense,
                     "separated": errors == 0 and dense == 0})
        if errors == 0 or seed % 10 == 9:
            print(f"intervention seed {seed}: eval={errors} dense={dense}", flush=True)
    _write(pd.DataFrame(rows), directory, "scaled_init_intervention")


def _write(frame: pd.DataFrame, directory: Path, stem_name: str) -> None:
    stem = directory / stem_name
    with artifact_lock(stem, stem_name):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    run_2a(directory)
    run_2b_intervention(directory)
    print("done", flush=True)


if __name__ == "__main__":
    main()
