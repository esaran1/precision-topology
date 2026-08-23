"""Parts 1a-1c: basin radius, initialization distance, path barriers.

Protocol registered in ``results/basin_prediction.md`` before this ran.

Solutions:
- constructed at a = 1.02, 1.05, 1.10: pure ``sin_family(a)`` depth-4
  networks; layers 1-2 hold the F.2 construction (shrink into the fold
  window, amplify), layers 3-4 + head trained with the construction frozen;
  the whole state is then a standard MLP state_dict and every parameter is
  trainable in the basin experiments.
- found at a = 1.09, 1.10, 1.25, 2.0, 3.0: deterministic reconstructions
  of dense-verified depth-5 sweep runs.
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
from .train import TrainingConfig, seed_everything, train_mlp


EPSILONS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
NOISE_SEEDS = tuple(range(20))
RECOVERY_STEPS = 2_000
DENSE_PER_CLASS = 50_000

CONSTRUCTED = {1.02: 9, 1.05: None, 1.10: None}  # continuation seed; None = search
FOUND = {1.09: (5, 4), 1.10: (5, 4), 1.25: (5, 0), 2.00: (5, 0), 3.00: (5, 0)}


def _config() -> SweepConfig:
    return SweepConfig(
        n_train_per_class=1_000, n_eval_per_class=1_000,
        max_steps=2_000, learning_rate=1e-2, tube_radius=0.2,
    )


def construction_constants(a: float) -> tuple[float, float, float, float]:
    """(t_star, shrink, amplification, yz_rescale) via the generalized recipe."""

    t_star = math.pi - math.acos(1.0 / a)
    window = 2.0 * math.acos(1.0 / a)
    shrink = 0.33 * window / 2.2  # left extent stays at ~33% of the window
    # realized fold depth over the right extent, numerically
    right = min(shrink * 1.2, 0.999 * window)
    xs = np.linspace(t_star, t_star + right, 20_001)
    f = xs + a * np.sin(xs)
    depth = float(f.max() - f.min())
    amplification = 3.2 / depth
    yz_rescale = 1.0 / (shrink * (1.0 + a))
    return t_star, shrink, amplification, yz_rescale


def build_constructed(a: float, continuation_seed: int, steps: int = 3_000) -> MLP:
    t_star, shrink, amplification, yz_rescale = construction_constants(a)
    f_t_star = t_star + a * math.sin(t_star)
    seed_everything(continuation_seed)
    model = MLP(3, 4, 3, "sin_family", activation_parameter=a)
    with torch.no_grad():
        model.hidden_layers[0].weight.copy_(torch.diag(torch.tensor([shrink] * 3)))
        model.hidden_layers[0].bias.copy_(torch.tensor([t_star - shrink, 0.0, 0.0]))
        model.hidden_layers[1].weight.copy_(
            torch.diag(torch.tensor([amplification, yz_rescale, yz_rescale]))
        )
        model.hidden_layers[1].bias.copy_(
            torch.tensor([-amplification * f_t_star, 0.0, 0.0])
        )
    frozen = {
        id(p)
        for layer in model.hidden_layers[:2]
        for p in layer.parameters()
    }
    trainable = [p for p in model.parameters() if id(p) not in frozen]
    train_data, eval_data, *_ = _make_data("linked_tori", continuation_seed, _config())
    tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(tf), tl).backward()
        optimizer.step()
    model.eval()
    return model


def build_found(a: float, depth: int, seed: int) -> MLP:
    train_data, eval_data, *_ = _make_data("linked_tori", seed, _config())
    result = train_mlp(
        train_data, eval_data, hidden_depth=depth, hidden_width=3,
        activation="sin_family",
        config=TrainingConfig(seed=seed, max_steps=2_000, learning_rate=1e-2),
        activation_parameter=a,
    )
    if result.final_eval_accuracy < 1.0:
        raise RuntimeError(f"found solution a={a} d={depth} s={seed} did not reproduce")
    return result.model


def _dense_ok(model: MLP, tag_seed: int) -> bool:
    dense = linked_tori(DENSE_PER_CLASS, tube_radius=0.2, seed=962_000 + tag_seed)
    features = torch.as_tensor(dense.features); labels = torch.as_tensor(dense.labels)
    with torch.no_grad():
        return int((model(features).argmax(1) != labels).sum().item()) == 0


def verify_solution(model: MLP, data_seed: int) -> bool:
    _, eval_data, *_ = _make_data("linked_tori", data_seed, _config())
    features = torch.as_tensor(eval_data.features); labels = torch.as_tensor(eval_data.labels)
    with torch.no_grad():
        errors = int((model(features).argmax(1) != labels).sum().item())
    return errors == 0 and _dense_ok(model, data_seed)


def perturb_and_recover(
    model: MLP, a: float, data_seed: int, epsilon: float, noise_seed: int
) -> bool:
    clone = MLP(3, model.hidden_depth, 3, "sin_family", activation_parameter=a)
    clone.load_state_dict(model.state_dict())
    generator = torch.Generator().manual_seed(noise_seed)
    with torch.no_grad():
        for parameter in clone.parameters():
            rms = float(parameter.pow(2).mean().sqrt().item())
            noise = torch.randn(parameter.shape, generator=generator) * epsilon * max(rms, 1e-12)
            parameter.add_(noise)
    train_data, eval_data, *_ = _make_data("linked_tori", data_seed, _config())
    tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
    ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
    optimizer = torch.optim.Adam(clone.parameters(), lr=1e-2)
    clone.train()
    for _ in range(RECOVERY_STEPS):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(clone(tf), tl).backward()
        optimizer.step()
    clone.eval()
    with torch.no_grad():
        errors = int((clone(ef).argmax(1) != el).sum().item())
    return errors == 0 and _dense_ok(clone, 970_000 + noise_seed)


def initialization_distances(model: MLP, a: float, n: int = 200) -> pd.DataFrame:
    solution = torch.cat([p.detach().flatten() for p in model.parameters()])
    solution_norm = float(solution.norm().item())
    rows = []
    for seed in range(n):
        torch.manual_seed(800_000 + seed)
        init = MLP(3, model.hidden_depth, 3, "sin_family", activation_parameter=a)
        vector = torch.cat([p.detach().flatten() for p in init.parameters()])
        raw = float((vector - solution).norm().item())
        rows.append({
            "seed": seed,
            "raw_distance": raw,
            "normalized_distance": raw / solution_norm,
            "init_norm": float(vector.norm().item()),
        })
    return pd.DataFrame(rows)


def interpolation_profile(
    model: MLP, a: float, data_seed: int, n_inits: int = 20, n_points: int = 21
) -> pd.DataFrame:
    train_data, *_ = _make_data("linked_tori", data_seed, _config())
    features = torch.as_tensor(train_data.features); labels = torch.as_tensor(train_data.labels)
    solution = [p.detach().clone() for p in model.parameters()]
    probe = MLP(3, model.hidden_depth, 3, "sin_family", activation_parameter=a)
    rows = []
    for init_seed in range(n_inits):
        torch.manual_seed(810_000 + init_seed)
        start = [
            p.detach().clone()
            for p in MLP(3, model.hidden_depth, 3, "sin_family", activation_parameter=a).parameters()
        ]
        for index in range(n_points):
            alpha = index / (n_points - 1)
            with torch.no_grad():
                for target, s0, s1 in zip(probe.parameters(), start, solution):
                    target.copy_((1 - alpha) * s0 + alpha * s1)
                loss = float(F.cross_entropy(probe(features), labels).item())
            rows.append({"init_seed": init_seed, "alpha": alpha, "loss": loss})
    return pd.DataFrame(rows)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    solutions: list[tuple[float, str, int, MLP]] = []
    for a, seed in CONSTRUCTED.items():
        if seed is None:
            for candidate in range(40):
                model = build_constructed(a, candidate)
                if verify_solution(model, candidate):
                    seed = candidate
                    break
            else:
                print(f"NO constructed solution found at a={a}", flush=True)
                continue
        else:
            model = build_constructed(a, seed)
            assert verify_solution(model, seed), f"constructed a={a} seed={seed} failed"
        solutions.append((a, "constructed", seed, model))
        print(f"constructed a={a} seed={seed} verified", flush=True)
    for a, (depth, seed) in FOUND.items():
        model = build_found(a, depth, seed)
        solutions.append((a, "found", seed, model))
        print(f"found a={a} d={depth} s={seed} verified", flush=True)

    basin_rows, dist_frames, profile_frames = [], [], []
    for a, kind, data_seed, model in solutions:
        for epsilon in EPSILONS:
            recovered = 0
            for noise_seed in NOISE_SEEDS:
                recovered += perturb_and_recover(model, a, data_seed, epsilon, noise_seed)
            basin_rows.append({
                "a": a, "kind": kind, "depth": model.hidden_depth,
                "data_seed": data_seed, "epsilon": epsilon,
                "recovered": recovered, "n": len(NOISE_SEEDS),
            })
            print(f"a={a} {kind} eps={epsilon}: {recovered}/{len(NOISE_SEEDS)}", flush=True)
            _write(pd.DataFrame(basin_rows), directory, "basin_recovery")
        distances = initialization_distances(model, a)
        distances["a"] = a; distances["kind"] = kind
        dist_frames.append(distances)
        profile = interpolation_profile(model, a, data_seed)
        profile["a"] = a; profile["kind"] = kind
        profile_frames.append(profile)
    _write(pd.concat(dist_frames, ignore_index=True), directory, "basin_distances")
    _write(pd.concat(profile_frames, ignore_index=True), directory, "basin_profiles")
    print("done", flush=True)


def _write(frame: pd.DataFrame, directory: Path, stem_name: str) -> None:
    stem = directory / stem_name
    with artifact_lock(stem, stem_name):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
