"""Block A5d at k = 1: capacity pilot, then (if it passes) the registered rerun.

Same construction as src/blockA5d.py (Ren & Lim's generator, targeted
thickening, rho = 0.5) with ONE copy of the linked pair, 10,000 training points
(their per-copy base) and a 20,000-point held-out sample.  Protocol unchanged:
full-batch Adam lr 1e-3, float32, one CPU thread per run, budgets as checkpoints
of one 64k-step run.

    python -m src.blockA5d_k1 pilot                 # GELU, a=3.0, ReLU x depth {5,8} x 10 seeds
    python -m src.blockA5d_k1 stratcheck DEPTH act:seed,...   # pilot seeds, full evaluation
    python -m src.blockA5d_k1 stage1 DEPTH          # registered grid, seeds 100-119
    python -m src.blockA5d_k1 stage2 DEPTH a1,a2    # +20 seeds (120-139) at bracketing a
"""

from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import math

import numpy as np

from .blockA5d import (BUDGETS, HELDOUT_SEED_OFFSET, LR, N_HELDOUT, N_TRAIN, RHO, SQRT2,
                       WIDTH, _sample_uniform_ball, _tilde_a, _tilde_b, generate)

# Stratified held-out set: the linking neighbourhood, i.e. the 10% of each core
# closest to the other core.  An A-core point's distance to the B core depends
# only on u1 and is smallest at u1 = -1 (0.828427 = 2(sqrt2-1)); the closest 10%
# is u1 in [-1, -0.8], and symmetrically v1 in [-1, -0.8] for B.  u1 is uniform on
# [-1, 1] for a uniform point on S^2, so the band is sampled exactly: u1 uniform on
# the band, azimuth uniform.  Thickening as in the generator (targeted, rho = 0.5).
BAND = (-1.0, -0.8)
N_STRAT = 20_000                 # 10x the uniform set's density in the band
STRAT_SEED_OFFSET = 2_000_000
N_BIG = 200_000                  # 10x uniform, used only when uniform errors == 0
BIG_SEED_OFFSET = 3_000_000


def generate_stratified(num_samples, seed):
    rng = np.random.default_rng(seed)
    half = num_samples // 2
    A, B = [], []
    for _ in range(half):
        u1 = rng.uniform(*BAND); phi = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - u1 * u1)
        base = _tilde_a(np.array([u1, r * math.cos(phi), r * math.sin(phi)]))
        noise = np.zeros(5); noise[3:] = _sample_uniform_ball(2, RHO, rng)
        A.append(base + noise)
    for _ in range(half):
        v1 = rng.uniform(*BAND); phi = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - v1 * v1)
        base = _tilde_b(np.array([v1, r * math.cos(phi), r * math.sin(phi)]))
        noise = np.zeros(5); noise[:2] = _sample_uniform_ball(2, RHO, rng)
        B.append(base + noise)
    X = np.vstack([np.array(A, dtype=np.float32), np.array(B, dtype=np.float32)])
    y = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.float32)
    return X, y

RESULTS = Path(__file__).resolve().parents[1] / "results"
PILOT = RESULTS / "blockA5d_k1_pilot.csv"
RUNS = RESULTS / "blockA5d_k1_runs.csv"
K = 1
A_VALUES = (0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0)
CONTROLS = ("relu", "gelu")
STAGE1_SEEDS = range(100, 120)    # disjoint from the pilot's seeds 0-9 (registration)
STAGE2_SEEDS = range(120, 140)
N_WORKERS = int(os.environ.get("A5D_WORKERS", "9"))
FIELDS = ["act", "a", "depth", "seed", "budget", "heldout_errors", "heldout_acc",
          "strat_errors", "big_errors", "train_acc", "perfect", "wout_fro", "train_loss"]
PILOT_FIELDS = ["act", "a", "depth", "seed", "budget", "heldout_errors", "heldout_acc",
                "train_acc", "perfect", "wout_fro", "train_loss"]


def _build(act, seed, depth):
    import torch
    from torch import nn

    class FA(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, x):
            return x + self.a * torch.sin(x)

    def make_act():
        return {"relu": nn.ReLU, "gelu": nn.GELU}.get(act, lambda: FA(float(act)))()

    torch.manual_seed(seed)
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(WIDTH, WIDTH), make_act()]
    head = nn.Linear(WIDTH, 1)
    return nn.Sequential(nn.Sequential(*layers), head), head


def run_one(args):
    act, seed, depth = args[:3]
    full_eval = len(args) > 3 and args[3]
    Xst = None
    import numpy as np
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    Xtr, ytr = generate(N_TRAIN, seed, num_copies=K)
    Xho, yho = generate(N_HELDOUT, seed + HELDOUT_SEED_OFFSET, num_copies=K)
    Xtr, ytr, Xho, yho = (torch.from_numpy(v) for v in (Xtr, ytr, Xho, yho))
    model, head = _build(act, seed, depth)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    a_val = "" if act in CONTROLS else float(act)
    rows, step = [], 0
    for budget in BUDGETS:
        while step < budget:
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(Xtr).squeeze(-1), ytr).backward()
            opt.step()
            step += 1
        with torch.no_grad():
            tr = model(Xtr).squeeze(-1)
            ho = model(Xho).squeeze(-1)
            errs = int(((ho > 0).float() != yho).sum())
            row = {"act": act, "a": a_val, "depth": depth, "seed": seed, "budget": budget,
                   "heldout_errors": errs, "heldout_acc": 1 - errs / N_HELDOUT,
                   "train_acc": float(((tr > 0).float() == ytr).float().mean()),
                   "wout_fro": float(torch.linalg.norm(head.weight)),
                   "train_loss": float(F.binary_cross_entropy_with_logits(tr, ytr))}
            if full_eval:
                if Xst is None:
                    Xst, yst = (torch.from_numpy(v) for v in
                                generate_stratified(N_STRAT, seed + STRAT_SEED_OFFSET))
                st = int(((model(Xst).squeeze(-1) > 0).float() != yst).sum())
                big = ""
                if errs == 0:
                    Xb, yb = (torch.from_numpy(v) for v in
                              generate(N_BIG, seed + BIG_SEED_OFFSET, num_copies=K))
                    big = int(((model(Xb).squeeze(-1) > 0).float() != yb).sum())
                row.update({"strat_errors": st, "big_errors": big,
                            "perfect": errs == 0 and st == 0})
            else:
                row["perfect"] = errs == 0
            rows.append(row)
    return rows


def _run(path, jobs, fields=None):
    fields = fields or FIELDS
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["act"], int(r["seed"]), int(r["depth"])) for r in csv.DictReader(f)}
    jobs = [j for j in jobs if tuple(j[:3]) not in done]
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if new:
            w.writeheader()
        with Pool(N_WORKERS) as pool:
            for rows in pool.imap_unordered(run_one, jobs):
                w.writerows(rows)
                f.flush()
                r = rows[-1]
                print(f"  {r['act']:>5} depth {r['depth']} seed {r['seed']:>2}: held-out errors "
                      + " / ".join(str(x["heldout_errors"]) for x in rows), flush=True)


def pilot():
    _run(PILOT, [(act, s, d) for d in (5, 8) for act in ("gelu", "3.0", "relu") for s in range(10)],
         fields=PILOT_FIELDS)


CHECK = RESULTS / "blockA5d_k1_stratcheck.csv"


def stratcheck(depth, jobs):
    """Re-run pilot seeds (deterministic) with the stratified and 10x uniform evaluation."""
    _run(CHECK, [(act, s, depth, True) for act, s in jobs])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        pilot()
    elif cmd == "stage1":
        d = int(sys.argv[2])
        _run(RUNS, [(act, s, d, True) for act in [str(a) for a in A_VALUES] + list(CONTROLS)
                    for s in STAGE1_SEEDS])
    elif cmd == "stage2":
        d = int(sys.argv[2])
        _run(RUNS, [(act, s, d, True) for act in sys.argv[3].split(",") for s in STAGE2_SEEDS])
    elif cmd == "stratcheck":
        d = int(sys.argv[2])
        stratcheck(d, [(p.split(":")[0], int(p.split(":")[1])) for p in sys.argv[3].split(",")])
