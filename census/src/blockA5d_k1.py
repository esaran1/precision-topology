"""Block A5d at k = 1: capacity pilot, then (if it passes) the registered rerun.

Same construction as src/blockA5d.py (Ren & Lim's generator, targeted
thickening, rho = 0.5) with ONE copy of the linked pair, 10,000 training points
(their per-copy base) and a 20,000-point held-out sample.  Protocol unchanged:
full-batch Adam lr 1e-3, float32, one CPU thread per run, budgets as checkpoints
of one 64k-step run.

    python -m src.blockA5d_k1 pilot                 # GELU, a=3.0, ReLU x depth {5,8} x 10 seeds
    python -m src.blockA5d_k1 stage1 DEPTH          # registered grid, 20 seeds
    python -m src.blockA5d_k1 stage2 DEPTH a1,a2    # +20 seeds at bracketing a
"""

from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from .blockA5d import BUDGETS, HELDOUT_SEED_OFFSET, LR, N_HELDOUT, N_TRAIN, WIDTH, generate

RESULTS = Path(__file__).resolve().parents[1] / "results"
PILOT = RESULTS / "blockA5d_k1_pilot.csv"
RUNS = RESULTS / "blockA5d_k1_runs.csv"
K = 1
A_VALUES = (0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0)
CONTROLS = ("relu", "gelu")
N_WORKERS = int(os.environ.get("A5D_WORKERS", "9"))
FIELDS = ["act", "a", "depth", "seed", "budget", "heldout_errors", "heldout_acc",
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
    act, seed, depth = args
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
            rows.append({"act": act, "a": a_val, "depth": depth, "seed": seed, "budget": budget,
                         "heldout_errors": errs, "heldout_acc": 1 - errs / N_HELDOUT,
                         "train_acc": float(((tr > 0).float() == ytr).float().mean()),
                         "perfect": errs == 0, "wout_fro": float(torch.linalg.norm(head.weight)),
                         "train_loss": float(F.binary_cross_entropy_with_logits(tr, ytr))})
    return rows


def _run(path, jobs):
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["act"], int(r["seed"]), int(r["depth"])) for r in csv.DictReader(f)}
    jobs = [j for j in jobs if j not in done]
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
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
    _run(PILOT, [(act, s, d) for d in (5, 8) for act in ("gelu", "3.0", "relu") for s in range(10)])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        pilot()
    elif cmd == "stage1":
        d = int(sys.argv[2])
        _run(RUNS, [(act, s, d) for act in [str(a) for a in A_VALUES] + list(CONTROLS)
                    for s in range(20)])
    elif cmd == "stage2":
        d = int(sys.argv[2])
        _run(RUNS, [(act, s, d) for act in sys.argv[3].split(",") for s in range(20, 40)])
