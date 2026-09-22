"""Block A5d: the expressivity threshold versus the capability onset in 5D.

Registered in results/blockA5d_prediction.md before any run.

Data: Ren & Lim's released multicopy generator
(exp_4_higher_dim_r5_multicopy/train_width_scaling_v7.py::
generate_multi_copy_dataset), reimplemented call for call so a given seed gives
the same points: L1-ordered copy centres scaled by `spacing`, targeted
thickening (A in its Y-subspace, B in its X-subspace), uniform ball of radius rho.

Network: their depth convention -- five hidden blocks Linear(5,5) -> act --
then a single-logit linear readout (BCE; identical function class and loss to
their two-logit cross-entropy).

Protocol: full-batch Adam, lr 1e-3, constant, float32, one CPU thread per run.
Budgets 1k/4k/16k/64k are checkpoints of one 64k-step run, which is exact for a
deterministic constant-lr full-batch trajectory.

Usage:
  python -m src.blockA5d stage1            # 11 activations x 20 seeds
  python -m src.blockA5d stage2            # +20 seeds at the bracketing a values
  python -m src.blockA5d protocol          # GELU under Appendix G.2 (private)
"""

from __future__ import annotations

import csv
import math
import os
import sys
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "blockA5d_runs.csv"
PROTO_OUT = RESULTS / "blockA5d_protocol_private.csv"

SQRT2 = math.sqrt(2.0)
N_SPHERE = 2
D = 2 * N_SPHERE + 1
K_COPIES = 10
SPACING = 10.0
RHO = 0.5
N_TRAIN = 10_000
N_HELDOUT = 20_000
DEPTH = 5
WIDTH = 5
LR = 1e-3
BUDGETS = (1_000, 4_000, 16_000, 64_000)
A_VALUES = (0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0)
CONTROLS = ("relu", "gelu")
SEEDS_STAGE1 = range(20)
SEEDS_STAGE2 = range(20, 40)
HELDOUT_SEED_OFFSET = 1_000_000
N_WORKERS = int(os.environ.get("A5D_WORKERS", "9"))

FIELDS = ["act", "a", "seed", "budget", "heldout_errors", "heldout_acc",
          "train_acc", "perfect", "wout_fro", "train_loss"]


# --- their generator, call for call ------------------------------------------
def _shell_vectors(d, k):
    if d < 0 or k < 0:
        return
    if d == 0:
        if k == 0:
            yield ()
        return
    if d == 1:
        if k == 0:
            yield (0,)
        else:
            yield (-k,)
            yield (k,)
        return
    for x in range(-k, k + 1):
        rem = k - abs(x)
        for tail in _shell_vectors(d - 1, rem):
            yield (x,) + tail


def _l1_nondecreasing(d):
    k = 0
    while True:
        yield from _shell_vectors(d, k)
        k += 1


def copy_centers(num_copies=K_COPIES, spacing=SPACING, dim=D):
    return np.array(list(islice(_l1_nondecreasing(dim), num_copies)),
                    dtype=float) * spacing


def _sample_on_sphere(dim, rng):
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def _sample_uniform_ball(dim, rho, rng):
    direction = _sample_on_sphere(dim, rng)
    r = rho * (rng.random() ** (1.0 / dim))
    return r * direction


def _tilde_a(u):
    n = u.size - 1
    u1, upr = u[0], u[1:]
    a = 1.0 - u1 / SQRT2
    return np.concatenate([upr / a, np.array([-u1 / (SQRT2 * a)]), np.zeros(n)])


def _tilde_b(v):
    n = v.size - 1
    v1, vpr = v[0], v[1:]
    b = 1.0 - v1 / SQRT2
    return np.concatenate([np.zeros(n), np.array([v1 / (SQRT2 * b)]), vpr / b])


def generate(num_samples, seed, num_copies=K_COPIES, spacing=SPACING, rho=RHO):
    """generate_multi_copy_dataset(..., targeted_thickening=True), call for call."""

    rng = np.random.default_rng(seed)
    n, d = N_SPHERE, D
    half = num_samples // 2
    per_a = half // num_copies
    per_b = half // num_copies
    centers = copy_centers(num_copies, spacing, d)
    A_pts, B_pts = [], []
    for ci in range(num_copies):
        c = centers[ci]
        for _ in range(per_a):
            base = _tilde_a(_sample_on_sphere(n + 1, rng))
            noise = np.zeros(d)
            noise[n + 1:] = _sample_uniform_ball(n, rho, rng)
            A_pts.append(base + noise + c)
        for _ in range(per_b):
            base = _tilde_b(_sample_on_sphere(n + 1, rng))
            noise = np.zeros(d)
            noise[:n] = _sample_uniform_ball(n, rho, rng)
            B_pts.append(base + noise + c)
    X = np.vstack([np.array(A_pts, dtype=np.float32), np.array(B_pts, dtype=np.float32)])
    y = np.concatenate([np.zeros(len(A_pts)), np.ones(len(B_pts))]).astype(np.float32)
    perm = rng.permutation(len(X))          # their final shuffle, same rng
    return X[perm], y[perm]


# --- network ----------------------------------------------------------------
def _build(act, seed):
    import torch
    from torch import nn

    class FA(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, x):
            return x + self.a * torch.sin(x)

    def make_act():
        if act == "relu":
            return nn.ReLU()
        if act == "gelu":
            return nn.GELU()
        return FA(float(act))

    torch.manual_seed(seed)
    layers = []
    for _ in range(DEPTH):
        layers += [nn.Linear(WIDTH, WIDTH), make_act()]
    body = nn.Sequential(*layers)
    head = nn.Linear(WIDTH, 1)
    return nn.Sequential(body, head), head


def run_one(args):
    """One (activation, seed): a 64k-step run, scored at every budget."""

    act, seed = args
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    Xtr, ytr = generate(N_TRAIN, seed)
    Xho, yho = generate(N_HELDOUT, seed + HELDOUT_SEED_OFFSET)
    Xtr, ytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xho, yho = torch.from_numpy(Xho), torch.from_numpy(yho)
    model, head = _build(act, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    a_val = "" if act in CONTROLS else float(act)
    rows = []
    step = 0
    for budget in BUDGETS:
        while step < budget:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(Xtr).squeeze(-1), ytr)
            loss.backward()
            opt.step()
            step += 1
        with torch.no_grad():
            tr_logit = model(Xtr).squeeze(-1)
            ho_logit = model(Xho).squeeze(-1)
            tr_acc = float(((tr_logit > 0).float() == ytr).float().mean())
            errs = int(((ho_logit > 0).float() != yho).sum())
            tl = float(F.binary_cross_entropy_with_logits(tr_logit, ytr))
            wfro = float(torch.linalg.norm(head.weight))
        rows.append({"act": act, "a": a_val, "seed": seed, "budget": budget,
                     "heldout_errors": errs, "heldout_acc": 1 - errs / N_HELDOUT,
                     "train_acc": tr_acc, "perfect": errs == 0,
                     "wout_fro": wfro, "train_loss": tl})
    return rows


def _done():
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {(r["act"], int(r["seed"])) for r in csv.DictReader(f)}


def _run(jobs):
    done = _done()
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with Pool(N_WORKERS) as pool:
            for rows in pool.imap_unordered(run_one, jobs):
                for r in rows:
                    w.writerow(r)
                f.flush()
                last = rows[-1]
                print(f"  {last['act']:>5} seed {last['seed']:>2}: held-out errors "
                      + " / ".join(str(r["heldout_errors"]) for r in rows),
                      flush=True)


def stage1():
    acts = [str(a) for a in A_VALUES] + list(CONTROLS)
    _run([(act, s) for act in acts for s in SEEDS_STAGE1])


def stage2(bracketing_acts):
    _run([(act, s) for act in bracketing_acts for s in SEEDS_STAGE2])


# --- protocol comparison (private): Appendix G.2 ------------------------------
def run_protocol(seed):
    import copy
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    X, y = generate(6_000, seed)
    rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(len(X))
    tr, va = idx[:4_800], idx[4_800:]
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    Xho, yho = generate(N_HELDOUT, seed + HELDOUT_SEED_OFFSET)
    Xho, yho = torch.from_numpy(Xho), torch.from_numpy(yho)
    model, _ = _build("gelu", seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    best_acc, best_loss, best_state, best_epoch, since = -1.0, math.inf, None, 0, 0
    epochs = 0
    for epoch in range(1, 801):
        epochs = epoch
        order = torch.randperm(len(Xtr), generator=gen)
        for s in range(0, len(Xtr), 128):
            b = order[s:s + 128]
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(Xtr[b]).squeeze(-1), ytr[b]).backward()
            opt.step()
        with torch.no_grad():
            lv = model(Xva).squeeze(-1)
            acc = float(((lv > 0).float() == yva).float().mean())
            lo = float(F.binary_cross_entropy_with_logits(lv, yva))
        if acc > best_acc or (acc == best_acc and lo < best_loss):
            best_acc, best_loss, best_epoch, since = acc, lo, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= 150:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        errs = int(((model(Xho).squeeze(-1) > 0).float() != yho).sum())
    return {"seed": seed, "protocol": "appendix_G2", "best_val_acc": best_acc,
            "best_epoch": best_epoch, "epochs_run": epochs,
            "heldout_errors": errs, "heldout_acc": 1 - errs / N_HELDOUT,
            "perfect": errs == 0}


def protocol():
    with Pool(N_WORKERS) as pool:
        rows = list(pool.imap_unordered(run_protocol, range(20)))
    rows.sort(key=lambda r: r["seed"])
    with PROTO_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {PROTO_OUT.name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        stage2(sys.argv[2].split(","))
    elif cmd == "protocol":
        protocol()
