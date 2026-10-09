"""EXPLORATORY (Track L follow-up, MOTIVATED BY THE TRACK L NULL; NOT a registration): one run on the MNIST-CIFAR
dominoes (../trackL_explore/L_data.py, on-disk data only, hashes verified by L_data.verify) with a FIXED OUTPUT
MULTIPLIER alpha:  f(x) = alpha * g(x),  g = the L2 network (MLP 6144 -> 256 -> 256 -> 1, ReLU, PyTorch default init
drawn inside torch.random.fork_rng from torch.manual_seed(seed); the multiplier has no parameters, so g at init is
the same function for every alpha on a seed).  Loss: BCE with logits f (mean).  SGD momentum 0.9, no weight decay,
batch 128, minibatch order from numpy default_rng(seed).  Everything except alpha and the learning rate is as in
../trackL_explore/L_explore.py (checks, measures, stop rule).
Learning-rate modes (the argument LR is the base rate, 0.01):
  fix    lr = LR on every parameter, whatever alpha (the parameter-space step is the same; the function-space step at
         init scales as alpha^2)
  fs     lr = LR / alpha^2 on every parameter (the first-order function-space step at a given parameter point is the
         same as at alpha = 1: df = alpha * dg, dg = -lr * alpha * dL/df * |grad g|^2)
  ARM    "std" = the mode's lr; "globG" = the mode's lr / G on every parameter (the analogue of arm N)
Checks (no interpolation): step 0, every 5 steps to 500, then every max(5, floor(0.02 t)).  At each check: full-train
BCE of f, train accuracy (predictive / flipped subsets), test accuracy on orig / rand / rev, output-weight and hidden
(layer 2) weight norms, and the mean |f| and mean |g| on the training set.
Stops at train BCE <= 0.002, at the step cap, or on a non-finite loss.  One torch thread inside a local scope;
exit 3 if the process max RSS exceeds 1.5 GB.
Exploration seeds only: 2,994,000-2,994,099 (A_seedscan.json; never to be registered).
Usage (from census/): python results/designs/trackL_alpha_explore/A_explore.py OUT ALPHA MODE ARM SEED [CAP] [LR]"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trackL_explore"))
import L_data  # noqa: E402
from L_explore import BATCH, RSS_CAP, STOP_LOSS, next_check, rss  # noqa: E402

EXPLORATION = (2_994_000, 2_994_100)
P = 0.8
WIDTH = 256


class Mult(torch.nn.Module):
    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, x):
        return self.a * x


def run(out: Path, alpha: float, mode: str, arm: str, seed: int, cap: int = 100_000, base_lr: float = 0.01) -> dict:
    if not (EXPLORATION[0] <= seed < EXPLORATION[1]):
        raise SystemExit(f"seed {seed} outside the exploration range")
    lr = {"fix": base_lr, "fs": base_lr / alpha ** 2}[mode]
    if arm.startswith("glob"):
        lr /= float(arm[4:])
    elif arm != "std":
        raise ValueError(arm)
    key = f"a{alpha}|{mode}|{arm}|{seed}|base{base_lr}"
    if out.exists() and any(json.loads(l).get("key") == key for l in out.read_text().splitlines() if l.strip()):
        print("skip", key)
        return {}
    d = L_data.build(P)
    Xtr, ytr = torch.from_numpy(d["Xtr"]), torch.from_numpy(d["ytr"]).float()
    flip = torch.from_numpy(d["flip"])
    tests = {k: torch.from_numpy(d[k]) for k in ("Xorig", "Xrand", "Xrev")}
    yte = torch.from_numpy(d["yte"]).float()
    n = len(ytr)
    nthreads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], WIDTH), torch.nn.ReLU(),
                                      torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, 1),
                                      Mult(alpha))
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        rng = np.random.default_rng(seed)
        lossf = torch.nn.functional.binary_cross_entropy_with_logits

        def logits(X):
            with torch.no_grad():
                return torch.cat([net(X[i:i + 2000]).squeeze(1) for i in range(0, len(X), 2000)])

        rec = {k: [] for k in ("t", "loss", "acc_pred", "acc_flip", "orig", "rand", "rev", "outw", "hid", "absf")}

        def check(t):
            z = logits(Xtr)
            rec["t"].append(t)
            rec["loss"].append(float(lossf(z, ytr)))
            corr = ((z > 0).float() == ytr)
            rec["acc_pred"].append(float(corr[~flip].float().mean()))
            rec["acc_flip"].append(float(corr[flip].float().mean()))
            for k, X in tests.items():
                rec[k[1:]].append(float(((logits(X) > 0).float() == yte).float().mean()))
            rec["outw"].append(float(net[4].weight.detach().norm()))
            rec["hid"].append(float(net[2].weight.detach().norm()))
            rec["absf"].append(float(z.abs().mean()))
            return rec["loss"][-1]

        t0 = time.time()
        t, tc = 0, 0
        order, pos = rng.permutation(n), 0
        finite = True
        while True:
            if t == tc:
                ell = check(t)
                tc = next_check(t)
                if len(rec["t"]) % 50 == 0:
                    print(f"  t {t} loss {ell:.4f} rand {rec['rand'][-1]:.4f} {time.time() - t0:.0f}s", flush=True)
                if not np.isfinite(ell):
                    finite = False
                    break
                if ell <= STOP_LOSS or t >= cap:
                    break
                if rss() > RSS_CAP:
                    print(f"RSS {rss() / 1e9:.2f} GB over cap: stop", flush=True)
                    raise SystemExit(3)
            if pos + BATCH > n:
                order, pos = rng.permutation(n), 0
            idx = torch.from_numpy(order[pos:pos + BATCH])
            pos += BATCH
            opt.zero_grad(set_to_none=True)
            lossf(net(Xtr[idx]).squeeze(1), ytr[idx]).backward()
            opt.step()
            t += 1
    finally:
        torch.set_num_threads(nthreads)
    row = {"key": key, "p": P, "alpha": alpha, "mode": mode, "arm": arm, "seed": seed, "base_lr": base_lr, "lr": lr,
           "width": WIDTH, "cap": cap, "steps": t, "finite": finite, "reached_stop": bool(finite and rec["loss"][-1] <= STOP_LOSS),
           "secs": round(time.time() - t0, 1), "maxrss_gb": round(rss() / 1e9, 3),
           "rec": {k: [round(v, 5) for v in vs] if k != "t" else vs for k, vs in rec.items()}}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(key, "lr", lr, "steps", t, "secs", row["secs"], "final loss", rec["loss"][-1], "rand", rec["rand"][-1],
          "rss", row["maxrss_gb"], flush=True)
    return row


if __name__ == "__main__":
    a = sys.argv[1:]
    run(Path(a[0]) if os.path.isabs(a[0]) else HERE / a[0], float(a[1]), a[2], a[3], int(a[4]),
        *(int(a[5]),) if len(a) > 5 else (), *(float(a[6]),) if len(a) > 6 else ())
