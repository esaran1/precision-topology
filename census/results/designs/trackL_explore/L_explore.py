"""EXPLORATORY (Track L, step L2; NOT a registration): train one run on the MNIST-CIFAR dominoes data (L_data.py)
and append one JSON line to OUT.  Exploration seeds only: 2,991,000-2,991,099 (scan: L_seedscan.json; never to be
registered).

Network: MLP 6144 -> W -> W -> 1, ReLU, PyTorch default init (nn.Linear), drawn inside torch.random.fork_rng from
torch.manual_seed(seed) (global torch RNG state restored).  Loss: BCE with logits (mean).  Optimiser: SGD, momentum
0.9, no weight decay, batch 128; minibatch order from numpy default_rng(seed) (reshuffled every epoch).
Arms (learning rates):  std   lr on every parameter
                        outF  lr/F on the output layer's weight (bias at lr), lr elsewhere          (arm 2)
                        outbF lr/F on output weight and bias                                         (arm 2 variant)
                        globG lr/G on every parameter                                                (arm 3 / 3cm)
Checks (no interpolation): at step 0, every 5 steps to 500, then every max(5, floor(0.02 t)) steps.  At each check:
full-train BCE (mean), train accuracy on the predictive / flipped subsets, test accuracy on orig / rand / rev.
Stops at train BCE <= STOP_LOSS or at the step cap.  Torch threads set to 1 inside a local scope (restored).
Stops (exit 3) if the process max RSS exceeds 1.5 GB.
Optimiser option (8th argument): "sgd" (default, above) or "adam" (torch.optim.Adam defaults, per-group lr).
Activation option (9th argument): "relu" (default) or "tanh".  10th argument VSCALE: output weight multiplied by it
after the default draw (1.0 = PyTorch default).
Usage (from census/): python results/designs/trackL_explore/L_explore.py OUT P ARM SEED [LR] [WIDTH] [CAP] [OPT] [ACT] [VSCALE]"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import L_data  # noqa: E402

EXPLORATION = (2_991_000, 2_991_100)
STOP_LOSS = 0.002
RSS_CAP = 1.5e9
BATCH = 128


def rss() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) if sys.platform == "darwin" else float(r) * 1024


def lr_groups(net, arm: str, lr: float):
    out_w, out_b = net[4].weight, net[4].bias
    if arm == "std":
        return [{"params": list(net.parameters()), "lr": lr}]
    if arm.startswith("outb"):
        f = float(arm[4:])
        slow = [out_w, out_b]
    elif arm.startswith("out"):
        f = float(arm[3:])
        slow = [out_w]
    elif arm.startswith("glob"):
        return [{"params": list(net.parameters()), "lr": lr / float(arm[4:])}]
    else:
        raise ValueError(arm)
    ids = {id(q) for q in slow}
    return [{"params": slow, "lr": lr / f}, {"params": [q for q in net.parameters() if id(q) not in ids], "lr": lr}]


def next_check(t: int) -> int:
    return t + 5 if t < 500 else t + max(5, int(0.02 * t))


def run(out: Path, p: float, arm: str, seed: int, lr: float = 0.01, width: int = 256, cap: int = 200_000,
        optname: str = "sgd", act: str = "relu", vscale: float = 1.0) -> dict:
    if not (EXPLORATION[0] <= seed < EXPLORATION[1]):
        raise SystemExit(f"seed {seed} outside the exploration range")
    key = f"p{p}|{arm}|{seed}|lr{lr}|w{width}" + ("" if optname == "sgd" else f"|{optname}") + ("" if act == "relu" else f"|{act}") + ("" if vscale == 1.0 else f"|v{vscale}")
    if out.exists() and any(json.loads(l).get("key") == key for l in out.read_text().splitlines() if l.strip()):
        print("skip", key)
        return {}
    d = L_data.build(p)
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
            A = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[act]
            net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], width), A(),
                                      torch.nn.Linear(width, width), A(), torch.nn.Linear(width, 1))
            if vscale != 1.0:  # output-layer weight scaled after the default draw (no extra randomness)
                with torch.no_grad():
                    net[4].weight.mul_(vscale)
        if optname == "sgd":
            opt = torch.optim.SGD(lr_groups(net, arm, lr), lr=lr, momentum=0.9)
        elif optname == "adam":  # torch defaults: betas (0.9, 0.999), eps 1e-8
            opt = torch.optim.Adam(lr_groups(net, arm, lr), lr=lr)
        else:
            raise ValueError(optname)
        rng = np.random.default_rng(seed)
        lossf = torch.nn.functional.binary_cross_entropy_with_logits

        def logits(X):
            with torch.no_grad():
                return torch.cat([net(X[i:i + 2000]).squeeze(1) for i in range(0, len(X), 2000)])

        rec = {k: [] for k in ("t", "loss", "acc_pred", "acc_flip", "orig", "rand", "rev", "outw", "hid")}

        def check(t):
            z = logits(Xtr)
            rec["t"].append(t)
            rec["loss"].append(float(lossf(z, ytr)))
            corr = ((z > 0).float() == ytr)
            rec["acc_pred"].append(float(corr[~flip].float().mean()))
            rec["acc_flip"].append(float(corr[flip].float().mean()) if flip.any() else float("nan"))
            for k, X in tests.items():
                rec[k[1:]].append(float(((logits(X) > 0).float() == yte).float().mean()))
            rec["outw"].append(float(net[4].weight.detach().norm()))
            rec["hid"].append(float(net[2].weight.detach().norm()))
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
    row = {"key": key, "p": p, "arm": arm, "seed": seed, "lr": lr, "width": width, "opt": optname, "act": act, "vscale": vscale, "steps": t, "finite": finite,
           "secs": round(time.time() - t0, 1), "maxrss_gb": round(rss() / 1e9, 3),
           "rec": {k: [round(v, 5) for v in vs] if k != "t" else vs for k, vs in rec.items()}}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(key, "steps", t, "secs", row["secs"], "final loss", rec["loss"][-1], "rand", rec["rand"][-1],
          "rss", row["maxrss_gb"], flush=True)
    return row


if __name__ == "__main__":
    a = sys.argv[1:]
    run(Path(a[0]) if os.path.isabs(a[0]) else HERE / a[0], float(a[1]), a[2], int(a[3]),
        *(float(a[4]),) if len(a) > 4 else (), *(int(a[5]),) if len(a) > 5 else (), *(int(a[6]),) if len(a) > 6 else (), *(a[7],) if len(a) > 7 else (), *(a[8],) if len(a) > 8 else (), *(float(a[9]),) if len(a) > 9 else ())
