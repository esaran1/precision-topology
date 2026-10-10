"""EXPLORATORY (Track LN; NOT a registration): Track L with ONE change, a LayerNorm WITHOUT affine parameters on the
penultimate features before the readout:

    MLP 6144 -> 256 -> ReLU -> 256 -> ReLU -> LayerNorm(256, elementwise_affine=False, default eps) -> Linear(256, 1)

Everything else is Track L's (src/trackL.py, registration a8ab6db), imported READ-ONLY (nothing registered is
modified): the data construction and check (trackL.build, which calls trackL.verify_data: on-disk caches only,
SHA-256 verified, nothing downloaded), the dataset hash (must equal trackL.EXPLORATION_DATA_SHA256), the check
schedule (trackL.next_check), the matching rule (trackL.matched: first check with train BCE <= l, l in {0.6, 0.3,
0.03}), the stop (train BCE <= 0.002, cap 200,000, non-finite), the optimiser (SGD momentum 0.9, lr 0.01, batch 128,
order from numpy default_rng(seed)), the init (PyTorch default, drawn inside torch.random.fork_rng after
torch.manual_seed(seed); the LayerNorm has no parameters, so the draw order is Track L's), and the arms:
    std     lr 0.01 everywhere
    out16   lr 0.01/16 on the READOUT WEIGHT (net[5].weight); readout bias and hidden layers 0.01
    glob16  lr 0.01/16 everywhere
Recorded at every check: Track L's measures (train BCE, train accuracy on predictive / flipped, test accuracy orig /
rand / rev, outw = ||readout weight||_2, hid = ||W2||_F) plus absf = mean |f| over the training set.

Driver: ONE process, nice 15, one torch thread; a memory gate (memory_pressure -Q free >= 25% and swap free >= 500 MB,
else wait 60 s; disk >= 20 GB, else STOP) logged to memory_gate.log before every run; exit 3 if the max RSS exceeds
1.5 GB; resumable per (seed, arm) (a key already in OUT is skipped).  No global numpy/torch RNG state is touched.
Exploration seeds only: 2,995,000-2,995,099 (LN_seedscan.json; never to be registered).
Usage (from census/): python results/designs/trackLN_explore/LN_explore.py OUT SEED:ARM [SEED:ARM ...]"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import trackL as TL  # noqa: E402  (read-only)

EXPLORATION = (2_995_000, 2_995_100)
ARMS = ("std", "out16", "glob16")
RSS_CAP = 1.5e9
MEASURES = TL.MEASURES + ("absf",)


def gate(tag):
    while True:
        try:
            p = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
            s = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
            f, w = TL.parse_memory(p, s)
        except Exception:  # noqa: BLE001
            f, w = -1, -1.0
        ok = TL.memory_ok(f, w)
        disk = shutil.disk_usage(ROOT).free
        st = "STOP" if disk < TL.DISK_MIN_BYTES else ("OK" if ok else "WAIT")
        with open(HERE / "memory_gate.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB "
                     f"disk_free={disk / 1024 ** 3:.1f}GB {st}\n")
        if st == "STOP":
            raise SystemExit("STOP: disk free < 20 GB")
        if ok:
            return
        time.sleep(60)


def make_net(seed, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(64 * 32 * 3, TL.WIDTH), torch.nn.ReLU(),
                                  torch.nn.Linear(TL.WIDTH, TL.WIDTH), torch.nn.ReLU(),
                                  torch.nn.LayerNorm(TL.WIDTH, elementwise_affine=False),
                                  torch.nn.Linear(TL.WIDTH, 1))
    assert not list(net[4].parameters())
    return net


def lr_groups(net, arm, lr=TL.LR):
    if arm == "std":
        return [{"params": list(net.parameters()), "lr": lr}]
    if arm == "glob16":
        return [{"params": list(net.parameters()), "lr": lr / TL.F_OUT}]
    if arm == "out16":
        slow = [net[5].weight]
        ids = {id(q) for q in slow}
        return [{"params": slow, "lr": lr / TL.F_OUT},
                {"params": [q for q in net.parameters() if id(q) not in ids], "lr": lr}]
    raise ValueError(arm)


def run_one(arm, seed, data, cap=TL.STEP_CAP):
    """Track L's run_one, operation for operation, with the LN network and the readout at net[5]."""
    import torch
    Xtr, ytr = torch.from_numpy(data["Xtr"]), torch.from_numpy(data["ytr"]).float()
    flip = torch.from_numpy(data["flip"])
    tests = {k: torch.from_numpy(data[k]) for k in ("Xorig", "Xrand", "Xrev")}
    yte = torch.from_numpy(data["yte"]).float()
    n = len(ytr)
    nthreads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        net = make_net(seed, torch)
        init_sha = TL._params_sha(net)
        opt = torch.optim.SGD(lr_groups(net, arm), lr=TL.LR, momentum=TL.MOMENTUM)
        rng = np.random.default_rng(seed)
        lossf = torch.nn.functional.binary_cross_entropy_with_logits

        def logits(X):
            with torch.no_grad():
                return torch.cat([net(X[i:i + 2000]).squeeze(1) for i in range(0, len(X), 2000)])

        rec = {k: [] for k in ("t",) + MEASURES}

        def check(t):
            z = logits(Xtr)
            rec["t"].append(t)
            rec["loss"].append(float(lossf(z, ytr)))
            corr = ((z > 0).float() == ytr)
            rec["acc_pred"].append(float(corr[~flip].float().mean()))
            rec["acc_flip"].append(float(corr[flip].float().mean()) if flip.any() else float("nan"))
            for k, X in tests.items():
                rec[k[1:]].append(float(((logits(X) > 0).float() == yte).float().mean()))
            rec["outw"].append(float(net[5].weight.detach().norm()))
            rec["hid"].append(float(net[2].weight.detach().norm()))
            rec["absf"].append(float(z.abs().mean()))
            return rec["loss"][-1]

        t0 = time.time()
        t, tc = 0, 0
        order, pos = rng.permutation(n), 0
        finite, stop, t_nonfinite = True, None, None
        while True:
            if t == tc:
                ell = check(t)
                tc = TL.next_check(t)
                if not (np.isfinite(ell) and all(bool(torch.isfinite(q).all()) for q in net.parameters())):
                    finite, stop, t_nonfinite = False, "nonfinite", t
                    break
                if ell <= TL.STOP_LOSS:
                    stop = "loss"
                    break
                if t >= cap:
                    stop = "cap"
                    break
                if TL._maxrss() > RSS_CAP:
                    print(f"STOP: RSS {TL._maxrss() / 1e9:.2f} GB > 1.5 GB (run {arm} {seed} not recorded)", flush=True)
                    raise SystemExit(3)
            if pos + TL.BATCH > n:
                order, pos = rng.permutation(n), 0
            idx = torch.from_numpy(order[pos:pos + TL.BATCH])
            pos += TL.BATCH
            opt.zero_grad(set_to_none=True)
            lossf(net(Xtr[idx]).squeeze(1), ytr[idx]).backward()
            opt.step()
            t += 1
        final_sha = TL._params_sha(net)
        bias = float(net[5].bias.detach()[0])
    finally:
        torch.set_num_threads(nthreads)
    mi = TL.matched(rec)
    at = {k: (None if i is None else {"t": rec["t"][i], **{q: rec[q][i] for q in MEASURES}}) for k, i in mi.items()}
    end = {"t": rec["t"][-1], **{q: rec[q][-1] for q in MEASURES}}
    return {"key": f"{seed}|{arm}", "arm": arm, "arm_label": TL.ARM_LABEL[arm], "seed": int(seed),
            "lr_groups": [g["lr"] for g in lr_groups(net, arm)], "steps": int(t), "stop": stop, "finite": bool(finite),
            "t_nonfinite": t_nonfinite, "cap": int(cap), "at": at, "end": end, "final_bias": bias, "rec": rec,
            "init_params_sha256": init_sha, "final_params_sha256": final_sha,
            "secs": round(time.time() - t0, 2), "max_rss_gb": round(TL._maxrss() / 1e9, 3)}


def main(argv):
    out = Path(argv[1]) if os.path.isabs(argv[1]) else HERE / argv[1]
    jobs = []
    for j in argv[2:]:
        s, a = j.split(":")
        s = int(s)
        if not (EXPLORATION[0] <= s < EXPLORATION[1]):
            raise SystemExit(f"seed {s} outside the exploration range")
        if a not in ARMS:
            raise SystemExit(f"arm {a}")
        jobs.append((s, a))
    cur = os.getpriority(os.PRIO_PROCESS, 0)
    if cur < 15:
        os.nice(15 - cur)
    done = {r["key"] for r in TL._rows(out)}
    todo = [(s, a) for s, a in jobs if f"{s}|{a}" not in done]
    print(f"{time.strftime('%F %T')} jobs {len(jobs)}, to do {len(todo)}", flush=True)
    if not todo:
        return
    gate("data")
    data = TL.build()
    dsha = TL.dataset_sha256(data)
    print("dataset_sha256", dsha, "matches Track L:", dsha == TL.EXPLORATION_DATA_SHA256, "rss",
          round(TL._maxrss() / 1e9, 3), flush=True)
    if dsha != TL.EXPLORATION_DATA_SHA256:
        raise SystemExit("STOP: dataset hash differs from Track L's")
    for s, a in todo:
        gate(f"run {a} {s}")
        print(f"{time.strftime('%F %T')} run {s} {a}", flush=True)
        rec = run_one(a, s, data)
        rec["dataset_sha256"] = dsha
        TL._append_write(out, rec)
        print(json.dumps({k: rec[k] for k in ("seed", "arm", "steps", "stop", "finite", "secs", "max_rss_gb")}),
              {k: (v or {}).get("t") for k, v in rec["at"].items()}, flush=True)
    print(f"{time.strftime('%F %T')} done", flush=True)


if __name__ == "__main__":
    main(sys.argv)
