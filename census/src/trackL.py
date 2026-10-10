"""Track L: the output-layer lever on MNIST-CIFAR dominoes (design: results/designs/trackL_dominoes_design.md, approved by
the author 2026-10-09 "approve the page as drafted", page commit 9d7e77c; registration: results/trackL_registration.md).

PREDICTED OUTCOME, stated up front: R FAILS and N PASSES (exploration power R 0.00, N 0.98 at n = 40).  The criteria are
the ones drafted before this prediction; the track registers this predicted null.

Data (on-disk caches only; nothing here can download: the .npy caches are read with np.load after their SHA-256 values are
checked against NPY_SHA256 and the recorded SHA256SUMS against the published values; src/mnist_data.py and
src/cifar_data.py, which would fetch, are NOT imported).  Construction (follows the author's summary of Shah et al.'s
released code; identical to the exploration's results/designs/trackL_explore/L_data.py, tested to the byte):
  class 0 = MNIST 0 with CIFAR automobile, class 1 = MNIST 1 with CIFAR truck; MNIST padded to 32x32, repeated to 3
  channels, ON TOP of the CIFAR image: 3 x 64 x 32, values in [0, 1]; random one-to-one pairing within class, truncated to
  the smallest count: 10,000 train / 1,960 test (Shah et al.'s text says 50,000 / 10,000).  ONE fixed dataset, built from
  default_rng(CONSTRUCTION_SEED = 20261009).  p = 0.8: exactly 1,000 of each class's 5,000 training images carry the other
  class's digit; CIFAR is always correct.  Tests: orig (true digit), rand (the 1,960 test MNIST halves under one fixed
  permutation; an MNIST-only classifier scores 0.479), rev (each class paired with the other class's digits).

Network and optimiser: MLP 6144 -> 256 -> 256 -> 1, ReLU, PyTorch default init drawn inside torch.random.fork_rng after
torch.manual_seed(seed) (global torch RNG restored); BCE with logits (mean); SGD momentum 0.9, lr 0.01, batch 128 (order
from numpy default_rng(seed), reshuffled each epoch), no weight decay.  A run stops at the first check with train BCE <=
0.002, or at the step cap STEP_CAP = 200,000 (the exploration's cap; never reached there), or at a non-finite check.
Checks (no interpolation): step 0, every 5 steps to 500, then every max(5, floor(0.02 t)) steps.

Arms (shared seeds 2,992,000-2,992,039; per seed run in the order 1, 2, 3):
  std     arm 1   lr 0.01 everywhere                                              reference
  out16   arm 2   lr 0.01/16 on the output WEIGHT; output bias and hidden 0.01    R
  glob16  arm 3   lr 0.01/16 everywhere                                           N
F = 16 is fixed a priori (as in 2B); it replaces the proposed pilot rule.  Arm 3cm is dropped.

Matching: each matched point is the first check with train BCE <= l, l in {0.6, 0.3, 0.03}.  Primary measure: rand
accuracy.  Criteria: D = arm - arm 1 on the same seed; 95% percentile bootstrap interval of the median D (10,000
resamples).  R (arm 2): lower end > 0 at all three levels.  N (arm 3): upper end < delta = 0.02 at all three levels.
Validity: UNRESOLVED if fewer than 90% of the seeds reach a level in both arms, or a run in an arm the criterion uses is
non-finite (2B's D5).  rev accuracy, orig accuracy, the flipped-20% train accuracy, matched steps, step costs and
convergence are DESCRIPTIVE (no endpoint advantage is claimed).

STRICT CAUSAL RULE: no criterion is a forecast.  Every criterion compares OBSERVED outcomes after every run has ended
(score_tables refuses an incomplete set of runs), so no cutoff applies.  Nothing here imports or calls a forecaster.

ORDER:
  scan / freeze / manifest / stamp   before training (no registered or pilot seed drawn; the freeze reproduces
                                     committed exploration runs on exploration seed 2,991,000).
  run      REFUSES TO START until results/trackL/registration_stamp.txt.ots exists (the OpenTimestamps proof of the
           registration), then per seed in order arms 1, 2, 3, resumable per (seed, arm).  -> runs.jsonl
  score    after all 120 runs.  -> scores.json

Machine rules: one process, nice 15, one thread; a memory gate (memory_pressure -Q free >= 25%, swap free >= 500 MB; waits)
and a disk check (>= 20 GB, else STOP) before every job and run, logged to results/trackL/memory_gate.log; a run stops the
process (exit 3) if its max RSS exceeds 1.5 GB.  No global numpy RNG; the bootstrap uses a local Generator.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        nice -n 15 .venv/bin/python -m src.trackL scan | freeze | manifest | stamp | run | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "trackL"
REGISTRATION_MD = RESULTS / "trackL_registration.md"
DESIGN_MD = RESULTS / "designs" / "trackL_dominoes_design.md"
EXPLORE = RESULTS / "designs" / "trackL_explore"
ALPHA_EXPLORE = RESULTS / "designs" / "trackL_alpha_explore"
DATA = ROOT / "data"

# ------------------------------------------------------------------------------------------ registered constants
CONSTRUCTION_SEED = 20261009
P = 0.8                                          # ‡ fraction of each training class whose MNIST half is predictive
CIFAR_CLASSES = (1, 9)                           # automobile, truck
MNIST_CLASSES = (0, 1)
N_TRAIN_PER_CLASS, N_TEST_PER_CLASS = 5000, 980
RAND_MNIST_AGREES = 938 / 1960                   # 0.4786: an MNIST-only classifier's rand accuracy
PUBLISHED = {  # published SHA-256 of the original downloads (= data/*/SHA256SUMS)
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    "cifar-10-python.tar.gz": "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
}
NPY_SHA256 = {  # the .npy caches actually read (trackL_explore/L_data_check.log)
    "mnist/train_images.npy": "c5b45806d970e809a5f376f9b8461b7ca3b3e0a321282d72d06d00ab6c6c418f",
    "mnist/train_labels.npy": "5dd4d822cab3e20099239bc9d433d587ae3ce00e084d191079dd30b38380b336",
    "mnist/test_images.npy": "4acfa5c2911a2f95015eda9a9b825fbd6bec0f6a6f66942979b1473d33943a11",
    "mnist/test_labels.npy": "ff7e84b144c037e7215dfa787d6773550c5db83029d9a4e7bae6e90f605f081d",
    "cifar10/train_images.npy": "304a769ab0682c43bbdc4f766303066aef2073d37441d820ecc2666ccb3432b3",
    "cifar10/train_labels.npy": "9e2ee9261f6a7c6509b35aaa7161bf15d6d99673bbcac04130702d12a6da88d3",
    "cifar10/test_images.npy": "efbfa3b24c4f91c6febe8a57d87b6448f2c1226d8b2aa24d14db1eea85ad46c6",
    "cifar10/test_labels.npy": "fc48d9ecfdbeacce2dacf004498170f2df12e75e3485475017d2663b587a92f3",
}
KEYS = ("train_images", "train_labels", "test_images", "test_labels")
DATA_KEYS = ("Xtr", "ytr", "flip", "Xorig", "Xrand", "Xrev", "yte")
# SHA-256 of the p = 0.8 dataset (dataset_sha256) built by the committed exploration code L_data.build(0.8) (b08843d).
EXPLORATION_DATA_SHA256 = "6867910ba670acfced6154d198fdf509edeb350afa5e936735ea5fbe082b332f"

WIDTH = 256
LR, MOMENTUM, BATCH = 0.01, 0.9, 128             # ‡ lr
STOP_LOSS = 0.002
STEP_CAP = 200_000
F_OUT = 16.0                                     # arm 2 (output weight) and arm 3 (global); fixed a priori
ARMS = ("std", "out16", "glob16")                # run order per seed
ARM_LABEL = {"std": "1", "out16": "2", "glob16": "3"}
LEVELS = (("bce_0.6", 0.6), ("bce_0.3", 0.3), ("bce_0.03", 0.03))   # ‡ matched train-BCE levels
LEVEL_KEYS = tuple(k for k, _ in LEVELS)
PRIMARY_Q = "rand"
DELTA = 0.02                                     # ‡ N margin (one-sided)
REACH_MIN_FRAC = 0.90
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 20261009, (2.5, 97.5)
NO_CRITERION_IS_A_FORECAST = True
SEEDS = tuple(range(2_992_000, 2_992_040))       # 40 registered seeds, shared by the three arms
PILOT_SEEDS = tuple(range(2_993_000, 2_993_010))  # reserved; UNUSED (no machine check needs them)
EXPLORE_SEEDS = tuple(range(2_991_000, 2_991_100))    # L2 exploration (never registered)
ALPHA_EXPLORE_SEEDS = tuple(range(2_994_000, 2_994_100))  # alpha follow-up exploration (never registered)
FREEZE_REPRO_SEED = 2_991_000
RSS_CAP = 1.5e9
DISK_MIN_BYTES = 20 * 1024 ** 3
MEASURES = ("loss", "acc_pred", "acc_flip", "orig", "rand", "rev", "outw", "hid")


# ------------------------------------------------------------------------------------------ data
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_data() -> dict:
    """Recorded SHA256SUMS == the published values, and every .npy cache's SHA-256 == NPY_SHA256.  Refuses (RuntimeError)
    on a missing cache or any mismatch: nothing is downloaded."""
    out = {"recorded": {}, "npy": {}}
    for ds in ("mnist", "cifar10"):
        d = DATA / ds
        for line in (d / "SHA256SUMS").read_text().split("\n"):
            if line.strip():
                h, name = line.split()
                if PUBLISHED.get(name) != h:
                    raise RuntimeError(f"{ds}/{name}: recorded {h} != published {PUBLISHED.get(name)}")
                out["recorded"][name] = h
        for k in KEYS:
            p = d / f"{k}.npy"
            if not p.exists():
                raise RuntimeError(f"missing cache {p}: refusing (no downloads)")
            h = _sha(p)
            if h != NPY_SHA256[f"{ds}/{k}.npy"]:
                raise RuntimeError(f"{ds}/{k}.npy: SHA-256 {h} != registered {NPY_SHA256[f'{ds}/{k}.npy']}")
            out["npy"][f"{ds}/{k}.npy"] = h
    if set(out["recorded"]) != set(PUBLISHED):
        raise RuntimeError("SHA256SUMS incomplete")
    return out


def _load():
    verify_data()
    return ({k: np.load(DATA / "mnist" / f"{k}.npy") for k in KEYS},
            {k: np.load(DATA / "cifar10" / f"{k}.npy") for k in KEYS})


def _fill(dst, mnist, cifar):
    """dst (k, 3, 64, 32) float32 <- MNIST (k, 28, 28) uint8 padded by 2 and repeated to 3 channels on top (rows 0-31),
    CIFAR (k, 3, 32, 32) uint8 below (rows 32-63); both / 255.  In place (the exploration's operations, in order)."""
    dst[:, :, :32] = 0.0
    dst[:, :, 2:30, 2:30] = (mnist.astype(np.float32) / 255.0)[:, None]
    dst[:, :, 32:] = cifar
    dst[:, :, 32:] /= 255.0


def build(p: float = P) -> dict:
    """The dominoes arrays (float32, flattened 3*64*32 in C,H,W order) and labels, from default_rng(CONSTRUCTION_SEED);
    the exploration's L_data.build, operation for operation.  Keys: Xtr, ytr, flip (train image whose MNIST half is of
    the other class), Xorig, Xrand, Xrev, yte, and provenance (source indices and labels; not hashed): src_train_mnist,
    src_train_cifar, src_test_mnist, src_test_cifar, rand_perm, n_train_per_class, n_test_per_class, rand_mnist_agrees."""
    m, c = _load()
    rng = np.random.default_rng(CONSTRUCTION_SEED)
    out = {}
    for split in ("train", "test"):
        mi, ml = m.pop(f"{split}_images"), m.pop(f"{split}_labels")
        ci, cl = c.pop(f"{split}_images"), c.pop(f"{split}_labels")
        midx = [rng.permutation(np.flatnonzero(ml == d)) for d in MNIST_CLASSES]
        cidx = [rng.permutation(np.flatnonzero(cl == k)) for k in CIFAR_CLASSES]
        n = min(min(len(a) for a in midx), min(len(a) for a in cidx))
        midx = [a[:n] for a in midx]
        cidx = [a[:n] for a in cidx]
        csrc = np.concatenate(cidx)
        y = np.repeat(np.array([0, 1], dtype=np.int64), n)
        if split == "train":
            order = [rng.permutation(n) for _ in range(2)]
            nflip = int(round((1.0 - p) * n))
            mnist_src = []
            flip = np.zeros(2 * n, dtype=bool)
            for k in (0, 1):
                fl = order[k] >= n - nflip
                src = np.empty(n, dtype=np.int64)
                src[~fl] = midx[k][: n - nflip]
                src[fl] = midx[1 - k][n - nflip:] if nflip else midx[1 - k][:0]
                mnist_src.append(src)
                flip[k * n:(k + 1) * n] = fl
            msrc = np.concatenate(mnist_src)
            X = np.empty((2 * n, 3, 64, 32), dtype=np.float32)
            _fill(X, mi[msrc], ci[csrc])
            out["Xtr"] = X.reshape(2 * n, -1)
            out["ytr"], out["flip"], out["n_train_per_class"] = y, flip, n
            out["src_train_mnist_label"], out["src_train_cifar_label"] = ml[msrc].astype(np.int64), cl[csrc].astype(np.int64)
        else:
            msrc = np.concatenate(midx)
            perm = rng.permutation(2 * n)
            rev = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
            for key, ms in (("Xorig", msrc), ("Xrand", msrc[perm]), ("Xrev", msrc[rev])):
                X = np.empty((2 * n, 3, 64, 32), dtype=np.float32)
                _fill(X, mi[ms], ci[csrc])
                out[key] = X.reshape(2 * n, -1)
                out[f"src_test_mnist_label_{key[1:]}"] = ml[ms].astype(np.int64)
            out["yte"], out["n_test_per_class"] = y, n
            out["src_test_cifar_label"] = cl[csrc].astype(np.int64)
            out["rand_perm"] = perm
            out["rand_mnist_agrees"] = float(np.mean((perm >= n).astype(int) == y))
        del mi, ci
    return out


def dataset_sha256(d) -> str:
    """SHA-256 over the seven data arrays in DATA_KEYS order (name, dtype, shape, C-order bytes of each)."""
    h = hashlib.sha256()
    for k in DATA_KEYS:
        a = np.ascontiguousarray(d[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ network and one run
def lr_groups(net, arm, lr=LR):
    """SGD parameter groups.  std: lr everywhere; out16: lr/16 on the output layer's WEIGHT (net[4].weight), lr on its
    bias and the hidden layers; glob16: lr/16 everywhere."""
    if arm == "std":
        return [{"params": list(net.parameters()), "lr": lr}]
    if arm == "glob16":
        return [{"params": list(net.parameters()), "lr": lr / F_OUT}]
    if arm == "out16":
        slow = [net[4].weight]
        ids = {id(q) for q in slow}
        return [{"params": slow, "lr": lr / F_OUT},
                {"params": [q for q in net.parameters() if id(q) not in ids], "lr": lr}]
    raise ValueError(arm)


def next_check(t: int) -> int:
    return t + 5 if t < 500 else t + max(5, int(0.02 * t))


def make_net(seed, torch):
    """MLP 6144 -> 256 -> 256 -> 1 (ReLU), PyTorch default init after torch.manual_seed(seed), drawn inside
    torch.random.fork_rng (the global torch RNG state is restored)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(64 * 32 * 3, WIDTH), torch.nn.ReLU(),
                                  torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, 1))
    return net


def _maxrss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) if sys.platform == "darwin" else float(r) * 1024


def _params_sha(net):
    h = hashlib.sha256()
    for q in net.parameters():
        h.update(q.detach().numpy().astype("<f4").tobytes())
    return h.hexdigest()


def matched(rec):
    """{level key: index of the first check with train BCE <= l, or None (not reached)}."""
    out = {}
    for k, ell in LEVELS:
        out[k] = next((i for i, v in enumerate(rec["loss"]) if v <= ell), None)
    return out


def run_one(arm, seed, data, cap=STEP_CAP, rss_cap=RSS_CAP):
    """One run (the exploration's L_explore.run, operation for operation, at p = 0.8, width 256, SGD, ReLU).  At every
    check: full-train BCE (mean), train accuracy on the predictive and flipped subsets, test accuracy on orig / rand /
    rev, and ||output weight||_2, ||W2||_F.  Stops at train BCE <= STOP_LOSS, at t >= cap, or at a non-finite check
    (train BCE or any parameter not finite: finite = False).  Process exit 3 if the max RSS exceeds rss_cap (None: no
    check).  Returns the run record (unrounded values; the matched points; the parameter hashes)."""
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
        init_sha = _params_sha(net)
        opt = torch.optim.SGD(lr_groups(net, arm), lr=LR, momentum=MOMENTUM)
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
            rec["outw"].append(float(net[4].weight.detach().norm()))
            rec["hid"].append(float(net[2].weight.detach().norm()))
            return rec["loss"][-1]

        t0 = time.time()
        t, tc = 0, 0
        order, pos = rng.permutation(n), 0
        finite, stop, t_nonfinite = True, None, None
        while True:
            if t == tc:
                ell = check(t)
                tc = next_check(t)
                if not (np.isfinite(ell) and all(bool(torch.isfinite(q).all()) for q in net.parameters())):
                    finite, stop, t_nonfinite = False, "nonfinite", t
                    break
                if ell <= STOP_LOSS:
                    stop = "loss"
                    break
                if t >= cap:
                    stop = "cap"
                    break
                if rss_cap is not None and _maxrss() > rss_cap:
                    print(f"STOP: RSS {_maxrss() / 1e9:.2f} GB > {rss_cap / 1e9:.1f} GB (run {arm} {seed} not "
                          f"recorded)", flush=True)
                    raise SystemExit(3)
            if pos + BATCH > n:
                order, pos = rng.permutation(n), 0
            idx = torch.from_numpy(order[pos:pos + BATCH])
            pos += BATCH
            opt.zero_grad(set_to_none=True)
            lossf(net(Xtr[idx]).squeeze(1), ytr[idx]).backward()
            opt.step()
            t += 1
        final_sha = _params_sha(net)
    finally:
        torch.set_num_threads(nthreads)
    mi = matched(rec)
    at = {k: (None if i is None else {"t": rec["t"][i], **{q: rec[q][i] for q in MEASURES}}) for k, i in mi.items()}
    end = {"t": rec["t"][-1], **{q: rec[q][-1] for q in MEASURES}}
    return {"arm": arm, "arm_label": ARM_LABEL[arm], "seed": int(seed), "lr_groups": [g["lr"] for g in lr_groups(net, arm)],
            "steps": int(t), "stop": stop, "finite": bool(finite), "t_nonfinite": t_nonfinite, "cap": int(cap),
            "at": at, "end": end, "rec": rec, "init_params_sha256": init_sha, "final_params_sha256": final_sha,
            "secs": round(time.time() - t0, 2), "max_rss_gb": round(_maxrss() / 1e9, 3)}


# ------------------------------------------------------------------------------------------ pure scoring rules
def verdict(ok, resolved=True):
    return "UNRESOLVED" if not resolved else ("PASS" if ok else "FAIL")


def paired(rows_arm, rows_ref, q=PRIMARY_Q, levels=LEVEL_KEYS):
    """(n_seeds, n_levels) paired differences arm - arm 1 on the same seed (rows aligned by seed); NaN where a seed does
    not reach the level in both arms."""
    assert [r["seed"] for r in rows_arm] == [r["seed"] for r in rows_ref], "rows must be aligned by seed"
    D = np.full((len(rows_arm), len(levels)), np.nan)
    for i, (a, b) in enumerate(zip(rows_arm, rows_ref)):
        for j, k in enumerate(levels):
            x, y = (a.get("at") or {}).get(k), (b.get("at") or {}).get(k)
            if x is not None and y is not None:
                D[i, j] = x[q] - y[q]
    return D


def boot_ci_median(D, n_boot=None, seed=None):
    """95% percentile bootstrap interval of the median paired difference per column (2B's registered procedure, 2B D1-D3):
    seeds (rows) resampled jointly across the columns with ONE index matrix (n_boot x n_seeds) from a FRESH
    default_rng(seed); the median over the resampled seeds that reach the cell (nanmedian); numpy's default (linear)
    percentiles 2.5 and 97.5.  Returns (lo, hi, point median)."""
    import warnings
    D = np.asarray(D, float)
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(D[idx], axis=1)
        lo, hi = np.percentile(med, BOOT_PCT[0], axis=0), np.percentile(med, BOOT_PCT[1], axis=0)
        point = np.nanmedian(D, axis=0)
    return lo, hi, point


def reach_ok(D, n_seeds):
    """The 90% reach rule: every level is reached in BOTH arms by >= 90% of the seeds (36 of 40)."""
    n_both = np.isfinite(np.asarray(D, float)).sum(axis=0)
    return bool((n_both >= REACH_MIN_FRAC * n_seeds).all()), [int(x) for x in n_both]


def criterion_R(D, lo, n_seeds, nonfinite_used):
    """R (arm 2): PASS if the lower end is > 0 at all three levels (strict).  UNRESOLVED (overriding PASS and FAIL) if a
    level is reached in both arms by < 90% of the seeds, or a run of arm 2 or arm 1 is non-finite; else FAIL."""
    ok_reach, n_both = reach_ok(D, n_seeds)
    lo = np.asarray(lo, float)
    ok = bool(np.all(lo > 0))
    return {"criterion": "R", "arm": "out16", "levels": list(LEVEL_KEYS), "lower": lo.tolist(), "n_both": n_both,
            "reach_ok": ok_reach, "nonfinite_in_used_arms": bool(nonfinite_used), "all_lower_gt_0": ok,
            "verdict": verdict(ok, ok_reach and not nonfinite_used)}


def criterion_N(D, hi, n_seeds, nonfinite_used):
    """N (arm 3): PASS if the upper end is < delta = 0.02 at all three levels (strict, one-sided: an arm worse than arm 1
    passes).  UNRESOLVED as for R (reach; a non-finite run of arm 3 or arm 1); else FAIL."""
    ok_reach, n_both = reach_ok(D, n_seeds)
    hi = np.asarray(hi, float)
    ok = bool(np.all(hi < DELTA))
    return {"criterion": "N", "arm": "glob16", "levels": list(LEVEL_KEYS), "upper": hi.tolist(), "delta": DELTA,
            "n_both": n_both, "reach_ok": ok_reach, "nonfinite_in_used_arms": bool(nonfinite_used),
            "all_upper_lt_delta": ok, "verdict": verdict(ok, ok_reach and not nonfinite_used)}


def arm_cells(D, lo, hi, med):
    return {k: {"median": float(med[j]) if np.isfinite(med[j]) else None, "lo": float(lo[j]), "hi": float(hi[j]),
                "n_both": int(np.isfinite(D[:, j]).sum()), "n_up": int((D[:, j] > 0).sum()),
                "n_down": int((D[:, j] < 0).sum())} for j, k in enumerate(LEVEL_KEYS)}


def outcome(v, lever_hurts=False):
    """The page's outcome statements from the verdicts v = {"R": ..., "N": ...}.  lever_hurts (DESCRIPTIVE, never a
    verdict): arm 2's upper end < 0 at all three levels."""
    res = {"verdicts": dict(v), "predicted": {"R": "FAIL", "N": "PASS"},
           "matches_prediction": v["R"] == "FAIL" and v["N"] == "PASS"}
    st = []
    if "UNRESOLVED" in v.values():
        res["outcome"] = "UNRESOLVED"
        st.append("UNRESOLVED: " + ", ".join(k for k, x in v.items() if x == "UNRESOLVED"))
        st += [f"Also {k} {x}." for k, x in v.items() if x != "UNRESOLVED"]
    else:
        res["outcome"] = f"R {v['R']}, N {v['N']}"
        if v["R"] == "FAIL" and v["N"] == "PASS":
            st.append("R FAIL, N PASS (predicted): in this MLP the readout's learning rate does not decide which "
                      "feature is used.")
        if v["R"] == "PASS":
            st.append("R PASS: the lever transfers, against the prediction.")
        if v["N"] == "FAIL":
            st.append("N FAIL: global slowing helps.")
    if lever_hurts:
        st.append("Arm 2 below arm 1 (descriptive: upper end < 0 at all three levels): the lever hurts.")
    res["statements"] = st
    return res


def score_tables(runs):
    """All registered verdicts from the complete set of runs: every arm x every seed exactly once, else an AssertionError
    (no partial scoring).  Arm a's criterion uses arm a and arm 1 (2B's D5): a non-finite arm-2 run voids R only, a
    non-finite arm-3 run voids N only, a non-finite arm-1 run voids both."""
    by = {(r["arm"], r["seed"]): r for r in runs}
    assert len(by) == len(runs), "a run appears twice"
    seeds = sorted({r["seed"] for r in runs})
    assert set(by) == {(a, s) for a in ARMS for s in seeds}, "every arm x every seed exactly once"
    rows = {a: [by[(a, s)] for s in seeds] for a in ARMS}
    nonfinite_arm = {a: any(not r["finite"] for r in rows[a]) for a in ARMS}
    n = len(seeds)
    per_arm, crit = {}, {}
    for a, name in (("out16", "R"), ("glob16", "N")):
        D = paired(rows[a], rows["std"])
        lo, hi, med = boot_ci_median(D)
        used = nonfinite_arm[a] or nonfinite_arm["std"]
        c = criterion_R(D, lo, n, used) if name == "R" else criterion_N(D, hi, n, used)
        per_arm[a] = {"arm": a, "arm_label": ARM_LABEL[a], "n_seeds": n, "cells_rand": arm_cells(D, lo, hi, med),
                      "criterion": c}
        crit[name] = c["verdict"]
    hurts = bool(np.all(np.asarray([per_arm["out16"]["cells_rand"][k]["hi"] for k in LEVEL_KEYS]) < 0))
    return {"n_seeds": n, "nonfinite_by_arm": nonfinite_arm,
            "nonfinite_runs": [[r["arm"], r["seed"], r["t_nonfinite"]] for r in runs if not r["finite"]],
            "arms": per_arm, "verdicts": crit, "outcome": outcome(crit, lever_hurts=hurts)}


# ------------------------------------------------------------------------------------------ descriptive (never a verdict)
def _q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


DESC_Q = ("rand", "rev", "orig", "acc_flip", "acc_pred", "outw", "hid")


def descriptive(runs):
    """DESCRIPTIVE, registered as such (never a verdict): values at the matched points and at the end of training; paired
    arm - arm 1 medians (seeds up/down) for every measure, with the same bootstrap interval for rev (secondary) and orig;
    steps to each level and their per-seed ratio to arm 1 (step cost; arm 2's realised step cost); convergence (steps to
    the stop, stop reasons).  NO endpoint advantage is claimed."""
    by = {(r["arm"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    rows = {a: [by[(a, s)] for s in seeds] for a in ARMS if all((a, s) in by for s in seeds)}
    out = {"label": "DESCRIPTIVE (never a verdict)", "endpoint_no_advantage_claimed": True, "arms": {}}
    for a, R in rows.items():
        d = {"values": {}, "paired_vs_arm1": {}, "steps_to": {}, "step_cost_vs_arm1": {},
             "convergence": {"steps": _q([r["steps"] for r in R]),
                             "stop": {s: sum(r["stop"] == s for r in R) for s in ("loss", "cap", "nonfinite")}}}
        for k in LEVEL_KEYS + ("end",):
            get = (lambda r, k=k: r["end"]) if k == "end" else (lambda r, k=k: r["at"].get(k))
            d["values"][k] = {q: _q([(get(r) or {}).get(q) for r in R]) for q in DESC_Q}
            if a != "std" and "std" in rows:
                for q in DESC_Q:
                    dd = [get(r)[q] - get(b)[q] for r, b in zip(R, rows["std"]) if get(r) and get(b)]
                    d["paired_vs_arm1"][f"{k}|{q}"] = {**(_q(dd) or {}), "n_up": int(sum(x > 0 for x in dd)),
                                                       "n_down": int(sum(x < 0 for x in dd))}
        for k in LEVEL_KEYS:
            d["steps_to"][k] = _q([(r["at"].get(k) or {}).get("t") for r in R])
            if a != "std" and "std" in rows:
                d["step_cost_vs_arm1"][k] = _q([r["at"][k]["t"] / b["at"][k]["t"] for r, b in zip(R, rows["std"])
                                                if r["at"].get(k) and b["at"].get(k) and b["at"][k]["t"] > 0])
        if a != "std" and "std" in rows:
            d["step_cost_vs_arm1"]["end"] = _q([r["steps"] / b["steps"] for r, b in zip(R, rows["std"]) if b["steps"]])
            for q in ("rev", "orig"):
                D = paired(R, rows["std"], q=q)
                lo, hi, med = boot_ci_median(D)
                d[f"bootstrap_{q}_DESCRIPTIVE"] = arm_cells(D, lo, hi, med)
        out["arms"][a] = d
    return out


# ------------------------------------------------------------------------------------------ machine rules and io
def parse_memory(pressure_text, swap_text):
    """(free %, swap free MB) from `memory_pressure -Q` and `sysctl vm.swapusage` outputs (-1 if unparsable)."""
    import re
    m = re.search(r"free percentage:\s*(\d+)%", pressure_text)
    s = re.search(r"free\s*=\s*([\d.]+)M", swap_text)
    return (int(m.group(1)) if m else -1), (float(s.group(1)) if s else -1.0)


def memory_ok(free_pct, swap_free_mb):
    return free_pct >= 25 and swap_free_mb >= 500


def memory_gate(tag):
    """memory_pressure -Q free >= 25% and swap free >= 500 MB (waits, re-checking every 60 s); disk free >= 20 GB (else
    STOP).  Every check logged to results/trackL/memory_gate.log."""
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            p = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
            s = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
            f, w = parse_memory(p, s)
        except Exception:                                                 # noqa: BLE001
            f, w = -1, -1.0
        ok = memory_ok(f, w)
        disk = shutil.disk_usage(ROOT).free
        with open(OUT / "memory_gate.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB "
                     f"disk_free={disk / 1024 ** 3:.1f}GB "
                     f"{'STOP' if disk < DISK_MIN_BYTES else ('OK' if ok else 'WAIT')}\n")
        if disk < DISK_MIN_BYTES:
            raise SystemExit(f"STOP: disk free {disk / 1024 ** 3:.1f} GB < 20 GB")
        if ok:
            return f, w
        time.sleep(60)


def _setup():
    cur = os.getpriority(os.PRIO_PROCESS, 0)
    if cur < 15:
        os.nice(15 - cur)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _append_write(p, row):
    """Append one JSON line (one write call, flushed and fsynced)."""
    with open(p, "a") as fh:
        fh.write(json.dumps(_jsonable(row)) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _rows(p):
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()] if Path(p).exists() else []


# ------------------------------------------------------------------------------------------ seed scan
SCAN_DIRS = ("src", "tests", "results", "paper", "notes", "independent", "data", "dist")
SCAN_OWN = ("src/trackL.py", "tests/test_trackL.py", "results/trackL_registration.md")
# files that NAME the candidate ranges (scan scripts and their outputs, the design page, the alpha README); skipped and
# listed in seed_scan.json
SCAN_NAMING = ("results/designs/trackL_dominoes_design.md", "results/designs/trackL_explore/L_seedscan.py",
               "results/designs/trackL_explore/L_seedscan.json", "results/designs/trackL_alpha_explore/A_seedscan.py",
               "results/designs/trackL_alpha_explore/A_seedscan.json", "results/designs/trackL_alpha_explore/README.md")
SCAN_SKIP_SUFFIX = (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip", ".pyc", ".jpg")
SCAN_PATTERNS = {
    "registered": r"(^|[^0-9.])29920[0-3][0-9]([^0-9]|$)|(^|[^0-9])2_992_0[0-3][0-9]([^0-9]|$)"
                  r"|(^|[^0-9.,])2,992,0[0-3][0-9]([^0-9,]|$)",
    "pilot": r"(^|[^0-9.])29930[0][0-9]([^0-9]|$)|(^|[^0-9])2_993_00[0-9]([^0-9]|$)"
             r"|(^|[^0-9.,])2,993,00[0-9]([^0-9,]|$)"}


def scan_tree(patterns=SCAN_PATTERNS, dirs=SCAN_DIRS, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under `dirs` (binary files, results/trackL/, Track L's own files and SCAN_NAMING
    skipped), streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
    import re
    rx = {k: re.compile(v.encode()) for k, v in patterns.items()}
    hits = {k: [] for k in patterns}
    skip = set(SCAN_OWN) | set(SCAN_NAMING)
    for top in dirs:
        for dp, dns, fns in os.walk(ROOT / top):
            dns[:] = [d for d in dns if d not in ("__pycache__", ".git")]
            if Path(dp).resolve() == OUT.resolve():
                dns[:] = []
                continue
            for fn in fns:
                p = Path(dp) / fn
                rel = str(p.relative_to(ROOT))
                if rel in skip or p.suffix in SCAN_SKIP_SUFFIX:
                    continue
                try:
                    with open(p, "rb") as fh:
                        if b"\0" in fh.read(4096):
                            continue
                        fh.seek(0)
                        found, tail = set(), b""
                        while True:
                            buf = fh.read(chunk)
                            if not buf:
                                break
                            data = tail + buf
                            for k, r in rx.items():
                                if k not in found and r.search(data):
                                    found.add(k)
                            tail = data[-overlap:]
                except OSError:
                    continue
                for k in found:
                    hits[k].append(rel)
    return hits


def range_literal_overlaps(ranges):
    """Every integer `range(a, b)` / `range(a)` literal in src/ and tests/ (Track L's own files excepted) that overlaps a
    candidate range.  Returns (n checked, overlaps)."""
    import re
    rx = re.compile(r"range\(\s*([0-9_]+)\s*(?:,\s*([0-9_]+)\s*)?[,)]")
    over, n = [], 0
    for d in ("src", "tests"):
        for p in sorted((ROOT / d).rglob("*.py")):
            rel = str(p.relative_to(ROOT))
            if rel in SCAN_OWN:
                continue
            for m in rx.finditer(p.read_text(errors="ignore")):
                a, b = m.group(1), m.group(2)
                lo, hi = (0, int(a)) if b is None else (int(a), int(b))
                n += 1
                for name, (c0, c1) in ranges.items():
                    if lo < c1 and c0 < hi:
                        over.append([rel, m.group(0), name])
    return n, over


def parquet_seed_hits(ranges):
    """Every column whose name contains 'seed' in every parquet file under SCAN_DIRS: count of values in a range."""
    import pyarrow.parquet as pqm
    out = []
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.parquet")):
            t = pqm.read_table(p)
            for c in t.column_names:
                if "seed" in c.lower():
                    col = t.column(c).to_pylist()
                    k = sum(1 for x in col if isinstance(x, (int, float)) and x == x
                            and any(c0 <= x < c1 for c0, c1 in ranges.values()))
                    out.append([str(p.relative_to(ROOT)), c, int(k)])
    return out


def scan():
    """The 40 registered seeds (2,992,000-2,992,039) and the 10 pilot seeds (2,993,000-2,993,009) are unused: no such
    number (also written with _ or ,) in any text file under SCAN_DIRS (Track L's own files and the files that name the
    candidate ranges excepted, listed), no overlapping range literal in src/ or tests/, no such value in a parquet seed
    column; and disjoint from the exploration seeds.  -> results/trackL/seed_scan.json"""
    OUT.mkdir(parents=True, exist_ok=True)
    memory_gate("scan")
    ranges = {"registered": (SEEDS[0], SEEDS[-1] + 1), "pilot": (PILOT_SEEDS[0], PILOT_SEEDS[-1] + 1)}
    hits = scan_tree()
    n_lit, over = range_literal_overlaps(ranges)
    pq = parquet_seed_hits(ranges)
    mine = set(SEEDS) | set(PILOT_SEEDS)
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)],
                      "pilot": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)]},
           "dirs": list(SCAN_DIRS), "patterns": SCAN_PATTERNS, "skipped_own": list(SCAN_OWN),
           "skipped_naming_the_ranges": list(SCAN_NAMING), "pattern_files": hits,
           "range_literals_checked": n_lit, "range_overlaps": over, "parquet_seed_columns": pq,
           "disjoint_from_explorations": bool(not (mine & (set(EXPLORE_SEEDS) | set(ALPHA_EXPLORE_SEEDS)))),
           "registered_pilot_disjoint": bool(not (set(SEEDS) & set(PILOT_SEEDS)))}
    out["unused"] = bool(not any(hits.values()) and not over and all(k == 0 for _, _, k in pq)
                         and out["disjoint_from_explorations"] and out["registered_pilot_disjoint"])
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("pattern_files", "range_literals_checked", "range_overlaps",
                                                     "unused")})))
    if not out["unused"]:
        raise SystemExit("STOP: the Track L seed ranges are not unused")


# ------------------------------------------------------------------------------------------ freeze (no registered seed)
EXPLORE_RUN_FILES = {"std": "explore_gate.jsonl", "out16": "explore_lever.jsonl", "glob16": "explore_lever.jsonl"}


def explore_row(arm, seed):
    """The committed exploration row of (p 0.8, arm, seed, lr 0.01, width 256, SGD, ReLU, default init)."""
    key = f"p0.8|{arm}|{seed}|lr0.01|w256"
    rows = [r for r in _rows(EXPLORE / EXPLORE_RUN_FILES[arm]) if r["key"] == key]
    assert len(rows) == 1, f"exploration row {key}"
    return rows[0]


def compare_with_exploration(rec, old):
    """A run against the committed exploration row (L_explore.py rounds every recorded value to 5 decimals): steps,
    finiteness, the check steps, and every recorded value rounded to 5 decimals equal.  exact = all equal."""
    d = {"steps_equal": rec["steps"] == old["steps"], "finite_equal": rec["finite"] == old["finite"],
         "t_equal": rec["rec"]["t"] == old["rec"]["t"], "n_values": 0, "n_unequal": 0}
    for k in MEASURES:
        a, b = rec["rec"][k], old["rec"][k]
        if len(a) != len(b):
            d["n_unequal"] += abs(len(a) - len(b))
        for x, y in zip(a, b):
            d["n_values"] += 1
            same = (round(x, 5) == y) or (isinstance(x, float) and math.isnan(x) and y != y)
            d["n_unequal"] += not same
    d["exact"] = bool(d["steps_equal"] and d["finite_equal"] and d["t_equal"] and d["n_unequal"] == 0
                      and d["n_values"] > 0)
    return d


def explore_as_runs(seeds=EXPLORE_SEEDS[:5]):
    """The committed exploration rows (arms std/out16/glob16, p 0.8) in the registered run-record form (matched points
    from the rounded records), for the exploration-seed scoring disclosure (NOT a verdict)."""
    out = []
    for arm in ARMS:
        for s in seeds:
            r = explore_row(arm, s)
            mi = matched(r["rec"])
            at = {k: (None if i is None else {"t": r["rec"]["t"][i], **{q: r["rec"][q][i] for q in MEASURES}})
                  for k, i in mi.items()}
            out.append({"arm": arm, "seed": s, "finite": r["finite"], "t_nonfinite": None, "at": at, "steps": r["steps"],
                        "stop": "loss", "end": {"t": r["rec"]["t"][-1], **{q: r["rec"][q][-1] for q in MEASURES}}})
    return out


def freeze():
    """results/trackL/frozen.json (no registered or pilot seed drawn): the data check (SHA-256 of every cache read), the
    dataset's counts and hash (== the exploration's), the registered constants, the exact reproduction of the committed
    exploration runs of all three arms on exploration seed 2,991,000, and the registered scoring applied to the 5
    exploration seeds (disclosure only, never a verdict)."""
    _setup()
    memory_gate("freeze data")
    vd = verify_data()
    data = build()
    dsha = dataset_sha256(data)
    counts = {"n_train": int(len(data["ytr"])), "n_test": int(len(data["yte"])),
              "n_train_per_class": int(data["n_train_per_class"]), "n_test_per_class": int(data["n_test_per_class"]),
              "flipped_per_class": [int(data["flip"][:5000].sum()), int(data["flip"][5000:].sum())],
              "rand_mnist_agrees": data["rand_mnist_agrees"]}
    repro = {}
    for arm in ARMS:
        memory_gate(f"freeze repro {arm} {FREEZE_REPRO_SEED}")
        rec = run_one(arm, FREEZE_REPRO_SEED, data)
        repro[arm] = {**compare_with_exploration(rec, explore_row(arm, FREEZE_REPRO_SEED)), "secs": rec["secs"],
                      "steps": rec["steps"], "final_params_sha256": rec["final_params_sha256"]}
        print(arm, json.dumps(repro[arm]), flush=True)
    del data
    ex = score_tables(explore_as_runs())
    fr = {"label": "Track L frozen inputs (before any registered seed is drawn)", "data_check": vd,
          "dataset_sha256": dsha, "dataset_matches_exploration": dsha == EXPLORATION_DATA_SHA256, "counts": counts,
          "constants": {"p": P, "construction_seed": CONSTRUCTION_SEED, "width": WIDTH, "lr": LR, "momentum": MOMENTUM,
                        "batch": BATCH, "stop_loss": STOP_LOSS, "step_cap": STEP_CAP, "F": F_OUT,
                        "levels": dict(LEVELS), "delta": DELTA, "reach_min_frac": REACH_MIN_FRAC,
                        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "pct": list(BOOT_PCT)}},
          "arms": list(ARMS), "seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)],
          "pilot_seeds_unused": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)],
          "exploration_reproduction_seed": FREEZE_REPRO_SEED, "exploration_reproduction": repro,
          "exploration_reproduction_exact": bool(all(r["exact"] for r in repro.values())),
          "exploration_seeds_scored_DISCLOSURE_NOT_A_VERDICT": {"seeds": list(EXPLORE_SEEDS[:5]),
                                                                "verdicts": ex["verdicts"],
                                                                "cells_rand": {a: ex["arms"][a]["cells_rand"]
                                                                               for a in ("out16", "glob16")}},
          "max_rss_gb": round(_maxrss() / 1e9, 3)}
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(fr), indent=1))
    print(json.dumps(_jsonable({k: v for k, v in fr.items() if k not in ("exploration_reproduction", "data_check")}),
                     indent=1))
    if not (fr["dataset_matches_exploration"] and fr["exploration_reproduction_exact"]
            and counts["n_train"] == 10_000 and counts["n_test"] == 1_960 and counts["flipped_per_class"] == [1000, 1000]):
        raise SystemExit("STOP: a frozen check failed")


# ------------------------------------------------------------------------------------------ registration manifest
EXPLORE_FILES = ("README.md", "L_data.py", "L_data_check.log", "L_explore.py", "L_summary.py", "L_power.py",
                 "L_tables.py", "L_tables.md", "L_seedscan.py", "L_seedscan.json", "power_sgd.log",
                 "explore_gate.jsonl", "explore_lever.jsonl")
ALPHA_FILES = ("README.md", "A_explore.py", "A_summary.py", "A_summary.json", "A_tables.md", "A_seedscan.json")
FROZEN_DATA = (("results/designs/trackL_dominoes_design.md", "results/trackL_registration.md",
                "results/trackL/seed_scan.json", "results/trackL/frozen.json", "tests/test_trackL.py",
                "data/mnist/SHA256SUMS", "data/cifar10/SHA256SUMS")
               + tuple(f"results/designs/trackL_explore/{f}" for f in EXPLORE_FILES)
               + tuple(f"results/designs/trackL_alpha_explore/{f}" for f in ALPHA_FILES))


def code_closure(start=("trackL",)):
    """Every src module trackL reaches by relative imports, recursively."""
    import ast
    seen, todo = set(), list(start)
    while todo:
        m = todo.pop()
        if m in seen or not (ROOT / "src" / f"{m}.py").exists():
            continue
        seen.add(m)
        tree = ast.parse((ROOT / "src" / f"{m}.py").read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                if n.module:
                    todo.append(n.module.split(".")[0])
                else:
                    todo.extend(a.name for a in n.names)
    return sorted(f"src/{m}.py" for m in seen)


def manifest_files():
    return sorted(set(code_closure()) | set(FROZEN_DATA))


def manifest():
    """results/trackL/registration.sha256: SHA-256 of the code, tests, page, registration text, frozen files, data
    checksums and exploration provenance."""
    lines = []
    for rel in manifest_files():
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        lines.append(f"{_sha(p)}  {rel}")
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_committed(p):
    rel = str(Path(p).resolve().relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"


def stamp():
    """registration_stamp.txt: the registration commit (the last commit touching the manifest) and the SHA-256 of the
    registration file and of the manifest."""
    _assert_committed(OUT / "registration.sha256")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(OUT / "registration.sha256")], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    txt = (f"Track L registration\nregistration commit: {h}\n"
           f"sha256 results/trackL_registration.md: {_sha(REGISTRATION_MD)}\n"
           f"sha256 results/trackL/registration.sha256: {_sha(OUT / 'registration.sha256')}\n")
    (OUT / "registration_stamp.txt").write_text(txt)
    print(txt)


def assert_registration():
    """The OpenTimestamps proof of the registration stamp must exist (checked FIRST: no training without it); then every
    manifest hash matches and the manifest and stamp are committed and unmodified."""
    if not (OUT / "registration_stamp.txt.ots").exists():
        raise SystemExit("REFUSED: results/trackL/registration_stamp.txt.ots does not exist (no OpenTimestamps proof of "
                         "the registration yet): no registered training")
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")


# ------------------------------------------------------------------------------------------ run / score
def run():
    """AFTER the registration commit is pushed and OpenTimestamped (refuses otherwise): for each registered seed in order,
    arms 1, 2, 3, resumable per (seed, arm) (a (seed, arm) already in runs.jsonl is skipped); a memory gate before every
    run.  The dataset is built once and must hash to the frozen value."""
    assert_registration()
    _setup()
    memory_gate("run data")
    data = build()
    assert dataset_sha256(data) == EXPLORATION_DATA_SHA256, "dataset changed"
    rf = OUT / "runs.jsonl"
    done = {(r["seed"], r["arm"]) for r in _rows(rf)}
    for s in SEEDS:
        for arm in ARMS:
            if (s, arm) in done:
                continue
            memory_gate(f"run {arm} {s}")
            rec = run_one(arm, s, data)
            _append_write(rf, rec)
            print(json.dumps({k: rec[k] for k in ("arm", "seed", "steps", "stop", "finite", "secs", "max_rss_gb")}),
                  flush=True)


def score():
    """After all 120 runs: the registered verdicts and the descriptive tables.  -> results/trackL/scores.json"""
    assert_registration()
    runs = _rows(OUT / "runs.jsonl")
    assert sorted({r["seed"] for r in runs}) == list(SEEDS), "every registered seed"
    ST = score_tables(runs)
    out = {**ST, "DESCRIPTIVE": descriptive(runs), "no_criterion_is_a_forecast": NO_CRITERION_IS_A_FORECAST}
    (OUT / "scores.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("verdicts", "outcome")}), indent=1))


def main(argv):
    fns = {"scan": scan, "freeze": freeze, "manifest": manifest, "stamp": stamp, "run": run, "score": score}
    if len(argv) < 2 or argv[1] not in fns:
        raise SystemExit(f"usage: python -m src.trackL {'|'.join(fns)}")
    fns[argv[1]]()


if __name__ == "__main__":
    main(sys.argv)
