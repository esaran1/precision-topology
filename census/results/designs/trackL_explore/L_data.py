"""EXPLORATORY (Track L, step L2): the MNIST-CIFAR dominoes construction, from the on-disk copies only.

Follows Shah et al. (NeurIPS 2020, arXiv:2006.07710) released code (github.com/harshays/simplicitybiaspitfalls,
scripts/mnistcifar_utils.py, data_utils.py):
  * class 0 = MNIST digit 0 with CIFAR-10 automobile (CIFAR label 1); class 1 = MNIST digit 1 with CIFAR-10 truck
    (CIFAR label 9);
  * MNIST 28x28 zero-padded by 2 on every side to 32x32 and repeated to 3 channels; stacked ON TOP of the CIFAR image
    (height axis) -> 3 x 64 x 32; values in [0, 1] (ToTensor only, no mean/std normalisation);
  * random one-to-one pairing within class; each class truncated to the smallest count -> 2 x 5000 train,
    2 x 980 test (the paper's text says 50,000 / 10,000; the released code gives 10,000 / 1,960).
Mechanism change (author; Kirichenko, Izmailov, Wilson, ICLR 2023, arXiv:2204.02937 use 100/99/95%): the MNIST half
is predictive on a fraction p of the TRAINING images of each class (exactly round(p * 5000) per class); on the rest
the MNIST half is a digit of the other class.  The CIFAR half is always the true class.
Test sets (2 x 980, CIFAR halves fixed):
  orig : MNIST half of the true class (the released code's test set);
  rand : the 1,960 test MNIST halves under one fixed uniform permutation (Shah's randomize-MNIST): MNIST half
         independent of the label; measures reliance on CIFAR;
  rev  : each class-c CIFAR image paired with a digit-(1-c) MNIST half (the class-(1-c) MNIST halves of `orig`, in
         order): MNIST half against the label.
No download: refuses to run unless the four .npy caches of each dataset exist (src/mnist_data.py and
src/cifar_data.py would otherwise fetch).  All randomness from a local numpy Generator (CONSTRUCTION_SEED); the
flipped sets are nested in p (the flipped images for p are the last round((1-p) n) of one fixed order)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CONSTRUCTION_SEED = 20261009
CIFAR_CLASSES = (1, 9)          # automobile, truck
MNIST_CLASSES = (0, 1)
PUBLISHED = {  # TFDS / Keras published SHA-256 of the original downloads
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    "cifar-10-python.tar.gz": "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
}
KEYS = ("train_images", "train_labels", "test_images", "test_labels")


def verify() -> dict:
    """Recorded SHA256SUMS == published values; SHA-256 of the on-disk .npy caches (the files actually read)."""
    out = {"recorded": {}, "npy": {}}
    for ds in ("mnist", "cifar10"):
        d = ROOT / "data" / ds
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
            out["npy"][f"{ds}/{k}.npy"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if set(out["recorded"]) != set(PUBLISHED):
        raise RuntimeError("SHA256SUMS incomplete")
    return out


def _load():
    verify()
    from src import cifar_data, mnist_data
    return mnist_data.load(), cifar_data.load()


def _fill(dst: np.ndarray, mnist: np.ndarray, cifar: np.ndarray) -> None:
    """dst (k, 3, 64, 32) float32 <- MNIST (k, 28, 28) uint8 padded by 2 and repeated to 3 channels on top (rows
    0-31), CIFAR (k, 3, 32, 32) uint8 below (rows 32-63); both / 255.  Filled in place (low peak memory)."""
    dst[:, :, :32] = 0.0
    dst[:, :, 2:30, 2:30] = (mnist.astype(np.float32) / 255.0)[:, None]
    dst[:, :, 32:] = cifar
    dst[:, :, 32:] /= 255.0


def build(p: float) -> dict:
    """Arrays (float32, flattened 3*64*32 in C,H,W order) and labels.  Keys: Xtr, ytr, flip (bool, train image whose
    MNIST half is of the other class), Xorig, Xrand, Xrev, yte."""
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
            order = [rng.permutation(n) for _ in range(2)]   # position of each pair in the flip order, per class
            nflip = int(round((1.0 - p) * n))
            mnist_src = []
            flip = np.zeros(2 * n, dtype=bool)
            for k in (0, 1):
                # class k: pairs at flip-order positions >= n - nflip take a digit of class 1-k.  Digit d is used
                # n - nflip times by class d and nflip times by class 1-d: exactly n images of each digit.
                fl = order[k] >= n - nflip
                src = np.empty(n, dtype=np.int64)
                src[~fl] = midx[k][: n - nflip]
                src[fl] = midx[1 - k][n - nflip:] if nflip else midx[1 - k][:0]
                mnist_src.append(src)
                flip[k * n:(k + 1) * n] = fl
            X = np.empty((2 * n, 3, 64, 32), dtype=np.float32)
            _fill(X, mi[np.concatenate(mnist_src)], ci[csrc])
            out["Xtr"] = X.reshape(2 * n, -1)
            out["ytr"], out["flip"], out["n_train_per_class"] = y, flip, n
        else:
            msrc = np.concatenate(midx)
            perm = rng.permutation(2 * n)
            rev = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
            for key, ms in (("Xorig", msrc), ("Xrand", msrc[perm]), ("Xrev", msrc[rev])):
                X = np.empty((2 * n, 3, 64, 32), dtype=np.float32)
                _fill(X, mi[ms], ci[csrc])
                out[key] = X.reshape(2 * n, -1)
            out["yte"], out["n_test_per_class"] = y, n
            out["rand_mnist_agrees"] = float(np.mean((perm >= n).astype(int) == y))
        del mi, ci
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=1))
    d = build(0.95)
    print({k: (v.shape, v.dtype.name) if isinstance(v, np.ndarray) else v for k, v in d.items()})
    print("flip fraction", d["flip"].mean(), "Xtr range", d["Xtr"].min(), d["Xtr"].max())
