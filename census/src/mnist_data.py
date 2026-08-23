"""MNIST download/parse for Task E Part 2.

Files are fetched once from the ossci S3 mirror and cached under
``census/data/mnist/`` as .npy with sha256 recorded alongside.  The venv
has no certifi, so the fetch uses an unverified TLS context; integrity is
therefore checked against the published IDX sizes and the sha256 of every
file is recorded in ``data/mnist/SHA256SUMS`` for reproducibility.
"""

from __future__ import annotations

import gzip
import hashlib
import ssl
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"


def _parse_idx(raw: bytes) -> np.ndarray:
    magic, = struct.unpack(">i", raw[:4])
    dims = magic & 0xFF
    shape = struct.unpack(">" + "i" * dims, raw[4:4 + 4 * dims])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def load() -> dict[str, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    sums_path = DATA_DIR / "SHA256SUMS"
    sums: list[str] = []
    context = ssl._create_unverified_context()
    for key, filename in FILES.items():
        cache = DATA_DIR / (key + ".npy")
        if cache.exists():
            arrays[key] = np.load(cache)
            continue
        with urllib.request.urlopen(MIRROR + filename, timeout=60, context=context) as r:
            compressed = r.read()
        sums.append(f"{hashlib.sha256(compressed).hexdigest()}  {filename}")
        arrays[key] = _parse_idx(gzip.decompress(compressed))
        np.save(cache, arrays[key])
    if sums:
        with sums_path.open("a") as f:
            f.write("\n".join(sums) + "\n")
    expected = {"train_images": (60_000, 28, 28), "train_labels": (60_000,),
                "test_images": (10_000, 28, 28), "test_labels": (10_000,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{key}: got {arrays[key].shape}, expected {shape}")
    return arrays


if __name__ == "__main__":
    data = load()
    for key, value in data.items():
        print(key, value.shape, value.dtype)
