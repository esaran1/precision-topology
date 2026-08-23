"""CIFAR-10 download/parse for Task F.

Fetched once from the University of Toronto origin, cached as .npy with
sha256 recorded in ``data/cifar10/SHA256SUMS`` (unverified TLS context —
no certifi in this venv — so the checksum record is the integrity trail;
the tarball's sha256 is also compared against the widely published value
when available).
"""

from __future__ import annotations

import hashlib
import io
import pickle
import ssl
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Published sha256 of cifar-10-python.tar.gz (CIFAR site / torchvision).
KNOWN_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cifar10"


def load() -> dict[str, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caches = {k: DATA_DIR / f"{k}.npy" for k in
              ("train_images", "train_labels", "test_images", "test_labels")}
    if all(p.exists() for p in caches.values()):
        return {k: np.load(p) for k, p in caches.items()}

    context = ssl._create_unverified_context()
    with urllib.request.urlopen(URL, timeout=300, context=context) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    with (DATA_DIR / "SHA256SUMS").open("a") as f:
        f.write(f"{digest}  cifar-10-python.tar.gz\n")
    if digest != KNOWN_SHA256:
        raise RuntimeError(f"CIFAR-10 sha256 mismatch: {digest}")

    train_x, train_y = [], []
    test_x, test_y = None, None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not (name.startswith("data_batch") or name == "test_batch"):
                continue
            batch = pickle.load(tar.extractfile(member), encoding="bytes")
            images = batch[b"data"].reshape(-1, 3, 32, 32)
            labels = np.array(batch[b"labels"], dtype=np.int64)
            if name == "test_batch":
                test_x, test_y = images, labels
            else:
                train_x.append(images)
                train_y.append(labels)
    arrays = {
        "train_images": np.concatenate(train_x),
        "train_labels": np.concatenate(train_y),
        "test_images": test_x,
        "test_labels": test_y,
    }
    assert arrays["train_images"].shape == (50_000, 3, 32, 32)
    assert arrays["test_images"].shape == (10_000, 3, 32, 32)
    for k, p in caches.items():
        np.save(p, arrays[k])
    return arrays


if __name__ == "__main__":
    for k, v in load().items():
        print(k, v.shape, v.dtype)
