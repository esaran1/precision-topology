"""Driver for the MNIST fold pilots (T48): regenerates the three committed pilot tables.

The pilots were produced by calling mnist_fold.train_one, but no driver was committed.
This driver runs exactly the (width, a, seed, steps) cells of each committed table and
compares the regenerated accuracies with the committed ones (train_one is fully seeded).

    python -m src.mnist_fold_driver check     # regenerate, compare, write mnist_fold_driver_check.csv
    python -m src.mnist_fold_driver write     # overwrite the three pilot tables with the regeneration
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
TABLES = ("mnist_fold_pilot", "mnist_fold_pilot_narrow", "mnist_fold_pilot_longbudget")
_DATA = None


def _cell(args):
    global _DATA
    import torch
    from .mnist_fold import mnist_tensors, train_one
    torch.set_num_threads(1)
    if _DATA is None:
        _DATA = mnist_tensors()
    width, a, seed, steps = args
    return train_one(int(width), float(a), int(seed), int(steps), _DATA)


def regenerate(name: str, workers: int = 10) -> pd.DataFrame:
    ref = pd.read_csv(RESULTS / f"{name}.csv")
    cells = [tuple(r) for r in ref[["width", "a", "seed", "steps"]].itertuples(index=False)]
    with Pool(workers) as p:
        out = p.map(_cell, cells)
    return pd.DataFrame(out)[list(ref.columns)]


def check():
    rows = []
    for name in TABLES:
        ref = pd.read_csv(RESULTS / f"{name}.csv")
        new = regenerate(name)
        m = ref.merge(new, on=["width", "a", "seed", "steps"], suffixes=("_ref", "_new"))
        rows.append({"table": name, "rows_ref": len(ref), "rows_new": len(new), "matched": len(m),
                     "max_abs_diff_test": float((m.test_accuracy_ref - m.test_accuracy_new).abs().max()),
                     "max_abs_diff_train": float((m.train_accuracy_ref - m.train_accuracy_new).abs().max())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "mnist_fold_driver_check.csv", index=False)


def write():
    for name in TABLES:
        regenerate(name).to_csv(RESULTS / f"{name}.csv", index=False)


if __name__ == "__main__":
    {"check": check, "write": write}[sys.argv[1]]()
