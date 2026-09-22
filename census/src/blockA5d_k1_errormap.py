"""Where a barred (ReLU) network's held-out errors lie, by position on the core.

Calibration for the stratified held-out set (POST HOC diagnostic, not a
registered test).  Each point's core coordinate is recovered exactly from its Z
coordinate (index 2), which neither class's targeted thickening touches:
A: Z = -u1/(sqrt2 - u1)  ->  u1 = sqrt2 Z/(Z - 1);  B: Z = v1/(sqrt2 - v1)  ->  v1 = sqrt2 Z/(1 + Z).
u1 = -1 (A) and v1 = -1 (B) are the closest approach of the two cores.

    python -m src.blockA5d_k1_errormap DEPTH SEED[,SEED...]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .blockA5d import SQRT2, generate
from .blockA5d_k1 import BUDGETS, K, LR, N_TRAIN, RESULTS, _build

OUT = RESULTS / "blockA5d_k1_errormap.csv"
N_MAP = 200_000
MAP_SEED_OFFSET = 4_000_000
EDGES = np.linspace(-1, 1, 11)


def core_coord(X, y):
    z = X[:, 2].astype(np.float64)
    return np.where(y == 0, SQRT2 * z / (z - 1), SQRT2 * z / (1 + z))


def job(args):
    act, seed, depth = args
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    Xtr, ytr = (torch.from_numpy(v) for v in generate(N_TRAIN, seed, num_copies=K))
    Xm, ym = generate(N_MAP, seed + MAP_SEED_OFFSET, num_copies=K)
    model, _ = _build(act, seed, depth)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(BUDGETS[-1]):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(Xtr).squeeze(-1), ytr).backward()
        opt.step()
    with torch.no_grad():
        wrong = ((model(torch.from_numpy(Xm)).squeeze(-1) > 0).numpy() != ym.astype(bool))
    c = core_coord(Xm, ym)
    rows = []
    for cls in (0, 1):
        m = ym == cls
        idx = np.clip(np.digitize(c[m], EDGES) - 1, 0, 9)
        for b in range(10):
            sel = idx == b
            rows.append({"act": act, "depth": depth, "seed": seed, "cls": "AB"[cls],
                         "coord_lo": EDGES[b], "coord_hi": EDGES[b + 1],
                         "n": int(sel.sum()), "errors": int(wrong[m][sel].sum())})
    return rows


def main(depth, seeds):
    with Pool(len(seeds)) as p:
        rows = [r for rs in p.map(job, [("relu", s, depth) for s in seeds]) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(d.pivot_table(index=["cls", "coord_lo"], columns="seed", values="errors").to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")])
