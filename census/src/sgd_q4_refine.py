"""q4 under SGD, refined grid in (0.15, 0.25] only.

The coarse sweep pinned q4's onset at eps=0.25 for all four budgets because
the next grid step (0.15) failed everywhere -- unresolved, not flat. This
resolves that interval to answer: is q4's onset genuinely budget-independent
under SGD, or merely unresolved?

Not a ratio test (q0.667 is unreachable under SGD and is not chased). This is
a same-family cross-optimizer comparison at fixed beta = 1.25, against Adam's
measured -0.8305.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.depth_families import make_family
from src.fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX, make_data

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS, LR, Q = 40, 0.3, 4.0
INNER = torch.linspace(-INNER_MAX, INNER_MAX, 2001)
_P = torch.linspace(OUTER_MIN, OUTER_MAX, 1001)
OUTER = torch.cat([_P, -_P])


def rate(eps: float, B: int) -> float:
    f, _ = make_family(Q); a = 1.0 + eps; n = 0
    for s in range(SEEDS):
        x, y = make_data(200, s); torch.manual_seed(s)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.SGD([th], lr=LR)
        for _ in range(B):
            opt.zero_grad(set_to_none=True)
            w1, b1, w2, b2 = th[0], th[1], th[2], th[3]
            F.binary_cross_entropy_with_logits(w2 * f(w1 * x + b1, a) + b2, y).backward()
            opt.step()
        with torch.no_grad():
            t = th.detach(); w1, b1, w2, b2 = t
            li = w2 * f(w1 * INNER + b1, a) + b2
            lo = w2 * f(w1 * OUTER + b1, a) + b2
            n += bool((li < 0).all() and (lo > 0).all())
    return n / SEEDS


def main() -> None:
    # 1.08x steps across (0.15, 0.25]
    grid = [round(0.25 / 1.08**k, 5) for k in range(8)]   # 0.25 .. 0.1459
    rows, curves = [], []
    for B in (2_000, 8_000, 32_000, 128_000):
        onset, brk = None, False
        for eps in grid:
            r = rate(eps, B)
            curves.append({"budget": B, "eps": eps, "rate": r})
            print(f"  q4 SGD B={B} eps={eps}: rate={r:.3f}", flush=True)
            if r >= 0.5: onset = eps
            elif onset is not None: brk = True; break
        rows.append({"budget": B, "onset_eps": onset, "bracketed": brk})
        print(f"q4 SGD B={B}: onset={onset} bracketed={brk}", flush=True)
        for nm, fr in (("sgd_q4_refined", pd.DataFrame(rows)),
                       ("sgd_q4_curves", pd.DataFrame(curves))):
            stem = RESULTS / nm
            with artifact_lock(stem, nm):
                tmp = stem.with_suffix(".csv.tmp"); fr.to_csv(tmp, index=False)
                tmp.replace(stem.with_suffix(".csv"))
    d = pd.DataFrame(rows); d = d[d.bracketed & d.onset_eps.notna()]
    print(f"\nBRACKETED CELLS: {len(d)} of 4")
    if len(d) >= 2:
        s = np.polyfit(np.log(d.budget.values), np.log(d.onset_eps.values), 1)[0]
        print(f"q4 SGD exponent = {s:.4f}   (q4 Adam = -0.8305)")
        print(f"predicted -alpha_SGD/beta = {-0.7188/1.25:.4f}")
        print(f"ratio SGD/Adam = {s/-0.8305:.4f}  (alpha ratio = {0.7188/1.1172:.4f})")
    print("done", flush=True)


if __name__ == "__main__":
    main()
