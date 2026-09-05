"""P-composition: does excluding stallers lower alpha toward 1.0?
Plus 1c: measure alpha_SGD against the registered band [0.38, 0.58].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifact_lock import artifact_lock
from .budget_law import run_full

RESULTS = Path(__file__).resolve().parents[1] / "results"


def sweep(optimizer_name: str, budgets, lr: float, a: float = 1.25,
          seeds: int = 30) -> pd.DataFrame:
    rows = []
    for B in budgets:
        for s in range(seeds):
            t, ok, loss, stalled = run_full(a, s, B, lr) if optimizer_name == "adam" \
                else _run_sgd(a, s, B, lr)
            rows.append({"optimizer": optimizer_name, "budget": B, "seed": s,
                         "w2": abs(float(t[2])), "solved": ok,
                         "loss": loss, "stalled": stalled})
        sub = [r for r in rows if r["budget"] == B]
        esc = [r["w2"] for r in sub if not r["stalled"]]
        print(f"{optimizer_name} B={B}: all_med={np.median([r['w2'] for r in sub]):.3f} "
              f"escaper_med={np.median(esc) if esc else float('nan'):.3f} "
              f"stall={sum(r['stalled'] for r in sub)}/{len(sub)}", flush=True)
    return pd.DataFrame(rows)


def _run_sgd(a, seed, budget, lr):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data, solves
    LOG2 = float(np.log(2.0))
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.SGD([theta], lr=lr)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        opt.step()
    t = theta.detach()
    final = float(F.binary_cross_entropy_with_logits(logits(t, x, f), y))
    return t, bool(solves(t, f)), final, abs(final - LOG2) < 1e-4


def fit(frame, label):
    out = {}
    for name, sel in (("all", frame), ("escapers", frame[~frame.stalled])):
        med = sel.groupby("budget").w2.median()
        s = np.polyfit(np.log(med.index.values), np.log(med.values), 1)[0]
        out[name] = s
        print(f"  {label} alpha ({name}) = {s:.4f}")
    return out


def main() -> None:
    budgets = (1_000, 2_000, 4_000, 8_000, 16_000, 40_000)
    adam = sweep("adam", budgets, 1e-2)
    print("\n--- Adam ---"); a_fit = fit(adam, "adam")
    print(f"  P-composition: escaper alpha {'LOWER' if a_fit['escapers'] < a_fit['all'] else 'NOT lower'} "
          f"than population alpha (predicted: lower)")

    sgd = sweep("sgd", budgets, 0.3)
    print("\n--- SGD ---"); s_fit = fit(sgd, "sgd")
    lo, hi = 0.38, 0.58
    print(f"  P-1c registered alpha_SGD = 0.479, band [{lo}, {hi}]")
    print(f"  VERDICT: {'WITHIN BAND' if lo <= s_fit['all'] <= hi else 'OUTSIDE BAND'}")

    frame = pd.concat([adam, sgd])
    stem = RESULTS / "alpha_composition"
    with artifact_lock(stem, "alpha composition"):
        tmp = stem.with_suffix(".csv.tmp"); frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    Path(RESULTS / "alpha_composition_fit.txt").write_text(
        f"adam_all={a_fit['all']}\nadam_escapers={a_fit['escapers']}\n"
        f"sgd_all={s_fit['all']}\nsgd_escapers={s_fit['escapers']}\n"
        f"registered_sgd=0.479 band=[0.38,0.58]\n"
        f"within={lo <= s_fit['all'] <= hi}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
