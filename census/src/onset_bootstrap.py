"""Bootstrap the FULL onset pipeline: resample seeds within each cell,
redetermine every onset by the bracketing rule, refit the exponent.

Propagating a per-cell uncertainty would assume independence across cells.
The observed consistent offset between two pipelines suggests correlated
errors within a window, which tilt the fit rather than averaging out.
This measures the exponent's sampling distribution directly.

Needs per-seed outcomes, not cell rates, so it re-runs the grid once and
caches every individual solve/fail.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.fold1d import activation, logits, make_data, solves

R = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40
GRIDS = {2_000:[1.70,1.60,1.55,1.50,1.45],
         8_000:[1.30,1.25,1.20,1.18,1.16,1.14],
         32_000:[1.12,1.10,1.08,1.06,1.05,1.04],
         128_000:[1.06,1.05,1.04,1.035,1.03,1.025]}


def outcomes(a: float, B: int) -> np.ndarray:
    f = activation("sin_family", a)
    out = np.zeros(SEEDS, dtype=bool)
    for s in range(SEEDS):
        x, y = make_data(200, s); torch.manual_seed(s)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=1e-2)
        for _ in range(B):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
            opt.step()
        out[s] = solves(th.detach(), f)
    return out


def main() -> None:
    cache = R / "onset_bootstrap_outcomes.csv"
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        rows = []
        for B, grid in GRIDS.items():
            for a in grid:
                o = outcomes(a, B)
                rows.append({"budget": B, "a": a,
                             "outcomes": "".join("1" if v else "0" for v in o)})
                print(f"  cached B={B} a={a}: rate={o.mean():.3f}", flush=True)
        d = pd.DataFrame(rows)
        with artifact_lock(R / "onset_bootstrap_outcomes", "bootstrap outcomes"):
            d.to_csv(cache, index=False)

    arr = {(int(r.budget), float(r.a)): np.array([c == "1" for c in r.outcomes])
           for r in d.itertuples()}
    rng = np.random.default_rng(0)

    def fit_once(resample: bool) -> float | None:
        onsets = {}
        for B, grid in GRIDS.items():
            onset = None
            for a in grid:
                o = arr[(B, a)]
                r = (o[rng.integers(0, SEEDS, SEEDS)].mean() if resample else o.mean())
                if r >= 0.5:
                    onset = a
                elif onset is not None:
                    break
            if onset is not None:
                onsets[B] = onset
        if len(onsets) < 3:
            return None
        b = np.array(sorted(onsets)); e = np.array([onsets[k] for k in b]) - 1.0
        return float(np.polyfit(np.log(b), np.log(e), 1)[0])

    point = fit_once(False)
    boot = [fit_once(True) for _ in range(600)]
    boot = np.array([v for v in boot if v is not None])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\npoint estimate (no resampling): {point:.4f}")
    print(f"bootstrap n={len(boot)}: mean {boot.mean():.4f}  sd {boot.std():.4f}")
    print(f"95% interval: [{lo:.4f}, {hi:.4f}]   half-width {(hi-lo)/2:.4f}")
    print(f"grid-resolution term previously propagated: +-0.1140")
    print(f"ratio: {((hi-lo)/2)/0.1140:.2f}x")
    Path(R / "onset_bootstrap_fit.txt").write_text(
        f"point={point}\nmean={boot.mean()}\nsd={boot.std()}\n"
        f"ci_lo={lo}\nci_hi={hi}\nhalf_width={(hi-lo)/2}\nn={len(boot)}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
