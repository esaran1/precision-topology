"""Block 4a (POST HOC, exploratory): what sets the residual?  From the deconfounded lag test's saved φ = 1 states.

For each run (a ∈ {1.30, 1.50}, the 48 included seeds per a, n = 6,400 training points), at two points of the same
φ = 1 trajectory -- the plateau exit (t*, `tstar_*.pt`) and the 0.7× own-threshold switch point (`primary_*.pt`):
  - the symmetry-aware distance from (w₁, b₁) to the occupied branch's conditional minimiser at the current scale
    |w₂| (canonical orientation w₂ > 0, b₁ mod 2π; `mirror_branches.half_min`, step 0.05, 16 refinements, as the
    lag test's own diagnostic);
  - the component of that displacement along the softest eigenvector of the profiled Hessian at the minimiser
    (finite differences of the envelope gradient);
  - Adam's bias-corrected second moments v̂ for w₁, b₁, w₂, b₂ (and log ratios).
Each is correlated (Spearman, permutation p) with the run's final residual R_cross/R_own − 1 at φ = 1 (the lag test's
R_cross and R_own).  Nothing here is registered; it motivates the 4b design only.

    python -m src.residual_posthoc [compute | score]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
STATES = RESULTS / "lag_test2_states"
PARTS = RESULTS / "residual_posthoc_parts.csv"
N = 6400


def _hessian(s, a, x, y, w, b, eps=1e-5):
    from .profiled_bnb import profile
    def g(ww, bb):
        _, _, gw, gb, _ = profile(np.array([ww]), np.array([bb]), s, a, x, y)
        return np.array([gw[0], gb[0]])
    Hw = (g(w + eps, b) - g(w - eps, b)) / (2 * eps)
    Hb = (g(w, b + eps) - g(w, b - eps)) / (2 * eps)
    H = np.array([Hw, Hb]); return 0.5 * (H + H.T)


def one(a, seed, point):
    import torch
    from .mirror_branches import half_min
    from .sample_size import _data
    st = torch.load(STATES / f"{point}_a{a:.2f}_s{seed}.pt", weights_only=False)
    w1, b1, w2, b2 = (float(v) for v in st["theta"])
    x, y = _data(N, seed)
    sg = np.sign(w2)
    cw, cb = w1 * sg, (b1 * sg) % (2 * math.pi)
    br = int(np.sign(cw))
    s = abs(w2)
    _, mw, mb, _ = half_min(s, a, x, y, br, step=0.05, n_refine=16)
    db = (cb - mb + math.pi) % (2 * math.pi) - math.pi
    d = np.array([cw - mw, db])
    H = _hessian(s, a, x, y, mw, mb)
    ev, V = np.linalg.eigh(H)
    soft = V[:, 0]
    opt = st["opt"]["state"][0]
    t = int(opt["step"])
    v = opt["exp_avg_sq"].numpy() / (1 - 0.999 ** t)
    m = opt["exp_avg"].numpy() / (1 - 0.9 ** t)
    return {"a": a, "seed": seed, "point": point, "step": t, "s": s, "branch": br,
            "dist": float(np.hypot(*d)), "soft_component": float(abs(d @ soft)), "stiff_component": float(abs(d @ V[:, 1])),
            "hess_min": float(ev[0]), "hess_max": float(ev[1]),
            **{f"sqrt_vhat_{k}": float(np.sqrt(v[i])) for i, k in enumerate(("w1", "b1", "w2", "b2"))},
            "log_vhat_ratio_w2_w1": float(np.log(v[2] / v[0])), "mhat_w2": float(m[2])}


def compute():
    ck = pd.read_csv(RESULTS / "lag_test2_checkpoints.csv")
    ck = ck[(ck.status_primary == "ok") & (ck.status_tstar == "ok")]
    done = set() if not PARTS.exists() else {(round(r.a, 2), int(r.seed), r.point) for r in pd.read_csv(PARTS).itertuples()}
    for r in ck.itertuples():
        for point in ("tstar", "primary"):
            key = (round(float(r.a), 2), int(r.seed), point)
            if key in done:
                continue
            row = one(float(r.a), int(r.seed), point)
            pd.DataFrame([row]).to_csv(PARTS, mode="a", header=not PARTS.exists(), index=False)
            print(row["a"], row["seed"], point, round(row["dist"], 4), flush=True)


def _perm_spearman(a, b, n_perm=10_000, seed=0):
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, float((1 + (np.abs(perm) >= abs(rho)).sum()) / (1 + n_perm))


def score():
    d = pd.read_csv(PARTS)
    ck = pd.read_csv(RESULTS / "lag_test2_checkpoints.csv")[["a", "seed", "R_own"]]
    ck["R_own"] = pd.to_numeric(ck.R_own, errors="coerce")       # excluded runs carry a text note; they drop out
    ck = ck.dropna()
    runs = pd.read_csv(RESULTS / "lag_test2_runs.csv")
    base = runs[(runs.rule == "primary") & (runs.factor == 1.0)][["a", "seed", "R_cross"]]
    res = base.merge(ck, on=["a", "seed"]); res["residual"] = res.R_cross / res.R_own - 1
    d = d.merge(res[["a", "seed", "residual"]], on=["a", "seed"])
    feats = ["dist", "soft_component", "stiff_component", "sqrt_vhat_w1", "sqrt_vhat_b1", "sqrt_vhat_w2", "sqrt_vhat_b2",
             "log_vhat_ratio_w2_w1"]
    rows = []
    for (a, point), g in d.groupby(["a", "point"]):
        for f in feats:
            rho, p = _perm_spearman(g[f].values, g.residual.values)
            rows.append({"a": a, "point": point, "feature": f, "n": len(g), "spearman_rho": rho, "perm_p_two_sided": p,
                         "median_feature": float(g[f].median())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "residual_posthoc_correlations.csv", index=False)
    d.to_csv(RESULTS / "residual_posthoc_runs.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    {"compute": compute, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "compute"]()
