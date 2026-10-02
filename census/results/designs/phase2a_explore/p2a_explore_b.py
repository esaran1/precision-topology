"""EXPLORATORY landing at a held scale s0 (NON-registered exploration seeds 2,930,000-2,930,199; never to be registered).
Init: the PyTorch-default distributions of Linear(2, 4) and Linear(4, 1) (U(±1/√2) for W, c; U(±1/2) for v, b), drawn
from a LOCAL numpy Generator(seed) (no global RNG; not bit-identical to torch.manual_seed draws).  Hold: v rescaled to
‖v‖₁ = s0, then local minimisation of the fixed-s0 landscape (v2 objective, b profiled, 16 coordinates, BFGS, gtol 1e-8:
sb_fold.local_min_batch, the Track 1A basin classifier).  Label: the branch (L0, M, S, S2) whose point at s0 lies within
1e-3 (function space) of the minimiser, else 'other'.  Also ρ₂ and G₊ of the minimiser.

    python p2a_explore_b.py S0 [N]
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from src import sb_fold as F

SEED0 = 2_930_000


def init_row(seed):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1 / np.sqrt(2), 1 / np.sqrt(2), (4, 2)); c = rng.uniform(-1 / np.sqrt(2), 1 / np.sqrt(2), 4)
    v = rng.uniform(-0.5, 0.5, 4); b = rng.uniform(-0.5, 0.5)
    return np.r_[W.ravel(), c, v, b]


def main(s0, n):
    rows = np.array([init_row(SEED0 + i) for i in range(n)])
    rows[:, 12:16] *= s0 / np.abs(rows[:, 12:16]).sum(axis=1, keepdims=True)
    P0 = F.run_to_full(rows)
    P, L, gn = F.local_min_batch(P0, np.full(n, s0), C.X, C.Y)
    grids = {}
    for name in ("L0", "M", "S", "S2"):
        z = np.load(f"results/sb_fold/grid_{name}.npz")
        grids[name] = (z["s"], z["P"], z["U"], F.Reduced(int(z["n_active"]), C.X, C.Y))
    lab, dist = F.classify_rows(P, np.full(n, s0), grids, C.X)
    from src import simplicity_bias as sb
    rho = np.array([sb.feature_usage(p, C.X)["rho2"] for p in P])
    gp = sb.gplus(P, C.X, C.Y)
    counts = {k: int((lab == k).sum()) for k in ("L0", "M", "S", "S2", "other")}
    oth = lab == "other"
    res = {"label": "EXPLORATORY (non-registered seeds 2,930,000+; local RNG)", "s0": s0, "n": n, "counts": counts,
           "gnorm_max": float(gn.max()), "n_gnorm_gt_1e-6": int((gn > 1e-6).sum()),
           "rho2_by_label": {k: [float(np.min(rho[lab == k])), float(np.median(rho[lab == k])), float(np.max(rho[lab == k]))]
                             for k in counts if counts[k]},
           "loss_by_label": {k: float(np.median(L[lab == k])) for k in counts if counts[k]},
           "other_min_dist": [float(min(dist[k][i] for k in dist)) for i in np.flatnonzero(oth)][:50],
           "per_seed": [{"seed": SEED0 + i, "label": str(lab[i]), "rho2": float(rho[i]), "gplus": float(gp[i]),
                         "loss": float(L[i])} for i in range(n)]}
    open(f"results/designs/phase2a_explore/p2a_explore_b_s{s0:.4f}.json", "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s0", "n", "counts", "gnorm_max", "n_gnorm_gt_1e-6", "rho2_by_label", "loss_by_label")}))


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
