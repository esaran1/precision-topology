"""EXPLORATORY follow check (population, release at the exact M(s0), scale-only form): states at s = 0.8·s_F (past the
switch) and 0.95·s_F are minimised at their own s (sb_fold.local_min_batch, the Track 1A basin classifier) and labelled;
raw function-space distance to M(s) as well.  ρ = 2^-11 and 2^-14."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from results.designs.phase2a_explore.p2a_explore_c import step_fn
from src import sb_fold as F

out = []
grids = {}
for name in ("L0", "M", "S", "S2"):
    z = np.load(f"results/sb_fold/grid_{name}.npz")
    grids[name] = (z["s"], z["P"], z["U"], F.Reduced(int(z["n_active"]), C.X, C.Y))
for l2 in (-11, -14):
    th = C.branch_theta("M", 1.7957); step = step_fn("scale", 2.0 ** l2, 1.0, [3] if th[15] == 0 else [k for k in range(4) if th[12 + k] == 0])
    for frac in (0.8, 0.95):
        while C.scale(th) < frac * C.S_FOLD:
            th, _ = step(th)
        s = C.scale(th)
        P0 = C.to_landscape(th)[None]
        P, L, gn = F.local_min_batch(P0, np.array([s]), C.X, C.Y)
        lab, d = F.classify_rows(P, np.array([s]), grids, C.X)
        raw = C.branch_dists(th, ("M",))["M"]
        r = {"log2rho": l2, "frac_of_fold": frac, "s": s, "basin": str(lab[0]), "basin_dist_M": float(d["M"][0]), "raw_dist_M": raw}
        out.append(r); print(json.dumps(r), flush=True)
open("results/designs/phase2a_explore/p2a_explore_e.json", "w").write(json.dumps({"label": "EXPLORATORY", "rows": out}, indent=1))
