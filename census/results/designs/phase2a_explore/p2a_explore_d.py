"""EXPLORATORY (population landscape; the start point comes from non-registered exploration seed 2,930,000): the
'other' landing class of p2a_explore_b (a mirror pair only, no x₁ unit; call it Mp) continued in s by sb_fold's
pseudo-arclength continuation (h = 0.05 and 0.025): stable range, folds, ρ₂ and G₊ along it, its loss against M, L0, S, S2."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from results.designs.phase2a_explore.p2a_explore_b import init_row
from src import sb_fold as F, simplicity_bias as sb

s0 = 1.7957
row = init_row(2_930_000); row[12:16] *= s0 / np.abs(row[12:16]).sum()
P, L, gn = F.local_min_batch(F.run_to_full(row[None]), np.array([s0]), C.X, C.Y)
u0, nA = F.Reduced.from_full(P[0], s0, C.X, C.Y)
R = F.Reduced(nA, C.X, C.Y)
u0, ok, fres = F.newton_fixed_s(R.F, R.JF, u0, s0)
x0 = np.r_[u0, s0]
out = {"label": "EXPLORATORY", "n_active": nA, "newton_ok": ok, "loss_s0": R.loss(u0, s0)}
for d in (+1, -1):
    for h in (0.05, 0.025):
        pts, folds = F.continuation(R.F, R.JF, x0, d, h, n_max=int(40 / h), s_bounds=(0.05, 15.0))
        key = f"{'fwd' if d > 0 else 'bwd'}_h{h}"
        out[key] = {"folds": [{k: f[k] for k in ("s_fold", "eig_min")} for f in folds], "s_end": pts[-1]["s"], "n": len(pts)}
        if h == 0.05:
            tab = []
            for p in pts[:: max(1, len(pts) // 25)] + [pts[-1]]:
                Pf, _ = R.to_full(p["x"][:-1]); s = p["s"]
                rb = {}
                for nm in ("M", "L0", "S", "S2"):
                    bp = F.branch_point(nm, s)
                    rb[nm] = None if bp is None else bp[1].loss(bp[0], s)
                tab.append({"s": s, "eig_min": p["eig_min"], "rho2": sb.feature_usage(Pf, C.X)["rho2"],
                            "gplus": float(sb.gplus(Pf[None], C.X, C.Y)[0]), "loss": R.loss(p["x"][:-1], s), "other_branch_losses": rb})
            out[key]["table"] = tab
        print(json.dumps({key: {k: v for k, v in out[key].items() if k != "table"}}), flush=True)
open("results/designs/phase2a_explore/p2a_explore_d.json", "w").write(json.dumps(out, indent=1, default=float))
for r in out["fwd_h0.05"]["table"] + out["bwd_h0.05"]["table"]:
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "other_branch_losses"}),
          {k: (round(v, 5) if v else None) for k, v in r["other_branch_losses"].items()})
