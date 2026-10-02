"""EXPLORATORY timing pilot (s only, no ρ₂): steps to each matched scale per condition on exploration seed 2,953,000."""
import json, sys, time
import numpy as np
sys.path.insert(0, "results/designs/phase2b_explore")
import p2b_explore as E
th0, _ = E.init_row(E.SEEDS[0])
for cond in sys.argv[1:]:
    spec = E.CONDS[cond]; st = E.make_stepper(spec); th = th0.copy(); t = 0; ts = {}; t0 = time.time()
    while t < spec["budget"]:
        s = float(np.abs(th[12:16]).sum())
        for k, v in E.SCALES.items():
            if k not in ts and s >= v: ts[k] = t
        if s >= E.SCALES["3s*"] or not np.isfinite(th).all(): break
        th = st(th); t += 1
    print(json.dumps({"cond": cond, "seed": E.SEEDS[0], "steps": t, "s_end": s, "t_scale": ts, "seconds": time.time() - t0}), flush=True)
