"""EXPLORATORY (population only; no candidate seed).  Where random starts land at the held output v₀ = (s₀/2, s₀/2), as a
function of s₀: 150 hidden starts z ~ U(−1, 1)⁵ (numpy default_rng(20260930)), held by full-batch GD on z at lr 1.0 for
4,000 steps, then damped Newton at v₀; counts of: accepted duplicate unplaced / two-unit placed / other, and the
release-state distance to the Newton point (max over runs), for s₀ in S0S.  Writes w2_explore_d.json."""
import json

import numpy as np

import w2core as W

W.nice()
x, y = W.setup(); ac = W.act()
S0S = (0.05, 0.1, 0.15, 0.2225, 0.3, 0.445)
N, WSTEPS, LR = 150, 4000, 1.0
out = {}
for s0 in S0S:
    v0 = np.array([s0 / 2, s0 / 2])
    rng = np.random.default_rng(20260930)
    cnt, dist, lam = {}, 0.0, []
    for i in range(N):
        z = rng.uniform(-1, 1, 5)
        for t in range(WSTEPS):
            z = z - LR * W.gz(z, v0, x, y)
        zn, ok, _ = W.newton(z, v0, x, y)
        if not ok:
            key = "not accepted"
        else:
            kind = W.canon(zn, v0)[0]
            key = f"{kind} {'placed' if W.gap(zn, v0, ac)[0] > 0 else 'unplaced'} {W.canon(zn, v0)[1]}"
            dist = max(dist, float(np.abs(z - zn).max()))
            lam.append(float(np.linalg.eigvalsh(W.Hz(zn, v0, x, y)).min()))
        cnt[key] = cnt.get(key, 0) + 1
    out[str(s0)] = {"counts": cnt, "max_release_dist": dist, "lam_min_range": [min(lam), max(lam)] if lam else None}
    print(s0, json.dumps(out[str(s0)]), flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
