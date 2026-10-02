"""EXPLORATORY (NOT a registration): power per arm for the revised 2B page (author's decisions 2026-10-02), by
resampling the exploration seeds' (2,953,000-2,953,019; never registered) paired differences. Local generators only.
Output arms (out16 primary, out16b secondary): R_s (9 scale cells: rho2, shuffled, reversed at 1.25/2/3 s*),
R_l-rho2 (rho2 at BCE 0.1/0.03/0.01), R_l-acc (shuffled and reversed at those BCE levels): lower end > 0; O: lasting
onset (to T_C = 40,000) above s* in >= 75%. Global arms (glob /12.18, globcm /16.14): N, 18 cells, upper end < delta.
    python p2b_lever_power2.py   ->  p2b_lever_power2.json
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import p2b_lever_power as P

SC = [i for i, p in enumerate(P.POINTS) if p.startswith("scale")]
LO = [i for i, p in enumerate(P.POINTS) if p.startswith("loss")]
nq = len(P.QTY)
cols = lambda pts, qs: [i * nq + k for i in pts for k in qs]  # noqa: E731
GROUPS = {"R_s": cols(SC, (0, 1, 2)), "R_l_rho2": cols(LO, (0,)), "R_l_acc": cols(LO, (1, 2))}
MARG = np.tile([P.DELTA[q] for q in P.QTY], len(P.POINTS))


def verdict_out(D, on, B, rng):
    lo, _ = P.ci_median(D, B, rng)
    v = {g: bool((lo[c] > 0).all()) for g, c in GROUPS.items()}
    v["O"] = bool(on.mean() >= P.ONSET_MIN); v["all"] = all(v.values()); return v


def verdict_glob(D, B, rng):
    _, hi = P.ci_median(D, B, rng)
    return {"N": bool((hi < MARG).all())}


def main():
    std = P.load("std")
    res = {"label": "EXPLORATORY", "arms": {}, "sims": 300, "inner_bootstrap": 1000, "rng": P.SEED_REG + 10}
    for arm, kind in (("out16", "out"), ("out16b", "out"), ("glob", "glob"), ("globcm", "glob")):
        A = P.load(arm); D = P.paired(std, A)
        on = np.array([bool(A[s]["onset"].get("T40000")) and A[s]["onset"]["T40000"]["s_over_s_star"] > 1 for s in P.SEEDS])
        rng = np.random.default_rng(P.SEED_REG)
        here = verdict_out(D, on, P.B_REG, rng) if kind == "out" else verdict_glob(D, P.B_REG, rng)
        rng = np.random.default_rng(P.SEED_REG + 10)
        pw = {}
        for n in (60, 80):
            acc = {}
            for _ in range(res["sims"]):
                rows = rng.integers(0, len(P.SEEDS), n)
                v = verdict_out(D[rows], on[rows], 1000, rng) if kind == "out" else verdict_glob(D[rows], 1000, rng)
                for k, x in v.items():
                    acc[k] = acc.get(k, 0) + x
            pw[n] = {k: x / res["sims"] for k, x in acc.items()}
        res["arms"][arm] = {"on_exploration_seeds": here, "power": pw, "onset_above_s*": int(on.sum())}
        print(arm, here, pw, flush=True)
    (P.HERE / "p2b_lever_power2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
