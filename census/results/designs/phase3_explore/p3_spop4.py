"""EXPLORATORY (population only; no seed of any sample).  The width-K conditional minimiser at fixed s = ‖v‖₁ on the
asymmetric windows and its placement status: minimise L(z, v) over z and v = s·softmax(w) (v ≥ 0 is no restriction:
f_a is odd, so a unit's sign flip (α, β, v) → (−α, −β, −v) is exact), by BFGS (Armijo) from R random starts
(α ~ U(−3, 3), β ~ U(−π, π), b ~ U(−1, 1), w ~ N(0, 1); numpy default_rng(seed)); the lowest loss is retained; status
from the exact G₊ enclosure.  NOT T2-1's validated search (no ladder, no CMA-ES): a feasibility estimate of s_pop4.
Usage: python p3_spop4.py K R s1 s2 ...   (bisect mode: python p3_spop4.py K R bisect s_lo s_hi n)"""
import json, math, sys
import numpy as np
import w4core as W
W.nice()
x, y = W.population()
K, R = int(sys.argv[1]), int(sys.argv[2])
T = W.Timer()


def unpack(p, s):
    z, w = p[:2 * K + 1], p[2 * K + 1:]
    e = np.exp(w - w.max()); sm = e / e.sum()
    return z, s * sm, sm


def fg(p, s):
    z, v, sm = unpack(p, s)
    L = W.loss(z, v, x, y)
    gz, gv = W.grad(z, v, x, y)
    gw = s * sm * (gv - sm @ gv)
    return L, np.concatenate([gz, gw])


def bfgs(p, s, gtol=1e-8, maxit=3000):
    L, g = fg(p, s)
    n = len(p); Hi = np.eye(n)
    for it in range(maxit):
        if np.abs(g).max() < gtol:
            break
        d = -Hi @ g
        if g @ d >= 0:
            Hi = np.eye(n); d = -g
        t = 1.0
        for _ in range(50):
            pn = p + t * d
            Ln, gn = fg(pn, s)
            if np.isfinite(Ln) and Ln <= L + 1e-4 * t * (g @ d):
                break
            t *= 0.5
        sv, yv = pn - p, gn - g
        sy = sv @ yv
        if sy > 1e-12:
            rho = 1 / sy; I = np.eye(n)
            Hi = (I - rho * np.outer(sv, yv)) @ Hi @ (I - rho * np.outer(yv, sv)) + rho * np.outer(sv, sv)
        p, L, g = pn, Ln, gn
        if np.abs(p[:2 * K]).max() > 50 or np.abs(p[2 * K + 1:]).max() > 40:
            break
    return p, L, float(np.abs(g).max())


def scale(s, seed):
    rng = np.random.default_rng(seed)
    best = []
    for r in range(R):
        p0 = np.concatenate([rng.uniform(-3, 3, K), rng.uniform(-math.pi, math.pi, K), rng.uniform(-1, 1, 1),
                             rng.normal(0, 1, K)])
        p, L, gm = bfgs(p0, s)
        best.append((L, gm, p))
    best.sort(key=lambda t: t[0])
    out = []
    seen = []
    for L, gm, p in best:
        if any(abs(L - l0) < 1e-9 for l0 in seen):
            continue
        seen.append(L)
        z, v, _ = unpack(p, s)
        lo, hi = W.gap_enc(z, v)
        out.append({"loss": L, "gmax": gm, "G_lo": lo, "G_hi": hi, "v_share": (v / s).tolist(), "type": W.btype(z, v),
                    "z": z.tolist(), "n_hit": sum(abs(b[0] - L) < 1e-9 for b in best)})
        if len(out) == 4:
            break
    top = out[0]
    st = "placed" if top["G_lo"] > 0 else ("unplaced" if top["G_hi"] <= 0 else "undecided")
    # near-tie warning: a candidate of the other status within 1e-6 of the retained loss
    tie = [o for o in out[1:] if o["loss"] - top["loss"] < 1e-6 and ((o["G_lo"] > 0) != (top["G_lo"] > 0))]
    print(f"s={s:.6g} [{T()} s] status {st} loss {top['loss']:.10f} gmax {top['gmax']:.1e} type {top['type']} "
          f"shares {np.round(top['v_share'], 3).tolist()} G [{top['G_lo']:.4g},{top['G_hi']:.4g}] hits {top['n_hit']}/{R} "
          f"| next: " + "; ".join(f"{o['loss'] - top['loss']:.2e} {o['type']} G {o['G_lo']:.3g}" for o in out[1:]),
          ("TIE-WARNING" if tie else ""), flush=True)
    W.rss_guard()
    return st, out


res = {}
if sys.argv[3] == "bisect":
    lo, hi, nb = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    for i in range(nb):
        mid = math.sqrt(lo * hi)
        st, out = scale(mid, 7000 + i)
        res[f"{mid:.8g}"] = out
        if st == "placed":
            hi = mid
        else:
            lo = mid
    print(f"bracket [{lo:.6g}, {hi:.6g}]  geometric mean {math.sqrt(lo * hi):.6g}")
    res["bracket"] = [lo, hi]
else:
    for j, a in enumerate(sys.argv[3:]):
        st, out = scale(float(a), 100 + j)
        res[a] = out
json.dump(res, open(f"p3_spop{K}_{sys.argv[3]}.json", "w"), indent=0)
print("peak RSS GB", round(W.rss_gb(), 3))
