"""Independent checker for the exported certificates (Block 2c).

It shares NO code with the searches: it imports nothing from src/ and reads only the certificate data files
(results/certificates/*.npz + *.json).  Every objective is re-implemented here from its mathematical definition,
in ball arithmetic (python-flint / Arb; the searches used float Lipschitz bounds, numpy outward rounding, or
mpmath.iv -- never Arb).  Every enclosure is rigorous, including rounding.

Finite-a conditional certificate (one status at one scale s):
  objective   L*(w₁, b₁; s) = min_{b₂} mean softplus(z) − y z,  z = s·f_a(w₁x + b₁) + b₂,  f_a(t) = t + a sin t
  class gap   G(w₁, b₁) = min_O f_a(w₁x + b₁) − max_I f_a(w₁x + b₁)   (orientation w₂ > 0)
  domain      |w₁| ≤ W (the localisation lemma, math_note_v2 §5.0, recomputed here), b₁ ∈ [0, 2π)
  claim       status 'minus': min over {G > 0} > min over {G ≤ 0};  status 'plus': the reverse.
  checks      (1) W recomputed from the lemma's formula on the exported data equals the exported W (to 1e−12)
              (2) the leaves tile the domain exactly (integer quadtree indices: areas sum to 1, no leaf is an
                  ancestor of another, all inside the grid)
              (3) the winning region's value: a rigorous upper bound U at the exported point, which is verified to
                  lie in that region (or the constant predictor, exactly log 2 with ȳ = ½)
              (4) every leaf of the losing region: rigorously outside that region, or a rigorous lower bound of L*
                  over the leaf exceeds U  (subdividing a leaf up to MAX_DEPTH if needed)
              (5) every leaf's own claimed lower bound (both regions) holds (same subdivision rule)
              (6) U < log 2 when the winning region is {G > 0}, so the lemma's region |w₁| > W (L* > log 2) loses.

    python -m src.verify_certificates [certificate-name ...]     # all certificates if none given
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from flint import arb, ctx

CERTS = Path(__file__).resolve().parents[1] / "results" / "certificates"
PREC = 80                      # bits; Arb tracks every radius, so the result is rigorous at any precision
MAX_DEPTH = 6
WORKERS = 3


# ------------------------------------------------------------------------------------------ ball helpers
def ball(lo, hi):
    """The arb ball containing the closed interval [lo, hi] (floats, exact)."""
    lo, hi = arb(lo), arb(hi)
    return (lo + hi) / 2 + arb(0, ((hi - lo) / 2).upper())


def lo_(b):
    return b.lower()


def hi_(b):
    return b.upper()


# ------------------------------------------------------------------------------------------ f_a and its exact range
class FA:
    def __init__(self, a):
        self.a = arb(a)
        self.af = float(a)
        # critical points of f_a'(t) = 1 + a cos t = 0: t = ±acos(−1/a) + 2πk (a > 1 only)
        self.c = (-1 / self.a).acos() if self.af > 1 else None
        self.twopi = 2 * arb.pi()

    def f(self, t):
        return t + self.a * t.sin()

    def df(self, t):
        return 1 + self.a * t.cos()

    def range(self, tlo, thi):
        """Rigorous [min, max] of f_a over [tlo, thi] (tlo, thi arb): endpoints and the critical points inside."""
        cand = [self.f(tlo), self.f(thi)]
        if self.c is not None:
            for sgn in (1, -1):
                cc = sgn * self.c
                k0 = int(math.floor((float(tlo.mid()) - float(cc.mid())) / (2 * math.pi))) - 1
                k1 = int(math.ceil((float(thi.mid()) - float(cc.mid())) / (2 * math.pi))) + 1
                for k in range(k0, k1 + 1):
                    tk = cc + k * self.twopi
                    if hi_(tk) < lo_(tlo) or lo_(tk) > hi_(thi):
                        continue
                    cand.append(self.f(tk))       # a critical point possibly inside: its value is attained or not;
                                                  # including it can only widen the enclosure (still rigorous)
        mn = min((lo_(c) for c in cand))
        mx = max((hi_(c) for c in cand))
        return mn, mx


def t_interval(wl, wh, bl, bh, xl, xh):
    """Range of w·x + b over w ∈ [wl, wh], x ∈ [xl, xh], b ∈ [bl, bh] (arb endpoints, exact products)."""
    prods = [wl * xl, wl * xh, wh * xl, wh * xh]
    lo = min(lo_(p) for p in prods) + lo_(bl)
    hi = max(hi_(p) for p in prods) + hi_(bh)
    return lo, hi


# ------------------------------------------------------------------------------------------ the finite-a objective
class Objective:
    def __init__(self, x, y, s, a):
        ctx.prec = PREC
        self.xf = np.asarray(x, float); self.yf = np.asarray(y, float)
        self.X = [arb(float(v)) for v in self.xf]
        self.Y = [int(v) for v in self.yf]
        self.n = len(self.X)
        self.s = arb(s); self.fa = FA(a)
        k = sum(self.Y)
        self.ybar = arb(k) / self.n

    def z0_ranges(self, W, B):
        """Per point, a ball for z0 = s·f_a(w x + b) over the cell W × B (balls)."""
        out = []
        for xi in self.X:
            t = W * xi + B
            out.append(self.s * self.fa.f(t))
        return out

    def b_bracket(self, W, B):
        """A ball containing b₂*(w, b₁) for every (w, b₁) in the cell, by monotone bounds of F(b) = mean σ(z0 + b) − ȳ."""
        Z = self.z0_ranges(W, B)
        zl = np.array([float(lo_(z)) for z in Z]); zh = np.array([float(hi_(z)) for z in Z])

        def root(z):
            lo, hi = -80.0, 80.0
            for _ in range(90):
                m = 0.5 * (lo + hi)
                v = (1 / (1 + np.exp(-(z + m)))).mean() - float(self.ybar.mid())
                lo, hi = (m, hi) if v < 0 else (lo, m)
            return 0.5 * (lo + hi)
        beta, gamma = root(zh) - 1e-12, root(zl) + 1e-12

        def F(Zb, b):
            acc = arb(0)
            for z in Zb:
                acc += 1 / (1 + (-(z + b)).exp())
            return acc / self.n - self.ybar
        Zhi = [arb(hi_(z)) for z in Z]; Zlo = [arb(lo_(z)) for z in Z]
        ok = (F(Zhi, arb(beta)) < 0) and (F(Zlo, arb(gamma)) > 0)
        return (ball(beta, gamma) if ok else None), Z

    def value_and_grad(self, W, B, Bb, Z=None):
        """Balls for L(w, b₁, b₂) and the envelope gradient over W × B × Bb."""
        L = arb(0); gw = arb(0); gb = arb(0)
        for i, (xi, yi) in enumerate(zip(self.X, self.Y)):
            t = W * xi + B
            z = (self.s * self.fa.f(t) if Z is None else Z[i]) + Bb
            L += (z.exp() + 1).log() - (z if yi else 0)
            r = 1 / (1 + (-z).exp()) - yi
            d = r * self.s * self.fa.df(t)
            gw += d * xi; gb += d
        return L / self.n, gw / self.n, gb / self.n

    def lower_bound(self, wc, bc, hw, hb):
        """Rigorous lower bound of L* over the cell [wc ± hw] × [bc ± hb] (mean-value form)."""
        Wc, Bc = arb(wc), arb(bc)
        Bpt, _ = self.b_bracket(Wc, Bc)
        if Bpt is None:
            return -math.inf
        # L*(c) = L(c, b*) with b* ∈ Bpt.  L(c, ·) is convex, so L(c, b*) ≥ L(c, b̂) + F(b̂)(b* − b̂) with
        # F = ∂L/∂b = mean σ(z) − ȳ: evaluate at the single point b̂ (no dependency loss over the bracket).
        bhat = arb(float(Bpt.mid()))
        Lpt, _, _ = self.value_and_grad(Wc, Bc, bhat)
        Fh = self.F_at(Wc, Bc, bhat)
        Lc = arb(lo_(Lpt)) - abs(Fh) * (2 * arb(Bpt.rad()) + abs(bhat - Bpt))
        W, B = ball(wc - hw, wc + hw), ball(bc - hb, bc + hb)
        Bcell, Z = self.b_bracket(W, B)
        if Bcell is None:
            return -math.inf
        _, gw, gb = self.value_and_grad(W, B, Bcell, Z)
        lb = lo_(Lc) - hi_(abs(gw)) * arb(hw) - hi_(abs(gb)) * arb(hb)
        return float(lo_(lb))

    def F_at(self, W, B, b):
        """F(b) = mean σ(z0 + b) − ȳ at a (w, b₁) ball."""
        acc = arb(0)
        for xi in self.X:
            z = self.s * self.fa.f(W * xi + B) + b
            acc += 1 / (1 + (-z).exp())
        return acc / self.n - self.ybar

    def upper_at(self, w, b1, b2):
        L, _, _ = self.value_and_grad(arb(w), arb(b1), arb(b2))
        return float(hi_(L))

    def gap_bounds(self, wl, wh, bl, bh, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
        """Rigorous (G_lower, G_upper) over the cell, G = min_O φ − max_I φ, φ = f_a(w x + b₁)."""
        wl, wh, bl, bh = arb(wl), arb(wh), arb(bl), arb(bh)
        fa = self.fa
        ti = t_interval(wl, wh, bl, bh, arb(inner[0]), arb(inner[1]))
        to1 = t_interval(wl, wh, bl, bh, arb(outer[0]), arb(outer[1]))
        to2 = t_interval(wl, wh, bl, bh, arb(-outer[1]), arb(-outer[0]))
        imn, imx = fa.range(*ti)
        o1mn, _ = fa.range(*to1); o2mn, _ = fa.range(*to2)
        G_lower = float(lo_(min(o1mn, o2mn) - imx))
        # upper: G(θ) ≤ φ(x_o, θ) − φ(x_i, θ) for any fixed x_o ∈ O, x_i ∈ I; take the best of a few pairs
        wm, bm = float(((wl + wh) / 2).mid()), float(((bl + bh) / 2).mid())
        xs_i = np.linspace(inner[0], inner[1], 81); xs_o = np.r_[np.linspace(*outer, 41), -np.linspace(*outer, 41)]
        fi = xs_i * wm + bm; fo = xs_o * wm + bm
        fi = fi + fa.af * np.sin(fi); fo = fo + fa.af * np.sin(fo)
        xo, xi = float(xs_o[np.argmin(fo)]), float(xs_i[np.argmax(fi)])
        _, fomax = fa.range(*t_interval(wl, wh, bl, bh, arb(xo), arb(xo)))
        fimin, _ = fa.range(*t_interval(wl, wh, bl, bh, arb(xi), arb(xi)))
        G_upper = float(hi_(fomax - fimin))
        return G_lower, G_upper


# ------------------------------------------------------------------------------------------ the localisation lemma
def lemma_W(x, y, s, a, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """W(s, a) from math_note_v2 §5.0, as stated: min over the grid C = {0.799k/79} of (2a + δ(π(c))/s)/(1.2 + c), per
    side, then the larger side.  In floats (the same formula the text states); compared with the exported W."""
    n = len(x)
    out = []
    for sign in (+1, -1):
        n_outer = int(((sign * x <= -outer[0]) & (y == 1)).sum())
        best = math.inf
        for k in range(80):
            c = 0.799 * k / 79
            n_inner = int(((sign * x >= c) & (y == 0)).sum())
            m = min(n_outer, n_inner)
            if m == 0:
                continue
            pi = m / n
            delta = 2 * math.log(2 ** (1 / pi) - 1) if 1 / pi < 1000 else 2 * (1 / pi) * math.log(2)
            best = min(best, (2 * a + delta / s) / (outer[0] + c))
        out.append(best)
    return max(out)


# ------------------------------------------------------------------------------------------ checks
def coverage_ok(level, iw, ib, nw, nb):
    """The leaves tile the grid exactly: areas sum to 1 (exact integer arithmetic), every leaf lies inside the grid,
    no two leaves coincide and no leaf is an ancestor of another."""
    L = int(level.max())
    area = sum(4 ** (L - int(l)) for l in level)
    if area != nw * nb * 4 ** L:
        return False, f"area {area} != {nw * nb * 4 ** L}"
    if ((iw < 0) | (ib < 0) | (iw >= nw * 2 ** level) | (ib >= nb * 2 ** level)).any():
        return False, "leaf outside the grid"
    keys = set(zip(level.tolist(), iw.tolist(), ib.tolist()))
    if len(keys) != len(level):
        return False, "duplicate leaf"
    for l, i, j in keys:
        for up in range(1, l + 1):
            if (l - up, i >> up, j >> up) in keys:
                return False, f"leaf ({l},{i},{j}) has an ancestor leaf"
    return True, "exact tiling"


_OBJ = {}


def _init_worker(npz_path, s, a):
    ctx.prec = PREC
    d = np.load(npz_path)
    _OBJ["obj"] = Objective(d["x"], d["y"], s, a)


def _leaf_ok(obj, wc, bc, hw, hb, kind, target, depth, stats):
    """kind 'lb': L* ≥ target over the cell; kind 'out+' / 'out-': the cell lies outside {G > 0} / {G ≤ 0}.
    Adaptive: a failed cell is split in four, up to MAX_DEPTH."""
    stats["evals"] += 1
    stats["max_depth"] = max(stats["max_depth"], depth)
    if kind == "lb":
        if obj.lower_bound(wc, bc, hw, hb) >= target:
            return True
    else:
        gl, gu = obj.gap_bounds(wc - hw, wc + hw, bc - hb, bc + hb)
        if (kind == "out+" and gu <= 0) or (kind == "out-" and gl > 0):
            return True
    if depth >= MAX_DEPTH:
        return False
    h2w, h2b = hw / 2, hb / 2
    return all(_leaf_ok(obj, wc + dw * h2w, bc + db * h2b, h2w, h2b, kind, target, depth + 1, stats)
               for dw in (-1, 1) for db in (-1, 1))


def _chunk(args):
    rows = args
    obj = _OBJ["obj"]
    out = []
    for (k, wc, bc, hw, hb, kind, target) in rows:
        st = {"evals": 0, "max_depth": 0}
        ok = _leaf_ok(obj, wc, bc, hw, hb, kind, target, 0, st)
        out.append((k, ok, st["evals"], st["max_depth"]))
    return out


def check_finite(name, verbose=True, workers=WORKERS):
    """See the module docstring.  Each leaf is verified against what it claims: reason 2 (discarded as outside the
    region) by a rigorous gap bound; reasons 0/1 (kept / discarded by its bound) by a rigorous lower bound of L* at
    least its recorded bound.  For the losing region the chain additionally needs every such bound > U."""
    from multiprocessing import Pool
    meta = json.loads((CERTS / f"{name}.json").read_text())
    dat = np.load(CERTS / f"{name}.npz")
    x, y = dat["x"], dat["y"]
    s, a = meta["s"], meta["a"]
    obj = Objective(x, y, s, a)
    res = {"name": name, "checks": {}}
    t0 = time.time()
    W = lemma_W(x, y, s, a)
    res["checks"]["lemma_W"] = abs(W - meta["W"]) <= 1e-12
    win, lose = meta["winning_region"], meta["losing_region"]
    if meta["U_point"] is None:
        U = float(hi_(arb(2).log()))
        res["checks"]["U_point_in_region"] = win == "-" and abs(float(obj.ybar.mid()) - 0.5) < 1e-15
    else:
        w, b1, b2 = meta["U_point"]
        U = obj.upper_at(w, b1, b2)
        gl, gu = obj.gap_bounds(w, w, b1, b1)
        res["checks"]["U_point_in_region"] = (gl > 0) if win == "+" else (gu <= 0)
    res["U"] = U
    res["checks"]["outside_W_loses"] = (U < math.log(2)) if win == "+" else True
    jobs = []
    for reg in ("-", "+"):
        lv, iw, ib = dat[f"level_{reg}"], dat[f"iw_{reg}"], dat[f"ib_{reg}"]
        ok, why = coverage_ok(lv, iw, ib, meta["nw"], meta["nb"])
        res["checks"][f"coverage_{reg}"] = ok
        hw = meta["hw0"] / 2.0 ** lv; hb = meta["hb0"] / 2.0 ** lv
        cw = -meta["W"] + (iw + 0.5) * 2 * hw; cb = (ib + 0.5) * 2 * hb
        reason, claim = dat[f"reason_{reg}"], dat[f"lb_{reg}"]
        if reg == lose:
            lbl = reason != 2
            res["checks"]["losing_claims_exceed_U"] = bool((claim[lbl] > U).all())
        for k in range(len(lv)):
            kind = f"out{reg}" if reason[k] == 2 else "lb"
            jobs.append(((reg, k), float(cw[k]), float(cb[k]), float(hw[k]), float(hb[k]), kind, float(claim[k])))
    chunks = [jobs[i:i + 200] for i in range(0, len(jobs), 200)]
    with Pool(workers, initializer=_init_worker, initargs=(str(CERTS / f"{name}.npz"), s, a)) as p:
        results = [r for part in p.imap_unordered(_chunk, chunks) for r in part]
    fails = [r for r in results if not r[1]]
    res["checks"]["every_leaf_claim_verified"] = not fails
    res["leaves"] = len(results)
    res["failed_leaves"] = [(r[0][0], int(r[0][1])) for r in fails[:20]]
    res["evaluations"] = int(sum(r[2] for r in results))
    res["max_subdivision_depth"] = int(max(r[3] for r in results))
    res["leaves_needing_subdivision"] = int(sum(r[2] > 1 for r in results))
    res["seconds"] = time.time() - t0
    res["pass"] = all(res["checks"].values())
    if verbose:
        print(json.dumps(res, indent=1))
    return res


def main(names):
    out = [check_finite(n) for n in names]
    bad = [r["name"] for r in out if not r["pass"]]
    print(f"{len(out)} certificate(s) checked; failures: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in CERTS.glob("*.json"))
    raise SystemExit(main(names))
