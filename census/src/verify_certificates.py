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

Rigour notes (every step below is an enclosure in Arb ball arithmetic; nothing is decided in floats):
  * Comparisons are made between Arb balls (`a > b` is True only if it holds for every point of both balls), or
    between an Arb ball and the exact binary value of a float (arb(float) is exact).  No Arb endpoint is converted
    to a float before a comparison.
  * The bias bracket.  For fixed (w₁, b₁), F(b) = ∂L/∂b = mean σ(z0 + b) − ȳ is strictly increasing in b.  On a cell,
    z0ᵢ ∈ [zloᵢ, zhiᵢ] pointwise, so mean σ(zlo + b) − ȳ ≤ F(b) ≤ mean σ(zhi + b) − ȳ.  If the upper function is < 0 at
    β (checked in Arb) then F(β) < 0 for every point of the cell, so b* > β; likewise b* < γ.  Hence b* ∈ [β, γ].
  * Lower bound of L* at the cell centre c (the boundary-leaf argument).  g(b) = L(c, b) = mean softplus(z0ᵢ + b)
    − yᵢ(z0ᵢ + b) is convex in b (softplus is convex; the other term is linear), with g′(b) = F(b).  For convex g
    and any b̂, g(b*) ≥ g(b̂) + g′(b̂)(b* − b̂) (the tangent line lies below a convex function).  So
        L*(c) = g(b*) ≥ g(b̂) − |F(b̂)|·|b* − b̂|,   |b* − b̂| ≤ width of the certified bracket [β, γ] ∋ b*, b̂ ∈ [β, γ].
    g(b̂) and F(b̂) are evaluated in Arb at the exact point b̂ (a float), which avoids the dependency loss of
    evaluating g over the whole bracket (softplus(z) and y·z do not cancel in interval arithmetic).
  * Mean-value step over the cell: for every θ in the cell, L*(θ) ≥ L*(c) − sup|∂_w L*|·h_w − sup|∂_b L*|·h_b, with
    the envelope gradient ∂L*/∂θ = mean((σ(z) − y)·s·f_a′(t)·(x, 1)) enclosed over the cell × the bias bracket.
  * The class gap uses the exact range of f_a over an interval (endpoints and the critical points t = ±acos(−1/a)
    + 2πk inside it, all evaluated in Arb).

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
WORKERS = int(__import__('os').environ.get('VC_WORKERS', '3'))


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


def range_inner(fa, tlo, thi):
    """(an upper bound of min, a lower bound of max) of f_a over the exact interval [tlo, thi] (tlo, thi arb balls
    enclosing the exact endpoints): only values ATTAINED in the interval are used -- the endpoints, and the critical
    points whose ball lies strictly inside -- so min over them >= the true min and max over them <= the true max."""
    cand = [fa.f(tlo), fa.f(thi)]
    if fa.c is not None:
        for sgn in (1, -1):
            cc = sgn * fa.c
            k0 = int(math.floor((float(tlo.mid()) - float(cc.mid())) / (2 * math.pi))) - 1
            k1 = int(math.ceil((float(thi.mid()) - float(cc.mid())) / (2 * math.pi))) + 1
            for k in range(k0, k1 + 1):
                tk = cc + k * fa.twopi
                if lo_(tk) > hi_(tlo) and hi_(tk) < lo_(thi):
                    cand.append(fa.f(tk))
    return min(hi_(c) for c in cand), max(lo_(c) for c in cand)


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
            return None
        # L*(c) = L(c, b*) with b* ∈ Bpt.  L(c, ·) is convex, so L(c, b*) ≥ L(c, b̂) + F(b̂)(b* − b̂) with
        # F = ∂L/∂b = mean σ(z) − ȳ: evaluate at the single point b̂ (no dependency loss over the bracket).
        bhat = arb(float(Bpt.mid()))
        Lpt, _, _ = self.value_and_grad(Wc, Bc, bhat)
        Fh = self.F_at(Wc, Bc, bhat)
        Lc = arb(lo_(Lpt)) - abs(Fh) * (2 * arb(Bpt.rad()) + abs(bhat - Bpt))
        W, B = ball(wc - hw, wc + hw), ball(bc - hb, bc + hb)
        Bcell, Z = self.b_bracket(W, B)
        if Bcell is None:
            return None
        _, gw, gb = self.value_and_grad(W, B, Bcell, Z)
        return Lc - abs(gw) * arb(hw) - abs(gb) * arb(hb)            # a ball; its lower end is the bound

    def F_at(self, W, B, b):
        """F(b) = mean σ(z0 + b) − ȳ at a (w, b₁) ball."""
        acc = arb(0)
        for xi in self.X:
            z = self.s * self.fa.f(W * xi + B) + b
            acc += 1 / (1 + (-z).exp())
        return acc / self.n - self.ybar

    def upper_at(self, w, b1, b2):
        """An Arb ball containing L(w, b₁, b₂) ≥ L*(w, b₁); its upper end is a rigorous upper bound of L*."""
        L, _, _ = self.value_and_grad(arb(w), arb(b1), arb(b2))
        return L

    def gap_bounds(self, wl, wh, bl, bh, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
        """Rigorous (G_lower, G_upper) over the cell, G = min_O φ − max_I φ, φ = f_a(w x + b₁)."""
        wl, wh, bl, bh = arb(wl), arb(wh), arb(bl), arb(bh)
        fa = self.fa
        ti = t_interval(wl, wh, bl, bh, arb(inner[0]), arb(inner[1]))
        to1 = t_interval(wl, wh, bl, bh, arb(outer[0]), arb(outer[1]))
        to2 = t_interval(wl, wh, bl, bh, arb(-outer[1]), arb(-outer[0]))
        imn, imx = fa.range(*ti)
        o1mn, _ = fa.range(*to1); o2mn, _ = fa.range(*to2)
        G_lower = min(o1mn, o2mn) - imx                            # Arb (an exact lower endpoint)
        # upper: G(θ) ≤ φ(x_o, θ) − φ(x_i, θ) for any fixed x_o ∈ O, x_i ∈ I; take the best of a few pairs
        wm, bm = float(((wl + wh) / 2).mid()), float(((bl + bh) / 2).mid())
        xs_i = np.linspace(inner[0], inner[1], 81); xs_o = np.r_[np.linspace(*outer, 41), -np.linspace(*outer, 41)]
        fi = xs_i * wm + bm; fo = xs_o * wm + bm
        fi = fi + fa.af * np.sin(fi); fo = fo + fa.af * np.sin(fo)
        xo, xi = float(xs_o[np.argmin(fo)]), float(xs_i[np.argmax(fi)])
        _, fomax = fa.range(*t_interval(wl, wh, bl, bh, arb(xo), arb(xo)))
        fimin, _ = fa.range(*t_interval(wl, wh, bl, bh, arb(xi), arb(xi)))
        G_upper = fomax - fimin
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


def files_match_manifest(name):
    """Both data files hash to the SHA-256 committed in results/certificates_manifest.csv."""
    import csv
    import hashlib
    man = CERTS.parent / "certificates_manifest.csv"
    if not man.exists():
        return False
    want = {r["file"]: r["sha256"] for r in csv.DictReader(man.open())}
    for ext in ("json", "npz"):
        key = f"results/certificates/{name}.{ext}"
        if want.get(key) != hashlib.sha256((CERTS / f"{name}.{ext}").read_bytes()).hexdigest():
            return False
    return True


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
        lb = obj.lower_bound(wc, bc, hw, hb)
        if lb is not None and lb >= arb(target):                    # certain: every point of the ball ≥ target
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
    res = {"name": name, "checks": {"files_match_committed_hashes": files_match_manifest(name)}}
    t0 = time.time()
    W = lemma_W(x, y, s, a)
    res["checks"]["lemma_W"] = abs(W - meta["W"]) <= 1e-12
    win, lose = meta["winning_region"], meta["losing_region"]
    if meta["U_point"] is None:
        U = arb(2).log()                                   # the constant predictor with ȳ = ½ has loss exactly log 2
        res["checks"]["U_point_in_region"] = win == "-" and (obj.ybar == arb(1) / 2)
    else:
        w, b1, b2 = meta["U_point"]
        U = obj.upper_at(w, b1, b2)
        gl, gu = obj.gap_bounds(w, w, b1, b1)
        res["checks"]["U_point_in_region"] = bool((gl > 0) if win == "+" else (gu <= 0))
    res["U_upper"] = float(hi_(U).mid()) if hasattr(hi_(U), "mid") else str(hi_(U))
    res["checks"]["outside_W_loses"] = bool(U < arb(2).log()) if win == "+" else True
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
            res["checks"]["losing_claims_exceed_U"] = all(arb(float(c)) > U for c in claim[lbl])
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


# ------------------------------------------------------------------------------------------ Ĝ enclosure certificates
def _oriented_gap_upper(fa, w, b, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Rigorous upper bound of G(w, b) = max(min_O φ − max_I φ, min_I φ − max_O φ) at the exact point enclosed by the
    balls w, b (attained values only; see range_inner)."""
    ti = (w * arb(inner[0]) + b, w * arb(inner[1]) + b)
    o1 = (w * arb(outer[0]) + b, w * arb(outer[1]) + b)
    o2 = (w * arb(-outer[1]) + b, w * arb(-outer[0]) + b)
    imn_u, imx_l = range_inner(fa, *ti)
    a_u, a_l = range_inner(fa, *o1)
    b_u, b_l = range_inner(fa, *o2)
    # every operand is an exact point (a ball endpoint), so min/max are exact; each difference is a ball whose
    # UPPER endpoint is taken, and the maximum of two exact points is exact
    return max(hi_(min(a_u, b_u) - imx_l), hi_(imn_u - max(a_l, b_l)))


def _oriented_gap_lower(fa, w, b, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Rigorous lower bound of G at the exact point (every possibly-contained critical point: FA.range)."""
    w, b = arb(w), arb(b)
    imn, imx = fa.range(w * arb(inner[0]) + b, w * arb(inner[1]) + b)
    amn, amx = fa.range(w * arb(outer[0]) + b, w * arb(outer[1]) + b)
    bmn, bmx = fa.range(w * arb(-outer[1]) + b, w * arb(-outer[0]) + b)
    return max(lo_(min(amn, bmn) - imx), lo_(imn - max(amx, bmx)))           # exact points throughout (as above)


_G = {}


def _ghat_init(a, nw, nb, hi):
    ctx.prec = PREC
    fa = FA(a)
    A = arb(a)
    _G.update(fa=fa, a=A, cw0=A / nw, cb0=2 * arb.pi() / nb, hi=arb(hi))


def _ghat_chunk(args):
    lv, iw, ib = args
    fa, A, cw0, cb0, hi = _G["fa"], _G["a"], _G["cw0"], _G["cb0"], _G["hi"]
    bad, worst = [], None
    for l, i, j in zip(lv.tolist(), iw.tolist(), ib.tolist()):
        dw, db = cw0 / 2 ** l, cb0 / 2 ** l                      # cell widths (exact reals, as balls)
        wc, bc = (i + arb(1) / 2) * dw, (j + arb(1) / 2) * db
        ub = hi_(_oriented_gap_upper(fa, wc, bc) + (1 + A) * (arb("2.8") * dw / 2 + 2 * db / 2))
        if not (ub <= hi):
            bad.append((l, i, j))
        worst = ub if worst is None else max(worst, ub)                  # exact points
    worst = None if worst is None else float(worst.mid())
    return bad, worst


def check_ghat(name, verbose=True, workers=WORKERS):
    """Ĝ(a) enclosure.  Checks: (1) files match the committed hashes; (2) the leaves tile [0, a] x [0, 2π] exactly;
    (3) on every leaf, sup G <= claim_hi: a rigorous upper bound of G at the leaf's exact centre plus the Lipschitz
    step (1 + a)(2.8 hw + 2 hb) (|f_a′| <= 1 + a; derivation in src/ghat_bnb.py), all in Arb; (4) the search's
    attained lower end: G(bnb_arg) >= bnb_lo; (5) the paper's Ĝ_cert is attained: G(witness) >= Ĝ_cert, rigorously;
    (6) Ĝ_cert <= claim_hi.  The reduction of the domain to w₁ ∈ (0, a/1.4], b₁ ∈ [0, 2π) is analytic (symmetries and
    f_a(u) − f_a(t) >= (u − t) − 2a); a/1.4 <= a is checked."""
    from multiprocessing import Pool
    ctx.prec = PREC
    meta = json.loads((CERTS / f"{name}.json").read_text())
    dat = np.load(CERTS / f"{name}.npz")
    a = meta["a"]
    fa = FA(a)
    res = {"name": name, "checks": {"files_match_committed_hashes": files_match_manifest(name)}}
    t0 = time.time()
    lv, iw, ib = dat["level"].astype(np.int64), dat["iw"], dat["ib"]
    ok, why = coverage_ok(lv, iw, ib, meta["nw"], meta["nb"])
    res["checks"]["coverage"] = ok
    res["checks"]["domain_contains_reduction"] = a / 1.4 <= a
    n = len(lv)
    parts = [(lv[i:i + 2000], iw[i:i + 2000], ib[i:i + 2000]) for i in range(0, n, 2000)]
    with Pool(workers, initializer=_ghat_init, initargs=(a, meta["nw"], meta["nb"], meta["claim_hi"])) as p:
        out = p.map(_ghat_chunk, parts)
    bad = [b for o in out for b in o[0]]
    res["checks"]["every_leaf_below_claim_hi"] = not bad
    res["worst_leaf_upper"] = max(o[1] for o in out)
    g_arg = _oriented_gap_lower(fa, *meta["bnb_arg"])
    g_wit = _oriented_gap_lower(fa, *meta["ghat_cert_witness"])
    res["checks"]["bnb_lo_attained"] = bool(g_arg >= arb(meta["bnb_lo"]))
    res["checks"]["ghat_cert_attained"] = bool(g_wit >= arb(meta["ghat_cert"]))
    res["checks"]["ghat_cert_below_hi"] = bool(arb(meta["ghat_cert"]) <= arb(meta["claim_hi"]))
    res["G_at_bnb_arg_lower"] = str(lo_(g_arg)); res["G_at_witness_lower"] = str(lo_(g_wit))
    # the rigorous enclosure this certificate proves, and how far each published float endpoint is from it
    # (reported, never used to pass a check)
    L_rig = max(lo_(g_arg), lo_(g_wit))
    res["rigorous_lower"] = float(L_rig.mid()); res["rigorous_upper"] = res["worst_leaf_upper"]
    res["claim_hi_excess_over_published"] = res["worst_leaf_upper"] - meta["claim_hi"]
    res["ghat_cert_deficit"] = float((arb(meta["ghat_cert"]) - lo_(g_wit)).mid())
    res["bnb_lo_deficit"] = float((arb(meta["bnb_lo"]) - lo_(g_arg)).mid())
    res["ghat_cert"] = meta["ghat_cert"]; res["claim_hi"] = meta["claim_hi"]
    res["leaves"] = n; res["failed_leaves"] = bad[:20]
    res["seconds"] = time.time() - t0
    res["pass"] = all(res["checks"].values())
    if verbose:
        print(json.dumps(res, indent=1))
    return res


def check(name):
    kind = json.loads((CERTS / f"{name}.json").read_text()).get("kind")
    return check_ghat(name) if kind == "ghat_enclosure" else check_finite(name)


def main(names):
    out = [check(n) for n in names]
    bad = [r["name"] for r in out if not r["pass"]]
    print(f"{len(out)} certificate(s) checked; failures: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in CERTS.glob("*.json"))
    raise SystemExit(main(names))
