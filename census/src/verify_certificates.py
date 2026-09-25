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


def float_down(x):
    """The largest double <= the exact lower endpoint of the ball x (checked by exact Arb comparison)."""
    e = lo_(x)
    f = float(e.mid())
    while arb(f) > e:
        f = math.nextafter(f, -math.inf)
    return f


def float_up(x):
    """The smallest double >= the exact upper endpoint of the ball x."""
    e = hi_(x)
    f = float(e.mid())
    while arb(f) < e:
        f = math.nextafter(f, math.inf)
    return f


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

    def np_f(self, t):
        return t + self.af * np.sin(t)

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
        if not all(c.is_finite() for c in cand):
            return arb("-inf"), arb("inf")                   # conservative (see HAct)
        mn = min((lo_(c) for c in cand))
        mx = max((hi_(c) for c in cand))
        return mn, mx


class HAct:
    """The limit-problem activation h(σ) = −σ + σ³/6 (critical points ±√2), with the interface of FA."""

    def __init__(self):
        self.c = arb(2).sqrt()                            # h′(σ) = σ²/2 − 1 = 0 at ±√2
        self.twopi = None

    # NOTE: python-flint's arb ** int returns nan for a ball containing 0; products are used instead (bug found
    # 2026-09-25, before which a nan could be skipped by min/max in range()).
    def f(self, t):
        return -t + t * t * t / 6

    def df(self, t):
        return t * t / 2 - 1

    def np_f(self, t):
        return -t + t ** 3 / 6

    def range(self, tlo, thi):
        """Rigorous [min, max] over [tlo, thi]: endpoints and ±√2 if possibly inside."""
        cand = [self.f(tlo), self.f(thi)]
        for cc in (self.c, -self.c):
            if not (hi_(cc) < lo_(tlo) or lo_(cc) > hi_(thi)):
                cand.append(self.f(cc))
        if not all(c.is_finite() for c in cand):
            return arb("-inf"), arb("inf")                   # conservative: never let a nan be skipped by min/max
        return min(lo_(c) for c in cand), max(hi_(c) for c in cand)


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
    if not all(c.is_finite() for c in cand):
        return arb("inf"), arb("-inf")                       # conservative: no attained value is claimed
    return min(hi_(c) for c in cand), max(lo_(c) for c in cand)


def t_interval(wl, wh, bl, bh, xl, xh):
    """Range of w·x + b over w ∈ [wl, wh], x ∈ [xl, xh], b ∈ [bl, bh] (arb endpoints, exact products)."""
    prods = [wl * xl, wl * xh, wh * xl, wh * xh]
    lo = min(lo_(p) for p in prods) + lo_(bl)
    hi = max(hi_(p) for p in prods) + hi_(bh)
    return lo, hi


# ------------------------------------------------------------------------------------------ the finite-a objective
class Objective:
    def __init__(self, x, y, s, a, act=None):
        """a: the f_a parameter (act None), or act = HAct() for the limit problem (s is then A)."""
        ctx.prec = PREC
        self.xf = np.asarray(x, float); self.yf = np.asarray(y, float)
        self.X = [arb(float(v)) for v in self.xf]
        self.Y = [int(v) for v in self.yf]
        self.n = len(self.X)
        self.s = arb(s); self.fa = act if act is not None else FA(a)
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

        yb = float(self.ybar.mid())
        L0 = math.log(yb / (1 - yb))
        k_y = sum(self.Y)

        def sign_F(z, m):
            """Sign of F(m) = mean σ(z + m) − ȳ in floats, robust to saturation: count the positive units exactly; when
            the count balances (a plateau where every tail underflows), compare the tails in the log domain."""
            u = z + m
            pos = u > 0
            d = int(pos.sum()) - k_y
            if d != 0:
                t = np.exp(-np.logaddexp(0, -u[~pos])).sum() - np.exp(-np.logaddexp(0, u[pos])).sum()   # Σσ(u) − Σσ(−u)
                v = d + t
                if v != 0:
                    return np.sign(v)
            lneg = np.logaddexp.reduce(-np.logaddexp(0, -u[~pos])) if (~pos).any() else -np.inf      # log Σ σ(u)
            lpos = np.logaddexp.reduce(-np.logaddexp(0, u[pos])) if pos.any() else -np.inf           # log Σ σ(−u)
            return np.sign(lneg - lpos) if d == 0 else np.sign(d)

        def root(z):
            # F is increasing in b; its root lies in [logit(ȳ) − max z, logit(ȳ) − min z] (monotonicity), which can be far
            # beyond ±80 when z is large (the limit problem's cubic logit).  Float bisection only PROPOSES the bracket;
            # the Arb sign checks below decide it.
            lo, hi = L0 - float(z.max()) - 1.0, L0 - float(z.min()) + 1.0
            for _ in range(200):
                m = 0.5 * (lo + hi)
                lo, hi = (m, hi) if sign_F(z, m) < 0 else (lo, m)
            return 0.5 * (lo + hi)
        r_hi, r_lo = root(zh), root(zl)
        beta = r_hi - 1e-12 * max(1.0, abs(r_hi))
        gamma = r_lo + 1e-12 * max(1.0, abs(r_lo))

        def F(Zb, b):
            return self._F_tails(Zb, b)
        Zhi = [arb(hi_(z)) for z in Z]; Zlo = [arb(lo_(z)) for z in Z]
        ok = (F(Zhi, arb(beta)) < 0) and (F(Zlo, arb(gamma)) > 0)
        return (ball(beta, gamma) if ok else None), Z

    def _F_tails(self, Zb, b):
        """F(b) = mean σ(z + b) − ȳ, computed so that it stays resolvable when every σ is saturated (|z| ~ 1e5 in the
        limit problem): for u = z + b certainly > 0, σ(u) = 1 − σ(−u), and the whole units are counted exactly as an
        integer; only the (tiny) tails are summed in Arb."""
        k_pos = 0
        tails = arb(0)
        for z in Zb:
            u = z + b
            if u > 0:                                        # certain (Arb): use σ(u) = 1 − σ(−u)
                k_pos += 1
                tails -= 1 / (1 + u.exp())                   # σ(−u) = 1 / (1 + e^u)
            else:
                tails += 1 / (1 + (-u).exp())
        k_y = sum(self.Y)                                    # n·ȳ, exact
        return (arb(k_pos - k_y) + tails) / self.n

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
        mv = Lc - abs(gw) * arb(hw) - abs(gb) * arb(hb)                # mean-value bound (a ball; lower end)
        # Direct (zeroth-order) bound: on the cell, b* ∈ Bcell and each point's z0 ∈ Z[i], so z = z0 + b* ∈
        # [lo Z[i] + lo Bcell, hi Z[i] + hi Bcell].  The per-point loss is monotone in z (softplus(−z) for y = 1,
        # softplus(z) for y = 0), so it is at least its value at the favourable endpoint.  Decisive where the loss is
        # large (far out in the box), where the mean-value bound is swamped by the gradient.
        acc = arb(0)
        bl, bh = lo_(Bcell), hi_(Bcell)
        for zi, yi in zip(Z, self.Y):
            e = -(hi_(zi) + bh) if yi else (lo_(zi) + bl)
            acc += (arb(lo_(e)).exp() + 1).log()                          # lower bound: softplus is increasing
        direct = arb(lo_(acc / self.n))
        return mv if lo_(mv) >= lo_(direct) else direct

    def F_at(self, W, B, b):
        """F(b) = mean σ(z0 + b) − ȳ at a (w, b₁) ball (tail form, see _F_tails)."""
        return self._F_tails([self.s * self.fa.f(W * xi + B) for xi in self.X], b)

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
        fi = fa.np_f(xs_i * wm + bm); fo = fa.np_f(xs_o * wm + bm)       # float heuristic: only picks the points
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
    no two leaves coincide and no leaf is an ancestor of another.  Vectorised (each leaf is encoded as one exact
    integer key; memory is linear in the leaf count)."""
    level = np.asarray(level, np.int64); iw = np.asarray(iw, np.int64); ib = np.asarray(ib, np.int64)
    if (level < 0).any():
        return False, "negative level"
    L = int(level.max())
    area = sum(int(c) * 4 ** (L - l) for l, c in enumerate(np.bincount(level)))
    if area != nw * nb * 4 ** L:
        return False, f"area {area} != {nw * nb * 4 ** L}"
    if ((iw < 0) | (ib < 0) | (iw >= nw * 2 ** level) | (ib >= nb * 2 ** level)).any():
        return False, "leaf outside the grid"
    bi = max(1, int(nw * 2 ** L - 1).bit_length()); bj = max(1, int(nb * 2 ** L - 1).bit_length())
    bl = max(1, L.bit_length())
    if bi + bj + bl > 63:
        raise ValueError("quadtree too deep for exact 63-bit keys")

    def key(l, i, j):
        return (l << (bi + bj)) | (i << bj) | j
    keys = key(level, iw, ib)
    if len(np.unique(keys)) != len(keys):
        return False, "duplicate leaf"
    skeys = np.sort(keys)
    for up in range(1, L + 1):
        m = level >= up
        if not m.any():
            continue
        par = key(level[m] - up, iw[m] >> up, ib[m] >> up)
        pos = np.searchsorted(skeys, par)
        hit = (pos < len(skeys)) & (skeys[np.minimum(pos, len(skeys) - 1)] == par)
        if hit.any():
            k = int(np.flatnonzero(hit)[0])
            return False, f"leaf ({int(level[m][k])},{int(iw[m][k])},{int(ib[m][k])}) has an ancestor leaf"
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
        h = hashlib.sha256()
        with (CERTS / f"{name}.{ext}").open("rb") as fh:                 # streamed (large certificates)
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        if want.get(key) != h.hexdigest():
            return False
    return True


_OBJ = {}


def _init_worker(npz_path, s, a, act_name=None):
    ctx.prec = PREC
    d = np.load(npz_path)
    _OBJ["obj"] = Objective(d["x"], d["y"], s, a, HAct() if act_name == "h" else None)
    _OBJ["max_depth"] = MAX_DEPTH_LIMIT if act_name == "h" else None


MAX_DEPTH_LIMIT = 10          # the limit check's near-margin leaves (margin ~1e-7) need finer cells


def _leaf_ok(obj, wc, bc, hw, hb, kind, target, depth, stats, max_depth=None):
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
    if depth >= (MAX_DEPTH if max_depth is None else max_depth):
        return False
    h2w, h2b = hw / 2, hb / 2
    return all(_leaf_ok(obj, wc + dw * h2w, bc + db * h2b, h2w, h2b, kind, target, depth + 1, stats, max_depth)
               for dw in (-1, 1) for db in (-1, 1))


def _chunk(args):
    rows = args
    obj = _OBJ["obj"]
    md = _OBJ.get("max_depth")
    out = []
    for (k, wc, bc, hw, hb, kind, target) in rows:
        st = {"evals": 0, "max_depth": 0}
        ok = _leaf_ok(obj, wc, bc, hw, hb, kind, target, 0, st, md)
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


# ------------------------------------------------------------------------------------------ limit-problem status certificates
def pava_logloss(ys):
    """Rigorous total log loss of the isotonic (non-decreasing) regression of the 0/1 sequence ys: PAVA with exact
    integer block sums and counts, then −Σ k·log(mean) in Arb."""
    blocks = []                                           # [ones, count], exact integers
    for v in ys:
        blocks.append([int(v), 1])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            s2, c2 = blocks.pop()
            blocks[-1][0] += s2; blocks[-1][1] += c2
    tot = arb(0)
    for k1, c in blocks:
        k0 = c - k1
        if k1:
            tot -= k1 * (arb(k1) / c).log()
        if k0:
            tot -= k0 * (arb(k0) / c).log()
    return tot


def localisation_bounds(x, y, P):
    """Rigorous lower bounds (Arb) of L0* for |p| >= P (B(P)) and for |q| > 2√2 + X|p| (B_full), math in
    src/limit_bnb.py: points with |σ| < 2√2 lie in an open x-window of length 4√2/|p| <= w = 4√2/P; outside it the
    logit is monotone in x, so L0* >= (1/n)·(best monotone log loss of the rest).  Windows starting at a data point
    dominate every window (they remove a superset), and removing points never increases the bound, so the minimum over
    windows [x_i, x_i + w) is a valid bound; a point whose membership is undecided in Arb is removed (conservative)."""
    n = len(x)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    w = 4 * arb(2).sqrt() / arb(P)
    X = [arb(float(v)) for v in xs]
    best = None
    for i in range(n):
        c = X[i]
        keep = [j for j in range(n) if (X[j] < c) or (X[j] >= c + w)]       # certainly outside [c, c + w)
        yk = ys[keep]
        v = min_ball(pava_logloss(yk), pava_logloss(yk[::-1])) / n
        best = v if best is None else min_ball(best, v)
    full = min_ball(pava_logloss(ys), pava_logloss(ys[::-1])) / n
    return best, full


def min_ball(a, b):
    """A ball whose LOWER end is min(lower(a), lower(b)) -- a rigorous lower bound of min(a, b)."""
    return arb(min(lo_(a), lo_(b)))


def check_limit(name, verbose=True, workers=WORKERS):
    """Limit-problem status certificate at scale A.  Checks: hashes; the data are exactly x-symmetric with the labels
    (the half-domain p >= 0) and the windows are symmetric; max|x| <= X; h is increasing on |σ| >= 2√2 with
    h(−2√2) < h(2√2) (the monotone-logit step of the localisation); the localisation bounds B(P), B_full recomputed
    rigorously and the winner's U below both (so the global minimiser lies in the certified box); exact tiling of each
    region; U's point in the winning region (or the constant predictor, exactly log 2); every LOSING-region leaf outside
    its region or with a rigorous lower bound of L0* above U, i.e. exactly what the status needs (author's decision
    2026-09-25). Winning-region leaves are tiled (coverage) but their own search bounds are not re-verified."""
    from multiprocessing import Pool
    ctx.prec = PREC
    meta = json.loads((CERTS / f"{name}.json").read_text())
    dat = np.load(CERTS / f"{name}.npz")
    x, y = dat["x"], dat["y"]
    A, P = meta["A"], meta["P"]
    hact = HAct()
    obj = Objective(x, y, A, None, hact)
    res = {"name": name, "checks": {"files_match_committed_hashes": files_match_manifest(name)}}
    t0 = time.time()
    pos = sorted(zip(x.tolist(), y.tolist())); neg = sorted(zip((-x).tolist(), y.tolist()))
    res["checks"]["data_x_symmetric"] = pos == neg
    inn, out = meta["inner"], meta["outer"]
    res["checks"]["windows_symmetric"] = inn[0] == -inn[1]
    res["checks"]["max_abs_x_le_X"] = float(np.abs(x).max()) <= meta["X"]
    r2 = 2 * arb(2).sqrt()
    res["checks"]["h_monotone_outside_2sqrt2"] = bool(hact.f(-r2) < hact.f(r2)) and bool(hact.df(r2) > 0)
    res["checks"]["Q_matches"] = abs(meta["Q"] - (2 * math.sqrt(2) + meta["X"] * P)) <= 1e-12
    BP, Bfull = localisation_bounds(x, y, P)
    res["B_P_lower"] = float(lo_(BP).mid()); res["B_full_lower"] = float(lo_(Bfull).mid())
    win, lose = meta["winning_region"], meta["losing_region"]
    if meta["U_point"] is None:
        U = arb(2).log()
        res["checks"]["U_point_in_region"] = win == "-" and (obj.ybar == arb(1) / 2)
    else:
        p_, q_, b2_ = meta["U_point"]
        U = obj.upper_at(p_, q_, b2_)
        gl, gu = obj.gap_bounds(p_, p_, q_, q_, tuple(inn), tuple(out))
        res["checks"]["U_point_in_region"] = bool((gl > 0) if win == "+" else (gu <= 0))
    res["U_upper"] = float(hi_(U).mid())
    res["checks"]["U_below_localisation_bounds"] = bool(U < BP) and bool(U < Bfull)
    # What the status needs (author's decision 2026-09-25): every LOSING-region leaf is outside its region or has a
    # rigorous lower bound of L0* above U.  The target is the smallest double above U's upper end, so a pass means
    # L0* > U on the leaf.  Winning-region leaves are tiled (coverage) but their own search bounds are not re-verified:
    # they do not enter the status claim.
    target = float(np.nextafter(float(hi_(U).mid()), np.inf))
    while not (arb(target) > U):
        target = float(np.nextafter(target, np.inf))
    jobs = []
    for reg in ("-", "+"):
        lv, iw, ib = dat[f"level_{reg}"], dat[f"iw_{reg}"], dat[f"ib_{reg}"]
        ok, why = coverage_ok(lv, iw, ib, meta["nw"], meta["nb"])
        res["checks"][f"coverage_{reg}"] = ok
        if reg != lose:
            res["winning_region_leaves_not_reverified"] = int(len(lv))
            continue
        hw = meta["hw0"] / 2.0 ** lv; hb = meta["hb0"] / 2.0 ** lv
        cw = meta["p0"] + (iw + 0.5) * 2 * hw; cb = -meta["Q"] + (ib + 0.5) * 2 * hb
        reason = dat[f"reason_{reg}"]
        for k in range(len(lv)):
            kind = f"out{reg}" if reason[k] == 2 else "lb"
            jobs.append(((reg, k), float(cw[k]), float(cb[k]), float(hw[k]), float(hb[k]), kind, target))
    chunks = [jobs[i:i + 200] for i in range(0, len(jobs), 200)]
    with Pool(workers, initializer=_init_worker, initargs=(str(CERTS / f"{name}.npz"), A, None, "h")) as p:
        results = [r for part in p.imap_unordered(_chunk, chunks) for r in part]
    fails = [r for r in results if not r[1]]
    res["checks"]["every_losing_leaf_outside_or_above_U"] = not fails
    res["leaves"] = len(results)
    res["failed_leaves"] = [(r[0][0], int(r[0][1])) for r in fails[:20]]
    res["evaluations"] = int(sum(r[2] for r in results))
    res["max_subdivision_depth"] = int(max(r[3] for r in results))
    res["seconds"] = time.time() - t0
    res["pass"] = all(res["checks"].values())
    out_dir = CERTS.parent / "certificate_checks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
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
    worst = None if worst is None else float_up(worst)                    # rounded upward: still an upper bound
    return bad, worst, len(lv)


def coverage_per_round(zf_npz, nw, nb, rounds):
    """Exact tiling for leaves stored per level (streamed certificates), with memory linear in one level: the open set
    at level 0 is the whole nw x nb grid; at each level the leaves must be distinct members of the open set, and the
    open set at the next level is the four children of every open cell that is not a leaf; after the last level no
    open cell may remain.  Every point is then covered by exactly one leaf."""
    names = set(zf_npz.files)
    bj = max(1, int(nb * 2 ** rounds - 1).bit_length())
    op = (np.arange(nw, dtype=np.int64)[:, None] << bj | np.arange(nb, dtype=np.int64)[None, :]).ravel()
    for r in range(rounds):
        parts = [zf_npz[k].astype(np.int64) for k in (f"r{r:02d}_pruned", f"r{r:02d}_kept") if k in names]
        leaf = np.concatenate(parts) if parts else np.empty((0, 2), np.int64)
        lk = np.sort((leaf[:, 0] << bj) | leaf[:, 1])
        if len(np.unique(lk)) != len(lk):
            return False, f"duplicate leaf at level {r}"
        op = np.sort(op)
        pos = np.searchsorted(op, lk)
        if not ((pos < len(op)).all() and (op[np.minimum(pos, len(op) - 1)] == lk).all()):
            return False, f"a level-{r} leaf is not an open cell (outside the grid or under an ancestor leaf)"
        hit = np.zeros(len(op), bool)                   # lk ⊂ op is established above: mark, then keep the rest
        hit[pos] = True
        del lk, leaf, pos
        rest = op[~hit]
        del op, hit
        if r == rounds - 1:
            return (len(rest) == 0), ("exact tiling" if len(rest) == 0 else f"{len(rest)} cells uncovered")
        op = np.empty(4 * len(rest), np.int64)          # the children, written in place (no temporaries per child)
        i, j = rest >> bj, rest & ((1 << bj) - 1)
        del rest
        for k, (di, dj) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            op[k::4] = ((2 * i + di) << bj) | (2 * j + dj)
        del i, j
    return False, "no levels"


def _ghat_leaf_iter(dat, meta):
    """(level, iw, ib) chunks of at most 2,000 leaves, from either storage format."""
    if meta.get("format") == "per_round":
        for k in sorted(dat.files):
            arr = dat[k]
            lv = int(k[1:3])
            for s in range(0, len(arr), 2000):
                c = arr[s:s + 2000].astype(np.int64)
                yield np.full(len(c), lv, np.int64), c[:, 0], c[:, 1]
    else:
        lv, iw, ib = dat["level"].astype(np.int64), dat["iw"].astype(np.int64), dat["ib"].astype(np.int64)
        for s in range(0, len(lv), 2000):
            yield lv[s:s + 2000], iw[s:s + 2000], ib[s:s + 2000]


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
    if meta.get("format") == "per_round":
        ok, why = coverage_per_round(dat, meta["nw"], meta["nb"], len(meta["rounds"]))
    else:
        lv, iw, ib = dat["level"].astype(np.int64), dat["iw"].astype(np.int64), dat["ib"].astype(np.int64)
        ok, why = coverage_ok(lv, iw, ib, meta["nw"], meta["nb"])
        del lv, iw, ib
    res["checks"]["coverage"] = ok
    res["checks"]["domain_contains_reduction"] = a / 1.4 <= a
    n = 0
    out = []
    with Pool(workers, initializer=_ghat_init, initargs=(a, meta["nw"], meta["nb"], meta["claim_hi"])) as p:
        for o in p.imap_unordered(_ghat_chunk, _ghat_leaf_iter(dat, meta), chunksize=8):
            out.append(o)
    n = sum(o[2] for o in out)
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
    res["rigorous_lower"] = float_down(L_rig); res["rigorous_upper"] = res["worst_leaf_upper"]
    res["witness_lower"] = float_down(g_wit)          # the rigorous value of the paper's Ĝ_cert (same witness)
    res["bnb_arg_lower"] = float_down(g_arg)
    res["claim_hi_excess_over_published"] = res["worst_leaf_upper"] - meta["claim_hi"]
    res["ghat_cert_deficit"] = float((arb(meta["ghat_cert"]) - lo_(g_wit)).mid())
    res["bnb_lo_deficit"] = float((arb(meta["bnb_lo"]) - lo_(g_arg)).mid())
    res["ghat_cert"] = meta["ghat_cert"]; res["claim_hi"] = meta["claim_hi"]
    res["leaves"] = n; res["failed_leaves"] = bad[:20]
    res["seconds"] = time.time() - t0
    res["pass"] = all(res["checks"].values())
    out_dir = CERTS.parent / "certificate_checks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1))
    return res



# ------------------------------------------------------------------------------------------ K = sup G₀ (limit class gap)
K_BOX = (0.0, 8.0, -12.0, 12.0)
K_H0 = 0.05
K_MAX_LEVEL = 30


def _h_attained(t1, t2):
    """(upper bound of min, lower bound of max) of h over the exact interval between the exact points t1, t2 (arb):
    attained values only (the endpoints and ±√2 when certainly strictly inside)."""
    H = HAct()
    lo, hi = (t1, t2) if float(t1.mid()) <= float(t2.mid()) else (t2, t1)
    cand = [H.f(lo), H.f(hi)]
    for cc in (H.c, -H.c):
        if lo_(cc) > hi_(lo) and hi_(cc) < lo_(hi):
            cand.append(H.f(cc))
    if not all(c.is_finite() for c in cand):
        return arb("inf"), arb("-inf")
    return min(hi_(c) for c in cand), max(lo_(c) for c in cand)


def g0_upper(u, v):
    """Rigorous upper bound of G₀(u, v) = max(min_O h − max_I h, min_I h − max_O h) at the exact point (u, v)."""
    u, v = arb(u), arb(v)
    i_minU, i_maxL = _h_attained(u * arb("-0.8") + v, u * arb("0.8") + v)
    p_minU, p_maxL = _h_attained(u * arb("1.2") + v, u * arb(2) + v)
    n_minU, n_maxL = _h_attained(u * arb(-2) + v, u * arb("-1.2") + v)
    # min over O <= min(p, n) of attained values; max over I >= attained lower values (exact points throughout)
    return max(hi_(min(p_minU, n_minU) - i_maxL), hi_(i_minU - max(p_maxL, n_maxL)))


def g0_lower(u, v):
    """Rigorous lower bound of G₀ at the exact point (HAct.range includes every possibly-contained critical point)."""
    H = HAct()
    u, v = arb(u), arb(v)
    rng = lambda x1, x2: H.range(*sorted((u * arb(x1) + v, u * arb(x2) + v), key=lambda b: float(b.mid())))
    imn, imx = rng("-0.8", "0.8")
    pmn, pmx = rng("1.2", "2")
    nmn, nmx = rng("-2", "-1.2")
    return max(lo_(min(pmn, nmn) - imx), lo_(imn - max(pmx, nmx)))


def _k_cells(cells, level, target):
    """One level of the branch and bound: returns the cells not yet shown to satisfy sup G₀ <= target, and the worst
    upper bound seen.  Cell = (i, j) at `level`; centre and half-widths are exact dyadic rationals."""
    ctx.prec = PREC
    u0, u1, v0, v1 = (arb(repr(z)) for z in K_BOX)
    hu = arb(K_H0) / 2 ** (level + 1); hv = hu
    T = arb(repr(target))
    keep, worst = [], None
    for i, j in cells:
        uc = u0 + (2 * i + 1) * hu; vc = v0 + (2 * j + 1) * hv
        ub_c = g0_upper(uc, vc)
        smax = 2 * (abs(uc) + hu) + abs(vc) + hv                           # |σ| <= 2|u| + |v| on the cell
        S = max(arb(1), smax * smax / 2 - 1)                                 # sup |h′| on the σ-range (h′ = σ²/2 − 1)
        ub = hi_(ub_c + S * (arb("2.8") * hu + 2 * hv))
        worst = ub if worst is None else max(worst, ub)
        if not (ub <= T):
            keep.append((i, j))
    return keep, (None if worst is None else float_up(worst))


def _k_chunk(args):
    return _k_cells(*args)


def k_domain_lemma_exact():
    """Exact (rational) checks of the domain lemma (math_note_v2 §8): the cubic identity
    h(a) − h(a + d) = d[1 − ((a + d/2)² + d²/12)/2] as a polynomial identity (verified on a 5 x 5 rational grid, which
    determines a polynomial of degree <= 3 in each variable), and 0.12·(50/3) = 2 (the u-bound), and that
    |v| >= √2 + 0.6u implies (v ∓ 0.6u)² >= 2 (sign cases, u >= 0)."""
    from fractions import Fraction as Fr
    h = lambda s: -s + s ** 3 / 6
    grid = [Fr(k, 3) - 2 for k in range(5)]
    ident = all(h(a) - h(a + d) == d * (1 - ((a + d / 2) ** 2 + d ** 2 / 12) / 2) for a in grid for d in grid)
    ubound = Fr(12, 100) * Fr(50, 3) == 2 and (Fr(12, 10) ** 2) / 12 == Fr(12, 100)
    # v >= r + 0.6u with r = √2, u >= 0: v − 0.6u >= r and v + 0.6u >= r + 1.2u >= r; symmetric for v <= −r − 0.6u
    samples = [(Fr(p, 7), Fr(q, 5)) for p in range(0, 60) for q in range(-60, 61)]
    r2 = Fr(2)
    sign_cases = all(((v - Fr(3, 5) * u) ** 2 >= r2 and (v + Fr(3, 5) * u) ** 2 >= r2)
                     for u, v in samples
                     if (v >= 0 and (v - Fr(3, 5) * u) ** 2 >= r2 and v - Fr(3, 5) * u > 0)
                     or (v <= 0 and (v + Fr(3, 5) * u) ** 2 >= r2 and v + Fr(3, 5) * u < 0))
    return {"cubic_identity_exact": ident, "u_bound_exact": ubound, "v_bound_sign_cases": sign_cases}


def check_K(target=None, workers=WORKERS, verbose=True, max_level=K_MAX_LEVEL, sample=None):
    """K = sup G₀ over [0, 8] x [−12, 12]: an independent branch and bound in Arb (every cell: rigorous upper bound at
    its exact centre + sup|h′|·(2.8 hu + 2 hv)), the attained lower end at the published argmax, and the domain lemma
    in exact arithmetic.  `sample`: time only the first `sample` levels."""
    import csv
    from multiprocessing import Pool
    ctx.prec = PREC
    row = next(csv.DictReader((CERTS.parent / "limit_K_base.csv").open()))
    K_lo, K_hi = float(row["K_lo"]), float(row["K_hi"])
    target = K_hi if target is None else target
    res = {"name": "K_base", "claim_lo": K_lo, "claim_hi": K_hi, "target": target, "checks": {}}
    t0 = time.time()
    nu = int(round((K_BOX[1] - K_BOX[0]) / K_H0)); nv = int(round((K_BOX[3] - K_BOX[2]) / K_H0))
    cells = [(i, j) for i in range(nu) for j in range(nv)]
    level, worst_all, per_level = 0, None, []
    with Pool(workers) as p:
        while cells and level <= max_level and (sample is None or level < sample):
            chunks = [(cells[k:k + 2000], level, target) for k in range(0, len(cells), 2000)]
            out = p.map(_k_chunk, chunks)
            keep = [c for o in out for c in o[0]]
            w = max(o[1] for o in out)
            per_level.append({"level": level, "cells": len(cells), "open": len(keep), "worst_upper": w,
                              "seconds": time.time() - t0})
            if verbose:
                print(json.dumps(per_level[-1]), flush=True)
            cells = [(2 * i + di, 2 * j + dj) for i, j in keep for di in (0, 1) for dj in (0, 1)]
            level += 1
    res["levels"] = per_level
    res["checks"]["every_cell_below_target"] = (not cells) and sample is None
    res["checks"]["lower_end_attained"] = bool(g0_lower(row["u"], row["v"]) >= arb(repr(K_lo)))
    res["G0_at_argmax_lower"] = float_down(g0_lower(row["u"], row["v"]))
    res["checks"].update(k_domain_lemma_exact())
    res["checks"]["box_contains_lemma_region"] = bool(math.sqrt(50 / 3) < K_BOX[1] and
                                                      math.sqrt(2) + 0.6 * math.sqrt(50 / 3) < K_BOX[3])
    res["seconds"] = time.time() - t0
    res["pass"] = all(res["checks"].values())
    if sample is None:
        out_dir = CERTS.parent / "certificate_checks"
        out_dir.mkdir(exist_ok=True)
        tag = "" if target == K_hi else f"_target{target:.10f}"
        (out_dir / f"K_base{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps({k: v for k, v in res.items() if k != "levels"}, indent=1, default=str))
    return res

def check(name):
    kind = json.loads((CERTS / f"{name}.json").read_text()).get("kind")
    return {"ghat_enclosure": check_ghat, "limit_status": check_limit}.get(kind, check_finite)(name)


def main(names):
    out = [check(n) for n in names]
    bad = [r["name"] for r in out if not r["pass"]]
    print(f"{len(out)} certificate(s) checked; failures: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["K"]:
        smp = int(sys.argv[2]) if len(sys.argv) > 2 else None
        raise SystemExit(0 if check_K(sample=smp)["pass"] or smp else 1)
    names = sys.argv[1:] or sorted(p.stem for p in CERTS.glob("*.json"))
    raise SystemExit(main(names))
