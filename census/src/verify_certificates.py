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

Track 4 (2026-09-28) adds, with the same rules: outer_exclusion_arb (Link 2 of the limit four-link chain, a fresh
branch and bound in Arb with A as a ball), limit_solve_end (the limit solve-bracket ends: Krawczyk's unique critical
point and the margin's sign there), and region_excluded_arb (check_solve's losing region, by a fresh branch and bound
instead of its exported leaves).  Driver: src/track4_certs.py.

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


def check_finite(name, verbose=True, workers=WORKERS, skip_regions=()):
    """See the module docstring.  Each leaf is verified against what it claims: reason 2 (discarded as outside the
    region) by a rigorous gap bound; reasons 0/1 (kept / discarded by its bound) by a rigorous lower bound of L* at
    least its recorded bound.  For the losing region the chain additionally needs every such bound > U.
    skip_regions: regions whose exported leaves are NOT used (neither tiled nor verified); the caller must then establish
    what that region contributes by other means (check_solve: a fresh branch and bound)."""
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
        if reg in skip_regions:
            continue
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



# ------------------------------------------------------------------------------------------ finite-a solve brackets
def _margin_signs(obj, wl, wh, bl, bh, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Over the box [wl, wh] × [bl, bh] (and b₂ ∈ its certified bracket): ('solves' | 'fails' | None).
    solves: s·max_I f + b₂ < 0 and s·min_O f + b₂ > 0 at every point (the enclosures' outer ends).
    fails:  at every point some sign is wrong -- s·(a lower bound of max_I f) + b₂ > 0, or s·(an upper bound of min_O f)
            + b₂ < 0, with those bounds from single sample points x (valid at every point of the box)."""
    W, B = ball(wl, wh), ball(bl, bh)
    Bb, _ = obj.b_bracket(W, B)
    if Bb is None:
        return None
    fa, S = obj.fa, obj.s
    A = lambda v: arb(repr(v))
    ti = t_interval(A(wl), A(wh), A(bl), A(bh), arb(inner[0]), arb(inner[1]))
    t1 = t_interval(A(wl), A(wh), A(bl), A(bh), arb(outer[0]), arb(outer[1]))
    t2 = t_interval(A(wl), A(wh), A(bl), A(bh), arb(-outer[1]), arb(-outer[0]))
    _, imx = fa.range(*ti)
    o1, _ = fa.range(*t1); o2, _ = fa.range(*t2)
    if hi_(S * imx + hi_(Bb)) < 0 and lo_(S * min(o1, o2) + lo_(Bb)) > 0:
        return "solves"
    for xi in np.linspace(inner[0], inner[1], 17):
        fmn, _ = fa.range(*t_interval(A(wl), A(wh), A(bl), A(bh), A(float(xi)), A(float(xi))))
        if lo_(S * fmn + lo_(Bb)) > 0:
            return "fails"
    for xo in np.r_[np.linspace(*outer, 9), -np.linspace(*outer, 9)]:
        _, fmx = fa.range(*t_interval(A(wl), A(wh), A(bl), A(bh), A(float(xo)), A(float(xo))))
        if hi_(S * fmx + hi_(Bb)) < 0:
            return "fails"
    return None


def solve_sign_on_box(obj, wl, wh, bl, bh, depth=0, max_depth=5):
    """The solve margin's sign on the whole box, subdividing (4-way) up to max_depth; None if undecided or mixed."""
    r = _margin_signs(obj, wl, wh, bl, bh)
    if r is not None or depth >= max_depth:
        return r
    wm, bm = 0.5 * (wl + wh), 0.5 * (bl + bh)
    parts = [solve_sign_on_box(obj, w0, w1, b0, b1, depth + 1, max_depth)
             for w0, w1 in ((wl, wm), (wm, wh)) for b0, b1 in ((bl, bm), (bm, bh))]
    return parts[0] if parts[0] is not None and all(p == parts[0] for p in parts) else None


def region_excluded_arb(obj, U, region, W, h0=0.125, max_depth=10, max_evals=200_000, box=None,
                        witness_from_depth=2):
    """A fresh branch and bound in Arb (it does not use any exported leaf): every (w₁, b₁) with |w₁| <= W, b₁ ∈ [0, 2π],
    in `region` ('-': G <= 0, '+': G > 0; orientation w₂ > 0, as in the certificates) has L* > U (an Arb ball; strict).
    Cells are dyadic squares of side h0 from (−h0·⌈W/h0⌉, 0) (every endpoint an exact double); a cell is dropped only if
    it lies in |w₁| >= W (exact comparison; the localisation lemma covers it); it is certified if rigorously outside the
    region (gap bounds) or if its rigorous lower bound of L* exceeds U; otherwise it is split in four.  b₁ ∈ [0, 2π]
    suffices: f_a(t + 2π) = f_a(t) + 2π, which the profiled b₂ absorbs, and G is unchanged.
    Verdicts: 'pass'; 'fail' (a witness: a cell centre rigorously in the region with a rigorous upper bound of L* below
    U's lower end, so the claim is false); 'unresolved' (a cap).  box = (wl, wh, bl, bh) restricts the domain (tests)."""
    from fractions import Fraction as Fr
    t0 = time.time()
    Wf = Fr(float(W))
    if box is None:
        nw = int(math.ceil(float(W) / h0)); nb = int(math.ceil(2 * math.pi / h0))
        box = (-nw * h0, nw * h0, 0.0, nb * h0)
        assert box[3] >= 2 * math.pi and box[1] >= W
    wl0, wh0, bl0, bh0 = box
    n_w, n_b = int(round((wh0 - wl0) / h0)), int(round((bh0 - bl0) / h0))
    assert wl0 + n_w * h0 == wh0 and bl0 + n_b * h0 == bh0
    stack = [(wl0 + i * h0, wl0 + (i + 1) * h0, bl0 + j * h0, bl0 + (j + 1) * h0, 0) for i in range(n_w) for j in range(n_b)]
    st = {"evals": 0, "outside_region": 0, "above_U": 0, "dropped_beyond_W": 0, "max_depth": 0, "witness_tries": 0}
    unresolved, witness, lb_min = [], None, math.inf
    while stack:
        wl, wh, bl, bh, depth = stack.pop()
        if Fr(wl) >= Wf or Fr(wh) <= -Wf:
            st["dropped_beyond_W"] += 1
            continue
        if st["evals"] >= max_evals:
            unresolved.append((wl, wh, bl, bh, "evaluation cap"))
            continue
        st["evals"] += 1
        st["max_depth"] = max(st["max_depth"], depth)
        wc, bc, hw, hb = 0.5 * (wl + wh), 0.5 * (bl + bh), 0.5 * (wh - wl), 0.5 * (bh - bl)
        assert wc - hw == wl and wc + hw == wh and bc - hb == bl and bc + hb == bh
        gl, gu = obj.gap_bounds(wl, wh, bl, bh)
        if (region == "-" and bool(gl > 0)) or (region == "+" and bool(gu <= 0)):
            st["outside_region"] += 1
            continue
        lb = obj.lower_bound(wc, bc, hw, hb)
        if lb is not None and bool(lb > U):
            st["above_U"] += 1
            lb_min = min(lb_min, float_down(lo_(lb)))
            continue
        if depth >= witness_from_depth:
            gpl, gpu = obj.gap_bounds(wc, wc, bc, bc)
            inside = bool(gpu <= 0) if region == "-" else bool(gpl > 0)
            Bb = obj.b_bracket(arb(wc), arb(bc))[0] if inside else None
            if Bb is not None:
                st["witness_tries"] += 1
                up = obj.upper_at(wc, bc, float(Bb.mid()))
                if bool(arb(hi_(up)) < arb(lo_(U))):
                    witness = {"w1": wc, "b1": bc, "L_upper": float_up(hi_(up)), "U_lower": float_down(lo_(U))}
                    break
        if depth < max_depth:
            stack += [(a0, a1, b0, b1, depth + 1) for a0, a1 in ((wl, wc), (wc, wh)) for b0, b1 in ((bl, bc), (bc, bh))]
        else:
            unresolved.append((wl, wh, bl, bh, "depth cap"))
    verdict = "fail" if witness is not None else ("pass" if not unresolved and not stack else "unresolved")
    return {"verdict": verdict, "witness": witness, "unresolved_n": len(unresolved) + len(stack),
            "unresolved_first": unresolved[:10], **st, "min_cell_lower_bound": lb_min if verdict == "pass" else None,
            "region": region, "domain": list(box), "h0": h0, "max_depth_cap": max_depth, "max_evals_cap": max_evals,
            "seconds": time.time() - t0}


def check_solve(name, verbose=True, workers=WORKERS):
    """Finite-a solve-bracket end.  (1) The WINNING region's exported leaves (tolerance 1e−11): hashes, lemma W, exact
    tiling, every leaf's claim verified (check_finite, losing region skipped), and every leaf discarded by its bound has
    claim > U.  (2) The LOSING region: its exported leaves are not used; a fresh Arb branch and bound
    (region_excluded_arb) shows L* > U on all of it.  (3) Hence the global minimiser lies in the union of the winning
    region's KEPT leaves (recomputed from the integer indices); on every kept leaf (b₂ in its certified bracket) the
    solve margin's sign is decided in Arb, and it must be the same on all of them and equal the published sign.
    (Track 4, 2026-09-28: (2) replaces verifying 1e−11 losing-region leaves, of which there were millions at a = 1.45.)"""
    meta = json.loads((CERTS / f"{name}.json").read_text())
    win, lose = meta["winning_region"], meta["losing_region"]
    res = check_finite(name, verbose=False, workers=workers, skip_regions=(lose,))
    res["name"] = name
    dat = np.load(CERTS / f"{name}.npz")
    obj = Objective(dat["x"], dat["y"], meta["s"], meta["a"])
    if meta["U_point"] is None:
        res["checks"]["solve_winner_not_constant"] = False
        res["pass"] = False
        return res
    U = obj.upper_at(*meta["U_point"])
    lv, iw, ib = dat[f"level_{win}"], dat[f"iw_{win}"], dat[f"ib_{win}"]
    reason, claim = dat[f"reason_{win}"], dat[f"lb_{win}"]
    res["checks"]["winning_discarded_exceed_U"] = all(arb(float(c)) > U for c in claim[reason == 1])
    ex = region_excluded_arb(obj, U, lose, meta["W"])
    res["losing_region_fresh_bnb"] = ex
    res["checks"]["losing_region_above_U"] = ex["verdict"] == "pass"
    kept = reason == 0
    hw = meta["hw0"] / 2.0 ** lv[kept]; hb = meta["hb0"] / 2.0 ** lv[kept]
    cw = -meta["W"] + (iw[kept] + 0.5) * 2 * hw; cb = (ib[kept] + 0.5) * 2 * hb
    # The sign is decided on EVERY kept leaf (not on their bounding box: the kept set has two mirror clusters, w₁ ↦ −w₁,
    # since the data and windows are x-symmetric), each widened by 1e−12 (>> the float rounding of the leaf edges).
    E = (float((cw - hw).min()), float((cw + hw).max()), float((cb - hb).min()), float((cb + hb).max()))
    res["E"] = E; res["kept_leaves"] = int(kept.sum())
    signs = [solve_sign_on_box(obj, float(c - h) - 1e-12, float(c + h) + 1e-12, float(d - g) - 1e-12,
                               float(d + g) + 1e-12) for c, h, d, g in zip(cw, hw, cb, hb)]
    res["kept_leaf_signs"] = {str(k): signs.count(k) for k in set(signs)}
    sign = signs[0] if signs and all(v == signs[0] for v in signs) else None
    res["solve_sign"] = sign
    res["published_sign"] = meta["published_sign"]
    res["checks"]["sign_decided"] = sign is not None
    res["checks"]["sign_equals_published"] = sign == meta["published_sign"]
    res["pass"] = all(res["checks"].values())
    out_dir = CERTS.parent / "certificate_checks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1, default=str))
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

# ------------------------------------------------------------------------------------------ limit problem: Krawczyk and PD boxes
def _hh(t):
    return -t + t * t * t / 6


def _h1(t):
    return t * t / 2 - 1


def _r(t):
    t2 = t * t
    return t2 * t / 6 - t2 * t2 * t / 120


def _r1(t):
    t2 = t * t
    return t2 / 2 - t2 * t2 / 24


def _sig(z):
    return 1 / (1 + (-z).exp())


def _limit_data():
    d = np.load(CERTS / "limit_A_lo.npz")
    return [arb(float(v)) for v in d["x"]], [int(v) for v in d["y"]]


def switch_system_arb(X, xs, ys, want_J=True):
    """Φ(p, q, b, A) = (∂_p L, ∂_q L, ∂_b L, G₀^act) with L = mean softplus(z) − y z, z = A h(p x + q) + b and the active
    gap G₀^act = h(q − 1.2p) − h(q + 0.8p); its Jacobian and ∂_εΦ (r(σ) = σ³/6 − σ⁵/120, the O(ε) profile term), in Arb."""
    p, q, b, A = X
    n = len(xs)
    F = [arb(0)] * 3
    J = [[arb(0)] * 4 for _ in range(3)]
    E = [arb(0)] * 3
    for xi, yi in zip(xs, ys):
        sg = p * xi + q
        hs, h1s = _hh(sg), _h1(sg)
        z = A * hs + b
        S = _sig(z); res = S - yi
        dz = [A * h1s * xi, A * h1s, arb(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if not want_J:
            continue
        w = S * (1 - S)
        h2s = sg
        d2 = [[A * h2s * xi * xi, A * h2s * xi, arb(0)], [A * h2s * xi, A * h2s, arb(0)], [arb(0), arb(0), arb(0)]]
        dA = [h1s * xi, h1s, arb(0)]
        rs, r1s = _r(sg), _r1(sg)
        de = [A * r1s * xi, A * r1s, arb(0)]
        for i in range(3):
            for j in range(3):
                J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
            J[i][3] += w * hs * dz[i] + res * dA[i]
            E[i] += w * A * rs * dz[i] + res * de[i]
    F = [f / n for f in F]
    sO, sI = q - arb("1.2") * p, q + arb("0.8") * p
    F.append(_hh(sO) - _hh(sI))
    if not want_J:
        return F
    J = [[J[i][j] / n for j in range(4)] for i in range(3)]
    J.append([-arb("1.2") * _h1(sO) - arb("0.8") * _h1(sI), _h1(sO) - _h1(sI), arb(0), arb(0)])
    E = [e / n for e in E] + [_r(sO) - _r(sI)]
    return F, J, E


def _mid(b):
    return float(b.mid())


def krawczyk_arb(fun, xt, rad):
    """Krawczyk test in Arb: K = x̃ − Y F(x̃) + (I − Y J(X))(X − x̃) ⊂ int X proves a unique zero in X.  Y is any float
    matrix (the inverse of mid J); every other quantity is an Arb enclosure."""
    k = len(xt)
    X = [ball(xt[i] - rad[i], xt[i] + rad[i]) for i in range(k)]
    F0 = fun([arb(repr(v)) for v in xt], False)
    J = fun(X, True)[1]
    Y = np.linalg.inv(np.array([[_mid(J[i][j]) for j in range(k)] for i in range(k)]))
    Kb = []
    for i in range(k):
        acc = arb(repr(xt[i]))
        for j in range(k):
            acc -= arb(repr(float(Y[i, j]))) * F0[j]
        for j in range(k):
            cij = arb(1 if i == j else 0) - sum((arb(repr(float(Y[i, l]))) * J[l][j] for l in range(k)), arb(0))
            acc += cij * (X[j] - arb(repr(xt[j])))
        Kb.append(acc)
    ok = all((lo_(Kb[i]) > lo_(X[i])) and (hi_(Kb[i]) < hi_(X[i])) for i in range(k))
    return ok, Kb, X, J


def lin_enclose_arb(J, b):
    """Enclose {d : J d = −b} for every J, b in the balls: d̃ ± e (∞-norm), e = ‖Y(J d̃ + b)‖/(1 − ‖I − YJ‖)."""
    k = len(b)
    Y = np.linalg.inv(np.array([[_mid(J[i][j]) for j in range(k)] for i in range(k)]))
    dt = -Y @ np.array([_mid(v) for v in b])
    Ya = [[arb(repr(float(Y[i, j]))) for j in range(k)] for i in range(k)]
    Rv = [sum((J[i][j] * arb(repr(float(dt[j]))) for j in range(k)), arb(0)) + b[i] for i in range(k)]
    YR = [sum((Ya[i][l] * Rv[l] for l in range(k)), arb(0)) for i in range(k)]
    nE = max(sum(abs(arb(1 if i == j else 0) - sum((Ya[i][l] * J[l][j] for l in range(k)), arb(0))) for j in range(k))
             for i in range(k))
    if not (nE < 1):
        return None
    e = max(abs(v) for v in YR) / (1 - nE)
    return [arb(repr(float(dt[i]))) + ball(-float(e.upper()), float(e.upper())) for i in range(k)]


def check_krawczyk(verbose=True):
    """The c₁ chain's Krawczyk boxes, re-derived in Arb: (1) the switch system at the published point, radius 1e−9:
    unique zero, A* enclosure; active set (inner max at 0.8, outer min at −1.2, all other candidates strictly
    dominated over the box; no interior critical point); A′(0)/A* by an interval linear solve; (2) the K vertex
    (the tied corner) by the 2×2 vertex system, and K there."""
    import csv
    ctx.prec = PREC
    row = next(csv.DictReader((CERTS.parent / "first_order_c1.csv").open()))
    xs, ys = _limit_data()
    xt = [float(row["p_star"]), float(row["q_star"]), float(row["b_star"]),
          0.5 * (float(row["A_star_lo"]) + float(row["A_star_hi"]))]
    fun = lambda X, J=True: switch_system_arb(X, xs, ys, J)[:2] if J else switch_system_arb(X, xs, ys, False)
    ok, Kb, X, _ = krawczyk_arb(fun, xt, [1e-9] * 4)
    p, q = X[0], X[1]
    I_act, O_act = _hh(q + arb("0.8") * p), _hh(q - arb("1.2") * p)
    others_I = [_hh(q - arb("0.8") * p)]
    others_O = [_hh(q - 2 * p), _hh(q + arb("1.2") * p), _hh(q + 2 * p)]
    R2 = arb(2).sqrt()
    crit_ok = (lo_(q - arb("0.8") * p) > hi_(-R2)) and (hi_(q - arb("1.2") * p) < lo_(R2)) and \
              (lo_(q + arb("1.2") * p) > hi_(R2))
    active_ok = all(lo_(I_act) > hi_(o) for o in others_I) and all(hi_(O_act) < lo_(o) for o in others_O) and crit_ok
    _, J, Eps = switch_system_arb(X, xs, ys, True)
    d = lin_enclose_arb(J, Eps)
    A1_over_A = d[3] / X[3] if d is not None else None
    res = {"switch_krawczyk_ok": bool(ok), "active_set_unique": bool(active_ok),
           "A_star_lo": float_down(lo_(Kb[3])), "A_star_hi": float_up(hi_(Kb[3])),
           "A1_over_A_lo": float_down(lo_(A1_over_A)) if d is not None else None,
           "A1_over_A_hi": float_up(hi_(A1_over_A)) if d is not None else None}
    # published enclosures must contain the Arb ones' points (consistency) -- the Arb enclosure is the verified one
    res["A_star_overlaps_published"] = bool(res["A_star_lo"] <= float(row["A_star_hi"]) and
                                            res["A_star_hi"] >= float(row["A_star_lo"]))
    # (2) the vertex
    def vfun(Xv, want_J=True):
        u, v = Xv
        s_ = {k: v + arb(repr(k)) * u for k in (-0.8, 0.8, -2.0, -1.2)}
        F = [_hh(s_[-0.8]) - _hh(s_[0.8]), _hh(s_[-2.0]) - _hh(s_[-1.2])]
        if not want_J:
            return F
        Jv = [[arb("-0.8") * _h1(s_[-0.8]) - arb("0.8") * _h1(s_[0.8]), _h1(s_[-0.8]) - _h1(s_[0.8])],
              [-2 * _h1(s_[-2.0]) + arb("1.2") * _h1(s_[-1.2]), _h1(s_[-2.0]) - _h1(s_[-1.2])]]
        return F, Jv
    vok, Kv, Xv, _ = krawczyk_arb(vfun, [float(row["vertex_u"]), float(row["vertex_v"])], [1e-9, 1e-9])
    u, v = Kv
    Kval = _hh(v - arb("1.2") * u) - _hh(v + arb("0.8") * u)
    res.update({"vertex_krawczyk_ok": bool(vok), "K_vertex_lo": float_down(lo_(Kval)), "K_vertex_hi": float_up(hi_(Kval))})
    res["pass"] = all(res[k] for k in ("switch_krawczyk_ok", "active_set_unique", "vertex_krawczyk_ok",
                                       "A_star_overlaps_published")) and d is not None
    out_dir = CERTS.parent / "certificate_checks"; out_dir.mkdir(exist_ok=True)
    (out_dir / "c1_krawczyk.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    return res


def _pd_box(args):
    """(p, q, A) sub-box: b* enclosed (monotone bracket), the (p, q, b) Hessian enclosed over box × bracket; PD of
    H − c·I by Sylvester's criterion on the Arb enclosure (every member of the enclosure is then PD)."""
    pl, ph, ql, qh, al, ah, c = args
    ctx.prec = PREC
    d = np.load(CERTS / "limit_A_lo.npz")
    obj = Objective(d["x"], d["y"], ball(al, ah), None, act=HAct())
    P, Q, A = ball(pl, ph), ball(ql, qh), ball(al, ah)
    Bb, _ = obj.b_bracket(P, Q)
    if Bb is None:
        return {"box": args[:6], "b_ok": False, "pd": False}
    H = [[arb(0)] * 3 for _ in range(3)]
    for xi, yi in zip(obj.X, obj.Y):
        sg = P * xi + Q
        z = A * _hh(sg) + Bb
        S = _sig(z); w = S * (1 - S); res = S - yi
        dz = [A * _h1(sg) * xi, A * _h1(sg), arb(1)]
        d2 = {(0, 0): A * sg * xi * xi, (0, 1): A * sg * xi, (1, 1): A * sg}
        for i in range(3):
            for j in range(3):
                H[i][j] += w * dz[i] * dz[j] + res * d2.get((min(i, j), max(i, j)), arb(0))
    n = len(obj.X)
    M = [[H[i][j] / n - (arb(repr(c)) if i == j else 0) for j in range(3)] for i in range(3)]
    m1 = M[0][0]
    m2 = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    m3 = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
          + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    return {"box": args[:6], "b_ok": True, "pd": bool(m1 > 0 and m2 > 0 and m3 > 0)}


def check_pd(name, p0, q0, rho, A_range, split, c, workers=WORKERS, verbose=True):
    from multiprocessing import Pool
    pe = np.linspace(p0 - rho, p0 + rho, split[0] + 1); qe = np.linspace(q0 - rho, q0 + rho, split[1] + 1)
    ae = np.linspace(A_range[0], A_range[1], split[2] + 1)
    boxes = [(pe[i], pe[i + 1], qe[j], qe[j + 1], ae[k], ae[k + 1], c)
             for i in range(split[0]) for j in range(split[1]) for k in range(split[2])]
    t0 = time.time()
    with Pool(workers) as pl:
        out = pl.map(_pd_box, boxes)
    res = {"name": name, "boxes": len(out), "b_bracket_all": all(r["b_ok"] for r in out),
           "pd_all_with_margin_c": all(r["pd"] for r in out), "c": c, "seconds": time.time() - t0,
           "failed": [r["box"] for r in out if not r["pd"]][:10]}
    res["pass"] = res["b_bracket_all"] and res["pd_all_with_margin_c"]
    out_dir = CERTS.parent / "certificate_checks"; out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    return res



def _ring_interval(args):
    """No critical point of L0*(·; A) in the ring rho_in <= |θ − c|_∞ <= rho_out for any A in [A_lo, A_hi]: every box
    (padded) has b* bracketed and 0 ∉ ∂_p L or 0 ∉ ∂_q L (the envelope gradient at b*), in Arb; boxes split up to
    `max_depth`.  Boxes inside the rho_in box (covered by the PD certificate) are dropped."""
    A_lo, A_hi, p0, q0, rho_in, rho_out, side, max_depth = args
    ctx.prec = PREC
    d = np.load(CERTS / "limit_A_lo.npz")
    obj = Objective(d["x"], d["y"], ball(A_lo, A_hi), None, act=HAct())
    pad = 1e-12
    n = int(round(2 * rho_out / side))
    cells = [(-rho_out + (i + 0.5) * side, -rho_out + (j + 0.5) * side, side / 2) for i in range(n) for j in range(n)]
    boxes, depth, gmin = 0, 0, math.inf
    while cells and depth <= max_depth:
        nxt = []
        for cp, cq, hh in cells:
            if abs(cp) + hh <= rho_in - pad and abs(cq) + hh <= rho_in - pad:
                continue
            boxes += 1
            P = ball(p0 + cp - hh - pad, p0 + cp + hh + pad); Q = ball(q0 + cq - hh - pad, q0 + cq + hh + pad)
            Bb, Z = obj.b_bracket(P, Q)
            ok = False
            if Bb is not None:
                _, gp, gq = obj.value_and_grad(P, Q, Bb, Z)
                ok = (not gp.contains(0)) or (not gq.contains(0))
                if ok:
                    mag = max(0 if gp.contains(0) else float(min(abs(lo_(gp)), abs(hi_(gp))).mid()),
                              0 if gq.contains(0) else float(min(abs(lo_(gq)), abs(hi_(gq))).mid()))
                    gmin = min(gmin, mag)
            if not ok:
                h2 = hh / 2
                nxt += [(cp + dx * h2, cq + dy * h2, h2) for dx in (-1, 1) for dy in (-1, 1)]
        cells, depth = nxt, depth + 1
    return {"A_lo": A_lo, "A_hi": A_hi, "certified": not cells, "boxes": boxes, "left": len(cells), "grad_lower": gmin}


def check_ring(name, centre, intervals, rho_in=0.05, rho_out=0.15, side=0.0125, max_depth=4, workers=WORKERS,
               verbose=True):
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(a0, a1, centre[0], centre[1], rho_in, rho_out, side, max_depth) for a0, a1 in intervals]
    with Pool(workers) as pl:
        out = pl.map(_ring_interval, jobs)
    res = {"name": name, "intervals": len(out), "all_certified": all(r["certified"] for r in out),
           "boxes": sum(r["boxes"] for r in out), "min_grad_lower": min(r["grad_lower"] for r in out),
           "failed": [(r["A_lo"], r["A_hi"], r["left"]) for r in out if not r["certified"]], "seconds": time.time() - t0}
    res["pass"] = res["all_certified"]
    out_dir = CERTS.parent / "certificate_checks"; out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------ Track 4: outer exclusion (Link 2)
# The limit four-link chain (math_note_v2 §3(c)) needs, for every A in an interval J and every θ = (p, q) in K(24)
# (p >= 0 by x-symmetry) outside the closed 0.15 box around θ_c:  L0*(θ; A) > L0*(θ_c; A) + margin.
# The producer (certificates_v2.outer_exclusion) is a FLOAT branch and bound per A-sub-interval: a second-order cell
# bound at the midpoint A_c, an A-term from convexity in A and a median-centred envelope bound, and a 1e−9 allowance
# for float rounding.  This check shares none of that.  It is a fresh branch and bound in Arb:
#   * A is carried as a BALL in the objective (never sampled, no convexity-in-A term on the competitor side);
#   * the logit is written z = A·(h(σ) − c) + b′ with a per-cell constant c.  This is the same model (b′ = b + A·c and b is
#     profiled over ℝ), so L0* is unchanged; it only narrows the enclosure where h(σ) is large and nearly constant;
#   * each cell's lower bound is Objective.lower_bound (the checker's first-order mean-value bound and its direct
#     bound), valid for every θ in the cell and every A in the ball;
#   * the branch side: L0*(θ_c; ·) is convex in A (the loss is jointly convex in (A, b) since z is jointly linear in them,
#     and a partial minimum over b of a jointly convex function is convex), so on J = [A_i, A_j] it is at most
#     max(U(A_i), U(A_j)), with U(A) a rigorous Arb upper bound of L(θ_c, b̂; A) >= L0*(θ_c; A) at a float b̂;
#   * cells are dyadic squares from the integer grid [0, P] × [−52, 52] (every endpoint an exact double); a cell is
#     dropped only if it lies in the closed rho box (exact rational test) or is disjoint from K(P) (Arb test);
#   * a failed cell is split in four (θ); A is never split (one run per published A-sub-interval).
# Verdicts: 'pass' (every cell certified), 'fail' (a witness: a point θ outside the box and a grid A with a rigorous
# upper bound of L0*(θ; A) below a rigorous lower bound of L0*(θ_c; A) + margin, i.e. the claim is false there),
# 'unresolved' (a cap reached, neither shown).
class HShift(HAct):
    """h(σ) − c for a constant c (an exact double): the logit A·h(σ) + b is A·(h(σ) − c) + b′ with b′ = b + A·c."""

    def __init__(self, c):
        super().__init__()
        self.cc = arb(c)
        self.cf = float(c)

    def f(self, t):
        return -t + t * t * t / 6 - self.cc

    def np_f(self, t):
        return -t + t ** 3 / 6 - self.cf


def _float_profile_loss(xf, yf, A, p, q):
    """Float estimate of L0*(p, q; A) (profiled b by bisection).  Only PROPOSES witness points; never decides."""
    z0 = A * (-(p * xf + q) + (p * xf + q) ** 3 / 6)
    z0 = z0 - np.median(z0)
    yb = yf.mean()
    lo, hi = -float(np.abs(z0).max()) - 50.0, float(np.abs(z0).max()) + 50.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        v = np.exp(-np.logaddexp(0, -(z0 + m))).mean() - yb
        lo, hi = (m, hi) if v < 0 else (lo, m)
    z = z0 + 0.5 * (lo + hi)
    return float((np.logaddexp(0, z) - yf * z).mean())


def _inside_closed_box(pl, ph, ql, qh, p0, q0, rho):
    """Exact (rational) test: the cell [pl, ph] × [ql, qh] lies in the closed box |θ − (p0, q0)|_∞ <= rho."""
    from fractions import Fraction as Fr
    P0, Q0, R = Fr(p0), Fr(q0), Fr(rho)
    return Fr(pl) >= P0 - R and Fr(ph) <= P0 + R and Fr(ql) >= Q0 - R and Fr(qh) <= Q0 + R


def _point_outside_box(p, q, p0, q0, rho):
    from fractions import Fraction as Fr
    return max(abs(Fr(p) - Fr(p0)), abs(Fr(q) - Fr(q0))) > Fr(rho)


def _disjoint_from_K(pl, ph, ql, qh, X=2):
    """K(P) ∩ {p >= 0} = {0 <= p <= P, |q| <= 2√2 + X·p}.  The cell (0 <= p <= ph) is disjoint from it if every q in it
    has |q| > 2√2 + X·ph (Arb comparison)."""
    lim = 2 * arb(2).sqrt() + arb(X) * arb(ph)
    return bool(arb(ql) > lim) or bool(arb(qh) < -lim)


def outer_exclusion_arb(x, y, centre, A_lo, A_hi, margin, P=24.0, h0=1.0, q_extent=52.0, rho=0.15,
                        max_depth=12, max_evals=400_000, region=None, witness_from_depth=2, verbose=False):
    """See the section comment.  One A-interval J = [A_lo, A_hi] (a ball in the objective).
    region: (pl, ph, ql, qh) to restrict the domain (tests); default [0, P] × [−q_extent, q_extent] ∩ K(P).
    Returns the verdict, the certified margin, counts and the branch values."""
    ctx.prec = PREC
    t0 = time.time()
    xf, yf = np.asarray(x, float), np.asarray(y, float)
    p0, q0 = float(centre[0]), float(centre[1])
    A_edges = [float(A_lo), float(A_hi)]
    K = 1
    assert 0 < A_edges[0] < A_edges[1]
    assert 2 * math.sqrt(2) + 2 * P < q_extent and float(np.abs(xf).max()) <= 2.0
    exact = [Objective(xf, yf, A_edges[k], None, act=HAct()) for k in range(K + 1)]
    U_hi, U_lo = [], []
    for o in exact:
        Bb, _ = o.b_bracket(arb(p0), arb(q0))
        if Bb is None:
            return {"verdict": "unresolved", "note": "branch bias not bracketed"}
        U_hi.append(arb(hi_(o.upper_at(p0, q0, float(Bb.mid())))))       # >= L0*(θ_c; A_k)
        U_lo.append(arb(lo_(o.lower_bound(p0, q0, 0.0, 0.0))))            # <= L0*(θ_c; A_k)
    mg = arb(float(margin))                               # the exact double
    o = Objective(xf, yf, ball(A_edges[0], A_edges[1]), None, act=HShift(0.0))
    branch = max(U_hi[0], U_hi[1])                      # exact points (upper ends); convexity in A: max over J
    Uf = [_mid(u) for u in U_lo]
    pl0, ph0, ql0, qh0 = region if region is not None else (0.0, P, -q_extent, q_extent)
    n_p, n_q = int(round((ph0 - pl0) / h0)), int(round((qh0 - ql0) / h0))
    assert abs(pl0 + n_p * h0 - ph0) == 0 and abs(ql0 + n_q * h0 - qh0) == 0
    stack = [(pl0 + a * h0, pl0 + (a + 1) * h0, ql0 + b * h0, ql0 + (b + 1) * h0, 0)
             for a in range(n_p) for b in range(n_q)]
    stats = {"evals": 0, "leaves_certified": 0, "dropped_in_box": 0, "dropped_outside_K": 0, "theta_splits": 0,
             "max_depth": 0, "witness_tries": 0}
    marg, lb_min = math.inf, math.inf          # min over certified leaves of (lb − branch), and of lb
    unresolved, witness = [], None
    while stack:
        pl, ph, ql, qh, depth = stack.pop()
        if _inside_closed_box(pl, ph, ql, qh, p0, q0, rho):
            stats["dropped_in_box"] += 1
            continue
        if _disjoint_from_K(pl, ph, ql, qh):
            stats["dropped_outside_K"] += 1
            continue
        if stats["evals"] >= max_evals:
            unresolved.append((pl, ph, ql, qh, "evaluation cap"))
            continue
        stats["evals"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        pc, qc, hp, hq = 0.5 * (pl + ph), 0.5 * (ql + qh), 0.5 * (ph - pl), 0.5 * (qh - ql)
        assert pc - hp == pl and pc + hp == ph and qc - hq == ql and qc + hq == qh      # exact dyadic cells
        sc = pc * xf + qc
        o.fa = HShift(float(np.median(-sc + sc ** 3 / 6)))
        lb = o.lower_bound(pc, qc, hp, hq)
        if lb is not None and bool(lb > branch + mg):
            stats["leaves_certified"] += 1
            marg = min(marg, float_down(lo_(lb) - hi_(branch)))
            lb_min = min(lb_min, float_down(lo_(lb)))
            continue
        # a failed cell: look for a witness that the claim is false (float proposes, Arb decides)
        if depth >= witness_from_depth and _point_outside_box(pc, qc, p0, q0, rho):
            for k in (0, 1):
                if _float_profile_loss(xf, yf, A_edges[k], pc, qc) < Uf[k] + float(margin) + 1e-6:
                    stats["witness_tries"] += 1
                    Bb, _ = exact[k].b_bracket(arb(pc), arb(qc))
                    if Bb is not None:
                        up = arb(hi_(exact[k].upper_at(pc, qc, float(Bb.mid()))))
                        if bool(up < U_lo[k] + mg):
                            witness = {"p": pc, "q": qc, "A": A_edges[k], "L_upper": float_up(up),
                                       "branch_lower_plus_margin": float_down(lo_(U_lo[k] + mg))}
                            break
            if witness is not None:
                break
        if depth < max_depth:
            stack += [(a0, a1, b0, b1, depth + 1) for a0, a1 in ((pl, pc), (pc, ph)) for b0, b1 in ((ql, qc), (qc, qh))]
            stats["theta_splits"] += 1
        else:
            unresolved.append((pl, ph, ql, qh, "depth cap"))
        if verbose and stats["evals"] % 5000 == 0:
            print(json.dumps({**stats, "stack": len(stack), "s": round(time.time() - t0, 1)}), flush=True)
    verdict = "fail" if witness is not None else ("pass" if not unresolved and not stack else "unresolved")
    return {"verdict": verdict, "witness": witness, "unresolved_n": len(unresolved) + len(stack),
            "unresolved_first": unresolved[:10], **stats,
            "min_margin_certified": marg if verdict == "pass" else None,
            "min_cell_lower_bound": lb_min if verdict == "pass" else None,
            "branch_upper": [float_up(u) for u in U_hi], "branch_lower": [float_down(u) for u in U_lo],
            "A_lo": A_edges[0], "A_hi": A_edges[1], "centre": [p0, q0], "margin_target": margin, "rho": rho, "P": P,
            "h0": h0, "max_depth_cap": max_depth, "max_evals_cap": max_evals,
            "seconds": time.time() - t0}


def _verified_limit_data():
    """The 800-point population from the hash-checked limit certificate file (the same data the searches used)."""
    if not files_match_manifest("limit_A_lo"):
        raise RuntimeError("limit_A_lo certificate files do not match the committed hashes")
    d = np.load(CERTS / "limit_A_lo.npz")
    x, y = d["x"], d["y"]
    sym = sorted(zip(x.tolist(), y.tolist())) == sorted(zip((-x).tolist(), y.tolist()))
    return x, y, sym


# ------------------------------------------------------------------------------------------ Track 4: limit solve bracket ends
def grad_system_arb(X, xs, ys, A, want_J=True):
    """∇_{(p, q, b)} L and its Jacobian (the (p, q, b) Hessian) at fixed A, L = mean softplus(z) − y z, z = A h(p x + q) + b."""
    p, q, b = X
    n = len(xs)
    F = [arb(0)] * 3
    J = [[arb(0)] * 3 for _ in range(3)]
    for xi, yi in zip(xs, ys):
        sg = p * xi + q
        h1s = _h1(sg)
        z = A * _hh(sg) + b
        S = _sig(z); res = S - yi
        dz = [A * h1s * xi, A * h1s, arb(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if want_J:
            w = S * (1 - S)
            d2 = [[A * sg * xi * xi, A * sg * xi, arb(0)], [A * sg * xi, A * sg, arb(0)], [arb(0)] * 3]
            for i in range(3):
                for j in range(3):
                    J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
    F = [f / n for f in F]
    if not want_J:
        return F
    return F, [[J[i][j] / n for j in range(3)] for i in range(3)]


def limit_solve_margin(P, Q, B, A, inner=(-0.8, 0.8), outer=(1.2, 2.0), n_pts=33):
    """Enclosure of the limit solve margin M = min(−(A·max_I h(p x + q) + b), A·min_O h(p x + q) + b) over the box P × Q × B
    (balls; A > 0 exact).  Lower end: the exact ranges over the windows (HAct.range, every possibly-contained critical
    point).  Upper end: M <= −(A h(σ(x_i)) + b) and M <= A h(σ(x_o)) + b at any fixed x_i ∈ I, x_o ∈ O, so the smallest
    upper end over a grid of fixed points is an upper bound.  Returns (lower, upper) as Arb exact endpoints."""
    hact = HAct()
    D = lambda v: arb(f"{v:.12f}")                     # the decimal window point (a ball containing it)
    pl, ph, ql, qh = lo_(P), hi_(P), lo_(Q), hi_(Q)
    _, imx = hact.range(*t_interval(pl, ph, ql, qh, D(inner[0]), D(inner[1])))
    o1, _ = hact.range(*t_interval(pl, ph, ql, qh, D(outer[0]), D(outer[1])))
    o2, _ = hact.range(*t_interval(pl, ph, ql, qh, D(-outer[1]), D(-outer[0])))
    low = min(lo_(-(A * imx + hi_(B))), lo_(A * min(o1, o2) + lo_(B)))
    ups = []
    for xi in np.linspace(inner[0], inner[1], n_pts):
        ups.append(hi_(-(A * hact.f(P * D(float(xi)) + Q) + B)))
    for xo in np.r_[np.linspace(*outer, n_pts), -np.linspace(*outer, n_pts)]:
        ups.append(hi_(A * hact.f(P * D(float(xo)) + Q) + B))
    return low, min(ups)


def limit_solve_end(x, y, A, start, box_centre, rho_in=0.05, rad=1e-9, newton_steps=12):
    """At fixed A: Krawczyk proves a unique zero of ∇_{(p, q, b)} L in a box of radius `rad` around a float Newton point
    started from `start` (p, q, b); the box's (p, q) must lie in the rho_in box around `box_centre` (where, by the chain,
    the global minimiser is the unique critical point); the solve margin is enclosed over the box and its sign decided."""
    ctx.prec = PREC
    xs = [arb(float(v)) for v in x]; ys = [int(v) for v in y]
    Aa = arb(A)

    def fun(X, want_J=True):
        return grad_system_arb(X, xs, ys, Aa, want_J)
    xt = [float(v) for v in start]
    for _ in range(newton_steps):                                     # float Newton on the Arb midpoints: proposes only
        Fm, Jm = fun([arb(v) for v in xt], True)
        step = np.linalg.solve(np.array([[_mid(Jm[i][j]) for j in range(3)] for i in range(3)]),
                               np.array([_mid(f) for f in Fm]))
        xt = [xt[i] - float(step[i]) for i in range(3)]
    ok, Kb, Xb, _ = krawczyk_arb(fun, xt, [rad] * 3)
    from fractions import Fraction as Fr
    p0, q0, r = Fr(box_centre[0]), Fr(box_centre[1]), Fr(rho_in)
    in_box = all(Fr(float_down(lo_(Xb[k]))) >= c - r and Fr(float_up(hi_(Xb[k]))) <= c + r
                 for k, c in ((0, p0), (1, q0)))
    lo, hi = limit_solve_margin(Xb[0], Xb[1], Xb[2], Aa)
    sign = "solves" if bool(lo > 0) else ("fails" if bool(hi < 0) else None)
    return {"A": A, "krawczyk_unique_zero": bool(ok), "zero_box_in_rho_in_box": bool(in_box),
            "p": [float_down(lo_(Xb[0])), float_up(hi_(Xb[0]))], "q": [float_down(lo_(Xb[1])), float_up(hi_(Xb[1]))],
            "b": [float_down(lo_(Xb[2])), float_up(hi_(Xb[2]))],
            "margin": [float_down(lo), float_up(hi)], "sign": sign}


# ------------------------------------------------------------------------------------------ Track 4 driver
def _published_inputs():
    """The published parameters the checks take (read as data, as check_krawczyk reads first_order_c1.csv): the annulus
    centre (limit_switch.csv), the solve bracket and its centre (mn2_solve_limit.csv), the published margins."""
    import ast
    import csv
    R = CERTS.parent
    ls = next(csv.DictReader((R / "limit_switch.csv").open()))
    sl = sorted(csv.DictReader((R / "mn2_solve_limit.csv").open()), key=lambda r: float(r["A"]))
    e = ast.literal_eval(sl[-1]["encl"])
    return {"annulus_centre": (float(ls["argmin_p"]), float(ls["argmin_q"])),
            "annulus_edges": np.linspace(0.66, 0.71, 41).tolist(),
            "solve_bracket": (float(sl[0]["A_solve_lo"]), float(sl[0]["A_solve_hi"])),
            "solve_centre": (0.5 * (e[0] + e[1]), 0.5 * (e[2] + e[3])),
            "solve_rows": [{"A": float(r["A"]), "sign": r["certified_sign"], "encl": ast.literal_eval(r["encl"]),
                            "margin": [float(r["margin_lo"]), float(r["margin_hi"])]} for r in sl]}


def track4_outer(which, sub=None, verbose=True):
    """Outer exclusion (Link 2), independently, margin target 1e−4 (the published statement: >= 1.00e−4 on every annulus
    sub-interval, >= 1.48e−4 on the solve bracket).  which = 'solve': the solve bracket as one interval ->
    certificate_checks/outer_solve.json.  which = 'annulus': each of the published 40 sub-intervals of [0.66, 0.71]
    (checkpointed to outer_glob_annulus_parts.jsonl, skipped on restart) -> outer_glob_annulus.json; sub = k runs only
    the k-th sub-interval (the timing sample) -> outer_glob_annulus_sub<k>.json."""
    pub = _published_inputs()
    x, y, sym = _verified_limit_data()
    BP, _ = localisation_bounds(x, y, 24.0)
    out_dir = CERTS.parent / "certificate_checks"; out_dir.mkdir(exist_ok=True)

    def one(centre, A_lo, A_hi):
        r = outer_exclusion_arb(x, y, centre, A_lo, A_hi, 1e-4, verbose=verbose)
        r["localised_branch_below_B24"] = bool(arb(max(r["branch_upper"])) < BP)
        r["pass"] = r["verdict"] == "pass" and r["localised_branch_below_B24"]
        return r
    base = {"files_match_committed_hashes": True, "data_x_symmetric": bool(sym), "B24_lower": float_down(lo_(BP))}
    if which == "solve":
        (A_lo, A_hi) = pub["solve_bracket"]
        res = {"name": "outer_solve", **base, **one(pub["solve_centre"], A_lo, A_hi)}
        res["pass"] = res["pass"] and sym
        (out_dir / "outer_solve.json").write_text(json.dumps(res, indent=1, default=str))
    elif sub is not None:
        e = pub["annulus_edges"]
        res = {"name": f"outer_glob_annulus_sub{sub:02d}", **base, **one(pub["annulus_centre"], e[sub], e[sub + 1])}
        res["pass"] = res["pass"] and sym
        (out_dir / f"outer_glob_annulus_sub{sub:02d}.json").write_text(json.dumps(res, indent=1, default=str))
    else:
        import csv
        e = pub["annulus_edges"]
        parts = out_dir / "outer_glob_annulus_parts.jsonl"
        done = {}
        if parts.exists():
            for line in parts.read_text().splitlines():
                r = json.loads(line); done[r["k"]] = r
        for k in range(len(e) - 1):
            if k in done:
                continue
            r = {"k": k, **one(pub["annulus_centre"], e[k], e[k + 1])}
            with parts.open("a") as fh:
                fh.write(json.dumps(r, default=str) + "\n")
            done[k] = r
            if verbose:
                print(json.dumps({q: r[q] for q in ("k", "A_lo", "A_hi", "verdict", "min_margin_certified", "evals",
                                                     "seconds")}), flush=True)
        rows = [done[k] for k in range(len(e) - 1)]
        prod = {round(float(r["A_lo"]), 6): float(r["outer_margin"])
                for r in csv.DictReader((CERTS.parent / "certv2_annulus_parts.csv").open())}
        res = {"name": "outer_glob_annulus", **base, "centre": list(pub["annulus_centre"]),
               "range": [e[0], e[-1]], "subintervals": len(rows),
               "subintervals_pass": sum(bool(r["pass"]) for r in rows),
               "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in ("pass", "fail", "unresolved")},
               "min_margin_certified": min((r["min_margin_certified"] for r in rows if r["verdict"] == "pass"),
                                           default=None),
               "max_branch_upper": max(max(r["branch_upper"]) for r in rows),
               "evals_total": sum(r["evals"] for r in rows), "max_depth": max(r["max_depth"] for r in rows),
               "seconds_total": sum(r["seconds"] for r in rows),
               "per_subinterval": [{"A_lo": r["A_lo"], "A_hi": r["A_hi"], "verdict": r["verdict"],
                                    "margin": r["min_margin_certified"], "evals": r["evals"], "seconds": r["seconds"],
                                    "producer_margin": prod.get(round(float(r["A_lo"]), 6))} for r in rows]}
        res["pass"] = res["subintervals_pass"] == len(rows) == 40 and sym
        (out_dir / "outer_glob_annulus.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps({k: v for k, v in res.items() if k != "per_subinterval"}, indent=1, default=str))
    return res


def track4_solve_limit(verbose=True, workers=1):
    """The limit solve bracket's two ends, independently: for each end A, Krawczyk's unique zero of the (p, q, b)
    gradient, inside the 0.05 box around the published solve centre; the solve margin's sign there in Arb.  That zero is
    the global minimiser by the chain, whose links are re-run here with recorded parameters: localisation (B(24)),
    outer exclusion (outer_solve.json, run by track4_outer), ring 0.05–0.15 (_ring_interval) and PD on the 0.05 box
    (_pd_box, margin 0) over the whole bracket.  Writes certificate_checks/solve_limit_ends.json."""
    from multiprocessing import Pool
    t0 = time.time()
    pub = _published_inputs()
    x, y, sym = _verified_limit_data()
    (A_lo, A_hi), c = pub["solve_bracket"], pub["solve_centre"]
    res = {"name": "solve_limit_ends", "bracket": [A_lo, A_hi], "centre": list(c), "data_x_symmetric": bool(sym)}
    outer_p = CERTS.parent / "certificate_checks" / "outer_solve.json"
    res["outer_pass"] = bool(outer_p.exists() and json.loads(outer_p.read_text())["pass"])
    ring = _ring_interval((A_lo, A_hi, c[0], c[1], 0.05, 0.15, 0.0125, 4))
    res["ring"] = ring
    pe = np.linspace(c[0] - 0.05, c[0] + 0.05, 11); qe = np.linspace(c[1] - 0.05, c[1] + 0.05, 11)
    assert pe[0] <= c[0] - 0.05 and pe[-1] >= c[0] + 0.05 and qe[0] <= c[1] - 0.05 and qe[-1] >= c[1] + 0.05
    boxes = [(pe[i], pe[i + 1], qe[j], qe[j + 1], A_lo, A_hi, 0.0) for i in range(10) for j in range(10)]
    with Pool(workers) as pl:
        pdr = pl.map(_pd_box, boxes)
    res["pd"] = {"boxes": len(pdr), "all_pd": all(r["pd"] for r in pdr), "b_bracket_all": all(r["b_ok"] for r in pdr)}
    ends = []
    for row in pub["solve_rows"]:
        e = row["encl"]
        p_, q_ = 0.5 * (e[0] + e[1]), 0.5 * (e[2] + e[3])
        o = Objective(x, y, row["A"], None, act=HAct())
        Bb, _ = o.b_bracket(arb(p_), arb(q_))
        r = limit_solve_end(x, y, row["A"], (p_, q_, float(Bb.mid())), c)
        r["published_sign"] = row["sign"]; r["published_margin"] = row["margin"]
        r["sign_equals_published"] = r["sign"] == row["sign"]
        r["margin_overlaps_published"] = bool(r["margin"][0] <= row["margin"][1] and r["margin"][1] >= row["margin"][0])
        ends.append(r)
    res["ends"] = ends
    BP, _ = localisation_bounds(x, y, 24.0)
    ub = []
    for A in (A_lo, A_hi):
        o = Objective(x, y, A, None, act=HAct())
        Bb, _ = o.b_bracket(arb(c[0]), arb(c[1]))
        ub.append(arb(hi_(o.upper_at(c[0], c[1], float(Bb.mid())))))
    res["branch_upper"] = [float_up(u) for u in ub]
    res["localised_branch_below_B24"] = all(bool(u < BP) for u in ub)
    res["chain_links"] = {"localisation": res["localised_branch_below_B24"], "outer": res["outer_pass"],
                          "ring": bool(ring["certified"]), "pd": res["pd"]["all_pd"] and res["pd"]["b_bracket_all"]}
    res["ends_pass"] = all(r["krawczyk_unique_zero"] and r["zero_box_in_rho_in_box"] and r["sign_equals_published"]
                           for r in ends)
    res["pass"] = res["ends_pass"] and all(res["chain_links"].values()) and sym
    res["seconds"] = time.time() - t0
    out_dir = CERTS.parent / "certificate_checks"; out_dir.mkdir(exist_ok=True)
    (out_dir / "solve_limit_ends.json").write_text(json.dumps(res, indent=1, default=str))
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------ Track U, step U2: ε-interval units
# check_u2_unit (results/uniform_eps/U2_design.md §5, author-approved 2026-10-16).  Written for this checker only: it shares
# no code with the producer (src/u2_sample.py) and imports nothing from src/.  Everything is re-derived here in Arb.
#
# The rescaled feature at a = 1 + ε, u = √ε σ, written through EVEN entire kernels of v = u² = ε σ² (never through √ε, so
# every enclosure is valid on ε-balls that contain 0, where φ_0 = h):
#     φ    = −σ + (1+ε) σ³ Gt(v),          Gt = (u − sin u)/u³  = Σ_m (−1)^m v^m/(2m+3)!
#     φ_σ  = −1 + (1+ε) σ² J(v),           J  = (1 − cos u)/u²  = Σ_m (−1)^m v^m/(2m+2)!
#     φ_σσ = (1+ε) σ S(v),                 S  = sin u / u       = Σ_m (−1)^m v^m/(2m+1)!
#     φ_ε  = σ³ Gt(v) + (1+ε) σ⁵ K1(v),    K1 = Gt′(u)/(2u)     = Σ_m (−1)^{m+1}(m+1) v^m/(2m+5)!
#     φ_εε = 2σ⁵ K1(v) + (1+ε) σ⁷ K2(v),   K2 = K1′(u)/(2u)     = Σ_m (−1)^m (m+1)(m+2) v^m/(2m+7)!
# (derivation: φ = (u − (1+ε) sin u)/ε^{3/2}; ∂_ε u = u/(2ε); each series is the Taylor series of its closed form.)
# Series evaluation: |c_{m+1}|/|c_m| <= 3/((2m+2)(2m+3)) for all five kernels (the polynomial factors grow by at most 3),
# so once 3V/((2M+2)(2M+3)) <= 1/2 with V >= |v|, the tail after M terms is at most 2|c_M| V^M (a geometric majorant).
# The series is used for v < U2_V_SERIES (= 400, |u| < 20: Horner on a v-ball has mild dependency loss, and 80 bits carry
# the largest term, ~e^20); closed forms only where v is certainly >= U2_V_SERIES (there u − sin u etc. lose little to
# dependency); a v-ball straddling the threshold is split.
_U2_KERNELS = {
    "Gt": (lambda m: (-1) ** m, lambda m: 1, lambda m: 2 * m + 3),
    "J": (lambda m: (-1) ** m, lambda m: 1, lambda m: 2 * m + 2),
    "S": (lambda m: (-1) ** m, lambda m: 1, lambda m: 2 * m + 1),
    "K1": (lambda m: (-1) ** (m + 1), lambda m: m + 1, lambda m: 2 * m + 5),
    "K2": (lambda m: (-1) ** m, lambda m: (m + 1) * (m + 2), lambda m: 2 * m + 7),
    "JJ": (lambda m: (-1) ** (m + 1), lambda m: m + 1, lambda m: 2 * m + 4),      # J′(u)/(2u); for φ_σε
}


U2_V_SERIES = 400


def _ival(lo, hi):
    """The ball containing [lo, hi] for Arb (or float) endpoints, exactly."""
    lo, hi = arb(lo), arb(hi)
    return (lo + hi) / 2 + arb(0, ((hi - lo) / 2).upper())


def _sq_nonneg(t):
    """t² as a ball with a nonnegative lower end (t·t of a ball straddling 0 would include negatives)."""
    a, b = lo_(t), hi_(t)                            # exact endpoints, so every comparison below is decided
    m = abs(a) if bool(abs(a) >= abs(b)) else abs(b)
    if bool(a <= 0) and bool(b >= 0):
        mn = arb(0)
    else:
        mn = abs(a) if bool(abs(a) <= abs(b)) else abs(b)
    return _ival(mn * mn, m * m)


def _pow(v, k):
    """v**k by repeated products (python-flint's arb ** int is NaN on a ball containing 0)."""
    out = arb(1)
    for _ in range(k):
        out = out * v
    return out


def _kernel_series(name, v):
    sgn, P, k = _U2_KERNELS[name]
    a, b = abs(lo_(v)), abs(hi_(v))
    V = a if bool(a >= b) else b
    M = 4
    while not bool(3 * V <= arb((2 * M + 2) * (2 * M + 3)) / 2):
        M += 1
        if M > 5000:
            return arb("nan")
    eps_tail = arb(2) ** -100                        # then extend until the geometric tail is negligible
    while bool(2 * arb(P(M)) / arb.fac_ui(k(M)) * _pow(V, M) > eps_tail) and M < 5000:
        M += 1
    acc = arb(0)
    for m in reversed(range(M)):
        acc = acc * v + arb(sgn(m) * P(m)) / arb.fac_ui(k(m))
    tail = 2 * arb(P(M)) / arb.fac_ui(k(M)) * _pow(V, M)
    return acc + arb(0, tail.upper())


def _kernel_closed(name, v):
    u = v.sqrt()                                     # v certainly >= U2_V_SERIES here (or >= 1 in the tests)
    s, c = u.sin(), u.cos()
    if name == "Gt":
        return (u - s) / (u * u * u)
    if name == "J":
        return (1 - c) / (u * u)
    if name == "S":
        return s / u
    u3, u4, u5 = _pow(u, 3), _pow(u, 4), _pow(u, 5)
    if name == "JJ":
        return (s / (u * u) - 2 * (1 - c) / u3) / (2 * u)
    g1 = (1 - c) / u3 - 3 * (u - s) / u4                              # Gt′(u)
    if name == "K1":
        return g1 / (2 * u)
    g2 = s / u3 - 6 * (1 - c) / u4 + 12 * (u - s) / u5                # Gt″(u)
    return (g2 / (2 * u) - g1 / (2 * u * u)) / (2 * u)               # K2 = K1′/(2u), K1′ = Gt″/(2u) − Gt′/(2u²)


def u2_kernel(name, v):
    """Ball enclosing kernel(v′) for every v′ in the ball v (v′ = ε σ² >= 0 in use; series valid for any real v′)."""
    one = arb(U2_V_SERIES)
    if not v.is_finite():
        return arb("nan")
    if bool(v >= one):
        return _kernel_closed(name, v)
    if bool(v < one):
        return _kernel_series(name, v)
    a = _kernel_series(name, _ival(lo_(v), U2_V_SERIES))      # straddles the threshold: split it
    b = _kernel_closed(name, _ival(U2_V_SERIES, hi_(v)))
    return _ival(min(lo_(a), lo_(b)), max(hi_(a), hi_(b)))


class U2Act:
    """φ_ε(σ) − c (and its derivatives) for every ε in the ball E (E may contain 0), the interface of FA/HAct.
    c: a per-cell constant (exact double), the HShift idea: the logit A(φ − c) + b′ is the same profiled model."""

    def __init__(self, E, c=0.0):
        self.E = arb(E)
        self.cc = arb(c)
        self.cf = float(c)
        self.ef = float(self.E.mid())
        self.twopi = None

    def _v(self, t):
        return self.E * _sq_nonneg(t)

    def f(self, t):
        v = self._v(t)
        return -t + (1 + self.E) * t * _sq_nonneg(t) * u2_kernel("Gt", v) - self.cc

    def df(self, t):
        return -1 + (1 + self.E) * _sq_nonneg(t) * u2_kernel("J", self._v(t))

    def d2f(self, t):
        return (1 + self.E) * t * u2_kernel("S", self._v(t))

    def dse(self, t):
        """φ_σε = σ² J(v) + (1+ε) σ⁴ JJ(v)."""
        t2 = _sq_nonneg(t); v = self.E * t2
        return t2 * u2_kernel("J", v) + (1 + self.E) * t2 * t2 * u2_kernel("JJ", v)

    def de(self, t):
        t2 = _sq_nonneg(t); v = self.E * t2
        return t * t2 * u2_kernel("Gt", v) + (1 + self.E) * t * t2 * t2 * u2_kernel("K1", v)

    def dee(self, t):
        t2 = _sq_nonneg(t); v = self.E * t2
        return 2 * t * t2 * t2 * u2_kernel("K1", v) + (1 + self.E) * t * t2 * t2 * t2 * u2_kernel("K2", v)

    def np_f(self, t):
        """Float φ at mid ε (heuristics only: proposes points, never decides)."""
        e = max(self.ef, 0.0)
        if e < 1e-12:
            return -t + t ** 3 / 6 - self.cf
        u = math.sqrt(e) * t
        return (u - (1 + e) * np.sin(u)) / e ** 1.5 - self.cf

    def range(self, tlo, thi):
        """Rigorous [min, max] over [tlo, thi] for every ε in E: the ball evaluation on the whole interval."""
        v = self.f(_ival(tlo, thi))
        if not v.is_finite():
            return arb("-inf"), arb("inf")
        return lo_(v), hi_(v)


def u2_slabs_closed_ok(slabs):
    """F2: the ε-slabs are closed intervals [e_k, e_{k+1}] that share endpoints exactly (floats compared exactly)."""
    if not slabs:
        return False
    for i, (lo, hi) in enumerate(slabs):
        if not (lo < hi):
            return False
        if i and slabs[i - 1][1] != lo:
            return False
    return True


def _u2_obj(x, y, A, E, c=0.0):
    return Objective(x, y, arb(A), None, act=U2Act(E, c))


def _u2_grad_eps(obj, W, B, Bb):
    """Ball of ∂_ε L* = A·mean((σ(z) − y)·φ_ε) over the cell W × B with b* ∈ Bb (envelope theorem)."""
    acc = arb(0)
    for xi, yi in zip(obj.X, obj.Y):
        t = W * xi + B
        z = obj.s * obj.fa.f(t) + Bb
        acc += (1 / (1 + (-z).exp()) - yi) * obj.fa.de(t)
    return obj.s * acc / obj.n


def _u2_curv_eps(obj, W, B, A_hi, cshift):
    """Upper bound of |∂²_ε L*| over the cell and the obj's (A, E) balls: ¼A²·mean((φ_ε − c)²) + A·mean|φ_εε| (the
    profiled second derivative is A²·(a weighted variance of φ_ε, weights σ′ <= ¼) + A·mean((σ − y) φ_εε))."""
    s1 = arb(0); s2 = arb(0)
    for xi in obj.X:
        t = W * xi + B
        s1 += _sq_nonneg(obj.fa.de(t) - arb(cshift))
        d = obj.fa.dee(t)
        s2 += max(abs(lo_(d)), abs(hi_(d)))
    A = arb(A_hi)
    return hi_(A * A * s1 / (4 * obj.n) + A * s2 / obj.n)


def _absmax(b):
    """An upper bound of |v| over the ball b (an exact Arb endpoint)."""
    a, c = abs(lo_(b)), abs(hi_(b))
    return a if bool(a >= c) else c


def _u2_point_grads(xf, yf, P, Q, A, act, cA=0.0, cE=0.0):
    """At the point (P, Q) (balls), A (ball) and every ε in act.E: an Arb bracket of b* and balls of ∂_pL*, ∂_qL*,
    ∂_AL*, ∂_εL* by the envelope theorem.  ∂_A L* = mean((σ − y)(φ − cA)) and ∂_ε L* = A·mean((σ − y)(φ_ε − cE)) for
    any constants cA, cE, because mean(σ(z) − y) = 0 at b*.  Returns (Bb, [gp, gq, gA, gE]) or (None, None)."""
    obj = Objective(xf, yf, A, None, act=act)
    Bb, _ = obj.b_bracket(P, Q)
    if Bb is None:
        return None, None
    g = [arb(0)] * 4
    cA, cE = arb(repr(float(cA))), arb(repr(float(cE)))
    for xi, yi in zip(obj.X, obj.Y):
        sg = P * xi + Q
        f0 = act.f(sg)
        z = A * f0 + Bb
        res = 1 / (1 + (-z).exp()) - yi
        f1 = act.df(sg)
        g[0] += res * A * f1 * xi
        g[1] += res * A * f1
        g[2] += res * (f0 - cA)
        g[3] += res * A * (act.de(sg) - cE)
    return Bb, [v / obj.n for v in g]


def u2_coupled_lower(xf, yf, pc, qc, hp, hq, A_lo, A_hi, e_lo, e_hi, pb, qb, xs=None, ys=None):
    """The coupled second-order bound of math note §23 (Proposition 23.3): for every (p, q) in the cell [pc ± hp] ×
    [qc ± hq], A in [A_lo, A_hi] and ε in [e_lo, e_hi],
        D := L*_ε(p, q; A) − L*_ε(pb, qb; A)
          >= D(v_c) − |∂_pD|h_p − |∂_qD|h_q − |∂_AD|h_A − |∂_εD|h_ε − ½(M_comp + M_br)
    with everything at the centre v_c = (pc, qc, A_c, ε_c) enclosed in Arb at that point, and the remainders
      M_comp = mean_i sup_B Σ_ab |∂_ab z̃_i| h_a h_b  (z̃ = A(φ − φ(q_c; ε)) + b′, Lemma 23.2(i)),
      M_br   = ¼ mean_i sup (h_A|φ_i − c1| + h_ε A_hi|φ_ε,i − c2|)² + mean_i sup (2h_A h_ε|φ_ε,i| + h_ε² A_hi|φ_εε,i|)
               (Lemma 23.2(ii), at the branch point),
    every sup a ball evaluation over the cell's σ-balls, the A-ball and the ε-ball (U2Act; no envelope).
    Returns an Arb lower bound of D over the box (an exact endpoint), or None if a bracket fails."""
    xs = xs if xs is not None else [arb(float(v)) for v in xf]
    ys = ys if ys is not None else [int(v) for v in yf]
    n = len(xs)
    Ac, ec = 0.5 * (A_lo + A_hi), 0.5 * (e_lo + e_hi)
    hA, he = 0.5 * (A_hi - A_lo), 0.5 * (e_hi - e_lo)
    Apt, Ept, Ab, Eb = arb(repr(float(Ac))), arb(repr(float(ec))), _ival(A_lo, A_hi), _ival(e_lo, e_hi)
    sc = pc * np.asarray(xf, float) + qc
    sb = pb * np.asarray(xf, float) + qb
    shift_c = float(np.median(U2Act(Ept).np_f(sc)))
    shift_b = float(np.median(U2Act(Ept).np_f(sb)))
    # centre values: competitor lower bound, branch upper bound, gradients (Lemma 23.1)
    oc = Objective(xf, yf, Apt, None, act=U2Act(Ept, shift_c))
    Lc = oc.lower_bound(pc, qc, 0.0, 0.0)
    ob = Objective(xf, yf, Apt, None, act=U2Act(Ept, shift_b))
    Bbr, _ = ob.b_bracket(arb(repr(float(pb))), arb(repr(float(qb))))
    if Lc is None or Bbr is None:
        return None
    Lb = arb(hi_(ob.upper_at(pb, qb, float(Bbr.mid()))))
    actp = U2Act(Ept)
    fe_c = [float(v.mid()) for v in (actp.de(arb(repr(float(t)))) for t in sc[::40])]
    cEc = float(np.median(fe_c))
    fe_b = [float(v.mid()) for v in (actp.de(arb(repr(float(t)))) for t in sb[::40])]
    cEb = float(np.median(fe_b))
    _, gc = _u2_point_grads(xf, yf, arb(repr(float(pc))), arb(repr(float(qc))), Apt, actp, shift_c, cEc)
    _, gb = _u2_point_grads(xf, yf, arb(repr(float(pb))), arb(repr(float(qb))), Apt, actp, shift_b, cEb)
    if gc is None or gb is None:
        return None
    first = _absmax(gc[0]) * arb(hp) + _absmax(gc[1]) * arb(hq) + _absmax(gc[2] - gb[2]) * arb(hA) \
        + _absmax(gc[3] - gb[3]) * arb(he)
    # competitor remainder, Lemma 23.2(i), with the per-cell shift c(ε) = φ(q_c; ε)
    actE = U2Act(Eb)
    Qc = arb(repr(float(qc)))
    fe_q, fee_q = actE.de(Qc), actE.dee(Qc)
    P, Q = _ival(pc - hp, pc + hp), _ival(qc - hq, qc + hq)
    Mc = arb(0)
    AhB = arb(repr(float(A_hi)))
    for xi in xs:
        sg = P * xi + Q
        d = abs(xi) * arb(hp) + arb(hq)
        t = AhB * _absmax(actE.d2f(sg)) * d * d + 2 * _absmax(actE.df(sg)) * d * arb(hA) \
            + 2 * AhB * _absmax(actE.dse(sg)) * d * arb(he) + 2 * _absmax(actE.de(sg) - fe_q) * arb(hA) * arb(he) \
            + AhB * _absmax(actE.dee(sg) - fee_q) * arb(he) * arb(he)
        Mc += t
    Mc = Mc / n
    # branch remainder, Lemma 23.2(ii)
    Pb, Qb = arb(repr(float(pb))), arb(repr(float(qb)))
    c1 = shift_b
    Mb1 = arb(0); Mb2 = arb(0)
    for xi in xs:
        sg = Pb * xi + Qb
        u = arb(hA) * _absmax(actE.f(sg) - arb(repr(c1))) + arb(he) * AhB * _absmax(actE.de(sg) - arb(repr(cEb)))
        Mb1 += u * u
        Mb2 += 2 * arb(hA) * arb(he) * _absmax(actE.de(sg)) + arb(he) * arb(he) * AhB * _absmax(actE.dee(sg))
    Mb = Mb1 / (4 * n) + Mb2 / n
    D = arb(lo_(Lc)) - Lb - first - (Mc + Mb) / 2
    return lo_(D)


def u2_ring_fast_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_in=0.05, rho_out=0.15, side=0.0125, boxes=None,
                     fast_depth=2):
    """Ring with the fast direct test first (u2_ring_arb: the profiled gradient's Arb ball over box × A × ε, split up to
    fast_depth) and the first-order 3-D form (u2_ring3d_arb) only for the tiles the fast test leaves open."""
    t0 = time.time()
    if boxes is None:
        n = int(round(2 * rho_out / side))
        g = -rho_out + (np.arange(n) + 0.5) * side
        boxes = [(float(a), float(b), side / 2) for a in g for b in g
                 if max(abs(a), abs(b)) + side / 2 > rho_in + 1e-12]
    n_fast, n_slow, left = 0, 0, []
    for bx in boxes:
        r = u2_ring_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_in=rho_in, rho_out=rho_out, boxes=[bx],
                        max_depth=fast_depth)
        if r["certified"]:
            n_fast += 1
            continue
        r3 = u2_ring3d_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_in=rho_in, rho_out=rho_out, boxes=[bx])
        n_slow += 1
        if not r3["certified"]:
            left.append(bx)
    return {"certified": not left, "tiles": len(boxes), "fast": n_fast, "fallback": n_slow, "left": len(left),
            "first_left": left[:3], "seconds": time.time() - t0}


def u2_outer_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, regime, P=24.0, h0=1.0, rho=0.15, max_depth=10,
                 max_evals=200_000, region=None, n_eps=1, B_table=None, B_full=None, target=0.0, witness_from_depth=2,
                 eps_split_from_depth=4, max_eps_depth=8, use_coupled=True):
    """Re-search of the outer exclusion of one U2 unit, in Arb: for every θ in the regime's domain outside the closed rho
    box around θ_c, every A in [A_lo, A_hi] and every ε in [e_lo, e_hi]:  L*_ε(θ; A) − L*_ε(θ_c; A) > target.
    The ε-interval is split into n_eps closed pieces; on each piece E = [ε_c ± h_ε] a cell passes by
      (direct)  the cell's Arb lower bound with ε as a ball > the branch's Arb upper bound with ε as a ball, or
      (taylor)  D(ε_c) − h_ε·|∂_ε D(ε_c)| − ½ h_ε²·sup|∂²_ε D| > target, with D(ε_c) from the point-ε cell lower bound and
                the convexity-in-A branch bound, ∂_ε D enclosed at ε_c on the cell and at θ_c, sup|∂²_ε D| from
                _u2_curv_eps on both;
      (iso / full, regime R only, Lemma G2, exact PAVA bounds B(P), B_full passed in as Arb balls).
    A cell that fails is split in ε when its point-ε_c bound already exceeds the branch (only the ε-terms fail; each
    cell carries its own closed ε-piece, up to max_eps_depth halvings), and in θ otherwise.
    Domain: R: K(P) ∩ {p >= 0} (Lemma G2, ε <= 0.0462); O: 0 <= p <= W(s_min, a_max)/√e_lo, |q| <= π/√e_lo (§5.0 lemma,
    one b₁-period for every ε in the slab).  Verdicts 'pass' / 'fail' (a witness point that beats the branch) / 'unresolved'."""
    ctx.prec = PREC
    t0 = time.time()
    xf, yf = np.asarray(x, float), np.asarray(y, float)
    p0, q0 = float(centre[0]), float(centre[1])
    assert 0 < A_lo < A_hi and 0 <= e_lo < e_hi
    Ab = _ival(A_lo, A_hi)
    if regime == "R":
        assert e_hi <= 0.0462
        pl0, ph0, ql0, qh0 = region if region is not None else (0.0, P, -52.0, 52.0)
    else:
        assert e_lo > 0
        W = lemma_W(xf, yf, A_lo / e_hi ** 1.5, 1 + e_hi)
        pmax = W / math.sqrt(e_lo) * (1 + 1e-12) + 1e-12
        qmax = math.pi / math.sqrt(e_lo) * (1 + 1e-12) + 1e-12
        n_p, n_q = math.ceil(pmax / h0), math.ceil(qmax / h0)
        pl0, ph0, ql0, qh0 = region if region is not None else (0.0, n_p * h0, -n_q * h0, n_q * h0)
        assert bool(arb(ph0) * arb(e_lo).sqrt() >= arb(W)) or region is not None
        assert bool(arb(qh0) * arb(e_lo).sqrt() >= arb.pi()) or region is not None
    edges = [e_lo + (e_hi - e_lo) * i / n_eps for i in range(n_eps)] + [e_hi]
    stats = {"evals": 0, "direct": 0, "taylor": 0, "iso": 0, "full": 0, "dropped_in_box": 0, "dropped_outside_K": 0,
             "max_depth": 0, "witness_tries": 0, "eps_splits": 0, "max_eps_depth": 0, "coupled": 0}
    witness, unresolved, marg = None, [], math.inf
    cache = {}
    xs_arb = [arb(float(v)) for v in xf]; ys_int = [int(v) for v in yf]

    def branch(el, eh):
        """Branch-side Arb quantities on the closed ε-piece [el, eh] (cached)."""
        key = (el, eh)
        if key in cache:
            return cache[key]
        ec = 0.5 * (el + eh)
        Ept, Eb = arb(repr(float(ec))), _ival(el, eh)
        U_pt, U_E, L_pt = [], [], []
        for A_k in (A_lo, A_hi):
            for EE, store in ((Ept, U_pt), (Eb, U_E)):
                o = _u2_obj(xf, yf, A_k, EE)
                Bb, _ = o.b_bracket(arb(p0), arb(q0))
                if Bb is None:
                    raise RuntimeError("branch bias not bracketed")
                store.append(arb(hi_(o.upper_at(p0, q0, float(Bb.mid())))))
            o = _u2_obj(xf, yf, A_k, Ept)
            L_pt.append(arb(lo_(o.lower_bound(p0, q0, 0.0, 0.0))))
        ob = Objective(xf, yf, Ab, None, act=U2Act(Ept))
        Bb_c, _ = ob.b_bracket(arb(p0), arb(q0))
        if Bb_c is None:
            raise RuntimeError("branch bias not bracketed (A ball)")
        G_b = _u2_grad_eps(ob, arb(p0), arb(q0), Bb_c)
        ob_E = Objective(xf, yf, Ab, None, act=U2Act(Eb))
        scb = p0 * xf + q0
        cbe = float(np.median(np.array([float(ob_E.fa.de(arb(float(sv))).mid()) for sv in scb])))
        C_b = _u2_curv_eps(ob_E, arb(p0), arb(q0), A_hi, cbe)
        br_pt, br_E = max(U_pt[0], U_pt[1]), max(U_E[0], U_E[1])
        iso_P = None
        if regime == "R" and B_table:
            okP = [Pk for Pk, Bk in B_table if bool(Bk > br_E + arb(target))]
            iso_P = min(okP) if okP else None
        full_ok = regime == "R" and B_full is not None and bool(B_full > br_E + arb(target))
        cache[key] = {"ec": ec, "he": 0.5 * (eh - el), "Ept": Ept, "Eb": Eb, "L_pt": L_pt, "G_b": G_b, "C_b": C_b,
                      "br_pt": br_pt, "br_E": br_E, "iso_P": iso_P, "full_ok": full_ok}
        return cache[key]

    try:
        for ie in range(n_eps):
            branch(edges[ie], edges[ie + 1])
    except RuntimeError as ex:
        return {"verdict": "unresolved", "note": str(ex)}
    n_p, n_q = int(round((ph0 - pl0) / h0)), int(round((qh0 - ql0) / h0))
    stack = [(pl0 + a * h0, pl0 + (a + 1) * h0, ql0 + b * h0, ql0 + (b + 1) * h0, 0, edges[ie], edges[ie + 1], 0)
             for ie in range(n_eps) for a in range(n_p) for b in range(n_q)]
    r8 = 2 * arb(2).sqrt()
    while stack:
        pl, ph, ql, qh, depth, el, eh, ed = stack.pop()
        if _inside_closed_box(pl, ph, ql, qh, p0, q0, rho):
            stats["dropped_in_box"] += 1; continue
        if regime == "R" and _disjoint_from_K(pl, ph, ql, qh):
            stats["dropped_outside_K"] += 1; continue
        try:
            br = branch(el, eh)
        except RuntimeError as ex:
            unresolved.append((pl, ph, ql, qh, el, eh, str(ex))); continue
        if br["iso_P"] is not None and pl >= br["iso_P"]:
            stats["iso"] += 1; continue
        pc, qc, hp, hq = 0.5 * (pl + ph), 0.5 * (ql + qh), 0.5 * (ph - pl), 0.5 * (qh - ql)
        if br["full_ok"]:                                 # every point's σ-range certainly outside (−2√2, 2√2)
            def _out8(xi):
                lo = arb(pl) * arb(xi) if xi >= 0 else arb(ph) * arb(xi)
                hi = arb(ph) * arb(xi) if xi >= 0 else arb(pl) * arb(xi)
                return bool(lo + arb(ql) >= r8) or bool(hi + arb(qh) <= -r8)
            if all(_out8(float(xi)) for xi in xf):
                stats["full"] += 1; continue
        if stats["evals"] >= max_evals:
            unresolved.append((pl, ph, ql, qh, el, eh, "evaluation cap")); continue
        stats["evals"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        stats["max_eps_depth"] = max(stats["max_eps_depth"], ed)
        sc = pc * xf + qc
        ok = False
        if use_coupled:                                   # Proposition 23.3 (math note §23)
            el_, eh_ = el, eh
            cl = u2_coupled_lower(xf, yf, pc, qc, hp, hq, A_lo, A_hi, el_, eh_, p0, q0, xs=xs_arb, ys=ys_int)
            if cl is not None and bool(cl > arb(target)):
                stats["coupled"] += 1; ok = True
                marg = min(marg, float_down(cl))
        cE = U2Act(br["Eb"]).np_f(sc)
        o = Objective(xf, yf, Ab, None, act=U2Act(br["Eb"], float(np.median(cE))))
        lb = o.lower_bound(pc, qc, hp, hq) if not ok else None
        if lb is not None and bool(lb > br["br_E"] + arb(target)):
            stats["direct"] += 1; ok = True
            marg = min(marg, float_down(lo_(lb) - hi_(br["br_E"])))
        point_ok = False                                  # the θ-cell bound alone suffices at ε_c (then split ε)
        if not ok:
            cP = U2Act(br["Ept"]).np_f(sc)
            o = Objective(xf, yf, Ab, None, act=U2Act(br["Ept"], float(np.median(cP))))
            lbp = o.lower_bound(pc, qc, hp, hq)
            point_ok = lbp is not None and bool(lbp > br["br_pt"] + arb(target))
            if point_ok:
                Wc, Bc = _ival(pl, ph), _ival(ql, qh)
                Bcell, _ = o.b_bracket(Wc, Bc)
                if Bcell is not None:
                    G_c = _u2_grad_eps(o, Wc, Bc, Bcell)
                    oE = Objective(xf, yf, Ab, None, act=U2Act(br["Eb"]))
                    cse = float(np.median(np.array([float(oE.fa.de(arb(float(sv))).mid()) for sv in sc])))
                    C_c = _u2_curv_eps(oE, Wc, Bc, A_hi, cse)
                    dG = G_c - br["G_b"]
                    he = br["he"]
                    bound = lbp - br["br_pt"] - arb(he) * max(abs(lo_(dG)), abs(hi_(dG))) \
                        - arb(he) ** 2 / 2 * (C_c + br["C_b"])
                    if bool(bound > arb(target)):
                        stats["taylor"] += 1; ok = True
                        marg = min(marg, float_down(lo_(bound)))
        if ok:
            continue
        if depth >= witness_from_depth and _point_outside_box(pc, qc, p0, q0, rho):
            for k, A_k in enumerate((A_lo, A_hi)):
                act = U2Act(br["Ept"])
                z0 = A_k * act.np_f(pc * xf + qc)
                z0 = z0 - np.median(z0)
                lo, hi = -float(np.abs(z0).max()) - 50.0, float(np.abs(z0).max()) + 50.0
                for _ in range(200):
                    mm = 0.5 * (lo + hi)
                    vv = np.exp(-np.logaddexp(0, -(z0 + mm))).mean() - yf.mean()
                    lo, hi = (mm, hi) if vv < 0 else (lo, mm)
                zz = z0 + 0.5 * (lo + hi)
                if float((np.logaddexp(0, zz) - yf * zz).mean()) < float(br["L_pt"][k].mid()) + 1e-6:
                    stats["witness_tries"] += 1
                    ow = _u2_obj(xf, yf, A_k, br["Ept"])
                    Bw, _ = ow.b_bracket(arb(pc), arb(qc))
                    if Bw is not None:
                        up = arb(hi_(ow.upper_at(pc, qc, float(Bw.mid()))))
                        if bool(up < br["L_pt"][k] + arb(target)):
                            witness = {"p": pc, "q": qc, "A": A_k, "eps": br["ec"], "L_upper": float_up(up),
                                       "branch_lower": float_down(lo_(br["L_pt"][k]))}
                            break
            if witness is not None:
                break
        # split: ε when the θ-cell bound already holds at ε_c (only the ε-terms fail), else θ
        if point_ok and ed < max_eps_depth:
            em = 0.5 * (el + eh)
            stack += [(pl, ph, ql, qh, depth, el, em, ed + 1), (pl, ph, ql, qh, depth, em, eh, ed + 1)]
            stats["eps_splits"] += 1
        elif depth < max_depth:
            stack += [(a0, a1, b0, b1, depth + 1, el, eh, ed) for a0, a1 in ((pl, pc), (pc, ph)) for b0, b1 in ((ql, qc), (qc, qh))]
        else:
            unresolved.append((pl, ph, ql, qh, el, eh, "depth cap"))
    verdict = "fail" if witness is not None else ("pass" if not unresolved else "unresolved")
    return {"verdict": verdict, "witness": witness, "unresolved_n": len(unresolved), "unresolved_first": unresolved[:10],
            **stats, "min_margin_certified": marg if verdict == "pass" else None, "eps": [e_lo, e_hi], "n_eps": n_eps,
            "A": [A_lo, A_hi], "regime": regime, "centre": [p0, q0], "rho": rho, "seconds": time.time() - t0}


def u2_ring_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_in=0.05, rho_out=0.15, side=0.0125, max_depth=5, n_eps=1,
                boxes=None):
    """No critical point of L*_ε(·; A) in the ring rho_in <= |θ − θ_c|_∞ <= rho_out, for every A in [A_lo, A_hi] and ε in
    [e_lo, e_hi]: every box (×A ball × ε ball) has b* bracketed and 0 ∉ ∂_p L* or 0 ∉ ∂_q L* (Arb); split up to max_depth.
    boxes: optional list of (cp, cq, half) offsets from θ_c (tests / replay); default the full ring tiling."""
    ctx.prec = PREC
    t0 = time.time()
    xf, yf = np.asarray(x, float), np.asarray(y, float)
    p0, q0 = float(centre[0]), float(centre[1])
    pad = 1e-12
    if boxes is None:
        n = int(round(2 * rho_out / side))
        boxes = [(-rho_out + (i + 0.5) * side, -rho_out + (j + 0.5) * side, side / 2) for i in range(n) for j in range(n)]
    edges = [e_lo + (e_hi - e_lo) * i / n_eps for i in range(n_eps)] + [e_hi]
    n_box, left, gmin, dmax = 0, [], math.inf, 0
    for ie in range(n_eps):
        obj = Objective(xf, yf, _ival(A_lo, A_hi), None, act=U2Act(_ival(edges[ie], edges[ie + 1])))
        cells, depth = list(boxes), 0
        while cells and depth <= max_depth:
            nxt = []
            for cp, cq, hh in cells:
                if abs(cp) + hh <= rho_in - pad and abs(cq) + hh <= rho_in - pad:
                    continue
                n_box += 1
                Pb = _ival(p0 + cp - hh - pad, p0 + cp + hh + pad); Qb = _ival(q0 + cq - hh - pad, q0 + cq + hh + pad)
                Bb, Z = obj.b_bracket(Pb, Qb)
                ok = False
                if Bb is not None:
                    _, gp, gq = obj.value_and_grad(Pb, Qb, Bb, Z)
                    ok = (not gp.contains(0)) or (not gq.contains(0))
                    if ok:
                        mag = max(0 if gp.contains(0) else float(min(abs(lo_(gp)), abs(hi_(gp))).mid()),
                                  0 if gq.contains(0) else float(min(abs(lo_(gq)), abs(hi_(gq))).mid()))
                        gmin = min(gmin, mag)
                if not ok:
                    h2 = hh / 2
                    nxt += [(cp + dx * h2, cq + dy * h2, h2) for dx in (-1, 1) for dy in (-1, 1)]
            cells, depth = nxt, depth + 1
            dmax = max(dmax, depth - 1)
        left += cells
    return {"certified": not left, "boxes": n_box, "left": len(left), "grad_lower": gmin, "max_depth": dmax,
            "eps": [e_lo, e_hi], "A": [A_lo, A_hi], "seconds": time.time() - t0}


def _u2_F3(xs, ys, P, Q, Bb, A, act, want_de=False):
    """Balls of F = (∂_p L, ∂_q L, ∂_b L), L = mean softplus(z) − y z, z = A φ(p x + q) + b, over P × Q × Bb × A and
    every ε in act.E; with want_de also ∂_ε F = mean(σ′(z)·A φ_ε·v + (σ(z) − y)·∂_ε v), v = (A φ_σ x, A φ_σ, 1)."""
    n = len(xs)
    F = [arb(0)] * 3; D = [arb(0)] * 3
    for xi, yi in zip(xs, ys):
        sg = P * xi + Q
        z = A * act.f(sg) + Bb
        S = 1 / (1 + (-z).exp()); res = S - yi
        f1 = act.df(sg)
        v = [A * f1 * xi, A * f1, arb(1)]
        for k in range(3):
            F[k] += res * v[k]
        if want_de:
            w = S * (1 - S)
            fe, fse = act.de(sg), act.dse(sg)
            dv = [A * fse * xi, A * fse, arb(0)]
            for k in range(3):
                D[k] += w * A * fe * v[k] + res * dv[k]
    return [f / n for f in F], [d / n for d in D]


def u2_ring3d_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_in=0.05, rho_out=0.15, side=0.0125, nb=8, max_depth=3,
                  boxes=None, max_evals=200_000):
    """Ring, first-order (mean-value) form in ε, in Arb: no zero of F = ∇_{(p,q,b)} L with rho_in <= |θ − θ_c|_∞ <=
    rho_out, b in the Arb bracket of b* over the ring hull (every A, ε), A in [A_lo, A_hi], ε in E = [e_lo, e_hi].
    For a box P × Q × Bs:  F(θ, b, A, ε) ∈ F(P, Q, Bs, A, ε_c) + [−h, h]·∂_ε F(P, Q, Bs, A, E)  (the mean-value theorem
    in ε at fixed (θ, b, A)); the box is excluded when 0 lies outside one component, else split in (p, q, b).
    boxes: optional (cp, cq, half) offsets from θ_c (tests / timing on selected boxes)."""
    ctx.prec = PREC
    t0 = time.time()
    xf, yf = np.asarray(x, float), np.asarray(y, float)
    xs = [arb(float(v)) for v in xf]; ys = [int(v) for v in yf]
    p0, q0 = float(centre[0]), float(centre[1])
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    A = _ival(A_lo, A_hi)
    act_c, act_E = U2Act(arb(ec)), U2Act(_ival(e_lo, e_hi))
    obj = Objective(xf, yf, A, None, act=act_E)
    hull = _ival(p0 - rho_out, p0 + rho_out), _ival(q0 - rho_out, q0 + rho_out)
    Bh, _ = obj.b_bracket(*hull)
    if Bh is None:
        return {"certified": False, "note": "b* not bracketed over the ring hull"}
    blo, bhi = float_down(lo_(Bh)), float_up(hi_(Bh))
    if boxes is None:
        n = int(round(2 * rho_out / side))
        boxes = [(-rho_out + (i + 0.5) * side, -rho_out + (j + 0.5) * side, side / 2) for i in range(n) for j in range(n)]
    hb = (bhi - blo) / (2 * nb)
    stack = [(cp, cq, hh, blo + (2 * k + 1) * hb, hb, 0) for cp, cq, hh in boxes for k in range(nb)]
    evals, left, dmax = 0, [], 0
    Eh = arb(0, he)
    while stack:
        cp, cq, hh, bc, hbb, d = stack.pop()
        if abs(cp) + hh <= rho_in - 1e-12 and abs(cq) + hh <= rho_in - 1e-12:
            continue
        evals += 1
        if evals > max_evals:
            left.append((cp, cq, bc, "evaluation cap")); break
        P = _ival(p0 + cp - hh, p0 + cp + hh); Q = _ival(q0 + cq - hh, q0 + cq + hh); Bs = _ival(bc - hbb, bc + hbb)
        Fc, _ = _u2_F3(xs, ys, P, Q, Bs, A, act_c)
        _, D = _u2_F3(xs, ys, P, Q, Bs, A, act_E, want_de=True)
        ok = any(not (Fc[k] + Eh * D[k]).contains(0) for k in range(3))
        if ok:
            continue
        if d >= max_depth:
            left.append((cp, cq, bc)); continue
        dmax = max(dmax, d + 1)
        h2, hb2 = hh / 2, hbb / 2
        stack += [(cp + sp * h2, cq + sq * h2, h2, bc + sb * hb2, hb2, d + 1)
                  for sp in (-1, 1) for sq in (-1, 1) for sb in (-1, 1)]
    return {"certified": not left, "evals": evals, "left": len(left), "first_left": left[:3], "max_depth": dmax,
            "b_bracket": [blo, bhi], "eps": [e_lo, e_hi], "A": [A_lo, A_hi], "seconds": time.time() - t0}


def u2_pd_box_arb(x, y, pl, ph, ql, qh, A_lo, A_hi, e_lo, e_hi, c=0.0):
    """PD of the (p, q, b) Hessian of L (minus c·I) over box × A ball × ε ball × b* bracket (Sylvester on the Arb
    enclosure)."""
    ctx.prec = PREC
    obj = Objective(np.asarray(x, float), np.asarray(y, float), _ival(A_lo, A_hi), None, act=U2Act(_ival(e_lo, e_hi)))
    Pb, Qb, A = _ival(pl, ph), _ival(ql, qh), _ival(A_lo, A_hi)
    Bb, _ = obj.b_bracket(Pb, Qb)
    if Bb is None:
        return {"b_ok": False, "pd": False}
    H = [[arb(0)] * 3 for _ in range(3)]
    for xi, yi in zip(obj.X, obj.Y):
        sg = Pb * xi + Qb
        z = A * obj.fa.f(sg) + Bb
        S = 1 / (1 + (-z).exp()); w = S * (1 - S); res = S - yi
        f1, f2 = obj.fa.df(sg), obj.fa.d2f(sg)
        dz = [A * f1 * xi, A * f1, arb(1)]
        d2 = {(0, 0): A * f2 * xi * xi, (0, 1): A * f2 * xi, (1, 1): A * f2}
        for i in range(3):
            for j in range(3):
                H[i][j] += w * dz[i] * dz[j] + res * d2.get((min(i, j), max(i, j)), arb(0))
    n = obj.n
    M = [[H[i][j] / n - (arb(repr(c)) if i == j else 0) for j in range(3)] for i in range(3)]
    m1 = M[0][0]
    m2 = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    m3 = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
          + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    return {"b_ok": True, "pd": bool(m1 > 0 and m2 > 0 and m3 > 0)}


def u2_pd_arb(x, y, centre, A_lo, A_hi, e_lo, e_hi, rho_u=0.05, split=4, max_depth=4, c=0.0):
    """PD on the rho_u box around θ_c for every A, ε in the unit: split × split sub-boxes, each split in four up to
    max_depth if Sylvester fails."""
    t0 = time.time()
    p0, q0 = float(centre[0]), float(centre[1])
    h = rho_u / split
    cells = [(p0 - rho_u + (2 * i + 1) * h, q0 - rho_u + (2 * j + 1) * h, h, 0) for i in range(split) for j in range(split)]
    n_box, failed = 0, []
    while cells:
        pc, qc, hh, d = cells.pop()
        n_box += 1
        r = u2_pd_box_arb(x, y, pc - hh, pc + hh, qc - hh, qc + hh, A_lo, A_hi, e_lo, e_hi, c)
        if r["pd"]:
            continue
        if d >= max_depth:
            failed.append((pc, qc, hh)); continue
        h2 = hh / 2
        cells += [(pc + dx * h2, qc + dy * h2, h2, d + 1) for dx in (-1, 1) for dy in (-1, 1)]
    return {"pd": not failed, "boxes": n_box, "failed": failed[:10], "seconds": time.time() - t0}


def u2_system_arb(X, xs, ys, act, A_par=None, with_gap=False, want_J=True):
    """(∂_p L, ∂_q L, ∂_b L[, G]) with L = mean softplus(z) − y z, z = A φ_ε(p x + q) + b, every ε in act.E, and the
    active gap G = φ(q − 1.2p) − φ(q + 0.8p); the Jacobian in (p, q, b[, A]).  A_par: the A ball when A is a parameter."""
    p, q, b = X[0], X[1], X[2]
    A = X[3] if with_gap else A_par
    n = len(xs)
    k = 4 if with_gap else 3
    F = [arb(0)] * 3
    J = [[arb(0)] * k for _ in range(3)]
    for xi, yi in zip(xs, ys):
        sg = p * xi + q
        f0, f1 = act.f(sg), act.df(sg)
        z = A * f0 + b
        S = 1 / (1 + (-z).exp()); res = S - yi
        dz = [A * f1 * xi, A * f1, arb(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if not want_J:
            continue
        w = S * (1 - S)
        f2 = act.d2f(sg)
        d2 = [[A * f2 * xi * xi, A * f2 * xi, arb(0)], [A * f2 * xi, A * f2, arb(0)], [arb(0), arb(0), arb(0)]]
        for i in range(3):
            for j in range(3):
                J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
            if with_gap:
                dA = [f1 * xi, f1, arb(0)]
                J[i][3] += w * f0 * dz[i] + res * dA[i]
    F = [f / n for f in F]
    sO, sI = q - arb("1.2") * p, q + arb("0.8") * p
    if with_gap:
        F.append(act.f(sO) - act.f(sI))
    if not want_J:
        return F
    J = [[J[i][j] / n for j in range(k)] for i in range(3)]
    if with_gap:
        J.append([-arb("1.2") * act.df(sO) - arb("0.8") * act.df(sI), act.df(sO) - act.df(sI), arb(0), arb(0)])
    return F, J


def u2_active_pair(P, Q, act):
    """(valid, G ball) over the box P × Q (balls), every ε in act.E: the class gap min_O φ − max_I φ equals the active pair
    φ(q − 1.2p) − φ(q + 0.8p) when (i) φ is convex on the inner σ-window (φ_σσ = (1+ε)σ·S(v) > 0 there), so its max is at
    an end; (ii) φ is concave on the negative outer window (φ_σσ < 0), so its min is at an end; (iii) φ_σ > 0 on the
    positive outer window, so its min is the left end; and the active ends strictly dominate the other ends."""
    def win(x1, x2):
        return _ival(min(lo_(P * x1 + Q), lo_(P * x2 + Q)), max(hi_(P * x1 + Q), hi_(P * x2 + Q)))
    I, On, Op = win(-0.8, 0.8), win(-2.0, -1.2), win(1.2, 2.0)
    # inner window: φ decreasing on [σ_lo, c] and convex on [c, σ_hi] (c = 0.3) gives its max at an end; when σ_lo >= c
    # convexity alone suffices.  (φ_σ < 0 on |σ| < σ* = arccos(1/a)/√ε, which exceeds 1.1 for every ε <= 0.6.)
    c_ = arb("0.3")

    def _all_sign(fn, a, b, sgn, pieces=16):
        """fn > 0 (sgn = 1) or < 0 (sgn = −1) on [a, b], checked on `pieces` closed sub-intervals (ball evaluation)."""
        if not bool(b > a):
            return True
        for j in range(pieces):
            v = fn(_ival(a + (b - a) * j / pieces, a + (b - a) * (j + 1) / pieces))
            if not (bool(v > 0) if sgn > 0 else bool(v < 0)):
                return False
        return True
    if bool(lo_(I) >= c_):
        inner_ok = _all_sign(act.d2f, lo_(I), hi_(I), 1)
    else:
        inner_ok = _all_sign(act.df, lo_(I), c_, -1) and (bool(hi_(I) <= c_) or _all_sign(act.d2f, c_, hi_(I), 1))
    conds = {"inner_convex": inner_ok, "neg_outer_concave": _all_sign(act.d2f, lo_(On), hi_(On), -1),
             "pos_outer_increasing": _all_sign(act.df, lo_(Op), hi_(Op), 1)}
    u2_active_pair.last = conds
    ok = all(conds.values())
    phi = lambda xx: act.f(P * arb(repr(xx)) + Q)
    I_act, I_oth = phi(0.8), phi(-0.8)
    O_act, O_m2, O_p1 = phi(-1.2), phi(-2.0), phi(1.2)
    dom = {"I_dominates": bool(I_act > I_oth), "O_below_m2": bool(O_act < O_m2), "O_below_p1": bool(O_act < O_p1)}
    u2_active_pair.last = {**conds, **dom}
    ok = ok and all(dom.values())
    return ok, O_act - I_act


def u2_switch_arb(x, y, e_lo, e_hi, Z, rows, J_tube, U_boxes=None):
    """The switch of one slab, replayed from the producer's published boxes (data): Z = 4 intervals (p, q, b, A); rows =
    [{A_lo, A_hi, box: 3 intervals}] covering J_tube.  Checks in Arb with ε as a ball: Krawczyk on Z (unique zero of
    (∇L, G)) with the active pair valid on Z; for each row, Krawczyk on its box with A as a ball parameter (unique critical
    point), the gap's sign on the Krawczyk image (left of Z_A: < 0; right: > 0; inside: the image lies in Z's θ-range);
    the rows tile J_tube exactly; optionally each image lies in its sub-interval's U box (U_boxes: list of (A_lo, A_hi,
    p0, q0, rho_u))."""
    ctx.prec = PREC
    t0 = time.time()
    xs = [arb(float(v)) for v in np.asarray(x, float)]; ys = [int(v) for v in np.asarray(y, float)]
    act = U2Act(_ival(e_lo, e_hi))
    zt = [0.5 * (a + b) for a, b in Z]; zr = [0.5 * (b - a) for a, b in Z]
    okZ, KZ, XZ, _ = krawczyk_arb(lambda X, Jf=True: u2_system_arb(X, xs, ys, act, with_gap=True, want_J=Jf), zt, zr)
    actZ, _ = u2_active_pair(XZ[0], XZ[1], act)
    ZA = (Z[3][0], Z[3][1])
    out = {"Z_krawczyk": bool(okZ), "Z_active_pair": bool(actZ), "rows": len(rows)}
    cover = abs(rows[0]["A_lo"] - J_tube[0]) == 0 and abs(rows[-1]["A_hi"] - J_tube[1]) == 0 and \
        all(rows[i]["A_hi"] == rows[i + 1]["A_lo"] for i in range(len(rows) - 1))
    bad = []
    for r in rows:
        Ab = _ival(r["A_lo"], r["A_hi"])
        bt = [0.5 * (a + b) for a, b in r["box"]]; br = [0.5 * (b - a) for a, b in r["box"]]
        ok, Kb, Xb, _ = krawczyk_arb(lambda X, Jf=True: u2_system_arb(X, xs, ys, act, A_par=Ab, want_J=Jf), bt, br)
        good = bool(ok)
        if good:
            left, right = r["A_hi"] <= ZA[0], r["A_lo"] >= ZA[1]
            if left or right:
                v, G = u2_active_pair(Xb[0], Xb[1], act)
                good = v and (bool(G < 0) if left else bool(G > 0))
            else:
                good = all(bool(lo_(Xb[i]) > lo_(XZ[i])) and bool(hi_(Xb[i]) < hi_(XZ[i])) for i in range(3))
        if good and U_boxes is not None:
            for (al, ah, pu, qu, ru) in U_boxes:
                if al <= r["A_lo"] and r["A_hi"] <= ah:
                    good = bool(lo_(Xb[0]) > pu - ru) and bool(hi_(Xb[0]) < pu + ru) and \
                        bool(lo_(Xb[1]) > qu - ru) and bool(hi_(Xb[1]) < qu + ru)
                    break
        if not good:
            bad.append((r["A_lo"], r["A_hi"]))
    out.update({"covered": bool(cover), "bad_rows": bad[:10], "n_bad": len(bad), "seconds": time.time() - t0})
    out["pass"] = bool(okZ and actZ and cover and not bad)
    return out


def _u2_dE_arb(X, xs, ys, act, A_par=None, with_gap=False):
    """∂_ε of (∂_p L, ∂_q L, ∂_b L[, G]) over the box X (balls) and every ε in act.E: the gradient part
    mean(σ′(z)·A φ_ε·v + (σ(z) − y)·∂_ε v), v = (A φ_σ x, A φ_σ, 1), ∂_ε v = (A φ_σε x, A φ_σε, 0); the gap part
    φ_ε(q − 1.2p) − φ_ε(q + 0.8p)."""
    p, q, b = X[0], X[1], X[2]
    A = X[3] if with_gap else A_par
    n = len(xs)
    D = [arb(0)] * 3
    for xi, yi in zip(xs, ys):
        sg = p * xi + q
        z = A * act.f(sg) + b
        S = 1 / (1 + (-z).exp()); res = S - yi; w = S * (1 - S)
        f1, fe, fse = act.df(sg), act.de(sg), act.dse(sg)
        v = [A * f1 * xi, A * f1, arb(1)]
        dv = [A * fse * xi, A * fse, arb(0)]
        for k in range(3):
            D[k] += w * A * fe * v[k] + res * dv[k]
    D = [d / n for d in D]
    if with_gap:
        D.append(act.de(q - arb("1.2") * p) - act.de(q + arb("0.8") * p))
    return D


def u2_krawczyk_shear_arb(xs, ys, xt, R, X1, e_lo, e_hi, A_par=None, with_gap=True, want_kmax=False):
    """Sheared Krawczyk in Arb: for every ε in E = [e_lo, e_hi], with x̃(ε) = xt + (ε − ε_c)·X1 (X1 any float vector,
    e.g. the producer's predicted ε-dependence -- the test is valid for every choice), the offset box [−R, R] around
    x̃(ε) contains a unique zero of Φ(·; ε) if
        K := −Y·F̃0 + (I − Y·J)·[−R, R] ⊂ (−R, R),
    F̃0 = Φ(xt; ε_c) + [−h, h]·(J(Xs; E)·X1 + ∂_εΦ(Xs; E)) ⊇ {Φ(x̃(ε); ε)} (mean value along the shear),
    J = ∂Φ over Xs + [−R, R] and E, Xs = xt + [−h, h]·X1.  Returns (ok, hull balls of x̃(E) + [−R, R])."""
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    k = len(xt)
    act_c, act_E = U2Act(arb(repr(float(ec)))), U2Act(_ival(e_lo, e_hi))
    pt = [arb(repr(float((t)))) for t in xt]
    F0 = u2_system_arb(pt, xs, ys, act_c, A_par=A_par, with_gap=with_gap, want_J=False)
    Xs = [arb(repr(float((xt[i])))) + arb(0, abs(he * X1[i])) for i in range(k)]
    _, Js = u2_system_arb(Xs, xs, ys, act_E, A_par=A_par, with_gap=with_gap)
    Ds = _u2_dE_arb(Xs, xs, ys, act_E, A_par=A_par, with_gap=with_gap)
    Hh = arb(0, he)
    Ft = [F0[i] + Hh * (sum((Js[i][j] * arb(repr(float((X1[j])))) for j in range(k)), arb(0)) + Ds[i]) for i in range(k)]
    Xb = [Xs[i] + arb(0, float(R[i])) for i in range(k)]
    _, Jb = u2_system_arb(Xb, xs, ys, act_E, A_par=A_par, with_gap=with_gap)
    Jm = np.array([[_mid(Jb[i][j]) for j in range(k)] for i in range(k)])
    if not np.isfinite(Jm).all() or abs(np.linalg.det(Jm)) < 1e-300:
        return (False, Xb, [math.inf] * k) if want_kmax else (False, Xb)
    Y = np.linalg.inv(Jm)
    Ya = [[arb(repr(float((Y[i, j])))) for j in range(k)] for i in range(k)]
    ok = True
    kmax = []
    for i in range(k):
        acc = -sum((Ya[i][j] * Ft[j] for j in range(k)), arb(0))
        for j in range(k):
            cij = arb(1 if i == j else 0) - sum((Ya[i][l] * Jb[l][j] for l in range(k)), arb(0))
            acc += cij * arb(0, float(R[j]))
        ok = ok and bool(lo_(acc) > -arb(float(R[i]))) and bool(hi_(acc) < arb(float(R[i])))
        kmax.append(float_up(_absmax(acc)) if acc.is_finite() else math.inf)
    if want_kmax:
        return ok, Xb, kmax
    return ok, Xb


def u2_krawczyk_shear_inflate(xs, ys, xt, X1, e_lo, e_hi, A_par=None, with_gap=True, r0=1e-7, iters=15, cap=0.05):
    """The checker's OWN radii: start from r0 and grow r to 1.5 × the Krawczyk image until the test contracts (at most
    `iters` times, radii <= cap).  Only the centre xt and the shear X1 are taken from the producer (as data); any
    centre and shear give a valid test.  Returns (ok, radii, hull balls)."""
    k = len(xt)
    r = [float(v) for v in r0] if isinstance(r0, (list, tuple)) else [float(r0)] * k
    Xb = None
    for _ in range(iters):
        ok, Xb, km = u2_krawczyk_shear_arb(xs, ys, xt, r, X1, e_lo, e_hi, A_par=A_par, with_gap=with_gap, want_kmax=True)
        if ok:
            return True, r, Xb
        r = [max(r[i], 1.5 * km[i]) for i in range(k)]
        if not all(math.isfinite(v) for v in r) or max(r) > cap:
            break
    return False, r, Xb


def u2_gap_shear_arb(xt, R, X1, e_lo, e_hi):
    """The active gap along the sheared box θ(ε) = xt + (ε − ε_c)X1 + [−R, R] (p, q), every ε in E (mean value along
    the shear): G(θ_c ± R; ε_c) + [−h, h]·(∇_θG·X1 + ∂_εG)(hull; E)."""
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    act_c, act_E = U2Act(arb(repr(float(ec)))), U2Act(_ival(e_lo, e_hi))
    P = arb(repr(float((xt[0])))) + arb(0, float(R[0])); Q = arb(repr(float((xt[1])))) + arb(0, float(R[1]))
    Ph = P + arb(0, abs(he * X1[0])); Qh = Q + arb(0, abs(he * X1[1]))
    sO, sI = (lambda PP, QQ: (QQ - arb("1.2") * PP, QQ + arb("0.8") * PP))(P, Q)
    hO, hI = Qh - arb("1.2") * Ph, Qh + arb("0.8") * Ph
    # G at ε_c on the box in centred (mean-value) form: G(centre) + ∇G(box)·[−R, R]
    Pc, Qc = arb(repr(float(xt[0]))), arb(repr(float(xt[1])))
    Gc = act_c.f(Qc - arb("1.2") * Pc) - act_c.f(Qc + arb("0.8") * Pc)
    gpb = -arb("1.2") * act_c.df(sO) - arb("0.8") * act_c.df(sI)
    gqb = act_c.df(sO) - act_c.df(sI)
    G0_mv = Gc + gpb * arb(0, float(R[0])) + gqb * arb(0, float(R[1]))
    G0_bx = act_c.f(sO) - act_c.f(sI)
    G0 = G0_mv if bool(G0_mv.rad() < G0_bx.rad()) else G0_bx
    gp = -arb("1.2") * act_E.df(hO) - arb("0.8") * act_E.df(hI)
    gq = act_E.df(hO) - act_E.df(hI)
    T = gp * arb(repr(float((X1[0])))) + gq * arb(repr(float((X1[1])))) + act_E.de(hO) - act_E.de(hI)
    return G0 + arb(0, he) * T


def u2_switch_shear_arb(x, y, e_lo, e_hi, Z, rows, J_tube, max_row_depth=0):
    """The switch of one ε-slab with SHEARED boxes (U2_design.md, item 3), replayed from the producer's published data
    (Z = {xt, X1} for (p, q, b, A); rows = [{A_lo, A_hi, xt, X1}] tiling J_tube; the producer's radii are ignored: the
    checker inflates its own, u2_krawczyk_shear_inflate), all in Arb:
      Z: sheared Krawczyk (unique zero of (∇L, G) in Z(ε) for every ε) and the active pair valid on Z's hull;
      each row: sheared Krawczyk with A as a ball (the unique critical point θ*(A, ε) in the row box for every ε);
      Z's A-range moves: L(ε) = A_Z − R_A + (ε − ε_c)A′, U(ε) = A_Z + R_A + (ε − ε_c)A′ (A′ = Z's X1[3]); the slab is cut
      where a row edge meets L or U; on each piece the row must be entirely left (G < 0 along the shear), entirely right
      (G > 0), entirely inside (its box inside Z's box) or partly outside (both).  Hence for every ε the branch's gap
      has exactly one zero in J_tube, inside Z(ε)."""
    ctx.prec = PREC
    t0 = time.time()
    xs = [arb(float(v)) for v in np.asarray(x, float)]; ys = [int(v) for v in np.asarray(y, float)]
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    zt, X1Z = Z["xt"], Z["X1"]                         # the producer's radii Z["R"] are NOT used
    # start from the §16.3 switch-box radii (a method constant of this checker, not the producer's output) and inflate
    okZ, RZ, XbZ = u2_krawczyk_shear_inflate(xs, ys, zt, X1Z, e_lo, e_hi, with_gap=True, r0=[6e-4, 6e-4, 2e-4, 5e-4])
    actZ, _ = u2_active_pair(XbZ[0], XbZ[1], U2Act(_ival(e_lo, e_hi)))
    actZ_why = dict(getattr(u2_active_pair, "last", {}))
    cover = rows[0]["A_lo"] == J_tube[0] and rows[-1]["A_hi"] == J_tube[1] and \
        all(rows[i]["A_hi"] == rows[i + 1]["A_lo"] for i in range(len(rows) - 1))
    A_Z, R_A, A1 = zt[3], RZ[3], X1Z[3]
    bad = []
    queue = [(r, 0) for r in rows]
    n_split = 0
    while queue:
        r, rdepth = queue.pop(0)
        xt, X1 = r["xt"], r["X1"]                      # the producer's radii r["R"] are NOT used
        a_lo, a_hi = r["A_lo"], r["A_hi"]
        ok, R, _ = u2_krawczyk_shear_inflate(xs, ys, xt, X1, e_lo, e_hi, A_par=_ival(a_lo, a_hi), with_gap=False)
        why = None if ok else "krawczyk"
        if ok:
            cuts = [e_lo, e_hi]
            if A1 != 0:
                for edge_A, off in ((a_lo, -R_A), (a_hi, -R_A), (a_lo, R_A), (a_hi, R_A)):
                    t = ec + (edge_A - A_Z - off) / A1
                    if e_lo < t < e_hi:
                        cuts.append(t)
            cuts = sorted(cuts)
            for lo_o, hi_o in zip(cuts[:-1], cuts[1:]):
                if not ok or hi_o <= lo_o:
                    continue
                em, ho = 0.5 * (lo_o + hi_o), 0.5 * (hi_o - lo_o)
                # the piece's A-range of Z, enclosed (Arb) over the piece
                Lb = arb(repr(float(A_Z - R_A))) + (arb(repr(float(em - ec))) + arb(0, ho)) * arb(repr(float(A1)))
                Ub = arb(repr(float(A_Z + R_A))) + (arb(repr(float(em - ec))) + arb(0, ho)) * arb(repr(float(A1)))
                left = bool(arb(repr(float(a_hi))) < Lb); right = bool(arb(repr(float(a_lo))) > Ub)
                need_in = not (left or right)
                signs = []                                    # the sides on which part of the row may lie outside Z_A
                if left or (need_in and not bool(arb(repr(float(a_lo))) >= Lb)):
                    signs.append(-1)
                if right or (need_in and not bool(arb(repr(float(a_hi))) <= Ub)):
                    signs.append(1)
                if need_in:                                   # row box(ε) ⊂ Z box(ε) for every ε in the piece
                    for i in range(3):
                        dlt = abs(arb(repr(float(xt[i]))) - arb(repr(float(zt[i]))) + arb(repr(float(em - ec))) * (arb(repr(float(X1[i]))) - arb(repr(float(X1Z[i]))))) \
                            + arb(repr(float(ho))) * abs(arb(repr(float(X1[i]))) - arb(repr(float(X1Z[i])))) + arb(float(R[i]))
                        ok = ok and bool(dlt < arb(float(RZ[i])))
                    if not ok and why is None:
                        why = "in_Z"
                for need_sign in signs:
                    if not ok:
                        break
                    xm = [xt[0] + (em - ec) * X1[0], xt[1] + (em - ec) * X1[1]]
                    G = u2_gap_shear_arb(xm, R, X1, lo_o, hi_o)
                    act_ok, _ = u2_active_pair(arb(repr(float(xm[0]))) + arb(0, float(R[0]) + abs(ho * X1[0])),
                                               arb(repr(float(xm[1]))) + arb(0, float(R[1]) + abs(ho * X1[1])),
                                               U2Act(_ival(lo_o, hi_o)))
                    if not act_ok and why is None:
                        why = "active"
                    elif not (bool(G < 0) if need_sign < 0 else bool(G > 0)) and why is None:
                        why = "sign"
                    ok = ok and act_ok and (bool(G < 0) if need_sign < 0 else bool(G > 0))
        if not ok and why in ("sign", "in_Z") and rdepth < max_row_depth:
            # the checker's own refinement: halve the row's A-range (same data centre and shear; Krawczyk re-inflates)
            m_ = 0.5 * (a_lo + a_hi)
            queue[:0] = [({**r, "A_lo": a_lo, "A_hi": m_}, rdepth + 1), ({**r, "A_lo": m_, "A_hi": a_hi}, rdepth + 1)]
            n_split += 1
            continue
        if not ok:
            bad.append((a_lo, a_hi, why))
    res = {"Z_krawczyk": bool(okZ), "Z_active_pair": bool(actZ), "covered": bool(cover), "rows": len(rows),
           "n_bad": len(bad), "bad_rows": bad[:10], "seconds": time.time() - t0,
           "bad_reasons": {k: sum(1 for b_ in bad if b_[2] == k) for k in ("krawczyk", "in_Z", "active", "sign")},
           "Z_active_pair_conditions": actZ_why, "Z_radii": [float(v) for v in RZ], "row_splits": n_split}
    res["pass"] = bool(okZ and actZ and cover and not bad)
    return res


def check_u2_unit(header, x, y, B_table=None, B_full=None, outer_kw=None, ring_kw=None, pd_kw=None):
    """One U2 unit (header: dict with eps, A, centre, regime), re-searched in Arb (the primary route): outer exclusion,
    ring and PD.  Data x, y are passed in (the driver hash-checks them).  Returns the verdicts and timings."""
    (e_lo, e_hi), (A_lo, A_hi), c, reg = header["eps"], header["A"], header["centre"], header["regime"]
    res = {"header": header}
    res["outer"] = u2_outer_arb(x, y, c, A_lo, A_hi, e_lo, e_hi, reg, B_table=B_table, B_full=B_full, **(outer_kw or {}))
    res["ring"] = u2_ring_arb(x, y, c, A_lo, A_hi, e_lo, e_hi, **(ring_kw or {}))
    res["pd"] = u2_pd_arb(x, y, c, A_lo, A_hi, e_lo, e_hi, **(pd_kw or {}))
    res["pass"] = res["outer"]["verdict"] == "pass" and res["ring"]["certified"] and res["pd"]["pd"]
    return res


def check(name):
    kind = json.loads((CERTS / f"{name}.json").read_text()).get("kind")
    return {"ghat_enclosure": check_ghat, "limit_status": check_limit, "finite_a_solve": check_solve}.get(kind, check_finite)(name)


def main(names):
    out = [check(n) for n in names]
    bad = [r["name"] for r in out if not r["pass"]]
    print(f"{len(out)} certificate(s) checked; failures: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["krawczyk"]:
        raise SystemExit(0 if check_krawczyk()["pass"] else 1)
    if sys.argv[1:2] == ["K"]:
        smp = int(sys.argv[2]) if len(sys.argv) > 2 else None
        raise SystemExit(0 if check_K(sample=smp)["pass"] or smp else 1)
    names = sys.argv[1:] or sorted(p.stem for p in CERTS.glob("*.json"))
    raise SystemExit(main(names))
