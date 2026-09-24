"""Width 2 (Route A), geometry: activations, exact extrema, gaps, the placement/bias identity, the margin bound
(Theorem 1), the symmetry group and canonical branches, and the Γ₂ search.  Design: results/width2_design.md.

Nothing here trains or registers anything.  Every check is exercised on constructed pass and fail cases in
tests/test_width2.py before any run.

Network: N(x) = v₁u(α₁x + β₁) + v₂u(α₂x + β₂) + b;  θ = (α₁, β₁, α₂, β₂), v = (v₁, v₂).
φ_v(x) = Σᵢ vᵢu(αᵢx + βᵢ);  G₊(φ) = min_O φ − max_I φ;  G₋(φ) = min_I φ − max_O φ;  G = max(G₊, G₋).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INNER = (-0.8, 0.8)
OUTER = ((-2.0, -1.2), (1.2, 2.0))
TWO_PI = 2 * math.pi


# ------------------------------------------------------------------------------------------ activations
@dataclass(frozen=True)
class Act:
    """u, u′, u″ and B2 = sup|u″| (for the extrema step bound).  `periodic`: β ↦ β + 2π adds a constant to u."""
    name: str
    a: float | None = None

    def u(self, t):
        return t + self.a * np.sin(t) if self.name == "fa" else np.tanh(t)

    def du(self, t):
        return 1 + self.a * np.cos(t) if self.name == "fa" else 1 - np.tanh(t) ** 2

    def d2u(self, t):
        if self.name == "fa":
            return -self.a * np.sin(t)
        th = np.tanh(t)
        return -2 * th * (1 - th ** 2)

    def d2u_bound(self, tlo, thi):
        """sup |u″| over [tlo, thi] (vectorised, exact for both activations)."""
        tlo, thi = np.minimum(tlo, thi), np.maximum(tlo, thi)
        if self.name == "fa":
            # |sin| reaches 1 iff the interval contains π/2 + kπ; else the larger end value
            k = np.ceil((tlo - math.pi / 2) / math.pi)
            hit = (math.pi / 2 + k * math.pi) <= thi
            return self.a * np.where(hit, 1.0, np.maximum(np.abs(np.sin(tlo)), np.abs(np.sin(thi))))
        # |u″(t)| = 2|tanh t|(1 − tanh² t) is even, increasing on [0, t*] and decreasing after, t* = atanh(1/√3)
        ts = math.atanh(1 / math.sqrt(3))
        g = lambda t: 2 * np.abs(np.tanh(t)) * (1 - np.tanh(t) ** 2)
        amin = np.where((tlo <= 0) & (thi >= 0), 0.0, np.minimum(np.abs(tlo), np.abs(thi)))
        amax = np.maximum(np.abs(tlo), np.abs(thi))
        hit = (amin <= ts) & (amax >= ts)
        return np.where(hit, self.B2, np.maximum(g(amin), g(amax)))

    @property
    def B2(self):
        return self.a if self.name == "fa" else 4 / (3 * math.sqrt(3))

    @property
    def periodic(self):
        return self.name == "fa"

    @property
    def tag(self):
        return f"f{self.a:.2f}" if self.name == "fa" else "tanh"


F130, F150, TANH = Act("fa", 1.30), Act("fa", 1.50), Act("tanh")
ACTS = {"f1.30": F130, "f1.50": F150, "tanh": TANH}


def phi(x, th, v, act):
    x = np.asarray(x, float)
    return v[0] * act.u(th[0] * x + th[1]) + v[1] * act.u(th[2] * x + th[3])


def dphi(x, th, v, act):
    x = np.asarray(x, float)
    return v[0] * th[0] * act.du(th[0] * x + th[1]) + v[1] * th[2] * act.du(th[2] * x + th[3])


# ------------------------------------------------------------------------------------------ exact extrema
def _m2_cells(m, h, th, v, act):
    """Per-cell bound on |φ″| = |Σ vᵢαᵢ²u″(αᵢx + βᵢ)| over [m − h, m + h]."""
    out = 0.0
    for i in (0, 1):
        al, be = th[2 * i], th[2 * i + 1]
        t1, t2 = al * (m - h) + be, al * (m + h) + be
        out = out + abs(v[i]) * al ** 2 * act.d2u_bound(t1, t2)
    return out


def extrema(th, v, act, lo, hi, tol=1e-9, vtol=1e-13, n0=64, max_cells=400_000):
    """Min and max of φ_v on [lo, hi] with a rigorous error bound (up to float rounding, ~1e-15 relative).

    On a cell of half-width h around m, M2 bounds |φ″| on the cell (per-cell bound of |u″| over each unit's
    argument interval).  If |φ′(m)| > M2·h, φ′ has one sign on the cell, so φ is monotone there and its extremes on
    the cell are at the cell's ends (evaluated exactly).  Otherwise the cell may hold a critical point: φ on it is
    enclosed by φ(m) ± r, r = |φ′(m)|h + M2h²/2, and it is split until r < vtol or h < tol.
    Returns (min_lo, min_hi, max_lo, max_hi): certified enclosures of the minimum and the maximum."""
    th = np.asarray(th, float); v = np.asarray(v, float)
    ends = phi(np.array([lo, hi]), th, v, act)
    mn_lo = mn_hi = float(ends.min()); mx_lo = mx_hi = float(ends.max())
    edges = np.linspace(lo, hi, n0 + 1)
    m = 0.5 * (edges[:-1] + edges[1:]); h = np.full(n0, 0.5 * (hi - lo) / n0)
    total = 0
    while len(m):
        total += len(m)
        if total > max_cells:
            raise RuntimeError("extrema: cell cap reached")
        d = dphi(m, th, v, act)
        M2 = _m2_cells(m, h, th, v, act)
        mono = np.abs(d) > M2 * h
        # monotone cells: extremes at the cell ends
        if mono.any():
            e = phi(np.concatenate([m[mono] - h[mono], m[mono] + h[mono]]), th, v, act)
            mn_lo = min(mn_lo, float(e.min())); mn_hi = min(mn_hi, float(e.min()))
            mx_lo = max(mx_lo, float(e.max())); mx_hi = max(mx_hi, float(e.max()))
        m, h, d, M2 = m[~mono], h[~mono], d[~mono], M2[~mono]
        r_all = np.abs(d) * h + 0.5 * M2 * h ** 2
        small = (h < tol) | (r_all < vtol)
        if small.any():
            f = phi(m[small], th, v, act)
            r = r_all[small]
            mn_lo = min(mn_lo, float((f - r).min())); mn_hi = min(mn_hi, float(f.min()))
            mx_lo = max(mx_lo, float(f.max())); mx_hi = max(mx_hi, float((f + r).max()))
        m, h = m[~small], h[~small]
        h = h / 2
        m = np.concatenate([m - h, m + h]); h = np.concatenate([h, h])
    return mn_lo, mn_hi, mx_lo, mx_hi


def window_extrema(th, v, act, tol=1e-9):
    """Enclosures of min/max of φ on I and on O = O_left ∪ O_right."""
    i = extrema(th, v, act, *INNER, tol=tol)
    oL = extrema(th, v, act, *OUTER[0], tol=tol)
    oR = extrema(th, v, act, *OUTER[1], tol=tol)
    o = (min(oL[0], oR[0]), min(oL[1], oR[1]), max(oL[2], oR[2]), max(oL[3], oR[3]))
    return {"I": i, "O": o}


def gaps(th, v, act, tol=1e-9):
    """Enclosures of G₊ = min_O φ − max_I φ and G₋ = min_I φ − max_O φ, as (lo, hi)."""
    e = window_extrema(th, v, act, tol)
    (iml, imh, ixl, ixh), (oml, omh, oxl, oxh) = e["I"], e["O"]
    return {"G+": (oml - ixh, omh - ixl), "G-": (iml - oxh, imh - oxl), "extrema": e}


def sign_correct(th, v, b, act, tol=1e-9):
    """N < 0 on I and N > 0 on O, from the extrema of N = φ + b directly.  Returns True / False / None (undecided:
    an enclosure straddles 0)."""
    e = window_extrema(th, v, act, tol)
    max_I = (e["I"][2] + b, e["I"][3] + b)
    min_O = (e["O"][0] + b, e["O"][1] + b)
    if max_I[1] < 0 and min_O[0] > 0:
        return True
    if max_I[0] >= 0 or min_O[1] <= 0:
        return False
    return None


def identity_predicts(th, v, b, act, tol=1e-9):
    """§2 identity: sign-correct ⟺ G₊(φ_v) > 0 and −min_O φ < b < −max_I φ.  True / False / None (undecided)."""
    g = gaps(th, v, act, tol)
    (oml, omh, _, _), (_, _, ixl, ixh) = g["extrema"]["O"], g["extrema"]["I"]
    if g["G+"][0] > 0 and -oml < b < -ixh:
        return True
    if g["G+"][1] <= 0 or b <= -omh or b >= -ixl:
        return False
    return None


def placed_sample(rng, act, tries=20000):
    """A random network with G₊(φ_v) > 0 (exactly), by rejection over a range where placement occurs, oriented with
    the global sign so that the positive class is O.  Used to make sure checks see placed networks."""
    for _ in range(tries):
        th = np.array([rng.uniform(0.3, 6), rng.uniform(0, TWO_PI), rng.uniform(0.3, 6), rng.uniform(0, TWO_PI)])
        if not act.periodic:
            th[1] = rng.uniform(-8, 8); th[3] = rng.uniform(-8, 8)
        v = rng.uniform(-2, 2, 2)
        g = gaps(th, v, act)
        if g["G+"][0] > 1e-6:
            return th, v
        if g["G-"][0] > 1e-6:
            return global_sign(th, v)
    raise RuntimeError("no placed network found")


def identity_check(n, act, seed=0, scale=3.0):
    """Compare sign_correct with the identity on n random networks.  Returns the number of decided mismatches
    (a stop condition if > 0), the number decided, and undecided."""
    rng = np.random.default_rng(seed)
    mism = dec = und = pos = 0
    for k in range(n):
        # half the networks are placed (G₊ > 0), so the identity's positive side is exercised
        if k % 2 == 0:
            th, v = placed_sample(rng, act)
        else:
            th = rng.uniform(-scale, scale, 4); v = rng.uniform(-scale, scale, 2)
        # bias near the feasible interval half the time, so both outcomes occur
        g = gaps(th, v, act)
        lo_b, hi_b = -g["extrema"]["O"][1], -g["extrema"]["I"][2]
        c0, c1 = min(lo_b, hi_b), max(lo_b, hi_b)
        if k % 4 == 0 and lo_b < hi_b:                    # placed network, bias inside the feasible interval
            b = rng.uniform(lo_b, hi_b)
        elif rng.random() < 0.5:
            b = rng.uniform(c0 - 0.5, c1 + 0.5)
        else:
            b = rng.uniform(-scale, scale)
        s, p = sign_correct(th, v, b, act), identity_predicts(th, v, b, act)
        if s is None or p is None:
            und += 1
            continue
        dec += 1
        pos += int(s)
        mism += int(s != p)
    return {"mismatches": mism, "decided": dec, "undecided": und, "n_sign_correct": pos}


def margin(th, v, b, act, tol=1e-9):
    """Largest m with N <= −m on I and N >= m on O (a lower enclosure; negative if not sign-correct)."""
    e = window_extrema(th, v, act, tol)
    return min(-(e["I"][3] + b), e["O"][0] + b)


def theorem1_ok(th, v, b, act, gamma2_hat, rel=1e-9):
    """Theorem 1: m <= ½‖v‖₁Γ₂.  With Γ̂₂ (a lower estimate of Γ₂) a violation beyond rel means Γ̂₂ is too low
    or the implementation is wrong: a stop condition."""
    m = margin(th, v, b, act)
    return m <= 0.5 * (abs(v[0]) + abs(v[1])) * gamma2_hat * (1 + rel), m


# ------------------------------------------------------------------------------------------ symmetry group
def _wrap(beta):
    return np.mod(beta, TWO_PI)


def unit_orient(th, v, i):
    """(αᵢ, βᵢ, vᵢ) → (−αᵢ, −βᵢ, −vᵢ): u is odd, so vᵢu(αᵢx+βᵢ) is unchanged."""
    th = np.array(th, float); v = np.array(v, float)
    th[2 * i] *= -1; th[2 * i + 1] *= -1; v[i] *= -1
    return th, v


def swap(th, v):
    th = np.asarray(th, float); v = np.asarray(v, float)
    return np.array([th[2], th[3], th[0], th[1]]), v[::-1].copy()


def mirror(th, v):
    """x → −x: (αᵢ, βᵢ) → (−αᵢ, βᵢ).  A symmetry of G (windows symmetric) and of the population loss."""
    th = np.array(th, float)
    th[0] *= -1; th[2] *= -1
    return th, np.array(v, float)


def global_sign(th, v):
    """φ → −φ (swaps G₊ and G₋): a symmetry of G only, not of the loss."""
    return np.array(th, float), -np.array(v, float)


def beta_shift(th, v, i, k=1):
    """βᵢ → βᵢ + 2πk: adds vᵢ·2πk (a constant) to φ for f_a.  G is unchanged; the profiled loss is unchanged."""
    th = np.array(th, float)
    th[2 * i + 1] += TWO_PI * k
    return th, np.array(v, float)


def _canon_no_mirror(th, v, act):
    th = np.array(th, float); v = np.array(v, float)
    for i in (0, 1):
        if th[2 * i] < 0 or (th[2 * i] == 0 and v[i] < 0):
            th, v = unit_orient(th, v, i)
    if act.periodic:
        th[1] = _wrap(th[1]); th[3] = _wrap(th[3])
    if (th[2], th[3]) < (th[0], th[1]):
        th, v = swap(th, v)
    return th, v


def canonical(th, v, act, use_mirror=True):
    """Canonical representative under unit orientation, β mod 2π (f_a), unit permutation and (optionally) the
    mirror; the lexicographically smaller of the two mirror images.  Idempotent."""
    a = _canon_no_mirror(th, v, act)
    if not use_mirror:
        return a
    b = _canon_no_mirror(*mirror(th, v), act)
    ka, kb = tuple(np.concatenate(a)), tuple(np.concatenate(b))
    return a if ka <= kb else b


def group_images(th, v, act, use_mirror=True):
    """All images under the group generated by unit orientation, permutation and the mirror (β reduced mod 2π for
    f_a), as a list of (θ, v)."""
    out = []
    for m in ((False, True) if use_mirror else (False,)):
        t0, v0 = mirror(th, v) if m else (np.array(th, float), np.array(v, float))
        for s in (False, True):
            t1, v1 = swap(t0, v0) if s else (t0, v0)
            for o0 in (False, True):
                for o1 in (False, True):
                    t2, v2 = t1, v1
                    if o0:
                        t2, v2 = unit_orient(t2, v2, 0)
                    if o1:
                        t2, v2 = unit_orient(t2, v2, 1)
                    if act.periodic:
                        t2 = t2.copy(); t2[1] = _wrap(t2[1]); t2[3] = _wrap(t2[3])
                    out.append((t2, v2))
    return out


def branch_distance(p, q, act, use_mirror=True):
    """min over group images of q of the Euclidean distance to p in (θ, v); β differences taken mod 2π for f_a."""
    tp, vp = p
    best = math.inf
    for tq, vq in group_images(*q, act, use_mirror):
        d = np.concatenate([tp - tq, vp - vq])
        if act.periodic:
            for k in (1, 3):
                d[k] = (d[k] + math.pi) % TWO_PI - math.pi
        best = min(best, float(np.sqrt((d ** 2).sum())))
    return best


# ------------------------------------------------------------------------------------------ Γ₂ search
def gap_of(z, act):
    """G(φ_ṽ) at z = (α₁, β₁, α₂, β₂, t) with ṽ = (t, 1 − |t|) (global sign handled by G = max(G₊, G₋)); a fast
    dense-grid value used inside searches (the exact value is recomputed on the returned point)."""
    th = z[:4]; t = float(np.clip(z[4], -1, 1)); v = np.array([t, 1 - abs(t)])
    return _dense_gap(th, v, act)


_XI = np.linspace(*INNER, 801)
_XO = np.concatenate([np.linspace(*OUTER[0], 401), np.linspace(*OUTER[1], 401)])


def _dense_gap(th, v, act):
    fi, fo = phi(_XI, th, v, act), phi(_XO, th, v, act)
    return max(fo.min() - fi.max(), fi.min() - fo.max())


def exact_gap(th, v, act):
    g = gaps(th, v, act)
    return max(g["G+"][0], g["G-"][0]), max(g["G+"][1], g["G-"][1])


def nelder_mead(f, x0, step=0.5, iters=2000, xtol=1e-10, ftol=1e-13):
    """Minimise f by Nelder–Mead (standard coefficients); returns (x, f(x))."""
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < ftol and max(np.abs(p - pts[0]).max() for p in pts) < xtol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def differential_evolution(f, lo, hi, pop=40, gens=400, F=0.7, CR=0.9, seed=0):
    """DE/rand/1/bin over the box [lo, hi]; returns (x, f(x))."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = lo + rng.random((pop, len(lo))) * (hi - lo)
    fx = np.array([f(x) for x in X])
    for _ in range(gens):
        for k in range(pop):
            a, b, c = X[rng.choice([j for j in range(pop) if j != k], 3, replace=False)]
            y = np.clip(a + F * (b - c), lo, hi)
            mask = rng.random(len(lo)) < CR
            mask[rng.integers(len(lo))] = True
            z = np.where(mask, y, X[k]); fz = f(z)
            if fz <= fx[k]:
                X[k], fx[k] = z, fz
    i = int(np.argmin(fx))
    return X[i], fx[i]


BOX_ALPHA = 10.0          # |αᵢ| <= 10, βᵢ ∈ [0, 2π): the stated box if Lemma 4 does not close


def gamma2_multistart(act, starts=4000, seed=0, box=BOX_ALPHA):
    """Multistart Nelder–Mead on −G over the reduced box (α₁, α₂ ∈ [0, box] by unit orientation; β ∈ [0, 2π) for
    f_a, [−box, box] for tanh; t ∈ [−1, 1]).  Returns the best point, with its exact gap enclosure."""
    rng = np.random.default_rng(seed)
    blo = 0.0 if act.periodic else -box
    bhi = TWO_PI if act.periodic else box
    best = (-math.inf, None)
    for _ in range(starts):
        z0 = np.array([rng.uniform(0, box), rng.uniform(blo, bhi), rng.uniform(0, box), rng.uniform(blo, bhi),
                       rng.uniform(-1, 1)])
        z, fv = nelder_mead(lambda z: -gap_of(_clip_box(z, box, act), act), z0, iters=600)
        if -fv > best[0]:
            best = (-fv, _clip_box(z, box, act))
    return _finish(best[1], act)


def gamma2_de(act, seed=1, box=BOX_ALPHA, pop=60, gens=500):
    """Independent search: DE with a different parametrisation of ṽ (angle ψ, ṽ = (cos ψ, sin ψ)/‖·‖₁)."""
    blo = 0.0 if act.periodic else -box
    bhi = TWO_PI if act.periodic else box

    def g(y):
        c, s = math.cos(y[4]), math.sin(y[4])
        v = np.array([c, s]) / (abs(c) + abs(s))
        return -_dense_gap(y[:4], v, act)
    y, fv = differential_evolution(g, [0, blo, 0, blo, -math.pi], [box, bhi, box, bhi, math.pi], pop, gens, seed=seed)
    c, s = math.cos(y[4]), math.sin(y[4]); v = np.array([c, s]) / (abs(c) + abs(s))
    th = y[:4]
    lo, hi = exact_gap(th, v, act)
    return {"gamma_lo": lo, "gamma_hi": hi, "theta": th, "v": v}


def _clip_box(z, box, act):
    z = np.array(z, float)
    z[0] = np.clip(z[0], 0, box); z[2] = np.clip(z[2], 0, box); z[4] = np.clip(z[4], -1, 1)
    if not act.periodic:
        z[1] = np.clip(z[1], -box, box); z[3] = np.clip(z[3], -box, box)
    return z


def _finish(z, act):
    th = z[:4]; t = float(z[4]); v = np.array([t, 1 - abs(t)])
    lo, hi = exact_gap(th, v, act)
    return {"gamma_lo": lo, "gamma_hi": hi, "theta": th, "v": v}


def searches_agree(r1, r2, rel=1e-6):
    """§5: the two searches' Γ̂₂ agree to rel (lower ends of their exact enclosures)."""
    a, b = r1["gamma_lo"], r2["gamma_lo"]
    return abs(a - b) <= rel * max(abs(a), abs(b))


def theorem2_ok(value):
    """Theorem 2 numerics: for tanh every search value must be < 1 (a supremum, not attained)."""
    return value < 1.0


def averaging_bound(act, lam):
    """A proved partial step toward Lemma 4 (not the lemma): if both |αᵢ| >= lam, then G₊(φ_ṽ) and G₋(φ_ṽ) are at
    most a·(2/(0.8·lam) + 2/(1.6·lam)) = 3.75a/lam.  Proof: min <= mean and max >= mean over one O side (length 0.8,
    mean x = ±1.6) and over I (length 1.6, mean x = 0); choose the O side with c·mean_x <= 0; the constants cancel;
    |mean of sin(αx+β)| over an interval of length L is <= 2/(|α|L).  f_a only."""
    assert act.periodic
    return act.a * (2 / (0.8 * lam) + 2 / (1.6 * lam))
