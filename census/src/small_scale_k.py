"""§21 (Track K): small-scale compactness of the width-1 sine-family conditional minimisers, on all three objectives.

Builds on §17 (`src/small_scale.py`, Theorem C).  Theorem C needs only (1) a parameter theta_1 that attains
S = sup Delta-mu, and (2) an UPPER bound V on Var(phi_{theta_1}); its constants s1 = sqrt(8 F psi(2)/V) and
Wc = (a + sqrt(1.2 V/F))/tau are monotone in V, so any V >= Var(phi_{theta_1}) gives a valid (weaker) statement.
This module computes those inputs rigorously:

  * continuous windows: ball arithmetic (python-flint arb), no float scan.  D_lo = a certified lower bound of
    |D_c(alpha_0)|; the decay bound |D_c| <= 3/(0.8 alpha) puts every maximiser in (0, A0], A0 = 3/(0.8 D_lo); an
    adaptive bisection of [0.5, A0] (with |D_c| <= (32/15) alpha^2 on (0, 0.5]) clears every cell with
    sup |D_c| < D_lo except cells inside alpha_0 +- R; V is enclosed over alpha_0 +- R.  Also the crude bound
    V <= A0^2 Var(x) + a^2 (no scan at all).
  * float 800-point quadrature (the objective the code evaluates): exact rationals.  Every x is k / 2^60 with
    integer k, so D is periodic with period 2 pi 2^60 / g (g = gcd of the k) and a maximiser exists with
    |w1| <= pi 2^60 / g; the data are exactly symmetric (so Delta-mu = a sin b1 D(w1) and Cov(x, cos) = 0).
  * exact linspace population (§17's object): recomputes §17's constants in ball arithmetic from exact rationals.

The spreading count N_tau (closed x-interval of length 0.8) is computed exactly with Fractions.  Nothing is sampled.

    python -m src.small_scale_k        (one process, nice 15, seconds; writes results/small_scale_k.json)
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from functools import reduce
from pathlib import Path

from flint import arb, ctx

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "small_scale_k.json"
PREC = 128
TAU = Fraction(2, 5)                        # half-length of the excluded x-window (Lemma C3), as in §17
A_ALL = (1.02, 1.05, 1.10, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
ALPHA0 = Fraction(17922917, 10 ** 7)        # the float-located continuous maximiser, used only as a test point
R_BOX = Fraction(1, 100)                    # localisation radius around ALPHA0
SMALL = Fraction(1, 2)                      # |D_c| <= (32/15) alpha^2 on (0, SMALL]
VAR_X_C = Fraction(106, 75)                 # Var(x) under the continuous population measure (exact)
F_C = Fraction(3, 4)                        # spreading fraction on the continuous windows (exact: 1 - 0.3125 * 0.8)


def _a(v) -> arb:
    """An arb enclosing the exact value v (int, Fraction or a decimal string)."""
    if isinstance(v, Fraction):
        return arb(v.numerator) / arb(v.denominator)
    return arb(v)


def _cell(lo: Fraction, hi: Fraction) -> arb:
    """A ball containing [lo, hi]."""
    return _a(lo).union(_a(hi))


# ---------------------------------------------------------------------------------------------- continuous windows
def Dc(alpha: arb) -> arb:
    """D_c(alpha) = E_O cos(alpha x) - E_I cos(alpha x) = [sin 2a - sin 1.2a - sin 0.8a]/(0.8a) (alpha ball, 0 excluded)."""
    num = (2 * alpha).sin() - (_a(Fraction(6, 5)) * alpha).sin() - (_a(Fraction(4, 5)) * alpha).sin()
    return num / (_a(Fraction(4, 5)) * alpha)


def Dc_small_bound(alpha_hi: Fraction) -> Fraction:
    """Upper bound of |D_c| on (0, alpha_hi]: the linear terms cancel (2 - 1.2 - 0.8 = 0) and |sin u - u| <= |u|^3/6,
    so |num| <= (8 + 1.728 + 0.512) alpha^3 / 6 and |D_c| <= (10.24/4.8) alpha^2 = (32/15) alpha^2."""
    return Fraction(32, 15) * alpha_hi ** 2


def C_c(beta: arb) -> arb:
    """E cos(beta x) under the population measure (1/2 on I, 1/2 on O): mean of the two class averages."""
    inner = (_a(Fraction(4, 5)) * beta).sin() / (_a(Fraction(4, 5)) * beta)
    outer = ((2 * beta).sin() - (_a(Fraction(6, 5)) * beta).sin()) / (_a(Fraction(4, 5)) * beta)
    return (inner + outer) / 2


def var_cos_c(alpha: arb) -> arb:
    """Var(cos(alpha x)) = E cos^2 - (E cos)^2 = 1/2 + C(2 alpha)/2 - C(alpha)^2."""
    return _a(Fraction(1, 2)) + C_c(2 * alpha) / 2 - C_c(alpha) ** 2


def D_lower_at(alpha: Fraction) -> arb:
    """A certified lower bound of |D_c(alpha)| (as an arb lower end)."""
    ctx.prec = PREC
    return abs(Dc(_a(alpha))).lower()


def certify_localisation(D_lo: arb, centre: Fraction = ALPHA0, r: Fraction = R_BOX, max_depth: int = 40,
                         keep_width: Fraction = Fraction(1, 10 ** 4)):
    """Clear (0, A0] of maximisers of |D_c| outside [centre - r, centre + r].

    A cell is CLEARED when the ball enclosure of |D_c| has upper end < D_lo; otherwise KEPT when it lies inside the
    box and is no wider than keep_width (so a D_lo above the true maximum is exposed); otherwise it is bisected.  Every maximiser has |D_c| >= max |D_c| >= D_lo, so it lies in a kept cell.
    Returns ok = all cells resolved within max_depth AND the test point's own cell is kept (a D_lo above the true
    maximum would clear everything; that is detected and reported as not ok)."""
    ctx.prec = PREC
    A0 = 3 / (_a(Fraction(4, 5)) * D_lo)                    # beyond A0, |D_c| <= 3/(0.8 alpha) < D_lo
    A0_hi = _upper_fraction(A0)
    small_ok = Dc_small_bound(SMALL) < _lower_fraction(D_lo)
    lo_box, hi_box = centre - r, centre + r
    stack = [(SMALL, A0_hi, 0)]
    cleared = kept = 0
    max_depth_used = 0
    kept_cells = []
    failed = []
    while stack:
        lo, hi, d = stack.pop()
        max_depth_used = max(max_depth_used, d)
        if abs(Dc(_cell(lo, hi))).upper() < D_lo:
            cleared += 1
            continue
        if lo >= lo_box and hi <= hi_box and hi - lo <= keep_width:
            kept += 1
            kept_cells.append((lo, hi))
            continue
        if d >= max_depth:
            failed.append((float(lo), float(hi)))
            continue
        mid = (lo + hi) / 2
        stack.append((lo, mid, d + 1))
        stack.append((mid, hi, d + 1))
    test_kept = any(lo <= centre <= hi for lo, hi in kept_cells)
    ok = small_ok and not failed and test_kept and lo_box > SMALL and hi_box < A0_hi
    return {"ok": bool(ok), "A0_upper": float(A0_hi), "small_bound_at_0.5": float(Dc_small_bound(SMALL)),
            "small_ok": bool(small_ok), "cleared_cells": cleared, "kept_cells": kept, "max_depth_used": max_depth_used,
            "failed_cells": failed[:10], "test_point_cell_kept": bool(test_kept),
            "kept_span": [float(min(c[0] for c in kept_cells)), float(max(c[1] for c in kept_cells))] if kept_cells else None}


def _exact_fraction(p: arb) -> Fraction:
    m, e = p.man_exp()
    m, e = int(m), int(e)
    return Fraction(m) * Fraction(2) ** e


def _upper_fraction(x: arb) -> Fraction:
    """A Fraction >= every point of the ball x (its exact upper end)."""
    return _exact_fraction(x.upper())


def _lower_fraction(x: arb) -> Fraction:
    """A Fraction <= every point of the ball x (its exact lower end)."""
    return _exact_fraction(x.lower())


def var_upper_on_box(a: float, centre: Fraction = ALPHA0, r: Fraction = R_BOX, cells: int = 400) -> arb:
    """Upper end of V(alpha) = alpha^2 Var(x) + a^2 Var(cos alpha x) over [centre - r, centre + r] (Var of phi at any
    (w1, b1) with |w1| = alpha and cos b1 = 0; Cov(x, cos alpha x) = 0 by symmetry, E x = 0)."""
    ctx.prec = PREC
    best = None
    lo = centre - r
    h = 2 * r / cells
    aa = _a(Fraction(a).limit_denominator(10 ** 6))
    for i in range(cells):
        al = _cell(lo + i * h, lo + (i + 1) * h)
        v = (al ** 2 * _a(VAR_X_C) + aa ** 2 * var_cos_c(al)).upper()
        best = v if best is None or v > best else best
    return best


def theorem_c_constants(V_hi, a: float, F: Fraction, tau: Fraction = TAU):
    """Rigorous ends of Theorem C's constants for any V >= Var(phi_1): s1 lower end and Wc upper end (floats rounded
    outward by taking the arb ends)."""
    ctx.prec = PREC
    V = V_hi if isinstance(V_hi, arb) else _a(V_hi)
    aa = _a(Fraction(a).limit_denominator(10 ** 6))
    psi2 = arb(1).cosh().log()
    s1 = (8 * _a(F) * psi2 / V).sqrt()
    Wc = (aa + (_a(Fraction(6, 5)) * V / _a(F)).sqrt()) / _a(tau)
    return float(s1.lower()), float(Wc.upper())


# ------------------------------------------------------------------------------------------------ finite objectives
def spread_count_exact(xs, length: Fraction = 2 * TAU) -> int:
    """Largest number of points in a closed interval of the given length (exact; Fractions)."""
    v = sorted(Fraction(t) for t in xs)
    j = best = 0
    for i in range(len(v)):
        while v[i] - v[j] > length:
            j += 1
        best = max(best, i - j + 1)
    return best


def dyadic_lattice(xs):
    """Every float is k / 2^E exactly; return (E, g, ints) with g = gcd of the integers k."""
    fr = [Fraction(float(t)) for t in xs]
    E = max(f.denominator for f in fr).bit_length() - 1
    assert all((f.denominator & (f.denominator - 1)) == 0 for f in fr)  # all dyadic
    ints = [int(f * 2 ** E) for f in fr]
    assert all(Fraction(k, 2 ** E) == f for k, f in zip(ints, fr))
    g = reduce(math.gcd, [abs(k) for k in ints if k != 0])
    return E, g, ints


def exact_symmetry(xs, ys):
    """Each class's multiset of x is symmetric about 0 (exact), and m = E_O x - E_I x (exact)."""
    sym = True
    m = Fraction(0)
    for c in (0, 1):
        v = sorted(Fraction(float(t)) for t, y in zip(xs, ys) if y == c)
        sym &= all(p == -q for p, q in zip(v, reversed(v)))
        m += (1 if c == 1 else -1) * sum(v) / len(v)
    return bool(sym), m


def var_exact(xs) -> Fraction:
    v = [Fraction(float(t)) if not isinstance(t, Fraction) else t for t in xs]
    mu = sum(v) / len(v)
    return sum((t - mu) ** 2 for t in v) / len(v)


def linspace_exact():
    """The exact linspace population on the lattice q = 0.4/79401: inner n = (-798 + 4k)*199, outer +-(597 + 2j)*399."""
    q = Fraction(2, 5) / 79401
    inner = [(-798 + 4 * k) * 199 * q for k in range(400)]
    pos = [(597 + 2 * j) * 399 * q for j in range(200)]
    return q, inner, pos + [-t for t in pos]


def alias_minimality(q: Fraction, inner, outer_pos):
    """Lemma K4's integer facts: consecutive inner gaps are all 796 q and consecutive same-side outer gaps all 798 q,
    so |D(alpha)| = 2 forces alpha * 796 q and alpha * 798 q in 2 pi Z, hence alpha * 2 q in 2 pi Z, i.e. |alpha| >= pi/q."""
    gi = {(b - a_) / q for a_, b in zip(inner, inner[1:])}
    go = {(b - a_) / q for a_, b in zip(outer_pos, outer_pos[1:])}
    return sorted(int(t) for t in gi), sorted(int(t) for t in go)


def D_float_at(alpha: arb, xs, ys) -> arb:
    """Ball enclosure of D(alpha) = mean_O cos(alpha x) - mean_I cos(alpha x) on the exact float data."""
    ctx.prec = PREC
    so, si, no, ni = arb(0), arb(0), 0, 0
    for t, y in zip(xs, ys):
        c = (alpha * _a(Fraction(float(t)))).cos()
        if y == 1:
            so, no = so + c, no + 1
        else:
            si, ni = si + c, ni + 1
    return so / no - si / ni


def main():
    os.nice(15)
    ctx.prec = PREC
    out = {}
    # ---- continuous windows
    D_lo = D_lower_at(ALPHA0)
    loc = certify_localisation(D_lo)
    A0 = 3 / (_a(Fraction(4, 5)) * D_lo)
    rows = []
    for a in A_ALL:
        Vb = var_upper_on_box(a)
        s1, Wc = theorem_c_constants(Vb, a, F_C)
        Vcrude = (A0 ** 2 * _a(VAR_X_C) + _a(Fraction(a).limit_denominator(10 ** 6)) ** 2).upper()
        s1c, Wcc = theorem_c_constants(Vcrude, a, F_C)
        rows.append({"a": a, "V_upper_box": float(Vb), "s1_lower_box": s1, "Wc_upper_box": Wc,
                     "V_upper_crude": float(Vcrude), "s1_lower_crude": s1c, "Wc_upper_crude": Wcc})
    out["continuous"] = {"D_lo_at_alpha0": float(D_lo), "alpha0": float(ALPHA0), "box_radius": float(R_BOX),
                         "A0_upper": float(A0.upper()), "var_x": float(VAR_X_C), "F": float(F_C),
                         "localisation": loc, "rows": rows}
    # ---- float quadrature (the code's objective)
    from .blockB_landscape import population_data
    x, y = population_data()
    xs, ys = x.numpy().tolist(), y.numpy().tolist()
    E, g, ints = dyadic_lattice(xs)
    sym, m = exact_symmetry(xs, ys)
    n_tau = spread_count_exact(xs)
    F_fl = 1 - Fraction(n_tau, len(xs))
    vx_fl = var_exact(xs)
    half_period = arb.pi() * _a(Fraction(2 ** E, g))                       # a maximiser exists with |w1| <= this
    frows = []
    for a in A_ALL:
        V = (half_period ** 2 * _a(vx_fl) + _a(Fraction(a).limit_denominator(10 ** 6)) ** 2).upper()
        s1, Wc = theorem_c_constants(V, a, F_fl)
        frows.append({"a": a, "V_upper": float(V), "s1_lower": s1, "Wc_upper": Wc})
    Dq = D_float_at(arb.pi() / _a(Fraction(2, 5) / 79401), xs, ys)      # diagnostic: the alias on the float data
    out["float_quadrature"] = {"two_plus_D_at_pi_over_q_upper": float((Dq + 2).upper()),
                               "D_at_pi_over_q_lower": float(Dq.lower()), "D_at_pi_over_q_upper": float(Dq.upper()),
                               "n": len(xs), "dyadic_exponent": E, "gcd": g, "half_period_upper": float(half_period.upper()),
                               "classes_symmetric_exact": sym, "m_exact": str(m), "spread_count": n_tau,
                               "F": float(F_fl), "var_x": float(vx_fl), "rows": frows}
    # ---- exact linspace population (§17's object), recomputed rigorously
    q, inner, outer = linspace_exact()
    max_dev = max(abs(Fraction(float(t)) - u) for t, u in zip(xs, inner + outer))
    allx = inner + outer
    n_ex = spread_count_exact(allx)
    F_ex = 1 - Fraction(n_ex, len(allx))
    vx_ex = var_exact(allx)
    alias = arb.pi() / _a(q)
    erows = []
    for a in A_ALL:
        V = (alias ** 2 * _a(vx_ex) + _a(Fraction(a).limit_denominator(10 ** 6)) ** 2)   # exact Var at the alias
        s1, Wc = theorem_c_constants(V, a, F_ex)
        erows.append({"a": a, "V": float(V.mid()), "s1_lower": s1, "Wc_upper": Wc})
    gi, go = alias_minimality(q, inner, outer[:200])
    out["exact_linspace"] = {"inner_gaps_in_q": gi, "outer_gaps_in_q": go, "q": str(q), "alias_alpha": float(alias.mid()), "spread_count": n_ex, "F": float(F_ex),
                             "var_x": float(vx_ex), "max_float_deviation": float(max_dev), "rows": erows}
    cr = {r_["a"]: r_ for r_ in rows}
    fr = {r_["a"]: r_ for r_ in frows}
    er = {r_["a"]: r_ for r_ in erows}
    out["ledger"] = {
        "K D_lo continuous": out["continuous"]["D_lo_at_alpha0"],
        "K A0 continuous": out["continuous"]["A0_upper"],
        "K localisation ok": float(loc["ok"]),
        "K cleared cells": loc["cleared_cells"], "K kept cells": loc["kept_cells"],
        "K kept span lo": loc["kept_span"][0], "K kept span hi": loc["kept_span"][1],
        "K Wc continuous box a=1.30": cr[1.30]["Wc_upper_box"], "K s1 continuous box a=1.30": cr[1.30]["s1_lower_box"],
        "K Wc continuous box a=1.60": cr[1.60]["Wc_upper_box"], "K s1 continuous box a=1.60": cr[1.60]["s1_lower_box"],
        "K Wc continuous crude a=1.30": cr[1.30]["Wc_upper_crude"], "K s1 continuous crude a=1.30": cr[1.30]["s1_lower_crude"],
        "K Wc continuous crude a=1.60": cr[1.60]["Wc_upper_crude"], "K s1 continuous crude a=1.60": cr[1.60]["s1_lower_crude"],
        "K float dyadic exponent": E, "K float gcd": g, "K float symmetric": float(sym),
        "K float spread count": n_tau, "K float half period (units 1e18)": out["float_quadrature"]["half_period_upper"] / 1e18,
        "K Wc float a=1.30 (units 1e19)": fr[1.30]["Wc_upper"] / 1e19, "K s1 float a=1.30": fr[1.30]["s1_lower"],
        "K Wc exact a=1.30": er[1.30]["Wc_upper"], "K s1 exact a=1.30": er[1.30]["s1_lower"],
        "K exact spread count": n_ex, "K exact inner gap": gi[0], "K exact outer gap": go[0],
        "K float 2 + D(pi/q) upper": out["float_quadrature"]["two_plus_D_at_pi_over_q_upper"], "K exact max float deviation": out["exact_linspace"]["max_float_deviation"]}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
