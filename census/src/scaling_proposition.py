"""Block B: symbolic and numerical checks for the scaling-reduction proposition.

Each check is numbered to match results/scaling_proposition.md.  Writes
results/scaling_proposition_checks.csv (one row per check: id, quantity,
value, bound or target, pass).

Run:  PYTHONPATH=. python -m src.scaling_proposition
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
import sympy as sp
import torch
from torch.nn import functional as F

RESULTS = Path(__file__).resolve().parents[1] / "results"
INNER = (-0.8, 0.8)
OUTER = (1.2, 2.0)
K = 0.579454926                     # sup class gap of h, T58/T64
# Compact placement set C = {(u, v): U_LO <= |u| <= U_HI, |v| <= V_MAX}.  On the
# windows (|x| <= 2) this gives |sigma| <= S_MAX = 2*U_HI + V_MAX.
U_LO, U_HI, V_MAX = 1.0, 2.2, 2.0
S_MAX = 2 * U_HI + V_MAX            # 6.4
ROWS: list[dict] = []


def record(cid, quantity, value, target, ok):
    ROWS.append({"check": cid, "quantity": quantity, "value": value,
                 "bound_or_target": target, "pass": bool(ok)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {quantity} = {value}  ({target})")


def h(s):
    return -s + s ** 3 / 6


def step1_2_symbolic():
    print("\nSTEPS 1-2: series (sympy)")
    s, e, sg = sp.symbols("s epsilon sigma", real=True)
    f = sp.pi + s + (1 + e) * sp.sin(sp.pi + s)            # f_a(pi + s), a = 1 + eps
    ser = sp.series(f, s, 0, 9).removeO()
    claim = sp.pi - e * s + (1 + e) * s**3 / 6 - (1 + e) * s**5 / 120 + (1 + e) * s**7 / 5040
    record("1", "f_a(pi+s) - [claimed series to s^7]", str(sp.simplify(ser - claim)), "0",
           sp.simplify(ser - claim) == 0)
    # exact identity underlying the remainder bound: f - pi = s - (1+eps) sin s
    record("1b", "f_a(pi+s) - pi - (s - (1+eps) sin s)",
           str(sp.simplify(f - sp.pi - (s - (1 + e) * sp.sin(s)))), "0",
           sp.simplify(f - sp.pi - (s - (1 + e) * sp.sin(s))) == 0)
    sub = sp.expand(claim.subs(s, sp.sqrt(e) * sg) - sp.pi)
    hh = -sg + sg**3 / 6
    rr = sg**3 / 6 - sg**5 / 120
    lead = sp.simplify(sp.expand(sub).coeff(e, 1) if False else 0)
    # collect powers of sqrt(eps): terms eps^{3/2}, eps^{5/2}, eps^{7/2}, ...
    t = sp.symbols("t", positive=True)                     # t = sqrt(eps)
    poly = sp.expand(sub.subs(e, t**2))
    c3, c5 = sp.simplify(poly.coeff(t, 3)), sp.simplify(poly.coeff(t, 5))
    record("2a", "coefficient of eps^{3/2} minus h(sigma)", str(sp.simplify(c3 - hh)), "0",
           sp.simplify(c3 - hh) == 0)
    record("2b", "coefficient of eps^{5/2} minus r(sigma)", str(sp.simplify(c5 - rr)), "0",
           sp.simplify(c5 - rr) == 0)
    lowest = min(k for k in range(0, 12) if sp.simplify(poly.coeff(t, k)) != 0)
    record("2c", "lowest power of sqrt(eps) present", str(lowest), "3", lowest == 3)


def remainder_bound():
    """Exact remainder after the eps^{5/2} term, with an explicit constant.

    f - pi - eps^{3/2} h - eps^{5/2} r = -eps^{7/2} sigma^5/120 + (1+eps) R7(s),
    |R7(s)| <= |s|^7/5040 (alternating series / Lagrange).  Hence on |sigma|<=S:
        |remainder| <= eps^{7/2} [ S^5/120 + (1+eps) S^7/5040 ].
    Checked here in 60-digit arithmetic on a grid of sigma, for several eps.
    """

    print("\nREMAINDER: explicit bound on |sigma| <= S_MAX, 60-digit arithmetic")
    mp.mp.dps = 60
    worst = 0.0
    for eps in (mp.mpf("0.3"), mp.mpf("0.1"), mp.mpf("0.02"), mp.mpf("0.001")):
        bound = eps ** mp.mpf("3.5") * (mp.mpf(S_MAX) ** 5 / 120
                                         + (1 + eps) * mp.mpf(S_MAX) ** 7 / 5040)
        mx = mp.mpf(0)
        for sg in np.linspace(-S_MAX, S_MAX, 1601):
            sg = mp.mpf(sg)
            s = mp.sqrt(eps) * sg
            fa = s - (1 + eps) * mp.sin(s)
            rem = fa - eps ** mp.mpf("1.5") * (-sg + sg**3 / 6) \
                - eps ** mp.mpf("2.5") * (sg**3 / 6 - sg**5 / 120)
            mx = max(mx, abs(rem))
        ratio = float(mx / bound)
        worst = max(worst, ratio)
        record(f"R({float(eps)})", "max|remainder| / explicit bound", f"{ratio:.4f}",
               "<= 1", ratio <= 1.0)
    return worst


def step4_ghat():
    """G_a(u,v) vs eps^{3/2} G_h(u,v): uniform bound |diff| <= 2 eps^{5/2} M_eps."""

    print("\nSTEP 4: class gap, finite a against the limit, uniformly on C")
    xi = np.linspace(*INNER, 801)
    xo = np.concatenate([np.linspace(*OUTER, 401), -np.linspace(*OUTER, 401)])
    rng = np.random.default_rng(0)
    us = rng.uniform(U_LO, U_HI, 4000) * rng.choice([-1, 1], 4000)
    vs = rng.uniform(-V_MAX, V_MAX, 4000)
    Mr = max(abs(s**3 / 6 - s**5 / 120) for s in np.linspace(-S_MAX, S_MAX, 20001))
    for eps in (0.3, 0.1, 0.02, 0.005):
        a = 1 + eps
        worst = 0.0
        for u, v in zip(us, vs):
            si, so = u * xi + v, u * xo + v
            ti, to = math.pi + math.sqrt(eps) * si, math.pi + math.sqrt(eps) * so
            fi, fo = ti + a * np.sin(ti), to + a * np.sin(to)
            ga = max(fo.min() - fi.max(), fi.min() - fo.max())
            hi, ho = h(si), h(so)
            gh = max(ho.min() - hi.max(), hi.min() - ho.max())
            worst = max(worst, abs(ga / eps**1.5 - gh))
        M_eps = Mr + eps * (S_MAX**5 / 120 + (1 + eps) * S_MAX**7 / 5040)
        bound = 2 * eps * M_eps
        record(f"4({eps})", "sup_C |G_a/eps^1.5 - G_h|  vs  2 eps M_eps",
               f"{worst:.3e}", f"<= {bound:.3e}", worst <= bound)
    return Mr


def step4_localisation():
    """Hypothesis H1 evidence: certified argmax placements, in limit coordinates."""

    print("\nH1 (localisation): certified argmax placements in (u, v) = limit coordinates")
    c = pd.read_csv(RESULTS / "kappa_certified_exact.csv")
    out = []
    for _, r in c.iterrows():
        eps = r.a - 1
        b = (r.b1 - math.pi + math.pi) % (2 * math.pi) - math.pi   # f_a(t+2pi) = f_a(t)+2pi
        u, v = r.w1 / math.sqrt(eps), b / math.sqrt(eps)
        out.append((r.a, u, v))
        inC = U_LO <= abs(u) <= U_HI and abs(v) <= V_MAX
        record(f"H1({r.a})", "argmax (u, v)", f"({u:+.3f}, {v:+.3f})",
               "inside C", inC)
    return out


def step5_loss():
    """|L_a - L_inf| <= (2R/K)(eps*M' + |delta|*H) on C, BCE 1-Lipschitz in the logit."""

    print("\nSTEP 5: conditional loss, finite a against the limit, uniformly on C")
    from .blockB_landscape import population_data
    x, y = population_data()
    x, y = x.numpy(), y.numpy()
    ce = pd.read_csv(RESULTS / "kappa_certified_smalleps.csv")
    rng = np.random.default_rng(1)
    us = rng.uniform(U_LO, U_HI, 1500) * rng.choice([-1, 1], 1500)
    vs = rng.uniform(-V_MAX, V_MAX, 1500)
    b2s = rng.uniform(-3, 3, 1500)
    H = S_MAX + S_MAX**3 / 6
    for eps in (0.1, 0.02, 0.005):
        a = 1 + eps
        from .fold1d_theorem import dip_depth
        from .kappa_certify import certify_exact
        ghat = certify_exact(a)["Ghat_lo"]
        delta = ghat / (K * eps**1.5) - 1
        for R in (0.2, 0.3):
            w2 = 2 * R / ghat
            W = 2 * R / K
            worst = 0.0
            for u, v, b in zip(us, vs, b2s):
                sg = u * x + v
                t = math.pi + math.sqrt(eps) * sg
                za = b + w2 * (t + a * np.sin(t) - math.pi)
                zi = b + W * h(sg)
                la = np.mean(np.logaddexp(0, za) - y * za)
                li = np.mean(np.logaddexp(0, zi) - y * zi)
                worst = max(worst, abs(la - li))
            Mp = max(abs(s**3 / 6 - s**5 / 120) for s in np.linspace(-S_MAX, S_MAX, 4001)) \
                + eps * (S_MAX**5 / 120 + S_MAX**7 / 5040)
            bound = W * (eps * Mp / (1 + delta) + abs(delta) / (1 + delta) * H)
            record(f"5({eps},R={R})", "sup_C |L_a - L_inf|  vs  (2R/K)(eps M'+|delta| H)/(1+delta)",
                   f"{worst:.3e}", f"<= {bound:.3e}", worst <= bound)


def step6_nondegeneracy():
    """H2: at R_glob^inf the limit minimiser is nondegenerate and the gap crosses transversally."""

    print("\nSTEP 6 / H2: nondegeneracy of the limit conditional minimiser at R_glob^inf")
    from .blockB_landscape import population_data
    from .scaling_limit_blockB import best_conditional, gap_of
    x, y = population_data()
    gaps = {}
    mins = {}
    for W in (0.680, 0.685, 0.690, 0.695, 0.700):
        b = best_conditional(W, x, y, restarts=24)
        gaps[W] = b["gap"]
        mins[W] = b
    slope = (gaps[0.695] - gaps[0.685]) / 0.010
    record("6a", "d(gap of limit minimiser)/dW near W* = 0.690", f"{slope:.4f}",
           "!= 0 (transversal crossing)", abs(slope) > 0.1)
    b = mins[0.690]
    p = torch.tensor([b["w1"], b["b1"], b["b2"]], dtype=torch.float64)

    def L(q):
        return F.binary_cross_entropy_with_logits(
            0.690 * (-(q[0] * x + q[1]) + (q[0] * x + q[1]) ** 3 / 6) + q[2], y)

    Hs = torch.autograd.functional.hessian(L, p)
    ev = torch.linalg.eigvalsh(Hs)
    record("6b", "min Hessian eigenvalue of L_inf at the minimiser, W = 0.690",
           f"{float(ev.min()):.4e}", "> 0 (nondegenerate)", float(ev.min()) > 0)
    # continuity of the minimiser across the crossing: no jump between branches
    jumps = [abs(mins[w2]["w1"] - mins[w1]["w1"]) + abs(mins[w2]["b1"] - mins[w1]["b1"])
             for w1, w2 in zip(list(mins)[:-1], list(mins)[1:])]
    record("6c", "max |Δ(u,v)| of the minimiser between adjacent W (step 0.005)",
           f"{max(jumps):.4f}", "small (same branch, no switch)", max(jumps) < 0.1)


def step6_measured():
    print("\nCONNECTION TO MEASURED RATES")
    ce = pd.read_csv(RESULTS / "kappa_certified_smalleps.csv")
    k0 = K / (4 * math.sqrt(2) / 3)
    m = ce.eps <= 0.1
    g = ce.kappa_lo / k0 - 1
    sl = float(np.polyfit(np.log(ce.eps[m]), np.log(g[m]), 1)[0])
    record("M1", "log-log slope of (kappa - kappa_0)/kappa_0 in eps, eps <= 0.1",
           f"{sl:.4f}", "1 (O(eps))", abs(sl - 1) < 0.05)
    sw = pd.read_csv(RESULTS / "blockB_switches.csv").sort_values("a")
    sl2 = pd.read_csv(RESULTS / "scaling_limit_switches.csv").iloc[0]
    dev = (sw.R_glob / sl2.R_glob - 1).values
    e = (sw.a - 1).values
    s2 = float(np.polyfit(np.log(e), np.log(dev), 1)[0])
    record("M2", "log-log slope of R_glob(a)/R_glob^inf - 1 in eps (six a)",
           f"{s2:.4f}", "1 (O(eps))", abs(s2 - 1) < 0.15)


def q_families():
    print("\nq-FAMILIES: exact reduction (sympy)")
    x, e, sg = sp.symbols("x epsilon sigma", positive=True)
    for q in (sp.Rational(2, 3), 1, 2, 4):
        a = 1 + e
        f = (1 - a) * x + a * x**(q + 1) / (q + 1)       # sigma > 0 branch; odd extension
        red = sp.simplify(f.subs(x, e**(1 / q) * sg) / e**(1 + 1 / q))
        hq = -sg + sg**(q + 1) / (q + 1)
        gq = sg**(q + 1) / (q + 1)
        diff = sp.simplify(red - hq - e * gq)
        record(f"q={q}", "f/eps^(1+1/q) - [h_q + eps g_q]", str(diff), "0 (exact, no higher terms)",
               diff == 0)
    # q = 2 against the sin family: h_2(sigma) = h(sigma sqrt2) / sqrt2 ... check
    s = sp.symbols("s", real=True)
    h2 = -s + s**3 / 3
    hs = -s + s**3 / 6
    record("q=2 vs sin", "h_2(tau/sqrt2) * sqrt2 - h(tau)",
           str(sp.simplify(h2.subs(s, s / sp.sqrt(2)) * sp.sqrt(2) - hs)), "0",
           sp.simplify(h2.subs(s, s / sp.sqrt(2)) * sp.sqrt(2) - hs) == 0)


def main():
    step1_2_symbolic()
    remainder_bound()
    step4_ghat()
    step4_localisation()
    step5_loss()
    step6_nondegeneracy()
    step6_measured()
    q_families()
    df = pd.DataFrame(ROWS)
    df.to_csv(RESULTS / "scaling_proposition_checks.csv", index=False)
    print(f"\n{int(df['pass'].sum())}/{len(df)} checks pass; "
          f"written results/scaling_proposition_checks.csv")


if __name__ == "__main__":
    main()
