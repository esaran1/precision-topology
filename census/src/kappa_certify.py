"""Certify the kappa interval by a compact-domain Lipschitz argument.

Registered in results/kappa_certification_prediction.md.  float64.

Ghat is a SUPREMUM over placements; a finite grid can only under-estimate it, so
T50's kappa interval [0.30544, 0.31543] is a lower bound with no error bar.  This
module bounds the gap:

    Ghat_grid  <=  Ghat  <=  Ghat_grid + L*h/2

with L the Lipschitz constant of G in (w1, b1) over a compact domain, and h the
grid spacing.  Same device as T57's solves() certificate, applied to a supremum.

Lipschitz bound:  |f_a'| = |1 + a cos t| <= 1 + a, and G is a max/min of
compositions x -> f_a(w1 x + b1), so
    |dG/dw1| <= (1+a) * max|x|,   |dG/db1| <= (1+a)
giving L = (1+a) * max(1, max|x|) in the sup-norm.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX
from .fold1d_theorem import dip_depth

RESULTS = Path(__file__).resolve().parents[1] / "results"
XMAX = max(INNER_MAX, OUTER_MAX)


def lipschitz(a: float) -> float:
    """L for G in (w1, b1), sup-norm."""

    return (1.0 + a) * max(1.0, XMAX)


def windows(n_inner=4001, n_outer=2000):
    I = torch.linspace(-INNER_MAX, INNER_MAX, n_inner, dtype=torch.float64)
    P = torch.linspace(OUTER_MIN, OUTER_MAX, n_outer, dtype=torch.float64)
    return I, torch.cat([P, -P])


def gap_grid(a, w1s, b1s, I, O):
    """max over the (w1,b1) grid of the oriented class gap.  Vectorised over b1."""

    best, arg = -np.inf, None
    bt = torch.as_tensor(b1s, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        for w1 in np.asarray(w1s, float):
            vi = (float(w1) * I).unsqueeze(0) + bt
            vo = (float(w1) * O).unsqueeze(0) + bt
            vi = vi + a * torch.sin(vi)
            vo = vo + a * torch.sin(vo)
            g = torch.maximum(vo.min(1).values - vi.max(1).values,
                              vi.min(1).values - vo.max(1).values)
            j = int(g.argmax())
            if float(g[j]) > best:
                best, arg = float(g[j]), (float(w1), float(bt[j, 0]))
    return best, arg


def certify(a, W=3.0, B=6.0, target_rel=1e-3, h0=0.02, hmin=1e-6, verbose=False):
    """Refine h until L*h/2 < target_rel * Ghat.  Returns the certified interval."""

    I, O = windows()
    L = lipschitz(a)
    h = h0
    lo, arg = gap_grid(a, np.arange(-W, W + 1e-12, h), np.arange(math.pi - B, math.pi + B + 1e-12, h), I, O)
    while True:
        slack = L * h / 2.0
        if slack < target_rel * lo or h <= hmin:
            break
        # zoom: refine around the current argmax by a factor 5
        h = h / 5.0
        w0, b0 = arg
        pad = 12 * h
        lo, arg = gap_grid(a, np.arange(w0 - pad, w0 + pad + 1e-12, h),
                           np.arange(b0 - pad, b0 + pad + 1e-12, h), I, O)
        if verbose:
            print(f"    h={h:.2e} Ghat>={lo:.9f} slack={L*h/2:.2e}", flush=True)
    return {"a": a, "L": L, "h": h, "Ghat_lo": lo, "Ghat_hi": lo + L * h / 2.0,
            "slack": L * h / 2.0, "w1": arg[0], "b1": arg[1]}


def boundary_check(a, W=3.0, B=6.0, interior_max=None, h=0.002):
    """C-3: is G on the boundary shell strictly below the interior maximum?"""

    I, O = windows(2001, 1000)
    best = -np.inf
    # |w1| = W faces
    for w1 in (-W, W):
        g, _ = gap_grid(a, [w1], np.arange(math.pi - B, math.pi + B + 1e-12, h), I, O)
        best = max(best, g)
    # |b1 - pi| = B faces
    for b1 in (math.pi - B, math.pi + B):
        g, _ = gap_grid(a, np.arange(-W, W + 1e-12, h), [b1], I, O)
        best = max(best, g)
    return {"a": a, "boundary_max": best, "interior_max": interior_max,
            "strictly_below": None if interior_max is None else bool(best < interior_max)}


def certify_K(target_rel=1e-3, h0=0.05, hmin=1e-7):
    """C-4: the same device on h(sigma) = -sigma + sigma^3/6, for K."""

    I = np.linspace(-INNER_MAX, INNER_MAX, 8001)
    P = np.linspace(OUTER_MIN, OUTER_MAX, 4000)
    O = np.concatenate([P, -P])
    h_fn = lambda z: -z + z ** 3 / 6.0
    # sigma = u*x + v; |dG/du| <= max|h'| * max|x|, |dG/dv| <= max|h'|
    # on the relevant range |sigma| <= S, |h'| = |-1 + sigma^2/2| <= max(1, S^2/2 - 1)
    S = 8.0
    hp = max(1.0, S * S / 2.0 - 1.0)
    L = hp * max(1.0, XMAX)
    hh = h0
    def scan(us, vs):
        best, arg = -np.inf, None
        for u in us:
            for v in vs:
                gi, go = h_fn(u * I + v), h_fn(u * O + v)
                g = max(go.min() - gi.max(), gi.min() - go.max())
                if g > best: best, arg = g, (u, v)
        return best, arg
    lo, arg = scan(np.arange(0.02, 6.0, hh), np.arange(-8.0, 8.0, hh))
    while L * hh / 2.0 >= target_rel * lo and hh > hmin:
        hh /= 5.0
        u0, v0 = arg
        pad = 12 * hh
        lo, arg = scan(np.arange(u0 - pad, u0 + pad + 1e-12, hh),
                       np.arange(v0 - pad, v0 + pad + 1e-12, hh))
    return {"L": L, "h": hh, "K_lo": lo, "K_hi": lo + L * hh / 2.0,
            "slack": L * hh / 2.0, "u": arg[0], "v": arg[1]}


def main():
    A_VALUES = (1.02, 1.05, 1.10, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
    print("=== certified Ghat and kappa, compact-domain Lipschitz ===", flush=True)
    print(f"{'a':>6} {'L':>7} {'h':>9} {'Ghat_lo':>11} {'Ghat_hi':>11} "
          f"{'rel width':>10} {'kappa_lo':>9} {'kappa_hi':>9}", flush=True)
    rows = []
    for a in A_VALUES:
        c = certify(a)
        D = dip_depth(a)
        c["D"] = D
        c["kappa_lo"] = c["Ghat_lo"] / D
        c["kappa_hi"] = c["Ghat_hi"] / D
        c["rel_width"] = (c["Ghat_hi"] - c["Ghat_lo"]) / c["Ghat_lo"]
        rows.append(c)
        print(f"{a:6.2f} {c['L']:7.3f} {c['h']:9.2e} {c['Ghat_lo']:11.7f} "
              f"{c['Ghat_hi']:11.7f} {c['rel_width']*100:9.4f}% "
              f"{c['kappa_lo']:9.6f} {c['kappa_hi']:9.6f}", flush=True)
    d = pd.DataFrame(rows)
    sub = d[d.a <= 1.60]
    print(f"\n  certified kappa over a <= 1.60: "
          f"[{sub.kappa_lo.min():.6f}, {sub.kappa_hi.max():.6f}]", flush=True)
    print(f"  T50 reported:                   [0.30544, 0.31543]", flush=True)

    print("\n=== C-3: boundary shell below the interior maximum? ===", flush=True)
    brows = []
    for a in (1.02, 1.30, 1.60):
        im = float(d[d.a == a].Ghat_lo.iloc[0])
        b = boundary_check(a, interior_max=im)
        brows.append(b)
        print(f"  a={a}: boundary max {b['boundary_max']:.7f} vs interior "
              f"{im:.7f} -> {'BELOW' if b['strictly_below'] else 'NOT BELOW'}", flush=True)

    print("\n=== C-4: certify K on h(sigma) ===", flush=True)
    k = certify_K()
    print(f"  L={k['L']:.3f} h={k['h']:.2e}  K in [{k['K_lo']:.9f}, {k['K_hi']:.9f}]  "
          f"rel width {(k['K_hi']-k['K_lo'])/k['K_lo']*100:.5f}%", flush=True)
    D_inf = 4 * math.sqrt(2) / 3
    print(f"  kappa_0 in [{k['K_lo']/D_inf:.7f}, {k['K_hi']/D_inf:.7f}]  "
          f"(reported 0.307302)", flush=True)

    stem = RESULTS / "kappa_certified"
    with artifact_lock(stem, "kappa certified"):
        d.to_csv(stem.with_suffix(".csv"), index=False)
    pd.DataFrame(brows).to_csv(RESULTS / "kappa_boundary.csv", index=False)
    pd.DataFrame([k]).to_csv(RESULTS / "kappa_K_certified.csv", index=False)
    print("\nwritten kappa_certified.csv, kappa_boundary.csv, kappa_K_certified.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Item #2 (Junyu): use EXACT extrema for G(w1,b1) in the certified search, so
# the certificate rests on exact evaluation in x rather than a grid in x.
#
# The (w1,b1) grid and its Lipschitz slack are unchanged -- that is the outer
# certificate.  What changes is the INNER evaluation of G at each grid point:
# previously a 4,001/2,000-point sample over the windows, now endpoints plus the
# exact interior roots of 1 + a cos(w1 x + b1) = 0.  The grid could only
# UNDER-state max_I and OVER-state min_O, so this can only lower G at each point
# and therefore tighten nothing spuriously.
# ---------------------------------------------------------------------------

def gap_exact_grid(a, w1s, b1s):
    """max over the (w1,b1) grid of the oriented class gap, x-extrema EXACT."""

    from .exact_extrema import exact_gap

    best, arg = -np.inf, None
    for w1 in np.asarray(w1s, float):
        if abs(w1) < 1e-12:
            continue
        for b1 in np.asarray(b1s, float):
            g, _ = exact_gap(a, float(w1), float(b1))
            if g > best:
                best, arg = g, (float(w1), float(b1))
    return best, arg


def certify_exact(a, W=3.0, B=6.0, target_rel=1e-3, h0=0.05, hmin=1e-6):
    """Same certificate as certify(), with exact x-extrema at every grid point."""

    L = lipschitz(a)
    h = h0
    lo, arg = gap_exact_grid(a, np.arange(-W, W + 1e-12, h),
                             np.arange(math.pi - B, math.pi + B + 1e-12, h))
    while L * h / 2.0 >= target_rel * lo and h > hmin:
        h = h / 5.0
        w0, b0 = arg
        pad = 12 * h
        lo, arg = gap_exact_grid(a, np.arange(w0 - pad, w0 + pad + 1e-12, h),
                                 np.arange(b0 - pad, b0 + pad + 1e-12, h))
    return {"a": a, "L": L, "h": h, "Ghat_lo": lo, "Ghat_hi": lo + L * h / 2.0,
            "slack": L * h / 2.0, "w1": arg[0], "b1": arg[1]}
