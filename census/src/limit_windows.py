"""Per-window limit switch A* (math_note_v2, Theorem (a)-(b), for every window geometry the paper uses).

Windows: base, Block G (G1-G4, blockG_windows.WINDOWS), Block 3 held-out (H10, H35, H65, H90).  Objective: the
window's 800-point quadrature population (base: blockB_landscape.population_data; others Window.population(400)),
as in the finite-a certificates for that window.

Per window:
  K      certified limit gap sup, every window including base (branch and bound as prospective_windows.K_certified,
         window-general, exact extrema, u >= 0 by the σ -> -σ symmetry of h with both orientations).
  A*     limit_bnb certify (window-general) on K(P) = {|p| <= P, |q| <= 2√2 + X P}, X = max|x|; half search only
         for x-symmetric windows.  Bracket from A0 = 2·0.20/K in steps of 0.1, then bisection to width 0.005.
         R^∞ = K A*/2 as an interval (K_lo·A_lo/2, K_hi·A_hi/2).
  Localisation over the bracket: branch loss m(A) <= max(L0*(θ; A_lo), L0*(θ; A_hi)) for the fixed θ = argmin at
         A_lo (convexity in A), against min(B(P), B_full) in exact arithmetic (math_note_v2_checks).
  Float diagnostics (not certificates): argmin shift across the bracket, dG0/dA by central difference.

    python -m src.limit_windows [workers]
    python -m src.limit_windows K_base      # base-window K interval (exact extrema) and the certified R_glob^∞

T58's K = 0.579454926 is a nested-refinement estimate slightly below the supremum (the attained value here is
0.5794558); certified R intervals therefore use (K_lo·A_lo/2, K_hi·A_hi/2) with this K interval.
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .limit_bnb import R2, _h_extrema, _status, gap0, profile

RESULTS = Path(__file__).resolve().parents[1] / "results"
P_BOX = 24.0
WIDTH = 0.005


def windows():
    from .blockG_windows import WINDOWS, Window
    out = list(WINDOWS)
    for r in pd.read_csv(RESULTS / "prospective_windows.csv").itertuples():
        out.append(Window(f"H{int(round(r.quantile * 100)):02d}", -r.i, r.i, r.o1, r.o2))
    return out


def _data(win):
    if win.tag == "base":
        from . import blockB_landscape as bb
        x, y = bb.population_data()
    else:
        x, y = win.population(400)
    return x.numpy(), y.numpy()


def _g0_both(u, v, inner, outer):
    def ext(lo, hi):
        a1, a2 = u * lo + v, u * hi + v
        return _h_extrema(np.minimum(a1, a2), np.maximum(a1, a2))
    imn, imx = ext(*inner)
    p_mn, p_mx = ext(*outer)
    n_mn, n_mx = ext(-outer[1], -outer[0])
    omn, omx = np.minimum(p_mn, n_mn), np.maximum(p_mx, n_mx)
    return np.maximum(omn - imx, imn - omx)


def K_certified(inner, outer, U=8.0, V=12.0, h0=0.05, rel=1e-4):
    X = max(abs(inner[0]), abs(inner[1]), outer[1])
    cG = max(abs(inner[0]), abs(inner[1])) + outer[1]
    nu, nv = int(U / h0), int(2 * V / h0)
    hu, hv = U / nu / 2, V / nv
    cu, cv = np.meshgrid((np.arange(nu) + 0.5) * 2 * hu, -V + (np.arange(nv) + 0.5) * 2 * hv, indexing="ij")
    cu, cv = cu.ravel(), cv.ravel()
    best, arg = -np.inf, None
    for _ in range(40):
        G = _g0_both(cu, cv, inner, outer)
        smax = X * (np.abs(cu) + hu) + np.abs(cv) + hv
        ub = G + (1 + smax ** 2 / 2) * (cG * hu + 2 * hv)
        j = int(np.argmax(G))
        if G[j] > best:
            best, arg = float(G[j]), (float(cu[j]), float(cv[j]))
        keep = ub > best
        hi = float(ub[keep].max()) if keep.any() else best
        if hi - best <= rel * best:
            return best, hi, arg, (arg[0] > U - 0.5) or (abs(arg[1]) > V - 0.5)
        cu, cv = cu[keep], cv[keep]
        hu, hv = hu / 2, hv / 2
        cu = np.concatenate([cu - hu, cu - hu, cu + hu, cu + hu])
        cv = np.concatenate([cv - hv, cv + hv, cv - hv, cv + hv])
    raise RuntimeError("K not converged")


def job(win):
    from .math_note_v2_checks import monotone_bound_exact, _logloss_exact, _pava_blocks
    x, y = _data(win)
    inner, outer = (win.i_lo, win.i_hi), (win.o_lo, win.o_hi)
    half = abs(win.i_lo + win.i_hi) < 1e-12
    kw = dict(half=half, inner=inner, outer=outer)
    K_lo, K_hi, _, K_edge = K_certified(inner, outer)
    evals = []

    def status(A):
        st, rm, rp = _status(A, x, y, P_BOX, **kw)
        if st == "unresolved":
            st, rm, rp = _status(A, x, y, P_BOX, tol=1e-9, **kw)
        evals.append({"window": win.tag, "A": A, "status": st, "m_minus_lo": rm["lower"], "m_minus_hi": rm["upper"],
                      "m_plus_lo": rp["lower"], "m_plus_hi": rp["upper"],
                      "converged": rm["converged"] and rp["converged"]})
        return st, rm, rp

    A0 = round(2 * 0.20 / K_lo, 2)
    lo = hi = A0
    st, *_ = status(A0)
    steps = 0
    if st == "plus":
        while st == "plus" and lo > 0.05 and steps < 40:
            hi = lo; lo = round(lo - 0.1, 6); st, *_ = status(lo); steps += 1
    else:
        while st != "plus" and steps < 40:
            lo = hi; hi = round(hi + 0.1, 6); st, *_ = status(hi); steps += 1
    unresolved = False
    while hi - lo > WIDTH + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        st, *_ = status(mid)
        if st == "plus":
            hi = mid
        elif st == "minus":
            lo = mid
        else:
            unresolved = True
            break
    s_lo, rm_lo, rp_lo = status(lo)
    s_hi, rm_hi, rp_hi = status(hi)
    g_lo = rm_lo if s_lo == "minus" else rp_lo
    g_hi = rp_hi if s_hi == "plus" else rm_hi
    theta = (g_lo["arg_p"], g_lo["arg_q"])
    L_at_hi = float(profile([theta[0]], [theta[1]], hi, x, y)[0][0])
    branch_upper = max(g_lo["upper"], L_at_hi)
    ys = y[np.argsort(x)].astype(int)
    B_full = float(min(_logloss_exact(_pava_blocks(ys)), _logloss_exact(_pava_blocks(ys[::-1]))) / len(x))
    B_P = float(monotone_bound_exact(x, y, 4 * R2 / P_BOX))
    Ac, dl = 0.5 * (lo + hi), 2e-3
    gs = []
    for A in (Ac - dl, Ac + dl):
        st, rm, rp = _status(A, x, y, P_BOX, **kw)
        g = rp if st == "plus" else rm
        gs.append(float(gap0([g["arg_p"]], [g["arg_q"]], inner, outer)[0]))
    return ({"window": win.tag, "i_lo": win.i_lo, "i_hi": win.i_hi, "o_lo": win.o_lo, "o_hi": win.o_hi,
             "symmetric": half, "K_lo": K_lo, "K_hi": K_hi, "K_argmax_near_box_edge": K_edge,
             "A_lo": lo, "A_hi": hi, "status_lo": s_lo, "status_hi": s_hi,
             "switch_certified": (s_lo == "minus") and (s_hi == "plus") and not unresolved,
             "R_inf_lo": K_lo * lo / 2, "R_inf_hi": K_hi * hi / 2,
             "branch_loss_upper_over_bracket": branch_upper, "B_P": B_P, "B_full": B_full,
             "localised": branch_upper < min(B_P, B_full), "localisation_margin": min(B_P, B_full) - branch_upper,
             "argmin_shift": math.hypot(g_hi["arg_p"] - g_lo["arg_p"], g_hi["arg_q"] - g_lo["arg_q"]),
             "dG_dA_float": (gs[1] - gs[0]) / (2 * dl),
             "all_converged": all(e["converged"] for e in evals), "evaluations": len(evals)}, evals)


def K_base():
    K_lo, K_hi, arg, edge = K_certified((-0.8, 0.8), (1.2, 2.0))
    ls = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    out = {"K_lo": K_lo, "K_hi": K_hi, "u": arg[0], "v": arg[1], "argmax_near_box_edge": edge,
           "A_lo": float(ls.A_lo), "A_hi": float(ls.A_hi),
           "R_glob_inf_lo": K_lo * float(ls.A_lo) / 2, "R_glob_inf_hi": K_hi * float(ls.A_hi) / 2}
    pd.DataFrame([out]).to_csv(RESULTS / "limit_K_base.csv", index=False)
    print(pd.Series(out).to_string())


def main(workers=2):
    with Pool(workers) as p:
        out = p.map(job, windows(), chunksize=1)
    t = pd.DataFrame([r for r, _ in out])
    t.to_csv(RESULTS / "limit_windows.csv", index=False)
    pd.DataFrame([e for _, ev in out for e in ev]).to_csv(RESULTS / "limit_windows_evaluations.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "K_base":
        K_base()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
