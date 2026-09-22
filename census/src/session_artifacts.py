"""Producing scripts for artifacts generated in analysis sessions.

Every number the ledger verifies must have a committed script that regenerates
its artifact.  These six were first produced in ad-hoc heredocs during the
2026-09-21/22 sessions; this module is the committed producer for each.

Run `python -m src.session_artifacts` to regenerate all of them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

RESULTS = Path(__file__).resolve().parents[1] / "results"


# --- alpha_of_eps.csv (T66) ------------------------------------------------
def alpha_of_eps(n_boot: int = 600, seed: int = 0) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "nu_sweep.csv")
    med = d.groupby(["a", "budget"]).w2_abs.median().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for a, g in med.groupby("a"):
        sl = float(np.polyfit(np.log(g.budget), np.log(g.w2_abs), 1)[0])
        sub = d[d.a == a]
        bs = []
        for _ in range(n_boot):
            s = sub.iloc[rng.choice(len(sub), len(sub), replace=True)]
            m = s.groupby("budget").w2_abs.median()
            if len(m) >= 4:
                bs.append(np.polyfit(np.log(m.index), np.log(m.values), 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"a": a, "eps": a - 1.0, "alpha": sl,
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


# --- blockG_scaling.csv (T61) ----------------------------------------------
def blockG_scaling() -> pd.DataFrame:
    from .blockG_windows import WINDOWS
    from .fold1d_theorem import dip_depth
    D_inf = 4 * math.sqrt(2) / 3

    def K_of(win, iters=12, n=130):
        I = np.linspace(win.i_lo, win.i_hi, 4001)
        P = np.linspace(win.o_lo, win.o_hi, 2000)
        O = np.concatenate([P, -P])
        h = lambda z: -z + z ** 3 / 6.0
        ulo, uhi, vlo, vhi = 0.02, 8.0, -12.0, 12.0
        for _ in range(iters):
            best = (-np.inf, None)
            for u in np.linspace(ulo, uhi, n):
                for v in np.linspace(vlo, vhi, n):
                    hi_, ho = h(u * I + v), h(u * O + v)
                    g = max(ho.min() - hi_.max(), hi_.min() - ho.max())
                    if g > best[0]:
                        best = (g, (u, v))
            K, (u, v) = best
            du = (uhi - ulo) / (n - 1) * 2.5
            dv = (vhi - vlo) / (n - 1) * 2.5
            ulo, uhi, vlo, vhi = u - du, u + du, v - dv, v + dv
        return K, (u, v)

    def ghat_scaled(win, a, u0, v0, n=700, span=2.5):
        eps = a - 1.0
        se = math.sqrt(eps)
        I = torch.linspace(win.i_lo, win.i_hi, 4001, dtype=torch.float64)
        P = torch.linspace(win.o_lo, win.o_hi, 2000, dtype=torch.float64)
        O = torch.cat([P, -P])
        best = -np.inf
        us = np.linspace(u0 / span, u0 * span, n)
        vs = np.linspace(v0 - 4, v0 + 4, n)
        with torch.no_grad():
            vt = torch.tensor(vs, dtype=torch.float64).unsqueeze(1)
            for u in us:
                w1 = u * se
                vi = (w1 * I).unsqueeze(0) + math.pi + se * vt
                vo = (w1 * O).unsqueeze(0) + math.pi + se * vt
                vi = vi + a * torch.sin(vi)
                vo = vo + a * torch.sin(vo)
                plus = vo.min(1).values - vi.max(1).values
                minus = vi.min(1).values - vo.max(1).values
                best = max(best, float(torch.maximum(plus, minus).max()))
        return best

    rows = []
    for win in WINDOWS:
        K, arg = K_of(win)
        k0 = K / D_inf
        g = ghat_scaled(win, 1.02, arg[0], arg[1])
        k = g / dip_depth(1.02)
        rows.append({"window": win.tag, "K": K, "kappa_0": k0,
                     "kappa_1p02": k, "err_pct": (k / k0 - 1) * 100})
    return pd.DataFrame(rows)


# --- blockK_*.csv (T62) ----------------------------------------------------
def blockK() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RESULTS / "r_family_b.csv")
    d = d.assign(gnorm=0.4 * d.parameter.abs())
    d = d.assign(prod=d.w1_abs * d.w2_abs)
    d = d.assign(R_B=d["prod"] * d.gnorm / 2.0)

    def auc(v, s):
        v = np.asarray(v, float)
        s = np.asarray(s, bool)
        pos, neg = v[s], v[~s]
        return float(sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
                     / (len(pos) * len(neg)))

    rows = []
    for p, g in d.groupby("parameter"):
        s = g.solved.values.astype(bool)
        ps, ns = g["prod"][s], g["prod"][~s]
        lo, hi = ns.max(), ps.min()
        band = g[(g["prod"] >= min(lo, hi)) & (g["prod"] <= max(lo, hi))]
        rows.append({"alpha": p, "n": len(g), "solved": int(s.sum()),
                     "gstar_box": g.gstar.iloc[0], "gnorm": g.gnorm.iloc[0],
                     "auc_R_B": auc(g.R_B, s), "auc_prod": auc(g["prod"], s),
                     "auc_w2": auc(g.w2_abs, s), "auc_w1": auc(g.w1_abs, s),
                     "unsolved_max_prod": float(ns.max()),
                     "solved_min_prod": float(ps.min()),
                     "band_runs": len(band)})
    return d, pd.DataFrame(rows)


# --- exceptions_above_050.csv (T55) ----------------------------------------
def exceptions_above_050() -> pd.DataFrame:
    """Re-derive both >0.50 exceptions from their seeds, measuring everything."""

    from .exact_extrema import exact_gap, separates
    from .fold1d import (INNER_MAX, LR, N_PER_CLASS, OUTER_MAX, OUTER_MIN,
                         activation, logits, make_data)

    gg = pd.read_csv(RESULTS / "ghat_certified_all.csv").set_index("a")
    a = 3.0
    g_cert = float(gg.loc[a, "Ghat_certified"])
    g_rest = float(gg.loc[a, "Ghat_restricted"])
    rows = []
    for seed in (13, 82):
        f = activation("sin_family", a)
        x, y = make_data(N_PER_CLASS, seed)
        torch.manual_seed(seed)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=LR)
        for _ in range(2000):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
            opt.step()
        t = th.detach().double()
        w1, b1, w2, b2 = (float(v) for v in t)
        G, _ = exact_gap(a, w1, b1)
        ok, im, om = separates(a, (w1, b1, w2, b2))
        bias_lo = bias_hi = bias_miss = None
        bias_ok = None
        if G > 0:
            i_lo, i_hi, o_lo, o_hi = exact_gap(a, w1, b1)[1]
            if w2 > 0:
                lo_, hi_ = -w2 * o_lo, -w2 * i_hi
            else:
                lo_, hi_ = -w2 * i_lo, -w2 * o_hi
            bias_lo, bias_hi = min(lo_, hi_), max(lo_, hi_)
            bias_ok = bool(bias_lo < b2 < bias_hi)
            if not bias_ok:
                bias_miss = float(min(abs(b2 - bias_lo), abs(b2 - bias_hi)))
        n = 4001
        I = torch.linspace(-INNER_MAX, INNER_MAX, n, dtype=torch.float64)
        P = torch.linspace(OUTER_MIN, OUTER_MAX, n // 2, dtype=torch.float64)
        O = torch.cat([P, -P])
        with torch.no_grad():
            li = w2 * f(w1 * I + b1) + b2
            lo_o = w2 * f(w1 * O + b1) + b2
            pred = (logits(t.float(), x, f) > 0).double()
        vi = int((li >= 0).sum())
        vo = int((lo_o <= 0).sum())
        bad = torch.cat([I[li >= 0], O[lo_o <= 0]])
        rows.append({
            "a": a, "seed": seed, "optimizer": "adam", "budget": 2000,
            "w2_abs": abs(w2), "R_certified": abs(w2) * g_cert / 2,
            "R_restricted": abs(w2) * g_rest / 2,
            "placement_G": G, "placement_ok": bool(G > 0),
            "bias_ok": bias_ok, "bias_miss": bias_miss,
            "bias_lo": bias_lo, "bias_hi": bias_hi, "b2": b2,
            "dense_viol_inner": vi, "dense_viol_outer": vo,
            "viol_x_lo": float(bad.min()) if len(bad) else None,
            "viol_x_hi": float(bad.max()) if len(bad) else None,
            "worst_logit": float(li.max()) if vi else float(lo_o.min()),
            "sample_errors": int((pred != y.double()).sum()),
            "w1": w1, "b1": b1,
            "kind": ("bias near-miss" if G > 0 else
                     "placement failure, w1~0 constant-predictor region")})
    return pd.DataFrame(rows)


# --- blockB_fine_windows.csv (T60) -----------------------------------------
def blockB_fine_windows(step: float = 0.01) -> pd.DataFrame:
    """The a>=1.35 switch points at grid step 0.01 (the T60 refinement).

    Same frozen Block B procedure; only the grid step changes, which is
    justified by resolution alone (results/blockB_spin_dropped.md).
    """

    from .blockB_landscape import (SCREEN, best_conditional, is_degenerate,
                                   minimise, population_data)
    from .fold1d_theorem import maximum_gap

    x, y = population_data()
    out = []
    for a, lo in ((1.35, 2.8), (1.40, 2.6), (1.45, 2.4), (1.50, 2.2), (1.60, 1.9)):
        gs = maximum_gap(a, resolution=600)
        glob = None
        for w2 in np.arange(lo, lo + 1.5, step):
            b = best_conditional(a, float(w2), x, y)
            if b is not None and b["gap"] > 0:
                glob = float(w2)
                break
        cur, spin = None, None
        for w2 in np.arange(max(0.5, lo - 1.0), lo + 1.8, step):
            if cur is None:
                c = [minimise(a, float(w2), x, y, seed=s, steps=SCREEN)
                     for s in range(16)]
                c = [z for z in c if not is_degenerate(z) and z["gap"] <= 0]
                if not c:
                    continue
                c = min(c, key=lambda z: z["loss"])
                cur = minimise(a, float(w2), x, y, start=[c["w1"], c["b1"], c["b2"]])
                continue
            cur = minimise(a, float(w2), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
            if is_degenerate(cur) or cur["gap"] > 0:
                spin = float(w2)
                break
        seed_w2, fold = None, None
        for v in np.arange(lo, lo + 3.5, 0.5):
            b = best_conditional(a, float(v), x, y, restarts=24)
            if b is not None and b["gap"] > 0:
                seed_w2 = float(v) + 1.0
                break
        if seed_w2 is not None:
            cur = best_conditional(a, seed_w2, x, y, restarts=24)
            if cur is not None:
                for w2 in np.arange(seed_w2 - step, max(0.3, lo - 1.5), -step):
                    nxt = minimise(a, float(w2), x, y,
                                   start=[cur["w1"], cur["b1"], cur["b2"]])
                    if is_degenerate(nxt) or nxt["gap"] <= 0:
                        fold = float(w2)
                        break
                    cur = nxt
        out.append({"a": a, "gstar": gs, "w2_fold": fold, "w2_glob": glob,
                    "w2_spin": spin,
                    "R_fold": None if fold is None else fold * gs / 2,
                    "R_glob": None if glob is None else glob * gs / 2,
                    "R_spin": None if spin is None else spin * gs / 2})
    return pd.DataFrame(out)


# --- theorem_perplacement.csv (T56) ----------------------------------------
def theorem_perplacement() -> pd.DataFrame:
    """The theorem's first inequality at each solver's OWN placement.

    Rebuilds every recorded solver from its seed and scores the bound in that
    network's own orientation.
    """

    from .exact_extrema import exact_gap
    from .fold1d import (INNER_MAX, LR, N_PER_CLASS, OUTER_MAX, OUTER_MIN,
                         activation, logits, make_data, solves)

    sweep = pd.read_csv(RESULTS / "fold1d_sweep.csv")
    rows = []
    for _, r in sweep[sweep.solved.astype(bool)].iterrows():
        a = float(r.parameter)
        seed = int(r.seed)
        f = activation("sin_family", a)
        x, y = make_data(N_PER_CLASS, seed)
        torch.manual_seed(seed)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=LR)
        for _ in range(2000):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
            opt.step()
        t = th.detach().double()
        w1, b1, w2, b2 = (float(v) for v in t)
        if not solves(t, f):
            continue
        n = 4001
        I = torch.linspace(-INNER_MAX, INNER_MAX, n, dtype=torch.float64)
        P = torch.linspace(OUTER_MIN, OUTER_MAX, n // 2, dtype=torch.float64)
        O = torch.cat([P, -P])
        with torch.no_grad():
            li = w2 * f(w1 * I + b1) + b2
            lo_o = w2 * f(w1 * O + b1) + b2
        m = float(min(-li.max(), lo_o.min()))
        g_own, _ = exact_gap(a, w1, b1)
        rows.append({"a": a, "seed": seed, "w2": w2, "g_own": g_own,
                     "margin": m,
                     "bound_perplace": (2 * m / g_own) if g_own > 0 else np.nan,
                     "ok": bool(g_own > 0 and abs(w2) >= 2 * m / g_own)})
    return pd.DataFrame(rows)


def main() -> None:
    print("regenerating session artifacts", flush=True)
    alpha_of_eps().to_csv(RESULTS / "alpha_of_eps.csv", index=False)
    print("  alpha_of_eps.csv", flush=True)
    nb, auc = blockK()
    nb.to_csv(RESULTS / "blockK_family_b_normalised.csv", index=False)
    auc.to_csv(RESULTS / "blockK_auc.csv", index=False)
    print("  blockK_family_b_normalised.csv, blockK_auc.csv", flush=True)
    exceptions_above_050().to_csv(RESULTS / "exceptions_above_050.csv", index=False)
    print("  exceptions_above_050.csv", flush=True)
    blockG_scaling().to_csv(RESULTS / "blockG_scaling.csv", index=False)
    print("  blockG_scaling.csv", flush=True)
    blockB_fine_windows().to_csv(RESULTS / "blockB_fine_windows.csv", index=False)
    print("  blockB_fine_windows.csv", flush=True)
    theorem_perplacement().to_csv(RESULTS / "theorem_perplacement.csv", index=False)
    print("  theorem_perplacement.csv", flush=True)


if __name__ == "__main__":
    main()
