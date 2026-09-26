"""Track 3B: the band task in R^d (d = 2, 4) with a single hidden unit.  Registration: results/band_rd_registration.md.

Task.  x = (x₁, x₂, …, x_d).  x₁ exactly as the width-1 task (fold1d.make_data(200, seed): inner U(−0.8, 0.8) class 0,
outer ±U(1.2, 2.0) class 1, 200 per class); the noise coordinates x₂ … x_d are i.i.d. U(−2, 2), independent of x₁ and
of the label, drawn from their own stream numpy default_rng([seed, 3]) (a 400 × 3 draw, of which the first d − 1
columns are used, so the d = 2 noise column is the first d = 4 noise column).  y = 1 iff |x₁| > 1.

Model.  z = w₂·f_a(w·x + b₁) + b₂, w = (w₁, w_noise) ∈ R^d, f_a(t) = t + a sin t.  Parameter vector
p = (w₁, b₁, w₂, b₂, w_noise) drawn as torch.manual_seed(seed); torch.empty(d + 3).uniform_(−1, 1) (float32, then
float64): the first four draws are exactly the width-1 initialisation of the same seed.

Placement gap on the continuous support.  Over the inner class the pre-activation ranges over
[min(±0.8w₁) + b₁ − N, max(±0.8w₁) + b₁ + N] with N = 2‖w_noise‖₁ (the noise box is [−2, 2]^{d−1}); each outer
window is widened the same way.  G₊ = min_O f_a − max_I f_a, G₋ = min_I f_a − max_O f_a, exact extrema of f_a on
intervals (profiled_bnb._interval_extrema: endpoints and the critical points ±acos(−1/a) + 2πk); G = G₊ if w₂ > 0
else G₋.  With w_noise = 0 it is profiled_bnb.gap bit for bit.

    python -m src.band_rd popcheck     # design check: conditional minimiser has w_noise = 0 (validated, not certified)
    python -m src.band_rd reference    # width-1 reference range from committed phase2b data (frozen)
    python -m src.band_rd own          # width-1 own thresholds of each registered seed's x₁ sample (frozen)
    python -m src.band_rd pilot        # NON-registered pilot seeds: crossing fraction / steps / timing only
    python -m src.band_rd validate     # batched vs single-seed training on a pilot seed
    python -m src.band_rd run          # the registered runs (after the registration commit); resumable
    python -m src.band_rd score        # registered scoring + registered descriptives
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .profiled_bnb import _interval_extrema

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "band_rd"
A_VALUES = (1.30, 1.50)
D_VALUES = (2, 4)
SEEDS = tuple(range(880_000, 880_040))
SECONDARY_SEEDS = tuple(range(880_040, 880_120))
PILOT_SEEDS = tuple(range(889_000, 889_012))
NOISE_C = 2.0
LR = 1e-2
PILOT_BUDGET = 128_000
BUDGET = 64_000                     # fixed from the pilot (results/band_rd_registration.md §3)
WINDOW = 100                        # growth rate d log s/dt over the last WINDOW steps before the crossing
RECORD_EVERY = 10
TWO_PI = 2 * math.pi

# registered scoring constants
MIN_CROSS = 30
P1_FRAC = 0.90
LAG_TOL_ABS, LAG_TOL_REL = 0.01, 0.25
BRANCH_STOP = 0.20                  # P2b UNRESOLVED if > 20% of crossing runs have no converged branch
RHO_NONNEG = 0.05                   # descriptive: "non-negligible noise weight" at crossing, ‖w_noise‖₂/|w₁| >= 0.05


# ------------------------------------------------------------------------------------------ data and init
def make_data_rd(seed, d, n_per_class=200):
    from .fold1d import make_data
    x1, y = make_data(n_per_class, seed)
    noise = np.random.default_rng([seed, 3]).uniform(-NOISE_C, NOISE_C, (2 * n_per_class, 3))[:, :d - 1]
    X = np.column_stack([x1.double().numpy(), noise])
    return X, y.double().numpy()


def init_rd(seed, d):
    import torch
    torch.manual_seed(seed)
    return torch.empty(d + 3).uniform_(-1.0, 1.0).double()


# ------------------------------------------------------------------------------------------ the gap in R^d
def gap_rd(w1, b1, wn, w2, a, c=NOISE_C, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Exact gap on the continuous support, oriented by w₂ (G₊ if w₂ > 0 else G₋).  wn: (..., d−1), may be empty."""
    w1 = np.asarray(w1, float)
    b1 = np.asarray(b1, float)
    wn = np.asarray(wn, float)
    N = c * np.abs(wn).sum(axis=-1) if wn.size else np.zeros_like(w1)

    def ext(lo, hi):
        t1, t2 = w1 * lo + b1, w1 * hi + b1
        return _interval_extrema(np.minimum(t1, t2) - N, np.maximum(t1, t2) + N, a)
    imin, imax = ext(*inner)
    o1min, o1max = ext(outer[0], outer[1])
    o2min, o2max = ext(-outer[1], -outer[0])
    gp = np.minimum(o1min, o2min) - imax
    gm = imin - np.maximum(o1max, o2max)
    return np.where(np.asarray(w2) > 0, gp, gm)


def _ext_t(tlo, thi, a):
    import torch
    f = lambda t: t + a * torch.sin(t)
    c = math.acos(-1.0 / a)
    vlo, vhi = f(tlo), f(thi)
    mx, mn = torch.maximum(vlo, vhi), torch.minimum(vlo, vhi)
    tm = c + TWO_PI * torch.floor((thi - c) / TWO_PI)
    mx = torch.where(tm >= tlo, torch.maximum(mx, f(tm)), mx)
    tn = -c + TWO_PI * torch.ceil((tlo + c) / TWO_PI)
    mn = torch.where(tn <= thi, torch.minimum(mn, f(tn)), mn)
    return mn, mx


def gap_rd_t(p, a, c=NOISE_C):
    """Batched torch gap for parameter rows p = (w₁, b₁, w₂, b₂, w_noise…): returns (G, G1) where G is the R^d gap and
    G1 the gap with w_noise set to 0 (the width-1 gap of the x₁ direction), both oriented by w₂."""
    import torch
    w1, b1, w2 = p[:, 0], p[:, 1], p[:, 2]
    N = c * p[:, 4:].abs().sum(dim=1) if p.shape[1] > 4 else torch.zeros_like(w1)
    out = []
    for M in (N, torch.zeros_like(N)):
        def ext(lo, hi):
            t1, t2 = w1 * lo + b1, w1 * hi + b1
            return _ext_t(torch.minimum(t1, t2) - M, torch.maximum(t1, t2) + M, a)
        imin, imax = ext(-0.8, 0.8)
        o1min, o1max = ext(1.2, 2.0)
        o2min, o2max = ext(-2.0, -1.2)
        out.append(torch.where(w2 > 0, torch.minimum(o1min, o2min) - imax, imin - torch.maximum(o1max, o2max)))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------ thresholds
def s_bracket(a):
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    b = br[br.a.round(2) == round(a, 2)].iloc[0]
    return float(b.w2_lo), float(b.w2_hi), float(0.5 * (b.w2_lo + b.w2_hi))


# ------------------------------------------------------------------------------------------ design check (population)
def _gl_nodes(d, m):
    """Tensor Gauss–Legendre nodes/weights for U(−2, 2)^{d−1} (weights sum to 1)."""
    z, w = np.polynomial.legendre.leggauss(m)
    z, w = NOISE_C * z, w / 2
    grids = np.meshgrid(*([z] * (d - 1)), indexing="ij")
    wg = np.meshgrid(*([w] * (d - 1)), indexing="ij")
    return np.stack([g.ravel() for g in grids], 1), np.prod(np.stack([g.ravel() for g in wg], 1), axis=1)


def pop_loss_fn(a, s, d, m):
    """Population loss in h = (w₁, b₁, b₂, w_noise) at w₂ = +s: x₁ over the population grid (width2_conditional.population),
    noise by tensor Gauss–Legendre (m nodes per coordinate)."""
    import torch
    from .width2_conditional import population
    x, y = population()
    Z, W = _gl_nodes(d, m)
    X = torch.tensor(x, dtype=torch.float64)[:, None]; Y = torch.tensor(y, dtype=torch.float64)[:, None]
    Zt = torch.tensor(Z, dtype=torch.float64); Wt = torch.tensor(W, dtype=torch.float64)[None, :]

    def L(h):
        t = h[0] * X + h[1] + (Zt @ h[3:])[None, :]
        z = s * (t + a * torch.sin(t)) + h[2]
        l = torch.nn.functional.softplus(z) - Y * z
        return (l * Wt).sum(dim=1).mean()
    return L


def newton_min(L, h0, fixed=None, iters=200, gtol=1e-11):
    """Damped (Levenberg) Newton on L over the free coordinates (fixed: boolean mask of coordinates held)."""
    import torch
    h = torch.tensor(np.asarray(h0, float), dtype=torch.float64)
    free = np.ones(len(h), bool) if fixed is None else ~np.asarray(fixed)
    fi = torch.tensor(np.nonzero(free)[0])

    def sub(v):
        hh = h.clone(); hh[fi] = v
        return L(hh)
    v = h[fi].clone()
    mu, f0 = 1e-6, float(sub(v))
    for _ in range(iters):
        vv = v.clone().requires_grad_(True)
        g = torch.autograd.grad(sub(vv), vv)[0]
        if float(g.abs().max()) < gtol:
            break
        H = torch.autograd.functional.hessian(sub, v)
        ok = False
        for _ in range(40):
            dlt = torch.linalg.solve(H + mu * torch.eye(len(v), dtype=torch.float64), -g)
            f1 = float(sub(v + dlt))
            if f1 <= f0:
                v, f0, ok, mu = v + dlt, f1, True, max(mu / 10, 1e-14)
                break
            mu *= 10
        if not ok:
            break
    vv = v.clone().requires_grad_(True)
    g = torch.autograd.grad(sub(vv), vv)[0]
    h[fi] = v
    return h.numpy(), f0, float(g.abs().max())


def popcheck(m2=32, m4=8, n_rand=8, seed=0):
    """VALIDATED, NOT CERTIFIED.  At s = both ends of the certified width-1 bracket, for d = 2, 4 and a = 1.30, 1.50:
    (i) the width-1 population branch θ₁(s) (lag_law.branch_point continued from the certified switch z*) and its gap
    sign; (ii) the noise block of the Hessian at (θ₁, w_noise = 0) equals E[x_n²]·∂²L/∂b₁² (> 0) and the w₁-noise and
    b₂-noise cross terms vanish; (iii) multistart Newton in all d + 2 hidden coordinates (starts: θ₁ with random w_noise
    of norm 0.05, 0.2, 0.5 and random starts) — no start ends below L₁*(s) with w_noise ≠ 0; (iv) the profile
    min_{w₁,b₁,b₂} L(w_noise = r·u) increases with r."""
    import torch
    torch.set_num_threads(1)
    from .lag_law import branch_point, gap as gap1
    from .width2_conditional import population
    x, y = population()
    kt = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    rng = np.random.default_rng(seed)
    rows, prof = [], []
    for a in A_VALUES:
        r = kt[kt.a.round(2) == round(a, 2)].iloc[0]
        z = np.array(json.loads(r.z_star))
        lo, hi, _ = s_bracket(a)
        for s in (lo, hi):
            zs = z.copy()
            for f in np.linspace(r.s_star, s, 6)[1:]:
                zs, gr = branch_point(zs, f, a, x, y)
            G1 = gap1(zs[0], zs[1], a)
            for d in D_VALUES:
                L = pop_loss_fn(a, s, d, m2 if d == 2 else m4)
                h1 = np.r_[zs, np.zeros(d - 1)]
                L1 = float(L(torch.tensor(h1)))
                H = torch.autograd.functional.hessian(L, torch.tensor(h1)).numpy()
                noise_block = H[3:, 3:]
                pred_block = (NOISE_C ** 2 / 3) * H[1, 1] * np.eye(d - 1)
                starts = []
                for nrm in (0.05, 0.2, 0.5):
                    for _ in range(2):
                        u = rng.standard_normal(d - 1); u /= np.linalg.norm(u)
                        starts.append(("branch+noise", np.r_[zs, nrm * u]))
                for _ in range(n_rand):
                    starts.append(("random", np.r_[rng.uniform(-3, 3), rng.uniform(0, TWO_PI), rng.uniform(-3 * s, 3 * s),
                                                   rng.uniform(-0.5, 0.5, d - 1)]))
                for kind, h0 in starts:
                    hf, Lf, gm = newton_min(L, h0)
                    wn = hf[3:]
                    rows.append({"a": a, "s": s, "d": d, "start": kind, "L_final_minus_L1star": Lf - L1,
                                 "wnoise_norm_final": float(np.linalg.norm(wn)), "grad_max": gm,
                                 "G_rd_final": float(gap_rd(hf[0], hf[1], wn, 1.0, a)), "w1": hf[0], "b1": hf[1],
                                 "G1_branch": G1, "L1star": L1, "branch_grad": gr,
                                 "noise_block_rel_err": float(np.abs(noise_block - pred_block).max() / abs(pred_block[0, 0])),
                                 "noise_block_min_eig": float(np.linalg.eigvalsh(noise_block).min()),
                                 "cross_max_abs": float(np.abs(H[:3, 3:]).max())})
                    print({k: rows[-1][k] for k in ("a", "s", "d", "start", "L_final_minus_L1star", "wnoise_norm_final")},
                          flush=True)
                dirs = [np.eye(d - 1)[0]] + ([np.ones(d - 1) / math.sqrt(d - 1)] if d > 2 else [])
                for j, u in enumerate(dirs):
                    hw = h1.copy()
                    for rr in (0.0, 0.005, 0.02, 0.05, 0.1, 0.2):
                        h0 = np.r_[hw[:3], rr * u]
                        hf, Lf, gm = newton_min(L, h0, fixed=np.r_[np.zeros(3, bool), np.ones(d - 1, bool)])
                        hw = hf
                        prof.append({"a": a, "s": s, "d": d, "direction": j, "r": rr, "L_profile_minus_L1star": Lf - L1,
                                     "grad_max": gm})
    OUT.mkdir(exist_ok=True)
    P = pd.DataFrame(rows); P.to_csv(OUT / "popcheck_starts.csv", index=False)
    Q = pd.DataFrame(prof); Q.to_csv(OUT / "popcheck_profile.csv", index=False)
    summ = []
    for (a, s, d), g in P.groupby(["a", "s", "d"]):
        q = Q[(Q.a == a) & (Q.s == s) & (Q.d == d)]
        mono = all(np.all(np.diff(h.sort_values("r").L_profile_minus_L1star.values) > 0) for _, h in q.groupby("direction"))
        below = g[g.L_final_minus_L1star < -1e-10]
        summ.append({"a": a, "s": s, "d": d, "G1_branch": float(g.G1_branch.iloc[0]),
                     "placed_width1": bool(g.G1_branch.iloc[0] > 0),
                     "n_starts": len(g), "n_below_L1star": len(below),
                     "n_reach_L1star": int((g.L_final_minus_L1star.abs() < 1e-10).sum()),
                     "max_wnoise_at_L1star": float(g[g.L_final_minus_L1star.abs() < 1e-10].wnoise_norm_final.max()),
                     "noise_block_rel_err": float(g.noise_block_rel_err.iloc[0]),
                     "noise_block_min_eig": float(g.noise_block_min_eig.iloc[0]),
                     "cross_max_abs": float(g.cross_max_abs.iloc[0]), "profile_increasing": mono,
                     "n_end_above_L1star": int((g.L_final_minus_L1star > 1e-10).sum()),
                     "n_end_above_with_wnoise_gt_0.1": int(((g.L_final_minus_L1star > 1e-10)
                                                             & (g.wnoise_norm_final > 0.1)).sum()),
                     "min_excess_loss_noise_stationary": float(g[g.wnoise_norm_final > 0.1].L_final_minus_L1star.min())
                     if (g.wnoise_norm_final > 0.1).any() else np.nan,
                     "max_grad_at_end": float(g.grad_max.max())})
    S = pd.DataFrame(summ); S.to_csv(OUT / "popcheck_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    return S


# ------------------------------------------------------------------------------------------ frozen inputs
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze(name):
    h = _sha(OUT / name)
    (OUT / (name + ".sha256")).write_text(h + "\n")
    return h


def check_frozen(name):
    h = _sha(OUT / name)
    assert h == (OUT / (name + ".sha256")).read_text().strip(), f"STOP: frozen input {name} changed"
    return h


def width1_reference():
    """Width-1 free-training reference (phase2b, budget 32,000, Adam lr 0.01, seeds 0–39): crossing residual vs the
    certified population threshold (bracket midpoint), r = |w₂|_c / s_mid − 1: quartiles and fraction at or above s_lo."""
    d = pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=["a", "budget", "seed", "w2", "crossing"])
    c = d[(d.crossing == True) & (d.budget == 32_000)]  # noqa: E712
    ref = {}
    for a in A_VALUES:
        lo, hi, mid = s_bracket(a)
        s = c[c.a.round(2) == round(a, 2)].w2.abs().to_numpy()
        r = s / mid - 1
        q1, med, q3 = (float(v) for v in np.percentile(r, [25, 50, 75]))
        ref[f"{a:.2f}"] = {"n_crossing": int(len(s)), "n_runs": 40, "s_lo": lo, "s_hi": hi, "s_mid": mid, "q1": q1,
                           "median": med, "q3": q3, "frac_at_or_above_s_lo": float((s >= lo).mean())}
    OUT.mkdir(exist_ok=True)
    (OUT / "width1_reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    print(json.dumps(ref, indent=1), "\nsha256", _freeze("width1_reference.json"))
    return ref


def _own_job(args):
    os.nice(15)
    from .own_threshold import _pop, own_threshold
    a, seed = args
    return own_threshold(a, seed, _pop(a)[1])


def own_x1():
    """Width-1 own-sample thresholds (own_threshold.own_threshold) of each registered seed's x₁ sample; frozen."""
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "own_x1_parts.csv"
    done = set() if not f.exists() else {(round(r.a, 2), r.seed) for r in pd.read_csv(f).itertuples()}
    jobs = [(a, s) for a in A_VALUES for s in SEEDS if (round(a, 2), s) not in done]
    with get_context("spawn").Pool(1) as pool:
        for r in pool.imap_unordered(_own_job, jobs):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(r, flush=True)
    d = pd.read_csv(f).sort_values(["a", "seed"])
    d["w2_own"] = 0.5 * (d.w2_lo + d.w2_hi)
    d.to_csv(OUT / "own_x1_frozen.csv", index=False)
    print(len(d), int(d.w2_own.isna().sum()), _freeze("own_x1_frozen.csv"))


# ------------------------------------------------------------------------------------------ training
def _forward(p, X, a):
    import torch
    w = torch.cat([p[:, 0:1], p[:, 4:]], dim=1)                          # (n, d) in x-coordinate order
    t = (X * w[:, None, :]).sum(dim=-1) + p[:, 1:2]
    return p[:, 2:3] * (t + a * torch.sin(t)) + p[:, 3:4]


def train_batch(a, d, seeds, budget, record_every=RECORD_EVERY, s_half=None, stop_when_done=True, p0=None):
    """Free Adam training (all d + 3 parameters, lr 0.01, full batch), vectorised over seeds (Adam is coordinate-wise
    and the per-seed losses are independent).  Crossing: the first step t >= 1 (after the t-th update) with G > 0 (R^d
    gap, oriented by w₂).  A run with G > 0 at initialisation is 'placed at init' and has no crossing.
    Returns (records, trace): per-seed dicts and a (steps/record_every, n, 6) float64 trace of
    (step, |w₁|, ‖w_noise‖₂, |w₂|, G, G1)."""
    import torch
    torch.set_num_threads(1)
    XY = [make_data_rd(s, d) for s in seeds]
    X = torch.tensor(np.stack([q[0] for q in XY])); Y = torch.tensor(np.stack([q[1] for q in XY]))
    p = (torch.stack([init_rd(s, d) for s in seeds]) if p0 is None
         else torch.tensor(np.asarray(p0, float))).clone().requires_grad_(True)       # p0: tests only
    opt = torch.optim.Adam([p], lr=LR)
    n = len(seeds)
    with torch.no_grad():
        G0, G10 = gap_rd_t(p, a)
    rec = []
    for i, s in enumerate(seeds):
        q = p.detach()[i]
        rec.append({"seed": s, "a": a, "d": d, "placed_at_init": bool(G0[i] > 0), "crossed": False, "step": np.nan,
                    "s_cross": np.nan, "s_cross_minus_window": np.nan,
                    "G1_first_step": 0 if bool(G10[i] > 0) else np.nan, "s_at_G1_first": float(q[2].abs()) if bool(G10[i] > 0) else np.nan,
                    "half_step": np.nan, "w1_init": float(q[0]), "wnoise_l2_init": float(q[4:].norm()),
                    "rho_init": float(q[4:].norm() / q[0].abs())})
    done = G0 > 0
    g1done = G10 > 0
    halfdone = torch.zeros(n, dtype=torch.bool)
    ring = torch.zeros(WINDOW + 1, n, dtype=torch.float64)
    ring[0] = p.detach()[:, 2].abs()
    trace = []
    t = 0
    for t in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        z = _forward(p, X, a)
        torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none").mean(dim=1).sum().backward()
        opt.step()
        with torch.no_grad():
            q = p.detach()
            s_abs = q[:, 2].abs()
            ring[t % (WINDOW + 1)] = s_abs
            G, G1 = gap_rd_t(q, a)
            wl2 = q[:, 4:].norm(dim=1) if d > 1 else torch.zeros(n, dtype=torch.float64)
            if t % record_every == 0:
                trace.append(torch.stack([torch.full((n,), float(t), dtype=torch.float64), q[:, 0].abs(), wl2, s_abs, G, G1], 1))
            h1 = (G1 > 0) & ~g1done
            for i in torch.nonzero(h1).flatten().tolist():
                rec[i].update(G1_first_step=t, s_at_G1_first=float(s_abs[i]))
            g1done |= h1
            if s_half is not None:
                hh = (s_abs >= s_half) & ~halfdone
                for i in torch.nonzero(hh).flatten().tolist():
                    rec[i].update(half_step=t, w1_half=float(q[i, 0]), wnoise_l2_half=float(wl2[i]),
                                  rho_half=float(wl2[i] / q[i, 0].abs()))
                halfdone |= hh
            hit = (G > 0) & ~done
            if hit.any():
                st = opt.state[p]
                vh = st["exp_avg_sq"] / (1 - 0.999 ** int(st["step"]))
                for i in torch.nonzero(hit).flatten().tolist():
                    rec[i].update(crossed=True, step=t, s_cross=float(s_abs[i]), G_at_cross=float(G[i]),
                                  G1_at_cross=float(G1[i]), wnoise_l2_cross=float(wl2[i]),
                                  wnoise_l1_cross=float(q[i, 4:].abs().sum()), rho_cross=float(wl2[i] / q[i, 0].abs()),
                                  s_cross_minus_window=float(ring[(t - WINDOW) % (WINDOW + 1), i]) if t >= WINDOW else np.nan,
                                  p_cross=json.dumps([float(v) for v in q[i]]),
                                  vhat_cross=json.dumps([float(v) for v in vh[i]]))
                done |= hit
            if stop_when_done and bool(done.all()) and (s_half is None or bool(halfdone[~(G0 > 0)].all())):
                break
    q = p.detach()
    for i in range(n):
        rec[i].update(steps_run=t, p_end=json.dumps([float(v) for v in q[i]]))
    tr = torch.stack(trace).numpy() if trace else np.zeros((0, n, 6))
    return rec, tr


def train_single(a, d, seed, steps):
    """Independent single-seed implementation (parameter vector of shape (d + 3,)) for validating the batching:
    parameters after `steps` updates and the crossing step (if any)."""
    import torch
    torch.set_num_threads(1)
    Xn, Yn = make_data_rd(seed, d)
    X, Y = torch.tensor(Xn), torch.tensor(Yn)
    p = init_rd(seed, d).clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=LR)
    cross = None
    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        w = torch.cat([p[0:1], p[4:]])
        tt = (X * w[None, :]).sum(dim=-1) + p[1]
        z = p[2] * (tt + a * torch.sin(tt)) + p[3]
        torch.nn.functional.binary_cross_entropy_with_logits(z, Y).backward()
        opt.step()
        q = p.detach().numpy()
        if cross is None and gap_rd(q[0], q[1], q[4:], q[2], a) > 0:
            cross = (t, abs(float(q[2])))
    return p.detach().numpy(), cross


# ------------------------------------------------------------------------------------------ per-run lag quantities
def branch_rd(a, X, Y, p):
    """Damped Newton on the hidden vector h = (w₁, b₁, b₂, w_noise) at w₂ fixed on the run's own sample
    (residual_timescale.branch_hessian generalised).  Returns (h*, H(h*), grad max, converged)."""
    import torch
    Xt, Yt = torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)
    w2 = float(p[2])

    def L(h):
        w = torch.cat([h[0:1], h[3:]])
        t = (Xt * w[None, :]).sum(dim=-1) + h[1]
        return torch.nn.functional.binary_cross_entropy_with_logits(w2 * (t + a * torch.sin(t)) + h[2], Yt)
    h = torch.tensor(np.r_[p[0], p[1], p[3], p[4:]], dtype=torch.float64)
    mu, f0 = 1e-3, float(L(h))
    for _ in range(200):
        hh = h.clone().requires_grad_(True)
        g = torch.autograd.grad(L(hh), hh)[0]
        if float(g.abs().max()) < 1e-12:
            break
        H = torch.autograd.functional.hessian(L, h)
        ok = False
        for _ in range(30):
            dl = torch.linalg.solve(H + mu * torch.eye(len(h), dtype=torch.float64) * max(1.0, float(H.diag().abs().max())), -g)
            f1 = float(L(h + dl))
            if f1 < f0:
                h, f0, ok, mu = h + dl, f1, True, max(mu / 3, 1e-12)
                break
            mu *= 10
        if not ok:
            break
    hh = h.clone().requires_grad_(True)
    g = torch.autograd.grad(L(hh), hh)[0]
    H = torch.autograd.functional.hessian(L, h)
    return h.numpy(), H.numpy(), float(g.abs().max()), float(g.abs().max()) < 1e-8


def _landscape():
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    return {round(r.a, 2): {"H": np.array(json.loads(r.H)), "tan": np.array(json.loads(r.tangent)),
                            "dG": np.array(json.loads(r.gradG)), "z": np.array(json.loads(r.z_star)),
                            "p": np.array(json.loads(r.p_median))} for r in k.itertuples()}


def lag_quantities(a, d, rec, L):
    """Registered per-run lag-law prediction for a crossing run:
      growth  = log(s_c / s_{c−100}) / 100;
      relax   = lr·λ_min(D^{-1/2} H_sig D^{-1/2}) (residual_timescale.relax_rate), H_sig the (w₁, b₁, b₂) block of the
                Hessian at the own-sample branch in all hidden coordinates at w₂ fixed, D from Adam's v̂ of (w₁, b₁, b₂);
      χ       = growth / relax;  κ_k = lag_law.kappa_k(width-1 population H, θ*′, ∇G, median P, k) with k the run's
                winding (canonical b₁, lag_law.canonical_b1);  pred_r = κ_k·χ."""
    from .lag_law import canonical_b1, kappa_k
    from .residual_timescale import relax_rate
    p = np.array(json.loads(rec["p_cross"])); vh = np.array(json.loads(rec["vhat_cross"]))
    X, Y = make_data_rd(int(rec["seed"]), d)
    h, H, gmax, conv = branch_rd(a, X, Y, p)
    relax = relax_rate(H[:3, :3], vh[[0, 1, 3]])
    relax_full = relax_rate(H, vh[[0, 1, 3] + list(range(4, len(p)))])
    growth = math.log(rec["s_cross"] / rec["s_cross_minus_window"]) / WINDOW if np.isfinite(rec["s_cross_minus_window"]) else np.nan
    chi = growth / relax if relax > 0 else np.nan
    b1c, w1c = canonical_b1(p[0], p[1], p[2])
    k = int(round((b1c - L["z"][1]) / TWO_PI))
    kap = kappa_k(L["H"], L["tan"], L["dG"], L["p"], k)
    return {"branch_converged": conv, "branch_grad": gmax, "branch_wnoise_l2": float(np.linalg.norm(h[3:])),
            "branch_dist": float(np.abs(h - np.r_[p[0], p[1], p[3], p[4:]]).max()),
            "growth": growth, "relax_sig": relax, "relax_full": relax_full, "chi": chi, "winding_k": k,
            "mirror": bool(np.sign(w1c) != np.sign(L["z"][0])), "kappa_k": kap, "pred_r": kap * chi,
            "G_numpy_at_cross": float(gap_rd(p[0], p[1], p[4:], p[2], a))}


# ------------------------------------------------------------------------------------------ registered scoring rules
def score_cell(s_cross, r_pop, obs_own, pred, usable, s_lo, ref, g_numpy):
    """One (d, a) cell.  s_cross, r_pop: all crossing runs (placed-at-init runs excluded).  obs_own, pred, usable: per
    crossing run (usable = finite χ > 0, converged branch, finite own threshold).  ref: width-1 quartiles {q1, q3}.
    g_numpy: the R^d gap recomputed in numpy at each recorded crossing state (validity: all > 0, else STOP).
      P1:  fraction of crossing runs with s_c >= s_lo >= 0.90.
      P2a: q1 <= median(r_pop) <= q3 (width-1 interquartile range of the crossing residual vs the population threshold).
      P2b: |median(obs_own) − median(pred)| <= max(0.01, 0.25·|median(pred)|) over usable runs.
    Any rule with < 30 crossing runs is UNRESOLVED; P2b is UNRESOLVED with < 30 usable runs or with non-converged branches
    in > 20% of crossing runs."""
    s_cross, r_pop, obs_own, pred = (np.asarray(v, float) for v in (s_cross, r_pop, obs_own, pred))
    usable = np.asarray(usable, bool)
    n = len(s_cross)
    out = {"n_crossing": n}
    if n and not np.all(np.asarray(g_numpy, float) > 0):
        return {**out, "P1": "STOP", "P2a": "STOP", "P2b": "STOP", "stop_reason": "recomputed gap <= 0 at a crossing"}
    if n < MIN_CROSS:
        return {**out, "P1": "UNRESOLVED", "P2a": "UNRESOLVED", "P2b": "UNRESOLVED"}
    frac = float((s_cross >= s_lo).mean())
    med = float(np.median(r_pop))
    out.update(frac_at_or_above_s_lo=frac, P1="PASS" if frac >= P1_FRAC else "FAIL",
               median_r_pop=med, ref_q1=ref["q1"], ref_q3=ref["q3"],
               P2a="PASS" if ref["q1"] <= med <= ref["q3"] else "FAIL",
               P2a_side="inside" if ref["q1"] <= med <= ref["q3"] else ("above" if med > ref["q3"] else "below"))
    nu = int(usable.sum())
    out["n_usable_P2b"] = nu
    return {**out, **score_lag(obs_own[usable], pred[usable], n, n - nu)}


def score_lag(obs, pred, n_cross, n_unusable_branch):
    if n_cross and n_unusable_branch / n_cross > BRANCH_STOP:
        return {"P2b": "UNRESOLVED", "P2b_reason": "branch not converged / chi undefined in > 20% of crossing runs"}
    if len(obs) < MIN_CROSS:
        return {"P2b": "UNRESOLVED", "P2b_reason": "< 30 usable runs"}
    pm, om = float(np.median(pred)), float(np.median(obs))
    tol = max(LAG_TOL_ABS, LAG_TOL_REL * abs(pm))
    return {"median_pred_r": pm, "median_obs_r_own": om, "tol": tol, "obs_minus_pred": om - pm,
            "P2b": "PASS" if abs(om - pm) <= tol else "FAIL"}


# ------------------------------------------------------------------------------------------ pilot / validation / runs
def pilot():
    """NON-registered pilot seeds: crossing counts, crossing steps and wall time only (no crossing scale is written or
    printed).  Disclosed in the registration; used only to fix the budget."""
    os.nice(15)
    OUT.mkdir(exist_ok=True)
    rows = []
    for d in D_VALUES:
        for a in A_VALUES:
            t0 = time.time()
            rec, _ = train_batch(a, d, PILOT_SEEDS, PILOT_BUDGET, record_every=1000)
            dt = time.time() - t0
            steps = [r["step"] for r in rec if r["crossed"]]
            rows.append({"d": d, "a": a, "n": len(rec), "n_crossed": len(steps),
                         "n_placed_at_init": sum(r["placed_at_init"] for r in rec),
                         "max_cross_step": max(steps) if steps else np.nan,
                         "crossing_steps": json.dumps(sorted(int(s) for s in steps)),
                         "steps_run": rec[0]["steps_run"], "seconds": dt})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "pilot.csv", index=False)


def pilot_diag(steps=40_000):
    """Pilot diagnosis (NON-registered seeds, d = 4): the final state of every pilot run that has not crossed after
    `steps` steps — |w₁|, ‖w_noise‖₂, |w₂|, G, G1, and the change of these over the last 36,000 steps."""
    os.nice(15)
    rows = []
    for a in A_VALUES:
        rec, tr = train_batch(a, 4, PILOT_SEEDS, steps, record_every=4000)
        for i, r in enumerate(rec):
            if r["crossed"]:
                continue
            p = np.array(json.loads(r["p_end"]))
            rows.append({"a": a, "d": 4, "seed": r["seed"], "steps": r["steps_run"], "w1_abs": abs(p[0]),
                         "wnoise_l2": float(np.linalg.norm(p[4:])), "w2_abs": abs(p[2]),
                         "G": float(gap_rd(p[0], p[1], p[4:], p[2], a)), "G1": float(gap_rd(p[0], p[1], p[4:0], p[2], a)),
                         "max_change_last_36000": float(np.abs(tr[-1, i, 1:4] - tr[0, i, 1:4]).max())})
    pd.DataFrame(rows).to_csv(OUT / "pilot_stuck.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def validate(seed=PILOT_SEEDS[0], steps=3000):
    """Batched (the pilot seeds as a batch) vs single-seed training for one pilot seed, d = 2 and 4, both a."""
    os.nice(15)
    rows = []
    for d in D_VALUES:
        for a in A_VALUES:
            rec, _ = train_batch(a, d, PILOT_SEEDS, steps, record_every=steps, stop_when_done=False)
            i = PILOT_SEEDS.index(seed)
            pb = np.array(json.loads(rec[i]["p_end"]))
            ps, cross = train_single(a, d, seed, steps)
            rows.append({"d": d, "a": a, "seed": seed, "steps": steps, "max_abs_diff": float(np.abs(pb - ps).max()),
                         "bitwise_equal": bool(np.array_equal(pb, ps)),
                         "cross_step_batched": rec[i]["step"], "cross_step_single": cross[0] if cross else np.nan,
                         "s_cross_equal": (cross is None and not rec[i]["crossed"]) or
                                          (cross is not None and cross[1] == rec[i]["s_cross"])})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "validate_batching.csv", index=False)


ARMS = (("primary", SEEDS, D_VALUES + (1,)), ("secondary", SECONDARY_SEEDS, D_VALUES))


def run():
    """Registered runs, in this order: primary (40 seeds; d = 2, 4 and the descriptive width-1 control d = 1), then the
    secondary arm (80 further seeds; d = 2, 4).  Resumable: an (arm, d, a) cell already in runs.jsonl is skipped;
    traces go to traces_{arm}_d{d}_a{a}.npz."""
    os.nice(15)
    for name in ("width1_reference.json", "own_x1_frozen.csv"):
        check_frozen(name)
    OUT.mkdir(exist_ok=True)
    f = OUT / "runs.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line); done.add((r["arm"], r["d"], round(r["a"], 2)))
    for arm, seeds, dims in ARMS:
        for d in dims:
            for a in A_VALUES:
                if (arm, d, round(a, 2)) in done:
                    continue
                t0 = time.time()
                rec, tr = train_batch(a, d, seeds, BUDGET, s_half=0.5 * s_bracket(a)[2])
                np.savez_compressed(OUT / f"traces_{arm}_d{d}_a{a:.2f}.npz", trace=tr, seeds=np.array(seeds))
                with open(f, "a") as fh:
                    for r in rec:
                        fh.write(json.dumps({"arm": arm, **r}, default=float) + "\n")
                print(arm, d, a, f"{time.time() - t0:.0f}s", flush=True)      # no outcome is printed


def load_runs():
    rows = [json.loads(l) for l in (OUT / "runs.jsonl").read_text().splitlines()]
    d = pd.DataFrame(rows)
    d["a"] = d.a.round(2)
    return d


def score():
    """Registered scoring (results/band_rd_registration.md) and the registered descriptives.
    Primary: 40 seeds per (d, a) cell, P1, P2a, P2b.  Secondary: the 120 seeds pooled (primary + secondary arm) per
    (d, a) cell, P1 and P2a (own thresholds are frozen for the primary seeds only, so P2b is not scored there).
    The d = 1 control is scored with the same functions for description only."""
    ref = json.loads((OUT / "width1_reference.json").read_text())
    hashes = {n: check_frozen(n) for n in ("width1_reference.json", "own_x1_frozen.csv")}
    own = pd.read_csv(OUT / "own_x1_frozen.csv"); own["a"] = own.a.round(2)
    L = _landscape()
    R = load_runs().merge(own[["a", "seed", "w2_own"]], on=["a", "seed"], how="left")
    rows = []
    for r in R.itertuples():
        rec = r._asdict()
        base = {k: rec[k] for k in ("arm", "d", "a", "seed", "placed_at_init", "crossed", "step", "s_cross", "w2_own",
                                    "G1_first_step", "s_at_G1_first", "half_step", "rho_init", "steps_run")}
        base.update({k: rec.get(k, np.nan) for k in ("rho_half", "rho_cross", "wnoise_l2_cross", "wnoise_l1_cross",
                                                      "G_at_cross", "G1_at_cross")})
        pe = np.array(json.loads(rec["p_end"]))
        base.update(w1_end=abs(pe[0]), wnoise_l2_end=float(np.linalg.norm(pe[4:])), s_end=abs(pe[2]),
                    rho_end=float(np.linalg.norm(pe[4:]) / abs(pe[0])), G_end=float(gap_rd(pe[0], pe[1], pe[4:], pe[2], r.a)),
                    G1_end=float(gap_rd(pe[0], pe[1], pe[4:0], pe[2], r.a)))
        if r.crossed:
            lo, hi, mid = s_bracket(r.a)
            base.update(r_pop=r.s_cross / mid - 1, obs_r_own=r.s_cross / r.w2_own - 1,
                        **lag_quantities(r.a, r.d, rec, L[r.a]))
            w1c = abs(json.loads(rec["p_cross"])[0])
            base["widening_rel"] = NOISE_C * base["wnoise_l1_cross"] / (0.8 * w1c) if r.d > 1 else 0.0
        rows.append(base)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "scored_runs.csv", index=False)
    verdicts, desc = [], []
    for arm in ("primary", "secondary"):
        Pa = P[P.arm == "primary"] if arm == "primary" else P[P.d > 1]
        for (d, a), g in Pa.groupby(["d", "a"]):
            c = g[g.crossed.astype(bool)]
            usable = (np.isfinite(c.chi) & (c.chi > 0) & c.branch_converged.astype(bool)
                      & np.isfinite(c.obs_r_own)).to_numpy()
            lo, hi, mid = s_bracket(a)
            sc = score_cell(c.s_cross, c.r_pop, c.obs_r_own, c.pred_r, usable, lo, ref[f"{a:.2f}"], c.G_numpy_at_cross)
            if arm == "secondary":
                sc = {k: v for k, v in sc.items() if not (k.startswith("P2b") or k in
                      ("median_pred_r", "median_obs_r_own", "tol", "obs_minus_pred", "n_usable_P2b"))}
                sc["P2b"] = "not scored (secondary)"
            role = ("registered primary" if arm == "primary" else "registered secondary (pooled 120 seeds)") if d > 1 \
                else "descriptive width-1 control (not scored)"
            verdicts.append({"arm": arm, "d": int(d), "a": a, "role": role, "n_runs": len(g),
                             "n_placed_at_init": int(g.placed_at_init.sum()), **sc})
            desc.append({"arm": arm, **describe_cell(g, c, d, a)})
    V = pd.DataFrame(verdicts); D = pd.DataFrame(desc)
    V.to_csv(OUT / "verdicts.csv", index=False); D.to_csv(OUT / "descriptives.csv", index=False)
    (OUT / "verdicts.json").write_text(json.dumps({"frozen_inputs_sha256": hashes, "cells": V.to_dict("records"),
                                                    "descriptives": D.to_dict("records")}, indent=1, default=float) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(V.to_string(index=False)); print(D.T.to_string())
    return V, D


def _trace(arm, d, a):
    z = np.load(OUT / f"traces_{arm}_d{d}_a{a:.2f}.npz")
    return z["trace"], list(z["seeds"])


def describe_cell(g, c, d, a):
    """Registered descriptives (not scored): how the unit finds the direction."""
    q = lambda v, p: float(np.nanpercentile(v, p)) if len(v) and np.isfinite(v).any() else np.nan
    out = {"d": int(d), "a": a, "n_runs": len(g), "n_crossing": len(c)}
    for name, col in (("rho_init", g.rho_init), ("rho_half", g.get("rho_half", pd.Series(dtype=float))),
                      ("rho_cross", c.rho_cross)):
        v = pd.to_numeric(col, errors="coerce").to_numpy(float)
        out.update({f"{name}_q25": q(v, 25), f"{name}_median": q(v, 50), f"{name}_q75": q(v, 75)})
    nc = g[~g.crossed.astype(bool) & ~g.placed_at_init.astype(bool)]
    out.update({"n_not_crossed": len(nc), "n_not_crossed_rho_end_gt_1": int((nc.rho_end > 1).sum()),
                "not_crossed_w1_end_median": q(nc.w1_end.to_numpy(float), 50),
                "not_crossed_wnoise_end_median": q(nc.wnoise_l2_end.to_numpy(float), 50),
                "not_crossed_s_end_median": q(nc.s_end.to_numpy(float), 50)})
    if len(c):
        out["n_cross_rho_ge_0.05"] = int((c.rho_cross >= RHO_NONNEG).sum())
        out["max_rho_cross"] = float(c.rho_cross.max())
        delay = (c.step - c.G1_first_step).to_numpy(float)
        out.update(noise_delay_steps_median=q(delay, 50), noise_delay_steps_q75=q(delay, 75),
                   noise_delay_steps_max=float(np.nanmax(delay)),
                   noise_delay_scale_median=q((c.s_cross / c.s_at_G1_first - 1).to_numpy(float), 50),
                   frac_cross_same_step_as_G1=float((delay == 0).mean()),
                   widening_rel_median=q(c.widening_rel.to_numpy(float), 50),
                   widening_rel_max=float(c.widening_rel.max()),
                   crossing_step_median=q(c.step.to_numpy(float), 50), crossing_step_max=float(c.step.max()),
                   median_r_pop=q(c.r_pop.to_numpy(float), 50), median_obs_r_own=q(c.obs_r_own.to_numpy(float), 50),
                   median_pred_r=q(c.pred_r.to_numpy(float), 50), median_chi=q(c.chi.to_numpy(float), 50),
                   winding_counts=json.dumps({str(k): int(v) for k, v in c.winding_k.value_counts().sort_index().items()}),
                   n_mirror=int(c.mirror.sum()), median_branch_wnoise_l2=q(c.branch_wnoise_l2.to_numpy(float), 50),
                   median_relax_full_over_sig=q((c.relax_full / c.relax_sig).to_numpy(float), 50))
    if d > 1 and len(c):
        first = []
        for arm, ca in c.groupby("arm"):
            tr, seeds = _trace(arm, d, a)                                    # (T, n, 6): step, |w1|, ‖wn‖, |w2|, G, G1
            rho = tr[:, :, 2] / tr[:, :, 1]
            for r in ca.itertuples():
                i = seeds.index(r.seed)
                below = np.nonzero(rho[:, i] < 0.1)[0]
                first.append(tr[below[0], i, 0] / r.step if len(below) else np.nan)
        first = np.array(first, float)
        out.update({"frac_rho_below_0.1_before_cross": float(np.mean(first < 1.0)),
                    "t_rho_0.1_over_t_cross_median": q(first, 50)})
    return out


if __name__ == "__main__":
    {"popcheck": popcheck, "reference": width1_reference, "own": own_x1, "pilot": pilot, "pilot_diag": pilot_diag, "validate": validate,
     "run": run, "score": score}[sys.argv[1]]()
