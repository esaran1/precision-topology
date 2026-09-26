"""Track 1B: the ramp experiment (registration: results/ramp_registration.md), and R4 (free Adam training at three
learning rates).

Output scale forced to grow: s(t) = s₀·exp(γ(t − W)) after a warm-up of W steps at s₀; the hidden parameters and output
bias (w₁, b₁, b₂) train normally (Adam lr 0.01 or SGD lr 0.3, full batch) with w₂ = +s(t) set every step (the held-scale
update of fixed_scale.replay, with the per-step reset replaced by the ramp).  All 40 seeds of a cell are trained in one
vectorised batch: Adam and SGD act coordinate-wise and the per-seed losses are independent, so this equals per-seed
training (validated against single-seed runs).  Crossing: the first step with G > 0 in w₂'s orientation on the dense
windows of phase2b_ordering (4,001 inner points, 2 × 2,000 outer points).

    python -m src.ramp own        # own thresholds (frozen before any run)
    python -m src.ramp design     # γ grid from κ and the landscape (a priori); written before any run
    python -m src.ramp run | free | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ramp"
A_VALUES = (1.30, 1.50)
SEEDS = tuple(range(860_000, 860_040))
FREE_SEEDS = tuple(range(860_100, 860_140))
LR = {"adam": 1e-2, "sgd": 0.3}
WARMUP = 4000
S0_FRAC = 0.5
END_FRAC = 3.0
N_GAMMA = 6
LAG_RANGE = (0.001, 0.10)
ADAM_LAG_RANGE = (0.0005, 0.006)      # Adam: every pilot run crosses only up to predicted lags ≈ 0.007 (speed limit ~η/step)
FREE_ETAS = (0.01, 0.005, 0.0025)


# ------------------------------------------------------------------------------------------ own thresholds
def _own_job(args):
    os.nice(15)
    from .own_threshold import _pop, own_threshold
    a, seed = args
    return own_threshold(a, seed, _pop(a)[1])


def own(workers=1, seeds=SEEDS, name="own_frozen.csv"):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / ("own_parts.csv" if name == "own_frozen.csv" else "own_parts_free.csv")
    done = set() if not f.exists() else {(round(r.a, 2), r.seed) for r in pd.read_csv(f).itertuples()}
    jobs = [(a, s) for a in A_VALUES for s in seeds if (round(a, 2), s) not in done]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_own_job, jobs):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
    d = pd.read_csv(f).sort_values(["a", "seed"])
    d["w2_own"] = 0.5 * (d.w2_lo + d.w2_hi)
    d.to_csv(OUT / name, index=False)
    h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    (OUT / (name + ".sha256")).write_text(h + "\n")
    print(len(d), int(d.w2_own.isna().sum()), h)


# ------------------------------------------------------------------------------------------ batched training
def _windows():
    import torch
    inner = torch.linspace(-0.8, 0.8, 4001, dtype=torch.float64)
    pos = torch.linspace(1.2, 2.0, 2000, dtype=torch.float64)
    return inner, torch.cat([pos, -pos])


def _data(seeds):
    import torch
    from .fold1d import make_data
    xs, ys = zip(*(make_data(200, s) for s in seeds))
    return torch.stack(xs).double(), torch.stack(ys).double()


def _init(seeds):
    import torch
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(torch.empty(4).uniform_(-1.0, 1.0).double())
    return torch.stack(out)


def branch_init(a, k=0):
    """The population branch point θ*(s₀), s₀ = S0_FRAC·s*, by Newton continuation from θ*(s*) in 25 steps of s
    (lag_law.branch_point; the gradient norm is checked at every step), on winding copy k:
    (w₁, b₁ + 2πk, b₂ − 2πk·s₀) has identical logits, loss and Hessian."""
    from .lag_law import _pop, branch_point
    kt = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    r = kt[kt.a.round(2) == round(a, 2)].iloc[0]
    x, y = _pop()
    z = np.array(json.loads(r.z_star))
    for f in np.linspace(1.0, S0_FRAC, 26):
        z, gr = branch_point(z, f * r.s_star, a, x, y)
        assert gr < 1e-10, (a, f, gr)
    return z + np.array([0.0, 2 * math.pi * k, -2 * math.pi * k * S0_FRAC * r.s_star])


def batch_ramp(a, seeds, opt_name, gamma, s_star_pop, warmup=WARMUP, max_steps=None, record_every=None, init="branch",
               winding=0):
    """Vectorised ramp over seeds.  Returns per-seed dict: crossed, step (after warm-up start), s_cross, placed_in_warmup,
    and the crossing state (w₁, b₁, b₂) plus Adam's v̂ at the crossing (for χ).  init="branch": every seed's hidden
    coordinates start at the population branch point θ*(s₀) (the warm-up relaxes them to the seed's own-sample branch);
    init="random": the standard U(−1, 1) initialisation."""
    import torch
    torch.set_num_threads(1)
    X, Y = _data(seeds)
    if init == "branch":
        p = torch.tensor(np.tile(branch_init(a, winding), (len(seeds), 1)), dtype=torch.float64).requires_grad_(True)
    else:
        p = _init(seeds)[:, [0, 1, 3]].clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=LR["adam"]) if opt_name == "adam" else torch.optim.SGD([p], lr=LR["sgd"])
    inner, outer = _windows()
    n = len(seeds)
    s0 = S0_FRAC * s_star_pop
    s_end = END_FRAC * s_star_pop
    ramp_steps = math.ceil(math.log(s_end / s0) / gamma) if gamma > 0 else 0
    total = warmup + ramp_steps if max_steps is None else max_steps
    done = torch.zeros(n, dtype=torch.bool)
    res = [dict(crossed=False, step=np.nan, s_cross=np.nan, placed_in_warmup=False) for _ in range(n)]
    hist = []
    for t in range(1, total + 1):
        s = s0 if t <= warmup else s0 * math.exp(gamma * (t - warmup))
        opt.zero_grad(set_to_none=True)
        tt = p[:, 0:1] * X + p[:, 1:2]
        z = s * (tt + a * torch.sin(tt)) + p[:, 2:3]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none").mean(dim=1).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w1, b1 = p[:, 0:1], p[:, 1:2]
            ti, to = w1 * inner + b1, w1 * outer + b1
            G = (to + a * torch.sin(to)).min(dim=1).values - (ti + a * torch.sin(ti)).max(dim=1).values
            hit = (G > 0) & ~done
            if hit.any():
                for i in torch.nonzero(hit).flatten().tolist():
                    r = res[i]
                    if t <= warmup:
                        r["placed_in_warmup"] = True
                    else:
                        r.update(crossed=True, step=t - warmup, s_cross=s, w1=float(p[i, 0]), b1=float(p[i, 1]),
                                 b2=float(p[i, 2]))
                        if opt_name == "adam":
                            st = opt.state[p]
                            vh = st["exp_avg_sq"][i] / (1 - 0.999 ** int(st["step"]))
                            r.update(vhat_w1=float(vh[0]), vhat_b1=float(vh[1]), vhat_b2=float(vh[2]))
                done |= hit
            if bool(done.all()):
                break
    for i, s_ in enumerate(seeds):
        res[i].update(seed=s_, w1_end=float(p.detach()[i, 0]), b1_end=float(p.detach()[i, 1]), b2_end=float(p.detach()[i, 2]), steps_run=t)
    return res


def single_ramp_reference(a, seed, opt_name, gamma, s_star_pop, steps, winding=0):
    """The same dynamics for one seed (validation of the batching): parameters after `steps` steps."""
    import torch
    X, Y = _data([seed])
    p = torch.tensor(branch_init(a, winding)[None, :], dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=LR["adam"]) if opt_name == "adam" else torch.optim.SGD([p], lr=LR["sgd"])
    s0 = S0_FRAC * s_star_pop
    for t in range(1, steps + 1):
        s = s0 if t <= WARMUP else s0 * math.exp(gamma * (t - WARMUP))
        opt.zero_grad(set_to_none=True)
        tt = p[:, 0:1] * X + p[:, 1:2]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s * (tt + a * torch.sin(tt)) + p[:, 2:3], Y)
        loss.backward(); opt.step()
    return p.detach().numpy()[0]


# ------------------------------------------------------------------------------------------ design (a priori γ grid)
SETTINGS = ((1.30, -1), (1.50, 0), (1.30, 0))        # (a, winding): the natural windings first, then the sign test
PILOT_SEEDS = tuple(range(869_000, 869_008))
ADAM_SWEEP = tuple(np.geomspace(5e-4, 5e-2, 11))


def _s_star_pop(a):
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    b = br[br.a.round(2) == round(a, 2)].iloc[0]
    return float(0.5 * (b.w2_lo + b.w2_hi))


def design():
    """γ grids, fixed before any registered run, so the predicted |lag| spans LAG_RANGE (6 log-spaced targets).
    SGD: χ = γ/(η λ_min(H)) is exact a priori, so γ = target/|κ_k| · η λ_min(H).
    Adam: v̂ adapts to the ramp, so χ is not known a priori.  γ is calibrated on PILOT_SEEDS (outside every registered
    range): an 11-point γ sweep, the median PREDICTED lag per γ (no observed residual is computed), and log-log
    interpolation to each target in ADAM_LAG_RANGE, using only sweep points where every pilot run crossed.""" 
    kb = pd.read_csv(RESULTS / "lag_law" / "kappa_by_winding.csv")
    L = _landscape()
    rows, sweep = [], []
    for a, kw in SETTINGS:
        sp = _s_star_pop(a)
        lam = float(np.linalg.eigvalsh(L[round(a, 2)]["H"]).min())
        kap = float(kb[(kb.a.round(2) == round(a, 2)) & (kb.k == kw)]["kappa_sgd"].iloc[0])
        for j, lag in enumerate(np.geomspace(*LAG_RANGE, N_GAMMA)):
            g = lag / abs(kap) * LR["sgd"] * lam
            rows.append({"a": a, "winding": kw, "opt": "sgd", "cell": j, "target_lag": lag, "gamma": g, "s_star_pop": sp,
                         "ramp_steps": math.ceil(math.log(END_FRAC / S0_FRAC) / g)})
        cache = OUT / "design_adam_sweep.csv"
        if cache.exists() and len(c := pd.read_csv(cache).query("a.round(2) == @a and winding == @kw")) == len(ADAM_SWEEP):
            sw = c.to_dict("records")
        else:
            sw = []
            for g in ADAM_SWEEP:
                res = batch_ramp(a, PILOT_SEEDS, "adam", g, sp, winding=kw)
                pr = [predict_run(L[round(a, 2)], "adam", g, x)["pred_r"] for x in res if x["crossed"]]
                sw.append({"a": a, "winding": kw, "gamma": g, "n_crossed": len(pr),
                           "median_abs_pred": float(np.median(np.abs(pr))) if pr else np.nan})
                print(sw[-1], flush=True)
        sweep += sw
        use = [r for r in sw if r["n_crossed"] == len(PILOT_SEEDS)]          # all pilot runs cross
        lg = np.log([r["gamma"] for r in use]); lm = np.maximum.accumulate(np.log([r["median_abs_pred"] for r in use]))
        for j, lag in enumerate(np.geomspace(*ADAM_LAG_RANGE, N_GAMMA)):
            assert lm[0] <= math.log(lag) <= lm[-1], (a, kw, lag)
            g = float(np.exp(np.interp(np.log(lag), lm, lg)))
            rows.append({"a": a, "winding": kw, "opt": "adam", "cell": j, "target_lag": lag, "gamma": g, "s_star_pop": sp,
                         "ramp_steps": math.ceil(math.log(END_FRAC / S0_FRAC) / g)})
    d = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(sweep).to_csv(OUT / "design_adam_sweep.csv", index=False)
    d.to_csv(OUT / "design.csv", index=False)
    print(d.to_string(index=False))
    return d


# ------------------------------------------------------------------------------------------ R4: free Adam training at three η
def batch_free(a, seeds, eta, budget):
    """Standard protocol (all four parameters trained, Adam at lr η), vectorised over seeds; every-step crossing."""
    import torch
    torch.set_num_threads(1)
    X, Y = _data(seeds)
    p = _init(seeds).clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=eta)
    inner, outer = _windows()
    n = len(seeds)
    done = torch.zeros(n, dtype=torch.bool)
    res = [dict(seed=s, crossed=False, step=np.nan, w2_cross=np.nan) for s in seeds]
    for t in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        tt = p[:, 0:1] * X + p[:, 1:2]
        z = p[:, 2:3] * (tt + a * torch.sin(tt)) + p[:, 3:4]
        torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none").mean(dim=1).sum().backward()
        opt.step()
        with torch.no_grad():
            w1, b1, w2 = p[:, 0:1], p[:, 1:2], p[:, 2]
            ti, to = w1 * inner + b1, w1 * outer + b1
            fi, fo = ti + a * torch.sin(ti), to + a * torch.sin(to)
            Gp = fo.min(dim=1).values - fi.max(dim=1).values
            Gm = fi.min(dim=1).values - fo.max(dim=1).values
            G = torch.where(w2 > 0, Gp, Gm)
            hit = (G > 0) & ~done
            for i in torch.nonzero(hit).flatten().tolist():
                res[i].update(crossed=True, step=t, w2_cross=abs(float(w2[i])))
            done |= hit
            if bool(done.all()):
                break
    return res


# ------------------------------------------------------------------------------------------ scoring (registered rules)
MIN_CELL = 30
WARMUP_STOP = 0.20


def slope_through_origin(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    return float((pred[ok] * obs[ok]).sum() / (pred[ok] ** 2).sum())


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def score_setting(cells, r3_bound=0.005):
    """cells: list of dicts per γ cell with arrays obs_r, pred_r (per crossing run), gamma, n_runs, n_warmup_placed.
    R1: through-origin slope of obs on pred (all resolved cells' runs) in [0.7, 1.3].
    R2: sign(κ)·Spearman(γ, cell median obs) >= 0.9 (the lag grows with γ in κ's direction).
    R3: |median obs| of the slowest resolved cell < r3_bound.
    A cell is resolved with >= MIN_CELL crossings; >= 4 resolved cells are needed; the setting STOPs if > 20% of runs
    were placed during warm-up."""
    tot = sum(c["n_runs"] for c in cells); wp = sum(c["n_warmup_placed"] for c in cells)
    if tot and wp / tot > WARMUP_STOP:
        return {"R1": "STOP", "R2": "STOP", "R3": "STOP", "warmup_placed_frac": wp / tot}
    res = [c for c in cells if len(c["obs_r"]) >= MIN_CELL]
    out = {"cells_resolved": len(res), "warmup_placed_frac": wp / tot if tot else 0.0}
    if len(res) < 4:
        return {**out, "R1": "UNRESOLVED", "R2": "UNRESOLVED", "R3": "UNRESOLVED"}
    obs = np.concatenate([c["obs_r"] for c in res]); pred = np.concatenate([c["pred_r"] for c in res])
    sl = slope_through_origin(obs, pred)
    sign = float(np.sign(np.median(pred)))
    meds = [float(np.median(c["obs_r"])) for c in res]
    sp = sign * spearman([c["gamma"] for c in res], meds)
    slow = min(res, key=lambda c: c["gamma"])
    m_slow = float(np.median(slow["obs_r"]))
    return {**out, "slope_obs_on_pred": sl, "R1": "PASS" if 0.7 <= sl <= 1.3 else "FAIL",
            "signed_spearman_gamma_median": sp, "R2": "PASS" if sp >= 0.9 else "FAIL",
            "slowest_median_r": m_slow, "R3": "PASS" if abs(m_slow) < r3_bound else "FAIL",
            "cell_medians_obs": meds, "cell_medians_pred": [float(np.median(c["pred_r"])) for c in res]}


def score_r4(res_by_eta, tol_abs=0.01, tol_rel=0.25):
    """R4: at each η, the median residual is within max(0.01, 0.25·|r(0.01)|) of the η = 0.01 median; each η needs
    >= MIN_CELL crossings."""
    base = res_by_eta[0.01]
    if len(base) < MIN_CELL:
        return {"R4": "UNRESOLVED"}
    m0 = float(np.median(base)); tol = max(tol_abs, tol_rel * abs(m0))
    out = {"median_0.01": m0, "tol": tol}
    ok = True
    for eta, v in res_by_eta.items():
        if len(v) < MIN_CELL:
            return {**out, "R4": "UNRESOLVED"}
        out[f"median_{eta}"] = float(np.median(v))
        ok &= abs(float(np.median(v)) - m0) <= tol
    return {**out, "R4": "PASS" if ok else "FAIL"}

# ------------------------------------------------------------------------------------------ per-run prediction (registered)
def _landscape():
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    return {round(r.a, 2): {"H": np.array(json.loads(r.H)), "tan": np.array(json.loads(r.tangent)),
                            "dG": np.array(json.loads(r.gradG)), "z": np.array(json.loads(r.z_star))} for r in k.itertuples()}


def predict_run(L, opt_name, gamma, rec):
    """r_pred = κ_k·χ with χ = γ/(η λ_min(P^{1/2}HP^{1/2})): H, θ*′, ∇G at the population switch (lag_law), the run's
    winding k = round((b₁ − b₁*)/2π) at its crossing, and P = I (SGD) or the run's own 1/(√v̂ + ε) at its crossing (Adam)."""
    from .lag_law import TWO_PI, kappa_k
    H, tan, dG, z = L["H"], L["tan"], L["dG"], L["z"]
    k = int(round((rec["b1"] - z[1]) / TWO_PI))
    p = np.ones(3) if opt_name == "sgd" else 1.0 / (np.sqrt([rec["vhat_w1"], rec["vhat_b1"], rec["vhat_b2"]]) + 1e-8)
    lam = float(np.linalg.eigvalsh((np.sqrt(p)[:, None] * H) * np.sqrt(p)[None, :]).min())
    chi = gamma / (LR[opt_name] * lam)
    kap = kappa_k(H, tan, dG, p, k)
    return {"winding_k": k, "b1_offset": float(rec["b1"] - z[1] - TWO_PI * k), "mirror": bool(np.sign(rec["w1"]) != np.sign(z[0])),
            "chi": chi, "kappa_k": kap, "pred_r": kap * chi}


# ------------------------------------------------------------------------------------------ runs (after the registration commit)
def _append(f, rows):
    pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)


def run():
    """Every design cell with the 40 registered seeds; resumable (cells already in runs.csv are skipped)."""
    os.nice(15)
    d = pd.read_csv(OUT / "design.csv"); f = OUT / "runs.csv"
    done = set() if not f.exists() else {(round(r.a, 2), r.winding, r.opt, r.cell) for r in pd.read_csv(f).itertuples()}
    for r in d.itertuples():
        if (round(r.a, 2), r.winding, r.opt, r.cell) in done:
            continue
        res = batch_ramp(r.a, SEEDS, r.opt, r.gamma, r.s_star_pop, winding=r.winding)
        _append(f, [{"a": r.a, "winding": r.winding, "opt": r.opt, "cell": r.cell, "gamma": r.gamma, **x} for x in res])
        print(r.a, r.winding, r.opt, r.cell, sum(x["crossed"] for x in res), flush=True)


def free():
    """R4: free Adam training (standard protocol) at FREE_ETAS with budget 32,000·(0.01/η); resumable."""
    os.nice(15)
    f = OUT / "free_runs.csv"
    done = set() if not f.exists() else {(round(r.a, 2), r.eta) for r in pd.read_csv(f).itertuples()}
    for a in A_VALUES:
        for eta in FREE_ETAS:
            if (round(a, 2), eta) in done:
                continue
            res = batch_free(a, FREE_SEEDS, eta, int(round(32_000 * 0.01 / eta)))
            _append(f, [{"a": a, "eta": eta, **x} for x in res])
            print(a, eta, sum(x["crossed"] for x in res), flush=True)


def _check_hash(name):
    h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    assert h == (OUT / (name + ".sha256")).read_text().strip(), name
    return h


def read_runs(path=None):
    """runs.csv was appended cell by cell with the first cell's (SGD) header, and Adam rows carry three extra v̂ fields,
    so rows are parsed by their field count (each record's key order is fixed by batch_ramp):
      14 fields: not crossed; 17: crossed, SGD; 20: crossed, Adam.  Every row must match one schema exactly."""
    import csv
    head = ["a", "winding", "opt", "cell", "gamma", "crossed", "step", "s_cross", "placed_in_warmup"]
    tail = ["seed", "w1_end", "b1_end", "b2_end", "steps_run"]
    schema = {14: head + tail, 17: head + ["w1", "b1", "b2"] + tail,
              20: head + ["w1", "b1", "b2", "vhat_w1", "vhat_b1", "vhat_b2"] + tail}
    rows = []
    with open(path or OUT / "runs.csv") as f:
        r = csv.reader(f)
        hdr = next(r)
        assert hdr == schema[17], hdr
        for line in r:
            cols = schema[len(line)]
            rec = dict(zip(cols, line))
            assert (len(line) == 20) == (rec["opt"] == "adam" and rec["crossed"] == "True"), line
            assert (len(line) == 14) == (rec["crossed"] == "False"), line
            rows.append(rec)
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c == "opt":
            continue
        d[c] = d[c].map({"True": True, "False": False}) if c in ("crossed", "placed_in_warmup") else pd.to_numeric(d[c])
    return d


def score():
    _check_hash("own_frozen.csv"); _check_hash("own_free_frozen.csv")
    L = _landscape()
    own = pd.read_csv(OUT / "own_frozen.csv"); own["a"] = own.a.round(2)
    d = read_runs(); d["a"] = d.a.round(2)
    d = d.merge(own[["a", "seed", "w2_own"]], on=["a", "seed"], how="left")
    rows = []
    for r in d.itertuples():
        rec = r._asdict()
        if r.crossed and np.isfinite(r.w2_own):
            pr = predict_run(L[r.a], r.opt, r.gamma, rec)
            rows.append({**{k: rec[k] for k in ("a", "winding", "opt", "cell", "gamma", "seed", "s_cross", "w2_own")},
                         **pr, "init_winding": r.winding, "obs_r": r.s_cross / r.w2_own - 1})
    P = pd.DataFrame(rows); P.to_csv(OUT / "scored_runs.csv", index=False)
    verdicts = []
    for (a, kw, opt), g in d.groupby(["a", "winding", "opt"]):
        cells = []
        for (cell, gamma), gc in g.groupby(["cell", "gamma"]):
            pc = P[(P.a == a) & (P.winding == kw) & (P.opt == opt) & (P.cell == cell)]
            cells.append({"gamma": gamma, "obs_r": pc.obs_r.to_numpy(), "pred_r": pc.pred_r.to_numpy(),
                          "n_runs": len(gc), "n_warmup_placed": int(gc.placed_in_warmup.sum())})
        sc = score_setting(cells)
        verdicts.append({"a": a, "winding": kw, "opt": opt,
                         "family": "R5 (winding sign test)" if (a == 1.3 and kw == 0) else "R1-R3",
                         "n_crossed": int(g.crossed.sum()), "n_runs": len(g),
                         "winding_changed": int((P[(P.a == a) & (P.init_winding == kw) & (P.opt == opt)].winding_k != kw).sum()),
                         **sc})
    V = pd.DataFrame(verdicts)
    ff = OUT / "free_runs.csv"
    if not ff.exists() or len(pd.read_csv(ff)) < len(A_VALUES) * len(FREE_ETAS) * len(FREE_SEEDS):
        out = {"settings": V.to_dict("records"), "R4": "pending (free runs incomplete)"}
        (OUT / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
        pd.set_option("display.width", 250)
        print(V.drop(columns=["cell_medians_obs", "cell_medians_pred"], errors="ignore").to_string(index=False))
        return out
    fo = pd.read_csv(OUT / "own_free_frozen.csv"); fo["a"] = fo.a.round(2)
    fr = pd.read_csv(ff); fr["a"] = fr.a.round(2)
    fr = fr.merge(fo[["a", "seed", "w2_own"]], on=["a", "seed"], how="left")
    fr["obs_r"] = fr.w2_cross / fr.w2_own - 1
    fr.to_csv(OUT / "free_scored.csv", index=False)
    r4 = []
    for a, g in fr.groupby("a"):
        ok = g[g.crossed & np.isfinite(g.obs_r)]
        r4.append({"a": a, **score_r4({e: ok[ok.eta == e].obs_r.to_numpy() for e in FREE_ETAS}),
                   **{f"n_crossed_{e}": int((ok.eta == e).sum()) for e in FREE_ETAS}})
    out = {"settings": V.to_dict("records"), "R4": r4}
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(V.drop(columns=["cell_medians_obs", "cell_medians_pred"], errors="ignore").to_string(index=False))
    print(pd.DataFrame(r4).to_string(index=False))
    return out


# ------------------------------------------------------------------------------------------ POST HOC (after the registered score)
def _branch_switch_job(args):
    """POST HOC: the switch of the branch the ramp actually tracks, on the seed's own sample.  Newton continuation
    (lag_law.branch_point) from the population branch point θ*(s₀) on winding k, relaxed on the own sample at s₀, then
    stepped up in s (1%) until its gap G(θ_b(s)) > 0, then bisected to 1e-6 relative."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    from .lag_law import branch_point, gap
    a, k, seed = args
    X, Y = _data([seed]); x, y = X[0].numpy(), Y[0].numpy()
    sp = _s_star_pop(a); s = S0_FRAC * sp
    z, gr = branch_point(branch_init(a, k), s, a, x, y)
    G = lambda z_: gap(z_[0], z_[1], a)
    out = {"a": a, "winding": k, "seed": seed, "grad_s0": gr}
    if G(z) > 0:
        return {**out, "note": "placed at s0", "s_branch": np.nan}
    lo, zlo = s, z
    while s < END_FRAC * sp:
        s_new = s * 1.01
        z_new, gr = branch_point(zlo, s_new, a, x, y)
        if gr > 1e-8 or np.linalg.norm(z_new - zlo) > 0.5:
            return {**out, "note": f"continuation lost at s={s_new:.4f} (grad {gr:.1e})", "s_branch": np.nan}
        if G(z_new) > 0:
            hi, zhi = s_new, z_new
            while (hi - lo) / lo > 1e-6:
                mid = 0.5 * (lo + hi)
                zm, _ = branch_point(zlo, mid, a, x, y)
                if G(zm) > 0:
                    hi, zhi = mid, zm
                else:
                    lo, zlo = mid, zm
            return {**out, "note": "", "s_branch": 0.5 * (lo + hi), "w1": float(zhi[0]), "b1": float(zhi[1])}
        s, lo, zlo = s_new, s_new, z_new
    return {**out, "note": "not placed by 3 s*", "s_branch": np.nan}


def posthoc_branch(workers=1):
    """POST HOC (labelled; after the registered score): residuals against the tracked branch's own-sample switch."""
    from multiprocessing import get_context
    f = OUT / "posthoc_branch_switch.csv"
    jobs = [(a, k, sd) for a, k in SETTINGS for sd in SEEDS]
    if not f.exists():
        with get_context("spawn").Pool(workers) as pool:
            rows = list(pool.imap(_branch_switch_job, jobs))
        pd.DataFrame(rows).to_csv(f, index=False)
    b = pd.read_csv(f); b["a"] = b.a.round(2)
    own = pd.read_csv(OUT / "own_frozen.csv"); own["a"] = own.a.round(2)
    b = b.merge(own[["a", "seed", "w2_own"]], on=["a", "seed"])
    b["branch_over_own"] = b.s_branch / b.w2_own
    P = pd.read_csv(OUT / "scored_runs.csv"); P["a"] = P.a.round(2)
    P = P.merge(b[["a", "winding", "seed", "s_branch", "branch_over_own"]], left_on=["a", "init_winding", "seed"],
                right_on=["a", "winding", "seed"], suffixes=("", "_b"))
    P["obs_r_branch"] = P.s_cross / P.s_branch - 1
    P.to_csv(OUT / "posthoc_scored_runs.csv", index=False)
    rows = []
    for (a, kw, opt), g in P.groupby(["a", "init_winding", "opt"]):
        ok = g[np.isfinite(g.obs_r_branch)]
        cells = [{"gamma": gm, "obs_r": gc.obs_r_branch.to_numpy(), "pred_r": gc.pred_r.to_numpy(), "n_runs": len(gc),
                  "n_warmup_placed": 0} for gm, gc in ok.groupby("gamma")]
        sc = score_setting(cells)
        rows.append({"a": a, "winding": kw, "opt": opt, "n_with_branch_switch": len(ok) // N_GAMMA,
                     "frac_branch_off_own_1pct": float((np.abs(b[(b.a == a) & (b.winding == kw)].branch_over_own - 1) > 0.01).mean()),
                     **{k_: v for k_, v in sc.items()}})
    d = pd.DataFrame(rows); d.to_json(OUT / "posthoc_branch.json", orient="records", indent=1)
    pd.set_option("display.width", 250)
    print(d.drop(columns=["cell_medians_obs", "cell_medians_pred"]).to_string(index=False))
    for r in rows:
        print(r["a"], r["winding"], r["opt"], np.round(r.get("cell_medians_obs", []), 4), np.round(r.get("cell_medians_pred", []), 4))
    return d


def adam_contrast():
    """POST HOC (author's request): what differs between the Adam ramp and free Adam training (where the lag law held,
    Track 1A)?  Per a: the measured preconditioner at crossing (√v̂ per coordinate), the growth rate d log s/dt, the
    relaxation time 1/(ηλ_min(P^{1/2}HP^{1/2})), χ, the number of steps from the start of growth to the crossing against
    Adam's v̂ memory 1/(1 − β₂) = 1,000 steps, and the branch (winding, mirror).  Free: the deconfounded lag test's
    φ = 1 primary runs (residual_timescale_runs.csv; preconditioner_runs.csv); ramp: all crossing Adam ramp runs on the
    natural winding.  Written to adam_contrast.csv."""
    d = read_runs(); d["a"] = d.a.round(2)
    P = pd.read_csv(OUT / "scored_runs.csv"); P["a"] = P.a.round(2)
    ft = pd.read_csv(RESULTS / "residual_timescale_runs.csv"); ft["a"] = ft.a.round(2)
    fp = pd.read_csv(RESULTS / "lag_law" / "preconditioner_runs.csv"); fp["a"] = fp.a.round(2)
    pr = pd.read_csv(RESULTS / "lag_law" / "predictions.csv"); pr["a"] = pr.a.round(2)
    rows = []
    for a, k in ((1.30, -1), (1.50, 0)):
        g = d[(d.a == a) & (d.opt == "adam") & (d.winding == k) & d.crossed]
        q = P[(P.a == a) & (P.opt == "adam") & (P.winding == k)]
        f = ft[(ft.a == a) & (ft.factor == 1.0)]; fpa = fp[fp.a == a]
        fr = pr[(pr.a == a) & (pr["set"] == "lag2-prim") & (pr.arm.astype(str) == "1.0")]
        rows.append({"a": a, "source": "ramp (Adam)", "n": len(g),
                     "sqrt_v_w1": float(np.sqrt(g.vhat_w1).median()), "sqrt_v_b1": float(np.sqrt(g.vhat_b1).median()),
                     "sqrt_v_b2": float(np.sqrt(g.vhat_b2).median()),
                     "growth_per_step": float(g.gamma.median()), "growth_min": float(g.gamma.min()), "growth_max": float(g.gamma.max()),
                     "relax_steps": float((q.chi / q.gamma).median()), "chi": float(q.chi.median()),
                     "steps_growth_to_cross": float(g.step.median()), "winding": k, "mirror_frac": float(q.mirror.mean()),
                     "pred_r_median": float(q.pred_r.median())})
        rows.append({"a": a, "source": "free Adam (lag test 2, phi = 1)", "n": len(f),
                     "sqrt_v_w1": float(fpa.sqrt_v_w1.median()), "sqrt_v_b1": float(fpa.sqrt_v_b1.median()),
                     "sqrt_v_b2": float(fpa.sqrt_v_b2.median()),
                     "growth_per_step": float(f.growth.median()), "growth_min": float(f.growth.min()), "growth_max": float(f.growth.max()),
                     "relax_steps": float((1 / f.relax).median()), "chi": float(f.ratio.median()),
                     "steps_growth_to_cross": float(f.cross_step.median()), "winding": int(fr.winding_k.median()),
                     "mirror_frac": float(fr.mirror.mean()), "pred_r_median": float(fr.pred_r.median())})
    out = pd.DataFrame(rows); out.to_csv(OUT / "adam_contrast.csv", index=False)
    pd.set_option("display.width", 250); print(out.T.to_string())
    return out



if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"own": lambda: own(w), "own_free": lambda: own(w, FREE_SEEDS, "own_free_frozen.csv"), "design": design,
     "run": lambda: run(), "free": lambda: free(), "score": lambda: score(), "posthoc": lambda: posthoc_branch(w), "adam_contrast": adam_contrast}[sys.argv[1]]()
