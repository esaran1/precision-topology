"""T2-3b (registered AFTER T2-3's failure; results/asym_registration.md, amendment 2): T2-3 with the approved sweep-rate
matching of the width-2 design (Revision 3).  Same geometry (a = 1.30, Δ = 0.4), criteria and matched initialisation;
the output weights' learning-rate factor φ₂ from Revision 3's steps-matching rule; fresh seeds 600,160 onward.

Rule (Revision 3): φ₂ is chosen so that
  the median number of steps for ‖w₂‖₁ to go from its (matched) initial value to s₂,glob (the T2-1 bracket midpoint),
  at width 2 on the Δ = 0.4 training sets, equals, within 2%,
  the median number of steps for |w₂| to go from its initial value to |w₂|_glob(1.30) = 4.95625, at width 1 with the
  width-1 protocol (fold1d.make_data(200, seed), torch.manual_seed(seed), U(−1, 1)⁴ in float32 then double, Adam 0.01).
  Calibration seeds 510,000–510,039; the pilot records ‖w₂‖₁ / |w₂| only (no placement is evaluated).  Bisection in
  log φ₂ on [1e−4, 1].
Implementation: after each Adam step, v ← v_before + φ₂(v_after − v_before) (exactly learning rate φ₂·lr for v).
Validity check (registered): the achieved median timescale ratio at crossing (asym_posthoc2's definition) must lie in
the width-1 Adam range [0.0014, 0.023]; otherwise T2-3b is UNRESOLVED.

    python -m src.asym_t23b pilot | train | score
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

from .asym_register import BUDGET, FROZEN, MIN_CROSS, STEP0_STOP, _act, _setup, score_t2_3, training_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_t23b"
CAL_SEEDS = tuple(range(510_000, 510_040))
SEEDS = tuple(range(600_160, 600_240))
SEEDS_EXT = tuple(range(600_240, 600_320))
W2_GLOB_W1 = 4.95625
PHI_RANGE = (1e-4, 1.0)
RATIO_RANGE = (0.0014, 0.023)
PILOT_CAP = 64_000
FROZEN_B = RESULTS / "asym_t23b_frozen.json"


def _k():
    return json.loads(FROZEN.read_text())["k"]


def _s_glob():
    f = json.loads(FROZEN.read_text())
    return math.sqrt(f["s_lo"] * f["s_hi"])


def steps_w2(seed, phi, target, cap=PILOT_CAP):
    """Width 2, matched initialisation, output lr factor φ₂: steps for ‖v‖₁ to first reach target (cap if never)."""
    import torch
    from .width2_train import LR, init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); k = _k(); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    for step in range(1, cap + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        vb = q.detach()[[2, 5]].clone()
        opt.step()
        with torch.no_grad():
            q[[2, 5]] = vb + phi * (q[[2, 5]] - vb)
        if float(q.detach()[2].abs() + q.detach()[5].abs()) >= target:
            return step
    return cap


def steps_w1(seed, target=W2_GLOB_W1, cap=PILOT_CAP):
    """Width 1, the width-1 protocol at a = 1.30: steps for |w₂| to first reach target."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", 1.30)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    x, y = x.double(), y.double()
    for step in range(1, cap + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        if abs(float(th.detach()[2])) >= target:
            return step
    return cap


def choose_phi(med_w1, med_w2_of, lo=PHI_RANGE[0], hi=PHI_RANGE[1], rel=0.02, max_iter=40):
    """Bisection in log φ₂ (median steps decrease as φ₂ grows).  Returns (φ₂, median) or (None, reason)."""
    m_lo, m_hi = med_w2_of(lo), med_w2_of(hi)
    if not (m_hi <= med_w1 <= m_lo):
        return None, f"no phi in range: median steps {m_hi} at phi = {hi}, {m_lo} at phi = {lo}; width-1 {med_w1}"
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        m = med_w2_of(mid)
        if abs(m / med_w1 - 1) <= rel:
            return mid, m
        lo, hi = (mid, hi) if m > med_w1 else (lo, mid)
    return None, "bisection did not reach 2%"


def _w1_job(seed):
    os.nice(15)
    return steps_w1(seed)


def _w2_job(args):
    os.nice(15)
    return steps_w2(*args)


def pilot(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    target = _s_glob()
    with get_context("spawn").Pool(workers) as pool:
        w1 = pool.map(_w1_job, CAL_SEEDS)
        med_w1 = float(np.median(w1))
        log = []

        def med_w2_of(phi):
            v = pool.map(_w2_job, [(s, phi, target) for s in CAL_SEEDS])
            m = float(np.median(v))
            log.append({"phi": phi, "median_steps": m, "capped": int(sum(t >= PILOT_CAP for t in v))})
            print(json.dumps(log[-1]), flush=True)
            return m
        phi, m = choose_phi(med_w1, med_w2_of)
    res = {"median_steps_w1": med_w1, "w1_capped": int(sum(t >= PILOT_CAP for t in w1)), "phi2": phi, "result": m,
           "bisection": log, "target_s_glob": target}
    (OUT / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("median_steps_w1", "phi2", "result")}, default=float))
    return res


def freeze():
    p = json.loads((OUT / "pilot.json").read_text())
    if p["phi2"] is None:
        raise SystemExit("no phi2: T2-3b is not registered")
    d = {"phi2": p["phi2"], "k": _k(), "seeds": [SEEDS[0], SEEDS[-1]], "seeds_ext": [SEEDS_EXT[0], SEEDS_EXT[-1]],
         "budget": BUDGET, "ratio_range": RATIO_RANGE, "median_steps_w1": p["median_steps_w1"],
         "median_steps_w2_at_phi2": p["result"]}
    txt = json.dumps(d, indent=1, sort_keys=True)
    FROZEN_B.write_text(txt)
    (RESULTS / "asym_t23b_frozen.sha256").write_text(hashlib.sha256(txt.encode()).hexdigest() + "\n")
    print(txt)


def train_one(seed, k, phi, budget=BUDGET):
    """As asym_register.train_one, with the output learning-rate factor φ₂; placement from step 0; also the timescale
    ratio at the crossing (asym_posthoc2's definition, growth under φ₂)."""
    import torch
    from .width2_train import LR, init_params, logits, placed, unpack
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    if placed(q0.numpy(), act)[0]:
        return {"seed": seed, "placed_at_init": True, "crossed": False, "step": 0, "s_cross": np.nan, "ratio": np.nan}
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    hist = {}
    for step in range(1, budget + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        vb = q.detach()[[2, 5]].clone()
        opt.step()
        with torch.no_grad():
            q[[2, 5]] = vb + phi * (q[[2, 5]] - vb)
        qn = q.detach().numpy().copy()
        hist[step] = float(abs(qn[2]) + abs(qn[5]))
        hist.pop(step - 101, None)
        if placed(qn, act)[0]:
            _, v, _ = unpack(qn)
            win = min(100, step - 1)
            growth = math.log(hist[step] / hist[step - win]) / win if win >= 1 else float("nan")
            relax = _relax(q.detach().clone(), opt.state[q], X, Y, act)
            return {"seed": seed, "placed_at_init": False, "crossed": True, "step": step,
                    "s_cross": float(np.abs(v).sum()), "growth": growth, "relax": relax,
                    "ratio": growth / relax if relax > 0 else np.nan}
    return {"seed": seed, "placed_at_init": False, "crossed": False, "step": np.nan, "s_cross": np.nan, "ratio": np.nan}


def _relax(qc, st, X, Y, act, lr=1e-2, eps=1e-8, beta2=0.999):
    import torch
    from .width2_train import logits
    idx = [0, 1, 3, 4, 6]
    vhat = (st["exp_avg_sq"].numpy() / (1 - beta2 ** int(st["step"])))[idx]

    def L(z):
        qq = qc.clone(); qq[idx] = z
        return torch.nn.functional.binary_cross_entropy_with_logits(logits(qq, X, act), Y)
    z = qc[idx].clone(); mu, f0 = 1e-3, float(L(z))
    for _ in range(200):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(L(zz), zz)[0]
        if float(g.abs().max()) < 1e-12:
            break
        H = torch.autograd.functional.hessian(L, z)
        ok = False
        for _ in range(30):
            d = torch.linalg.solve(H + mu * torch.eye(5, dtype=torch.float64) * max(1.0, float(H.diag().abs().max())), -g)
            f1 = float(L(z + d))
            if f1 < f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 3, 1e-12)
                break
            mu *= 10
        if not ok:
            break
    H = torch.autograd.functional.hessian(L, z).numpy()
    dd = 1.0 / np.sqrt(np.sqrt(vhat) + eps)
    M = dd[:, None] * H * dd[None, :]
    return lr * float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


def _train_job(seed):
    os.nice(15)
    fz = json.loads(FROZEN_B.read_text())
    r = train_one(seed, fz["k"], fz["phi2"])
    print(json.dumps({"seed": seed, "done": True}), flush=True)
    return r


def train(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "train.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    todo = [s for s in SEEDS if s not in done]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_train_job, todo):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
    d = pd.read_csv(f)
    if d.placed_at_init.mean() <= STEP0_STOP and d.crossed.sum() < MIN_CROSS:
        todo = [s for s in SEEDS_EXT if s not in set(d.seed)]
        with get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_train_job, todo):
                pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def validity(ratios, lo=RATIO_RANGE[0], hi=RATIO_RANGE[1]):
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r) & (r > 0)]
    if not len(r):
        return False, float("nan")
    m = float(np.median(r))
    return bool(lo <= m <= hi), m


def score():
    fz = json.loads(FROZEN.read_text())
    d = pd.read_csv(OUT / "train.csv")
    out = {"runs": len(d), "placed_at_init": int(d.placed_at_init.sum()), "crossed": int(d.crossed.sum())}
    if d.placed_at_init.mean() > STEP0_STOP:
        out["T2-3b"] = "STOP (more than 20% placed at step 0)"
    else:
        c = d[d.crossed & ~d.placed_at_init]
        ok, m = validity(c.ratio)
        out["median_ratio_at_cross"] = m
        out["validity_ratio_in_width1_range"] = ok
        sc = score_t2_3(c.s_cross.values, fz["s_lo"], fz["s_hi"])
        out["criteria"] = sc
        out["T2-3b"] = sc["verdict"] if ok else f"UNRESOLVED (median ratio {m:.4g} outside {RATIO_RANGE})"
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"pilot": lambda: pilot(w), "freeze": freeze, "train": lambda: train(w), "score": score}[sys.argv[1]]()
