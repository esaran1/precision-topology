"""Blocks 4 and 5: fixed-scale replays from checkpoints with full optimiser state.

Training (shared): a = 1.30, fresh seeds (SEED_OFFSET + s), fold1d.make_data(200, seed), float64,
Adam lr 1e-2, initialisation torch.manual_seed(seed); U(-1,1)^4 (the phase 2b protocol).  Full state
(θ = (w1, b1, w2, b2), Adam exp_avg, exp_avg_sq, step) saved at log-spaced steps and at the first
placement (oriented dense-grid gap > 0, the phase 2b crossing definition, checked every step).

Replay: from a checkpoint, jointly rescale (w2, b2) by k > 0 — every decision sign(N(x)) is unchanged at
the moment of intervention — so that R/R_glob lands on a target level (R = |w2| Ĝ/2, certified Ĝ and
R_glob); then optimise (w1, b1, b2) with |w2| held fixed (w2 excluded from the optimiser) for a
registered horizon.  Optimiser state variants:
  "preserved": the checkpoint's Adam moments and step count for w1, b1, b2 (the b2 moments are kept as
               saved, not rescaled; w2's moments are dropped since w2 is frozen);
  "reset":     fresh Adam (zero moments, step 0).
Recorded: ΔG of the first update, G(t) every 25 steps, first step with G > 0 (Block 4) or with G <= 0
(Block 5), gradient norm of (w1, b1, b2) at the start and end, fraction of training points with
saturated logits (|σ(z) - y| < 1e-3) at the start, and the distance in (w1, b1) to the certified
conditional branch argmin at the held scale (Block 1c).

    python -m src.fixed_scale train N            # N fresh runs with full-state checkpoints
    python -m src.fixed_scale replay4            # Block 4 (before-placement checkpoints)
    python -m src.fixed_scale replay5            # Block 5 (just-after-placement checkpoints)
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
STATE_DIR = RESULTS / "fixed_scale_states"
A = 1.30
LR = 1e-2
SEED_OFFSET = 200_000
BUDGET = 12_000
LEVELS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
WORKERS = int(os.environ.get("FS_WORKERS", "4"))


def _ghat_and_glob():
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    G = float(g[g.a.round(2) == A].Ghat_certified.iloc[0])
    t = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    r = t[(t.a.round(2) == A) & (t.kind == "glob")].iloc[0]
    w2_glob = 0.5 * (r.w2_lo + r.w2_hi)
    return G, w2_glob, w2_glob * G / 2


def _gap(f, w1, b1, w2):
    from .blockB_landscape import gap_of
    return gap_of(f, w1, b1, w2)


def train_one(seed):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", A)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    marks = set(np.unique(np.logspace(0, np.log10(BUDGET), 60).astype(int)))
    states, placed_at = [], None
    for step in range(1, BUDGET + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        w1, b1, w2, b2 = (float(v) for v in th.detach())
        G = _gap(f, w1, b1, w2)
        first = placed_at is None and G > 0
        if first:
            placed_at = step
        if step in marks or first:
            st = opt.state[th]
            states.append({"seed": seed, "step": step, "theta": th.detach().clone().numpy(),
                           "exp_avg": st["exp_avg"].clone().numpy(),
                           "exp_avg_sq": st["exp_avg_sq"].clone().numpy(),
                           "adam_step": int(st["step"]), "G": G, "first_placement": first,
                           "placed_before": placed_at is not None and not first})
        if placed_at is not None and step >= placed_at and step not in marks and not first:
            pass
    return states


def train(n):
    STATE_DIR.mkdir(exist_ok=True)
    seeds = [SEED_OFFSET + s for s in range(int(n))]
    with Pool(WORKERS) as p:
        out = p.map(train_one, seeds)
    allst = [s for ss in out for s in ss]
    with open(STATE_DIR / "checkpoints.pkl", "wb") as fh:
        pickle.dump(allst, fh)
    meta = pd.DataFrame([{k: v for k, v in s.items() if k not in ("theta", "exp_avg", "exp_avg_sq")}
                         | {"w1": s["theta"][0], "b1": s["theta"][1], "w2": s["theta"][2], "b2": s["theta"][3]}
                         for s in allst])
    meta.to_csv(RESULTS / "fixed_scale_checkpoints.csv", index=False)
    print(meta.groupby("seed").first_placement.any().value_counts().to_string())


def _branch_distance(w1, b1, w2, branch):
    """Distance in (w1, b1) to the certified branch argmin (|w1*|, b1* in [0, 2π)), modulo the problem's
    symmetries: orientation (w1, b1, w2) -> (-w1, -b1, -w2) maps w2 < 0 to w2 > 0; the population is
    x-symmetric, so (w1, b1) -> (-w1, b1); and b1 is 2π-periodic."""
    if w2 < 0:
        w1, b1 = -w1, -b1
    db = (b1 - branch[1]) % (2 * math.pi)
    db = min(db, 2 * math.pi - db)
    return math.hypot(abs(w1) - branch[0], db)


def replay(ck, level, variant, horizon, stop_on, G_hat, w2_glob, branch=None, x=None, y=None):
    """Rescale to R/R_glob = level, hold |w2|, train (w1, b1, b2) for `horizon` steps."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    f = activation("sin_family", A)
    if x is None:
        x, y = make_data(200, ck["seed"])
        x, y = x.double(), y.double()
    w1, b1, w2, b2 = (float(v) for v in ck["theta"])
    k = level * w2_glob / abs(w2)
    w2n, b2n = w2 * k, b2 * k
    p = torch.tensor([w1, b1, b2n], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=LR)
    if variant == "preserved":
        opt.state[p] = {"step": torch.tensor(float(ck["adam_step"])),
                        "exp_avg": torch.tensor(ck["exp_avg"][[0, 1, 3]], dtype=torch.float64),
                        "exp_avg_sq": torch.tensor(ck["exp_avg_sq"][[0, 1, 3]], dtype=torch.float64)}
    w2t = torch.tensor(w2n, dtype=torch.float64)

    def lossf():
        z = w2t * f(p[0] * x + p[1]) + p[2]
        return F.binary_cross_entropy_with_logits(z, y), z
    with torch.no_grad():
        _, z0 = lossf()
        sat0 = float(((torch.sigmoid(z0) - y).abs() < 1e-3).double().mean())
    G0 = _gap(f, w1, b1, w2n)
    traj, first_hit, g_start, dG1 = [], None, None, None
    for t in range(1, horizon + 1):
        opt.zero_grad(set_to_none=True)
        L, _ = lossf()
        L.backward()
        if t == 1:
            g_start = float(p.grad.norm())
        opt.step()
        if t == 1 or t % 25 == 0 or t == horizon:
            G = _gap(f, float(p[0]), float(p[1]), w2n)
            if t == 1:
                dG1 = G - G0
            traj.append((t, G))
            hit = (G > 0) if stop_on == "place" else (G <= 0)
            if first_hit is None and hit:
                first_hit = t
    opt.zero_grad(set_to_none=True)
    L, _ = lossf()
    L.backward()
    Gend = traj[-1][1]
    ew1, eb1 = float(p[0]), float(p[1])
    dist = _branch_distance(ew1, eb1, w2n, branch) if branch is not None else np.nan
    return {"seed": ck["seed"], "ck_step": ck["step"], "level": level, "variant": variant,
            "k": k, "G_start": G0, "dG_first": dG1, "G_end": Gend, "first_hit": first_hit,
            "placed_end": Gend > 0, "grad_start": g_start, "grad_end": float(p.grad.norm()),
            "sat_start": sat0, "dist_branch_end": dist, "end_w1": ew1, "end_b1": eb1, "w2_sign": float(np.sign(w2n)),
            "traj_G": ";".join(f"{t}:{g:.6g}" for t, g in traj)}


if __name__ == "__main__" and sys.argv[1] == "train":
    train(sys.argv[2])


# ------------------------------------------------------------------ checkpoint selection and replays
STRATA = ((0.2, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.3))
H4, H5 = 4_000, 12_000


def _load():
    with open(STATE_DIR / "checkpoints.pkl", "rb") as fh:
        return pickle.load(fh)


def _branch_argmins(G_hat, w2_glob):
    """Certified global conditional minimiser (w1, b1) at each held scale s = level * w2_glob."""
    from . import blockB_landscape as bb
    from .profiled_bnb import certify
    x, y = bb.population_data()
    x, y = x.numpy(), y.numpy()
    out = {}
    for lv in LEVELS:
        s = lv * w2_glob
        rm = certify(s, A, x, y, "-")
        rp = certify(s, A, x, y, "+")
        g = rp if rp["upper"] < rm["upper"] else rm
        out[lv] = (abs(g["arg_w1"]), g["arg_b1"])            # w1 folded: the data are x-symmetric
    return out


def select4(states, G_hat, w2_glob):
    """Block 4: before placement (G < 0), latest checkpoint per run per R/R_glob stratum."""
    R_glob = w2_glob * G_hat / 2
    by_seed = {}
    for s in states:
        by_seed.setdefault(s["seed"], []).append(s)
    chosen = []
    for seed, ss in by_seed.items():
        placed = [s["step"] for s in ss if s["first_placement"]]
        t_place = placed[0] if placed else math.inf
        pre = [s for s in ss if s["step"] < t_place and s["G"] < 0]
        for lo, hi in STRATA:
            cand = [s for s in pre if lo <= abs(s["theta"][2]) * G_hat / 2 / R_glob < hi]
            if cand:
                c = max(cand, key=lambda s: s["step"])
                chosen.append({**c, "stratum": f"[{lo},{hi})"})
    return chosen


def select5(states):
    return [s for s in states if s["first_placement"]]


def _decisions_preserved(ck, k):
    import torch
    from .fold1d import activation, make_data
    f = activation("sin_family", A)
    x, _ = make_data(200, ck["seed"])
    x = x.double()
    w1, b1, w2, b2 = (float(v) for v in ck["theta"])
    z = w2 * f(w1 * x + b1) + b2
    return bool(torch.equal(torch.sign(z), torch.sign(k * z)))


def reference_freeze(ck, variant, steps, x=None, y=None):
    """Independent reference for the replay at k = 1 (amended check, 2026-09-23).

    Restores the full 4-parameter Adam optimiser from the checkpoint ('preserved') or starts it fresh
    ('reset'), zeroes w2's gradient AND resets w2 to its saved value after every step, so w2 is truly
    frozen.  (Zeroing the gradient alone does not freeze w2 under Adam: the stored first moment keeps
    moving it — the reason the original check fired.)  Returns (w1, b1, b2) and the w2 trajectory.
    """
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    f = activation("sin_family", A)
    if x is None:
        x, y = make_data(200, ck["seed"])
        x, y = x.double(), y.double()
    th = torch.tensor(ck["theta"], dtype=torch.float64, requires_grad=True)
    w2_0 = th.detach()[2].clone()
    opt = torch.optim.Adam([th], lr=LR)
    if variant == "preserved":
        opt.state[th] = {"step": torch.tensor(float(ck["adam_step"])),
                         "exp_avg": torch.tensor(ck["exp_avg"], dtype=torch.float64),
                         "exp_avg_sq": torch.tensor(ck["exp_avg_sq"], dtype=torch.float64)}
    w2_traj = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        th.grad[2] = 0.0
        opt.step()
        with torch.no_grad():
            th[2] = w2_0
        w2_traj.append(float(th.detach()[2]))
    return th.detach().numpy()[[0, 1, 3]], w2_traj


def replay_params(ck, variant, steps, k=1.0, x=None, y=None):
    """The replay's own implementation (w2 excluded from the optimiser), returning (w1, b1, b2)."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    f = activation("sin_family", A)
    if x is None:
        x, y = make_data(200, ck["seed"])
        x, y = x.double(), y.double()
    w1, b1, w2, b2 = (float(v) for v in ck["theta"])
    p = torch.tensor([w1, b1, b2 * k], dtype=torch.float64, requires_grad=True)
    o = torch.optim.Adam([p], lr=LR)
    if variant == "preserved":
        o.state[p] = {"step": torch.tensor(float(ck["adam_step"])),
                      "exp_avg": torch.tensor(ck["exp_avg"][[0, 1, 3]], dtype=torch.float64),
                      "exp_avg_sq": torch.tensor(ck["exp_avg_sq"][[0, 1, 3]], dtype=torch.float64)}
    w2t = torch.tensor(w2 * k, dtype=torch.float64)
    for _ in range(steps):
        o.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(w2t * f(p[0] * x + p[1]) + p[2], y).backward()
        o.step()
    return p.detach().numpy()


def k1_check(ck, variant="preserved", steps=25):
    """Amended registered check: max |replay - reference_freeze| over the first `steps` steps."""
    ref, _ = reference_freeze(ck, variant, steps)
    return float(np.abs(replay_params(ck, variant, steps) - ref).max())


K1_TOL = 1e-10


def stop_conditions(ck, k, variant="preserved"):
    """Both registered stop conditions for one replay: returns (fires, reasons)."""
    reasons = []
    if not _decisions_preserved(ck, k):
        reasons.append("rescaling changed a decision")
    d = k1_check(ck, variant)
    if not d <= K1_TOL:
        reasons.append(f"k=1 check failed ({d:.3e})")
    return bool(reasons), reasons


def _replay_job(args):
    ck, level, variant, horizon, stop_on, G_hat, w2_glob, branch = args
    import torch
    torch.set_num_threads(1)
    r = replay(ck, level, variant, horizon, stop_on, G_hat, w2_glob, branch=branch)
    r["decisions_preserved"] = _decisions_preserved(ck, r["k"])
    r["stratum"] = ck.get("stratum", "")
    r["ck_G"] = ck["G"]
    r["ck_R_over_glob"] = abs(ck["theta"][2]) * G_hat / 2 / (w2_glob * G_hat / 2)
    return r


def run_replays(block):
    G_hat, w2_glob, R_glob = _ghat_and_glob()
    states = _load()
    cks = select4(states, G_hat, w2_glob) if block == 4 else select5(states)
    horizon, stop_on = (H4, "place") if block == 4 else (H5, "lose")
    br = _branch_argmins(G_hat, w2_glob)
    jobs = [(ck, lv, var, horizon, stop_on, G_hat, w2_glob, br[lv])
            for ck in cks for lv in LEVELS for var in ("preserved", "reset")]
    k1 = [k1_check(ck) for ck in cks[:20]]
    if max(k1) > K1_TOL:
        raise SystemExit(f"STOP: amended k=1 check failed, max diff {max(k1):.3e}")
    print(f"block {block}: {len(cks)} checkpoints, {len(jobs)} replays; k=1 check max diff {max(k1):.2e}",
          flush=True)
    with Pool(WORKERS) as p:
        out = p.map(_replay_job, jobs, chunksize=4)
    d = pd.DataFrame(out)
    d["k1_check_max_diff"] = max(k1)
    d.to_csv(RESULTS / f"fixed_scale_block{block}.csv", index=False)
    print(f"decisions preserved in {int(d.decisions_preserved.sum())} of {len(d)}", flush=True)


if __name__ == "__main__" and sys.argv[1] in ("replay4", "replay5"):
    run_replays(4 if sys.argv[1] == "replay4" else 5)


# ------------------------------------------------------------------ scoring (as registered)
def _mcnemar_one_sided(b, c):
    """Exact one-sided McNemar: P(X >= b) with X ~ Bin(b + c, 1/2); b = discordant pairs in the tested direction."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def _clopper(k, n, alpha=0.05):
    def tail_ge(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if tail_ge(m) < alpha / 2 else (a, m)
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if tail_le(m) > alpha / 2 else (a, m)
        hi = a
    return lo, hi


def _crossing(levels, frac, target=0.5):
    for i in range(1, len(levels)):
        if (frac[i - 1] - target) * (frac[i] - target) <= 0 and frac[i] != frac[i - 1]:
            return levels[i - 1] + (target - frac[i - 1]) * (levels[i] - levels[i - 1]) / (frac[i] - frac[i - 1])
    return float("nan")


def score(block):
    """Writes fixed_scale_block4_curve.csv / fixed_scale_block4_tests.csv (block 4) or
    fixed_scale_block5_curve.csv / fixed_scale_block5_tests.csv (block 5)."""
    d = pd.read_csv(RESULTS / f"fixed_scale_block{block}.csv")
    if block == 4:
        d["success"] = d.placed_end.astype(bool)
    else:
        d["success"] = d.first_hit.isna()                       # retained: never lost
    rows, tests = [], []
    for var, g in d.groupby("variant"):
        piv = g.pivot_table(index=["seed", "ck_step"], columns="level", values="success", aggfunc="first")
        lv = sorted(piv.columns)
        frac = [float(piv[l].mean()) for l in lv]
        for l, f_ in zip(lv, frac):
            k, n = int(piv[l].sum()), int(piv[l].notna().sum())
            lo, hi = _clopper(k, n)
            rows.append({"variant": var, "level": l, "k": k, "n": n, "frac": f_, "ci95_lo": lo, "ci95_hi": hi})
        # monotonicity: any higher level significantly BELOW a lower level (paired, one-sided)
        viol = []
        for i, li in enumerate(lv):
            for lj in lv[i + 1:]:
                b = int(((piv[li] == True) & (piv[lj] == False)).sum())   # lower level success, higher fail
                c = int(((piv[li] == False) & (piv[lj] == True)).sum())
                p = _mcnemar_one_sided(b, c)
                if p < 0.05:
                    viol.append((li, lj, b, c, p))
        x50 = _crossing(lv, frac)
        x10, x90 = _crossing(lv, frac, 0.1), _crossing(lv, frac, 0.9)
        rec = {"variant": var, "x50": x50, "x10": x10, "x90": x90, "width_10_90": x90 - x10,
               "monotonic_violations": len(viol), "violations": str(viol)}
        if block == 4:
            imax = int(np.argmax(frac))
            b = int(((piv[lv[imax]] == True) & (piv[2.0] == False)).sum())
            c = int(((piv[lv[imax]] == False) & (piv[2.0] == True)).sum())
            rec.update({"D1_x50_in_band": 0.9 <= x50 <= 1.25, "D1_pass": (0.9 <= x50 <= 1.25) and not viol
                        and frac[-1] >= frac[0],
                        "D2_level_of_max": lv[imax], "D2_p": _mcnemar_one_sided(b, c),
                        "D2_pass": _mcnemar_one_sided(b, c) < 0.05 and lv[imax] != 2.0})
        else:
            rec.update({"location_in_band": 0.9 <= x50 <= 1.1, "monotone_pass": not viol})
        tests.append(rec)
    pd.DataFrame(rows).to_csv(RESULTS / f"fixed_scale_block{block}_curve.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / f"fixed_scale_block{block}_tests.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(tests).T.to_string())


if __name__ == "__main__" and sys.argv[1] in ("score4", "score5"):
    score(4 if sys.argv[1] == "score4" else 5)


def score5_splits():
    """Block 5, reported without prediction: 10-90% width by starting state (median splits of the
    first-placement step and of G at placement), per variant."""
    d = pd.read_csv(RESULTS / "fixed_scale_block5.csv")
    d["success"] = d.first_hit.isna()
    rows = []
    for split, col in (("first-placement step", "ck_step"), ("G at placement", "ck_G")):
        med = d.drop_duplicates("seed")[col].median()
        for half, g in (("below median", d[d[col] <= med]), ("above median", d[d[col] > med])):
            for var, gv in g.groupby("variant"):
                lv = sorted(gv.level.unique())
                frac = [float(gv[gv.level == l].success.mean()) for l in lv]
                x10, x50, x90 = (_crossing(lv, frac, t) for t in (0.1, 0.5, 0.9))
                rows.append({"split": split, "median": med, "half": half, "variant": var,
                             "n_checkpoints": gv.seed.nunique(), "x10": x10, "x50": x50, "x90": x90,
                             "width_10_90": x90 - x10})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "fixed_scale_block5_splits.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "score5_splits":
    score5_splits()


# ------------------------------------------------------------------ horizon extension (registered cdfbf9d)
HORIZONS = (4_000, 16_000, 64_000)
H_LEVELS = (0.9, 0.95, 1.0, 1.05, 1.1)


def replay_multi(ck, level, variant, w2_glob, branch):
    """Same dynamics as `replay` (identical update sequence), recording the outcome at each horizon."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", A)
    x, y = make_data(200, ck["seed"])
    x, y = x.double(), y.double()
    w1, b1, w2, b2 = (float(v) for v in ck["theta"])
    k = level * w2_glob / abs(w2)
    w2n, b2n = w2 * k, b2 * k
    p = torch.tensor([w1, b1, b2n], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=LR)
    if variant == "preserved":
        opt.state[p] = {"step": torch.tensor(float(ck["adam_step"])),
                        "exp_avg": torch.tensor(ck["exp_avg"][[0, 1, 3]], dtype=torch.float64),
                        "exp_avg_sq": torch.tensor(ck["exp_avg_sq"][[0, 1, 3]], dtype=torch.float64)}
    w2t = torch.tensor(w2n, dtype=torch.float64)
    out = {"seed": ck["seed"], "ck_step": ck["step"], "level": level, "variant": variant, "k": k}
    first_hit, last_nonpos = None, 0
    for t in range(1, HORIZONS[-1] + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(w2t * f(p[0] * x + p[1]) + p[2], y).backward()
        opt.step()
        # exploratory settling time: every step up to 1,000, then every 25 (checks do not alter dynamics)
        if t <= 1000 or t % 25 == 0 or t in HORIZONS:
            G = _gap(f, float(p[0]), float(p[1]), w2n)
            if first_hit is None and G > 0:
                first_hit = t
            if G <= 0:
                last_nonpos = t
            if t in HORIZONS:
                ew1, eb1 = float(p[0]), float(p[1])
                out[f"G_{t}"] = G
                out[f"placed_{t}"] = G > 0
                out[f"dist_{t}"] = _branch_distance(ew1, eb1, w2n, branch)
                out[f"w1_{t}"], out[f"b1_{t}"] = ew1, eb1
    out["first_hit"] = first_hit
    out["settle_step"] = (last_nonpos + 1) if out[f"placed_{HORIZONS[-1]}"] else None   # exploratory
    out["w2_sign"] = float(np.sign(w2n))
    return out


def _multi_job(args):
    return replay_multi(*args)


def run_horizons():
    G_hat, w2_glob, R_glob = _ghat_and_glob()
    cks = select4(_load(), G_hat, w2_glob)
    br = _branch_argmins_levels(H_LEVELS, w2_glob)
    jobs = [(ck, lv, var, w2_glob, br[lv]) for ck in cks for lv in H_LEVELS for var in ("preserved", "reset")]
    print(f"{len(jobs)} long replays", flush=True)
    with Pool(WORKERS) as p:
        out = p.map(_multi_job, jobs, chunksize=2)
    d = pd.DataFrame(out)
    d.to_csv(RESULTS / "fixed_scale_horizons.csv", index=False)
    check_horizons()


def check_horizons(results=None):
    """Registered validity check, as amended (fixed_scale_horizon_prediction.md, amendment 1): the 4,000-step
    outcome must reproduce Block 4 exactly.  Both stored files are read with exact round-trip float parsing;
    tolerance zero; all overlapping (seed, ck_step, level, variant) rows."""
    R = results or RESULTS
    d = pd.read_csv(R / "fixed_scale_horizons.csv", float_precision="round_trip")
    b4 = pd.read_csv(R / "fixed_scale_block4.csv", float_precision="round_trip")
    m = d.merge(b4[["seed", "ck_step", "level", "variant", "placed_end", "G_end"]],
                on=["seed", "ck_step", "level", "variant"], how="inner")
    ok = bool((m.placed_4000 == m.placed_end).all() and np.array_equal(m.G_4000.values, m.G_end.values))
    out = pd.DataFrame([{"rows_compared": len(m), "placement_mismatches": int((m.placed_4000 != m.placed_end).sum()),
                         "G_not_bit_identical": int((m.G_4000.values != m.G_end.values).sum()), "reproduced": ok}])
    out.to_csv(R / "fixed_scale_horizons_check.csv", index=False)
    print("Block 4 reproduced at 4,000 steps (exact parsing):", ok, "on", len(m), "rows", flush=True)
    return ok


def _branch_argmins_levels(levels, w2_glob):
    from . import blockB_landscape as bb
    from .profiled_bnb import certify
    x, y = bb.population_data()
    x, y = x.numpy(), y.numpy()
    out = {}
    for lv in levels:
        s = lv * w2_glob
        rm = certify(s, A, x, y, "-"); rp = certify(s, A, x, y, "+")
        g = rp if rp["upper"] < rm["upper"] else rm
        out[lv] = (abs(g["arg_w1"]), g["arg_b1"] % (2 * math.pi))
    return out


def score_horizons():
    d = pd.read_csv(RESULTS / "fixed_scale_horizons.csv")
    rows = []
    for var, g in d.groupby("variant"):
        for H in HORIZONS:
            fr = g.groupby("level")[f"placed_{H}"].mean()
            lv = list(fr.index)
            rows.append({"variant": var, "horizon": H, **{f"frac_{l}": fr[l] for l in lv},
                         "x50": _crossing(lv, list(fr.values))})
    t = pd.DataFrame(rows)
    res = []
    for var, g in t.groupby("variant"):
        x = dict(zip(g.horizon, g.x50))
        f09 = dict(zip(g.horizon, g["frac_0.9"])); f095 = dict(zip(g.horizon, g["frac_0.95"]))
        q1 = (x[4000] >= x[16000] >= x[64000]) and abs(x[64000] - 1) < abs(x[4000] - 1) and 0.98 <= x[64000] <= 1.02
        q2 = (f09[4000] >= f09[16000] >= f09[64000]) and (f095[4000] >= f095[16000] >= f095[64000]) \
            and f09[64000] == 0 and f095[64000] <= 0.01
        comp = 1.03 <= x[64000] <= 1.07
        res.append({"variant": var, "x50_4k": x[4000], "x50_16k": x[16000], "x50_64k": x[64000],
                    "Q1_pass": q1, "Q2_pass": q2, "competing_outcome": comp,
                    "frac09_64k": f09[64000], "frac095_64k": f095[64000]})
    t.to_csv(RESULTS / "fixed_scale_horizons_curve.csv", index=False)
    pd.DataFrame(res).to_csv(RESULTS / "fixed_scale_horizons_tests.csv", index=False)
    print(t.to_string(index=False)); print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__" and sys.argv[1] in ("horizons", "score_horizons", "check_horizons"):
    {"horizons": run_horizons, "score_horizons": score_horizons, "check_horizons": check_horizons}[sys.argv[1]]()
