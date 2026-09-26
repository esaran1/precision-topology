"""Track 2A (POST HOC, existing data; no verdict changes): are the no-gating test's stuck states basins?

Endpoints: the Track 7 no-gating replays (src/width2_nogating.py; symmetric windows; f1.30, f1.50 and tanh; seeds
630,000–630,079; ‖v‖₁ held at s = 2R₂/Γ̂₂ by projection; R₂ ∈ {0.003, 0.01, 0.03, 0.1}; variants preserved / reset) at
their scored step H = 16,000 that are UNPLACED (exact G₊ ≤ 0) and STATIONARY by Track 7's criterion (max|∇L|/‖v‖₁ ≤ 1e−6
over the free directions).  Each is re-created by deterministic replay (replay_ng's operations; the recorded unit shares,
α and β must reproduce to 1e−9).

Per endpoint (definitions fixed before any endpoint was evaluated):
  polish     damped Newton on the fixed-scale conditional loss in the joint coordinates (α₁, β₁, α₂, β₂, v₁, v₂, b),
             φ = (v₁u₁ + v₂u₂)/‖v‖₁ at the held s (width2_finish.newton), to max|∇| ≤ 1e−13; recorded: the distance moved.
  reduced Hessian   the training-loss Hessian in (θ, v, b) restricted to the free directions of the held ℓ₁ sphere
             (width2_nogating.free_basis: 6 dimensions; with the signs of v fixed the sphere is a hyperplane, so this is
             the constrained second-order form).  Eigenvalues relative to the largest; |λ| ≤ ZERO_REL·λ_max is a zero mode.
  class      'strict minimum' (all > 0); 'saddle' (some < 0); 'duplicate-unit minimum (Morse–Bott)': the two units are
             copies (αᵢ, βᵢ equal up to the unit orientation (α, β, v) → (−α, −β, −v), to 1e−6), exactly one zero mode, aligned
             (|cos| ≥ 0.999) with the weight-transfer direction (δ|v₁| = −δ|v₂|, which leaves the function unchanged), and all
             other eigenvalues > 0 -- a line of equal-loss critical points with a positive-definite transverse Hessian,
             i.e. a (non-isolated) conditional local minimum; 'degenerate, undetermined' otherwise.
  loss gap   L(endpoint) − L(best placed minimiser found at the same s on the run's own training set): starts = the
             population direct-check minimiser (width2_direct_check.csv, same act, nearest R₂, and its unit-orientation /
             mirror images) and a 100-restart batched BFGS search (width2_conditional.search_batch, cap 1,500) whose 30 lowest
             distinct candidates are Newton-polished (computed once per act, seed and R₂; shared by both variants).  'Placed' by exact extrema.  Not certified: the best placed found.
             Also recorded: the lowest loss of any status found by the search (is the stuck state itself the lowest?).

Added after the first 136 endpoints had been scored (disclosed): 'check' re-runs the deterministic placed search for every
(act, seed, R₂) where a placed minimiser was found, records the gradient at the best placed point and polishes it further
(2,000 Newton iterations), because the polish in placed_best stops at 300 iterations without a convergence flag.

    python -m src.width2_basins run        # one process, nice 15, resumable
    python -m src.width2_basins check      # the convergence check of the best placed minimisers
    python -m src.width2_basins score
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "width2_basins"
NG = RESULTS / "width2_nogating_parts"
STATIONARY_TOL, H_STEP = 1e-6, 16_000
ZERO_REL, DUP_TOL, ALIGN = 1e-9, 1e-6, 0.999
RESTARTS, N_POLISH, MAXIT = 100, 10, 1500


def endpoints():
    d = pd.read_csv(NG / "replays.csv", float_precision="round_trip")
    e = d[(d.step == H_STEP) & (d.placed == False) & (d.grad_rel <= STATIONARY_TOL)]            # noqa: E712
    return e.sort_values(["act", "seed", "R2", "variant"]).reset_index(drop=True)


def replay_endpoint(name, seed, R2, variant, steps=H_STEP):
    """width2_nogating.replay_ng's operations, returning the final q."""
    import torch
    from .width2_conditional import training_set
    from .width2_nogating import act_of, gamma2
    from .width2_train import LR, _project, logits, rescale
    torch.set_num_threads(1)
    with open(NG / f"ck_{name}_{seed}.pkl", "rb") as fh:
        ck = pickle.load(fh)["ck"]
    x, y = training_set(seed)
    act = act_of(name)
    radius = 2 * R2 / gamma2(name)
    k = radius / (abs(ck["q"][2]) + abs(ck["q"][5]))
    q = torch.tensor(rescale(ck["q"], k), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=LR)
    if variant == "preserved" and ck["adam"]:
        opt.state[q] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in ck["adam"].items()}
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        _project(q, radius)
    return q.detach().numpy().copy(), radius, x, y, act


def reproduces(q, rec, tol=1e-9):
    n1 = abs(q[2]) + abs(q[5])
    got = {"share1": abs(q[2]) / n1, "alpha1": q[0], "alpha2": q[3], "beta1": q[1], "beta2": q[4]}
    return all(abs(got[k] - rec[k]) <= tol for k in got)


def to_joint(q, s, x, y, act):
    from .width2_conditional import profile_b
    v = np.array([q[2], q[5]]) / (abs(q[2]) + abs(q[5]))
    return np.r_[q[0], q[1], q[3], q[4], v, q[6]]


def from_joint(z, s):
    v = z[4:6] / (abs(z[4]) + abs(z[5])) * s
    return np.array([z[0], z[1], v[0], z[2], z[3], v[1], z[6]])


def reduced_hessian(q, x, y, act):
    import torch
    from .width2_nogating import free_basis
    from .width2_train import logits
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    f = lambda t: torch.nn.functional.binary_cross_entropy_with_logits(logits(t, X, act), Y)
    H = torch.autograd.functional.hessian(f, torch.tensor(np.asarray(q, float), dtype=torch.float64)).numpy()
    B = free_basis(q)
    Hr = B.T @ H @ B
    return 0.5 * (Hr + Hr.T), B


def transfer_direction(q):
    """δ|v₁| = +1, δ|v₂| = −1 in q's coordinates (unit norm)."""
    e = np.zeros(7)
    e[2], e[5] = np.sign(q[2]), -np.sign(q[5])
    return e / np.linalg.norm(e)


def duplicate(q, tol=DUP_TOL):
    """Units are copies up to orientation: (α₂, β₂) = σ(α₁, β₁), σ = ±1 (β compared mod 2π for f_a is not needed: the
    replays keep β unwrapped and a 2π shift is not a copy of the function without a bias change)."""
    a1, b1, a2, b2 = q[0], q[1], q[3], q[4]
    for sg in (1.0, -1.0):
        if abs(a2 - sg * a1) <= tol * max(1, abs(a1)) and abs(b2 - sg * b1) <= tol * max(1, abs(b1)):
            return True, sg
    return False, 0.0


def classify(evals, evecs, B, q, zero_rel=ZERO_REL):
    """Class of a stationary point from its reduced-Hessian spectrum (see module docstring)."""
    lmax = float(np.abs(evals).max())
    zero = np.abs(evals) <= zero_rel * lmax
    neg = evals < -zero_rel * lmax
    if neg.any():
        return "saddle"
    if not zero.any():
        return "strict minimum"
    dup, _ = duplicate(q)
    if dup and zero.sum() == 1:
        vec = B @ evecs[:, int(np.flatnonzero(zero)[0])]
        if abs(float(vec @ transfer_direction(q))) >= ALIGN:
            return "duplicate-unit minimum (Morse-Bott)"
    return "degenerate, undetermined"


def _starts_population(name, R2, s):
    """The population direct-check minimiser at the nearest R₂ (f_a only), with its orientation/mirror images."""
    f = RESULTS / "width2_direct_check.csv"
    if name == "tanh" or not f.exists():
        return []
    d = pd.read_csv(f, float_precision="round_trip")
    d = d[d.act == name]
    if not len(d):
        return []
    r = d.iloc[int(np.argmin(np.abs(np.log(d.R2.values / R2))))]
    t = float(r.p4); v = np.array([t, r.sigma * (1 - abs(t))])
    base = (np.array([r.p0, r.p1, r.p2, r.p3]), v)
    from .width2_geometry import group_images, Act
    act = Act("fa", float(name[1:]))
    return [np.r_[th[:4], vv] for th, vv in group_images(*base, act)]


def placed_best(name, R2, s, x, y, act, seed):
    """Best placed conditional minimiser found at s on (x, y); also the lowest loss of any status among polished."""
    from .width2_conditional import profile_b, search_batch
    from .width2_finish import newton
    from .width2_unplaced import g_exact, loss
    starts = _starts_population(name, R2, s)
    _, cands = search_batch(s, x, y, act, restarts=RESTARTS, seed=seed, maxit=MAXIT)
    cands = [c for c in cands[:-1] if np.all(np.isfinite(c["p"]))]
    cands.sort(key=lambda c: c["loss"])
    picked, seen = [], []
    for c in cands:
        if all(abs(c["loss"] - l0) > 1e-10 for l0 in seen):
            picked.append(c); seen.append(c["loss"])
        if len(picked) == 3 * N_POLISH:
            break
    for c in picked:
        t = float(np.clip(c["p"][4], -1, 1))
        starts.append(np.r_[c["p"][:4], t, c["sigma"] * (1 - abs(t))])
    best_pl, best_any = (math.inf, None, math.nan), (math.inf, None, math.nan)
    for q6 in starts:
        z0 = np.r_[q6, profile_b(s * (q6[4] * act.u(q6[0] * x + q6[1]) + q6[5] * act.u(q6[2] * x + q6[3])), y)]
        z, it, gmax, conv = newton(z0, s, x, y, act, maxit=300)
        L = loss(z[:6], s, x, y, act)
        g = g_exact(z[:6], act)
        if L < best_any[0]:
            best_any = (L, z, gmax)
        if g[0] > 0 and L < best_pl[0]:
            best_pl = (L, z, gmax)
    return best_pl, best_any, len(starts)


_CACHE = {}


def analyse(row, search_seed):
    from .width2_finish import newton
    from .width2_unplaced import g_exact, loss
    q, s, x, y, act = replay_endpoint(row.act, int(row.seed), float(row.R2), row.variant)
    rep = reproduces(q, row._asdict())
    z0 = to_joint(q, s, x, y, act)
    L0 = loss(z0[:6], s, x, y, act)
    z, it, gmax, conv = newton(z0, s, x, y, act, maxit=300, gtol=1e-13)
    qp = from_joint(z, s)
    L1 = loss(z[:6], s, x, y, act)
    G = g_exact(z[:6], act)
    Hr, B = reduced_hessian(qp, x, y, act)
    ev, V = np.linalg.eigh(Hr)
    klass = classify(ev, V, B, qp)
    dup, sg = duplicate(qp)
    comp = None
    if dup:                                                   # spectrum on the complement of the transfer direction
        e = B.T @ transfer_direction(qp); e /= np.linalg.norm(e)
        Pc = np.eye(len(e)) - np.outer(e, e)
        w, U = np.linalg.eigh(Pc)
        C = U[:, w > 0.5]
        comp = float(np.linalg.eigvalsh(C.T @ Hr @ C).min())
    key = (row.act, int(row.seed), round(float(row.R2), 6))
    if key not in _CACHE:
        _CACHE[key] = placed_best(row.act, float(row.R2), s, x, y, act, search_seed)
    bp, ba, n_starts = _CACHE[key]
    from .width2_unplaced import g_exact as _ge
    lowest_placed = bool(ba[1] is not None and _ge(ba[1][:6], act)[0] > 0)
    return {"act": row.act, "seed": int(row.seed), "R2": float(row.R2), "variant": row.variant, "s": s,
            "reproduced": rep, "loss_endpoint": L0, "loss_polished": L1, "polish_move": float(np.abs(qp - q).max()),
            "polish_grad": gmax, "G_lo": float(G[0]), "G_hi": float(G[1]), "duplicate": dup, "orientation": sg,
            "lam_min": float(ev.min()), "lam_max": float(ev.max()), "lam_min_rel": float(ev.min() / np.abs(ev).max()),
            "lam_second_rel": float(np.sort(ev)[1] / np.abs(ev).max()),
            "n_zero": int((np.abs(ev) <= ZERO_REL * np.abs(ev).max()).sum()),
            "lam_min_complement_rel": comp / float(np.abs(ev).max()) if comp is not None else float("nan"),
            "class": klass, "loss_placed_best": bp[0], "gap_to_placed": L1 - bp[0] if np.isfinite(bp[0]) else float("nan"),
            "loss_any_best": ba[0], "lowest_found_is_placed": lowest_placed, "gap_to_lowest": L1 - ba[0],
            "placed_min_found": bool(np.isfinite(bp[0])), "stuck_is_lowest_found": bool(L1 <= ba[0] + 1e-12), "n_starts": n_starts,
            "evals": json.dumps([float(v) for v in ev])}


def run():
    try:
        os.nice(15)
    except OSError:
        pass
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    part = OUT / "endpoints.csv"
    done = set() if not part.exists() else {(r.act, int(r.seed), round(float(r.R2), 6), r.variant)
                                             for r in pd.read_csv(part).itertuples()}
    E = endpoints()
    for i, r in enumerate(E.itertuples()):
        if (r.act, int(r.seed), round(float(r.R2), 6), r.variant) in done:
            continue
        row = analyse(r, 40_000 + int(r.seed) % 1000 + int(round(float(r.R2) * 1000)) * 1000)
        pd.DataFrame([row]).to_csv(part, mode="a", header=not part.exists(), index=False)
        print(json.dumps({k: row[k] for k in ("act", "seed", "R2", "variant", "class", "gap_to_placed")}, default=float),
              flush=True)


def check_placed(maxit=2000):
    """Convergence check of the best placed minimiser (the search in placed_best is deterministic per act, seed, R₂):
    for every (act, seed, R₂) with a placed minimiser found, re-run it, record the gradient at the best placed point,
    polish that point further (maxit Newton iterations) and record whether it stays placed and its loss."""
    from .width2_conditional import training_set
    from .width2_finish import newton
    from .width2_nogating import act_of
    from .width2_unplaced import g_exact, loss
    d = pd.read_csv(OUT / "endpoints.csv", float_precision="round_trip")
    E = endpoints()
    seeds = {(r.act, int(r.seed), round(float(r.R2), 6)): 40_000 + int(r.seed) % 1000 + int(round(float(r.R2) * 1000)) * 1000
             for r in E.itertuples()}
    rows = []
    for (a, sd, R2), g in d[d.placed_min_found.astype(bool)].groupby(["act", "seed", "R2"]):
        s = float(g.s.iloc[0]); x, y = training_set(int(sd)); act = act_of(a)
        bp, ba, _ = placed_best(a, float(R2), s, x, y, act, seeds[(a, int(sd), round(float(R2), 6))])
        z2, it, gm2, conv = newton(bp[1], s, x, y, act, maxit=maxit, gtol=1e-13)
        G2 = g_exact(z2[:6], act)
        rows.append({"act": a, "seed": int(sd), "R2": float(R2), "loss_placed_best": bp[0], "recorded": float(g.loss_placed_best.iloc[0]),
                     "grad_at_best_placed": bp[2], "loss_after_polish": loss(z2[:6], s, x, y, act), "grad_after_polish": gm2,
                     "still_placed": bool(G2[0] > 0), "G_lo_after": float(G2[0]),
                     "loss_stuck_min": float(g.loss_polished.min())})
        print(json.dumps(rows[-1], default=float), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "placed_check.csv", index=False)


LOCAL_MIN = ("strict minimum", "duplicate-unit minimum (Morse-Bott)")


def score():
    d = pd.read_csv(OUT / "endpoints.csv", float_precision="round_trip")
    rows = []
    groups = [(a, g) for a, g in d.groupby("act")] + [("f_a pooled", d[d.act != "tanh"])]
    for a, g in groups:
        pm = g[g.placed_min_found.astype(bool)]
        rows.append({"act": a, "n": len(g), "reproduced": int(g.reproduced.sum()), "duplicates": int(g.duplicate.sum()),
                     "strict_min": int((g["class"] == "strict minimum").sum()),
                     "morse_bott_min": int((g["class"] == "duplicate-unit minimum (Morse-Bott)").sum()),
                     "saddle": int((g["class"] == "saddle").sum()),
                     "undetermined": int((g["class"] == "degenerate, undetermined").sum()),
                     "frac_reduced_hessian_pd": float((g["class"] == "strict minimum").mean()),
                     "frac_conditional_local_min": float(g["class"].isin(LOCAL_MIN).mean()),
                     "median_lam_min_complement_rel": float(g.lam_min_complement_rel.median()),
                     "n_placed_min_found": int(len(pm)),
                     "gap_to_placed_median": float(pm.gap_to_placed.median()) if len(pm) else float("nan"),
                     "gap_to_placed_min": float(pm.gap_to_placed.min()) if len(pm) else float("nan"),
                     "gap_to_placed_max": float(pm.gap_to_placed.max()) if len(pm) else float("nan"),
                     "n_gap_to_placed_positive": int((pm.gap_to_placed > 0).sum()),
                     "n_lowest_found_is_placed": int(g.lowest_found_is_placed.astype(bool).sum()),
                     "n_stuck_is_lowest_found": int(g.stuck_is_lowest_found.astype(bool).sum()),
                     "gap_to_lowest_median": float(g.gap_to_lowest.median()),
                     "median_polish_move": float(g.polish_move.median())})
    S = pd.DataFrame(rows)
    by = []
    for (a, R2), g in d.groupby(["act", "R2"]):
        pm = g[g.placed_min_found.astype(bool)]
        by.append({"act": a, "R2": R2, "s": float(g.s.iloc[0]), "n": len(g),
                   "conditional_local_min": int(g["class"].isin(LOCAL_MIN).sum()),
                   "n_placed_min_found": int(len(pm)),
                   "gap_to_placed_median": float(pm.gap_to_placed.median()) if len(pm) else float("nan"),
                   "gap_to_placed_rel_median": float((pm.gap_to_placed / pm.loss_placed_best).median()) if len(pm) else float("nan"),
                   "n_lowest_found_is_placed": int(g.lowest_found_is_placed.astype(bool).sum()),
                   "n_stuck_is_lowest_found": int(g.stuck_is_lowest_found.astype(bool).sum())})
    B = pd.DataFrame(by)
    S.to_csv(OUT / "summary.csv", index=False); B.to_csv(OUT / "by_scale.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.T.to_string()); print(B.to_string(index=False))
    return S, B


if __name__ == "__main__":
    {"run": run, "score": score, "check": check_placed}[sys.argv[1]]()
