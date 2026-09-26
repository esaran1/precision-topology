"""Track T: transfer to simplicity bias (linear + 3-slab, width-4 tanh).  Design: results/simplicity_bias_design.md.

Base construction: Shah, Tamuly, Raghunathan, Jain, Netrapalli, "The Pitfalls of Simplicity Bias in Neural Networks",
NeurIPS 2020 (§3.1 building blocks: the noisy-linear block and the k-slab block; the LSN / Figure 1 linear + 3-slab
pairing).  Deviation: the linear coordinate is the *noisy* linear block (overlapping classes), d = 2, one 3-slab.

Data (per class n = N1·N2 points, a deterministic product grid of class-conditional quantiles, so x₁ ⟂ x₂ | y exactly):
  x₁ (noisy linear block, noise p):  y = 1: (1 − p)·U[0.1, 1] + p·U[−0.1, 0.1];  y = 0: the mirror image.
  x₂ (3-slab block, equal widths, gaps 0.2 = the linear block's gap):  y = 0 in the middle slab [−h, h];
      y = 1 in the two outer slabs ±[h + 0.2, 1], half the class mass in each, with 3w + 0.4 = 2, w = 2h.
The same fixed point set is the population objective of the landscape and (Step 2) the training set.

Network: N(x) = s·Σₖ ṽₖ tanh(wₖ·x + cₖ) + b, width 4, ‖ṽ‖₁ = 1, s = ‖w₂‖₁.  tanh is odd, so each output sign is
absorbed into its hidden unit and ṽ lies on the simplex: ṽₖ = ηₖ²/‖η‖² (smooth, no ℓ₁ kink).  b is profiled exactly.

Rule for the geometry (computed from the objective, not from training): with balanced classes and b profiled,
  L*(s) = log 2 − (s/4)·Δ* + O(s²),   Δ* = sup_φ (E[φ | y=1] − E[φ | y=0]) over the unit ℓ₁ hull of tanh units.
Δ is linear in φ, so the sup over the hull is the sup over single units; tanh(z) = ∫ sign(z − τ) dF(τ) with
F = (1 + tanh)/2 a CDF, so a unit's gap is an average of halfspace gaps and the sup over units is the sup over
halfspaces: Δ* = sup_θ Δ(θ),  Δ(θ) = 2·max_τ |P₁(u > τ) − P₀(u > τ)|,  u = x·(cos θ, sin θ).  Computed exactly on the
point set for every direction of a fine grid (`gap_rule`).

    python -m src.simplicity_bias rule              # the parameter rule (design commit)
    python -m src.simplicity_bias pilot SET         # Step 1 pilot, one restart set per worker (A or B), resumable
    python -m src.simplicity_bias summarise         # gate verdict (EXPLORATORY pilot)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from .width2_conditional import bfgs_batch, profile_b_batch, _sig, _softplus

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "simplicity_bias"
WIDTH = 4
NPAR = 4 * WIDTH                     # W (4×2), c (4), η (4)
GAP = 0.2                            # the linear block's gap [−0.1, 0.1], reused between slabs
PILOT_P = 0.20                       # noise fraction: the best x₁ threshold misclassifies p/2 = 10% of each class
N1, N2 = 20, 20                      # quantile levels per coordinate per class: 400 points per class


# ------------------------------------------------------------------------------------------ data
def _uniform_quantiles(lo, hi, k):
    return lo + (hi - lo) * (np.arange(k) + 0.5) / k


X1_OFFSET = 0.25


def linear_block_levels(p, n=N1):
    """Class-1 x₁ quantile levels of (1 − p)·U[0.1, 1] + p·U[−0.1, 0.1] at (k + ¼)/n (mixture quantile function).
    Class 0 takes the mirror image, so the two classes' levels interleave in the noise band with no cross-class tie
    (with (k + ½)/n the noise levels would coincide across classes and an oblique unit could use x₂ to break the
    ties, a discretisation artefact)."""
    u = (np.arange(n) + X1_OFFSET) / n
    return np.where(u < p, -0.1 + 0.2 * u / max(p, 1e-300), 0.1 + 0.9 * (u - p) / max(1 - p, 1e-300))


def slab_levels(n=N2):
    """(class-0 middle-slab levels, class-1 outer-slab levels); equal widths w, gaps GAP: 3w + 2·GAP = 2."""
    w = (2 - 2 * GAP) / 3; h = w / 2
    mid = _uniform_quantiles(-h, h, n)
    upper = _uniform_quantiles(h + GAP, 1.0, n // 2)
    return mid, np.concatenate([-upper[::-1], upper])


def make_data(p=PILOT_P, n1=N1, n2=N2):
    """The fixed point set: per class the product grid (x₁ levels) × (x₂ levels); y ∈ {0, 1}; x₁ ⟂ x₂ | y exactly."""
    l1 = linear_block_levels(p, n1)
    mid, outer = slab_levels(n2)
    X1 = np.array([(a, b) for a in l1 for b in outer]); X0 = np.array([(-a, b) for a in l1 for b in mid])
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    return X, y


# ------------------------------------------------------------------------------------------ the parameter rule
def halfspace_gap(u, y):
    """Δ = 2·max over thresholds τ of |P₁(u > τ) − P₀(u > τ)|, exact on a finite set (tied u kept on one side)."""
    order = np.argsort(u, kind="stable"); us = u[order]; ys = y[order]
    n1, n0 = ys.sum(), (1 - ys).sum()
    c1 = np.cumsum(ys[::-1])[::-1]; c0 = np.cumsum((1 - ys)[::-1])[::-1]     # counts with u ≥ us[i]
    first = np.r_[True, us[1:] != us[:-1]]                                    # thresholds just below each distinct u
    d = np.abs(c1[first] / n1 - c0[first] / n0)
    return float(2 * d.max())


def gap_rule(X, y, n_dir=7200):
    """Δ(θ) for θ on a uniform grid of [0, π) (Δ(θ + π) = Δ(θ)).  Returns Δ_lin = Δ(0) (x₁ only), Δ_slab = Δ(π/2)
    (any x₂-only function in the ℓ₁ hull: by linearity its gap is at most the best single x₂ unit's), Δ_all and the
    set of maximising directions."""
    th = np.pi * np.arange(n_dir) / n_dir
    D = np.array([halfspace_gap(X @ np.array([math.cos(t), math.sin(t)]), y) for t in th])
    d_lin = D[0]; d_slab = D[n_dir // 2]; d_all = D.max()
    arg = np.degrees(th[D >= d_all - 1e-12])
    return {"delta_lin": d_lin, "delta_slab": d_slab, "delta_all": float(d_all), "ratio_lin_slab": d_lin / d_slab,
            "n_maximising_directions": int(len(arg)),
            "max_tilt_from_x1_deg_among_maximisers": float(np.minimum(arg, 180 - arg).max()),
            "pure_x1_attains_sup": bool(d_lin >= d_all - 1e-12), "n_dir": n_dir,
            "best_x1_error_rate": best_threshold_error(X[:, 0], y)}


def best_threshold_error(u, y):
    """Smallest balanced error of any threshold classifier on u (both orientations)."""
    order = np.argsort(u, kind="stable"); us = u[order]; ys = y[order]
    n1, n0 = ys.sum(), (1 - ys).sum()
    c1 = np.r_[np.cumsum(ys[::-1])[::-1], 0]; c0 = np.r_[np.cumsum((1 - ys)[::-1])[::-1], 0]
    first = np.r_[np.r_[True, us[1:] != us[:-1]], True]
    err = 0.5 * ((n1 - c1[first]) / n1 + c0[first] / n0)
    return float(min(err.min(), (1 - err).min()))


RULE_MIN_RATIO = 1.2


def rule_ok(r):
    """The design rule: the linear feature's small-scale gap exceeds every x₂-only configuration's by the stated
    factor, and no direction (oblique units included) beats the pure-x₁ direction."""
    return bool(r["ratio_lin_slab"] >= RULE_MIN_RATIO and r["pure_x1_attains_sup"])


# ------------------------------------------------------------------------------------------ the network
def unpack(P):
    P = np.atleast_2d(P)
    m = len(P)
    return P[:, :8].reshape(m, WIDTH, 2), P[:, 8:12], P[:, 12:16]


def vtilde(eta):
    q = eta ** 2
    return q / q.sum(axis=-1, keepdims=True)


def phi(P, X):
    """Unit-ℓ₁ network function φ (m, n) for parameter rows P (m, 16)."""
    W, c, eta = unpack(P)
    H = np.tanh(np.einsum("mkj,nj->mkn", W, X) + c[:, :, None])
    return np.einsum("mk,mkn->mn", vtilde(eta), H)


def loss_grad_batch(P, s, X, y):
    """L*(P; s) (b profiled) and its gradient in P (envelope theorem), for a batch of rows."""
    W, c, eta = unpack(P)
    m, n = len(P), len(y)
    H = np.tanh(np.einsum("mkj,nj->mkn", W, X) + c[:, :, None])
    q = eta ** 2; Q = q.sum(axis=1, keepdims=True); v = q / Q
    Z0 = s * np.einsum("mk,mkn->mn", v, H)
    b = profile_b_batch(Z0, y)
    Z = Z0 + b[:, None]
    L = (_softplus(Z) - y[None, :] * Z).mean(axis=1)
    R = s * (_sig(Z) - y[None, :]) / n                                     # dL/dφ_n
    D = (1 - H ** 2) * (v[:, :, None] * R[:, None, :])                      # (m, k, n)
    gW = np.einsum("mkn,nj->mkj", D, X).reshape(m, 8)
    gc = D.sum(axis=2)
    gv = np.einsum("mkn,mn->mk", H, R)
    geta = 2 * eta / Q * (gv - (v * gv).sum(axis=1, keepdims=True))
    return L, np.concatenate([gW, gc, geta], axis=1)


def gplus(P, X, y):
    """G₊(φ) = ½(min_{y=1} φ − max_{y=0} φ): > 0 iff some bias separates every point (class 1 above)."""
    F = phi(P, X)
    return 0.5 * (F[:, y == 1].min(axis=1) - F[:, y == 0].max(axis=1))


def all_correct(P, s, X, y):
    """Sign-correct everywhere with the profiled bias (positive margin at every point)."""
    Z0 = s * phi(P, X); b = profile_b_batch(Z0, y)
    Z = Z0 + b[:, None]
    return ((2 * y[None, :] - 1) * Z > 0).all(axis=1)


# ------------------------------------------------------------------------------------------ feature usage
def feature_usage(p, X):
    """For one parameter row: V_j = mean over all pairs (i, k) of |φ(x_i) − φ(x_i with coordinate j := x_{k,j})|
    (the paper's S-randomisation, made deterministic by averaging over every replacement), their share
    ρ₂ = V₂/(V₁ + V₂), and the gradient sensitivities S_j = mean_i |∂φ/∂x_j(x_i)| (logged; not the classifier, since
    saturated units have near-zero derivative at the data)."""
    p = np.asarray(p, float)
    W, c, eta = unpack(p[None]); W, c, v = W[0], c[0], vtilde(eta)[0]
    base = phi(p[None], X)[0]
    V = []
    for j in range(2):
        vals = np.unique(X[:, j])                                          # replacement values, weighted by counts
        cnt = np.array([(X[:, j] == t).sum() for t in vals], float)
        tot = 0.0
        for t, w in zip(vals, cnt):
            Xr = X.copy(); Xr[:, j] = t
            tot += w * np.abs(phi(p[None], Xr)[0] - base).sum()
        V.append(tot / (len(X) ** 2))
    Hd = 1 - np.tanh(X @ W.T + c) ** 2                                     # (n, k)
    S = np.abs(Hd @ (v[:, None] * W)).mean(axis=0)
    V1, V2 = V
    rho2 = V2 / (V1 + V2) if V1 + V2 > 0 else float("nan")
    return {"V1": V1, "V2": V2, "rho2": rho2, "S1": float(S[0]), "S2": float(S[1]),
            "feature": classify_feature(rho2)}


def classify_feature(rho2):
    """Dominant feature by the S-randomised share: 'linear' if ρ₂ < ½, 'slab' if ρ₂ > ½."""
    if not np.isfinite(rho2):
        return "none"
    return "slab" if rho2 > 0.5 else ("linear" if rho2 < 0.5 else "tie")


# ------------------------------------------------------------------------------------------ the validated search
GTOL = 1e-8
LADDER = (200, 400, 800)
LADDER_TOL = 1e-7
CMA_TOL = 1e-7


def starts(rng, k, family):
    """Set A: uniform box (W, c ~ U[−6, 6], η ~ U[0.2, 1]); set B: Gaussian (W ~ N(0, 3²), c ~ N(0, 2²),
    η ~ |N(0, 1)| + 0.05).  Different seeds and different start distributions: independent restart sets."""
    if family == "A":
        return np.concatenate([rng.uniform(-6, 6, (k, 12)), rng.uniform(0.2, 1, (k, 4))], axis=1)
    return np.concatenate([rng.normal(0, 3, (k, 8)), rng.normal(0, 2, (k, 4)),
                           np.abs(rng.normal(0, 1, (k, 4))) + 0.05], axis=1)


SET_SEED = {"A": 11, "B": 23}


def flags(gn, f_end, f90, gtol=GTOL):
    """Excluding flags: non-finite; not converged = gradient above 100·gtol AND still improving over the last 10% of
    iterations by more than 1e−10 (a saturating tail that has stopped improving is 'info:stalled_tail')."""
    fl = []
    if not np.isfinite(f_end) or not np.isfinite(gn):
        fl.append("nonfinite")
    elif gn > 100 * gtol:
        if np.isfinite(f90) and f90 - f_end <= 1e-10:
            fl.append("info:stalled_tail")
        else:
            fl.append("not_converged")
    return fl


EXCLUDING = {"nonfinite", "not_converged"}


def constant_loss(y):
    p = y.mean()
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def search(s, X, y, restarts, family, seed, maxit=3000, batch=200):
    rng = np.random.default_rng(seed)
    P0 = starts(rng, restarts, family)
    cands = []
    for i in range(0, restarts, batch):
        P, L, G, it = bfgs_batch(lambda Q, rows: loss_grad_batch(Q, s, X, y), P0[i:i + batch], gtol=GTOL, maxit=maxit)
        f90 = bfgs_batch.last_f90
        for k in range(len(P)):
            eta = P[k, 12:]; P[k, 12:] = eta / np.linalg.norm(eta)
            gn = float(np.abs(G[k]).max())
            cands.append({"k": i + k, "p": P[k].copy(), "loss": float(L[k]), "gnorm": gn, "iters": int(it[k]),
                          "flags": flags(gn, float(L[k]), float(f90[k]))})
    return cands


def retain(cands, y):
    ok = [c for c in cands if not (set(c["flags"]) & EXCLUDING)]
    const = {"k": -1, "p": None, "loss": constant_loss(y), "flags": ["constant_predictor"]}
    return min(ok + [const], key=lambda c: c["loss"])


def audit(cands, retained, tol=LADDER_TOL):
    """Stop condition: an excluded (unconverged / non-finite) candidate below the retained loss by more than tol."""
    bad = [c for c in cands if (set(c["flags"]) & EXCLUDING) and np.isfinite(c["loss"])
           and c["loss"] < retained["loss"] - tol]
    return {"ok": not bad, "n_below": len(bad)}


def independent_search(s, X, y, starts_n=16, seed=7, gens=300):
    """CMA-ES on a different parametrisation: signed output weights u ∈ ℝ⁴ normalised to the ℓ₁ sphere, no
    simplex/odd-symmetry reduction.  Returns the best loss and the equivalent 16-vector (signs moved into units)."""
    from .cmaes import cma_es
    rng = np.random.default_rng(seed)

    def to_P(q):
        W = q[:8].reshape(4, 2).copy(); c = q[8:12].copy(); u = q[12:16]
        sg = np.where(u < 0, -1.0, 1.0); W *= sg[:, None]; c *= sg
        return np.concatenate([W.ravel(), c, np.sqrt(np.abs(u) / max(np.abs(u).sum(), 1e-300))])

    def f(q):
        if np.abs(q[12:16]).sum() < 1e-12:
            return 1e9
        return float(loss_grad_batch(to_P(q)[None], s, X, y)[0][0])
    best = (math.inf, None)
    for _ in range(starts_n):
        q0 = np.concatenate([rng.uniform(-6, 6, 12), rng.uniform(-1, 1, 4)])
        r = cma_es(f, q0, 2.0, max_generations=gens, seed=int(rng.integers(1 << 31)))
        if r.best_f < best[0]:
            best = (r.best_f, to_P(r.best_x))
    return {"loss": best[0], "p": best[1]}


# ------------------------------------------------------------------------------------------ the gate (committed before the pilot)
AGREE_TOL = 0.02
BISECT_REL = 0.005


def switch_from_sequence(scales, g):
    """Sign sequence of G₊ of the retained minimiser over increasing scales.  Returns the number of sign changes and
    the bracket of the first ≤ 0 → > 0 change."""
    scales = np.asarray(scales, float); pos = np.asarray(g) > 0
    order = np.argsort(scales); scales, pos = scales[order], pos[order]
    changes = int((pos[1:] != pos[:-1]).sum())
    up = np.flatnonzero(~pos[:-1] & pos[1:])
    br = (float(scales[up[0]]), float(scales[up[0] + 1])) if len(up) else None
    return {"sign_changes": changes, "bracket": br, "starts_negative": bool(not pos[0]), "ends_positive": bool(pos[-1])}


def gate(sets):
    """sets: {name: {"points": [{"s", "gplus", "feature", "validated"}...], "switch": s_hat or None}}.
    Gate (all must hold):
      G1 one clean switch: in every set, G₊ of the retained minimiser changes sign exactly once over all computed
         scales, from ≤ 0 (below) to > 0 (above), inside the scanned range;
      G2 feature usage: every computed scale below the switch has the linear feature dominant (ρ₂ < ½) and every one
         above has the slab feature dominant (ρ₂ > ½);
      G3 agreement: the two sets' refined switch scales agree within 2% (|s_A − s_B| / min ≤ 0.02);
      G4 validation: every point flagged for validation passed (ladder, audit, CMA-ES).
    Returns the verdict with every sub-result."""
    res = {"per_set": {}}
    ok1 = ok2 = ok4 = True
    sw = {}
    for name, S in sets.items():
        pts = sorted(S["points"], key=lambda r: r["s"])
        seq = switch_from_sequence([r["s"] for r in pts], [r["gplus"] for r in pts])
        g1 = seq["sign_changes"] == 1 and seq["starts_negative"] and seq["ends_positive"]
        g2 = g1 and all((r["feature"] == "linear") if r["gplus"] <= 0 else (r["feature"] == "slab") for r in pts)
        g4 = all(r.get("validated", True) for r in pts)
        res["per_set"][name] = {"G1": g1, "G2": g2, "G4": g4, **seq, "switch": S.get("switch")}
        ok1 &= g1; ok2 &= g2; ok4 &= g4
        sw[name] = S.get("switch")
    vals = [v for v in sw.values() if v is not None]
    if len(vals) == len(sw) and len(vals) >= 2:
        rel = (max(vals) - min(vals)) / min(vals)
        ok3 = rel <= AGREE_TOL
    else:
        rel, ok3 = None, False
    res.update({"G1": ok1, "G2": ok2, "G3": ok3, "G4": ok4, "agreement_rel": rel,
                "pass": bool(ok1 and ok2 and ok3 and ok4)})
    return res


# ------------------------------------------------------------------------------------------ pilot (EXPLORATORY)
PILOT_GRID = tuple(float(v) for v in np.round(0.5 * 1.2 ** np.arange(24), 6))      # 0.5 … 32.9, ratio 1.2


def _init_worker():
    try:
        os.nice(15)
    except OSError:
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def evaluate_scale(s, X, y, family, validate=False):
    """The validated search at one scale: 800 restarts (nested ladder 200 → 400 → 800), retained minimiser, audit;
    with validate=True also the CMA-ES independent search."""
    cands = search(s, X, y, max(LADDER), family, SET_SEED[family] * 100_003 + int(round(s * 1e6)) % 99_991)
    lad = []
    for n in LADDER:
        r = retain(cands[:n], y)
        g = float(gplus(r["p"][None], X, y)[0]) if r["p"] is not None else float("-inf")
        fu = feature_usage(r["p"], X)["feature"] if r["p"] is not None else "none"
        lad.append({"n": n, "loss": r["loss"], "gplus_pos": bool(g > 0), "feature": fu})
    ret = retain(cands, y)
    au = audit(cands, ret)
    p = ret["p"]
    fu = feature_usage(p, X) if p is not None else {"feature": "none"}
    g = float(gplus(p[None], X, y)[0]) if p is not None else float("-inf")
    ladder_ok = (abs(lad[1]["loss"] - lad[2]["loss"]) <= LADDER_TOL
                 and len({(l["gplus_pos"], l["feature"]) for l in lad}) == 1)
    losses = np.array([c["loss"] for c in cands])
    within = int((np.abs(losses - ret["loss"]) <= LADDER_TOL).sum())
    rec = {"s": s, "set": family, "loss": ret["loss"], "gplus": g, "all_correct": bool(all_correct(p[None], s, X, y)[0])
           if p is not None else False, **{k: fu[k] for k in fu}, "p": None if p is None else [float(t) for t in p],
           "ladder": lad, "ladder_ok": bool(ladder_ok), "audit": au, "hits_within_tol": within,
           "n_excluded": int(sum(bool(set(c["flags"]) & EXCLUDING) for c in cands)),
           "n_stalled_tail": int(sum("info:stalled_tail" in c["flags"] for c in cands))}
    if validate:
        ind = independent_search(s, X, y, seed=SET_SEED[family] + 1)
        rec["cma"] = {"loss": ind["loss"], "ok": bool(ind["loss"] >= ret["loss"] - CMA_TOL)}
    rec["validated"] = bool(rec["ladder_ok"] and au["ok"] and rec.get("cma", {"ok": True})["ok"])
    return rec


def _part(family, s):
    return OUT / "pilot_parts" / f"{family}_{s:.6f}.json"


def _run_point(s, X, y, family, validate=False):
    f = _part(family, s)
    if f.exists():
        r = json.loads(f.read_text())
        if not validate or "cma" in r:
            return r
    t0 = time.time()
    r = evaluate_scale(s, X, y, family, validate=validate)
    r["seconds"] = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(r)); tmp.replace(f)
    print(json.dumps({k: r[k] for k in ("set", "s", "loss", "gplus", "rho2", "feature", "validated", "seconds")}),
          flush=True)
    return r


def run_pilot(family):
    """One restart set: the grid, then bisection of the first G₊ sign change to 0.5% (geometric midpoints), then
    validation (ladder, audit, CMA-ES) at the final bracket ends.  Resumable (one JSON per scale)."""
    _init_worker()
    X, y = make_data(PILOT_P)
    pts = [_run_point(s, X, y, family) for s in PILOT_GRID]
    seq = switch_from_sequence([r["s"] for r in pts], [r["gplus"] for r in pts])
    if seq["bracket"] is None:
        print(json.dumps({"set": family, "status": "no switch in grid", **seq}), flush=True)
        return
    lo, hi = seq["bracket"]
    while hi / lo - 1 > BISECT_REL:
        mid = float(np.round(math.sqrt(lo * hi), 6))
        r = _run_point(mid, X, y, family)
        if r["gplus"] > 0:
            hi = mid
        else:
            lo = mid
    for s in (lo, hi):
        _run_point(s, X, y, family, validate=True)
    print(json.dumps({"set": family, "status": "done", "bracket": [lo, hi]}), flush=True)


def summarise():
    X, y = make_data(PILOT_P)
    sets = {}
    for fam in ("A", "B"):
        rows = [json.loads(f.read_text()) for f in sorted((OUT / "pilot_parts").glob(f"{fam}_*.json"))]
        seq = switch_from_sequence([r["s"] for r in rows], [r["gplus"] for r in rows])
        switch = None
        if seq["bracket"] is not None:
            switch = math.sqrt(seq["bracket"][0] * seq["bracket"][1])
        # validation is required at the bracket ends (only those carry the CMA-ES search); other points report theirs
        br = seq["bracket"] or (None, None)
        pts = [{"s": r["s"], "gplus": r["gplus"], "feature": r["feature"],
                "validated": r["validated"] if r["s"] in br else True} for r in rows]
        sets[fam] = {"points": pts, "switch": switch, "rows": rows}
    verdict = gate({k: {"points": v["points"], "switch": v["switch"]} for k, v in sets.items()})
    table = []
    for fam, v in sets.items():
        for r in sorted(v["rows"], key=lambda r: r["s"]):
            table.append({"set": fam, "s": r["s"], "loss": r["loss"], "gplus": r["gplus"],
                          "all_correct": r["all_correct"], "V1": r["V1"], "V2": r["V2"], "rho2": r["rho2"],
                          "S1": r["S1"], "S2": r["S2"], "feature": r["feature"], "ladder_ok": r["ladder_ok"],
                          "audit_ok": r["audit"]["ok"], "cma_ok": r.get("cma", {}).get("ok"),
                          "cma_loss": r.get("cma", {}).get("loss"), "hits_within_tol": r["hits_within_tol"],
                          "n_excluded": r["n_excluded"], "n_stalled_tail": r["n_stalled_tail"],
                          "grid": r["s"] in PILOT_GRID})
    import pandas as pd
    pd.DataFrame(table).to_csv(OUT / "pilot_scan.csv", index=False)
    allrows = [r for v in sets.values() for r in v["rows"]]
    hedge = [{"set": r["set"], "s": r["s"], "loss": r["loss"], "hedge_closed_form": hedge_loss(r["s"]),
              "diff": r["loss"] - hedge_loss(r["s"])} for r in allrows if r["s"] <= 1.0]
    cross = []
    for s_ in PILOT_GRID:
        a = [r for r in sets["A"]["rows"] if r["s"] == s_]; b = [r for r in sets["B"]["rows"] if r["s"] == s_]
        if a and b:
            cross.append({"s": s_, "loss_A_minus_B": a[0]["loss"] - b[0]["loss"]})
    diag = {"n_points": len(allrows), "n_ladder_fail": sum(not r["ladder_ok"] for r in allrows),
            "n_audit_fail": sum(not r["audit"]["ok"] for r in allrows),
            "hits_within_tol_median": float(np.median([r["hits_within_tol"] for r in allrows])),
            "n_points_single_hit": sum(r["hits_within_tol"] == 1 for r in allrows),
            "max_abs_hidden_weight_median": float(np.median([np.abs(np.array(r["p"])[:8]).max() for r in allrows])),
            "cma": {f'{r["set"]}_{r["s"]}': r["cma"] for r in allrows if "cma" in r},
            "small_s_hedge_check": hedge, "cross_set_loss_difference_on_grid": cross,
            "max_abs_cross_set_loss_difference": max(abs(c["loss_A_minus_B"]) for c in cross),
            "rho2_at_first_separating_point": {k: min((r for r in v["rows"] if r["gplus"] > 0), key=lambda r: r["s"])["rho2"]
                                               for k, v in sets.items()},
            "feature_class_changes_above_switch": {k: _changes([r["feature"] for r in sorted(v["rows"], key=lambda r: r["s"])
                                                                if r["gplus"] > 0]) for k, v in sets.items()}}
    summ = {"label": "EXPLORATORY (Step 1 pilot; nothing registered)", "p": PILOT_P, "n_per_class": N1 * N2,
            "grid": list(PILOT_GRID), "gate": verdict,
            "switch": {k: v["switch"] for k, v in sets.items()}, "diagnostics": diag}
    (OUT / "pilot_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def run_hull(iters=80):
    """EXPLORATORY diagnostic (added after the pilot showed saturated units): the hard-unit hull minimum (any width)
    at every pilot grid scale, with its G₊ and feature share, beside the width-4 retained losses."""
    _init_worker()
    import pandas as pd
    X, y = make_data(PILOT_P)
    scan = pd.read_csv(OUT / "pilot_scan.csv")
    rows = []
    for s in PILOT_GRID:
        r = hard_hull_minimum(s, X, y, iters=iters)
        w4 = scan[np.isclose(scan.s, s)].loss.min()
        rows.append({k: r[k] for k in ("s", "loss", "fw_gap", "iters", "n_active", "gplus", "V1", "V2", "rho2")} |
                    {"width4_best_retained_loss": float(w4), "width4_minus_hull": float(w4 - r["loss"])})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "hull_diagnostic.csv", index=False)


def hedge_loss(s):
    """Closed-form loss of the hard 'hedged' linear function ½[sign(x₁ − t₋) + sign(x₁ − t₊)] (t± at the noise-band
    edges): class-clean points at ±1, the whole noise band (20% of each class) at 0, b = 0 by symmetry."""
    return (1 - PILOT_P) * math.log1p(math.exp(-s)) + PILOT_P * math.log(2)


def _changes(seq):
    return int(sum(a != b for a, b in zip(seq[:-1], seq[1:])))


def run_rule():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = make_data(PILOT_P)
    r = gap_rule(X, y)
    r.update({"p": PILOT_P, "n_per_class": int((y == 1).sum()), "rule_min_ratio": RULE_MIN_RATIO,
              "rule_ok": rule_ok(r), "slab_levels_mid": [float(t) for t in slab_levels()[0][[0, -1]]],
              "slab_levels_outer": [float(t) for t in slab_levels()[1][[0, len(slab_levels()[1]) // 2 - 1,
                                                                          len(slab_levels()[1]) // 2, -1]]],
              "first_order_slope_lin": r["delta_lin"] / 4, "first_order_slope_slab": r["delta_slab"] / 4})
    (OUT / "rule.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


# ------------------------------------------------------------------------------------------ diagnostic (EXPLORATORY, added after the
# first pilot points showed saturated units, |w| ~ 1e4): the infimum over the whole ℓ₁ hull of HARD halfspace units
def _halfspace_oracle(X, r, n_dir=1440):
    """argmax over halfspaces h(x) = sign(x·d − τ) of |Σ r_i h(x_i)| (exact per direction on the finite set).
    Returns (value, d, τ, sign) with the maximiser oriented so that Σ r_i h(x_i) is minimised (negative)."""
    best = (-1.0, None, None, 1)
    tot = r.sum()
    for t in np.pi * np.arange(n_dir) / n_dir:
        d = np.array([math.cos(t), math.sin(t)])
        u = X @ d; o = np.argsort(u, kind="stable"); us = u[o]; rs = r[o]
        above = tot - np.r_[0.0, np.cumsum(rs)]                              # Σ r over u ≥ us[i] (i = 0..n)
        first = np.r_[True, us[1:] != us[:-1], True]
        val = 2 * above[first] - tot                                          # Σ r_i sign(u_i − τ) for τ just below
        k = int(np.argmax(np.abs(val)))
        if abs(val[k]) > best[0]:
            idx = np.flatnonzero(first)[k]
            tau = us[idx] - 1e-9 if idx < len(us) else us[-1] + 1e-9
            best = (abs(val[k]), d, tau, -1.0 if val[k] > 0 else 1.0)
    return best


def hard_hull_minimum(s, X, y, iters=200, n_dir=1440, tol=1e-10):
    """Fully corrective Frank–Wolfe over the convex hull of hard units ±sign(x·d − τ) (unit ℓ₁ output, any width),
    b profiled.  Returns the loss (an upper bound on the hull infimum that the duality gap brackets), the gap, the
    active units with weights, G₊ and the feature share of the hull minimiser."""
    cols = []                                                                  # each column: h(x_i) values
    meta = []
    w = np.zeros(0)

    def fg(wv):
        F = (np.array(cols).T @ wv) if cols else np.zeros(len(y))
        Z0 = s * F; b = profile_b_batch(Z0[None], y)[0]; Z = Z0 + b
        L = float((_softplus(Z) - y * Z).mean()); R = s * (_sig(Z) - y) / len(y)
        return L, R, F
    L, R, F = fg(w)
    gap = math.inf
    for it in range(iters):
        val, d, tau, sg = _halfspace_oracle(X, R, n_dir)
        h = sg * np.sign(X @ d - tau)
        gap = float((R * F).sum() - (R * h).sum()) if cols else math.inf     # FW duality gap ⟨∇, φ − h⟩
        if cols and gap <= tol:
            break
        cols.append(h); meta.append((float(d[0]), float(d[1]), float(tau), float(sg)))
        w = np.r_[w * (1 - 1e-3), 1e-3] if len(w) else np.array([1.0])
        w = w / w.sum()
        eta = 1.0
        L, R, F = fg(w)
        for _ in range(600):                                                   # exponentiated gradient, backtracking
            gw = np.array(cols) @ R
            while True:
                w2 = w * np.exp(-eta * (gw - gw.min()) / (np.abs(gw).max() + 1e-300)); w2 /= w2.sum()
                L2, R2, F2 = fg(w2)
                if L2 <= L or eta < 1e-12:
                    break
                eta *= 0.5
            done = np.abs(w2 - w).max() < 1e-13
            w, L, R, F = w2, L2, R2, F2
            eta = min(eta * 2, 1e3)
            if done:
                break
    keep = w > 1e-6
    G = 0.5 * (F[y == 1].min() - F[y == 0].max())
    # feature share of the hull minimiser (V_j as in feature_usage, on the hard function)
    V = []
    for j in range(2):
        vals, cnt = np.unique(X[:, j], return_counts=True)
        tot = 0.0
        for t, c in zip(vals, cnt):
            Xr = X.copy(); Xr[:, j] = t
            Fr = np.array([sg * np.sign(Xr @ np.array([a, b]) - tau) for a, b, tau, sg in meta]).T @ w
            tot += c * np.abs(Fr - F).sum()
        V.append(tot / len(X) ** 2)
    return {"s": s, "loss": L, "fw_gap": gap, "iters": it + 1, "n_active": int(keep.sum()), "gplus": float(G),
            "V1": V[0], "V2": V[1], "rho2": V[1] / (V[0] + V[1]) if V[0] + V[1] > 0 else float("nan"),
            "units": [dict(zip(("d1", "d2", "tau", "sign"), m), weight=float(wt)) for m, wt in zip(meta, w) if wt > 1e-6]}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rule":
        run_rule()
    elif cmd == "pilot":
        run_pilot(sys.argv[2])
    elif cmd == "summarise":
        summarise()
    elif cmd == "hull":
        run_hull()
