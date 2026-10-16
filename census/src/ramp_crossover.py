"""Ramp-limit crossover on Track 2's registered asymmetric landscape (POST HOC, EXPLORATORY; float, not a certificate).

Question (author, 2026-10-16): below which scale does the ramp-limit minimum (math note §22, Lemma K7) beat the
registered width-2 retained loss on the asymmetric windows, and does every registered width-2 quantity lie above it?

Objective: Track 2's registered landscape P_an = asym_pilot.population(0.4) (800 points), f_a with a = 1.30, the
conditional loss L*(theta; s) = min_b mean l(s*phi + b, y) (width2_conditional).  Every registered width-2 quantity on
the asymmetric windows is at a = 1.30 and Delta = 0.4 (results/ramp_crossover_check.md lists them).

Ramp-limit minimum R(s) = min over (kappa, b, alpha, beta) of mean l(kappa x + s a sin(alpha x + beta) + b, y), with
c_bar = kappa / s the total slope (phi_bar = c_bar x + const + a sin(alpha x + beta); sigma = -1 is beta + pi, and
alpha >= 0 by symmetry).  For fixed (alpha, beta) the problem is convex in (kappa, b) and is solved by Newton
(vectorised); the outer (alpha, beta) search uses two INDEPENDENT start sets (set A: the §22.6 grid, 150 x 72 on
alpha in (0, 10], beta in [0, 2 pi); set B: 3,000 seeded uniform draws on alpha in (0, 12], beta in [0, 2 pi)), then a
4-D damped Newton polish of the lowest distinct candidates of each set.  The two sets must agree (loss within 1e-12).

Mirror range (Lemma K8): the ramp-limit point is realised by a compact width-2 configuration iff |c_bar| <= alpha.
s_ramp is bracketed by bisection (in log s) on the sign of alpha - |c_bar| between the committed scan scales 0.1334
(outside) and 0.1778 (inside).  The registered retained loss is the committed value (asym_parts/*.csv) at registered
scales; at the new scales near s_ramp and at s0 it is recomputed POST HOC with the registered code path, unmodified
(asym_register.eval_scale, 1,000 restarts, cap 3,000, 5 polished), seeds 910 + i.

One process, nice 15, one thread; about 15 minutes (the registered-path evaluations dominate, ~160 s each, 0.35 GB).

    python -m src.ramp_crossover
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ramp_crossover.json"
A = 1.30
DELTA = 0.4
TWO_PI = 2 * math.pi
AGREE_TOL = 1e-12
BISECT_STEPS = 16
REG_SEED0 = 910


# --------------------------------------------------------------------------------------------- the ramp-limit loss
def _softplus(z):
    return np.logaddexp(0, z)


def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def inner_newton(W, x, y, iters=60):
    """For each row w of W (m, n): min over (kappa, b) of mean l(kappa x + w + b, y).  Damped Newton (strictly convex
    when x is not constant).  Returns (L, kappa, b, gnorm), each of shape (m,)."""
    m = W.shape[0]
    k = np.zeros(m); b = np.zeros(m)
    f = lambda k, b: (_softplus(k[:, None] * x + W + b[:, None]) - y * (k[:, None] * x + W + b[:, None])).mean(1)
    L = f(k, b)
    for _ in range(iters):
        z = k[:, None] * x + W + b[:, None]
        r = _sig(z) - y; q = _sig(z) * (1 - _sig(z))
        gk, gb = (r * x).mean(1), r.mean(1)
        hkk, hkb, hbb = (q * x * x).mean(1), (q * x).mean(1), q.mean(1)
        det = hkk * hbb - hkb ** 2
        dk = (hbb * gk - hkb * gb) / det
        db = (hkk * gb - hkb * gk) / det
        t = np.ones(m)
        for _ in range(30):
            Ln = f(k - t * dk, b - t * db)
            bad = Ln > L + 1e-15
            if not bad.any():
                break
            t = np.where(bad, t / 2, t)
        k, b, L = k - t * dk, b - t * db, np.minimum(L, f(k - t * dk, b - t * db))
        if np.max(np.abs(gk) + np.abs(gb)) < 1e-15:
            break
    z = k[:, None] * x + W + b[:, None]
    r = _sig(z) - y
    return L, k, b, np.abs((r * x).mean(1)) + np.abs(r.mean(1))


def ramp_profile(alpha, beta, s, x, y, a=A, chunk=1500):
    """R(alpha, beta; s): the (kappa, b)-profiled ramp-limit loss for arrays alpha, beta."""
    alpha, beta = np.atleast_1d(alpha).astype(float), np.atleast_1d(beta).astype(float)
    out = [np.empty(len(alpha)) for _ in range(4)]
    for i in range(0, len(alpha), chunk):
        sl = slice(i, i + chunk)
        W = s * a * np.sin(alpha[sl, None] * x + beta[sl, None])
        for o, v in zip(out, inner_newton(W, x, y)):
            o[sl] = v
    return tuple(out)


def loss_grad_hess(p, s, x, y, a=A):
    """Full ramp-limit loss in p = (kappa, b, alpha, beta), with its gradient and Hessian."""
    k, b, al, be = p
    u = al * x + be
    S, C = np.sin(u), np.cos(u)
    z = k * x + b + s * a * S
    L = float((_softplus(z) - y * z).mean())
    r = _sig(z) - y; q = _sig(z) * (1 - _sig(z))
    J = np.stack([x, np.ones_like(x), s * a * C * x, s * a * C])          # dz/dp
    g = (r * J).mean(1)
    H = (q * J) @ J.T / len(x)
    t = -s * a * S
    H[2, 2] += (r * t * x * x).mean(); H[2, 3] += (r * t * x).mean(); H[3, 2] = H[2, 3]; H[3, 3] += (r * t).mean()
    return L, g, H


def newton_polish(p0, s, x, y, a=A, iters=100):
    """Damped (Levenberg) Newton on the 4-D ramp-limit loss.  Returns (p, L, gnorm)."""
    p = np.asarray(p0, float).copy()
    L, g, H = loss_grad_hess(p, s, x, y, a)
    mu = 0.0
    for _ in range(iters):
        if np.max(np.abs(g)) < 1e-14:
            break
        lam_min = float(np.linalg.eigvalsh(H)[0])
        mu = max(0.0, -lam_min + 1e-10) if lam_min <= 0 else 0.0
        while True:
            d = np.linalg.solve(H + mu * np.eye(4), g)
            pn = p - d
            Ln, gn, Hn = loss_grad_hess(pn, s, x, y, a)
            if Ln <= L + 1e-15:
                break
            mu = max(10 * mu, 1e-8)
            if mu > 1e8:
                return p, L, float(np.max(np.abs(g)))
        p, L, g, H = pn, Ln, gn, Hn
    return p, L, float(np.max(np.abs(g)))


def canonical(p, s):
    """(kappa, b, alpha, beta) -> alpha >= 0, beta in [0, 2 pi); c_bar = kappa / s."""
    k, b, al, be = p
    if al < 0:                                   # sin(-|al| x + be) = sin(|al| x + pi - be)
        al, be = -al, math.pi - be
    return {"kappa": float(k), "b": float(b), "alpha": float(al), "beta": float(be % TWO_PI), "c_bar": float(k / s)}


def start_set(name, seed=2026):
    if name == "A":
        al, be = np.meshgrid(np.linspace(10 / 150, 10, 150), np.linspace(0, TWO_PI, 72, endpoint=False), indexing="ij")
        return al.ravel(), be.ravel()
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 12, 3000), rng.uniform(0, TWO_PI, 3000)


def ramp_min_from(alpha, beta, s, x, y, a=A, keep=8):
    """Profile every start, polish the `keep` lowest distinct ones by 4-D Newton, return the best."""
    L, k, b, _ = ramp_profile(alpha, beta, s, x, y, a)
    order = np.argsort(L)
    picked = []
    for i in order:
        if all(abs(alpha[i] - alpha[j]) > 0.2 or abs(((beta[i] - beta[j] + math.pi) % TWO_PI) - math.pi) > 0.3
               for j in picked):
            picked.append(i)
        if len(picked) == keep:
            break
    best = None
    for i in picked:
        p, Lp, gn = newton_polish([k[i], b[i], alpha[i], beta[i]], s, x, y, a)
        if best is None or Lp < best[1]:
            best = (p, Lp, gn)
    p, Lp, gn = best
    return {"loss": float(Lp), "gnorm": gn, **canonical(p, s)}


def agree(rA, rB, tol=AGREE_TOL):
    """Two independent restart sets agree: loss within tol and the same minimiser (alpha, c_bar within 1e-6, beta
    within 1e-6 mod 2 pi)."""
    dbeta = abs(((rA["beta"] - rB["beta"] + math.pi) % TWO_PI) - math.pi)
    return bool(abs(rA["loss"] - rB["loss"]) <= tol and abs(rA["alpha"] - rB["alpha"]) <= 1e-6
                and abs(rA["c_bar"] - rB["c_bar"]) <= 1e-6 and dbeta <= 1e-6)


def mirror_margin(r):
    """alpha - |c_bar|: > 0 inside the mirror range (compactly realisable, Lemma K8), < 0 outside."""
    return r["alpha"] - abs(r["c_bar"])


def ramp_min(s, x, y, a=A):
    """The validated ramp-limit minimum at s: both start sets, and their agreement."""
    rA = ramp_min_from(*start_set("A"), s, x, y, a)
    rB = ramp_min_from(*start_set("B"), s, x, y, a)
    best = rA if rA["loss"] <= rB["loss"] else rB
    return {"s": s, **best, "loss_A": rA["loss"], "loss_B": rB["loss"], "agree": agree(rA, rB),
            "mirror_margin": mirror_margin(best)}


def bisect_sign(h, lo, hi, steps=BISECT_STEPS):
    """Bisection in log s for the sign change of h between lo (h < 0) and hi (h > 0).  Returns (lo, hi, evals)."""
    hl, hh = h(lo), h(hi)
    if not (hl[0] < 0 < hh[0]):
        raise ValueError("no sign change on the bracket")
    ev = [(lo, hl[1]), (hi, hh[1])]
    for _ in range(steps):
        m = math.exp((math.log(lo) + math.log(hi)) / 2)
        hm = h(m)
        ev.append((m, hm[1]))
        if hm[0] > 0:
            hi = m
        elif hm[0] < 0:
            lo = m
        else:
            return m, m, ev
    return lo, hi, ev


def classify(quantities, s_ramp_hi):
    """Each registered quantity (name, scale) against the crossover: 'above' iff scale > s_ramp_hi (the bracket's upper
    end, so a quantity inside the bracket is not called above), with the relative margin scale/s_ramp_hi - 1."""
    rows = []
    for q in quantities:
        s = q["s"]
        rows.append({**q, "above": bool(s > s_ramp_hi), "margin_rel": float(s / s_ramp_hi - 1)})
    return rows


# ----------------------------------------------------------------------------------------- registered inputs
def committed_retained():
    import pandas as pd
    P = RESULTS / "asym_parts"
    rows = []
    for f in ("scan.csv", "bisect.csv"):
        d = pd.read_csv(P / f)
        rows += [{"s": float(r.s), "retained_loss": float(r.retained_loss), "status": r.status, "source": f}
                 for r in d.itertuples()]
    v = pd.read_csv(P / "validate.csv")
    rows += [{"s": float(r.s), "retained_loss": float(r.retained_loss), "status": r.status, "source": "validate.csv"}
             for r in v.itertuples()]
    return rows


def init_scales(seeds, k):
    """Matched initial ||v||_1 = k(|v1| + |v2|) of width2_train.init_params(seed) (no training)."""
    from .width2_train import init_params
    out = []
    for sd in seeds:
        q = init_params(sd).numpy()
        out.append(float(k * (abs(q[2]) + abs(q[5]))))
    return out


def training_scales(s_ramp_hi):
    """Observed crossing scales and matched initial scales of T2-3, T2-3b, T2-3c, T2-3d (committed CSVs)."""
    import pandas as pd
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    arms = {"T2-3": RESULTS / "asym_parts" / "train.csv", "T2-3b": RESULTS / "asym_t23b" / "train.csv",
            "T2-3c": RESULTS / "asym_t23c" / "train.csv", "T2-3d": RESULTS / "asym_t23d" / "train.csv"}
    out = {}
    for arm, f in arms.items():
        d = pd.read_csv(f)
        sc = d.loc[d.crossed == True, "s_cross"].dropna().to_numpy()          # noqa: E712
        si = init_scales([int(v) for v in d.seed], k)
        out[arm] = {"runs": int(len(d)), "crossed": int(len(sc)), "s_cross_min": float(sc.min()),
                    "s_cross_median": float(np.median(sc)),
                    "n_cross_below_s_ramp": int((sc <= s_ramp_hi).sum()),
                    "s_init_min": float(min(si)), "s_init_median": float(np.median(si)), "s_init_max": float(max(si)),
                    "n_init_below_s_ramp": int(sum(v <= s_ramp_hi for v in si))}
    return out


def registered_quantities():
    """Every registered width-2 scale on the asymmetric windows (a = 1.30, Delta = 0.4), read from committed files."""
    sc = json.loads((RESULTS / "asym_scores.json").read_text())
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    w2 = json.loads((RESULTS / "width2_asym" / "landscape.json").read_text())
    p3 = json.loads((RESULTS / "phase3" / "landscape.json").read_text())
    q = [{"name": "T2-1 bracket s_lo", "s": sc["s_lo"], "src": "asym_scores.json"},
         {"name": "T2-1 bracket s_hi", "s": sc["s_hi"], "src": "asym_scores.json"},
         {"name": "T2-1 bisection step s = 0.4870", "s": 0.4869675251658631, "src": "asym_parts/bisect.csv"},
         {"name": "T2-1/W2-A s_pop2 = sqrt(s_lo s_hi) (T2-3d, t2g comparator 0.44508)", "s": w2["s_pop2"],
          "src": "width2_asym/landscape.json"},
         {"name": "T2-3/3b/3c s_glob = (s_lo + s_hi)/2 (k and phi2 matching)", "s": (sc["s_lo"] + sc["s_hi"]) / 2,
          "src": "asym_frozen.json (k = %.5f)" % k},
         {"name": "T2-2 pilot comparator 0.5623", "s": 0.5623413251903491, "src": "asym_register.PILOT_LO"},
         {"name": "W2-A hold scale s0 (also Phase 1C W2-A arm, Phase 3)", "s": w2["s0"],
          "src": "width2_asym/landscape.json"}]
    for c, lab in (("D", "D"), ("Dp", "D'"), ("T", "T"), ("Tp", "T'")):
        q.append({"name": f"W2-A population switch {lab}", "s": w2["copies"][c]["s_pop_branch"],
                  "src": "width2_asym/landscape.json"})
    for c in ("Q", "S", "T4"):
        q.append({"name": f"Phase 3 (width 4) population switch {c}", "s": p3["copies"][c]["s_pop_branch"],
                  "src": "phase3/landscape.json"})
    return q


# ============================================================================================================ main
def main():
    os.nice(15)
    from .asym_pilot import population
    x, y = population(DELTA)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {"label": "POST HOC, EXPLORATORY: validated float search, not a certificate; no registered verdict changes",
           "a": A, "delta": DELTA, "objective": "asym_pilot.population(0.4), 800 points (Track 2's registered P_an)"}
    # ---- the committed registered retained losses against the ramp-limit minimum
    comm = []
    for r in committed_retained():
        rm = ramp_min(r["s"], x, y)
        comm.append({**r, "ramp_loss": rm["loss"], "gap": r["retained_loss"] - rm["loss"], "agree": rm["agree"],
                     "alpha": rm["alpha"], "c_bar": rm["c_bar"], "mirror_margin": rm["mirror_margin"],
                     "gnorm": rm["gnorm"], "loss_A": rm["loss_A"], "loss_B": rm["loss_B"]})
        print(f"s={r['s']:.6g} retained={r['retained_loss']:.15f} ramp={rm['loss']:.15f} "
              f"gap={comm[-1]['gap']:.3e} margin={rm['mirror_margin']:.4f} agree={rm['agree']}", flush=True)
    out["committed"] = comm
    # ---- bisection on the mirror-range sign between the committed scan scales 0.1334 (outside) and 0.1778 (inside)
    cache = {}

    def h(s):
        rm = ramp_min(s, x, y)
        cache[s] = rm
        return rm["mirror_margin"], rm
    lo0, hi0 = 0.1333521432163324, 0.1778279410038923
    lo, hi, ev = bisect_sign(h, lo0, hi0)
    out["bisection"] = {"lo0": lo0, "hi0": hi0, "steps": BISECT_STEPS, "s_ramp_lo": lo, "s_ramp_hi": hi,
                        "rel_width": hi / lo - 1, "all_agree": bool(all(e[1]["agree"] for e in ev)),
                        "evals": [{"s": s_, "loss": e["loss"], "alpha": e["alpha"], "c_bar": e["c_bar"],
                                   "mirror_margin": e["mirror_margin"], "agree": e["agree"]} for s_, e in ev]}
    print(f"s_ramp in [{lo:.7f}, {hi:.7f}]", flush=True)
    # ---- the registered retained loss recomputed by the registered code path near s_ramp and at s0
    from .asym_register import eval_scale
    s0 = json.loads((RESULTS / "width2_asym" / "landscape.json").read_text())["s0"]
    reg = []
    for i, s in enumerate((hi * 1.01, hi * 1.05, lo * 0.99, lo * 0.95, s0)):
        r = eval_scale(s, seed=REG_SEED0 + i)
        rm = ramp_min(s, x, y)
        reg.append({"s": s, "seed": REG_SEED0 + i, "retained_loss": float(r["retained_loss"]), "status": r["status"],
                    "ramp_loss": rm["loss"], "gap": float(r["retained_loss"]) - rm["loss"], "agree": rm["agree"],
                    "alpha": rm["alpha"], "c_bar": rm["c_bar"], "mirror_margin": rm["mirror_margin"],
                    "q": r["q"]})
        print(f"[registered path] s={s:.7f} retained={r['retained_loss']:.15f} ramp={rm['loss']:.15f} "
              f"gap={reg[-1]['gap']:.3e} margin={rm['mirror_margin']:.5f}", flush=True)
    out["registered_path"] = reg
    # ---- the registered quantities against s_ramp
    out["quantities"] = classify(registered_quantities(), hi)
    out["training"] = training_scales(hi)
    w2 = json.loads((RESULTS / "width2_asym" / "landscape.json").read_text())
    out["w2a_branch_losses_s0"] = {c: w2["copies"][c]["loss"] for c in ("D", "Dp", "T", "Tp")}
    s0row = reg[-1]
    above = [c for c in comm if c["s"] > hi]
    below = [c for c in comm if c["s"] < lo]
    out["ledger"] = {
        "RC s_ramp lo": lo, "RC s_ramp hi": hi, "RC bisection rel width": hi / lo - 1,
        "RC bisection all agree": float(out["bisection"]["all_agree"]),
        "RC committed all agree": float(all(c["agree"] for c in comm)),
        "RC max gap above": max(c["gap"] for c in above),                 # > 0 would mean the ramp limit beats
        "RC min gap above": min(c["gap"] for c in above),                 # < 0: the retained loss beats the ramp
        "RC gap s=10": [c["gap"] for c in above if abs(c["s"] - 10.0) < 1e-12][0],
        "RC max abs gap 0.178 to 7.5": max(abs(c["gap"]) for c in above if c["s"] < 9),
        "RC min mirror margin above": min(c["mirror_margin"] for c in above),
        "RC gap s=0.1": [c["gap"] for c in below if abs(c["s"] - 0.1) < 1e-12][0],
        "RC gap s=0.1334": [c["gap"] for c in below if abs(c["s"] - lo0) < 1e-12][0],
        "RC regpath gap hi*1.01": reg[0]["gap"], "RC regpath gap hi*1.05": reg[1]["gap"],
        "RC regpath gap lo*0.99": reg[2]["gap"], "RC regpath gap lo*0.95": reg[3]["gap"],
        "RC regpath gap s0": s0row["gap"], "RC mirror margin s0": s0row["mirror_margin"],
        "RC ramp loss s0": s0row["ramp_loss"],
        "RC quantities below": float(sum(not r["above"] for r in out["quantities"])),
        "RC min quantity margin": min(r["margin_rel"] for r in out["quantities"]),
        "RC s0 over s_ramp": s0 / hi}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["ledger"], indent=1))


if __name__ == "__main__":
    main()
