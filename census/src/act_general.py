"""Track 3A: the width-1 scale-gating machinery generalised to smooth non-periodic activations with a single dip
(GELU, exact erf form; SiLU = t·σ(t); Mish = t·tanh(softplus(t))).

Nothing here modifies an existing module.  The activation object below is Act-like (u, du, d2u, d2u_bound, B2,
periodic, tag), so `width2_geometry.gaps` / `extrema` work unchanged with θ = (w₁, b₁, 0, 0), v = (±1, 0).

d2u_bound is VALIDATED, not certified: sup |u″| over [tlo, thi] is bounded by the maximum of |u″| on a 9-point
sub-grid of the interval plus (h/2)·(max of |u‴| on the sub-grid + (h/2)·B4) (h the sub-grid spacing, B4 a validated
global bound on |u⁗|, grid maximum over [−60, 60] at step 1e−4 plus a 5% margin), capped by the validated global
bound B2.  (A first version used a global |u‴| margin; it made flat tails refine without end — the extrema cell cap
stopped it at the first small-scale evaluation — and was replaced before any result was recorded.)

Width 1 (task: I = [−0.8, 0.8] class 0, O = ±[1.2, 2.0] class 1): z = w₂·u(w₁x + b₁) + b₂, s = |w₂|, σ = sign w₂.
The width-1 conditional search is the width-2 search with ṽ fixed at (σ, 0) (asym_pilot._width1_batch), written
directly in (w₁, b₁) with b₂ profiled; BFGS is width2_conditional.bfgs_batch.  Placement of a configuration:
G₊(σ·u(w₁x + b₁)) > 0 (the lower end of the exact-extrema enclosure).

    python -m src.act_general validate_fa        # step 0: reproduce the certified f_a (a = 1.30) bracket
    python -m src.act_general ghat               # step 1: Ĝ (validated search) and the solve check
    python -m src.act_general criterion          # step 2 (after the registration commit)
    python -m src.act_general bracket NAME       # step 3
    python -m src.act_general kappa NAME         # step 4 calibration + κ (before the registered runs)
    python -m src.act_general train NAME         # step 4 registered runs
    python -m src.act_general score              # step 4 scoring
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "act_general"
NAMES = ("gelu", "silu", "mish")
BOX = 10.0
TIE = 1e-9


# ------------------------------------------------------------------------------------------ activations
def _torch_fn(name):
    import torch
    from torch.nn import functional as F
    if name == "gelu":
        return lambda t: F.gelu(t)                      # exact (erf) form
    if name == "silu":
        return F.silu
    if name == "mish":
        return F.mish
    raise ValueError(name)


def _ndtr(t):
    import torch
    return np.exp(torch.special.log_ndtr(torch.as_tensor(np.asarray(t, float))).numpy())   # tail-accurate


def _sig(t):
    """Logistic σ with full relative precision in both tails (0.5(1 + tanh) loses it for t << 0)."""
    t = np.asarray(t, float)
    e = np.exp(-np.abs(t))
    return np.where(t >= 0, 1 / (1 + e), e / (1 + e))


class GAct:
    """u, u′, u″ (closed forms, numpy, vectorised) and a validated d2u_bound, for gelu / silu / mish."""

    periodic = False
    a = None

    def __init__(self, name):
        if name not in NAMES:
            raise ValueError(name)
        self.name = name
        self.B3 = self._global_d3_bound()
        self.B4 = self._global_d4_bound()
        grid = np.arange(-60.0, 60.0 + 1e-9, 1e-4)
        self.B2 = float(np.abs(self.d2u(grid)).max() + self.B3 * 0.5e-4) * 1.0001

    @property
    def tag(self):
        return self.name

    def u(self, t):
        t = np.asarray(t, float)
        if self.name == "gelu":
            return t * _ndtr(t)
        if self.name == "silu":
            return t * _sig(t)
        return t * np.tanh(np.logaddexp(0, t))

    def du(self, t):
        t = np.asarray(t, float)
        if self.name == "gelu":
            return _ndtr(t) + t * np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
        if self.name == "silu":
            s = _sig(t)
            return s * (1 + t * (1 - s))
        g = np.tanh(np.logaddexp(0, t)); s = _sig(t)
        return g + t * (1 - g * g) * s

    def d2u(self, t):
        t = np.asarray(t, float)
        if self.name == "gelu":
            return np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi) * (2 - t * t)
        if self.name == "silu":
            s = _sig(t)
            return s * (1 - s) * (2 + t * (1 - 2 * s))
        g = np.tanh(np.logaddexp(0, t)); s = _sig(t)
        return (1 - g * g) * s * (2 + t * (1 - s - 2 * g * s))

    def d3u(self, t):
        t = np.asarray(t, float)
        if self.name == "gelu":
            return np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi) * (t ** 3 - 4 * t)
        if self.name == "silu":
            s = _sig(t); q = s * (1 - s); e = 1 - 2 * s
            return q * (e * (3 + t * e) - 2 * t * q)
        g = np.tanh(np.logaddexp(0, t)); s = _sig(t); q = s * (1 - s)
        A = (1 - g * g) * s; E = 1 - s - 2 * g * s; C = 2 + t * E
        Ep = -q - 2 * A * s - 2 * g * q
        return A * (E * C + E + t * Ep)

    def torch_u(self, t):
        return _torch_fn(self.name)(t)

    def _global_d3_bound(self):
        """max |u‴| on [−60, 60] (step 1e−4) by autograd in double, plus a 5% margin (validated, not certified)."""
        import torch
        t = torch.arange(-60.0, 60.0 + 1e-9, 1e-4, dtype=torch.float64).requires_grad_(True)
        g = _torch_fn(self.name)(t)
        for _ in range(3):
            g = torch.autograd.grad(g.sum(), t, create_graph=True)[0]
        return float(g.detach().abs().max()) * 1.05

    def _global_d4_bound(self):
        """max |u⁗| on [−60, 60] (step 1e−4) by autograd in double, plus a 5% margin (validated, not certified)."""
        import torch
        t = torch.arange(-60.0, 60.0 + 1e-9, 1e-4, dtype=torch.float64).requires_grad_(True)
        g = _torch_fn(self.name)(t)
        for _ in range(4):
            g = torch.autograd.grad(g.sum(), t, create_graph=True)[0]
        return float(g.detach().abs().max()) * 1.05

    def d2u_bound(self, tlo, thi, k=9):
        """sup |u″| on [tlo, thi] <= max_grid |u″| + (h/2)·(max_grid |u‴| + (h/2)·B4), h the sub-grid spacing (every
        point is within h/2 of a grid point; |u‴| on the interval is bounded the same way from B4); capped by B2."""
        tlo, thi = np.minimum(tlo, thi), np.maximum(tlo, thi)
        tlo, thi = np.broadcast_arrays(np.asarray(tlo, float), np.asarray(thi, float))
        w = np.linspace(0.0, 1.0, k)
        G = tlo[..., None] + (thi - tlo)[..., None] * w
        h = (thi - tlo) / (k - 1)
        m2 = np.abs(self.d2u(G)).max(axis=-1); m3 = np.abs(self.d3u(G)).max(axis=-1)
        return np.minimum(m2 + 0.5 * h * (m3 + 0.5 * h * self.B4), self.B2)


def get_act(name):
    if name == "fa1.30":
        from .width2_geometry import Act
        return Act("fa", 1.30)
    return GAct(name)


def torch_u(act):
    """A torch version of u for any act (f_a or GAct)."""
    import torch
    if getattr(act, "name", None) == "fa":
        return lambda t: t + act.a * torch.sin(t)
    return act.torch_u


# ------------------------------------------------------------------------------------------ geometry (width 1)
MAX_CELLS = 4_000_000


def gplus(w1, b1, sigma, act, tol=1e-9, max_cells=MAX_CELLS):
    """Exact-extrema enclosure (lo, hi) of G₊(σ·u(w₁x + b₁)) on the continuous windows (width2_geometry.extrema; the
    cell cap raised from 400,000 to MAX_CELLS, a resource limit).  If the cap is still reached the enclosure is
    (−inf, +inf), i.e. undecided."""
    from .width2_geometry import INNER, OUTER, extrema
    th = np.array([w1, b1, 0.0, 0.0]); v = np.array([float(sigma), 0.0])
    try:
        i = extrema(th, v, act, *INNER, tol=tol, max_cells=max_cells)
        oL = extrema(th, v, act, *OUTER[0], tol=tol, max_cells=max_cells)
        oR = extrema(th, v, act, *OUTER[1], tol=tol, max_cells=max_cells)
    except RuntimeError:
        return (-math.inf, math.inf)
    o_min_lo, o_min_hi = min(oL[0], oR[0]), min(oL[1], oR[1])
    return (o_min_lo - i[3], o_min_hi - i[2])


def mp_u(name, t):
    import mpmath as mp
    if name == "gelu":
        return t * mp.ncdf(t)
    if name == "silu":
        return t / (1 + mp.exp(-t))
    if name == "mish":
        return t * mp.tanh(mp.log1p(mp.exp(t)))
    raise ValueError(name)


def mp_dip(name, dps=60):
    """The unique critical point of u (its minimum), by mpmath root finding on u′."""
    import mpmath as mp
    with mp.workdps(dps):
        t0 = {"gelu": -0.7518, "silu": -1.2785, "mish": -1.1924}[name]
        return mp.findroot(lambda t: mp.diff(lambda q: mp_u(name, q), t), t0)


def mp_gap(w1, b1, sigma, name, dps=60):
    """POST HOC diagnostic: G₊(σ·u(w₁x + b₁)) in arbitrary precision (mpmath), using that u is unimodal (decreasing
    then increasing, one critical point; checked in tests): on an interval, max u is at an end, min u is at the dip if
    the dip is inside, else at an end.  Returns an mpf (no underflow)."""
    import mpmath as mp
    from .width2_geometry import INNER, OUTER
    with mp.workdps(dps):
        tm = mp_dip(name, dps)
        w1, b1 = mp.mpf(w1), mp.mpf(b1)

        def ext(lo, hi):
            a, b = w1 * lo + b1, w1 * hi + b1
            a, b = min(a, b), max(a, b)
            ua, ub = mp_u(name, a), mp_u(name, b)
            mn = mp_u(name, tm) if a <= tm <= b else min(ua, ub)
            return mn, max(ua, ub)
        iI = ext(*INNER); oL = ext(*OUTER[0]); oR = ext(*OUTER[1])
        if sigma > 0:
            return min(oL[0], oR[0]) - iI[1]
        return iI[0] - max(oL[1], oR[1])


def mp_log10_gap(w1, b1, sigma, name):
    """(sign, log10|G|) of the arbitrary-precision gap (post hoc diagnostic)."""
    import mpmath as mp
    g = mp_gap(w1, b1, sigma, name)
    if g == 0:
        return 0, -math.inf
    return (1 if g > 0 else -1), float(mp.log10(abs(g)))


def dense_gplus(w1, b1, sigma, act):
    from .width2_geometry import _XI, _XO
    fi, fo = sigma * act.u(w1 * _XI + b1), sigma * act.u(w1 * _XO + b1)
    return fo.min() - fi.max()


def ghat_search(act, starts=400, seed=0, box=BOX):
    """sup over (w₁, b₁, σ) of G₊ by multistart Nelder–Mead on the dense gap; the best point's exact enclosure."""
    from .width2_geometry import nelder_mead
    rng = np.random.default_rng(seed)
    blo, bhi = (0.0, 2 * math.pi) if act.periodic else (-box, box)
    best = (-math.inf, None)
    for _ in range(starts):
        sg = float(rng.choice([-1.0, 1.0]))
        z0 = np.array([rng.uniform(0, box), rng.uniform(blo, bhi)])
        z, fv = nelder_mead(lambda z: -dense_gplus(z[0], z[1], sg, act), z0, step=0.5, iters=800)
        if -fv > best[0]:
            best = (-fv, (float(z[0]), float(z[1]), sg))
    w1, b1, sg = best[1]
    lo, hi = gplus(w1, b1, sg, act)
    return {"G_lo": float(lo), "G_hi": float(hi), "w1": w1, "b1": b1, "sigma": sg, "dense": best[0]}


def ghat_de(act, seed=1, box=BOX, pop=40, gens=300):
    """Independent search: differential evolution over (w₁, b₁, ψ), σ = sign(ψ)."""
    from .width2_geometry import differential_evolution
    blo, bhi = (0.0, 2 * math.pi) if act.periodic else (-box, box)
    f = lambda z: -dense_gplus(z[0], z[1], 1.0 if z[2] >= 0 else -1.0, act)
    z, fv = differential_evolution(f, [-box, blo, -1], [box, bhi, 1], pop, gens, seed=seed)
    sg = 1.0 if z[2] >= 0 else -1.0
    lo, hi = gplus(z[0], z[1], sg, act)
    return {"G_lo": float(lo), "G_hi": float(hi), "w1": float(z[0]), "b1": float(z[1]), "sigma": sg}


def solve_check(w1, b1, sigma, s, act):
    """With the output bias at the midpoint of the feasible interval, is z = σs·u(w₁x+b₁) + b sign-correct on the
    windows (exact extrema)?  Returns (verdict, b)."""
    from .width2_geometry import sign_correct, window_extrema
    th = np.array([w1, b1, 0.0, 0.0]); v = np.array([sigma * s, 0.0])
    e = window_extrema(th, v, act)
    b = -0.5 * (e["O"][0] + e["I"][3])
    return sign_correct(th, v, b, act), float(b)


# ------------------------------------------------------------------------------------------ width-1 conditional search
def population():
    from .width2_conditional import population as p
    return p()


def loss_grad_w1(P, sg, s, x, y, act):
    """L*(w₁, b₁; σ, s) (b₂ profiled exactly) and its gradient, batched over rows of P (m, 2)."""
    from .width2_conditional import _sig as sgm, _softplus, profile_b_batch
    W, B = P[:, :1], P[:, 1:2]
    T = W * x[None, :] + B
    U, D = act.u(T), act.du(T)
    c = s * sg[:, None]
    Z0 = c * U
    b = profile_b_batch(Z0, y)
    Z = Z0 + b[:, None]
    L = (_softplus(Z) - y[None, :] * Z).mean(axis=1)
    R = sgm(Z) - y[None, :]
    G = np.stack([(R * c * D * x[None, :]).mean(axis=1), (R * c * D).mean(axis=1)], axis=1)
    return L, G


def loss_w1(w1, b1, sigma, s, x, y, act):
    return float(loss_grad_w1(np.array([[w1, b1]]), np.array([float(sigma)]), s, x, y, act)[0][0])


def starts_w1(n, seed, act, box=BOX):
    rng = np.random.default_rng(seed)
    blo, bhi = (0.0, 2 * math.pi) if act.periodic else (-box, box)
    P0 = np.column_stack([rng.uniform(0, box, n), rng.uniform(blo, bhi, n)])
    sg = rng.choice([-1.0, 1.0], n)
    return P0, sg


def search_w1(s, x, y, act, restarts=200, seed=0, gtol=1e-8, maxit=3000, batch=100, box=BOX):
    """Every restart retained as a candidate: (w₁, b₁, σ, loss, gnorm); plus the constant predictor."""
    from .width2_conditional import bfgs_batch, constant_predictor_loss
    P0, SG = starts_w1(restarts, seed, act, box)
    cands = []
    for i in range(0, restarts, batch):
        p0, sg = P0[i:i + batch], SG[i:i + batch]
        P, L, G, it = bfgs_batch(lambda Q, rows, sg=sg: loss_grad_w1(Q, sg[rows], s, x, y, act), p0,
                                 gtol=gtol, maxit=maxit)
        for k in range(len(p0)):
            gn = float(np.abs(G[k]).max())
            flags = []
            if not np.all(np.isfinite(P[k])) or not np.isfinite(L[k]):
                flags.append("nonfinite")
            if gn > 100 * gtol:
                flags.append("not_converged")
            cands.append({"k": i + k, "w1": float(P[k, 0]), "b1": float(P[k, 1]), "sigma": float(sg[k]),
                          "loss": float(L[k]), "gnorm": gn, "iters": int(it[k]), "flags": flags})
    cands.append({"k": -1, "w1": 0.0, "b1": 0.0, "sigma": 1.0, "loss": constant_predictor_loss(y), "gnorm": 0.0,
                  "iters": 0, "flags": ["constant_predictor"]})
    return cands


def newton_w1(w1, b1, sigma, s, x, y, act, iters=100):
    """Damped Newton on the joint loss in z = (w₁, b₁, b₂) at w₂ = σs (torch, double).  Returns (z, max|grad|)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    u = torch_u(act)
    w2 = float(sigma) * s
    f = lambda z: torch.nn.functional.binary_cross_entropy_with_logits(w2 * u(z[0] * X + z[1]) + z[2], Y)
    from .width2_conditional import profile_b
    b0 = profile_b(w2 * act.u(w1 * x + b1), y)
    z = torch.tensor([w1, b1, b0], dtype=torch.float64)
    mu, f0 = 1e-8, float(f(z))
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(f(zz), zz)[0]
        if float(g.abs().max()) < 1e-14:
            break
        H = torch.autograd.functional.hessian(f, z)
        ok = False
        for _ in range(40):
            try:
                d = torch.linalg.solve(H + mu * torch.eye(3, dtype=torch.float64), -g)
            except RuntimeError:
                mu *= 10
                continue
            f1 = float(f(z + d))
            if f1 <= f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 10, 1e-14)
                break
            mu *= 10
        if not ok:
            break
    zz = z.clone().requires_grad_(True)
    g = torch.autograd.grad(f(zz), zz)[0]
    return z.numpy(), float(g.abs().max()), f0


def eligible(c):
    return not ({"nonfinite", "not_converged"} & set(c["flags"]))


def classify(g):
    """'placed' iff the exact enclosure's lower end > 0; 'unplaced' iff its upper end <= 0; else 'undecided'."""
    lo, hi = g
    return "placed" if lo > 0 else ("unplaced" if hi <= 0 else "undecided")


def finish(cands, s, x, y, act, polish=5):
    """Polish the `polish` lowest distinct eligible candidates by Newton (loss recomputed exactly), attach exact G and
    status to every polished candidate; the retained one is the lowest loss (constant predictor included, status
    'unplaced': a constant φ has G = 0)."""
    ok = sorted([c for c in cands if eligible(c) and c["k"] >= 0], key=lambda c: c["loss"])
    picked, seen = [], []
    for c in ok:
        if all(abs(c["loss"] - l0) > 1e-10 for l0 in seen):
            picked.append(c); seen.append(c["loss"])
        if len(picked) == polish:
            break
    pol = []
    for c in picked:
        z, gmax, _ = newton_w1(c["w1"], c["b1"], c["sigma"], s, x, y, act)
        L = loss_w1(z[0], z[1], c["sigma"], s, x, y, act)
        if not (L <= c["loss"] + 1e-12):                   # Newton must not make it worse; keep the BFGS point
            z = np.array([c["w1"], c["b1"], np.nan]); L = c["loss"]; gmax = c["gnorm"]
        g = gplus(z[0], z[1], c["sigma"], act)
        pol.append({**c, "w1": float(z[0]), "b1": float(z[1]), "loss": float(L), "gnorm_polished": gmax,
                    "G_lo": float(g[0]), "G_hi": float(g[1]), "status": classify(g)})
    const = next(c for c in cands if c["k"] == -1)
    pol.append({**const, "G_lo": 0.0, "G_hi": 0.0, "status": "unplaced"})
    ret = min(pol, key=lambda c: c["loss"])
    return ret, pol


def status_at(s, x, y, act, restarts=200, seed=0, **kw):
    cands = search_w1(s, x, y, act, restarts=restarts, seed=seed, **kw)
    ret, pol = finish(cands, s, x, y, act)
    return ret, pol, cands


# ------------------------------------------------------------------------------------------ validity checks (pure)
def ladder_ok(ret_a, ret_b, tol=TIE):
    """Restart ladder: retained loss unchanged within tol and the same status."""
    return bool(abs(ret_a["loss"] - ret_b["loss"]) <= tol and ret_a["status"] == ret_b["status"])


def independent_ok(retained_loss, other_loss, tol=TIE):
    """Independent search must not find a loss lower than the retained one by more than tol."""
    return bool(other_loss >= retained_loss - tol)


def audit_ok(ret, others, tol=TIE):
    """No candidate of another status (placed / unplaced / undecided) within tol of the retained loss, and the retained
    status decided.  `others`: dicts with 'loss' and 'status' (exact G of every eligible candidate near the top)."""
    bad = [c for c in others if c is not ret and c["status"] != ret["status"] and c["loss"] <= ret["loss"] + tol]
    return bool(not bad and ret["status"] != "undecided"), len(bad)


def near_top_status(cands, ret, act, window=1e-7):
    """Exact status of every eligible candidate whose loss is within `window` of the retained loss (for the audit)."""
    out = []
    for c in cands:
        if c["k"] < 0:
            if c["loss"] <= ret["loss"] + window:
                out.append({"loss": c["loss"], "status": "unplaced", "k": -1})
            continue
        if eligible(c) and c["loss"] <= ret["loss"] + window:
            out.append({"loss": c["loss"], "status": classify(gplus(c["w1"], c["b1"], c["sigma"], act)), "k": c["k"]})
    return out


def criterion_verdict(status_small, ghat_lo):
    """Registered decision rule.  status_small: statuses of the small-scale conditional minimiser (list, one per small
    scale).  'switch predicted' iff all unplaced and Ĝ > 0; 'no switch predicted' iff all placed; else
    'undetermined'."""
    if all(st == "unplaced" for st in status_small) and ghat_lo > 0:
        return "switch predicted"
    if all(st == "placed" for st in status_small):
        return "no switch predicted"
    return "undetermined"


def first_switch(ss, statuses):
    """From a scan (increasing s): the first unplaced→placed change (bracket), number of status changes, and whether
    any scan point is undecided."""
    ss = list(ss); st = list(statuses)
    changes = sum(st[i] != st[i + 1] for i in range(len(st) - 1))
    for i in range(1, len(st)):
        if st[i] == "placed" and st[i - 1] == "unplaced":
            return {"bracket": (ss[i - 1], ss[i]), "changes": changes, "undecided": "undecided" in st}
    return {"bracket": None, "changes": changes, "undecided": "undecided" in st}


# ------------------------------------------------------------------------------------------ independent (CMA-ES)
def cma_w1(s, x, y, act, starts=20, seed=7, sigma0=1.0, gens=300, box=BOX):
    """CMA-ES on q = (w₁, b₁, v) with φ = sign(v)·u(w₁x + b₁): a different optimiser and parametrisation."""
    from .cmaes import cma_es
    from .width2_conditional import _softplus, profile_b
    rng = np.random.default_rng(seed)

    def f(q):
        sg = 1.0 if q[2] >= 0 else -1.0
        z0 = s * sg * act.u(q[0] * x + q[1])
        if not np.all(np.isfinite(z0)):
            return 1e9
        z = z0 + profile_b(z0, y)
        return float((_softplus(z) - y * z).mean())
    best = (math.inf, None)
    for _ in range(starts):
        q0 = np.array([rng.uniform(-box, box), rng.uniform(-box, box), rng.uniform(-1, 1)])
        r = cma_es(f, q0, sigma0, max_generations=gens, seed=int(rng.integers(1 << 31)))
        if r.best_f < best[0]:
            best = (r.best_f, r.best_x)
    return {"loss": float(best[0]), "q": [float(v) for v in best[1]]}


# ------------------------------------------------------------------------------------------ step 0: f_a validation
def validate_fa(restarts=200):
    """Scan the certified bracket's neighbourhood with the generalised search and bisect to 1e−4 relative."""
    OUT.mkdir(exist_ok=True)
    act = get_act("fa1.30")
    x, y = population()
    t0 = time.time()
    rows = []

    def st(s, seed):
        r, pol, _ = status_at(s, x, y, act, restarts=restarts, seed=seed)
        rows.append({"s": s, "status": r["status"], "loss": r["loss"], "G_lo": r["G_lo"], "G_hi": r["G_hi"],
                     "w1": r["w1"], "b1": r["b1"], "sigma": r["sigma"]})
        print(json.dumps(rows[-1]), flush=True)
        return r["status"]
    grid = [4.0, 4.5, 4.9, 5.0, 5.5, 6.0]
    stats = [st(s, 100 + i) for i, s in enumerate(grid)]
    fs = first_switch(grid, stats)
    lo, hi = fs["bracket"]
    j = 0
    while hi / lo - 1 > 1e-4:
        mid = math.sqrt(lo * hi); j += 1
        sm = st(mid, 200 + j)
        if sm == "placed":
            hi = mid
        elif sm == "unplaced":
            lo = mid
        else:
            break
    out = {"act": "f_a, a = 1.30", "scan": list(zip(grid, stats)), "scan_changes": fs["changes"],
           "bracket_lo": lo, "bracket_hi": hi, "certified_lo": 4.95, "certified_hi": 4.9625,
           "reproduced": bool(4.95 <= lo and hi <= 4.9625), "restarts": restarts, "seconds": time.time() - t0}
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "validate_fa_evals.csv", index=False)
    (OUT / "validate_fa.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------ step 1: placement, solve
def step1():
    """Ĝ by the validated search (multistart NM, 400 starts) and an independent DE search; agreement required to 1e−6
    relative; the solve check at the maximiser with the output bias at the feasible midpoint (s = 1).  Also the
    orientation of every placed maximiser and the dip location (information)."""
    OUT.mkdir(exist_ok=True)
    rows = {}
    for n in NAMES:
        act = GAct(n)
        t0 = time.time()
        a = ghat_search(act)
        b = ghat_de(act)
        agree = abs(a["G_lo"] - b["G_lo"]) <= 1e-6 * max(abs(a["G_lo"]), abs(b["G_lo"]))
        sc, bias = solve_check(a["w1"], a["b1"], a["sigma"], 1.0, act)
        # a placed configuration with the wrong output bias is not sign-correct (control)
        sc_bad, _ = (solve_check(a["w1"], a["b1"], a["sigma"], 1.0, act)[0], None)
        from .width2_geometry import sign_correct
        wrong = sign_correct(np.array([a["w1"], a["b1"], 0, 0]), np.array([a["sigma"], 0.0]), bias + 10.0, act)
        tt = np.linspace(-5, 5, 1_000_001); uu = act.u(tt)
        rows[n] = {"Ghat_nm": a, "Ghat_de": b, "searches_agree_1e-6": bool(agree),
                   "solve_check_s1": sc, "solve_bias": bias, "wrong_bias_control_sign_correct": wrong,
                   "dip_t": float(tt[np.argmin(uu)]), "dip_depth": float(uu.min()),
                   "B2": act.B2, "B3": act.B3, "seconds": time.time() - t0}
        print(n, json.dumps(rows[n], default=float), flush=True)
    (OUT / "ghat.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


def bound_check(act, bound_fn, n=2000, seed=0, dense=2001):
    """Validity check of a sup|u″| bound: on n random intervals, the number where the bound is below the maximum of
    |u″| on a dense sampling of the interval (should be 0)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-12, 12, n); w = np.exp(rng.uniform(np.log(1e-6), np.log(20), n))
    lo, hi = c - w / 2, c + w / 2
    B = bound_fn(lo, hi)
    bad = 0
    for i in range(n):
        m = np.abs(act.d2u(np.linspace(lo[i], hi[i], dense))).max()
        bad += int(B[i] < m * (1 - 1e-12))
    return bad


# ------------------------------------------------------------------------------------------ step 2: the criterion
SMALL_S = (0.05, 0.1)


def validate_point(s, x, y, act, seed, ladder=(200, 800), cma_starts=20):
    """Validation of the retained conditional minimiser at scale s: restart ladder (retained unchanged within 1e−9 and
    same status), independent CMA-ES not lower by more than 1e−9, audit (no polished candidate of the other status
    within 1e−9; retained not undecided)."""
    r_small, pol_small, _ = status_at(s, x, y, act, restarts=ladder[0], seed=seed)
    r_big, pol_big, c_big = status_at(s, x, y, act, restarts=ladder[1], seed=seed + 1)
    cm = cma_w1(s, x, y, act, starts=cma_starts, seed=seed + 2)
    lad = ladder_ok(r_small, r_big)
    ind = independent_ok(r_big["loss"], cm["loss"])
    near = near_top_status(c_big, r_big, act) + [c for c in pol_big if c is not r_big]
    aud, n_other = audit_ok(r_big, near)
    n_conv = sum(1 for c in c_big if c["k"] >= 0 and eligible(c))
    mp_sign, mp_l10 = (mp_log10_gap(r_big["w1"], r_big["b1"], r_big["sigma"], act.name)
                       if isinstance(act, GAct) and r_big["k"] >= 0 else (0, -math.inf))
    return {"s": s, "status": r_big["status"], "loss": r_big["loss"], "G_lo": r_big["G_lo"], "G_hi": r_big["G_hi"],
            "w1": r_big["w1"], "b1": r_big["b1"], "sigma": r_big["sigma"],
            "loss_ladder_small": r_small["loss"], "status_ladder_small": r_small["status"], "ladder_ok": lad,
            "cma_loss": cm["loss"], "cma_q": json.dumps(cm["q"]), "independent_ok": ind,
            "audit_ok": aud, "audit_n_other_within_tie": n_other, "audit_n_near_top": len(near), "n_converged": n_conv,
            "validated": bool(lad and ind and aud),
            "posthoc_mp_sign_G": mp_sign, "posthoc_mp_log10_absG": mp_l10}


def criterion():
    """Step 2 (run only after results/act_criterion_registration.md is committed)."""
    import pandas as pd
    gh = json.loads((OUT / "ghat.json").read_text())
    x, y = population()
    rows, verdicts = [], {}
    for n in NAMES:
        act = GAct(n)
        pts = [validate_point(s, x, y, act, seed=41_000 + 10 * i + 100 * NAMES.index(n)) for i, s in enumerate(SMALL_S)]
        for p in pts:
            p["act"] = n
            rows.append(p)
            print(json.dumps(p, default=float), flush=True)
        ok = all(p["validated"] for p in pts)
        v = criterion_verdict([p["status"] for p in pts], gh[n]["Ghat_nm"]["G_lo"]) if ok else "undetermined"
        verdicts[n] = {"verdict": v, "all_validated": ok, "Ghat_lo": gh[n]["Ghat_nm"]["G_lo"],
                       "statuses": {str(p["s"]): p["status"] for p in pts}}
    pd.DataFrame(rows).to_csv(OUT / "criterion_points.csv", index=False)
    (OUT / "criterion_verdicts.json").write_text(json.dumps(verdicts, indent=1, default=float))
    print(json.dumps(verdicts, indent=1, default=float))
    return verdicts
