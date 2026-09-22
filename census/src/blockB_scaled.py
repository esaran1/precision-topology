"""Block B's frozen conditional minimisation, run in limit coordinates.

WHY (convergence diagnostics only, logged in results/blockB_scaled_validity.md):
at a = 1.02 the frozen procedure's random starts (w1 in [-2,2], b1 in [-4,4],
Adam lr 3e-2) are set in raw units, while the fold is ~sqrt(eps) wide and |w2| is
in the hundreds.  48 of 50 starts collapse to the constant predictor, and even a
start at the certified optimum collapses for R >= 0.35.

WHAT CHANGES: only the coordinates.  With c the fold centre (pi for the sin
family, 0 for q-families) and s = eps**(1/q)  (q = 2 for the sin family),
    w1 = s*u,   b1 = c + s*v,   logit = W*h_a(sigma) + b2',
    h_a(sigma) = (f(c + s*sigma) - f(c)) / eps**(1+1/q),   W = |w2| eps**(1+1/q) * sign
so the problem at every eps has the scale of its limit.  Restarts, screen/keep,
inner steps, Adam and its schedule, the log-2 exclusion and the population data
are the frozen values; the random box is the frozen box read in (u, v, b2') with
W in place of |w2|.  Class gap and the dense regional check are evaluated on the
raw parameters.

VALIDITY TEST (must pass before use): at a = 1.30 this reproduces the frozen
procedure's thresholds to within one frozen grid step.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from . import blockB_landscape as bb
from .fold1d import solves

LOG2 = bb.LOG2


def make(fam, a):
    """(f, centre c, scale s, amplitude power p) for 'A' (sin) or 'q2'."""
    eps = a - 1.0
    if fam == "A":
        from .fold1d import activation
        return activation("sin_family", a), math.pi, math.sqrt(eps), 1.5
    from .depth_families import make_family
    fq = make_family(2.0)[0]
    return (lambda v: fq(v, a)), 0.0, math.sqrt(eps), 1.5


def minimise(fam, a, w2, x, y, start=None, seed=0, steps=bb.STEPS):
    f, c, s, p_amp = make(fam, a)
    eps = a - 1.0
    amp = eps ** p_amp
    W = w2 * amp
    fc = float(f(torch.tensor(c, dtype=torch.float64)))
    if start is None:
        g = torch.Generator().manual_seed(seed)
        p = torch.stack([torch.rand(1, generator=g).double()[0] * 4 - 2,
                         torch.rand(1, generator=g).double()[0] * 8 - 4,
                         torch.rand(1, generator=g).double()[0] * 2 * abs(W) - abs(W)])
    else:
        p = torch.tensor(start, dtype=torch.float64)
    p = p.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=3e-2)
    Wt = torch.tensor(W, dtype=torch.float64)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        t = c + s * (p[0] * x + p[1])
        out = Wt * (f(t) - fc) / amp + p[2]
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        if i == steps // 2:
            for gp in opt.param_groups:
                gp["lr"] = 5e-3
    q = p.detach()
    with torch.no_grad():
        t = c + s * (q[0] * x + q[1])
        loss = float(F.binary_cross_entropy_with_logits(Wt * (f(t) - fc) / amp + q[2], y))
    u, v, b2p = (float(z) for z in q)
    w1, b1 = s * u, c + s * v
    b2 = b2p - w2 * fc                                   # raw bias
    return {"u": u, "v": v, "b2p": b2p, "w1": w1, "b1": b1, "b2": b2, "loss": loss,
            "gap": bb.gap_of(f, w1, b1, w2),
            "solves": bool(solves(torch.tensor([w1, b1, w2, b2], dtype=torch.float64), f))}


def degenerate(r):
    return abs(r["loss"] - LOG2) < bb.DEGEN or abs(r["u"]) < 1e-3


def best_conditional(fam, a, w2, x, y, restarts=bb.RESTARTS):
    cheap = [minimise(fam, a, w2, x, y, seed=s, steps=bb.SCREEN) for s in range(restarts)]
    cheap = [r for r in cheap if not degenerate(r)]
    if not cheap:
        return None
    cheap.sort(key=lambda r: r["loss"])
    full = [minimise(fam, a, w2, x, y, start=[r["u"], r["v"], r["b2p"]]) for r in cheap[:bb.KEEP]]
    full = [r for r in full if not degenerate(r)]
    return min(full, key=lambda r: r["loss"]) if full else None
