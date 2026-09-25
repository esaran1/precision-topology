"""Finish the direct check's unfinished restarts with a fast local optimiser (author's request 2026-09-24).

The direct check (width2_w0.smallscale) keeps its candidates only in memory; its summary row counts the restarts still
improving at the final 100,000-iteration cap (45 at a = 1.30, R₂ = 0.001) without saving them.  Identifying exactly those
would need the 5-hour cap ladder again.  Instead this reproduces the check's first stage exactly (search_batch with the
same seed and starts, deterministic, row-independent) and finishes EVERY restart that was still improving at the first
cap (2,000 iterations): a superset of the ones still improving at 100,000.

Optimiser: damped Newton (Levenberg-Marquardt) with the exact Hessian (torch autograd) on the joint variables
z = (α₁, β₁, α₂, β₂, v₁, v₂, b), φ = (v₁u₁ + v₂u₂)/(|v₁| + |v₂|).  Minimising jointly over b has the same minimisers as the
profiled loss; the reported loss is the PROFILED loss (width2_unplaced.loss) at the end point, directly comparable with
the retained minimiser's.  (scipy is not installed, so no L-BFGS; with 7 variables Newton is cheap and is not slowed by
the flat landscape at small scale.)

    python -m src.width2_finish <act> [R2 ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "width2_w0_parts"


def _joint_loss(z, s, X, Y, act):
    import torch
    u = (lambda t: t + act.a * torch.sin(t)) if act.name == "fa" else torch.tanh
    n1 = z[4].abs() + z[5].abs()
    phi = (z[4] * u(z[0] * X + z[1]) + z[5] * u(z[2] * X + z[3])) / n1
    Z = s * phi + z[6]
    return (torch.nn.functional.softplus(Z) - Y * Z).mean()


def newton(z0, s, x, y, act, maxit=300, gtol=1e-15):
    """Damped Newton on the joint objective.  Returns (z, iterations, final max|grad|, converged)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    f = lambda z: _joint_loss(z, s, X, Y, act)
    z = torch.tensor(np.asarray(z0, float), dtype=torch.float64)
    mu = 1e-3
    fz = float(f(z))
    it = 0
    for it in range(1, maxit + 1):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(f(zz), zz)[0].numpy()
        if np.abs(g).max() <= gtol:
            return z.numpy(), it, float(np.abs(g).max()), True
        H = torch.autograd.functional.hessian(f, z).numpy()
        scale = max(1.0, float(np.abs(np.diag(H)).max()))
        accepted = False
        for _ in range(30):
            try:
                d = np.linalg.solve(H + mu * scale * np.eye(7), -g)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            zn = z + torch.tensor(d)
            fn = float(f(zn))
            if fn < fz:
                z, fz, accepted = zn, fn, True
                mu = max(mu / 3, 1e-12)
                break
            mu *= 10
        if not accepted:                                    # no decrease possible at float precision
            zz = z.clone().requires_grad_(True)
            g = torch.autograd.grad(f(zz), zz)[0].numpy()
            return z.numpy(), it, float(np.abs(g).max()), True
    zz = z.clone().requires_grad_(True)
    g = torch.autograd.grad(f(zz), zz)[0].numpy()
    return z.numpy(), it, float(np.abs(g).max()), False


def finish(act_name, R2):
    from .width2_conditional import population, profile_b, search_batch
    from .width2_unplaced import g_exact, loss
    from .width2_w0 import ACTS, CAPS, _classify_caps, _is_pair, _read
    act = ACTS[act_name]
    gh = float(_read(f"gamma_{act_name}.csv").query("search == 'nm'").gamma_lo.iloc[0])
    x, y = population()
    s = 2 * R2 / gh
    _, cands = search_batch(s, x, y, act, restarts=4000, seed=7000, maxit=CAPS[0])     # the check's first stage
    improving, _ = _classify_caps(cands, CAPS[0])
    rows = []
    for c in improving:
        t = float(np.clip(c["p"][4], -1, 1))
        q = np.r_[c["p"][:4], t, c["sigma"] * (1 - abs(t))]
        z0 = np.r_[q, profile_b(s * (q[4] * act.u(q[0] * x + q[1]) + q[5] * act.u(q[2] * x + q[3])), y)]
        z, it, gmax, conv = newton(z0, s, x, y, act)
        qf = z[:6]
        Lf = loss(qf, s, x, y, act)
        ge = g_exact(qf, act)
        n1 = abs(qf[4]) + abs(qf[5])
        tt = qf[4] / n1
        pair, c_lin = _is_pair(np.r_[qf[:4], tt], 1.0 if qf[5] >= 0 else -1.0)
        rows.append({"act": act_name, "R2": R2, "k": c["k"], "loss_at_cap": c["loss"], "loss_final": Lf,
                     "newton_iterations": it, "grad_max": gmax, "converged": conv, "G_lo": ge[0], "G_hi": ge[1],
                     "placed": bool(ge[0] > 0), "cancelling_pair": bool(pair), "linear_c": float(c_lin),
                     "min_share": float(min(abs(qf[4]), abs(qf[5])) / n1)})
    return pd.DataFrame(rows)


def run(act_name, R2s):
    out = PARTS / f"finish_{act_name}.csv"
    done = set() if not out.exists() else set(np.round(pd.read_csv(out).R2, 10))
    for R2 in R2s:
        if round(R2, 10) in done:
            continue
        d = finish(act_name, R2)
        d.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(json.dumps({"act": act_name, "R2": R2, "finished": len(d), "converged": int(d.converged.sum()),
                          "placed_pair": int((d.placed & d.cancelling_pair).sum()), "min_loss": float(d.loss_final.min())}),
              flush=True)


def summary():
    """Join with the direct check's retained minimiser: where the finished restarts converge, and whether any ends
    below it (by more than 1e-12, the resolution of the losses compared)."""
    rows = []
    for f in sorted(PARTS.glob("finish_f*.csv")):
        d = pd.read_csv(f, float_precision="round_trip")
        act = d.act.iloc[0]
        smf = PARTS / f"smallscale_{act}.csv"
        sm = pd.read_csv(smf, float_precision="round_trip") if smf.exists() else pd.DataFrame(columns=["R2"])
        for R2, g in d.groupby("R2"):
            r = sm[np.isclose(sm.R2, R2)]
            ret = float(r.retained_loss.iloc[0]) if len(r) else np.nan
            below = g[g.loss_final < ret - 1e-12] if len(r) else g.iloc[0:0]
            rows.append({"act": act, "R2": R2, "finished": len(g), "converged": int(g.converged.sum()),
                         "end_placed_pair": int((g.placed & g.cancelling_pair).sum()),
                         "end_placed_other": int((g.placed & ~g.cancelling_pair).sum()),
                         "end_unplaced_single_unit": int((~g.placed & (g.min_share <= 0.01)).sum()),
                         "end_unplaced_other": int((~g.placed & (g.min_share > 0.01)).sum()),
                         "retained_loss": ret, "min_final_minus_retained": float(g.loss_final.min() - ret),
                         "n_below_retained": len(below)})
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "width2_finish_summary.csv", index=False)
    return s


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        pd.set_option("display.width", 250)
        print(summary().to_string(index=False))
    else:
        from .width2_w0 import SMALL_R2
        run(sys.argv[1], [float(v) for v in sys.argv[2:]] or list(SMALL_R2))
