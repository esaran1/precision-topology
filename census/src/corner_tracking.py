"""EXPLORATORY (characterises, does not test): the active set of the gap-maximising placement from ε = 0.05 to 2.0.

At each a = 1 + ε: the global certified Ĝ(a) branch and bound (ghat_bnb.certify, full placement domain, target
rel 1e-8) gives a maximising placement (w1, b1).  If the second orientation is the active one, the placement is
replaced by its mirror (w1, 2π − b1) (x -> −x with φ -> −φ), so the canonical orientation is min_O φ − max_I φ.
Candidates: window edges and interior critical points of f_a (cos t = −1/a: local max at t = π − δ + 2πk, local
min at π + δ + 2πk, δ = arccos(1/a)).  Active: within TOL·Ĝ of the inner maximum / outer minimum.  The margin to
the nearest inactive candidate is recorded, so borderline classifications are visible.

Grid ε = 0.05 (0.025) 2.0; each change between neighbours is bisected to 0.001.  Also: the conditional
minimiser's active pair at the six certified switch brackets (cond_certified_bracket_evaluations.csv).

    python -m src.corner_tracking [workers]
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
TOL = 1e-6
INNER = (-0.8, 0.8)
OUTERS = {"O-": (-2.0, -1.2), "O+": (1.2, 2.0)}


def f(t, a):
    return t + a * np.sin(t)


def _cands(w, b, lo, hi, a, kind, tag):
    """Edges and interior local extrema of f_a on x ∈ [lo, hi] (kind 'max' or 'min')."""
    out = [(f"{tag}:x={lo:+.1f}", f(w * lo + b, a)), (f"{tag}:x={hi:+.1f}", f(w * hi + b, a))]
    t1, t2 = sorted((w * lo + b, w * hi + b))
    d = math.acos(1 / a)
    base = math.pi - d if kind == "max" else math.pi + d
    for k in range(math.floor((t1 - base) / (2 * math.pi)) - 1, math.ceil((t2 - base) / (2 * math.pi)) + 2):
        t = base + 2 * math.pi * k
        if t1 < t < t2:
            out.append((f"{tag}:crit-{kind}", f(t, a)))
    return out


def _reflect(n):
    tag, rest = n.split(":")
    tag = {"O+": "O-", "O-": "O+"}.get(tag, tag)
    if rest.startswith("x="):
        rest = f"x={-float(rest[2:]):+.1f}"
    return f"{tag}:{rest}"


def classify(w, b, a):
    def orient1(w, b):
        inner = _cands(w, b, *INNER, a, "max", "I")
        outer = sum((_cands(w, b, *OUTERS[k], a, "min", k) for k in OUTERS), [])
        imax = max(v for _, v in inner)
        omin = min(v for _, v in outer)
        return omin - imax, inner, outer, imax, omin
    G1 = orient1(w, b)[0]
    G2 = orient1(w, 2 * math.pi - b)[0]              # mirror placement carries the second orientation
    if G2 > G1:
        b = 2 * math.pi - b
    G, inner, outer, imax, omin = orient1(w, b)
    tol = TOL * max(G, 1e-12)
    act_i = sorted(n for n, v in inner if imax - v <= tol)
    act_o = sorted(n for n, v in outer if v - omin <= tol)
    if act_o and all(n.startswith("O+") for n in act_o):         # canonical mirror: x -> −x
        act_i, act_o = sorted(_reflect(n) for n in act_i), sorted(_reflect(n) for n in act_o)
    marg_i = min([imax - v for n, v in inner if n not in act_i], default=np.inf)
    marg_o = min([v - omin for n, v in outer if n not in act_o], default=np.inf)
    return {"G": G, "inner_active": "|".join(act_i), "outer_active": "|".join(act_o),
            "active_set": "|".join(act_i + act_o), "margin_inner_inactive": marg_i / G,
            "margin_outer_inactive": marg_o / G, "margin_inner_abs": marg_i, "margin_outer_abs": marg_o,
            "w1_canon": w, "b1_canon": b}


def _job(eps):
    from .ghat_bnb import certify
    a = 1 + eps
    r = certify(a, target_rel=1e-8)
    return {"eps": round(eps, 6), "a": a, "Ghat_lo": r["Ghat_lo"], "Ghat_hi": r["Ghat_hi"],
            "converged": r["converged"], "surviving": r["surviving"],
            "u": r["w1"] / math.sqrt(eps), **classify(r["w1"], r["b1"], a)}


def main(workers=2):
    grid = [round(e, 6) for e in np.arange(0.05, 2.0 + 1e-9, 0.025)]
    with Pool(workers) as p:
        rows = p.map(_job, grid, chunksize=1)
        d = pd.DataFrame(rows).sort_values("eps").reset_index(drop=True)
        changes = [(d.eps[i], d.eps[i + 1]) for i in range(len(d) - 1)
                   if d.active_set[i] != d.active_set[i + 1]]
        extra = []
        for lo, hi in changes:
            slo = d[d.eps == lo].active_set.iloc[0]
            while hi - lo > 0.001 + 1e-12:
                mids = list(np.linspace(lo, hi, 4)[1:-1])      # two interior points per round, in parallel
                res = p.map(_job, [round(m, 6) for m in mids], chunksize=1)
                extra += res
                seq = [(lo, slo)] + [(r["eps"], r["active_set"]) for r in res]
                j = next((i for i in range(1, len(seq)) if seq[i][1] != slo), None)
                if j is None:
                    lo = seq[-1][0]
                else:
                    lo, hi = seq[j - 1][0], seq[j][0]
    d = pd.concat([d, pd.DataFrame(extra)]).sort_values("eps").reset_index(drop=True)
    d["K_eps"] = d.Ghat_lo / d.eps ** 1.5
    d.to_csv(RESULTS / "corner_tracking.csv", index=False)
    ch = [{"eps_before": d.eps[i], "eps_after": d.eps[i + 1], "from": d.active_set[i], "to": d.active_set[i + 1]}
          for i in range(len(d) - 1) if d.active_set[i] != d.active_set[i + 1]]
    pd.DataFrame(ch).to_csv(RESULTS / "corner_tracking_changes.csv", index=False)
    print(pd.DataFrame(ch).to_string(index=False))
    switch_active()


def switch_active():
    """The conditional minimiser's active pair at the six certified switch brackets (both ends)."""
    ev = pd.read_csv(RESULTS / "cond_certified_bracket_evaluations.csv")
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    rows = []
    for r in br[br.kind == "glob"].itertuples():
        for s in (r.w2_lo, r.w2_hi):
            e = ev[(ev.a.round(2) == round(r.a, 2)) & (np.isclose(ev.s, s))].iloc[0]
            w, b = abs(e.glob_w1), e.glob_b1 % (2 * math.pi)   # (−w, b) is the x-reflection (symmetric windows)
            c = classify(w, b, float(r.a))
            rows.append({"a": r.a, "s": s, **{k: c[k] for k in ("G", "inner_active", "outer_active",
                                                                  "margin_inner_abs", "margin_outer_abs")}})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "corner_tracking_switch_active.csv", index=False)
    print(t.to_string(index=False))




def decompose():
    """EXPLORATORY.  (i) At ε = 0.05, 0.3, 0.6, 1.0, 1.5, 2.0: the corner solved from its two tie equations with the
    exact f_a against the branch-and-bound supremum.  (ii) At the six certified a: R/R_inf - 1 = α + κ + ακ with
    α = A/A* - 1 (A = s ε^{3/2}, certified bracket midpoint) and κ = K(ε)/K0 - 1 (certified Ĝ), against the
    first-order terms A'/A*·ε and k1·ε."""
    from .first_order import _phi_exact
    d = pd.read_csv(RESULTS / "corner_tracking.csv").set_index("eps")
    fo = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    K0 = 0.5 * (fo.K_vertex_lo + fo.K_vertex_hi)
    rows, t = [], np.array([1.6056, 1.2042])
    for e in (0.05, 0.3, 0.6, 1.0, 1.5, 2.0):
        a = 1 + e
        E = lambda tt: np.array([_phi_exact(tt[1] - 0.8 * tt[0], a, e) - _phi_exact(tt[1] + 0.8 * tt[0], a, e),
                                 _phi_exact(tt[1] - 2.0 * tt[0], a, e) - _phi_exact(tt[1] - 1.2 * tt[0], a, e)])
        for _ in range(60):
            J = np.array([(E(t + 1e-8 * np.eye(2)[i]) - E(t)) / 1e-8 for i in range(2)]).T
            t = t - np.linalg.solve(J, E(t))
        Kv = _phi_exact(t[1] - 1.2 * t[0], a, e) - _phi_exact(t[1] + 0.8 * t[0], a, e)
        r = d.loc[round(e, 6)]
        rows.append({"part": "vertex_check", "eps": e, "K_vertex": Kv, "K_bnb_lo": r.Ghat_lo / e ** 1.5,
                     "K_bnb_hi": r.Ghat_hi / e ** 1.5,
                     "vertex_in_bnb": r.Ghat_lo / e ** 1.5 - 1e-9 <= Kv <= r.Ghat_hi / e ** 1.5 + 1e-9})
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    for r in br[br.kind == "glob"].sort_values("a").itertuples():
        e = r.a - 1
        al = 0.5 * (r.w2_lo + r.w2_hi) * e ** 1.5 / fo.A_star_lo - 1
        ka = r.Ghat_cert / e ** 1.5 / K0 - 1
        rows.append({"part": "decomposition", "eps": e, "alpha": al, "alpha_first_order": fo.A1_over_A_lo * e,
                     "kappa": ka, "kappa_first_order": fo.k1_lo * e, "dev": al + ka + al * ka,
                     "dev_first_order": fo.c1_lo * e, "excess_over_first_order": al + ka + al * ka - fo.c1_lo * e,
                     "A_higher_order": al - fo.A1_over_A_lo * e, "K_higher_order": ka - fo.k1_lo * e,
                     "product_term": al * ka})
    t_ = pd.DataFrame(rows)
    t_.to_csv(RESULTS / "corner_tracking_decomposition.csv", index=False)
    print(t_.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "switch":
        switch_active()
    elif len(sys.argv) > 1 and sys.argv[1] == "decompose":
        decompose()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
