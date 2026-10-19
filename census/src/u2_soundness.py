"""Track U, U2: soundness spot check of the switch rows the checker's replay fails (the author's step 1, 2026-10-19).

For failing rows spread across ε: the active gap G(p, q; ε) = φ_ε(q − 1.2p) − φ_ε(q + 0.8p) is evaluated at 60 digits
(mpmath, closed form φ_ε = (u − (1 + ε) sin u)/ε^{3/2}, u = √ε σ, and h(σ) at ε = 0 -- a code path independent of both
the producer and the checker) at the centre and the four (p, q) corners of the sheared box, at the ends and the middle of
the ε-piece.  Each point value is reported next to the producer's enclosure (src.u2_sample.gap_sheared with the
producer's radii) and the checker's enclosure (verify_certificates.u2_gap_shear_arb with the checker's own radii).
A point of the wrong sign, or outside the producer's enclosure, is a PRODUCER problem (the run then stops there).

All producer output used here is PRODUCER-ONLY: not independently checked.

    python -m src.u2_soundness <out.json> <prefix:subslab> [<prefix:subslab> ...]     (gated entry point)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "uniform_eps" / "u2_sample"


def gap60(p, q, eps):
    from mpmath import mp, mpf, sin, sqrt
    with mp.workdps(60):
        p, q, e = mpf(p), mpf(q), mpf(eps)

        def phi(s):
            if e == 0:
                return -s + s ** 3 / 6
            u = sqrt(e) * s
            return (u - (1 + e) * sin(u)) / e ** mpf("1.5")
        return phi(q - mpf("1.2") * p) - phi(q + mpf("0.8") * p)


def main(out_path, targets, per_slab):
    from . import verify_certificates as vc
    from . import u2_sample as m
    x, y, _ = vc._verified_limit_data()
    report = {"check_status_of_inputs": "PRODUCER-ONLY: not independently checked", "rows": []}
    for tg in targets:
        prefix, sub = tg.split(":")
        data = json.loads((OUT / f"{prefix}_switch.json").read_text())
        L = data["sub_slabs"] if isinstance(data, dict) else data
        S = L[int(sub)]
        e_lo, e_hi = S["eps"]
        ec = 0.5 * (e_lo + e_hi)
        t0 = time.time()
        rep = vc.u2_switch_shear_arb(x, y, e_lo, e_hi, S["Z"], S["rows"], S["J_tube"], return_rows=True)
        byA = {}
        for rl in rep["row_log"]:
            byA.setdefault(tuple(rl["A"]), []).append(rl)
        bad = [b for b in rep["bad_all"] if b[2] == "sign" and tuple(b[:2]) in byA]
        pick = [bad[j] for j in np.unique(np.linspace(0, len(bad) - 1, min(per_slab, len(bad))).round().astype(int))] \
            if bad else []
        prod_rows = {(r["A_lo"], r["A_hi"]): r for r in S["rows"]}
        for b in pick:
            pr = prod_rows.get((b[0], b[1]))
            for rl in byA[tuple(b[:2])]:
                lo_o, hi_o = rl["piece"]
                em = 0.5 * (lo_o + hi_o)
                X1 = pr["X1"]; Rp = pr["R"]
                xm = [pr["xt"][0] + (em - ec) * X1[0], pr["xt"][1] + (em - ec) * X1[1]]
                gp = m.gap_sheared(xm, Rp, X1, lo_o, hi_o)              # the producer's enclosure (its radii)
                pts = []
                for e in (lo_o, em, hi_o):
                    cp = pr["xt"][0] + (e - ec) * X1[0]; cq = pr["xt"][1] + (e - ec) * X1[1]
                    for dp, dq in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                        v = gap60(cp + dp * Rp[0], cq + dq * Rp[1], e)
                        pts.append({"eps": e, "corner": [dp, dq], "G60": float(v),
                                    "in_producer": bool(gp[0] <= float(v) <= gp[1]),
                                    "sign_ok": bool(float(v) < 0) if rl["need_sign"] < 0 else bool(float(v) > 0)})
                report["rows"].append({"slab": [e_lo, e_hi], "A": list(b[:2]), "piece": [lo_o, hi_o],
                                       "need_sign": rl["need_sign"], "producer_enclosure": list(gp),
                                       "producer_width": gp[1] - gp[0], "producer_R": Rp,
                                       "checker_enclosure": rl["G"], "checker_width": rl["G"][1] - rl["G"][0],
                                       "checker_R": rl["R_checker"], "points": pts, "checker_piece": {k_: rl.get(k_) for k_ in ("Lb", "Ub", "cuts")},
                                       "all_points_in_producer": all(q_["in_producer"] for q_ in pts),
                                       "all_signs_ok": all(q_["sign_ok"] for q_ in pts)})
        report.setdefault("slabs", []).append({"target": tg, "rows_failing": len(rep["bad_all"]),
                                               "reasons": rep["bad_reasons"], "seconds": time.time() - t0})
        Path(out_path).write_text(json.dumps(report, indent=1, default=float))
        print(tg, len(pick), "rows; any producer problem:",
              any(not (r_["all_points_in_producer"] and r_["all_signs_ok"]) for r_ in report["rows"]), flush=True)
    rows = report["rows"]
    report["summary"] = {"rows_checked": len({(tuple(r_["slab"]), tuple(r_["A"])) for r_ in rows}),
                         "pieces_checked": len(rows), "points": sum(len(r_["points"]) for r_ in rows),
                         "producer_problem": any(not (r_["all_points_in_producer"] and r_["all_signs_ok"]) for r_ in rows)}
    Path(out_path).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["summary"]), flush=True)


def cells(out_path, units, n_small=5, n_rand=5):
    """Step 2: on outer-exclusion leaves of the given units (prefix_kNNN), the producer's cell bound against the
    checker's coupled bound, term by term (both evaluated at the same cell, A-sub-interval and ε-slab)."""
    from . import verify_certificates as vc
    from . import u2_sample as m
    x, y, _ = vc._verified_limit_data()
    rng = np.random.default_rng(0)
    rep = {"check_status_of_inputs": "PRODUCER-ONLY: not independently checked", "cells": []}
    for u in units:
        h = json.loads((OUT / f"{u}.meta.json").read_text())
        L = np.load(OUT / f"{u}.npz")
        pert = np.nonzero(L["reason"] == 3)[0]
        D = L["D"][pert]
        pick = list(pert[np.argsort(D)[:n_small]]) + list(rng.choice(pert, size=min(n_rand, len(pert)), replace=False))
        for i in pick:
            pc, qc, hp, hq = (float(L[k][i]) for k in ("p", "q", "hp", "hq"))
            tp = m.outer_cell_terms(pc, qc, hp, hq, h["A"][0], h["A"][1], h["eps"][0], h["eps"][1], h["centre"], x, y)
            tc = {}
            t0 = time.time()
            vc.u2_coupled_lower(x, y, pc, qc, hp, hq, h["A"][0], h["A"][1], h["eps"][0], h["eps"][1],
                                h["centre"][0], h["centre"][1], terms=tc)
            rep["cells"].append({"unit": u, "cell": [pc, qc, hp, hq], "producer_D_recorded": float(L["D"][i]),
                                 "producer_terms": tp, "checker_terms": tc, "checker_seconds": time.time() - t0})
            Path(out_path).write_text(json.dumps(rep, indent=1, default=float))
        print(u, "done", flush=True)


if __name__ == "__main__":
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(" ".join(["u2_soundness"] + sys.argv[1:]))
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        sys.exit(3)
    if os.environ.get("SND_CELLS"):
        cells(sys.argv[1], sys.argv[2:])
    else:
        main(sys.argv[1], sys.argv[2:], int(os.environ.get("SND_PER_SLAB", "4")))
