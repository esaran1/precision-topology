"""Export the certificates the paper claims as data files (Block 2b).  The SEARCH side: it re-runs each certifying
search with its recording hook (results unchanged -- asserted against the committed artifacts) and writes
results/certificates/<name>.npz (cells, data points) and <name>.json (the claim, constants, domain).  Nothing about
the search needs to be trusted: src/verify_certificates.py re-checks every file independently.

    python -m src.cert_export finite <a> <s> <name>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CERTS = Path(__file__).resolve().parents[1] / "results" / "certificates"


def finite(a, s, name, x=None, y=None):
    """One finite-a status certificate at scale s: both regions' leaves, with the status rule of
    conditional_certified.evaluate (tolerance 1e−7, then 1e−9 if unresolved)."""
    from .conditional_certified import _population
    from .profiled_bnb import certify, profile
    if x is None:
        x, y = _population()
    for tol in (1e-7, 1e-9):
        rec = {"-": {}, "+": {}}
        rm = certify(s, a, x, y, "-", tol=tol, record=rec["-"])
        rp = certify(s, a, x, y, "+", tol=tol, record=rec["+"])
        status = "minus" if rp["lower"] > rm["upper"] else "plus" if rm["lower"] > rp["upper"] else "unresolved"
        if status != "unresolved":
            break
    assert rec["-"]["W"] == rec["+"]["W"] and rec["-"]["nw"] == rec["+"]["nw"]
    win, lose = ("-", "+") if status == "minus" else ("+", "-")
    wr = rm if win == "-" else rp
    if win == "-" and wr["upper"] == math.log(2) and (wr["arg_w1"], wr["arg_b1"]) == (0.0, 0.0):
        U_point = None                                   # the constant predictor
    else:
        _, b2, _, _, _ = profile(np.array([wr["arg_w1"]]), np.array([wr["arg_b1"]]), s, a, x, y)
        U_point = [float(wr["arg_w1"]), float(wr["arg_b1"]), float(b2[0])]
    arrays = {"x": np.asarray(x, float), "y": np.asarray(y, float)}
    for reg in ("-", "+"):
        L = np.concatenate(rec[reg]["leaves"])
        for k in ("level", "iw", "ib", "cw", "cb", "lb", "G", "reason"):
            arrays[f"{k}_{reg}"] = np.asarray(L[k])
    meta = {"kind": "finite_a_status", "name": name, "a": a, "s": s, "tol": tol, "status": status,
            "winning_region": win, "losing_region": lose, "U_search": wr["upper"], "U_point": U_point,
            "W": rec["-"]["W"], "nw": int(rec["-"]["nw"]), "nb": int(rec["-"]["nb"]), "hw0": rec["-"]["hw0"],
            "hb0": rec["-"]["hb0"], "h0": rec["-"]["h0"], "domain": "w1 in [-W, W], b1 in [0, 2pi)",
            "objective": "L*(w1, b1; s) = min_b2 mean softplus(z) - y z, z = s f_a(w1 x + b1) + b2, f_a(t) = t + a sin t",
            "gap": "G = min_O f_a(w1 x + b1) - max_I f_a(w1 x + b1), I = [-0.8, 0.8], O = +-[1.2, 2.0]",
            "lower_bound_search": "L*(c) - |dL/dw| hw - |dL/db| hb - 0.5 s a mean((|x| hw + hb)^2)",
            "gap_step_search": "(1 + a)(2.8 hw + 2 hb)",
            "search_result": {"minus": {k: rm[k] for k in ("lower", "upper", "rounds")},
                              "plus": {k: rp[k] for k in ("lower", "upper", "rounds")}}}
    CERTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CERTS / f"{name}.npz", **arrays)
    (CERTS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


if __name__ == "__main__":
    if sys.argv[1] == "finite":
        print(json.dumps(finite(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]), indent=1, default=float)[:1500])
