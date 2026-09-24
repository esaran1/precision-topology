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
            "regenerate": f"python -m src.cert_export finite {a} {s} {name}",
            "search_result": {"minus": {k: rm[k] for k in ("lower", "upper", "rounds")},
                              "plus": {k: rp[k] for k in ("lower", "upper", "rounds")}}}
    CERTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CERTS / f"{name}.npz", **arrays)
    (CERTS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def ghat(a, name):
    """One Ĝ(a) enclosure certificate: the global branch and bound (ghat_bnb.certify, recorded; its result asserted
    equal to results/ghat_bnb.csv) and the witness of the paper's Ĝ_cert (kappa_certify.certify_exact's attained
    point; its value asserted equal to results/ghat_certified_all.csv)."""
    import pandas as pd
    from .ghat_bnb import certify
    from .kappa_certify import certify_exact
    rec = {}
    r = certify(a, record=rec)
    ref = pd.read_csv(CERTS.parent / "ghat_bnb.csv", float_precision="round_trip")
    ref = ref[ref.a.round(2) == round(a, 2)].iloc[0]
    assert r["Ghat_lo"] == ref.Ghat_lo and r["Ghat_hi"] == ref.Ghat_hi, "search result differs from ghat_bnb.csv"
    z = certify_exact(a)
    gc = pd.read_csv(CERTS.parent / "ghat_certified_all.csv", float_precision="round_trip")
    gc = gc[gc.a.round(2) == round(a, 2)].iloc[0]
    assert z["Ghat_lo"] == gc.Ghat_certified, "zoom result differs from ghat_certified_all.csv"
    lv = np.concatenate([np.full(len(ix), rnd, dtype=np.int8) for rnd, ix, _ in rec["leaves"]])
    ij = np.concatenate([ix for _, ix, _ in rec["leaves"]])
    reason = np.concatenate([np.full(len(ix), rs, dtype=np.int8) for _, ix, rs in rec["leaves"]])
    arrays = {"level": lv, "iw": ij[:, 0].astype(np.int64), "ib": ij[:, 1].astype(np.int64), "reason": reason}
    meta = {"kind": "ghat_enclosure", "name": name, "a": a, "nw": rec["nw"], "nb": rec["nb"],
            "hw0": rec["hw0"], "hb0": rec["hb0"], "domain": "w1 in [0, a], b1 in [0, 2pi] (contains (0, a/1.4] x [0, 2pi))",
            "gap": "G = max(min_O phi - max_I phi, min_I phi - max_O phi), phi = f_a(w1 x + b1), I = [-0.8, 0.8], O = +-[1.2, 2.0]",
            "step": "(1 + a)(2.8 hw + 2 hb)", "claim_hi": r["Ghat_hi"], "bnb_lo": r["Ghat_lo"],
            "bnb_arg": [r["w1"], r["b1"]], "ghat_cert": float(gc.Ghat_certified), "ghat_cert_witness": [z["w1"], z["b1"]],
            "regenerate": f"python -m src.cert_export ghat {a} {name}"}
    CERTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CERTS / f"{name}.npz", **arrays)
    (CERTS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def manifest():
    """results/certificates_manifest.csv: every exported file's SHA-256 and size, and the command that regenerates
    it bit-identically.  The data files themselves are not committed."""
    import csv
    import hashlib
    rows = []
    for p in sorted(CERTS.glob("*.json")):
        meta = json.loads(p.read_text())
        for f in (p, p.with_suffix(".npz")):
            if f.exists():
                rows.append({"file": f"results/certificates/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                             "bytes": f.stat().st_size, "regenerate": meta.get("regenerate", "")})
    out = CERTS.parent / "certificates_manifest.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "sha256", "bytes", "regenerate"])
        w.writeheader(); w.writerows(rows)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        print(manifest())
    elif sys.argv[1] == "ghat":
        print(json.dumps(ghat(float(sys.argv[2]), sys.argv[3]), indent=1, default=float)[:1500])
    elif sys.argv[1] == "finite":
        print(json.dumps(finite(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]), indent=1, default=float)[:1500])
