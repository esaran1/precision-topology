"""Track U, U2: timing of check_u2_unit (the independent checker's ε-interval functions) on the producer's exported
units.  It imports ONLY the checker (src/verify_certificates.py) and the gate; it reads the producer's exports as data
(their labels say "PRODUCER-ONLY: not independently checked"); the population comes from the hash-checked certificate
file.  Entry point gated like every U2 compute entry point: it refuses to start without a separate, earlier gate entry.

    python -m src.u2_checker_timing <out.json> <prefix> [<prefix> ...]
      env: CHK_UNITS = comma list of unit indices (default: first, middle, last); CHK_MAX_EVALS (outer re-search cap);
           CHK_RING_TILES (tiles sampled per unit, default all); CHK_SWITCH_SUBSLABS (default 1); CHK_SKIP_OUTER
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "uniform_eps" / "u2_sample"


def _rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def diag_active(prefix):
    """Print the active-pair window balls for the first row of the first switch sub-slab (diagnosis only)."""
    from . import verify_certificates as vc
    from flint import arb
    data = json.loads((OUT / f"{prefix}_switch.json").read_text())
    S = (data["sub_slabs"] if isinstance(data, dict) else data)[0]
    r = S["rows"][0]
    act = vc.U2Act(vc._ival(*S["eps"]))
    P = arb(repr(float(r["xt"][0]))) + arb(0, 1e-4); Q = arb(repr(float(r["xt"][1]))) + arb(0, 1e-4)
    I = vc._ival(min(vc.lo_(P * arb("-0.8") + Q), vc.lo_(P * arb("0.8") + Q)), max(vc.hi_(P * arb("-0.8") + Q), vc.hi_(P * arb("0.8") + Q)))
    c_ = arb("0.3")
    print(json.dumps({"eps": S["eps"], "row": [r["A_lo"], r["A_hi"]], "xt": r["xt"], "I": str(I),
                      "df_lo_part": str(act.df(vc._ival(vc.lo_(I), c_))), "d2f_hi_part": str(act.d2f(vc._ival(c_, vc.hi_(I))))}))


def main(out_path, prefixes):
    from . import verify_certificates as vc
    x, y, sym = vc._verified_limit_data()
    assert sym
    rows = []

    def save():
        Path(out_path).write_text(json.dumps(rows, indent=1, default=float))
    t0 = time.time()
    Bt = [(P, vc.localisation_bounds(x, y, P)[0]) for P in (18.0, 20.0, 22.0, 24.0)]
    Bf = vc.localisation_bounds(x, y, 24.0)[1]
    rows.append({"B_table_seconds": time.time() - t0}); save()
    max_evals = int(os.environ.get("CHK_MAX_EVALS", "40000"))
    for pre in prefixes:
        metas = sorted(glob.glob(str(OUT / f"{pre}_k*.meta.json")))
        if not metas:
            continue
        if os.environ.get("CHK_SWITCH_ONLY"):
            idx = []
        elif os.environ.get("CHK_UNITS"):
            idx = [int(v) for v in os.environ["CHK_UNITS"].split(",") if v.strip() and int(v) < len(metas)]
        else:
            idx = sorted({0, len(metas) // 2, len(metas) - 1})
        for i in idx:
            h = json.loads(Path(metas[i]).read_text())
            r = {"unit": Path(metas[i]).name, "header": h, "input_label": h.get("check_status")}
            reg = h["regime"]
            if os.environ.get("CHK_SKIP_OUTER"):
                o = {"verdict": "skipped", "evals": 0, "seconds": 0.0}
            else:
                o = vc.u2_outer_arb(x, y, h["centre"], h["A"][0], h["A"][1], h["eps"][0], h["eps"][1], reg,
                                    B_table=Bt if reg == "R" else None, B_full=Bf if reg == "R" else None,
                                    max_evals=max_evals)
            r["outer"] = {k: o[k] for k in o if k != "unresolved_first"}
            r["outer"]["seconds_per_eval"] = o["seconds"] / max(1, o["evals"])
            n = int(round(2 * 0.15 / 0.0125)); g = -0.15 + (np.arange(n) + 0.5) * 0.0125
            tiles = [(float(a), float(b), 0.00625) for a in g for b in g if max(abs(a), abs(b)) + 0.00625 > 0.05 + 1e-12]
            nt = int(os.environ.get("CHK_RING_TILES", str(len(tiles))))
            sel = [tiles[j] for j in np.linspace(0, len(tiles) - 1, nt).round().astype(int)]
            rr = vc.u2_ring_fast_arb(x, y, h["centre"], h["A"][0], h["A"][1], h["eps"][0], h["eps"][1], boxes=sel)
            rr["tiles_total"] = len(tiles); rr["seconds_per_tile"] = rr["seconds"] / len(sel)
            r["ring_fast"] = rr
            r["pd"] = vc.u2_pd_arb(x, y, h["centre"], h["A"][0], h["A"][1], h["eps"][0], h["eps"][1])
            r["rss_gb"] = _rss_gb()
            rows.append(r); save()
            print(json.dumps({"unit": r["unit"], "outer": [o["verdict"], o["evals"], o.get("coupled"), o.get("direct"),
                              round(o["seconds"], 1)], "ring": [rr["certified"], rr["fast"], rr["fallback"],
                              round(rr["seconds_per_tile"], 2)], "pd": [r["pd"]["pd"], r["pd"]["boxes"],
                              round(r["pd"]["seconds"], 1)], "rss_gb": round(r["rss_gb"], 2)}), flush=True)
        sw = glob.glob(str(OUT / f"{pre}_switch.json"))
        if sw:
            data = json.loads(Path(sw[0]).read_text())
            L = data["sub_slabs"] if isinstance(data, dict) else data
            for S in L[:int(os.environ.get("CHK_SWITCH_SUBSLABS", "1"))]:
                res = vc.u2_switch_shear_arb(x, y, S["eps"][0], S["eps"][1], S["Z"], S["rows"], S["J_tube"])
                res.update({"sub_slab": S["eps"], "n_sub_slabs_in_file": len(L), "prefix": pre, "rss_gb": _rss_gb()})
                rows.append({"switch": res}); save()
                print(json.dumps({"switch": {k: res[k] for k in res if k != "bad_rows"}}), flush=True)


if __name__ == "__main__":
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(" ".join(["u2_checker_timing"] + sys.argv[1:]))
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        sys.exit(3)
    if os.environ.get("CHK_DIAG_ACTIVE"):
        diag_active(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2:])
