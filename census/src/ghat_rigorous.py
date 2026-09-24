"""The certified Ĝ(a) as rigorous enclosures (author's decision 2026-09-24, option (a); Block 2).

The independent Arb checker (src/verify_certificates.py, check_ghat) writes results/certificate_checks/ghat_a*.json.
From them this producer writes results/ghat_rigorous.csv:
  Ghat_cert_rigorous  the rigorous lower bound of G at the witness of the published Ĝ_cert (kappa_certify's attained
                      point), rounded DOWN to a double: the value every R now uses
  Ghat_hi_rigorous    the rigorous upper bound over every leaf of the branch and bound, rounded UP to a double
  Ghat_cert_published, Ghat_hi_published   the old float endpoints (ghat_certified_all.csv, ghat_bnb.csv), kept for
                      the record; the strict checks on them keep reporting as they are
  bnb_arg_lower       the rigorous value at the branch and bound's own attained point: a larger proven lower bound at
                      some a, reported and NOT adopted (it would change the quantity Ĝ_cert, not just its rounding)

    python -m src.ghat_rigorous
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ghat_rigorous.csv"


def build():
    rows = []
    for p in sorted((RESULTS / "certificate_checks").glob("ghat_a*.json")):
        r = json.loads(p.read_text())
        a = float(p.stem.split("_a")[1])
        rows.append({"a": round(a, 2), "Ghat_cert_rigorous": r["witness_lower"], "Ghat_hi_rigorous": r["rigorous_upper"],
                     "Ghat_cert_published": r["ghat_cert"], "Ghat_hi_published": r["claim_hi"],
                     "bnb_arg_lower": r["bnb_arg_lower"], "structure_pass": all(
                         r["checks"][k] for k in ("files_match_committed_hashes", "coverage", "domain_contains_reduction",
                                                  "ghat_cert_below_hi")),
                     "strict_ghat_cert_attained": r["checks"]["ghat_cert_attained"],
                     "strict_every_leaf_below_claim_hi": r["checks"]["every_leaf_below_claim_hi"],
                     "leaves": r["leaves"]})
    d = pd.DataFrame(rows).sort_values("a")
    d["delta_cert"] = d.Ghat_cert_rigorous - d.Ghat_cert_published
    d["delta_hi"] = d.Ghat_hi_rigorous - d.Ghat_hi_published
    d["rel_delta_cert"] = d.delta_cert / d.Ghat_cert_published
    d["rel_delta_hi"] = d.delta_hi / d.Ghat_hi_published
    d.to_csv(OUT, index=False)
    return d


def ghat_R():
    """{a: the rigorous Ĝ_cert} -- the Ĝ every R uses.  Every a of ghat_certified_all.csv must be present."""
    d = pd.read_csv(OUT, float_precision="round_trip")
    g = dict(zip(d.a.round(2), d.Ghat_cert_rigorous))
    want = set(pd.read_csv(RESULTS / "ghat_certified_all.csv").a.round(2))
    missing = want - set(g)
    if missing:
        raise RuntimeError(f"no rigorous Ĝ for a = {sorted(missing)}")
    return g


def ghat_R_of(a):
    return ghat_R()[round(float(a), 2)]


def max_rel_change():
    d = pd.read_csv(OUT, float_precision="round_trip")
    return float(max(d.rel_delta_cert.abs().max(), d.rel_delta_hi.abs().max()))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(build().to_string(index=False))
