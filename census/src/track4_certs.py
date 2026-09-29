"""Track 4 (discussion phase): independent checks of the outer-exclusion step and of the solve brackets.

This is the DRIVER.  The checks themselves are in src/verify_certificates.py (the independent Arb checker, which imports
nothing from src/).  The finite-a solve-bracket certificates are exported by the producer side (src/cert_export.solve,
which re-runs the status branch and bound at tolerance 1e−11 with its recording hook); their SHA-256 hashes are appended
to results/certificates_manifest.csv, and verify_certificates.check_solve then checks the hash-verified files.

    python -m src.track4_certs finite [name ...]    # export + manifest + check_solve, one end at a time (checkpointed)
    python -m src.track4_certs outer_solve          # verify_certificates.track4_outer('solve')
    python -m src.track4_certs outer_annulus        # verify_certificates.track4_outer('annulus'), 40 sub-intervals
    python -m src.track4_certs solve_limit          # verify_certificates.track4_solve_limit()
    python -m src.track4_certs summary              # certificate_checks/track4_summary.json

Run each command as one process under nice 15 (the caller sets the niceness).  Results: results/certificate_checks/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKS = RESULTS / "certificate_checks"


def finite_jobs():
    """The 12 finite-a solve-bracket ends (a = 1.30-1.60, lo and hi), from the published mn2_solve_finite.csv."""
    d = pd.read_csv(RESULTS / "mn2_solve_finite.csv", float_precision="round_trip")
    jobs = []
    for a, g in d.groupby("a"):
        g = g.sort_values("s")
        for end, (_, r) in zip(("lo", "hi"), g.iterrows()):
            jobs.append({"name": f"solve_a{a:.2f}_{end}", "a": float(a), "s": float(r.s),
                         "published_sign": r.certified_sign})
    return jobs


def append_manifest(name):
    """Append the SHA-256 of a newly exported certificate's two files to certificates_manifest.csv (if absent)."""
    from .cert_export import CERTS
    man = RESULTS / "certificates_manifest.csv"
    have = {r["file"] for r in csv.DictReader(man.open())}
    meta = json.loads((CERTS / f"{name}.json").read_text())
    with man.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "sha256", "bytes", "regenerate"], lineterminator="\n")
        for ext in ("json", "npz"):
            f = CERTS / f"{name}.{ext}"
            key = f"results/certificates/{f.name}"
            if key not in have:
                h = hashlib.sha256()
                with f.open("rb") as fh2:
                    for blk in iter(lambda: fh2.read(1 << 24), b""):
                        h.update(blk)
                w.writerow({"file": key, "sha256": h.hexdigest(), "bytes": f.stat().st_size,
                            "regenerate": meta.get("regenerate", "")})


def finite(names=None, workers=1):
    """Export (if absent), hash and check each finite-a solve end; one record per end in
    certificate_checks/solve_finite_checks.json (checkpointed: a finished end is skipped on restart)."""
    from .cert_export import CERTS, solve as export_solve
    from .verify_certificates import check_solve
    outp = CHECKS / "solve_finite_checks.json"
    CHECKS.mkdir(exist_ok=True)
    done = json.loads(outp.read_text()) if outp.exists() else {}
    for j in finite_jobs():
        if (names and j["name"] not in names) or j["name"] in done:
            continue
        t0 = time.time()
        if not (CERTS / f"{j['name']}.json").exists():
            meta = export_solve(j["a"], j["s"], j["name"])
        else:
            meta = json.loads((CERTS / f"{j['name']}.json").read_text())
        append_manifest(j["name"])
        t1 = time.time()
        res = check_solve(j["name"], verbose=False, workers=workers)
        done[j["name"]] = {"a": j["a"], "s": j["s"], "published_sign": j["published_sign"],
                           "exported_published_sign": meta.get("published_sign"), "status_exported": meta["status"],
                           "leaves": int(res.get("leaves", 0)) if "leaves" in res else None,
                           "solve_sign": res.get("solve_sign"), "E": res.get("E"), "kept_leaves": res.get("kept_leaves"),
                           "checks": res["checks"], "checker_pass": bool(res["pass"]),
                           "export_seconds": t1 - t0, "check_seconds": time.time() - t1}
        outp.write_text(json.dumps(done, indent=1, default=str))
        print(j["name"], json.dumps(done[j["name"]], default=str), flush=True)
    return done


def summary():
    """certificate_checks/track4_summary.json: what was checked, by what, the verdicts and timings, and what was not."""
    J = lambda n: json.loads((CHECKS / n).read_text()) if (CHECKS / n).exists() else None
    ann, sol, sub0 = J("outer_glob_annulus.json"), J("outer_solve.json"), J("outer_glob_annulus_sub00.json")
    lim, fin = J("solve_limit_ends.json"), J("solve_finite_checks.json")
    jobs = finite_jobs()
    out = {
        "outer_timing_sample": None if sub0 is None else {
            "A": [sub0["A_lo"], sub0["A_hi"]], "verdict": sub0["verdict"], "evals": sub0["evals"],
            "seconds": sub0["seconds"], "margin": sub0["min_margin_certified"]},
        "outer_annulus": None if ann is None else {
            k: ann[k] for k in ("subintervals", "subintervals_pass", "verdicts", "min_margin_certified",
                                "max_branch_upper", "B24_lower", "evals_total", "max_depth", "seconds_total", "pass")},
        "outer_solve": None if sol is None else {
            k: sol[k] for k in ("A_lo", "A_hi", "verdict", "min_margin_certified", "branch_upper", "evals", "max_depth",
                                "seconds", "pass")},
        "solve_limit": None if lim is None else {
            "pass": lim["pass"], "chain_links": lim["chain_links"], "seconds": lim["seconds"],
            "ends": [{k: e[k] for k in ("A", "krawczyk_unique_zero", "zero_box_in_rho_in_box", "margin", "sign",
                                          "published_sign", "sign_equals_published", "margin_overlaps_published")}
                     for e in lim["ends"]]},
        "solve_finite": {"ends_total": len(jobs), "ends_checked": 0 if fin is None else len(fin),
                         "ends_pass": 0 if fin is None else sum(bool(v["checker_pass"]) for v in fin.values()),
                         "per_end": fin or {},
                         "not_checked": [j["name"] for j in jobs if not fin or j["name"] not in fin]},
    }
    # timing: the extrapolation from the samples (one outer sub-interval, the limit ends, the a = 1.45 hi end), and the
    # actual single-process total (every run above, including the sample's own)
    smp = (fin or {}).get("solve_a1.45_hi")
    if sub0 and lim and smp:
        per_end = smp["export_seconds"] + smp["check_seconds"]
        out["timing"] = {"sample_outer_s": sub0["seconds"], "sample_limit_s": lim["seconds"],
                         "sample_finite_end_s": per_end,
                         "extrapolated_total_h": (41 * sub0["seconds"] + lim["seconds"] + len(jobs) * per_end) / 3600}
        if ann and sol and fin:
            out["timing"]["actual_total_h"] = (sub0["seconds"] + ann["seconds_total"] + sol["seconds"] + lim["seconds"]
                                               + sum(v["export_seconds"] + v["check_seconds"] for v in fin.values())) / 3600
    CHECKS.mkdir(exist_ok=True)
    (CHECKS / "track4_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    from . import verify_certificates as vc
    if cmd == "finite":
        finite(sys.argv[2:] or None)
    elif cmd == "outer_solve":
        vc.track4_outer("solve")
    elif cmd == "outer_sub":
        vc.track4_outer("annulus", sub=int(sys.argv[2]))
    elif cmd == "outer_annulus":
        vc.track4_outer("annulus", verbose=False)
    elif cmd == "solve_limit":
        vc.track4_solve_limit()
    elif cmd == "summary":
        summary()
