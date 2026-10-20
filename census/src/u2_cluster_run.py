"""Track U, U2: the CLUSTER RUN of the independent checker over run1's exports (PRODUCER-ONLY inputs, labelled
"not independently checked" until this run passes).

One script, in this order:
  1. verifies EVERY file listed in results/uniform_eps/u2_cluster/MANIFEST.sha256 (refuses on any mismatch or missing
     file), and the pinned versions of python-flint and numpy;
  2. refuses to start without a fresh, separate, earlier gate entry (python -m src.u2_gate gate ..., its own step);
  3. phase "reference": runs the 5 reference units and compares their verdicts EXACTLY with the committed local ones
     (reference_verdicts.json); any difference stops the run (exit 4) and the full phase refuses afterwards;
     phase "full": refuses unless the reference phase passed against this manifest; runs every unit (outer exclusion
     by re-search with a time cap and ε-splitting, the replay of the producer's leaves -- geometry only -- as fallback;
     ring, fast test first; PD) and every switch sub-slab (sheared Krawczyk replay with the checker's own radii and
     ε-split), in a pool of --workers processes;
  4. writes ONE result file per job atomically (u2_store) and resumes by skipping finished jobs.
The checker (src/verify_certificates.py) imports nothing from the producer.

    python -m src.u2_cluster_run --phase reference --workers 5 [--make-reference]   # local: --make-reference
    python -m src.u2_cluster_run --phase full --workers 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results" / "uniform_eps" / "u2_cluster"
MANIFEST = PKG / "MANIFEST.sha256"
PINS = {"python-flint": "0.9.0", "numpy": "2.4.6"}
REFERENCE_IDS = ["s000_k020", "s006_k020", "s013_k020", "s025_k020", "s026_k039"]
OUTER_CAP = float(os.environ.get("U2_OUTER_CAP", "3600"))
FALLBACK_CAP = float(os.environ.get("U2_FALLBACK_CAP", "3600"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def verify_manifest():
    """Every listed file exists and hashes as listed; returns (manifest sha256, {path: sha})."""
    if not MANIFEST.exists():
        raise SystemExit("REFUSED: no MANIFEST.sha256")
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        entries[p.strip()] = h
    bad = [p for p, h in entries.items() if not (ROOT / p).exists() or sha256(ROOT / p) != h]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} manifest mismatch(es), first: {bad[:5]}")
    import numpy
    import flint
    vers = {"python-flint": getattr(flint, "__version__", "?"), "numpy": numpy.__version__}
    if vers != PINS:
        raise SystemExit(f"REFUSED: versions {vers} != pinned {PINS}")
    return sha256(MANIFEST), entries


_W = {}


def _init(manifest_sha):
    from . import verify_certificates as vc
    x, y, sym = vc._verified_limit_data()
    assert sym
    Bt = [(P, vc.localisation_bounds(x, y, P)[0]) for P in (18.0, 20.0, 22.0, 24.0)]
    Bf = vc.localisation_bounds(x, y, 24.0)[1]
    _W.update(vc=vc, x=x, y=y, Bt=Bt, Bf=Bf, manifest=manifest_sha)


def _outer(vc, u, lo, hi, depth=0):
    """Re-search on [lo, hi] with a time cap; on 'unresolved' split ε (to depth 3); verdicts per closed piece."""
    x, y = _W["x"], _W["y"]
    R = u["regime"] == "R"
    r = vc.u2_outer_arb(x, y, u["centre"], u["A"][0], u["A"][1], lo, hi, u["regime"], B_table=_W["Bt"] if R else None,
                        B_full=_W["Bf"] if R else None, max_seconds=OUTER_CAP, max_evals=10 ** 12)
    piece = {"eps": [lo, hi], "verdict": r["verdict"], "evals": r["evals"], "seconds": r["seconds"],
             "route": "re-search", "witness": r.get("witness")}
    if r["verdict"] != "unresolved" or depth >= 3:
        return [piece]
    m = 0.5 * (lo + hi)
    return _outer(vc, u, lo, m, depth + 1) + _outer(vc, u, m, hi, depth + 1)


def run_job(job):
    vc = _W["vc"]
    t0 = time.time()
    out = {"id": job["id"], "kind": job["kind"], "manifest_sha256": _W["manifest"],
           "check_status": "independent checker (Arb) result"}
    if job["kind"] == "unit":
        u = job
        pieces = _outer(vc, u, u["eps"][0], u["eps"][1])
        if any(p_["verdict"] == "unresolved" for p_ in pieces):          # fallback: replay of the producer's leaves
            import numpy as np
            for lf in u.get("leaves", []):
                L = np.load(ROOT / lf["npz"])
                hdr = {"eps": lf["eps"], "A": u["A"], "centre": u["centre"], "regime": u["regime"]}
                rr = vc.u2_outer_replay(_W["x"], _W["y"], hdr, {k: L[k] for k in ("p", "q", "hp", "hq")},
                                        B_table=_W["Bt"] if u["regime"] == "R" else None,
                                        B_full=_W["Bf"] if u["regime"] == "R" else None, fallback_seconds=FALLBACK_CAP)
                pieces.append({"eps": lf["eps"], "verdict": rr["verdict"], "route": "replay", "seconds": rr["seconds"]})
            unres = [p_ for p_ in pieces if p_["route"] == "re-search" and p_["verdict"] == "unresolved"]
            covered = all(any(p_["route"] == "replay" and p_["verdict"] == "pass" and p_["eps"][0] <= q_["eps"][0]
                              and q_["eps"][1] <= p_["eps"][1] for p_ in pieces) for q_ in unres)
            outer_pass = covered and not any(p_["verdict"] == "fail" for p_ in pieces)
        else:
            outer_pass = all(p_["verdict"] == "pass" for p_ in pieces)
        ring = vc.u2_ring_fast_arb(_W["x"], _W["y"], u["centre"], u["A"][0], u["A"][1], u["eps"][0], u["eps"][1])
        pd = vc.u2_pd_arb(_W["x"], _W["y"], u["centre"], u["A"][0], u["A"][1], u["eps"][0], u["eps"][1])
        out.update(outer={"pass": bool(outer_pass), "pieces": pieces},
                   ring={k: ring[k] for k in ("certified", "tiles", "fast", "fallback", "left", "seconds")},
                   pd={k: pd[k] for k in ("pd", "boxes", "seconds")})
        out["pass"] = bool(outer_pass and ring["certified"] and pd["pd"])
    else:
        data = json.loads((ROOT / job["file"]).read_text())
        L = data["sub_slabs"] if isinstance(data, dict) else data
        S = L[job["index"]]
        r = vc.u2_switch_shear_split(_W["x"], _W["y"], S["eps"][0], S["eps"][1], S["Z"], S["rows"], S["J_tube"])
        out.update(switch={"leaves": r["leaves"]}, eps=S["eps"])
        out["pass"] = bool(r["pass"])
    out["seconds"] = time.time() - t0
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin"
                                                                               else 2 ** 20)
    return out


def _work(job):
    from .u2_store import UnitStore
    st = UnitStore(Path(_W["results"]))
    prev = st.get(job["id"])
    if prev is not None and prev.get("manifest_sha256") == _W["manifest"]:
        return job["id"], prev["pass"], True              # resume: finished against this same manifest
    r = run_job(job)
    st.put(job["id"], r)
    return job["id"], r["pass"], False


def _init_pool(manifest_sha, results):
    _init(manifest_sha)
    _W["results"] = results


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reference", "full"], required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--make-reference", action="store_true", help="local only: write reference_verdicts.json")
    a = ap.parse_args(argv)
    msha, entries = verify_manifest() if not a.make_reference else (None, {})
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(f"u2_cluster_run {a.phase} workers={a.workers}")
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 3
    jobs = json.loads((PKG / "jobs.json").read_text())
    # the reference phase always computes afresh on this platform (local results are never reused on the VM)
    results = PKG / (f"results_reference_{sys.platform}" if a.phase == "reference" else "results_full")
    results.mkdir(parents=True, exist_ok=True)
    if a.phase == "full":
        ok_file = PKG / f"results_reference_{sys.platform}" / "REFERENCE_MATCH.json"
        if not ok_file.exists() or json.loads(ok_file.read_text()).get("manifest_sha256") != msha:
            print("REFUSED: the reference phase has not passed against this manifest", flush=True)
            return 5
        todo = jobs
    else:
        todo = [j for j in jobs if j["id"] in REFERENCE_IDS]
    from multiprocessing import get_context
    print(f"{len(todo)} jobs, {a.workers} workers, manifest {msha}", flush=True)
    with get_context("spawn").Pool(a.workers, initializer=_init_pool, initargs=(msha, str(results))) as pool:
        for jid, ok, skipped in pool.imap_unordered(_work, todo):
            print(jid, "PASS" if ok else "NOT PASSED", "(resumed)" if skipped else "", flush=True)
    from .u2_store import UnitStore
    st = UnitStore(results)
    verdicts = {j["id"]: st.get(j["id"])["pass"] for j in todo}
    if a.phase == "reference":
        ref = PKG / "reference_verdicts.json"
        if a.make_reference:
            ref.write_text(json.dumps({"verdicts": verdicts, "platform": sys.platform,
                                       "versions": PINS}, indent=1, sort_keys=True))
            print("reference verdicts written", verdicts, flush=True)
            return 0
        want = json.loads(ref.read_text())["verdicts"]
        if verdicts != want:
            print(f"STOP: reference verdicts differ: local {want} vs here {verdicts}", flush=True)
            return 4
        (results / "REFERENCE_MATCH.json").write_text(json.dumps({"manifest_sha256": msha, "verdicts": verdicts}))
        print("reference verdicts match exactly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
