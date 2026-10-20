"""Track U, U2: package the cluster run (local; gated entry point).

  jobs    -> results/uniform_eps/u2_cluster/jobs.json: one job per run1 unit (1,080; its slab ε-range, A-sub-interval,
             branch centre, regime, and its leaf files for the replay fallback) and one per switch sub-slab (from run1's
             switch exports).  All inputs are PRODUCER-ONLY ("not independently checked").
  manifest -> results/uniform_eps/u2_cluster/MANIFEST.sha256 (sha256 of every input export, the checker source, the
             run script, the verify-back script, the gate/store helpers, the requirements pins, jobs.json and the
             reference verdicts) and BUNDLE_FILES.txt (the inputs that are not in git, to transfer with the checkout).

    python -m src.u2_cluster_pack jobs
    python -m src.u2_cluster_pack manifest
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "results" / "uniform_eps" / "u2_sample"
PKG = ROOT / "results" / "uniform_eps" / "u2_cluster"
CODE = ["src/__init__.py", "src/verify_certificates.py", "src/u2_cluster_run.py", "src/u2_cluster_verify.py",
        "src/u2_gate.py", "src/u2_store.py", "results/uniform_eps/u2_cluster/requirements-u2-cluster.txt"]
DATA = ["results/certificates/limit_A_lo.npz", "results/certificates/limit_A_lo.json",
        "results/certificates_manifest.csv"]


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def build_jobs():
    jobs = []
    units = SAMPLE / "units" / "run1"
    for s in range(27):
        for k in range(40):
            r = json.loads((units / f"s{s:03d}_k{k:03d}.json").read_text())
            leaves = []
            whole = SAMPLE / f"run1_s{s:02d}_k{k:03d}.npz"
            if whole.exists() and "outer_split" not in r:
                leaves.append({"npz": rel(whole), "eps": r["eps"]})
            j = 0
            while (SAMPLE / f"run1_s{s:02d}_k{k:03d}_e{j}.npz").exists():
                m = json.loads((SAMPLE / f"run1_s{s:02d}_k{k:03d}_e{j}.meta.json").read_text())
                leaves.append({"npz": rel(SAMPLE / f"run1_s{s:02d}_k{k:03d}_e{j}.npz"), "eps": m["eps"]})
                j += 1
            jobs.append({"id": f"s{s:03d}_k{k:03d}", "kind": "unit", "eps": r["eps"], "A": r["A"], "centre": r["centre"],
                         "regime": r["regime"], "leaves": leaves,
                         "producer_status": "PRODUCER-ONLY: not independently checked"})
    for s in range(27):
        f = SAMPLE / f"run1_s{s:02d}_switch.json"
        data = json.loads(f.read_text())
        L = data["sub_slabs"] if isinstance(data, dict) else data
        for i in range(len(L)):
            jobs.append({"id": f"sw_s{s:03d}_{i:02d}", "kind": "switch", "file": rel(f), "index": i, "eps": L[i]["eps"]})
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "jobs.json").write_text(json.dumps(jobs, indent=0))
    n_u = sum(j["kind"] == "unit" for j in jobs)
    print(f"{n_u} units ({sum(1 for j in jobs if j['kind'] == 'unit' and not j['leaves'])} without leaves), "
          f"{len(jobs) - n_u} switch sub-slabs", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def build_manifest():
    jobs = json.loads((PKG / "jobs.json").read_text())
    inputs = set()
    for j in jobs:
        for lf in j.get("leaves", []):
            inputs.add(lf["npz"])
        if j["kind"] == "switch":
            inputs.add(j["file"])
    files = CODE + DATA + ["results/uniform_eps/u2_cluster/jobs.json"]
    if (PKG / "reference_verdicts.json").exists():
        files.append("results/uniform_eps/u2_cluster/reference_verdicts.json")
    files += sorted(inputs)
    lines = ["# sha256  path (relative to census/); the run script refuses on any mismatch"]
    for f in files:
        lines.append(f"{sha256(ROOT / f)}  {f}")
    (PKG / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    bundle = [f for f in files if f.startswith("results/uniform_eps/u2_sample/") or f.startswith("results/certificates/")]
    (PKG / "BUNDLE_FILES.txt").write_text("\n".join(bundle) + "\n")
    tot = sum((ROOT / f).stat().st_size for f in bundle)
    print(f"{len(files)} files in the manifest; {len(bundle)} to transfer ({tot / 2 ** 20:.1f} MB); "
          f"manifest sha256 {sha256(PKG / 'MANIFEST.sha256')}", flush=True)


if __name__ == "__main__":
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(" ".join(["u2_cluster_pack"] + sys.argv[1:]))
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        sys.exit(3)
    {"jobs": build_jobs, "manifest": build_manifest}[sys.argv[1]]()
