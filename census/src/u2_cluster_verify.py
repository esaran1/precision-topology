"""Track U, U2: LOCAL verify-back of the cluster run's returned results.

Checks, against the committed results/uniform_eps/u2_cluster/MANIFEST.sha256 and jobs.json:
  * every job (1,080 units + every switch sub-slab) has exactly one result file, written against this manifest
    (its manifest_sha256 equals the sha256 of the committed MANIFEST);
  * every unit passed all three links (outer, ring, PD) and every switch sub-slab passed;
  * the VM's reference phase matched the committed reference verdicts (REFERENCE_MATCH.json, same manifest);
and rebuilds the summary (counts, CPU seconds, peak memory) into results_full/SUMMARY.json.  Reads JSON only; runs
no checker code.

    python -m src.u2_cluster_verify [<results_full dir>] [<reference dir>]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results" / "uniform_eps" / "u2_cluster"


def main(argv):
    res_dir = Path(argv[0]) if argv else PKG / "results_full"
    ref_dir = Path(argv[1]) if len(argv) > 1 else PKG / "results_reference_linux"
    msha = hashlib.sha256((PKG / "MANIFEST.sha256").read_bytes()).hexdigest()
    jobs = json.loads((PKG / "jobs.json").read_text())
    problems, cpu, rss, n_pass = [], 0.0, 0.0, {"unit": 0, "switch": 0}
    for j in jobs:
        f = res_dir / f"{j['id']}.json"
        if not f.exists():
            problems.append(f"missing {j['id']}"); continue
        r = json.loads(f.read_text())
        if r.get("manifest_sha256") != msha:
            problems.append(f"{j['id']}: manifest {r.get('manifest_sha256')} != {msha}"); continue
        if not r.get("pass"):
            problems.append(f"{j['id']}: NOT PASSED"); continue
        if j["kind"] == "unit" and not (r["outer"]["pass"] and r["ring"]["certified"] and r["pd"]["pd"]):
            problems.append(f"{j['id']}: inconsistent link verdicts"); continue
        n_pass[j["kind"]] += 1
        cpu += r.get("seconds", 0.0)
        rss = max(rss, r.get("peak_rss_gb", 0.0))
    extra = sorted(set(p.stem for p in res_dir.glob("*.json")) - {j["id"] for j in jobs} - {"SUMMARY"})
    ref_ok = False
    rm = ref_dir / "REFERENCE_MATCH.json"
    if rm.exists():
        ref_ok = json.loads(rm.read_text()).get("manifest_sha256") == msha
    else:
        problems.append("no REFERENCE_MATCH.json from the VM's reference phase")
    summary = {"manifest_sha256": msha, "jobs": len(jobs), "units_passed": n_pass["unit"],
               "switch_subslabs_passed": n_pass["switch"], "reference_match": ref_ok,
               "unexpected_result_files": extra, "cpu_hours": cpu / 3600, "peak_rss_gb": rss,
               "problems": problems[:50], "n_problems": len(problems),
               "verdict": "ALL PASSED" if (not problems and ref_ok and not extra) else "NOT VERIFIED"}
    (res_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
    return 0 if summary["verdict"] == "ALL PASSED" else 1


if __name__ == "__main__":
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(" ".join(["u2_cluster_verify"] + sys.argv[1:]))
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        sys.exit(3)
    sys.exit(main(sys.argv[1:]))
