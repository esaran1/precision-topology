"""Phase 3 reproducibility check (POST HOC, not registered; after the worker stall of 2026-10-06).

The Q `run` stalled at 14:36 with 13 per-seed rows written; it was restarted at 15:28:59 with the registered command,
which resumes per seed (a seed with a written row is skipped; 7,430,013, in progress at the stall, had no row and was
retrained from scratch; there is no in-seed checkpoint).  This check recomputes three pre-stall Q seeds with the
registered `phase3_w4.run_one` and compares each with its committed row in runs_Q.jsonl, timing fields excluded.

    python -m src.phase3_repro_check        -> results/phase3/repro_check.json
"""

from __future__ import annotations

import json
import time

from . import phase3_w4 as P

SEEDS = (7_430_009, 7_430_005, 7_430_012)
OUT_FILE = P.OUT / "repro_check.json"
TIMING_KEYS = ("rss", "peak_rss", "wall")


def strip(o):
    """Drop timing fields (keys starting with 'secs', and rss/peak_rss/wall) recursively."""
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not (k.startswith("secs") or k in TIMING_KEYS)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def differing_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def main():
    P._setup()
    P.memory_gate("repro check")
    fr, rows = P._frozen_by_seed("Q")
    land = P._land()
    rho = fr["rho"]["Q"]
    committed = {r["seed"]: r for r in P._rows(P._runs_file("Q"))}
    res = {}
    for seed in SEEDS:
        t0 = time.time()
        r = json.loads(json.dumps(P._jsonable(P.run_one(rows[seed], rho, land))))
        a, b = strip(r), strip(committed[seed])
        res[str(seed)] = {"identical": a == b, "differing_keys": differing_keys(a, b),
                          "secs": round(time.time() - t0, 1)}
        print(seed, res[str(seed)], flush=True)
    out = {"label": "POST HOC reproducibility check (not registered): pre-stall Q seeds recomputed with the registered "
                    "run_one vs their committed runs_Q.jsonl rows, timing fields excluded",
           "runs_Q_sha256": P._sha(P._runs_file("Q")), "seeds": res,
           "all_identical": all(v["identical"] for v in res.values())}
    OUT_FILE.write_text(json.dumps(out, indent=1))
    print("ALL_IDENTICAL" if out["all_identical"] else "SOME_DIFFER")


if __name__ == "__main__":
    main()
