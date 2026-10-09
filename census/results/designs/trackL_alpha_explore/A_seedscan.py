"""EXPLORATORY seed-range check (Track L, output-multiplier exploration, MOTIVATED BY THE TRACK L NULL; NOT a
registration; no seed is drawn by this script).  Candidate range: 2,994,000-2,994,099 (fresh exploration seeds for
A_explore.py; never to be registered).  Same method as ../trackL_explore/L_seedscan.py (2A-PS2's method, unchanged):
  1. regex scan (2A-PS's scan_tree; digits, '_' and ',' spellings) of every text file under src, tests, results,
     paper, notes, independent, data, dist (this folder skipped: it names the range);
  2. every integer range literal `range(a, b)` / `range(a)` in src/ and tests/ for overlap;
  3. every column whose name contains 'seed' in every parquet file under the same directories.
The range is also checked to be disjoint from the L2 exploration (2,991,000-099), the proposed registered
(2,992,000-039) and the proposed pilot (2,993,000-019) ranges.
Usage (from census/): python results/designs/trackL_alpha_explore/A_seedscan.py"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import phase2a_ps as PS

CANDS = {"alpha_exploration": (2_994_000, 2_994_100)}
OTHER = {"L2_exploration": (2_991_000, 2_991_100), "registered": (2_992_000, 2_992_040), "pilot": (2_993_000, 2_993_020)}
PAT = {"alpha_exploration": r"(^|[^0-9.])29940[0-9][0-9]([^0-9]|$)|(^|[^0-9])2_994_0[0-9][0-9]([^0-9]|$)"
                            r"|(^|[^0-9.,])2,994,0[0-9][0-9]([^0-9,]|$)"}
SELF_DIR = "trackL_alpha_explore"

out = {"candidates": CANDS, "disjoint_from": OTHER,
       "disjoint": all(not (a0 < b1 and b0 < a1) for a0, a1 in CANDS.values() for b0, b1 in OTHER.values())}
hits = PS.scan_tree(PAT, PS.SCAN_DIRS)
out["text_hits"] = {k: [h for h in v if SELF_DIR not in h] for k, v in hits.items()}

rx = re.compile(r"range\(\s*([0-9_]+)\s*(?:,\s*([0-9_]+)\s*)?[,)]")
over, n_ranges = [], 0
for d in ("src", "tests"):
    for p in (ROOT / d).rglob("*.py"):
        for m in rx.finditer(p.read_text(errors="ignore")):
            a, b = m.group(1), m.group(2)
            lo, hi = (0, int(a)) if b is None else (int(a), int(b))
            n_ranges += 1
            for name, (c0, c1) in CANDS.items():
                if lo < c1 and c0 < hi:
                    over.append([str(p.relative_to(ROOT)), m.group(0), name])
out["range_literals_checked"] = n_ranges
out["range_overlaps"] = over

pq = []
try:
    import pyarrow.parquet as pqm
    for d in PS.SCAN_DIRS:
        for p in (ROOT / d).rglob("*.parquet"):
            t = pqm.read_table(p)
            for c in t.column_names:
                if "seed" in c.lower():
                    col = t.column(c).to_pylist()
                    n = sum(1 for x in col if isinstance(x, (int, float)) and x == x
                            and any(c0 <= x < c1 for c0, c1 in CANDS.values()))
                    pq.append([str(p.relative_to(ROOT)), c, n])
except ImportError as e:
    pq = f"pyarrow unavailable: {e}"
out["parquet_seed_columns"] = pq
out["clean"] = (out["disjoint"] and not any(out["text_hits"].values()) and not over
                and (isinstance(pq, list) and all(n == 0 for _, _, n in pq)))
(HERE / "A_seedscan.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("disjoint", "text_hits", "range_literals_checked", "range_overlaps", "clean")}))
