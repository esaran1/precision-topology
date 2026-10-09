"""EXPLORATORY seed-range check for Track L (no seed is drawn by this script).  Candidate ranges:
exploration 2,991,000-2,991,099 (used by L_explore.py; never to be registered), proposed registered
2,992,000-2,992,099 and proposed pilot 2,993,000-2,993,019.  Checks (2A-PS2's p2_seedscan.py method, unchanged):
  1. regex scan (2A-PS's scan_tree; digits, '_' and ',' spellings) of every text file under src, tests, results,
     paper, notes, independent, data, dist (this folder and the design page skipped: they name the ranges);
  2. every integer range literal `range(a, b)` / `range(a)` in src/ and tests/ for overlap;
  3. every column whose name contains 'seed' in every parquet file under the same directories.
Usage (from census/): python results/designs/trackL_explore/L_seedscan.py"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import phase2a_ps as PS

CANDS = {"exploration": (2_991_000, 2_991_100), "registered": (2_992_000, 2_992_100), "pilot": (2_993_000, 2_993_020)}
PAT = {"exploration": r"(^|[^0-9.])29910[0-9][0-9]([^0-9]|$)|(^|[^0-9])2_991_0[0-9][0-9]([^0-9]|$)"
                      r"|(^|[^0-9.,])2,991,0[0-9][0-9]([^0-9,]|$)",
       "registered": r"(^|[^0-9.])29920[0-9][0-9]([^0-9]|$)|(^|[^0-9])2_992_0[0-9][0-9]([^0-9]|$)"
                     r"|(^|[^0-9.,])2,992,0[0-9][0-9]([^0-9,]|$)",
       "pilot": r"(^|[^0-9.])29930[01][0-9]([^0-9]|$)|(^|[^0-9])2_993_0[01][0-9]([^0-9]|$)"
                r"|(^|[^0-9.,])2,993,0[01][0-9]([^0-9,]|$)"}
SELF_DIR = "trackL_explore"
SELF_FILES = ("trackL_dominoes_design.md",)

out = {"candidates": CANDS}
hits = PS.scan_tree(PAT, PS.SCAN_DIRS)
out["text_hits"] = {k: [h for h in v if SELF_DIR not in h and not h.endswith(SELF_FILES)] for k, v in hits.items()}

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
out["clean"] = (not any(out["text_hits"].values()) and not over
                and (isinstance(pq, list) and all(n == 0 for _, _, n in pq)))
(HERE / "L_seedscan.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("text_hits", "range_literals_checked", "range_overlaps", "clean")}))
