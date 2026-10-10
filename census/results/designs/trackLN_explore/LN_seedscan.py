"""EXPLORATORY seed-range check for Track LN (Track L + LayerNorm without affine before the readout; NOT a
registration; no seed is drawn by this script).  Candidate range: 2,995,000-2,995,099 (exploration seeds for
LN_explore.py; never to be registered).  Method of ../trackL_alpha_explore/A_seedscan.py, plus a JSONL check:
  1. regex scan (2A-PS's scan_tree; digits, '_' and ',' spellings) of every text file under src, tests, results,
     paper, notes, independent, data, dist (this folder skipped: it names the range);
  2. every integer range literal `range(a, b)` / `range(a)` in src/ and tests/ for overlap;
  3. every column whose name contains 'seed' in every parquet file under the same directories;
  4. every top-level key containing 'seed' in every line of every .jsonl file under the same directories.
Also checked: disjoint from 2,991,000-099 (L2 exploration), 2,992,000-039 (Track L registered), 2,993,000-009
(Track L pilot, unused), 2,994,000-007 (alpha exploration; its whole range 2,994,000-099 is checked).
Usage (from census/): python results/designs/trackLN_explore/LN_seedscan.py"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import phase2a_ps as PS

CANDS = {"LN_exploration": (2_995_000, 2_995_100)}
OTHER = {"L2_exploration": (2_991_000, 2_991_100), "trackL_registered": (2_992_000, 2_992_040),
         "trackL_pilot": (2_993_000, 2_993_010), "alpha_exploration": (2_994_000, 2_994_100)}
PAT = {"LN_exploration": r"(^|[^0-9.])29950[0-9][0-9]([^0-9]|$)|(^|[^0-9])2_995_0[0-9][0-9]([^0-9]|$)"
                         r"|(^|[^0-9.,])2,995,0[0-9][0-9]([^0-9,]|$)"}
SELF_DIR = "trackLN_explore"


def inrange(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and \
        any(c0 <= x < c1 for c0, c1 in CANDS.values())


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
                    n = sum(1 for x in t.column(c).to_pylist() if inrange(x))
                    pq.append([str(p.relative_to(ROOT)), c, n])
except ImportError as e:
    pq = f"pyarrow unavailable: {e}"
out["parquet_seed_columns"] = pq

jl, n_jsonl = [], 0
for d in PS.SCAN_DIRS:
    for p in (ROOT / d).rglob("*.jsonl"):
        if SELF_DIR in str(p):
            continue
        n_jsonl += 1
        n_keys, n_hit = 0, 0
        for line in p.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                for k, v in row.items():
                    if "seed" in k.lower():
                        n_keys += 1
                        vals = v if isinstance(v, list) else [v]
                        n_hit += sum(1 for x in vals if inrange(x))
        if n_keys:
            jl.append([str(p.relative_to(ROOT)), n_keys, n_hit])
out["jsonl_files_checked"] = n_jsonl
out["jsonl_seed_keys"] = jl
out["clean"] = (out["disjoint"] and not any(out["text_hits"].values()) and not over
                and (isinstance(pq, list) and all(n == 0 for _, _, n in pq)) and all(n == 0 for _, _, n in jl))
(HERE / "LN_seedscan.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("disjoint", "text_hits", "range_literals_checked", "range_overlaps",
                                      "jsonl_files_checked", "clean")}))
print("parquet seed columns:", len(pq) if isinstance(pq, list) else pq,
      "jsonl files with seed keys:", len(jl))
