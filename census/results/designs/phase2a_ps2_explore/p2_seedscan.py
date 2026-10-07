"""EXPLORATORY seed-range check for the 2A-PS2 design page (no seed is drawn).  Candidate ranges:
registered 2,987,000-2,987,599 and pilot 2,988,000-2,988,059.  Checks:
  1. a regex scan (2A-PS's scan_tree, unchanged; digits, '_' and ',' spellings) of every text file under src, tests,
     results, paper, notes, independent, data, dist;
  2. every integer range literal `range(a, b)` / `range(a)` in src/ and tests/ (the SEEDS*/PILOT_SEEDS* constants of
     every module, written with or without '_') for overlap with the candidates;
  3. every column whose name contains 'seed' in every parquet file under the same directories.
Usage (from census/): python results/designs/phase2a_ps2_explore/p2_seedscan.py"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import phase2a_ps as PS

REG = (2_987_000, 2_987_600)
PIL = (2_988_000, 2_988_060)
PAT = {"registered": r"(^|[^0-9.])2987[0-5][0-9][0-9]([^0-9]|$)|(^|[^0-9])2_987_[0-5][0-9][0-9]([^0-9]|$)"
                     r"|(^|[^0-9.,])2,987,[0-5][0-9][0-9]([^0-9,]|$)",
       "pilot": r"(^|[^0-9.])29880[0-5][0-9]([^0-9]|$)|(^|[^0-9])2_988_0[0-5][0-9]([^0-9]|$)"
                r"|(^|[^0-9.,])2,988,0[0-5][0-9]([^0-9,]|$)"}
SELF = ("p2_seedscan.py", "p2_seedscan.json", "phase2a_ps2_design.md", "README.md")

out = {"candidates": {"registered": REG, "pilot": PIL}}
hits = PS.scan_tree(PAT, PS.SCAN_DIRS)
out["text_hits"] = {k: [h for h in v if not h.endswith(SELF)] for k, v in hits.items()}

rx = re.compile(r"range\(\s*([0-9_]+)\s*(?:,\s*([0-9_]+)\s*)?[,)]")
over = []
n_ranges = 0
for d in ("src", "tests"):
    for p in (ROOT / d).rglob("*.py"):
        for m in rx.finditer(p.read_text(errors="ignore")):
            a, b = m.group(1), m.group(2)
            lo, hi = (0, int(a)) if b is None else (int(a), int(b))
            n_ranges += 1
            for name, (c0, c1) in (("registered", REG), ("pilot", PIL)):
                if lo < c1 and c0 < hi:
                    over.append([str(p.relative_to(ROOT)), m.group(0), name])
out["range_literals_checked"] = n_ranges
out["range_overlaps"] = over
out["seed_constant_modules"] = sorted({str(p.relative_to(ROOT)) for d in ("src", "tests") for p in (ROOT / d).rglob("*.py")
                                       if re.search(r"^\s*[A-Z_]*SEEDS[A-Z_0-9]*\s*=", p.read_text(errors="ignore"), re.M)})

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
                            and (REG[0] <= x < REG[1] or PIL[0] <= x < PIL[1]))
                    pq.append([str(p.relative_to(ROOT)), c, n])
except ImportError as e:
    pq = f"pyarrow unavailable: {e}"
out["parquet_seed_columns"] = pq
out["clean"] = (not any(out["text_hits"].values()) and not over
                and (isinstance(pq, list) and all(n == 0 for _, _, n in pq)))
(HERE / "p2_seedscan.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("text_hits", "range_literals_checked", "range_overlaps", "clean")}))
print("modules with SEEDS constants:", len(out["seed_constant_modules"]), "parquet seed columns:",
      len(pq) if isinstance(pq, list) else pq)
