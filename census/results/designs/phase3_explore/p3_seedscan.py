"""EXPLORATORY seed-range check for the Phase 3 exploration own samples 7,410,000-7,410,099 (prefix 7410): no 7-digit
number with this prefix (also written 7_410_xxx or 7,410,xxx) in any text file under src/, tests/, results/, paper/
(src.phase1c._scan_tree, the 1C scan; this directory's own files excepted), and no overlap with the registered or pilot
seed constants (SEEDS*, PILOT_SEEDS*) of the test modules."""
import importlib, json, sys
sys.path.insert(0, "/Users/Evan/precision-topology/census")
from src import phase1c as P
P.SCAN_SKIP_FILES = P.SCAN_SKIP_FILES + ("p3_seedscan.py", "p3_seedscan.log", "README.md") + tuple(
    f for f in __import__("os").listdir(".") if f.startswith("p3_"))
p = "7410"
pat = {"7410": rf"(^|[^0-9.]){p}[0-9]{{3}}([^0-9]|$)|{p[0]}_{p[1:4]}_[0-9]{{3}}|(^|[^0-9.,]){p[0]},{p[1:4]},[0-9]{{3}}([^0-9,]|$)"}
hits = P._scan_tree(pat)
hits["7410"] = [h for h in hits["7410"] if "phase3_explore" not in h]
mine = set(range(7_410_000, 7_410_100))
ov = {}
for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1c",
            "phase2a", "phase2a_ps", "phase2b"):
    try:
        m = importlib.import_module(f"src.{mod}")
    except Exception as e:
        ov[mod] = f"import failed: {e}"
        continue
    for name in dir(m):
        if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
            ov[f"{mod}.{name}"] = sorted(mine & set(P._flat_ints(getattr(m, name))))
print(json.dumps({"pattern_files": hits, "registered_overlap": {k: v for k, v in ov.items() if v}}, indent=1))
print("UNUSED" if not hits["7410"] and not any(v for v in ov.values() if isinstance(v, list) and v) else "USED")
