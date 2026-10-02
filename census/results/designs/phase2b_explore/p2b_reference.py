"""EXPLORATORY reference: ρ₂ and G₊ of each committed branch (results/sb_fold/branch_points.csv, stable points, linear
interpolation in s) at the matched scales, and the global (lowest-loss) branch there (switches L0→M 1.48955, M→S 3.58906,
S→S2 8.61658, from results/sb_fold_report.md).  python p2b_reference.py  (writes p2b_reference.json)"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2b_explore as E  # noqa: E402
rows = list(csv.DictReader(open(E.ROOT / "results/sb_fold/branch_points.csv")))
out = {}
for nm, s in E.SCALES.items():
    d = {}
    for br in ("L0", "M", "S", "S2"):
        P = sorted((float(r["s"]), float(r["rho2"]), float(r["gplus"]), float(r["loss"])) for r in rows
                   if r["branch"] == br and r["stable"] == "True")
        ss = np.array([p[0] for p in P])
        if len(ss) and ss.min() <= s <= ss.max():
            d[br] = {k: float(np.interp(s, ss, [p[i] for p in P])) for i, k in ((1, "rho2"), (2, "gplus"), (3, "loss"))}
    glob = min(d, key=lambda b: d[b]["loss"])
    out[nm] = {"s": s, "branches": d, "global": glob}
    print(nm, round(s, 4), "global", glob, {b: round(v["rho2"], 3) for b, v in d.items()})
(HERE / "p2b_reference.json").write_text(json.dumps(out, indent=1))
