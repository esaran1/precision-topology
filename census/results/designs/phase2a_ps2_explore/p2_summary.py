"""EXPLORATORY summary of p2_explore.jsonl (arithmetic only) and of the committed 2A-PS observations (observed.jsonl;
end-of-observation step / t_F per class; no recomputation).  Usage (from census/):
python results/designs/phase2a_ps2_explore/p2_summary.py"""
import collections, json, math
from pathlib import Path
import numpy as np


class binom:
    @staticmethod
    def sf(k, n, p):
        """P(X > k), X ~ Bin(n, p) (exact sum; no scipy)."""
        return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(int(k) + 1, n + 1)))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = [json.loads(l) for l in open(HERE / "p2_explore.jsonl")]
print("seeds", len(R), R[0]["seed"], "-", R[-1]["seed"])
st = collections.Counter(("untraceable" if r["status"].startswith("untraceable") else r["status"]) for r in R)
print("status", dict(st))
sc = [r for r in R if r["status"] == "scoreable"]
print("scoreable registered class", dict(collections.Counter(r["class_registered"] for r in sc)))
# (a) activity
rel = [r for r in R if r.get("n_reg_active") is not None]
drops = [r["seed"] for r in rel if r["activity_drops_unit"]]
ms = sorted(r["min_share"] for r in rel)
print(f"(a) releases {len(rel)}; theta drops a unit on {len(drops)} {drops}; smallest live share {ms[:4]}")
print("    3 units active (registered rule) on", sum(r["n_reg_active"] == 3 for r in rel))
# (b) budget
cal = [r for r in sc if r.get("calib")]
rts = np.array([r["rho_t_star"] for r in sc if r.get("rho_t_star") is not None and math.isfinite(r["rho_t_star"])])
print(f"(b) rho*t* (s0 -> s_F) n {len(rts)}: min {rts.min():.2f} med {np.median(rts):.2f} q90 {np.quantile(rts, .9):.2f} "
      f"max {rts.max():.2f}; stall (inf/None) {sum(1 for r in sc if r.get('rho_t_star') in (None,) or not math.isfinite(r['rho_t_star']))}")
ratF = np.array([r["calib"]["rt_F"] / r["rho_t_star"] for r in cal if r["calib"]["rt_F"]])
print(f"    2^-12 calibration: t_F/t* n {len(ratF)} min {ratF.min():.4f} med {np.median(ratF):.4f} max {ratF.max():.4f}")
for cls in ("clean", "none", "mixed"):
    e = [r["calib"]["rt_end"] / r["rho_t_star"] for r in cal if r["class_registered"] == cls]
    x = [r for r in cal if r["class_registered"] == cls]
    if e:
        print(f"    {cls}: n {len(e)} end/t* min {min(e):.3f} med {np.median(e):.3f} max {max(e):.3f}; crossed "
              f"{sum(r['calib']['crossed'] for r in x)}; reached 1.25 {sum(r['calib']['reached_1.25'] for r in x)}; "
              f"sign change {sum(r['calib']['sign_change_t'] is not None for r in x)}")
for C in (96, 128, 160):
    for k in (1.5, 2.5):
        over = sum(k * t > C for t in rts)
        print(f"    cap rho*t {C}: k {k}: over cap {over}/{len(rts)}")
# (c) classes
print("(c) registered vs wide:", dict(collections.Counter((r["class_registered"], r["wide"]["pooled"]) for r in sc)))
srule = [(r["class_registered"], (r["rho2_S_101"] is not None and r["rho2_S_101"] < 0.3914103370353161)) for r in sc]
print("    S rho2 < q at 1.01 s_F (by registered class):", dict(collections.Counter(srule)))
print("    S first rho2>=q beyond 1.25 s_F (never in window):",
      sum(1 for r in sc if r.get("S_first_ge_q_over_sF") is None))
tab = collections.Counter((r["wide"]["pooled"], r["class_registered"], r["calib"]["crossed"]) for r in cal)
print("    (wide, registered, crossed at 2^-12):", dict(tab))
for key in ("class_registered",):
    pass
for w in ("clean", "mixed", "none"):
    x = [r for r in cal if r["wide"]["pooled"] == w]
    print(f"    wide {w}: n {len(x)} crossed {sum(r['calib']['crossed'] for r in x)}")
for w in ("clean", "mixed", "none"):
    x = [r for r in cal if r["class_registered"] == w]
    print(f"    registered {w}: n {len(x)} crossed {sum(r['calib']['crossed'] for r in x)}")
print("    secs per scoreable seed median", np.median([r["secs"] for r in sc]))

# 2A-PS committed observations: steps observed / t_F (end of observation relative to the fold time)
O = [json.loads(l) for l in open(ROOT / "results/phase2a_ps/observed.jsonl")]
print("2A-PS observed.jsonl (committed; seed 2,975,052 and 127 excluded: dead unit):")
for cls in ("clean", "none", "mixed"):
    x = [o["steps_observed"] / o["t_F"] for o in O if o["class"] == cls and o["t_F"] and o["seed"] not in (2975052, 2975127)]
    print(f"    {cls}: n {len(x)} end/t_F med {np.median(x):.3f} q90 {np.quantile(x, .9):.3f} max {max(x):.3f}; "
          f"> 1.5: {sum(v > 1.5 for v in x)}")

# gate / H arithmetic
print("gate arithmetic (binomial):")
for N in (400, 500, 600):
    for p in (0.15, 0.185):
        print(f"    N {N} p_clean {p}: P(clean >= 40) {binom.sf(39, N, p):.3f}  P(>=24) {binom.sf(23, N, p):.3f}")
    for p in (0.03, 0.05):
        print(f"    N {N} p_none {p}: E {N*p:.0f}; P(none >= 10) {binom.sf(9, N, p):.3f} P(>=15) {binom.sf(14, N, p):.3f} "
              f"P(>=20) {binom.sf(19, N, p):.3f}")
for n in (10, 15, 20, 30):
    for p in (0.7, 0.8, 0.875, 0.95):
        print(f"    H power n {n} p_nocross {p}: P(frac >= 0.8) {binom.sf(math.ceil(0.8 * n) - 1, n, p):.3f}")
