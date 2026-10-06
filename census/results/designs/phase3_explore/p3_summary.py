"""EXPLORATORY arithmetic on p3_runs.jsonl and p3_own_*.json (no new runs): per copy and ρ, the exploration-seed run
statistics, and the 1C pilot-error rule applied to them for illustration only (τ = 1.5 × q90, rounded up to the next
multiple of 5 steps).  The registered tolerances would come from the registered pilot seeds, not from these."""
import json, math, glob
import numpy as np
R = [json.loads(l) for l in open("p3_runs.jsonl")]
q90 = lambda a: float(np.quantile(np.abs(a), 0.9, method="linear"))
for c, rho in (("Q", 2 ** -13), ("S", 2 ** -12), ("T4", 2 ** -10)):
    rr = [r for r in R if r["copy"] == c and r["sample"] != "pop" and abs(r["rho"] - rho) < 1e-15]
    ec = [r["err_cross"] for r in rr]; el = [r["err_lag"] for r in rr]
    print(f"{c} rho 2^{round(math.log2(rho))}: n {len(rr)}; lag_obs {[r['lag_obs'] for r in rr]}; |err_cross| q90 {q90(ec):.1f} "
          f"-> tau_cross {5 * math.ceil(1.5 * q90(ec) / 5)}; |err_lag| q90 {q90(el):.1f} -> tau_lag {max(5, 5 * math.ceil(1.5 * q90(el) / 5))}; "
          f"median r_obs/r_fc {np.median([r['r_obs'] / r['r_fc'] for r in rr]):.3f}; "
          f"q90 window max chi {q90([r['chi_win_max'] for r in rr]):.3f}; t_obs {min(r['t_obs'] for r in rr)}-{max(r['t_obs'] for r in rr)}; "
          f"t_obs - t_c median {np.median([r['t_obs'] - r['t_c'] for r in rr]):.0f}; time/run median {np.median([r['time_s'] for r in rr]):.1f} s; "
          f"|lag_fc| >= 6: {sum(abs(r['lag_fc']) >= 6 for r in rr)}/{len(rr)}; sign agree {sum((r['lag_fc'] < 0) == (r['lag_obs'] < 0) for r in rr)}/{len(rr)}; "
          f"closer than no-lag {sum(abs(r['t_fc'] - r['t_obs']) < abs(r['t_sw_fc'] - r['t_obs']) for r in rr)}/{len(rr)}")
for f in sorted(glob.glob("p3_own_*.json")):
    d = json.load(open(f))
    for c in ("Q", "S", "T4"):
        v = [r[c] for r in d if r[c].get("valid") and r[c].get("status") == "switch"]
        k = np.array([x["kappa0"] for x in v]); kl = np.array([x["kappa_over_lam"] for x in v])
        bp = [r[c].get("bp_on") for r in d if c != "T4"]
        print(f"{f} {c}: valid+switch {len(v)}/{len(d)}; kappa0 < 0 {int((k < 0).sum())}; kappa/lam median {np.median(kl):.2f} "
              f"[{kl.min():.2f}, {kl.max():.2f}]; |kappa/(0.03 lam)| < 6: {int((np.abs(kl / 0.03) < 6).sum())}; s_switch "
              f"[{min(x['s_switch'] for x in v):.3f}, {max(x['s_switch'] for x in v):.3f}]; lam_s0 min {min(x['lam_s0'] for x in v):.2e}"
              + (f"; branch-point on copy {sum(map(bool, bp))}/{len(bp)}; hold G>0 {sum(r[c].get('bp_hold_pos', 0) > 0 for r in d)}; "
                 f"max W {max(r[c].get('W_hold', 0) for r in d)}" if c != "T4" else ""))
