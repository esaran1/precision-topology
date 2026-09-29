"""EXPLORATORY arithmetic (no training; after the author's change D): the L5 resolution rule N_min = 2(g + τ + d) = 6 steps
applied to the population runs of w2_explore_c.log / w2_explore_c_Tprime.log at the candidate settings (D: η 0.3, ρ 1;
T and T′: η 0.03, ρ 0.001).  Slaved predicted lag in steps n = κ/(η·λ_min); relative resolution r_min = N_min·ṡ/s_sw
= N_min·χ·η·λ_min; compared with the observed lag r_obs."""
import json

N_MIN = 2 * (1 + 1 + 1)
pick = {("D", "w2_explore_c.log"): (0, "duplicate", 0.3, 1.0), ("T", "w2_explore_c.log"): (0, "two-unit", 0.03, 0.001),
        ("T'", "w2_explore_c_Tprime.log"): (2, "two-unit", 0.03, 0.001)}
for (arm, f), (cls, kind, eta, rho) in pick.items():
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    r = next(r for r in rows if r["class"] == cls and r["kind"] == kind and r["eta"] == eta and r["rho"] == rho)
    lam = r["lam_min_sw"]
    print(arm, "n_pred(slaved)", round(r["lag_steps_slaved"], 1), "n_obs", r["lag_steps_obs"], "N_min", N_MIN,
          "r_min", f"{N_MIN * r['chi_sw'] * eta * lam:.1e}", "r_obs", f"{r['r_obs_path']:.2e}")
