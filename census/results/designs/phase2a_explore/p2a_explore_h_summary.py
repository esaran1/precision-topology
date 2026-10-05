"""EXPLORATORY: summary of p2a_explore_h.jsonl (per-seed-sample survey; not registered).
Usage: python p2a_explore_h_summary.py  -> prints the numbers quoted in ../phase2a_perseed_design.md."""
import json, math
import numpy as np

Q = 0.3914103370353161
rows = [json.loads(l) for l in open("results/designs/phase2a_explore/p2a_explore_h.jsonl")]
seen = {}
for r in rows:                                    # last record of each kind per (mode, seed)
    if "run" in r or "skipped" in r:                  # "run skipped" markers carry no data
        continue
    seen[(r["mode"], r["seed"], r.get("part", "freeze") + (str(r["log2rho"]) if r.get("part") == "run" else ""))] = r
for mode in ("grid", "iid"):
    fr = [r for (m, s, p), r in sorted(seen.items()) if m == mode and p == "freeze" and "traceable" in r]
    if not fr:
        continue
    ok = [r for r in fr if r["traceable"]]
    print(f"\n== {mode}: {len(fr)} seeds, traceable M {len(ok)} "
          f"(untraceable: {[r['seed'] for r in fr if not r['traceable']]})")
    if not ok:
        continue
    q = lambda xs: "n/a" if not len(xs) else f"median {np.median(xs):.4g}, range {min(xs):.4g}-{max(xs):.4g} (n {len(xs)})"
    sF = [r["M"]["fold_up"] for r in ok]; lo = [r["M_stable"][0] for r in ok]
    ss = [r["s_star"] for r in ok if r["s_star"]]
    print("s_F:", q(sF)); print("M lower end:", q(lo)); print("s*:", q(ss))
    print("s_F/s*:", q([r["sF_over_sstar"] for r in ok if r["sF_over_sstar"]]))
    print("|m'c'|:", q([r["abs_mc"] for r in ok])); print("Lambda_F:", q([r["Lambda_F"] for r in ok]))
    print("fold h agreement max |diff|:", max(abs(r["M"]["fold_up"] - r["fold_h0.025"]) for r in ok if isinstance(r["fold_h0.025"], float)))
    print("homotopy n_steps 200 needed:", sum(r["M"]["n_steps"] == 200 for r in ok), "; min eig along homotopy:",
          q([r["M"]["homotopy_min_eig"] for r in ok]), "; fdist to pop point:", q([r["M"]["fdist_to_pop_point"] for r in ok]))
    print("rho2 max on M:", q([r["rho2_M_max"] for r in ok]))
    print("fold validation: below 1% all on M:", sum(r["validation"]["below_1pct"]["n_on_M"] == 30 for r in ok),
          "; above 0.1% none on M:", sum(r["validation"]["above_0.1pct"]["n_on_M"] == 0 for r in ok),
          "; above 1%: n(rho2>=q) of 30 per seed:", [r["validation"]["above_1pct"]["n_rho2_ge_q"] for r in ok])
    print("static H-F5 (>= 24/30 above-1% minima with rho2 >= q):",
          sum(r["validation"]["above_1pct"]["n_rho2_ge_q"] >= 24 for r in ok), "of", len(ok))
    print("1.7957 on M:", sum(r["s0_1.7957_on_M"] for r in ok), "; s*/2 on M:", sum(r["s0_half_sstar_on_M"] for r in ok),
          "; frac rule s0:", q([r["s0_frac"] for r in ok]))
    print("landing own init on M:", sum(r["landing"]["own_init_on_M"] for r in ok), "of", len(ok),
          "; 11-init landings total:", sum(r["landing"]["n_on_M_of_11"] for r in ok), "of", 11 * len(ok))
    print("freeze sec:", q([r["sec_freeze"] for r in fr]))
    for l2 in sorted({r["log2rho"] for r in rows if r.get("part") == "run" and r["mode"] == mode}, reverse=True):
     runs = [r for (m, s, p), r in sorted(seen.items()) if m == mode and p == "run" + str(l2)]
     print(f"runs at 2^{l2}: {len(runs)}")
     for r in runs:
         fo, fp = r.get("fc_own", {}), r.get("fc_pop", {})
         fz = seen[(mode, r["seed"], "freeze")]
         print(f"  {r['seed']} eps {fo.get('eps_fc') or float('nan'):.2e} LamF {fz['Lambda_F']:.2e} sF {r['s_F']:.4f} s* {fz['s_star'] or float('nan'):.4f} H-F5(static) "
               f"{fz['validation']['above_1pct']['n_rho2_ge_q']:2d}/30  cross "
               f"{'%.4f' % r['s_obs_over_sF'] if 't_obs' in r else 'none'} F {r.get('in_F_window')} fals {r.get('falsifier')} "
               f"rho2_end {r['rho2_end']:.3f} signs {r['signs_fixed']} | own: C1 {fo.get('C1_rel', float('nan')):.3f} "
               f"C2 {fo.get('C2_rel', float('nan')):.3f} ratio {fo.get('ratio_r', float('nan')):.3f} "
               f"lnerr {fo.get('abs_log_scale_err', float('nan')):.4f} | pop: {fp.get('status', '-')[:12]} "
               f"lnerr {fp.get('abs_log_scale_err', float('nan')):.4f} terr {fp.get('abs_time_err', float('nan'))}")
     both = [r for r in runs if "abs_log_scale_err" in r.get("fc_own", {}) and "abs_log_scale_err" in r.get("fc_pop", {})]
     if both:
         d = [r["fc_own"]["abs_log_scale_err"] - r["fc_pop"]["abs_log_scale_err"] for r in both]
         print(f"  per-seed minus fixed-dataset |ln s_c/s_obs|: mean {np.mean(d):.4f}, own better at {sum(x < 0 for x in d)}/{len(d)}")
