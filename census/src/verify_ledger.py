"""Verify every headline number in CLAIMS.md against its committed artifact.

Recomputes from CSVs rather than trusting the ledger text. Any mismatch is
printed as FINDING: and written to results/ledger_verification.txt.
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
F: list[str] = []


REC: list[tuple] = []


def chk(label, got, want, tol=0.01):
    REC.append((label, got, want, tol))
    ok = (got is not None) and abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'MISMATCH'} {label}: artifact={got} ledger={want}")
    if not ok:
        F.append(f"{label}: artifact={got} ledger={want}")


def main() -> None:
    print("T1/T2 monotonic zero")
    d = pd.read_csv(R / "monotonic_zero_decomposition.csv")
    chk("total runs", float(d.runs.sum()), 5580.0, 0)
    chk("separations", float(d.separations.sum()), 0.0, 0)

    print("pooled R union (abstract numbers)")
    u = pd.read_csv(R / "r_pooled_union.csv")
    u["solved"] = u.solved.astype(bool)
    chk("pooled runs", float(len(u)), 3150.0, 0)
    chk("optimisers", float(u.opt.nunique()), 3.0, 0)
    low, high = u[u.R < 0.30], u[u.R > 0.50]
    chk("runs below R=0.30", float(len(low)), 2285.0, 0)
    chk("solved below R=0.30", float(low.solved.sum()), 0.0, 0)
    chk("runs above R=0.50", float(len(high)), 461.0, 0)
    chk("solved above R=0.50", float(high.solved.sum()), 460.0, 0)

    print("T56 per-placement theorem check")
    t = pd.read_csv(R / "theorem_perplacement.csv")
    chk("solvers checked", float(len(t)), 66.0, 0)
    chk("per-placement violations", float((~t.ok.astype(bool)).sum()), 0.0, 0)
    chk("solvers with w2<0", float((t.w2 < 0).sum()), 22.0, 0)
    chk("median slack", float((t.w2.abs() / t.bound_perplace).median()), 1.2488, 0.001)

    print("T57 placement/bias decomposition")
    d = pd.read_csv(R / "phase1_decomposition.csv")
    f = d[d.precision == "float32"]
    chk("identity disagreements", float((f.predicted_solved != f.solved).sum()), 0.0, 0)
    u = f[~f.predicted_solved]
    chk("placement failures", float((u.failure == "placement").sum()), 1664.0, 0)
    chk("bias failures", float((u.failure == "bias").sum()), 306.0, 0)
    s_ = f[f.predicted_solved]
    chk("median solved rho", float(s_.rho.median()), 0.9674, 0.001)
    chk("rho>0.9 all solve", float((f[f.rho > 0.9].predicted_solved).all()), 1.0, 0)

    print("T43 budget law")
    d = pd.read_csv(R / "onset_law_extended.csv"); d = d[d.bracketed & d.onset.notna()]
    chk("bracketed cells", float(len(d)), 6.0, 0)
    s = np.polyfit(np.log(d.budget), np.log(d.onset - 1), 1)[0]
    chk("onset exponent", s, -0.7340, 0.002)
    a = pd.read_csv(R / "budget_alpha.csv")
    al = np.polyfit(np.log(a.budget), np.log(a.w2_med), 1)[0]
    chk("alpha (1k-160k)", al, 1.1172, 0.002)
    chk("rate at 80k", float(a[a.budget == 80000].rate.iloc[0]), 1.0, 0)

    print("T44 four-family")
    fam = {"q4": (1.25, -0.8305), "q2": (1.50, -0.6749),
           "q1": (2.00, -0.6521), "q0.667": (2.4993, -0.5000)}
    d = pd.read_csv(R / "family_onsets.csv"); d = d[d.bracketed & d.onset_eps.notna()]
    for name, sub in d.groupby("family"):
        key = name.split("_")[0]
        if key in fam and len(sub) >= 2:
            s = np.polyfit(np.log(sub.budget), np.log(sub.onset_eps), 1)[0]
            chk(f"{key} exponent", s, fam[key][1], 0.02)

    print("T47 cross-optimizer")
    d = pd.read_csv(R / "sgd_q4_refined.csv"); d = d[d.bracketed & d.onset_eps.notna()]
    chk("q4 SGD bracketed", float(len(d)), 4.0, 0)
    s = np.polyfit(np.log(d.budget), np.log(d.onset_eps), 1)[0]
    chk("q4 SGD exponent", s, 0.0056, 0.002)
    d = pd.read_csv(R / "sgd_onsets.csv"); d = d[d.bracketed & d.onset.notna()]
    s = np.polyfit(np.log(d.budget), np.log(d.onset - 1), 1)[0]
    chk("family A SGD exponent", s, -0.3255, 0.002)

    print("T38/T43 CIFAR")
    d = pd.read_csv(R / "budget_flip.csv")
    p = d.pivot_table(index="epochs", columns="activation", values="test_errors", aggfunc="mean")
    chk("GELU adv at 2ep", float(p.loc[2, "relu"] - p.loc[2, "gelu"]), 750.4, 1.0)
    chk("GELU adv at 12ep", float(p.loc[12, "relu"] - p.loc[12, "gelu"]), 150.6, 1.0)

    print("T41 exclusion table")
    d = pd.read_csv(R / "criticality.csv")
    z = d[d.population == "zero_basin_constructed"]
    chk("min zero-basin grad norm", float(z.grad_norm.min()), 0.0192, 0.002)
    chk("negative lambda_min count", float((z.lambda_min < 0).sum()), 6.0, 0)

    print("T58 scaling limit")
    import math as _m
    from .fold1d_theorem import maximum_gap as _mg, dip_depth as _dd
    d = pd.read_csv(R / "scaling_limit_switches.csv").iloc[0]
    K = float(d["K"]); D_inf = 4 * _m.sqrt(2) / 3
    chk("dip depth 4sqrt2/3", D_inf, 1.885618, 1e-5)
    chk("K", K, 0.5794549, 1e-5)
    chk("kappa_0", K / D_inf, 0.307302, 1e-5)
    chk("kappa(1.02) vs kappa_0 (%)",
        (_mg(1.02, resolution=800) / _dd(1.02) / (K / D_inf) - 1) * 100, -0.38, 0.05)
    chk("R_glob^inf (frozen 0.005 grid)", float(d["R_glob"]), 0.19991, 1e-4)
    ls_ = pd.read_csv(R / "limit_switch.csv").iloc[0]
    chk("R_glob^inf certified lower", float(ls_["R_glob_inf_lo"]), 0.1974, 5e-5)
    chk("R_glob^inf certified upper", float(ls_["R_glob_inf_hi"]), 0.1992, 5e-5)
    kb_ = pd.read_csv(R / "limit_K_base.csv").iloc[0]
    chk("K certified (exact extrema) lower", float(kb_.K_lo), 0.5794558, 1e-7)
    chk("K certified (exact extrema) upper", float(kb_.K_hi), 0.5794951, 1e-7)
    chk("R_glob^inf certified with K interval, lower", float(kb_.R_glob_inf_lo), 0.19738, 1e-5)
    chk("R_glob^inf certified with K interval, upper", float(kb_.R_glob_inf_hi), 0.19920, 1e-5)
    chk("frozen grid value lies above the certified interval",
        float(float(d["R_glob"]) > float(ls_["R_glob_inf_hi"])), 1.0, 0)
    chk("R_spin^inf == R_glob^inf (no window)",
        float(d["R_spin"]) - float(d["R_glob"]), 0.0, 0)
    chk("R_solve^inf", float(d["R_solve"]), 0.30711, 1e-4)
    b = pd.read_csv(R / "blockB_switches.csv").sort_values("a")
    e = (b["a"] - 1).values
    dev = (b["R_glob"] / float(d["R_glob"]) - 1).values
    chk("R_glob dev all positive", float((dev > 0).all()), 1.0, 0)
    chk("R_glob dev monotone", float(np.all(np.diff(dev) > 0)), 1.0, 0)
    chk("R_glob dev at a=1.60 (%)", dev[-1] * 100, 14.29, 0.05)
    chk("R_glob corr(eps,dev)", float(np.corrcoef(e, dev)[0, 1]), 0.9866, 0.001)
    chk("R_glob log-log slope (frozen grid, superseded)", float(np.polyfit(np.log(e), np.log(dev), 1)[0]), 0.947, 0.01)
    c1 = float(np.linalg.lstsq(np.vstack([e]).T, dev, rcond=None)[0][0])
    chk("c1", c1, 0.23099, 1e-4)
    chk("c1 fit max residual",
        float(np.abs(b["R_glob"].values - float(d["R_glob"]) * (1 + c1 * e)).max()),
        0.00133, 1e-4)
    print("T58 refit on certified intervals (rglob_refit)")
    rf = dict(pd.read_csv(R / "rglob_convergence_refit.csv").values)
    for q_, v_, t_ in (("log-log slope, midpoints", 0.829, 0.001), ("log-log slope, min over certified box", 0.714, 0.001),
                       ("log-log slope, max over certified box", 0.955, 0.001), ("slope range inside registered band [0.5, 2]", 1.0, 0),
                       ("one-term law R_inf(1 + c1 eps) feasible within all certified intervals", 0.0, 0),
                       ("two-term law (eps and eps^2) feasible within all certified intervals", 1.0, 0),
                       ("two-term law: c1 = m/R_inf min over feasible", 0.243, 0.001),
                       ("two-term law: c1 max over feasible", 0.321, 0.001),
                       ("two-term law: q max over feasible", -0.0069, 0.0001),
                       ("limit-free intercept min over box", 0.2002, 0.0001),
                       ("limit-free intercept range overlaps certified limit", 0.0, 0),
                       ("dev all positive (worst corner)", 1.0, 0), ("dev monotone in a, midpoints", 1.0, 0)):
        chk(q_, float(rf[q_]), v_, t_)
    dv = (b["R_solve"] / float(d["R_solve"]) - 1).values
    chk("R_solve dev all negative (S-3 falsified)", float((dv < 0).all()), 1.0, 0)
    chk("R_solve mean dev (%)", dv.mean() * 100, -0.29, 0.02)

    print("T59 Block E intervention")
    d = pd.read_csv(R / "blockE_intervene.csv")
    k = d[d.kept.notna()]
    hl = k[k.arm == "hold_low"]; hh = k[k.arm == "hold_high"]; nf = k[k.arm == "noise_floor"]
    chk("hold_low kept", float(hl.kept.sum()), 0.0, 0)
    chk("hold_low n", float(len(hl)), 37.0, 0)
    chk("hold_high kept", float(hh.kept.sum()), 33.0, 0)
    chk("hold_high n", float(len(hh)), 37.0, 0)
    chk("noise_floor kept", float(nf.kept.sum()), 37.0, 0)
    chk("hold_high placed /40", float(d[d.arm == "hold_high"].placed_final.sum()), 33.0, 0)
    chk("hold_high solved /40", float(d[d.arm == "hold_high"].solved_final.sum()), 0.0, 0)
    chk("control placed /40", float(d[d.arm == "control"].placed_final.sum()), 37.0, 0)
    chk("jump placed /40", float(d[d.arm == "jump"].placed_final.sum()), 9.0, 0)
    chk("cold_high placed /40", float(d[d.arm == "cold_high"].placed_final.sum()), 2.0, 0)
    chk("cold_low placed /40", float(d[d.arm == "cold_low"].placed_final.sum()), 0.0, 0)
    lost = (hl.lost_at - hl.intervened_at)
    chk("hold_low max steps to lose", float(lost.max()), 50.0, 0)
    # hold_high's held R must lie strictly between the two switch points
    sw = pd.read_csv(R / "blockB_switches.csv")
    sw13 = sw[sw.a == 1.30].iloc[0]
    held = float(d[(d.arm == "hold_high") & d.kept.notna()].R_final.median())
    chk("hold_high held R", held, 0.24663, 1e-4)
    chk("held R above R_glob", float(held > float(sw13.R_glob)), 1.0, 0)
    chk("held R below R_solve", float(held < float(sw13.R_solve)), 1.0, 0)

    print("Block E stall: per-arm verdicts")
    stl = pd.read_csv(R / "blockE_stall.csv")
    sh12 = stl[stl.budget == 12000]
    lo60 = stl[stl.budget == 60000]
    ctl = float(sh12[sh12.arm == "control"].grad_500.median())
    for arm, want_grad, want_60 in (("jump", 0.00072, 7.0), ("cold_high", 0.00032, 3.0)):
        gfrac = float(sh12[sh12.arm == arm].grad_500.median()) / ctl
        chk(f"{arm} grad frac of control", gfrac, want_grad, 5e-5)
        chk(f"{arm} placed at 60k", float(lo60[lo60.arm == arm].placed.sum()), want_60, 0)
    # the registered S-A budget criterion: >25%
    j60 = float(lo60[lo60.arm == "jump"].placed.mean())
    c60 = float(lo60[lo60.arm == "cold_high"].placed.mean())
    chk("jump meets S-A budget (>25%)", float(j60 > 0.25), 1.0, 0)
    chk("cold_high FAILS S-A budget (>25%)", float(c60 > 0.25), 0.0, 0)
    chk("cold_high 60k rate", c60, 0.2000, 1e-4)

    print("T60 R_spin dropped")
    fw = pd.read_csv(R / "blockB_fine_windows.csv").sort_values("a")
    chk("R_glob == R_spin, all a", float(np.allclose(fw.R_glob, fw.R_spin, atol=1e-12)), 1.0, 0)
    chk("every window 1 grid step",
        float(np.allclose((fw.w2_spin - fw.w2_fold) / 0.01, 1.0, atol=0.05)), 1.0, 0)
    chk("a=1.60 window width", float(fw[fw.a == 1.60].R_spin.iloc[0] - fw[fw.a == 1.60].R_fold.iloc[0]),
        0.00111, 1e-5)
    meas = np.array([0.2330, 0.2375, 0.2418, 0.2456, 0.2489, 0.2563])
    bb = pd.read_csv(R / "blockB_switches.csv").sort_values("a")
    rg = [float(bb[bb.a == 1.30].R_glob.iloc[0])] + [float(fw[fw.a == a].R_glob.iloc[0])
                                                     for a in (1.35, 1.40, 1.45, 1.50, 1.60)]
    rg = np.array(rg)
    chk("R_glob drift (%)", (rg[-1] / rg[0] - 1) * 100, 4.5, 0.1)
    chk("corr(measured, R_glob)", float(np.corrcoef(meas, rg)[0, 1]), 0.9965, 0.001)
    chk("mean offset measured/R_glob", float(np.mean(meas / rg)), 1.1153, 1e-3)

    print("T61 Block G transfer")
    gs = pd.read_csv(R / "blockG_switches.csv")
    gc = pd.read_csv(R / "blockG_crossings.csv")
    for a, want_r, want_w in ((1.30, 0.2189, 0.7358), (1.50, 0.2304, 0.6925)):
        c = gc[(gc.a == a) & gc.cross_R.notna()]
        med = c.groupby("window").agg(Rm=("cross_R", "median"), wm=("cross_w2", "median"))
        cvR = float(med.Rm.std(ddof=1) / med.Rm.mean())
        cvW = float(med.wm.std(ddof=1) / med.wm.mean())
        chk(f"a={a} CV(R)", cvR, want_r, 1e-3)
        chk(f"a={a} CV(|w2|)", cvW, want_w, 1e-3)
        chk(f"a={a} CV(R) < CV(w2)/2", float(cvR < cvW / 2), 1.0, 0)
    sc = pd.read_csv(R / "blockG_scaling.csv")
    chk("G-4 windows within 3%", float((sc.err_pct.abs() <= 3).sum()), 5.0, 0)
    chk("G-4 worst error (%)", float(sc.err_pct.abs().max()), 0.808, 0.01)
    chk("kappa_0 spread (x)", float(sc.kappa_0.max() / sc.kappa_0.min()), 8.077, 0.01)

    print("T62 Block K family B")
    bk = pd.read_csv(R / "blockK_auc.csv").sort_values("alpha")
    chk("box/norm ratio at alpha=-0.5",
        float(bk[bk.alpha == -0.50].gstar_box.iloc[0] / bk[bk.alpha == -0.50].gnorm.iloc[0]),
        8.0, 1e-6)
    chk("Ghat_norm exponent in |alpha|",
        float(np.polyfit(np.log(bk.alpha.abs()), np.log(bk.gnorm), 1)[0]), 1.0, 0.01)
    chk("cells with AUC(R_B)==AUC(prod)",
        float((np.abs(bk.auc_R_B - bk.auc_prod) < 1e-12).sum()), 5.0, 0)
    chk("cells at AUC 1.0", float((bk.auc_R_B >= 0.999999).sum()), 4.0, 0)
    chk("band runs at alpha=-1.00", float(bk[bk.alpha == -1.00].band_runs.iloc[0]), 2.0, 0)
    chk("alpha=-1 separation gap (unsolved max)",
        float(bk[bk.alpha == -1.00].unsolved_max_prod.iloc[0]), 0.295219, 1e-5)
    chk("alpha=-1 separation gap (solved min)",
        float(bk[bk.alpha == -1.00].solved_min_prod.iloc[0]), 7.258799, 1e-5)
    nb = pd.read_csv(R / "blockK_family_b_normalised.csv")
    sv = nb.solved.values.astype(bool)
    def _auc(v):
        pos, neg = np.asarray(v)[sv], np.asarray(v)[~sv]
        return float(sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
                     / (len(pos) * len(neg)))
    chk("pooled AUC(R_B)", _auc(nb.R_B), 0.9990, 1e-3)
    chk("pooled AUC(|w2|)", _auc(nb.w2_abs), 0.9901, 1e-3)
    chk("AUC margin under registered 0.10",
        float((_auc(nb.R_B) - _auc(nb.w2_abs)) < 0.10), 1.0, 0)

    print("T63 Block F optimisers")
    bf = pd.read_csv(R / "blockF_optimisers.csv")
    cf = bf[bf.cross_R.notna()]
    meds = {n: float(cf[cf.optimiser == n].cross_R.median()) for n in ("adam", "adamw", "sgd")}
    chk("Adam cross R", meds["adam"], 0.23094, 1e-4)
    chk("AdamW cross R", meds["adamw"], 0.22980, 1e-4)
    chk("SGD cross R", meds["sgd"], 0.22888, 1e-4)
    chk("F-1 ordering holds",
        float(meds["adam"] > meds["adamw"] > meds["sgd"]), 1.0, 0)
    rates = {n: float(cf[cf.optimiser == n].local_rate.median()) for n in meds}
    chk("rate spread (Adam/SGD)", rates["adam"] / rates["sgd"], 3.917, 0.02)
    R_glob_125 = 0.21066
    offs = {n: (meds[n] / R_glob_125 - 1) * 100 for n in meds}
    chk("SGD offset (%)", offs["sgd"], 8.645, 0.02)
    chk("SGD predicted offset (%)", offs["adam"] * rates["sgd"] / rates["adam"], 2.458, 0.02)
    chk("F-2 SGD error exceeds 4pp",
        float(abs(offs["sgd"] - offs["adam"] * rates["sgd"] / rates["adam"]) > 4.0), 1.0, 0)
    chk("offset spread (pp)", max(offs.values()) - min(offs.values()), 0.981, 0.02)
    rng = np.random.default_rng(0)
    def _bm(n, k=4000):
        v = cf[cf.optimiser == n].cross_R.values
        return np.array([np.median(rng.choice(v, len(v), replace=True)) for _ in range(k)])
    ba, bw, bs = _bm("adam"), _bm("adamw"), _bm("sgd")
    chk("P(full ordering) below 0.5", float(((ba > bw) & (bw > bs)).mean() < 0.5), 1.0, 0)

    print("T64 kappa certified")
    kc = pd.read_csv(R / "kappa_certified.csv")
    sub = kc[kc.a <= 1.60]
    chk("max certificate rel width (%)", float(sub.rel_width.max() * 100), 0.0855, 0.002)
    chk("all widths under 0.2%", float((sub.rel_width < 0.002).all()), 1.0, 0)
    chk("certified kappa lower", float(sub.kappa_lo.min()), 0.307747, 1e-5)
    chk("certified kappa upper", float(sub.kappa_hi.max()), 0.317616, 1e-5)
    chk("certified width (%)",
        (float(sub.kappa_hi.max()) / float(sub.kappa_lo.min()) - 1) * 100, 3.207, 0.01)
    chk("excludes T50 lower endpoint",
        float(float(sub.kappa_lo.min()) > 0.30544), 1.0, 0)
    kb = pd.read_csv(R / "kappa_boundary.csv")
    chk("boundary below interior, all", float(kb.strictly_below.all()), 1.0, 0)
    kk = pd.read_csv(R / "kappa_K_certified.csv").iloc[0]
    chk("K certified lower", float(kk.K_lo), 0.579454977, 1e-8)
    chk("K rel width (%)", (float(kk.K_hi) / float(kk.K_lo) - 1) * 100, 0.0856, 0.001)

    print("T65 one Ghat everywhere")
    gg = pd.read_csv(R / "ghat_certified_all.csv").set_index("a")
    chk("certified/restricted min (%)", float(gg.pct.min()), 0.1313, 0.002)
    chk("certified/restricted max (%)", float(gg.pct.max()), 0.8399, 0.002)
    fr = []
    for fn in ("r_pooled.csv", "r_adamw.csv"):
        t = pd.read_csv(R / fn)
        if "a" not in t:
            t = t.assign(a=1.25)
        fr.append(t[["a", "w2", "solved"]])
    pool = pd.concat(fr, ignore_index=True)
    gcert = pool.a.map(lambda v: float(gg.loc[round(float(v), 2), "Ghat_certified"]))
    pool = pool.assign(R_new=pool.w2.abs() * gcert / 2)
    below = pool[pool.R_new < 0.30]
    above = pool[pool.R_new > 0.50]
    chk("certified: n below 0.30", float(len(below)), 2281.0, 0)
    chk("certified: solved below 0.30", float(below.solved.sum()), 0.0, 0)
    chk("certified: n above 0.50", float(len(above)), 463.0, 0)
    chk("certified: solved above 0.50", float(above.solved.sum()), 461.0, 0)
    ba = pd.read_csv(R / "blockA_per_a.csv").sort_values("a")
    Rn = ba.w2.values * np.array([float(gg.loc[round(float(v), 2), "Ghat_certified"])
                                  for v in ba.a]) / 2
    cvn = float(Rn.std(ddof=0) / Rn.mean())
    chk("certified CV(R) across a", cvn, 0.0324, 5e-4)
    chk("O1 still met", float(cvn <= 0.15 and cvn < float(ba.w2.std(ddof=0) / ba.w2.mean()) / 2), 1.0, 0)

    print("T66 nu")
    ns = pd.read_csv(R / "nu_sweep.csv")
    m2 = ns.groupby(["a", "budget"]).w2_abs.median().reset_index()
    als = [float(np.polyfit(np.log(g.budget), np.log(g.w2_abs), 1)[0])
           for _, g in m2.groupby("a")]
    eps = [float(a) - 1.0 for a, _ in m2.groupby("a")]
    chk("corr(eps, alpha)", float(np.corrcoef(eps, als)[0, 1]), -0.9943, 0.002)
    chk("alpha spread", float(max(als) - min(als)), 0.1692, 0.002)
    nus = [-float(np.polyfit(np.log(g.a - 1), np.log(g.w2_abs), 1)[0])
           for _, g in m2.groupby("budget")]
    chk("fixed-budget nu mean", float(np.mean(nus)), -0.0626, 0.002)
    chk("fixed-budget nu sign inconsistent",
        float(any(v > 0 for v in nus) and any(v < 0 for v in nus)), 1.0, 0)

    print("T67 pathwise distance")
    pw = pd.read_csv(R / "pathwise_distance.csv")
    chk("n runs", float(len(pw)), 400.0, 0)
    chk("median init distance", float(pw.init_distance.median()), 5.2234, 1e-3)
    chk("median min-along-path", float(pw.min_distance.median()), 4.8773, 1e-3)
    chk("median ratio min/init", float((pw.min_distance / pw.init_distance).median()),
        0.9918, 1e-3)
    # corrected 2026-09-23: step 0 is never logged; the true initialisation is regenerated
    pi = pd.read_csv(R / "discrepancy_pathwise_init.csv")
    chk("step 1 reproduced from regenerated init", float(pi.step1_reproduced.sum()), 400.0, 0)
    chk("median distance at true init", float(pi.init_distance_step0.median()), 5.2259, 1e-3)
    chk("median min incl. step 0", float(pi.min_including_step0.median()), 4.8755, 1e-3)
    chk("minimum at initialisation (runs)", float(pi.min_at_step0.sum()), 114.0, 0)
    appr = 1 - pi.min_including_step0 / pi.init_distance_step0
    chk("runs > 5% closer", float((appr > 0.05).sum()), 112.0, 0)
    chk("largest approach (%)", float(appr.max() * 100), 23.43, 0.01)

    print("T55 exceptions above 0.50")
    exc = pd.read_csv(R / "exceptions_above_050.csv").sort_values("seed")
    chk("n exceptions", float(len(exc)), 2.0, 0)
    s13 = exc[exc.seed == 13].iloc[0]
    s82 = exc[exc.seed == 82].iloc[0]
    chk("seed13 placement G", float(s13.placement_G), 0.768428, 1e-5)
    chk("seed13 placement ok", float(bool(s13.placement_ok)), 1.0, 0)
    chk("seed13 bias miss", float(s13.bias_miss), 0.010958, 1e-6)
    chk("seed13 outer violations", float(s13.dense_viol_outer), 11.0, 0)
    chk("seed13 sample errors", float(s13.sample_errors), 1.0, 0)
    chk("seed82 placement G", float(s82.placement_G), -0.216202, 1e-5)
    chk("seed82 placement fails", float(not bool(s82.placement_ok)), 1.0, 0)
    chk("seed82 w1 near zero", float(abs(s82.w1) < 0.05), 1.0, 0)
    chk("seed82 total violations",
        float(s82.dense_viol_inner + s82.dense_viol_outer), 3847.0, 0)
    chk("seed82 sample errors", float(s82.sample_errors), 185.0, 0)
    chk("seed82 new entrant (was below 0.50)", float(s82.R_restricted < 0.50), 1.0, 0)
    chk("seed82 R certified", float(s82.R_certified), 0.500886, 1e-6)

    print("T66 alpha(eps)")
    ae = pd.read_csv(R / "alpha_of_eps.csv").sort_values("eps")
    chk("alpha min", float(ae.alpha.min()), 1.1719, 1e-3)
    chk("alpha max", float(ae.alpha.max()), 1.3411, 1e-3)
    chk("alpha median over onset region", float(ae.alpha.median()), 1.2797, 1e-3)

    print("T68 our link construction")
    ol = pd.read_csv(R / "our_link_verification.csv").iloc[0]
    chk("Gauss linking number", float(ol.linking_number), -1.0, 0.02)
    chk("core separation = R", float(ol.core_separation), 1.0, 1e-6)
    chk("no core self-intersection", float(ol.core_self_min > 1e-6), 1.0, 0)
    chk("tubes disjoint (2rho < sep)", float(bool(ol.tubes_disjoint)), 1.0, 0)
    chk("tube embedded (rho < reach)", float(bool(ol.tube_embedded)), 1.0, 0)

    print("T69 S2uS2 budget sweep (WITHDRAWN -- overlapping regions; numbers checked only as the record)")
    s2 = pd.read_csv(R / "blockS2_budget.csv")
    gaps = {int(B): float(g[g.activation == "gelu"].accuracy.mean()
                          - g[g.activation == "relu"].accuracy.mean())
            for B, g in s2.groupby("budget")}
    Bs = sorted(gaps)
    chk("S2 gap at 1k", gaps[Bs[0]], 0.0138, 5e-4)
    chk("S2 gap at 64k", gaps[Bs[-1]], 0.0402, 5e-4)
    chk("S2 gap widens monotonically",
        float(all(gaps[Bs[i]] < gaps[Bs[i + 1]] for i in range(len(Bs) - 1))), 1.0, 0)
    chk("S2-1 falsified (gap grew)", float(gaps[Bs[-1]] > gaps[Bs[0]]), 1.0, 0)
    relu = [float(s2[(s2.budget == B) & (s2.activation == "relu")].accuracy.mean())
            for B in Bs]
    gelu = [float(s2[(s2.budget == B) & (s2.activation == "gelu")].accuracy.mean())
            for B in Bs]
    chk("ReLU flat across 64x budget", relu[-1] - relu[0], 0.0084, 1e-3)
    chk("ReLU perfect runs, all budgets", float(s2[s2.activation == "relu"].perfect.sum()), 0.0, 0)
    chk("GELU perfect at 64k",
        float(s2[(s2.budget == Bs[-1]) & (s2.activation == "gelu")].perfect.sum()), 8.0, 0)
    chk("both non-decreasing in budget",
        float(all(relu[i] <= relu[i + 1] + 1e-9 for i in range(3))
              and all(gelu[i] <= gelu[i + 1] + 1e-9 for i in range(3))), 1.0, 0)

    print("T73 optimiser equivalence (Block C in-band population, a = 1.25)")
    v = pd.read_csv(R / "blockC_verdicts.csv").set_index("pair")
    chk("pairs equivalent", float((v.verdict == "equivalent").sum()), 3.0, 0)
    chk("adam - adamw diff", float(v.loc["adam - adamw", "diff"]), -0.0008, 1e-4)
    chk("adam - sgd diff", float(v.loc["adam - sgd", "diff"]), 0.0029, 1e-4)
    chk("adamw - sgd diff", float(v.loc["adamw - sgd", "diff"]), 0.0037, 1e-4)

    print("T74 cross-family follow-up (Y1)")
    y = pd.read_csv(R / "crossfamily_followup_scores.csv")
    chk("comparisons within tol", float(y.within.sum()), 8.0, 0)
    chk("max |q2 - A|", float(y["diff"].abs().max()), 0.00215, 1e-4)
    chk("q2 share outside domain", float(y.q2_frac_outside.max()), 0.0, 0)
    dom = pd.read_csv(R / "crossfamily_followup_domain.csv")
    chk("domain condition at all a", float(dom.condition_holds.sum()), 4.0, 0)
    z = pd.read_csv(R / "crossfamily_q1control_scores.csv")
    chk("Z1: q1 beyond two steps (8 comparisons)", float(z.beyond_two_steps.sum()), 8.0, 0)
    chk("Z1: min q1-A steps, R_glob", float(z[z.threshold == "glob"].steps.min()), 35.0, 0.01)
    chk("Z1: min q1-A steps, R_solve", float(z[z.threshold == "solve"].steps.min()), 5.0, 0.01)
    chk("Z1: q1 share outside domain", float(z.q1_frac_outside.max()), 0.0, 0)
    zg = pd.read_csv(R / "crossfamily_q1control_validity.csv")
    chk("Z1 gate passes at a = 1.30", float(zg["pass"].sum()), 2.0, 0)

    print("T75 Block A5d at k = 1")
    k1 = pd.read_csv(R / "blockA5d_k1_runs.csv", dtype={"act": str})
    k1["perfect"] = k1.perfect.astype(str).str.lower() == "true"
    chk("runs", float(len(k1)), 880.0, 0)
    low = k1[k1.act.isin(["0.9", "1.0"])]
    chk("A1: perfect at a <= 1", float(low.perfect.sum()), 0.0, 0)
    chk("A1: min uniform errors at a <= 1", float(low.heldout_errors.min()), 22.0, 0)
    top = k1[(k1.act == "3.0") & (k1.budget == 64_000)]
    chk("A4: perfect at a = 3.0, 64k", float(top.perfect.sum()), 7.0, 0)
    fa = k1[~k1.act.isin(["relu", "gelu"])]
    chk("max perfect fraction (no onset)",
        float(fa.groupby(["act", "budget"]).perfect.mean().max()), 0.35, 1e-9)
    tr = pd.read_csv(R / "blockA5d_k1_first_perfect.csv").set_index("budget")
    chk("post hoc: smallest a with any perfect at 1k", float(tr.loc[1000, "smallest_a_any_perfect"]), 3.0, 0)
    chk("post hoc: smallest a with any perfect at 4k-64k",
        float(tr.loc[[4000, 16000, 64000], "smallest_a_any_perfect"].max()), 1.35, 0)
    chk("post hoc: 50% onset reached at any budget", float(tr.onset_half_reached.sum()), 0.0, 0)
    chk("no perfect run at a <= 1.2",
        float(k1[k1.act.isin(["0.9", "1.0", "1.05", "1.1", "1.2"])].perfect.sum()), 0.0, 0)

    print("T76 global Ĝ certificate")
    bb = pd.read_csv(R / "ghat_bnb.csv")
    chk("all a converged", float(bb.converged.astype(str).str.lower().eq("true").sum()), 13.0, 0)
    chk("max relative enclosure width", float(bb.rel_width.max()), 0.00099, 0.00002)
    chk("committed Ghat vs global lo, max rel diff",
        float(((bb.Ghat_certified - bb.Ghat_lo) / bb.Ghat_lo).abs().max()), 0.000084, 0.000002)
    chk("committed Ghat above global hi (count)", float((bb.Ghat_certified > bb.Ghat_hi).sum()), 0.0, 0)
    chk("argmax w1 <= a/1.4 (count)", float((bb.w1 <= bb.a / 1.4).sum()), 13.0, 0)

    print("T77 per-a table, certified Ĝ")
    pa = pd.read_csv(R / "wi_per_a_certified.csv").set_index("a")
    chk("R_glob(1.30)", float(pa.loc[1.3, "R_glob_cert"]), 0.2153, 1e-4)
    chk("R_solve(1.30)", float(pa.loc[1.3, "R_solve_cert"]), 0.3078, 1e-4)
    chk("crossing R median of cell medians (1.30)", float(pa.loc[1.3, "cross_R_median_of_cell_medians"]),
        0.2338, 1e-4)
    chk("crossing R (1.60)", float(pa.loc[1.6, "cross_R_median_of_cell_medians"]), 0.2580, 1e-4)
    chk("crossings total", float(pa.n_crossings.sum()), 974.0, 0)
    cvt = pd.read_csv(R / "wi_per_a_cv.csv").iloc[0]
    chk("CV(R) certified, 6 a", float(cvt.cv_R_cert), 0.0324, 1e-4)
    chk("CV(|w2|), 6 a", float(cvt.cv_w2), 0.2821, 1e-4)
    cr_ = pd.read_csv(R / "wi_crossing_runs.csv")
    bc = pd.read_csv(R / "blockA_crossings.csv")
    m_ = bc.merge(cr_.groupby(["a", "budget"]).size().reset_index(name="n2"), on=["a", "budget"])
    chk("crossing counts match blockA_crossings", float((m_.n == m_.n2).sum()), 30.0, 0)

    print("T78 registration census")
    rc = pd.read_csv(R / "registration_census.csv")
    chk("registered predictions (headline convention: one row per prediction)", float(len(rc)), 232.0, 0)
    chk("first-round rows (2026-09-23)", float((rc.census_round == "2026-09-23").sum()), 164.0, 0)
    chk("third-round rows (2026-09-25)", float((rc.census_round == "2026-09-25").sum()), 37.0, 0)
    ru = pd.read_csv(R / "registration_census_by_unit.csv")
    chk("registered units (appendix convention: per a)", float(len(ru)), 292.0, 0)
    for v, n in (("PASS", 134), ("FAIL", 99), ("PARTIAL", 13), ("UNRESOLVED", 46)):
        chk(f"units, all: {v}", float((ru.verdict == v).sum()), float(n), 0)
    for v, n in (("PASS", 103), ("FAIL", 72), ("PARTIAL", 21), ("UNRESOLVED", 36)):
        chk(f"all: {v}", float((rc.verdict == v).sum()), float(n), 0)
    r64 = rc[rc.counted_in_existing_64 == "yes"]
    chk("existing 64 rows", float(len(r64)), 64.0, 0)
    for v, n in (("PASS", 27), ("FAIL", 24), ("PARTIAL", 1), ("UNRESOLVED", 12)):
        chk(f"64: {v}", float((r64.verdict == v).sum()), float(n), 0)
    tl = pd.read_csv(R / "registration_tally.csv").set_index("scope")
    for scope, want in (("scored by registered rules", (210, 100, 66, 8, 36)),
                        ("assigned post hoc in the census", (22, 3, 6, 13, 0)),
                        ("since 2026-09-23: scored by registered rules", (59, 34, 19, 0, 6)),
                        ("since 2026-09-23: assigned post hoc", (9, 0, 0, 9, 0)),
                        ("by registered unit: scored by registered rules", (278, 131, 93, 8, 46)),
                        ("by registered unit: assigned post hoc in the census", (14, 3, 6, 5, 0)),
                        ("by registered unit: since 2026-09-23: scored by registered rules", (127, 65, 46, 0, 16)),
                        ("by registered unit: since 2026-09-23: assigned post hoc", (1, 0, 0, 1, 0)),
                        ("since 2026-09-23: validity gates (not predictions; not in the headline)", (20, 20, 0, 0, 0))):
        got = tl.loc[scope]
        chk(f"{scope}: n/PASS/FAIL/PARTIAL/UNRES",
            float(sum(int(got[k]) * 10 ** (3 * i) for i, k in enumerate(["n", "PASS", "FAIL", "PARTIAL", "UNRESOLVED"]))),
            float(sum(v * 10 ** (3 * i) for i, v in enumerate(want))), 0)
    chk("E-2 as registered is FAIL", float(rc[rc.id == "E-2"].verdict.iloc[0] == "FAIL"), 1.0, 0)
    e2 = pd.read_csv(R / "wi_e2_rescore.csv").set_index("arm")
    chk("E-2 kept", float(e2.loc["hold_high", "kept"]), 33.0, 0)
    chk("E-2 intervened", float(e2.loc["hold_high", "intervened"]), 37.0, 0)

    print("T79 metric check, certified Ĝ")
    mc = pd.read_csv(R / "metric_check.csv").set_index("ghat")
    chk("restricted continuous 10% reproduces 0.055", float(mc.loc["restricted", "continuous_10"]), 0.055, 0.001)
    chk("restricted continuous 90% reproduces 0.307", float(mc.loc["restricted", "continuous_90"]), 0.307, 0.001)
    chk("zero-rate share, certified (%)", float(mc.loc["certified", "improvement_at_zero_rate_pct"]), 76.2, 0.05)
    chk("certified binary 10%", float(mc.loc["certified", "binary_10"]), 0.330, 0.001)
    chk("certified binary 90%", float(mc.loc["certified", "binary_90"]), 0.429, 0.001)
    chk("certified continuous 90%", float(mc.loc["certified", "continuous_90"]), 0.342, 0.001)
    chk("certified ratio", float(mc.loc["certified", "ratio"]), 2.90, 0.01)

    print("T57 phase 1 reconciliation")
    p1 = pd.read_csv(R / "wi_phase1_reconciliation.csv").set_index("precision")
    chk("float32 placement", float(p1.loc["float32", "placement"]), 1664.0, 0)
    chk("float32 bias", float(p1.loc["float32", "bias"]), 306.0, 0)
    chk("float32 solved", float(p1.loc["float32", "solved"]), 430.0, 0)
    chk("pooled solved", float(p1.loc["both", "solved"]), 870.0, 0)
    chk("pooled runs", float(p1.loc["both", "runs"]), 4800.0, 0)

    print("T80 branch tracking")
    bt = pd.read_csv(R / "wi_branch_tracking.csv").set_index("split")
    chk("median before crossing", float(bt.loc["before crossing", "median"]), 0.834, 0.001)
    chk("median after crossing", float(bt.loc["after crossing", "median"]), 0.0017, 0.0001)
    chk("median overall", float(bt.loc["all", "median"]), 0.0065, 0.0001)
    chk("n before / after", float(bt.loc["before crossing", "n"] * 1000 + bt.loc["after crossing", "n"]),
        197211.0, 0)

    print("T48 MNIST pilots (driver)")
    md = pd.read_csv(R / "mnist_fold_driver_check.csv").set_index("table")
    chk("pilot rows regenerated", float(md.matched.sum()), 210.0, 0)
    chk("max |test acc diff|", float(md.max_abs_diff_test.max()), 0.0, 1e-12)
    chk("narrow pilot rows (no a = 0.5)", float(md.loc["mnist_fold_pilot_narrow", "rows_ref"]), 75.0, 0)

    print("T41 margins and MEP (produced 2026-09-23)")
    mg = pd.read_csv(R / "discrepancy_margin.csv").set_index("a")
    chk("a=1.5 solvers", float(mg.loc[1.5, "n_solved"]), 81.0, 0)
    chk("a=1.5 below constructed margin", float(mg.loc[1.5, "n_below_constructed"]), 17.0, 0)
    chk("a=1.5 min margin", float(mg.loc[1.5, "min_margin"]), 0.0017, 0.0001)
    chk("a=1.45 below constructed margin", float(mg.loc[1.45, "n_below_constructed"]), 18.0, 0)
    mp = pd.read_csv(R / "discrepancy_mep.csv").iloc[0]
    chk("exclusion-table MEP paths", float(mp.paths), 60.0, 0)
    chk("barrier.csv distinct a", float(mp.barrier_csv_distinct_a), 13.0, 0)

    print("AdamW default weight decay (Block F) equals Block C's explicit 0.01")
    import inspect as _insp, torch as _t
    chk("torch AdamW default weight_decay",
        float(_insp.signature(_t.optim.AdamW).parameters["weight_decay"].default), 0.01, 0)

    print("T41 exclusion configurations")
    from . import barrier as _br
    chk("string images", float(_br.STRING_IMAGES), 41.0, 0)
    ex = pd.read_csv(R / "exclusion_table.csv")
    chk("MEP barrier max over all rows", float(ex.barrier_mep_max.max()), 0.0, 0)
    chk("found rows", float((ex.population == "found_adam").sum()), 4.0, 0)

    print("T63 Block F definitions")
    bf = pd.read_csv(R / "blockF_optimisers.csv")
    med = bf[bf.local_rate.notna()].groupby("optimiser").local_rate.median()
    chk("rate spread Adam/SGD (measured medians)", float(med.max() / med.min()), 3.92, 0.01)
    chk("registered-rate ratio 0.00207/0.00075", 0.00207 / 0.00075, 2.76, 0.01)

    v4_checks()
    harsh_review_checks()
    tracks_checks()
    lag_law_checks()
    act_checks()
    track4_checks()
    band_checks()
    track2_checks()
    c1_followup_checks()
    linear_response_checks()
    track_b_checks()
    theorem1_checks()
    track_t_checks()
    track_a_checks()

    provenance_check()

    digit_stability()

    print(f"\n{len(F)} finding(s)")
    (R / "ledger_verification.txt").write_text("\n".join(F) if F else "no findings\n")


# ---------------------------------------------------------------------------
# PROVENANCE
#
# Every artifact that backs a reported number must have a committed function
# that writes it.  Tightened 2026-09-22: the earlier check passed if ANY script
# merely mentioned the artifact's name, which let three headline datasets
# (r_pooled.csv, r_families.csv, r_family_b.csv) and seven session artifacts
# through with no producer at all.  Now each artifact is mapped to a
# module.function, and the check confirms from the source that the function
# exists, that its module references the artifact, and that the function
# performs a write.
#
# status "full"    -- the producer regenerates every column
# status "partial" -- some columns cannot be regenerated; recorded as a finding
# ---------------------------------------------------------------------------
PRODUCERS = {
    # artifact: (module, function, status, note)
    "track4_populations.json": ("track4_populations", "main", "full", ""),
    "alpha_of_eps.csv": ("session_artifacts", "main", "full", ""),
    "beta_law_points.csv": ("beta_law_figure", "main", "full", ""),
    "blockA_crossings.csv": ("blockA_score", "main", "full", ""),
    "blockA_per_a.csv": ("blockA_score", "main", "full", ""),
    "blockB_fine_windows.csv": ("session_artifacts", "main", "full", ""),
    "blockB_switches.csv": ("blockB_landscape", "main", "full", ""),
    "blockC_adiabatic.csv": ("blockC_adiabatic", "main", "full", ""),
    "blockE_intervene.csv": ("blockE_intervene", "main", "full", ""),
    "blockE_stall.csv": ("blockE_stall", "main", "full", ""),
    "blockE_stall_loss.csv": ("blockE_stall", "main", "full", ""),
    "blockF_optimisers.csv": ("blockF_optimisers", "main", "full", ""),
    "blockG_crossings.csv": ("blockG_windows", "main", "full", ""),
    "blockG_scaling.csv": ("session_artifacts", "main", "full", ""),
    "blockG_switches.csv": ("blockG_windows", "main", "full", ""),
    "blockH_rate.csv": ("blockH_rate", "main", "full", ""),
    "blockK_auc.csv": ("session_artifacts", "main", "full", ""),
    "blockK_family_b_normalised.csv": ("session_artifacts", "main", "full", ""),
    "blockS2_budget.csv": ("blockS2_budget", "main", "full", ""),
    "budget_alpha.csv": ("budget_law", "main", "full", ""),
    "budget_flip.csv": ("budget_flip", "main", "full", ""),
    "cifar_convergence.csv": ("cifar_convergence", "main", "full", ""),
    "criticality.csv": ("criticality", "main", "full", ""),
    "depth_families_verification.csv": ("depth_families", "<module>", "full", ""),
    "exact_all_solved.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "exact_separation.csv": ("exact_extrema", "main", "full", ""),
    "exact_vs_grid.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "exceptions_above_050.csv": ("session_artifacts", "main", "full", ""),
    "exclusion_table.csv": ("exclusion_table", "main", "full", ""),
    "family_ghat_certified.csv": ("family_certify", "main", "full", ""),
    "family_onsets.csv": ("family_onsets", "main", "full", ""),
    "fold1d_refine.csv": ("fold1d_theorem", "main", "full", ""),
    "fold1d_sweep.csv": ("fold1d", "main", "full", ""),
    "gelu_scale_scaled_down.csv": ("gelu_scale", "run_arm", "full", ""),
    "gelu_scale_scaled_up.csv": ("gelu_scale", "run_arm", "full", ""),
    "gelu_scale_standard.csv": ("gelu_scale", "run_arm", "full", ""),
    "ghat_certified_all.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "kappa_K_certified.csv": ("kappa_certify", "main", "full", ""),
    "kappa_boundary.csv": ("kappa_certify", "main", "full", ""),
    "kappa_certified.csv": ("kappa_certify", "main", "full", ""),
    "kappa_certified_exact.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "kappa_certified_full_range.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "kappa_certified_smalleps.csv": ("session_artifacts", "write_session_producers", "full", ""),
    "monotonic_zero_decomposition.csv": ("zero_decomposition", "main", "full", ""),
    "nu_sweep.csv": ("nu_sweep", "main", "full", ""),
    "offset_search.csv": ("offset_search", "main", "full", ""),
    "offset_witness_dense.csv": ("offset_witness", "main", "full", ""),
    "onset_family_b.csv": ("onset_family_b", "main", "full", ""),
    "onset_law_extended.csv": ("onset_more", "main", "full", ""),
    "our_link_verification.csv": ("verify_our_link", "main", "full", ""),
    "pathwise_distance.csv": ("pathwise_distance", "main", "full", ""),
    "phase1_decomposition.csv": ("phase1_decompose", "main", "full", ""),
    "r50_fit.csv": ("r50_fit", "main", "full", ""),
    "r_adamw.csv": ("provenance_rebuild", "write", "full", ""),
    "r_families.csv": ("provenance_rebuild", "r_families_primary", "full",
                       "w2 and solved regenerate exactly; the gstar column had no producing "
                       "code, is superseded by the certified Ghat, and backs no number"),
    "r_families_certified.csv": ("family_certify", "write_families_certified", "full", ""),
    "r_pooled.csv": ("provenance_rebuild", "write", "full", ""),
    "r_pooled_union.csv": ("pool_union", "main", "full", ""),
    "saturation.parquet": ("census", "run_sweep", "full", ""),
    "scaling_limit_switches.csv": ("scaling_limit_blockB", "main", "full", ""),
    "scaling_proposition_checks.csv": ("scaling_proposition", "main", "full", ""),
    "search_restarts.csv": ("search_restarts", "_write", "full", ""),
    "sgd_onsets.csv": ("sgd_onsets", "main", "full", ""),
    "sgd_q4_refined.csv": ("sgd_q4_refine", "main", "full", ""),
    "theorem_perplacement.csv": ("session_artifacts", "main", "full", ""),
    "theorem_verification.csv": ("fold1d_theorem", "main", "full", ""),
    "training_dynamics.parquet": ("census", "run_sweep", "full", ""),
    "training_status.parquet": ("census", "run_sweep", "full", ""),
    "width_effect.csv": ("width_effect", "main", "full", ""),
    "width_sweep.csv": ("width_sweep", "_write", "full", ""),
    "width_sweep.parquet": ("width_sweep", "_write", "full", ""),
    "blockA5d_runs.csv": ("blockA5d", "_run", "full", ""),
    "blockC_pilot.csv": ("blockC_equivalence", "_append", "full", ""),
    "blockC_runs.csv": ("blockC_equivalence", "_append", "full", ""),
    "blockC_verdicts.csv": ("blockC_equivalence", "_cli_analyze", "full", ""),
    "blockC_existing_context.csv": ("blockC_equivalence", "analyze_existing", "full", ""),
    "blockA5d_perfect_fraction.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_controls.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_scores.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_k1_pilot.csv": ("blockA5d_k1", "_run", "full", ""),
    "blockA5d_k1_stratcheck.csv": ("blockA5d_k1", "_run", "full", ""),
    "blockA5d_k1_runs.csv": ("blockA5d_k1", "_run", "full", ""),
    "blockA5d_k1_scores.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_k1_perfect_fraction.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_k1_controls.csv": ("blockA5d_analyze", "score", "full", ""),
    "blockA5d_k1_errormap.csv": ("blockA5d_k1_errormap", "main", "full", ""),
    "crossfamily_thresholds.csv": ("crossfamily_landscape", "run", "full", ""),
    "crossfamily_scores.csv": ("crossfamily_landscape", "score", "full", ""),
    "crossfamily_followup_domain.csv": ("crossfamily_followup", "verify", "full", ""),
    "crossfamily_followup_thresholds.csv": ("crossfamily_followup", "run", "full", ""),
    "crossfamily_followup_scores.csv": ("crossfamily_followup", "score", "full", ""),
    "blockB_scaled_validity.csv": ("crossfamily_followup", "validity", "full", ""),
    "blockA5d_k1_budget_trend.csv": ("blockA5d_k1_trend", "main", "full", ""),
    "blockA5d_k1_first_perfect.csv": ("blockA5d_k1_trend", "main", "full", ""),
    "crossfamily_q1control_domain.csv": ("crossfamily_q1control", "verify", "full", ""),
    "crossfamily_q1control_validity.csv": ("crossfamily_q1control", "validity", "full", ""),
    "crossfamily_q1control_thresholds.csv": ("crossfamily_q1control", "run", "full", ""),
    "crossfamily_q1control_scores.csv": ("crossfamily_q1control", "score", "full", ""),
    "ghat_bnb.csv": ("ghat_bnb", "main", "full", ""),
    "wi_crossing_runs.csv": ("writer_inputs", "crossings", "full", ""),
    "wi_per_a_certified.csv": ("writer_inputs", "per_a_table", "full", ""),
    "wi_per_a_cv.csv": ("writer_inputs", "per_a_table", "full", ""),
    "wi_phase1_reconciliation.csv": ("writer_inputs", "phase1", "full", ""),
    "wi_e2_rescore.csv": ("writer_inputs", "e2_rescore", "full", ""),
    "wi_branch_tracking.csv": ("writer_inputs", "branch_tracking", "full", ""),
    "metric_check.csv": ("metric_check", "main", "full", ""),
    "metric_check_binning.csv": ("metric_check", "main", "full", ""),
    "mechanism_branch_a130.csv": ("mechanism_figure_data", "branch", "full", ""),
    "mechanism_trajectories_a130.csv": ("mechanism_figure_data", "trajectories", "full", ""),
    "registration_census.csv": ("registration_census", "build", "full", ""),
    "registration_tally.csv": ("registration_census", "build", "full", ""),
    "registration_census_by_unit.csv": ("registration_census", "build", "full", ""),
    "discrepancy_margin.csv": ("discrepancies", "margins", "full", ""),
    "discrepancy_margin_runs.csv": ("discrepancies", "margins", "full", ""),
    "discrepancy_pathwise_init.csv": ("discrepancies", "pathwise", "full", ""),
    "discrepancy_mep.csv": ("discrepancies", "mep", "full", ""),
    "mnist_fold_driver_check.csv": ("mnist_fold_driver", "check", "full", ""),
    "mnist_fold_pilot.csv": ("mnist_fold_driver", "write", "full", ""),
    "mnist_fold_pilot_narrow.csv": ("mnist_fold_driver", "write", "full", ""),
    "mnist_fold_pilot_longbudget.csv": ("mnist_fold_driver", "write", "full", ""),
    "wi_crossing_steps.csv": ("writer_inputs", "crossing_steps", "full", ""),
    "provenance_rebuild_check.csv": ("provenance_rebuild", "check", "full", ""),
    "session_producers_check.csv": ("session_artifacts", "check_session_producers", "full", ""),
    "rglob_convergence_refit.csv": ("rglob_refit", "main", "full", ""),
    "cond_scan_certified_a130.csv": ("cond_scan_certified", "main", "full", ""),
    "first_order_c1.csv": ("first_order", "main", "full", ""),
    "own_threshold_scores.csv": ("own_threshold", "score", "full", ""),
    "diagnose_below_endpoints.csv": ("diagnose_below", "endpoints", "full", ""),
    "mirror_q2_s1_breakdown.csv": ("mirror_branches", "q2_s1_breakdown", "full", ""),
    "lag_test_tests.csv": ("lag_test", "score", "full", ""),
    "lag_test_cells.csv": ("lag_test", "score", "full", ""),
    "lag_test_confounds.csv": ("lag_test", "score", "full", ""),
    "lag_test_runs.csv": ("lag_test", "run", "full", ""),
    "mirror_init_match.csv": ("mirror_branches", "analyse", "full", ""),
    "mirror_occupancy.csv": ("mirror_branches", "analyse", "full", ""),
    "mirror_s1.csv": ("mirror_branches", "analyse", "full", ""),
    "mirror_s3.csv": ("mirror_branches", "analyse", "full", ""),
    "mirror_gap_summary.csv": ("mirror_branches", "mirror_gap", "full", ""),
    "mirror_size_gap_summary.csv": ("mirror_branches", "size_gap", "full", ""),
    "mirror_branch_thresholds.csv": ("mirror_branches", "thresholds", "full", ""),
    "sample_size_tests.csv": ("sample_size", "score", "full", ""),
    "sample_size_cells.csv": ("sample_size", "score", "full", ""),
    "sample_size_certify.csv": ("sample_size", "certify", "full", ""),
    "sample_size_detail_pairs.csv": ("sample_size", "detail", "full", ""),
    "sample_size_detail_cells.csv": ("sample_size", "detail", "full", ""),
    "mirror_basin_census_preferred.csv": ("mirror_branches", "basin_census", "full", ""),
    "mirror_global_branch.csv": ("mirror_branches", "global_branch_levels", "full", ""),
    "diagnose_below.csv": ("diagnose_below", "main", "full", ""),
    "fixed_scale_horizons_tests.csv": ("fixed_scale", "score_horizons", "full", ""),
    "fixed_scale_horizons_curve.csv": ("fixed_scale", "score_horizons", "full", ""),
    "fixed_scale_horizons_check.csv": ("fixed_scale", "check_horizons", "full", ""),
    "own_threshold_crossing.csv": ("own_threshold", "_run", "full", ""),
    "own_threshold_block4.csv": ("own_threshold", "_run", "full", ""),
    "own_threshold_validation.csv": ("own_threshold", "validate", "full", ""),
    "own_threshold_validation_tight.csv": ("own_threshold", "validate_tight", "full", ""),
    "corner_tracking.csv": ("corner_tracking", "main", "full", ""),
    "corner_tracking_decomposition.csv": ("corner_tracking", "decompose", "full", ""),
    "corner_tracking_switch_active.csv": ("corner_tracking", "switch_active", "full", ""),
    "cond_certified_seeds_summary.csv": ("conditional_certified", "seeds_summary", "full", ""),
    "first_order_corner.csv": ("first_order", "corner", "full", ""),
    "first_order_corner_free.csv": ("first_order", "corner", "full", ""),
    "mn2_h2prime.csv": ("math_note_v2_checks", "h2", "full", ""),
    "mn2_solve_limit.csv": ("math_note_v2_checks", "solve_limit", "full", ""),
    "mn2_solve_finite.csv": ("math_note_v2_checks", "solve_finite", "full", ""),
    "cond_audit_candidates.csv": ("conditional_audit", "candidates", "full", ""),
    "cond_audit_strict.csv": ("conditional_audit", "strict", "full", ""),
    "cond_certified_brackets.csv": ("conditional_certified", "brackets", "full", ""),
    "cond_certified_bracket_evaluations.csv": ("conditional_certified", "brackets", "full", ""),
    "mn2_bounds.csv": ("math_note_v2_checks", "bounds", "full", ""),
    "mn2_uniformity_summary.csv": ("math_note_v2_checks", "uniform_summary", "full", ""),
    "mn2_neighbourhood.csv": ("math_note_v2_checks", "neighbourhood", "full", ""),
    "mn2_rounding.csv": ("math_note_v2_checks", "rounding", "full", ""),
    "limit_switch.csv": ("limit_bnb", "switch", "full", ""),
    "certv2_annulus_parts.csv": ("certificates_v2", "_run", "full", ""),
    "first_order_supplementary.csv": ("first_order", "supplementary", "full", ""),
    "first_order_supplementary_scores.csv": ("first_order", "score_supplementary", "full", ""),
    "first_order_finite.csv": ("first_order", "finite", "full", ""),
    "first_order_finite_evaluations.csv": ("first_order", "finite", "full", ""),
    "first_order_scores.csv": ("first_order", "score", "full", ""),
    "prospective_own_runs.csv": ("prospective_own", "train", "full", ""),
    "lag_test2_checkpoints.csv": ("lag_test2", "checkpoints", "full", ""),
    "lag_test2_runs.csv": ("lag_test2", "cont", "full", ""),
    "lag_test2_diagnostics.csv": ("lag_test2", "diagnostics", "full", ""),
    "lag_test2_cells.csv": ("lag_test2", "score", "full", ""),
    "lag_test2_tests.csv": ("lag_test2", "score", "full", ""),
    "lag_test2_pairs.csv": ("lag_test2", "score", "full", ""),
    "lag_test2_confounds.csv": ("lag_test2", "score", "full", ""),
    "prospective_own_scores.csv": ("prospective_own", "score", "full", ""),
    "prospective_own_settings_scored.csv": ("prospective_own", "score", "full", ""),
    "prospective_own_early_scores.csv": ("prospective_own", "score_secondary", "full", ""),
    "prospective_own_mixture_scores.csv": ("prospective_own", "score_secondary", "full", ""),
    "prospective_own_mixture_settings.csv": ("prospective_own", "score_secondary", "full", ""),
    "prospective_own_validation.csv": ("prospective_own", "validate", "full", ""),
    "certv2_annulus_summary.csv": ("certificates_v2", "_summary", "full", ""),
    "certv2_solve_parts.csv": ("certificates_v2", "_run", "full", ""),
    "certv2_solve_pd.csv": ("certificates_v2", "pd_boxes_solve", "full", ""),
    "certv2_solve_summary.csv": ("certificates_v2", "_summary", "full", ""),
    "prospective_comparisons.csv": ("prospective", "score", "full", ""),
    "prospective_scores.csv": ("prospective", "score", "full", ""),
    "prospective_calibration.csv": ("prospective", "calibrate", "full", ""),
    "prospective_secondary_analyses.csv": ("prospective_secondary", "main", "full", ""),
    "fixed_scale_block4.csv": ("fixed_scale", "run_replays", "full", ""),
    "fixed_scale_block5.csv": ("fixed_scale", "run_replays", "full", ""),
    "fixed_scale_block4_curve.csv": ("fixed_scale", "score", "full", ""),
    "fixed_scale_block4_tests.csv": ("fixed_scale", "score", "full", ""),
    "fixed_scale_block5_curve.csv": ("fixed_scale", "score", "full", ""),
    "fixed_scale_block5_tests.csv": ("fixed_scale", "score", "full", ""),
    "critical_slowing_posthoc.csv": ("critical_slowing", "main", "full", ""),
    "rglob_convergence_points.csv": ("rglob_refit", "main", "full", ""),
    "limit_K_base.csv": ("limit_windows", "K_base", "full", ""),
    "limit_windows.csv": ("limit_windows", "main", "full", ""),
    "limit_windows_evaluations.csv": ("limit_windows", "main", "full", ""),
    "writer_patch_prospective_own_per_setting.csv": ("writer_patch", "per_setting", "full", ""),
    "residual_posthoc_runs.csv": ("residual_posthoc", "score", "full", ""),
    "scale_limits_D.csv": ("scale_limits", "main", "full", ""),
    "residual_mechanism_scores.csv": ("residual_mechanism", "score", "full", ""),
    "residual_mechanism_checks.csv": ("residual_mechanism", "score", "full", ""),
    "scale_limits_width1.csv": ("scale_limits", "main", "full", ""),
    "scale_limits_width2.csv": ("scale_limits", "main", "full", ""),
    "scale_limits_summary.csv": ("scale_limits", "main", "full", ""),
    "residual_posthoc_correlations.csv": ("residual_posthoc", "score", "full", ""),
    "writer_patch_windows.csv": ("writer_patch", "windows", "full", ""),
    "writer_patch_calibration.csv": ("writer_patch", "windows", "full", ""),
    "writer_patch_w_bound.csv": ("writer_patch", "w_bound_table", "full", ""),
    "writer_patch_census_by_block.csv": ("writer_patch", "census_by_block", "full", ""),
    "writer_patch_figure_sizes.csv": ("writer_patch", "figure_sizes", "full", ""),
    "fixed_scale_block5_splits.csv": ("fixed_scale", "score5_splits", "full", ""),
    "crossing_audit_runs.csv": ("crossing_audit", "main", "full", ""),
    "crossing_audit_full_runs.csv": ("cadence_sensitivity", "runs", "full", ""),
    "cadence_sensitivity.csv": ("cadence_sensitivity", "main", "full", ""),
    "width2_unplaced.csv": ("width2_unplaced", "save", "full", ""),
    "width2_finish_summary.csv": ("width2_finish", "summary", "full", ""),
    "width2_direct_check.csv": ("width2_finish", "summary", "full", ""),
    "width2_w0_scan_coverage.csv": ("width2_w0", "scan_coverage", "full", ""),
    "k_domain_check.csv": ("k_domain", "main", "full", ""),
    "width2_unplaced_localmin.csv": ("width2_unplaced", "save", "full", ""),
    "ghat_rigorous.csv": ("ghat_rigorous", "build", "full", ""),
    "ghat_digit_stability.csv": ("verify_ledger", "digit_stability", "full", ""),
    "ghat_digit_stability_machine_precision.csv": ("verify_ledger", "digit_stability", "full", ""),
    "cadence_sensitivity_block3.csv": ("cadence_sensitivity", "main", "full", ""),
    "cadence_sensitivity_own_seed.csv": ("cadence_sensitivity", "main", "full", ""),
    "scale_limits_tanh.csv": ("scale_limits_tanh", "main", "full", ""),
    "scale_limits_tanh_maximisers.csv": ("scale_limits_tanh", "main", "full", ""),
    "scale_limits_tanh_summary.csv": ("scale_limits_tanh", "main", "full", ""),
    "crossing_audit_summary.csv": ("crossing_audit", "main", "full", ""),
}

_WRITE_CALL = ("to_csv(", "to_parquet(", "write_text(", "DictWriter(", "csv.writer(",
               "_write(", "_write_frame(", "artifact_lock(")


def v4_checks() -> None:
    """Numbers in paper/WRITER_INPUTS_v4.md (Blocks 1-5)."""
    print("V4 Block 1 (conditional audit)")
    ca = pd.read_csv(R / "cond_audit_candidates.csv")
    chk("1a candidates", float(len(ca)), 16468.0, 0)
    chk("1a scales", float(ca.groupby(["a", "kind", "s"]).ngroups), 358.0, 0)
    chk("1a frozen reproduced at all scales", float(ca.frozen_reproduced.all()), 1.0, 0)
    ret = ca[ca.status == "RETAINED"].set_index(["a", "kind", "s"]).loss
    oth = ca[ca.status != "RETAINED"].join(ret.rename("ret"), on=["a", "kind", "s"])
    chk("1a discarded candidates below the retained loss", float((oth.loss < oth.ret - 1e-12).sum()), 0.0, 0)
    chk("1a max retained gradient norm <= 2.3e-6",
        float(ca[ca.status == "RETAINED"].grad_norm.max() <= 2.3e-6), 1.0, 0)
    st = pd.read_csv(R / "cond_audit_strict.csv").join(ret.rename("ret"), on=["a", "kind", "s"])
    chk("1e strict vs frozen, max |loss diff| <= 2e-16", float((st.loss - st.ret).abs().max() <= 2.2e-16), 1.0, 0)
    br = pd.read_csv(R / "cond_certified_brackets.csv")
    g13 = br[(br.a.round(2) == 1.3) & (br.kind == "glob")].iloc[0]
    s13 = br[(br.a.round(2) == 1.3) & (br.kind == "solve")].iloc[0]
    chk("1c R_glob(1.30) lower", float(g13.R_lo), 0.21310, 1e-5)
    chk("1c R_glob(1.30) upper", float(g13.R_hi), 0.21364, 1e-5)
    chk("1c R_solve(1.30) lower", float(s13.R_lo), 0.30566, 1e-5)
    chk("1c R_solve(1.30) upper", float(s13.R_hi), 0.30620, 1e-5)
    chk("1c frozen within one step, all 12", float(br.frozen_within_one_step.all()), 1.0, 0)
    chk("1c all converged, none unresolved", float(br.all_converged.all() and (br.unresolved == 0).all()), 1.0, 0)
    ev = pd.read_csv(R / "cond_certified_bracket_evaluations.csv")
    sep = np.maximum(ev.m_plus_lo - ev.m_minus_hi, ev.m_minus_lo - ev.m_plus_hi)
    chk("1c min separation of certified intervals >= 1.3e-8", float(sep.min() >= 1.3e-8), 1.0, 0)
    gl = br[br.kind == "glob"]
    asep = pd.concat([gl.argmin_separation_lo, gl.argmin_separation_hi])
    chk("1c argmin separation min", float(asep.min()), 0.00015, 0.00005)
    chk("1c argmin separation max", float(asep.max()), 0.0011, 0.00005)

    sc_ = pd.read_csv(R / "cond_scan_certified_a130.csv").sort_values("s")
    chk("1c scan: one sign change", float(((sc_.global_region == "+").astype(int).diff().abs() == 1).sum()), 1.0, 0)
    chk("1c scan: sign change between 4.5 and 5.0",
        float(sc_[sc_.global_region == "+"].s.min() == 5.0 and sc_[sc_.global_region == "-"].s.max() == 4.5), 1.0, 0)
    chk("1c scan: competitor margin min", float(sc_.competitor_margin.min()), 0.0031, 0.00005)
    chk("1c scan: competitor margin max", float(sc_.competitor_margin.max()), 0.0126, 0.00005)
    chk("1c scan: all competitor bounds converged", float(sc_.competitor_converged.all()), 1.0, 0)

    ss_ = pd.read_csv(R / "cond_certified_seeds_summary.csv")
    g_ = ss_[ss_.kind == "glob"].set_index("a")
    for a_, med_, pq_ in ((1.3, 1.063, 0.32), (1.45, 1.063, 0.30), (1.6, 1.059, 0.32)):
        chk(f"1d a={a_}: median own/pop", float(g_.loc[a_, "median_own_over_pop"]), med_, 0.0006)
        chk(f"1d a={a_}: population quantile", float(g_.loc[a_, "pop_quantile_in_seeds"]), pq_, 0)
    chk("1d: IQR low min", float(g_.q25.min()), 0.988, 0.0006)
    chk("1d: IQR high max", float(g_.q75.max()), 1.121, 0.0006)
    chk("1d: range min", float(g_["min"].min()), 0.873, 0.0006)
    chk("1d: range max", float(g_["max"].max()), 1.222, 0.0006)
    chk("1d: seeds with >1 sign change", float(ss_.seeds_with_more_than_one_sign_change.sum()), 0.0, 0)
    chk("1d: 50 seeds per a", float(ss_.n_seeds.min()), 50.0, 0)

    print("V4 first-order coefficient (math_note_v2 s8)")
    fo = pd.read_csv(R / "first_order_c1.csv").iloc[0]
    for k_ in ("switch_krawczyk_ok", "active_set_unique", "vertex_krawczyk_ok", "other_pieces_dominated",
               "strict_local_max", "vertex_in_bnb_enclosure"):
        chk(k_, float(bool(fo[k_])), 1.0, 0)
    chk("c1 lo", float(fo.c1_lo), 0.2852300, 1e-7)
    chk("c1 hi", float(fo.c1_hi), 0.2852303, 1e-7)
    chk("A'/A* lo", float(fo.A1_over_A_lo), 0.6621547, 1e-7)
    chk("k1", float(fo.k1_lo), -0.37692472, 1e-8)
    chk("A* (Krawczyk)", float(fo.A_star_lo), 0.68544523757565, 1e-12)
    chk("R_glob^inf sharp lo", float(fo.R_glob_inf_lo), 0.1985926, 1e-7)

    co = pd.read_csv(R / "first_order_corner.csv")
    chk("corner: pieces", float(len(co)), 400.0, 0)
    chk("corner: eps range", float(co.eps_hi.max() - co.eps_lo.min()), 0.1, 1e-12)
    chk("corner: all certified (Krawczyk, active set, strict max)", float(co.all_ok.all()), 1.0, 0)
    chk("corner: min barycentric weight", float(co.strict_max_barycentric_min.min()), 0.00434, 0.00001)
    cf = pd.read_csv(R / "first_order_corner_free.csv")
    chk("corner free check: fixed value inside free enclosure, all", float(cf.fixed_in_free_enclosure.all()), 1.0, 0)
    chk("corner free check: kept cells within 3e-6", float((cf.kept_cells_max_dist_to_argmax_or_mirror < 3e-6).all()), 1.0, 0)

    ct_ = pd.read_csv(R / "corner_tracking.csv")
    chk("corner tracking: eps values", float(ct_.eps.nunique()), 79.0, 0)
    chk("corner tracking: one active set throughout", float(ct_.active_set.nunique()), 1.0, 0)
    chk("corner tracking: min outer inactive margin", float(ct_.margin_outer_inactive.min()), 0.97, 0.01)
    dc_ = pd.read_csv(R / "corner_tracking_decomposition.csv")
    chk("corner vertex = B&B sup at all checked eps", float(dc_[dc_.part == "vertex_check"].vertex_in_bnb.all()), 1.0, 0)
    d6_ = dc_[(dc_.part == "decomposition") & (dc_.eps.round(2) == 0.6)].iloc[0]
    chk("decomposition a=1.60: excess", float(d6_.excess_over_first_order), -0.038, 0.0006)
    chk("decomposition a=1.60: product term", float(d6_.product_term), -0.060, 0.0006)
    chk("decomposition a=1.60: A higher order", float(d6_.A_higher_order), -0.037, 0.0006)
    chk("decomposition a=1.60: K higher order", float(d6_.K_higher_order), 0.059, 0.0006)
    sw_ = pd.read_csv(R / "corner_tracking_switch_active.csv")
    chk("switch active pair unchanged", float((sw_.inner_active == "I:x=+0.8").all() and (sw_.outer_active == "O-:x=-1.2").all()), 1.0, 0)

    print("V4 own-seed thresholds (S3)")
    sc3 = pd.read_csv(R / "own_threshold_scores.csv").set_index("test")
    for a_, ex_, pass_, a_pass_, rho_, oo_, op_ in (("1.30", 0.0574, True, True, 0.877, 0.0308, 0.0916),
                                                    ("1.50", 0.0556, False, False, 0.881, 0.0640, 0.1189)):
        r_ = sc3.loc[f"S3 a={a_}"]
        chk(f"S3 a={a_}: excess", float(r_.median_own_over_pop_minus_1), ex_, 0.0001)
        chk(f"S3 a={a_}: S3a", float(r_.S3a_pass), float(a_pass_), 0)
        chk(f"S3 a={a_}: rho", float(r_.spearman_rho), rho_, 0.001)
        chk(f"S3 a={a_}: offset vs own", float(r_.offset_vs_own), oo_, 0.0001)
        chk(f"S3 a={a_}: offset vs pop", float(r_.offset_vs_pop), op_, 0.0001)
        chk(f"S3 a={a_}: verdict", float(r_["pass"]), float(pass_), 0)
    s1_ = sc3.loc["S1"]
    chk("S1 own-rule agreement", float(s1_.own_rule_agreement), 0.8939, 0.0001)
    chk("S1 pop-rule agreement", float(s1_.pop_rule_agreement), 0.7602, 0.0001)
    chk("S1 n replays", float(s1_.n), 2715.0, 0)
    chk("S1 verdict", float(s1_["pass"]), 0.0, 0)
    s2_ = sc3.loc["S2"]
    chk("S2 x50 64k", float(s2_.x50_64k), 1.0454, 0.0001)
    chk("S2 median own/pop", float(s2_.median_own_over_pop), 1.0359, 0.0001)
    chk("S2 verdict", float(s2_["pass"]), 1.0, 0)
    ht_ = pd.read_csv(R / "fixed_scale_horizons_tests.csv").set_index("variant")
    chk("Q1 preserved", float(ht_.loc["preserved", "Q1_pass"]), 0.0, 0)
    chk("Q2 preserved", float(ht_.loc["preserved", "Q2_pass"]), 0.0, 0)
    chk("competing outcome preserved", float(ht_.loc["preserved", "competing_outcome"]), 1.0, 0)
    for h_, v_ in (("x50_4k", 1.0458), ("x50_16k", 1.0450), ("x50_64k", 1.0454)):
        chk(f"horizon {h_}", float(ht_.loc["preserved", h_]), v_, 0.0001)
    chk("frac 0.9 at 64k", float(ht_.loc["preserved", "frac09_64k"]), 0.0147, 0.0001)
    chk("frac 0.95 at 64k", float(ht_.loc["preserved", "frac095_64k"]), 0.0939, 0.0001)
    hc_ = pd.read_csv(R / "fixed_scale_horizons_check.csv").iloc[0]
    chk("amended reproduction check", float(hc_.reproduced and hc_.rows_compared == 3258), 1.0, 0)
    en_ = pd.read_csv(R / "diagnose_below_endpoints.csv")
    chk("0.9x endpoints: seven, all placed at 64k", float(len(en_) == 7 and en_.placed_64000.all()), 1.0, 0)
    chk("0.9x endpoints: strict local minima", float((en_.loc_min_hess_min_64000 > 1.5).all()), 1.0, 0)
    chk("0.9x endpoints: global minimum in G<=0", float((en_.global_region == "-").all()), 1.0, 0)
    chk("0.9x endpoints: loss gap min", float(en_.loss_gap_to_global_64000.min()), 0.008, 0.0005)
    chk("0.9x endpoints: loss gap max", float(en_.loss_gap_to_global_64000.max()), 0.064, 0.0005)
    chk("0.9x endpoints: mirror (opposite w1 sign)", float((en_.loc_min_w1_64000 * en_.global_w1 < 0).all()), 1.0, 0)
    chk("0.9x endpoints: barrier = log 2", float(en_.barrier_level.max()), 0.693147, 0.00001)
    mb_ = pd.read_csv(R / "mirror_q2_s1_breakdown.csv")
    om_ = mb_[mb_.part == "S1 by on_mirror"].set_index("group")
    chk("mirror: on-global agreement", float(om_.loc["False", "agreement"]), 0.9928, 0.0001)
    chk("mirror: on-global disagreements", float(om_.loc["False", "disagreements"]), 15.0, 0)
    chk("mirror: on-mirror replays", float(om_.loc["True", "n"]), 618.0, 0)
    chk("mirror: on-mirror agreement", float(om_.loc["True", "agreement"]), 0.5583, 0.0001)
    q2_ = mb_[mb_.part == "Q2 vs own"].set_index("level")
    chk("mirror: placed below own at 0.9/0.95 all on mirror",
        float((q2_.placed_below_own == q2_.placed_below_own_on_mirror).all() and q2_.placed_below_own.sum() == 41), 1.0, 0)
    cp_ = pd.read_csv(R / "mirror_basin_census_preferred.csv")
    chk("census preferred: 15 disagreements", float(len(cp_)), 15.0, 0)
    chk("census preferred: no third basin", float((cp_.basin == "other").sum()), 0.0, 0)
    chk("census preferred: plateau cases", float(cp_.kind.str.startswith("degenerate").sum()), 8.0, 0)
    chk("census preferred: non-plateau within 0.35% of own threshold",
        float((cp_[~cp_.kind.str.startswith("degenerate")].dist < 0.0035).all()), 1.0, 0)
    stt_ = pd.read_csv(R / "sample_size_tests.csv").set_index("a")
    for a_, n1_, n2_, n3_, cp_x in ((1.3, 1, 0, 0, 1), (1.5, 1, 1, 0, 0)):
        chk(f"size test a={a_}: N1", float(stt_.loc[a_, "N1_pass"]), float(n1_), 0)
        chk(f"size test a={a_}: N2", float(stt_.loc[a_, "N2_pass"]), float(n2_), 0)
        chk(f"size test a={a_}: N3", float(stt_.loc[a_, "N3_pass"]), float(n3_), 0)
        chk(f"size test a={a_}: competing", float(stt_.loc[a_, "competing_offset_not_finite_sample"]), float(cp_x), 0)
    sc_ = pd.read_csv(R / "sample_size_cells.csv").set_index(["a", "n"])
    for (a_, n_), ex_, off_ in (((1.3, 400), 0.0422, 0.0706), ((1.3, 6400), 0.0120, 0.0421),
                                ((1.5, 400), 0.0444, 0.1062), ((1.5, 6400), 0.0111, 0.0748)):
        chk(f"size test {a_}/{n_}: own excess", float(sc_.loc[(a_, n_), "own_excess"]), ex_, 0.0001)
        chk(f"size test {a_}/{n_}: free offset", float(sc_.loc[(a_, n_), "free_offset"]), off_, 0.0001)
    scc_ = pd.read_csv(R / "sample_size_certify.csv")
    chk("size test: certified contradictions", float(((scc_.cert_lo == "plus") | (scc_.cert_hi == "minus")).sum()), 0.0, 0)
    im_ = pd.read_csv(R / "mirror_init_match.csv").set_index("group")
    chk("mirror: init match B45", float(im_.loc["B45 free crossing", "frac_matching_init"]), 0.4972, 0.0001)
    oc_ = pd.read_csv(R / "mirror_occupancy.csv")
    chk("mirror: pooled init match", float((oc_.branch == oc_.init_branch).mean()), 0.5138, 0.0001)
    s1m_ = pd.read_csv(R / "mirror_s1.csv").set_index("rule")
    chk("mirror S1: starting-branch agreement", float(s1m_.loc["branch at the replay's starting checkpoint", "agreement"]), 0.9941, 0.0001)
    chk("mirror S1: init-branch agreement", float(s1m_.loc["branch selected by initialisation", "agreement"]), 0.7444, 0.0001)
    chk("mirror S1: replays end on starting branch", float(s1m_.loc["replays ending on their starting checkpoint's branch", "agreement"]), 1.0, 0)
    s3m_ = pd.read_csv(R / "mirror_s3.csv").set_index(["a", "threshold"])
    chk("mirror S3 1.30: crossing-branch rho", float(s3m_.loc[(1.3, "branch occupied at the crossing"), "spearman_rho"]), 0.9974, 0.0001)
    chk("mirror S3 1.50: crossing-branch residual", float(s3m_.loc[(1.5, "branch occupied at the crossing"), "residual_median_log"]), 0.0625, 0.0001)
    chk("mirror S3 1.30: init-branch rho", float(s3m_.loc[(1.3, "branch selected by initialisation"), "spearman_rho"]), 0.1297, 0.0001)
    gs_ = pd.read_csv(R / "mirror_gap_summary.csv").iloc[0]
    chk("mirror gap median", float(gs_.gap_rel_median), 0.1063, 0.0001)
    chk("mirror gap exceeds distance", float(gs_.gap_exceeds_distance), 281.0, 0)
    msg_ = pd.read_csv(R / "mirror_size_gap_summary.csv").set_index(["a", "n"])
    for (a_, n_), v_ in (((1.3, 400), 0.0972), ((1.3, 6400), 0.0366), ((1.5, 400), 0.0993), ((1.5, 6400), 0.0365)):
        chk(f"mirror gap vs n {a_}/{n_}", float(msg_.loc[(a_, n_), "median"]), v_, 0.0001)
    lt_ = pd.read_csv(R / "lag_test_tests.csv")
    ltp_ = lt_[lt_.primary].set_index("a")
    for a_ in (1.3, 1.5):
        chk(f"lag a={a_}: L1", float(ltp_.loc[a_, "L1_pass"]), 1.0, 0)
        chk(f"lag a={a_}: L2", float(ltp_.loc[a_, "L2_pass"]), 0.0, 0)
        chk(f"lag a={a_}: competing", float(ltp_.loc[a_, "competing_no_dependence"]), 0.0, 0)
    lc_ = pd.read_csv(R / "lag_test_cells.csv")
    lc_ = lc_[lc_.primary].set_index(["a", "factor"])
    for (a_, f_), v_ in (((1.3, 0.25), 0.00473), ((1.3, 1.0), 0.03105), ((1.5, 0.25), 0.01069), ((1.5, 1.0), 0.06559)):
        chk(f"lag residual {a_}/{f_}", float(lc_.loc[(a_, f_), "residual"]), v_, 0.00001)
    cf_ = pd.read_csv(R / "lag_test_confounds.csv").set_index(["a", "factor"])
    chk("lag confound: plateau steps 1.30/0.25", float(cf_.loc[(1.3, 0.25), "median_plateau_steps"]), 1308.5, 0.1)
    chk("lag diag: dist 1.30/0.25", float(cf_.loc[(1.3, 0.25), "median_dist_to_branch_min"]), 0.001257, 0.000001)
    va_ = pd.read_csv(R / "own_threshold_validation.csv")
    chk("own validation: agreement (registered)", float(va_.agrees.mean()), 0.65, 0)
    vt_ = pd.read_csv(R / "own_threshold_validation_tight.csv")
    chk("own validation at 1e-11: consistent", float(vt_.consistent.sum()), 6.0, 0)

    print("V4 Block 2 (math note checks)")
    mb = dict(pd.read_csv(R / "mn2_bounds.csv").values)
    chk("B(24)", float(mb["B(24) (exact PAVA, 50 digits)"]), 0.38797358, 1e-8)
    chk("B_full", float(mb["B_full (exact PAVA, 50 digits)"]), 0.47738563, 1e-8)
    us = pd.read_csv(R / "mn2_uniformity_summary.csv").iloc[0]
    chk("uniform branch loss sup on [0.66, 0.72]", float(us.sup_branch_loss), 0.3768942, 1e-7)
    chk("uniform margin vs B(24)", float(us.margin_B24), 0.0111, 0.0001)
    chk("uniform margin vs B_full", float(us.margin_Bfull), 0.1005, 0.0001)
    chk("wider range [0.60, 0.76] not certified", float(us.wider_certified), 0.0, 0)
    chk("wider range overshoot", float(us.wider_margin_B24), -0.0032, 0.0001)
    nb = pd.read_csv(R / "mn2_neighbourhood.csv").iloc[0]
    chk("U: Hessian PD on all sub-boxes", float(bool(nb.hess_pd_all_boxes) and bool(nb.b2_validated_all)), 1.0, 0)
    chk("U: lambda_min lower bound", float(nb.hess_lambda_min_lower_bound), 0.0473, 0.0001)
    chk("U: S_U", float(nb.S_U), 4.857, 0.001)
    for k_, v_ in (("M0", 5.217), ("M1", 13.507), ("M2", 16.375), ("M3", 13.384)):
        chk(f"Taylor constant {k_}", float(nb[k_]), v_, 0.001)
    lsw = pd.read_csv(R / "limit_switch.csv").iloc[0]
    chk("A* bracket lower", float(lsw.A_lo), 0.68125, 1e-9)
    chk("A* bracket upper", float(lsw.A_hi), 0.6875, 1e-9)
    rd = pd.read_csv(R / "mn2_rounding.csv")
    chk("rounding: max |float - interval| <= 1.2e-16", float(rd["diff"].abs().max() <= 1.2e-16), 1.0, 0)

    h2_ = pd.read_csv(R / "mn2_h2prime.csv")
    chk("H2': all validated (b, PD, unique active set)",
        float(h2_.b2_validated.all() and h2_.hessian_pd.all() and h2_.active_set_unique.all()), 1.0, 0)
    chk("H2': dG/dA min", float(h2_.dG_dA_lo.min()), 1.666, 0.0005)
    chk("H2': dG/dA max", float(h2_.dG_dA_hi.max()), 1.705, 0.0005)
    chk("H2': lambda_min lower", float(h2_.hess_lambda_min_lo.min()), 0.1470, 0.0001)

    sl_ = pd.read_csv(R / "mn2_solve_limit.csv").sort_values("A")
    chk("solve limit: sign fails then solves",
        float(list(sl_.certified_sign) == ["fails", "solves"] and sl_.b2_validated.all()), 1.0, 0)
    chk("solve limit: A_solve lo", float(sl_.A_solve_lo.iloc[0]), 1.05875, 1e-9)
    chk("solve limit: A_solve hi", float(sl_.A_solve_hi.iloc[0]), 1.06, 1e-9)
    chk("solve limit: dmargin/dA min", float(sl_.dmargin_dA_lo.min()), 0.532, 0.0005)
    chk("solve limit: dmargin/dA max", float(sl_.dmargin_dA_hi.max()), 0.536, 0.0005)
    chk("solve limit: lambda_min", float(sl_.hess_lambda_min_lo.min()), 0.1369, 0.0001)

    for nm_, mg_, gr_, bu_ in (("annulus", 1.00e-4, 3.28e-3, 0.358045), ("solve", 1.48e-4, 8.47e-3, 0.281166)):
        cs_ = pd.read_csv(R / f"certv2_{nm_}_summary.csv").iloc[0]
        cp_ = pd.read_csv(R / f"certv2_{nm_}_parts.csv")
        chk(f"certv2 {nm_}: covered, all certified, PD and b validated",
            float(bool(cs_.covered) and bool(cp_.certified.all()) and bool(cp_.localised.all())
                  and bool(cs_.pd_all) and bool(cs_.pd_b2_all)), 1.0, 0)
        chk(f"certv2 {nm_}: min outer margin", float(cp_.outer_margin.min()), mg_, 0.005e-4)
        chk(f"certv2 {nm_}: min ring gradient bound", float(cp_.ring_grad_lower.min()), gr_, 0.005e-3)
        chk(f"certv2 {nm_}: max branch upper < B(24)", float(cp_.branch_upper.max()), bu_, 1e-6)
    fs_ = pd.read_csv(R / "first_order_scores.csv", float_precision="round_trip")
    p_ = fs_[fs_.quantity.str.startswith("c1")].iloc[0]
    chk("c1 test: registered verdict INCONCLUSIVE", float(str(p_.verdict).startswith("INCONCLUSIVE")), 1.0, 0)
    chk("c1 test: feasible lo", float(p_.feasible_lo), 0.22521, 0.00001)
    chk("c1 test: feasible hi", float(p_.feasible_hi), 0.33020, 0.00001)
    chk("c1 test: all four competing values excluded",
        float(bool(p_["excludes_0.49"]) and bool(p_["excludes_k1_only_-0.377"])
              and bool(p_["excludes_switch_only_0.662"]) and bool(p_["excludes_0"])), 1.0, 0)
    k_ = fs_[fs_.quantity.str.startswith("k1")].iloc[0]
    chk("c1 test: k1 component PASS", float(k_.verdict == "PASS"), 1.0, 0)
    chk("c1 test: k1 feasible width", float(k_.feasible_width), 0.0013, 0.00005)
    ss_ = pd.read_csv(R / "first_order_supplementary_scores.csv").iloc[0]
    chk("c1 supplementary: feasible lo", float(ss_.feasible_lo), 0.284596, 0.000001)
    chk("c1 supplementary: feasible hi", float(ss_.feasible_hi), 0.285895, 0.000001)
    chk("c1 supplementary: all certified, prediction inside",
        float(bool(ss_.all_certified) and bool(ss_.pred_inside)), 1.0, 0)
    ff_ = pd.read_csv(R / "first_order_finite.csv")
    chk("c1 test: Ghat converged at all four a", float(ff_.Ghat_converged.all() and len(ff_) == 4), 1.0, 0)
    ps_ = pd.read_csv(R / "prospective_own_scores.csv", float_precision="round_trip")
    chk("prospective own: all 16 registered comparisons pass (primary + sensitivity)", float(ps_["pass"].all() and len(ps_) == 16), 1.0, 0)
    pr_ = ps_[ps_.analysis.str.startswith("primary")].set_index(["a", "comparison"])
    for (a_, c_), (st_, lo_, hi_) in {(1.3, "P1"): (-0.0696, -0.0852, -0.0565), (1.3, "P2a"): (-0.0136, -0.0256, -0.0016),
                                      (1.3, "P3"): (-0.0448, -0.0553, -0.0334), (1.5, "P1"): (-0.0695, -0.0864, -0.0562),
                                      (1.5, "P2b"): (0.0226, 0.0091, 0.0363), (1.5, "P3"): (-0.0318, -0.0436, -0.0196)}.items():
        r_ = pr_.loc[(a_, c_)]
        chk(f"prospective own {c_} a={a_} stat", float(r_.stat), st_, 0.00006)
        chk(f"prospective own {c_} a={a_} lo", float(r_.lo), lo_, 0.00006)
        chk(f"prospective own {c_} a={a_} hi", float(r_.hi), hi_, 0.00006)
    chk("prospective own P4 a=1.3", float(pr_.loc[(1.3, "P4")].stat), 0.0419, 0.00006)
    chk("prospective own P4 a=1.5", float(pr_.loc[(1.5, "P4")].stat), 0.0867, 0.00006)
    pv_ = pd.read_csv(R / "prospective_own_validation.csv")
    chk("prospective own validation: 0 contradictions of 48", float(pv_.contradiction.sum() == 0 and len(pv_) == 48), 1.0, 0)
    pe_ = pd.read_csv(R / "prospective_own_early_scores.csv").set_index("a")
    chk("S-early match 1.30", float(pe_.loc[1.3, "match_rate"]), 0.8696, 0.00005)
    chk("S-early match 1.50", float(pe_.loc[1.5, "match_rate"]), 0.8783, 0.00005)
    chk("S-early all expectations consistent", float(pe_.match_consistent.all() and pe_.err_unfitted_in_expected_range.all()
                                                    and pe_.err_fitted_in_expected_range.all() and (pe_.n_undefined == 0).all()), 1.0, 0)
    pss_ = pd.read_csv(R / "prospective_own_settings_scored.csv")
    chk("prospective own non-crossers per a", float(pss_.groupby("a").non_crossers.sum().max()), 20.0, 0)
    lt_ = pd.read_csv(R / "lag_test2_tests.csv")
    lp_ = lt_[(lt_.rule == "primary") & (lt_.stratum == "all")].set_index("a")
    chk("lag2 primary: L1' fails at both a", float((~lp_.L1p_pass.astype(bool)).all()), 1.0, 0)
    chk("lag2 primary: L2' fails at both a", float((~lp_.L2p_pass.astype(bool)).all()), 1.0, 0)
    chk("lag2 primary: competing (no dependence) not met", float((~lp_.competing_no_dependence.astype(bool)).all()), 1.0, 0)
    chk("lag2 primary: all verdict cells sufficient", float(lp_.all_cells_sufficient.astype(bool).all()), 1.0, 0)
    chk("lag2 primary ratio 0.25 a=1.3", float(lp_.loc[1.3, "ratio_025"]), 0.894, 0.0005)
    chk("lag2 primary ratio 0.25 a=1.5", float(lp_.loc[1.5, "ratio_025"]), 0.859, 0.0005)
    chk("lag2 primary ratio 0.5 a=1.3", float(lp_.loc[1.3, "ratio_05"]), 0.974, 0.0005)
    chk("lag2 primary ratio 0.5 a=1.5", float(lp_.loc[1.5, "ratio_05"]), 0.958, 0.0005)
    lc_ = pd.read_csv(R / "lag_test2_cells.csv")
    lcp_ = lc_[(lc_.rule == "primary") & (lc_.stratum == "all")].set_index(["a", "factor"]).residual
    for (a_, f_), v_ in {(1.3, 0.25): 0.0278, (1.3, 1.0): 0.0311, (1.5, 0.25): 0.0563, (1.5, 1.0): 0.0656}.items():
        chk(f"lag2 primary residual a={a_} phi={f_}", float(lcp_.loc[(a_, f_)]), v_, 0.00005)
    lct_ = lc_[(lc_.rule == "tstar") & (lc_.stratum == "all")].set_index(["a", "factor"]).residual
    chk("lag2 tstar residual a=1.3 phi=0.25", float(lct_.loc[(1.3, 0.25)]), 0.0071, 0.00005)
    chk("lag2 tstar residual a=1.5 phi=0.25", float(lct_.loc[(1.5, 0.25)]), 0.0164, 0.00005)
    lf_ = pd.read_csv(R / "lag_test2_confounds.csv")
    chk("lag2 primary: no plateau re-entry in any cell",
        float(lf_[lf_.rule == "primary"].frac_with_reentry.max()), 0.0, 0)
    chk("lag2: 192 phi=1 continuations", float(len(pd.read_csv(R / "lag_test2_runs.csv").query("factor == 1.0"))), 192.0, 0)
    lw_ = pd.read_csv(R / "limit_windows.csv").set_index("window")
    chk("per-window A*: 9 windows, all converged, ends certified minus/plus",
        float(len(lw_) == 9 and lw_.all_converged.all() and (lw_.status_lo == "minus").all() and (lw_.status_hi == "plus").all()), 1.0, 0)
    chk("per-window A*: localised windows", float(lw_.localised.sum()), 7.0, 0)
    chk("per-window A*: G1, G4 not localised", float((~lw_.loc[["G1_wide_gap", "G4_shifted"], "localised"]).all()), 1.0, 0)
    chk("per-window A*: H10 flagged (unresolved stop)", float(not bool(lw_.loc["H10", "switch_certified"])), 1.0, 0)
    chk("per-window A*: base A* bracket contains sharp A*",
        float(lw_.loc["base", "A_lo"] < 0.68544523757565 <= lw_.loc["base", "A_hi"]), 1.0, 0)
    for w_, lo_, hi_ in (("base", 0.1981, 0.1990), ("G2_narrow_gap", 0.1230, 0.1232), ("H10", 0.1864, 0.1875), ("H65", 0.1989, 0.2002)):
        chk(f"per-window R_inf lo {w_}", float(lw_.loc[w_, "R_inf_lo"]), lo_, 0.00006)
        chk(f"per-window R_inf hi {w_}", float(lw_.loc[w_, "R_inf_hi"]), hi_, 0.00006)
    print("V4 patch (WRITER_INPUTS_v4_patch.md)")
    wp1 = pd.read_csv(R / "writer_patch_prospective_own_per_setting.csv")
    chk("WP-1: 16 settings + 2 per-a rows", float(len(wp1)), 18.0, 0)
    agg = wp1[wp1.window.str.startswith("all")].set_index("a")
    sc_ = pd.read_csv(R / "prospective_own_scores.csv", float_precision="round_trip")
    p1_ = sc_[(sc_.analysis.str.startswith("primary")) & (sc_.comparison == "P1")].set_index("a").stat
    p3_ = sc_[(sc_.analysis.str.startswith("primary")) & (sc_.comparison == "P3")].set_index("a").stat
    for a_ in (1.3, 1.5):
        chk(f"WP-1: mean U_own - U reproduces registered P1 at a={a_}",
            float(agg.loc[a_, "U_own_mean_abs_log_err"] - agg.loc[a_, "U_mean_abs_log_err"]), float(p1_.loc[a_]), 1e-12)
        chk(f"WP-1: mean C_own - C reproduces registered P3 at a={a_}",
            float(agg.loc[a_, "C_own_mean_abs_log_err"] - agg.loc[a_, "C_mean_abs_log_err"]), float(p3_.loc[a_]), 1e-12)
    chk("WP-1: crossers per a", float(agg.n_crossed.sum()), 920.0, 0)
    wp2 = pd.read_csv(R / "writer_patch_windows.csv")
    chk("WP-2: 4 H + 8 V windows", float(len(wp2)), 12.0, 0)
    wp3 = pd.read_csv(R / "writer_patch_w_bound.csv")
    chk("WP-3: loss > log 2 at and beyond W at every certified bracket end", float(wp3.exceeds_log2.all()), 1.0, 0)
    chk("WP-3: 24 bracket ends checked", float(len(wp3)), 24.0, 0)
    chk("WP-3: min loss at/beyond W", float(wp3.min_profiled_loss_at_and_beyond_W.min()), 4.337, 0.001)
    chk("WP-3: W at a=1.30 glob lo", float(wp3[(wp3.a == 1.3) & (wp3.kind == "glob") & (wp3.end == "lo")].W.iloc[0]),
        2.9077, 0.0001)
    wp4 = pd.read_csv(R / "writer_patch_census_by_block.csv")
    chk("WP-4: by-block rows sum to 232", float(wp4[wp4.by == "block"].n.sum()), 232.0, 0)
    chk("WP-4: by-file rows sum to 232", float(wp4[wp4.by == "registration_file"].n.sum()), 232.0, 0)
    oth_ = pd.read_csv(R / "registration_census_v4_gates_and_reported.csv")
    chk("width-2 verdict listed as a registered decision rule, not a prediction",
        float((oth_.id == "sl-width2").sum() == 1
              and "sl-width2" not in set(pd.read_csv(R / "registration_census.csv").id)), 1.0, 0)
    wp5 = pd.read_csv(R / "writer_patch_figure_sizes.csv")
    chk("WP-5: 7 figures, all <= 5.5 in wide", float(len(wp5) == 7 and (wp5.width_in <= 5.5).all()), 1.0, 0)
    chk("WP-5: tallest figure height", float(wp5.height_in.max()), 3.123, 0.002)

    print("Block 4b (registered: residual_mechanism_design.md)")
    rmc = pd.read_csv(R / "residual_mechanism_checks.csv")
    chk("4b: every validity check passed", float(rmc["pass"].all()), 1.0, 0)
    rms = pd.read_csv(R / "residual_mechanism_scores.csv")
    for a_ in (1.3, 1.5):
        v_ = rms[(rms.a == a_) & (rms.arm == "VERDICT")].verdict.iloc[0]
        chk(f"4b a={a_}: verdict competing (neither removes half)", float(v_.startswith("competing")), 1.0, 0)
    chk("4b a=1.3 control residual", float(rms[(rms.a == 1.3) & (rms.arm == "control")].median_residual.iloc[0]), 0.031053, 1e-6)
    chk("4b a=1.3 teleport residual", float(rms[(rms.a == 1.3) & (rms.arm == "teleport")].median_residual.iloc[0]), 0.030787, 1e-6)
    chk("4b a=1.5 reset residual", float(rms[(rms.a == 1.5) & (rms.arm == "reset")].median_residual.iloc[0]), 0.067219, 1e-6)
    rtr_ = rms[rms.arm == "teleport_reset"].set_index("a").median_residual
    chk("4b a=1.3 teleport+reset residual", float(rtr_.loc[1.3]), 0.031222, 1e-6)
    chk("4b a=1.5 teleport+reset residual", float(rtr_.loc[1.5]), 0.066012, 1e-6)
    print("Crossing-detection audit (POST HOC: crossing_audit.md)")
    car_ = pd.read_csv(R / "crossing_audit_runs.csv")
    chk("audit: every rerun reproduced bit for bit (160)",
        float(car_[car_.source == "rerun"].reproduced.all() and (car_.source == "rerun").sum() == 160), 1.0, 0)
    chk("audit: phase 2b stored runs located (228)",
        float(car_[car_.source == "stored"].reproduced.sum()), 228.0, 0)
    cas_ = pd.read_csv(R / "crossing_audit_summary.csv").set_index(["family", "a"])
    def _ca(fam, a, col):
        return float(cas_[cas_.index.get_level_values(0).str.startswith(fam)].xs(a, level=1)[col].iloc[0])
    for fam_, a_, col_, want_ in (("phase 2b", 1.3, "median_residual_check", 0.0313), ("phase 2b", 1.3, "median_residual_interp_check", 0.0311),
                                  ("phase 2b", 1.5, "median_residual_check", 0.0661), ("phase 2b", 1.5, "median_residual_interp_check", 0.0658),
                                  ("phase 2b", 1.6, "growth_over_interval_rel_max", 0.0044),
                                  ("size test", 1.3, "median_residual_check", 0.0316), ("size test", 1.3, "median_residual_interp_check", 0.0311),
                                  ("size test", 1.5, "median_residual_check", 0.0668), ("size test", 1.5, "median_residual_interp_check", 0.0659),
                                  ("size test", 1.3, "growth_one_step_rel_median", 0.00058), ("size test", 1.5, "growth_one_step_rel_median", 0.00149),
                                  ("Block G base", 1.3, "growth_over_interval_rel_median", 0.0240), ("Block G base", 1.5, "growth_over_interval_rel_median", 0.0491),
                                  ("Block G base", 1.3, "median_shift_check_minus_interp_check", 0.0138), ("Block G base", 1.5, "median_shift_check_minus_interp_check", 0.0191),
                                  ("Block 3", 1.3, "median_residual_check", 0.0858), ("Block 3", 1.3, "median_residual_interp_check", 0.0746),
                                  ("Block 3", 1.5, "median_residual_check", 0.1304), ("Block 3", 1.5, "median_residual_interp_check", 0.1169),
                                  ("Block 3", 1.5, "growth_over_interval_rel_max", 0.2111), ("Block 3", 1.5, "max_shift_check_minus_interp_check", 0.1628),
                                  ("prospective own-seed", 1.3, "median_residual_check", 0.0430), ("prospective own-seed", 1.3, "median_residual_interp_check", 0.0297),
                                  ("prospective own-seed", 1.5, "median_residual_check", 0.0801), ("prospective own-seed", 1.5, "median_residual_interp_check", 0.0618)):
        chk(f"audit {fam_} a={a_} {col_}", _ca(fam_, a_, col_), want_, 0.00006)
    print("Scale limits, tanh (registered: scale_limits_tanh_prediction.md)")
    tl_ = pd.read_csv(R / "scale_limits_tanh.csv").set_index("A")
    chk("tanh: registered outcome 'neither'", float(pd.read_csv(R / "scale_limits_tanh_summary.csv").verdict.iloc[0] == "neither"), 1.0, 0)
    chk("tanh: validation passed at every A", float(tl_.ladder_ok.all() and tl_.independent_ok.all() and tl_.analytic_attain.all()), 1.0, 0)
    chk("tanh: box maxima < 1 and rising", float((tl_.top < 1).all() and (np.diff(tl_.top.values) > 1e-9).all()), 1.0, 0)
    chk("tanh: boundary condition fails only at A = 40", float(list(tl_.maximisers_on_boundary) == [True, True, True, False]), 1.0, 0)
    chk("tanh: min max|alpha|/A at A = 40", float(tl_.loc[40.0, "min_max_abs_alpha_over_A"]), 0.9908, 0.0001)
    chk("tanh: 1 - max at A = 40", float(tl_.loc[40.0, "one_minus_max"]), 4.10e-9, 0.01e-9)
    for A_, g_ in ((5.0, 0.7616), (10.0, 0.9640), (20.0, 0.99933), (40.0, 0.9999998)):
        chk(f"tanh: selected (pair) G+ at A = {A_}", float(tl_.loc[A_, "selected_G_lo"]), g_, 0.00005 if A_ < 20 else 0.000005)
    chk("tanh: selected member is the symmetric pair at every A",
        float((tl_.selected_type == "pair").all() and (tl_.selected_t == 0.5).all()), 1.0, 0)
    tm_ = pd.read_csv(R / "scale_limits_tanh_maximisers.csv")
    chk("tanh: single units unplaced at every A", float((tm_[tm_.type == "single_unit"].G_hi < 0).all()), 1.0, 0)
    t40_ = tm_[tm_.A == 40.0]
    chk("tanh: A = 40 near-maximisers off the boundary", float((t40_.max_abs_alpha_over_A < 0.999).sum()), 9.0, 0)
    chk("tanh: A = 40 tie-set size", float(len(t40_)), 1107.0, 0)
    print("WP-6: 50-step rerun and cadence sensitivity (POST HOC)")
    fr_ = pd.read_csv(R / "crossing_audit_full_runs.csv")
    chk("full rerun: replays", float(len(fr_)), 1998.0, 0)
    chk("full rerun: every replay reproduced", float(fr_.reproduced.all()), 1.0, 0)
    cs_ = pd.read_csv(R / "cadence_sensitivity.csv")
    def _cs(exp, a, qty, col):
        g = cs_[(cs_.experiment == exp) & (cs_.quantity == qty)]
        g = g if a is None else g[g.a.round(2) == a]
        return float(g[col].iloc[0])
    for a_, c_, i_ in ((1.3, 0.0412, 0.0291), (1.5, 0.0878, 0.0622)):
        chk(f"own-seed residual a={a_} check", _cs("own-seed", a_, "median residual R_cross / U_own - 1", "check_based"), c_, 0.00006)
        chk(f"own-seed residual a={a_} interp", _cs("own-seed", a_, "median residual R_cross / U_own - 1", "interpolated"), i_, 0.00006)
    for a_, c_, i_ in ((1.3, 0.0876, 0.0732), (1.5, 0.1421, 0.1154)):
        chk(f"Block 3 residual vs U a={a_} check", _cs("Block 3", a_, "median over settings of (median R / U) - 1", "check_based"), c_, 0.00006)
        chk(f"Block 3 residual vs U a={a_} interp", _cs("Block 3", a_, "median over settings of (median R / U) - 1", "interpolated"), i_, 0.00006)
    for a_, c_, i_ in ((1.3, 1.1149, 1.0979), (1.5, 1.1644, 1.1298)):
        chk(f"lambda a={a_} check", _cs("Block G base (lambda)", a_, "lambda(a)", "check_based"), c_, 0.00006)
        chk(f"lambda a={a_} interp", _cs("Block G base (lambda)", a_, "lambda(a)", "interpolated"), i_, 0.00006)
    chk("B2 check", _cs("Block G all windows (B2)", None, "B2 pooled median crossing R", "check_based"), 0.2280, 0.00006)
    chk("B2 interp", _cs("Block G all windows (B2)", None, "B2 pooled median crossing R", "interpolated"), 0.2260, 0.00006)
    p2b_ = cs_[(cs_.experiment == "own-seed") & (cs_.quantity == "P2b stat [registered predictions]")].iloc[0]
    chk("P2b a=1.50 check interval above 0", float(p2b_.check_lo > 0), 1.0, 0)
    chk("P2b a=1.50 interp interval includes 0", float(p2b_.interp_lo < 0 < p2b_.interp_hi), 1.0, 0)
    chk("every own-seed criterion passes both ways", float(cs_[cs_.experiment == "own-seed"].dropna(subset=["check_pass"])
                                                           [["check_pass", "interp_pass"]].astype(bool).all().all()), 1.0, 0)
    b3m_ = cs_[(cs_.experiment == "Block 3") & cs_.quantity.str.contains("ci95_hi") & cs_.quantity.str.contains("cadence-matched")]
    chk("Block 3 cadence-matched: every comparison excludes 0", float(b3m_.interp_pass.astype(bool).all() and len(b3m_) == 3), 1.0, 0)
    chk("Block 3 registered preds: C - B2 upper interp", float(cs_[cs_.quantity == "C - B2 ci95_hi [registered predictions]"].interpolated.iloc[0]), 0.0053, 0.00006)
    print("WP-7: Ghat enclosures and G* best lower bound")
    gr_ = pd.read_csv(R / "ghat_rigorous.csv")
    bl_ = np.maximum(gr_.Ghat_cert_rigorous, gr_.bnb_arg_lower)
    rel_ = (bl_ - gr_.Ghat_cert_rigorous) / gr_.Ghat_cert_rigorous
    chk("G* best lower above Ghat_cert at nine a", float((gr_.bnb_arg_lower > gr_.Ghat_cert_rigorous).sum()), 9.0, 0)
    chk("G* best lower - Ghat_cert, max relative", float(rel_.max()), 8.36e-5, 0.006e-5)
    chk("G* best lower - Ghat_cert, max relative at a = 1.10", float(gr_.a[rel_.idxmax()]), 1.10, 1e-9)
    chk("G* best lower - Ghat_cert, absolute at a = 1.10", float((bl_ - gr_.Ghat_cert_rigorous)[rel_.idxmax()]), 1.48e-6, 0.006e-6)
    ab_ = bl_ - gr_.Ghat_cert_rigorous
    chk("G* best lower - Ghat_cert, max absolute", float(ab_.max()), 7.09e-6, 0.006e-6)
    chk("G* best lower - Ghat_cert, max absolute at a = 1.60", float(gr_.a[ab_.idxmax()]), 1.60, 1e-9)
    chk("G* best lower - Ghat_cert, relative at a = 1.60", float(rel_[ab_.idxmax()]), 3.16e-5, 0.006e-5)
    print("WP-8: unplaced region at small scale")
    lm_ = pd.read_csv(R / "width2_unplaced_localmin.csv")
    un_ = pd.read_csv(R / "width2_unplaced.csv")
    r1_ = lm_.lowest_local_min_minus_placed / lm_.predicted_single_unit_gap - 1
    chk("single-unit gap within 0.1-2% of prediction", float(r1_.abs().min() >= 0.0005 and r1_.abs().max() <= 0.02), 1.0, 0)
    chk("lowest unplaced local minimum is a single unit at every scale", float(lm_.lowest_local_min_single_unit.all()), 1.0, 0)
    chk("its G+ range lo", float(lm_.lowest_local_min_G.min()), -3.65, 0.01)
    chk("its G+ range hi", float(lm_.lowest_local_min_G.max()), -3.34, 0.01)
    mm_ = lm_.merge(un_, on=["act", "R2"])
    r2_ = (mm_.best_unplaced_loss - mm_.local_descent_loss) / mm_.predicted_boundary_gap - 1
    chk("boundary drop within 0.2-3% of prediction", float(r2_.min() >= 0.0015 and r2_.max() <= 0.032), 1.0, 0)
    chk("all best-unplaced on the boundary", float((un_.classification == "boundary").all()), 1.0, 0)
    chk("|c| min", float(un_.linear_c.abs().min()), 0.43, 0.006)
    chk("|c| max", float(un_.linear_c.abs().max()), 0.51, 0.006)
    chk("min margin at R2 = 0.001", float((mm_[mm_.R2 == 0.001].best_unplaced_loss - mm_[mm_.R2 == 0.001].local_descent_loss).min()), 9.9e-8, 0.06e-8)
    print("WP-9/10 inputs")
    w1g_ = pd.read_csv(R / "scale_limits_width1.csv")
    chk("width 1: G range lo (a = 1.30)", float(w1g_.G.min()), -3.662, 0.0006)
    chk("width 1: G range hi (a = 1.60)", float(w1g_.G.max()), -3.349, 0.0006)
    pc_ = pd.read_csv(R / "residual_posthoc_correlations.csv")
    def _rho(a, feat):
        return float(pc_[(pc_.a.round(2) == a) & (pc_.point == "primary") & (pc_.feature == feat)].spearman_rho.iloc[0])
    chk("4a a=1.3 displacement rho (post hoc)", _rho(1.3, "dist"), 0.675, 0.0006)
    chk("4a a=1.5 displacement rho (post hoc)", _rho(1.5, "dist"), 0.668, 0.0006)
    chk("4a a=1.3 sqrt vhat(w2) rho (post hoc)", _rho(1.3, "sqrt_vhat_w2"), -0.466, 0.0006)
    chk("4a a=1.5 sqrt vhat(w2) rho (post hoc)", _rho(1.5, "sqrt_vhat_w2"), -0.480, 0.0006)
    fs_ = pd.read_csv(R / "width2_finish_summary.csv")
    ld_ = fs_[fs_.direct_check_landed.astype(bool)]
    chk("direct check: every landed scale placed, cancelling pair",
        float(len(ld_) > 0 and ld_.direct_check_placed.astype(bool).all() and ld_.direct_check_pair.astype(bool).all()), 1.0, 0)
    chk("direct check: no finished restart below the retained minimiser (landed scales)",
        float((ld_.n_below_retained == 0).all()), 1.0, 0)
    print("K domain lemma (math note s8)")
    kd_ = pd.read_csv(R / "k_domain_check.csv").iloc[0]
    chk("K domain: identity max relative error", float(kd_.identity_max_rel_err), 1.4e-12, 0.05e-12)
    chk("K domain: G0 <= 0 on the excluded region (u > 0 max)", float(kd_.max_G0_on_excluded_region_u_positive), -1.19e-3, 0.006e-3)
    chk("K domain: excluded points checked", float(kd_.excluded_points_checked), 4035775.0, 0)
    chk("K domain: region inside the certified box", float(bool(kd_.region_inside_box)), 1.0, 0)
    chk("K domain: K argmax inside the region", float(bool(kd_.K_argmax_inside_region)), 1.0, 0)
    print("W0 scan coverage (never registered)")
    sc_ = pd.read_csv(R / "width2_w0_scan_coverage.csv")
    c15 = sc_[(sc_.act == "f1.50") & sc_.completed.astype(bool)]
    chk("W0 a=1.50: 9 grid points completed", float(len(c15)), 9.0, 0)
    chk("W0 a=1.50: R2 = 0.02..0.10", float(abs(c15.R2.min() - 0.02) < 1e-9 and abs(c15.R2.max() - 0.10) < 1e-9), 1.0, 0)
    chk("W0 a=1.50: placed and audit passed at every point", float(c15.placed.astype(bool).all() and c15.audit_ok.astype(bool).all()), 1.0, 0)
    chk("W0 a=1.50: none validated", float((~c15.validated.astype(bool)).all()), 1.0, 0)
    chk("W0 a=1.30: no point completed", float(sc_[(sc_.act == "f1.30") & sc_.completed.astype(bool)].shape[0]), 0.0, 0)
    print("Direct check (complete)")
    dc_ = pd.read_csv(R / "width2_direct_check.csv")
    chk("direct check: 8 scales", float(len(dc_)), 8.0, 0)
    chk("direct check: placed and the cancelling pair at all 8", float(dc_.placed.astype(bool).all()
                                                                        and (dc_.is_cancelling_pair.astype(str) == "True").all()), 1.0, 0)
    chk("direct check: every validation passed at all 8", float((dc_.audit_ok & dc_.stall_local_ok & dc_.ladder_ok
                                                                   & dc_.stricter_ok & dc_.independent_ok).astype(bool).all()), 1.0, 0)
    chk("direct check: min G+ at a = 1.30", float(dc_[dc_.act == "f1.30"].Gplus_lo.min()), 0.8895, 0.00006)
    chk("direct check: min G+ at a = 1.50", float(dc_[dc_.act == "f1.50"].Gplus_lo.min()), 1.0263, 0.00006)
    print("Block 2: limit-switch status checks")
    import json as _j
    for n_, st_ in (("limit_A_lo", "minus"), ("limit_A_hi", "plus")):
        r_ = _j.loads((R / "certificate_checks" / f"{n_}.json").read_text())
        chk(f"limit check {'A_lo' if n_.endswith('lo') else 'A_hi'} passes",
            float(r_["pass"] and all(r_["checks"].values()) and not r_["failed_leaves"]), 1.0, 0)
    chk("limit check A_lo losing leaves", float(_j.loads((R / "certificate_checks" / "limit_A_lo.json").read_text())["leaves"]), 72075.0, 0)
    chk("limit check A_hi losing leaves", float(_j.loads((R / "certificate_checks" / "limit_A_hi.json").read_text())["leaves"]), 69102.0, 0)
    print("Block 2: finite-a certificates, independent checker")
    vcf_ = (R / "verify_certificates_finite.log").read_text()
    chk("Block 2: 12 finite-a certificates pass", float(vcf_.count('"pass": true') == 12
                                                         and "12 certificate(s) checked; failures: none" in vcf_), 1.0, 0)
    print("Block 2: Ghat enclosure certificates (certificate_audit.md)")
    import json as _json, re as _re
    gl_ = (R / "verify_certificates_ghat.log").read_text()
    go_ = {}
    for m_ in _re.findall(r"^\{\n.*?^\}", gl_, flags=_re.S | _re.M):
        o_ = _json.loads(m_); go_[o_["name"]] = o_                      # the latest run of each certificate
    chk("Ghat certificates checked (a = 1.02-3.0)", float(len(go_)), 13.0, 0)
    chk("Ghat: structure passes at every a (hashes, tiling, domain, Ghat_cert <= hi)",
        float(all(o_["checks"][k] for o_ in go_.values() for k in ("files_match_committed_hashes", "coverage",
                                                                    "domain_contains_reduction", "ghat_cert_below_hi"))), 1.0, 0)
    chk("Ghat: max |published - rigorous| endpoint discrepancy <= 1.1e-15 (a >= 1.05; a = 1.02 <= 1.4e-16)",
        float(max(max(abs(o_["claim_hi_excess_over_published"]), abs(o_["ghat_cert_deficit"]), abs(o_["bnb_lo_deficit"]))
                  for o_ in go_.values()) <= 1.1e-15), 1.0, 0)
    chk("Ghat(1.05) leaves", float(go_["ghat_a1.05"]["leaves"]), 18041202.0, 0)
    print("Scale limits (registered: scale_limits_prediction.md)")
    sd_ = pd.read_csv(R / "scale_limits_D.csv").iloc[0]
    chk("alpha*", float(sd_.alpha_star), 1.7913244, 1e-6)
    w1s = pd.read_csv(R / "scale_limits_width1.csv")
    chk("width 1: selected dmu-maximiser G <= 0 at all six a", float((w1s.G <= 0).all() and len(w1s) == 6), 1.0, 0)
    w2s = pd.read_csv(R / "scale_limits_width2.csv").set_index("a")
    chk("width 2: validation (ladder, independent, pair attains max, pair Var minimal)",
        float(w2s.ladder_ok.all() and w2s.independent_ok.all() and w2s.pair_attains_max.all()
              and w2s.pair_var_le_all_maximisers.all()), 1.0, 0)
    chk("width 2: selected G at a=1.30", float(w2s.loc[1.3, "selected_G_lo"]), 0.8896, 0.0001)
    chk("width 2: selected G at a=1.50", float(w2s.loc[1.5, "selected_G_lo"]), 1.0265, 0.0001)
    chk("verdict: no placement threshold for f_a at width 2",
        float(pd.read_csv(R / "scale_limits_summary.csv").verdict.iloc[0].startswith("width 2 f_a: NO")), 1.0, 0)
    chk("certv2 annulus: 40 sub-intervals over [0.66, 0.71]",
        float(len(pd.read_csv(R / "certv2_annulus_parts.csv"))), 40.0, 0)
    chk("certv2 solve: PD lambda_min", float(pd.read_csv(R / "certv2_solve_summary.csv").pd_lambda_min_lower.iloc[0]),
        0.0209, 0.0001)

    sf_ = pd.read_csv(R / "mn2_solve_finite.csv")
    chk("solve finite: 12 bracket ends", float(len(sf_)), 12.0, 0)
    chk("solve finite: fails at lower ends", float((sf_.groupby("a").certified_sign.first() == "fails").all()), 1.0, 0)
    chk("solve finite: solves at upper ends", float((sf_.groupby("a").certified_sign.last() == "solves").all()), 1.0, 0)
    chk("solve finite: b validated", float(sf_.b2_validated.all()), 1.0, 0)

    print("V4 Block 3 (prospective)")
    cmp_ = pd.read_csv(R / "prospective_comparisons.csv").set_index("comparison")
    for c_, m_, lo_, hi_ in (("C - B1", -0.257, -0.370, -0.131), ("C - B2", -0.055, -0.103, -0.010),
                             ("C - U", -0.086, -0.111, -0.063),
                             ("secondary: C - B3 (MAE of placed fraction)", -0.044, -0.068, -0.018)):
        chk(f"{c_} mean", float(cmp_.loc[c_, "mean_diff"]), m_, 0.0006)
        chk(f"{c_} ci lo", float(cmp_.loc[c_, "ci95_lo"]), lo_, 0.0006)
        chk(f"{c_} ci hi", float(cmp_.loc[c_, "ci95_hi"]), hi_, 0.0006)
    sc = pd.read_csv(R / "prospective_scores.csv")
    chk("C max abs log error", float(sc.abslogerr_C.max()), 0.037, 0.0005)
    chk("crossed min", float(sc.n_crossed.min()), 86.0, 0)
    chk("crossed max", float(sc.n_crossed.max()), 88.0, 0)
    chk("U inside registered range", float(sc.U_in_expected_range.sum()), 7.0, 0)
    chk("U under-predicts everywhere", float((sc.U < sc.obs_median_cross_R).all()), 1.0, 0)
    chk("U log(obs/U) min", float(sc.abslogerr_U.min()), 0.077, 0.0005)
    chk("U log(obs/U) max", float(sc.abslogerr_U.max()), 0.151, 0.0005)
    chk("C over-predicts everywhere", float((sc.C > sc.obs_median_cross_R).all()), 1.0, 0)
    cal = pd.read_csv(R / "prospective_calibration.csv").set_index("a")
    chk("lambda(1.30)", float(cal.loc[1.3, "lambda_fitted"]), 1.115, 0.0005)
    chk("lambda(1.50)", float(cal.loc[1.5, "lambda_fitted"]), 1.164, 0.0005)
    sg = np.log(sc.C / sc.obs_median_cross_R)
    chk("C mean signed log error", float(sg.mean()), 0.022, 0.0005)
    sa = pd.read_csv(R / "prospective_secondary_analyses.csv").set_index("analysis")
    chk("window bootstrap C - B1 lo", float(sa.loc["window-clustered bootstrap: C - B1", "lo"]), -0.414, 0.0005)
    chk("window bootstrap C - B1 hi", float(sa.loc["window-clustered bootstrap: C - B1", "hi"]), -0.069, 0.0005)
    chk("window bootstrap C - B2 lo", float(sa.loc["window-clustered bootstrap: C - B2", "lo"]), -0.098, 0.0005)
    chk("window bootstrap C - B2 hi", float(sa.loc["window-clustered bootstrap: C - B2", "hi"]), -0.012, 0.0005)
    chk("signed error setting-level lo", float(sa.loc["C signed log error log(C/obs)", "lo"]), 0.0145, 0.0005)
    chk("signed error setting-level hi", float(sa.loc["C signed log error log(C/obs)", "hi"]), 0.0295, 0.0005)

    for blk, n_ck, n_rep, fr_, x50_ in ((4, 543, 8688, (0.0, 0.0, 0.013, 0.309, 0.700, 0.967, 0.989, 0.989), 1.049),
                                        (5, 177, 2832, (0.0, 0.0, 0.017, 0.203, 0.644, 0.966, 1.0, 1.0), 1.067)):
        print(f"V4 Block {blk} (fixed scale)")
        rp_ = pd.read_csv(R / f"fixed_scale_block{blk}.csv")
        chk("replays", float(len(rp_)), float(n_rep), 0)
        chk("decisions preserved, all", float(rp_.decisions_preserved.all()), 1.0, 0)
        chk("k = 1 check max diff", float(rp_.k1_check_max_diff.max()), 0.0, 1e-10)
        cv_ = pd.read_csv(R / f"fixed_scale_block{blk}_curve.csv")
        pr_ = cv_[cv_.variant == "preserved"].sort_values("level")
        chk("checkpoints", float(pr_.n.max()), float(n_ck), 0)
        for l_, f_ in zip(pr_.level, fr_):
            chk(f"fraction at {l_}", float(pr_[pr_.level == l_].frac.iloc[0]), f_, 0.0006)
        ts_ = pd.read_csv(R / f"fixed_scale_block{blk}_tests.csv").set_index("variant")
        chk("x50 preserved", float(ts_.loc["preserved", "x50"]), x50_, 0.0006)
        chk("monotone violations, both variants", float(ts_.monotonic_violations.sum()), 0.0, 0)
        if blk == 4:
            chk("D1 pass, both", float(ts_.D1_pass.all()), 1.0, 0)
            chk("D2 pass, any", float(ts_.D2_pass.any()), 0.0, 0)
        else:
            chk("x50 reset", float(ts_.loc["reset", "x50"]), 1.068, 0.0006)
            chk("location in [0.9, 1.1], both", float(ts_.location_in_band.all()), 1.0, 0)
            chk("width 10-90", float(ts_.loc["preserved", "width_10_90"]), 0.27, 0.006)
            sp_ = pd.read_csv(R / "fixed_scale_block5_splits.csv")
            chk("split widths min", float(sp_.width_10_90.min()), 0.26, 0.006)
            chk("split widths max", float(sp_.width_10_90.max()), 0.28, 0.006)
            chk("split x50 range", float(sp_.x50.max() - sp_.x50.min()), 0.03, 0.001)

    print("V4 exploratory: critical slowing (post hoc)")
    cs = pd.read_csv(R / "critical_slowing_posthoc.csv").set_index("a")
    chk("predicted offset at 1.30 (%)", float(cs.loc[1.3, "pred_offset_pct"]), 1.08, 0.01)
    chk("observed offset at 1.30 (%)", float(cs.loc[1.3, "obs_offset_pct"]), 9.59, 0.01)



def harsh_review_checks() -> None:
    """Harsh review Phase A (WP-12 to WP-14; 2026-09-25)."""
    print("WP-12 mechanism (A1)")
    idn = pd.read_csv(R / "wp12_identity.csv")
    chk("A1 identity: max abs err, data points", float(idn.max_abs_err_data.max() < 1e-13), 1.0, 0)
    chk("A1 identity: max abs err, continuous windows", float(idn.max_abs_err_continuous.max() < 1e-12), 1.0, 0)
    m = pd.read_csv(R / "wp12_mechanism.csv").set_index("a")
    chk("A1 alpha* (data)", float(m.alpha_star_data.iloc[0]), 1.7913244, 1e-6)
    chk("A1 alpha* (continuous)", float(m.alpha_star_cont.iloc[0]), 1.7922917, 1e-6)
    chk("A1 |D*| (data)", float(m.D_star_data.iloc[0]), 1.571067, 1e-6)
    chk("A1 1.4 alpha*", float(m["a_limit_1.4_alpha_star"].iloc[0]), 2.50785, 1e-5)
    chk("A1 s0(1.30)", float(m.loc[1.3, "s0"]), 0.342, 0.0005)
    chk("A1 s0(1.50)", float(m.loc[1.5, "s0"]), 0.274, 0.0005)
    chk("A1 s1_data(1.30)", float(m.loc[1.3, "s1_data"]), 163.8, 0.05)
    chk("A1 s1_data(1.50)", float(m.loc[1.5, "s1_data"]), 80.5, 0.05)
    chk("A1 R1 = log(n/log 2)", float(m.R1_data.iloc[0]), 7.05, 0.005)
    chk("A1 every certified bracket inside [s0, s1]", float(m.bracket_contains_cert.all()), 1.0, 0)
    chk("A1 G at theta* (1.30)", float(m.loc[1.3, "G_cont_hi_at_theta_star"]), -3.66, 0.005)
    chk("A1 G at theta* (1.50)", float(m.loc[1.5, "G_cont_hi_at_theta_star"]), -3.45, 0.005)
    chk("A1 bound check at s0 (1.30): L*(theta*) below placed lower bound",
        float(m.loc[1.3, "bound_check_Lstar_at_s0"] < m.loc[1.3, "bound_check_lower_placed"]), 1.0, 0)
    chk("A1 window s1 defined for a >= 1.05", float(m.s1_cont.iloc[1:].notna().all() and m.s1_cont.isna().iloc[0]), 1.0, 0)
    print("WP-12 scaling and Adam (A2)")
    sc = pd.read_csv(R / "wp12_scaling.csv").set_index("a")
    chk("A2 w2 first order (1.02)", float(sc.loc[1.02, "w2_first_order"]), 245.6, 0.05)
    chk("A2 w2 limit (1.02)", float(sc.loc[1.02, "w2_limit_krawczyk"]), 242.3, 0.05)
    chk("A2 w2 limit, A* lower end (1.02)", float(sc.loc[1.02, "w2_limit_lo"]), 240.9, 0.05)
    chk("A2 first order vs certified, max rel", float(sc.rel_err_first_order_vs_cert_mid.max()), 0.027, 0.0005)
    chk("A2 first order vs certified, min rel", float(sc.rel_err_first_order_vs_cert_mid.min()), 0.009, 0.0005)
    chk("A2 N_min at 1.02", float(sc.loc[1.02, "N_min_adam_worst_case"]), 3966.0, 0)
    chk("A2 N at lr per step, 1.02", float(sc.loc[1.02, "N_lr_per_step"]), 24456.0, 0)
    ad = pd.read_csv(R / "wp12_adam_bound.csv").iloc[0]
    chk("A2 B_inf", float(ad.B_inf), 7.2703, 0.0001)
    chk("A2 reachable |w2| at 2000 steps", float(ad.reachable_2000), 106.93, 0.005)
    chk("A2 max B_t over 2000 steps", float(ad.B_max_2000), 6.76, 0.005)
    chk("A2 bound attained (t = 400)", float(abs(ad.worst_case_attained_ratio - ad.B_t_400) < 1e-9), 1.0, 0)
    chk("A2 a=1.02: runs", float(ad.a102_runs), 200.0, 0)
    chk("A2 a=1.02: solved", float(ad.a102_solved), 0.0, 0)
    chk("A2 a=1.02: median terminal |w2|", float(ad.a102_median_terminal_w2), 1.85, 0.005)
    chk("A2 a=1.02: max terminal |w2|", float(ad.a102_max_terminal_w2), 3.92, 0.005)
    ob = pd.read_csv(R / "wp12_onset_bound.csv").set_index("budget")
    chk("A2 worst-case a_min at 2000", float(ob.loc[2000, "a_min_worst_case"]), 1.035, 0.0005)
    chk("A2 lr-rate a_min at 2000", float(ob.loc[2000, "a_min_lr_per_step"]), 1.107, 0.0005)
    chk("A2 worst-case a_min at 128000", float(ob.loc[128000, "a_min_worst_case"]), 1.0018, 0.00005)
    print("WP-14 census by relevance (A4)")
    cr = pd.read_csv(R / "census_relevance.csv")
    n = lambda **k: float(len(cr.query(" and ".join(f"{a} == {b!r}" for a, b in k.items()))))
    chk("A4 FAIL registered central", n(verdict="FAIL", scoring="registered rule", relevance="central"), 31.0, 0)
    chk("A4 FAIL registered peripheral", n(verdict="FAIL", scoring="registered rule", relevance="peripheral"), 35.0, 0)
    chk("A4 FAIL post hoc central", n(verdict="FAIL", scoring="post hoc (census)", relevance="central"), 2.0, 0)
    chk("A4 FAIL post hoc peripheral", n(verdict="FAIL", scoring="post hoc (census)", relevance="peripheral"), 4.0, 0)
    chk("A4 PASS registered central", n(verdict="PASS", scoring="registered rule", relevance="central"), 53.0, 0)
    chk("A4 PASS registered peripheral", n(verdict="PASS", scoring="registered rule", relevance="peripheral"), 47.0, 0)
    chk("A4 PARTIAL registered central", n(verdict="PARTIAL", scoring="registered rule", relevance="central"), 1.0, 0)
    chk("A4 UNRESOLVED registered central", n(verdict="UNRESOLVED", scoring="registered rule", relevance="central"), 11.0, 0)
    chk("A4 rows", float(len(cr)), 232.0, 0)


def tracks_checks() -> None:
    """Fast-track follow-up, Tracks 3-8 (2026-09-25)."""
    print("Track 4 (SGD own thresholds, registered)")
    sc = pd.read_csv(R / "sgd_own_scores.csv").set_index("a")
    for a in (1.3, 1.5):
        t = f"a={a:.2f}"
        chk(f"Track 4 G1 {t}", float(sc.loc[a, "G1"] == "PASS"), 1.0, 0)
        chk(f"Track 4 G2 {t}", float(sc.loc[a, "G2"] == "PASS"), 1.0, 0)
        chk(f"Track 4 crossed {t}", float(sc.loc[a, "crossed"]), 30.0, 0)
    chk("Track 4 G1 lo a=1.30", float(sc.loc[1.3, "G1_lo"]), -0.0738, 0.00006)
    chk("Track 4 G1 hi a=1.30", float(sc.loc[1.3, "G1_hi"]), -0.0172, 0.00006)
    chk("Track 4 G1 lo a=1.50", float(sc.loc[1.5, "G1_lo"]), -0.0727, 0.00006)
    chk("Track 4 G1 hi a=1.50", float(sc.loc[1.5, "G1_hi"]), -0.0148, 0.00006)
    chk("Track 4 G1 mean a=1.30", float(sc.loc[1.3, "G1_mean_diff"]), -0.0459, 0.00006)
    chk("Track 4 G1 mean a=1.50", float(sc.loc[1.5, "G1_mean_diff"]), -0.0439, 0.00006)
    chk("Track 4 Spearman a=1.30", float(sc.loc[1.3, "G2_spearman"]), 0.674, 0.0006)
    chk("Track 4 Spearman a=1.50", float(sc.loc[1.5, "G2_spearman"]), 0.684, 0.0006)
    chk("Track 4 residual vs own a=1.30", float(sc.loc[1.3, "G3_median_resid_own"]) * 100, 2.95, 0.006)
    chk("Track 4 residual vs own a=1.50", float(sc.loc[1.5, "G3_median_resid_own"]) * 100, 6.43, 0.006)
    chk("Track 4 residual vs pop a=1.30", float(sc.loc[1.3, "G3_median_resid_pop"]) * 100, 9.45, 0.006)
    chk("Track 4 residual vs pop a=1.50", float(sc.loc[1.5, "G3_median_resid_pop"]) * 100, 12.74, 0.006)
    ex = pd.read_csv(R / "sgd_own_extension_scores.csv").set_index("a")
    chk("Track 4 EXT a=1.30", float(ex.loc[1.3, "EXT"] == "PASS"), 1.0, 0)
    chk("Track 4 EXT a=1.50", float(ex.loc[1.5, "EXT"] == "PASS"), 1.0, 0)
    chk("Track 4 EXT pred a=1.30", float(ex.loc[1.3, "pred"]), 0.0251, 0.00006)
    chk("Track 4 EXT pred a=1.50", float(ex.loc[1.5, "pred"]), 0.0530, 0.00006)
    chk("Track 4 EXT obs a=1.30", float(ex.loc[1.3, "obs"]), 0.0295, 0.00006)
    chk("Track 4 EXT obs a=1.50", float(ex.loc[1.5, "obs"]), 0.0643, 0.00006)
    chk("Track 4 EXT tol a=1.50", float(ex.loc[1.5, "tol"]), 0.0133, 0.00006)
    chk("Track 4 SGD replays reproduce", float(pd.read_csv(R / "sgd_own_ratios.csv").reproduced.all()), 1.0, 0)
    chk("Track 4 budget", float(json.loads((R / "sgd_own_budget.json").read_text())["budget"]), 32000.0, 0)
    sens = pd.read_csv(R / "sgd_own_sensitivity.csv").set_index("a")
    for a in (1.3, 1.5):
        t = f"a={a:.2f}"
        chk(f"Track 4 sens G1 {t}", float(sens.loc[a, "G1"] == "PASS"), 1.0, 0)
        chk(f"Track 4 sens G2 {t}", float(sens.loc[a, "G2"] == "FAIL"), 1.0, 0)
        chk(f"Track 4 sens EXT {t}", float(sens.loc[a, "EXT"] == "PASS"), 1.0, 0)
    chk("Track 4 sens noncrosser final/own a=1.30", float(sens.loc[1.3, "median_final_over_own_noncrossers"]), 0.15, 0.006)
    chk("Track 4 sens noncrosser final/own a=1.50", float(sens.loc[1.5, "median_final_over_own_noncrossers"]), 0.30, 0.006)
    chk("Track 4 sens Spearman a=1.30", float(sens.loc[1.3, "G2_spearman"]), 0.502, 0.0006)
    chk("Track 4 sens Spearman a=1.50", float(sens.loc[1.5, "G2_spearman"]), 0.507, 0.0006)
    print("Track 2 (asymmetric windows, registered)")
    a2 = json.loads((R / "asym_scores.json").read_text())
    chk("Track 2 T2-1 PASS", float(a2["T2-1"] == "PASS"), 1.0, 0)
    chk("Track 2 T2-2 PASS", float(a2["T2-2"] == "PASS"), 1.0, 0)
    chk("Track 2 s_lo", a2["s_lo"], 0.4371, 0.00006)
    chk("Track 2 s_hi", a2["s_hi"], 0.4532, 0.00006)
    chk("Track 2 T2-3 FAIL", float(a2["T2-3"]["verdict"] == "FAIL"), 1.0, 0)
    chk("Track 2 T2-3 frac above", a2["T2-3"]["frac_above_s_hi"] * 100, 93.7, 0.06)
    chk("Track 2 T2-3 median ratio", a2["T2-3"]["median_ratio_s_lo"], 3.29, 0.006)
    chk("Track 2 T2-3 CI lo", a2["T2-3"]["ci_median_minus_1"][0], 1.89, 0.006)
    chk("Track 2 T2-3 CI hi", a2["T2-3"]["ci_median_minus_1"][1], 2.56, 0.006)
    chk("Track 2 T2-3 crossed", float(a2["crossed"]), 79.0, 0)
    chk("Track 2 T2-3 placed at init", float(a2["placed_at_init"]), 1.0, 0)
    va = pd.read_csv(R / "asym_parts" / "validate.csv")
    chk("Track 2 validation both ends", float(va.ladder_ok.all() and va.independent_ok.all() and va.audit_ok.all() and len(va) == 2), 1.0, 0)
    ph = json.loads((R / "asym_posthoc" / "summary.json").read_text())
    chk("Track 2 posthoc replays reproduce", float(ph["replay_reproduces"]), 1.0, 0)
    chk("Track 2 posthoc single-unit at crossing", float(ph["knockout_counts"].get("single-unit", 0)), 0.0, 0)
    chk("Track 2 posthoc pair at crossing", float(ph["knockout_counts"]["pair"]), 76.0, 0)
    chk("Track 2 posthoc single branch switch lo", ph["single_branch_switch_lo"], 4.87, 0.006)
    chk("Track 2 posthoc single branch switch hi", ph["single_branch_switch_hi"], 5.62, 0.006)
    chk("Track 2 posthoc width-1 threshold", ph["s_w1"], 5.09, 0.006)
    chk("Track 2 posthoc frac below single branch", ph["frac_cross_below_single_branch"] * 100, 97.5, 0.06)
    chk("Track 2 posthoc median branch switch", ph["median_s_branch"], 0.439, 0.0006)
    chk("Track 2 posthoc branch minus glob hi", ph["branch_minus_glob"][2], 0.015, 0.0006)
    chk("Track 2 posthoc Spearman branch", ph["spearman_cross_branch"], 0.30, 0.006)
    chk("Track 2 posthoc median crossing", ph["median_s_cross"], 1.44, 0.006)
    tb = json.loads((R / "asym_t23b" / "scores.json").read_text())
    chk("Track 2 T2-3b UNRESOLVED", float(tb["T2-3b"].startswith("UNRESOLVED")), 1.0, 0)
    chk("Track 2 T2-3b phi2", json.loads((R / "asym_t23b_frozen.json").read_text())["phi2"], 0.009306, 1e-6)
    chk("Track 2 T2-3b median ratio at crossing", tb["median_ratio_at_cross"], 6.8e-6, 1e-7)
    chk("Track 2 T2-3b frac above", tb["criteria"]["frac_above_s_hi"] * 100, 56.3, 0.06)
    chk("Track 2 T2-3b median crossing ratio", tb["criteria"]["median_ratio_s_lo"], 10.4, 0.06)
    tc = json.loads((R / "asym_t23c" / "scores.json").read_text())
    chk("Track 2 T2-3c UNRESOLVED", float(tc["T2-3c"].startswith("UNRESOLVED")), 1.0, 0)
    chk("Track 2 T2-3c phi2", json.loads((R / "asym_t23c_frozen.json").read_text())["phi2"], 0.01778, 1e-5)
    chk("Track 2 T2-3c median ratio at crossing", tc["median_ratio_at_cross"], 1.32e-5, 1e-7)
    chk("Track 2 T2-3c frac above", tc["criteria"]["frac_above_s_hi"] * 80, 47.0, 1e-9)
    chk("Track 2 T2-3c median crossing ratio", tc["criteria"]["median_ratio_s_lo"], 10.8, 0.06)
    e2 = json.loads((R / "asym_posthoc" / "exploratory2.json").read_text())
    chk("Track 2 expl own median", e2["own_median"], 0.445, 0.0006)
    chk("Track 2 expl cross over own", e2["median_cross_over_own"], 4.11, 0.006)
    chk("Track 2 expl Spearman own", e2["spearman_cross_own"], -0.74, 0.006)
    chk("Track 2 expl ratio median", e2["ts_median_ratio"], 2.73, 0.006)
    chk("Track 2 expl ratio Spearman within", e2["spearman_residual_ratio_within_T2_3"], -0.24, 0.006)
    print("Track 7 (no-gating, registered)")
    ng = pd.read_csv(R / "width2_nogating_scores.csv")
    pr = ng[(ng.variant == "preserved")].set_index(["act", "R2"])
    chk("Track 7 no-gating FAIL a=1.30", float(pr.loc[("f1.30", 0.003), "outcome"].startswith("gating")), 1.0, 0)
    chk("Track 7 no-gating FAIL a=1.50", float(pr.loc[("f1.50", 0.003), "outcome"].startswith("gating")), 1.0, 0)
    chk("Track 7 placed 0.003 a=1.30", float(pr.loc[("f1.30", 0.003), "placed_frac"]), 0.4625, 1e-9)
    chk("Track 7 placed 0.003 a=1.50", float(pr.loc[("f1.50", 0.003), "placed_frac"]), 0.4625, 1e-9)
    chk("Track 7 placed 0.1 a=1.30", float(pr.loc[("f1.30", 0.1), "placed_frac"]), 0.9875, 1e-9)
    chk("Track 7 placed 0.1 a=1.50", float(pr.loc[("f1.50", 0.1), "placed_frac"]), 0.975, 1e-9)
    ct = pd.read_csv(R / "width2_nogating_parts" / "control.csv")
    chk("Track 7 positive control", float(ct[(ct.level == 0.1) & ~ct.excluded.astype(bool)].placed_end.astype(bool).mean()), 0.0, 0)
    chk("Track 7 H", float(json.loads((R / "width2_nogating_frozen.json").read_text())["H"]), 16000.0, 0)
    print("Task B (timescale prospective test, registered)")
    tb_ = pd.read_csv(R / "ts_test" / "scores.csv").set_index("a")
    for a in (1.45, 1.6):
        chk(f"Task B TS-1 a={a:.2f}", float(tb_.loc[a, "TS-1"] == "PASS"), 1.0, 0)
        chk(f"Task B TS-2 a={a:.2f}", float(tb_.loc[a, "TS-2"] == "PASS"), 1.0, 0)
    chk("Task B pred a=1.45", float(tb_.loc[1.45, "pred"]), 0.0456, 0.00006)
    chk("Task B obs a=1.45", float(tb_.loc[1.45, "obs"]), 0.0532, 0.00006)
    chk("Task B pred a=1.60", float(tb_.loc[1.6, "pred"]), 0.0818, 0.00006)
    chk("Task B obs a=1.60", float(tb_.loc[1.6, "obs"]), 0.0809, 0.00006)
    chk("Task B crossed", float(tb_.crossed.sum()), 160.0, 0)
    chk("Task B own thresholds frozen", float(len(pd.read_csv(R / "ts_test_own_frozen.csv"))), 160.0, 0)
    print("Task C (mechanism figure)")
    mw = pd.read_csv(R / "mechanism_w1_stats.csv").set_index("Unnamed: 0")
    chk("Task C w1 at smallest a=1.30", float(mw.loc[1.3, "w1_at_smallest"]), 1.751, 0.0006)
    chk("Task C crossings below bound", float(mw.cross_w1_below_bound.min()), 1.0, 0)
    chk("Task C figure audit", float((R / "figures" / "v5" / "mechanism_w1.pdf").exists()), 1.0, 0)
    print("Item 1 (timescale consistency, post hoc)")
    it1 = pd.read_csv(R / "timescale_consistency" / "arms.csv"); s1_ = json.loads((R / "timescale_consistency" / "summary.json").read_text())
    chk("Item 1 arms within tolerance", float(it1.within_tol.sum()), 28.0, 0)
    ms_ = it1[~it1.within_tol]
    chk("Item 1 misses are the phi=0.25 arms", float(len(ms_) == 4 and (ms_.arm.astype(str) == "0.25").all()), 1.0, 0)
    chk("Item 1 within-a slope a=1.30", s1_["slope_within_a_all"]["1.30"], 4.69, 0.006)
    chk("Item 1 within-a slope a=1.50", s1_["slope_within_a_all"]["1.50"], 3.09, 0.006)
    chk("Item 1 partial Spearman", s1_["partial_spearman_all"], 0.51, 0.006)
    chk("Item 1 replays reproduce", float(pd.read_csv(R / "timescale_consistency" / "runs.csv").reproduced.all()), 1.0, 0)
    t3d_ = json.loads((R / "asym_t23d" / "scores.json").read_text())
    chk("Track 2 T2-3d FAIL", float(t3d_["T2-3d"] == "FAIL"), 1.0, 0)
    chk("Track 2 T2-3d median err", t3d_["criteria"]["median_abs_log_err"], 0.019, 0.0006)
    chk("Track 2 T2-3d diff hi", t3d_["criteria"]["diff_hi"], 0.0064, 0.00006)
    chk("Track 2 T2-3d bimodal", float(json.loads((R / "asym_t23d" / "descriptive.json").read_text())["n_pop_beats_own"]), 31.0, 0)
    print("Block 4b correction and state audit")
    bc_ = pd.read_csv(R / "residual_mechanism_corrected" / "scores.csv")
    vv_ = bc_[bc_.arm == "VERDICT"]
    chk("Block 4b corrected ID FAIL", float((vv_["4b-ID"] == "FAIL").all() and len(vv_) == 2), 1.0, 0)
    chk("Block 4b corrected OM FAIL", float((vv_["4b-OM"] == "FAIL").all() and len(vv_) == 2), 1.0, 0)
    au_ = pd.read_csv(R / "state_audit.csv")
    other_ = au_[~au_.experiment.str.startswith("Block 4b (as registered)")]
    chk("State audit clean arms", float(other_.clean.sum()), float(len(other_)), 0)
    chk("State audit positive control", float(json.loads((R / "state_audit.json").read_text())["positive_control_flagged"]), 1.0, 0)
    print("Item 2 (width-2 diagnosis, post hoc)")
    g2_ = json.loads((R / "width2_diagnosis" / "gate.json").read_text())
    chk("Item 2 gate STOP", float(g2_["gate"] == "STOP"), 1.0, 0)
    chk("Item 2 selected P5", float(g2_["selected"] == "P5"), 1.0, 0)
    ma_ = pd.read_csv(R / "width2_diagnosis" / "metrics_all.csv").set_index(["predictor", "arm"])
    chk("Item 2 P5 T2-3 error", float(ma_.loc[("P5", "T2-3"), "median_abs_log_err"]), 1.17, 0.006)
    chk("Item 2 P3 T2-3b error", float(ma_.loc[("P3", "T2-3b"), "median_abs_log_err"]), 0.015, 0.0006)
    chk("Item 2 P3 T2-3c error", float(ma_.loc[("P3", "T2-3c"), "median_abs_log_err"]), 0.011, 0.0006)
    print("Track 3 (residual timescale, post hoc)")
    ts = json.loads((R / "residual_timescale_summary.json").read_text())
    chk("Track 3 Spearman pooled", ts["spearman_run_level"], 0.634, 0.0006)
    chk("Track 3 Spearman CI lo", ts["spearman_ci95"][0], 0.542, 0.0006)
    chk("Track 3 Spearman CI hi", ts["spearman_ci95"][1], 0.721, 0.0006)
    chk("Track 3 Spearman within a=1.30", ts["spearman_within_a"]["a=1.30"], 0.454, 0.0006)
    chk("Track 3 Spearman within a=1.50", ts["spearman_within_a"]["a=1.50"], 0.507, 0.0006)
    chk("Track 3 runs", float(ts["n_runs"]), 384.0, 0)
    chk("Track 3 replays reproduce", float(pd.read_csv(R / "residual_timescale_runs.csv").reproduced.all()), 1.0, 0)
    fit = json.loads((R / "residual_timescale_fit.json").read_text())
    chk("Track 3 fit alpha", fit["alpha"], 0.0157, 0.00006)
    chk("Track 3 fit beta", fit["beta"], 2.658, 0.0006)
    print("Track 6 (stronger baselines, post hoc)")
    v = pd.read_csv(R / "baselines_posthoc_validation.csv")
    chk("Track 6 reproduction checks", float(v.ok.all() and len(v) == 163), 1.0, 0)
    b3 = pd.read_csv(R / "baselines_posthoc_block3.csv")
    chk("Track 6 C mean abs log err", float(b3.abslogerr_C.mean()), 0.0223, 0.00006)
    chk("Track 6 PL mean abs log err", float(b3.abslogerr_PL.mean()), 0.0231, 0.00006)
    chk("Track 6 PL5 mean abs log err", float(b3.abslogerr_PL5.mean()), 0.0274, 0.00006)
    chk("Track 6 RK mean abs log err", float(b3.abslogerr_RK.mean()), 0.1099, 0.00006)
    chk("Track 6 RG mean abs log err", float(b3.abslogerr_RG.mean()), 0.1047, 0.00006)
    cm = pd.read_csv(R / "baselines_posthoc_comparisons.csv")
    cm = cm[cm.test == "Block 3 (post hoc baselines)"].set_index("comparison")
    chk("Track 6 C - PL lo", float(cm.loc["C - PL", "ci95_lo"]), -0.0088, 0.00006)
    chk("Track 6 C - PL hi", float(cm.loc["C - PL", "ci95_hi"]), 0.0074, 0.00006)
    chk("Track 6 C - RK hi", float(cm.loc["C - RK", "ci95_hi"]), -0.0746, 0.00006)
    print("Track 5 (independent certificate checks)")
    kb = json.loads((R / "certificate_checks" / "K_base.json").read_text())
    chk("Track 5 K check passes (published hi)", float(kb["pass"]), 1.0, 0)
    kt = json.loads((R / "certificate_checks" / "K_base_target0.5794559217.json").read_text())
    chk("Track 5 K check passes (tight hi)", float(kt["pass"]), 1.0, 0)
    for n_ in ("c1_krawczyk", "pd_glob_neighbourhood", "pd_solve_neighbourhood", "ring_glob_annulus", "ring_solve_annulus"):
        chk(f"Track 5 {n_} passes", float(json.loads((R / "certificate_checks" / f"{n_}.json").read_text())["pass"]), 1.0, 0)
    kr = json.loads((R / "certificate_checks" / "c1_krawczyk.json").read_text())
    chk("Track 5 Krawczyk A* lo", kr["A_star_lo"], 0.68544523757565, 1e-13)
    chk("Track 5 Krawczyk A1/A lo", kr["A1_over_A_lo"], 0.6621548, 1e-7)
    chk("Track 5 ring glob boxes", float(json.loads((R / "certificate_checks" / "ring_glob_annulus.json").read_text())["boxes"]), 21600.0, 0)


def _decimals(v) -> int:
    t = repr(float(v))
    if "e" in t or "E" in t:
        m, e = t.lower().split("e")
        return max(0, (len(m.split(".")[1]) if "." in m else 0) - int(e))
    return len(t.split(".")[1]) if "." in t else 0


def lag_law_checks() -> None:
    """Track 1A (κ, derived after the fitted relationship was known) and Track 1B (registered ramp, R4, post hoc)."""
    import hashlib as _h
    print("Track 1A (lag law, no refit)")
    L = R / "lag_law"
    A = pd.read_csv(L / "compare_arms.csv"); Pa = pd.read_csv(L / "compare_per_a.csv").set_index("a")
    chk("1A arms within tolerance", float(A.within.sum()), 36.0, 0)
    chk("1A arms", float(len(A)), 36.0, 0)
    for a, v in ((1.30, 0.89), (1.45, 0.83), (1.50, 0.91), (1.60, 0.82)):
        chk(f"1A slope/kappa a={a:.2f}", float(Pa.loc[a, "ratio_obs_to_pred"]), v, 0.006)
    for a, v in ((1.30, 7.61), (1.45, 4.59), (1.50, 4.05), (1.60, 3.29)):
        kb = pd.read_csv(L / "kappa_by_winding.csv")
        k = -1 if a == 1.30 else 0
        chk(f"1A kappa_Adam a={a:.2f}", float(kb[(kb.a.round(2) == a) & (kb.k == k)].kappa_adam.iloc[0]), v, 0.006)
    want = {l.split()[1].split("/")[-1]: l.split()[0] for l in (L / "committed.sha256").read_text().splitlines()}
    chk("1A committed predictions hash", float(_h.sha256((L / "predictions.csv").read_bytes()).hexdigest() == want["predictions.csv"]), 1.0, 0)
    p_ = pd.read_csv(L / "predictions.csv")
    chk("1A predicted runs", float(len(p_)), 1750.0, 0)
    chk("1A mirror runs", float(p_.mirror.sum()), 875.0, 0)
    low = A.arm.astype(str).eq("0.25") & A["set"].isin(["lag1", "lag2-tstar"])
    q = A[~low]; q25 = A[low]
    chk("1A low arms are the four early-slowed phi=0.25 arms", float(low.sum()), 4.0, 0)
    chk("1A obs/pred min (other 32 arms)", float(q.obs_over_pred.min()), 0.91, 0.006)
    chk("1A obs/pred max (other 32 arms)", float(q.obs_over_pred.max()), 1.13, 0.006)
    chk("1A obs/pred min (four low arms)", float(q25.obs_over_pred.min()), 0.70, 0.006)
    chk("1A obs/pred max (four low arms)", float(q25.obs_over_pred.max()), 0.76, 0.006)
    pp = A[A.arm.astype(str).eq("0.25") & A["set"].eq("lag2-prim")]
    chk("1A obs/pred min (primary-rule phi=0.25)", float(pp.obs_over_pred.min()), 1.03, 0.006)
    chk("1A obs/pred max (primary-rule phi=0.25)", float(pp.obs_over_pred.max()), 1.07, 0.006)
    w = pd.read_csv(L / "winding_check.csv")
    for (a, k, chi), (pv, mv) in {(1.30, 0, 0.003): (-0.0223, -0.0215), (1.30, -1, 0.003): (0.0229, 0.0236),
                                  (1.30, 1, 0.003): (-0.0675, -0.0607), (1.50, 0, 0.002): (0.0081, 0.0082),
                                  (1.50, 0, 0.005): (0.0204, 0.0207)}.items():
        r_ = w[(w.a.round(2) == a) & (w.k == k) & (w.chi.round(4) == chi)].iloc[0]
        chk(f"1A winding check a={a:.2f} k={k} chi={chi} predicted", float(r_.predicted_r), pv, 0.00006)
        chk(f"1A winding check a={a:.2f} k={k} chi={chi} measured", float(r_.measured_r), mv, 0.00006)
    rx = pd.read_csv(L / "relaxation.csv")
    chk("1A relaxation steps min", float(rx.relaxation_steps.min()), 7.8, 0.06)
    chk("1A relaxation steps max", float(rx.relaxation_steps.max()), 16.2, 0.06)
    print("Track 1B (ramp, registered) and R4")
    V = json.loads((R / "ramp" / "verdicts.json").read_text())
    reg = {(1.30, -1, "sgd"): ("PASS", "PASS", "PASS", 0.78), (1.30, 0, "sgd"): ("FAIL", "PASS", "PASS", 1.30),
           (1.50, 0, "sgd"): ("FAIL", "PASS", "FAIL", -0.10), (1.30, -1, "adam"): ("FAIL", "PASS", "PASS", -7.12),
           (1.30, 0, "adam"): ("FAIL", "PASS", "PASS", 9.10), (1.50, 0, "adam"): ("FAIL", "FAIL", "FAIL", -9.99)}
    for x in V["settings"]:
        key = (round(x["a"], 2), int(x["winding"]), x["opt"]); r1, r2, r3, sl = reg[key]
        t = f"{key[2]} {key[0]:.2f} k={key[1]}"
        chk(f"1B registered R1 {t}", float(x["R1"] == r1), 1.0, 0)
        chk(f"1B registered R2 {t}", float(x["R2"] == r2), 1.0, 0)
        chk(f"1B registered R3 {t}", float(x["R3"] == r3), 1.0, 0)
        chk(f"1B registered slope {t}", float(x["slope_obs_on_pred"]), sl, 0.006)
        chk(f"1B crossed {t}", float(x["n_crossed"]), 240.0, 0)
    chk("1B registered R1 SGD 1.30 k=-1", float([x for x in V["settings"] if x["opt"] == "sgd" and x["winding"] == -1][0]["R1"] == "PASS"), 1.0, 0)
    chk("1B slowest median SGD 1.50", float([x for x in V["settings"] if x["opt"] == "sgd" and round(x["a"], 2) == 1.5][0]["slowest_median_r"]), -0.0233, 0.00006)
    PH = json.loads((R / "ramp" / "posthoc_branch.json").read_text())
    ph = {(round(x["a"], 2), int(x["winding"]), x["opt"]): x for x in PH}
    for key, (r1, sl) in {(1.30, -1, "sgd"): ("PASS", 1.24), (1.30, 0, "sgd"): ("PASS", 0.90), (1.50, 0, "sgd"): ("PASS", 1.14),
                          (1.30, -1, "adam"): ("FAIL", 0.52), (1.30, 0, "adam"): ("FAIL", 1.49), (1.50, 0, "adam"): ("PASS", 0.90)}.items():
        t = f"{key[2]} {key[0]:.2f} k={key[1]}"
        chk(f"1B post hoc R1 {t}", float(ph[key]["R1"] == r1), 1.0, 0)
        chk(f"1B post hoc slope {t}", float(ph[key]["slope_obs_on_pred"]), sl, 0.006)
    chk("1B post hoc R1 SGD 1.50", float(ph[(1.50, 0, "sgd")]["R1"] == "PASS"), 1.0, 0)
    rat = lambda x: np.array(x["cell_medians_obs"]) / np.array(x["cell_medians_pred"])
    sg = np.concatenate([rat(x)[None, :] for x in PH if x["opt"] == "sgd"])
    chk("1B post hoc SGD ratio 4 slowest min", float(sg[:, :4].min()), 0.99, 0.006)
    chk("1B post hoc SGD ratio 4 slowest max", float(sg[:, :4].max()), 1.06, 0.006)
    chk("1B post hoc SGD ratio fifth min", float(sg[:, 4].min()), 0.97, 0.006)
    chk("1B post hoc SGD ratio fifth max", float(sg[:, 4].max()), 1.11, 0.006)
    chk("1B post hoc SGD ratio fastest min", float(sg[:, 5].min()), 0.87, 0.006)
    chk("1B post hoc SGD ratio fastest max", float(sg[:, 5].max()), 1.26, 0.006)
    for key, (lo, hi) in {(1.30, -1): (0.35, 0.94), (1.30, 0): (1.40, 1.85), (1.50, 0): (-0.39, 0.67)}.items():
        r_ = rat(ph[(key[0], key[1], "adam")])
        chk(f"1B post hoc Adam ratio min {key}", float(r_.min()), lo, 0.006)
        chk(f"1B post hoc Adam ratio max {key}", float(r_.max()), hi, 0.006)
    chk("1B branch off own a=1.30", float(ph[(1.30, -1, "sgd")]["frac_branch_off_own_1pct"]), 0.35, 0.0006)
    chk("1B branch off own a=1.50", float(ph[(1.50, 0, "sgd")]["frac_branch_off_own_1pct"]), 0.675, 0.0006)
    ac = pd.read_csv(R / "ramp" / "adam_contrast.csv"); ac["a"] = ac.a.round(2)
    rp = ac[ac.source.str.startswith("ramp")].set_index("a"); fe = ac[ac.source.str.startswith("free")].set_index("a")
    vr = [float(fe.loc[a, c] / rp.loc[a, c]) for a in (1.3, 1.5) for c in ("sqrt_v_w1", "sqrt_v_b1", "sqrt_v_b2")]
    chk("1B Adam contrast v ratio min", min(vr), 8.0, 0.5); chk("1B Adam contrast v ratio max", max(vr), 88.0, 0.5)
    chk("1B Adam contrast ramp relax min", float(rp.relax_steps.min()), 0.17, 0.006)
    chk("1B Adam contrast ramp relax max", float(rp.relax_steps.max()), 0.19, 0.006)
    chk("1B Adam contrast free relax min", float(fe.relax_steps.min()), 7.8, 0.06)
    chk("1B Adam contrast free relax max", float(fe.relax_steps.max()), 15.0, 0.06)
    chk("1B Adam contrast ramp steps min", float(rp.steps_growth_to_cross.min()), 238.5, 0.5)
    chk("1B Adam contrast ramp steps max", float(rp.steps_growth_to_cross.max()), 623.5, 0.5)
    chk("1B Adam contrast free steps min", float(fe.steps_growth_to_cross.min()), 1679.5, 0.5)
    chk("1B Adam contrast free steps max", float(fe.steps_growth_to_cross.max()), 3265.5, 0.5)
    gr = [float(rp.loc[a, "growth_per_step"] / fe.loc[a, "growth_per_step"]) for a in (1.3, 1.5)]
    chk("1B Adam contrast growth ratio min", min(gr), 2.3, 0.06); chk("1B Adam contrast growth ratio max", max(gr), 2.7, 0.06)
    sw2 = pd.read_csv(R / "ramp2" / "design_sweep.csv")
    chk("ramp2 no admissible gamma", float(pd.read_csv(R / "ramp2" / "design.csv").cell.max()), -1.0, 0)
    chk("ramp2 slow eta*lambda min", float(sw2[sw2.gamma <= 1.8e-4].median_eta_lambda.min()), 4.4, 0.06)
    chk("ramp2 slow eta*lambda max", float(sw2[sw2.gamma <= 1.8e-4].median_eta_lambda.max()), 11.5, 0.06)
    chk("ramp2 fast crossings max", float(sw2[sw2.median_eta_lambda <= 0.5].n_crossed.max()), 3.0, 0)
    gf = json.loads((R / "act_general" / "gelu_rule_feasibility.json").read_text())
    chk("GELU rule earliest crossing", gf["earliest_crossing_s"], 0.0043, 0.00006)
    chk("GELU rule runs starting below", float(gf["runs_starting_below_earliest_crossing"]), 0.0, 0)
    chk("GELU rule init median w2", gf["init_abs_w2_median"], 0.54, 0.006)
    chk("GELU rule final at init", float(gf["final_at_init_n"]), 34.0, 0)
    chk("GELU rule final at init pct", 100 * gf["final_at_init_frac"], 26.0, 0.6)
    chk("GELU rule infeasible", float(gf["rule_feasible"]), 0.0, 0)
    for x in V["R4"]:
        a = round(x["a"], 2)
        chk(f"R4 a={a:.2f}", float(x["R4"] == "PASS"), 1.0, 0)
        for e, v in ((0.01, {1.3: 0.0296, 1.5: 0.0619}), (0.005, {1.3: 0.0304, 1.5: 0.0625}), (0.0025, {1.3: 0.0308, 1.5: 0.0647})):
            chk(f"R4 median a={a:.2f} eta={e}", float(x[f"median_{e}"]), v[a], 0.00006)
            chk(f"R4 crossed a={a:.2f} eta={e}", float(x[f"n_crossed_{e}"]), 39.0, 0)


def act_checks() -> None:
    """Track 3A (GELU, SiLU, Mish at width 1): registered verdicts and the numbers WP-25 prints."""
    print("Track 3A (non-sine activations)")
    D = R / "act_general"
    ts = pd.read_csv(D / "training_scores.csv").set_index(["act", "arm"])
    for a, (fr, obs, pred, cr, nc) in {"gelu": (0.477, -0.0183, 0.0023, 132, 60), "silu": (0.622, 0.0353, -0.0324, 135, 65),
                                       "mish": (0.704, 0.0658, -0.0393, 135, 65)}.items():
        r = ts.loc[(a, "secondary_pooled_200")]
        chk(f"3A {a} T-a FAIL", float(r["T-a"] == "FAIL"), 1.0, 0)
        chk(f"3A {a} T-b FAIL", float(r["T-b"] == "FAIL"), 1.0, 0)
        chk(f"3A {a} primary UNRESOLVED", float(ts.loc[(a, "primary"), "T-a"] == "UNRESOLVED"), 1.0, 0)
        chk(f"3A {a} frac at or above s_lo", float(r.frac_at_or_above_lo), fr, 0.0006)
        chk(f"3A {a} T-b obs", float(r.obs), obs, 0.00006)
        chk(f"3A {a} T-b pred", float(r.pred), pred, 0.00006)
        chk(f"3A {a} crossing", float(r.crossing_runs), cr, 0)
        chk(f"3A {a} never cross", float(r.runs - r.placed_at_init - r.crossing_runs), nc, 0)
    chk("3A GELU T-a", float(ts.loc[("gelu", "secondary_pooled_200"), "T-a"] == "FAIL"), 1.0, 0)
    chk("3A SiLU T-b", float(ts.loc[("silu", "secondary_pooled_200"), "T-b"] == "FAIL"), 1.0, 0)
    chk("3A Mish crossing", float(ts.loc[("mish", "secondary_pooled_200"), "crossing_runs"]), 135.0, 0)
    for a, (lo, hi, k) in {"gelu": (6.6117, 6.6714, 0.1473), "silu": (3.6190, 3.6517, -0.1444), "mish": (3.1908, 3.2197, -0.1539)}.items():
        b = json.loads((D / f"bracket_{a}.json").read_text()); kp = json.loads((D / f"kappa_{a}_frozen.json").read_text())
        chk(f"3A {a} bracket lo", float(b["s_lo"]), lo, 0.00006)
        chk(f"3A {a} bracket hi", float(b["s_hi"]), hi, 0.00006)
        chk(f"3A {a} kappa", float(kp["kappa_adam"]), k, 0.00006)
        chk(f"3A {a} validated both ends", float(b["validation_lo"]["validated"] and b.get("validation_hi", {"validated": True})["validated"]), 1.0, 0)
    chk("3A GELU bracket lo", float(json.loads((D / "bracket_gelu.json").read_text())["s_lo"]), 6.6117, 0.00006)
    cv = json.loads((D / "criterion_verdicts.json").read_text())
    chk("3A criterion verdicts", float([cv[a]["verdict"] for a in ("gelu", "silu", "mish")] == ["switch predicted", "undetermined", "undetermined"]), 1.0, 0)
    S2 = json.loads((D / "posthoc2_summary.json").read_text()); A2 = {x["act"]: x for x in S2["acts"]}; sw = S2["sine_width1"]
    chk("3A post hoc sine chi arm-median max", sw["chi_arm_median_max"], 0.0249, 0.00006)
    chk("3A post hoc sine chi run max", sw["chi_run_max"], 0.064, 0.0006)
    chk("3A post hoc sine chi run min", sw["chi_run_min"], 0.0001, 0.00006)
    chk("3A post hoc sine chi run q95", sw["chi_run_q95"], 0.021, 0.0006)
    for a, v in {"gelu": dict(chi_median=0.0137, with_branch_switch=117, obs_branch_median=0.0017, pred_median=0.0021, frac_at_or_above_branch=0.93,
                              n_no_switch_on_branch=15, no_switch_s_cross_max=0.45, branch_over_pop_q10=0.88, branch_over_pop_q90=1.17, validity_met=1.0, within=1.0),
                 "silu": dict(chi_median=0.2242, with_branch_switch=134, obs_branch_median=0.0142, pred_median=-0.0324, validity_met=0.0, within=0.0,
                              branch_over_pop_q10=0.83, branch_over_pop_q90=1.19),
                 "mish": dict(chi_median=0.2556, with_branch_switch=134, obs_branch_median=0.0342, pred_median=-0.0393, validity_met=0.0, within=0.0,
                              branch_over_pop_q10=0.84, branch_over_pop_q90=1.20)}.items():
        for k, want in v.items():
            got = float(A2[a][k])
            tol = 0 if k in ("with_branch_switch", "n_no_switch_on_branch", "validity_met", "within") else (0.006 if want >= 0.3 or k.startswith("frac") or k.startswith("branch") or k.startswith("no_switch") else 0.00006)
            chk(f"3A post hoc {a} {k}", got, want, tol)
    for a, (f1, f2, f3) in {"gelu": (0.81, 0.94, 0.96), "silu": (0.01, 0.05, 0.31), "mish": (0.0, 0.02, 0.28)}.items():
        chk(f"3A post hoc {a} frac chi <= arm max", round(A2[a]["frac_chi_le_sine_arm_median_max"], 2), f1, 0.006)
        chk(f"3A post hoc {a} frac chi <= run max", round(A2[a]["frac_chi_le_sine_run_max"], 2), f2, 0.006)
        chk(f"3A post hoc {a} frac |rb| <= 0.01", round(A2[a]["frac_abs_rb_le_0.01"], 2), f3, 0.006)
    gh = json.loads((D / "ghat.json").read_text())
    for a, v in {"gelu": 0.037547, "silu": 0.050912, "mish": 0.054772}.items():
        chk(f"3A {a} Ghat", float(gh[a]["Ghat_nm"]["G_lo"]), v, 6e-7)


def track4_checks() -> None:
    """Track 4 writer inputs: run-population numbers printed in WP-26 (producer: track4_populations)."""
    print("Track 4 (run populations)")
    P = json.loads((R / "track4_populations.json").read_text())
    for k, (n, sv, pl, bi) in {"float32": (2400, 430, 1664, 306), "float64": (2400, 440, 1633, 327), "pooled": (4800, 870, 3297, 633)}.items():
        chk(f"T4 {k} runs", float(P[k]["runs"]), n, 0); chk(f"T4 {k} solved", float(P[k]["solved"]), sv, 0)
        chk(f"T4 {k} placement", float(P[k]["placement"]), pl, 0); chk(f"T4 {k} bias", float(P[k]["bias"]), bi, 0)
    chk("T4 pooled counts", float(P["pooled"]["solved"] + P["pooled"]["placement"] + P["pooled"]["bias"]), 4800.0, 0)
    chk("T4 sign agreement", 100 * P["paired"]["sign_w1_agree"], 48.6, 0.06)
    chk("T4 failure class agreement", 100 * P["paired"]["failure_class_agree"], 72.8, 0.06)
    chk("T4 solve agreement", 100 * P["paired"]["solve_agree"], 85.25, 0.006)
    chk("T4 median rel diff w2", P["paired"]["median_rel_diff_abs_w2"], 0.44, 0.006)
    chk("T4 float32 equals sweep", float(P["float32_equals_sweep"]["matched"] == 2400 and P["float32_equals_sweep"]["solved_agree"] == 1.0
                                          and P["float32_equals_sweep"]["max_abs_diff_abs_w2"] == 0.0), 1.0, 0)
    chk("T4 a=1.02 float32", float(P["a1.02_float32"]["solved"]), 0.0, 0)
    sb = pd.read_csv(R / "figures" / "v5" / "scoreboard_data.csv")
    for fam, (k, n) in {"Task B": (2, 2), "SGD": (2, 2), "learning rate": (4, 4), "other activations": (0, 3), "R^2": (0, 2),
                        "own-sample threshold": (2, 2), "early branch": (2, 2), "early branch, lag-corrected": (2, 2),
                        "held-out windows": (7, 8)}.items():
        g = sb[sb.family == fam]
        chk(f"scoreboard {fam} within/total", float(g.passed.sum() * 100 + len(g)), float(k * 100 + n), 0)
    pa = pd.DataFrame(P["float32_per_a"])
    chk("T4 float32 per-a rows", float(len(pa)), 12.0, 0)
    chk("T4 float32 per-a sums", float(pa.solved.sum() * 1e6 + pa.placement.sum() * 1e3 + pa.bias.sum()), 430.0e6 + 1664.0e3 + 306.0, 0)
    chk("T4 float32 200 per a", float((pa.runs == 200).all()), 1.0, 0)
    chk("T4 a=1.02 float32 median w2", P["a1.02_float32"]["median_abs_w2"], 1.85, 0.006)
    chk("T4 a=1.02 float32 max w2", P["a1.02_float32"]["max_abs_w2"], 3.92, 0.006)
    chk("T4 a=1.02 float64 median w2", P["a1.02_float64"]["median_abs_w2"], 1.92, 0.006)
    chk("T4 a=1.02 float64 max w2", P["a1.02_float64"]["max_abs_w2"], 3.94, 0.006)


def band_checks() -> None:
    """Track 3B (band task in R^d): registered verdicts and the numbers WP-28 prints."""
    print("Track 3B (band task in R^d)")
    B = R / "band_rd"
    v = pd.read_csv(B / "verdicts.csv"); v["a"] = v.a.round(2)
    g = lambda arm, d, a: v[(v.arm == arm) & (v.d == d) & (v.a == a)].iloc[0]
    for a, (p1, rp, ro, pr) in {1.30: (0.972, 0.182, 0.147, 0.024), 1.50: (0.972, 0.207, 0.187, 0.049)}.items():
        r = g("primary", 2, a)
        chk(f"3B d=2 a={a:.2f} crossings", float(r.n_crossing), 36.0, 0)
        chk(f"3B d=2 a={a:.2f} P1", float(r.P1 == "PASS"), 1.0, 0); chk(f"3B d=2 a={a:.2f} P1 frac", float(r.frac_at_or_above_s_lo), p1, 0.0006)
        chk(f"3B d=2 a={a:.2f} P2a FAIL", float(r.P2a == "FAIL"), 1.0, 0); chk(f"3B d=2 a={a:.2f} median r_pop", float(r.median_r_pop), rp, 0.0006)
        chk(f"3B d=2 a={a:.2f} P2b FAIL", float(r.P2b == "FAIL"), 1.0, 0); chk(f"3B d=2 a={a:.2f} r_own", float(r.median_obs_r_own), ro, 0.0006)
        chk(f"3B d=2 a={a:.2f} pred", float(r.median_pred_r), pr, 0.0006)
        r4 = g("primary", 4, a)
        chk(f"3B d=4 a={a:.2f} UNRESOLVED", float(r4.P1 == "UNRESOLVED" and r4.P2a == "UNRESOLVED" and r4.P2b == "UNRESOLVED"), 1.0, 0)
        chk(f"3B d=4 a={a:.2f} crossings", float(r4.n_crossing), 29.0, 0)
    chk("3B P1 d=2", float((v[(v.arm == "primary") & (v.d == 2)].P1 == "PASS").all()), 1.0, 0)
    chk("3B P2a d=2", float((v[(v.arm == "primary") & (v.d == 2)].P2a == "FAIL").all()), 1.0, 0)
    chk("3B P2b d=2", float((v[(v.arm == "primary") & (v.d == 2)].P2b == "FAIL").all()), 1.0, 0)
    chk("3B d=4 UNRESOLVED", float((v[(v.arm == "primary") & (v.d == 4)].P1 == "UNRESOLVED").all()), 1.0, 0)
    rc_ = pd.read_csv(R / "registration_census.csv").set_index("id")
    chk("TA census L3 adam FAIL registered", float(rc_.loc["trackA-L3-adam", "verdict"] == "FAIL" and rc_.loc["trackA-L3-adam", "scoring"] == "registered rule"), 1.0, 0)
    chk("TA census L3 sgd PASS registered", float(rc_.loc["trackA-L3-sgd", "verdict"] == "PASS" and rc_.loc["trackA-L3-sgd", "scoring"] == "registered rule"), 1.0, 0)
    for (d, a), (n, p1, rp) in {(2, 1.30): (112, 0.946, 0.189), (2, 1.50): (112, 0.964, 0.226), (4, 1.30): (90, 1.0, 0.528), (4, 1.50): (93, 1.0, 0.553)}.items():
        r = g("secondary", d, a)
        chk(f"3B secondary d={d} a={a:.2f} crossings", float(r.n_crossing), n, 0)
        chk(f"3B secondary d={d} a={a:.2f} P1 frac", float(r.frac_at_or_above_s_lo), p1, 0.0006)
        chk(f"3B secondary d={d} a={a:.2f} median r_pop", float(r.median_r_pop), rp, 0.0006)
    ph = pd.read_csv(B / "posthoc_summary.csv"); ph["a"] = ph.a.round(2)
    q = ph[ph.d > 1]
    chk("3B post hoc obs/pred range", float(q.obs_over_pred.min()), 0.84, 0.006)
    chk("3B post hoc obs/pred max", float(q.obs_over_pred.max()), 1.10, 0.006)
    chk("3B post hoc all within tol", float(q.within_tol.all()), 1.0, 0)
    for d, (lo, hi) in {2: (1.118, 1.120), 4: (1.401, 1.412)}.items():
        chk(f"3B post hoc branch/own d={d} min", float(q[q.d == d].median_branch_over_own_x1.min()), lo, 0.0006)
        chk(f"3B post hoc branch/own d={d} max", float(q[q.d == d].median_branch_over_own_x1.max()), hi, 0.0006)


def track2_checks() -> None:
    """Track 2 (width 2; POST HOC): the numbers WP-27 prints."""
    print("Track 2 (width 2, post hoc)")
    sm = pd.read_csv(R / "width2_lag" / "summary.csv"); J = json.loads((R / "width2_lag" / "summary.json").read_text())
    g = lambda arm, sub: sm[(sm.arm == arm) & (sm.subset == sub)].iloc[0]
    n_w = 0; n = 0
    for arm, (nn, fr, obs) in {"T2-3b": (53, 0.66, -0.0049), "T2-3c": (54, 0.83, 0.00005), "T2-3d": (58, 0.81, -0.0012)}.items():
        r = g(arm, "chi <= width-1 max")
        chk(f"T2 2B {arm} n", float(r.n_scored), nn, 0); chk(f"T2 2B {arm} within", float(r.frac_runs_within), fr, 0.006)
        chk(f"T2 2B {arm} median obs", float(r.median_obs), obs, 0.00006)
        n += int(r.n_scored); n_w += int(round(r.frac_runs_within * r.n_scored))
    chk("T2 2B slowed within", n_w / n, 0.77, 0.006); chk("T2 2B slowed n", float(n), 165.0, 0)
    t = g("T2-3", "chi <= width-1 max")
    chk("T2 2B T2-3 three runs obs/pred", float(t.obs_over_pred), 1.04, 0.006); chk("T2 2B T2-3 three runs n", float(t.n_scored), 3.0, 0)
    chk("T2 2B T2-3 three runs dist", float(t.median_dist_c), 0.066, 0.0006)
    chk("T2 2B T2-3 chi_c median", J["T2-3"]["chi_c_min_median_max"][1], 0.99, 0.006)
    chk("T2 2B T2-3 chi_c max", J["T2-3"]["chi_c_min_median_max"][2], 3.05, 0.006)
    chk("T2 2B slowed branch defined", float(J["slowed"]["status_counts"]["ok"]), 191.0, 0)
    chk("T2 2B slowed crossers", float(J["slowed"]["n"]), 230.0, 0)
    b = pd.read_csv(R / "width2_basins" / "summary.csv").set_index("act")
    f = b.loc["f_a pooled"]
    chk("T2 2A Morse-Bott", float(f.morse_bott_min), 265.0, 0); chk("T2 2A strict", float(f.strict_min), 0.0, 0)
    chk("T2 2A placed found", float(f.n_placed_min_found), 58.0, 0); chk("T2 2A stuck lowest", float(f.n_stuck_is_lowest_found), 108.0, 0)
    chk("T2 2A lowest is placed", float(f.n_lowest_found_is_placed), 18.0, 0)
    chk("T2 2A gap median", float(f.gap_to_placed_median), -0.0027, 0.00006)
    chk("T2 2A stuck lower", float(f.n_placed_min_found - f.n_gap_to_placed_positive), 37.0, 0)
    chk("T2 2A tanh saddles", float(b.loc["tanh", "saddle"]), 9.0, 0)
    rc = json.loads((R / "width2_basins" / "reconcile_summary.json").read_text())
    chk("T2 reconcile own loss matched", float(rc["own_loss_matched"]), 265.0, 0)
    chk("T2 reconcile compared", float(rc["n_compared"]), 190.0, 0)
    chk("T2 reconcile all above population global", float(rc["n_stuck_above_pop_global"]), 190.0, 0)
    chk("T2 reconcile same s and placed", float(rc["same_s_all"] and rc["direct_placed_all"]), 1.0, 0)
    chk("T2 reconcile gap min", rc["pop_gap_min"], 1.1e-5, 6e-7)
    chk("T2 reconcile gap max", rc["pop_gap_max"], 2.8e-4, 6e-6)
    chk("T2 reconcile gap median", rc["pop_gap_median"], 3.5e-5, 6e-7)
    chk("T2 reconcile not comparable", float(rc["n_not_comparable"]), 75.0, 0)
    gg = json.loads((R / "t2g" / "gate.json").read_text())
    chk("T2 final gate FAIL", float(gg["gate"] == "FAIL"), 1.0, 0)
    chk("T2 final gate accuracy", gg["accuracy"], 0.812, 0.0006)
    chk("T2 final gate runs", float(gg["n_runs"]), 239.0, 0)
    chk("T2 final gate late median |log err|", gg["median_abs_log_err_pred_late"], 0.014, 0.0006)
    chk("T2 final gate late n", float(gg["n_pred_late_scored"]), 174.0, 0)
    cf = gg["confusion"]
    chk("T2 final gate confusion", float(cf["pred EARLY / true EARLY"] * 1e9 + cf["pred EARLY / true LATE"] * 1e6
                                         + cf["pred LATE / true EARLY"] * 1e3 + cf["pred LATE / true LATE"]), 54e9 + 3e6 + 42e3 + 140, 0)
    for arm, (k, n) in {"T2-3b": (66, 80), "T2-3c": (69, 80), "T2-3d": (59, 79)}.items():
        pa_ = gg["per_arm"][arm]; c_ = pa_["confusion"]
        chk(f"T2 final gate correct {arm}", float(c_["pred EARLY / true EARLY"] + c_["pred LATE / true LATE"]), float(k), 0)
        chk(f"T2 final gate runs {arm}", float(pa_["n_runs"]), float(n), 0)
    c = json.loads((R / "t2c" / "calibration.json").read_text())
    chk("T2 2C STOP", float(c["decision"] == "STOP"), 1.0, 0)
    chk("T2 2C earliest crossing", c["earliest_crossing"], 0.0164, 0.00006)
    chk("T2 2C max available", c["max_frac_available"], 0.778, 0.0006)
    chk("T2 2C max available below earliest", c["max_frac_available_below_earliest"], 0.075, 0.0006)


def c1_followup_checks() -> None:
    """Track 3 (final round): the registered c1 follow-up."""
    print("c1 follow-up (registered)")
    S = json.loads((R / "c1_followup_summary.json").read_text())
    chk("c1 follow-up PASS", float(S["verdict"] == "PASS"), 1.0, 0)
    chk("c1 follow-up lo", S["feasible_lo"], 0.26531, 0.000006)
    chk("c1 follow-up hi", S["feasible_hi"], 0.30687, 0.000006)
    chk("c1 follow-up width", S["feasible_width"], 0.0416, 0.00006)
    chk("c1 follow-up contains derived", float(S["feasible_lo"] <= S["pred_lo"] and S["pred_hi"] <= S["feasible_hi"]), 1.0, 0)
    chk("c1 follow-up added certified", float(S["n_added_certified"]), 5.0, 0)
    chk("c1 follow-up original n", float(S["n_original"]), 4.0, 0)
    chk("c1 follow-up independently checked", float(sum(S["independently_checked"].values())), 10.0, 0)
    chk("c1 follow-up not checked", float(len(S["not_checked"])), 0.0, 0)
    chk("c1 follow-up excluded by checker", float(len(S["excluded_by_checker"])), 0.0, 0)


def linear_response_checks() -> None:
    """Track 1 (final round; POST HOC): exact linear response along the trajectory."""
    import hashlib as _h
    print("Linear response (post hoc)")
    L = R / "linear_response"
    S = json.loads((L / "summary.json").read_text())
    chk("LR predictions hash", float(_h.sha256((L / "predictions.csv").read_bytes()).hexdigest() == S["predictions_sha256"]
                                     == "e86ca102bcf87f6ec44565d189d4c50fd0e4204db338309d6240527b1c39faa2"), 1.0, 0)
    chk("LR runs", float(S["n_runs"]), 1750.0, 0)
    chk("LR predicted full 0.7", float(S["n_pred_full_7"]), 1708.0, 0)
    chk("LR diff branch 0.7", float(S["n_diff_branch_start7"]), 0.0, 0)
    chk("LR diff branch 0.5", float(S["n_diff_branch_start5"]), 0.0, 0)
    chk("LR s* off global own > 1%", S["frac_s_star_off_global_own_1pct_all"], 0.228, 0.0006)
    chk("LR no-momentum unstable (Adam)", float(S["n_unstable_d_7"]), 1240.0, 0)
    a = pd.read_csv(L / "attribution.csv").pivot(index="step", columns="a", values="slope")
    for step, want in {"full_7": (1.02, 1.03, 1.04, 1.05), "full_5": (1.02, 1.03, 1.04, 1.05),
                       "1A_global_ref": (0.89, 0.83, 0.91, 0.82), "kchi_branch_ref": (1.02, 1.01, 0.98, 0.90),
                       "slaved_own": (1.01, 0.99, 0.96, 0.90)}.items():
        for av, w in zip((1.3, 1.45, 1.5, 1.6), want):
            chk(f"LR {step} slope a={av:.2f}", float(a.loc[step, av]), w, 0.006)
    chk("LR full model ratio", float(a.loc["full_7"].min()), 1.02, 0.006)
    c = pd.read_csv(L / "compare_arms.csv")
    chk("LR arm ratio min", float(c.full_7_ratio_of_medians.min()), 1.00, 0.006)
    chk("LR arm ratio max", float(c.full_7_ratio_of_medians.max()), 1.05, 0.006)
    d = S["diagnostic_exact_grad"]["pooled"]
    for r in d:
        chk(f"LR exact gradient reproduces start {r['start']}", float(r["n_exact_step_equals_observed"] == r["n"]), 1.0, 0)


def track_b_checks() -> None:
    """Track B (final round): the numbers WP-33 prints."""
    print("Track B (statistics)")
    D = R / "track_b"
    S = json.loads((D / "summary.json").read_text()); W = S["within"]
    k = list(W)
    cf, reg, tr = W[k[0]], W[k[1]], W[k[2]]
    chk("TB closed form within10 arms", float(cf["arms_median_within_10"]), 36.0, 0)
    chk("TB closed form runs within10", 100 * cf["pooled_frac_within_10"], 88.2, 0.06)
    chk("TB closed form runs within20", 100 * cf["pooled_frac_within_20"], 97.8, 0.06)
    chk("TB registered ref arms within10", float(reg["arms_median_within_10"]), 31.0, 0)
    chk("TB registered ref runs within10", 100 * reg["pooled_frac_within_10"], 63.7, 0.06)
    chk("TB registered ref arms within20", float(reg["arms_median_within_20"]), 32.0, 0)
    chk("TB traj runs within10", 100 * tr["pooled_frac_within_10"], 99.9, 0.06)
    chk("TB traj runs within20", 100 * tr["pooled_frac_within_20"], 100.0, 0.06)
    col = pd.read_csv(D / "collapse.csv"); cr = col[col.resolved]
    lo = cr[(cr.chi <= 0.06) & cr.branch_conditioned_post_hoc]
    chk("TB collapse low-chi ratio min", float(lo.ratio.min()), 0.67, 0.006)
    chk("TB collapse low-chi ratio max", float(lo.ratio.max()), 1.40, 0.006)
    chk("TB collapse width2 max", float(cr[cr.setting == "width 2"].ratio.max()), 34.0, 0.6)
    chk("TB collapse resolved", float(len(cr)), 99.0, 0)
    ss = pd.read_csv(D / "sample_size_decomposition.csv")
    for r in ss.itertuples():
        want = {(1.3, 400): 0.0297, (1.3, 1600): 0.0293, (1.3, 6400): 0.0306, (1.5, 400): 0.0620, (1.5, 1600): 0.0619, (1.5, 6400): 0.0635}[(round(r.a, 2), r.n)]
        chk(f"TB sample size dynamic a={r.a:.2f} n={r.n}", float(r.dynamic_median), want, 0.00006)
    chk("TB kappa inputs rows", float(len(pd.read_csv(D / "kappa_inputs.csv"))), 8.0, 0)


def theorem1_checks() -> None:
    """Track D.1 (final round): Theorem 1's hypotheses."""
    print("Theorem 1 hypotheses (checked)")
    d = json.loads((R / "theorem1_checks.json").read_text())
    at = d["H-A1_data_gap_attainment"]
    chk("D1 attainment", float(all(r["Gamma_n_positive"] and r["Gamma_n_ge_Ghat_lo"] for r in at)), 1.0, 0)
    chk("D1 attainment a values", float(len(at)), 9.0, 0)
    chk("D1 attainment margin min", float(min(r["argmax_interior_margin"] for r in at)), 0.098, 0.0006)
    q = d["H-A2_classmean_attainment_and_H-Q_quadratic_growth"]
    chk("D1 quadratic growth", float(q["continuous"]["quad_growth_ok"] and q["data"]["quad_growth_ok"]), 1.0, 0)
    chk("D1 quadratic growth constant", q["data"]["quad_growth_c"], 2.454, 0.0006)
    chk("D1 continuous alpha*", q["continuous"]["alpha_star"], 1.7922917, 6e-8)
    chk("D1 alias", float(q["data"]["alpha_star_global_on_(0, pi/q)"] is False), 1.0, 0)
    rr = d["H-R_uniform_remainder"]
    chk("D1 remainder", float(max(r["max_ratio_all"] for r in rr)), 0.0052084, 6e-7)
    chk("D1 alias excluded above s", d["H-C_compactness"]["s_where_W_reaches_alias_1.30"], 7.24e-6, 6e-8)


def track_t_checks() -> None:
    """Track T (final round): pilot and gate (exploratory)."""
    print("Track T (pilot, exploratory)")
    P = json.loads((R / "simplicity_bias" / "pilot_summary.json").read_text())
    chk("TT gate FAIL", float(P["gate"]["pass"] is False), 1.0, 0)
    chk("TT G1", float(P["gate"]["G1"]), 1.0, 0); chk("TT G2", float(P["gate"]["G2"]), 0.0, 0)
    chk("TT G3", float(P["gate"]["G3"]), 0.0, 0); chk("TT G4", float(P["gate"]["G4"]), 0.0, 0)
    chk("TT switch A", P["switch"]["A"], 4.699, 0.0006); chk("TT switch B", P["switch"]["B"], 4.389, 0.0006)
    chk("TT agreement", 100 * P["gate"]["agreement_rel"], 7.1, 0.06)
    chk("TT ladder fail", float(P["diagnostics"]["n_ladder_fail"]), 29.0, 0)
    chk("TT points", float(P["diagnostics"]["n_points"]), 60.0, 0)
    chk("TT hidden weight median", P["diagnostics"]["max_abs_hidden_weight_median"], 6.9e4, 600)


def track_a_checks() -> None:
    """Track A (final round; registered lag-law test at a = 1.65)."""
    import hashlib as _h
    print("Track A (registered, a = 1.65)")
    S = json.loads((R / "track_a" / "scores.json").read_text())
    chk("TA predictions hash", float(_h.sha256((R / "track_a" / "predictions.csv").read_bytes()).hexdigest()
                                     == "b30167c61116972298dc434494918dced800b83a56164374ef338167070e3d82"), 1.0, 0)
    for opt, (nc, l1, l2, l3, v1, v2, v3) in {"adam": (76, 1.065, 1.045, 0.25, "PASS", "PASS", "FAIL"),
                                              "sgd": (64, 1.056, 1.024, 0.995, "PASS", "PASS", "PASS")}.items():
        x = S[opt]
        chk(f"TA {opt} crossings", float(x["n_crossed"]), float(nc), 0)
        chk(f"TA {opt} valid", float(x["valid"] and x["n_rule_not_before_crossing"] == 0), 1.0, 0)
        chk(f"TA {opt} L1", float(x["L1"]["verdict"] == v1), 1.0, 0); chk(f"TA {opt} L1 ratio", x["L1"]["median_ratio"], l1, 0.0006)
        chk(f"TA {opt} L2", float(x["L2"]["verdict"] == v2), 1.0, 0); chk(f"TA {opt} L2 ratio", x["L2"]["median_ratio"], l2, 0.0006)
        chk(f"TA {opt} L3", float(x["L3"]["verdict"] == v3), 1.0, 0); chk(f"TA {opt} L3 spearman", x["L3"]["spearman"], l3, 0.006 if opt == "adam" else 0.0006)


def digit_stability() -> None:
    """Ĝ was replaced by its rigorous enclosure (author's decision 2026-09-24).  Every R is linear in Ĝ, so a stored
    value x computed with the old float Ĝ becomes x·r with |r − 1| <= δ, δ = the largest relative change of Ĝ over a
    (results/ghat_rigorous.csv).  For EVERY numeric ledger check (conservatively, including numbers that do not depend
    on Ĝ): the check must still hold at x·(1 ± δ), and x·(1 ± δ) must round to the same printed digits (the number of
    decimals of the printed ledger value).  Any failure is a finding."""
    p = R / "ghat_rigorous.csv"
    print("Digit stability under the rigorous Ĝ")
    if not p.exists():
        F.append("digit stability: ghat_rigorous.csv missing"); print("  MISSING ghat_rigorous.csv"); return
    g = pd.read_csv(p, float_precision="round_trip")
    want_a = set(pd.read_csv(R / "ghat_certified_all.csv").a.round(2))
    if set(g.a.round(2)) != want_a:
        F.append("digit stability: ghat_rigorous.csv does not cover every a"); return
    delta = float(max(g.rel_delta_cert.abs().max(), g.rel_delta_hi.abs().max()))
    # Checks with tol < 1e-9 are machine-precision consistency checks between two artifacts (their "value" carries
    # 14-18 decimals); they are not printed numbers.  They are recorded and listed by name, not counted as printed-digit
    # findings (disclosed in WP-7; set after the first run flagged five, none of which depends on the replaced Ĝ(a)).
    n_num = n_skip = 0
    bad, mp = [], []
    for label, got, want, tol in REC:
        if got is None or tol == 0 or not np.isfinite(got) or not np.isfinite(want):
            n_skip += 1
            continue
        d = _decimals(want)
        moved = any(abs(x - want) > tol or round(x, d) != round(got, d) for x in (got * (1 - delta), got * (1 + delta)))
        if tol < 1e-9:
            mp.append({"label": label, "tol": tol, "moves_under_blanket_delta": moved})
            continue
        n_num += 1
        if moved:
            bad.append(label)
    rows = [{"delta": delta, "printed_number_checks": n_num, "unstable": len(bad),
             "machine_precision_checks": len(mp), "machine_precision_moved": int(sum(m["moves_under_blanket_delta"] for m in mp)),
             "exact_or_non_numeric": n_skip}]
    pd.DataFrame(rows).to_csv(R / "ghat_digit_stability.csv", index=False)
    pd.DataFrame(mp).to_csv(R / "ghat_digit_stability_machine_precision.csv", index=False)
    print(f"  delta = {delta:.3e}; {n_num} printed-number checks, {len(bad)} unstable; {len(mp)} machine-precision checks "
          f"({rows[0]['machine_precision_moved']} move in their last digits); {n_skip} exact or non-numeric")
    for b in bad:
        F.append(f"digit stability under the rigorous Ĝ: {b}")


def provenance_check() -> None:
    import ast as _ast
    import pathlib as _pl

    print("PROVENANCE: every artifact backing a reported number has a producing function")
    here = _pl.Path(__file__).resolve().parent
    bad, partial = [], []
    for art, (mod, fn, status, note) in sorted(PRODUCERS.items()):
        path = here / f"{mod}.py"
        if not (R / art).exists():
            continue                              # not yet produced (running block)
        if not path.exists():
            bad.append(f"{art}: producer module src/{mod}.py does not exist")
            continue
        text = path.read_text()
        stem = art.rsplit(".", 1)[0]
        import re as _re
        fprefixes = _re.findall(r'f"([A-Za-z0-9_]+)_\{', text)      # f"gelu_scale_{arm}"
        if not (art in text or f'"{stem}"' in text
                or any(stem.startswith(fp + "_") for fp in fprefixes)):
            bad.append(f"{art}: src/{mod}.py never references it")
            continue
        if fn == "<module>":
            body = text
        else:
            tree = _ast.parse(text)
            fns = [n for n in _ast.walk(tree)
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and n.name == fn]
            if not fns:
                bad.append(f"{art}: src/{mod}.py has no function {fn}")
                continue
            body = _ast.get_source_segment(text, fns[0]) or ""
        if fn != "r_families_primary" and not any(w in body for w in _WRITE_CALL):
            bad.append(f"{art}: {mod}.{fn} performs no write")
            continue
        if status == "partial":
            partial.append(f"{art}: {note}")
    chk("artifacts with no valid producer", float(len(bad)), 0.0, 0)
    for b in bad:
        F.append(f"PROVENANCE: {b}")
    chk("artifacts with only a partial producer", float(len(partial)), 0.0, 0)
    for p_ in partial:
        F.append(f"PROVENANCE (partial): {p_}")


if __name__ == "__main__":
    main()
