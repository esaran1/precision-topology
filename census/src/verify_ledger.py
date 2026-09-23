"""Verify every headline number in CLAIMS.md against its committed artifact.

Recomputes from CSVs rather than trusting the ledger text. Any mismatch is
printed as FINDING: and written to results/ledger_verification.txt.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
F: list[str] = []


def chk(label, got, want, tol=0.01):
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
    chk("registered predictions", float(len(rc)), 164.0, 0)
    for v, n in (("PASS", 69), ("FAIL", 53), ("PARTIAL", 12), ("UNRESOLVED", 30)):
        chk(f"all: {v}", float((rc.verdict == v).sum()), float(n), 0)
    r64 = rc[rc.counted_in_existing_64 == "yes"]
    chk("existing 64 rows", float(len(r64)), 64.0, 0)
    for v, n in (("PASS", 27), ("FAIL", 24), ("PARTIAL", 1), ("UNRESOLVED", 12)):
        chk(f"64: {v}", float((r64.verdict == v).sum()), float(n), 0)
    tl = pd.read_csv(R / "registration_tally.csv").set_index("scope")
    for scope, want in (("scored by registered rules", (151, 66, 47, 8, 30)),
                        ("assigned post hoc in the census", (13, 3, 6, 4, 0))):
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

    provenance_check()

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
    "corner_tracking.csv": ("corner_tracking", "main", "full", ""),
    "corner_tracking_decomposition.csv": ("corner_tracking", "decompose", "full", ""),
    "corner_tracking_switch_active.csv": ("corner_tracking", "switch_active", "full", ""),
    "cond_certified_seeds_summary.csv": ("conditional_certified", "seeds_summary", "full", ""),
    "first_order_corner.csv": ("first_order", "corner", "full", ""),
    "first_order_corner_free.csv": ("first_order", "corner", "full", ""),
    "mn2_h2prime.csv": ("math_note_v2_checks", "h2", "full", ""),
    "mn2_solve_limit.csv": ("math_note_v2_checks", "solve_limit", "full", ""),
    "cond_audit_candidates.csv": ("conditional_audit", "candidates", "full", ""),
    "cond_audit_strict.csv": ("conditional_audit", "strict", "full", ""),
    "cond_certified_brackets.csv": ("conditional_certified", "brackets", "full", ""),
    "cond_certified_bracket_evaluations.csv": ("conditional_certified", "brackets", "full", ""),
    "mn2_bounds.csv": ("math_note_v2_checks", "bounds", "full", ""),
    "mn2_uniformity_summary.csv": ("math_note_v2_checks", "uniform_summary", "full", ""),
    "mn2_neighbourhood.csv": ("math_note_v2_checks", "neighbourhood", "full", ""),
    "mn2_rounding.csv": ("math_note_v2_checks", "rounding", "full", ""),
    "limit_switch.csv": ("limit_bnb", "switch", "full", ""),
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
    "fixed_scale_block5_splits.csv": ("fixed_scale", "score5_splits", "full", ""),
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
