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
    chk("R_glob^inf", float(d["R_glob"]), 0.19991, 1e-4)
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
    chk("R_glob log-log slope", float(np.polyfit(np.log(e), np.log(dev), 1)[0]), 0.947, 0.01)
    c1 = float(np.linalg.lstsq(np.vstack([e]).T, dev, rcond=None)[0][0])
    chk("c1", c1, 0.23099, 1e-4)
    chk("c1 fit max residual",
        float(np.abs(b["R_glob"].values - float(d["R_glob"]) * (1 + c1 * e)).max()),
        0.00133, 1e-4)
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

    print(f"\n{len(F)} finding(s)")
    (R / "ledger_verification.txt").write_text("\n".join(F) if F else "no findings\n")


if __name__ == "__main__":
    main()
