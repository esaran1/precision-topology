"""Phase 2C (DESCRIPTIVE; no registration, no criteria): the distribution of κχ at the crossing in standard training.

For standard initialisation and the standard learning rate of every setting run so far (width-1 sine, Adam and SGD;
width-1 GELU; the band task in R^d; width 2; simplicity bias), how often ordinary training lies inside the regime where
the linear lag law r = κχ applies (math note §13; validity boundary |κχ| ≤ 0.1 from Test 2B C1).

Every κχ, κ and χ is READ from a committed producer output and cited; nothing is retrained. The only quantity computed
here is the per-run fold rate ε_F of the simplicity-bias runs at M's fold (`sb_eps_f`), from the committed v3 run
records and the committed M grid with sb_fold's own construction; the population value it is checked against is
`sb_fold/fold_constant_adam.json`.

    python -m src.phase2c_map eps     # results/phase2c/sb_eps_F.json (one process, ~1 min)
    python -m src.phase2c_map         # results/phase2c/kappa_chi_map.json and results/phase2c_kappa_chi_map.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "phase2c"
MD = RESULTS / "phase2c_kappa_chi_map.md"
BOUNDS = (0.1, 0.02)          # Test 2B C1 (|κχ| ≤ 0.1 within ±25%), and the slowest C1 cell
QUANTS = (0.10, 0.50, 0.90)
LABEL = "DESCRIPTIVE (Phase 2C; no registration, no criteria, no verdict)"


# ------------------------------------------------------------------------------------------ summary functions
def summarize(values) -> dict:
    """Distribution of the finite values: n, min, q10, median, q90, max (signed), the same of |·|, the number of
    negative values and the fractions with |value| ≤ each bound (over the finite values)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    out = {"n": int(v.size)}
    if v.size == 0:
        for k in ("min", "q10", "median", "q90", "max"):
            out[k] = None
            out[f"abs_{k}"] = None
        out["n_negative"] = 0
        for b in BOUNDS:
            out[f"frac_abs_le_{b}"] = None
            out[f"n_abs_le_{b}"] = 0
        return out
    a = np.abs(v)
    for tag, x in (("", v), ("abs_", a)):
        q = np.quantile(x, QUANTS)
        out.update({f"{tag}min": float(x.min()), f"{tag}q10": float(q[0]), f"{tag}median": float(q[1]),
                    f"{tag}q90": float(q[2]), f"{tag}max": float(x.max())})
    out["n_negative"] = int((v < 0).sum())
    for b in BOUNDS:
        out[f"n_abs_le_{b}"] = int((a <= b).sum())
        out[f"frac_abs_le_{b}"] = float((a <= b).mean())
    return out


def signed_differs(s: dict) -> bool:
    """True when the signed and |·| summaries differ (some value is negative)."""
    return s["n"] > 0 and s["n_negative"] > 0


def cell(setting, name, values, *, n_runs, n_cross, source, column, optimiser, lr, kappa_def, chi_def, note="",
         quantity="kappa_chi") -> dict:
    return {"setting": setting, "cell": name, "quantity": quantity, "optimiser": optimiser, "lr": lr,
            "n_runs": int(n_runs), "n_cross": int(n_cross), "source": source, "column": column,
            "kappa_def": kappa_def, "chi_def": chi_def, "note": note, "summary": summarize(values)}


# ------------------------------------------------------------------------------------------ width-1 sine
CHI_S13 = "χ = (ṡ/s*)/(ηλ_min(P^{1/2}HP^{1/2})) (§13.1)"


def width1_cells() -> list:
    S = "width-1 sine"
    out = []
    pr = pd.read_csv(RESULTS / "lag_law" / "predictions.csv")
    pr["arm"] = pr.arm.astype(str)
    lt = pd.read_csv(RESULTS / "lag_test_runs.csv")
    lt = lt[lt.factor == 1.0]
    k_lag = ("population κ_k at the certified switch (lag_law/kappa.csv; Adam P = per-coordinate median of the crossing "
             "P over existing runs; winding k measured at the crossing)")
    chi_rt = ("run's own timescale ratio at the crossing (residual_timescale): d log s/dt over the last ≤ 100 steps "
              "/ (lr·λ_min(D^{-1/2}HD^{-1/2})), D = √v̂+ε the run's own Adam state; = χ/(1+r), i.e. ṡ divided by s_c "
              "not s* (§13.3(i))")
    for a in (1.30, 1.50):
        g = pr[(pr.set == "lag1") & (pr.arm == "1.0") & (pr.a.round(2) == a)]
        n = lt[lt.a.round(2) == a]
        out.append(cell(S, f"Adam a={a:.2f} (n=6,400, seeds 300000-049)", g.pred_r, n_runs=len(n),
                        n_cross=int(n.cross_step.notna().sum()), source="results/lag_law/predictions.csv "
                        "(set lag1, arm 1.0 = φ = 1, the unmodified reference path; lag2-prim/1.0, lag2-tstar/1.0 and "
                        "4b/control are bit-identical duplicates and are not counted again)", column="pred_r",
                        optimiser="Adam", lr=0.01, kappa_def=k_lag, chi_def=chi_rt))
    tb = pd.read_csv(RESULTS / "ts_test" / "runs.csv")
    for a in (1.45, 1.60):
        g = pr[(pr.set == "TaskB") & (pr.a.round(2) == a)]
        n = tb[tb.a.round(2) == a]
        out.append(cell(S, f"Adam a={a:.2f} (Task B, n=200, seeds 830000-079)", g.pred_r, n_runs=len(n),
                        n_cross=int(n.crossed.sum()), source="results/lag_law/predictions.csv (set TaskB)",
                        column="pred_r", optimiser="Adam", lr=0.01, kappa_def=k_lag, chi_def=chi_rt))
    ta = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
    dg = pd.read_csv(RESULTS / "track_a" / "diag_p_at_crossing.csv")
    ad = ta[ta.opt == "adam"]
    out.append(cell(S, "Adam a=1.65 (Track A, seeds 1650000-079), P at the crossing", dg[dg.crossed].p_cross_r_cf,
                    n_runs=len(ad), n_cross=int(ad.crossed.sum()),
                    source="results/track_a/diag_p_at_crossing.csv (POST HOC diagnostic, track_a_diag)",
                    column="p_cross_r_cf", optimiser="Adam", lr=0.01,
                    kappa_def="κ_k from the a = 1.65 landscape (H, θ*′, ∇G at s*_pop) with the run's P at its crossing",
                    chi_def="(ṡ/s*_run)/(lr·λ_min) at t_sw (ṡ over the last ≤ 100 steps), P at the crossing; = §13.1"))
    ac = ad[ad.crossed.astype(bool)]
    out.append(cell(S, "Adam a=1.65 (Track A), registered P at the rule point [sensitivity; not pooled]", ac.r_cf,
                    n_runs=len(ad), n_cross=int(ad.crossed.sum()), source="results/track_a/observed_runs.csv",
                    column="r_cf", optimiser="Adam", lr=0.01,
                    kappa_def="as above, P frozen at the rule point (last upward passage of 0.5·s*)",
                    chi_def="as above, P at the rule point", note="sensitivity to the P rule"))
    t2 = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
    out.append(cell(S, "Adam a=1.85 (Test 2A, seeds 1850000-079), P at t_sw", t2[t2.crossed.astype(bool)].r_cf,
                    n_runs=len(t2), n_cross=int(t2.crossed.sum()), source="results/track2a/observed_runs.csv",
                    column="r_cf", optimiser="Adam", lr=0.01,
                    kappa_def="κ_k from the a = 1.85 landscape with the run's P at t_sw (first s ≥ s*_run)",
                    chi_def="(ṡ/s*_run)/(lr·λ_min) at t_sw, P at t_sw; = §13.1",
                    note="the minimum is seed 1850004 (χ = 1.7e-5 at t_sw; not in 2A's scored set)"))
    so = pd.read_csv(RESULTS / "sgd_own_runs.csv")
    for a in (1.30, 1.50):
        g = pr[(pr.set == "SGD") & (pr.a.round(2) == a)]
        n = so[so.a.round(2) == a]
        out.append(cell(S, f"SGD a={a:.2f} (n=200, seeds 0-39)", g.pred_r, n_runs=len(n), n_cross=int(n.crossed.sum()),
                        source="results/lag_law/predictions.csv (set SGD)", column="pred_r", optimiser="SGD", lr=0.3,
                        kappa_def="population κ_k (lag_law/kappa.csv, P = I)",
                        chi_def="run's own ratio at the crossing (sgd_own_ratios.csv), P = I; ṡ divided by s_c"))
    sg = ta[ta.opt == "sgd"]
    out.append(cell(S, "SGD a=1.65 (Track A, seeds 1650000-079)", sg[sg.crossed.astype(bool)].r_cf, n_runs=len(sg),
                    n_cross=int(sg.crossed.sum()), source="results/track_a/observed_runs.csv", column="r_cf",
                    optimiser="SGD", lr=0.3, kappa_def="κ_k from the a = 1.65 landscape, P = I",
                    chi_def="(ṡ/s*_run)/(lr·λ_min(H)) at t_sw; = §13.1"))
    w1 = [json.loads(l) for l in (RESULTS / "phase1c" / "observed_W1.jsonl").read_text().splitlines() if l.strip()]
    v = [r["r_cf"] if r.get("crossed") and r.get("r_cf") is not None else np.nan for r in w1]
    out.append(cell(S, "SGD a=1.58 (Phase 1C W1, seeds 9350000-099)", v, n_runs=len(w1),
                    n_cross=sum(bool(r.get("crossed")) for r in w1), source="results/phase1c/observed_W1.jsonl",
                    column="r_cf", optimiser="SGD", lr=0.3,
                    kappa_def="κ_k from the a = 1.58 landscape (Track A construction), P = I",
                    chi_def="(ṡ/s*_run)/(lr·λ_min(H)) at the actual t_sw (Track A's closed form); = §13.1"))
    return out


# ------------------------------------------------------------------------------------------ GELU
def gelu_runs():
    t = pd.concat([pd.read_csv(RESULTS / "act_general" / "train_gelu.csv"),
                   pd.read_csv(RESULTS / "act_general" / "train_ext_gelu.csv")], ignore_index=True)
    k = json.loads((RESULTS / "act_general" / "kappa_gelu_frozen.json").read_text())
    c = t[t.crossed.astype(bool) & ~t.placed_at_init.astype(bool)].copy()
    c["kchi"] = k["kappa_adam"] * c.chi
    c["early"] = c.s_cross < 0.5 * k["s_glob"]
    return t, c, k


def gelu_cells() -> list:
    t, c, k = gelu_runs()
    S = "width-1 GELU"
    kd = (f"one frozen population κ_Adam = {k['kappa_adam']:.4f} (act_general/kappa_gelu_frozen.json; P = median "
          "crossing P of 5 calibration runs), times the run's χ; the product is act_posthoc2's `pred` for the 117 runs "
          "with a branch switch (identical to 1e-16)")
    cd = ("run's χ at the crossing (act_general._at_crossing): d log s/dt over the last ≤ 100 steps / (lr·λ_min) with "
          "the run's own Adam v̂ and the own-sample Hessian at the branch point; ṡ divided by s_c")
    src = "results/act_general/train_gelu.csv + train_ext_gelu.csv (column chi) × kappa_gelu_frozen.json kappa_adam"
    base = dict(n_runs=len(t), n_cross=len(c), source=src, column="kappa_adam·chi", optimiser="Adam", lr=0.01,
                kappa_def=kd, chi_def=cd)
    return [cell(S, "Adam (Track 3A, seeds 850000-199), crossings at s ≥ 0.5·s_glob", c[~c.early].kchi, **base,
                 note=f"{int(t.placed_at_init.sum())} runs placed at init excluded; s_glob = {k['s_glob']:.4f}"),
            cell(S, "Adam, EARLY crossings s < 0.5·s_glob [reported separately; not pooled]", c[c.early].kchi, **base,
                 note="tail-placed one-sided-ramp branch with no switch (§13.7); s falling or relax ≈ 0, so κχ has no "
                      "lag meaning here")]


# ------------------------------------------------------------------------------------------ band task in R^d
def band_cells() -> list:
    S = "band task in R^d"
    out = []
    b = pd.read_csv(RESULTS / "band_rd" / "scored_runs.csv")
    kd = ("population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing "
          "(band_rd.lag_quantities)")
    cd = ("at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam "
          "v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c")
    for (arm, d, a), g in b.groupby(["arm", "d", "a"]):
        out.append(cell(S, f"Adam 3B {arm} d={d} a={a:.2f}", g[g.crossed.astype(bool)].pred_r, n_runs=len(g),
                        n_cross=int(g.crossed.sum()), source="results/band_rd/scored_runs.csv", column="pred_r",
                        optimiser="Adam", lr=0.01, kappa_def=kd, chi_def=cd,
                        note="d = 1 is the width-1 control of the band task" if d == 1 else ""))
    t = pd.read_csv(RESULTS / "track2c" / "observed_runs.csv")
    kd2 = "as 3B (track2c/kappa_k.csv), the run's winding k at the rule point"
    cd2 = ("at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with "
           "the run's Adam P at t_sw (track2c.predict_run)")
    for (d, a), g in t.groupby(["d", "a"]):
        c = g[g.crossed.astype(bool)]
        post = c[c.t_sw >= c.step_obs]
        out.append(cell(S, f"Adam 2C d={d} a={a:.2f} (seeds 2030000-059), all crossers", c.r_pred, n_runs=len(g),
                        n_cross=len(c), source="results/track2c/observed_runs.csv", column="r_pred", optimiser="Adam",
                        lr=0.01, kappa_def=kd2, chi_def=cd2,
                        note=f"{len(post)} crossers have t_sw ≥ the crossing step (χ read after the crossing)"))
        pre = c[c.t_sw < c.step_obs]
        out.append(cell(S, f"Adam 2C d={d} a={a:.2f}, t_sw before the crossing [sensitivity; not pooled]", pre.r_pred,
                        n_runs=len(g), n_cross=len(c), source="results/track2c/observed_runs.csv", column="r_pred",
                        optimiser="Adam", lr=0.01, kappa_def=kd2, chi_def=cd2, note="= the registered scored set"))
    return out


# ------------------------------------------------------------------------------------------ width 2
def width2_cells() -> list:
    S = "width 2"
    d = pd.read_csv(RESULTS / "width2_lag" / "parts_T2-3.csv")
    tr = pd.read_csv(RESULTS / "asym_parts" / "train.csv")
    ok = d[d.status == "ok"]
    st = ", ".join(f"{k} {v}" for k, v in sorted(d.status.value_counts().items()))
    return [cell(S, "Adam T2-3 (a=1.30, Δ=0.4, matched init, seeds 600000-079)", ok.r_pred, n_runs=len(tr),
                 n_cross=int(tr.crossed.sum()), source="results/width2_lag/parts_T2-3.csv (width2_lag.analyse)", column="r_pred",
                 optimiser="Adam", lr=0.01,
                 kappa_def="κ₂ = λ_min·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′] at the run's own-path switch, Adam P at the crossing",
                 chi_def="(ṡ/s*)/(lr·λ_min) with ṡ at the crossing (≤ 100-step window), s* the own-path switch",
                 note=f"{int(tr.placed_at_init.sum())} placed at init (asym_parts/train.csv); analysed crossings: {st}; κχ only for status ok; no run within 0.05 of its branch at the crossing "
                      f"(min dist_c {d.dist_c.min():.3f})")]


# ------------------------------------------------------------------------------------------ simplicity bias
def sb_cells() -> list:
    S = "simplicity bias"
    r = pd.read_csv(RESULTS / "simplicity_bias_v3" / "runs_summary.csv")
    p = pd.read_csv(RESULTS / "sb_fold" / "per_run.csv")
    m = r.merge(p[["seed", "group"]], on="seed")
    cd = ("v3's χ at the ρ₂ upward passage (simplicity_bias_v3.chi_at): d log s/dt over ≤ 100 steps (÷ s_c, not s*) "
          "/ (lr·λ_min(P^{1/2}HP^{1/2})), H = the v2 13×13 weight-decayed Hessian at s_switch on the S side "
          "(not at the run's branch), P = the run's Adam block medians at the crossing (no v block)")
    out = []
    for grp, lab in (("late", "late crossers (s_c 8.9-10.7, above s_switch = 3.59)"),
                     ("early", "early crossers (s_c 0.20-0.95, s falling: χ < 0) [not pooled]")):
        g = m[m.group == grp]
        out.append(cell(S, f"Adam χ only, {lab}", g.chi, n_runs=len(m), n_cross=int(m.s_cross.notna().sum()),
                        source="results/simplicity_bias_v3/runs_summary.csv (chi); groups from sb_fold/per_run.csv",
                        column="chi", optimiser="Adam", lr=0.01, quantity="chi",
                        kappa_def="UNDEFINED: M has no switch of its own (ρ₂ ≤ 0.279, G₊ < 0); no κ for any branch",
                        chi_def=cd))
    e = OUT / "sb_eps_F.json"
    if e.exists():
        E = json.loads(e.read_text())
        v = [x["eps_F"] for x in E["runs"]]
        out.append(cell(S, "Adam ε_F at M's fold, per run (own P at the first s ≥ s_F)", v, n_runs=len(m),
                        n_cross=int(m.s_cross.notna().sum()), source="results/phase2c/sb_eps_F.json (phase2c_map.sb_eps_f)",
                        column="eps_F", optimiser="Adam", lr=0.01, quantity="eps_F",
                        kappa_def="not applicable (fold)",
                        chi_def="ε_F = (ṡ_F/s_F)/(lr·Λ_F), Λ_F = √(|m_c|·s_F), m_c from λ_min(P^{1/2}HP^{1/2})² ≈ "
                                "4|m_c|(s_F − s) on M's grid within 0.05 of s_F (sb_fold.run_precond_constant), with the "
                                "run's own block-median P instead of the median over runs",
                        note=f"population ε_F (median P, median ṡ) = {E['population']['eps_F_committed']:.3f}; no run is "
                             "on M at the fold"))
    return out


# ------------------------------------------------------------------------------------------ per-run ε_F at M's fold
def _sb_hessian_fn(nA, X, y, lam):
    """sb_fold.run_precond_constant's H_train (nested there), unchanged: Hessian of the v3 training loss in the
    training coordinates (W, c, v[:-1], b) at the M-grid point u, scale s."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(y, dtype=torch.float64)

    def H_train(u, s):
        n = nA
        W = u[:2 * n]; c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]; b = u[-1]
        v = s * eta ** 2 / (eta ** 2).sum()
        q0 = np.r_[W, c, v[:-1], b]

        def L(q):
            Wt = q[:2 * n].reshape(n, 2); ct = q[2 * n:3 * n]; vf = q[3 * n:4 * n - 1]
            vt = torch.cat([vf, (s - vf.sum()).reshape(1)]); bt = q[-1]
            zz = torch.tanh(Xt @ Wt.T + ct) @ vt + bt
            return (torch.nn.functional.binary_cross_entropy_with_logits(zz, Yt)
                    + 0.5 * lam * ((Wt ** 2).sum() + (ct ** 2).sum()))
        return torch.autograd.functional.hessian(L, torch.tensor(q0, dtype=torch.float64)).numpy()
    return H_train


def fold_rate(Hs, ss, s_f, Pb, nA, sdot, lr=0.01, window=0.05):
    """(Λ_F, ε_F, |m_c|) for block P = (W, c, v, b) from Hessians Hs at scales ss (sb_fold's fit, window 0.05)."""
    pdiag = np.r_[np.full(2 * nA, Pb[0]), np.full(nA, Pb[1]), np.full(nA - 1, Pb[2]), Pb[3]]
    ph = np.sqrt(pdiag)
    lam = np.array([float(np.linalg.eigvalsh((ph[:, None] * H) * ph[None, :])[0]) for H in Hs])
    m = ss >= s_f - window
    dd = s_f - ss[m]
    coef, *_ = np.linalg.lstsq(np.vstack([dd, dd ** 2]).T, lam[m] ** 2, rcond=None)
    mc = abs(coef[0]) / 4
    Lam = math.sqrt(mc * s_f)
    return Lam, (sdot / s_f) / (lr * Lam), mc, lam


def sb_eps_f(window=0.05) -> dict:
    from . import sb_fold as SF
    from . import simplicity_bias_v2 as v2
    v2._torch()
    X, y = v2.data()
    s_f = SF.fold_of("M")
    gz = np.load(SF.OUT / "grid_M.npz"); nA = int(gz["n_active"])
    sel = gz["s"] >= s_f - window
    ss = gz["s"][sel]
    Hf = _sb_hessian_fn(nA, X, y, SF.LAM)
    Hs = [Hf(u, s) for u, s in zip(gz["U"][sel], ss)]
    per = pd.read_csv(RESULTS / "sb_fold" / "per_run.csv").set_index("seed")
    runs, Pall, sdots = [], [], []
    for f in sorted(SF.V3_RUNS.glob("run_*.npz")):
        seed = int(f.stem.split("_")[1])
        z = np.load(f)
        if not (z["s"] >= s_f).any():
            continue
        k = int(np.argmax(z["s"] >= s_f))
        vh = SF.reconstruct_vhat(z["params"][:k + 1])
        p = 1 / (np.sqrt(vh[k]) + 1e-8)
        Pb = [float(np.median(p[:8])), float(np.median(p[8:12])), float(np.median(p[12:16])), float(p[16])]
        w = min(100, k)
        sdot = float((z["s"][k] - z["s"][k - w]) / w)
        Lam, eps, mc, _ = fold_rate(Hs, ss, s_f, Pb, nA, sdot, window=window)
        Pall.append(Pb); sdots.append(sdot)
        runs.append({"seed": seed, "k_F": k, "s_at_k_F": float(z["s"][k]), "sdot_F": sdot, "P_blocks": Pb,
                     "abs_mc": mc, "Lambda_F": Lam, "eps_F": eps, "group": str(per.loc[seed, "group"]),
                     "basin_at_fold": str(per.loc[seed, "basin_at_fold"])})
    C = json.loads((SF.OUT / "fold_constant_adam.json").read_text())
    Pmed = np.median(np.array(Pall), axis=0)
    Lam0, eps0, mc0, lam0 = fold_rate(Hs, ss, s_f, Pmed, nA, float(np.median(sdots)), window=window)
    ref = np.array(C["lambda_min_path"]["adamP"])
    res = {"label": LABEL, "s_F": s_f, "lr": 0.01, "window": window, "n_runs": len(runs),
           "population": {"eps_F_committed": C["fold_delay_law_post_hoc"]["eps_F"],
                          "Lambda_F_committed": C["fold_delay_law_post_hoc"]["Lambda_F"],
                          "eps_F_recomputed_median_P": eps0, "Lambda_F_recomputed_median_P": Lam0,
                          "lambda_path_max_rel_diff": float(np.max(np.abs(lam0 - ref) / np.abs(ref))),
                          "P_median_max_rel_diff": float(np.max(np.abs(Pmed - np.array(C["P_block_medians_W_c_v_b"]))
                                                                / np.array(C["P_block_medians_W_c_v_b"])))},
           "runs": runs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sb_eps_F.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------ assembly
SETTINGS = ("width-1 sine", "width-1 GELU", "band task in R^d", "width 2", "simplicity bias")

EXCLUDED = [
    ("width-1 sine", "lag tests φ ≠ 1 (lag1, lag2-prim, lag2-tstar; w₂ slowed or sped up) and residual-mechanism arms "
                     "4b reset / teleport / teleport_reset", "non-standard (slowed / intervened)"),
    ("width-1 sine", "Track 1B ramp (results/ramp/runs.csv), Test 2B (track2b), ramp_boundary, ramp2",
     "imposed growth of s (ramped), warm-up at a stationary point"),
    ("width-1 sine", "ramp R4 free Adam (results/ramp/free_runs.csv; η = 0.01, 0.005, 0.0025)",
     "free, but no χ or κχ committed (η = 0.01 is standard; would need a replay of 80 runs)"),
    ("width-1 sine", "Phase 1A W1 dev/est runs (results/phase1a/runs_w1_sgd.jsonl)",
     "free SGD, but no κχ at the actual switch committed"),
    ("width-1 GELU", "GELU-T (results/gelu_transfer), Phase 1C G, Phase 1A gelu_random",
     "held at s₀ on a copy, then released with SGD η = 0.03"),
    ("width-1 GELU", "act_fold (results/act_fold)", "the same Track 3A runs; no κ/χ"),
    ("width 2", "W2-A arms T, D, T′ (results/width2_asym), Phase 1C T, T′, Phase 1A w2a_T/Tp",
     "started at a branch point, held, then slowed SGD"),
    ("width 2", "T2-3b, T2-3c, T2-3d (results/width2_lag/parts_T2-3{b,c,d}.csv)", "output lr × φ₂ (slowed)"),
    ("width 2", "standard-init T2-3 runs (results/asym_sealed/train_uniform_init_seeds0-79.csv)",
     "sealed / withdrawn before analysis; no κ/χ"),
    ("width 2", "width2_fold", "0 folds found in the 79 T2-3 runs; no ε_F"),
    ("simplicity bias", "Phase 2A exploration (results/designs/phase2a_explore)",
     "slow releases from the exact M point, plain GD; not free training"),
]


def build() -> dict:
    cells = width1_cells() + gelu_cells() + band_cells() + width2_cells() + sb_cells()
    pooled = {}
    for s in SETTINGS:
        for opt in ("Adam", "SGD"):
            cs = [c for c in cells if c["setting"] == s and c["optimiser"] == opt and c["quantity"] == "kappa_chi"
                  and "not pooled" not in c["cell"]]
            if cs:
                pooled[f"{s} | {opt}"] = {"cells": [c["cell"] for c in cs], "summary": None}
    return {"label": LABEL, "bounds": list(BOUNDS), "quantiles": list(QUANTS), "cells": cells, "pooled": pooled,
            "excluded": [{"setting": a, "runs": b, "reason": c} for a, b, c in EXCLUDED]}


def _pool_values() -> dict:
    """The finite κχ values of each pooled group, re-read from the cell loaders (the JSON stores summaries only)."""
    vals = {}
    pr = pd.read_csv(RESULTS / "lag_law" / "predictions.csv"); pr["arm"] = pr.arm.astype(str)
    ta = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
    dg = pd.read_csv(RESULTS / "track_a" / "diag_p_at_crossing.csv")
    t2 = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
    w1 = [json.loads(l) for l in (RESULTS / "phase1c" / "observed_W1.jsonl").read_text().splitlines() if l.strip()]
    adam = np.r_[pr[(pr.set == "lag1") & (pr.arm == "1.0")].pred_r, pr[pr.set == "TaskB"].pred_r,
                 dg[dg.crossed].p_cross_r_cf, t2[t2.crossed.astype(bool)].r_cf]
    sgd = np.r_[pr[pr.set == "SGD"].pred_r, ta[(ta.opt == "sgd") & ta.crossed.astype(bool)].r_cf,
                [r["r_cf"] for r in w1 if r.get("crossed") and r.get("r_cf") is not None]]
    vals["width-1 sine | Adam"], vals["width-1 sine | SGD"] = adam, sgd
    _, c, _ = gelu_runs()
    vals["width-1 GELU | Adam"] = c[~c.early].kchi.to_numpy()
    b = pd.read_csv(RESULTS / "band_rd" / "scored_runs.csv")
    t = pd.read_csv(RESULTS / "track2c" / "observed_runs.csv")
    vals["band task in R^d | Adam"] = np.r_[b[b.crossed.astype(bool)].pred_r, t[t.crossed.astype(bool)].r_pred]
    d = pd.read_csv(RESULTS / "width2_lag" / "parts_T2-3.csv")
    vals["width 2 | Adam"] = d[d.status == "ok"].r_pred.to_numpy()
    return vals


def finalize(J: dict) -> dict:
    vals = _pool_values()
    for k, p in J["pooled"].items():
        n_cells = sum(c["summary"]["n"] for c in J["cells"] if c["cell"] in p["cells"] and c["setting"] == k.split(" | ")[0])
        s = summarize(vals[k])
        assert s["n"] == n_cells, (k, s["n"], n_cells)
        p["summary"] = s
    return J


def _f(x, nd=4):
    if x is None:
        return "–"
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _row(name, c, s):
    fr = lambda b: "–" if s[f"frac_abs_le_{b}"] is None else f"{s[f'n_abs_le_{b}']}/{s['n']} ({100 * s[f'frac_abs_le_{b}']:.0f}%)"
    return (f"| {name} | {c} | {s['n']} | {_f(s['min'])} | {_f(s['q10'])} | {_f(s['median'])} | {_f(s['q90'])} | "
            f"{_f(s['max'])} | {fr(0.1)} | {fr(0.02)} |")


HDR = ("| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |·| ≤ 0.1 | |·| ≤ 0.02 |\n"
       "|---|---|---|---|---|---|---|---|---|---|")


def _eps_min(J):
    c = [c for c in J["cells"] if c["quantity"] == "eps_F"]
    return _f(c[0]["summary"]["min"], 2) if c else "not computed"


def render_md(J: dict) -> str:
    L = ["# Phase 2C: κχ at the crossing in standard training (DESCRIPTIVE)", "",
         "**DESCRIPTIVE. No registration, no criteria, no verdict.** Producer `src/phase2c_map.py`; data "
         "`results/phase2c/kappa_chi_map.json`. Every κχ is read from a committed producer output (cited per cell); "
         "nothing was retrained. The one computed quantity is the per-run ε_F of the simplicity-bias runs "
         "(`results/phase2c/sb_eps_F.json`).", "",
         "Question: for standard initialisation and the standard learning rate of each setting, how often does ordinary "
         "training lie inside the regime where the linear lag law r = κχ applies? The reference is Test 2B's C1: the "
         "law held within ±25% at κχ ≤ 0.1 (all six cells, κχ ∈ {0.02, 0.05, 0.1}); κχ\\* = 0.2 at both a.", "",
         "Definitions (§13.1): κ = λ_min·[∇G·(PH)⁻¹θ\\*′]/[∇G·θ\\*′], χ = (ṡ/s\\*)/(ηλ_min(P^{1/2}HP^{1/2})). Fractions "
         "are over the runs with a finite value and use |κχ|. Quantiles are linear-interpolation (numpy default). Signed "
         "and |·| summaries differ only where a value is negative; those cells list the negatives.", "",
         "## Pooled per setting and optimiser (cells marked \"not pooled\" excluded)", "",
         "| setting | optimiser | n | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, p in J["pooled"].items():
        s = p["summary"]; st, opt = k.split(" | ")
        fr = lambda b: f"{s[f'n_abs_le_{b}']}/{s['n']} ({100 * s[f'frac_abs_le_{b}']:.0f}%)"
        L.append(f"| {st} | {opt} | {s['n']} | {_f(s['min'])} | {_f(s['q10'])} | {_f(s['median'])} | {_f(s['q90'])} "
                 f"| {_f(s['max'])} | {fr(0.1)} | {fr(0.02)} |")
    L += ["", "Simplicity bias has no pooled κχ: κ is undefined (below).", ""]
    for st in SETTINGS:
        cs = [c for c in J["cells"] if c["setting"] == st]
        L += [f"## {st}", ""]
        for q in ("kappa_chi", "chi", "eps_F"):
            cq = [c for c in cs if c["quantity"] == q]
            if not cq:
                continue
            if q != "kappa_chi":
                qn = "χ only (κ undefined)" if q == "chi" else "ε_F at M's fold"
                L += [f"Quantity: **{qn}**; the |·| ≤ 0.1 / 0.02 "
                      "columns apply to this quantity, not to κχ.", ""]
            L += [HDR.replace("|·|", "|κχ|") if q == "kappa_chi" else HDR]
            for c in cq:
                L.append(_row(c["cell"], f"{c['n_runs']} / {c['n_cross']}", c["summary"]))
            L.append("")
        L += ["Sources and definitions:", ""]
        for c in cs:
            s = c["summary"]
            neg = (f" Signed and |·| differ: {s['n_negative']} negative; |·| median {_f(s['abs_median'])}, "
                   f"q10 {_f(s['abs_q10'])}, q90 {_f(s['abs_q90'])}, max {_f(s['abs_max'])}." if signed_differs(s) else "")
            L.append(f"- **{c['cell']}** — {c['optimiser']} lr {c['lr']}; `{c['source']}`, column `{c['column']}`. "
                     f"κ: {c['kappa_def']}. χ: {c['chi_def']}.{(' ' + c['note'] + '.') if c['note'] else ''}{neg}")
        L.append("")
    L += ["## Caveats", "",
          "- **Width-1 sine.** χ in the lag_law sets divides ṡ by s_c, not s\\* (factor 1/(1 + r); per-arm median r ≤ 0.081, §13.3(i)). Track A "
          "Adam is shown with P at the crossing (POST HOC diagnostic) and, unpooled, with the registered rule-point P. "
          "Task B (n = 200) and the lag-test reference runs (n = 6,400) differ only in sample size.",
          "- **GELU.** One frozen population κ for every run. The early crossings (s_c < 0.5·s_glob = 3.32; s_c 0.004-0.84, "
          "against ≥ 5.09 for every other crossing) have κχ of either sign with |κχ| up to ~7·10¹², because s is falling "
          "or relax ≈ 0. §13.7 places GELU's 15 early crossings (s ≤ 0.45) on the tail-placed one-sided-ramp branch, "
          "where the continuation finds no switch. They are listed separately and not pooled. One early crossing (seed "
          "850068) has NaN χ.",
          "- **Band task.** 3B measures χ at the crossing; 2C measures it at t_sw (the first |w₂| ≥ s_own,d), which for "
          "26 crossers lies at or after the crossing step (flagged per cell; the t_sw-before-crossing subset is shown "
          "unpooled). 3B and 2C use the width-1 population κ_k; d = 2 and d = 4 cells share seeds by design. Three 3B "
          "runs sit on a negative-κ winding.",
          "- **Width 2.** Only T2-3 has per-run κ and χ, and it uses the registered MATCHED initialisation (output "
          "weights scaled by k = 0.04209), not the standard one; the standard-init version is sealed with no κ/χ. No "
          "T2-3 run is on its branch at the crossing (every dist_c > 0.05), so the law's premise (tracking the branch "
          "whose switch is used, §13.3(iv)(d)) fails for every run; κχ here is formal. Seven runs have κ₂ < 0.",
          "- **Simplicity bias.** κ is undefined: M has no switch of its own. Only v3's χ at the ρ₂ upward passage exists, "
          "and its definition is not §13's (ṡ/s_c; H at s_switch on the S side, not at the run's branch; P without the v "
          "block). v3's initialisation is PyTorch default with the output scale capped at 0.5·s_switch. ε_F per run is "
          "computed here with sb_fold's construction and the run's own P; the population ε_F with median P and median "
          "ṡ is the committed 5.58 (reproduced). No run is on M at the fold (basin at the fold: S 30, other 5, L0 4, "
          "S2 1), so ε_F describes the drive rate at s_F, not an event of the run. sb_fold calls ε_F > 0.3 outside the "
          "small-ε regime of §14; the per-run minimum is " + _eps_min(J) + ".",
          "- Fractions are over finite values. Runs that never cross, or cross with no defined κχ, are counted in "
          "\"runs / crossed\" but not in n.", "",
          "## Excluded (non-standard, or no κχ committed)", "", "| setting | runs | reason |", "|---|---|---|"]
    L += [f"| {e['setting']} | {e['runs']} | {e['reason']} |" for e in J["excluded"]]
    L += ["", "## Not computed", "",
          "- χ for the ramp R4 free-Adam runs at η = 0.01 (no χ committed; a replay of 80 runs would be needed).",
          "- κχ at width 2 with standard initialisation (no such runs analysed; the sealed runs carry no κ/χ).",
          "- Any κ for the simplicity-bias benchmark (undefined on M).", ""]
    return "\n".join(L)


def main() -> dict:
    J = json.loads(json.dumps(finalize(build())))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kappa_chi_map.json").write_text(json.dumps(J, indent=1))
    MD.write_text(render_md(J))
    return J


if __name__ == "__main__":
    if sys.argv[1:] == ["eps"]:
        sb_eps_f()
    else:
        main()
