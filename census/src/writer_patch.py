"""Producers for WRITER_INPUTS_v4_patch.md (completion inputs for the submission).  Every table here is derived
from committed artifacts with the registered definitions (imported from the scorers, not re-implemented).

    python -m src.writer_patch            # all five tables
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGDIR = RESULTS / "figures" / "v4"


# ------------------------------------------------------------------------------------------ 1. per setting
def per_setting(n_boot=10_000):
    """The prospective own-seed test per setting (16 = 8 windows × 2 a): crossings and the mean per-run
    |log(R_cross/pred)| for U_own (own threshold), U, C, C_own and S-early (unfitted; misses excluded and counted).
    Per-setting intervals are run-level bootstrap 95% (descriptive: the registered unit of uncertainty is the
    window, so no per-setting interval was registered).  Per-a rows carry the registered window-level bootstrap."""
    from .prospective_own import A_VALUES, B_WIN, _win_boot, early_prediction
    pr = pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip")
    rn = pd.read_csv(RESULTS / "prospective_own_runs.csv", float_precision="round_trip")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2
    rng = np.random.default_rng(0)
    rows = []
    models = ("U_own", "U", "C", "C_own", "S_early")
    for (w, a), g in d.groupby(["window", d.a.round(2)], sort=True):
        x = g[g.R_cross.notna()].copy()
        x["S_early"] = [early_prediction(r) if early_prediction(r) is not None else np.nan for r in x.to_dict("records")]
        r = {"window": w, "a": a, "n_runs": len(g), "n_crossed": len(x), "S_early_undefined": int(x.S_early.isna().sum())}
        for m in models:
            e = np.abs(np.log(x.R_cross / x[m])).dropna().values
            boots = e[rng.integers(0, len(e), (n_boot, len(e)))].mean(axis=1)
            r[f"{m}_mean_abs_log_err"] = float(e.mean())
            r[f"{m}_runlevel_lo"] = float(np.percentile(boots, 2.5))
            r[f"{m}_runlevel_hi"] = float(np.percentile(boots, 97.5))
        rows.append(r)
    t = pd.DataFrame(rows)
    agg = []
    for a in A_VALUES:
        s = t[t.a == a]
        r = {"window": "all 8 (window-level)", "a": a, "n_runs": int(s.n_runs.sum()), "n_crossed": int(s.n_crossed.sum()),
             "S_early_undefined": int(s.S_early_undefined.sum())}
        for m in models:
            m0, lo, hi = _win_boot(s[f"{m}_mean_abs_log_err"].values, B_WIN)
            r[f"{m}_mean_abs_log_err"], r[f"{m}_windowlevel_lo"], r[f"{m}_windowlevel_hi"] = m0, lo, hi
        agg.append(r)
    t = pd.concat([t, pd.DataFrame(agg)], ignore_index=True)
    t.to_csv(RESULTS / "writer_patch_prospective_own_per_setting.csv", index=False)
    return t


# ------------------------------------------------------------------------------------------ 2. windows
def windows():
    """Exact window edges: I = [−i, i] (class 0), O = [−o2, −o1] ∪ [o1, o2] (class 1), for Block 3's held-out H
    windows and the own-seed test's V windows; plus the frozen calibration constants."""
    from .prospective_own import LAMBDA, RHO_RES
    h = pd.read_csv(RESULTS / "prospective_windows.csv")
    v = pd.read_csv(RESULTS / "prospective_own_windows.csv")
    rows = []
    for r in h.itertuples():
        rows.append({"window": f"H{int(round(r.quantile * 100)):02d}", "test": "Block 3", "I": f"[-{r.i}, {r.i}]",
                     "O": f"[-{r.o2}, -{r.o1}] U [{r.o1}, {r.o2}]", "i": r.i, "o1": r.o1, "o2": r.o2,
                     "kappa0_lo": r.kappa0_lo, "kappa0_hi": r.kappa0_hi})
    for r in v.itertuples():
        rows.append({"window": r.name, "test": "own-seed", "I": f"[-{r.i}, {r.i}]",
                     "O": f"[-{r.o2}, -{r.o1}] U [{r.o1}, {r.o2}]", "i": r.i, "o1": r.o1, "o2": r.o2,
                     "kappa0_lo": r.kappa0_lo, "kappa0_hi": r.kappa0_hi})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "writer_patch_windows.csv", index=False)
    c = pd.DataFrame([{"a": a, "lambda_C": LAMBDA[a], "rho_res_C_own": RHO_RES[a]} for a in sorted(RHO_RES)])
    c.to_csv(RESULTS / "writer_patch_calibration.csv", index=False)
    return t, c


# ------------------------------------------------------------------------------------------ 3. W(s, a)
def w_parts(s, a, x, y, win=(-0.8, 0.8, 1.2, 2.0)):
    """The localisation bound as implemented (profiled_bnb.w_bound), with its ingredients per side: the opposing
    group counts, the chosen inner cut c, π = m/n, and W for that side."""
    i_lo, i_hi, o_lo, o_hi = win
    n = len(x)
    out = []
    for sign in (+1, -1):
        n_outer = int(((sign * x <= -o_lo) & (y == 1)).sum())
        top = i_hi if sign > 0 else -i_lo
        best = (math.inf, None, None, None)
        for c in np.linspace(min(0.0, top - 1e-3), top - 1e-3, 80):
            n_inner = int(((sign * x >= c) & (y == 0)).sum())
            m = min(n_outer, n_inner)
            if m == 0 or o_lo + c <= 0:
                continue
            pi = m / n
            delta = 2 * math.log(2 ** (1 / pi) - 1) if 1 / pi < 1000 else 2 * (1 / pi) * math.log(2)
            W = (2 * a + delta / s) / (o_lo + c)
            if W < best[0]:
                best = (W, float(c), n_inner, pi)
        out.append({"side": "w1 > 0" if sign > 0 else "w1 < 0", "n_outer": n_outer, "c": best[1],
                    "n_inner": best[2], "pi": best[3], "W_side": best[0]})
    return out


def w_bound_table():
    """W(s, a) at every certified a (population objective) at both ends of the certified R_glob and R_solve
    brackets, with the ingredients, agreement with profiled_bnb.w_bound, and a direct numerical check: the
    profiled loss exceeds log 2 on a dense b₁ grid at |w₁| = W and beyond (W, 1.5W, 3W)."""
    from .conditional_certified import _population
    from .profiled_bnb import profile, w_bound
    x, y = _population()
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    rows = []
    b1 = np.linspace(0, 2 * math.pi, 721)[:-1]
    for r in br.itertuples():
        for end, w2 in (("lo", r.w2_lo), ("hi", r.w2_hi)):
            s = float(w2)
            parts = w_parts(s, r.a, x, y)
            W = max(p["W_side"] for p in parts)
            assert abs(W - w_bound(s, r.a, x, y)) <= 1e-12
            minL = math.inf
            for fac in (1.0, 1.5, 3.0):
                for sg in (+1, -1):
                    L = profile(np.full(len(b1), sg * fac * W), b1, s, r.a, x, y)[0]
                    minL = min(minL, float(np.min(L)))
            rows.append({"a": r.a, "kind": r.kind, "end": end, "s": s, "W": W,
                         **{f"{k}_{p['side'][3]}": p[k] for p in parts for k in ("n_outer", "c", "n_inner", "pi", "W_side")},
                         "min_profiled_loss_at_and_beyond_W": minL, "exceeds_log2": minL > math.log(2)})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "writer_patch_w_bound.csv", index=False)
    return t


# ------------------------------------------------------------------------------------------ 4. census by block and file
def census_by_block():
    """Headline-convention census (one row per registered prediction) counted by block and by registration file."""
    d = pd.read_csv(RESULTS / "registration_census.csv")
    out = []
    for key in ("block", "registration_file"):
        g = d.groupby([key, "scoring"]).verdict.value_counts().unstack(fill_value=0)
        for v in ("PASS", "FAIL", "PARTIAL", "UNRESOLVED"):
            if v not in g.columns:
                g[v] = 0
        g = g[["PASS", "FAIL", "PARTIAL", "UNRESOLVED"]].reset_index()
        g["n"] = g[["PASS", "FAIL", "PARTIAL", "UNRESOLVED"]].sum(axis=1)
        g.insert(0, "by", key)
        g = g.rename(columns={key: "group"})
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(RESULTS / "writer_patch_census_by_block.csv", index=False)
    return t


# ------------------------------------------------------------------------------------------ 5. figure sizes
def figure_sizes():
    import pypdf
    rows = []
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        box = pypdf.PdfReader(str(pdf)).pages[0].mediabox
        rows.append({"file": f"results/figures/v4/{pdf.name}", "width_in": round(float(box.width) / 72, 3),
                     "height_in": round(float(box.height) / 72, 3)})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "writer_patch_figure_sizes.csv", index=False)
    return t


def main():
    pd.set_option("display.width", 250)
    for f in (per_setting, windows, w_bound_table, census_by_block, figure_sizes):
        r = f()
        print(f"== {f.__name__}")
        for x in (r if isinstance(r, tuple) else (r,)):
            print(x.to_string(index=False)[:3000])


# ------------------------------------------------------------------------------------------ render the patch file
LEMMA = r"""**Lemma (localisation of the finite-a conditional search).**

- **Setting**:
  - z(x) = s·f_a(w₁x + b₁) + b₂, with s = |w₂| > 0 in the orientation w₂ > 0, and f_a(t) = t + a sin t, so
    t − a ≤ f_a(t) ≤ t + a.
  - Class 1 lies on O = [−2, −1.2] ∪ [1.2, 2] and class 0 on I = [−0.8, 0.8], with n points in total and balanced
    classes (ȳ = ½; true of the 800-point population and of every 400-point training set).
  - L* is the profiled loss, minimised over b₂.
- **Data**: the population objective has n = 800 points:
  - 400 inner points x = linspace(−0.8, 0.8, 400), class 0;
  - 200 outer points x = linspace(1.2, 2.0, 200), class 1, and their 200 negatives, class 1.
  (`blockB_landscape.population_data`; a training set is `fold1d.make_data(200, seed)`.)
- **Notation**: for a cut c ∈ [0, 0.8), let n_O⁻ = #{class 1: x ≤ −1.2} and n_I^{≥c} = #{class 0: x ≥ c}. Set
  π(c) = min(n_O⁻, n_I^{≥c})/n and δ(π) = 2 log(2^{1/π} − 1).
- **The cut grid**: C = {0.799·k/79 : k = 0, 1, …, 79}, the 80 cuts the certified code evaluates
  (`np.linspace(0, 0.799, 80)`).
- **Claim**: if w₁ > W₊(s, a) = min over c ∈ C of (2a + δ(π(c))/s)/(1.2 + c), then L*(w₁, b₁; s) > log 2 for every b₁.
  (The argument holds for every single c; minimising over the grid C reproduces the code exactly.)
  The case w₁ < 0 is the mirror image: use the right-outer class-1 points (x ≥ 1.2) against the class-0 points
  with x ≤ −c, which gives W₋.
- **Conclusion**: with W(s, a) = max(W₊, W₋), every global conditional minimiser has |w₁| ≤ W. The constant
  predictor attains exactly log 2, so the global minimum is at most log 2.
- **b₁**: it ranges over [0, 2π). A shift of b₁ by 2π adds 2πs to every logit, which the profiled b₂ absorbs.

**Proof** (w₁ > 0).
1. For x ≤ −1.2, w₁x + b₁ ≤ −1.2w₁ + b₁, so z ≤ A := s(−1.2w₁ + b₁ + a) + b₂.
2. For x ≥ c, z ≥ s(cw₁ + b₁ − a) + b₂ = A + Δ, with Δ = s((1.2 + c)w₁ − 2a).
3. Take Δ > 0 and let M = A + Δ/2.
   - If M ≥ 0, every class-0 point with x ≥ c has z ≥ Δ/2, so its loss softplus(z) is at least softplus(Δ/2).
   - If M < 0, every class-1 point with x ≤ −1.2 has z < −Δ/2, so its loss softplus(−z) exceeds softplus(Δ/2).
4. All other losses are non-negative. So for every b₂, L ≥ π(c)·softplus(Δ/2), and hence
   L* ≥ π(c)·softplus(Δ/2).
5. π·softplus(Δ/2) > log 2 ⟺ Δ > δ(π) ⟺ w₁ > (2a + δ(π)/s)/(1.2 + c).
6. Every c gives a valid bound, so the minimum over the grid C does too.
   (For 1/π ≥ 1000 the code replaces δ by the larger 2 log 2/π, which is conservative; this never occurs here.) ∎

**Illustration** (a = 1.30, s = 4.95, the lower end of the certified R_glob bracket; population data):
- **With the cut c = 0.4**: n_O⁻ = 200, n_I^{≥0.4} = 100, π = 1/8, δ = 2 log 255 = 11.08. This gives
  W₊ = (2.6 + 11.08/4.95)/1.6 = 3.024.
- **With the grid-optimal cut c = 0.799·22/79 = 0.22251**: n_I^{≥c} = 145, π = 145/800 = 0.18125,
  δ = 2 log(2^{1/0.18125} − 1) = 7.6045, W₊ = (2.6 + 7.6045/4.95)/(1.2 + 0.22251) = 2.9077.
  W₋ is the same, because the data are x-symmetric, so **W = 2.908**: the value the certified search used.
"""


def _md(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in df.to_dict("records"):
        body += "| " + " | ".join(fmt.get(c, lambda v: str(v))(r[c]) for c in cols) + " |\n"
    return head + body


def render():
    ps = pd.read_csv(RESULTS / "writer_patch_prospective_own_per_setting.csv")
    wi = pd.read_csv(RESULTS / "writer_patch_windows.csv")
    ca = pd.read_csv(RESULTS / "writer_patch_calibration.csv")
    wb = pd.read_csv(RESULTS / "writer_patch_w_bound.csv")
    cb = pd.read_csv(RESULTS / "writer_patch_census_by_block.csv")
    fs = pd.read_csv(RESULTS / "writer_patch_figure_sizes.csv")
    f3 = lambda v: f"{v:.3f}"
    f4 = lambda v: f"{v:.4f}"

    def cell(r, m):
        lo = r.get(f"{m}_runlevel_lo") if not pd.isna(r.get(f"{m}_runlevel_lo", np.nan)) else r.get(f"{m}_windowlevel_lo")
        hi = r.get(f"{m}_runlevel_hi") if not pd.isna(r.get(f"{m}_runlevel_hi", np.nan)) else r.get(f"{m}_windowlevel_hi")
        return f"{r[f'{m}_mean_abs_log_err']:.3f} [{lo:.3f}, {hi:.3f}]"
    rows = []
    for r in ps.to_dict("records"):
        rows.append({"setting": f"{r['window']}, a = {r['a']:.2f}", "crossed": f"{r['n_crossed']}/{r['n_runs']}",
                     **{m: cell(r, m) for m in ("U_own", "U", "C", "C_own", "S_early")}})
    t1 = _md(pd.DataFrame(rows), ["setting", "crossed", "U_own", "U", "C", "C_own", "S_early"])
    t2 = _md(wi, ["window", "test", "I", "O", "kappa0_lo", "kappa0_hi"], {"kappa0_lo": lambda v: f"{v:.5f}",
                                                                         "kappa0_hi": lambda v: f"{v:.5f}"})
    wbg = wb[wb.end == "lo"]
    t3 = _md(wbg, ["a", "kind", "s", "W", "c_>", "n_inner_>", "pi_>", "min_profiled_loss_at_and_beyond_W"],
             {"a": lambda v: f"{v:.2f}", "s": f4, "W": f3, "c_>": f4, "pi_>": lambda v: f"{v:.5f}",
              "min_profiled_loss_at_and_beyond_W": f3})
    cbb = cb[cb.by == "block"]; cbf = cb[cb.by == "registration_file"]
    t4 = _md(cbb, ["group", "scoring", "PASS", "FAIL", "PARTIAL", "UNRESOLVED", "n"])
    t4f = _md(cbf, ["group", "scoring", "PASS", "FAIL", "PARTIAL", "UNRESOLVED", "n"])
    desc = {
        "v4_decomposition.pdf": "Phase 1: fraction solved, failed on placement (G ≤ 0) or failed on bias (G > 0, b₂ outside its interval) against ε = a − 1; n = 400 runs per a; Clopper–Pearson 95%.",
        "v4_thresholds.pdf": "(a) certified R_glob and R_solve at a = 1.30–1.60 against free-training crossing violins, with the certified ε → 0 limits; (b) small ε: certified brackets at a = 1.01–1.04, the sharp limit and the first-order line; the registered c₁ test INCONCLUSIVE.",
        "v4_cond_candidates.pdf": "Block 1 at a = 1.30: (a) every candidate of the frozen search above the single retained branch; (b) the branch's gap crossing G = 0 once, with the certified R_glob and R_solve brackets.",
        "v4_fixed_scale.pdf": "(a) Block 4 placement, (b) Block 5 retention with E-2's criterion and Block E's 33/37 (FAIL), (c) horizon extension (Q1, Q2 FAIL), against held R/R_glob; Clopper–Pearson 95%.",
        "v4_mirror_branches.pdf": "Post hoc: crossing R against the own threshold of the mirror branch occupied at the crossing, marked by branch; Spearman ρ = 0.997 (a = 1.30), 0.995 (a = 1.50), bootstrap 95%; n = 38 per a.",
        "v4_prospective.pdf": "Block 3: predicted against observed median crossing R in 8 held-out settings; C (λ fitted) filled, U (nothing fitted) open, B1, B2; bootstrap 95%.",
        "v4_prospective_own.pdf": "Prospective own-seed test: (a) per-run crossing R against the frozen own threshold, with the fitted C_own lines; (b) mean per-run |log error| of U, C, U_own, C_own at each a with the registered verdicts.",
    }
    fs["description"] = fs.file.map(lambda f: desc[Path(f).name])
    t5 = _md(fs, ["file", "width_in", "height_in", "description"])
    tl = pd.read_csv(RESULTS / "registration_tally.csv").set_index("scope")
    h = tl.loc["scored by registered rules"]; p_ = tl.loc["assigned post hoc in the census"]
    text = f"""# Writer inputs v4 — completion patch (2026-09-24)

Written for the paper's writer. Each item is tied to a ledger ID in `src/verify_ledger.py` (group "V4 patch").
Producer: `src/writer_patch.py` (`python -m src.writer_patch`), which also renders this file. Every item here is
**ready for the submission**.

## WP-1. Prospective own-seed test, per setting

Per setting: the number of runs that crossed, and the mean per-run |log(R_cross / prediction)| for five
predictors:
- **U_own**: the run's own threshold, nothing fitted;
- **U**: the population R_glob;
- **C**: λ(a)·R_glob, fitted;
- **C_own**: ρ_res(a)·R_own, fitted;
- **S-early**: the early branch's own threshold, unfitted. No run was undefined for S-early.

**Intervals**:
- **Per-setting rows**: a run-level bootstrap 95% (10,000 resamples, seed 0). This is **descriptive**. The
  registered unit of uncertainty is the window, so no per-setting interval was registered.
- **The "all 8" rows**: the registered window-level bootstrap (10,000 resamples of windows, seed 0).

Source: `writer_patch_prospective_own_per_setting.csv`, built from `prospective_own_predictions.csv` and
`prospective_own_runs.csv` with the registered definitions.

{t1}
## WP-2. Window definitions and the calibration rules

Class 0 on I, class 1 on O. Every window is x-symmetric. Source: `writer_patch_windows.csv`, built from
`prospective_windows.csv` and `prospective_own_windows.csv`.

{t2}
**Calibration rules as registered** (`writer_patch_calibration.csv`):
- **C (Block 3)**: C = λ(a)·R_glob, with λ(1.30) = {ca.lambda_C[0]} and λ(1.50) = {ca.lambda_C[1]}. λ is the base
  window's median crossing R divided by its certified R_glob, frozen before Block 3.
- **C_own (own-seed test)**: C_own = ρ_res(a)·R_own, with ρ_res(1.30) = {ca.rho_res_C_own[0]} and
  ρ_res(1.50) = {ca.rho_res_C_own[1]}.
  - ρ_res = exp(median log(R_cross/R_own)) over the base window's phase 2b seeds 0–39 that crossed within
    12,000 steps.
  - It was frozen before any computation on the new settings (`prospective_own_prediction.md`).

## WP-3. The localisation bound W(s, a) for the finite-a certificates

{LEMMA}
**Checked at every certified a** (`writer_patch_w_bound.csv`: population objective, both ends of every
certified R_glob and R_solve bracket):
- The W computed here equals the W used by the certified search (`profiled_bnb.w_bound`) to 1e−12.
- The profiled loss at |w₁| = W, 1.5W and 3W, on 720 values of b₁ and both signs of w₁, is at least 4.34 > log 2
  everywhere.
- The table shows the lower end of each bracket; both ends are in the CSV. The two sides (w₁ > 0 and w₁ < 0)
  give equal W because the population is x-symmetric.

{t3}
## WP-4. Registration census by block and by registration file

**Headline convention** (as in the 2026-09-23 census): each registered prediction is counted once across a.
- **Headline**: {int(h.n)} scored by their registered rules: {int(h.PASS)} PASS, {int(h.FAIL)} FAIL, {int(h.PARTIAL)} PARTIAL,
  {int(h.UNRESOLVED)} UNRESOLVED.
- **Post hoc**: {int(p_.n)} assigned post hoc: {int(p_.PASS)} / {int(p_.FAIL)} / {int(p_.PARTIAL)} / {int(p_.UNRESOLVED)}.

**Row-level tables shipped in the supplementary**:
- `results/registration_census.csv`: the headline, one row per prediction;
- `results/registration_census_by_unit.csv`: the appendix view, per a;
- `results/registration_census_v4_gates_and_reported.csv`: validity gates and no-criterion items;
- `results/registration_tally.csv`: the tallies.

Source for the tables below: `writer_patch_census_by_block.csv`.

### By block
{t4}
### By registration file
{t4f}
## WP-5. Figure manifest after the page-width rebuild

All figures are built at ICLR's text width (5.5 in) and placed at their built size. Every glyph is at least 8 pt.
Producer: `src/figures_v4.py`. Sizes are read from the PDFs (`writer_patch_figure_sizes.csv`).

{t5}"""
    text += wp6()
    text += wp7()
    out = RESULTS.parent / "paper" / "WRITER_INPUTS_v4_patch.md"
    out.write_text(text)
    return out


def wp7():
    """WP-7: Ĝ(a) as rigorous enclosures (author's decision 2026-09-24, Block 2 option (a))."""
    g = pd.read_csv(RESULTS / "ghat_rigorous.csv", float_precision="round_trip")
    ds = pd.read_csv(RESULTS / "ghat_digit_stability.csv").iloc[0]
    e = lambda v: f"{v:+.1e}"
    rows = "".join(f"| {r.a:.2f} | [{r.Ghat_cert_rigorous:.12f}, {r.Ghat_hi_rigorous:.12f}] | {r.Ghat_cert_published:.12f} | "
                   f"{e(r.delta_cert)} | {e(r.delta_hi)} | {r.bnb_arg_lower:.12f} | {int(r.leaves):,} |\n"
                   for r in g.itertuples())
    return f"""
## WP-7. Ĝ(a) as rigorous enclosures (replaces the float values; Block 2)

Source: `ghat_rigorous.csv` (producer `src/ghat_rigorous.py`, from the independent Arb checker's results in
`certificate_checks/`). **Every R now uses the rigorous lower end** `Ĝ_cert` below: the checker's lower bound on G at
the same witness point as the published float value, rounded down to a double. The upper end is the checker's bound
over every leaf of the branch and bound, rounded up. The domain reduction to w₁ ∈ (0, a/1.4] is analytic.

| a | Ĝ(a) ∈ [Ĝ_cert, Ĝ_hi] (rigorous) | old float Ĝ_cert | change of Ĝ_cert | change of Ĝ_hi | larger proven lower bound (not adopted) | leaves |
|---|---|---|---|---|---|---|
{rows}
**No printed digit of Ĝ or R changes.** The largest relative change of Ĝ is δ = {ds.delta:.1e}. Every R is linear in Ĝ.
All {int(ds.printed_number_checks)} printed-number checks of the ledger (`src/verify_ledger.py`, which verifies every
printed number against its artifact) still hold, and round to the same printed digits, with their artifact value
scaled by 1 ± δ (conservatively applied to every check, Ĝ-dependent or not); {int(ds.unstable)} are unstable
(`ghat_digit_stability.csv`). A further {int(ds.machine_precision_checks)} checks compare two artifacts to 1e−12; they are
not printed numbers. {int(ds.machine_precision_moved)} of them move in their last digits under the blanket δ, and none of
those depends on the replaced Ĝ(a) (`ghat_digit_stability_machine_precision.csv`: A* is a limit constant; WP-1's P1 and
P3 use the own-seed windows' Ĝ, which is not replaced). That separation was set after the first run flagged them. The old float endpoints are kept in the table; the strict
checks on them keep reporting that they are not rigorous in the last one or two ulps.

Artifacts computed before the switch (stored R columns) keep the old float Ĝ; they differ by at most δ relative, which
the check above covers. Every code path that computes R now reads `ghat_rigorous.ghat_R`.

The "larger proven lower bound" column is the rigorous value at the branch and bound's own attained point. At nine a it
exceeds Ĝ_cert, by up to {float(((g.bnb_arg_lower - g.Ghat_cert_rigorous) / g.Ghat_cert_rigorous).max()):.1e} relative. It is a valid lower bound but a different number from the one R has
always used, so it is not adopted.
"""


def wp6():
    """WP-6: crossing detection and the residual (post hoc audit, author's decision 2026-09-24)."""
    sm = pd.read_csv(RESULTS / "crossing_audit_summary.csv")
    cs = pd.read_csv(RESULTS / "cadence_sensitivity.csv")
    pct = lambda v: f"{100 * v:.1f}%"
    pp = lambda v: f"{100 * v:.2f}%"

    def fam(prefix, a, col):
        return float(sm[sm.family.str.startswith(prefix) & (sm.a.round(2) == a)][col].iloc[0])

    def q(exp, a, quantity, col):
        g = cs[(cs.experiment == exp) & (cs.quantity == quantity)]
        if a is not None:
            g = g[g.a.round(2) == a]
        return float(g[col].iloc[0])
    every = sm[sm.check_interval == 1]
    own = {a: (q("own-seed", a, "median residual R_cross / U_own - 1", "check_based"),
               q("own-seed", a, "median residual R_cross / U_own - 1", "interpolated")) for a in (1.3, 1.5)}
    b3 = {a: (q("Block 3", a, "median over settings of (median R / U) - 1", "check_based"),
              q("Block 3", a, "median over settings of (median R / U) - 1", "interpolated")) for a in (1.3, 1.5)}
    lam = {a: (q("Block G base (lambda)", a, "lambda(a)", "check_based"),
               q("Block G base (lambda)", a, "lambda(a)", "interpolated")) for a in (1.3, 1.5)}
    b2 = (q("Block G all windows (B2)", None, "B2 pooled median crossing R", "check_based"),
          q("Block G all windows (B2)", None, "B2 pooled median crossing R", "interpolated"))
    s3 = {a: (fam("phase 2b", a, "median_residual_check"), fam("phase 2b", a, "median_residual_interp_check")) for a in (1.3, 1.5)}
    sz = {a: (fam("size test", a, "median_residual_check"), fam("size test", a, "median_residual_interp_check")) for a in (1.3, 1.5)}
    o = cs[(cs.experiment == "own-seed") & cs.quantity.str.endswith("[registered predictions]")]
    b = cs[(cs.experiment == "Block 3") & cs.quantity.str.contains("ci95_hi")]
    rows_o = "".join(f"| own-seed {r.quantity.split()[0]}, a = {r.a:.2f} | {r.check_based:.4f} [{r.check_lo:.4f}, {r.check_hi:.4f}] "
                     f"| {r.interpolated:.4f} [{r.interp_lo:.4f}, {r.interp_hi:.4f}] | {'pass' if r.check_pass else 'FAIL'} / "
                     f"{'pass' if r.interp_pass else 'FAIL'} |\n" for r in o.itertuples())
    rows_b = "".join(f"| Block 3 {r.quantity.split(' ci95')[0]}, {r.quantity.split('[')[1].rstrip(']')} | upper {r.check_based:.4f} "
                     f"| upper {r.interpolated:.4f} | {'excludes 0' if r.check_pass else 'includes 0'} / "
                     f"{'excludes 0' if r.interp_pass else 'includes 0'} |\n" for r in b.itertuples())
    return f"""
## WP-6. Crossing detection and the residual (post hoc audit; no registered verdict changes)

Sources: `crossing_audit.md`, `crossing_audit_summary.csv`, `cadence_sensitivity.csv` (producers `src/crossing_audit.py`,
`src/cadence_sensitivity.py`). Every rerun reproduced its stored crossing step and |w₂| bit for bit.

**Check interval, stated beside each number.** Phase 2b (S3), the size test, both lag tests (every φ arm; the interval
did not scale with the budget, which was 32,000/φ) and Block 4b check placement **every step**. Block 3, the prospective
own-seed test and the Block G runs behind λ, B1 and B2 check **every 50 steps**.

**Every-step experiments: the residual is unaffected by detection.** The upward bias of the recorded crossing is at most
one step's growth of R: median {pp(every.growth_over_interval_rel_median.min())}–{pp(every.growth_over_interval_rel_median.max())}
of the threshold, at most {pp(every.growth_over_interval_rel_max.max())}. The residual against the run's own threshold
moves by at most 0.09 points under interpolation: S3 {pct(s3[1.3][0])} → {pct(s3[1.3][1])} (a = 1.30) and
{pct(s3[1.5][0])} → {pct(s3[1.5][1])} (a = 1.50); the size test (the lag tests' φ = 1 arms and 4b's control)
{pct(sz[1.3][0])} → {pct(sz[1.3][1])} and {pct(sz[1.5][0])} → {pct(sz[1.5][1])}. Detection spacing is ruled out as the
cause of the every-step residual.

**50-step experiments: detection overstates the residual; interpolated values as a labelled sensitivity analysis.**
Crossing located by linear interpolation of G between the last negative and the first positive 50-step check (all
crossing runs):

| quantity (check interval 50 steps) | check-based | interpolated (post hoc) |
|---|---|---|
| own-seed: median R_cross/U_own − 1, a = 1.30 (n = 460) | {pct(own[1.3][0])} | {pct(own[1.3][1])} |
| own-seed: median R_cross/U_own − 1, a = 1.50 (n = 460) | {pct(own[1.5][0])} | {pct(own[1.5][1])} |
| Block 3: median over settings of (median R/U) − 1, a = 1.30 | {pct(b3[1.3][0])} | {pct(b3[1.3][1])} |
| Block 3: median over settings of (median R/U) − 1, a = 1.50 | {pct(b3[1.5][0])} | {pct(b3[1.5][1])} |
| λ(1.30) (Block G base) | {lam[1.3][0]:.4f} | {lam[1.3][1]:.4f} |
| λ(1.50) (Block G base) | {lam[1.5][0]:.4f} | {lam[1.5][1]:.4f} |
| B2 pooled median crossing R (five Block G windows) | {b2[0]:.4f} | {b2[1]:.4f} |

**The residual statement (narrower form, finalised on the full rerun).** Against the run's own threshold, the residual
is about 3% at a = 1.30 and about 6–6.5% at a = 1.50 whether placement is checked every step (S3 {pct(s3[1.3][1])} and
{pct(s3[1.5][1])}) or every 50 steps with the crossing interpolated (own-seed {pct(own[1.3][1])} and {pct(own[1.5][1])}).
Against a population threshold it stays larger after interpolation (Block 3 against U: {pct(b3[1.3][1])} and
{pct(b3[1.5][1])}), so that excess is not a detection effect. The check-spacing explanation is ruled out for the every-step residual; in the 50-step
experiments detection adds about {100 * (own[1.3][0] - own[1.3][1]):.1f} (a = 1.30) and {100 * (own[1.5][0] - own[1.5][1]):.1f}
(a = 1.50) points on top of it.

**Registered comparisons under interpolation (sensitivity; the registered verdicts stand as scored).** Own-seed
criteria as registered (statistic [95% interval]; pass/fail check-based / interpolated):

| criterion | check-based | interpolated | verdict check / interpolated |
|---|---|---|---|
{rows_o}
For P4 the bracket is the registered acceptance range, not an interval. At a = 1.50 the P2b reading "C better than U_own (interval above 0)" holds check-based but not interpolated.

Block 3, C against each baseline (upper end of the 95% interval of the |log error| difference; C better if < 0):

| comparison | check-based | interpolated | check / interpolated |
|---|---|---|---|
{rows_b}
With the registered predictions (built from 50-step data) and interpolated observations the detection is mismatched;
with C, B1 and B2 rebuilt from interpolated Block G crossings (cadence-matched) every registered comparison keeps its
sign and excludes 0.
"""



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        print(render())
    else:
        main()
        print(render())
