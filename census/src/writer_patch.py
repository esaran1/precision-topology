"""Producers for WRITER_INPUTS_v4_patch.md (completion inputs for the submission).  Every table here is derived
from committed artifacts with the registered definitions (imported from the scorers, not re-implemented).

    python -m src.writer_patch            # all five tables
"""

from __future__ import annotations

import json

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
- **Total**: {int(h.n) + int(p_.n)} registered predictions.

**Added in the 2026-09-25 round** (registrations through the current commit):
- Block 4b (`residual_mechanism_design.md`): the two competing hypotheses, inherited displacement and optimiser memory,
  are FAIL at both a (neither intervention removes half the residual; the registered competing outcome holds). Its
  four validity gates pass; teleport + reset is reported with no criterion.
- Width-1 consistency check (`scale_limits_prediction.md`): PASS.
- tanh (`scale_limits_tanh_prediction.md`): FAIL. The registered expectation was not met as written ("neither", by
  the author's decision; the E-stall convention).

**Not in the headline.**
- *Registered decision rule (outcome, not a prediction)*: the width-2 verdict (`scale_limits_prediction.md`). Its
  registration stated a two-outcome rule (G > 0: no placement threshold; G <= 0: a threshold exists) rather than a
  predicted outcome. **Outcome: no placement threshold for f_a at width 2.** The outcome is decided by the
  second-order (Var) selection, which the registration added; its direct check confirms it (below).
- *Registered check, scored*: the width-2 direct check. All 8 scales are placed (the cancelling pair), so the decision
  rule's outcome is confirmed directly (2026-09-25).
- *Pending (registered, not yet scored)*: the width-2 no-gating test (not run).
- *Designed but never registered*: W0–W4 and the tanh criterion of `width2_design.md`. The design registers them
  with frozen thresholds at its step 5, which the small-scale verdict made moot. They are not applicable at width 2.

**Row-level tables shipped in the supplementary**:
- `results/registration_census.csv`: the headline, one row per prediction;
- `results/registration_census_by_unit.csv`: the appendix view, per a;
- `results/registration_census_v4_gates_and_reported.csv`: validity gates, no-criterion items, the registered
  decision rule, pending registrations and designed-but-unregistered predictions;
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
    text += wp8()
    text += wp9() + wp10() + wp11() + wp12() + wp13() + wp14()
    text += wp15() + wp16() + wp17() + wp18() + wp19() + wp20() + wp21() + wp22() + wp23()
    text += wp24() + wp25() + wp26() + wp27() + wp28() + wp30() + wp31() + wp32() + wp33() + wp34() + wp35() + wp29()
    out = RESULTS.parent / "paper" / "WRITER_INPUTS_v4_patch.md"
    out.write_text(text)
    return out


def _id(*labels):
    return "; ".join(f"`{l}`" for l in labels)


def wp9():
    """WP-9: width 2 (rebuttal revision)."""
    w1 = pd.read_csv(RESULTS / "scale_limits_width1.csv")
    w2 = pd.read_csv(RESULTS / "scale_limits_width2.csv").set_index("a")
    sm = [pd.read_csv(f) for f in sorted((RESULTS / "width2_w0_parts").glob("smallscale_f*.csv"))]
    landed = pd.concat(sm) if sm else pd.DataFrame(columns=["act", "R2"])       # the direct check's own results
    land = ", ".join(f"a = {float(r.act[1:]):.2f} at R₂ = {r.R2:g}" for r in landed.itertuples()) or "none yet"
    n_pend = 8 - len(landed)
    return f"""
## WP-9. Width 2: the small-scale criterion, the registered verdict and the tanh case (rebuttal revision)

Sources: `math_note_v2.md` §10 and §10.1 (at `census/results/math_note_v2.md`), `scale_limits_prediction.md`,
`scale_limits_tanh_prediction.md`, WP-8. IDs are `src/verify_ledger.py` check labels.

**The small-scale criterion (math note §10.1, summary).** For fixed hidden parameters θ, the profiled loss at output
scale s is L*(θ; s) = log 2 − (s/4)·Δμ(θ) + (s²/8)·Var(φ_θ) + O(s⁴) (g‴(0) = 0; remainder checked numerically). So the
small-scale conditional minimiser maximises the class-mean gap Δμ, and among tied maximisers the s² term selects the
one with the smallest Var(φ). If the data gap Γ_n is attained (Lemma 2), output scale gates placement if and only if
this small-scale minimiser is unplaced.

**Width-1 consistency check (registered, passed).** The selected Δμ-maximiser has G ≤ 0 at all six a
(G = {w1.G.max():.3f} … {w1.G.min():.3f}), as required by the certified width-1 thresholds.
IDs: {_id("alpha*", "width 1: selected dmu-maximiser G <= 0 at all six a", "width 1: G range lo (a = 1.30)", "width 1: G range hi (a = 1.60)")}.

**The registered width-2 verdict: no placement threshold for f_a.** The Var-selected Δμ-maximiser is the cancelling
cosine pair, placed with G = +{w2.loc[1.3, "selected_G_lo"]:.3f} (a = 1.30) and +{w2.loc[1.5, "selected_G_lo"]:.3f} (a = 1.50);
W1 and W4 were not run. IDs: {_id("width 2: validation (ladder, independent, pair attains max, pair Var minimal)", "width 2: selected G at a=1.30", "width 2: selected G at a=1.50", "verdict: no placement threshold for f_a at width 2")}.

**Its dependency on the variance selection.** At first order the Δμ-maximisers tie: single units (unplaced) and
pairs (placed). Δμ alone does not decide the verdict; the s² (Var) term does. That selection rule was added in the
registration (committed before any maximiser was computed) and was not in the author's original rule. The direct
check (WP-8) tests the verdict without the expansion: at each scale the validated W0 search finds the conditional
minimiser itself. {("Complete: all 8 scales" if n_pend == 0 else "Landed so far: " + land)} (placed cancelling pair at each,
every validation passed, the best unplaced configuration on the G = 0 boundary, at least 99× the tie tolerance above)
{"— the verdict is confirmed directly" if n_pend == 0 else f"; **{n_pend} scale(s) PENDING**"} (table in WP-8, not repeated
here). IDs: {_id("direct check: every landed scale placed, cancelling pair", "direct check: no finished restart below the retained minimiser (landed scales)", "min margin at R2 = 0.001", "all best-unplaced on the boundary")}.

**Coverage of the width-2 threshold scan (W0; never registered).** At a = 1.30 no R₂ value was completed: the scan
never started, because its Γ̂₂ stop fired first. At a = 1.50 the scan completed R₂ = 0.02, 0.03, …, 0.10 (9 of 99
planned points; 2,000 restarts each, audit passed, placed at every point), not validated in the design's sense
(validation runs only near a threshold, and there was none), and was terminated. The validated small-scale results
are the direct check's 8 scales (WP-8). Source: `width2_design.md`, "W0 threshold-scan coverage, as run".

**tanh: registered outcome "neither".** Every registered expectation held except one: validation passed at every box
size, the maxima are < 1 and rising, the first-order tie is present, the Var-selected symmetric pair is placed at every
box (G₊ → 1) and the single units are unplaced. The boundary criterion as written fails at A = 40, because the 1e−9 tie
tolerance is of the order of 1 − max (4.1e−9) there. Non-attainment of the supremum is proved analytically
(Δμ < 1 = sup Δμ). No re-registration. tanh lies outside the criterion's attainment hypothesis.
IDs: {_id("tanh: registered outcome 'neither'", "tanh: validation passed at every A", "tanh: box maxima < 1 and rising", "tanh: boundary condition fails only at A = 40", "tanh: selected member is the symmetric pair at every A", "tanh: single units unplaced at every A")}.

**Say**
- "At width 2, output scale does not gate the placement of the conditional minimiser for f_a: the registered
  small-scale prediction selects the placed cancelling pair, and a direct search at small scales finds that pair as the
  conditional minimiser. Training at small held scale is nevertheless gated (WP-20)."
- "The prediction rests on a second-order selection: at first order, unplaced single units and placed pairs tie."
- "For tanh the supremum of Δμ is not attained; the registered test returned 'neither' because its boundary criterion
  failed at the largest box, although every other expectation held."

**Do not say**
- "proved" or "certified" for the width-2 verdict (it is a registered, validated computation plus a lemma, not a
  certificate); or that Δμ alone decides it.
- that the direct check is complete while any scale is PENDING (it completed on 2026-09-25: all 8 scales placed).
- that the unplaced region has no local minima, or that the boundary point is a competing minimum (WP-8 wording).
- that tanh "confirmed" or "passed" the registered expectation.
- that output scale does not gate width-2 *training*: the registered training test found gating (WP-20).
"""


def wp10():
    """WP-10: the residual mechanism."""
    pc = pd.read_csv(RESULTS / "residual_posthoc_correlations.csv")
    r = lambda a, f: float(pc[(pc.a.round(2) == a) & (pc.point == "primary") & (pc.feature == f)].spearman_rho.iloc[0])
    rm = pd.read_csv(RESULTS / "residual_mechanism_scores.csv")
    m = lambda a, arm: 100 * float(rm[(rm.a.round(2) == a) & (rm.arm == arm)].median_residual.iloc[0])
    ci = lambda a, arm: rm[(rm.a.round(2) == a) & (rm.arm == arm)].iloc[0]
    rows = ""
    for a in (1.3, 1.5):
        cells = [f"{m(a, 'control'):.3f}%"]
        for arm in ("teleport", "reset", "teleport_reset"):
            c = ci(a, arm)
            cells.append(f"{m(a, arm):.3f}% [{100 * c.control_minus_arm_ci_lo:+.2f}, {100 * c.control_minus_arm_ci_hi:+.2f}]")
        rows += f"| {a:.2f} | " + " | ".join(cells) + " |\n"
    return f"""
## WP-10. The residual mechanism: Block 4a (post hoc) and Block 4b (registered)

Sources: `residual_posthoc_correlations.csv`, `residual_mechanism_design.md`, `residual_mechanism_scores.csv`.

**Block 4a (POST HOC correlations).** At the deconfounded switch point (0.7× the own threshold), the final residual
correlates with the displacement from the occupied branch's conditional minimiser (Spearman ρ = {r(1.3, "dist"):.2f} at
a = 1.30, {r(1.5, "dist"):.2f} at a = 1.50; n = 48 each; permutation p ≈ 1e−4) and anti-correlates with Adam's
√v̂(w₂) (ρ = {r(1.3, "sqrt_vhat_w2"):.2f}, {r(1.5, "sqrt_vhat_w2"):.2f}).
IDs: {_id("4a a=1.3 displacement rho (post hoc)", "4a a=1.5 displacement rho (post hoc)", "4a a=1.3 sqrt vhat(w2) rho (post hoc)", "4a a=1.5 sqrt vhat(w2) rho (post hoc)")}.

**Block 4b (registered): competing outcome at both a — neither teleporting to the branch minimiser nor resetting
Adam's moments removes half the residual; the mechanism is unresolved.** Every validity check passed (control
bit-identical to the lag test's φ = 1 continuation, branch preserved, teleports on the minimiser, 0 excluded, 48/48
crossings per arm). Median residual against the own threshold [95% interval of control − arm]:

| a | control | teleport | reset | teleport + reset (reported, no criterion) |
|---|---|---|---|---|
{rows}
IDs: {_id("4b: every validity check passed", "4b a=1.3: verdict competing (neither removes half)", "4b a=1.5: verdict competing (neither removes half)", "4b a=1.3 control residual", "4b a=1.3 teleport residual", "4b a=1.5 reset residual", "4b a=1.3 teleport+reset residual", "4b a=1.5 teleport+reset residual")}.
Detection does not explain the residual either (WP-6).

**Say**
- "Post hoc, the residual correlates with the run's displacement from its branch minimiser at the switch point; a
  registered intervention that removes the displacement does not remove the residual, so the correlation is not causal
  there. Neither inherited displacement nor optimiser memory alone accounts for it; the mechanism is open."

**Do not say**
- that displacement, optimiser memory or adiabatic lag causes or explains the residual;
- that Block 4a's correlations are registered, or that 4b found a small effect of any arm (every interval includes 0).

**Validity note (found 2026-09-25 while replaying these runs; no verdict changed).** In the registered Block 4b run, the
arms shared optimiser-state tensors, and PyTorch's `load_state_dict` does not copy them.
- The control arm's continuation therefore mutated the stored Adam state in place. The **teleport** arm (registered as
  "Adam state kept") started from the control run's *end-of-run* Adam state.
- Likewise, the **teleport_reset** arm started from the reset run's end state rather than zeroed moments.
- Control and reset are unaffected. Replaying in the original order reproduces every recorded crossing (WP-23).
- **Corrected rerun (author's instruction; `residual_mechanism_corrected`, 1838f21).** Both teleport arms were rerun
  from fresh deep copies of each checkpoint, with each start state verified bit for bit.
  - Under the registered criteria, **4b-ID and 4b-OM still FAIL at both a.** No verdict changes.
  - Teleport median residual: 0.03066 (a = 1.30) and 0.06504 (a = 1.50), against as-run 0.03079 and 0.06547.
  - Teleport_reset: 0.03130 and 0.06542, against as-run 0.03122 and 0.06601.
- **Audit of every multi-arm experiment that restarts from saved optimiser state** (`state_audit`). Each arm's actual
  starting parameters and Adam state were compared bit for bit with a fresh load of its checkpoint, with the arms run
  in their original order.
  - Clean: Blocks 4 and 5 (128 arms), the horizon extension (40), both lag-test-2 rules (32) and the no-gating
    replays (48).
  - Only Block 4b's registered teleport arms are flagged. This was the audit's positive control, and the corrected
    arms are clean.
- **Say:** "A shared-state bug affected two arms of one block. A corrected rerun leaves its registered verdicts
  unchanged, and an audit of every other multi-arm experiment found no contamination."
IDs: {_id("Block 4b corrected ID FAIL", "Block 4b corrected OM FAIL", "State audit clean arms", "State audit positive control")}.
"""


def wp11():
    """WP-11: the independent certificate checker (Block 2)."""
    gr = pd.read_csv(RESULTS / "ghat_rigorous.csv")
    return f"""
## WP-11. The independent certificate checker: what is verified, what is pending, what ships

Sources: `certificate_audit.md`, `verify_certificates_finite.log`, `verify_certificates_ghat.log`,
`certificate_checks/`, `certificates_manifest.csv`. Checker: `src/verify_certificates.py` (python-flint / Arb, 80-bit
ball arithmetic; imports nothing from `src/`; every decision an Arb comparison).

**Verified so far.**
- **Finite-a status certificates**, both ends of the global bracket at a = 1.30, 1.35, 1.40, 1.45, 1.50, 1.60 (12
  certificates): every leaf's claim verified, exact tiling, the localisation lemma W recomputed, hashes match. All pass.
  ID: `Block 2: 12 finite-a certificates pass`.
- **Ĝ(a) enclosures**, a = {", ".join(f"{a:.2f}" for a in gr.a)} ({len(gr)} certificates, 42 thousand to 111 million leaves):
  hashes, exact tiling of the domain, and the analytic domain reduction pass at every a. The float endpoints of the
  original searches miss the rigorous enclosure by at most 1.1e−15, so the rigorous enclosures replace them (WP-7).
  IDs: `Ghat certificates checked (a = 1.02-3.0)`; `Ghat: structure passes at every a (hashes, tiling, domain, Ghat_cert <= hi)`.

- **Limit switch**: the status certificates at both ends of the search-certified bracket, A = 0.68125 ("minus") and
  A = 0.6875 ("plus"), both pass. So **A* ∈ [0.68125, 0.6875] is independently verified**, the same bracket as the
  search's. The checks cover every losing-region leaf (outside its region, or loss above U), exact tiling, data
  symmetry, and the localisation bounds B(24) and B_full, recomputed by exact PAVA (so the localisation is verified
  too). IDs: `limit check A_lo passes`; `limit check A_hi passes`.

**Pending (not yet exported or checked):** K = sup G₀ (its domain lemma is written, math note §8), the Krawczyk boxes
for c₁, the solve brackets (finite and limit), outer exclusion and ring, and the PD boxes.

**K's domain (lemma written and checked, 2026-09-24).** K's branch and bound searches (u, v) ∈ [0, 8] × [−12, 12].
The domain lemma (math note §8, "Domain lemma for K") proves G₀(u, v) ≤ 0 whenever |u| ≥ √(50/3) ≈ 4.08 or
|v| ≥ √2 + 0.6|u|. So the supremum lies inside the box, and K is the supremum over the whole plane. The lemma is an exact
algebraic argument, checked numerically: the identity holds to 1.4e−12, and G₀ ≤ 0 at 4.0 million excluded points
(`k_domain_check.csv`). The Arb check of K's branch and bound over the box is still pending, as listed above.
IDs: `K domain: G0 <= 0 on the excluded region (u > 0 max)`; `K domain: region inside the certified box`;
`K domain: K argmax inside the region`.

**What the supplementary will ship** (not rebuilt yet; built once every claim is final): the checker
(`src/verify_certificates.py`) and its tests; the manifest of SHA-256 hashes, each with the command that regenerates
the file bit-identically (`certificates_manifest.csv`); the exporting searches and `src/cert_export.py`. The certificate
data files are not shipped (regenerated from the commands and verified against the hashes); a single command runs the
checker, with its expected output.

**Say**
- "An independent checker in ball arithmetic, sharing no code with the searches, verifies the finite-a certificates,
  the Ĝ enclosures and the limit-switch bracket; the remaining certificate families are being exported."

**Say** (the sharp limit threshold)
- "K = sup G₀ is certified by branch and bound over a box, and a domain lemma shows the supremum lies inside it."

**Do not say**
- that K (or the sharp limit threshold built on it) has been independently checked in Arb, until that check is done;
- that every certificate has been independently checked, until the pending families are;
- that the checker verified the original float endpoints of Ĝ (it verified rigorous enclosures that differ from them
  by at most 1.1e−15).
"""


def wp8():
    """WP-8: the small-scale selection resolved against the unplaced region (wording approved by the author 2026-09-24)."""
    lm = pd.read_csv(RESULTS / "width2_unplaced_localmin.csv")
    un = pd.read_csv(RESULTS / "width2_unplaced.csv")
    sm = []
    for act in ("f1.30", "f1.50"):
        f = RESULTS / "width2_w0_parts" / f"smallscale_{act}.csv"
        if f.exists():
            sm.append(pd.read_csv(f, float_precision="round_trip"))
    sm = pd.concat(sm) if sm else pd.DataFrame(columns=["act", "R2", "retained_loss", "placed", "is_cancelling_pair"])
    rows = ""
    for r in lm.merge(un, on=["act", "R2"]).itertuples():
        d = sm[(sm.act == r.act) & (np.isclose(sm.R2, r.R2))]
        if len(d):
            q = d.iloc[0]
            placed_l = q.retained_loss
            chk = (f"{placed_l:.13f} ({'placed' if q.placed else 'UNPLACED'}, "
                   f"{'cancelling pair' if q.is_cancelling_pair else 'not the pair'})")
            diff = r.best_unplaced_loss - placed_l
            dd = f"{diff:.2e} ({diff / 1e-9:,.0f}× 1e−9)"
        else:
            chk, dd = "PENDING", "PENDING"
        rows += (f"| a = {float(r.act[1:]):.2f}, R₂ = {r.R2:g} | {chk} | {r.best_unplaced_loss:.13f} (boundary, G₊ = {r.G_hi:.1e}) "
                 f"| {dd} | {r.lowest_local_min_minus_placed:.2e} (single unit, G₊ = {r.lowest_local_min_G:.2f}) |\n")
    return f"""
## WP-8. The small-scale selection, resolved against the unplaced region (direct check; rebuttal revision)

Sources: `width2_unplaced.csv`, `width2_unplaced_localmin.csv` (producer `src/width2_unplaced.py`), the direct check
`width2_w0 smallscale` (validated W0 search, 4,000 restarts per scale).

At these scales the unplaced region contains local minima: single units at α* with an idle second unit (G₊ ≈ −3.3 to
−3.7), whose loss exceeds the placed cancelling pair's by the predicted single-unit second-order gap (s²/8)·α*²·Var(x)
to within 0.1–2%. The infimum over the unplaced region is lower: it lies on the placement boundary G₊ = 0, at a
partially cancelling pair (|c| ≈ 0.43–0.51). Descent from there reaches the placed cancelling pair, and the loss drop
matches the second-order prediction (s²/8)·c²·Var(x) to within 0.2–3%. In every search the placed minimiser lies below
every unplaced configuration, by at least 9.9e−8 at R₂ = 0.001 (about 99× the 1e−9 tie tolerance).

| scale | best placed: the direct check's retained minimiser | best unplaced (infimum over G₊ ≤ 0) | difference (× tie tolerance) | lowest unplaced local minimum, above the placed pair |
|---|---|---|---|---|
{rows}"""


def wp7():
    """WP-7: Ĝ(a) as rigorous enclosures (author's decision 2026-09-24, Block 2 option (a))."""
    g = pd.read_csv(RESULTS / "ghat_rigorous.csv", float_precision="round_trip")
    ds = pd.read_csv(RESULTS / "ghat_digit_stability.csv").iloc[0]
    e = lambda v: f"{v:+.1e}"
    rows = "".join(f"| {r.a:.2f} | {r.Ghat_cert_rigorous:.12f} | {r.Ghat_cert_published:.12f} | "
                   f"{e(r.delta_cert)} | {int(r.leaves):,} |\n" for r in g.itertuples())
    g["best_lo"] = np.maximum(g.Ghat_cert_rigorous, g.bnb_arg_lower)
    g["d_abs"] = g.best_lo - g.Ghat_cert_rigorous
    g["d_rel"] = g.d_abs / g.Ghat_cert_rigorous
    rows_g = "".join(f"| {r.a:.2f} | [{r.best_lo:.12f}, {r.Ghat_hi_rigorous:.12f}] | "
                     f"{'branch and bound point' if r.bnb_arg_lower > r.Ghat_cert_rigorous else 'Ĝ_cert witness'} | "
                     f"{r.d_abs:.2e} | {r.d_rel:.2e} | {e(r.delta_hi)} |\n" for r in g.itertuples())
    imax = int(g.d_rel.idxmax()); jmax = int(g.d_abs.idxmax())
    return f"""
## WP-7. Ĝ(a) as rigorous enclosures (replaces the float values; Block 2)

Source: `ghat_rigorous.csv` (producer `src/ghat_rigorous.py`, from the independent Arb checker's results in
`certificate_checks/`). **Every R now uses the rigorous lower end** `Ĝ_cert` below: the checker's lower bound on G at
the same witness point as the published float value, rounded down to a double. The upper end is the checker's bound
over every leaf of the branch and bound, rounded up. The domain reduction to w₁ ∈ (0, a/1.4] is analytic.

**(i) Ĝ_cert, the value every R uses** (the rigorous value at the same witness as the old float value):

| a | Ĝ_cert (rigorous, rounded down) | old float Ĝ_cert | change | leaves checked |
|---|---|---|---|---|
{rows}
**(ii) The enclosure of the supremum G* = Ĝ(a), reported separately.** Lower end: the best certified lower bound at each
a, the larger of the rigorous values at the two recorded attained points (Ĝ_cert's witness and the branch and bound's
own point). Upper end: the rigorous leaf bound, rounded up. The difference between the best certified lower bound and
Ĝ_cert is given **both absolute and relative**. The largest relative difference is {g.d_rel[imax]:.2e} (at
a = {g.a[imax]:.2f}, where it is {g.d_abs[imax]:.2e} absolute); the "up to 8.4e−5" is this relative figure. The largest
absolute difference is {g.d_abs[jmax]:.2e} (at a = {g.a[jmax]:.2f}, {g.d_rel[jmax]:.2e} relative).

| a | G* ∈ [best certified lower, upper] | lower end from | best lower − Ĝ_cert (absolute) | (relative) | change of upper end vs old float |
|---|---|---|---|---|---|
{rows_g}
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

R keeps Ĝ_cert (table (i)); the best certified lower bound in table (ii) is a valid bound on G* but a different number
from the one R has always used (author's decision 2026-09-24).
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
    p2 = o[o.quantity.str.startswith("P2b")].iloc[0]
    p2b_c = f"{p2.check_based:+.3f} [{p2.check_lo:+.3f}, {p2.check_hi:+.3f}]"
    p2b_i = f"{p2.interpolated:+.3f} [{p2.interp_lo:+.3f}, {p2.interp_hi:+.3f}]"
    rows_o = "".join(f"| own-seed {r.quantity.split()[0]}, a = {r.a:.2f} | {r.check_based:.4f} [{r.check_lo:.4f}, {r.check_hi:.4f}] "
                     f"| {r.interpolated:.4f} [{r.interp_lo:.4f}, {r.interp_hi:.4f}] | {'pass' if r.check_pass else 'FAIL'} / "
                     f"{'pass' if r.interp_pass else 'FAIL'} |\n" for r in o.itertuples())
    def _b3(comp, lab):
        r = b[b.quantity == f"{comp} ci95_hi [{lab}]"].iloc[0]
        return r
    rows_b = ""
    for comp in ("C - B1", "C - B2", "C - U"):
        full, mix = _b3(comp, "cadence-matched C, B1 and B2"), _b3(comp, "registered predictions")
        ex = lambda ok: "excludes 0" if ok else "includes 0"
        rows_b += (f"| {comp} | {full.check_based:+.4f} ({ex(full.check_pass)}) | {full.interpolated:+.4f} "
                   f"({ex(full.interp_pass)}) | {mix.interpolated:+.4f} ({ex(mix.interp_pass)}) |\n")
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
For P4 the bracket is the registered acceptance range, not an interval.

**Own-seed at a = 1.50.** P2b passes under both definitions. The stronger reading, that C beats U_own at a = 1.50
(interval above 0), holds check-based ({p2b_c}) but does **not** survive interpolation ({p2b_i}).
**Flag for softening:** the draft's "C is better, as registered" (`WRITER_INPUTS_v4.md` line 13, "P2b: C better at 1.50,
as registered", and line 662, "C is better, +0.023 [+0.009, +0.036]") should say that P2b passes, and that C's advantage
at a = 1.50 depends on the detection rule (check-based +0.023 [+0.009, +0.036]; interpolated {p2b_i}).

**Block 3.** The sensitivity analysis is the **fully interpolated** version: calibration (λ, hence C; B1; B2, all from
the Block G runs) and observations both use interpolated crossings, so one detection rule applies throughout. The
observations-only version is reported beside it, **labelled as mixing detection rules** (predictions calibrated on
50-step checks, observations interpolated). Upper end of the 95% interval of the |log error| difference (C better if
< 0):

| comparison | check-based (as registered) | fully interpolated (sensitivity) | observations only (mixes detection rules) |
|---|---|---|---|
{rows_b}
In the fully interpolated version every registered comparison keeps its sign and excludes 0. Mixing the detection
rules makes C − B2 include 0; that version is shown for completeness, not as the analysis.
"""




def wp12():
    """WP-12: the mechanism stated analytically, the scaling law and Adam's per-step bound (harsh review A1-A2)."""
    m = pd.read_csv(RESULTS / "wp12_mechanism.csv").set_index("a")
    sc = pd.read_csv(RESULTS / "wp12_scaling.csv").set_index("a")
    ad = pd.read_csv(RESULTS / "wp12_adam_bound.csv").iloc[0]
    ob = pd.read_csv(RESULTS / "wp12_onset_bound.csv").set_index("budget")
    rows = "\n".join(f"| {a:.2f} | {r.s0:.3f} | {r.s1_data:.1f} | [{r.w2_glob_cert_lo:.4g}, {r.w2_glob_cert_hi:.4g}] |"
                     for a, r in m.iterrows() if np.isfinite(r.w2_glob_cert_lo))
    srows = "\n".join(f"| {a:.2f} | {r.w2_limit_krawczyk:.4g} | {r.w2_first_order:.4g} | "
                      + (f"[{r.w2_cert_lo:.4g}, {r.w2_cert_hi:.4g}] | {100 * r.rel_err_first_order_vs_cert_mid:+.1f}% |"
                         if np.isfinite(r.w2_cert_lo) else "not certified | — |")
                      for a, r in sc.iterrows() if a <= 1.6)
    return f"""
## WP-12. The mechanism, stated analytically; the scaling law; why Adam is delayed (for the submission)

Sources: math note §11 and §11.1 (`math_note_for_writer.md`); producer `src/harsh_review_a.py` → `wp12_identity.csv`,
`wp12_mechanism.csv`, `wp12_scaling.csv`, `wp12_onset_bound.csv`, `wp12_adam_bound.csv`. This answers the review's
point that the mechanism is in the appendix but not the main text. The recommended placement is a short main-text
proposition (Step 1, Step 2 and the Proposition) with the proof in the appendix.

**What the loss rewards (the mechanism in one paragraph).** At small output scale the profiled loss is
log 2 − (s/4)Δμ + O(s²), so it rewards the **class-mean gap** Δμ. At large scale it is controlled by e^(−sG/2), so it
rewards the **worst-case gap** G. The threshold R_glob is where the conditional minimiser switches from the first
kind of solution to the second.

**Step 1 (identity).** On windows symmetric about 0, the linear part of f_a contributes nothing to Δμ, and
Δμ = sign(w₂)·a·sin b₁·D(w₁) with D(α) = E_O cos αx − E_I cos αx. The identity is checked to
{pd.read_csv(RESULTS / "wp12_identity.csv").max_abs_err_data.max():.1e} on the data points and
{pd.read_csv(RESULTS / "wp12_identity.csv").max_abs_err_continuous.max():.1e} on the continuous windows. The maximiser is
|w₁| = α* = {m.alpha_star_data.iloc[0]:.7f} (data points; {m.alpha_star_cont.iloc[0]:.7f} on the continuous windows), with
sin b₁ = ±1, and it does not depend on a.
IDs: {_id("A1 identity: max abs err, data points", "A1 identity: max abs err, continuous windows", "A1 alpha* (data)", "A1 alpha* (continuous)")}.

**Step 2 (placement needs a small first-layer weight).** G ≤ 2a − 2.8|w₁|, using the endpoint pairs (−2.0, 0.8) and
(−0.8, 2.0). These four points are window endpoints and data points, so the bound holds for the continuous gap G and
the data gap G_n alike. Placement therefore needs |w₁| < a/1.4. Since α* > a/1.4 for every a < 1.4α* =
{m["a_limit_1.4_alpha_star"].iloc[0]:.5f}, the class-mean maximiser is unplaced there: G = {m.loc[1.3, "G_cont_hi_at_theta_star"]:.2f}
at a = 1.30 and {m.loc[1.5, "G_cont_hi_at_theta_star"]:.2f} at a = 1.50.
IDs: {_id("A1 1.4 alpha*", "A1 G at theta* (1.30)", "A1 G at theta* (1.50)")}.

**Proposition (an explicit analytic bracket for the switch, width 1).** For 1 < a < 1.4α* and s < s₀(a), every placed
parameter has a strictly higher profiled loss than the unplaced class-mean maximiser. For s > s₁(a), every unplaced
parameter has a strictly higher loss than Ĝ's placed witness. So the switch exists and lies in [s₀, s₁]. The proof
uses only convexity of the logistic loss, ℓ″ ≤ 1/4 and Step 2, plus four finite evaluations. No compactness and no
attainment are needed.

| a | s₀ | s₁ (data gap) | certified \\|w₂\\|_glob |
|---|---|---|---|
{rows}

Every certified threshold lies inside the bracket. The bracket is loose, by roughly 15× below and 30× above: it
proves the switch **exists**, and the certificates **locate** it. In R units the upper end is R₁ = log(n/log 2) =
{m.R1_data.iloc[0]:.2f} for every a.
IDs: {_id("A1 s0(1.30)", "A1 s0(1.50)", "A1 s1_data(1.30)", "A1 s1_data(1.50)", "A1 R1 = log(n/log 2)", "A1 every certified bracket inside [s0, s1]", "A1 bound check at s0 (1.30): L*(theta*) below placed lower bound")}.
The window-gap version of s₁ needs Ĝ > η (a Lipschitz correction). That holds at every tabulated a except 1.02
({_id("A1 window s1 defined for a >= 1.05")}).

**Width 2, in the same terms.** On symmetric windows, the second-order term selects the pair ṽ₁α₁ = −ṽ₂α₂. It
cancels the ramp and leaves a pure cosine, which is placed. So the class-mean maximiser is already placed and there
is no switch (WP-9). The same criterion predicts a switch on **asymmetric** windows, where the ramp enters Δμ. That
prediction was registered and passed: a validated width-2 threshold exists there. The registered training prediction
failed: training crosses at about 3× the threshold scale (WP-15).

**The scaling law (A2).** |w₂|_glob(a) = A*·ε^(−3/2)·(1 + c_s ε + O(ε²)), with ε = a − 1 and c_s := A′(0)/A* = 0.66215
(math note §8). A* ∈ [0.68125, 0.6875] is certified and independently checked (Krawczyk value 0.6854452).
**Notation:** c_s is the first-order coefficient in s-units. It is distinct from c₁ = 0.28523, the coefficient of R_glob
in R-units; the two differ by k₁ = −0.377, which comes from K(ε).

| a | limit A*ε^(−3/2) | with first-order term | certified \\|w₂\\|_glob | first order vs certified |
|---|---|---|---|---|
{srows}

IDs: {_id("A2 w2 first order (1.02)", "A2 w2 limit (1.02)", "A2 first order vs certified, max rel", "A2 first order vs certified, min rel")}.

**Adam's per-step bound (rigorous, for every gradient sequence).** Take torch Adam with lr = 0.01 and
(β₁, β₂) = (0.9, 0.999), bias-corrected. By Cauchy–Schwarz, every coordinate moves at most lr·B_t per step, with
B_t ≤ B_∞ = {ad.B_inf:.4f}. The bound is attained by a geometric gradient sequence. Summed, |w₂| ≤ {ad.reachable_2000:.2f}
after 2,000 steps from |w₂(0)| ≤ 1. At a = 1.02 the required scale is {sc.loc[1.02, "w2_first_order"]:.1f} (≥ {sc.loc[1.02, "w2_limit_lo"]:.1f}
even at A*'s certified lower end). So any Adam run needs **at least {int(sc.loc[1.02, "N_min_adam_worst_case"]):,} steps**, and
about {int(sc.loc[1.02, "N_lr_per_step"]):,} at the typical rate of lr per step. Observed at 2,000 steps: {int(ad.a102_solved)}/{int(ad.a102_runs)}
solves, with terminal |w₂| median {ad.a102_median_terminal_w2:.2f} and maximum {ad.a102_max_terminal_w2:.2f}.
IDs: {_id("A2 B_inf", "A2 reachable |w2| at 2000 steps", "A2 bound attained (t = 400)", "A2 N_min at 1.02", "A2 N at lr per step, 1.02", "A2 a=1.02: solved", "A2 a=1.02: median terminal |w2|", "A2 a=1.02: max terminal |w2|")}.

**What the law explains and what it does not.**
- **It explains** why the required scale diverges as ε^(−3/2), with a certified constant and first-order term (within
  3% of every certified finite-a threshold). It also shows that, with a bounded per-step move, the number of steps must
  diverge at least as fast.
- **It explains** the 0/200 at a = 1.02 in 2,000 steps. The threshold scale cannot be reached, whatever the gradients.
  This makes the 0/200 *consistent with and forced by* the threshold-crossing picture; it is not independent evidence
  for that picture.
- **It does not explain** Adam's actual growth. The worst-case bound is loose by one to two orders of magnitude. At
  2,000 steps it permits every a ≥ {ob.loc[2000, "a_min_worst_case"]:.3f}, but the observed onset is a = 1.60. At 128,000
  steps it permits a ≥ {ob.loc[128000, "a_min_worst_case"]:.4f}; observed 1.03. The observed onset follows the *measured*
  growth |w₂| ∝ B^α, α = 1.1172, and α's derivation failed (SGD). So −0.745 (registered) against −0.734 (measured)
  tests the derived ε^(−3/2) combined with a measured α; it is not a derivation of α.
- **It does not explain** the residual (why crossings sit above the conditional threshold).
IDs: {_id("A2 worst-case a_min at 2000", "A2 lr-rate a_min at 2000", "A2 worst-case a_min at 128000")}.

**Say:** "At small output scale the loss rewards the class-mean gap, whose maximiser is provably unplaced for
a < 2.51; at large scale it rewards the worst-case gap, whose maximiser is placed. The conditional threshold is where
the minimiser switches, and its existence follows analytically; its location is certified." / "Adam's per-step move
is bounded (attainably) by 7.27·lr, so at a = 1.02 the conditional threshold scale cannot be reached within 2,000
steps."

**Do not say:** "we derive Adam's growth rate" or "the scaling law predicts the onset" (α is measured). Do not say
"the per-step bound explains the delay quantitatively" (it is loose by one to two orders of magnitude). Do not say
"the proposition locates the threshold" (the bracket spans more than two decades). Do not say "the 0/200 at a = 1.02
confirms the mechanism" (it is forced by it).
"""


BIB_VERIFIED = r"""% verified: https://proceedings.neurips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html
@inproceedings{ahn2023threshold,
  title     = {Learning threshold neurons via edge of stability},
  author    = {Ahn, Kwangjun and Bubeck, Sebastien and Chewi, Sinho and Lee, Yin Tat and Suarez, Felipe and Zhang, Yi},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {19540--19569},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/075280-0858},
  year      = {2023}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/63ed15a46a143ff57484b38cd6b85d91-Abstract-Conference.html
@inproceedings{kumar2024grokking,
  title     = {Grokking as the transition from lazy to rich training dynamics},
  author    = {Kumar, Tanishq and Bordelon, Blake and Gershman, Samuel and Pehlevan, Cengiz},
  booktitle = {International Conference on Learning Representations},
  pages     = {23010--23035},
  year      = {2024}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html
@inproceedings{pesme2023saddle,
  title     = {Saddle-to-Saddle Dynamics in Diagonal Linear Networks},
  author    = {Pesme, Scott and Flammarion, Nicolas},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {7475--7505},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/075280-0329},
  year      = {2023}
}

% verified: https://proceedings.mlr.press/v202/refinetti23a.html
@inproceedings{refinetti2023neural,
  title     = {Neural networks trained with {SGD} learn distributions of increasing complexity},
  author    = {Refinetti, Maria and Ingrosso, Alessandro and Goldt, Sebastian},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {28843--28863},
  publisher = {PMLR},
  year      = {2023}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html
@inproceedings{barak2022hidden,
  title     = {Hidden Progress in Deep Learning: {SGD} Learns Parities Near the Computational Limit},
  author    = {Barak, Boaz and Edelman, Benjamin and Goel, Surbhi and Kakade, Sham and Malach, Eran and Zhang, Cyril},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {21750--21764},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/068431-1581},
  year      = {2022}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9e6ac15e689e06139d7b39e1667b165-Abstract-Conference.html
@inproceedings{prieto2025grokking,
  title     = {Grokking at the Edge of Numerical Stability},
  author    = {Prieto, Lucas and Barsbey, Melih and Mediano, Pedro and Birdal, Tolga},
  booktitle = {International Conference on Learning Representations},
  pages     = {81151--81168},
  year      = {2025}
}
"""

BIB_ICLR_SITE_ONLY = r"""% verified: https://iclr.cc/virtual/2023/oral/12716 (OpenReview unreachable; arXiv 2210.01117 lists "Eric J. Michaud")
@inproceedings{liu2023omnigrok,
  title     = {Omnigrok: Grokking Beyond Algorithmic Data},
  author    = {Liu, Ziming and Michaud, Eric and Tegmark, Max},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

% verified: https://iclr.cc/virtual/2023/oral/12746 (OpenReview forum QC10RmRbZy9 unreachable; no arXiv version)
@inproceedings{chiang2023loss,
  title     = {Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent},
  author    = {Chiang, Ping-yeh and Ni, Renkun and Miller, David Y. and Bansal, Arpit and Geiping, Jonas and Goldblum, Micah and Goldstein, Tom},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
"""


def wp13():
    """WP-13: related-work comparators named by the second reviewer (verified citations)."""
    return f"""
## WP-13. Related work: the reviewer's comparators (for the submission)

How these were verified (2026-09-25): each field was read from the proceedings page (NeurIPS proceedings, PMLR,
proceedings.iclr.cc) and from arXiv, using the pages' own citation metadata and official BibTeX. **Cite the
proceedings metadata, not arXiv's.** They differ as follows:
- Ahn et al.: the title differs.
- Kumar et al., Barak et al., Prieto et al.: middle initials are dropped in the proceedings.
- Pesme and Flammarion, Refinetti et al., Prieto et al.: the abstracts differ.

openreview.net refused every automated request, so no OpenReview decision string is verified.

**Six verified on proceedings pages.** Each has one sentence on what it shares with this paper and how it differs, written from the
paper's own abstract.

- **Ahn et al. (NeurIPS 2023), `ahn2023threshold`.** They prove, for gradient descent on simplified two-layer models,
  a sharp step-size transition below which the network fails to learn threshold neurons (non-zero first-layer bias).
  Shared: a sharp transition decides whether a unit acquires a useful bias/placement. Differs: their control
  parameter is the learning rate, acting through edge-of-stability dynamics. Ours is output scale, and our threshold
  is a property of the loss at fixed scale, certified independently of any trajectory.
- **Kumar et al. (ICLR 2024), `kumar2024grokking`.** Grokking arises from a transition from lazy to rich
  (feature-learning) dynamics, whose rate is controlled by the parameters that scale the network output. Shared:
  output scale governs when features are learned. Differs: they study the rate of feature learning along gradient
  descent on polynomial regression. We locate a scale at which the preferred solution of the conditional loss changes,
  a static property of the loss.
- **Pesme and Flammarion (NeurIPS 2023), `pesme2023saddle`.** They prove that gradient flow on diagonal linear networks
  from vanishing initialisation jumps from saddle to saddle, with the saddles and jump times given exactly. Shared: an
  exact account of discrete transitions in a small model. Differs: their transitions happen in time along the flow;
  ours are in output scale on the conditional loss, and training's relation to them is measured, not derived.
- **Refinetti et al. (ICML 2023), `refinetti2023neural`.** Networks trained with SGD first classify with lower-order
  input statistics (mean, covariance) and use higher-order statistics later. Shared: a low-order statistic comes
  first. Our small-scale lemma (WP-12) shows the conditional minimiser at small scale maximises the class-mean gap, and
  at large scale the worst-case gap. Differs: their ordering is in training time and over input statistics; ours is in
  output scale, for a fixed finite objective, and is proved.
- **Barak et al. (NeurIPS 2022), `barak2022hidden`.** Learning sparse parities shows abrupt transitions at about
  n^O(k) iterations. SGD makes continual progress through a Fourier gap that loss and error do not show. Shared: an
  abrupt change in training preceded by steady growth in a hidden variable (for us, the output scale grows before
  placement). Differs: their delay is computational; ours is a scale threshold of the loss, with a certified location.
- **Prieto et al. (ICLR 2025), `prieto2025grokking`.** Without regularisation, gradients align with a direction that
  scales the logits without changing predictions. That scaling delays generalisation and ends in Softmax Collapse.
  Shared: logit-scale growth as the variable behind a delayed transition. Differs: in their account the growth
  delays generalisation and ends in floating-point failure. In ours, output-scale growth is what makes placement the
  preferred solution, and the transition occurs when the scale crosses a certified threshold.

**Two verified against the official ICLR 2023 conference pages (iclr.cc).** The author accepted iclr.cc as the source
on 2026-09-25; OpenReview was unreachable, and proceedings.iclr.cc has no 2023 volume. Every field below matches iclr.cc.
- **Liu, Michaud and Tegmark (ICLR 2023), `liu2023omnigrok`.** Grokking is explained by the "LU mechanism": training
  and test losses plotted against weight norm look like "L" and "U". Weight norm, controlled by initialisation and
  weight decay, then decides whether grokking occurs. Shared: a norm/scale variable organises a qualitative change.
  Differs: their variable indexes a train–test mismatch in generalisation. Ours indexes which minimiser of the
  training loss itself is placed at fixed scale.
- **Chiang et al. (ICLR 2023), `chiang2023loss`.** Gradient-free optimisers, including guess-and-check (sample random
  parameters until training accuracy is perfect), reach test accuracy comparable to SGD in low-sample regimes. So the
  implicit-regularisation behaviour is largely independent of the optimiser. Shared: attributing an outcome to the
  loss landscape rather than to the optimiser; our threshold is a property of the loss. Differs: they study
  generalisation; we study which training-loss minimiser is placed at a given output scale.

**Do not say** that any of these papers studies a conditional threshold in output scale. Do not cite an OpenReview
decision ("oral", "poster") from these notes; iclr.cc lists the two 2023 papers as "Oral presentation / top 25%", and
that is the only source checked.

BibTeX (all eight; the two 2023 ICLR entries are verified on iclr.cc):
```bibtex
{BIB_VERIFIED}
{BIB_ICLR_SITE_ONLY}```
"""


CENTRAL_FAILS = {
    "X1": "cross-family: q2 matches family A on R_glob at 6/6 a but on R_solve at 0/6, so the joint prediction fails "
          "(the solve threshold does not transfer).",
    "K-2b": "numeric sub-claim 'roughly 115-145x' was an arithmetic error (actual 11.8-145x).",
    "K-3": "registered margin >= 0.10 not met (0.0089); the claim was removed.",
    "C-2 (kappa)": "kappa's magnitude clause passes (0.69-0.76%), its containment clause fails; FAIL under the "
                   "conjunctive falsifier.",
    "G-1": "training-free out-of-distribution windows: 6 of 10 within 15%, short of the registered count.",
    "G-3b": "sub-claim (far-outer window threshold >= base) was wrong by a provable inequality: a registration error.",
    "sl-tanh": "tanh at width 2: the boundary condition failed at A = 40 (criterion flaw); recorded 'neither'.",
    "H-3": "growth-rate dose-response validity gate failed (m = 1 placed 0.500 vs control 0.925); block inconclusive.",
    "F-2": "offsets scaling with the optimiser's rate ratio: SGD misses by 6.19 pp.",
    "rc-H": "'R is the controlling variable' fails across activation families (2 of 4 failure conditions): R is a "
            "correlate there.",
    "P1-b": "median solved rho predicted in [0.5, 0.95]; failed as registered.",
    "p2b-Hreverse": "median R at crossing 0.2330 against the registered < 0.20.",
    "p2bA-Rrho": "coefficient-of-variation comparison across a (R 0.083 vs R*rho 0.193) went against the prediction.",
    "B4-D2": "the competing saturation outcome at fixed scale failed (favourable to D1).",
    "B4h-Q1": "x50 did not move toward 1.0 with a longer horizon (1.046 / 1.045 / 1.045); the registered competing "
              "outcome, a persistent offset, held.",
    "B4h-Q2": "the placed fraction below threshold did not vanish at 64k steps (0.015 at 0.9, 0.094 at 0.95).",
    "own-S1": "own-threshold rule agreement 0.894 < 0.95 (its +13.4 pp over the population rule passes).",
    "size-N3": "the own-sample excess is below half the free offset at some n (a = 1.30 at n = 6,400; a = 1.50 at all n).",
    "lag-L2": "residual proportional to rate: ratio 0.858 (1.30) and 0.815 (1.50), outside [0.30, 0.70].",
    "lag2-L1'": "deconfounded lag test: the residual depends on rate too weakly (ratio 0.894 / 0.859 > 0.5).",
    "lag2-L2'": "deconfounded proportionality fails (0.974 / 0.958).",
    "4b-ID": "inherited displacement: neither teleport nor reset removes half the residual (the competing outcome held).",
    "4b-OM": "optimiser memory: likewise, neither arm removes half the residual.",
    "ramp-R5-R1": "Track 1B ramp, winding sign test (a = 1.30, k = 0): the pooled through-origin slope of r on kappa*chi is "
                  "1.3003 (SGD; bound 1.3) and 9.1 (Adam); R2 and R3 pass and the crossings do come early as predicted.",
    "act-Ta-200": "GELU/SiLU/Mish, 200 seeds: fewer than 90% of crossings at or above the validated switch (0.48 / 0.62 / 0.70).",
    "act-Tb-200": "GELU/SiLU/Mish, 200 seeds: the median residual against the population threshold is not within tolerance of "
                  "kappa*chi (wrong sign or size).",
    "trackA-L3-adam": "lag law at the unseen a = 1.65, Adam (registered per optimiser): the per-run Spearman between predicted "
                      "and observed lag is 0.25 < 0.5; the median lag is predicted (L1, L2 PASS) but not which runs lag more.",
    "band-P2a-120": "band task in R^d, 120 seeds: the median residual against the population threshold lies above the width-1 "
                    "range at d = 2 and d = 4 (0.19-0.55).",
    "T2-3d": "slowed regime, fresh seeds: the width-1 own-sample threshold has median |log err| 0.019 but does not beat "
             "the population width-2 threshold (paired interval [-1.04, +0.006]); crossings bimodal.",
    "T2-3": "asymmetric windows, width 2: training crosses above the validated threshold (93.7%) but at a median 3.29x "
            "the threshold scale, above the registered bound of 1.25.",
    "nogating": "width 2, symmetric windows, fixed-scale training: placement is gated at small held scale (placed "
                "0.46 at R2 = 0.003, rising to 0.98-0.99 at 0.1, both a), against the predicted no gating.",
    "B-spin": "(post hoc scoring) the registered falsifier of the spinodal / hysteresis picture fired at a = 1.30; the "
              "framing was dropped.",
    "B-solve": "(post hoc scoring) R at solve exceeds R_solve by 16-29%, against a 15% tolerance.",
}


def wp14():
    """WP-14: the census sorted by relevance (harsh review A4; post hoc classification)."""
    cr = pd.read_csv(RESULTS / "census_relevance.csv")
    f = cr[(cr.verdict == "FAIL") & (cr.relevance == "central")]
    missing = set(f.id) - set(CENTRAL_FAILS)
    if missing or set(CENTRAL_FAILS) - set(f.id):
        raise RuntimeError(f"WP-14 central-failure list out of step with census_relevance.csv: {missing}")
    t = cr.pivot_table(index=["scoring", "relevance"], columns="verdict", values="id", aggfunc="size", fill_value=0)
    tab = "\n".join(f"| {s} | {r} | " + " | ".join(str(int(t.loc[(s, r)].get(v, 0))) for v in ("PASS", "FAIL", "PARTIAL", "UNRESOLVED")) + " |"
                    for s, r in t.index)
    lines = []
    for claim in ("threshold", "scaling", "prospective", "width2"):
        g = f[f.claim == claim]
        lines.append(f"\n*{claim}* ({len(g)}):")
        lines += [f"- `{i}`: {CENTRAL_FAILS[i]}" for i in g.id]
    return f"""
## WP-14. The census sorted by relevance (for the submission; POST HOC classification)

Producer: `src/census_relevance.py` → `census_relevance.csv`, `census_relevance_counts.csv`. The rule was fixed by
**block topic, never by outcome**, and ties were resolved toward *central*, so that no failure is hidden as
peripheral.
- **Central:** the block's predictions concern the conditional threshold (its value, what it predicts about training
  crossings, the residual), the scaling reduction (κ, K, A*, c₁, the scaling limit, cross-family), the prospective
  held-out predictions (Block G, Block 3, own-seed), or width 2.
- **Peripheral:** budget laws, barriers and sharpness, trapping, optimiser equivalence, and the early exploratory
  probes.
- The classification is post hoc: it was made on 2026-09-25, after every verdict was known.

| scoring | relevance | PASS | FAIL | PARTIAL | UNRESOLVED |
|---|---|---|---|---|---|
{tab}

IDs: {_id("A4 FAIL registered central", "A4 FAIL registered peripheral", "A4 FAIL post hoc central", "A4 FAIL post hoc peripheral", "A4 PASS registered central", "A4 PASS registered peripheral", "A4 rows")}.

**Failed central predictions ({len(f)}: {int((f.scoring == "registered rule").sum())} under the registered rule, {int((f.scoring != "registered rule").sum())} by post hoc scoring), with one line each:**
{chr(10).join(lines)}

**The pattern, stated plainly.** The central failures cluster in the **training-side** claims, and most of them
concern the **residual**, i.e. where crossings sit relative to the threshold and why:
- the lag tests;
- Block 4b;
- the horizon extension;
- own-S1;
- sample size.

The rest are:
- sub-claims of the scaling reduction (two of them arithmetic or registration errors);
- the solve-threshold transfer (X1);
- tanh at width 2.

Tonight's program (2026-09-25) added four central failures, all training-side:
- the ramp's winding sign test (R5-R1; pooled slope 1.3003 against a bound of 1.3);
- the two 200-seed training tests for GELU, SiLU and Mish;
- the R^d band task's width-1 lag range (120 seeds).

The final round added:
- one central PASS: the registered c₁ follow-up (WP-30), whose interval of width 0.042 contains the derived c₁. The
  original c₁ test stays INCONCLUSIVE.
- the registered lag-law test at the unseen a = 1.65 (Track A, WP-35): L1 and L2 PASS for both optimisers. L3 was
  registered per optimiser and is scored as registered, one row each: SGD PASS, and Adam FAIL (a central failure).
- the registered simplicity-bias transfer test (Track T v3, WP-37), with C1 and C2 both UNRESOLVED on its validity
  conditions: 22 of 40 runs crossed, and the median χ at crossing was 7.4.

It also added seven PARTIAL rows, assigned post hoc because verdicts differ across units: the ramp's R1–R3 (across a
and optimiser) and the band task's primary P1, P2a and P2b (across d). Designs that failed their own rules before
registration (2C, the GELU prospective test, the Adam ramp from initialisation) are not registrations and are not in the
census.

None is a failure of the certified threshold values themselves, and the primary prospective comparisons (Block 3,
own-seed primary) passed.

**Say:** "Of {int((cr.verdict == "FAIL").sum())} failed predictions ({int(((cr.verdict == "FAIL") & (cr.scoring == "registered rule")).sum())} under the registered rule, {int(((cr.verdict == "FAIL") & (cr.scoring != "registered rule")).sum())} by post hoc scoring), {len(f)} bear on a central
claim. Most of these concern the residual and training's tracking of the threshold (lag tests, Block 4b, the ramp's sign
test, training outside the sine family, the lag in R^d); others are the width-2 training predictions (WP-15, WP-20)." **Do not say** "the central claims never failed" or "the failures are
peripheral". Do not present the relevance classification as registered;
it is post hoc.
"""


def wp15():
    """WP-15: a width-2 threshold on asymmetric windows (Track 2, registered)."""
    sc = json.loads((RESULTS / "asym_scores.json").read_text())
    t3 = sc["T2-3"]
    ph = json.loads((RESULTS / "asym_posthoc" / "summary.json").read_text())
    tb = json.loads((RESULTS / "asym_t23b" / "scores.json").read_text())
    tb_f = json.loads((RESULTS / "asym_t23b_frozen.json").read_text())
    e2 = json.loads((RESULTS / "asym_posthoc" / "exploratory2.json").read_text())
    tc = json.loads((RESULTS / "asym_t23c" / "scores.json").read_text())
    t3d = json.loads((RESULTS / "asym_t23d" / "scores.json").read_text())["criteria"]
    t3dd = json.loads((RESULTS / "asym_t23d" / "descriptive.json").read_text())
    _ma = pd.read_csv(RESULTS / "width2_diagnosis" / "metrics_all.csv")
    dg_sel = dict(zip(_ma[_ma.predictor == "P5"].arm, _ma[_ma.predictor == "P5"].median_abs_log_err))
    dg_p3 = dict(zip(_ma[_ma.predictor == "P3"].arm, _ma[_ma.predictor == "P3"].median_abs_log_err))
    _pr = pd.read_csv(RESULTS / "width2_diagnosis" / "predictors.csv")
    dg_undef = int(((_pr.kind == "P5") & _pr.value.isna()).sum())
    tc_f = json.loads((RESULTS / "asym_t23c_frozen.json").read_text())
    pk = ph["knockout_counts"]
    return f"""
## WP-15. Width 2 on asymmetric windows: the landscape threshold survives; training crosses far above it (Track 2, registered; for the submission)

Exploratory pilot (Δ = 0.8, a = 1.30): `asym_pilot_design.md`, with GO/no-go fixed before running (374b86a, GO
129a487). Registration: `asym_registration.md` (53dce22), with amendment 1 (2505b2d, matched initialisation, the
author's instruction). Producer: `src/asym_register.py` → `asym_parts/`, `asym_scores.json`. Windows:
I = [−0.8, 0.8], O = [−2.0, −1.2] ∪ [1.2, 2.4], so Δ = 0.4. This answers "the width-2 analysis contains no threshold
test where one is predicted".

**Why a threshold was predicted.** On asymmetric windows the linear ramp enters the class-mean gap (m = E_O x − E_I x
= 0.44). So the small-scale conditional minimiser uses the ramp and is unplaced, and the criterion (WP-12, math note
§10.1) predicts a switch. On symmetric windows it predicts none, and none was found (WP-9).

| prediction | criterion | result |
|---|---|---|
| **T2-1** (landscape) | a validated width-2 search finds exactly one switch (competing: placed at every scale) | **PASS**: unplaced at s ≤ 0.4217, placed at s ≥ 0.5623 on a 17-scale grid; bisection bracket **s ∈ [{sc["s_lo"]:.4f}, {sc["s_hi"]:.4f}]**, validated at both ends (restart ladder 500–4,000 unchanged; independent CMA-ES agrees; audit clean) |
| **T2-2** (secondary) | s_hi(Δ = 0.4) < 0.5623 (the pilot's lower end at Δ = 0.8) | **PASS** ({sc["s_hi"]:.4f}) |
| **T2-3** (training, W1's criterion) | ≥ 90% of crossings at or above s_hi; bootstrap CI of the median ratio − 1 above 0; **median(s_cross/s_lo) ≤ 1.25** | **FAIL**: {100 * t3["frac_above_s_hi"]:.1f}% at or above s_hi; CI [{t3["ci_median_minus_1"][0]:.2f}, {t3["ci_median_minus_1"][1]:.2f}]; **median ratio {t3["median_ratio_s_lo"]:.2f} > 1.25** |

- T2-3 used matched initialisation, seeds 600,000–600,079: {sc["placed_at_init"]} run placed at initialisation
  (excluded; under the 20% stop), and {sc["crossed"]} crossed.
- **Disclosed:** a first T2-3 arm with standard initialisation ran before amendment 1. Its outputs were sealed unread
  (hashes committed), and it is withdrawn and unscored.
IDs: {_id("Track 2 T2-1 PASS", "Track 2 s_lo", "Track 2 s_hi", "Track 2 T2-2 PASS", "Track 2 T2-3 FAIL", "Track 2 T2-3 frac above", "Track 2 T2-3 median ratio")}.

**Reading.**
- The criterion's landscape prediction holds at width 2: where the small-scale class-mean maximiser is unplaced, a
  placement threshold exists, validated like W0.
- Training does cross above that threshold (94% of runs), but at a median of about **3.3×** the threshold scale, not
  within the registered 25%. So at width 2 the landscape threshold does not predict *where* training crosses, as the
  width-1 threshold does (within 3–7%).


**POST HOC, EXPLORATORY (the author's request, after T2-3 failed): what the runs cross on.** Producer:
`src/asym_posthoc.py` → `asym_posthoc/`. The definitions were fixed before running: knockout class at the crossing;
weight shares along the trajectory; each run's branch switch scale by continuation from its crossing configuration;
the explicit single-unit branch; and the width-1 threshold for these windows. All 79 replays reproduce their
crossings exactly.
- **No run crosses on a single-unit branch.** At the crossing, {pk["pair"]} of 79 are a cooperating pair (neither unit
  alone is placed) and {pk.get("redundant", 0)} are redundant; **0 are single-unit**. Before the crossing, the median run
  has a dominant unit (share ≥ 0.75) at only {100 * ph["median_frac_pre_dominant"]:.0f}% of its logged steps.
- **The single-unit branch** (the width-1 minimiser embedded in width 2, with 99% of the weight on one unit) switches
  at s ∈ ({ph["single_branch_switch_lo"]:.2f}, {ph["single_branch_switch_hi"]:.2f}]. The width-1 threshold for these windows
  is s = {ph["s_w1"]:.2f}. **{100 * ph["frac_cross_below_single_branch"]:.1f}% of crossings lie below it**: median crossing
  s = {ph["median_s_cross"]:.2f} (10–90%: {ph["s_cross_p10"]:.2f}–{ph["s_cross_p90"]:.2f}), i.e. {ph["median_ratio_w1"]:.2f}× the width-1
  threshold.
- **The branch the runs actually cross on** (continuation from each crossing configuration) switches at a median
  s = {ph["median_s_branch"]:.3f}, essentially the width-2 global threshold ({ph["s_glob_w2"]:.3f}). The runs cross at
  {ph["median_ratio_branch"]:.1f}× their own branch's switch.
- **No reference tracks the crossings.** Mean |log error|: branch {ph["mean_abslog_branch"][0]:.2f}, width-2 global
  {ph["mean_abslog_glob"][0]:.2f}, width-1 {ph["mean_abslog_w1"][0]:.2f}. Branch − global is
  {ph["branch_minus_glob"][0]:+.3f} [{ph["branch_minus_glob"][1]:+.3f}, {ph["branch_minus_glob"][2]:+.3f}] and width-1 − global is
  {ph["w1_minus_glob"][0]:+.3f} [{ph["w1_minus_glob"][1]:+.3f}, {ph["w1_minus_glob"][2]:+.3f}]; both intervals contain 0.
  Spearman(crossing, branch switch) = {ph["spearman_cross_branch"]:.2f}.
IDs: {_id("Track 2 posthoc single-unit at crossing", "Track 2 posthoc pair at crossing", "Track 2 posthoc single branch switch lo", "Track 2 posthoc single branch switch hi", "Track 2 posthoc width-1 threshold", "Track 2 posthoc frac below single branch", "Track 2 posthoc median branch switch", "Track 2 posthoc branch minus glob hi", "Track 2 posthoc Spearman branch")}.

**Say (post hoc, beside the failure):** "The registered training prediction failed. Post hoc, the runs cross as
cooperating pairs, on the branch whose switch is the width-2 threshold, but about three times above it, and below the
single-unit (width-1) threshold. Neither threshold, nor the branch's own switch, predicts the crossing scale."

**Do not say:** that the failure is explained by a single-unit branch (no run is single-unit at its crossing), that
crossings "track" any of these thresholds, or that the post hoc analysis rescues T2-3. It is exploratory, and the
registered verdict is FAIL.


**T2-3b (registered after T2-3's failure; amendment 2, 61f3353): T2-3 plus the approved sweep-rate matching.**
- When T2-3 was amended, only matched initialisation was specified. The approved design's sweep-rate matching
  (Revision 3) was left out, and T2-3b adds it.
- The output learning-rate factor φ₂ = {tb_f["phi2"]:.4f} came from the steps-matching pilot (calibration seeds,
  ‖w₂‖₁ only). The validity check requires the achieved median timescale ratio at crossing to lie in the width-1 Adam
  range [0.0014, 0.023].
- **Result: UNRESOLVED.** The achieved median ratio is {tb["median_ratio_at_cross"]:.1e}, far *below* the range. Steps
  matching over-slows the scale's growth at the crossing, so the two notions of rate matching disagree by about three
  orders of magnitude.
- For information only, the criteria alone would give FAIL: {tb["crossed"]}/{tb["runs"]} crossed,
  {100 * tb["criteria"]["frac_above_s_hi"]:.1f}% at or above s_hi, median ratio {tb["criteria"]["median_ratio_s_lo"]:.1f}, and a
  bootstrap interval [{tb["criteria"]["ci_median_minus_1"][0]:.2f}, {tb["criteria"]["ci_median_minus_1"][1]:.2f}] that contains 0.
- **T2-3 stays FAIL.**
IDs: {_id("Track 2 T2-3b UNRESOLVED", "Track 2 T2-3b phi2", "Track 2 T2-3b median ratio at crossing", "Track 2 T2-3b frac above", "Track 2 T2-3b median crossing ratio")}.

**EXPLORATORY note (author's request; changes nothing above).**
- *Training data.* The T2-3 runs trained on their own sampled sets, not on the population objective.
- *Own-sample thresholds.* Own-sample width-2 thresholds for 10 seeds (200 restarts) have median
  s = {e2["own_median"]:.3f} (range {e2["own_range"][0]:.3f}–{e2["own_range"][1]:.3f}), the population value.
  - These seeds cross at {e2["median_cross_over_own"]:.1f}× their own threshold, against {e2["median_cross_over_pop"]:.1f}× the
    population's.
  - Mean |log error| is {e2["mean_abslog_own"]:.2f} against own and {e2["mean_abslog_pop"]:.2f} against the population;
    {e2["runs_closer_to_own"]}/10 runs are closer to their own; Spearman(crossing, own) = {e2["spearman_cross_own"]:.2f} (n = 10).
  - So own-sample thresholds do not explain the width-2 residual.
- *Timescale ratio.* At the T2-3 crossings the median ratio is {e2["ts_median_ratio"]:.2f}, about 100× beyond the width-1
  range. The width-1 fit, extrapolated, predicts a residual of {e2["width1_fit_predicted_median_residual"]:.1f} against
  {e2["observed_median_residual_vs_pop"]:.2f} observed. Within T2-3, Spearman(residual, ratio) = {e2["spearman_residual_ratio_within_T2_3"]:.2f}.
IDs: {_id("Track 2 expl own median", "Track 2 expl cross over own", "Track 2 expl Spearman own", "Track 2 expl ratio median", "Track 2 expl ratio Spearman within")}.

**T2-3c (registered after T2-3's FAIL and T2-3b's UNRESOLVED; amendment 3, 9ba85f5): timescale-ratio matching.**
- φ₂ = {tc_f["phi2"]:.5f} was chosen by a pilot on fresh calibration seeds. The pilot recorded only the timescale ratio
  at the step where ‖v‖₁ first reaches the threshold scale (median {tc_f["pilot_median_ratio"]:.4f}, within 20% of 0.0057),
  and never evaluated placement.
- **Result: UNRESOLVED.** The achieved median ratio **at crossing** is {tc["median_ratio_at_cross"]:.1e}, far below the
  width-1 range. The risk stated in the registration materialised: the runs cross only much later (median
  {tc["criteria"]["median_ratio_s_lo"]:.1f}× s_lo), after the scale has nearly stopped growing.
- For information only, the criteria alone would give FAIL: {tc["crossed"]}/{tc["runs"]} crossed,
  {round(tc["criteria"]["frac_above_s_hi"] * tc["crossed"])}/{tc["crossed"]} at or above s_hi, median ratio {tc["criteria"]["median_ratio_s_lo"]:.1f}.
IDs: {_id("Track 2 T2-3c UNRESOLVED", "Track 2 T2-3c phi2", "Track 2 T2-3c median ratio at crossing", "Track 2 T2-3c frac above", "Track 2 T2-3c median crossing ratio")}.

**EXPLORATORY diagnosis (item 2; POST HOC; gate STOP, so no further test and no change to this section's claims).**
The rules and the gate were committed before any result (`width2_diagnosis_gate.md`, 4b7826e). Producer:
`src/width2_diagnosis.py` → `width2_diagnosis/`.
- **Predictors of each run's crossing scale:** the population width-2 threshold; the own-sample width-2 threshold; the
  own-sample width-1 threshold; the population width-1 threshold (s = 5.09); and the switch of the branch reached from
  the run's own crossing configuration.
- **Selected (lowest pooled error): the branch switch.** Median |log error| is {dg_sel["T2-3b"]:.3f} (T2-3b) and
  {dg_sel["T2-3c"]:.3f} (T2-3c), but **{dg_sel["T2-3"]:.2f} (T2-3)**, so the gate (≤ 0.10 in every φ₂ setting) STOPS.
- In the two slowed arms, the **width-1 own-sample threshold on the same training set** also matches the crossings closely:
  median |log error| {dg_p3["T2-3b"]:.3f} and {dg_p3["T2-3c"]:.3f} (10 seeds each). Nothing tracks the full-speed T2-3
  crossings.
- Caveats:
  - the branch switch is computed from the crossing configuration itself, so it is close to circular;
  - it is undefined for {dg_undef} slowed-arm runs;
  - the own-sample predictors rest on 10 seeds per arm.
IDs: {_id("Item 2 gate STOP", "Item 2 selected P5", "Item 2 P5 T2-3 error", "Item 2 P3 T2-3b error", "Item 2 P3 T2-3c error")}.
**Do not say** that any predictor explains where width-2 training crosses, or that width-2 crossings follow the
width-1 threshold. This is a post hoc, 10-seed observation in the slowed arms only, and it failed the gate.

**T2-3d (registered as designed after the item 2 diagnosis; amendment 4, 394ebcc): the slowed regime only.**
- φ₂ = 0.01778 (T2-3c's), fresh seeds 600,480–600,559.
- Each seed's width-1 own-sample threshold was frozen with a hash before any training (68323b2).
- Criteria: (i) median |log error| ≤ 0.10, and (ii) it beats the population width-2 threshold (paired interval below 0).
- **Result: FAIL.**
  - (i) holds: {t3d["median_abs_log_err"]:.3f}. The median crossing is {t3d["median_cross_over_own"]:.2f}× the width-1 own
    threshold.
  - (ii) fails: [{t3d["diff_lo"]:.2f}, {t3d["diff_hi"]:+.4f}] contains 0.
- The crossings are bimodal: {t3dd["n_near_own"]} of {t3dd["n"]} runs cross at their width-1 own threshold, and
  {t3dd["n_pop_beats_own"]} cross early at small scale (s = {t3dd["early_small_scale_range"][0]:.3f}–{t3dd["early_small_scale_range"][1]:.2f}).
- **Scope:** T2-3d covers the slowed regime only. The full-speed regime (T2-3) remains unexplained.
IDs: {_id("Track 2 T2-3d FAIL", "Track 2 T2-3d median err", "Track 2 T2-3d diff hi", "Track 2 T2-3d bimodal")}.
**Say (T2-3d):** "A registered test designed after the diagnosis found that, in the slowed regime, the median run crosses at
its own width-1 threshold, but about 40% of runs cross early at small scale, so the registered comparison with the
width-2 threshold fails. The full-speed regime remains unexplained."
**Do not say** that the width-1 threshold predicts width-2 crossings, or that T2-3d rescues T2-3, T2-3b or T2-3c.

**Say (T2-3, T2-3b and T2-3c, in order):** "The registered width-2 training prediction on asymmetric windows, T2-3,
failed: runs crossed at a median 3.3× the threshold. Two follow-ups were registered after that failure, each labelled as
such, to put width-2 training in width 1's timescale regime. T2-3b (steps-matched output rate) is unresolved: its runs
crossed at a timescale ratio far below width 1's range. T2-3c (ratio-matched at the threshold scale) is unresolved for
the same reason: the ratio at crossing fell far below the range. Neither follow-up tested the prediction, and T2-3 stands
as a failure."

**Do not say:** that T2-3c (or T2-3b) is a retry of a failed test, that either rescues or qualifies T2-3's failure,
that width-2 training "would" track the threshold in width 1's regime (never achieved), or that own-sample thresholds
or the timescale ratio explain the width-2 residual.

**Say:** "On asymmetric windows, where the criterion predicts a switch, a validated width-2 placement threshold exists
(registered). Width-2 training crosses above it, but at about three times the threshold scale, so the registered
training prediction fails."

**Do not say** that the width-2 threshold predicts training crossings. Do not cite the Δ = 0.8 pilot as evidence:
it is exploratory.
"""


def wp16():
    """WP-16: the conditional threshold under SGD (Track 4, registered) and the residual timescale (Track 3, post hoc)."""
    _sc = pd.read_csv(RESULTS / "sgd_own_scores.csv"); _sn = pd.read_csv(RESULTS / "sgd_own_sensitivity.csv")
    sc_ = {round(r.a, 2): r._asdict() for r in _sc.itertuples()}; sn_ = {round(r.a, 2): r._asdict() for r in _sn.itertuples()}
    sc = pd.read_csv(RESULTS / "sgd_own_scores.csv").set_index("a")
    ex = pd.read_csv(RESULTS / "sgd_own_extension_scores.csv").set_index("a")
    ts = json.loads((RESULTS / "residual_timescale_summary.json").read_text())
    fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
    B = json.loads((RESULTS / "sgd_own_budget.json").read_text())["budget"]
    sens = pd.read_csv(RESULTS / "sgd_own_sensitivity.csv").set_index("a")
    srow = lambda a: (f"| {a:.2f} | [{sens.loc[a, 'G1_lo']:+.3f}, {sens.loc[a, 'G1_hi']:+.3f}] **{sens.loc[a, 'G1']}** | "
                      f"{sens.loc[a, 'G2_spearman']:.3f} **{sens.loc[a, 'G2']}** | {sens.loc[a, 'EXT_obs_imputed']:.4f} vs "
                      f"{sens.loc[a, 'EXT_pred']:.4f} (±{sens.loc[a, 'EXT_tol']:.4f}) **{sens.loc[a, 'EXT']}** |")
    row = lambda a: (f"| {a:.2f} | {int(sc.loc[a, 'crossed'])}/{int(sc.loc[a, 'runs'])} | {sc.loc[a, 'G1_mean_diff']:+.4f} "
                     f"[{sc.loc[a, 'G1_lo']:+.4f}, {sc.loc[a, 'G1_hi']:+.4f}] **{sc.loc[a, 'G1']}** | "
                     f"{sc.loc[a, 'G2_spearman']:.3f} **{sc.loc[a, 'G2']}** | {100 * sc.loc[a, 'G3_median_resid_own']:+.2f}% | "
                     f"{100 * sc.loc[a, 'G3_median_resid_pop']:+.2f}% |")
    erow = lambda a: (f"| {a:.2f} | {ex.loc[a, 'median_ratio']:.5f} | {ex.loc[a, 'pred']:.4f} | {ex.loc[a, 'obs']:.4f} | "
                      f"±{ex.loc[a, 'tol']:.4f} | **{ex.loc[a, 'EXT']}** |")
    return f"""
## WP-16. Does the conditional threshold predict SGD crossings? (Track 4, registered; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Registration: `sgd_own_registration.md` (0506650); budget fixed before any registered run (988c1d7); amendment 1,
the timescale extension (1df48d1, 13:13 EDT), registered before any SGD run was scored. Producer: `src/sgd_own.py` →
`sgd_own_runs.csv`, `sgd_own_scores.csv`, `sgd_own_ratios.csv`, `sgd_own_extension_scores.csv`. This answers "does the
conditional threshold predict SGD crossings, or only the a = 1.25 midpoint?"

**Design.**
- SGD at lr 0.3 (the only rate that solves in this setting), full batch.
- Exactly the phase-2b training sets (seeds 0–39, a = 1.30 and 1.50, n = 400), whose own-sample global thresholds were
  already computed. So there is no new threshold computation.
- The same initialisation as the Adam runs of each seed, and every-step crossing detection.
- Budget {B:,} steps, fixed by a rule on calibration seeds that recorded only |w₂| growth.

| a | crossed | G1: mean(e_own − e_pop) [95% CI] | G2: Spearman(cross, own) | residual vs own | vs population |
|---|---|---|---|---|---|
{row(1.3)}
{row(1.5)}

- **G1 passes at both a.** Each run's own threshold predicts its SGD crossing better than the population threshold.
- **G2 passes at both a.**
- The SGD residuals against the own threshold, +2.95% and +6.43%, are close to Adam's in the deconfounded lag test
  (+3.1% and +6.6% at φ = 1).
- **Caveat.** The budget rule aimed at ≥ 90% crossing; 75% crossed at each a (30/40), which is exactly the registered
  minimum of 30. Report the crossing fraction.
IDs: {_id("Track 4 G1 a=1.30", "Track 4 G1 a=1.50", "Track 4 G2 a=1.30", "Track 4 G2 a=1.50", "Track 4 crossed a=1.30", "Track 4 crossed a=1.50", "Track 4 residual vs own a=1.30", "Track 4 residual vs own a=1.50")}.

**The residual's timescale (Track 3, POST HOC, exploratory).**
- For each of the 384 crossings of the deconfounded lag test (both a, four learning-rate arms), the run was replayed
  to its crossing. Every replay reproduces its crossing exactly.
- Two rates were measured at the crossing: the output scale's growth rate d log s/dt, and the branch's
  Adam-preconditioned relaxation rate, lr·λ_min(D^(−1/2) H D^(−1/2)).
- Spearman(residual, growth/relaxation) is **{ts["spearman_run_level"]:.3f} [{ts["spearman_ci95"][0]:.3f}, {ts["spearman_ci95"][1]:.3f}]**
  pooled. Within each a it is **{ts["spearman_within_a"]["a=1.30"]:.3f}** and **{ts["spearman_within_a"]["a=1.50"]:.3f}**; much
  of the pooled value is the difference between the two a. Across the 8 cells it is 1.0.
- The growth rate varies only about 25% across an 8× range of w₂'s learning rate, because Adam's normalisation
  compensates. This matches L1′'s registered failure.
IDs: {_id("Track 3 Spearman pooled", "Track 3 Spearman CI lo", "Track 3 Spearman CI hi", "Track 3 Spearman within a=1.30", "Track 3 Spearman within a=1.50", "Track 3 replays reproduce")}.

**The timescale account, tested prospectively on SGD (EXT, registered amendment).**
- The residual = α + β·ratio fit on Adam (α = {fit["alpha"]:.4f}, β = {fit["beta"]:.3f}) was frozen before scoring.
- Each SGD run's own ratio at its crossing was measured with the identity preconditioner.

| a | median SGD ratio | predicted residual | observed | tolerance | verdict |
|---|---|---|---|---|---|
{erow(1.3)}
{erow(1.5)}

- It passes at both a. At a = 1.50 the miss (0.0113) is close to the tolerance (0.0133).
- The SGD ratios lie inside the fitted Adam range, so this is interpolation across optimisers, not extrapolation.
IDs: {_id("Track 4 EXT a=1.30", "Track 4 EXT a=1.50", "Track 4 EXT pred a=1.30", "Track 4 EXT pred a=1.50")}.

**Sensitivity analysis (POST HOC, author's request): non-crossers treated as crossing at their final |w₂|.**
- 25% of runs did not cross (10 per a). Here each is scored as crossing at (or above) its final |w₂|, a lower bound on
  its unobserved crossing. A replay confirms that none of them crosses within the budget.

| a | G1 [95% CI] | G2: Spearman | EXT: observed median vs predicted |
|---|---|---|---|
{srow(1.3)}
{srow(1.5)}

- **One verdict changes: G2 fails at both a under this imputation.** G1 and EXT are unchanged.
- The non-crossers ended at a median **{sens.loc[1.3, "median_final_over_own_noncrossers"]:.2f}×** (a = 1.30) and
  **{sens.loc[1.5, "median_final_over_own_noncrossers"]:.2f}×** (a = 1.50) of their own threshold. They stalled well
  below the threshold scale, so the imputed values are very loose lower bounds, not crossings.
IDs: {_id("Track 4 sens G1 a=1.30", "Track 4 sens G1 a=1.50", "Track 4 sens G2 a=1.30", "Track 4 sens G2 a=1.50", "Track 4 sens EXT a=1.30", "Track 4 sens EXT a=1.50", "Track 4 sens noncrosser final/own a=1.30", "Track 4 sens noncrosser final/own a=1.50")}.

**How to describe the timescale account.**
- It is **supported by two prospective tests, not established**: the SGD extension (this section) and Task B (WP-21:
  Adam at a = 1.45 and 1.60, which the fit never saw, on fresh training sets).
- The Adam relationship is a post hoc correlation. Always give the within-a values beside the pooled one: pooled
  Spearman {ts["spearman_run_level"]:.2f}, but only {ts["spearman_within_a"]["a=1.30"]:.2f} (a = 1.30) and
  {ts["spearman_within_a"]["a=1.50"]:.2f} (a = 1.50) within each a.
- Two registered prospective checks passed: EXT on a different optimiser (at a = 1.50 near the edge of its tolerance),
  and TS-1 at two unseen activation values (WP-21).

**Say:** "Trained with SGD on the same samples, each run's own conditional threshold predicted its crossing better
than the population threshold (registered; both a; 75% of runs crossed). A post hoc timescale ratio, correlated with
the residual on Adam runs (Spearman 0.63 pooled; 0.45 and 0.51 within each a), predicted the median SGD residual
within the registered tolerance, and a second registered test at unseen activation values also passed (WP-21). The
timescale account is supported by these two prospective tests; it is not established."

**Do not say:** that the timescale ratio is the residual's mechanism, or that the account is established (it did not
carry over to width 2: WP-15's exploratory note). Do not quote
the pooled Spearman without the within-a values. Do not say that SGD crossings were universal. Do not report G2 without
noting that it fails when the non-crossers are imputed. Do not merge these runs with Block F's a = 1.25 SGD arm.

**Exact wording for the SGD own-sample result (final round, D.4): a partial pass.**
- Say: "With SGD, the registered own-sample predictions pass on the runs that crossed (30 of 40 at each a): each run's
  own threshold predicts its crossing better than the population threshold, and the rank correlation between crossing
  and own threshold is {sc_[1.3]["G2_spearman"]:.2f} and {sc_[1.5]["G2_spearman"]:.2f} (registered bound 0.6). When
  the 10 non-crossing runs at each a are counted at their final scale, the first prediction still holds, but the rank
  correlation falls to {sn_[1.3]["G2_spearman"]:.2f} and {sn_[1.5]["G2_spearman"]:.2f} and fails the bound. We
  therefore report the SGD result as a partial pass: it holds for crossers only."
- Do not say: "the SGD own-sample test passed" without "for the runs that crossed".
"""


def wp18():
    """WP-18: stronger baselines (Track 6, post hoc)."""
    c = pd.read_csv(RESULTS / "baselines_posthoc_block3.csv") if (RESULTS / "baselines_posthoc_block3.csv").exists() else None
    return f"""
## WP-18. Stronger baselines (Track 6; POST HOC; for the submission)

Producer: `src/baselines_posthoc.py` → `baselines_posthoc*.csv`, `baselines_posthoc.md` (d7d88d8). Every baseline is fitted
only on the ten Block G settings, the data available before the held-out runs. The re-implementation first reproduces
the committed C, B1, B2, U and own-seed numbers (163 checks, largest difference 2.2e−16). This answers "B1 and B2 are
weak".

| model | what it uses | Block 3 mean \|log err\| | C − model [95% CI] |
|---|---|---|---|
| C (registered) | U × λ fitted on the base window | 0.0223 | — |
| PL | U × λ pooled over all five Block G windows | 0.0231 | −0.0007 [−0.0088, +0.0074]: **matches C** |
| PL5 | U × median of the five windows' λ | 0.0274 | −0.0051 [−0.0120, +0.0019]: not separated |
| B2 (registered) | pooled median crossing R | 0.0773 | −0.0550 [−0.1027, −0.0105] |
| RG | per-a regression on log Ĝ | 0.1047 | −0.0823 [−0.0926, −0.0722] |
| RK | regression on log κ₀ and a (no conditional threshold) | 0.1099 | −0.0875 [−0.0993, −0.0746] |
| B1 (registered) | base-window median \|w₂\| | 0.2796 | −0.2573 [−0.3704, −0.1312] |

- On the own-seed test, PL is not separated from C at a = 1.50. At a = 1.30, C is better by 0.001 (the interval
  just excludes 0).
- **Stated plainly: a stronger baseline (PL) matches C.** PL still uses the conditional threshold U; it differs from C
  only in fitting the lag factor on five windows instead of one.
- **Every baseline that does not use the conditional threshold loses to C by a factor of 4–5 in error, and also to
  B2.**
IDs: {_id("Track 6 reproduction checks", "Track 6 PL mean abs log err", "Track 6 C - PL lo", "Track 6 C - PL hi", "Track 6 C - RK hi")}.

**Say:** "Baselines that do not use the conditional threshold do 4–5 times worse than the registered predictor. A
variant that pools the lag factor over all calibration windows matches it (post hoc)."

**Do not say:** "C beats every baseline". Do not present PL as registered: it is post hoc.
"""


def wp19():
    """WP-19: H1 proved for small ε (Track 8)."""
    return """
## WP-19. H1 is proved for small ε (Track 8; for the submission)

Source: math note §12 (`math_note_for_writer.md`), every step checked by hand. The numerical sanity checks
(`src/h1_checks.py` → `h1_checks.log`) are separate from the proof. This answers "Proposition 2 also retains
hypothesis H1".

**Statement.** For 0 < ε ≤ 0.029 (a ≤ 1.029), Ĝ(a) > 0 is attained, and every maximiser has, up to the symmetries,
rescaled coordinates with 1.0155 ≤ p < 1.74964 and |q| < 1.7956, inside the compact set C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}.
Moreover |K(ε) − K| ≤ 23.3ε.

**Proof idea, for the text.**
- An exact identity: f_a(t_o) − f_a(t_i) = 2a cos(m) sin(d/2) − d. Applied to three pairs of window endpoints it
  gives G ≤ 0.4εw₁, that G > 0 forces 1.4w₁ below the root of sin x = x/a, and an ellipse bound on the bias.
- An exact rational evaluation, G₀(1.6, 1.2) = 27088/46875, bounds K(ε) from below.
- No numerical certificate is used.

**Consequence.** Proposition 2 is asymptotic ("for all sufficiently small ε"), so **H1 is no longer a hypothesis: it
is proved.**

**What remains certified rather than proved.**
- That the maximiser is the tied corner (the K(ε) = K_loc(ε) step of the c₁ calculation). This needs uniqueness of
  G₀'s maximiser, which K's branch and bound certifies; that check is now independently verified in Arb (WP-17).
- The argument does not cover a ≥ 1.30: there the q-bound gives 2.0008 > 2. At a = 1.02–1.25 the location in C
  follows from the proof's lemmas together with the certified Ĝ values.

**Say:** "Hypothesis H1 of Proposition 2 is proved for ε ≤ 0.029 (Appendix …)."
**Do not say:** "H1 is proved for all tested a", or "the corner is proved to be the global maximiser". The latter is
certified, not proved.
"""


def wp17():
    """WP-17: independent certificate checks, Track 5 status (updates WP-11's pending list)."""
    cc = RESULTS / "certificate_checks"
    J = lambda n: json.loads((cc / f"{n}.json").read_text()) if (cc / f"{n}.json").exists() else None
    kb, kt, kr = J("K_base"), J("K_base_target0.5794559217"), J("c1_krawczyk")
    pdg, pds = J("pd_glob_neighbourhood"), J("pd_solve_neighbourhood")
    rg, rs = J("ring_glob_annulus"), J("ring_solve_annulus")
    st = lambda r: "**verified**" if (r is not None and r.get("pass")) else ("**failed**" if r is not None else "not run")
    return f"""
## WP-17. Independent certificate checks: status after Track 5 (for the submission; replaces WP-11's "pending" list)

Checker: `src/verify_certificates.py` (python-flint / Arb, 80-bit balls; imports nothing from `src/`). Each item is
verified, failed, or not run. Tests exercise every new check on constructed pass and fail cases.

| certificate | status | what the checker did |
|---|---|---|
| K = sup G₀ over [0, 8] × [−12, 12] | {st(kb)} | a fresh Arb branch and bound over the whole box: sup G₀ ≤ 0.5794951 (the published hi) and ≤ 0.5794559217 (the tight hi); the lower end is attained at the published argmax; the domain lemma is checked in exact rational arithmetic |
| c₁ Krawczyk boxes (switch and K vertex) | {st(kr)} | switch: unique zero in the 1e−9 box, A* ∈ [{kr["A_star_lo"] if kr else ""}, {kr["A_star_hi"] if kr else ""}], A′(0)/A* ∈ [{kr["A1_over_A_lo"] if kr else ""}, {kr["A1_over_A_hi"] if kr else ""}], active set unique; K vertex unique, K = {kr["K_vertex_lo"] if kr else ""} |
| PD boxes, glob chain (500 boxes, U × [0.66, 0.71]) | {st(pdg)} | b* bracketed; the Hessian enclosure minus 0.0473·I is PD by Sylvester on every box |
| PD boxes, solve chain (100 boxes, × [1.05875, 1.06]) | {st(pds)} | as above, with margin 0.0209 |
| ring (no critical point, 0.05–0.15), glob chain, 40 A-sub-intervals | {st(rg)} | the envelope gradient excludes 0 in p or q on every ring box, with b* bracketed, uniformly over each A-sub-interval |
| ring, solve chain | {st(rs)} | as above, over [1.05875, 1.06] |
| outer exclusion (K(24) minus the 0.15 box), both chains | not run | projected > 10 CPU-hours in Arb (about 39,000 cells per A-sub-interval before refinement, × 40 sub-intervals) |
| limit solve bracket (A_solve ∈ (1.05875, 1.06]) | not run | needs the solve-chain outer exclusion above |
| finite-a solve brackets (12 certificates) | not run | timing sample: one bracket end's search at the published 1e−11 tolerance took 19 min and produced 25.2 million losing-region leaves; at about 20 ms per leaf in Arb, that is several CPU-days per certificate |

**What is now fully independently verified.**
- **R_glob^∞ ∈ [0.19738, 0.19920]:** A*'s bracket (WP-11) and K's enclosure (above) are both independently checked.
- **A* and A′(0)/A* on the branch (Krawczyk), and K at the vertex.** The sharp value R_glob^∞ ∈ [0.1985926, 0.1985927]
  additionally needs the Krawczyk switch to be the *global* switch. That needs the uniqueness chain: localisation
  (verified, WP-11), outer exclusion (not run), ring (above) and PD (above). So the sharp value is **not yet fully
  independent**; the bracket is.

**Say (certificate status paragraph):** "An independent checker in ball arithmetic (Arb) re-verifies the finite-a
placement brackets, the Ĝ(a) enclosures, the limit switch bracket and its localisation, K = sup G₀ with its domain
lemma, the Krawczyk boxes of the first-order calculation, and the positive-definite and ring certificates of the
uniqueness chain. The outer-exclusion certificates and the solve brackets are verified by the original searches only."

**Do not say:** "all certificates are independently verified", or that the sharp R_glob^∞ = 0.19859 is independently
verified.
"""


def wp20():
    """WP-20: the width-2 no-gating training test (Track 7, registered)."""
    s = pd.read_csv(RESULTS / "width2_nogating_scores.csv")
    fz = json.loads((RESULTS / "width2_nogating_frozen.json").read_text())
    ctl = pd.read_csv(RESULTS / "width2_nogating_parts" / "control.csv") if (RESULTS / "width2_nogating_parts" / "control.csv").exists() else None
    row = lambda act, var: " / ".join(f"{r.placed_frac:.2f}" for r in s[(s.act == act) & (s.variant == var)].sort_values("R2").itertuples())
    return f"""
## WP-20. Does output scale gate placement in width-2 training? (Track 7, registered; for the submission)

Design and registration: `width2_nogating_design.md` (approved 2026-09-24, with amendments before any run). Horizon
H = {int(fz["H"]):,}, frozen with a hash from a pilot that cannot see the outcome (5922e24). Producer:
`src/width2_nogating.py` → `width2_nogating_scores.csv`. This answers "the width-2 analysis contains no width-2 training
test".

**Design.**
- Width 2, symmetric windows, a = 1.30 and 1.50.
- The latest pre-placement checkpoints of 80 training runs per a (matched initialisation) are replayed with ‖w₂‖₁
  held at R₂ ∈ {{0.003, 0.01, 0.03, 0.1}}, far below width 1's switch. The primary arm preserves the optimiser state.
- The outcome is placement at H, from exact extrema.
- The landscape verdict (WP-9) says the conditional minimiser is placed at every scale, so the prediction was
  **no gating**.

| arm | placed fraction at R₂ = 0.003 / 0.01 / 0.03 / 0.1 |
|---|---|
| a = 1.30, preserved (primary) | {row("f1.30", "preserved")} |
| a = 1.50, preserved (primary) | {row("f1.50", "preserved")} |
| a = 1.30, reset | {row("f1.30", "reset")} |
| a = 1.50, reset | {row("f1.50", "reset")} |
| tanh, preserved (descriptive) | {row("tanh", "preserved")} |

- **Registered verdict: the predicted "no gating" FAILS at both a. The registered competing outcome, gating at small
  scale, holds.**
- Validity: the positive control passed (width 1: 0.0 placed at R/R_glob = 0.1, both a). The validity checks passed
  on constructed cases before the run.
- **Where the unplaced runs sit (descriptive, reported beside the verdict).**
  - At H they are stationary two-unit configurations: median scale-relative gradient 1.4e−7, and 59% below the 1e−6
    tolerance.
  - None is a single-unit local minimum, so the registered longer-horizon extension applied to 0 runs.
  - Only 5 of 640 endpoints changed status between H/4 and H, all to placed.
  - Placed endpoints are almost all the cancelling pair.
IDs: {_id("Track 7 no-gating FAIL a=1.30", "Track 7 no-gating FAIL a=1.50", "Track 7 placed 0.003 a=1.30", "Track 7 placed 0.1 a=1.30", "Track 7 positive control", "Track 7 H")}.

**Reading.**
- At width 2 on symmetric windows, the *conditional minimiser* is placed at every scale (WP-9, registered).
- *Training* at a fixed small output scale nevertheless stays unplaced about half the time, at stationary unplaced
  two-unit configurations.
- So gating in training does not require a threshold of the conditional minimiser. Here it comes from where training
  gets stuck, not from what the loss prefers globally.

**Say:** "The landscape predicts no gating at width 2 on symmetric windows, and none exists in the conditional
minimiser. The registered training test nevertheless found gating: at small held scale about half the runs remain
unplaced, at stationary two-unit configurations. So the conditional-threshold account does not by itself predict
width-2 training."

**Do not say:** "width 2 confirms no gating", or that the width-2 result is a training-level confirmation. The training
prediction failed.
"""


def wp21():
    """WP-21: a second prospective test of the timescale account (Task B, registered)."""
    sc = pd.read_csv(RESULTS / "ts_test" / "scores.csv").set_index("a")
    fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
    row = lambda a: (f"| {a:.2f} | {int(sc.loc[a, 'crossed'])}/{int(sc.loc[a, 'runs'])} | {sc.loc[a, 'median_ratio']:.4f} | "
                     f"{sc.loc[a, 'pred']:.4f} | {sc.loc[a, 'obs']:.4f} | ±{sc.loc[a, 'tol']:.4f} | **{sc.loc[a, 'TS-1']}** | "
                     f"[{sc.loc[a, 'TS2_lo']:+.3f}, {sc.loc[a, 'TS2_hi']:+.3f}] **{sc.loc[a, 'TS-2']}** |")
    return f"""
## WP-21. A second prospective test of the timescale account (Task B, registered; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Registration: `ts_test_registration.md` (db2a915), written before any run. The own thresholds for the fresh training
sets were frozen with a hash before any Adam run (0da303c). Producer: `src/ts_test.py` → `ts_test/runs.csv`,
`ts_test/scores.csv`.

**Design.**
- Adam, standard protocol, every-step crossing detection.
- **Fresh training sets:** seeds 830,000–830,079, never used anywhere, at **a = 1.45 and 1.60**. The timescale fit
  used only a = 1.30 and 1.50.
- The residual–ratio relationship is the one frozen for the SGD extension (α = {fit["alpha"]:.4f}, β = {fit["beta"]:.3f}).

| a | crossed | median ratio at crossing | TS-1 predicted | observed | tolerance | TS-1 | TS-2: own − population [95% CI] | TS-2 |
|---|---|---|---|---|---|---|---|---|
{row(1.45)}
{row(1.6)}

- Every run crossed, so the registered non-crosser sensitivity analysis is identical to the primary scores.
- At a = 1.60 the median ratio ({sc.loc[1.6, "median_ratio"]:.4f}) is slightly above the fit's range (maximum
  {fit["ratio_range_fitted"][1]:.4f}), so that prediction is a slight extrapolation.
IDs: {_id("Task B TS-1 a=1.45", "Task B TS-1 a=1.60", "Task B TS-2 a=1.45", "Task B TS-2 a=1.60", "Task B pred a=1.45", "Task B obs a=1.45", "Task B pred a=1.60", "Task B obs a=1.60", "Task B crossed")}.

**Say:** "A second registered test, at activation values the fit never saw and on fresh training sets, predicted the
median residual from the Adam timescale relationship within tolerance at both values, and each run's own threshold
again beat the population threshold."

**Do not say:** that this establishes the timescale account as the residual's mechanism (it rests on a post hoc fit
whose within-a correlations are weak, and it did not carry over to width 2), or that the a = 1.60 prediction is an
interpolation.
"""


def wp22():
    """WP-22: the mechanism figure (Task C)."""
    ms = pd.read_csv(RESULTS / "mechanism_w1_stats.csv").set_index("Unnamed: 0")
    return f"""
## WP-22. The mechanism figure (Task C; for the submission, main text)

File: `results/figures/v5/mechanism_w1.pdf`. Caption entry in `results/figures/v5/captions.md` ("mechanism_w1"), with
its one-sentence message, population and every number. Producer: `src/mechanism_w1_figure.py`. It passes the v5 audit
(5.49 × 2.38 in, smallest glyph 8 pt).

**What it shows.** For width 1 at a = 1.30 and 1.50, the conditional minimiser's |w₁| against R/R_glob (log scale).
- It starts near the class-mean optimum α* = 1.79 at small scale: {ms.loc[1.3, "w1_at_smallest"]:.3f} and
  {ms.loc[1.5, "w1_at_smallest"]:.3f} at the smallest scale shown.
- It falls below the placement bound a/1.4 before R_glob. The bound is necessary, not sufficient.
- Placement switches on at R_glob, where all {int(ms.loc[1.3, "n_cross"])} + {int(ms.loc[1.5, "n_cross"])} free-training
  crossings sit, every one with |w₁| < a/1.4.
- It is the picture of WP-12: the loss moves from rewarding the class-mean gap to rewarding the worst-case gap.
IDs: {_id("Task C figure audit", "Task C w1 at smallest a=1.30", "Task C crossings below bound")}.

**Say:** "As output scale grows, the conditional minimiser's first-layer weight leaves the class-mean optimum and
crosses below the placement bound; placement switches on at R_glob, where training crosses."

**Do not say:** that the minimiser path is certified (it is a validated search; only R_glob is certified), or that
crossing the bound a/1.4 is the switch (the bound is necessary, not sufficient).
"""


def wp23():
    """WP-23: is the timescale relationship consistent with the within-a interventions? (Item 1, POST HOC)"""
    t = pd.read_csv(RESULTS / "timescale_consistency" / "arms.csv")
    sm = json.loads((RESULTS / "timescale_consistency" / "summary.json").read_text())
    miss = t[~t.within_tol]
    rows = "\n".join(f"| {r.test} | {r.a:.2f} | {r.arm} | {r.n} | {r.median_ratio:.4f} | {r.pred:.4f} | {r.obs:.4f} | ±{r.tol:.4f} | "
                     f"{'yes' if r.within_tol else '**no**'} |" for r in t.itertuples())
    return f"""
## WP-23. Is the timescale relationship consistent with the within-a interventions? (Item 1; POST HOC; for the submission)

**Update (WP-24).** The no-fit law r = κ(a)·χ (math note §13; derived after the fitted relationship used here was known)
accounts for the same residuals with no intercept and no fitted parameter. The verdicts in this section were scored
with the fitted line and stand as registered. Where the paper states the account, use κ(a)·χ (WP-24).

Producer: `src/timescale_consistency.py` → `timescale_consistency/runs.csv`, `arms.csv`, `summary.json`.
- **Frozen relationship:** residual = α + β·ratio, with α = {sm["frozen_fit"]["alpha"]:.4f} and
  β = {sm["frozen_fit"]["beta"]:.3f}. This is the file used by the SGD extension and Task B (SHA-256 16792946…).
- **Runs:** every crossing run of the first lag test (all φ arms), the deconfounded lag test (primary and t\* rules,
  all φ arms) and Block 4b (all arms), {sm["runs"]} in total.
- Each run was replayed to its recorded crossing, and every replay reproduces its crossing exactly.
- For each arm, the predicted median residual (from each run's own ratio at crossing) is compared with the observed
  one, at the registered tests' tolerance, max(0.01, 0.25·|pred|).

| test | a | arm (φ or intervention) | n | median ratio | predicted | observed | tolerance | within |
|---|---|---|---|---|---|---|---|---|
{rows}

- **{sm["arms_within_tol"]} of {sm["arms"]} arms are within tolerance.** The {len(miss)} misses are exactly the φ = 0.25 arms of
  the first lag test and of the t\* rule, at both a.
- There, slowing w₂'s learning rate from early on lowers the ratio at crossing 3–4.5×. The residual falls in the
  predicted direction but by more than predicted: observed {miss.obs.min():.3f}–{miss.obs.max():.3f} against predicted
  {miss.pred.min():.3f}–{miss.pred.max():.3f}.
- **Where the interventions barely move the ratio,** the relationship predicts every arm. That covers the deconfounded
  primary rule (φ applied from 0.7 of the threshold), where Adam's normalisation absorbs the change, so the ratio moves
  by at most 20% and the residual by 11–14%. It also covers all four Block 4b arms, where the ratio does not move.
- **Slopes:** within a, over all arms, the residual–ratio slope is {sm["slope_within_a_all"]["1.30"]:.2f} (a = 1.30) and
  {sm["slope_within_a_all"]["1.50"]:.2f} (a = 1.50), against the pooled fit's {sm["frozen_fit"]["beta"]:.2f}.
- **Partial correlation:** the Spearman partial correlation of residual and ratio, controlling for a, is
  {sm["partial_spearman_all"]:.2f} over all arms and {sm["partial_spearman_fitdata"]:.2f} on the fit's own data.
- **Validity note (Block 4b; disclosed, no verdict changed).** In the registered Block 4b run, the arms shared optimiser-state
  tensors, which the preceding arm's continuation mutated in place (`load_state_dict` shares them).
  - The **teleport** arm therefore started from the control run's end-of-run Adam state, not the state at the switch.
  - The **teleport_reset** arm started from the reset run's end state, not zeroed moments.
  - Control and reset are unaffected.
  - Here the arms were replayed *as run*, which reproduces them exactly. The corrected rerun leaves 4b-ID and 4b-OM
    FAIL (WP-10).
IDs: {_id("Item 1 arms within tolerance", "Item 1 misses are the phi=0.25 arms", "Item 1 within-a slope a=1.30", "Item 1 within-a slope a=1.50", "Item 1 partial Spearman", "Item 1 replays reproduce")}.

**Statement (in between, stated exactly).**
- The frozen relationship predicts the residual across activation values and optimisers (two registered prospective
  tests). Within a, it predicts every intervention arm that leaves the timescale ratio near its unperturbed value.
- It gets the *direction* of a large within-a slowdown right but not its *size*. When the output rate is slowed from
  early on, the residual falls further than the linear relationship predicts; the within-a dependence is steeper.
- So the account is quantitatively supported across a and optimisers and for small within-a changes, not for large
  within-a changes of the ratio.

**Say:** "A single linear relation between the residual and the ratio of output growth to branch relaxation predicts
the residual across activation values and optimisers (two registered prospective tests) and for within-a interventions
that leave the ratio nearly unchanged. It captures the direction but underestimates the size of the effect when the
output rate is slowed substantially (post hoc)."

**Do not say:** that the relation holds quantitatively within a for arbitrary interventions, that it is the residual's
mechanism, or that the lag tests confirm it. The registered lag-test predictions L2, L1′ and L2′ failed, and those
verdicts stand.
"""


def wp24():
    """WP-24: the lag law r = κ(a)·χ (Track 1A, derived after the fitted relationship was known; Track 1B, registered ramp)."""
    L = RESULTS / "lag_law"
    A = pd.read_csv(L / "compare_arms.csv"); Pa = pd.read_csv(L / "compare_per_a.csv")
    kb = pd.read_csv(L / "kappa_by_winding.csv"); wc = pd.read_csv(L / "winding_check.csv")
    pr = pd.read_csv(L / "predictions.csv"); rx = pd.read_csv(L / "relaxation.csv")
    V = json.loads((RESULTS / "ramp" / "verdicts.json").read_text())
    PH = json.loads((RESULTS / "ramp" / "posthoc_branch.json").read_text())
    kap = {(round(r.a, 2), int(r.k)): (r.kappa_adam, r.kappa_sgd) for r in kb.itertuples()}
    used = {1.3: -1, 1.45: 0, 1.5: 0, 1.6: 0}
    ktab = "\n".join(f"| {a:.2f} | {k:+d} | {kap[(a, k)][0]:.2f} | {kap[(a, k)][1]:.2f} | {Pa.set_index(Pa.a.round(2)).loc[a, 'ratio_obs_to_pred']:.2f} |"
                     for a, k in used.items())
    wtab = "\n".join(f"| {r.a:.2f} | {int(r.k):+d} | {r.chi:.3f} | {r.predicted_r:+.4f} | {r.measured_r:+.4f} |" for r in wc.itertuples())
    arms = A.copy(); arms["a"] = arms.a.round(2)
    grp = arms.groupby("set").agg(n=("within", "size"), within=("within", "sum"), lo=("obs_over_pred", "min"), hi=("obs_over_pred", "max"))
    atab = "\n".join(f"| {st} | {int(g.n)} | {int(g.within)} | {g.lo:.2f}–{g.hi:.2f} |" for st, g in grp.iterrows())
    low = arms.arm.astype(str).eq("0.25") & arms["set"].isin(["lag1", "lag2-tstar"])
    q = arms[~low]; q25 = arms[low]
    def vrow(x, lab):
        return (f"| {x['a']:.2f} | {int(x['winding']):+d} | {x['opt']} | {lab} | {x['slope_obs_on_pred']:.2f} **{x['R1']}** | "
                f"{x['signed_spearman_gamma_median']:.2f} **{x['R2']}** | {x['slowest_median_r']:+.4f} **{x['R3']}** |")
    fam = lambda x: "R5 (sign test)" if (round(x["a"], 2) == 1.3 and int(x["winding"]) == 0) else "R1–R3"
    vt = "\n".join(vrow(x, fam(x)) for x in V["settings"])
    pt = "\n".join(vrow(x, fam(x)) + f" {x['frac_branch_off_own_1pct']:.3f} |" for x in PH)
    cm = "\n".join(f"| {x['a']:.2f} | {int(x['winding']):+d} | {x['opt']} | " + " / ".join(f"{o:+.4f}" for o in x["cell_medians_obs"]) + " | "
                   + " / ".join(f"{p_:+.4f}" for p_ in x["cell_medians_pred"]) + " |" for x in PH)
    r4 = V.get("R4")
    if isinstance(r4, list):
        r4t = "\n".join(f"| {x['a']:.2f} | {x['n_crossed_0.01']} / {x['n_crossed_0.005']} / {x['n_crossed_0.0025']} | "
                        f"{x['median_0.01']:+.4f} / {x['median_0.005']:+.4f} / {x['median_0.0025']:+.4f} | ±{x['tol']:.4f} | **{x['R4']}** |"
                        for x in r4)
        r4s = ("| a | crossed (η = 0.01 / 0.005 / 0.0025) | median residual | tolerance | R4 |\n|---|---|---|---|---|\n" + r4t)
    else:
        r4s = f"R4: {r4}."
    n_m = int(pr.mirror.sum())
    sw2 = pd.read_csv(RESULTS / "ramp2" / "design_sweep.csv")
    slow = sw2[sw2.gamma <= 1.8e-4]; fast = sw2[sw2.median_eta_lambda <= 0.5]
    r2s = {"slow_el_min": float(slow.median_eta_lambda.min()), "slow_el_max": float(slow.median_eta_lambda.max()),
           "fast_cross_max": int(fast.n_crossed.max())}
    ac = pd.read_csv(RESULTS / "ramp" / "adam_contrast.csv"); ac["a"] = ac.a.round(2)
    rp = ac[ac.source.str.startswith("ramp")].set_index("a"); fe = ac[ac.source.str.startswith("free")].set_index("a")
    vr = [float(fe.loc[a, c] / rp.loc[a, c]) for a in (1.3, 1.5) for c in ("sqrt_v_w1", "sqrt_v_b1", "sqrt_v_b2")]
    gr = [float(rp.loc[a, "growth_per_step"] / fe.loc[a, "growth_per_step"]) for a in (1.3, 1.5)]
    adam_par = (
        "Three things could differ between the Adam ramp and free Adam training, where the law held (Track 1A): the "
        "measured preconditioner, the growth rate and the branch. The data single out the **preconditioner**.\n"
        f"- **Preconditioner.** At crossing, Adam's √v̂ is {min(vr):.0f}–{max(vr):.0f}× smaller in the ramp than in free "
        "training, per coordinate and at both a. The 4,000-step warm-up holds the hidden coordinates at a stationary point, "
        "where the gradient vanishes, so v̂ decays; in free training v̂ still carries the larger gradients of the approach.\n"
        "  - With P = 1/(√v̂ + ε), the frozen-P relaxation time 1/(ηλ_min(P^{1/2}HP^{1/2})) is "
        f"{rp.relax_steps.min():.2f}–{rp.relax_steps.max():.2f} steps in the ramp, against "
        f"{fe.relax_steps.min():.1f}–{fe.relax_steps.max():.1f} steps in free training.\n"
        "  - A relaxation time below one step means ηλ_min > 1. The linear update with that P would overshoot, so the "
        "linearisation behind the law (small ηλ, P fixed over the relaxation) does not describe the ramp. Adam there is in "
        "its self-normalising regime: steps of about η per coordinate, with v̂ adapting to the ramp's own gradients.\n"
        f"  - Consistently, the ramp reaches its crossing {rp.steps_growth_to_cross.min():.0f}–"
        f"{rp.steps_growth_to_cross.max():.0f} steps after growth starts (median), within v̂'s 1,000-step memory. Free "
        f"training takes {fe.steps_growth_to_cross.min():.0f}–{fe.steps_growth_to_cross.max():.0f} steps.\n"
        f"- **Growth rate.** The ramp's rate is {min(gr):.1f}–{max(gr):.1f}× the free-training rate at crossing (medians), "
        "the same order. SGD in the ramp covers far wider rates and follows the law, so the rate alone does not explain "
        "Adam's failure.\n"
        "- **Branch.** The ramp runs sit on the same windings as free training (k = −1 at a = 1.30, k = 0 at a = 1.50). "
        "They are all on the non-mirror branch, where κ is the same. The post hoc scoring already uses each run's tracked "
        "branch, and SGD passes on the same branches.\n"
        "- **So:** the law's Adam form needs a preconditioner that is stationary and small enough that ηλ_min ≪ 1 over the "
        "relaxation. Free training satisfies this; the ramp, started at a stationary point, does not. This is a "
        "limit of the frozen-preconditioner linearisation (math note §13.3(iv)(c)), identified after scoring.")

    def rng(opt, sl, a=None, k=None):
        v = []
        for x in PH:
            if x["opt"] != opt or (a is not None and (round(x["a"], 2) != a or int(x["winding"]) != k)):
                continue
            v += list((np.array(x["cell_medians_obs"]) / np.array(x["cell_medians_pred"]))[sl])
        return f"{min(v):.2f} to {max(v):.2f}" if min(v) < 0 else f"{min(v):.2f}–{max(v):.2f}"
    return f"""
## WP-24. The lag law r = κ(a)·χ: a no-fit constant, and a registered ramp test (Tracks 1A and 1B; for the submission)

Producers: `src/lag_law.py` → `lag_law/` (κ, predictions, comparison, winding check, relaxation times);
`src/ramp.py` → `ramp/` (design, runs, scores, post hoc). Math note §13 has the derivation.

**Label for everything in Track 1A: derived after the fitted relationship (residual = α + β·ratio, WP-16/21/23) was known.**

### 1A. What was committed before any comparison (b6ae433)

- **The law.** Hidden coordinates θ = (w₁, b₁, b₂) track the conditional stationary branch θ\\*(s) of the loss at fixed
  output scale s. To first order the steady lag gives

  r = (s_c − s\\*)/s\\* = κ(a)·χ, with χ = (ṡ/s\\*)/(ηλ_min) and κ(a) = λ_min·[∇G·(PH)⁻¹θ\\*′]/[∇G·θ\\*′].

  - H is the Hessian at the switch s\\* and θ\\*′ the branch tangent.
  - P = I for SGD. For Adam, P = diag(1/(√v̂ + ε)) is the median preconditioner at crossing.
  - λ_min = λ_min(P^{{1/2}}HP^{{1/2}}).
  - κ has no free parameter. χ is the paper's timescale ratio, up to s\\* against s_c (a factor 1 + r).
- **Winding.** f_a(t + 2π) = f_a(t) + 2π, so b₁ has 2π copies with the same loss and threshold but a different output-bias
  drift. κ depends on the copy k. Every existing run sits on k = −1 at a = 1.30 and on k = 0 elsewhere.
  - This was found, and checked with controlled population ramps, before any comparison (`winding_check.csv`):

| a | k | χ | predicted r | measured r |
|---|---|---|---|---|
{wtab}

- **Values and predictions.** Per-run predictions were committed with SHA-256 af6bb137… for {len(pr):,} runs: 32
  intervention arms, SGD at two a, and Task B at two a. {n_m} of these runs sit on the mirror branch, where κ is
  unchanged.

| a | k | κ_Adam | κ_SGD | observed/predicted slope (post hoc, through the origin) |
|---|---|---|---|---|
{ktab}

### 1A. Result (899850d; no refit)

| set | arms | within max(0.01, 0.25·pred) | observed/predicted per arm |
|---|---|---|---|
{atab}

- **{int(arms.within.sum())} of {len(arms)} arms are within tolerance.**
  - The four φ = 0.25 arms of the first lag test and the t\\* rule (w₂ slowed from early on) are at
    {q25.obs_over_pred.min():.2f}–{q25.obs_over_pred.max():.2f}. They are within tolerance only through the 0.01 floor.
  - The other {len(q)} arms are at {q.obs_over_pred.min():.2f}–{q.obs_over_pred.max():.2f}. This corrects the 899850d
    commit message, which said 0.99–1.13 and called the four low arms "the φ = 0.25 arms"; the deconfounded primary
    rule's φ = 0.25 arms are at 1.03–1.07.
- **Per a (post hoc through-origin fit):** observed slope / κ = {", ".join(f"{r.ratio_obs_to_pred:.2f} ({r.a:.2f})" for r in Pa.itertuples())}.
  That is within 30% at every a. The law over-predicts by 9–18%.
- **Timescales.** The relaxation time 1/(ηλ_min) is {rx.relaxation_steps.min():.1f}–{rx.relaxation_steps.max():.1f}
  steps; Adam's momentum time is 10 steps. Momentum does not change the steady lag (math note §13.3).

### 1B. The ramp experiment (registered c4b4c6d; own thresholds frozen 83f66f5 and 15b4744 before any crossing was read)

**Design.**
- Output scale forced to s = s₀e^{{γt}} from 0.5·s\\* after a 4,000-step warm-up. Hidden coordinates and output bias
  train normally (Adam lr 0.01 or SGD lr 0.3).
- 40 fresh seeds per cell, 6 γ cells per setting.
- Every run starts on the population branch at winding k:
  - (1.30, k = −1) and (1.50, k = 0) are the natural windings (R1–R3);
  - (1.30, k = 0) has κ < 0, so the law predicts crossing *before* the threshold (R5).
- Predictions: r_pred = κ_k·χ. SGD's χ is fully a priori from γ, η and the landscape. Adam's χ uses each run's measured
  v̂ at crossing.
- Observed r uses each seed's own-sample global threshold.
- **Rules:**
  - R1: through-origin slope in [0.7, 1.3];
  - R2: sign(κ)·Spearman(γ, cell median) ≥ 0.9;
  - R3: |slowest-cell median| < 0.005.
- **Adam's γ grid.** It was calibrated on pilot seeds from predictions only, and targets predicted lags of only
  0.0005–0.006. Adam cannot follow faster ramps: its step is capped near η per coordinate, and the pilot runs stopped
  crossing.

**Registered verdicts (all 1,440 runs crossed; none placed during warm-up; no run changed winding).**

| a | k | optimiser | rule set | R1: slope | R2: signed Spearman | R3: slowest median |
|---|---|---|---|---|---|---|
{vt}

**Diagnosis (after scoring).**
- Every γ cell contains the same subpopulation of seeds at a large static offset: 5% quantiles ≈ −0.25 and 95% ≈ +0.09
  in every cell.
- For these seeds the own-sample *global* threshold is not the switch of the branch the ramp forces. For 35% of seeds at
  a = 1.30 and 68% at a = 1.50, that branch's own-sample switch differs from s_own by more than 1%.
- Free training is not forced onto a branch; the ramp is. The offsets pull the pooled slopes (R1) and, at a = 1.50, the
  medians (R3).

**POST HOC (labelled): the same rules against each seed's tracked-branch own-sample switch** (`ramp.posthoc_branch`;
the last column is the fraction of seeds whose branch switch is > 1% from s_own).

| a | k | optimiser | rule set | R1 | R2 | R3 | branch ≠ own |
|---|---|---|---|---|---|---|---|
{pt}

Cell medians, post hoc (observed vs predicted, slowest to fastest γ):

| a | k | optimiser | observed | predicted |
|---|---|---|---|---|
{cm}

- **SGD:** observed/predicted cell medians are {rng("sgd", slice(0, 4))} in the four slowest cells,
  {rng("sgd", slice(4, 5))} in the fifth (κχ = 0.04) and {rng("sgd", slice(5, 6))} in the fastest (κχ = 0.1, where the
  linearisation starts to fail; math note §13.3(iv)).
  - The sign test holds: on the k = 0 copy at a = 1.30 the crossings come *early*, as predicted.
- **Adam:** the predicted lags are ≤ 0.006, and the cell medians do not follow them. Observed/predicted is
  {rng("adam", slice(0, 6), 1.3, -1)} (a = 1.30, k = −1), {rng("adam", slice(0, 6), 1.3, 0)} (k = 0) and
  {rng("adam", slice(0, 6), 1.5, 0)} (a = 1.50).
  - The post hoc R1 pass at a = 1.50 comes from the pooled per-run slope, not from agreement at the cell level.
  - Adam's frozen-preconditioner linearisation is not supported at these small lags. Its v̂ adapts during the ramp
    (math note §13.3(iv)(c)).

### Why Adam differs in the ramp (POST HOC; `ramp.adam_contrast` → `ramp/adam_contrast.csv`)

{adam_par}

### Follow-up requested after scoring: the Adam ramp from initialisation (NOT registered: infeasible by its design rules)

- **Design, fixed before any registered run** (`src/ramp2.py`; `ramp2/design_sweep.csv`; pilot seeds 869,100–869,107,
  predictions only):
  - no warm-up; standard initialisation; s = 0.5·e^{{γt}};
  - each run to be scored against its tracked-branch switch;
  - γ admissible only if ≥ 6 of 8 pilot runs cross **and** the median ηλ_min(P^{{1/2}}HP^{{1/2}}) at crossing is ≤ 0.5,
    so that the frozen-preconditioner law can apply.
- **Outcome of the design sweep:** no admissible γ at either a.
  - Where most runs cross (γ ≤ 1.8e-4), v̂ collapses during the slow ramp and ηλ_min is {r2s["slow_el_min"]:.1f}–{r2s["slow_el_max"]:.1f}.
  - Where ηλ_min ≤ 0.5 (γ ≈ 1e-3), at most {r2s["fast_cross_max"]} of 8 runs cross.
- So no forced exponential ramp at these rates gives a test in the law's Adam regime, and nothing was registered or
  run. Free training meets the condition because its growth is not imposed (relaxation 7.8–15 steps; R4, 1A).
- **Say:** "Adam's version of the law needs a stationary preconditioner. A forced ramp either lets it collapse (slow ramps)
  or outruns the unit (fast ramps), so the law's Adam form is supported by free training only."

### R4. Free Adam training at three learning rates (registered with 1B; seeds 860,100–860,139)

{r4s}

- The law predicts η-invariance. Growth ṡ and relaxation ηλ both scale with η, so χ does not change.
- A pass means that the lag does not vanish in the gradient-flow limit of free training.

### Notation for the main text

- κ(a) is the lag constant. The plan's notation table uses κ(a). The existing κ₀ (the limiting-cubic constant in Block
  3's window design) must then get another symbol in the main text (suggestion: ν₀), and Ĝ/D is not called κ anywhere
  in the main text.
- χ = (ṡ/s\\*)/(ηλ_min). The tests used each run's measured ratio at crossing, which divides by s_c instead of s\\*.
  State this once.
- The fitted line residual = α + β·ratio (WP-16, WP-21, WP-23) stays as the registered basis of those verdicts
  (EXT, TS-1). Where the paper states the account, use r = κ(a)χ, labelled "derived after the fitted relationship was
  known". It has no intercept and no fitted parameter.

### Say / Do not say

**The κ outcome (1A) is "within 30%"** (post hoc slope/κ = 0.82–0.91 at every a; 36/36 arms within tolerance).

- **Say (observed outcome, within 30%):** "A first-order tracking analysis gives the lag constant κ(a) with no fitted
  parameter. Computed before comparison, it predicts the median residual of all 36 existing arms within tolerance, and
  the observed slope is within 30% of κ(a) at every a (the analysis was derived after a fitted relationship was
  known)."
- **If it had been within a factor of two (not observed),** the sentence would be: "predicts the residual's scale within
  a factor of two".
- **If worse (not observed),** it would be: "does not predict the residual quantitatively".
- **Say (ramp, registered):** "In a registered test that forces the output scale to grow at set rates, the lag increases
  monotonically with the rate in every setting (R2 passes in 5 of 6) and vanishes at the slowest rate at a = 1.30. The
  pooled magnitude criterion (R1) passes only for SGD at a = 1.30 on the natural winding. The failures trace to seeds
  whose own-sample global threshold is not the switch of the branch the ramp forces."
- **Say (ramp, post hoc, labelled):** "Measured against the switch of the branch each run actually tracks, SGD's
  crossing lag matches κ(a)χ with no fitted parameter across two decades of rate: within 6% in the four slowest rate
  cells and within 30% in every cell, including the predicted *early* crossing on the winding copy with negative κ.
  Adam's lags in the ramp do not follow the law."
- **Do not say:**
  - that the ramp test passed as registered;
  - that κ was predicted before the fitted relationship was known;
  - that the law holds for Adam in the ramp (its cell medians do not follow κχ, even post hoc);
  - that the law holds in the nonlinear regime κχ ≳ 0.1;
  - that the lag law explains width 2 (WP-15; Track 2 below).
IDs: {_id("1A arms within tolerance", "1A slope/kappa a=1.30", "1A slope/kappa a=1.50", "1B registered R1 SGD 1.30 k=-1", "1B post hoc R1 SGD 1.50", "1B branch off own a=1.50")}.
"""


def wp25():
    """WP-25: GELU, SiLU and Mish at width 1 (Track 3A)."""
    D = RESULTS / "act_general"
    gh = json.loads((D / "ghat.json").read_text()); cv = json.loads((D / "criterion_verdicts.json").read_text())
    ts = pd.read_csv(D / "training_scores.csv"); ph = pd.read_csv(D / "posthoc_training.csv").set_index("act")
    names = {"gelu": "GELU", "silu": "SiLU", "mish": "Mish"}
    br = {a: json.loads((D / f"bracket_{a}.json").read_text()) for a in names}
    kp = {a: json.loads((D / f"kappa_{a}_frozen.json").read_text()) for a in names}
    t1 = "\n".join(f"| {names[a]} | {gh[a]['Ghat_nm']['G_lo']:.6f} | ({gh[a]['Ghat_nm']['w1']:.3f}, {gh[a]['Ghat_nm']['b1']:.3f}) | "
                   f"{cv[a]['statuses']['0.05']} / {cv[a]['statuses']['0.1']} | **{cv[a]['verdict']}** | [{br[a]['s_lo']:.4f}, {br[a]['s_hi']:.4f}] | "
                   f"{kp[a]['s_star']:.4f} | {kp[a]['kappa_adam']:+.4f} |" for a in names)
    tt = "\n".join(f"| {names[r.act]} | {r.arm.replace('_', ' ')} | {r.runs} | {r.crossing_runs} | **{r['T-a']}**"
                   + (f" ({r.frac_at_or_above_lo:.3f})" if r['T-a'] != "UNRESOLVED" else "") + f" | **{r['T-b']}**"
                   + (f" (obs {r.obs:+.4f}, pred {r.pred:+.4f}, median χ {r.median_chi:.3f})" if r['T-b'] != "UNRESOLVED" else "") + " |"
                   for _, r in ts.iterrows())
    pt = "\n".join(f"| {names[a]} | [{ph.loc[a, 'r_q25']:+.3f}, {ph.loc[a, 'r_q75']:+.3f}] | {ph.loc[a, 'frac_abs_r_le_0.10']:.2f} | "
                   f"{int(ph.loc[a, 'n_early_below_half_s_glob'])} | {ph.loc[a, 'spearman_r_chi']:+.2f} | {ph.loc[a, 'noncross_median_w2_final']:.2f} |" for a in names)
    sec = ts[ts.arm == "secondary_pooled_200"].set_index("act")
    S2 = json.loads((D / "posthoc2_summary.json").read_text()); sw = S2["sine_width1"]
    A2 = {x["act"]: x for x in S2["acts"]}; ga, sa, ma = A2["gelu"], A2["silu"], A2["mish"]
    gf = json.loads((D / "gelu_rule_feasibility.json").read_text())
    import re as _re
    notes = ga["no_switch_notes"]
    n_lost = sum(int(m) for m in _re.findall(r"continuation lost at s=[0-9.]+: (\d+)", notes))
    n_nosign = sum(int(m) for m in _re.findall(r"no sign change within 400 steps: (\d+)", notes))
    n_und = sum(int(m) for m in _re.findall(r"gap undecided at the cell cap: (\d+)", notes))
    assert n_lost + n_nosign + n_und == ga["n_no_switch_on_branch"]
    ns = f"no sign change within 400 continuation steps in {n_nosign}, continuation lost in {n_lost}, gap undecided at the cell cap in {n_und}"
    vt = "\n".join(f"| {names[a]} | {A2[a]['chi_median']:.4f} [{A2[a]['chi_q25']:.4f}, {A2[a]['chi_q75']:.4f}] | "
                   f"{100 * A2[a]['frac_chi_le_sine_arm_median_max']:.0f}% | {100 * A2[a]['frac_chi_le_sine_run_max']:.0f}% | "
                   f"{'yes' if A2[a]['validity_met'] else '**no**'} |" for a in names)
    bt = "\n".join(f"| {names[a]} | {A2[a]['with_branch_switch']} of {A2[a]['crossing_runs']} | {A2[a]['obs_branch_median']:+.4f} | "
                   f"{A2[a]['pred_median']:+.4f} | ±{A2[a]['tol']:.2f} | {'yes' if A2[a]['within'] else '**no**'} | "
                   f"{100 * A2[a]['frac_at_or_above_branch']:.0f}% | {100 * A2[a]['frac_abs_rb_le_0.01']:.0f}% | "
                   f"{A2[a]['branch_over_pop_q10']:.2f}–{A2[a]['branch_over_pop_q90']:.2f} |" for a in names)
    nc = ", ".join(f"{int(sec.loc[a, 'runs'] - sec.loc[a, 'placed_at_init'] - sec.loc[a, 'crossing_runs'])} ({names[a]})" for a in names)
    return f"""
## WP-25. Outside the sine family: GELU, SiLU and Mish at width 1 (Track 3A; for the submission)

Producers: `src/act_general.py` (steps 0–4) and `src/act_posthoc.py` → `act_general/`. Summary: `results/act_summary.md`.
Registrations: criterion `act_criterion_registration.md` (b707e86, amendments a251e80 and cf2d68b, both implementation
only and made before any criterion result); training `act_training_registration.md` (ad99053; κ frozen with SHA-256
before any run). Labels: registered, validated (not certified), post hoc.

**Step 0 (validated).** The generalised machinery reproduces the certified f_a bracket at a = 1.30: the switch is in
[4.95796, 4.95835], inside [4.95, 4.9625].

**Steps 1–3 (validated; the criterion registered).**
- A single unit places and solves for all three activations. The dip sits under the inner window (σ = +1). A midpoint
  output bias is sign-correct and a wrong bias is not.
- Differences from the sine case:
  - the dip has a fixed size, so there is no winding;
  - the ramp is not odd, so the class-mean maximiser is not attained.
- The registered criterion: a switch is predicted iff the validated conditional minimiser is unplaced at s = 0.05 and
  at s = 0.1.

| activation | Ĝ | maximiser (w₁, b₁) | status at s = 0.05 / 0.1 | criterion (registered) | validated switch bracket in s | s\\* (tracked branch) | κ_Adam |
|---|---|---|---|---|---|---|---|
{t1}

- **Post hoc, disclosed:** every small-scale minimiser is a one-sided ramp. It is "placed" in exact arithmetic by a
  margin of 10⁻¹⁵³ or smaller, and its gap is 0 in double precision. SiLU and Mish are "undetermined" because of this.
- Brackets are validated (restart ladder, independent CMA-ES, audit), not certified.

**Step 4: training (registered).**
- Adam lr 0.01, budget 32,000, fresh seeds.
- T-a: ≥ 90% of crossing runs at s ≥ s_lo.
- T-b: median residual within max(0.01, 0.25|pred|) of median κχ.
- A cell with < 30 crossings is UNRESOLVED.
- The 200-seed arm was registered before any run, after calibration showed crossing rates near 50%.

| activation | arm | runs | crossing | T-a | T-b |
|---|---|---|---|---|---|
{tt}

**Post hoc** (`posthoc_training.csv`). The residual is measured against the *population* threshold.

| activation | IQR of r | fraction with \\|r\\| ≤ 0.10 | crossings below ½·s_glob | Spearman(r, χ) | final \\|w₂\\| of non-crossing runs (median) |
|---|---|---|---|---|---|
{pt}

- About a third of runs never cross ({nc}, of 200). They stall near a trivial point with small |w₂|.
- SiLU's and Mish's crossings happen at χ ≈ 0.2, which is outside the regime where the linear lag law applies (math note
  §13.3(iv)).
- Own-sample thresholds were not computed. Each training set has 400 points (200 per class); `act_summary.md` says
  "200 points per run", which means 200 per class.

**POST HOC checks on the same runs (author's request; `src/act_posthoc2.py` → `act_general/posthoc2_*`; no new runs;
the registered verdicts above stand as scored).**

*(1) Validity: χ at crossing against the width-1 sine range.*
- The quantity is the same growth-to-relaxation ratio as for the sine runs, with each run's own Adam preconditioner.
- The width-1 lag law was verified (Track 1A) on runs with χ = {sw["chi_run_min"]:.4f}–{sw["chi_run_max"]:.3f} (95% of runs
  ≤ {sw["chi_run_q95"]:.3f}). The largest arm-median χ at which it held is {sw["chi_arm_median_max"]:.4f}.
- **The validity condition used here:** the median χ at crossing is ≤ {sw["chi_arm_median_max"]:.4f}.

| activation | median χ [IQR] | runs with χ ≤ {sw["chi_arm_median_max"]:.4f} | runs with χ ≤ {sw["chi_run_max"]:.3f} | condition met |
|---|---|---|---|---|
{vt}

*(2) Crossings scored against each run's tracked-branch switch* (Newton continuation on the run's own 400-point sample
from its crossing state, as for the ramp).

| activation | runs with a branch switch | median r vs branch | median κχ | tolerance | within | at or above the branch switch | \|r\| ≤ 0.01 | branch switch / population threshold, 10–90% |
|---|---|---|---|---|---|---|---|---|
{bt}

- **GELU (condition met):**
  - Against its tracked branch, the crossing follows the lag law: median r {ga["obs_branch_median"]:+.4f} against
    κχ {ga["pred_median"]:+.4f}.
  - {100 * ga["frac_at_or_above_branch"]:.0f}% of runs cross at or above their branch switch, and
    {100 * ga["frac_abs_rb_le_0.01"]:.0f}% within 1% of it.
  - The registered tests compared crossings with the population threshold. Each run's own branch switch (on its own
    400-point sample) lies {ga["branch_over_pop_q10"]:.2f}–{ga["branch_over_pop_q90"]:.2f}× the population threshold (10–90%).
    Since the crossings sit within 1% of those switches, that spread is what the registered tests measured.
  - The other {ga["n_no_switch_on_branch"]} GELU crossings all happen early (s ≤ {ga["no_switch_s_cross_max"]:.2f}, against a population threshold of
    6.64), from states far from the retained branch. Continuing their branch from the crossing finds no switch:
    {ns}.
- **Requested follow-up, a GELU prospective test with an early-scale branch rule: NOT registered (infeasible).**
  - The rule scale had to lie below the earliest crossing observed, s = {gf["earliest_crossing_s"]:.4f}. None of the
    {gf["n_runs"]} existing runs starts below it (median initial |w₂| is {gf["init_abs_w2_median"]:.2f}), so the only
    admissible "early scale" is initialisation.
  - There, the branch classification is final for only {gf["final_at_init_n"]} of {gf["n_crossing"]} crossing runs
    ({100 * gf["final_at_init_frac"]:.0f}%), against the required 95% (`src/gelu_rule_feasibility.py`).
  - Nothing was registered or trained.
- **SiLU and Mish (condition not met):**
  - χ at crossing is about 10× beyond the largest χ at which the sine law was verified.
  - Against the tracked branch, the residual has the wrong sign for κχ, and only {100 * sa["frac_abs_rb_le_0.01"]:.0f}% and
    {100 * ma["frac_abs_rb_le_0.01"]:.0f}% of runs cross within 1% of their branch switch.
  - This is outside the regime of the linear lag law (math note §13.3(iv)(a)), so these runs neither test nor
    contradict it.

**Say:**
- "For GELU, SiLU and Mish a single unit can solve the task, and the loss at fixed output scale has a validated
  unplaced-to-placed switch."
- "(Post hoc) For GELU, whose runs cross in the timescale regime where the sine lag law was verified, crossings track
  within 1% of each run's own branch switch and follow the lag law; the registered tests, which used the population
  threshold, measured the spread of those switches. For SiLU and Mish, output growth at crossing is about ten times too fast for the law to apply."
- "Free Adam training does not track it. Registered tests on 200 seeds per activation fail: 48–70% of crossings lie
  above the switch, and the median residual has the wrong sign or size for the lag law. About a third of runs never
  cross."
- "The scale-gating account is therefore specific to activations whose non-monotone part scales with the pre-activation
  (the sine family here). Its extension to practical activations is not supported by the registered tests."

**Do not say:**
- that the switch for these activations is certified;
- that the criterion predicted the switch for SiLU or Mish (it was undetermined);
- that training confirms scale gating outside the sine family;
- that the registered tests passed, or that the post hoc branch scoring was registered;
- that the lag law holds for SiLU or Mish (their runs are outside its validity condition).
IDs: {_id("3A GELU T-a", "3A SiLU T-b", "3A Mish crossing", "3A GELU bracket lo")}.
"""


def wp26():
    """WP-26: Track 4 writer inputs: citations, main-text notation, the checker paragraph, run populations."""
    t4 = (RESULTS / "track4_writer_inputs.md").read_text()
    cit = t4[t4.index("**Result: all six are INCLUDED."):t4.index("## 2. Notation")].rstrip().rstrip("-").rstrip()
    chk_par = t4[t4.index("> An independent checker"):t4.index("**Where each statement comes from:**")].rstrip()
    P = json.loads((RESULTS / "track4_populations.json").read_text())
    f32, f64, pl, pr = P["float32"], P["float64"], P["pooled"], P["paired"]
    a32, a64 = P["a1.02_float32"], P["a1.02_float64"]
    p32t = "\n".join(f"| {r['a']:.2f} | {int(r['solved'])} | {int(r['placement'])} | {int(r['bias'])} |" for r in P["float32_per_a"])
    return f"""
## WP-26. Citations, main-text notation, the checker paragraph, and run populations (Track 4; for the submission)

Sources: `results/track4_writer_inputs.md` (the full Track 4 report, including the appendix-only symbol list and the
list of symbol clashes) and `src/track4_populations.py` → `track4_populations.json` (every population number below).

### 1. Six citations: verified against the proceedings or iclr.cc pages

{cit}

### 2. Main-text notation

| symbol | meaning | where defined |
|---|---|---|
| s | output scale: \\|w₂\\| at width 1; ‖w₂‖₁ at width 2 | math note §1, §10 |
| R | output scale in gap units, R = sĜ(a)/2 (certified Ĝ_cert) | math note §1; WP-7 |
| Ĝ(a) | certified maximum class gap over first-layer parameters | math note "Three objects"; WP-7 |
| G | worst-case class gap of the hidden unit on the continuous windows (G > 0: placed) | math note §10–§11; WP-12 |
| R\\* ≡ R_glob | the conditional placement threshold: where the global minimiser of the loss at fixed output scale becomes placed (certified brackets) | math note "Three objects"; v4 Block 1 |
| R_solve | where that minimiser becomes sign-correct everywhere (certified brackets) | v4 Block 1; math note §2(b) |
| R_own | R_glob computed on a run's own training set (400 points, 200 per class), before training | v4 Block 1 (1d) |
| χ | timescale ratio (ṡ/s\\*)/(ηλ_min(P^{{1/2}}HP^{{1/2}})); the tests use each run's measured ratio at crossing (ṡ/s_c), a factor 1 + r apart | math note §13.1, §13.3(i); WP-24 |
| κ(a) | lag constant in r = κ(a)·χ, with no free parameter; depends on the winding of b₁ | math note §13; WP-24 |
| c₁ | first-order coefficient of R_glob in R-units: R_glob = R_glob^∞(1 + c₁ε + …), c₁ = 0.28523 | math note §8; WP-30 |
| c_s | first-order coefficient of the switch in s-units: \|w₂\|_glob = A*ε^(−3/2)(1 + c_sε + …), c_s = A′(0)/A* = 0.66215 | math note §8 |

- **Clashes to resolve in the main text.** κ₀ (the limiting-cubic constant in Block 3's window design) needs another
  symbol there, for example ν₀. "A\\*" is the limit switch in rescaled units, not R\\*.
- Every other symbol is appendix only. The full list and all clashes (α, β, U, C, K, D, φ, σ, λ, ε) are in the Track 4
  report, §2.
- That report's notes 1–3 predate Track 1A's commit: κ(a) and χ are now defined in math note §13 and WP-24.

### 3. AI Use Statement: the independent checker (one paragraph, under 150 words)

{chk_par}

- Every statement comes from the checker's docstring, `PREC = 80`, the manifest hash check, `certificate_audit.md` and
  WP-17's table.
- Keep "for exported certificates": the K, Krawczyk, positive-definite and ring checks take their inputs differently
  (Track 4 report, §3).
- The files do not say who wrote the checker or whether AI tools were used. The author adds that.

### 4. Run populations (a report, not a choice)

| population | runs | solved | placement failures | bias failures |
|---|---|---|---|---|
| 2,400 float32 | {f32["runs"]:,} | {f32["solved"]} | {f32["placement"]:,} | {f32["bias"]} |
| 2,400 float64 | {f64["runs"]:,} | {f64["solved"]} | {f64["placement"]:,} | {f64["bias"]} |
| 4,800 pooled | {pl["runs"]:,} | {pl["solved"]} | {pl["placement"]:,} | {pl["bias"]} |

- The float32 half is the width-1 sweep plus the refinement, run for run: {P["float32_equals_sweep"]["matched"]:,}
  matched, solve outcomes identical, and |w₂| identical (maximum difference {P["float32_equals_sweep"]["max_abs_diff_abs_w2"]:.1f}).
- **The float64 half is not a re-run of the float32 runs at higher precision.** At the same (a, seed):
  - the sign of w₁ agrees in {100 * pr["sign_w1_agree"]:.1f}% of {pr["pairs"]:,} pairs;
  - the failure class agrees in {100 * pr["failure_class_agree"]:.1f}%;
  - the solve outcome agrees in {100 * pr["solve_agree"]:.2f}%;
  - the median relative difference in |w₂| is {pr["median_rel_diff_abs_w2"]:.2f} (relative to float32; the Track 4
    report's 0.50 used another denominator).

  These are two independent initialisation draws labelled by precision (the per-dtype RNG problem recorded in the
  retraction, commit 1abaf09).
- **Which main-text number uses which population:**
  - The decomposition figure and its caption use the 4,800 pooled, labelled "(200 seeds × 2 precisions)". That label
    describes the tags, not two precisions of the same runs.
  - The metric check (`metric_check.py`; caption and VERIFIED_NUMBERS §6) uses the 2,400 float32.
  - The a = 1.02 numbers in WP-12 (A2) and math note §11.1 use the 2,400 float32: {a32["solved"]}/{a32["runs"]} solved,
    median terminal |w₂| {a32["median_abs_w2"]:.2f}, maximum {a32["max_abs_w2"]:.2f}.
    - The float64 half gives {a64["solved"]}/{a64["runs"]}, median {a64["median_abs_w2"]:.2f}, maximum
      {a64["max_abs_w2"]:.2f}.
  - WP-12's onset values (1.60 at 2,000 steps; 1.18, 1.06 and 1.03 at 8k, 32k and 128k) are constants in
    `harsh_review_a.py` taken from the onset analyses, not from either population.
- **So the main text currently mixes the two populations.** Whichever is chosen, the decomposition caption should not
  say "2 precisions" as if the runs were paired.

**The population the main text should use (final round): the 2,400 float32 runs.**
- **Why:**
  - They are one initialisation draw per seed at one precision, so they have no hidden pairing.
  - They are exactly the width-1 sweep plus refinement.
  - The metric check, the a = 1.02 numbers and VERIFIED_NUMBERS §6 already use them.
- **What changes:** only the decomposition figure and caption. Use `results/figures/v5/decomposition_float32.pdf`
  (captions.md entry "decomposition_float32") in place of `decomposition.pdf`.
- The pooled 4,800 can appear once, as a robustness line: "an independent second draw of 2,400 runs gives 440 / 1,633 / 327".
- **Decomposition** (`track4_float32_decomposition.csv`): {f32["solved"]} solved, {f32["placement"]:,} placement failures and {f32["bias"]}
  bias failures in {f32["runs"]:,} runs (200 per a):

| a | solved | placement failure | bias failure |
|---|---|---|---|
{p32t}
IDs: {_id("T4 pooled counts", "T4 sign agreement", "T4 a=1.02 float32")}.
"""


def wp28():
    """WP-28: the band task in R^d (Track 3B), from the agent's writer inputs (sections 1–6, verbatim)."""
    t = (RESULTS / "band_rd_writer_inputs.md").read_text()
    body = t[t.index("## 1. What was committed"):t.index("## 7. Files")].rstrip()
    body = body.replace("\n## ", "\n### ").replace("## 1. What", "### 1. What", 1)
    return f"""
## WP-28. Higher input dimension: the band task in R^d, d = 2 and 4 (Track 3B; for the submission)

Producer: `src/band_rd.py` → `band_rd/`. Registration: `results/band_rd_registration.md` (f19535b). Tests:
`tests/test_band_rd.py`. The per-step traces (`band_rd/traces_*.npz`, 135 MB) are regenerated by
`python -m src.band_rd run`; their SHA-256 are in `band_rd/traces_sha256.txt`. The text below is the Track 3B writer
input, sections 1–6.

{body}

IDs: {_id("3B P1 d=2", "3B P2a d=2", "3B P2b d=2", "3B d=4 UNRESOLVED", "3B post hoc obs/pred range")}.
"""


def wp27():
    """WP-27: width 2 (Track 2): 2B lag law, 2A stuck states, 2C validity condition (from the agent's writer inputs)."""
    t = (RESULTS / "track2_writer_inputs.md").read_text()
    body = t[t.index("## 2B. Does the lag law"):].rstrip()
    body = "\n" + body
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    sm = pd.read_csv(RESULTS / "width2_lag" / "summary.csv")
    tr = sm[(sm.arm == "T2-3") & (sm.subset == "chi <= width-1 max")].iloc[0]
    rc = json.loads((RESULTS / "width2_basins" / "reconcile_summary.json").read_text())
    return f"""
## WP-27. Width 2: the lag law where the account applies, the stuck states, and the validity condition (Track 2; POST HOC; for the submission)

Producers: `src/width2_lag.py` → `width2_lag/`; `src/width2_basins.py` → `width2_basins/`; `src/t2c.py` → `t2c/`
(commit 764c9dc). **Everything here is POST HOC, and no verdict changes: T2-3 FAIL, T2-3b UNRESOLVED, T2-3c UNRESOLVED,
T2-3d FAIL.** 2C was not registered: the author's calibration rule found no admissible rule scale.

**Coordinator note (checked against `width2_lag/summary.csv`).** The three full-speed runs called "tracking" below have
crossing states {tr.median_dist_c:.3f} from their branch (median). That is just above the module's 0.05 on-branch
threshold, so the summary files list them as "off branch". Describe them as "the three full-speed runs with χ in width 1's
range", not as on-branch.

**Reconciliation of the two loss statements (coordinator, final round; `src/width2_reconcile.py` →
`width2_basins/reconcile.csv`, `reconcile_summary.json`; POST HOC).** Both statements below are exactly true. They are about
different objectives.
- **Population objective (symmetric windows, 800 points).** At R₂ = 0.003 and 0.01 the validated global conditional
  minimiser is placed (a cancelling pair; direct check). Every f_a stuck configuration, evaluated on the population
  objective at the same scale, lies above it: {rc["n_stuck_above_pop_global"]} of {rc["n_compared"]}, by
  {rc["pop_gap_min"]:.1e} to {rc["pop_gap_max"]:.1e} (median {rc["pop_gap_median"]:.1e}).
- **Each run's own 400-point training set (search, not validated).** A placed minimiser was found for 58 of 265 stuck
  states, and in 37 of those 58 the stuck state has the lower loss.
- There is no validated population minimiser at R₂ = 0.03 and 0.1, so {rc["n_not_comparable"]} stuck states have no
  population comparison.
- **Wording:** do not write "the stuck states are often lower in loss than any placed state" without "on the run's own
  training set". Write: "On the population objective the global minimiser at the same scale is placed and every stuck
  state lies above it (by 10⁻⁵–10⁻⁴); on the finite training sets the placed and stuck minima are nearly degenerate, and
  the stuck one is often lower."

The text below is the Track 2 writer input.
{body}

IDs: {_id("T2 2B slowed within", "T2 2B T2-3 three runs obs/pred", "T2 2A Morse-Bott", "T2 2C STOP")}.
"""


def wp29():
    """WP-29: open items for the rebuttal (final round, not in scope)."""
    return """
## WP-29. Open items for the rebuttal (final round; not run, not in the paper's results)

These were out of scope for the final round. The paper may list them as open; it must not state results for them.
- **The outer-exclusion certificate check** by the independent checker (over 10 CPU-hours). Only the original search
  supports the outer-exclusion certificates (WP-17).
- **Further activation families** beyond GELU, SiLU and Mish (WP-25).
- **Any other width-2 variant** (WP-15, WP-20, WP-27).
- **Transfer to simplicity bias (scope; the only sentence the paper may use, exploratory, never as a result):** "In an
  exploratory pilot on a linear-plus-slab task, the fixed-scale minimizer became separating at a single output scale by
  adding the slab feature gradually rather than by switching features; no training test was registered, and a
  redesign is left for future work." (Source: `results/simplicity_bias_design.md`; pilot gate FAIL; nothing
  registered.)
- **Designs that failed their own rules before registration** (tonight; recorded, not run):
  - the width-2 early-basin prospective test (2C, WP-27);
  - the GELU early-scale prospective test (WP-25);
  - the Adam ramp from initialisation (WP-24).

**Do not say** that any of these was tested.
"""


def wp30():
    """WP-30: the registered c1 follow-up (Track 3, final round), from the agent's writer inputs."""
    f = RESULTS / "c1_followup_writer_inputs.md"
    if not f.exists():
        return ""
    t = f.read_text()
    body = t[t.index("\n## ") + 1:] if "\n## " in t else t
    body = "\n" + body
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    return f"""
## WP-30. A decisive test of the first-order coefficient c₁ (Track 3, final round; registered follow-up; for the submission)

Producer: `src/c1_followup.py` → `c1_followup_*`. Registration: `results/c1_followup_registration.md` (e233ef5), committed
before any added certificate was computed. The original test stays INCONCLUSIVE as registered. Independent Arb check:
all 10 added bracket-end certificates (a = 1.08–1.12, both ends) pass (`c1_followup_checks.json`; the last 7 checked on
2026-09-26 with the checker's standard 3 workers). The rescaled Ĝ at the added a has no independent checker. **Author's wording (binding):** the coefficient is "confirmed with certified thresholds at
ε = 0.08 to 0.12"; never call it a small-ε confirmation. The text below is the Track 3 writer input.
{body}

IDs: {_id("c1 follow-up PASS", "c1 follow-up width", "c1 follow-up contains derived")}.
"""


def wp31():
    """WP-31: exact linear response along the trajectory (Track 1, final round; POST HOC), from the agent's writer inputs."""
    f = RESULTS / "linear_response_writer_inputs.md"
    if not f.exists():
        return ""
    t = f.read_text()
    body = t[t.index("\n## ") + 1:] if "\n## " in t else t
    body = "\n" + body
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    return f"""
## WP-31. Exact linear response along the trajectory: what accounts for the 0.82–0.91 shortfall (Track 1, final round; POST HOC; for the submission)

Producer: `src/linear_response.py` → `linear_response/`. Every prediction was committed and hashed before any observed
crossing was read. Everything here is POST HOC and changes no registered verdict. The text below is the Track 1 writer
input.
{body}

IDs: {_id("LR predictions hash", "LR full model ratio")}.
"""


def wp32():
    """WP-32: verified positioning (Track C, final round), from the agent's writer inputs."""
    f = RESULTS / "track_c_writer_inputs.md"
    if not f.exists():
        return ""
    t = f.read_text()
    body = "\n" + t[t.index("## 1. Verification table"):].rstrip()
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    return f"""
## WP-32. Positioning against the implicit-bias, simplicity-bias and slow-manifold literature (Track C, final round; for the submission)

Source: `results/track_c_writer_inputs.md`. Ten citations were verified on their official pages and all are included.
Fenichel was verified on the DOI record, which the author accepted (2026-09-26). None is in `references.bib` yet; add the BibTeX below. The tracking concept itself is not
new, and the text says so. For the novelty claims, keep the labels from WP-24 and WP-31: the within-30% κ result was
derived after a fitted relationship was known; the exact linear response (1.00–1.05) is post hoc; the learning-rate
invariance (R4) is registered; the validity boundary χ ≲ 0.06 comes from the data and is not registered.
{body}
"""


def wp33():
    """WP-33: statistics (Track B, final round; existing data)."""
    D = RESULTS / "track_b"
    if not (D / "summary.json").exists():
        return ""
    S = json.loads((D / "summary.json").read_text()); W = S["within"]
    ss = pd.read_csv(D / "sample_size_decomposition.csv")
    ki = pd.read_csv(D / "kappa_inputs.csv")
    col = pd.read_csv(D / "collapse.csv"); cr = col[col.resolved]
    lo = cr[(cr.chi <= 0.06) & cr.branch_conditioned_post_hoc]
    names = list(W)
    wt = "\n".join(f"| {k} | {v['runs']:,} | {100 * v['pooled_frac_within_10']:.1f}% | {100 * v['pooled_frac_within_20']:.1f}% | "
                   f"{v['arms_median_within_10']} / {v['arms']} | {v['arms_median_within_20']} / {v['arms']} |" for k, v in W.items())
    st = "\n".join(f"| {r.a:.2f} | {r.n:,} | {r.runs} | {r.total_median:.4f} [{r.total_lo:.4f}, {r.total_hi:.4f}] | "
                   f"{r.static_median:.4f} [{r.static_lo:.4f}, {r.static_hi:.4f}] | {r.dynamic_median:.4f} [{r.dynamic_lo:.4f}, {r.dynamic_hi:.4f}] |"
                   for r in ss.itertuples())
    kt = "\n".join(f"| {r.input} | {r.origin} | {r.detail} |" for r in ki.itertuples())
    return f"""
## WP-33. Statistics for the lag law (Track B, final round; existing data; for the submission)

Producer: `src/track_b.py` → `track_b/` (bootstrap: run-level, {S["bootstrap"]["B"]:,} resamples, seed {S["bootstrap"]["seed"]}).
Figures: `lag.pdf` (now with bootstrap 95% intervals on every arm median) and `collapse.pdf` (new); captions in `captions.md`.

**1. Bootstrap intervals** (`arm_ci.csv`): every arm median in the lag figure carries a run-level bootstrap 95% interval.
The lag is measured from the tracked-branch switch, as in the figure.

**2. Fraction within ±10% and ±20% (relative, no absolute floor)** (`within.csv`, `within_summary.json`)

| model | runs | runs within ±10% | runs within ±20% | arm medians within ±10% | arm medians within ±20% |
|---|---|---|---|---|---|
{wt}

- The committed closed form κχ, measured from the tracked-branch switch, has every arm median within ±10%. Per run, 88%
  are within ±10% and 98% within ±20%.
- Measured from the registered reference, the global own-sample threshold (the committed 1A comparison), fewer runs
  are within: this is the reference effect identified in WP-31.
- The trajectory-integrated model (post hoc) has essentially every run within ±10%.

**3. Collapse across settings** (`collapse.csv`, `collapse.pdf`)
- Branch-conditioned points (post hoc) with χ ≤ 0.06 have observed/predicted lag {lo.ratio.min():.2f}–{lo.ratio.max():.2f}. That
  covers width-1 free training, the SGD and Adam ramp cells, the R^d band task, and the three width-2 runs in width 1's range.
- Above χ ≈ 0.06 the ratio departs from 1: SiLU and Mish (χ ≈ 0.2) have the wrong sign, and width 2 at χ ≳ 0.3 reaches
  up to {cr[cr.setting == "width 2"].ratio.max():.0f}.
- Points with |predicted lag| < 0.005 are omitted as below resolution. This excludes GELU and most slowed width-2 runs.

**4. Sample size: the free-training offset splits into a static threshold shift and a dynamic lag**
(`sample_size_decomposition.csv`). Per run, log(s_cross/s_pop) = log(s_own/s_pop) [static] + log(s_cross/s_own)
[dynamic]. Medians are shown with bootstrap 95% intervals.

| a | n (training points) | runs | total offset | static shift | dynamic lag |
|---|---|---|---|---|---|
{st}

- The dynamic lag does not depend on n: about 0.030 at a = 1.30 and 0.062 at a = 1.50 at every n.
- The static finite-sample shift shrinks as n grows.
- So the free-training offset above the population threshold is a finite-sample threshold shift plus an n-independent
  lag.

**5. What κ is built from** (`kappa_inputs.csv`)

| input | origin | detail |
|---|---|---|
{kt}

- The landscape supplies H, θ\\*′, ∇G and s\\*.
- Two inputs to the committed κ came from measured runs: the winding index k, read from each run's crossing state, and
  Adam's preconditioner shape, the median at crossing over existing runs. Track A fixes both before any crossing at an
  unseen a.

**Say:**
- "Every arm median of the lag lies within 10% of the no-fit prediction κ(a)χ when the lag is measured from the branch
  each run tracks (bootstrap 95% intervals shown)."
- "Across settings, observed/predicted lag stays near 1 below χ ≈ 0.06 and departs above it."
- "With more training data the finite-sample threshold shift shrinks, while the lag behind the own threshold is
  unchanged."

**Do not say:**
- "within 10%" for the registered reference: there it is 31 of 36 arm medians, and 64% of runs.
- that κ's winding index and preconditioner were fixed before crossing data at a = 1.30–1.60 (they were not; Track A
  addresses this).
- that the collapse is registered (the branch-conditioned points are post hoc).
"""


def wp34():
    """WP-34: Theorem 1's hypotheses (Track D.1, final round), from the agent's deliverable."""
    f = RESULTS / "theorem1_hypotheses.md"
    if not f.exists():
        return ""
    t = f.read_text()
    body = "\n" + t[t.index("## Which statement"):].rstrip()
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    return f"""
## WP-34. Theorem 1's hypotheses: which are proved and which are checked (Track D.1, final round; for the submission)

Producer: `src/theorem1_checks.py` → `theorem1_checks.json`; tests `tests/test_theorem1_checks.py`. The "checked"
items are float64 Lipschitz-grid arguments, not interval arithmetic. "Theorem 1" is the small-scale criterion of math
note §10.1 (confirmed by the author, 2026-09-26).

**Finding outside the four hypotheses (important for the writer).**
- On the 800-point grid population, the class-mean gap D(α) is periodic: every point is a multiple of q = 0.4/79,401.
  So α\\* = 1.7913244 is the class-mean maximiser only on (0, ≈6.2·10⁵). An aliased weight |w₁| ≈ π/q gives a larger
  gap.
- **Width 1: the conclusion is unchanged.** The alias wins only below s ≈ 2·10⁻¹² and is itself unplaced; the
  localisation lemma excludes it for s > 7.2·10⁻⁶.
- **Width 2:** the literal s → 0 statement on the data population needs the qualifier "on the continuous windows" or
  "for s above about 10⁻¹¹". A cancelling pair of alias units is placed on the data gap but not on the window gap.
  This width-2 consequence has not been checked numerically.
{body}

IDs: {_id("D1 attainment", "D1 remainder", "D1 quadratic growth", "D1 alias")}.
"""


def wp35():
    """WP-35: registered lag-law test at an unseen activation value (Track A, final round), from the agent's writer inputs."""
    f = RESULTS / "track_a_writer_inputs.md"
    if not f.exists():
        return ""
    t = f.read_text()
    body = "\n" + (t[t.index("\n## ") + 1:] if "\n## " in t else t).rstrip()
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    return f"""
## WP-35. A registered test of the lag law at an unseen activation value (Track A, final round; for the submission)

Producer: `src/track_a.py` → `track_a/`; registration `results/track_a_registration.md`, with everything (κ, the winding
rule, the preconditioner rule, the branch rule, own thresholds, the prediction pipeline) frozen and hashed before any
run.

**Census:** each registered criterion is scored as registered.
- L1 and L2 pass for both optimisers.
- L3 was registered per optimiser, so it is two rows: SGD PASS and Adam FAIL. It is not merged into a PARTIAL.

The text below is the Track A writer input.
{body}
"""


def wp36():
    """WP-36: Track T (simplicity-bias transfer), final round: pilot and gate verdict (EXPLORATORY; nothing registered)."""
    f = RESULTS / "simplicity_bias" / "pilot_summary.json"
    if not f.exists():
        return ""
    P = json.loads(f.read_text()); g = P["gate"]; dg = P["diagnostics"]
    h = pd.read_csv(RESULTS / "simplicity_bias" / "hull_diagnostic.csv") if (RESULTS / "simplicity_bias" / "hull_diagnostic.csv").exists() else None
    return f"""
## WP-36. Transfer to simplicity bias (Track T, final round): pilot result and gate verdict (EXPLORATORY; nothing registered)

Producer: `src/simplicity_bias.py` → `simplicity_bias/`; design and gate `results/simplicity_bias_design.md`, committed
before any landscape computation (efde241); pilot and verdict d9dfb96.

**Setting.**
- The linear-plus-slab benchmark of Shah et al. (NeurIPS 2020; verified on the proceedings page), in 2D.
  - x₁ is noisy-linear with p = {P["p"]}: the best threshold misclassifies 10% of each class.
  - x₂ is a 3-slab coordinate that separates every point.
  - {P["n_per_class"]} points per class.
- The small-scale minimiser is predicted to use the linear feature: the class-mean gap per unit scale is 1.6 for the
  linear feature and 1.0 for the slab, computed from the objective.
- The network is two-layer tanh, width 4, with s = ‖w₂‖₁ and the output bias profiled.
- The feature event is "every training point correct with positive margin". The feature-usage measure is a
  deterministic randomisation test, ρ₂ = V₂/(V₁ + V₂).

**Gate (committed before the pilot): FAIL.**
- **One switch:** passes as computed. Set A switches at s = {P["switch"]["A"]:.3f}, set B at {P["switch"]["B"]:.3f}.
- **Linear below and slab above the switch, by feature usage:** fails. Just above the switch the minimiser is still
  linear-dominant (ρ₂ ≈ 0.3), and further up the feature class flips several times.
- **Two restart sets within 2%:** fails; they are {100 * g["agreement_rel"]:.1f}% apart.
- **Validation at the bracket ends:** fails. An independent CMA-ES found a lower loss at one end, and the restart
  ladder failed at {dg["n_ladder_fail"]} of {dg["n_points"]} scales.
- So Step 2 (the registered training test) was not registered and no training was run.

**What the landscape does instead (EXPLORATORY).**
- The hidden weights diverge (median largest weight {dg["max_abs_hidden_weight_median"]:.1e}), so the fixed-scale
  minimum is not attained and restarts cannot validate it.
- At small scale the minimiser is a "hedged" linear function: it matches the closed form 0.8·log(1 + e⁻ˢ) + 0.2·log 2
  to 1e−7.
- The slab coordinate enters gradually. A convex diagnostic over any number of hard units (exploratory, added after the
  pilot) shows the slab share rising smoothly and never reaching ½. The minimiser becomes separating at a single scale,
  s ≈ 3.7–4.7, by adding the slab to a hedged linear solution, not by switching from linear to slab.

**Say (if anything):** "In a pilot on a linear-plus-slab task (exploratory), the fixed-scale minimizer becomes
separating at a single output scale, but it does so by adding the slab feature gradually to a linear solution, not by
switching features, and the width-4 minimizer could not be validated (its weights diverge). We did not run a
registered training test there."

**Do not say:**
- that the account transfers to simplicity bias;
- that a registered test was run or is ongoing;
- that the fixed-scale minimizer switches from the linear to the slab feature. The pilot did not show that.
- Any redesign (bounded hidden weights, or data without gaps; an onset-of-slab-use event) would be new work for the
  rebuttal and is not started.
"""


def wp36_v2():
    """WP-36 (separate file, so the main patch the writer uses is unchanged): the gated Track T redesign (weight decay,
    slab-usage path), from the agent's writer inputs.  Written to paper/WRITER_INPUTS_WP36_track_T.md."""
    f = RESULTS / "simplicity_bias_v2_writer_inputs.md"
    if not f.exists():
        return None
    t = f.read_text()
    body = "\n" + (t[t.index("\n## ") + 1:] if "\n## " in t else t).rstrip()
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    text = f"""# WP-36. Transfer to simplicity bias, redesigned and gated (Track T v2; final night)

A separate file: the main patch (`WRITER_INPUTS_v4_patch.md`) is unchanged. There, WP-29 keeps its one-sentence scope
note about the first pilot. Producer: `src/simplicity_bias*.py` → `results/simplicity_bias*/`.

This attempt:
- adds weight decay on the hidden weights (the same λ in the fixed-scale objective and in training);
- replaces the switch with the scale s_q at which the minimizer's slab usage reaches a level q fixed in advance;
- predicts the training crossing as s_q(1 + κ_q·χ).

The gate, and the registration if the gate passed, were committed before the corresponding computation. The text below
is the Track T writer input.
{body}
"""
    out = RESULTS.parent / "paper" / "WRITER_INPUTS_WP36_track_T.md"
    out.write_text(text)
    return out


def wp37_v3():
    """WP-37 (separate file; the main patch the writer uses is unchanged): the registered Track T training test at the
    attained switch (v3), from the agent's writer inputs.  Written to paper/WRITER_INPUTS_WP37_track_T.md."""
    f = RESULTS / "simplicity_bias_v3_writer_inputs.md"
    if not f.exists():
        return None
    t = f.read_text()
    sc = json.loads((RESULTS / "simplicity_bias_v3" / "score.json").read_text())
    body = "\n" + (t[t.index("\n## ") + 1:] if "\n## " in t else t).rstrip()
    body = body.replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
    text = f"""# WP-37. Does training acquire the slab feature just above the attained fixed-scale switch? (Track T v3; registered)

A separate file: the main patch (`WRITER_INPUTS_v4_patch.md`) and WP-36 are unchanged. Setting: the weight-decayed
linear-plus-slab task of WP-36 (λ = 1e−4), with the attained, reproducible fixed-scale switch at s = 3.5914. The
registered prediction is that training's slab share first reaches q = 0.3914 just above the switch. It uses no κ.
Everything was frozen before any training. The text below is the Track T writer input.
{body}

## Census rows for this registration (folded into the main census on 2026-09-26, at the author's request; WP-14)

| id | block | registration | verdict | scored |
|---|---|---|---|---|
| trackT-C1 | simplicity-bias transfer (Track T v3) | results/simplicity_bias_v3_registration.md (91ef9cf) | UNRESOLVED | fraction of crossing runs at s ≥ 3.5914: {sc["fraction_at_or_above_switch"]:.3f} (would be FAIL); validity failed: {sc["n_cross"]} crossings < 30 and median χ {sc["median_chi"]:.2f} > 0.06 |
| trackT-C2 | simplicity-bias transfer (Track T v3) | results/simplicity_bias_v3_registration.md (91ef9cf) | UNRESOLVED | median crossing/switch {sc["median_ratio"]:.2f} (would be FAIL); same validity failure |

With these two rows the census headline is 234 predictions: 212 scored by a registered rule (100 / 66 / 8 / 38) and 22
assigned post hoc (WP-14).
"""
    out = RESULTS.parent / "paper" / "WRITER_INPUTS_WP37_track_T.md"
    out.write_text(text)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        print(render())
    else:
        main()
        print(render())
        wp36_v2()
        wp37_v3()
