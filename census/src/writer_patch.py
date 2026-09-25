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
- 4b-ID and 4b-OM (both FAIL) were scored on the affected arms. A corrected rerun of the two teleport arms is the
  author's decision; until then, do not rely on the teleport-arm comparisons.
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

**The scaling law (A2).** |w₂|_glob(a) = A*·ε^(−3/2)·(1 + 0.66215ε + O(ε²)), with ε = a − 1. A* ∈ [0.68125, 0.6875]
is certified and independently checked (Krawczyk value 0.6854452); the correction A′(0)/A* comes from math note §8.

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

None is a failure of the certified threshold values themselves, and the primary prospective comparisons (Block 3,
own-seed primary) passed.

**Say:** "Of {int((cr.verdict == "FAIL").sum())} failed predictions ({int(((cr.verdict == "FAIL") & (cr.scoring == "registered rule")).sum())} under the registered rule, {int(((cr.verdict == "FAIL") & (cr.scoring != "registered rule")).sum())} by post hoc scoring), {len(f)} bear on a central
claim; most of these concern the residual's mechanism, which the paper reports as open, and one is the width-2 training
prediction on asymmetric windows (WP-15)." **Do not say** "the central claims never failed" or "the failures are
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
  - Here the arms were replayed *as run*, which reproduces them exactly. 4b-ID and 4b-OM (both FAIL) were scored on the
    affected arms, and a corrected rerun is the author's decision.
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        print(render())
    else:
        main()
        print(render())
