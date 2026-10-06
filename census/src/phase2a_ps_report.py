"""Phase 2A-PS results page (results/phase2a_ps_results.md) from results/phase2a_ps/scores.json.

score_result() rebuilds the scores dict exactly as the registered src/phase2a_ps.py `score` assembles it (the same
registered functions, in the same order) from the committed frozen.json, frozen_parts.jsonl, runs.jsonl and
observed.jsonl; the ledger (phase2a_ps_checks) asserts it equals the committed scores.json and that this page
regenerates from it.

    python -m src.phase2a_ps_report
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import phase2a_ps as P

OUT = P.OUT
RESULTS_MD = P.RESULTS / "phase2a_ps_results.md"


def score_result():
    """The scores dict exactly as phase2a_ps.score assembles it."""
    fr, rows = P._frozen()
    runs = P._rows(OUT / "runs.jsonl")
    obs = {(o["seed"], o["log2rho"]): o for o in P._rows(OUT / "observed.jsonl")}
    T = [P.run_row(P.seed_record(rows, r["seed"]), r, obs.get((r["seed"], r["log2rho"]), {})) for r in runs]
    ST = P.score_tables(T, fr["gate"]["pass"])
    return {"counts_PROMINENT": fr["counts"], "gate": fr["gate"], "outcome": ST["outcome"],
            "verdicts": ST["verdicts"], "secondary_verdicts": ST.get("secondary_verdicts"),
            "criteria": ST["criteria"], "secondary": ST.get("secondary"), "validity": ST["validity"],
            "seed_exponents": ST.get("seed_exponents"), "rows": T, "DESCRIPTIVE": P.descriptive(T),
            "DESCRIPTIVE_split_by_mc_agreement": P.descriptive_split(T) if fr["gate"]["pass"] else None}


def _f(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.{d}e}"
        return f"{v:.{d}f}"
    return str(v)


def _ci(c):
    lo, hi = c.get("ci95", [None, None])
    return f"[{_f(lo, 4)}, {_f(hi, 4)}]"


def crit_line(k, c):
    if k == "F":
        return (f"{c['n_within']}/{c['n']} in [s_F, 1.25·s_F] (frac {_f(c['frac'])}, needs ≥ 0.90); "
                f"no crossing {c['n_no_crossing']}; below s_F {c['n_below_sF']}; falsifier {c['n_falsifier']}")
    if k == "H":
        return (f"{c['n_no_crossing_to_1.25sF']}/{c['n']} no crossing by 1.25·s_F (frac {_f(c['frac'])}, needs ≥ 0.80); "
                f"crossed {c['n_crossed']}; ended at the budget without crossing {c['n_budget_end_no_crossing']}")
    if k == "E_seed":
        return (f"median exponent {_f(c['median'], 4)}, 95% CI {_ci(c)} (band [0.55, 0.80]); n {c['n']}, "
                f"excluded {c['n_excluded']}")
    if k == "C3":
        return (f"{c['n_gt_1']}/{c['n']} with r_obs/r_fc > 1 (frac {_f(c['frac'])}, needs ≥ 0.90); ratio min "
                f"{_f(c['ratio_min'])}, median {_f(c['ratio_median'])}, max {_f(c['ratio_max'])}")
    if k == "C4":
        return (f"mean D {_f(c['mean_D'], 1)} steps, 95% CI {_ci(c)}; n {c['n']}; no forecast {c['n_no_forecast']}; "
                f"forecast closer at {_f(c['DESCRIPTIVE_frac_forecast_closer'])}")
    if k == "P":
        return (f"mean D {_f(c['mean_D'], 5)}, 95% CI {_ci(c)}; n {c['n']}; own ŝ_c missing {c['n_own_missing']}; "
                f"fixed missing (excluded) {c['n_fixed_missing_excluded']}; own closer at "
                f"{_f(c['DESCRIPTIVE_frac_own_closer'])}")
    if k in ("C1", "C2"):
        return (f"{c['n_within']}/{c['n']} within τ = {c['tau']} (frac {_f(c['frac_within'])}, needs ≥ 0.80); "
                f"no forecast {c['n_no_forecast']}")
    return ""


def posthoc_summary(S):
    """DESCRIPTIVE: 2A's fitted correction (POST HOC) on the clean 2⁻¹⁶ seeds: r_obs/r_posthoc and the scale error of
    ŝ_c^posthoc = s_F(1 + r_posthoc) beside the registered leading-order ŝ_c."""
    rows = {(r["seed"], r["log2rho"]): r for r in S["rows"]}
    rat, e_ph, e_lo = [], [], []
    for d in S["DESCRIPTIVE"]["per_seed"]:
        if d["class"] != "clean" or d["log2rho"] != P.LOG2_SCORED:
            continue
        r = rows[(d["seed"], d["log2rho"])]
        if d["ratio_obs_posthoc"] is not None:
            rat.append(d["ratio_obs_posthoc"])
        if r["s_obs"] is not None and d["r_posthoc"] is not None and r["s_c_fc"]:
            e_ph.append(abs(math.log(r["s_F"] * (1 + d["r_posthoc"]) / r["s_obs"])))
            e_lo.append(abs(math.log(r["s_c_fc"] / r["s_obs"])))
    return {"n": len(rat), "ratio_median": float(np.median(rat)) if rat else None,
            "ratio_range": [float(min(rat)), float(max(rat))] if rat else None,
            "frac_ratio_gt_1": float(np.mean(np.array(rat) > 1)) if rat else None,
            "mean_abs_log_err_posthoc": float(np.mean(e_ph)) if e_ph else None,
            "mean_abs_log_err_leading": float(np.mean(e_lo)) if e_lo else None}


def misses(S):
    """Every trained run that misses a rule, grouped by reason (seed lists)."""
    R = S["rows"]
    out = []

    def grp(title, rows, extra=None):
        if rows:
            out.append(f"- **{title}** ({len(rows)}): " + ", ".join(
                f"{r['seed']:,} (2^{r['log2rho']:g}{'; ' + extra(r) if extra else ''})" for r in rows) + ".")
        else:
            out.append(f"- **{title}**: none.")
    clean = [r for r in R if r["class"] == "clean"]
    nofc = [r for r in clean if not r["has_fc"]]
    for st in sorted({r["status"] for r in nofc}):
        grp(f"Clean, no forecast — {st}", [r for r in nofc if r["status"] == st])
    grp("Clean, no crossing by 1.25·s_F or the budget", [r for r in clean if r["t_obs"] is None])
    grp("Clean, crossing outside [s_F, 1.25·s_F]", [r for r in clean if r["s_obs"] is not None
                                                    and not r["s_F"] <= r["s_obs"] <= 1.25 * r["s_F"]],
        lambda r: f"s_obs/s_F {r['s_obs'] / r['s_F']:.4f}")
    grp("Clean 2⁻¹⁶, r_obs/r_fc ≤ 1 (C3)", [r for r in clean if r["log2rho"] == P.LOG2_SCORED and r["has_fc"]
                                            and r["r_obs"] is not None and r["r_obs"] / r["r_fc"] <= 1],
        lambda r: f"ratio {r['r_obs'] / r['r_fc']:.3f}")
    grp("Clean, an active sign changed before the crossing (validity)", [r for r in clean if not r["sign_ok"]])
    grp("Clean, idle unit left 0 (validity)", [r for r in clean if not r["idle_ok"]])
    grp("Clean 2⁻¹⁶, follow check off the seed's own M (validity)",
        [r for r in clean if r["log2rho"] == P.LOG2_SCORED and not r["follow_in_M"]])
    grp("Clean 2⁻¹⁶, t_c not before t_obs (validity)",
        [r for r in clean if r["log2rho"] == P.LOG2_SCORED and not (r["t_c"] is not None and (
            r["t_obs"] is None or r["t_c"] < r["t_obs"]))])
    none = [r for r in R if r["class"] == "none"]
    grp("None, crossed (H)", [r for r in none if r["t_obs"] is not None],
        lambda r: f"s_obs/s_F {r['s_obs'] / r['s_F']:.3f}")
    grp("None, ended at the budget below 1.25·s_F without crossing (H)",
        [r for r in none if r["t_obs"] is None and not r["reached_obs_end"]])
    out.append(f"- DESCRIPTIVE (not in validity): an active sign changed before the crossing or the end on "
               f"{sum(1 for r in none if not r['sign_ok'])}/{len(none)} none and "
               f"{sum(1 for r in R if r['class'] == 'mixed' and not r['sign_ok'])}/"
               f"{sum(1 for r in R if r['class'] == 'mixed')} mixed runs.")
    return out


def render_md(S):
    c, g = S["counts_PROMINENT"], S["gate"]
    bs = c["by_status"]
    n = c["n_seeds"]
    fr = json.loads((OUT / "frozen.json").read_text())
    L = ["# Phase 2A-PS results: the fold prediction with a sample per seed", "",
         "Generated by `python -m src.phase2a_ps_report` from `results/phase2a_ps/scores.json` (registered "
         "`src/phase2a_ps.py` `score`). Registration: `results/phase2a_ps_registration.md` (commit c819290; stamp "
         "9a10df4; OpenTimestamps proof a2065bf). Forecasts committed before any observation (`forecasts.sha256`).", "",
         f"**Outcome: {S['outcome']}** (F, H, E_seed, C3, C4, P; C1 and C2 secondary: "
         f"C1 {S['secondary_verdicts']['C1']}, C2 {S['secondary_verdicts']['C2']}).", "",
         "## Freeze counts (all 400 registered seeds; read these first)", "",
         f"- **Untraceable: {c['n_untraceable']} of {n} ({100 * c['n_untraceable'] / n:.1f}%)** — M homotopy failed "
         f"{bs['untraceable: M homotopy failed']}, no upper fold {bs['untraceable: no upper fold']}, ρ₂ ≥ q on the "
         f"stable part {bs['untraceable: rho2 >= q on the stable part']}.",
         f"- s_F not validated {bs['s_F not validated']}; Λ_F not validated {bs['Lambda_F not validated']} (D1 (c): "
         f"the |m′c′| agreement check is descriptive); **no s\\* {c['n_no_s_star']}**; release Newton failed "
         f"{c['n_release_failed']}; **ineligible {c['n_ineligible']}**.",
         f"- **Scoreable {c['n_scoreable']}: clean {c['scoreable_by_class']['clean']}, none "
         f"{c['scoreable_by_class']['none']}, mixed {c['scoreable_by_class']['mixed']}.** Gate: "
         f"{g['n_clean_eligible']} clean scoreable ≥ {g['min']}: {'PASS' if g['pass'] else 'FAIL'}.",
         f"- Trained (caps 40/30/20 in seed order): clean {fr['plan_counts']['clean']} (2⁻¹⁴ and 2⁻¹⁶), none "
         f"{fr['plan_counts']['none']}, mixed {fr['plan_counts']['mixed']} (2⁻¹⁴). **Counted, not trained:** "
         + ", ".join(f"{k} {len(v)}" for k, v in fr["surplus_counted_not_trained"].items()) + ".",
         "- Under the first evaluation (D1 as (a), before the author's clarification at freeze): Λ_F not validated "
         "91 (clean 45); clean scoreable 37.", "",
         "## Verdicts (clean seeds at 2⁻¹⁶ unless stated; 10,000 seed resamples)", "",
         "| criterion | verdict | statistic |", "|---|---|---|"]
    for k in P.PRIMARY:
        L.append(f"| {k}{' (none seeds, 2⁻¹⁴)' if k == 'H' else ''} | **{S['verdicts'][k]}** | "
                 f"{crit_line(k, S['criteria'][k])} |")
    for k in P.SECONDARY:
        L.append(f"| {k} (secondary; registered expected FAIL) | {S['secondary_verdicts'][k]} | "
                 f"{crit_line(k, S['secondary'][k])} |")
    v = S["validity"]
    L += ["", f"**Validity: {'OK' if v['ok'] else 'NOT OK'}.** Follow check on the seed's own M at 0.8·s_F "
          f"{_f(v['follow_frac'])} (needs ≥ 0.90); t_c < t_obs {_f(v['cutoff_before_frac'])} (needs ≥ 0.90; "
          f"not before: {v['n_cutoff_not_before']}); idle unit exactly 0 at every clean run "
          f"{_f(v['idle_zero_all'])}; signs fixed {_f(v['signs_fixed_all'])} ({v['n_runs_idle_sign']} clean runs).",
          f"F's falsifier (a crossing below s_F and at or below 1.25·s\\*): "
          f"{'FIRED' if S['criteria']['F']['falsified'] else 'did not fire'}."]
    c4 = S["criteria"]["C4"]
    if c4["verdict"] == "FAIL" and c4["n_no_forecast"] and c4["ci95"][1] is not None and c4["ci95"][1] < 0:
        L += ["", f"C4 FAILs by the registered no-forecast rule (D12: any trained clean seed without a forecast fails C4; "
                  f"{c4['n_no_forecast']} such seeds). Over the {c4['n']} seeds with a forecast and a crossing the "
                  f"interval lies wholly below 0."]
    L += ["",
          "## DESCRIPTIVE: split by the |m′c′| two-window agreement check (not a criterion)", "",
          "| subset | clean seeds | none seeds | F | H | E_seed | C3 | C4 | P | C1 | C2 | validity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    D = S["DESCRIPTIVE_split_by_mc_agreement"]
    for tag in ("agree", "disagree"):
        d = D[tag]
        vv, sv = d["verdicts_DESCRIPTIVE"], d["secondary_verdicts_DESCRIPTIVE"]
        L.append(f"| {tag} | {d['n_clean_scored']} | {d['n_none']} | "
                 + " | ".join(vv[k] for k in P.PRIMARY) + " | " + " | ".join(sv[k] for k in P.SECONDARY)
                 + f" | {'OK' if d['validity_DESCRIPTIVE']['ok'] else 'NOT OK'} |")
    L += ["", "Per subset: " + "; ".join(
        f"{tag}: F {D[tag]['criteria_DESCRIPTIVE']['F']['n_within']}/{D[tag]['criteria_DESCRIPTIVE']['F']['n']}, "
        f"E_seed median {_f(D[tag]['criteria_DESCRIPTIVE']['E_seed']['median'], 4)} "
        f"{_ci(D[tag]['criteria_DESCRIPTIVE']['E_seed'])}, C3 {D[tag]['criteria_DESCRIPTIVE']['C3']['n_gt_1']}/"
        f"{D[tag]['criteria_DESCRIPTIVE']['C3']['n']}, P mean D {_f(D[tag]['criteria_DESCRIPTIVE']['P']['mean_D'], 5)} "
        f"{_ci(D[tag]['criteria_DESCRIPTIVE']['P'])}" for tag in ("agree", "disagree")) + "."]
    L += ["", "## Misses by reason (every trained run; from the scored rows)", ""] + misses(S)
    ph = posthoc_summary(S)
    L += ["", "## POST HOC (descriptive): 2A's fitted correction as a forecast", "",
          f"r = A ε^{{2/3}} + C ε ln(1/ε), A = {_f(S['DESCRIPTIVE']['posthoc_A'], 4)}, "
          f"C = {_f(S['DESCRIPTIVE']['posthoc_C'], 4)} (fitted on 2A's results; `phase2a_posthoc.json`), with each "
          f"seed's ε̂_F. Clean seeds at 2⁻¹⁶ (n {ph['n']}): r_obs/r_posthoc median {_f(ph['ratio_median'])}, range "
          f"{_f((ph['ratio_range'] or [None])[0])}–{_f((ph['ratio_range'] or [None, None])[1])}, > 1 at "
          f"{_f(ph['frac_ratio_gt_1'])}; mean |ln ŝ_c/s_obs| {_f(ph['mean_abs_log_err_posthoc'], 5)} (POST HOC) vs "
          f"{_f(ph['mean_abs_log_err_leading'], 5)} (registered leading order). Never a verdict.", "",
          "## DESCRIPTIVE: every trained cell", "",
          "| class, log₂ρ | runs | crossed | in [s_F, 1.25·s_F] | falsifier flag |", "|---|---|---|---|---|"]
    for k, b in sorted(S["DESCRIPTIVE"]["by_class_rate"].items()):
        L.append(f"| {k} | {b['n']} | {b['n_crossed']} | {b['n_in_F_window']} | {b['n_falsifier_flag']} |")
    L += ["", "Mixed seeds enter no criterion. Exploration and design: `results/designs/phase2a_perseed_design.md`.", ""]
    return "\n".join(L)


def main():
    S = json.loads((OUT / "scores.json").read_text())
    RESULTS_MD.write_text(render_md(S))
    print(RESULTS_MD.read_text())


if __name__ == "__main__":
    main()
