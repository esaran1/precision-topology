"""Census additions for the 2026-09-25 night program (Tracks 1B, 2C, 3A, 3B) and the 2026-09-29 discussion phase
(Tests 2A, 2B, 2C, GELU-T; census_round 2026-09-29): registered units appended to
results/registration_census_enumeration.csv from the committed verdict files (idempotent: rows with these id prefixes
are replaced).  Track 1A (κ) is not a registered prediction (a no-fit comparison derived after the fitted relationship
was known) and is not in the census.

    python -m src.census_tonight && python -m src.registration_census && python -m src.census_relevance
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ENUM = RESULTS / "registration_census_enumeration.csv"
PREFIXES = ("ramp-", "act-", "t2c-", "band-", "c1-followup", "trackA-", "trackT-", "boundary-",
            "test2A-", "test2B-", "geluT-", "test2C-")
ROUND = "2026-09-25"
ROUND_DISCUSSION = "2026-09-29"      # discussion phase: Tests 2A, 2B, 2C and GELU-T


def _row(id_, block, reg, commit, verdict, text, scored, note="", round_=ROUND):
    return {"id": id_, "block": block, "registration_file": reg, "registration_commit": commit, "verdict": verdict,
            "original_verdict_text": text, "scored_in": scored, "counted_in_existing_64": "no", "note": note,
            "census_round": round_}


def ramp_rows():
    V = json.loads((RESULTS / "ramp" / "verdicts.json").read_text())
    blk, reg, com = "ramp (Track 1B)", "results/ramp_registration.md", "c4b4c6d"
    note = ("Own thresholds frozen before any crossing was read (83f66f5, 15b4744). POST HOC beside it (ramp.posthoc_branch): "
            "against each seed's tracked-branch switch, SGD passes R1-R3 in all settings.")
    rows = []
    rule = {"R1": "through-origin slope of r on kappa_k*chi in [0.7, 1.3]",
            "R2": "sign(kappa)*Spearman(gamma, cell median r) >= 0.9",
            "R3": "|median r| in the slowest gamma cell < 0.005"}
    val = {"R1": ("slope_obs_on_pred", "{:.2f}"), "R2": ("signed_spearman_gamma_median", "{:.2f}"),
           "R3": ("slowest_median_r", "{:+.4f}")}
    for x in V["settings"]:
        a, k, opt = round(x["a"], 2), int(x["winding"]), x["opt"]
        pre = "ramp-R5-" if (a == 1.3 and k == 0) else "ramp-"
        for R in ("R1", "R2", "R3"):
            col, fmt = val[R]
            unit = f"{opt} {a:.2f}" if pre == "ramp-" else opt
            rows.append(_row(f"{pre}{R}@{unit}", blk, reg, com, x[R],
                             f"{'R5 (winding sign test, a = 1.30, k = 0): ' if pre != 'ramp-' else ''}{R}: {rule[R]}. "
                             f"Scored {opt} a = {a:.2f} (k = {k}): {fmt.format(x[col])}: {x[R]}.",
                             "results/ramp/verdicts.json", note))
    for x in V["R4"]:
        a = round(x["a"], 2)
        rows.append(_row(f"ramp-R4@{a:.2f}", blk, reg, com, x["R4"],
                         f"R4: free Adam at eta = 0.01/0.005/0.0025, median residual within max(0.01, 0.25|r_0.01|) of the "
                         f"eta = 0.01 median. Scored a = {a:.2f}: {x['median_0.01']:.4f} / {x['median_0.005']:.4f} / "
                         f"{x['median_0.0025']:.4f}, tol {x['tol']:.4f}: {x['R4']}.", "results/ramp/verdicts.json"))
    return rows


def act_rows():
    ts = pd.read_csv(RESULTS / "act_general" / "training_scores.csv")
    blk, reg, com = "non-sine activations (Track 3A)", "results/act_training_registration.md", "ad99053"
    rows = []
    for r in ts.itertuples():
        arm = "" if r.arm == "primary" else "-200"
        ta = getattr(r, "_10"); tb = getattr(r, "_11")          # columns 'T-a', 'T-b'
        rows.append(_row(f"act-Ta{arm}@{r.act}", blk, reg, com, ta,
                         f"T-a ({'40 seeds' if not arm else '200 seeds, registered secondary arm'}): >= 90% of crossing runs at "
                         f"s >= s_lo of the validated bracket. Scored {r.act}: {r.crossing_runs} crossing"
                         + (f", fraction {r.frac_at_or_above_lo:.3f}" if ta != "UNRESOLVED" else " (< 30)") + f": {ta}.",
                         "results/act_general/training_scores.csv"))
        rows.append(_row(f"act-Tb{arm}@{r.act}", blk, reg, com, tb,
                         f"T-b ({'40 seeds' if not arm else '200 seeds'}): median residual within max(0.01, 0.25|pred|) of "
                         f"median kappa*chi. Scored {r.act}: "
                         + (f"obs {r.obs:+.4f}, pred {r.pred:+.4f}" if tb != "UNRESOLVED" else f"{r.crossing_runs} crossing (< 30)")
                         + f": {tb}.", "results/act_general/training_scores.csv"))
    return rows


def band_rows():
    v = pd.read_csv(RESULTS / "band_rd" / "verdicts.csv")
    blk, reg, com = "band task in R^d (Track 3B)", "results/band_rd_registration.md", "f19535b"
    rule = {"P1": ">= 90% of crossing runs at s_c >= s_lo of the certified width-1 bracket",
            "P2a": "median residual vs the population threshold inside the width-1 phase2b IQR",
            "P2b": "median residual vs the frozen x1-sample own threshold within max(0.01, 0.25|pred|) of median kappa_k*chi"}
    rows = []
    for r in v[v.d > 1].itertuples():
        sec = r.arm == "secondary"
        for R in (("P1", "P2a") if sec else ("P1", "P2a", "P2b")):
            verdict = getattr(r, R)
            detail = {"P1": f"fraction {r.frac_at_or_above_s_lo:.3f}", "P2a": f"median r_pop {r.median_r_pop:.3f} vs [{r.ref_q1:.3f}, {r.ref_q3:.3f}]",
                      "P2b": f"obs {r.median_obs_r_own:.3f} vs pred {r.median_pred_r:.3f}"}[R] if verdict != "UNRESOLVED" else f"{r.n_crossing} crossing (< 30)"
            rows.append(_row(f"band-{R}{'-120' if sec else ''}@d{r.d} {r.a:.2f}", blk, reg, com, verdict,
                             f"{R} ({'secondary, 120 seeds pooled' if sec else 'primary, 40 seeds'}): {rule[R]}. Scored d = {r.d}, "
                             f"a = {r.a:.2f}: {detail}: {verdict}.", "results/band_rd/verdicts.csv",
                             "POST HOC beside it: against each run's own R^d branch switch the lag law holds in every cell (obs/pred 0.84-1.10)."))
    return rows


def c1_rows():
    import json as _j
    f = RESULTS / "c1_followup_summary.json"
    if not f.exists():
        return []
    S = _j.loads(f.read_text())
    return [_row("c1-followup", "c1 first order", "results/c1_followup_registration.md", "e233ef5", S["verdict"],
                 f"Follow-up (registered after c1-primary's INCONCLUSIVE): the registered feasible-set estimator unchanged, on the "
                 f"four original brackets plus five added certified brackets (a = 1.08-1.12); validity width <= 0.1; PASS iff the "
                 f"interval contains the derived c1. Scored: C = [{S['feasible_lo']:.5f}, {S['feasible_hi']:.5f}], width "
                 f"{S['feasible_width']:.4f}, contains [0.2852300, 0.2852303]: {S['verdict']}.",
                 "results/c1_followup_summary.json", "c1-primary stays UNRESOLVED (INCONCLUSIVE) as registered.")]


def track_a_rows():
    f = RESULTS / "track_a" / "scores.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    rule = {"L1": "median observed/predicted lag (trajectory-integrated) in [0.90, 1.10]",
            "L2": "median observed/predicted lag (closed form) in [0.80, 1.20]",
            "L3": "per-run Spearman(predicted, observed lag) >= 0.5"}
    rows = []
    for opt in ("adam", "sgd"):
        for L in ("L1", "L2", "L3"):
            x = S[opt][L]; v = x["verdict"]
            val = f"median ratio {x['median_ratio']:.3f}" if L != "L3" else f"Spearman {x['spearman']:.3f}"
            pid = f"trackA-{L}-{opt}" if L == "L3" else f"trackA-{L}@{opt}"      # L3 registered per optimiser: one row each
            rows.append(_row(pid, "lag law at an unseen a (Track A)", "results/track_a_registration.md", "fcd2e46", v,
                             f"{L}: {rule[L]}, per optimiser, a = 1.65, 80 fresh seeds; everything frozen and hashed before any run. "
                             f"Scored {opt}: {S[opt]['n_crossed']} crossed, {val}: {v}.", "results/track_a/scores.json"))
    return rows


def track_t_rows():
    f = RESULTS / "simplicity_bias_v3" / "score.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    blk, reg, com = "simplicity-bias transfer (Track T)", "results/simplicity_bias_v3_registration.md", "91ef9cf"
    why = (f"validity failed: {S['n_cross']} of {S['n_runs']} crossed (< 30) and median chi at crossing "
           f"{S['median_chi']:.2f} (> 0.06)")
    return [_row("trackT-C1", blk, reg, com, S["C1"],
                 f"C1: >= 90% of crossing runs at s >= 3.5914 (the attained weight-decayed switch). Scored: {why}: "
                 f"{S['C1']} (would-be {S['fraction_at_or_above_switch']:.3f}, FAIL).", "results/simplicity_bias_v3/score.json"),
            _row("trackT-C2", blk, reg, com, S["C2"],
                 f"C2: median crossing/switch in [1.00, 1.25]. Scored: {why}: {S['C2']} (would-be "
                 f"{S['median_ratio']:.2f}, FAIL).", "results/simplicity_bias_v3/score.json")]


def boundary_rows():
    f = RESULTS / "ramp_boundary" / "verdicts.json"
    if not f.exists():
        return []
    V = json.loads(f.read_text())
    blk, reg, com = "boundary test (SGD forced ramps)", "results/ramp_boundary_registration.md", "a22e1aa"
    nvalid = sum(c["valid"] for c in V["cells"])
    return [_row("boundary-B1", blk, reg, com, V["B1"],
                 f"B1: median obs/pred in [0.75, 1.25] in the chi = 0.03 and 0.06 cells at a = 1.30 and 1.50. Scored: "
                 f"{nvalid} of 10 cells valid; the chi = 0.06 cell at a = 1.30 has fewer than 30 crossings: {V['B1']} (the valid "
                 f"cell chi = 0.03 at a = 1.30 is out of band).", "results/ramp_boundary/verdicts.json"),
            _row("boundary-B2", blk, reg, com, V["B2"],
                 f"B2: median obs/pred outside [0.75, 1.25] in the chi = 0.4 cells. Scored: both cells have fewer than 30 "
                 f"crossings: {V['B2']}.", "results/ramp_boundary/verdicts.json")]


def test2a_rows():
    """Test 2A (discussion phase): Adam per-run ordering at the unseen a = 1.85; one unit, A1-A3 one row each."""
    f = RESULTS / "track2a" / "scores.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    blk, reg, com = "Adam per-run ordering at an unseen a (Test 2A)", "results/track2a_registration.md", "9fd1f32"
    rule = {"A1": "Spearman(pred, obs) >= 0.5", "A2": "fraction of scored runs with |obs/pred - 1| <= 0.10 is >= 0.75",
            "A3": "median obs/pred in [0.9, 1.1]"}
    val = {"A1": lambda x: f"Spearman {x['spearman']:.3f}", "A2": lambda x: f"fraction {x['frac_within_10pct']:.3f}",
           "A3": lambda x: f"median ratio {x['median_ratio']:.3f}"}
    return [_row(f"test2A-{A}", blk, reg, com, S[A]["verdict"],
                 f"{A}: {rule[A]}; Adam, a = 1.85, 80 fresh seeds, P frozen at t_sw (the rule chosen post hoc at 1.65, "
                 f"tested prospectively); everything frozen and hashed before any run. Scored: {S['n_crossed']} crossed, "
                 f"{S['n_scored']} scored, validity {'met' if S['valid'] else 'failed'}, {val[A](S[A])}: {S[A]['verdict']}.",
                 "results/track2a/scores.json", round_=ROUND_DISCUSSION) for A in ("A1", "A2", "A3")]


def test2b_rows():
    """Test 2B (discussion phase): validity boundary of the lag law in kappa*chi; C1, C2 each one verdict over its cells."""
    f = RESULTS / "track2b" / "scores.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    blk, reg, com = "lag-law validity boundary (Test 2B)", "results/track2b_registration.md", "0708ae9"
    c1 = [c for c in S["cells"] if c["kx_target"] <= 0.1 + 1e-12]
    c2 = [c for c in S["cells"] if c["kx_target"] >= 0.3 - 1e-12]
    lo, hi = min(c["median_ratio"] for c in c1), max(c["median_ratio"] for c in c1)
    return [_row("test2B-C1", blk, reg, com, S["C1"],
                 f"C1: median obs/pred in [0.75, 1.25] in every cell with kappa*chi in {{0.02, 0.05, 0.1}} at a = 1.30 and "
                 f"1.50 (6 cells); an invalid cell makes it UNRESOLVED. Scored: {sum(c['valid'] for c in c1)} of {len(c1)} "
                 f"cells valid, median ratios {lo:.3f}-{hi:.3f}: {S['C1']}.", "results/track2b/scores.json",
                 round_=ROUND_DISCUSSION),
            _row("test2B-C2", blk, reg, com, S["C2"],
                 f"C2: median obs/pred outside [0.75, 1.25] in every cell with kappa*chi in {{0.3, 0.4}} (4 cells). Scored: "
                 f"{sum(c['valid'] for c in c2)} of {len(c2)} cells valid (realised eta*lambda_max > 1): {S['C2']}.",
                 "results/track2b/scores.json", round_=ROUND_DISCUSSION)]


def gelu_t_rows():
    """GELU-T (discussion phase): two registered arms, each with its own gate, validity and L1-L5; one row per
    criterion per arm (author, 2026-09-29: the control arm's criteria are separate registered rows)."""
    f = RESULTS / "gelu_transfer" / "scores.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    blk, reg, com = "lag law at GELU from a declared start (GELU-T)", "results/gelu_transfer_registration.md", "736b6bf"
    rule = {"L1": "median r_obs/r_traj in [0.90, 1.10]", "L2": "median r_obs/r_cf in [0.80, 1.20]",
            "L3": "Spearman(r_traj, r_obs) >= 0.5",
            "L4": "upper end of the bootstrap 95% interval of mean D < 0, D against the population threshold s_glob",
            "L5": "upper end of the bootstrap 95% interval of mean D < 0, D against the occupied branch's switch"}
    rows = []
    for arm, suf, label in (("random", "", "primary arm, random start (the headline verdict)"),
                            ("branch", "-control", "mechanism-control arm, branch-point start")):
        A = S["arms"][arm]
        for L in ("L1", "L2", "L3", "L4", "L5"):
            x = A[L]
            val = (f"median ratio {x['median_ratio']:.3f}" if L in ("L1", "L2") else f"Spearman {x['spearman']:.3f}"
                   if L == "L3" else f"95% CI [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]")
            rows.append(_row(f"geluT-{L}{suf}", blk, reg, com, x["verdict"],
                             f"{L} ({label}): {rule[L]}; GELU, free SGD, 80 seeds, everything frozen and hashed before any "
                             f"run. Scored: gate {'passed' if A['gate']['pass'] else 'failed'}, validity "
                             f"{'met' if A['valid'] else 'failed'}, {A['n_scored']} scored, {val}: {x['verdict']}.",
                             "results/gelu_transfer/scores.json", round_=ROUND_DISCUSSION))
    return rows


def test2c_rows():
    """Test 2C (discussion phase): C1-C3 registered per cell (d, a), no pooled verdict; one row per criterion per cell
    (author, 2026-09-29, as Track A's L3 ruling), not merged into a post hoc PARTIAL."""
    f = RESULTS / "track2c" / "scores.json"
    if not f.exists():
        return []
    S = json.loads(f.read_text())
    blk, reg, com = "band task in R^d vs own R^d switch (Test 2C)", "results/track2c_registration.md", "b22ebd0"
    rule = {"C1": "median |log(s_obs/s_pred)| <= 0.05", "C2": "median signed r_obs/r_pred in [0.75, 1.25]",
            "C3": "upper end of the bootstrap 95% interval of mean D < 0 against the x1-only own threshold"}
    rows = []
    for key in ("d2_a1.30", "d2_a1.50", "d4_a1.30", "d4_a1.50"):
        c = S["cells"][key]
        why = ("validity met" if c["valid"] else
               "validity failed (V3: " + f"{c['n_scored']} of {c['n_crossing']} crossing runs scored, {c['frac_scored']:.3f} < 0.90)")
        for C in ("C1", "C2", "C3"):
            x = c[C]
            val = (f"median |log err| {x['median_abs_log_err']:.4f}" if C == "C1" else
                   f"median ratio {x['median_ratio']:.3f}" if C == "C2" else f"95% CI [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]")
            rows.append(_row(f"test2C-{C}-d{c['d']}-{c['a']:.2f}", blk, reg, com, x["verdict"],
                             f"{C} (registered per cell, no pooled verdict): {rule[C]}; Adam, 60 seeds per cell, own-sample "
                             f"R^d switch frozen before any run. Scored d = {c['d']}, a = {c['a']:.2f}: {c['n_crossing']} "
                             f"crossings, {why}; statistic {val}"
                             + (" (not a verdict)" if x["verdict"] == "UNRESOLVED" else "") + f": {x['verdict']}.",
                             "results/track2c/scores.json", round_=ROUND_DISCUSSION))
    return rows


def extra_rows():
    """Track 2C and Track 3B rows, from their agents' committed files (added when those tracks are scored)."""
    rows = []
    f = RESULTS / "census_rows_t2c_band.csv"
    if f.exists():
        rows += pd.read_csv(f).to_dict("records")
    return rows


def main():
    d = pd.read_csv(ENUM)
    d = d[~d.id.str.startswith(PREFIXES)]
    new = pd.DataFrame(ramp_rows() + act_rows() + band_rows() + c1_rows() + track_a_rows() + track_t_rows() + boundary_rows() + extra_rows()
                       + test2a_rows() + test2b_rows() + gelu_t_rows() + test2c_rows())
    for c in d.columns:
        if c not in new.columns:
            new[c] = ""
    out = pd.concat([d, new[d.columns]], ignore_index=True)
    out.to_csv(ENUM, index=False)
    print(new[["id", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
