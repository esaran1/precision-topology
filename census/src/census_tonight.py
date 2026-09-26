"""Census additions for the 2026-09-25 night program (Tracks 1B, 2C, 3A, 3B): registered units appended to
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
PREFIXES = ("ramp-", "act-", "t2c-", "band-", "c1-followup", "trackA-")
ROUND = "2026-09-25"


def _row(id_, block, reg, commit, verdict, text, scored, note=""):
    return {"id": id_, "block": block, "registration_file": reg, "registration_commit": commit, "verdict": verdict,
            "original_verdict_text": text, "scored_in": scored, "counted_in_existing_64": "no", "note": note,
            "census_round": ROUND}


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
            rows.append(_row(f"trackA-{L}@{opt}", "lag law at an unseen a (Track A)", "results/track_a_registration.md", "fcd2e46", v,
                             f"{L}: {rule[L]}, per optimiser, a = 1.65, 80 fresh seeds; everything frozen and hashed before any run. "
                             f"Scored {opt}: {S[opt]['n_crossed']} crossed, {val}: {v}.", "results/track_a/scores.json"))
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
    new = pd.DataFrame(ramp_rows() + act_rows() + band_rows() + c1_rows() + track_a_rows() + extra_rows())
    for c in d.columns:
        if c not in new.columns:
            new[c] = ""
    out = pd.concat([d, new[d.columns]], ignore_index=True)
    out.to_csv(ENUM, index=False)
    print(new[["id", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
