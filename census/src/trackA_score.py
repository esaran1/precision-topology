"""Track A scoring entry point: a POST-REGISTRATION PLUMBING FIX (author's approval, 2026-10-10).

The registered `python -m src.trackA_causal score` crashed before any verdict: `table()` returns two columns that
`score_track_a`'s docstring does not list ("status", a string; "P_relerr", descriptive only), and `score_track_a`
converts every column it is given to float (ValueError: could not convert string to float: 'ok').  The registered
module (src/trackA_causal.py, hashed in results/trackA/registration.sha256) is NOT modified.  This module calls the
registered functions unchanged and passes `score_track_a` only the columns its docstring documents.  Tolerances,
criteria, the miss rule, the bootstrap seed and the descriptive statistics are the registered ones.

  pilot_check   the registered scorer through this entry point on the PILOT data (pilot_runs.jsonl); asserts it
                reproduces the in-sample pilot numbers in pilot.json exactly; writes nothing
  score         -> results/trackA/scores.json (the registered verdict and DESCRIPTIVE statistics)
"""
from __future__ import annotations

import json
import sys

from . import trackA_causal as T

RESULTS = T.RESULTS
OUT = RESULTS / "trackA"

# The columns score_track_a's docstring lists (plus "seed", which it never reads).
SCORE_COLUMNS = ("seed", "frozen_ok", "crossed", "t_obs", "s_obs", "t_sw", "s_sw", "t_c", "t_fc", "t_sw_fc", "r_fc",
                 "nan_identical")


def documented_columns(R):
    """table()'s output restricted to the columns score_track_a documents."""
    return {k: R[k] for k in SCORE_COLUMNS}


def score_table(rows, obs, tol):
    """The registered verdict on table(rows, obs) at the registered f and primary forecast."""
    return T.score_track_a(documented_columns(T.table(rows, obs, T.F, "primary")), tol)


def pilot_check():
    """The pilot data through this entry point; every in-sample pilot number must be reproduced exactly."""
    rows = T._rows(OUT / "pilot_runs.jsonl")
    assert sorted(r["seed"] for r in rows) == list(T.PILOT_SEEDS), "every pilot seed exactly once"
    obs = [r["observed_PILOT_ONLY"] for r in rows if r.get("observed_PILOT_ONLY")]
    pj = json.loads((OUT / "pilot.json").read_text())
    tol = {k: pj["tolerances"][k] for k in ("tau_cross", "tau_lag", "band")}
    assert tol == T.TOLERANCES, "pilot tolerances differ from the registered ones"
    res = score_table(rows, obs, tol)
    ref = pj["primary_with_criteria"]["criteria_at_pilot_tolerances_IN_SAMPLE"]
    got = {k: {kk: res[k].get(kk) for kk in ref[k]} for k in ref}
    # the pilot has 40 seeds < 60, so validity is not met there; the criteria are compared before that overlay
    for k in ref:
        if not res["valid"]:
            got[k]["verdict"] = res[k]["verdict_if_valid"]
    out = {"n_scored": res["n_scored"], "n_scored_miss": res["n_scored_miss"], "valid": res["valid"],
           "validity": res["validity"], "criteria": got, "reference": ref, "identical": got == ref}
    print(json.dumps(T._jsonable(out), indent=1))
    assert got == ref, "the entry point does not reproduce the pilot numbers"
    return out


def score():
    """scores.json, as the registered score() would have written it, with the documented columns passed."""
    T.assert_registration()
    T._assert_forecasts()
    rows, obs = T._rows(OUT / "runs.jsonl"), T._rows(OUT / "observed.jsonl")
    tol = T.tolerances()
    res = score_table(rows, obs, tol)
    res["DESCRIPTIVE"] = {"primary_errors": T._desc_stats(T.table(rows, obs, T.F, "primary"), tol),
                          **{k: T._desc_stats(T.table(rows, obs, T.F, k), tol) for k in T.DESC_VARIANTS},
                          "f_0.95_primary_rule": T._desc_stats(T.table(rows, obs, T.F_DESC, "primary"), tol)}
    res["forecasts_sha256"] = T._sha(OUT / "forecasts.sha256")
    res["scoring_entry_point"] = ("src/trackA_score.py (post-registration plumbing fix, author 2026-10-10): "
                                  "score_track_a given only its documented columns; registered module unchanged")
    (OUT / "scores.json").write_text(json.dumps(T._jsonable(res), indent=1) + "\n")
    print(json.dumps(T._jsonable({k: res[k] for k in ("outcome", "validity")})))


if __name__ == "__main__":
    {"pilot_check": pilot_check, "score": score}[sys.argv[1]]()
