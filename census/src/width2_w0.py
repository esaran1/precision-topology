"""Width 2, W0: Γ̂₂ and the conditional thresholds, validated (design §1, §3; revisions 2–3).  Nothing here trains.

    python -m src.width2_w0 gamma <act>        # act ∈ {f1.30, f1.50, tanh}: NM (4,000 starts) + DE (20 × 200 × 2,000)
    python -m src.width2_w0 scan <act>         # the registered scan R₂ ∈ [0.02, 1.00] step 0.01, 2,000 restarts
    python -m src.width2_w0 refine <act>       # bisection inside the bracketing grid step (to 2e−4 relative in s)
    python -m src.width2_w0 validate <act>     # within ±0.1 in R₂ of the threshold: ladder, 10× stricter, CMA-ES,
                                               #   cap raising (f_a); tanh: non-convergence stop waived (disclosed)
    python -m src.width2_w0 summary

Each finished unit (one scale, one search) is appended to results/width2_w0_parts/*.csv and skipped on restart.
Stop conditions (write a STOP line to the summary and exit non-zero): Γ̂₂ searches disagree beyond 1e−6 relative;
the audit finds a discarded candidate below the retained one (f_a; tanh waived); the restart ladder moves beyond
1e−9; the stricter or independent search finds a lower loss beyond 1e−9.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .width2_geometry import ACTS, gamma2_de, gamma2_multistart, searches_agree
from . import width2_conditional as wc

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "width2_w0_parts"
CAPS = (2000, 5000, 20000, 100000)
R2_LO, R2_HI, R2_STEP = 0.02, 1.00, 0.01
RESTARTS = 2000
REFINE_REL = 2e-4


def _append(fname, row):
    PARTS.mkdir(parents=True, exist_ok=True)
    f = PARTS / fname
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)


def _read(fname):
    f = PARTS / fname
    return pd.read_csv(f, float_precision="round_trip") if f.exists() else pd.DataFrame()


def gamma(act_name):
    act = ACTS[act_name]
    done = _read(f"gamma_{act_name}.csv")
    if not done.empty and set(done.search) >= {"nm", "de"}:
        return done
    r1 = gamma2_multistart(act, starts=4000, seed=0)
    _append(f"gamma_{act_name}.csv", {"search": "nm", "gamma_lo": r1["gamma_lo"], "gamma_hi": r1["gamma_hi"],
                                        **{f"theta{i}": v for i, v in enumerate(r1["theta"])}, "v0": r1["v"][0], "v1": r1["v"][1]})
    best = None
    for seed in range(1, 21):                                     # 20 independent DE runs, pop 200 × 2,000 gens
        r = gamma2_de(act, seed=seed, pop=200, gens=2000)
        if best is None or r["gamma_lo"] > best["gamma_lo"]:
            best = r
    _append(f"gamma_{act_name}.csv", {"search": "de", "gamma_lo": best["gamma_lo"], "gamma_hi": best["gamma_hi"],
                                        **{f"theta{i}": v for i, v in enumerate(best["theta"])}, "v0": best["v"][0], "v1": best["v"][1]})
    d = _read(f"gamma_{act_name}.csv")
    agree = searches_agree(r1, best) if act.name == "fa" else True    # tanh: sup not attained (Theorem 2), both < 1
    if not agree:
        _stop(act_name, f"Gamma2 searches disagree: NM {r1['gamma_lo']} vs DE {best['gamma_lo']}")
    return d


def gamma_hat(act_name):
    d = _read(f"gamma_{act_name}.csv")
    return float(d.gamma_lo.max())


def _stop(act_name, why):
    _append("stops.csv", {"act": act_name, "stop": why})
    print("STOP:", why, flush=True)
    raise SystemExit(2)


def _eval_scale(act_name, s, restarts, seed, cap, gtol=1e-8):
    act = ACTS[act_name]
    x, y = wc.population()
    r, cands = wc.search_batch(s, x, y, act, restarts=restarts, seed=seed, gtol=gtol, maxit=cap)
    aud = wc.audit(cands, r)
    gp = wc.directional_gplus(r["p"], r["sigma"], act) if r["p"] is not None else (float("nan"), float("nan"))
    cap_hits = sum(c["iters"] >= cap for c in cands[:-1])
    nonconv = sum("not_converged" in c["flags"] for c in cands)
    p = r["p"] if r["p"] is not None else [np.nan] * 5
    return {"act": act_name, "s": s, "restarts": restarts, "seed": seed, "cap": cap, "gtol": gtol,
            "retained_loss": r["loss"], "retained_constant": r["p"] is None, "sigma": r["sigma"],
            **{f"p{i}": float(v) for i, v in enumerate(p)}, "Gplus_lo": gp[0], "Gplus_hi": gp[1],
            "audit_ok": aud["ok"], "audit_n_below": aud["n_below"], "cap_hits": cap_hits, "not_converged": nonconv}


def scan(act_name):
    gh = gamma_hat(act_name)
    R = np.round(np.arange(R2_LO, R2_HI + R2_STEP / 2, R2_STEP), 10)
    done = _read(f"scan_{act_name}.csv")
    have = set() if done.empty else set(np.round(done.R2, 10))
    for k, r2 in enumerate(R):
        if r2 in have:
            continue
        row = _eval_scale(act_name, 2 * r2 / gh, RESTARTS, seed=1000 + k, cap=CAPS[0])
        row["R2"] = r2
        _append(f"scan_{act_name}.csv", row)
        print(act_name, r2, row["Gplus_lo"], row["audit_ok"], row["cap_hits"], flush=True)
        if not row["audit_ok"] and ACTS[act_name].name == "fa":
            _stop(act_name, f"audit: discarded candidate below the retained one at R2 = {r2}")
    d = _read(f"scan_{act_name}.csv").sort_values("R2")
    return wc.first_sign_change(d.R2.values, d.Gplus_lo.values)


def refine(act_name):
    """Bisection in s inside the bracketing grid step, to REFINE_REL relative width (the registered grid gives the
    bracket; bisection narrows it)."""
    gh = gamma_hat(act_name)
    d = _read(f"scan_{act_name}.csv").sort_values("R2")
    fs = wc.first_sign_change(d.R2.values, d.Gplus_lo.values)
    if fs["status"] != "defined" or fs["bracket"] is None:
        return fs
    lo, hi = 2 * fs["bracket"][0] / gh, 2 * fs["bracket"][1] / gh
    done = _read(f"refine_{act_name}.csv")
    k = 0 if done.empty else len(done)
    if not done.empty:
        for r in done.itertuples():
            if r.Gplus_lo > 0:
                hi = min(hi, r.s)
            else:
                lo = max(lo, r.s)
    while (hi - lo) / lo > REFINE_REL:
        mid = 0.5 * (lo + hi)
        row = _eval_scale(act_name, mid, RESTARTS, seed=5000 + k, cap=CAPS[0]); k += 1
        row["R2"] = mid * gh / 2
        _append(f"refine_{act_name}.csv", row)
        if row["Gplus_lo"] > 0:
            hi = mid
        else:
            lo = mid
        print(act_name, "refine", lo, hi, flush=True)
    return {"s_lo": lo, "s_hi": hi, "R2_lo": lo * gh / 2, "R2_hi": hi * gh / 2}


def validate(act_name):
    """Within ±0.1 in R₂ of the threshold (grid points): restart ladder (500 → 4,000), 10× stricter search
    (maxit ×10, gtol 1e−10), independent CMA-ES search; iteration caps raised for f_a until no restart hits the cap."""
    act = ACTS[act_name]
    gh = gamma_hat(act_name)
    d = _read(f"scan_{act_name}.csv").sort_values("R2")
    fs = wc.first_sign_change(d.R2.values, d.Gplus_lo.values)
    centre = 0.5 * sum(fs["bracket"]) if fs.get("bracket") else float(d.R2.iloc[0])
    pts = d[(d.R2 >= centre - 0.1 - 1e-12) & (d.R2 <= centre + 0.1 + 1e-12)]
    done = _read(f"validate_{act_name}.csv")
    have = set() if done.empty else set(np.round(done.R2, 10))
    x, y = wc.population()
    for r in pts.itertuples():
        if round(r.R2, 10) in have:
            continue
        s = 2 * r.R2 / gh
        cap = CAPS[0]; row0 = None
        for cap in CAPS:                                       # cap-raising rule (f_a): until no restart hits it
            row0 = _eval_scale(act_name, s, 4000, seed=9000, cap=cap)
            if row0["cap_hits"] == 0 or act.name != "fa":
                break
        lad = wc.convergence_ladder(s, x, y, act, seed=9000, maxit=cap) if act.name == "fa" else {"ok": True, "retained": []}
        strict = _eval_scale(act_name, s, RESTARTS, seed=9100, cap=cap * 10, gtol=1e-10)
        ind = wc.independent_search(s, x, y, act)
        row = {"act": act_name, "R2": r.R2, "s": s, "final_cap": cap, "cap_hits_at_final_cap": row0["cap_hits"],
               "retained_4000": row0["retained_loss"], "ladder_ok": lad["ok"], "ladder": json.dumps(lad["retained"]),
               "stricter_loss": strict["retained_loss"], "independent_loss": ind["loss"],
               "stricter_ok": wc.stricter_check(row0["retained_loss"], strict["retained_loss"]),
               "independent_ok": wc.stricter_check(row0["retained_loss"], ind["loss"]),
               "audit_ok": row0["audit_ok"], "Gplus_lo_4000": row0["Gplus_lo"]}
        _append(f"validate_{act_name}.csv", row)
        print(act_name, "validate", r.R2, {k: row[k] for k in ("final_cap", "ladder_ok", "stricter_ok", "independent_ok", "audit_ok")}, flush=True)
        if act.name == "fa" and not (row["ladder_ok"] and row["stricter_ok"] and row["independent_ok"] and row["audit_ok"]):
            _stop(act_name, f"validation failed at R2 = {r.R2}: {row}")
    return _read(f"validate_{act_name}.csv")


def summary():
    rows = []
    for a in ACTS:
        g = _read(f"gamma_{a}.csv"); sc = _read(f"scan_{a}.csv")
        if g.empty or sc.empty:
            continue
        sc = sc.sort_values("R2")
        fs = wc.first_sign_change(sc.R2.values, sc.Gplus_lo.values)
        rf = _read(f"refine_{a}.csv")
        rows.append({"act": a, "gamma2_hat": float(g.gamma_lo.max()), "status": fs["status"],
                     "grid_bracket": fs.get("bracket"), "sign_changes": fs.get("sign_changes"),
                     "cap_hit_rate_scan": float(sc.cap_hits.sum() / sc.restarts.sum()),
                     "refined_evaluations": 0 if rf.empty else len(rf)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "width2_w0_summary.csv", index=False)
    print(out.to_string(index=False))


# ------------------------------------------------------------------------------------------ direct small-scale check
SMALL_R2 = (0.001, 0.003, 0.01, 0.02)


def _classify_caps(cands, cap):
    """Cap-rule amendment: a cap hit counts only if still improving (loss fell > 1e−12 over the last 10% of iterations);
    otherwise it is a stall, located at the kink t = 0, the clip |t| = 1, or 'other'."""
    improving, stalls = [], []
    for c in cands[:-1]:
        if c["iters"] < cap:
            continue
        if np.isfinite(c.get("loss_at_90pct", np.nan)) and c["loss_at_90pct"] - c["loss"] > 1e-12:
            improving.append(c)
        else:
            t = abs(float(c["p"][4]))
            c["stall_at"] = "kink t=0" if t < 1e-6 else ("clip |t|=1" if t >= 1 - 1e-9 else "other")
            stalls.append(c)
    return improving, stalls


def _continue(c, s, x, y, act, extra):
    P, L, G, it = wc.bfgs_batch(lambda Q, rows: wc.loss_grad_batch(Q, np.array([float(c["sigma"])])[rows * 0], s, x, y, act),
                                np.array([c["p"]]), gtol=1e-8, maxit=extra)
    c["p"] = P[0]; c["loss"] = float(L[0]); c["gnorm"] = float(np.abs(G[0]).max()); c["iters"] += int(it[0])
    c["loss_at_90pct"] = float(wc.bfgs_batch.last_f90[0])
    c["flags"] = wc.degenerate(c["p"], c["gnorm"], 1e-8)


def _local_from_stall(c, s, x, y, act):
    """Local CMA-ES in the alternative parametrisation (v unconstrained, normalised), σ₀ = 0.05, from the stall."""
    from .cmaes import cma_es
    t = float(np.clip(c["p"][4], -1, 1)); v = np.array([t, c["sigma"] * (1 - abs(t))])
    q0 = np.r_[c["p"][:4], v]

    def f(q):
        vv = q[4:6]; n1 = abs(vv[0]) + abs(vv[1])
        if n1 < 1e-12:
            return 1e9
        vv = vv / n1
        z0 = s * (vv[0] * act.u(q[0] * x + q[1]) + vv[1] * act.u(q[2] * x + q[3]))
        b = wc.profile_b(z0, y); z = z0 + b
        return float((wc._softplus(z) - y * z).mean())
    return cma_es(f, q0, 0.05, max_generations=100, seed=int(c["k"]) + 1).best_f


def _is_pair(p, sigma):
    t = float(np.clip(p[4], -1, 1)); v = np.array([t, sigma * (1 - abs(t))])
    c = v[0] * p[0] + v[1] * p[2]
    return (abs(abs(t) - 0.5) < 1e-3 and abs(abs(p[0]) - abs(p[2])) < 1e-3 * max(abs(p[0]), 1e-9)
            and abs(c) < 1e-3 * (abs(v[0] * p[0]) + abs(v[1] * p[2]) + 1e-12)), float(c)


def smallscale(act_name):
    act = ACTS[act_name]
    gh = float(_read(f"gamma_{act_name}.csv").query("search == 'nm'").gamma_lo.iloc[0])
    x, y = wc.population()
    done = _read(f"smallscale_{act_name}.csv")
    have = set() if done.empty else set(np.round(done.R2, 10))
    for R2 in SMALL_R2:
        if round(R2, 10) in have:
            continue
        s = 2 * R2 / gh
        r, cands = wc.search_batch(s, x, y, act, restarts=4000, seed=7000, maxit=CAPS[0])
        cap = CAPS[0]
        for nxt in CAPS[1:]:                                   # cap raising for still-improving restarts only
            improving, _ = _classify_caps(cands, cap)
            if not improving:
                break
            for c in improving:
                _continue(c, s, x, y, act, nxt - cap)
            cap = nxt
        improving, stalls = _classify_caps(cands, cap)
        r = wc.retain(cands)
        aud = wc.audit(cands, r)
        stall_mins = [_local_from_stall(c, s, x, y, act) for c in stalls]
        stall_ok = all(m >= r["loss"] - 1e-9 for m in stall_mins)
        ladder = wc.retain(cands[:500] + [cands[-1]])["loss"]
        strict, _ = wc.search_batch(s, x, y, act, restarts=RESTARTS, seed=7100, gtol=1e-10, maxit=CAPS[0] * 10)
        ind = wc.independent_search(s, x, y, act)
        gp = wc.directional_gplus(r["p"], r["sigma"], act) if r["p"] is not None else (np.nan, np.nan)
        pair, c_lin = _is_pair(r["p"], r["sigma"]) if r["p"] is not None else (False, np.nan)
        locs = pd.Series([c["stall_at"] for c in stalls]).value_counts().to_dict() if stalls else {}
        row = {"act": act_name, "R2": R2, "s": s, "retained_loss": r["loss"], "Gplus_lo": gp[0], "Gplus_hi": gp[1],
               "placed": bool(gp[0] > 0), "is_cancelling_pair": pair, "linear_c": c_lin,
               **({f"p{i}": float(v) for i, v in enumerate(r["p"])} if r["p"] is not None else {}), "sigma": r["sigma"],
               "final_cap": cap, "still_improving_at_final_cap": len(improving), "stalls": len(stalls),
               "stall_locations": json.dumps(locs), "stall_local_min": min(stall_mins) if stall_mins else np.nan,
               "stall_local_ok": stall_ok, "audit_ok": aud["ok"], "ladder_500": ladder,
               "ladder_ok": abs(ladder - r["loss"]) <= 1e-9, "stricter_loss": strict["loss"],
               "stricter_ok": wc.stricter_check(r["loss"], strict["loss"]), "independent_loss": ind["loss"],
               "independent_ok": wc.stricter_check(r["loss"], ind["loss"])}
        _append(f"smallscale_{act_name}.csv", row)
        print(json.dumps({k: row[k] for k in ("act", "R2", "placed", "is_cancelling_pair", "Gplus_lo", "audit_ok",
                                              "stall_local_ok", "ladder_ok", "stricter_ok", "independent_ok", "stalls",
                                              "stall_locations", "still_improving_at_final_cap")}, default=str), flush=True)
        if not row["placed"]:
            print(f"UNPLACED at R2 = {R2}: the verdict would be wrong -- report immediately", flush=True)
    return _read(f"smallscale_{act_name}.csv")


if __name__ == "__main__":
    cmd, act = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    {"gamma": lambda: gamma(act), "scan": lambda: scan(act), "refine": lambda: refine(act),
     "validate": lambda: validate(act), "summary": summary, "smallscale": lambda: smallscale(act)}[cmd]()
