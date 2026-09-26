"""Follow-up registered test of the first-order coefficient c1 (Track 3; registration results/c1_followup_registration.md).

The ORIGINAL registered test (results/first_order_prediction.md, scored INCONCLUSIVE: feasible width 0.105 > 0.1) is
not re-scored and stays INCONCLUSIVE.  This module adds certified R_glob(a) brackets at further a, computed with the
registered bracket procedure (first_order._finite_job: start at A*/eps^1.5, 1% steps, bisection to 2e-4 relative,
stop at the first unresolved midpoint; conditional_certified.evaluate at tolerance 1e-7, tightened to 1e-9), with
Ghat(a) from the certified rescaled search (first_order.ghat_rescaled; Amendment 2's procedure, relative target 1e-6),
and scores the union of the original four and the added brackets with the registered estimator
first_order._feasible_c1, unchanged.

    python -m src.c1_followup model              # error model of the bracket procedure from the committed evaluations
    python -m src.c1_followup timing A           # Step 1: one full certified threshold at a = A (timing only)
    python -m src.c1_followup run                # the registered added a (resumable; one process, nice 15)
    python -m src.c1_followup score              # registered scoring
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "c1_followup_parts"
MEM_LIMIT_GB = 3.0
VALID_WIDTH = 0.1
A_ADD = (1.08, 1.09, 1.10, 1.11, 1.12)                   # registered added a (c1_followup_registration.md)
FROZEN = ("results/first_order_finite.csv", "results/first_order_c1.csv", "results/c1_followup_model.json",
          "results/c1_followup_timing_a1.0550.json")

# ------------------------------------------------------------------ scoring rule (pure; exercised in tests first)


def verdict(f_lo, f_hi, pred_lo, pred_hi, valid_width=VALID_WIDTH):
    """Registered rule of the follow-up.  INVALID (not a test) if the feasible set is wider than valid_width;
    FAIL if it is empty (no two-term law fits) or does not meet the derived c1 interval; PASS iff it meets it."""
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return "FAIL (no two-term law fits)"
    if f_hi - f_lo > valid_width:
        return "INVALID (feasible width > 0.1)"
    return "PASS" if (pred_hi >= f_lo and pred_lo <= f_hi) else "FAIL"


def feasible(eps, lo, hi, base_lo, base_hi):
    from .first_order import _feasible_c1
    return _feasible_c1(np.asarray(eps, float), np.asarray(lo, float), np.asarray(hi, float), base_lo, base_hi)


# ------------------------------------------------------------------ memory guard (stop, never pause)


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:                                          # macOS ru_maxrss is in bytes (peak)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def _watchdog(limit=MEM_LIMIT_GB):
    def run():
        while True:
            if _rss_gb() > limit:
                print(f"MEMORY STOP: rss > {limit} GB", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------ the registered bracket procedure (resumable)


def _cache_path(a):
    return PARTS / f"a{a:.4f}_evaluations.csv"


def threshold(a, log_timing=None):
    """first_order._finite_job, step for step, with Ghat from the rescaled certified search (Amendment 2) and every
    evaluation cached on disk (evaluate is deterministic; a restart replays the cache)."""
    from .conditional_certified import _population, evaluate
    from .first_order import REL_WIDTH, ghat_rescaled
    PARTS.mkdir(exist_ok=True)
    eps = a - 1.0
    x, y = _population()
    cp = _cache_path(a)
    cache = {}
    if cp.exists():
        for r_ in pd.read_csv(cp, float_precision="round_trip").to_dict("records"):
            cache[float(r_["s"])] = r_
    t0 = time.time()
    gp = PARTS / f"a{a:.4f}_ghat.json"
    if gp.exists():
        g = json.loads(gp.read_text())
    else:
        g = ghat_rescaled(a)
        g = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in g.items()}
        g["seconds"] = time.time() - t0
        gp.write_text(json.dumps(g, indent=1, default=float))
    assert bool(g["exclusion_closed"]), "Ghat exclusion did not close"
    A_star = float(pd.read_csv(RESULTS / "first_order_c1.csv").A_star_lo.iloc[0])
    rows = []

    def status(sv):
        if sv in cache:
            r_ = cache[sv]
        else:
            t1 = time.time()
            r_ = evaluate(sv, a, x, y, True)
            r_["seconds"] = time.time() - t1
            r_["rss_gb"] = _rss_gb()
            cache[sv] = r_
            pd.DataFrame(list(cache.values())).to_csv(cp, index=False)
        rows.append(r_)
        print(a, sv, r_["status"], f"{r_.get('seconds', float('nan')):.0f}s", flush=True)
        if log_timing:
            log_timing(r_)
        return r_["status"]
    lo = hi = round(A_star / eps ** 1.5, 6)
    st = status(lo)
    if st == "plus":
        while st == "plus":
            hi = lo; lo = round(lo * 0.99, 6); st = status(lo)
    else:
        while st != "plus":
            if st == "unresolved":
                break
            lo = hi; hi = round(hi * 1.01, 6); st = status(hi)
    unresolved = st == "unresolved"
    while not unresolved and (hi - lo) / lo > REL_WIDTH:
        mid = round(0.5 * (lo + hi), 9)
        st = status(mid)
        if st == "plus":
            hi = mid
        elif st == "minus":
            lo = mid
        else:
            unresolved = True
    G_lo, G_hi = float(g["Ghat_lo"]), float(g["Ghat_hi"])
    return {"a": a, "eps": eps, "s_lo": lo, "s_hi": hi, "s_rel_width": (hi - lo) / lo,
            "stopped_unresolved": unresolved, "Ghat_lo": G_lo, "Ghat_hi": G_hi, "Ghat_rel_width": (G_hi - G_lo) / G_lo,
            "Ghat_seconds": g.get("seconds", np.nan),
            "R_lo": lo * G_lo / 2, "R_hi": hi * G_hi / 2,
            "all_converged": all(bool(r_["converged"]) for r_ in rows), "evaluations": len(rows),
            "eval_seconds": float(sum(float(r_.get("seconds", 0) or 0) for r_ in rows))}


# ------------------------------------------------------------------ Step 1: error model and timing


def model():
    """Error model of the registered bracket procedure, from the committed a = 1.01-1.04 evaluations and the
    supplementary branch-root switch s*: the certified gap m- - m+ is quadratic in the relative offset d = s/s* - 1
    (continuous crossing), so a status resolves only for |d| above a floor set by the 1e-9 tolerance."""
    d = pd.read_csv(RESULTS / "first_order_finite_evaluations.csv", float_precision="round_trip")
    sp = pd.read_csv(RESULTS / "first_order_supplementary.csv", float_precision="round_trip").set_index("a")
    d["s_star"] = d.a.map(0.5 * (sp.s_lo + sp.s_hi))
    d["rel"] = d.s / d.s_star - 1
    d["diff"] = 0.5 * (d.m_minus_lo + d.m_minus_hi) - 0.5 * (d.m_plus_lo + d.m_plus_hi)
    d["halfwidth_sum"] = 0.5 * ((d.m_minus_hi - d.m_minus_lo) + (d.m_plus_hi - d.m_plus_lo))
    t9 = d[(d.tol == 1e-9) & (d.status != "unresolved")]
    kq = t9["diff"].abs() / t9.rel ** 2                       # diff = k d|d|
    k_med = float(kq.median())
    hw = float(d[d.tol == 1e-9].halfwidth_sum.median())
    zone = math.sqrt(hw / k_med)
    res = {"k_quadratic_median": k_med, "k_quadratic_min": float(kq.min()), "k_quadratic_max": float(kq.max()),
           "tol1e-9_half_width_sum_median": hw, "unresolved_zone_half_width_rel_s": zone,
           "largest_unresolved_abs_rel": float(d[d.status == "unresolved"].rel.abs().max()),
           "smallest_resolved_abs_rel_tol1e-9": float(t9.rel.abs().min()),
           "min_attainable_rel_width_s": 0.01 / 2 ** 4,
           "reason_min_width": "a bracket of relative width 0.01/2^5 = 3.1e-4 containing s* has an endpoint within "
                               "1.56e-4 < zone of s*, which cannot resolve; so bisection from the 1% bracket ends at "
                               "0.01/2^4 = 6.25e-4 at best (the registered 2e-4 target is unreachable at tolerance 1e-9)",
           "rows": d[["a", "s", "status", "tol", "rel", "diff", "halfwidth_sum"]].to_dict("records")}
    return res


def design_widths(new_eps, rel_w=0.01 / 16, positions=(0.32, 0.5, 0.68), seed=0, draws=200):
    """Projected feasible-c1 width from the registered estimator when each added a has a bracket of relative width
    rel_w in s and its truth at a relative position in the resolvable part of the bracket; truth from the four
    supplementary branch-root points (quadratic in eps beyond c1; design only, not a result)."""
    c = pd.read_csv(RESULTS / "first_order_c1.csv", float_precision="round_trip").iloc[0]
    t = pd.read_csv(RESULTS / "first_order_finite.csv", float_precision="round_trip")
    sp = pd.read_csv(RESULTS / "first_order_supplementary.csv", float_precision="round_trip")
    Rinf = 0.5 * (c.R_glob_inf_lo + c.R_glob_inf_hi)
    c1 = 0.5 * (c.c1_lo + c.c1_hi)
    e = sp.eps.values
    y = (0.5 * (sp.R_lo + sp.R_hi)) / Rinf - 1 - c1 * e
    p = np.polyfit(e, y / e ** 2, 1)
    R = lambda ep: Rinf * (1 + c1 * ep + (p[1] + p[0] * ep) * ep ** 2)
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(draws):
        E, L, H = list(t.eps), list(t.R_lo), list(t.R_hi)
        for ep in new_eps:
            q = rng.uniform(positions[0], positions[-1])
            E.append(ep); L.append(R(ep) * (1 - rel_w * q)); H.append(R(ep) * (1 + rel_w * (1 - q)))
        f = feasible(E, L, H, c.R_glob_inf_lo, c.R_glob_inf_hi)
        ws.append(f[1] - f[0])
    return {"new_eps": [float(v) for v in new_eps], "rel_w": rel_w, "width_min": float(np.min(ws)),
            "width_median": float(np.median(ws)), "width_max": float(np.max(ws))}


WIDTH_OUTCOMES = (0.01 / 16, 0.01 / 8, 0.01 / 4)      # attained so far: 1.01, 1.03 | 1.02, 1.055 (timing) | 1.04
WIDTH_PROBS = (0.4, 0.4, 0.2)


def _truth():
    c = pd.read_csv(RESULTS / "first_order_c1.csv", float_precision="round_trip").iloc[0]
    sp = pd.read_csv(RESULTS / "first_order_supplementary.csv", float_precision="round_trip")
    Rinf = 0.5 * (c.R_glob_inf_lo + c.R_glob_inf_hi)
    c1 = 0.5 * (c.c1_lo + c.c1_hi)
    e = sp.eps.values
    y = (0.5 * (sp.R_lo + sp.R_hi)) / Rinf - 1 - c1 * e
    p = np.polyfit(e, y / e ** 2, 1)
    return c, (lambda ep: Rinf * (1 + c1 * ep + (p[1] + p[0] * ep) * ep ** 2)), p


def design_mc(new_eps, draws=400, seed=1):
    """Monte Carlo of the projected feasible width when each added bracket's relative width in s is drawn from the
    attained outcomes (WIDTH_OUTCOMES, WIDTH_PROBS) and its truth sits in the resolvable middle of the bracket."""
    c, R, _ = _truth()
    t = pd.read_csv(RESULTS / "first_order_finite.csv", float_precision="round_trip")
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(draws):
        E, L, H = list(t.eps), list(t.R_lo), list(t.R_hi)
        for ep in new_eps:
            w = rng.choice(WIDTH_OUTCOMES, p=WIDTH_PROBS)
            q = rng.uniform(0.32, 0.68) if w == WIDTH_OUTCOMES[0] else rng.uniform(0.175 / (w * 1e4), 1 - 0.175 / (w * 1e4))
            E.append(ep); L.append(R(ep) * (1 - w * q)); H.append(R(ep) * (1 + w * (1 - q)))
        f = feasible(E, L, H, c.R_glob_inf_lo, c.R_glob_inf_hi)
        ws.append(f[1] - f[0])
    ws = np.array(ws)
    return {"new_eps": [float(v) for v in new_eps], "draws": draws, "width_median": float(np.median(ws)),
            "width_q90": float(np.quantile(ws, 0.9)), "width_max": float(ws.max()), "p_below_0.05": float(np.mean(ws < 0.05)),
            "p_below_0.1": float(np.mean(ws <= 0.1))}


def timing(a):
    """Step 1: one full certified threshold (registered bracket procedure + rescaled Ghat) at a = A, wall time and
    memory per evaluation.  The timing a is not used in the registered set."""
    os.nice(15)
    _watchdog()
    t0 = time.time()
    r = threshold(a)
    r["wall_seconds"] = time.time() - t0
    r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    ev = pd.read_csv(_cache_path(a))
    r["per_evaluation_seconds"] = ev.seconds.tolist()
    r["per_evaluation_status"] = ev.status.tolist()
    r["per_evaluation_tol"] = ev.tol.tolist()
    out = RESULTS / f"c1_followup_timing_a{a:.4f}.json"
    out.write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
    return r


# ------------------------------------------------------------------ Step 2: the registered computation and scoring


def sha256(path):
    import hashlib
    return hashlib.sha256((RESULTS.parent / path).read_bytes()).hexdigest()


def run():
    """The registered added a, one at a time, in one process (nice 15, memory stop at 3 GB); each finished a is written
    to c1_followup_parts/ and skipped on restart."""
    os.nice(15)
    _watchdog()
    PARTS.mkdir(exist_ok=True)
    for a in A_ADD:
        out = PARTS / f"a{a:.4f}.csv"
        if out.exists():
            continue
        t0 = time.time()
        r = threshold(a)
        r["wall_seconds"] = time.time() - t0
        pd.DataFrame([r]).to_csv(out, index=False)
        print("finished a =", a, r, flush=True)
    done = [PARTS / f"a{a:.4f}.csv" for a in A_ADD if (PARTS / f"a{a:.4f}.csv").exists()]
    t = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in done])
    t.to_csv(RESULTS / "c1_followup_finite.csv", index=False)
    ev = [pd.read_csv(_cache_path(a), float_precision="round_trip") for a in A_ADD if _cache_path(a).exists()]
    pd.concat(ev).to_csv(RESULTS / "c1_followup_finite_evaluations.csv", index=False)
    print(t.to_string(index=False))


def score():
    """Registered scoring: the original four registered brackets (first_order_finite.csv, frozen) and the added
    brackets that certified (all_converged, Ghat exclusion closed), through first_order._feasible_c1 unchanged."""
    orig = pd.read_csv(RESULTS / "first_order_finite.csv", float_precision="round_trip")
    add = pd.read_csv(RESULTS / "c1_followup_finite.csv", float_precision="round_trip")
    c = pd.read_csv(RESULTS / "first_order_c1.csv", float_precision="round_trip").iloc[0]
    ok = add[add.all_converged.astype(bool)]
    eps = np.concatenate([orig.eps.values, ok.eps.values])
    lo = np.concatenate([orig.R_lo.values, ok.R_lo.values])
    hi = np.concatenate([orig.R_hi.values, ok.R_hi.values])
    f_lo, f_hi = feasible(eps, lo, hi, c.R_glob_inf_lo, c.R_glob_inf_hi)
    v = verdict(f_lo, f_hi, c.c1_lo, c.c1_hi)
    empty = not np.isfinite(f_lo)
    comp = {f"competing_{n}": ("n/a" if empty else "inside" if f_lo <= x <= f_hi else "below" if x < f_lo else "above")
            for n, x in (("0.49_earlier", 0.49), ("-0.377_K_only", -0.377), ("0.662_switch_only", 0.662),
                         ("0_no_first_order", 0.0))}
    ref = pd.read_csv(RESULTS / "first_order_supplementary_scores.csv").iloc[0]
    row = {"verdict": v, "feasible_lo": f_lo, "feasible_hi": f_hi, "feasible_width": (f_hi - f_lo) if not empty else np.nan,
           "pred_lo": c.c1_lo, "pred_hi": c.c1_hi, "n_original": len(orig), "n_added_registered": len(A_ADD),
           "n_added_certified": len(ok), "added_a": [float(x) for x in ok.a], "dropped_a": [float(x) for x in add.a[
               ~add.all_converged.astype(bool)]] + [float(a) for a in A_ADD if a not in set(add.a.round(4))],
           "added_s_rel_widths": [float(x) for x in ok.s_rel_width], "added_stopped_unresolved": [bool(x) for x in ok.stopped_unresolved],
           "added_Ghat_rel_widths": [float(x) for x in ok.Ghat_rel_width],
           "original_test_verdict_unchanged": "INCONCLUSIVE (first_order_scores.csv)",
           "supplementary_branch_root_set_for_reference": [float(ref.feasible_lo), float(ref.feasible_hi)],
           "frozen_sha256": {p: sha256(p) for p in FROZEN}, **comp}
    (RESULTS / "c1_followup_summary.json").write_text(json.dumps(row, indent=1, default=float))
    pd.DataFrame([{k: (json.dumps(x) if isinstance(x, (list, dict)) else x) for k, x in row.items()}]).to_csv(
        RESULTS / "c1_followup_scores.csv", index=False)
    print(json.dumps(row, indent=1, default=float))
    return row


def export_and_check(names_only=False):
    """Independent check (verify_certificates.check_finite, one worker): export the status certificate at each added
    a's bracket ends (s_lo: 'minus', s_hi: 'plus'), append their SHA-256 to certificates_manifest.csv, check them.
    Order: largest a first (those bind the feasible set).  Results: results/certificate_checks/<name>.json and
    c1_followup_checks.json."""
    import csv
    import hashlib
    from .cert_export import CERTS, finite as export_finite
    os.nice(15)
    add = pd.read_csv(RESULTS / "c1_followup_finite.csv", float_precision="round_trip").sort_values("a", ascending=False)
    jobs = []
    for _, r in add.iterrows():
        for end, sv, want in (("hi", r.s_hi, "plus"), ("lo", r.s_lo, "minus")):
            jobs.append((f"c1f_a{r.a:.2f}_{end}", float(r.a), float(sv), want))
    if names_only:
        return jobs
    outp = RESULTS / "c1_followup_checks.json"
    done = json.loads(outp.read_text()) if outp.exists() else {}
    man = RESULTS / "certificates_manifest.csv"
    for name, a, sv, want in jobs:
        if name in done:
            continue
        t0 = time.time()
        if not (CERTS / f"{name}.json").exists():
            meta = export_finite(a, sv, name)
        else:
            meta = json.loads((CERTS / f"{name}.json").read_text())
        have = {r["file"] for r in csv.DictReader(man.open())}
        with man.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "sha256", "bytes", "regenerate"], lineterminator="\n")
            for f in (CERTS / f"{name}.json", CERTS / f"{name}.npz"):
                key = f"results/certificates/{f.name}"
                if key not in have:
                    w.writerow({"file": key, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                "bytes": f.stat().st_size, "regenerate": meta.get("regenerate", "")})
        t1 = time.time()
        from .verify_certificates import check_finite
        res = check_finite(name, verbose=False, workers=1)
        done[name] = {"a": a, "s": sv, "status_exported": meta["status"], "status_expected": want,
                      "status_matches": meta["status"] == want, "checker_pass": bool(res["pass"]),
                      "checks": res["checks"], "export_seconds": t1 - t0, "check_seconds": time.time() - t1}
        outp.write_text(json.dumps(done, indent=1, default=str))
        print(name, done[name], flush=True)
    return done


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "model"
    if cmd == "model":
        m = model()
        m["design"] = [design_widths(v) for v in (
            [0.005, 0.0075, 0.015, 0.025, 0.035], list(np.arange(0.005, 0.0401, 0.005)),
            list(np.arange(0.0025, 0.0376, 0.0025)), [0.015, 0.025, 0.035],
            [0.05, 0.075, 0.1], [0.06, 0.08, 0.1], [0.05, 0.06, 0.07, 0.08, 0.09, 0.1], [0.03, 0.06, 0.09],
            [0.07, 0.1], [0.08, 0.1], [0.1]) ] + \
            [design_widths(v, rel_w=0.01 / 8) for v in ([0.05, 0.075, 0.1], [0.06, 0.08, 0.1], [0.07, 0.1])]
        m["design_at_registered_target_2e-4"] = [design_widths(v, rel_w=0.01 / 64) for v in (
            [0.005, 0.0075, 0.015, 0.025, 0.035], [0.015, 0.025, 0.035])]
        m["design_mc"] = [design_mc(v, draws=150) for v in (
            [0.1], [0.1, 0.12], [0.08, 0.1, 0.12], [0.08, 0.09, 0.1, 0.11, 0.12], [0.07, 0.08, 0.09, 0.1],
            list(np.arange(0.005, 0.0401, 0.005)))]
        m["registered_design"] = [round(a - 1, 4) for a in A_ADD]
        (RESULTS / "c1_followup_model.json").write_text(json.dumps(m, indent=1, default=float))
        print(json.dumps({k: v for k, v in m.items() if k != "rows"}, indent=1, default=float))
    elif cmd == "timing":
        timing(float(sys.argv[2]))
    elif cmd == "run":
        run()
    elif cmd == "score":
        score()
    elif cmd == "check":
        export_and_check()
