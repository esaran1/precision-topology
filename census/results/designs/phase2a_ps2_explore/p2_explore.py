"""EXPLORATORY (design 2A-PS2; NOT registered).  Exploration seeds 2,930,112-2,930,199 ONLY (the 2A-PS exploration
range 2,930,000-2,930,199, the part never drawn by p2a_explore_g/h/i).  No registered or pilot seed of 2A-PS (2,975,xxx,
2,976,xxx) or of 2A-PS2 is drawn.

Per seed, with the REGISTERED 2A-PS functions unchanged (src/phase2a_ps.py freeze_raw/evaluate; posthoc seed_M,
seed_branch, wider_class):
  (a) activity test at the release: shares |v_k|/s of the units the 2A-PS rule (v != 0.0) counts active; the Phase 3
      test (|v_k| >= 1e-8*s) and whether it drops a unit;
  (b) per-seed budget input: rho*t* = int_{s0}^{0.9999 s_F} ds / (n*(-dL_M/ds)) on the seed's own M (posthoc's
      quasi_static_rho_time; n = 3 theta-active units); and a CALIBRATION run at rho = 2^-12 (scale-only rule from the
      exact release, theta-active units in a_hat): rho*t_F, rho*t_c(0.95), rho2 every 64 steps -> approximate crossing,
      and the run end (crossing and s_F, or 1.25*s_F, or a hard cap of ceil(4*64/rho) steps);
  (c) classes: registered (30 minima at 1.01 s_F), the posthoc WIDER search (1.01/1.02/1.05/1.10 s_F x amplitudes
      1/2/4 x 30, default_rng([seed, 20261005])), S's rho2 at 1.01 s_F and the first s/s_F on 1.00..1.25 with S's
      rho2 >= q.
(b) and (c) for SCOREABLE seeds only.  One process, nice 15, one thread; stop above 1 GB RSS; wall-clock deadline.
Usage (from census/):  python results/designs/phase2a_ps2_explore/p2_explore.py [deadline_minutes]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
try:
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
except OSError:
    pass
import json, math, resource, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import numpy as np
from src import phase2a_ps as PS
from src import phase2a_ps_posthoc as PH
from src import sb_fold as SBF

SEEDS = range(2_930_112, 2_930_200)
THETA = 1e-8
L2_CAL = -12.0
OUT = HERE / "p2_explore.jsonl"
RSS_STOP = 1.0e9


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS


def guard():
    if rss() > RSS_STOP:
        raise SystemExit(f"STOP: RSS {rss() / 1e9:.2f} GB > 1 GB")


def theta_active(v, theta=THETA):
    s = float(np.abs(v).sum())
    return [k for k in range(len(v)) if s > 0 and abs(v[k]) >= theta * s]


def rho_t_star(M, s0, s_F, n):
    ss = np.linspace(s0, 0.9999 * s_F, 121)
    h = PH.SLOPE_DS_REL * s_F
    d = []
    for s in ss:
        a, b = M.loss(min(s + h, 0.99995 * s_F)), M.loss(s - h)
        d.append(np.nan if a is None or b is None else (a - b) / (min(s + h, 0.99995 * s_F) - (s - h)))
    d = np.array(d)
    if not np.all(np.isfinite(d)):
        return None, d
    return PH.quasi_static_rho_time(ss, d, s0, 0.9999 * s_F, n_active=n), d


def calib_run(th0, act, s_F, X, Y):
    rho = PS.rho_of(L2_CAL)
    cap = int(math.ceil(4 * 64 / rho))
    step = PS.make_step(rho, act, X, Y)
    g2 = PS.Rho2Grid(X)
    th = np.array(th0, float)
    sg0 = np.sign(th[12:16][act])
    below = g2(th) < PS.Q
    t_c = t_F = t_x = sign_t = None
    s = PS.scale(th)
    t = 0
    for t in range(1, cap + 1):
        th = step(th)
        s = PS.scale(th)
        if sign_t is None and np.any(np.sign(th[12:16][act]) != sg0):
            sign_t = t
        if t_c is None and s >= 0.95 * s_F:
            t_c = t
        if t_F is None and s >= s_F:
            t_F = t
        if t_x is None and t % 64 == 0:
            r = g2(th)
            if r >= PS.Q and below:
                t_x = t
            elif r < PS.Q:
                below = True
        if s >= 1.25 * s_F or (t_x is not None and t_F is not None):
            break
        if t % 100_000 == 0:
            guard()
    f = lambda x: None if x is None else rho * x
    return {"rho": rho, "cap": cap, "rt_c": f(t_c), "rt_F": f(t_F), "rt_cross": f(t_x), "rt_end": rho * t,
            "s_end_over_sF": s / s_F, "crossed": t_x is not None, "s_cross_over_sF": None,
            "reached_1.25": bool(s >= 1.25 * s_F), "sign_change_t": sign_t}


def main():
    deadline = time.time() + 60 * float(sys.argv[1] if len(sys.argv) > 1 else 40)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["seed"] for l in open(OUT)}
    for seed in SEEDS:
        if seed in done:
            continue
        if time.time() > deadline:
            print("deadline reached", flush=True)
            break
        t0 = time.time()
        rec = PS.freeze_raw(seed, timing=False)
        ev = PS.evaluate(rec)
        row = {"seed": seed, "status": ev["status"], "class_registered": ev.get("class"),
               "n_ge_q_101": ev.get("class_n_rho2_ge_q"), "s_F": rec.get("s_F"), "s0": rec.get("s0"),
               "sF_over_sstar": rec.get("sF_over_sstar"), "rho2_S_101": rec.get("rho2_S_at_1.01sF")}
        if rec.get("release_theta") is not None:
            th0 = np.array(rec["release_theta"], float)
            v = th0[12:16]
            s = float(np.abs(v).sum())
            reg = [k for k in range(4) if v[k] != 0.0]
            act = theta_active(v)
            row.update(n_reg_active=len(reg), n_theta_active=len(act),
                       min_share=float(min(abs(v[k]) / s for k in reg)) if reg else None,
                       activity_drops_unit=bool(len(act) < len(reg)))
        if ev["status"] == "scoreable":
            X, Y = PS.sample(seed)
            fs = dict(rec)
            M = PH.seed_M(fs, X, Y)
            s_F = rec["s_F"]
            rt, d = rho_t_star(M, rec["s0"], s_F, row["n_theta_active"])
            row["rho_t_star"] = rt
            row["dLds_at_s0_and_sF"] = [float(d[0]), float(d[-1])]
            if row["n_theta_active"] == 3:
                row["calib"] = calib_run(th0, theta_active(th0[12:16]), s_F, X, Y)
            # wider search (posthoc job_none's blocks, no S labels)
            Pb = SBF.run_to_full(M.theta(0.9999 * s_F)[None])[0]
            rng = np.random.default_rng([seed, PH.WIDE_STREAM])
            blocks = []
            for f in PH.WIDE_SCALES:
                for amp in PH.WIDE_AMPS:
                    P0 = Pb[None] + rng.normal(0, 1, (PH.WIDE_N, 16)) * amp * (0.05 * np.abs(Pb) + 0.02)
                    P0[:, 12:] = np.abs(P0[:, 12:])
                    Pm, L, gn = SBF.local_min_batch(P0, np.full(PH.WIDE_N, f * s_F), X, Y)
                    r = np.array([PS.rho2_static(p, X) for p in Pm])
                    blocks.append({"scale": f, "amp": amp, "n": PH.WIDE_N, "n_ge_q": int((r >= PS.Q).sum())})
            row["wide"] = PH.wider_class(blocks)
            S = PH.seed_branch("S", X, Y)
            prof = []
            for f in PH.S_PROFILE:
                P = PH.branch_P(S, f * s_F)
                prof.append(PS.rho2_static(P, X) if P is not None else np.nan)
            row["S_rho2_profile"] = [None if not np.isfinite(x) else float(x) for x in prof]
            row["S_first_ge_q_over_sF"] = PH.first_crossing_scale(PH.S_PROFILE, prof)
        row["secs"] = round(time.time() - t0, 1)
        row["peak_rss_gb"] = round(rss() / 1e9, 3)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("seed", "status", "class_registered", "n_theta_active",
                                                  "rho_t_star", "secs")}
                         | {"wide": row.get("wide", {}).get("pooled"),
                            "calib": {k: row.get("calib", {}).get(k) for k in ("rt_F", "rt_end", "crossed")}}),
              flush=True)
        guard()
    print("peak RSS GB", round(rss() / 1e9, 3), flush=True)


if __name__ == "__main__":
    main()
