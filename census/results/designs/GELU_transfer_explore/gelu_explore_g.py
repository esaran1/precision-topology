"""EXPLORATORY (GELU-T, after approval, before registration; coordinator request 2026-09-29).  Population and pilot seeds
only; no registered seed (876,000-876,079) is used.  Nothing here is registered.  Writes gelu_explore_g.json beside it.

(1) Gate options.  N_POP population random starts with the registered random-start rule (src.gelu_transfer.init_state:
    coordinates 0, 1, 3 of U(-1, 1)^4 from a torch Generator seeded with the start's seed, w2 = +s0) and the registered
    hold (GD lr 0.3 on (w1, b1, b2), w2 fixed; W = max(4000, ceil(25/(0.3 lambda_min,pop(s0)))) = 4000), G (dense
    phase2b_ordering.state) at the start and after every hold step; release classification by damped Newton at s0 on the
    POPULATION against the frozen theta*_pop(s0) (target) and its mirror (w1 -> -w1; a stationary point of the symmetric
    population objective), with the registered tolerances.  Generator seeds 8,760,000 + i (exploratory; outside every
    registered range).  Categories: (a) G > 0 at the start; (b) G > 0 at some hold step after the start, not at the start;
    (c) G <= 0 at the start and every hold step AND on target or mirror at release; (d) on target or mirror regardless of
    G; (e) on neither.  Clopper-Pearson 95% interval for (c); P(>= 64 of 80 satisfy (c)) under Binomial(80, p) at the
    point estimate and at the interval ends, and under the Beta(1 + k, 1 + n - k) posterior predictive (beta-binomial).
(2) V7 windows on the 20 pilot runs (both arms, pilot seeds 876,900-876,909, rho = 1, to t_sw; the registered pipeline,
    reproduced bit for bit against pilot.json's chi_path_max): chi_t = ((s_{t+1} - s_t)/s_t)/(eta lambda_min(H(s_t)))
    for 0 <= t < t_sw on the occupied branch; per run the step and s/s_switch of the maximum; q90 and median of the
    per-run maximum over three windows: all t < t_sw (the registered text), s_t >= 0.8 s_switch, and the last
    5/(eta lambda_min(s_switch)) steps before t_sw; and q90 / median of chi at t_sw (the closed form's chi)."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
os.nice(15)
import torch  # noqa: E402

torch.set_num_threads(1)
from src import gelu_transfer as G  # noqa: E402
from src.act_general import population  # noqa: E402

N_POP = 600
SEED0 = 8_760_000
OUT = HERE / "gelu_explore_g.json"


def _logC(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_sf(kmin, n, p):
    """P(X >= kmin), X ~ Binomial(n, p), exact summation."""
    if p <= 0:
        return 0.0 if kmin > 0 else 1.0
    if p >= 1:
        return 1.0
    return float(sum(math.exp(_logC(n, k) + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(kmin, n + 1)))


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% interval by bisection on the binomial tails."""
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if f(m) < target:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)
    low = 0.0 if k == 0 else solve(lambda q: binom_sf(k, n, q), alpha / 2)          # P(X >= k | q) = α/2
    high = 1.0 if k == n else solve(lambda q: binom_sf(k + 1, n, q), 1 - alpha / 2)  # P(X <= k | q) = α/2
    return low, high


def betabinom_sf(kmin, n, a, b):
    """P(X >= kmin), X ~ BetaBinomial(n, a, b)."""
    lB = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return float(sum(math.exp(_logC(n, k) + lB(k + a, n - k + b) - lB(a, b)) for k in range(kmin, n + 1)))


def memory_gate():
    from src.act_fold import check_memory
    ok, f, w = check_memory()
    if not ok:
        raise SystemExit(f"STOP (memory): free {f}%, swap free {w} MB")
    return f, w


def part1(land):
    s0 = land["s0"]
    zt = np.array(land["z_pop_s0"]); zm = np.array([-zt[0], zt[1], zt[2]])
    P = G.PopProblem()
    zt_n, _, _, okt = G.newton_at(zt, s0, P); zm_n, gm, lm, okm = G.newton_at(zm, s0, P)
    assert okt and okm and np.abs(zt_n - zt).max() < 1e-9 and np.abs(zm_n - zm).max() < 1e-9, "mirror not stationary"
    W = G.w_hold(land["lam_min_s0"])
    x, y = population()
    X = torch.tensor(np.asarray(x, float), dtype=torch.float64); Y = torch.tensor(np.asarray(y, float), dtype=torch.float64)
    u = G._act().torch_u
    rows = []
    for i in range(N_POP):
        th0 = G.init_state(G.PRIMARY, SEED0 + i, land["z_pop_s0"], s0)
        th, rec = G.hold(th0, W, X, Y, u)
        z = th[[0, 1, 3]]
        zn, gn, ln, okn = G.newton_at(z, s0, P)
        c = G.classify_release(z, zn, okn, {1: zt, -1: zm})
        rows.append({"seed": SEED0 + i, "init_placed": rec["init_placed"], "hold_n_G_pos": rec["hold_n_G_pos"],
                     "hold_G_positive": rec["hold_G_positive"], "copy": c, "newton_ok": okn,
                     "newton_w1": float(zn[0]), "release_w1": float(z[0])})
        if i % 50 == 49:
            print("part1", i + 1, time.strftime("%H:%M:%S"), flush=True)
            G._rss_guard()
    n = len(rows)
    a = sum(r["init_placed"] for r in rows)
    b = sum((not r["init_placed"]) and r["hold_G_positive"] for r in rows)
    onb = [r["copy"] != 0 for r in rows]
    c = sum((not r["hold_G_positive"]) and o for r, o in zip(rows, onb))
    d = sum(onb); e = n - d
    lo, hi = clopper_pearson(c, n)
    p = c / n
    p64 = lambda q: binom_sf(64, 80, q)
    neither = [r for r in rows if r["copy"] == 0]
    out = {"n": n, "W_hold": W, "generator_seeds": [SEED0, SEED0 + n - 1],
           "a_G_pos_at_init": [a, a / n], "b_G_pos_in_hold_not_init": [b, b / n],
           "c_no_G_pos_and_on_branch": [c, p], "d_on_branch_any_G": [d, d / n], "e_neither": [e, e / n],
           "n_target": sum(r["copy"] == 1 for r in rows), "n_mirror": sum(r["copy"] == -1 for r in rows),
           "d_and_G_pos": d - c,
           "neither_newton_ok": sum(r["newton_ok"] for r in neither),
           "neither_abs_newton_w1_lt_1e-3": sum(r["newton_ok"] and abs(r["newton_w1"]) < 1e-3 for r in neither),
           "c_clopper_pearson_95": [lo, hi],
           "P_ge64of80_c_at_point": p64(p), "P_ge64of80_c_at_ci_lo": p64(lo), "P_ge64of80_c_at_ci_hi": p64(hi),
           "P_ge64of80_c_beta_binomial": betabinom_sf(64, 80, 1 + c, 1 + n - c),
           "P_ge64of80_d_at_point": p64(d / n)}
    print(json.dumps(out, indent=1), flush=True)
    return out


def part2(land, fz, pil):
    ref = {(r["arm"], r["seed"]): r for r in pil["runs"]["1.0"]}
    rows = []
    for arm in G.ARMS:
        for seed in G.PILOT_SEEDS:
            fr = fz[seed]
            th_rel, rec, (X, Y, u, P) = G.run_start(arm, seed, fr, land)
            cf = fr["copies"][str(rec["copy_at_release"])]
            s_sw, lam_sw = cf["s_switch"], cf["lam_min_switch"]
            Wp = G.release_train(th_rel, X, Y, u, 1.0, G.BUDGET, stop_s=s_sw)
            s = np.abs(Wp[:, 2])
            t_sw = G.t_switch(s, s_sw)
            B = G.GBranch(land["s0"], np.array(cf["z_s0"]), P, 0.95 * land["s0"], G.S_HI_FRAC * land["s_pop"],
                          G.GRID_H_FRAC * land["s_pop"])
            cp = G.chi_path(s, t_sw, B.lam_min)
            _, chi_tsw, _ = G.closed_form(s, t_sw, s_sw, cf["kappa"], lam_sw)
            r0 = ref[(arm, seed)]
            assert r0["t_sw"] == t_sw and r0["chi_path_max"] == float(cp.max()), (arm, seed, "not reproduced")
            k = int(np.argmax(cp))
            win08 = s[:t_sw] >= 0.8 * s_sw
            relax = 1.0 / (G.ETA * lam_sw)
            t0 = max(0, t_sw - int(math.ceil(5 * relax)))
            rows.append({"arm": arm, "seed": seed, "copy": rec["copy_at_release"], "t_sw": int(t_sw),
                         "argmax_step": k, "argmax_s_over_s_switch": float(s[k] / s_sw),
                         "max_all": float(cp.max()), "max_s_ge_0p8": float(cp[win08].max()),
                         "relax_steps_at_switch": relax, "window_5relax_start": t0,
                         "max_last_5_relax": float(cp[t0:].max()), "chi_tsw": chi_tsw,
                         "chi_at_release_step0": float(cp[0])})
            print(json.dumps(rows[-1]), flush=True)
            G._rss_guard()
    summ = {}
    for key in ("max_all", "max_s_ge_0p8", "max_last_5_relax", "chi_tsw"):
        for grp, sel in (("pooled", rows), ("random", [r for r in rows if r["arm"] == "random"]),
                         ("branch", [r for r in rows if r["arm"] == "branch"])):
            v = np.array([r[key] for r in sel])
            summ[f"{key}_{grp}"] = {"q90": float(np.percentile(v, 90)), "median": float(np.median(v)),
                                    "n_gt_0p25": int((v > 0.25).sum()), "n": len(v)}
    out = {"runs": rows, "summary": summ}
    print(json.dumps(summ, indent=1), flush=True)
    return out


if __name__ == "__main__":
    f, w = memory_gate()
    print("memory free", f, "% swap free", w, "MB", flush=True)
    land = json.loads((G.OUT / "landscape.json").read_text())
    fz = G._frozen()
    pil = json.loads((G.OUT / "pilot.json").read_text())
    res = {"label": "EXPLORATORY (population and pilot seeds only; not registered)", "part2_V7": part2(land, fz, pil)}
    memory_gate()
    res["part1_gate"] = part1(land)
    OUT.write_text(json.dumps(res, indent=1, default=float))
