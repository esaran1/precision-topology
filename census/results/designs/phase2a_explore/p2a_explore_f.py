"""EXPLORATORY forecast-error estimate (population; release at the exact M(s0 = 0.5·s*); scale-only form; η = 1; no
seed).  For ρ = 2^-11 … 2^-14 in half-octaves: train to s ≥ 1.25·s_F (s every step), locate the first upward passage of
q by ρ₂ (every-20-step ρ₂, then every step in the bracketing 20), and emulate the proposed causal fold forecast at
f = 0.90 and 0.95: t_c = src.causal_forecast.cutoff_step(s, f·s_F); ŝ = src.causal_forecast.extrapolate_scale(s[:t_c],
quad, window 5%, ≥ 50, horizon 1.6·s_F) (the frozen 25aac9e functions, unchanged); t̂_F = first ŝ ≥ s_F; ṡ̂_F = the
extrapolated one-step increment there; ε̂ = (ṡ̂_F/s_F)/(ηΛ_F) with Λ_F from p2a_explore_a (P = I on the fixed-s tangent);
ŝ_c = s_F(1 + Ω₀ε̂^{2/3}); t_fc = first ŝ ≥ ŝ_c.  Compared with the observed t_cross and t_F (first s ≥ s_F)."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from results.designs.phase2a_explore.p2a_explore_c import step_fn
from src import simplicity_bias_v2 as v2
from src import causal_forecast as CF

REC = 20
A = json.load(open("results/designs/phase2a_explore/p2a_explore_a.json"))
LAMF = A["fold"]["1"]["Lambda_F"]


def forecast(s_vis, eta=1.0):
    cfg = CF.ForecastConfig(family="quad", window_frac=0.05, min_window=50, horizon_factor=1.6)
    sh, st, info = CF.extrapolate_scale(s_vis, cfg, C.S_FOLD, 10 ** 8)
    t_c = len(s_vis)
    kF = int(np.argmax(sh >= C.S_FOLD))
    sdot = sh[kF] - (sh[kF - 1] if kF > 0 else s_vis[-1])
    eps = (sdot / C.S_FOLD) / (eta * LAMF)
    s_c = C.S_FOLD * (1 + C.OMEGA0 * eps ** (2 / 3))
    kc = int(np.argmax(sh >= s_c))
    return {"t_F_fc": t_c + kF, "eps_fc": float(eps), "r_fc": float(s_c / C.S_FOLD - 1), "t_fc": t_c + kc,
            "s_c_fc": float(s_c), "rate_at_cutoff": info.get("growth_rate_at_cutoff")}


def main():
    import torch
    torch.set_num_threads(1)
    out = []
    for l2 in (-11, -11.5, -12, -12.5, -13, -13.5, -14):
        rho = 2.0 ** l2
        th = C.branch_theta("M", 1.7957)
        idle = [k for k in range(4) if th[12 + k] == 0.0]
        step = step_fn("scale", rho, 1.0, idle)
        S = [C.scale(th)]; rows = [th.copy()]; t = 0; t0 = time.time()
        while S[-1] < 1.25 * C.S_FOLD:
            th, _ = step(th); t += 1; S.append(C.scale(th))
            if t % REC == 0:
                rows.append(th.copy())
        S = np.array(S); rows = np.array(rows)
        r2 = v2.rho2_batch(rows, C.X, torch)
        tc = v2.crossing_step(r2, C.Q)
        th = rows[tc - 1].copy(); seg = [th.copy()]
        for _ in range(REC):
            th, _ = step(th); seg.append(th.copy())
        j = int(np.argmax(v2.rho2_batch(np.array(seg), C.X, torch) >= C.Q))
        t_obs = (tc - 1) * REC + j
        t_F = int(np.argmax(S >= C.S_FOLD))
        rec = {"log2rho": l2, "steps": t, "seconds": time.time() - t0, "t_obs": t_obs, "s_obs": float(S[t_obs]),
               "r_obs": float(S[t_obs] / C.S_FOLD - 1), "t_F": t_F, "delay_obs": t_obs - t_F,
               "t_switch": int(np.argmax(S >= C.S_STAR))}
        for f in (0.9, 0.95):
            t_c = CF.cutoff_step(S, f * C.S_FOLD)
            fc = forecast(S[:t_c])
            fc.update({"t_c": t_c, "err_cross": fc["t_fc"] - t_obs, "err_nodelay": fc["t_F_fc"] - t_obs,
                       "delay_fc": fc["t_fc"] - fc["t_F_fc"], "err_delay": (fc["t_fc"] - fc["t_F_fc"]) - (t_obs - t_F),
                       "ratio_r": rec["r_obs"] / fc["r_fc"], "ratio_delay": (t_obs - t_F) / (fc["t_fc"] - fc["t_F_fc"]),
                       "horizon_steps": t_obs - t_c})
            rec[f"f{f}"] = fc
        out.append(rec); print(json.dumps(rec), flush=True)
    open("results/designs/phase2a_explore/p2a_explore_f.json", "w").write(json.dumps({"label": "EXPLORATORY", "rows": out}, indent=1))


if __name__ == "__main__":
    main()
