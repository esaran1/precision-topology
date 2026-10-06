"""EXPLORATORY training runs (population, or exploration own samples 7,410,000 + i only).  Copy c ∈ {Q, S, T4} at
v₀ = s₀·(0.1, 0.2, 0.3, 0.4), s₀ = 0.2225397; branch-point start: the population point → own copy by Newton → held
W = max(4000, ⌈25/λ(s₀)⌉) steps (lr 1.0) → released: full-batch SGD, η on z, ρη on v, until CONT steps after the observed
crossing (enclosure lower end > 0) or the budget.  Then (all from the v path and the release state):
  own-path switch (W2-A's procedure, w4core.own_path_switch) t_sw, s_sw; R4 recursion t_traj (W2-A's, along the whole
  own v path, i.e. the NON-causal registered-style prediction); κ₀, λ at the own copy's adiabatic switch; r_cf = κχ;
  V7's window max χ_t (s_t ≥ 0.8 s_sw, t < t_sw; λ_t the reduced λ at z*(v_t));
  CAUSAL forecast at f (default 0.95): t_c = first t ≥ 1 with s_t ≥ f·s_ref (s_ref = the own copy's adiabatic switch);
  only rows < t_c are used: quad extrapolation of log s over the last max(50, ⌈0.05 t_c⌉) rows and of each share u_i
  (quad in t, clipped at 0, renormalised; the width-4 analogue of causal_forecast.extrapolate_v), horizon 1.6·s_ref;
  own-path switch and R4 along [visible, extrapolated] → t_fc, t_sw,fc.
Usage: python p3_runs.py COPY SAMPLE(pop|seed) eta rho [f] [CONT]"""
import json, math, sys
import numpy as np
import w4core as W
W.nice()
c, samp, eta, rho = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
F = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95
CONT = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
s0 = 0.2225397
v0 = s0 * np.array([0.1, 0.2, 0.3, 0.4])
b3 = json.load(open("p3_land_pop_box3.json"))["classes"]
b1 = json.load(open("p3_land_pop_0.1_0.2_0.3_0.4.json"))["classes"]
POP = {"Q": np.array(b3[49]["zc"]), "S": np.array(b3[39]["zc"]), "T4": np.array(b1[2]["zc"]),
       "Q11": np.array(b3[11]["zc"]), "D4": None}
x, y = W.population() if samp == "pop" else W.own_sample(int(samp))
T = W.Timer()
zo, ok = W.newton(POP[c], v0, x, y)
assert ok and W.btype(zo, v0) == W.btype(POP[c], v0), "own copy invalid"
lam0 = W.reduced_lam(W.blocks(zo, v0, x, y)[0], zo, v0)[0]
ad = W.continue_adiabatic(zo, v0, x, y, 20.0)
assert ad["status"] == "switch", ad["status"]
kp = W.kappa_at(ad["z"], ad["v"], ad["vprime"], x, y)
s_ref = ad["s_switch"]
Wh = max(4000, int(math.ceil(25 / lam0)))
z_rel, npos = W.hold(POP[c], v0, Wh, x, y, check_every=50)
zn, okn = W.newton(z_rel, v0, x, y)
on = okn and np.abs(zn - zo).max() <= 1e-6 and np.abs(z_rel - zn).max() <= 1e-3
budget = int(min(400_000, 100_000 * 2 ** -10 / rho))
Z, V, t_obs = W.train(z_rel, v0, eta, rho, budget, x, y, stop_after_cross=CONT)
s = np.abs(V).sum(1)
t_train = T()
br = W.OwnBranch(V, zo, x, y)
t_sw, s_sw = W.own_path_switch(br, len(V) - 1)
t_traj, st_traj = W.r4(br, z_rel, eta, len(V) - 1)
res = {"copy": c, "sample": samp, "eta": eta, "rho": rho, "on_copy": bool(on), "W": Wh, "hold_pos": npos,
       "lam_s0": lam0, "s_ref": s_ref, "kappa0": kp["kappa0"], "lam_switch": kp["lam"], "pred_lag_steps": kp["kappa0"] / (eta * kp["lam"]),
       "t_obs": t_obs, "s_obs": float(s[t_obs]) if t_obs else None, "t_sw": t_sw, "s_sw": s_sw, "t_traj": t_traj,
       "traj_status": st_traj, "lost_at": br.lost_at, "n_steps": len(V) - 1}
if t_sw:
    w = min(100, t_sw)
    sdot = (s[t_sw] - s[t_sw - w]) / w
    chi = (sdot / s_sw) / (eta * kp["lam"])
    res["chi_sw"] = chi; res["r_cf"] = kp["kappa0"] * chi
    win = [t for t in range(t_sw) if s[t] >= 0.8 * s_sw]
    if win and br.ensure(max(win)):
        res["chi_win_max"] = float(max(((s[t + 1] - s[t]) / s[t]) / (eta * br.lam[t]) for t in win))
if t_obs and t_sw:
    res["lag_obs"] = t_obs - t_sw; res["r_obs"] = float(s[t_obs] / s_sw - 1)
if t_traj and t_sw:
    res["lag_traj"] = t_traj - t_sw; res["r_traj"] = float(s[t_traj] / s_sw - 1)
# ---------------- causal forecast (rows < t_c only)
t_c = next((t for t in range(1, len(s)) if s[t] >= F * s_ref), None)
res["t_c"] = t_c
if t_c:
    Vv = V[:t_c]; sv = s[:t_c]
    nw = int(min(t_c, max(50, math.ceil(0.05 * t_c))))
    tau = np.arange(-nw + 1, 1, dtype=float); sc = max(1.0, abs(tau).max())
    yv = np.log(sv[-nw:])
    cy = np.polyfit(tau / sc, yv, 2)
    rate = np.polyval(np.polyder(cy), 0.0) / sc
    if rate > 0:
        need = (math.log(1.6 * s_ref) - yv[-1]) / rate
        nA = int(max(1, min(budget - t_c + 1, math.ceil(4 * max(need, 1)) + 10)))
        k = np.arange(1, nA + 1, dtype=float)
        yh = yv[-1] + np.polyval(cy, k / sc) - np.polyval(cy, 0.0)
        hit = np.nonzero(yh >= math.log(1.6 * s_ref))[0]
        if len(hit):
            yh = yh[:hit[0] + 1]; k = k[:hit[0] + 1]
        U = Vv[-nw:] / sv[-nw:, None]
        Uh = np.empty((len(k), 4))
        for i in range(4):
            cu = np.polyfit(tau / sc, U[:, i], 2)
            Uh[:, i] = U[-1, i] + np.polyval(cu, k / sc) - np.polyval(cu, 0.0)
        Uh = np.clip(Uh, 0, None); Uh /= Uh.sum(1, keepdims=True)
        Vext = np.vstack([Vv, np.exp(yh)[:, None] * Uh])
        bf = W.OwnBranch(Vext, zo, x, y)
        n_reuse = min(t_c, len(br.Z))          # z*(v_t), t < t_c: computed from visible rows only (identical)
        bf.Z, bf.H, bf.lam = br.Z[:n_reuse], br.H[:n_reuse], br.lam[:n_reuse]
        if n_reuse:
            bf._hzv = W.blocks(bf.Z[-1], Vext[n_reuse - 1], x, y)[1]
        t_swf, s_swf = W.own_path_switch(bf, len(Vext) - 1)
        t_fc, st_fc = W.r4(bf, z_rel, eta, len(Vext) - 1)
        sext = np.abs(Vext).sum(1)
        res.update(t_sw_fc=t_swf, t_fc=t_fc, fc_status=st_fc, n_window=nw)
        if t_fc and t_swf:
            res["lag_fc"] = t_fc - t_swf; res["r_fc"] = float(sext[t_fc] / s_swf - 1)
        if t_fc and t_obs:
            res["err_cross"] = t_fc - t_obs
        if res.get("lag_fc") is not None and res.get("lag_obs") is not None:
            res["err_lag"] = res["lag_fc"] - res["lag_obs"]
        res["cutoff_before_crossing"] = bool(t_obs is None or t_c < t_obs)
    else:
        res["fc_status"] = "not growing"
res["time_s"] = T(); res["train_s"] = t_train; res["rss_gb"] = round(W.rss_gb(), 3)
print(json.dumps(res, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), flush=True)
with open("p3_runs.jsonl", "a") as fh:
    fh.write(json.dumps(res, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")
