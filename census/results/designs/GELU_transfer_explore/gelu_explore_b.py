"""EXPLORATORY (GELU transfer design feasibility). POPULATION objective only (800 points); no training seed is used.
B: warm start at s0 = 0.5 s*_pop on the population branch point, release, free training (SGD at several eta, Adam at
   several lr with fresh state at release); per config: crossing lag r_obs = s_cross/s* - 1 (population switch on the
   tracked branch), chi at t_sw (Track A R5 closed form), kappa*chi, lag in steps, ratio.
C: random hidden init U(-1,1)^3 (numpy rng, exploratory seed) with w2 = +s0 fixed: relax, Newton, classify."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, "/Users/Evan/precision-topology/census")
os.nice(15)
import torch
torch.set_num_threads(1)
from torch.nn import functional as Fn
from src.act_general import GAct, population, branch_point, hessian_and_tangent, grad_gap
from src.lag_law import kappa
from src.phase2b_ordering import state

K = json.load(open("/Users/Evan/precision-topology/census/results/act_general/kappa_gelu_frozen.json"))
act = GAct("gelu"); u = act.torch_u
x, y = population(); x = np.asarray(x, float); y = np.asarray(y, float)
X, Y = torch.tensor(x), torch.tensor(y)
S = K["s_star"]; s0 = 0.5 * S
H_s, tan_s, dG_s = (np.array(K[k]) for k in ("H", "tangent", "gradG"))
z0, gr = branch_point(np.array(K["z_star"]), s0, x, y, act)
print("branch point at s0", s0, z0, "grad", gr)


def run(opt_name, lr, rho=1.0, budget=400000):
    th = torch.tensor(np.r_[z0[0], z0[1], s0, z0[2]], dtype=torch.float64, requires_grad=True)
    if opt_name == "sgd":
        opt = torch.optim.SGD([th], lr=lr)
    else:
        opt = torch.optim.Adam([th], lr=lr)
    hist = []; t_sw = None; rec = None
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y).backward()
        if rho != 1.0:
            th.grad[2] *= rho
        opt.step()
        s = abs(float(th.detach()[2])); hist.append(s)
        if t_sw is None and s >= S:
            t_sw = step
            sdot = (hist[-1] - hist[-101]) / 100
            if opt_name == "sgd":
                p = np.ones(3)
            else:
                st = opt.state[th]
                vh = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
                p = 1.0 / (np.sqrt(vh[[0, 1, 3]]) + 1e-8)
            kap, lamP, _, _ = kappa(H_s, tan_s, dG_s, p)
            chi = (sdot / S) / (lr * lamP)
            rec = {"t_sw": t_sw, "sdot": sdot, "kappa": kap, "chi": chi, "pred": kap * chi, "relax": lr * lamP}
        if t_sw is not None:
            st_ = state(th.detach(), u, None, 1.0)
            if st_["placement_ok"]:
                g_prev = rec.get("g_prev")
                rec.update({"step": step, "s_cross": s, "r_obs": s / S - 1, "lag_steps": step - t_sw,
                            "step_growth": hist[-1] / hist[-2] - 1})
                return rec
            rec["g_prev"] = st_["gap"]
    return rec


t0 = time.time()
for cfg in (("sgd", 0.3), ("sgd", 0.1), ("sgd", 0.03), ("adam", 0.01), ("adam", 0.003), ("adam", 0.001)):
    r = run(*cfg)
    print(cfg, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()},
          "obs/pred", r["r_obs"] / r["pred"], f"{time.time()-t0:.0f}s", flush=True)

# C: random hidden init at fixed s0
rng = np.random.default_rng(20260928)
n = 60
res = {"target": 0, "mirror": 0, "other": 0, "noconv": 0}
others = []
for i in range(n):
    q = rng.uniform(-1, 1, 3)
    th = torch.tensor([q[0], q[1], s0, q[2]], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([th], lr=0.3)
    for step in range(4000):
        opt.zero_grad(set_to_none=True)
        Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y).backward()
        th.grad[2] = 0.0
        opt.step()
    w = th.detach().numpy()
    z, g = branch_point(np.array([w[0], w[1], w[3]]), s0, x, y, act)
    Hh, _ = hessian_and_tangent(z, s0, x, y, act)
    pd = np.linalg.eigvalsh(Hh).min() > 0
    if g > 1e-8 or not pd:
        res["noconv"] += 1; others.append(("noconv", w.round(3).tolist(), z.round(3).tolist(), g)); continue
    if np.linalg.norm(z - z0) < 1e-3:
        res["target"] += 1
    elif np.linalg.norm(z - np.array([-z0[0], z0[1], z0[2]])) < 1e-3:
        res["mirror"] += 1
    else:
        res["other"] += 1; others.append(("other", w.round(3).tolist(), z.round(3).tolist()))
print("C random init at s0 (SGD 0.3, 4000 steps):", res)
for o in others[:10]:
    print(o)
