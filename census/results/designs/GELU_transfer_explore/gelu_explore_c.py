"""EXPLORATORY. POPULATION objective only; numpy exploratory seeds for inits (no training seed).
Adam regime check: relax = lr*lambda_min(P^1/2 H P^1/2) at t_sw (P = 1/(sqrt(vhat)+eps) at t_sw), chi, kappa*chi,
observed lag on the population (switch s* on the tracked branch; mirror handled via |w1|), lag in steps.
 D1: free Adam from U(-1,1)^4 (w2 sign forced +) at lr 0.01, 0.001.
 D2: warm start: hidden U(-1,1)^3, w2 = +s0 fixed, Adam warm-up W steps, NO state reset, then release.
 D3: warm start from the population branch point with an Adam warm-up of W steps starting from a random hidden init
     is D2; D3 = SGD warm-up (lr 0.3, 4000 steps) from random init, then fresh Adam at release."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/Evan/precision-topology/census")
import os; os.nice(15)
import torch
torch.set_num_threads(1)
from torch.nn import functional as Fn
from src.act_general import GAct, population, branch_point
from src.lag_law import kappa
from src.phase2b_ordering import state

K = json.load(open("/Users/Evan/precision-topology/census/results/act_general/kappa_gelu_frozen.json"))
act = GAct("gelu"); u = act.torch_u
x, y = population(); x = np.asarray(x, float); y = np.asarray(y, float)
X, Y = torch.tensor(x), torch.tensor(y)
S = K["s_star"]; s0 = 0.5 * S
H_s, tan_s, dG_s = (np.array(K[k]) for k in ("H", "tangent", "gradG"))


def lossf(th):
    return Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y)


def release(th, opt, lr, budget=200000):
    hist = [abs(float(th.detach()[2]))]; t_sw = None; rec = {}
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True); lossf(th).backward(); opt.step()
        s = abs(float(th.detach()[2])); hist.append(s)
        if t_sw is None and s >= S and len(hist) > 101:
            t_sw = step
            st = opt.state[th]
            vh = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
            p = 1.0 / (np.sqrt(vh[[0, 1, 3]]) + 1e-8)
            kap, lamP, _, _ = kappa(H_s, tan_s, dG_s, p)
            sdot = (hist[-1] - hist[-101]) / 100
            chi = (sdot / S) / (lr * lamP)
            rec = {"t_sw": t_sw, "relax": lr * lamP, "chi": chi, "kappa": kap, "pred": kap * chi}
        if t_sw is not None and state(th.detach(), u, None, 1.0)["placement_ok"]:
            rec.update({"r_obs": s / S - 1, "lag_steps": step - t_sw, "obs/pred": (s / S - 1) / (kap * chi)})
            return rec
        if s < 1e-3 and step > 5000:
            return {"stalled": True}
    return rec or {"never": True}


def fmt(r):
    return {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()}


rng = np.random.default_rng(7)
print("D1 free Adam from random init (population; w2 forced positive):")
for lr in (0.01, 0.001):
    for i in range(4):
        q = rng.uniform(-1, 1, 4); q[2] = abs(q[2])
        th = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        r = release(th, torch.optim.Adam([th], lr=lr), lr)
        print(" lr", lr, fmt(r), flush=True)

print("D2 warm start random hidden init, Adam warm-up W steps at w2 = s0, no reset:")
for W in (500, 1000, 4000):
    for i in range(3):
        q = rng.uniform(-1, 1, 4); q[2] = s0
        th = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.01)
        for _ in range(W):
            opt.zero_grad(set_to_none=True); lossf(th).backward(); th.grad[2] = 0.0; opt.step()
        th.data[2] = s0
        w = th.detach().numpy()
        r = release(th, opt, 0.01)
        print(" W", W, "w1 at release", round(float(w[0]), 3), fmt(r), flush=True)
