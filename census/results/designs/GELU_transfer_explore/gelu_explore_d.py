"""EXPLORATORY. (i) Own-sample landscape on ALREADY-USED Track 3A seeds 850,000-850,011 (no training; Newton +
continuation only): target branch at s0 from the population branch point, its switch (and the mirror copy's), kappa_SGD
at the own switch, lambda_min at s0; compared with act_fold's switch_s where that seed was analysed.
(ii) Population SGD release (eta 0.03) at output-lr factors rho = 1, 3, 10, 30: obs/pred (closed form)."""
import csv, json, sys
import numpy as np
sys.path.insert(0, "/Users/Evan/precision-topology/census")
import os; os.nice(15)
import torch
torch.set_num_threads(1)
from torch.nn import functional as Fn
from src.act_general import GAct, population, branch_point, hessian_and_tangent, grad_gap
from src.act_fold import Problem, continue_branch
from src.lag_law import kappa
from src.phase2b_ordering import state

K = json.load(open("/Users/Evan/precision-topology/census/results/act_general/kappa_gelu_frozen.json"))
act = GAct("gelu"); u = act.torch_u
S = K["s_star"]; s0 = 0.5 * S
xp, yp = population(); xp = np.asarray(xp, float); yp = np.asarray(yp, float)
zpop0, _ = branch_point(np.array(K["z_star"]), s0, xp, yp, act)
fold = {int(r["seed"]): float(r["switch_s"]) for r in csv.DictReader(open(
    "/Users/Evan/precision-topology/census/results/act_fold/runs_gelu_pre95.csv")) if r["switch_s"] not in ("", "nan")}

print("(i) own-sample target branch (seeds already used by Track 3A; landscape only)")
for seed in range(850000, 850012):
    P = Problem("gelu", seed, 1.0)
    out = []
    for m in (+1, -1):
        z, g = branch_point(np.array([m * zpop0[0], zpop0[1], zpop0[2]]), s0, P.x, P.y, act)
        H, _ = hessian_and_tangent(z, s0, P.x, P.y, act)
        lam0 = np.linalg.eigvalsh(H).min()
        ev, path = continue_branch(P.F, P.J, np.r_[z, s0], +1, s_stop=1.6 * S, gapf=P.gap, h0=0.01, hmax=0.05)
        sw = ev["switch"]["s"] if ev["switch"] else float("nan")
        kap = float("nan")
        if ev["switch"]:
            zs = np.array(ev["switch"]["z"])
            Hs, tan = hessian_and_tangent(zs, sw, P.x, P.y, act)
            dG, _ = grad_gap(zs[0], zs[1], act) if zs[0] > 0 else (None, None)
            if dG is None:  # mirror: gap of sigma*u(w1 x + b1) with w1<0; finite differences directly
                from src.act_general import gap_mid
                h = 1e-6
                dG = np.array([(gap_mid(zs[0] + h, zs[1], act) - gap_mid(zs[0] - h, zs[1], act)) / (2 * h),
                               (gap_mid(zs[0], zs[1] + h, act) - gap_mid(zs[0], zs[1] - h, act)) / (2 * h), 0.0])
            kap = kappa(Hs, tan, dG, np.ones(3))[0]
        out.append((m, round(float(z[0]), 4), g < 1e-8, round(lam0, 4), ev["end"], ev["fold"] is not None,
                    round(sw, 5), round(sw / S, 4), round(kap, 4), round(ev["min_lam_stable"], 4)))
    print(seed, "act_fold switch_s:", fold.get(seed), out, flush=True)

print("(ii) population SGD release, eta = 0.03, output lr factor rho")
x, y = xp, yp
X, Y = torch.tensor(x), torch.tensor(y)
H_s, tan_s, dG_s = (np.array(K[k]) for k in ("H", "tangent", "gradG"))
kap, lam, _, _ = kappa(H_s, tan_s, dG_s, np.ones(3))
for rho in (1.0, 3.0, 10.0, 30.0):
    eta = 0.03
    th = torch.tensor(np.r_[zpop0[0], zpop0[1], s0, zpop0[2]], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([th], lr=eta)
    hist = [s0]; t_sw = None
    for step in range(1, 400000):
        opt.zero_grad(set_to_none=True)
        Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y).backward()
        th.grad[2] *= rho
        opt.step()
        s = float(th.detach()[2]); hist.append(s)
        if t_sw is None and s >= S:
            t_sw = step; sdot = (hist[-1] - hist[-101]) / 100; chi = (sdot / S) / (eta * lam)
        if t_sw is not None and state(th.detach(), u, None, 1.0)["placement_ok"]:
            r = s / S - 1
            print(f" rho={rho}: chi={chi:.4f} kappa*chi={kap*chi:.5f} r_obs={r:.5f} obs/pred={r/(kap*chi):.3f} "
                  f"lag steps={step-t_sw} t_sw={t_sw}", flush=True)
            break
