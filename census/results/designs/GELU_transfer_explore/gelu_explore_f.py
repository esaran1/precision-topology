"""EXPLORATORY (GELU-T, after approval, before registration; population only; numpy exploratory seeds; no candidate
seed).  (i) Fraction of random (w1, b1) ~ U(-1,1)^2 placed (dense state, sigma = +1, w2 = s0).  (ii) 150 random hidden
starts U(-1,1)^3 at w2 = +s0, held by GD lr 0.3 for 4,000 steps with G (dense) checked at the start and every step.
(iii) Population SGD release at eta = 0.03 from the branch point theta*_pop(s0): t_sw (first s >= s_pop)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.nice(15)
import torch
torch.set_num_threads(1)
from torch.nn import functional as Fn
from src.act_general import GAct, population, branch_point
from src.phase2b_ordering import state
K = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "act_general", "kappa_gelu_frozen.json")))
u = GAct("gelu").torch_u
x, y = population(); X = torch.tensor(np.asarray(x, float), dtype=torch.float64); Y = torch.tensor(np.asarray(y, float), dtype=torch.float64)
S = K["s_star"]; s0 = 0.5 * S
q = np.random.default_rng(12345).uniform(-1, 1, (4000, 2))
n = sum(state(torch.tensor([a, b, s0, 0.0], dtype=torch.float64), u, None, 1.0)["placement_ok"] for a, b in q)
print("(i) placed fraction of 4000 (w1, b1) draws at w2 = +s0:", n / 4000, flush=True)
rng = np.random.default_rng(424242)
res = {"n": 150, "placed_at_start": 0, "G_pos_after_a_hold_step": 0, "G_pos_start_or_hold": 0,
       "placed_start_still_placed_after_step1": 0, "unplaced_start_placed_in_hold": 0}
for i in range(150):
    qq = rng.uniform(-1, 1, 3)
    th = torch.tensor([qq[0], qq[1], s0, qq[2]], dtype=torch.float64, requires_grad=True)
    p0 = state(th.detach(), u, None, 1.0)["placement_ok"]
    npos, p1 = 0, None
    for step in range(1, 4001):
        g = torch.autograd.grad(Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y), th)[0]
        with torch.no_grad():
            th[[0, 1, 3]] -= 0.3 * g[[0, 1, 3]]
        pl = state(th.detach(), u, None, 1.0)["placement_ok"]
        p1 = pl if step == 1 else p1
        npos += int(pl)
    res["placed_at_start"] += int(p0); res["G_pos_after_a_hold_step"] += int(npos > 0)
    res["G_pos_start_or_hold"] += int(p0 or npos > 0)
    res["placed_start_still_placed_after_step1"] += int(p0 and p1)
    res["unplaced_start_placed_in_hold"] += int((not p0) and npos > 0)
print("(ii)", res, flush=True)
z0, _ = branch_point(np.array(K["z_star"]), s0, np.asarray(x, float), np.asarray(y, float), GAct("gelu"))
th = torch.tensor(np.r_[z0[0], z0[1], s0, z0[2]], dtype=torch.float64, requires_grad=True)
opt = torch.optim.SGD([th], lr=0.03)
for step in range(1, 200001):
    opt.zero_grad(set_to_none=True)
    Fn.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y).backward(); opt.step()
    if float(th.detach()[2]) >= S:
        print("(iii) population t_sw at eta 0.03:", step); break
