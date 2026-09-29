"""EXPLORATORY. POPULATION only; numpy exploratory seed. Adam warm start variants that keep v-hat memory:
hidden U(-1,1)^3, w2 held at +s0 by reassignment after each step (its moments see its true gradient),
Adam lr 0.01 for W steps, NO reset at release. Report: distance to branch point at release, relax at t_sw, obs/pred."""
import sys; sys.path.insert(0, "/Users/Evan/precision-topology/census")
import os; os.nice(15)
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "gelu_explore_c.py")).read().split("rng = np.random.default_rng(7)")[0])
from src.act_general import branch_point
zb, _ = branch_point(np.array(K["z_star"]), s0, x, y, act)
rng = np.random.default_rng(11)
for W in (300, 1000, 2000):
    for i in range(4):
        q = rng.uniform(-1, 1, 4); q[2] = s0
        th = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.01)
        for _ in range(W):
            opt.zero_grad(set_to_none=True); lossf(th).backward(); opt.step()
            with torch.no_grad(): th[2] = s0
        w = th.detach().numpy(); zz = np.array([abs(w[0]), w[1], w[3]])
        d = float(np.linalg.norm(zz - zb))
        r = release(th, opt, 0.01)
        print(" W", W, "dist to branch pt (mod mirror)", f"{d:.2e}", fmt(r), flush=True)
