"""EXPLORATORY (GELU transfer design feasibility). Population landscape only; no training seeds.
A: continue the population branch from s* (kappa_gelu_frozen.json) down to s=0.3 and up to 2 s*.
B: along it: G, lambda_min(H), dL/ds, chi_SGD(rho=1) = |dL/ds|/(s lambda_min), kappa_SGD per s (info)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/Users/Evan/precision-topology/census")
os.nice(15)
import torch
torch.set_num_threads(1)
from src.act_general import GAct, population
from src.act_fold import Problem, continue_branch
from src.lag_law import kappa

R = "/Users/Evan/precision-topology/census/results/act_general/kappa_gelu_frozen.json"
K = json.load(open(R))


class Pop(Problem):
    def __init__(self):
        self.act = GAct("gelu"); self.sigma = 1.0
        x, y = population()
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.X, self.Y = torch.tensor(self.x), torch.tensor(self.y)


P = Pop()
X0 = np.r_[K["z_star"], K["s_star"]]
print("F at s*", np.abs(P.F(X0)).max(), "G", P.gap(X0))
out = {}
for d, stop in ((-1, 0.3), (+1, 2 * K["s_star"])):
    ev, path = continue_branch(P.F, P.J, X0, d, s_stop=stop, gapf=P.gap, h0=0.01, hmax=0.05)
    out[d] = (ev, path)
    print("dir", d, "end", ev["end"], "fold", ev["fold"], "switch", ev["switch"] and ev["switch"]["s"],
          "bp", ev["branch_point"], "minlam", ev["min_lam_stable"])


def dLds(z, s):
    q = torch.tensor(np.r_[z, s], dtype=torch.float64, requires_grad=True)
    L = P._L(q)
    return float(L), float(torch.autograd.grad(L, q)[0][3])


rows = []
for d in (-1, 1):
    for p in out[d][1]:
        rows.append(p)
rows.sort(key=lambda p: p["s"])
targets = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3228, 4.0, 5.0, 6.0, 6.6456, 7.5, 9.0, 11.0, 13.0]
for t in targets:
    p = min(rows, key=lambda r: abs(r["s"] - t))
    z, s = np.array(p["z"]), p["s"]
    L, g = dLds(z, s)
    H = P.J(np.r_[z, s])[:, :3]
    lam = np.linalg.eigvalsh(0.5 * (H + H.T))
    chi = abs(g) / (s * lam.min())
    print(f"s={s:7.4f} w1={z[0]:+.4f} b1={z[1]:+.4f} b2={z[2]:+.4f} G={p['G']:+.5f} L={L:.5f} dLds={g:+.5f} "
          f"lam={lam.min():.4f}/{lam.max():.4f} chiSGD(rho=1)={chi:.4f}")

H, tan, dG = (np.array(K[k]) for k in ("H", "tangent", "gradG"))
ks = kappa(H, tan, dG, np.ones(3))
print("kappa_SGD at s*:", ks)
L, g = dLds(np.array(K["z_star"]), K["s_star"])
chi = abs(g) / (K["s_star"] * ks[1])
print("s* dLds", g, "chi_SGD(rho=1)", chi, "kappa*chi", ks[0] * chi, "rho for kc=0.1:", 0.1 / (ks[0] * chi))
print("lag in steps kappa/(eta lam) at eta=0.3, 0.1, 0.03, 0.01:", [ks[0] / (e * ks[1]) for e in (0.3, 0.1, 0.03, 0.01)])
