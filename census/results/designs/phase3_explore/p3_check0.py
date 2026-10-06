"""EXPLORATORY (population only).  Check of w4core: (i) analytic gradient/Hessian vs finite differences at K = 4;
(ii) K = 2 reproduces W2-A's frozen population numbers (landscape.json: T λ_min(s₀) 0.00971, switch 0.481586,
κ₀ 0.0486, a (−0.139, −0.127); D switch 5.079637, κ₀ 7.674); (iii) timings."""
import json, math, sys, time
import numpy as np
import w4core as W
W.nice()
print("memory gate:", sys.argv[1:] or "see log header", flush=True)
x, y = W.population()
rng = np.random.default_rng(1)
z, v = rng.uniform(-1, 1, 9), rng.uniform(0.1, 1, 4)
gz_, gv_ = W.grad(z, v, x, y); H = W.hess(z, v, x, y)
q = np.concatenate([z, v]); e = 1e-6
def Lq(q): return W.loss(q[:9], q[9:], x, y)
def Gq(q): return np.concatenate(W.grad(q[:9], q[9:], x, y))
gfd = np.array([(Lq(q + e * u) - Lq(q - e * u)) / (2 * e) for u in np.eye(13)])
Hfd = np.array([(Gq(q + e * u) - Gq(q - e * u)) / (2 * e) for u in np.eye(13)])
print("K=4 grad err", np.abs(np.concatenate([gz_, gv_]) - gfd).max(), "hess err", np.abs(H - Hfd).max())
# (ii) K = 2 vs W2-A frozen numbers
s0 = 0.5 * math.sqrt(0.43714448126110894 * 0.4531583637600818)
def w2(p): return np.array([p[0], p[2], p[1], p[3], p[4]])
T = (-1.707199, -1.450349, 1.670937, -1.692048, 0.296007)
D = (1.641692, -1.708088, 1.641692, -1.708088, 0.296361)
vT = s0 * np.array([0.1, 0.9]); vD = s0 * np.array([0.5, 0.5])
t0 = time.time()
zT, ok = W.newton(w2(T), vT, x, y)
print("T newton", ok, "lam(s0)", np.linalg.eigvalsh(W.blocks(zT, vT, x, y)[0]).min(), "type", W.btype(zT, vT))
r = W.continue_adiabatic(zT, vT, x, y, 3.0)
k = W.kappa_at(r["z"], r["v"], r["vprime"], x, y)
print("T switch", r["status"], r["s_switch"], "kappa0", k["kappa0"], "a", k["a"], "lam", k["lam"], round(time.time() - t0, 1), "s")
t0 = time.time()
zD, ok = W.newton(w2(D), vD, x, y)
print("D newton", ok, "type", W.btype(zD, vD), "groups", W.groups(zD))
r = W.continue_ray(zD, vD, x, y, 30.0)
k = W.kappa_at(r["z"], r["v"], r["vprime"], x, y)
print("D switch", r["status"], r["s_switch"], "kappa0", k["kappa0"], "a", k["a"], "lam", k["lam"], "split", k["lam_split"], round(time.time() - t0, 1), "s")
# (iii) timings at K = 4
z4 = np.concatenate([[1.6, 1.6, -1.3, 1.7], [-1.7, -1.7, -1.6, -1.5], [0.3]]); v4 = np.array([0.02, 0.04, 0.06, 0.08])
for name, fn in (("grad", lambda: W.grad(z4, v4, x, y)), ("hess", lambda: W.hess(z4, v4, x, y)),
                 ("gap_dense", lambda: W.gap_dense(z4, v4)), ("gap_enc", lambda: W.gap_enc(z4, v4))):
    t0 = time.time(); n = 0
    while time.time() - t0 < 0.5:
        fn(); n += 1
    print(name, round(1e6 * (time.time() - t0) / n), "us")
print("peak RSS GB", round(W.rss_gb(), 3))
