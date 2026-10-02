"""EXPLORATORY (NOT a registration) for the Phase 2B decision ("the lever"): does a plain output-layer learning-rate
reduction change what training learns on the v3 simplicity-bias benchmark, and how does it compare with slowing only the
output SCALE (the Phase 2A rule)?

Benchmark (v3, unchanged): v2's 800 points, tanh width 4, BCE (mean) + (λ/2)(‖W‖² + ‖c‖²), λ = 1e−4, float64, full
batch; s = ‖v‖₁; q = 0.3914; s* = 3.5914; M's fold s_F = 4.7677 (src/sb_fold.py).
Initialisation: src/simplicity_bias_v3.init_net (PyTorch default init of Linear(2, 4), Linear(4, 1) after
torch.manual_seed(seed), float64, output weights capped at 0.5·s*), called inside torch.random.fork_rng so that the global
torch RNG state is restored afterwards. No idle-unit zeroing.
Seeds: NON-registered exploration seeds 2,953,000+ (verified unused; never to be registered).

Conditions (training in numpy; the Adam update reproduces torch.optim.Adam's formula, checked against torch by `check`):
  adam_r1            Adam lr 0.01 on every parameter (the v3 protocol)
  adam_r4/16/64      Adam, lr 0.01·r on the output weights v (all of v), lr 0.01 on W, c and the output bias b
  gd_rho1            full-batch GD, η = 1 on every parameter (reference for the GD family)
  gd_scale_r6/r9     GD, η on (W, c, b); v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/‖sign(v)‖ (Phase 2A rule), ρ = 2⁻⁶/2⁻⁹
  gd_plain_r6/r9     GD, η on (W, c, b); v ← v − ρη∇_vL (ρ on all of v) at the same ρ (mechanism contrast)
Stop: the first step with s ≥ 3·s* AND t ≥ the same seed's adam_r1 step count to 3·s* (the matched step budget), or the
per-condition step budget.

Per step: s and ρ₂ (v2/v3 definition, evaluated exactly on the product grid of the data's unique coordinate values;
checked equal to src.simplicity_bias_v2.rho2_batch by `check`). At the first step reaching each matched scale, at the
matched step budget, and at the ρ₂ events: loss, ρ₂, G₊ (src.simplicity_bias.gplus, the separating gap of the unit-ℓ₁
function), train accuracy, and accuracy on two shifted test sets built from the 800 points:
  shuffled  x₁ replaced by an independent draw from the x₁ multiset: the exact expectation over a uniform random
            permutation of x₁ across examples (every (i, j) pair, deterministic; no RNG)
  reversed  x₁ → −x₁ for every point. In the generator class 1 has x₁ = a and class 0 has x₁ = −a with the same levels
            a (src/simplicity_bias.make_data), so this swaps the classes' x₁ distributions exactly: the x₁–label
            correlation becomes −0.80 (from +0.80); x₂ (the slab) is unchanged.

    python p2b_explore.py check                # numpy Adam vs torch Adam; grid ρ₂ / G₊ vs the src implementations
    python p2b_explore.py run COND SEED [BUDGET]
    python p2b_explore.py summarise
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import simplicity_bias as sb          # noqa: E402
from src import simplicity_bias_v2 as v2       # noqa: E402
from src import simplicity_bias_v3 as v3       # noqa: E402

LAM = 1e-4
S_STAR = 3.5913755424683727
S_FOLD = 4.767689442106793
Q = 0.3914103370353161
CAP = 0.5 * S_STAR
SEEDS = tuple(range(2_953_000, 2_953_010))
RUNS = HERE / "runs"
SCALES = {"0.5s*": 0.5 * S_STAR, "s*": S_STAR, "1.25s*": 1.25 * S_STAR, "s_F": S_FOLD, "2s*": 2 * S_STAR,
          "3s*": 3 * S_STAR}
RSS_STOP = 3e9
CHUNK = 250

CONDS = {
    "adam_r1": {"kind": "adam", "r": 1.0, "budget": 200_000},
    "adam_r4": {"kind": "adam", "r": 1 / 4, "budget": 400_000},
    "adam_r16": {"kind": "adam", "r": 1 / 16, "budget": 800_000},
    "adam_r64": {"kind": "adam", "r": 1 / 64, "budget": 1_600_000},
    "gd_rho1": {"kind": "gd", "form": "plain", "rho": 1.0, "budget": 2_000_000},
    "gd_scale_r6": {"kind": "gd", "form": "scale", "rho": 2.0 ** -6, "budget": 3_000_000},
    "gd_plain_r6": {"kind": "gd", "form": "plain", "rho": 2.0 ** -6, "budget": 3_000_000},
    "gd_scale_r9": {"kind": "gd", "form": "scale", "rho": 2.0 ** -9, "budget": 6_000_000},
    "gd_plain_r9": {"kind": "gd", "form": "plain", "rho": 2.0 ** -9, "budget": 6_000_000},
    "gd_wn_r6": {"kind": "gd_wn", "rho": 2.0 ** -6, "budget": 3_000_000},
    "gd_wn_r9": {"kind": "gd_wn", "rho": 2.0 ** -9, "budget": 1_000_000},
    "adam_wn_r64": {"kind": "adam_wn", "r": 1 / 64, "budget": 1_600_000},
}
ETA = 1.0
ADAM_LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8

X, Y = v2.data()
N = len(Y)
U1, I1 = np.unique(X[:, 0], return_inverse=True)
U2, I2 = np.unique(X[:, 1], return_inverse=True)
W1 = np.bincount(I1).astype(float)                  # multiplicity of each x₁ value (20 each)
W2 = np.bincount(I2).astype(float)
NEG1 = np.searchsorted(U1, -U1)                     # index of −x₁ (the x₁ level set is exactly symmetric)
assert np.array_equal(U1[NEG1], -U1)
YB = Y.astype(bool)


# ------------------------------------------------------------------------------------------ model
def loss_grad(th):
    W = th[:8].reshape(4, 2); c = th[8:12]; v = th[12:16]; b = th[16]
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    L = float(np.mean(np.logaddexp(0.0, z) - Y * z) + 0.5 * LAM * ((W ** 2).sum() + (c ** 2).sum()))
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / N
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + LAM * W).ravel()
    g[8:12] = D.sum(0) + LAM * c
    g[12:16] = H.T @ r
    g[16] = r.sum()
    return L, g


def init_row(seed):
    """v3.init_net (torch.manual_seed(seed), default init, float64, cap 0.5·s*) inside fork_rng (global state restored)."""
    import torch
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        hid, out, s0 = v3.init_net(seed, CAP, torch)
    th = np.concatenate([hid.weight.detach().numpy().ravel(), hid.bias.detach().numpy(),
                         out.weight.detach().numpy().ravel(), out.bias.detach().numpy()]).astype(float)
    return th, float(s0)


def wn_grads(th, u, g_s):
    """ℓ₁ weight-normalised output v = g·u/‖u‖₁ (so s = ‖v‖₁ = g exactly): ∂L/∂g = ∇_vL·u/‖u‖₁ and
    ∂L/∂u = (g/‖u‖₁)(∇_vL − sign(u)(u·∇_vL)/‖u‖₁)."""
    _, gr = loss_grad(th)
    gv = gr[12:16]; n1 = np.abs(u).sum()
    return gr, float(gv @ u / n1), (g_s / n1) * (gv - np.sign(u) * (u @ gv) / n1)


def make_stepper(spec):
    if spec["kind"] in ("gd_wn", "adam_wn"):
        st = {"u": None, "g": None, "m": np.zeros(18), "v": np.zeros(18), "t": 0}
        lr = np.full(18, ADAM_LR); lr[17] *= spec.get("r", 1.0)

        def step(th):
            if st["u"] is None:
                st["u"] = th[12:16].copy(); st["g"] = float(np.abs(th[12:16]).sum())
            gr, dg, du = wn_grads(th, st["u"], st["g"])
            if spec["kind"] == "gd_wn":
                new = th - ETA * gr
                st["u"] = st["u"] - ETA * du; st["g"] = st["g"] - spec["rho"] * ETA * dg
            else:                                  # Adam on (W, c, b, u, g), lr 0.01·r on the gain only
                x = np.r_[th[:12], th[16], st["u"], st["g"]]; G = np.r_[gr[:12], gr[16], du, dg]
                st["t"] += 1; t = st["t"]
                st["m"] = st["m"] + (1 - B1) * (G - st["m"]); st["v"] = st["v"] * B2 + (1 - B2) * G * G
                x = x - (lr / (1 - B1 ** t)) * st["m"] / (np.sqrt(st["v"]) / math.sqrt(1 - B2 ** t) + EPS)
                new = th.copy(); new[:12] = x[:12]; new[16] = x[12]; st["u"] = x[13:17]; st["g"] = float(x[17])
            new[12:16] = st["g"] * st["u"] / np.abs(st["u"]).sum()
            return new
        return step
    if spec["kind"] == "adam":
        lr = np.full(17, ADAM_LR); lr[12:16] *= spec["r"]
        st = {"m": np.zeros(17), "v": np.zeros(17), "t": 0}

        def step(th):
            _, g = loss_grad(th)
            st["t"] += 1; t = st["t"]
            st["m"] = st["m"] + (1 - B1) * (g - st["m"])          # torch: exp_avg.lerp_(grad, 1 − β₁)
            st["v"] = st["v"] * B2 + (1 - B2) * g * g
            bc1 = 1 - B1 ** t; bc2 = 1 - B2 ** t
            denom = np.sqrt(st["v"]) / math.sqrt(bc2) + EPS
            return th - (lr / bc1) * st["m"] / denom
        return step
    rho, form = spec["rho"], spec["form"]

    def step(th):
        _, g = loss_grad(th)
        new = th - ETA * g
        gv = g[12:16]
        if form == "plain":
            new[12:16] = th[12:16] - rho * ETA * gv
        else:
            a = np.sign(th[12:16]); a = a / np.linalg.norm(a)
            new[12:16] = th[12:16] - ETA * (gv - (1 - rho) * a * (a @ gv))
        return new
    return step


# ------------------------------------------------------------------------------------------ measurements
def grid_z(P):
    """Network output without the bias on the 40 × 40 grid of unique (x₁, x₂) values: (m, 40, 40)."""
    W = P[:, :8].reshape(-1, 4, 2); c = P[:, 8:12]; v = P[:, 12:16]
    A = W[:, None, None, :, 0] * U1[None, :, None, None] + W[:, None, None, :, 1] * U2[None, None, :, None]
    return np.einsum("mabk,mk->mab", np.tanh(A + c[:, None, None, :]), v)


def rho2_rows(P):
    """ρ₂ for parameter rows (m, 17): V_j = (1/N²) Σ_i Σ_t mult(t) |φ(x_i, x_j := t) − φ(x_i)|, ρ₂ = V₂/(V₁ + V₂)."""
    Z = grid_z(P)
    base = Z[:, I1, I2]                                            # (m, N)
    V1 = np.einsum("a,man->m", W1, np.abs(Z[:, :, I2] - base[:, None, :])) / N ** 2
    V2 = np.einsum("b,mbn->m", W2, np.abs(np.transpose(Z, (0, 2, 1))[:, :, I1] - base[:, None, :])) / N ** 2
    return V2 / (V1 + V2)


def metrics(th):
    P = th[None]
    Z = grid_z(P)[0]
    b = th[16]
    base = Z[I1, I2]
    sv = np.abs(th[12:16]).sum()
    corr = lambda z: (z + b > 0) == YB                                                    # noqa: E731
    shuf = (W1[:, None] * corr(Z[:, I2])).sum() / (W1.sum() * N)                           # (40, N): x₁ := t
    rev = corr(Z[NEG1[I1], I2]).mean()
    L, _ = loss_grad(th)
    return {"s": float(sv), "loss": L, "rho2": float(rho2_rows(P)[0]),
            "gplus": float(0.5 * (base[YB].min() - base[~YB].max()) / sv),
            "acc_train": float(corr(base).mean()), "acc_shuffled": float(shuf), "acc_reversed": float(rev)}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS


# ------------------------------------------------------------------------------------------ one run
def part(cond, seed):
    return RUNS / f"{cond}_{seed}.json"


def run(cond, seed, budget=None):
    spec = CONDS[cond]
    budget = spec["budget"] if budget is None else int(budget)
    t_match = None
    if cond != "adam_r1":
        std = json.loads(part("adam_r1", seed).read_text())
        t_match = std["t_scale"]["3s*"]
    th, s_raw = init_row(seed)
    step = make_stepper(spec)
    t0 = time.time()
    S, R = [], []
    buf, buf_t = [], []
    t_scale, keep = {}, {}
    ev = {"below_seen": False, "first_literal": None, "first_up": None, "last_up": None, "prev": None, "n_up": 0}
    status = "budget"

    def flush():
        if not buf:
            return
        P = np.array(buf)
        r = rho2_rows(P)
        for j, (t, val) in enumerate(zip(buf_t, r)):
            if val >= Q:
                if ev["first_literal"] is None:
                    ev["first_literal"] = t; keep["first_literal"] = P[j].copy()
                if ev["prev"] is not None and ev["prev"] < Q:
                    ev["n_up"] += 1; ev["last_up"] = t; keep["last_up"] = P[j].copy()
                    if ev["first_up"] is None:
                        ev["first_up"] = t; keep["first_up"] = P[j].copy()
            ev["prev"] = val
        R.extend(r.tolist())
        buf.clear(); buf_t.clear()
        if rss() > RSS_STOP:
            raise SystemExit(f"RSS {rss() / 1e9:.2f} GB > 3 GB: stopped")

    t = 0; n_flip = 0; sg_prev = np.sign(th[12:16])
    while True:
        s = float(np.abs(th[12:16]).sum())
        sg = np.sign(th[12:16]); n_flip += int((sg != sg_prev).sum()); sg_prev = sg
        if not np.isfinite(th).all():
            status = "nonfinite"; break
        S.append(s); buf.append(th.copy()); buf_t.append(t)
        for nm, val in SCALES.items():
            if nm not in t_scale and s >= val:
                t_scale[nm] = t; keep["scale_" + nm] = th.copy()
        if t_match is not None and t == t_match:
            keep["match_step"] = th.copy()
        if s >= SCALES["3s*"] and (t_match is None or t >= t_match):
            status = "reached_3s*"; break
        if t >= budget:
            break
        if len(buf) >= CHUNK:
            flush()
        th = step(th); t += 1
    flush()
    S = np.array(S); R = np.array(R)
    res = {"label": "EXPLORATORY (non-registered seeds 2,953,000+; never to be registered)", "cond": cond,
           "seed": seed, "spec": spec, "s_init_raw": s_raw, "s_init": float(S[0]), "rho2_init": float(R[0]),
           "steps": int(t), "status": status, "s_end": float(S[-1]), "rho2_end": float(R[-1]),
           "t_match": t_match, "t_scale": t_scale, "rho2_min": float(R.min()), "rho2_max": float(R.max()),
           "n_upward_passages": ev["n_up"], "n_output_sign_changes": n_flip,
           "events": {k: (None if ev[k] is None else {"t": ev[k], "s": float(S[ev[k]])})
                      for k in ("first_literal", "first_up", "last_up")},
           "rho2_ge_q_at_end": bool(R[-1] >= Q),
           "seconds": time.time() - t0, "max_rss_gb": rss() / 1e9}
    res["at"] = {k: metrics(v) for k, v in keep.items()}
    if t_match is not None and "match_step" not in keep:
        res["at"]["match_step"] = None                # run stopped (nonfinite/budget) before the matched step
    idx = np.unique(np.r_[0, np.geomspace(1, max(len(S) - 1, 1), 250).astype(int), len(S) - 1])
    res["traj"] = {"t": idx.tolist(), "s": S[idx].tolist(), "rho2": R[idx].tolist()}
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = part(cond, seed).with_suffix(".tmp"); tmp.write_text(json.dumps(res, default=float)); tmp.replace(part(cond, seed))
    print(json.dumps({k: res[k] for k in ("cond", "seed", "steps", "status", "s_end", "rho2_end", "events", "t_scale",
                                          "seconds", "max_rss_gb")}, default=float), flush=True)


# ------------------------------------------------------------------------------------------ checks
def check(seed=SEEDS[-1] + 90):
    """Exploration seed 2,953,099 (reserved for this check): torch Adam (the v3 loop, λ on W, c) vs this numpy Adam for
    the standard condition; grid ρ₂ and G₊ vs src.simplicity_bias_v2.rho2_batch and src.simplicity_bias.gplus."""
    import torch
    torch.set_num_threads(1)
    th0, _ = init_row(seed)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)
    with torch.random.fork_rng(devices=[]):
        hid, out, _ = v3.init_net(seed, CAP, torch)
    params = [hid.weight, hid.bias, out.weight, out.bias]
    opt = torch.optim.Adam(params, lr=ADAM_LR, betas=(B1, B2), eps=EPS)
    step = make_stepper(CONDS["adam_r1"])
    th = th0.copy(); rows_t, rows_n = [], []
    for t in range(1500):
        rows_t.append(np.concatenate([p.detach().numpy().ravel() for p in params])); rows_n.append(th.copy())
        z = out(torch.tanh(hid(Xt))).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * LAM * (
            (hid.weight ** 2).sum() + (hid.bias ** 2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
        th = step(th)
    A = np.array(rows_t); B = np.array(rows_n)
    rel = np.abs(A - B).max(axis=1) / np.abs(A).max(axis=1)
    rt = v2.rho2_batch(A[::25], X, torch); rg = rho2_rows(A[::25])
    from src import sb_fold as F
    full = F.run_to_full(A[::100])
    gs = sb.gplus(full, X, Y); gm = np.array([metrics(r)["gplus"] for r in A[::100]])
    res = {"label": "EXPLORATORY check (seed 2,953,099)", "adam_numpy_vs_torch_max_rel_param_diff": {
               "t<=100": float(rel[:101].max()), "t<=500": float(rel[:501].max()), "t<=1499": float(rel.max())},
           "s_torch_vs_numpy_at_1499": [float(np.abs(A[-1, 12:16]).sum()), float(np.abs(B[-1, 12:16]).sum())],
           "rho2_grid_vs_src_max_abs_diff": float(np.abs(rt - rg).max()), "rho2_rows_compared": len(rt),
           "gplus_vs_src_max_abs_diff": float(np.abs(gs - gm).max()), "gplus_rows_compared": len(gs)}
    (HERE / "p2b_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------ summary
POINTS = list(SCALES) + ["match_step"]
QTY = ("rho2", "gplus", "acc_train", "acc_shuffled", "acc_reversed")


def _q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def onset_class(s):
    if s is None:
        return "none"
    if s < S_STAR:
        return "below s*"
    if s <= 1.25 * S_STAR:
        return "at s* [1, 1.25]s*"
    if s < S_FOLD:
        return "between 1.25s* and s_F"
    if s <= 1.25 * S_FOLD:
        return "at s_F [1, 1.25]s_F"
    return "above 1.25s_F"


def summarise():
    runs = {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text()); runs.setdefault(r["cond"], {})[r["seed"]] = r
    seeds = sorted(set.intersection(*[set(v) for v in runs.values()])) if runs else []
    out = {"label": "EXPLORATORY (non-registered seeds 2,953,000+; never to be registered)",
           "seeds_complete_in_every_condition": seeds, "conditions": {}}
    std = runs.get("adam_r1", {})
    for cond in CONDS:
        if cond not in runs:
            continue
        R = [runs[cond][s] for s in seeds]
        c = {"n_runs": len(R), "status": {k: sum(r["status"] == k for r in R) for k in {r["status"] for r in R}},
             "steps_total": _q([r["steps"] for r in R]), "seconds": _q([r["seconds"] for r in R]),
             "rho2_init": _q([r["rho2_init"] for r in R]), "s_init": _q([r["s_init"] for r in R]),
             "n_output_sign_changes": _q([r["n_output_sign_changes"] for r in R]), "points": {}, "onset": {}}
        for pt in POINTS:
            # adam_r1's matched step IS its step to 3 s* (the budget is defined from it)
            key = "scale_3s*" if (pt == "match_step" and cond == "adam_r1") else (pt if pt == "match_step" else "scale_" + pt)
            rows = [(r["at"].get(key), r, s) for r, s in zip(R, seeds)]
            reached = [(m, r, s) for m, r, s in rows if m is not None]
            d = {"n_reached": len(reached)}
            if pt != "match_step":
                d["steps"] = _q([r["t_scale"][pt] for _, r, _ in reached])
                d["steps_over_standard"] = _q([r["t_scale"][pt] / std[s]["t_scale"][pt] for _, r, s in reached
                                               if s in std and std[s]["t_scale"].get(pt)])
            else:
                d["s"] = _q([m["s"] for m, _, _ in reached])
            for k in QTY:
                d[k] = _q([m[k] for m, _, _ in reached])
            d["frac_rho2_ge_q"] = (sum(m["rho2"] >= Q for m, _, _ in reached) / len(reached)) if reached else None
            if cond != "adam_r1":
                key = "scale_3s*" if pt == "match_step" else "scale_" + pt
                for k in ("rho2", "acc_shuffled", "acc_reversed"):
                    diff = [m[k] - std[s]["at"][key][k] for m, _, s in reached if std.get(s, {}).get("at", {}).get(key)]
                    d[f"paired_diff_vs_adam_r1_{k}"] = _q(diff)
                    d[f"n_above_adam_r1_{k}"] = int(sum(x > 0 for x in diff)); d[f"n_below_adam_r1_{k}"] = int(sum(x < 0 for x in diff))
            c["points"][pt] = d
        for ev in ("first_up", "first_literal", "last_up"):
            ss = [None if r["events"][ev] is None else r["events"][ev]["s"] for r in R]
            cls = [onset_class(x) for x in ss]
            c["onset"][ev] = {"s": _q(ss), "s_over_s_star": _q([x / S_STAR for x in ss if x is not None]),
                              "class_counts": {k: cls.count(k) for k in dict.fromkeys(cls)},
                              "n_rho2_init_ge_q": int(sum(r["rho2_init"] >= Q for r in R))}
        c["rho2_ge_q_at_end"] = int(sum(r["rho2_ge_q_at_end"] for r in R))
        c["s_end"] = _q([r["s_end"] for r in R])
        out["conditions"][cond] = c
    (HERE / "p2b_summary.json").write_text(json.dumps(out, indent=1))
    print(tables(out))


def _f(d, k="median", fmt="{:.3f}"):
    if d is None:
        return "—"
    return fmt.format(d[k]) if k != "iqr" else (fmt + "–" + fmt).format(d["q25"], d["q75"])


def tables(out):
    L = []
    for pt in POINTS:
        L.append(f"\n### {pt}\n")
        L.append("| condition | n reached | steps (median [IQR]) | ρ₂ | frac ρ₂ ≥ q | acc shuffled | acc reversed | G₊ | acc train |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for cond, c in out["conditions"].items():
            d = c["points"][pt]
            st = d.get("steps") if pt != "match_step" else d.get("s")
            stt = "—" if st is None else (f"{st['median']:.0f} [{st['q25']:.0f}–{st['q75']:.0f}]" if pt != "match_step"
                                          else f"s {st['median']:.2f} [{st['q25']:.2f}–{st['q75']:.2f}]")
            cell = lambda k: "—" if d[k] is None else f"{d[k]['median']:.3f} [{d[k]['q25']:.3f}–{d[k]['q75']:.3f}]"  # noqa: E731
            fr = "—" if d["frac_rho2_ge_q"] is None else f"{d['frac_rho2_ge_q']:.2f}"
            L.append(f"| {cond} | {d['n_reached']}/{c['n_runs']} | {stt} | {cell('rho2')} | {fr} | {cell('acc_shuffled')} | "
                     f"{cell('acc_reversed')} | {cell('gplus')} | {cell('acc_train')} |")
    L.append("\n### onset of slab use (first upward passage of q; first literal ρ₂ ≥ q; last upward passage)\n")
    L.append("| condition | first up: s/s* median [IQR] | classes (first up) | first literal: classes | last up: s/s* median | ρ₂ ≥ q at end | ρ₂(0) ≥ q |")
    L.append("|---|---|---|---|---|---|---|")
    for cond, c in out["conditions"].items():
        o = c["onset"]
        q1 = o["first_up"]["s_over_s_star"]
        fu = "—" if q1 is None else "{:.2f} [{:.2f}–{:.2f}]".format(q1["median"], q1["q25"], q1["q75"])
        L.append(f"| {cond} | {fu} "
                 f"| {o['first_up']['class_counts']} | {o['first_literal']['class_counts']} | "
                 f"{'—' if o['last_up']['s_over_s_star'] is None else round(o['last_up']['s_over_s_star']['median'], 2)} | "
                 f"{c['rho2_ge_q_at_end']}/{c['n_runs']} | {o['first_up']['n_rho2_init_ge_q']}/{c['n_runs']} |")
    return "\n".join(L)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "summarise":
        summarise()
