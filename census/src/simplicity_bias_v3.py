"""Track T v3: one registered test at the attained v2 switch (registration: results/simplicity_bias_v3_registration.md).

Prediction (frozen before any training): with the same data, λ = 1e−4 and Adam, training acquires the slab feature
just above the fixed-scale switch s_switch = 3.5914 (the committed v2 s_q; the path jumps ρ₂ 0.182 → 0.448 across it).
Event: the first upward passage of q = 0.3914 by the run's slab share ρ₂ (the path's measure), evaluated at every step.
No κ.

    python -m src.simplicity_bias_v3 freeze         # frozen inputs + SHA-256 (before any training)
    python -m src.simplicity_bias_v3 train W NW     # seeds W::NW (resumable)
    python -m src.simplicity_bias_v3 score          # every-step ρ₂, crossings, χ, criteria
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

from . import simplicity_bias as sb
from . import simplicity_bias_v2 as v2

OUT = sb.RESULTS / "simplicity_bias_v3"
RUNS = OUT / "runs"
V2_FROZEN = sb.RESULTS / "simplicity_bias_v2" / "frozen.json"
SEEDS = tuple(range(2_720_000, 2_720_040))
LR = 0.01
BUDGET = 40_000
STOP_FACTOR = 3.0
INIT_CAP_FACTOR = 0.5
CHI_WINDOW = 100
C1_MIN = 0.90
C2_LO, C2_HI = 1.00, 1.25
MIN_CROSS = 30
CHI_MAX = 0.06


def frozen_inputs():
    fr = json.loads(V2_FROZEN.read_text())
    return {"s_switch": fr["s_q"], "q": fr["q"], "lambda": fr["lambda"], "v2_frozen_sha256": v2.sha256_file(V2_FROZEN),
            "H_min_eig": fr["H_min_eig"], "seeds": [SEEDS[0], SEEDS[-1]], "n_seeds": len(SEEDS), "lr": LR,
            "betas": [0.9, 0.999], "eps": 1e-8, "budget": BUDGET, "stop_factor": STOP_FACTOR,
            "init_cap": INIT_CAP_FACTOR * fr["s_q"], "chi_window": CHI_WINDOW,
            "criteria": {"C1_min_fraction_at_or_above_switch": C1_MIN, "C2_median_ratio_range": [C2_LO, C2_HI],
                         "validity_min_crossings": MIN_CROSS, "validity_max_median_chi": CHI_MAX}}


def run_freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "frozen_inputs.json"
    f.write_text(json.dumps(frozen_inputs(), indent=1))
    (OUT / "frozen_inputs.sha256").write_text(v2.sha256_file(f) + "  frozen_inputs.json\n")
    print(f.read_text())


# ------------------------------------------------------------------------------------------ training
def init_net(seed, cap, torch):
    """PyTorch default init of Linear(2, 4) and Linear(4, 1) after torch.manual_seed(seed), float64; then, if the
    output scale ‖w₂‖₁ exceeds cap = 0.5·s_switch, the output weights are multiplied by cap/‖w₂‖₁ (same rule for every
    seed; the output bias is unchanged)."""
    torch.manual_seed(seed)
    hid = torch.nn.Linear(2, 4).double(); out = torch.nn.Linear(4, 1).double()
    with torch.no_grad():
        s0 = float(out.weight.abs().sum())
        if s0 > cap:
            out.weight.mul_(cap / s0)
    return hid, out, s0


def train_run(seed, fz):
    torch = v2._torch()
    X, y = v2.data()
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(y)
    hid, out, s0 = init_net(seed, fz["init_cap"], torch)
    params = [hid.weight, hid.bias, out.weight, out.bias]
    opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    lam = fz["lambda"]; stop = STOP_FACTOR * fz["s_switch"]
    S, PR, VH = [], [], []
    for t in range(BUDGET + 1):
        with torch.no_grad():
            s = float(out.weight.abs().sum())
            S.append(s)
            PR.append(np.concatenate([p.detach().numpy().ravel() for p in params]))
            VH.append(np.full(13, np.nan) if t == 0 else np.concatenate(
                [opt.state[p]["exp_avg_sq"].numpy().ravel() for p in (hid.weight, hid.bias, out.bias)]) / (1 - 0.999 ** t))
        if s >= stop or t == BUDGET:
            break
        z = out(torch.tanh(hid(Xt))).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * lam * (
            (hid.weight ** 2).sum() + (hid.bias ** 2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = RUNS / f"run_{seed}.tmp.npz"
    np.savez_compressed(tmp, s=np.array(S), params=np.array(PR), vhat=np.array(VH), s_init_raw=s0)
    tmp.replace(RUNS / f"run_{seed}.npz")
    return len(S) - 1, S[-1], s0


def run_train(w, nw):
    sb._init_worker()
    fz = json.loads((OUT / "frozen_inputs.json").read_text())
    for seed in SEEDS[w::nw]:
        if (RUNS / f"run_{seed}.npz").exists():
            continue
        t0 = time.time()
        steps, s_end, s0 = train_run(seed, fz)
        print(json.dumps({"seed": seed, "steps": steps, "s_end": s_end, "s_init_raw": s0,
                          "seconds": time.time() - t0}), flush=True)


# ------------------------------------------------------------------------------------------ χ and the decision rules
def chi_at(s_traj, vhat, t, H, lr=LR, window=CHI_WINDOW):
    """χ at step t: (d log s/dt over the last min(window, t) steps) / (lr·λ_min(P^{1/2}HP^{1/2})), H the committed
    weight-decayed Hessian at the switch (v2 frozen.json, coordinates W, c, b), P = the run's block-median Adam
    1/(√v̂ + ε) at step t (v2.block_p)."""
    if t is None or t < 1:
        return None
    w = min(window, t)
    rate = (math.log(s_traj[t]) - math.log(s_traj[t - w])) / w
    p = v2.block_p(vhat[t])
    ph = np.sqrt(p)
    lmin = float(np.linalg.eigvalsh((ph[:, None] * np.asarray(H)) * ph[None, :]).min())
    return rate / (lr * lmin)


def decide(s_cross, chis, n_runs, s_switch):
    """C1: fraction of crossing runs with s_cross ≥ s_switch ≥ 0.90.  C2: median(s_cross/s_switch) ∈ [1.00, 1.25].
    Validity: ≥ 30 crossings of n_runs AND median χ at crossing ≤ 0.06; otherwise both criteria UNRESOLVED."""
    s_cross = np.asarray(s_cross, float); chis = np.asarray(chis, float)
    n = len(s_cross)
    res = {"n_runs": n_runs, "n_cross": n}
    if n == 0:
        res.update({"valid": False, "unresolved_by": ["crossings"], "C1": "UNRESOLVED", "C2": "UNRESOLVED"})
        return res
    frac = float((s_cross >= s_switch).mean()); med = float(np.median(s_cross / s_switch)); mchi = float(np.median(chis))
    res.update({"fraction_at_or_above_switch": frac, "median_ratio": med, "median_chi": mchi})
    why = []
    if n < MIN_CROSS:
        why.append("crossings")
    if not mchi <= CHI_MAX:
        why.append("chi")
    if why:
        res.update({"valid": False, "unresolved_by": why, "C1": "UNRESOLVED", "C2": "UNRESOLVED",
                    "C1_would_be": "PASS" if frac >= C1_MIN else "FAIL",
                    "C2_would_be": "PASS" if C2_LO <= med <= C2_HI else "FAIL"})
    else:
        res.update({"valid": True, "unresolved_by": [], "C1": "PASS" if frac >= C1_MIN else "FAIL",
                    "C2": "PASS" if C2_LO <= med <= C2_HI else "FAIL"})
    return res


def run_score():
    import pandas as pd
    torch = v2._torch()
    fz = json.loads((OUT / "frozen_inputs.json").read_text())
    H = np.array(json.loads(V2_FROZEN.read_text())["H"])
    X, _ = v2.data()
    rows = []
    for seed in SEEDS:
        d = np.load(RUNS / f"run_{seed}.npz")
        rho = v2.rho2_batch(d["params"], X, torch)                  # every step
        tc = v2.crossing_step(rho, fz["q"])
        lit = np.flatnonzero(rho >= fz["q"])
        rows.append({"seed": seed, "steps": len(d["s"]) - 1, "s_init_raw": float(d["s_init_raw"]), "s_init": float(d["s"][0]),
                     "s_end": float(d["s"][-1]), "rho2_init": float(rho[0]), "rho2_min": float(rho.min()),
                     "rho2_end": float(rho[-1]), "crossing_step": tc,
                     "s_cross": None if tc is None else float(d["s"][tc]),
                     "chi": None if tc is None else chi_at(d["s"], d["vhat"], tc, H),
                     "first_step_rho2_ge_q_literal": int(lit[0]) if len(lit) else None,
                     "s_at_literal_first": float(d["s"][lit[0]]) if len(lit) else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs_summary.csv", index=False)
    c = df[df.crossing_step.notna()]
    res = decide(c.s_cross.values, c.chi.values, len(df), fz["s_switch"])
    res["s_cross_quartiles"] = [float(v) for v in np.percentile(c.s_cross, [25, 50, 75])] if len(c) else None
    res["chi_quartiles"] = [float(v) for v in np.percentile(c.chi, [25, 50, 75])] if len(c) else None
    res["n_rho2_init_ge_q"] = int((df.rho2_init >= fz["q"]).sum())
    (OUT / "score.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "freeze":
        run_freeze()
    elif cmd == "train":
        run_train(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "score":
        run_score()
