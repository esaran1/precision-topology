"""Track A: a REGISTERED test of the lag law at an unseen activation value (a = 1.65).

Registration: results/track_a_registration.md.  Everything below is frozen in one hashed commit before any registered
run.  Rules use only (i) the population landscape at a = 1.65, (ii) each fresh seed's own training sample (before
training), (iii) the run's state up to the rule point (the first step at which |w₂| reaches 0.5 of the seed's frozen
branch switch), and (iv) the run's output-scale trajectory s_t = |w₂(t)|.

Choice of a (author's decision: a value with NO existing crossing data).  1.40 (and 1.35) have width-1 crossing data;
1.55 and 1.70 appear in onset_curves / sgd_onset_curves / onset_bootstrap_outcomes (placement rates by budget = crossing
information).  a = 1.65 appears only in corner_tracking.csv (a landscape table, no training) and in search_anneal.csv's
`parameter` column (a depth-3 annealing search, not width-1 training).  No src seed/a list trains width 1 at 1.65.

Landscape at 1.65 (no certified glob bracket exists at 1.65): the switch s*_pop is the root of G(θ*(s)) on the branch
reached by Newton at a = 1.65 from the 1.60 certified switch point (lag_law kappa.csv z_star), continued in s on the
800-point population objective; VALIDATED by the existing conditional search (own_threshold.global_min, the search used
for every own-sample threshold): global minimiser unplaced at 0.995·s*_pop and placed at 1.005·s*_pop, and the global
minimiser at 1.005·s*_pop on the same branch (up to the mirror x → −x).

Winding rule (landscape alone): the copy of the landscape branch with canonical b₁ ∈ (−π, π] (orientation w₂ > 0).  This
rule was DEFINED USING EXISTING DATA AT OTHER a: all 1,750 existing crossings at a = 1.30/1.45/1.50/1.60
(lag_law/crossing_states.csv) have canonical b₁ in (−2.47, −2.22) ⊂ (−π, π], i.e. the rule predicts the observed
windings k = −1 (1.30) and 0 (1.45-1.60) of 1A.  κ_k(1.65) is computed on that copy.

Per run (Adam: torch Adam lr 0.01; SGD: plain torch SGD lr 0.3; data make_data(200, seed); init as ts_test / sgd_own;
budget 32,000 steps; every-step detection with phase2b_ordering.state, the dense 4,001-point windows):
  - frozen before training: s*_frozen = the own-sample switch of the landscape branch (Newton from the landscape switch
    point onto the own sample, continued), and the global own-sample threshold (own_threshold rule, bracket 0.01);
  - rule point t_R: the step at which s_t last passes 0.5·s*_frozen upward before first reaching s*_frozen;
  - branch rule: the branch occupied at t_R = Newton (fixed s = s_{t_R}) from the run's state at t_R, continued on the
    own sample (grid [0.3, 1.7]·s*_frozen, spacing 0.002·s*_frozen); s*_run = its switch nearest s*_frozen;
  - preconditioner rule: P_R = 1/(√v̂ + ε) at t_R (Adam); P = I (SGD);
  - trajectory-integrated prediction: Track 1's linear-response recursion from t_R with δ₀ = θ_{t_R} − θ*(s_{t_R}),
    m₀ = the run's Adam first moment at t_R, P FROZEN at P_R, H(s_t) and the exact θ*(s_t) path of the occupied branch
    along the run's own s_t, Adam's bias correction 1 − β₁^t; crossing at the first step with G(θ*(s_t) + δ_t) > 0;
    s_traj = s_t there; no prediction if the iterated linear map has spectral radius > 1 or sup|δ| > 1 (Track 1 rule);
  - closed form: r_cf = κ_k·χ with κ_k from the landscape (H, θ*′ of the principal copy, ∇G at s*_pop; P shape P_R),
    χ = (ṡ/s*_run)/(η·λ_min(P_R^{1/2} H_pop P_R^{1/2})), ṡ = (s_{t_sw} − s_{t_sw−100})/100 at t_sw = the first step with
    s_t ≥ s*_run (the output-scale trajectory only);
  - lags against s*_run: r_traj = s_traj/s*_run − 1, r_obs = s_obs/s*_run − 1.

How the crossing is kept out of the predictions: `train` runs the full 32,000-step budget WITHOUT any placement or gap
evaluation, saves the parameter path (paths/*.npz, SHA-256 listed in the predictions file) and computes the predictions
from the rule-point state and the s path.  Predictions (with the path hashes) are committed and hashed.  Only then does
`observe` evaluate phase2b_ordering.state at every saved step (after asserting the committed hash and each path's hash):
the first placed step is the observed crossing (the path is deterministic, so this equals every-step detection during
training).

    python -m src.track_a landscape        # s*_pop, κ_k, validation      -> results/track_a/landscape.json
    python -m src.track_a freeze           # per-seed frozen inputs        -> results/track_a/frozen_seeds.csv
    python -m src.track_a train            # registered runs, predictions  -> results/track_a/predictions.csv
    python -m src.track_a finalize         # predictions hash
    python -m src.track_a observe          # after the commit: crossings, scores
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import linear_response as LR

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "track_a"
PATHS = OUT / "paths"
A = 1.65
SEEDS = tuple(range(1_650_000, 1_650_080))
OPTS = ("adam", "sgd")
BUDGET = 32_000
LR_ADAM, LR_SGD = 1e-2, 0.3
RULE_FRAC = 0.5
L1_BAND, L2_BAND, L3_MIN, MIN_CROSS = (0.90, 1.10), (0.80, 1.20), 0.5, 60
TWO_PI = 2 * math.pi


# ------------------------------------------------------------------------------------------ rules (pure functions)
def principal_copy(th, s):
    """Winding rule: shift (w₁, b₁, b₂) to the copy with b₁ ∈ (−π, π] (b₂ − 2πk·s keeps the logits)."""
    th = np.asarray(th, float).copy()
    k = -math.floor((th[1] + math.pi) / TWO_PI)          # b₁ + 2πk ∈ [−π, π)
    if th[1] + TWO_PI * k <= -math.pi:
        k += 1
    th[1] += TWO_PI * k
    th[2] -= TWO_PI * k * s
    return th, k


def rule_step(s_path, s_frozen, frac=RULE_FRAC):
    """Rule point: the step at which s_t last passes frac·s_frozen upward before s_t first reaches s_frozen, i.e. the
    first step t_R ≥ 1 such that s_t ≥ frac·s_frozen for every t_R ≤ t ≤ t_top (t_top = first step with s_t ≥ s_frozen).
    Uses the output-scale trajectory only (at a = 1.65, 0.5·s* ≈ 0.9 is below the initial |w₂| range, so "first
    reaches" would land in the initial transient).  None if s never reaches s_frozen."""
    s_path = np.asarray(s_path, float)
    top = np.nonzero(s_path[1:] >= s_frozen)[0]
    if len(top) == 0:
        return None
    t_top = int(top[0] + 1)
    below = np.nonzero(s_path[1:t_top] < frac * s_frozen)[0]
    return int(below[-1] + 2) if len(below) else 1


def kappa_closed(H, tan, dG, p):
    """κ for P = diag(p) (lag_law.kappa) and λ_min(P^{1/2}HP^{1/2})."""
    H, tan, dG, p = (np.asarray(v, float) for v in (H, tan, dG, p))
    ph = np.sqrt(p)
    lam = float(np.linalg.eigvalsh((ph[:, None] * H) * ph[None, :]).min())
    return lam * float(dG @ np.linalg.solve(p[:, None] * H, tan)) / float(dG @ tan), lam


def closed_form_r(H, tan, dG, p, sdot, s_star, lr):
    kap, lam = kappa_closed(H, tan, dG, p)
    chi = (sdot / s_star) / (lr * lam)
    return kap * chi, kap, chi


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def score_opt(r_obs, r_traj, r_cf, crossed, rule_before_cross):
    """Registered verdicts for one optimiser.  Arrays over the 80 runs; r_* NaN where no prediction/crossing.
    Runs whose rule point is not strictly before the crossing are excluded (and counted).  L1/L2: median over runs of
    r_obs/r_pred in the band; L3: Spearman(r_traj, r_obs) ≥ 0.5.  Validity: ≥ 60 crossings; each criterion also needs
    ≥ 60 crossing runs with its prediction, else UNRESOLVED."""
    r_obs, r_traj, r_cf = (np.asarray(v, float) for v in (r_obs, r_traj, r_cf))
    crossed = np.asarray(crossed, bool); rb = np.asarray(rule_before_cross, bool)
    n_cross = int(crossed.sum())
    use = crossed & rb
    out = {"n_runs": int(len(crossed)), "n_crossed": n_cross, "n_rule_not_before_crossing": int((crossed & ~rb).sum()),
           "valid": n_cross >= MIN_CROSS}

    def one(pred, band):
        ok = use & np.isfinite(pred) & np.isfinite(r_obs)
        n = int(ok.sum())
        if not out["valid"] or n < MIN_CROSS:
            return {"n": n, "median_ratio": float(np.median(r_obs[ok] / pred[ok])) if n else float("nan"),
                    "verdict": "UNRESOLVED"}
        med = float(np.median(r_obs[ok] / pred[ok]))
        return {"n": n, "median_ratio": med, "verdict": "PASS" if band[0] <= med <= band[1] else "FAIL"}
    out["L1"] = one(r_traj, L1_BAND)
    out["L2"] = one(r_cf, L2_BAND)
    ok = use & np.isfinite(r_traj) & np.isfinite(r_obs)
    n = int(ok.sum())
    rho = spearman(r_traj[ok], r_obs[ok]) if n >= 3 else float("nan")
    out["L3"] = {"n": n, "spearman": rho,
                 "verdict": "UNRESOLVED" if (not out["valid"] or n < MIN_CROSS) else ("PASS" if rho >= L3_MIN else "FAIL")}
    ok2 = use & np.isfinite(r_cf) & np.isfinite(r_obs)
    out["L3_closed_form_secondary"] = spearman(r_cf[ok2], r_obs[ok2]) if ok2.sum() >= 3 else float("nan")
    return out


def first_placed(gaps):
    """Index of the first positive gap (every-step detection on the saved path), None if never."""
    idx = np.nonzero(np.asarray(gaps) > 0)[0]
    return None if len(idx) == 0 else int(idx[0])


# ------------------------------------------------------------------------------------------ landscape
def _pop():
    from .width2_conditional import population
    return population()


def landscape():
    os.nice(15)
    from .own_threshold import global_min
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = _pop()
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    r = k[k.a.round(2) == 1.60].iloc[0]
    z0 = np.array(json.loads(r.z_star)); s0 = float(r.s_star)
    # continuation in a at fixed s from the 1.60 certified switch point, then to the principal copy
    z = z0.copy()
    for aa in np.linspace(1.60, A, 11)[1:]:
        z, res, H = LR.newton(z, s0, aa, x, y)
        assert res < 1e-9 and np.linalg.eigvalsh(H).min() > 0, (aa, res)
    z, kcopy = principal_copy(z, s0)
    z, res, _ = LR.newton(z, s0, A, x, y)
    B = LR.Branch(s0, z, A, x, y, 0.5 * s0, 1.5 * s0, 0.001 * s0)
    s_star = B.switch(s0)
    th, Hs, tan, res = B.exact(s_star)
    dG = LR.grad_gap(th[0], th[1], A)
    # validation by the existing conditional search
    val = {}
    for f_ in (0.995, 1.005):
        Lb, w1, b1, G, _ = global_min(f_ * s_star, A, x, y)
        thb, *_ = B.exact(f_ * s_star)
        same = min(abs(abs(w1) - abs(thb[0])), 9) < 2e-3 and \
            min(abs((b1 - thb[1]) % TWO_PI), TWO_PI - abs((b1 - thb[1]) % TWO_PI)) < 2e-3
        val[f"{f_}"] = {"global_min_w1": w1, "global_min_b1": b1, "global_min_G": G, "branch_w1": float(thb[0]),
                        "branch_b1": float(thb[1]), "same_branch_mod_mirror_2pi": bool(same)}
    ok = val["0.995"]["global_min_G"] <= 0 < val["1.005"]["global_min_G"] and val["1.005"]["same_branch_mod_mirror_2pi"]
    k_sgd, lam_sgd = kappa_closed(Hs, tan, dG, np.ones(3))
    out = {"a": A, "s_star_pop": s_star, "theta_star": th.tolist(), "H": Hs.tolist(), "tangent": tan.tolist(),
           "gradG": dG.tolist(), "grad_residual": res, "H_pd": bool(np.linalg.eigvalsh(Hs).min() > 0),
           "winding_shift_from_1p60_copy": int(kcopy), "b1_principal": float(th[1]), "kappa_sgd": k_sgd,
           "validation_global_min": val, "validated": bool(ok),
           "certified_bracket": "none exists at a = 1.65; validated by continuation + own_threshold.global_min"}
    (OUT / "landscape.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k_: v for k_, v in out.items() if k_ not in ("H",)}, indent=1))
    if not ok:
        raise SystemExit("STOP: landscape switch not validated by the conditional search")
    return out


def _land():
    d = json.loads((OUT / "landscape.json").read_text())
    return d, np.array(d["theta_star"]), np.array(d["H"]), np.array(d["tangent"]), np.array(d["gradG"])


# ------------------------------------------------------------------------------------------ per-seed frozen inputs
def _sample(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def frozen_one(seed):
    from .own_threshold import own_threshold
    d, th_pop, *_ = _land()
    s_pop = d["s_star_pop"]
    x, y = _sample(seed)
    th, res, H = LR.newton(th_pop, s_pop, A, x, y)
    row = {"seed": seed, "newton_res": res}
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        return {**row, "s_frozen": float("nan"), "note": "no own-sample minimum from the landscape point"}
    B = LR.Branch(s_pop, th, A, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop)
    s_fr = B.switch(s_pop)
    row.update(s_frozen=s_fr if s_fr is not None else float("nan"), branch_w1=float(th[0]), branch_b1=float(th[1]))
    ot = own_threshold(A, seed, s_fr if s_fr is not None else s_pop)
    row.update(w2_own_lo=ot["w2_lo"], w2_own_hi=ot["w2_hi"], w2_own=0.5 * (ot["w2_lo"] + ot["w2_hi"]), own_evals=ot["evals"],
               note=ot["note"])
    return row


def freeze():
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    f = OUT / "frozen_parts.jsonl"
    done = set() if not f.exists() else {json.loads(l)["seed"] for l in f.read_text().splitlines()}
    for s in SEEDS:
        if s in done:
            continue
        r = frozen_one(s)
        with open(f, "a") as fh:
            fh.write(json.dumps(LR._jsonable(r)) + "\n")
        print(json.dumps(LR._jsonable(r)), flush=True)
    d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()]).sort_values("seed")
    d.to_csv(OUT / "frozen_seeds.csv", index=False)
    print(len(d), int(d.s_frozen.isna().sum()))


# ------------------------------------------------------------------------------------------ training (no gap evaluated)
def train_one(opt_name, seed):
    """The registered run for the full budget, recording θ, and Adam's m, v̂, step at every step.  No placement or gap
    is evaluated here."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", A)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    x, y = x.double(), y.double()
    adam = opt_name == "adam"
    opt = torch.optim.Adam([th], lr=LR_ADAM) if adam else torch.optim.SGD([th], lr=LR_SGD)
    W = np.empty((BUDGET + 1, 4)); M = np.zeros((BUDGET + 1, 4)); V = np.full((BUDGET + 1, 4), np.nan)
    K = np.zeros(BUDGET + 1)
    W[0] = th.detach().numpy()
    for t in range(1, BUDGET + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        W[t] = th.detach().numpy()
        if adam:
            st = opt.state[th]; k = int(st["step"]); K[t] = k
            M[t] = st["exp_avg"].numpy(); V[t] = st["exp_avg_sq"].numpy() / (1 - 0.999 ** k)
    return W, M, V, K


def predict_one(opt_name, seed, W, M, V, K, s_frozen):
    d, th_pop, H_pop, tan_pop, dG_pop = _land()
    x, y = _sample(seed)
    adam = opt_name == "adam"
    lr = LR_ADAM if adam else LR_SGD
    sign = np.sign(W[:, 2]); s = np.abs(W[:, 2])
    flip = np.where(sign[:, None] < 0, LR.D_FLIP[None, :], 1.0)
    TH = W[:, [0, 1, 3]] * flip; MM = M[:, [0, 1, 3]] * flip
    out = {"opt": opt_name, "seed": seed, "s_frozen": s_frozen}
    tR = rule_step(s, s_frozen)
    if tR is None:
        return {**out, "status": "never reaches the rule point"}
    out["t_rule"] = tR; out["s_rule"] = float(s[tR])
    th_b, res, Hn = LR.newton(TH[tR], float(s[tR]), A, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(Hn).min() <= 0:
        return {**out, "status": "branch rule: no minimum at the rule point"}
    B = LR.Branch(float(s[tR]), th_b, A, x, y, 0.3 * s_frozen, 1.7 * s_frozen, 0.002 * s_frozen)
    s_run = B.switch(s_frozen)
    if s_run is None:
        return {**out, "status": "branch rule: occupied branch has no switch"}
    out.update(s_run=s_run, branch_b1_rule=float(th_b[1]), branch_w1_rule=float(th_b[0]),
               winding_rule_agrees=bool(-math.pi < th_b[1] <= math.pi), branch_lo=B.lo, branch_hi=B.hi)
    P_R = (1.0 / (np.sqrt(V[tR, [0, 1, 3]]) + LR.EPS)) if adam else np.ones(3)
    out.update(P_w1=float(P_R[0]), P_b1=float(P_R[1]), P_b2=float(P_R[2]))
    # closed form (landscape κ with P_R's shape; χ from the s path)
    t_sw = LR._first_ge(s, s_run)
    if t_sw is None:
        out["r_cf"] = float("nan"); out["status_cf"] = "s never reaches s*_run"
    else:
        win = min(100, t_sw); sdot = (s[t_sw] - s[t_sw - win]) / win
        r_cf, kap, chi = closed_form_r(H_pop, tan_pop, dG_pop, P_R, sdot, s_run, lr)
        out.update(r_cf=r_cf, kappa=kap, chi=chi, sdot=float(sdot), status_cf="ok")
    # trajectory-integrated (P frozen at P_R)
    s_seg = s[tR:]
    inside = (s_seg >= B.lo) & (s_seg <= B.hi)
    T_end = len(s_seg) if inside.all() else int(np.argmin(inside))
    s_seg = s_seg[:T_end]
    th_seg = [B.theta(v) for v in s_seg]
    Hs = [B.hess(v) for v in s_seg]
    d0 = TH[tR] - th_seg[0]
    dth = [th_seg[i + 1] - th_seg[i] for i in range(T_end - 1)]
    PP = np.tile(P_R, (T_end, 1))
    bc1 = (1 - LR.BETA1 ** K[tR:tR + T_end]) if adam else None
    gapf = lambda i, dd: LR.gap_exact(th_seg[i][0] + dd[0], th_seg[i][1] + dd[1], A)
    t_hit, path = LR.simulate(s_seg, lambda t: Hs[t], PP, dth, gapf, d0, m0=MM[tR] if adam else None, bc1=bc1, lr=lr)
    n_sim = len(path)
    idx = sorted(set(range(0, n_sim, 10)) | {n_sim - 1})
    rho = max(LR.step_map_radius(Hs[i], P_R, lr, adam) for i in idx)
    mx = float(np.nanmax(np.abs(np.array(path))))
    out.update(traj_rho_max=rho, traj_max_abs_delta=mx, traj_T=T_end, delta0_norm=float(np.abs(d0).max()))
    if t_hit is None or rho > 1.0 or not np.isfinite(mx) or mx > LR.DIVERGED:
        out.update(r_traj=float("nan"), s_traj=float("nan"), status_traj="no prediction (no hit / unstable / diverged)")
    else:
        out.update(s_traj=float(s_seg[t_hit]), r_traj=float(s_seg[t_hit]) / s_run - 1, t_traj=int(tR + t_hit), status_traj="ok")
    out["status"] = "ok"
    return out


def _path_file(opt_name, seed):
    return PATHS / f"{opt_name}_{seed}.npz"


def train(opts=OPTS):
    """One process per optimiser is allowed (independent, deterministic; each writes its own parts file)."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    PATHS.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(OUT / "frozen_seeds.csv").set_index("seed")
    for opt_name in opts:
        f = OUT / f"predictions_parts_{opt_name}.jsonl"
        done = set() if not f.exists() else {(r["opt"], r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
        for seed in SEEDS:
            if (opt_name, seed) in done:
                continue
            W, M, V, K = train_one(opt_name, seed)
            pf = _path_file(opt_name, seed)
            np.savez(pf, W=W)
            h = hashlib.sha256(pf.read_bytes()).hexdigest()
            s_fr = float(fr.loc[seed, "s_frozen"])
            try:
                r = predict_one(opt_name, seed, W, M, V, K, s_fr) if np.isfinite(s_fr) else \
                    {"opt": opt_name, "seed": seed, "status": "no frozen switch"}
            except Exception as e:
                r = {"opt": opt_name, "seed": seed, "status": f"error: {type(e).__name__}: {e}"}
            r["path_sha256"] = h
            with open(f, "a") as fh:
                fh.write(json.dumps(LR._jsonable(r)) + "\n")
            print(json.dumps({k: r.get(k) for k in ("opt", "seed", "status", "r_traj", "r_cf")}), flush=True)


def finalize():
    d = pd.DataFrame([json.loads(l) for o in OPTS for l in (OUT / f"predictions_parts_{o}.jsonl").read_text().splitlines()])
    d = d.sort_values(["opt", "seed"])
    forbidden = {"cross_step", "s_obs", "r_obs", "step_obs", "placed"}
    assert not (forbidden & set(d.columns))
    p = OUT / "predictions.csv"
    d.to_csv(p, index=False)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "predictions.sha256").write_text(f"{h}  predictions.csv\n")
    print(h, len(d), d.status.value_counts().to_dict())


# ------------------------------------------------------------------------------------------ observation and scoring
def _assert_committed(p, sha):
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == sha.read_text().split()[0], "hash mismatch"
    rel = str(p.relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"
    return h


def observe():
    import torch
    from .fold1d import activation
    from .phase2b_ordering import state
    _assert_committed(OUT / "predictions.csv", OUT / "predictions.sha256")
    pr = pd.read_csv(OUT / "predictions.csv")
    f = activation("sin_family", A)
    rows = []
    for r in pr.itertuples():
        pf = _path_file(r.opt, r.seed)
        assert hashlib.sha256(pf.read_bytes()).hexdigest() == r.path_sha256, f"path hash mismatch {pf}"
        W = np.load(pf)["W"]
        t_c = None
        for t in range(1, len(W)):
            if state(torch.tensor(W[t]), f, A, 1.0)["placement_ok"]:
                t_c = t
                break
        rows.append({"opt": r.opt, "seed": r.seed, "crossed": t_c is not None, "step_obs": t_c,
                     "s_obs": abs(float(W[t_c, 2])) if t_c is not None else float("nan")})
    o = pd.DataFrame(rows)
    fr = pd.read_csv(OUT / "frozen_seeds.csv")
    m = pr.merge(o, on=["opt", "seed"]).merge(fr[["seed", "w2_own"]], on="seed", how="left")
    m["r_obs"] = m.s_obs / m.s_run - 1
    m["r_obs_vs_global_own"] = m.s_obs / m.w2_own - 1
    m["rule_before_cross"] = m.t_rule < m.step_obs
    m.to_csv(OUT / "observed_runs.csv", index=False)
    res = {}
    for opt_name, g in m.groupby("opt"):
        res[opt_name] = score_opt(g.r_obs, g.r_traj, g.r_cf, g.crossed, g.rule_before_cross.fillna(False))
        ok = g[g.crossed & np.isfinite(g.r_traj)]
        res[opt_name]["descriptive"] = {
            "median_r_obs": float(g.r_obs.median()), "median_r_traj": float(g.r_traj.median()),
            "median_r_cf": float(g.r_cf.median()), "n_winding_rule_agrees": int(g.winding_rule_agrees.fillna(False).sum()),
            "n_status_ok": int((g.status == "ok").sum()), "n_traj_pred": int(np.isfinite(g.r_traj).sum()),
            "n_cf_pred": int(np.isfinite(g.r_cf).sum()),
            "median_s_run_over_s_frozen": float((g.s_run / g.s_frozen).median()),
            "frac_s_run_off_frozen_1pct": float(((g.s_run / g.s_frozen - 1).abs() > 0.01).mean()),
            "slope_obs_on_traj": LR._slope(ok.r_traj.to_numpy(), ok.r_obs.to_numpy())[0],
            "median_ratio_ts_obs_over_traj_step": float((ok.step_obs / ok.t_traj).median()) if "t_traj" in ok else float("nan")}
    res["predictions_sha256"] = (OUT / "predictions.sha256").read_text().split()[0]
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(tuple(sys.argv[2:]) or OPTS)
    else:
        {"landscape": landscape, "freeze": freeze, "finalize": finalize, "observe": observe}[sys.argv[1]]()
