"""Width-2 fixed-scale test that output scale does not gate placement (design, approved with additions:
results/width2_nogating_design.md, revision 1).  For the rebuttal revision.  Runs only after the direct small-scale
check (scale_limits_prediction.md, last section) confirms the registered verdict.

    python -m src.width2_nogating validate            # validity checks on constructed pass and fail cases (tests too)
    python -m src.width2_nogating pilot [workers]     # horizon pilot (seeds 500,000-500,019, R₂ = 0.5); writes H
    python -m src.width2_nogating freeze              # width2_nogating_frozen.json (k, H, seeds, levels) + SHA-256
    python -m src.width2_nogating train [workers]     # checkpoints: seeds 630,000-630,079, f1.30, f1.50, tanh
    python -m src.width2_nogating control [workers]   # width-1 positive control, seeds 630,000-630,019
    python -m src.width2_nogating replay [workers]    # the scored replays
    python -m src.width2_nogating score
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "width2_nogating_parts"
FROZEN = RESULTS / "width2_nogating_frozen.json"

K_INIT = 0.46862                                   # approved: median ‖v‖₁ at init = 0.45899 (width-1 median |w₂|)
SEEDS = tuple(range(630_000, 630_080))
PILOT_SEEDS = tuple(range(500_000, 500_020))
CTRL_SEEDS = tuple(range(630_000, 630_020))
R2_LEVELS = (0.003, 0.01, 0.03, 0.1)
PILOT_R2 = 0.5
PILOT_STEPS, PILOT_EVERY = 128_000, 100
H_CHOICES = (16_000, 32_000, 64_000, 128_000)
CTRL_LEVELS = (0.1, 0.3, 0.5)
VARIANTS = ("preserved", "reset")
ACT_NAMES = ("f1.30", "f1.50", "tanh")
EXCLUDE_STOP = 0.20
CTRL_STOP = 0.20
PAIR_SHARE, PAIR_ALPHA, PAIR_CANCEL = (0.4, 0.6), 0.10, 0.10


def act_of(name):
    from .width2_geometry import Act
    return Act("tanh") if name == "tanh" else Act("fa", float(name[1:]))


def gamma2(name):
    """Γ̂₂: the validated NM value for f_a (width2_w0 gamma_*.csv); 1 for tanh (its supremum, not attained)."""
    if name == "tanh":
        return 1.0
    g = pd.read_csv(RESULTS / "width2_w0_parts" / f"gamma_{name}.csv")
    return float(g[g.search == "nm"].gamma_lo.iloc[0])


# ------------------------------------------------------------------------------------------ training (checkpoints)
def train_matched(seed, act, k=K_INIT, budget=32_000, ck_steps=None):
    """width2_train.train with the output weights v scaled by k at initialisation (b unchanged), placement checked at
    every step FROM STEP 0, and full states saved at step 0 and at log-spaced steps while G₊ ≤ 0.

    Returns {"seed", "placed_at_init", "cross_step" (None if never), "ck" (the latest pre-placement state or None),
    "ck_step", "n_saved"}."""
    import torch
    from .width2_conditional import training_set
    from .width2_train import LR, init_params, log_spaced_steps, logits, placed
    torch.set_num_threads(1)
    ck_steps = set(ck_steps if ck_steps is not None else log_spaced_steps(budget))
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed)
    q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    if placed(q.detach().numpy(), act)[0]:
        return {"seed": seed, "placed_at_init": True, "cross_step": 0, "ck": None, "ck_step": None, "n_saved": 0}
    last = {"q": q.detach().numpy().copy(), "adam": {}}
    last_step, n_saved = 0, 1
    for step in range(1, budget + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        qn = q.detach().numpy().copy()
        if placed(qn, act)[0]:
            return {"seed": seed, "placed_at_init": False, "cross_step": step, "ck": last, "ck_step": last_step,
                    "n_saved": n_saved}
        if step in ck_steps:
            last = {"q": qn, "adam": {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                                      for kk, vv in opt.state[q].items()}}
            last_step, n_saved = step, n_saved + 1
    return {"seed": seed, "placed_at_init": False, "cross_step": None, "ck": last, "ck_step": last_step,
            "n_saved": n_saved}


# ------------------------------------------------------------------------------------------ per-unit breakdown
def breakdown(q, act):
    """Weight shares, α, β, the cancellation index, the knockout class and the cancelling-pair flags."""
    from .width2_geometry import gaps
    from .width2_train import unpack
    from .width2_w0 import _is_pair
    th, v, b = unpack(q)
    n1 = abs(v[0]) + abs(v[1])
    share = (abs(v[0]) / n1, abs(v[1]) / n1)
    al = (th[0], th[2])
    denom = abs(v[0] * al[0]) + abs(v[1] * al[1])
    cancel = abs(v[0] * al[0] + v[1] * al[1]) / denom if denom > 0 else float("nan")

    def G(vv):
        return gaps(th, np.asarray(vv, float), act)["G+"][0]
    both = G(v) > 0
    solo = (G([v[0], 0.0]) > 0, G([0.0, v[1]]) > 0)
    klass = ("unplaced" if not both else "redundant" if all(solo) else "single-unit" if any(solo) else "shared")
    pair = (PAIR_SHARE[0] <= share[0] <= PAIR_SHARE[1] and PAIR_SHARE[0] <= share[1] <= PAIR_SHARE[1]
            and abs(abs(al[0]) - abs(al[1])) <= PAIR_ALPHA * max(abs(al[0]), abs(al[1]))
            and cancel <= PAIR_CANCEL and klass == "shared")
    t = v[0] / n1
    strict, _ = _is_pair(np.array([th[0], th[1], th[2], th[3], t]), 1.0 if v[1] >= 0 else -1.0)
    return {"share1": share[0], "share2": share[1], "alpha1": al[0], "alpha2": al[1], "beta1": th[1], "beta2": th[3],
            "cancel_index": cancel, "knockout": klass, "pair": bool(pair), "pair_strict": bool(strict),
            "max_abs_alpha": max(abs(al[0]), abs(al[1]))}


# ------------------------------------------------------------------------------------------ replays
def replay_ng(ck, radius, steps, x, y, act, variant, record, every=None):
    """width2_train.replay's operations (rescale to ‖v‖₁ = radius, Adam, radial ℓ₁ projection after every step), recording
    placement (exact G₊), sign-correctness and the per-unit breakdown at the `record` steps; if `every` is given, the
    placement indicator every `every` steps as well (the horizon pilot)."""
    import torch
    from .width2_geometry import sign_correct
    from .width2_train import LR, _project, logits, placed, rescale, unpack
    torch.set_num_threads(1)
    k = radius / (abs(ck["q"][2]) + abs(ck["q"][5]))
    q = torch.tensor(rescale(ck["q"], k), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=LR)
    if variant == "preserved" and ck["adam"]:
        opt.state[q] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in ck["adam"].items()}
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    out, trace, drift = {}, [], 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        _project(q, radius)
        qd = q.detach()
        drift = max(drift, abs(float(qd[2].abs() + qd[5].abs()) - radius))
        if every and step % every == 0:
            try:
                trace.append(placed(qd.numpy(), act)[0])
            except RuntimeError:
                trace.append(None)
        if step in record:
            qn = qd.numpy().copy()
            th, v, b = unpack(qn)
            try:
                pl, glo = placed(qn, act)
            except RuntimeError:
                pl, glo = None, None
            out[step] = {"placed": pl, "G_lo": glo, "sign_correct": sign_correct(th, v, b, act), **breakdown(qn, act)}
    return {"record": out, "trace": trace, "drift": drift, "k": k}


def settling_time(trace, every=PILOT_EVERY, total=PILOT_STEPS):
    """First check after which the placement indicator never changes again; total if it still changes in the last
    10% of the run.  An undecided check (None) counts as a change."""
    tr = list(trace)
    last_change = 0
    for i in range(1, len(tr)):
        if tr[i] != tr[i - 1] or tr[i] is None:
            last_change = i
    t = (last_change + 1) * every if last_change > 0 else every
    if last_change > 0 and (last_change + 1) * every > 0.9 * total:
        return total
    return t


def choose_H(T):
    """Smallest of H_CHOICES that is >= 4 x the 95th percentile of the settling times; None (stop) if none is."""
    need = 4 * float(np.percentile(np.asarray(T, float), 95))
    for H in H_CHOICES:
        if H >= need:
            return H, need
    return None, need


# ------------------------------------------------------------------------------------------ width-1 positive control
def _w1_consts(a):
    from .ghat_rigorous import ghat_R_of            # the rigorous Ĝ_cert (Block 2, author's decision 2026-09-24)
    G = ghat_R_of(a)
    t = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    r = t[(t.a.round(2) == round(a, 2)) & (t.kind == "glob")].iloc[0]
    return G, 0.5 * (r.w2_lo + r.w2_hi)


def w1_train(seed, a, budget=12_000):
    """fixed_scale.train_one's protocol at any a, placement (oriented dense gap > 0) checked from step 0; returns the
    latest pre-placement state (step 0 allowed) as a fixed_scale checkpoint dict."""
    import torch
    from torch.nn import functional as F
    from .fixed_scale import _gap
    from .fold1d import activation, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    marks = set(np.unique(np.logspace(0, np.log10(budget), 60).astype(int)))

    def snap(step):
        st = opt.state[th]
        z = np.zeros(4)
        return {"seed": seed, "step": step, "theta": th.detach().clone().numpy(),
                "exp_avg": st["exp_avg"].clone().numpy() if st else z, "exp_avg_sq": st["exp_avg_sq"].clone().numpy() if st else z,
                "adam_step": int(st["step"]) if st else 0}
    w1, b1, w2, b2 = (float(v) for v in th.detach())
    if _gap(f, w1, b1, w2) > 0:
        return {"seed": seed, "placed_at_init": True, "ck": None, "cross_step": 0}
    last = snap(0)
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        w1, b1, w2, b2 = (float(v) for v in th.detach())
        if _gap(f, w1, b1, w2) > 0:
            return {"seed": seed, "placed_at_init": False, "ck": last, "cross_step": step}
        if step in marks:
            last = snap(step)
    return {"seed": seed, "placed_at_init": False, "ck": last, "cross_step": None}


def w1_replay(ck, a, level, variant, horizon, G_hat, w2_glob):
    """fixed_scale.replay at any a (w₂ excluded from the optimiser; preserved or reset Adam state for w₁, b₁, b₂).  At
    a = 1.30 it must equal fixed_scale.replay exactly (checked in validate())."""
    from . import fixed_scale as FS
    old = FS.A
    try:
        FS.A = a
        return FS.replay(ck, level, variant, horizon, "place", G_hat, w2_glob)
    finally:
        FS.A = old


# ------------------------------------------------------------------------------------------ scoring
def clopper_lo(k, n, alpha=0.05):
    from .fixed_scale import _clopper
    return _clopper(k, n, alpha)[0]


def score_levels(frac_by_level, n, placed_matrix):
    """frac_by_level: {R₂: placed fraction at H} (primary variant); placed_matrix: {R₂: array of 0/1 per seed, aligned}.
    Returns the registered outcome."""
    from .width2_train import mcnemar_one_sided
    levels = sorted(frac_by_level)
    lo = {L: clopper_lo(int(round(frac_by_level[L] * n)), n) for L in levels}
    no_gating = all(frac_by_level[L] >= 0.90 and lo[L] >= 0.80 for L in levels)
    no_sig_decrease = True
    for L1, L2 in zip(levels, levels[1:]):                   # decrease from L1 to L2: placed at L1, not at L2
        a_, b_ = np.asarray(placed_matrix[L1]), np.asarray(placed_matrix[L2])
        dec, inc = int(((a_ == 1) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ == 1)).sum())
        if mcnemar_one_sided(dec, inc) < 0.05:
            no_sig_decrease = False
    gating = frac_by_level[levels[0]] <= 0.50 and no_sig_decrease
    return ("no gating (predicted)" if no_gating else "gating at small scale (competing)" if gating else "neither"), lo


def control_ok(frac_at_01):
    return frac_at_01 <= CTRL_STOP


def exclusion_ok(n_excluded, n):
    return n_excluded <= EXCLUDE_STOP * n


# ------------------------------------------------------------------------------------------ frozen parameters
def freeze(H, need, pilot_T):
    d = {"k_init": K_INIT, "H": H, "four_p95_T": need, "pilot_T": [int(t) for t in pilot_T], "R2_levels": R2_LEVELS,
         "ctrl_levels": CTRL_LEVELS, "seeds": [SEEDS[0], SEEDS[-1]], "ctrl_seeds": [CTRL_SEEDS[0], CTRL_SEEDS[-1]],
         "pilot_seeds": [PILOT_SEEDS[0], PILOT_SEEDS[-1]]}
    s = json.dumps(d, indent=1, sort_keys=True)
    FROZEN.write_text(s)
    h = hashlib.sha256(s.encode()).hexdigest()
    (RESULTS / "width2_nogating_frozen.sha256").write_text(h + "\n")
    return h


def frozen():
    s = FROZEN.read_text()
    want = (RESULTS / "width2_nogating_frozen.sha256").read_text().strip()
    if hashlib.sha256(s.encode()).hexdigest() != want:
        raise SystemExit("STOP: frozen parameters do not match their committed hash")
    return json.loads(s)


# ------------------------------------------------------------------------------------------ jobs
def _append(name, rows):
    PARTS.mkdir(exist_ok=True)
    p = PARTS / name
    pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


def _done(name, keys):
    p = PARTS / name
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(map(tuple, d[list(keys)].astype(str).values.tolist()))


def _pilot_job(args):
    name, seed = args
    from .width2_conditional import training_set
    act = act_of(name)
    tr = train_matched(seed, act)
    if tr["ck"] is None:
        return {"act": name, "seed": seed, "excluded": True, "T": np.nan}
    x, y = training_set(seed)
    r = replay_ng(tr["ck"], 2 * PILOT_R2 / gamma2(name), PILOT_STEPS, x, y, act, "preserved", record=(),
                  every=PILOT_EVERY)
    return {"act": name, "seed": seed, "excluded": False, "ck_step": tr["ck_step"], "T": settling_time(r["trace"]),
            "final_placed": r["trace"][-1], "drift": r["drift"]}


def pilot(workers=2):
    jobs = [(n, s) for n in ("f1.30", "f1.50") for s in PILOT_SEEDS]
    done = _done("pilot.csv", ("act", "seed"))
    jobs = [j for j in jobs if (j[0], str(j[1])) not in done]
    with Pool(workers) as p:
        for row in p.imap_unordered(_pilot_job, jobs, chunksize=1):
            _append("pilot.csv", [row]); print(row, flush=True)
    d = pd.read_csv(PARTS / "pilot.csv")
    T = d[~d.excluded].T.values
    H, need = choose_H(T)
    print("pilot: p95(T) =", np.percentile(T, 95), "4 x p95 =", need, "H =", H)
    if H is None:
        print("STOP: 4 x p95(T) > 128,000; no H is chosen"); raise SystemExit(2)
    return H, need, T


def _train_job(args):
    name, seed = args
    import pickle
    tr = train_matched(seed, act_of(name))
    PARTS.mkdir(exist_ok=True)
    with open(PARTS / f"ck_{name}_{seed}.pkl", "wb") as fh:
        pickle.dump(tr, fh)
    return {"act": name, "seed": seed, "placed_at_init": tr["placed_at_init"], "cross_step": tr["cross_step"],
            "ck_step": tr["ck_step"], "n_saved": tr["n_saved"]}


def train(workers=2):
    jobs = [(n, s) for n in ACT_NAMES for s in SEEDS]
    done = _done("train.csv", ("act", "seed"))
    jobs = [j for j in jobs if (j[0], str(j[1])) not in done]
    with Pool(workers) as p:
        for row in p.imap_unordered(_train_job, jobs, chunksize=1):
            _append("train.csv", [row])
    d = pd.read_csv(PARTS / "train.csv")
    for n, g in d.groupby("act"):
        ex = int(g.placed_at_init.sum())
        print(n, "placed at init (excluded):", ex, "of", len(g), "| step-0 checkpoints:", int((g.ck_step == 0).sum()),
              "| never placed:", int(g.cross_step.isna().sum()))
        if n != "tanh" and not exclusion_ok(ex, len(g)):
            print("STOP: more than 20% placed at initialisation at", n); raise SystemExit(2)


def _replay_job(args):
    import pickle
    from .width2_conditional import training_set
    name, seed, R2, variant, H = args
    with open(PARTS / f"ck_{name}_{seed}.pkl", "rb") as fh:
        tr = pickle.load(fh)
    x, y = training_set(seed)
    act = act_of(name)
    rec = (H // 16, H // 4, H)
    r = replay_ng(tr["ck"], 2 * R2 / gamma2(name), H, x, y, act, variant, record=rec)
    rows = []
    for t in rec:
        rows.append({"act": name, "seed": seed, "R2": R2, "variant": variant, "step": t, "H": H,
                     "ck_step": tr["ck_step"], "drift": r["drift"], **r["record"][t]})
    return rows


def replay(workers=3):
    fz = frozen()
    H = int(fz["H"])
    tr = pd.read_csv(PARTS / "train.csv")
    ok = tr[~tr.placed_at_init.astype(bool)]
    jobs = [(r.act, int(r.seed), R2, v, H) for r in ok.itertuples() for R2 in R2_LEVELS for v in VARIANTS]
    done = _done("replays.csv", ("act", "seed", "R2", "variant"))
    jobs = [j for j in jobs if (j[0], str(j[1]), str(j[2]), j[3]) not in done]
    with Pool(workers) as p:
        for rows in p.imap_unordered(_replay_job, jobs, chunksize=1):
            _append("replays.csv", rows)
            if rows[-1]["drift"] > 1e-12:
                print("STOP: ‖v‖₁ drift", rows[-1]["drift"], rows[-1]); raise SystemExit(2)


def _ctrl_job(args):
    a, seed, H = args
    G, w2g = _w1_consts(a)
    tr = w1_train(seed, a)
    if tr["ck"] is None:
        return [{"a": a, "seed": seed, "excluded": True}]
    rows = []
    for L in CTRL_LEVELS:
        for v in VARIANTS:
            r = w1_replay(tr["ck"], a, L, v, H, G, w2g)
            rows.append({"a": a, "seed": seed, "excluded": False, "ck_step": tr["ck"]["step"], "level": L, "variant": v,
                         "placed_end": r["placed_end"], "G_end": r["G_end"], "first_hit": r["first_hit"]})
    return rows


def control(workers=2):
    H = int(frozen()["H"])
    jobs = [(a, s, H) for a in (1.30, 1.50) for s in CTRL_SEEDS]
    done = _done("control.csv", ("a", "seed"))
    jobs = [j for j in jobs if (str(j[0]), str(j[1])) not in done]
    with Pool(workers) as p:
        for rows in p.imap_unordered(_ctrl_job, jobs, chunksize=1):
            _append("control.csv", rows)
    d = pd.read_csv(PARTS / "control.csv")
    d = d[~d.excluded.astype(bool)]
    for a, g in d.groupby("a"):
        f01 = float(g[(g.level == 0.1) & (g.variant == "preserved")].placed_end.astype(bool).mean())
        print("control a =", a, "placed fraction at R/R_glob = 0.1:", f01)
        if not control_ok(f01):
            print("STOP: the machinery cannot detect gating (control placed fraction > 0.2 at 0.1)"); raise SystemExit(2)


def score():
    fz = frozen()
    H = int(fz["H"])
    d = pd.read_csv(PARTS / "replays.csv")
    rows = []
    for name in ("f1.30", "f1.50", "tanh"):
        g = d[(d.act == name) & (d.step == H)]
        for variant in VARIANTS:
            gv = g[g.variant == variant]
            piv = gv.pivot_table(index="seed", columns="R2", values="placed", aggfunc="first").dropna()
            piv = piv.astype(float)
            n = len(piv)
            fr = {float(L): float(piv[L].mean()) for L in piv.columns}
            out, lo = score_levels(fr, n, {float(L): piv[L].values for L in piv.columns})
            for L in piv.columns:
                gl = gv[gv.R2 == L]
                rows.append({"act": name, "variant": variant, "R2": float(L), "n": n, "placed_frac": fr[float(L)],
                             "cp_lo": lo[float(L)], "sign_correct_frac": float((gl.sign_correct == True).mean()),
                             "pair_frac_placed": float(gl[gl.placed == True].pair.mean()) if (gl.placed == True).any() else np.nan,
                             "pair_frac_unplaced": float(gl[gl.placed == False].pair.mean()) if (gl.placed == False).any() else np.nan,
                             "pair_strict_frac": float(gl.pair_strict.mean()),
                             "max_abs_alpha_median": float(gl.max_abs_alpha.median()),
                             "outcome": out if (variant == "preserved" and name != "tanh") else f"descriptive: {out}"})
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "width2_nogating_scores.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False))


# ------------------------------------------------------------------------------------------ validity (constructed cases)
def validate():
    """The design's validity checks, each on a pass and a fail case, on 20 checkpoints (f1.30, pilot seeds' first 20
    log-spaced states).  Any failure stops."""
    import torch
    from .width2_conditional import training_set
    from .width2_train import (decisions_preserved, dense_crossing_check, determinism_ok, freeze_ok, k1_check,
                               log_spaced_steps, placed, reference_true_freeze, train)
    act = act_of("f1.30")
    res = []
    # k = 1 reproduction on 20 checkpoints (pass) and a gradient-only freeze reference (fail)
    cks = []
    for s in PILOT_SEEDS:                                  # up to 5 per seed until 20 are collected
        r = train(s, act, gamma2("f1.30"), budget=400, checkpoint_steps=log_spaced_steps(400, 12))
        x, y = training_set(s)
        cks += [(c, x, y) for st, c in sorted(r["checkpoints"].items()) if st > 5][:5]
        if len(cks) >= 20:
            cks = cks[:20]
            break
    diffs = [k1_check(c, x, y, act)[1] for c, x, y in cks]
    res.append((f"k=1 reproduction, {len(cks)} checkpoints (20 required), <= 1e-10",
                len(cks) >= 20 and max(diffs) <= 1e-10, max(diffs)))
    c, x, y = cks[0]
    bad = reference_true_freeze(c, 25, x, y, act, reset=False)
    res.append(("fail case: gradient-only freeze reference rejected", not k1_check(c, x, y, act, reference=bad)[0], None))
    res.append(("decisions preserved at k = 0.01", decisions_preserved(c["q"], 0.01, x, act), None))
    res.append(("fail case: k < 0 flips decisions", not decisions_preserved(c["q"], -1.0, x, act), None))
    r = replay_ng(c, 0.02, 200, x, y, act, "preserved", record=(200,))
    res.append(("drift <= 1e-12 with projection", freeze_ok(r["drift"]), r["drift"]))
    torch.manual_seed(0)
    from . import width2_train as W
    old = W._project
    try:
        W._project = lambda q, radius: None
        r2 = replay_ng(c, 0.02, 200, x, y, act, "preserved", record=(200,))
    finally:
        W._project = old
    res.append(("fail case: no projection drifts", not freeze_ok(r2["drift"]), r2["drift"]))
    res.append(("training determinism", determinism_ok(PILOT_SEEDS[0], act, gamma2("f1.30"), budget=300), None))
    # K_INIT = 1 reproduces width2_train.train's crossing (the matched init is the only change)
    a_ = train_matched(PILOT_SEEDS[0], act, k=1.0, budget=3000)
    b_ = train(PILOT_SEEDS[0], act, gamma2("f1.30"), budget=3000)
    same = (a_["cross_step"] == (b_["cross"]["step"] if b_["cross"] else None))
    res.append(("k_init = 1 reproduces width2_train.train's crossing step", same, a_["cross_step"]))
    # crossing detection against the independent dense check, on the replay endpoints
    q_end = r["record"][200]
    res.append(("placement agrees with the independent 20,001-point dense check",
                bool(placed(c["q"], act)[0]) == dense_crossing_check(c["q"], act), None))
    # width-1 control replay at a = 1.30 equals fixed_scale.replay exactly
    from . import fixed_scale as FS
    tr1 = w1_train(CTRL_SEEDS[0], 1.30)
    G, w2g = _w1_consts(1.30)
    r_a = w1_replay(tr1["ck"], 1.30, 0.3, "preserved", 200, G, w2g)
    r_b = FS.replay(tr1["ck"], 0.3, "preserved", 200, "place", G, w2g)
    res.append(("width-1 control replay = fixed_scale.replay at a = 1.30", r_a["traj_G"] == r_b["traj_G"], None))
    df = pd.DataFrame(res, columns=["check", "pass", "value"])
    df.to_csv(RESULTS / "width2_nogating_validity.csv", index=False)
    print(df.to_string(index=False))
    if not df["pass"].all():
        print("STOP: a validity check failed"); raise SystemExit(2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if cmd == "validate":
        validate()
    elif cmd == "pilot":
        pilot(w)
    elif cmd == "freeze":
        H, need, T = pilot(w)
        print("sha256", freeze(H, need, T))
    elif cmd == "train":
        train(w)
    elif cmd == "control":
        control(w)
    elif cmd == "replay":
        replay(w)
    elif cmd == "score":
        score()
