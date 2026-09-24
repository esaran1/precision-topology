"""Width 2 (Route A): unconstrained training, crossing detection, the W4 fixed-scale replays and their validity
checks, and the scorers for W1–W4 and the tanh criterion (design §4–§5).  Nothing here is run until reported.

Protocol: width 2, float64, full batch, Adam lr 1e−2; torch.manual_seed(seed), then (α₁, β₁, v₁, α₂, β₂, v₂, b)
~ U(−1, 1)⁷; fold1d.make_data(200, seed); budget 32,000 steps.  Crossing: the first step at which G₊(φ_v) > 0
with exact extrema.  The dense-grid G₊ is an upper bound on the exact one (the grid is a subset of each window),
so the exact extrema are evaluated only when the grid value is positive: the every-step check is exact.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .width2_geometry import INNER, OUTER, Act, gaps, phi, sign_correct

LR, BUDGET = 1e-2, 32_000
_XI = np.linspace(*INNER, 401)
_XO = np.concatenate([np.linspace(*OUTER[0], 201), np.linspace(*OUTER[1], 201)])


def _u_torch(act, t):
    return t + act.a * torch.sin(t) if act.name == "fa" else torch.tanh(t)


def init_params(seed):
    torch.manual_seed(seed)
    return torch.empty(7, dtype=torch.float64).uniform_(-1, 1)      # α₁, β₁, v₁, α₂, β₂, v₂, b


def unpack(q):
    q = np.asarray(q, float)
    return np.array([q[0], q[1], q[3], q[4]]), np.array([q[2], q[5]]), float(q[6])


def logits(q, x, act):
    return q[2] * _u_torch(act, q[0] * x + q[1]) + q[5] * _u_torch(act, q[3] * x + q[4]) + q[6]


def dense_gplus(th, v, act):
    return float(phi(_XO, th, v, act).min() - phi(_XI, th, v, act).max())


def placed(q, act):
    """G₊(φ_v) > 0, exactly.  Dense screen first (an upper bound), exact extrema only if the screen is positive.
    Returns (placed, exact G₊ lower end or None)."""
    th, v, _ = unpack(q)
    if dense_gplus(th, v, act) <= 0:
        return False, None
    lo, hi = gaps(th, v, act)["G+"]
    if lo > 0:
        return True, lo
    if hi <= 0:
        return False, lo
    raise RuntimeError("crossing undecided at extrema tolerance")


def train(seed, act, gamma2_hat, budget=BUDGET, log_every=50, checkpoint_steps=None, x=None, y=None):
    """One run.  Returns the crossing (step, R₂ = ‖v‖₁Γ̂₂/2) or None, a log of (step, q) every log_every steps, and
    full-state checkpoints (params + Adam state) at the requested steps while G₊ <= 0."""
    from .width2_conditional import training_set
    if x is None:
        x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q = init_params(seed).requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    ck, log, cross = {}, [], None
    ck_steps = set(checkpoint_steps or [])
    for step in range(1, budget + 1):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y)
        loss.backward()
        opt.step()
        qn = q.detach().numpy().copy()
        if step % log_every == 0:
            log.append((step, qn))
        pl, _ = placed(qn, act)
        if pl:
            _, v, _ = unpack(qn)
            cross = {"step": step, "R2": float(np.abs(v).sum() * gamma2_hat / 2), "q": qn}
            break
        if step in ck_steps:
            ck[step] = {"q": qn, "adam": {k: (v_.clone() if torch.is_tensor(v_) else v_)
                                          for k, v_ in opt.state[q].items()}}
    return {"seed": seed, "cross": cross, "log": log, "checkpoints": ck}


def dense_crossing_check(q, act, n=20001):
    """Independent every-step check (validity): the dense-grid G₊ at a much finer grid."""
    th, v, _ = unpack(q)
    xi = np.linspace(*INNER, n)
    xo = np.concatenate([np.linspace(*OUTER[0], n // 2), np.linspace(*OUTER[1], n // 2)])
    return float(phi(xo, th, v, act).min() - phi(xi, th, v, act).max()) > 0


def log_spaced_steps(budget=BUDGET, n=60):
    return sorted(set(np.unique(np.round(np.logspace(0, math.log10(budget), n)).astype(int)).tolist()))


# ------------------------------------------------------------------------------------------ W4 replays
def rescale(q, k):
    """(w₂, b) → k(w₂, b), k > 0: every decision sign(N(x)) is unchanged."""
    q = np.array(q, float)
    q[2] *= k; q[5] *= k; q[6] *= k
    return q


def decisions_preserved(q, k, x, act):
    X = torch.tensor(x, dtype=torch.float64)
    a = torch.sign(logits(torch.tensor(q), X, act))
    b = torch.sign(logits(torch.tensor(rescale(q, k)), X, act))
    return bool(torch.equal(a, b))


def _project(q, radius):
    """Radial projection of v = (q₂, q₅) onto the ℓ₁ sphere of the held radius (direction and signs kept)."""
    with torch.no_grad():
        n1 = q[2].abs() + q[5].abs()
        q[2] *= radius / n1; q[5] *= radius / n1


def replay(ck, k, steps, x, y, act, gamma2_hat, variant="preserved", project=True, record=(4000, 16000, 64000)):
    """Train (θ, v, b) from a rescaled checkpoint with ‖v‖₁ held fixed by projection after every Adam step.
    Returns the correctness at the recorded steps and the maximum drift |‖v‖₁ − held|."""
    q0 = rescale(ck["q"], k)
    radius = abs(q0[2]) + abs(q0[5])
    q = torch.tensor(q0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=LR)
    if variant == "preserved":
        st = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in ck["adam"].items()}
        opt.state[q] = st
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    out, drift = {}, 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        if project:
            _project(q, radius)
        drift = max(drift, abs(float(q.detach()[2].abs() + q.detach()[5].abs()) - radius))
        if step in record:
            th, v, b = unpack(q.detach().numpy())
            out[step] = sign_correct(th, v, b, act)
    return {"correct": out, "drift": drift, "radius": radius, "q_end": q.detach().numpy().copy()}


def reference_true_freeze(ck, steps, x, y, act, reset=True):
    """Independent reference for the k = 1 check: a hand-written Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e−8, continuing
    the checkpoint's moments and step count) that resets v onto the held ℓ₁ sphere after every step.  With
    reset=False it only zeroes nothing and lets v move -- the constructed fail case (the width-1 amendment-2
    error was a reference that did not truly freeze)."""
    q = torch.tensor(ck["q"], dtype=torch.float64, requires_grad=True)
    radius = float(abs(ck["q"][2]) + abs(ck["q"][5]))
    m = ck["adam"]["exp_avg"].clone(); v2 = ck["adam"]["exp_avg_sq"].clone()
    t = int(ck["adam"]["step"])
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    traj = []
    for _ in range(steps):
        if q.grad is not None:
            q.grad = None
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        g = q.grad.detach()
        t += 1
        m = 0.9 * m + 0.1 * g
        v2 = 0.999 * v2 + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v2 / (1 - 0.999 ** t)
        with torch.no_grad():
            q -= LR * mh / (vh.sqrt() + 1e-8)
            if reset:
                n1 = q[2].abs() + q[5].abs()
                q[2] *= radius / n1; q[5] *= radius / n1
        traj.append(q.detach().numpy().copy())
    return traj


def replay_trajectory(ck, steps, x, y, act):
    """The replay code path at k = 1 (preserved moments), returning its trajectory for the k = 1 check."""
    q = torch.tensor(rescale(ck["q"], 1.0), dtype=torch.float64, requires_grad=True)
    radius = float(q.detach()[2].abs() + q.detach()[5].abs())
    opt = torch.optim.Adam([q], lr=LR)
    opt.state[q] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in ck["adam"].items()}
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        _project(q, radius)
        traj.append(q.detach().numpy().copy())
    return traj


def k1_check(ck, x, y, act, steps=25, tol=1e-10, reference=None):
    """§5: at k = 1 the replay reproduces the independent true-freeze reference to tol over its first steps."""
    ref = reference if reference is not None else reference_true_freeze(ck, steps, x, y, act)
    rep = replay_trajectory(ck, steps, x, y, act)
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, rep))
    return diff <= tol, diff


def freeze_ok(drift, tol=1e-12):
    return drift <= tol


def determinism_ok(seed, act, gamma2_hat, budget=500):
    a = train(seed, act, gamma2_hat, budget=budget)
    b = train(seed, act, gamma2_hat, budget=budget)
    return all(np.array_equal(p[1], q[1]) for p, q in zip(a["log"], b["log"])) and \
        ((a["cross"] is None) == (b["cross"] is None))


# ------------------------------------------------------------------------------------------ scorers (registered)
def _boot_median_ci(v, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    meds = np.median(rng.choice(v, (n, len(v))), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def score_w1(cross_R2, R2_glob, upper=1.25):
    """W1: ≥ 90% of crossing runs have R₂ ≥ R₂,glob; the bootstrap 95% interval of median(R₂/R₂,glob) − 1 lies
    above 0; and median(R₂/R₂,glob) <= 1.25."""
    r = np.asarray(cross_R2, float) / R2_glob
    frac = float((r >= 1).mean()); med = float(np.median(r)); lo, hi = _boot_median_ci(r - 1)
    return {"frac_above": frac, "median_ratio": med, "ci": (lo, hi),
            "pass": frac >= 0.90 and lo > 0 and med <= upper}


def score_w2(own_over_pop, cross_R2, own_R2, n_perm=100_000, seed=0):
    """W2a: IQR(own/pop) ≥ 2%.  W2b: Spearman(crossing R₂, own R₂) > 0, one-sided permutation p < 0.05.
    W2c: median(own/pop) − 1 with its interval, no sign registered."""
    r = np.asarray(own_over_pop, float)
    iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
    rho = _spearman(cross_R2, own_R2)
    rng = np.random.default_rng(seed)
    b = np.asarray(own_R2, float)
    perm = np.array([_spearman(cross_R2, rng.permutation(b)) for _ in range(n_perm)])
    p = float((1 + (perm >= rho).sum()) / (1 + n_perm))
    return {"W2a_iqr": iqr, "W2a_pass": iqr >= 0.02, "W2b_rho": rho, "W2b_p": p, "W2b_pass": rho > 0 and p < 0.05,
            "W2c_median_minus_1": float(np.median(r) - 1), "W2c_ci": _boot_median_ci(r - 1)}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def mcnemar_one_sided(b, c):
    """One-sided exact McNemar p for 'the lower level is significantly higher' (b = discordant favouring lower)."""
    n = b + c
    return float(sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n) if n else 1.0


def score_w4(levels, correct, own_over_pop_median, agree_start):
    """W4a: correct fraction at 64k rises with held R₂ (no significant decrease between any lower/higher level,
    paired exact McNemar, one-sided p < 0.05) and its 50% point lies within ±0.05 of median(own/pop).
    W4b: agreement of 'held R₂ above the replay's own threshold on its starting branch' with correctness >= 0.90.
    `correct`: array (replays × levels) of 0/1, paired across levels."""
    C = np.asarray(correct, int)
    frac = C.mean(axis=0)
    viol = 0
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            b = int(((C[:, i] == 1) & (C[:, j] == 0)).sum()); c = int(((C[:, i] == 0) & (C[:, j] == 1)).sum())
            viol += int(mcnemar_one_sided(b, c) < 0.05)
    x50 = _crossing(np.asarray(levels, float), frac)
    agree = float(np.mean(agree_start))
    return {"frac": frac.tolist(), "violations": viol, "x50": x50,
            "W4a_pass": viol == 0 and x50 is not None and abs(x50 - own_over_pop_median) <= 0.05,
            "W4b_agreement": agree, "W4b_pass": agree >= 0.90}


def _crossing(levels, frac, target=0.5):
    for i in range(len(levels) - 1):
        if (frac[i] - target) * (frac[i + 1] - target) <= 0 and frac[i] != frac[i + 1]:
            return float(levels[i] + (target - frac[i]) * (levels[i + 1] - levels[i]) / (frac[i + 1] - frac[i]))
    return None


def tanh_verdict(fa_pass, tanh_pass, tanh_applicable):
    """§4: H-general iff W1 and W4 pass for tanh and both f_a arms; H-nonmonotone iff both f_a pass and tanh fails;
    neither if an f_a arm fails; tanh 'not applicable' (threshold undefined) decides nothing."""
    if not all(fa_pass):
        return "neither (an f_a arm failed W1 or W4)"
    if not tanh_applicable:
        return "not decided (tanh not applicable)"
    return "H-general" if tanh_pass else "H-nonmonotone"
