"""Harsh-review Phase A, items A1 and A2 (for the submission; author's request 2026-09-25).  Writes
results/wp12_mechanism.csv (A1: the width-1 mechanism with explicit constants) and results/wp12_scaling.csv (A2: the
scaling law and Adam's per-step bound), and results/wp12_identity.csv (the class-mean identity checks).
Proposition and proof: results/math_note_v2.md §11.

A1.  f_a(t) = t + a sin t on I = [−0.8, 0.8], O = ±[1.2, 2.0]; φ = sign(w₂)·f_a(w₁x + b₁), s = |w₂|.
  Identity:  Δμ(w₁, b₁) = sign(w₂)·a·sin b₁·D(w₁),  D(α) = E_O cos αx − E_I cos αx   (data points; continuous windows).
  Domain:    G(θ) > 0 or G_n(θ) > 0  ⟹  |w₁| < a/1.4       (endpoint pair −2.0, 0.8 and −0.8, 2.0; all four are data points)
  Small s:   L*(θ; s) ≥ log 2 − (s/4)Δμ(θ)  for every θ, s  (convexity of ℓ, tangent at 0, balanced classes)
             L*(θ*; s) ≤ log 2 − (s/4)Δμ(θ*) + (s²/8)Var(φ*)   (ℓ″ ≤ 1/4, b = −s·mean φ*)
             so for s < s₀ = 2a(|D*| − D_P)/Var(φ*), with D_P = max_{|α| ≤ a/1.4}|D(α)|, every placed θ has a loss
             above the unplaced θ* = (α*, b*): no placed parameter is within that margin of the infimum.
  Large s:   every θ with G_n(θ) ≤ η has L* ≥ (log 2/n)·e^{−sη/2};  the Ĝ witness has L* < e^{−sĜ/2}.  So for
             s > s₁ = 2·log(n/log 2)/(Ĝ − η) every θ with G_n ≤ η loses to the witness.  η = 0: data placement.
             η = (a/1.4)(1 + a)(h_I + h_O): continuous placement (G ≥ G_n − Lip(φ)(h_I + h_O), Lip ≤ (a/1.4)(1 + a)
             on the domain where G_n > 0 is possible).
  Hence the switch lies in [s₀, s₁] -- analytic given four finite evaluations (D*, D_P, Var(φ*), Ĝ's witness).

A2.  |w₂|_glob(a) ≈ A*·ε^{−3/2}·(1 + (A′(0)/A*)·ε), ε = a − 1 (math note §8).  Adam (β₁, β₂) = (0.9, 0.999),
  lr = 0.01, bias-corrected: |Δθᵢ| ≤ lr·B_t with B_t = (1−β₁)/(1−β₁ᵗ)·√((1−β₂ᵗ)/(1−β₂))·√((1−γᵗ)/(1−γ)),
  γ = β₁²/β₂ (Cauchy–Schwarz on m_t against v_t; eps only lowers it).  Worst case over every gradient sequence.

    python -m src.harsh_review_a
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .width2_conditional import population

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "results"
A_LIST = (1.02, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 2.0)
PAIR = 2.8                                   # (−2.0, 0.8) and (−0.8, 2.0)
H_I, H_O = 0.8 / 399, 0.4 / 199              # half-spacings of the population grid (window point to nearest data point)
LR, B1, B2, STEPS = 1e-2, 0.9, 0.999, 2000
W2_INIT_MAX = 1.0                            # θ ~ U(−1, 1)


def _data():
    x, y = population()
    return x, y, x[y == 1], x[y == 0]


def D_n(alpha, xO, xI):
    a = np.atleast_1d(np.asarray(alpha, float))[:, None]
    return np.cos(a * xO[None, :]).mean(axis=1) - np.cos(a * xI[None, :]).mean(axis=1)


def D_c(alpha):
    """Continuous windows (uniform densities), closed form."""
    a = np.atleast_1d(np.asarray(alpha, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        eI = np.where(a == 0, 1.0, np.sin(0.8 * a) / (0.8 * a))
        eO = np.where(a == 0, 1.0, (np.sin(2.0 * a) - np.sin(1.2 * a)) / (0.8 * a))
    return eO - eI


def _argmax_abs(f, lo=1e-3, hi=50.0, step=1e-3):
    grid = np.arange(lo, hi, step)
    v = np.abs(np.concatenate([f(grid[i:i + 5000]) for i in range(0, len(grid), 5000)]))
    k = int(np.argmax(v))
    l, h = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    F = lambda t: float(np.abs(f(t))[0])
    c, d = h - g * (h - l), l + g * (h - l)
    for _ in range(80):
        if F(c) > F(d):
            h = d
        else:
            l = c
        c, d = h - g * (h - l), l + g * (h - l)
    t = 0.5 * (l + h)
    return t, F(t)


def sup_abs_on(f, amax, lip, h=1e-5):
    """Upper bound of max_{0 ≤ α ≤ amax}|f| (f even in α): grid max + Lipschitz step (float, allowance 1e−12)."""
    grid = np.arange(0.0, amax + h, h)
    grid[-1] = amax
    v = np.abs(np.concatenate([f(grid[i:i + 20000]) for i in range(0, len(grid), 20000)]))
    return float(v.max() + lip * h / 2 + 1e-12)


def identity(n_samples=2000, seed=0):
    """Δμ direct vs a·sin b·D(w₁): on the data points and on the continuous windows (Gauss–Legendre, 200 nodes per
    window, against the closed form)."""
    rng = np.random.default_rng(seed)
    x, y, xO, xI = _data()
    gl, gw = np.polynomial.legendre.leggauss(200)
    quad = lambda f, lo, hi: 0.5 * (hi - lo) * (gw * f(0.5 * (hi - lo) * gl + 0.5 * (hi + lo))).sum()
    rows = []
    for a in (1.3, 1.5):
        W = rng.uniform(-10, 10, n_samples); B = rng.uniform(0, 2 * math.pi, n_samples)
        err_n = err_c = 0.0
        for w, b in zip(W, B):
            f = lambda t: t + a * np.sin(t)
            dmu_n = f(w * xO + b).mean() - f(w * xI + b).mean()
            err_n = max(err_n, abs(dmu_n - a * math.sin(b) * float(D_n(w, xO, xI)[0])))
            eO = 0.5 * (quad(lambda t: f(w * t + b), -2.0, -1.2) + quad(lambda t: f(w * t + b), 1.2, 2.0)) / 0.8
            eI = quad(lambda t: f(w * t + b), -0.8, 0.8) / 1.6
            err_c = max(err_c, abs((eO - eI) - a * math.sin(b) * float(D_c(w)[0])))
        rows.append({"a": a, "samples": n_samples, "w1_range": "[-10, 10]", "max_abs_err_data": err_n,
                     "max_abs_err_continuous": err_c})
    return pd.DataFrame(rows)


def _profiled(z0, y):
    from .width2_conditional import _softplus, profile_b
    z = z0 + profile_b(z0, y)
    return float((_softplus(z) - y * z).mean())


def mechanism():
    from .ghat_rigorous import ghat_R
    from .width2_geometry import Act, gaps
    x, y, xO, xI = _data()
    n = len(x)
    al_n, Dn_star = _argmax_abs(lambda t: D_n(t, xO, xI))
    al_c, Dc_star = _argmax_abs(D_c)
    lip_n = float(np.abs(xO).mean() + np.abs(xI).mean())            # |D′| ≤ E_O|x| + E_I|x|
    lip_c = 1.6 + 0.4
    sgn = float(np.sign(D_n(al_n, xO, xI)[0]))
    b_star = sgn * math.pi / 2                                        # sin b* · D* = |D*|
    gh = ghat_R()
    brk = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    brk = {round(r.a, 2): (r.w2_lo, r.w2_hi) for r in brk.itertuples()}
    rows = []
    for a in A_LIST:
        f = lambda t: t + a * np.sin(t)
        phis = f(al_n * x + b_star)
        dmu = float(phis[y == 1].mean() - phis[y == 0].mean())
        var = float(phis.var())
        DP_n = sup_abs_on(lambda t: D_n(t, xO, xI), a / 1.4, lip_n)
        DP_c = sup_abs_on(D_c, a / 1.4, lip_c)
        s0 = 2 * (dmu - a * DP_n) / var if dmu > a * DP_n else float("nan")
        # sanity: the two bounds at s0 (the proof's inequality, evaluated)
        L_star_at_s0 = _profiled(s0 * phis, y) if np.isfinite(s0) else float("nan")
        g = gaps(np.array([al_n, b_star, 0.0, 0.0]), np.array([1.0, 0.0]), Act("fa", a))
        Gc = max(g["G+"][1], g["G-"][1])                              # upper end: unplaced if ≤ 0
        Gn = max(phis[y == 1].min() - phis[y == 0].max(), phis[y == 0].min() - phis[y == 1].max())
        G_hat = gh[round(a, 2)]
        eta = (a / 1.4) * (1 + a) * (H_I + H_O)
        s1_data = 2 * math.log(n / math.log(2)) / G_hat
        s1_cont = 2 * math.log(n / math.log(2)) / (G_hat - eta) if G_hat > eta else float("nan")
        w2c = brk.get(round(a, 2), (np.nan, np.nan))
        rows.append({"a": a, "alpha_star_data": al_n, "D_star_data": Dn_star, "alpha_star_cont": al_c,
                     "D_star_cont": Dc_star, "a_over_1.4": a / 1.4, "a_limit_1.4_alpha_star": 1.4 * al_n,
                     "D_P_data": DP_n, "D_P_cont": DP_c, "dmu_star": dmu, "var_phi_star": var, "s0": s0,
                     "R0": s0 * G_hat / 2, "bound_check_Lstar_at_s0": L_star_at_s0,
                     "bound_check_lower_placed": math.log(2) - s0 / 4 * a * DP_n if np.isfinite(s0) else np.nan,
                     "pair_bound_G": 2 * a - PAIR * al_n, "pair_bound_G_sharp": 2 * a * abs(math.sin(1.4 * al_n)) - PAIR * al_n,
                     "G_cont_hi_at_theta_star": Gc, "G_n_at_theta_star": float(Gn), "Ghat_rigorous": G_hat, "eta": eta,
                     "s1_data": s1_data, "s1_cont": s1_cont, "R1_data": s1_data * G_hat / 2,
                     "w2_glob_cert_lo": w2c[0], "w2_glob_cert_hi": w2c[1],
                     "bracket_contains_cert": bool(np.isnan(w2c[0]) or (s0 < w2c[0] and w2c[1] < s1_data))})
    return pd.DataFrame(rows)


def adam_bound(steps=STEPS, b1=B1, b2=B2):
    t = np.arange(1, steps + 1, dtype=float)
    gam = b1 ** 2 / b2
    B = (1 - b1) / (1 - b1 ** t) * np.sqrt((1 - b2 ** t) / (1 - b2)) * np.sqrt((1 - gam ** t) / (1 - gam))
    Binf = (1 - b1) / math.sqrt(1 - b2) / math.sqrt(1 - gam)
    return t, B, Binf


def adam_worst_case_check(b1=B1, b2=B2, t=400):
    """The bound is attained (Cauchy–Schwarz equality): g_k ∝ (β₁/β₂)^{t−k} gives |m̂_t|/√v̂_t = B_t exactly."""
    k = np.arange(1, t + 1)
    g = (b1 / b2) ** (t - k)
    m = (1 - b1) * (b1 ** (t - k) * g).sum() / (1 - b1 ** t)
    v = (1 - b2) * (b2 ** (t - k) * g ** 2).sum() / (1 - b2 ** t)
    _, B, _ = adam_bound(t)
    return float(m / math.sqrt(v)), float(B[-1])


def scaling():
    c1 = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    A_lo, A_hi = 0.68125, 0.6875                                      # certified bracket (limit_bnb; independently checked)
    A_k = 0.5 * (c1.A_star_lo + c1.A_star_hi)                         # Krawczyk A*
    r1 = 0.5 * (c1.A1_over_A_lo + c1.A1_over_A_hi)                    # A′(0)/A*
    brk = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    brk = {round(r.a, 2): (r.w2_lo, r.w2_hi) for r in brk.itertuples()}
    t, B, Binf = adam_bound(200_000)
    cum = 1 + LR * np.cumsum(B)                                       # max reachable |w₂| after T steps (|w₂(0)| ≤ 1)
    rows = []
    for a in A_LIST:
        e = a - 1
        w0 = A_k * e ** -1.5
        w1_ = w0 * (1 + r1 * e)
        lo, hi = brk.get(round(a, 2), (np.nan, np.nan))
        Nmin = int(np.searchsorted(cum, w1_) + 1) if w1_ <= cum[-1] else None
        rows.append({"a": a, "eps": e, "w2_limit_lo": A_lo * e ** -1.5, "w2_limit_hi": A_hi * e ** -1.5,
                     "w2_limit_krawczyk": w0, "w2_first_order": w1_, "w2_cert_lo": lo, "w2_cert_hi": hi,
                     "rel_err_first_order_vs_cert_mid": (w1_ / (0.5 * (lo + hi)) - 1) if np.isfinite(lo) else np.nan,
                     "N_min_adam_worst_case": Nmin, "N_lr_per_step": math.ceil((w1_ - W2_INIT_MAX) / LR),
                     "reachable_w2_at_2000_worst_case": float(cum[STEPS - 1]),
                     "unreachable_at_2000": bool(w1_ > cum[STEPS - 1])})
    d = pd.DataFrame(rows)
    # the smallest a whose predicted threshold is reachable within B steps, for the budget-law budgets
    grid = np.linspace(1.0005, 2.0, 200_000)
    wg = A_k * (grid - 1) ** -1.5 * (1 + r1 * (grid - 1))
    onset = []
    for Bud, obs in ((2000, 1.60), (8000, 1.18), (32000, 1.06), (128000, 1.03)):
        reach = float(cum[Bud - 1])
        a_min = float(grid[np.argmax(wg <= reach)])
        a_typ = float(grid[np.argmax(wg <= W2_INIT_MAX + LR * Bud)])
        onset.append({"budget": Bud, "reachable_w2_worst_case": reach, "a_min_worst_case": a_min,
                      "reachable_w2_lr_per_step": W2_INIT_MAX + LR * Bud, "a_min_lr_per_step": a_typ,
                      "observed_onset": obs})
    sw = pd.read_csv(RESULTS / "fold1d_sweep.csv").query("activation == 'sin_family' and parameter == 1.02")
    return d, pd.DataFrame(onset), {"B_inf": Binf, "B_max_2000": float(B[:STEPS].max()),
                                    "reachable_2000": float(cum[STEPS - 1]),
                                    "a102_runs": len(sw), "a102_solved": int(sw.solved.sum()),
                                    "a102_median_terminal_w2": float(sw.w2_abs.median()),
                                    "a102_max_terminal_w2": float(sw.w2_abs.max())}


def main():
    idn = identity()
    idn.to_csv(RESULTS / "wp12_identity.csv", index=False)
    m = mechanism()
    m.to_csv(RESULTS / "wp12_mechanism.csv", index=False)
    d, onset, info = scaling()
    ach, bt = adam_worst_case_check()
    info.update({"worst_case_attained_ratio": ach, "B_t_400": bt})
    d.to_csv(RESULTS / "wp12_scaling.csv", index=False)
    onset.to_csv(RESULTS / "wp12_onset_bound.csv", index=False)
    pd.DataFrame([info]).to_csv(RESULTS / "wp12_adam_bound.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(idn.to_string(index=False)); print(m.T.to_string()); print(d.to_string(index=False))
    print(onset.to_string(index=False)); print(info)


if __name__ == "__main__":
    main()
