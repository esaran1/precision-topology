Approved by the author 2026-09-29 as recommended, with changes A–D; **NEW** = not yet approved. Draft and background:
ccce4fa (background moves to the registration). Exploratory numbers: `width2_asym_explore/`.

# Design W2-A: the lag law at width 2 on the asymmetric windows

**Setting.** a = 1.30, Δ = 0.4; own sample `asym_register.training_set(seed)`; float64; s = ‖v‖₁; exact G₊ every step;
s₀ = 0.5·s_pop2 = 0.2225. Hold: v fixed, hidden GD (lr 1.0), W = max(4000, ⌈25/λ_min,own(s₀)⌉) steps; then full-batch
SGD, η hidden, ρη on v.

| arm | v held | hidden start | η | seeds | gate |
|---|---|---|---|---|---|
| **T** (headline) | s₀·(0.1, 0.9) | seed's U(−1, 1)⁷, coords 0, 1, 3, 4, 6 | 0.03 | 884,000–884,119 | ≥ 60/120 on T |
| **D** (control) | s₀·(½, ½) | the same | 0.3 | the same | ≥ 60/120 on D, D′ |
| **T′** (**NEW**) | s₀·(0.1, 0.9) | θ\*_T′,pop(s₀) | 0.03 | 884,200–884,319 | ≥ 108/120 on T′, no G > 0 in hold |

Pilot 884,900–884,909, all arms (all verified unused). Budgets: D 40,000; T, T′ 100,000·2⁻¹⁰/ρ.

**Release.** Newton at v₀: on copy c if within 1e−6 of frozen θ\*_own,c(s₀) up to 2π windings (state within 1e−3);
κ_k by winding (k₁ + k₂ = −3..3). Others (unconverged too) counted, never replaced; G > 0 in a T or D hold: scored,
flagged.

**Frozen (hashed, OpenTimestamps).** Per seed and copy: θ\*_own(s₀), λ_min, W; s_switch by validated continuation
(checks as ccce4fa; D diagonal; T, T′ adiabatic reference); κ_k; follow points; population switches (D 5.080, T 0.482,
T′ 0.328); ρ; code, **including the procedure reading T's and T′'s lag-free switch from the run's own v path (v only,
before any gap is read); their predictions are conditional on that path** (C).

**κ** = λ_min[∇G·H⁻¹θ\*′]/[dG/ds], hidden Hessian at fixed v, P = I, D's split directions excluded,
dG/ds = ∇_zG·θ\*′ + ∂_vG·v′; signed. **Disclosure (B):** this total derivative was adopted after exploratory
population runs gave the wrong sign with GELU-T's ∇_zG·θ\*′ (T: κ ≈ −1.4, lag positive); with it, obs/pred 0.997 (T),
1.005 (D), 1.02 (T′).

**Predictions.** R4 along the run's own v_t; r_cf = κ_kχ at t_sw; s_cross = s_switch(1 + κ_kχ).

**Pilot** (to t_sw). D: GELU-T's. T, T′: from ρ = 2⁻¹⁰ halve (≤ 3, then STOP) until q90 of window max χ_t ≤ 0.1.

**Criteria, validity** (per arm): GELU-T's L1–L5, V1–V7; V1 ≥ 60/120; L4 vs the branch's population switch; L5 vs the
lag-free own switch; D must still be duplicate at the follow check.

**L5 resolution (NEW, D).** N_min = 2(g + τ + d) = 6 steps: g = 1 (first-step detection), τ = 1 (Corollary L3's
discreteness), d = ⌈ε_G/(|dG/ds|·ṡ)⌉ = 1 (one exact enclosure, ε_G = 1e−9, for observation and prediction, so GELU-T's
dense-vs-exact offset, up to 191 steps, cannot arise); r_min = 6·ṡ_sw/s_sw. Runs with |t_pred − t_sw| < 6 are left out
of L5 (counted); L5 is UNRESOLVED if fewer than half the scored runs, or fewer than 2, remain. Exploratory: 200, 28, −48
steps (D, T, T′); r_min 9e−4, 3e−4, 5e−4 vs lags 0.031, 0.0013, −0.0040.

**T′ (NEW, A).** Not T's unit-swap image; random holds reach it in 12–13% at any share 0.02–0.3 (~1,000 seeds needed).
Branch-point start (as GELU-T's control): own copy in 12/12 used seeds, κ −0.026 to −0.065, early crossing as predicted
(−49 vs −47.9 steps).

**Readings.** PASS T: T2-3's branch type crosses at its switch plus the lag (T2-3's 2.8× was speed); T′: the lag's sign
is predicted; D: the law holds on T2-3b–d's branch. Compute ≈ 3 h, one worker.
