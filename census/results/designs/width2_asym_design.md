DRAFT for the author's approval; nothing registered, trained or piloted. (expl.) = exploratory, population or used T2-3
seeds 600,000–600,011, from `width2_asym_explore/`; recomputed and frozen at registration.

# Design W2-A: the lag law at width 2 on the asymmetric windows, tracking regime

**Setting.** a = 1.30, Δ = 0.4 (as T2-1–T2-3); own sample `asym_register.training_set(seed)`; float64; s = ‖v‖₁; exact G₊
every step. Seeds 884,000–884,119, both arms; pilot 884,900–884,909 (verified unused).

**Branches at s₀ = 0.5·s_pop2 = 0.2225** (T2-1: 0.44508; expl.). With v held at equal shares, random hidden starts reach
an unplaced **duplicate** branch D (units equal mod 2π: the width-1 function) in 72% (own samples pooled 63%, 13–83%), a
two-unit branch *already placed* in 27%, a duplicate D′ in 1%. With v = s₀·(0.1, 0.9): an unplaced **two-unit** branch T
in 59%, T′ 8%, D 33%. At s₀ = 2.54 (half D's switch, the GELU-T analogue) all are placed. Symmetries: sign flip per unit
(fixed by v > 0); unit swap (no copy: D is invariant, T's v₀ is not); a 2π winding per unit (β_i + 2πk_i, b − 2πk·v):
same switch, different κ (D: 7.67, 0.18, −7.31 at k₁ + k₂ = 0, 1, 2). Per seed: copies D, D′, T, T′, seven κ_k each.

**Arms** (separate verdicts). v held at s₀·(½, ½) (**D**) or s₀·(0.1, 0.9) (**T**); hidden = coordinates 0, 1, 3, 4, 6 of
the seed's U(−1, 1)⁷ draw, GD at lr 1.0 for W = max(4000, ⌈25/λ_min,own(s₀)⌉). Then full-batch SGD, η hidden, ρη on v:
D η = 0.3, 40,000 steps; T η = 0.03, 100,000·2⁻¹⁰/ρ steps.

**Gate.** Newton at v₀ from the release state: on copy c if within 1e−6 of frozen θ\*_own,c(s₀) up to windings (state
within 1e−3). Target D (T); D′ (T′) scored like GELU-T's mirror; other runs counted, never replaced. Gate ≥ 60/120 on a
scored copy (P ≈ 0.998 at pooled own rates; 80% would give < 1e−4).

**Frozen (hashed, OpenTimestamps), per seed and copy:** θ\*_own(s₀), λ_min, W; s_switch,branch by validated continuation
(step halving, no turning point, reduced λ_min > 0; D's split block PD) along the diagonal (D) or the adiabatic path,
the reduced gradient flow of v (T); κ_k, k₁ + k₂ = −3..3; follow points at 0.8·s_switch; s_pop,D, s_pop,T; ρ; code.

**κ.** Hidden-layer Hessian at fixed v, P = I, D's split directions (δθ₁ = v₂e, δθ₂ = −v₁e; first-order neutral,
decoupled, PD ≥ 0.15) excluded. κ = λ_min[∇G·H⁻¹θ\*′]/[dG/ds], dG/ds = ∇_zG·θ\*′ + ∂_vG·v′ (v-term 0 for D). Signed.

**Predictions.** r_traj: R4 along the run's own v_t; r_cf = κ_kχ at t_sw; s_cross = s_switch(1 + κ_kχ). D's G depends on
s only, so its frozen switch holds on any v path. T's does not: r_cf and L5 use the lag-free switch on the run's own v
path (v only, before any gap is read); expl. it differs from the adiabatic one by the lag itself.

**Feasibility (population, expl.).** D, η 0.3, ρ 1: lag 203 steps (slaved 200), r_obs 0.0313 vs κχ 0.0311, window max χ_t
0.006 (η = 1: no tracking). T: ρ = 1 jumps to s = 0.48 in one step; η 0.03, ρ 0.001: lag 28 steps (28.1), r_obs 0.00128 vs
κχ 0.00129, κ 0.048; switch 0.482 on the adiabatic path (own 0.42–0.54; fixed ray 1.92), where T2-3's occupied branches
switched (median 0.486). T′ crossed 0.5–4% before its switch at ρ ≤ 0.01 (not understood).

**Pilot rule** (to t_sw). D: GELU-T's. T: from ρ = 2⁻¹⁰, halve (≤ 3, then STOP) until q90 of window max χ_t ≤ 0.1 (κ ≈ 0.05
makes κχ ≤ 0.1 too loose).

**Criteria, per arm:** GELU-T's L1–L5; L4 vs the same branch's population switch (D 5.080, T 0.482; T2-1's 0.445
descriptive). **Validity:** GELU-T's V1–V7; V1 ≥ 60/120; the follow check also needs D still duplicate.

**Readings.** PASS D: the law holds on the branch T2-3b–d's slow runs followed. PASS T: T2-3's branch type is crossed at
its switch plus the predicted lag; T2-3's 2.8× was speed. Follow fails: runs leave D for the placed branch, or T splits.
L1/L2 fail: T2-3-like lag. L5 fails: lag unresolved (T's ≈ 0.1%). Compute ≈ 2 h, one worker (rough).

**Needs your decision.** (1) Both arms; headline. (2) s₀. (3) T's shares. (4) Gate 60/120, 120 seeds, V1. (5) Score D′,
T′? (6) T's own-path switch (not frozen). (7) κ: dG/ds, split block. (8) T's pilot rule, η_T. (9) L4 comparators. (10)
Winding table. (11) G > 0 in the hold: scored, flagged? Unconverged holds (up to 40% of one own sample) = neither. (12)
Budgets. (13) All other GELU-T registration details carried over.
