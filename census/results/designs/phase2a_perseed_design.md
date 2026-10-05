DRAFT for the author's approval (2026-10-05). NOT a registration: no registered or pilot seed drawn, no pilot run.
Every numbered rule is a PROPOSAL. Exploratory numbers: `phase2a_explore/p2a_explore_g`, `p2a_explore_h`
(exploration seeds 2,930,001–2,930,111 only). ‡ = set after seeing exploratory data.

# Design 2A-PS: the fold prediction with a sample per seed

**Question.** 2A passed on ONE landscape (7b15d9f). Here each seed has its own 800-point sample, M, fold, constants and
causal forecast: **seeds are the replicates.** Everything else is 2A's (scale-only rule, η = 1, crossing, three units).

**Exploration** (25 grid, 12 i.i.d.-point seeds; runs at 2⁻¹³, three also at 2⁻¹⁵).
- M traceable: 16/25 grid, 9/12 i.i.d. s_F 4.55–6.04 (median 5.46); s\* 3.35–5.36; s_F/s\* 1.06–1.43; Λ_F
  5.5e−5–1.7e−3 (2A 1.51e−3). 1.7957 lies on M for 9/16. Fold step-halving agreement ≤ 4.4e−9.
- **H-F5 is seed-dependent.** Minima at 1.01·s_F with ρ₂ ≥ q: 29–30/30 on 5 seeds, 0/30 on 6, 13–22/30 on 5.
  Of the 0/30 seeds, 5 of 6 did not cross by 1.25·s_F (they escape to a low-ρ₂ minimum).
- **Clean seeds** (≥ 27/30): exponent between rates 0.688 and 0.689 (2 seeds). r_obs/r_fc 1.24–2.14 at 2⁻¹³ and
  1.20 and 1.62 at 2⁻¹⁵: the ratio does not fall to 2A's correction on every seed. C1 error/delay_fc 0.15–1.01.
- **Mixed seed** 2,930,022: exponent 0.08. Its crossing is not the fold delay.
- **Own fold vs fixed-dataset fold:** |ln(ŝ_c/s_obs)| was smaller with the seed's own constants at 9 of 11 crossing
  seeds (mean difference −0.098).

**Rules (proposals)**
1. **Sample ‡.** i.i.d. levels on the product grid, from `default_rng([seed, 7])`, 800 points, as explored. The
   i.i.d.-point alternative traced M as often but S on only 2 of 9 (no s\*).
2. **Hold ‡.** s₀ = lo + 0.1776·(s_F − lo), where lo is the lower end of the seed's own stable M; 0.1776 places 2A's
   1.7957 on the fixed M. The seed is eligible only if s₀ ≤ 0.6·s_F (13/16 traced seeds), which leaves room for the
   0.8·s_F follow check. **No BFGS hold:** each eligible seed is released at its exact M(s₀) (as in 2A, where the init
   only gated). An own-init hold landed on M for 4/16.
3. **Branch identity.** A data homotopy at fixed s ≈ 4.49, X(λ) = (1 − λ)X_fixed + λX_seed, Newton in 50 equal λ
   steps (200 if that fails). M is accepted iff Newton converges at every λ, the reduced λ_min > 0 at every λ, the
   branch has an upper fold, and ρ₂ < q on its stable part. Otherwise the seed is untraceable: counted, not scored.
4. **s_F, s\*, Λ_F per seed.**
   - s_F: continuation at h and h/2 agreeing to ≤ 1e−6, λ_min → 0 at the fold, and 0/30 perturbed minima on M at
     1.001·s_F.
   - s\*: equal loss of the seed's M and S (S by the same homotopy). No s\*: not scored.
   - |m′c′|: the 0.05 and 0.025 windows must agree to ≤ 5%. This is not yet explored per seed.
5. **H-F5 as a frozen per-seed PREDICTION ‡.**
   - The class is set before training, from the 30 perturbed minima at 1.01·s_F with ρ₂ ≥ q: clean (≥ 27), none (0),
     mixed (1–26).
   - Clean seeds are scored. None seeds are predicted NOT to cross by 1.25·s_F (criterion H). Mixed seeds are
     descriptive only.
6. **Rates ‡.** Every clean seed runs at 2⁻¹⁶ (F, C1–C4, P) and at 2⁻¹⁴ (for E_seed). None and mixed seeds run at
   2⁻¹⁴. The forecast stays leading order (Ω₀ε^{2/3}).
   - 2A's post hoc correction (C = 0.803, fitted on one landscape) is shown alongside as a descriptive forecast,
     labelled POST HOC-derived. The exploration shows it does not transfer.
7. **Cutoff and forecaster.**
   - f = 0.95 of the seed's s_F. `causal_forecast_fold` is unchanged, with the seed's (s_F, Λ_F).
   - Budget = ⌈64/ρ⌉ − t_c. An unbounded budget drove the exploration to 4.2 GB.
   - The competitor is the same forecaster at the same t_c with the fixed-dataset constants (4.7677, 1.50995e−3).

**Criteria (over clean scored seeds; bootstrap 10,000 over seeds).** PASS requires all of the following except C1
and C2:

| | PASS if |
|---|---|
| F | s_F ≤ s_obs ≤ 1.25·s_F at ≥ 90% of seeds, AND no falsifier: s_obs < s_F with s_obs ≤ 1.25·s\* (this matters on seeds with s_F/s\* near 1.25) |
| H | none seeds: no crossing by 1.25·s_F at ≥ 80% |
| E_seed | median of Δln r_obs/Δln ε̂_F (2⁻¹⁴→2⁻¹⁶) has its 95% CI inside [0.55, 0.80] |
| C3 | r_obs/r_fc > 1 at ≥ 90% of seeds |
| C4 | upper end of mean(\|t_fc − t_obs\| − \|t̂_F − t_obs\|) < 0 |
| P | upper end of mean(\|ln ŝ_c/s_obs\| − \|ln ŝ_c^fixed/s_obs\|) < 0 (the point of per-seed folds) |
| C1, C2 | 2A's τ₁ = 0.15 and τ₂ = 0.25 at ≥ 80% of seeds. **Secondary; registered expectation FAIL ‡** (1/5 clean seeds within 0.15 in the exploration) |

**Validity.** At ≥ 90% of seeds the follow check at 0.8·s_F lands on the seed's own M and t_c < t_obs. The idle
unit and the signs are fixed up to the crossing (clean seeds).

**Gate and power ‡.** 300 registered seeds, all frozen. The gate is ≥ 24 clean eligible seeds. The exploration had
5/25 (Wilson 0.089–0.391), so P(gate) is 1.00 at 0.20, 0.99 at 0.12 and 0.73 at the Wilson low. The first 40 clean
seeds are trained, then the first 30 none seeds and 20 mixed seeds; the rest are counted. Resampling the exploration's
P differences gives power ≥ 0.98 at n = 10. That figure is optimistic: it rests on 11 values.

**Compute (one worker, nice 15).**

| step | time |
|---|---|
| freeze, 300 × 13 s | 65 min |
| clean seeds, 40 × (2⁻¹⁴ ≈ 80 s + 2⁻¹⁶ ≈ 320 s, with ρ₂ every step) | 4.4 h |
| none and mixed seeds, 50 × 80 s | 67 min |
| pilot (3 clean pilot seeds at 2⁻¹⁵·⁵) | ≈ 15 min |
| **total** | **≈ 7 h** |

Peak RSS ≈ 0.8 GB; output < 50 MB. Candidate seeds 2,975,000–2,975,299, pilot 2,975,900–2,975,959 (to be scanned).

**Needs your decision**
1. Grid levels (proposed) or i.i.d. points. The latter needs a better S tracer.
2. The s₀ fraction rule plus the 0.6·s_F cap ‡, and dropping the BFGS hold.
3. H-F5: clean / none / mixed classes, with thresholds 27 and 0 ‡, and none seeds as criterion H.
4. Two rates per clean seed (2⁻¹⁴, 2⁻¹⁶) with E_seed, or one fixed rate (E across seeds only, descriptive).
5. Leading-order forecast, with the 2A correction descriptive only.
6. C1 and C2 kept at 2A's τ as a secondary criterion expected to FAIL, or τ from a pilot (likely ≥ 1), or dropped.
7. The falsifier reading for seeds with s_F/s\* < 1.25.
8. N = 300, gate 24, and the training caps 40 / 30 / 20.
9. Untraceable seeds (36% in the exploration) and seeds without s\*: counted, never scored.
