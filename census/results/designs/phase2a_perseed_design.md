Approved by the author 2026-10-05 (decisions 1–9 as recommended; 400 seeds). NOT a registration: no registered or
pilot seed drawn, no pilot run. Detail moves to the registration text. Exploratory numbers: `phase2a_explore/p2a_explore_h`
(seeds 2,930,001–2,930,111 only). ‡ = set after exploratory data.

# Design 2A-PS: the fold prediction with a sample per seed

**Question.** 2A passed on ONE landscape (7b15d9f). Here each seed has its own 800-point sample, M, fold, constants and
causal forecast: **seeds are the replicates.** Otherwise as 2A (scale-only rule, η = 1, crossing, three units).

**Untraceable fraction (reported prominently in the results).** In exploration M could not be traced on **9 of 25**
grid seeds (36%). Untraceable seeds, and seeds without s\*, are counted and never scored.

**Exploration.** Traced seeds: s_F 4.55–6.04; s_F/s\* 1.06–1.43; Λ_F 5.5e−5–1.7e−3 (2A 1.51e−3). Past the fold,
5 seeds land cleanly on S, 6 never do (5 of 6 did not cross), 5 are mixed. Clean seeds: exponent 0.688/0.689;
r_obs/r_fc 1.20–2.14. The seed's own fold beat the fixed-dataset fold on 9 of 11 crossing seeds.

**Rules**
1. **Sample ‡.** i.i.d. levels on the product grid (`default_rng([seed, 7])`), 800 points.
2. **Release ‡.** s₀ = lo + 0.1776·(s_F − lo) on the seed's own stable M (0.1776 maps 2A's 1.7957); eligible iff
   s₀ ≤ 0.6·s_F. Released at the exact M(s₀); no BFGS hold.
3. **Branch identity.** Data homotopy at s ≈ 4.49 from the fixed dataset (50 λ steps, 200 on failure). Accepted iff
   Newton converges and reduced λ_min > 0 at every λ, the branch has an upper fold, and ρ₂ < q on its stable part.
4. **Per-seed constants.** s_F by validated continuation (h, h/2 agree; λ_min → 0; no perturbed minimum on M above
   it); s\* = equal loss of own M and S; |m′c′| from two windows agreeing ≤ 5%.
5. **Class ‡.** From 30 perturbed minima at 1.01·s_F with ρ₂ ≥ q: clean ≥ 27, none 0, mixed 1–26.
6. **Rates ‡.** Clean seeds at 2⁻¹⁶ (scored) and 2⁻¹⁴ (per-seed exponent); none and mixed at 2⁻¹⁴.
7. **Forecast.** f = 0.95·s_F(seed); `causal_forecast_fold` unchanged, leading order (Ω₀ε^{2/3}); budget ⌈64/ρ⌉ − t_c.
   Competitor: the same forecaster at the same t_c with the fixed-dataset fold (4.7677, 1.50995e−3). 2A's fitted
   correction is shown descriptively only, labelled POST HOC.

**Criteria** (clean seeds unless stated; 10,000 bootstrap resamples over seeds). PASS needs all except C1 and C2.

| | PASS if |
|---|---|
| F | s_F ≤ s_obs ≤ 1.25·s_F at ≥ 90%; falsifier: a crossing below s_F and ≤ 1.25·s\* |
| H | none seeds: no crossing by 1.25·s_F at ≥ 80% |
| E_seed | median of Δln r_obs/Δln ε̂_F (2⁻¹⁴→2⁻¹⁶): 95% CI inside [0.55, 0.80] |
| C3 | r_obs/r_fc > 1 at ≥ 90% |
| C4 | upper end of mean(\|t_fc − t_obs\| − \|t̂_F − t_obs\|) < 0 |
| P | upper end of mean(\|ln ŝ_c/s_obs\| − \|ln ŝ_c^fixed/s_obs\|) < 0 |
| C1, C2 | τ₁ = 0.15, τ₂ = 0.25 at ≥ 80%. Secondary; **registered expected FAIL ‡** (1/5 in exploration) |

**Validity.** At ≥ 90% of clean seeds, the follow check at 0.8·s_F lands on the seed's own M and t_c < t_obs. The idle
unit and the signs stay fixed up to the crossing.

**Gate and power ‡.** 400 seeds, all frozen. Gate: ≥ 24 clean eligible seeds. Exploration had 5/25 (Wilson low 0.089),
so P(gate) = 0.987 at 0.089, and 1.00 at 0.12 and above. Trained: the first 40 clean, 30 none and 20 mixed seeds in seed
order; the rest are counted. P's power is ≥ 0.98 at n = 10, which is optimistic because it rests on 11 values.

**Compute (one worker, nice 15).** Freeze 400 × 13 s ≈ 87 min. Clean runs: 40 × (80 + 320) s ≈ 4.4 h. None and
mixed runs: 50 × 80 s ≈ 67 min. Pilot ≈ 15 min. Total ≈ 7.2 h; peak RSS ≈ 0.8 GB; output < 50 MB.

**Disclosures**
- **Grid levels over i.i.d. points:** with i.i.d.-point resampling S was untraceable on 7 of 9 seeds (no s\*).
- **A MECHANISM test:** each run is released at its seed's exact M point. This is not training from initialisation.
- **Classes defined after exploration (‡):** the classification rule and EVERY seed's class are frozen and hashed
  (with the OpenTimestamped registration) before any training.
- **Untraceable fraction:** reported prominently in the results, with counts of seeds lacking s\* and of ineligible seeds.
- **Falsifier:** read as stated in F; it does not fire on a crossing at or above s_F (seeds with s_F/s\* < 1.25).
