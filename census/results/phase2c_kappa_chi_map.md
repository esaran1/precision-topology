# Phase 2C: κχ at the crossing in standard training (DESCRIPTIVE)

**DESCRIPTIVE. No registration, no criteria, no verdict.** Producer `src/phase2c_map.py`; data `results/phase2c/kappa_chi_map.json`. Every κχ is read from a committed producer output (cited per cell); nothing was retrained. The one computed quantity is the per-run ε_F of the simplicity-bias runs (`results/phase2c/sb_eps_F.json`).

Question: for standard initialisation and the standard learning rate of each setting, how often does ordinary training lie inside the regime where the linear lag law r = κχ applies? The reference is Test 2B's C1: the law held within ±25% at κχ ≤ 0.1 (all six cells, κχ ∈ {0.02, 0.05, 0.1}); κχ\* = 0.2 at both a.

Definitions (§13.1): κ = λ_min·[∇G·(PH)⁻¹θ\*′]/[∇G·θ\*′], χ = (ṡ/s\*)/(ηλ_min(P^{1/2}HP^{1/2})). Fractions are over the runs with a finite value and use |κχ|. Quantiles are linear-interpolation (numpy default). Signed and |·| summaries differ only where a value is negative; those cells list the negatives.

## Pooled per setting and optimiser (cells marked "not pooled" excluded)

| setting | optimiser | n | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| width-1 sine | Adam | 408 | 4.19e-05 | 0.0321 | 0.0747 | 0.1471 | 0.2209 | 297/408 (73%) | 1/408 (0%) |
| width-1 sine | SGD | 207 | 0.0223 | 0.0287 | 0.0741 | 0.1006 | 0.1376 | 184/207 (89%) | 0/207 (0%) |
| width-1 GELU | Adam | 114 | 3.80e-04 | 9.00e-04 | 0.0021 | 0.0043 | 0.0075 | 114/114 (100%) | 114/114 (100%) |
| band task in R^d | Adam | 705 | -0.0689 | 0.0162 | 0.0296 | 0.0587 | 0.0981 | 705/705 (100%) | 136/705 (19%) |
| width 2 | Adam | 75 | -1.2726 | 0.0279 | 0.2576 | 0.8344 | 28.6255 | 9/75 (12%) | 0/75 (0%) |

Simplicity bias has no pooled κχ: κ is undefined (below).

## width-1 sine

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam a=1.30 (n=6,400, seeds 300000-049) | 50 / 48 | 48 | 0.0237 | 0.0268 | 0.0304 | 0.0328 | 0.0344 | 48/48 (100%) | 0/48 (0%) |
| Adam a=1.50 (n=6,400, seeds 300000-049) | 50 / 48 | 48 | 0.0496 | 0.0547 | 0.0634 | 0.0831 | 0.0925 | 48/48 (100%) | 0/48 (0%) |
| Adam a=1.45 (Task B, n=200, seeds 830000-079) | 80 / 80 | 80 | 0.0342 | 0.0425 | 0.0516 | 0.0635 | 0.0749 | 80/80 (100%) | 0/80 (0%) |
| Adam a=1.60 (Task B, n=200, seeds 830000-079) | 80 / 80 | 80 | 0.0514 | 0.0642 | 0.0819 | 0.1155 | 0.2099 | 68/80 (85%) | 0/80 (0%) |
| Adam a=1.65 (Track A, seeds 1650000-079), P at the crossing | 80 / 76 | 74 | 0.0498 | 0.0701 | 0.0910 | 0.1516 | 0.2193 | 48/74 (65%) | 0/74 (0%) |
| Adam a=1.65 (Track A), registered P at the rule point [sensitivity; not pooled] | 80 / 76 | 74 | 0.0464 | 0.0656 | 0.0887 | 0.1328 | 0.1506 | 47/74 (64%) | 0/74 (0%) |
| Adam a=1.85 (Test 2A, seeds 1850000-079), P at t_sw | 80 / 79 | 78 | 4.19e-05 | 0.1066 | 0.1396 | 0.1840 | 0.2209 | 5/78 (6%) | 1/78 (1%) |
| SGD a=1.30 (n=200, seeds 0-39) | 40 / 30 | 30 | 0.0223 | 0.0238 | 0.0270 | 0.0311 | 0.0328 | 30/30 (100%) | 0/30 (0%) |
| SGD a=1.50 (n=200, seeds 0-39) | 40 / 30 | 30 | 0.0472 | 0.0502 | 0.0572 | 0.0660 | 0.0693 | 30/30 (100%) | 0/30 (0%) |
| SGD a=1.65 (Track A, seeds 1650000-079) | 80 / 64 | 63 | 0.0616 | 0.0715 | 0.0884 | 0.1112 | 0.1376 | 45/63 (71%) | 0/63 (0%) |
| SGD a=1.58 (Phase 1C W1, seeds 9350000-099) | 100 / 84 | 84 | 0.0538 | 0.0601 | 0.0763 | 0.0909 | 0.1208 | 79/84 (94%) | 0/84 (0%) |

Sources and definitions:

- **Adam a=1.30 (n=6,400, seeds 300000-049)** — Adam lr 0.01; `results/lag_law/predictions.csv (set lag1, arm 1.0 = φ = 1, the unmodified reference path; lag2-prim/1.0, lag2-tstar/1.0 and 4b/control are bit-identical duplicates and are not counted again)`, column `pred_r`. κ: population κ_k at the certified switch (lag_law/kappa.csv; Adam P = per-coordinate median of the crossing P over existing runs; winding k measured at the crossing). χ: run's own timescale ratio at the crossing (residual_timescale): d log s/dt over the last ≤ 100 steps / (lr·λ_min(D^{-1/2}HD^{-1/2})), D = √v̂+ε the run's own Adam state; = χ/(1+r), i.e. ṡ divided by s_c not s* (§13.3(i)).
- **Adam a=1.50 (n=6,400, seeds 300000-049)** — Adam lr 0.01; `results/lag_law/predictions.csv (set lag1, arm 1.0 = φ = 1, the unmodified reference path; lag2-prim/1.0, lag2-tstar/1.0 and 4b/control are bit-identical duplicates and are not counted again)`, column `pred_r`. κ: population κ_k at the certified switch (lag_law/kappa.csv; Adam P = per-coordinate median of the crossing P over existing runs; winding k measured at the crossing). χ: run's own timescale ratio at the crossing (residual_timescale): d log s/dt over the last ≤ 100 steps / (lr·λ_min(D^{-1/2}HD^{-1/2})), D = √v̂+ε the run's own Adam state; = χ/(1+r), i.e. ṡ divided by s_c not s* (§13.3(i)).
- **Adam a=1.45 (Task B, n=200, seeds 830000-079)** — Adam lr 0.01; `results/lag_law/predictions.csv (set TaskB)`, column `pred_r`. κ: population κ_k at the certified switch (lag_law/kappa.csv; Adam P = per-coordinate median of the crossing P over existing runs; winding k measured at the crossing). χ: run's own timescale ratio at the crossing (residual_timescale): d log s/dt over the last ≤ 100 steps / (lr·λ_min(D^{-1/2}HD^{-1/2})), D = √v̂+ε the run's own Adam state; = χ/(1+r), i.e. ṡ divided by s_c not s* (§13.3(i)).
- **Adam a=1.60 (Task B, n=200, seeds 830000-079)** — Adam lr 0.01; `results/lag_law/predictions.csv (set TaskB)`, column `pred_r`. κ: population κ_k at the certified switch (lag_law/kappa.csv; Adam P = per-coordinate median of the crossing P over existing runs; winding k measured at the crossing). χ: run's own timescale ratio at the crossing (residual_timescale): d log s/dt over the last ≤ 100 steps / (lr·λ_min(D^{-1/2}HD^{-1/2})), D = √v̂+ε the run's own Adam state; = χ/(1+r), i.e. ṡ divided by s_c not s* (§13.3(i)).
- **Adam a=1.65 (Track A, seeds 1650000-079), P at the crossing** — Adam lr 0.01; `results/track_a/diag_p_at_crossing.csv (POST HOC diagnostic, track_a_diag)`, column `p_cross_r_cf`. κ: κ_k from the a = 1.65 landscape (H, θ*′, ∇G at s*_pop) with the run's P at its crossing. χ: (ṡ/s*_run)/(lr·λ_min) at t_sw (ṡ over the last ≤ 100 steps), P at the crossing; = §13.1.
- **Adam a=1.65 (Track A), registered P at the rule point [sensitivity; not pooled]** — Adam lr 0.01; `results/track_a/observed_runs.csv`, column `r_cf`. κ: as above, P frozen at the rule point (last upward passage of 0.5·s*). χ: as above, P at the rule point. sensitivity to the P rule.
- **Adam a=1.85 (Test 2A, seeds 1850000-079), P at t_sw** — Adam lr 0.01; `results/track2a/observed_runs.csv`, column `r_cf`. κ: κ_k from the a = 1.85 landscape with the run's P at t_sw (first s ≥ s*_run). χ: (ṡ/s*_run)/(lr·λ_min) at t_sw, P at t_sw; = §13.1. the minimum is seed 1850004 (χ = 1.7e-5 at t_sw; not in 2A's scored set).
- **SGD a=1.30 (n=200, seeds 0-39)** — SGD lr 0.3; `results/lag_law/predictions.csv (set SGD)`, column `pred_r`. κ: population κ_k (lag_law/kappa.csv, P = I). χ: run's own ratio at the crossing (sgd_own_ratios.csv), P = I; ṡ divided by s_c.
- **SGD a=1.50 (n=200, seeds 0-39)** — SGD lr 0.3; `results/lag_law/predictions.csv (set SGD)`, column `pred_r`. κ: population κ_k (lag_law/kappa.csv, P = I). χ: run's own ratio at the crossing (sgd_own_ratios.csv), P = I; ṡ divided by s_c.
- **SGD a=1.65 (Track A, seeds 1650000-079)** — SGD lr 0.3; `results/track_a/observed_runs.csv`, column `r_cf`. κ: κ_k from the a = 1.65 landscape, P = I. χ: (ṡ/s*_run)/(lr·λ_min(H)) at t_sw; = §13.1.
- **SGD a=1.58 (Phase 1C W1, seeds 9350000-099)** — SGD lr 0.3; `results/phase1c/observed_W1.jsonl`, column `r_cf`. κ: κ_k from the a = 1.58 landscape (Track A construction), P = I. χ: (ṡ/s*_run)/(lr·λ_min(H)) at the actual t_sw (Track A's closed form); = §13.1.

## width-1 GELU

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam (Track 3A, seeds 850000-199), crossings at s ≥ 0.5·s_glob | 200 / 132 | 114 | 3.80e-04 | 9.00e-04 | 0.0021 | 0.0043 | 0.0075 | 114/114 (100%) | 114/114 (100%) |
| Adam, EARLY crossings s < 0.5·s_glob [reported separately; not pooled] | 200 / 132 | 17 | -7.06e+12 | -3.05e+12 | -1.11e+04 | 1.60e+06 | 4.63e+06 | 0/17 (0%) | 0/17 (0%) |

Sources and definitions:

- **Adam (Track 3A, seeds 850000-199), crossings at s ≥ 0.5·s_glob** — Adam lr 0.01; `results/act_general/train_gelu.csv + train_ext_gelu.csv (column chi) × kappa_gelu_frozen.json kappa_adam`, column `kappa_adam·chi`. κ: one frozen population κ_Adam = 0.1473 (act_general/kappa_gelu_frozen.json; P = median crossing P of 5 calibration runs), times the run's χ; the product is act_posthoc2's `pred` for the 117 runs with a branch switch (identical to 1e-16). χ: run's χ at the crossing (act_general._at_crossing): d log s/dt over the last ≤ 100 steps / (lr·λ_min) with the run's own Adam v̂ and the own-sample Hessian at the branch point; ṡ divided by s_c. 8 runs placed at init excluded; s_glob = 6.6415.
- **Adam, EARLY crossings s < 0.5·s_glob [reported separately; not pooled]** — Adam lr 0.01; `results/act_general/train_gelu.csv + train_ext_gelu.csv (column chi) × kappa_gelu_frozen.json kappa_adam`, column `kappa_adam·chi`. κ: one frozen population κ_Adam = 0.1473 (act_general/kappa_gelu_frozen.json; P = median crossing P of 5 calibration runs), times the run's χ; the product is act_posthoc2's `pred` for the 117 runs with a branch switch (identical to 1e-16). χ: run's χ at the crossing (act_general._at_crossing): d log s/dt over the last ≤ 100 steps / (lr·λ_min) with the run's own Adam v̂ and the own-sample Hessian at the branch point; ṡ divided by s_c. tail-placed one-sided-ramp branch with no switch (§13.7); s falling or relax ≈ 0, so κχ has no lag meaning here. Signed and |·| differ: 10 negative; |·| median 1.06e+06, q10 581.4331, q90 3.05e+12, max 7.06e+12.

## band task in R^d

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam 3B primary d=1 a=1.30 | 40 / 40 | 40 | 0.0190 | 0.0250 | 0.0289 | 0.0324 | 0.0352 | 40/40 (100%) | 1/40 (2%) |
| Adam 3B primary d=1 a=1.50 | 40 / 40 | 40 | 0.0392 | 0.0513 | 0.0626 | 0.0751 | 0.0873 | 40/40 (100%) | 0/40 (0%) |
| Adam 3B primary d=2 a=1.30 | 40 / 36 | 36 | -0.0305 | 0.0195 | 0.0238 | 0.0282 | 0.0323 | 36/36 (100%) | 4/36 (11%) |
| Adam 3B primary d=2 a=1.50 | 40 / 36 | 36 | -0.0689 | 0.0403 | 0.0490 | 0.0609 | 0.0688 | 36/36 (100%) | 0/36 (0%) |
| Adam 3B primary d=4 a=1.30 | 40 / 29 | 29 | 0.0097 | 0.0137 | 0.0175 | 0.0210 | 0.0237 | 29/29 (100%) | 23/29 (79%) |
| Adam 3B primary d=4 a=1.50 | 40 / 29 | 29 | 0.0193 | 0.0293 | 0.0358 | 0.0412 | 0.0482 | 29/29 (100%) | 1/29 (3%) |
| Adam 3B secondary d=2 a=1.30 | 80 / 76 | 76 | 0.0114 | 0.0180 | 0.0236 | 0.0305 | 0.0363 | 76/76 (100%) | 15/76 (20%) |
| Adam 3B secondary d=2 a=1.50 | 80 / 76 | 76 | 0.0238 | 0.0357 | 0.0498 | 0.0649 | 0.0790 | 76/76 (100%) | 0/76 (0%) |
| Adam 3B secondary d=4 a=1.30 | 80 / 61 | 61 | 0.0075 | 0.0107 | 0.0157 | 0.0206 | 0.0249 | 61/61 (100%) | 46/61 (75%) |
| Adam 3B secondary d=4 a=1.50 | 80 / 64 | 64 | -0.0354 | 0.0201 | 0.0323 | 0.0426 | 0.0522 | 64/64 (100%) | 6/64 (9%) |
| Adam 2C d=2 a=1.30 (seeds 2030000-059), all crossers | 60 / 57 | 57 | 0.0154 | 0.0200 | 0.0269 | 0.0312 | 0.0377 | 57/57 (100%) | 6/57 (11%) |
| Adam 2C d=2 a=1.30, t_sw before the crossing [sensitivity; not pooled] | 60 / 57 | 50 | 0.0154 | 0.0200 | 0.0270 | 0.0314 | 0.0377 | 50/50 (100%) | 5/50 (10%) |
| Adam 2C d=2 a=1.50 (seeds 2030000-059), all crossers | 60 / 57 | 57 | 0.0298 | 0.0419 | 0.0567 | 0.0747 | 0.0981 | 57/57 (100%) | 0/57 (0%) |
| Adam 2C d=2 a=1.50, t_sw before the crossing [sensitivity; not pooled] | 60 / 57 | 53 | 0.0298 | 0.0417 | 0.0567 | 0.0768 | 0.0981 | 53/53 (100%) | 0/53 (0%) |
| Adam 2C d=4 a=1.30 (seeds 2030000-059), all crossers | 60 / 51 | 51 | 0.0082 | 0.0131 | 0.0179 | 0.0232 | 0.0277 | 51/51 (100%) | 33/51 (65%) |
| Adam 2C d=4 a=1.30, t_sw before the crossing [sensitivity; not pooled] | 60 / 51 | 41 | 0.0082 | 0.0131 | 0.0181 | 0.0232 | 0.0277 | 41/41 (100%) | 25/41 (61%) |
| Adam 2C d=4 a=1.50 (seeds 2030000-059), all crossers | 60 / 53 | 53 | 0.0146 | 0.0273 | 0.0383 | 0.0507 | 0.0635 | 53/53 (100%) | 1/53 (2%) |
| Adam 2C d=4 a=1.50, t_sw before the crossing [sensitivity; not pooled] | 60 / 53 | 48 | 0.0146 | 0.0259 | 0.0380 | 0.0498 | 0.0635 | 48/48 (100%) | 1/48 (2%) |

Sources and definitions:

- **Adam 3B primary d=1 a=1.30** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c. d = 1 is the width-1 control of the band task.
- **Adam 3B primary d=1 a=1.50** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c. d = 1 is the width-1 control of the band task.
- **Adam 3B primary d=2 a=1.30** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c. Signed and |·| differ: 1 negative; |·| median 0.0243, q10 0.0200, q90 0.0288, max 0.0323.
- **Adam 3B primary d=2 a=1.50** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c. Signed and |·| differ: 1 negative; |·| median 0.0498, q10 0.0417, q90 0.0637, max 0.0689.
- **Adam 3B primary d=4 a=1.30** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c.
- **Adam 3B primary d=4 a=1.50** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c.
- **Adam 3B secondary d=2 a=1.30** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c.
- **Adam 3B secondary d=2 a=1.50** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c.
- **Adam 3B secondary d=4 a=1.30** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c.
- **Adam 3B secondary d=4 a=1.50** — Adam lr 0.01; `results/band_rd/scored_runs.csv`, column `pred_r`. κ: population width-1 κ_k (lag_law/kappa.csv, median Adam P) for the run's winding k at the crossing (band_rd.lag_quantities). χ: at the crossing: growth = log(s_c/s_{c−100})/100, relax = lr·λ_min(P^{1/2}H_sig P^{1/2}), P = run's Adam v̂ at the crossing, H_sig the own-sample signal block at the run's branch; ṡ divided by s_c. Signed and |·| differ: 1 negative; |·| median 0.0326, q10 0.0214, q90 0.0426, max 0.0522.
- **Adam 2C d=2 a=1.30 (seeds 2030000-059), all crossers** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). 7 crossers have t_sw ≥ the crossing step (χ read after the crossing).
- **Adam 2C d=2 a=1.30, t_sw before the crossing [sensitivity; not pooled]** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). = the registered scored set.
- **Adam 2C d=2 a=1.50 (seeds 2030000-059), all crossers** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). 4 crossers have t_sw ≥ the crossing step (χ read after the crossing).
- **Adam 2C d=2 a=1.50, t_sw before the crossing [sensitivity; not pooled]** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). = the registered scored set.
- **Adam 2C d=4 a=1.30 (seeds 2030000-059), all crossers** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). 10 crossers have t_sw ≥ the crossing step (χ read after the crossing).
- **Adam 2C d=4 a=1.30, t_sw before the crossing [sensitivity; not pooled]** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). = the registered scored set.
- **Adam 2C d=4 a=1.50 (seeds 2030000-059), all crossers** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). 5 crossers have t_sw ≥ the crossing step (χ read after the crossing).
- **Adam 2C d=4 a=1.50, t_sw before the crossing [sensitivity; not pooled]** — Adam lr 0.01; `results/track2c/observed_runs.csv`, column `r_pred`. κ: as 3B (track2c/kappa_k.csv), the run's winding k at the rule point. χ: at t_sw (first |w₂| ≥ s_own,d), not at the crossing: growth = log(s_{t_sw}/s_{t_sw−100})/100, relax with the run's Adam P at t_sw (track2c.predict_run). = the registered scored set.

## width 2

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |κχ| ≤ 0.1 | |κχ| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam T2-3 (a=1.30, Δ=0.4, matched init, seeds 600000-079) | 80 / 79 | 75 | -1.2726 | 0.0279 | 0.2576 | 0.8344 | 28.6255 | 9/75 (12%) | 0/75 (0%) |

Sources and definitions:

- **Adam T2-3 (a=1.30, Δ=0.4, matched init, seeds 600000-079)** — Adam lr 0.01; `results/width2_lag/parts_T2-3.csv (width2_lag.analyse)`, column `r_pred`. κ: κ₂ = λ_min·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′] at the run's own-path switch, Adam P at the crossing. χ: (ṡ/s*)/(lr·λ_min) with ṡ at the crossing (≤ 100-step window), s* the own-path switch. 1 placed at init (asym_parts/train.csv); analysed crossings: no_branch_at_crossing 2, ok 75, placed_back_to_init 2; κχ only for status ok; no run within 0.05 of its branch at the crossing (min dist_c 0.061). Signed and |·| differ: 7 negative; |·| median 0.2677, q10 0.0936, q90 0.8537, max 28.6255.

## simplicity bias

Quantity: **χ only (κ undefined)**; the |·| ≤ 0.1 / 0.02 columns apply to this quantity, not to κχ.

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |·| ≤ 0.1 | |·| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam χ only, late crossers (s_c 8.9-10.7, above s_switch = 3.59) | 40 / 22 | 16 | 3.9949 | 5.9934 | 9.2398 | 15.9788 | 22.0170 | 0/16 (0%) | 0/16 (0%) |
| Adam χ only, early crossers (s_c 0.20-0.95, s falling: χ < 0) [not pooled] | 40 / 22 | 6 | -279.0396 | -275.7313 | -238.7484 | -182.8531 | -145.0229 | 0/6 (0%) | 0/6 (0%) |

Quantity: **ε_F at M's fold**; the |·| ≤ 0.1 / 0.02 columns apply to this quantity, not to κχ.

| cell | runs / crossed | n finite | min | q10 | median | q90 | max | |·| ≤ 0.1 | |·| ≤ 0.02 |
|---|---|---|---|---|---|---|---|---|---|
| Adam ε_F at M's fold, per run (own P at the first s ≥ s_F) | 40 / 22 | 40 | 0.9879 | 2.9663 | 5.5834 | 9.2645 | 15.0252 | 0/40 (0%) | 0/40 (0%) |

Sources and definitions:

- **Adam χ only, late crossers (s_c 8.9-10.7, above s_switch = 3.59)** — Adam lr 0.01; `results/simplicity_bias_v3/runs_summary.csv (chi); groups from sb_fold/per_run.csv`, column `chi`. κ: UNDEFINED: M has no switch of its own (ρ₂ ≤ 0.279, G₊ < 0); no κ for any branch. χ: v3's χ at the ρ₂ upward passage (simplicity_bias_v3.chi_at): d log s/dt over ≤ 100 steps (÷ s_c, not s*) / (lr·λ_min(P^{1/2}HP^{1/2})), H = the v2 13×13 weight-decayed Hessian at s_switch on the S side (not at the run's branch), P = the run's Adam block medians at the crossing (no v block).
- **Adam χ only, early crossers (s_c 0.20-0.95, s falling: χ < 0) [not pooled]** — Adam lr 0.01; `results/simplicity_bias_v3/runs_summary.csv (chi); groups from sb_fold/per_run.csv`, column `chi`. κ: UNDEFINED: M has no switch of its own (ρ₂ ≤ 0.279, G₊ < 0); no κ for any branch. χ: v3's χ at the ρ₂ upward passage (simplicity_bias_v3.chi_at): d log s/dt over ≤ 100 steps (÷ s_c, not s*) / (lr·λ_min(P^{1/2}HP^{1/2})), H = the v2 13×13 weight-decayed Hessian at s_switch on the S side (not at the run's branch), P = the run's Adam block medians at the crossing (no v block). Signed and |·| differ: 6 negative; |·| median 238.7484, q10 182.8531, q90 275.7313, max 279.0396.
- **Adam ε_F at M's fold, per run (own P at the first s ≥ s_F)** — Adam lr 0.01; `results/phase2c/sb_eps_F.json (phase2c_map.sb_eps_f)`, column `eps_F`. κ: not applicable (fold). χ: ε_F = (ṡ_F/s_F)/(lr·Λ_F), Λ_F = √(|m_c|·s_F), m_c from λ_min(P^{1/2}HP^{1/2})² ≈ 4|m_c|(s_F − s) on M's grid within 0.05 of s_F (sb_fold.run_precond_constant), with the run's own block-median P instead of the median over runs. population ε_F (median P, median ṡ) = 5.576; no run is on M at the fold.

## Caveats

- **Width-1 sine.** χ in the lag_law sets divides ṡ by s_c, not s\* (factor 1/(1 + r); per-arm median r ≤ 0.081, §13.3(i)). Track A Adam is shown with P at the crossing (POST HOC diagnostic) and, unpooled, with the registered rule-point P. Task B (n = 200) and the lag-test reference runs (n = 6,400) differ only in sample size.
- **GELU.** One frozen population κ for every run. The early crossings (s_c < 0.5·s_glob = 3.32; s_c 0.004-0.84, against ≥ 5.09 for every other crossing) have κχ of either sign with |κχ| up to ~7·10¹², because s is falling or relax ≈ 0. §13.7 places GELU's 15 early crossings (s ≤ 0.45) on the tail-placed one-sided-ramp branch, where the continuation finds no switch. They are listed separately and not pooled. One early crossing (seed 850068) has NaN χ.
- **Band task.** 3B measures χ at the crossing; 2C measures it at t_sw (the first |w₂| ≥ s_own,d), which for 26 crossers lies at or after the crossing step (flagged per cell; the t_sw-before-crossing subset is shown unpooled). 3B and 2C use the width-1 population κ_k; d = 2 and d = 4 cells share seeds by design. Three 3B runs sit on a negative-κ winding.
- **Width 2.** Only T2-3 has per-run κ and χ, and it uses the registered MATCHED initialisation (output weights scaled by k = 0.04209), not the standard one; the standard-init version is sealed with no κ/χ. No T2-3 run is on its branch at the crossing (every dist_c > 0.05), so the law's premise (tracking the branch whose switch is used, §13.3(iv)(d)) fails for every run; κχ here is formal. Seven runs have κ₂ < 0.
- **Simplicity bias.** κ is undefined: M has no switch of its own. Only v3's χ at the ρ₂ upward passage exists, and its definition is not §13's (ṡ/s_c; H at s_switch on the S side, not at the run's branch; P without the v block). v3's initialisation is PyTorch default with the output scale capped at 0.5·s_switch. ε_F per run is computed here with sb_fold's construction and the run's own P; the population ε_F with median P and median ṡ is the committed 5.58 (reproduced). No run is on M at the fold (basin at the fold: S 30, other 5, L0 4, S2 1), so ε_F describes the drive rate at s_F, not an event of the run. sb_fold calls ε_F > 0.3 outside the small-ε regime of §14; the per-run minimum is 0.99.
- Fractions are over finite values. Runs that never cross, or cross with no defined κχ, are counted in "runs / crossed" but not in n.

## Excluded (non-standard, or no κχ committed)

| setting | runs | reason |
|---|---|---|
| width-1 sine | lag tests φ ≠ 1 (lag1, lag2-prim, lag2-tstar; w₂ slowed or sped up) and residual-mechanism arms 4b reset / teleport / teleport_reset | non-standard (slowed / intervened) |
| width-1 sine | Track 1B ramp (results/ramp/runs.csv), Test 2B (track2b), ramp_boundary, ramp2 | imposed growth of s (ramped), warm-up at a stationary point |
| width-1 sine | ramp R4 free Adam (results/ramp/free_runs.csv; η = 0.01, 0.005, 0.0025) | free, but no χ or κχ committed (η = 0.01 is standard; would need a replay of 80 runs) |
| width-1 sine | Phase 1A W1 dev/est runs (results/phase1a/runs_w1_sgd.jsonl) | free SGD, but no κχ at the actual switch committed |
| width-1 GELU | GELU-T (results/gelu_transfer), Phase 1C G, Phase 1A gelu_random | held at s₀ on a copy, then released with SGD η = 0.03 |
| width-1 GELU | act_fold (results/act_fold) | the same Track 3A runs; no κ/χ |
| width 2 | W2-A arms T, D, T′ (results/width2_asym), Phase 1C T, T′, Phase 1A w2a_T/Tp | started at a branch point, held, then slowed SGD |
| width 2 | T2-3b, T2-3c, T2-3d (results/width2_lag/parts_T2-3{b,c,d}.csv) | output lr × φ₂ (slowed) |
| width 2 | standard-init T2-3 runs (results/asym_sealed/train_uniform_init_seeds0-79.csv) | sealed / withdrawn before analysis; no κ/χ |
| width 2 | width2_fold | 0 folds found in the 79 T2-3 runs; no ε_F |
| simplicity bias | Phase 2A exploration (results/designs/phase2a_explore) | slow releases from the exact M point, plain GD; not free training |

## Not computed

- χ for the ramp R4 free-Adam runs at η = 0.01 (no χ committed; a replay of 80 runs would be needed).
- κχ at width 2 with standard initialisation (no such runs analysed; the sealed runs carry no κ/χ).
- Any κ for the simplicity-bias benchmark (undefined on M).
