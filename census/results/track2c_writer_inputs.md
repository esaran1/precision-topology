# Test 2C writer inputs: the band task in R^d against each seed's own-sample R^d switch, prospectively

All numbers come from committed producers:
- `census/src/track2c.py` (registered; `freeze`, `kappa`, `train`, `finalize`, `observe`);
- `census/src/track2c_posthoc.py` (POST HOC, labelled).

Outputs are in `census/results/track2c/`. Registration: `census/results/track2c_registration.md`. Design (approved
2026-09-29, unchanged): `census/results/designs/2C_band_own_sample_design.md`.

## 1. What was committed before any comparison

| commit | content |
|---|---|
| 8dcc690 | Implementation and tests (33) of every rule. Timing on a non-registered seed. NON-REGISTERED mirror diagnostic on 3B seeds. |
| 4e1fbe7 | Registration text (criteria, validity, 13 details the page left open, a-priori risks). |
| b22ebd0 | **REGISTRATION:** frozen inputs and the SHA-256 manifest `registration.sha256` (18 files). |
| 906c07b | OpenTimestamps proof of b22ebd0 (`registration_stamp.txt`, `.ots`, 4 calendars). |
| d8d3569 | **Predictions** of all 240 runs, committed before any crossing was evaluated. `predictions.csv` SHA-256 27c0b676…, with per-run path hashes. |
| (b863a54) | Observation, scores, POST HOC, these writer inputs, ledger checks. |

**Order.**
- The registration text was committed before any registered seed was searched.
- The frozen inputs were computed next (02:48–06:47) and committed in b22ebd0 at 06:51:43.
- Training started after that commit. It evaluated no gap.
- The predictions were committed in d8d3569 at 07:16:24.
- `observe` asserted the manifest, the committed predictions hash and every path hash before evaluating any gap.

## 2. Design (short)

- **Setting:** Track 3B's band task.
  - x₁ is the width-1 task (400 points); x₂…x_d are i.i.d. U(−2, 2).
  - d = 2, 4 and a = 1.30, 1.50: four cells.
  - One f_a unit, free Adam lr 0.01, 64,000 steps, every-step crossing detection.
  - 60 fresh seeds (2,030,000–2,030,059), the same seeds in every cell.
- **Prediction per run:** s_pred = s_own,d·(1 + κ_k·χ).
  - **s_own,d** is frozen before training. It is the switch of the seed's own-sample conditional minimiser in all
    hidden coordinates, noise weights included, from a validated search. **240 of 240 seed-cells validated** (restart
    ladder, independent CMA-ES, audit).
  - **κ_k** is the width-1 κ_k, frozen: 7.611 at a = 1.30 (k = −1) and 4.046 at a = 1.50 (k = 0).
  - **χ** uses pre-crossing information only. The rule point is Track A's (the last upward passage of 0.5·s_own,d).
    **P is frozen at t_sw**, the first step with |w₂| ≥ s_own,d (the 2A rule; 2A passed).
- **Frozen offsets:** median s_own,d/s_own,x1 is 1.075 and 1.078 at d = 2, and 1.359 and 1.355 at d = 4. Here s_own,x1
  is the x₁-only own threshold.
- **Criteria per cell,** on scored runs (crossing runs with t_sw strictly before the crossing):
  - C1: median |log(s_obs/s_pred)| ≤ 0.05;
  - C2: median signed r_obs/r_pred ∈ [0.75, 1.25];
  - C3: the upper end of the 95% percentile bootstrap interval of mean(|log err_pred| − |log err_x1-only|) < 0.
- **Validity per cell:**
  - V1: ≥ 30 crossings;
  - V2: ρ at crossing < 0.05 in ≥ 90% of crossing runs;
  - V3: a pre-crossing prediction in ≥ 90% of crossing runs. V3 is not in the design page; the registration added it
    (§7 item 9, 2A's convention) so that the scored set is not selected by the outcome.

## 3. Registered results (`scores.json`, `observed_runs.csv`)

| cell | crossings | ρ < 0.05 | scored / crossing (V3) | C1 median \|log err\| | C2 median ratio | C3 95% CI of mean D | outcome |
|---|---|---|---|---|---|---|---|
| d = 2, a = 1.30 | 57/60 | 57/57 | 50/57 = 0.877 **(fails)** | 0.0032 | 0.956 | [−0.132, −0.094] | **UNRESOLVED** (V3) |
| d = 2, a = 1.50 | 57/60 | 57/57 | 53/57 = 0.930 | 0.0076 | 0.940 | [−0.158, −0.121] | **PASS** |
| d = 4, a = 1.30 | 51/60 | 51/51 | 41/51 = 0.804 **(fails)** | 0.0068 | 1.010 | [−0.394, −0.310] | **UNRESOLVED** (V3) |
| d = 4, a = 1.50 | 53/60 | 53/53 | 48/53 = 0.906 | 0.0091 | 1.071 | [−0.403, −0.332] | **PASS** |

- **Registered outcome:** PASS in the two a = 1.50 cells. **UNRESOLVED in the two a = 1.30 cells**, because V3 failed:
  7 of 57 and 10 of 51 crossing runs crossed before |w₂| reached s_own,d, so they had no pre-crossing prediction.
- In the UNRESOLVED cells, the statistics on the scored runs are reported, **not verdicts**.
- Validity V1 and V2 held in all four cells. No run was placed at initialisation. The largest ρ at crossing was 0.045.
- Non-crossing runs: 3, 3, 9 and 7 of 60. In 3, 3, 9 and 6 of them |w₂| never reached s_own,d.
- **Registered descriptives (scored runs):**

  | cell | median r_obs | median r_pred | Spearman(r_pred, r_obs) | median \|log err\| of s_own,x1 alone | of s_own,d alone (no lag) | stricter comparator s_own,x1(1 + κχ): 95% CI of mean D |
  |---|---|---|---|---|---|---|
  | d = 2, a = 1.30 | 0.0263 | 0.0270 | 0.59 | 0.107 | 0.026 | [−0.107, −0.067] |
  | d = 2, a = 1.50 | 0.0551 | 0.0567 | 0.67 | 0.132 | 0.054 | [−0.104, −0.063] |
  | d = 4, a = 1.30 | 0.0194 | 0.0181 | 0.31 | 0.329 | 0.019 | [−0.376, −0.290] |
  | d = 4, a = 1.50 | 0.0420 | 0.0380 | 0.51 | 0.365 | 0.041 | [−0.367, −0.292] |

  - Every scored run had the typical winding: k = −1 at a = 1.30 and k = 0 at a = 1.50.
  - t_sw preceded the crossing by a median 54–63 steps; the rule point, by 629–1,613 steps.
- **Mirror-matched prediction** (registered descriptive, NOT scored). The same rules, with the frozen switch of the
  mirror occupied at the rule point:
  - d = 2, a = 1.30: PASS pattern, 56/57 scored, C1 0.0027, C2 0.948, C3 upper end −0.106.
  - d = 2, a = 1.50: PASS pattern, 57/57 scored, C1 0.0067, C2 0.922, C3 upper end −0.125.
  - d = 4, a = 1.30: V3 fails, 42/51 = 0.824.
  - d = 4, a = 1.50: PASS pattern, 52/53 scored, C1 0.0076, C2 1.064, C3 upper end −0.323.
  - The rule-point mirror was the global minimiser's in 43/57, 42/57, 41/51 and 43/53 crossing runs.

## 4. POST HOC (labelled; after the registered scores; `posthoc_unscored.csv`, `posthoc_summary.csv`)

**The 26 unscored crossing runs.** All crossed below s_own,d, at 0.813–0.999·s_own,d.
- **17 were on the other mirror at the rule point.** That mirror's frozen switch was lower. They crossed 1.1–9.0% above
  it, which is a positive lag of the usual size.
  - By cell: d = 2, 6 of 7 (a = 1.30) and 4 of 4 (a = 1.50); d = 4, 3 of 10 (a = 1.30) and 4 of 5 (a = 1.50).
- **9 were on the global minimiser's mirror.** They crossed 0.08–3.8% below s_own,d: 1 at d = 2, a = 1.30; 7 at d = 4,
  a = 1.30; 1 at d = 4, a = 1.50.
- **All crossing runs, with the static floor** s_pred = s_own,d for runs without a pre-crossing prediction (NOT a
  registered rule):
  - median |log err| 0.0033, 0.0082, 0.0073, 0.0112;
  - C3-type upper end −0.085, −0.108, −0.288, −0.305.

## 5. Writer section

**Outcome that occurred: PASS at a = 1.50 (d = 2 and d = 4); UNRESOLVED at a = 1.30 (d = 2 and d = 4, validity V3).**

**Outcome: a = 1.50 cells PASS**
- Say: "In a prospective, registered test, each seed's own-sample R^d threshold was computed before training. That
  threshold plus the width-1 lag law predicted the crossing scale at a = 1.50 to a median 0.8% at d = 2 and 0.9% at
  d = 4. The observed-to-predicted lag ratio was 0.94 and 1.07, and the prediction beat the noise-blind x₁-only
  threshold (95% intervals of the mean log-error difference entirely below 0)."
- Say: "The frozen own-sample R^d threshold sat a median 8% (d = 2) and 36% (d = 4) above the x₁-only threshold. This
  confirms prospectively that the noise coordinates shift the finite-sample threshold, and that the shift is computable
  from the sample before training."
- Do not say: that 2C passed as a whole, or that the decomposition was confirmed at both activation values. Two of the
  four registered cells are UNRESOLVED.
- Do not say: that the population threshold moves with d (3B proved it does not). What moves is the finite-sample
  threshold.

**Outcome: a = 1.30 cells UNRESOLVED (V3)**
- Say: "At a = 1.30 the registered test is unresolved. 12% (d = 2) and 20% (d = 4) of crossing runs crossed before
  reaching the frozen threshold, so they had no pre-crossing prediction; the registered minimum was 90%."
- Say, if needed: "On the runs that could be scored, the statistics were within the registered bands (median error 0.3%
  and 0.7%, lag ratio 0.96 and 1.01). They are reported, not scored."
- Do not say: that C1–C3 passed at a = 1.30. They were not scored.
- Do not say: that the a = 1.30 cells failed.
- Do not present V3 as part of the approved design. It was added in the registration (2A's convention) before any run,
  so that the scored set would not be selected by the outcome.

**Mirror branches (POST HOC and registered descriptive)**
- Say, labelled post hoc: "Most runs that crossed below the frozen threshold had settled on the mirror branch
  (w₁ → −w₁) of the finite sample. Its own threshold is lower: 17 of 26 such runs, which crossed 1.1–9.0% above their
  mirror's frozen threshold."
- Say: "Before training we registered, as a descriptive, the mirror-matched version of the prediction (the frozen
  threshold of the mirror occupied at the rule point). It gave the same picture at d = 2 and at d = 4, a = 1.50, and did
  not resolve d = 4, a = 1.30 (82% predicted before crossing)."
- Do not say: that the mirror-matched prediction is the registered test, or that it passed as a registered criterion.
- Do not say: that mirror choice explains every unscored run. 9 of 26 were on the global mirror and crossed slightly
  (0.1–3.8%) below its threshold, 7 of them at d = 4, a = 1.30.

**If asked about the lag itself**
- Say: "On the scored runs, the median ratio of the observed lag (from the frozen threshold) to the predicted lag (κχ,
  from pre-crossing information) was 0.94–0.96 at d = 2 and 1.01–1.07 at d = 4. Per-run ordering was moderate:
  Spearman 0.31–0.67."
- Do not say: that per-run ordering was a registered criterion here. It was not.

## 6. Files

- **Registered producer:** `census/src/track2c.py`. Tests: `census/tests/test_track2c.py` (33 tests).
- **POST HOC producer:** `census/src/track2c_posthoc.py`.
- **Frozen before training:** `frozen_seeds.csv`, `frozen_parts.jsonl`, `kappa_k.csv`, `registration.sha256`,
  `registration_stamp.txt(.ots)`, `freeze.log`.
- **Predictions:** `predictions.csv` (SHA-256 27c0b676…), `predictions.sha256`, `predictions_parts.jsonl`, `train.log`.
- **Observation:** `observed_parts.jsonl`, `observed_runs.csv`, `scores.json`, `observe.log`.
- **POST HOC:** `posthoc_unscored.csv`, `posthoc_summary.csv`, `posthoc.log`.
- **NON-REGISTERED pre-registration:** `timing.csv`, `timing.log`, `diag_mirror_3b.csv`, `diag_mirror_3b.log`.
- Paths (`paths/`, 704 MB) are untracked. Their SHA-256 are in `predictions.csv`, and `train` regenerates them.
