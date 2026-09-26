# Track A: a registered test of the lag law at an unseen activation value — writer inputs

**REGISTERED** (`results/track_a_registration.md`). Code: `src/track_a.py`. Outputs: `results/track_a/`.

## 1. Frozen before any run or comparison

| commit | content |
|---|---|
| `fcd2e46` | Registration: md, code, tests, landscape and per-seed frozen inputs. SHA-256 in `results/track_a/registration.sha256`. No registered run had been trained. Gate: 516 + 27 tests passed. |
| `406b32c` | All 160 registered runs trained to budget with no gap or placement evaluated. Predictions (trajectory-integrated and closed form) committed; `predictions.csv` SHA-256 `b30167c61116972298dc434494918dced800b83a56164374ef338167070e3d82`. Each run's saved parameter path is hashed in that file. |
| this commit | `observe`: asserts the committed hashes, then runs every-step detection on the saved paths; scores. |

- **a = 1.65**, the author's decision: no existing crossing data.
  - 1.35 and 1.40 have width-1 crossing data.
  - 1.55 and 1.70 appear in the onset placement-rate files.
  - 1.65 appears only in a landscape table and in a depth-3 annealing search.
- **Seeds.** 80 fresh seeds (1,650,000–1,650,079) per optimiser.
- **Landscape.** No certified bracket exists at 1.65, so the landscape was validated by continuation plus the existing
  conditional search. The global minimiser is unplaced at 0.995·s\*_pop and placed at 1.005·s\*_pop, on the same branch.
  s\*_pop = 1.81002.
- **Winding rule.** Canonical b₁ ∈ (−π, π]. The rule was defined from existing crossings at other a (all 1,750 at
  1.30–1.60 obey it).
- **Per-run rules.** Rule point, branch rule, preconditioner rule, trajectory-integrated model with P frozen at the rule
  point, and closed form. They use only information up to the rule point plus the run's s_t trajectory.
- **Deviation from the literal spec (stated in the registration before any run).** The rule point is the last upward
  passage of 0.5·s\*_frozen before s\*_frozen, not the first. At 1.65, 0.5·s\* ≈ 0.9 lies inside the initial |w₂| range.

## 2. Verdicts (per optimiser; relative only)

| | crossings (valid ≥ 60) | runs with prediction | **L1** median r_obs/r_traj ∈ [0.90, 1.10] | **L2** median r_obs/r_cf ∈ [0.80, 1.20] | **L3** Spearman(r_traj, r_obs) ≥ 0.5 |
|---|---|---|---|---|---|
| Adam | 76 / 80 (valid) | 74 | **1.065 PASS** | **1.045 PASS** | **0.25 FAIL** |
| SGD | 64 / 80 (valid) | 63 | **1.056 PASS** | **1.024 PASS** | **0.995 PASS** |

- No run had its rule point at or after its crossing.
- Runs without a prediction were not replaced:
  - Adam: 4 never reach s\*_frozen, and the branch rule failed in 2.
  - SGD: 16 never reach s\*_frozen (the known small-|w₂| plateau), and the branch rule failed in 1.
- Crossing runs without a prediction: 2 Adam, 1 SGD.
- Secondary (registered as reported, not scored): the Spearman of the closed form r_cf with r_obs is 0.55 (Adam) and
  0.81 (SGD).

**Descriptive** (`scores.json`, `scores_descriptive.json`):
- Median lags: r_obs 0.094 for both optimisers; r_traj 0.080 (Adam), 0.090 (SGD); r_cf 0.089 (Adam), 0.088 (SGD).
- The winding rule agreed with the occupied branch in every run with a prediction (74 of 74 Adam, 63 of 63 SGD).
- About half the runs occupy the mirror branch.
- s\*_run differs from s\*_frozen by more than 1% in 40% (Adam) and 38% (SGD) of runs, mostly mirror-branch runs. The
  lag is measured against s\*_run, as registered.
- Per-run observed/predicted (trajectory-integrated), 10th–90th percentile:
  - SGD: 1.04–1.06. Every run is within 10%, and the median |r_obs − r_traj| is 0.005.
  - Adam: 0.74–1.75. Only 22% of runs are within 10%, and the median |r_obs − r_traj| is 0.023.
- The observed r_obs spread for Adam is narrow (0.079–0.106), narrower than the prediction spread (0.054–0.129). The
  per-run Adam predictions are therefore noisier than the lags they rank.
- **Interpretation (POST HOC, not tested here).** The registered Adam model freezes P at the rule point, about 0.5·s\*.
  In Track 1 (post hoc, existing runs), the same recursion with the run's actual time-varying P_t matched per run.
  Adam's v̂ memory is 1,000 steps, so P at the rule point need not equal P near the crossing. This is the likely source
  of the Adam per-run scatter. It is not demonstrated in this track.

## 3. Writer section

**Outcome for SGD: all three pass.**
- Say: "In a registered test at an activation value with no prior crossing data (a = 1.65, 80 fresh seeds), everything
  was frozen before training: the winding index, the branch rule and the preconditioner rule. For plain SGD the lag law
  predicts the crossing lag with median observed/predicted 1.06 (trajectory-integrated) and 1.02 (closed form κχ). The
  per-run rank correlation is 0.99."
- Do not say: that the rules were chosen without any data. The winding rule was defined from existing crossings at other
  a, and the rule point was adapted to the initial transient before any run; say both. Do not say that the test covers
  other widths or activations.

**Outcome for Adam: L1 and L2 pass, L3 fails.**
- Say: "For Adam, with the preconditioner frozen at the half-switch point, the median observed/predicted lag is 1.07
  (trajectory-integrated) and 1.05 (closed form). Both are inside the registered bands. The per-run ranking fails
  (Spearman 0.25 < 0.5): with only pre-crossing information, the law predicts Adam's typical lag but not which runs lag
  more."
- Do not say: "the lag law predicts each Adam run's lag". Do not say that the L3 failure is explained by the
  preconditioner drift. That is a post hoc hypothesis, supported only indirectly by Track 1, and untested here. Do not
  report the closed-form Spearman (0.55) as passing L3: L3 was registered on the trajectory-integrated prediction.

**Overall.**
- Say: "5 of 6 registered criteria pass; the one failure is Adam's per-run rank correlation."
- Do not say: "the lag law passed its registered test" without the Adam L3 qualification.

## 4. Files

- `results/track_a_registration.md`
- `results/track_a/`:
  - `registration.sha256`, `landscape.json`, `frozen_seeds.csv`, `freeze.log`
  - `predictions.csv`, `predictions.sha256`, `train_*.log`
  - `observed_runs.csv`, `scores.json`, `scores_descriptive.json`
- Parameter paths `paths/*.npz` (160 MB) are untracked. Their SHA-256 hashes are in `predictions.csv`.

## 5. POST HOC diagnostic: Adam ordering (author-approved, after the registered score)

**Label: POST HOC.**
- Freezing P at the observed crossing uses crossing-time information. It is a diagnostic, never a prediction. **The
  registered Adam L3 FAIL stands.**
- Producer: `src/track_a_diag.py`, outputs `results/track_a/diag_p_at_crossing.csv` and `.json`.

**Method.** The registered pipeline (`track_a.predict_one`) is re-run unchanged with only the step at which P is frozen
changed.
- Rule point, branch, δ₀, m₀, H and θ\* path, s path and stability rule are all identical.
- Adam's moments come from deterministic retraining. Each retrained path equals its saved, hashed path bit for bit.
- The registered predictions are reproduced to round-trip precision (74 of 74).

All 74 Adam runs that crossed and have a prediction:

| P frozen at | Spearman (traj.) | within 10% (traj.) | median obs/pred (traj.) | predicted r, q10–q90 (traj.) | Spearman (closed form) | within 10% (closed form) |
|---|---|---|---|---|---|---|
| rule point (registered) | 0.25 | 0.22 | 1.065 | 0.054–0.129 | 0.55 | 0.30 |
| observed crossing (diagnostic) | **0.99** | **1.00** | 1.057 | 0.074–0.102 | 0.71 | 0.45 |
| occupied branch's switch t_sw (candidate rule) | **0.95** | **0.91** | 1.042 | 0.074–0.102 | 0.74 | 0.50 |

- The observed r spans 0.079–0.106 (q10–q90).
- t_sw is the first step with |w₂| ≥ s\*_run. It came before the crossing in 74 of 74 runs.
- **Outcome: the failure is a measurement-point artifact.** With P taken near the crossing, the same linear-response
  model ranks the Adam runs almost perfectly, and every run falls within 10%. Frozen at the half-switch point, P is
  stale: Adam's v̂ keeps evolving over its ~1,000-step memory between the rule point and the crossing.

**Corrected rule for a future registered test (pre-crossing information only).**
- Freeze Adam's preconditioner at t_sw, the first step at which |w₂| reaches the switch s\*_run of the branch occupied at
  the rule point. s\*_run is known at the rule point, and t_sw is read from the output-scale trajectory in real time.
- It precedes the crossing whenever the lag is positive: 74 of 74 here.
- This rule was examined post hoc on these runs. It is a candidate for a new registration, not a validated one.

**Paper sentence:**
"The registered Adam failure of per-run ordering (Spearman 0.25) is a measurement-point artifact. Post hoc, freezing
the preconditioner at the crossing instead of at half the switch scale raises the per-run Spearman to 0.99, with every
run within 10%. The pre-crossing rule 'preconditioner at the first step the output scale reaches the occupied branch's
switch' gives 0.95 (91% within 10%), and we name it as the rule for a future registered test."

- Do not say: that L3 passes, or that this diagnostic is registered.
