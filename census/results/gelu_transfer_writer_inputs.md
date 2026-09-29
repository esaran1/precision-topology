# GELU-T: the lag law at GELU under free SGD from a declared start — writer inputs

**REGISTERED** (`results/gelu_transfer_registration.md`). Design: `results/designs/GELU_transfer_design.md`
(author-approved 2026-09-29; draft 5e6323f). Code: `src/gelu_transfer.py`. Outputs: `results/gelu_transfer/`. Ledger
block: `verify_ledger.gelu_transfer_checks` (every number below).

## 1. Order of events

| commit | content |
|---|---|
| `4ecc97d` | Approved one-page design. |
| `9b2be59`, `9f39328` | Implementation with tests of every decision rule; registration text (not yet registered). |
| `09e2fea`, `29b0a7e`, `7027f75` | Pre-registration computations (landscape, frozen inputs, pilot on pilot seeds only); exploration on the population and pilot seeds; the author's decisions of 2026-09-29 implemented. No registered seed trained. |
| `736b6bf` | **Registration.** Code, tests, text and frozen inputs hashed (`registration.sha256`, 17 files). No registered run trained. |
| `d7996d7` | OpenTimestamps proof of `736b6bf` (`registration_stamp.txt` + `.ots`). |
| `8d39e75` | 160 runs (2 arms × 80 seeds) trained 40,000 steps after release with no gap or placement evaluated after release. Predictions committed; `predictions.csv` SHA-256 `218309c59fae3d921deb0e5ef42736705029051eef9edb6dd8f63ae70ff74044`; each path hashed there. |
| this commit | `observe`: asserted the registration hashes, the committed predictions hash and all 160 path hashes; gates first; then every-step detection; scored. |

**Fixed after the pilot, before registration** (pilot seeds 876,900–876,909 only, disjoint from the registered seeds;
registration §12–§13; the author was informed):
- V7's window (s_t ≥ 0.8·s_switch,branch). On the pilot, max χ_t over the whole path had q90 / median 0.274 / 0.199,
  with the maximum at the first step after release; over the window, 0.046 / 0.036.
- The follow check's state condition (Newton condition only). On the pilot the state trailed its branch by 0.012–0.028
  (sup) at 0.8·s_switch, and 20 of 20 pilot runs followed their branch.
- The primary gate ignores G during the hold (author's decision after exploration).

**κ's sign is physical** (registration §10a). It is invariant under the mirror coordinates, the sign of G and the
direction of s. Recomputed from scratch on the mirrored sample, κ equals the frozen value exactly in 180 of 180 copies,
including the 13 negative ones (4 target copies, 9 mirror copies; κ from −0.168 to −0.004). A negative κ predicts a
crossing before the switch, and the sign is part of what is tested.

## 2. Say / Do not say for every possible outcome (written before the observation was run)

The same table applies to each arm. The primary (random-start) arm carries the headline; the branch-point arm is a
mechanism control and is never reported as the headline.

| outcome (per arm) | Say | Do not say |
|---|---|---|
| **Gate fails** (primary: < 64 of 80 on a branch at release; control: < 72 of 80 on target, or any run with G > 0 in the hold) | "The registered gate failed: only k of 80 runs reached (the target copy of) their own-sample small-scale branch at release, so the test is UNRESOLVED for this arm; no crossing was evaluated." | That the lag law failed or passed; any L1–L5 number (none were computed). |
| **UNRESOLVED (validity)** | "The registered validity condition Vn failed (value, bound); L1–L5 are UNRESOLVED and say nothing about the law." Name every failed condition with its number. | That the law passed or failed; the L1–L5 statistics as if they were verdicts (they may be given only as POST HOC, labelled). |
| **UNRESOLVED (a criterion not computable)** | "Criterion Ln could not be computed (reason); the arm's outcome is UNRESOLVED." | A verdict for that criterion. |
| **PASS** (all of L1–L5) | Primary: "Registered before training, with no fitted constant, the lag law predicts where free SGD crosses at GELU from a random start: median observed/predicted lag x (L1), closed form y (L2), per-run Spearman z (L3); the prediction beats the population switch (L4) and a lag-free prediction at the run's own branch switch (L5)." Control: the same, "from a start on the population branch point (mechanism control)". Give n scored, the on-branch and follow counts, the neither count. | "Proved for GELU"; "holds for Adam"; "holds at any learning rate" (only η = 0.03 was tested; the invariance is a theorem for fixed-P GD and registered only for Adam on the sine family); that negative-κ runs were excluded (they were scored). |
| **L1 FAIL** (alone or with others) | "The trajectory-integrated prediction missed the typical lag: median observed/predicted x, outside [0.90, 1.10]." | That the law holds in magnitude. |
| **L2 FAIL** | "The slaved closed form κχ missed the typical lag: median x, outside [0.80, 1.20]" (and, if L1 passed: "the trajectory-integrated recursion did not"). | That κχ is an accurate one-number summary here. |
| **L3 FAIL** | "The law gets the typical lag but not which runs lag more: Spearman x < 0.5." | That per-run lags are predicted. |
| **L4 FAIL** | "Measured against the population switch s_glob, the per-run prediction is not closer to the observed crossing (mean D x, 95% interval [lo, hi])." | That the own-branch switch carries the sample variation. |
| **L5 FAIL** | "Against a crossing exactly at the run's own branch switch (no lag), the lag prediction is not resolved: mean D x, 95% interval [lo, hi] (upper end ≥ 0)." | That a lag was detected or predicted beyond the switch itself. |
| **Sensitivity analysis (primary; DESCRIPTIVE)** | "Excluding the k runs with G > 0 at initialisation or in the hold (a registered descriptive check, not a criterion) gives …" | That it changes or confirms the registered verdict; it is reported beside it only. |

Any failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 3. What occurred: PASS in both arms (registered; `results/gelu_transfer/scores.json`)

ρ = 1 (pilot rule inactive: pilot q90 of κχ 0.0037).

**Gates.**
- Primary (random start): 80 of 80 on a branch at release (38 target, 42 mirror), 0 on neither. PASS (≥ 64). 6 runs
  had G > 0 at initialisation or in the hold (2 placed at initialisation); they are scored and flagged.
- Control (branch-point start): 80 of 80 on target; 0 with G > 0 in the hold. PASS.

**Counts.** In each arm: 80 crossed; 80 followed their release branch at 0.8·s_switch (0 not following; no follow check
at or after a crossing); 80 scored. Runs with κ_seed < 0: 8 (primary), 2 (control), all scored.

**Validity (all seven hold in both arms).**

| | primary | control | bound |
|---|---|---|---|
| V1 scored runs | 80 | 80 | ≥ 60 |
| V2 t_sw before crossing, κ_seed > 0 | 67 of 72 (0.931) | 73 of 78 (0.936) | ≥ 0.90 |
| V3 η·λ_min ≤ 0.5 | all | all | ≥ 80% |
| V4 median predicted lag (steps) | 40.6 | 42.6 | ≥ 10 |
| V5 q90 of κχ at t_sw | 0.0039 | 0.0038 | ≤ 0.1 |
| V6 median χ at t_sw (pilot) | 0.0222 (0.0224) | 0.0208 (0.0230) | within 30% |
| V7 q90 of max χ_t, s ≥ 0.8·s_switch | 0.060 | 0.053 | ≤ 0.25 |

**Criteria.**

| criterion | primary (random start) | control (branch-point start) | verdict |
|---|---|---|---|
| **L1** median r_obs/r_traj ∈ [0.90, 1.10] | 1.000 | 1.000 | PASS, PASS |
| **L2** median r_obs/r_cf ∈ [0.80, 1.20] | 0.977 | 0.985 | PASS, PASS |
| **L3** Spearman(r_traj, r_obs) ≥ 0.5 | 0.999 | 0.999 | PASS, PASS |
| **L4** mean D vs s_glob, 95% interval upper end < 0 | −0.104 [−0.121, −0.088] | −0.103 [−0.119, −0.087] | PASS, PASS |
| **L5** mean D vs s_switch,branch, upper end < 0 | −0.0031 [−0.0036, −0.0026] | −0.0026 [−0.0029, −0.0023] | PASS, PASS |

**Outcome: PASS in the primary arm (the headline) and PASS in the mechanism control.** The Say line of the PASS row
applies, with x = 1.000, y = 0.977, z = 0.999 (primary).

**Sensitivity analysis (primary; registered, DESCRIPTIVE, not a criterion).** Excluding the 6 runs with G > 0 at
initialisation or in the hold: 74 scored, all validity conditions hold; L1 1.000, L2 0.977, L3 0.999, L4 −0.100
[−0.118, −0.083], L5 −0.0032 [−0.0037, −0.0027]; PASS.

**Descriptive (from the committed predictions and the observed crossings; not criteria).**
- Median observed lag r_obs: 0.00275 (primary), 0.00290 (control); median steps from t_sw to the crossing 39 and 41.5.
- The recursion's predicted crossing step equals the observed step in 36 of 80 runs (primary) and 42 of 80 (control);
  within 1 step in 60 and 65; within 3 steps in 68 and 75. The observed step is never after the predicted one (largest
  gaps 191 and 19 steps).
- Every negative-κ run was predicted early (r_traj < 0) and crossed before its switch and before t_sw (8 of 8 primary,
  2 of 2 control).
- In 8 seeds the random-start hold reached the same release point as the branch-point start bit for bit (152 distinct
  paths of 160), so the arms are not independent samples.

## 4. POST HOC readings (not registered; beside the verdicts)

- **Why the observed step is never after the predicted one (POST HOC).** The prediction uses the exact-extrema gap on
  the continuous windows; the observed crossing uses `phase2b_ordering.state` on the dense windows. For the same
  parameters the dense-window gap is never below the exact one (a grid minimum over O is ≥ the continuum minimum and a
  grid maximum over I is ≤ the continuum maximum), so the dense detector can fire at or before the exact one. This is
  a candidate explanation, not checked here.
- **Negative-κ runs lag by more than predicted (POST HOC).** In those runs |r_obs| exceeds |r_traj| (ratios above 1);
  they are few (8 and 2) and do not move the medians. Not analysed further.

## 5. Scope and caveats (for the writer)

- One activation (GELU), width 1, full-batch SGD at η = 0.03 only. The η-invariance of the lag is a theorem for fixed-P
  gradient descent (Corollary L3) and registered evidence only for free Adam on the sine family (R4). No registered
  SGD or GELU invariance result exists.
- The lag itself is small (median r_obs about 0.3%, about 40 steps). L5 is what shows it is resolved against a
  lag-free prediction; L4 mostly reflects the sample variation of the own-branch switch.
- V7's window and the follow check's state condition were fixed after the pilot (pilot seeds only), before registration.
- The primary gate ignores G during the hold (author's decision before registration); the 6 affected runs are scored
  and the sensitivity analysis without them gives the same outcome.
