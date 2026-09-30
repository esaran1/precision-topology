# W2-A: the lag law at width 2 on the asymmetric windows — writer inputs

**REGISTERED** (`results/width2_asym_registration.md`). The design is `results/designs/width2_asym_design.md`,
approved by the author 2026-09-29 with changes A–D, plus the author's five changes of 2026-09-30, made after the pilot
on pilot seeds only (registration §0). Code: `src/width2_asym.py`. Outputs: `results/width2_asym/`.

## 1. Order of events

| commit | content |
|---|---|
| `907e225` | Approved one-page design (changes A–D). Draft with the width-2 background: `ccce4fa`. |
| `596942c` | Implementation with tests of every decision rule (not registered). |
| `fd119b2` | Registration text, not yet registered; committed before any pilot run. |
| `b3f483d` | Pre-registration computations: landscape, frozen inputs, first pilot (pilot seeds only). Arm D's random start landed on D in 3/10: STOP for the author. |
| `28b2432` | The author's five changes (2026-09-30); D re-piloted from its branch-point start; code, tests, text and frozen inputs hashed (`registration.sha256`). |
| `7800b48` | OpenTimestamps proof of the registration commit. |
| `2c630f2` | 440 runs (T 200, D 120, T′ 120) trained to their budgets, with no gap or placement of the state evaluated after release. `predictions.csv` (SHA-256 `e78d85c80ad7287db698fcf299bb22854fa811d620f382bc64a2db531bb409ab`) and every path hash committed, together with §2 of this file. |
| this commit | `observe`: asserted the registration hashes, the committed predictions hash and all 440 path hashes; gates first; every-step detection; scored (§3). |

## 2. Say / Do not say for every possible outcome, per arm (written before the observation was run)

**Rules for every arm**
- Each arm has its own verdict.
  - T (random start, 200 seeds) is the headline.
  - D (branch-point start on the duplicate, width-1-function branch) is the control.
  - T′ (branch-point start on the two-unit branch with κ < 0) tests the sign of the lag.
- D and T′ are never reported as the headline. No arm's result is transferred to another arm.
- **Changes after the pilot.** Every sentence that states a W2-A verdict also says that five registration details were
  changed after the pilot, on pilot seeds only, before any registered seed was touched (registration §0): D's start
  and gate, T's 200 seeds, V4/V5 on magnitudes, D′'s L4 comparator, and κ at the actual winding pair.
- **Validity failures.** Several validity conditions can fail together. Name each one with its value and bound. Any
  UNRESOLVED leaves the arm UNRESOLVED.
- **Failures stay failures.** POST HOC readings may stand beside a verdict, labelled POST HOC, and never replace it.

### Arm T (headline)

| outcome | Say | Do not say |
|---|---|---|
| **Gate fails** (< 60 of 200 runs on T at release) | "T (headline): the registered gate failed ({value}); the arm is UNRESOLVED and no crossing was evaluated." | That the lag law failed or passed for this arm; any L1–L5 number (none were computed). |
| **V1 fails** (< 60 scored runs) | "T (headline): only n runs were scored (on-branch, following, crossing, with predictions; bound 60), so L1–L5 are UNRESOLVED." Give the counts for not following, not crossing and no prediction. | A verdict; L1–L5 values as results (POST HOC only). |
| **V2 fails** (t_sw precedes the crossing in < 90% of κ > 0 crossing runs) | "T (headline): in only x% of the runs predicted to cross late did the lag-free switch come before the crossing (bound 90%); UNRESOLVED." | That the runs crossed early 'as predicted'; a verdict. |
| **V3 fails** (η·λ_min ≤ 0.5 in < 80% of scored runs) | "T (headline): outside the registered tracking regime (x% with η·λ_min ≤ 0.5); UNRESOLVED." | A verdict. |
| **V4 fails** (median \|predicted lag\| < 10 steps) | "T (headline): the median magnitude of the predicted lag was x steps (bound 10), too small to resolve; UNRESOLVED." | That the lag is absent or zero; a verdict. |
| **V5 fails** (q90 \|κχ\| > 0.1) | "T (headline): the lag was outside the linear regime (q90 of \|κχ\| = x > 0.1); UNRESOLVED." | A verdict; that the law fails at large κχ (not tested here). |
| **V6 fails** (median χ off the pilot's by > 30%) | "T (headline): the sweep rate differed from the pilot's (median χ x vs pilot y); UNRESOLVED." | A verdict. |
| **V7 fails** (q90 of the window max χ_t > 0.25, or undefined) | "T (headline): the runs did not track their branch near the switch (q90 of max χ_t = x > 0.25); UNRESOLVED." | A verdict. |
| **UNRESOLVED, a criterion not computable** (e.g. Spearman NaN, or < 2 runs for a bootstrap) | "T (headline): criterion Ln could not be computed (reason); the arm is UNRESOLVED." | A verdict for Ln. |
| **L5 UNRESOLVED** (fewer than half the scored runs, or fewer than 2, have \|t_pred − t_sw\| ≥ 6 steps) | "T (headline): only m of n scored runs had a predicted lag of at least 6 steps (the registered resolution N_min), so L5 is UNRESOLVED, and so is the arm." | That the lag was resolved; an L5 verdict. |
| **PASS** (L1–L5) | "T (headline): registered before training, with no fitted constant. At width 2 on the asymmetric windows, full-batch SGD from a random start on the two-unit branch type T crosses at that branch's own switch plus the predicted lag: median observed/predicted x (L1), closed form y (L2), per-run Spearman z (L3). The prediction beats the branch's population switch (L4) and a lag-free crossing at the run's own switch (L5). The switch is read from each run's own output path, and the prediction is conditional on that path." Give n scored, k of 200 on T at release, and the neither and T′ counts. | That T2-3's failure is overturned or 'explained' (T2-3 stays FAIL; this is a different, slow-output design at η 0.03, ρ 2⁻¹⁰). That the switch was predicted in advance (it is read from the run's own v path; disclosure C). That the law holds for Adam or at any learning rate. That random starts usually reach T (98 of 200 did). |
| **L1 FAIL** | "T (headline): the trajectory-integrated prediction missed the typical lag: median observed/predicted x, outside [0.90, 1.10]." | That the law holds in magnitude for this arm. |
| **L2 FAIL** | "T (headline): the slaved closed form κχ missed the typical lag: median x, outside [0.80, 1.20]." Add "(the recursion did not)" if L1 passed. | That κχ is an accurate one-number summary here. |
| **L3 FAIL** | "T (headline): Spearman x < 0.5; the typical lag is right but not which runs lag more." | That per-run lags are predicted. |
| **L4 FAIL** | "T (headline): against the branch's population switch (bisected: T 0.481586, T′ 0.327167, D 5.079637, D′ 4.970592), the per-run prediction was not closer: mean D x, 95% interval [lo, hi]." | That the own-sample switch carries the sample variation. |
| **L5 FAIL** | "T (headline): against a lag-free crossing at the run's own switch, the predicted lag was not resolved: mean D x, [lo, hi], over m runs with \|t_pred − t_sw\| ≥ 6." | That a lag beyond the switch was detected or predicted. |
| **Several criteria FAIL** | "T (headline): FAIL on Ln, Lm (values as above)." Name each failing criterion. | A partial pass; 'mostly passed'. |
| **Sensitivity analysis (T only; DESCRIPTIVE)** | "Excluding the k runs with G > 0 at the start or in the hold (a registered descriptive check, not a criterion) gives …" | That it changes or confirms the registered verdict. |

### Arm D (control)

| outcome | Say | Do not say |
|---|---|---|
| **Gate fails** (< 108/120 on D or D′ at release, or any run with G > 0 at the start or in the hold) | "D (control): the registered gate failed ({value}); the arm is UNRESOLVED and no crossing was evaluated." | That the lag law failed or passed for this arm; any L1–L5 number (none were computed). |
| **V1 fails** (< 60 scored runs) | "D (control): only n runs were scored (on-branch, following, crossing, with predictions; bound 60), so L1–L5 are UNRESOLVED." Give the counts for not following, not crossing and no prediction. | A verdict; L1–L5 values as results (POST HOC only). |
| **V2 fails** (t_sw precedes the crossing in < 90% of κ > 0 crossing runs) | "D (control): in only x% of the runs predicted to cross late did the lag-free switch come before the crossing (bound 90%); UNRESOLVED." | That the runs crossed early 'as predicted'; a verdict. |
| **V3 fails** (η·λ_min ≤ 0.5 in < 80% of scored runs) | "D (control): outside the registered tracking regime (x% with η·λ_min ≤ 0.5); UNRESOLVED." | A verdict. |
| **V4 fails** (median \|predicted lag\| < 10 steps) | "D (control): the median magnitude of the predicted lag was x steps (bound 10), too small to resolve; UNRESOLVED." | That the lag is absent or zero; a verdict. |
| **V5 fails** (q90 \|κχ\| > 0.1) | "D (control): the lag was outside the linear regime (q90 of \|κχ\| = x > 0.1); UNRESOLVED." | A verdict; that the law fails at large κχ (not tested here). |
| **V6 fails** (median χ off the pilot's by > 30%) | "D (control): the sweep rate differed from the pilot's (median χ x vs pilot y); UNRESOLVED." | A verdict. |
| **V7 fails** (q90 of the window max χ_t > 0.25, or undefined) | "D (control): the runs did not track their branch near the switch (q90 of max χ_t = x > 0.25); UNRESOLVED." | A verdict. |
| **UNRESOLVED, a criterion not computable** (e.g. Spearman NaN, or < 2 runs for a bootstrap) | "D (control): criterion Ln could not be computed (reason); the arm is UNRESOLVED." | A verdict for Ln. |
| **L5 UNRESOLVED** (fewer than half the scored runs, or fewer than 2, have \|t_pred − t_sw\| ≥ 6 steps) | "D (control): only m of n scored runs had a predicted lag of at least 6 steps (the registered resolution N_min), so L5 is UNRESOLVED, and so is the arm." | That the lag was resolved; an L5 verdict. |
| **PASS** (L1–L5) | "D (control): from a start on the duplicate (width-1-function) branch point, the lag law predicts where free SGD crosses at width 2: x, y, z; L4 and L5 pass." Give the counts, and say it is the control with a branch-point start. | That this is the headline. That random starts reach D (the first pilot gave 3/10; D was moved to a branch-point start after the pilot). |
| **L1 FAIL** | "D (control): the trajectory-integrated prediction missed the typical lag: median observed/predicted x, outside [0.90, 1.10]." | That the law holds in magnitude for this arm. |
| **L2 FAIL** | "D (control): the slaved closed form κχ missed the typical lag: median x, outside [0.80, 1.20]." Add "(the recursion did not)" if L1 passed. | That κχ is an accurate one-number summary here. |
| **L3 FAIL** | "D (control): Spearman x < 0.5; the typical lag is right but not which runs lag more." | That per-run lags are predicted. |
| **L4 FAIL** | "D (control): against the branch's population switch (bisected: T 0.481586, T′ 0.327167, D 5.079637, D′ 4.970592), the per-run prediction was not closer: mean D x, 95% interval [lo, hi]." | That the own-sample switch carries the sample variation. |
| **L5 FAIL** | "D (control): against a lag-free crossing at the run's own switch, the predicted lag was not resolved: mean D x, [lo, hi], over m runs with \|t_pred − t_sw\| ≥ 6." | That a lag beyond the switch was detected or predicted. |
| **Several criteria FAIL** | "D (control): FAIL on Ln, Lm (values as above)." Name each failing criterion. | A partial pass; 'mostly passed'. |

### Arm T′

| outcome | Say | Do not say |
|---|---|---|
| **Gate fails** (< 108/120 on T′ at release, or any run with G > 0 at the start or in the hold) | "T′: the registered gate failed ({value}); the arm is UNRESOLVED and no crossing was evaluated." | That the lag law failed or passed for this arm; any L1–L5 number (none were computed). |
| **V1 fails** (< 60 scored runs) | "T′: only n runs were scored (on-branch, following, crossing, with predictions; bound 60), so L1–L5 are UNRESOLVED." Give the counts for not following, not crossing and no prediction. | A verdict; L1–L5 values as results (POST HOC only). |
| **V2 fails** (t_sw precedes the crossing in < 90% of κ > 0 crossing runs) | "T′: in only x% of the runs predicted to cross late did the lag-free switch come before the crossing (bound 90%); UNRESOLVED." | That the runs crossed early 'as predicted'; a verdict. |
| **V3 fails** (η·λ_min ≤ 0.5 in < 80% of scored runs) | "T′: outside the registered tracking regime (x% with η·λ_min ≤ 0.5); UNRESOLVED." | A verdict. |
| **V4 fails** (median \|predicted lag\| < 10 steps) | "T′: the median magnitude of the predicted lag was x steps (bound 10), too small to resolve; UNRESOLVED." | That the lag is absent or zero; a verdict. |
| **V5 fails** (q90 \|κχ\| > 0.1) | "T′: the lag was outside the linear regime (q90 of \|κχ\| = x > 0.1); UNRESOLVED." | A verdict; that the law fails at large κχ (not tested here). |
| **V6 fails** (median χ off the pilot's by > 30%) | "T′: the sweep rate differed from the pilot's (median χ x vs pilot y); UNRESOLVED." | A verdict. |
| **V7 fails** (q90 of the window max χ_t > 0.25, or undefined) | "T′: the runs did not track their branch near the switch (q90 of max χ_t = x > 0.25); UNRESOLVED." | A verdict. |
| **UNRESOLVED, a criterion not computable** (e.g. Spearman NaN, or < 2 runs for a bootstrap) | "T′: criterion Ln could not be computed (reason); the arm is UNRESOLVED." | A verdict for Ln. |
| **L5 UNRESOLVED** (fewer than half the scored runs, or fewer than 2, have \|t_pred − t_sw\| ≥ 6 steps) | "T′: only m of n scored runs had a predicted lag of at least 6 steps (the registered resolution N_min), so L5 is UNRESOLVED, and so is the arm." | That the lag was resolved; an L5 verdict. |
| **PASS** (L1–L5) | "T′: on a branch with negative κ, the law predicts an early crossing, and runs cross early by the predicted amount: signed median observed/predicted x (L1), y (L2), Spearman z (L3); L4 and L5 pass." Give the counts, and say it is a branch-point start. | That T′ is the headline. That random starts reach T′. That the negative-κ explanation was found on registered seeds (it came from exploratory runs on the population and used T2-3 seeds; clarification 1). |
| **L1 FAIL** | "T′: the trajectory-integrated prediction missed the typical lag: median observed/predicted x, outside [0.90, 1.10]." | That the law holds in magnitude for this arm. |
| **L2 FAIL** | "T′: the slaved closed form κχ missed the typical lag: median x, outside [0.80, 1.20]." Add "(the recursion did not)" if L1 passed. | That κχ is an accurate one-number summary here. |
| **L3 FAIL** | "T′: Spearman x < 0.5; the typical lag is right but not which runs lag more." | That per-run lags are predicted. |
| **L4 FAIL** | "T′: against the branch's population switch (bisected: T 0.481586, T′ 0.327167, D 5.079637, D′ 4.970592), the per-run prediction was not closer: mean D x, 95% interval [lo, hi]." | That the own-sample switch carries the sample variation. |
| **L5 FAIL** | "T′: against a lag-free crossing at the run's own switch, the predicted lag was not resolved: mean D x, [lo, hi], over m runs with \|t_pred − t_sw\| ≥ 6." | That a lag beyond the switch was detected or predicted. |
| **Several criteria FAIL** | "T′: FAIL on Ln, Lm (values as above)." Name each failing criterion. | A partial pass; 'mostly passed'. |
| **Wrong sign** (T′ runs cross AFTER their switch although predicted early; L1/L2 medians negative or L3 negative) | "T′: the observed lags had the wrong sign in k of n scored runs (median signed ratio x); the negative-κ prediction failed." This comes with the L1/L2 (and L3) FAIL rows. | That the magnitude was right; that the sign is a convention (it is physical, registration §7). |
| **V2 (information)** | V2 is vacuous for T′ (no κ > 0 runs), as GELU-T's rule defines it. Say so if asked. | That V2 checked T′'s timing. |

## 3. What occurred: PASS in all three arms (registered; `results/width2_asym/scores.json`)

ρ (from the pilot): T 2⁻¹⁰, D 1, T′ 2⁻¹².

**Gates**
- **T:** 98 of 200 on T at release. PASS (≥ 60).
  - The rest: 15 on T′ and 87 on neither (Newton landed on D 58, not accepted 24, D′ 3, T 2 off the frozen copy
    point). These are counted and not scored.
  - 1 run had G > 0 in the hold (seed 884,192, one hold state). It is on T, scored and flagged.
- **D:** 120 of 120 on D, none on D′, no run with G > 0 in the hold. PASS (≥ 108).
- **T′:** 120 of 120 on T′, no run with G > 0 in the hold. PASS (≥ 108).
- Every on-branch run released at winding (0, 0), fixed at release.

**Counts**

| | T | D | T′ |
|---|---|---|---|
| on-branch at release | 98 | 120 | 120 |
| neither / other copy | 102 | 0 | 0 |
| unconverged (Newton not accepted) | 24 (inside "neither") | 0 | 0 |
| crossed (all runs) | 139 | 120 | 120 |
| on-branch and crossed | 98 | 120 | 120 |
| not following their branch | 0 | 0 | 0 |
| scored | 98 | 120 | 120 |
| κ ≤ 0 among scored | 1 | 0 | 120 |
| L5-eligible (\|t_pred − t_sw\| ≥ 6) | 92 of 98 | 120 of 120 | 120 of 120 |

**Validity (all seven hold in every arm)**

| | T | D | T′ | bound |
|---|---|---|---|---|
| V1 scored runs | 98 | 120 | 120 | ≥ 60 |
| V2 t_sw before crossing, κ > 0 | 94 of 97 (0.969) | 120 of 120 | no κ > 0 run (holds) | ≥ 0.90 |
| V3 η·λ_min ≤ 0.5 | all | all | all | ≥ 80% |
| V4 median \|predicted lag\| (steps) | 24.8 | 202.4 | 43.3 | ≥ 10 |
| V5 q90 \|κχ\| at t_sw | 0.00167 | 0.0359 | 0.00142 | ≤ 0.1 |
| V6 median χ at t_sw (pilot) | 0.0284 (0.0259) | 0.00411 (0.00388) | 0.0165 (0.0162) | within 30% |
| V7 q90 window max χ_t | 0.077 | 0.0064 | 0.075 | ≤ 0.25 |

**Criteria**

| criterion | T (headline) | D (control) | T′ |
|---|---|---|---|
| **L1** median r_obs/r_traj ∈ [0.90, 1.10] | 1.000 PASS | 1.021 PASS | 1.000 PASS |
| **L2** median r_obs/r_cf ∈ [0.80, 1.20] | 1.009 PASS | 1.008 PASS | 0.993 PASS |
| **L3** Spearman ≥ 0.5 | 1.000 PASS | 0.997 PASS | 1.000 PASS |
| **L4** mean D vs the branch's population switch, upper end < 0 | −0.109 [−0.125, −0.094] PASS | −0.077 [−0.088, −0.065] PASS | −0.062 [−0.070, −0.054] PASS |
| **L5** mean D vs the own switch (eligible runs), upper end < 0 | −0.00121 [−0.00129, −0.00113] (n 92) PASS | −0.0307 [−0.0314, −0.0300] (n 120) PASS | −0.00088 [−0.00095, −0.00082] (n 120) PASS |

**Outcome: PASS in T (the headline), PASS in D (the control) and PASS in T′.** Each arm's PASS row in §2 applies, with
the numbers above.
- **T:** 98 scored; 98 of 200 on T at release; 15 on T′ and 87 on neither.
- **T′:** every one of the 120 scored runs crossed before its own switch, as predicted. The median signed lag is −43.5
  steps, and r_obs ranges from −0.00196 to −0.00017.
- **The pilot changes:** each Say line carries the registered five-change disclosure (§2 rules).

**Sensitivity analysis (T; registered, DESCRIPTIVE, not a criterion).** Excluding the 1 run with G > 0 in the hold:
97 scored, all validity conditions hold; L1 1.000, L2 1.008, L3 1.000, L4 −0.109 [−0.125, −0.094], L5 −0.00121
[−0.00129, −0.00113]; PASS.

**Descriptive** (from the committed predictions and the observed crossings; not criteria)
- **Median observed lag r_obs:** T 0.00116 (median 24 steps from t_sw to the crossing); D 0.0320 (206 steps); T′
  −0.00084 (−43.5 steps).
- **Predicted against observed crossing step:**
  - T: equal in 97 of 98 runs; 1 step earlier than predicted in 1.
  - T′: equal in 118 of 120; 1 step later in 2.
  - D: the observed step is always 1–6 steps after the predicted one (within 1 in 14, within 3 in 28, of 120). The
    predicted lag there is about 200 steps.
- **T's one scored run with κ < 0** (seed 884,149, κ = −0.0041): predicted and observed to cross 3 steps before its
  switch, at the same step.
- **Off-branch crossings:** 41 of T's 102 runs that were not on T at release crossed during training. They are not
  scored and have no prediction.
- **Undecided enclosure states:** 0 during observation, in every arm.

## 4. POST HOC readings (not registered; beside the verdicts)

- **D's observed crossing always trails the recursion by 1–6 steps** (POST HOC). This is 0.5–3% of its ~200-step lag,
  and it moves L1 to 1.021, inside the band. A candidate reason is D's split directions: the R4 recursion is linear in
  them, and they relax at a rate of their own. Not analysed.

## 5. Scope and caveats (for the writer)

- **Setting.** One activation family (f_a, a = 1.30), width 2, asymmetric windows (Δ = 0.4), full-batch SGD at one
  (η, ρ) per arm: T 0.03 / 2⁻¹⁰, D 0.3 / 1, T′ 0.03 / 2⁻¹². T2-3 (Adam) stays FAIL.
- **Own-path switch (disclosure C).** T's and T′'s lag-free switch is read from each run's own output path, so their
  predictions are conditional on that path. D's switch is frozen.
- **Changes before registration.** Disclosure B: the total-derivative κ was adopted after exploratory runs. The five
  changes of registration §0 were made after the pilot, on pilot seeds only, before any registered seed was touched.
- **Arm starts.** D and T′ are branch-point starts. Only T starts at random, and fewer than half of its runs (98 of
  200) reached T.

## 6. Erratum (found by the ledger after the observation)

- **Registration §14** (hashed in 28b2432) says "All 840 copy points are accepted". The frozen file
  (`frozen_seeds.json`, unchanged and hashed) has **920** copy points, all accepted and of their type: 40 on the
  pilot seeds, 480 on 884,000–884,119, 160 on 884,120–884,199 and 240 on 884,200–884,319.
- The count 840 was an arithmetic slip in the text only. The registration commit message repeats it. No frozen value,
  rule or verdict is affected, and the registration text is left unchanged because it is hashed.
