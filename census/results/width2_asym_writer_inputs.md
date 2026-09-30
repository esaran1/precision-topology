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
| (registration) | The author's five changes (2026-09-30); D re-piloted from its branch-point start; code, tests, text and frozen inputs hashed (`registration.sha256`). |
| (stamp) | OpenTimestamps proof of the registration commit. |
| (predictions) | All runs trained to their budgets with no gap or placement of the state evaluated after release; `predictions.csv` and every path hash committed, together with §2 of this file. |
| (observation) | `observe`: hashes asserted; gates first; every-step detection; scored (§3). |

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

## 3. What occurred

(Written after the observation; §2 was committed with the predictions, before it.)
