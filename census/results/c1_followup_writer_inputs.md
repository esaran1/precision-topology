# Writer inputs: follow-up registered test of c₁ (Track 3)

Registration: `results/c1_followup_registration.md` (commit `e233ef5`, 2026-09-25 22:18 EDT, before any added
certificate). Result: commit `aa26bec`; checks `ae6a0ce` and the final commit. Producer: `src/c1_followup.py`. Summary JSON:
`results/c1_followup_summary.json` (scores: `c1_followup_scores.csv`; brackets: `c1_followup_finite.csv`,
evaluations: `c1_followup_finite_evaluations.csv`; independent checks: `c1_followup_checks.json`).

## 1. What was committed before computing (`e233ef5`)

- **Added a**: 1.08, 1.09, 1.10, 1.11, 1.12.
- **Bracket procedure**: the registered one, unchanged.
  - Start at A*/ε^{3/2}, step outward by 1%, bisect to 2e−4, stop at the first unresolved midpoint.
  - `conditional_certified.evaluate`, tolerance 1e−7 tightened to 1e−9.
- **Ĝ(a)**: the rescaled certified search, Amendment 2's procedure.
- **Estimator**: `first_order._feasible_c1`, unchanged. Its inputs are the four original registered brackets
  (frozen, SHA-256) plus the added ones.
- **Validity and criterion**:
  - valid iff the feasible width is ≤ 0.1, otherwise INVALID/UNRESOLVED;
  - PASS iff C meets the derived c₁ ∈ [0.2852300, 0.2852303];
  - otherwise FAIL, including when C is empty.
- **Drop rule**: an added a that does not certify is dropped.
- **Checker rule**: an a whose checked endpoint certificate fails the independent checker is removed, and the
  verdict is recomputed without it.
- **Tests**: the scoring rule and the validity check were exercised on constructed pass, fail, no-law and
  invalid cases, and the original four alone reproduce width 0.10498. The tests passed before any added
  certificate was computed.

## 2. Why these ε (Step 1; `c1_followup_model.json`, `c1_followup_timing_a1.0550.json`)

- **The bracket's resolution is bounded below, whatever ε is.**
  - At the switch, the certified minima m₋ and m₊ meet tangentially: m₋ − m₊ ≈ 0.0215·d|d|, with
    d = s/s* − 1.
  - At tolerance 1e−9 a status therefore resolves only for |d| ≳ 1.75e−4.
  - So the smallest bracket the registered bisection can reach is 0.01/16 = 6.25e−4 relative in s. The
    registered 2e−4 target is unreachable.
- **Small ε cannot reach the design goal.** With that floor, added points at ε ≤ 0.04 cannot bring the
  feasible width below 0.05: Monte Carlo median 0.069, P(< 0.05) = 0.15.
- **ε ≈ 0.1 can.** There the ±ε³ allowance is still small relative to the signal: the projection for
  ε = 0.08–0.12 had median 0.042 and max 0.045.
- **The law holds at these ε.** The two-term law with the small-ε c₂ (−0.13) predicts R(1.30) = 0.21319,
  inside the certified (0.21310, 0.21364]. So ε ≤ 0.12 lies well inside the |c₃| ≤ 1 allowance.
- **Timing**: one full certified threshold at a = 1.055 (not in the set, not used) took 78 s at 0.75 GB.

## 3. The certificates

All five added a certified, so none was dropped.

| a | s bracket | rel. width | Ĝ(a) | R_glob(a) interval | evaluations | wall s |
|---|---|---:|---|---|---:|---:|
| 1.08 | [31.837918, 31.91751275] | 2.50e−3 | [0.01273485, 0.01273485] | [0.202725, 0.203232] | 10 | 69 |
| 1.09 | [26.865283375, 26.8819595] | 6.21e−4 | [0.01514250, 0.01514251] | [0.203404, 0.203530] | 12 | 60 |
| 1.10 | [23.08107775, 23.0954585] | 6.23e−4 | [0.01767344, 0.01767345] | [0.203961, 0.204088] | 13 | 65 |
| 1.11 | [20.118496, 20.143426] | 1.24e−3 | [0.02031933, 0.02031935] | [0.204397, 0.204651] | 12 | 61 |
| 1.12 | [17.7671015, 17.778150688] | 6.22e−4 | [0.02307296, 0.02307297] | [0.204970, 0.205097] | 14 | 61 |

- **Brackets**:
  - Every bracket stopped at an unresolved midpoint, as the error model predicts.
  - Three reached the model's minimum width, 6.2e−4.
  - Every evaluation converged.
- **Ĝ**: the relative width is ≤ 5.7e−7 at every a, and every exclusion closed.
- **Compute**: one process at nice 15, peak memory below 1 GB, about 5 min in total.

**Independent check** (`verify_certificates.check_finite`, Arb, one worker, largest a first; about 23 min per
certificate): **3 of the 10 bracket-end certificates were independently checked, and all 3 pass.**
- **Passed**: `c1f_a1.12_hi` ('plus'), `c1f_a1.12_lo` ('minus') and `c1f_a1.11_hi` ('plus').
  - Every sub-check is true: hashes, lemma W, tiling/coverage, the winning point in its region, the losing
    claims above U, and every leaf claim verified.
  - Check times were 1375 s, 1305 s and 1720 s.
- **Not checked** (stopped at 23:59 EDT on the coordinator's wrap-up instruction):
  - `c1f_a1.11_lo` was exported and hashed in the manifest; its check was interrupted.
  - `c1f_a1.10_hi/lo`, `c1f_a1.09_hi/lo` and `c1f_a1.08_hi/lo` were not exported.
  - These are search-certified only.
- **No a was excluded by the checker rule.**

**Ĝ(a)** from the rescaled search has no independent checker. It is search-certified only.

## 4. Fitted interval and verdict

- **Feasible set** (original four + five added, registered estimator): **C = [0.26531, 0.30687], width 0.0416.**
  - The width is valid (≤ 0.1), and the design goal of < 0.05 is met.
- **The derived c₁** ∈ [0.2852300, 0.2852303] lies inside C.
- **Registered follow-up verdict: PASS.**
- **Competing values**: 0.49 (the earlier incomplete prediction) and 0.662 (the switch shift alone) are
  excluded above; −0.377 (the K correction alone) and 0 (no first-order term) are excluded below.
- **The original test** (`first_order_prediction.md`) stays **INCONCLUSIVE** as registered (width 0.105).
  This is a follow-up registered after it.
- **For comparison only** (no verdict attaches): the supplementary branch-root set is [0.2846, 0.2859],
  inside C.

## 5. Say / Do not say

**PASS (this outcome)**
- Say: "In a follow-up registered after the original test was inconclusive, we added certified thresholds at
  a = 1.08–1.12 with the registered bracket procedure. The unchanged feasible-set estimator then gives
  c₁ ∈ [0.265, 0.307] (width 0.042 ≤ 0.1), which contains the derived c₁ = 0.28523 and excludes 0, 0.49,
  −0.377 and 0.662."
- Say: "The original registered test remains inconclusive (width 0.105 > 0.1). The follow-up needed ε up to
  0.12, because at tolerance 1e−9 the bracket procedure cannot resolve the switch below a relative width of
  6e−4."
- Do not say: "the original c₁ test passed"; "c₁ is measured to six digits" (the fitted interval is 0.042 wide;
  six digits is the derivation); "confirmed at small ε ≤ 0.04" (the decisive points are at ε = 0.08–0.12,
  within an O(ε³) allowance |c₃| ≤ 1).
- Do not say "all certificates independently checked" unless §3 lists all ten as checked. Ĝ from the rescaled
  search is not independently checked.

**FAIL** (not this outcome)
- It would have read: "a certified follow-up excludes the derived c₁, or no two-term law with |c₃| ≤ 1 fits". At
  ε ≤ 0.12 that would implicate either c₁ or the higher-order allowance.

**INVALID/UNRESOLVED** (width > 0.1; not this outcome)
- It would have read: "the follow-up was too coarse to test c₁". It would not be reported as a pass.

**Not run** (not this outcome)
- It would have read: "the follow-up was not run; the c₁ test remains inconclusive".
