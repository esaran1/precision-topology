# Registration: follow-up test of the first-order coefficient c₁ (Track 3)

**Written and committed before any certified R_glob(a) at a = 1.08, 1.09, 1.10, 1.11 or 1.12 is computed.**
Date: 2026-09-25, 22:18 EDT. Producer: `src/c1_followup.py` (`model`, `timing`, `run`, `score`, `check`).
Tests: `tests/test_c1_followup.py`. Both are committed with this file.

## Status of the original test

- The original registered test (`first_order_prediction.md`, scored 2026-09-24 03:20) stays **INCONCLUSIVE**.
  - Its feasible set was C = [0.22521, 0.33020], width 0.105, against a registered limit of 0.1.
  - It is not re-scored.
- This is a **follow-up, registered after that result**. It adds certified brackets at further a and scores
  them together with the four original ones, using the registered estimator unchanged.

## Step 1: error model and timing (`c1_followup_model.json`, `c1_followup_timing_a1.0550.json`)

**The error model** comes from the committed a = 1.01–1.04 evaluations (`first_order_finite_evaluations.csv`)
and the supplementary branch-root switch s*.

- **What limits the bracket.** The certified difference m₋ − m₊ is quadratic in the relative offset
  d = s/s* − 1: m₋ − m₊ ≈ k·d|d|, with k = 0.0215 (range 0.0207–0.0224, the same at all four a). The
  minima meet tangentially at a continuous crossing.
  - At tolerance 1e−9, the two certified intervals have half-widths that sum to 6.6e−10, so a status
    resolves only for |d| ≳ √(6.6e−10/0.0215) = **1.75e−4**.
  - Observed: the largest unresolved |d| is 1.71e−4, and the smallest resolved |d| at 1e−9 is 2.63e−4.
- **The smallest attainable bracket.** Bisection starts from a 1% bracket. The smallest relative width in s
  it can attain is 0.01/2⁴ = **6.25e−4**.
  - A 3.1e−4 bracket that contains s* has an endpoint within 1.56e−4 of s*, which cannot resolve.
  - So the registered 2e−4 bisection target is unreachable, whatever a is. The four original a attained
    6.2e−4, 1.2e−3, 6.2e−4 and 2.5e−3.
- **Consequence for small ε.**
  - The O(ε³) allowance is negligible there, so C is set by ~6e−4-wide brackets divided by ε.
  - Projected feasible width with added points at ε ≤ 0.04: grid floor 0.058. Monte Carlo over the attained
    width outcomes, with ε = 0.005, 0.010, …, 0.040: median 0.069, and P(width < 0.05) = 0.15.
  - **No set of added ε ≤ 0.04 brings C below 0.05 with the registered procedure.**
- **Consequence for larger ε.** A point at ε ≈ 0.1 constrains c₁ + εc₂ to about ±(6e−4 + ε³)/ε, which with
  the original four brings C to about 0.044.
  - Registered design ε = 0.08, 0.09, 0.10, 0.11, 0.12 (Monte Carlo, 150 draws): median 0.0416, 90% quantile
    0.0429, max 0.0446, P(< 0.05) = 1.
  - The five points give redundancy against brackets that stop early. A single ε = 0.1 would give
    P(< 0.05) = 0.83.

**Timing** (`c1_followup_timing_a1.0550.json`): one full certified threshold at **a = 1.055**, which is not in
the registered set and not used.

- The registered bracket procedure plus the rescaled certified Ĝ took **78 s wall time** for 9 evaluations
  (2–23 s each), with **0.75 GB peak RSS**, in one process at nice 15. Ĝ took 0.6 s.
- The expected cost of the five added a is minutes, well inside the 01:00 target. Only the optional
  independent check is long: at a = 1.30 it took about 10 min per certificate with 3 workers.

## Procedure (added a = 1.08, 1.09, 1.10, 1.11, 1.12; ε = a − 1)

- **Switch bracket**: the registered procedure, unchanged (`first_order._finite_job`, reproduced step for step in
  `c1_followup.threshold`).
  - `conditional_certified.evaluate` at tolerance 1e−7, tightened to 1e−9.
  - Start at s₀ = round(A*/ε^{3/2}, 6), step outward by 1% until the status flips, then bisect to a
    relative width ≤ 2e−4.
  - Stop at the first unresolved midpoint and use the last resolved bracket.
- **Ĝ(a)**: the certified rescaled search `first_order.ghat_rescaled` (the box branch and bound plus the
  exclusion branch and bound, relative target 1e−6). This is Amendment 2's procedure, applied at every
  added a.
  - Reasons: the registered `ghat_bnb` step exceeded memory at small ε, and the rescaled search agreed
    with `ghat_bnb` at a = 1.02–1.04.
- **R interval**: [s_lo·Ĝ_lo/2, s_hi·Ĝ_hi/2].
- **Compute**: one process (`python -m src.c1_followup run`), nice 15, stopped (not paused) if RSS exceeds
  3 GB. Each evaluation is cached on disk and a restart resumes.
- **Drop rule**: an added a that does not certify is dropped and reported, with no rerun under changed
  settings. Not certifying means any unconverged evaluation, an exclusion that does not close, or a memory
  stop.

## Estimator (unchanged)

`first_order._feasible_c1`, called unchanged, as in the registered test:
- **C** = the set of c₁ for which some R∞ in the certified limit interval [0.19859264, 0.19859265]
  (`first_order_c1.csv`) and some c₂ ∈ ℝ make R∞(1 + c₁ε + c₂ε²) pass through every interval, each widened
  by ±ε³·R∞ (|c₃| ≤ 1).
- **Inputs**: the four original registered brackets (`first_order_finite.csv`, frozen) and the added brackets
  that certified.

## Validity and criterion

- **Valid** iff C has width ≤ 0.1. Otherwise **INVALID/UNRESOLVED**, reported as such and not as a pass.
- **PASS** iff C is valid and meets the derived interval c₁ ∈ [0.2852300, 0.2852303].
- **FAIL** iff C is valid and does not meet it, or if C is empty (no two-term law fits).
- **Design goal**: width < 0.05. This is not a validity condition. If 0.05 ≤ width ≤ 0.1, the verdict stands
  and the width is reported.
- **What a FAIL would mean.** At ε = 0.08–0.12, the ±ε³ allowance (5e−4 to 1.7e−3 of R) is a leading part
  of the tolerance. A FAIL would therefore mean that c₁ is wrong, or that the law needs |c₃| > 1 or
  higher-order terms at ε ≤ 0.12. It would be reported with both readings.
- **Competing values** (reported as excluded or not, as in the original): 0.49, −0.377, 0.662 and 0.
- **Scoring**: `python -m src.c1_followup score` → `c1_followup_scores.csv`, `c1_followup_summary.json`.

## Independent check (as time allows)

- **What is checked**: the status certificate at each added bracket's two ends (s_lo: 'minus', s_hi: 'plus').
  - Exported with `cert_export.finite`; SHA-256 appended to `certificates_manifest.csv`.
  - Checked by `verify_certificates.check_finite` (Arb ball arithmetic, no shared code), with one worker.
  - Order: largest a first.
- **If a checked certificate fails**: that a is excluded, and the verdict recomputed without it is the
  registered verdict. Unchecked certificates are reported as unchecked.
- **Ĝ(a)** from the rescaled search has no independent checker; it is reported as search-certified only.
- **Hard stop 01:30 EDT**: anything not complete is reported as not done.

## Frozen inputs (SHA-256; checked by `tests/test_c1_followup.py`)

| file | SHA-256 |
|---|---|
| `results/first_order_finite.csv` | ff68d364027fbdd1aee5a6f67edf7c76d1a18ac602e6edaf4cf2b10ef5b9c5bb |
| `results/first_order_c1.csv` | e7fb2f5a1373b6182848c8e1b36ad68dbd72dcaa2a83447a7da45c6b154c2397 |
| `results/c1_followup_model.json` | f8bb88aa1e5101cf5111b6870457a540069e1f3c919d1a774a64d578d6ee0926 |
| `results/c1_followup_timing_a1.0550.json` | 5bf17f20d0cbeff242f2cebc1feb2a02e573e8496b19785555ab531930849f60 |

The scoring rule and the validity check were exercised on constructed cases in the tests (pass, fail with the
wrong c₁, fail with no law, invalid-wide, and the original four alone reproducing width 0.10498) before any
added certificate was computed.

## Disclosed prior exposure

- **The supplementary branch-root set** [0.284596, 0.285895] (`first_order_supplementary_scores.csv`) is known.
  It is not a registered result.
- **The timing bracket at a = 1.055** was seen: R ∈ [0.201511, 0.201761], relative width in s 1.24e−3,
  stopped unresolved. It is not used.
- **Design check at a = 1.30.** The certified R_glob(1.30) ∈ (0.21310, 0.21364] (math_note_v2 §5) was used
  to check that a two-term law with the small-ε c₂ remains adequate to ε = 0.3. The design truth model
  R∞(1 + c₁ε − 0.13ε² − 0.016ε³), fitted to the four supplementary points, gives 0.21319 there. This is
  why ε up to 0.12 is within the |c₃| ≤ 1 allowance.
- **Certified Ĝ(1.10)** (`ghat_a1.10` certificate) exists and was not recomputed or looked at for this design.
- No R_glob at a = 1.08–1.12 was computed before this registration.
