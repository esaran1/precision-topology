# Writer inputs, v4 (supersedes WRITER_INPUTS.md where they differ)

Written for the paper's writer. Every item names its registration, producing script and artifact. The v3
handoff (`WRITER_INPUTS.md`, tag `paper-submitted-v3`) remains correct except where this file changes it.

## Status

| block | status | for the 25 September submission? |
|---|---|---|
| 1. Audit of the conditional threshold | **complete**; registered decision rule met; 1d done (below) | yes |
| 2. Proposition 2 made precise | **certified**: localisation, switch, H2′, neighbourhood, solve threshold, and global uniqueness (annulus and solve bracket, four-link chain); the registered c₁ test is running | yes |
| 3. Prospective held-out prediction | **complete; primary criterion PASSED** | yes |
| 4. Fixed-scale dynamics | running under an amended validity check (see below) | if scored and verified in time; otherwise rebuttal |
| 5. Retention curve | queued after Block 4 | if scored and verified in time; otherwise rebuttal |
| 6. Width 2 | design committed for review (`block6_width2_design.md`); nothing run | rebuttal |

## Terminology (all paper text)

Replace "training-free" with **"computed independently of unconstrained training trajectories"**. The
conditional search is itself an optimisation. Done in every paper-facing file of the repository; four
frozen registration documents keep their original wording.

## Block 1 — the conditional threshold is not an artifact of the search

Registered `conditional_audit_prediction.md` (`6b17508`, before any audit computation). Results
`conditional_audit.md`; producers `src/conditional_audit.py`, `src/profiled_bnb.py`,
`src/conditional_certified.py`.

- **Decision rule met; no stop condition fired.**
- **Every candidate retained (1a).** The frozen search was replayed at all 358 scales it evaluates:
  16,468 candidates, and the frozen result reproduced at 358 of 358.
  - No discarded candidate — degenerate, screened out, not lowest, or the exact constant predictor —
    has lower loss than the retained branch, anywhere.
  - Retained minimisers are stationary: gradient norm at most 2.3e-6.
  - The candidate-loss plots (`results/figures/cond_audit_candidates.pdf`) show one continuous branch
    at every a.
- **Objective check (1d)**: each of 50 seeds' own 400-point training sets gives its own certified
  threshold, at a = 1.30, 1.45 and 1.60 (`cond_certified_seeds_summary.csv`).
  - The per-seed median is **1.06×** the quadrature value. The IQR is 0.99–1.12× and the range
    0.87–1.22×.
  - The quadrature value sits at the 30th–32nd percentile. Every seed has one switch.
  - **Bias of the procedure, stated beside the 6%**: evaluations that could not be resolved at tolerance
    1e−9 were treated as unplaced. That biases the affected thresholds **upward** by at most one bisection
    step: ≤ 0.08% / 0.14% / 0.19% of R at a = 1.30 / 1.45 / 1.60, in 20 / 18 / 6 of 50 seeds. This is far
    too small to produce the +6% median excess.
  - **Say**: "the quadrature threshold is the population-level object; individual training sets have
    thresholds spread ±12% (IQR) around a median 6% above it". Whether that spread and offset explain
    the fixed-scale curve width and the free-training offset is tested in S1–S3 (pending).
- **Certified full-range scan at a = 1.30** (`cond_scan_certified_a130.csv`): at every |w₂| from 1.5 to
  11 (step 0.5) the certified global conditional minimiser lies on one continuous branch.
  - Its gap changes sign once, between 4.5 and 5.0.
  - No other basin comes within +0.0031 to +0.0126 of it (certified competitor margins, radius 0.1).
- **Stricter optimisation (1e)**: 10× steps and a gradient-norm stop at 1e-8. It agrees with the
  frozen search at all 12 thresholds, and losses agree to 2e-16.
- **Certified thresholds (1c).**
  - **Method**: exhaustive branch and bound on the profiled conditional loss. The output bias is
    solved exactly (a strictly convex scalar problem). The domain is the full placement domain: b₁ is
    2π-periodic, and |w₁| is bounded analytically.
  - **Every R_glob sign decision is rigorous.** The two certified intervals are separated by at least
    1.3e-8. That is 100× the stated rounding margin, with attained values re-checked in interval
    arithmetic to 1.1e-16.
  - **The switch is a continuous crossing of one basin**, not an exchange: the minimisers on either
    side of G = 0 lie 0.00015–0.0011 apart.

**Paper-facing thresholds.** Quote the certified intervals. The frozen values are the first grid point
at or above each certified switch, so they overstate it by less than one grid step (0.05 in |w₂|).

| a | Ĝ (certified) | **R_glob, certified** | frozen R_glob | **R_solve, certified bracket** | frozen R_solve | \|w₂\| intervals (glob / solve) |
|---|---:|---|---:|---|---:|---|
| 1.30 | 0.086102 | (0.21310, 0.21364] | 0.21525 | (0.30566, 0.30620] | 0.30781 | (4.9500, 4.9625] / (7.1000, 7.1125] |
| 1.35 | 0.106914 | (0.21516, 0.21583] | 0.21650 | (0.30537, 0.30604] | 0.30738 | (4.0250, 4.0375] / (5.7125, 5.7250] |
| 1.40 | 0.128772 | (0.21730, 0.21811] | 0.21891 | (0.30503, 0.30583] | 0.30583 | (3.3750, 3.3875] / (4.7375, 4.7500] |
| 1.45 | 0.151545 | (0.21879, 0.21974] | 0.21974 | (0.30404, 0.30498] | 0.30688 | (2.8875, 2.9000] / (4.0125, 4.0250] |
| 1.50 | 0.175126 | (0.22110, 0.22219] | 0.22329 | (0.30428, 0.30538] | 0.30647 | (2.5250, 2.5375] / (3.4750, 3.4875] |
| 1.60 | 0.224360 | (0.22436, 0.22576] | 0.22997 | (0.30289, 0.30429] | 0.30850 | (2.0000, 2.0125] / (2.7000, 2.7125] |

(R_solve is the global minimiser's solve margin changing sign. **Certified**: over the minimiser's certified
enclosure, the margin is certified negative at every lower bracket end and positive at every upper end, at
all six a (`mn2_solve_finite.csv`).)

## Block 3 — prospective prediction on held-out geometries: primary criterion PASSED

Registered `prospective_prediction.md` (`8d8dec6`, before any registered training; predictions frozen
in `prospective_predictions.csv`, SHA-256 in the registration). Results `prospective_scores.csv`,
`prospective_comparisons.csv`; producer `src/prospective.py`.

**Design.**
- **Held-out windows**: four windows chosen by a structural rule with no training — the certified
  limiting-cubic κ₀ at quantiles 0.1, 0.35, 0.65 and 0.9 of an 88-window family that excludes every
  window used before. Crossed with a = 1.30 and 1.50, that gives 8 settings, each with 86–88 crossings
  out of 90 runs.
- **Change from the plan, registered with its reason**: a = 1.30 and 1.50 instead of 1.35 and 1.45.
  Only those values have base runs under the identical protocol, which the lag calibration needs.
- **Models**:
  - **U**: the certified conditional threshold R_glob of the new window.
  - **C** = λ(a)·R_glob. **λ is a lag factor fitted on the base window at the same a** (1.115 at 1.30,
    1.164 at 1.50) and must be labelled fitted.
  - **B1**: the base window's crossing |w₂|, converted to R with the new window's Ĝ (output weight
    without geometry).
  - **B2**: one pooled crossing R over all existing settings (no geometry-specific minimisation).
- **Disclosed**: one timing run (seed 999999) was made in H35 before registration. It is excluded
  from scoring, and no prediction depends on it.

**Result.** Metric: |log(pred/obs)| of the per-setting median crossing R; setting-level paired bootstrap.

| comparison | mean difference | 95% interval | |
|---|---:|---|---|
| **C − B1** (registered) | −0.257 | [−0.370, −0.131] | **C better by a wide margin** |
| **C − B2** (registered) | −0.055 | [−0.103, −0.010] | **C better by a modest margin** |
| C − U | −0.086 | [−0.111, −0.063] | C better |
| secondary target, C − B3 (fraction placed by budget) | −0.044 | [−0.068, −0.018] | C better |

**Every setting** (absolute log errors; "best" is the lowest):

| setting | crossed | observed median crossing R | U | C | B1 | B2 | best | C signed log error |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| H10, a = 1.30 | 88 | 0.2119 | 0.077 | 0.032 | 0.358 | 0.073 | C | +0.0320 |
| H10, a = 1.50 | 88 | 0.2259 | 0.116 | 0.037 | 0.339 | 0.009 | B2 | +0.0366 |
| H35, a = 1.30 | 86 | 0.2296 | 0.080 | 0.028 | 0.013 | 0.007 | B2 | +0.0284 |
| H35, a = 1.50 | 87 | 0.2476 | 0.124 | 0.028 | 0.019 | 0.082 | B1 | +0.0284 |
| H65, a = 1.30 | 87 | 0.2415 | 0.088 | 0.021 | 0.291 | 0.057 | C | +0.0212 |
| H65, a = 1.50 | 87 | 0.2679 | 0.142 | 0.010 | 0.267 | 0.161 | C | +0.0104 |
| H90, a = 1.30 | 88 | 0.2417 | 0.088 | 0.021 | 0.492 | 0.058 | C | +0.0205 |
| H90, a = 1.50 | 88 | 0.2704 | 0.151 | 0.001 | 0.458 | 0.170 | C | +0.0013 |

- **C's error is at most 3.7% in every setting.**
- **Systematic limitation of the calibration: C over-predicts in all 8 settings**, with a mean signed
  log error of **+0.022** (setting-level 95% interval [0.015, 0.030]). The reason is that the lag factor
  fitted on the base window exceeds the held-out windows' lag by about 2%. It was **not recalibrated**
  on the held-out data. State it beside the 3.7% maximum error.
- **C loses in four pairwise comparisons**:
  - to B2 at H10, a = 1.50 (0.037 against 0.009);
  - to B2 at H35, a = 1.30 (0.028 against 0.007);
  - to B1 at H35, a = 1.30 (0.028 against 0.013);
  - to B1 at H35, a = 1.50 (0.028 against 0.019).

  These are the windows whose R_glob sits nearest the pooled value, or whose Ĝ is nearest the base
  window's.
- **U under-predicts in all 8 settings**, by log(obs/U) = 0.077–0.151 (U is 7.4–14.0% below the
  observed value).
  - U's registered range [0.080, 0.182] holds at 7 of 8 settings. H10 at a = 1.30 is outside at
    0.0767.
  - Under the registered per-setting rule this is 7 of 8. Read as a joint prediction, it **fails
    narrowly**. The primary result is unaffected.

**Secondary analyses** (labelled secondary; the registered analysis above is primary):
- **Window-clustered bootstrap**: the 8 settings are 4 windows × 2 a. Enumerating all 256 window
  resamples gives C − B1 in [−0.414, −0.069] and C − B2 in [−0.098, −0.012]. Both still exclude zero.
- **Leave one window out**: C beats both baselines with each of the four windows removed. The C − B2
  mean ranges from −0.039 (without H90) to −0.071 (without H10).
- **Residual bias** (the systematic limitation above): the window-level 95% interval for C's mean
  signed log error is [0.013, 0.031].
- **Threshold definition is consistent**: λ and the held-out predictions both use the certified
  R_glob bracket midpoint from the same procedure, so there is no definition bias. Bracket
  resolution adds at most ±0.6%.

**Suggested wording.** "On four held-out window geometries chosen by a structural rule before any
training, the conditional threshold, scaled by a lag factor fitted on the base window at the same a,
predicted the median crossing within 3.7% in all eight settings, with a systematic over-prediction
of 2.2% on average (95% interval 1.5–3.0%), because the base-window lag exceeds the held-out
windows'. It outperformed an output-weight baseline (mean absolute log error 0.257 lower, 95% interval [0.131, 0.370]) and a common-R baseline
(0.055 lower, [0.010, 0.103])."

## Block 2 — Proposition 2 made precise

- **The limit threshold is an interval**: R_glob^∞ ∈ [0.1974, 0.1992], certified. Never quote 0.19991;
  that is the old grid's first point above the interval.
- **Single branch, certified globally** (`math_note_v2.md` §2(c), §3(c); `certificates_v2.py`,
  `certv2_annulus_summary.csv`, `certv2_solve_summary.csv`):
  - For every A ∈ [0.66, 0.71], and for every A over the solve bracket [1.05875, 1.06], the limit problem has
    a **unique global minimiser** (up to the mirror p → −p), and it is the branch minimiser.
  - The proof is a four-link chain, with A carried as an interval and never sampled:
    1. localisation to K(24);
    2. every point outside a 0.15 box loses to the branch by a certified margin (≥ 1.0e−4);
    3. no critical point in the ring between the 0.05 and 0.15 boxes (interval gradient bounds);
    4. a positive-definite Hessian on the 0.05 box.
  - **Consequences**:
    - The sharp value **R_glob^∞ ∈ [0.1985926, 0.1985927]** is now certified. It is no longer
      conditional, and it lies inside [0.1974, 0.1992].
    - The limit **R_solve^∞ ∈ [0.30675, 0.30712]** is global, not branch-only.
  - **Say**: "certified: for every A in the neighbourhood of the switch, the limit problem has a unique global
    minimiser, which is the branch followed by the implicit-function argument."
  - **Do not say** this for finite a. Global preference at finite a rests on Block 1c at a = 1.30–1.60 only.
- **Convergence, on certified values** (replaces "slope 0.947" and "R_glob(a) = 0.19991(1 + 0.231ε)"):
  - finite-a thresholds exceed the limit by 7.6–13.5%, increasing in a;
  - log-log slope 0.83 (exact range 0.71–0.96 over the certified intervals);
  - a single O(ε) term does **not** fit the certified values; a leading term plus a negative ε² term
    does, with c₁ between 0.24 and 0.32.
  - **The one-term law is withdrawn.** The O(ε) rate rests on the theorem, not on the fit.
  - **Warning sentence for the paper**: "A straight line through the finite-a thresholds (ε = 0.3–0.6)
    extrapolates to 0.2002–0.2035 at ε → 0, outside the certified limit, so moderate-ε values should
    not be extrapolated to the limit."
  - **Say**: "The O(ε) rate is proved (under the stated hypotheses); at the a we measure (ε = 0.3–0.6)
    the approach is concave and the ε² term is visible."
  - **Do not say**: "R_glob(a) is fitted by R_glob^∞(1 + c₁ε)", or quote 0.947 or 0.231.
- **The first-order coefficient is now computed, not fitted** (`math_note_v2.md` §8, `first_order_c1.csv`):
  - c₁ ∈ [0.2852300, 0.2852303], certified. It combines the switch shift, +0.662 (implicit-function
    theorem at a Krawczyk-certified switch point), and the gap-maximiser correction, −0.377.
  - The earlier "c₁ ≈ 0.49, a factor 2.4 too large" came from an incomplete calculation (K alone). Do
    not quote it.
  - **Chronology, state it in this order**:
    1. The original 0.49 came from an incomplete argument that used only the K correction.
    2. The certified refit showed it was inconsistent with the data: two-term laws allow only
       [0.243, 0.321].
    3. The full first-order calculation was then done and gives c₁ ∈ [0.2852300, 0.2852303].
    4. The registered small-ε test (`f92b1b5`, a = 1.01–1.04) is the independent check. It is
       **pending**.
  - **Until that test is scored**, the paper may say only: "the corrected first-order calculation is
    consistent with the certified large-ε values". It may **not** say that the first-order coefficient
    is confirmed.
  - **Range of validity**: the first-order expansion of K is certified for |ε| ≤ 0.05, i.e.
    a ∈ [0.95, 1.05] (active set unchanged, strict maximum, two-sided; `math_note_v2.md` §8).
    - The registered c₁ test (a = 1.01–1.04) lies inside that range, so its premises hold where it is
      evaluated.
    - **Do not quote first-order statements** (c₁, k₁, A′(0), or "R_glob(a) ≈ R_glob^∞(1 + c₁ε)") as
      applying at the larger a the paper also uses (1.30–1.60, ε = 0.3–0.6). There the certified values
      show a visible ε² term.
    - The corner's smallest multiplier is small (0.004). **Exploratory tracking up to ε = 2.0** found no
      change in the gap maximiser's active set at any of 79 ε (math note §9). The conditional minimiser's
      active pair is also unchanged at a = 1.30–1.60.
    - **So the concavity at ε = 0.3–0.6 is genuine higher order, not a structural change.** Its largest
      part is the product of the two first-order effects (a₁k₁ = −0.250). Say it this way if needed; do
      not call it a regime change.

MATH_NOTE_SECTION

## Blocks 4 and 5 — fixed-scale dynamics and retention

**A registered validity check fired, and was amended before any outcome was computed.**
- **What fired**: the registration (`fixed_scale_prediction.md`, `888344d`) required the k = 1 replay
  to reproduce an independent frozen-w₂ continuation to 1e-10. On launch the check gave 0.105.
- **Why the check was invalid**: its reference zeroed w₂'s gradient under the full Adam optimiser.
  That does not freeze w₂ — Adam's stored momentum kept moving it by up to 0.033 in 25 steps.
- **The replay itself was correct**: a reference that truly freezes w₂ (reset after every step)
  reproduces it exactly.
- **The amendment** (`10c1ea5`) replaced only that reference. No outcome prediction, horizon, grid,
  sample size or criterion changed.
- **Tests before any replay**: 11 tests covering the reference, the replay, and every stop condition
  in a should-fire and a should-not-fire case (`tests/test_fixed_scale.py`). All pass.
- **Record**: this was the second error in that registration's stop conditions; both concerned the
  implementation check, not an outcome.
- **E-2 stays failed** whatever these blocks show.

### Block 4 — fixed-scale dynamics: D1 passed, D2 failed

Registered `fixed_scale_prediction.md` (`888344d`; validity check amended at `10c1ea5`, before any outcome).
Results `fixed_scale_block4_curve.csv`, `fixed_scale_block4_tests.csv`; producer `src/fixed_scale.py`.

**Design.**
- **Checkpoints**: 543 before-placement checkpoints (G < 0) from 180 fresh runs.
- **Rescaling**: (w₂, b₂) jointly rescaled so that R/R_glob lands on a fixed grid. This preserves
  every decision; verified in 8,688 of 8,688 replays.
- **Replay**: |w₂| held fixed while (w₁, b₁, b₂) train for 4,000 steps, with Adam moments either
  preserved or reset.

| R/R_glob | 0.6 | 0.8 | 0.9 | 1.0 | 1.1 | 1.25 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| placed at 4,000 steps (n = 543) | 0.000 | 0.000 | 0.013 | 0.309 | 0.700 | 0.967 | 0.989 | 0.989 |

- **D1 (equilibrium) passed**, in both optimiser-state variants. The 50% point is **1.049**, inside
  the registered [0.9, 1.25], with no significant monotonicity violation.
- **D2 (saturation) failed**: placement does not fall at 2× the threshold.
- **Scope of the claim — state it exactly this way**:

  > At a = 1.30 in the base window, under Adam, with moments preserved or reset, where training ends
  > up at a fixed output scale is set by the conditional threshold: the placed fraction switches
  > around R/R_glob ≈ 1.05 at a 4,000-step horizon.

  Do not generalise beyond this setting, optimiser or horizon.
- **The two variants**: they differ early (first-step ΔG by up to 0.14; time to first G > 0 in 25% of
  replays) but end with the same placement outcome in every pair.
- **Fixed-scale verdicts, stated plainly**:
  - **Q1 FAILED.**
  - **Q2 FAILED.**
  - **S1 FAILED.**
  - **S2 PASSED.**
- **Interpretation**:
  - Q1 and Q2 registered that fixed-scale placement would switch at the **population** threshold, and
    they failed.
  - S2 registered that it would switch at the **median of each training set's own** threshold, and it
    passed.
  - So at fixed scale, placement is governed by the training set's own conditional threshold, about 3–6%
    above the population value.
  - Per-replay prediction from own thresholds reaches **0.894**, against **0.760** for the population
    rule, short of the registered 0.95.
  - The explanation of that shortfall is left to the mirror-branch analysis (exploratory, pending).
- **What the own-seed account does and does not predict (post hoc, `mirror_q2_s1_breakdown.csv` pending;
  numbers from the scored horizon replays)**:
  - It predicts **the fraction of replays placed at each scale** (S2).
  - It does **not** predict **which individual replays place**. At 0.95×, 33 of the 51 placed replays
    belong to seeds whose own threshold lies above that level. At 0.9×, all 8 do.
  - Two-thirds of S1's 288 disagreements (195) lie more than 5% from the run's own threshold. So the
    misses are not threshold-resolution effects near each run's switch.
  - **Say**: "the training set's own threshold sets where the population of fixed-scale runs switches,
    not the outcome of an individual run". The mirror-branch analysis and a basin census are examining
    what determines individual outcomes; both are exploratory.
  - **Mirror occupancy accounts for almost all of S1's disagreements** (post hoc, exploratory;
    `mirror_q2_s1_breakdown.csv`, `mirror_global_branch.csv`). A replay's 64k endpoint either lies on the
    globally preferred mirror branch of its own training loss at the held scale, or on the other mirror
    branch:

    | 64k endpoint | replays | own-seed rule agreement | disagreements |
    |---|---:|---:|---:|
    | on the globally preferred branch | 2,097 | **0.993** | 15 |
    | on the mirror branch | 618 (23%) | 0.558 | 273 |

    - All 41 replays placed below their own threshold at 0.9× and 0.95× are on the mirror.
    - **The 99.3% is conditional on the branch where the replay ends, which is known only after the
      run.** It explains the disagreements; it does not predict them.
    - **Say**: "among replays that end on the globally preferred branch of their own loss, the own
      threshold matches the outcome in 99% of cases; the disagreements are runs that end in the mirror
      basin".
    - **Dynamics steer runs toward the global branch** (post hoc):
      - an equal-weight mixture of both branches' thresholds predicts the base window's median crossing R
        **11–16% too low** (−0.115 at 1.30, −0.159 at 1.50);
      - by the crossing, 77–90% of runs sit on the globally preferred branch, although initialisation
        chooses it at chance;
      - **Exploratory — when and why runs commit** (base-window runs; `prospective_own_switch_losses.csv`,
        `prospective_own_commit_by_*.csv`):
        - A branch switch must pass through w₁ = 0 or w₂ = 0, where the network is constant and the loss is
          log 2. So a switch occurring at log 2 is **forced, not an observation**.
        - The finding: **branch switches occur only early, while the loss is still at log 2** (all 30 of
          them, none afterwards).
        - During that phase, runs drift toward the lower-loss mirror. That takes the global-branch share
          from chance at initialisation to 77–90% at crossing.
        - Once the loss falls below log 2, a switch would require the loss to rise back to it, and that
          does not happen.
        - **Commitment curve**: the branch matches the crossing branch at every check **from 10% of the
          time to crossing onward**, and **once R exceeds 0.10 at a = 1.30 or 0.15 at a = 1.50**. Before
          that the match rate is 62% in the first 1% of the time, 78–86% at 1–5%, and 95–98% at 5–10%.
    - **Summary, all post hoc**:
      - initialisation selects the mirror branch at chance (51.4%);
      - every replay ends on the branch it started on;
      - with the starting branch's threshold, per-replay agreement is 0.994;
      - with the branch occupied at the crossing, crossing R correlates with that branch's threshold at
        ρ = 0.997 (1.30) and 0.995 (1.50), with a residual of 3.0% and 6.3% that branch-matching does not
        change.
      - **The registered S1 still failed.**
    - **Can the branch be known before the run?** (post hoc, `mirror_*.csv`)
      - **Initialisation does not select the mirror branch.** The initialisation-selected branch matches
        the branch at the free-training crossing in **51%** of runs: 0.50 for Block 4/5 (177 runs), 0.55
        at 1.30 and 0.55 at 1.50 (38 runs each). That is a coin flip.
      - The **starting checkpoint's branch** does determine a fixed-scale replay's end branch (100% of
        replays). With that branch's threshold, S1's per-replay agreement is **0.994**, against 0.894
        for the global own threshold and 0.760 for the population rule.
      - The starting branch is known when a replay starts. For a free-training run it is not known before
        training.
    - **The mirror thresholds differ substantially**: |T₊ − T₋|/T_global has median 0.106, IQR
      0.042–0.183 and maximum 0.40 at n = 400. That gap exceeds the replay's distance from its own
      threshold for 281 of the 288 disagreements.
    - **S3 with branch-matched thresholds**:
      - using the branch occupied at the crossing (known only then), crossing R tracks the threshold with
        **ρ = 0.997 (1.30) and 0.995 (1.50)**;
      - using the initialisation-selected branch, ρ is 0.13 and 0.12.
      - **The residual is unchanged either way**: 3.0% and 6.3%, against 3.1% and 6.4% with the global
        threshold. So the residual is **not** mirror occupancy.
    - **The mirror gap shrinks with training-set size** (`mirror_size_gap_summary.csv`; size-test seeds,
      20 per cell). The median |T₊ − T₋|/T_global goes 9.7% → 8.8% → 3.7% at a = 1.30 and 9.9% → 8.8% →
      3.6% at a = 1.50, for n = 400 → 1,600 → 6,400.
      - That is consistent with finite samples breaking the x → −x symmetry.
      - So mirror asymmetry is a finite-sample effect, and it cannot account for the residual (about 4%
        and 7.5%) that persists at large n.
    - **The 15 disagreements on the preferred branch** (`mirror_basin_census_preferred.csv`):
      - **No third basin.**
      - **8 are on the degenerate plateau**: w₁ ≈ 0, flat Hessian, loss ≈ log 2. The run never left the
        constant predictor, so its branch label is arbitrary.
      - **4 are slow relaxation within 0.35% of the run's own threshold.**
      - **3 sit at the branch minimiser** within 0.023% of the threshold, below the own threshold's
        bracket resolution.
    - **Separating the plateau** (24 of 2,715 endpoints, 0.9%):
      - off-plateau, preferred branch: 2,078 replays, agreement **0.9966** (7 disagreements, all within
        0.35% of the own threshold);
      - off-plateau, mirror branch: 613 replays, agreement 0.561;
      - plateau: 24 replays, 12 disagreements.
    - Whether the mirror branch's own threshold then predicts those runs is pending (branch thresholds
      computing).
- **Horizon extension, registered** (`fixed_scale_horizon_prediction.md`, `cdfbf9d`): does the 50%
  point move toward 1.0 at 4× and 16× the horizon? **Q1 FAILED, Q2 FAILED; the registered competing
  outcome holds** (`fixed_scale_horizons_tests.csv`).

  | horizon | 4,000 | 16,000 | 64,000 |
  |---|---:|---:|---:|
  | x₅₀ (preserved) | 1.0458 | 1.0450 | 1.0454 |
  | placed at 0.9× | 0.0129 | 0.0129 | 0.0147 |
  | placed at 0.95× | 0.094 | 0.092 | 0.094 |

  - **The 50% point does not move with horizon.** It stays at 1.045 from 4k to 64k, the same under reset
    moments, inside the competing band [1.03, 1.07].
  - **Say**: "at a fixed output scale the switch sits about 4.5% above the conditional threshold, and
    this is an equilibrium, not a finite-horizon effect."
  - **Q2**: placement below the threshold does not vanish at long horizons: 8/543 at 0.9× and 9.4% at
    0.95× at 64k. The 7 runs placed at 0.9× at 4k all persist, and one more places by 64k.
  - **Why some runs stay placed below the threshold (exploratory, `diagnose_below.md`)**:
    - The seven 0.9× placements sit in **metastable correct-placement basins**: strict local minima of
      their own training loss with G > 0, 0.008–0.064 above that run's global minimum.
    - Each basin is the **mirror image (x → −x) of the global branch**. The population objective is
      symmetric, so the two are equivalent there. A finite sample breaks the symmetry, and at this scale
      the placed mirror is the higher-loss one.
    - They are separated from the global branch by a barrier of exactly log 2, since the path must pass
      through the constant predictor.
    - The runs' own thresholds (1.18–1.23×) do not explain them.
    - The single-branch certificate at a = 1.30 is for the population objective, modulo the mirror
      symmetry. Say so if citing it.
  - **A validity check fired and was amended before scoring.** The registered check requires the
    4,000-step outcome to reproduce Block 4 exactly, and it returned False.
  - **Cause**: Block 4's stored values were read with a CSV parser that is off by up to 1e−16, then
    compared exactly.
  - **With exact parsing on both sides**, all 3,258 rows reproduce bit for bit.
  - **The amendment** changed only how stored values are read. Tolerance stayed at zero, and no
    prediction or criterion changed. It was made before any Q1, Q2, S1 or S2 outcome was scored.
  - This is the third implementation error in this registration family's validity checks; none
    concerned an outcome criterion.
- **Exploratory, post hoc — critical slowing.** The recorded 25-step first-placement times cannot
  support a relaxation fit: placement above the threshold happens within 25–50 steps at every level.
  Integrating that along free training predicts an offset of about 1–2%, far below the observed 9.6%.
  Fixed-scale relaxation therefore does not appear to explain the free-training offset. Per-step
  settling times are being recorded in the long-horizon replays. No prospective design has been
  written.
- **Caveat for any reading of the placement curve as dynamics**: first-placement times come only from
  replays that place (at 1.1×, 30% never place within the horizon). The curve's width around the
  threshold reflects heterogeneity across replays, not a relaxation timescale.
- **Sample-specific thresholds, registered** (`own_threshold_prediction.md`, `312829f`): each run
  trains on its own 400 points. S1–S3 test whether the run's own conditional threshold explains the
  replay-level placements, the 50% point and the free-training offset (`own_threshold_scores.csv`).
  - **S3 (free-training offset): PASS at a = 1.30, FAIL at a = 1.50.**
    - The own-seed threshold sits above the population threshold by 5.7% (1.30) and 5.6% (1.50). The
      registered requirement was at least half the free-training offset: ≥ 4.8% at 1.30 (met) and
      ≥ 6.3% at 1.50 (**missed**).
    - Crossing R tracks each run's own threshold (Spearman ρ = 0.88 at both a, p ≈ 1e−5).
    - Measured against the own threshold, the offset shrinks from 9.2% to 3.1% (1.30) and from 11.9% to
      6.4% (1.50).
    - The competing account (own thresholds centred on the population value) is rejected at both a.
  - **Decomposition of the free-training offset, state it this way**:

    | | a = 1.30 | a = 1.50 |
    |---|---:|---:|
    | free-training offset against the population threshold | 9.6% | 12.6% |
    | finite-sample shift (median own-seed threshold over population) | 5.7% | 5.6% |
    | share of the offset it accounts for | ≈ 60% | ≈ 44% |
    | residual against each run's own threshold (median, log) | 3.1% | 6.4% |

    - The finite-sample shift is about 6% at both a.
    - The residual grows with a and is **unexplained**.
    - Crossing R tracks each run's own threshold with Spearman ρ ≈ 0.88 at both a.
  - **Say**: "about 6% of the 9.6–12.6% free-training offset reflects each run's own finite training set;
    this accounts for the registered half at a = 1.30 but not at a = 1.50, and leaves a residual of 3.1%
    and 6.4% that grows with a and is not explained".
  - **Validation**:
    - The fast search agrees seed by seed with 1d's certified search on seeds 0–39: 40/40 brackets
      overlap, the mean difference is −0.012%, and the range is −0.09% to +0.06%.
    - Registered certified check on 20 Block 4 seeds: 13/20 certified at both ends, 7 with one end
      unresolved at 1e−9, and 0 contradicted. At 1e−11, 19/20.
    - The 1.033 (Block 4/5 seeds) against 1.063 (1d seeds) medians differ by sampling variation: the
      difference is 0.029, 95% interval [−0.017, 0.067].
  - **S1 (per replay): FAIL.** The own-seed rule (placed iff held R > the run's own R_glob) agrees
    with 89.4% of the 2,715 replays at 64k, below the registered 95%. It does beat the population rule
    (76.0%) by 13.4 points, above the registered 5.
  - **S2 (50% point): PASS.** x₅₀(64k) = 1.0454 against the median own/pop = 1.0359 over the replays'
    seeds, a difference of 0.0095 (tolerance ±0.02). So the fixed-scale switch's position above the
    population threshold is accounted for by the runs' own thresholds.
  - Scored with the scorer committed before the data (`a015ace`), unchanged; the current refactored
    scorer gives identical output.

## Adiabatic-lag test (registered `lag_test_prediction.md`, `bd8cb9c`; scored)

- **Design**: w₂'s learning rate scaled by φ ∈ {0.25, 0.5, 1, 2} at n = 6,400, a = 1.30 and 1.50, on the size
  test's 50 training sets per a.
- **Validity**:
  - φ = 1 reproduces the size test's runs **bit-identically on all 100 runs**;
  - 44–49 of 50 runs cross in every cell.
- **Primary threshold**: the global own threshold. The registered branch rule chose it because the
  initialisation-selected branch matched only 51.4% of crossings, against the required 90%.

| a | φ = 0.25 | 0.5 | 1 | 2 |
|---|---:|---:|---:|---:|
| 1.30: residual (95%) | 0.47% [0.26, 1.05] | 2.66% [2.56, 2.77] | 3.11% [2.98, 3.17] | 3.12% [3.04, 3.19] |
| 1.50: residual (95%) | 1.07% [0.64, 2.18] | 5.34% [5.11, 5.65] | 6.56% [6.21, 6.79] | 6.64% [6.40, 6.78] |

- **L1 (primary) PASS at both a.** The residual decreases with w₂'s learning rate and tends toward zero:
  the ordering is strict; the interval of residual(1) − residual(0.25) is above 0 ([2.1, 2.8]% and
  [4.4, 5.9]%); and residual(0.25) ≤ ½·residual(1).
  - The competing outcome (no dependence) is rejected at both a.
  - The same verdicts hold with the initialisation-selected branch threshold (secondary).
- **L2 (secondary, proportionality) FAIL at both a.**
  - residual(0.5)/residual(1) = 0.86 and 0.81, outside [0.30, 0.70];
  - residual(0.25)/residual(1) = 0.15 and 0.16, inside [0.05, 0.45].
  - The dependence is not linear: the residual is nearly flat from φ = 0.5 to 2 and collapses between 0.5
    and 0.25. (Between φ = 1 and 2 the ordering holds, but the difference interval includes 0.)
- **Direct lag diagnostic.** At the crossing, the median distance from the hidden parameters to the
  branch minimiser at the current scale is:
  - a = 1.30: 0.0013 at φ = 0.25, against 0.0050–0.0059 above;
  - a = 1.50: 0.0029, against 0.012–0.015.
  - It shrinks with φ, with the same shape as the residual.
- **Confound checks (per φ), stated beside the verdicts**:
  - **The mirror-branch share at the crossing does not shift** (w₁·w₂ > 0 in 50–57% of runs at every φ).
  - **Time on the constant-predictor plateau shifts materially with φ.** The median number of steps with
    |w₁| < 0.05 before the crossing is:
    - 1,309 and 1,691 at φ = 0.25;
    - 702 and 1,244 at φ = 0.5;
    - 21 and 13 at φ = 1;
    - 0 at φ = 2.
    - The fraction of runs with any plateau time falls from 0.68 to 0.40–0.42.
  - Slower w₂ keeps runs near the constant predictor longer. That could move the residual independently
    of lag, so it has to be weighed with the L1 verdict.
  - **Post hoc observation (it does not remove the caveat)**: plateau time does not track the residual's
    collapse step for step.
    - Between φ = 1 and 0.5, plateau time rises from 21 to 702 steps (a = 1.30) and from 13 to 1,244
      (1.50), while the residual changes little (3.11 → 2.66% and 6.56 → 5.34%).
    - Between 0.5 and 0.25, plateau time roughly doubles, while the residual collapses (to 0.47% and
      1.07%).
    - A deconfounded test that switches φ only after the plateau has been left is registered separately
      (`lag_test2_prediction.md`).
- **Say**: "slowing w₂'s growth removes most of the residual (to ≈ 0.5% and 1% at a quarter of the
  learning rate), consistent with an adiabatic lag of the output scale; the dependence is not
  proportional, and slower w₂ also lengthens the time spent on the constant-predictor plateau, which is a
  possible confound".

## Sample-size test (registered `sample_size_prediction.md`, `04e7668`; scored)

- **Design**: a ∈ {1.30, 1.50} × n ∈ {400, 1,600, 6,400}, with 50 fresh seeds per cell for both arms.
- **Validity**:
  - 50/50 brackets overlap 1d's at n = 400;
  - the certified check agrees on 10/12, with 2 unresolved ends and **0 contradictions**;
  - 48–49 of 50 runs cross in every cell.
- **Verdicts**:
  - **N1 PASS at both a.**
  - **N2 FAIL at 1.30, PASS at 1.50.**
  - **N3 FAIL at both a.**
  - The competing outcome ("the offset is not a finite-sample effect") **holds at 1.30** and does not
    hold at 1.50.

| a | n | own excess (95%) | free offset (95%) |
|---|---:|---|---|
| 1.30 | 400 | 4.2% [1.8, 7.4] | 7.1% [3.7, 9.8] |
| 1.30 | 1,600 | 3.9% [1.5, 5.1] | 5.6% [3.2, 7.3] |
| 1.30 | 6,400 | 1.2% [0.5, 2.2] | 4.2% [3.1, 5.2] |
| 1.50 | 400 | 4.4% [1.4, 7.3] | 10.6% [8.3, 13.7] |
| 1.50 | 1,600 | 3.6% [1.2, 5.1] | 8.5% [6.9, 11.0] |
| 1.50 | 6,400 | 1.1% [0.2, 2.1] | 7.5% [6.4, 8.3] |

- Every pairwise difference and its interval is in `sample_size_detail_pairs.csv`. No ordering step
  failed, so the "indistinguishable" flag does not apply.
- **N2 at 1.30 fails** only on its interval condition: the offset decreases 7.1 → 5.6 → 4.2%, but the
  400 − 6,400 difference interval [−0.005, 0.056] includes 0.
- **The offset decomposition, state it this way**:
  - N1 passes at both a: the own-seed excess shrinks with training-set size, to about 1% at n = 6,400.
  - N2 fails at 1.30 (its interval includes zero) and passes at 1.50.
  - N3 fails at both a.
  - The free-training offset tends not to zero but to about **4.2% at a = 1.30 and 7.5% at 1.50**, close
    to S3's residual against the own threshold.
  - So the offset has **a finite-sample part that vanishes with n**, and **a residual that persists and
    grows with a, unexplained so far**.
- **Reading, and the connection to S3's residual**:
  - As n grows, the own-seed excess shrinks toward zero: to about 1% at 6,400.
  - The free-training offset shrinks less, to 4.2% (1.30) and 7.5% (1.50). Those are close to S3's
    residual against the own threshold (3.1% and 6.4% at n = 400).
  - So the free-training offset tends to the residual, not to zero. The finite-sample part disappears
    with n; the residual does not, and it grows with a.
  - **Say**: "the finite-sample shift of the threshold vanishes as the training set grows; a residual
    lag of about 4% (a = 1.30) and 7.5% (a = 1.50) remains and is not a finite-sample effect".

## Block 5 — retention just after placement: both registered predictions PASSED

Registered `fixed_scale_prediction.md` (`888344d`, amendments `0f9f67e`, `10c1ea5`); producer
`src/fixed_scale.py` (`run_replays(5)`, `score(5)`, `score5_splits`); artifacts
`fixed_scale_block5.csv`, `_curve.csv`, `_tests.csv`, `_splits.csv`.

- **Design**: the first-placement checkpoint of each of 177 runs (180 seeds; the registered minimum was
  150), rescaled to each level and trained 12,000 steps with |w₂| held. **Retained** = G > 0 at every
  25-step check (Block E's "kept").
- **Validity**: decisions preserved in 2,832 of 2,832 replays; k = 1 check difference 0.0.

| R/R_glob | 0.6 | 0.8 | 0.9 | 1.0 | 1.1 | 1.25 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retained, preserved (n = 177) | 0.000 | 0.000 | 0.017 | 0.203 | 0.644 | 0.966 | 1.000 | 1.000 |

- **Monotone: PASS** (no paired McNemar violation, both variants).
- **Location: PASS**: the 50% point is **1.067** (reset 1.068), inside the registered [0.9, 1.1].
- **Reported without prediction**: the 10–90% width is 0.27 (0.94 to 1.22). By starting state
  (median splits of first-placement step and of G at placement) the width stays 0.26–0.28 and the 50%
  point moves by at most 0.03 (1.054 vs 1.083).
- **E-2 stays failed as registered.** Block 5 is a new registered test at fixed scale, not a rescoring
  of E-2.
- **Scope, same as Block 4**: a = 1.30, base window, Adam, moments preserved or reset, 12,000-step
  horizon. Say: "Just after placement, whether a correct configuration is kept at a fixed output scale
  switches around R/R_glob ≈ 1.07; below 0.9 it is almost always lost, above 1.25 almost always kept."
- The same caveat as Block 4 applies to the width: it reflects heterogeneity across runs (S1–S3 test
  whether own-seed thresholds account for it).


## Figure manifest (v4)

These are **not in `paper/figures` yet.** They are held until S1–S3 and the c₁ test are scored, since
those can change what the panels should show.
- Producer: `src/figures_v4.py` (`PYTHONPATH=. python -m src.figures_v4`).
- Output: `results/figures/v4/*.pdf` (vector) and `.png` previews.
- All figures are single column, 3.25 in wide, with nothing below 6 pt.
- Every panel states n and shows uncertainty.

| file | content | n | uncertainty shown | source artifacts | pending changes |
|---|---|---|---|---|---|
| `v4_prospective.pdf` | Block 3: predicted against observed median crossing R, 8 held-out settings × 4 models. C is filled, U open and joined to C by a segment (the fitted λ). λ(1.30) = 1.115 and λ(1.50) = 1.164 are labelled as fitted on the base window. Identity line. | 86–88 crossings of 90 per setting | observed medians: bootstrap 95% (10,000 resamples, seed 0) | `prospective_runs.csv`, `prospective_scores.csv`, `prospective_calibration.csv` | none expected |
| `v4_fixed_scale.pdf` | (a) Block 4: placed at 4,000 steps against held R/R_glob, both optimiser-state variants, x₅₀ in the legend, registered band [0.9, 1.25] shaded. (b) Block 5: retained through 12,000 steps, band [0.9, 1.1], with **E-2's criterion** (kept ≥ 0.9 at 1.15 × Block E's R_glob = 1.156 certified units) and Block E's observed 33/37 marked FAIL. | 543 checkpoints per level (a); 177 (b); Block E 37 | Clopper–Pearson 95% | `fixed_scale_block{4,5}_curve.csv`, `_tests.csv`; `blockE_results.md` | **Room kept** for the 16k and 64k curves from the registered horizon extension (Q1). Hook: `fig_fixed_scale(horizons=True)`, added once Q1/Q2 are scored. S1/S2 may add the own-seed prediction. |
| `v4_thresholds.pdf` | Certified R_glob and R_solve intervals at a = 1.30–1.60 and in the limit ε → 0, with free-training crossing-R distributions (budget 32k) on the same axes. | 38 runs per a | certified interval widths; crossing medians bootstrap 95% | `cond_certified_brackets.csv`, `limit_K_base.csv`, `mn2_solve_limit.csv`, `wi_crossing_runs.csv` | S3 may add the own-seed thresholds. The c₁ test may add the small-ε points (a = 1.01–1.04) and the first-order line. The limit R_solve is now global (competitor exclusion certified, `certv2_solve_summary.csv`). |
