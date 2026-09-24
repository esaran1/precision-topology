# Crossing-detection discretisation audit — POST HOC (exploratory; no registered verdict changes)

**For the rebuttal revision** (not the submission). Requested by the author on 2026-09-24. Producer:
`src/crossing_audit.py` (`python -m src.crossing_audit 1`, 98 s at one worker, peak RSS 0.72 GB). Outputs:
`crossing_audit_runs.csv` (per run) and `crossing_audit_summary.csv`. Tests: `tests/test_crossing_audit.py`
(constructed linear, convex and grid-offset series).

**Registered verdicts are unchanged, whatever this shows.** Every interpolated number below is post hoc.

## 1. Check intervals, from the code

| experiment | checks placement every | budget | φ arms: did the interval scale with the budget? |
|---|---|---|---|
| phase 2b (S3's crossings; `phase2b_ordering.run`) | 1 step | 2,000–32,000 | no φ arms |
| size test (`sample_size.free_one`) | 1 step | 32,000 | no φ arms |
| lag test 1 (`lag_test.run_one`) | 1 step, in every φ arm | 32,000/φ | **no**: the budget scales by 1/φ, the check interval stays at 1 step |
| lag test 2, deconfounded (`lag_test2.continue_one`) | 1 step, in every φ arm | t* + ⌈(32,000 − t*)/φ⌉ | **no**: as above |
| Block 4b, all four arms (`residual_mechanism.continue_from`) | 1 step | to 32,000 | no φ arms (φ = 1) |
| Block G base window, the runs λ was fitted on (`blockG_windows.train`) | **50 steps** (i ≡ 0 mod 50, 0-based) | 12,000 | no φ arms |
| Block 3, prospective (`blockG_windows.train`) | **50 steps** | 12,000 | no φ arms |
| prospective own-seed (`prospective_own.train_logged`) | **50 steps** | 12,000 | no φ arms |

In the φ arms, the interval is one step, but the step itself scales. The w₂ update is multiplied by φ, so the growth
of R over one check interval scales with φ as well.

## 2. Upper bound on the upward bias: growth of R over one check interval, relative to the threshold

- A check-based crossing can lag the true crossing by at most one check interval. So R(first positive check) −
  R(previous check) bounds the upward bias of the recorded crossing R.
- Measured here at the recorded crossing, relative to each experiment's reference threshold:

| experiment | a | runs | interval | median bound | max bound |
|---|---|---|---|---|---|
| phase 2b (all 38 crossing runs per a, stored rows) | 1.30 | 38 | 1 | 0.047% | 0.083% |
| | 1.35 | 38 | 1 | 0.055% | 0.113% |
| | 1.40 | 38 | 1 | 0.066% | 0.156% |
| | 1.45 | 38 | 1 | 0.077% | 0.210% |
| | 1.50 | 38 | 1 | 0.095% | 0.299% |
| | 1.60 | 38 | 1 | 0.117% | 0.444% |
| size test n = 6,400 (= lag tests at φ = 1, 4b control) | 1.30 | 20 | 1 | 0.058% | 0.072% |
| | 1.50 | 20 | 1 | 0.149% | 0.253% |
| Block G base window (λ calibration) | 1.30 | 20 | 50 | **2.40%** | 3.46% |
| | 1.50 | 20 | 50 | **4.91%** | 12.07% |
| Block 3 | 1.30 | 20 | 50 | **2.57%** | 6.36% |
| | 1.50 | 20 | 50 | **4.91%** | 21.11% |
| prospective own-seed | 1.30 | 20 | 50 | **2.02%** | 2.96% |
| | 1.50 | 20 | 50 | **3.83%** | 6.05% |

- **Reference thresholds.**
  - Phase 2b: the run's own threshold at n = 400 (`own_threshold_crossing.csv`) at a = 1.30 and 1.50. At the other
    four a, the growth is taken relative to the recorded R.
  - Size test: the own threshold at n = 6,400.
  - Block G base and Block 3: the setting's certified R_glob (U).
  - Own-seed: U_own.
- **φ arms (lag tests 1 and 2, every step).** Two figures are given.
  - *Estimate*: φ × the φ = 1 one-step growth above. That is 0.015%, 0.029%, 0.058% and 0.116% at a = 1.30 for
    φ = 0.25, 0.5, 1 and 2, and 0.037%, 0.074%, 0.149% and 0.297% at a = 1.50 (medians).
  - *Hard bound*: Adam's step bound (Kingma and Ba, §2.1) is |Δw₂| ≤ φ·lr·(1 − β₁)/√(1 − β₂) = 3.16·φ·lr. Relative
    to the smallest own |w₂| at n = 6,400 (4.797 and 2.447), that gives 0.66%·φ at a = 1.30 and 1.29%·φ at 1.50.

## 3. Residuals under both definitions

- **Definitions.**
  - *Check*: the recorded crossing (first check with G > 0).
  - *Interpolated*: the crossing located by linear interpolation of G between the last negative and the first
    positive check, with R interpolated at the same fraction.
  - For the 50-step experiments, the interpolation between checks is compared with the same interpolation between
    single steps (per-step G, from the reruns). They agree to ≤ 0.04 percentage points at the median, so linear
    interpolation between 50-step checks is accurate.
- **Per-step G.** It was stored only for phase 2b: `phase2b_checkpoints.csv` logs every step once |G| < 0.05·G*.
  All 228 crossing runs at budget 32,000 are covered, with the last negative and the first positive step on disk.
- **Reruns.** 20 runs per a (φ = 1, the lowest seeds) were rerun with G logged at every step up to the crossing:
  - the size test (its n = 6,400 runs are the lag tests' φ = 1 runs and 4b's control);
  - the Block G base window;
  - Block 3 (5 runs per held-out window);
  - the own-seed test (3 runs per window for V10–V60 and 2 for V70, per a).
  - The Block G base window is exactly λ's data source (λ(a) = median crossing R / R_glob(base, a), 40 runs per a).
    The 20-run subset's check-based residual (7.1% / 12.6%) is below λ − 1 on all 40 runs (11.5% / 16.4%): a
    subset difference.

  **All 160 reruns reproduced the stored crossing step and |w₂| bit for bit.**

| experiment | a | reference | median residual: check | median residual: interpolated | median per-run shift | max per-run shift |
|---|---|---|---|---|---|---|
| phase 2b (S3), all runs | 1.30 | R_own (n = 400) | 3.13% | 3.11% | 0.02 pp | 0.06 pp |
| | 1.50 | | 6.61% | 6.58% | 0.04 pp | 0.28 pp |
| size test (lag φ = 1, 4b control) | 1.30 | R_own (n = 6,400) | 3.16% | 3.11% | 0.02 pp | 0.06 pp |
| | 1.50 | | 6.68% | 6.59% | 0.06 pp | 0.20 pp |
| Block G base (λ calibration) | 1.30 | R_glob (base) | 7.11% | 5.64% | 1.38 pp | 2.29 pp |
| | 1.50 | | 12.56% | 9.36% | 1.91 pp | 9.32 pp |
| Block 3 | 1.30 | U = R_glob (window) | 8.58% | 7.46% | 0.85 pp | 1.95 pp |
| | 1.50 | | 13.04% | 11.69% | 2.23 pp | 16.28 pp |
| prospective own-seed | 1.30 | U_own | 4.30% | 2.97% | 1.19 pp | 2.45 pp |
| | 1.50 | | 8.01% | 6.18% | 1.30 pp | 3.73 pp |

(pp = percentage points of residual. "Median per-run shift" is the median of check − interpolated over runs; it
differs from the difference of the two medians.)

## 4. Reading (post hoc)

- **Every-step experiments** (phase 2b / S3, the size test, both lag tests, 4b): the discretisation bias is at most
  one step. The median is 0.05–0.15% of the threshold and the maximum 0.44%. The residuals move by ≤ 0.09
  percentage points. **Negligible for every claim.**
  - S3 remains 3.1% / 6.6%.
  - The lag tests' and 4b's control residual is 3.1% / 6.6% under both definitions.
  - One caution for 4b. The arm differences are of the order of one detection step: the reset arm's median is
    0.16 pp above control at a = 1.50, against a median one-step growth of 0.15%. The registered 4b verdict needs
    an arm to remove half the residual (1.6 / 3.3 pp), which is 20–30 steps' worth, so it is unaffected.
  - Lag test 1's criterion L2 allows ±0.20 on residual(φ)/residual(1). The φ-dependent bias moves that ratio by
    about 0.02 at φ = 2 (estimate). The hard Adam bound would allow up to 0.2, but the measured steps are 10× below
    it.
- **50-step experiments** (Block G base / λ, Block 3, own-seed): the recorded residual is **overstated by 0.9–2.2
  percentage points at the median**, and by up to 16 pp for single runs whose |w₂| moves fast at the crossing.
  That is 10–30% of the residual those experiments report.
  - Block 3's calibrated predictor C uses λ fitted on 50-step crossings, so the same cadence bias is largely built
    into λ. The median shift on the base window (1.38 / 1.91 pp) is close to that on the held-out windows
    (0.85 / 2.23 pp). C is therefore approximately cadence-consistent, although this rests on 20-run subsets.
  - U and U_own are uncalibrated, so the cadence bias counts against them in full. Under interpolation, the
    own-seed residual against U_own falls from 4.3% to 3.0% (a = 1.30) and from 8.0% to 6.2% (a = 1.50), in this
    20-run subset.
- **No registered verdict changes.**
  - The own-seed, Block 3 and λ numbers in the paper are check-based at 50 steps, as registered.
  - For the rebuttal revision the check interval could be stated beside each of them, with the interpolated values
    as a labelled post hoc sensitivity. Doing that for the full run sets needs G at each 50-step check, which is
    not stored (the own-seed log keeps w₁ and w₂ but not b₁). That would be a rerun of all 720 Block 3 runs, 960
    own-seed runs and 80 base-window runs, logging G at checks only: minutes of compute.

## 5. Full 50-step rerun and sensitivity analysis (approved by the author, 2026-09-24; POST HOC)

- **Rerun.** Every crossing run of the 50-step experiments was replayed to its stored crossing, logging G and |w₂| at
  each 50-step check (`python -m src.crossing_audit full`; `crossing_audit_full_runs.csv`):
  - Block 3: 699 runs;
  - own-seed: 920 runs;
  - Block G base (λ, B1): 74 runs;
  - the other four Block G windows: 305 runs. These were added so that B2, pooled over all five windows, could be
    cadence-matched. That goes slightly beyond the 1,760 approved; the addition is disclosed here.
- **Not rerun**: 67 of the approved 1,760 runs never crossed, so they have no crossing to locate.
- **Reproduction.** Every one of the 1,998 replays reproduced its stored crossing step and |w₂| bit for bit.
  - Resources: 337 s plus about 1 min, at ≤ 0.35 GB.
  - The 305-run addition briefly ran as a fourth process beside three others, over the three-worker cap, for about
    one minute.
- **Validation of the sensitivity code** (`src/cadence_sensitivity.py`). The check-based path reproduces the committed
  numbers exactly:
  - λ(1.30) = 1.11487 and λ(1.50) = 1.16440;
  - B2 = 0.22803;
  - every Block 3 comparison (`prospective_comparisons.csv`) to 1e−12;
  - every own-seed criterion (`prospective_own_scores.csv`, primary) to 1e−12.
- **Results.** In `cadence_sensitivity.csv` (all quantities), `cadence_sensitivity_block3.csv` and
  `cadence_sensitivity_own_seed.csv`. They are written into the writer inputs as WP-6, with the check interval stated
  beside each number.
  - Own-seed, median R_cross/U_own − 1 (460 crossers per a): 4.1% → 2.9% (a = 1.30) and 8.8% → 6.2% (a = 1.50).
  - λ: 1.1149 → 1.0979 and 1.1644 → 1.1298.
  - Every registered own-seed criterion passes under both definitions. At a = 1.50 the reading "C better than U_own"
    (P2b, interval above 0) holds check-based but not interpolated.
  - Block 3 (author's decision 2026-09-24). The sensitivity analysis is the **fully interpolated** version:
    calibration (λ, so C; B1; B2) and observations both use interpolated crossings, one detection rule throughout.
    In it, every registered comparison keeps its sign and excludes 0. The observations-only version (50-step
    predictions against interpolated observations) is reported as **mixing detection rules**; in it, C − B2
    includes 0.
  - Own-seed at a = 1.50 (author's decision). P2b passes under both definitions; the stronger reading "C beats U_own"
    does not survive interpolation. The draft's "C is better, as registered" is flagged for softening in WP-6.
- **The residual statement** (the author's narrower wording, finalised on the full rerun, WP-6). Against the run's
  own threshold, the residual is about 3% at a = 1.30 and 6–6.5% at a = 1.50, both with every-step checks and with
  50-step checks when the crossing is interpolated.
