# Track 2 writer inputs (width 2): 2A stuck states, 2B lag law, 2C validity condition

Written 2026-09-25 (evening). **All of 2A and 2B is POST HOC.** Both use existing data only. **2C was not
registered, and no fresh-seed run was made:** the binding calibration rule found no admissible rule scale (details
below).

**Verdicts that do not change: T2-3 FAIL; T2-3b UNRESOLVED; T2-3c UNRESOLVED; T2-3d FAIL.** Nothing here re-scores
them.

Producers:
- 2B: `src/width2_lag.py`, with tests in `tests/test_width2_lag.py`. Outputs are in `results/width2_lag/`: per-run
  `parts_<arm>.csv`, `summary.csv`, `summary.json` and `chi_bins.csv`.
- 2A: `src/width2_basins.py`, with tests in `tests/test_width2_basins.py`. Outputs are in `results/width2_basins/`:
  `endpoints.csv`, `summary.csv` and `by_scale.csv`.
- 2C: `src/t2c.py`, with tests in `tests/test_t2c.py`. Outputs are in `results/t2c/`: `availability.csv`,
  `availability_runs.csv`, `calibration_scales.csv` and `calibration.json` (plus `finality_runs.csv` if present).
- Replays were checked: every run and endpoint reproduced its recorded crossing ‖v‖₁ (to 1e−12) or its recorded
  endpoint (to 1e−9).

---------------------------------------------------------------------------------------------------------------------

## 2B. Does the lag law hold at width 2 where the account applies? (POST HOC)

### Method (fixed before any comparison was computed; in the module docstring)

- **Which runs.** Every crossing run of T2-3 (full speed, φ₂ = 1), T2-3b (φ₂ = 0.0093) and T2-3c / T2-3d
  (φ₂ = 0.01778). Asymmetric windows, a = 1.30, Δ = 0.4, each run on its own training set.
- **Replay.** Each run was replayed to its recorded crossing step.
- **Width-2 analogue of 1A's coordinates.**
  - Fast coordinates are z = (α₁, β₁, α₂, β₂, b), trained at η = 0.01.
  - The slow variables are the output weights v = (v₁, v₂). In the slowed arms both components move at φ₂η.
- **The tracked branch.** z*(v) is the stationary point in z at fixed v, found by damped Newton from the crossing
  state, with a positive-definite Hessian.
- **Following it.** Newton continuation (predictor–corrector, steps ≤ 2% of ‖v‖₁) along the run's own recorded
  v-path:
  - backward from the crossing if the branch is placed there;
  - forward along the crossing velocity if it is not.
- **The own-branch switch s\*.** The scale at which the branch's exact placement gap G₊ changes sign, bisected.
- **κ₂.** Computed as in 1A at the switch, with P = diag(1/(√v̂+ε)) from the run's own Adam state at the crossing.
  - Along the path, the gap's rate also includes the rotation of v's direction:
    ġ = ∇_zG·θ̇* + D_ṽG[ṽ̇].
  - With no rotation this is exactly 1A's formula.
  - The run's actual coordinates are used, so the 2π winding enters as κ_k does in 1A.
- **χ and the comparison.** χ = (ṡ/s\*)/(ηλ_min(P^{1/2}HP^{1/2})). The observed r_obs = s_c/s\* − 1 is compared
  with r_pred = κ₂χ, with no fit.
- **Secondary check.** The ray version (v = s·ṽ_c, 1A's formula literally) is kept in the parts files as
  `ray_*`.
- **Disclosed change.** The path version was made primary after the ray version was tried on three T2-3c runs. One
  of them had v's direction rotating at 0.88 of the scale's growth rate.

### Results (`results/width2_lag/summary.csv`, `summary.json`, `chi_bins.csv`)

**Branch defined.**
- **Slowed arms: 191 of 230 crossers.**
- The 39 without a branch are all but one early crossers:
  - s_cross ≤ 0.18, median step 48, about 0.5 relaxation times after initialisation;
  - the statuses are placed-back-to-init 12, no PD branch at the crossing 8, jump 8, branch lost 6, no switch in
    range 5;
  - these crossings happen during the initial relaxation and are not tracking any branch.
- **T2-3: 75 of 79.**

**Width-1 reference** (`lag_law/compare_arms.csv`, `predictions.csv`). The lag law held in all 36 of 36 arms,
with per-run χ up to **0.064** (arm medians up to 0.025).

**Slowed arms, branch-defined runs with χ within the width-1 range (χ ≤ 0.064; n = 53 / 54 / 58 for b / c / d):**

| arm | median r_obs | median r_pred | per-run within max(0.01, 0.25\|pred\|) | median \|log(s_c/s\*)\| |
|---|---|---|---|---|
| T2-3b | −0.0049 | +0.00003 | 66% | 0.0059 |
| T2-3c | +0.00005 | +0.00005 | 83% | 0.0015 |
| T2-3d | −0.0012 | +0.00005 | 81% | 0.0024 |

- Pooled over the slowed arms: 165 runs, 77% within, median |log(s_c/s\*)| = 0.0033.
- The per-arm medians are within tolerance in every arm, but only through the 0.01 floor.

**How weak this test is (say this).**
- In the slowed arms the predicted lag is tiny, |r_pred| ≤ 0.005. The late crossers (s ≈ 4.2–6.6, 133 of 191, the majority)
  have χ ≈ 1e−5.
- The mid crossers (s ≈ 0.05–0.49) have κ₂ ≈ 0.01–0.1: their branch's gap is driven mainly by the rotation of v, so a
  lag in time is a small lag in s.
- **So the slowed arms test the zero-lag limit:** the run crosses at its own tracked branch's switch to within about
  1%. They do not test κ quantitatively.
- The mid crossers on the branch (n = 9 / 9 / 11) have median obs 0.0011 / 0.0016 / 0.0022 against pred
  0.0002 / 0.0010 / 0.0009. That is within the floor, but obs/pred is 1.6–6.6, not about 1.

**A systematic departure not predicted by the lag law.**
- The late slowed crossers cross slightly **before** their branch's switch: median r_obs −0.0066 (b), −0.0005 (c),
  −0.0038 (d), against pred +0.00005.
- Their crossing states sit about 1e−3 from the branch (dist_c median 0.0013 / 0.0002 / 0.0008).
- A plausible reading, **not tested**: Adam's step-scale fluctuation about a nearly flat branch lets the gap first
  turn positive early.
- It is largest in T2-3b, the slowest arm, where the gap grows most slowly per step.
- This is what drives the within-tolerance fraction below 100% (66–83%).

**Full speed (T2-3).**
- **χ at the crossing:** χ_c = (ṡ/s_c)/(ηλ_min,c) ranges 0.0031–3.05, median 0.99 (77 runs with a PD branch at the
  crossing).
- With s\* in place of s_c: χ = 0.0032–1.0e5, median 1.73 (75 branch-defined runs).
- 72 of the 75 branch-defined runs have χ ≥ 0.41. They cross about 1 relaxation time after initialisation (median
  step·ηλ_min = 1.0), far from the branch they are assigned (median ‖z_c − z\*‖∞ = 1.75). They do not track it:
  median r_obs 1.22 against pred 0.27, and 4% within.
- **3 T2-3 runs do track** (seeds 600002, 600058, 600073):
  - late crossings (s_c = 4.8–6.2, 2,100–2,600 steps, about 250 relaxation times);
  - χ = 0.0032–0.0045;
  - two units of nearly equal weight share (0.47–0.49);
  - κ₂ = 7.47–8.25, close to the width-1 κ(1.30) = 7.61;
  - **r_obs / r_pred = 0.0359/0.0345, 0.0356/0.0342, 0.0245/0.0236, i.e. obs/pred = 1.04, 1.04, 1.04.**
  - This is the only width-2 case where the predicted lag exceeds the 0.01 floor. There the no-fit prediction agrees
    to 4%.

**χ bins, all arms, branch-defined runs** (`chi_bins.csv`; per-run fraction within tolerance):

| χ bin | n | fraction within |
|---|---|---|
| (0, 1e−4] | 133 | 77% |
| (1e−4, 0.01] | 5 | 100% |
| (0.01, 0.064] | 30 | 73% |
| (0.064, 0.3] | 19 | 32% |
| (0.3, 3] | 51 | 6% |
| > 3 | 28 | 0% |

**Verdict (POST HOC, descriptive).**
- Where the account applies (χ in width 1's range and the run relaxed onto a branch), width-2 crossings occur at the
  run's own tracked branch's switch, to about 1% (77% of runs within max(0.01, 0.25|pred|)).
- In the 3 runs where the lag is large enough to measure, the no-fit lag law holds (obs/pred = 1.04).
- The slowed arms confirm only the zero-lag limit.
- Above χ ≈ 0.06–0.3 the account fails, and T2-3 sits at χ ≈ 1.

---------------------------------------------------------------------------------------------------------------------

## 2A. Are the stuck states basins? (POST HOC)

### Method (fixed before any endpoint was evaluated; in the `src/width2_basins.py` docstring)

- **Endpoints.** Every Track 7 no-gating replay endpoint at the scored step H = 16,000 that is **unplaced** (exact
  G₊ ≤ 0) and **stationary** by Track 7's criterion (max|∇L|/‖v‖₁ ≤ 1e−6).
  - That is 281 endpoints: f1.30 125, f1.50 140, tanh 16.
  - They span all four R₂ levels and both variants.
  - Each was re-created by deterministic replay; all 281 reproduced to 1e−9.
- **Polish.** Newton on the fixed-scale conditional loss.
- **Reduced Hessian.** The Hessian in (θ, v, b), restricted to the 6 free directions of the held ℓ₁ sphere.
- **Classes:**
  - strict minimum (PD);
  - saddle;
  - duplicate-unit (Morse–Bott) minimum: the two units are copies up to orientation, the only zero mode is the weight
    transfer between them (an exact symmetry), and the rest is PD;
  - undetermined.
- **Loss gap.** The endpoint's loss minus that of the best **placed** conditional minimiser found at the same s on the
  run's own training set.
  - Starts: the population direct-check minimiser and its images, plus a 100-restart search with the 30 lowest
    candidates Newton-polished.
  - This is the best placed minimiser found, not a certified global one.
  - Convergence check, added after the first 136 endpoints and disclosed: all 50 (act, seed, R₂) groups with a placed
    minimiser were re-run and polished to 2,000 Newton iterations. Every best placed point had max|∇| ≤ 2e−9, stayed
    placed and did not move (`placed_check.csv`).

### Results (`results/width2_basins/summary.csv`, `by_scale.csv`, `endpoints.csv`, `placed_check.csv`)

**f_a (f1.30 and f1.50), 265 endpoints.**
- **All 265 are duplicate-unit configurations.** The two units are copies up to orientation, each with weight share
  0.5 and cancellation index 1. The function is a single width-1 unit with output weight s, embedded in width 2.
- **The reduced Hessian is PD in 0 of 265.** Each has exactly one zero eigenvalue (|λ|/λ_max ≤ 2e−12). Its
  eigenvector is the weight transfer between the copies, along which the function is exactly unchanged.
- **On the complement the Hessian is PD in 265 of 265** (λ_min/λ_max median 0.017). No endpoint is a saddle.
- **So 100% are conditional local minima of the fixed-scale loss, of Morse–Bott type:** a line of equal-loss minima,
  not isolated ones. They are basins. The weight-transfer direction is flat, but it leads only to other points of the
  same line.

**Loss gap to the best placed minimiser (same s, own training set).**
- A placed conditional minimiser was found for only **58 of 265** f_a endpoints:
  - R₂ = 0.003: 4 of 101;
  - R₂ = 0.01: 13 of 89;
  - R₂ = 0.03: 36 of 70;
  - R₂ = 0.1: 5 of 5.
- Where one was found, the stuck state is **lower** in 37 of 58 (median gap −0.0027; range −0.019 to +0.0018).
- By scale:
  - R₂ = 0.003: +1e−5 to +2e−5 (4 cases);
  - R₂ = 0.01: −0.0018 (f1.30) and −0.0030 (f1.50);
  - R₂ = 0.03: −0.0049 (f1.30) and +0.0006 (f1.50);
  - R₂ = 0.1: −0.015 and −0.016.
- The stuck state itself is the lowest loss found in **108 of 265**. A placed configuration is the lowest found in
  only **18 of 265**.
- **So on the runs' own training sets the stuck states are not high-loss traps.** They are usually as low as, or lower
  than, any placed configuration found at that scale.
- **Contrast with the population** (symmetric), where the validated direct check found the placed cancelling pair
  as the global minimiser at R₂ = 0.003 and 0.01.
- **Not tested:** that finite-sample asymmetry of the 400-point training sets is the cause.

**tanh, 16 endpoints (descriptive).**
- None is a duplicate.
- Polishing moved them far (median max-coordinate move 2.8), so they were not near stationary points.
- Afterwards: 9 saddles and 7 undetermined, **none a local minimum**.

### Writer section, 2A

**Say:**
- (POST HOC) Every stationary unplaced endpoint of the f_a no-gating test (265 of 265) is a conditional local minimum
  of the fixed-scale loss.
  - Each is a width-1 unit duplicated across the two hidden units.
  - Its reduced Hessian has one exact zero mode, the weight transfer between the copies, and is positive definite on
    the rest (a Morse–Bott line of minima).
  - So the stuck states are basins, which is why training at fixed scale does not leave them.
- (POST HOC) At the same scale on the run's own training set, a placed minimiser was found for only 58 of these
  endpoints. In 37 of those 58 the stuck state has the lower loss. The stuck state is the lowest found in 108 of 265.
- The reduced Hessian is never strictly positive definite: 0 of 265, because of the exact symmetry.

**Do not say:**
- "the stuck states are strict (isolated) local minima";
- "the stuck states are suboptimal / higher-loss traps" or "the placed minimiser is the global minimiser on the
  training set". On the own training sets the opposite is more common, and the placed search is not certified;
- anything that changes Track 7's registered outcome;
- anything about tanh beyond "no local minima among its 16 stationary-by-criterion endpoints".

---------------------------------------------------------------------------------------------------------------------

## 2C. The validity condition, and the registered test (NOT registered: the calibration rule stopped it)

### Validity condition, from the data (`width2_lag/summary.json`, `chi_bins.csv`; `lag_law/compare_arms.csv`)

- **Width 1:** tracking held in all 36 arms, at per-run χ up to 0.064 (arm medians ≤ 0.025).
- **Width 2, where it held:**
  - branch-defined runs with χ ≤ 0.064 are 73–100% within tolerance per bin (overall 77%);
  - with 0.064 < χ ≤ 0.3, 32%;
  - with 0.3 < χ ≤ 3, 6%;
  - above 3, 0%.
- **Width 2, where it failed:** at full speed (T2-3), χ at the crossing has median 0.99 (range 0.0031–3.05), and
  72 of 75 runs are at χ ≥ 0.41.
- **The condition as stated:** χ ≲ 0.06, and the crossing comes after the run has relaxed onto a branch.
  - Branch-defined tracking runs had ≥ 6.8 relaxation times before the crossing.
  - The runs with no branch crossed at a median 0.5 relaxation times.
  - The early slowed crossers fail this second part even when their χ is small. Their crossing is part of the initial
    relaxation into a basin that is already placed (12 are placed along their whole path back to initialisation).

### Calibration of the rule scale (author's binding amendment), BEFORE any registration

**Rule and definitions** (fixed in the `src/t2c.py` docstring and tests before computing):
- The candidate rule scale s_r lies on the grid 10^(−2.5 + k/8).
- A run is classified at s_r at the first passage of its ‖v‖₁ through s_r, and only if that passage is strictly before
  its crossing.
- **"Final":** the basin found there (the fixed-v Newton branch point), continued along the run's own path, ends at
  its crossing branch.
- **Choice:** the earliest s_r with the final fraction ≥ 0.95 **and** s_r < the earliest crossing observed.

**Result** (`results/t2c/calibration.json`): 239 runs (T2-3b, T2-3c, T2-3d, not placed at step 0).
- The earliest crossing is **s = 0.0164**.
- Under matched initialisation the runs start at ‖v‖₁ ≈ 0.001–0.081 (median 0.041).
- So below 0.0164, at most **7.5%** of runs ever pass through s_r before crossing.
- At larger s_r, runs that cross first have no classification. The fraction with any classification peaks at **0.778
  (s_r = 0.075)** and falls to 0.60 by s_r = 0.56, as the early crossers (40%) drop out.
- The available fraction is an upper bound on the final fraction. **No scale can be final in ≥ 95% of runs, and
  none below the earliest crossing comes close.**
- **Finality where it was computed** (`results/t2c/finality_runs.csv`):
  - at s_r = 0.075 (the peak of availability), 131 of 239 runs are final (0.548). Of the 177 classifiable crossers,
    28 continued to their crossing on a different branch and 18 hit a jump or had no basin at the rule scale;
  - at s_r = 0.0133 (below the earliest crossing), 4 of 239 are final (0.017).
- **Decision: STOP.** Nothing was registered, no fresh seed was trained, and no seed range was consumed.

**Possible redesign, for the author only** (not done, not registered):
- Classify each run's basin at step 0 (the fixed-v Newton branch point from the initial state), which is before any
  crossing for every run not placed at step 0.
- Predict the crossing from that basin's path-continued switch.
- The 2B data suggest this would separate the relaxation crossers (placed basins) from the trackers. It is a different
  rule from the one the author fixed.

---------------------------------------------------------------------------------------------------------------------

## Writer section

### 2B

**Say:**
- (POST HOC) At width 2, in the regime where the account applies, a run crosses at the switch of the branch it
  tracks:
  - the regime is χ within width 1's range (≤ 0.06) and the run relaxed onto a branch;
  - the branch is followed by Newton continuation along the run's own output-weight path;
  - 77% of 165 slowed runs are within max(0.01, 0.25|pred|), with median |log(s_c/s\*)| = 0.003.
- (POST HOC) The three full-speed runs that track a branch have a measurable lag. There the no-fit width-1 formula
  (κ₂ ≈ 7.5–8.2, computed on the run's own branch) predicts it to 4% (obs/pred 1.04).
- The slowed arms test only the zero-lag limit: the predicted lags there are ≤ 0.005, below the 0.01 floor.
- The full-speed arm T2-3 sits at χ ≈ 1 (0.003–3), about 15–30× beyond width 1's range. There runs cross within about one
  relaxation time, far from any branch. That is why T2-3 FAILED; the verdict stands.

**Do not say:**
- that the lag law is "confirmed at width 2" in general, or that κ is validated by the slowed arms (their predicted
  lags are below resolution);
- that T2-3, T2-3b, T2-3c or T2-3d change verdict;
- that the late slowed crossers' early crossing (median r_obs −0.0005 to −0.0066) is explained; it is not tested;
- "own-branch switch" as if it were the population threshold, or the width-1 own-sample threshold. It is a per-run,
  per-path quantity computed after the fact.

### 2A (see the 2A section above)

### 2C

**Say:**
- The validity condition, from the data: tracking holds up to χ ≈ 0.06 at both widths. At width 2 it also needs the
  run to have relaxed onto a branch before crossing.
- About 17% of slowed runs (39/230) cross during the initial relaxation (≈ 0.5 relaxation times) into a basin that
  is already placed. The account does not apply to them.
- The registered prospective test was not run: the pre-set calibration rule found no admissible rule scale. At most
  78% of runs are ever classifiable before crossing at any scale, and at most 7.5% below the earliest crossing
  (0.016).

**Do not say:**
- that a prospective width-2 basin test was registered or passed;
- that the early-scale rule "fails". It was never applied.

### If asked about the other possible outcomes (so wording is ready)
- Had 2C been registered and PASSED: "a fresh-seed test predicted each run's crossing from its early basin". This
  **did not happen**, so do not use it.
- Had it FAILED: the FAIL would have been reported beside T2-3's FAIL. **Not applicable.**
