# Width-2 T2-3: is there an occupied branch whose fold or switch matches the crossings? (POST HOC)

**Label.** This is POST HOC, done at the author's request on 2026-09-28, on existing data only. **T2-3's registered
FAIL stands.**

**Producer.** `src/width2_fold.py`, with tests in `tests/test_width2_fold.py`. The tests cover a constructed
saddle-node: the fold is located to 1e−6, the λ_min slope is checked, and fold validation is exercised on pass and
fail cases, as are the switch and boundary events.

**Outputs.**
- `results/width2_fold/runs.csv`: one row per run, with each run's occupied basins at 11 pre-crossing times and the
  λ_min path near every event.
- `results/width2_fold/summary.json`.

**Code commit.** `d9c4d85`, made before the T2-3 runs were processed. Two runs (600,000 and 600,002) had been timed
first. The basin-placed addendum was added after that timing, and this is disclosed in the docstring.

**Status.** The computation finished for all 79 T2-3 crossing runs, and every replay reproduced its crossing. The
coordinator then asked me to stop, and nothing further was started. **Nothing is left unfinished.**

## Method (docstring)

- **The branch.** The occupied branch is the fixed-scale conditional loss's stationary point in the reduced
  coordinates, followed as a curve in (w, log s). The coordinates are (α₁, β₁, α₂, β₂, u, b), with the output
  direction u on the run's ℓ₁ orthant.
  - For duplicate units, u is dropped: it is an exact symmetry.
  - For a single unit, the idle unit is dropped.
- **Finding the basin at a given time.** Replay the run to step t, relax at its fixed scale s(t) for 8,000 Adam
  steps (the replay operations, with ℓ₁ projection), then polish with Levenberg–Marquardt. The reduced Hessian must
  be PD.
- **Times.** t = f·t_c for f = 0, 0.1, …, 0.9, and t_c − 1.
- **The primary branch.** The basin at the latest of these times that is unplaced.
- **Continuation.** Pseudo-arclength continuation upward in s, stopping at the first event:
  - a **fold** (the s-component of the tangent changes sign);
  - a **switch** (the branch's exact gap G₊ changes sign);
  - the orthant boundary;
  - or none by 10·s_cross.
- **Fold validation** (never needed; see below):
  - the step is halved;
  - λ_min/λ_max at the fold is ≤ 1e−3;
  - an independent search of 20 minimisations 1% beyond the fold finds no minimum there.
- **Switch validation.** Recompute the switch with the step halved; it must agree to 1e−4.
- **Addendum.** The basin just before the crossing, and the first step at which the run's fixed-scale basin is
  placed (bisection over steps).

## Results (`results/width2_fold/summary.json`)

**No folds.**
- No run's occupied branch reaches a fold before its first event:
  - 77 of 79 primary branches end at their **own switch**;
  - 2 have no event within 10·s_cross (seeds 600,052 and 600,068, which cross at steps 4 and 2).
- Along every continued branch, the reduced Hessian stays well conditioned: λ_min/λ_max ≥ 0.030 (median of the
  per-run minima 0.064).
- **So the fold-delay constant d(λ_min²)/ds (math note §14) is not applicable to T2-3.** There is no fold to read it
  from.
- All 77 switches passed step halving.

**Where the branches switch.**
- The primary occupied branch is mostly a two-unit configuration (66 of 79; 13 duplicates). It is found at a median
  of half the crossing time, at s ≈ 0.42.
- Its own switch is at a median s = **0.486**, range 0.24–6.03. That is close to the population threshold
  s_pop = 0.445.
- The run's fixed-scale basin becomes placed at a median s = **0.472**, which is 1.06 × s_pop (41% of runs within
  15% of s_pop).
- This is the same event as the branch switch: median s_exit / s_basin_placed = 0.99.
- For the 10 seeds with a width-2 own-sample threshold (exploratory, `asym_posthoc/own_thresholds.csv`):
  s_basin_placed / s_own has median 1.02, with 9 of 10 within 15%.

**Crossings do not match the switch.**
- Median s_cross / s_switch is **2.79**; only **3 of 77 (3.9%)** are within 15%.
- Those three are seeds 600,002, 600,058 and 600,073: the late, duplicate-branch runs that Track 2B found tracking
  (χ ≈ 0.004, obs/pred lag 1.04).
- Just before crossing, the run's fixed-scale basin is already placed in 75 of 79 runs.

**Reading (POST HOC).**
- At full speed the runs do occupy an unplaced branch, and it does switch. The switch is at about the population
  (or own-sample) width-2 threshold, and there is no fold.
- The run's state then lags far behind the scale: χ ≈ 1 (Track 2B). It crosses only at about 2.8–2.9 × the switch,
  while its own fixed-scale basin has been placed all along.
- **The 3.3× overshoot of T2-3 is a relaxation lag inside an already-placed basin, not a delayed exit from an
  occupied branch.**

## Say / Do not say

**Say:**
- (POST HOC) For the full-speed width-2 runs (T2-3), no occupied branch has a fold: 0 of 79.
- The occupied unplaced branch switches at s ≈ 0.49 (median), about the population threshold (0.445). Each run's
  fixed-scale basin is placed from there on.
- The crossings come a median 2.8× later (3 of 77 within 15%), consistent with the χ ≈ 1 relaxation lag.
- Only the 3 slowly moving, tracking runs cross at their branch's switch (within 4%).

**Do not say:**
- that T2-3's crossings are explained by a fold, or that "runs leave at the fold" at width 2;
- that T2-3's FAIL is revised;
- that the fold-delay constant was measured at width 2 (there is no fold);
- that the own-sample match (9 of 10 within 15%) is general: it covers only 10 exploratory seeds.
