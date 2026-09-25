# Asymmetric windows, Track 2 Step 1: exploratory pilot (design, written before any run)

Status: EXPLORATORY. This is not a registered test and nothing from it enters the paper. Its only purpose is a
go/no-go decision on whether to register Step 2 (author's fast-track plan, 2026-09-25). Written and committed
before the pilot's first evaluation.

## Geometry

- Inner window I = [−0.8, 0.8] (class 0). Outer window O = [−2.0, −1.2] ∪ [1.2, 2.0 + Δ] (class 1).
- The pilot uses Δ = 0.8. The later test uses Δ = 0.4 and is not run in the pilot.
- Population: 400 inner points, `linspace(−0.8, 0.8, 400)` as the base population. There are 400 outer points,
  split between the two sides in proportion to length: n_L = round(400·0.8/(1.6 + Δ)) on the left,
  `−linspace(1.2, 2.0, n_L)`, and n_R = 400 − n_L on the right, `linspace(1.2, 2.0 + Δ, n_R)`. At Δ = 0 this is the
  base population exactly (200 + 200), which is tested. At Δ = 0.8, n_L = 133 and n_R = 267.
- Placement is the continuous-window G₊ = min_O φ − max_I φ on the asymmetric windows, from the exact extrema
  (`width2_geometry.gaps` with the windows reset). A candidate is placed iff the enclosure's lower end is > 0.

## Why a threshold is expected (the criterion's prediction, stated before the run)

On symmetric windows the ramp contributes nothing to the class-mean gap. On these windows it contributes
m·(ṽ₁α₁ + ṽ₂α₂), with m = E_O x − E_I x ≠ 0 (m = 0.803 at Δ = 0.8 in the population; see
`asym_pilot.population`). The small-scale class-mean maximiser therefore uses the ramp, which is unplaced. At
large scale the minimiser maximises the worst-case gap, which is placed. So the criterion predicts a width-2
placement switch.

## Pilot

- a = 1.30, Δ = 0.8. Width-2 conditional minimisation at scale s, with ṽ on the ℓ₁ circle and b profiled
  (`width2_conditional.search_batch`, the validated search's optimiser and starts). 13 scales,
  s = 10^(−1 + k/4) for k = 0..12 (0.1 to 100). 400 restarts per scale, BFGS cap 3,000.
- The five lowest distinct candidates at each scale are polished by damped Newton (`width2_finish.newton`) and the
  profiled loss is recomputed. The retained candidate is the lowest polished loss, with the constant predictor
  included as a candidate.
- A scale is **undecided** if the best placed and the best unplaced candidate differ in loss by less than 1e-9.
- Width-1 reference: the same search with ṽ fixed at (±1, 0), same scales, 200 restarts.

## Go / no-go (fixed now)

**Go** iff every condition below holds for the width-2 retained minimiser:
1. No scale is undecided.
2. There is a switch scale s_c such that the minimiser is unplaced at every scanned s < s_c and placed at every
   scanned s > s_c. That is exactly one change of placement status, from unplaced to placed, and no alternation.
3. The unplaced scales span at least a factor of 3 (max/min ≥ 3), and so do the placed scales.

Anything else is a **no-go**: stop Track 2, and nothing enters the paper. The width-1 reference does not affect
the decision.

Budget: at most 1.5 h of wall time including code; three workers at nice 15.
