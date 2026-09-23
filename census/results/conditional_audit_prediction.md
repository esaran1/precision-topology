# Registration: audit of the conditional threshold (Block 1)

**Written before any audit computation.** Date: 2026-09-23, 11:45 EDT (15:45 UTC).
Frozen state being audited: `paper-submitted-v3` (62c7581). Everything below — objects,
procedures, tolerances, the decision rule and the stop conditions — is fixed here.

## What is being audited

The frozen Block B procedure (`src/blockB_landscape.py`; `R_glob`, `R_solve` in T60/T77)
and its scale-equivariant variant (`src/blockB_scaled.py`, T74). At fixed output scale
`s = |w₂|` it minimises the logistic loss over `(w₁, b₁, b₂)` with 50 (glob) or 24 (solve)
restarts, a 600-step screen, the best 8 carried to 3,000 steps, and a degeneracy filter
(`|loss − log 2| < 1e-4` or `|w₁| < 1e-3`). `R_glob` = first grid `s` whose retained minimiser
has class gap `G > 0`; `R_solve` = first grid `s` whose retained minimiser solves.
Grid: coarse 0.5 then fine 0.05 in `|w₂|`. **One grid step = 0.05 in `|w₂|`.**

## Procedures

- **1a (retain everything).** Rerun the frozen search at every `s` it evaluates when locating
  `R_glob` and `R_solve` at a ∈ {1.30, 1.35, 1.40, 1.45, 1.50, 1.60}, and record, for every
  restart: final parameters, loss, `G`, gradient norm, step count, and the discard reason.
  Add the exact constant predictor (`w₁ = 0`, `b₂ = 0`, balanced classes, loss `log 2`).
  Scale-equivariant variant: the same at a ∈ {1.01, 1.02, 1.03, 1.04} for family A
  (secondary).
- **1b (profiled bias).** At fixed `(w₁, b₁, s)`, `b₂` = the unique root of
  `mean(σ(z) − y) = 0` by safeguarded Newton (bisection fallback) to |residual| ≤ 1e-14.
  `L*(w₁, b₁; s) = min_{b₂} L`.
- **1c (certified, exhaustive).** Branch and bound on `L*` over `w₂ = s > 0` (the symmetry
  `(w₁, b₁, w₂) ↦ (−w₁, −b₁, −w₂)` covers the other sign), `b₁ ∈ [0, 2π)` (periodicity to be
  verified numerically), `w₁ ∈ [−W, W]` with `W(s, a)` from the linear-growth bound, derived
  in `conditional_audit.md` before use. Lipschitz bounds for `L*` and for `G` are derived and
  verified numerically. The procedure certifies `m₋(s) = min_{G ≤ 0} L*` and `m₊(s) = min_{G > 0} L*`
  to **absolute tolerance 1e-7**, with a float64 rounding margin of 1e-10 stated separately.
  `G` is the exact one-sided gap for `w₂ > 0`, `min_O φ − max_I φ`, with exact x-extrema.
  Certified `R_glob` = the `s` interval where `m₊ − m₋` changes sign, found by bisection to
  `|w₂|` width ≤ 0.005 (a tenth of the frozen fine step). The certified minimiser's solve
  status is reported at the grid points bracketing `R_solve`. Objective: the frozen 800-point
  quadrature population. Primary a: the six above. The small-ε a (1.01–1.04) run in
  rescaled coordinates if time allows (secondary).
- **1d (objective used).** The certified procedure applied to each of **50 seeds'** own
  400-point training sets (`fold1d.make_data(200, seed)`, seeds 0–49) at **a = 1.30 and
  a = 1.45**. Report the per-seed threshold spread and the population threshold's quantile in it.
- **1e (stricter Adam).** At every frozen grid `s` within ±0.1 of each reported threshold:
  the same search with 10× the steps (screen 6,000, full 30,000) and a stopping tolerance of
  gradient norm ≤ 1e-8. Report agreement with 1c.

## Registered decision rule

The placement transition is identifiable **without excluding any lower-loss candidate**, and
**1c, 1e and the frozen search agree within their reported numerical uncertainty** (1c: its
certified interval; frozen search and 1e: one grid step, 0.05 in `|w₂|`). If they disagree,
the certified procedure 1c defines the threshold from now on, the discrepancy is reported,
and every downstream number is recomputed.

## Stop and report if

- any discarded candidate (degenerate, or not carried past the screen) has a **lower loss
  than the retained branch** at an `s` within one coarse step (0.5) of a reported threshold;
- the certified threshold differs from the reported one by **more than one grid step** (0.05);
- the switch is a **basin exchange** (the `G ≤ 0` and `G > 0` minimisers are separated and
  `m₊ − m₋` changes sign), where the paper describes a continuous crossing of one branch.

## Terminology

From now on, "training-free" is replaced by "computed independently of unconstrained
training trajectories". The conditional search is itself an optimisation.

## Addendum — 2026-09-23 12:41 EDT, before any 1d computation

Scope restored and extended by decision, before any per-seed threshold was computed. 1d runs at
**50 seeds per a** (seeds 0–49) at **a = 1.30, 1.45 and 1.60**. The third value, a = 1.60, extends
the comparison between the quadrature and per-seed training objectives to the top of the range the
paper uses.

Procedure per seed: the full coarse scan (`conditional_certified.scan`, |w₂| from 1.5 to 11 in steps
of 0.5, so a second sign change would be seen), then bisection of each switch to width 0.005.
Reported: the per-seed threshold spread, and the quantile of the population threshold within it.
No prediction is attached; 1d is a check on the choice of objective.

Also added: a certified coarse scan across the full scale range at a = 1.30. At each grid scale it
records the certified global minimiser, with an enclosure of the surviving cells, and a certified
lower bound on the loss outside a ball around it. This certifies, at those scales, the single-branch
picture the candidate plots show.
